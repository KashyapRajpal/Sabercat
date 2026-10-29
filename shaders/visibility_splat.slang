// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Baked visibility as a shadow source (ShadowMode::BakedSH).
// Each Gaussian's baked visibility SH (the `visibility` relight stream,
// bake_visibility.slang) is evaluated toward each shadow-casting light --
//     V_i(l) = saturate(sum_k c_ik Y_k(w_il)),
// w_il = -direction for a directional light, the direction from the
// Gaussian's mean to the light's center otherwise -- and splatted exactly
// like gbuffer_rasterize splats Surfels (same sorted lists, same
// evaluateSplat, same early-outs), normalized by coverage:
//     visibility[l][p] = sum_i w_i V_i(l) / coverage   (1 where uncovered)
// Real time: no rays. A degree-3 SH is a smooth, band-limited stand-in for
// the true visibility, so shadows are soft and contact detail is lost --
// the |error| debug view grades it against the ray-traced reference.
// Layers of lights outside `lightMask` are left untouched (cleared to 1).

#include "forward_common.slangh"
#include "relight_common.slangh"

// One light, as light_visibility.slangh's TraceLight (only positionType and
// direction are read here).
struct SplatLight
{
    float4 positionType;
    float4 direction;
    float4 axisX;
    float4 axisY;
    float4 sizeLength;
};

struct SplatParams
{
    uint lightMask; // bit l: light l casts shadows (l < SABERCAT_MAX_LIGHTS)
    uint lightCount;
    uint pad0;
    uint pad1;
};

StructuredBuffer<ProjectedGaussian> projected;
StructuredBuffer<uint> sortedValues;
StructuredBuffer<uint2> tileRanges;
StructuredBuffer<float4> means;
StructuredBuffer<float> visibilitySH; // SABERCAT_VISIBILITY_COEFFS per Gaussian
StructuredBuffer<SplatLight> lights;
StructuredBuffer<SplatParams> paramsBuffer;
RWStructuredBuffer<float> visibility; // SABERCAT_MAX_LIGHTS layers of width*height

// Hanning sigma-factors w_l = (1 + cos(pi l / 4)) / 2 per band: suppress the
// ringing a sharp feature (e.g. a thin horizon notch from coplanar neighbors)
// leaks across the sphere when a degree-3 SH is evaluated directly.
static const float kVisibilityWindow[4] = { 1.0f, 0.853553391f, 0.5f, 0.146446609f };

[vk::push_constant]
ConstantBuffer<CameraUniforms> camera;

groupshared float2 batchCenter[SABERCAT_TILE_THREADS];
groupshared float3 batchConic[SABERCAT_TILE_THREADS];
groupshared float batchAlpha[SABERCAT_TILE_THREADS];
groupshared float batchVisibility[SABERCAT_MAX_LIGHTS][SABERCAT_TILE_THREADS];
groupshared uint doneCount;

[shader("compute")]
[numthreads(SABERCAT_TILE_SIZE, SABERCAT_TILE_SIZE, 1)]
void main(uint3 groupId : SV_GroupID, uint3 localId : SV_GroupThreadID)
{
    SplatParams params = paramsBuffer[0];
    uint tileId = groupId.y * camera.tilesX + groupId.x;
    uint2 pixel = uint2(groupId.x * SABERCAT_TILE_SIZE + localId.x,
                        groupId.y * SABERCAT_TILE_SIZE + localId.y);
    float2 pixelPosition = float2(pixel);

    uint2 range = tileRanges[tileId];
    uint linearThread = localId.y * SABERCAT_TILE_SIZE + localId.x;

    float sums[SABERCAT_MAX_LIGHTS];
    for (uint l = 0; l < SABERCAT_MAX_LIGHTS; ++l)
    {
        sums[l] = 0.0f;
    }
    float transmittance = 1.0f;
    bool done = pixel.x >= camera.imageWidth || pixel.y >= camera.imageHeight;
    bool reportedDone = false;

    if (linearThread == 0)
    {
        doneCount = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    if (done)
    {
        InterlockedAdd(doneCount, 1);
        reportedDone = true;
    }

    for (uint batchStart = range.x; batchStart < range.y; batchStart += SABERCAT_TILE_THREADS)
    {
        uint batchSize = min(SABERCAT_TILE_THREADS, range.y - batchStart);

        if (linearThread < batchSize)
        {
            uint index = sortedValues[batchStart + linearThread];
            ProjectedGaussian g = projected[index];
            batchCenter[linearThread] = g.center;
            batchConic[linearThread] = g.conic;
            batchAlpha[linearThread] = g.alpha;

            float3 mean = means[index].xyz;
            uint base = index * SABERCAT_VISIBILITY_COEFFS;
            for (uint l = 0; l < params.lightCount; ++l)
            {
                float v = 1.0f;
                if (((params.lightMask >> l) & 1u) != 0u)
                {
                    SplatLight light = lights[l];
                    float3 w = uint(light.positionType.w + 0.5f) == 0u
                                   ? -light.direction.xyz
                                   : light.positionType.xyz - mean;
                    float wLen = length(w);
                    w = wLen > 1e-12f ? w / wLen : float3(0.0f, 1.0f, 0.0f);
                    float b[16];
                    shBasis16(w, b);
                    float sum = 0.0f;
                    for (uint k = 0; k < SABERCAT_VISIBILITY_COEFFS; ++k)
                    {
                        uint band = k == 0u ? 0u : (k < 4u ? 1u : (k < 9u ? 2u : 3u));
                        sum += kVisibilityWindow[band] * visibilitySH[base + k] * b[k];
                    }
                    v = saturate(sum);
                }
                batchVisibility[l][linearThread] = v;
            }
        }
        GroupMemoryBarrierWithGroupSync();

        if (!done)
        {
            for (uint i = 0; i < batchSize; ++i)
            {
                float alpha = 0.0f;
                float gaussian = 0.0f;
                bool clamped = false;
                if (!evaluateSplat(pixelPosition, batchCenter[i], batchConic[i], batchAlpha[i],
                                   alpha, gaussian, clamped))
                {
                    continue;
                }

                float w = transmittance * alpha;
                for (uint l = 0; l < params.lightCount; ++l)
                {
                    sums[l] += w * batchVisibility[l][i];
                }
                transmittance *= (1.0f - alpha);

                if (transmittance < 1e-4f)
                {
                    done = true;
                    break;
                }
            }

            if (done && !reportedDone)
            {
                InterlockedAdd(doneCount, 1);
                reportedDone = true;
            }
        }

        GroupMemoryBarrierWithGroupSync();
        if (doneCount == SABERCAT_TILE_THREADS)
        {
            break;
        }
    }

    if (pixel.x < camera.imageWidth && pixel.y < camera.imageHeight)
    {
        uint p = pixel.y * camera.imageWidth + pixel.x;
        uint layerSize = camera.imageWidth * camera.imageHeight;
        float coverage = 1.0f - transmittance;
        for (uint l = 0; l < params.lightCount; ++l)
        {
            if (((params.lightMask >> l) & 1u) != 0u)
            {
                visibility[l * layerSize + p] = coverage > 1e-6f ? sums[l] / coverage : 1.0f;
            }
        }
    }
}
