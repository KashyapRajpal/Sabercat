// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The splatted G-buffer. Walks ForwardPass's tile-sorted
// instance lists exactly like tile_rasterize -- same batches, same
// evaluateSplat, same 1e-4 transmittance and group-wide early-outs -- but
// blends per-Gaussian surface data (relight_prepare's Surfels) instead of
// color. Every pixel's coverage (1 - T) therefore equals the color
// image's alpha bit for bit, which relight_test checks.
//
// Outputs are premultiplied sums (sum of w_i * x_i, w_i = T_i * alpha_i);
// deferred_lighting divides by coverage. Row-major width*height float4.

#include "forward_common.slangh"
#include "relight_common.slangh"

StructuredBuffer<ProjectedGaussian> projected;
StructuredBuffer<uint> sortedValues;
StructuredBuffer<uint2> tileRanges;
StructuredBuffer<Surfel> surfels;
RWStructuredBuffer<float4> gNormalDepth;   // sum w*n (xyz), sum w*depth (w)
RWStructuredBuffer<float4> gBaseRoughness; // sum w*base (rgb), sum w*roughness
RWStructuredBuffer<float4> gMetalAmbientCoverage; // sum w*metallic, sum w*ambient, coverage, 0

[vk::push_constant]
ConstantBuffer<CameraUniforms> camera;

groupshared float2 batchCenter[SABERCAT_TILE_THREADS];
groupshared float3 batchConic[SABERCAT_TILE_THREADS];
groupshared float batchAlpha[SABERCAT_TILE_THREADS];
groupshared float batchDepth[SABERCAT_TILE_THREADS];
groupshared float4 batchNormalMetal[SABERCAT_TILE_THREADS];
groupshared float4 batchBaseRough[SABERCAT_TILE_THREADS];
groupshared float batchAmbient[SABERCAT_TILE_THREADS];
groupshared uint doneCount;

[shader("compute")]
[numthreads(SABERCAT_TILE_SIZE, SABERCAT_TILE_SIZE, 1)]
void main(uint3 groupId : SV_GroupID, uint3 localId : SV_GroupThreadID)
{
    uint tileId = groupId.y * camera.tilesX + groupId.x;
    uint2 pixel = uint2(groupId.x * SABERCAT_TILE_SIZE + localId.x,
                        groupId.y * SABERCAT_TILE_SIZE + localId.y);
    float2 pixelPosition = float2(pixel);

    uint2 range = tileRanges[tileId];
    uint linearThread = localId.y * SABERCAT_TILE_SIZE + localId.x;

    float3 normalSum = float3(0.0f, 0.0f, 0.0f);
    float depthSum = 0.0f;
    float3 baseSum = float3(0.0f, 0.0f, 0.0f);
    float roughSum = 0.0f;
    float metalSum = 0.0f;
    float ambientSum = 0.0f;
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
            Surfel s = surfels[index];
            batchCenter[linearThread] = g.center;
            batchConic[linearThread] = g.conic;
            batchAlpha[linearThread] = g.alpha;
            batchDepth[linearThread] = g.depth;
            batchNormalMetal[linearThread] = s.normalMetallic;
            batchBaseRough[linearThread] = s.baseColorRoughness;
            batchAmbient[linearThread] = s.ambient.x;
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
                normalSum += w * batchNormalMetal[i].xyz;
                metalSum += w * batchNormalMetal[i].w;
                baseSum += w * batchBaseRough[i].xyz;
                roughSum += w * batchBaseRough[i].w;
                ambientSum += w * batchAmbient[i];
                depthSum += w * batchDepth[i];
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
        gNormalDepth[p] = float4(normalSum, depthSum);
        gBaseRoughness[p] = float4(baseSum, roughSum);
        gMetalAmbientCoverage[p] = float4(metalSum, ambientSum, 1.0f - transmittance, 0.0f);
    }
}
