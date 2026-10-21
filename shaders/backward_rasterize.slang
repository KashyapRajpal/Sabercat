// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Backward pass, stage 1: differentiate the tile blend.
//
// Mirror image of tile_rasterize: one workgroup per 16x16 tile, one thread
// per pixel, the tile's sorted Gaussians staged cooperatively through shared
// memory -- but walked BACK to front. Each thread starts from the final
// transmittance tile_rasterize saved for its pixel and divides each
// Gaussian's (1 - alpha) back out, recovering the transmittance in front of
// it without re-running the forward blend.
//
// Per (pixel, Gaussian j) contribution, with T_j the transmittance in front
// of j and C_behind the color composited behind j:
//
//     dRGB/dcolor_j  = alpha_j T_j
//     dRGB/dalpha_j  = T_j (color_j - C_behind)
//     dA/dalpha_j    = T_final / (1 - alpha_j)        (A = 1 - T_final)
//
// and through alpha_j = min(opacity_j * G, MAX_ALPHA), G = exp(-q/2),
// q = delta^T conic delta, delta = pixel - center, into dL/d(center, conic,
// projected opacity). Those 9 floats per Gaussian are summed over every pixel
// with fp32 atomics (PERFORMANCE.md invariant #7; no subgroup pre-reduction
// yet -- profile first) and chained to the parameters by backward_project.
//
// The image gradient is RGBA: the alpha channel is the coverage the forward
// writes, so a trainer can composite its own background outside the renderer
// (the reference trainer adds one) and still get correct gradients.

#include "forward_common.slangh"

StructuredBuffer<ProjectedGaussian> projected;
StructuredBuffer<uint> sortedValues;   // Gaussian index, tile-major/depth-sorted
StructuredBuffer<uint2> tileRanges;    // per tile: (firstInstance, lastInstance)
StructuredBuffer<uint2> pixelState;    // per pixel: (asuint(final T), nContrib)
StructuredBuffer<float4> dLossDImage;  // per pixel, row-major, RGBA

// Per Gaussian, SABERCAT_PROJECTED_GRADIENT_FLOATS floats:
// [0,1] dCenter  [2,3,4] dConic (a, b, c)  [5] dAlpha (projected opacity)
// [6,7,8] dColor  [9..11] pad. Zeroed by the host before this dispatch.
RWStructuredBuffer<float> projectedGradients;

[vk::push_constant]
ConstantBuffer<CameraUniforms> camera;

groupshared float2 batchCenter[SABERCAT_TILE_THREADS];
groupshared float3 batchConic[SABERCAT_TILE_THREADS];
groupshared float3 batchColor[SABERCAT_TILE_THREADS];
groupshared float batchAlpha[SABERCAT_TILE_THREADS];
groupshared uint batchIndex[SABERCAT_TILE_THREADS];
// Largest nContrib in the tile: entries at or past it blended into no pixel
// here, so the walk starts from it instead of the end of the tile's range.
groupshared uint tileContributors;

void accumulate(uint base, uint slot, float value)
{
    float unused;
    InterlockedAdd(projectedGradients[base + slot], value, unused);
}

[shader("compute")]
[numthreads(SABERCAT_TILE_SIZE, SABERCAT_TILE_SIZE, 1)]
void main(uint3 groupId : SV_GroupID, uint3 localId : SV_GroupThreadID)
{
    uint tileId = groupId.y * camera.tilesX + groupId.x;
    uint2 pixel = uint2(groupId.x * SABERCAT_TILE_SIZE + localId.x,
                        groupId.y * SABERCAT_TILE_SIZE + localId.y);
    float2 pixelPosition = float2(pixel); // integer pixel centers, as forward
    uint linearThread = localId.y * SABERCAT_TILE_SIZE + localId.x;
    bool inside = pixel.x < camera.imageWidth && pixel.y < camera.imageHeight;

    uint2 range = tileRanges[tileId];

    if (linearThread == 0)
    {
        tileContributors = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    float finalTransmittance = 1.0f;
    uint contributorCount = 0;
    float4 dPixel = float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (inside)
    {
        uint pixelIndex = pixel.y * camera.imageWidth + pixel.x;
        uint2 state = pixelState[pixelIndex];
        finalTransmittance = asfloat(state.x);
        contributorCount = state.y;
        dPixel = dLossDImage[pixelIndex];
        uint unused;
        InterlockedMax(tileContributors, contributorCount, unused);
    }
    GroupMemoryBarrierWithGroupSync();
    uint walkCount = tileContributors; // uniform across the workgroup

    float transmittance = finalTransmittance;
    float3 colorBehind = float3(0.0f, 0.0f, 0.0f);
    float lastAlpha = 0.0f;
    float3 lastColor = float3(0.0f, 0.0f, 0.0f);

    // Batches of the tile's relative range [0, walkCount), last batch first.
    for (uint batchEnd = walkCount; batchEnd > 0;)
    {
        uint batchStart = batchEnd > SABERCAT_TILE_THREADS ? batchEnd - SABERCAT_TILE_THREADS : 0;
        uint batchSize = batchEnd - batchStart;

        if (linearThread < batchSize)
        {
            uint gaussianIndex = sortedValues[range.x + batchStart + linearThread];
            ProjectedGaussian g = projected[gaussianIndex];
            batchCenter[linearThread] = g.center;
            batchConic[linearThread] = g.conic;
            batchColor[linearThread] = g.color;
            batchAlpha[linearThread] = g.alpha;
            batchIndex[linearThread] = gaussianIndex;
        }
        GroupMemoryBarrierWithGroupSync();

        if (inside)
        {
            for (uint k = batchSize; k > 0; --k)
            {
                uint i = k - 1;
                if (batchStart + i >= contributorCount)
                {
                    continue; // past this pixel's last blend (early-out)
                }

                // Same function and inputs as tile_rasterize, so exactly the
                // pairs that blended forward are revisited here.
                float alpha = 0.0f;
                float gaussian = 0.0f;
                bool clamped = false;
                if (!evaluateSplat(pixelPosition, batchCenter[i], batchConic[i], batchAlpha[i],
                                   alpha, gaussian, clamped))
                {
                    continue;
                }

                float oneMinusAlpha = 1.0f - alpha;
                transmittance /= oneMinusAlpha; // now T_j, in front of j
                float3 color = batchColor[i];
                uint base = batchIndex[i] * SABERCAT_PROJECTED_GRADIENT_FLOATS;

                // Color.
                float weight = alpha * transmittance;
                accumulate(base, 6, weight * dPixel.x);
                accumulate(base, 7, weight * dPixel.y);
                accumulate(base, 8, weight * dPixel.z);

                // Alpha: RGB through the "behind" color, plus coverage.
                colorBehind = lastAlpha * lastColor + (1.0f - lastAlpha) * colorBehind;
                lastAlpha = alpha;
                lastColor = color;
                float dAlpha = transmittance * dot(color - colorBehind, dPixel.xyz)
                               + dPixel.w * finalTransmittance / oneMinusAlpha;

                // Clamped at MAX_ALPHA: alpha is constant there, so neither
                // G nor the opacity receives gradient (gsplat's choice; the
                // INRIA reference passes it straight through instead).
                if (clamped)
                {
                    continue;
                }

                float opacity = batchAlpha[i];
                accumulate(base, 5, gaussian * dAlpha);

                // dL/dq = dL/dG * dG/dq = (opacity dAlpha) * (-G/2).
                float dExponent = -0.5f * gaussian * opacity * dAlpha;
                float2 delta = pixelPosition - batchCenter[i];
                float3 conic = batchConic[i];
                accumulate(base, 2, dExponent * delta.x * delta.x);
                accumulate(base, 3, dExponent * 2.0f * delta.x * delta.y);
                accumulate(base, 4, dExponent * delta.y * delta.y);
                // dq/dcenter = -dq/ddelta = -2 (conic . delta).
                accumulate(base, 0, -2.0f * dExponent * (conic.x * delta.x + conic.y * delta.y));
                accumulate(base, 1, -2.0f * dExponent * (conic.y * delta.x + conic.z * delta.y));
            }
        }

        // Every thread must reach this before the next batch overwrites
        // shared memory, including out-of-image threads.
        GroupMemoryBarrierWithGroupSync();
        batchEnd = batchStart;
    }
}
