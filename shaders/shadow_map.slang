// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// A light-view shadow map of splats, for PCSS.
//
// Walks a light-camera ForwardPass's tile-sorted lists exactly like
// tile_rasterize (same evaluateSplat, order, 1e-4 early-out) but stores,
// per texel, the depth at which transmittance first falls to 0.5 -- the
// MEDIAN-opacity depth. The front-most contributing depth (what
// tile_rasterize's outputDepth keeps) is set by any splat with alpha >= 1/255
// and would let faint haze cast hard shadows; the median depth is where the
// splats become more opaque than not, the usual choice for transmittance
// shadow maps. Texels whose transmittance never reaches 0.5 hold a huge depth (no
// blocker; stored as 1e30). y = final transmittance, kept for diagnostics.

#include "forward_common.slangh"

StructuredBuffer<ProjectedGaussian> projected;
StructuredBuffer<uint> sortedValues;
StructuredBuffer<uint2> tileRanges;
RWStructuredBuffer<float2> shadowMap; // (median depth, final T), row-major

[vk::push_constant]
ConstantBuffer<CameraUniforms> camera;

groupshared float2 batchCenter[SABERCAT_TILE_THREADS];
groupshared float3 batchConic[SABERCAT_TILE_THREADS];
groupshared float batchAlpha[SABERCAT_TILE_THREADS];
groupshared float batchDepth[SABERCAT_TILE_THREADS];
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

    float transmittance = 1.0f;
    float medianDepth = 1e30f; // "no blocker"
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
            ProjectedGaussian g = projected[sortedValues[batchStart + linearThread]];
            batchCenter[linearThread] = g.center;
            batchConic[linearThread] = g.conic;
            batchAlpha[linearThread] = g.alpha;
            batchDepth[linearThread] = g.depth;
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
                float before = transmittance;
                transmittance *= (1.0f - alpha);
                if (before > 0.5f && transmittance <= 0.5f)
                {
                    medianDepth = batchDepth[i];
                }
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
        shadowMap[pixel.y * camera.imageWidth + pixel.x] = float2(medianDepth, transmittance);
    }
}
