// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Forward pass, stage 2: expand each visible Gaussian into one
// (tileID, depth) key per tile it overlaps.
//
// The key is packed so a single unsigned sort orders by tile first and by
// depth within a tile -- exactly the order the rasterizer wants:
//
//   key   = (tileID << 32) | depthBits      (64-bit, split across two
//   value = Gaussian index                   uint32 buffers)
//
// Depth is a positive float, and for positive floats the IEEE-754 bit
// pattern compares identically to the value itself when read as an
// unsigned int -- so no extra key transform is needed.

#include "forward_common.slangh"

StructuredBuffer<ProjectedGaussian> projected;

[vk::push_constant]
ConstantBuffer<CameraUniforms> camera;

// Exclusive prefix sum of tileCount, giving each Gaussian its base slot in
// the instance list. Computed on the host side of the dispatch chain.
StructuredBuffer<uint> instanceOffsets;

RWStructuredBuffer<uint> keysTile;   // high word: tile id
RWStructuredBuffer<uint> keysDepth;  // low word: depth bits
RWStructuredBuffer<uint> values;     // Gaussian index

// GPU-side instance sizing (instance_dispatch.slang). Its instanceCount is
// the number of slots the sort buffers can take this frame -- 0 when the
// frame overflowed their capacity -- so no write below can land past the
// end of a buffer the host sized from an earlier frame.
StructuredBuffer<InstanceDispatch> dispatch;

[shader("compute")]
[numthreads(SABERCAT_WORKGROUP_SIZE, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    uint index = threadId.x;
    if (index >= camera.gaussianCount)
    {
        return;
    }

    ProjectedGaussian g = projected[index];
    if (g.tileCount == 0)
    {
        return; // culled in the projection pass
    }

    int minTileX = max((int)floor((g.center.x - g.radius) / SABERCAT_TILE_SIZE), 0);
    int minTileY = max((int)floor((g.center.y - g.radius) / SABERCAT_TILE_SIZE), 0);
    int maxTileX = min((int)floor((g.center.x + g.radius) / SABERCAT_TILE_SIZE), (int)camera.tilesX - 1);
    int maxTileY = min((int)floor((g.center.y + g.radius) / SABERCAT_TILE_SIZE), (int)camera.tilesY - 1);

    uint slot = instanceOffsets[index];
    uint slotLimit = dispatch[0].instanceCount;
    uint depthBits = asuint(g.depth);

    for (int ty = minTileY; ty <= maxTileY; ++ty)
    {
        for (int tx = minTileX; tx <= maxTileX; ++tx)
        {
            if (slot >= slotLimit)
            {
                return;
            }
            uint tileId = (uint)(ty * (int)camera.tilesX + tx);
            keysTile[slot] = tileId;
            keysDepth[slot] = depthBits;
            values[slot] = index;
            ++slot;
        }
    }
}
