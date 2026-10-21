// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Find each tile's [first, last) slice of the sorted instance list.
//
// After the sort, instances are grouped by tile id. A boundary exists
// wherever the tile id differs from the previous element, so one thread per
// instance is enough: write the start of the run to the new tile and the
// end of the run to the previous one.

#include "forward_common.slangh"

StructuredBuffer<uint> sortedKeysTile;
RWStructuredBuffer<uint2> tileRanges;
// Instance count (instance_dispatch.slang) -- also what sized this
// indirect dispatch.
StructuredBuffer<InstanceDispatch> dispatch;

[shader("compute")]
[numthreads(SABERCAT_WORKGROUP_SIZE, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    uint index = threadId.x;
    uint instanceCount = dispatch[0].instanceCount;
    if (index >= instanceCount)
    {
        return;
    }

    uint tile = sortedKeysTile[index];

    if (index == 0)
    {
        tileRanges[tile].x = 0;
    }
    else
    {
        uint previousTile = sortedKeysTile[index - 1];
        if (previousTile != tile)
        {
            tileRanges[tile].x = index;         // this tile's run starts here
            tileRanges[previousTile].y = index; // previous tile's run ends here
        }
    }

    if (index == instanceCount - 1)
    {
        tileRanges[tile].y = instanceCount;
    }
}
