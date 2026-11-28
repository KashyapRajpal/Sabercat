// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Sorting study, Stable mode: replace each instance's depth key with the
// Gaussian's sticky bucket (see sort_stable_update.slang).
//
// Runs indirectly over the instance list, in the same slot order the radix
// sort's key buffer uses, so keysDepth[i] pairs with values[i]. The tile
// half of the key is untouched: the radix sort still orders tile first and
// the bucket within the tile.

#include "sort_common.slangh"

StructuredBuffer<InstanceDispatch> dispatch;
StructuredBuffer<uint> values;          // Gaussian index per instance slot
StructuredBuffer<uint> sticky;          // per-Gaussian sticky bucket
RWStructuredBuffer<uint> keysDepth;     // the radix sort's low-word key

[shader("compute")]
[numthreads(SABERCAT_WORKGROUP_SIZE, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    uint i = threadId.x;
    // Slots past the live instance count (capacity headroom) are not sorted.
    if (i >= dispatch[0].instanceCount)
    {
        return;
    }

    keysDepth[i] = sticky[values[i]];
}
