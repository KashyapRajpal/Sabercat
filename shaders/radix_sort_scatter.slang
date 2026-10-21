// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Radix sort, step 3 of 3: write each key/value to its sorted position.
//
// Destination = the scanned global offset for this (digit, workgroup),
// plus this element's rank among same-digit elements EARLIER IN ITS OWN
// CHUNK. That rank is what makes the sort stable, and stability is not
// optional: an LSD radix sort that reorders equal digits destroys the
// ordering established by previous passes.

#include "forward_common.slangh"

struct SortParams
{
    uint bitShift;
    uint keySelect;
};

StructuredBuffer<uint> keysTile;
StructuredBuffer<uint> keysDepth;
StructuredBuffer<uint> values;

[vk::push_constant]
ConstantBuffer<SortParams> params;
StructuredBuffer<uint> offsets; // scanned (digit, group) start offsets (prefix_sum)

RWStructuredBuffer<uint> outKeysTile;
RWStructuredBuffer<uint> outKeysDepth;
RWStructuredBuffer<uint> outValues;

// Element and group counts (instance_dispatch.slang).
StructuredBuffer<InstanceDispatch> dispatch;

groupshared uint chunkDigits[SABERCAT_WORKGROUP_SIZE];

[shader("compute")]
[numthreads(SABERCAT_WORKGROUP_SIZE, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID, uint3 localId : SV_GroupThreadID,
          uint3 groupId : SV_GroupID)
{
    uint index = threadId.x;
    bool active = index < dispatch[0].instanceCount;

    uint key = 0;
    uint digit = SABERCAT_RADIX_BUCKETS; // sentinel: never matches a real digit
    if (active)
    {
        key = params.keySelect == 0 ? keysDepth[index] : keysTile[index];
        digit = (key >> params.bitShift) & (SABERCAT_RADIX_BUCKETS - 1);
    }
    chunkDigits[localId.x] = digit;
    GroupMemoryBarrierWithGroupSync();

    if (!active)
    {
        return;
    }

    // Rank among earlier same-digit elements in this chunk. A serial scan
    // over at most 256 shared-memory entries -- the simple, obviously
    // stable formulation. (A ballot/match-based rank would be faster; worth
    // revisiting once the pipeline is proven correct.)
    uint rank = 0;
    for (uint i = 0; i < localId.x; ++i)
    {
        if (chunkDigits[i] == digit)
        {
            ++rank;
        }
    }

    uint destination = offsets[digit * dispatch[0].sortGroups + groupId.x] + rank;

    outKeysTile[destination] = keysTile[index];
    outKeysDepth[destination] = keysDepth[index];
    outValues[destination] = values[index];
}
