// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Radix sort, step 1 of 3: per-workgroup digit histogram.
//
// Each workgroup owns a contiguous chunk of SABERCAT_WORKGROUP_SIZE keys
// and counts how many fall in each of the 256 digit buckets. Counts are
// written DIGIT-MAJOR -- counters[digit * groupCount + group] -- because
// then a plain exclusive scan of the flattened array (step 2,
// prefix_sum.slang) yields every (digit, group)'s global offset.

#include "forward_common.slangh"

struct SortParams
{
    uint bitShift;   // which 8-bit digit of the key this pass sorts on
    uint keySelect;  // 0 = sort on the depth word, 1 = the tile word
};

StructuredBuffer<uint> keysTile;
StructuredBuffer<uint> keysDepth;

[vk::push_constant]
ConstantBuffer<SortParams> params;

RWStructuredBuffer<uint> counters;

// Element and group counts come from the GPU (instance_dispatch.slang), the
// same values that sized this indirect dispatch. Declared last: bindings are
// assigned in declaration order, and the host binds this at 3.
StructuredBuffer<InstanceDispatch> dispatch;

groupshared uint localHistogram[SABERCAT_RADIX_BUCKETS];

uint digitOf(uint index)
{
    uint key = params.keySelect == 0 ? keysDepth[index] : keysTile[index];
    return (key >> params.bitShift) & (SABERCAT_RADIX_BUCKETS - 1);
}

[shader("compute")]
[numthreads(SABERCAT_WORKGROUP_SIZE, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID, uint3 localId : SV_GroupThreadID,
          uint3 groupId : SV_GroupID)
{
    // SABERCAT_WORKGROUP_SIZE == SABERCAT_RADIX_BUCKETS == 256, so one
    // thread clears and later writes exactly one bucket.
    localHistogram[localId.x] = 0;
    GroupMemoryBarrierWithGroupSync();

    uint index = threadId.x;
    if (index < dispatch[0].instanceCount)
    {
        uint unused;
        InterlockedAdd(localHistogram[digitOf(index)], 1, unused);
    }
    GroupMemoryBarrierWithGroupSync();

    counters[localId.x * dispatch[0].sortGroups + groupId.x] = localHistogram[localId.x];
}
