// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Device-wide exclusive prefix sum, used twice per frame:
//
//   - per-Gaussian tile counts -> each Gaussian's base slot in the instance
//     list (input = projected[].tileCount, read with a stride);
//   - the radix sort's digit-major (digit, workgroup) counts -> global
//     scatter offsets, once per radix pass. Because the counts are stored
//     digit-major, the exclusive scan of the flattened array IS the offset
//     of every (digit, workgroup) pair -- earlier digits first, then earlier
//     workgroups of the same digit -- which is what keeps the sort stable.
//
// Three phases of one kernel, selected by params.phase, with a barrier
// between dispatches:
//
//   0  block scan: workgroup g scans elements [g*BLOCK, (g+1)*BLOCK) into
//      `output` and writes the block's total to blockSums[g].
//   1  block-sum scan: ONE workgroup scans blockSums in place (chunked with
//      a carry, so any block count works; a few thousand at most here).
//   2  add: every element of block g gets blockSums[g] added.
//
// Replaces two single-workgroup scans (the old prefix_sum looped over all
// Gaussians in 256-element chunks; the old radix_sort_scan had one thread
// per digit walk its whole row serially) that together cost ~120 ms per
// frame on a 1.16M-Gaussian scene. All integer, so results are bit-identical.

#include "forward_common.slangh"

#define SCAN_PHASE_BLOCKS 0
#define SCAN_PHASE_BLOCK_SUMS 1
#define SCAN_PHASE_ADD 2

// Push-constant `countSlot` value meaning "use params.elementCount".
#define SCAN_COUNT_FROM_PARAMS 0xFFFFFFFFu

struct ScanParams
{
    uint elementCount; // used when countSlot == SCAN_COUNT_FROM_PARAMS
    uint countSlot;    // else: uint index into `dispatchState` holding the count
    uint inputStride;  // in uints (12 for ProjectedGaussian, 1 for a plain array)
    uint inputOffset;  // in uints, within each element
    uint phase;
};

[vk::push_constant]
ConstantBuffer<ScanParams> params;

StructuredBuffer<uint> input;
RWStructuredBuffer<uint> output;
RWStructuredBuffer<uint> blockSums;
// InstanceDispatch viewed as uints: the radix counter scan's length is only
// known on the GPU (instance_dispatch.slang).
StructuredBuffer<uint> dispatchState;

groupshared uint blockValues[SABERCAT_SCAN_BLOCK];
groupshared uint threadTotals[SABERCAT_WORKGROUP_SIZE];

uint elementCount()
{
    return params.countSlot == SCAN_COUNT_FROM_PARAMS ? params.elementCount
                                                      : dispatchState[params.countSlot];
}

// Exclusive scan of blockValues (already loaded) in place; returns the
// block total. Each thread owns SCAN_ITEMS_PER_THREAD consecutive values:
// a serial scan of those, then a Hillis-Steele scan of the per-thread totals.
uint scanSharedBlock(uint thread)
{
    uint first = thread * SABERCAT_SCAN_ITEMS_PER_THREAD;
    uint running = 0;
    for (uint k = 0; k < SABERCAT_SCAN_ITEMS_PER_THREAD; ++k)
    {
        uint value = blockValues[first + k];
        blockValues[first + k] = running;
        running += value;
    }
    threadTotals[thread] = running;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = 1; stride < SABERCAT_WORKGROUP_SIZE; stride <<= 1)
    {
        uint addend = thread >= stride ? threadTotals[thread - stride] : 0;
        GroupMemoryBarrierWithGroupSync();
        threadTotals[thread] += addend;
        GroupMemoryBarrierWithGroupSync();
    }

    // Inclusive -> exclusive base for this thread's run.
    uint base = threadTotals[thread] - running;
    for (uint k = 0; k < SABERCAT_SCAN_ITEMS_PER_THREAD; ++k)
    {
        blockValues[first + k] += base;
    }
    uint total = threadTotals[SABERCAT_WORKGROUP_SIZE - 1];
    GroupMemoryBarrierWithGroupSync();
    return total;
}

[shader("compute")]
[numthreads(SABERCAT_WORKGROUP_SIZE, 1, 1)]
void main(uint3 localId : SV_GroupThreadID, uint3 groupId : SV_GroupID)
{
    uint thread = localId.x;
    uint count = elementCount();

    if (params.phase == SCAN_PHASE_BLOCKS)
    {
        uint blockStart = groupId.x * SABERCAT_SCAN_BLOCK;
        // Coalesced load (consecutive threads, consecutive elements) into
        // shared memory; the scan then reads it back per-thread-contiguous.
        for (uint k = 0; k < SABERCAT_SCAN_ITEMS_PER_THREAD; ++k)
        {
            uint local = k * SABERCAT_WORKGROUP_SIZE + thread;
            uint index = blockStart + local;
            blockValues[local] =
                index < count ? input[index * params.inputStride + params.inputOffset] : 0;
        }
        GroupMemoryBarrierWithGroupSync();

        uint total = scanSharedBlock(thread);

        for (uint k = 0; k < SABERCAT_SCAN_ITEMS_PER_THREAD; ++k)
        {
            uint local = k * SABERCAT_WORKGROUP_SIZE + thread;
            uint index = blockStart + local;
            if (index < count)
            {
                output[index] = blockValues[local];
            }
        }
        if (thread == 0)
        {
            blockSums[groupId.x] = total;
        }
    }
    else if (params.phase == SCAN_PHASE_BLOCK_SUMS)
    {
        uint blockCount = (count + SABERCAT_SCAN_BLOCK - 1) / SABERCAT_SCAN_BLOCK;
        uint carry = 0;
        for (uint chunk = 0; chunk < blockCount; chunk += SABERCAT_SCAN_BLOCK)
        {
            for (uint k = 0; k < SABERCAT_SCAN_ITEMS_PER_THREAD; ++k)
            {
                uint local = k * SABERCAT_WORKGROUP_SIZE + thread;
                uint index = chunk + local;
                blockValues[local] = index < blockCount ? blockSums[index] : 0;
            }
            GroupMemoryBarrierWithGroupSync();

            uint total = scanSharedBlock(thread);

            for (uint k = 0; k < SABERCAT_SCAN_ITEMS_PER_THREAD; ++k)
            {
                uint local = k * SABERCAT_WORKGROUP_SIZE + thread;
                uint index = chunk + local;
                if (index < blockCount)
                {
                    blockSums[index] = carry + blockValues[local];
                }
            }
            carry += total;
            // blockValues is reloaded next chunk; every thread must be done
            // reading it first.
            GroupMemoryBarrierWithGroupSync();
        }
    }
    else // SCAN_PHASE_ADD
    {
        if (groupId.x == 0)
        {
            return; // block 0's base is 0
        }
        uint base = blockSums[groupId.x];
        uint blockStart = groupId.x * SABERCAT_SCAN_BLOCK;
        for (uint k = 0; k < SABERCAT_SCAN_ITEMS_PER_THREAD; ++k)
        {
            uint index = blockStart + k * SABERCAT_WORKGROUP_SIZE + thread;
            if (index < count)
            {
                output[index] += base;
            }
        }
    }
}
