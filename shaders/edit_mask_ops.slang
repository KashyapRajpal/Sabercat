// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Selection mask operations (Studio phase C3): one kernel with a mode in the push
// constants, the mask-level operations EditPass runs on a SelectionMask. The mask
// holds one uint per 32 Gaussians: bit (i & 31) of word (i >> 5) is Gaussian i,
// and bits at or beyond the layer's count are always zero.
//
// Every mode is a grid-stride loop. The host dispatches min(ceil(total / 256), 65535)
// workgroups, so a layer of more than 16.7M Gaussians still fits the dispatch limit:
// each invocation walks its items with stride push.groups * SABERCAT_WORKGROUP_SIZE.
// With at most one item per invocation (the layers of up to 16.7M Gaussians) the
// loop runs once, as the per-item kernel always did.
//
//   0 CLEAR     per word: the word is 0.
//   1 ALL       per word: bits [0, count) set, 0 past count.
//   2 INVERT    per word: flip the bits below count, 0 past count.
//   3 POPCOUNT  per word: countbits, summed per invocation and reduced per workgroup;
//               thread 0 writes countPartials[group].
//   4 CENTROID  per Gaussian: a selected Gaussian adds its local-space mean and a
//               count of 1, summed per invocation and reduced per workgroup into
//               centroidPartials[group] = (sum.xyz, count).
//   5 EXPAND    per Gaussian: flags[i] = 1 for a selected Gaussian, or the inverse
//               when push.flag is 1 (flag 1 = keep the unselected).
//
// Every reduction partial is per workgroup, so the host adds them in double. No
// invocation writes a word it does not own, so the word-wise modes need no atomics.
//
// Plain if/else throughout, no ternaries choosing between temporaries (a known
// Slang GLSL miscompile risk).

#include "config.slangh"

struct EditMaskPush
{
    uint mode;      // EDIT_MASK_MODE_*
    uint count;     // Gaussians in the layer (or the mask's live prefix)
    uint wordCount; // mask words the word-wise modes cover
    uint flag;      // EXPAND: 1 writes the inverse (keep unselected)
    uint groups;    // workgroups the host dispatched: the grid stride is groups * WORKGROUP_SIZE
};

#define EDIT_MASK_MODE_CLEAR 0u
#define EDIT_MASK_MODE_ALL 1u
#define EDIT_MASK_MODE_INVERT 2u
#define EDIT_MASK_MODE_POPCOUNT 3u
#define EDIT_MASK_MODE_CENTROID 4u
#define EDIT_MASK_MODE_EXPAND 5u

[vk::push_constant]
ConstantBuffer<EditMaskPush> push;

// Binding 0: the selection mask, one uint per 32 Gaussians.
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> mask;
// Binding 1: the layer's local-space means, read by CENTROID only.
[[vk::binding(1, 0)]] StructuredBuffer<float4> means;
// Binding 2: per-Gaussian flags, written by EXPAND only.
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> flags;
// Binding 3: one popcount per workgroup, written by POPCOUNT only.
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> countPartials;
// Binding 4: one (sum.xyz, count) per workgroup, written by CENTROID only.
[[vk::binding(4, 0)]] RWStructuredBuffer<float4> centroidPartials;

groupshared uint sCount[SABERCAT_WORKGROUP_SIZE];
groupshared float4 sCentroid[SABERCAT_WORKGROUP_SIZE];

// The bits [0, count) of word w: the bits of the word that lie below count.
uint validBits(uint w, uint count)
{
    const uint lo = w * 32u;
    if (lo >= count)
    {
        return 0u;
    }
    const uint n = count - lo;
    if (n >= 32u)
    {
        return 0xFFFFFFFFu;
    }
    return (1u << n) - 1u;
}

// Whether Gaussian i is selected.
bool isSet(uint i)
{
    return ((mask[i >> 5u] >> (i & 31u)) & 1u) != 0u;
}

[shader("compute")]
[numthreads(SABERCAT_WORKGROUP_SIZE, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uint3 groupId : SV_GroupID,
          uint3 localId : SV_GroupThreadID)
{
    const uint mode = push.mode;
    const uint lane = localId.x;
    const uint group = groupId.x;
    const uint stride = push.groups * SABERCAT_WORKGROUP_SIZE;

    if (mode == EDIT_MASK_MODE_CLEAR)
    {
        for (uint w = id.x; w < push.wordCount; w += stride)
        {
            mask[w] = 0u;
        }
        return;
    }

    if (mode == EDIT_MASK_MODE_ALL)
    {
        for (uint w = id.x; w < push.wordCount; w += stride)
        {
            mask[w] = validBits(w, push.count);
        }
        return;
    }

    if (mode == EDIT_MASK_MODE_INVERT)
    {
        for (uint w = id.x; w < push.wordCount; w += stride)
        {
            mask[w] = (~mask[w]) & validBits(w, push.count);
        }
        return;
    }

    // The reductions below reach their barriers in every thread of the group: the
    // mode is a push constant, so the branch is uniform across the workgroup.
    if (mode == EDIT_MASK_MODE_POPCOUNT)
    {
        uint bits = 0u;
        for (uint w = id.x; w < push.wordCount; w += stride)
        {
            bits = bits + countbits(mask[w] & validBits(w, push.count));
        }
        sCount[lane] = bits;
        GroupMemoryBarrierWithGroupSync();

        for (uint s = SABERCAT_WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u)
        {
            if (lane < s)
            {
                sCount[lane] = sCount[lane] + sCount[lane + s];
            }
            GroupMemoryBarrierWithGroupSync();
        }

        if (lane == 0u)
        {
            countPartials[group] = sCount[0];
        }
        return;
    }

    if (mode == EDIT_MASK_MODE_CENTROID)
    {
        float4 sum = float4(0.0f);
        for (uint i = id.x; i < push.count; i += stride)
        {
            if (isSet(i))
            {
                sum = sum + float4(means[i].xyz, 1.0f);
            }
        }
        sCentroid[lane] = sum;
        GroupMemoryBarrierWithGroupSync();

        for (uint s = SABERCAT_WORKGROUP_SIZE / 2u; s > 0u; s = s >> 1u)
        {
            if (lane < s)
            {
                sCentroid[lane] = sCentroid[lane] + sCentroid[lane + s];
            }
            GroupMemoryBarrierWithGroupSync();
        }

        if (lane == 0u)
        {
            centroidPartials[group] = sCentroid[0];
        }
        return;
    }

    if (mode == EDIT_MASK_MODE_EXPAND)
    {
        for (uint i = id.x; i < push.count; i += stride)
        {
            uint bit = 0u;
            if (isSet(i))
            {
                bit = 1u;
            }
            if (push.flag != 0u)
            {
                bit = 1u - bit;
            }
            flags[i] = bit;
        }
    }
}
