// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Compaction and expansion of a layer's Gaussian streams (Studio S2, phase C4): one
// generic word-level kernel for the delete, duplicate and undo paths of EditPass. A
// stream is viewed as an array of uint32 words, `wordsPerElement` words per element,
// so the same kernel serves every stream: vec4 streams 4 words, SH 64, scalars 1,
// visibility 16, and the index list 1.
//
// The dispatch is a grid-stride loop. A host dispatch is capped at 65535 workgroups,
// and the SH stream needs count * 64 threads, so each invocation walks the flat work
// index t = id.x, id.x + stride, ... with stride = push.groups * SABERCAT_WORKGROUP_SIZE
// (push.groups is the number of workgroups the host dispatched). For the word-wise
// modes the work index is t = element * wordsPerElement + word; for INDEX and REMAP it
// is the element itself.
//
// flags holds the keep (or select) flag of each element and scan its exclusive scan.
//
// Modes (push.mode):
//   GATHER   flags[g] != 0: dst[scan[g] * wpe + w] = src[g * wpe + w]
//   INDEX    flags[g] != 0: dst[scan[g]] = g
//   SCATTER  element j < count: g = aux[j]; dst[g * wpe + w] = src[j * wpe + w]
//   COMPACT  as GATHER (delete: compact a stream into scratch)
//   APPEND   flags[g] != 0: dst[(base + scan[g]) * wpe + w] = src[g * wpe + w]
//   EXPAND   g < count: flags[g] != 0: dst[g * wpe + w] = src[scan[g] * wpe + w];
//            otherwise dst[g * wpe + w] = aux[(g - scan[g]) * wpe + w]
//   REMAP    dst[g] = flags[g] != 0 ? scan[g] : 0xFFFFFFFF (written as if/else)
//
// Plain if/else throughout, no ternaries choosing between temporaries (a known
// Slang GLSL miscompile risk).

#include "config.slangh"

#define EDIT_COMPACT_MODE_GATHER 0u
#define EDIT_COMPACT_MODE_INDEX 1u
#define EDIT_COMPACT_MODE_SCATTER 2u
#define EDIT_COMPACT_MODE_COMPACT 3u
#define EDIT_COMPACT_MODE_APPEND 4u
#define EDIT_COMPACT_MODE_EXPAND 5u
#define EDIT_COMPACT_MODE_REMAP 6u

struct EditCompactPush
{
    uint mode;            // EDIT_COMPACT_MODE_*
    uint count;           // elements (the original count for EXPAND and APPEND)
    uint wordsPerElement; // words per element (ignored by INDEX and REMAP)
    uint base;            // APPEND: element offset of the first appended element
    uint groups;          // workgroups the host dispatched: stride = groups * WORKGROUP_SIZE
};

[vk::push_constant]
ConstantBuffer<EditCompactPush> push;

// Binding 0: keep (or select) flag per element.
[[vk::binding(0, 0)]] StructuredBuffer<uint> flags;
// Binding 1: exclusive scan of flags.
[[vk::binding(1, 0)]] StructuredBuffer<uint> scan;
// Binding 2: the source words. The compacted stream for EXPAND, the packed values for
// SCATTER, the stream itself for GATHER, COMPACT and APPEND.
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> src;
// Binding 3: the destination words. src and dst may be the same buffer (APPEND): the
// appended range [base, base + kept) is disjoint from the source range [0, count).
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> dst;
// Binding 4: the index list for SCATTER, the packed deleted elements for EXPAND.
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> aux;

[shader("compute")]
[numthreads(SABERCAT_WORKGROUP_SIZE, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    const uint mode = push.mode;
    const uint count = push.count;
    const uint wpe = push.wordsPerElement;
    const uint base = push.base;
    const uint stride = push.groups * SABERCAT_WORKGROUP_SIZE;

    if (mode == EDIT_COMPACT_MODE_INDEX)
    {
        for (uint g = id.x; g < count; g += stride)
        {
            if (flags[g] != 0u)
            {
                dst[scan[g]] = g;
            }
        }
        return;
    }

    if (mode == EDIT_COMPACT_MODE_REMAP)
    {
        for (uint g = id.x; g < count; g += stride)
        {
            if (flags[g] != 0u)
            {
                dst[g] = scan[g];
            }
            else
            {
                dst[g] = 0xFFFFFFFFu;
            }
        }
        return;
    }

    // The word-wise modes: t indexes the words of the stream, e = t / wpe is the
    // element and w = t - e * wpe the word within it.
    const uint total = count * wpe;
    for (uint t = id.x; t < total; t += stride)
    {
        const uint e = t / wpe;
        const uint w = t - e * wpe;

        if (mode == EDIT_COMPACT_MODE_SCATTER)
        {
            dst[aux[e] * wpe + w] = src[t];
        }
        else if (mode == EDIT_COMPACT_MODE_EXPAND)
        {
            if (flags[e] != 0u)
            {
                dst[t] = src[scan[e] * wpe + w];
            }
            else
            {
                dst[t] = aux[(e - scan[e]) * wpe + w];
            }
        }
        else if (mode == EDIT_COMPACT_MODE_APPEND)
        {
            if (flags[e] != 0u)
            {
                dst[(base + scan[e]) * wpe + w] = src[t];
            }
        }
        else if (mode == EDIT_COMPACT_MODE_GATHER || mode == EDIT_COMPACT_MODE_COMPACT)
        {
            if (flags[e] != 0u)
            {
                dst[scan[e] * wpe + w] = src[t];
            }
        }
    }
}
