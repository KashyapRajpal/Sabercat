// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Selection merge (Studio S2, phase C3): gathers one layer's selection bits into the
// merged scene's selection mask, one dispatch per layer. The layer's mask is local to
// the layer: bit g of its words is the layer's Gaussian g, counted from the layer's
// first Gaussian. The merged mask has one bit per Gaussian of the merged scene.
//
// Thread t handles merged word w = firstWord + t, which covers merged Gaussians
// [32w, 32w + 32). It ORs in the bits of the Gaussians this layer owns and writes the
// word once, so no two threads of a dispatch touch the same word and no atomics are
// needed. A word that straddles two layers is written by both layers' dispatches; the
// host puts a barrier between dispatches, so the second one ORs onto the first. The
// host zero-fills the merged mask before the first dispatch.

#include "config.slangh"

// Push constants: four 4-byte uints, 16 bytes, no padding.
struct SelectionMergePush
{
    uint first;        // the layer's offset in the merged scene
    uint count;        // the layer's Gaussian count
    uint firstWord;    // first >> 5: the first merged word the layer touches
    uint wordsInRange; // merged words the layer touches
};

[vk::push_constant]
ConstantBuffer<SelectionMergePush> push;

// Binding 0: the layer's selection mask, layer-local bits.
[[vk::binding(0, 0)]] StructuredBuffer<uint> layerMask;
// Binding 1: the merged selection mask, read-modify-written word by word.
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> mergedMask;

[shader("compute")]
[numthreads(SABERCAT_WORKGROUP_SIZE, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    const uint t = id.x;
    if (t >= push.wordsInRange)
    {
        return;
    }

    const uint w = push.firstWord + t;
    const uint layerEnd = push.first + push.count;
    uint bits = 0u;
    for (uint j = 0; j < 32; ++j)
    {
        const uint g = 32u * w + j; // merged Gaussian this bit stands for
        if (g >= push.first && g < layerEnd)
        {
            const uint local = g - push.first;
            if ((layerMask[local >> 5] & (1u << (local & 31u))) != 0u)
            {
                bits |= 1u << j;
            }
        }
    }

    mergedMask[w] = mergedMask[w] | bits;
}
