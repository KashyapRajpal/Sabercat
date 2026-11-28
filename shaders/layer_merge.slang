// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Layer merge (Studio phase C2): bakes one layer's Gaussians into the destination
// cloud at a Gaussian offset. This is the GPU port of bakeLayer() in
// src/renderer/layer_transform.cpp, one dispatch per layer. Thread i reads source
// Gaussian i and writes destination Gaussian first + i. Layers write disjoint
// destination ranges and the source is never the destination, so the dispatches
// need no barriers between them.
//
// The host fills params[layer] (SimilarityParams, edit_common.slangh) from the
// layer's BakeFrame and transform; the flags say which optional relight streams
// are carried across. An absent stream is bound to a dummy buffer and is never
// read or written.
//
// params[layer] is read member by member where it is used, never copied into a
// local struct: the struct is 480 bytes and copies of it were a large part of the
// non-identity cost.

#include "edit_common.slangh"

struct LayerMergePush
{
    uint first;    // destination Gaussian offset of this layer
    uint layer;    // index into params
    uint flags;    // LAYER_MERGE_FLAG_* bits
    uint reserved; // zero
};

#define LAYER_MERGE_FLAG_NORMALS 1u
#define LAYER_MERGE_FLAG_BASE_COLOR_ROUGHNESS 2u
#define LAYER_MERGE_FLAG_METALLIC 4u
#define LAYER_MERGE_FLAG_VISIBILITY 8u

[vk::push_constant]
ConstantBuffer<LayerMergePush> push;

// Binding 0: the per-layer placements.
[[vk::binding(0, 0)]] StructuredBuffer<SimilarityParams> params;

// Bindings 1..9: source (the layer's cloud), the GaussianCloud GPU layout.
[[vk::binding(1, 0)]] StructuredBuffer<float4> srcMeans;
[[vk::binding(2, 0)]] StructuredBuffer<float4> srcScales;
[[vk::binding(3, 0)]] StructuredBuffer<float4> srcRotations;
[[vk::binding(4, 0)]] StructuredBuffer<float> srcOpacity;
[[vk::binding(5, 0)]] StructuredBuffer<float4> srcSh;
[[vk::binding(6, 0)]] StructuredBuffer<float4> srcNormals;
[[vk::binding(7, 0)]] StructuredBuffer<float4> srcBaseColorRoughness;
[[vk::binding(8, 0)]] StructuredBuffer<float> srcMetallic;
[[vk::binding(9, 0)]] StructuredBuffer<float> srcVisibility;

// Bindings 10..18: destination, the same streams in the same order.
[[vk::binding(10, 0)]] RWStructuredBuffer<float4> dstMeans;
[[vk::binding(11, 0)]] RWStructuredBuffer<float4> dstScales;
[[vk::binding(12, 0)]] RWStructuredBuffer<float4> dstRotations;
[[vk::binding(13, 0)]] RWStructuredBuffer<float> dstOpacity;
[[vk::binding(14, 0)]] RWStructuredBuffer<float4> dstSh;
[[vk::binding(15, 0)]] RWStructuredBuffer<float4> dstNormals;
[[vk::binding(16, 0)]] RWStructuredBuffer<float4> dstBaseColorRoughness;
[[vk::binding(17, 0)]] RWStructuredBuffer<float> dstMetallic;
[[vk::binding(18, 0)]] RWStructuredBuffer<float> dstVisibility;

[shader("compute")]
[numthreads(SABERCAT_WORKGROUP_SIZE, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    const uint layer = push.layer;
    const uint i = id.x;
    if (i >= params[layer].count)
    {
        return;
    }

    const uint o = push.first + i;
    const uint flags = push.flags;
    const bool identity = params[layer].identity != 0;

    // Coefficients above the source degree carry nothing and are written as zero.
    const uint degree = min(params[layer].shDegree, 3u);
    const uint keptCoefficients = (degree + 1) * (degree + 1);

    dstOpacity[o] = srcOpacity[i];

    if (identity)
    {
        dstMeans[o] = srcMeans[i];
        dstScales[o] = srcScales[i];
        dstRotations[o] = srcRotations[i];
        for (uint k = 0; k < SABERCAT_SH_COEFFS; ++k)
        {
            if (k < keptCoefficients)
            {
                dstSh[o * SABERCAT_SH_COEFFS + k] = srcSh[i * SABERCAT_SH_COEFFS + k];
            }
            else
            {
                dstSh[o * SABERCAT_SH_COEFFS + k] = float4(0.0f);
            }
        }
    }
    else
    {
        // Mean: the full node matrix (w = 1), stored with a zero pad lane.
        const float3 mean = eTransformPoint(params[layer].matrix, srcMeans[i].xyz);
        dstMeans[o] = float4(mean, 0.0f);

        dstRotations[o] = eComposeRotation(params[layer].frameQuat, srcRotations[i]);
        dstScales[o] = float4(params[layer].uniformScale * srcScales[i].xyz, 0.0f);

        // SH: DC unchanged, bands 1..degree rotated. The canonical 16-entry
        // array is zero above the source degree before and after the rotation.
        // The loads are fully unrolled so every index into c is constant.
        float3 c[SABERCAT_SH_COEFFS];
        [ForceUnroll]
        for (uint k = 0; k < SABERCAT_SH_COEFFS; ++k)
        {
            if (k < keptCoefficients)
            {
                c[k] = srcSh[i * SABERCAT_SH_COEFFS + k].xyz;
            }
            else
            {
                c[k] = float3(0.0f);
            }
        }
        eRotateShBands(params, layer, degree, c);
        [ForceUnroll]
        for (uint k = 0; k < SABERCAT_SH_COEFFS; ++k)
        {
            dstSh[o * SABERCAT_SH_COEFFS + k] = float4(c[k], 0.0f);
        }
    }

    if ((flags & LAYER_MERGE_FLAG_NORMALS) != 0)
    {
        const float4 n = srcNormals[i];
        if (identity)
        {
            dstNormals[o] = n;
        }
        else
        {
            dstNormals[o] = float4(eRotateNormal(params[layer].rotRows[0].xyz,
                                                 params[layer].rotRows[1].xyz,
                                                 params[layer].rotRows[2].xyz, n.xyz),
                                   n.w);
        }
    }

    if ((flags & LAYER_MERGE_FLAG_BASE_COLOR_ROUGHNESS) != 0)
    {
        dstBaseColorRoughness[o] = srcBaseColorRoughness[i];
    }

    if ((flags & LAYER_MERGE_FLAG_METALLIC) != 0)
    {
        dstMetallic[o] = srcMetallic[i];
    }

    if ((flags & LAYER_MERGE_FLAG_VISIBILITY) != 0)
    {
        for (uint k = 0; k < SABERCAT_VISIBILITY_COEFFS; ++k)
        {
            dstVisibility[o * SABERCAT_VISIBILITY_COEFFS + k] =
                srcVisibility[i * SABERCAT_VISIBILITY_COEFFS + k];
        }
    }
}
