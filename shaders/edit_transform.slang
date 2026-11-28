// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Layer-space transform of the selected Gaussians, in place (Studio S2, phase C4): the
// kernel behind EditPass::transformSelected. A selected Gaussian takes the non-identity
// path of layer_merge.slang with the same SimilarityParams, so the result is bit-identical
// to a merge of the same placement: the mean through the node matrix, the rotation composed
// with the frame quaternion, the scale by the uniform scale, SH bands 1..degree rotated
// (degree capped at 3), and the normal rotated by the frame rotation.
//
// Dispatch: one invocation per Gaussian. The host dispatches min(groups, 65535) workgroups
// in x and the rest in y, so one row of the dispatch spans 65535 workgroups and the
// Gaussian index is id.x + id.y * EDIT_TRANSFORM_ROW_THREADS. A layer that fits one row
// (up to 16.7M Gaussians) is dispatched with y = 1.
//
// params[0] is read member by member where it is used, never copied: the struct is 480
// bytes. The host never dispatches an identity placement.

#include "edit_common.slangh"

struct EditTransformPush
{
    uint count;      // Gaussians in the layer
    uint hasNormals; // 1: the normals stream is carried and is transformed
};

[vk::push_constant]
ConstantBuffer<EditTransformPush> push;

// Binding 0: the selection mask, one bit per Gaussian.
[[vk::binding(0, 0)]] StructuredBuffer<uint> mask;
// Binding 1: the layer's placement (element 0).
[[vk::binding(1, 0)]] StructuredBuffer<SimilarityParams> params;
// Bindings 2..6: the layer's GPU streams, transformed in place.
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> means;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> scales;
[[vk::binding(4, 0)]] RWStructuredBuffer<float4> rotations;
[[vk::binding(5, 0)]] RWStructuredBuffer<float4> shCoefficients;
[[vk::binding(6, 0)]] RWStructuredBuffer<float4> normals;

// Invocations in one row of the dispatch: 65535 workgroups of WORKGROUP_SIZE threads.
#define EDIT_TRANSFORM_ROW_THREADS (65535u * SABERCAT_WORKGROUP_SIZE)

[shader("compute")]
[numthreads(SABERCAT_WORKGROUP_SIZE, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    const uint i = id.x + id.y * EDIT_TRANSFORM_ROW_THREADS;
    if (i >= push.count)
    {
        return;
    }
    if (((mask[i >> 5] >> (i & 31u)) & 1u) == 0u)
    {
        return;
    }

    // Coefficients above the source degree carry nothing and are written as zero.
    const uint degree = min(params[0].shDegree, 3u);
    const uint keptCoefficients = (degree + 1) * (degree + 1);

    // Mean: the full node matrix (w = 1), stored with a zero pad lane.
    const float3 mean = eTransformPoint(params[0].matrix, means[i].xyz);
    means[i] = float4(mean, 0.0f);

    rotations[i] = eComposeRotation(params[0].frameQuat, rotations[i]);
    scales[i] = float4(params[0].uniformScale * scales[i].xyz, 0.0f);

    // SH: DC unchanged, bands 1..degree rotated. Every coefficient is loaded into c
    // before any is stored, and the loads are fully unrolled so every index into c is
    // constant.
    float3 c[SABERCAT_SH_COEFFS];
    [ForceUnroll]
    for (uint k = 0; k < SABERCAT_SH_COEFFS; ++k)
    {
        if (k < keptCoefficients)
        {
            c[k] = shCoefficients[i * SABERCAT_SH_COEFFS + k].xyz;
        }
        else
        {
            c[k] = float3(0.0f);
        }
    }
    eRotateShBands(params, 0, degree, c);
    [ForceUnroll]
    for (uint k = 0; k < SABERCAT_SH_COEFFS; ++k)
    {
        shCoefficients[i * SABERCAT_SH_COEFFS + k] = float4(c[k], 0.0f);
    }

    if (push.hasNormals != 0u)
    {
        const float4 n = normals[i];
        normals[i] = float4(eRotateNormal(params[0].rotRows[0].xyz,
                                          params[0].rotRows[1].xyz,
                                          params[0].rotRows[2].xyz, n.xyz),
                            n.w);
    }
}
