// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Raw trainer parameters -> activated Gaussian arrays, one thread per Gaussian.
//
// Per Gaussian g, 32 floats in `params` at base = g * 32:
//   [0..2]   mean xyz            [3]      pad
//   [4..6]   log scale xyz       [7]      pad
//   [8..11]  raw quaternion xyzw (unnormalized)
//   [12]     logit opacity       [13..15] pad
//   [16 + 4k + c]  SH coefficient k (0..3), channel c (0..2); lane 3 is pad.

struct ActivateParams
{
    uint count;
};

[[vk::binding(0, 0)]] StructuredBuffer<float> params;
[[vk::binding(1, 0)]] RWStructuredBuffer<float4> outMeans;
[[vk::binding(2, 0)]] RWStructuredBuffer<float4> outScales;
[[vk::binding(3, 0)]] RWStructuredBuffer<float4> outRotations;
[[vk::binding(4, 0)]] RWStructuredBuffer<float> outOpacities;
[[vk::binding(5, 0)]] RWStructuredBuffer<float4> outSh;

[vk::push_constant]
ConstantBuffer<ActivateParams> pc;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    const uint g = id.x;
    if (g >= pc.count)
        return;

    const uint base = g * 32u;

    outMeans[g] = float4(params[base + 0u], params[base + 1u], params[base + 2u], 0.0f);

    outScales[g] = float4(exp(params[base + 4u]),
                          exp(params[base + 5u]),
                          exp(params[base + 6u]),
                          0.0f);

    const float4 q = float4(params[base + 8u], params[base + 9u],
                            params[base + 10u], params[base + 11u]);
    const float qq = dot(q, q);
    if (qq == 0.0f)
        outRotations[g] = float4(0.0f, 0.0f, 0.0f, 1.0f);
    else
        outRotations[g] = q / sqrt(qq);

    outOpacities[g] = 1.0f / (1.0f + exp(-params[base + 12u]));

    // Only the DC and band-1 coefficients are trained; the rest stay as they are.
    [unroll]
    for (uint k = 0; k < 4; ++k)
    {
        const uint o = base + 16u + 4u * k;
        outSh[g * 16u + k] = float4(params[o + 0u], params[o + 1u], params[o + 2u], 0.0f);
    }
}
