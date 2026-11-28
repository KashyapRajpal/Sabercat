// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Per-pixel loss gradients and partial loss sums, one thread per pixel.
//
//   e  = image - targetImage          (RGBA, premultiplied)
//   ed = depth - targetDepth
//   dLdI[p] = colorScale * e
//   dLdD[p] = depthScale * ed
//   partials[group] = (sum of dot(e, e), sum of ed * ed) over the workgroup.
//
// The host divides the sums: loss = sumC / (P * 4) + 0.1 * sumD / P with the
// scales passed as colorScale = 2 / (P * 4) and depthScale = 0.2 / P.

struct LossParams
{
    uint pixelCount;
    float colorScale;
    float depthScale;
};

[[vk::binding(0, 0)]] StructuredBuffer<float4> image;
[[vk::binding(1, 0)]] StructuredBuffer<float> depth;
[[vk::binding(2, 0)]] StructuredBuffer<float4> targetImage;
[[vk::binding(3, 0)]] StructuredBuffer<float> targetDepth;
[[vk::binding(4, 0)]] RWStructuredBuffer<float4> dLdI;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> dLdD;
[[vk::binding(6, 0)]] RWStructuredBuffer<float2> partials;

[vk::push_constant]
ConstantBuffer<LossParams> pc;

static const uint kGroupSize = 256u;

groupshared float sumC[kGroupSize];
groupshared float sumD[kGroupSize];

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uint lane : SV_GroupIndex, uint3 groupId : SV_GroupID)
{
    const uint p = id.x;

    float c = 0.0f;
    float d = 0.0f;
    if (p < pc.pixelCount)
    {
        const float4 e = image[p] - targetImage[p];
        dLdI[p] = pc.colorScale * e;
        const float ed = depth[p] - targetDepth[p];
        dLdD[p] = pc.depthScale * ed;
        c = dot(e, e);
        d = ed * ed;
    }

    sumC[lane] = c;
    sumD[lane] = d;
    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint s = kGroupSize / 2u; s > 0u; s = s / 2u)
    {
        if (lane < s)
        {
            sumC[lane] = sumC[lane] + sumC[lane + s];
            sumD[lane] = sumD[lane] + sumD[lane + s];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (lane == 0u)
        partials[groupId.x] = float2(sumC[0], sumD[0]);
}
