// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Sabercat Compare metrics: per-workgroup partial sums of squared differences over
// the overlap rectangle, in display-space RGB as displayRgb() in flip_hdr.cpp defines
// it (Display samples as stored, Linear samples clamped to [0,1] and sRGB encoded,
// premultiplied rgb unchanged). Each workgroup covers 256 consecutive overlap pixels
// (row-major). It reduces them with a shared-memory tree and writes one float. The
// host sums the partials in double.
//
// Every group with index >= groupCount returns before the barrier. That is uniform
// over the group, so the barriers stay valid.

static const uint kGroupSize = 256u;

// Push constants, mirrored by GpuMetricsParams in metrics_gpu.cpp (16 x 4 bytes).
struct MetricsParams
{
    int aOffsetX;
    int aOffsetY;
    int bOffsetX;
    int bOffsetY;
    int overlapX;
    int overlapY;
    uint overlapWidth;
    uint overlapHeight;
    uint aWidth;
    uint aHeight;
    uint bWidth;
    uint bHeight;
    uint aLinear;
    uint bLinear;
    uint groupCount;
    uint groupRowLength;
};

[vk::push_constant] ConstantBuffer<MetricsParams> params;

[vk::binding(0, 0)] StructuredBuffer<float4> sourceA;
[vk::binding(1, 0)] StructuredBuffer<float4> sourceB;
[vk::binding(2, 0)] RWStructuredBuffer<float> partials;

groupshared float partialSums[256];

float displayValue(float v, uint linear)
{
    if (linear == 0u)
    {
        return v;
    }
    // Clamp to [0,1] (NaN is 0), then the sRGB transfer function.
    const float c = (v > 0.0f) ? min(v, 1.0f) : 0.0f;
    return (c <= 0.0031308f) ? 12.92f * c : 1.055f * pow(c, 1.0f / 2.4f) - 0.055f;
}

[numthreads(256, 1, 1)]
void main(uint3 groupId : SV_GroupID, uint3 threadId : SV_GroupThreadID)
{
    const uint group = groupId.y * params.groupRowLength + groupId.x;
    if (group >= params.groupCount)
    {
        return;
    }

    const uint local = threadId.x;
    const uint pixel = group * kGroupSize + local;
    float value = 0.0f;
    if (pixel < params.overlapWidth * params.overlapHeight)
    {
        const uint lx = pixel % params.overlapWidth;
        const uint ly = pixel / params.overlapWidth;
        const int cx = params.overlapX + int(lx);
        const int cy = params.overlapY + int(ly);

        const uint indexA = uint(cy - params.aOffsetY) * params.aWidth + uint(cx - params.aOffsetX);
        const uint indexB = uint(cy - params.bOffsetY) * params.bWidth + uint(cx - params.bOffsetX);
        const float4 a = sourceA[indexA];
        const float4 b = sourceB[indexB];

        const float d0 = displayValue(a.r, params.aLinear) - displayValue(b.r, params.bLinear);
        const float d1 = displayValue(a.g, params.aLinear) - displayValue(b.g, params.bLinear);
        const float d2 = displayValue(a.b, params.aLinear) - displayValue(b.b, params.bLinear);
        value = d0 * d0 + d1 * d1 + d2 * d2;
    }

    partialSums[local] = value;
    GroupMemoryBarrierWithGroupSync();
    for (uint stride = kGroupSize / 2u; stride > 0u; stride >>= 1u)
    {
        if (local < stride)
        {
            partialSums[local] += partialSums[local + stride];
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (local == 0u)
    {
        partials[group] = partialSums[0];
    }
}
