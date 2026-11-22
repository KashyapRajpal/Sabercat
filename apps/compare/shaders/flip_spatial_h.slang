// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0
// Portions derived from NVlabs FLIP, BSD-3-Clause

// LDR-FLIP pass 2: the horizontal half of FLIP's separable spatial (contrast
// sensitivity) filter, for A and B at once. Taps run from -R to R in FLIP's order,
// with x clamped to the overlap. Output per pixel, from the YCxCz of pass 1:
//   (sum wY * Y, sum wCx * Cx, sum wCz0 * Cz, sum wCz1 * Cz)
// where the weights are kernel[10 + t + R] = (wY, wCx, wCz0, wCz1).

#include "flip_common.slangh"

[vk::binding(1, 0)] StructuredBuffer<float4> yA;
[vk::binding(2, 0)] StructuredBuffer<float4> yB;
[vk::binding(3, 0)] RWStructuredBuffer<float4> interA;
[vk::binding(4, 0)] RWStructuredBuffer<float4> interB;

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= params.width || id.y >= params.height)
    {
        return;
    }

    const int width = int(params.width);
    const int radius = int(params.spatialRadius);
    const uint row = id.y * params.width;

    float4 sumA = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 sumB = float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int t = -radius; t <= radius; t++)
    {
        const int xx = clampEdge(int(id.x) + t, width);
        const uint index = row + uint(xx);
        const float4 weights = kernel[uint(10 + t + radius)];
        const float4 ca = yA[index];
        const float4 cb = yB[index];

        sumA.x += weights.x * ca.x;
        sumA.y += weights.y * ca.y;
        sumA.z += weights.z * ca.z;
        sumA.w += weights.w * ca.z;

        sumB.x += weights.x * cb.x;
        sumB.y += weights.y * cb.y;
        sumB.z += weights.z * cb.z;
        sumB.w += weights.w * cb.z;
    }

    const uint i = row + id.x;
    interA[i] = sumA;
    interB[i] = sumB;
}
