// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0
// Portions derived from NVlabs FLIP, BSD-3-Clause

// LDR-FLIP pass 4: the horizontal half of FLIP's feature (edge and point) filters, for
// A and B at once. The Y channel is normalized as FLIP does (Y / 116 + 16 / 116), and
// filtered with the taps kernel[featureBase + t + R] = (gaussian, first derivative,
// second derivative). Output per pixel: (dx, ddx, gaussian, 0), as FLIP's
// intermediateFeatures. x is clamped to the overlap.

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
    const int radius = int(params.featureRadius);
    const uint row = id.y * params.width;
    const float oneOver116 = 1.0f / 116.0f;
    const float sixteenOver116 = 16.0f / 116.0f;

    float dxA = 0.0f;
    float ddxA = 0.0f;
    float gaussA = 0.0f;
    float dxB = 0.0f;
    float ddxB = 0.0f;
    float gaussB = 0.0f;
    for (int t = -radius; t <= radius; t++)
    {
        const int xx = clampEdge(int(id.x) + t, width);
        const uint index = row + uint(xx);
        const float4 weights = kernel[params.featureBase + uint(t + radius)];
        const float yNormA = yA[index].x * oneOver116 + sixteenOver116;
        const float yNormB = yB[index].x * oneOver116 + sixteenOver116;

        dxA += weights.y * yNormA;
        ddxA += weights.z * yNormA;
        gaussA += weights.x * yNormA;

        dxB += weights.y * yNormB;
        ddxB += weights.z * yNormB;
        gaussB += weights.x * yNormB;
    }

    const uint i = row + id.x;
    interA[i] = float4(dxA, ddxA, gaussA, 0.0f);
    interB[i] = float4(dxB, ddxB, gaussB, 0.0f);
}
