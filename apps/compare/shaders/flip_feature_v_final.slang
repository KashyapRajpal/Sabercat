// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0
// Portions derived from NVlabs FLIP, BSD-3-Clause

// LDR-FLIP pass 5: the vertical half of the feature filters (y clamped), the edge and
// point magnitudes of A and B, their differences, and FLIP's final error:
//   featureDifference = pow(sqrt(0.5) * max(|edgeA - edgeB|, |pointA - pointB|), gqf)
//   error = pow(colorDifference, 1 - featureDifference)
// errorMap[i] holds the color difference from pass 3 on entry and the final error on
// exit (read and written by the same thread). A zero color difference stays exactly 0,
// which also avoids pow(0, negative) when featureDifference exceeds 1.

#include "flip_common.slangh"

[vk::binding(1, 0)] StructuredBuffer<float4> interA;
[vk::binding(2, 0)] StructuredBuffer<float4> interB;
[vk::binding(3, 0)] RWStructuredBuffer<float> errorMap;

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= params.width || id.y >= params.height)
    {
        return;
    }

    const int height = int(params.height);
    const int radius = int(params.featureRadius);
    const uint width = params.width;

    float dxA = 0.0f;
    float ddxA = 0.0f;
    float dyA = 0.0f;
    float ddyA = 0.0f;
    float dxB = 0.0f;
    float ddxB = 0.0f;
    float dyB = 0.0f;
    float ddyB = 0.0f;
    for (int t = -radius; t <= radius; t++)
    {
        const int yy = clampEdge(int(id.y) + t, height);
        const uint index = uint(yy) * width + id.x;
        const float4 weights = kernel[params.featureBase + uint(t + radius)];
        const float4 ia = interA[index];
        const float4 ib = interB[index];

        // Intermediates: x = dx, y = ddx (first and second derivative in x), z = gaussian.
        dxA += weights.x * ia.x;
        ddxA += weights.x * ia.y;
        dyA += weights.y * ia.z;
        ddyA += weights.z * ia.z;

        dxB += weights.x * ib.x;
        ddxB += weights.x * ib.y;
        dyB += weights.y * ib.z;
        ddyB += weights.z * ib.z;
    }

    const float edgeA = sqrt(dxA * dxA + dyA * dyA);
    const float edgeB = sqrt(dxB * dxB + dyB * dyB);
    const float pointA = sqrt(ddxA * ddxA + ddyA * ddyA);
    const float pointB = sqrt(ddxB * ddxB + ddyB * ddyB);

    const float edgeDifference = abs(edgeA - edgeB);
    const float pointDifference = abs(pointA - pointB);
    const float maxDifference = (edgeDifference > pointDifference) ? edgeDifference : pointDifference;

    const float normalizationFactor = 1.0f / sqrt(2.0f);
    const float gqf = kernel[1].x;
    const float featureDifference = pow(normalizationFactor * maxDifference, gqf);

    const uint i = id.y * width + id.x;
    const float colorDifference = errorMap[i];
    errorMap[i] = (colorDifference > 0.0f) ? pow(colorDifference, 1.0f - featureDifference) : 0.0f;
}
