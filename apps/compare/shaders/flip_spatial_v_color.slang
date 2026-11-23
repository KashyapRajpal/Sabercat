// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0
// Portions derived from NVlabs FLIP, BSD-3-Clause

// LDR-FLIP pass 3: the vertical half of the spatial filter (y clamped to the overlap),
// then FLIP's color difference: filtered YCxCz to linear RGB (clamped to [0,1]) to
// CIE L*a*b*, Hunt adjustment on a and b, HyAB, the gqc power and the re-map to [0,1].
// Writes errorMap[i] for the overlap pixel i = y * width + x.

#include "flip_common.slangh"

[vk::binding(1, 0)] StructuredBuffer<float4> interA;
[vk::binding(2, 0)] StructuredBuffer<float4> interB;
[vk::binding(3, 0)] RWStructuredBuffer<float> errorMap;

// Filtered YCxCz (x = Y, y = Cx, z = Cz0 + Cz1) to Hunt-adjusted CIE L*a*b*.
float3 filteredToLab(float4 f)
{
    const float3 ycc = float3(f.x, f.y, f.z + f.w);
    const float3 rgb = clampUnit3(xyzToLinearRgb(yCxCzToXyz(ycc)));
    float3 lab = xyzToLab(linearRgbToXyz(rgb));
    lab.y = hunt(lab.x, lab.y);
    lab.z = hunt(lab.x, lab.z);
    return lab;
}

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= params.width || id.y >= params.height)
    {
        return;
    }

    const int height = int(params.height);
    const int radius = int(params.spatialRadius);
    const uint width = params.width;

    float4 filteredA = float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 filteredB = float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int t = -radius; t <= radius; t++)
    {
        const int yy = clampEdge(int(id.y) + t, height);
        const uint index = uint(yy) * width + id.x;
        const float4 weights = kernel[uint(10 + t + radius)];
        const float4 ia = interA[index];
        const float4 ib = interB[index];

        filteredA.x += weights.x * ia.x;
        filteredA.y += weights.y * ia.y;
        filteredA.z += weights.z * ia.z;
        filteredA.w += weights.w * ia.w;

        filteredB.x += weights.x * ib.x;
        filteredB.y += weights.y * ib.y;
        filteredB.z += weights.z * ib.z;
        filteredB.w += weights.w * ib.w;
    }

    // Equal filtered values have a distance of exactly 0. The driver may fuse the two
    // copies of the color chain below differently, which would leave a rounding residue
    // of about 1e-5 for identical inputs; this keeps them at exactly 0.
    float d = 0.0f;
    if (any(filteredA != filteredB))
    {
        const float3 labA = filteredToLab(filteredA);
        const float3 labB = filteredToLab(filteredB);
        const float distance = hyab(labA, labB);
        // FLIP: colorDifference = HyAB^gqc, then the re-map. pow(0, y) is not guaranteed to be 0.
        const float gqc = kernel[0].w;
        d = (distance > 0.0f) ? pow(distance, gqc) : 0.0f;
    }
    errorMap[id.y * width + id.x] = colorRemap(d);
}
