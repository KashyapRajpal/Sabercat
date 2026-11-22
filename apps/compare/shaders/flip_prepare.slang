// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0
// Portions derived from NVlabs FLIP, BSD-3-Clause

// LDR-FLIP pass 1: per overlap pixel, the display value of A and B as displayRgb()
// defines it (Display as stored, Linear clamped and sRGB encoded, premultiplied rgb
// unchanged, alpha ignored), clamped to [0,1] (NaN to 0), looked up in the sRGB table
// to linear RGB, and converted to YCxCz (FLIP's LinearRGBToYCxCz). Non-finite rgb
// samples are counted into diag[0]. Overlap-local thread (x, y); pixel i = y * width + x.

#include "flip_common.slangh"

[vk::binding(1, 0)] StructuredBuffer<float4> sourceA;
[vk::binding(2, 0)] StructuredBuffer<float4> sourceB;
[vk::binding(3, 0)] StructuredBuffer<float> srgbTable;   // 65536 entries
[vk::binding(4, 0)] RWStructuredBuffer<float4> yA;
[vk::binding(5, 0)] RWStructuredBuffer<float4> yB;
[vk::binding(6, 0)] RWStructuredBuffer<uint> diag;       // diag[0]: non-finite sample count

// True for NaN and both infinities (the comparison is false for NaN).
bool isNonFiniteSample(float v)
{
    return !(abs(v) <= 3.402823466e38f);
}

// The linear value of one display sample: displayRgb(), then clamp and the sRGB table.
float linearFromSample(float v, uint linear)
{
    // Clamp to [0,1] first (NaN to 0). Linear inputs are sRGB encoded from the clamped value.
    const float display = (linear == 0u) ? v : srgbTransfer(clampUnit(v));
    const float c = clampUnit(display);
    const uint index = uint(c * 65535.0f + 0.5f);
    return srgbTable[index];
}

float3 linearRgbOfSample(float4 s, uint linear)
{
    return float3(linearFromSample(s.r, linear), linearFromSample(s.g, linear), linearFromSample(s.b, linear));
}

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= params.width || id.y >= params.height)
    {
        return;
    }

    const uint i = id.y * params.width + id.x;
    const int cx = params.overlapX + int(id.x);
    const int cy = params.overlapY + int(id.y);
    const uint indexA = uint(cy - params.aOffsetY) * params.aWidth + uint(cx - params.aOffsetX);
    const uint indexB = uint(cy - params.bOffsetY) * params.bWidth + uint(cx - params.bOffsetX);
    const float4 a = sourceA[indexA];
    const float4 b = sourceB[indexB];

    uint nonFinite = 0u;
    nonFinite += isNonFiniteSample(a.r) ? 1u : 0u;
    nonFinite += isNonFiniteSample(a.g) ? 1u : 0u;
    nonFinite += isNonFiniteSample(a.b) ? 1u : 0u;
    nonFinite += isNonFiniteSample(b.r) ? 1u : 0u;
    nonFinite += isNonFiniteSample(b.g) ? 1u : 0u;
    nonFinite += isNonFiniteSample(b.b) ? 1u : 0u;
    if (nonFinite > 0u)
    {
        InterlockedAdd(diag[0], nonFinite);
    }

    // XYZ = M * rgb, then FLIP's LinearRGBToYCxCz (kernel[2..4] is the matrix).
    const float3 xyzA = linearRgbToXyz(linearRgbOfSample(a, params.aLinear));
    const float3 xyzB = linearRgbToXyz(linearRgbOfSample(b, params.bLinear));
    const float3 yccA = xyzToYCxCz(xyzA);
    // Equal samples must give bit-equal YCxCz (the two copies of this code may be scheduled
    // differently by the driver), so that identical inputs give an error of exactly 0.
    const bool same = all(a.rgb == b.rgb) && params.aLinear == params.bLinear;
    yA[i] = float4(yccA, 0.0f);
    yB[i] = same ? float4(yccA, 0.0f) : float4(xyzToYCxCz(xyzB), 0.0f);
}
