// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0
// Portions derived from NVlabs FLIP, BSD-3-Clause

// HDR-FLIP pass 1, once per exposure: per overlap pixel, the linear RGB of A and B (Linear
// sources as stored, Display sources decoded through the sRGB table), scaled by the exposure,
// clamped to a finite range, tone mapped, clamped to [0,1], and converted to YCxCz. Overlap-local
// thread (x, y); pixel i = y * width + x. Non-finite rgb samples are counted into diag[0] only
// for the first exposure (hdrFlags bit 0).

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

// The linear value of one sample. A non-finite value (NaN, +inf, -inf) is 0 for a Linear and for a
// Display source alike, so +inf of a Display source is 0, not white. A Linear source is taken as
// stored. A Display source goes through the sRGB table after clamping to [0,1].
float linearFromSample(float v, uint linear)
{
    if (isNonFiniteSample(v))
    {
        return 0.0f;
    }
    if (linear != 0u)
    {
        return v;
    }
    const float c = clampUnit(v);
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

    if ((params.hdrFlags & 1u) != 0u)
    {
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
    }

    // Exposure, then the overflow guard, then the tone mapper, then FLIP's clamp to [0,1].
    // The guard keeps c * c finite in the rational tone mapper; it is far above any value
    // a [-126, 126] exposure range can reach from a float sample.
    float3 rgbA = linearRgbOfSample(a, params.aLinear) * params.exposureScale;
    float3 rgbB = linearRgbOfSample(b, params.bLinear) * params.exposureScale;
    rgbA = clamp(rgbA, float3(-1e15f), float3(1e15f));
    rgbB = clamp(rgbB, float3(-1e15f), float3(1e15f));
    rgbA = clampUnit3(toneMap(rgbA, params.toneMapper));
    rgbB = clampUnit3(toneMap(rgbB, params.toneMapper));

    // XYZ = M * rgb, then FLIP's LinearRGBToYCxCz (kernel[2..4] is the matrix).
    const float3 yccA = xyzToYCxCz(linearRgbToXyz(rgbA));
    // Equal samples must give bit-equal YCxCz (the two copies of this code may be scheduled
    // differently by the driver), so that identical inputs give an error of exactly 0.
    const bool same = all(a.rgb == b.rgb) && params.aLinear == params.bLinear;
    yA[i] = float4(yccA, 0.0f);
    yB[i] = same ? float4(yccA, 0.0f) : float4(xyzToYCxCz(linearRgbToXyz(rgbB)), 0.0f);
}
