// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// One VkAabbPositionsKHR per Gaussian, the procedural-geometry input of
// GaussianAccelerationStructure's BLAS. Each box is the exact
// axis-aligned bound of the Gaussian's support ellipsoid -- 3 sigma,
// tightened by opacity (gaussianSupportSigmas: a faint Gaussian's alpha
// clears the 1/255 skip only well inside 3 sigma) -- so a ray that can see a
// Gaussian always reaches it as a candidate, and no other ray does.
//
// A Gaussian that can never contribute (gaussianRayFrame false: opacity below
// 1/255, zero/non-finite scale or quaternion, non-finite mean) gets a
// COLLAPSED box -- min = max = its mean (the origin if the mean is not
// finite) -- rather than Vulkan's inactive "min.x = NaN" box. Vulkan forbids
// an update from changing which primitives are active, and opacity moves
// across 1/255 freely between refits during training; a collapsed box stays
// active, costs a leaf, and is skipped by every ray that grazes it.

#include "gaussian_ray.slangh"

struct GaussianAabbParams
{
    uint gaussianCount;
    float minAlpha;    // the alpha skip rays will use (RayCutoffs); 1/255 = exact
    float invMinAlpha; // 1 / minAlpha
    float pad0;
};

StructuredBuffer<float4> means;
StructuredBuffer<float4> scales;
StructuredBuffer<float4> rotations;
StructuredBuffer<float> opacities;

[vk::push_constant]
ConstantBuffer<GaussianAabbParams> params;

// VkAabbPositionsKHR is 6 tightly packed floats (24 B, a multiple of the
// 8-byte stride Vulkan requires); a float3 pair struct would be padded to
// 32 B under std430, so the boxes are written as a flat float array.
RWStructuredBuffer<float> aabbs;

[shader("compute")]
[numthreads(SABERCAT_WORKGROUP_SIZE, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    const uint i = threadId.x;
    if (i >= params.gaussianCount)
    {
        return;
    }

    const float3 mean = means[i].xyz;
    float3 lo = allFinite(mean) ? mean : float3(0.0f, 0.0f, 0.0f);
    float3 hi = lo;

    GaussianRayFrame frame;
    // A Gaussian fainter than the alpha skip can never contribute either.
    if (gaussianRayFrame(mean, scales[i].xyz, rotations[i], opacities[i], frame)
        && frame.opacity >= params.minAlpha)
    {
        // Finite by construction: gaussianRayFrame rejects boxes that overflow.
        const float3 extent = gaussianAabbHalfExtent(frame, params.invMinAlpha);
        lo = mean - extent;
        hi = mean + extent;
    }

    const uint base = i * 6u;
    aabbs[base + 0u] = lo.x;
    aabbs[base + 1u] = lo.y;
    aabbs[base + 2u] = lo.z;
    aabbs[base + 3u] = hi.x;
    aabbs[base + 4u] = hi.y;
    aabbs[base + 5u] = hi.z;
}
