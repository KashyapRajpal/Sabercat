// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Bake per-Gaussian visibility -- R3DG's baked visibility and
// RGCA's precomputed (mono) radiance transfer -- into the `visibility`
// relight stream: 16 SH coefficients (degree 3) of
//     V_i(w) = transmittance along direction w from where a ray out of
//              Gaussian i's mean leaves its own 3-sigma ellipsoid,
// traced through every OTHER Gaussian (gaussianTransmittanceSkipping: the
// Gaussian itself peaks exactly at the origin, t* = 0, and would otherwise
// count). Projection by Monte Carlo over a fixed spherical Fibonacci set of
// N near-uniform directions (equal solid angle 4pi/N each):
//     c_k = (4pi / N) * sum_j V_i(w_j) * Y_k(w_j)
// in the 3DGS-signed basis (relight_common.slangh's shBasis16), so a fully
// visible Gaussian bakes c_0 = sqrt(4pi) and relight_prepare's cosine
// convolution of it gives ambient visibility 1. Deterministic: same scene,
// same directions, same coefficients.
//
// One thread per Gaussian, a window [firstGaussian, firstGaussian + count)
// per dispatch: the host splits the cloud into submissions small enough that
// none can trip a driver timeout (TDR) on a million-Gaussian scene.
//
// Compiled with the ray-query flags (shaders/CMakeLists.txt).

#include "gaussian_ray.slangh"
#include "relight_common.slangh"

struct BakeConstants
{
    uint firstGaussian;
    uint count;          // Gaussians in this dispatch
    uint directionCount; // N
    float rayLength;     // world units: how far a visibility ray reaches
};

RaytracingAccelerationStructure tlas;
StructuredBuffer<float4> means;
StructuredBuffer<float4> scales;
StructuredBuffer<float4> rotations;
StructuredBuffer<float> opacities;
RWStructuredBuffer<float> visibility; // SABERCAT_VISIBILITY_COEFFS per Gaussian

[vk::push_constant]
ConstantBuffer<BakeConstants> bake;

// Direction j of N on the spherical Fibonacci lattice: z uniform in (-1, 1)
// at cell centres, azimuth advancing by the golden angle. sabercat's
// visibility_test mirrors this exactly for its CPU reference.
float3 fibonacciDirection(uint j, uint n)
{
    const float goldenAngle = 2.39996322972865332f; // pi * (3 - sqrt(5))
    float z = 1.0f - (2.0f * float(j) + 1.0f) / float(n);
    float r = sqrt(max(0.0f, 1.0f - z * z));
    float phi = goldenAngle * float(j);
    return float3(r * cos(phi), r * sin(phi), z);
}

[shader("compute")]
[numthreads(SABERCAT_WORKGROUP_SIZE, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= bake.count)
    {
        return;
    }
    uint i = bake.firstGaussian + id.x;
    float3 origin = means[i].xyz;
    uint base = i * SABERCAT_VISIBILITY_COEFFS;

    float coeffs[SABERCAT_VISIBILITY_COEFFS];
    for (uint k = 0; k < SABERCAT_VISIBILITY_COEFFS; ++k)
    {
        coeffs[k] = 0.0f;
    }
    GaussianRayFrame self;
    const bool hasFrame = gaussianRayFrame(origin, scales[i].xyz, rotations[i], opacities[i], self);
    if (all(isfinite(origin)))
    {
        for (uint j = 0; j < bake.directionCount; ++j)
        {
            float3 w = fibonacciDirection(j, bake.directionCount);
            // The ray starts where it leaves this Gaussian's own 3-sigma
            // ellipsoid (t = k / |S^-1 R^T w|), its support -- the bake's
            // analog of shadow_rt's normal offset. From the bare mean, a
            // ray out of a thin surfel crosses its overlapping coplanar
            // neighbours at t* ~ 0+ and, under the peak-response model, takes
            // their full peak alpha: a flat, fully open ground would bake as
            // almost fully occluded.
            float tStart = 0.0f;
            if (hasFrame)
            {
                tStart = SABERCAT_RADIUS_SIGMAS / length(mul(w, self.R) * self.invScale);
            }
            float v = gaussianTransmittanceSkipping(tlas, means, scales, rotations, opacities,
                                                    origin, w, min(tStart, bake.rayLength),
                                                    bake.rayLength, i);
            float b[16];
            shBasis16(w, b);
            for (uint k = 0; k < SABERCAT_VISIBILITY_COEFFS; ++k)
            {
                coeffs[k] += v * b[k];
            }
        }
    }
    float weight = 4.0f * kPi / float(max(bake.directionCount, 1u));
    for (uint k = 0; k < SABERCAT_VISIBILITY_COEFFS; ++k)
    {
        visibility[base + k] = coeffs[k] * weight;
    }
}
