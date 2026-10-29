// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Ray-traced teacher samples for an external trainer to learn visibility
// (and, with the G-buffer, materials) from. Draws from sample_select's two
// candidate lists:
//   randomTarget samples from the smooth list, edgeTarget from the edge list
//   (defaults 75% / 25%; both runtime-tunable), each drawn STRATIFIED: the
//   list is cut into K equal strata (K = min(target, list size)) and one
//   jittered pick taken per stratum, so samples spread over the whole image
//   and never repeat. The jitter is a hash of (sample, seed): vary the seed
//   per call for fresh samples, keep it for reproducibility.
// Each sample is assigned one shadow-casting light round-robin and traced
// with lightVisibility -- the SAME function and emitter grid as shadow_rt's
// ray-traced shadows, so visibilityTarget equals the reference shadow value
// at that pixel bit for bit (at the same samplesPerAxis).
//
// Output, compacted: [0, kS) the smooth samples, [kS, kS + kE) the edge
// samples; counts[1] = (kS + kE, kS). samplePixels mirrors the
// samples' pixel indices (top bit set for edge samples) for the debug view.
//
// Compiled with the ray-query flags (shaders/CMakeLists.txt).

#include "light_visibility.slangh"

// One sample, std430 32 B -- sabercat::VisibilityTrainingSample.
struct VisibilityTrainingSample
{
    float3 position; // asset space, on the surface (NOT offset along the normal)
    float visibilityTarget;
    float3 normal; // unit, or zero where the splatted normals canceled
    uint lightIndex;
};

struct TeacherParams
{
    float4x4 receiverCameraToWorld;
    float4 receiverIntrinsics; // fx, fy, cx, cy
    uint randomTarget;
    uint edgeTarget;
    uint samplesPerAxis; // emitter grid per axis (4 => 16 emitter samples)
    uint seed;
    uint shadowLightCount;
    uint width;
    uint height;
    uint pad0;
    float normalOffset; // world units
    float pad1;
    float pad2;
    float pad3;
    uint4 shadowLights[2]; // indices of the shadow-casting lights, in order
};

RaytracingAccelerationStructure tlas;
StructuredBuffer<float4> means;
StructuredBuffer<float4> scales;
StructuredBuffer<float4> rotations;
StructuredBuffer<float> opacities;
StructuredBuffer<float4> gNormalDepth;
StructuredBuffer<float4> gMetalAmbientCoverage;
StructuredBuffer<uint> smoothList;
StructuredBuffer<uint> edgeList;
RWStructuredBuffer<uint4> counts;
StructuredBuffer<TraceLight> lights;
StructuredBuffer<TeacherParams> paramsBuffer;
RWStructuredBuffer<VisibilityTrainingSample> samples;
RWStructuredBuffer<uint> samplePixels;

uint pcgHash(uint v)
{
    uint state = v * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Pick k of K equal strata over a list of `size` entries (K <= size).
uint stratifiedPick(uint k, uint K, uint size, float jitter)
{
    float ratio = float(size) / float(K);
    uint start = uint(floor(float(k) * ratio));
    uint end = k + 1u == K ? size : uint(floor(float(k + 1u) * ratio));
    end = max(end, start + 1u);
    return min(start + min(uint(jitter * float(end - start)), end - start - 1u), size - 1u);
}

[shader("compute")]
[numthreads(SABERCAT_WORKGROUP_SIZE, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    TeacherParams p = paramsBuffer[0];
    uint4 c = counts[0];
    uint kS = min(p.randomTarget, c.x);
    uint kE = min(p.edgeTarget, c.y);
    if (id.x == 0u)
    {
        counts[1] = uint4(kS + kE, kS, 0u, 0u);
    }

    uint i = id.x;
    uint k, outIndex, pixel;
    bool edge;
    float jitter = float(pcgHash(i ^ pcgHash(p.seed)) >> 8) * (1.0f / 16777216.0f);
    if (i < p.randomTarget)
    {
        k = i;
        if (k >= kS)
        {
            return;
        }
        pixel = smoothList[stratifiedPick(k, kS, c.x, jitter)];
        outIndex = k;
        edge = false;
    }
    else
    {
        k = i - p.randomTarget;
        if (k >= kE)
        {
            return;
        }
        pixel = edgeList[stratifiedPick(k, kE, c.y, jitter)];
        outIndex = kS + k;
        edge = true;
    }

    uint slot = outIndex % max(p.shadowLightCount, 1u);
    uint lightIndex = slot < 4u ? p.shadowLights[0][slot] : p.shadowLights[1][slot - 4u];

    uint x = pixel % p.width;
    uint y = pixel / p.width;
    float3 position, N;
    gbufferReceiver(gNormalDepth[pixel], gMetalAmbientCoverage[pixel].z, x, y,
                    p.receiverCameraToWorld, p.receiverIntrinsics, position, N);

    VisibilityTrainingSample s;
    s.position = position;
    s.visibilityTarget = lightVisibility(tlas, means, scales, rotations, opacities,
                                         lights[lightIndex], position + N * p.normalOffset,
                                         p.samplesPerAxis, exactRayCutoffs());
    s.normal = N;
    s.lightIndex = lightIndex;
    samples[outIndex] = s;
    // The tag is a shift, not `edge ? 0x80000000u : 0u`: Slang 2025.6 (via GLSL)
    // reused outIndex's variable for that ternary and wrote to the wrong slot.
    samplePixels[outIndex] = pixel | (uint(edge) << 31u);
}
