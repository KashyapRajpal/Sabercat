// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Test kernel for examples/gaussian_raytrace_test: one thread per ray,
// gaussianTransmittance against a GaussianAccelerationStructure's TLAS. The
// per-pixel shadow kernel calls the same function. The cutoffs come from the
// host so one kernel traces both the exact model and a structure built with a
// raised minAlpha.

#include "gaussian_ray.slangh"

struct TraceParams
{
    uint rayCount;
    float minAlpha;         // RayCutoffs::minAlpha
    float invMinAlpha;      // RayCutoffs::invMinAlpha
    float minTransmittance; // RayCutoffs::minTransmittance
};

// One ray: (origin, tMin), (direction, tMax).
struct TestRay
{
    float4 originTMin;
    float4 directionTMax;
};

RaytracingAccelerationStructure tlas;
StructuredBuffer<float4> means;
StructuredBuffer<float4> scales;
StructuredBuffer<float4> rotations;
StructuredBuffer<float> opacities;
StructuredBuffer<TestRay> rays;
RWStructuredBuffer<float> transmittance;

[vk::push_constant]
ConstantBuffer<TraceParams> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    const uint i = threadId.x;
    if (i >= params.rayCount)
    {
        return;
    }
    RayCutoffs cutoffs;
    cutoffs.minAlpha = params.minAlpha;
    cutoffs.invMinAlpha = params.invMinAlpha;
    cutoffs.minTransmittance = params.minTransmittance;
    cutoffs.pad0 = 0.0f;
    const TestRay ray = rays[i];
    transmittance[i] = gaussianTransmittance(tlas, means, scales, rotations, opacities,
                                             ray.originTMin.xyz, ray.directionTMax.xyz,
                                             ray.originTMin.w, ray.directionTMax.w, cutoffs);
}
