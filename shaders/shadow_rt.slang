// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Ray-traced soft shadows through the splats -- the ground truth
// the PCSS baseline and any baked or learned visibility are graded against.
//
// One thread per receiver, one light per dispatch. The receiver comes from the
// G-buffer (position from expected depth, offset along the normal); its
// visibility is the mean transmittance over a fixed n x n stratified emitter
// grid (light_visibility.slangh), traced THROUGH the Gaussians
// (order-independent, no sorting). The teacher samples use the same
// rays, so their targets equal this reference bit for bit.
//
// Progressive accumulation: a dispatch traces grid samples
// [firstSample, firstSample + sampleCount) and adds them to this layer's
// running sum (accum; firstSample 0 restarts it), then writes
// visibility = sum / (firstSample + sampleCount). The samples are summed in
// the same order as lightVisibility's loop, so once firstSample +
// sampleCount = n * n the result is the full-grid value bit for bit. A
// dispatch with sampleCount 0 only re-resolves a converged layer.
//
// Approximations (ShadowSettings; the defaults are exact):
//   minAlpha / minTransmittance  RayCutoffs -- skip fainter Gaussians, stop
//                                darker rays (the BVH is built to match)
//   downsample s > 1             trace one receiver per s x s block -- pixel
//                                (s*lx, s*ly) -- into lowRes (lw x lh per
//                                layer); shadow_upsample.slang fills the rest
//
// Compiled with the ray-query flags (shaders/CMakeLists.txt).

#include "light_visibility.slangh"

struct RayShadowParams
{
    float4x4 receiverCameraToWorld;
    float4 receiverIntrinsics; // fx, fy, cx, cy
    float4 lightPositionType;  // xyz position; w = type (0 dir, 1 point, 2 rect)
    float4 lightDirection;     // xyz: direction the light travels (Directional)
    float4 lightAxisX;         // xyz: axisX * halfWidth (Rect)
    float4 lightAxisY;         // xyz: axisY * halfHeight (Rect)
    uint width;
    uint height;
    uint layer;
    uint samplesPerAxis;
    float lightSize;    // Point: disk radius; Directional: tan(angular radius)
    float normalOffset; // world units
    float rayLength;    // world units: how far a directional ray reaches
    uint firstSample;   // progressive: first grid sample this dispatch traces
    uint sampleCount;   // progressive: how many (n * n = everything at once)
    float minAlpha;     // RayCutoffs
    float minTransmittance;
    uint downsample;    // 1 = every pixel; s = one receiver per s x s block
};

RaytracingAccelerationStructure tlas;
StructuredBuffer<float4> means;
StructuredBuffer<float4> scales;
StructuredBuffer<float4> rotations;
StructuredBuffer<float> opacities;
StructuredBuffer<float4> gNormalDepth;
StructuredBuffer<float4> gMetalAmbientCoverage;
StructuredBuffer<RayShadowParams> paramsBuffer;
RWStructuredBuffer<float> visibility; // full resolution (downsample 1)
RWStructuredBuffer<float> accum;      // per layer and traced receiver: the running sum
RWStructuredBuffer<float> lowRes;     // downsample > 1: per layer, lw x lh results

[shader("compute")]
[numthreads(SABERCAT_TILE_SIZE, SABERCAT_TILE_SIZE, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    RayShadowParams p = paramsBuffer[0];
    uint s = max(p.downsample, 1u);
    uint lw = (p.width + s - 1u) / s;
    uint lh = (p.height + s - 1u) / s;
    if (id.x >= lw || id.y >= lh)
    {
        return;
    }
    uint x = id.x * s;
    uint y = id.y * s;
    uint pixelIndex = y * p.width + x;
    uint index = s == 1u ? p.layer * p.width * p.height + pixelIndex : p.layer * lw * lh + id.y * lw + id.x;
    float result = 1.0f;

    float3 position, N;
    if (gbufferReceiver(gNormalDepth[pixelIndex], gMetalAmbientCoverage[pixelIndex].z, x, y,
                        p.receiverCameraToWorld, p.receiverIntrinsics, position, N))
    {
        TraceLight light;
        light.positionType = p.lightPositionType;
        light.direction = p.lightDirection;
        light.axisX = p.lightAxisX;
        light.axisY = p.lightAxisY;
        light.sizeLength = float4(p.lightSize, p.rayLength, 0.0f, 0.0f);
        RayCutoffs cutoffs = exactRayCutoffs();
        cutoffs.minAlpha = p.minAlpha;
        cutoffs.minTransmittance = p.minTransmittance;
        uint n = max(p.samplesPerAxis, 1u);
        float3 origin = position + N * p.normalOffset;
        float sum = p.firstSample == 0u ? 0.0f : accum[index];
        for (uint k = p.firstSample; k < p.firstSample + p.sampleCount; ++k)
        {
            sum += emitterSampleTransmittance(tlas, means, scales, rotations, opacities, light, origin, k, n,
                                              cutoffs);
        }
        accum[index] = sum;
        result = sum / float(p.firstSample + p.sampleCount);
    }
    if (s == 1u)
    {
        visibility[index] = result;
    }
    else
    {
        lowRes[index] = result;
    }
}
