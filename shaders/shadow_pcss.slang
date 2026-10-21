// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Percentage-closer soft shadows over a splat shadow map --
// the conventional baseline any baked or learned visibility must beat, and
// the fallback on devices without ray query.
//
// Per receiver pixel (from the G-buffer): asset-space position offset along
// the normal, projected into the light camera; (1) blocker search -- the mean
// median-depth of shadow-map texels closer than the receiver within a window
// scaled by the light's size; (2) penumbra width from similar triangles
// (local light: size * (dR - dB) / dB; directional: (dR - dB) * tan(angular
// radius)); (3) PCF over that width. Writes visibility layer `layer`.
// Every map comparison uses the receiver's plane in the light's view (slope
// from the G-buffer normal) plus a slope-scaled bias, so tilted receivers do
// not shadow themselves.
// A heuristic by construction: it is graded against the ray-traced
// reference, not taken as truth.

#include "config.slangh"

struct ShadowParams
{
    float4x4 receiverCameraToWorld;
    float4x4 lightWorldToCamera;
    float4 receiverIntrinsics; // fx, fy, cx, cy
    float4 lightIntrinsics;    // fx, fy, cx, cy
    uint width;
    uint height;
    uint mapWidth;
    uint mapHeight;
    uint layer;
    uint blockerSamplesPerAxis;
    uint filterSamplesPerAxis;
    uint directional;
    float lightSize;    // local: emitter radius (world); directional: tan(angular radius)
    float depthBias;    // world units
    float normalOffset; // world units
    float maxFilterTexels;
};

StructuredBuffer<float4> gNormalDepth;
StructuredBuffer<float4> gMetalAmbientCoverage;
StructuredBuffer<float2> shadowMap;
StructuredBuffer<ShadowParams> paramsBuffer;
RWStructuredBuffer<float> visibility;

// Slope bias, in shadow-map texels of depth error per unit tan(theta): covers
// the median-depth map's splat-centre error on tilted receivers.
static const float kSlopeBiasTexels = 8.0f;

float mapDepth(ShadowParams p, float2 uv)
{
    int2 t = int2(round(uv));
    if (t.x < 0 || t.y < 0 || t.x >= int(p.mapWidth) || t.y >= int(p.mapHeight))
    {
        return 1e30f; // outside the light's view: unshadowed
    }
    return shadowMap[uint(t.y) * p.mapWidth + uint(t.x)].x;
}

[shader("compute")]
[numthreads(SABERCAT_TILE_SIZE, SABERCAT_TILE_SIZE, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    ShadowParams p = paramsBuffer[0];
    if (id.x >= p.width || id.y >= p.height)
    {
        return;
    }
    uint pixelIndex = id.y * p.width + id.x;
    float4 nd = gNormalDepth[pixelIndex];
    float coverage = gMetalAmbientCoverage[pixelIndex].z;
    float result = 1.0f;

    if (coverage > 1e-6f)
    {
        float depth = nd.w / coverage;
        float nLen = length(nd.xyz);
        float3 N = nLen > 1e-8f ? nd.xyz / nLen : float3(0.0f, 0.0f, 0.0f);
        float3 local = float3((float(id.x) - p.receiverIntrinsics.z) * depth / p.receiverIntrinsics.x,
                              (p.receiverIntrinsics.w - float(id.y)) * depth / p.receiverIntrinsics.y,
                              -depth);
        float3 P = mul(p.receiverCameraToWorld, float4(local, 1.0f)).xyz + N * p.normalOffset;

        float3 lp = mul(p.lightWorldToCamera, float4(P, 1.0f)).xyz;
        float dR = -lp.z;
        if (dR > 1e-6f)
        {
            float f = p.lightIntrinsics.x;
            float2 uv = float2(p.lightIntrinsics.x * lp.x / dR + p.lightIntrinsics.z,
                               p.lightIntrinsics.w - p.lightIntrinsics.y * lp.y / dR);
            // The receiver's plane as the light sees it: depth changes by slope.x
            // per texel in u and slope.y per texel in v. A tap at texel t is
            // compared against the plane's depth there, so a tilted receiver
            // does not find its own surface as a blocker in the search or the
            // filter. Grazing planes are clamped so the slope stays finite.
            float3 Nl = mul(p.lightWorldToCamera, float4(N, 0.0f)).xyz;
            float facing = dot(Nl, float3(lp.x / dR, lp.y / dR, -1.0f));
            facing = facing < 0.0f ? min(facing, -0.05f) : max(facing, 0.05f);
            float2 slope = float2(-Nl.x * dR / (facing * f), Nl.y * dR / (facing * p.lightIntrinsics.y));

            // Depth error of the map itself: a splat's median depth is its centre's
            // depth, which lies up to a splat radius off the receiver's plane
            // laterally, and that offset maps to depth by tan(theta) between the
            // view ray and the surface. Scale the bias by it, in texels.
            float cosTheta = abs(dot(Nl, normalize(lp)));
            float tanTheta = sqrt(max(1.0f - cosTheta * cosTheta, 0.0f)) / max(cosTheta, 0.05f);
            float totalBias = p.depthBias + kSlopeBiasTexels * (dR / f) * tanTheta;

            // (1) Blocker search over a window the light could be seen through.
            float searchTexels = p.directional != 0u ? p.lightSize * f
                                                     : p.lightSize * f / dR;
            searchTexels = clamp(searchTexels, 1.0f, p.maxFilterTexels);
            uint nb = max(p.blockerSamplesPerAxis, 1u);
            float blockerSum = 0.0f;
            float receiverSum = 0.0f;
            uint blockers = 0;
            for (uint by = 0; by < nb; ++by)
            {
                for (uint bx = 0; bx < nb; ++bx)
                {
                    float2 o = (float2(bx, by) + 0.5f) / float(nb) * 2.0f - 1.0f;
                    float2 t = round(uv + o * searchTexels);
                    float d = mapDepth(p, t);
                    float planeDepth = dR + dot(slope, t - uv);
                    if (d < planeDepth - totalBias)
                    {
                        blockerSum += d;
                        receiverSum += planeDepth;
                        ++blockers;
                    }
                }
            }

            if (blockers > 0)
            {
                // (2) Penumbra width at the receiver, in shadow-map texels. Each blocker
                // is measured against the receiver's plane at its own texel, so a tilted
                // receiver and an occluder parallel to it give the true gap.
                float dB = blockerSum / float(blockers);
                float dRecv = receiverSum / float(blockers);
                float worldWidth = p.directional != 0u ? (dRecv - dB) * p.lightSize
                                                       : p.lightSize * (dRecv - dB) / max(dB, 1e-6f);
                float filterTexels = clamp(worldWidth * f / dR, 0.5f, p.maxFilterTexels);

                // (3) PCF.
                uint nf = max(p.filterSamplesPerAxis, 1u);
                uint lit = 0;
                for (uint fy = 0; fy < nf; ++fy)
                {
                    for (uint fx = 0; fx < nf; ++fx)
                    {
                        float2 o = (float2(fx, fy) + 0.5f) / float(nf) * 2.0f - 1.0f;
                        float2 t = round(uv + o * filterTexels);
                        lit += mapDepth(p, t) >= dR + dot(slope, t - uv) - totalBias ? 1u : 0u;
                    }
                }
                result = float(lit) / float(nf * nf);
            }
        }
    }
    visibility[p.layer * p.width * p.height + pixelIndex] = result;
}
