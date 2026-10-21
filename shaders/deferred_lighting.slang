// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Deferred lighting over the splatted G-buffer.
//
// Per pixel: normalize the premultiplied G-buffer sums by coverage,
// reconstruct the asset-space position from the expected depth and the
// forward camera, and shade every light with the metallic-roughness BRDF
// (relight_common.slangh). Direct lights are multiplied by this pixel's
// layer of the VISIBILITY buffer, which this kernel never produces: it is
// all ones by default, and ray-traced / PCSS shadows or an external
// (learned) source fills it. The environment is degree-2 radiance SH: exact
// cosine-convolved irradiance for diffuse, the radiance in the reflection
// direction for a rough specular approximation, both scaled by the splatted
// per-Gaussian ambient visibility. Result = lit * coverage + background *
// (1 - coverage), times exposure; alpha = coverage.

#include "relight_common.slangh"

StructuredBuffer<float4> gNormalDepth;
StructuredBuffer<float4> gBaseRoughness;
StructuredBuffer<float4> gMetalAmbientCoverage;
StructuredBuffer<GpuLight> lights;
StructuredBuffer<LightingParams> paramsBuffer;
StructuredBuffer<float> visibility; // [light][pixel], SABERCAT_MAX_LIGHTS layers
RWStructuredBuffer<float4> litBuffer;
RWTexture2D<float4> litImage;

static const uint kOutLit = 0u;
static const uint kOutNormal = 1u;
static const uint kOutBaseColor = 2u;
static const uint kOutRoughness = 3u;
static const uint kOutMetallic = 4u;
static const uint kOutAmbient = 5u;
static const uint kOutVisibility = 6u;
static const uint kOutDepth = 7u;

float3 environmentSH(LightingParams p, float3 dir, bool convolve)
{
    float b[16];
    shBasis16(dir, b);
    float3 sum = float3(0.0f, 0.0f, 0.0f);
    for (uint k = 0; k < 9; ++k)
    {
        uint band = k == 0 ? 0u : (k < 4 ? 1u : 2u);
        sum += (convolve ? kCosineBand[band] : 1.0f) * p.environment[k].xyz * b[k];
    }
    return max(sum, 0.0f);
}

[shader("compute")]
[numthreads(SABERCAT_TILE_SIZE, SABERCAT_TILE_SIZE, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    LightingParams p = paramsBuffer[0];
    if (id.x >= p.width || id.y >= p.height)
    {
        return;
    }
    uint pixelIndex = id.y * p.width + id.x;
    uint pixelCount = p.width * p.height;

    float4 nd = gNormalDepth[pixelIndex];
    float4 br = gBaseRoughness[pixelIndex];
    float4 mac = gMetalAmbientCoverage[pixelIndex];
    float coverage = mac.z;
    float3 background = p.backgroundExposure.xyz;
    float exposure = p.backgroundExposure.w;

    float3 color = float3(0.0f, 0.0f, 0.0f);
    if (coverage > 1e-6f)
    {
        float invCoverage = 1.0f / coverage;
        float nLen = length(nd.xyz);
        float3 N = nLen > 1e-8f ? nd.xyz / nLen : float3(0.0f, 0.0f, 1.0f);
        float depth = nd.w * invCoverage;
        float3 baseColor = saturate(br.xyz * invCoverage);
        float roughness = saturate(br.w * invCoverage);
        float metallic = saturate(mac.x * invCoverage);
        float ambient = saturate(mac.y * invCoverage);

        // Pixel centres at integer coordinates; v = cy - fy*y/d, d = -z.
        float3 local = float3((float(id.x) - p.intrinsics.z) * depth / p.intrinsics.x,
                              (p.intrinsics.w - float(id.y)) * depth / p.intrinsics.y, -depth);
        float3 P = mul(p.cameraToWorld, float4(local, 1.0f)).xyz;
        float3 V = normalize(p.cameraPosition.xyz - P);

        if (p.output == kOutLit)
        {
            float3 lit = float3(0.0f, 0.0f, 0.0f);
            for (uint li = 0; li < p.lightCount; ++li)
            {
                GpuLight light = lights[li];
                uint type = uint(light.positionType.w + 0.5f);
                float shadow = light.axisXShadow.w > 0.5f
                                   ? visibility[li * pixelCount + pixelIndex]
                                   : 1.0f;
                if (shadow <= 0.0f)
                {
                    continue;
                }
                float3 contribution = float3(0.0f, 0.0f, 0.0f);
                if (type == 0u)
                {
                    float3 L = -normalize(light.direction.xyz);
                    contribution = shadeBRDF(N, V, L, baseColor, roughness, metallic)
                                   * light.radiance.xyz;
                }
                else if (type == 1u)
                {
                    float3 toLight = light.positionType.xyz - P;
                    float d2 = max(dot(toLight, toLight), 1e-8f);
                    float3 L = toLight * rsqrt(d2);
                    contribution = shadeBRDF(N, V, L, baseColor, roughness, metallic)
                                   * light.radiance.xyz / d2;
                }
                else
                {
                    // Stratified midpoints over the rectangle; one-sided
                    // emitter of radiance Le facing cross(axisX, axisY).
                    float3 ax = light.axisXShadow.xyz;
                    float3 ay = light.axisY.xyz;
                    float3 emitNormal = normalize(cross(ax, ay));
                    float area = 4.0f * length(ax) * length(ay);
                    uint n = max(p.rectSamplesPerAxis, 1u);
                    float sampleArea = area / float(n * n);
                    for (uint sy = 0; sy < n; ++sy)
                    {
                        for (uint sx = 0; sx < n; ++sx)
                        {
                            float u = (float(sx) + 0.5f) / float(n) * 2.0f - 1.0f;
                            float v = (float(sy) + 0.5f) / float(n) * 2.0f - 1.0f;
                            float3 S = light.positionType.xyz + u * ax + v * ay;
                            float3 toLight = S - P;
                            float d2 = max(dot(toLight, toLight), 1e-8f);
                            float3 L = toLight * rsqrt(d2);
                            float cosLight = dot(emitNormal, -L);
                            if (cosLight <= 0.0f)
                            {
                                continue;
                            }
                            contribution += shadeBRDF(N, V, L, baseColor, roughness, metallic)
                                            * cosLight * sampleArea / d2;
                        }
                    }
                    contribution *= light.radiance.xyz;
                }
                lit += shadow * contribution;
            }

            if (p.hasEnvironment != 0u)
            {
                // Diffuse: E(n) = pi * sum_l w_l L_lm Y_lm(n); BRDF base/pi.
                float3 irradiance = kPi * environmentSH(p, N, true);
                float NdotV = max(dot(N, V), 1e-4f);
                float3 f0 = lerp(float3(0.04f, 0.04f, 0.04f), baseColor, metallic);
                float3 F = fresnelSchlick(NdotV, f0);
                float3 diffuse = (1.0f - F) * (1.0f - metallic) * baseColor / kPi * irradiance;
                // Rough specular approximation: degree-2 SH is already a
                // heavily blurred environment; weight it down as roughness
                // rises (no split-sum LUT at this degree).
                float3 R = reflect(-V, N);
                float3 specular = F * environmentSH(p, R, false) * (1.0f - 0.5f * roughness);
                lit += ambient * (diffuse + specular);
            }
            color = lit;
        }
        else if (p.output == kOutNormal)
        {
            color = N * 0.5f + 0.5f;
        }
        else if (p.output == kOutBaseColor)
        {
            color = baseColor;
        }
        else if (p.output == kOutRoughness)
        {
            color = float3(roughness, roughness, roughness);
        }
        else if (p.output == kOutMetallic)
        {
            color = float3(metallic, metallic, metallic);
        }
        else if (p.output == kOutAmbient)
        {
            color = float3(ambient, ambient, ambient);
        }
        else if (p.output == kOutVisibility)
        {
            float v = visibility[min(p.visibilityLight, SABERCAT_MAX_LIGHTS - 1u) * pixelCount
                                 + pixelIndex];
            color = float3(v, v, v);
        }
        else
        {
            float d = depth * p.depthScale.x;
            color = float3(d, d, d);
        }
    }

    float3 composed = (color * coverage + background * (1.0f - coverage)) * exposure;
    float4 result = float4(composed, coverage);
    litBuffer[pixelIndex] = result;
    litImage[int2(id.xy)] = result;
}
