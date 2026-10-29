// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Presentation-only debug view: reads a channel ForwardPass already
// produced (or, for tile density, derives cheaply from tileRanges) and
// writes a normalized RGBA image for present() to blit to the swapchain.
// Not part of the differentiable rasterizer -- nothing here is consumed by
// any forward-pass stage, and this kernel never runs for the default
// Final view (present() blits ForwardPass's own output image directly).

#include "forward_common.slangh"

RWTexture2D<float4> colorImage;
RWTexture2D<float> depthImage;
StructuredBuffer<uint2> tileRanges;
RWTexture2D<float4> displayImage;
// SHDegree view: re-walks each pixel's blend over the same sorted
// instance list tile_rasterize used, reading every Gaussian's effective
// (possibly fractional) SH degree from forward_project's shDegrees.
StructuredBuffer<ProjectedGaussian> projected;
StructuredBuffer<uint> sortedValues;
StructuredBuffer<float> shDegrees;

struct DebugParams
{
    uint mode; // 1 = alpha, 2 = depth, 3 = tile density, 4 = SH degree (0/Final never dispatches this kernel)
    uint imageWidth;
    uint imageHeight;
    uint tilesX;
    float depthScale;   // depth / depthScale -> [0,1] grayscale
    float densityScale; // instance count / densityScale -> [0,1] heatmap
};

[vk::push_constant]
ConstantBuffer<DebugParams> params;

// SH degree 0..3 -> blue, green, yellow, red, linear between stops so a
// faded (fractional) degree reads as an in-between color. Red = full
// quality (the expensive end), matching TileDensity's red = hot.
static const float3 kSHDegreeStops[4] = {
    float3(0.15f, 0.25f, 0.95f),
    float3(0.15f, 0.85f, 0.25f),
    float3(0.95f, 0.85f, 0.15f),
    float3(0.95f, 0.20f, 0.15f),
};

float3 shDegreeHeat(float degree)
{
    float d = clamp(degree, 0.0f, (float)SABERCAT_SH_DEGREE);
    uint low = min((uint)floor(d), (uint)SABERCAT_SH_DEGREE - 1u);
    float t = d - (float)low;
    return kSHDegreeStops[low] + t * (kSHDegreeStops[low + 1] - kSHDegreeStops[low]);
}

[shader("compute")]
[numthreads(SABERCAT_TILE_SIZE, SABERCAT_TILE_SIZE, 1)]
void main(uint3 pixel : SV_DispatchThreadID)
{
    if (pixel.x >= params.imageWidth || pixel.y >= params.imageHeight)
    {
        return;
    }

    float4 result = float4(0.0f, 0.0f, 0.0f, 1.0f);
    if (params.mode == 1)
    {
        float a = colorImage[pixel.xy].a;
        result = float4(a, a, a, 1.0f);
    }
    else if (params.mode == 2)
    {
        float d = clamp(depthImage[pixel.xy] / max(params.depthScale, 1e-4f), 0.0f, 1.0f);
        result = float4(d, d, d, 1.0f);
    }
    else if (params.mode == 3)
    {
        uint tileX = pixel.x / SABERCAT_TILE_SIZE;
        uint tileY = pixel.y / SABERCAT_TILE_SIZE;
        uint tileId = tileY * params.tilesX + tileX;
        uint2 range = tileRanges[tileId];
        float density = clamp(float(range.y - range.x) / max(params.densityScale, 1.0f), 0.0f, 1.0f);
        // Blue (cold/empty) -> red (hot/dense).
        result = float4(density, 0.2f * (1.0f - density), 1.0f - density, 1.0f);
    }
    else if (params.mode == 4)
    {
        // Effective per-pixel SH degree = the blend-weighted mean of the
        // contributing Gaussians' degrees, weights T_i * alpha_i -- the
        // exact weights that composite the pixel's color, so this is the
        // degree of the color actually shown. (The front-most contributor
        // alone would be dominated by faint alpha >= 1/255 splats.) Same
        // evaluateSplat, order and 1e-4 early-out as tile_rasterize, so the
        // walk visits exactly the Gaussians that blended. Tinted by coverage
        // (1 - T) so empty background stays black.
        uint tileX = pixel.x / SABERCAT_TILE_SIZE;
        uint tileY = pixel.y / SABERCAT_TILE_SIZE;
        uint2 range = tileRanges[tileY * params.tilesX + tileX];
        float2 pixelPosition = float2(pixel.xy);

        float transmittance = 1.0f;
        float weightSum = 0.0f;
        float weightedDegree = 0.0f;
        for (uint i = range.x; i < range.y; ++i)
        {
            uint gaussianIndex = sortedValues[i];
            ProjectedGaussian g = projected[gaussianIndex];
            float alpha = 0.0f;
            float gaussian = 0.0f;
            bool clamped = false;
            if (!evaluateSplat(pixelPosition, g.center, g.conic, g.alpha, alpha, gaussian, clamped))
            {
                continue;
            }
            float weight = transmittance * alpha;
            weightSum += weight;
            weightedDegree += weight * shDegrees[gaussianIndex];
            transmittance *= (1.0f - alpha);
            if (transmittance < 1e-4f)
            {
                break;
            }
        }

        float degree = weightSum > 0.0f ? weightedDegree / weightSum : 0.0f;
        result = float4(shDegreeHeat(degree) * (1.0f - transmittance), 1.0f);
    }

    displayImage[pixel.xy] = result;
}
