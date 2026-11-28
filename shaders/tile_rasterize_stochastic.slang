// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Sorting study: the stochastic-transparency rasterizer (Stochastic mode). It
// draws one raw frame of an order-independent estimate of the blend: every
// pixel keeps SABERCAT_ST_SAMPLES samples, and each sample keeps the nearest
// candidate that covers it, where "covers" is a per-sample threshold test on
// the candidate's alpha. Because a sample only keeps a minimum t*, the order of
// the candidates does not matter, so the kernel needs no transmittance
// early-out and no group early-out. Averaged over frames (the resolve stage),
// the samples converge to the front-to-back blend in t* order.
//
// The thresholds combine a blue-noise rank per pixel and sample with a
// hash of the Gaussian index and a per-(frame, sample) seed, plus a per-frame
// offset from the host, so each frame
// draws different samples while the rank table keeps the samples of one frame
// evenly spread over the pixel.
//
// One workgroup per 16x16 tile, one thread per pixel, staged like
// tile_rasterize_window.slang: each batch of the tile's slice is copied into
// shared memory once, and every thread runs its own pixel against it. Every
// thread takes part in every barrier, including pixels outside the image,
// which skip the work and do not write.
//
// SABERCAT_ST_SAMPLES is the sample count S: one compiled variant per value
// (tile_rasterize_stochastic_s4 / _s8).

#include "sort_common.slangh"

#ifndef SABERCAT_ST_SAMPLES
#error "tile_rasterize_stochastic.slang needs -DSABERCAT_ST_SAMPLES=<S>; build it through the sabercat_add_shader_variant calls in shaders/CMakeLists.txt"
#endif

StructuredBuffer<ProjectedGaussian> projected;
StructuredBuffer<uint> sortedValues;   // Gaussian index, tile-major, default order
StructuredBuffer<uint2> tileRanges;    // per tile: (firstInstance, lastInstance)
// SABERCAT_RAY_FORM_FLOATS floats per Gaussian, indexed index * 12 + k.
StructuredBuffer<float> rayForm;
// The 64x64 blue-noise rank table: a permutation of 0..4095, index y * 64 + x.
StructuredBuffer<uint> ranks;
// Element 0: x = the per-frame noise offset (stochasticFrameOffset on the host).
StructuredBuffer<uint4> params;

[vk::push_constant]
ConstantBuffer<CameraUniforms> camera;

RWTexture2D<float4> outputImage;
RWTexture2D<float> outputDepth;

// One batch of the tile's candidates, staged cooperatively by the workgroup.
// The same staging as tile_rasterize_window.slang: the contributor test and the
// ray depth read the shared copy, and the color is read from projected[] only
// when a sample takes it.
groupshared uint batchIndex[SABERCAT_TILE_THREADS];
groupshared float2 batchCenter[SABERCAT_TILE_THREADS];
groupshared float3 batchConic[SABERCAT_TILE_THREADS];
groupshared float batchAlpha[SABERCAT_TILE_THREADS];
groupshared float batchDepth[SABERCAT_TILE_THREADS];
// The 12 ray-form floats, as three float4 per Gaussian:
//   batchForm0 = f0..f3, batchForm1 = f4..f7, batchForm2 = f8..f11
// (see rayDepthFromStaged for the unpacking).
groupshared float4 batchForm0[SABERCAT_TILE_THREADS];
groupshared float4 batchForm1[SABERCAT_TILE_THREADS];
groupshared float4 batchForm2[SABERCAT_TILE_THREADS];

// Peak depth along the pixel ray for the staged Gaussian i. This repeats the
// arithmetic of rayDepthAlongRay (sort_common.slangh) operation for operation,
// because that function reads an SSBO and this one reads the shared copy; the
// sample depths must agree with the window and sort-key paths on t*.
float rayDepthFromStaged(uint i, float2 dxdy)
{
    float3 c0 = batchForm0[i].xyz;
    float3 c1 = float3(batchForm0[i].w, batchForm1[i].x, batchForm1[i].y);
    float3 c2 = float3(batchForm1[i].z, batchForm1[i].w, batchForm2[i].x);
    float3 u = batchForm2[i].yzw;

    float3 dp = c0 * dxdy.x + c1 * dxdy.y - c2;
    float dd = dot(dp, dp);
    // Degenerate form (all zeros, or non-finite): the Gaussian's center depth,
    // exactly as rayDepthAlongRay falls back.
    if (!(dd > 0.0f) || !isfinite(dd))
    {
        return batchDepth[i];
    }
    float t = dot(u, dp) / dd;
    return isfinite(t) ? t : batchDepth[i];
}

[shader("compute")]
[numthreads(SABERCAT_TILE_SIZE, SABERCAT_TILE_SIZE, 1)]
void main(uint3 groupId : SV_GroupID, uint3 localId : SV_GroupThreadID)
{
    uint tileId = groupId.y * camera.tilesX + groupId.x;
    uint2 pixel = uint2(groupId.x * SABERCAT_TILE_SIZE + localId.x,
                        groupId.y * SABERCAT_TILE_SIZE + localId.y);
    bool inside = pixel.x < camera.imageWidth && pixel.y < camera.imageHeight;

    // Pixel centers at integer coordinates, as in tile_rasterize.slang.
    float2 pixelPosition = float2(pixel);
    // Camera-space ray through this pixel, (dx, dy) of d = (dx, dy, -1), the
    // same parameterization as the sort keys and the window model.
    float2 dxdy = float2((pixelPosition.x - camera.cx) / camera.fx,
                         -(pixelPosition.y - camera.cy) / camera.fy);

    // +infinity, written as a bit pattern so it is exact on every backend.
    float infinity = asfloat(0x7f800000u);

    // Per-sample state. thresholdBase[s] = u_s + frameOffset depends only on the
    // pixel and the sample, so it is formed once here, in the same order the
    // per-candidate threshold adds it (the two adds are not reassociated).
    // params[0] = {asuint(frameOffset), frame index, 0, 0}. The per-(frame, sample)
    // seed makes the hash rotation of u differ for every frame and sample: with a
    // rotation fixed per Gaussian, two candidates at one pixel would keep the same
    // relative offset for the whole run, their coverage would stay perfectly
    // dependent and the accumulated image would converge to a biased value
    // (measured: MAE stalled at 0.022 over 256 frames).
    float frameOffset = asfloat(params[0].x);
    uint frameIndex = params[0].y;
    uint seedS[SABERCAT_ST_SAMPLES];
    float thresholdBase[SABERCAT_ST_SAMPLES];
    float depthS[SABERCAT_ST_SAMPLES];
    float3 colorS[SABERCAT_ST_SAMPLES];
    for (uint s = 0; s < SABERCAT_ST_SAMPLES; ++s)
    {
        thresholdBase[s] = 0.0f;
        seedS[s] = stochasticHash(frameIndex * 8u + s);
        depthS[s] = infinity;
        colorS[s] = float3(0.0f, 0.0f, 0.0f);
        if (inside)
        {
            // The rank table is 64x64 and tiled by wrapping the pixel index;
            // the per-sample offsets (23, 41) decorrelate the samples' lookups.
            uint rank = ranks[((pixel.y + 23u * s) & 63u) * 64u + ((pixel.x + 41u * s) & 63u)];
            thresholdBase[s] = stochasticNoiseUnit(rank) + frameOffset;
        }
    }

    uint2 range = tileRanges[tileId];
    uint linearThread = localId.y * SABERCAT_TILE_SIZE + localId.x;

    for (uint batchStart = range.x; batchStart < range.y; batchStart += SABERCAT_TILE_THREADS)
    {
        uint batchSize = min(SABERCAT_TILE_THREADS, range.y - batchStart);

        // Cooperative load: one candidate per thread.
        if (linearThread < batchSize)
        {
            uint g = sortedValues[batchStart + linearThread];
            ProjectedGaussian splat = projected[g];
            batchIndex[linearThread] = g;
            batchCenter[linearThread] = splat.center;
            batchConic[linearThread] = splat.conic;
            batchAlpha[linearThread] = splat.alpha;
            batchDepth[linearThread] = splat.depth;

            uint base = g * SABERCAT_RAY_FORM_FLOATS;
            batchForm0[linearThread] = float4(rayForm[base + 0], rayForm[base + 1],
                                              rayForm[base + 2], rayForm[base + 3]);
            batchForm1[linearThread] = float4(rayForm[base + 4], rayForm[base + 5],
                                              rayForm[base + 6], rayForm[base + 7]);
            batchForm2[linearThread] = float4(rayForm[base + 8], rayForm[base + 9],
                                              rayForm[base + 10], rayForm[base + 11]);
        }
        GroupMemoryBarrierWithGroupSync();

        if (inside)
        {
            for (uint i = 0; i < batchSize; ++i)
            {
                // Same contributor test as the other rasterizers: a candidate
                // that does not contribute at this pixel has no alpha to cover
                // a sample with, so it is skipped.
                float alpha = 0.0f;
                float gaussian = 0.0f;
                bool clamped = false;
                if (!evaluateSplat(pixelPosition, batchCenter[i], batchConic[i], batchAlpha[i],
                                   alpha, gaussian, clamped))
                {
                    continue;
                }

                uint g = batchIndex[i];
                float t = rayDepthFromStaged(i, dxdy);
                for (uint s = 0; s < SABERCAT_ST_SAMPLES; ++s)
                {
                    // A 24-bit hash of the Gaussian index and the (frame, sample)
                    // seed, in [0, 1): a per-candidate rotation of the threshold.
                    float hashTerm =
                        (float)(stochasticHash(g + seedS[s]) >> 8u) * (1.0f / 16777216.0f);
                    float xi = fractf(thresholdBase[s] + hashTerm);
                    // Strict <: on a depth tie the first candidate in the
                    // tile's list keeps the sample.
                    if (alpha > xi && t < depthS[s])
                    {
                        depthS[s] = t;
                        colorS[s] = projected[g].color;
                    }
                }
            }
        }

        // Every thread reaches this barrier before the next batch overwrites
        // shared memory, including pixels outside the image.
        GroupMemoryBarrierWithGroupSync();
    }

    if (inside)
    {
        // Samples are summed in ascending order, then scaled by 1/S. The
        // coverage is the fraction of samples that took any candidate.
        float3 acc = float3(0.0f, 0.0f, 0.0f);
        uint covered = 0u;
        float nearest = infinity;
        for (uint s = 0; s < SABERCAT_ST_SAMPLES; ++s)
        {
            if (depthS[s] < infinity)
            {
                acc += colorS[s];
                covered += 1u;
                nearest = min(nearest, depthS[s]);
            }
        }

        float invSamples = 1.0f / (float)SABERCAT_ST_SAMPLES;
        outputImage[pixel] = float4(acc * invSamples, (float)covered * invSamples);
        // No covered sample means no surface: depth 0, as the other rasterizers
        // write for an empty pixel.
        outputDepth[pixel] = covered > 0u ? nearest : 0.0f;
    }
}
