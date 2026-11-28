// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Sorting study: the per-pixel resort window rasterizer (PixelWindow and
// Hybrid modes). It implements the WindowRayDepth model of the reference
// (cpu_reference.cpp): the tile's list is walked in its own order, and each
// pixel reorders the contributors it sees through a small window of K
// entries sorted by (t*, Gaussian index), where t* is the peak depth along
// the pixel ray (sort_common.slangh). K is the SABERCAT_SORT_WINDOW define:
// one compiled variant per window size (tile_rasterize_window_k8/_k16/_k32).
//
// One workgroup per 16x16 tile, one thread per pixel, structured like
// tile_rasterize.slang: the workgroup stages each batch of the tile's slice
// into shared memory once, and every thread then runs its own pixel against
// the batch. The per-pixel window lives in private arrays, so it never
// touches shared memory and needs no barriers of its own.
//
// Writes only outputImage and outputDepth; the sort-study modes read nothing
// else back from the forward pass.

#include "sort_common.slangh"

#ifndef SABERCAT_SORT_WINDOW
#error "tile_rasterize_window.slang needs -DSABERCAT_SORT_WINDOW=<K>; build it through the sabercat_add_shader_variant calls in shaders/CMakeLists.txt"
#endif

StructuredBuffer<ProjectedGaussian> projected;
StructuredBuffer<uint> sortedValues;   // Gaussian index, tile-major, default order
StructuredBuffer<uint2> tileRanges;    // per tile: (firstInstance, lastInstance)
// SABERCAT_RAY_FORM_FLOATS floats per Gaussian, indexed index * 12 + k.
StructuredBuffer<float> rayForm;

[vk::push_constant]
ConstantBuffer<CameraUniforms> camera;

RWTexture2D<float4> outputImage;
RWTexture2D<float> outputDepth;

// One batch of the tile's candidates, staged cooperatively by the workgroup.
// Only what the contributor test and the ray depth need is staged: the blend
// of a popped or flushed entry re-reads its ProjectedGaussian, which keeps
// the shared-memory footprint near 20 KB for a 256-thread group.
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
// Threads that have collapsed (transmittance below threshold, or outside the
// image). Same role as in tile_rasterize.slang: once every thread has, no
// later batch can change the output.
groupshared uint doneCount;

// Strict total order on window entries: t* first, then the Gaussian index as
// the tie-break, so two entries never compare equal and the window order is
// well defined. Matches the reference's (t*, Gaussian index) sort.
bool keyLess(float ta, uint ga, float tb, uint gb)
{
    return ta < tb || (ta == tb && ga < gb);
}

// Peak depth along the pixel ray for the staged Gaussian i. This repeats the
// arithmetic of rayDepthAlongRay (sort_common.slangh) operation for operation,
// because that function reads an SSBO and this one reads the shared copy; the
// two must stay in step, since the sort keys and the window must agree on t*.
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

// Blend one Gaussian into the pixel, if it contributes there. Called for the
// popped entries and for the final flush; the candidate test already ran on
// the staged copy, and evaluateSplat is deterministic, so a popped entry that
// contributed at arrival contributes again here. Returns true when the pixel
// has become done (transmittance below the threshold after this blend).
bool blendGaussian(uint g, float2 pixelPosition, inout float3 accumulated,
                   inout float transmittance, inout float frontDepth, inout bool hasFrontDepth)
{
    ProjectedGaussian splat = projected[g];
    float alpha = 0.0f;
    float gaussian = 0.0f;
    bool clamped = false;
    if (evaluateSplat(pixelPosition, splat.center, splat.conic, splat.alpha, alpha, gaussian, clamped))
    {
        if (!hasFrontDepth)
        {
            frontDepth = splat.depth;
            hasFrontDepth = true;
        }
        accumulated += transmittance * alpha * splat.color;
        transmittance *= (1.0f - alpha);
    }
    return transmittance < 1e-4f;
}

[shader("compute")]
[numthreads(SABERCAT_TILE_SIZE, SABERCAT_TILE_SIZE, 1)]
void main(uint3 groupId : SV_GroupID, uint3 localId : SV_GroupThreadID)
{
    uint tileId = groupId.y * camera.tilesX + groupId.x;
    uint2 pixel = uint2(groupId.x * SABERCAT_TILE_SIZE + localId.x,
                        groupId.y * SABERCAT_TILE_SIZE + localId.y);

    // Pixel centers at integer coordinates, as in tile_rasterize.slang.
    float2 pixelPosition = float2(pixel);
    // Camera-space ray through this pixel, (dx, dy) of d = (dx, dy, -1) --
    // the same parameterization the reference and the sort keys use.
    float2 dxdy = float2((pixelPosition.x - camera.cx) / camera.fx,
                         -(pixelPosition.y - camera.cy) / camera.fy);

    uint2 range = tileRanges[tileId];
    uint linearThread = localId.y * SABERCAT_TILE_SIZE + localId.x;

    // The window: entries sorted ascending by (t*, Gaussian index). Element 0
    // is the smallest, so the pop removes it and a flush walks front to back.
    float bufT[SABERCAT_SORT_WINDOW];
    uint bufG[SABERCAT_SORT_WINDOW];
    uint bufCount = 0;
    for (uint j = 0; j < SABERCAT_SORT_WINDOW; ++j)
    {
        bufT[j] = 0.0f;
        bufG[j] = 0u;
    }

    float3 accumulated = float3(0.0f, 0.0f, 0.0f);
    float transmittance = 1.0f;
    float frontDepth = 0.0f;
    bool hasFrontDepth = false;
    bool done = pixel.x >= camera.imageWidth || pixel.y >= camera.imageHeight;
    bool reportedDone = false;

    if (linearThread == 0)
    {
        doneCount = 0;
    }
    GroupMemoryBarrierWithGroupSync();

    if (done)
    {
        InterlockedAdd(doneCount, 1);
        reportedDone = true;
    }

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

        // A done pixel never touches its window again. It still reaches the
        // barriers below, so the whole group stays in step.
        if (!done)
        {
            for (uint i = 0; i < batchSize; ++i)
            {
                // Same contributor test as tile_rasterize: non-contributors
                // are skipped and do not enter the window.
                float alpha = 0.0f;
                float gaussian = 0.0f;
                bool clamped = false;
                if (!evaluateSplat(pixelPosition, batchCenter[i], batchConic[i], batchAlpha[i],
                                   alpha, gaussian, clamped))
                {
                    continue;
                }

                float t = rayDepthFromStaged(i, dxdy);
                uint g = batchIndex[i];

                // A full window pops its smallest entry first and blends it,
                // then the arriving contributor is inserted. Popping before the
                // insert is what the model does, so the arrival can never pop
                // itself.
                if (bufCount == SABERCAT_SORT_WINDOW)
                {
                    uint popped = bufG[0];
                    for (uint j = 0; j + 1 < bufCount; ++j)
                    {
                        bufT[j] = bufT[j + 1];
                        bufG[j] = bufG[j + 1];
                    }
                    bufCount -= 1;

                    if (blendGaussian(popped, pixelPosition, accumulated, transmittance,
                                      frontDepth, hasFrontDepth))
                    {
                        // Done: the window is abandoned, not flushed.
                        done = true;
                        break;
                    }
                }

                // Insert in ascending order: the new entry goes before the first
                // entry that compares greater. bufCount < K here, so the shift
                // below stays in range.
                uint slot = bufCount;
                for (uint j = 0; j < bufCount; ++j)
                {
                    if (keyLess(t, g, bufT[j], bufG[j]))
                    {
                        slot = j;
                        break;
                    }
                }
                for (uint j = bufCount; j > slot; --j)
                {
                    bufT[j] = bufT[j - 1];
                    bufG[j] = bufG[j - 1];
                }
                bufT[slot] = t;
                bufG[slot] = g;
                bufCount += 1;
            }

            if (done && !reportedDone)
            {
                InterlockedAdd(doneCount, 1);
                reportedDone = true;
            }
        }

        // Every thread reaches this barrier before the next batch overwrites
        // shared memory, including threads that finished early.
        GroupMemoryBarrierWithGroupSync();

        // The whole tile has collapsed: no later batch can change the output.
        // Every in-image pixel is done here, so no flush is owed either.
        if (doneCount == SABERCAT_TILE_THREADS)
        {
            break;
        }
    }

    // The candidate list ended with the pixel still open: flush the window in
    // ascending order, with the same early-out as the pop.
    if (!done)
    {
        for (uint j = 0; j < bufCount; ++j)
        {
            if (blendGaussian(bufG[j], pixelPosition, accumulated, transmittance,
                              frontDepth, hasFrontDepth))
            {
                break;
            }
        }
    }

    if (pixel.x < camera.imageWidth && pixel.y < camera.imageHeight)
    {
        // Alpha is coverage: 1 - remaining transmittance.
        outputImage[pixel] = float4(accumulated, 1.0f - transmittance);
        outputDepth[pixel] = frontDepth;
    }
}
