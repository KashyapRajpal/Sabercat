// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Forward pass, stage 4: blend the sorted Gaussians into the image.
//
// One workgroup per 16x16 tile, one thread per pixel. The workgroup walks
// its tile's slice of the sorted instance list in batches, cooperatively
// staging each batch into shared memory (so the SSBO is read once per
// workgroup rather than once per pixel), then every thread blends that
// batch front-to-back against its own pixel.
//
// Front-to-back compositing:
//     C += T * alpha_i * color_i        T *= (1 - alpha_i)
// with T (transmittance) starting at 1. Iteration stops early once T falls
// below a threshold -- everything behind is invisible.

#include "forward_common.slangh"

StructuredBuffer<ProjectedGaussian> projected;
StructuredBuffer<uint> sortedValues;   // Gaussian index, tile-major/depth-sorted
StructuredBuffer<uint2> tileRanges;    // per tile: (firstInstance, lastInstance)

[vk::push_constant]
ConstantBuffer<CameraUniforms> camera;

RWTexture2D<float4> outputImage;
// Depth of the frontmost Gaussian that actually contributes at this pixel
// (first alpha >= 1/255 hit in front-to-back order) -- a debug/visualize
// channel, not consumed by any later forward-pass stage. 0 where nothing
// contributes, matching the "no coverage" sentinel outputImage.a == 0 uses.
RWTexture2D<float> outputDepth;
// Verification-only: one counter per tile, incremented once per batch this
// workgroup actually fetches. Not read by any rendering path -- forward_test
// uses it to prove the group-wide early-out below actually skips batches,
// rather than trusting the output image alone (an early-out that silently
// never fires would still produce a correct image).
RWStructuredBuffer<uint> batchesFetched;
// Per-pixel state the backward pass needs and cannot cheaply
// recompute -- (asuint(final transmittance), nContrib), where nContrib is
// 1 + the position (relative to the tile's range start) of the last
// Gaussian that blended into this pixel, 0 if none did. backward_rasterize
// walks [0, nContrib) back to front, dividing (1 - alpha) back out of T.
// Final T is stored directly rather than recovered from 1 - outputImage.a,
// which loses ~3 significant digits right where T approaches the 1e-4
// early-out. Written only under SABERCAT_RENDER_FLAG_SAVE_BACKWARD_STATE,
// which ForwardPass sets in differentiable mode, so the viewer skips it.
RWStructuredBuffer<uint2> pixelState;
// The same float4 outputImage gets, as a width*height row-major
// buffer -- the readImage()/dL/dImage layout -- so a trainer computes its
// loss on the GPU straight from it (Renderer::imageView()). Written under
// the same flag as pixelState, so the viewer pays nothing.
RWStructuredBuffer<float4> outputBuffer;

// One batch of Gaussians staged cooperatively by the whole workgroup.
groupshared float2 batchCenter[SABERCAT_TILE_THREADS];
groupshared float3 batchConic[SABERCAT_TILE_THREADS];
groupshared float3 batchColor[SABERCAT_TILE_THREADS];
groupshared float batchAlpha[SABERCAT_TILE_THREADS];
groupshared float batchDepth[SABERCAT_TILE_THREADS];
// Count of threads in this workgroup that have collapsed (transmittance
// below threshold, or outside the image). Once every thread has, the whole
// tile is done and no later batch can change the output -- the outer loop
// below breaks instead of continuing to cooperatively load dead batches.
groupshared uint doneCount;

[shader("compute")]
[numthreads(SABERCAT_TILE_SIZE, SABERCAT_TILE_SIZE, 1)]
void main(uint3 groupId : SV_GroupID, uint3 localId : SV_GroupThreadID)
{
    uint tileId = groupId.y * camera.tilesX + groupId.x;
    uint2 pixel = uint2(groupId.x * SABERCAT_TILE_SIZE + localId.x,
                        groupId.y * SABERCAT_TILE_SIZE + localId.y);

    // Pixel centers are at integer coordinates, matching the reference
    // rasterizer's grid (NOT offset by half a pixel).
    float2 pixelPosition = float2(pixel);

    uint2 range = tileRanges[tileId];
    uint linearThread = localId.y * SABERCAT_TILE_SIZE + localId.x;

    float3 accumulated = float3(0.0f, 0.0f, 0.0f);
    float transmittance = 1.0f;
    float frontDepth = 0.0f;
    bool hasFrontDepth = false;
    uint contributorCount = 0; // nContrib, see pixelState
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

        if (linearThread == 0)
        {
            InterlockedAdd(batchesFetched[tileId], 1);
        }

        // Cooperative load: one Gaussian per thread.
        if (linearThread < batchSize)
        {
            ProjectedGaussian g = projected[sortedValues[batchStart + linearThread]];
            batchCenter[linearThread] = g.center;
            batchConic[linearThread] = g.conic;
            batchColor[linearThread] = g.color;
            batchAlpha[linearThread] = g.alpha;
            batchDepth[linearThread] = g.depth;
        }
        GroupMemoryBarrierWithGroupSync();

        if (!done)
        {
            for (uint i = 0; i < batchSize; ++i)
            {
                // Shared with backward_rasterize (forward_common.slangh), so
                // both passes skip exactly the same (pixel, Gaussian) pairs.
                // evaluateSplat writes all three on every path; initialized
                // anyway so a later edit to it can't leave them undefined.
                float alpha = 0.0f;
                float gaussian = 0.0f;
                bool clamped = false;
                if (!evaluateSplat(pixelPosition, batchCenter[i], batchConic[i], batchAlpha[i],
                                   alpha, gaussian, clamped))
                {
                    continue;
                }
                contributorCount = batchStart - range.x + i + 1;

                if (!hasFrontDepth)
                {
                    frontDepth = batchDepth[i];
                    hasFrontDepth = true;
                }

                accumulated += transmittance * alpha * batchColor[i];
                transmittance *= (1.0f - alpha);

                if (transmittance < 1e-4f)
                {
                    done = true;
                    break;
                }
            }

            if (done && !reportedDone)
            {
                InterlockedAdd(doneCount, 1);
                reportedDone = true;
            }
        }

        // Every thread must reach this barrier before the next batch
        // overwrites shared memory -- including threads that finished early,
        // which is why `done` breaks the inner loop rather than returning.
        GroupMemoryBarrierWithGroupSync();

        // Every thread's doneCount report for this batch is visible past the
        // barrier above; if the whole tile has collapsed, no later batch can
        // change the output, so stop fetching them.
        if (doneCount == SABERCAT_TILE_THREADS)
        {
            break;
        }
    }

    if (pixel.x < camera.imageWidth && pixel.y < camera.imageHeight)
    {
        // Alpha is coverage: 1 - remaining transmittance.
        const float4 color = float4(accumulated, 1.0f - transmittance);
        outputImage[pixel] = color;
        outputDepth[pixel] = frontDepth;
        if ((camera.renderFlags & SABERCAT_RENDER_FLAG_SAVE_BACKWARD_STATE) != 0)
        {
            const uint pixelIndex = pixel.y * camera.imageWidth + pixel.x;
            pixelState[pixelIndex] = uint2(asuint(transmittance), contributorCount);
            outputBuffer[pixelIndex] = color;
        }
    }
}
