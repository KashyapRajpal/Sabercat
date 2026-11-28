// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Selection (Studio phase C3): evaluates one shape predicate over a layer's
// Gaussians and combines it into the layer's selection mask. One thread per
// 32-Gaussian mask word: each thread reads the 32 Gaussians of its word and writes
// that word alone, so there are no atomics and the result is deterministic.
//
// The predicate runs on the world position of each mean, mul(layerToWorld,
// float4(mean, 1)) in fp32, with the same placement the layer merge bakes. Box and
// sphere are tested in world space. A rectangle projects the world position with
// the camera of the view it was drawn in and tests the framebuffer pixel
// inclusively, through every depth. A non-finite value anywhere makes the test
// false: every comparison below is written so that NaN fails it.
//
// The op combines the predicate with the word's old bits: replace, add (or) or
// subtract (and not). Bits at or beyond `count` are always written as zero.
//
// Plain if/else throughout, no ternaries choosing between temporaries: that form
// is a known Slang GLSL miscompile risk.

#include "config.slangh"
#include "edit_select_common.slangh"

// Binding 0: the layer's selection mask, one uint per 32 Gaussians.
[[vk::binding(0, 0)]] RWStructuredBuffer<uint> mask;
// Binding 1: the layer's local-space means (the GaussianCloud means stream).
[[vk::binding(1, 0)]] StructuredBuffer<float4> means;
// Binding 2: the one SelectParams for this call.
[[vk::binding(2, 0)]] StructuredBuffer<SelectParams> params;

// Whether Gaussian g satisfies the query's shape. g must be below params[0].count.
bool isSelected(uint g)
{
    const float3 local = means[g].xyz;
    const float3 world = mul(params[0].layerToWorld, float4(local, 1.0f)).xyz;

    if (params[0].shape == SABERCAT_SELECT_SHAPE_BOX)
    {
        const float3 d = world - params[0].centerRadius.xyz;
        const float3 extents = params[0].halfExtents.xyz;
        return abs(dot(params[0].boxAxis[0].xyz, d)) <= extents.x
               && abs(dot(params[0].boxAxis[1].xyz, d)) <= extents.y
               && abs(dot(params[0].boxAxis[2].xyz, d)) <= extents.z;
    }

    if (params[0].shape == SABERCAT_SELECT_SHAPE_SPHERE)
    {
        const float3 d = world - params[0].centerRadius.xyz;
        const float radius = params[0].centerRadius.w;
        return dot(d, d) <= radius * radius;
    }

    // Rectangle: the pixel the Gaussian's center projects to, in the drawn view.
    // The same projection as computeProjectionGeometry in forward_common.slangh.
    const float3 cam = mul(params[0].worldToCamera, float4(world, 1.0f)).xyz;
    const float depth = -cam.z;
    // `depth > MIN` is false for NaN, so a non-finite depth is never selected.
    if (depth > SABERCAT_MIN_DEPTH)
    {
        const float fx = params[0].intrinsics.x;
        const float fy = params[0].intrinsics.y;
        const float cx = params[0].intrinsics.z;
        const float cy = params[0].intrinsics.w;
        const float u = fx * cam.x / depth + cx;
        const float v = cy - fy * cam.y / depth;
        const float4 rect = params[0].cameraRect; // x0, y0, x1, y1
        return u >= rect.x && u <= rect.z && v >= rect.y && v <= rect.w;
    }
    return false;
}

[shader("compute")]
[numthreads(SABERCAT_WORKGROUP_SIZE, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    const uint w = id.x;
    if (w >= params[0].wordCount)
    {
        return;
    }

    const uint count = params[0].count;
    const uint op = params[0].op;
    const uint oldWord = mask[w];

    uint newWord = 0u;
    for (uint j = 0; j < 32u; ++j)
    {
        const uint g = 32u * w + j;
        if (g < count)
        {
            const bool hit = isSelected(g);
            const uint oldBit = (oldWord >> j) & 1u;

            uint bit = 0u;
            if (op == SABERCAT_SELECT_OP_REPLACE)
            {
                if (hit)
                {
                    bit = 1u;
                }
            }
            else if (op == SABERCAT_SELECT_OP_ADD)
            {
                bit = oldBit;
                if (hit)
                {
                    bit = 1u;
                }
            }
            else
            {
                // Subtract: keep the old bit only where the predicate misses.
                bit = oldBit;
                if (hit)
                {
                    bit = 0u;
                }
            }
            newWord = newWord | (bit << j);
        }
    }

    mask[w] = newWord;
}
