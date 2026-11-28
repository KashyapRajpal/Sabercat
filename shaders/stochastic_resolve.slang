// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Sorting study: the temporal resolve of the Stochastic mode. The raster stage
// writes one raw frame of stochastic coverage into outputImage; this stage folds
// it into a per-pixel history buffer and writes the result back as the frame.
//
// The host computes the weight per run (ForwardPass): 1 takes the raw frame
// outright (a reset, or a camera change with the plain-accumulate mode), 1/k is
// the running mean over k static frames, and the EMA weight is used when the
// exponential mode is on. Weights of 1 or more copy the raw frame exactly, so a
// reset never depends on what the history buffer held before it.

// Bound in declaration order, like the other kernels: 0 outputImage, 1 hist.
RWTexture2D<float4> outputImage;
RWStructuredBuffer<float4> hist;

struct ResolveParams
{
    uint width;
    uint height;
    float weight;
    uint pad;
};

[vk::push_constant]
ConstantBuffer<ResolveParams> params;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    uint2 pixel = dispatchThreadId.xy;
    if (pixel.x >= params.width || pixel.y >= params.height)
    {
        return;
    }

    uint index = pixel.y * params.width + pixel.x;
    float4 current = outputImage[pixel];
    float4 resolved;
    if (params.weight >= 1.0f)
    {
        resolved = current;
    }
    else
    {
        float4 previous = hist[index];
        resolved = previous + (current - previous) * params.weight;
    }

    hist[index] = resolved;
    outputImage[pixel] = resolved;
}
