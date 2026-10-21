// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Debug views over the visibility buffer (SABERCAT_MAX_LIGHTS
// layers of width*height floats), so LightingOutput::Visibility can show
// where a visibility source fails and where the teacher samples landed:
//   mode 0  visibility = |visibility - reference|, every layer
//           (reference: the ray-traced shadows, snapshotted first)
//   mode 1  clear every layer to 0
//   mode 2  one thread per teacher sample: mark its pixel in its light's
//           layer, 0.5 for a smooth sample, 1.0 for an edge sample
// (modes 1 + 2 = the sample-locations view)

#include "config.slangh"

struct DebugConstants
{
    uint mode;
    uint width;
    uint height;
    uint sampleCount;
};

// sabercat::VisibilityTrainingSample (std430, 32 B); only lightIndex is read.
struct VisibilityTrainingSample
{
    float3 position;
    float visibilityTarget;
    float3 normal;
    uint lightIndex;
};

RWStructuredBuffer<float> visibility;
StructuredBuffer<float> reference;
StructuredBuffer<uint> samplePixels; // pixel index, top bit = edge sample
StructuredBuffer<VisibilityTrainingSample> samples;

[vk::push_constant]
ConstantBuffer<DebugConstants> debug;

[shader("compute")]
[numthreads(SABERCAT_WORKGROUP_SIZE, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint layerSize = debug.width * debug.height;
    if (debug.mode == 2u)
    {
        if (id.x >= debug.sampleCount)
        {
            return;
        }
        uint tagged = samplePixels[id.x];
        uint pixel = tagged & 0x7FFFFFFFu;
        uint light = samples[id.x].lightIndex;
        visibility[light * layerSize + pixel] = (tagged & 0x80000000u) != 0u ? 1.0f : 0.5f;
        return;
    }
    if (id.x >= layerSize)
    {
        return;
    }
    for (uint l = 0; l < SABERCAT_MAX_LIGHTS; ++l)
    {
        uint i = l * layerSize + id.x;
        visibility[i] = debug.mode == 0u ? abs(visibility[i] - reference[i]) : 0.0f;
    }
}
