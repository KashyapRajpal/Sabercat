// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0
// Portions derived from NVlabs FLIP, BSD-3-Clause

// HDR-FLIP, once per exposure after the LDR-FLIP error of that exposure: keeps the largest
// error per pixel and the exposure at which it occurred, as FLIP's image::setMaxExposure does.
// The first exposure (hdrFlags bit 1) starts both maps: the error is clamped at 0 and the
// exposure index is 0. Later exposures replace a value only when strictly greater. Overlap-local
// thread (x, y); pixel i = y * width + x.

#include "flip_common.slangh"

[vk::binding(1, 0)] StructuredBuffer<float> tmpError;        // this exposure's LDR-FLIP error
[vk::binding(2, 0)] RWStructuredBuffer<float> maxError;     // running maximum over the exposures
[vk::binding(3, 0)] RWStructuredBuffer<float> exposureMap;  // i / (count - 1) of the maximum

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= params.width || id.y >= params.height)
    {
        return;
    }

    const uint i = id.y * params.width + id.x;
    const float error = tmpError[i];
    if ((params.hdrFlags & 2u) != 0u)
    {
        maxError[i] = (error > 0.0f) ? error : 0.0f;
        exposureMap[i] = 0.0f;
    }
    else if (error > maxError[i])
    {
        maxError[i] = error;
        exposureMap[i] = params.exposureFraction;
    }
}
