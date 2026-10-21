// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <glm/glm.hpp>

#include <cstdint>

namespace sabercat
{

// Pinhole camera matching the canonical convention shared with the
// creator/trainer side:
//
//   - Right-handed, +Y up, the camera looks along its local -Z.
//   - Depth is positive in front of the camera: d = -local.z.
//   - u = fx * x / d + cx
//   - v = cy - fy * y / d        (note the SIGN FLIP: +Y in the world is
//                                 up, but +v goes DOWN the image)
//
// The v flip is the easy thing to get wrong; it also propagates into the
// projection Jacobian's second row, which is why that row is negated.
struct Camera
{
    uint32_t width = 0;
    uint32_t height = 0;

    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;

    // Camera-to-world transform; the shaders receive its inverse.
    glm::mat4 cameraToWorld {1.0f};

    glm::mat4 worldToCamera() const
    {
        return glm::inverse(cameraToWorld);
    }

    bool valid() const
    {
        return width > 0 && height > 0 && fx > 0.0f && fy > 0.0f;
    }
};

// GPU-side camera block. Must match the CameraUniforms layout declared in
// shaders/forward_common.slangh -- std140/std430 rules put mat4 on a 16-byte
// boundary and pack the trailing scalars into vec4-sized slots, which the
// field order below already satisfies.
struct CameraUniforms
{
    glm::mat4 worldToCamera {1.0f};

    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;

    uint32_t imageWidth = 0;
    uint32_t imageHeight = 0;
    uint32_t tilesX = 0;
    uint32_t tilesY = 0;

    uint32_t gaussianCount = 0;
    uint32_t shDegree = 0; // max SH degree to evaluate (0-3)
    // SABERCAT_RENDER_FLAG_* bits (render_config.h). Took over what
    // was pad0, so the struct is still exactly 128 bytes. Only
    // kRenderFlagAntialiased is the caller's to set; ForwardPass owns
    // kRenderFlagSaveBackwardState and overwrites it on every run().
    uint32_t renderFlags = 0;
    uint32_t pad1 = 0;

    // World-space camera position, for the SH view direction.
    // xyz used, w padding -- same padded-vec3 convention as the canonical
    // Gaussian streams. This lands the struct at exactly 128 bytes, the
    // Vulkan-guaranteed minimum maxPushConstantsSize (asserted in
    // forward_pass.cpp), so anything added after this needs its own
    // binding rather than another field here.
    glm::vec4 cameraPositionWorld {0.0f};
};

} // namespace sabercat
