// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <string>

#include <glm/glm.hpp>

#include "sabercat/layer_transform.h"
#include "sh_rotation.h"

namespace sabercat
{

// A layer's node transform, decomposed exactly as the glTF reader does it for a
// node (nodeLocalMatrix, then decomposeTransform), so the baked numbers match
// what the reader produces from the same .glb.
struct BakeFrame
{
    glm::dmat4 matrix{1.0};                                      // T * R * S
    sh_rotation::Mat3 rotation{};                                // rotation factor
    std::array<double, 4> rotationQuaternion{0.0, 0.0, 0.0, 1.0}; // xyzw
    double uniformScale = 1.0;
    std::array<sh_rotation::BandMatrix, 3> shBands{}; // bands 1..3
};

// The double-precision frame shared by the host bake and the GPU layer merge.
BakeFrame makeBakeFrame(const LayerTransform & transform);

// Rejects a placement that cannot be baked (non-finite values, a scale that is not
// positive, a rotation that is not unit length); throws std::runtime_error naming `where`.
void validateLayerTransform(const LayerTransform & transform, const std::string & where);

} // namespace sabercat
