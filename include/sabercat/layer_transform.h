// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <memory>
#include <utility>
#include <vector>

namespace sabercat
{

class Device;
class GaussianCloud;

// Rigid placement of one layer: translation, rotation, UNIFORM scale. Uniform
// scale is what keeps every per-Gaussian quantity exact: means move with the
// full matrix, rotations compose as quaternions, scales multiply, and SH bands
// 1..3 rotate by the rotation alone (directions are unaffected by translation
// and uniform scale). A non-uniform placement would need re-diagonalizing every
// covariance, which the glTF reader does for node transforms; a layer does not.
struct LayerTransform
{
    glm::quat rotation{1.0f, 0.0f, 0.0f, 0.0f}; // glm (w, x, y, z) constructor; unit length
    glm::vec3 translation{0.0f};
    float scale = 1.0f;                          // > 0

    // T * R * S, as a glTF node matrix would hold it.
    glm::mat4 matrix() const;
    // Exactly identity: unit rotation, zero translation, scale 1.
    bool isIdentity() const;
};

// A copy of `source`'s host vectors with `transform` applied exactly as the glTF
// reader bakes a node transform: mean -> M * mean, rotation -> transform.rotation * q
// (renormalized, xyzw canonical order), scale -> scale * s, SH bands 1..3 rotated by
// transform.rotation (DC unchanged), normals rotated, other relight streams copied.
// shDegree/antialiased copied. Identity -> bit-exact copy.
// Host-side only (no GPU work); call computeBounds() on the result.
std::shared_ptr<GaussianCloud> transformedCopy(Device & device, const GaussianCloud & source,
                                               const LayerTransform & transform);

// Concatenates layers (each transformed) into one cloud in order. shDegree = max; lower-
// degree layers zero-padded. A relight stream is kept only if EVERY layer has it
// (otherwise dropped with one Logger::warning naming the stream). antialiased: true only
// if all layers are. Throws on an empty list.
std::shared_ptr<GaussianCloud> mergeLayers(
    Device & device, const std::vector<std::pair<const GaussianCloud *, LayerTransform>> & layers);

} // namespace sabercat
