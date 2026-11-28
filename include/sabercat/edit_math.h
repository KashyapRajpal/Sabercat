// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <glm/glm.hpp>

#include "sabercat/camera.h"
#include "sabercat/layer_transform.h"

namespace sabercat
{

// GPU-free edit math in double precision. A Similarity maps x to
// scale * rotation * x + translation, where rotation is a proper rotation
// (orthonormal, determinant +1) and scale is positive wherever an inverse or a
// layer edit is taken.
struct Similarity
{
    glm::dmat3 rotation{1.0};     // column-major, as glm
    glm::dvec3 translation{0.0};
    double scale = 1.0;

    // scale * rotation * x + translation.
    glm::dvec3 apply(const glm::dvec3 & x) const;
};

// a after b: x -> a(b(x)). Scales and rotations multiply; the translation is
// a.scale * a.rotation * b.translation + a.translation.
Similarity compose(const Similarity & a, const Similarity & b);

// The inverse, x -> (1/scale) * rotation^T * (y - translation). Throws
// std::invalid_argument unless scale > 0.
Similarity inverse(const Similarity & s);

// The placement's T * R * S as a Similarity. The rotation is the placement's
// quaternion normalized in double. Throws std::invalid_argument unless the scale
// is positive and the quaternion is non-zero and finite.
Similarity similarityOf(const LayerTransform & placement);

// A world edit about a pivot: x' = scale * rotation * (x - pivot) + pivot + translation.
// Takes the rotation as given; no validation.
Similarity worldEditAboutPivot(const glm::dmat3 & rotation, double scale, const glm::dvec3 & translation,
                               const glm::dvec3 & pivot);

// S_l = L^-1 o T o L in double, with L = layer and T = worldEdit: applied to a
// layer's Gaussians (in layer space), S_l gives the same points as T applied after
// the layer placement L. Expanded: rotation R_L^T R_T R_L, scale s_T, translation
// (1/s_L) R_L^T (s_T R_T t_L + t_T - t_L). Throws std::invalid_argument unless both
// scales are positive.
Similarity layerSpaceEditD(const Similarity & layer, const Similarity & worldEdit);

// layerSpaceEditD(similarityOf(layerPlacement), worldEdit), cast once to the float
// LayerTransform: the quaternion comes from the double rotation, normalized, in glm
// (w, x, y, z) order. Throws std::invalid_argument unless both scales are positive.
LayerTransform layerSpaceEdit(const LayerTransform & layerPlacement, const Similarity & worldEdit);

// The rectangle (x0, y0, x1, y1) spanned by a drag from a to b, in either direction:
// x0 <= x1 and y0 <= y1.
glm::vec4 normalizeRect(const glm::vec2 & a, const glm::vec2 & b);

// The CPU references the GPU select kernel is checked against. Each takes a
// WORLD-space point, computes in double, and treats its boundaries as inclusive.
// A NaN point or parameter is never contained.

// True when |dot(axes[i], p - center)| <= halfExtents[i] for each axis i. The
// columns of `axes` are the box's world-space axes, assumed orthonormal.
bool boxContains(const glm::dvec3 & p, const glm::dvec3 & center, const glm::dvec3 & halfExtents,
                 const glm::dmat3 & axes);

// True when |p - center| <= radius. False for a negative radius.
bool sphereContains(const glm::dvec3 & p, const glm::dvec3 & center, double radius);

// Projects a world point with the forward pass's camera, as computeProjectionGeometry
// in shaders/forward_common.slangh does: cam = worldToCamera * (p, 1); depth = -cam.z;
// u = fx * cam.x / depth + cx; v = cy - fy * cam.y / depth. Returns false, leaving the
// outputs untouched, when depth <= kMinDepth (render_config.h) or the depth or pixel
// is not finite.
bool projectToPixels(const glm::dvec3 & p, const CameraUniforms & camera, glm::dvec2 & pixel,
                     double & depth);

// True when projectToPixels() succeeds and the pixel lies in the rectangle
// (x0, y0, x1, y1), edges included.
bool rectContains(const glm::dvec3 & p, const CameraUniforms & camera, const glm::vec4 & rect);

} // namespace sabercat
