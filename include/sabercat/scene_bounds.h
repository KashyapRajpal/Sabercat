// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <glm/glm.hpp>

#include <limits>

namespace sabercat
{

// Axis-aligned bounds of a loaded scene, in world space.
//
// Lives in its own header so both GaussianCloud (which computes it) and
// Renderer (which republishes it) can use the type without either public
// header having to include the other.
//
// Loaded scenes vary enormously in scale -- the reference 3DGS .ply scenes sit
// near the origin a few units across, while the Niantic .spz samples are
// ~500-unit outdoor captures with a full background shell. A viewer with one
// fixed default camera shows the second kind from the inside, which reads as a
// soft sphere filling the screen rather than as a scene. Bounds are what let a
// camera frame any of them.
struct SceneBounds
{
    glm::vec3 min{0.0f};
    glm::vec3 max{0.0f};
    bool valid = false;

    // --- Robust framing target ---
    //
    // The AABB is the wrong thing to point a camera at. Outdoor captures carry
    // a distant background shell that dwarfs the subject: measured on
    // racoonfamily.spz, 90% of Gaussians lie within 1.83 units of the center
    // while the bounds reach 241 -- a 132x ratio. Framing the full bounds there
    // puts the camera ~1000 units out and the actual subject occupies a
    // fraction of a percent of the screen, which looks like a broken render.
    //
    // So framing uses a median center and a 90th-percentile radius, which
    // ignores the shell. Compact scenes are unaffected (luigi's max/p90 ratio
    // is 1.1, bonsai's 2.1); it only bites where it should.
    glm::vec3 focusCenter{0.0f};
    float focusRadius = 0.0f;

    glm::vec3 center() const
    {
        return (min + max) * 0.5f;
    }

    glm::vec3 extent() const
    {
        return max - min;
    }

    // Radius of the bounding SPHERE, not the half-extent of the box. This is
    // the FULL extent including any background shell -- useful for reporting
    // and for depth-range heuristics, but see focusRadius for framing.
    float radius() const
    {
        return glm::length(extent()) * 0.5f;
    }
};

} // namespace sabercat
