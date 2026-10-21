// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Orbit camera for the apps (apps/common): a bounded-scene orbit / pan / zoom /
// roll camera, and the GLFW mouse + keyboard input that drives it.
#pragma once

#include "sabercat/camera.h"
#include "sabercat/scene_bounds.h"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>

// The GLFW window type (declared the same way glfw3.h declares it).
struct GLFWwindow;

namespace sabercat::app
{

// Orbit camera: mouse-drag rotates around `pivot` at `distance`, matching
// how every 3DGS viewer handles a bounded trained scene (vs. a fly camera,
// better suited to unbounded/large scenes).
//
// Orientation is a quaternion, accumulated directly from each frame's
// incremental yaw/pitch/roll deltas rather than stored as separate angles
// and reconstructed via glm::lookAt every frame. That reconstruction is
// what previously needed a +-89 degree pitch clamp: lookAt derives "right"
// from cross(forward, worldUp), which collapses to zero -- a singular,
// NaN-producing view matrix -- once forward and worldUp become collinear
// (pitch -> +-90). Accumulating the quaternion sidesteps that: there is no
// per-frame reconstruction from a forward/up pair, so no cross product to
// degenerate, and no pole to avoid. Full range, including straight up/down
// and past-vertical orbits (like Blender), just works.
struct OrbitCamera
{
    glm::vec3 pivot{0.0f, 0.0f, 0.0f};
    float distance = 3.0f;
    glm::quat orientation{1.0f, 0.0f, 0.0f, 0.0f}; // identity: -Z forward, +Y up, world-aligned

    static constexpr float kMinDistance = 0.05f;

    glm::vec3 forward() const
    {
        return orientation * glm::vec3(0.0f, 0.0f, -1.0f);
    }
    glm::vec3 right() const
    {
        return orientation * glm::vec3(1.0f, 0.0f, 0.0f);
    }
    glm::vec3 up() const
    {
        return orientation * glm::vec3(0.0f, 1.0f, 0.0f);
    }

    // Applies incremental rotations and re-normalizes (repeated quaternion
    // multiplication drifts off unit length over many frames). World-space
    // yaw (pre-multiply by a world +Y rotation) keeps the horizon level
    // during orbiting; pitch and roll are camera-local (post-multiply)
    // since they're relative to however the camera is currently oriented.
    void applyDelta(float yawDelta, float pitchDelta, float rollDelta)
    {
        if (yawDelta != 0.0f)
        {
            orientation = glm::angleAxis(yawDelta, glm::vec3(0.0f, 1.0f, 0.0f)) * orientation;
        }
        if (pitchDelta != 0.0f)
        {
            orientation = orientation * glm::angleAxis(pitchDelta, glm::vec3(1.0f, 0.0f, 0.0f));
        }
        if (rollDelta != 0.0f)
        {
            orientation = orientation * glm::angleAxis(rollDelta, glm::vec3(0.0f, 0.0f, 1.0f));
        }
        if (yawDelta != 0.0f || pitchDelta != 0.0f || rollDelta != 0.0f)
        {
            orientation = glm::normalize(orientation);
        }
    }

    // Places the camera so the whole scene fits the vertical field of view.
    //
    // Without this every scene opens at the fixed default distance of 3.0,
    // which is fine for the reference .ply scenes (a few units across, near the
    // origin) and useless for the .spz samples (~500 units across): the camera
    // starts INSIDE those, and what you see is the far background shell from
    // within, which reads as a soft sphere filling the screen rather than as a
    // broken loader. Scrolling out would take ~50 notches at 0.9 per notch.
    //
    // Frames the bounding SPHERE, not the box half-extent, so the corners of a
    // wide scene stay inside the frustum. Orientation is left alone -- this
    // answers "where do I stand", not "which way do I look".
    void frame(const SceneBounds & bounds, uint32_t height, float fy)
    {
        if (!bounds.valid)
        {
            return;
        }

        // focusCenter/focusRadius, NOT center()/radius(): the AABB includes any
        // distant background shell, and framing that leaves the subject a speck.
        pivot = bounds.focusCenter;

        // toCamera() below sets fx = fy = height, i.e. tan(halfFovY) =
        // (height/2) / fy. Derived from the same numbers rather than hardcoded,
        // so changing the FOV there cannot silently break framing here.
        const float tanHalfFov = (static_cast<float>(height) * 0.5f) / fy;

        const float radius = bounds.focusRadius;
        constexpr float kMargin = 1.15f; // a little air around the scene
        distance = std::max(kMinDistance, (radius / tanHalfFov) * kMargin);
    }

    Camera toCamera(uint32_t width, uint32_t height) const
    {
        glm::vec3 eye = pivot - distance * forward();
        glm::mat4 cameraToWorld = glm::translate(glm::mat4(1.0f), eye) * glm::mat4_cast(orientation);

        Camera camera;
        camera.width = width;
        camera.height = height;
        camera.fx = camera.fy = static_cast<float>(height); // ~53 degree vertical FOV
        camera.cx = static_cast<float>(width) * 0.5f;
        camera.cy = static_cast<float>(height) * 0.5f;
        camera.cameraToWorld = cameraToWorld;
        return camera;
    }
};

// Polls mouse/keyboard each frame (matching the V-key polling style already
// used for the debug view) rather than installing GLFW callbacks -- avoids
// having to chain through Dear ImGui's own installed callbacks
// (ImGui_ImplGlfw_InitForVulkan's install_callbacks=true). Scroll is the one
// input ImGui's backend owns exclusively via callback; read it back from
// ImGuiIO::MouseWheel instead of installing a competing scroll callback.
//
// Left-drag orbits, right-drag pans, scroll zooms, Q/E rolls. Skips
// everything when ImGui wants the input (dragging over the debug panel
// shouldn't also spin the camera).
void updateOrbitCamera(GLFWwindow * window, OrbitCamera & cam, float deltaTime);

} // namespace sabercat::app
