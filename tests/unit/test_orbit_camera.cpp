// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Orbit camera math (apps/common/orbit_camera.h): no window, no input.
#include "unit_test.h"

#include "orbit_camera.h"
#include "sabercat/camera.h"
#include "sabercat/scene_bounds.h"

#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cmath>
#include <random>

namespace sabercat::unit
{

namespace
{

using app::OrbitCamera;

bool finite(const glm::vec3 & v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

// The camera's basis must stay orthonormal however it got there. Each vector
// comes from the quaternion, so its length error is a few float ulps.
void checkBasis(Context & ctx, const OrbitCamera & cam)
{
    UNIT_CHECK(ctx, finite(cam.forward()) && finite(cam.right()) && finite(cam.up()));
    UNIT_CHECK_NEAR(ctx, glm::length(cam.forward()), 1.0f, 1e-5);
    UNIT_CHECK_NEAR(ctx, glm::length(cam.right()), 1.0f, 1e-5);
    UNIT_CHECK_NEAR(ctx, glm::length(cam.up()), 1.0f, 1e-5);
    UNIT_CHECK_NEAR(ctx, glm::dot(cam.right(), cam.up()), 0.0f, 1e-5);
    UNIT_CHECK_NEAR(ctx, glm::dot(cam.forward(), cam.right()), 0.0f, 1e-5);
}

// 100000 random deltas: applyDelta renormalises after every step, so the
// quaternion stays unit length to a few ulps instead of drifting with the count.
void testNormalisationAfterManyDeltas(Context & ctx)
{
    OrbitCamera cam;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> angle(-0.2f, 0.2f);
    for (int i = 0; i < 100000; ++i)
    {
        cam.applyDelta(angle(rng), angle(rng), angle(rng));
    }
    UNIT_CHECK_NEAR(ctx, glm::length(cam.orientation), 1.0f, 1e-6);
    checkBasis(ctx, cam);
}

// Straight up, straight down and a full turn through both poles: the
// quaternion has no pitch clamp to hit, so nothing goes NaN or degenerates.
void testPoleStability(Context & ctx)
{
    OrbitCamera cam;
    cam.pivot = glm::vec3(1.0f, 2.0f, 3.0f);
    cam.distance = 4.0f;

    cam.applyDelta(0.0f, glm::half_pi<float>(), 0.0f);
    UNIT_CHECK_NEAR(ctx, std::abs(cam.forward().y), 1.0f, 1e-5); // looking straight up or down
    checkBasis(ctx, cam);

    for (int i = 0; i < 360; ++i)
    {
        cam.applyDelta(0.0f, glm::radians(1.0f), 0.0f);
        checkBasis(ctx, cam);
    }
    const sabercat::Camera view = cam.toCamera(1280, 720);
    const glm::mat4 & m = view.cameraToWorld;
    bool finiteMatrix = true;
    for (int c = 0; c < 4; ++c)
    {
        for (int r = 0; r < 4; ++r)
        {
            finiteMatrix = finiteMatrix && std::isfinite(m[c][r]);
        }
    }
    UNIT_CHECK(ctx, finiteMatrix);
}

// The pivot projects to the image centre, whatever the orientation and distance.
// Tolerance: the pivot is taken through one float inverse and one matrix product,
// each costing a few ulps of the largest magnitude involved (pivot plus distance).
void testPivotMapsToCentre(Context & ctx)
{
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    std::uniform_real_distribution<float> distanceDist(OrbitCamera::kMinDistance, 100.0f);
    for (int trial = 0; trial < 50; ++trial)
    {
        OrbitCamera cam;
        cam.orientation = glm::normalize(glm::quat(unit(rng), unit(rng), unit(rng), unit(rng)));
        cam.pivot = glm::vec3(unit(rng), unit(rng), unit(rng)) * 10.0f;
        cam.distance = distanceDist(rng);

        const sabercat::Camera view = cam.toCamera(1280, 720);
        const glm::vec4 inCamera = glm::inverse(view.cameraToWorld) * glm::vec4(cam.pivot, 1.0f);
        const double tolerance = 1e-5 * (1.0 + glm::length(cam.pivot) + cam.distance);
        UNIT_CHECK_NEAR(ctx, inCamera.x, 0.0, tolerance);
        UNIT_CHECK_NEAR(ctx, inCamera.y, 0.0, tolerance);
        UNIT_CHECK_NEAR(ctx, inCamera.z, -cam.distance, tolerance);
    }

    // The image centre and focal length the Camera carries, for a 1280x720 image.
    const sabercat::Camera view = OrbitCamera{}.toCamera(1280, 720);
    UNIT_CHECK_NEAR(ctx, view.cx, 640.0f, 0.0);
    UNIT_CHECK_NEAR(ctx, view.cy, 360.0f, 0.0);
    UNIT_CHECK_NEAR(ctx, view.fx, 720.0f, 0.0);
    UNIT_CHECK_NEAR(ctx, view.fy, 720.0f, 0.0);
}

// frame() puts the whole bounding sphere inside the vertical field of view. The
// sphere at distance d fits when r <= d * sin(halfFov). frame() takes d = 1.15 r /
// tan(halfFov), which fits whenever cos(halfFov) >= 1/1.15, i.e. halfFov <= 29.6
// degrees. toCamera() uses fy = height, a half-FOV of atan(0.5) = 26.6 degrees,
// so the worst case here is r / (d sin) = tan / (1.15 sin) = 0.97.
void testFrameEnclosesSphere(Context & ctx)
{
    const uint32_t height = 720;
    const float radii[] = {1e-3f, 0.5f, 3.0f, 250.0f};
    const glm::vec3 centres[] = {glm::vec3(0.0f), glm::vec3(-4.0f, 2.0f, 9.0f)};
    double worstRatio = 0.0;
    for (const float radius : radii)
    {
        for (const glm::vec3 & centre : centres)
        {
            SceneBounds bounds;
            bounds.valid = true;
            bounds.focusCenter = centre;
            bounds.focusRadius = radius;
            bounds.min = centre - glm::vec3(radius);
            bounds.max = centre + glm::vec3(radius);

            OrbitCamera cam;
            cam.orientation = glm::angleAxis(0.6f, glm::normalize(glm::vec3(0.2f, 1.0f, -0.4f)));
            const glm::quat orientationBefore = cam.orientation;
            cam.frame(bounds, height, static_cast<float>(height));

            UNIT_CHECK(ctx, cam.pivot == centre);
            UNIT_CHECK(ctx, cam.orientation == orientationBefore); // frame() answers "where", not "which way"
            UNIT_CHECK(ctx, cam.distance >= OrbitCamera::kMinDistance);
            const double tanHalfFov = 0.5; // height / (2 fy), with fy = height
            const double sinHalfFov = tanHalfFov / std::sqrt(1.0 + tanHalfFov * tanHalfFov);
            const double ratio = radius / (cam.distance * sinHalfFov);
            worstRatio = std::max(worstRatio, ratio);
            UNIT_CHECK(ctx, ratio <= 1.0);
        }
    }
    std::cout << "  frame: worst r / (d sin(halfFov)) = " << worstRatio << " (must be <= 1)" << std::endl;
}

// Invalid bounds leave the camera exactly as it was.
void testFrameIgnoresInvalidBounds(Context & ctx)
{
    OrbitCamera cam;
    cam.pivot = glm::vec3(1.0f, -2.0f, 3.0f);
    cam.distance = 7.0f;
    SceneBounds bounds;
    bounds.valid = false;
    bounds.focusCenter = glm::vec3(100.0f);
    bounds.focusRadius = 50.0f;
    cam.frame(bounds, 720, 720.0f);
    UNIT_CHECK(ctx, cam.pivot == glm::vec3(1.0f, -2.0f, 3.0f));
    UNIT_CHECK_NEAR(ctx, cam.distance, 7.0f, 0.0);
}

} // namespace

void testOrbitCamera(Context & ctx)
{
    testNormalisationAfterManyDeltas(ctx);
    testPoleStability(ctx);
    testPivotMapsToCentre(ctx);
    testFrameEnclosesSphere(ctx);
    testFrameIgnoresInvalidBounds(ctx);
}

} // namespace sabercat::unit
