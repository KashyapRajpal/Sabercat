// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The GPU-free edit math (sabercat/edit_math.h): the layer-space form of a world edit,
// in double and after the float cast; the identity, translation and pivot cases;
// rectangle normalization; the inclusive world-space predicates at their boundaries;
// the forward camera's projection; and inverses. The inputs come from a fixed
// xorshift32 stream, so every run checks the same cases.
#include "unit_test.h"

#include "sabercat/camera.h"
#include "sabercat/edit_math.h"
#include "sabercat/layer_transform.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>

namespace sabercat::unit
{

namespace
{

// xorshift32: the same sequence on every run.
class Rng
{
  public:
    explicit Rng(uint32_t seed) : m_state(seed) {}

    // Uniform in [lo, hi).
    double uniform(double lo, double hi)
    {
        m_state ^= m_state << 13;
        m_state ^= m_state >> 17;
        m_state ^= m_state << 5;
        return lo + (hi - lo) * (static_cast<double>(m_state) / 4294967296.0);
    }

    glm::dvec3 point(double lo, double hi)
    {
        const double x = uniform(lo, hi);
        const double y = uniform(lo, hi);
        const double z = uniform(lo, hi);
        return glm::dvec3(x, y, z);
    }

    // A unit quaternion, as (w, x, y, z): a normalized sample of the 4-cube.
    glm::dquat unitQuaternion()
    {
        for (;;)
        {
            const double w = uniform(-1.0, 1.0);
            const double x = uniform(-1.0, 1.0);
            const double y = uniform(-1.0, 1.0);
            const double z = uniform(-1.0, 1.0);
            const double length = std::sqrt(w * w + x * x + y * y + z * z);
            if (length >= 0.1)
            {
                return glm::dquat(w / length, x / length, y / length, z / length);
            }
        }
    }

  private:
    uint32_t m_state;
};

// The largest absolute component difference of two points.
double maxAbsDiff(const glm::dvec3 & a, const glm::dvec3 & b)
{
    const glm::dvec3 d = a - b;
    return std::max({std::abs(d.x), std::abs(d.y), std::abs(d.z)});
}

double lengthOf(const glm::dvec3 & v)
{
    return glm::length(v);
}

// A placement in the ranges the Studio uses: unit rotation, translation in [-20, 20],
// scale in [0.1, 5]. Returned as the float LayerTransform the layers hold.
LayerTransform randomPlacement(Rng & rng)
{
    const glm::dquat q = rng.unitQuaternion();
    LayerTransform layer;
    layer.rotation = glm::quat(static_cast<float>(q.w), static_cast<float>(q.x), static_cast<float>(q.y),
                               static_cast<float>(q.z));
    layer.translation = glm::vec3(rng.point(-20.0, 20.0));
    layer.scale = static_cast<float>(rng.uniform(0.1, 5.0));
    return layer;
}

// The forward pass's camera for the projection checks: identity world-to-camera,
// fx = fy = 100, principal point (64, 48), a 128 x 96 image.
CameraUniforms testCamera()
{
    CameraUniforms camera{};
    camera.worldToCamera = glm::mat4(1.0f);
    camera.fx = 100.0f;
    camera.fy = 100.0f;
    camera.cx = 64.0f;
    camera.cy = 48.0f;
    camera.imageWidth = 128;
    camera.imageHeight = 96;
    return camera;
}

// S_l = layerSpaceEdit(L, T) must satisfy L(S_l(x)) = T(L(x)). The double form must do
// so to double precision. The float cast adds error of about 1e-7 times the largest
// intermediate magnitude: s_L s_T |x| <= 20 |x| for the linear part and
// |v| <= 10 (|t_L| + |p| + |t_T|) for the translation, so the bound scales with those.
void testLayerSpaceComposition(Context & ctx)
{
    Rng rng(0x1A2B3C4Du);
    for (int c = 0; c < 500; ++c)
    {
        const LayerTransform layer = randomPlacement(rng);
        const Similarity placed = similarityOf(layer);
        const glm::dvec3 pivot = rng.point(-20.0, 20.0);
        const glm::dmat3 rotation = glm::mat3_cast(rng.unitQuaternion());
        const double editScale = rng.uniform(0.2, 4.0);
        const glm::dvec3 editTranslation = rng.point(-20.0, 20.0);
        const Similarity world = worldEditAboutPivot(rotation, editScale, editTranslation, pivot);

        const Similarity doubleForm = layerSpaceEditD(placed, world);
        const Similarity floatForm = similarityOf(layerSpaceEdit(layer, world));
        const double placementMagnitude
            = lengthOf(placed.translation) + lengthOf(pivot) + lengthOf(editTranslation);

        for (int k = 0; k < 50; ++k)
        {
            const glm::dvec3 x = rng.point(-20.0, 20.0);
            const glm::dvec3 reference = compose(world, placed).apply(x);
            const double magnitude = 1.0 + 20.0 * lengthOf(x) + 10.0 * placementMagnitude;

            UNIT_CHECK_NEAR(ctx, maxAbsDiff(compose(placed, doubleForm).apply(x), reference), 0.0,
                            1e-12 * magnitude);
            UNIT_CHECK_NEAR(ctx, maxAbsDiff(compose(placed, floatForm).apply(x), reference), 0.0,
                            1e-5 * magnitude);
        }
    }
}

// The identity edit gives the identity layer-space edit. A pure translation t about any
// pivot gives the identity rotation, unit scale, and translation (1/s_L) R_L^T t.
void testLayerSpaceSpecialCases(Context & ctx)
{
    Rng rng(0x2B3C4D5Eu);
    const glm::dmat3 identity(1.0);
    for (int c = 0; c < 100; ++c)
    {
        const LayerTransform layer = randomPlacement(rng);
        const Similarity placed = similarityOf(layer);
        const glm::dvec3 pivot = rng.point(-20.0, 20.0);

        const LayerTransform same
            = layerSpaceEdit(layer, worldEditAboutPivot(identity, 1.0, glm::dvec3(0.0), pivot));
        UNIT_CHECK_NEAR(ctx, std::abs(same.rotation.w), 1.0, 1e-7);
        UNIT_CHECK_NEAR(ctx,
                        std::abs(same.rotation.x) + std::abs(same.rotation.y) + std::abs(same.rotation.z),
                        0.0, 1e-7);
        UNIT_CHECK_NEAR(ctx, same.scale, 1.0, 1e-7);
        UNIT_CHECK_NEAR(ctx, maxAbsDiff(glm::dvec3(same.translation), glm::dvec3(0.0)), 0.0, 1e-7);

        const glm::dvec3 shift = rng.point(-20.0, 20.0);
        const LayerTransform moved
            = layerSpaceEdit(layer, worldEditAboutPivot(identity, 1.0, shift, pivot));
        const glm::dvec3 expected = glm::transpose(placed.rotation) * shift / placed.scale;
        UNIT_CHECK_NEAR(ctx, std::abs(moved.rotation.w), 1.0, 1e-6);
        UNIT_CHECK_NEAR(ctx,
                        std::abs(moved.rotation.x) + std::abs(moved.rotation.y) + std::abs(moved.rotation.z),
                        0.0, 1e-6);
        UNIT_CHECK_NEAR(ctx, moved.scale, 1.0, 1e-6);
        UNIT_CHECK_NEAR(ctx, maxAbsDiff(glm::dvec3(moved.translation), expected), 0.0,
                        1e-5 * (1.0 + lengthOf(expected)));
    }
}

// A rotation about a pivot with unit scale and no translation fixes the pivot.
void testPivotFixed(Context & ctx)
{
    Rng rng(0x3C4D5E6Fu);
    for (int c = 0; c < 200; ++c)
    {
        const glm::dmat3 rotation = glm::mat3_cast(rng.unitQuaternion());
        const glm::dvec3 pivot = rng.point(-20.0, 20.0);
        const Similarity edit = worldEditAboutPivot(rotation, 1.0, glm::dvec3(0.0), pivot);
        UNIT_CHECK_NEAR(ctx, maxAbsDiff(edit.apply(pivot), pivot), 0.0, 1e-12 * (1.0 + lengthOf(pivot)));
    }
}

// A drag gives the same rectangle in any direction; a drag from a point to itself is
// degenerate, with x0 == x1 and y0 == y1.
void testRectNormalization(Context & ctx)
{
    const glm::vec2 lowerLeft(10.0f, 5.0f);
    const glm::vec2 upperRight(30.0f, 20.0f);
    const glm::vec2 upperLeft(10.0f, 20.0f);
    const glm::vec2 lowerRight(30.0f, 5.0f);
    const glm::vec4 expected(10.0f, 5.0f, 30.0f, 20.0f);

    UNIT_CHECK(ctx, normalizeRect(lowerLeft, upperRight) == expected);
    UNIT_CHECK(ctx, normalizeRect(upperRight, lowerLeft) == expected);
    UNIT_CHECK(ctx, normalizeRect(upperLeft, lowerRight) == expected);
    UNIT_CHECK(ctx, normalizeRect(lowerRight, upperLeft) == expected);

    const glm::vec2 same(7.0f, 3.0f);
    const glm::vec4 degenerate = normalizeRect(same, same);
    UNIT_CHECK(ctx, degenerate == glm::vec4(7.0f, 3.0f, 7.0f, 3.0f));
    UNIT_CHECK(ctx, degenerate.x == degenerate.z && degenerate.y == degenerate.w);
}

// The box and sphere predicates at their boundaries. Points exactly on a face or the
// surface are inside; 1e-9 beyond are outside; 1e-9 within are inside.
void testPredicateBoundaries(Context & ctx)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const glm::dvec3 center(1.0, 2.0, 3.0);
    const glm::dvec3 halfExtents(2.0, 3.0, 4.0);

    // The axis-aligned box's faces are exact in double, so the on-face points are tested
    // as they are. A box rotated 30 degrees about z has faces that rounding cannot place
    // exactly, so its on-face points are nudged 1e-12 inside.
    const double angle = std::acos(-1.0) / 6.0;
    const glm::dmat3 rotated(glm::dvec3(std::cos(angle), std::sin(angle), 0.0),
                             glm::dvec3(-std::sin(angle), std::cos(angle), 0.0),
                             glm::dvec3(0.0, 0.0, 1.0));
    const auto checkFaces = [&](const glm::dmat3 & axes, double onFaceNudge)
    {
        UNIT_CHECK(ctx, boxContains(center, center, halfExtents, axes));
        for (int axis = 0; axis < 3; ++axis)
        {
            for (const double sign : {-1.0, 1.0})
            {
                const glm::dvec3 normal = sign * axes[axis];
                const glm::dvec3 face = center + halfExtents[axis] * normal;
                UNIT_CHECK(ctx, boxContains(face - onFaceNudge * normal, center, halfExtents, axes));
                UNIT_CHECK(ctx, boxContains(face - 1e-9 * normal, center, halfExtents, axes));
                UNIT_CHECK(ctx, !boxContains(face + 1e-9 * normal, center, halfExtents, axes));
            }
        }
        UNIT_CHECK(ctx, !boxContains(glm::dvec3(nan, 2.0, 3.0), center, halfExtents, axes));
    };
    checkFaces(glm::dmat3(1.0), 0.0);
    checkFaces(rotated, 1e-12);

    const glm::dvec3 sphereCenter(1.0, -2.0, 0.5);
    const double radius = 2.0;
    for (int axis = 0; axis < 3; ++axis)
    {
        for (const double sign : {-1.0, 1.0})
        {
            glm::dvec3 unit(0.0);
            unit[axis] = sign;
            const glm::dvec3 surface = sphereCenter + radius * unit;
            UNIT_CHECK(ctx, sphereContains(surface, sphereCenter, radius));
            UNIT_CHECK(ctx, sphereContains(surface - 1e-9 * unit, sphereCenter, radius));
            UNIT_CHECK(ctx, !sphereContains(surface + 1e-9 * unit, sphereCenter, radius));
        }
    }
    UNIT_CHECK(ctx, sphereContains(sphereCenter, sphereCenter, 0.0));
    UNIT_CHECK(ctx, !sphereContains(sphereCenter, sphereCenter, -1.0));
    UNIT_CHECK(ctx, !sphereContains(glm::dvec3(nan, 0.0, 0.0), sphereCenter, radius));
}

// The projection in double, as the forward pass does it, and the rectangle test on it.
void testProjection(Context & ctx)
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const CameraUniforms camera = testCamera();
    glm::dvec2 pixel(0.0);
    double depth = 0.0;

    // The optical axis at depth 5 projects to the principal point.
    UNIT_CHECK(ctx, projectToPixels(glm::dvec3(0.0, 0.0, -5.0), camera, pixel, depth));
    UNIT_CHECK_NEAR(ctx, pixel.x, 64.0, 1e-12);
    UNIT_CHECK_NEAR(ctx, pixel.y, 48.0, 1e-12);
    UNIT_CHECK_NEAR(ctx, depth, 5.0, 1e-12);

    // (x, y, -5) lands at (64 + 100 x / 5, 48 - 100 y / 5): world +y is up, which is
    // smaller v.
    UNIT_CHECK(ctx, projectToPixels(glm::dvec3(0.5, -0.25, -5.0), camera, pixel, depth));
    UNIT_CHECK_NEAR(ctx, pixel.x, 64.0 + 100.0 * 0.5 / 5.0, 1e-12);
    UNIT_CHECK_NEAR(ctx, pixel.y, 48.0 - 100.0 * (-0.25) / 5.0, 1e-12);
    UNIT_CHECK_NEAR(ctx, depth, 5.0, 1e-12);

    // The depth must exceed kMinDepth (0.2). z = -0.2 gives depth 0.2 in double, which is
    // below the float bound; z = 0 and z = +1 are not in front of the camera.
    UNIT_CHECK(ctx, !projectToPixels(glm::dvec3(0.0, 0.0, -0.2), camera, pixel, depth));
    UNIT_CHECK(ctx, !projectToPixels(glm::dvec3(0.0, 0.0, 0.0), camera, pixel, depth));
    UNIT_CHECK(ctx, !projectToPixels(glm::dvec3(0.0, 0.0, 1.0), camera, pixel, depth));
    UNIT_CHECK(ctx, projectToPixels(glm::dvec3(0.0, 0.0, -0.3), camera, pixel, depth));
    UNIT_CHECK(ctx, !projectToPixels(glm::dvec3(nan, 0.0, -5.0), camera, pixel, depth));
    UNIT_CHECK(ctx, !projectToPixels(glm::dvec3(0.0, 0.0, nan), camera, pixel, depth));

    // The rectangle (44, 48) to (84, 68), edges included: (-1, 0, -5) projects to (44, 48)
    // and (1, -1, -5) to (84, 68).
    const glm::vec4 rect(44.0f, 48.0f, 84.0f, 68.0f);
    UNIT_CHECK(ctx, rectContains(glm::dvec3(-1.0, 0.0, -5.0), camera, rect));
    UNIT_CHECK(ctx, rectContains(glm::dvec3(1.0, -1.0, -5.0), camera, rect));
    UNIT_CHECK(ctx, rectContains(glm::dvec3(0.0, 0.0, -5.0), camera, rect));
    UNIT_CHECK(ctx, !rectContains(glm::dvec3(-1.0 - 1e-9, 0.0, -5.0), camera, rect));
    UNIT_CHECK(ctx, !rectContains(glm::dvec3(1.0 + 1e-9, -1.0, -5.0), camera, rect));
    UNIT_CHECK(ctx, !rectContains(glm::dvec3(0.0, 1e-9, -5.0), camera, rect));
    UNIT_CHECK(ctx, !rectContains(glm::dvec3(1.0, -1.0 - 1e-9, -5.0), camera, rect));

    // Depth alone decides at (64, 48): the whole image contains that pixel.
    const glm::vec4 image(0.0f, 0.0f, 128.0f, 96.0f);
    UNIT_CHECK(ctx, !rectContains(glm::dvec3(0.0, 0.0, -0.2), camera, image));
    UNIT_CHECK(ctx, rectContains(glm::dvec3(0.0, 0.0, -0.3), camera, image));
    UNIT_CHECK(ctx, !rectContains(glm::dvec3(nan, 0.0, -5.0), camera, image));
}

// inverse(s) undoes s on both sides, and a non-positive scale has no inverse.
void testInverse(Context & ctx)
{
    Rng rng(0x4D5E6F70u);
    for (int c = 0; c < 200; ++c)
    {
        Similarity s;
        s.rotation = glm::mat3_cast(rng.unitQuaternion());
        s.translation = rng.point(-20.0, 20.0);
        s.scale = rng.uniform(0.5, 4.0);
        const Similarity inv = inverse(s);
        for (int k = 0; k < 20; ++k)
        {
            const glm::dvec3 x = rng.point(-1.0, 1.0);
            UNIT_CHECK_NEAR(ctx, maxAbsDiff(compose(inv, s).apply(x), x), 0.0, 1e-12);
            UNIT_CHECK_NEAR(ctx, maxAbsDiff(compose(s, inv).apply(x), x), 0.0, 1e-12);
        }
    }

    Similarity flat;
    flat.scale = 0.0;
    bool threw = false;
    try
    {
        (void)inverse(flat);
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    UNIT_CHECK(ctx, threw);
}

} // namespace

void testEditMath(Context & ctx)
{
    testLayerSpaceComposition(ctx);
    testLayerSpaceSpecialCases(ctx);
    testPivotFixed(ctx);
    testRectNormalization(ctx);
    testPredicateBoundaries(ctx);
    testProjection(ctx);
    testInverse(ctx);
}

} // namespace sabercat::unit
