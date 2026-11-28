// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// CPU reference renderer for the sort study (examples/sort_study/cpu_reference.h):
// the ray-depth evaluation and the per-pixel orders, on a hand-built scene whose
// pixel values are exact binary fractions, so each expectation is a hand sum.
#include "unit_test.h"

#include "cpu_reference.h"

#include <glm/glm.hpp>

#include <cstring>
#include <stdexcept>
#include <vector>

namespace sabercat::unit
{

namespace
{

namespace study = sabercat::study;

// Two 16x16 tiles side by side (32x16), so tile 1 can be empty. Three Gaussians
// are centered on pixel (8, 8) with conic (0.5, 0, 0.5), so each reaches that
// pixel at full opacity. Their opacities are 0.5 (red), 0.25 (green) and 0.5
// (blue), and their center depths are 3, 1 and 2, in that list order. With no
// ray-depth form, the depth order is the center-depth order.
study::StudyCandidates buildScene()
{
    study::StudyCandidates c;
    c.width = 32;
    c.height = 16;
    c.tilesX = 2;
    c.tilesY = 1;
    c.fx = 16.0f;
    c.fy = 16.0f;
    c.cx = 8.0f;
    c.cy = 8.0f;

    const glm::vec3 colors[3] = {glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f),
                                 glm::vec3(0.0f, 0.0f, 1.0f)};
    const float opacities[3] = {0.5f, 0.25f, 0.5f};
    const float depths[3] = {3.0f, 1.0f, 2.0f};
    for (int i = 0; i < 3; ++i)
    {
        ProjectedGaussian g {};
        g.center = glm::vec2(8.0f, 8.0f);
        g.depth = depths[i];
        g.radius = 4.0f;
        g.conic = glm::vec3(0.5f, 0.0f, 0.5f);
        g.alpha = opacities[i];
        g.color = colors[i];
        g.tileCount = 1;
        c.projected.push_back(g);
    }
    c.sortedValues = {0u, 1u, 2u};
    c.tileRanges = {TileRange {0u, 3u}, TileRange {3u, 3u}};
    return c; // rayForm stays empty: every Gaussian uses its center depth
}

// Pixel (8, 8) in the list order, by hand: red 0.5 blends first, then green 0.25,
// then blue 0.5. T goes 1 -> 0.5 -> 0.375 -> 0.1875.
//   acc = (0.5, 0, 0) + (0, 0.125, 0) + (0, 0, 0.1875)
const glm::vec4 kTileOrderAtCenter(0.5f, 0.125f, 0.1875f, 0.8125f);

// Pixel (8, 8) in ascending center depth: green (1), blue (2), red (3).
//   acc = (0, 0.25, 0) + (0, 0, 0.375) + (0.1875, 0, 0), T -> 0.75 -> 0.375 -> 0.1875
const glm::vec4 kExactAtCenter(0.1875f, 0.25f, 0.375f, 0.8125f);

// Bitwise equality of two images, so -0 and +0 or a single ulp count as different.
bool sameBits(const std::vector<glm::vec4> & a, const std::vector<glm::vec4> & b)
{
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(glm::vec4)) == 0;
}

bool sameVec(const glm::vec4 & a, const glm::vec4 & b)
{
    return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

std::vector<glm::vec4> render(const study::StudyCandidates & c, study::ReferenceOrder order, uint32_t window = 16)
{
    study::ReferenceOptions o;
    o.order = order;
    o.window = window;
    return study::renderReference(c, o);
}

// evalRayDepth: the Gaussian at camera (0, 0, -5) with identity inverse covariance.
void testEvalRayDepth(Context & ctx)
{
    // Whitened form: P = I (columns c0, c1, c2) and u = P * mu_cam = (0, 0, -5).
    const float form[12] = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, -5.0f};

    // The ray straight down the axis, d = (0, 0, -1): t* = a.d = 5.
    UNIT_CHECK_NEAR(ctx, study::evalRayDepth(form, 99.0, 0.0, 0.0), 5.0, 1e-12);

    // d = (0.1, 0, -1): num = a.d = 5, den = 0.01 + 1 = 1.01.
    UNIT_CHECK_NEAR(ctx, study::evalRayDepth(form, 99.0, 0.1, 0.0), 5.0 / 1.01, 1e-12);

    // An all-zero form is degenerate (den = 0), so the fallback is returned.
    const float zero[12] = {};
    UNIT_CHECK(ctx, study::evalRayDepth(zero, 4.25, 0.1, 0.2) == 4.25);

    // A null form means "no form": the fallback is returned.
    UNIT_CHECK(ctx, study::evalRayDepth(nullptr, 4.25, 0.1, 0.2) == 4.25);
}

// renderReference on the three-Gaussian scene: orders, window bounds and the
// empty tile.
void testRenderReference(Context & ctx)
{
    const study::StudyCandidates c = buildScene();

    const std::vector<glm::vec4> tile = render(c, study::ReferenceOrder::TileOrder);
    const std::vector<glm::vec4> exact = render(c, study::ReferenceOrder::ExactRayDepth);
    const std::vector<glm::vec4> window1 = render(c, study::ReferenceOrder::WindowRayDepth, 1);
    const std::vector<glm::vec4> window2 = render(c, study::ReferenceOrder::WindowRayDepth, 2);
    const std::vector<glm::vec4> window16 = render(c, study::ReferenceOrder::WindowRayDepth, 16);

    UNIT_CHECK(ctx, tile.size() == 32 * 16);

    // Hand-computed values at the center pixel, in both orders.
    const size_t center = 8 * 32 + 8;
    UNIT_CHECK(ctx, sameVec(tile[center], kTileOrderAtCenter));
    UNIT_CHECK(ctx, sameVec(exact[center], kExactAtCenter));

    // The two orders differ at the center pixel, so the test can tell them apart.
    UNIT_CHECK(ctx, !sameVec(tile[center], exact[center]));

    // K = 1 is the list order, bit for bit.
    UNIT_CHECK(ctx, sameBits(window1, tile));

    // K = 2 holds the two smallest depths at the center pixel, so it matches the exact order there.
    UNIT_CHECK(ctx, sameVec(window2[center], kExactAtCenter));

    // K >= the contributor count is the exact order, bit for bit.
    UNIT_CHECK(ctx, sameBits(window16, exact));

    // The second tile has an empty range, so every pixel in it is zero in every order.
    bool emptyTileZero = true;
    for (uint32_t y = 0; y < 16; ++y)
    {
        for (uint32_t x = 16; x < 32; ++x)
        {
            const size_t idx = size_t(y) * 32 + x;
            emptyTileZero = emptyTileZero && sameVec(tile[idx], glm::vec4(0.0f)) && sameVec(exact[idx], glm::vec4(0.0f))
                            && sameVec(window16[idx], glm::vec4(0.0f));
        }
    }
    UNIT_CHECK(ctx, emptyTileZero);

    // A pixel no Gaussian reaches stays (0, 0, 0, 0). Pixel (0, 0) is far from the
    // Gaussians, so its alpha falls below 1/255 and it does not contribute.
    UNIT_CHECK(ctx, sameVec(tile[0], glm::vec4(0.0f)));
}

// renderReference rejects inconsistent candidates instead of indexing past them.
void testRenderReferenceValidation(Context & ctx)
{
    study::StudyCandidates bad = buildScene();
    bad.tileRanges.pop_back(); // one range for two tiles

    bool threw = false;
    try
    {
        (void)render(bad, study::ReferenceOrder::TileOrder);
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    UNIT_CHECK(ctx, threw);

    study::StudyCandidates zeroWindow = buildScene();
    threw = false;
    try
    {
        (void)render(zeroWindow, study::ReferenceOrder::WindowRayDepth, 0);
    }
    catch (const std::invalid_argument &)
    {
        threw = true;
    }
    UNIT_CHECK(ctx, threw);
}

} // namespace

void testSortReference(Context & ctx)
{
    testEvalRayDepth(ctx);
    testRenderReference(ctx);
    testRenderReferenceValidation(ctx);
}

} // namespace sabercat::unit
