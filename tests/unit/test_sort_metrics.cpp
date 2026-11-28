// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Sort-study metrics (examples/sort_study/sort_metrics.h): image error, popping
// between frames, near-tie swaps in the default sort, and percentiles. Inputs are
// chosen so each expectation is an exact hand computation.
#include "unit_test.h"

#include "sort_metrics.h"

#include <glm/glm.hpp>

#include <cmath>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sabercat::unit
{

namespace
{

namespace study = sabercat::study;

using Image = std::vector<glm::vec4>;

// A 2x2 image (four pixels) with the same color everywhere.
Image flat(float r, float g, float b)
{
    return Image(4, glm::vec4(r, g, b, 1.0f));
}

// The flat image with pixel 0 changed to `(r, g, b)`.
Image withPixelZero(Image image, float r, float g, float b)
{
    image[0] = glm::vec4(r, g, b, 1.0f);
    return image;
}

bool throwsInvalidArgument(const std::function<void()> & fn)
{
    try
    {
        fn();
    }
    catch (const std::invalid_argument &)
    {
        return true;
    }
    return false;
}

void testImageError(Context & ctx)
{
    const Image a = flat(0.2f, 0.4f, 0.6f);

    // Identical images: zero mean error, and infinite PSNR.
    UNIT_CHECK_NEAR(ctx, study::meanAbsoluteError(a, a), 0.0, 0.0);
    const double same = study::psnrDb(a, a);
    UNIT_CHECK(ctx, std::isinf(same) && same > 0.0);

    // Constant difference. The float inputs cannot hold 0.1 exactly, so the
    // expectation uses the exact double difference of the stored values. The
    // 0.1 and 20 dB checks use the looser tolerance that float storage allows.
    const Image ref = flat(0.25f, 0.25f, 0.25f);
    const Image test = flat(0.35f, 0.35f, 0.35f);
    const double diff = double(0.35f) - double(0.25f);
    UNIT_CHECK_NEAR(ctx, study::meanAbsoluteError(ref, test), diff, 1e-12);
    UNIT_CHECK_NEAR(ctx, study::meanAbsoluteError(ref, test), 0.1, 1e-6);
    UNIT_CHECK_NEAR(ctx, study::psnrDb(ref, test), -20.0 * std::log10(diff), 1e-9);
    UNIT_CHECK_NEAR(ctx, study::psnrDb(ref, test), 20.0, 1e-5);

    // Mismatched sizes are rejected.
    UNIT_CHECK(ctx, throwsInvalidArgument([&] { (void)study::meanAbsoluteError(a, Image(3)); }));
}

void testPopping(Context & ctx)
{
    // One frame: the reference holds still while the mode moves pixel 0 by 0.5 in
    // every channel. The per-frame score is 1.5 / (4 pixels * 3 channels) = 0.125,
    // which is 0.5 / pixelCount, and exactly one pixel pops.
    const Image gray = flat(0.25f, 0.25f, 0.25f);
    const Image grayMoved = withPixelZero(gray, 0.75f, 0.75f, 0.75f);
    const Image reference = flat(0.5f, 0.5f, 0.5f);

    study::PoppingAccumulator flip;
    flip.addFrame(gray, grayMoved, reference, reference);
    UNIT_CHECK(ctx, flip.frames() == 1);
    UNIT_CHECK_NEAR(ctx, flip.meanScore(), 0.125, 1e-12);
    UNIT_CHECK_NEAR(ctx, flip.meanPopPixels(), 1.0, 0.0);

    // Identical motion: the reference moves by the same 0.5 as the mode, so nothing
    // is unexplained. Both the score and the pop count are zero.
    const Image referenceMoved = withPixelZero(reference, 1.0f, 1.0f, 1.0f);
    study::PoppingAccumulator matched;
    matched.addFrame(gray, grayMoved, reference, referenceMoved);
    UNIT_CHECK_NEAR(ctx, matched.meanScore(), 0.0, 1e-12);
    UNIT_CHECK_NEAR(ctx, matched.meanPopPixels(), 0.0, 0.0);

    // Two frames together: the mean score is (0.125 + 0) / 2 and the mean pop
    // count is (1 + 0) / 2.
    study::PoppingAccumulator both;
    both.addFrame(gray, grayMoved, reference, reference);
    both.addFrame(gray, grayMoved, reference, referenceMoved);
    UNIT_CHECK(ctx, both.frames() == 2);
    UNIT_CHECK_NEAR(ctx, both.meanScore(), 0.0625, 1e-12);
    UNIT_CHECK_NEAR(ctx, both.meanPopPixels(), 0.5, 1e-12);

    // An empty accumulator reports zeros.
    const study::PoppingAccumulator none;
    UNIT_CHECK(ctx, none.frames() == 0);
    UNIT_CHECK_NEAR(ctx, none.meanScore(), 0.0, 0.0);
    UNIT_CHECK_NEAR(ctx, none.meanPopPixels(), 0.0, 0.0);

    // Mismatched image sizes are rejected.
    study::PoppingAccumulator bad;
    UNIT_CHECK(ctx, throwsInvalidArgument([&] { bad.addFrame(gray, grayMoved, reference, Image(3)); }));
}

// One tile, two Gaussians (A = 0, B = 1), with the given depths in the current frame.
study::SortOrderFrame oneTile(std::vector<uint32_t> order, float depthA, float depthB)
{
    study::SortOrderFrame f;
    f.sortedValues = std::move(order);
    f.tileRanges = {TileRange {0u, 2u}};
    f.depth = {depthA, depthB};
    return f;
}

void testNearTieSwaps(Context & ctx)
{
    // Depths 1.0 and 1.0001 are within the 1e-3 relative tolerance. The previous
    // frame had (A, B) and the current frame has (B, A), so one swap is counted.
    const study::SortOrderFrame prevAB = oneTile({0u, 1u}, 1.0f, 1.0001f);
    const study::SortOrderFrame curBA = oneTile({1u, 0u}, 1.0f, 1.0001f);
    const study::SortOrderFrame curAB = oneTile({0u, 1u}, 1.0f, 1.0001f);
    UNIT_CHECK(ctx, study::countNearTieSwaps(prevAB, curBA) == 1);

    // The same order in both frames is not a swap.
    UNIT_CHECK(ctx, study::countNearTieSwaps(prevAB, curAB) == 0);

    // Depths 10% apart are outside the tolerance, so the swap is not a near tie.
    const study::SortOrderFrame farBA = oneTile({1u, 0u}, 1.0f, 1.1f);
    UNIT_CHECK(ctx, study::countNearTieSwaps(prevAB, farBA) == 0);

    // A pair must share a tile in both frames. Here A and B sit in different tiles
    // in each frame, so the swap is not counted.
    study::SortOrderFrame split;
    split.sortedValues = {0u, 1u};
    split.tileRanges = {TileRange {0u, 1u}, TileRange {1u, 2u}};
    split.depth = {1.0f, 1.0001f};
    study::SortOrderFrame splitSwapped = split;
    splitSwapped.sortedValues = {1u, 0u};
    UNIT_CHECK(ctx, study::countNearTieSwaps(split, splitSwapped) == 0);

    // Tile counts that differ are rejected.
    study::SortOrderFrame twoTiles = curBA;
    twoTiles.tileRanges.push_back(TileRange {2u, 2u});
    UNIT_CHECK(ctx, throwsInvalidArgument([&] { (void)study::countNearTieSwaps(prevAB, twoTiles); }));
}

void testPercentile(Context & ctx)
{
    const std::vector<double> values = {5.0, 1.0, 3.0, 2.0, 4.0};
    UNIT_CHECK_NEAR(ctx, study::percentile(values, 0.5), 3.0, 0.0);
    UNIT_CHECK_NEAR(ctx, study::percentile(values, 0.95), 5.0, 0.0);
    UNIT_CHECK_NEAR(ctx, study::percentile(values, 0.0), 1.0, 0.0);
    UNIT_CHECK_NEAR(ctx, study::percentile({}, 0.5), 0.0, 0.0);
}

} // namespace

void testSortMetrics(Context & ctx)
{
    testImageError(ctx);
    testPopping(ctx);
    testNearTieSwaps(ctx);
    testPercentile(ctx);
}

} // namespace sabercat::unit
