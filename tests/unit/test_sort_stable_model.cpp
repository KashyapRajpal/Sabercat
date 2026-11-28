// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Stable-mode key model (examples/sort_study/stable_key_model.h): the first step,
// the hysteresis edges, the empty-tile sentinel, the coplanar pair and the
// ambiguity flags. Depths are built from chosen bucket coordinates, so each
// expectation is an exact hand computation.
#include "unit_test.h"

#include "stable_key_model.h"

#include <cmath>
#include <cstdint>

namespace sabercat::unit
{

namespace
{

namespace study = sabercat::study;

constexpr float kBuckets = 1024.0f;
constexpr float kHysteresis = 1.0f;

// The depth whose bucket coordinate is x: inverts x = (log2(depth) + 64) * B.
float depthAtBucket(float x, float bucketsPerOctave)
{
    return std::exp2(x / bucketsPerOctave - 64.0f);
}

void testFirstStep(Context & ctx)
{
    // From the sentinel the key is floor(x). Depth 1 gives x = 64 * 1024 = 65536.
    // Depth 2 adds log2(2) = 1 octave, so x = 65 * 1024 = 66560.
    UNIT_CHECK(ctx, study::stableKeyStep(study::kStableSentinel, 1.0f, 1, kBuckets, kHysteresis) == 65536u);
    UNIT_CHECK(ctx, study::stableKeyStep(study::kStableSentinel, 2.0f, 1, kBuckets, kHysteresis) == 66560u);
}

void testHysteresis(Context & ctx)
{
    // Depth 5 gives x = 67913.66, so the first key is its floor, 67913.
    const uint32_t stored = study::stableKeyStep(study::kStableSentinel, 5.0f, 1, kBuckets, kHysteresis);
    UNIT_CHECK(ctx, stored == 67913u);

    // A +1e-6 relative change moves x by about 0.0015 buckets, so the key is kept.
    UNIT_CHECK(ctx, study::stableKeyStep(stored, 5.0f * (1.0f + 1e-6f), 1, kBuckets, kHysteresis) == stored);

    // A 5% change moves x by log2(1.05) * 1024, about 72 buckets. The key moves to
    // the floor of the new coordinate, 67985 at depth 5.25.
    UNIT_CHECK(ctx, study::stableKeyStep(stored, 5.25f, 1, kBuckets, kHysteresis) == 67985u);

    // Edges with the stored key fixed at 67913 and h = 1. The keep band is
    // |x - (stored + 0.5)| <= 0.5 + h, so the edges are x = stored - h and
    // x = stored + 1 + h. Each probe sits 0.25 buckets on one side of an edge,
    // far more than the float error of the depth round trip.
    const float s = static_cast<float>(stored);
    UNIT_CHECK(ctx, study::stableKeyStep(stored, depthAtBucket(s + 1.0f + kHysteresis - 0.25f, kBuckets), 1, kBuckets, kHysteresis) == stored);
    UNIT_CHECK(ctx, study::stableKeyStep(stored, depthAtBucket(s + 1.0f + kHysteresis + 0.25f, kBuckets), 1, kBuckets, kHysteresis) == 67915u);
    UNIT_CHECK(ctx, study::stableKeyStep(stored, depthAtBucket(s - kHysteresis + 0.25f, kBuckets), 1, kBuckets, kHysteresis) == stored);
    UNIT_CHECK(ctx, study::stableKeyStep(stored, depthAtBucket(s - kHysteresis - 0.25f, kBuckets), 1, kBuckets, kHysteresis) == 67911u);
}

void testEmptyTileSentinel(Context & ctx)
{
    // A Gaussian with no tile coverage returns the sentinel, whatever it stored and
    // however far its depth has moved.
    UNIT_CHECK(ctx, study::stableKeyStep(12345u, 1.0f, 0, kBuckets, kHysteresis) == study::kStableSentinel);
    UNIT_CHECK(ctx, study::stableKeyStep(study::kStableSentinel, 1.0f, 0, kBuckets, kHysteresis) == study::kStableSentinel);
    UNIT_CHECK(ctx, study::stableKeyStep(67913u, 5.25f, 0, kBuckets, kHysteresis) == study::kStableSentinel);
}

void testCoplanarPair(Context & ctx)
{
    // Two Gaussians at depths 5.0 and 5.0 * (1 + 1e-6). Their coordinates are a
    // few thousandths of a bucket apart and sit well inside bucket 67913, so
    // their keys agree from the first step on.
    const float base = 5.0f;
    const float other = base * (1.0f + 1e-6f);
    UNIT_CHECK(ctx, other != base);

    uint32_t a = study::stableKeyStep(study::kStableSentinel, base, 1, kBuckets, kHysteresis);
    uint32_t b = study::stableKeyStep(study::kStableSentinel, other, 1, kBuckets, kHysteresis);
    UNIT_CHECK(ctx, a == b);
    const uint32_t first = a;

    // Over 100 frames both depths move by alternating +-1e-6 relative. Each key
    // stays at the first value, so the pair keeps its order in the sorted list.
    bool stayed = true;
    for (int frame = 0; frame < 100; ++frame)
    {
        const float sign = (frame % 2 == 0) ? 1.0f : -1.0f;
        a = study::stableKeyStep(a, base * (1.0f + sign * 1e-6f), 1, kBuckets, kHysteresis);
        b = study::stableKeyStep(b, other * (1.0f + sign * 1e-6f), 1, kBuckets, kHysteresis);
        stayed = stayed && a == first && b == first;
    }
    UNIT_CHECK(ctx, stayed);
}

void testAmbiguous(Context & ctx)
{
    // B = 1 keeps x near 64, where the float spacing is about 8e-6. That is far
    // below the 0.02 band, so each probe's flag is exact.
    constexpr float kOneBucket = 1.0f;

    // Floor boundary: x = 64.001 is 0.001 above an integer. Sentinel stored, so only
    // the floor case applies, and it is ambiguous.
    UNIT_CHECK(ctx, study::stableKeyAmbiguous(study::kStableSentinel, depthAtBucket(64.001f, kOneBucket), 1, kOneBucket, kHysteresis));

    // x = 64.5 is half a bucket from any integer, so it is not ambiguous.
    UNIT_CHECK(ctx, !study::stableKeyAmbiguous(study::kStableSentinel, depthAtBucket(64.5f, kOneBucket), 1, kOneBucket, kHysteresis));

    // Hysteresis boundary with h = 0.5 and stored 64: the edge is at x = 64 + 0.5 + 1.0 = 65.5,
    // which is half a bucket from any integer, so only the hysteresis case can fire here.
    // x = 65.51 is 0.01 past the edge: ambiguous. x = 65.75 is 0.25 past it: not.
    UNIT_CHECK(ctx, study::stableKeyAmbiguous(64u, depthAtBucket(65.51f, kOneBucket), 1, kOneBucket, 0.5f));
    UNIT_CHECK(ctx, !study::stableKeyAmbiguous(64u, depthAtBucket(65.75f, kOneBucket), 1, kOneBucket, 0.5f));

    // A kept key is not flagged for the floor. Stored 64, h = 1: x = 65.001 is inside the keep
    // band (distance 0.501, edge 1.5), so the key stays and x's nearness to 65 does not matter.
    UNIT_CHECK(ctx, !study::stableKeyAmbiguous(64u, depthAtBucket(65.001f, kOneBucket), 1, kOneBucket, kHysteresis));

    // An update past the edge is flagged when x is near an integer. Stored 64, h = 0.755:
    // x = stored + 1 + h + 0.25 = 66.005 is 0.25 past the edge (edge 1.255, distance 1.505), so the
    // key is rewritten, and its fraction 0.005 is within 0.01 of the integer 66.
    UNIT_CHECK(ctx, study::stableKeyAmbiguous(64u, depthAtBucket(66.005f, kOneBucket), 1, kOneBucket, 0.755f));
    // The same update with x = 66.25 (fraction 0.25) is not ambiguous.
    UNIT_CHECK(ctx, !study::stableKeyAmbiguous(64u, depthAtBucket(66.25f, kOneBucket), 1, kOneBucket, 0.755f));

    // An empty tile is never ambiguous, even on a boundary.
    UNIT_CHECK(ctx, !study::stableKeyAmbiguous(64u, depthAtBucket(66.001f, kOneBucket), 0, kOneBucket, kHysteresis));
}

} // namespace

void testSortStableModel(Context & ctx)
{
    testFirstStep(ctx);
    testHysteresis(ctx);
    testEmptyTileSentinel(ctx);
    testCoplanarPair(ctx);
    testAmbiguous(ctx);
}

} // namespace sabercat::unit
