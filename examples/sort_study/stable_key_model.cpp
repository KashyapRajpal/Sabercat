// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Keep the bucket coordinate uncontracted. A fused multiply-add rounds
// differently from the separate multiply and add the GPU kernel performs, and
// the sticky rule compares against a float edge, so the rounding matters.
#pragma fp_contract(off)

#include "stable_key_model.h"

#include <cmath>

namespace sabercat::study
{

namespace
{

// Half-width of the band around each decision edge in which the host and the
// GPU may disagree. Float log2 differs by a few ulps between implementations,
// which is far below this band.
constexpr float kAmbiguityBand = 0.02f;

// The bucket coordinate, in the kernel's order of operations. The offset of 64
// octaves keeps depths down to 2^-64 positive before the floor.
float bucketCoordinate(float depth, float bucketsPerOctave)
{
    return (std::log2(depth) + 64.0f) * bucketsPerOctave;
}

} // namespace

uint32_t stableKeyStep(uint32_t stored, float depth, uint32_t tileCount, float bucketsPerOctave, float hysteresis)
{
    if (tileCount == 0)
    {
        return kStableSentinel;
    }

    const float x = bucketCoordinate(depth, bucketsPerOctave);
    // The stored bucket is kept while x stays within half a bucket of its center,
    // widened by the hysteresis. The sentinel always takes the new bucket.
    if (stored == kStableSentinel || std::fabs(x - (static_cast<float>(stored) + 0.5f)) > 0.5f + hysteresis)
    {
        // A compare in place of max(x, 0): a NaN x (log2 of a negative depth)
        // becomes 0, which is what the GPU's max returns for it.
        const float clamped = x > 0.0f ? x : 0.0f;
        return static_cast<uint32_t>(std::floor(clamped));
    }
    return stored;
}

bool stableKeyAmbiguous(uint32_t stored, float depth, uint32_t tileCount, float bucketsPerOctave, float hysteresis)
{
    if (tileCount == 0)
    {
        return false;
    }

    const float x = bucketCoordinate(depth, bucketsPerOctave);
    // The floor only matters when the step writes a new key, so a close-to-integer
    // x with a kept key is not ambiguous. The sentinel always writes.
    const bool nearInteger = std::fabs(x - std::round(x)) <= kAmbiguityBand;
    if (stored == kStableSentinel)
    {
        return nearInteger;
    }

    // Hysteresis boundary: the comparison against 0.5 + hysteresis may flip.
    const float edge = 0.5f + hysteresis;
    const float distance = std::fabs(x - (static_cast<float>(stored) + 0.5f));
    if (std::fabs(distance - edge) <= kAmbiguityBand)
    {
        return true;
    }
    // Past the edge the step writes a new key, so the floor matters there.
    return distance > edge && nearInteger;
}

} // namespace sabercat::study
