// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Blue-noise rank table and the stochastic helpers (src/renderer/blue_noise.h). The
// table must be a permutation, deterministic, and blue: its 8x8 block means vary far
// less than white noise would, and horizontal neighbors differ by a large rank.
#include "unit_test.h"

#include "blue_noise.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace sabercat::unit
{

void testBlueNoise(Context & ctx)
{
    constexpr uint32_t kCount = kBlueNoiseSize * kBlueNoiseSize;
    const std::vector<uint16_t> ranks = generateBlueNoiseRanks();
    UNIT_CHECK(ctx, ranks.size() == kCount);

    // A permutation of 0..4095: sorted, the value at index i is i.
    bool permutation = ranks.size() == kCount;
    if (permutation)
    {
        std::vector<uint16_t> sorted = ranks;
        std::sort(sorted.begin(), sorted.end());
        for (uint32_t i = 0; i < kCount && permutation; ++i)
        {
            permutation = static_cast<uint32_t>(sorted[i]) == i;
        }
    }
    UNIT_CHECK(ctx, permutation);

    // Deterministic: a second call returns the same table, element for element.
    UNIT_CHECK(ctx, generateBlueNoiseRanks() == ranks);

    // Block means over the 64 blocks of 8x8 pixels. Each block sums 64 ranks around a
    // mean of 2047.5. White noise gives a standard deviation of 147.8 for these means;
    // blue noise must be below 74 and every mean must stay in [1500, 2600].
    std::vector<double> blockMeans;
    for (uint32_t by = 0; by < 8; ++by)
    {
        for (uint32_t bx = 0; bx < 8; ++bx)
        {
            double sum = 0.0;
            for (uint32_t y = 0; y < 8; ++y)
            {
                for (uint32_t x = 0; x < 8; ++x)
                {
                    sum += ranks[(by * 8 + y) * kBlueNoiseSize + bx * 8 + x];
                }
            }
            blockMeans.push_back(sum / 64.0);
        }
    }
    double meanOfMeans = 0.0;
    double lowest = blockMeans[0];
    double highest = blockMeans[0];
    for (const double m : blockMeans)
    {
        meanOfMeans += m;
        lowest = std::min(lowest, m);
        highest = std::max(highest, m);
    }
    meanOfMeans /= static_cast<double>(blockMeans.size());
    double variance = 0.0;
    for (const double m : blockMeans)
    {
        variance += (m - meanOfMeans) * (m - meanOfMeans);
    }
    variance /= static_cast<double>(blockMeans.size());
    UNIT_CHECK(ctx, std::sqrt(variance) < 74.0);
    UNIT_CHECK(ctx, lowest >= 1500.0 && highest <= 2600.0);

    // Horizontally adjacent pixels, with wraparound at the edge. Independent uniform
    // ranks would give a mean absolute difference of about 4096 / 3 = 1365; blue noise
    // keeps neighbors apart, so it must stay above 1000.
    double totalDifference = 0.0;
    for (uint32_t y = 0; y < kBlueNoiseSize; ++y)
    {
        for (uint32_t x = 0; x < kBlueNoiseSize; ++x)
        {
            const int here = ranks[y * kBlueNoiseSize + x];
            const int right = ranks[y * kBlueNoiseSize + (x + 1) % kBlueNoiseSize];
            totalDifference += static_cast<double>(std::abs(here - right));
        }
    }
    UNIT_CHECK(ctx, totalDifference / static_cast<double>(kCount) > 1000.0);

    // Frame offsets: the golden-ratio step, fract(frame * 0.6180339887).
    UNIT_CHECK_NEAR(ctx, stochasticFrameOffset(0), 0.0, 0.0);
    UNIT_CHECK_NEAR(ctx, stochasticFrameOffset(1), 0.6180339887, 1e-6);
    UNIT_CHECK_NEAR(ctx, stochasticFrameOffset(2), 0.2360679775, 1e-6);
    bool offsetsInRange = true;
    for (uint32_t frame = 0; frame < 4096; ++frame)
    {
        const float offset = stochasticFrameOffset(frame);
        offsetsInRange = offsetsInRange && offset >= 0.0f && offset < 1.0f;
    }
    offsetsInRange = offsetsInRange && stochasticFrameOffset(0xFFFFFFFFu) >= 0.0f
        && stochasticFrameOffset(0xFFFFFFFFu) < 1.0f;
    UNIT_CHECK(ctx, offsetsInRange);

    // lowbias32 by hand for x = 1, all arithmetic mod 2^32:
    //   x ^= x >> 16    -> 0x00000001
    //   x *= 0x7feb352d -> 0x7FEB352D
    //   x ^= x >> 15    -> 0x7FEB352D ^ 0x0000FFD6 = 0x7FEBCAFB
    //   x *= 0x846ca68b -> 0x6889F849 (the product, mod 2^32, was checked by a second
    //                      multiplication in decimal)
    //   x ^= x >> 16    -> 0x6889F849 ^ 0x00006889 = 0x688990C0
    UNIT_CHECK(ctx, stochasticHash(0) == 0u);
    UNIT_CHECK(ctx, stochasticHash(1) == 0x688990C0u);
    UNIT_CHECK(ctx, stochasticHash(1) != stochasticHash(2));

    // The hash is a bijection on 32 bits, so 65536 inputs must give 65536 distinct outputs.
    std::vector<uint32_t> hashed;
    hashed.reserve(65536);
    for (uint32_t x = 0; x < 65536; ++x)
    {
        hashed.push_back(stochasticHash(x));
    }
    std::sort(hashed.begin(), hashed.end());
    UNIT_CHECK(ctx, std::adjacent_find(hashed.begin(), hashed.end()) == hashed.end());
}

} // namespace sabercat::unit
