// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Image and sort-order metrics for the sort study: error between renders,
// popping between frames, and near-tie swaps in the default sort. No GPU
// calls. sort_metrics.h uses TileRange from forward_pass.h, a header type only.
#pragma once

#include <glm/glm.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

#include "sabercat/forward_pass.h"

namespace sabercat::study
{

// Mean over pixels and RGB of |a - b|. Throws std::invalid_argument when the
// sizes differ. Empty images give 0.
double meanAbsoluteError(const std::vector<glm::vec4> & a, const std::vector<glm::vec4> & b);

// Peak-1 PSNR over RGB, with the mse accumulated in double. +infinity when the
// mse is 0 (and for empty images).
double psnrDb(const std::vector<glm::vec4> & reference, const std::vector<glm::vec4> & test);

// Accumulates popping over a sequence of frames. Each frame compares a mode's
// two images with the reference renderer's two images for the same camera
// motion. Popping is what the mode shows and the reference does not.
class PoppingAccumulator
{
  public:
    // One frame step: prevM/curM are the mode's images, prevRef/curRef the
    // reference's. The per-frame score is the mean over pixels and RGB of
    // max(0, |cur_m - prev_m| - |cur_ref - prev_ref|). A pop pixel has
    // max over RGB of |cur_m - prev_m| > 0.05 and max over RGB of
    // |cur_ref - prev_ref| < 0.01. Throws std::invalid_argument when the four
    // images differ in size.
    void addFrame(const std::vector<glm::vec4> & prevM, const std::vector<glm::vec4> & curM,
                  const std::vector<glm::vec4> & prevRef, const std::vector<glm::vec4> & curRef);

    size_t frames() const
    {
        return m_frames;
    }
    // Means over the frames added so far, 0 when there are none.
    double meanScore() const;
    double meanPopPixels() const;

  private:
    // Grouped by size (all 8 bytes), so the class carries no padding.
    uint64_t m_popTotal = 0;
    double m_scoreSum = 0.0;
    size_t m_frames = 0;
};

// One frame's default sort, for near-tie swap counting.
struct SortOrderFrame
{
    std::vector<uint32_t> sortedValues;
    std::vector<TileRange> tileRanges;
    std::vector<float> depth; // per Gaussian: ProjectedGaussian::depth
};

// Counts, over all tiles, the pairs of Gaussians present in both frames' lists
// for that tile, whose depths are within relTol of each other and whose relative
// order differs between the two lists. Depths come from cur. Cost is
// O(n log n + pairs) per tile. Throws std::invalid_argument when the tile
// counts differ, when an index is out of range, or when relTol is outside [0, 1).
uint64_t countNearTieSwaps(const SortOrderFrame & prev, const SortOrderFrame & cur, double relTol = 1e-3);

// Nearest-rank percentile of a copy of `values`, p in [0, 1]: index =
// ceil(p * n) - 1, clamped. Returns 0 for an empty input.
double percentile(std::vector<double> values, double p);

} // namespace sabercat::study
