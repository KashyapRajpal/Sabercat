// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sort_metrics.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace sabercat::study
{

namespace
{

// Popping thresholds: the mode must move a pixel by more than kModeMove while
// the reference stays within kRefStill.
constexpr double kModeMove = 0.05;
constexpr double kRefStill = 0.01;

// Marks a Gaussian that is absent from the previous frame's tile list.
constexpr uint32_t kAbsent = std::numeric_limits<uint32_t>::max();

void requireSameSize(const std::vector<glm::vec4> & a, const std::vector<glm::vec4> & b, const char * what)
{
    if (a.size() != b.size())
    {
        throw std::invalid_argument(std::string(what) + ": images differ in size");
    }
}

void requireRange(TileRange r, size_t listSize, const char * what)
{
    if (r.first > r.last || r.last > listSize)
    {
        throw std::invalid_argument(std::string(what) + ": a tile range lies outside sortedValues");
    }
}

// One Gaussian present in both frames' lists for the tile being scanned.
struct TileItem
{
    // Grouped by size: the double leads, then the 4-byte fields.
    double depth;
    uint32_t gaussian;
    uint32_t curPos;  // position in the current frame's tile list
    uint32_t prevPos; // position in the previous frame's tile list
};

} // namespace

double meanAbsoluteError(const std::vector<glm::vec4> & a, const std::vector<glm::vec4> & b)
{
    requireSameSize(a, b, "meanAbsoluteError");
    if (a.empty())
    {
        return 0.0;
    }
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        for (int ch = 0; ch < 3; ++ch)
        {
            sum += std::fabs(double(a[i][ch]) - double(b[i][ch]));
        }
    }
    return sum / (3.0 * double(a.size()));
}

double psnrDb(const std::vector<glm::vec4> & reference, const std::vector<glm::vec4> & test)
{
    requireSameSize(reference, test, "psnrDb");
    if (reference.empty())
    {
        return std::numeric_limits<double>::infinity();
    }
    double sumSquares = 0.0;
    for (size_t i = 0; i < reference.size(); ++i)
    {
        for (int ch = 0; ch < 3; ++ch)
        {
            const double d = double(reference[i][ch]) - double(test[i][ch]);
            sumSquares += d * d;
        }
    }
    const double mse = sumSquares / (3.0 * double(reference.size()));
    if (mse == 0.0)
    {
        return std::numeric_limits<double>::infinity();
    }
    // Peak value 1: 10 log10(1 / mse).
    return -10.0 * std::log10(mse);
}

void PoppingAccumulator::addFrame(const std::vector<glm::vec4> & prevM, const std::vector<glm::vec4> & curM,
                                  const std::vector<glm::vec4> & prevRef, const std::vector<glm::vec4> & curRef)
{
    requireSameSize(prevM, curM, "PoppingAccumulator::addFrame");
    requireSameSize(prevM, prevRef, "PoppingAccumulator::addFrame");
    requireSameSize(prevM, curRef, "PoppingAccumulator::addFrame");

    const size_t pixelCount = prevM.size();
    double scoreSum = 0.0;
    uint64_t pops = 0;
    for (size_t i = 0; i < pixelCount; ++i)
    {
        double maxMode = 0.0;
        double maxRef = 0.0;
        for (int ch = 0; ch < 3; ++ch)
        {
            const double modeMove = std::fabs(double(curM[i][ch]) - double(prevM[i][ch]));
            const double refMove = std::fabs(double(curRef[i][ch]) - double(prevRef[i][ch]));
            scoreSum += std::max(0.0, modeMove - refMove);
            maxMode = std::max(maxMode, modeMove);
            maxRef = std::max(maxRef, refMove);
        }
        if (maxMode > kModeMove && maxRef < kRefStill)
        {
            ++pops;
        }
    }

    // The per-frame score is the mean over pixels and RGB.
    const double frameScore = pixelCount == 0 ? 0.0 : scoreSum / (3.0 * double(pixelCount));
    m_scoreSum += frameScore;
    m_popTotal += pops;
    ++m_frames;
}

double PoppingAccumulator::meanScore() const
{
    return m_frames == 0 ? 0.0 : m_scoreSum / double(m_frames);
}

double PoppingAccumulator::meanPopPixels() const
{
    return m_frames == 0 ? 0.0 : double(m_popTotal) / double(m_frames);
}

uint64_t countNearTieSwaps(const SortOrderFrame & prev, const SortOrderFrame & cur, double relTol)
{
    if (prev.tileRanges.size() != cur.tileRanges.size())
    {
        throw std::invalid_argument("countNearTieSwaps: the two frames have different tile counts");
    }
    if (!(relTol >= 0.0 && relTol < 1.0))
    {
        throw std::invalid_argument("countNearTieSwaps: relTol must lie in [0, 1)");
    }

    // The scan below stops at the first pair outside the tolerance. That is
    // exact because, with depths >= 0 and relTol < 1, the window of partners
    // for a sorted depth is contiguous.
    const size_t gaussianCount = cur.depth.size();
    std::vector<uint32_t> prevPos(gaussianCount, kAbsent);
    std::vector<TileItem> items;
    uint64_t swaps = 0;

    for (size_t tile = 0; tile < cur.tileRanges.size(); ++tile)
    {
        const TileRange prevRange = prev.tileRanges[tile];
        const TileRange curRange = cur.tileRanges[tile];
        requireRange(prevRange, prev.sortedValues.size(), "countNearTieSwaps");
        requireRange(curRange, cur.sortedValues.size(), "countNearTieSwaps");
        if (prevRange.last - prevRange.first < 2 || curRange.last - curRange.first < 2)
        {
            continue; // fewer than two entries: no pair can swap
        }

        for (uint32_t i = prevRange.first; i < prevRange.last; ++i)
        {
            const uint32_t gaussian = prev.sortedValues[i];
            if (gaussian >= gaussianCount)
            {
                throw std::invalid_argument("countNearTieSwaps: sortedValues names a Gaussian without a depth");
            }
            prevPos[gaussian] = i - prevRange.first;
        }

        // Current-frame entries also present in the previous frame's tile list,
        // sorted by depth so each candidate pair is visited once, low to high.
        items.clear();
        for (uint32_t i = curRange.first; i < curRange.last; ++i)
        {
            const uint32_t gaussian = cur.sortedValues[i];
            if (gaussian >= gaussianCount)
            {
                throw std::invalid_argument("countNearTieSwaps: sortedValues names a Gaussian without a depth");
            }
            if (prevPos[gaussian] == kAbsent)
            {
                continue;
            }
            items.push_back(TileItem {double(cur.depth[gaussian]), gaussian, i - curRange.first, prevPos[gaussian]});
        }
        std::sort(items.begin(), items.end(), [](const TileItem & a, const TileItem & b) {
            return a.depth < b.depth || (a.depth == b.depth && a.gaussian < b.gaussian);
        });

        for (size_t i = 0; i < items.size(); ++i)
        {
            for (size_t j = i + 1; j < items.size(); ++j)
            {
                const double gap = items[j].depth - items[i].depth;
                const double tolerance = relTol * std::max(items[i].depth, items[j].depth);
                if (!(gap < tolerance))
                {
                    break;
                }
                const bool prevLess = items[i].prevPos < items[j].prevPos;
                const bool curLess = items[i].curPos < items[j].curPos;
                if (prevLess != curLess)
                {
                    ++swaps;
                }
            }
        }

        // Reset only the entries this tile set, so the scratch map is clean for
        // the next tile without an O(gaussianCount) clear.
        for (uint32_t i = prevRange.first; i < prevRange.last; ++i)
        {
            prevPos[prev.sortedValues[i]] = kAbsent;
        }
    }
    return swaps;
}

double percentile(std::vector<double> values, double p)
{
    if (values.empty())
    {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double n = double(values.size());
    const double rank = std::ceil(p * n);
    size_t index = 0;
    if (rank >= n)
    {
        index = values.size() - 1;
    }
    else if (rank >= 1.0)
    {
        index = size_t(rank) - 1;
    }
    return values[index];
}

} // namespace sabercat::study
