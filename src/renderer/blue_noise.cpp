// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "blue_noise.h"

#include <cmath>
#include <cstddef>

namespace sabercat
{

namespace
{

constexpr int kSize = static_cast<int>(kBlueNoiseSize);
constexpr int kCount = kSize * kSize;

// About 10% of the pixels start as ones, the minority pattern of void-and-cluster.
constexpr int kInitialOnes = 410;

// The kernel covers offsets of at most kRadius on each axis: sigma 1.5 has
// negligible weight beyond 6, and 2 * kRadius + 1 must stay below kSize.
constexpr int kRadius = 6;
constexpr int kSpan = 2 * kRadius + 1;
constexpr double kSigma = 1.5;

// Kernel weights are rounded to integers (scale 2^20). Every energy is then an exact
// integer sum in double, so the strict-decrease test in phase 1 never depends on
// rounding and the relaxation must end.
constexpr double kWeightScale = 1048576.0;

// Seed of the xorshift32 that places the initial ones. Any nonzero value works; this
// one is fixed so the table never changes.
constexpr uint32_t kSeed = 0x2545F491u;

std::size_t slot(int index)
{
    return static_cast<std::size_t>(index);
}

// Wraps a coordinate into [0, kSize).
int wrap(int value)
{
    return ((value % kSize) + kSize) % kSize;
}

// Signed offset from `from` to `to` on the torus, in [-kSize / 2, kSize / 2).
int torusOffset(int from, int to)
{
    const int d = wrap(to - from);
    return d >= kSize / 2 ? d - kSize : d;
}

// Gaussian weights for offsets in [-kRadius, kRadius]^2, row-major with dy as the row.
// std::exp runs once per entry and the result is rounded at once, so last-bit
// differences between math libraries cannot change any weight.
std::vector<double> makeKernel()
{
    std::vector<double> kernel(slot(kSpan * kSpan));
    for (int dy = -kRadius; dy <= kRadius; ++dy)
    {
        for (int dx = -kRadius; dx <= kRadius; ++dx)
        {
            const double distanceSquared = static_cast<double>(dx * dx + dy * dy);
            const double weight = std::exp(-distanceSquared / (2.0 * kSigma * kSigma));
            kernel[slot((dy + kRadius) * kSpan + dx + kRadius)] = std::round(weight * kWeightScale);
        }
    }
    return kernel;
}

// A binary pattern on the torus. energy[i] is the sum of kernel weights between pixel i
// and every one, including the self weight when i is a one. Setting or clearing a pixel
// updates only the window around it, so a change costs kSpan * kSpan operations.
class Pattern
{
public:
    Pattern()
        : m_kernel(makeKernel())
        , m_energy(slot(kCount), 0.0)
        , m_one(slot(kCount), uint8_t{0})
        , m_center(m_kernel[slot(kRadius * kSpan + kRadius)])
    {
    }

    bool isOne(int index) const
    {
        return m_one[slot(index)] != 0;
    }

    double energy(int index) const
    {
        return m_energy[slot(index)];
    }

    // Energy of a one from the other ones: its self weight is removed.
    double clusterEnergy(int index) const
    {
        return m_energy[slot(index)] - m_center;
    }

    // Kernel weight between two pixels; zero outside the window.
    double weight(int a, int b) const
    {
        const int dx = torusOffset(a % kSize, b % kSize);
        const int dy = torusOffset(a / kSize, b / kSize);
        if (dx < -kRadius || dx > kRadius || dy < -kRadius || dy > kRadius)
        {
            return 0.0;
        }
        return m_kernel[slot((dy + kRadius) * kSpan + dx + kRadius)];
    }

    void set(int index)
    {
        addWindow(index, 1.0);
        m_one[slot(index)] = 1;
    }

    void clear(int index)
    {
        addWindow(index, -1.0);
        m_one[slot(index)] = 0;
    }

private:
    // Each window offset maps to a distinct pixel because kSpan < kSize, so no weight
    // is counted twice.
    void addWindow(int index, double sign)
    {
        const int x = index % kSize;
        const int y = index / kSize;
        for (int dy = -kRadius; dy <= kRadius; ++dy)
        {
            for (int dx = -kRadius; dx <= kRadius; ++dx)
            {
                const int target = wrap(y + dy) * kSize + wrap(x + dx);
                m_energy[slot(target)] += sign * m_kernel[slot((dy + kRadius) * kSpan + dx + kRadius)];
            }
        }
    }

    std::vector<double> m_kernel;
    std::vector<double> m_energy;
    std::vector<uint8_t> m_one;
    double m_center;
};

// The tightest cluster: the one with the largest energy from the other ones. The lowest
// index wins a tie, because the scan is in index order and only a strictly larger value
// replaces the current choice.
int tightestCluster(const Pattern & pattern)
{
    int best = -1;
    double bestEnergy = 0.0;
    for (int i = 0; i < kCount; ++i)
    {
        if (!pattern.isOne(i))
        {
            continue;
        }
        const double e = pattern.clusterEnergy(i);
        if (best < 0 || e > bestEnergy)
        {
            best = i;
            bestEnergy = e;
        }
    }
    return best;
}

// The largest void: the zero with the least energy. The lowest index wins a tie.
int largestVoid(const Pattern & pattern)
{
    int best = -1;
    double bestEnergy = 0.0;
    for (int i = 0; i < kCount; ++i)
    {
        if (pattern.isOne(i))
        {
            continue;
        }
        const double e = pattern.energy(i);
        if (best < 0 || e < bestEnergy)
        {
            best = i;
            bestEnergy = e;
        }
    }
    return best;
}

} // namespace

std::vector<uint16_t> generateBlueNoiseRanks()
{
    Pattern pattern;

    // Place the initial ones with a fixed xorshift32 sequence. A draw that lands on a
    // pixel that is already a one is skipped, so exactly kInitialOnes distinct pixels
    // are set. The sequence has period 2^32 - 1 and each residue mod 4096 recurs many
    // times, so the loop ends after a few hundred draws.
    uint32_t state = kSeed;
    for (int placed = 0; placed < kInitialOnes;)
    {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        const int index = static_cast<int>(state % static_cast<uint32_t>(kCount));
        if (!pattern.isOne(index))
        {
            pattern.set(index);
            ++placed;
        }
    }

    // Phase 1: relax the pattern. The tightest cluster moves to the largest void, but
    // only when that strictly lowers the total energy H = sum over pairs of ones of the
    // kernel weight. H is an integer and drops on every move, while there are finitely
    // many patterns with kInitialOnes ones, so the loop ends. The pattern it stops at
    // is stable for this move.
    for (;;)
    {
        const int clusterPixel = tightestCluster(pattern);
        const int voidPixel = largestVoid(pattern);
        // Removing the cluster pixel lowers the void's energy by their mutual weight, and
        // the cluster's own energy leaves H. The change in H is therefore:
        const double change = (pattern.energy(voidPixel) - pattern.weight(clusterPixel, voidPixel))
            - pattern.clusterEnergy(clusterPixel);
        if (!(change < 0.0))
        {
            break;
        }
        pattern.clear(clusterPixel);
        pattern.set(voidPixel);
    }

    // Phase 2: rank the ones from kInitialOnes - 1 down to 0. Each step removes the
    // tightest cluster, so the most clustered pixels get the highest ranks.
    std::vector<int> ranks(slot(kCount), -1);
    for (int rank = kInitialOnes - 1; rank >= 0; --rank)
    {
        const int clusterPixel = tightestCluster(pattern);
        ranks[slot(clusterPixel)] = rank;
        pattern.clear(clusterPixel);
    }

    // Phase 3: the pixels with ranks 0..kInitialOnes-1 form the pattern. Each further rank
    // goes to the largest void, and that pixel then joins the pattern. Ranked pixels are
    // exactly the ones, so largestVoid only considers unranked pixels.
    for (int i = 0; i < kCount; ++i)
    {
        if (ranks[slot(i)] >= 0)
        {
            pattern.set(i);
        }
    }
    for (int rank = kInitialOnes; rank < kCount; ++rank)
    {
        const int voidPixel = largestVoid(pattern);
        ranks[slot(voidPixel)] = rank;
        pattern.set(voidPixel);
    }

    std::vector<uint16_t> result(slot(kCount));
    for (int i = 0; i < kCount; ++i)
    {
        result[slot(i)] = static_cast<uint16_t>(ranks[slot(i)]);
    }
    return result;
}

float stochasticFrameOffset(uint32_t frame)
{
    // The product is exact enough in double for every 32-bit frame index.
    const double product = static_cast<double>(frame) * 0.6180339887498949;
    const double fractional = product - std::floor(product);
    const float offset = static_cast<float>(fractional);
    // A fraction just below 1 can round up to 1.0f in float. Map that to 0 so the
    // result stays in [0, 1).
    return offset < 1.0f ? offset : 0.0f;
}

uint32_t stochasticHash(uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

} // namespace sabercat
