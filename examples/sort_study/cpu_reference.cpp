// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Keep the CPU blend uncontracted. A fused multiply-add rounds differently
// from the separate multiply and add the GPU shaders do, and the study compares
// the two bit for bit in its tests.
#pragma fp_contract(off)

#include "cpu_reference.h"

#include "sabercat/parallel_for.h"

// The stochastic model reads the same hash and frame offset as the shader, so
// it uses the renderer's helpers rather than a copy that could drift.
#include "blue_noise.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace sabercat::study
{

namespace
{

constexpr uint32_t kTileSize = 16;
// The stochastic rank lookup masks coordinates with 63 and strides by 64, so it
// only works for the 64 x 64 table the shared helpers generate.
static_assert(sabercat::kBlueNoiseSize == 64u, "the stochastic model assumes a 64 x 64 noise table");
constexpr uint32_t kMaxStochasticSamples = 8;
constexpr float kTransmittanceFloor = 1e-4f;
constexpr float kMinAlpha = 1.0f / 255.0f;
constexpr float kMaxAlpha = 0.99f;

// evaluateSplat from forward_common.slangh, in float, same operations in the
// same order. Returns false when the Gaussian does not contribute to the pixel.
bool evaluateSplat(float px, float py, const ProjectedGaussian & g, float & alphaOut)
{
    const float dx = px - g.center.x;
    const float dy = py - g.center.y;
    const float e = g.conic.x * dx * dx + 2.0f * g.conic.y * dx * dy + g.conic.z * dy * dy;
    if (e < 0.0f)
    {
        return false;
    }
    const float gaussian = std::exp(-0.5f * e);
    const float raw = g.alpha * gaussian;
    alphaOut = std::min(raw, kMaxAlpha);
    return alphaOut >= kMinAlpha;
}

// Front-to-back accumulation for one pixel. The two statements keep the GPU's
// operation order: the color is added with the transmittance before this
// contributor, then the transmittance is updated.
struct PixelBlend
{
    // Grouped by size: vec3 then float, so the struct has no padding.
    glm::vec3 acc {0.0f};
    float transmittance = 1.0f;

    void add(float alpha, const glm::vec3 & color)
    {
        acc += transmittance * alpha * color;
        transmittance *= (1.0f - alpha);
    }

    bool saturated() const
    {
        return transmittance < kTransmittanceFloor;
    }

    glm::vec4 result() const
    {
        return glm::vec4(acc, 1.0f - transmittance);
    }
};

// A pixel's contributor, keyed for the ray-depth orders.
struct Contributor
{
    // Grouped by size: the double leads, so only the tail pads to 8 bytes.
    double depth = 0.0; // t* along the pixel ray, or the center depth
    glm::vec3 color {0.0f};
    uint32_t gaussian = 0;
    float alpha = 0.0f;
};

// Ascending by depth, ties broken by Gaussian index. Keys are unique within a
// tile's list, so the order is total and deterministic.
bool contributorLess(const Contributor & a, const Contributor & b)
{
    return a.depth < b.depth || (a.depth == b.depth && a.gaussian < b.gaussian);
}

// The ray depth of Gaussian `gaussian` along (dx, dy, -1). A NaN depth would
// break the sort's strict weak ordering, so it is treated as infinitely far.
double rayDepthOf(const StudyCandidates & c, uint32_t gaussian, double dx, double dy)
{
    const float * form = c.rayForm.empty() ? nullptr : c.rayForm.data() + size_t(gaussian) * 12;
    double depth = evalRayDepth(form, c.projected[gaussian].depth, dx, dy);
    if (std::isnan(depth))
    {
        depth = std::numeric_limits<double>::infinity();
    }
    return depth;
}

// TileOrder: the tile's list, blended in order.
glm::vec4 shadeTileOrder(const StudyCandidates & c, TileRange range, float px, float py)
{
    PixelBlend blend;
    for (uint32_t i = range.first; i < range.last; ++i)
    {
        const ProjectedGaussian & g = c.projected[c.sortedValues[i]];
        float alpha = 0.0f;
        if (!evaluateSplat(px, py, g, alpha))
        {
            continue;
        }
        blend.add(alpha, g.color);
        if (blend.saturated())
        {
            break;
        }
    }
    return blend.result();
}

// ExactRayDepth: every contributor, sorted by ray depth, then blended.
glm::vec4 shadeExactRayDepth(const StudyCandidates & c, TileRange range, uint32_t x, uint32_t y,
                             std::vector<Contributor> & scratch)
{
    const double dx = (double(x) - double(c.cx)) / double(c.fx);
    const double dy = -(double(y) - double(c.cy)) / double(c.fy);
    const float px = float(x);
    const float py = float(y);

    scratch.clear();
    for (uint32_t i = range.first; i < range.last; ++i)
    {
        const uint32_t gaussian = c.sortedValues[i];
        const ProjectedGaussian & g = c.projected[gaussian];
        float alpha = 0.0f;
        if (!evaluateSplat(px, py, g, alpha))
        {
            continue;
        }
        scratch.push_back(Contributor {rayDepthOf(c, gaussian, dx, dy), g.color, gaussian, alpha});
    }
    std::sort(scratch.begin(), scratch.end(), contributorLess);

    PixelBlend blend;
    for (const Contributor & s : scratch)
    {
        blend.add(s.alpha, s.color);
        if (blend.saturated())
        {
            break;
        }
    }
    return blend.result();
}

// WindowRayDepth: a buffer of at most `window` contributors, kept sorted. When
// it is full, the smallest is blended before the next arrival is inserted. The
// buffer is flushed in ascending order after the list ends. With window = 1 this
// is the list order; with window >= the contributor count it is the exact order.
glm::vec4 shadeWindowRayDepth(const StudyCandidates & c, TileRange range, uint32_t window,
                              uint32_t x, uint32_t y, std::vector<Contributor> & buffer)
{
    const double dx = (double(x) - double(c.cx)) / double(c.fx);
    const double dy = -(double(y) - double(c.cy)) / double(c.fy);
    const float px = float(x);
    const float py = float(y);

    buffer.clear();
    PixelBlend blend;
    bool stopped = false;
    for (uint32_t i = range.first; i < range.last && !stopped; ++i)
    {
        const uint32_t gaussian = c.sortedValues[i];
        const ProjectedGaussian & g = c.projected[gaussian];
        float alpha = 0.0f;
        if (!evaluateSplat(px, py, g, alpha))
        {
            continue;
        }

        if (buffer.size() >= window)
        {
            const Contributor & smallest = buffer.front();
            blend.add(smallest.alpha, smallest.color);
            buffer.erase(buffer.begin());
            if (blend.saturated())
            {
                stopped = true;
                break;
            }
        }
        const Contributor cur {rayDepthOf(c, gaussian, dx, dy), g.color, gaussian, alpha};
        const auto pos = std::upper_bound(buffer.begin(), buffer.end(), cur, contributorLess);
        buffer.insert(pos, cur);
    }

    if (!stopped)
    {
        for (const Contributor & s : buffer)
        {
            blend.add(s.alpha, s.color);
            if (blend.saturated())
            {
                break;
            }
        }
    }
    return blend.result();
}

// The structural checks every renderer makes before it indexes the candidates.
// Throws std::invalid_argument, naming the renderer through `who`.
void validateLists(const StudyCandidates & c, const char * who)
{
    const size_t tileCount = size_t(c.tilesX) * c.tilesY;
    if (c.tileRanges.size() != tileCount)
    {
        throw std::invalid_argument(std::string(who) + ": tileRanges must hold tilesX * tilesY entries");
    }
    if (size_t(c.tilesX) * kTileSize < c.width || size_t(c.tilesY) * kTileSize < c.height)
    {
        throw std::invalid_argument(std::string(who) + ": the tile grid does not cover the image");
    }
    if (!c.rayForm.empty() && c.rayForm.size() != 12 * c.projected.size())
    {
        throw std::invalid_argument(std::string(who) + ": rayForm must hold 12 floats per Gaussian");
    }
    for (const TileRange & r : c.tileRanges)
    {
        if (r.first > r.last || r.last > c.sortedValues.size())
        {
            throw std::invalid_argument(std::string(who) + ": a tile range lies outside sortedValues");
        }
    }
    for (const uint32_t gaussian : c.sortedValues)
    {
        if (gaussian >= c.projected.size())
        {
            throw std::invalid_argument(std::string(who) + ": sortedValues names a missing Gaussian");
        }
    }
}

// Throws std::invalid_argument for candidates the renderer cannot index safely.
void validateCandidates(const StudyCandidates & c, const ReferenceOptions & o)
{
    validateLists(c, "renderReference");
    if (o.order != ReferenceOrder::TileOrder && !(c.fx > 0.0f && c.fy > 0.0f))
    {
        throw std::invalid_argument("renderReference: a ray-depth order needs positive fx and fy");
    }
    if (o.order == ReferenceOrder::WindowRayDepth && o.window == 0)
    {
        throw std::invalid_argument("renderReference: WindowRayDepth needs window >= 1");
    }
}

// Peak depth along the ray (dx, dy, -1) in float, the operation order of
// rayDepthAlongRay (sort_common.slangh): dp = c0*dx + c1*dy - c2 per component,
// dd = (x*x + y*y) + z*z, t = dot(u, dp) / dd. The fallbacks match the shader.
// A null form means "no form" and returns the center depth.
float rayDepthFloat(const float * form12, float fallbackDepth, float dx, float dy)
{
    if (form12 == nullptr)
    {
        return fallbackDepth;
    }
    const glm::vec3 c0(form12[0], form12[1], form12[2]);
    const glm::vec3 c1(form12[3], form12[4], form12[5]);
    const glm::vec3 c2(form12[6], form12[7], form12[8]);
    const glm::vec3 u(form12[9], form12[10], form12[11]);

    const glm::vec3 dp = c0 * dx + c1 * dy - c2;
    const float dd = dp.x * dp.x + dp.y * dp.y + dp.z * dp.z;
    if (!(dd > 0.0f) || !std::isfinite(dd))
    {
        return fallbackDepth;
    }
    const float t = (u.x * dp.x + u.y * dp.y + u.z * dp.z) / dd;
    return std::isfinite(t) ? t : fallbackDepth;
}

// fract(x) = x - floor(x), as fractf in sort_common.slangh.
float fractFloat(float x)
{
    return x - std::floor(x);
}

// One pixel of the stochastic rasterizer. Each sample keeps the nearest covered
// candidate it has seen, so the candidates can arrive in any order. A candidate
// covers sample s when its alpha exceeds the sample's threshold xi_s. The
// operations and their order follow tile_rasterize_stochastic.
glm::vec4 shadeStochastic(const StudyCandidates & c, TileRange range, uint32_t x, uint32_t y,
                          uint32_t samples, float frameOffset, uint32_t frameIndex,
                          const std::vector<uint16_t> & ranks)
{
    const float px = float(x);
    const float py = float(y);
    // Camera-space ray through the pixel, as the shader's dxdy.
    const float dx = (px - c.cx) / c.fx;
    const float dy = -(py - c.cy) / c.fy;

    // Each sample reads the noise table at its own offset, so the samples of one
    // pixel are decorrelated. The offsets are the shader's, so the masks stay 63.
    float noise[kMaxStochasticSamples];
    for (uint32_t s = 0; s < samples; ++s)
    {
        const uint32_t rank = ranks[((y + 23u * s) & 63u) * 64u + ((x + 41u * s) & 63u)];
        noise[s] = float(rank) * (1.0f / 4096.0f);
    }

    // Each sample gets its own hash seed per frame. With one shared hash per
    // candidate, every sample would see the same per-candidate offset, so the
    // candidates' thresholds would move together and the frame average would
    // stall. The seed hashes (frame, sample), so it changes every frame and is the
    // same for every pixel and candidate of that sample. The uint32 add in the
    // candidate loop wraps, as it does in the shader.
    uint32_t seed[kMaxStochasticSamples];
    for (uint32_t s = 0; s < samples; ++s)
    {
        seed[s] = sabercat::stochasticHash(frameIndex * 8u + s);
    }

    const float infinity = std::numeric_limits<float>::infinity();
    float depthS[kMaxStochasticSamples];
    glm::vec3 colorS[kMaxStochasticSamples];
    for (uint32_t s = 0; s < samples; ++s)
    {
        depthS[s] = infinity;
        colorS[s] = glm::vec3(0.0f);
    }

    for (uint32_t i = range.first; i < range.last; ++i)
    {
        const uint32_t gaussian = c.sortedValues[i];
        const ProjectedGaussian & g = c.projected[gaussian];
        float alpha = 0.0f;
        if (!evaluateSplat(px, py, g, alpha))
        {
            continue;
        }

        const float * form = c.rayForm.empty() ? nullptr : c.rayForm.data() + size_t(gaussian) * 12;
        const float t = rayDepthFloat(form, g.depth, dx, dy);
        for (uint32_t s = 0; s < samples; ++s)
        {
            // A 24-bit hash of (Gaussian, sample seed), in [0, 1): a per-candidate
            // offset that differs between Gaussians and between samples. The hash
            // keeps 24 bits, so the scale by a power of two is exact in float.
            const float hashTerm = float(sabercat::stochasticHash(gaussian + seed[s]) >> 8) * (1.0f / 16777216.0f);
            // The two adds are in this order to match the shader's rounding.
            const float xi = fractFloat((noise[s] + frameOffset) + hashTerm);
            // Strict "<": on a depth tie the earlier candidate in list order keeps the sample.
            if (alpha > xi && t < depthS[s])
            {
                depthS[s] = t;
                colorS[s] = g.color;
            }
        }
    }

    // Average over the samples: a covered sample adds its color, an uncovered one
    // adds nothing, and the alpha is the covered fraction.
    glm::vec3 acc(0.0f);
    uint32_t covered = 0;
    for (uint32_t s = 0; s < samples; ++s)
    {
        if (depthS[s] < infinity)
        {
            acc += colorS[s];
            ++covered;
        }
    }
    const float invSamples = 1.0f / float(samples);
    return glm::vec4(acc * invSamples, float(covered) * invSamples);
}

} // namespace

double evalRayDepth(const float * form12, double fallbackDepth, double dx, double dy)
{
    if (form12 == nullptr)
    {
        return fallbackDepth;
    }

    // The stored columns of P (c0, c1, c2) and the vector u. Each column is three
    // floats, so c0 = form[0..2], c1 = form[3..5], c2 = form[6..8], u = form[9..11].
    const double c0x = form12[0], c0y = form12[1], c0z = form12[2];
    const double c1x = form12[3], c1y = form12[4], c1z = form12[5];
    const double c2x = form12[6], c2y = form12[7], c2z = form12[8];
    const double ux = form12[9], uy = form12[10], uz = form12[11];

    // d' = P d for the ray d = (dx, dy, -1): d' = c0*dx + c1*dy - c2.
    const double vx = c0x * dx + c1x * dy - c2x;
    const double vy = c0y * dx + c1y * dy - c2y;
    const double vz = c0z * dx + c1z * dy - c2z;

    // t* = u.d' / d'.d', the peak depth along the ray.
    const double dd = vx * vx + vy * vy + vz * vz;
    if (!(dd > 0.0) || !std::isfinite(dd))
    {
        return fallbackDepth;
    }
    const double t = (ux * vx + uy * vy + uz * vz) / dd;
    return std::isfinite(t) ? t : fallbackDepth;
}

std::vector<glm::vec4> renderReference(const StudyCandidates & c, const ReferenceOptions & o)
{
    validateCandidates(c, o);

    std::vector<glm::vec4> image(size_t(c.width) * c.height);
    // Rows are independent and write disjoint slices of image, so no locking is
    // needed. The scratch buffer is per chunk and reused across its pixels.
    parallelFor(
        size_t(c.height),
        [&](size_t rowBegin, size_t rowEnd) {
            std::vector<Contributor> scratch;
            for (size_t y = rowBegin; y < rowEnd; ++y)
            {
                const size_t tileRow = y / kTileSize;
                for (uint32_t x = 0; x < c.width; ++x)
                {
                    const TileRange range = c.tileRanges[tileRow * c.tilesX + x / kTileSize];
                    glm::vec4 & out = image[y * c.width + x];
                    switch (o.order)
                    {
                    case ReferenceOrder::TileOrder:
                        out = shadeTileOrder(c, range, float(x), float(y));
                        break;
                    case ReferenceOrder::ExactRayDepth:
                        out = shadeExactRayDepth(c, range, x, uint32_t(y), scratch);
                        break;
                    case ReferenceOrder::WindowRayDepth:
                        out = shadeWindowRayDepth(c, range, o.window, x, uint32_t(y), scratch);
                        break;
                    }
                }
            }
        },
        1);
    return image;
}

std::vector<glm::vec4> renderStochasticModel(const StudyCandidates & c, const std::vector<uint16_t> & ranks,
                                             const StochasticOptions & o)
{
    if (ranks.size() != size_t(sabercat::kBlueNoiseSize) * sabercat::kBlueNoiseSize)
    {
        throw std::invalid_argument("renderStochasticModel: ranks must hold 4096 entries (64 x 64)");
    }
    if (o.samples != 4 && o.samples != 8)
    {
        throw std::invalid_argument("renderStochasticModel: samples must be 4 or 8");
    }
    validateLists(c, "renderStochasticModel");
    if (!(c.fx > 0.0f && c.fy > 0.0f))
    {
        throw std::invalid_argument("renderStochasticModel: needs positive fx and fy");
    }

    // The frame offset is one float for the whole image, as the shader's params buffer.
    const float frameOffset = sabercat::stochasticFrameOffset(o.frameIndex);
    std::vector<glm::vec4> image(size_t(c.width) * c.height);
    // Rows write disjoint slices of image, so no locking is needed.
    parallelFor(
        size_t(c.height),
        [&](size_t rowBegin, size_t rowEnd) {
            for (size_t y = rowBegin; y < rowEnd; ++y)
            {
                const size_t tileRow = y / kTileSize;
                for (uint32_t x = 0; x < c.width; ++x)
                {
                    const TileRange range = c.tileRanges[tileRow * c.tilesX + x / kTileSize];
                    image[y * c.width + x] = shadeStochastic(c, range, x, uint32_t(y), o.samples, frameOffset, o.frameIndex, ranks);
                }
            }
        },
        1);
    return image;
}

} // namespace sabercat::study
