// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Stochastic transparency model (examples/sort_study/cpu_reference.h,
// renderStochasticModel) on hand-built one-tile scenes, with a fake rank table
// that is a permutation. The checks cover the one-color-per-sample structure,
// the convergence of the frame average to the front-to-back blend, determinism,
// and candidates too faint to be covered.
#include "unit_test.h"

#include "cpu_reference.h"

#include <glm/glm.hpp>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace sabercat::unit
{

namespace
{

namespace study = sabercat::study;

// rank[i] = (i * 2503) % 4096. 2503 is odd and 4096 is a power of two, so the
// multiplier is coprime with 4096 and the table is a permutation of 0..4095.
std::vector<uint16_t> buildRanks()
{
    std::vector<uint16_t> ranks(4096);
    for (uint32_t i = 0; i < 4096; ++i)
    {
        ranks[i] = uint16_t((i * 2503u) % 4096u);
    }
    return ranks;
}

// A projected Gaussian with the fields the stochastic model reads. Its depth is
// its center depth, because the scenes carry no ray-depth form.
ProjectedGaussian makeSplat(glm::vec2 center, glm::vec3 conic, float alpha, glm::vec3 color, float depth)
{
    ProjectedGaussian g {};
    g.center = center;
    g.depth = depth;
    g.radius = 4.0f;
    g.conic = conic;
    g.alpha = alpha;
    g.color = color;
    g.tileCount = 1;
    return g;
}

// A 16 x 16 image that is one tile, with the Gaussians listed in candidate order.
study::StudyCandidates buildTile(const std::vector<ProjectedGaussian> & splats)
{
    study::StudyCandidates c;
    c.width = 16;
    c.height = 16;
    c.tilesX = 1;
    c.tilesY = 1;
    c.fx = 16.0f;
    c.fy = 16.0f;
    c.cx = 8.0f;
    c.cy = 8.0f;
    c.projected = splats;
    for (uint32_t i = 0; i < splats.size(); ++i)
    {
        c.sortedValues.push_back(i);
    }
    c.tileRanges = {TileRange {0u, uint32_t(splats.size())}};
    return c; // rayForm stays empty: every Gaussian uses its center depth
}

// Three overlapping Gaussians at different depths, one per primary color. Their
// reaches overlap, so some pixels see two or three candidates and others see one.
study::StudyCandidates buildScene()
{
    return buildTile({
        makeSplat(glm::vec2(6.0f, 6.0f), glm::vec3(0.1f, 0.0f, 0.1f), 0.8f, glm::vec3(1.0f, 0.0f, 0.0f), 3.0f),
        makeSplat(glm::vec2(9.0f, 9.0f), glm::vec3(0.1f, 0.0f, 0.1f), 0.6f, glm::vec3(0.0f, 1.0f, 0.0f), 1.0f),
        makeSplat(glm::vec2(8.0f, 11.0f), glm::vec3(0.1f, 0.0f, 0.1f), 0.9f, glm::vec3(0.0f, 0.0f, 1.0f), 2.0f),
    });
}

study::StochasticOptions options(uint32_t samples, uint32_t frameIndex)
{
    study::StochasticOptions o;
    o.samples = samples;
    o.frameIndex = frameIndex;
    return o;
}

std::vector<glm::vec4> render(const study::StudyCandidates & c, const std::vector<uint16_t> & ranks,
                              uint32_t samples, uint32_t frameIndex)
{
    return study::renderStochasticModel(c, ranks, options(samples, frameIndex));
}

// Bitwise equality of two images, so -0 and +0 or a single ulp count as different.
bool sameBits(const std::vector<glm::vec4> & a, const std::vector<glm::vec4> & b)
{
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(glm::vec4)) == 0;
}

// True when renderStochasticModel(c, ranks, options) throws std::invalid_argument.
template <typename Fn> bool throwsInvalid(Fn && fn)
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

// Each sample keeps one candidate's color, so rgb * S counts the samples that
// landed on each primary, and those counts add up to the covered samples. A
// blended color would leave fractions in rgb * S, and a wrong count would break
// the sum. With S = 4 the alpha must also be one of 0, .25, .5, .75, 1.
void testOneColorPerSample(Context & ctx)
{
    const study::StudyCandidates c = buildScene();
    const std::vector<uint16_t> ranks = buildRanks();

    bool structured = true;
    bool anyCovered = false;
    for (uint32_t samples : {4u, 8u})
    {
        for (uint32_t frame : {0u, 1u, 7u})
        {
            const std::vector<glm::vec4> image = render(c, ranks, samples, frame);
            for (const glm::vec4 & p : image)
            {
                const float counts[3] = {p.r * float(samples), p.g * float(samples), p.b * float(samples)};
                int total = 0;
                for (float n : counts)
                {
                    const float rounded = std::round(n);
                    if (std::abs(n - rounded) > 1e-5f || rounded < 0.0f)
                    {
                        structured = false;
                    }
                    total += int(rounded);
                }
                const float coveredF = p.a * float(samples);
                const float coveredRounded = std::round(coveredF);
                if (std::abs(coveredF - coveredRounded) > 1e-5f || coveredRounded > float(samples)
                    || int(coveredRounded) != total)
                {
                    structured = false;
                }
                if (samples == 4)
                {
                    const bool onGrid = p.a == 0.0f || p.a == 0.25f || p.a == 0.5f || p.a == 0.75f || p.a == 1.0f;
                    if (!onGrid)
                    {
                        structured = false;
                    }
                }
                if (total > 0)
                {
                    anyCovered = true;
                }
            }
        }
    }
    UNIT_CHECK(ctx, structured);
    UNIT_CHECK(ctx, anyCovered);
}

// Two candidates with alpha 0.99 at the pixel (8, 8), their common center, so the
// alpha there is its maximum. The front one is red at depth 1, the back one is
// green at depth 2. The front-to-back blend is red * 0.99 + green * 0.99 * 0.01.
// Averaging 400 frames of the stochastic model converges to it.
void testConvergesToBlend(Context & ctx)
{
    const study::StudyCandidates c = buildTile({
        makeSplat(glm::vec2(8.0f, 8.0f), glm::vec3(0.5f, 0.0f, 0.5f), 0.99f, glm::vec3(1.0f, 0.0f, 0.0f), 1.0f),
        makeSplat(glm::vec2(8.0f, 8.0f), glm::vec3(0.5f, 0.0f, 0.5f), 0.99f, glm::vec3(0.0f, 1.0f, 0.0f), 2.0f),
    });
    const std::vector<uint16_t> ranks = buildRanks();

    const double alphaFront = 0.99;
    const double alphaBack = 0.99;
    const glm::dvec3 colorFront(1.0, 0.0, 0.0);
    const glm::dvec3 colorBack(0.0, 1.0, 0.0);
    const glm::dvec3 expected = colorFront * alphaFront + colorBack * alphaBack * (1.0 - alphaFront);

    const size_t pixel = size_t(8) * 16 + 8;
    const uint32_t frames = 400;
    glm::dvec3 mean(0.0);
    for (uint32_t f = 0; f < frames; ++f)
    {
        const glm::vec4 p = render(c, ranks, 4, f)[pixel];
        mean += glm::dvec3(p.r, p.g, p.b);
    }
    mean /= double(frames);

    for (int k = 0; k < 3; ++k)
    {
        UNIT_CHECK_NEAR(ctx, mean[k], expected[k], 0.02);
    }
}

// Same inputs give the same bits. A different frame index moves the noise, so the
// image changes somewhere.
void testDeterminism(Context & ctx)
{
    const study::StudyCandidates c = buildScene();
    const std::vector<uint16_t> ranks = buildRanks();
    for (uint32_t samples : {4u, 8u})
    {
        const std::vector<glm::vec4> a = render(c, ranks, samples, 0);
        const std::vector<glm::vec4> b = render(c, ranks, samples, 0);
        UNIT_CHECK(ctx, sameBits(a, b));

        const std::vector<glm::vec4> next = render(c, ranks, samples, 1);
        UNIT_CHECK(ctx, !sameBits(a, next));
    }
}

// A candidate with alpha below 1/255 is skipped by evaluateSplat, so it never
// covers a sample. It sits in front of the scene, so if it contributed it would
// change the output, and the image must stay bit-identical.
void testFaintCandidateIgnored(Context & ctx)
{
    const study::StudyCandidates base = buildScene();
    study::StudyCandidates withFaint = base;
    withFaint.projected.push_back(
        makeSplat(glm::vec2(8.0f, 8.0f), glm::vec3(0.1f, 0.0f, 0.1f), 0.001f, glm::vec3(1.0f, 1.0f, 1.0f), 0.5f));
    withFaint.sortedValues.push_back(uint32_t(base.projected.size()));
    withFaint.tileRanges = {TileRange {0u, uint32_t(withFaint.sortedValues.size())}};

    const std::vector<uint16_t> ranks = buildRanks();
    for (uint32_t samples : {4u, 8u})
    {
        for (uint32_t frame : {0u, 1u, 2u, 3u})
        {
            UNIT_CHECK(ctx, sameBits(render(base, ranks, samples, frame), render(withFaint, ranks, samples, frame)));
        }
    }
}

// Inputs the model cannot index safely are rejected with std::invalid_argument.
void testValidation(Context & ctx)
{
    const study::StudyCandidates c = buildScene();
    const std::vector<uint16_t> ranks = buildRanks();

    UNIT_CHECK(ctx, throwsInvalid([&] { (void)render(c, std::vector<uint16_t>(4095), 4, 0); }));
    UNIT_CHECK(ctx, throwsInvalid([&] { (void)render(c, ranks, 0, 0); }));
    UNIT_CHECK(ctx, throwsInvalid([&] { (void)render(c, ranks, 5, 0); }));

    study::StudyCandidates noTiles = c;
    noTiles.tileRanges.clear();
    UNIT_CHECK(ctx, throwsInvalid([&] { (void)render(noTiles, ranks, 4, 0); }));

    study::StudyCandidates noFocal = c;
    noFocal.fx = 0.0f;
    UNIT_CHECK(ctx, throwsInvalid([&] { (void)render(noFocal, ranks, 4, 0); }));
}

} // namespace

void testSortStochastic(Context & ctx)
{
    testOneColorPerSample(ctx);
    testConvergesToBlend(ctx);
    testDeterminism(ctx);
    testFaintCandidateIgnored(ctx);
    testValidation(ctx);
}

} // namespace sabercat::unit
