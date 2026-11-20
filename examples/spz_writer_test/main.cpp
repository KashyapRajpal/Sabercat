// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Headless test for SPZWriter, the exact inverse of SpzReader. Everything is
// checked by writing a .spz and reading it back through SpzReader, except the
// byte-level check in extra_checks.cpp, which inspects the file with the
// reference library directly.
//
//  (a) random clouds (seeded) of 2000 Gaussians at SH degree 0..3, versions 3
//      and 4, antialiased off and on: every attribute within its quantizer
//      tolerance (worst error per attribute printed); SH slots above the written
//      degree read back exactly zero; the reader reports the written degree.
//  (b) idempotence: write(read(write(x))) decodes to the same streams as
//      read(write(x)) -- positions, scales, opacities and SH bit for bit, and
//      rotations bit for bit or equal as rotations (see testIdempotence).
//  (c) single splat: identity rotation, each of the four largest-component
//      indices, q against -q, a zero quaternion (identity), a non-unit one.
//  (e) clamps and saturation: scale 0 / negative / 1e6, opacity outside [0, 1],
//      DC and SH beyond the quantizer range: no throw, values at the limits.
//  (f) position boundary: 2047.99 and -2048.0 are accepted and round-trip.
//  (g) a non-ASCII directory and file name; overwriting an existing file.
//  (h) a 1,000,000-Gaussian degree-3 cloud: timed version 3 write, read back,
//      1000 random indices checked against the tolerances.
//
// (d) refusals and (i) the byte-level spec check are in extra_checks.cpp.
// Validation messages are logged as "[validation]"; the run must have none.

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "sabercat/device.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/render_config.h"
#include "sabercat/spz_reader.h"
#include "sabercat/spz_writer.h"

using namespace sabercat;

// extra_checks.cpp: writer refusals (d) and the byte-level spec check (i).
int runExtraChecks(Device & device, const std::filesystem::path & dir);

namespace
{

// Each tolerance is the largest error the spz quantizer can make on the value,
// plus float rounding (about 1e-7 relative). The quantizer is spz's own
// (external/spz/src/cc/load-spz.cc); the derivation is in each comment.

// Position: round(x * 4096), so the error is at most half a step, 0.5 / 4096 = 1.22e-4.
constexpr double kPositionTolerance = 1.3e-4;
// Log scale: clamp(round((ln s + 10) * 16)), so at most half a step, 0.5 / 16 = 0.03125.
constexpr double kLogScaleTolerance = 0.5 / 16.0 + 1e-4;
// Rotation: smallest-three with 9 bits per stored component; 2e-6 is the bound on
// 1 - |dot| that the project states for this format.
constexpr double kRotationTolerance = 2e-6;
// Opacity: round(sigmoid(logit(p)) * 255), so at most 0.5 / 255 = 1.96e-3.
constexpr double kOpacityTolerance = 2.0e-3;
// DC: round(dc * 0.15 * 255 + 127.5), so at most 0.5 / (0.15 * 255) = 0.01307 for
// |dc| <= 3.3 (no saturation in that range).
constexpr double kDcTolerance = 0.0132;
// SH degree 1 (slots 1..3, bucket 8 in 1/128 units): the bucket moves q by -3..+4
// units, and round(x * 128) adds up to 0.5 unit, so the worst case is 4.5 / 128
// = 0.03516. (The spz bucket rounding is not symmetric, which is why this is not
// the 4 / 128 half bucket.)
constexpr double kShDegree1Tolerance = 4.5 / 128.0 + 1e-4;
// SH rest (slots 4..15, bucket 16): the bucket moves q by -7..+8 units plus the
// 0.5 unit of rounding, so the worst case is 8.5 / 128 = 0.06641.
constexpr double kShRestTolerance = 8.5 / 128.0 + 1e-4;
// Values at a quantizer limit read back as exact multiples of 1/128 or of the DC
// step, so the only error is float rounding.
constexpr double kSaturationTolerance = 1e-4;

int g_failures = 0;

void check(bool ok, const std::string & what)
{
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok)
    {
        ++g_failures;
    }
}

std::filesystem::path g_tempDir;

static_assert(kShCoefficientsPerGaussian == 16, "the SH layout below assumes degree 3 (16 slots)");

// Bit-for-bit equality of two streams (the same elements, the same bytes).
template <typename T> bool bitEqual(const std::vector<T> & a, const std::vector<T> & b)
{
    return a.size() == b.size()
           && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

glm::quat randomRotation(std::mt19937 & rng)
{
    std::normal_distribution<float> normal(0.0f, 1.0f);
    const float w = normal(rng);
    const float x = normal(rng);
    const float y = normal(rng);
    const float z = normal(rng);
    return glm::normalize(glm::quat(w, x, y, z));
}

// A cloud of `count` random Gaussians, every value inside the quantizer range:
// positions in +-100, log scales in [-9.9, 5.9], opacity in [0.01, 0.99], DC in
// +-3.2, SH in +-0.99, and zero above the degree.
std::shared_ptr<GaussianCloud> makeCloud(Device & device, size_t count, uint32_t degree,
                                         uint32_t seed, bool antialiased)
{
    auto cloud = std::make_shared<GaussianCloud>(device);
    cloud->initialize(count);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    std::uniform_real_distribution<float> position(-100.0f, 100.0f);
    std::uniform_real_distribution<float> logScale(-9.9f, 5.9f);
    std::uniform_real_distribution<float> opacity(0.01f, 0.99f);
    std::uniform_real_distribution<float> dc(-3.2f, 3.2f);
    const size_t kept = static_cast<size_t>((degree + 1) * (degree + 1));
    for (size_t i = 0; i < count; ++i)
    {
        const float px = position(rng);
        const float py = position(rng);
        const float pz = position(rng);
        cloud->positions()[i] = glm::vec4(px, py, pz, 0.0f);
        const float sx = std::exp(logScale(rng));
        const float sy = std::exp(logScale(rng));
        const float sz = std::exp(logScale(rng));
        cloud->scales()[i] = glm::vec4(sx, sy, sz, 0.0f);
        const glm::quat q = randomRotation(rng);
        cloud->rotations()[i] = glm::vec4(q.x, q.y, q.z, q.w);
        cloud->opacities()[i] = opacity(rng);
        for (size_t k = 0; k < kShCoefficientsPerGaussian; ++k)
        {
            glm::vec4 & c = cloud->shCoefficients()[i * kShCoefficientsPerGaussian + k];
            if (k == 0)
            {
                const float r = dc(rng);
                const float g = dc(rng);
                const float b = dc(rng);
                c = glm::vec4(r, g, b, 0.0f);
            }
            else if (k < kept)
            {
                const float r = 0.99f * unit(rng);
                const float g = 0.99f * unit(rng);
                const float b = 0.99f * unit(rng);
                c = glm::vec4(r, g, b, 0.0f);
            }
            else
            {
                c = glm::vec4(0.0f);
            }
        }
    }
    cloud->setShDegree(degree);
    cloud->setAntialiased(antialiased);
    cloud->computeBounds();
    return cloud;
}

// A cloud of `count` Gaussians with neutral values, for the tests that set a few
// fields by hand: unit scale 0.1, identity rotation, opacity 0.5, zero SH, degree 3.
std::shared_ptr<GaussianCloud> makeBase(Device & device, size_t count)
{
    auto cloud = std::make_shared<GaussianCloud>(device);
    cloud->initialize(count);
    for (size_t i = 0; i < count; ++i)
    {
        cloud->positions()[i] = glm::vec4(0.0f);
        cloud->scales()[i] = glm::vec4(0.1f, 0.1f, 0.1f, 0.0f);
        cloud->rotations()[i] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        cloud->opacities()[i] = 0.5f;
    }
    for (glm::vec4 & c : cloud->shCoefficients())
    {
        c = glm::vec4(0.0f);
    }
    cloud->setShDegree(3);
    cloud->setAntialiased(false);
    cloud->computeBounds();
    return cloud;
}

// Writes `source` as `<name>.spz` under the temp directory, checks that no
// temporary file is left behind, and reads it back through SpzReader.
std::shared_ptr<GaussianCloud> writeAndRead(Device & device, const GaussianCloud & source,
                                            const std::string & name,
                                            uint32_t version = SPZWriter::kDefaultVersion)
{
    const std::string path = (g_tempDir / (name + ".spz")).string();
    SPZWriter::write(source, path, version);
    check(!std::filesystem::exists(path + ".tmp"), name + ": no temporary file left behind");
    return SpzReader::read(device, path);
}

// Worst error per attribute over a set of Gaussians.
struct Worst
{
    double position = 0.0;
    double logScale = 0.0;
    double rotation = 0.0;
    double opacity = 0.0;
    double dc = 0.0;
    double shDegree1 = 0.0;
    double shRest = 0.0;

    void merge(const Worst & other)
    {
        position = std::max(position, other.position);
        logScale = std::max(logScale, other.logScale);
        rotation = std::max(rotation, other.rotation);
        opacity = std::max(opacity, other.opacity);
        dc = std::max(dc, other.dc);
        shDegree1 = std::max(shDegree1, other.shDegree1);
        shRest = std::max(shRest, other.shRest);
    }

    bool within() const
    {
        return position <= kPositionTolerance && logScale <= kLogScaleTolerance
               && rotation <= kRotationTolerance && opacity <= kOpacityTolerance
               && dc <= kDcTolerance && shDegree1 <= kShDegree1Tolerance
               && shRest <= kShRestTolerance;
    }

    std::string str() const
    {
        char buffer[400];
        std::snprintf(buffer, sizeof(buffer),
                      "position %.3e, log-scale %.3e, rotation %.3e, opacity %.3e, DC %.3e, "
                      "SH1 %.3e, SHrest %.3e",
                      position, logScale, rotation, opacity, dc, shDegree1, shRest);
        return buffer;
    }
};

// 1 - |dot| of the unit quaternions: zero for the same rotation, q and -q alike.
double rotationError(const glm::vec4 & expected, const glm::vec4 & got)
{
    const glm::dvec4 e = glm::normalize(glm::dvec4(expected));
    const glm::dvec4 g = glm::normalize(glm::dvec4(got));
    return std::max(0.0, 1.0 - std::abs(glm::dot(e, g)));
}

// Folds the error of Gaussian `i` of `read` against `source` into `worst`.
void accumulate(Worst & worst, const GaussianCloud & source, const GaussianCloud & read, size_t i)
{
    const glm::vec4 p0 = source.positions()[i];
    const glm::vec4 p1 = read.positions()[i];
    const glm::vec4 s0 = source.scales()[i];
    const glm::vec4 s1 = read.scales()[i];
    for (int a = 0; a < 3; ++a)
    {
        worst.position = std::max(worst.position, std::abs(static_cast<double>(p0[a]) - p1[a]));
        worst.logScale = std::max(worst.logScale,
                                  std::abs(std::log(static_cast<double>(s1[a]))
                                           - std::log(static_cast<double>(s0[a]))));
    }

    worst.rotation = std::max(worst.rotation,
                              rotationError(source.rotations()[i], read.rotations()[i]));

    worst.opacity = std::max(worst.opacity,
                             std::abs(static_cast<double>(source.opacities()[i])
                                      - read.opacities()[i]));

    const size_t base = i * kShCoefficientsPerGaussian;
    for (int c = 0; c < 3; ++c)
    {
        worst.dc = std::max(worst.dc,
                            std::abs(static_cast<double>(source.shCoefficients()[base][c])
                                     - read.shCoefficients()[base][c]));
    }
    for (size_t slot = 1; slot < kShCoefficientsPerGaussian; ++slot)
    {
        for (int c = 0; c < 3; ++c)
        {
            const double error = std::abs(static_cast<double>(source.shCoefficients()[base + slot][c])
                                          - read.shCoefficients()[base + slot][c]);
            if (slot <= 3)
            {
                worst.shDegree1 = std::max(worst.shDegree1, error);
            }
            else
            {
                worst.shRest = std::max(worst.shRest, error);
            }
        }
    }
}

// Test (a): random clouds at every degree, version and antialiased setting.
void testRandomClouds(Device & device)
{
    std::printf("\n-- (a) random clouds: every attribute within its tolerance --\n");
    Worst overall;
    for (uint32_t version : {3u, 4u})
    {
        for (uint32_t degree = 0; degree <= 3; ++degree)
        {
            for (bool antialiased : {false, true})
            {
                const std::string label = "version " + std::to_string(version) + ", degree "
                                          + std::to_string(degree) + ", antialiased "
                                          + (antialiased ? "on" : "off");
                const uint32_t seed = 1000 + version * 100 + degree * 10 + (antialiased ? 1u : 0u);
                auto source = makeCloud(device, 2000, degree, seed, antialiased);
                auto read = writeAndRead(device, *source,
                                         "random_v" + std::to_string(version) + "_d"
                                             + std::to_string(degree) + (antialiased ? "_aa" : "_plain"),
                                         version);

                check(read->count() == source->count(), label + ": count matches");
                check(read->shDegree() == degree, label + ": reader reports the written SH degree");
                check(read->antialiased() == antialiased, label + ": antialiased flag round-trips");

                Worst combo;
                for (size_t i = 0; i < source->count(); ++i)
                {
                    accumulate(combo, *source, *read, i);
                }
                overall.merge(combo);
                check(combo.within(), label + ": within tolerance; worst " + combo.str());

                const size_t kept = static_cast<size_t>((degree + 1) * (degree + 1));
                bool zeroAbove = true;
                for (size_t i = 0; i < read->count(); ++i)
                {
                    for (size_t slot = kept; slot < kShCoefficientsPerGaussian; ++slot)
                    {
                        const glm::vec4 & c = read->shCoefficients()[i * kShCoefficientsPerGaussian + slot];
                        zeroAbove = zeroAbove && c.x == 0.0f && c.y == 0.0f && c.z == 0.0f;
                    }
                }
                check(zeroAbove, label + ": SH slots above the written degree read back as exactly zero");
            }
        }
    }
    std::printf("  worst over all combinations: %s\n", overall.str().c_str());
    std::printf("  tolerances: position %.3e, log-scale %.3e, rotation %.1e, opacity %.3e, "
                "DC %.4f, SH1 %.5f, SHrest %.5f\n",
                kPositionTolerance, kLogScaleTolerance, kRotationTolerance, kOpacityTolerance,
                kDcTolerance, kShDegree1Tolerance, kShRestTolerance);
}

// Test (b): write(read(write(x))) against read(write(x)). Positions, scales,
// opacities and SH must be bit-identical. Rotations are compared bit for bit
// too, but a second encode can break a near-tie between two largest components
// (their magnitudes agree to about the quantizer step), so a rotation that differs
// in bits must still be the same rotation to kRotationTolerance. The number of such
// bit-level differences is printed.
void testIdempotence(Device & device)
{
    std::printf("\n-- (b) idempotence: write(read(write(x))) == read(write(x)) --\n");
    for (uint32_t version : {3u, 4u})
    {
        const std::string label = "version " + std::to_string(version) + ": ";
        auto x = makeCloud(device, 2000, 3, 2000 + version, true);
        auto once = writeAndRead(device, *x, "idem_once_v" + std::to_string(version), version);
        auto twice = writeAndRead(device, *once, "idem_twice_v" + std::to_string(version), version);

        check(twice->count() == once->count() && twice->shDegree() == once->shDegree()
                  && twice->antialiased() == once->antialiased(),
              label + "count, SH degree and antialiased match");
        check(bitEqual(twice->positions(), once->positions()), label + "positions bit-identical");
        check(bitEqual(twice->scales(), once->scales()), label + "scales bit-identical");
        check(bitEqual(twice->opacities(), once->opacities()), label + "opacities bit-identical");
        check(bitEqual(twice->shCoefficients(), once->shCoefficients()),
              label + "SH bit-identical");

        size_t bitMismatches = 0;
        double worstRotation = 0.0;
        for (size_t i = 0; i < once->count(); ++i)
        {
            if (std::memcmp(&once->rotations()[i], &twice->rotations()[i], sizeof(glm::vec4)) != 0)
            {
                ++bitMismatches;
                worstRotation = std::max(worstRotation,
                                         rotationError(once->rotations()[i], twice->rotations()[i]));
            }
        }
        std::printf("  %s%zu of %zu rotations differ in bits after the second encode "
                    "(worst 1-|dot| %.3e)\n",
                    label.c_str(), bitMismatches, once->count(), worstRotation);
        check(worstRotation <= kRotationTolerance,
              label + "rotations identical, or the same rotation within tolerance");
    }
}

// Writes a single Gaussian with the given rotation (degree 0) and reads it back.
std::shared_ptr<GaussianCloud> singleSplatRoundTrip(Device & device, const glm::vec4 & rotation,
                                                    const std::string & name,
                                                    std::shared_ptr<GaussianCloud> * sourceOut = nullptr)
{
    auto cloud = makeBase(device, 1);
    cloud->positions()[0] = glm::vec4(0.5f, -0.25f, 1.0f, 0.0f);
    cloud->scales()[0] = glm::vec4(0.1f, 0.2f, 0.3f, 0.0f);
    cloud->rotations()[0] = rotation;
    cloud->setShDegree(0);
    cloud->computeBounds();
    if (sourceOut != nullptr)
    {
        *sourceOut = cloud;
    }
    return writeAndRead(device, *cloud, name);
}

// A unit quaternion whose largest-magnitude component is `largest`.
glm::vec4 quatWithLargest(int largest)
{
    const float others[3] = {0.31f, -0.42f, 0.17f};
    glm::vec4 q(0.0f);
    int j = 0;
    for (int c = 0; c < 4; ++c)
    {
        if (c == largest)
        {
            q[c] = 0.85f;
        }
        else
        {
            q[c] = others[j];
            ++j;
        }
    }
    return glm::normalize(q);
}

// Test (c): single splats and the rotation encoding.
void testSingleSplat(Device & device)
{
    std::printf("\n-- (c) single splat and rotations --\n");

    std::shared_ptr<GaussianCloud> source;
    auto identity = singleSplatRoundTrip(device, glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), "single_identity",
                                         &source);
    check(identity->count() == 1, "single splat: count is 1");
    Worst worst;
    accumulate(worst, *source, *identity, 0);
    check(worst.within(), "single splat (identity): every attribute within tolerance; worst " + worst.str());
    check(rotationError(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), identity->rotations()[0]) <= kRotationTolerance,
          "single splat (identity): rotation is identity");

    for (int largest = 0; largest < 4; ++largest)
    {
        const glm::vec4 q = quatWithLargest(largest);
        const std::string label = "rotation with largest component " + std::to_string(largest);
        auto out = singleSplatRoundTrip(device, q, "single_largest_" + std::to_string(largest));
        const double error = rotationError(q, out->rotations()[0]);
        check(error <= kRotationTolerance, label + ": decoded within tolerance (1-|dot| " + std::to_string(error) + ")");

        auto negated = singleSplatRoundTrip(device, -q, "single_negated_" + std::to_string(largest));
        check(rotationError(out->rotations()[0], negated->rotations()[0]) <= kRotationTolerance,
              label + ": q and -q decode to the same rotation");
    }

    auto zero = singleSplatRoundTrip(device, glm::vec4(0.0f), "single_zero_quaternion");
    check(rotationError(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), zero->rotations()[0]) <= kRotationTolerance,
          "zero quaternion is written as the identity");

    const glm::vec4 unnormalized(1.0f, 2.0f, 3.0f, 4.0f);
    auto nonUnit = singleSplatRoundTrip(device, unnormalized, "single_non_unit");
    check(rotationError(unnormalized, nonUnit->rotations()[0]) <= kRotationTolerance,
          "non-unit quaternion (1,2,3,4) is normalized before packing");

    auto scaledIdentity = singleSplatRoundTrip(device, glm::vec4(0.0f, 0.0f, 0.0f, 3.0f), "single_scaled_identity");
    check(rotationError(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), scaledIdentity->rotations()[0]) <= kRotationTolerance,
          "non-unit quaternion (0,0,0,3) is normalized to the identity");
}

// Test (e): values outside the quantizer range clamp or saturate; nothing throws.
void testClamps(Device & device)
{
    std::printf("\n-- (e) clamps and saturation --\n");

    // Scale: zero, negative, huge. Clamped to the byte range, ln in [-10, 5.9375].
    {
        auto cloud = makeBase(device, 3);
        const float scales[3] = {0.0f, -2.0f, 1.0e6f};
        for (size_t i = 0; i < 3; ++i)
        {
            cloud->scales()[i] = glm::vec4(scales[i], scales[i], scales[i], 0.0f);
        }
        auto read = writeAndRead(device, *cloud, "clamp_scale");
        check(read->count() == 3, "scale clamps: written without throwing");
        const double low = -10.0;
        const double high = 5.9375;
        for (size_t i = 0; i < 2; ++i)
        {
            for (int a = 0; a < 3; ++a)
            {
                const double logValue = std::log(static_cast<double>(read->scales()[i][a]));
                check(std::abs(logValue - low) <= kLogScaleTolerance,
                      "scale " + std::to_string(scales[i]) + " reads back at exp(-10)");
            }
        }
        for (int a = 0; a < 3; ++a)
        {
            const double logValue = std::log(static_cast<double>(read->scales()[2][a]));
            check(std::abs(logValue - high) <= kLogScaleTolerance,
                  "scale 1e6 reads back at exp(5.9375)");
        }
    }

    // Opacity: outside [0, 1] clamps to the ends.
    {
        auto cloud = makeBase(device, 4);
        const float opacities[4] = {0.0f, 1.0f, 1.5f, -0.2f};
        for (size_t i = 0; i < 4; ++i)
        {
            cloud->opacities()[i] = opacities[i];
        }
        auto read = writeAndRead(device, *cloud, "clamp_opacity");
        check(read->opacities()[0] >= 0.0f && read->opacities()[0] <= 0.003f,
              "opacity 0 reads back <= 0.003");
        check(read->opacities()[1] >= 0.997f && read->opacities()[1] <= 1.0f,
              "opacity 1 reads back >= 0.997");
        check(read->opacities()[2] >= 0.997f && read->opacities()[2] <= 1.0f,
              "opacity 1.5 reads back >= 0.997 and within [0, 1]");
        check(read->opacities()[3] >= 0.0f && read->opacities()[3] <= 0.003f,
              "opacity -0.2 reads back <= 0.003 and within [0, 1]");
    }

    // DC and SH beyond the quantizer range saturate at the limits.
    {
        auto cloud = makeBase(device, 1);
        cloud->shCoefficients()[0] = glm::vec4(10.0f, -10.0f, 0.5f, 0.0f);
        cloud->shCoefficients()[1] = glm::vec4(5.0f, -5.0f, 0.5f, 0.0f);
        cloud->shCoefficients()[15] = glm::vec4(5.0f, -5.0f, 0.5f, 0.0f);
        auto read = writeAndRead(device, *cloud, "clamp_dc_sh");

        const double dcHigh = (255.0 - 127.5) / (0.15 * 255.0);
        const double shHigh = 127.0 / 128.0;
        const double shLow = -1.0;
        const glm::vec4 dc = read->shCoefficients()[0];
        check(std::abs(dc.x - dcHigh) <= kSaturationTolerance, "DC 10 saturates at (255-127.5)/(0.15*255)");
        check(std::abs(dc.y + dcHigh) <= kSaturationTolerance, "DC -10 saturates at -(255-127.5)/(0.15*255)");
        check(std::abs(dc.z - 0.5) <= kDcTolerance, "DC 0.5 (in range) reads back within the DC tolerance");

        const glm::vec4 slot1 = read->shCoefficients()[1];
        check(std::abs(slot1.x - shHigh) <= kSaturationTolerance, "SH slot 1 = 5.0 saturates at 127/128");
        check(std::abs(slot1.y - shLow) <= kSaturationTolerance, "SH slot 1 = -5.0 saturates at -1");
        check(std::abs(slot1.z - 0.5) <= kShDegree1Tolerance, "SH slot 1 = 0.5 (in range) within the degree-1 tolerance");

        const glm::vec4 slot15 = read->shCoefficients()[15];
        check(std::abs(slot15.x - shHigh) <= kSaturationTolerance, "SH slot 15 = 5.0 saturates at 127/128");
        check(std::abs(slot15.y - shLow) <= kSaturationTolerance, "SH slot 15 = -5.0 saturates at -1");
        check(std::abs(slot15.z - 0.5) <= kShRestTolerance, "SH slot 15 = 0.5 (in range) within the rest tolerance");
    }
}

// Test (f): the position range edges.
void testPositionBoundary(Device & device)
{
    std::printf("\n-- (f) position boundary --\n");
    auto cloud = makeBase(device, 2);
    cloud->positions()[0] = glm::vec4(2047.99f, -2048.0f, 2047.99f, 0.0f);
    cloud->positions()[1] = glm::vec4(-2048.0f, 2047.99f, -2047.99f, 0.0f);
    auto read = writeAndRead(device, *cloud, "boundary");
    check(read->count() == 2, "position boundary: both splats accepted");

    double worst = 0.0;
    for (size_t i = 0; i < 2; ++i)
    {
        for (int a = 0; a < 3; ++a)
        {
            worst = std::max(worst, std::abs(static_cast<double>(cloud->positions()[i][a])
                                             - read->positions()[i][a]));
        }
    }
    std::printf("  worst position error at the boundary: %.3e\n", worst);
    check(worst <= kPositionTolerance, "position boundary: round-trip within tolerance");
    check(read->positions()[0].y == -2048.0f && read->positions()[1].x == -2048.0f,
          "position -2048.0 (exactly representable) reads back exactly");
}

// Test (g): a non-ASCII directory and file name, and overwriting an existing file.
void testNonAsciiPathAndOverwrite(Device & device)
{
    std::printf("\n-- (g) non-ASCII path and overwrite --\n");
    // "Ünïcödé 層" and "проект.spz" as UTF-8 byte escapes.
    const std::string unicodeDir = "\xC3\x9C"
                                   "n"
                                   "\xC3\xAF"
                                   "c"
                                   "\xC3\xB6"
                                   "d"
                                   "\xC3\xA9"
                                   " \xE5\xB1\xA4";
    const std::string cyrillicFile = "\xD0\xBF\xD1\x80\xD0\xBE\xD0\xB5\xD0\xBA\xD1\x82.spz";
    const std::filesystem::path dir = g_tempDir / std::filesystem::u8path(unicodeDir);
    std::filesystem::create_directories(dir);
    const std::filesystem::path file = dir / std::filesystem::u8path(cyrillicFile);
    const std::u8string fileUtf8 = file.u8string();
    const std::string path(reinterpret_cast<const char *>(fileUtf8.data()), fileUtf8.size());

    auto first = makeCloud(device, 300, 1, 700, false);
    SPZWriter::write(*first, path);
    check(std::filesystem::exists(file), "non-ASCII path: .spz exists under its UTF-8 name");
    check(!std::filesystem::exists(std::filesystem::u8path(path + ".tmp")),
          "non-ASCII path: no temporary file left behind");
    auto readFirst = SpzReader::read(device, path);
    check(readFirst->count() == first->count(), "non-ASCII path: count matches");

    // Overwrite the same file with a different cloud.
    auto second = makeCloud(device, 500, 2, 701, true);
    SPZWriter::write(*second, path);
    check(!std::filesystem::exists(std::filesystem::u8path(path + ".tmp")),
          "overwrite: no temporary file left behind");
    auto readSecond = SpzReader::read(device, path);
    check(readSecond->count() == 500 && readSecond->shDegree() == 2 && readSecond->antialiased(),
          "overwrite: the file now holds the second cloud (count, degree, antialiased)");
    Worst worst;
    for (size_t i = 0; i < second->count(); ++i)
    {
        accumulate(worst, *second, *readSecond, i);
    }
    check(worst.within(), "overwrite: second cloud within tolerance; worst " + worst.str());
    std::filesystem::remove_all(dir);
}

// Test (h): a 1,000,000-Gaussian degree-3 cloud, timed.
void testMillion(Device & device)
{
    std::printf("\n-- (h) 1,000,000 Gaussians, degree 3 --\n");
    auto source = makeCloud(device, 1000000, 3, 9000, false);
    const std::string path = (g_tempDir / "million.spz").string();

    const auto start = std::chrono::steady_clock::now();
    SPZWriter::write(*source, path, 3);
    const auto stop = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(stop - start).count();
    const auto bytes = std::filesystem::file_size(path);
    std::printf("  write (version 3): %.3f s, %llu bytes\n", seconds,
                static_cast<unsigned long long>(bytes));
    check(!std::filesystem::exists(path + ".tmp"), "1M: no temporary file left behind");

    auto read = SpzReader::read(device, path);
    check(read->count() == 1000000 && read->shDegree() == 3, "1M: count and SH degree read back");

    std::mt19937 rng(9001);
    std::uniform_int_distribution<size_t> pick(0, 999999);
    Worst worst;
    for (int n = 0; n < 1000; ++n)
    {
        accumulate(worst, *source, *read, pick(rng));
    }
    std::printf("  worst over 1000 random indices: %s\n", worst.str().c_str());
    check(worst.within(), "1M: 1000 random indices within tolerance");
    std::filesystem::remove(path);
}

} // namespace

int main()
{
    g_tempDir = std::filesystem::temp_directory_path() / "sabercat_spz_writer_test";
    std::filesystem::create_directories(g_tempDir);

    try
    {
        Device device;
        device.initialize({}, /*enableValidation=*/true);
        testRandomClouds(device);
        testIdempotence(device);
        testSingleSplat(device);
        testClamps(device);
        testPositionBoundary(device);
        testNonAsciiPathAndOverwrite(device);
        testMillion(device);
        g_failures += runExtraChecks(device, g_tempDir);
        device.shutdown();
    }
    catch (const std::exception & e)
    {
        std::printf("[FAIL] unexpected exception: %s\n", e.what());
        ++g_failures;
    }

    std::filesystem::remove_all(g_tempDir);
    std::printf("\n%s (%d failure%s)\n",
                g_failures == 0 ? "spz_writer_test OK" : "spz_writer_test FAILED",
                g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}
