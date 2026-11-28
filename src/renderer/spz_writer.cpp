// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/spz_writer.h"

#include <load-spz.h>
#include <splat-types.h>

#include <glm/glm.hpp>

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "sabercat/gaussian_cloud.h"
#include "sabercat/logger.h"
#include "sabercat/parallel_for.h"
#include "sabercat/ply_writer.h"

// The packed .spz bytes are little-endian by definition; the streams are copied
// as the reference packer lays them out.
static_assert(std::endian::native == std::endian::little,
              "SPZWriter assumes a little-endian host (spz data is little-endian)");

namespace sabercat
{

namespace
{

// spz packs position as 24-bit fixed point with 12 fractional bits.
constexpr double kPositionScale = 4096.0;
constexpr double kPositionMin = -8388608.0; // -2^23
constexpr double kPositionMax = 8388607.0;  // 2^23 - 1

// spz packs log-scale as one byte: b / 16 - 10.
constexpr float kLogScaleMin = -10.0f;
constexpr float kLogScaleMax = 255.0f / 16.0f - 10.0f;

// spz packs DC color as one byte: (c / 255 - 0.5) / 0.15.
constexpr double kDcScale = 0.15;
constexpr double kByteScale = 255.0;
constexpr double kDcOffset = 127.5;

// spz packs SH coefficients as one byte: (b - 128) / 128.
constexpr float kShMin = -1.0f;
constexpr float kShMax = 127.0f / 128.0f;

// Failure kinds, one first-offender slot each. Ordered as the post-pass check.
constexpr size_t kPositionInvalid = 0;
constexpr size_t kScaleInvalid = 1;
constexpr size_t kRotationInvalid = 2;
constexpr size_t kOpacityInvalid = 3;
constexpr size_t kDcInvalid = 4;
constexpr size_t kShInvalid = 5;
constexpr size_t kFailureKinds = 6;

constexpr size_t kNoIndex = std::numeric_limits<size_t>::max();

// Lowers `slot` to `value` if smaller. Makes the reported offender the lowest
// index across workers, so the error message does not depend on thread timing.
void atomicMin(std::atomic<size_t> & slot, size_t value)
{
    size_t current = slot.load(std::memory_order_relaxed);
    while (value < current && !slot.compare_exchange_weak(current, value, std::memory_order_relaxed))
    {
    }
}

void requireSize(size_t actual, size_t expected, const char * name, const std::string & path)
{
    if (actual != expected)
    {
        throw std::runtime_error("SPZWriter: " + std::string(name) + " has "
                                 + std::to_string(actual) + " entries, expected "
                                 + std::to_string(expected)
                                 + " (call downloadFromGPU() if the host copy was dropped): "
                                 + path);
    }
}

// True when v is finite and round(v * 4096) fits spz's 24-bit position field.
inline bool isRepresentablePosition(float v)
{
    if (!std::isfinite(v))
    {
        return false;
    }
    const double q = std::round(static_cast<double>(v) * kPositionScale);
    return q >= kPositionMin && q <= kPositionMax;
}

// Inverse of SpzReader's exp(): stores ln(s), clamped to spz's byte range.
// s <= 0 has no logarithm and maps to the bottom of the range. Each clamp bumps
// `clampedCount`. The caller has rejected non-finite s.
inline float encodeLogScale(float s, size_t & clampedCount)
{
    if (!(s > 0.0f))
    {
        ++clampedCount;
        return kLogScaleMin;
    }
    const float logScale = std::log(s);
    if (logScale < kLogScaleMin)
    {
        ++clampedCount;
        return kLogScaleMin;
    }
    if (logScale > kLogScaleMax)
    {
        ++clampedCount;
        return kLogScaleMax;
    }
    return logScale;
}

// Inverse of SpzReader's sigmoid: logit(p), with p clamped to [eps, 1 - eps].
// Out-of-[0, 1] and clamped values both count toward `clampedCount`. The caller
// has rejected non-finite opacity.
inline float encodeLogit(float opacity, size_t & clampedCount)
{
    const double eps = static_cast<double>(PLYWriter::kLogitEpsilon);
    bool clamped = opacity < 0.0f || opacity > 1.0f;
    double p = std::min(std::max(static_cast<double>(opacity), 0.0), 1.0);
    if (p < eps)
    {
        p = eps;
        clamped = true;
    }
    else if (p > 1.0 - eps)
    {
        p = 1.0 - eps;
        clamped = true;
    }
    clampedCount += clamped ? 1 : 0;
    return static_cast<float>(std::log(p / (1.0 - p)));
}

// True when a DC channel would saturate spz's byte: dc * 0.15 * 255 + 127.5
// outside [0, 255].
inline bool isDcSaturated(float dc)
{
    const double byte = static_cast<double>(dc) * kDcScale * kByteScale + kDcOffset;
    return byte < 0.0 || byte > kByteScale;
}

// Finds the first non-finite SH coefficient of Gaussian i, for the message.
// Returns the SH degree-style index k (1-based, DC excluded) and channel c.
void locateNonFiniteSh(const std::vector<glm::vec4> & shCoefficients, size_t i, size_t coeffs,
                       size_t & k, int & c)
{
    for (k = 1; k <= coeffs; ++k)
    {
        const glm::vec4 & v = shCoefficients[i * kShCoefficientsPerGaussian + k];
        for (c = 0; c < 3; ++c)
        {
            if (!std::isfinite(v[c]))
            {
                return;
            }
        }
    }
    k = 0;
    c = 0;
}

} // namespace

void SPZWriter::write(const GaussianCloud & cloud, const std::string & path, uint32_t version)
{
    if (version != 3 && version != 4)
    {
        throw std::runtime_error("SPZWriter: version " + std::to_string(version)
                                 + " is not supported (use 3 for gzip or 4 for zstd): " + path);
    }
    const size_t count = cloud.count();
    if (count < 1)
    {
        throw std::runtime_error("SPZWriter: cloud is empty, nothing to write: " + path);
    }
    if (count > kMaxGaussians)
    {
        throw std::runtime_error("SPZWriter: " + std::to_string(count)
                                 + " Gaussians exceeds the spz limit of "
                                 + std::to_string(kMaxGaussians) + ": " + path);
    }
    const uint32_t shDegree = cloud.shDegree();
    if (shDegree > kMaxShDegree)
    {
        throw std::runtime_error("SPZWriter: SH degree " + std::to_string(shDegree)
                                 + " is above " + std::to_string(kMaxShDegree)
                                 + " and would be truncated; refusing to write: " + path);
    }
    // Higher-order coefficients per channel: 0 for degree 0, 3 for 1, 8 for 2, 15 for 3.
    const size_t coeffs = static_cast<size_t>((shDegree + 1) * (shDegree + 1) - 1);

    const auto & positions = cloud.positions();
    const auto & scales = cloud.scales();
    const auto & rotations = cloud.rotations();
    const auto & shCoefficients = cloud.shCoefficients();
    const auto & opacities = cloud.opacities();

    requireSize(positions.size(), count, "positions", path);
    requireSize(scales.size(), count, "scales", path);
    requireSize(rotations.size(), count, "rotations", path);
    requireSize(opacities.size(), count, "opacities", path);
    requireSize(shCoefficients.size(), count * kShCoefficientsPerGaussian, "shCoefficients", path);

    // The reference packer's cloud. Every stream is sized once here; the parallel
    // pass below writes disjoint index ranges into it and allocates nothing.
    spz::GaussianCloud packed;
    packed.numPoints = static_cast<int32_t>(count);
    packed.shDegree = static_cast<int32_t>(shDegree);
    packed.antialiased = cloud.antialiased();
    packed.positions.resize(count * 3);
    packed.scales.resize(count * 3);
    packed.rotations.resize(count * 4);
    packed.alphas.resize(count);
    packed.colors.resize(count * 3);
    packed.sh.resize(count * coeffs * 3);

    float * const outPositions = packed.positions.data();
    float * const outScales = packed.scales.data();
    float * const outRotations = packed.rotations.data();
    float * const outAlphas = packed.alphas.data();
    float * const outColors = packed.colors.data();
    float * const outSh = packed.sh.data();

    // Lowest offending index per failure kind, merged across workers.
    std::atomic<size_t> firstBad[kFailureKinds];
    for (std::atomic<size_t> & slot : firstBad)
    {
        slot.store(kNoIndex, std::memory_order_relaxed);
    }
    std::atomic<size_t> scaleClamped{0};
    std::atomic<size_t> opacityClamped{0};
    std::atomic<size_t> rotationDegenerate{0};
    std::atomic<size_t> dcSaturated{0};
    std::atomic<size_t> shSaturated{0};

    // One pass. Invalid values are recorded (first index per kind) rather than
    // thrown mid-loop, so every worker finishes its range and the error names the
    // lowest offender. Invalid values are written as zero to keep the streams
    // well-formed; the cloud is rejected below before anything is saved.
    parallelFor(count,
                [&](size_t begin, size_t end)
                {
                    size_t localFirst[kFailureKinds];
                    std::fill(localFirst, localFirst + kFailureKinds, kNoIndex);
                    size_t nScale = 0;
                    size_t nOpacity = 0;
                    size_t nDegenerate = 0;
                    size_t nDc = 0;
                    size_t nSh = 0;

                    for (size_t i = begin; i < end; ++i)
                    {
                        // Position: xyz, the w lane is padding.
                        const glm::vec4 & p = positions[i];
                        if (isRepresentablePosition(p.x) && isRepresentablePosition(p.y)
                            && isRepresentablePosition(p.z))
                        {
                            outPositions[i * 3 + 0] = p.x;
                            outPositions[i * 3 + 1] = p.y;
                            outPositions[i * 3 + 2] = p.z;
                        }
                        else
                        {
                            if (localFirst[kPositionInvalid] == kNoIndex)
                            {
                                localFirst[kPositionInvalid] = i;
                            }
                            outPositions[i * 3 + 0] = 0.0f;
                            outPositions[i * 3 + 1] = 0.0f;
                            outPositions[i * 3 + 2] = 0.0f;
                        }

                        // Scale: linear in the cloud, log in spz.
                        const glm::vec4 & s = scales[i];
                        for (int c = 0; c < 3; ++c)
                        {
                            if (std::isfinite(s[c]))
                            {
                                outScales[i * 3 + c] = encodeLogScale(s[c], nScale);
                            }
                            else
                            {
                                if (localFirst[kScaleInvalid] == kNoIndex)
                                {
                                    localFirst[kScaleInvalid] = i;
                                }
                                outScales[i * 3 + c] = kLogScaleMin;
                            }
                        }

                        // Rotation: xyzw in both, normalized here so spz's
                        // smallest-three packing sees unit quaternions.
                        const glm::vec4 & q = rotations[i];
                        if (std::isfinite(q.x) && std::isfinite(q.y) && std::isfinite(q.z)
                            && std::isfinite(q.w))
                        {
                            const double lengthSquared =
                                static_cast<double>(q.x) * q.x + static_cast<double>(q.y) * q.y
                                + static_cast<double>(q.z) * q.z + static_cast<double>(q.w) * q.w;
                            if (lengthSquared > 1e-12)
                            {
                                const double invLength = 1.0 / std::sqrt(lengthSquared);
                                outRotations[i * 4 + 0] = static_cast<float>(q.x * invLength);
                                outRotations[i * 4 + 1] = static_cast<float>(q.y * invLength);
                                outRotations[i * 4 + 2] = static_cast<float>(q.z * invLength);
                                outRotations[i * 4 + 3] = static_cast<float>(q.w * invLength);
                            }
                            else
                            {
                                ++nDegenerate;
                                outRotations[i * 4 + 0] = 0.0f;
                                outRotations[i * 4 + 1] = 0.0f;
                                outRotations[i * 4 + 2] = 0.0f;
                                outRotations[i * 4 + 3] = 1.0f;
                            }
                        }
                        else
                        {
                            if (localFirst[kRotationInvalid] == kNoIndex)
                            {
                                localFirst[kRotationInvalid] = i;
                            }
                            outRotations[i * 4 + 0] = 0.0f;
                            outRotations[i * 4 + 1] = 0.0f;
                            outRotations[i * 4 + 2] = 0.0f;
                            outRotations[i * 4 + 3] = 1.0f;
                        }

                        // Opacity: linear in the cloud, logit in spz.
                        const float opacity = opacities[i];
                        if (std::isfinite(opacity))
                        {
                            outAlphas[i] = encodeLogit(opacity, nOpacity);
                        }
                        else
                        {
                            if (localFirst[kOpacityInvalid] == kNoIndex)
                            {
                                localFirst[kOpacityInvalid] = i;
                            }
                            outAlphas[i] = 0.0f;
                        }

                        // DC color (SH coefficient 0) goes straight through; spz
                        // quantizes it. Only saturation is counted here.
                        const glm::vec4 * shIn = &shCoefficients[i * kShCoefficientsPerGaussian];
                        for (int c = 0; c < 3; ++c)
                        {
                            const float dc = shIn[0][c];
                            if (std::isfinite(dc))
                            {
                                nDc += isDcSaturated(dc) ? 1 : 0;
                                outColors[i * 3 + c] = dc;
                            }
                            else
                            {
                                if (localFirst[kDcInvalid] == kNoIndex)
                                {
                                    localFirst[kDcInvalid] = i;
                                }
                                outColors[i * 3 + c] = 0.0f;
                            }
                        }

                        // Higher-order coefficients, coefficient-major with the
                        // channel as the inner axis (sh[k*3 + c], k 1-based): the
                        // layout SpzReader reads.
                        const size_t base = i * coeffs * 3;
                        for (size_t k = 1; k <= coeffs; ++k)
                        {
                            const glm::vec4 & v = shIn[k];
                            for (int c = 0; c < 3; ++c)
                            {
                                const float value = v[c];
                                float * const dst = &outSh[base + (k - 1) * 3 + static_cast<size_t>(c)];
                                if (std::isfinite(value))
                                {
                                    nSh += (value < kShMin || value > kShMax) ? 1 : 0;
                                    *dst = value;
                                }
                                else
                                {
                                    if (localFirst[kShInvalid] == kNoIndex)
                                    {
                                        localFirst[kShInvalid] = i;
                                    }
                                    *dst = 0.0f;
                                }
                            }
                        }
                    }

                    for (size_t kind = 0; kind < kFailureKinds; ++kind)
                    {
                        if (localFirst[kind] != kNoIndex)
                        {
                            atomicMin(firstBad[kind], localFirst[kind]);
                        }
                    }
                    scaleClamped.fetch_add(nScale, std::memory_order_relaxed);
                    opacityClamped.fetch_add(nOpacity, std::memory_order_relaxed);
                    rotationDegenerate.fetch_add(nDegenerate, std::memory_order_relaxed);
                    dcSaturated.fetch_add(nDc, std::memory_order_relaxed);
                    shSaturated.fetch_add(nSh, std::memory_order_relaxed);
                });

    // Reject on the lowest offender of each kind, in the order the fields are
    // checked. The message names the Gaussian and the value, read back from the
    // host cloud.
    const size_t badPosition = firstBad[kPositionInvalid].load(std::memory_order_relaxed);
    if (badPosition != kNoIndex)
    {
        const glm::vec4 & p = positions[badPosition];
        throw std::runtime_error("SPZWriter: Gaussian " + std::to_string(badPosition) + " position ("
                                 + std::to_string(p.x) + ", " + std::to_string(p.y) + ", "
                                 + std::to_string(p.z)
                                 + ") is not finite or lies outside the SPZ range of about +-2048 "
                                   "(24-bit fixed point, 12 fractional bits): " + path);
    }
    const size_t badScale = firstBad[kScaleInvalid].load(std::memory_order_relaxed);
    if (badScale != kNoIndex)
    {
        const glm::vec4 & s = scales[badScale];
        throw std::runtime_error("SPZWriter: Gaussian " + std::to_string(badScale) + " scale ("
                                 + std::to_string(s.x) + ", " + std::to_string(s.y) + ", "
                                 + std::to_string(s.z) + ") is not finite: " + path);
    }
    const size_t badRotation = firstBad[kRotationInvalid].load(std::memory_order_relaxed);
    if (badRotation != kNoIndex)
    {
        throw std::runtime_error("SPZWriter: Gaussian " + std::to_string(badRotation)
                                 + " rotation has a non-finite component: " + path);
    }
    const size_t badOpacity = firstBad[kOpacityInvalid].load(std::memory_order_relaxed);
    if (badOpacity != kNoIndex)
    {
        throw std::runtime_error("SPZWriter: Gaussian " + std::to_string(badOpacity)
                                 + " opacity is not finite (" + std::to_string(opacities[badOpacity])
                                 + "): " + path);
    }
    const size_t badDc = firstBad[kDcInvalid].load(std::memory_order_relaxed);
    if (badDc != kNoIndex)
    {
        throw std::runtime_error("SPZWriter: Gaussian " + std::to_string(badDc)
                                 + " DC color (SH coefficient 0) is not finite: " + path);
    }
    const size_t badSh = firstBad[kShInvalid].load(std::memory_order_relaxed);
    if (badSh != kNoIndex)
    {
        size_t k = 0;
        int c = 0;
        locateNonFiniteSh(shCoefficients, badSh, coeffs, k, c);
        throw std::runtime_error("SPZWriter: Gaussian " + std::to_string(badSh) + " SH coefficient "
                                 + std::to_string(k) + " channel " + std::to_string(c)
                                 + " is not finite (degree " + std::to_string(shDegree) + "): "
                                 + path);
    }

    const size_t nScale = scaleClamped.load(std::memory_order_relaxed);
    const size_t nOpacity = opacityClamped.load(std::memory_order_relaxed);
    const size_t nDegenerate = rotationDegenerate.load(std::memory_order_relaxed);
    const size_t nDc = dcSaturated.load(std::memory_order_relaxed);
    const size_t nSh = shSaturated.load(std::memory_order_relaxed);
    if (nScale + nOpacity + nDegenerate + nDc + nSh > 0)
    {
        Logger::warning("SPZWriter: values were adjusted to fit the spz format -- "
                        + std::to_string(nScale) + " scale(s) clamped to ln(s) in ["
                        + std::to_string(kLogScaleMin) + ", " + std::to_string(kLogScaleMax) + "], "
                        + std::to_string(nOpacity) + " opacity value(s) clamped to ["
                        + std::to_string(PLYWriter::kLogitEpsilon) + ", 1 - eps], "
                        + std::to_string(nDegenerate) + " degenerate rotation(s) written as identity, "
                        + std::to_string(nDc) + " DC color value(s) saturated, "
                        + std::to_string(nSh) + " SH coefficient value(s) saturated: " + path);
    }

    spz::PackOptions options;
    options.version = version;
    options.from = spz::CoordinateSystem::RUB;
    std::vector<uint8_t> bytes;
    if (!spz::saveSpz(packed, options, &bytes))
    {
        throw std::runtime_error("SPZWriter: spz failed to pack the cloud (version "
                                 + std::to_string(version) + "): " + path);
    }

    // Via a temporary file renamed over the target, so a failed write never
    // leaves a truncated .spz in place of a good one.
    const std::string tmpPath = path + ".tmp";
    const std::filesystem::path target = std::filesystem::u8path(path);
    const std::filesystem::path tmpFile = std::filesystem::u8path(tmpPath);
    try
    {
        std::ofstream file(tmpFile, std::ios::binary | std::ios::trunc);
        if (!file)
        {
            throw std::runtime_error("SPZWriter: failed to open " + tmpPath + " for writing");
        }
        file.write(reinterpret_cast<const char *>(bytes.data()),
                   static_cast<std::streamsize>(bytes.size()));
        file.close();
        if (!file)
        {
            throw std::runtime_error("SPZWriter: failed to write " + tmpPath);
        }

        std::error_code ec;
        std::filesystem::rename(tmpFile, target, ec);
        if (ec)
        {
            throw std::runtime_error("SPZWriter: failed to rename " + tmpPath + " over " + path
                                     + ": " + ec.message());
        }
    }
    catch (...)
    {
        std::error_code ignored;
        std::filesystem::remove(tmpFile, ignored);
        throw;
    }
}

} // namespace sabercat
