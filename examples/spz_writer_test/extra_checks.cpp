// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Further SPZWriter checks, run from main.cpp:
//
//  (d) Refusals. Each is a std::runtime_error, and its message names the
//      Gaussian where there is one: zero splats; NaN and inf in position, scale,
//      rotation, opacity, DC and an SH coefficient; positions 2048.0, -2049 and
//      1e6; SH degree 4; versions 2 and 5; host vectors of the wrong size. A
//      refused write leaves no file and no temporary file, and a file already at
//      the path is left untouched.
//  (i) Byte-level spec check. A hand-built two-splat degree-3 cloud is written as
//      version 3 and inspected with the reference library directly
//      (spz::loadSpzPacked gunzips and unpacks the streams; SpzReader is not
//      involved): the header fields, then each byte stream against the packing
//      formula. Version 4 must unpack to the same streams.

#include <load-spz.h>
#include <splat-types.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "sabercat/device.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/render_config.h"
#include "sabercat/spz_writer.h"

using namespace sabercat;
namespace fs = std::filesystem;

namespace
{

// The rotation bytes are decoded to within this of the normalized input per
// component. One stored step is sqrt(0.5) / 511 = 1.38e-3 (so +-6.9e-4 after
// rounding); the largest component is recovered from the other three, which adds
// at most about three of those, so 2e-3 holds with margin.
constexpr double kRotationByteTolerance = 2e-3;

int g_failures = 0;
fs::path g_dir;

void check(bool ok, const std::string & what)
{
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok)
    {
        ++g_failures;
    }
}

fs::path caseFile(const std::string & tag)
{
    return g_dir / ("refused_" + tag + ".spz");
}

std::vector<char> readBytes(const fs::path & path)
{
    std::ifstream file(path, std::ios::binary);
    return std::vector<char>(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

// A valid degree-3 cloud of `count` Gaussians: every field finite and in range.
std::shared_ptr<GaussianCloud> validCloud(Device & device, size_t count)
{
    auto cloud = std::make_shared<GaussianCloud>(device);
    cloud->initialize(count);
    for (size_t i = 0; i < count; ++i)
    {
        cloud->positions()[i] = glm::vec4(0.25f * static_cast<float>(i), -1.0f, 2.0f, 0.0f);
        cloud->scales()[i] = glm::vec4(0.1f, 0.2f, 0.3f, 0.0f);
        cloud->rotations()[i] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        cloud->opacities()[i] = 0.5f;
        for (size_t k = 0; k < kShCoefficientsPerGaussian; ++k)
        {
            const float v = 0.1f * static_cast<float>(k) - 0.7f;
            const float w = 0.05f * static_cast<float>(k);
            cloud->shCoefficients()[i * kShCoefficientsPerGaussian + k] = glm::vec4(v, w, -0.2f, 0.0f);
        }
    }
    cloud->setShDegree(3);
    cloud->setAntialiased(false);
    cloud->computeBounds();
    return cloud;
}

// Expects SPZWriter::write to refuse with a std::runtime_error. When `mention` is
// not empty, the message must contain it. Unless `expectAbsent` is false, no file
// may be left at `path`, nor a temporary file.
void expectRefused(const std::string & label, const GaussianCloud & cloud, const fs::path & path,
                   uint32_t version, const std::string & mention, bool expectAbsent = true)
{
    bool threw = false;
    bool typed = false;
    std::string message;
    try
    {
        SPZWriter::write(cloud, path.string(), version);
    }
    catch (const std::runtime_error & e)
    {
        threw = true;
        typed = true;
        message = e.what();
    }
    catch (const std::exception & e)
    {
        threw = true;
        message = e.what();
    }
    check(threw && typed, label + ": refused with std::runtime_error");
    check(mention.empty() || message.find(mention) != std::string::npos,
          label + ": message names '" + mention + "' (got: " + message + ")");
    if (expectAbsent)
    {
        check(!fs::exists(path) && !fs::exists(path.string() + ".tmp"),
              label + ": no file or temporary file left behind");
    }
}

// Test (d): every refusal.
void testRefusals(Device & device)
{
    std::printf("\n-- (d) refusals --\n");

    {
        auto empty = std::make_shared<GaussianCloud>(device);
        empty->initialize(0);
        expectRefused("zero splats", *empty, caseFile("zero"), 3, "");
    }

    // One non-finite value in Gaussian 7 of a valid 10-splat cloud, per field.
    struct Field
    {
        const char * tag;
        void (*set)(GaussianCloud & cloud, size_t i, float value);
    };
    const Field fields[] = {
        {"position", [](GaussianCloud & c, size_t i, float v) { c.positions()[i].y = v; }},
        {"scale", [](GaussianCloud & c, size_t i, float v) { c.scales()[i].z = v; }},
        {"rotation", [](GaussianCloud & c, size_t i, float v) { c.rotations()[i].x = v; }},
        {"opacity", [](GaussianCloud & c, size_t i, float v) { c.opacities()[i] = v; }},
        {"dc", [](GaussianCloud & c, size_t i, float v) { c.shCoefficients()[i * kShCoefficientsPerGaussian].y = v; }},
        {"sh", [](GaussianCloud & c, size_t i, float v) { c.shCoefficients()[i * kShCoefficientsPerGaussian + 5].z = v; }},
    };
    const float nonFinite[2] = {std::numeric_limits<float>::quiet_NaN(),
                                std::numeric_limits<float>::infinity()};
    const char * nonFiniteTag[2] = {"nan", "inf"};
    for (const Field & field : fields)
    {
        for (int v = 0; v < 2; ++v)
        {
            auto cloud = validCloud(device, 10);
            field.set(*cloud, 7, nonFinite[v]);
            const std::string tag = std::string(field.tag) + "_" + nonFiniteTag[v];
            expectRefused("non-finite " + std::string(field.tag) + " (" + nonFiniteTag[v] + ")", *cloud,
                          caseFile(tag), 3, "7");
        }
    }

    // Positions outside round(x * 4096) in [-2^23, 2^23 - 1], i.e. beyond about +-2048.
    const float outOfRange[3] = {2048.0f, -2049.0f, 1.0e6f};
    for (int n = 0; n < 3; ++n)
    {
        auto cloud = validCloud(device, 10);
        cloud->positions()[3].x = outOfRange[n];
        expectRefused("position x " + std::to_string(outOfRange[n]), *cloud,
                      caseFile("range_" + std::to_string(n)), 3, "3");
    }

    {
        auto cloud = validCloud(device, 10);
        cloud->setShDegree(4);
        expectRefused("SH degree 4", *cloud, caseFile("degree_4"), 3, "");
    }

    expectRefused("version 2", *validCloud(device, 10), caseFile("version_2"), 2, "");
    expectRefused("version 5", *validCloud(device, 10), caseFile("version_5"), 5, "");

    {
        auto cloud = validCloud(device, 10);
        cloud->positions().resize(9);
        expectRefused("positions of the wrong size", *cloud, caseFile("size_positions"), 3, "");
    }
    {
        auto cloud = validCloud(device, 10);
        cloud->opacities().resize(11);
        expectRefused("opacities of the wrong size", *cloud, caseFile("size_opacities"), 3, "");
    }
    {
        auto cloud = validCloud(device, 10);
        cloud->shCoefficients().resize(10 * kShCoefficientsPerGaussian - 1);
        expectRefused("SH of the wrong size", *cloud, caseFile("size_sh"), 3, "");
    }

    // A refused write over an existing file leaves that file as it was.
    const fs::path prior = caseFile("prior");
    SPZWriter::write(*validCloud(device, 4), prior.string());
    check(fs::exists(prior), "prior file written");
    const std::vector<char> before = readBytes(prior);
    {
        auto bad = validCloud(device, 4);
        bad->opacities()[2] = std::numeric_limits<float>::quiet_NaN();
        expectRefused("NaN opacity over an existing file", *bad, prior, 3, "2", false);
        check(readBytes(prior) == before, "prior file untouched by the refused NaN write");
        check(!fs::exists(prior.string() + ".tmp"), "no temporary file after the refused NaN write");
    }
    expectRefused("version 2 over an existing file", *validCloud(device, 4), prior, 2, "", false);
    check(readBytes(prior) == before, "prior file untouched by the refused version-2 write");
}

// A two-splat degree-3 cloud for the byte check: every field off the rounding
// boundaries of its quantizer, with antialiased on.
std::shared_ptr<GaussianCloud> handCloud(Device & device)
{
    constexpr size_t kCount = 2;
    auto cloud = std::make_shared<GaussianCloud>(device);
    cloud->initialize(kCount);

    const glm::vec3 positions[kCount] = {{1.5f, -2.25f, 3.0f}, {-0.3f, 0.001f, 2047.99f}};
    const float logScales[kCount][3] = {{-1.23f, 0.77f, -0.5f}, {-4.0f, 2.2f, 0.4f}};
    const glm::vec4 rotations[kCount] = {glm::normalize(glm::vec4(0.3f, -0.4f, 0.5f, 0.7f)),
                                         glm::normalize(glm::vec4(-0.6f, 0.1f, 0.2f, -0.3f))};
    const float opacities[kCount] = {0.2f, 0.6f};
    const glm::vec3 dc[kCount] = {{0.8f, -0.6f, 0.2f}, {-0.1f, 0.45f, 1.2f}};

    for (size_t i = 0; i < kCount; ++i)
    {
        cloud->positions()[i] = glm::vec4(positions[i], 0.0f);
        cloud->scales()[i] = glm::vec4(std::exp(logScales[i][0]), std::exp(logScales[i][1]),
                                       std::exp(logScales[i][2]), 0.0f);
        cloud->rotations()[i] = rotations[i];
        cloud->opacities()[i] = opacities[i];
        cloud->shCoefficients()[i * kShCoefficientsPerGaussian] = glm::vec4(dc[i], 0.0f);
        for (int slot = 1; slot < kShCoefficientsPerGaussian; ++slot)
        {
            glm::vec4 value(0.0f);
            for (int c = 0; c < 3; ++c)
            {
                const int n = (slot * 7 + c * 3 + static_cast<int>(i) * 5) % 19 - 9;
                value[c] = static_cast<float>(n) * 0.0931f;
            }
            cloud->shCoefficients()[i * kShCoefficientsPerGaussian + slot] = value;
        }
    }
    cloud->setShDegree(3);
    cloud->setAntialiased(true);
    cloud->computeBounds();
    return cloud;
}

// Decoded smallest-three quaternion: which component was largest, and the four
// components (the largest one positive), as the format defines them.
struct Smallest3
{
    int largest = 0;
    std::array<double, 4> q{};
};

Smallest3 decodeSmallestThree(const uint8_t * b)
{
    const uint32_t comp = static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8)
                          | (static_cast<uint32_t>(b[2]) << 16) | (static_cast<uint32_t>(b[3]) << 24);
    Smallest3 out;
    out.largest = static_cast<int>(comp >> 30);
    double sumSquares = 0.0;
    int j = 0;
    for (int c = 0; c < 4; ++c)
    {
        if (c == out.largest)
        {
            continue;
        }
        // Field j (ascending component index) sits at bits 20 - 10j.
        const uint32_t field = (comp >> (20 - 10 * j)) & 0x3FFu;
        ++j;
        const uint32_t mag = field & 0x1FFu;
        const bool negative = ((field >> 9) & 1u) != 0u;
        double v = std::sqrt(0.5) * static_cast<double>(mag) / 511.0;
        if (negative)
        {
            v = -v;
        }
        out.q[c] = v;
        sumSquares += v * v;
    }
    out.q[out.largest] = std::sqrt(std::max(0.0, 1.0 - sumSquares));
    return out;
}

// Test (i): the version 3 bytes against the packing formula, read through the
// reference library; then version 4 must unpack to the same streams.
void testByteSpec(Device & device)
{
    std::printf("\n-- (i) byte-level spec check against the reference library --\n");
    auto cloud = handCloud(device);
    const std::string path3 = (g_dir / "byte_v3.spz").string();
    SPZWriter::write(*cloud, path3, 3);
    const spz::PackedGaussians packed = spz::loadSpzPacked(path3);

    check(packed.version == 3, "v3 header: version 3");
    check(packed.numPoints == 2, "v3 header: numPoints 2");
    check(packed.shDegree == 3, "v3 header: shDegree 3");
    check(packed.fractionalBits == 12, "v3 header: fractionalBits 12");
    check(packed.antialiased, "v3 header: antialiased flag set");
    check(packed.usesQuaternionSmallestThree, "v3 header: smallest-three quaternions");

    const bool sizes = packed.positions.size() == 18 && packed.scales.size() == 6
                       && packed.rotations.size() == 8 && packed.alphas.size() == 2
                       && packed.colors.size() == 6 && packed.sh.size() == 90;
    check(sizes, "v3 streams: sizes 18, 6, 8, 2, 6, 90");
    if (!sizes)
    {
        return;
    }

    // Positions: 24-bit little-endian two's complement of round(x * 4096).
    size_t bad = 0;
    for (size_t i = 0; i < 2; ++i)
    {
        for (int a = 0; a < 3; ++a)
        {
            const long long fixed = std::llround(static_cast<double>(cloud->positions()[i][a]) * 4096.0);
            const uint32_t twos = static_cast<uint32_t>(fixed) & 0xFFFFFFu;
            const uint8_t expected[3] = {static_cast<uint8_t>(twos & 0xFFu),
                                         static_cast<uint8_t>((twos >> 8) & 0xFFu),
                                         static_cast<uint8_t>((twos >> 16) & 0xFFu)};
            for (int b = 0; b < 3; ++b)
            {
                if (packed.positions[(i * 3 + a) * 3 + b] != expected[b])
                {
                    ++bad;
                }
            }
        }
    }
    check(bad == 0, "positions: 24-bit fixed point, round(x * 4096) (" + std::to_string(bad) + " bad bytes)");

    // Scales: clamp(round((ln s + 10) * 16)), s linear in the cloud.
    bad = 0;
    for (size_t i = 0; i < 2; ++i)
    {
        for (int a = 0; a < 3; ++a)
        {
            const double ln = std::log(static_cast<double>(cloud->scales()[i][a]));
            const double expected = std::clamp(std::round((ln + 10.0) * 16.0), 0.0, 255.0);
            if (packed.scales[i * 3 + a] != static_cast<uint8_t>(expected))
            {
                ++bad;
            }
        }
    }
    check(bad == 0, "scales: clamp(round((ln s + 10) * 16)) (" + std::to_string(bad) + " bad bytes)");

    // Alphas: clamp(round(opacity * 255)).
    bad = 0;
    for (size_t i = 0; i < 2; ++i)
    {
        const double expected = std::clamp(std::round(static_cast<double>(cloud->opacities()[i]) * 255.0), 0.0, 255.0);
        if (packed.alphas[i] != static_cast<uint8_t>(expected))
        {
            ++bad;
        }
    }
    check(bad == 0, "alphas: round(opacity * 255) (" + std::to_string(bad) + " bad bytes)");

    // Colors: clamp(round(dc * 0.15 * 255 + 127.5)).
    bad = 0;
    for (size_t i = 0; i < 2; ++i)
    {
        for (int c = 0; c < 3; ++c)
        {
            const double dc = static_cast<double>(cloud->shCoefficients()[i * kShCoefficientsPerGaussian][c]);
            const double expected = std::clamp(std::round(dc * 0.15 * 255.0 + 127.5), 0.0, 255.0);
            if (packed.colors[i * 3 + c] != static_cast<uint8_t>(expected))
            {
                ++bad;
            }
        }
    }
    check(bad == 0, "colors: clamp(round(dc * 0.15 * 255 + 127.5)) (" + std::to_string(bad) + " bad bytes)");

    // SH: coefficient k (0-based, slot k + 1) channel c at (k * 3 + c) per splat,
    // quantizeSH with buckets 8 for degree 1 and 16 above it.
    bad = 0;
    for (size_t i = 0; i < 2; ++i)
    {
        for (int k = 0; k < 15; ++k)
        {
            for (int c = 0; c < 3; ++c)
            {
                const float x = cloud->shCoefficients()[i * kShCoefficientsPerGaussian + k + 1][c];
                int q = static_cast<int>(std::round(x * 128.0f) + 128.0f);
                const int bucket = k < 3 ? 8 : 16;
                q = (q + bucket / 2) / bucket * bucket;
                q = std::clamp(q, 0, 255);
                if (packed.sh[i * 45 + static_cast<size_t>(k) * 3 + c] != static_cast<uint8_t>(q))
                {
                    ++bad;
                }
            }
        }
    }
    check(bad == 0, "SH: quantizeSH buckets 8 (slots 1..3) and 16 (slots 4..15) (" + std::to_string(bad)
                        + " bad bytes)");

    // Rotations: smallest-three, decoded as the format defines it, equal to the
    // normalized input up to sign, with the largest component positive.
    bad = 0;
    double worstRotation = 0.0;
    for (size_t i = 0; i < 2; ++i)
    {
        const Smallest3 got = decodeSmallestThree(&packed.rotations[i * 4]);
        glm::dvec4 want = glm::normalize(glm::dvec4(cloud->rotations()[i]));
        int largest = 0;
        for (int c = 1; c < 4; ++c)
        {
            if (std::abs(want[c]) > std::abs(want[largest]))
            {
                largest = c;
            }
        }
        if (want[largest] < 0.0)
        {
            want = -want;
        }
        bool ok = got.largest == largest;
        for (int c = 0; c < 4; ++c)
        {
            const double error = std::abs(got.q[c] - want[c]);
            worstRotation = std::max(worstRotation, error);
            ok = ok && error <= kRotationByteTolerance;
        }
        if (!ok)
        {
            ++bad;
        }
    }
    std::printf("  worst rotation component error from the packed bytes: %.3e\n", worstRotation);
    check(bad == 0, "rotations: smallest-three bytes decode to the input within 2e-3 per component");

    // Version 4 unpacks to the same fields.
    const std::string path4 = (g_dir / "byte_v4.spz").string();
    SPZWriter::write(*cloud, path4, 4);
    const spz::PackedGaussians v4 = spz::loadSpzPacked(path4);
    check(v4.version == 4, "v4 header: version 4");
    check(v4.numPoints == packed.numPoints && v4.shDegree == packed.shDegree
              && v4.fractionalBits == packed.fractionalBits && v4.antialiased == packed.antialiased
              && v4.usesQuaternionSmallestThree == packed.usesQuaternionSmallestThree,
          "v4 header fields equal the v3 ones");
    check(v4.positions == packed.positions, "v4 positions equal the v3 bytes");
    check(v4.scales == packed.scales, "v4 scales equal the v3 bytes");
    check(v4.rotations == packed.rotations, "v4 rotations equal the v3 bytes");
    check(v4.alphas == packed.alphas, "v4 alphas equal the v3 bytes");
    check(v4.colors == packed.colors, "v4 colors equal the v3 bytes");
    check(v4.sh == packed.sh, "v4 SH equal the v3 bytes");
}

} // namespace

int runExtraChecks(Device & device, const fs::path & dir)
{
    g_failures = 0;
    g_dir = dir;
    testRefusals(device);
    testByteSpec(device);
    return g_failures;
}
