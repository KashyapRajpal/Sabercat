// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Checks for PLYWriter and the plain-PLY reader that the round-trip test does not
// reach (WP6):
//
//  (a) Golden: a fixed two-Gaussian degree-1 cloud is written and the file's
//      header text is compared with the exact expected text, and the first
//      record's 26 float words with hand-derived bit patterns. A writer and
//      reader that were symmetrically wrong would still round-trip, so the
//      expected bytes here come from the format, not from the reader.
//  (b) A non-finite position is refused, naming the Gaussian, and no file is
//      left behind.
//  (c) Plain-PLY negatives, each rejected by name: ascii, big-endian, no
//      end_header, missing x / scale_0 / rot_0, a double property, f_rest
//      count 4 and 48, zero vertices, a truncated body. Positive controls with
//      f_rest 0, 9 and 45 (degrees 0, 1 and 3) read fine, so the negatives are
//      not over-strict.
//  (d) A 1,000,000-Gaussian degree-3 cloud round-trips through PLYWriter and
//      PLYReader (timed), with the documented error bounds.

#include <glm/glm.hpp>

#include <algorithm>
#include <cfloat>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "sabercat/device.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/ply_reader.h"
#include "sabercat/ply_writer.h"
#include "sabercat/render_config.h"

using namespace sabercat;
namespace fs = std::filesystem;

namespace
{

int g_failures = 0;

void check(bool ok, const std::string & what)
{
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok)
    {
        ++g_failures;
    }
}

uint32_t bitsOf(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof bits);
    return bits;
}

std::string readFile(const fs::path & path)
{
    std::ifstream file(path, std::ios::binary);
    std::ostringstream text;
    text << file.rdbuf();
    return text.str();
}

void writeFile(const fs::path & path, const std::string & bytes)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

// ---- (a) golden ----------------------------------------------------------

void testGolden(Device & device, const fs::path & dir)
{
    GaussianCloud cloud(device);
    cloud.initialize(2);
    cloud.setShDegree(1);
    cloud.positions()[0] = glm::vec4(1.0f, 2.0f, 3.0f, 0.0f);
    cloud.positions()[1] = glm::vec4(-1.0f, 0.5f, 7.0f, 0.0f);
    cloud.opacities()[0] = 0.5f; // logit(0.5) == 0 exactly
    cloud.opacities()[1] = 0.25f;
    cloud.scales()[0] = glm::vec4(1.0f, 2.0f, 4.0f, 0.0f);
    cloud.scales()[1] = glm::vec4(0.5f, 0.5f, 0.5f, 0.0f);
    cloud.rotations()[0] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f); // identity, xyzw
    cloud.rotations()[1] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    for (size_t g = 0; g < 2; ++g)
    {
        glm::vec4 * sh = &cloud.shCoefficients()[g * kShCoefficientsPerGaussian];
        sh[0] = glm::vec4(0.5f, 0.25f, -0.5f, 0.0f);
        // Rest coefficient k (0-based) of channel c is 10 * c + k + 1, so the
        // channel-major order is visible in the bytes.
        for (int k = 0; k < 3; ++k)
        {
            for (int c = 0; c < 3; ++c)
            {
                sh[1 + k][c] = static_cast<float>(10 * c + k + 1);
            }
        }
    }

    const fs::path path = dir / "golden.ply";
    PLYWriter::write(cloud, path.string());
    const std::string bytes = readFile(path);

    // Header: exact text, including the writer's comment line.
    std::string expectedHeader = "ply\nformat binary_little_endian 1.0\n"
                                 "comment Written by Sabercat PLYWriter\n"
                                 "element vertex 2\n";
    const char * names[] = {"x",       "y",       "z",       "nx",      "ny",      "nz",      "f_dc_0",
                            "f_dc_1",  "f_dc_2",  "f_rest_0", "f_rest_1", "f_rest_2", "f_rest_3", "f_rest_4",
                            "f_rest_5", "f_rest_6", "f_rest_7", "f_rest_8", "opacity", "scale_0", "scale_1",
                            "scale_2", "rot_0",   "rot_1",   "rot_2",   "rot_3"};
    for (const char * name : names)
    {
        expectedHeader += std::string("property float ") + name + "\n";
    }
    expectedHeader += "end_header\n";
    check(bytes.compare(0, expectedHeader.size(), expectedHeader) == 0,
          "golden header text matches the format exactly (" + std::to_string(expectedHeader.size())
              + " bytes)");

    // First record: 26 float words, derived by hand:
    //   x y z = 1, 2, 3; nx ny nz = 0 (no normals); f_dc = 0.5, 0.25, -0.5;
    //   f_rest channel-major = R (1 2 3), G (11 12 13), B (21 22 23);
    //   opacity = logit(0.5) = 0; scale = log(1), log(2), log(4) = 0,
    //   ln 2 = 0x3F317218, ln 4 = 0x3FB17218; rotation (w x y z) = 1 0 0 0.
    const uint32_t expected[26] = {
        0x3F800000, 0x40000000, 0x40400000,             // x y z = 1 2 3
        0x00000000, 0x00000000, 0x00000000,             // nx ny nz
        0x3F000000, 0x3E800000, 0xBF000000,             // f_dc = 0.5 0.25 -0.5
        0x3F800000, 0x40000000, 0x40400000,             // f_rest R = 1 2 3
        0x41300000, 0x41400000, 0x41500000,             // f_rest G = 11 12 13
        0x41A80000, 0x41B00000, 0x41B80000,             // f_rest B = 21 22 23
        0x00000000,                                     // opacity = 0
        0x00000000, 0x3F317218, 0x3FB17218,             // scale = 1 2 4 (log)
        0x3F800000, 0x00000000, 0x00000000, 0x00000000, // rot w x y z = 1 0 0 0
    };
    // The hand-derived constants above are checked against the IEEE values
    // they name, so a typo in this table cannot pass.
    check(bitsOf(0.5f) == 0x3F000000 && bitsOf(0.25f) == 0x3E800000 && bitsOf(-0.5f) == 0xBF000000
              && bitsOf(11.0f) == 0x41300000 && bitsOf(21.0f) == 0x41A80000 && bitsOf(23.0f) == 0x41B80000
              && bitsOf(std::log(2.0f)) == 0x3F317218 && bitsOf(std::log(4.0f)) == 0x3FB17218,
          "golden constants are the IEEE bit patterns of their values");

    const size_t recordOffset = expectedHeader.size();
    check(bytes.size() == recordOffset + 2 * 26 * sizeof(float),
          "golden file is header + 2 records (" + std::to_string(bytes.size()) + " bytes)");
    size_t mismatches = 0;
    for (size_t w = 0; w < 26; ++w)
    {
        uint32_t got = 0;
        std::memcpy(&got, bytes.data() + recordOffset + 4 * w, 4);
        if (got != expected[w])
        {
            std::printf("  word %zu: expected 0x%08X got 0x%08X\n", w, expected[w], got);
            ++mismatches;
        }
    }
    check(mismatches == 0, "golden first record: all 26 float words match the hand-derived bits");
}

// ---- (b) non-finite positions ---------------------------------------------

void testNonFinitePosition(Device & device, const fs::path & dir)
{
    const fs::path path = dir / "nonfinite.ply";
    fs::remove(path);

    GaussianCloud cloud(device);
    cloud.initialize(3);
    cloud.setShDegree(0);
    for (size_t g = 0; g < 3; ++g)
    {
        cloud.positions()[g] = glm::vec4(static_cast<float>(g), 0.0f, 0.0f, 0.0f);
        cloud.opacities()[g] = 0.5f;
        cloud.scales()[g] = glm::vec4(1.0f);
        cloud.rotations()[g] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    }
    cloud.positions()[1].y = std::numeric_limits<float>::quiet_NaN();
    cloud.positions()[2].z = std::numeric_limits<float>::infinity();

    try
    {
        PLYWriter::write(cloud, path.string());
        check(false, "PLYWriter refuses a NaN position (no exception)");
    }
    catch (const std::exception & error)
    {
        const std::string message = error.what();
        // Gaussian 1 is the first bad one; the check names it, not Gaussian 2.
        check(message.find("Gaussian 1") != std::string::npos
                  && message.find("non-finite y position") != std::string::npos,
              "PLYWriter refuses a NaN position and names the first bad Gaussian (" + message + ")");
    }
    check(!fs::exists(path) && !fs::exists(path.string() + ".tmp"),
          "a refused write leaves neither the file nor a temporary behind");
}

// ---- (c) plain-PLY negatives -----------------------------------------------

// Float properties the reader needs, with `restCount` f_rest properties,
// minus any name in `omit`.
std::vector<std::string> plainProperties(size_t restCount, const std::string & omit = "")
{
    std::vector<std::string> names = {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
                                      "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"};
    for (size_t k = 0; k < restCount; ++k)
    {
        names.push_back("f_rest_" + std::to_string(k));
    }
    std::vector<std::string> lines;
    for (const std::string & name : names)
    {
        if (name != omit)
        {
            lines.push_back("property float " + name);
        }
    }
    return lines;
}

// Bytes per vertex of a property list (all are 4-byte floats unless "double").
size_t recordBytes(const std::vector<std::string> & lines)
{
    size_t bytes = 0;
    for (const std::string & line : lines)
    {
        bytes += line.find(" double ") != std::string::npos ? 8 : 4;
    }
    return bytes;
}

// Writes a PLY: header lines, then `bodyBytes` zero bytes.
void writePly(const fs::path & path, const std::string & format, size_t declaredCount,
              const std::vector<std::string> & properties, bool endHeader, size_t bodyBytes)
{
    std::string text = "ply\nformat " + format + "\nelement vertex " + std::to_string(declaredCount) + "\n";
    for (const std::string & line : properties)
    {
        text += line + "\n";
    }
    if (endHeader)
    {
        text += "end_header\n";
    }
    text.append(bodyBytes, '\0');
    writeFile(path, text);
}

struct Negative
{
    const char * what;
    const char * needle;
    std::string format;
    std::vector<std::string> properties;
    bool endHeader;
    size_t declaredCount;
    size_t bodyRecords; // records of body actually written
};

void testPlainPlyNegatives(Device & device, const fs::path & dir)
{
    const fs::path path = dir / "plain.ply";
    const std::string binary = "binary_little_endian 1.0";
    const size_t good = 4;
    const auto ok = plainProperties(9);

    std::vector<std::string> doubleX = plainProperties(9, "x");
    doubleX.insert(doubleX.begin(), "property double x");

    const Negative negatives[] = {
        {"ascii format", "only binary_little_endian PLY is supported, got 'ascii'", "ascii 1.0", ok, true, good, good},
        {"big-endian format", "got 'binary_big_endian'", "binary_big_endian 1.0", ok, true, good, good},
        {"no end_header", "missing 'end_header'", binary, ok, false, good, 0},
        {"missing x", "missing required property 'x'", binary, plainProperties(9, "x"), true, good, good},
        {"missing scale_0", "missing required property 'scale_0'", binary, plainProperties(9, "scale_0"), true, good, good},
        {"missing rot_0", "missing required property 'rot_0'", binary, plainProperties(9, "rot_0"), true, good, good},
        {"double x", "property 'x' is not 4 bytes", binary, doubleX, true, good, good},
        {"f_rest count 4", "f_rest property count (4) is not divisible by 3", binary, plainProperties(4), true, good, good},
        {"f_rest count 48 (SH degree 4)", "SH degree too high", binary, plainProperties(48), true, good, good},
        {"zero vertices", "no vertices", binary, ok, true, 0, 0},
        {"truncated body (10 declared, 4 written)", "truncated 'vertex' element data", binary, ok, true, 10, good},
    };

    for (const Negative & negative : negatives)
    {
        writePly(path, negative.format, negative.declaredCount, negative.properties, negative.endHeader,
                 negative.bodyRecords * recordBytes(negative.properties));
        try
        {
            PLYReader::read(device, path.string());
            check(false, std::string("rejects ") + negative.what + " (no exception)");
        }
        catch (const std::exception & error)
        {
            const std::string message = error.what();
            check(message.find(negative.needle) != std::string::npos,
                  std::string("rejects ") + negative.what + " (" + message + ")");
        }
    }

    // Positive controls: f_rest 0, 9 and 45 (SH degrees 0, 1 and 3) read fine.
    for (const size_t rest : {size_t(0), size_t(9), size_t(45)})
    {
        const std::vector<std::string> props = plainProperties(rest);
        const uint32_t degree = rest == 0 ? 0u : (rest == 9 ? 1u : 3u);
        writePly(path, binary, good, props, true, good * recordBytes(props));
        std::string detail;
        bool accepted = false;
        try
        {
            const auto cloud = PLYReader::read(device, path.string());
            accepted = cloud->count() == good && cloud->shDegree() == degree;
            detail = "count " + std::to_string(cloud->count()) + ", shDegree " + std::to_string(cloud->shDegree());
        }
        catch (const std::exception & error)
        {
            detail = error.what();
        }
        check(accepted, "accepts f_rest " + std::to_string(rest) + " as SH degree " + std::to_string(degree)
                            + " (" + detail + ")");
    }
}

// ---- (d) one million Gaussians --------------------------------------------

void testMillionRoundTrip(Device & device, const fs::path & dir)
{
    const size_t count = 1000000;
    std::mt19937 rng(20261020u);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    std::normal_distribution<float> normal(0.0f, 10.0f);
    std::uniform_real_distribution<float> opacityDist(0.01f, 0.99f);
    std::uniform_real_distribution<float> scaleDist(0.01f, 5.0f);

    GaussianCloud cloud(device);
    cloud.initialize(count);
    cloud.setShDegree(3);
    for (size_t i = 0; i < count; ++i)
    {
        cloud.positions()[i] = glm::vec4(normal(rng), normal(rng), normal(rng), 0.0f);
        glm::vec4 * sh = &cloud.shCoefficients()[i * kShCoefficientsPerGaussian];
        for (int k = 0; k < kShCoefficientsPerGaussian; ++k)
        {
            sh[k] = glm::vec4(unit(rng), unit(rng), unit(rng), 0.0f);
        }
        cloud.opacities()[i] = opacityDist(rng);
        cloud.scales()[i] = glm::vec4(scaleDist(rng), scaleDist(rng), scaleDist(rng), 0.0f);
        cloud.rotations()[i] = glm::normalize(glm::vec4(unit(rng), unit(rng), unit(rng), unit(rng)));
    }

    const fs::path path = dir / "million.ply";
    const auto writeStart = std::chrono::steady_clock::now();
    PLYWriter::write(cloud, path.string());
    const double writeMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - writeStart).count();
    const auto readStart = std::chrono::steady_clock::now();
    const auto back = PLYReader::read(device, path.string());
    const double readMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - readStart).count();
    std::printf("1M Gaussians: write %.0f ms, read %.0f ms, file %.1f MB\n", writeMs, readMs,
                static_cast<double>(fs::file_size(path)) / 1e6);

    check(back->count() == count && back->shDegree() == 3,
          "1M: count and shDegree round trip (" + std::to_string(back->count()) + " Gaussians)");

    // Positions and SH are a byte copy in both directions: bit-exact.
    size_t exactMismatch = 0;
    for (size_t i = 0; i < count; ++i)
    {
        const glm::vec4 & a = cloud.positions()[i];
        const glm::vec4 & b = back->positions()[i];
        if (bitsOf(a.x) != bitsOf(b.x) || bitsOf(a.y) != bitsOf(b.y) || bitsOf(a.z) != bitsOf(b.z))
        {
            ++exactMismatch;
        }
        for (int k = 0; k < kShCoefficientsPerGaussian; ++k)
        {
            const glm::vec4 & s = cloud.shCoefficients()[i * kShCoefficientsPerGaussian + k];
            const glm::vec4 & t = back->shCoefficients()[i * kShCoefficientsPerGaussian + k];
            if (bitsOf(s.x) != bitsOf(t.x) || bitsOf(s.y) != bitsOf(t.y) || bitsOf(s.z) != bitsOf(t.z))
            {
                ++exactMismatch;
            }
        }
    }
    check(exactMismatch == 0, "1M: positions and SH bit-exact (" + std::to_string(exactMismatch) + " mismatches)");

    // Opacity bound: d sigmoid/dx = o(1 - o) <= 1/4, and |logit(o)| <= 4.6 for
    // o in [0.01, 0.99], so one float rounding of the logit (2^-23 * 4.6 ~ 5.5e-7)
    // moves the result by at most ~1.4e-7. The bound 1e-6 leaves a 7x margin.
    // Scale bound: log and exp each round to float with |log s| <= 1.61, so the
    // relative error stays near 2e-7; 1e-6 leaves a 5x margin.
    // Rotation bound: the reader renormalises a quaternion whose squared length
    // is more than 1e-6 from one; the flip-tolerant error is a few ulp, and 8 ulp
    // is the bound.
    double worstOpacity = 0.0;
    double worstScale = 0.0;
    double worstRotationUlp = 0.0;
    for (size_t i = 0; i < count; ++i)
    {
        worstOpacity = std::max(worstOpacity, std::fabs(static_cast<double>(back->opacities()[i]) - cloud.opacities()[i]));
        for (int c = 0; c < 3; ++c)
        {
            const double a = cloud.scales()[i][c];
            const double b = back->scales()[i][c];
            worstScale = std::max(worstScale, std::fabs(b / a - 1.0));
        }
        const glm::vec4 & qa = cloud.rotations()[i];
        const glm::vec4 & qb = back->rotations()[i];
        const glm::vec4 diff = qa - qb;
        const glm::vec4 sum = qa + qb;
        const double direct = std::max({std::fabs(static_cast<double>(diff.x)), std::fabs(static_cast<double>(diff.y)),
                                        std::fabs(static_cast<double>(diff.z)), std::fabs(static_cast<double>(diff.w))});
        const double flipped = std::max({std::fabs(static_cast<double>(sum.x)), std::fabs(static_cast<double>(sum.y)),
                                         std::fabs(static_cast<double>(sum.z)), std::fabs(static_cast<double>(sum.w))});
        worstRotationUlp = std::max(worstRotationUlp, std::min(direct, flipped) / static_cast<double>(FLT_EPSILON));
    }
    std::printf("1M worst errors: opacity %.3g (bound 1e-6), scale relative %.3g (bound 1e-6), rotation %.2f ulp (bound 8)\n",
                worstOpacity, worstScale, worstRotationUlp);
    check(worstOpacity <= 1e-6, "1M: opacity within 1e-6 after sigmoid(logit)");
    check(worstScale <= 1e-6, "1M: scale within relative 1e-6 after exp(log)");
    check(worstRotationUlp <= 8.0, "1M: rotation within 8 ulp (up to sign) after renormalisation");

    fs::remove(path);
}

} // namespace

// Entry point called by ply_writer_test's main. Returns the failure count.
int runPlainPlyChecks(Device & device, const fs::path & dir)
{
    g_failures = 0;
    testGolden(device, dir);
    testNonFinitePosition(device, dir);
    testPlainPlyNegatives(device, dir);
    testMillionRoundTrip(device, dir);
    std::printf("plain PLY checks: %s (%d failure%s)\n", g_failures == 0 ? "OK" : "FAILED", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures;
}
