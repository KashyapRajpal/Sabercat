// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Headless test for PLYWriter (Studio S1, Task B).
//
// Clouds are synthesized AT RUNTIME from a fixed seed, so the test needs no
// assets/. Each cloud is written with PLYWriter, read back with PLYReader, and
// compared with the original host data:
//
//   * positions, raw SH and visibility, metallic: BIT-EXACT. The writer and
//     reader do no arithmetic on these, so any difference is a layout bug.
//   * rotations: up to sign, within 8 float32 ulps of 1. The reader
//     renormalizes, which can move each component by a few ulps.
//   * scales (exp/log) and opacities (sigmoid/logit): within the propagated
//     float32 error bound, which each check states in its comment.
//   * base color and roughness (affine sigmoid): the same bound, scaled by
//     the affine factor.
//   * normals: within 8 ulps (the reader renormalizes them).
//
// Also checked: shDegree and count round trip; a relight-free cloud carries no
// R3DG property; the rejection paths throw; a failed write leaves the existing
// file intact; and, when a real .ply is on disk, load -> write -> load.

#include <glm/glm.hpp>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>
#include <string>
#include <system_error>
#include <vector>

#include "../common/scratch_dir.h"
#include "sabercat/device.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/ply_reader.h"
#include "sabercat/ply_writer.h"
#include "sabercat/render_config.h"

using namespace sabercat;

// plain_ply_checks.cpp: golden bytes, NaN refusal, plain-PLY negatives, 1M round trip.
int runPlainPlyChecks(Device & device, const std::filesystem::path & dir);

namespace
{

#define CHECK(condition, message)                                                                  \
    do                                                                                             \
    {                                                                                              \
        if (!(condition))                                                                          \
        {                                                                                          \
            std::cerr << "FAILED: " << (message) << "\n  at " << __FILE__ << ":" << __LINE__       \
                      << std::endl;                                                                \
            return 1;                                                                              \
        }                                                                                          \
    } while (false)

// 2^-24: half an ulp of a value in [0.5, 1), the unit that the bounds below use.
constexpr double kHalfUlpAtOne = 0x1p-24;
// One float32 ulp at 1 (2^-23), the tolerance unit for quantities near unit size.
constexpr double kUlp = static_cast<double>(FLT_EPSILON);
constexpr double kClampEps = static_cast<double>(PLYWriter::kLogitEpsilon);

bool bitEqual(float a, float b)
{
    uint32_t x;
    uint32_t y;
    std::memcpy(&x, &a, sizeof x);
    std::memcpy(&y, &b, sizeof y);
    return x == y;
}

// Keeps the worst error and worst error/tolerance ratio. The test passes while
// every ratio is <= 1, so each tolerance has to be justified where it is set.
struct Worst
{
    double error = 0.0;
    double ratio = 0.0;
    void add(double err, double tolerance)
    {
        error = std::max(error, err);
        ratio = std::max(ratio, err / tolerance);
    }
};

// The writer's clamp, restated here so the expected value does not depend on
// the code under test.
double clampUnit(double s)
{
    return std::min(std::max(s, kClampEps), 1.0 - kClampEps);
}

// The reader's sigmoid argument for an activated value, as the writer inverts
// it: x = logit(clamp(s)).
double logitOf(double s)
{
    return std::log(s / (1.0 - s));
}

// Tolerance for `y = float(sigmoid(x))` when x = float(logit(s)) is stored as
// float32: |dy| <= sigmoid' * |dx| + evaluation, with sigmoid' <= 1/4 and
// |dx| <= |x| * 2^-24 (the float rounding of x). The evaluation term covers
// the float exp and the final divide, each within a few half-ulps.
double sigmoidTolerance(double x)
{
    return 0.25 * std::abs(x) * kHalfUlpAtOne + 3.0 * kHalfUlpAtOne;
}

// Synthesizes a cloud with every feature the format carries: edge opacities
// (0, 1, just inside each end), tiny and large scales, all quaternion signs,
// and SH coefficients up to `shDegree` (zero above it, as the reader leaves them).
std::shared_ptr<GaussianCloud> makeCloud(Device & device, size_t count, uint32_t shDegree,
                                         bool relight, bool withVisibility, bool withMetallic,
                                         uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_real_distribution<double> symmetric(-1.0, 1.0);
    std::normal_distribution<double> gauss(0.0, 1.0);
    const double logScaleLo = std::log(1e-6);
    const double logScaleHi = std::log(50.0);

    auto cloud = std::make_shared<GaussianCloud>(device);
    cloud->initialize(count);
    cloud->setShDegree(shDegree);
    const size_t restPerChannel = static_cast<size_t>((shDegree + 1) * (shDegree + 1) - 1);

    auto & positions = cloud->positions();
    auto & scales = cloud->scales();
    auto & rotations = cloud->rotations();
    auto & sh = cloud->shCoefficients();
    auto & opacities = cloud->opacities();
    for (size_t i = 0; i < count; ++i)
    {
        positions[i] = glm::vec4(static_cast<float>(symmetric(rng) * 5.0),
                                 static_cast<float>(symmetric(rng) * 5.0),
                                 static_cast<float>(symmetric(rng) * 5.0),
                                 0.0f);

        switch (i % 8)
        {
        case 0:
            opacities[i] = 0.0f;
            break;
        case 1:
            opacities[i] = 1.0f;
            break;
        case 2:
            opacities[i] = 1e-6f;
            break;
        case 3:
            opacities[i] = 1.0f - 1e-6f;
            break;
        case 4:
            opacities[i] = 1e-4f;
            break;
        case 5:
            opacities[i] = 0.9999f;
            break;
        default:
            opacities[i] = static_cast<float>(unit(rng));
            break;
        }

        for (int c = 0; c < 3; ++c)
        {
            scales[i][c] = static_cast<float>(std::exp(logScaleLo + (logScaleHi - logScaleLo) * unit(rng)));
        }

        // Unit quaternion (w, x, y, z), with its sign flipped for half of them.
        glm::dvec4 q(gauss(rng), gauss(rng), gauss(rng), gauss(rng));
        q /= glm::length(q);
        const double sign = (rng() & 1u) ? -1.0 : 1.0;
        // Canonical storage is xyzw.
        rotations[i] = glm::vec4(static_cast<float>(sign * q.x), static_cast<float>(sign * q.y),
                                 static_cast<float>(sign * q.z), static_cast<float>(sign * q.w));

        glm::vec4 * shOut = &sh[i * kShCoefficientsPerGaussian];
        for (int c = 0; c < 3; ++c)
        {
            shOut[0][c] = static_cast<float>(symmetric(rng) * 2.0);
        }
        for (size_t k = 0; k < restPerChannel; ++k)
        {
            for (int c = 0; c < 3; ++c)
            {
                shOut[1 + k][c] = static_cast<float>(symmetric(rng));
            }
        }
    }

    if (relight)
    {
        auto & normals = cloud->normals();
        auto & baseColorRoughness = cloud->baseColorRoughness();
        normals.resize(count);
        baseColorRoughness.resize(count);
        for (size_t i = 0; i < count; ++i)
        {
            glm::dvec3 n(gauss(rng), gauss(rng), gauss(rng));
            n /= glm::length(n);
            normals[i] = glm::vec4(glm::vec3(n), 0.0f);
            // Base color in the reader's open range (0.03, 0.80), roughness in (0.09, 0.99).
            baseColorRoughness[i] =
                glm::vec4(static_cast<float>(0.05 + 0.70 * unit(rng)),
                          static_cast<float>(0.05 + 0.70 * unit(rng)),
                          static_cast<float>(0.05 + 0.70 * unit(rng)),
                          static_cast<float>(0.12 + 0.83 * unit(rng)));
        }
        if (withVisibility)
        {
            auto & visibility = cloud->visibility();
            visibility.resize(count * kVisibilityCoefficients);
            for (float & v : visibility)
            {
                v = static_cast<float>(symmetric(rng));
            }
        }
        if (withMetallic)
        {
            auto & metallic = cloud->metallic();
            metallic.resize(count);
            for (size_t i = 0; i < count; ++i)
            {
                metallic[i] = i % 5 == 0 ? (i % 2 == 0 ? 0.0f : 1.0f) : static_cast<float>(unit(rng));
            }
        }
    }

    cloud->computeBounds();
    return cloud;
}

// Reads the .ply header's property names, in order.
std::vector<std::string> headerProperties(const std::string & path, std::string & elementLine)
{
    std::ifstream file(path, std::ios::binary);
    std::vector<std::string> names;
    std::string line;
    while (std::getline(file, line))
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        if (line == "end_header")
        {
            break;
        }
        if (line.rfind("element ", 0) == 0)
        {
            elementLine = line;
        }
        if (line.rfind("property ", 0) == 0)
        {
            names.push_back(line.substr(line.rfind(' ') + 1));
        }
    }
    return names;
}

bool hasProperty(const std::vector<std::string> & names, const std::string & name)
{
    for (const std::string & n : names)
    {
        if (n == name)
        {
            return true;
        }
    }
    return false;
}

// Compares a loaded cloud with the original host data it was written from.
// `relight` says whether the original carries relight streams.
int compareClouds(GaussianCloud & original, GaussianCloud & loaded, const std::string & label,
                  bool relight)
{
    CHECK(loaded.count() == original.count(),
          label + ": count " + std::to_string(loaded.count()) + ", expected "
              + std::to_string(original.count()));
    CHECK(loaded.shDegree() == original.shDegree(), label + ": shDegree round trip differs");

    const size_t n = original.count();
    const auto & pa = original.positions();
    const auto & pl = loaded.positions();
    const auto & sa = original.scales();
    const auto & sl = loaded.scales();
    const auto & ra = original.rotations();
    const auto & rl = loaded.rotations();
    const auto & ha = original.shCoefficients();
    const auto & hl = loaded.shCoefficients();
    const auto & oa = original.opacities();
    const auto & ol = loaded.opacities();

    Worst scale;
    Worst opacity;
    Worst rotation;
    Worst baseColor;
    Worst roughness;
    Worst normal;
    double worstScaleRel = 0.0;
    double worstClampDeviation = 0.0;

    for (size_t i = 0; i < n; ++i)
    {
        for (int c = 0; c < 3; ++c)
        {
            CHECK(bitEqual(pa[i][c], pl[i][c]), label + ": position not bit-exact at " + std::to_string(i));
        }
        // All 16 slots: coefficients above the cloud's degree are zero on both sides.
        for (int k = 0; k < kShCoefficientsPerGaussian; ++k)
        {
            for (int c = 0; c < 3; ++c)
            {
                CHECK(bitEqual(ha[i * kShCoefficientsPerGaussian + k][c],
                               hl[i * kShCoefficientsPerGaussian + k][c]),
                      label + ": SH not bit-exact at " + std::to_string(i) + " slot "
                          + std::to_string(k));
            }
        }

        // Rotation up to sign: the reader normalizes, which moves each component by a few ulps.
        {
            const glm::vec4 & a = ra[i];
            const glm::vec4 & b = rl[i];
            const double same = std::max({std::abs(a.x - b.x), std::abs(a.y - b.y),
                                          std::abs(a.z - b.z), std::abs(a.w - b.w)});
            const double flipped = std::max({std::abs(a.x + b.x), std::abs(a.y + b.y),
                                             std::abs(a.z + b.z), std::abs(a.w + b.w)});
            rotation.add(std::min(same, flipped), 8.0 * kUlp);
        }

        // Scale: float(log) then float(exp). Error: |dlog| <= |ln s| * 2^-24 from the
        // float rounding of the log, plus about one ulp for float exp, so the
        // relative error is bounded by (|ln s| + 2) * 2^-23.
        for (int c = 0; c < 3; ++c)
        {
            const double s = static_cast<double>(sa[i][c]);
            const double r = static_cast<double>(sl[i][c]);
            const double rel = std::abs(r - s) / s;
            scale.add(std::abs(r - s), s * kUlp * (std::abs(std::log(s)) + 2.0));
            worstScaleRel = std::max(worstScaleRel, rel);
        }

        // Opacity: the expected value is the writer's clamp of the original. The
        // reader's value is float sigmoid(float logit(clamp)), bounded by sigmoidTolerance.
        {
            const double o = static_cast<double>(oa[i]);
            const double expected = clampUnit(o);
            worstClampDeviation = std::max(worstClampDeviation, std::abs(expected - o));
            opacity.add(std::abs(static_cast<double>(ol[i]) - expected),
                        sigmoidTolerance(logitOf(expected)));
        }

        if (relight)
        {
            const auto & bca = original.baseColorRoughness()[i];
            const auto & bcl = loaded.baseColorRoughness()[i];
            // base color = sigmoid * 0.77 + 0.03: the error is 0.77 times the sigmoid
            // error, plus two float roundings of the affine step (|v| <= 1).
            for (int c = 0; c < 3; ++c)
            {
                const double v = static_cast<double>(bca[c]);
                const double s = clampUnit((v - 0.03) / 0.77);
                const double expected = 0.03 + 0.77 * s;
                baseColor.add(std::abs(static_cast<double>(bcl[c]) - expected),
                              0.77 * sigmoidTolerance(logitOf(s)) + 4.0 * kHalfUlpAtOne);
            }
            // roughness = sigmoid * 0.9 + 0.09, the same way.
            {
                const double v = static_cast<double>(bca.w);
                const double s = clampUnit((v - 0.09) / 0.9);
                const double expected = 0.09 + 0.9 * s;
                roughness.add(std::abs(static_cast<double>(bcl.w) - expected),
                              0.9 * sigmoidTolerance(logitOf(s)) + 4.0 * kHalfUlpAtOne);
            }

            // Normals: the reader divides by max(|n|, 1e-3). With |n| = 1 to a few
            // ulps, each component moves by at most ~8 ulps.
            const glm::vec4 & na = original.normals()[i];
            const glm::vec4 & nl = loaded.normals()[i];
            const double dn = std::max({std::abs(na.x - nl.x), std::abs(na.y - nl.y),
                                        std::abs(na.z - nl.z)});
            normal.add(dn, 8.0 * kUlp);
        }
    }

    // Visibility and metallic are raw: bit-exact or a bug.
    if (relight)
    {
        const auto & va = original.visibility();
        const auto & vl = loaded.visibility();
        CHECK(va.size() == vl.size(), label + ": visibility stream size differs");
        for (size_t k = 0; k < va.size(); ++k)
        {
            CHECK(bitEqual(va[k], vl[k]), label + ": visibility not bit-exact at " + std::to_string(k));
        }
        const auto & ma = original.metallic();
        const auto & ml = loaded.metallic();
        CHECK(ma.size() == ml.size(), label + ": metallic stream size differs");
        for (size_t k = 0; k < ma.size(); ++k)
        {
            CHECK(bitEqual(ma[k], ml[k]), label + ": metallic not bit-exact at " + std::to_string(k));
        }
    }

    std::printf("  %-28s positions, SH%s bit-exact\n", label.c_str(),
                relight ? ", visibility, metallic" : "");
    std::printf("    rotation   worst %.3g  (tol 8 ulp, ratio %.3f)\n", rotation.error, rotation.ratio);
    std::printf("    scale      worst rel %.3g  (tol ulp*(|ln s|+2), worst ratio %.3f)\n",
                worstScaleRel, scale.ratio);
    std::printf("    opacity    worst abs %.3g  (tol sigmoid bound, worst ratio %.3f; "
                "clamp moved an input by at most %.3g)\n",
                opacity.error, opacity.ratio, worstClampDeviation);
    if (relight)
    {
        std::printf("    base color worst abs %.3g  (ratio %.3f)\n", baseColor.error, baseColor.ratio);
        std::printf("    roughness  worst abs %.3g  (ratio %.3f)\n", roughness.error, roughness.ratio);
        std::printf("    normal     worst abs %.3g  (tol 8 ulp, ratio %.3f)\n", normal.error, normal.ratio);
    }

    CHECK(rotation.ratio <= 1.0, label + ": rotation outside tolerance");
    CHECK(scale.ratio <= 1.0, label + ": scale outside tolerance");
    CHECK(opacity.ratio <= 1.0, label + ": opacity outside tolerance");
    if (relight)
    {
        CHECK(baseColor.ratio <= 1.0, label + ": base color outside tolerance");
        CHECK(roughness.ratio <= 1.0, label + ": roughness outside tolerance");
        CHECK(normal.ratio <= 1.0, label + ": normal outside tolerance");
    }
    return 0;
}

// shDegree 0..3, relight-free: count/degree round trip, the expected f_rest
// count, no R3DG property, and the reader's empty relight streams.
int testRoundTrip(Device & device, const std::filesystem::path & dir)
{
    std::cout << "round trip, relight-free (SH degree 0..3)" << std::endl;
    constexpr size_t kCount = 4000;
    for (uint32_t degree = 0; degree <= 3; ++degree)
    {
        auto original = makeCloud(device, kCount, degree, false, false, false, 100 + degree);
        const std::string path = (dir / ("degree" + std::to_string(degree) + ".ply")).string();
        PLYWriter::write(*original, path);

        std::string elementLine;
        const auto names = headerProperties(path, elementLine);
        CHECK(elementLine == "element vertex " + std::to_string(kCount), "header element line: " + elementLine);
        const size_t restCount = static_cast<size_t>(std::count_if(
            names.begin(), names.end(), [](const std::string & n) { return n.rfind("f_rest_", 0) == 0; }));
        CHECK(restCount == 3 * static_cast<size_t>((degree + 1) * (degree + 1) - 1),
              "degree " + std::to_string(degree) + ": f_rest count " + std::to_string(restCount));
        CHECK(hasProperty(names, "nx") && hasProperty(names, "ny") && hasProperty(names, "nz"),
              "nx ny nz missing (INRIA layout)");
        for (const char * r3dg : {"base_color_0", "base_color_1", "base_color_2", "roughness",
                                  "visibility_dc_0", "metallic"})
        {
            CHECK(!hasProperty(names, r3dg),
                  std::string("relight-free cloud wrote R3DG property ") + r3dg);
        }

        auto loaded = PLYReader::read(device, path);
        CHECK(loaded->normals().empty() && loaded->baseColorRoughness().empty()
                  && loaded->visibility().empty() && loaded->metallic().empty(),
              "relight-free cloud read back with relight streams");
        if (int r = compareClouds(*original, *loaded, "degree " + std::to_string(degree), false))
        {
            return r;
        }
    }
    return 0;
}

// Relight streams: a full R3DG cloud (degree 3, visibility, metallic) and a
// partial one (degree 1, no visibility, no metallic).
int testRelight(Device & device, const std::filesystem::path & dir)
{
    std::cout << "round trip, relight (R3DG attributes)" << std::endl;
    {
        auto original = makeCloud(device, 4000, 3, true, true, true, 200);
        const std::string path = (dir / "r3dg_full.ply").string();
        PLYWriter::write(*original, path);

        std::string elementLine;
        const auto names = headerProperties(path, elementLine);
        for (const char * required : {"base_color_0", "base_color_2", "roughness", "visibility_dc_0",
                                      "visibility_rest_14", "metallic", "nx"})
        {
            CHECK(hasProperty(names, required), std::string("R3DG cloud missing property ") + required);
        }
        auto loaded = PLYReader::read(device, path);
        CHECK(!loaded->metallic().empty() && !loaded->visibility().empty(),
              "R3DG cloud: metallic or visibility did not read back");
        if (int r = compareClouds(*original, *loaded, "R3DG degree 3", true))
        {
            return r;
        }
    }
    {
        // No visibility, no metallic: those properties must be absent and stay absent.
        auto original = makeCloud(device, 2000, 1, true, false, false, 201);
        const std::string path = (dir / "r3dg_partial.ply").string();
        PLYWriter::write(*original, path);

        std::string elementLine;
        const auto names = headerProperties(path, elementLine);
        CHECK(!hasProperty(names, "visibility_dc_0") && !hasProperty(names, "metallic"),
              "partial R3DG cloud wrote visibility or metallic it does not have");
        auto loaded = PLYReader::read(device, path);
        CHECK(loaded->visibility().empty() && loaded->metallic().empty(),
              "partial R3DG cloud read back visibility or metallic");
        if (int r = compareClouds(*original, *loaded, "R3DG degree 1, partial", true))
        {
            return r;
        }
    }
    {
        // Relight without normals: the documented rule is that nx ny nz are zeros and
        // the reader hands back zero normals (not an absent stream).
        auto original = makeCloud(device, 500, 0, true, false, false, 202);
        original->normals().clear();
        const std::string path = (dir / "r3dg_no_normals.ply").string();
        PLYWriter::write(*original, path);
        auto loaded = PLYReader::read(device, path);
        CHECK(loaded->normals().size() == original->count(), "no-normals R3DG: normals stream absent");
        for (const glm::vec4 & n : loaded->normals())
        {
            CHECK(n.x == 0.0f && n.y == 0.0f && n.z == 0.0f, "no-normals R3DG: normal is not zero");
        }
        std::cout << "  normals absent in the cloud -> zero normals on read (documented)" << std::endl;
    }
    return 0;
}

// Base color and roughness outside the representable open range are clamped,
// and the file still reads back inside it, at the clamp boundary.
int testRangeClamp(Device & device, const std::filesystem::path & dir)
{
    std::cout << "base color / roughness clamp" << std::endl;
    auto original = makeCloud(device, 4, 0, true, false, false, 300);
    const float bc[4][3] = {{0.0f, 0.03f, 0.5f}, {0.8f, 1.0f, 0.2f}, {0.5f, 0.5f, 0.5f}, {0.3f, 0.3f, 0.3f}};
    const float rough[4] = {0.09f, 0.0f, 1.0f, 0.5f};
    for (size_t i = 0; i < 4; ++i)
    {
        original->baseColorRoughness()[i] = glm::vec4(bc[i][0], bc[i][1], bc[i][2], rough[i]);
    }
    const std::string path = (dir / "clamp.ply").string();
    PLYWriter::write(*original, path);
    auto loaded = PLYReader::read(device, path);
    // Clamp boundary in the reader's output: 0.03 + 0.77 * eps, 0.80 - 0.77 * eps.
    const double lo = 0.03 + 0.77 * kClampEps;
    const double hi = 0.80 - 0.77 * kClampEps;
    for (size_t i = 0; i < 2; ++i)
    {
        for (int c = 0; c < 3; ++c)
        {
            const double v = loaded->baseColorRoughness()[i][c];
            CHECK(std::isfinite(v) && v >= lo - 1e-6 && v <= hi + 1e-6,
                  "clamped base color outside the reader's range");
        }
    }
    CHECK(std::abs(loaded->baseColorRoughness()[0].x - lo) < 1e-5, "0.0 base color not clamped to the low edge");
    CHECK(std::abs(loaded->baseColorRoughness()[1].x - hi) < 1e-5, "0.8 base color not clamped to the high edge");
    // Roughness: 0.09 and 0.0 clamp to the low edge, 1.0 to the high edge (0.09 + 0.9).
    CHECK(std::abs(loaded->baseColorRoughness()[0].w - (0.09 + 0.9 * kClampEps)) < 1e-6,
          "0.09 roughness not clamped to the low edge");
    CHECK(std::abs(loaded->baseColorRoughness()[1].w - (0.09 + 0.9 * kClampEps)) < 1e-6,
          "0.0 roughness not clamped to the low edge");
    CHECK(std::abs(loaded->baseColorRoughness()[2].w - (0.99 - 0.9 * kClampEps)) < 1e-5,
          "1.0 roughness not clamped to the high edge");
    std::cout << "  base color 0.0 -> " << loaded->baseColorRoughness()[0].x << ", 1.0 -> "
              << loaded->baseColorRoughness()[1].y << " (clamp edges " << lo << " / " << hi << ")"
              << std::endl;
    return 0;
}

// Things that must throw, and the write failures that must not damage an existing file.
int testRejectionsAndAtomicity(Device & device, const std::filesystem::path & dir)
{
    std::cout << "rejections and atomic replacement" << std::endl;
    const std::string path = (dir / "reject.ply").string();

    auto expectThrow = [&](const std::string & label, auto && write, const std::string & needle) -> int
    {
        try
        {
            write();
        }
        catch (const std::exception & e)
        {
            const std::string message = e.what();
            CHECK(message.find(needle) != std::string::npos,
                  label + ": message lacks '" + needle + "': " + message);
            std::cout << "  " << label << ": threw as expected" << std::endl;
            return 0;
        }
        std::cerr << "FAILED: " << label << " did not throw" << std::endl;
        return 1;
    };

    // Empty cloud.
    {
        GaussianCloud empty(device);
        if (int r = expectThrow("empty cloud", [&] { PLYWriter::write(empty, path); }, "empty")) return r;
    }
    // Zero scale on Gaussian 7 names it.
    {
        auto bad = makeCloud(device, 50, 0, false, false, false, 400);
        bad->scales()[7].y = 0.0f;
        if (int r = expectThrow("zero scale", [&] { PLYWriter::write(*bad, path); }, "Gaussian 7")) return r;
    }
    // Negative scale.
    {
        auto bad = makeCloud(device, 50, 0, false, false, false, 401);
        bad->scales()[3].x = -1.0f;
        if (int r = expectThrow("negative scale", [&] { PLYWriter::write(*bad, path); }, "Gaussian 3")) return r;
    }
    // SH degree above what the 16-coefficient storage can hold.
    {
        auto bad = makeCloud(device, 50, 0, false, false, false, 402);
        bad->setShDegree(4);
        if (int r = expectThrow("shDegree 4", [&] { PLYWriter::write(*bad, path); }, "above 3")) return r;
    }
    // Normals without base color cannot be read back.
    {
        auto bad = makeCloud(device, 50, 0, false, false, false, 403);
        bad->normals().resize(50, glm::vec4(0.0f, 0.0f, 1.0f, 0.0f));
        if (int r = expectThrow("normals without base color", [&] { PLYWriter::write(*bad, path); },
                                "baseColorRoughness"))
        {
            return r;
        }
    }
    // Unwritable location.
    {
        auto good = makeCloud(device, 50, 0, false, false, false, 404);
        const std::string missing = (dir / "no_such_dir" / "x.ply").string();
        if (int r = expectThrow("missing directory", [&] { PLYWriter::write(*good, missing); }, "failed to open"))
        {
            return r;
        }
    }

    // Atomic replacement: a second write replaces the first, and leaves no temp file.
    auto first = makeCloud(device, 100, 1, false, false, false, 410);
    auto second = makeCloud(device, 200, 2, false, false, false, 411);
    PLYWriter::write(*first, path);
    PLYWriter::write(*second, path);
    CHECK(!std::filesystem::exists(path + ".tmp"), "temp file left behind after a successful write");
    {
        auto loaded = PLYReader::read(device, path);
        if (int r = compareClouds(*second, *loaded, "replacement (200, degree 2)", false)) return r;
    }

    // A failed write (bad scale, mid-stream) must leave the existing file as it was.
    auto broken = makeCloud(device, 300, 2, false, false, false, 412);
    broken->scales()[250].z = 0.0f;
    if (int r = expectThrow("failed write over existing file", [&] { PLYWriter::write(*broken, path); },
                            "Gaussian 250"))
    {
        return r;
    }
    CHECK(!std::filesystem::exists(path + ".tmp"), "temp file left behind after a failed write");
    {
        auto loaded = PLYReader::read(device, path);
        CHECK(loaded->count() == 200, "failed write damaged the existing file");
        if (int r = compareClouds(*second, *loaded, "after failed write (unchanged)", false)) return r;
    }
    return 0;
}

// Load -> write -> load on a real scene, when one is on disk. Searches the
// argument, then the main checkout's assets/ (git-ignored, so often absent).
// `requested` is an absolute path given on the command line, or empty.
// Writes a .ply under a directory and file name that are not ASCII. The writer
// takes a UTF-8 std::string, so the bytes must match a plain-ASCII write of the
// same cloud (PLYReader itself is not exercised here).
int testNonAsciiPath(Device & device, const std::filesystem::path & dir)
{
    std::cout << "non-ASCII path" << std::endl;
    // "Ünïcödé 層" and "проект.ply" as UTF-8 byte escapes.
    const std::string unicodeDir = "\xC3\x9C"
                                   "n"
                                   "\xC3\xAF"
                                   "c"
                                   "\xC3\xB6"
                                   "d"
                                   "\xC3\xA9"
                                   " \xE5\xB1\xA4";
    const std::string cyrillicFile = "\xD0\xBF\xD1\x80\xD0\xBE\xD0\xB5\xD0\xBA\xD1\x82.ply";
    const std::filesystem::path nonAsciiDir = dir / std::filesystem::u8path(unicodeDir);
    std::filesystem::create_directories(nonAsciiDir);
    const std::filesystem::path file = nonAsciiDir / std::filesystem::u8path(cyrillicFile);
    const std::filesystem::path plain = dir / "non_ascii_reference.ply";
    const std::u8string fileUtf8 = file.u8string();
    const std::string path(reinterpret_cast<const char *>(fileUtf8.data()), fileUtf8.size());

    auto cloud = makeCloud(device, 64, 1, false, false, false, 800);
    PLYWriter::write(*cloud, plain.string());
    PLYWriter::write(*cloud, path);

    std::string plainBytes;
    std::string nonAsciiBytes;
    {
        std::ifstream plainIn(plain, std::ios::binary);
        plainBytes.assign(std::istreambuf_iterator<char>(plainIn), std::istreambuf_iterator<char>());
        std::ifstream nonAsciiIn(file, std::ios::binary);
        nonAsciiBytes.assign(std::istreambuf_iterator<char>(nonAsciiIn), std::istreambuf_iterator<char>());
    }
    CHECK(!nonAsciiBytes.empty(), "non-ASCII path: .ply not written under its UTF-8 name");
    CHECK(nonAsciiBytes == plainBytes, "non-ASCII path: bytes differ from the ASCII write of the same cloud");
    CHECK(!std::filesystem::exists(std::filesystem::u8path(path + ".tmp")),
          "non-ASCII path: no temporary file left behind");
    std::filesystem::remove_all(nonAsciiDir);
    std::filesystem::remove(plain);
    return 0;
}

int testRealScene(Device & device, const std::filesystem::path & dir, const std::string & requested)
{
    std::cout << "real scene round trip" << std::endl;
    std::string source = requested;
    if (source.empty())
    {
        const std::filesystem::path assets = "assets";
        const std::filesystem::path preferred = assets / "bonsai-7k.ply";
        std::error_code ec;
        if (std::filesystem::exists(preferred, ec))
        {
            source = preferred.string();
        }
        else if (std::filesystem::is_directory(assets, ec))
        {
            for (const auto & entry : std::filesystem::directory_iterator(assets, ec))
            {
                if (entry.path().extension() == ".ply")
                {
                    source = entry.path().string();
                    break;
                }
            }
        }
    }
    if (source.empty() || !std::filesystem::exists(source))
    {
        std::cout << "  skipped: no .ply found (assets/ is git-ignored and absent here)" << std::endl;
        return 0;
    }
    std::cout << "  source: " << source << std::endl;

    auto original = PLYReader::read(device, source);
    const std::string path = (dir / "real_roundtrip.ply").string();
    PLYWriter::write(*original, path);
    auto loaded = PLYReader::read(device, path);
    return compareClouds(*original, *loaded, "real scene", !original->baseColorRoughness().empty());
}

} // namespace

int main(int argc, char ** argv)
{
    // Resolved before ScratchDir changes the working directory.
    const std::string requestedScene =
        argc > 1 ? std::filesystem::absolute(argv[1]).string() : std::string();

    // Scratch files go to a temp folder, not the directory the test runs from.
    examples::ScratchDir scratch("ply_writer_test");
    Device device;
    int result = 0;
    try
    {
        // Headless: no surface, no shaders -- the reader and writer never touch the render path.
        device.initialize({}, /*enableValidation=*/true);

        const std::filesystem::path dir = scratch.path();
        result = testRoundTrip(device, dir);
        if (result == 0)
        {
            result = testRelight(device, dir);
        }
        if (result == 0)
        {
            result = testRangeClamp(device, dir);
        }
        if (result == 0)
        {
            result = testRejectionsAndAtomicity(device, dir);
        }
        if (result == 0)
        {
            result = testNonAsciiPath(device, dir);
        }
        if (result == 0)
        {
            result = testRealScene(device, dir, requestedScene);
        }
        if (result == 0)
        {
            // plain_ply_checks.cpp: golden bytes, NaN refusal, plain-PLY negatives, 1M round trip.
            result = runPlainPlyChecks(device, dir) == 0 ? 0 : 1;
        }

        if (result == 0)
        {
            std::cout << "PLYWriter test OK" << std::endl;
        }
    }
    catch (const std::exception & e)
    {
        std::cerr << "PLYWriter test FAILED: " << e.what() << std::endl;
        result = 1;
    }

    device.shutdown();
    return result;
}
