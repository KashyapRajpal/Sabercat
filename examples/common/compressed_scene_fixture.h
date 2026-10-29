// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Shared by ksplat_reader_test and sog_reader_test: the synthetic source scene
// both formats' fixtures were converted from, plus the machinery to compare a
// decoded scene against its source.
//
// Why a shared SOURCE scene: .ksplat and .sog are produced by their reference
// tools (GaussianSplats3D's SplatBufferGenerator, PlayCanvas splat-transform)
// FROM a .ply, and both tools reorder Gaussians (spatial buckets / Morton
// order) and quantize them. So the honest check of a reader is "does the file
// decode back to the .ply it was made from, within that format's documented
// quantization error" -- with the .ply loaded through the already-verified
// PLYReader. The fixtures in each test's fixtures/ directory are the reference
// tools' output for makeSyntheticScene(); generate_fixtures.mjs (beside this
// file) regenerates them from `<test> --write-source-ply`.
//
// The scene is generated, not committed, and deterministic (a fixed xorshift
// sequence, no libm calls), so it is bit-identical on every machine -- which
// the committed fixtures depend on.

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

#include "sabercat/camera.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/renderer.h"

namespace fixture
{

// --- The synthetic source scene ---

// One 3DGS .ply vertex, in file order (the reference trainer's property
// order, normals included so converters see a typical file).
struct PlyVertex
{
    float position[3];
    float normal[3];
    float fdc[3];
    float frest[45]; // channel-major: R1..R15, G1..G15, B1..B15
    float opacity;   // pre-sigmoid
    float scale[3];  // log-space
    float rot[4];    // w, x, y, z, unnormalized
};
static_assert(sizeof(PlyVertex) == 62 * sizeof(float), "PlyVertex must be tightly packed");

// 300 Gaussians, SH degree 3. Shaped to exercise what the formats encode:
//   - 270 on a jittered grid inside ONE 5-unit .ksplat bucket block (blocks
//     are anchored at the scene minimum, pinned to (-12,-12,-12) by vertex 0),
//     so a level-1/2 file has a FULL 256-splat bucket and a partially-filled
//     one in the same block; 30 spread over [-11, 11]^3 for many partial
//     buckets and both signs through SOG's log-encoded positions.
//   - minimum spacing ~0.16, far above any format's position error (<2e-3),
//     so decoded Gaussians match their source unambiguously by position.
//   - scales straddle 1.0 (log in [-4.5, 0.7]) so a spurious or missing exp()
//     is visible in both directions; four distinct non-zero quaternion lanes
//     of mixed sign and length != 1, so a lane-order error cannot hide.
//   - opacities with sigmoid in ~[0.05, 0.98] (all above .ksplat's default
//     1/255 alpha-removal threshold, so no Gaussian is dropped).
inline std::vector<PlyVertex> makeSyntheticScene()
{
    uint32_t state = 0x9E3779B9u;
    // xorshift32 -> [0, 1) with 24 exact bits: no libm, no platform variance.
    const auto next = [&state]()
    {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        return static_cast<float>(state >> 8) * (1.0f / 16777216.0f);
    };
    const auto uniform = [&next](float lo, float hi) { return lo + (hi - lo) * next(); };

    std::vector<PlyVertex> scene(300);
    for (size_t i = 0; i < scene.size(); ++i)
    {
        PlyVertex & v = scene[i];
        if (i == 0)
        {
            v.position[0] = v.position[1] = v.position[2] = -12.0f;
        }
        else if (i < 271)
        {
            const size_t slot = i - 1; // 7 x 7 x 6 grid, spacing 0.28
            const float grid[3] = {static_cast<float>(slot % 7), static_cast<float>((slot / 7) % 7),
                                   static_cast<float>(slot / 49)};
            for (int a = 0; a < 3; ++a)
            {
                v.position[a] = -1.5f + grid[a] * 0.28f + uniform(-0.06f, 0.06f);
            }
        }
        else
        {
            for (int a = 0; a < 3; ++a)
            {
                v.position[a] = uniform(-11.0f, 11.0f);
            }
        }
        v.normal[0] = v.normal[1] = v.normal[2] = 0.0f;
        for (float & c : v.fdc)
        {
            c = uniform(-1.6f, 1.6f);
        }
        for (float & c : v.frest)
        {
            c = uniform(-0.45f, 0.45f);
        }
        v.opacity = uniform(-3.0f, 4.0f);
        for (float & s : v.scale)
        {
            s = uniform(-4.5f, 0.7f);
        }
        for (float & q : v.rot)
        {
            // |lane| in [0.15, 1], random sign: never ~0, rarely equal.
            const float magnitude = uniform(0.15f, 1.0f);
            q = next() < 0.5f ? -magnitude : magnitude;
        }
    }
    return scene;
}

inline void writePly(const std::string & path, const std::vector<PlyVertex> & scene)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file << "ply\nformat binary_little_endian 1.0\nelement vertex " << scene.size() << "\n";
    for (const char * name : {"x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"})
    {
        file << "property float " << name << "\n";
    }
    for (int k = 0; k < 45; ++k)
    {
        file << "property float f_rest_" << k << "\n";
    }
    for (const char * name : {"opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"})
    {
        file << "property float " << name << "\n";
    }
    file << "end_header\n";
    file.write(reinterpret_cast<const char *>(scene.data()),
               static_cast<std::streamsize>(scene.size() * sizeof(PlyVertex)));
}

// --- Matching a reordered decode back to its source ---

// For each decoded Gaussian, the index of the nearest source Gaussian within
// `tolerance` (or -1). A uniform grid of cell size `tolerance`, so each query
// looks at 27 cells; O(n) for any scene whose spacing is above the tolerance.
inline std::vector<int64_t> matchByPosition(const std::vector<glm::vec4> & decoded,
                                            const std::vector<glm::vec4> & source, float tolerance)
{
    const auto cellOf = [tolerance](float v) { return static_cast<int64_t>(std::floor(v / tolerance)); };
    const auto key = [](int64_t x, int64_t y, int64_t z)
    {
        return static_cast<uint64_t>((x * 73856093) ^ (y * 19349663) ^ (z * 83492791));
    };
    std::unordered_multimap<uint64_t, size_t> grid;
    grid.reserve(source.size());
    for (size_t i = 0; i < source.size(); ++i)
    {
        grid.emplace(key(cellOf(source[i].x), cellOf(source[i].y), cellOf(source[i].z)), i);
    }

    std::vector<int64_t> match(decoded.size(), -1);
    for (size_t i = 0; i < decoded.size(); ++i)
    {
        const glm::vec3 p(decoded[i]);
        const int64_t cx = cellOf(p.x);
        const int64_t cy = cellOf(p.y);
        const int64_t cz = cellOf(p.z);
        float best = tolerance;
        for (int64_t dx = -1; dx <= 1; ++dx)
        {
            for (int64_t dy = -1; dy <= 1; ++dy)
            {
                for (int64_t dz = -1; dz <= 1; ++dz)
                {
                    const auto range = grid.equal_range(key(cx + dx, cy + dy, cz + dz));
                    for (auto it = range.first; it != range.second; ++it)
                    {
                        const float d = glm::length(p - glm::vec3(source[it->second]));
                        if (d <= best)
                        {
                            best = d;
                            match[i] = static_cast<int64_t>(it->second);
                        }
                    }
                }
            }
        }
    }
    return match;
}

// Angle of the rotation taking one unit quaternion to the other (q and -q are
// the same rotation, hence the sign alignment). Chord form, in double:
// |a - b| = 2 sin(angle / 4) stays accurate near zero, where acos(dot) in
// float quantizes small angles to ~1e-3 rad.
inline float rotationAngle(const glm::vec4 & a, const glm::vec4 & b)
{
    const double sign = glm::dot(a, b) < 0.0f ? -1.0 : 1.0;
    double chord = 0.0;
    for (int i = 0; i < 4; ++i)
    {
        const double d = static_cast<double>(a[i]) - sign * static_cast<double>(b[i]);
        chord += d * d;
    }
    return static_cast<float>(4.0 * std::asin(std::min(1.0, std::sqrt(chord) / 2.0)));
}

// Running max / mean of one error measure.
struct ErrorStat
{
    double max = 0.0;
    double sum = 0.0;
    size_t count = 0;

    void add(double value)
    {
        max = std::max(max, value);
        sum += value;
        ++count;
    }
    double mean() const
    {
        return count ? sum / static_cast<double>(count) : 0.0;
    }
};

inline void printStat(const char * label, const ErrorStat & stat, double bound = -1.0)
{
    if (bound >= 0.0)
    {
        std::printf("    %-34s max %-12.4g mean %-12.4g (bound %.4g)\n", label, stat.max, stat.mean(), bound);
    }
    else
    {
        std::printf("    %-34s max %-12.4g mean %.4g\n", label, stat.max, stat.mean());
    }
}

// --- Per-attribute comparison against the source ---

// Errors of a decoded scene against its source, over matched pairs. Scale is
// RELATIVE per axis (the formats store it linear-f16 or log-codebook, both of
// which make relative error the meaningful one); everything else absolute.
struct AttributeErrors
{
    ErrorStat position;      // |decoded - source|, per axis
    ErrorStat scaleRelative; // |decoded / source - 1|, per axis
    ErrorStat rotationAngle; // radians between the two rotations
    ErrorStat opacity;       // [0,1] units
    ErrorStat dc;            // raw f_dc units, per channel
    ErrorStat sh[3];         // bands 1, 2, 3, per component
    ErrorStat shAboveDegree; // |decoded| in bands above the decoded degree -- must be 0
    size_t matched = 0;
    size_t unmatched = 0;
    size_t sourceUsedTwice = 0;
};

// `ordered`: the two clouds are in the same order (a reference decode of the
// same file); otherwise pairs are found by position within `matchTolerance`.
// `perPair(decodedIndex, sourceIndex)` lets a caller apply per-Gaussian bounds.
inline AttributeErrors compareToSource(sabercat::GaussianCloud & decoded, sabercat::GaussianCloud & source,
                                       bool ordered, float matchTolerance,
                                       const std::function<void(size_t, size_t)> & perPair = {})
{
    using sabercat::kShCoefficientsPerGaussian;
    AttributeErrors errors;
    std::vector<int64_t> match;
    if (ordered)
    {
        match.resize(decoded.count());
        for (size_t i = 0; i < match.size(); ++i)
        {
            match[i] = i < source.count() ? static_cast<int64_t>(i) : -1;
        }
    }
    else
    {
        match = matchByPosition(decoded.positions(), source.positions(), matchTolerance);
    }

    std::vector<uint8_t> used(source.count(), 0);
    const uint32_t degree = decoded.shDegree();
    for (size_t d = 0; d < decoded.count(); ++d)
    {
        if (match[d] < 0)
        {
            ++errors.unmatched;
            continue;
        }
        const size_t s = static_cast<size_t>(match[d]);
        errors.sourceUsedTwice += used[s]++ ? 1 : 0;
        ++errors.matched;

        for (int a = 0; a < 3; ++a)
        {
            errors.position.add(std::abs(decoded.positions()[d][a] - source.positions()[s][a]));
            errors.scaleRelative.add(std::abs(decoded.scales()[d][a] / source.scales()[s][a] - 1.0f));
        }
        errors.rotationAngle.add(rotationAngle(decoded.rotations()[d], source.rotations()[s]));
        errors.opacity.add(std::abs(decoded.opacities()[d] - source.opacities()[s]));

        const glm::vec4 * dsh = &decoded.shCoefficients()[d * kShCoefficientsPerGaussian];
        const glm::vec4 * ssh = &source.shCoefficients()[s * kShCoefficientsPerGaussian];
        for (int c = 0; c < 3; ++c)
        {
            errors.dc.add(std::abs(dsh[0][c] - ssh[0][c]));
        }
        for (int k = 1; k < kShCoefficientsPerGaussian; ++k)
        {
            const uint32_t band = k < 4 ? 1 : (k < 9 ? 2 : 3);
            for (int c = 0; c < 3; ++c)
            {
                if (band <= degree)
                {
                    errors.sh[band - 1].add(std::abs(dsh[k][c] - ssh[k][c]));
                }
                else
                {
                    errors.shAboveDegree.add(std::abs(dsh[k][c]));
                }
            }
        }
        if (perPair)
        {
            perPair(d, s);
        }
    }
    return errors;
}

inline void printAttributeErrors(const AttributeErrors & e)
{
    std::printf("    matched %zu, unmatched %zu, source matched twice %zu\n", e.matched, e.unmatched,
                e.sourceUsedTwice);
    printStat("position (abs)", e.position);
    printStat("scale (relative)", e.scaleRelative);
    printStat("rotation (radians)", e.rotationAngle);
    printStat("opacity (abs)", e.opacity);
    printStat("SH DC (abs, raw f_dc)", e.dc);
    printStat("SH band 1 (abs)", e.sh[0]);
    printStat("SH band 2 (abs)", e.sh[1]);
    printStat("SH band 3 (abs)", e.sh[2]);
    printStat("SH above file degree (|value|)", e.shAboveDegree);
}

// --- Rendering a decoded scene against its source ---

struct ImageDiff
{
    double rmse = 0.0;      // over RGB, display units [0,1]
    double maxAbs = 0.0;    // worst single channel
    double psnr = 0.0;      // dB, peak 1.0
    double meanCoverage = 0.0; // mean alpha of the reference image, so an empty render can't pass
};

// Renders `reference` and `candidate` through sabercat::Renderer from the same
// camera(s) -- framed on the reference's focus sphere, as apps/viewer frames a
// scene -- and diffs the images. Two views (front, and from +X) so view-
// dependent SH is seen from more than one direction. `maxShDegree` caps both
// renders (a format that stores fewer bands is compared at its own degree).
inline ImageDiff renderAndDiff(sabercat::Renderer & renderer, const std::string & reference,
                               const std::string & candidate, uint32_t maxShDegree)
{
    const uint32_t width = renderer.width();
    const uint32_t height = renderer.height();
    renderer.setMaxSHDegree(maxShDegree);

    renderer.loadScene(reference.c_str());
    const sabercat::SceneBounds bounds = renderer.sceneBounds();
    const float fy = static_cast<float>(height);
    const float distance = (bounds.focusRadius / 0.5f) * 1.15f; // tan(half fov) = (h/2)/fy
    const glm::vec3 directions[2] = {glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(1.0f, 0.0f, 0.0f)};

    std::vector<sabercat::Camera> cameras;
    for (const glm::vec3 & back : directions)
    {
        sabercat::Camera camera;
        camera.width = width;
        camera.height = height;
        camera.fx = camera.fy = fy;
        camera.cx = static_cast<float>(width) * 0.5f;
        camera.cy = static_cast<float>(height) * 0.5f;
        // The camera looks along its local -Z, so +Z of the view basis is
        // `back`; for the +X view that basis is a 90-degree turn about Y.
        const glm::vec3 eye = bounds.focusCenter + distance * back;
        camera.cameraToWorld = glm::inverse(glm::lookAt(eye, bounds.focusCenter, glm::vec3(0.0f, 1.0f, 0.0f)));
        cameras.push_back(camera);
    }

    const auto renderAll = [&]()
    {
        std::vector<std::vector<float>> images;
        for (const sabercat::Camera & camera : cameras)
        {
            renderer.setCamera(camera);
            renderer.forward();
            images.push_back(renderer.readImage());
        }
        return images;
    };
    const std::vector<std::vector<float>> referenceImages = renderAll();
    renderer.loadScene(candidate.c_str());
    const std::vector<std::vector<float>> candidateImages = renderAll();

    ImageDiff diff;
    double squared = 0.0;
    double coverage = 0.0;
    size_t samples = 0;
    size_t pixels = 0;
    for (size_t v = 0; v < referenceImages.size(); ++v)
    {
        for (size_t p = 0; p + 3 < referenceImages[v].size(); p += 4)
        {
            for (size_t c = 0; c < 3; ++c)
            {
                const double d = static_cast<double>(referenceImages[v][p + c]) - candidateImages[v][p + c];
                squared += d * d;
                diff.maxAbs = std::max(diff.maxAbs, std::abs(d));
                ++samples;
            }
            coverage += referenceImages[v][p + 3];
            ++pixels;
        }
    }
    diff.rmse = std::sqrt(squared / static_cast<double>(samples));
    diff.psnr = diff.rmse > 0.0 ? 20.0 * std::log10(1.0 / diff.rmse) : 999.0;
    diff.meanCoverage = coverage / static_cast<double>(pixels);
    return diff;
}

inline void printImageDiff(const char * label, const ImageDiff & diff)
{
    std::printf("    %-34s RMSE %.5f  max %.4f  PSNR %.2f dB  (reference coverage %.3f)\n", label,
                diff.rmse, diff.maxAbs, diff.psnr, diff.meanCoverage);
}

} // namespace fixture
