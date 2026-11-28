// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Helpers shared by the edit_test sources (main.cpp, merge_section.cpp,
// renderer_section.cpp): the check counters, a deterministic xorshift32, bit
// comparisons, the synthetic layer clouds, and the comparison of a merged cloud
// against the host mergeLayers() reference with the Section F tolerances.
// Internal to the test; not installed.

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "sabercat/edit_pass.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/layer_transform.h"
#include "sabercat/render_config.h"

namespace edit_test
{

inline int g_checks = 0;
inline int g_failures = 0;

inline void check(bool ok, const std::string & what)
{
    ++g_checks;
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok)
    {
        ++g_failures;
    }
}

// Deterministic xorshift32. The state is never zero, so it never sticks.
class XorShift32
{
  public:
    explicit XorShift32(uint32_t seed) : m_state(seed != 0 ? seed : 0x2545F491u) {}

    uint32_t next()
    {
        uint32_t x = m_state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        m_state = x;
        return x;
    }

    // Uniform in [0, 1), from the top 24 bits.
    float unit()
    {
        return static_cast<float>(next() >> 8) * (1.0f / 16777216.0f);
    }

    // Uniform in [-1, 1).
    float signedUnit()
    {
        return 2.0f * unit() - 1.0f;
    }

    // Uniform in [0, bound).
    uint32_t below(uint32_t bound)
    {
        return next() % bound;
    }

  private:
    uint32_t m_state;
};

// The bit pattern of a float, for comparisons that must not treat -0.0 and 0.0 as equal.
inline uint32_t bitsOf(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof bits);
    return bits;
}

inline bool sameBits(float a, float b)
{
    return bitsOf(a) == bitsOf(b);
}

// True when both vectors have the same size and the same bytes. The comparison
// is on bit patterns, so -0.0 and 0.0 differ.
template <typename T>
bool bitEqual(const std::vector<T> & a, const std::vector<T> & b)
{
    static_assert(std::is_trivially_copyable<T>::value,
                  "bitEqual needs trivially copyable elements");
    return a.size() == b.size()
           && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

// Every host stream of a cloud, copied out.
struct Streams
{
    std::vector<glm::vec4> positions;
    std::vector<glm::vec4> scales;
    std::vector<glm::vec4> rotations;
    std::vector<glm::vec4> shCoefficients;
    std::vector<float> opacities;
    std::vector<glm::vec4> normals;
    std::vector<glm::vec4> baseColorRoughness;
    std::vector<float> metallic;
    std::vector<float> visibility;
};

inline Streams copyStreams(const sabercat::GaussianCloud & cloud)
{
    Streams s;
    s.positions = cloud.positions();
    s.scales = cloud.scales();
    s.rotations = cloud.rotations();
    s.shCoefficients = cloud.shCoefficients();
    s.opacities = cloud.opacities();
    s.normals = cloud.normals();
    s.baseColorRoughness = cloud.baseColorRoughness();
    s.metallic = cloud.metallic();
    s.visibility = cloud.visibility();
    return s;
}

// --- Synthetic layers -------------------------------------------------------------

struct SyntheticSpec
{
    size_t count = 0;
    uint32_t shDegree = 0;
    bool relight = false; // all four relightable streams
    bool antialiased = false;
    uint32_t seed = 1;
};

inline glm::vec3 signedVec3(XorShift32 & rng)
{
    const float x = rng.signedUnit();
    const float y = rng.signedUnit();
    const float z = rng.signedUnit();
    return glm::vec3(x, y, z);
}

inline glm::vec3 unitVec3(XorShift32 & rng)
{
    const float x = rng.unit();
    const float y = rng.unit();
    const float z = rng.unit();
    return glm::vec3(x, y, z);
}

// A host cloud with every host vector filled from `spec.seed`, uploaded to the GPU.
// Positions lie within +-5. Every 97th quaternion is scaled by 1.0001, so it is
// deliberately not unit length. SH coefficients above the degree are zero, as a
// loader pads them. times() stays 0, the reserved schema value.
inline std::unique_ptr<sabercat::GaussianCloud> makeSynthetic(sabercat::Device & device,
                                                              const SyntheticSpec & spec)
{
    auto cloud = std::make_unique<sabercat::GaussianCloud>(device);
    const size_t n = spec.count;
    cloud->initialize(n);
    cloud->setShDegree(spec.shDegree);
    cloud->setAntialiased(spec.antialiased);

    XorShift32 rng(spec.seed);
    const uint32_t usedCoefficients = (spec.shDegree + 1) * (spec.shDegree + 1);
    const size_t perGaussian = static_cast<size_t>(sabercat::kShCoefficientsPerGaussian);
    for (size_t i = 0; i < n; ++i)
    {
        cloud->positions()[i] = glm::vec4(5.0f * signedVec3(rng), 0.0f);
        cloud->scales()[i] = glm::vec4(glm::vec3(0.02f) + 0.3f * unitVec3(rng), 0.0f);

        const glm::vec3 qxyz = signedVec3(rng);
        const float qw = rng.signedUnit();
        glm::vec4 q(qxyz, qw);
        const float qLength = glm::length(q);
        q = qLength > 1e-3f ? q / qLength : glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        if (i % 97 == 0)
        {
            q *= 1.0001f;
        }
        cloud->rotations()[i] = q;

        cloud->opacities()[i] = rng.unit();
        for (size_t k = 0; k < perGaussian; ++k)
        {
            glm::vec4 & c = cloud->shCoefficients()[i * perGaussian + k];
            c = glm::vec4(0.0f);
            if (static_cast<uint32_t>(k) < usedCoefficients)
            {
                c = glm::vec4(0.5f * signedVec3(rng), 0.0f);
            }
        }
    }

    if (spec.relight)
    {
        cloud->normals().resize(n);
        cloud->baseColorRoughness().resize(n);
        cloud->metallic().resize(n);
        cloud->visibility().resize(n * sabercat::kVisibilityCoefficients);
        for (size_t i = 0; i < n; ++i)
        {
            glm::vec3 normal = signedVec3(rng);
            const float normalLength = glm::length(normal);
            normal = normalLength > 1e-3f ? normal / normalLength : glm::vec3(0.0f, 0.0f, 1.0f);
            cloud->normals()[i] = glm::vec4(normal, 0.0f);

            const glm::vec3 albedo = unitVec3(rng);
            const float roughness = rng.unit();
            cloud->baseColorRoughness()[i] = glm::vec4(albedo, roughness);
            cloud->metallic()[i] = rng.unit();
        }
        for (float & v : cloud->visibility())
        {
            v = rng.signedUnit();
        }
    }

    cloud->computeBounds();
    cloud->uploadToGPU();
    return cloud;
}

// --- Transforms -------------------------------------------------------------------

inline sabercat::LayerTransform identityTransform()
{
    return sabercat::LayerTransform{};
}

inline sabercat::LayerTransform transformT1()
{
    sabercat::LayerTransform t;
    t.rotation = glm::angleAxis(0.7f, glm::normalize(glm::vec3(1.0f, 2.0f, 3.0f)));
    t.translation = glm::vec3(1.5f, -2.0f, 0.25f);
    t.scale = 1.7f;
    return t;
}

inline sabercat::LayerTransform transformT2()
{
    sabercat::LayerTransform t;
    t.rotation = glm::angleAxis(-2.1f, glm::normalize(glm::vec3(0.0f, 1.0f, 0.0f)));
    t.translation = glm::vec3(-10.0f, 3.0f, 7.0f);
    t.scale = 0.35f;
    return t;
}

// --- Merged-stream comparison -----------------------------------------------------

// Where one layer's Gaussians sit in the merged cloud.
struct LayerSpan
{
    size_t first = 0;
    size_t count = 0;
    const sabercat::GaussianCloud * cloud = nullptr;
    sabercat::LayerTransform transform;
};

inline std::vector<LayerSpan> layerSpans(const std::vector<sabercat::MergeLayer> & layers)
{
    std::vector<LayerSpan> spans;
    size_t first = 0;
    for (const sabercat::MergeLayer & layer : layers)
    {
        LayerSpan s;
        s.first = first;
        s.count = layer.first->count();
        s.cloud = layer.first;
        s.transform = layer.second;
        spans.push_back(s);
        first += s.count;
    }
    return spans;
}

// The worst error of one stream. `bad` counts the components outside their limit.
struct StreamReport
{
    double maxErr = 0.0; // largest |d| (ulps for scales)
    double worst = 0.0;  // largest |d| / limit
    size_t bad = 0;
    bool bitExact = false;
};

inline StreamReport sizeMismatch()
{
    StreamReport r;
    r.bad = 1;
    r.maxErr = std::numeric_limits<double>::infinity();
    return r;
}

// Records one component error d against its limit. A NaN fails.
inline void account(StreamReport & r, double d, double limit)
{
    if (!(d <= limit))
    {
        ++r.bad;
    }
    r.maxErr = std::max(r.maxErr, d);
    r.worst = std::max(r.worst, d / limit);
}

// Distance in units in the last place between two floats (the ordered-integer
// difference). Zero for +0.0 against -0.0; infinite when either is NaN.
inline double ulpDistance(float a, float b)
{
    if (std::isnan(a) || std::isnan(b))
    {
        return std::numeric_limits<double>::infinity();
    }
    int32_t ra = 0;
    int32_t rb = 0;
    std::memcpy(&ra, &a, sizeof ra);
    std::memcpy(&rb, &b, sizeof rb);
    const int64_t ia = ra < 0 ? static_cast<int64_t>(std::numeric_limits<int32_t>::min()) - ra : ra;
    const int64_t ib = rb < 0 ? static_cast<int64_t>(std::numeric_limits<int32_t>::min()) - rb : rb;
    return static_cast<double>(ia > ib ? ia - ib : ib - ia);
}

// Every component bit-equal.
template <typename T>
StreamReport bitReport(const std::vector<T> & want, const std::vector<T> & got)
{
    static_assert(sizeof(T) % sizeof(float) == 0, "bitReport compares float lanes");
    StreamReport r;
    r.bitExact = true;
    if (want.size() != got.size())
    {
        return sizeMismatch();
    }
    const size_t floats = want.size() * (sizeof(T) / sizeof(float));
    const float * a = reinterpret_cast<const float *>(want.data());
    const float * b = reinterpret_cast<const float *>(got.data());
    for (size_t i = 0; i < floats; ++i)
    {
        if (!sameBits(a[i], b[i]))
        {
            ++r.bad;
            r.maxErr = std::max(r.maxErr, static_cast<double>(std::fabs(a[i] - b[i])));
        }
    }
    return r;
}

// Means: |d| <= 2e-6 * (1 + |t| + s * |mu|) per xyz component, with t and s the
// layer's translation and scale and mu the source mean. The w lane is exact.
inline StreamReport meansReport(const std::vector<LayerSpan> & spans,
                                const std::vector<glm::vec4> & want,
                                const std::vector<glm::vec4> & got)
{
    StreamReport r;
    if (want.size() != got.size())
    {
        return sizeMismatch();
    }
    for (const LayerSpan & s : spans)
    {
        const double t = glm::length(glm::dvec3(s.transform.translation));
        const double scale = static_cast<double>(s.transform.scale);
        for (size_t j = 0; j < s.count; ++j)
        {
            const size_t i = s.first + j;
            const double mu = glm::length(glm::dvec3(glm::vec3(s.cloud->positions()[j])));
            const double tol = 2e-6 * (1.0 + t + scale * mu);
            for (int c = 0; c < 3; ++c)
            {
                account(r,
                        std::fabs(static_cast<double>(want[i][c]) - static_cast<double>(got[i][c])),
                        tol);
            }
            if (!sameBits(want[i].w, got[i].w))
            {
                ++r.bad;
            }
        }
    }
    return r;
}

// Rotations: the sign resolved (negate when dot < 0), then |d| <= 1e-6 per component.
inline StreamReport rotationsReport(const std::vector<glm::vec4> & want,
                                    const std::vector<glm::vec4> & got)
{
    StreamReport r;
    if (want.size() != got.size())
    {
        return sizeMismatch();
    }
    for (size_t i = 0; i < want.size(); ++i)
    {
        const glm::dvec4 q(want[i]);
        glm::dvec4 g(got[i]);
        if (glm::dot(q, g) < 0.0)
        {
            g = -g;
        }
        for (int c = 0; c < 4; ++c)
        {
            account(r, std::fabs(q[c] - g[c]), 1e-6);
        }
    }
    return r;
}

// Scales: within 2 ulp per xyz lane; the w lane exact.
inline StreamReport scalesReport(const std::vector<glm::vec4> & want,
                                 const std::vector<glm::vec4> & got)
{
    StreamReport r;
    if (want.size() != got.size())
    {
        return sizeMismatch();
    }
    for (size_t i = 0; i < want.size(); ++i)
    {
        for (int c = 0; c < 3; ++c)
        {
            account(r, ulpDistance(want[i][c], got[i][c]), 2.0);
        }
        if (!sameBits(want[i].w, got[i].w))
        {
            ++r.bad;
        }
    }
    return r;
}

// Normals: |d| <= 1e-6 per xyz component; the w lane exact.
inline StreamReport normalsReport(const std::vector<glm::vec4> & want,
                                  const std::vector<glm::vec4> & got)
{
    StreamReport r;
    if (want.size() != got.size())
    {
        return sizeMismatch();
    }
    for (size_t i = 0; i < want.size(); ++i)
    {
        for (int c = 0; c < 3; ++c)
        {
            account(r,
                    std::fabs(static_cast<double>(want[i][c]) - static_cast<double>(got[i][c])),
                    1e-6);
        }
        if (!sameBits(want[i].w, got[i].w))
        {
            ++r.bad;
        }
    }
    return r;
}

// SH: per Gaussian and band, |d| <= 2e-6 * max(1, max|c|) over the band's coefficients
// and the 3 channels of the reference. Coefficients of a band above the layer's
// degree must be bit-exact +0.0 in the merged cloud. The w lane is exact everywhere.
inline StreamReport shReport(const std::vector<LayerSpan> & spans,
                             const std::vector<glm::vec4> & want,
                             const std::vector<glm::vec4> & got)
{
    constexpr size_t kPerGaussian = static_cast<size_t>(sabercat::kShCoefficientsPerGaussian);
    constexpr size_t kBandStart[5] = {0, 1, 4, 9, 16};
    StreamReport r;
    if (want.size() != got.size())
    {
        return sizeMismatch();
    }
    for (const LayerSpan & s : spans)
    {
        const uint32_t degree = s.cloud->shDegree();
        for (size_t j = 0; j < s.count; ++j)
        {
            const size_t base = (s.first + j) * kPerGaussian;
            for (uint32_t band = 0; band < 4; ++band)
            {
                const size_t k0 = kBandStart[band];
                const size_t k1 = kBandStart[band + 1];
                if (band <= degree)
                {
                    double maxAbs = 0.0;
                    for (size_t k = k0; k < k1; ++k)
                    {
                        for (int c = 0; c < 3; ++c)
                        {
                            maxAbs = std::max(maxAbs,
                                              std::fabs(static_cast<double>(want[base + k][c])));
                        }
                    }
                    const double tol = 2e-6 * std::max(1.0, maxAbs);
                    for (size_t k = k0; k < k1; ++k)
                    {
                        for (int c = 0; c < 3; ++c)
                        {
                            account(r,
                                    std::fabs(static_cast<double>(want[base + k][c])
                                              - static_cast<double>(got[base + k][c])),
                                    tol);
                        }
                    }
                }
                else
                {
                    for (size_t k = k0; k < k1; ++k)
                    {
                        for (int c = 0; c < 3; ++c)
                        {
                            if (bitsOf(got[base + k][c]) != 0u)
                            {
                                ++r.bad;
                            }
                            r.maxErr = std::max(
                                r.maxErr, std::fabs(static_cast<double>(got[base + k][c])));
                        }
                    }
                }
                for (size_t k = k0; k < k1; ++k)
                {
                    if (!sameBits(want[base + k].w, got[base + k].w))
                    {
                        ++r.bad;
                    }
                }
            }
        }
    }
    return r;
}

inline void printReport(const std::string & label, const char * name, const StreamReport & r,
                        const char * unit)
{
    if (r.bitExact)
    {
        std::printf("  %s %-19s bit-exact, %zu mismatches (max |d| %.3g)\n", label.c_str(), name,
                    r.bad, r.maxErr);
    }
    else
    {
        std::printf("  %s %-19s max |d| %.3g%s, worst err/limit %.3g, failing %zu\n",
                    label.c_str(), name, r.maxErr, unit, r.worst, r.bad);
    }
}

inline void checkReport(const std::string & label, const char * name, const StreamReport & r,
                        const char * unit)
{
    printReport(label, name, r, unit);
    check(r.bad == 0, label + ": " + name + (r.bitExact ? " bit-exact" : " within tolerance"));
}

// Compares a merged cloud's streams against the host reference. `identity` (every
// layer at identity) demands bit-exact in every stream; otherwise the Section F
// rules apply. Absent streams must be absent in both (empty vectors).
inline void compareMergedStreams(const std::vector<LayerSpan> & spans, const Streams & want,
                                 const Streams & got, bool identity, const std::string & label)
{
    const StreamReport positions = identity ? bitReport(want.positions, got.positions)
                                            : meansReport(spans, want.positions, got.positions);
    const StreamReport scales = identity ? bitReport(want.scales, got.scales)
                                         : scalesReport(want.scales, got.scales);
    const StreamReport rotations = identity ? bitReport(want.rotations, got.rotations)
                                            : rotationsReport(want.rotations, got.rotations);
    const StreamReport sh = identity ? bitReport(want.shCoefficients, got.shCoefficients)
                                     : shReport(spans, want.shCoefficients, got.shCoefficients);
    const StreamReport opacities = bitReport(want.opacities, got.opacities);
    const StreamReport normals = identity ? bitReport(want.normals, got.normals)
                                          : normalsReport(want.normals, got.normals);
    const StreamReport baseColorRoughness =
        bitReport(want.baseColorRoughness, got.baseColorRoughness);
    const StreamReport metallic = bitReport(want.metallic, got.metallic);
    const StreamReport visibility = bitReport(want.visibility, got.visibility);

    checkReport(label, "positions", positions, "");
    checkReport(label, "scales", scales, " (ulp)");
    checkReport(label, "rotations", rotations, "");
    checkReport(label, "shCoefficients", sh, "");
    checkReport(label, "opacities", opacities, "");
    checkReport(label, "normals", normals, "");
    checkReport(label, "baseColorRoughness", baseColorRoughness, "");
    checkReport(label, "metallic", metallic, "");
    checkReport(label, "visibility", visibility, "");
}

// --- Sections (each in its own source file) ---------------------------------------

// F: EditPass::mergeLayers against the host mergeLayers(); merge_section.cpp.
void checkMergeLayers(sabercat::Device & device);
// G: Renderer::mergeLayersOnGpu; renderer_section.cpp.
void checkRendererMerge(sabercat::Device & device);
// H: visibility refit; renderer_section.cpp.
void checkRefit(sabercat::Device & device);
// I: informational merge timing on SABERCAT_EDIT_TEST_SCENE; renderer_section.cpp.
void timeMergeScene(sabercat::Device & device);
// J: selection (EditPass::select, the mask operations, SelectionMask) and K: the selection
// highlight (Renderer::setSelectionHighlight); selection_section.cpp.
void checkSelection(sabercat::Device & device);
void checkHighlight(sabercat::Device & device);
// L and M: the C4 stream edits (EditPass transform, snapshot, delete, duplicate) and the
// seeded random op sequence with undo and redo; ops_section.cpp.
void checkOps(sabercat::Device & device);
// N: the mask operations and selection above the 65535-workgroup limit (17,000,003
// Gaussians), with the elapsed time; large_section.cpp.
void checkLarge(sabercat::Device & device);
// O: the borrowed-buffer guard (ownership model B): the copying operations are refused, and
// select and an identity transform accept a borrowed layer; borrow_section.cpp.
void checkBorrowGuard(sabercat::Device & device);

} // namespace edit_test
