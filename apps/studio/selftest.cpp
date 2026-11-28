// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "selftest.h"

#include "borrowed_vulkan_host.h"
#include "shader_dir.h"
#include "studio_state.h"

#include "sabercat/device.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/layer_transform.h"
#include "sabercat/ply_writer.h"
#include "sabercat/project.h"
#include "sabercat/renderer.h"
#include "sabercat/scene_loader.h"
#include "sabercat/spz_reader.h"
#include "sabercat/spz_writer.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sabercat::studio
{

using sabercat::app::BorrowedVulkanHost;
using sabercat::app::resolveShaderDir;

namespace
{

namespace fs = std::filesystem;
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

void require(bool condition, const std::string & what)
{
    if (!condition)
    {
        throw std::runtime_error("selftest: " + what);
    }
}

double secondsSince(Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

uintmax_t fileBytes(const std::string & path)
{
    return fs::file_size(fs::u8path(path));
}

// Bit-for-bit equality of two host arrays (so -0.0 and 0.0 differ, as they should).
template <typename T>
bool sameBits(const std::vector<T> & a, const std::vector<T> & b)
{
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

bool sameBits(float a, float b)
{
    return std::memcmp(&a, &b, sizeof(float)) == 0;
}

bool sameTransform(const LayerTransform & a, const LayerTransform & b)
{
    return sameBits(a.rotation.w, b.rotation.w) && sameBits(a.rotation.x, b.rotation.x)
           && sameBits(a.rotation.y, b.rotation.y) && sameBits(a.rotation.z, b.rotation.z)
           && sameBits(a.translation.x, b.translation.x)
           && sameBits(a.translation.y, b.translation.y)
           && sameBits(a.translation.z, b.translation.z) && sameBits(a.scale, b.scale);
}

// Names the first stream that differs, or returns "" when the clouds match bit for bit.
std::string firstDifference(const GaussianCloud & a, const GaussianCloud & b)
{
    if (a.count() != b.count())
    {
        return "count";
    }
    if (a.shDegree() != b.shDegree())
    {
        return "shDegree";
    }
    if (a.antialiased() != b.antialiased())
    {
        return "antialiased";
    }
    if (!sameBits(a.positions(), b.positions()))
    {
        return "positions";
    }
    if (!sameBits(a.scales(), b.scales()))
    {
        return "scales";
    }
    if (!sameBits(a.rotations(), b.rotations()))
    {
        return "rotations";
    }
    if (!sameBits(a.opacities(), b.opacities()))
    {
        return "opacities";
    }
    if (!sameBits(a.shCoefficients(), b.shCoefficients()))
    {
        return "shCoefficients";
    }
    if (!sameBits(a.normals(), b.normals()))
    {
        return "normals";
    }
    if (!sameBits(a.baseColorRoughness(), b.baseColorRoughness()))
    {
        return "baseColorRoughness";
    }
    if (!sameBits(a.metallic(), b.metallic()))
    {
        return "metallic";
    }
    if (!sameBits(a.visibility(), b.visibility()))
    {
        return "visibility";
    }
    return "";
}

// Largest absolute difference over the first `lanes` components of two vec4 streams.
// A size mismatch or any non-finite difference counts as infinite.
double worstError(const std::vector<glm::vec4> & a, const std::vector<glm::vec4> & b, int lanes)
{
    if (a.size() != b.size())
    {
        return INFINITY;
    }
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        for (int k = 0; k < lanes; ++k)
        {
            const double d = std::fabs(static_cast<double>(a[i][k]) - static_cast<double>(b[i][k]));
            if (!std::isfinite(d))
            {
                return INFINITY;
            }
            worst = std::max(worst, d);
        }
    }
    return worst;
}

double worstError(const std::vector<float> & a, const std::vector<float> & b)
{
    if (a.size() != b.size())
    {
        return INFINITY;
    }
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        const double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (!std::isfinite(d))
        {
            return INFINITY;
        }
        worst = std::max(worst, d);
    }
    return worst;
}

// Records the largest error and the stream it came from.
struct Worst
{
    double value = 0.0;
    std::string stream;
    void add(const std::string & name, double error)
    {
        if (error > value)
        {
            value = error;
            stream = name;
        }
    }
};

// A cloud with every stream the formats carry: SH up to `shDegree` (zero above
// it), edge opacities (0, 1 and just inside each end), scales over six decades,
// both quaternion signs and, when `relight`, normals, base color and roughness,
// metallic and visibility. Fixed seed, so every run builds the same scene.
std::shared_ptr<GaussianCloud> makeSyntheticCloud(Device & device, size_t count, uint32_t shDegree,
                                                  bool relight, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_real_distribution<double> symmetric(-1.0, 1.0);
    std::normal_distribution<double> gauss(0.0, 1.0);
    const double logScaleLo = std::log(1e-6);
    const double logScaleHi = std::log(50.0);
    constexpr float kEdgeOpacities[6] = {0.0f, 1.0f, 1e-6f, 1.0f - 1e-6f, 1e-4f, 0.9999f};

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
                                 static_cast<float>(symmetric(rng) * 5.0), 0.0f);
        opacities[i] = i % 8 < 6 ? kEdgeOpacities[i % 8] : static_cast<float>(unit(rng));
        for (int c = 0; c < 3; ++c)
        {
            scales[i][c] = static_cast<float>(std::exp(logScaleLo + (logScaleHi - logScaleLo) * unit(rng)));
        }
        glm::dvec4 q(gauss(rng), gauss(rng), gauss(rng), gauss(rng));
        q /= glm::length(q);
        const double sign = (rng() & 1u) ? -1.0 : 1.0;
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
        auto & metallic = cloud->metallic();
        auto & visibility = cloud->visibility();
        normals.resize(count);
        baseColorRoughness.resize(count);
        metallic.resize(count);
        visibility.resize(count * kVisibilityCoefficients);
        for (size_t i = 0; i < count; ++i)
        {
            glm::dvec3 n(gauss(rng), gauss(rng), gauss(rng));
            n /= glm::length(n);
            normals[i] = glm::vec4(glm::vec3(n), 0.0f);
            baseColorRoughness[i] = glm::vec4(static_cast<float>(0.05 + 0.70 * unit(rng)),
                                              static_cast<float>(0.05 + 0.70 * unit(rng)),
                                              static_cast<float>(0.05 + 0.70 * unit(rng)),
                                              static_cast<float>(0.12 + 0.83 * unit(rng)));
            metallic[i] = i % 5 == 0 ? (i % 2 == 0 ? 0.0f : 1.0f) : static_cast<float>(unit(rng));
        }
        for (float & v : visibility)
        {
            v = static_cast<float>(unit(rng));
        }
    }

    cloud->computeBounds();
    return cloud;
}

// The synthetic project behind `--selftest <outDir>` with no scenes: layer A
// (SH degree 3, no relight streams) and layer B (SH degree 1, relight streams,
// rotated, moved and scaled). mergeLayers keeps a relight stream only when every
// layer carries it, so the merge, and the PLY written from it, has none.
void addSyntheticLayers(StudioState & state, Device & device)
{
    struct Spec
    {
        const char * name;
        size_t count;
        uint32_t shDegree;
        bool relight;
        uint32_t seed;
    };
    const Spec specs[] = {{"synthetic A", 3000, 3, false, 1001}, {"synthetic B", 2000, 1, true, 1002}};
    for (const Spec & spec : specs)
    {
        ProjectLayer layer;
        layer.cloud = makeSyntheticCloud(device, spec.count, spec.shDegree, spec.relight, spec.seed);
        layer.id = nextLayerId(state.project);
        layer.name = spec.name;
        state.project.layers.push_back(std::move(layer));
    }
    LayerTransform & placed = state.project.layers.back().transform;
    placed.rotation = glm::normalize(glm::angleAxis(glm::radians(30.0f), glm::vec3(0.0f, 1.0f, 0.0f)));
    placed.translation = glm::vec3(1.5f, 0.25f, -0.75f);
    placed.scale = 0.75f;
    state.selectedLayer = 0;
    state.dirty = true;
    state.sceneDirty = true;
}

// The largest error of one stream, and the largest ratio of an error to its
// tolerance (the check passes while every ratio is <= 1).
struct Bound
{
    double error = 0.0;
    double ratio = 0.0;
    void add(double err, double tolerance)
    {
        error = std::max(error, err);
        ratio = std::max(ratio, err / tolerance);
    }
};

// Compares a PLY reload with the in-memory merge it was written from, using the
// bounds examples/ply_writer_test derives. Positions and SH are stored raw, so
// they match bit for bit. Rotation is normalized on read (8 ulp, up to sign).
// Scale goes through log and exp (ulp * (|ln s| + 2), relative). Opacity goes
// through the writer's logit clamp and the reader's sigmoid, whose error is at
// most sigmoid' * |x| * 2^-24 plus a few half-ulps for exp and divide.
void requirePlyMatchesMerged(const GaussianCloud & merged, const GaussianCloud & loaded)
{
    constexpr double kUlp = std::numeric_limits<float>::epsilon();
    constexpr double kHalfUlpAtOne = 0x1p-24;
    const double eps = static_cast<double>(PLYWriter::kLogitEpsilon);
    const auto sigmoidTolerance = [&](double x) { return 0.25 * std::abs(x) * kHalfUlpAtOne + 3.0 * kHalfUlpAtOne; };

    require(loaded.shDegree() == merged.shDegree(), "PLY shDegree differs from the merge");
    require(merged.normals().empty() && loaded.normals().empty() && merged.visibility().empty()
                && loaded.visibility().empty() && merged.metallic().empty() && loaded.metallic().empty()
                && merged.baseColorRoughness().empty() && loaded.baseColorRoughness().empty(),
            "relight streams present after the merge or the PLY reload");

    Bound rotation;
    Bound scale;
    Bound opacity;
    double worstScaleRelative = 0.0;
    for (size_t i = 0; i < merged.count(); ++i)
    {
        for (int c = 0; c < 3; ++c)
        {
            require(sameBits(merged.positions()[i][c], loaded.positions()[i][c]),
                    "PLY position not bit-exact at " + std::to_string(i));
        }
        for (size_t k = 0; k < static_cast<size_t>(kShCoefficientsPerGaussian); ++k)
        {
            for (int c = 0; c < 3; ++c)
            {
                require(sameBits(merged.shCoefficients()[i * kShCoefficientsPerGaussian + k][c],
                                 loaded.shCoefficients()[i * kShCoefficientsPerGaussian + k][c]),
                        "PLY SH not bit-exact at " + std::to_string(i));
            }
        }

        const glm::vec4 & a = merged.rotations()[i];
        const glm::vec4 & b = loaded.rotations()[i];
        const double same = std::max({std::abs(a.x - b.x), std::abs(a.y - b.y), std::abs(a.z - b.z), std::abs(a.w - b.w)});
        const double flipped = std::max({std::abs(a.x + b.x), std::abs(a.y + b.y), std::abs(a.z + b.z), std::abs(a.w + b.w)});
        rotation.add(std::min(same, flipped), 8.0 * kUlp);

        for (int c = 0; c < 3; ++c)
        {
            const double s = static_cast<double>(merged.scales()[i][c]);
            const double r = static_cast<double>(loaded.scales()[i][c]);
            scale.add(std::abs(r - s), s * kUlp * (std::abs(std::log(s)) + 2.0));
            worstScaleRelative = std::max(worstScaleRelative, std::abs(r - s) / s);
        }

        const double expected = std::min(std::max(static_cast<double>(merged.opacities()[i]), eps), 1.0 - eps);
        opacity.add(std::abs(static_cast<double>(loaded.opacities()[i]) - expected),
                    sigmoidTolerance(std::log(expected / (1.0 - expected))));
    }

    std::cout << "PLY vs mergeLayers: positions and SH bit-exact" << std::endl;
    std::cout << "  rotation   worst " << rotation.error << " (tol 8 ulp, ratio " << rotation.ratio << ")" << std::endl;
    std::cout << "  scale      worst rel " << worstScaleRelative << " (tol ulp*(|ln s|+2), ratio " << scale.ratio << ")" << std::endl;
    std::cout << "  opacity    worst abs " << opacity.error << " (tol sigmoid bound, ratio " << opacity.ratio << ")" << std::endl;
    require(rotation.ratio <= 1.0, "PLY rotation outside tolerance");
    require(scale.ratio <= 1.0, "PLY scale outside tolerance");
    require(opacity.ratio <= 1.0, "PLY opacity outside tolerance");
}

// --- GPU editing model (studio_edit.h): checks E1 to E7 ---------------------

using MergeInputs = std::vector<std::pair<const GaussianCloud *, LayerTransform>>;

glm::dvec3 asDouble(const glm::vec3 & v)
{
    return glm::dvec3(v.x, v.y, v.z);
}

glm::dvec3 asDouble(const glm::vec4 & v)
{
    return glm::dvec3(v.x, v.y, v.z);
}

// The Gaussian `i` of a flat (4 floats per Gaussian) array.
glm::vec4 at4(const std::vector<float> & values, size_t i)
{
    return glm::vec4(values[4 * i], values[4 * i + 1], values[4 * i + 2], values[4 * i + 3]);
}

bool contains(const std::string & text, const std::string & needle)
{
    return text.find(needle) != std::string::npos;
}

// Captures what is written to std::cerr while it lives (the Logger's warnings).
class CerrCapture
{
  public:
    CerrCapture() : m_saved(std::cerr.rdbuf(m_text.rdbuf())) {}
    ~CerrCapture()
    {
        std::cerr.rdbuf(m_saved);
    }
    CerrCapture(const CerrCapture &) = delete;
    CerrCapture & operator=(const CerrCapture &) = delete;

    // How many times `needle` appears in the captured text.
    int count(const std::string & needle) const
    {
        const std::string text = m_text.str();
        int matches = 0;
        for (size_t at = text.find(needle); at != std::string::npos; at = text.find(needle, at + needle.size()))
        {
            ++matches;
        }
        return matches;
    }

  private:
    std::ostringstream m_text; // declared first: the redirect below needs it constructed
    std::streambuf * m_saved;
};

// One layer's host streams, copied bit for bit (a GaussianCloud cannot be copied).
struct LayerCopy
{
    size_t count = 0;
    uint32_t shDegree = 0;
    bool antialiased = false;
    std::vector<glm::vec4> positions;
    std::vector<glm::vec4> scales;
    std::vector<glm::vec4> rotations;
    std::vector<float> opacities;
    std::vector<glm::vec4> shCoefficients;
    std::vector<glm::vec4> normals;
    std::vector<glm::vec4> baseColorRoughness;
    std::vector<float> metallic;
    std::vector<float> visibility;
};

LayerCopy copyLayer(const GaussianCloud & cloud)
{
    LayerCopy copy;
    copy.count = cloud.count();
    copy.shDegree = cloud.shDegree();
    copy.antialiased = cloud.antialiased();
    copy.positions = cloud.positions();
    copy.scales = cloud.scales();
    copy.rotations = cloud.rotations();
    copy.opacities = cloud.opacities();
    copy.shCoefficients = cloud.shCoefficients();
    copy.normals = cloud.normals();
    copy.baseColorRoughness = cloud.baseColorRoughness();
    copy.metallic = cloud.metallic();
    copy.visibility = cloud.visibility();
    return copy;
}

// Names the first stream that differs, or "" when the two copies match bit for bit.
std::string differenceOf(const LayerCopy & a, const LayerCopy & b)
{
    if (a.count != b.count)
    {
        return "count";
    }
    if (a.shDegree != b.shDegree)
    {
        return "shDegree";
    }
    if (a.antialiased != b.antialiased)
    {
        return "antialiased";
    }
    if (!sameBits(a.positions, b.positions))
    {
        return "positions";
    }
    if (!sameBits(a.scales, b.scales))
    {
        return "scales";
    }
    if (!sameBits(a.rotations, b.rotations))
    {
        return "rotations";
    }
    if (!sameBits(a.opacities, b.opacities))
    {
        return "opacities";
    }
    if (!sameBits(a.shCoefficients, b.shCoefficients))
    {
        return "shCoefficients";
    }
    if (!sameBits(a.normals, b.normals))
    {
        return "normals";
    }
    if (!sameBits(a.baseColorRoughness, b.baseColorRoughness))
    {
        return "baseColorRoughness";
    }
    if (!sameBits(a.metallic, b.metallic))
    {
        return "metallic";
    }
    if (!sameBits(a.visibility, b.visibility))
    {
        return "visibility";
    }
    return "";
}

// A scene as flat arrays at the canonical strides (the layout of readGaussians).
struct SceneCopy
{
    uint32_t count = 0;
    std::vector<float> means;
    std::vector<float> scales;
    std::vector<float> rotations;
    std::vector<float> opacities;
    std::vector<float> shCoefficients;
};

// The renderer's current scene, read back from the GPU.
SceneCopy readScene(Renderer & renderer)
{
    const size_t sh = static_cast<size_t>(kShCoefficientsPerGaussian);
    SceneCopy scene;
    scene.count = renderer.gaussianCount();
    const size_t n = scene.count;
    scene.means.resize(n * 4);
    scene.scales.resize(n * 4);
    scene.rotations.resize(n * 4);
    scene.opacities.resize(n);
    scene.shCoefficients.resize(n * sh * 4);
    GaussianHostOutput out;
    out.count = scene.count;
    out.means = scene.means.data();
    out.scales = scene.scales.data();
    out.rotations = scene.rotations.data();
    out.opacities = scene.opacities.data();
    out.shCoefficients = scene.shCoefficients.data();
    renderer.readGaussians(out);
    return scene;
}

// A host cloud's streams in the same layout.
SceneCopy copyCloudScene(const GaussianCloud & cloud)
{
    const size_t sh = static_cast<size_t>(kShCoefficientsPerGaussian);
    const size_t n = cloud.count();
    SceneCopy scene;
    scene.count = static_cast<uint32_t>(n);
    scene.means.resize(n * 4);
    scene.scales.resize(n * 4);
    scene.rotations.resize(n * 4);
    scene.opacities.resize(n);
    scene.shCoefficients.resize(n * sh * 4);
    std::memcpy(scene.means.data(), cloud.positions().data(), n * 4 * sizeof(float));
    std::memcpy(scene.scales.data(), cloud.scales().data(), n * 4 * sizeof(float));
    std::memcpy(scene.rotations.data(), cloud.rotations().data(), n * 4 * sizeof(float));
    std::memcpy(scene.opacities.data(), cloud.opacities().data(), n * sizeof(float));
    std::memcpy(scene.shCoefficients.data(), cloud.shCoefficients().data(), n * sh * 4 * sizeof(float));
    return scene;
}

bool sameScene(const SceneCopy & a, const SceneCopy & b)
{
    return a.count == b.count && sameBits(a.means, b.means) && sameBits(a.scales, b.scales)
           && sameBits(a.rotations, b.rotations) && sameBits(a.opacities, b.opacities)
           && sameBits(a.shCoefficients, b.shCoefficients);
}

// The GPU merge's tolerances against a reference scene (the host merge of `layers`):
// means 2e-6 * (1 + |t| + s |mu|) for a Gaussian of a layer placed with translation t and
// scale s, mu its layer-local mean; rotations 1e-6, up to sign; scales 2 ulp; SH 2e-6 times
// max(1, the largest |c| of its band in the reference); opacity bit-exact. Prints the
// worst error of each stream and fails on the first one out of tolerance.
void requireSceneNear(const SceneCopy & got, const SceneCopy & want, const MergeInputs & layers,
                      const std::string & label)
{
    require(got.count == want.count, label + ": count " + std::to_string(got.count) + " != "
                                         + std::to_string(want.count));
    const size_t n = want.count;
    const size_t sh = static_cast<size_t>(kShCoefficientsPerGaussian);

    std::vector<double> bandMax(sh, 1.0);
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t k = 0; k < sh; ++k)
        {
            for (size_t c = 0; c < 3; ++c)
            {
                bandMax[k] = std::max(bandMax[k], std::fabs(static_cast<double>(want.shCoefficients[(i * sh + k) * 4 + c])));
            }
        }
    }

    Bound means;
    Bound rotations;
    Bound scales;
    Bound shBands;
    Bound opacity;
    size_t opacityMismatches = 0;
    size_t first = 0;
    for (const auto & layer : layers)
    {
        const GaussianCloud & cloud = *layer.first;
        const double t = glm::length(asDouble(layer.second.translation));
        const double s = static_cast<double>(layer.second.scale);
        for (size_t j = 0; j < cloud.count(); ++j)
        {
            const size_t i = first + j;
            const double muLength = glm::length(asDouble(cloud.positions()[j]));
            means.add(glm::length(asDouble(at4(got.means, i)) - asDouble(at4(want.means, i))),
                      2e-6 * (1.0 + t + s * muLength));

            const glm::vec4 a = at4(got.rotations, i);
            const glm::vec4 b = at4(want.rotations, i);
            double same = 0.0;
            double flipped = 0.0;
            for (int c = 0; c < 4; ++c)
            {
                same = std::max(same, std::fabs(static_cast<double>(a[c]) - static_cast<double>(b[c])));
                flipped = std::max(flipped, std::fabs(static_cast<double>(a[c]) + static_cast<double>(b[c])));
            }
            rotations.add(std::min(same, flipped), 1e-6);

            const glm::vec4 gs = at4(got.scales, i);
            const glm::vec4 ws = at4(want.scales, i);
            for (int c = 0; c < 3; ++c)
            {
                const float h = ws[c];
                const double ulp = static_cast<double>(std::nextafter(h, std::numeric_limits<float>::infinity()) - h);
                scales.add(std::fabs(static_cast<double>(gs[c]) - static_cast<double>(h)), 2.0 * ulp);
            }

            opacityMismatches += sameBits(got.opacities[i], want.opacities[i]) ? 0 : 1;

            for (size_t k = 0; k < sh; ++k)
            {
                for (size_t c = 0; c < 3; ++c)
                {
                    const size_t at = (i * sh + k) * 4 + c;
                    shBands.add(std::fabs(static_cast<double>(got.shCoefficients[at]) - static_cast<double>(want.shCoefficients[at])),
                                2e-6 * bandMax[k]);
                }
            }
        }
        first += cloud.count();
    }
    require(first == n, label + ": the layers hold " + std::to_string(first) + " Gaussians, not " + std::to_string(n));

    std::cout << label << ": " << n << " Gaussians" << std::endl;
    std::cout << "  means      worst " << means.error << " (ratio " << means.ratio << ")" << std::endl;
    std::cout << "  rotations  worst " << rotations.error << " (ratio " << rotations.ratio << ")" << std::endl;
    std::cout << "  scales     worst " << scales.error << " (ratio " << scales.ratio << ")" << std::endl;
    std::cout << "  SH         worst " << shBands.error << " (ratio " << shBands.ratio << ")" << std::endl;
    std::cout << "  opacities  " << (opacityMismatches == 0 ? "bit-exact" : "NOT bit-exact") << std::endl;
    require(means.ratio <= 1.0, label + ": means outside tolerance");
    require(rotations.ratio <= 1.0, label + ": rotations outside tolerance");
    require(scales.ratio <= 1.0, label + ": scales outside tolerance");
    require(shBands.ratio <= 1.0, label + ": SH outside tolerance");
    require(opacityMismatches == 0, label + ": " + std::to_string(opacityMismatches) + " opacities differ");
}

bool sameVec3(const glm::vec3 & a, const glm::vec3 & b)
{
    return sameBits(a.x, b.x) && sameBits(a.y, b.y) && sameBits(a.z, b.z);
}

// Every field of the bounds, bit for bit.
bool sameBounds(const SceneBounds & a, const SceneBounds & b)
{
    return a.valid == b.valid && sameVec3(a.min, b.min) && sameVec3(a.max, b.max)
           && sameVec3(a.focusCenter, b.focusCenter) && sameBits(a.focusRadius, b.focusRadius);
}

// A world box around the middle of a layer: its half extents are 0.67 of the layer's, a box
// that covers about 0.67^3 = 30 percent of the volume of a layer spread through its bounds.
void middleBox(const ProjectLayer & layer, glm::vec3 & center, glm::vec3 & halfExtents)
{
    const SceneBounds & local = layer.cloud->bounds();
    require(local.valid, "the layer has no bounds to select in");
    center = glm::vec3(layer.transform.matrix() * glm::vec4(local.center(), 1.0f));
    halfExtents = 0.5f * local.extent() * 0.67f;
}

void selectMiddle(StudioState & state, Renderer & renderer)
{
    glm::vec3 center(0.0f);
    glm::vec3 halfExtents(0.0f);
    middleBox(state.project.layers[0], center, halfExtents);
    selectBox(state, renderer, SelectOp::Replace, center, halfExtents);
}

TransformParams editTransform()
{
    TransformParams params;
    params.translation = glm::vec3(0.4f, -0.2f, 0.3f);
    params.rotationDegrees = glm::vec3(10.0f, 20.0f, 30.0f);
    params.scale = 1.1f;
    params.pivot = PivotMode::SelectionCentroid;
    return params;
}

// Layer 0 and the renderer's scene, read back after a rebuild and a sync.
struct EditSnapshot
{
    LayerCopy layer;
    SceneCopy scene;
};

EditSnapshot settle(StudioState & state, Renderer & renderer)
{
    rebuildScene(state, renderer);
    ProjectLayer & layer = state.project.layers[0];
    syncLayerToHost(state, layer);
    EditSnapshot snapshot;
    snapshot.layer = copyLayer(*layer.cloud);
    snapshot.scene = readScene(renderer);
    require(snapshot.scene.count == layer.cloud->count() + state.project.layers[1].cloud->count(),
            "the renderer's scene is not the merge of the two layers");
    return snapshot;
}

void requireSnapshot(const EditSnapshot & got, const EditSnapshot & want, const std::string & label)
{
    const std::string difference = differenceOf(got.layer, want.layer);
    require(difference.empty(), label + ": layer 0 differs in " + difference);
    require(sameScene(got.scene, want.scene), label + ": the renderer's scene differs");
}

// A selection on a locked or hidden layer throws, naming the reason, and changes nothing.
void requireEditRefused(StudioState & state, Renderer & renderer, const std::string & word)
{
    // Hiding a layer is a recorded command, so the baseline is taken here, not assumed to be 0.
    const size_t historyBefore = state.edit.history.size();
    const uint32_t selectedBefore = selectedCountOfActiveLayer(state);
    bool threw = false;
    std::string message;
    try
    {
        selectBox(state, renderer, SelectOp::Replace, glm::vec3(0.0f), glm::vec3(10.0f));
    }
    catch (const std::exception & e)
    {
        threw = true;
        message = e.what();
    }
    require(threw, "selectBox on a " + word + " layer did not throw");
    require(contains(message, word), "the refusal does not name the layer as " + word + ": " + message);
    require(selectedCountOfActiveLayer(state) == selectedBefore && state.edit.history.size() == historyBefore,
            "a refused selection changed the state");
}

// --- Layer operations and save/reopen (checks E9 and E10) -------------------

// One layer's metadata, as the project file and the Layers panel hold it.
struct LayerMeta
{
    std::string id;
    std::string name;
    bool visible = true;
    bool locked = false;
    LayerTransform transform;
    size_t count = 0;
};

// What an undo must restore: the layer list, the selected layer and the selection
// count of the active layer.
struct ProjectMeta
{
    std::vector<LayerMeta> layers;
    int selectedLayer = -1;
    uint32_t selectedCount = 0;
};

// The project and the renderer's scene (after a rebuild) at one point of a history.
struct EditCapture
{
    ProjectMeta meta;
    SceneCopy scene;
};

ProjectMeta metaOf(const StudioState & state)
{
    ProjectMeta meta;
    for (const ProjectLayer & layer : state.project.layers)
    {
        LayerMeta entry;
        entry.id = layer.id;
        entry.name = layer.name;
        entry.visible = layer.visible;
        entry.locked = layer.locked;
        entry.transform = layer.transform;
        entry.count = layer.cloud ? layer.cloud->count() : size_t{0};
        meta.layers.push_back(entry);
    }
    meta.selectedLayer = state.selectedLayer;
    meta.selectedCount = selectedCountOfActiveLayer(state);
    return meta;
}

// Names the first difference between two layer lists, or "" when they match.
std::string layersDifference(const std::vector<LayerMeta> & a, const std::vector<LayerMeta> & b)
{
    if (a.size() != b.size())
    {
        return "the layer count " + std::to_string(a.size()) + " != " + std::to_string(b.size());
    }
    for (size_t i = 0; i < a.size(); ++i)
    {
        const std::string where = "layer " + std::to_string(i) + ": ";
        if (a[i].id != b[i].id)
        {
            return where + "id";
        }
        if (a[i].name != b[i].name)
        {
            return where + "name";
        }
        if (a[i].visible != b[i].visible || a[i].locked != b[i].locked)
        {
            return where + "flags";
        }
        if (!sameTransform(a[i].transform, b[i].transform))
        {
            return where + "transform";
        }
        if (a[i].count != b[i].count)
        {
            return where + "count " + std::to_string(a[i].count) + " != " + std::to_string(b[i].count);
        }
    }
    return "";
}

// Names the first difference between two project metas, or "" when they match.
std::string metaDifference(const ProjectMeta & a, const ProjectMeta & b)
{
    const std::string layersDiff = layersDifference(a.layers, b.layers);
    if (!layersDiff.empty())
    {
        return layersDiff;
    }
    if (a.selectedLayer != b.selectedLayer)
    {
        return "the selected layer " + std::to_string(a.selectedLayer) + " != " + std::to_string(b.selectedLayer);
    }
    if (a.selectedCount != b.selectedCount)
    {
        return "the selection count " + std::to_string(a.selectedCount) + " != " + std::to_string(b.selectedCount);
    }
    return "";
}

// The project and the renderer's scene as they are now, after a rebuild.
EditCapture captureNow(StudioState & state, Renderer & renderer)
{
    rebuildScene(state, renderer);
    EditCapture capture;
    capture.meta = metaOf(state);
    capture.scene = readScene(renderer);
    return capture;
}

// Checks the project against `want`, then rebuilds the scene and checks the renderer's
// arrays against it, bit for bit. Throws on the first difference.
void requireCapture(StudioState & state, Renderer & renderer, const EditCapture & want, const std::string & label)
{
    const std::string project = metaDifference(metaOf(state), want.meta);
    require(project.empty(), label + ": the project differs in " + project);
    rebuildScene(state, renderer);
    require(sameScene(readScene(renderer), want.scene), label + ": the renderer's scene differs");
}

// The index of the layer `id` in the project, or -1.
int indexOfId(const StudioState & state, const std::string & id)
{
    for (size_t i = 0; i < state.project.layers.size(); ++i)
    {
        if (state.project.layers[i].id == id)
        {
            return static_cast<int>(i);
        }
    }
    return -1;
}

// A layer's host streams taken from its GPU copy. The layer's edit state is left alone, so
// a stale layer stays stale: only the host vectors are refreshed.
LayerCopy gpuCopyOf(ProjectLayer & layer)
{
    layer.cloud->downloadFromGPU();
    return copyLayer(*layer.cloud);
}

// Everything the edit checks do. Throws on the first failure. Each state is reset
// before it goes out of scope, so its masks die before the renderer does.
void runEditChecks(Renderer & renderer, const std::string & outDir)
{
    Device & device = renderer.sabercatDevice();

    // E1. The GPU merge (rebuildScene) against the host merge of the same layers, and the
    //     renderer's bounds against the host merged cloud's bounds, exactly.
    StudioState s1;
    addSyntheticLayers(s1, device);
    rebuildScene(s1, renderer);
    require(!s1.edit.gpuMergeFailed, "E1: the GPU layer merge failed");
    require(renderer.gaussianCount() == 5000,
            "E1: renderer count " + std::to_string(renderer.gaussianCount()) + " != 5000");
    require(s1.edit.boundsExact, "E1: the rebuilt scene's bounds are not the exact ones");
    const MergeInputs layers = visibleLayers(s1);
    const std::shared_ptr<GaussianCloud> merged = mergeLayers(device, layers);
    require(merged->count() == 5000, "E1: host merge count " + std::to_string(merged->count()));
    const SceneCopy gpuScene = readScene(renderer);
    const SceneCopy hostScene = copyCloudScene(*merged);
    requireSceneNear(gpuScene, hostScene, layers, "E1 GPU merge vs mergeLayers");
    require(sameBounds(renderer.sceneBounds(), merged->bounds()),
            "E1: the renderer's bounds differ from the host merge's bounds");
    std::cout << "  bounds     exact: renderer == host merge in every field" << std::endl;

    // E2. The host fallback is bit-exact to mergeLayers, and within the GPU tolerance of
    //     the GPU merge; the flag cleared, the GPU path comes back.
    s1.edit.gpuMergeFailed = true;
    s1.sceneDirty = true;
    rebuildScene(s1, renderer);
    const SceneCopy fallbackScene = readScene(renderer);
    require(!s1.edit.boundsExact, "E2: the host fallback claims exact bounds");
    require(sameScene(fallbackScene, hostScene), "E2: the host fallback is not bit-exact to mergeLayers");
    requireSceneNear(fallbackScene, gpuScene, layers, "E2 host fallback vs GPU merge");
    s1.edit.gpuMergeFailed = false;
    s1.sceneDirty = true;
    rebuildScene(s1, renderer);
    require(!s1.edit.gpuMergeFailed && s1.edit.boundsExact, "E2: the GPU merge did not resume");
    require(renderer.gaussianCount() == 5000, "E2: count after the GPU rebuild");
    std::cout << "E2 host fallback: bit-exact to mergeLayers, GPU path resumed" << std::endl;

    // E3. An edit sequence on layer 0 (select, transform, duplicate, delete), undone and
    //     redone; every state is bit-exact in the layer and in the renderer's scene.
    ProjectLayer & layer = s1.project.layers[0];
    const uint32_t n0 = static_cast<uint32_t>(layer.cloud->count());
    std::vector<EditSnapshot> states;
    states.push_back(settle(s1, renderer));
    selectMiddle(s1, renderer);
    const uint32_t picked = selectedCountOfActiveLayer(s1);
    require(picked >= 100 && picked + 100 <= n0, "E3: the box selected " + std::to_string(picked) + " of "
                                                     + std::to_string(n0) + " Gaussians (want 100 to n-100)");
    states.push_back(settle(s1, renderer));
    commitTransform(s1, renderer, editTransform());
    states.push_back(settle(s1, renderer));
    std::string whyNot;
    require(duplicateSelection(s1, renderer, whyNot), "E3: duplicate refused: " + whyNot);
    require(layer.cloud->count() == n0 + picked, "E3: duplicate count " + std::to_string(layer.cloud->count()));
    require(selectedCountOfActiveLayer(s1) == picked, "E3: the copies are not the selection");
    states.push_back(settle(s1, renderer));
    require(deleteSelection(s1, renderer, whyNot), "E3: delete refused: " + whyNot);
    require(layer.cloud->count() == n0, "E3: delete count " + std::to_string(layer.cloud->count()));
    states.push_back(settle(s1, renderer));
    require(s1.edit.history.size() == 4 && s1.edit.history.cursor() == 4,
            "E3: history holds " + std::to_string(s1.edit.history.size()) + " commands at cursor "
                + std::to_string(s1.edit.history.cursor()) + ", want 4 at 4");
    requireSnapshot(states[4], states[2], "E3 delete of the copies vs post-transform");

    // Undo four times: states[3], [2], [1], [0].
    const size_t undoTo[4] = {3, 2, 1, 0};
    for (int step = 0; step < 4; ++step)
    {
        require(undo(s1, renderer), "E3: undo " + std::to_string(step + 1) + " refused");
        requireSnapshot(settle(s1, renderer), states[undoTo[step]], "E3 undo " + std::to_string(step + 1));
    }
    require(!s1.edit.history.canUndo() && s1.edit.history.canRedo(), "E3: history flags after four undos");
    require(!undo(s1, renderer), "E3: undo past the start was accepted");

    // Redo four times: states[1], [2], [3], [4].
    for (size_t step = 1; step <= 4; ++step)
    {
        require(redo(s1, renderer), "E3: redo " + std::to_string(step) + " refused");
        requireSnapshot(settle(s1, renderer), states[step], "E3 redo " + std::to_string(step));
    }
    require(!s1.edit.history.canRedo() && s1.edit.history.canUndo(), "E3: history flags after four redos");
    require(!redo(s1, renderer), "E3: redo past the end was accepted");
    std::cout << "E3 edit sequence: select " << picked << " of " << n0 << ", transform, duplicate " << picked
              << ", delete; undo x4 and redo x4 bit-exact (layer 0 and renderer scene)" << std::endl;

    // E7. Exports after the edits: the export syncs the GPU copies, and the PLY reads back as
    //     the host merge of the synced layers.
    const std::string plyPath = outDir + "/edit_selftest.ply";
    exportPly(s1, device, plyPath);
    require(fileBytes(plyPath) > 0, "E7: the PLY export is empty");
    const MergeInputs exported = visibleLayers(s1);
    const std::shared_ptr<GaussianCloud> mergedAfterEdits = mergeLayers(device, exported);
    const std::shared_ptr<GaussianCloud> fromPly = sabercat::loadScene(device, plyPath);
    require(fromPly->count() == mergedAfterEdits->count(), "E7: PLY count " + std::to_string(fromPly->count())
                                                               + " != merged "
                                                               + std::to_string(mergedAfterEdits->count()));
    requirePlyMatchesMerged(*mergedAfterEdits, *fromPly);
    rebuildScene(s1, renderer);
    require(renderer.gaussianCount() == fromPly->count(), "E7: renderer count after the export "
                                                              + std::to_string(renderer.gaussianCount()));
    std::cout << "E7 export after edits: " << fromPly->count() << " Gaussians, matches the host merge" << std::endl;
    s1.edit.reset();

    // E4. A cancelled preview restores the layer bit for bit; a preview then a commit equals
    //     a direct commit on an identical state.
    TransformParams other;
    other.translation = glm::vec3(1.0f, 0.0f, 0.0f);
    other.rotationDegrees = glm::vec3(0.0f, 45.0f, 0.0f);
    other.scale = 1.3f;
    const TransformParams commit = editTransform();

    StudioState previewed;
    addSyntheticLayers(previewed, device);
    selectMiddle(previewed, renderer);
    ProjectLayer & previewedLayer = previewed.project.layers[0];
    syncLayerToHost(previewed, previewedLayer);
    const LayerCopy before = copyLayer(*previewedLayer.cloud);
    previewTransform(previewed, renderer, other);
    previewTransform(previewed, renderer, commit);
    require(previewed.edit.preview.active, "E4: the preview is not active");
    cancelTransformPreview(previewed, renderer);
    require(!previewed.edit.preview.active, "E4: the cancel left the preview active");
    syncLayerToHost(previewed, previewedLayer);
    const std::string afterCancel = differenceOf(copyLayer(*previewedLayer.cloud), before);
    require(afterCancel.empty(), "E4 cancel after two previews differs in " + afterCancel);
    // Preview then commit, against a direct commit on an identical state.
    previewTransform(previewed, renderer, commit);
    commitTransform(previewed, renderer, commit);
    syncLayerToHost(previewed, previewedLayer);

    StudioState direct;
    addSyntheticLayers(direct, device);
    selectMiddle(direct, renderer);
    commitTransform(direct, renderer, commit);
    ProjectLayer & directLayer = direct.project.layers[0];
    syncLayerToHost(direct, directLayer);
    const std::string difference = differenceOf(copyLayer(*previewedLayer.cloud), copyLayer(*directLayer.cloud));
    require(difference.empty(), "E4: preview then commit differs from a direct commit in " + difference);
    std::cout << "E4 preview: cancel restores the layer bit-exact; preview then commit == direct commit" << std::endl;
    previewed.edit.reset();
    direct.edit.reset();

    // E5. A 1 MiB budget evicts as 40 transforms of the whole layer push past it.
    StudioState budget;
    addSyntheticLayers(budget, device);
    selectAllGaussians(budget, renderer);
    budget.edit.history.setBudget(std::size_t{1} << 20);
    uint64_t evicted = 0;
    std::size_t kept = 0;
    int warnings = 0;
    {
        CerrCapture captured;
        for (int i = 0; i < 40; ++i)
        {
            TransformParams step;
            step.translation = glm::vec3(0.01f * static_cast<float>(i + 1), 0.0f, 0.0f);
            step.rotationDegrees = glm::vec3(0.0f, 0.0f, 2.0f);
            commitTransform(budget, renderer, step);
        }
        evicted = budget.edit.history.evictedCount();
        kept = budget.edit.history.size();
        warnings = captured.count("edit history");
    }
    require(evicted > 0, "E5: the 1 MiB budget evicted nothing");
    require(kept >= 1, "E5: the history is empty after the evictions");
    require(warnings > 0, "E5: no edit history warning was logged");
    budget.edit.history.setBudget(std::size_t{1} << 30);
    std::cout << "E5 history budget 1 MiB: 40 transforms of " << budget.project.layers[0].cloud->count()
              << " Gaussians, " << evicted << " evicted, " << kept << " kept, " << warnings << " warnings" << std::endl;
    budget.edit.reset();

    // E6. Edit gates: a locked or hidden active layer refuses edits, and deleting every
    //     Gaussian is refused.
    StudioState gates;
    addSyntheticLayers(gates, device);
    require(setLayerLocked(gates, 0, true), "E6: could not lock layer 0");
    require(editableLayer(gates) == nullptr && contains(whyNotEditable(gates), "locked"),
            std::string("E6: a locked layer reads as editable: ") + whyNotEditable(gates));
    requireEditRefused(gates, renderer, "locked");
    require(setLayerLocked(gates, 0, false), "E6: could not unlock layer 0");
    require(setLayerVisible(gates, 0, false), "E6: could not hide layer 0");
    require(editableLayer(gates) == nullptr && contains(whyNotEditable(gates), "hidden"),
            std::string("E6: a hidden layer reads as editable: ") + whyNotEditable(gates));
    requireEditRefused(gates, renderer, "hidden");
    require(setLayerVisible(gates, 0, true), "E6: could not show layer 0");
    require(editableLayer(gates) != nullptr, "E6: layer 0 is not editable once unlocked and shown");
    selectAllGaussians(gates, renderer);
    require(!deleteSelection(gates, renderer, whyNot), "E6: deleting every Gaussian was accepted");
    require(contains(whyNot, "remove the layer"), "E6: the refusal does not say to remove the layer: " + whyNot);
    require(gates.project.layers.size() == 2 && gates.project.layers[0].cloud->count() == 3000,
            "E6: a refused delete changed the layers");
    std::cout << "E6 edit gates: locked and hidden refuse edits; deleting every Gaussian is refused" << std::endl;
    gates.edit.reset();

    // E8. Saving and exporting cancel an active transform preview, and a selection change
    //     does not mark the project dirty.
    StudioState sel;
    addSyntheticLayers(sel, device);
    selectMiddle(sel, renderer);
    syncLayerToHost(sel, sel.project.layers[0]);
    const LayerCopy selBefore = copyLayer(*sel.project.layers[0].cloud);
    sel.dirty = false;
    glm::vec3 selCenter(0.0f);
    glm::vec3 selHalf(0.0f);
    middleBox(sel.project.layers[0], selCenter, selHalf);
    selectBox(sel, renderer, SelectOp::Replace, selCenter + glm::vec3(1.0f, 0.0f, 0.0f), selHalf);
    require(!sel.dirty, "E8: a selectBox marked the project dirty");
    selectNone(sel, renderer);
    require(!sel.dirty, "E8: selectNone marked the project dirty");
    const std::string selDifference = differenceOf(copyLayer(*sel.project.layers[0].cloud), selBefore);
    require(selDifference.empty(), "E8: selection changed layer 0 in " + selDifference);
    sel.edit.reset();

    StudioState pv;
    addSyntheticLayers(pv, device);
    selectMiddle(pv, renderer);
    ProjectLayer & pvLayer = pv.project.layers[0];
    syncLayerToHost(pv, pvLayer);
    const LayerCopy preview0 = copyLayer(*pvLayer.cloud);
    const LayerCopy preview1 = copyLayer(*pv.project.layers[1].cloud);

    // Save during a preview: the file holds the pre-preview values, and the preview ends.
    previewTransform(pv, renderer, editTransform());
    require(pv.edit.preview.active, "E8: the preview did not start");
    const std::string savePath = outDir + "/preview_save.sabercat";
    saveProject(pv, savePath);
    require(!pv.edit.preview.active, "E8: saveProject left the preview active");
    const Project savedProject = ProjectIO::load(device, savePath);
    require(savedProject.layers.size() == 2, "E8: the saved project has no two layers");
    const std::string saved0 = differenceOf(copyLayer(*savedProject.layers[0].cloud), preview0);
    require(saved0.empty(), "E8: the saved layer 0 differs from the pre-preview copy in " + saved0);
    const std::string saved1 = differenceOf(copyLayer(*savedProject.layers[1].cloud), preview1);
    require(saved1.empty(), "E8: the saved layer 1 changed in " + saved1);

    // Export PLY during a preview: the preview ends, and the PLY is the host merge of the
    // restored layers.
    previewTransform(pv, renderer, editTransform());
    require(pv.edit.preview.active, "E8: the preview did not restart for the PLY export");
    const std::string previewPlyPath = outDir + "/preview_export.ply";
    exportPly(pv, device, previewPlyPath);
    require(!pv.edit.preview.active, "E8: exportPly left the preview active");
    const std::string ply0 = differenceOf(copyLayer(*pvLayer.cloud), preview0);
    require(ply0.empty(), "E8: layer 0 after exportPly differs from the pre-preview copy in " + ply0);
    const MergeInputs plyLayers = visibleLayers(pv);
    const std::shared_ptr<GaussianCloud> plyMerged = mergeLayers(device, plyLayers);
    const std::shared_ptr<GaussianCloud> plyLoaded = sabercat::loadScene(device, previewPlyPath);
    require(plyLoaded->count() == plyMerged->count(), "E8: PLY count differs from the host merge");
    requirePlyMatchesMerged(*plyMerged, *plyLoaded);

    // Export GLB and SPZ during a preview: each ends the preview.
    previewTransform(pv, renderer, editTransform());
    require(pv.edit.preview.active, "E8: the preview did not restart for the GLB export");
    const std::string glbPath = outDir + "/preview_export.glb";
    exportGlb(pv, glbPath);
    require(!pv.edit.preview.active, "E8: exportGlb left the preview active");
    require(fileBytes(glbPath) > 0, "E8: the GLB export is empty");

    previewTransform(pv, renderer, editTransform());
    require(pv.edit.preview.active, "E8: the preview did not restart for the SPZ export");
    const std::string spzPath = outDir + "/preview_export.spz";
    exportSpz(pv, device, spzPath);
    require(!pv.edit.preview.active, "E8: exportSpz left the preview active");
    require(fileBytes(spzPath) > 0, "E8: the SPZ export is empty");

    std::cout << "E8 save/export cancel the transform preview, selection does not mark dirty" << std::endl;
    pv.edit.reset();

    // E9. Layer operations on real clouds, all recorded: a rename, a hide and a show, an
    //     Inspector drag of layer 1, a selection and a transform of layer 0, a move, the
    //     removal of the edited layer and the add of a file. Each is one command. Undo and
    //     redo are bit-exact in the project and in the renderer's scene, the removed layer
    //     comes back with its GPU edits, and jumpToHistory reaches both ends in one call.
    StudioState ops;
    addSyntheticLayers(ops, device);
    const std::string edited = ops.project.layers[0].id;
    const EditCapture start = captureNow(ops, renderer);

    // A lock toggle is not recorded; a rename of a locked layer is refused and records nothing.
    require(setLayerLocked(ops, 0, true), "E9: could not lock layer 0");
    require(ops.edit.history.size() == 0, "E9: locking a layer was recorded");
    require(!renameLayer(ops, 0, "refused"), "E9: a rename of a locked layer was accepted");
    require(ops.project.layers[0].name == start.meta.layers[0].name && ops.edit.history.size() == 0,
            "E9: a refused rename changed the layer or the history");
    require(setLayerLocked(ops, 0, false), "E9: could not unlock layer 0");
    require(ops.edit.history.size() == 0, "E9: unlocking a layer was recorded");
    requireCapture(ops, renderer, start, "E9 after the lock checks");

    // captures[k] is the state after the first k operations; sizes[k] the history size then.
    std::vector<EditCapture> captures = {start};
    std::vector<size_t> sizes = {0};
    auto checkpoint = [&](const std::string & what) {
        require(ops.edit.history.size() == sizes.back() + 1,
                "E9 " + what + ": the history holds " + std::to_string(ops.edit.history.size())
                    + " commands, want " + std::to_string(sizes.back() + 1));
        captures.push_back(captureNow(ops, renderer));
        sizes.push_back(ops.edit.history.size());
    };

    require(renameLayer(ops, 1, "synthetic B, renamed"), "E9: the rename of layer 1 was refused");
    checkpoint("rename layer 1");

    require(setLayerVisible(ops, 0, false), "E9: could not hide layer 0");
    checkpoint("hide layer 0");
    require(setLayerVisible(ops, 0, true), "E9: could not show layer 0");
    checkpoint("show layer 0");

    // An Inspector drag of layer 1: every frame is an unrecorded setLayerTransform, and the
    // finished drag is one command.
    const LayerTransform dragStart = ops.project.layers[1].transform;
    const size_t beforeDrag = ops.edit.history.size();
    for (int frame = 1; frame <= 3; ++frame)
    {
        const float f = static_cast<float>(frame);
        LayerTransform moving;
        moving.rotation = glm::normalize(glm::angleAxis(glm::radians(15.0f * f), glm::vec3(0.0f, 1.0f, 0.0f))
                                         * dragStart.rotation);
        moving.translation = dragStart.translation + glm::vec3(0.3f * f, -0.1f * f, 0.2f * f);
        moving.scale = dragStart.scale * (1.0f + 0.1f * f);
        require(setLayerTransform(ops, 1, moving), "E9: drag frame " + std::to_string(frame) + " was refused");
    }
    require(ops.edit.history.size() == beforeDrag, "E9: a drag frame was recorded");
    require(commitLayerTransformEdit(ops, 1, dragStart), "E9: the finished drag was not recorded");
    require(!commitLayerTransformEdit(ops, 1, ops.project.layers[1].transform),
            "E9: a commit of an unchanged transform was recorded");
    checkpoint("drag layer 1");

    // A selection and a transform of layer 0, the layer that is then moved and removed.
    selectMiddle(ops, renderer);
    checkpoint("select on layer 0");
    require(selectedCountOfActiveLayer(ops) >= 100, "E9: the selection on layer 0 is too small");
    commitTransform(ops, renderer, editTransform());
    checkpoint("transform layer 0");

    require(moveSelectedLayer(ops, 1), "E9: the move of layer 0 was refused");
    checkpoint("move layer 0 down");
    const int editedIndex = indexOfId(ops, edited);
    require(editedIndex == 1, "E9: the edited layer is at index " + std::to_string(editedIndex) + " after the move");

    // Remove the edited layer: it holds GPU edits (its host copy is stale), and its selection.
    ops.selectedLayer = editedIndex;
    require(ops.edit.layers.at(edited).hostStale, "E9: the layer to remove has no GPU edits");
    const LayerCopy removedGpu = gpuCopyOf(ops.project.layers[static_cast<size_t>(editedIndex)]);
    const uint32_t removedSelected = selectedCountOfActiveLayer(ops);
    require(removeSelectedLayer(ops), "E9: the removal of the edited layer was refused");
    require(indexOfId(ops, edited) < 0, "E9: the removed layer is still in the project");
    checkpoint("remove the edited layer");
    const size_t removeOperation = sizes.size() - 1;

    // Add a file: a small synthetic cloud written as PLY, then added through the recorded path.
    const std::string extraPath = outDir + "/e9_extra.ply";
    const size_t extraCount = 500;
    const std::shared_ptr<GaussianCloud> extra = makeSyntheticCloud(device, extraCount, 1, false, 1003);
    PLYWriter::write(*extra, extraPath);
    addLayerFromFile(ops, device, extraPath);
    require(ops.project.layers.size() == 2 && ops.selectedLayer == 1,
            "E9: the added layer is not the second layer, selected");
    require(ops.project.layers[1].cloud->count() == extraCount,
            "E9: the added layer has " + std::to_string(ops.project.layers[1].cloud->count()) + " Gaussians, not "
                + std::to_string(extraCount));
    checkpoint("add e9_extra.ply");

    // Undo everything, one command at a time. Operation k's undo returns to captures[k - 1].
    // The removal's undo brings the layer back with its GPU edits: a host sync reads them back.
    const size_t opCount = sizes.size() - 1;
    require(ops.edit.history.size() == opCount, "E9: the history does not hold one command per operation");
    size_t undone = 0;
    while (ops.edit.history.canUndo())
    {
        const size_t op = ops.edit.history.cursor(); // the operation undone now
        require(undo(ops, renderer), "E9: undo of operation " + std::to_string(op) + " was refused");
        ++undone;
        requireCapture(ops, renderer, captures[op - 1], "E9 undo of operation " + std::to_string(op));
        if (op == removeOperation)
        {
            const int back = indexOfId(ops, edited);
            require(back >= 0, "E9: the undone removal did not bring the layer back");
            ProjectLayer & restored = ops.project.layers[static_cast<size_t>(back)];
            syncLayerToHost(ops, restored);
            const std::string restoredDiff = differenceOf(copyLayer(*restored.cloud), removedGpu);
            require(restoredDiff.empty(), "E9: the restored layer differs from its GPU copy in " + restoredDiff);
            require(selectedCountOfActiveLayer(ops) == removedSelected, "E9: the restored layer's selection differs");
        }
    }
    require(undone == opCount && !ops.edit.history.canUndo(),
            "E9: undo stopped after " + std::to_string(undone) + " of " + std::to_string(opCount));

    // Redo everything. Operation k's redo returns to captures[k].
    size_t redone = 0;
    while (ops.edit.history.canRedo())
    {
        const size_t op = ops.edit.history.cursor() + 1; // the operation redone now
        require(redo(ops, renderer), "E9: redo of operation " + std::to_string(op) + " was refused");
        ++redone;
        requireCapture(ops, renderer, captures[op], "E9 redo of operation " + std::to_string(op));
    }
    require(redone == opCount && !ops.edit.history.canRedo(),
            "E9: redo stopped after " + std::to_string(redone) + " of " + std::to_string(opCount));

    // jumpToHistory reaches any point in one call: the start, the middle and the end.
    require(jumpToHistory(ops, renderer, 0) && ops.edit.history.cursor() == 0, "E9: jumpToHistory(0) did not reach the start");
    requireCapture(ops, renderer, captures[0], "E9 jumpToHistory(0)");
    require(jumpToHistory(ops, renderer, 4) && ops.edit.history.cursor() == 4, "E9: jumpToHistory(4) did not reach 4");
    requireCapture(ops, renderer, captures[4], "E9 jumpToHistory(4)");
    require(jumpToHistory(ops, renderer, opCount) && ops.edit.history.cursor() == opCount,
            "E9: jumpToHistory(size) did not reach the end");
    requireCapture(ops, renderer, captures[opCount], "E9 jumpToHistory(size)");

    std::cout << "E9 layer operations: " << opCount << " ops, undo/redo bit-exact (project and scene); "
              << "the lock toggle and a locked rename are not recorded; the removed layer came back with its GPU edits; "
              << "jumpToHistory reaches 0, 4 and the end" << std::endl;
    ops.edit.reset();

    // E10. Save and reopen after edits. The file holds the edited layers; a reopen restores them
    //      bit for bit (the layers, the host streams, the scene and the bounds after a rebuild),
    //      the GPU copies reload exactly, and the reopened state edits, undoes and redoes the
    //      same way as a second reopen of the same file.
    StudioState saved;
    addSyntheticLayers(saved, device);
    selectMiddle(saved, renderer);
    commitTransform(saved, renderer, editTransform());
    require(duplicateSelection(saved, renderer, whyNot), "E10: duplicate refused: " + whyNot);
    require(renameLayer(saved, 1, "synthetic B, renamed"), "E10: rename refused");
    rebuildScene(saved, renderer);
    refreshExactBounds(saved, renderer);
    syncAllToHost(saved);
    std::vector<LayerCopy> preSave;
    for (ProjectLayer & entry : saved.project.layers)
    {
        preSave.push_back(copyLayer(*entry.cloud));
    }
    const ProjectMeta preMeta = metaOf(saved);
    const SceneCopy preScene = readScene(renderer);
    const SceneBounds preBounds = renderer.sceneBounds();
    require(preBounds.valid, "E10: the pre-save scene has no bounds");

    const std::string reopenPath = outDir + "/e10.sabercat";
    saveProject(saved, reopenPath);
    require(!saved.dirty, "E10: saveProject left the project dirty");

    StudioState reopened;
    openProject(reopened, renderer, reopenPath);
    require(reopened.edit.history.size() == 0 && !reopened.dirty,
            "E10: a reopened project has history or is dirty");
    const std::string layersDiff = layersDifference(metaOf(reopened).layers, preMeta.layers);
    require(layersDiff.empty(), "E10: the reopened layers differ in " + layersDiff);
    for (size_t i = 0; i < preSave.size(); ++i)
    {
        const std::string hostDiff = differenceOf(copyLayer(*reopened.project.layers[i].cloud), preSave[i]);
        require(hostDiff.empty(), "E10: reopened layer " + std::to_string(i) + " differs in " + hostDiff);
    }
    rebuildScene(reopened, renderer);
    require(sameScene(readScene(renderer), preScene), "E10: the reopened scene differs from the pre-save scene");
    require(sameBounds(renderer.sceneBounds(), preBounds), "E10: the reopened bounds differ from the pre-save bounds");

    // The GPU copies after the reload download to the pre-save host copies, bit for bit.
    ensureLayerGpu(reopened, renderer, reopened.project.layers[0]);
    for (size_t i = 0; i < preSave.size(); ++i)
    {
        const std::string gpuDiff = differenceOf(gpuCopyOf(reopened.project.layers[i]), preSave[i]);
        require(gpuDiff.empty(), "E10: the GPU copy of reopened layer " + std::to_string(i) + " differs in " + gpuDiff);
    }

    // Edit the reopened layer 0 (select the middle, transform it), then undo and redo the transform.
    ProjectLayer & reopenedZero = reopened.project.layers[0];
    glm::vec3 center(0.0f);
    glm::vec3 halfExtents(0.0f);
    middleBox(reopenedZero, center, halfExtents);
    selectBox(reopened, renderer, SelectOp::Replace, center, halfExtents);
    commitTransform(reopened, renderer, editTransform());
    require(reopened.edit.history.size() == 2 && reopened.edit.history.cursor() == 2,
            "E10: the history after the selection and the transform is not 2 at 2");
    syncLayerToHost(reopened, reopenedZero);
    const LayerCopy editedCopy = copyLayer(*reopenedZero.cloud);
    require(!differenceOf(editedCopy, preSave[0]).empty(), "E10: the edit left layer 0 unchanged");

    require(undo(reopened, renderer), "E10: undo of the transform was refused");
    require(reopened.edit.history.size() == 2 && reopened.edit.history.cursor() == 1,
            "E10: the history after the undo is not 2 at 1");
    syncLayerToHost(reopened, reopenedZero);
    const std::string afterUndo = differenceOf(copyLayer(*reopenedZero.cloud), preSave[0]);
    require(afterUndo.empty(), "E10: the undo is not the reopened layer 0, differs in " + afterUndo);

    require(redo(reopened, renderer), "E10: redo of the transform was refused");
    require(reopened.edit.history.cursor() == 2, "E10: the history after the redo is not at 2");
    syncLayerToHost(reopened, reopenedZero);
    const std::string afterRedo = differenceOf(copyLayer(*reopenedZero.cloud), editedCopy);
    require(afterRedo.empty(), "E10: the redo differs from the edit in " + afterRedo);

    // A second reopen of the same file, given the same selection and transform, lands on the same bits.
    StudioState twin;
    openProject(twin, renderer, reopenPath);
    rebuildScene(twin, renderer);
    selectBox(twin, renderer, SelectOp::Replace, center, halfExtents);
    commitTransform(twin, renderer, editTransform());
    syncLayerToHost(twin, twin.project.layers[0]);
    const std::string twinDiff = differenceOf(copyLayer(*twin.project.layers[0].cloud), editedCopy);
    require(twinDiff.empty(), "E10: a second reopen with the same edit differs in " + twinDiff);

    std::cout << "E10 save/reopen after edits: " << preSave.size() << " layers (layer 0 edited and duplicated to "
              << preSave[0].count << " Gaussians, layer 1 renamed); reopened layers, host streams, scene and bounds "
              << "bit-exact; GPU reload bit-exact; edit, undo, redo and a second reopen agree" << std::endl;
    twin.edit.reset();
    reopened.edit.reset();
    saved.edit.reset();
}

// Everything the checks do, in order. Throws on the first failure. The states
// are locals, so they (and their clouds) are gone before the renderer shuts down.
void runChecks(Renderer & renderer, const std::string & outDir, const std::vector<std::string> & scenes)
{
    fs::create_directories(fs::u8path(outDir));
    Device & device = renderer.sabercatDevice();

    // 1. Author a project: each scene a layer; every layer after the first is placed.
    //    With no scenes, the synthetic project instead.
    StudioState authored;
    if (scenes.empty())
    {
        addSyntheticLayers(authored, device);
    }
    for (size_t i = 0; i < scenes.size(); ++i)
    {
        addLayerFromFile(authored, device, scenes[i]);
        if (i > 0)
        {
            LayerTransform & t = authored.project.layers.back().transform;
            t.rotation = glm::normalize(glm::angleAxis(glm::radians(30.0f), glm::vec3(0.0f, 1.0f, 0.0f)));
            t.translation = glm::vec3(1.5f, 0.25f, -0.75f);
            t.scale = 0.75f;
        }
    }
    require(!authored.project.layers.empty(), "no layers loaded");
    authored.project.layers[0].locked = true; // exercise the flags in the round trip
    // Non-default app state, so the round trip has something to carry.
    authored.camera.distance = 7.5f;
    authored.camera.pivot = glm::vec3(0.5f, -0.25f, 2.0f);
    authored.camera.orientation = glm::normalize(glm::angleAxis(0.7f, glm::normalize(glm::vec3(0.3f, 1.0f, 0.2f))));
    authored.shDegree = 2;
    authored.debugViewIndex = 1;
    authored.relight.enabled = true;
    authored.relight.exposure = 1.25f;
    authored.relight.shadowMode = 2;
    authored.relight.environmentPreset = 1;
    authored.relight.progressiveShadows = false;

    // 2. Save, then read the file back and compare every layer bit for bit.
    const std::string projectPath = outDir + "/selftest.sabercat";
    auto start = Clock::now();
    saveProject(authored, projectPath);
    const double saveSeconds = secondsSince(start);
    std::cout << "project save: " << saveSeconds << " s, " << fileBytes(projectPath) << " bytes" << std::endl;

    start = Clock::now();
    const Project loaded = ProjectIO::load(device, projectPath);
    const double loadSeconds = secondsSince(start);
    std::cout << "project load: " << loadSeconds << " s" << std::endl;

    require(loaded.layers.size() == authored.project.layers.size(), "layer count after load");
    for (size_t i = 0; i < loaded.layers.size(); ++i)
    {
        const ProjectLayer & a = authored.project.layers[i];
        const ProjectLayer & b = loaded.layers[i];
        const std::string where = "layer " + std::to_string(i) + ": ";
        require(a.id == b.id, where + "id");
        require(a.name == b.name, where + "name");
        require(a.visible == b.visible && a.locked == b.locked, where + "flags");
        require(sameTransform(a.transform, b.transform), where + "transform");
        require(a.cloud && b.cloud, where + "missing cloud");
        const std::string difference = firstDifference(*a.cloud, *b.cloud);
        require(difference.empty(), where + "differs in " + difference);
    }
    const json appBefore = json::parse(authored.project.appStateJson);
    require(json::parse(loaded.appStateJson) == appBefore, "app state differs after load");

    // 3. Opening through the app path restores the viewer state.
    StudioState reopened;
    openProject(reopened, renderer, projectPath);
    const json appRestored = json::parse(buildAppStateJson(reopened));
    require(appRestored == appBefore,
            "restored app state differs: saved " + appBefore.dump() + " restored " + appRestored.dump());
    require(reopened.camera.distance == authored.camera.distance, "camera distance not restored");
    require(reopened.shDegree == 2 && reopened.debugViewIndex == 1, "SH degree or debug view not restored");
    require(reopened.relight.enabled && reopened.relight.exposure == 1.25f
                && reopened.relight.shadowMode == 2 && !reopened.relight.progressiveShadows,
            "relight settings not restored");
    std::cout << "project round trip: bit-exact, " << loaded.layers.size() << " layers" << std::endl;

    // 4. Export PLY, GLB and SPZ from the reopened project.
    const std::string previewPlyPath = outDir + "/selftest.ply";
    const std::string glbPath = outDir + "/selftest.glb";
    const std::string spzPath = outDir + "/selftest.spz";
    start = Clock::now();
    exportPly(reopened, device, previewPlyPath);
    const double plySeconds = secondsSince(start);
    start = Clock::now();
    exportGlb(reopened, glbPath);
    const double glbSeconds = secondsSince(start);
    start = Clock::now();
    exportSpz(reopened, device, spzPath);
    const double spzSeconds = secondsSince(start);
    std::cout << "export PLY: " << plySeconds << " s, " << fileBytes(previewPlyPath) << " bytes" << std::endl;
    std::cout << "export GLB: " << glbSeconds << " s, " << fileBytes(glbPath) << " bytes" << std::endl;
    std::cout << "export SPZ: " << spzSeconds << " s, " << fileBytes(spzPath) << " bytes" << std::endl;

    // 5. Reload the exports. The reference is mergeLayers of the same layers.
    const auto visible = visibleLayers(reopened);
    const std::shared_ptr<GaussianCloud> merged = mergeLayers(device, visible);
    require(merged->count() > 0, "merged scene is empty");

    const std::shared_ptr<GaussianCloud> fromPly = sabercat::loadScene(device, previewPlyPath);
    require(fromPly->count() == merged->count(), "PLY count " + std::to_string(fromPly->count())
                                                     + " != merged " + std::to_string(merged->count()));
    if (scenes.empty())
    {
        requirePlyMatchesMerged(*merged, *fromPly);
    }

    const std::shared_ptr<GaussianCloud> fromGlb = sabercat::loadScene(device, glbPath);
    require(fromGlb->count() == merged->count(), "GLB count " + std::to_string(fromGlb->count())
                                                     + " != merged " + std::to_string(merged->count()));
    Worst worst;
    worst.add("positions", worstError(fromGlb->positions(), merged->positions(), 3));
    worst.add("scales", worstError(fromGlb->scales(), merged->scales(), 3));
    worst.add("rotations", worstError(fromGlb->rotations(), merged->rotations(), 4));
    worst.add("opacities", worstError(fromGlb->opacities(), merged->opacities()));
    worst.add("shCoefficients", worstError(fromGlb->shCoefficients(), merged->shCoefficients(), 3));
    require(worst.value <= 1e-5, "GLB streams differ from mergeLayers by " + std::to_string(worst.value)
                                     + " in " + worst.stream);
    std::cout << "GLB reload vs mergeLayers: worst error " << worst.value << " (" << worst.stream
              << "), " << merged->count() << " Gaussians" << std::endl;

    // SPZ stores positions as 24-bit fixed point with 12 fractional bits, so the
    // rounding error is at most 1/8192 (about 1.22e-4); the bound is 1.3e-4.
    const std::shared_ptr<GaussianCloud> fromSpz = SpzReader::read(device, spzPath);
    require(fromSpz->count() == merged->count(), "SPZ count " + std::to_string(fromSpz->count())
                                                     + " != merged " + std::to_string(merged->count()));
    const double spzPositionError = worstError(fromSpz->positions(), merged->positions(), 3);
    require(spzPositionError <= 1.3e-4, "SPZ positions differ from mergeLayers by "
                                            + std::to_string(spzPositionError));
    std::cout << "SPZ reload vs mergeLayers: worst position error " << spzPositionError << ", "
              << merged->count() << " Gaussians" << std::endl;

    // 6. One forward frame of the merged scene through the app's rebuild path.
    rebuildScene(reopened, renderer);
    require(renderer.gaussianCount() == merged->count(), "renderer count != merged count");
    std::cout << "scene rebuild (mergeLayers + setGaussians): " << reopened.lastRebuildMs << " ms" << std::endl;
    reopened.camera.frame(renderer.sceneBounds(), renderer.height(), static_cast<float>(renderer.height()));
    renderer.setCamera(reopened.camera.toCamera(renderer.width(), renderer.height()));
    renderer.setMaxSHDegree(static_cast<uint32_t>(reopened.shDegree));
    renderer.forward();
    std::vector<float> image = renderer.readImage();
    const size_t pixels = static_cast<size_t>(renderer.width()) * renderer.height();
    require(image.size() == pixels * 4, "frame size");
    size_t covered = 0;
    for (size_t p = 0; p < pixels; ++p)
    {
        for (size_t c = 0; c < 4; ++c)
        {
            require(std::isfinite(image[p * 4 + c]), "frame contains a non-finite value");
        }
        covered += image[p * 4 + 3] > 0.0f ? 1 : 0;
    }
    require(covered > 0, "the merged scene covers no pixels");
    std::cout << "forward frame: " << covered << " of " << pixels << " pixels covered" << std::endl;

    // 7. Hiding every layer clears the scene, so nothing is drawn: no
    // Gaussians, no image to read (readImage throws), and exporting is refused.
    for (ProjectLayer & layer : reopened.project.layers)
    {
        layer.visible = false;
    }
    reopened.sceneDirty = true;
    rebuildScene(reopened, renderer);
    require(renderer.gaussianCount() == 0 && !renderer.sceneBounds().valid,
            "hidden project still has a scene");
    renderer.forward();
    bool hasImage = true;
    try
    {
        renderer.readImage();
    }
    catch (const std::exception &)
    {
        hasImage = false;
    }
    require(!hasImage, "hidden project still produces an image");
    fs::remove(fs::u8path(outDir + "/unused.glb"));
    fs::remove(fs::u8path(outDir + "/unused.ply"));
    fs::remove(fs::u8path(outDir + "/unused.spz"));
    bool refused = false;
    try
    {
        exportGlb(reopened, outDir + "/unused.glb");
    }
    catch (const std::runtime_error &)
    {
        refused = true;
    }
    require(refused, "export with no visible layers was not refused");
    bool plyRefused = false;
    try
    {
        exportPly(reopened, device, outDir + "/unused.ply");
    }
    catch (const std::runtime_error &)
    {
        plyRefused = true;
    }
    require(plyRefused, "PLY export with no visible layers was not refused");
    bool spzRefused = false;
    try
    {
        exportSpz(reopened, device, outDir + "/unused.spz");
    }
    catch (const std::runtime_error &)
    {
        spzRefused = true;
    }
    require(spzRefused, "SPZ export with no visible layers was not refused");
    require(!fs::exists(fs::u8path(outDir + "/unused.ply")) && !fs::exists(fs::u8path(outDir + "/unused.glb"))
                && !fs::exists(fs::u8path(outDir + "/unused.spz")),
            "a refused export left a file behind");
    std::cout << "no visible layers: nothing rendered, PLY, GLB and SPZ export refused" << std::endl;
    runEditChecks(renderer, outDir);
}

} // namespace

int runSelfTest(int argc, char * argv[], bool enableValidation)
{
    if (argc < 3)
    {
        std::cerr << "usage: sabercat_studio.exe --selftest <outDir> [<scene>...]" << std::endl;
        return 1;
    }
    const std::string outDir = argv[2];
    // No scenes: the synthetic project (addSyntheticLayers).
    const std::vector<std::string> scenes(argv + 3, argv + argc);

    // Headless: no window extensions, no presentation requirement.
    BorrowedVulkanHost host;
    Renderer renderer(1280, 720);
    int code = 0;
    try
    {
        host.create({}, /*presentation=*/false, enableValidation);
        renderer.initializeBorrowed(host.context(), host.enabled, resolveShaderDir());
        runChecks(renderer, outDir, scenes);
        std::cout << "studio selftest OK" << std::endl;
    }
    catch (const std::exception & e)
    {
        std::cout << "studio selftest FAILED: " << e.what() << std::endl;
        code = 1;
    }
    renderer.shutdown();
    host.destroy();
    return code;
}

} // namespace sabercat::studio
