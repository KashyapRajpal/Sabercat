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

    // 4. Export PLY and GLB from the reopened project.
    const std::string plyPath = outDir + "/selftest.ply";
    const std::string glbPath = outDir + "/selftest.glb";
    start = Clock::now();
    exportPly(reopened, device, plyPath);
    const double plySeconds = secondsSince(start);
    start = Clock::now();
    exportGlb(reopened, glbPath);
    const double glbSeconds = secondsSince(start);
    std::cout << "export PLY: " << plySeconds << " s, " << fileBytes(plyPath) << " bytes" << std::endl;
    std::cout << "export GLB: " << glbSeconds << " s, " << fileBytes(glbPath) << " bytes" << std::endl;

    // 5. Reload both exports. The reference is mergeLayers of the same layers.
    const auto visible = visibleLayers(reopened);
    const std::shared_ptr<GaussianCloud> merged = mergeLayers(device, visible);
    require(merged->count() > 0, "merged scene is empty");

    const std::shared_ptr<GaussianCloud> fromPly = sabercat::loadScene(device, plyPath);
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
    require(!fs::exists(fs::u8path(outDir + "/unused.ply")) && !fs::exists(fs::u8path(outDir + "/unused.glb")),
            "a refused export left a file behind");
    std::cout << "no visible layers: nothing rendered, PLY and GLB export refused" << std::endl;
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
