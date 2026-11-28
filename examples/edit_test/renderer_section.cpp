// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Sections G, H and I of edit_test, on a Renderer.
//
//  G. Renderer::mergeLayersOnGpu: the merged count and SH degree, the bounds after
//     setSceneBounds (bit-equal to the host merge), the readback against the host
//     merge, in-place storage reuse and growth, forward parity with the host merge
//     (identity: bit-identical; transformed: mean difference), clearScene, and the
//     rendered frame ending at a merge (until the next forward()).
//  H. The ray-traced visibility refit: refit on a moved layer, a fresh build equal
//     to it, the refit cap of 16 in a row (with the refit count per frame), a build
//     on a count change.
//  I. Informational merge timing on the scene named by SABERCAT_EDIT_TEST_SCENE.
//
// The layers are built on the Renderer's own Device (Renderer::sabercatDevice),
// since a GaussianCloud can be merged only on the device it was created on.

#include "edit_test_common.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <memory>
#include <utility>
#include <string>
#include <vector>

#include "sabercat/camera.h"
#include "sabercat/device.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/layer_transform.h"
#include "sabercat/lighting.h"
#include "sabercat/renderer.h"
#include "sabercat/scene_loader.h"

#include "../common/shader_dir.h"

using namespace sabercat;
using namespace edit_test;

namespace edit_test
{
namespace
{

constexpr uint32_t kWidth = 128;
constexpr uint32_t kHeight = 96;
constexpr size_t kCountA = 100003;
constexpr size_t kCountA2 = 5003;
constexpr size_t kCountB = 4099;

// True when `fn` throws a std::exception.
template <typename Fn>
bool throwsException(Fn && fn)
{
    try
    {
        fn();
    }
    catch (const std::exception &)
    {
        return true;
    }
    return false;
}

// Camera 14 units back on +z, looking down -z, with a modest field of view.
Camera framingCamera()
{
    Camera c;
    c.width = kWidth;
    c.height = kHeight;
    c.fx = 130.0f;
    c.fy = 130.0f;
    c.cx = 0.5f * static_cast<float>(kWidth - 1);
    c.cy = 0.5f * static_cast<float>(kHeight - 1);
    c.cameraToWorld = glm::mat4(1.0f);
    c.cameraToWorld[3] = glm::vec4(0.0f, 0.0f, 14.0f, 1.0f);
    return c;
}

// The host arrays of a host-resident cloud, as setGaussians takes them.
GaussianHostInput hostInput(const GaussianCloud & cloud)
{
    GaussianHostInput in;
    in.count = static_cast<uint32_t>(cloud.count());
    in.means = &cloud.positions()[0].x;
    in.scales = &cloud.scales()[0].x;
    in.rotations = &cloud.rotations()[0].x;
    in.opacities = cloud.opacities().data();
    in.shCoefficients = &cloud.shCoefficients()[0].x;
    in.shDegree = cloud.shDegree();
    in.antialiased = cloud.antialiased();
    if (!cloud.normals().empty())
    {
        in.normals = &cloud.normals()[0].x;
    }
    if (!cloud.baseColorRoughness().empty())
    {
        in.baseColorRoughness = &cloud.baseColorRoughness()[0].x;
    }
    if (!cloud.metallic().empty())
    {
        in.metallic = cloud.metallic().data();
    }
    if (!cloud.visibility().empty())
    {
        in.visibility = cloud.visibility().data();
    }
    return in;
}

// The renderer's current scene, read back through readGaussians (core streams only).
Streams readBack(Renderer & r)
{
    const size_t n = r.gaussianCount();
    const size_t perGaussian = static_cast<size_t>(sabercat::kShCoefficientsPerGaussian);
    std::vector<float> means(n * 4);
    std::vector<float> scales(n * 4);
    std::vector<float> rotations(n * 4);
    std::vector<float> opacities(n);
    std::vector<float> sh(n * perGaussian * 4);
    GaussianHostOutput out;
    out.count = static_cast<uint32_t>(n);
    out.means = means.data();
    out.scales = scales.data();
    out.rotations = rotations.data();
    out.opacities = opacities.data();
    out.shCoefficients = sh.data();
    r.readGaussians(out);

    Streams s;
    s.positions.resize(n);
    std::memcpy(s.positions.data(), means.data(), means.size() * sizeof(float));
    s.scales.resize(n);
    std::memcpy(s.scales.data(), scales.data(), scales.size() * sizeof(float));
    s.rotations.resize(n);
    std::memcpy(s.rotations.data(), rotations.data(), rotations.size() * sizeof(float));
    s.opacities = opacities;
    s.shCoefficients.resize(n * perGaussian);
    std::memcpy(s.shCoefficients.data(), sh.data(), sh.size() * sizeof(float));
    return s;
}

std::vector<float> renderImage(Renderer & r)
{
    r.forward(RenderMode::Viewer);
    return r.readImage();
}

bool sameVec3(const glm::vec3 & a, const glm::vec3 & b)
{
    return sameBits(a.x, b.x) && sameBits(a.y, b.y) && sameBits(a.z, b.z);
}

// --- G: Renderer::mergeLayersOnGpu -----------------------------------------------

void checkGpuMergeScene(Renderer & r)
{
    Device & device = r.sabercatDevice();
    const auto a = makeSynthetic(device, SyntheticSpec{kCountA, 3, true, false, 0xA0A0001u});
    const auto a2 = makeSynthetic(device, SyntheticSpec{kCountA2, 3, true, true, 0xA2A2002u});
    const auto b = makeSynthetic(device, SyntheticSpec{kCountB, 1, false, true, 0xB0B0003u});
    const GaussianCloud * A = a.get();
    const GaussianCloud * A2 = a2.get();
    const GaussianCloud * B = b.get();
    const LayerTransform id = identityTransform();
    const LayerTransform t1 = transformT1();
    const LayerTransform t2 = transformT2();

    // G1: the merged scene and its bounds.
    const std::vector<MergeLayer> g1 = {{A2, t1}, {B, t2}};
    r.mergeLayersOnGpu(g1);
    check(r.gaussianCount() == static_cast<uint32_t>(kCountA2 + kCountB),
          "G1: gaussianCount is the merged count");
    check(r.sceneSHDegree() == 3, "G1: sceneSHDegree is the merged max (3)");
    check(!r.sceneBounds().valid, "G1: sceneBounds invalid after mergeLayersOnGpu");
    r.setSceneBounds(mergedLayerBounds(g1));
    const std::shared_ptr<GaussianCloud> host1 = sabercat::mergeLayers(device, g1);
    const SceneBounds got = r.sceneBounds();
    const SceneBounds & want = host1->bounds();
    check(got.valid == want.valid, "G1: bounds.valid matches mergeLayers()");
    check(sameVec3(got.min, want.min), "G1: bounds.min bit-equal to mergeLayers()");
    check(sameVec3(got.max, want.max), "G1: bounds.max bit-equal to mergeLayers()");
    check(sameVec3(got.focusCenter, want.focusCenter),
          "G1: bounds.focusCenter bit-equal to mergeLayers()");
    check(sameBits(got.focusRadius, want.focusRadius),
          "G1: bounds.focusRadius bit-equal to mergeLayers()");

    // G2: the readback against the host merge, within the Section F tolerances.
    compareMergedStreams(layerSpans(g1), copyStreams(*host1), readBack(r), false, "G2 readGaussians");

    // G3: storage reuse. Capacity reuse needs the same relight presence; a larger
    // total or a different presence allocates.
    const uint64_t gen0 = r.storageGeneration();
    r.mergeLayersOnGpu({{A2, t1}, {B, t1}});
    check(r.storageGeneration() == gen0 && r.gaussianCount() == static_cast<uint32_t>(kCountA2 + kCountB),
          "G3: same count and presence merge in place (generation unchanged)");
    r.mergeLayersOnGpu({{B, id}});
    check(r.storageGeneration() == gen0 && r.gaussianCount() == static_cast<uint32_t>(kCountB),
          "G3: a smaller merge fits the capacity (generation unchanged)");

    r.mergeLayersOnGpu({{A2, t1}, {A, id}});
    const uint64_t gen1 = r.storageGeneration();
    check(gen1 != gen0 && r.gaussianCount() == static_cast<uint32_t>(kCountA2 + kCountA),
          "G3: a change of relight presence allocates (generation advances)");
    r.mergeLayersOnGpu({{A2, t2}});
    check(r.storageGeneration() == gen1 && r.gaussianCount() == static_cast<uint32_t>(kCountA2),
          "G3: a smaller all-relight merge is in place (generation unchanged)");
    r.mergeLayersOnGpu({{A, id}, {A2, id}});
    check(r.storageGeneration() == gen1 && r.gaussianCount() == static_cast<uint32_t>(kCountA + kCountA2),
          "G3: a merge up to the allocated capacity is in place (generation unchanged)");
    r.mergeLayersOnGpu({{A, id}, {A2, id}, {A2, t1}});
    check(r.storageGeneration() > gen1
              && r.gaussianCount() == static_cast<uint32_t>(kCountA + 2 * kCountA2),
          "G3: a merge over the allocated capacity allocates (generation advances)");

    // G4: forward parity. The identity merge renders bit-identical to the host merge;
    // the transformed one differs by the placement's rounding only.
    const std::vector<MergeLayer> identityPair = {{A2, id}, {B, id}};
    r.mergeLayersOnGpu(identityPair);
    const std::vector<float> gpuIdentity = renderImage(r);
    double coverage = 0.0;
    for (size_t p = 3; p < gpuIdentity.size(); p += 4)
    {
        coverage += gpuIdentity[p];
    }
    check(coverage > 0.0, "G4: the identity-merged frame has coverage");
    const std::shared_ptr<GaussianCloud> hostIdentity = sabercat::mergeLayers(device, identityPair);
    r.setGaussians(hostInput(*hostIdentity));
    const std::vector<float> hostIdentityImage = renderImage(r);
    check(bitEqual(gpuIdentity, hostIdentityImage),
          "G4: identity GPU merge renders bit-identical to the host merge");

    r.mergeLayersOnGpu(g1);
    const std::vector<float> gpuImage = renderImage(r);
    r.setGaussians(hostInput(*host1));
    const std::vector<float> hostImage = renderImage(r);
    double sum = 0.0;
    double worst = 0.0;
    const bool sizesMatch = gpuImage.size() == hostImage.size();
    for (size_t i = 0; sizesMatch && i < gpuImage.size(); ++i)
    {
        const double d = std::fabs(static_cast<double>(gpuImage[i]) - static_cast<double>(hostImage[i]));
        sum += d;
        worst = std::max(worst, d);
    }
    const double mean = sizesMatch && !gpuImage.empty() ? sum / static_cast<double>(gpuImage.size()) : 1.0;
    std::printf("  G4 transformed frame: mean |d| %.3g, max |d| %.3g\n", mean, worst);
    check(sizesMatch && mean <= 1e-4, "G4: transformed GPU merge matches the host merge (mean |d| <= 1e-4)");

    // G5: an empty list is refused; clearScene then a merge.
    check(throwsException([&] { r.mergeLayersOnGpu({}); }), "G5: an empty layer list throws");
    r.clearScene();
    check(r.gaussianCount() == 0, "G5: clearScene empties the scene");
    r.mergeLayersOnGpu({{A2, id}});
    check(r.gaussianCount() == static_cast<uint32_t>(kCountA2),
          "G5: a merge after clearScene has the right count");

    // G6: a merge ends the rendered frame: readImage() throws until the next forward().
    // The scene is A2 placed by id. First an in-place merge (same count and presence,
    // new content), then a merge into a new cloud (the relight stream is dropped).
    const std::vector<float> g6Before = renderImage(r);
    const uint64_t g6Gen = r.storageGeneration();
    r.mergeLayersOnGpu({{A2, t1}});
    check(r.storageGeneration() == g6Gen, "G6: the in-place merge keeps the storage");
    check(throwsException([&] { r.readImage(); }),
          "G6: readImage() throws after an in-place merge, until the next forward()");
    const std::vector<float> g6Moved = renderImage(r);
    check(!bitEqual(g6Before, g6Moved),
          "G6: the forward after an in-place merge differs from the frame before it");

    r.mergeLayersOnGpu({{B, id}});
    check(r.storageGeneration() != g6Gen, "G6: a merge that drops the relight stream allocates");
    check(throwsException([&] { r.readImage(); }),
          "G6: readImage() throws after a merge into a new cloud, until the next forward()");
    const std::vector<float> g6New = renderImage(r);
    check(!bitEqual(g6Moved, g6New),
          "G6: the forward after a merge into a new cloud differs from the frame before it");
}

// --- H: visibility refit ----------------------------------------------------------

Light refitLight()
{
    Light light;
    light.type = LightType::Point;
    light.position = glm::vec3(2.0f, 3.0f, 6.0f);
    return light;
}

ShadowSettings rayTracedShadows()
{
    ShadowSettings st;
    st.mode = ShadowMode::RayTraced;
    st.focusCenter = glm::vec3(0.0f);
    st.focusRadius = 3.0f; // explicit focus: the merged scene has no bounds
    st.pointLightRadius = 0.1f;
    st.raySamplesPerAxis = 2;
    st.cacheAcrossFrames = 1;
    return st;
}

// S2's placement: transformT1 with its translation moved by dx along x.
LayerTransform s2Placed(float dx)
{
    LayerTransform t = transformT1();
    t.translation.x += dx;
    return t;
}

void checkRefitSection(Renderer & r)
{
    Device & device = r.sabercatDevice();
    const auto s1 = makeSynthetic(device, SyntheticSpec{3000, 1, false, false, 0x51515151u});
    const auto s2 = makeSynthetic(device, SyntheticSpec{1200, 1, false, false, 0x52525252u});
    const GaussianCloud * S1 = s1.get();
    const GaussianCloud * S2 = s2.get();
    const LayerTransform id = identityTransform();

    r.setCamera(framingCamera());
    r.setLightingEnabled(true);
    r.setLights({refitLight()});
    r.setShadowSettings(rayTracedShadows());

    // H1: the first frame builds.
    r.mergeLayersOnGpu({{S1, id}, {S2, s2Placed(0.0f)}});
    r.forward(RenderMode::Viewer);
    ShadowFrameStats st = r.shadowFrameStats();
    check(st.accelerationBuilds == 1 && st.accelerationRefits == 0,
          "H1: the first frame builds the acceleration structure");

    // H2: an in-place merge with S2 moved refits, and the frame is traced.
    r.mergeLayersOnGpu({{S1, id}, {S2, s2Placed(0.02f)}});
    r.forward(RenderMode::Viewer);
    st = r.shadowFrameStats();
    check(st.accelerationRefits == 1 && st.accelerationBuilds == 0 && st.reusedFrame == 0,
          "H2: an in-place merge with a moved layer refits");
    const std::vector<float> visRefit = r.readVisibility();

    // H3: a forced build of the same scene gives the same visibility as the refit.
    r.markSceneChanged();
    r.forward(RenderMode::Viewer);
    st = r.shadowFrameStats();
    check(st.accelerationBuilds == 1, "H3: markSceneChanged forces a fresh build");
    const std::vector<float> visBuild = r.readVisibility();
    double worst = 0.0;
    const bool sizesMatch = visRefit.size() == visBuild.size();
    for (size_t i = 0; sizesMatch && i < visRefit.size(); ++i)
    {
        worst = std::max(worst, std::fabs(static_cast<double>(visRefit[i]) - static_cast<double>(visBuild[i])));
    }
    std::printf("  H3 refit vs fresh build: max |dV| %.3g\n", worst);
    check(sizesMatch && worst <= 1e-4, "H3: refit visibility matches a fresh build (max |dV| <= 1e-4)");

    // H4: the refit cap. After the build of H3, 16 refits in a row, then a build.
    std::string sequence;
    bool sequenceOk = true;
    bool countsOk = true;
    for (int i = 1; i <= 17; ++i)
    {
        r.mergeLayersOnGpu({{S1, id}, {S2, s2Placed(0.02f + 0.01f * static_cast<float>(i))}});
        r.forward(RenderMode::Viewer);
        st = r.shadowFrameStats();
        const bool expectBuild = i == 17;
        const bool ok = expectBuild ? (st.accelerationBuilds == 1 && st.accelerationRefits == 0)
                                    : (st.accelerationBuilds == 0 && st.accelerationRefits == 1);
        sequenceOk = sequenceOk && ok;
        // Refits since the last build: frame i is the i-th refit; the build resets it to 0.
        const uint32_t sinceBuild = expectBuild ? 0u : static_cast<uint32_t>(i);
        countsOk = countsOk && st.accelerationRefitsSinceBuild == sinceBuild;
        sequence += st.accelerationBuilds != 0 ? " B" : (st.accelerationRefits != 0 ? " R" : " -");
    }
    check(sequenceOk, "H4: 16 refits in a row, then a build on frame 17 (frames:" + sequence + ")");
    check(countsOk, "H4: accelerationRefitsSinceBuild counts the refits since the last build "
                    "(frame k reports k, the build reports 0)");

    // H5: a different total count builds.
    r.mergeLayersOnGpu({{S1, id}});
    r.forward(RenderMode::Viewer);
    st = r.shadowFrameStats();
    check(st.accelerationBuilds == 1 && st.accelerationRefits == 0,
          "H5: a merge with a different total count builds");
}

} // namespace

void checkRendererMerge(Device &)
{
    Renderer r(kWidth, kHeight);
    if (!r.initialize({}, true, testShaderDir()))
    {
        check(false, "G: renderer initialize");
        return;
    }
    r.setCamera(framingCamera());
    r.setMaxSHDegree(3);
    checkGpuMergeScene(r);
}

void checkRefit(Device &)
{
    Renderer r(kWidth, kHeight);
    if (!r.initialize({}, true, testShaderDir()))
    {
        check(false, "H: renderer initialize");
        return;
    }
    if (!r.deviceFeatures().rayQuery)
    {
        std::printf("[SKIP] ray query not enabled on this device: no refit checks\n");
        return;
    }
    checkRefitSection(r);
}

void timeMergeScene(Device &)
{
    const char * path = std::getenv("SABERCAT_EDIT_TEST_SCENE");
    if (path == nullptr || *path == '\0')
    {
        return;
    }
    // The file name only: the path itself stays out of the output.
    const std::string name = std::filesystem::path(path).filename().string();

    Renderer r(kWidth, kHeight);
    if (!r.initialize({}, true, testShaderDir()))
    {
        std::printf("merge timing: skipped (renderer initialize failed)\n");
        return;
    }
    Device & device = r.sabercatDevice();
    std::shared_ptr<GaussianCloud> scene;
    try
    {
        scene = loadScene(device, std::string(path));
    }
    catch (const std::exception &)
    {
        std::printf("merge timing: skipped (%s did not load)\n", name.c_str());
        return;
    }
    if (scene->positionBuffer() == nullptr)
    {
        scene->uploadToGPU();
    }

    using Clock = std::chrono::steady_clock;
    const std::vector<MergeLayer> layers = {{scene.get(), transformT1()}};
    r.mergeLayersOnGpu(layers);
    r.mergeLayersOnGpu(layers);
    std::vector<double> gpuMs;
    for (int k = 0; k < 10; ++k)
    {
        const Clock::time_point t0 = Clock::now();
        r.mergeLayersOnGpu(layers);
        gpuMs.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
    }
    std::sort(gpuMs.begin(), gpuMs.end());
    const double median = 0.5 * (gpuMs[4] + gpuMs[5]);
    const double gpuMin = gpuMs.front();

    const Clock::time_point h0 = Clock::now();
    const std::shared_ptr<GaussianCloud> host = sabercat::mergeLayers(device, layers);
    r.setGaussians(hostInput(*host));
    const double hostMs = std::chrono::duration<double, std::milli>(Clock::now() - h0).count();

    std::printf("merge timing: file %s\n", name.c_str());
    std::printf("merge timing: gpu median %.3f ms min %.3f ms (%zu Gaussians); host mergeLayers+setGaussians %.3f ms\n",
                median, gpuMin, scene->count(), hostMs);

    // Breakdown: each case runs 2 warmups, then 10 timed calls; median and min.
    const auto timeCalls = [](const auto & call) -> std::pair<double, double>
    {
        call();
        call();
        std::vector<double> samples;
        for (int k = 0; k < 10; ++k)
        {
            const Clock::time_point t0 = Clock::now();
            call();
            samples.push_back(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
        }
        std::sort(samples.begin(), samples.end());
        return {0.5 * (samples[4] + samples[5]), samples.front()};
    };

    // (a) Renderer::mergeLayersOnGpu with an identity transform on the full scene.
    const std::vector<MergeLayer> identityLayers = {{scene.get(), identityTransform()}};
    const std::pair<double, double> identityTimes =
        timeCalls([&] { r.mergeLayersOnGpu(identityLayers); });
    std::printf("merge timing: identity renderer median %.3f ms min %.3f ms\n",
                identityTimes.first, identityTimes.second);

    // (b) EditPass::mergeLayers directly on the scene's layout, with the same T1
    // transform: the Renderer's overhead is excluded.
    EditPass edit(device, testShaderDir());
    const MergedLayout sceneLayout = planLayerMerge(layers);
    GaussianCloud sceneDest(device);
    sceneDest.allocateGPU(sceneLayout.count, 0, sceneLayout.presence);
    sceneDest.setShDegree(sceneLayout.shDegree);
    const std::pair<double, double> directTimes =
        timeCalls([&] { edit.mergeLayers(layers, sceneDest); });
    std::printf("merge timing: direct EditPass T1 median %.3f ms min %.3f ms\n",
                directTimes.first, directTimes.second);

    // (c) The fixed per-call cost: a 1000-Gaussian layer through the same direct call.
    const auto tiny = makeSynthetic(device, SyntheticSpec{1000, 3, false, false, 0x1000A1u});
    const std::vector<MergeLayer> tinyLayers = {{tiny.get(), transformT1()}};
    const MergedLayout tinyLayout = planLayerMerge(tinyLayers);
    GaussianCloud tinyDest(device);
    tinyDest.allocateGPU(tinyLayout.count, 0, tinyLayout.presence);
    tinyDest.setShDegree(tinyLayout.shDegree);
    const std::pair<double, double> tinyTimes =
        timeCalls([&] { edit.mergeLayers(tinyLayers, tinyDest); });
    std::printf("merge timing: direct EditPass 1000 Gaussians median %.3f ms min %.3f ms\n",
                tinyTimes.first, tinyTimes.second);
}

} // namespace edit_test
