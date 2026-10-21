// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// SH band-fade transitions + DebugView::SHDegree.
//
// Headless, numeric (no screenshots), in three parts:
//
//   (a) Fade off is the pre-fade renderer. With adaptive LOD off, the fade
//       width is inert (image bit-identical to default LODParams); with the
//       fade width 0, an adaptive degree d renders bit-identically to a
//       global cap of d (the plain evaluateSH path); a Differentiable run
//       ignores the viewer's fade like every other viewer knob.
//   (b) The fade is continuous. A 3x3 grid of Gaussians with energy in every
//       SH band is dollied across all three degree thresholds in small steps,
//       once with hard switches and once per fade width. Reported per
//       config: the largest per-step change of the image, of the LOD error
//       image (LOD image minus the full-quality image at the same camera --
//       this removes the dolly's own geometric change and isolates the
//       degree transition), and of the per-Gaussian effective degree. The
//       faded colors are also checked against a C++ reference of the fade.
//   (c) The SHDegree view's output matches a C++ reference: per-Gaussian
//       degrees recomputed from the LOD formula, blended per pixel over the
//       GPU's own sorted instance list with the blend weights T_i * alpha_i,
//       mapped through the same color ramp and scaled by coverage.

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "sabercat/buffer.h"
#include "sabercat/camera.h"
#include "sabercat/debug_visualizer.h"
#include "sabercat/device.h"
#include "sabercat/forward_pass.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/render_config.h"

using namespace sabercat;

namespace
{

constexpr uint32_t kWidth = 256;
constexpr uint32_t kHeight = 256;
constexpr float kFocal = 256.0f;

struct TestGaussian
{
    glm::vec3 position;
    float scale;
    float opacity;
    glm::vec3 sh[kShCoefficients]; // band-ascending, like the loader
};

// LOD settings shared by (b) and (c): distance only (density isolated off,
// it is the same pressure input forward_test already covers), and a full
// distance drop of 1.0 so degreeSteps = (depth - 5) / 10 -- thresholds at
// depths 15, 25 and 35.
constexpr float kNear = 5.0f;
constexpr float kFar = 45.0f;
constexpr float kMaxDrop = 1.0f;

LODParams appearanceLOD(float fadeWidth)
{
    LODParams p{};
    p.enabled = 0;
    p.appearanceEnabled = 1;
    p.nearDistance = kNear;
    p.farDistance = kFar;
    p.densityReference = 1.0f;
    p.appearanceMaxDropDistance = kMaxDrop;
    p.appearanceMaxDropDensity = 0.0f;
    p.appearanceFadeWidth = fadeWidth;
    return p;
}

// -- C++ references, mirroring forward_project.slang / forward_common.slangh. --

struct DegreeChoice
{
    uint32_t degree;     // effective (integer) degree evaluated
    float topBandWeight; // 1 unless the band fade is active
    float fractional() const
    {
        return static_cast<float>(degree) - (1.0f - topBandWeight);
    }
};

DegreeChoice referenceDegree(float depth, uint32_t cap, const LODParams & lod)
{
    float distanceFraction = std::clamp(
        (depth - lod.nearDistance) / std::max(lod.farDistance - lod.nearDistance, 1e-4f), 0.0f,
        1.0f);
    float degrade = distanceFraction * lod.appearanceMaxDropDistance; // density term is 0
    float steps = degrade * static_cast<float>(kShDegree + 1);
    int stepsLost = static_cast<int>(std::floor(steps));
    int adaptive = std::clamp(static_cast<int>(kShDegree) - stepsLost, 0, static_cast<int>(kShDegree));
    DegreeChoice c{std::min(cap, static_cast<uint32_t>(adaptive)), 1.0f};
    float fadeWidth = std::min(lod.appearanceFadeWidth, 1.0f);
    if (fadeWidth > 0.0f && adaptive >= 1 && static_cast<uint32_t>(adaptive) <= cap)
    {
        c.topBandWeight =
            std::clamp((1.0f - (steps - static_cast<float>(stepsLost))) / fadeWidth, 0.0f, 1.0f);
    }
    return c;
}

glm::vec3 shBand(uint32_t band, const glm::vec3 * c, const glm::vec3 & dir)
{
    float x = dir.x, y = dir.y, z = dir.z;
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, yz = y * z, xz = x * z;
    switch (band)
    {
    case 0:
        return kShC0 * c[0];
    case 1:
        return kShC1 * (-y * c[1] + z * c[2] - x * c[3]);
    case 2:
        return kShC2_0 * xy * c[4] + kShC2_1 * yz * c[5] + kShC2_2 * (2.0f * zz - xx - yy) * c[6]
               + kShC2_3 * xz * c[7] + kShC2_4 * (xx - yy) * c[8];
    default:
        return kShC3_0 * y * (3.0f * xx - yy) * c[9] + kShC3_1 * xy * z * c[10]
               + kShC3_2 * y * (4.0f * zz - xx - yy) * c[11]
               + kShC3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * c[12]
               + kShC3_4 * x * (4.0f * zz - xx - yy) * c[13] + kShC3_5 * z * (xx - yy) * c[14]
               + kShC3_6 * x * (xx - 3.0f * yy) * c[15];
    }
}

glm::vec3 referenceColor(const DegreeChoice & choice, const glm::vec3 * c, const glm::vec3 & dir)
{
    glm::vec3 raw(0.0f);
    for (uint32_t band = 0; band <= choice.degree; ++band)
    {
        float weight = band == choice.degree && band >= 1 ? choice.topBandWeight : 1.0f;
        raw += weight * shBand(band, c, dir);
    }
    return glm::clamp(raw + 0.5f, 0.0f, 1.0f);
}

// Mirrors debug_visualize.slang's shDegreeHeat.
glm::vec3 referenceHeat(float degree)
{
    static const glm::vec3 stops[4] = {
        glm::vec3(0.15f, 0.25f, 0.95f),
        glm::vec3(0.15f, 0.85f, 0.25f),
        glm::vec3(0.95f, 0.85f, 0.15f),
        glm::vec3(0.95f, 0.20f, 0.15f),
    };
    float d = std::clamp(degree, 0.0f, static_cast<float>(kShDegree));
    uint32_t low = std::min(static_cast<uint32_t>(std::floor(d)), kShDegree - 1);
    float t = d - static_cast<float>(low);
    return stops[low] + t * (stops[low + 1] - stops[low]);
}

// -- Harness helpers. --

uint32_t g_rng = 12345u;
float nextUnit()
{
    g_rng = g_rng * 1664525u + 1013904223u;
    return static_cast<float>(g_rng >> 8) * (1.0f / 16777216.0f);
}

void fillCloud(GaussianCloud & cloud, const std::vector<TestGaussian> & gaussians)
{
    cloud.initialize(gaussians.size());
    for (size_t i = 0; i < gaussians.size(); ++i)
    {
        const TestGaussian & g = gaussians[i];
        cloud.positions()[i] = glm::vec4(g.position, 0.0f);
        cloud.scales()[i] = glm::vec4(glm::vec3(g.scale), 0.0f);
        cloud.rotations()[i] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        cloud.opacities()[i] = g.opacity;
        for (uint32_t k = 0; k < kShCoefficients; ++k)
        {
            cloud.shCoefficients()[i * kShCoefficients + k] = glm::vec4(g.sh[k], 0.0f);
        }
    }
    cloud.uploadToGPU();
}

// Every band carries visible energy, different per Gaussian and per band.
void randomSH(TestGaussian & g)
{
    g.sh[0] = glm::vec3(nextUnit() - 0.5f, nextUnit() - 0.5f, nextUnit() - 0.5f);
    for (uint32_t k = 1; k < kShCoefficients; ++k)
    {
        g.sh[k] = 0.6f * glm::vec3(nextUnit() - 0.5f, nextUnit() - 0.5f, nextUnit() - 0.5f);
    }
}

// Camera on the +Z axis at `distance`, looking down -Z (no rotation).
CameraUniforms dollyCamera(float distance, uint32_t gaussianCount, uint32_t shCap = kShDegree)
{
    CameraUniforms c{};
    c.worldToCamera = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -distance));
    c.fx = kFocal;
    c.fy = kFocal;
    c.cx = 0.5f * kWidth;
    c.cy = 0.5f * kHeight;
    c.imageWidth = kWidth;
    c.imageHeight = kHeight;
    c.tilesX = dispatchGroups(kWidth, kTileSize);
    c.tilesY = dispatchGroups(kHeight, kTileSize);
    c.gaussianCount = gaussianCount;
    c.shDegree = shCap;
    c.cameraPositionWorld = glm::vec4(0.0f, 0.0f, distance, 0.0f);
    return c;
}

std::vector<glm::vec4> render(ForwardPass & pass, const CanonicalGaussianBufferViews & views,
                              const CameraUniforms & camera, const LODParams & lod,
                              ForwardMode mode = ForwardMode::Viewer)
{
    pass.setLODParams(lod);
    pass.run(views, camera, ForwardStage::Rasterize, mode);
    return pass.downloadOutputImage();
}

template <typename T> std::vector<T> download(const Buffer & buffer, size_t count)
{
    std::vector<T> out(count);
    buffer.download(out.data(), count * sizeof(T));
    return out;
}

bool bitIdentical(const std::vector<glm::vec4> & a, const std::vector<glm::vec4> & b)
{
    return a.size() == b.size()
           && std::equal(a.begin(), a.end(), b.begin(),
                         [](const glm::vec4 & x, const glm::vec4 & y)
                         { return std::memcmp(&x, &y, sizeof(glm::vec4)) == 0; });
}

float maxAbsRGB(const std::vector<glm::vec4> & a, const std::vector<glm::vec4> & b)
{
    float worst = 0.0f;
    for (size_t i = 0; i < a.size(); ++i)
    {
        glm::vec3 d = glm::abs(glm::vec3(a[i]) - glm::vec3(b[i]));
        worst = std::max({worst, d.x, d.y, d.z});
    }
    return worst;
}

// The 3x3 grid (b) dollies across; all at z = 0, so every Gaussian's depth is
// exactly the camera distance and they cross each threshold together -- the
// worst case for a pop.
std::vector<TestGaussian> gridScene()
{
    std::vector<TestGaussian> grid;
    for (int gy = -1; gy <= 1; ++gy)
    {
        for (int gx = -1; gx <= 1; ++gx)
        {
            TestGaussian g{};
            g.position = glm::vec3(1.2f * gx, 1.2f * gy, 0.0f);
            g.scale = 0.5f;
            g.opacity = 0.8f;
            randomSH(g);
            grid.push_back(g);
        }
    }
    return grid;
}

void partA(Device & device, ForwardPass & pass)
{
    std::vector<TestGaussian> grid = gridScene();
    GaussianCloud cloud(device);
    fillCloud(cloud, grid);
    auto views = cloud.views();

    // Adaptive LOD off: the fade width must be inert.
    CameraUniforms camera = dollyCamera(21.0f, views.gaussianCount);
    std::vector<glm::vec4> plain = render(pass, views, camera, LODParams{});
    LODParams offWithFade = appearanceLOD(1.0f);
    offWithFade.appearanceEnabled = 0;
    if (!bitIdentical(plain, render(pass, views, camera, offWithFade)))
    {
        throw std::runtime_error("(a) fade width changed the image with appearance LOD off");
    }

    // Fade 0: adaptive degree d renders exactly like a global cap of d. At
    // distance 21, steps = 1.6 -> degree 2; at 31, steps = 2.6 -> degree 1.
    for (float distance : {21.0f, 31.0f})
    {
        CameraUniforms lodCamera = dollyCamera(distance, views.gaussianCount);
        uint32_t degree = referenceDegree(distance, kShDegree, appearanceLOD(0.0f)).degree;
        std::vector<glm::vec4> hard = render(pass, views, lodCamera, appearanceLOD(0.0f));
        CameraUniforms capped = lodCamera;
        capped.shDegree = degree;
        if (!bitIdentical(hard, render(pass, views, capped, LODParams{})))
        {
            throw std::runtime_error("(a) fade-0 adaptive degree differs from the same global cap");
        }
        // Non-degeneracy: the fade does change this frame.
        if (bitIdentical(hard, render(pass, views, lodCamera, appearanceLOD(1.0f))))
        {
            throw std::runtime_error("(a) fixture degenerate: fade 1.0 changed nothing");
        }
    }

    // Differentiable ignores the viewer's fade (and adaptive LOD) entirely.
    std::vector<glm::vec4> differentiable = render(pass, views, camera, appearanceLOD(1.0f),
                                                   ForwardMode::Differentiable);
    if (!bitIdentical(plain, differentiable))
    {
        throw std::runtime_error("(a) viewer fade reached a Differentiable run");
    }

    std::cout << "(a) fade-off identity OK: fade inert with appearance LOD off, fade-0 "
                 "adaptive == same global cap (degrees 2 and 1), Differentiable unaffected -- "
                 "all bit-identical\n";
}

void partB(Device & device, ForwardPass & pass)
{
    std::vector<TestGaussian> grid = gridScene();
    GaussianCloud cloud(device);
    fillCloud(cloud, grid);
    auto views = cloud.views();

    // Distance 10 -> 40 crosses the thresholds at 15, 25, 35.
    constexpr float kStart = 10.0f;
    constexpr float kStep = 0.1f;
    constexpr int kSteps = 300;
    const float fadeWidths[] = {0.0f, 0.25f, 0.5f, 1.0f};
    constexpr size_t kConfigs = sizeof(fadeWidths) / sizeof(fadeWidths[0]);

    struct Stats
    {
        float maxImageStep = 0.0f;          // max over steps of max |I_k - I_{k-1}|
        float maxImageStepAtThreshold = 0.0f;
        float maxImageStepElsewhere = 0.0f;
        float maxErrorStep = 0.0f;          // same, for E_k = I_lod - I_full
        float maxErrorStepAtThreshold = 0.0f;
        float maxDegreeStep = 0.0f;         // max |D_k - D_{k-1}| of the centre Gaussian
        float worstColorError = 0.0f;       // vs referenceColor
    };
    Stats stats[kConfigs];
    std::vector<glm::vec4> previousImage[kConfigs];
    std::vector<glm::vec3> previousError[kConfigs];
    float previousDegree[kConfigs] = {};
    int thresholdSteps = 0;

    for (int k = 0; k <= kSteps; ++k)
    {
        float distance = kStart + kStep * static_cast<float>(k);
        CameraUniforms camera = dollyCamera(distance, views.gaussianCount);
        bool atThreshold =
            k > 0
            && referenceDegree(distance, kShDegree, appearanceLOD(0.0f)).degree
                   != referenceDegree(distance - kStep, kShDegree, appearanceLOD(0.0f)).degree;
        thresholdSteps += atThreshold ? 1 : 0;

        std::vector<glm::vec4> full = render(pass, views, camera, LODParams{});

        for (size_t c = 0; c < kConfigs; ++c)
        {
            LODParams lod = appearanceLOD(fadeWidths[c]);
            std::vector<glm::vec4> image = render(pass, views, camera, lod);
            std::vector<ProjectedGaussian> projected =
                download<ProjectedGaussian>(pass.projected(), grid.size());
            std::vector<float> degrees = download<float>(pass.shDegrees(), grid.size());

            // Faded colors against the C++ reference, every Gaussian.
            for (size_t i = 0; i < grid.size(); ++i)
            {
                float depth = distance - grid[i].position.z;
                DegreeChoice choice = referenceDegree(depth, kShDegree, lod);
                glm::vec3 dir = glm::normalize(grid[i].position - glm::vec3(0.0f, 0.0f, distance));
                glm::vec3 expected = referenceColor(choice, grid[i].sh, dir);
                glm::vec3 diff = glm::abs(projected[i].color - expected);
                stats[c].worstColorError =
                    std::max({stats[c].worstColorError, diff.x, diff.y, diff.z,
                              std::fabs(degrees[i] - choice.fractional())});
            }

            std::vector<glm::vec3> error(image.size());
            for (size_t p = 0; p < image.size(); ++p)
            {
                error[p] = glm::vec3(image[p]) - glm::vec3(full[p]);
            }

            if (k > 0)
            {
                float imageStep = maxAbsRGB(image, previousImage[c]);
                float errorStep = 0.0f;
                for (size_t p = 0; p < error.size(); ++p)
                {
                    glm::vec3 d = glm::abs(error[p] - previousError[c][p]);
                    errorStep = std::max({errorStep, d.x, d.y, d.z});
                }
                float degreeStep = std::fabs(degrees[4] - previousDegree[c]); // centre Gaussian

                Stats & s = stats[c];
                s.maxImageStep = std::max(s.maxImageStep, imageStep);
                s.maxErrorStep = std::max(s.maxErrorStep, errorStep);
                s.maxDegreeStep = std::max(s.maxDegreeStep, degreeStep);
                if (atThreshold)
                {
                    s.maxImageStepAtThreshold = std::max(s.maxImageStepAtThreshold, imageStep);
                    s.maxErrorStepAtThreshold = std::max(s.maxErrorStepAtThreshold, errorStep);
                }
                else
                {
                    s.maxImageStepElsewhere = std::max(s.maxImageStepElsewhere, imageStep);
                }
            }
            previousImage[c] = std::move(image);
            previousError[c] = std::move(error);
            previousDegree[c] = degrees[4];
        }
    }

    if (thresholdSteps != 3)
    {
        throw std::runtime_error("(b) sweep fixture did not cross exactly three thresholds");
    }

    std::printf("(b) dolly %.2f -> %.2f in %d steps of %.2f, %d threshold crossings\n", kStart,
                kStart + kStep * kSteps, kSteps, kStep, thresholdSteps);
    std::printf("    fade  | max image step  (at threshold / elsewhere) | max LOD-error step "
                "(at threshold) | max degree step | worst ref error\n");
    for (size_t c = 0; c < kConfigs; ++c)
    {
        const Stats & s = stats[c];
        std::printf("    %.2f  | %.5f        (%.5f / %.5f)          | %.5f          (%.5f)     "
                    "  | %.4f          | %.2e\n",
                    fadeWidths[c], s.maxImageStep, s.maxImageStepAtThreshold,
                    s.maxImageStepElsewhere, s.maxErrorStep, s.maxErrorStepAtThreshold,
                    s.maxDegreeStep, s.worstColorError);
    }

    const Stats & hard = stats[0];
    // Non-degeneracy: the hard switch must actually pop here, or nothing
    // below proves anything.
    if (hard.maxErrorStepAtThreshold < 0.05f || hard.maxDegreeStep != 1.0f)
    {
        throw std::runtime_error("(b) fixture degenerate: hard switches do not pop");
    }
    for (size_t c = 0; c < kConfigs; ++c)
    {
        const Stats & s = stats[c];
        if (s.worstColorError > 1e-5f)
        {
            throw std::runtime_error("(b) faded color/degree differs from the C++ reference");
        }
        if (c == 0)
        {
            continue;
        }
        // Continuous: the degree moves at most step / (10 * width) per step
        // (10 = depth units per degree step here), plus float slack ...
        float degreeBound = kStep / (10.0f * fadeWidths[c]) * 1.01f + 1e-5f;
        // ... and no image jump at the thresholds beyond the dolly's own
        // per-step change elsewhere.
        if (s.maxDegreeStep > degreeBound || s.maxImageStepAtThreshold > s.maxImageStepElsewhere
            || s.maxErrorStep > 0.25f * hard.maxErrorStep)
        {
            throw std::runtime_error("(b) band fade is not continuous");
        }
    }
    std::cout << "(b) band fade continuous OK: no step at the thresholds exceeds the dolly's "
                 "own per-step change; hard switches pop\n";
}

void partC(Device & device, ForwardPass & pass)
{
    // ~40 overlapping Gaussians spread over depths 8-40, so pixels blend
    // several different (fractional) degrees.
    std::vector<TestGaussian> scene;
    for (int i = 0; i < 40; ++i)
    {
        TestGaussian g{};
        float depth = 8.0f + 32.0f * nextUnit();
        g.position = glm::vec3((nextUnit() - 0.5f) * 0.6f * depth, (nextUnit() - 0.5f) * 0.6f * depth,
                               -depth);
        g.scale = 0.08f * depth * (0.5f + nextUnit());
        g.opacity = 0.3f + 0.6f * nextUnit();
        randomSH(g);
        scene.push_back(g);
    }
    GaussianCloud cloud(device);
    fillCloud(cloud, scene);
    auto views = cloud.views();

    DebugVisualizer visualizer(device, kWidth, kHeight, SHADER_DIR);
    Buffer readback(device, static_cast<VkDeviceSize>(kWidth) * kHeight * sizeof(glm::vec4));

    // cap 3 (fade active on every band), cap 2 (Gaussians the cap binds get
    // no fade), and LOD off (a uniform cap-3 view).
    struct Case
    {
        uint32_t cap;
        LODParams lod;
        const char * name;
    };
    const Case cases[] = {
        {3, appearanceLOD(0.6f), "cap 3, fade 0.6"},
        {2, appearanceLOD(0.6f), "cap 2, fade 0.6"},
        {3, LODParams{}, "LOD off"},
    };
    for (const Case & testCase : cases)
    {
        CameraUniforms camera = dollyCamera(0.0f, views.gaussianCount, testCase.cap);
        pass.setLODParams(testCase.lod);
        pass.run(views, camera);

        device.immediateSubmit(
            [&](VkCommandBuffer cmd)
            {
                visualizer.record(cmd, DebugView::SHDegree, pass, 50.0f, 64.0f, 0);
                VkBufferImageCopy region{};
                region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                region.imageSubresource.layerCount = 1;
                region.imageExtent = {kWidth, kHeight, 1};
                vkCmdCopyImageToBuffer(cmd, visualizer.displayImage(0).getHandle(),
                                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.handle(), 1,
                                       &region);
            });
        std::vector<glm::vec4> gpu = download<glm::vec4>(readback, size_t(kWidth) * kHeight);

        // GPU-side inputs the walk reads (verified stage by stage in
        // forward_test); the degrees are recomputed independently below.
        std::vector<ProjectedGaussian> projected =
            download<ProjectedGaussian>(pass.projected(), scene.size());
        std::vector<float> gpuDegrees = download<float>(pass.shDegrees(), scene.size());
        uint32_t tilesX = camera.tilesX;
        std::vector<TileRange> ranges =
            download<TileRange>(pass.tileRanges(), size_t(tilesX) * camera.tilesY);
        std::vector<uint32_t> sorted =
            download<uint32_t>(pass.sortedValues(), std::max(pass.lastInstanceCount(), 1u));

        std::vector<float> degrees(scene.size(), 0.0f);
        float worstDegreeError = 0.0f;
        bool sawFractional = false;
        for (size_t i = 0; i < scene.size(); ++i)
        {
            if (projected[i].tileCount == 0)
            {
                continue;
            }
            DegreeChoice choice = testCase.lod.appearanceEnabled != 0
                                      ? referenceDegree(projected[i].depth, testCase.cap, testCase.lod)
                                      : DegreeChoice{testCase.cap, 1.0f};
            degrees[i] = choice.fractional();
            sawFractional |= choice.topBandWeight > 0.0f && choice.topBandWeight < 1.0f;
            worstDegreeError = std::max(worstDegreeError, std::fabs(gpuDegrees[i] - degrees[i]));
        }

        float worstPixelError = 0.0f;
        float minDegree = 1e9f, maxDegree = -1e9f;
        uint32_t coveredPixels = 0;
        for (uint32_t y = 0; y < kHeight; ++y)
        {
            for (uint32_t x = 0; x < kWidth; ++x)
            {
                TileRange range = ranges[(y / kTileSize) * tilesX + x / kTileSize];
                glm::vec2 pixel(static_cast<float>(x), static_cast<float>(y));
                float transmittance = 1.0f, weightSum = 0.0f, weightedDegree = 0.0f;
                for (uint32_t idx = range.first; idx < range.last; ++idx)
                {
                    uint32_t gi = sorted[idx];
                    const ProjectedGaussian & g = projected[gi];
                    glm::vec2 d = pixel - g.center;
                    float exponent =
                        g.conic.x * d.x * d.x + 2.0f * g.conic.y * d.x * d.y + g.conic.z * d.y * d.y;
                    if (exponent < 0.0f)
                    {
                        continue;
                    }
                    float alpha = std::min(g.alpha * std::exp(-0.5f * exponent), kMaxAlpha);
                    if (alpha < 1.0f / 255.0f)
                    {
                        continue;
                    }
                    float weight = transmittance * alpha;
                    weightSum += weight;
                    weightedDegree += weight * degrees[gi];
                    transmittance *= 1.0f - alpha;
                    if (transmittance < 1e-4f)
                    {
                        break;
                    }
                }
                float degree = weightSum > 0.0f ? weightedDegree / weightSum : 0.0f;
                glm::vec3 expected = referenceHeat(degree) * (1.0f - transmittance);
                const glm::vec4 & got = gpu[y * kWidth + x];
                glm::vec3 diff = glm::abs(glm::vec3(got) - expected);
                worstPixelError = std::max({worstPixelError, diff.x, diff.y, diff.z,
                                            std::fabs(got.a - 1.0f)});
                if (weightSum > 0.5f)
                {
                    ++coveredPixels;
                    minDegree = std::min(minDegree, degree);
                    maxDegree = std::max(maxDegree, degree);
                }
            }
        }

        std::printf("(c) %-16s: %u covered px, pixel degree %.3f..%.3f, worst per-Gaussian "
                    "degree error %.2e, worst pixel error %.2e\n",
                    testCase.name, coveredPixels, minDegree, maxDegree, worstDegreeError,
                    worstPixelError);
        if (worstDegreeError > 1e-6f || worstPixelError > 1e-5f)
        {
            throw std::runtime_error("(c) SHDegree view differs from the C++ reference");
        }
        if (coveredPixels < 1000)
        {
            throw std::runtime_error("(c) fixture degenerate: too little coverage");
        }
        if (testCase.lod.appearanceEnabled != 0 && (!sawFractional || maxDegree - minDegree < 1.0f))
        {
            throw std::runtime_error("(c) fixture degenerate: no fractional / spread degrees");
        }
    }
    std::cout << "(c) SHDegree view OK: matches the C++ reference (blend-weighted degree, "
                 "ramp, coverage)\n";
}

} // namespace

int main()
{
    Device device;
    int result = 0;
    try
    {
        device.initialize(/*instanceExtensions=*/{}, /*enableValidation=*/true);
        ForwardPass pass(device, kWidth, kHeight, SHADER_DIR);
        partA(device, pass);
        partB(device, pass);
        partC(device, pass);
        std::cout << "sh_lod_test OK\n";
    }
    catch (const std::exception & e)
    {
        std::cerr << "sh_lod_test FAILED: " << e.what() << std::endl;
        result = 1;
    }
    device.shutdown();
    return result;
}
