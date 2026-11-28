// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "study_scenes.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "sabercat/device.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/render_config.h"
#include "sabercat/scene_bounds.h"
#include "sabercat/scene_loader.h"

namespace sabercat::study
{
namespace
{

// Synthetic scene layout. Two flat sheets in the z = 0 and z = kSheetGap
// planes, interleaved in the array, plus a ring of elongated splats in z = 0.
constexpr uint32_t kSheetCount = 20000; // per sheet
constexpr uint32_t kRingCount = 2000;
constexpr uint32_t kSeed = 0x5A8E1D2Bu;
constexpr float kSheetHalfExtent = 1.5f;
constexpr float kSheetGap = 1e-4f;
constexpr float kSheetOpacity = 0.6f;
constexpr float kRingRadius = 1.0f;
constexpr float kRingOpacity = 0.6f;
// Sheets fill the view at 15 degrees elevation: the vertical half-height at
// 1.8 * radius is about radius, so ~1.6 keeps the +-1.5 sheets on screen.
constexpr float kSyntheticRadius = 1.6f;
constexpr float kCameraDistanceFactor = 1.8f;
constexpr float kFocalPerHeight = 0.9f; // fx = fy = 0.9 * height, as sort_modes_test --timing
constexpr float kDegToRad = 3.14159265358979323846f / 180.0f;

// Small xorshift32, so the synthetic scene is bit-identical in Debug and Release
// and does not depend on host math library output.
class Xorshift32
{
  public:
    explicit Xorshift32(uint32_t seed) : m_state(seed != 0 ? seed : 0x9E3779B9u) {}

    uint32_t next()
    {
        m_state ^= m_state << 13;
        m_state ^= m_state >> 17;
        m_state ^= m_state << 5;
        return m_state;
    }

    // Uniform in [0, 1) with 24 random bits, exact in float.
    float unit()
    {
        return static_cast<float>(next() >> 8) * (1.0f / 16777216.0f);
    }

  private:
    uint32_t m_state;
};

struct SplatSpec
{
    glm::vec3 position;
    glm::vec3 scale;
    glm::vec4 rotation; // xyzw
    float opacity;
    glm::vec3 color;
};

// Degree-0 SH coefficient whose evaluated color is `color`: the shader computes
// color = clamp(kShC0 * sh0 + 0.5, 0, 1) (forward_common.slangh).
glm::vec3 dcFromColor(const glm::vec3 & color)
{
    return (color - 0.5f) / kShC0;
}

void writeSplat(GaussianCloud & cloud, size_t index, const SplatSpec & spec)
{
    cloud.positions()[index] = glm::vec4(spec.position, 0.0f);
    cloud.scales()[index] = glm::vec4(spec.scale, 0.0f);
    cloud.rotations()[index] = spec.rotation;
    cloud.opacities()[index] = spec.opacity;

    const size_t base = index * static_cast<size_t>(kShCoefficientsPerGaussian);
    cloud.shCoefficients()[base] = glm::vec4(dcFromColor(spec.color), 0.0f);
    for (size_t k = 1; k < static_cast<size_t>(kShCoefficientsPerGaussian); ++k)
    {
        cloud.shCoefficients()[base + k] = glm::vec4(0.0f);
    }
}

glm::vec3 randomColor(Xorshift32 & rng)
{
    const float r = 0.1f + 0.8f * rng.unit();
    const float g = 0.1f + 0.8f * rng.unit();
    const float b = 0.1f + 0.8f * rng.unit();
    return glm::vec3(r, g, b);
}

// Unit point of the first quadrant, from the rational parametrization
// ((1 - t^2), 2t) / (1 + t^2) with t in (0, 1). Only arithmetic, no sin/cos.
glm::vec2 firstQuadrantPoint(float t)
{
    const float inverse = 1.0f / (1.0f + t * t);
    return glm::vec2((1.0f - t * t) * inverse, 2.0f * t * inverse);
}

// Rotates p by q * 90 degrees counter-clockwise (exact).
glm::vec2 rotateQuarter(const glm::vec2 & p, uint32_t q)
{
    switch (q & 3u)
    {
    case 0:
        return p;
    case 1:
        return glm::vec2(-p.y, p.x);
    case 2:
        return glm::vec2(-p.x, -p.y);
    default:
        return glm::vec2(p.y, -p.x);
    }
}

// Rotation about +Z that takes the splat's local +X to the ring tangent (-Y, X)
// at the unit point (X, Y). Built from the half-angle identity
// (1 + e^{i phi}) ~ e^{i phi/2}, so it needs only sqrt, which is correctly
// rounded and therefore identical in Debug and Release. Valid for Y < 1.
glm::vec4 tangentRotation(float x, float y)
{
    const float w = 1.0f - y;
    const float z = x;
    const float norm = std::sqrt(w * w + z * z);
    return glm::vec4(0.0f, 0.0f, z / norm, w / norm); // xyzw
}

// Replaces every occurrence of the full scene path in a loader message with the
// file name, so the directory never reaches the output.
std::string withoutDirectory(std::string message, const std::string & path, const std::string & name)
{
    if (path.empty())
    {
        return message;
    }
    size_t at = 0;
    while ((at = message.find(path, at)) != std::string::npos)
    {
        message.replace(at, path.size(), name);
        at += name.size();
    }
    return message;
}

} // namespace

StudyScene makeSyntheticScene(Device & device)
{
    const size_t sheetSplats = 2 * static_cast<size_t>(kSheetCount);
    const size_t count = sheetSplats + kRingCount;

    auto cloud = std::make_shared<GaussianCloud>(device);
    cloud->initialize(count);
    cloud->setShDegree(0);

    Xorshift32 rng(kSeed);
    const glm::vec4 identity(0.0f, 0.0f, 0.0f, 1.0f); // xyzw
    const glm::vec3 redColor(0.9f, 0.1f, 0.1f);
    const glm::vec3 blueColor(0.1f, 0.1f, 0.9f);
    const glm::vec3 sheetScale(0.03f, 0.03f, 0.0005f);
    const float sheetSpan = 2.0f * kSheetHalfExtent;

    // Sheet A (z = 0, red) at even indices, sheet B (z = kSheetGap, blue) at odd
    // indices: the array interleaves the two sheets.
    for (uint32_t k = 0; k < kSheetCount; ++k)
    {
        for (size_t sheet = 0; sheet < 2; ++sheet)
        {
            SplatSpec spec{};
            spec.position.x = -kSheetHalfExtent + sheetSpan * rng.unit();
            spec.position.y = -kSheetHalfExtent + sheetSpan * rng.unit();
            spec.position.z = sheet == 0 ? 0.0f : kSheetGap;
            spec.scale = sheetScale;
            spec.rotation = identity;
            spec.opacity = kSheetOpacity;
            spec.color = sheet == 0 ? redColor : blueColor;
            writeSplat(*cloud, 2 * static_cast<size_t>(k) + sheet, spec);
        }
    }

    // Ring: 4 quadrants of kRingCount / 4 points, visited in order around the
    // circle. Each splat's long axis (0.12) is along the tangent.
    const uint32_t stepsPerQuadrant = kRingCount / 4;
    for (uint32_t i = 0; i < kRingCount; ++i)
    {
        const uint32_t quadrant = i % 4;
        const uint32_t step = i / 4;
        const float t = (static_cast<float>(step) + 0.5f) / static_cast<float>(stepsPerQuadrant);
        const glm::vec2 unit = rotateQuarter(firstQuadrantPoint(t), quadrant);

        SplatSpec spec{};
        spec.position = glm::vec3(kRingRadius * unit.x, kRingRadius * unit.y, 0.0f);
        spec.scale = glm::vec3(0.12f, 0.02f, 0.02f);
        spec.rotation = tangentRotation(unit.x, unit.y);
        spec.opacity = kRingOpacity;
        spec.color = randomColor(rng);
        writeSplat(*cloud, sheetSplats + i, spec);
    }

    cloud->computeBounds();
    cloud->uploadToGPU();

    StudyScene scene;
    scene.name = "synthetic";
    scene.cloud = std::move(cloud);
    scene.center = glm::vec3(0.0f);
    scene.radius = kSyntheticRadius;
    return scene;
}

StudyScene loadStudyScene(Device & device, const std::string & path)
{
    const std::string name = std::filesystem::path(path).filename().string();

    std::shared_ptr<GaussianCloud> cloud;
    try
    {
        cloud = loadScene(device, path);
    }
    catch (const std::exception & e)
    {
        throw std::runtime_error("loadScene failed for " + name + ": "
                                 + withoutDirectory(e.what(), path, name));
    }
    if (!cloud || cloud->count() == 0)
    {
        throw std::runtime_error("scene " + name + " has no Gaussians");
    }

    // loadScene() computes bounds but does not upload; Renderer::loadScene does
    // the upload, so the harness must do it too.
    cloud->uploadToGPU();

    const SceneBounds & bounds = cloud->bounds();
    if (!bounds.valid)
    {
        throw std::runtime_error("scene " + name + " has no bounds");
    }

    // Robust focus sphere (median center, 90th-percentile radius), as in
    // sort_modes_test --timing. Fall back to the full radius if degenerate.
    float radius = bounds.focusRadius;
    if (!(radius > 0.0f))
    {
        radius = 0.5f * bounds.radius();
    }
    if (!(radius > 0.0f))
    {
        throw std::runtime_error("scene " + name + " has a degenerate extent");
    }

    StudyScene scene;
    scene.name = name;
    scene.cloud = std::move(cloud);
    scene.center = bounds.focusCenter;
    scene.radius = radius;
    return scene;
}

CameraUniforms makeStudyCamera(const StudyScene & scene, uint32_t width, uint32_t height, float azimuthDeg)
{
    const float elevation = scene.elevationDeg * kDegToRad;
    const float azimuth = azimuthDeg * kDegToRad;
    const float distance = kCameraDistanceFactor * scene.radius;
    const glm::vec3 offset(std::sin(azimuth) * std::cos(elevation), std::sin(elevation),
                           std::cos(azimuth) * std::cos(elevation));
    const glm::vec3 eye = scene.center + distance * offset;
    const glm::mat4 worldToCamera = glm::lookAt(eye, scene.center, glm::vec3(0.0f, 1.0f, 0.0f));

    // Same intrinsics as sort_modes_test --timing: square pixels, principal point
    // at the image center.
    const float fy = kFocalPerHeight * static_cast<float>(height);

    CameraUniforms uniforms{};
    uniforms.worldToCamera = worldToCamera;
    uniforms.fx = fy;
    uniforms.fy = fy;
    uniforms.cx = 0.5f * static_cast<float>(width);
    uniforms.cy = 0.5f * static_cast<float>(height);
    uniforms.imageWidth = width;
    uniforms.imageHeight = height;
    uniforms.tilesX = dispatchGroups(width, kTileSize);
    uniforms.tilesY = dispatchGroups(height, kTileSize);
    uniforms.gaussianCount = static_cast<uint32_t>(scene.cloud->count());
    uniforms.shDegree = std::min(scene.cloud->shDegree(), kShDegree);
    uniforms.renderFlags = scene.cloud->antialiased() ? kRenderFlagAntialiased : 0u;
    uniforms.cameraPositionWorld = glm::vec4(eye, 0.0f);
    return uniforms;
}

} // namespace sabercat::study
