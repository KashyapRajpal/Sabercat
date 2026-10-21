// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Numeric diagnostic for a real scene file.
//
// Screen capture of Vulkan content is unreliable in this environment, so "the
// render looks wrong" has to be turned into numbers before it can be debugged.
// This loads a real asset through loadScene(), places the camera exactly the
// way apps/viewer does (frame the 90th-percentile focus sphere), runs
// forward_project, and reports what actually came out:
//
//   - what the loader produced (scale/opacity distributions, bounds)
//   - what forward_project produced (projected radius distribution, how many
//     Gaussians survived, how far the visible surface is)
//
// Usage: sabercat_scene_diag <path-to-scene>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "sabercat/buffer.h"
#include "sabercat/camera.h"
#include "sabercat/device.h"
#include "sabercat/forward_pass.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/render_config.h"
#include "sabercat/scene_loader.h"

using namespace sabercat;

namespace
{

constexpr uint32_t kWidth = 1280;
constexpr uint32_t kHeight = 720;

float percentile(std::vector<float> & v, double p)
{
    if (v.empty())
    {
        return std::nanf("");
    }
    const size_t i = static_cast<size_t>(p * static_cast<double>(v.size() - 1));
    std::nth_element(v.begin(), v.begin() + i, v.end());
    return v[i];
}

void printPercentiles(const char * label, std::vector<float> values)
{
    if (values.empty())
    {
        std::printf("  %-22s (none)\n", label);
        return;
    }
    const float p01 = percentile(values, 0.01);
    const float p50 = percentile(values, 0.50);
    const float p90 = percentile(values, 0.90);
    const float p99 = percentile(values, 0.99);
    float mx = values[0];
    for (float f : values)
    {
        mx = std::max(mx, f);
    }
    std::printf("  %-22s p01=%-10.5g p50=%-10.5g p90=%-10.5g p99=%-10.5g max=%.5g\n",
                label, p01, p50, p90, p99, mx);
}

} // namespace

int main(int argc, char * argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: sabercat_scene_diag <path-to-scene>" << std::endl;
        return 1;
    }
    const std::string path = argv[1];

    Device device;
    int result = 0;
    try
    {
        device.initialize({}, /*enableValidation=*/false);

        auto cloud = loadScene(device, path);
        cloud->uploadToGPU();

        const SceneBounds bounds = cloud->bounds();
        std::printf("\n=== %s ===\n", path.c_str());
        std::printf("  gaussians              %zu\n", cloud->count());
        std::printf("  SH degree              %u\n", cloud->shDegree());
        std::printf("  antialiased            %s\n", cloud->antialiased() ? "yes" : "no");
        std::printf("  bounds min             (%.3f, %.3f, %.3f)\n",
                    bounds.min.x, bounds.min.y, bounds.min.z);
        std::printf("  bounds max             (%.3f, %.3f, %.3f)\n",
                    bounds.max.x, bounds.max.y, bounds.max.z);
        std::printf("  focus center           (%.3f, %.3f, %.3f)\n",
                    bounds.focusCenter.x, bounds.focusCenter.y, bounds.focusCenter.z);
        std::printf("  focus radius / full    %.3f / %.3f\n", bounds.focusRadius, bounds.radius());

        // --- what the loader produced ---
        std::printf("\n-- loader output (canonical values) --\n");
        {
            std::vector<float> scales;
            std::vector<float> opac;
            std::vector<float> dc;
            scales.reserve(cloud->count() * 3);
            opac.reserve(cloud->count());
            dc.reserve(cloud->count() * 3);
            for (size_t i = 0; i < cloud->count(); ++i)
            {
                scales.push_back(cloud->scales()[i].x);
                scales.push_back(cloud->scales()[i].y);
                scales.push_back(cloud->scales()[i].z);
                opac.push_back(cloud->opacities()[i]);
                const glm::vec4 & c = cloud->shCoefficients()[i * kShCoefficientsPerGaussian];
                dc.push_back(c.x);
                dc.push_back(c.y);
                dc.push_back(c.z);
            }
            printPercentiles("scale (linear)", scales);
            printPercentiles("opacity [0,1]", opac);
            printPercentiles("SH DC (raw f_dc)", dc);
        }

        // --- camera placed exactly as apps/viewer does ---
        const float fy = static_cast<float>(kHeight);
        const float tanHalfFov = (static_cast<float>(kHeight) * 0.5f) / fy;
        const float distance = (bounds.focusRadius / tanHalfFov) * 1.15f;
        const glm::vec3 eye = bounds.focusCenter - distance * glm::vec3(0.0f, 0.0f, -1.0f);

        Camera cam;
        cam.width = kWidth;
        cam.height = kHeight;
        cam.fx = cam.fy = fy;
        cam.cx = static_cast<float>(kWidth) * 0.5f;
        cam.cy = static_cast<float>(kHeight) * 0.5f;
        cam.cameraToWorld = glm::translate(glm::mat4(1.0f), eye);

        std::printf("\n-- camera (as apps/viewer frames it) --\n");
        std::printf("  eye                    (%.3f, %.3f, %.3f)\n", eye.x, eye.y, eye.z);
        std::printf("  distance to focus      %.3f\n", distance);

        ForwardPass pass(device, kWidth, kHeight, SHADER_DIR);
        CameraUniforms uniforms{};
        uniforms.worldToCamera = glm::inverse(cam.cameraToWorld);
        uniforms.cameraPositionWorld = glm::vec4(eye, 1.0f);
        uniforms.fx = cam.fx;
        uniforms.fy = cam.fy;
        uniforms.cx = cam.cx;
        uniforms.cy = cam.cy;
        uniforms.imageWidth = kWidth;
        uniforms.imageHeight = kHeight;
        uniforms.tilesX = (kWidth + kTileSize - 1) / kTileSize;
        uniforms.tilesY = (kHeight + kTileSize - 1) / kTileSize;
        uniforms.gaussianCount = static_cast<uint32_t>(cloud->count());
        uniforms.shDegree = cloud->shDegree();
        // Compensation is the scene's call, as in Renderer.
        uniforms.renderFlags = cloud->antialiased() ? kRenderFlagAntialiased : 0u;

        pass.run(cloud->views(), uniforms, ForwardStage::Project);

        std::vector<ProjectedGaussian> projected(cloud->count());
        pass.projected().download(projected.data(),
                                  projected.size() * sizeof(ProjectedGaussian));

        size_t survived = 0;
        std::vector<float> radii;
        std::vector<float> depths;
        std::vector<float> alphas;
        radii.reserve(cloud->count());
        for (const auto & p : projected)
        {
            if (p.tileCount > 0)
            {
                ++survived;
                radii.push_back(p.radius);
                depths.push_back(p.depth);
                alphas.push_back(p.alpha);
            }
        }

        std::printf("\n-- forward_project output --\n");
        std::printf("  survived               %zu of %zu  (%.1f%%)\n",
                    survived, cloud->count(),
                    100.0 * static_cast<double>(survived) / static_cast<double>(cloud->count()));
        std::printf("  tile instances         %u\n", pass.lastInstanceCount());
        printPercentiles("projected radius (px)", radii);
        printPercentiles("depth", depths);
        printPercentiles("alpha", alphas);

        // The single most diagnostic number: how much of the screen one
        // median-sized Gaussian covers. If this is a large fraction, the scene
        // cannot resolve detail no matter how many Gaussians there are.
        if (!radii.empty())
        {
            std::vector<float> copy = radii;
            const float medianRadius = percentile(copy, 0.50);
            std::printf("\n  median footprint is %.2f px across on a %ux%u image\n",
                        medianRadius * 2.0f, kWidth, kHeight);
            std::printf("  => roughly %.0f such footprints would tile the screen\n",
                        (static_cast<double>(kWidth) * kHeight)
                            / std::max(1.0, 3.14159 * medianRadius * medianRadius));
        }
        // Per-pixel CPU composite of the projected Gaussians (same alpha rules
        // as tile_rasterize.slang): WHERE does each pixel's colour come from?
        // "offscreen" = centre outside the image, "near" = depth < 0.2 (the
        // reference rasterizer's near cull), "subject" = inside focus sphere.
        std::printf("-- per-pixel composite (weight = T * alpha) --\n");
        const glm::vec2 samplePixels[] = {
            {kWidth * 0.5f, kHeight * 0.5f}, {kWidth * 0.25f, kHeight * 0.25f},
            {kWidth * 0.75f, kHeight * 0.25f}, {kWidth * 0.25f, kHeight * 0.75f},
            {kWidth * 0.75f, kHeight * 0.75f}};
        for (const glm::vec2 & px : samplePixels)
        {
            struct Hit
            {
                float depth;
                float alpha;
                size_t index;
            };
            std::vector<Hit> hits;
            for (size_t i = 0; i < projected.size(); ++i)
            {
                const auto & p = projected[i];
                if (p.tileCount == 0)
                {
                    continue;
                }
                const glm::vec2 d = p.center - px;
                if (glm::dot(d, d) > p.radius * p.radius)
                {
                    continue;
                }
                const float e = p.conic.x * d.x * d.x + 2.0f * p.conic.y * d.x * d.y
                                + p.conic.z * d.y * d.y;
                const float a = std::min(p.alpha * std::exp(-0.5f * e), 0.99f);
                if (e < 0.0f || a < 1.0f / 255.0f)
                {
                    continue;
                }
                hits.push_back({p.depth, a, i});
            }
            std::sort(hits.begin(), hits.end(),
                      [](const Hit & l, const Hit & r) { return l.depth < r.depth; });
            float T = 1.0f;
            double wTotal = 0, wDepth = 0, wOff = 0, wNear = 0, wSubject = 0, wHuge = 0;
            for (const Hit & h : hits)
            {
                const auto & p = projected[h.index];
                const double w = T * h.alpha;
                wTotal += w;
                wDepth += w * h.depth;
                const bool off = p.center.x < 0 || p.center.y < 0 || p.center.x >= kWidth
                                 || p.center.y >= kHeight;
                wOff += off ? w : 0;
                wNear += h.depth < 0.2f ? w : 0;
                wHuge += p.radius > static_cast<float>(kHeight) ? w : 0;
                const glm::vec3 pos = glm::vec3(cloud->positions()[h.index]);
                wSubject += glm::length(pos - bounds.focusCenter) <= bounds.focusRadius ? w : 0;
                T *= 1.0f - h.alpha;
                if (T < 1e-4f)
                {
                    break;
                }
            }
            const double inv = wTotal > 0 ? 1.0 / wTotal : 0.0;
            std::printf("  px(%4.0f,%4.0f) hits=%-7zu T=%.3f meanDepth=%-8.3g subject=%.0f%% "
                        "offscreen=%.0f%% near=%.0f%% radius>H=%.0f%%\n",
                        px.x, px.y, hits.size(), T, wDepth * inv, 100 * wSubject * inv,
                        100 * wOff * inv, 100 * wNear * inv, 100 * wHuge * inv);
        }
        std::printf("\n");
    }
    catch (const std::exception & e)
    {
        std::cerr << "scene_diag FAILED: " << e.what() << std::endl;
        result = 1;
    }

    device.shutdown();
    return result;
}
