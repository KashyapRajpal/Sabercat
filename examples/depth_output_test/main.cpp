// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The per-pixel depth output of a Differentiable forward (DepthLayout,
// Renderer::readDepth(), Renderer::depthView()), checked against the image,
// the G-buffer and the lighting path. Image 200x150 (not a multiple of the
// 16-pixel tile), two cameras over a deterministic scatter of Gaussians.
//
//  1. Each layout's readDepth() holds width*height*n floats and depthView()
//     reports stride 4n, size width*height*n*4, offset 0.
//  2. Color is bit-identical with DepthLayout::None and DepthCoverageFront.
//  3. Coverage (.y) is bit-identical to the image alpha; front depth (.z) is
//     > 0 exactly where coverage > 0; .w is 0.
//  4. D (.x) is bit-identical across the three layouts.
//  5. With lighting on, D equals the G-buffer's premultiplied depth
//     (normalDepth.w) per pixel.
//  6. The Depth lighting output (depthScale 1, background 0, exposure 1) is
//     D to within 1e-6 relative.
//  7. The throw rules: layout None, no forward, a Viewer forward, and a
//     resize that needs a new forward.
//
// Validation messages are logged as "[validation]"; the run must have none.

#include <vulkan/vulkan.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include "sabercat/camera.h"
#include "sabercat/renderer.h"

using namespace sabercat;

namespace
{

constexpr uint32_t kWidth = 200;
constexpr uint32_t kHeight = 150;
constexpr uint32_t kCount = 300;
constexpr uint32_t kCoeffs = 16;

// A resize target that is not a tile multiple either.
constexpr uint32_t kResizedWidth = 120;
constexpr uint32_t kResizedHeight = 90;

int g_failures = 0;

void check(bool ok, const std::string & what)
{
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok)
    {
        ++g_failures;
    }
}

std::string fmt(const char * format, double value)
{
    char buf[200];
    std::snprintf(buf, sizeof(buf), format, value);
    return buf;
}

// Deterministic 64-bit generator (SplitMix64), so the scene is the same on
// every machine and every run.
class SplitMix
{
  public:
    explicit SplitMix(uint64_t seed) : m_state(seed) {}

    uint64_t next()
    {
        uint64_t z = (m_state += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    float uniform(float lo, float hi)
    {
        return lo + (hi - lo) * static_cast<float>((next() >> 40) * (1.0 / 16777216.0));
    }

  private:
    uint64_t m_state;
};

// Host arrays at the canonical strides.
struct Scene
{
    std::vector<float> means, scales, rotations, opacities, sh;

    explicit Scene(uint32_t n)
        : means(n * 4), scales(n * 4), rotations(n * 4), opacities(n), sh(n * kCoeffs * 4)
    {
    }
};

// Gaussians scattered in a box in front of the first camera (depth 2.8 to 6):
// overlapping enough that many pixels are fully covered, with the box edges
// and the image corners left empty, and every SH band non-zero.
Scene makeScene()
{
    SplitMix rng(0x5ABE12CA7ull);
    Scene s(kCount);
    for (uint32_t i = 0; i < kCount; ++i)
    {
        s.means[i * 4 + 0] = rng.uniform(-1.6f, 1.6f);
        s.means[i * 4 + 1] = rng.uniform(-1.1f, 1.1f);
        s.means[i * 4 + 2] = rng.uniform(-2.0f, 1.2f);
        s.means[i * 4 + 3] = 0.0f;
        for (uint32_t k = 0; k < 3; ++k)
        {
            s.scales[i * 4 + k] = rng.uniform(0.04f, 0.22f);
        }
        s.scales[i * 4 + 3] = 0.0f;
        glm::vec4 q(rng.uniform(-1.0f, 1.0f), rng.uniform(-1.0f, 1.0f), rng.uniform(-1.0f, 1.0f),
                    rng.uniform(-1.0f, 1.0f));
        q = glm::normalize(q);
        std::memcpy(&s.rotations[i * 4], &q, sizeof(q));
        s.opacities[i] = rng.uniform(0.2f, 0.9f);
        for (uint32_t k = 0; k < kCoeffs; ++k)
        {
            for (uint32_t c = 0; c < 3; ++c)
            {
                const float amplitude = k == 0 ? 0.5f : 0.08f;
                s.sh[(i * kCoeffs + k) * 4 + c] = rng.uniform(-amplitude, amplitude);
            }
            s.sh[(i * kCoeffs + k) * 4 + 3] = 0.0f;
        }
    }
    return s;
}

GaussianHostInput sceneInput(const Scene & s)
{
    GaussianHostInput in;
    in.count = kCount;
    in.means = s.means.data();
    in.scales = s.scales.data();
    in.rotations = s.rotations.data();
    in.opacities = s.opacities.data();
    in.shCoefficients = s.sh.data();
    in.shDegree = 3;
    return in;
}

// Principal point at the center of the pixel grid, as in the other examples.
Camera makeCamera(uint32_t width, uint32_t height, float focal, const glm::mat4 & cameraToWorld)
{
    Camera c;
    c.width = width;
    c.height = height;
    c.fx = c.fy = focal;
    c.cx = 0.5f * static_cast<float>(width - 1);
    c.cy = 0.5f * static_cast<float>(height - 1);
    c.cameraToWorld = cameraToWorld;
    return c;
}

Camera firstCamera()
{
    glm::mat4 pose(1.0f);
    pose[3] = glm::vec4(0.0f, 0.0f, 4.0f, 1.0f);
    return makeCamera(kWidth, kHeight, 150.0f, pose);
}

// Turned about the vertical axis and moved sideways.
Camera secondCamera()
{
    glm::mat4 pose = glm::rotate(glm::mat4(1.0f), 0.35f, glm::vec3(0.0f, 1.0f, 0.0f));
    pose[3] = glm::vec4(0.9f, 0.25f, 3.6f, 1.0f);
    return makeCamera(kWidth, kHeight, 140.0f, pose);
}

// The same view at another image size: focal and principal point scale with it.
Camera resized(const Camera & c, uint32_t width, uint32_t height)
{
    const float scale = static_cast<float>(width) / static_cast<float>(c.width);
    return makeCamera(width, height, c.fx * scale, c.cameraToWorld);
}

// Distance in representable floats: 0 when equal (and for +0 vs -0), huge
// when either is NaN.
int64_t ulpDistance(float a, float b)
{
    if (a == b)
    {
        return 0;
    }
    if (std::isnan(a) || std::isnan(b))
    {
        return INT64_MAX;
    }
    int32_t ia = 0;
    int32_t ib = 0;
    std::memcpy(&ia, &a, sizeof(float));
    std::memcpy(&ib, &b, sizeof(float));
    if (ia < 0)
    {
        ia = INT32_MIN - ia;
    }
    if (ib < 0)
    {
        ib = INT32_MIN - ib;
    }
    return std::llabs(static_cast<int64_t>(ia) - static_cast<int64_t>(ib));
}

bool sameBits(float a, float b)
{
    return std::memcmp(&a, &b, sizeof(float)) == 0;
}

bool throws(const std::function<void()> & fn)
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

bool viewMatches(const BufferView & view, uint32_t floatsPerPixel)
{
    return view.valid() && view.offset == 0 && view.stride == 4 * floatsPerPixel
           && view.size == static_cast<VkDeviceSize>(kWidth) * kHeight * floatsPerPixel * 4;
}

} // namespace

int main()
{
    try
    {
        const Scene scene = makeScene();
        const Camera cameras[2] = {firstCamera(), secondCamera()};

        // One renderer per layout. None is the reference: nothing written.
        Renderer none(kWidth, kHeight, DepthLayout::None);
        Renderer depth1(kWidth, kHeight, DepthLayout::Depth);
        Renderer depth2(kWidth, kHeight, DepthLayout::DepthCoverage);
        Renderer depth4(kWidth, kHeight, DepthLayout::DepthCoverageFront);
        Renderer * renderers[] = {&none, &depth1, &depth2, &depth4};
        for (Renderer * r : renderers)
        {
            if (!r->initialize({}, true, SHADER_DIR))
            {
                std::printf("[FAIL] renderer initialize\n");
                return 1;
            }
            r->setGaussians(sceneInput(scene));
        }
        check(depth2.depthLayout() == DepthLayout::DepthCoverage
                  && none.depthLayout() == DepthLayout::None,
              "depthLayout() reports the layout the renderer was created with");

        // 7a. Before any forward.
        check(throws([&] { depth2.readDepth(); }), "readDepth() throws before any forward");
        check(throws([&] { depth2.depthView(); }), "depthView() throws before any forward");

        const uint32_t pixels = kWidth * kHeight;
        for (uint32_t ci = 0; ci < 2; ++ci)
        {
            const std::string cam = "camera " + std::to_string(ci + 1) + ": ";
            for (Renderer * r : renderers)
            {
                r->setCamera(cameras[ci]);
                r->forward(RenderMode::Differentiable);
            }

            // 1. Sizes and views per layout.
            check(depth1.readDepth().size() == pixels * 1 && viewMatches(depth1.depthView(), 1),
                  cam + "layout 1: readDepth() size and depthView() stride/size/offset");
            check(depth2.readDepth().size() == pixels * 2 && viewMatches(depth2.depthView(), 2),
                  cam + "layout 2: readDepth() size and depthView() stride/size/offset");
            check(depth4.readDepth().size() == pixels * 4 && viewMatches(depth4.depthView(), 4),
                  cam + "layout 4: readDepth() size and depthView() stride/size/offset");

            // 2. Color is untouched by the depth output.
            const std::vector<float> color = none.readImage();
            const std::vector<float> color4 = depth4.readImage();
            check(color.size() == color4.size()
                      && std::memcmp(color.data(), color4.data(), color.size() * sizeof(float))
                             == 0,
                  cam + "color bit-identical with DepthLayout::None");

            // 3. Coverage, front depth and the zero word of layout 4.
            const std::vector<float> depth4Data = depth4.readDepth();
            const std::vector<float> image = depth4.readImage();
            size_t full = 0, partial = 0, empty = 0;
            size_t coverageMismatch = 0, frontMismatch = 0, zeroMismatch = 0;
            for (uint32_t p = 0; p < pixels; ++p)
            {
                const float * d = &depth4Data[4 * static_cast<size_t>(p)];
                const float alpha = image[4 * static_cast<size_t>(p) + 3];
                if (!sameBits(d[1], alpha))
                {
                    ++coverageMismatch;
                }
                if ((d[1] > 0.0f) != (d[2] > 0.0f) || (d[1] == 0.0f && d[2] != 0.0f))
                {
                    ++frontMismatch;
                }
                if (d[3] != 0.0f)
                {
                    ++zeroMismatch;
                }
                if (alpha >= 0.99f)
                {
                    ++full;
                }
                else if (alpha > 0.0f)
                {
                    ++partial;
                }
                else
                {
                    ++empty;
                }
            }
            check(full > 0 && partial > 0 && empty > 0,
                  cam + fmt("scene has fully covered, partial and empty pixels (empty %.0f)",
                            static_cast<double>(empty)));
            // Guards checks 4-6 against passing on an all-zero D: D must be positive
            // exactly where the image has coverage, and that set must not be empty.
            size_t depthPositive = 0, covered = 0;
            for (uint32_t p = 0; p < pixels; ++p)
            {
                depthPositive += depth4Data[4 * static_cast<size_t>(p)] > 0.0f ? 1 : 0;
                covered += image[4 * static_cast<size_t>(p) + 3] > 0.0f ? 1 : 0;
            }
            check(depthPositive > 0 && depthPositive == covered,
                  cam + "pixels with D > 0 equal pixels with alpha > 0: "
                      + std::to_string(depthPositive) + " vs " + std::to_string(covered));
            check(coverageMismatch == 0,
                  cam + fmt("coverage bit-identical to the image alpha (%.0f pixels differ)",
                            static_cast<double>(coverageMismatch)));
            check(frontMismatch == 0,
                  cam + fmt("front depth > 0 exactly where coverage > 0 (%.0f pixels differ)",
                            static_cast<double>(frontMismatch)));
            check(zeroMismatch == 0,
                  cam + fmt("the fourth component is 0 (%.0f pixels differ)",
                            static_cast<double>(zeroMismatch)));

            // 4. D is the same in every layout.
            const std::vector<float> d1 = depth1.readDepth();
            const std::vector<float> d2 = depth2.readDepth();
            size_t dMismatch = 0;
            for (uint32_t p = 0; p < pixels; ++p)
            {
                const float value = depth4Data[4 * static_cast<size_t>(p)];
                if (!sameBits(d1[p], value) || !sameBits(d2[2 * static_cast<size_t>(p)], value))
                {
                    ++dMismatch;
                }
            }
            check(dMismatch == 0,
                  cam + fmt("D bit-identical across layouts 1, 2 and 4 (%.0f pixels differ)",
                            static_cast<double>(dMismatch)));

            // 5. With lighting on, D is the G-buffer's premultiplied depth.
            depth4.setLightingEnabled(true);
            depth4.forward(RenderMode::Differentiable);
            const std::vector<float> lightDepth = depth4.readDepth();
            const Renderer::GBufferData gbuffer = depth4.readGBuffer();
            size_t gDiffering = 0;
            int64_t worstUlp = 0;
            for (uint32_t p = 0; p < pixels; ++p)
            {
                const float value = lightDepth[4 * static_cast<size_t>(p)];
                const int64_t ulps = ulpDistance(value, gbuffer.normalDepth[p].w);
                if (ulps != 0)
                {
                    ++gDiffering;
                }
                worstUlp = std::max(worstUlp, ulps);
            }
            check(gDiffering == 0,
                  cam + "G-buffer normalDepth.w == D: " + std::to_string(gDiffering) + " of "
                      + std::to_string(pixels) + " pixels differ, worst "
                      + std::to_string(worstUlp) + " ulp");

            // 6. The Depth lighting output is D (depthScale 1, background 0,
            // exposure 1).
            LightingSettings settings;
            settings.output = LightingOutput::Depth;
            settings.depthScale = 1.0f;
            settings.background = glm::vec3(0.0f);
            settings.exposure = 1.0f;
            depth4.setLightingSettings(settings);
            depth4.forward(RenderMode::Differentiable);
            const std::vector<float> lit = depth4.readLitImage();
            const std::vector<float> depthForLit = depth4.readDepth();
            double worstRelative = 0.0;
            size_t litOutside = 0;
            for (uint32_t p = 0; p < pixels; ++p)
            {
                const double value = depthForLit[4 * static_cast<size_t>(p)];
                const double lightValue = lit[4 * static_cast<size_t>(p)];
                const double error = std::fabs(lightValue - value);
                const double bound = 1e-6 * std::max(std::fabs(value), 1e-30) + 1e-30;
                if (error > bound)
                {
                    ++litOutside;
                }
                worstRelative = std::max(worstRelative, error / std::max(std::fabs(value), 1e-30));
            }
            check(litOutside == 0,
                  cam + fmt("Depth lighting output == D within 1e-6 relative (worst %.3g)",
                            worstRelative));

            settings = LightingSettings{};
            depth4.setLightingSettings(settings);
            depth4.setLightingEnabled(false);
        }

        // 7b-d. Throw rules after the forwards above.
        check(throws([&] { none.readDepth(); }) && throws([&] { none.depthView(); }),
              "layout None: readDepth() and depthView() throw after a forward");

        depth2.forward(RenderMode::Viewer);
        check(throws([&] { depth2.readDepth(); }) && throws([&] { depth2.depthView(); }),
              "readDepth() and depthView() throw after a Viewer forward");
        depth2.setCamera(cameras[0]);
        depth2.forward(RenderMode::Differentiable);
        const BufferView before = depth2.depthView();
        check(depth2.readDepth().size() == pixels * 2,
              "readDepth() valid again after a Differentiable forward");

        depth2.setImageSize(kResizedWidth, kResizedHeight);
        check(throws([&] { depth2.readDepth(); }),
              "readDepth() throws after setImageSize, before a new forward");
        depth2.setCamera(resized(cameras[0], kResizedWidth, kResizedHeight));
        depth2.forward(RenderMode::Differentiable);
        const uint32_t resizedPixels = kResizedWidth * kResizedHeight;
        const BufferView after = depth2.depthView();
        check(depth2.readDepth().size() == static_cast<size_t>(resizedPixels) * 2
                  && after.stride == 8
                  && after.size == static_cast<VkDeviceSize>(resizedPixels) * 8
                  && after.offset == 0 && after.buffer != before.buffer,
              "after setImageSize and a new forward: size w*h*2 and a new depthView handle");

        for (Renderer * r : renderers)
        {
            r->shutdown();
        }
    }
    catch (const std::exception & e)
    {
        std::printf("[FAIL] exception: %s\n", e.what());
        return 1;
    }

    std::printf("\n%s (%d failure%s)\n", g_failures == 0 ? "ALL PASSED" : "FAILED", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}
