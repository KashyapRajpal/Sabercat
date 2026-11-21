// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Renderer::renderDebugView() and readDebugView(), the debug views rendered
// headless, checked against the forward outputs they are derived from.
//
//  1. The throw rules: no scene, no forward, a scene change, setGaussianCount,
//     clearScene, setImageSize, non-positive or NaN scales and an unknown view
//     all throw; a new forward makes the views available again.
//  2. Deterministic: for each RenderMode every view is bit-identical across
//     calls, the forward's own outputs (readImage(), readDepth()) are unchanged
//     by the views, and a second forward reproduces them. BufferView reports
//     offset 0, stride 16, size width*height*16 and a stable handle.
//  3. Final is the forward color image, bit for bit, in both modes.
//  4. Alpha is the image alpha in grayscale. Depth is the front depth over the
//     depth scale, clamped to [0,1] (abs tolerance 1e-6), also with custom
//     scales; the Viewer and Differentiable Depth views are bit-identical.
//  5. TileDensity is the instance count per tile over the density scale: one
//     value per kTileSize tile, opaque, and saturated with a density scale of 1.
//  6. SHDegree is the blend-weighted effective SH degree. At max degree 0 it is
//     the degree-0 color times coverage (abs tolerance 2e-4); at degree 3 it
//     differs; a scene with only DC coefficients renders like max degree 0
//     (abs tolerance 1e-5). The forward uses the renderer's max degree, not the
//     scene's shDegree metadata.
//  7. A resize invalidates the views until a forward, and the views follow the
//     new size both ways (96x40, not tile multiples, and back).
//  8. A DepthLayout::None renderer gives the same Depth view, and its Final
//     view is its readImage().
//  9. A renderer on a borrowed device gives the same five views.
// 10. Interleaving the views keeps each result bit-identical (the buffer is reused).
//
// Validation messages are logged as "[validation]"; the run must have none.

#include <vulkan/vulkan.h>

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "sabercat/camera.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/render_config.h"
#include "sabercat/renderer.h"

using namespace sabercat;

namespace
{

constexpr uint32_t kWidth = 64;
constexpr uint32_t kHeight = 48;
constexpr uint32_t kCount = 12;
constexpr uint32_t kCoeffs = 16;
constexpr float kFocal = 52.0f;

// Indexed by static_cast<size_t>(DebugView): Final = 0 ... SHDegree = 4.
constexpr size_t kViewCount = 5;
constexpr DebugView kViews[kViewCount] = {DebugView::Final, DebugView::Alpha, DebugView::Depth,
                                          DebugView::TileDensity, DebugView::SHDegree};
constexpr size_t kPixels = static_cast<size_t>(kWidth) * kHeight;

// The SHDegree color of degree 0 (kSHDegreeStops[0] in debug_visualize.slang).
constexpr float kDegree0Color[3] = {0.15f, 0.25f, 0.95f};

int g_failures = 0;

void check(bool ok, const std::string & what)
{
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok)
    {
        ++g_failures;
    }
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

bool identical(const std::vector<float> & a, const std::vector<float> & b)
{
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// FNV-1a over the bytes of the floats.
uint64_t fnv1a(const std::vector<float> & v)
{
    uint64_t h = 1469598103934665603ull;
    const unsigned char * p = reinterpret_cast<const unsigned char *>(v.data());
    for (size_t i = 0, n = v.size() * sizeof(float); i < n; ++i)
    {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

std::string num(double v)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.3g", v);
    return buf;
}

DebugViewScales makeScales(float depth, float density)
{
    DebugViewScales s;
    s.depth = depth;
    s.density = density;
    return s;
}

// Host arrays at the canonical strides.
struct Scene
{
    std::vector<float> means, scales, rotations, opacities, sh;

    explicit Scene(uint32_t n)
        : means(n * 4), scales(n * 4), rotations(n * 4), opacities(n), sh(n * kCoeffs * 4)
    {
    }
};

// Deterministic, all-visible, overlapping Gaussians with every SH band
// non-zero (same construction as image_view_test).
Scene makeScene()
{
    Scene s(kCount);
    for (uint32_t i = 0; i < kCount; ++i)
    {
        const float t = static_cast<float>(i);
        s.means[i * 4 + 0] = -0.9f + 0.16f * t;
        s.means[i * 4 + 1] = 0.35f * std::sin(1.7f * t);
        s.means[i * 4 + 2] = -0.4f * std::cos(0.9f * t);
        s.scales[i * 4 + 0] = 0.10f + 0.02f * static_cast<float>(i % 3);
        s.scales[i * 4 + 1] = 0.07f + 0.015f * static_cast<float>(i % 4);
        s.scales[i * 4 + 2] = 0.09f;
        glm::vec4 q(0.1f * t, 0.3f, -0.2f + 0.05f * t, 1.0f);
        q = glm::normalize(q);
        std::memcpy(&s.rotations[i * 4], &q, sizeof(q));
        s.opacities[i] = 0.35f + 0.05f * static_cast<float>(i % 9);
        for (uint32_t k = 0; k < kCoeffs; ++k)
        {
            for (uint32_t c = 0; c < 3; ++c)
            {
                const float base = k == 0 ? 0.6f : 0.08f;
                s.sh[(i * kCoeffs + k) * 4 + c] =
                    base * std::sin(0.37f * t + 1.3f * static_cast<float>(k) + 2.1f * c);
            }
        }
    }
    return s;
}

GaussianHostInput sceneInput(const Scene & s, uint32_t shDegree)
{
    GaussianHostInput in;
    in.count = kCount;
    in.means = s.means.data();
    in.scales = s.scales.data();
    in.rotations = s.rotations.data();
    in.opacities = s.opacities.data();
    in.shCoefficients = s.sh.data();
    in.shDegree = shDegree;
    return in;
}

// Principal point at the center of the pixel grid. The focal length follows
// the width, as in depth_output_test's resized().
Camera makeCamera(uint32_t width, uint32_t height)
{
    const float scale = static_cast<float>(width) / static_cast<float>(kWidth);
    Camera c;
    c.width = width;
    c.height = height;
    c.fx = c.fy = kFocal * scale;
    c.cx = 0.5f * static_cast<float>(width - 1);
    c.cy = 0.5f * static_cast<float>(height - 1);
    c.cameraToWorld = glm::mat4(1.0f);
    c.cameraToWorld[3] = glm::vec4(0.05f, -0.02f, 3.0f, 1.0f);
    return c;
}

// The outputs of one forward, and every view over it.
struct Capture
{
    std::vector<float> image;
    std::vector<float> depth; // Differentiable only
    std::vector<float> views[kViewCount];
};

// The buffer handle is shared by every view of one size: the first call sets
// it and every later call must match.
bool debugBufferOk(const BufferView & bv, VkBuffer & handle, size_t pixels)
{
    if (!bv.valid() || bv.offset != 0 || bv.stride != 16
        || bv.size != static_cast<VkDeviceSize>(pixels) * 16)
    {
        return false;
    }
    if (handle == VK_NULL_HANDLE)
    {
        handle = bv.buffer;
    }
    return bv.buffer == handle;
}

// Runs a forward in `mode`, then every view twice, checking determinism,
// the BufferView contract and that the forward outputs are unchanged.
Capture capture(Renderer & r, RenderMode mode, VkBuffer & handle, const std::string & label)
{
    Capture c;
    const bool diff = mode == RenderMode::Differentiable;
    r.forward(mode);
    c.image = r.readImage();
    if (diff)
    {
        c.depth = r.readDepth();
    }
    const uint64_t hashImage = fnv1a(c.image);
    const uint64_t hashDepth = diff ? fnv1a(c.depth) : 0;

    bool bufferOk = true, deterministic = true;
    for (DebugView v : kViews)
    {
        const size_t slot = static_cast<size_t>(v);
        bufferOk &= debugBufferOk(r.renderDebugView(v), handle, kPixels);
        bufferOk &= debugBufferOk(r.renderDebugView(v), handle, kPixels);
        const std::vector<float> first = r.readDebugView(v);
        const std::vector<float> second = r.readDebugView(v);
        deterministic &= identical(first, second) && first.size() == kPixels * 4;
        c.views[slot] = first;
    }
    check(bufferOk, label + "every view: valid, offset 0, stride 16, size w*h*16, one stable handle");
    check(deterministic, label + "every view is bit-identical across calls");

    bool outputsKept = fnv1a(r.readImage()) == hashImage;
    if (diff)
    {
        outputsKept &= fnv1a(r.readDepth()) == hashDepth;
    }
    check(outputsKept, label + "the views leave readImage()/readDepth() unchanged");

    r.forward(mode);
    bool repeat = fnv1a(r.readImage()) == hashImage;
    if (diff)
    {
        repeat &= fnv1a(r.readDepth()) == hashDepth;
    }
    check(repeat, label + "a second forward() reproduces readImage()/readDepth()");
    return c;
}

// Alpha view: every RGB channel is the image alpha exactly, alpha channel 1.
bool alphaViewOk(const std::vector<float> & view, const std::vector<float> & image, size_t pixels)
{
    if (view.size() != pixels * 4 || image.size() != pixels * 4)
    {
        return false;
    }
    for (size_t p = 0; p < pixels; ++p)
    {
        const float a = image[p * 4 + 3];
        if (view[p * 4] != a || view[p * 4 + 1] != a || view[p * 4 + 2] != a
            || view[p * 4 + 3] != 1.0f)
        {
            return false;
        }
    }
    return true;
}

// Depth view against clamp(front / max(scale, 1e-4), 0, 1), where front is
// the front depth of readDepth() (layout 4: [D, coverage, front, 0]).
// Returns the worst abs error over RGB; alpha must be 1.
double depthViewError(const std::vector<float> & view, const std::vector<float> & depth,
                      float scale, bool & alphaOne)
{
    double worst = 0.0;
    alphaOne = true;
    for (size_t p = 0; p < kPixels; ++p)
    {
        const float expected =
            std::clamp(depth[p * 4 + 2] / std::max(scale, 1e-4f), 0.0f, 1.0f);
        for (size_t c = 0; c < 3; ++c)
        {
            worst = std::max(worst, std::fabs(static_cast<double>(view[p * 4 + c]) - expected));
        }
        alphaOne &= view[p * 4 + 3] == 1.0f;
    }
    return worst;
}

// TileDensity: opaque; red in [0,1]; blue = 1 - red and green = 0.2 (1 - red)
// to 1e-6; every pixel equals its tile's top-left pixel exactly. `worst` is
// the largest blue/green deviation.
bool densityViewOk(const std::vector<float> & v, uint32_t width, uint32_t height, double & worst)
{
    worst = 0.0;
    bool ok = true;
    for (uint32_t y = 0; y < height; ++y)
    {
        for (uint32_t x = 0; x < width; ++x)
        {
            const size_t p = (static_cast<size_t>(y) * width + x) * 4;
            const uint32_t tileY = y / kTileSize * kTileSize;
            const uint32_t tileX = x / kTileSize * kTileSize;
            const size_t corner = (static_cast<size_t>(tileY) * width + tileX) * 4;
            const float red = v[p];
            const double eBlue = std::fabs(static_cast<double>(v[p + 2]) - (1.0 - red));
            const double eGreen = std::fabs(static_cast<double>(v[p + 1]) - 0.2 * (1.0 - red));
            worst = std::max({worst, eBlue, eGreen});
            ok &= v[p + 3] == 1.0f && red >= 0.0f && red <= 1.0f && eBlue <= 1e-6
                  && eGreen <= 1e-6;
            ok &= v[p] == v[corner] && v[p + 1] == v[corner + 1] && v[p + 2] == v[corner + 2];
        }
    }
    return ok;
}

// SHDegree at max degree 0: the degree-0 color times coverage (the image
// alpha). Returns the worst abs error over RGB; `exact` is set when uncovered
// pixels are exactly black and alpha is 1 everywhere.
double degree0Error(const std::vector<float> & view, const std::vector<float> & image, bool & exact)
{
    double worst = 0.0;
    exact = true;
    for (size_t p = 0; p < kPixels; ++p)
    {
        const float a = image[p * 4 + 3];
        for (size_t c = 0; c < 3; ++c)
        {
            const float expected = kDegree0Color[c] * a;
            worst = std::max(worst, std::fabs(static_cast<double>(view[p * 4 + c]) - expected));
            if (a == 0.0f)
            {
                exact &= view[p * 4 + c] == 0.0f;
            }
        }
        exact &= view[p * 4 + 3] == 1.0f;
    }
    return worst;
}

// The largest abs RGB difference between two views of the same size.
double maxRgbDifference(const std::vector<float> & a, const std::vector<float> & b)
{
    double worst = 0.0;
    for (size_t p = 0; p < a.size() / 4; ++p)
    {
        for (size_t c = 0; c < 3; ++c)
        {
            worst = std::max(worst, std::fabs(static_cast<double>(a[p * 4 + c]) - b[p * 4 + c]));
        }
    }
    return worst;
}

// The checks that need only the image, for one capture.
void checkImageViews(const Capture & c, const std::string & label)
{
    check(identical(c.views[0], c.image), label + "Final view bit-identical to readImage()");
    check(alphaViewOk(c.views[1], c.image, kPixels),
          label + "Alpha view: RGB = image alpha, alpha = 1");

    double worst = 0.0;
    check(densityViewOk(c.views[3], kWidth, kHeight, worst),
          label + "TileDensity: opaque, blue = 1 - red, green = 0.2 (1 - red), uniform per tile"
              " (worst " + num(worst) + ", tol 1e-6)");
}

} // namespace

int main()
{
    try
    {
        const Scene scene = makeScene();
        VkBuffer handle = VK_NULL_HANDLE;

        // 1a. No scene loaded, then a scene without any forward.
        {
            Renderer e(kWidth, kHeight);
            if (!e.initialize({}, true, SHADER_DIR))
            {
                std::printf("[FAIL] renderer initialize\n");
                return 1;
            }
            check(throws([&] { e.readDebugView(DebugView::Final); }),
                  "readDebugView() throws with no scene loaded");
            e.setCamera(makeCamera(kWidth, kHeight));
            e.setGaussians(sceneInput(scene, 3));
            check(throws([&] { e.renderDebugView(DebugView::Final); }),
                  "renderDebugView() throws after setGaussians, before any forward");
            e.forward(RenderMode::Differentiable);
            check(!throws([&] { e.readDebugView(DebugView::Depth); }),
                  "readDebugView() works after the first forward");
            e.shutdown();
        }

        Renderer r(kWidth, kHeight, DepthLayout::DepthCoverageFront);
        if (!r.initialize({}, true, SHADER_DIR))
        {
            std::printf("[FAIL] renderer initialize\n");
            return 1;
        }
        r.setCamera(makeCamera(kWidth, kHeight));
        r.setGaussians(sceneInput(scene, 3));

        // 1b. Rejected calls leave a valid forward usable.
        r.forward(RenderMode::Differentiable);
        check(!throws([&] { r.readDebugView(DebugView::Final); }),
              "readDebugView(Final) works after a Differentiable forward");
        check(throws([&] { r.readDebugView(DebugView::Depth, makeScales(0.0f, 64.0f)); }),
              "scales.depth = 0 throws");
        check(throws([&] { r.readDebugView(DebugView::Depth, makeScales(50.0f, 0.0f)); }),
              "scales.density = 0 throws");
        check(throws([&] {
                  r.readDebugView(DebugView::Depth,
                                  makeScales(std::numeric_limits<float>::quiet_NaN(), 64.0f));
              }),
              "NaN scales.depth throws");
        check(throws([&] { r.renderDebugView(static_cast<DebugView>(99)); }),
              "an unknown DebugView throws");
        r.forward(RenderMode::Differentiable);
        check(!throws([&] { r.readDebugView(DebugView::Final); }),
              "readDebugView() valid again after the rejected calls and a forward");

        // 1c. Invalidation by scene changes.
        r.setGaussians(sceneInput(scene, 3));
        check(throws([&] { r.readDebugView(DebugView::Final); }),
              "readDebugView() throws after setGaussians (same scene), before a forward");
        r.forward(RenderMode::Differentiable);
        check(!throws([&] { r.renderDebugView(DebugView::Final); }),
              "renderDebugView() valid again after a new forward");
        r.setGaussianCount(kCount - 1);
        check(throws([&] { r.renderDebugView(DebugView::Alpha); }),
              "renderDebugView() throws after setGaussianCount");
        r.forward(RenderMode::Differentiable);
        check(!throws([&] { r.renderDebugView(DebugView::Alpha); }),
              "renderDebugView() valid again after a forward at the new count");
        r.clearScene();
        check(throws([&] { r.readDebugView(DebugView::Final); })
                  && throws([&] { r.readImage(); }),
              "readDebugView() and readImage() throw after clearScene()");
        r.setGaussians(sceneInput(scene, 3));
        r.forward(RenderMode::Differentiable);
        check(!throws([&] { r.readDebugView(DebugView::Final); }),
              "readDebugView() valid again after a new scene and forward");

        // 2 and 3. Every view, deterministic, in both modes.
        const Capture viewer = capture(r, RenderMode::Viewer, handle, "[Viewer] ");
        const Capture diff = capture(r, RenderMode::Differentiable, handle, "[Differentiable] ");

        // Degeneracy guard: most pixels are covered and have depth, so the
        // comparisons below are not vacuous.
        {
            size_t covered = 0, front = 0;
            for (size_t p = 0; p < kPixels; ++p)
            {
                covered += diff.image[p * 4 + 3] > 0.0f;
                front += diff.depth[p * 4 + 2] > 0.0f;
            }
            check(covered > kPixels / 4 && front > kPixels / 4,
                  "reference frame: substantial coverage and non-zero front depth");
        }

        checkImageViews(viewer, "[Viewer] ");
        checkImageViews(diff, "[Differentiable] ");
        check(identical(viewer.views[0], viewer.image) && identical(diff.views[0], diff.image),
              "Final view bit-identical to readImage() in both modes");

        // 4. Depth view.
        {
            bool alphaOne = true;
            const double worst = depthViewError(diff.views[2], diff.depth, 50.0f, alphaOne);
            check(worst <= 1e-6 && alphaOne,
                  "[Differentiable] Depth view = clamp(front / 50, 0, 1) (worst " + num(worst)
                      + ", tol 1e-6)");
        }
        check(identical(viewer.views[2], diff.views[2]),
              "Depth view of a Viewer forward bit-identical to a Differentiable one");
        for (const DebugViewScales & s : {makeScales(5.0f, 64.0f), makeScales(50.0f, 8.0f)})
        {
            const std::vector<float> view = r.readDebugView(DebugView::Depth, s);
            bool alphaOne = true;
            const double worst = depthViewError(view, diff.depth, s.depth, alphaOne);
            check(worst <= 1e-6 && alphaOne,
                  "Depth view with depth scale " + num(s.depth) + " (worst " + num(worst)
                      + ", tol 1e-6)");
        }

        // 5. TileDensity with a density scale of 1: every non-empty tile saturates.
        {
            const std::vector<float> sat =
                r.readDebugView(DebugView::TileDensity, makeScales(50.0f, 1.0f));
            bool saturated = true, differs = false;
            for (size_t p = 0; p < kPixels; ++p)
            {
                const float def = diff.views[3][p * 4];
                if (def > 0.0f)
                {
                    saturated &= sat[p * 4] == 1.0f;
                }
                differs |= sat[p * 4] != def;
            }
            check(saturated, "TileDensity at density scale 1: every non-empty tile is saturated");
            check(differs, "TileDensity at density scale 1 differs from the default scale");
        }

        // 6. SHDegree: degree 0 is the degree-0 color times coverage.
        r.setMaxSHDegree(0);
        r.forward(RenderMode::Differentiable);
        const std::vector<float> sh0 = r.readDebugView(DebugView::SHDegree);
        check(identical(sh0, r.readDebugView(DebugView::SHDegree)),
              "SHDegree view bit-identical across calls");
        // The degree-0 frame, kept for the DC-only scene check below.
        const std::vector<float> image0 = r.readImage();
        {
            bool exact = false;
            const double worst = degree0Error(sh0, image0, exact);
            check(exact && worst <= 2e-4,
                  "SHDegree at max degree 0 = (0.15, 0.25, 0.95) * coverage (worst " + num(worst)
                      + ", tol 2e-4; uncovered pixels exactly black)");
        }
        r.setMaxSHDegree(3);
        r.forward(RenderMode::Differentiable);
        {
            const double largest = maxRgbDifference(r.readDebugView(DebugView::SHDegree), sh0);
            check(largest > 0.01,
                  "SHDegree at max degree 3 differs from degree 0 (largest difference " + num(largest)
                      + ")");
        }
        {
            // A scene with only DC coefficients, at the default max degree
            // (the forward uses the renderer's max degree, not the scene's
            // shDegree metadata), renders like max degree 0.
            Scene dc = scene;
            for (uint32_t i = 0; i < kCount; ++i)
            {
                for (uint32_t k = 1; k < kCoeffs; ++k)
                {
                    for (uint32_t c = 0; c < 4; ++c)
                    {
                        dc.sh[(i * kCoeffs + k) * 4 + c] = 0.0f;
                    }
                }
            }
            Renderer z(kWidth, kHeight);
            if (!z.initialize({}, true, SHADER_DIR))
            {
                std::printf("[FAIL] renderer initialize\n");
                return 1;
            }
            z.setCamera(makeCamera(kWidth, kHeight));
            z.setGaussians(sceneInput(dc, 0));
            z.forward(RenderMode::Differentiable);
            const std::vector<float> dcImage = z.readImage();
            check(identical(z.readDebugView(DebugView::Final), dcImage),
                  "DC-only scene: Final view bit-identical to readImage()");
            double worst = 0.0;
            for (size_t i = 0; i < kPixels * 4; ++i)
            {
                worst = std::max(worst, std::fabs(static_cast<double>(dcImage[i]) - image0[i]));
            }
            check(worst <= 1e-5,
                  "DC-only scene at max degree 3 renders like max degree 0 (worst " + num(worst)
                      + ", tol 1e-5)");
            const std::vector<float> dcSH = z.readDebugView(DebugView::SHDegree);
            check(identical(dcSH, z.readDebugView(DebugView::SHDegree)),
                  "DC-only scene: SHDegree view bit-identical across calls");
            z.shutdown();
        }
        r.setMaxSHDegree(3);

        // 7. Resize: invalidated until a forward; the views follow the size both ways.
        {
            constexpr uint32_t kResizedWidth = 96;
            constexpr uint32_t kResizedHeight = 40;
            constexpr size_t kResizedPixels = static_cast<size_t>(kResizedWidth) * kResizedHeight;

            r.setImageSize(kResizedWidth, kResizedHeight);
            check(throws([&] { r.renderDebugView(DebugView::Final); }),
                  "renderDebugView() throws after setImageSize, before a new forward");
            r.setCamera(makeCamera(kResizedWidth, kResizedHeight));
            r.forward(RenderMode::Differentiable);
            const BufferView bv = r.renderDebugView(DebugView::Alpha);
            check(bv.valid() && bv.offset == 0 && bv.stride == 16
                      && bv.size == static_cast<VkDeviceSize>(kResizedPixels) * 16,
                  "resized (96x40): view size w*h*16 after a forward");
            const std::vector<float> image = r.readImage();
            const std::vector<float> finalView = r.readDebugView(DebugView::Final);
            check(finalView.size() == kResizedPixels * 4 && identical(finalView, image),
                  "resized: Final view bit-identical to readImage()");
            check(alphaViewOk(r.readDebugView(DebugView::Alpha), image, kResizedPixels),
                  "resized: Alpha view matches the image alpha");

            r.setImageSize(kWidth, kHeight);
            r.setCamera(makeCamera(kWidth, kHeight));
            check(throws([&] { r.readDebugView(DebugView::Depth); }),
                  "readDebugView() throws after setImageSize back to 64x48, before a forward");
            r.forward(RenderMode::Differentiable);
            const BufferView back = r.renderDebugView(DebugView::Final);
            check(back.valid() && back.size == static_cast<VkDeviceSize>(kPixels) * 16,
                  "back at 64x48: view size w*h*16 again (no stale buffer)");
            check(identical(r.readDebugView(DebugView::Final), r.readImage()),
                  "back at 64x48: Final view bit-identical to readImage()");
        }

        // The round trip reproduces the first Differentiable views exactly.
        VkBuffer freshHandle = VK_NULL_HANDLE;
        const Capture base = capture(r, RenderMode::Differentiable, freshHandle, "[after resize] ");
        {
            bool same = true;
            for (size_t v = 0; v < kViewCount; ++v)
            {
                same &= identical(base.views[v], diff.views[v]);
            }
            check(same, "after the resize round trip all five views match the first ones");
        }

        // 8. DepthLayout::None: the same Depth view, and Final is its readImage().
        {
            Renderer none(kWidth, kHeight, DepthLayout::None);
            if (!none.initialize({}, true, SHADER_DIR))
            {
                std::printf("[FAIL] renderer initialize\n");
                return 1;
            }
            none.setCamera(makeCamera(kWidth, kHeight));
            none.setGaussians(sceneInput(scene, 3));
            none.forward(RenderMode::Differentiable);
            std::vector<float> noneDepth;
            const bool noneOk = !throws([&] { noneDepth = none.readDebugView(DebugView::Depth); });
            check(noneOk, "DepthLayout::None: the Depth view is available after a Differentiable forward");
            check(noneOk && identical(noneDepth, diff.views[2]),
                  "DepthLayout::None: Depth view bit-identical to the DepthCoverageFront renderer's");
            check(identical(none.readDebugView(DebugView::Final), none.readImage()),
                  "DepthLayout::None: Final view bit-identical to readImage()");
            none.shutdown();
        }

        // 9. Borrowed device: the same five views as the owning renderer.
        {
            Renderer b(kWidth, kHeight, DepthLayout::DepthCoverageFront);
            b.initializeBorrowed(r.vulkanContext(), r.enabledDeviceFeatures(), SHADER_DIR);
            b.setCamera(makeCamera(kWidth, kHeight));
            b.setGaussians(sceneInput(scene, 3));
            b.forward(RenderMode::Differentiable);
            r.forward(RenderMode::Differentiable);
            bool same = true;
            for (DebugView v : kViews)
            {
                same &= identical(b.readDebugView(v), r.readDebugView(v));
            }
            check(same, "borrowed device: all five views bit-identical to the owning renderer");
            b.shutdown(); // before the owner's device
        }

        // 10. Interleaved views: each result still matches its first one.
        r.forward(RenderMode::Differentiable);
        std::vector<float> reference[kViewCount];
        for (DebugView v : kViews)
        {
            reference[static_cast<size_t>(v)] = r.readDebugView(v);
        }
        const DebugView order[] = {
            DebugView::Final,       DebugView::SHDegree, DebugView::Depth,
            DebugView::Final,       DebugView::Alpha,    DebugView::TileDensity,
            DebugView::SHDegree,    DebugView::Depth,    DebugView::Alpha,
            DebugView::TileDensity, DebugView::Final,    DebugView::Depth,
            DebugView::SHDegree,    DebugView::Alpha,    DebugView::TileDensity,
        };
        bool alternates = true;
        for (DebugView v : order)
        {
            alternates &= identical(r.readDebugView(v), reference[static_cast<size_t>(v)]);
        }
        check(alternates, "interleaved views: each result bit-identical to its first result");

        r.shutdown();
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
