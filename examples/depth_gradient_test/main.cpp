// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The depth gradient through the Renderer API: backward(image, depth) and its
// BufferView overload, checked against the color-only path, the layout-None
// renderer, and a freshly built renderer after a size or count change.
//
//  1. Layouts 1, 2 and 4, with L = 0.5|I|^2 + 0.5 D^2 (dL/dImage = readImage(),
//     dL/dD = readDepth(), so components 1..3 are the real coverage and front
//     values): the host overload and the view overload agree, and the depth
//     term changes the gradients.
//  2. Layouts 2 and 4: depth components 1..n-1 are ignored (garbage == zeros).
//  3. Color-only: backward(img) == backward(img, nullptr) == backward(img,
//     zeros) on a depth layout; backward(view) == backward(view, BufferView{}).
//  4. Layout None: backward(img, depth) throws; backward(img, nullptr) and
//     backward(view, BufferView{}) work.
//  5. A depth view one float short throws, and the gradients are unchanged.
//  6. After setImageSize (and a new forward) and after setGaussianCount, the
//     depth-gradient gradients equal a freshly built renderer's.
//  7. Depth-only loss (dL/dImage = 0): means, scales, rotations and opacities
//     are non-zero, and the SH gradient is exactly zero.
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
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "sabercat/buffer.h"
#include "sabercat/camera.h"
#include "sabercat/device.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/renderer.h"

using namespace sabercat;

namespace
{

constexpr uint32_t kWidth = 100; // not a multiple of the 16-pixel tile
constexpr uint32_t kHeight = 75;
constexpr uint32_t kResizedWidth = 120;
constexpr uint32_t kResizedHeight = 90;
constexpr uint32_t kCount = 60;
constexpr uint32_t kPrefix = 25;
constexpr uint32_t kCoeffs = 16;

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

// Gaussians scattered in a box in front of the camera (depth 2.8 to 6), with
// fully covered, partial and empty pixels, and every SH band non-zero.
Scene makeScene()
{
    SplitMix rng(0xD3F7A2C1ull);
    Scene s(kCount);
    for (uint32_t i = 0; i < kCount; ++i)
    {
        s.means[i * 4 + 0] = rng.uniform(-1.6f, 1.6f);
        s.means[i * 4 + 1] = rng.uniform(-1.1f, 1.1f);
        s.means[i * 4 + 2] = rng.uniform(-2.0f, 1.2f);
        s.means[i * 4 + 3] = 0.0f;
        for (uint32_t k = 0; k < 3; ++k)
        {
            s.scales[i * 4 + k] = rng.uniform(0.05f, 0.25f);
        }
        s.scales[i * 4 + 3] = 0.0f;
        glm::vec4 q(rng.uniform(-1.0f, 1.0f), rng.uniform(-1.0f, 1.0f), rng.uniform(-1.0f, 1.0f),
                    rng.uniform(-1.0f, 1.0f));
        q = glm::normalize(q);
        std::memcpy(&s.rotations[i * 4], &q, sizeof(q));
        s.opacities[i] = rng.uniform(0.3f, 0.9f);
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

GaussianHostInput sceneInput(const Scene & s, uint32_t count)
{
    GaussianHostInput in;
    in.count = count;
    in.means = s.means.data();
    in.scales = s.scales.data();
    in.rotations = s.rotations.data();
    in.opacities = s.opacities.data();
    in.shCoefficients = s.sh.data();
    in.shDegree = 3;
    return in;
}

// The same view at every size: focal scales with the width.
Camera cameraFor(uint32_t width, uint32_t height)
{
    Camera c;
    c.width = width;
    c.height = height;
    c.fx = c.fy = 1.5f * static_cast<float>(width);
    c.cx = 0.5f * static_cast<float>(width - 1);
    c.cy = 0.5f * static_cast<float>(height - 1);
    c.cameraToWorld = glm::mat4(1.0f);
    c.cameraToWorld[3] = glm::vec4(0.05f, -0.02f, 4.0f, 1.0f);
    return c;
}

std::unique_ptr<Renderer> makeRenderer(uint32_t width, uint32_t height, DepthLayout layout,
                                       const Scene & s, uint32_t count)
{
    auto r = std::make_unique<Renderer>(width, height, layout);
    if (!r->initialize({}, true, SHADER_DIR))
    {
        throw std::runtime_error("Renderer::initialize failed");
    }
    r->setGaussians(sceneInput(s, count));
    r->setCamera(cameraFor(width, height));
    return r;
}

// Worst |a-b| over a stream, relative to the stream's largest magnitude in b.
template <typename T>
double relativeError(const std::vector<T> & a, const std::vector<T> & b)
{
    if (a.size() != b.size())
    {
        return 1e30;
    }
    const float * pa = reinterpret_cast<const float *>(a.data());
    const float * pb = reinterpret_cast<const float *>(b.data());
    const size_t n = a.size() * sizeof(T) / sizeof(float);
    double worst = 0.0, scale = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        worst = std::max(worst, std::fabs(static_cast<double>(pa[i]) - pb[i]));
        scale = std::max(scale, std::fabs(static_cast<double>(pb[i])));
    }
    return scale > 0.0 ? worst / scale : worst;
}

double worstGradientError(const GaussianGradients & a, const GaussianGradients & b)
{
    return std::max({relativeError(a.means, b.means), relativeError(a.scales, b.scales),
                     relativeError(a.rotations, b.rotations),
                     relativeError(a.opacities, b.opacities),
                     relativeError(a.shCoefficients, b.shCoefficients),
                     relativeError(a.screenMeans, b.screenMeans)});
}

// Every gradient stream, flattened with the pad lanes.
std::vector<float> flatten(const GaussianGradients & g)
{
    std::vector<float> out;
    for (const std::vector<glm::vec4> * stream : {&g.means, &g.scales, &g.rotations,
                                                  &g.shCoefficients, &g.screenMeans})
    {
        for (const glm::vec4 & v : *stream)
        {
            out.insert(out.end(), &v.x, &v.x + 4);
        }
    }
    out.insert(out.end(), g.opacities.begin(), g.opacities.end());
    return out;
}

bool identical(const GaussianGradients & a, const GaussianGradients & b)
{
    const std::vector<float> fa = flatten(a), fb = flatten(b);
    return fa.size() == fb.size()
           && std::memcmp(fa.data(), fb.data(), fa.size() * sizeof(float)) == 0;
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

// dL/dDepth for `n` floats per pixel: component 0 is the depth itself, the
// others are zero (scale 0) or garbage of that scale. Only component 0 may be read.
std::vector<float> withComponents(const std::vector<float> & depth, uint32_t n, float scale)
{
    std::vector<float> host = depth;
    SplitMix rng(0x6A7B8C9Dull);
    for (size_t p = 0; p < depth.size() / n; ++p)
    {
        for (uint32_t k = 1; k < n; ++k)
        {
            host[p * n + k] = scale == 0.0f ? 0.0f : rng.uniform(-scale, scale);
        }
    }
    return host;
}

} // namespace

int main()
{
    try
    {
        const Scene scene = makeScene();
        const DepthLayout layouts[3] = {DepthLayout::Depth, DepthLayout::DepthCoverage,
                                        DepthLayout::DepthCoverageFront};
        const uint32_t pixels = kWidth * kHeight;

        // -- 1. Host and view overloads agree; the depth term matters. --
        std::unique_ptr<Renderer> byLayout[3];
        for (int i = 0; i < 3; ++i)
        {
            byLayout[i] = makeRenderer(kWidth, kHeight, layouts[i], scene, kCount);
            Renderer & r = *byLayout[i];
            const uint32_t n = static_cast<uint32_t>(layouts[i]);
            r.forward(RenderMode::Differentiable);
            const std::vector<float> image = r.readImage();
            const std::vector<float> depth = r.readDepth();
            r.backward(image.data(), depth.data());
            const GaussianGradients host = r.readGradients();
            r.backward(image.data());
            const GaussianGradients color = r.readGradients();
            r.backward(r.imageView(), r.depthView());
            const GaussianGradients view = r.readGradients();

            const double viewError = worstGradientError(host, view);
            const double depthEffect = worstGradientError(host, color);
            check(viewError <= 1e-5,
                  "layout " + std::to_string(n) + ": backward(image, depth) == backward(imageView(), "
                      "depthView()), worst relative " + fmt("%.3g", viewError));
            check(depthEffect > 1e-2,
                  "layout " + std::to_string(n) + ": the depth gradient changes the color-only "
                      "gradients, worst relative " + fmt("%.3g", depthEffect));
        }

        // -- 2. Garbage in components 1..n-1 is ignored (layouts 2 and 4). --
        for (int i = 1; i < 3; ++i)
        {
            Renderer & r = *byLayout[i];
            const uint32_t n = static_cast<uint32_t>(layouts[i]);
            r.forward(RenderMode::Differentiable);
            const std::vector<float> image = r.readImage();
            const std::vector<float> depth = r.readDepth();
            r.backward(image.data(), withComponents(depth, n, 0.0f).data());
            const GaussianGradients zeros = r.readGradients();
            r.backward(image.data(), withComponents(depth, n, 1.0e3f).data());
            const GaussianGradients garbage = r.readGradients();
            const double error = worstGradientError(zeros, garbage);
            check(error <= 1e-5,
                  "layout " + std::to_string(n) + ": garbage in components 1.." + std::to_string(n - 1)
                      + " == zeros there, worst relative " + fmt("%.3g", error));
        }

        // -- 3. Color-only equivalences, on the layout-2 renderer. --
        {
            Renderer & r = *byLayout[1];
            r.forward(RenderMode::Differentiable);
            const std::vector<float> image = r.readImage();
            const std::vector<float> zerosDepth(static_cast<size_t>(pixels) * 2, 0.0f);
            r.backward(image.data());
            const GaussianGradients plain = r.readGradients();
            r.backward(image.data(), static_cast<const float *>(nullptr));
            const double nullError = worstGradientError(plain, r.readGradients());
            r.backward(image.data(), zerosDepth.data());
            const double zeroError = worstGradientError(plain, r.readGradients());
            check(nullError <= 1e-5 && zeroError <= 1e-5,
                  "color-only: backward(img) == backward(img, nullptr) (worst " + fmt("%.3g", nullError)
                      + ") == backward(img, zeros) (worst " + fmt("%.3g", zeroError) + ")");

            r.backward(r.imageView());
            const GaussianGradients viewPlain = r.readGradients();
            r.backward(r.imageView(), BufferView{});
            const double viewError = worstGradientError(viewPlain, r.readGradients());
            check(viewError <= 1e-5,
                  "color-only: backward(view) == backward(view, BufferView{}) (worst "
                      + fmt("%.3g", viewError) + ")");

            // -- 5. A depth view one float short is refused, gradients untouched. --
            r.backward(r.imageView(), r.depthView());
            const GaussianGradients before = r.readGradients();
            Buffer shortDepth(r.sabercatDevice(), static_cast<VkDeviceSize>(pixels) * 8 - 4);
            check(throws([&] { r.backward(r.imageView(), shortDepth.view(8)); }),
                  "depth view one float short is rejected");
            check(identical(before, r.readGradients()),
                  "gradients unchanged after the rejected depth view");
        }

        // -- 4. Layout None. --
        {
            auto none = makeRenderer(kWidth, kHeight, DepthLayout::None, scene, kCount);
            none->forward(RenderMode::Differentiable);
            const std::vector<float> image = none->readImage();
            const std::vector<float> anyDepth(pixels, 0.0f);
            check(throws([&] { none->backward(image.data(), anyDepth.data()); }),
                  "layout None: backward(image, depth) throws");
            check(!throws([&] { none->backward(image.data(), static_cast<const float *>(nullptr)); })
                      && !throws([&] { none->backward(none->imageView(), BufferView{}); }),
                  "layout None: backward(image, nullptr) and backward(view, BufferView{}) work");
        }

        // -- 6a. setImageSize, then a new forward, equals a fresh renderer. --
        {
            auto sized = makeRenderer(kWidth, kHeight, DepthLayout::DepthCoverage, scene, kCount);
            sized->setImageSize(kResizedWidth, kResizedHeight);
            sized->setCamera(cameraFor(kResizedWidth, kResizedHeight));
            sized->forward(RenderMode::Differentiable);
            const std::vector<float> image = sized->readImage();
            const std::vector<float> depth = sized->readDepth();
            sized->backward(image.data(), depth.data());
            const GaussianGradients host = sized->readGradients();
            sized->backward(sized->imageView(), sized->depthView());
            const GaussianGradients view = sized->readGradients();

            auto fresh = makeRenderer(kResizedWidth, kResizedHeight, DepthLayout::DepthCoverage,
                                      scene, kCount);
            fresh->forward(RenderMode::Differentiable);
            const std::vector<float> freshImage = fresh->readImage();
            const std::vector<float> freshDepth = fresh->readDepth();
            fresh->backward(freshImage.data(), freshDepth.data());
            const GaussianGradients reference = fresh->readGradients();

            const double hostError = worstGradientError(host, reference);
            const double viewError = worstGradientError(view, reference);
            check(hostError <= 1e-5 && viewError <= 1e-5,
                  "setImageSize " + std::to_string(kResizedWidth) + "x" + std::to_string(kResizedHeight)
                      + ": depth-gradient backward == fresh renderer (host worst "
                      + fmt("%.3g", hostError) + ", view worst " + fmt("%.3g", viewError) + ")");
        }

        // -- 6b. setGaussianCount to a prefix equals a fresh renderer of that prefix. --
        {
            auto counted = makeRenderer(kWidth, kHeight, DepthLayout::DepthCoverage, scene, kCount);
            counted->setGaussianCount(kPrefix);
            counted->forward(RenderMode::Differentiable);
            const std::vector<float> image = counted->readImage();
            const std::vector<float> depth = counted->readDepth();
            counted->backward(image.data(), depth.data());
            const GaussianGradients got = counted->readGradients();

            auto prefix = makeRenderer(kWidth, kHeight, DepthLayout::DepthCoverage, scene, kPrefix);
            prefix->forward(RenderMode::Differentiable);
            const std::vector<float> prefixImage = prefix->readImage();
            const std::vector<float> prefixDepth = prefix->readDepth();
            prefix->backward(prefixImage.data(), prefixDepth.data());
            const GaussianGradients reference = prefix->readGradients();

            const double error = worstGradientError(got, reference);
            check(got.means.size() == kPrefix && error <= 1e-5,
                  "setGaussianCount " + std::to_string(kPrefix) + ": depth-gradient backward == "
                      "fresh renderer of that prefix, worst relative " + fmt("%.3g", error));
        }

        // -- 7. Depth-only loss: geometry gradients non-zero, SH exactly zero. --
        {
            Renderer & r = *byLayout[2];
            r.forward(RenderMode::Differentiable);
            const std::vector<float> depth = r.readDepth();
            const std::vector<float> zeroImage(static_cast<size_t>(pixels) * 4, 0.0f);
            r.backward(zeroImage.data(), depth.data());
            const GaussianGradients g = r.readGradients();
            auto maxAbs = [](const std::vector<glm::vec4> & v)
            {
                double m = 0.0;
                for (const glm::vec4 & x : v)
                {
                    for (int c = 0; c < 4; ++c)
                    {
                        m = std::max(m, std::fabs(static_cast<double>(x[c])));
                    }
                }
                return m;
            };
            double opacityMax = 0.0;
            for (float o : g.opacities)
            {
                opacityMax = std::max(opacityMax, std::fabs(static_cast<double>(o)));
            }
            const bool shZero = std::all_of(g.shCoefficients.begin(), g.shCoefficients.end(),
                                            [](const glm::vec4 & v) { return v == glm::vec4(0.0f); });
            check(maxAbs(g.means) > 0.0 && maxAbs(g.scales) > 0.0 && maxAbs(g.rotations) > 0.0
                      && opacityMax > 0.0,
                  "depth-only: means, scales, rotations and opacities gradients are non-zero (max "
                      + fmt("%.3g", maxAbs(g.means)) + ", " + fmt("%.3g", maxAbs(g.scales)) + ", "
                      + fmt("%.3g", maxAbs(g.rotations)) + ", " + fmt("%.3g", opacityMax) + ")");
            check(shZero, "depth-only: the SH gradient is exactly zero");
        }

        std::printf("\n%s (%d failure%s)\n", g_failures == 0 ? "ALL PASSED" : "FAILED", g_failures,
                    g_failures == 1 ? "" : "s");
        return g_failures == 0 ? 0 : 1;
    }
    catch (const std::exception & e)
    {
        std::printf("[FAIL] exception: %s\n", e.what());
        return 1;
    }
}
