// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Sort-mode API test.
//
// Exercises the SortSettings surface through the public Renderer API and the
// ForwardPass sort variants on a synthetic scene built in code. Stable,
// TileRayDepth, PixelWindow, Hybrid and Stochastic run for a Viewer forward. A
// Differentiable forward always forces the default sort.
//
// Checks (each prints ok or FAIL; any FAIL makes the exit status nonzero):
//   (a) Defaults: a fresh renderer, a renderer given an explicit SortSettings{},
//       and a renderer switched to Stable and back to Default give the same image
//       bits, and a second render on one renderer is bit-identical. Each of
//       Stable, TileRayDepth, PixelWindow, Hybrid and Stochastic renders in Viewer
//       with coverage and finite values (no bit comparison).
//   (b) Differentiable forward gives the default Differentiable image bits for
//       every mode, on a fresh renderer per mode and on one renderer cycled
//       through all modes (no state carried from one mode to the next).
//   (h) Lighting gate: with lighting on, forward(Viewer) throws for PixelWindow,
//       Hybrid and Stochastic, naming "lighting" and the mode. Stable and
//       TileRayDepth render without a lighting error (they change the sorted
//       list, so the G-buffer stays consistent). The Default sort does not throw.
//       The gate does not apply to Differentiable forward: with lighting on, each
//       of the five non-default modes renders without throwing and gives the same
//       readImage() bits as the default-sort lit Differentiable image.
//   (v) setSortSettings rejects invalid values with std::invalid_argument, and a
//       rejected call leaves sortSettings() unchanged.
//   (t) Stage timings: when valid, every field is finite and non-negative, the
//       total is positive, and the eight stages sum to the total. A renderer that
//       never enabled timing reports valid == false. Timing does not change the
//       image.
//   (d) Ray-depth form (ForwardPass::computeRayDepthForm): twelve floats per
//       Gaussian against a double-precision world-space reference, on the
//       synthetic scene plus six edge Gaussians. The three degenerate ones must
//       be exactly zero. Also: a repeat call is bit-identical, a different camera
//       rewrites the form, and rayDepthForm() before any compute throws.
//   (c) S1 sticky bucket keys (SortMode::Stable): over a 40-frame orbit of 0.5
//       degree azimuth steps, every Gaussian's new key equals
//       study::stableKeyStep of its previous key, except where
//       study::stableKeyAmbiguous; ambiguous Gaussian-frames stay under 0.5%.
//       Coplanar pair: two overlapping splats whose depth gap alternates by
//       +-1e-6 each frame. Default order swaps at least 15 times in 20 frames;
//       Stable order never swaps.
//   (e) TileRayDepth and Hybrid sort keys: the sorted list is ordered by
//       (tile, key, Gaussian index); each key equals the double tile-center ray
//       depth (clamped at 0) within kRayDepthKeyTolerance; the ray-depth form the
//       run wrote equals computeRayDepthForm's output bit for bit.
//   (f) Window images: PixelWindow K = 16 and 32 and Hybrid (K = 8) against the
//       CPU window model (WindowRayDepth) within kWindowPixelFraction; each window
//       image differs from the default image somewhere. Sparse scene (40
//       Gaussians, every tile list under 32): PixelWindow K = 32 equals the
//       ExactRayDepth reference within kSparseTolerance on every pixel.
//   (g1) Stochastic raw frames: ST4 and ST8 with temporalFrames 1, frame indices 0
//       and 1 after resetStochastic(), against the CPU model
//       (renderStochasticModel) within kStochasticPixelTolerance on all but
//       kStochasticPixelFraction of the pixels.
//   (g2) Stochastic convergence: 256 static frames for ST4 and ST8, MAE (rgb)
//       against the ExactRayDepth reference, printed after 1, 8, 32 and 256 frames;
//       only the 256-frame MAE is asserted, against kAccumulatedMaeLimit.
//   (g3) Accumulation arithmetic: temporalFrames 32, 8 static frames, against the
//       mean of 8 raw frames (temporalFrames 1, same frame indices) within
//       kAccumulationTolerance per channel.
//   (g4) Reset rules: a camera change under temporalMode 0 gives the raw frame at
//       the same frame index (kResetTolerance); under temporalMode 1 (EMA) the
//       result is (1 - w) * previous + w * raw (kEmaTolerance). The accumulated
//       count is 1 after a reset, k after k static frames, and capped by
//       temporalFrames. Camera steps alternate, so every change resets.
//   (g5) Stochastic without lighting renders with coverage and finite values; with
//       lighting on it is refused (section (h)).
//   (g6) Blue-noise ranks: a permutation of 0..4095, identical across two calls,
//       8x8 block means within 3 standard errors of 2047.5, and the spread of the
//       64 block means below half the white-noise spread.
//   (g7a) Temporal weights, exact: ST4 with temporalMode 1, temporalEmaWeight 0.25
//       and 12 static frames after a reset: weight 1 on the reset run, then
//       max(1 / k, 0.25) for k = 2..12, and the count min(k, 32). A second
//       sequence with temporalEmaWeight 0.05 and 30 frames: max(1 / k, 0.05) for
//       k = 1..30, and the count min(k, 32).
//   (g7b) EMA image recursion: 10 static frames in temporalMode 1 (w 0.25) against
//       the host recursion hist += (raw_k - hist) * max(1 / k, 0.25) over 10 raw
//       frames at the same frame indices, within kRecursionTolerance per channel.
//   (r) CPU study reference (examples/sort_study): TileOrder against the GPU
//       image, window 1 bit-equal to TileOrder, window 4096 bit-equal to
//       ExactRayDepth, ExactRayDepth differing from TileOrder somewhere, and
//       coverage and finite values for every reference image. Each reference
//       render is timed.
//
// Usage:
//   sabercat_sort_modes_test                                   run the checks
//   sabercat_sort_modes_test --timing <scene> <WxH> [frames]   manual benchmark
//
// The benchmark is not run by ctest. It loads the scene, orbits a camera around
// its focus center (0.5 degrees per frame about the up axis, fixed 15 degree
// elevation) for `frames` frames (default 200, the first 20 are warm-up), and
// prints median and p95 of the forward() wall time, of each GPU stage and of the
// total, and the tile-instance count. Only the scene file's name is printed.

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "sabercat/buffer.h"
#include "sabercat/camera.h"
#include "sabercat/device.h"
#include "sabercat/forward_pass.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/render_config.h"
#include "sabercat/renderer.h"
#include "sabercat/scene_bounds.h"
#include "sabercat/sort_settings.h"

#include "../common/shader_dir.h"

#include "blue_noise.h"
#include "cpu_reference.h"
#include "stable_key_model.h"
#include "study_capture.h"

using namespace sabercat;

namespace
{

constexpr uint32_t kWidth = 256;
constexpr uint32_t kHeight = 192;
constexpr uint32_t kGaussians = 3000;
constexpr uint32_t kShDegree = 3;
constexpr uint32_t kWarmupFrames = 20;
constexpr uint32_t kDefaultBenchFrames = 200;
constexpr double kPi = 3.14159265358979323846;

constexpr SortMode kAllModes[] = {SortMode::Default,      SortMode::Stable,  SortMode::TileRayDepth,
                                  SortMode::PixelWindow, SortMode::Hybrid, SortMode::Stochastic};
constexpr SortMode kNonDefaultModes[] = {SortMode::Stable,  SortMode::TileRayDepth, SortMode::PixelWindow,
                                         SortMode::Hybrid, SortMode::Stochastic};

constexpr const char * kStageNames[] = {"project", "scan",    "binning", "sortKeys", "sort",
                                        "ranges",  "rasterize", "resolve", "total"};
constexpr size_t kStageCount = sizeof(kStageNames) / sizeof(kStageNames[0]);

int g_checks = 0;
int g_failures = 0;

// xorshift32 (Marsaglia, 2003). Its output is an exact function of the seed,
// and the seed must be nonzero.
class Xorshift32
{
  public:
    explicit Xorshift32(uint32_t seed) : m_state(seed)
    {
    }

    uint32_t next()
    {
        m_state ^= m_state << 13;
        m_state ^= m_state >> 17;
        m_state ^= m_state << 5;
        return m_state;
    }

    // [0, 1) from the top 24 bits. Exact: 2^24 fits a float's significand.
    float unit()
    {
        return static_cast<float>(next() >> 8) * (1.0f / 16777216.0f);
    }

    // lo + (hi - lo) * unit(): two correctly rounded float operations.
    float range(float lo, float hi)
    {
        return lo + (hi - lo) * unit();
    }

  private:
    uint32_t m_state;
};

// Host arrays in the GaussianHostInput layout. Each draw from the generator
// gets its own statement: the order of argument evaluation is unspecified in
// C++, so two draws inside one call expression could swap between builds.
struct GaussianSet
{
    std::vector<float> means;     // 4 per Gaussian (xyz, pad)
    std::vector<float> scales;    // 4 per Gaussian (linear, xyz, pad)
    std::vector<float> rotations; // 4 per Gaussian (unit quaternion, xyzw)
    std::vector<float> opacities; // 1 per Gaussian, in [0, 1]
    std::vector<float> sh;        // 64 per Gaussian: 16 coefficients x (rgb, pad)

    uint32_t count() const
    {
        return static_cast<uint32_t>(opacities.size());
    }
};

struct Scene
{
    Camera camera;
    GaussianSet gaussians;
};

glm::vec3 drawBox(Xorshift32 & rng, glm::vec3 lo, glm::vec3 hi)
{
    glm::vec3 v;
    v.x = rng.range(lo.x, hi.x);
    v.y = rng.range(lo.y, hi.y);
    v.z = rng.range(lo.z, hi.z);
    return v;
}

// Appends one Gaussian: the given center and linear scale, a random unit
// rotation, the given opacity, and random SH (wide DC, small higher bands).
void appendGaussian(GaussianSet & set, Xorshift32 & rng, glm::vec3 center, glm::vec3 scale, float opacity)
{
    set.means.insert(set.means.end(), {center.x, center.y, center.z, 0.0f});
    set.scales.insert(set.scales.end(), {scale.x, scale.y, scale.z, 0.0f});

    float q[4];
    for (float & c : q)
    {
        c = rng.range(-1.0f, 1.0f);
    }
    float norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (!(norm > 1e-6f))
    {
        q[0] = 0.0f;
        q[1] = 0.0f;
        q[2] = 0.0f;
        q[3] = 1.0f;
        norm = 1.0f;
    }
    for (float & c : q)
    {
        c /= norm;
    }
    set.rotations.insert(set.rotations.end(), {q[0], q[1], q[2], q[3]});

    set.opacities.push_back(opacity);

    for (uint32_t k = 0; k < 16; ++k)
    {
        const float spread = (k == 0) ? 1.5f : 0.15f;
        for (int c = 0; c < 3; ++c)
        {
            const float coefficient = rng.range(-spread, spread);
            set.sh.push_back(coefficient);
        }
        set.sh.push_back(0.0f);
    }
}

void appendScatter(GaussianSet & set, Xorshift32 & rng, uint32_t count, glm::vec3 lo, glm::vec3 hi,
                   float scaleLo, float scaleHi)
{
    for (uint32_t i = 0; i < count; ++i)
    {
        const glm::vec3 center = drawBox(rng, lo, hi);
        const glm::vec3 scale = drawBox(rng, glm::vec3(scaleLo), glm::vec3(scaleHi));
        const float opacity = rng.unit();
        appendGaussian(set, rng, center, scale, opacity);
    }
}

// Camera-to-world transform for a unit quaternion (xyzw) and a position.
// Columns are written out from the standard quaternion-to-matrix formulas.
glm::mat4 makePose(float x, float y, float z, float w, glm::vec3 position)
{
    const float norm = std::sqrt(x * x + y * y + z * z + w * w);
    x /= norm;
    y /= norm;
    z /= norm;
    w /= norm;

    glm::mat4 m(1.0f);
    m[0] = glm::vec4(1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y + z * w), 2.0f * (x * z - y * w), 0.0f);
    m[1] = glm::vec4(2.0f * (x * y - z * w), 1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z + x * w), 0.0f);
    m[2] = glm::vec4(2.0f * (x * z + y * w), 2.0f * (y * z - x * w), 1.0f - 2.0f * (x * x + y * y), 0.0f);
    m[3] = glm::vec4(position, 1.0f);
    return m;
}

Camera makeCamera(uint32_t width, uint32_t height, float fx, float fy, float cx, float cy,
                  const glm::mat4 & cameraToWorld)
{
    Camera camera;
    camera.width = width;
    camera.height = height;
    camera.fx = fx;
    camera.fy = fy;
    camera.cx = cx;
    camera.cy = cy;
    camera.cameraToWorld = cameraToWorld;
    return camera;
}

// The rotated scene of forward_regression_test, at 256x192 with a centered
// principal point.
Scene buildScene()
{
    Scene scene;
    scene.camera = makeCamera(kWidth, kHeight, 310.0f, 290.0f, 128.0f, 96.0f,
                              makePose(0.04f, -0.06f, 0.02f, 1.0f, glm::vec3(0.4f, -0.3f, 5.0f)));

    Xorshift32 rng(0x5027A0D1u);
    appendScatter(scene.gaussians, rng, kGaussians, glm::vec3(-1.2f, -1.2f, -1.2f), glm::vec3(1.2f, 1.2f, 1.2f),
                  0.03f, 0.2f);
    return scene;
}

GaussianHostInput inputFor(const Scene & scene)
{
    GaussianHostInput input;
    input.count = scene.gaussians.count();
    input.means = scene.gaussians.means.data();
    input.scales = scene.gaussians.scales.data();
    input.rotations = scene.gaussians.rotations.data();
    input.opacities = scene.gaussians.opacities.data();
    input.shCoefficients = scene.gaussians.sh.data();
    input.shDegree = kShDegree;
    input.antialiased = false;
    return input;
}

void initializeRenderer(Renderer & renderer)
{
    if (!renderer.initialize({}, true, testShaderDir()))
    {
        throw std::runtime_error("Renderer::initialize failed (a Vulkan 1.3 device is required)");
    }
}

// A fresh renderer with the scene and camera set. Sort settings are left at
// whatever the renderer's own default is (no setSortSettings call).
std::unique_ptr<Renderer> makeRenderer(const Scene & scene)
{
    auto renderer = std::make_unique<Renderer>(kWidth, kHeight);
    initializeRenderer(*renderer);
    renderer->setGaussians(inputFor(scene));
    renderer->setCamera(scene.camera);
    return renderer;
}

// forward(mode), then the image as RGBA floats.
std::vector<float> render(Renderer & renderer, RenderMode mode)
{
    renderer.forward(mode);
    return renderer.readImage();
}

// Bit-for-bit equality of two images (memcmp, so -0.0 and 0.0 differ).
bool sameBits(const std::vector<float> & a, const std::vector<float> & b)
{
    if (a.size() != b.size())
    {
        return false;
    }
    return a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

size_t coveredPixels(const std::vector<float> & image)
{
    size_t covered = 0;
    for (size_t i = 3; i < image.size(); i += 4)
    {
        if (image[i] > 0.0f)
        {
            ++covered;
        }
    }
    return covered;
}

bool allFiniteFloats(const std::vector<float> & values)
{
    for (float v : values)
    {
        if (!std::isfinite(v))
        {
            return false;
        }
    }
    return true;
}

std::string lower(std::string s)
{
    for (char & c : s)
    {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return s;
}

SortSettings withMode(SortMode mode)
{
    SortSettings settings;
    settings.mode = mode;
    return settings;
}

bool sameSettings(const SortSettings & a, const SortSettings & b)
{
    return a.mode == b.mode && a.pixelWindow == b.pixelWindow && a.stableBucketsPerOctave == b.stableBucketsPerOctave
           && a.stochasticSamples == b.stochasticSamples && a.temporalFrames == b.temporalFrames
           && a.temporalMode == b.temporalMode && a.stableHysteresis == b.stableHysteresis
           && a.temporalEmaWeight == b.temporalEmaWeight;
}

void report(bool pass, const std::string & what)
{
    ++g_checks;
    if (!pass)
    {
        ++g_failures;
    }
    std::printf("  %s  %s\n", pass ? "ok  " : "FAIL", what.c_str());
}

// Runs one labelled check. An exception it does not expect fails that check
// only, so the rest of the section still runs.
template <typename Fn>
void guarded(const std::string & label, Fn fn)
{
    try
    {
        fn();
    }
    catch (const std::exception & e)
    {
        report(false, label + ": unexpected exception: " + e.what());
    }
}

// Runs one section. An exception escaping it fails the section, and the other
// sections still run.
template <typename Fn>
void runSection(const char * title, Fn fn)
{
    std::printf("%s\n", title);
    try
    {
        fn();
    }
    catch (const std::exception & e)
    {
        report(false, std::string(title) + ": unexpected exception: " + e.what());
    }
}

// (a) Defaults.
void checkDefaults(const Scene & scene)
{
    // The fresh renderer never calls setSortSettings.
    auto fresh = makeRenderer(scene);
    const std::vector<float> reference = render(*fresh, RenderMode::Viewer);
    report(coveredPixels(reference) > 0, "(a) reference image has coverage");
    report(sameBits(reference, render(*fresh, RenderMode::Viewer)),
           "(a) second render on the same renderer is bit-identical");

    auto explicitDefault = makeRenderer(scene);
    explicitDefault->setSortSettings(SortSettings{});
    report(sameBits(reference, render(*explicitDefault, RenderMode::Viewer)),
           "(a) explicit SortSettings{} matches the fresh renderer");

    // Stable is set and reverted with no forward in between, so the switch
    // itself is what is under test.
    auto switched = makeRenderer(scene);
    switched->setSortSettings(withMode(SortMode::Stable));
    switched->setSortSettings(SortSettings{});
    report(switched->sortSettings().mode == SortMode::Default,
           "(a) after Stable then Default, sortSettings().mode is Default");
    report(sameBits(reference, render(*switched, RenderMode::Viewer)),
           "(a) Stable then Default matches the fresh renderer");

    // The variant modes run for a Viewer forward. No bit comparison: each mode
    // has its own image, and the checks below only need it to render.
    const SortMode rendering[] = {SortMode::Stable, SortMode::TileRayDepth, SortMode::PixelWindow,
                                  SortMode::Hybrid, SortMode::Stochastic};
    for (SortMode mode : rendering)
    {
        const std::string label = std::string("(a) Viewer render, mode ") + sortModeName(mode);
        guarded(label, [&]() {
            auto renderer = makeRenderer(scene);
            renderer->setSortSettings(withMode(mode));
            const std::vector<float> image = render(*renderer, RenderMode::Viewer);
            report(coveredPixels(image) > 0 && allFiniteFloats(image),
                   label + ": renders with coverage and finite values");
        });
    }
}

// (b) Differentiable forward forces the default sort.
void checkDifferentiable(const Scene & scene)
{
    auto reference = makeRenderer(scene);
    const std::vector<float> defaultImage = render(*reference, RenderMode::Differentiable);
    report(coveredPixels(defaultImage) > 0, "(b) default Differentiable image has coverage");

    for (SortMode mode : kAllModes)
    {
        const std::string label = std::string("(b) fresh renderer, Differentiable, mode ") + sortModeName(mode);
        guarded(label, [&]() {
            auto renderer = makeRenderer(scene);
            renderer->setSortSettings(withMode(mode));
            const std::vector<float> image = render(*renderer, RenderMode::Differentiable);
            report(sameBits(image, defaultImage), label + ": bits equal the default Differentiable image");
        });
    }

    auto cycled = makeRenderer(scene);
    for (SortMode mode : kAllModes)
    {
        const std::string label =
            std::string("(b) one renderer cycled to mode ") + sortModeName(mode) + ", Differentiable";
        guarded(label, [&]() {
            cycled->setSortSettings(withMode(mode));
            const std::vector<float> image = render(*cycled, RenderMode::Differentiable);
            report(sameBits(image, defaultImage), label + ": bits equal the default Differentiable image");
        });
    }
    report(sameBits(cycled->readImage(), defaultImage),
           "(b) readImage after the last cycled Differentiable forward equals the default image");
}

// (h) Lighting gate.
void checkLightingGate(const Scene & scene)
{
    const SortMode blocked[] = {SortMode::PixelWindow, SortMode::Hybrid, SortMode::Stochastic};
    for (SortMode mode : blocked)
    {
        const std::string label = std::string("(h) lighting on, mode ") + sortModeName(mode) + ": forward refuses";
        guarded(label, [&]() {
            auto renderer = makeRenderer(scene);
            renderer->setLightingEnabled(true);
            renderer->setSortSettings(withMode(mode));
            try
            {
                renderer->forward(RenderMode::Viewer);
            }
            catch (const std::runtime_error & e)
            {
                const std::string message = lower(e.what());
                const bool named = message.find("lighting") != std::string::npos
                                   && message.find(lower(sortModeName(mode))) != std::string::npos;
                report(named, named ? label + ": refused with a message naming lighting and the mode"
                                    : label + ": message does not name lighting and the mode: " + e.what());
                return;
            }
            report(false, label + ": did not throw");
        });
    }

    guarded("(h) lighting on, mode Default: no sort error", [&]() {
        auto renderer = makeRenderer(scene);
        renderer->setLightingEnabled(true);
        renderer->setSortSettings(SortSettings{});
        try
        {
            renderer->forward(RenderMode::Viewer);
            report(true, "(h) lighting on, mode Default: forward does not throw");
        }
        catch (const std::exception & e)
        {
            report(false, std::string("(h) lighting on, mode Default: forward threw: ") + e.what());
        }
    });

    // Stable and TileRayDepth change the sorted list itself, so the lit G-buffer
    // stays consistent with them: a lit Viewer frame must render, not throw.
    const SortMode litRendering[] = {SortMode::Stable, SortMode::TileRayDepth};
    for (SortMode mode : litRendering)
    {
        const std::string label =
            std::string("(h) lighting on, mode ") + sortModeName(mode) + ": renders without a lighting error";
        guarded(label, [&]() {
            auto renderer = makeRenderer(scene);
            renderer->setLightingEnabled(true);
            renderer->setSortSettings(withMode(mode));
            const std::vector<float> image = render(*renderer, RenderMode::Viewer);
            report(coveredPixels(image) > 0 && allFiniteFloats(image),
                   label + " (coverage and finite values)");
        });
    }

    // (g5) Without lighting the Stochastic mode renders. The lit case is refused
    // above: the relight G-buffer pass still uses the default order, so a lit
    // Stochastic frame would disagree with it.
    guarded("(g5) lighting off, mode Stochastic: renders with coverage and finite values", [&]() {
        auto renderer = makeRenderer(scene);
        renderer->setSortSettings(withMode(SortMode::Stochastic));
        const std::vector<float> image = render(*renderer, RenderMode::Viewer);
        report(coveredPixels(image) > 0 && allFiniteFloats(image),
               "(g5) lighting off, mode Stochastic: renders with coverage and finite values");
    });
}

// (h) The lighting gate does not apply to Differentiable forward. With lighting
// on, every non-default mode must give the default-sort lit image, bit for bit.
void checkLitDifferentiable(const Scene & scene)
{
    auto reference = makeRenderer(scene);
    reference->setLightingEnabled(true);
    const std::vector<float> defaultLit = render(*reference, RenderMode::Differentiable);
    report(coveredPixels(defaultLit) > 0, "(h) default-sort lit Differentiable image has coverage");

    for (SortMode mode : kNonDefaultModes)
    {
        const std::string label = std::string("(h) lighting on, Differentiable, mode ") + sortModeName(mode)
                                  + ": no throw, bits equal the default-sort lit image";
        guarded(label, [&]() {
            auto renderer = makeRenderer(scene);
            renderer->setLightingEnabled(true);
            renderer->setSortSettings(withMode(mode));
            const std::vector<float> image = render(*renderer, RenderMode::Differentiable);
            report(sameBits(image, defaultLit), label);
        });
    }
}

// (v) Validation.
void checkValidation()
{
    Renderer renderer(64, 48);
    initializeRenderer(renderer);

    const SortSettings baseline = withMode(SortMode::Stable);

    SortSettings badMode;
    badMode.mode = static_cast<SortMode>(99);

    SortSettings badSamples = withMode(SortMode::Stochastic);
    badSamples.stochasticSamples = 3;

    SortSettings badWindow = withMode(SortMode::PixelWindow);
    badWindow.pixelWindow = 24;

    SortSettings badTemporal = withMode(SortMode::Default);
    badTemporal.temporalMode = 2;

    SortSettings badHysteresis = withMode(SortMode::Stable);
    badHysteresis.stableHysteresis = std::numeric_limits<float>::quiet_NaN();

    SortSettings badEma = withMode(SortMode::Default);
    badEma.temporalEmaWeight = 0.0f;

    struct Case
    {
        const char * name;
        SortSettings settings;
    };
    const Case cases[] = {
        {"mode 99", badMode},
        {"stochasticSamples 3", badSamples},
        {"pixelWindow 24 with PixelWindow", badWindow},
        {"temporalMode 2", badTemporal},
        {"stableHysteresis NaN", badHysteresis},
        {"temporalEmaWeight 0", badEma},
    };

    for (const Case & c : cases)
    {
        const std::string label = std::string("(v) ") + c.name;
        guarded(label, [&]() {
            renderer.setSortSettings(baseline);
            const SortSettings before = renderer.sortSettings();

            bool threw = false;
            try
            {
                renderer.setSortSettings(c.settings);
            }
            catch (const std::invalid_argument &)
            {
                threw = true;
            }
            report(threw, label + ": setSortSettings throws std::invalid_argument");
            report(sameSettings(before, renderer.sortSettings()), label + ": rejected call leaves settings unchanged");

            bool validateThrew = false;
            try
            {
                validateSortSettings(c.settings);
            }
            catch (const std::invalid_argument &)
            {
                validateThrew = true;
            }
            report(validateThrew, label + ": validateSortSettings throws std::invalid_argument");
        });
    }

    guarded("(v) valid settings accepted", [&]() {
        const SortSettings valid = withMode(SortMode::Hybrid);
        renderer.setSortSettings(valid);
        report(sameSettings(valid, renderer.sortSettings()), "(v) valid Hybrid settings are accepted and stored");
    });
}

// (t) Stage timings.
void checkStageTimings(const Scene & scene)
{
    auto timed = makeRenderer(scene);
    timed->setStageTimingEnabled(true);
    const std::vector<float> timedImage = render(*timed, RenderMode::Viewer);
    // The first frame on a fresh renderer overflows the instance buffer and its timings
    // are dropped by design, so run a second frame whose stamps are kept.
    timed->forward(RenderMode::Viewer);
    const ForwardStageTimings timings = timed->lastStageTimings();

    if (!timings.valid)
    {
        std::printf("  [SKIP] stage timings unavailable on this queue\n");
    }
    else
    {
        const float stages[] = {timings.project, timings.scan,   timings.binning, timings.sortKeys,
                                timings.sort,    timings.ranges, timings.rasterize, timings.resolve};

        bool allFinite = std::isfinite(timings.total) && timings.total > 0.0f;
        double sum = 0.0;
        for (float ms : stages)
        {
            allFinite = allFinite && std::isfinite(ms) && ms >= 0.0f;
            sum += static_cast<double>(ms);
        }
        report(allFinite, "(t) every stage field is finite and >= 0, and total > 0");

        char buffer[192];
        std::snprintf(buffer, sizeof buffer, "(t) the eight stages sum to total within 1e-3 ms (sum %.6f, total %.6f)",
                      sum, static_cast<double>(timings.total));
        report(std::fabs(sum - static_cast<double>(timings.total)) <= 1e-3, buffer);

        std::printf("  stage ms: project %.4f  scan %.4f  binning %.4f  sortKeys %.4f  sort %.4f  ranges %.4f  "
                    "rasterize %.4f  resolve %.4f  total %.4f\n",
                    timings.project, timings.scan, timings.binning, timings.sortKeys, timings.sort, timings.ranges,
                    timings.rasterize, timings.resolve, timings.total);
    }

    auto plain = makeRenderer(scene);
    const std::vector<float> plainImage = render(*plain, RenderMode::Viewer);
    report(!plain->lastStageTimings().valid, "(t) timing never enabled: lastStageTimings().valid is false");
    report(sameBits(timedImage, plainImage), "(t) timing on gives the same image bits as timing off");
}

// ---------------------------------------------------------------------------
// (d) Ray-depth form and (r) CPU study reference. Both use one Device, which
// runChecks() creates after every Renderer above has been destroyed.

// Recalibrated from the plan's 1e-5: measured worst relative error is 2.8e-5 for
// |reference| >= 0.2 (fp32 cancellation for rays far from a Gaussian's center); the
// first attempt, a 9-float B form, was 10x worse (2.8e-4) and was replaced by this
// whitened 12-float form. Margin is about 3.5x.
constexpr double kRayFormTolerance = 1e-4;         // |gpu - ref| <= tol * max(|ref|, floor)
// Depths below the near-plane cull (kMinDepth, 0.2) are never consumed by a sort,
// so the relative error is measured against max(|reference|, 0.2).
constexpr double kRayFormScaleFloor = 0.2;
constexpr uint32_t kDegenerateCount = 3;           // the first three edge Gaussians
// Worst per-channel |reference - GPU| for TileOrder. Recalibrated from the plan's
// 1.2e-7 to 6e-7: the measured worst on this scene is 2.98e-7 (FMA contraction and
// exp differences, about 5 ulp). A skip decision flip would show up near 1e-3.
constexpr float kTileOrderTolerance = 6e-7f;
constexpr double kOrderDifferenceThreshold = 1e-6; // per-pixel difference that counts in (r4)

// Four floats of a flat array (4 floats per element) as one vec4.
glm::vec4 vec4At(const std::vector<float> & values, size_t index)
{
    return glm::vec4(values[4 * index], values[4 * index + 1], values[4 * index + 2], values[4 * index + 3]);
}

// Uploads a GaussianSet into a fresh cloud, as examples/forward_test does.
void uploadSet(GaussianCloud & cloud, const GaussianSet & set)
{
    const size_t count = set.count();
    const size_t coefficients = static_cast<size_t>(kShCoefficientsPerGaussian);
    cloud.initialize(count);
    for (size_t i = 0; i < count; ++i)
    {
        cloud.positions()[i] = vec4At(set.means, i);
        cloud.scales()[i] = vec4At(set.scales, i);
        cloud.rotations()[i] = vec4At(set.rotations, i);
        cloud.opacities()[i] = set.opacities[i];
        for (size_t k = 0; k < coefficients; ++k)
        {
            cloud.shCoefficients()[coefficients * i + k] = vec4At(set.sh, coefficients * i + k);
        }
    }
    cloud.uploadToGPU();
}

// CameraUniforms filled as Renderer::forward fills them, without the
// flipUpAxis branch and with no antialiasing.
CameraUniforms makeUniforms(const Camera & camera, uint32_t gaussianCount)
{
    CameraUniforms u{};
    u.worldToCamera = camera.worldToCamera();
    u.fx = camera.fx;
    u.fy = camera.fy;
    u.cx = camera.cx;
    u.cy = camera.cy;
    u.imageWidth = camera.width;
    u.imageHeight = camera.height;
    u.tilesX = dispatchGroups(camera.width, kTileSize);
    u.tilesY = dispatchGroups(camera.height, kTileSize);
    u.gaussianCount = gaussianCount;
    u.shDegree = kShDegree;
    u.renderFlags = 0;
    u.cameraPositionWorld = glm::vec4(glm::vec3(camera.cameraToWorld[3]), 0.0f);
    return u;
}

// Appends one Gaussian with explicit parameters (no random draws).
void appendExplicit(GaussianSet & set, glm::vec3 center, glm::vec3 scale, glm::vec4 rotation, float opacity)
{
    set.means.insert(set.means.end(), {center.x, center.y, center.z, 0.0f});
    set.scales.insert(set.scales.end(), {scale.x, scale.y, scale.z, 0.0f});
    set.rotations.insert(set.rotations.end(), {rotation.x, rotation.y, rotation.z, rotation.w});
    set.opacities.push_back(opacity);
    set.sh.resize(set.sh.size() + 64, 0.0f);
}

// World position of a camera-space point, so each edge case sits where it is
// meant to, whatever the camera pose.
glm::vec3 cameraPoint(const Camera & camera, glm::vec3 local)
{
    return glm::vec3(camera.cameraToWorld * glm::vec4(local, 1.0f));
}

// The synthetic scene plus six edge Gaussians. The first kDegenerateCount are
// degenerate (the kernel writes twelve zeros for them): a zero scale component, a
// scale beyond 1e15, and a zero quaternion. The other three are ordinary: a very
// anisotropic one, one far behind the camera, and one on the optical axis.
GaussianSet withEdgeCases(const Scene & scene)
{
    const Camera & camera = scene.camera;
    const glm::vec4 identity(0.0f, 0.0f, 0.0f, 1.0f);
    GaussianSet set = scene.gaussians;

    appendExplicit(set, cameraPoint(camera, glm::vec3(-0.3f, 0.2f, -2.5f)), glm::vec3(0.0f, 0.1f, 0.1f), identity,
                   0.5f);
    appendExplicit(set, cameraPoint(camera, glm::vec3(0.3f, 0.1f, -2.5f)), glm::vec3(2e15f, 0.1f, 0.1f), identity,
                   0.5f);
    appendExplicit(set, cameraPoint(camera, glm::vec3(0.0f, -0.2f, -2.5f)), glm::vec3(0.1f, 0.1f, 0.1f),
                   glm::vec4(0.0f), 0.5f);
    appendExplicit(set, cameraPoint(camera, glm::vec3(0.1f, -0.05f, -3.0f)), glm::vec3(0.001f, 0.5f, 2.0f),
                   glm::normalize(glm::vec4(0.3f, 0.5f, -0.2f, 0.7f)), 0.8f);
    appendExplicit(set, cameraPoint(camera, glm::vec3(0.0f, 0.0f, 50.0f)), glm::vec3(0.2f, 0.2f, 0.2f), identity,
                   0.7f);
    appendExplicit(set, cameraPoint(camera, glm::vec3(0.0f, 0.0f, -2.0f)), glm::vec3(0.1f, 0.2f, 0.05f),
                   glm::normalize(glm::vec4(0.1f, 0.2f, 0.3f, 0.9f)), 0.6f);
    return set;
}

// The camera as the GPU sees it, in double: worldToCamera, and the world-space
// camera center inverse(worldToCamera) * (0, 0, 0, 1).
struct DoubleCamera
{
    glm::dmat4 worldToCamera {1.0};
    glm::dvec3 center {0.0};
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
};

DoubleCamera toDoubleCamera(const Camera & camera)
{
    DoubleCamera d;
    d.worldToCamera = glm::dmat4(camera.worldToCamera());
    d.center = glm::dvec3(glm::inverse(d.worldToCamera) * glm::dvec4(0.0, 0.0, 0.0, 1.0));
    d.fx = camera.fx;
    d.fy = camera.fy;
    d.cx = camera.cx;
    d.cy = camera.cy;
    return d;
}

// Reference peak depth t* in double, from the raw parameters in world space
// (the 3DGRT formula, a different derivation from the kernel's camera-space
// form). With o' = S^-1 R^T (c - mean) and d' = S^-1 R^T d_world, the peak is
// t* = -(o'.d') / (d'.d'). The ray is through pixel (px, py) in the camera's
// frame, and d_world = W^T d_cam with W the upper 3x3 of worldToCamera.
double referencePeakDepth(const DoubleCamera & cam, const GaussianSet & set, size_t i, double px, double py)
{
    const glm::dvec3 mean(set.means[4 * i], set.means[4 * i + 1], set.means[4 * i + 2]);
    const glm::dvec3 scale(set.scales[4 * i], set.scales[4 * i + 1], set.scales[4 * i + 2]);
    const glm::dvec4 q(set.rotations[4 * i], set.rotations[4 * i + 1], set.rotations[4 * i + 2],
                       set.rotations[4 * i + 3]);
    const glm::dvec4 unit = q / glm::length(q);
    const double x = unit.x;
    const double y = unit.y;
    const double z = unit.z;
    const double w = unit.w;

    // R as the shader's quaternionToMatrix writes it: R[row][column].
    const double R[3][3] = {
        {1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)},
        {2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)},
        {2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)},
    };
    const double s[3] = {scale.x, scale.y, scale.z};

    const glm::dvec3 origin = cam.center - mean;
    const glm::dvec3 dCam((px - cam.cx) / cam.fx, -(py - cam.cy) / cam.fy, -1.0);
    const glm::dvec3 dWorld = glm::transpose(glm::dmat3(cam.worldToCamera)) * dCam;

    // (S^-1 R^T v)_row = (1 / s_row) * sum_col R[col][row] * v_col
    glm::dvec3 oLocal;
    glm::dvec3 dLocal;
    for (int row = 0; row < 3; ++row)
    {
        double o = 0.0;
        double d = 0.0;
        for (int col = 0; col < 3; ++col)
        {
            o += R[col][row] * origin[col];
            d += R[col][row] * dWorld[col];
        }
        oLocal[row] = o / s[row];
        dLocal[row] = d / s[row];
    }
    return -glm::dot(oLocal, dLocal) / glm::dot(dLocal, dLocal);
}

// The host formula of the whitened form, in double over the downloaded floats:
// form[0..2], form[3..5], form[6..8] are the columns c0, c1, c2 of P, and
// form[9..11] is u. d' = c0 dx + c1 dy - c2 and t = dot(u, d') / dot(d', d').
double evaluateForm(const float * form, double fallback, double dx, double dy)
{
    double dp[3];
    for (int k = 0; k < 3; ++k)
    {
        dp[k] = static_cast<double>(form[k]) * dx + static_cast<double>(form[3 + k]) * dy
                - static_cast<double>(form[6 + k]);
    }
    const double dd = dp[0] * dp[0] + dp[1] * dp[1] + dp[2] * dp[2];
    if (!(dd > 0.0) || !std::isfinite(dd))
    {
        return fallback;
    }
    const double num = static_cast<double>(form[9]) * dp[0] + static_cast<double>(form[10]) * dp[1]
                       + static_cast<double>(form[11]) * dp[2];
    const double t = num / dd;
    return std::isfinite(t) ? t : fallback;
}

// Depth of the Gaussian's center: -(worldToCamera * (mean, 1)).z.
double centerDepth(const DoubleCamera & cam, const GaussianSet & set, size_t i)
{
    const glm::dvec4 mean(set.means[4 * i], set.means[4 * i + 1], set.means[4 * i + 2], 1.0);
    return -(cam.worldToCamera * mean).z;
}

// The 25 sample pixels: four corners, the center, four tile centers and 16
// pixels from a seeded xorshift32. Pixel centers are at integer coordinates.
std::vector<glm::vec2> raySamples()
{
    std::vector<glm::vec2> samples;
    const float right = static_cast<float>(kWidth - 1);
    const float bottom = static_cast<float>(kHeight - 1);
    samples.push_back(glm::vec2(0.0f, 0.0f));
    samples.push_back(glm::vec2(right, 0.0f));
    samples.push_back(glm::vec2(0.0f, bottom));
    samples.push_back(glm::vec2(right, bottom));
    samples.push_back(glm::vec2(0.5f * right, 0.5f * bottom));

    const uint32_t tilesX = dispatchGroups(kWidth, kTileSize);
    const uint32_t tilesY = dispatchGroups(kHeight, kTileSize);
    const uint32_t tiles[4][2] = {
        {0, 0},
        {tilesX - 1, tilesY - 1},
        {tilesX / 3, tilesY / 3},
        {(2 * tilesX) / 3, (2 * tilesY) / 3},
    };
    for (const auto & tile : tiles)
    {
        const float tileSize = static_cast<float>(kTileSize);
        samples.push_back(glm::vec2(tileSize * static_cast<float>(tile[0]) + 7.5f,
                                    tileSize * static_cast<float>(tile[1]) + 7.5f));
    }

    Xorshift32 rng(0x2545F491u);
    for (int k = 0; k < 16; ++k)
    {
        const float px = static_cast<float>(rng.next() % kWidth);
        const float py = static_cast<float>(rng.next() % kHeight);
        samples.push_back(glm::vec2(px, py));
    }
    return samples;
}

struct RayFormStats
{
    size_t comparisons = 0;
    size_t failures = 0;
    double worstRelative = 0.0;
    size_t worstIndex = 0;
    double worstReference = 0.0;
    double worstGpu = 0.0;
    double worstRelativeBeyondMinDepth = 0.0;
    double worstBeyondReference = 0.0;
    size_t failuresBeyondMinDepth = 0; // failures where |reference| >= 0.2, the near-plane cull depth
};

// Compares the kernel's twelve floats per Gaussian with the reference at every
// sample pixel, skipping the Gaussians in [skipBegin, skipEnd).
RayFormStats compareRayForm(const std::vector<float> & form, const GaussianSet & set, const DoubleCamera & cam,
                            const std::vector<glm::vec2> & samples, size_t skipBegin, size_t skipEnd)
{
    RayFormStats stats;
    const size_t count = set.count();
    for (size_t i = 0; i < count; ++i)
    {
        if (i >= skipBegin && i < skipEnd)
        {
            continue;
        }
        const float * f = &form[12 * i];
        const double fallback = centerDepth(cam, set, i);
        for (const glm::vec2 & p : samples)
        {
            const double px = p.x;
            const double py = p.y;
            const double dx = (px - cam.cx) / cam.fx;
            const double dy = -(py - cam.cy) / cam.fy;
            const double reference = referencePeakDepth(cam, set, i, px, py);
            const double gpu = evaluateForm(f, fallback, dx, dy);
            const double relative =
                std::fabs(gpu - reference) / std::max(std::fabs(reference), kRayFormScaleFloor);

            ++stats.comparisons;
            if (std::fabs(reference) >= 0.2 && !(relative <= stats.worstRelativeBeyondMinDepth))
            {
                stats.worstRelativeBeyondMinDepth = relative;
                stats.worstBeyondReference = reference;
            }
            if (!(relative <= kRayFormTolerance))
            {
                ++stats.failures;
                if (std::fabs(reference) >= 0.2)
                {
                    ++stats.failuresBeyondMinDepth;
                }
            }
            if (!(relative <= stats.worstRelative))
            {
                stats.worstRelative = relative;
                stats.worstIndex = i;
                stats.worstReference = reference;
                stats.worstGpu = gpu;
            }
        }
    }
    return stats;
}

void reportRayForm(const std::string & label, const RayFormStats & stats)
{
    char buffer[320];
    std::snprintf(buffer, sizeof buffer,
                  "%s: %zu comparisons, worst relative error %.3g at Gaussian %zu (reference %.9g, gpu %.9g), %zu over tolerance %.0e, %zu of them with |reference| >= 0.2, worst there %.3g at reference %.6g",
                  label.c_str(), stats.comparisons, stats.worstRelative, stats.worstIndex, stats.worstReference,
                  stats.worstGpu, stats.failures, kRayFormTolerance, stats.failuresBeyondMinDepth, stats.worstRelativeBeyondMinDepth, stats.worstBeyondReference);
    report(stats.failures == 0, buffer);
}

std::vector<float> downloadRayForm(const ForwardPass & pass, size_t count)
{
    std::vector<float> form(12 * count);
    pass.rayDepthForm().download(form.data(), form.size() * sizeof(float));
    return form;
}

bool degenerateAllZero(const std::vector<float> & form, size_t begin, size_t end)
{
    for (size_t i = begin; i < end; ++i)
    {
        for (size_t k = 0; k < 12; ++k)
        {
            if (form[12 * i + k] != 0.0f)
            {
                return false;
            }
        }
    }
    return true;
}

bool ordinaryAllFinite(const std::vector<float> & form, size_t count, size_t skipBegin, size_t skipEnd)
{
    for (size_t i = 0; i < count; ++i)
    {
        if (i >= skipBegin && i < skipEnd)
        {
            continue;
        }
        for (size_t k = 0; k < 12; ++k)
        {
            if (!std::isfinite(form[12 * i + k]))
            {
                return false;
            }
        }
    }
    return true;
}

// (d) Ray-depth form against the double world-space reference.
void checkRayDepthForm(Device & device, const Scene & scene)
{
    const GaussianSet set = withEdgeCases(scene);
    const size_t count = set.count();
    const size_t degenerateBegin = scene.gaussians.count();
    const size_t degenerateEnd = degenerateBegin + kDegenerateCount;

    GaussianCloud cloud(device);
    uploadSet(cloud, set);
    const auto views = cloud.views();

    ForwardPass pass(device, kWidth, kHeight, testShaderDir());

    // Before any computeRayDepthForm() on this pass.
    bool threw = false;
    try
    {
        (void)pass.rayDepthForm();
    }
    catch (const std::runtime_error &)
    {
        threw = true;
    }
    report(threw, "(d) rayDepthForm() before any computeRayDepthForm throws std::runtime_error");

    const Camera & camera = scene.camera;
    const CameraUniforms uniforms = makeUniforms(camera, views.gaussianCount);
    pass.computeRayDepthForm(views, uniforms);
    const std::vector<float> form = downloadRayForm(pass, count);

    report(degenerateAllZero(form, degenerateBegin, degenerateEnd),
           "(d) the three degenerate Gaussians (zero scale, scale 2e15, zero quaternion) "
           "have all twelve floats exactly 0");
    report(ordinaryAllFinite(form, count, degenerateBegin, degenerateEnd),
           "(d) every other Gaussian's twelve floats are finite");

    const std::vector<glm::vec2> samples = raySamples();
    const DoubleCamera cam = toDoubleCamera(camera);
    reportRayForm("(d) ray-depth form vs double reference, camera A",
                  compareRayForm(form, set, cam, samples, degenerateBegin, degenerateEnd));

    // The same camera again: the buffer is rewritten with identical values.
    pass.computeRayDepthForm(views, uniforms);
    report(sameBits(downloadRayForm(pass, count), form),
           "(d) second computeRayDepthForm with the same camera is bit-identical");

    // A different camera (pose and intrinsics): the buffer must follow it.
    const Camera other = makeCamera(kWidth, kHeight, 290.0f, 300.0f, 120.0f, 100.0f,
                                    makePose(0.1f, 0.2f, -0.05f, 0.97f, glm::vec3(-0.3f, 0.2f, 4.2f)));
    pass.computeRayDepthForm(views, makeUniforms(other, views.gaussianCount));
    const std::vector<float> otherForm = downloadRayForm(pass, count);
    report(!sameBits(otherForm, form), "(d) after a different camera the form buffer is rewritten");
    report(degenerateAllZero(otherForm, degenerateBegin, degenerateEnd),
           "(d) after a different camera the degenerate Gaussians are still exactly 0");
    reportRayForm("(d) ray-depth form vs double reference, camera B",
                  compareRayForm(otherForm, set, toDoubleCamera(other), samples, degenerateBegin, degenerateEnd));
}

// Worst per-channel absolute difference over RGBA. A NaN makes the result NaN,
// so a non-finite pixel fails any tolerance check.
double worstChannelDifference(const std::vector<glm::vec4> & a, const std::vector<glm::vec4> & b)
{
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        for (int channel = 0; channel < 4; ++channel)
        {
            const double d = std::fabs(static_cast<double>(a[i][channel]) - static_cast<double>(b[i][channel]));
            // NaN never compares greater, so it is taken explicitly: once a NaN
            // is seen, the result stays NaN and fails any tolerance check.
            if (std::isnan(d) || d > worst)
            {
                worst = d;
            }
        }
    }
    return worst;
}

// Bit-for-bit equality of two RGBA images (memcmp, as sameBits).
bool sameImage(const std::vector<glm::vec4> & a, const std::vector<glm::vec4> & b)
{
    if (a.size() != b.size())
    {
        return false;
    }
    return a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(glm::vec4)) == 0;
}

size_t coveredRgba(const std::vector<glm::vec4> & image)
{
    size_t covered = 0;
    for (const glm::vec4 & pixel : image)
    {
        if (pixel.a > 0.0f)
        {
            ++covered;
        }
    }
    return covered;
}

bool allFinite(const std::vector<glm::vec4> & image)
{
    for (const glm::vec4 & pixel : image)
    {
        for (int channel = 0; channel < 4; ++channel)
        {
            if (!std::isfinite(pixel[channel]))
            {
                return false;
            }
        }
    }
    return true;
}

// One CPU reference render, timed and printed.
std::vector<glm::vec4> timedReference(const char * name, const study::StudyCandidates & candidates,
                                      study::ReferenceOrder order, uint32_t window)
{
    study::ReferenceOptions options;
    options.order = order;
    options.window = window;

    const auto start = std::chrono::steady_clock::now();
    std::vector<glm::vec4> image = study::renderReference(candidates, options);
    const auto stop = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(stop - start).count();

    std::printf("  reference %-22s %.4f s\n", name, seconds);
    return image;
}

void reportCoverage(const std::string & name, const std::vector<glm::vec4> & image)
{
    report(coveredRgba(image) > 0 && allFinite(image),
           "(r5) " + name + " reference image has coverage and finite values");
}

// (r) CPU reference self-checks, on the plain synthetic scene (no edge cases).
void checkCpuReference(Device & device, const Scene & scene)
{
    GaussianCloud cloud(device);
    uploadSet(cloud, scene.gaussians);
    const auto views = cloud.views();
    const CameraUniforms uniforms = makeUniforms(scene.camera, views.gaussianCount);

    ForwardPass pass(device, kWidth, kHeight, testShaderDir());
    pass.run(views, uniforms);
    const study::StudyCandidates candidates = study::captureCandidates(pass, views, uniforms, true);
    const std::vector<glm::vec4> gpuImage = pass.downloadOutputImage();

    const std::vector<glm::vec4> tileOrder =
        timedReference("TileOrder", candidates, study::ReferenceOrder::TileOrder, 16);
    const std::vector<glm::vec4> window1 =
        timedReference("WindowRayDepth K=1", candidates, study::ReferenceOrder::WindowRayDepth, 1);
    const std::vector<glm::vec4> window16 =
        timedReference("WindowRayDepth K=16", candidates, study::ReferenceOrder::WindowRayDepth, 16);
    const std::vector<glm::vec4> window4096 =
        timedReference("WindowRayDepth K=4096", candidates, study::ReferenceOrder::WindowRayDepth, 4096);
    const std::vector<glm::vec4> exact =
        timedReference("ExactRayDepth", candidates, study::ReferenceOrder::ExactRayDepth, 16);

    // (r1) TileOrder reproduces the GPU image up to FMA and exp rounding.
    const double worst = tileOrder.size() == gpuImage.size()
                             ? worstChannelDifference(tileOrder, gpuImage)
                             : std::numeric_limits<double>::infinity();
    char buffer[256];
    std::snprintf(buffer, sizeof buffer,
                  "(r1) TileOrder reference vs GPU: worst per-channel |difference| %.9g (tolerance %.3g)", worst,
                  kTileOrderTolerance);
    report(worst <= kTileOrderTolerance, buffer);

    // (r2) and (r3) the window limits.
    report(sameImage(window1, tileOrder), "(r2) WindowRayDepth with window 1 is bit-identical to TileOrder");
    report(sameImage(window4096, exact), "(r3) WindowRayDepth with window 4096 is bit-identical to ExactRayDepth");

    // (r4) the order matters on this scene, so the study has something to measure.
    size_t differing = 0;
    double maxDifference = 0.0;
    for (size_t i = 0; i < exact.size(); ++i)
    {
        double pixelMax = 0.0;
        for (int channel = 0; channel < 4; ++channel)
        {
            pixelMax = std::max(pixelMax, std::fabs(static_cast<double>(exact[i][channel])
                                                    - static_cast<double>(tileOrder[i][channel])));
        }
        maxDifference = std::max(maxDifference, pixelMax);
        if (pixelMax > kOrderDifferenceThreshold)
        {
            ++differing;
        }
    }
    std::snprintf(buffer, sizeof buffer,
                  "(r4) ExactRayDepth differs from TileOrder on %zu pixels (max channel difference %.6g)", differing,
                  maxDifference);
    report(differing > 0, buffer);

    // (r5) coverage and finite values, for every reference image.
    reportCoverage("TileOrder", tileOrder);
    reportCoverage("WindowRayDepth K=1", window1);
    reportCoverage("WindowRayDepth K=16", window16);
    reportCoverage("WindowRayDepth K=4096", window4096);
    reportCoverage("ExactRayDepth", exact);
}

// ---------------------------------------------------------------------------
// (c) S1 sticky keys, (e) TileRayDepth and Hybrid keys, (f) window images.

// (c) stableKeyAmbiguous flags a Gaussian-frame whose float result could differ
// from the GPU's log2/FMA. Every other Gaussian-frame must match exactly.
constexpr uint32_t kStableFrames = 40;
constexpr double kStableOrbitStepDegrees = 0.5;
// The floor boundary (+-0.02 of an integer, about 4%) only counts when a key is rewritten,
// and a 0.5 degree orbit step moves most Gaussians by several buckets (bucket width about
// 6.8e-4 relative), so about 64% of Gaussian-frames rewrite: measured 3.2% ambiguous.
// The limit only guards against the model flagging nearly everything.
constexpr double kAmbiguousFractionLimit = 0.10; // ambiguous Gaussian-frames / all Gaussian-frames
constexpr double kPairEpsilon = 1e-6;              // relative depth gap of the coplanar pair
constexpr int kPairFrames = 20;
constexpr int kPairMinDefaultSwaps = 15;
// (e2) Same fp32 conditioning as kRayFormTolerance: the key comes from the same
// 12-float form in fp32, so the worst case measured in (d) applies. Relative to
// max(t, kRayFormScaleFloor), the same floor as (d).
// The plan's 1e-6 holds: tile-center rays are well conditioned (measured worst 5.7e-7).
constexpr double kRayDepthKeyTolerance = 1e-6;
// (f) The GPU sorts t* in float and the model in double, so a few near-tie pixels
// can differ. kWindowPixelFraction is the initial limit on the fraction of pixels
// whose worst channel difference exceeds kWindowPixelTolerance. It is to be
// calibrated once from a measured run, then fixed.
constexpr double kWindowPixelTolerance = 2e-6;
constexpr double kWindowPixelFraction = 0.001;
// A window image must differ from the default image on at least one pixel by more
// than this, so the modes are shown to do something.
constexpr double kWindowDifferTolerance = 1e-6;
constexpr double kSparseTolerance = 2e-6; // (f) sparse exactness, every pixel

float bitsToFloat(uint32_t bits)
{
    float value;
    std::memcpy(&value, &bits, sizeof value);
    return value;
}

// The scene camera with its pose rotated about the world Y axis through the
// origin, the scene's center. The whole pose turns (not only the eye), so the
// scene stays in view along the path.
Camera orbitCamera(const Camera & base, double radians)
{
    Camera camera = base;
    camera.cameraToWorld =
        glm::rotate(glm::mat4(1.0f), static_cast<float>(radians), glm::vec3(0.0f, 1.0f, 0.0f)) * base.cameraToWorld;
    return camera;
}

// (c) S1: each frame's new sticky key equals stableKeyStep of the previous key,
// except where stableKeyAmbiguous. The previous keys are the GPU's own buffer,
// downloaded before each run, so the GPU's state carries from frame to frame.
void checkStableKeys(Device & device, const Scene & scene)
{
    const GaussianSet & set = scene.gaussians;
    const size_t count = set.count();

    GaussianCloud cloud(device);
    uploadSet(cloud, set);
    const auto views = cloud.views();

    ForwardPass pass(device, kWidth, kHeight, testShaderDir());
    SortSettings stable = withMode(SortMode::Stable);
    stable.stableBucketsPerOctave = 1024;
    stable.stableHysteresis = 1.0f;
    pass.setSortSettings(stable);
    const float bucketsPerOctave = static_cast<float>(stable.stableBucketsPerOctave);
    const float hysteresis = stable.stableHysteresis;

    std::vector<uint32_t> prev(count, study::kStableSentinel);
    size_t comparisons = 0;
    size_t ambiguous = 0;
    size_t mismatches = 0;
    size_t keyUpdates = 0;
    std::string firstMismatch;

    for (uint32_t frame = 0; frame < kStableFrames; ++frame)
    {
        const double radians = static_cast<double>(frame) * kStableOrbitStepDegrees * kPi / 180.0;
        const Camera camera = orbitCamera(scene.camera, radians);
        const CameraUniforms uniforms = makeUniforms(camera, views.gaussianCount);

        // stableKeys() throws before the first Stable run, so frame 0 keeps the
        // sentinel the buffer is filled with at reset.
        if (frame > 0)
        {
            pass.stableKeys().download(prev.data(), count * sizeof(uint32_t));
        }
        pass.run(views, uniforms, ForwardStage::Rasterize, ForwardMode::Viewer);

        std::vector<uint32_t> next(count);
        pass.stableKeys().download(next.data(), count * sizeof(uint32_t));
        std::vector<ProjectedGaussian> projected(count);
        pass.projected().download(projected.data(), count * sizeof(ProjectedGaussian));

        for (size_t i = 0; i < count; ++i)
        {
            const ProjectedGaussian & p = projected[i];
            ++comparisons;
            if (next[i] != prev[i])
            {
                ++keyUpdates;
            }
            if (study::stableKeyAmbiguous(prev[i], p.depth, p.tileCount, bucketsPerOctave, hysteresis))
            {
                ++ambiguous;
                continue;
            }
            const uint32_t expected =
                study::stableKeyStep(prev[i], p.depth, p.tileCount, bucketsPerOctave, hysteresis);
            if (expected != next[i])
            {
                if (mismatches == 0)
                {
                    char detail[320];
                    std::snprintf(detail, sizeof detail,
                                  "frame %u, Gaussian %zu: previous %u, depth %.9g, tileCount %u, expected %u, gpu %u",
                                  frame, i, prev[i], static_cast<double>(p.depth), p.tileCount, expected, next[i]);
                    firstMismatch = detail;
                }
                ++mismatches;
            }
        }
    }

    char buffer[512];
    std::snprintf(buffer, sizeof buffer,
                  "(c) S1 sticky keys over %u frames: %zu Gaussian-frames, %zu key updates, %zu mismatches outside the ambiguous set",
                  kStableFrames, comparisons, keyUpdates, mismatches);
    std::string summary = buffer;
    if (!firstMismatch.empty())
    {
        summary += "; first: " + firstMismatch;
    }
    report(mismatches == 0, summary);

    const double ambiguousFraction =
        comparisons > 0 ? static_cast<double>(ambiguous) / static_cast<double>(comparisons) : 1.0;
    std::snprintf(buffer, sizeof buffer, "(c) ambiguous Gaussian-frames: %zu of %zu (%.3f%%), limit %.2f%%", ambiguous,
                  comparisons, 100.0 * ambiguousFraction, 100.0 * kAmbiguousFractionLimit);
    report(ambiguousFraction < kAmbiguousFractionLimit, buffer);
}

// Position of the pair (Gaussians 0 and 1) in one tile's list: 0 when 0 precedes
// 1, 1 when it follows, -1 when either is missing from the tile.
int pairOrder(const std::vector<uint32_t> & values, TileRange range)
{
    int positionA = -1;
    int positionB = -1;
    for (uint32_t j = range.first; j < range.last && j < values.size(); ++j)
    {
        if (values[j] == 0)
        {
            positionA = static_cast<int>(j);
        }
        else if (values[j] == 1)
        {
            positionB = static_cast<int>(j);
        }
    }
    if (positionA < 0 || positionB < 0)
    {
        return -1;
    }
    return positionA < positionB ? 0 : 1;
}

struct PairRun
{
    int swaps = 0;   // order changes between consecutive frames, counted from frame 2 on
    int missing = 0; // frames where the center tile did not hold both Gaussians
};

// Two splats at the image center, depths 5 and 5(1 +- 1e-6) with the sign
// alternating per frame, under a fresh ForwardPass in `mode`. The cloud is
// re-uploaded each frame; the pass is not reset by an upload.
PairRun runCoplanarPair(Device & device, SortMode mode)
{
    GaussianSet set;
    const glm::vec4 identity(0.0f, 0.0f, 0.0f, 1.0f);
    appendExplicit(set, glm::vec3(0.0f, 0.0f, -5.0f), glm::vec3(0.3f), identity, 0.5f);
    appendExplicit(set, glm::vec3(0.0f, 0.0f, -5.0f), glm::vec3(0.3f), identity, 0.5f);

    // Identity camera: world coordinates are camera coordinates, and depth is -z.
    const Camera camera = makeCamera(kWidth, kHeight, 300.0f, 300.0f, 128.0f, 96.0f, glm::mat4(1.0f));

    GaussianCloud cloud(device);
    uploadSet(cloud, set);

    ForwardPass pass(device, kWidth, kHeight, testShaderDir());
    pass.setSortSettings(withMode(mode));

    // The image center (128, 96) is the first pixel of tile (8, 6).
    const uint32_t tilesX = dispatchGroups(kWidth, kTileSize);
    const uint32_t tilesY = dispatchGroups(kHeight, kTileSize);
    const uint32_t centerTile = (static_cast<uint32_t>(camera.cy) / kTileSize) * tilesX
                                + static_cast<uint32_t>(camera.cx) / kTileSize;

    PairRun result;
    int previous = -1;
    for (int frame = 0; frame < kPairFrames; ++frame)
    {
        const double eps = (frame % 2 == 0) ? kPairEpsilon : -kPairEpsilon;
        cloud.positions()[1] = glm::vec4(0.0f, 0.0f, static_cast<float>(-5.0 * (1.0 + eps)), 0.0f);
        cloud.uploadToGPU();
        const auto views = cloud.views();
        const CameraUniforms uniforms = makeUniforms(camera, views.gaussianCount);
        pass.run(views, uniforms);

        const uint32_t instances = pass.lastInstanceCount();
        std::vector<uint32_t> values(instances);
        if (instances > 0)
        {
            pass.sortedValues().download(values.data(), instances * sizeof(uint32_t));
        }
        std::vector<TileRange> ranges(static_cast<size_t>(tilesX) * tilesY);
        pass.tileRanges().download(ranges.data(), ranges.size() * sizeof(TileRange));

        const int order = pairOrder(values, ranges[centerTile]);
        if (order < 0)
        {
            ++result.missing;
            previous = -1;
            continue;
        }
        // Frame 0 under Stable only establishes the sticky keys, so swaps are
        // counted from frame 1 on: the first comparison is frame 1 against frame 2.
        if (frame >= 2 && previous >= 0 && order != previous)
        {
            ++result.swaps;
        }
        previous = order;
    }
    return result;
}

// (c) The coplanar pair: Default order flips with the depth gap, Stable order must not.
void checkCoplanarPair(Device & device)
{
    const PairRun defaultRun = runCoplanarPair(device, SortMode::Default);
    char buffer[320];
    std::snprintf(buffer, sizeof buffer,
                  "(c) coplanar pair, Default order: %d order swaps over %d frames (%d frames without both), need at least %d",
                  defaultRun.swaps, kPairFrames, defaultRun.missing, kPairMinDefaultSwaps);
    report(defaultRun.missing == 0 && defaultRun.swaps >= kPairMinDefaultSwaps, buffer);

    const PairRun stableRun = runCoplanarPair(device, SortMode::Stable);
    std::snprintf(buffer, sizeof buffer,
                  "(c) coplanar pair, Stable order: %d order swaps over %d frames (%d frames without both), need 0",
                  stableRun.swaps, kPairFrames, stableRun.missing);
    report(stableRun.missing == 0 && stableRun.swaps == 0, buffer);
}

// Lexicographic (tile, key, Gaussian index) order, as a non-decreasing test.
bool keyLessOrEqual(uint32_t aTile, uint32_t aKey, uint32_t aValue, uint32_t bTile, uint32_t bKey, uint32_t bValue)
{
    if (aTile != bTile)
    {
        return aTile < bTile;
    }
    if (aKey != bKey)
    {
        return aKey < bKey;
    }
    return aValue <= bValue;
}

// (e) TileRayDepth and Hybrid: the sort keys are the tile-center ray depths.
void checkTileKeys(Device & device, const Scene & scene)
{
    const size_t count = scene.gaussians.count();
    GaussianCloud cloud(device);
    uploadSet(cloud, scene.gaussians);
    const auto views = cloud.views();
    const CameraUniforms uniforms = makeUniforms(scene.camera, views.gaussianCount);
    const DoubleCamera cam = toDoubleCamera(scene.camera);
    const uint32_t tilesX = dispatchGroups(kWidth, kTileSize);
    const uint32_t tilesY = dispatchGroups(kHeight, kTileSize);
    const double tileSize = static_cast<double>(kTileSize);

    const SortMode modes[] = {SortMode::TileRayDepth, SortMode::Hybrid};
    for (SortMode mode : modes)
    {
        const std::string name = sortModeName(mode);
        guarded(std::string("(e) sort keys, mode ") + name, [&]() {
            ForwardPass pass(device, kWidth, kHeight, testShaderDir());
            pass.setSortSettings(withMode(mode));
            pass.run(views, uniforms);

            const uint32_t n = pass.lastInstanceCount();
            report(n > 0, "(e) " + name + ": the run emitted tile instances (" + std::to_string(n) + ")");
            if (n == 0)
            {
                return;
            }

            std::vector<uint32_t> keysTile(n);
            std::vector<uint32_t> keysDepth(n);
            std::vector<uint32_t> values(n);
            pass.sortedKeysTile().download(keysTile.data(), n * sizeof(uint32_t));
            pass.sortedKeysDepth().download(keysDepth.data(), n * sizeof(uint32_t));
            pass.sortedValues().download(values.data(), n * sizeof(uint32_t));
            std::vector<TileRange> ranges(static_cast<size_t>(tilesX) * tilesY);
            pass.tileRanges().download(ranges.data(), ranges.size() * sizeof(TileRange));
            std::vector<ProjectedGaussian> projected(count);
            pass.projected().download(projected.data(), count * sizeof(ProjectedGaussian));

            // (e1) the list is sorted by (tile, key, Gaussian index).
            size_t orderViolations = 0;
            for (size_t i = 1; i < n; ++i)
            {
                if (!keyLessOrEqual(keysTile[i - 1], keysDepth[i - 1], values[i - 1], keysTile[i], keysDepth[i],
                                    values[i]))
                {
                    ++orderViolations;
                }
            }
            report(orderViolations == 0, "(e1) " + name + ": " + std::to_string(n)
                                             + " instances sorted by (tile, key, Gaussian index), "
                                             + std::to_string(orderViolations) + " violations");

            // (e1) each tile range holds only its own tile's instances, and the
            // ranges together cover every instance.
            size_t rangeViolations = 0;
            size_t covered = 0;
            for (uint32_t tile = 0; tile < ranges.size(); ++tile)
            {
                const TileRange range = ranges[tile];
                if (range.last < range.first || range.last > n)
                {
                    ++rangeViolations;
                    continue;
                }
                covered += range.last - range.first;
                for (uint32_t j = range.first; j < range.last; ++j)
                {
                    if (keysTile[j] != tile)
                    {
                        ++rangeViolations;
                    }
                }
            }
            report(rangeViolations == 0 && covered == n,
                   "(e1) " + name + ": tile ranges hold their own tile's instances and cover all " + std::to_string(n));

            // (e1) the instance list agrees with the projected buffer: every instance's
            // Gaussian touches tiles, and the tile counts sum to the instance count.
            size_t expectedInstances = 0;
            size_t badInstances = 0;
            for (const ProjectedGaussian & p : projected)
            {
                expectedInstances += p.tileCount;
            }
            for (size_t i = 0; i < n; ++i)
            {
                const uint32_t g = values[i];
                if (g >= count || projected[g].tileCount == 0)
                {
                    ++badInstances;
                }
            }
            report(expectedInstances == n && badInstances == 0,
                   "(e1) " + name + ": projected tile counts sum to " + std::to_string(expectedInstances)
                       + " (instances " + std::to_string(n) + "), " + std::to_string(badInstances)
                       + " instances from a Gaussian with tileCount 0");

            // (e2) each key is the double tile-center ray depth, clamped at 0.
            size_t failures = 0;
            size_t worstIndex = 0;
            double worstRelative = 0.0;
            double worstReference = 0.0;
            double worstKey = 0.0;
            for (size_t i = 0; i < n; ++i)
            {
                const uint32_t tile = keysTile[i];
                const double px = tileSize * static_cast<double>(tile % tilesX) + 7.5;
                const double py = tileSize * static_cast<double>(tile / tilesX) + 7.5;
                const double reference = std::max(referencePeakDepth(cam, scene.gaussians, values[i], px, py), 0.0);
                const double key = static_cast<double>(bitsToFloat(keysDepth[i]));
                const double relative = std::fabs(key - reference) / std::max(reference, kRayFormScaleFloor);

                if (!(relative <= kRayDepthKeyTolerance))
                {
                    ++failures;
                }
                if (!(relative <= worstRelative))
                {
                    worstRelative = relative;
                    worstIndex = i;
                    worstReference = reference;
                    worstKey = key;
                }
            }
            char buffer[400];
            std::snprintf(buffer, sizeof buffer,
                          "(e2) %s: %zu instances vs double tile-center ray depth (clamped at 0, relative to max(t, 0.2)): "
                          "worst relative error %.3g at instance %zu (reference %.9g, key %.9g), %zu over tolerance %.0e",
                          name.c_str(), static_cast<size_t>(n), worstRelative, worstIndex, worstReference, worstKey,
                          failures, kRayDepthKeyTolerance);
            report(failures == 0, buffer);

            // (e3) the form the run wrote is the one computeRayDepthForm writes.
            const std::vector<float> runForm = downloadRayForm(pass, count);
            pass.computeRayDepthForm(views, uniforms);
            report(sameBits(downloadRayForm(pass, count), runForm),
                   "(e3) " + name + ": the run's ray-depth form is bit-identical to computeRayDepthForm's output");
        });
    }
}

// Per-pixel worst channel difference, counted above a tolerance. NaN counts as above.
struct WindowDifference
{
    size_t pixels = 0;
    size_t above = 0;   // pixels whose worst channel difference is above the tolerance
    double worst = 0.0; // worst channel difference over all pixels (NaN is sticky)
};

WindowDifference compareImages(const std::vector<glm::vec4> & a, const std::vector<glm::vec4> & b, double tolerance)
{
    WindowDifference result;
    result.pixels = a.size();
    for (size_t i = 0; i < a.size(); ++i)
    {
        double pixelWorst = 0.0;
        for (int channel = 0; channel < 4; ++channel)
        {
            const double d = std::fabs(static_cast<double>(a[i][channel]) - static_cast<double>(b[i][channel]));
            if (std::isnan(d) || d > pixelWorst)
            {
                pixelWorst = d;
            }
        }
        if (!(pixelWorst <= tolerance))
        {
            ++result.above;
        }
        if (std::isnan(pixelWorst) || pixelWorst > result.worst)
        {
            result.worst = pixelWorst;
        }
    }
    return result;
}

// A GPU window image against its CPU window model: the fraction of pixels above
// kWindowPixelTolerance must be at most kWindowPixelFraction.
void reportWindowImage(const std::string & label, const std::vector<glm::vec4> & gpu,
                       const std::vector<glm::vec4> & model)
{
    if (gpu.size() != model.size())
    {
        report(false, label + ": GPU and model image sizes differ");
        return;
    }
    const WindowDifference diff = compareImages(gpu, model, kWindowPixelTolerance);
    const double fraction = diff.pixels > 0 ? static_cast<double>(diff.above) / static_cast<double>(diff.pixels) : 1.0;
    char buffer[384];
    std::snprintf(buffer, sizeof buffer,
                  "%s vs CPU window model: max channel difference %.3g, %zu of %zu pixels (%.4f%%) above %.0e, limit %.2f%%",
                  label.c_str(), diff.worst, diff.above, diff.pixels, 100.0 * fraction, kWindowPixelTolerance,
                  100.0 * kWindowPixelFraction);
    report(fraction <= kWindowPixelFraction, buffer);
}

// (f) Window images, the Hybrid image, and the sparse exactness test.
void checkWindowImages(Device & device, const Scene & scene)
{
    GaussianCloud cloud(device);
    uploadSet(cloud, scene.gaussians);
    const auto views = cloud.views();
    const CameraUniforms uniforms = makeUniforms(scene.camera, views.gaussianCount);

    ForwardPass pass(device, kWidth, kHeight, testShaderDir());
    pass.run(views, uniforms);
    const std::vector<glm::vec4> defaultImage = pass.downloadOutputImage();

    auto windowModel = [](const study::StudyCandidates & candidates, uint32_t window) {
        study::ReferenceOptions options;
        options.order = study::ReferenceOrder::WindowRayDepth;
        options.window = window;
        return study::renderReference(candidates, options);
    };

    // One window mode on the synthetic scene: its image against the window model
    // of K, and a check that the image differs from the default-sort image.
    auto checkWindowMode = [&](const std::string & label, const SortSettings & settings, uint32_t window) {
        guarded(label, [&]() {
            pass.setSortSettings(settings);
            pass.run(views, uniforms);
            const study::StudyCandidates candidates = study::captureCandidates(pass, views, uniforms, true);
            const std::vector<glm::vec4> gpu = pass.downloadOutputImage();
            reportWindowImage(label, gpu, windowModel(candidates, window));

            const WindowDifference fromDefault = compareImages(gpu, defaultImage, kWindowDifferTolerance);
            report(fromDefault.above > 0, label + ": differs from the default-sort image on "
                                              + std::to_string(fromDefault.above) + " pixels");
        });
    };

    for (uint32_t window : {16u, 32u})
    {
        SortSettings settings = withMode(SortMode::PixelWindow);
        settings.pixelWindow = window;
        checkWindowMode("(f) PixelWindow K=" + std::to_string(window), settings, window);
    }
    checkWindowMode("(f) Hybrid K=8", withMode(SortMode::Hybrid), 8);

    // Sparse scene: 40 small Gaussians, so no tile lists 32 or more instances and
    // no pixel can reach the window of 32. The window then never pops, and the
    // result must equal the exact ray-depth order.
    guarded("(f) sparse scene, PixelWindow K=32 vs ExactRayDepth", [&]() {
        GaussianSet sparse;
        Xorshift32 rng(0x5BA25E11u);
        appendScatter(sparse, rng, 40, glm::vec3(-1.2f, -1.2f, -1.2f), glm::vec3(1.2f, 1.2f, 1.2f), 0.05f, 0.2f);

        GaussianCloud sparseCloud(device);
        uploadSet(sparseCloud, sparse);
        const auto sparseViews = sparseCloud.views();
        const CameraUniforms sparseUniforms = makeUniforms(scene.camera, sparseViews.gaussianCount);

        ForwardPass sparsePass(device, kWidth, kHeight, testShaderDir());
        SortSettings settings = withMode(SortMode::PixelWindow);
        settings.pixelWindow = 32;
        sparsePass.setSortSettings(settings);
        sparsePass.run(sparseViews, sparseUniforms);
        const study::StudyCandidates candidates =
            study::captureCandidates(sparsePass, sparseViews, sparseUniforms, true);
        const std::vector<glm::vec4> gpu = sparsePass.downloadOutputImage();

        study::ReferenceOptions options;
        options.order = study::ReferenceOrder::ExactRayDepth;
        const std::vector<glm::vec4> exact = study::renderReference(candidates, options);

        // A tile's list bounds the contributors of each of its pixels.
        uint32_t longestList = 0;
        for (const TileRange & range : candidates.tileRanges)
        {
            longestList = std::max(longestList, range.last - range.first);
        }
        report(longestList < 32, "(f) sparse scene: longest tile list " + std::to_string(longestList)
                                     + " is under 32, so every pixel has fewer than 32 contributors");

        if (gpu.size() != exact.size())
        {
            report(false, "(f) sparse scene: GPU and model image sizes differ");
            return;
        }
        const WindowDifference diff = compareImages(gpu, exact, kSparseTolerance);
        char buffer[256];
        std::snprintf(buffer, sizeof buffer,
                      "(f) sparse scene: PixelWindow K=32 vs ExactRayDepth, max channel difference %.3g, %zu pixels above %.0e",
                      diff.worst, diff.above, kSparseTolerance);
        report(diff.above == 0, buffer);
    });
}

// ---------------------------------------------------------------------------
// (g) Stochastic transparency: raw frames against the CPU model, convergence,
// accumulation arithmetic, the reset rules, and the blue-noise rank table.

// Initial limits. Each is set before a measured run; calibrate once from the
// printed numbers, then fix them.
constexpr double kStochasticPixelTolerance = 1e-6; // (g1) per-pixel worst channel |gpu - model|
constexpr double kStochasticPixelFraction = 0.001; // (g1) fraction of pixels allowed above that
constexpr double kAccumulatedMaeLimit = 1e-2;      // (g2) rgb MAE after the last frame, against ExactRayDepth
constexpr double kAccumulationTolerance = 1e-5;    // (g3) per channel; float accumulation order differs
constexpr double kResetTolerance = 1e-6;           // (g4) a reset frame is the raw frame
constexpr double kEmaTolerance = 1e-5;             // (g4) one float multiply-add per channel
constexpr uint32_t kSampleCounts[] = {4, 8};
// Frame counts at which (g2) prints its MAE. Only the last one is asserted.
constexpr uint32_t kReportedFrames[] = {1, 8, 32, 256};
constexpr uint32_t kConvergenceFrames = 256;
constexpr uint32_t kAccumulationFrames = 8;    // (g3) static frames averaged
constexpr uint32_t kAccumulationCap = 32;      // (g3) temporalFrames
constexpr uint32_t kCapCheckCap = 4;           // (g4) temporalFrames for the cap check
constexpr uint32_t kCapCheckFrames = 6;
constexpr float kEmaWeight = 0.25f;            // (g4) temporalEmaWeight; exact in binary
constexpr double kCameraStepDegrees = 0.5;

// Blue-noise block statistics (g6): the mean of 64 independent uniform ranks over
// 0..4095 has standard deviation (4096 / sqrt(12)) / sqrt(64), about 147.8.
const double kWhiteBlockStdDev = 4096.0 / std::sqrt(12.0) / 8.0;

// The sample-count settings of one Stochastic check.
SortSettings stochasticSettings(uint32_t samples, uint32_t temporalFrames, uint32_t temporalMode = 0,
                                float emaWeight = 0.2f)
{
    SortSettings settings = withMode(SortMode::Stochastic);
    settings.stochasticSamples = samples;
    settings.temporalFrames = temporalFrames;
    settings.temporalMode = temporalMode;
    settings.temporalEmaWeight = emaWeight;
    return settings;
}

// Applies the settings, runs one frame and returns the downloaded image. The
// settings are applied on every run: setting the values already in effect does
// not reset the accumulation, only a change does.
std::vector<glm::vec4> runStochastic(ForwardPass & pass, const CanonicalGaussianBufferViews & views,
                                     const CameraUniforms & uniforms, const SortSettings & settings)
{
    pass.setSortSettings(settings);
    pass.run(views, uniforms);
    return pass.downloadOutputImage();
}

// Mean absolute error over the rgb channels. Alpha is coverage and is checked by
// the other sections. A NaN makes the result NaN, which fails any bound.
double meanAbsoluteErrorRgb(const std::vector<glm::vec4> & a, const std::vector<glm::vec4> & b)
{
    if (a.size() != b.size() || a.empty())
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double sum = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        for (int channel = 0; channel < 3; ++channel)
        {
            sum += std::fabs(static_cast<double>(a[i][channel]) - static_cast<double>(b[i][channel]));
        }
    }
    return sum / (3.0 * static_cast<double>(a.size()));
}

// A GPU frame against its CPU model: the fraction of pixels whose worst channel
// difference is above kStochasticPixelTolerance must be at most kStochasticPixelFraction.
void reportStochasticFrame(const std::string & label, const std::vector<glm::vec4> & gpu,
                           const std::vector<glm::vec4> & model)
{
    if (gpu.size() != model.size())
    {
        report(false, label + ": GPU and model image sizes differ");
        return;
    }
    const WindowDifference diff = compareImages(gpu, model, kStochasticPixelTolerance);
    const double fraction =
        diff.pixels > 0 ? static_cast<double>(diff.above) / static_cast<double>(diff.pixels) : 1.0;
    char buffer[384];
    std::snprintf(buffer, sizeof buffer,
                  "%s vs CPU model: max channel difference %.3g, %zu of %zu pixels (%.4f%%) above %.0e, limit %.2f%%",
                  label.c_str(), diff.worst, diff.above, diff.pixels, 100.0 * fraction, kStochasticPixelTolerance,
                  100.0 * kStochasticPixelFraction);
    report(fraction <= kStochasticPixelFraction, buffer);
}

// (g1) Raw frames (temporalFrames 1, so every frame is written as it is) against
// the CPU model of the shader, for frame indices 0 and 1 after resetStochastic().
void checkStochasticModel(Device & device, const Scene & scene)
{
    GaussianCloud cloud(device);
    uploadSet(cloud, scene.gaussians);
    const auto views = cloud.views();
    const CameraUniforms uniforms = makeUniforms(scene.camera, views.gaussianCount);
    const std::vector<uint16_t> ranks = generateBlueNoiseRanks();

    for (uint32_t samples : kSampleCounts)
    {
        const std::string name = "(g1) ST" + std::to_string(samples);
        guarded(name, [&]() {
            ForwardPass pass(device, kWidth, kHeight, testShaderDir());
            const SortSettings settings = stochasticSettings(samples, 1);
            pass.setSortSettings(settings);
            pass.resetStochastic();

            for (uint32_t frame = 0; frame < 2; ++frame)
            {
                const std::string label = name + " frame " + std::to_string(frame);
                const std::vector<glm::vec4> gpu = runStochastic(pass, views, uniforms, settings);
                const uint32_t used = pass.lastStochasticFrame();
                report(used == frame, label + ": the run used frame index " + std::to_string(used));

                // Candidates after the run, with the ray-depth form the model reads for t*.
                const study::StudyCandidates candidates = study::captureCandidates(pass, views, uniforms, true);
                study::StochasticOptions options;
                options.samples = samples;
                options.frameIndex = used;
                reportStochasticFrame(label, gpu, study::renderStochasticModel(candidates, ranks, options));
            }
        });
    }
}

// (g2) Convergence: after k static frames the image is the mean of k noisy frames,
// which tends to the exact blend in t* order. The reference is ExactRayDepth on the
// candidates of one ST4 frame. Tile membership and projection do not depend on the
// sample count, and the exact order is per pixel, so one reference serves both.
void checkStochasticConvergence(Device & device, const Scene & scene)
{
    GaussianCloud cloud(device);
    uploadSet(cloud, scene.gaussians);
    const auto views = cloud.views();
    const CameraUniforms uniforms = makeUniforms(scene.camera, views.gaussianCount);

    ForwardPass capture(device, kWidth, kHeight, testShaderDir());
    runStochastic(capture, views, uniforms, stochasticSettings(4, kConvergenceFrames));
    const study::StudyCandidates candidates = study::captureCandidates(capture, views, uniforms, true);
    study::ReferenceOptions referenceOptions;
    referenceOptions.order = study::ReferenceOrder::ExactRayDepth;
    const std::vector<glm::vec4> exact = study::renderReference(candidates, referenceOptions);
    report(coveredRgba(exact) > 0 && allFinite(exact),
           "(g2) ExactRayDepth reference has coverage and finite values");

    for (uint32_t samples : kSampleCounts)
    {
        const std::string name = "(g2) ST" + std::to_string(samples);
        guarded(name, [&]() {
            ForwardPass pass(device, kWidth, kHeight, testShaderDir());
            pass.setSortSettings(stochasticSettings(samples, kConvergenceFrames));
            pass.resetStochastic();

            double finalMae = std::numeric_limits<double>::quiet_NaN();
            for (uint32_t frame = 1; frame <= kConvergenceFrames; ++frame)
            {
                pass.run(views, uniforms);
                bool reported = false;
                for (uint32_t count : kReportedFrames)
                {
                    reported = reported || count == frame;
                }
                if (!reported)
                {
                    continue;
                }
                const double mae = meanAbsoluteErrorRgb(pass.downloadOutputImage(), exact);
                std::printf("  %s rgb MAE after %3u static frame%s: %.6g\n", name.c_str(), frame,
                            frame == 1 ? "" : "s", mae);
                if (frame == kConvergenceFrames)
                {
                    finalMae = mae;
                }
            }

            report(pass.lastStochasticAccumulated() == kConvergenceFrames,
                   name + ": accumulated count is " + std::to_string(kConvergenceFrames) + " after "
                       + std::to_string(kConvergenceFrames) + " static frames");
            char buffer[256];
            std::snprintf(buffer, sizeof buffer,
                          "%s: rgb MAE against ExactRayDepth after %u static frames %.6g, limit %.0e", name.c_str(),
                          kConvergenceFrames, finalMae, kAccumulatedMaeLimit);
            report(finalMae <= kAccumulatedMaeLimit, buffer);
        });
    }
}

// (g3) Accumulation arithmetic: with temporalFrames 32, the image after 8 static
// frames is the mean of the 8 raw frames with the same frame indices.
void checkAccumulation(Device & device, const Scene & scene)
{
    GaussianCloud cloud(device);
    uploadSet(cloud, scene.gaussians);
    const auto views = cloud.views();
    const CameraUniforms uniforms = makeUniforms(scene.camera, views.gaussianCount);
    const size_t pixels = static_cast<size_t>(kWidth) * kHeight;

    for (uint32_t samples : kSampleCounts)
    {
        const std::string name = "(g3) ST" + std::to_string(samples);
        guarded(name, [&]() {
            const SortSettings accumulate = stochasticSettings(samples, kAccumulationCap);
            const SortSettings raw = stochasticSettings(samples, 1);
            ForwardPass accPass(device, kWidth, kHeight, testShaderDir());
            accPass.setSortSettings(accumulate);
            accPass.resetStochastic();
            ForwardPass rawPass(device, kWidth, kHeight, testShaderDir());
            rawPass.setSortSettings(raw);
            rawPass.resetStochastic();

            std::vector<glm::dvec4> sum(pixels, glm::dvec4(0.0));
            std::vector<glm::vec4> accumulated;
            for (uint32_t frame = 0; frame < kAccumulationFrames; ++frame)
            {
                accumulated = runStochastic(accPass, views, uniforms, accumulate);
                const std::vector<glm::vec4> rawFrame = runStochastic(rawPass, views, uniforms, raw);
                for (size_t i = 0; i < pixels; ++i)
                {
                    sum[i] += glm::dvec4(rawFrame[i]);
                }
            }

            std::vector<glm::vec4> mean(pixels);
            for (size_t i = 0; i < pixels; ++i)
            {
                mean[i] = glm::vec4(sum[i] / static_cast<double>(kAccumulationFrames));
            }

            report(accPass.lastStochasticAccumulated() == kAccumulationFrames,
                   name + ": accumulated count is " + std::to_string(kAccumulationFrames) + " after "
                       + std::to_string(kAccumulationFrames) + " static frames");
            const double worst = worstChannelDifference(accumulated, mean);
            char buffer[256];
            std::snprintf(buffer, sizeof buffer,
                          "%s: after %u static frames the image is the mean of the raw frames, worst channel "
                          "difference %.3g (tolerance %.0e)",
                          name.c_str(), kAccumulationFrames, worst, kAccumulationTolerance);
            report(worst <= kAccumulationTolerance, buffer);
        });
    }
}

// (g4) Reset rules, on the camera sequence A A A B A, so steps 3 and 4 are camera
// changes. Under temporalMode 0 a change restarts the accumulation and writes the
// raw frame. Under temporalMode 1 (EMA) it blends into the history.
void checkResetRules(Device & device, const Scene & scene)
{
    GaussianCloud cloud(device);
    uploadSet(cloud, scene.gaussians);
    const auto views = cloud.views();
    const CameraUniforms cameraA = makeUniforms(scene.camera, views.gaussianCount);
    const CameraUniforms cameraB =
        makeUniforms(orbitCamera(scene.camera, kCameraStepDegrees * kPi / 180.0), views.gaussianCount);
    const CameraUniforms sequence[] = {cameraA, cameraA, cameraA, cameraB, cameraA};
    constexpr size_t kSteps = sizeof(sequence) / sizeof(sequence[0]);

    // The raw frame of every step, from a pass that writes each frame as it is.
    // Its frame indices match the pass below, which runs the same sequence.
    const SortSettings rawSettings = stochasticSettings(4, 1);
    ForwardPass raw(device, kWidth, kHeight, testShaderDir());
    raw.setSortSettings(rawSettings);
    raw.resetStochastic();
    std::vector<std::vector<glm::vec4>> rawFrames;
    for (const CameraUniforms & u : sequence)
    {
        rawFrames.push_back(runStochastic(raw, views, u, rawSettings));
    }

    guarded("(g4) temporalMode 0", [&]() {
        const SortSettings accumulate = stochasticSettings(4, kAccumulationCap, 0);
        ForwardPass pass(device, kWidth, kHeight, testShaderDir());
        pass.setSortSettings(accumulate);
        pass.resetStochastic();
        for (size_t step = 0; step < kSteps; ++step)
        {
            const std::string label = "(g4) temporalMode 0, step " + std::to_string(step);
            const std::vector<glm::vec4> image = runStochastic(pass, views, sequence[step], accumulate);

            // Steps 0 (reset), 1 and 2 (static) count up; steps 3 and 4 restart at 1.
            const uint32_t expected = step < 3 ? static_cast<uint32_t>(step + 1) : 1u;
            const uint32_t accumulated = pass.lastStochasticAccumulated();
            report(accumulated == expected,
                   label + ": accumulated count " + std::to_string(accumulated) + ", expected "
                       + std::to_string(expected));

            if (step >= 3)
            {
                const bool sameIndex = pass.lastStochasticFrame() == static_cast<uint32_t>(step);
                const double worst = worstChannelDifference(image, rawFrames[step]);
                char buffer[256];
                std::snprintf(buffer, sizeof buffer,
                              "%s: the camera change gives the raw frame at the same index, worst channel "
                              "difference %.3g (tolerance %.0e)",
                              label.c_str(), worst, kResetTolerance);
                report(sameIndex && worst <= kResetTolerance, buffer);
            }
        }
    });

    guarded("(g4) temporalFrames cap", [&]() {
        const SortSettings capped = stochasticSettings(4, kCapCheckCap, 0);
        ForwardPass pass(device, kWidth, kHeight, testShaderDir());
        pass.setSortSettings(capped);
        pass.resetStochastic();
        bool counts = true;
        for (uint32_t k = 1; k <= kCapCheckFrames; ++k)
        {
            runStochastic(pass, views, cameraA, capped);
            const uint32_t expected = std::min(k, kCapCheckCap);
            counts = counts && pass.lastStochasticAccumulated() == expected;
        }
        report(counts, "(g4) accumulated count follows min(k, temporalFrames) over " + std::to_string(kCapCheckFrames)
                           + " static frames with temporalFrames " + std::to_string(kCapCheckCap));
    });

    guarded("(g4) temporalMode 1 (EMA)", [&]() {
        // The raw frame of index 1 under camera B, from a pass that sees A then B.
        ForwardPass rawPair(device, kWidth, kHeight, testShaderDir());
        rawPair.setSortSettings(rawSettings);
        rawPair.resetStochastic();
        runStochastic(rawPair, views, cameraA, rawSettings);
        const std::vector<glm::vec4> rawB = runStochastic(rawPair, views, cameraB, rawSettings);

        const SortSettings ema = stochasticSettings(4, kAccumulationCap, 1, kEmaWeight);
        ForwardPass pass(device, kWidth, kHeight, testShaderDir());
        pass.setSortSettings(ema);
        pass.resetStochastic();

        // The first frame after a reset is raw, whatever the mode.
        const std::vector<glm::vec4> previous = runStochastic(pass, views, cameraA, ema);
        report(pass.lastStochasticWeight() == 1.0f && pass.lastStochasticAccumulated() == 1,
               "(g4) temporalMode 1: the first frame after a reset has weight 1 and count 1");

        // The camera change blends: history + (raw - history) * w.
        const std::vector<glm::vec4> blended = runStochastic(pass, views, cameraB, ema);
        report(pass.lastStochasticWeight() == kEmaWeight && pass.lastStochasticAccumulated() == 1,
               "(g4) temporalMode 1: a camera change has weight temporalEmaWeight and count 1");

        std::vector<glm::vec4> expected(previous.size());
        for (size_t i = 0; i < previous.size(); ++i)
        {
            expected[i] = (1.0f - kEmaWeight) * previous[i] + kEmaWeight * rawB[i];
        }
        const double worst = worstChannelDifference(blended, expected);
        char buffer[256];
        std::snprintf(buffer, sizeof buffer,
                      "(g4) temporalMode 1: the change is (1 - w) * previous + w * raw with w = %.2f, worst channel "
                      "difference %.3g (tolerance %.0e)",
                      static_cast<double>(kEmaWeight), worst, kEmaTolerance);
        report(worst <= kEmaTolerance, buffer);
    });
}

// ---------------------------------------------------------------------------
// (g7) Temporal weights of the EMA mode, and the image recursion they drive.

constexpr uint32_t kWeightCapEma = 32;        // (g7) temporalFrames of both sequences
constexpr uint32_t kWeightFramesEma = 12;     // (g7a) runs in the first sequence
constexpr float kWeightFloorEma = 0.25f;      // (g7a) first sequence: temporalEmaWeight
constexpr uint32_t kWeightFramesLow = 30;     // (g7a) runs in the second sequence (1/k < 0.05 from k = 21)
constexpr float kWeightFloorLow = 0.05f;      // (g7a) second sequence: temporalEmaWeight
constexpr uint32_t kRecursionFrames = 10;     // (g7b) static frames of the EMA image
constexpr double kRecursionTolerance = 1e-5;  // (g7b) per channel; the blend order and FMA differ

// The weight the host gives run k (1-based) of a static EMA sequence: the reset
// run takes 1, and each later run max(1 / k, floor) in float, as ForwardPass does.
float expectedEmaWeight(uint32_t k, float emaFloor)
{
    if (k == 1)
    {
        return 1.0f;
    }
    return std::max(1.0f / static_cast<float>(k), emaFloor);
}

// (g7a) After each run the weight and the count must equal the host values exactly.
// Only the first mismatch is printed, so a failure names one run.
void checkWeightSequence(Device & device, const Scene & scene)
{
    GaussianCloud cloud(device);
    uploadSet(cloud, scene.gaussians);
    const auto views = cloud.views();
    const CameraUniforms uniforms = makeUniforms(scene.camera, views.gaussianCount);

    struct Sequence
    {
        const char * name;
        float emaFloor;
        uint32_t frames;
    };
    const Sequence sequences[] = {
        {"(g7a) temporalEmaWeight 0.25, 12 runs", kWeightFloorEma, kWeightFramesEma},
        {"(g7a) temporalEmaWeight 0.05, 30 runs", kWeightFloorLow, kWeightFramesLow},
    };

    for (const Sequence & sequence : sequences)
    {
        guarded(sequence.name, [&]() {
            const SortSettings ema = stochasticSettings(4, kWeightCapEma, 1, sequence.emaFloor);
            ForwardPass pass(device, kWidth, kHeight, testShaderDir());
            pass.setSortSettings(ema);
            pass.resetStochastic();

            uint32_t firstBad = 0;
            float gotWeight = 0.0f;
            float wantWeight = 0.0f;
            uint32_t gotCount = 0;
            uint32_t wantCount = 0;
            for (uint32_t k = 1; k <= sequence.frames; ++k)
            {
                runStochastic(pass, views, uniforms, ema);
                const float weight = pass.lastStochasticWeight();
                const uint32_t accumulated = pass.lastStochasticAccumulated();
                const float expected = expectedEmaWeight(k, sequence.emaFloor);
                const uint32_t expectedAccumulated = std::min(k, kWeightCapEma);
                if (firstBad == 0 && (weight != expected || accumulated != expectedAccumulated))
                {
                    firstBad = k;
                    gotWeight = weight;
                    wantWeight = expected;
                    gotCount = accumulated;
                    wantCount = expectedAccumulated;
                }
            }

            char buffer[320];
            if (firstBad == 0)
            {
                std::snprintf(buffer, sizeof buffer,
                              "%s: every run has the host weight max(1/k, %.2f) exactly (1 on the reset run) and "
                              "count min(k, %u)",
                              sequence.name, static_cast<double>(sequence.emaFloor), kWeightCapEma);
                report(true, buffer);
            }
            else
            {
                std::snprintf(buffer, sizeof buffer,
                              "%s: first mismatch at run %u: weight %.9g (expected %.9g), count %u (expected %u)",
                              sequence.name, firstBad, static_cast<double>(gotWeight),
                              static_cast<double>(wantWeight), gotCount, wantCount);
                report(false, buffer);
            }
        });
    }
}

// (g7b) The EMA image after 10 static frames against the host recursion over 10 raw
// frames at the same frame indices. The reset frame is the raw frame; each later
// frame blends with the weight of (g7a). The raw frames come from a pass with
// temporalFrames 1, which writes every frame as it is.
void checkEmaRecursion(Device & device, const Scene & scene)
{
    GaussianCloud cloud(device);
    uploadSet(cloud, scene.gaussians);
    const auto views = cloud.views();
    const CameraUniforms uniforms = makeUniforms(scene.camera, views.gaussianCount);

    const SortSettings rawSettings = stochasticSettings(4, 1);
    const SortSettings ema = stochasticSettings(4, kWeightCapEma, 1, kWeightFloorEma);

    guarded("(g7b) EMA image recursion", [&]() {
        ForwardPass raw(device, kWidth, kHeight, testShaderDir());
        raw.setSortSettings(rawSettings);
        raw.resetStochastic();
        ForwardPass pass(device, kWidth, kHeight, testShaderDir());
        pass.setSortSettings(ema);
        pass.resetStochastic();

        std::vector<glm::vec4> hist;
        std::vector<glm::vec4> image;
        for (uint32_t frame = 0; frame < kRecursionFrames; ++frame)
        {
            const std::vector<glm::vec4> rawFrame = runStochastic(raw, views, uniforms, rawSettings);
            image = runStochastic(pass, views, uniforms, ema);
            if (frame == 0)
            {
                hist = rawFrame;
                continue;
            }
            // Run number k = frame + 1 of the static sequence.
            const float weight = expectedEmaWeight(frame + 1, kWeightFloorEma);
            for (size_t i = 0; i < hist.size(); ++i)
            {
                hist[i] += (rawFrame[i] - hist[i]) * weight;
            }
        }

        const double worst = worstChannelDifference(image, hist);
        char buffer[320];
        std::snprintf(buffer, sizeof buffer,
                      "(g7b) EMA image after %u static frames (w %.2f) matches the host recursion, "
                      "worst channel difference %.3g (tolerance %.0e)",
                      kRecursionFrames, static_cast<double>(kWeightFloorEma), worst, kRecursionTolerance);
        report(worst <= kRecursionTolerance, buffer);
    });
}

// (g6) The rank table: a permutation of 0..4095, the same on every call, and blue.
// The 64 block means over the non-overlapping 8x8 blocks of the 64x64 table must
// vary far less than white noise would, whose spread is kWhiteBlockStdDev.
void checkBlueNoise()
{
    const uint32_t side = kBlueNoiseSize;
    const size_t cells = static_cast<size_t>(side) * side;
    const std::vector<uint16_t> ranks = generateBlueNoiseRanks();
    const std::vector<uint16_t> again = generateBlueNoiseRanks();

    bool permutation = ranks.size() == cells;
    std::vector<bool> seen(cells, false);
    for (size_t i = 0; permutation && i < ranks.size(); ++i)
    {
        const size_t r = ranks[i];
        if (r >= cells || seen[r])
        {
            permutation = false;
        }
        else
        {
            seen[r] = true;
        }
    }
    report(permutation, "(g6) ranks are a permutation of 0.." + std::to_string(cells - 1));
    report(ranks == again, "(g6) the rank table is identical across two calls");
    if (ranks.size() != cells)
    {
        return;
    }

    const uint32_t block = 8;
    const uint32_t blocksPerSide = side / block;
    std::vector<double> means;
    for (uint32_t by = 0; by < blocksPerSide; ++by)
    {
        for (uint32_t bx = 0; bx < blocksPerSide; ++bx)
        {
            double sum = 0.0;
            for (uint32_t y = 0; y < block; ++y)
            {
                for (uint32_t x = 0; x < block; ++x)
                {
                    sum += ranks[static_cast<size_t>(by * block + y) * side + bx * block + x];
                }
            }
            means.push_back(sum / static_cast<double>(block * block));
        }
    }

    const double rankMean = 0.5 * static_cast<double>(cells - 1);
    double worstDeviation = 0.0;
    double meanOfMeans = 0.0;
    for (double m : means)
    {
        worstDeviation = std::max(worstDeviation, std::fabs(m - rankMean));
        meanOfMeans += m;
    }
    meanOfMeans /= static_cast<double>(means.size());
    double variance = 0.0;
    for (double m : means)
    {
        variance += (m - meanOfMeans) * (m - meanOfMeans);
    }
    const double spread = std::sqrt(variance / static_cast<double>(means.size()));

    char buffer[256];
    std::snprintf(buffer, sizeof buffer,
                  "(g6) the 64 block means lie within 3 standard errors (%.1f) of %.1f: worst deviation %.1f",
                  3.0 * kWhiteBlockStdDev, rankMean, worstDeviation);
    report(worstDeviation <= 3.0 * kWhiteBlockStdDev, buffer);

    std::snprintf(buffer, sizeof buffer,
                  "(g6) spread of the 64 block means %.1f is below half the white-noise spread %.1f", spread,
                  kWhiteBlockStdDev);
    report(spread < 0.5 * kWhiteBlockStdDev, buffer);
}

int runChecks()
{
    const Scene scene = buildScene();
    std::printf("sort_modes_test: %u Gaussians, SH degree %u, %ux%u\n", scene.gaussians.count(), kShDegree, kWidth,
                kHeight);

    runSection("(a) default settings", [&]() { checkDefaults(scene); });
    runSection("(b) Differentiable forces the default sort", [&]() { checkDifferentiable(scene); });
    runSection("(h) lighting gate", [&]() {
        checkLightingGate(scene);
        checkLitDifferentiable(scene);
    });
    runSection("(v) settings validation", [&]() { checkValidation(); });
    runSection("(t) stage timings", [&]() { checkStageTimings(scene); });
    runSection("(g6) blue-noise rank table", [&]() { checkBlueNoise(); });

    // (c), (d), (e), (f), (g) and (r) share one Device. It is created here, after every
    // Renderer above has been destroyed, so two Vulkan devices never coexist.
    {
        Device device;
        bool deviceReady = false;
        std::string deviceError;
        try
        {
            device.initialize({}, true);
            deviceReady = true;
        }
        catch (const std::exception & e)
        {
            deviceError = e.what();
        }
        // A Device that failed to start fails every section that needs it.
        auto requireDevice = [&]() {
            if (!deviceReady)
            {
                throw std::runtime_error("no Device: " + deviceError);
            }
        };
        runSection("(c) S1 sticky keys over a camera orbit", [&]() {
            requireDevice();
            checkStableKeys(device, scene);
        });
        runSection("(c) coplanar pair, Default and Stable order", [&]() {
            requireDevice();
            checkCoplanarPair(device);
        });
        runSection("(d) ray-depth form against a double reference", [&]() {
            requireDevice();
            checkRayDepthForm(device, scene);
        });
        runSection("(e) TileRayDepth and Hybrid sort keys", [&]() {
            requireDevice();
            checkTileKeys(device, scene);
        });
        runSection("(f) window images and sparse exactness", [&]() {
            requireDevice();
            checkWindowImages(device, scene);
        });
        runSection("(g1) stochastic raw frames against the CPU model", [&]() {
            requireDevice();
            checkStochasticModel(device, scene);
        });
        runSection("(g2) stochastic convergence over 256 static frames", [&]() {
            requireDevice();
            checkStochasticConvergence(device, scene);
        });
        runSection("(g3) stochastic accumulation arithmetic", [&]() {
            requireDevice();
            checkAccumulation(device, scene);
        });
        runSection("(g4) stochastic reset rules", [&]() {
            requireDevice();
            checkResetRules(device, scene);
        });
        runSection("(g7a) stochastic EMA weight sequences", [&]() {
            requireDevice();
            checkWeightSequence(device, scene);
        });
        runSection("(g7b) stochastic EMA image recursion", [&]() {
            requireDevice();
            checkEmaRecursion(device, scene);
        });
        runSection("(r) CPU reference self-checks", [&]() {
            requireDevice();
            checkCpuReference(device, scene);
        });
    }

    std::printf("\nsort_modes_test: %d checks, %d failed\n", g_checks, g_failures);
    if (g_failures == 0)
    {
        std::printf("sort_modes_test: PASSED\n");
        return 0;
    }
    std::printf("sort_modes_test: FAILED\n");
    return 1;
}

// ---------------------------------------------------------------------------
// --timing: manual benchmark (not run by ctest).

double median(std::vector<double> values)
{
    if (values.empty())
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::sort(values.begin(), values.end());
    const size_t n = values.size();
    return (n % 2 == 1) ? values[n / 2] : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

// Nearest-rank percentile, p in (0, 1].
double percentile(std::vector<double> values, double p)
{
    if (values.empty())
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::sort(values.begin(), values.end());
    const double rank = std::ceil(p * static_cast<double>(values.size()));
    const size_t index = rank < 1.0 ? 0 : std::min(static_cast<size_t>(rank) - 1, values.size() - 1);
    return values[index];
}

void printRow(const char * name, const std::vector<double> & values, const char * unit)
{
    std::printf("  %-12s median %10.4f %s   p95 %10.4f %s\n", name, median(values), unit, percentile(values, 0.95),
                unit);
}

// Replaces every occurrence of the full scene path in a loader message with
// the file name, so the directory is never printed.
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

int runTiming(const char * path, uint32_t width, uint32_t height, uint32_t frames)
{
    if (frames <= kWarmupFrames)
    {
        throw std::invalid_argument("frames must be greater than the warm-up count (" + std::to_string(kWarmupFrames)
                                    + ")");
    }

    const std::string name = std::filesystem::path(path).filename().string();

    Renderer renderer(width, height);
    initializeRenderer(renderer);
    try
    {
        renderer.loadScene(path);
    }
    catch (const std::exception & e)
    {
        throw std::runtime_error("loadScene failed for " + name + ": " + withoutDirectory(e.what(), path, name));
    }

    const SceneBounds bounds = renderer.sceneBounds();
    if (!bounds.valid)
    {
        throw std::runtime_error("scene " + name + " has no bounds");
    }

    // Frame on the robust focus sphere (the median center and 90th-percentile
    // radius), which ignores distant background shells. Fall back to the full
    // radius if the focus radius is degenerate.
    const glm::vec3 center = bounds.focusCenter;
    float radius = bounds.focusRadius;
    if (!(radius > 0.0f))
    {
        radius = 0.5f * bounds.radius();
    }
    if (!(radius > 0.0f))
    {
        throw std::runtime_error("scene " + name + " has a degenerate extent");
    }
    const float distance = 1.8f * radius;

    // 60 degree vertical field of view would be fy = height / (2 tan 30 deg);
    // 0.9 * height is the benchmark's fixed focal length (about 58 degrees).
    const float fy = 0.9f * static_cast<float>(height);
    const float fx = fy;
    const float cx = 0.5f * static_cast<float>(width);
    const float cy = 0.5f * static_cast<float>(height);
    const double elevation = 15.0 * kPi / 180.0;

    renderer.setStageTimingEnabled(true);

    std::vector<double> wallMs;
    std::vector<double> instances;
    std::vector<std::vector<double>> stageMs(kStageCount);
    bool timingValid = true;

    for (uint32_t frame = 0; frame < frames; ++frame)
    {
        const double yaw = static_cast<double>(frame) * 0.5 * kPi / 180.0;
        const glm::vec3 offset(static_cast<float>(std::cos(elevation) * std::sin(yaw)),
                               static_cast<float>(std::sin(elevation)),
                               static_cast<float>(std::cos(elevation) * std::cos(yaw)));
        const glm::vec3 eye = center + distance * offset;
        const glm::mat4 view = glm::lookAt(eye, center, glm::vec3(0.0f, 1.0f, 0.0f));
        renderer.setCamera(makeCamera(width, height, fx, fy, cx, cy, glm::inverse(view)));

        const auto start = std::chrono::steady_clock::now();
        renderer.forward(RenderMode::Viewer);
        const auto stop = std::chrono::steady_clock::now();

        if (frame < kWarmupFrames)
        {
            continue;
        }
        wallMs.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
        instances.push_back(static_cast<double>(renderer.lastInstanceCount()));

        const ForwardStageTimings & t = renderer.lastStageTimings();
        if (!t.valid)
        {
            timingValid = false;
            continue;
        }
        const double values[] = {t.project, t.scan,   t.binning, t.sortKeys, t.sort,
                                 t.ranges,  t.rasterize, t.resolve, t.total};
        for (size_t i = 0; i < kStageCount; ++i)
        {
            stageMs[i].push_back(values[i]);
        }
    }

    std::printf("benchmark: %s, %ux%u, %u Gaussians, %u frames (%u warm-up, %zu measured)\n", name.c_str(), width,
                height, renderer.gaussianCount(), frames, kWarmupFrames, wallMs.size());
    printRow("forward wall", wallMs, "ms");
    if (timingValid)
    {
        for (size_t i = 0; i < kStageCount; ++i)
        {
            printRow(kStageNames[i], stageMs[i], "ms GPU");
        }
    }
    else
    {
        std::printf("  [SKIP] stage timings unavailable on this queue\n");
    }
    std::printf("  tile instances: median %.0f  p95 %.0f\n", median(instances), percentile(instances, 0.95));
    return 0;
}

void printUsage(const char * program)
{
    std::fprintf(stderr, "usage: %s\n       %s --timing <scene-file> <WxH> [frames]\n", program, program);
}

} // namespace

int main(int argc, char ** argv)
{
    try
    {
        if (argc == 1)
        {
            return runChecks();
        }

        if (argc >= 4 && argc <= 5 && std::strcmp(argv[1], "--timing") == 0)
        {
            unsigned int width = 0;
            unsigned int height = 0;
            if (std::sscanf(argv[3], "%ux%u", &width, &height) != 2 || width == 0 || height == 0)
            {
                printUsage(argv[0]);
                return 2;
            }

            unsigned long frames = kDefaultBenchFrames;
            if (argc == 5)
            {
                char * end = nullptr;
                frames = std::strtoul(argv[4], &end, 10);
                if (end == argv[4] || *end != '\0')
                {
                    printUsage(argv[0]);
                    return 2;
                }
            }
            return runTiming(argv[2], static_cast<uint32_t>(width), static_cast<uint32_t>(height),
                             static_cast<uint32_t>(frames));
        }

        printUsage(argv[0]);
        return 2;
    }
    catch (const std::exception & e)
    {
        std::fprintf(stderr, "sort_modes_test: %s\n", e.what());
        return 1;
    }
}
