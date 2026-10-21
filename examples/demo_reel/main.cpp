// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Feature reel: renders a fixed, deterministic sequence of Sabercat's rendering
// features from one scene, headless (no window, no screen capture), and writes
// each output frame as a binary PPM (P6, 8-bit sRGB) plus a segments.txt index
// and a stats.txt of per-segment numeric sanity values. tools/make_demo.ps1
// turns the frames into the MP4 and GIF.
//
//   sabercat_demo_reel <scene> <outDir> [--width 1280] [--height 720] [--fps 30]
//                      [--max-frames N] [--flip-up]
//
// Segments: orbit; SH degree 0->3; relighting (point light, sky); PCSS shadows;
// ray-traced shadows; baked visibility; debug views; layers. Ray-traced
// segments are skipped with a log line when the device has no ray query.
//
// Captions and durations are fixed in this file; frame times are fixed
// (frame i of a segment is at parameter i / n), so the output is reproducible.
//
// Limits: the debug views TileDensity and SHDegree are presentation-only
// (DebugVisualizer, reached through present() to a surface), so they cannot be
// read back headless and are not in the reel. Ray-traced segments use the
// full emitter grid per frame (no progressive accumulation).

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sabercat/camera.h"
#include "sabercat/device.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/layer_transform.h"
#include "sabercat/lighting.h"
#include "sabercat/logger.h"
#include "sabercat/renderer.h"
#include "sabercat/scene_loader.h"

using namespace sabercat;

namespace
{

namespace fs = std::filesystem;

const glm::vec3 kBackground(0.05f, 0.07f, 0.10f);
constexpr float kPi = 3.14159265358979f;
constexpr float kHalfFovTan = 0.41421356f; // tan(22.5 deg): 45 degree vertical FOV

std::string shaderDir()
{
    const char * env = std::getenv("SABERCAT_SHADER_DIR");
    return env != nullptr && *env != '\0' ? std::string(env) : std::string(SHADER_DIR);
}

// ---- output ---------------------------------------------------------------

// Linear -> 8-bit sRGB through a lookup table over [0, 1].
struct SrgbLut
{
    uint8_t table[4096];
    SrgbLut()
    {
        for (int i = 0; i < 4096; ++i)
        {
            const float x = static_cast<float>(i) / 4095.0f;
            const float s = x <= 0.0031308f ? 12.92f * x : 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
            table[i] = static_cast<uint8_t>(std::lround(std::clamp(s, 0.0f, 1.0f) * 255.0f));
        }
    }
    uint8_t encode(float linear) const
    {
        const float c = std::clamp(linear, 0.0f, 1.0f);
        return table[static_cast<int>(c * 4095.0f + 0.5f)];
    }
};

const SrgbLut & srgb()
{
    static const SrgbLut lut;
    return lut;
}

// How a readback becomes pixels.
enum class Look
{
    Colour,   // rgb premultiplied over the background (readImage), sRGB
    Lit,      // lighting output, already composited over the background, sRGB
    Coverage, // alpha as grayscale, no encoding
};

struct SegmentRecord
{
    std::string caption;
    uint32_t first = 0;
    uint32_t count = 0;
    double sumMean = 0.0;
    double maxY = 0.0;
    double sumFrac = 0.0;
    double minFrac = 1.0;
    double renderMs = 0.0;
};

class Recorder
{
  public:
    Recorder(std::string outDir, uint32_t width, uint32_t height, uint32_t maxFrames)
        : m_outDir(std::move(outDir)), m_width(width), m_height(height), m_maxFrames(maxFrames)
    {
    }

    bool full() const
    {
        return m_maxFrames != 0 && m_frame >= m_maxFrames;
    }

    void beginSegment(const std::string & caption)
    {
        m_segments.push_back(SegmentRecord{});
        m_segments.back().caption = caption;
        m_segments.back().first = m_frame;
        m_t0 = std::chrono::steady_clock::now();
        std::printf("segment %zu: %s\n", m_segments.size(), caption.c_str());
    }

    void endSegment()
    {
        SegmentRecord & s = m_segments.back();
        s.renderMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - m_t0).count();
        std::printf("  frames %u..%u (%u), %.0f ms\n", s.first, m_frame ? m_frame - 1 : 0, s.count, s.renderMs);
    }

    // Writes one frame and folds its statistics into the current segment.
    void emit(const std::vector<float> & rgba, Look look)
    {
        const size_t pixels = static_cast<size_t>(m_width) * m_height;
        const glm::vec3 bg = look == Look::Coverage ? glm::vec3(0.0f) : kBackground;
        const uint8_t bg8[3] = {srgb().encode(bg.r), srgb().encode(bg.g), srgb().encode(bg.b)};

        std::vector<uint8_t> rgb(pixels * 3);
        for (size_t p = 0; p < pixels; ++p)
        {
            const float a = rgba[p * 4 + 3];
            for (int c = 0; c < 3; ++c)
            {
                float v = rgba[p * 4 + c];
                if (look == Look::Colour)
                {
                    v += (1.0f - a) * bg[c];
                }
                if (look == Look::Coverage)
                {
                    v = a;
                    rgb[p * 3 + c] = static_cast<uint8_t>(std::lround(std::clamp(v, 0.0f, 1.0f) * 255.0f));
                    continue;
                }
                rgb[p * 3 + c] = srgb().encode(v);
            }
        }

        // Statistics on the 8-bit output: sRGB-domain luma, and the fraction of
        // pixels that differ from the background colour.
        double sumY = 0.0, maxY = 0.0;
        size_t nonBg = 0;
        for (size_t p = 0; p < pixels; ++p)
        {
            const uint8_t * px = &rgb[p * 3];
            const double y = (0.2126 * px[0] + 0.7152 * px[1] + 0.0722 * px[2]) / 255.0;
            sumY += y;
            maxY = std::max(maxY, y);
            const int diff = std::max({std::abs(px[0] - bg8[0]), std::abs(px[1] - bg8[1]), std::abs(px[2] - bg8[2])});
            if (diff > 2)
            {
                ++nonBg;
            }
        }
        const double mean = sumY / static_cast<double>(pixels);
        const double frac = static_cast<double>(nonBg) / static_cast<double>(pixels);
        SegmentRecord & s = m_segments.back();
        s.sumMean += mean;
        s.maxY = std::max(s.maxY, maxY);
        s.sumFrac += frac;
        s.minFrac = std::min(s.minFrac, frac);
        ++s.count;

        char name[64];
        std::snprintf(name, sizeof(name), "frame_%05u.ppm", m_frame);
        const std::string path = (fs::path(m_outDir) / name).string();
        std::FILE * f = std::fopen(path.c_str(), "wb");
        if (f == nullptr)
        {
            throw std::runtime_error("cannot write " + path);
        }
        std::fprintf(f, "P6\n%u %u\n255\n", m_width, m_height);
        std::fwrite(rgb.data(), 1, rgb.size(), f);
        std::fclose(f);
        ++m_frame;
    }

    void writeIndex() const
    {
        std::ofstream seg(fs::path(m_outDir) / "segments.txt");
        std::ofstream stats(fs::path(m_outDir) / "stats.txt");
        stats << "# caption\tframes\tmean_luma_8bit\tmax_luma\tmin_nonbg_frac\tmean_nonbg_frac\trender_ms\n";
        for (const SegmentRecord & s : m_segments)
        {
            if (s.count == 0)
            {
                continue;
            }
            seg << s.first << '\t' << s.count << '\t' << s.caption << '\n';
            stats << s.caption << '\t' << s.count << '\t' << (s.sumMean / s.count * 255.0) << '\t' << s.maxY << '\t'
                  << s.minFrac << '\t' << (s.sumFrac / s.count) << '\t' << s.renderMs << '\n';
        }
    }

    uint32_t frames() const
    {
        return m_frame;
    }
    double totalRenderMs() const
    {
        double t = 0.0;
        for (const SegmentRecord & s : m_segments)
        {
            t += s.renderMs;
        }
        return t;
    }

  private:
    std::string m_outDir;
    uint32_t m_width;
    uint32_t m_height;
    uint32_t m_maxFrames;
    uint32_t m_frame = 0;
    std::vector<SegmentRecord> m_segments;
    std::chrono::steady_clock::time_point m_t0;
};

// ---- camera and scene -----------------------------------------------------

Camera lookCamera(uint32_t width, uint32_t height, const glm::vec3 & eye, const glm::vec3 & target)
{
    const float fy = static_cast<float>(height) / (2.0f * kHalfFovTan);
    Camera c;
    c.width = width;
    c.height = height;
    c.fx = c.fy = fy;
    c.cx = static_cast<float>(width) * 0.5f;
    c.cy = static_cast<float>(height) * 0.5f;
    c.cameraToWorld = glm::inverse(glm::lookAt(eye, target, glm::vec3(0.0f, 1.0f, 0.0f)));
    return c;
}

// Orbit around `centre` at `radius` (the scene's focus radius) for azimuth phi.
Camera orbitCamera(uint32_t width, uint32_t height, const glm::vec3 & centre, float radius, float phi,
                   float elevation)
{
    // 1.35: the focus radius is the 90th-percentile radius, so the plinth at
    // the bottom of the cactus scene sits outside it and needs the margin.
    const float dist = radius / kHalfFovTan * 1.35f;
    const glm::vec3 dir(std::sin(phi) * std::cos(elevation), std::sin(elevation), std::cos(phi) * std::cos(elevation));
    return lookCamera(width, height, centre + dist * dir, centre);
}

GaussianHostInput inputOf(const GaussianCloud & c)
{
    GaussianHostInput in;
    in.count = static_cast<uint32_t>(c.count());
    in.means = reinterpret_cast<const float *>(c.positions().data());
    in.scales = reinterpret_cast<const float *>(c.scales().data());
    in.rotations = reinterpret_cast<const float *>(c.rotations().data());
    in.opacities = c.opacities().data();
    in.shCoefficients = reinterpret_cast<const float *>(c.shCoefficients().data());
    in.shDegree = c.shDegree();
    in.antialiased = c.antialiased();
    if (!c.normals().empty())
    {
        in.normals = reinterpret_cast<const float *>(c.normals().data());
    }
    if (!c.baseColorRoughness().empty())
    {
        in.baseColorRoughness = reinterpret_cast<const float *>(c.baseColorRoughness().data());
    }
    if (!c.metallic().empty())
    {
        in.metallic = c.metallic().data();
    }
    if (!c.visibility().empty())
    {
        in.visibility = c.visibility().data();
    }
    return in;
}

// Light orbiting the scene: directional light travelling along a sweep angle.
Light sunAt(float angle)
{
    Light l;
    l.type = LightType::Directional;
    l.direction = glm::normalize(glm::vec3(0.8f * std::sin(angle), -1.0f, 0.8f * std::cos(angle)));
    l.intensity = 2.0f;
    l.color = glm::vec3(1.0f, 0.96f, 0.9f);
    return l;
}

EnvironmentLight skyEnvironment(float intensity)
{
    EnvironmentLight env = EnvironmentLight::sky(glm::vec3(0.4f, 0.5f, 0.9f), glm::vec3(0.8f),
                                                 glm::vec3(0.2f, 0.15f, 0.1f), glm::vec3(0.0f, 1.0f, 0.0f));
    env.intensity = intensity;
    return env;
}

// Everything the segments share.
struct Context
{
    Renderer & renderer;
    Recorder & rec;
    Device & device;
    std::shared_ptr<GaussianCloud> cloud;
    uint32_t width;
    uint32_t height;
    uint32_t fps;
    bool rayQuery;
    glm::vec3 centre;
    float radius;

    uint32_t frames(double seconds) const
    {
        return std::max<uint32_t>(1, static_cast<uint32_t>(std::lround(seconds * fps)));
    }
};

LightingSettings lightingSettings(LightingOutput output)
{
    LightingSettings st;
    st.background = kBackground;
    st.exposure = 1.0f;
    st.output = output;
    st.depthScale = 1.0f;
    return st;
}

void renderColour(Context & ctx, const Camera & cam)
{
    ctx.renderer.setCamera(cam);
    ctx.renderer.forward(RenderMode::Viewer);
    ctx.rec.emit(ctx.renderer.readImage(), Look::Colour);
}

void renderLit(Context & ctx, const Camera & cam)
{
    ctx.renderer.setCamera(cam);
    ctx.renderer.forward(RenderMode::Viewer);
    ctx.rec.emit(ctx.renderer.readLitImage(), Look::Lit);
}

// ---- segments -------------------------------------------------------------

void segmentOrbit(Context & ctx)
{
    ctx.rec.beginSegment("Gaussian splats, real time");
    const uint32_t n = ctx.frames(8.0);
    for (uint32_t i = 0; i < n && !ctx.rec.full(); ++i)
    {
        const float t = static_cast<float>(i) / n;
        renderColour(ctx, orbitCamera(ctx.width, ctx.height, ctx.centre, ctx.radius, 2.0f * kPi * t, 0.2f));
    }
    ctx.rec.endSegment();
}

void segmentSH(Context & ctx)
{
    ctx.rec.beginSegment("View-dependent colour (SH degree 0 -> 3)");
    const uint32_t per = ctx.frames(2.0);
    for (uint32_t d = 0; d <= 3 && !ctx.rec.full(); ++d)
    {
        ctx.renderer.setMaxSHDegree(d);
        for (uint32_t i = 0; i < per && !ctx.rec.full(); ++i)
        {
            const float t = static_cast<float>(d * per + i) / (4 * per);
            renderColour(ctx, orbitCamera(ctx.width, ctx.height, ctx.centre, ctx.radius, kPi * 0.25f + 0.5f * kPi * t, 0.2f));
        }
    }
    ctx.renderer.setMaxSHDegree(3);
    ctx.rec.endSegment();
}

void segmentRelight(Context & ctx)
{
    ctx.rec.beginSegment("Relighting");
    ctx.renderer.setLightingEnabled(true);
    ctx.renderer.setLightingSettings(lightingSettings(LightingOutput::Lit));
    const float r = ctx.radius;

    // Point light circling the scene, no environment.
    ctx.renderer.setEnvironment(EnvironmentLight{});
    const uint32_t n1 = ctx.frames(4.0);
    for (uint32_t i = 0; i < n1 && !ctx.rec.full(); ++i)
    {
        const float t = static_cast<float>(i) / n1;
        const float a = 2.0f * kPi * t;
        Light p;
        p.type = LightType::Point;
        p.position = ctx.centre + glm::vec3(1.6f * r * std::cos(a), 0.9f * r, 1.6f * r * std::sin(a));
        p.intensity = (1.6f * r) * (1.6f * r);
        p.color = glm::vec3(1.0f, 0.95f, 0.85f);
        ctx.renderer.setLights({p});
        renderLit(ctx, orbitCamera(ctx.width, ctx.height, ctx.centre, ctx.radius, 0.6f, 0.2f));
    }

    // Sky environment only, camera orbiting a quarter turn.
    ctx.renderer.setLights({});
    ctx.renderer.setEnvironment(skyEnvironment(1.0f));
    const uint32_t n2 = ctx.frames(4.0);
    for (uint32_t i = 0; i < n2 && !ctx.rec.full(); ++i)
    {
        const float t = static_cast<float>(i) / n2;
        renderLit(ctx, orbitCamera(ctx.width, ctx.height, ctx.centre, ctx.radius, 0.6f + 0.5f * kPi * t, 0.2f));
    }

    ctx.renderer.setEnvironment(EnvironmentLight{});
    ctx.renderer.setLights({});
    ctx.renderer.setLightingEnabled(false);
    ctx.rec.endSegment();
}

// Directional light sweep under one shadow mode.
void shadowSweep(Context & ctx, ShadowMode mode, double seconds)
{
    ShadowSettings sh;
    sh.mode = mode;
    sh.raySamplesPerAxis = 4;
    sh.progressiveSamplesPerFrame = 0; // full emitter grid per frame: converged in one pass
    ctx.renderer.setShadowSettings(sh);
    const uint32_t n = ctx.frames(seconds);
    for (uint32_t i = 0; i < n && !ctx.rec.full(); ++i)
    {
        const float t = static_cast<float>(i) / n;
        ctx.renderer.setLights({sunAt(-kPi / 3.0f + (2.0f * kPi / 3.0f) * t)});
        renderLit(ctx, orbitCamera(ctx.width, ctx.height, ctx.centre, ctx.radius, 0.35f, 0.5f));
    }
}

void segmentShadows(Context & ctx)
{
    ctx.renderer.setLightingEnabled(true);
    ctx.renderer.setLightingSettings(lightingSettings(LightingOutput::Lit));
    ctx.renderer.setEnvironment(skyEnvironment(0.3f));

    ctx.rec.beginSegment("Soft shadows (PCSS)");
    shadowSweep(ctx, ShadowMode::PCSS, 3.0);
    ctx.rec.endSegment();

    if (!ctx.rayQuery)
    {
        Logger::warning("demo_reel: device has no ray query, skipping ray-traced shadows");
        std::printf("[SKIP] Ray-traced shadows: device has no ray query\n");
    }
    else
    {
        ctx.rec.beginSegment("Ray-traced shadows");
        shadowSweep(ctx, ShadowMode::RayTraced, 3.0);
        ctx.rec.endSegment();
    }
    ctx.renderer.setShadowSettings(ShadowSettings{});
    ctx.renderer.setEnvironment(EnvironmentLight{});
    ctx.renderer.setLights({});
    ctx.renderer.setLightingEnabled(false);
}

void segmentBaked(Context & ctx)
{
    if (!ctx.rayQuery)
    {
        std::printf("[SKIP] Baked visibility: device has no ray query (the bake traces rays)\n");
        return;
    }
    // Default settings: the bake sizes its submissions to a time budget, so
    // dense scenes stay clear of the driver timeout.
    VisibilityBakeSettings bake;
    const auto t0 = std::chrono::steady_clock::now();
    ctx.renderer.bakeVisibility(bake);
    const double bakeMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("  bakeVisibility: %.0f ms (not part of the reel timing)\n", bakeMs);

    ctx.renderer.setLightingEnabled(true);
    ctx.renderer.setEnvironment(skyEnvironment(0.3f));
    ctx.rec.beginSegment("Baked visibility (ambient occlusion)");
    shadowSweep(ctx, ShadowMode::BakedSH, 3.0);
    ctx.renderer.setLightingSettings(lightingSettings(LightingOutput::AmbientOcclusion));
    const uint32_t n = ctx.frames(2.0);
    for (uint32_t i = 0; i < n && !ctx.rec.full(); ++i)
    {
        const float t = static_cast<float>(i) / n;
        renderLit(ctx, orbitCamera(ctx.width, ctx.height, ctx.centre, ctx.radius, 0.35f + 0.5f * t, 0.5f));
    }
    ctx.rec.endSegment();
    ctx.renderer.setLightingSettings(lightingSettings(LightingOutput::Lit));
    ctx.renderer.setShadowSettings(ShadowSettings{});
    ctx.renderer.setEnvironment(EnvironmentLight{});
    ctx.renderer.setLights({});
    ctx.renderer.setLightingEnabled(false);
}

void segmentDebug(Context & ctx)
{
    ctx.rec.beginSegment("Debug views");
    const uint32_t per = ctx.frames(1.5);
    // Depth: the lighting pass's expected-depth output, scaled into [0, 1].
    ctx.renderer.setLightingEnabled(true);
    LightingSettings depth = lightingSettings(LightingOutput::Depth);
    depth.depthScale = 1.0f / (2.6f * ctx.radius + ctx.radius);
    ctx.renderer.setLightingSettings(depth);
    for (uint32_t i = 0; i < per && !ctx.rec.full(); ++i)
    {
        const float t = static_cast<float>(i) / per;
        renderLit(ctx, orbitCamera(ctx.width, ctx.height, ctx.centre, ctx.radius, 0.6f * t, 0.2f));
    }
    // Alpha: the coverage channel of the colour pass, grayscale.
    for (uint32_t i = 0; i < per && !ctx.rec.full(); ++i)
    {
        const float t = static_cast<float>(i) / per;
        ctx.renderer.setCamera(orbitCamera(ctx.width, ctx.height, ctx.centre, ctx.radius, 0.6f + 0.6f * t, 0.2f));
        ctx.renderer.forward(RenderMode::Viewer);
        ctx.rec.emit(ctx.renderer.readImage(), Look::Coverage);
    }
    // Ambient occlusion: the per-Gaussian visibility splatted to the image.
    ctx.renderer.setLightingSettings(lightingSettings(LightingOutput::AmbientOcclusion));
    for (uint32_t i = 0; i < per && !ctx.rec.full(); ++i)
    {
        const float t = static_cast<float>(i) / per;
        renderLit(ctx, orbitCamera(ctx.width, ctx.height, ctx.centre, ctx.radius, 1.2f + 0.6f * t, 0.2f));
    }
    ctx.renderer.setLightingSettings(lightingSettings(LightingOutput::Lit));
    ctx.renderer.setLightingEnabled(false);
    ctx.rec.endSegment();
}

void segmentLayers(Context & ctx)
{
    ctx.rec.beginSegment("Layers");
    // Two copies side by side; the second one spins, bobs and breathes.
    const float offset = 1.6f * ctx.radius;
    const glm::vec3 centre = ctx.centre + glm::vec3(0.5f * offset, 0.0f, 0.0f);
    const float frameRadius = 2.0f * ctx.radius;
    const uint32_t n = ctx.frames(6.0);
    LayerTransform fixed;
    for (uint32_t i = 0; i < n && !ctx.rec.full(); ++i)
    {
        const float t = static_cast<float>(i) / n;
        const float a = 2.0f * kPi * t;
        LayerTransform moving;
        moving.rotation = glm::angleAxis(a, glm::vec3(0.0f, 1.0f, 0.0f));
        moving.translation = glm::vec3(offset, 0.1f * ctx.radius * std::sin(a), 0.0f);
        moving.scale = 0.875f + 0.125f * std::sin(a);
        const std::shared_ptr<GaussianCloud> merged =
            mergeLayers(ctx.device, {{ctx.cloud.get(), fixed}, {ctx.cloud.get(), moving}});
        ctx.renderer.setGaussians(inputOf(*merged));
        renderColour(ctx, orbitCamera(ctx.width, ctx.height, centre, frameRadius, 0.3f + 0.5f * t, 0.25f));
    }
    ctx.rec.endSegment();
}

// ---- main -----------------------------------------------------------------

void usage()
{
    std::fprintf(stderr,
                 "usage: sabercat_demo_reel <scene> <outDir> [--width N] [--height N] [--fps N] "
                 "[--max-frames N] [--flip-up]\n");
}

} // namespace

int main(int argc, char ** argv)
{
    if (argc < 3)
    {
        usage();
        return 2;
    }
    const std::string scenePath = argv[1];
    const std::string outDir = argv[2];
    uint32_t width = 1280, height = 720, fps = 30, maxFrames = 0;
    bool flipUp = false;
    for (int i = 3; i < argc; ++i)
    {
        const std::string a = argv[i];
        const bool hasValue = i + 1 < argc;
        if (a == "--width" && hasValue)
        {
            width = static_cast<uint32_t>(std::atoi(argv[++i]));
        }
        else if (a == "--height" && hasValue)
        {
            height = static_cast<uint32_t>(std::atoi(argv[++i]));
        }
        else if (a == "--fps" && hasValue)
        {
            fps = static_cast<uint32_t>(std::atoi(argv[++i]));
        }
        else if (a == "--max-frames" && hasValue)
        {
            maxFrames = static_cast<uint32_t>(std::atoi(argv[++i]));
        }
        else if (a == "--flip-up")
        {
            flipUp = true;
        }
        else
        {
            usage();
            return 2;
        }
    }
    if (width == 0 || height == 0 || fps == 0)
    {
        std::fprintf(stderr, "error: width, height and fps must be positive\n");
        return 2;
    }
    if (!fs::exists(scenePath))
    {
        std::fprintf(stderr, "error: scene not found: %s\n", scenePath.c_str());
        return 2;
    }
    fs::create_directories(outDir);

    try
    {
        Renderer renderer(width, height);
        if (!renderer.initialize({}, true, shaderDir()))
        {
            std::fprintf(stderr, "error: renderer initialize failed\n");
            return 1;
        }
        Device & device = renderer.sabercatDevice();
        const bool rayQuery = renderer.deviceFeatures().rayQuery;
        std::printf("device: ray query %s, backward %s\n", rayQuery ? "yes" : "no",
                    renderer.backwardSupported() ? "yes" : "no");

        const auto loadStart = std::chrono::steady_clock::now();
        std::shared_ptr<GaussianCloud> cloud = sabercat::loadScene(device, scenePath);
        renderer.setGaussians(inputOf(*cloud));
        // glTF (and KHR_gaussian_splatting) is Y-up by specification, so only
        // a Y-down .ply export needs the flip; --flip-up forces it.
        renderer.setFlipUpAxis(flipUp);
        renderer.setLightingEnabled(false);
        const double loadMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - loadStart).count();

        const SceneBounds bounds = renderer.sceneBounds();
        if (!bounds.valid || bounds.focusRadius <= 0.0f)
        {
            std::fprintf(stderr, "error: scene has no bounds\n");
            return 1;
        }
        std::printf("scene: %zu Gaussians, SH degree %u, load %.0f ms, focus centre (%.3f %.3f %.3f) radius %.3f\n",
                    cloud->count(), renderer.sceneSHDegree(), loadMs, bounds.focusCenter.x, bounds.focusCenter.y,
                    bounds.focusCenter.z, bounds.focusRadius);

        Recorder rec(outDir, width, height, maxFrames);
        Context ctx{renderer, rec, device, cloud, width, height, fps, rayQuery, bounds.focusCenter, bounds.focusRadius};

        const auto start = std::chrono::steady_clock::now();
        segmentOrbit(ctx);
        segmentSH(ctx);
        segmentRelight(ctx);
        segmentShadows(ctx);
        segmentBaked(ctx);
        segmentDebug(ctx);
        segmentLayers(ctx);
        const double wallMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

        rec.writeIndex();
        std::printf("done: %u frames at %ux%u, %.1f s wall (%.1f ms per frame)\n", rec.frames(), width, height,
                    wallMs / 1000.0, rec.frames() ? wallMs / rec.frames() : 0.0);
        // Release the cloud before the renderer's device goes away.
        ctx.cloud.reset();
        cloud.reset();
        renderer.shutdown();
        return 0;
    }
    catch (const std::exception & e)
    {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
