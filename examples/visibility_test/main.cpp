// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Baked visibility + teacher samples, verified against CPU
// references on a scene with a known shape -- a ground grid of flat opaque
// Gaussians (z = 0) under a square occluder slab, plus a few rotated
// ellipsoids so the frames are not all axis-aligned. Checks:
//
//  1. bake: every coefficient equals a CPU brute-force bake (same Fibonacci
//     directions, transmittance in double over ALL other Gaussians); an
//     isolated Gaussian bakes c0 = sqrt(4 pi); ground under the slab gets
//     less ambient than far ground; a bake split into many submissions is
//     bit-identical to one submission (the TDR guard changes nothing).
//  2. ShadowMode::BakedSH: each visibility layer equals the color pass
//     splatting the same per-Gaussian values (rendered as colors) --
//     exactly the blend, no new approximation.
//  3. teacher samples: the mix (exact counts, smooth first), every sample's
//     pixel in the CPU's candidate class, stratified and distinct, targets
//     equal to the ray-traced shadow at that pixel BIT FOR BIT and to a CPU
//     recomputation within fp32 tolerance, round-robin lights, positions and
//     normals from the G-buffer, determinism per seed.
//  4. debug views: |error| vs reference (exactly |source - reference|, and 0
//     when the source is the reference), sample locations (exact marks), and
//     ShadowMode::External fed by a second renderer on the same device.
//  5. contracts: the default, 1-direction and capped bakes against the CPU
//     interval; a borrowed scene refuses rayLength 0; teacher options (more
//     samples than candidates, 0 samples, 0 smooth, custom thresholds, emitter
//     grids 2 and 8, a Directional caster, before any lit frame) against the
//     candidate lists and the RayTraced targets; BakedSH for Directional and
//     Rect lights; the BakedSH frame cache; External size checks; and a device
//     without ray query (named errors, PCSS still renders).
//
// Optional argv[1]: a real scene to bake (e.g. assets/bonsai-7k.ply) with the
// default adaptive submissions -- reports the bake time.
//
// Validation messages are logged as "[validation]"; the run must have none.

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "sabercat/borrowed.h"
#include "sabercat/buffer.h"
#include "sabercat/camera.h"
#include "sabercat/device.h"
#include "sabercat/lighting.h"
#include "sabercat/render_config.h"
#include "sabercat/renderer.h"

#include "../common/shader_dir.h"

using namespace sabercat;

namespace
{

constexpr uint32_t kWidth = 96;
constexpr uint32_t kHeight = 96;
constexpr double kPi = 3.14159265358979323846;

int g_failures = 0;

void check(bool ok, const std::string & what)
{
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok)
    {
        ++g_failures;
    }
}

std::string fmt(const char * f, double v)
{
    char buf[128];
    std::snprintf(buf, sizeof(buf), f, v);
    return buf;
}

template <typename F>
bool throwsWith(F && f, const std::string & needle)
{
    try
    {
        f();
    }
    catch (const std::exception & e)
    {
        return std::string(e.what()).find(needle) != std::string::npos;
    }
    return false;
}

// ---- scene ----------------------------------------------------------------

struct Scene
{
    std::vector<glm::vec4> means, scales, rotations, sh, normals, baseColorRoughness;
    std::vector<float> opacities;
    uint32_t count = 0;

    void add(glm::vec3 mean, glm::vec3 scale, glm::quat q, glm::vec3 normal, float opacity)
    {
        means.push_back(glm::vec4(mean, 0.0f));
        scales.push_back(glm::vec4(scale, 0.0f));
        rotations.push_back(glm::vec4(q.x, q.y, q.z, q.w)); // xyzw, w real
        for (int k = 0; k < 16; ++k)
        {
            sh.push_back(k == 0 ? glm::vec4(0.3f, 0.3f, 0.3f, 0.0f) : glm::vec4(0.0f));
        }
        normals.push_back(glm::vec4(normal, 0.0f));
        baseColorRoughness.push_back(glm::vec4(0.8f, 0.8f, 0.8f, 0.9f));
        opacities.push_back(opacity);
        ++count;
    }
};

constexpr float kSlabHeight = 0.4f;

Scene makeScene()
{
    Scene s;
    const glm::quat flat(1.0f, 0.0f, 0.0f, 0.0f); // w, x, y, z: identity, thin axis = z
    for (int iy = -10; iy <= 10; ++iy)
    {
        for (int ix = -10; ix <= 10; ++ix)
        {
            s.add(glm::vec3(0.1f * ix, 0.1f * iy, 0.0f), glm::vec3(0.09f, 0.09f, 0.006f), flat,
                  glm::vec3(0, 0, 1), 0.99f);
        }
    }
    for (int iy = -5; iy <= 5; ++iy)
    {
        for (int ix = -5; ix <= 5; ++ix)
        {
            s.add(glm::vec3(0.06f * ix, 0.06f * iy, kSlabHeight), glm::vec3(0.055f, 0.055f, 0.006f), flat,
                  glm::vec3(0, 0, 1), 0.99f);
        }
    }
    // A few rotated, elongated ellipsoids beside the slab.
    const glm::vec3 spots[4] = {{0.6f, 0.2f, 0.15f}, {-0.55f, -0.4f, 0.2f}, {0.2f, -0.65f, 0.1f},
                                {-0.3f, 0.6f, 0.25f}};
    for (int i = 0; i < 4; ++i)
    {
        const glm::quat q = glm::normalize(glm::angleAxis(0.7f + 0.5f * i, glm::normalize(glm::vec3(1.0f, 0.3f * i, 0.5f))));
        s.add(spots[i], glm::vec3(0.12f, 0.04f, 0.03f), q, q * glm::vec3(0, 0, 1), 0.8f);
    }
    return s;
}

GaussianHostInput input(const Scene & s, const float * visibility = nullptr)
{
    GaussianHostInput in;
    in.count = s.count;
    in.means = &s.means[0].x;
    in.scales = &s.scales[0].x;
    in.rotations = &s.rotations[0].x;
    in.opacities = s.opacities.data();
    in.shCoefficients = &s.sh[0].x;
    in.shDegree = 0;
    in.normals = &s.normals[0].x;
    in.baseColorRoughness = &s.baseColorRoughness[0].x;
    in.visibility = visibility;
    return in;
}

Camera topCamera()
{
    Camera c;
    c.width = kWidth;
    c.height = kHeight;
    c.fx = c.fy = 60.0f;
    c.cx = 0.5f * (kWidth - 1);
    c.cy = 0.5f * (kHeight - 1);
    c.cameraToWorld = glm::mat4(1.0f);
    c.cameraToWorld[3] = glm::vec4(0.0f, 0.0f, 2.4f, 1.0f);
    return c;
}

// ---- CPU ray tracing (double), the same peak-response model ---------------

struct CpuFrame
{
    glm::dvec3 rt[3]; // columns of R
    glm::dvec3 invScale;
    glm::dvec3 mean;
    double opacity = 0.0;
};

bool cpuFrame(const Scene & s, uint32_t i, CpuFrame & f)
{
    const glm::vec4 q = s.rotations[i];
    const glm::dvec4 n = glm::dvec4(q) / std::sqrt(glm::dot(glm::dvec4(q), glm::dvec4(q)));
    const double x = n.x, y = n.y, z = n.z, w = n.w;
    const double r[3][3] = {{1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)},
                            {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)},
                            {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)}};
    for (int k = 0; k < 3; ++k)
    {
        f.rt[k] = glm::dvec3(r[0][k], r[1][k], r[2][k]);
    }
    f.invScale = 1.0 / glm::dvec3(glm::vec3(s.scales[i]));
    f.mean = glm::dvec3(glm::vec3(s.means[i]));
    f.opacity = std::min(static_cast<double>(s.opacities[i]), 1.0);
    return f.opacity >= 1.0 / 255.0;
}

// A Gaussian whose include/skip decision lies within rounding of a threshold
// may legitimately go either way on the GPU, so the reference is an interval:
// [lo, hi] = every borderline Gaussian counted / none counted.
struct CpuTrace
{
    double transmittance = 1.0; // the nominal decisions
    double lo = 1.0;
    double hi = 1.0;
    bool borderline = false;
};

CpuTrace cpuTransmittance(const Scene & s, const std::vector<CpuFrame> & frames, const glm::dvec3 & origin,
                          const glm::dvec3 & dir, double tMin, double tMax, uint32_t skip)
{
    const double minAlpha = 1.0 / 255.0;
    const double maxD2 = static_cast<double>(kRadiusSigmas) * kRadiusSigmas;
    CpuTrace result;
    for (uint32_t i = 0; i < s.count; ++i)
    {
        if (i == skip || frames[i].opacity < minAlpha)
        {
            continue;
        }
        const CpuFrame & f = frames[i];
        const glm::dvec3 v = origin - f.mean;
        const glm::dvec3 o = glm::dvec3(glm::dot(f.rt[0], v), glm::dot(f.rt[1], v), glm::dot(f.rt[2], v)) * f.invScale;
        const glm::dvec3 d = glm::dvec3(glm::dot(f.rt[0], dir), glm::dot(f.rt[1], dir), glm::dot(f.rt[2], dir)) * f.invScale;
        const double dd = glm::dot(d, d);
        const glm::dvec3 c = glm::cross(o, d);
        const double d2 = glm::dot(c, c) / dd;
        const double t = -glm::dot(o, d) / dd;
        const double alpha = std::min(f.opacity * std::exp(-0.5 * d2), static_cast<double>(kMaxAlpha));
        const double tTol = 1e-4 * std::max(1.0, std::fabs(t));
        const bool counted = !(d2 > maxD2 || alpha < minAlpha || t < tMin || t > tMax);
        if (std::fabs(d2 - maxD2) < 1e-3 * maxD2 || std::fabs(alpha - minAlpha) < 2e-3 * minAlpha
            || std::fabs(t - tMin) < tTol || std::fabs(t - tMax) < tTol)
        {
            result.borderline = true;
            result.lo *= 1.0 - alpha;
        }
        else if (counted)
        {
            result.lo *= 1.0 - alpha;
            result.hi *= 1.0 - alpha;
        }
        if (counted)
        {
            result.transmittance *= 1.0 - alpha;
        }
    }
    // The 1e-4 cut-off is monotone, so it maps the interval's ends.
    const auto cut = [](double v) { return v < 1e-4 ? 0.0 : v; };
    if (std::fabs(result.transmittance - 1e-4) < 2e-6)
    {
        result.borderline = true;
    }
    result.transmittance = cut(result.transmittance);
    result.lo = cut(result.lo * (1.0 - 1e-6));
    result.hi = cut(result.hi);
    return result;
}

// bake_visibility.slang's ray start: where direction w leaves Gaussian i's
// own 3-sigma ellipsoid.
double bakeStart(const CpuFrame & f, const glm::dvec3 & w)
{
    const glm::dvec3 d = glm::dvec3(glm::dot(f.rt[0], w), glm::dot(f.rt[1], w), glm::dot(f.rt[2], w)) * f.invScale;
    return static_cast<double>(kRadiusSigmas) / glm::length(d);
}

// bake_visibility.slang's fibonacciDirection, in the same float arithmetic.
glm::vec3 fibonacciDirection(uint32_t j, uint32_t n)
{
    const float goldenAngle = 2.39996322972865332f;
    const float z = 1.0f - (2.0f * static_cast<float>(j) + 1.0f) / static_cast<float>(n);
    const float r = std::sqrt(std::max(0.0f, 1.0f - z * z));
    const float phi = goldenAngle * static_cast<float>(j);
    return glm::vec3(r * std::cos(phi), r * std::sin(phi), z);
}

// (1/pi) * integral V max(0, n.w) dw from the SH (relight_prepare's ambient).
double ambientAt(const float * c, const glm::vec3 & n)
{
    float b[16];
    shBasis16(n, b);
    const double band[4] = {1.0, 2.0 / 3.0, 0.25, 0.0};
    double sum = 0.0;
    for (int k = 0; k < 16; ++k)
    {
        const int l = k == 0 ? 0 : (k < 4 ? 1 : (k < 9 ? 2 : 3));
        sum += band[l] * c[k] * b[k];
    }
    return sum;
}

// sample_select.slang's classify, in the same float arithmetic: 0 none,
// 1 smooth, 2 edge.
uint32_t classify(const Renderer::GBufferData & g, uint32_t x, uint32_t y, const TeacherSettings & t)
{
    const uint32_t p = y * kWidth + x;
    const float coverage = g.metalAmbientCoverage[p].z;
    if (!(coverage >= t.minCoverage))
    {
        return 0;
    }
    const glm::vec4 nd = g.normalDepth[p];
    const float depth = nd.w / coverage;
    const float nLen = glm::length(glm::vec3(nd));
    const glm::vec3 n = nLen > 1e-8f ? glm::vec3(nd) / nLen : glm::vec3(0.0f);
    const int offsets[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    for (const auto & o : offsets)
    {
        const int qx = static_cast<int>(x) + o[0], qy = static_cast<int>(y) + o[1];
        if (qx < 0 || qy < 0 || qx >= static_cast<int>(kWidth) || qy >= static_cast<int>(kHeight))
        {
            continue;
        }
        const uint32_t q = static_cast<uint32_t>(qy) * kWidth + static_cast<uint32_t>(qx);
        const float qCoverage = g.metalAmbientCoverage[q].z;
        if (!(qCoverage >= t.minCoverage))
        {
            return 2;
        }
        const glm::vec4 qnd = g.normalDepth[q];
        if (std::fabs(qnd.w / qCoverage - depth) > t.edgeDepthRatio * depth)
        {
            return 2;
        }
        const float qLen = glm::length(glm::vec3(qnd));
        if (nLen > 1e-8f && qLen > 1e-8f && glm::dot(n, glm::vec3(qnd) / qLen) < t.edgeNormalCos)
        {
            return 2;
        }
    }
    return 1;
}

Light pointAbove(float height)
{
    Light l;
    l.type = LightType::Point;
    l.position = glm::vec3(0.05f, -0.03f, height);
    l.intensity = 4.0f;
    l.castsShadows = 1;
    return l;
}

std::vector<float> layer(const std::vector<float> & all, uint32_t l)
{
    return std::vector<float>(all.begin() + l * kWidth * kHeight, all.begin() + (l + 1) * kWidth * kHeight);
}

void realSceneBake(Renderer & r, const char * path)
{
    r.loadScene(path);
    VisibilityBakeSettings bake; // defaults: 256 directions, adaptive submissions (100 ms target)
    const auto t0 = std::chrono::steady_clock::now();
    r.bakeVisibility(bake);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const uint32_t count = static_cast<uint32_t>(r.gaussianCapacity());
    std::printf("real scene %s: %u Gaussians x %u directions baked in %.2f s (%.1f Mrays/s; the "
                "submission count and final batch size are in the log)\n",
                path, count, bake.directions, seconds, 1e-6 * count * bake.directions / seconds);
    check(seconds > 0.0, "real-scene bake completed");
}

// ---- contract checks: bake settings, teacher options, BakedSH lights, the frame
// cache, External sources, and a device without ray query ----------------------

// The 16 coefficients of every Gaussian in `s`, from `r`'s visibility stream.
std::vector<float> readStream(Renderer & r, const Scene & s)
{
    std::vector<float> stream(s.count * kVisibilityCoefficients);
    std::vector<glm::vec4> m(s.count), sc(s.count), ro(s.count), sh(s.count * 16);
    std::vector<float> op(s.count);
    GaussianHostOutput out;
    out.count = s.count;
    out.means = &m[0].x;
    out.scales = &sc[0].x;
    out.rotations = &ro[0].x;
    out.opacities = op.data();
    out.shCoefficients = &sh[0].x;
    out.visibility = stream.data();
    r.readGaussians(out);
    return stream;
}

struct BakeCheck
{
    double worst = 0.0;     // worst error outside the CPU interval
    double meanWidth = 0.0; // mean width of the interval (borderline rays' reach)
    uint32_t checked = 0;
};

// Every `step`-th Gaussian's coefficients against the CPU brute-force bake
// for `directions` Fibonacci directions and `rayLength`. Tolerance 2e-4, as
// the all-Gaussian check above: per ray |dT| <= ~1e-5, |Y| <= 0.75, and the
// weights 4pi/N sum to 4pi, so the error is bounded independently of N.
BakeCheck cpuBakeCheck(const Scene & s, const std::vector<CpuFrame> & frames, const std::vector<float> & stream,
                       uint32_t directions, float rayLength, uint32_t step)
{
    BakeCheck out;
    double widthSum = 0.0;
    const double weight = 4.0 * kPi / directions;
    for (uint32_t i = 0; i < s.count; i += step)
    {
        double lo[16] = {}, hi[16] = {};
        for (uint32_t j = 0; j < directions; ++j)
        {
            const glm::vec3 w = fibonacciDirection(j, directions);
            const double tMin = std::min(bakeStart(frames[i], glm::dvec3(w)), static_cast<double>(rayLength));
            const CpuTrace t = cpuTransmittance(s, frames, frames[i].mean, glm::dvec3(w), tMin, rayLength, i);
            float b[16];
            shBasis16(w, b);
            for (int k = 0; k < 16; ++k)
            {
                lo[k] += std::min(t.lo * b[k], t.hi * b[k]) * weight;
                hi[k] += std::max(t.lo * b[k], t.hi * b[k]) * weight;
            }
        }
        for (int k = 0; k < 16; ++k)
        {
            const double g = stream[i * 16 + k];
            out.worst = std::max(out.worst, std::max({0.0, g - hi[k], lo[k] - g}));
            widthSum += hi[k] - lo[k];
        }
        ++out.checked;
    }
    out.meanWidth = widthSum / std::max(1.0, 16.0 * out.checked);
    return out;
}

// A lit RayTraced frame of `lights` with `s`'s emitter grid; returns its layers.
std::vector<float> rayTracedFrame(Renderer & r, const std::vector<Light> & lights, ShadowSettings s)
{
    s.mode = ShadowMode::RayTraced;
    r.setShadowSettings(s);
    r.setLights(lights);
    r.forward(RenderMode::Viewer);
    return r.readVisibility();
}

// The candidate lists of a G-buffer under `t`, as classify() sees them.
struct Candidates
{
    std::vector<uint32_t> smooth, edge;
};

Candidates candidatesOf(const Renderer::GBufferData & g, const TeacherSettings & t)
{
    Candidates c;
    for (uint32_t y = 0; y < kHeight; ++y)
    {
        for (uint32_t x = 0; x < kWidth; ++x)
        {
            const uint32_t k = classify(g, x, y, t);
            if (k == 1)
            {
                c.smooth.push_back(y * kWidth + x);
            }
            else if (k == 2)
            {
                c.edge.push_back(y * kWidth + x);
            }
        }
    }
    return c;
}

// One generateTeacherSamples call on the current lit frame, against the
// candidate lists and `rayTraced`, the RayTraced visibility of the same lights
// and emitter grid. Expects min(randomSamples, smooth) smooth samples first,
// then min(edgeSamples, edge) edge ones, each in its class, lights round-robin
// over `casters`, and each target equal bit for bit to the ray-traced layer.
uint32_t checkTeacherCall(Renderer & r, const TeacherSettings & ts, const Renderer::GBufferData & g,
                          const std::vector<float> & rayTraced, const std::vector<uint32_t> & casters,
                          const std::string & tag)
{
    const uint32_t count = r.generateTeacherSamples(ts);
    const std::vector<VisibilityTrainingSample> samples = r.readTeacherSamples();
    const std::vector<uint32_t> pixels = r.readTeacherSamplePixels();
    const Candidates c = candidatesOf(g, ts);
    const std::set<uint32_t> smoothSet(c.smooth.begin(), c.smooth.end());
    const std::set<uint32_t> edgeSet(c.edge.begin(), c.edge.end());
    const uint32_t kS = std::min<uint32_t>(ts.randomSamples, static_cast<uint32_t>(c.smooth.size()));
    const uint32_t kE = std::min<uint32_t>(ts.edgeSamples, static_cast<uint32_t>(c.edge.size()));
    check(count == kS + kE && r.teacherSmoothSampleCount() == kS && samples.size() == count
              && pixels.size() == count,
          tag + "mix " + std::to_string(kS) + " smooth + " + std::to_string(kE) + " edge (requested "
              + std::to_string(ts.randomSamples) + " + " + std::to_string(ts.edgeSamples) + "; "
              + std::to_string(c.smooth.size()) + " / " + std::to_string(c.edge.size()) + " candidates), returned "
              + std::to_string(count));

    const uint32_t n = std::min<uint32_t>(count, static_cast<uint32_t>(std::min(samples.size(), pixels.size())));
    bool classes = true, lights = true, exact = true;
    double meanTarget = 0.0;
    for (uint32_t i = 0; i < n; ++i)
    {
        const bool isEdge = i >= kS;
        const bool edgeBit = (pixels[i] & 0x80000000u) != 0;
        const uint32_t pixel = pixels[i] & 0x7FFFFFFFu;
        classes &= edgeBit == isEdge && (isEdge ? edgeSet : smoothSet).count(pixel) == 1;
        lights &= samples[i].lightIndex == casters[i % casters.size()];
        exact &= samples[i].visibilityTarget == rayTraced[samples[i].lightIndex * kWidth * kHeight + pixel];
        meanTarget += samples[i].visibilityTarget;
    }
    check(classes, tag + "every sample's pixel is in its CPU class, edge flag matching");
    check(lights, tag + "lights round-robin over the shadow casters");
    check(exact, tag + "every target == the RayTraced visibility at its pixel and light, bit for bit"
                     + fmt(" (mean target %.3f)", meanTarget / std::max<uint32_t>(n, 1)));
    return count;
}

// ShadowMode::BakedSH with one light against the color pass splatting each
// Gaussian's V(light) (the method of section 2, for any light type): returns
// the worst difference over covered pixels and their count.
double bakedSplatWorst(Renderer & r, const Scene & s, const std::vector<float> & stream, const Light & light,
                       ShadowSettings settings, uint32_t & covered)
{
    r.setGaussians(input(s, stream.data()));
    r.setLightingEnabled(true);
    settings.mode = ShadowMode::BakedSH;
    r.setShadowSettings(settings);
    r.setLights({light});
    r.forward(RenderMode::Viewer);
    const std::vector<float> baked = layer(r.readVisibility(), 0);

    // visibility_splat.slang: w = -direction (directional) or the way to the
    // light's center; the Hanning window per band; saturate.
    Scene colored = s;
    const float window[4] = {1.0f, 0.853553391f, 0.5f, 0.146446609f};
    for (uint32_t i = 0; i < s.count; ++i)
    {
        glm::vec3 w = light.type == LightType::Directional ? -light.direction
                                                           : light.position - glm::vec3(s.means[i]);
        w /= glm::length(w);
        float b[16];
        shBasis16(w, b);
        float v = 0.0f;
        for (int k = 0; k < 16; ++k)
        {
            const int l = k == 0 ? 0 : (k < 4 ? 1 : (k < 9 ? 2 : 3));
            v += window[l] * stream[i * 16 + k] * b[k];
        }
        v = std::min(std::max(v, 0.0f), 1.0f);
        colored.sh[i * 16] = glm::vec4(glm::vec3((v - 0.5f) / kShC0), 0.0f);
    }
    r.setLightingEnabled(false);
    r.setGaussians(input(colored));
    r.forward(RenderMode::Viewer);
    const std::vector<float> image = r.readImage();
    double worst = 0.0;
    covered = 0;
    for (uint32_t p = 0; p < kWidth * kHeight; ++p)
    {
        const float a = image[p * 4 + 3];
        if (a > 0.05f)
        {
            worst = std::max(worst, static_cast<double>(std::fabs(image[p * 4] / a - baked[p])));
            ++covered;
        }
    }
    r.setLightingEnabled(true);
    return worst;
}

// An application-owned device created WITHOUT ray query, for the renderer to
// borrow (the same instance and device setup as gaussian_raytrace_test).
VKAPI_ATTR VkBool32 VKAPI_CALL plainDebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                  VkDebugUtilsMessageTypeFlagsEXT,
                                                  const VkDebugUtilsMessengerCallbackDataEXT * data, void *)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
    {
        std::printf("[validation] %s\n", data->pMessage);
    }
    return VK_FALSE;
}

class PlainDevice
{
  public:
    PlainDevice()
    {
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "visibility_test (no ray query)";
        appInfo.apiVersion = kRequiredVulkanApiVersion;
        const char * layer = "VK_LAYER_KHRONOS_validation";
        const char * debugExt = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
        VkDebugUtilsMessengerCreateInfoEXT debugInfo{};
        debugInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debugInfo.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debugInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                                | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                                | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debugInfo.pfnUserCallback = plainDebugCallback;
        VkInstanceCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo = &appInfo;
        ici.enabledLayerCount = 1;
        ici.ppEnabledLayerNames = &layer;
        ici.enabledExtensionCount = 1;
        ici.ppEnabledExtensionNames = &debugExt;
        ici.pNext = &debugInfo;
        if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS)
        {
            throw std::runtime_error("PlainDevice: vkCreateInstance");
        }
        auto createMessenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
        createMessenger(instance, &debugInfo, nullptr, &messenger);

        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance, &count, nullptr);
        std::vector<VkPhysicalDevice> gpus(count);
        vkEnumeratePhysicalDevices(instance, &count, gpus.data());
        gpu = gpus.at(0);
        for (VkPhysicalDevice candidate : gpus)
        {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(candidate, &props);
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            {
                gpu = candidate;
                break;
            }
        }
        vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, families.data());
        for (uint32_t i = 0; i < count; ++i)
        {
            if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
            {
                family = i;
                break;
            }
        }

        requirements = std::make_unique<DeviceRequirements>(
            gpu, DeviceRequirements::Options{DeviceRequirements::Need::None, DeviceRequirements::Need::None,
                                             DeviceRequirements::Need::None});
        const float priority = 1.0f;
        VkDeviceQueueCreateInfo qci{};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = family;
        qci.queueCount = 1;
        qci.pQueuePriorities = &priority;
        VkDeviceCreateInfo dci{};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.pNext = requirements->featureChain();
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = static_cast<uint32_t>(requirements->deviceExtensions().size());
        dci.ppEnabledExtensionNames = requirements->deviceExtensions().data();
        if (vkCreateDevice(gpu, &dci, nullptr, &device) != VK_SUCCESS)
        {
            throw std::runtime_error("PlainDevice: vkCreateDevice");
        }
        vkGetDeviceQueue(device, family, 0, &queue);
    }

    ~PlainDevice()
    {
        vkDeviceWaitIdle(device);
        vkDestroyDevice(device, nullptr);
        auto destroyMessenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
        destroyMessenger(instance, messenger, nullptr);
        vkDestroyInstance(instance, nullptr);
    }

    VulkanContext context() const
    {
        VulkanContext c;
        c.instance = instance;
        c.physicalDevice = gpu;
        c.device = device;
        c.computeQueue = queue;
        c.computeFamily = family;
        return c;
    }

    std::unique_ptr<DeviceRequirements> requirements;

  private:
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    VkPhysicalDevice gpu = VK_NULL_HANDLE;
    uint32_t family = UINT32_MAX;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
};

// A device without ray query: bake, teacher samples, RayTraced and the
// error-vs-reference view each throw a named error and the renderer keeps
// working (PCSS renders, BakedSH renders from a stream it was given).
void noRayQueryChecks(const std::vector<float> & stream)
{
    PlainDevice app;
    Renderer no(kWidth, kHeight);
    no.initializeBorrowed(app.context(), app.requirements->enabled(), testShaderDir());
    check(!no.deviceFeatures().rayQuery, "device created without ray query: rayQuery reported unsupported");

    const Scene scene = makeScene();
    no.setGaussians(input(scene));
    no.setCamera(topCamera());
    no.setLightingEnabled(true);
    no.setLights({pointAbove(1.2f)});
    ShadowSettings s;
    s.focusCenter = glm::vec3(0.0f, 0.0f, 0.2f);
    s.focusRadius = 1.6f;
    s.raySamplesPerAxis = 4;
    s.mode = ShadowMode::PCSS;
    no.setShadowSettings(s);
    no.forward(RenderMode::Viewer);
    const std::vector<float> lit = no.readLitImage();
    double coverage = 0.0;
    for (uint32_t p = 0; p < kWidth * kHeight; ++p)
    {
        coverage += lit[p * 4 + 3];
    }
    check(lit.size() == kWidth * kHeight * 4 && coverage > 1000.0,
          fmt("PCSS renders on the device without ray query (coverage %.0f pixels)", coverage));

    check(throwsWith([&] { no.generateTeacherSamples(TeacherSettings{}); }, "ray query"),
          "generateTeacherSamples throws, naming ray query (a lit PCSS frame exists)");
    check(throwsWith([&] { no.bakeVisibility(VisibilityBakeSettings{}); }, "ray query"),
          "bakeVisibility throws, naming ray query");
    ShadowSettings rt = s;
    rt.mode = ShadowMode::RayTraced;
    check(throwsWith(
              [&]
              {
                  no.setShadowSettings(rt);
                  no.forward(RenderMode::Viewer);
              },
              "ray query"),
          "ShadowMode::RayTraced throws, naming ray query");
    ShadowSettings errorView = s;
    errorView.debug = VisibilityDebug::ErrorVsReference;
    check(throwsWith(
              [&]
              {
                  no.setShadowSettings(errorView);
                  no.forward(RenderMode::Viewer);
              },
              "ray query"),
          "VisibilityDebug::ErrorVsReference throws, naming ray query");

    ShadowSettings baked = s;
    baked.mode = ShadowMode::BakedSH;
    no.setGaussians(input(scene)); // reload: the refused bake above allocated a stream
    no.setLightingEnabled(true);
    no.setShadowSettings(baked);
    check(throwsWith([&] { no.forward(RenderMode::Viewer); }, "visibility stream"),
          "BakedSH without a visibility stream throws, naming it (no ray query needed to splat)");
    no.setGaussians(input(scene, stream.data()));
    no.setLightingEnabled(true);
    bool rendered = true;
    try
    {
        no.setShadowSettings(baked);
        no.forward(RenderMode::Viewer);
        no.readLitImage();
    }
    catch (const std::exception &)
    {
        rendered = false;
    }
    check(rendered, "BakedSH with a stream renders without ray query (it traces nothing; not in the spec's list)");

    no.setShadowSettings(s);
    no.forward(RenderMode::Viewer);
    const std::vector<float> again = no.readLitImage();
    double coverageAgain = 0.0;
    for (uint32_t p = 0; p < kWidth * kHeight; ++p)
    {
        coverageAgain += again[p * 4 + 3];
    }
    check(coverageAgain == coverage, "after those errors the renderer still renders PCSS, unchanged");
    no.shutdown();
}

} // namespace

int main(int argc, char ** argv)
{
    std::setvbuf(stdout, nullptr, _IONBF, 0); // keep results visible if a check crashes
    try
    {
        Renderer r(kWidth, kHeight);
        if (!r.initialize({}, true, testShaderDir()))
        {
            std::printf("[FAIL] renderer initialize\n");
            return 1;
        }
        if (!r.deviceFeatures().rayQuery)
        {
            std::printf("[SKIP] this device has no ray query -- the bake and teacher samples trace rays\n");
            r.shutdown();
            return 0;
        }

        Scene scene = makeScene();
        std::vector<CpuFrame> frames(scene.count);
        for (uint32_t i = 0; i < scene.count; ++i)
        {
            cpuFrame(scene, i, frames[i]);
        }
        r.setGaussians(input(scene));
        r.setCamera(topCamera());

        // ---- 1. bake ------------------------------------------------------
        constexpr uint32_t kDirections = 64;
        const float rayLength = 6.0f;
        VisibilityBakeSettings one;
        one.directions = kDirections;
        one.rayLength = rayLength;
        one.maxRaysPerSubmit = 1u << 30;
        one.targetSubmitMilliseconds = 0.0f; // fixed batches: one submission
        r.bakeVisibility(one);
        std::vector<float> single(scene.count * kVisibilityCoefficients);
        std::vector<glm::vec4> m(scene.count), sc(scene.count), ro(scene.count), shOut(scene.count * 16);
        std::vector<float> op(scene.count);
        GaussianHostOutput out;
        out.count = scene.count;
        out.means = &m[0].x;
        out.scales = &sc[0].x;
        out.rotations = &ro[0].x;
        out.opacities = op.data();
        out.shCoefficients = &shOut[0].x;
        out.visibility = single.data();
        r.readGaussians(out);

        VisibilityBakeSettings split = one;
        split.maxRaysPerSubmit = kDirections * 37; // 37 Gaussians per submission
        r.bakeVisibility(split);
        std::vector<float> batched(single.size());
        out.visibility = batched.data();
        r.readGaussians(out);
        check(std::memcmp(single.data(), batched.data(), single.size() * sizeof(float)) == 0,
              "bake split into " + std::to_string((scene.count + 36) / 37)
                  + " submissions == one submission, bit for bit");

        // Adaptive submission sizes (the default: a time target) change only the
        // batching, so they must match a fixed single-submission bake bit for
        // bit. The scene has more Gaussians than the first adaptive batch
        // (64K rays / 256 directions), so the default run really is split.
        const auto bakedStream = [&](const VisibilityBakeSettings & settings)
        {
            r.bakeVisibility(settings);
            std::vector<float> stream(scene.count * kVisibilityCoefficients);
            std::vector<glm::vec4> pos(scene.count), scl(scene.count), rot(scene.count), sh(scene.count * 16);
            std::vector<float> opa(scene.count);
            GaussianHostOutput host;
            host.count = scene.count;
            host.means = &pos[0].x;
            host.scales = &scl[0].x;
            host.rotations = &rot[0].x;
            host.opacities = opa.data();
            host.shCoefficients = &sh[0].x;
            host.visibility = stream.data();
            r.readGaussians(host);
            return stream;
        };
        VisibilityBakeSettings fixedDefault;
        fixedDefault.rayLength = rayLength;
        fixedDefault.maxRaysPerSubmit = 1u << 30;
        fixedDefault.targetSubmitMilliseconds = 0.0f;
        const std::vector<float> reference = bakedStream(fixedDefault);
        VisibilityBakeSettings adaptiveDefault; // defaults except the ray length
        adaptiveDefault.rayLength = rayLength;
        const std::vector<float> adaptive = bakedStream(adaptiveDefault);
        check(adaptive.size() == reference.size()
                  && std::memcmp(adaptive.data(), reference.data(), reference.size() * sizeof(float)) == 0,
              "adaptive bake (default settings) == fixed single-submission bake, bit for bit");
        // The adaptive floor (1M rays) exceeds this small scene, so cap the
        // batch to make the adaptive path split it into many submissions.
        VisibilityBakeSettings tiny = adaptiveDefault;
        tiny.targetSubmitMilliseconds = 0.01f;
        tiny.maxRaysPerSubmit = tiny.directions * 37; // 37 Gaussians per submission
        check(scene.count > 37 * 4, "adaptive bake test scene spans several capped submissions");
        const std::vector<float> tinyStream = bakedStream(tiny);
        check(tinyStream.size() == reference.size()
                  && std::memcmp(tinyStream.data(), reference.data(), reference.size() * sizeof(float)) == 0,
              "adaptive bake with a 0.01 ms target, capped at 37 Gaussians per submission == fixed bake, bit "
              "for bit");

        {
            // Each coefficient must lie in the interval the CPU's borderline
            // decisions allow (exact where no ray is borderline), plus fp32
            // slack: per ray |dT| <= ~1e-5 (products of <= ~10 factors), times
            // |Y| <= 0.75, summed with weight 4pi/N: <= 4pi * 0.75e-5 ~ 1e-4.
            double worst = 0.0, widthSum = 0.0;
            const double weight = 4.0 * kPi / kDirections;
            for (uint32_t i = 0; i < scene.count; ++i)
            {
                double lo[16] = {}, hi[16] = {};
                for (uint32_t j = 0; j < kDirections; ++j)
                {
                    const glm::vec3 w = fibonacciDirection(j, kDirections);
                    const double tMin = std::min(bakeStart(frames[i], glm::dvec3(w)), static_cast<double>(rayLength));
                    const CpuTrace t = cpuTransmittance(scene, frames, frames[i].mean, glm::dvec3(w), tMin,
                                                        rayLength, i);
                    float b[16];
                    shBasis16(w, b);
                    for (int k = 0; k < 16; ++k)
                    {
                        lo[k] += std::min(t.lo * b[k], t.hi * b[k]) * weight;
                        hi[k] += std::max(t.lo * b[k], t.hi * b[k]) * weight;
                    }
                }
                for (int k = 0; k < 16; ++k)
                {
                    const double g = single[i * 16 + k];
                    worst = std::max(worst, std::max({0.0, g - hi[k], lo[k] - g}));
                    widthSum += hi[k] - lo[k];
                }
            }
            // The interval must stay narrow, or the comparison would prove nothing.
            const double meanWidth = widthSum / (16.0 * scene.count);
            check(worst < 2e-4 && meanWidth < 1e-2,
                  "bake == CPU brute-force bake for all " + std::to_string(scene.count) + " Gaussians"
                      + fmt(": worst error outside the borderline interval %.2e (tol 2e-4)", worst)
                      + fmt(", mean interval width %.2e", meanWidth));
        }
        {
            // Ground Gaussians: index (iy+10)*21 + (ix+10).
            const uint32_t under = 10 * 21 + 10;  // (0, 0), under the slab's center
            const uint32_t far = 19 * 21 + 19;    // (0.9, 0.9)
            const double aUnder = ambientAt(&single[under * 16], glm::vec3(0, 0, 1));
            const double aFar = ambientAt(&single[far * 16], glm::vec3(0, 0, 1));
            check(aUnder < aFar - 0.3 && aFar > 0.8,
                  fmt("ambient under the slab %.3f", aUnder) + fmt(" < far ground %.3f", aFar));
        }
        {
            // An isolated Gaussian sees every direction: c0 = (4pi/N) N C0.
            Scene lone;
            lone.add(glm::vec3(0.0f), glm::vec3(0.1f), glm::quat(1, 0, 0, 0), glm::vec3(0, 0, 1), 0.9f);
            lone.add(glm::vec3(50.0f, 0.0f, 0.0f), glm::vec3(0.1f), glm::quat(1, 0, 0, 0), glm::vec3(0, 0, 1), 0.9f);
            r.setGaussians(input(lone));
            VisibilityBakeSettings b = one;
            b.rayLength = 10.0f; // the far one is out of reach
            r.bakeVisibility(b);
            std::vector<float> c(2 * 16);
            std::vector<glm::vec4> m2(2), s2(2), r2(2), sh2(32);
            std::vector<float> o2(2);
            GaussianHostOutput o{2, &m2[0].x, &s2[0].x, &r2[0].x, o2.data(), &sh2[0].x, c.data()};
            r.readGaussians(o);
            const double expected = std::sqrt(4.0 * kPi);
            check(std::fabs(c[0] - expected) < 1e-5,
                  fmt("isolated Gaussian bakes c0 = %.6f", c[0]) + fmt(" (sqrt(4pi) = %.6f)", expected));
            check(std::fabs(ambientAt(c.data(), glm::vec3(0, 0, 1)) - 1.0) < 0.02,
                  "isolated Gaussian: ambient visibility ~1 in any normal direction");
        }

        // ---- 2. ShadowMode::BakedSH == color-splatting the same values ----
        r.setGaussians(input(scene, single.data()));
        r.setLightingEnabled(true);
        ShadowSettings shadows;
        shadows.focusCenter = glm::vec3(0.0f, 0.0f, 0.2f);
        shadows.focusRadius = 1.6f;
        shadows.raySamplesPerAxis = 4;
        const Light point = pointAbove(1.2f);
        std::vector<float> bakedLayer;
        {
            shadows.mode = ShadowMode::BakedSH;
            r.setShadowSettings(shadows);
            r.setLights({point});
            r.forward(RenderMode::Viewer);
            const std::vector<float> baked = layer(r.readVisibility(), 0);

            // Per-Gaussian V toward the light, as colors: rgb = C0*dc + 0.5 = V.
            Scene colored = scene;
            for (uint32_t i = 0; i < scene.count; ++i)
            {
                glm::vec3 w = point.position - glm::vec3(scene.means[i]);
                w /= glm::length(w);
                float b[16];
                shBasis16(w, b);
                // visibility_splat.slang's Hanning window per band.
                const float window[4] = {1.0f, 0.853553391f, 0.5f, 0.146446609f};
                float v = 0.0f;
                for (int k = 0; k < 16; ++k)
                {
                    const int l = k == 0 ? 0 : (k < 4 ? 1 : (k < 9 ? 2 : 3));
                    v += window[l] * single[i * 16 + k] * b[k];
                }
                v = std::min(std::max(v, 0.0f), 1.0f);
                colored.sh[i * 16] = glm::vec4(glm::vec3((v - 0.5f) / kShC0), 0.0f);
            }
            r.setLightingEnabled(false);
            r.setGaussians(input(colored));
            r.forward(RenderMode::Viewer);
            const std::vector<float> image = r.readImage();
            double worst = 0.0;
            uint32_t pixels = 0;
            for (uint32_t p = 0; p < kWidth * kHeight; ++p)
            {
                const float a = image[p * 4 + 3];
                if (a > 0.05f)
                {
                    worst = std::max(worst, static_cast<double>(std::fabs(image[p * 4] / a - baked[p])));
                    ++pixels;
                }
            }
            check(pixels > 2000 && worst < 2e-5,
                  "BakedSH layer == color pass splatting each Gaussian's V(light) over "
                      + std::to_string(pixels) + fmt(" pixels, worst %.2e (tol 2e-5)", worst));
            bakedLayer = baked;
            r.setGaussians(input(scene, single.data()));
            r.setLightingEnabled(true);
        }

        // ---- 3. teacher samples --------------------------------------------
        Light sun;
        sun.type = LightType::Directional;
        sun.direction = glm::normalize(glm::vec3(0.3f, 0.2f, -1.0f));
        sun.castsShadows = 0; // skipped by the round-robin
        Light rect;
        rect.type = LightType::Rect;
        rect.position = glm::vec3(-0.2f, 0.1f, 1.0f);
        rect.halfWidth = rect.halfHeight = 0.2f;
        rect.castsShadows = 1;
        const std::vector<Light> lights = {point, sun, rect};
        r.setLights(lights);
        shadows.mode = ShadowMode::RayTraced;
        r.setShadowSettings(shadows);
        r.forward(RenderMode::Viewer);
        const std::vector<float> rtVisibility = r.readVisibility();
        const Renderer::GBufferData g = r.readGBuffer();
        {
            // Grade BakedSH against the ray-traced reference (same point
            // light, layer 0). Reported, not required to beat "no shadows":
            // a degree-3 SH smooths a nearby occluder's large solid angle
            // into the light's direction (ground beside the slab reads ~0.8
            // toward a light it sees clearly), so the bake serves ambient and
            // environment light best; sharp shadows need a higher degree.
            // The |error| debug view exists to show exactly this. What must
            // hold: in-shadow pixels read darker than lit ones.
            double errBaked = 0.0, errNone = 0.0, litSum = 0.0, shadowSum = 0.0;
            uint32_t n = 0, litN = 0, shadowN = 0;
            for (uint32_t p = 0; p < kWidth * kHeight; ++p)
            {
                if (g.metalAmbientCoverage[p].z > 0.5f)
                {
                    errBaked += std::fabs(bakedLayer[p] - rtVisibility[p]);
                    errNone += std::fabs(1.0f - rtVisibility[p]);
                    ++n;
                    if (rtVisibility[p] > 0.95f)
                    {
                        litSum += bakedLayer[p];
                        ++litN;
                    }
                    else if (rtVisibility[p] < 0.05f)
                    {
                        shadowSum += bakedLayer[p];
                        ++shadowN;
                    }
                }
            }
            const double lit = litSum / std::max(litN, 1u), shadow = shadowSum / std::max(shadowN, 1u);
            std::printf("  BakedSH vs ray-traced reference: mean |error| %.3f (no shadows: %.3f)\n",
                        errBaked / std::max(n, 1u), errNone / std::max(n, 1u));
            check(shadowN > 20 && litN > 1000 && shadow < lit - 0.05,
                  fmt("BakedSH: ray-traced umbra pixels read darker (mean %.3f", shadow)
                      + fmt(") than lit pixels (%.3f)", lit));
        }

        TeacherSettings ts;
        ts.randomSamples = 600;
        ts.edgeSamples = 200;
        ts.emitterSamplesPerAxis = shadows.raySamplesPerAxis;
        ts.seed = 7;
        const uint32_t count = r.generateTeacherSamples(ts);
        const std::vector<VisibilityTrainingSample> samples = r.readTeacherSamples();
        const std::vector<uint32_t> pixels = r.readTeacherSamplePixels();

        std::vector<uint32_t> smoothList, edgeList;
        for (uint32_t y = 0; y < kHeight; ++y)
        {
            for (uint32_t x = 0; x < kWidth; ++x)
            {
                const uint32_t c = classify(g, x, y, ts);
                (c == 1 ? smoothList : edgeList).push_back(y * kWidth + x);
                if (c == 0)
                {
                    edgeList.pop_back();
                }
            }
        }
        const uint32_t kS = std::min<uint32_t>(ts.randomSamples, static_cast<uint32_t>(smoothList.size()));
        const uint32_t kE = std::min<uint32_t>(ts.edgeSamples, static_cast<uint32_t>(edgeList.size()));
        check(count == kS + kE && r.teacherSmoothSampleCount() == kS && samples.size() == count,
              "sample mix: " + std::to_string(r.teacherSmoothSampleCount()) + " smooth + "
                  + std::to_string(count - r.teacherSmoothSampleCount()) + " edge of "
                  + std::to_string(smoothList.size()) + " / " + std::to_string(edgeList.size())
                  + " candidates (expected " + std::to_string(kS) + " + " + std::to_string(kE) + ")");
        check(kS == ts.randomSamples && kE == ts.edgeSamples,
              "fixture has enough candidates for the exact 75/25 mix");
        {
            bool classesOk = true, stratified = true, increasing = true;
            for (uint32_t i = 0; i < count; ++i)
            {
                const bool edge = (pixels[i] & 0x80000000u) != 0;
                const uint32_t pixel = pixels[i] & 0x7FFFFFFFu;
                const bool isEdgeSample = i >= kS;
                const std::vector<uint32_t> & list = isEdgeSample ? edgeList : smoothList;
                const uint32_t k = isEdgeSample ? i - kS : i;
                const uint32_t K = isEdgeSample ? kE : kS;
                classesOk &= edge == isEdgeSample;
                const auto it = std::find(list.begin(), list.end(), pixel);
                classesOk &= it != list.end();
                if (it != list.end())
                {
                    const double ratio = static_cast<double>(list.size()) / K;
                    const uint32_t idx = static_cast<uint32_t>(it - list.begin());
                    stratified &= idx >= static_cast<uint32_t>(std::floor(k * ratio)) - (k > 0 ? 1u : 0u)
                                  && idx <= static_cast<uint32_t>(std::floor((k + 1) * ratio));
                }
                if (k > 0)
                {
                    increasing &= pixel > (pixels[i - 1] & 0x7FFFFFFFu);
                }
            }
            check(classesOk, "every sample's pixel is in its CPU class (smooth / edge)");
            check(stratified && increasing, "samples are stratified over each list: one per stratum, distinct, in order");
        }
        {
            bool exact = true, lightsOk = true, unit = true;
            double worstPos = 0.0, worstCpu = 0.0;
            uint32_t cpuCompared = 0;
            const glm::mat4 camToWorld = topCamera().cameraToWorld;
            for (uint32_t i = 0; i < count; ++i)
            {
                const VisibilityTrainingSample & s = samples[i];
                const uint32_t pixel = pixels[i] & 0x7FFFFFFFu;
                lightsOk &= s.lightIndex == (i % 2 == 0 ? 0u : 2u);
                exact &= s.visibilityTarget == rtVisibility[s.lightIndex * kWidth * kHeight + pixel];
                const float nl = glm::length(s.normal);
                unit &= std::fabs(nl - 1.0f) < 1e-5f || nl == 0.0f;

                const glm::vec4 nd = g.normalDepth[pixel];
                const float cov = g.metalAmbientCoverage[pixel].z;
                const float depth = nd.w / cov;
                const Camera cam = topCamera();
                const glm::vec3 local(((pixel % kWidth) - cam.cx) * depth / cam.fx,
                                      (cam.cy - (pixel / kWidth)) * depth / cam.fy, -depth);
                const glm::vec3 P = glm::vec3(camToWorld * glm::vec4(local, 1.0f));
                worstPos = std::max(worstPos, static_cast<double>(glm::length(P - s.position)));

                if (i < 300)
                {
                    // CPU recomputation: same emitter grid, double transmittance.
                    const Light & L = lights[s.lightIndex];
                    const glm::dvec3 origin = glm::dvec3(s.position)
                                              + glm::dvec3(s.normal) * static_cast<double>(shadows.normalOffset * shadows.focusRadius);
                    const uint32_t n = ts.emitterSamplesPerAxis;
                    double sumLo = 0.0, sumHi = 0.0;
                    for (uint32_t sy = 0; sy < n; ++sy)
                    {
                        for (uint32_t sx = 0; sx < n; ++sx)
                        {
                            const double u = (sx + 0.5) / n * 2.0 - 1.0, v = (sy + 0.5) / n * 2.0 - 1.0;
                            glm::dvec3 S;
                            if (L.type == LightType::Point)
                            {
                                const glm::dvec3 toC = glm::normalize(glm::dvec3(L.position) - origin);
                                const glm::dvec3 helper = std::fabs(toC.y) < 0.99 ? glm::dvec3(0, 1, 0) : glm::dvec3(1, 0, 0);
                                const glm::dvec3 t = glm::normalize(glm::cross(helper, toC));
                                const glm::dvec3 b = glm::cross(toC, t);
                                S = glm::dvec3(L.position) + (u * t + v * b) * (shadows.pointLightRadius * shadows.focusRadius * 0.8862);
                            }
                            else
                            {
                                S = glm::dvec3(L.position) + u * glm::dvec3(glm::normalize(L.axisX) * L.halfWidth)
                                    + v * glm::dvec3(glm::normalize(L.axisY) * L.halfHeight);
                            }
                            const glm::dvec3 toS = S - origin;
                            const double dist = glm::length(toS);
                            const CpuTrace t = cpuTransmittance(scene, frames, origin, toS / dist, 0.0, dist, UINT32_MAX);
                            sumLo += t.lo;
                            sumHi += t.hi;
                        }
                    }
                    const double lo = sumLo / (n * n), hi = sumHi / (n * n);
                    worstCpu = std::max({worstCpu, lo - s.visibilityTarget, s.visibilityTarget - hi});
                    ++cpuCompared;
                }
            }
            check(exact, "every visibilityTarget == the RayTraced shadow at its pixel and light, bit for bit");
            check(cpuCompared > 200 && worstCpu < 1e-4,
                  "visibilityTarget == CPU recomputation for " + std::to_string(cpuCompared)
                      + fmt(" samples, worst distance outside the borderline interval %.2e (tol 1e-4)", worstCpu));
            check(lightsOk, "lights assigned round-robin over the shadow-casting ones (0, 2, 0, 2, ...)");
            check(worstPos < 2e-5 && unit,
                  fmt("positions from the G-buffer (worst %.2e), normals unit", worstPos));
            double meanTarget = 0.0;
            for (const auto & s : samples)
            {
                meanTarget += s.visibilityTarget;
            }
            meanTarget /= std::max<size_t>(samples.size(), 1);
            std::printf("  mean visibilityTarget %.3f\n", meanTarget);
        }
        {
            const BufferView view = r.teacherSampleView();
            check(view.valid() && view.stride == 32 && view.size == count * 32ull,
                  "teacherSampleView: count x 32 B, borrowable");
            r.generateTeacherSamples(ts);
            const std::vector<uint32_t> again = r.readTeacherSamplePixels();
            const std::vector<VisibilityTrainingSample> againSamples = r.readTeacherSamples();
            check(again == pixels
                      && std::memcmp(againSamples.data(), samples.data(), samples.size() * sizeof(samples[0])) == 0,
                  "same seed: identical samples");
            TeacherSettings other = ts;
            other.seed = 8;
            r.generateTeacherSamples(other);
            const std::vector<uint32_t> reseeded = r.readTeacherSamplePixels();
            uint32_t differ = 0;
            for (size_t i = 0; i < std::min(reseeded.size(), pixels.size()); ++i)
            {
                differ += reseeded[i] != pixels[i] ? 1 : 0;
            }
            check(differ > count / 2, "another seed: " + std::to_string(differ) + " of " + std::to_string(count)
                                          + " samples move");
            // Supervision is always the exact model, whatever the shadows use.
            ShadowSettings approx = shadows;
            approx.rayMinAlpha = 0.1f;
            approx.rayMinTransmittance = 0.05f;
            approx.rayDownsample = 2;
            r.setShadowSettings(approx);
            r.forward(RenderMode::Viewer);
            r.generateTeacherSamples(ts);
            const std::vector<VisibilityTrainingSample> underApprox = r.readTeacherSamples();
            check(underApprox.size() == samples.size()
                      && std::memcmp(underApprox.data(), samples.data(), samples.size() * sizeof(samples[0])) == 0,
                  "teacher samples stay exact (bit for bit) while shadows use approximations");
            r.setShadowSettings(shadows);
            r.forward(RenderMode::Viewer);
            r.generateTeacherSamples(ts); // back to seed 7 for the location view
        }
        {
            Light dark = point;
            dark.castsShadows = 0;
            r.setLights({dark});
            r.forward(RenderMode::Viewer);
            check(throwsWith([&] { r.generateTeacherSamples(ts); }, "castsShadows"),
                  "no shadow-casting light: generateTeacherSamples refuses, naming castsShadows");
            r.setLights(lights);
        }

        // ---- 4. debug views ------------------------------------------------
        {
            ShadowSettings d = shadows;
            d.mode = ShadowMode::RayTraced;
            d.debug = VisibilityDebug::ErrorVsReference;
            d.referenceSamplesPerAxis = d.raySamplesPerAxis;
            r.setShadowSettings(d);
            r.forward(RenderMode::Viewer);
            const std::vector<float> e = r.readVisibility();
            check(std::all_of(e.begin(), e.end(), [](float v) { return v == 0.0f; }),
                  "|error| of the reference against itself is exactly 0");

            ShadowSettings pcss = shadows;
            pcss.mode = ShadowMode::PCSS;
            r.setShadowSettings(pcss);
            r.forward(RenderMode::Viewer);
            const std::vector<float> vP = r.readVisibility();
            ShadowSettings ref = shadows;
            ref.mode = ShadowMode::RayTraced;
            ref.raySamplesPerAxis = 8;
            r.setShadowSettings(ref);
            r.forward(RenderMode::Viewer);
            const std::vector<float> vR = r.readVisibility();
            pcss.debug = VisibilityDebug::ErrorVsReference;
            pcss.referenceSamplesPerAxis = 8;
            r.setShadowSettings(pcss);
            r.forward(RenderMode::Viewer);
            const std::vector<float> vE = r.readVisibility();
            bool exact = vE.size() == vP.size();
            double mean = 0.0;
            for (size_t i = 0; exact && i < vE.size(); ++i)
            {
                exact &= vE[i] == std::fabs(vP[i] - vR[i]);
                mean += vE[i];
            }
            check(exact, "PCSS |error| view == |PCSS - RayTraced(8x8)| exactly"
                             + fmt(", mean over layers %.4f", mean / std::max<size_t>(vE.size(), 1)));
        }
        {
            ShadowSettings d = shadows;
            d.debug = VisibilityDebug::SampleLocations;
            r.setShadowSettings(d);
            r.forward(RenderMode::Viewer);
            const std::vector<float> marks = r.readVisibility();
            std::vector<float> expected(marks.size(), 0.0f);
            for (uint32_t i = 0; i < count; ++i)
            {
                expected[samples[i].lightIndex * kWidth * kHeight + (pixels[i] & 0x7FFFFFFFu)] =
                    (pixels[i] & 0x80000000u) ? 1.0f : 0.5f;
            }
            check(marks == expected, "sample-locations view: 0.5 smooth / 1 edge at each sample, in its light's layer");
        }
        {
            // External: a second renderer on the SAME device renders the
            // ray-traced visibility; this one copies it in.
            Renderer other(kWidth, kHeight);
            other.initializeBorrowed(r.vulkanContext(), r.enabledDeviceFeatures(), testShaderDir());
            other.setGaussians(input(scene));
            other.setCamera(topCamera());
            other.setLightingEnabled(true);
            other.setLights(lights);
            ShadowSettings rt = shadows;
            rt.mode = ShadowMode::RayTraced;
            other.setShadowSettings(rt);
            other.forward(RenderMode::Viewer);
            const std::vector<float> source = other.readVisibility();

            r.setShadowSettings(rt);
            r.forward(RenderMode::Viewer);
            const std::vector<float> ownLit = r.readLitImage();
            ShadowSettings ext = shadows;
            ext.mode = ShadowMode::External;
            r.setShadowSettings(ext);
            check(throwsWith([&] { r.forward(RenderMode::Viewer); }, "setExternalVisibility"),
                  "External without a source refuses, naming setExternalVisibility");
            r.setExternalVisibility(other.visibilityView());
            r.forward(RenderMode::Viewer);
            check(r.readVisibility() == source, "External: visibility == the other renderer's buffer, exactly");
            check(r.readLitImage() == ownLit, "External: lit image == this renderer's own RayTraced lit image");
            r.setExternalVisibility(BufferView{});
            other.shutdown();
        }

        // ---- 5. Bake settings, teacher options, BakedSH lights, cache, External. ----
        const std::vector<Light> lights3 = {point, sun, rect};
        {
            // Defaults (256 directions, rayLength 0 = 4 x the scene radius, adaptive
            // submissions), then 1 direction -- both against the CPU interval.
            r.setGaussians(input(scene));
            check(r.sceneBounds().valid, "an owned scene has bounds (rayLength 0 is 4 x their radius)");
            const float radius = r.sceneBounds().radius();
            r.bakeVisibility(VisibilityBakeSettings{});
            const BakeCheck d = cpuBakeCheck(scene, frames, readStream(r, scene), 256, 4.0f * radius, 7);
            check(d.checked > 0 && d.worst < 2e-4 && d.meanWidth < 1e-2,
                  "default bake (256 directions, rayLength 0) == CPU for " + std::to_string(d.checked)
                      + " sampled Gaussians" + fmt(": worst outside the interval %.2e (tol 2e-4)", d.worst)
                      + fmt(", mean width %.2e", d.meanWidth));

            VisibilityBakeSettings oneDirection;
            oneDirection.directions = 1;
            oneDirection.rayLength = 6.0f;
            r.bakeVisibility(oneDirection);
            const BakeCheck o = cpuBakeCheck(scene, frames, readStream(r, scene), 1, 6.0f, 1);
            check(o.checked == scene.count && o.worst < 2e-4 && o.meanWidth < 1e-2,
                  "bake with 1 direction == CPU for all " + std::to_string(o.checked) + " Gaussians"
                      + fmt(": worst outside the interval %.2e (tol 2e-4)", o.worst));

            // A Gaussian's directions are one submission, so a cap below them is
            // refused; a cap equal to them gives the one-submission result.
            VisibilityBakeSettings capped;
            capped.directions = 64;
            capped.rayLength = 6.0f;
            capped.maxRaysPerSubmit = 63;
            capped.targetSubmitMilliseconds = 0.0f;
            check(throwsWith([&] { r.bakeVisibility(capped); }, "maxRaysPerSubmit"),
                  "maxRaysPerSubmit below directions is rejected, naming maxRaysPerSubmit");
            capped.maxRaysPerSubmit = 64;
            r.setGaussians(input(scene));
            r.bakeVisibility(capped);
            check(readStream(r, scene) == single,
                  "maxRaysPerSubmit == directions: one Gaussian per submission, bit for bit with the one-submission bake");

            // A borrowed scene has no bounds: rayLength 0 is refused.
            Device & dev = r.sabercatDevice();
            const uint32_t n = scene.count;
            // The SH stream holds 16 coefficients per Gaussian whatever shDegree says.
            Buffer means(dev, n * 16ull), scales(dev, n * 16ull), rotations(dev, n * 16ull), opacity(dev, n * 4ull),
                coefficients(dev, n * 256ull);
            means.upload(scene.means.data(), n * 16ull);
            scales.upload(scene.scales.data(), n * 16ull);
            rotations.upload(scene.rotations.data(), n * 16ull);
            opacity.upload(scene.opacities.data(), n * 4ull);
            coefficients.upload(scene.sh.data(), n * 256ull);
            BorrowedGaussians borrowedScene;
            CanonicalGaussianBufferViews & v = borrowedScene.parameters;
            v.means = means.view(kMeansStride);
            v.means.size = n * 16ull;
            v.scales = scales.view(kScalesStride);
            v.scales.size = n * 16ull;
            v.rotations = rotations.view(kRotationsStride);
            v.rotations.size = n * 16ull;
            v.opacity = opacity.view(kOpacityStride);
            v.opacity.size = n * 4ull;
            v.shCoefficients = coefficients.view(kShStride);
            v.shCoefficients.size = n * 256ull;
            v.gaussianCount = n;
            v.shDegree = 0;
            Renderer borrowed(kWidth, kHeight);
            borrowed.initializeBorrowed(r.vulkanContext(), r.enabledDeviceFeatures(), testShaderDir());
            borrowed.borrowGaussians(borrowedScene);
            check(throwsWith([&] { borrowed.bakeVisibility(VisibilityBakeSettings{}); }, "rayLength"),
                  "borrowed scene with rayLength 0 throws, naming VisibilityBakeSettings::rayLength");
            borrowed.shutdown();
        }
        {
            // Teacher options on the default lit frame (point, sun, rect) and on others.
            r.setGaussians(input(scene));
            r.setLightingEnabled(true);
            const std::vector<float> rt3 = rayTracedFrame(r, lights3, shadows);
            const Renderer::GBufferData g3 = r.readGBuffer();
            const std::vector<uint32_t> casters = {0, 2};

            TeacherSettings more;
            more.randomSamples = 20000; // more than the smooth candidates
            more.edgeSamples = 3000;    // more than the edge candidates
            more.emitterSamplesPerAxis = shadows.raySamplesPerAxis;
            more.seed = 11;
            checkTeacherCall(r, more, g3, rt3, casters, "more samples than candidates: ");

            TeacherSettings none;
            none.randomSamples = 0;
            none.edgeSamples = 0;
            check(throwsWith([&] { r.generateTeacherSamples(none); }, "sample counts"),
                  "0 samples: generateTeacherSamples throws (sample counts must be 1..2^24)");

            TeacherSettings edgeOnly;
            edgeOnly.randomSamples = 0;
            edgeOnly.edgeSamples = 50;
            edgeOnly.emitterSamplesPerAxis = shadows.raySamplesPerAxis;
            edgeOnly.seed = 5;
            checkTeacherCall(r, edgeOnly, g3, rt3, casters, "0 smooth samples: ");

            TeacherSettings custom;
            custom.randomSamples = 600;
            custom.edgeSamples = 200;
            custom.minCoverage = 0.9f;
            custom.edgeDepthRatio = 0.2f;
            custom.edgeNormalCos = 0.99f;
            custom.emitterSamplesPerAxis = shadows.raySamplesPerAxis;
            custom.seed = 3;
            checkTeacherCall(r, custom, g3, rt3, casters, "custom minCoverage 0.9 / depth 20% / normal 8 deg: ");

            for (uint32_t grid : {2u, 8u})
            {
                ShadowSettings gridShadows = shadows;
                gridShadows.raySamplesPerAxis = grid;
                const std::vector<float> rtGrid = rayTracedFrame(r, lights3, gridShadows);
                TeacherSettings t;
                t.randomSamples = 600;
                t.edgeSamples = 200;
                t.emitterSamplesPerAxis = grid;
                t.seed = 20 + grid;
                checkTeacherCall(r, t, g3, rtGrid, casters,
                                 "emitter grid " + std::to_string(grid) + "x" + std::to_string(grid) + ": ");
            }

            Light sunCaster = sun;
            sunCaster.castsShadows = 1;
            const std::vector<float> rtSun = rayTracedFrame(r, {sunCaster}, shadows);
            TeacherSettings sunSamples;
            sunSamples.randomSamples = 600;
            sunSamples.edgeSamples = 200;
            sunSamples.emitterSamplesPerAxis = shadows.raySamplesPerAxis;
            sunSamples.seed = 9;
            checkTeacherCall(r, sunSamples, g3, rtSun, {0}, "Directional shadow caster as the only light: ");

            Renderer fresh(kWidth, kHeight);
            fresh.initializeBorrowed(r.vulkanContext(), r.enabledDeviceFeatures(), testShaderDir());
            fresh.setGaussians(input(scene));
            fresh.setLightingEnabled(true);
            fresh.setLights(lights3);
            fresh.setShadowSettings(shadows);
            check(throwsWith([&] { fresh.generateTeacherSamples(); }, "lit forward"),
                  "generateTeacherSamples before any lit frame throws, naming the lit forward");
            fresh.shutdown();
            r.setLights(lights3);
        }
        {
            // BakedSH with the directional and rect lights, against the color pass.
            Light sunShadow = sun;
            sunShadow.castsShadows = 1;
            uint32_t sunPixels = 0, rectPixels = 0;
            const double sunWorst = bakedSplatWorst(r, scene, single, sunShadow, shadows, sunPixels);
            check(sunPixels > 2000 && sunWorst < 2e-5,
                  "BakedSH Directional == color pass splatting V(sun) over " + std::to_string(sunPixels)
                      + fmt(" pixels, worst %.2e (tol 2e-5)", sunWorst));
            const double rectWorst = bakedSplatWorst(r, scene, single, rect, shadows, rectPixels);
            check(rectPixels > 2000 && rectWorst < 2e-5,
                  "BakedSH Rect == color pass splatting V(rect) over " + std::to_string(rectPixels)
                      + fmt(" pixels, worst %.2e (tol 2e-5)", rectWorst));
        }
        {
            // A bake or a setGaussians between two BakedSH frames invalidates the frame cache.
            r.setGaussians(input(scene, single.data()));
            r.setLightingEnabled(true);
            r.setLights({point});
            ShadowSettings bakedShadows = shadows;
            bakedShadows.mode = ShadowMode::BakedSH;
            r.setShadowSettings(bakedShadows);
            r.forward(RenderMode::Viewer);
            const uint32_t firstReuse = r.shadowFrameStats().reusedFrame;
            const std::vector<float> first = layer(r.readVisibility(), 0);
            r.forward(RenderMode::Viewer);
            const uint32_t secondReuse = r.shadowFrameStats().reusedFrame;
            const std::vector<float> second = layer(r.readVisibility(), 0);
            check(firstReuse == 0 && secondReuse == 1 && second == first,
                  "BakedSH: an unchanged frame is reused from the cache, identical");

            VisibilityBakeSettings coarse;
            coarse.directions = 32;
            coarse.rayLength = 3.0f;
            r.bakeVisibility(coarse);
            r.forward(RenderMode::Viewer);
            const std::vector<float> afterBake = layer(r.readVisibility(), 0);
            double change = 0.0;
            for (size_t i = 0; i < first.size(); ++i)
            {
                change = std::max(change, static_cast<double>(std::fabs(afterBake[i] - first[i])));
            }
            check(r.shadowFrameStats().reusedFrame == 0 && change > 1e-3,
                  fmt("a bake between BakedSH frames: recomputed, not reused (max change %.3f)", change));

            r.setGaussians(input(scene, single.data()));
            r.forward(RenderMode::Viewer);
            check(r.shadowFrameStats().reusedFrame == 0 && layer(r.readVisibility(), 0) == first,
                  "setGaussians between BakedSH frames: recomputed, not reused (back to the same values)");
        }
        {
            // ShadowMode::External: a source too small, and one from a half-size image.
            r.setGaussians(input(scene));
            r.setLightingEnabled(true);
            r.setLights(lights3);
            rayTracedFrame(r, lights3, shadows);
            const BufferView full = r.visibilityView();
            BufferView undersized = full;
            undersized.size -= 4;
            ShadowSettings external = shadows;
            external.mode = ShadowMode::External;
            r.setShadowSettings(external);
            r.setExternalVisibility(undersized);
            check(throwsWith([&] { r.forward(RenderMode::Viewer); }, "must hold"),
                  "External: a view 4 bytes short of the layers is rejected");

            Renderer half(kWidth / 2, kHeight / 2);
            half.initializeBorrowed(r.vulkanContext(), r.enabledDeviceFeatures(), testShaderDir());
            Camera halfCamera = topCamera();
            halfCamera.width = kWidth / 2;
            halfCamera.height = kHeight / 2;
            halfCamera.cx = 0.5f * (halfCamera.width - 1);
            halfCamera.cy = 0.5f * (halfCamera.height - 1);
            half.setGaussians(input(scene));
            half.setCamera(halfCamera);
            half.setLightingEnabled(true);
            half.setLights(lights3);
            ShadowSettings halfRt = shadows;
            halfRt.mode = ShadowMode::RayTraced;
            half.setShadowSettings(halfRt);
            half.forward(RenderMode::Viewer);
            r.setExternalVisibility(half.visibilityView());
            check(throwsWith([&] { r.forward(RenderMode::Viewer); }, "must hold"),
                  "External: a half-size image's visibility view is rejected (size mismatch)");
            half.shutdown();
            r.setExternalVisibility(BufferView{});
        }
        noRayQueryChecks(single);

        if (argc > 1)
        {
            r.setLightingEnabled(false);
            realSceneBake(r, argv[1]);
        }
        r.shutdown();
    }
    catch (const std::exception & e)
    {
        std::printf("[FAIL] exception: %s\n", e.what());
        return 1;
    }
    std::printf(g_failures == 0 ? "visibility_test: all checks passed\n" : "visibility_test: %d check(s) FAILED\n",
                g_failures);
    return g_failures == 0 ? 0 : 1;
}
