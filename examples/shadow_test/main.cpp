// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// PCSS shadows over splats, verified numerically on a scene whose
// answer is known: a ground plane of flat opaque Gaussians (z = 0, facing +z)
// with a square occluder slab of opaque Gaussians above it, lit from straight
// above. Checks:
//
//  1. pixels whose ground point lies under the occluder read visibility ~0,
//     pixels far outside it read ~1;
//  2. the shadow lands where geometry says (the umbra's centre of mass sits
//     under the occluder's centre, within a tolerance of the sampling grid);
//  3. penumbra width grows with light size (point-light radius sweep) and
//     with occluder-receiver distance;
//  4. lights with castsShadows = 0 stay fully visible, and ShadowMode::None
//     leaves every visibility layer at 1;
//  5. the lit image darkens exactly where visibility is low.
//
// Validation messages are logged as "[validation]"; the run must have none.

#include <glm/glm.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "sabercat/camera.h"
#include "sabercat/device.h"
#include "sabercat/lighting.h"
#include "sabercat/render_config.h"
#include "sabercat/renderer.h"

using namespace sabercat;

namespace
{

constexpr uint32_t kWidth = 96;
constexpr uint32_t kHeight = 96;

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
    char buf[96];
    std::snprintf(buf, sizeof(buf), f, v);
    return buf;
}

struct Scene
{
    std::vector<glm::vec4> means, scales, rotations, sh, normals, baseColorRoughness;
    std::vector<float> opacities;
    uint32_t count = 0;

    void add(glm::vec3 mean, glm::vec3 scale, glm::vec3 normal, float opacity)
    {
        means.push_back(glm::vec4(mean, 0.0f));
        scales.push_back(glm::vec4(scale, 0.0f));
        rotations.push_back(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)); // thin axis = z
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

// Ground: a dense grid of flat disks at z = 0 covering [-2,2]^2. Occluder: a
// dense slab of flat disks at z = height covering [-0.5,0.5]^2.
Scene makeScene(float occluderHeight)
{
    Scene s;
    const float step = 0.08f;
    for (float y = -2.0f; y <= 2.001f; y += step)
    {
        for (float x = -2.0f; x <= 2.001f; x += step)
        {
            s.add(glm::vec3(x, y, 0.0f), glm::vec3(0.07f, 0.07f, 0.005f), glm::vec3(0, 0, 1), 0.99f);
        }
    }
    for (float y = -0.5f; y <= 0.501f; y += 0.05f)
    {
        for (float x = -0.5f; x <= 0.501f; x += 0.05f)
        {
            s.add(glm::vec3(x, y, occluderHeight), glm::vec3(0.045f, 0.045f, 0.005f),
                  glm::vec3(0, 0, 1), 0.99f);
        }
    }
    return s;
}

GaussianHostInput input(const Scene & s)
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
    return in;
}

// Looks straight down at the ground from above (camera on +z, ground below).
Camera topCamera()
{
    Camera c;
    c.width = kWidth;
    c.height = kHeight;
    c.fx = c.fy = 60.0f;
    c.cx = 0.5f * (kWidth - 1);
    c.cy = 0.5f * (kHeight - 1);
    c.cameraToWorld = glm::mat4(1.0f);
    c.cameraToWorld[3] = glm::vec4(0.0f, 0.0f, 3.4f, 1.0f);
    return c;
}

// Ground point (x, y, 0) seen by the camera -> pixel.
glm::vec2 pixelOfGround(const glm::vec2 & xy)
{
    const Camera c = topCamera();
    const float depth = 3.4f;
    return glm::vec2(c.fx * xy.x / depth + c.cx, c.cy - c.fy * xy.y / depth);
}

struct Result
{
    std::vector<float> visibility; // layer 0
    std::vector<float> lit;
    std::vector<float> coverage;
};

Result render(Renderer & r, const std::vector<Light> & lights, ShadowMode mode, const ShadowSettings & base)
{
    ShadowSettings st = base;
    st.mode = mode;
    r.setShadowSettings(st);
    r.setLights(lights);
    r.forward(RenderMode::Viewer);
    Result out;
    const std::vector<float> all = r.readVisibility();
    out.visibility.assign(all.begin(), all.begin() + kWidth * kHeight);
    out.lit = r.readLitImage();
    for (uint32_t p = 0; p < kWidth * kHeight; ++p)
    {
        out.coverage.push_back(out.lit[p * 4 + 3]);
    }
    return out;
}

float visibilityAt(const Result & r, const glm::vec2 & groundXY)
{
    const glm::vec2 px = pixelOfGround(groundXY);
    return r.visibility[static_cast<uint32_t>(std::lround(px.y)) * kWidth +
                        static_cast<uint32_t>(std::lround(px.x))];
}

// Width, in ground units, of the transition band along the +x axis through
// the occluder: x positions where visibility is strictly between 0.1 and 0.9.
double penumbraWidth(const Result & r)
{
    double lo = 1e9, hi = -1e9;
    for (float x = 0.0f; x <= 1.9f; x += 0.01f)
    {
        const float v = visibilityAt(r, glm::vec2(x, 0.0f));
        if (v > 0.1f && v < 0.9f)
        {
            lo = std::min<double>(lo, x);
            hi = std::max<double>(hi, x);
        }
    }
    return hi >= lo ? hi - lo : 0.0;
}

Light pointAbove(float height, float intensity)
{
    Light l;
    l.type = LightType::Point;
    l.position = glm::vec3(0.0f, 0.0f, height);
    l.intensity = intensity;
    l.castsShadows = 1;
    return l;
}

// ---- CPU port of the ray-traced shadow kernel (double precision) ----
//
// The same model as shadow_rt.slang and light_visibility.slangh: a ray's
// transmittance is the product over Gaussians whose peak (on the ray, within
// [tMin, tMax]) lies inside the 3-sigma ellipsoid, each with alpha =
// min(opacity * exp(-d^2 / 2), max alpha), and the emitter grid mean. Near a
// threshold the GPU and CPU may legitimately disagree, so the reference is an
// interval: every borderline Gaussian counted / none counted (the method of
// visibility_test).

struct CpuGaussian
{
    glm::dvec3 rt[3]; // columns of R
    glm::dvec3 invScale;
    glm::dvec3 mean;
    double opacity = 0.0;
};

CpuGaussian cpuGaussian(const Scene & s, uint32_t i)
{
    CpuGaussian f;
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
    return f;
}

struct CpuTrace
{
    double transmittance = 1.0; // nominal decisions, cut below 1e-4 (the kernel's ray cut-off)
    double lo = 1.0;            // every borderline Gaussian counted
    double hi = 1.0;            // none counted
};

CpuTrace cpuTrace(const std::vector<CpuGaussian> & gs, const glm::dvec3 & origin, const glm::dvec3 & dir,
                  double tMin, double tMax)
{
    const double minAlpha = 1.0 / 255.0;
    const double maxD2 = static_cast<double>(kRadiusSigmas) * kRadiusSigmas;
    const double maxAlpha = static_cast<double>(kMaxAlpha);
    CpuTrace result;
    for (const CpuGaussian & f : gs)
    {
        const glm::dvec3 v = origin - f.mean;
        const glm::dvec3 o = glm::dvec3(glm::dot(f.rt[0], v), glm::dot(f.rt[1], v), glm::dot(f.rt[2], v)) * f.invScale;
        const glm::dvec3 d = glm::dvec3(glm::dot(f.rt[0], dir), glm::dot(f.rt[1], dir), glm::dot(f.rt[2], dir)) * f.invScale;
        const double dd = glm::dot(d, d);
        const glm::dvec3 c = glm::cross(o, d);
        const double d2 = glm::dot(c, c) / dd;
        const double t = -glm::dot(o, d) / dd;
        const double alpha = std::min(f.opacity * std::exp(-0.5 * d2), maxAlpha);
        const double tTol = 1e-4 * std::max(1.0, std::fabs(t));
        const bool counted = !(d2 > maxD2 || alpha < minAlpha || t < tMin || t > tMax);
        if (std::fabs(d2 - maxD2) < 1e-3 * maxD2 || std::fabs(alpha - minAlpha) < 2e-3 * minAlpha
            || std::fabs(t - tMin) < tTol || std::fabs(t - tMax) < tTol)
        {
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
    const auto cut = [](double v) { return v < 1e-4 ? 0.0 : v; };
    result.transmittance = cut(result.transmittance);
    result.lo = cut(result.lo * (1.0 - 1e-6));
    result.hi = cut(result.hi);
    return result;
}

// shadow_pass's TraceLight, in double.
struct CpuLight
{
    LightType type = LightType::Point;
    glm::dvec3 position{0.0}, direction{0.0}, axisX{0.0}, axisY{0.0};
    double lightSize = 0.0; // Point: disk radius; Directional: tan(angular radius)
    double rayLength = 0.0;
};

CpuLight cpuLightOf(const Light & l, const ShadowSettings & st)
{
    CpuLight c;
    c.type = l.type;
    c.position = glm::dvec3(l.position);
    c.direction = glm::dvec3(l.direction);
    c.axisX = glm::dvec3(glm::normalize(l.axisX)) * static_cast<double>(l.halfWidth);
    c.axisY = glm::dvec3(glm::normalize(l.axisY)) * static_cast<double>(l.halfHeight);
    c.lightSize = l.type == LightType::Directional ? std::tan(static_cast<double>(st.sunAngularRadius))
                                                   : static_cast<double>(st.pointLightRadius * st.focusRadius);
    c.rayLength = 8.0 * static_cast<double>(st.focusRadius);
    return c;
}

// tangentFrame of light_visibility.slangh.
void cpuTangentFrame(const glm::dvec3 & n, glm::dvec3 & t, glm::dvec3 & b)
{
    const glm::dvec3 helper = std::fabs(n.y) < 0.99 ? glm::dvec3(0.0, 1.0, 0.0) : glm::dvec3(1.0, 0.0, 0.0);
    t = glm::normalize(glm::cross(helper, n));
    b = glm::cross(n, t);
}

// Emitter sample s (row-major n x n grid) of the light from origin P.
CpuTrace cpuEmitterSample(const std::vector<CpuGaussian> & gs, const CpuLight & L, const glm::dvec3 & P, uint32_t s,
                          uint32_t n)
{
    const double u = (static_cast<double>(s % n) + 0.5) / n * 2.0 - 1.0;
    const double v = (static_cast<double>(s / n) + 0.5) / n * 2.0 - 1.0;
    glm::dvec3 dir;
    double tMax = 0.0;
    if (L.type == LightType::Directional)
    {
        const glm::dvec3 toLight = -glm::normalize(L.direction);
        glm::dvec3 t, b;
        cpuTangentFrame(toLight, t, b);
        dir = glm::normalize(toLight + (u * t + v * b) * L.lightSize * 0.8862);
        tMax = L.rayLength;
    }
    else
    {
        glm::dvec3 S;
        if (L.type == LightType::Point)
        {
            glm::dvec3 t, b;
            cpuTangentFrame(glm::normalize(L.position - P), t, b);
            S = L.position + (u * t + v * b) * L.lightSize * 0.8862;
        }
        else
        {
            S = L.position + u * L.axisX + v * L.axisY;
        }
        const glm::dvec3 toS = S - P;
        const double dist = glm::length(toS);
        dir = toS / dist;
        tMax = dist;
    }
    return cpuTrace(gs, P, dir, 0.0, tMax);
}

// The receiver of G-buffer pixel p in asset space, offset along its normal as
// the kernel does (ShadowSettings::normalOffset * focusRadius).
glm::dvec3 cpuReceiver(const Renderer::GBufferData & g, uint32_t p, const Camera & cam, const ShadowSettings & st,
                       glm::dvec3 & normal)
{
    const glm::vec4 nd = g.normalDepth[p];
    const float cov = g.metalAmbientCoverage[p].z;
    const double depth = static_cast<double>(nd.w / cov);
    const double len = glm::length(glm::dvec3(nd));
    normal = len > 1e-8 ? glm::dvec3(nd) / len : glm::dvec3(0.0);
    const double u = static_cast<double>(p % cam.width), v = static_cast<double>(p / cam.width);
    const glm::dvec3 local((u - cam.cx) * depth / cam.fx, (cam.cy - v) * depth / cam.fy, -depth);
    const glm::dvec3 position = glm::dvec3(glm::dmat4(cam.cameraToWorld) * glm::dvec4(local, 1.0));
    return position + normal * static_cast<double>(st.normalOffset * st.focusRadius);
}

} // namespace

int main(int argc, char ** argv)
{
    try
    {
        Renderer r(kWidth, kHeight);
        if (!r.initialize({}, true, SHADER_DIR))
        {
            std::printf("[FAIL] renderer initialize\n");
            return 1;
        }
        r.setCamera(topCamera());
        r.setLightingEnabled(true);
        LightingSettings ls;
        ls.exposure = 1.0f;
        r.setLightingSettings(ls);

        ShadowSettings shadows;
        shadows.focusCenter = glm::vec3(0.0f, 0.0f, 0.3f);
        shadows.focusRadius = 2.6f;
        shadows.mapResolution = 1024;
        shadows.pointLightRadius = 0.02f;

        Scene scene = makeScene(0.6f);
        r.setGaussians(input(scene));

        // 1 + 2: a light straight above the occluder.
        // A low-ish light so the shadow (+-0.83 for the 0.6-high slab) extends
        // past the slab's footprint as the top-down camera sees it (+-0.61):
        // the camera sees the slab's lit top over the ground directly beneath
        // it, so the umbra is checked in the ring between the two.
        const std::vector<Light> overhead = {pointAbove(1.5f, 6.0f)};
        const Result shadowed = render(r, overhead, ShadowMode::PCSS, shadows);
        const float under = visibilityAt(shadowed, glm::vec2(0.72f, 0.0f));
        const float outside = visibilityAt(shadowed, glm::vec2(1.6f, 1.2f));
        check(under < 0.1f, "ground in the occluder's shadow ring is dark " + fmt("(visibility %.3f)", under));
        check(outside > 0.9f, "ground far outside the occluder is lit " + fmt("(visibility %.3f)", outside));
        {
            // Centre of mass of the umbra (visibility < 0.5) among covered pixels.
            double sx = 0.0, sy = 0.0, n = 0.0;
            for (uint32_t y = 0; y < kHeight; ++y)
            {
                for (uint32_t x = 0; x < kWidth; ++x)
                {
                    const uint32_t p = y * kWidth + x;
                    // Only the ground: skip pixels the occluder itself covers.
                    const float gx = (x - 0.5f * (kWidth - 1)) * 3.4f / 60.0f;
                    const float gy = -(y - 0.5f * (kHeight - 1)) * 3.4f / 60.0f;
                    if (shadowed.coverage[p] > 0.5f && shadowed.visibility[p] < 0.5f &&
                        std::fabs(gx) < 1.5f && std::fabs(gy) < 1.5f)
                    {
                        sx += gx;
                        sy += gy;
                        n += 1.0;
                    }
                }
            }
            const double cx = n > 0 ? sx / n : 99.0, cy = n > 0 ? sy / n : 99.0;
            check(n > 20 && std::fabs(cx) < 0.12 && std::fabs(cy) < 0.12,
                  "umbra is centred under the occluder " + fmt("(centre %.3f", cx) + fmt(", %.3f)", cy));
        }

        // 3a: penumbra grows with the light's size.
        double previous = -1.0;
        bool monotonic = true;
        std::string widths;
        for (float radius : {0.01f, 0.06f, 0.15f})
        {
            ShadowSettings s2 = shadows;
            s2.pointLightRadius = radius;
            const double w = penumbraWidth(render(r, overhead, ShadowMode::PCSS, s2));
            widths += fmt(" %.3f", w);
            monotonic &= w >= previous - 0.005;
            previous = w;
        }
        check(monotonic && previous > 0.05, "penumbra width grows with light size (widths:" + widths + ")");

        // 3c: a directional light's penumbra grows with its angular radius.
        {
            // Tilted, so the shadow (offset 0.48 in +x) lands on the visible ground
            // beside the slab rather than under it.
            std::vector<Light> sunDown(1);
            sunDown[0].type = LightType::Directional;
            sunDown[0].direction = glm::normalize(glm::vec3(0.8f, 0.0f, -1.0f));
            sunDown[0].intensity = 6.0f;
            double prevWidth = -1.0;
            bool growing = true;
            std::string widths;
            for (float angle : {0.005f, 0.03f, 0.1f})
            {
                ShadowSettings s2 = shadows;
                s2.sunAngularRadius = angle;
                s2.depthBias = 0.02f; // the oblique sun sees the slab's disks at varying depth
                const double w = penumbraWidth(render(r, sunDown, ShadowMode::PCSS, s2));
                widths += fmt(" %.3f", w);
                growing &= w >= prevWidth - 0.005;
                prevWidth = w;
            }
            check(growing && prevWidth > 0.05,
                  "directional penumbra width grows with sunAngularRadius (widths:" + widths + ")");
        }

        // 3b: penumbra grows with occluder-receiver distance.
        double near = 0.0, far = 0.0;
        {
            ShadowSettings s2 = shadows;
            s2.pointLightRadius = 0.08f;
            Scene lowScene = makeScene(0.25f);
            r.setGaussians(input(lowScene));
            near = penumbraWidth(render(r, overhead, ShadowMode::PCSS, s2));
            Scene highScene = makeScene(1.0f);
            r.setGaussians(input(highScene));
            far = penumbraWidth(render(r, overhead, ShadowMode::PCSS, s2));
            check(far > near + 0.02, "penumbra widens with occluder height " + fmt("(0.25 -> %.3f", near) +
                                         fmt(", 1.0 -> %.3f)", far));
            r.setGaussians(input(scene));
        }

        // 4: opt-outs.
        {
            std::vector<Light> noCast = overhead;
            noCast[0].castsShadows = 0;
            const Result a = render(r, noCast, ShadowMode::PCSS, shadows);
            const Result b = render(r, overhead, ShadowMode::None, shadows);
            const bool allOne = std::all_of(a.visibility.begin(), a.visibility.end(), [](float v) { return v == 1.0f; });
            const bool noneOne = std::all_of(b.visibility.begin(), b.visibility.end(), [](float v) { return v == 1.0f; });
            check(allOne, "a light with castsShadows = 0 keeps visibility 1 under PCSS");
            check(noneOne, "ShadowMode::None leaves visibility at 1");
        }

        // 5: the lit image darkens where visibility is low.
        {
            const Result unshadowed = render(r, overhead, ShadowMode::None, shadows);
            double darkSum = 0.0, litSum = 0.0, darkN = 0.0, litN = 0.0;
            for (uint32_t p = 0; p < kWidth * kHeight; ++p)
            {
                if (shadowed.coverage[p] < 0.5f)
                {
                    continue;
                }
                const float ratio = shadowed.lit[p * 4] / std::max(unshadowed.lit[p * 4], 1e-6f);
                if (shadowed.visibility[p] < 0.05f)
                {
                    darkSum += ratio;
                    darkN += 1.0;
                }
                else if (shadowed.visibility[p] > 0.95f)
                {
                    litSum += ratio;
                    litN += 1.0;
                }
            }
            const double darkMean = darkN > 0 ? darkSum / darkN : 9.0;
            const double litMean = litN > 0 ? litSum / litN : 0.0;
            check(darkN > 20 && darkMean < 0.05,
                  "fully shadowed pixels lose their direct light " + fmt("(lit/unshadowed %.4f)", darkMean));
            check(litN > 100 && std::fabs(litMean - 1.0) < 1e-3,
                  "fully visible pixels are unchanged " + fmt("(ratio %.4f)", litMean));
        }

        // 6: the ray-traced reference (needs ray query).
        if (!r.deviceFeatures().rayQuery)
        {
            std::printf("[SKIP] ray query not enabled on this device: no ray-traced checks\n");
        }
        else
        {
            ShadowSettings rt = shadows;
            rt.pointLightRadius = 0.08f;
            rt.raySamplesPerAxis = 16;
            const Result traced = render(r, overhead, ShadowMode::RayTraced, rt);
            check(visibilityAt(traced, glm::vec2(0.72f, 0.0f)) < 0.05f &&
                      visibilityAt(traced, glm::vec2(1.6f, 1.2f)) > 0.99f,
                  "RT: shadow ring dark, far ground fully lit " +
                      fmt("(%.3f", visibilityAt(traced, glm::vec2(0.72f, 0.0f))) +
                      fmt(", %.3f)", visibilityAt(traced, glm::vec2(1.6f, 1.2f))));

            // Deterministic: the sample grid is fixed. (Reuse off, so both
            // frames really trace.)
            ShadowSettings fresh = rt;
            fresh.cacheAcrossFrames = 0;
            const Result again = render(r, overhead, ShadowMode::RayTraced, fresh);
            check(again.visibility == traced.visibility && r.shadowFrameStats().reusedFrame == 0,
                  "RT is deterministic frame to frame (bit-identical, both traced)");

            // Penumbra grows with light size.
            double previousWidth = -1.0;
            bool monotone = true;
            std::string widths;
            for (float radius : {0.01f, 0.06f, 0.15f})
            {
                ShadowSettings s2 = rt;
                s2.pointLightRadius = radius;
                const double w = penumbraWidth(render(r, overhead, ShadowMode::RayTraced, s2));
                widths += fmt(" %.3f", w);
                monotone &= w >= previousWidth - 0.005;
                previousWidth = w;
            }
            check(monotone && previousWidth > 0.05, "RT penumbra width grows with light size (widths:" + widths + ")");

            // Convergence: error against a 32x32 reference falls as samples rise.
            ShadowSettings ref = rt;
            ref.raySamplesPerAxis = 32;
            const Result reference = render(r, overhead, ShadowMode::RayTraced, ref);
            double previousError = 1e9;
            bool falling = true;
            std::string errors;
            for (uint32_t n : {2u, 4u, 8u, 16u})
            {
                ShadowSettings s2 = rt;
                s2.raySamplesPerAxis = n;
                const Result t = render(r, overhead, ShadowMode::RayTraced, s2);
                double err = 0.0, count = 0.0;
                for (uint32_t p = 0; p < kWidth * kHeight; ++p)
                {
                    if (reference.coverage[p] > 0.5f)
                    {
                        err += std::fabs(t.visibility[p] - reference.visibility[p]);
                        count += 1.0;
                    }
                }
                err /= std::max(count, 1.0);
                errors += fmt(" %.4f", err);
                falling &= err <= previousError + 1e-4;
                previousError = err;
            }
            check(falling && previousError < 0.01, "RT converges as samples rise (mean |dV| vs 32x32:" + errors + ")");

            // PCSS is a baseline for this reference: report and bound the gap.
            ShadowSettings pc = rt;
            pc.mapResolution = 2048;
            const Result pcss = render(r, overhead, ShadowMode::PCSS, pc);
            double gap = 0.0, count = 0.0;
            for (uint32_t p = 0; p < kWidth * kHeight; ++p)
            {
                if (reference.coverage[p] > 0.5f)
                {
                    gap += std::fabs(pcss.visibility[p] - reference.visibility[p]);
                    count += 1.0;
                }
            }
            gap /= std::max(count, 1.0);
            check(gap < 0.08, "PCSS agrees with the ray-traced reference " + fmt("(mean |dV| %.4f, bound 0.08)", gap));

            // Directional light: PCSS against the ray-traced reference, same
            // angular radius (the sun's disk, traced with the same grid).
            {
                std::vector<Light> sunDown(1);
                sunDown[0].type = LightType::Directional;
                sunDown[0].direction = glm::normalize(glm::vec3(0.8f, 0.0f, -1.0f));
                sunDown[0].intensity = 6.0f;
                ShadowSettings sun = rt;
                sun.sunAngularRadius = 0.05f;
                sun.depthBias = 0.02f;
                sun.raySamplesPerAxis = 16;
                sun.mapResolution = 2048;
                const Result sunTraced = render(r, sunDown, ShadowMode::RayTraced, sun);
                const Result sunPcss = render(r, sunDown, ShadowMode::PCSS, sun);
                double sunGap = 0.0, sunCount = 0.0, sunDark = 0.0;
                for (uint32_t p = 0; p < kWidth * kHeight; ++p)
                {
                    if (sunTraced.coverage[p] > 0.5f)
                    {
                        sunGap += std::fabs(sunPcss.visibility[p] - sunTraced.visibility[p]);
                        sunDark += sunTraced.visibility[p] < 0.5f ? 1.0 : 0.0;
                        sunCount += 1.0;
                    }
                }
                sunGap /= std::max(sunCount, 1.0);
                // The same bound as the point light: the receiver-plane term keeps the
                // oblique sun's penumbra on the reference's scale.
                check(sunDark > 20 && sunGap < 0.08,
                      "directional PCSS vs the ray-traced reference " +
                          fmt("(mean |dV| %.4f, bound 0.08, as the point light", sunGap) + fmt(", %.0f shadowed px)", sunDark));
            }

            // Opt-out and the lit image.
            std::vector<Light> noCast = overhead;
            noCast[0].castsShadows = 0;
            const Result a = render(r, noCast, ShadowMode::RayTraced, rt);
            check(std::all_of(a.visibility.begin(), a.visibility.end(), [](float v) { return v == 1.0f; }),
                  "RT: a light with castsShadows = 0 keeps visibility 1");

            // Reuse across frames: shared acceleration structure, frame cache,
            // progressive accumulation (all only while the scene is tracked).
            {
                Light second = overhead[0];
                second.position = glm::vec3(0.5f, -0.3f, 1.8f);
                const std::vector<Light> two = {overhead[0], second};
                ShadowSettings full = rt;
                full.raySamplesPerAxis = 4;
                full.mode = ShadowMode::RayTraced;
                r.setLights(two);
                const auto frame = [&](const ShadowSettings & s)
                {
                    r.setShadowSettings(s);
                    r.forward(RenderMode::Viewer);
                    return r.readVisibility();
                };
                full.cacheAcrossFrames = 0;
                const std::vector<float> reference = frame(full); // two lights, full grid, no reuse
                const ShadowFrameStats s0 = r.shadowFrameStats();
                full.cacheAcrossFrames = 1;
                const std::vector<float> first = frame(full);
                const ShadowFrameStats s1 = r.shadowFrameStats();
                const std::vector<float> second2 = frame(full);
                const ShadowFrameStats s2 = r.shadowFrameStats();
                check(s0.accelerationBuilds == 1 && s0.emitterSamplesDispatched == 2ull * kWidth * kHeight * 16,
                      "one acceleration-structure build per frame for two lights, full grid each");
                check(first == reference && s1.accelerationBuilds == 0,
                      "tracked scene: the structure is reused, result unchanged");
                check(second2 == reference && s2.reusedFrame == 1 && s2.emitterSamplesDispatched == 0,
                      "unchanged frame: visibility restored from the frame cache, no rays");

                // Progressive: one emitter sample per frame converges to the
                // full grid bit for bit, then traces nothing.
                ShadowSettings prog = full;
                prog.progressiveSamplesPerFrame = 1;
                std::vector<float> v;
                bool pendingUntilLast = true;
                for (int f = 0; f < 16; ++f)
                {
                    v = frame(prog);
                    const ShadowFrameStats s = r.shadowFrameStats();
                    pendingUntilLast &= f < 15 ? (s.pendingLights == 2 && s.convergedLights == 0)
                                               : (s.convergedLights == 2 && s.pendingLights == 0);
                    pendingUntilLast &= s.emitterSamplesDispatched == 2ull * kWidth * kHeight;
                }
                check(v == reference && pendingUntilLast,
                      "progressive 1 sample/frame: after 16 frames == the full 4x4 grid, bit for bit");

                const std::vector<float> settled = frame(prog);
                check(settled == reference && r.shadowFrameStats().emitterSamplesDispatched == 0
                          && r.shadowFrameStats().convergedLights == 2,
                      "converged: later frames trace no rays and keep the result");

                // Moving one light restarts only that light.
                std::vector<Light> moved = two;
                moved[1].position.x += 0.2f;
                r.setLights(moved);
                frame(prog);
                const ShadowFrameStats sm = r.shadowFrameStats();
                check(sm.convergedLights == 1 && sm.pendingLights == 1
                          && sm.emitterSamplesDispatched == 1ull * kWidth * kHeight,
                      "moving one light restarts only that light's accumulation");
                r.setLights(two);

                // Other step sizes on the same 16-sample grid: k = 3, 7, 32
                // converge in ceil(16 / k) frames to the same bits. markSceneChanged()
                // restarts the accumulation each time. (After the checks above,
                // which depend on the k = 1 state.)
                for (uint32_t k : {3u, 7u, 32u})
                {
                    r.markSceneChanged();
                    ShadowSettings pk = full;
                    pk.progressiveSamplesPerFrame = k;
                    const uint32_t frames = (16 + k - 1) / k;
                    std::vector<float> vk;
                    bool stepsOk = true;
                    for (uint32_t f = 0; f < frames; ++f)
                    {
                        vk = frame(pk);
                        const ShadowFrameStats s = r.shadowFrameStats();
                        const bool last = f + 1 == frames;
                        stepsOk &= last ? (s.convergedLights == 2 && s.pendingLights == 0)
                                        : (s.pendingLights == 2 && s.convergedLights == 0);
                    }
                    check(vk == reference && stepsOk,
                          fmt("progressive %.0f samples/frame: the full 4x4 grid, bit for bit", k) +
                              fmt(" (converged in %.0f frames)", frames));
                }

                // In-place edits: markSceneChanged() and handed-out views.
                frame(full);
                r.markSceneChanged();
                frame(full);
                const ShadowFrameStats sc = r.shadowFrameStats();
                check(sc.accelerationBuilds == 1 && sc.reusedFrame == 0,
                      "markSceneChanged(): rebuild and retrace");
                (void)r.gaussianViews();
                frame(full);
                frame(full);
                const ShadowFrameStats sv = r.shadowFrameStats();
                check(sv.accelerationBuilds == 1 && sv.reusedFrame == 0
                          && sv.emitterSamplesDispatched == 2ull * kWidth * kHeight * 16,
                      "after gaussianViews() is handed out: every frame rebuilds and retraces");
                frame(prog);
                check(r.shadowFrameStats().pendingLights == 0
                          && r.shadowFrameStats().emitterSamplesDispatched == 2ull * kWidth * kHeight * 16,
                      "untracked scene: progressive falls back to the full grid every frame");

                // A fresh upload is tracked again; PCSS frames are cached too.
                r.setGaussians(input(scene));
                ShadowSettings pcssReuse = full;
                pcssReuse.mode = ShadowMode::PCSS;
                const std::vector<float> p1 = frame(pcssReuse);
                const std::vector<float> p2 = frame(pcssReuse);
                check(p1 == p2 && r.shadowFrameStats().reusedFrame == 1, "PCSS: an unchanged frame is reused");
                Camera nudged = topCamera();
                nudged.cameraToWorld[3].x += 0.01f;
                r.setCamera(nudged);
                frame(pcssReuse);
                check(r.shadowFrameStats().reusedFrame == 0, "a camera change invalidates the frame cache");
                r.setCamera(topCamera());

                // Approximations, each against the exact trace of the same frame.
                full.cacheAcrossFrames = 0;
                const std::vector<float> exact = frame(full);
                double worst = 0.0;
                bool changed = false;
                ShadowSettings cut = full;
                cut.rayMinTransmittance = 0.05f;
                const std::vector<float> vCut = frame(cut);
                for (size_t k = 0; k < exact.size(); ++k)
                {
                    worst = std::max(worst, static_cast<double>(std::fabs(vCut[k] - exact[k])));
                    changed |= vCut[k] != exact[k];
                }
                check(worst <= 0.05 + 1e-6 && changed,
                      fmt("rayMinTransmittance 0.05: per-pixel error <= 0.05 (worst %.4f)", worst));

                double previous = -1.0;
                bool lighterOnly = true, growing = true;
                std::string errors;
                for (float minAlpha : {0.02f, 0.1f, 0.3f})
                {
                    ShadowSettings faint = full;
                    faint.rayMinAlpha = minAlpha;
                    const std::vector<float> v = frame(faint);
                    const bool rebuilt = r.shadowFrameStats().accelerationBuilds == 1;
                    double mean = 0.0;
                    for (size_t k = 0; k < exact.size(); ++k)
                    {
                        lighterOnly &= v[k] >= exact[k] - 1e-6f; // skipped occluders can only add light
                        mean += std::fabs(v[k] - exact[k]);
                    }
                    mean /= exact.size();
                    growing &= rebuilt && mean >= previous;
                    previous = mean;
                    errors += fmt(" %.4f", mean);
                }
                check(lighterOnly && growing,
                      "rayMinAlpha: never darker than exact, error grows with the cutoff, structure rebuilt"
                      " (mean |dV|:" + errors + ")");

                ShadowSettings half = full;
                half.rayDownsample = 2;
                const std::vector<float> vHalf = frame(half);
                bool tracedExact = true;
                double halfMean = 0.0;
                for (uint32_t l = 0; l < 2; ++l)
                {
                    for (uint32_t y = 0; y < kHeight; ++y)
                    {
                        for (uint32_t x = 0; x < kWidth; ++x)
                        {
                            const size_t k = static_cast<size_t>(l) * kWidth * kHeight + y * kWidth + x;
                            if (x % 2 == 0 && y % 2 == 0)
                            {
                                tracedExact &= vHalf[k] == exact[k];
                            }
                            halfMean += std::fabs(vHalf[k] - exact[k]);
                        }
                    }
                }
                halfMean /= 2.0 * kWidth * kHeight;
                check(tracedExact && halfMean < 0.03
                          && r.shadowFrameStats().emitterSamplesDispatched == 2ull * 48 * 48 * 16,
                      fmt("rayDownsample 2: traced pixels exact, a quarter of the rays, mean |dV| %.4f", halfMean));
                r.setLights(overhead);
            }
        }

        // Per-pixel ray-traced shadows against the CPU port, one light of each
        // type; Rect PCSS against the reference; the penumbra's scaling with the
        // sun's angle; and the PCSS filter / blocker sample sweeps.
        if (r.deviceFeatures().rayQuery)
        {
            ShadowSettings rtp = shadows;
            rtp.mode = ShadowMode::RayTraced;
            rtp.pointLightRadius = 0.08f;
            rtp.sunAngularRadius = 0.05f;
            rtp.raySamplesPerAxis = 4;
            rtp.cacheAcrossFrames = 0;

            std::vector<CpuGaussian> gs;
            for (uint32_t i = 0; i < scene.count; ++i)
            {
                const CpuGaussian g = cpuGaussian(scene, i);
                if (g.opacity >= 1.0 / 255.0)
                {
                    gs.push_back(g);
                }
            }

            Light sun;
            sun.type = LightType::Directional;
            sun.direction = glm::normalize(glm::vec3(0.8f, 0.0f, -1.0f));
            sun.intensity = 6.0f;
            Light rect;
            rect.type = LightType::Rect;
            rect.position = glm::vec3(0.0f, 0.0f, 1.5f);
            rect.axisX = glm::vec3(1.0f, 0.0f, 0.0f);
            rect.axisY = glm::vec3(0.0f, -1.0f, 0.0f); // emits along cross(x, -y): down, at the scene
            rect.halfWidth = 0.4f;
            rect.halfHeight = 0.25f;
            rect.intensity = 4.0f;

            // (1) Per pixel: the GPU's visibility lies in the CPU's borderline
            // interval (+- 1e-4) for ~200 covered pixels.
            const auto perPixel = [&](const char * name, const Light & light)
            {
                const Result res = render(r, {light}, ShadowMode::RayTraced, rtp);
                const Renderer::GBufferData gb = r.readGBuffer();
                const CpuLight cl = cpuLightOf(light, rtp);
                const uint32_t n = rtp.raySamplesPerAxis;
                uint32_t covered = 0;
                for (uint32_t p = 0; p < kWidth * kHeight; ++p)
                {
                    covered += res.coverage[p] > 0.5f ? 1u : 0u;
                }
                const uint32_t stride = std::max(1u, covered / 200u);
                uint32_t seen = 0, picked = 0, shadowed = 0;
                double worstOutside = 0.0, worstDiff = 0.0;
                for (uint32_t p = 0; p < kWidth * kHeight && picked < 200; ++p)
                {
                    if (res.coverage[p] <= 0.5f || seen++ % stride != 0)
                    {
                        continue;
                    }
                    ++picked;
                    glm::dvec3 N(0.0);
                    const glm::dvec3 P = cpuReceiver(gb, p, topCamera(), rtp, N);
                    double v = 0.0, lo = 0.0, hi = 0.0;
                    for (uint32_t s = 0; s < n * n; ++s)
                    {
                        const CpuTrace t = cpuEmitterSample(gs, cl, P, s, n);
                        v += t.transmittance;
                        lo += t.lo;
                        hi += t.hi;
                    }
                    v /= n * n;
                    lo /= n * n;
                    hi /= n * n;
                    const double gpu = res.visibility[p];
                    worstOutside = std::max(worstOutside, std::max({0.0, lo - 1e-4 - gpu, gpu - hi - 1e-4}));
                    worstDiff = std::max(worstDiff, std::fabs(v - gpu));
                    shadowed += v < 0.5 ? 1u : 0u;
                }
                check(picked >= 150 && shadowed >= 10 && worstOutside <= 1e-4,
                      std::string(name) + " ray-traced shadows == CPU port, per pixel " +
                          fmt("(%.0f px", static_cast<double>(picked)) + fmt(", %.0f shadowed", shadowed) +
                          fmt(", worst outside interval %.2g", worstOutside) +
                          fmt(", worst |dV| %.2g, tol 1e-4)", worstDiff));
            };
            perPixel("Point", pointAbove(1.5f, 6.0f));
            perPixel("Directional", sun);
            perPixel("Rect", rect);

            // (2) Rect PCSS against the ray-traced reference.
            {
                ShadowSettings pc = rtp;
                pc.mode = ShadowMode::PCSS;
                pc.mapResolution = 2048;
                const Result traced = render(r, {rect}, ShadowMode::RayTraced, rtp);
                const Result pcss = render(r, {rect}, ShadowMode::PCSS, pc);
                double gap = 0.0, count = 0.0;
                for (uint32_t p = 0; p < kWidth * kHeight; ++p)
                {
                    if (traced.coverage[p] > 0.5f)
                    {
                        gap += std::fabs(pcss.visibility[p] - traced.visibility[p]);
                        count += 1.0;
                    }
                }
                gap /= std::max(count, 1.0);
                check(gap < 0.08, "Rect PCSS agrees with the ray-traced reference " +
                                      fmt("(mean |dV| %.4f, bound 0.08, as the point light)", gap));
            }

            // (3) Directional penumbra. The ray-traced edge's 50% point sits at the
            // geometric edge for any sun angle (the ray-traced band is pixel-limited at
            // 0.057 units per pixel, hence the 50% point here). PCSS's band is checked
            // against its analytic width further down.
            {
                const auto rtBand = [&](float angle)
                {
                    ShadowSettings s = rtp;
                    s.sunAngularRadius = angle;
                    s.raySamplesPerAxis = 16;
                    s.depthBias = 0.02f;
                    return s;
                };
                // The 50% point of the far shadow edge, from the ground row through
                // the image centre (0.057 units per pixel), interpolated between pixels.
                // A symmetric disk light puts the 50% point of a sharp edge where the
                // edge's shadow ray meets it: ground x = 0.5 + 0.6 * 0.8 = 0.98 for this
                // slab. The slab is a sheet of splats (sigma 0.045, spacing 0.05), so
                // its transmittance edge sits outside the last centre by about 2 sigma,
                // i.e. up to 0.07 in ground x: the 50% point lies in [0.98, 1.05] and
                // must not move with the sun's angle.
                const auto halfPoint = [](const Result & res) -> double
                {
                    double prevX = 0.0, prevV = 0.0;
                    bool have = false;
                    for (uint32_t px = 0; px < kWidth; ++px)
                    {
                        const double gx = (static_cast<double>(px) - 0.5 * (kWidth - 1)) * 3.4 / 60.0;
                        const double v = res.visibility[48 * kWidth + px];
                        if (gx < 0.7 || gx > 1.2)
                        {
                            continue;
                        }
                        if (have && prevV < 0.5 && v >= 0.5)
                        {
                            return prevX + (0.5 - prevV) / (v - prevV) * (gx - prevX);
                        }
                        prevX = gx;
                        prevV = v;
                        have = true;
                    }
                    return -1.0;
                };
                const double edge05 = halfPoint(render(r, {sun}, ShadowMode::RayTraced, rtBand(0.05f)));
                const double edge10 = halfPoint(render(r, {sun}, ShadowMode::RayTraced, rtBand(0.1f)));
                check(edge05 >= 0.96 && edge05 <= 1.06 && std::fabs(edge05 - edge10) < 0.06,
                      "directional ray-traced shadow edge sits at the slab's edge and does not move with the sun's angle " +
                          fmt("(0.05 rad %.3f", edge05) + fmt(", 0.10 rad %.3f)", edge10));

                // PCSS's penumbra band at the far shadow edge, against the analytic width.
                // The 96 x 96 image's 0.057 units per pixel is coarser than this band at
                // 0.02 rad, so the band is measured on a 512 x 512 render of the same
                // scene (0.011 units per pixel). Along the ground row y = 0 the visibility
                // rises from the occluder's shadow (0) to lit (1) past the edge; the band
                // is the distance between its 10% and 90% crossings, interpolated between
                // pixel centres.
                //
                // Analytic width: the sun's disk of angular radius a, seen past an
                // occluder edge at height h, blurs the edge over +-R on the ground with
                // R = h * (1 + t^2) * tan(a), t = 0.8 the sun's tilt (ground x per unit
                // of light-plane offset). A uniform disk's 10% and 90% points lie at
                // +-0.6875 R, so band = 2 * 0.6875 * h * (1 + t^2) * tan(a), h = 0.6 here.
                constexpr uint32_t kFine = 512;
                Renderer fine(kFine, kFine);
                if (!fine.initialize({}, true, SHADER_DIR))
                {
                    check(false, "fine renderer initialize");
                }
                else
                {
                    Camera fc = topCamera();
                    fc.width = fc.height = kFine;
                    fc.fx = fc.fy = 60.0f * kFine / kWidth;
                    fc.cx = fc.cy = 0.5f * (kFine - 1);
                    fine.setCamera(fc);
                    fine.setLightingEnabled(true);
                    fine.setLightingSettings(ls);
                    fine.setGaussians(input(scene));
                    const uint32_t row = kFine / 2 - 1; // ground y = 0 sits between rows 255 and 256
                    const auto fineBand = [&](float angle)
                    {
                        ShadowSettings s = shadows;
                        s.mode = ShadowMode::PCSS;
                        s.sunAngularRadius = angle;
                        s.depthBias = 0.02f;
                        fine.setShadowSettings(s);
                        fine.setLights({sun});
                        fine.forward(RenderMode::Viewer);
                        const std::vector<float> all = fine.readVisibility();
                        const auto gx = [&](uint32_t px) { return (static_cast<double>(px) - fc.cx) * 3.4 / fc.fx; };
                        // The first rising crossing of `level` between ground x 0.6 and 1.5.
                        const auto crossing = [&](float level)
                        {
                            for (uint32_t px = 0; px + 1 < kFine; ++px)
                            {
                                const double x0 = gx(px), x1 = gx(px + 1);
                                const double v0 = all[row * kFine + px], v1 = all[row * kFine + px + 1];
                                if (x0 >= 0.6 && x1 <= 1.5 && v0 < level && v1 >= level)
                                {
                                    return x0 + (level - v0) / (v1 - v0) * (x1 - x0);
                                }
                            }
                            return -1.0;
                        };
                        const double x10 = crossing(0.1f), x90 = crossing(0.9f);
                        return x10 < 0.0 || x90 < 0.0 ? 0.0 : x90 - x10;
                    };
                    for (float angle : {0.02f, 0.1f})
                    {
                        const double measured = fineBand(angle);
                        const double expected = 2.0 * 0.6875 * 0.6 * (1.0 + 0.8 * 0.8) * std::tan(static_cast<double>(angle));
                        check(measured > expected / 1.5 && measured < expected * 1.5,
                              fmt("directional PCSS edge band, sun %.2f rad, within 1.5x of the analytic width ", angle) +
                                  fmt("(band %.4f", measured) + fmt(", analytic %.4f)", expected));
                    }
                }
            }

            // Unoccluded oblique ground must stay lit: the slab shadows only +x, so the
            // sun's side (x in [-2, -0.7]) and the far ground past the penumbra
            // (x in [1.3, 1.9]) see the whole sun, and PCSS must not self-shadow there.
            {
                for (float angle : {0.02f, 0.1f})
                {
                    ShadowSettings s = rtp;
                    s.mode = ShadowMode::PCSS;
                    s.sunAngularRadius = angle;
                    s.depthBias = 0.02f;
                    const Result pr = render(r, {sun}, ShadowMode::PCSS, s);
                    double minV = 2.0;
                    for (float x = -2.0f; x <= 1.9f; x += 0.02f)
                    {
                        if (x > -0.7f && x < 1.3f)
                        {
                            continue;
                        }
                        for (float y = -0.6f; y <= 0.6f; y += 0.02f)
                        {
                            minV = std::min<double>(minV, visibilityAt(pr, glm::vec2(x, y)));
                        }
                    }
                    check(minV >= 0.99, fmt("PCSS leaves unoccluded oblique ground lit, sun %.2f rad ", angle) +
                                            fmt("(min visibility %.4f, bound 0.99)", minV));
                }
            }

            // (4) PCSS filter and blocker sample sweeps: at every sample count
            // the penumbra grows with the light's size.
            {
                const auto sweep = [&](const char * what, uint32_t ShadowSettings::*field, uint32_t count)
                {
                    double previous = -1.0;
                    bool monotone = true;
                    std::string widths;
                    for (float radius : {0.01f, 0.06f, 0.15f})
                    {
                        ShadowSettings s2 = shadows;
                        s2.pointLightRadius = radius;
                        s2.*field = count;
                        const double w = penumbraWidth(render(r, overhead, ShadowMode::PCSS, s2));
                        widths += fmt(" %.3f", w);
                        monotone &= w >= previous - 0.005;
                        previous = w;
                    }
                    check(monotone && previous > 0.05,
                          std::string("PCSS ") + what + " " + std::to_string(count) +
                              " samples: penumbra grows with light size (widths:" + widths + ")");
                };
                for (uint32_t f : {2u, 4u, 8u, 16u})
                {
                    sweep("filter", &ShadowSettings::filterSamplesPerAxis, f);
                }
                for (uint32_t b : {1u, 2u, 4u, 8u})
                {
                    sweep("blocker", &ShadowSettings::blockerSamplesPerAxis, b);
                }
            }
        }

        // rayDownsample 3 and 4 on 97x95, where neither size divides the
        // image: the last block row and column are partial.
        if (r.deviceFeatures().rayQuery)
        {
            constexpr uint32_t kOddW = 97, kOddH = 95;
            constexpr uint32_t kOddPixels = kOddW * kOddH;
            Renderer odd(kOddW, kOddH);
            if (!odd.initialize({}, true, SHADER_DIR))
            {
                check(false, "odd-size renderer initialize");
            }
            else
            {
                Camera oc = topCamera();
                oc.width = kOddW;
                oc.height = kOddH;
                oc.cx = 0.5f * (kOddW - 1);
                oc.cy = 0.5f * (kOddH - 1);
                odd.setCamera(oc);
                odd.setLightingEnabled(true);
                odd.setLightingSettings(ls);
                odd.setGaussians(input(scene));
                odd.setLights(overhead);
                ShadowSettings base = shadows;
                base.mode = ShadowMode::RayTraced;
                base.pointLightRadius = 0.08f;
                base.raySamplesPerAxis = 4;
                base.cacheAcrossFrames = 0;
                const auto oddFrame = [&](uint32_t s, ShadowFrameStats & stats)
                {
                    ShadowSettings st = base;
                    st.rayDownsample = s;
                    odd.setShadowSettings(st);
                    odd.forward(RenderMode::Viewer);
                    stats = odd.shadowFrameStats();
                    const std::vector<float> all = odd.readVisibility();
                    return std::vector<float>(all.begin(), all.begin() + kOddPixels);
                };
                ShadowFrameStats exactStats;
                const std::vector<float> exact = oddFrame(1, exactStats);
                check(exactStats.emitterSamplesDispatched == 97ull * 95 * 16,
                      "rayDownsample 1 on 97x95: every pixel traces its 16 emitter samples");
                for (uint32_t s : {3u, 4u})
                {
                    ShadowFrameStats stats;
                    const std::vector<float> v = oddFrame(s, stats);
                    bool tracedExact = true;
                    double mean = 0.0, worst = 0.0, maxOut = 0.0;
                    for (uint32_t y = 0; y < kOddH; ++y)
                    {
                        for (uint32_t x = 0; x < kOddW; ++x)
                        {
                            const uint32_t k = y * kOddW + x;
                            if (x % s == 0 && y % s == 0)
                            {
                                tracedExact &= v[k] == exact[k];
                            }
                            maxOut = std::max(maxOut, std::max(static_cast<double>(v[k]) - 1.0,
                                                               -static_cast<double>(v[k])));
                            const double e = std::fabs(static_cast<double>(v[k]) - exact[k]);
                            mean += e;
                            worst = std::max(worst, e);
                        }
                    }
                    mean /= kOddPixels;
                    // Receivers traced: ceil(97 / s) * ceil(95 / s), 16 emitter samples each.
                    const uint64_t traced = static_cast<uint64_t>((kOddW + s - 1) / s) * ((kOddH + s - 1) / s);
                    const std::string name = "rayDownsample " + std::to_string(s) + " on 97x95";
                    check(tracedExact, name + ": traced pixels bit-exact vs downsample 1 " +
                                           fmt("(%.0f traced receivers)", static_cast<double>(traced)));
                    // The blend sums w*v and w in separate fp32 accumulations, so it can
                    // land a few ulps past 1 (exact downsample 1 never does); 1e-6 is ~8 ulps there.
                    check(maxOut <= 1e-6, name + ": visibility within [0,1] up to fp32 rounding " +
                                              fmt("(worst excess %.3g, bound 1e-6)", maxOut));
                    check(stats.emitterSamplesDispatched == traced * 16,
                          name + ": emitter samples dispatched == receivers x 16");
                    check(mean < 0.03, name + fmt(" mean |dV| vs exact %.4f", mean) + fmt(" (worst %.4f, bound 0.03)", worst));
                }
                odd.shutdown();
            }
        }

        // 7 (optional, `shadow_test <scene>`): a real scene end to end -- both
        // shadow modes must run, differ from "no shadows", and agree roughly.
        if (argc > 1)
        {
            r.loadScene(argv[1]);
            const SceneBounds bounds = r.sceneBounds();
            Camera cam = topCamera();
            cam.cameraToWorld[3] = glm::vec4(bounds.focusCenter + glm::vec3(0, 0, 2.5f * bounds.focusRadius), 1.0f);
            cam.fx = cam.fy = 1.2f * kWidth;
            r.setCamera(cam);
            std::vector<Light> sun(1);
            sun[0].type = LightType::Directional;
            sun[0].direction = glm::normalize(glm::vec3(-0.4f, -0.5f, -1.0f));
            sun[0].intensity = 3.0f;
            ShadowSettings real;
            real.raySamplesPerAxis = 4;
            const auto time = [&](ShadowMode mode)
            {
                const auto t0 = std::chrono::steady_clock::now();
                Result res = render(r, sun, mode, real);
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                return std::make_pair(res, ms);
            };
            auto none = time(ShadowMode::None);
            auto pcss = time(ShadowMode::PCSS);
            double meanPcss = 0.0, count = 0.0;
            for (uint32_t p = 0; p < kWidth * kHeight; ++p)
            {
                if (pcss.first.coverage[p] > 0.5f)
                {
                    meanPcss += pcss.first.visibility[p];
                    count += 1.0;
                }
            }
            meanPcss /= std::max(count, 1.0);
            check(count > 100 && meanPcss < 0.995 && meanPcss > 0.2,
                  std::string("real scene: PCSS shadows some of ") + argv[1] +
                      fmt(" (mean visibility %.3f", meanPcss) + fmt(", %.0f ms)", pcss.second));
            if (r.deviceFeatures().rayQuery)
            {
                auto rt = time(ShadowMode::RayTraced);
                double meanRt = 0.0, gap = 0.0;
                for (uint32_t p = 0; p < kWidth * kHeight; ++p)
                {
                    if (rt.first.coverage[p] > 0.5f)
                    {
                        meanRt += rt.first.visibility[p];
                        gap += std::fabs(rt.first.visibility[p] - pcss.first.visibility[p]);
                    }
                }
                meanRt /= std::max(count, 1.0);
                gap /= std::max(count, 1.0);
                check(meanRt < 0.995 && gap < 0.25,
                      "real scene: ray-traced shadows run and roughly agree with PCSS " +
                          fmt("(mean visibility %.3f", meanRt) + fmt(", mean |dV| %.3f", gap) + fmt(", %.0f ms)", rt.second));
            }
            (void)none;
        }

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
