// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Splatted G-buffer + deferred lighting, verified numerically.
//
// The colour pass is the reference: forward_test proves it pixel-exact
// against a CPU rasterizer. The G-buffer walks the same sorted lists with the
// same weights, so rendering the SAME geometry with each Gaussian's colour
// set to some per-Gaussian value v_i (via a degree-0 SH DC term) must
// reproduce the G-buffer channel that splats v_i:
//
//  1. coverage == the colour image's alpha, bit for bit;
//  2. base colour, roughness, metallic, normal (as n*0.5+0.5), depth/10 and
//     ambient visibility each equal the colour splat of that value;
//  3. scenes without relightable streams get the documented defaults
//     (DC colour, roughness 0.8, metallic 0, ambient 1, shortest-axis normal
//     facing the camera);
//  4. a relightable .ply (pre-activation base_color/roughness, nx/ny/nz,
//     visibility SH) loads into the same G-buffer as the activated streams
//     given directly; a plain splat .ply's zero normals are ignored;
//  5. the lit image equals a CPU port of deferred_lighting run on the
//     G-buffer readback, for every light type + an environment, and every
//     LightingOutput mode.
//
// Validation messages are logged as "[validation]"; the run must have none.

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "sabercat/camera.h"
#include "sabercat/lighting.h"
#include "sabercat/render_config.h"
#include "sabercat/renderer.h"

#include "../common/shader_dir.h"

using namespace sabercat;

namespace
{

constexpr uint32_t kWidth = 64;
constexpr uint32_t kHeight = 48;
constexpr uint32_t kCount = 14;
constexpr float kPi = 3.14159265358979f;

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

// ---- Scene ----

struct Scene
{
    std::vector<glm::vec4> means, scales, rotations;
    std::vector<float> opacities;
    std::vector<glm::vec4> sh; // 16 per Gaussian
    // Relightable streams.
    std::vector<glm::vec4> normals, baseColorRoughness;
    std::vector<float> metallic, visibility;
};

Camera makeCamera()
{
    Camera c;
    c.width = kWidth;
    c.height = kHeight;
    c.fx = c.fy = 52.0f;
    c.cx = 0.5f * (kWidth - 1);
    c.cy = 0.5f * (kHeight - 1);
    c.cameraToWorld = glm::mat4(1.0f);
    c.cameraToWorld[3] = glm::vec4(0.05f, -0.02f, 3.0f, 1.0f);
    return c;
}

glm::vec3 cameraPosition()
{
    return glm::vec3(makeCamera().cameraToWorld[3]);
}

uint32_t g_rng = 2024u;
float rnd()
{
    g_rng = g_rng * 1664525u + 1013904223u;
    return static_cast<float>(g_rng >> 8) / static_cast<float>(1u << 24);
}

// Overlapping, anisotropic, all visible.
Scene makeScene()
{
    Scene s;
    for (uint32_t i = 0; i < kCount; ++i)
    {
        const float t = static_cast<float>(i);
        const glm::vec3 mean(-0.9f + 0.14f * t, 0.35f * std::sin(1.7f * t), -0.4f * std::cos(0.9f * t));
        s.means.push_back(glm::vec4(mean, 0.0f));
        s.scales.push_back(glm::vec4(0.10f + 0.02f * (i % 3), 0.07f + 0.015f * (i % 4), 0.03f, 0.0f));
        glm::vec4 q = glm::normalize(glm::vec4(0.1f * t, 0.3f, -0.2f + 0.05f * t, 1.0f));
        s.rotations.push_back(q);
        s.opacities.push_back(0.45f + 0.05f * static_cast<float>(i % 9));
        for (uint32_t k = 0; k < 16; ++k)
        {
            s.sh.push_back(k == 0 ? glm::vec4(0.3f * std::sin(t), 0.3f * std::cos(t), 0.2f, 0.0f)
                                  : glm::vec4(0.0f));
        }
        // A normal facing the camera (so relight_prepare's flip is a no-op).
        glm::vec3 n = glm::normalize(glm::vec3(rnd() - 0.5f, rnd() - 0.5f, 1.0f));
        if (glm::dot(n, cameraPosition() - mean) < 0.0f)
        {
            n = -n;
        }
        s.normals.push_back(glm::vec4(n, 0.0f));
        s.baseColorRoughness.push_back(glm::vec4(0.1f + 0.8f * rnd(), 0.1f + 0.8f * rnd(),
                                                 0.1f + 0.8f * rnd(), 0.05f + 0.9f * rnd()));
        s.metallic.push_back(rnd());
        // Visibility: a smooth, partly occluded function, projected to SH.
        const glm::vec3 axis = glm::normalize(glm::vec3(rnd() - 0.5f, rnd() - 0.5f, rnd() - 0.5f));
        float coeff[16] = {};
        const int kT = 48, kP = 96;
        for (int a = 0; a < kT; ++a)
        {
            const float th = (a + 0.5f) * kPi / kT;
            for (int b = 0; b < kP; ++b)
            {
                const float ph = (b + 0.5f) * 2.0f * kPi / kP;
                const glm::vec3 d(std::sin(th) * std::cos(ph), std::cos(th), std::sin(th) * std::sin(ph));
                const float v = 0.5f + 0.5f * std::tanh(4.0f * glm::dot(d, axis));
                float basis[16];
                shBasis16(d, basis);
                for (int k = 0; k < 16; ++k)
                {
                    coeff[k] += v * basis[k] * std::sin(th) * (kPi / kT) * (2.0f * kPi / kP);
                }
            }
        }
        s.visibility.insert(s.visibility.end(), coeff, coeff + 16);
    }
    return s;
}

GaussianHostInput input(const Scene & s, bool relight)
{
    GaussianHostInput in;
    in.count = kCount;
    in.means = &s.means[0].x;
    in.scales = &s.scales[0].x;
    in.rotations = &s.rotations[0].x;
    in.opacities = s.opacities.data();
    in.shCoefficients = &s.sh[0].x;
    in.shDegree = 0;
    if (relight)
    {
        in.normals = &s.normals[0].x;
        in.baseColorRoughness = &s.baseColorRoughness[0].x;
        in.metallic = s.metallic.data();
        in.visibility = s.visibility.data();
    }
    return in;
}

// The colour pass's splat of per-Gaussian values (each component in [0,1]).
std::vector<float> splatValues(Renderer & r, const Scene & s, const std::vector<glm::vec3> & values)
{
    Scene c = s;
    for (uint32_t i = 0; i < kCount; ++i)
    {
        c.sh[i * 16] = glm::vec4((values[i] - 0.5f) / kShC0, 0.0f);
    }
    r.setGaussians(input(c, false));
    r.forward(RenderMode::Differentiable);
    return r.readImage();
}

// Worst |a - b| over channel `ca` of `a` (float4 per pixel) vs `b` mapped.
template <typename F>
double worstDiff(const std::vector<float> & image, const F & gbufferValue, int channel)
{
    double worst = 0.0;
    for (uint32_t p = 0; p < kWidth * kHeight; ++p)
    {
        worst = std::max(worst, std::fabs(static_cast<double>(image[p * 4 + channel]) - gbufferValue(p)));
    }
    return worst;
}

float ambientOf(const Scene & s, uint32_t i, const glm::vec3 & n)
{
    float basis[16];
    shBasis16(n, basis);
    const float band[4] = {1.0f, 2.0f / 3.0f, 0.25f, 0.0f};
    float sum = 0.0f;
    for (int k = 0; k < 16; ++k)
    {
        const int l = k == 0 ? 0 : (k < 4 ? 1 : (k < 9 ? 2 : 3));
        sum += band[l] * s.visibility[i * 16 + k] * basis[k];
    }
    return std::clamp(sum, 0.0f, 1.0f);
}

float depthOf(const Scene & s, uint32_t i)
{
    const glm::mat4 worldToCamera = glm::inverse(makeCamera().cameraToWorld);
    return -(worldToCamera * glm::vec4(glm::vec3(s.means[i]), 1.0f)).z;
}

// ---- CPU port of deferred_lighting.slang ----

glm::vec3 fresnel(float c, glm::vec3 f0)
{
    return f0 + (1.0f - f0) * std::pow(1.0f - std::clamp(c, 0.0f, 1.0f), 5.0f);
}

glm::vec3 brdfCos(glm::vec3 N, glm::vec3 V, glm::vec3 L, glm::vec3 base, float rough, float metal)
{
    const float NdotL = glm::dot(N, L);
    if (NdotL <= 0.0f)
    {
        return glm::vec3(0.0f);
    }
    const float NdotV = std::max(glm::dot(N, V), 1e-4f);
    const glm::vec3 H = glm::normalize(L + V);
    const float NdotH = std::clamp(glm::dot(N, H), 0.0f, 1.0f);
    const float VdotH = std::clamp(glm::dot(V, H), 0.0f, 1.0f);
    const float a = std::max(rough * rough, 1e-3f);
    const float a2 = a * a;
    const glm::vec3 f0 = glm::mix(glm::vec3(0.04f), base, metal);
    const glm::vec3 F = fresnel(VdotH, f0);
    const float d = NdotH * NdotH * (a2 - 1.0f) + 1.0f;
    const float D = a2 / std::max(kPi * d * d, 1e-12f);
    const float gv = NdotL * std::sqrt(NdotV * NdotV * (1.0f - a2) + a2);
    const float gl = NdotV * std::sqrt(NdotL * NdotL * (1.0f - a2) + a2);
    const float Vis = 0.5f / std::max(gv + gl, 1e-12f);
    const glm::vec3 spec = F * D * Vis;
    const glm::vec3 diff = (1.0f - F) * (1.0f - metal) * base / kPi;
    return (diff + spec) * NdotL;
}

glm::vec3 envSH(const EnvironmentLight & env, glm::vec3 dir, bool convolve)
{
    float b[16];
    shBasis16(dir, b);
    glm::vec3 sum(0.0f);
    for (int k = 0; k < 9; ++k)
    {
        const float w = !convolve ? 1.0f : (k == 0 ? 1.0f : (k < 4 ? 2.0f / 3.0f : 0.25f));
        sum += w * env.sh[k] * env.intensity * b[k];
    }
    return glm::max(sum, glm::vec3(0.0f));
}

// `vis` is the visibility buffer (layer i = light i, width*height floats each).
glm::vec4 cpuLighting(const Renderer::GBufferData & g, uint32_t p, const Camera & cam,
                      const std::vector<Light> & lights, const EnvironmentLight & env, const LightingSettings & st,
                      const std::vector<float> & vis)
{
    const uint32_t layerSize = cam.width * cam.height;
    const glm::vec4 nd = g.normalDepth[p], br = g.baseRoughness[p], mac = g.metalAmbientCoverage[p];
    const float coverage = mac.z;
    glm::vec3 color(0.0f);
    if (coverage > 1e-6f)
    {
        const float inv = 1.0f / coverage;
        const float nLen = glm::length(glm::vec3(nd));
        const glm::vec3 N = nLen > 1e-8f ? glm::vec3(nd) / nLen : glm::vec3(0, 0, 1);
        const float depth = nd.w * inv;
        const glm::vec3 base = glm::clamp(glm::vec3(br) * inv, 0.0f, 1.0f);
        const float rough = std::clamp(br.w * inv, 0.0f, 1.0f);
        const float metal = std::clamp(mac.x * inv, 0.0f, 1.0f);
        const float ambient = std::clamp(mac.y * inv, 0.0f, 1.0f);
        const float u = static_cast<float>(p % cam.width), v = static_cast<float>(p / cam.width);
        const glm::vec3 local((u - cam.cx) * depth / cam.fx, (cam.cy - v) * depth / cam.fy, -depth);
        const glm::vec3 P = glm::vec3(cam.cameraToWorld * glm::vec4(local, 1.0f));
        const glm::vec3 V = glm::normalize(glm::vec3(cam.cameraToWorld[3]) - P);
        switch (st.output)
        {
        case LightingOutput::Lit:
        {
            glm::vec3 lit(0.0f);
            for (size_t li = 0; li < lights.size(); ++li)
            {
                const Light & l = lights[li];
                const glm::vec3 radiance = l.color * l.intensity;
                glm::vec3 c(0.0f);
                if (l.type == LightType::Directional)
                {
                    c = brdfCos(N, V, -glm::normalize(l.direction), base, rough, metal) * radiance;
                }
                else if (l.type == LightType::Point)
                {
                    const glm::vec3 to = l.position - P;
                    const float d2 = std::max(glm::dot(to, to), 1e-8f);
                    c = brdfCos(N, V, to / std::sqrt(d2), base, rough, metal) * radiance / d2;
                }
                else
                {
                    const glm::vec3 ax = glm::normalize(l.axisX) * l.halfWidth;
                    const glm::vec3 ay = glm::normalize(l.axisY) * l.halfHeight;
                    const glm::vec3 en = glm::normalize(glm::cross(ax, ay));
                    const float area = 4.0f * glm::length(ax) * glm::length(ay);
                    const uint32_t n = std::max(
                        1u, static_cast<uint32_t>(std::lround(std::sqrt(static_cast<double>(st.rectSamples)))));
                    const float sa = area / static_cast<float>(n * n);
                    for (uint32_t sy = 0; sy < n; ++sy)
                    {
                        for (uint32_t sx = 0; sx < n; ++sx)
                        {
                            const float a = (sx + 0.5f) / n * 2.0f - 1.0f;
                            const float b = (sy + 0.5f) / n * 2.0f - 1.0f;
                            const glm::vec3 S = l.position + a * ax + b * ay;
                            const glm::vec3 to = S - P;
                            const float d2 = std::max(glm::dot(to, to), 1e-8f);
                            const glm::vec3 L = to / std::sqrt(d2);
                            const float cl = glm::dot(en, -L);
                            if (cl <= 0.0f)
                            {
                                continue;
                            }
                            c += brdfCos(N, V, L, base, rough, metal) * cl * sa / d2;
                        }
                    }
                    c *= radiance;
                }
                lit += c * vis[li * layerSize + p];
            }
            const glm::vec3 irradiance = kPi * envSH(env, N, true);
            const float NdotV = std::max(glm::dot(N, V), 1e-4f);
            const glm::vec3 f0 = glm::mix(glm::vec3(0.04f), base, metal);
            const glm::vec3 F = fresnel(NdotV, f0);
            const glm::vec3 diffuse = (1.0f - F) * (1.0f - metal) * base / kPi * irradiance;
            const glm::vec3 R = glm::reflect(-V, N);
            const glm::vec3 spec = F * envSH(env, R, false) * (1.0f - 0.5f * rough);
            lit += ambient * (diffuse + spec);
            color = lit;
            break;
        }
        case LightingOutput::Normal: color = N * 0.5f + 0.5f; break;
        case LightingOutput::BaseColor: color = base; break;
        case LightingOutput::Roughness: color = glm::vec3(rough); break;
        case LightingOutput::Metallic: color = glm::vec3(metal); break;
        case LightingOutput::AmbientOcclusion: color = glm::vec3(ambient); break;
        case LightingOutput::Visibility: color = glm::vec3(vis[st.visibilityLight * layerSize + p]); break;
        case LightingOutput::Depth: color = glm::vec3(depth * st.depthScale); break;
        }
    }
    const glm::vec3 composed = (color * coverage + st.background * (1.0f - coverage)) * st.exposure;
    return glm::vec4(composed, coverage);
}

// ---- Relightable .ply writer (pre-activation values, as a trainer saves them) ----

float logit(float y)
{
    return std::log(y / (1.0f - y));
}

void writeRelightablePly(const std::string & path, const Scene & s, bool relightable)
{
    std::vector<std::string> props = {"x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
                                      "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1",
                                      "rot_2", "rot_3"};
    if (relightable)
    {
        for (const char * p : {"base_color_0", "base_color_1", "base_color_2", "roughness", "visibility_dc_0"})
        {
            props.push_back(p);
        }
        for (int k = 0; k < 15; ++k)
        {
            props.push_back("visibility_rest_" + std::to_string(k));
        }
    }
    std::ofstream f(path, std::ios::binary);
    f << "ply\nformat binary_little_endian 1.0\nelement vertex " << kCount << "\n";
    for (const std::string & p : props)
    {
        f << "property float " << p << "\n";
    }
    f << "end_header\n";
    for (uint32_t i = 0; i < kCount; ++i)
    {
        const glm::vec4 & q = s.rotations[i];
        // A plain splat .ply writes zero normals; a relightable one writes real ones.
        const glm::vec3 n = relightable ? glm::vec3(s.normals[i]) : glm::vec3(0.0f);
        std::vector<float> v = {s.means[i].x, s.means[i].y, s.means[i].z, n.x, n.y, n.z,
                                s.sh[i * 16].x, s.sh[i * 16].y, s.sh[i * 16].z, logit(s.opacities[i]),
                                std::log(s.scales[i].x), std::log(s.scales[i].y), std::log(s.scales[i].z),
                                q.w, q.x, q.y, q.z};
        if (relightable)
        {
            const glm::vec4 & b = s.baseColorRoughness[i];
            v.push_back(logit((b.x - 0.03f) / 0.77f));
            v.push_back(logit((b.y - 0.03f) / 0.77f));
            v.push_back(logit((b.z - 0.03f) / 0.77f));
            v.push_back(logit((b.w - 0.09f) / 0.9f));
            for (int k = 0; k < 16; ++k)
            {
                v.push_back(s.visibility[i * 16 + k]);
            }
        }
        f.write(reinterpret_cast<const char *>(v.data()), static_cast<std::streamsize>(v.size() * 4));
    }
}

// ---- SH basis and environment, checked without the renderer ----

// Closed-form real SH up to degree 3 in double, from the standard coefficients
// (not shBasis16's code), so a shared mistake in one cannot cancel.
void shClosedForm(double x, double y, double z, double b[16])
{
    const double c1 = 0.4886025119029199;
    const double c2[5] = {1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
                          -1.0925484305920792, 0.5462742152960396};
    const double c3[7] = {-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154,
                          -0.4570457994644658, 1.445305721320277, -0.5900435899266435};
    const double xx = x * x, yy = y * y, zz = z * z;
    b[0] = 0.28209479177387814;
    b[1] = -c1 * y;
    b[2] = c1 * z;
    b[3] = -c1 * x;
    b[4] = c2[0] * x * y;
    b[5] = c2[1] * y * z;
    b[6] = c2[2] * (2.0 * zz - xx - yy);
    b[7] = c2[3] * x * z;
    b[8] = c2[4] * (xx - yy);
    b[9] = c3[0] * y * (3.0 * xx - yy);
    b[10] = c3[1] * x * y * z;
    b[11] = c3[2] * y * (4.0 * zz - xx - yy);
    b[12] = c3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy);
    b[13] = c3[4] * x * (4.0 * zz - xx - yy);
    b[14] = c3[5] * z * (xx - yy);
    b[15] = c3[6] * x * (xx - 3.0 * yy);
}

void checkBasisAndEnvironment()
{
    uint32_t state = 77u;
    const auto unit = [&state]() // [0, 1)
    {
        state = state * 1103515245u + 12345u;
        return static_cast<double>(state >> 8) / static_cast<double>(1u << 24);
    };
    const auto randomDir = [&unit]()
    {
        const double z = 2.0 * unit() - 1.0, phi = 6.283185307179586 * unit();
        const double s = std::sqrt(std::max(0.0, 1.0 - z * z));
        return glm::vec3(static_cast<float>(s * std::cos(phi)), static_cast<float>(z),
                         static_cast<float>(s * std::sin(phi)));
    };

    // Orthonormality: midpoint quadrature over the sphere (uniform in
    // cos(theta) and phi), in double. Error is O(h^2) in the latitude bins.
    {
        constexpr int kU = 2048, kP = 512;
        double gram[16][16] = {};
        for (int a = 0; a < kU; ++a)
        {
            const double u = -1.0 + (a + 0.5) * 2.0 / kU;
            const double s = std::sqrt(1.0 - u * u);
            for (int q = 0; q < kP; ++q)
            {
                const double phi = (q + 0.5) * 6.283185307179586 / kP;
                double b[16];
                shClosedForm(s * std::cos(phi), u, s * std::sin(phi), b);
                const double w = (2.0 / kU) * (6.283185307179586 / kP);
                for (int i = 0; i < 16; ++i)
                {
                    for (int j = 0; j < 16; ++j)
                    {
                        gram[i][j] += b[i] * b[j] * w;
                    }
                }
            }
        }
        double worst = 0.0;
        for (int i = 0; i < 16; ++i)
        {
            for (int j = 0; j < 16; ++j)
            {
                worst = std::max(worst, std::fabs(gram[i][j] - (i == j ? 1.0 : 0.0)));
            }
        }
        check(worst < 1e-4, "SH closed form is orthonormal over the sphere " + fmt("(worst %.3g, bound 1e-4)", worst));
    }

    // shBasis16 (float) against the closed form (double) in 500 directions:
    // each term is a few float products of unit-scale values, so the error
    // stays within a few ulps times the term count, far below 1e-6.
    {
        double worst = 0.0;
        for (int n = 0; n < 500; ++n)
        {
            const glm::vec3 d = randomDir();
            float f[16];
            shBasis16(d, f);
            double b[16];
            shClosedForm(d.x, d.y, d.z, b);
            for (int k = 0; k < 16; ++k)
            {
                worst = std::max(worst, std::fabs(static_cast<double>(f[k]) - b[k]));
            }
        }
        check(worst < 1e-6, "shBasis16 == closed-form SH in 500 directions " + fmt("(worst %.3g, bound 1e-6)", worst));
    }

    // Uniform environment: constant radiance from every direction, scaled by intensity.
    {
        const glm::vec3 c(0.7f, 0.2f, 1.3f);
        EnvironmentLight env = EnvironmentLight::uniform(c);
        double worst = 0.0;
        for (int n = 0; n < 200; ++n)
        {
            const glm::vec3 got = env.radiance(randomDir());
            for (int k = 0; k < 3; ++k)
            {
                worst = std::max(worst, std::fabs(static_cast<double>(got[k]) - c[k]));
            }
        }
        // Only the DC term is non-zero: one product c/kShC0 * kShC0 * basis[0], a few float roundings.
        check(worst < 1e-6, "EnvironmentLight::uniform radiance == the constant from every direction " +
                                fmt("(worst %.3g)", worst));
        env.intensity = 2.0f;
        double worstScaled = 0.0;
        for (int n = 0; n < 200; ++n)
        {
            const glm::vec3 got = env.radiance(randomDir());
            for (int k = 0; k < 3; ++k)
            {
                worstScaled = std::max(worstScaled, std::fabs(static_cast<double>(got[k]) - 2.0 * c[k]));
            }
        }
        check(worstScaled < 1e-6, "EnvironmentLight intensity scales the radiance " + fmt("(worst %.3g)", worstScaled));
    }

    // Arbitrary degree-2 coefficients: radiance == sum of closed-form terms times intensity.
    {
        EnvironmentLight env;
        env.intensity = 1.5f;
        for (int k = 0; k < 9; ++k)
        {
            env.sh[k] = glm::vec3(2.0f * unit() - 1.0, 2.0f * unit() - 1.0, 2.0f * unit() - 1.0);
        }
        double worst = 0.0;
        for (int n = 0; n < 200; ++n)
        {
            const glm::vec3 d = randomDir();
            double b[16];
            shClosedForm(d.x, d.y, d.z, b);
            const glm::vec3 got = env.radiance(d);
            for (int k = 0; k < 3; ++k)
            {
                double want = 0.0;
                for (int j = 0; j < 9; ++j)
                {
                    want += static_cast<double>(env.sh[j][k]) * b[j];
                }
                want *= 1.5;
                worst = std::max(worst, std::fabs(static_cast<double>(got[k]) - want));
            }
        }
        // Coefficients and terms are O(1); 9 products per channel in float: ~1e-6 in practice.
        check(worst < 1e-5, "EnvironmentLight::radiance == closed-form SH sum " + fmt("(worst %.3g, bound 1e-5)", worst));
    }
}

// EnvironmentLight::sky(A, B, B, up = y) is B + (A - B) * max(y, 0), so its
// nine coefficients (sh_k = integral of L * b_k) are analytic:
//   sh0 = 2 sqrt(pi) B + pi C0 (A - B),  sh1 = -(2 pi / 3) C1 (A - B),
//   sh6 = -(pi / 4) C2_2 (A - B),        sh8 = -(pi / 4) C2_4 (A - B),
//   every other k < 9 is zero.
// (integral over the upper hemisphere of y^2 is 2 pi / 3; of y (2z^2 - x^2 - y^2)
// and of y (x^2 - y^2) are both -pi / 4 after the azimuthal average.) The
// projection is a 1024 x 256 midpoint grid, uniform in cos(theta). Its error on
// the polynomial terms is ~(dU)^2 / 2 * 2 pi * C2 with dU = 2 / 1024, i.e. ~3e-6
// absolute, under the 1e-5 bound. The horizon kink (u = 0) is a bin edge for an
// even bin count, so the non-uniform sky has the same order of error; it is
// checked at 1e-4, a few times that estimate.
void checkSky()
{
    const double pi = 3.14159265358979323846;
    const double c0 = 0.28209479177387814, c1 = 0.4886025119029199, c22 = 0.31539156525252005,
                 c24 = 0.5462742152960396;
    const glm::vec3 zenith(0.4f, 0.5f, 0.9f), base(0.8f, 0.7f, 0.6f);
    for (int variant = 0; variant < 2; ++variant)
    {
        const glm::vec3 a = variant == 0 ? zenith : base; // variant 1: uniform (A = B)
        const EnvironmentLight env = EnvironmentLight::sky(a, base, base, glm::vec3(0.0f, 1.0f, 0.0f));
        double worst = 0.0;
        for (int c = 0; c < 3; ++c)
        {
            const double d = static_cast<double>(zenith[c] - base[c]) * (variant == 0 ? 1.0 : 0.0);
            const double b = base[c];
            const double expect[9] = {2.0 * std::sqrt(pi) * b + pi * c0 * d, -(2.0 * pi / 3.0) * c1 * d, 0.0, 0.0,
                                      0.0, 0.0, -(pi / 4.0) * c22 * d, 0.0, -(pi / 4.0) * c24 * d};
            for (int k = 0; k < 9; ++k)
            {
                worst = std::max(worst, std::fabs(static_cast<double>(env.sh[k][c]) - expect[k]));
            }
        }
        const std::string name = variant == 0 ? "EnvironmentLight::sky (zenith != horizon = ground)"
                                              : "EnvironmentLight::sky with zenith = horizon = ground";
        check(worst < (variant == 0 ? 1e-4 : 1e-5), name + " == analytic SH coefficients " +
                                                         fmt("(worst %.3g, bound ", worst) +
                                                         (variant == 0 ? "1e-4)" : "1e-5)"));
    }
}

// A single Gaussian rendered alone. Its G-buffer normal is the blend of one
// splat, so nd / coverage is exactly that Gaussian's normal.
struct OneSplat
{
    glm::vec4 mean{0.0f};
    glm::vec4 scale{0.1f, 0.1f, 0.1f, 0.0f};
    glm::vec4 rotation{0.0f, 0.0f, 0.0f, 1.0f}; // x, y, z, w
    float opacity = 0.9f;
    bool hasNormal = false;
    glm::vec4 normal{0.0f, 0.0f, 1.0f, 0.0f};
};

// relight_prepare's normal on the CPU: the given normal, or the shortest
// ellipsoid axis (first on ties, x before y before z), then flipped to face
// the camera.
glm::vec3 cpuPreparedNormal(const OneSplat & g, const glm::vec3 & cameraPos)
{
    glm::vec3 n;
    if (g.hasNormal)
    {
        n = glm::normalize(glm::vec3(g.normal));
    }
    else
    {
        const glm::mat3 R = glm::mat3_cast(glm::quat(g.rotation.w, g.rotation.x, g.rotation.y, g.rotation.z));
        const glm::vec3 s = glm::vec3(g.scale);
        const int axis = (s.x <= s.y && s.x <= s.z) ? 0 : (s.y <= s.z ? 1 : 2);
        n = glm::normalize(R[axis]);
    }
    if (glm::dot(n, cameraPos - glm::vec3(g.mean)) < 0.0f)
    {
        n = -n;
    }
    return n;
}

// Worst |G-buffer normal - expected| over the covered pixels.
double oneSplatNormalError(Renderer & r, const Camera & cam, const OneSplat & input1, const glm::vec3 & expected,
                           int & covered)
{
    OneSplat g = input1;
    glm::vec4 sh[16];
    for (glm::vec4 & c : sh)
    {
        c = glm::vec4(0.0f);
    }
    sh[0] = glm::vec4(0.3f, 0.3f, 0.3f, 0.0f);
    GaussianHostInput in;
    in.count = 1;
    in.means = &g.mean.x;
    in.scales = &g.scale.x;
    in.rotations = &g.rotation.x;
    in.opacities = &g.opacity;
    in.shCoefficients = &sh[0].x;
    in.shDegree = 0;
    if (g.hasNormal)
    {
        in.normals = &g.normal.x;
    }
    r.setCamera(cam);
    r.setGaussians(in);
    r.forward(RenderMode::Differentiable);
    const Renderer::GBufferData gb = r.readGBuffer();
    double worst = 0.0;
    covered = 0;
    for (uint32_t p = 0; p < kWidth * kHeight; ++p)
    {
        const float cov = gb.metalAmbientCoverage[p].z;
        if (cov <= 1e-6f)
        {
            continue;
        }
        ++covered;
        for (int c = 0; c < 3; ++c)
        {
            worst = std::max(worst, std::fabs(static_cast<double>(gb.normalDepth[p][c] / cov) - expected[c]));
        }
    }
    return worst;
}

} // namespace

int main()
{
    try
    {
        checkBasisAndEnvironment();
        checkSky();

        Renderer r(kWidth, kHeight);
        if (!r.initialize({}, true, testShaderDir()))
        {
            std::printf("[FAIL] renderer initialize\n");
            return 1;
        }
        r.setCamera(makeCamera());
        r.setLightingEnabled(true);
        Scene scene = makeScene();
        // The relightable .ply's activation ranges, so the round trip (4) is invertible.
        for (glm::vec4 & b : scene.baseColorRoughness)
        {
            b = glm::vec4(glm::clamp(glm::vec3(b), 0.04f, 0.79f), std::clamp(b.w, 0.1f, 0.98f));
        }

        // ---- 1 + 2: relightable streams ----
        r.setGaussians(input(scene, true));
        r.forward(RenderMode::Differentiable);
        const std::vector<float> colour = r.readImage();
        const Renderer::GBufferData g = r.readGBuffer();
        {
            bool exact = true;
            uint32_t covered = 0;
            for (uint32_t p = 0; p < kWidth * kHeight; ++p)
            {
                exact &= g.metalAmbientCoverage[p].z == colour[p * 4 + 3];
                covered += colour[p * 4 + 3] > 0.01f ? 1u : 0u;
            }
            check(covered > kWidth * kHeight / 8,
                  "scene covers enough pixels (" + std::to_string(covered) + " above 1% coverage)");
            check(exact, "G-buffer coverage == colour alpha, bit for bit");
        }
        const auto channelCheck = [&](const std::string & name, const std::vector<glm::vec3> & values,
                                      auto gbufferValue, int channels)
        {
            const std::vector<float> ref = splatValues(r, scene, values);
            double worst = 0.0;
            for (int c = 0; c < channels; ++c)
            {
                worst = std::max(worst, worstDiff(ref, [&](uint32_t p) { return gbufferValue(p, c); }, c));
            }
            check(worst <= 2e-6, name + " == colour splat of the per-Gaussian value " + fmt("(worst %.3g)", worst));
        };
        std::vector<glm::vec3> vals(kCount);
        for (uint32_t i = 0; i < kCount; ++i) vals[i] = glm::vec3(scene.baseColorRoughness[i]);
        channelCheck("base colour", vals, [&](uint32_t p, int c) { return g.baseRoughness[p][c]; }, 3);
        for (uint32_t i = 0; i < kCount; ++i) vals[i] = glm::vec3(scene.baseColorRoughness[i].w);
        channelCheck("roughness", vals, [&](uint32_t p, int) { return g.baseRoughness[p].w; }, 1);
        for (uint32_t i = 0; i < kCount; ++i) vals[i] = glm::vec3(scene.metallic[i]);
        channelCheck("metallic", vals, [&](uint32_t p, int) { return g.metalAmbientCoverage[p].x; }, 1);
        for (uint32_t i = 0; i < kCount; ++i) vals[i] = glm::vec3(scene.normals[i]) * 0.5f + 0.5f;
        channelCheck("normal", vals,
                     [&](uint32_t p, int c)
                     { return 0.5 * g.normalDepth[p][c] + 0.5 * g.metalAmbientCoverage[p].z; }, 3);
        for (uint32_t i = 0; i < kCount; ++i) vals[i] = glm::vec3(depthOf(scene, i) / 10.0f);
        channelCheck("depth", vals, [&](uint32_t p, int) { return g.normalDepth[p].w / 10.0; }, 1);
        for (uint32_t i = 0; i < kCount; ++i) vals[i] = glm::vec3(ambientOf(scene, i, glm::vec3(scene.normals[i])));
        channelCheck("ambient visibility (SH cosine convolution)", vals,
                     [&](uint32_t p, int) { return g.metalAmbientCoverage[p].y; }, 1);
        {
            // Fully visible (V = 1, projected exactly) gives ambient 1.
            float fullyVisible = 4.0f * kPi * kShC0 * kShC0;
            check(std::fabs(fullyVisible - 1.0f) < 1e-6f, "V = 1 projects to ambient 1 (DC only)");
        }

        // ---- 3: derived defaults (no relightable streams) ----
        r.setGaussians(input(scene, false));
        r.forward(RenderMode::Differentiable);
        const Renderer::GBufferData d = r.readGBuffer();
        {
            const std::vector<float> dcColour = r.readImage(); // degree 0: exactly the DC colour
            double worstBase = 0.0, worstOther = 0.0;
            for (uint32_t p = 0; p < kWidth * kHeight; ++p)
            {
                const float cov = d.metalAmbientCoverage[p].z;
                for (int c = 0; c < 3; ++c)
                {
                    worstBase = std::max(worstBase, std::fabs(double(d.baseRoughness[p][c]) - dcColour[p * 4 + c]));
                }
                worstOther = std::max({worstOther, std::fabs(double(d.baseRoughness[p].w) - kDefaultRoughness * cov),
                                       std::fabs(double(d.metalAmbientCoverage[p].x)),
                                       std::fabs(double(d.metalAmbientCoverage[p].y) - cov)});
            }
            check(worstBase <= 2e-6, "default base colour == the DC colour " + fmt("(worst %.3g)", worstBase));
            check(worstOther <= 2e-6, "default roughness 0.8, metallic 0, ambient 1 " + fmt("(worst %.3g)", worstOther));
            std::vector<glm::vec3> derived(kCount);
            for (uint32_t i = 0; i < kCount; ++i)
            {
                const glm::mat3 R = glm::mat3_cast(glm::quat(scene.rotations[i].w, scene.rotations[i].x,
                                                             scene.rotations[i].y, scene.rotations[i].z));
                const glm::vec3 s = glm::vec3(scene.scales[i]);
                const int axis = (s.x <= s.y && s.x <= s.z) ? 0 : (s.y <= s.z ? 1 : 2);
                glm::vec3 n = glm::normalize(R[axis]); // glm: R[axis] is column `axis`
                if (glm::dot(n, cameraPosition() - glm::vec3(scene.means[i])) < 0.0f)
                {
                    n = -n;
                }
                derived[i] = n * 0.5f + 0.5f;
            }
            const std::vector<float> ref = splatValues(r, scene, derived);
            double worst = 0.0;
            for (int c = 0; c < 3; ++c)
            {
                worst = std::max(worst, worstDiff(ref, [&](uint32_t p) { return 0.5 * d.normalDepth[p][c]
                                                                              + 0.5 * d.metalAmbientCoverage[p].z; }, c));
            }
            check(worst <= 2e-5, "default normal = shortest ellipsoid axis facing the camera " + fmt("(worst %.3g)", worst));
        }

        // ---- 4: relightable .ply ----
        {
            const std::string dir = (std::filesystem::temp_directory_path() / "sabercat_relight_test").string();
            std::filesystem::create_directories(dir);
            const std::string relightable = dir + "/relightable.ply", plain = dir + "/plain.ply";
            writeRelightablePly(relightable, scene, true);
            writeRelightablePly(plain, scene, false);

            Scene noMetal = scene; // the relightable .ply has no metallic attribute
            std::fill(noMetal.metallic.begin(), noMetal.metallic.end(), 0.0f);
            GaussianHostInput in = input(noMetal, true);
            in.metallic = nullptr;
            r.setGaussians(in);
            r.forward(RenderMode::Differentiable);
            const Renderer::GBufferData want = r.readGBuffer();
            r.loadScene(relightable.c_str());
            r.forward(RenderMode::Differentiable);
            const Renderer::GBufferData got = r.readGBuffer();
            double worst = 0.0;
            for (uint32_t p = 0; p < kWidth * kHeight; ++p)
            {
                for (int c = 0; c < 4; ++c)
                {
                    worst = std::max({worst, std::fabs(double(got.normalDepth[p][c]) - want.normalDepth[p][c]),
                                      std::fabs(double(got.baseRoughness[p][c]) - want.baseRoughness[p][c]),
                                      std::fabs(double(got.metalAmbientCoverage[p][c]) - want.metalAmbientCoverage[p][c])});
                }
            }
            check(worst <= 2e-5, "relightable .ply (activations applied) == the activated streams " + fmt("(worst %.3g)", worst));

            r.loadScene(plain.c_str());
            r.forward(RenderMode::Differentiable);
            const Renderer::GBufferData v = r.readGBuffer();
            double worstV = 0.0;
            for (uint32_t p = 0; p < kWidth * kHeight; ++p)
            {
                for (int c = 0; c < 4; ++c)
                {
                    worstV = std::max(worstV, std::fabs(double(v.normalDepth[p][c]) - d.normalDepth[p][c]));
                }
            }
            check(worstV <= 2e-5, "plain .ply's zero normals are ignored (derived normals used) " + fmt("(worst %.3g)", worstV));
            std::filesystem::remove_all(dir);
        }

        // ---- 5: lighting vs a CPU port ----
        {
            std::vector<Light> lights(3);
            lights[0].type = LightType::Directional;
            lights[0].direction = glm::normalize(glm::vec3(0.3f, -0.5f, -1.0f));
            lights[0].color = glm::vec3(1.0f, 0.9f, 0.8f);
            lights[0].intensity = 2.0f;
            lights[1].type = LightType::Point;
            lights[1].position = glm::vec3(-0.5f, 0.6f, 1.5f);
            lights[1].color = glm::vec3(0.4f, 0.6f, 1.0f);
            lights[1].intensity = 3.0f;
            lights[2].type = LightType::Rect;
            lights[2].position = glm::vec3(0.6f, 0.2f, 1.2f);
            lights[2].axisX = glm::vec3(1.0f, 0.0f, 0.0f);
            lights[2].axisY = glm::vec3(0.0f, -1.0f, 0.0f); // emits along cross(x, -y) = -z: at the scene
            lights[2].halfWidth = 0.4f;
            lights[2].halfHeight = 0.25f;
            lights[2].intensity = 4.0f;
            const EnvironmentLight env = EnvironmentLight::sky(glm::vec3(0.4f, 0.5f, 0.9f), glm::vec3(0.8f),
                                                               glm::vec3(0.2f, 0.15f, 0.1f), glm::vec3(0, 1, 0));
            r.setLights(lights);
            r.setEnvironment(env);
            LightingSettings st;
            st.background = glm::vec3(0.05f, 0.1f, 0.2f);
            st.exposure = 1.3f;
            st.rectSamples = 16;
            const char * names[] = {"Lit", "Normal", "BaseColor", "Roughness", "Metallic",
                                    "AmbientOcclusion", "Visibility", "Depth"};
            // Two material sets. GGX's lobe scales with 1/alpha^2 (alpha =
            // roughness^2), so near-mirror specular amplifies fp32 rounding
            // differences between GPU and CPU op order: every output mode is
            // checked tightly on roughness >= 0.3, and the glossy set
            // (roughness down to 0.1) is checked on Lit with a looser bound.
            Scene rough = scene;
            for (glm::vec4 & b : rough.baseColorRoughness)
            {
                b.w = std::max(b.w, 0.3f);
            }
            for (uint32_t run = 0; run < 9; ++run)
            {
                const bool glossy = run == 8;
                const uint32_t mode = glossy ? 0u : run;
                r.setGaussians(input(glossy ? scene : rough, true));
                st.output = static_cast<LightingOutput>(mode);
                r.setLightingSettings(st);
                r.forward(RenderMode::Viewer);
                const std::vector<float> lit = r.readLitImage();
                const Renderer::GBufferData gb = r.readGBuffer();
                const std::vector<float> vis = r.readVisibility(); // ShadowMode::None: all ones
                double worst = 0.0, scale = 0.0;
                uint32_t worstPixel = 0;
                for (uint32_t p = 0; p < kWidth * kHeight; ++p)
                {
                    const glm::vec4 want = cpuLighting(gb, p, makeCamera(), lights, env, st, vis);
                    for (int c = 0; c < 4; ++c)
                    {
                        const double e = std::fabs(double(lit[p * 4 + c]) - want[c]);
                        if (e > worst)
                        {
                            worst = e;
                            worstPixel = p;
                        }
                        scale = std::max(scale, std::fabs(double(want[c])));
                    }
                }
                const float cov = gb.metalAmbientCoverage[worstPixel].z;
                const double tolerance = (glossy ? 2e-3 : 1e-4) * std::max(scale, 1.0);
                check(worst <= tolerance,
                      std::string("LightingOutput::") + names[mode]
                          + (glossy ? " (glossy, roughness >= 0.1)" : " (roughness >= 0.3)")
                          + " == CPU port " + fmt("(worst %.3g", worst) + fmt(", max %.3g", scale)
                          + fmt(", worst pixel roughness %.2f)", gb.baseRoughness[worstPixel].w / cov));
            }
            r.setGaussians(input(scene, true));
            // Lights matter: lit differs from no lights.
            st.output = LightingOutput::Lit;
            r.setLightingSettings(st);
            r.forward(RenderMode::Viewer);
            const std::vector<float> withLights = r.readLitImage();
            r.setLights({});
            r.setEnvironment(EnvironmentLight{});
            r.forward(RenderMode::Viewer);
            const std::vector<float> dark = r.readLitImage();
            double darkMax = 0.0, litMax = 0.0;
            for (uint32_t p = 0; p < kWidth * kHeight; ++p)
            {
                const float cov = dark[p * 4 + 3];
                for (int c = 0; c < 3; ++c)
                {
                    darkMax = std::max(darkMax, double(std::fabs(dark[p * 4 + c] - st.background[c] * (1.0f - cov) * st.exposure)));
                    litMax = std::max(litMax, double(withLights[p * 4 + c]));
                }
            }
            check(darkMax <= 1e-6 && litMax > 0.1, "no lights => background only; lights add light");
        }

        // ---- 6: per-light visibility composition, and the Visibility / Depth outputs ----
        {
            // BakedSH gives each light its own visibility layer from the
            // scene's visibility stream, so a light-to-layer mix-up changes the
            // lit image. The CPU port multiplies light i by layer i.
            std::vector<Light> casting(2);
            casting[0].type = LightType::Point;
            casting[0].position = glm::vec3(-0.6f, 0.4f, 1.6f);
            casting[0].color = glm::vec3(1.0f, 0.8f, 0.6f);
            casting[0].intensity = 4.0f;
            casting[1].type = LightType::Directional;
            casting[1].direction = glm::normalize(glm::vec3(0.5f, -0.2f, -1.0f));
            casting[1].color = glm::vec3(0.7f, 0.8f, 1.0f);
            casting[1].intensity = 2.5f;
            const EnvironmentLight env = EnvironmentLight::sky(glm::vec3(0.4f, 0.5f, 0.9f), glm::vec3(0.8f),
                                                               glm::vec3(0.2f, 0.15f, 0.1f), glm::vec3(0, 1, 0));
            r.setLights(casting);
            r.setEnvironment(env);
            ShadowSettings baked;
            baked.mode = ShadowMode::BakedSH;
            r.setShadowSettings(baked);
            Scene rough = scene;
            for (glm::vec4 & b : rough.baseColorRoughness)
            {
                b.w = std::max(b.w, 0.3f);
            }
            r.setGaussians(input(rough, true));

            // Each run: the output, the layer it shows, and the depth scale.
            const auto compare = [&](LightingOutput mode, uint32_t visLight, float depthScale, const std::string & name)
            {
                LightingSettings st;
                st.background = glm::vec3(0.05f, 0.1f, 0.2f);
                st.exposure = 1.3f;
                st.rectSamples = 16;
                st.output = mode;
                st.visibilityLight = visLight;
                st.depthScale = depthScale;
                r.setLightingSettings(st);
                r.forward(RenderMode::Viewer);
                const std::vector<float> lit = r.readLitImage();
                const std::vector<float> vis = r.readVisibility();
                const Renderer::GBufferData gb = r.readGBuffer();
                double worst = 0.0, scale = 0.0, layerGap = 0.0;
                for (uint32_t p = 0; p < kWidth * kHeight; ++p)
                {
                    layerGap = std::max(layerGap, static_cast<double>(std::fabs(vis[p] - vis[kWidth * kHeight + p])));
                    const glm::vec4 want = cpuLighting(gb, p, makeCamera(), casting, env, st, vis);
                    for (int c = 0; c < 4; ++c)
                    {
                        worst = std::max(worst, std::fabs(double(lit[p * 4 + c]) - want[c]));
                        scale = std::max(scale, std::fabs(double(want[c])));
                    }
                }
                const double tolerance = 1e-4 * std::max(scale, 1.0);
                check(worst <= tolerance, name + " == CPU port " + fmt("(worst %.3g", worst)
                                              + fmt(", max %.3g)", scale));
                return layerGap;
            };
            const double gap = compare(LightingOutput::Lit, 0, 0.1f, "Lit with 2 shadow-casting lights (BakedSH layers)");
            check(gap > 0.02, "the two visibility layers differ, so a swapped layer would show " + fmt("(max gap %.3f)", gap));
            compare(LightingOutput::Visibility, 1, 0.1f, "Visibility output, layer 1");
            compare(LightingOutput::Visibility, 0, 0.1f, "Visibility output, layer 0");
            compare(LightingOutput::Depth, 0, 0.37f, "Depth output with depthScale 0.37");
        }

        // ---- 7: odd sizes -- G-buffer and lighting against the CPU port ----
        {
            constexpr uint32_t kOddW = 37, kOddH = 29;
            Renderer odd(kOddW, kOddH);
            if (!odd.initialize({}, true, testShaderDir()))
            {
                check(false, "odd-size renderer initialize");
            }
            else
            {
                Camera cam = makeCamera();
                cam.width = kOddW;
                cam.height = kOddH;
                cam.fx = cam.fy = makeCamera().fx * kOddW / kWidth; // the same field of view
                cam.cx = 0.5f * (kOddW - 1);
                cam.cy = 0.5f * (kOddH - 1);
                odd.setCamera(cam);
                odd.setLightingEnabled(true);
                Scene rough = scene;
                for (glm::vec4 & b : rough.baseColorRoughness)
                {
                    b.w = std::max(b.w, 0.3f);
                }
                odd.setGaussians(input(rough, true));
                std::vector<Light> lights(3);
                lights[0].type = LightType::Directional;
                lights[0].direction = glm::normalize(glm::vec3(0.3f, -0.5f, -1.0f));
                lights[0].color = glm::vec3(1.0f, 0.9f, 0.8f);
                lights[0].intensity = 2.0f;
                lights[1].type = LightType::Point;
                lights[1].position = glm::vec3(-0.5f, 0.6f, 1.5f);
                lights[1].color = glm::vec3(0.4f, 0.6f, 1.0f);
                lights[1].intensity = 3.0f;
                lights[2].type = LightType::Rect;
                lights[2].position = glm::vec3(0.6f, 0.2f, 1.2f);
                lights[2].axisX = glm::vec3(1.0f, 0.0f, 0.0f);
                lights[2].axisY = glm::vec3(0.0f, -1.0f, 0.0f);
                lights[2].halfWidth = 0.4f;
                lights[2].halfHeight = 0.25f;
                lights[2].intensity = 4.0f;
                const EnvironmentLight env = EnvironmentLight::sky(glm::vec3(0.4f, 0.5f, 0.9f), glm::vec3(0.8f),
                                                                   glm::vec3(0.2f, 0.15f, 0.1f), glm::vec3(0, 1, 0));
                odd.setLights(lights);
                odd.setEnvironment(env);
                LightingSettings st;
                st.background = glm::vec3(0.05f, 0.1f, 0.2f);
                st.exposure = 1.3f;
                st.rectSamples = 16;
                st.output = LightingOutput::Lit;
                odd.setLightingSettings(st);
                odd.forward(RenderMode::Viewer);
                const std::vector<float> lit = odd.readLitImage();
                const std::vector<float> vis = odd.readVisibility();
                const Renderer::GBufferData gb = odd.readGBuffer();
                double worst = 0.0, scale = 0.0;
                uint32_t covered = 0;
                for (uint32_t p = 0; p < kOddW * kOddH; ++p)
                {
                    covered += gb.metalAmbientCoverage[p].z > 0.01f ? 1u : 0u;
                    const glm::vec4 want = cpuLighting(gb, p, cam, lights, env, st, vis);
                    for (int c = 0; c < 4; ++c)
                    {
                        worst = std::max(worst, std::fabs(double(lit[p * 4 + c]) - want[c]));
                        scale = std::max(scale, std::fabs(double(want[c])));
                    }
                }
                const double tolerance = 1e-4 * std::max(scale, 1.0);
                check(covered > kOddW * kOddH / 8 && worst <= tolerance,
                      "37x29 (odd): G-buffer and Lit == CPU port " + fmt("(worst %.3g", worst) +
                          fmt(", max %.3g", scale) + fmt(", %.0f covered px)", static_cast<double>(covered)));
                odd.shutdown();
            }
        }

        // ---- 8: relight_prepare's normals (flip, tie-break, rotation), one splat at a time ----
        {
            r.setLights({}); // section 6 left BakedSH on: no lights, no shadows
            r.setShadowSettings(ShadowSettings{});
            const Camera front = makeCamera();
            Camera back = makeCamera(); // on the far side of the splat, looking back at it
            back.cameraToWorld = glm::translate(glm::mat4(1.0f), glm::vec3(0.05f, -0.02f, -3.0f)) *
                                 glm::rotate(glm::mat4(1.0f), kPi, glm::vec3(0.0f, 1.0f, 0.0f));
            const auto check8 = [&](const std::string & name, const OneSplat & g, const Camera & cam)
            {
                const glm::vec3 expected = cpuPreparedNormal(g, glm::vec3(cam.cameraToWorld[3]));
                int covered = 0;
                const double worst = oneSplatNormalError(r, cam, g, expected, covered);
                check(covered > 20 && worst <= 2e-5, name + " " + fmt("(worst %.3g, bound 2e-5", worst) +
                                                         fmt(", %.0f px)", static_cast<double>(covered)));
            };
            OneSplat g;
            g.hasNormal = true;
            g.normal = glm::vec4(0.6f, 0.0f, -0.8f, 0.0f);
            check8("back-facing given normal is flipped to the camera", g, front);
            g.normal = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);
            check8("front-facing given normal is kept", g, front);
            check8("camera on the far side: the normal flips to face it", g, back);

            OneSplat d; // no normal stream: the shortest ellipsoid axis
            d.scale = glm::vec4(0.1f, 0.1f, 0.1f, 0.0f);
            check8("isotropic scale: tie takes x", d, front);
            d.scale = glm::vec4(0.2f, 0.05f, 0.05f, 0.0f);
            check8("shortest y = z (x longer): tie takes y, flipped", d, front);
            d.scale = glm::vec4(0.05f, 0.2f, 0.05f, 0.0f);
            check8("shortest x = z: tie takes x", d, front);
            d.scale = glm::vec4(0.05f, 0.05f, 0.2f, 0.0f);
            check8("shortest x = y: tie takes x", d, front);
            d.scale = glm::vec4(0.05f, 0.2f, 0.2f, 0.0f);
            d.rotation = glm::vec4(0.0f, std::sqrt(0.5f), 0.0f, std::sqrt(0.5f)); // 90 deg about y
            check8("rotated: the shortest axis follows the rotation", d, front);
            d.rotation = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
            d.scale = glm::vec4(0.05f, 0.2f, 0.2f, 0.0f);
            check8("far-side camera with no normal stream", d, back);
            r.setCamera(front);
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
