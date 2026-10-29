// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Finite-difference gradient check of the backward pass.
//
// A tiny scene (8 Gaussians, 60x44 image, rotated + translated camera) is
// rendered by ForwardPass in Differentiable mode and differentiated by
// BackwardPass against a fixed pseudo-random image gradient w, i.e. the loss
// L = sum(w * image) over all four RGBA channels. Every GPU gradient (every
// scalar of every Gaussian: position 3, scale 3, rotation 4, opacity 1, SH 48)
// is compared against a central finite difference of an INDEPENDENT
// double-precision CPU reimplementation of the forward pass.
//
// FROZEN-BRANCH finite differences. The forward pass is only piecewise
// smooth: a (pixel, Gaussian) pair is skipped below alpha 1/255, alpha clamps
// at 0.99, the Jacobian clamps at 1.3x the FOV, color clamps to [0,1], and a
// pixel stops at T < 1e-4. A naive finite difference straddling one of those
// kinks measures the jump, not the derivative: one pixel crossing the 1/255
// contour moves L by ~w*color/255 ~ 4e-3, which divided by 2h = 2e-6 is an
// "error" of ~2000. The analytic backward is the derivative of the smooth
// piece the base point sits on, so the reference freezes every discrete
// decision at the base point -- which pairs blend, in what order, which are
// clamped, where each pixel stops, which side of each clamp each Gaussian is
// on -- and differentiates the resulting smooth function. Those decisions
// are reconstructed from the GPU's own projected data in float (so they are
// the decisions the GPU actually made), and checked against the per-pixel
// state the GPU saved: nContrib must match exactly, which also verifies the
// forward-state storage backward depends on.
//
// Runs: compensation off and on (the per-scene ANTIALIASED flag), and an SH
// degree cap of 1 (bands 2-3 must come out exactly zero). screenMeans is
// checked against dL/d(center) of the double model. A 202-Gaussian random
// scene (a Gaussian just beyond kMinDepth, a camera inside another's 3-sigma)
// is checked on a sample of 300 scalars at SH degree 3 and on a sample at
// SH degree 2. Plus structural checks: viewer LOD knobs cannot reach a
// Differentiable run, backward refuses a Viewer-mode or incomplete forward,
// rejects mismatched or unbound gradient views and a short image gradient
// without touching the gradients, and a second call overwrites them.
//
// Depth gradient: the same frozen-branch machinery differentiates the depth
// output D = sum_j T_j alpha_j z_j (z_j the camera depth), with the loss
// L = sum w*RGBA + sum_p g_p D_p. Depth-only runs (dL/dImage = 0) isolate the
// z path: SH gradients must be exactly zero, the rest non-zero, and dL/dz
// flows through alpha-clamped pairs. Mixed runs add the color term. The GPU
// reads only component 0 of the dL/dDepth buffer; the others hold garbage.
// Checks that the no-depth paths are unchanged (zero and null depth gradients,
// the view overloads, layout-independence of the color gradient), that
// depth-gradient garbage in components 1..n-1 is ignored, and that the depth
// overloads reject a layout-None forward and a short depth buffer.

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sabercat/backward_pass.h"
#include "sabercat/buffer.h"
#include "sabercat/camera.h"
#include "sabercat/device.h"
#include "sabercat/forward_pass.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/render_config.h"

using namespace sabercat;

namespace
{

constexpr uint32_t kWidth = 60; // not a multiple of 16: partial tiles
constexpr uint32_t kHeight = 44;
constexpr float kFx = 60.0f;
constexpr float kFy = 58.0f;
constexpr float kCx = 30.0f;
constexpr float kCy = 22.0f;
constexpr int kCoeffs = static_cast<int>(kShCoefficients);

struct Params
{
    glm::dvec3 position;
    glm::dvec3 scale;
    glm::dvec4 rotation; // xyzw, as stored (normalized at the base point)
    double opacity;
    glm::dvec3 sh[kCoeffs];
};

// Deterministic LCG in [-1, 1] -- fixtures must not change run to run.
struct Lcg
{
    uint32_t state;
    double next()
    {
        state = state * 1664525u + 1013904223u;
        return static_cast<double>(state >> 8) / 8388608.0 - 1.0;
    }
};

struct Context
{
    glm::mat4 worldToCamera;   // exactly what the GPU receives
    glm::vec3 cameraPosition;
    uint32_t renderFlags = 0;
    uint32_t shDegree = kShDegree;
};

// -- Frozen discrete decisions -------------------------------------------
struct Contribution
{
    uint32_t gaussian;
    bool alphaClamped;
};

struct Frozen
{
    std::vector<bool> active;                 // forward did not cull it
    std::vector<int> clampSignX, clampSignY;  // frustum clamp side
    std::vector<std::array<int, 3>> colorClamp; // -1 low, 0 free, +1 high
    std::vector<std::vector<Contribution>> pixels; // blend sequence per pixel
};

// -- Independent double-precision forward ---------------------------------
struct Projected
{
    bool valid = false;
    glm::dvec2 center{0.0};
    double conic[3] = {0, 0, 0};
    double alpha = 0.0; // projected opacity (incl. compensation)
    double depth = 0.0; // camera depth z = -local.z, the integrand of D
    glm::dvec3 color{0.0};
    int clampSignX = 0, clampSignY = 0;   // only meaningful from base
    std::array<int, 3> colorClamp{0, 0, 0};
};

void rotationMatrix(const glm::dvec4 & q, double R[3][3])
{
    double x = q.x, y = q.y, z = q.z, w = q.w;
    R[0][0] = 1 - 2 * (y * y + z * z);
    R[0][1] = 2 * (x * y - z * w);
    R[0][2] = 2 * (x * z + y * w);
    R[1][0] = 2 * (x * y + z * w);
    R[1][1] = 1 - 2 * (x * x + z * z);
    R[1][2] = 2 * (y * z - x * w);
    R[2][0] = 2 * (x * z - y * w);
    R[2][1] = 2 * (y * z + x * w);
    R[2][2] = 1 - 2 * (x * x + y * y);
}

glm::dvec3 shRaw(uint32_t degree, const glm::dvec3 * c, const glm::dvec3 & d)
{
    glm::dvec3 r = static_cast<double>(kShC0) * c[0];
    if (degree >= 1)
    {
        double x = d.x, y = d.y, z = d.z;
        r += static_cast<double>(kShC1) * (-y * c[1] + z * c[2] - x * c[3]);
        if (degree >= 2)
        {
            double xx = x * x, yy = y * y, zz = z * z, xy = x * y, yz = y * z, xz = x * z;
            r += static_cast<double>(kShC2_0) * xy * c[4] + static_cast<double>(kShC2_1) * yz * c[5]
                 + static_cast<double>(kShC2_2) * (2 * zz - xx - yy) * c[6]
                 + static_cast<double>(kShC2_3) * xz * c[7]
                 + static_cast<double>(kShC2_4) * (xx - yy) * c[8];
            if (degree >= 3)
            {
                r += static_cast<double>(kShC3_0) * y * (3 * xx - yy) * c[9]
                     + static_cast<double>(kShC3_1) * xy * z * c[10]
                     + static_cast<double>(kShC3_2) * y * (4 * zz - xx - yy) * c[11]
                     + static_cast<double>(kShC3_3) * z * (2 * zz - 3 * xx - 3 * yy) * c[12]
                     + static_cast<double>(kShC3_4) * x * (4 * zz - xx - yy) * c[13]
                     + static_cast<double>(kShC3_5) * z * (xx - yy) * c[14]
                     + static_cast<double>(kShC3_6) * x * (xx - 3 * yy) * c[15];
            }
        }
    }
    return r;
}

// `frozen` == nullptr: base evaluation, which RECORDS the clamp sides.
Projected projectDouble(const Params & p, uint32_t index, const Context & ctx,
                        const Frozen * frozen)
{
    Projected out;
    // glm is column-major: M[col][row]. local = W p + t.
    const glm::mat4 & M = ctx.worldToCamera;
    double W[3][3];
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            W[r][c] = M[c][r];
        }
    }
    glm::dvec3 local;
    for (int r = 0; r < 3; ++r)
    {
        local[r] = W[r][0] * p.position.x + W[r][1] * p.position.y + W[r][2] * p.position.z
                   + static_cast<double>(M[3][r]);
    }
    double d = -local.z;
    if (d <= static_cast<double>(kMinDepth))
    {
        return out;
    }
    const double fx = kFx, fy = kFy;
    out.depth = d;
    out.center = glm::dvec2(fx * local.x / d + kCx, kCy - fy * local.y / d);

    double R[3][3];
    rotationMatrix(p.rotation, R);
    double S2[3] = {p.scale.x * p.scale.x, p.scale.y * p.scale.y, p.scale.z * p.scale.z};
    double Sigma[3][3];
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            Sigma[i][j] = R[i][0] * S2[0] * R[j][0] + R[i][1] * S2[1] * R[j][1]
                          + R[i][2] * S2[2] * R[j][2];
        }
    }
    double WS[3][3], Sc[3][3];
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            WS[i][j] = W[i][0] * Sigma[0][j] + W[i][1] * Sigma[1][j] + W[i][2] * Sigma[2][j];
        }
    }
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            Sc[i][j] = WS[i][0] * W[j][0] + WS[i][1] * W[j][1] + WS[i][2] * W[j][2];
        }
    }

    const double limX = static_cast<double>(kFrustumClamp) * (0.5 * kWidth) / fx;
    const double limY = static_cast<double>(kFrustumClamp) * (0.5 * kHeight) / fy;
    int sx, sy;
    if (frozen)
    {
        sx = frozen->clampSignX[index];
        sy = frozen->clampSignY[index];
    }
    else
    {
        double tx = local.x / d, ty = local.y / d;
        sx = tx < -limX ? -1 : (tx > limX ? 1 : 0);
        sy = ty < -limY ? -1 : (ty > limY ? 1 : 0);
    }
    out.clampSignX = sx;
    out.clampSignY = sy;
    double jx = sx == 0 ? local.x : sx * limX * d;
    double jy = sy == 0 ? local.y : sy * limY * d;
    double J[2][3] = {{fx / d, 0, fx * jx / (d * d)}, {0, -fy / d, -fy * jy / (d * d)}};

    double cov[2][2];
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            double s = 0;
            for (int k = 0; k < 3; ++k)
            {
                for (int l = 0; l < 3; ++l)
                {
                    s += J[i][k] * Sc[k][l] * J[j][l];
                }
            }
            cov[i][j] = s;
        }
    }
    const double dil = static_cast<double>(kCovarianceDilation);
    double det0 = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
    double A = cov[0][0] + dil, B = cov[0][1], C = cov[1][1] + dil;
    double det = A * C - cov[0][1] * cov[1][0];
    if (det <= 0)
    {
        return out;
    }
    out.conic[0] = C / det;
    out.conic[1] = -B / det;
    out.conic[2] = A / det;
    double comp = (ctx.renderFlags & kRenderFlagAntialiased) ? std::sqrt(std::max(det0, 0.0) / det)
                                                              : 1.0;
    out.alpha = p.opacity * comp;

    glm::dvec3 dir =
        glm::normalize(p.position - glm::dvec3(ctx.cameraPosition));
    glm::dvec3 raw = shRaw(ctx.shDegree, p.sh, dir) + 0.5;
    for (int ch = 0; ch < 3; ++ch)
    {
        int state;
        if (frozen)
        {
            state = frozen->colorClamp[index][ch];
        }
        else
        {
            state = raw[ch] <= 0.0 ? -1 : (raw[ch] >= 1.0 ? 1 : 0);
        }
        out.colorClamp[ch] = state;
        out.color[ch] = state < 0 ? 0.0 : (state > 0 ? 1.0 : raw[ch]);
    }
    out.valid = true;
    return out;
}

// Renders RGBA with frozen blend sequences. `centerShift` (optional) offsets
// each projected center in pixels, for differentiating dL/d(center). `depthOut`
// (optional) receives the depth output D = sum_j T_j alpha_j z_j per pixel.
std::vector<glm::dvec4> renderDouble(const std::vector<Params> & params, const Context & ctx,
                                     const Frozen & frozen,
                                     const std::vector<glm::dvec2> * centerShift = nullptr,
                                     std::vector<double> * depthOut = nullptr)
{
    std::vector<Projected> proj(params.size());
    for (uint32_t i = 0; i < params.size(); ++i)
    {
        if (frozen.active[i])
        {
            proj[i] = projectDouble(params[i], i, ctx, &frozen);
            if (centerShift)
            {
                proj[i].center += (*centerShift)[i];
            }
        }
    }
    std::vector<glm::dvec4> image(static_cast<size_t>(kWidth) * kHeight);
    if (depthOut)
    {
        depthOut->assign(image.size(), 0.0);
    }
    for (uint32_t py = 0; py < kHeight; ++py)
    {
        for (uint32_t px = 0; px < kWidth; ++px)
        {
            size_t pi = static_cast<size_t>(py) * kWidth + px;
            glm::dvec3 color(0.0);
            double T = 1.0;
            double depthSum = 0.0;
            for (const Contribution & c : frozen.pixels[pi])
            {
                const Projected & g = proj[c.gaussian];
                double alpha;
                if (c.alphaClamped)
                {
                    alpha = static_cast<double>(kMaxAlpha);
                }
                else
                {
                    double dx = px - g.center.x, dy = py - g.center.y;
                    double q = g.conic[0] * dx * dx + 2 * g.conic[1] * dx * dy
                               + g.conic[2] * dy * dy;
                    alpha = g.alpha * std::exp(-0.5 * q);
                }
                color += T * alpha * g.color;
                depthSum += T * alpha * g.depth;
                T *= 1.0 - alpha;
            }
            image[pi] = glm::dvec4(color, 1.0 - T);
            if (depthOut)
            {
                (*depthOut)[pi] = depthSum;
            }
        }
    }
    return image;
}

// The scalar loss L = sum_i rgba[i] * RGBA_i + sum_p depth[p] * D_p. Either
// part may be empty. A Loss built from a plain weight vector is color-only.
// `depthGarbage` only affects the GPU side: the dL/dDepth buffer's components
// 1..n-1 are filled with garbage of that scale (0 = zeros), which must not
// change any gradient.
struct Loss
{
    Loss(std::vector<float> color = {}, std::vector<float> depthWeights = {},
         float garbage = 0.0f)
        : rgba(std::move(color)), depth(std::move(depthWeights)), depthGarbage(garbage)
    {
    }

    std::vector<float> rgba;
    std::vector<float> depth;
    float depthGarbage;
};

double lossDouble(const std::vector<Params> & params, const Context & ctx, const Frozen & frozen,
                  const Loss & loss, const std::vector<glm::dvec2> * centerShift = nullptr)
{
    std::vector<double> depth;
    std::vector<glm::dvec4> image = renderDouble(params, ctx, frozen, centerShift, &depth);
    double total = 0.0;
    for (size_t i = 0; i < image.size(); ++i)
    {
        for (int c = 0; c < 4; ++c)
        {
            total += static_cast<double>(loss.rgba[i * 4 + c]) * image[i][c];
        }
    }
    for (size_t p = 0; p < loss.depth.size(); ++p)
    {
        total += static_cast<double>(loss.depth[p]) * depth[p];
    }
    return total;
}

bool tileRange(const glm::vec2 & center, float radius, uint32_t tilesX, uint32_t tilesY, int & x0,
               int & y0, int & x1, int & y1)
{
    x0 = std::max(static_cast<int>(std::floor((center.x - radius) / kTileSize)), 0);
    y0 = std::max(static_cast<int>(std::floor((center.y - radius) / kTileSize)), 0);
    x1 = std::min(static_cast<int>(std::floor((center.x + radius) / kTileSize)),
                  static_cast<int>(tilesX) - 1);
    y1 = std::min(static_cast<int>(std::floor((center.y + radius) / kTileSize)),
                  static_cast<int>(tilesY) - 1);
    return x0 <= x1 && y0 <= y1;
}

struct FreezeStats
{
    uint32_t clampedPairs = 0;
    uint32_t earlyOutPixels = 0; // stopped with tile-list entries left that would blend
};

// Reconstructs the GPU's discrete decisions from ITS projected data, in
// float, mirroring tile_rasterize -- then cross-checked against pixelState.
Frozen freeze(const std::vector<Params> & params, const Context & ctx,
              const std::vector<ProjectedGaussian> & gpu, const std::vector<uint32_t> & gpuState,
              FreezeStats & stats)
{
    const uint32_t n = static_cast<uint32_t>(params.size());
    const uint32_t tilesX = dispatchGroups(kWidth, kTileSize);
    const uint32_t tilesY = dispatchGroups(kHeight, kTileSize);

    Frozen f;
    f.active.resize(n);
    f.clampSignX.resize(n);
    f.clampSignY.resize(n);
    f.colorClamp.resize(n);
    for (uint32_t i = 0; i < n; ++i)
    {
        f.active[i] = gpu[i].tileCount != 0;
        Projected base = projectDouble(params[i], i, ctx, nullptr);
        f.clampSignX[i] = base.clampSignX;
        f.clampSignY[i] = base.clampSignY;
        f.colorClamp[i] = base.colorClamp;
    }

    // Depth-sorted order (radix sort is stable; ties fall back to index).
    std::vector<uint32_t> order;
    for (uint32_t i = 0; i < n; ++i)
    {
        if (f.active[i])
        {
            order.push_back(i);
        }
    }
    std::stable_sort(order.begin(), order.end(),
                     [&](uint32_t a, uint32_t b) { return gpu[a].depth < gpu[b].depth; });

    f.pixels.resize(static_cast<size_t>(kWidth) * kHeight);
    for (uint32_t py = 0; py < kHeight; ++py)
    {
        for (uint32_t px = 0; px < kWidth; ++px)
        {
            int tx = static_cast<int>(px / kTileSize), ty = static_cast<int>(py / kTileSize);
            std::vector<uint32_t> list;
            for (uint32_t g : order)
            {
                int x0, y0, x1, y1;
                if (tileRange(gpu[g].center, gpu[g].radius, tilesX, tilesY, x0, y0, x1, y1)
                    && tx >= x0 && tx <= x1 && ty >= y0 && ty <= y1)
                {
                    list.push_back(g);
                }
            }
            size_t pi = static_cast<size_t>(py) * kWidth + px;
            float T = 1.0f;
            uint32_t contributorCount = 0;
            bool stopped = false;
            for (uint32_t k = 0; k < list.size(); ++k)
            {
                const ProjectedGaussian & g = gpu[list[k]];
                float dx = static_cast<float>(px) - g.center.x;
                float dy = static_cast<float>(py) - g.center.y;
                float e = g.conic.x * dx * dx + 2.0f * g.conic.y * dx * dy + g.conic.z * dy * dy;
                if (stopped)
                {
                    // Would a later entry have blended? Then early-out cut it.
                    if (e >= 0.0f
                        && std::min(g.alpha * std::exp(-0.5f * e), kMaxAlpha) >= 1.0f / 255.0f)
                    {
                        ++stats.earlyOutPixels;
                        break;
                    }
                    continue;
                }
                if (e < 0.0f)
                {
                    continue;
                }
                float raw = g.alpha * std::exp(-0.5f * e);
                float alpha = std::min(raw, kMaxAlpha);
                if (alpha < 1.0f / 255.0f)
                {
                    continue;
                }
                bool clamped = raw > kMaxAlpha;
                stats.clampedPairs += clamped ? 1u : 0u;
                f.pixels[pi].push_back({list[k], clamped});
                contributorCount = k + 1;
                T *= 1.0f - alpha;
                if (T < 1e-4f)
                {
                    stopped = true;
                }
            }
            if (gpuState[pi * 2 + 1] != contributorCount)
            {
                throw std::runtime_error("frozen decisions disagree with the GPU's saved nContrib at pixel ("
                                         + std::to_string(px) + "," + std::to_string(py) + "): gpu "
                                         + std::to_string(gpuState[pi * 2 + 1]) + " cpu "
                                         + std::to_string(contributorCount));
            }
        }
    }
    return f;
}

// -- GPU side ------------------------------------------------------------
struct GpuResult
{
    std::vector<glm::vec4> image;
    std::vector<uint32_t> pixelState;
    std::vector<ProjectedGaussian> projected;
    std::vector<glm::vec4> gMeans, gScales, gRotations, gSh, gScreen;
    std::vector<float> gOpacity;
    std::vector<float> depth;      // forward D per pixel (depth layouts only)
    std::vector<float> projGrad;   // backward_rasterize's per-Gaussian slots
};

void uploadParams(GaussianCloud & cloud, const std::vector<Params> & params)
{
    cloud.initialize(params.size());
    for (size_t i = 0; i < params.size(); ++i)
    {
        const Params & p = params[i];
        cloud.positions()[i] = glm::vec4(glm::vec3(p.position), 0.0f);
        cloud.scales()[i] = glm::vec4(glm::vec3(p.scale), 0.0f);
        cloud.rotations()[i] = glm::vec4(p.rotation);
        cloud.opacities()[i] = static_cast<float>(p.opacity);
        for (int k = 0; k < kCoeffs; ++k)
        {
            cloud.shCoefficients()[i * kCoeffs + k] = glm::vec4(glm::vec3(p.sh[k]), 0.0f);
        }
    }
    cloud.uploadToGPU();
    cloud.ensureGradientBuffers();
}

CameraUniforms makeUniforms(const Context & ctx, uint32_t count)
{
    CameraUniforms u{};
    u.worldToCamera = ctx.worldToCamera;
    u.fx = kFx;
    u.fy = kFy;
    u.cx = kCx;
    u.cy = kCy;
    u.imageWidth = kWidth;
    u.imageHeight = kHeight;
    u.tilesX = dispatchGroups(kWidth, kTileSize);
    u.tilesY = dispatchGroups(kHeight, kTileSize);
    u.gaussianCount = count;
    u.shDegree = ctx.shDegree;
    u.renderFlags = ctx.renderFlags;
    u.cameraPositionWorld = glm::vec4(ctx.cameraPosition, 0.0f);
    return u;
}

// The dL/dDepth buffer the host hands the GPU: width*height*n floats, with
// component 0 of each pixel = weights[p] and components 1..n-1 garbage (scale *
// uniform [-1, 1)) or zero when scale is 0.
std::vector<float> depthGradientHost(const std::vector<float> & weights, uint32_t n,
                                     float scale, uint32_t seed)
{
    std::vector<float> host(weights.size() * n, 0.0f);
    Lcg rng{seed};
    for (size_t p = 0; p < weights.size(); ++p)
    {
        host[p * n] = weights[p];
        for (uint32_t k = 1; k < n; ++k)
        {
            host[p * n + k] = scale == 0.0f ? 0.0f : static_cast<float>(scale * rng.next());
        }
    }
    return host;
}

// The GPU side of the loss: the color weights always go in; a depth term
// switches to the depth overload with the host dL/dDepth buffer.
GpuResult runGpu(ForwardPass & forward, BackwardPass & backward, GaussianCloud & cloud,
                 const Context & ctx, const Loss & loss)
{
    const size_t n = cloud.count();
    forward.run(cloud.views(), makeUniforms(ctx, static_cast<uint32_t>(n)), ForwardStage::Rasterize,
                ForwardMode::Differentiable);
    if (loss.depth.empty())
    {
        backward.run(forward, loss.rgba.data(), cloud.gradientViews());
    }
    else
    {
        const uint32_t layout = static_cast<uint32_t>(forward.depthLayout());
        const std::vector<float> depthHost =
            depthGradientHost(loss.depth, layout, loss.depthGarbage, 77u);
        backward.run(forward, loss.rgba.data(), depthHost.data(), cloud.gradientViews());
    }

    GpuResult r;
    r.image = forward.downloadOutputImage();
    r.pixelState.resize(static_cast<size_t>(kWidth) * kHeight * 2);
    forward.pixelState().download(r.pixelState.data(), r.pixelState.size() * sizeof(uint32_t));
    r.projected.resize(n);
    forward.projected().download(r.projected.data(), n * sizeof(ProjectedGaussian));
    if (forward.depthLayout() != DepthLayout::None)
    {
        // Component 0 only: D itself.
        std::vector<float> all(static_cast<size_t>(kWidth) * kHeight
                               * static_cast<uint32_t>(forward.depthLayout()));
        forward.depthBuffer().download(all.data(), all.size() * sizeof(float));
        const uint32_t layoutFloats = static_cast<uint32_t>(forward.depthLayout());
        r.depth.resize(static_cast<size_t>(kWidth) * kHeight);
        for (size_t p = 0; p < r.depth.size(); ++p)
        {
            r.depth[p] = all[p * layoutFloats];
        }
    }
    r.projGrad.resize(n * SABERCAT_PROJECTED_GRADIENT_FLOATS);
    backward.projectedGradients().download(r.projGrad.data(), r.projGrad.size() * sizeof(float));

    GaussianGradients g = cloud.downloadGradients();
    r.gMeans = std::move(g.means);
    r.gScales = std::move(g.scales);
    r.gRotations = std::move(g.rotations);
    r.gOpacity = std::move(g.opacities);
    r.gSh = std::move(g.shCoefficients);
    r.gScreen = std::move(g.screenMeans);
    return r;
}

// Scalar parameter addressing: 0-2 position, 3-5 scale, 6-9 rotation xyzw,
// 10 opacity, 11.. SH (coefficient k, channel c) at 11 + 3k + c.
constexpr int kScalarsPerGaussian = 11 + 3 * kCoeffs;

double & scalarRef(Params & p, int s)
{
    if (s < 3) return p.position[s];
    if (s < 6) return p.scale[s - 3];
    if (s < 10) return p.rotation[s - 6];
    if (s == 10) return p.opacity;
    int k = (s - 11) / 3, c = (s - 11) % 3;
    return p.sh[k][c];
}

double gpuScalar(const GpuResult & r, uint32_t g, int s)
{
    if (s < 3) return r.gMeans[g][s];
    if (s < 6) return r.gScales[g][s - 3];
    if (s < 10) return r.gRotations[g][s - 6];
    if (s == 10) return r.gOpacity[g];
    int k = (s - 11) / 3, c = (s - 11) % 3;
    return r.gSh[g * kCoeffs + k][c];
}

enum Group
{
    kGroupPosition,
    kGroupScale,
    kGroupRotation,
    kGroupOpacity,
    kGroupShDc,
    kGroupShHigher,
    kGroupCount
};
const char * kGroupNames[kGroupCount] = {"position", "scale", "rotation", "opacity", "SH DC",
                                         "SH bands 1-3"};

int groupOf(int s)
{
    if (s < 3) return kGroupPosition;
    if (s < 6) return kGroupScale;
    if (s < 10) return kGroupRotation;
    if (s == 10) return kGroupOpacity;
    return s < 14 ? kGroupShDc : kGroupShHigher;
}

// One scalar of one Gaussian, to be finite-differenced.
struct Pick
{
    uint32_t gaussian;
    int scalar;
};

std::vector<Pick> allPicks(uint32_t n)
{
    std::vector<Pick> picks;
    for (uint32_t g = 0; g < n; ++g)
    {
        for (int s = 0; s < kScalarsPerGaussian; ++s)
        {
            picks.push_back({g, s});
        }
    }
    return picks;
}

// Every scalar of `fixtures`, plus a stratified random sample: perGroup[group]
// scalars drawn uniformly from that group's scalars over all Gaussians. Only
// SH coefficients below `shCoefficients` are eligible (the degree's band count).
std::vector<Pick> samplePicks(uint32_t n, uint32_t seed, int shCoefficients,
                              const int (&perGroup)[kGroupCount],
                              const std::vector<uint32_t> & fixtures)
{
    const int scalars = 11 + 3 * shCoefficients;
    std::vector<Pick> picks;
    for (uint32_t f : fixtures)
    {
        for (int s = 0; s < scalars; ++s)
        {
            picks.push_back({f, s});
        }
    }
    Lcg rng{seed};
    for (int group = 0; group < kGroupCount; ++group)
    {
        std::vector<Pick> candidates;
        for (uint32_t g = 0; g < n; ++g)
        {
            for (int s = 0; s < scalars; ++s)
            {
                if (groupOf(s) == group)
                {
                    candidates.push_back({g, s});
                }
            }
        }
        for (int i = 0; i < perGroup[group]; ++i)
        {
            size_t index = static_cast<size_t>((rng.next() * 0.5 + 0.5) * candidates.size());
            picks.push_back(candidates[std::min(index, candidates.size() - 1)]);
        }
    }
    return picks;
}

// Tolerance, per scalar: |gpu - fd| <= kRelTol * |fd| + kAbsTol * (largest
// |fd| in the same parameter group of the same run).
//
// The finite difference is effectively exact: double precision, central
// difference with h = 1e-6, so truncation (~h^2 * f''') and round-off
// (~1e-16 * |L| / h, |L| ~ 1e2) are both below 1e-8 -- the error budget is
// all on the GPU side, which is fp32 (eps 1.2e-7) and
//   - recovers transmittance by dividing (1 - alpha) back out, which
//     amplifies relative error by up to 1 / (1 - 0.99) = 100 per clamped
//     step in the stacked fixture;
//   - sums each gradient over hundreds of pixels with mixed-sign weights, so
//     a small result can be the difference of large partial sums (relative
//     error inflates exactly there -- hence the group-scaled absolute floor
//     rather than a pure relative test);
//   - accumulates with atomics in a nondeterministic order.
// 2e-3 relative leaves ~10x headroom over what those predict for this
// scene, while a wrong derivative term (sign, missing factor, dropped
// chain-rule path) is off by O(1) relative and fails by orders of magnitude.
constexpr double kRelTol = 2e-3;
constexpr double kAbsTol = 1e-4;
constexpr double kStep = 1e-6;

struct RunReport
{
    double groupMax[kGroupCount] = {};
    double worstRatio[kGroupCount] = {};
    int checked = 0;
};

RunReport checkGradients(const char * label, std::vector<Params> params, const Context & ctx,
                         const Frozen & frozen, const Loss & loss,
                         const GpuResult & gpu, const std::vector<Pick> & picks)
{
    std::vector<double> fd(picks.size());
    RunReport rep;
    for (size_t i = 0; i < picks.size(); ++i)
    {
        const uint32_t g = picks[i].gaussian;
        const int s = picks[i].scalar;
        double & v = scalarRef(params[g], s);
        const double original = v;
        v = original + kStep;
        double plus = lossDouble(params, ctx, frozen, loss);
        v = original - kStep;
        double minus = lossDouble(params, ctx, frozen, loss);
        v = original;
        fd[i] = (plus - minus) / (2.0 * kStep);
        rep.groupMax[groupOf(s)] = std::max(rep.groupMax[groupOf(s)], std::fabs(fd[i]));
    }

    int failures = 0;
    for (size_t i = 0; i < picks.size(); ++i)
    {
        const uint32_t g = picks[i].gaussian;
        const int s = picks[i].scalar;
        {
            double expected = fd[i];
            double actual = gpuScalar(gpu, g, s);
            int group = groupOf(s);
            double allowed = kRelTol * std::fabs(expected) + kAbsTol * rep.groupMax[group];
            double err = std::fabs(actual - expected);
            double ratio = allowed > 0.0 ? err / allowed : (err == 0.0 ? 0.0 : 1e30);
            rep.worstRatio[group] = std::max(rep.worstRatio[group], ratio);
            ++rep.checked;
            if (ratio > 1.0)
            {
                if (++failures <= 20)
                {
                    std::cerr << label << ": Gaussian " << g << " scalar " << s << " ("
                              << kGroupNames[group] << ") gpu=" << actual << " fd=" << expected
                              << " err=" << err << " allowed=" << allowed << "\n";
                }
            }
        }
    }
    std::printf("%s: %d scalars checked\n", label, rep.checked);
    for (int gi = 0; gi < kGroupCount; ++gi)
    {
        std::printf("  %-13s max |grad| %.4g, worst error / allowed %.3f\n", kGroupNames[gi],
                    rep.groupMax[gi], rep.worstRatio[gi]);
    }
    if (failures > 0)
    {
        throw std::runtime_error(std::string(label) + ": " + std::to_string(failures)
                                 + " gradients outside tolerance");
    }
    return rep;
}

// screenMeans is dL/d(center) in pixels. Each center component is
// differentiated in the double model by shifting the projected center. No
// discrete decision reads the center, so the frozen render is smooth in it and
// the same tolerance model applies.
void checkScreenMeans(const char * label, const std::vector<Params> & params, const Context & ctx,
                      const Frozen & frozen, const Loss & loss, const GpuResult & gpu)
{
    const uint32_t n = static_cast<uint32_t>(params.size());
    std::vector<double> fd(static_cast<size_t>(n) * 2);
    std::vector<glm::dvec2> shift(n, glm::dvec2(0.0));
    double groupMax = 0.0;
    for (uint32_t g = 0; g < n; ++g)
    {
        for (int c = 0; c < 2; ++c)
        {
            double & v = c == 0 ? shift[g].x : shift[g].y;
            v = kStep;
            double plus = lossDouble(params, ctx, frozen, loss, &shift);
            v = -kStep;
            double minus = lossDouble(params, ctx, frozen, loss, &shift);
            v = 0.0;
            fd[static_cast<size_t>(g) * 2 + c] = (plus - minus) / (2.0 * kStep);
            groupMax = std::max(groupMax, std::fabs(fd[static_cast<size_t>(g) * 2 + c]));
        }
    }

    int failures = 0;
    double worstRatio = 0.0;
    for (uint32_t g = 0; g < n; ++g)
    {
        for (int c = 0; c < 2; ++c)
        {
            double expected = fd[static_cast<size_t>(g) * 2 + c];
            double actual = c == 0 ? gpu.gScreen[g].x : gpu.gScreen[g].y;
            double allowed = kRelTol * std::fabs(expected) + kAbsTol * groupMax;
            double err = std::fabs(actual - expected);
            double ratio = allowed > 0.0 ? err / allowed : (err == 0.0 ? 0.0 : 1e30);
            worstRatio = std::max(worstRatio, ratio);
            if (ratio > 1.0 && ++failures <= 20)
            {
                std::cerr << label << ": Gaussian " << g << " screenMeans." << (c == 0 ? 'x' : 'y')
                          << " gpu=" << actual << " fd=" << expected << " err=" << err
                          << " allowed=" << allowed << "\n";
            }
        }
    }
    std::printf("%s: screenMeans %u components vs double-model dL/d(center), max |grad| %.4g, "
                "worst error / allowed %.3f\n",
                label, 2 * n, groupMax, worstRatio);
    if (failures > 0)
    {
        throw std::runtime_error(std::string(label) + ": " + std::to_string(failures)
                                 + " screenMeans outside tolerance");
    }
}

void quantizeToFloat(std::vector<Params> & params)
{
    // The GPU sees float parameters; the double reference must start from the
    // identical values or the two differentiate slightly different scenes.
    for (Params & p : params)
    {
        p.position = glm::dvec3(glm::vec3(p.position));
        p.scale = glm::dvec3(glm::vec3(p.scale));
        p.rotation = glm::dvec4(glm::vec4(p.rotation));
        p.opacity = static_cast<double>(static_cast<float>(p.opacity));
        for (glm::dvec3 & c : p.sh)
        {
            c = glm::dvec3(glm::vec3(c));
        }
    }
}

void expect(bool condition, const std::string & what)
{
    if (!condition)
    {
        throw std::runtime_error(what);
    }
}

struct FullRun
{
    GpuResult gpu;
    Frozen frozen;
    FreezeStats stats;
};

FullRun verifyRun(const char * label, ForwardPass & forward, BackwardPass & backward,
                  GaussianCloud & cloud, const std::vector<Params> & params, const Context & ctx,
                  const Loss & loss, const std::vector<Pick> & picks)
{
    FullRun run;
    run.gpu = runGpu(forward, backward, cloud, ctx, loss);
    run.frozen = freeze(params, ctx, run.gpu.projected, run.gpu.pixelState, run.stats);

    // The double reference must reproduce what the GPU rendered before its
    // derivatives mean anything -- image and saved transmittance both.
    std::vector<double> refDepth;
    std::vector<glm::dvec4> image = renderDouble(params, ctx, run.frozen, nullptr, &refDepth);
    double worstImage = 0.0, worstT = 0.0;
    for (size_t i = 0; i < image.size(); ++i)
    {
        for (int c = 0; c < 4; ++c)
        {
            worstImage =
                std::max(worstImage, std::fabs(image[i][c] - static_cast<double>(run.gpu.image[i][c])));
        }
        float gpuT;
        std::memcpy(&gpuT, &run.gpu.pixelState[i * 2], sizeof(float));
        worstT = std::max(worstT, std::fabs((1.0 - image[i].w) - static_cast<double>(gpuT)));
    }
    std::printf("%s: double reference vs GPU image worst %.3g, final T worst %.3g; "
                "%u clamped pairs, %u early-out pixels\n",
                label, worstImage, worstT, run.stats.clampedPairs, run.stats.earlyOutPixels);
    expect(worstImage < 1e-4, std::string(label) + ": reference image disagrees with the GPU");
    expect(worstT < 1e-4, std::string(label) + ": reference transmittance disagrees with pixelState");

    // The depth reference must reproduce the GPU's D as well (depth layouts only).
    if (!run.gpu.depth.empty())
    {
        double worstDepth = 0.0, maxDepth = 0.0;
        for (size_t p = 0; p < run.gpu.depth.size(); ++p)
        {
            worstDepth = std::max(worstDepth, std::fabs(refDepth[p] - run.gpu.depth[p]));
            maxDepth = std::max(maxDepth, std::fabs(static_cast<double>(run.gpu.depth[p])));
        }
        const double relDepth = worstDepth / std::max(maxDepth, 1e-12);
        std::printf("%s: double reference vs GPU depth D worst |error| %.3g, relative to max D "
                    "%.3g: %.3g\n",
                    label, worstDepth, maxDepth, relDepth);
        expect(relDepth < 1e-3, std::string(label) + ": reference depth disagrees with the GPU");
    }

    checkGradients(label, params, ctx, run.frozen, loss, run.gpu, picks);
    checkScreenMeans(label, params, ctx, run.frozen, loss, run.gpu);
    return run;
}

// Scene of `count` pseudo-random Gaussians after two fixtures, placed in camera
// space like the hand-built cases. Index 0 sits just beyond kMinDepth (0.2).
// Index 1 is centered 0.5 in front of the camera with 3-sigma reaching 1.2, so
// the camera lies inside its 3-sigma ellipsoid.
std::vector<Params> randomScene(const glm::mat4 & cameraToWorld, uint32_t count, uint32_t seed)
{
    auto world = [&](double x, double y, double depth)
    {
        glm::vec4 w = cameraToWorld * glm::vec4(static_cast<float>(x), static_cast<float>(y),
                                                static_cast<float>(-depth), 1.0f);
        return glm::dvec3(w);
    };
    Lcg rng{seed};
    auto unit = [&]() { return 0.5 * (rng.next() + 1.0); }; // [0, 1)
    const glm::dvec4 identity(0.0, 0.0, 0.0, 1.0);

    std::vector<Params> params;
    params.push_back({world(0.02, 0.0, 0.25), {0.05, 0.05, 0.05}, identity, 0.5, {}});
    params.push_back({world(0.0, 0.0, 0.5), {0.4, 0.4, 0.4}, identity, 0.6, {}});
    for (uint32_t i = 0; i < count; ++i)
    {
        Params p{};
        const double depth = 1.0 + 4.0 * unit();
        const double x = depth * 0.6 * (2.0 * unit() - 1.0);
        const double y = depth * 0.45 * (2.0 * unit() - 1.0);
        p.position = world(x, y, depth);
        p.scale = glm::dvec3(0.03 + 0.25 * unit(), 0.03 + 0.25 * unit(), 0.03 + 0.25 * unit());
        p.rotation = glm::normalize(glm::dvec4(rng.next(), rng.next(), rng.next(), rng.next()));
        p.opacity = 0.1 + 0.8 * unit();
        for (int k = 0; k < kCoeffs; ++k)
        {
            const double scale = k == 0 ? 0.6 : 0.15;
            const double r = rng.next();
            const double g = rng.next();
            const double b = rng.next();
            p.sh[k] = glm::dvec3(r, g, b) * scale;
        }
        params.push_back(p);
    }
    return params;
}

// Every gradient stream of a download, flattened with the pad lanes.
std::vector<float> flatten(const GaussianGradients & g)
{
    std::vector<float> out;
    for (const std::vector<glm::vec4> * stream : {&g.means, &g.scales, &g.rotations, &g.shCoefficients,
                                                  &g.screenMeans})
    {
        for (const glm::vec4 & v : *stream)
        {
            out.insert(out.end(), &v.x, &v.x + 4);
        }
    }
    out.insert(out.end(), g.opacities.begin(), g.opacities.end());
    return out;
}

// Runs `attempt`, which must throw a message containing `expected`, and checks
// the gradient streams of `cloud` are byte-identical afterwards: validation
// runs before the pass writes anything.
template <typename Attempt>
void expectRejected(const std::string & what, const char * expected, const GaussianCloud & cloud,
                    Attempt attempt)
{
    const std::vector<float> before = flatten(cloud.downloadGradients());
    bool threw = false;
    std::string message;
    try
    {
        attempt();
    }
    catch (const std::exception & e)
    {
        threw = true;
        message = e.what();
    }
    expect(threw, what + ": accepted, expected a rejection");
    expect(message.find(expected) != std::string::npos,
           what + ": wrong message: " + message);
    const std::vector<float> after = flatten(cloud.downloadGradients());
    expect(before.size() == after.size()
               && std::memcmp(before.data(), after.data(), before.size() * sizeof(float)) == 0,
           what + ": gradients changed on a rejected call");
    std::printf("  rejected (state unchanged): %s -> \"%s\"\n", what.c_str(), expected);
}

// The same flattening as flatten(), from a checked-out GpuResult.
std::vector<float> flattenResult(const GpuResult & r)
{
    std::vector<float> out;
    for (const std::vector<glm::vec4> * stream : {&r.gMeans, &r.gScales, &r.gRotations, &r.gSh,
                                                  &r.gScreen})
    {
        for (const glm::vec4 & v : *stream)
        {
            out.insert(out.end(), &v.x, &v.x + 4);
        }
    }
    out.insert(out.end(), r.gOpacity.begin(), r.gOpacity.end());
    return out;
}

// Largest |a - b| and largest |a| over two equally sized gradient streams.
struct StreamDiff
{
    double worst = 0.0;
    double largest = 0.0;
};

StreamDiff compareStreams(const std::vector<float> & a, const std::vector<float> & b)
{
    expect(a.size() == b.size(), "gradient stream sizes differ");
    StreamDiff d;
    for (size_t i = 0; i < a.size(); ++i)
    {
        d.worst = std::max(d.worst, std::fabs(static_cast<double>(a[i]) - b[i]));
        d.largest = std::max(d.largest, std::fabs(static_cast<double>(a[i])));
    }
    return d;
}

} // namespace

int main()
{
    Device device;
    int result = 0;
    try
    {
        device.initialize(/*instanceExtensions=*/{}, /*enableValidation=*/true);

        // Rotated + translated camera, so the world->camera rotation W in
        // every chain-rule step is non-trivial.
        const glm::mat4 cameraToWorld =
            glm::translate(glm::mat4(1.0f), glm::vec3(0.3f, -0.2f, 2.0f))
            * glm::rotate(glm::mat4(1.0f), 0.15f, glm::vec3(0.0f, 1.0f, 0.0f))
            * glm::rotate(glm::mat4(1.0f), -0.1f, glm::vec3(1.0f, 0.0f, 0.0f));
        Context ctx;
        ctx.worldToCamera = glm::inverse(cameraToWorld);
        ctx.cameraPosition = glm::vec3(cameraToWorld[3]);

        // Gaussians are PLACED in camera space (x right, y up, looking down
        // -z) so each lands where its case needs it, then moved to world.
        auto world = [&](double x, double y, double depth)
        {
            glm::vec4 w = cameraToWorld * glm::vec4(static_cast<float>(x), static_cast<float>(y),
                                                    static_cast<float>(-depth), 1.0f);
            return glm::dvec3(w);
        };
        auto quat = [](double x, double y, double z, double w)
        { return glm::normalize(glm::dvec4(x, y, z, w)); };

        std::vector<Params> params(8);
        // 0: large, anisotropic, rotated -- the bread-and-butter case.
        params[0] = {world(0.0, 0.0, 4.0), {0.45, 0.25, 0.3}, quat(0.2, 0.3, 0.1, 0.9), 0.7, {}};
        // 1: red channel deliberately clamped high (DC red 3.0 -> raw > 1).
        params[1] = {world(0.6, 0.3, 3.2), {0.15, 0.4, 0.2}, quat(-0.3, 0.1, 0.4, 0.85), 0.6, {}};
        // 2: sub-pixel (true sigma ~0.1px) -- the compensation dominates here.
        params[2] = {world(-0.9, -0.5, 4.5), {0.008, 0.008, 0.008}, quat(0.0, 0.0, 0.0, 1.0), 0.8,
                     {}};
        // 3: center at x/z = 0.8 > 0.65 = 1.3 * half-FOV tangent, so the
        // Jacobian clamp is active; its footprint still reaches the image.
        params[3] = {world(3.2, 0.2, 4.0), {0.6, 0.35, 0.3}, quat(0.05, 0.02, 0.1, 1.0), 0.75, {}};
        // 4-7: a stack on one spot: three near-opaque (alpha clamps at 0.99,
        // T collapses below 1e-4) and one behind them that the early-out
        // cuts off at the center but that still shows at the rim.
        const double stackOpacity[4] = {0.999, 0.998, 0.997, 0.9};
        for (int i = 0; i < 4; ++i)
        {
            double depth = 3.6 + 0.2 * i;
            params[4 + i] = {world(-0.3 * depth + 0.01 * i, 0.225 * depth - 0.01 * i, depth),
                             {0.25 + 0.02 * i, 0.22, 0.2},
                             quat(0.1 * i, 0.05, -0.05 * i, 1.0),
                             stackOpacity[i],
                             {}};
        }
        Lcg rng{12345u};
        for (Params & p : params)
        {
            for (int k = 0; k < kCoeffs; ++k)
            {
                double scale = k == 0 ? 0.6 : 0.15; // keeps colors inside (0,1)
                p.sh[k] = glm::dvec3(rng.next(), rng.next(), rng.next()) * scale;
            }
        }
        params[1].sh[0].x = 3.0;
        quantizeToFloat(params);

        // dL/dImage: fixed pseudo-random weights over all of RGBA.
        std::vector<float> weights(static_cast<size_t>(kWidth) * kHeight * 4);
        Lcg wrng{777u};
        for (float & w : weights)
        {
            w = static_cast<float>(wrng.next());
        }

        GaussianCloud cloud(device);
        uploadParams(cloud, params);
        ForwardPass forward(device, kWidth, kHeight, SHADER_DIR);
        BackwardPass backward(device, SHADER_DIR);
        // Same scene, with the depth output (n = 4) for the depth-gradient runs.
        ForwardPass forwardDepth(device, kWidth, kHeight, SHADER_DIR,
                                 DepthLayout::DepthCoverageFront);

        // -- Compensation OFF (vanilla 3DGS) --
        Context off = ctx;
        off.renderFlags = 0;
        FullRun runOff = verifyRun("compensation off, SH3", forward, backward, cloud, params, off,
                                   weights, allPicks(static_cast<uint32_t>(params.size())));

        // Fixture guards: each special case must actually be in effect, or
        // the run above proved less than it claims.
        expect(runOff.frozen.clampSignX[3] != 0, "frustum clamp fixture is not clamped");
        expect(runOff.frozen.active[3], "frustum clamp fixture was culled");
        expect(runOff.frozen.colorClamp[1][0] == 1, "color clamp fixture is not clamped");
        expect(runOff.gpu.gSh[1 * kCoeffs][0] == 0.0f && runOff.gpu.gSh[1 * kCoeffs][1] != 0.0f,
               "clamped red channel must get exactly zero SH gradient (green must not)");
        expect(runOff.stats.clampedPairs > 0, "no (pixel, Gaussian) pair hit the 0.99 alpha clamp");
        expect(runOff.stats.earlyOutPixels > 0, "no pixel's early-out cut off a contributor");
        expect(runOff.frozen.active[2], "sub-pixel fixture was culled");

        // -- Compensation ON (antialiased / mip-splatting) --
        Context on = ctx;
        on.renderFlags = kRenderFlagAntialiased;
        FullRun runOn = verifyRun("compensation on, SH3", forward, backward, cloud, params, on,
                                  weights, allPicks(static_cast<uint32_t>(params.size())));
        const double offOpacity = runOff.gpu.gOpacity[2], onOpacity = runOn.gpu.gOpacity[2];
        expect(std::fabs(onOpacity - offOpacity) > 0.1 * std::fabs(offOpacity),
               "sub-pixel Gaussian's opacity gradient ignores the compensation flag");
        std::printf("compensation reaches backward: sub-pixel opacity gradient %.4g (off) vs "
                    "%.4g (on)\n",
                    offOpacity, onOpacity);

        // -- SH degree cap 1: bands 2-3 must be EXACTLY zero on the GPU. --
        Context capped = on;
        capped.shDegree = 1;
        FullRun runCap = verifyRun("compensation on, SH1", forward, backward, cloud, params, capped,
                                   weights, allPicks(static_cast<uint32_t>(params.size())));
        for (uint32_t g = 0; g < params.size(); ++g)
        {
            for (int k = 4; k < kCoeffs; ++k)
            {
                expect(runCap.gpu.gSh[g * kCoeffs + k] == glm::vec4(0.0f),
                       "SH coefficient above the degree cap received gradient");
            }
        }
        std::printf("SH degree cap OK: bands 2-3 exactly zero at degree 1\n");

        // screenMeans is backward_rasterize's dL/d(center), passed through.
        {
            std::vector<float> pg(params.size() * SABERCAT_PROJECTED_GRADIENT_FLOATS);
            backward.projectedGradients().download(pg.data(), pg.size() * sizeof(float));
            for (uint32_t g = 0; g < params.size(); ++g)
            {
                const glm::vec4 & s = runCap.gpu.gScreen[g];
                expect(s.x == pg[g * SABERCAT_PROJECTED_GRADIENT_FLOATS + 0]
                           && s.y == pg[g * SABERCAT_PROJECTED_GRADIENT_FLOATS + 1]
                           && s.z == 0.0f && s.w == 0.0f,
                       "screenMeans does not carry dL/d(center)");
            }
        }

        // -- Structural: viewer knobs can't reach a Differentiable run. --
        {
            LODParams aggressive{};
            aggressive.enabled = 1;
            aggressive.nearDistance = 0.0f;
            aggressive.farDistance = 0.001f;
            aggressive.maxDropDistance = 1.0f;
            aggressive.maxDropDensity = 1.0f;
            aggressive.densityReference = 1.0f;
            aggressive.appearanceEnabled = 1;
            aggressive.appearanceMaxDropDistance = 1.0f;
            aggressive.minScreenRadius = 1000.0f;
            forward.setLODParams(aggressive);

            CameraUniforms u = makeUniforms(off, static_cast<uint32_t>(params.size()));
            forward.run(cloud.views(), u, ForwardStage::Rasterize, ForwardMode::Viewer);
            expect(forward.lastInstanceCount() == 0,
                   "the aggressive LOD should cull everything in Viewer mode");

            bool threw = false;
            try
            {
                backward.run(forward, weights.data(), cloud.gradientViews());
            }
            catch (const std::exception &)
            {
                threw = true;
            }
            expect(threw, "backward accepted a Viewer-mode forward");

            forward.run(cloud.views(), u, ForwardStage::Rasterize, ForwardMode::Differentiable);
            std::vector<glm::vec4> image = forward.downloadOutputImage();
            expect(std::memcmp(image.data(), runOff.gpu.image.data(),
                               image.size() * sizeof(glm::vec4))
                       == 0,
                   "viewer LOD knobs changed a Differentiable render");

            forward.run(cloud.views(), u, ForwardStage::Project, ForwardMode::Differentiable);
            threw = false;
            try
            {
                backward.run(forward, weights.data(), cloud.gradientViews());
            }
            catch (const std::exception &)
            {
                threw = true;
            }
            expect(threw, "backward accepted an incomplete forward");
            forward.setLODParams(LODParams{});
            std::printf("structural OK: viewer LOD culled all %zu Gaussians in Viewer mode, "
                        "Differentiable render bit-identical; backward refused Viewer-mode "
                        "and incomplete forwards\n",
                        params.size());
        }

        // -- Scale: 200 random Gaussians (+2 fixtures), rotated camera. --
        // Finite differences on a stratified sample of 300 scalars across all
        // five streams (plus every scalar of the two fixtures); screenMeans on
        // every Gaussian. Sampling only narrows which scalars are checked: the
        // tolerance floor is the largest sampled |fd| in each group.
        const uint32_t sceneRandom = 200;
        const std::vector<uint32_t> fixtures = {0, 1};
        const int perGroup[kGroupCount] = {60, 60, 60, 30, 30, 60};
        std::vector<Params> scene = randomScene(cameraToWorld, sceneRandom, 4242u);
        quantizeToFloat(scene);
        const uint32_t sceneCount = static_cast<uint32_t>(scene.size());
        GaussianCloud bigCloud(device);
        uploadParams(bigCloud, scene);

        Context sceneOn = on;
        sceneOn.shDegree = 3;
        FullRun runScene = verifyRun("202 Gaussians, compensation on, SH3", forward, backward,
                                     bigCloud, scene, sceneOn, weights,
                                     samplePicks(sceneCount, 11u, kCoeffs, perGroup, fixtures));
        expect(runScene.frozen.active[0] && runScene.frozen.active[1],
               "scene fixtures were culled before the depth/3-sigma checks ran");

        // The second backward call with doubled dL/dImage must OVERWRITE the
        // gradients: 2*g1, not g1 + 2*g1 (the sum would be 3*g1).
        {
            const std::vector<float> first = flatten(bigCloud.downloadGradients());
            std::vector<float> doubled(weights.size());
            for (size_t i = 0; i < weights.size(); ++i)
            {
                doubled[i] = 2.0f * weights[i];
            }
            backward.run(forward, doubled.data(), bigCloud.gradientViews());
            const std::vector<float> second = flatten(bigCloud.downloadGradients());
            expect(first.size() == second.size(), "gradient stream sizes changed");

            // Tolerance: fp32 atomics reorder sums of up to a few hundred terms
            // per gradient; that noise is ~1e-6 of the largest entry, so 1e-4
            // of the largest entry leaves ~100x headroom.
            double maxFirst = 0.0, worstOverwrite = 0.0, worstFromSum = 0.0;
            for (float v : first)
            {
                maxFirst = std::max(maxFirst, std::fabs(static_cast<double>(v)));
            }
            for (size_t i = 0; i < first.size(); ++i)
            {
                const double f = first[i], s = second[i];
                worstOverwrite = std::max(worstOverwrite, std::fabs(s - 2.0 * f));
                worstFromSum = std::max(worstFromSum, std::fabs(s - 3.0 * f));
            }
            std::printf("second backward (dL/dImage x2): worst |g2 - 2*g1| %.3g, "
                        "worst |g2 - 3*g1| %.3g (largest |g1| %.3g)\n",
                        worstOverwrite, worstFromSum, maxFirst);
            expect(worstOverwrite <= 1e-4 * maxFirst,
                   "second backward call did not overwrite the gradients");
            expect(worstFromSum >= 0.5 * maxFirst,
                   "overwrite check cannot tell overwrite from accumulation");
        }

        // -- Scale, compensation off, SH degree 2. --
        // Bands 3 (coefficients 9-15) must be exactly zero on the GPU; the
        // sample covers only the degree-2 coefficients (0-8).
        Context sceneDeg2 = off;
        sceneDeg2.shDegree = 2;
        FullRun runDeg2 = verifyRun("202 Gaussians, compensation off, SH2", forward, backward,
                                    bigCloud, scene, sceneDeg2, weights,
                                    samplePicks(sceneCount, 12u, 9, perGroup, fixtures));
        for (uint32_t g = 0; g < sceneCount; ++g)
        {
            for (int k = 9; k < kCoeffs; ++k)
            {
                expect(runDeg2.gpu.gSh[g * kCoeffs + k] == glm::vec4(0.0f),
                       "SH degree 2 gave a gradient to a band-3 coefficient");
            }
        }
        std::printf("SH degree 2 OK: band-3 coefficients exactly zero for all %u Gaussians\n",
                    sceneCount);

        // -- Depth gradient: L = sum w RGBA + sum_p g_p D_p, D the forward's
        // depth output. Depth-only runs (dL/dImage = 0) isolate the z path. --
        {
            const uint32_t count8 = static_cast<uint32_t>(params.size());
            std::vector<float> depthWeights(static_cast<size_t>(kWidth) * kHeight);
            Lcg drng{4321u};
            for (float & g : depthWeights)
            {
                g = static_cast<float>(drng.next());
            }
            const std::vector<float> zeroImage(weights.size(), 0.0f);
            // The GPU's components 1..3 get garbage of scale 1e3 (only component 0 may be read).
            const Loss depthOnly(zeroImage, depthWeights, 1000.0f);
            const Loss mixed(weights, depthWeights, 1000.0f);

            FullRun dOff = verifyRun("depth only, compensation off, SH3", forwardDepth, backward,
                                     cloud, params, off, depthOnly, allPicks(count8));
            expect(dOff.frozen.clampSignX[3] != 0 && dOff.frozen.active[3],
                   "depth run: frustum clamp fixture is not in effect");
            expect(dOff.stats.clampedPairs > 0 && dOff.stats.earlyOutPixels > 0,
                   "depth run: alpha clamp / early-out fixture is not in effect");

            FullRun dOn = verifyRun("depth only, compensation on, SH3", forwardDepth, backward,
                                    cloud, params, on, depthOnly, allPicks(count8));
            FullRun mOn = verifyRun("depth + color, compensation on, SH3", forwardDepth, backward,
                                    cloud, params, on, mixed, allPicks(count8));

            // Structural checks on the depth-only run.
            const GpuResult & g = dOn.gpu;
            for (uint32_t gi = 0; gi < count8; ++gi)
            {
                for (int k = 0; k < kCoeffs; ++k)
                {
                    expect(g.gSh[gi * kCoeffs + k] == glm::vec4(0.0f),
                           "depth-only run gave a color (SH) gradient");
                }
            }
            auto maxAbs4 = [](const std::vector<glm::vec4> & v)
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
            const double meansMax = maxAbs4(g.gMeans), scalesMax = maxAbs4(g.gScales);
            const double rotationsMax = maxAbs4(g.gRotations), screenMax = maxAbs4(g.gScreen);
            double opacityMax = 0.0;
            for (float o : g.gOpacity)
            {
                opacityMax = std::max(opacityMax, std::fabs(static_cast<double>(o)));
            }
            expect(meansMax > 0.0 && scalesMax > 0.0 && rotationsMax > 0.0 && opacityMax > 0.0
                       && screenMax > 0.0,
                   "depth-only run: a geometry gradient stream is identically zero");
            // The camera's view axis in world space: row 2 of world-to-camera.
            const glm::vec3 viewAxis(on.worldToCamera[0][2], on.worldToCamera[1][2],
                                     on.worldToCamera[2][2]);
            double alongAxis = 0.0;
            for (const glm::vec4 & m : g.gMeans)
            {
                const double along = glm::dot(glm::vec3(m), viewAxis);
                alongAxis = std::max(alongAxis, std::fabs(along));
            }
            expect(alongAxis > 0.0, "depth-only run: no mean gradient along the camera axis");
            // dL/dz reaches the alpha-clamped stack (4..6) through the clamp.
            for (uint32_t gi : {4u, 5u, 6u})
            {
                expect(g.projGrad[gi * SABERCAT_PROJECTED_GRADIENT_FLOATS + 9] != 0.0f,
                       "depth-only run: dL/dz is zero for an alpha-clamped Gaussian");
            }
            // Color-only runs must leave dL/dz (slot 9) exactly zero.
            for (uint32_t gi = 0; gi < count8; ++gi)
            {
                expect(runOn.gpu.projGrad[gi * SABERCAT_PROJECTED_GRADIENT_FLOATS + 9] == 0.0f,
                       "color-only run wrote the depth slot");
            }
            std::printf("depth-only structure OK: SH gradients exactly zero; max |grad| means "
                        "%.4g, scales %.4g, rotations %.4g, opacity %.4g, screenMeans %.4g; means "
                        "along the camera axis up to %.4g; dL/dz nonzero on the alpha-clamped "
                        "stack; color-only dL/dz exactly zero\n",
                        meansMax, scalesMax, rotationsMax, opacityMax, screenMax, alongAxis);

            // Linearity: mixed = color-only + depth-only, within fp32 atomic order.
            const std::vector<float> colorFlat = flattenResult(runOn.gpu);
            const std::vector<float> depthFlat = flattenResult(dOn.gpu);
            std::vector<float> summed(colorFlat.size());
            for (size_t i = 0; i < summed.size(); ++i)
            {
                summed[i] = colorFlat[i] + depthFlat[i];
            }
            const StreamDiff lin = compareStreams(flattenResult(mOn.gpu), summed);
            std::printf("mixed = color + depth (linearity): worst |diff| %.3g, largest %.3g, "
                        "worst / largest %.3g\n",
                        lin.worst, lin.largest, lin.worst / lin.largest);
            expect(lin.worst <= 1e-4 * lin.largest, "mixed gradient is not color + depth");

            // -- No-depth path, 8-Gaussian scene, compensation on, SH3. --
            const CameraUniforms onUniforms = makeUniforms(on, count8);
            auto forwardOn = [&](ForwardPass & fp)
            {
                fp.run(cloud.views(), onUniforms, ForwardStage::Rasterize,
                       ForwardMode::Differentiable);
            };
            forwardOn(forward);
            backward.run(forward, weights.data(), cloud.gradientViews());
            const std::vector<float> gOld = flatten(cloud.downloadGradients());

            forwardOn(forward);
            backward.run(forward, weights.data(), static_cast<const float *>(nullptr),
                         cloud.gradientViews());
            const StreamDiff nullDepth = compareStreams(gOld, flatten(cloud.downloadGradients()));

            Buffer imageDev(device, static_cast<VkDeviceSize>(kWidth) * kHeight * 16);
            imageDev.upload(weights.data(), weights.size() * sizeof(float));
            forwardOn(forward);
            backward.run(forward, imageDev.view(16), BufferView{}, cloud.gradientViews());
            const StreamDiff viewNone = compareStreams(gOld, flatten(cloud.downloadGradients()));

            // A zero depth gradient on a depth-layout forward is the color gradient.
            const std::vector<float> zeroDepth(static_cast<size_t>(kWidth) * kHeight * 4, 0.0f);
            forwardOn(forwardDepth);
            backward.run(forwardDepth, weights.data(), zeroDepth.data(), cloud.gradientViews());
            const StreamDiff zeroD = compareStreams(gOld, flatten(cloud.downloadGradients()));
            const double colorLargest = compareStreams(gOld, gOld).largest;
            std::printf("no-depth paths vs color-only (largest |g| %.4g): null depth worst %.3g, "
                        "view with BufferView{} worst %.3g, zero depth worst %.3g\n",
                        colorLargest, nullDepth.worst, viewNone.worst, zeroD.worst);
            expect(nullDepth.worst <= 1e-4 * colorLargest,
                   "null depth gradient changed the color gradient");
            expect(viewNone.worst <= 1e-4 * colorLargest,
                   "BufferView{} depth changed the color gradient");
            expect(zeroD.worst <= 1e-4 * colorLargest,
                   "zero depth gradient changed the color gradient");

            // -- Garbage independence: components 1..n-1 of the depth gradient
            // are ignored, on layouts 1, 2 and 4. --
            ForwardPass forwardL1(device, kWidth, kHeight, SHADER_DIR, DepthLayout::Depth);
            ForwardPass forwardL2(device, kWidth, kHeight, SHADER_DIR, DepthLayout::DepthCoverage);
            ForwardPass * layouts[] = {&forwardL1, &forwardL2, &forwardDepth};
            for (ForwardPass * fp : layouts)
            {
                const uint32_t n = static_cast<uint32_t>(fp->depthLayout());
                forwardOn(*fp);
                const std::vector<float> zeroHost =
                    depthGradientHost(depthWeights, n, 0.0f, 77u);
                const std::vector<float> garbageHost =
                    depthGradientHost(depthWeights, n, 1000.0f, 77u);
                forwardOn(*fp);
                backward.run(*fp, weights.data(), zeroHost.data(), cloud.gradientViews());
                const std::vector<float> zerosRun = flatten(cloud.downloadGradients());
                forwardOn(*fp);
                backward.run(*fp, weights.data(), garbageHost.data(), cloud.gradientViews());
                const StreamDiff garbage =
                    compareStreams(zerosRun, flatten(cloud.downloadGradients()));
                std::printf("garbage in %u unread components (layout %u): worst |diff| %.3g, "
                            "largest |g| %.4g\n",
                            n - 1, n, garbage.worst, garbage.largest);
                expect(garbage.worst <= 1e-4 * garbage.largest,
                       "garbage in unread depth components changed the gradient");
                expect(garbage.largest > 1e-3, "garbage check is vacuous: gradient is tiny");
            }

            // -- Scale: 202 Gaussians, mixed loss, compensation on, SH3. --
            FullRun scene202 = verifyRun("202 Gaussians, color + depth, compensation on, SH3",
                                         forwardDepth, backward, bigCloud, scene, sceneOn, mixed,
                                         samplePicks(sceneCount, 13u, kCoeffs, perGroup, fixtures));
            expect(scene202.frozen.active[0] && scene202.frozen.active[1],
                   "scene fixtures were culled in the depth run");
        }

        // -- Negatives on the scale scene's Differentiable forward. --
        forward.run(bigCloud.views(), makeUniforms(sceneOn, sceneCount), ForwardStage::Rasterize,
                    ForwardMode::Differentiable);
        {
            std::vector<Params> bigger = scene;
            bigger.push_back(scene[0]);
            GaussianCloud other(device);
            uploadParams(other, bigger);
            expectRejected("gradient views sized for 203 Gaussians", "sized for a different Gaussian count",
                           bigCloud, [&]()
                           { backward.run(forward, weights.data(), other.gradientViews()); });

            CanonicalGaussianGradientViews unbound = bigCloud.gradientViews();
            unbound.opacity = BufferView{};
            expectRejected("opacity gradient view unbound", "every gradient view must be bound",
                           bigCloud, [&]()
                           { backward.run(forward, weights.data(), unbound); });

            Buffer shortImage(device, static_cast<VkDeviceSize>(kWidth) * kHeight * 16 - 16);
            expectRejected("image gradient one RGBA texel short", "smaller than width*height RGBA",
                           bigCloud, [&]()
                           { backward.run(forward, shortImage.view(16), bigCloud.gradientViews()); });

            // Depth gradient rejections: a layout-None forward has no depth to
            // differentiate, and a depth view one float short is refused.
            const std::vector<float> depthAny(static_cast<size_t>(kWidth) * kHeight * 4, 0.0f);
            expectRejected("depth gradient on a layout-None forward", "no depth output", bigCloud,
                           [&]()
                           { backward.run(forward, weights.data(), depthAny.data(),
                                          bigCloud.gradientViews()); });

            forwardDepth.run(bigCloud.views(), makeUniforms(sceneOn, sceneCount),
                             ForwardStage::Rasterize, ForwardMode::Differentiable);
            Buffer imageDev(device, static_cast<VkDeviceSize>(kWidth) * kHeight * 16);
            Buffer shortDepth(device, static_cast<VkDeviceSize>(kWidth) * kHeight * 16 - 4);
            expectRejected("depth view one float short", "smaller than width*height*n floats",
                           bigCloud, [&]()
                           { backward.run(forwardDepth, imageDev.view(16), shortDepth.view(16),
                                          bigCloud.gradientViews()); });
        }

        std::printf("backward_test PASSED\n");
    }
    catch (const std::exception & e)
    {
        std::cerr << "backward_test FAILED: " << e.what() << std::endl;
        result = 1;
    }
    device.shutdown();
    return result;
}
