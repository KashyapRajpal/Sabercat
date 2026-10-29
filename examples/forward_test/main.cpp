// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Headless verification harness for ForwardPass.
//
// Stage 1 (this file, for now): forward_project alone. A tiny synthetic
// 2-Gaussian cloud + a fixed identity-view camera is projected on the GPU;
// the result is checked against an independent C++ reference
// implementation of forward_project.slang's math (mirrors how
// gaussian_cloud_test recomputes expected sigmoid/exp/quaternion values in
// C++ rather than hardcoding pre-computed decimal literals -- safer than
// hand-deriving the projection math's many intermediate steps by hand).
//
// The camera's cameraToWorld is the identity, chosen deliberately: it
// makes world space and camera space coincide, so this reference doesn't
// need to get glm's column-major mat4/mat3 layout to agree with Slang's
// row-major-authored float4x4/float3x3 -- multiplying by identity is
// unambiguous either way. Later stages that need a non-trivial camera
// transform will need to settle that question; not needed here.

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "sabercat/approx.h"
#include "sabercat/buffer.h"
#include "sabercat/camera.h"
#include "sabercat/device.h"
#include "sabercat/forward_pass.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/render_config.h"

#include "../common/shader_dir.h"

using namespace sabercat;

namespace
{

constexpr uint32_t kImageWidth = 256;
constexpr uint32_t kImageHeight = 256;
constexpr float kFx = 500.0f;
constexpr float kFy = 500.0f;
constexpr float kCx = 128.0f;
constexpr float kCy = 128.0f;

struct SyntheticGaussian
{
    glm::vec3 position;
    glm::vec3 scale;
    glm::vec4 rotation; // xyzw, normalized
    float opacity;
    glm::vec3 sh0;
    // Higher-order SH coefficients (bands 1-3), band-ascending to
    // match the loader's layout. Value-initializes to zero, so every
    // fixture below predating the higher bands keeps behaving exactly as it did when only
    // the DC term existed.
    glm::vec3 shRest[kShCoefficientsPerGaussian - 1] = {};
};

// -- Reference math, mirroring shaders/forward_common.slangh +
// forward_project.slang line for line. Plain scalar 3x3/2x3 arrays, not
// glm::mat3, so there's no row/column-major ambiguity to get wrong. --

// Mirrors forward_common.slangh's evaluateSH, including its band-ascending
// coefficient order and the +0.5 / clamp tail. `coefficients` is the
// Gaussian's full 16-entry run; only the first (degree+1)^2 are read.
glm::vec3 referenceEvaluateSH(uint32_t degree, const glm::vec3 * coefficients,
                              const glm::vec3 & dir)
{
    glm::vec3 result = kShC0 * coefficients[0];

    if (degree >= 1)
    {
        float x = dir.x, y = dir.y, z = dir.z;
        result += kShC1 * (-y * coefficients[1] + z * coefficients[2] - x * coefficients[3]);

        if (degree >= 2)
        {
            float xx = x * x, yy = y * y, zz = z * z;
            float xy = x * y, yz = y * z, xz = x * z;
            result += kShC2_0 * xy * coefficients[4] + kShC2_1 * yz * coefficients[5]
                      + kShC2_2 * (2.0f * zz - xx - yy) * coefficients[6]
                      + kShC2_3 * xz * coefficients[7] + kShC2_4 * (xx - yy) * coefficients[8];

            if (degree >= 3)
            {
                result += kShC3_0 * y * (3.0f * xx - yy) * coefficients[9]
                          + kShC3_1 * xy * z * coefficients[10]
                          + kShC3_2 * y * (4.0f * zz - xx - yy) * coefficients[11]
                          + kShC3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * coefficients[12]
                          + kShC3_4 * x * (4.0f * zz - xx - yy) * coefficients[13]
                          + kShC3_5 * z * (xx - yy) * coefficients[14]
                          + kShC3_6 * x * (xx - 3.0f * yy) * coefficients[15];
            }
        }
    }

    return glm::clamp(result + 0.5f, 0.0f, 1.0f);
}

// Packs a SyntheticGaussian's DC + higher-order terms into one contiguous
// 16-entry run, matching how the loader/GPU buffer stores them.
void packSHCoefficients(const SyntheticGaussian & g, glm::vec3 * out)
{
    out[0] = g.sh0;
    for (int i = 1; i < kShCoefficientsPerGaussian; ++i)
    {
        out[i] = g.shRest[i - 1];
    }
}

void quaternionToMatrix(const glm::vec4 & q, float R[3][3])
{
    float x = q.x, y = q.y, z = q.z, w = q.w;
    R[0][0] = 1.0f - 2.0f * (y * y + z * z);
    R[0][1] = 2.0f * (x * y - z * w);
    R[0][2] = 2.0f * (x * z + y * w);
    R[1][0] = 2.0f * (x * y + z * w);
    R[1][1] = 1.0f - 2.0f * (x * x + z * z);
    R[1][2] = 2.0f * (y * z - x * w);
    R[2][0] = 2.0f * (x * z - y * w);
    R[2][1] = 2.0f * (y * z + x * w);
    R[2][2] = 1.0f - 2.0f * (x * x + y * y);
}

void matMul3(const float A[3][3], const float B[3][3], float Out[3][3])
{
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            float s = 0.0f;
            for (int k = 0; k < 3; ++k)
            {
                s += A[i][k] * B[k][j];
            }
            Out[i][j] = s;
        }
    }
}

void transpose3(const float A[3][3], float Out[3][3])
{
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            Out[j][i] = A[i][j];
        }
    }
}

void covarianceFromScaleRotation(const glm::vec3 & scale, const glm::vec4 & rotation,
                                 float Sigma[3][3])
{
    float R[3][3];
    quaternionToMatrix(rotation, R);
    float S[3][3] = {
        {scale.x * scale.x, 0.0f, 0.0f},
        {0.0f, scale.y * scale.y, 0.0f},
        {0.0f, 0.0f, scale.z * scale.z},
    };
    float RS[3][3];
    matMul3(R, S, RS);
    float Rt[3][3];
    transpose3(R, Rt);
    matMul3(RS, Rt, Sigma);
}

// Mirrors forward_common.slangh's wangHash/hashToUnitFloat exactly -- uint32_t
// arithmetic wraps mod 2^32 identically in C++ and Slang, so this is a
// bit-for-bit port, not an approximation.
uint32_t wangHash(uint32_t seed)
{
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed = seed ^ (seed >> 4u);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15u);
    return seed;
}

float hashToUnitFloat(uint32_t index)
{
    // >> 8u leaves a 24-bit value in [0, 2^24); 1/16777216 == 1/2^24 rescales
    // that to [0, 1) -- must match forward_common.slangh's hashToUnitFloat
    // bit-for-bit.
    return static_cast<float>(wangHash(index) >> 8u) * (1.0f / 16777216.0f);
}

// Mirrors forward_project.slang's LOD decision exactly (see its
// comment for the formula's rationale).
struct LODTestParams
{
    float nearDistance;
    float farDistance;
    float maxDropDistance;
    float maxDropDensity;
    float densityReference;
};

// Mirrors forward_project.slang's adaptive degree selection.
uint32_t expectedAdaptiveDegree(uint32_t globalDegree, float degradeFraction)
{
    int stepsLost = static_cast<int>(
        std::floor(degradeFraction * static_cast<float>(kShDegree + 1)));
    int adaptive = std::clamp(static_cast<int>(kShDegree) - stepsLost, 0,
                              static_cast<int>(kShDegree));
    return std::min(globalDegree, static_cast<uint32_t>(adaptive));
}

float expectedDropFraction(float depth, float previousInstances, const LODTestParams & lod)
{
    float distanceFraction =
        std::clamp((depth - lod.nearDistance) / std::max(lod.farDistance - lod.nearDistance, 1e-4f),
                   0.0f, 1.0f);
    float densityFraction =
        std::clamp(previousInstances / std::max(lod.densityReference, 1.0f), 0.0f, 1.0f);
    return std::max(distanceFraction * lod.maxDropDistance, densityFraction * lod.maxDropDensity);
}

// Mirrors the min/max tile computation duplicated in forward_project.slang
// (to decide tileCount) and tile_binning.slang (to emit the actual keys).
// Returns false if the footprint misses the tile grid entirely.
bool tileRangeFor(const glm::vec2 & center, float radius, uint32_t tilesX, uint32_t tilesY,
                  int & minTileX, int & minTileY, int & maxTileX, int & maxTileY)
{
    minTileX = std::max(static_cast<int>(std::floor((center.x - radius) / kTileSize)), 0);
    minTileY = std::max(static_cast<int>(std::floor((center.y - radius) / kTileSize)), 0);
    maxTileX = std::min(static_cast<int>(std::floor((center.x + radius) / kTileSize)),
                        static_cast<int>(tilesX) - 1);
    maxTileY = std::min(static_cast<int>(std::floor((center.y + radius) / kTileSize)),
                        static_cast<int>(tilesY) - 1);
    return minTileX <= maxTileX && minTileY <= maxTileY;
}

// `shDegree`/`cameraPosition` feed the SH evaluation. Defaults match
// the fixtures that predate it: full degree, camera at the world origin
// (this file's camera uses an identity cameraToWorld -- see the file-level
// comment).
ProjectedGaussian referenceProject(const SyntheticGaussian & g, uint32_t tilesX, uint32_t tilesY,
                                   uint32_t shDegree = kShDegree,
                                   const glm::vec3 & cameraPosition = glm::vec3(0.0f),
                                   // Screen-space size cull threshold in px.
                                   // Defaults to 0 (disabled), matching
                                   // LODParams::minScreenRadius -- the cull is
                                   // opt-in now, so every fixture that does not
                                   // ask for it must not get it.
                                   float minScreenRadius = 0.0f,
                                   // SABERCAT_RENDER_FLAG_ANTIALIASED.
                                   // Off by default, matching CameraUniforms'
                                   // default renderFlags == 0 -- vanilla 3DGS
                                   // scenes train without the compensation.
                                   bool antialiased = false)
{
    ProjectedGaussian r{};
    r.center = glm::vec2(0.0f);
    r.depth = 0.0f;
    r.radius = 0.0f;
    r.conic = glm::vec3(0.0f);
    r.alpha = 0.0f;
    r.color = glm::vec3(0.0f);
    r.tileCount = 0;

    // worldToCamera is identity for this fixture's camera, so localMean ==
    // worldMean (see the file-level comment on why that's deliberate).
    glm::vec3 localMean = g.position;
    float depth = -localMean.z;
    if (depth <= kMinDepth)
    {
        return r;
    }

    glm::vec2 center(kFx * localMean.x / depth + kCx, kCy - kFy * localMean.y / depth);

    float covWorld[3][3];
    covarianceFromScaleRotation(g.scale, g.rotation, covWorld);
    // covarianceCamera == covarianceWorld: W (worldToCamera's upper 3x3) is
    // identity for this fixture.

    // Same frustum clamp as forward_project.slang: only J uses the clamped point.
    const float limX = kFrustumClamp * (0.5f * static_cast<float>(kImageWidth)) / kFx;
    const float limY = kFrustumClamp * (0.5f * static_cast<float>(kImageHeight)) / kFy;
    const float jx = std::clamp(localMean.x / depth, -limX, limX) * depth;
    const float jy = std::clamp(localMean.y / depth, -limY, limY) * depth;
    float invDepth = 1.0f / depth;
    float invDepthSq = invDepth * invDepth;
    // J is 2x3: row 0 = (fx/d, 0, fx*x/d^2), row 1 = (0, -fy/d, -fy*y/d^2).
    float J[2][3] = {
        {kFx * invDepth, 0.0f, kFx * jx * invDepthSq},
        {0.0f, -kFy * invDepth, -kFy * jy * invDepthSq},
    };

    // cov2d = J * covWorld * J^T (2x2).
    float JC[2][3];
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            float s = 0.0f;
            for (int k = 0; k < 3; ++k)
            {
                s += J[i][k] * covWorld[k][j];
            }
            JC[i][j] = s;
        }
    }
    float cov2d[2][2];
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            float s = 0.0f;
            for (int k = 0; k < 3; ++k)
            {
                s += JC[i][k] * J[j][k];
            }
            cov2d[i][j] = s;
        }
    }

    // Captured before dilation, for the opacity compensation below -- see
    // forward_project.slang for why an uncompensated dilation destroys the
    // image.
    float detBeforeDilation = cov2d[0][0] * cov2d[1][1] - cov2d[0][1] * cov2d[1][0];

    cov2d[0][0] += kCovarianceDilation;
    cov2d[1][1] += kCovarianceDilation;

    float determinant = cov2d[0][0] * cov2d[1][1] - cov2d[0][1] * cov2d[1][0];
    if (determinant <= 0.0f)
    {
        return r;
    }

    float invDet = 1.0f / determinant;
    glm::vec3 conic(cov2d[1][1] * invDet, -cov2d[0][1] * invDet, cov2d[0][0] * invDet);

    float midpoint = 0.5f * (cov2d[0][0] + cov2d[1][1]);
    float offset = std::sqrt(std::max(midpoint * midpoint - determinant, 0.0f));
    float maxEigenvalue = midpoint + offset;
    float minEigenvalue = midpoint - offset;
    if (minEigenvalue <= 0.0f)
    {
        return r;
    }
    float radius = kRadiusSigmas * std::sqrt(maxEigenvalue);

    if (minScreenRadius > 0.0f && radius < minScreenRadius)
    {
        return r;
    }

    if (center.x + radius < 0.0f || center.x - radius > static_cast<float>(kImageWidth)
        || center.y + radius < 0.0f || center.y - radius > static_cast<float>(kImageHeight))
    {
        return r;
    }

    int minTileX, minTileY, maxTileX, maxTileY;
    if (!tileRangeFor(center, radius, tilesX, tilesY, minTileX, minTileY, maxTileX, maxTileY))
    {
        return r;
    }

    r.center = center;
    r.depth = depth;
    r.radius = radius;
    // Same opacity compensation the shader applies for the covariance
    // dilation -- reimplemented here from the formula rather than copied, so
    // this stays an independent check (the established pattern in this file).
    // Preserves integrated contribution instead of peak, which is what stops
    // sub-pixel Gaussians from over-inking the image.
    float compensation =
        antialiased ? std::sqrt(std::max(detBeforeDilation, 0.0f) / determinant) : 1.0f;
    r.alpha = g.opacity * compensation;
    glm::vec3 coefficients[kShCoefficientsPerGaussian];
    packSHCoefficients(g, coefficients);
    r.color = referenceEvaluateSH(shDegree, coefficients,
                                  glm::normalize(g.position - cameraPosition));
    r.tileCount = static_cast<uint32_t>((maxTileX - minTileX + 1) * (maxTileY - minTileY + 1));
    r.conic = conic;
    return r;
}

bool checkProjected(const char * label, const ProjectedGaussian & gpu,
                    const ProjectedGaussian & expected)
{
    bool ok = true;
    auto fail = [&](const char * field)
    {
        std::cerr << label << ": " << field << " mismatch\n";
        ok = false;
    };

    if (!approxEqualRelative(gpu.center.x, expected.center.x)
        || !approxEqualRelative(gpu.center.y, expected.center.y))
    {
        fail("center");
        std::cerr << "  gpu=(" << gpu.center.x << ", " << gpu.center.y << ") expected=("
                  << expected.center.x << ", " << expected.center.y << ")\n";
    }
    if (!approxEqualRelative(gpu.depth, expected.depth))
    {
        fail("depth");
        std::cerr << "  gpu=" << gpu.depth << " expected=" << expected.depth << "\n";
    }
    if (!approxEqualRelative(gpu.radius, expected.radius))
    {
        fail("radius");
        std::cerr << "  gpu=" << gpu.radius << " expected=" << expected.radius << "\n";
    }
    if (!approxEqualRelative(gpu.conic.x, expected.conic.x)
        || !approxEqualRelative(gpu.conic.y, expected.conic.y)
        || !approxEqualRelative(gpu.conic.z, expected.conic.z))
    {
        fail("conic");
        std::cerr << "  gpu=(" << gpu.conic.x << ", " << gpu.conic.y << ", " << gpu.conic.z
                  << ") expected=(" << expected.conic.x << ", " << expected.conic.y << ", "
                  << expected.conic.z << ")\n";
    }
    if (!approxEqual(gpu.alpha, expected.alpha))
    {
        fail("alpha");
    }
    if (!approxEqualRelative(gpu.color.x, expected.color.x)
        || !approxEqualRelative(gpu.color.y, expected.color.y)
        || !approxEqualRelative(gpu.color.z, expected.color.z))
    {
        fail("color");
    }
    if (gpu.tileCount != expected.tileCount)
    {
        fail("tileCount");
        std::cerr << "  gpu=" << gpu.tileCount << " expected=" << expected.tileCount << "\n";
    }
    return ok;
}

// One (tileID, depth-bits, Gaussian index) tile-binning instance -- reused
// as the Stage 3 oracle (in emission order) and, stably sorted by (tile,
// depth), as the Stage 4 radix-sort oracle.
struct Instance
{
    uint32_t tile;
    uint32_t depth;
    uint32_t value;
};

std::vector<Instance> generateExpectedInstances(const std::vector<ProjectedGaussian> & projected,
                                                uint32_t tilesX, uint32_t tilesY)
{
    std::vector<Instance> instances;
    for (uint32_t i = 0; i < projected.size(); ++i)
    {
        const ProjectedGaussian & expected = projected[i];
        if (expected.tileCount == 0)
        {
            continue;
        }
        int minTileX, minTileY, maxTileX, maxTileY;
        tileRangeFor(expected.center, expected.radius, tilesX, tilesY, minTileX, minTileY,
                    maxTileX, maxTileY);

        uint32_t depthBits;
        std::memcpy(&depthBits, &expected.depth, sizeof(depthBits));

        for (int ty = minTileY; ty <= maxTileY; ++ty)
        {
            for (int tx = minTileX; tx <= maxTileX; ++tx)
            {
                uint32_t tileId = static_cast<uint32_t>(ty) * tilesX + static_cast<uint32_t>(tx);
                instances.push_back({tileId, depthBits, i});
            }
        }
    }
    return instances;
}

// ---------------------------------------------------------------------------
// Scene-level verification. Everything here is generated at run time from
// fixed seeds and checked against a double-precision CPU renderer that shares
// no code with the GPU path (only the numeric constants in render_config.h).
// The camera is general: rotated, translated, fx != fy, off-center principal
// point. Tile membership and the sort/binning/range checks are driven by the
// GPU's own projected records, so those stages are compared exactly and the
// projection is compared separately within the tolerances below.
// ---------------------------------------------------------------------------

// Tolerances. Each one bounds fp32 rounding of the quantity it guards; the
// worst error actually measured is printed for every case.
//   - center: a few fp32 ulps of an image coordinate. The widest frame here is
//     4096 px, where one ulp is 4.9e-4 px; the measured worst is 3.4e-4 px.
//   - radius: 3 * sqrt(mid + offset) with offset = sqrt(mid^2 - det). For a
//     near-isotropic footprint mid^2 - det cancels, so fp32 error in the
//     eigenvalues reaches sqrt(eps) ~ 3e-4 relative; measured up to ~1.3e-4
//     relative across the probes and scenes. 1e-3 keeps a few-times margin.
//   - conic: fp32 subtraction in cov2d = J S J^T + dilation, which cancels
//     when a footprint is very elongated; measured relative to the largest
//     conic term of the same Gaussian.
//   - alpha, color: fp32 rounding of opacity * exp() and of the SH polynomial
//     (~30 terms of O(1) magnitude, so ~1e-6 absolute).
//   - image color: each blend step rounds at ~1e-7, but the transmittance
//     early-out (T < 1e-4) is a hard threshold. A pixel whose T lands within
//     rounding of it can include or skip one more Gaussian, which moves its
//     color by T * alpha * color <= 1e-4 per Gaussian. 5e-4 covers a few such
//     pixels in a dense scene; the summary line prints the measured worst.
constexpr double kCenterTolerancePx = 1e-3;
constexpr double kRadiusTolerance = 1e-3; // relative
constexpr double kConicTolerance = 1e-4;
constexpr double kAlphaTolerance = 1e-5;
constexpr double kColorTolerance = 1e-5;
constexpr double kDepthTolerance = 1e-5;
constexpr double kImageColorTolerance = 5e-4;
constexpr double kImageDepthTolerance = 1e-4;
//   - depth sum D = sum T*alpha*z: a sum of z-weighted terms, so its error is
//     bounded against the scene's largest depth rather than against D itself
//     (D can be near zero where a pixel is empty). Per-step fp32 rounding is
//     tiny; the T < 1e-4 early-out moves D by at most 1e-4 * z per Gaussian, the
//     same pixel-level effect as the color tolerance above.
constexpr double kImageDepthSumTolerance = 5e-4; // relative to the scene's max reference depth

std::string formatError(double value)
{
    std::ostringstream out;
    out << std::scientific << std::setprecision(2) << value;
    return out.str();
}

// splitmix64: a fixed, compiler-independent generator. The standard library's
// distributions are not portable, so the same seed could build a different
// scene on another toolchain.
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

    // Uniform in [lo, hi).
    double uniform(double lo, double hi)
    {
        const double unit = static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0);
        return lo + (hi - lo) * unit;
    }

    // Log-uniform in [lo, hi): even coverage across decades of depth and scale.
    double logUniform(double lo, double hi)
    {
        return std::exp(uniform(std::log(lo), std::log(hi)));
    }

  private:
    uint64_t m_state;
};

// A pinhole camera plus the exact values the GPU receives. The reference reads
// worldToCamera and cameraPosition as stored (float), so both sides evaluate
// the same numbers.
struct CpuCamera
{
    glm::mat4 worldToCamera{1.0f};
    glm::vec3 cameraPosition{0.0f}; // world space; the SH view direction starts here
    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t shDegree = kShDegree;
    bool antialiased = false;

    uint32_t tilesX() const
    {
        return dispatchGroups(width, kTileSize);
    }
    uint32_t tilesY() const
    {
        return dispatchGroups(height, kTileSize);
    }
};

CpuCamera makeCamera(const glm::mat4 & worldToCamera, float fx, float fy, float cx, float cy,
                     uint32_t width, uint32_t height)
{
    CpuCamera cam;
    cam.worldToCamera = worldToCamera;
    cam.cameraPosition = glm::vec3(glm::inverse(worldToCamera) * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
    cam.fx = fx;
    cam.fy = fy;
    cam.cx = cx;
    cam.cy = cy;
    cam.width = width;
    cam.height = height;
    return cam;
}

// Rigid world-to-camera transform: rotate by `angle` about `axis`, then translate.
glm::mat4 rigidWorldToCamera(const glm::vec3 & axis, float angle, const glm::vec3 & translation)
{
    const glm::mat4 rotation = glm::mat4_cast(glm::angleAxis(angle, glm::normalize(axis)));
    return glm::translate(glm::mat4(1.0f), translation) * rotation;
}

glm::vec4 randomUnitQuaternion(SplitMix & rng)
{
    for (;;)
    {
        const double x = rng.uniform(-1.0, 1.0);
        const double y = rng.uniform(-1.0, 1.0);
        const double z = rng.uniform(-1.0, 1.0);
        const double w = rng.uniform(-1.0, 1.0);
        const double norm2 = x * x + y * y + z * z + w * w;
        if (norm2 > 0.04 && norm2 <= 1.0)
        {
            const double norm = std::sqrt(norm2);
            return glm::vec4(static_cast<float>(x / norm), static_cast<float>(y / norm),
                             static_cast<float>(z / norm), static_cast<float>(w / norm));
        }
    }
}

// Scatters `count` Gaussians so each lands on a random pixel of `cam`. Depth is
// log-uniform; the world scale is chosen so the projected sigma (fx * s / depth)
// is log-uniform over [sigmaMinPx, sigmaMaxPx]. `randomShape` adds random
// rotations and per-axis scale factors; otherwise each Gaussian is isotropic.
// Pixels are drawn over a 10% margin so frustum clipping is exercised too.
std::vector<SyntheticGaussian> makeScatteredScene(SplitMix & rng, const CpuCamera & cam,
                                                  uint32_t count, double depthMin,
                                                  double depthMax, double sigmaMinPx,
                                                  double sigmaMaxPx, bool randomShape)
{
    const glm::mat4 cameraToWorld = glm::inverse(cam.worldToCamera);
    std::vector<SyntheticGaussian> scene(count);
    for (SyntheticGaussian & g : scene)
    {
        const double depth = rng.logUniform(depthMin, depthMax);
        const double u = rng.uniform(-0.1 * cam.width, 1.1 * cam.width);
        const double v = rng.uniform(-0.1 * cam.height, 1.1 * cam.height);
        const glm::vec3 local(static_cast<float>((u - cam.cx) * depth / cam.fx),
                              static_cast<float>((cam.cy - v) * depth / cam.fy),
                              static_cast<float>(-depth));
        g.position = glm::vec3(cameraToWorld * glm::vec4(local, 1.0f));

        const double sigma = rng.logUniform(sigmaMinPx, sigmaMaxPx);
        const double base = sigma * depth / cam.fx;
        g.scale = glm::vec3(static_cast<float>(base));
        g.rotation = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        if (randomShape)
        {
            for (int axis = 0; axis < 3; ++axis)
            {
                g.scale[axis] = static_cast<float>(base * rng.logUniform(0.4, 2.5));
            }
            g.rotation = randomUnitQuaternion(rng);
        }

        g.opacity = static_cast<float>(rng.uniform(0.05, 0.95));
        g.sh0 = glm::vec3(static_cast<float>(rng.uniform(-0.8, 0.8)),
                          static_cast<float>(rng.uniform(-0.8, 0.8)),
                          static_cast<float>(rng.uniform(-0.8, 0.8)));
        for (int k = 0; k < kShCoefficientsPerGaussian - 1; ++k)
        {
            g.shRest[k] = glm::vec3(static_cast<float>(rng.uniform(-0.12, 0.12)),
                                    static_cast<float>(rng.uniform(-0.12, 0.12)),
                                    static_cast<float>(rng.uniform(-0.12, 0.12)));
        }
    }
    return scene;
}

// -- Double-precision reference. Written from the shader's formulas, not
// copied from them: every intermediate is a double, and nothing is shared
// with the GPU path except the numeric constants. --

// One Gaussian as the reference projects it. `visible` is false wherever the
// GPU must cull (tileCount == 0); fields are only meaningful when visible.
struct CpuSplat
{
    bool visible = false;
    double center[2] = {0.0, 0.0};
    double depth = 0.0;
    double radius = 0.0;
    double conic[3] = {0.0, 0.0, 0.0}; // (a, b, c) of [[a, b], [b, c]]
    double alpha = 0.0;
    double color[3] = {0.0, 0.0, 0.0};
    uint32_t tileCount = 0;
    int tileMin[2] = {0, 0};
    int tileMax[2] = {-1, -1};
};

// Sigma = R diag(s^2) R^T, with the quaternion used as stored (no
// normalization), as quaternionToMatrix does.
void cpuCovariance(const glm::vec3 & scale, const glm::vec4 & q, double sigma[3][3])
{
    const double x = q.x, y = q.y, z = q.z, w = q.w;
    const double R[3][3] = {
        {1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)},
        {2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)},
        {2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)},
    };
    const double s2[3] = {static_cast<double>(scale.x) * scale.x,
                          static_cast<double>(scale.y) * scale.y,
                          static_cast<double>(scale.z) * scale.z};
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            double s = 0.0;
            for (int k = 0; k < 3; ++k)
            {
                s += R[i][k] * s2[k] * R[j][k];
            }
            sigma[i][j] = s;
        }
    }
}

// Mirrors evaluateSH in forward_common.slangh, including the band-ascending
// coefficient order and the +0.5 / [0, 1] tail, in double precision.
void cpuEvaluateSH(uint32_t degree, const glm::vec3 * c, const double dir[3], double out[3])
{
    const double x = dir[0], y = dir[1], z = dir[2];
    const double xx = x * x, yy = y * y, zz = z * z;
    const double xy = x * y, yz = y * z, xz = x * z;
    const double c0 = kShC0, c1 = kShC1;
    const double c20 = kShC2_0, c21 = kShC2_1, c22 = kShC2_2, c23 = kShC2_3, c24 = kShC2_4;
    const double c30 = kShC3_0, c31 = kShC3_1, c32 = kShC3_2, c33 = kShC3_3;
    const double c34 = kShC3_4, c35 = kShC3_5, c36 = kShC3_6;
    for (int ch = 0; ch < 3; ++ch)
    {
        auto k = [&](int i)
        {
            return static_cast<double>(c[i][ch]);
        };
        double v = c0 * k(0);
        if (degree >= 1)
        {
            v += c1 * (-y * k(1) + z * k(2) - x * k(3));
        }
        if (degree >= 2)
        {
            v += c20 * xy * k(4) + c21 * yz * k(5) + c22 * (2.0 * zz - xx - yy) * k(6)
                 + c23 * xz * k(7) + c24 * (xx - yy) * k(8);
        }
        if (degree >= 3)
        {
            v += c30 * y * (3.0 * xx - yy) * k(9) + c31 * xy * z * k(10)
                 + c32 * y * (4.0 * zz - xx - yy) * k(11)
                 + c33 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * k(12)
                 + c34 * x * (4.0 * zz - xx - yy) * k(13) + c35 * z * (xx - yy) * k(14)
                 + c36 * x * (xx - 3.0 * yy) * k(15);
        }
        out[ch] = std::clamp(v + 0.5, 0.0, 1.0);
    }
}

// Projects one Gaussian with the same frustum clamp, dilation, footprint and
// tile rules as forward_project.slang, but in double precision. Every test is
// written NaN-safe: a value that is not a number is culled, never rendered.
CpuSplat cpuProject(const SyntheticGaussian & g, const CpuCamera & cam)
{
    CpuSplat out;
    const glm::mat4 & m = cam.worldToCamera;

    // Column-major: row i of worldToCamera * (p, 1) is sum_j m[j][i] p_j + m[3][i].
    double local[3];
    for (int i = 0; i < 3; ++i)
    {
        local[i] = static_cast<double>(m[3][i]);
        for (int j = 0; j < 3; ++j)
        {
            local[i] += static_cast<double>(m[j][i]) * static_cast<double>(g.position[j]);
        }
    }

    const double depth = -local[2];
    if (!(depth > static_cast<double>(kMinDepth)))
    {
        return out;
    }

    const double fx = cam.fx, fy = cam.fy;
    const double center[2] = {fx * local[0] / depth + cam.cx, cam.cy - fy * local[1] / depth};

    double W[3][3];
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            W[i][j] = static_cast<double>(m[j][i]);
        }
    }
    double sigmaWorld[3][3];
    cpuCovariance(g.scale, g.rotation, sigmaWorld);
    double WS[3][3], sigmaCam[3][3];
    for (int i = 0; i < 3; ++i)
    {
        for (int l = 0; l < 3; ++l)
        {
            double s = 0.0;
            for (int k = 0; k < 3; ++k)
            {
                s += W[i][k] * sigmaWorld[k][l];
            }
            WS[i][l] = s;
        }
    }
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            double s = 0.0;
            for (int l = 0; l < 3; ++l)
            {
                s += WS[i][l] * W[j][l];
            }
            sigmaCam[i][j] = s;
        }
    }

    // Projection Jacobian with x/z and y/z clamped to the widened frustum (the
    // center itself is projected unclamped).
    const double limX = static_cast<double>(kFrustumClamp) * (0.5 * cam.width) / fx;
    const double limY = static_cast<double>(kFrustumClamp) * (0.5 * cam.height) / fy;
    const double tx = local[0] / depth, ty = local[1] / depth;
    const double jx = std::clamp(tx, -limX, limX) * depth;
    const double jy = std::clamp(ty, -limY, limY) * depth;
    const double inv = 1.0 / depth, invSq = inv * inv;
    const double J[2][3] = {
        {fx * inv, 0.0, fx * jx * invSq},
        {0.0, -fy * inv, -fy * jy * invSq},
    };

    double JS[2][3], cov[2][2];
    for (int i = 0; i < 2; ++i)
    {
        for (int l = 0; l < 3; ++l)
        {
            double s = 0.0;
            for (int k = 0; k < 3; ++k)
            {
                s += J[i][k] * sigmaCam[k][l];
            }
            JS[i][l] = s;
        }
    }
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            double s = 0.0;
            for (int l = 0; l < 3; ++l)
            {
                s += JS[i][l] * J[j][l];
            }
            cov[i][j] = s;
        }
    }

    const double detBeforeDilation = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
    cov[0][0] += static_cast<double>(kCovarianceDilation);
    cov[1][1] += static_cast<double>(kCovarianceDilation);
    const double det = cov[0][0] * cov[1][1] - cov[0][1] * cov[1][0];
    if (!(det > 0.0))
    {
        return out;
    }

    const double invDet = 1.0 / det;
    const double conic[3] = {cov[1][1] * invDet, -cov[0][1] * invDet, cov[0][0] * invDet};

    const double mid = 0.5 * (cov[0][0] + cov[1][1]);
    const double offset = std::sqrt(std::max(mid * mid - det, 0.0));
    const double minEig = mid - offset;
    if (!(minEig > 0.0))
    {
        return out;
    }
    const double radius = static_cast<double>(kRadiusSigmas) * std::sqrt(mid + offset);

    // Recorded before any cull, so a culled Gaussian's footprint can still be measured.
    out.center[0] = center[0];
    out.center[1] = center[1];
    out.radius = radius;

    if (!std::isfinite(center[0]) || !std::isfinite(center[1]) || !std::isfinite(radius)
        || !std::isfinite(conic[0]) || !std::isfinite(conic[1]) || !std::isfinite(conic[2]))
    {
        return out;
    }

    if (center[0] + radius < 0.0 || center[0] - radius > cam.width || center[1] + radius < 0.0
        || center[1] - radius > cam.height)
    {
        return out;
    }

    const int tileSize = static_cast<int>(kTileSize);
    const int tilesX = static_cast<int>(cam.tilesX()), tilesY = static_cast<int>(cam.tilesY());
    const int minTileX = std::max(static_cast<int>(std::floor((center[0] - radius) / tileSize)), 0);
    const int minTileY = std::max(static_cast<int>(std::floor((center[1] - radius) / tileSize)), 0);
    const int maxTileX =
        std::min(static_cast<int>(std::floor((center[0] + radius) / tileSize)), tilesX - 1);
    const int maxTileY =
        std::min(static_cast<int>(std::floor((center[1] + radius) / tileSize)), tilesY - 1);
    if (minTileX > maxTileX || minTileY > maxTileY)
    {
        return out;
    }

    out.visible = true;
    out.depth = depth;
    out.conic[0] = conic[0];
    out.conic[1] = conic[1];
    out.conic[2] = conic[2];
    out.tileMin[0] = minTileX;
    out.tileMin[1] = minTileY;
    out.tileMax[0] = maxTileX;
    out.tileMax[1] = maxTileY;
    out.tileCount = static_cast<uint32_t>((maxTileX - minTileX + 1) * (maxTileY - minTileY + 1));

    const double compensation =
        cam.antialiased ? std::sqrt(std::max(detBeforeDilation, 0.0) / det) : 1.0;
    out.alpha = static_cast<double>(g.opacity) * compensation;

    glm::vec3 coefficients[kShCoefficientsPerGaussian];
    packSHCoefficients(g, coefficients);
    const double mean[3] = {g.position.x, g.position.y, g.position.z};
    const double camPos[3] = {cam.cameraPosition.x, cam.cameraPosition.y, cam.cameraPosition.z};
    double dir[3] = {mean[0] - camPos[0], mean[1] - camPos[1], mean[2] - camPos[2]};
    const double len = std::sqrt(dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]);
    for (double & d : dir)
    {
        d /= len;
    }
    cpuEvaluateSH(cam.shDegree, coefficients, dir, out.color);
    return out;
}

// Double-precision image: per tile, the Gaussians whose footprint covers it,
// in (depth, index) order; per pixel, the same front-to-back blend and the
// same early-outs the shader uses.
struct CpuImage
{
    std::vector<double> rgba; // 4 per pixel, row-major
    std::vector<double> depth;
    std::vector<double> depthSum; // sum of T*alpha*z per pixel: the depth output's D
};

CpuImage cpuRender(const std::vector<CpuSplat> & splats, const CpuCamera & cam)
{
    const uint32_t tilesX = cam.tilesX(), tilesY = cam.tilesY();
    std::vector<std::vector<uint32_t>> tiles(static_cast<size_t>(tilesX) * tilesY);
    for (uint32_t i = 0; i < splats.size(); ++i)
    {
        const CpuSplat & s = splats[i];
        if (!s.visible)
        {
            continue;
        }
        for (int ty = s.tileMin[1]; ty <= s.tileMax[1]; ++ty)
        {
            for (int tx = s.tileMin[0]; tx <= s.tileMax[0]; ++tx)
            {
                tiles[static_cast<size_t>(ty) * tilesX + tx].push_back(i);
            }
        }
    }
    for (std::vector<uint32_t> & list : tiles)
    {
        std::stable_sort(list.begin(), list.end(), [&](uint32_t a, uint32_t b)
                         { return splats[a].depth < splats[b].depth; });
    }

    CpuImage img;
    img.rgba.assign(static_cast<size_t>(cam.width) * cam.height * 4, 0.0);
    img.depth.assign(static_cast<size_t>(cam.width) * cam.height, 0.0);
    img.depthSum.assign(static_cast<size_t>(cam.width) * cam.height, 0.0);
    for (uint32_t y = 0; y < cam.height; ++y)
    {
        for (uint32_t x = 0; x < cam.width; ++x)
        {
            const std::vector<uint32_t> & list = tiles[(y / kTileSize) * tilesX + x / kTileSize];
            double acc[3] = {0.0, 0.0, 0.0};
            double depthSum = 0.0;
            double transmittance = 1.0;
            double front = 0.0;
            bool hasFront = false;
            for (uint32_t i : list)
            {
                const CpuSplat & s = splats[i];
                const double dx = static_cast<double>(x) - s.center[0];
                const double dy = static_cast<double>(y) - s.center[1];
                const double exponent = s.conic[0] * dx * dx + 2.0 * s.conic[1] * dx * dy
                                        + s.conic[2] * dy * dy;
                if (exponent < 0.0)
                {
                    continue;
                }
                const double alpha = std::min(s.alpha * std::exp(-0.5 * exponent),
                                              static_cast<double>(kMaxAlpha));
                if (alpha < 1.0 / 255.0)
                {
                    continue;
                }
                if (!hasFront)
                {
                    front = s.depth;
                    hasFront = true;
                }
                for (int ch = 0; ch < 3; ++ch)
                {
                    acc[ch] += transmittance * alpha * s.color[ch];
                }
                depthSum += transmittance * alpha * s.depth;
                transmittance *= 1.0 - alpha;
                if (transmittance < 1e-4)
                {
                    break;
                }
            }
            const size_t p = static_cast<size_t>(y) * cam.width + x;
            img.rgba[4 * p + 0] = acc[0];
            img.rgba[4 * p + 1] = acc[1];
            img.rgba[4 * p + 2] = acc[2];
            img.rgba[4 * p + 3] = 1.0 - transmittance;
            img.depth[p] = front;
            img.depthSum[p] = depthSum;
        }
    }
    return img;
}

void uploadScene(GaussianCloud & cloud, const std::vector<SyntheticGaussian> & scene)
{
    cloud.initialize(scene.size());
    for (size_t i = 0; i < scene.size(); ++i)
    {
        const SyntheticGaussian & g = scene[i];
        cloud.positions()[i] = glm::vec4(g.position, 0.0f);
        cloud.scales()[i] = glm::vec4(g.scale, 0.0f);
        cloud.rotations()[i] = g.rotation;
        cloud.opacities()[i] = g.opacity;
        glm::vec3 packed[kShCoefficientsPerGaussian];
        packSHCoefficients(g, packed);
        for (int k = 0; k < kShCoefficientsPerGaussian; ++k)
        {
            cloud.shCoefficients()[i * kShCoefficientsPerGaussian + k] =
                glm::vec4(packed[k], 0.0f);
        }
    }
    cloud.uploadToGPU();
}

CameraUniforms makeUniforms(const CpuCamera & cam, uint32_t gaussianCount)
{
    CameraUniforms u{};
    u.worldToCamera = cam.worldToCamera;
    u.fx = cam.fx;
    u.fy = cam.fy;
    u.cx = cam.cx;
    u.cy = cam.cy;
    u.imageWidth = cam.width;
    u.imageHeight = cam.height;
    u.tilesX = cam.tilesX();
    u.tilesY = cam.tilesY();
    u.gaussianCount = gaussianCount;
    u.shDegree = cam.shDegree;
    u.renderFlags = cam.antialiased ? kRenderFlagAntialiased : 0u;
    u.cameraPositionWorld = glm::vec4(cam.cameraPosition, 0.0f);
    return u;
}

struct SceneRun
{
    std::vector<ProjectedGaussian> projected;
    std::vector<glm::vec4> image;
    std::vector<float> depth;
    uint32_t instanceCount = 0;
};

struct ProjectionWorst
{
    double center = 0.0;
    double radius = 0.0;
    double conic = 0.0;
    double alpha = 0.0;
    double color = 0.0;
    double depth = 0.0;
    uint32_t mismatches = 0;
};

// Compares one GPU record with the reference. tileCount must match exactly
// (visibility included); the fields are compared within tolerance.
void compareRecord(const ProjectedGaussian & gpu, const CpuSplat & ref, uint32_t index,
                   ProjectionWorst & worst)
{
    if (gpu.tileCount != ref.tileCount)
    {
        if (++worst.mismatches <= 5)
        {
            std::cerr << "Gaussian " << index << ": tileCount gpu=" << gpu.tileCount
                      << " reference=" << ref.tileCount << "\n";
        }
        return;
    }
    if (!ref.visible)
    {
        return;
    }

    const double center = std::max(std::fabs(gpu.center.x - ref.center[0]),
                                   std::fabs(gpu.center.y - ref.center[1]));
    const double radius = std::fabs(gpu.radius - ref.radius) / ref.radius;
    const double conicScale =
        std::max({std::fabs(ref.conic[0]), std::fabs(ref.conic[1]), std::fabs(ref.conic[2]),
                  1e-30});
    const double conic = std::max({std::fabs(gpu.conic.x - ref.conic[0]),
                                   std::fabs(gpu.conic.y - ref.conic[1]),
                                   std::fabs(gpu.conic.z - ref.conic[2])})
                         / conicScale;
    const double alpha = std::fabs(gpu.alpha - ref.alpha);
    const double color = std::max({std::fabs(gpu.color.x - ref.color[0]),
                                   std::fabs(gpu.color.y - ref.color[1]),
                                   std::fabs(gpu.color.z - ref.color[2])});
    const double depth = std::fabs(gpu.depth - ref.depth) / ref.depth;

    worst.center = std::max(worst.center, center);
    worst.radius = std::max(worst.radius, radius);
    worst.conic = std::max(worst.conic, conic);
    worst.alpha = std::max(worst.alpha, alpha);
    worst.color = std::max(worst.color, color);
    worst.depth = std::max(worst.depth, depth);

    if (center > kCenterTolerancePx || radius > kRadiusTolerance || conic > kConicTolerance
        || alpha > kAlphaTolerance || color > kColorTolerance || depth > kDepthTolerance)
    {
        if (++worst.mismatches <= 5)
        {
            std::cerr << "Gaussian " << index << ": projection differs (gpu center=("
                      << gpu.center.x << ", " << gpu.center.y << ") reference=(" << ref.center[0]
                      << ", " << ref.center[1] << "), gpu radius=" << gpu.radius
                      << " reference=" << ref.radius << ")\n";
        }
    }
}

// Runs the whole chain on `scene` and checks every stage: the projection
// against the reference, emission order out of tile_binning, the radix sort,
// tile ranges, and the blended image and front depth.
SceneRun checkScene(Device & device, const std::string & label,
                    const std::vector<SyntheticGaussian> & scene, const CpuCamera & cam)
{
    GaussianCloud cloud(device);
    uploadScene(cloud, scene);
    auto views = cloud.views();
    const uint32_t count = static_cast<uint32_t>(scene.size());
    const CameraUniforms uniforms = makeUniforms(cam, count);
    const uint32_t tilesX = cam.tilesX();
    const uint32_t tileTotal = tilesX * cam.tilesY();
    // DepthCoverageFront writes the depth output (D, coverage, front, 0) on the
    // full-frame Differentiable run below; it changes nothing else checked here.
    ForwardPass pass(device, cam.width, cam.height, testShaderDir(),
                     DepthLayout::DepthCoverageFront);

    std::vector<CpuSplat> splats(count);
    uint64_t referenceInstances = 0;
    for (uint32_t i = 0; i < count; ++i)
    {
        splats[i] = cpuProject(scene[i], cam);
        referenceInstances += splats[i].tileCount;
    }

    // tile_binning alone, so its emission order can be read before the sort.
    pass.run(views, uniforms, ForwardStage::TileBinning);
    SceneRun run;
    run.instanceCount = pass.lastInstanceCount();
    run.projected.resize(count);
    pass.projected().download(run.projected.data(), count * sizeof(ProjectedGaussian));

    uint64_t recordInstances = 0;
    for (const ProjectedGaussian & r : run.projected)
    {
        recordInstances += r.tileCount;
    }
    if (run.instanceCount != referenceInstances || run.instanceCount != recordInstances)
    {
        std::cerr << label << ": instance count gpu=" << run.instanceCount
                  << " records=" << recordInstances << " reference=" << referenceInstances
                  << "\n";
        throw std::runtime_error("tile instance count mismatch");
    }

    // The GPU's records drive the binning and sort oracle, so those checks are
    // exact regardless of projection rounding (checked separately below).
    const std::vector<Instance> emitted = generateExpectedInstances(run.projected, tilesX,
                                                                    cam.tilesY());
    if (!emitted.empty())
    {
        std::vector<uint32_t> tile(emitted.size()), depthBits(emitted.size()),
            value(emitted.size());
        pass.sortedKeysTile().download(tile.data(), tile.size() * sizeof(uint32_t));
        pass.sortedKeysDepth().download(depthBits.data(), depthBits.size() * sizeof(uint32_t));
        pass.sortedValues().download(value.data(), value.size() * sizeof(uint32_t));
        for (size_t slot = 0; slot < emitted.size(); ++slot)
        {
            const Instance & e = emitted[slot];
            if (tile[slot] != e.tile || depthBits[slot] != e.depth || value[slot] != e.value)
            {
                std::cerr << label << ": tile_binning mismatch at slot " << slot << "\n";
                throw std::runtime_error("tile_binning mismatch");
            }
        }
    }

    // Expected sorted order: stable by (tile, depth), i.e. emission order breaks ties.
    std::vector<Instance> sorted = emitted;
    std::stable_sort(sorted.begin(), sorted.end(), [](const Instance & a, const Instance & b)
                     {
                         if (a.tile != b.tile)
                         {
                             return a.tile < b.tile;
                         }
                         return a.depth < b.depth;
                     });

    // Full frame, Differentiable so the depth output is written. The LOD knobs
    // are at their defaults here, so the color path is the same as a Viewer run.
    pass.run(views, uniforms, ForwardStage::Rasterize, ForwardMode::Differentiable);
    if (pass.lastInstanceCount() != run.instanceCount)
    {
        std::cerr << label << ": full run instance count differs from tile_binning run\n";
        throw std::runtime_error("instance count changed between runs");
    }
    std::vector<ProjectedGaussian> again(count);
    pass.projected().download(again.data(), count * sizeof(ProjectedGaussian));
    if (std::memcmp(again.data(), run.projected.data(), count * sizeof(ProjectedGaussian)) != 0)
    {
        std::cerr << label << ": projection is not deterministic across runs\n";
        throw std::runtime_error("projection nondeterministic");
    }

    if (!emitted.empty())
    {
        std::vector<uint32_t> tile(emitted.size()), depthBits(emitted.size()),
            value(emitted.size());
        pass.sortedKeysTile().download(tile.data(), tile.size() * sizeof(uint32_t));
        pass.sortedKeysDepth().download(depthBits.data(), depthBits.size() * sizeof(uint32_t));
        pass.sortedValues().download(value.data(), value.size() * sizeof(uint32_t));

        for (size_t i = 0; i < sorted.size(); ++i)
        {
            const Instance & e = sorted[i];
            if (tile[i] != e.tile || depthBits[i] != e.depth || value[i] != e.value)
            {
                std::cerr << label << ": radix sort mismatch at index " << i << "\n";
                throw std::runtime_error("radix sort mismatch");
            }
        }
    }

    std::vector<TileRange> gpuRanges(tileTotal);
    pass.tileRanges().download(gpuRanges.data(), gpuRanges.size() * sizeof(TileRange));
    std::vector<TileRange> expectedRanges(tileTotal, TileRange{0, 0});
    for (uint32_t i = 0; i < sorted.size();)
    {
        const uint32_t tile = sorted[i].tile;
        const uint32_t start = i;
        while (i < sorted.size() && sorted[i].tile == tile)
        {
            ++i;
        }
        expectedRanges[tile] = TileRange{start, i};
    }
    for (uint32_t t = 0; t < tileTotal; ++t)
    {
        if (gpuRanges[t].first != expectedRanges[t].first
            || gpuRanges[t].last != expectedRanges[t].last)
        {
            std::cerr << label << ": tile_ranges mismatch at tile " << t << "\n";
            throw std::runtime_error("tile_ranges mismatch");
        }
    }

    run.image = pass.downloadOutputImage();
    run.depth = pass.downloadDepthImage();

    ProjectionWorst projection;
    for (uint32_t i = 0; i < count; ++i)
    {
        compareRecord(run.projected[i], splats[i], i, projection);
    }
    if (projection.mismatches > 0)
    {
        std::cerr << label << ": " << projection.mismatches << " projection mismatches\n";
        throw std::runtime_error("projection mismatch");
    }

    const CpuImage ref = cpuRender(splats, cam);
    const uint32_t pixels = cam.width * cam.height;

    // The depth output (DepthCoverageFront): per pixel D, coverage, front, 0.
    std::vector<float> depthOutput(static_cast<size_t>(pixels) * 4);
    pass.depthBuffer().download(depthOutput.data(), depthOutput.size() * sizeof(float));
    double maxDepth = 0.0;
    for (const CpuSplat & s : splats)
    {
        if (s.visible)
        {
            maxDepth = std::max(maxDepth, static_cast<double>(s.depth));
        }
    }
    double worstDepthSum = 0.0; // absolute, reported relative to maxDepth
    double worstCoverage = 0.0;
    uint32_t depthBadPixels = 0;
    for (uint32_t p = 0; p < pixels; ++p)
    {
        const float * out = &depthOutput[4 * static_cast<size_t>(p)];
        const double sumError = std::fabs(static_cast<double>(out[0]) - ref.depthSum[p]);
        const double coverageError = std::fabs(static_cast<double>(out[1]) - ref.rgba[4 * p + 3]);
        worstDepthSum = std::max(worstDepthSum, sumError);
        worstCoverage = std::max(worstCoverage, coverageError);
        // The front depth is the same frontDepth value the color check reads.
        if (sumError > kImageDepthSumTolerance * maxDepth || coverageError > kImageColorTolerance
            || out[2] != run.depth[p] || out[3] != 0.0f)
        {
            ++depthBadPixels;
        }
    }
    if (depthBadPixels > 0)
    {
        std::cerr << label << ": depth output " << depthBadPixels << " pixels outside tolerance\n";
        throw std::runtime_error("depth output mismatch");
    }

    double worstColor = 0.0;
    double worstDepth = 0.0;
    bool finite = true;
    uint32_t badPixels = 0;
    uint32_t covered = 0;
    for (uint32_t p = 0; p < pixels; ++p)
    {
        if (run.image[p].a > 1.0f / 255.0f)
        {
            ++covered;
        }
        double pixelColor = 0.0;
        for (int ch = 0; ch < 4; ++ch)
        {
            const float gpuValue = run.image[p][ch];
            if (!std::isfinite(gpuValue))
            {
                finite = false;
            }
            pixelColor = std::max(pixelColor, std::fabs(gpuValue - ref.rgba[4 * p + ch]));
        }
        const double pixelDepth = std::fabs(run.depth[p] - ref.depth[p]);
        if (!std::isfinite(run.depth[p]))
        {
            finite = false;
        }
        worstColor = std::max(worstColor, pixelColor);
        worstDepth = std::max(worstDepth, pixelDepth);
        if (pixelColor > kImageColorTolerance || pixelDepth > kImageDepthTolerance)
        {
            if (++badPixels <= 5)
            {
                std::cerr << label << ": pixel " << p % cam.width << "," << p / cam.width
                          << " gpu=(" << run.image[p].r << ", " << run.image[p].g << ", "
                          << run.image[p].b << ", " << run.image[p].a << ") reference=("
                          << ref.rgba[4 * p] << ", " << ref.rgba[4 * p + 1] << ", "
                          << ref.rgba[4 * p + 2] << ", " << ref.rgba[4 * p + 3] << ")\n";
            }
        }
    }
    if (!finite || badPixels > 0)
    {
        std::cerr << label << ": image " << (finite ? "" : "not finite, ") << badPixels
                  << " pixels outside tolerance\n";
        throw std::runtime_error("image mismatch");
    }
    // A scene with visible Gaussians must cover something, or the image check
    // above could pass with both sides empty.
    if (covered == 0 && referenceInstances > 0)
    {
        std::cerr << label << ": visible Gaussians but an empty image\n";
        throw std::runtime_error("empty image");
    }

    std::cout << "[" << label << "] " << count << " Gaussians, " << run.instanceCount
              << " tile instances, " << tileTotal << " tiles\n"
              << "    tile_binning emission, radix sort and tile ranges: exact\n"
              << "    projection worst: center " << formatError(projection.center) << " px, radius "
              << formatError(projection.radius) << " (relative), conic " << formatError(projection.conic)
              << " (relative), alpha " << formatError(projection.alpha) << ", color "
              << formatError(projection.color) << ", depth " << formatError(projection.depth)
              << " (relative); tileCount exact\n"
              << "    image " << cam.width << "x" << cam.height << " worst color "
              << formatError(worstColor) << ", depth " << formatError(worstDepth) << ", "
              << covered << " of " << pixels << " pixels covered\n"
              << "    depth output worst D error " << formatError(worstDepthSum)
              << " (relative to max depth " << formatError(maxDepth) << ": "
              << formatError(maxDepth > 0.0 ? worstDepthSum / maxDepth : worstDepthSum)
              << "), coverage " << formatError(worstCoverage) << "; front and 0 exact\n";
    return run;
}

// Projection of one view direction per Gaussian, for every SH degree. Each
// direction is taken from a different camera-space offset, so the world-space
// view directions cover the sphere rather than sitting near -Z.
void checkShDirections(Device & device)
{
    SplitMix rng(0x5E0D1EC7ull);
    CpuCamera cam = makeCamera(rigidWorldToCamera(glm::vec3(0.6f, -0.3f, 0.7f), 0.9f,
                                                  glm::vec3(0.4f, -0.8f, 2.0f)),
                               500.0f, 480.0f, 120.0f, 130.0f, 256, 256);

    const glm::mat4 cameraToWorld = glm::inverse(cam.worldToCamera);
    const std::vector<glm::vec2> presets = {{0.0f, 0.0f},  {0.2f, 0.0f},  {-0.2f, 0.0f},
                                            {0.0f, 0.2f},  {0.0f, -0.2f}, {0.15f, 0.15f},
                                            {-0.15f, 0.15f}, {0.15f, -0.15f}};
    std::vector<SyntheticGaussian> scene;
    for (size_t i = 0; i < 64; ++i)
    {
        glm::vec2 offset = i < presets.size()
                               ? presets[i]
                               : glm::vec2(static_cast<float>(rng.uniform(-0.2, 0.2)),
                                           static_cast<float>(rng.uniform(-0.2, 0.2)));
        const double depth = rng.logUniform(2.0, 6.0);
        const glm::vec3 local(static_cast<float>(offset.x * depth),
                              static_cast<float>(offset.y * depth), static_cast<float>(-depth));
        SyntheticGaussian g{};
        g.position = glm::vec3(cameraToWorld * glm::vec4(local, 1.0f));
        g.scale = glm::vec3(0.02f);
        g.rotation = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        g.opacity = 0.5f;
        g.sh0 = glm::vec3(static_cast<float>(rng.uniform(-0.6, 0.6)),
                          static_cast<float>(rng.uniform(-0.6, 0.6)),
                          static_cast<float>(rng.uniform(-0.6, 0.6)));
        for (int k = 0; k < kShCoefficientsPerGaussian - 1; ++k)
        {
            g.shRest[k] = glm::vec3(static_cast<float>(rng.uniform(-0.15, 0.15)),
                                    static_cast<float>(rng.uniform(-0.15, 0.15)),
                                    static_cast<float>(rng.uniform(-0.15, 0.15)));
        }
        scene.push_back(g);
    }

    // The fixture only means something if the view directions really spread.
    double minDirZ = 1.0, maxAbsDirX = 0.0, maxAbsDirY = 0.0;
    for (const SyntheticGaussian & g : scene)
    {
        glm::vec3 dir = glm::normalize(g.position - cam.cameraPosition);
        minDirZ = std::min(minDirZ, static_cast<double>(dir.z));
        maxAbsDirX = std::max(maxAbsDirX, static_cast<double>(std::fabs(dir.x)));
        maxAbsDirY = std::max(maxAbsDirY, static_cast<double>(std::fabs(dir.y)));
    }
    if (maxAbsDirX < 0.2 || maxAbsDirY < 0.2)
    {
        throw std::runtime_error("SH direction fixture is degenerate (directions near -Z only)");
    }

    GaussianCloud cloud(device);
    uploadScene(cloud, scene);
    auto views = cloud.views();
    ForwardPass pass(device, cam.width, cam.height, testShaderDir());

    std::cout << "[SH view directions] " << scene.size() << " Gaussians; world view directions "
              << "span z down to " << minDirZ << ", |x| up to " << maxAbsDirX << ", |y| up to "
              << maxAbsDirY << "\n";
    for (uint32_t degree = 0; degree <= kShDegree; ++degree)
    {
        cam.shDegree = degree;
        pass.run(views, makeUniforms(cam, views.gaussianCount), ForwardStage::Project);
        std::vector<ProjectedGaussian> gpu(scene.size());
        pass.projected().download(gpu.data(), gpu.size() * sizeof(ProjectedGaussian));

        double worst = 0.0;
        for (size_t i = 0; i < scene.size(); ++i)
        {
            const CpuSplat ref = cpuProject(scene[i], cam);
            if (!ref.visible || gpu[i].tileCount == 0)
            {
                std::cerr << "SH fixture Gaussian " << i << " is not visible\n";
                throw std::runtime_error("SH fixture not visible");
            }
            const double color = std::max({std::fabs(gpu[i].color.x - ref.color[0]),
                                            std::fabs(gpu[i].color.y - ref.color[1]),
                                            std::fabs(gpu[i].color.z - ref.color[2])});
            worst = std::max(worst, color);
            if (color > kColorTolerance)
            {
                std::cerr << "SH degree " << degree << " Gaussian " << i << ": gpu=("
                          << gpu[i].color.x << ", " << gpu[i].color.y << ", " << gpu[i].color.z
                          << ") reference=(" << ref.color[0] << ", " << ref.color[1] << ", "
                          << ref.color[2] << ")\n";
                throw std::runtime_error("SH color mismatch");
            }
        }
        std::cout << "    degree " << degree << ": color worst " << formatError(worst) << "\n";
    }
}

// Image sizes that stress the tile grid: single pixel, partial tiles in both
// directions, a one-pixel-high strip, and a wide strip spanning 256 tiles.
void checkImageSizes(Device & device)
{
    const std::vector<std::pair<uint32_t, uint32_t>> sizes = {
        {1, 1}, {15, 17}, {17, 15}, {33, 1}, {4096, 16}};
    for (const auto & size : sizes)
    {
        const uint32_t w = size.first, h = size.second;
        SplitMix rng(0x51CE5ull + w * 131u + h);
        const float focal = 0.9f * static_cast<float>(std::max(w, h));
        const CpuCamera cam = makeCamera(glm::mat4(1.0f), focal, focal, 0.5f * w, 0.5f * h, w, h);
        const double sigmaMax = 0.1 * std::max(w, h) + 4.0;
        const auto scene = makeScatteredScene(rng, cam, 300, 1.0, 8.0, 0.3, sigmaMax, false);
        std::string label = "image size " + std::to_string(w) + "x" + std::to_string(h);
        checkScene(device, label, scene, cam);
    }
    std::cout << "image sizes OK: 1x1, 15x17, 17x15, 33x1, 4096x16 match the CPU blend\n";
}

// A large random scene under a rotated, translated camera with fx != fy and an
// off-center principal point, with SH degree 3 and varied scales, rotations
// and opacities. Also rerun with the antialiased compensation enabled.
void checkRandomScene(Device & device)
{
    SplitMix rng(0x5ABE12CA7ull);
    CpuCamera cam = makeCamera(rigidWorldToCamera(glm::vec3(1.0f, 2.0f, 3.0f), 0.4f,
                                                  glm::vec3(0.3f, -0.5f, 1.0f)),
                               520.0f, 488.0f, 131.5f, 118.25f, 256, 240);
    const auto scene = makeScatteredScene(rng, cam, 5000, 1.5, 30.0, 0.3, 60.0, true);
    checkScene(device, "random scene, 5000 Gaussians, rotated camera", scene, cam);

    cam.antialiased = true;
    checkScene(device, "random scene, antialiased compensation", scene, cam);
}

// Enough tile instances to push the radix sort across many workgroups and the
// counter scan into a second level (the scan's block sums need a second pass).
void checkLargeInstanceCount(Device & device)
{
    SplitMix rng(0x1A56E5CAull);
    const CpuCamera cam = makeCamera(rigidWorldToCamera(glm::vec3(-0.2f, 1.0f, 0.4f), -0.7f,
                                                        glm::vec3(-0.2f, 0.1f, 0.5f)),
                                     500.0f, 500.0f, 128.0f, 128.0f, 256, 256);
    const auto scene = makeScatteredScene(rng, cam, 8000, 6.0, 14.0, 4.0, 8.0, true);
    const SceneRun run = checkScene(device, "large instance count, 8000 Gaussians", scene, cam);
    if (run.instanceCount <= 65536)
    {
        std::cerr << "large scene produced only " << run.instanceCount << " instances\n";
        throw std::runtime_error("large scene is not large enough");
    }
}

// Capacity regrow: a sparse frame, then a dense one at >= 10x the instances
// (forcing the instance buffers to grow), then dense again (no growth), then
// sparse again (reusing the grown buffers). Every frame must match a fresh
// ForwardPass bit for bit.
void checkCapacityRegrow(Device & device)
{
    SplitMix rng(0x6E0E7A11ull);
    const CpuCamera cam = makeCamera(rigidWorldToCamera(glm::vec3(0.4f, 1.0f, -0.6f), 0.3f,
                                                        glm::vec3(0.1f, 0.2f, 1.5f)),
                                     500.0f, 500.0f, 128.0f, 128.0f, 256, 256);
    const auto sparse = makeScatteredScene(rng, cam, 40, 1.0, 5.0, 0.5, 1.5, false);
    const auto dense = makeScatteredScene(rng, cam, 3000, 2.0, 12.0, 4.0, 10.0, true);

    auto cpuInstances = [&](const std::vector<SyntheticGaussian> & scene)
    {
        uint64_t total = 0;
        for (const SyntheticGaussian & g : scene)
        {
            total += cpuProject(g, cam).tileCount;
        }
        return total;
    };
    const uint64_t sparseExpected = cpuInstances(sparse);
    const uint64_t denseExpected = cpuInstances(dense);
    if (denseExpected < 10 * sparseExpected)
    {
        std::cerr << "dense scene " << denseExpected << " is not 10x sparse " << sparseExpected
                  << "\n";
        throw std::runtime_error("regrow fixture too small");
    }

    GaussianCloud sparseCloud(device);
    uploadScene(sparseCloud, sparse);
    auto sparseViews = sparseCloud.views();
    const CameraUniforms sparseUniforms = makeUniforms(cam, sparseViews.gaussianCount);
    GaussianCloud denseCloud(device);
    uploadScene(denseCloud, dense);
    auto denseViews = denseCloud.views();
    const CameraUniforms denseUniforms = makeUniforms(cam, denseViews.gaussianCount);

    struct Frame
    {
        std::vector<glm::vec4> image;
        std::vector<float> depth;
    };
    auto freshFrame = [&](const auto & views, const CameraUniforms & uniforms)
    {
        ForwardPass fresh(device, cam.width, cam.height, testShaderDir());
        fresh.run(views, uniforms);
        return Frame{fresh.downloadOutputImage(), fresh.downloadDepthImage()};
    };
    ForwardPass grown(device, cam.width, cam.height, testShaderDir());
    auto runFrame = [&](const char * name, const auto & views, const CameraUniforms & uniforms,
                        uint64_t expected)
    {
        grown.run(views, uniforms);
        if (grown.lastInstanceCount() != expected)
        {
            std::cerr << name << ": lastInstanceCount " << grown.lastInstanceCount()
                      << " reference " << expected << "\n";
            throw std::runtime_error("regrow instance count mismatch");
        }
        return Frame{grown.downloadOutputImage(), grown.downloadDepthImage()};
    };
    auto requireSame = [](const char * name, const Frame & a, const Frame & b)
    {
        if (a.image != b.image || a.depth != b.depth)
        {
            std::cerr << name << ": frame differs from a fresh ForwardPass\n";
            throw std::runtime_error("regrow frame mismatch");
        }
    };

    const Frame sparseFresh = freshFrame(sparseViews, sparseUniforms);
    const Frame denseFresh = freshFrame(denseViews, denseUniforms);

    requireSame("sparse", runFrame("sparse", sparseViews, sparseUniforms, sparseExpected),
                sparseFresh);
    requireSame("dense (regrow)",
                runFrame("dense (regrow)", denseViews, denseUniforms, denseExpected), denseFresh);
    requireSame("dense again",
                runFrame("dense again", denseViews, denseUniforms, denseExpected), denseFresh);
    requireSame("sparse again",
                runFrame("sparse again", sparseViews, sparseUniforms, sparseExpected),
                sparseFresh);

    std::cout << "capacity regrow OK: sparse " << sparseExpected << " -> dense "
              << denseExpected << " instances (" << std::fixed << std::setprecision(1)
              << static_cast<double>(denseExpected) / sparseExpected << "x) -> dense -> sparse; "
              << "all four frames bit-identical to a fresh ForwardPass" << std::defaultfloat
              << "\n";
}

// Culling edges. Each probe sits next to a base Gaussian that is always
// visible, and the scene is checked end to end. Culled probes must leave the
// image bit-identical to the base-only frame; no probe may leave a non-finite
// value in the image.
void checkCullingEdges(Device & device)
{
    const CpuCamera cam = makeCamera(glm::mat4(1.0f), 500.0f, 500.0f, 128.0f, 128.0f, 256, 256);

    SyntheticGaussian base{};
    base.position = glm::vec3(0.0f, 0.0f, -20.0f);
    base.scale = glm::vec3(0.5f);
    base.rotation = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    base.opacity = 0.8f;
    base.sh0 = glm::vec3(0.3f, -0.2f, 0.5f);
    const SceneRun baseRun = checkScene(device, "culling base", {base}, cam);

    auto probeAt = [&](glm::vec3 position, glm::vec3 scale)
    {
        SyntheticGaussian g = base;
        g.position = position;
        g.scale = scale;
        return g;
    };
    // A Gaussian at depth 10 whose image position is given in pixels, so its
    // footprint edges can be placed relative to the frame and the tile grid.
    auto probeAtPixel = [&](float px, float py)
    {
        const float depth = 10.0f;
        return probeAt(glm::vec3((px - cam.cx) * depth / cam.fx, (cam.cy - py) * depth / cam.fy,
                                 -depth),
                       glm::vec3(0.1f));
    };
    // Radius at the image center (sigma 5 px, so ~15 px); edge probes refine it.
    const float r = static_cast<float>(cpuProject(probeAtPixel(128.0f, 128.0f), cam).radius);
    const float m = 0.02f; // margin in pixels, far above fp32 error at 256 px
    // Probe whose footprint satisfies center + sign * radius = target along one axis.
    // Perspective skews the footprint away from the optical axis, so the radius
    // depends on position; iterate to the fixed point.
    auto solveEdge = [&](bool horizontal, float target, float sign)
    {
        float center = target - sign * r;
        for (int i = 0; i < 8; ++i)
        {
            const SyntheticGaussian g =
                horizontal ? probeAtPixel(center, 128.0f) : probeAtPixel(128.0f, center);
            center = target - sign * static_cast<float>(cpuProject(g, cam).radius);
        }
        return horizontal ? probeAtPixel(center, 128.0f) : probeAtPixel(128.0f, center);
    };
    const float W = static_cast<float>(cam.width), H = static_cast<float>(cam.height);
    const float minDepth = kMinDepth;

    struct Probe
    {
        std::string name;
        SyntheticGaussian g;
        bool culled;
    };
    SyntheticGaussian nanMean = probeAt(glm::vec3(std::nanf(""), 0.0f, -10.0f), glm::vec3(0.1f));
    SyntheticGaussian nanDepth = probeAt(glm::vec3(0.0f, 0.0f, std::nanf("")), glm::vec3(0.1f));
    SyntheticGaussian nanScale = probeAt(glm::vec3(0.0f, 0.0f, -10.0f),
                                         glm::vec3(std::nanf(""), 0.1f, 0.1f));
    SyntheticGaussian infScale = probeAt(glm::vec3(0.0f, 0.0f, -10.0f),
                                         glm::vec3(INFINITY, 0.1f, 0.1f));
    SyntheticGaussian zeroScale = probeAt(glm::vec3(0.0f, 0.0f, -10.0f), glm::vec3(0.0f));
    SyntheticGaussian zeroQuat = probeAt(glm::vec3(0.0f, 0.0f, -10.0f), glm::vec3(0.1f));
    zeroQuat.rotation = glm::vec4(0.0f);

    const std::vector<Probe> probes = {
        {"depth exactly kMinDepth", probeAt(glm::vec3(0.0f, 0.0f, -minDepth), glm::vec3(0.1f)),
         true},
        {"depth one ulp beyond kMinDepth",
         probeAt(glm::vec3(0.0f, 0.0f, -std::nextafter(minDepth, 1.0f)), glm::vec3(0.1f)), false},
        {"depth one ulp inside kMinDepth",
         probeAt(glm::vec3(0.0f, 0.0f, -std::nextafter(minDepth, 0.0f)), glm::vec3(0.1f)), true},
        {"depth 0.1 (inside the near plane)",
         probeAt(glm::vec3(0.0f, 0.0f, -0.1f), glm::vec3(0.1f)), true},
        {"behind the camera", probeAt(glm::vec3(0.0f, 0.0f, 2.0f), glm::vec3(0.1f)), true},
        {"left edge, footprint reaches 0.02 px into the frame", solveEdge(true, m, 1.0f), false},
        {"left edge, footprint stops 0.02 px short of it", solveEdge(true, -m, 1.0f), true},
        {"right edge, footprint reaches 0.02 px into the frame", solveEdge(true, W - m, -1.0f),
         false},
        {"right edge, footprint starts 0.02 px past it", solveEdge(true, W + m, -1.0f), true},
        {"top edge, footprint reaches 0.02 px into the frame", solveEdge(false, m, 1.0f), false},
        {"top edge, footprint stops 0.02 px short of it", solveEdge(false, -m, 1.0f), true},
        {"bottom edge, footprint reaches 0.02 px into the frame", solveEdge(false, H - m, -1.0f),
         false},
        {"bottom edge, footprint starts 0.02 px past it", solveEdge(false, H + m, -1.0f), true},
        {"tile boundary x=128, footprint reaches 0.02 px past it", solveEdge(true, 128.0f + m, 1.0f),
         false},
        {"tile boundary x=128, footprint stops 0.02 px short of it",
         solveEdge(true, 128.0f - m, 1.0f), false},
        {"NaN mean", nanMean, true},
        {"NaN depth", nanDepth, true},
        {"NaN scale", nanScale, true},
        {"infinite scale", infScale, true},
        {"zero scale (dilation floor only)", zeroScale, false},
        {"zero quaternion (treated as identity)", zeroQuat, false},
    };

    for (const Probe & probe : probes)
    {
        const SceneRun run = checkScene(device, "culling: " + probe.name, {base, probe.g}, cam);
        const bool culled = run.projected[1].tileCount == 0;
        if (culled != probe.culled)
        {
            std::cerr << "culling: " << probe.name << " culled=" << culled << " expected "
                      << probe.culled << " (tileCount " << run.projected[1].tileCount << ")\n";
            throw std::runtime_error("culling decision mismatch");
        }
        for (size_t p = 0; p < run.image.size(); ++p)
        {
            if (!std::isfinite(run.image[p].r) || !std::isfinite(run.image[p].g)
                || !std::isfinite(run.image[p].b) || !std::isfinite(run.image[p].a))
            {
                std::cerr << "culling: " << probe.name << " left a non-finite pixel\n";
                throw std::runtime_error("non-finite pixel");
            }
        }
        if (culled && (run.image != baseRun.image || run.depth != baseRun.depth))
        {
            std::cerr << "culling: " << probe.name << " changed the image although culled\n";
            throw std::runtime_error("culled probe changed the image");
        }
    }
    std::cout << "culling edges OK: " << probes.size()
              << " probes; GPU tileCount matches the reference for each, every decision is the "
              << "expected one, culled probes leave the image bit-identical, no non-finite pixels\n";
}

} // namespace

int main()
{
    std::vector<SyntheticGaussian> synthetic = {
        // Front-and-center, radius ~30px, spans a handful of tiles.
        {glm::vec3(0.0f, 0.0f, -50.0f), glm::vec3(1.0f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), 0.9f,
        glm::vec3(1.0f)},
        // Closer, smaller, offset -- should overlap the first Gaussian's
        // tile range partially, exercising depth ordering in later stages.
        {glm::vec3(0.5f, 0.0f, -40.0f), glm::vec3(0.5f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f), 0.5f,
        glm::vec3(-1.0f)},
    };

    Device device;
    int result = 0;
    try
    {
        device.initialize(/*instanceExtensions=*/{}, /*enableValidation=*/true);

        GaussianCloud cloud(device);
        cloud.initialize(synthetic.size());
        for (size_t i = 0; i < synthetic.size(); ++i)
        {
            cloud.positions()[i] = glm::vec4(synthetic[i].position, 0.0f);
            cloud.scales()[i] = glm::vec4(synthetic[i].scale, 0.0f);
            cloud.rotations()[i] = synthetic[i].rotation;
            cloud.opacities()[i] = synthetic[i].opacity;
            cloud.shCoefficients()[i * kShCoefficientsPerGaussian] =
                glm::vec4(synthetic[i].sh0, 0.0f);
        }
        cloud.uploadToGPU();
        auto views = cloud.views();

        CameraUniforms camera{};
        camera.worldToCamera = glm::mat4(1.0f);
        camera.fx = kFx;
        camera.fy = kFy;
        camera.cx = kCx;
        camera.cy = kCy;
        camera.imageWidth = kImageWidth;
        camera.imageHeight = kImageHeight;
        camera.tilesX = dispatchGroups(kImageWidth, kTileSize);
        camera.tilesY = dispatchGroups(kImageHeight, kTileSize);
        camera.gaussianCount = views.gaussianCount;
        camera.shDegree = kShDegree;

        ForwardPass pass(device, kImageWidth, kImageHeight, testShaderDir());
        pass.run(views, camera);

        std::vector<ProjectedGaussian> gpuProjected(synthetic.size());
        pass.projected().download(gpuProjected.data(),
                                  gpuProjected.size() * sizeof(ProjectedGaussian));

        uint32_t expectedTotalInstances = 0;
        std::vector<uint32_t> expectedTileCounts(synthetic.size());
        std::vector<ProjectedGaussian> expectedProjected(synthetic.size());
        for (size_t i = 0; i < synthetic.size(); ++i)
        {
            ProjectedGaussian expected = referenceProject(synthetic[i], camera.tilesX, camera.tilesY);
            expectedProjected[i] = expected;
            expectedTileCounts[i] = expected.tileCount;
            expectedTotalInstances += expected.tileCount;

            char label[32];
            std::snprintf(label, sizeof(label), "Gaussian %zu", i);
            if (!checkProjected(label, gpuProjected[i], expected))
            {
                throw std::runtime_error("forward_project mismatch");
            }
        }

        if (pass.lastInstanceCount() != expectedTotalInstances)
        {
            std::cerr << "tileInstanceCount mismatch: gpu=" << pass.lastInstanceCount()
                      << " expected=" << expectedTotalInstances << std::endl;
            throw std::runtime_error("tileInstanceCount mismatch");
        }

        std::cout << "forward_project OK: " << synthetic.size() << " Gaussians projected, "
                  << expectedTotalInstances << " tile instances\n";

        // -- Stage 2: prefix_sum's exclusive scan, already dispatched as
        // part of run(). --
        std::vector<uint32_t> gpuOffsets(synthetic.size());
        pass.instanceOffsets().download(gpuOffsets.data(), gpuOffsets.size() * sizeof(uint32_t));

        uint32_t runningOffset = 0;
        for (size_t i = 0; i < synthetic.size(); ++i)
        {
            if (gpuOffsets[i] != runningOffset)
            {
                std::cerr << "instanceOffsets[" << i << "] mismatch: gpu=" << gpuOffsets[i]
                          << " expected=" << runningOffset << std::endl;
                throw std::runtime_error("prefix_sum mismatch");
            }
            runningOffset += expectedTileCounts[i];
        }
        std::cout << "prefix_sum OK (small case, " << synthetic.size() << " Gaussians)\n";

        // Small case never spans a chunk boundary (SABERCAT_WORKGROUP_SIZE
        // == 256 elements per chunk) -- prefix_sum's carry-between-chunks
        // logic is untested by the 2-Gaussian fixture above. Re-run the
        // same ForwardPass (exercising its gaussianCount-changed
        // reallocation path too) over 300 IDENTICAL Gaussians so the
        // expected offsets reduce to simple arithmetic (i * tileCount)
        // while still crossing two chunk boundaries.
        {
            constexpr size_t kLargeCount = 300;
            GaussianCloud largeCloud(device);
            largeCloud.initialize(kLargeCount);
            const SyntheticGaussian & proto = synthetic[0];
            for (size_t i = 0; i < kLargeCount; ++i)
            {
                largeCloud.positions()[i] = glm::vec4(proto.position, 0.0f);
                largeCloud.scales()[i] = glm::vec4(proto.scale, 0.0f);
                largeCloud.rotations()[i] = proto.rotation;
                largeCloud.opacities()[i] = proto.opacity;
                largeCloud.shCoefficients()[i * kShCoefficientsPerGaussian] =
                    glm::vec4(proto.sh0, 0.0f);
            }
            largeCloud.uploadToGPU();
            auto largeViews = largeCloud.views();

            CameraUniforms largeCamera = camera;
            largeCamera.gaussianCount = largeViews.gaussianCount;

            pass.run(largeViews, largeCamera);

            uint32_t protoTileCount = expectedTileCounts[0];
            uint32_t expectedLargeTotal = protoTileCount * static_cast<uint32_t>(kLargeCount);
            if (pass.lastInstanceCount() != expectedLargeTotal)
            {
                std::cerr << "large-case tileInstanceCount mismatch: gpu="
                          << pass.lastInstanceCount() << " expected=" << expectedLargeTotal
                          << std::endl;
                throw std::runtime_error("large-case tileInstanceCount mismatch");
            }

            std::vector<uint32_t> largeOffsets(kLargeCount);
            pass.instanceOffsets().download(largeOffsets.data(), kLargeCount * sizeof(uint32_t));
            for (size_t i = 0; i < kLargeCount; ++i)
            {
                uint32_t expected = protoTileCount * static_cast<uint32_t>(i);
                if (largeOffsets[i] != expected)
                {
                    std::cerr << "large-case instanceOffsets[" << i << "] mismatch: gpu="
                              << largeOffsets[i] << " expected=" << expected << std::endl;
                    throw std::runtime_error("large-case prefix_sum mismatch");
                }
            }
            std::cout << "prefix_sum OK (large case, " << kLargeCount
                      << " Gaussians, spans multiple 256-element chunks)\n";

            // -- Culling, stage: tile_rasterize's group-wide early-out. The
            // 300 identical Gaussians above (opacity 0.9, radius ~30px,
            // centered at pixel (128,128) i.e. tile (8,8) for this camera)
            // all land in the SAME single tile -- every pixel in that tile
            // sees transmittance collapse well within the first 256-element
            // batch (worst case, the tile's far corner at conic-weighted
            // distance ~4.49, needs ~92 identical hits to cross the 1e-4
            // threshold), so the early-out should skip fetching the second
            // batch entirely. Without the early-out, batchesFetched would be
            // ceil(300/256) = 2; proving it's 1 proves the early-out fired,
            // not just that the image happens to be correct (a no-op
            // "optimization" would still produce a correct image and pass
            // every other check in this file). --
            {
                constexpr uint32_t kCenterTileX = 8;
                constexpr uint32_t kCenterTileY = 8;
                uint32_t centerTile = kCenterTileY * largeCamera.tilesX + kCenterTileX;

                uint32_t totalTiles = largeCamera.tilesX * largeCamera.tilesY;
                std::vector<uint32_t> gpuBatchesFetched(totalTiles);
                pass.batchesFetched().download(gpuBatchesFetched.data(),
                                               gpuBatchesFetched.size() * sizeof(uint32_t));

                uint32_t expectedBatches =
                    dispatchGroups(kLargeCount, kTileThreads); // == 2 without the early-out
                if (gpuBatchesFetched[centerTile] == 0
                    || gpuBatchesFetched[centerTile] >= expectedBatches)
                {
                    std::cerr << "tile early-out did not fire: batchesFetched[" << centerTile
                              << "]=" << gpuBatchesFetched[centerTile]
                              << " (expected < " << expectedBatches << ", > 0)" << std::endl;
                    throw std::runtime_error("tile_rasterize early-out mismatch");
                }
                std::cout << "tile_rasterize early-out OK: tile " << centerTile << " fetched "
                          << gpuBatchesFetched[centerTile] << " of " << expectedBatches
                          << " possible batches for " << kLargeCount << " overlapping Gaussians\n";
            }
        }

        // -- Stage 3: tile_binning. Re-run over the original 2-Gaussian
        // fixture, stopping right after tile_binning's dispatch -- the
        // radix sort (stopAfter's default) would otherwise ping-pong set A
        // (tile_binning's output buffer) into something else before we get
        // to look at it. --
        pass.run(views, camera, ForwardStage::TileBinning);

        std::vector<uint32_t> emittedTile(expectedTotalInstances);
        std::vector<uint32_t> emittedDepth(expectedTotalInstances);
        std::vector<uint32_t> emittedValue(expectedTotalInstances);
        pass.sortedKeysTile().download(emittedTile.data(), emittedTile.size() * sizeof(uint32_t));
        pass.sortedKeysDepth().download(emittedDepth.data(),
                                        emittedDepth.size() * sizeof(uint32_t));
        pass.sortedValues().download(emittedValue.data(), emittedValue.size() * sizeof(uint32_t));

        std::vector<Instance> expectedInstances =
            generateExpectedInstances(expectedProjected, camera.tilesX, camera.tilesY);
        for (size_t slot = 0; slot < expectedInstances.size(); ++slot)
        {
            const Instance & expected = expectedInstances[slot];
            if (emittedTile[slot] != expected.tile || emittedDepth[slot] != expected.depth
                || emittedValue[slot] != expected.value)
            {
                std::cerr << "tile_binning mismatch at slot " << slot << ": gpu=(tile="
                          << emittedTile[slot] << ", depthBits=" << emittedDepth[slot]
                          << ", value=" << emittedValue[slot] << ") expected=(tile=" << expected.tile
                          << ", depthBits=" << expected.depth << ", value=" << expected.value << ")"
                          << std::endl;
                throw std::runtime_error("tile_binning mismatch");
            }
        }
        std::cout << "tile_binning OK: " << expectedTotalInstances << " (tile, depth, index) "
                  << "triples match\n";

        // -- Stage 4: radix sort (8 passes over the two key words). Expected
        // order is the same instances, stably sorted by (tile, depth) --
        // exactly what a correct LSD radix sort finishing on the
        // most-significant word (tile) produces. --
        pass.run(views, camera, ForwardStage::RadixSort);

        std::vector<uint32_t> sortedTile(expectedTotalInstances);
        std::vector<uint32_t> sortedDepth(expectedTotalInstances);
        std::vector<uint32_t> sortedValue(expectedTotalInstances);
        pass.sortedKeysTile().download(sortedTile.data(), sortedTile.size() * sizeof(uint32_t));
        pass.sortedKeysDepth().download(sortedDepth.data(), sortedDepth.size() * sizeof(uint32_t));
        pass.sortedValues().download(sortedValue.data(), sortedValue.size() * sizeof(uint32_t));

        std::vector<Instance> expectedSorted = expectedInstances;
        std::stable_sort(expectedSorted.begin(), expectedSorted.end(),
                         [](const Instance & a, const Instance & b)
                         {
                             if (a.tile != b.tile)
                             {
                                 return a.tile < b.tile;
                             }
                             return a.depth < b.depth;
                         });

        for (size_t i = 0; i < expectedSorted.size(); ++i)
        {
            const Instance & expected = expectedSorted[i];
            if (sortedTile[i] != expected.tile || sortedDepth[i] != expected.depth
                || sortedValue[i] != expected.value)
            {
                std::cerr << "radix sort mismatch at index " << i << ": gpu=(tile=" << sortedTile[i]
                          << ", depthBits=" << sortedDepth[i] << ", value=" << sortedValue[i]
                          << ") expected=(tile=" << expected.tile << ", depthBits=" << expected.depth
                          << ", value=" << expected.value << ")" << std::endl;
                throw std::runtime_error("radix sort mismatch");
            }
        }
        std::cout << "radix sort OK: " << expectedSorted.size()
                  << " instances sorted tile-major, depth-minor, stable\n";

        // -- Stage 5: tile_ranges. Expected [first,last) per tile falls out
        // of the same stably-sorted instance list -- contiguous runs of
        // equal tile id. Tiles with no run default to (0,0): tile_ranges
        // never writes them, and run() clears the buffer every call
        // specifically so that untouched tiles read as empty rather than
        // inheriting a previous call's span (the fixture reuses `pass`
        // across every stage, so this also checks that the clear actually
        // happens each time, not just once). --
        pass.run(views, camera, ForwardStage::TileRanges);

        uint32_t totalTiles = camera.tilesX * camera.tilesY;
        std::vector<TileRange> gpuRanges(totalTiles);
        pass.tileRanges().download(gpuRanges.data(), gpuRanges.size() * sizeof(TileRange));

        std::vector<TileRange> expectedRanges(totalTiles, TileRange{0, 0});
        for (uint32_t i = 0; i < expectedSorted.size();)
        {
            uint32_t tile = expectedSorted[i].tile;
            uint32_t start = i;
            while (i < expectedSorted.size() && expectedSorted[i].tile == tile)
            {
                ++i;
            }
            expectedRanges[tile] = TileRange{start, i};
        }

        for (uint32_t tile = 0; tile < totalTiles; ++tile)
        {
            if (gpuRanges[tile].first != expectedRanges[tile].first
                || gpuRanges[tile].last != expectedRanges[tile].last)
            {
                std::cerr << "tile_ranges mismatch at tile " << tile << ": gpu=[" << gpuRanges[tile].first
                          << ", " << gpuRanges[tile].last << ") expected=["
                          << expectedRanges[tile].first << ", " << expectedRanges[tile].last << ")"
                          << std::endl;
                throw std::runtime_error("tile_ranges mismatch");
            }
        }
        std::cout << "tile_ranges OK: " << totalTiles << " tiles checked (incl. cleared-empty case)\n";

        // -- Stage 6: tile_rasterize. Full default run (through Rasterize),
        // checked against a CPU front-to-back blend over the SAME sorted
        // instance list + per-tile ranges already verified above -- the
        // shader's cooperative batching only changes where data comes from,
        // not the blend order, so a plain per-pixel loop over each tile's
        // [first,last) slice reproduces it exactly. --
        pass.run(views, camera);

        std::vector<glm::vec4> gpuColor = pass.downloadOutputImage();
        std::vector<float> gpuDepth = pass.downloadDepthImage();

        uint32_t mismatches = 0;
        float worstColorError = 0.0f;
        float worstDepthError = 0.0f;
        for (uint32_t y = 0; y < kImageHeight && mismatches < 5; ++y)
        {
            for (uint32_t x = 0; x < kImageWidth && mismatches < 5; ++x)
            {
                uint32_t tileId = (y / kTileSize) * camera.tilesX + (x / kTileSize);
                TileRange range = expectedRanges[tileId];

                glm::vec2 pixelPosition(static_cast<float>(x), static_cast<float>(y));
                glm::vec3 accumulated(0.0f);
                float transmittance = 1.0f;
                float frontDepth = 0.0f;
                bool hasFrontDepth = false;

                for (uint32_t idx = range.first; idx < range.last; ++idx)
                {
                    const ProjectedGaussian & g = expectedProjected[expectedSorted[idx].value];
                    glm::vec2 delta = pixelPosition - g.center;
                    float exponent = g.conic.x * delta.x * delta.x
                                     + 2.0f * g.conic.y * delta.x * delta.y
                                     + g.conic.z * delta.y * delta.y;
                    if (exponent < 0.0f)
                    {
                        continue;
                    }
                    float alpha = std::min(g.alpha * std::exp(-0.5f * exponent), kMaxAlpha);
                    if (alpha < 1.0f / 255.0f)
                    {
                        continue;
                    }
                    if (!hasFrontDepth)
                    {
                        frontDepth = g.depth;
                        hasFrontDepth = true;
                    }
                    accumulated += transmittance * alpha * g.color;
                    transmittance *= (1.0f - alpha);
                    if (transmittance < 1e-4f)
                    {
                        break;
                    }
                }

                // Measured worst color error for this scene is ~1e-7 (two blend steps of
            // fp32 rounding), so 1e-5 keeps two orders of magnitude of margin. Depth is
            // a stored float copied through unchanged; 1e-4 covers ~25 ulps at depth 50.
            glm::vec4 expectedColor(accumulated, 1.0f - transmittance);
                const glm::vec4 & gpu = gpuColor[y * kImageWidth + x];
                float colorError = std::max({std::fabs(gpu.r - expectedColor.r),
                                             std::fabs(gpu.g - expectedColor.g),
                                             std::fabs(gpu.b - expectedColor.b),
                                             std::fabs(gpu.a - expectedColor.a)});
                float depthError = std::fabs(gpuDepth[y * kImageWidth + x] - frontDepth);
                worstColorError = std::max(worstColorError, colorError);
                worstDepthError = std::max(worstDepthError, depthError);

                if (colorError > 1e-5f || depthError > 1e-4f)
                {
                    std::cerr << "tile_rasterize mismatch at (" << x << ", " << y
                              << "): gpu color=(" << gpu.r << ", " << gpu.g << ", " << gpu.b << ", "
                              << gpu.a << ") expected=(" << expectedColor.r << ", " << expectedColor.g
                              << ", " << expectedColor.b << ", " << expectedColor.a
                              << "); gpu depth=" << gpuDepth[y * kImageWidth + x]
                              << " expected=" << frontDepth << std::endl;
                    ++mismatches;
                }
            }
        }
        if (mismatches > 0)
        {
            throw std::runtime_error("tile_rasterize mismatch");
        }
        std::cout << "tile_rasterize OK: " << kImageWidth << "x" << kImageHeight
                  << " pixels match (worst color error " << worstColorError
                  << ", worst depth error " << worstDepthError << ")\n";

        // -- Culling: screen-space size cull. A second Gaussian at the same
        // depth/position as synthetic[0] but with a near-zero scale: its true
        // covariance is negligible, so cov2d is dominated by the isotropic
        // dilation floor (radius -> ~1.5px, see config.slangh's comment on
        // why that's the floor), which sits below the 2px threshold and
        // should cull. synthetic[0] itself (radius ~30px) must stay unculled
        // in the same run, proving this isn't just culling everything.
        //
        // The cull is now OFF by default (LODParams::minScreenRadius == 0),
        // so this fixture turns it on explicitly and turns it back off after.
        // That default changed deliberately: an unconditional 2px threshold
        // was measured culling >95% of real scenes, erasing the subject and
        // leaving only huge background Gaussians. See config.slangh. --
        {
            std::vector<SyntheticGaussian> sizeCullCloud = {
                synthetic[0],
                {glm::vec3(0.0f, 0.0f, -50.0f), glm::vec3(0.001f), glm::vec4(0.0f, 0.0f, 0.0f, 1.0f),
                0.5f, glm::vec3(0.2f)},
            };

            GaussianCloud cloud2(device);
            cloud2.initialize(sizeCullCloud.size());
            for (size_t i = 0; i < sizeCullCloud.size(); ++i)
            {
                cloud2.positions()[i] = glm::vec4(sizeCullCloud[i].position, 0.0f);
                cloud2.scales()[i] = glm::vec4(sizeCullCloud[i].scale, 0.0f);
                cloud2.rotations()[i] = sizeCullCloud[i].rotation;
                cloud2.opacities()[i] = sizeCullCloud[i].opacity;
                cloud2.shCoefficients()[i * kShCoefficientsPerGaussian] =
                    glm::vec4(sizeCullCloud[i].sh0, 0.0f);
            }
            cloud2.uploadToGPU();
            auto cloud2Views = cloud2.views();

            CameraUniforms cloud2Camera = camera;
            cloud2Camera.gaussianCount = cloud2Views.gaussianCount;

            LODParams cullParams{};
            cullParams.minScreenRadius = kDefaultMinScreenRadius;
            pass.setLODParams(cullParams);

            pass.run(cloud2Views, cloud2Camera, ForwardStage::Project);

            std::vector<ProjectedGaussian> gpuCull(sizeCullCloud.size());
            pass.projected().download(gpuCull.data(), gpuCull.size() * sizeof(ProjectedGaussian));

            ProjectedGaussian expectedKept =
                referenceProject(sizeCullCloud[0], camera.tilesX, camera.tilesY, kShDegree,
                                 glm::vec3(0.0f), kDefaultMinScreenRadius);
            ProjectedGaussian expectedCulled =
                referenceProject(sizeCullCloud[1], camera.tilesX, camera.tilesY, kShDegree,
                                 glm::vec3(0.0f), kDefaultMinScreenRadius);
            if (!checkProjected("size-cull kept", gpuCull[0], expectedKept))
            {
                throw std::runtime_error("screen-space cull mismatch (should be kept)");
            }
            if (gpuCull[1].tileCount != 0 || expectedCulled.tileCount != 0)
            {
                std::cerr << "screen-space cull did not fire: gpu tileCount="
                          << gpuCull[1].tileCount
                          << " expected tileCount=" << expectedCulled.tileCount << std::endl;
                throw std::runtime_error("screen-space cull mismatch (should be culled)");
            }
            // A second run with the cull back at its real default proves the
            // default is genuinely OFF -- the same ~1.5px Gaussian that was
            // just culled must now survive. Without this the fixture would
            // still pass if the cull had been left unconditionally on.
            pass.setLODParams(LODParams{});
            pass.run(cloud2Views, cloud2Camera, ForwardStage::Project);
            pass.projected().download(gpuCull.data(),
                                      gpuCull.size() * sizeof(ProjectedGaussian));
            if (gpuCull[1].tileCount == 0)
            {
                std::cerr << "screen-space cull fired with minScreenRadius=0, but 0 must "
                             "disable it\n";
                throw std::runtime_error("screen-space cull default is not disabled");
            }

            // Compensation is a per-scene flag, OFF by default
            // (vanilla 3DGS trains without it). With the flag clear the
            // sub-pixel Gaussian must keep its stored opacity EXACTLY -- a
            // compensation that leaked through regardless of the flag would
            // fail here.
            const float storedOpacity = sizeCullCloud[1].opacity;
            if (gpuCull[1].alpha != storedOpacity)
            {
                std::cerr << "compensation applied with the antialiased flag clear: alpha "
                          << gpuCull[1].alpha << " vs stored opacity " << storedOpacity << "\n";
                throw std::runtime_error("opacity compensation not gated by the flag");
            }

            // Degeneracy guard for the dilation opacity compensation, flag
            // SET. Every OTHER fixture in this file uses Gaussians large
            // enough that the compensation factor is ~1, so they would all
            // still pass even if it were dropped entirely. This Gaussian is
            // deliberately sub-pixel (radius pinned at the ~1.5px dilation
            // floor), which is exactly where the compensation must bite hard
            // -- for an antialiased scene, leaving it out deposits full
            // opacity over a vastly inflated area and saturates the image.
            CameraUniforms antialiasedCamera = cloud2Camera;
            antialiasedCamera.renderFlags = kRenderFlagAntialiased;
            pass.run(cloud2Views, antialiasedCamera, ForwardStage::Project);
            pass.projected().download(gpuCull.data(),
                                      gpuCull.size() * sizeof(ProjectedGaussian));
            for (size_t i = 0; i < sizeCullCloud.size(); ++i)
            {
                ProjectedGaussian expectedAA =
                    referenceProject(sizeCullCloud[i], camera.tilesX, camera.tilesY, kShDegree,
                                     glm::vec3(0.0f), 0.0f, /*antialiased=*/true);
                if (!checkProjected("antialiased compensation", gpuCull[i], expectedAA))
                {
                    throw std::runtime_error("antialiased compensation mismatch");
                }
            }
            if (!(gpuCull[1].alpha < storedOpacity * 0.5f))
            {
                std::cerr << "dilation opacity compensation did not fire: sub-pixel "
                             "Gaussian kept alpha " << gpuCull[1].alpha
                          << " against stored opacity " << storedOpacity << "\n";
                throw std::runtime_error("dilation opacity compensation missing");
            }
            std::cout << "dilation opacity compensation OK: sub-pixel Gaussian attenuated "
                      << storedOpacity / std::max(gpuCull[1].alpha, 1e-9f)
                      << "x (opacity " << storedOpacity << " -> alpha "
                      << gpuCull[1].alpha << "), while the ~30px Gaussian keeps "
                      << gpuCull[0].alpha << "\n";

            std::cout << "screen-space size cull OK: at 2px threshold radius-~30px Gaussian "
                         "kept and radius-~1.5px Gaussian culled; at the default (0) the "
                         "same small Gaussian survives\n";
        }

        // -- SH evaluation: real SH1-3 evaluation + the global degree cap. One
        // Gaussian with distinct NON-ZERO coefficients in every band, so a
        // degree cap that were silently ignored (or a band evaluated at the
        // wrong index) changes the color and fails. Camera sits at the world
        // origin with the Gaussian at (0,0,-50), making the view direction
        // exactly (0,0,-1) -- no floating-point ambiguity in the fixture
        // itself, only in the SH math under test. --
        {
            SyntheticGaussian shGaussian = synthetic[0];
            // Distinct per-band, per-channel values: a wrong coefficient
            // index produces a different color rather than coincidentally
            // matching.
            shGaussian.sh0 = glm::vec3(0.3f, -0.2f, 0.5f);
            for (int i = 0; i < kShCoefficientsPerGaussian - 1; ++i)
            {
                float k = static_cast<float>(i + 1);
                shGaussian.shRest[i] =
                    glm::vec3(0.05f * k, -0.03f * k, 0.02f * k);
            }

            GaussianCloud shCloud(device);
            shCloud.initialize(1);
            shCloud.positions()[0] = glm::vec4(shGaussian.position, 0.0f);
            shCloud.scales()[0] = glm::vec4(shGaussian.scale, 0.0f);
            shCloud.rotations()[0] = shGaussian.rotation;
            shCloud.opacities()[0] = shGaussian.opacity;
            glm::vec3 packed[kShCoefficientsPerGaussian];
            packSHCoefficients(shGaussian, packed);
            for (int i = 0; i < kShCoefficientsPerGaussian; ++i)
            {
                shCloud.shCoefficients()[i] = glm::vec4(packed[i], 0.0f);
            }
            shCloud.uploadToGPU();
            auto shViews = shCloud.views();

            const glm::vec3 shCameraPosition(0.0f);
            for (uint32_t degree = 0; degree <= kShDegree; ++degree)
            {
                CameraUniforms shCamera = camera;
                shCamera.gaussianCount = shViews.gaussianCount;
                shCamera.shDegree = degree;
                shCamera.cameraPositionWorld = glm::vec4(shCameraPosition, 0.0f);

                pass.run(shViews, shCamera, ForwardStage::Project);

                ProjectedGaussian gpuSH{};
                pass.projected().download(&gpuSH, sizeof(ProjectedGaussian));

                ProjectedGaussian expectedSH = referenceProject(
                    shGaussian, camera.tilesX, camera.tilesY, degree, shCameraPosition);

                char label[32];
                std::snprintf(label, sizeof(label), "SH degree %u", degree);
                if (!checkProjected(label, gpuSH, expectedSH))
                {
                    throw std::runtime_error("SH evaluation mismatch");
                }
            }

            // Sanity check on the fixture itself: if every degree produced
            // the same color, the checks above would pass trivially even
            // with the cap ignored. Confirm the bands actually move the
            // result.
            glm::vec3 coefficients[kShCoefficientsPerGaussian];
            packSHCoefficients(shGaussian, coefficients);
            glm::vec3 viewDir = glm::normalize(shGaussian.position - shCameraPosition);
            glm::vec3 atDegree0 = referenceEvaluateSH(0, coefficients, viewDir);
            glm::vec3 atDegree3 = referenceEvaluateSH(3, coefficients, viewDir);
            float degreeSpread = maxAbsDifference(&atDegree0.x, &atDegree3.x, 3);
            if (degreeSpread < 1e-3f)
            {
                throw std::runtime_error(
                    "SH fixture is degenerate: degree 0 and 3 produce the same color");
            }

            std::cout << "SH evaluation OK: degrees 0-" << kShDegree
                      << " each match an independent C++ reference "
                      << "(degree 0 vs 3 differ by " << degreeSpread << ")\n";
        }

        // -- LOD: distance + tile-density hybrid dynamic LOD. --
        {
            LODTestParams lodTest{/*nearDistance=*/10.0f, /*farDistance=*/200.0f,
                                  /*maxDropDistance=*/0.6f, /*maxDropDensity=*/0.6f,
                                  /*densityReference=*/200.0f};

            // Seed run: 77 identical, deliberately tiny (single-tile-only,
            // see the screen-space-cull fixture above for why scale 0.05
            // still clears SABERCAT_MIN_SCREEN_RADIUS) Gaussians, all at
            // pixel (128,128) i.e. tile (8,8) -- so after this run,
            // ForwardPass's tileRanges buffer (still resident until the
            // NEXT run()'s round 2 rewrites it -- see forward_pass.cpp's
            // binding-7 comment) holds exactly 77 instances for that tile.
            // This is the exact mechanism forward_project's previousTileRanges
            // reads, reproduced end to end rather than faked.
            constexpr uint32_t kSeedCount = 77;
            constexpr uint32_t kDensityTile = 8 * 16 + 8; // tile (8,8), 16 = tilesX at 256x256
            GaussianCloud seedCloud(device);
            seedCloud.initialize(kSeedCount);
            for (uint32_t i = 0; i < kSeedCount; ++i)
            {
                seedCloud.positions()[i] = glm::vec4(0.0f, 0.0f, -50.0f, 0.0f);
                seedCloud.scales()[i] = glm::vec4(0.05f, 0.05f, 0.05f, 0.0f);
                seedCloud.rotations()[i] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
                seedCloud.opacities()[i] = 0.5f;
                seedCloud.shCoefficients()[i * kShCoefficientsPerGaussian] =
                    glm::vec4(0.1f, 0.1f, 0.1f, 0.0f);
            }
            seedCloud.uploadToGPU();
            auto seedViews = seedCloud.views();
            CameraUniforms seedCamera = camera;
            seedCamera.gaussianCount = seedViews.gaussianCount;
            pass.run(seedViews, seedCamera);

            std::vector<TileRange> seedRanges(camera.tilesX * camera.tilesY);
            pass.tileRanges().download(seedRanges.data(), seedRanges.size() * sizeof(TileRange));
            uint32_t seedDensity = seedRanges[kDensityTile].last - seedRanges[kDensityTile].first;
            if (seedDensity != kSeedCount)
            {
                std::cerr << "LOD fixture setup error: seed tile density=" << seedDensity
                          << " expected=" << kSeedCount << std::endl;
                throw std::runtime_error("LOD fixture seed density mismatch");
            }

            // Test cloud: on-axis (so every Gaussian's center tile is the
            // SAME tile the seed just populated), varying depth, scale large
            // enough to clear the screen-space cull at every depth tested.
            std::vector<float> testDepths = {5.0f, 20.0f, 50.0f, 80.0f, 150.0f, 300.0f};
            GaussianCloud lodCloud(device);
            lodCloud.initialize(testDepths.size());
            for (size_t i = 0; i < testDepths.size(); ++i)
            {
                lodCloud.positions()[i] = glm::vec4(0.0f, 0.0f, -testDepths[i], 0.0f);
                lodCloud.scales()[i] = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f);
                lodCloud.rotations()[i] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
                lodCloud.opacities()[i] = 0.5f;
                lodCloud.shCoefficients()[i * kShCoefficientsPerGaussian] =
                    glm::vec4(0.1f, 0.1f, 0.1f, 0.0f);
            }
            lodCloud.uploadToGPU();
            auto lodViews = lodCloud.views();
            CameraUniforms lodCamera = camera;
            lodCamera.gaussianCount = lodViews.gaussianCount;

            LODParams lodParams{};
            lodParams.enabled = 1;
            lodParams.nearDistance = lodTest.nearDistance;
            lodParams.farDistance = lodTest.farDistance;
            lodParams.maxDropDistance = lodTest.maxDropDistance;
            lodParams.maxDropDensity = lodTest.maxDropDensity;
            lodParams.densityReference = lodTest.densityReference;
            pass.setLODParams(lodParams);

            // stopAfter=Project only -- round 2 never runs, so tileRanges
            // (still holding the seed's 77-instance tile) is untouched by
            // this call, matching the real per-frame mechanism exactly.
            pass.run(lodViews, lodCamera, ForwardStage::Project);

            std::vector<ProjectedGaussian> gpuLod(testDepths.size());
            pass.projected().download(gpuLod.data(), gpuLod.size() * sizeof(ProjectedGaussian));

            for (size_t i = 0; i < testDepths.size(); ++i)
            {
                float dropFraction =
                    expectedDropFraction(testDepths[i], static_cast<float>(kSeedCount), lodTest);
                bool expectedDropped = hashToUnitFloat(static_cast<uint32_t>(i)) < dropFraction;
                bool gpuDropped = gpuLod[i].tileCount == 0;
                if (expectedDropped != gpuDropped)
                {
                    std::cerr << "LOD decision mismatch at depth " << testDepths[i]
                              << ": gpu dropped=" << gpuDropped
                              << " expected dropped=" << expectedDropped
                              << " (dropFraction=" << dropFraction << ")" << std::endl;
                    throw std::runtime_error("hybrid LOD decision mismatch");
                }
            }
            std::cout << "hybrid LOD OK: " << testDepths.size()
                      << " depths matched an independent hash+threshold reimplementation "
                      << "(previous-tile density=" << kSeedCount << ")\n";

            // Nested-threshold property: isolate the distance term
            // (maxDropDensity=0) and confirm raising maxDropDistance never
            // un-drops a Gaussian that was already dropped at a lower value
            // -- the property that keeps the surviving set from reshuffling
            // frame to frame as the camera moves.
            LODParams lowParams = lodParams;
            lowParams.maxDropDensity = 0.0f;
            lowParams.maxDropDistance = 0.2f;
            pass.setLODParams(lowParams);
            pass.run(lodViews, lodCamera, ForwardStage::Project);
            std::vector<ProjectedGaussian> gpuLow(testDepths.size());
            pass.projected().download(gpuLow.data(), gpuLow.size() * sizeof(ProjectedGaussian));

            LODParams highParams = lowParams;
            highParams.maxDropDistance = 0.9f;
            pass.setLODParams(highParams);
            pass.run(lodViews, lodCamera, ForwardStage::Project);
            std::vector<ProjectedGaussian> gpuHigh(testDepths.size());
            pass.projected().download(gpuHigh.data(), gpuHigh.size() * sizeof(ProjectedGaussian));

            for (size_t i = 0; i < testDepths.size(); ++i)
            {
                bool droppedLow = gpuLow[i].tileCount == 0;
                bool droppedHigh = gpuHigh[i].tileCount == 0;
                if (droppedLow && !droppedHigh)
                {
                    std::cerr << "LOD nested-threshold violation at depth " << testDepths[i]
                              << ": dropped at maxDropDistance=0.2 but kept at 0.9" << std::endl;
                    throw std::runtime_error("LOD nested-threshold violation");
                }
            }
            std::cout << "hybrid LOD nested-threshold OK: raising maxDropDistance never "
                      << "un-dropped an already-dropped Gaussian\n";

            // -- Adaptive per-Gaussian SH degree. Same pressure
            // signals, mapped onto degree instead of keep/drop. Geometry
            // dropping is switched OFF here so every Gaussian survives and
            // its color can be compared -- this isolates the appearance axis.
            // The seed cloud's 77-instance tile density from above is still
            // resident (Project-only runs never rewrite tileRanges), so the
            // density term is live, not stubbed. --
            {
                std::vector<float> shDepths = {5.0f, 60.0f, 120.0f, 300.0f};
                GaussianCloud shLodCloud(device);
                shLodCloud.initialize(shDepths.size());
                std::vector<SyntheticGaussian> shLodSynthetic(shDepths.size());
                for (size_t i = 0; i < shDepths.size(); ++i)
                {
                    SyntheticGaussian g{};
                    g.position = glm::vec3(0.0f, 0.0f, -shDepths[i]);
                    g.scale = glm::vec3(1.0f);
                    g.rotation = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
                    g.opacity = 0.5f;
                    g.sh0 = glm::vec3(0.3f, -0.2f, 0.5f);
                    for (int k = 0; k < kShCoefficientsPerGaussian - 1; ++k)
                    {
                        float s = static_cast<float>(k + 1);
                        g.shRest[k] = glm::vec3(0.05f * s, -0.03f * s, 0.02f * s);
                    }
                    shLodSynthetic[i] = g;

                    shLodCloud.positions()[i] = glm::vec4(g.position, 0.0f);
                    shLodCloud.scales()[i] = glm::vec4(g.scale, 0.0f);
                    shLodCloud.rotations()[i] = g.rotation;
                    shLodCloud.opacities()[i] = g.opacity;
                    glm::vec3 packedLod[kShCoefficientsPerGaussian];
                    packSHCoefficients(g, packedLod);
                    for (int k = 0; k < kShCoefficientsPerGaussian; ++k)
                    {
                        shLodCloud.shCoefficients()[i * kShCoefficientsPerGaussian + k] =
                            glm::vec4(packedLod[k], 0.0f);
                    }
                }
                shLodCloud.uploadToGPU();
                auto shLodViews = shLodCloud.views();

                CameraUniforms shLodCamera = camera;
                shLodCamera.gaussianCount = shLodViews.gaussianCount;
                shLodCamera.shDegree = kShDegree;
                shLodCamera.cameraPositionWorld = glm::vec4(0.0f);

                LODParams appearanceParams{};
                appearanceParams.enabled = 0; // geometry untouched
                appearanceParams.appearanceEnabled = 1;
                appearanceParams.nearDistance = lodTest.nearDistance;
                appearanceParams.farDistance = lodTest.farDistance;
                appearanceParams.densityReference = lodTest.densityReference;
                appearanceParams.appearanceMaxDropDistance = 0.9f;
                appearanceParams.appearanceMaxDropDensity = 0.0f; // isolate distance
                pass.setLODParams(appearanceParams);
                pass.run(shLodViews, shLodCamera, ForwardStage::Project);

                std::vector<ProjectedGaussian> gpuAppearance(shDepths.size());
                pass.projected().download(gpuAppearance.data(),
                                          gpuAppearance.size() * sizeof(ProjectedGaussian));

                uint32_t previousDegree = kShDegree + 1;
                bool sawDegreeStepDown = false;
                for (size_t i = 0; i < shDepths.size(); ++i)
                {
                    float distanceFraction =
                        std::clamp((shDepths[i] - lodTest.nearDistance)
                                       / std::max(lodTest.farDistance - lodTest.nearDistance,
                                                  1e-4f),
                                   0.0f, 1.0f);
                    float degradeFraction =
                        distanceFraction * appearanceParams.appearanceMaxDropDistance;
                    uint32_t expectedDegree = expectedAdaptiveDegree(kShDegree, degradeFraction);

                    ProjectedGaussian expected = referenceProject(
                        shLodSynthetic[i], camera.tilesX, camera.tilesY, expectedDegree,
                        glm::vec3(0.0f));

                    char label[48];
                    std::snprintf(label, sizeof(label), "appearance LOD @ depth %.0f",
                                  shDepths[i]);
                    if (!checkProjected(label, gpuAppearance[i], expected))
                    {
                        throw std::runtime_error("appearance LOD mismatch");
                    }

                    // Monotonic falloff: degree must never rise with depth.
                    if (expectedDegree > previousDegree)
                    {
                        throw std::runtime_error("appearance LOD degree rose with distance");
                    }
                    if (previousDegree <= kShDegree && expectedDegree < previousDegree)
                    {
                        sawDegreeStepDown = true;
                    }
                    previousDegree = expectedDegree;
                }

                // Guard against a degenerate fixture: if the chosen depths
                // never actually crossed a degree boundary, the per-depth
                // color checks above would pass even with adaptive degree
                // wired to a constant.
                if (!sawDegreeStepDown)
                {
                    throw std::runtime_error(
                        "appearance LOD fixture is degenerate: degree never stepped down");
                }

                std::cout << "appearance LOD OK: " << shDepths.size()
                          << " depths stepped SH degree down monotonically, colors match "
                          << "the C++ reference at each selected degree\n";
            }
        }

    // -- Scene-level checks against a double-precision CPU renderer.
    checkCullingEdges(device);
    checkShDirections(device);
    checkImageSizes(device);
    checkRandomScene(device);
    checkLargeInstanceCount(device);
    checkCapacityRegrow(device);
    }
    catch (const std::exception & e)
    {
        std::cerr << "forward_test FAILED: " << e.what() << std::endl;
        result = 1;
    }

    device.shutdown();
    return result;
}
