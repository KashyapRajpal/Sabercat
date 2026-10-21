// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Host-side view of the launch configuration the Slang kernels compile
// against. The values live in shaders/config.slangh and are included here
// verbatim, so host and device can never disagree about tile size,
// workgroup size, or the constants baked into the rasterizer math
// (PERFORMANCE.md invariant #9). The build copies it into
// <build>/include/sabercat/shaders/ and installs it to the same relative
// path, so this include resolves identically in a build tree and an install.
#include <sabercat/shaders/config.slangh>

namespace sabercat
{

constexpr uint32_t kTileSize = SABERCAT_TILE_SIZE;
constexpr uint32_t kTileThreads = SABERCAT_TILE_THREADS;
constexpr uint32_t kWorkgroupSize = SABERCAT_WORKGROUP_SIZE;

constexpr uint32_t kRadixBits = SABERCAT_RADIX_BITS;
constexpr uint32_t kRadixBuckets = SABERCAT_RADIX_BUCKETS;
constexpr uint32_t kRadixPasses = SABERCAT_RADIX_PASSES;

constexpr uint32_t kScanBlock = SABERCAT_SCAN_BLOCK;

constexpr uint32_t kShDegree = SABERCAT_SH_DEGREE;
constexpr uint32_t kShCoefficients = SABERCAT_SH_COEFFS;

// Canonical stream strides. Fixed by std430 alignment + the shared buffer
// contract -- see the long note in shaders/config.slangh for why these are
// not tunable.
constexpr uint32_t kMeansStride = SABERCAT_STRIDE_MEANS;
constexpr uint32_t kScalesStride = SABERCAT_STRIDE_SCALES;
constexpr uint32_t kRotationsStride = SABERCAT_STRIDE_ROTATIONS;
constexpr uint32_t kShStride = SABERCAT_STRIDE_SH;
constexpr uint32_t kOpacityStride = SABERCAT_STRIDE_OPACITY;
constexpr uint32_t kConfidenceStride = SABERCAT_STRIDE_CONFIDENCE;

// Optional relightable streams and lighting limits.
constexpr uint32_t kNormalsStride = SABERCAT_STRIDE_NORMALS;
constexpr uint32_t kBaseColorRoughnessStride = SABERCAT_STRIDE_BASE_COLOR_ROUGHNESS;
constexpr uint32_t kMetallicStride = SABERCAT_STRIDE_METALLIC;
constexpr uint32_t kVisibilityStride = SABERCAT_STRIDE_VISIBILITY;
constexpr uint32_t kVisibilityCoefficients = SABERCAT_VISIBILITY_COEFFS;
constexpr uint32_t kMaxLights = SABERCAT_MAX_LIGHTS;
constexpr float kDefaultRoughness = SABERCAT_DEFAULT_ROUGHNESS;

constexpr float kMinDepth = SABERCAT_MIN_DEPTH;
constexpr float kFrustumClamp = SABERCAT_FRUSTUM_CLAMP;
constexpr float kCovarianceDilation = SABERCAT_COVARIANCE_DILATION;
constexpr float kRadiusSigmas = SABERCAT_RADIUS_SIGMAS;
// The viewer slider's starting value, NOT an applied default -- the cull is
// off unless LODParams::minScreenRadius is set. See config.slangh.
constexpr float kDefaultMinScreenRadius = SABERCAT_DEFAULT_MIN_SCREEN_RADIUS;
constexpr float kMaxAlpha = SABERCAT_MAX_ALPHA;

// CameraUniforms::renderFlags bits -- see config.slangh.
constexpr uint32_t kRenderFlagAntialiased = SABERCAT_RENDER_FLAG_ANTIALIASED;
constexpr uint32_t kRenderFlagSaveBackwardState = SABERCAT_RENDER_FLAG_SAVE_BACKWARD_STATE;

// Spherical-harmonic basis constants (see config.slangh for the coefficient
// ordering these assume). Bands 1-3 are consumed by forward_common.slangh's
// evaluateSH and mirrored by its C++ verification reference.
constexpr float kShC0 = SABERCAT_SH_C0;
constexpr float kShC1 = SABERCAT_SH_C1;
constexpr float kShC2_0 = SABERCAT_SH_C2_0;
constexpr float kShC2_1 = SABERCAT_SH_C2_1;
constexpr float kShC2_2 = SABERCAT_SH_C2_2;
constexpr float kShC2_3 = SABERCAT_SH_C2_3;
constexpr float kShC2_4 = SABERCAT_SH_C2_4;
constexpr float kShC3_0 = SABERCAT_SH_C3_0;
constexpr float kShC3_1 = SABERCAT_SH_C3_1;
constexpr float kShC3_2 = SABERCAT_SH_C3_2;
constexpr float kShC3_3 = SABERCAT_SH_C3_3;
constexpr float kShC3_4 = SABERCAT_SH_C3_4;
constexpr float kShC3_5 = SABERCAT_SH_C3_5;
constexpr float kShC3_6 = SABERCAT_SH_C3_6;

// Number of workgroups needed to cover `count` items at `groupSize` each.
constexpr uint32_t dispatchGroups(uint32_t count, uint32_t groupSize)
{
    return (count + groupSize - 1) / groupSize;
}

} // namespace sabercat
