// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <limits>

namespace sabercat
{

// Which order the forward rasterizer blends the Gaussians of a tile in. Default
// is the shipped depth-sorted path; the others are opt-in variants. Differentiable
// runs always use Default, because the backward pass assumes that order.
enum class SortMode : uint32_t
{
    Default = 0,
    Stable = 1,
    TileRayDepth = 2,
    PixelWindow = 3,
    Hybrid = 4,
    Stochastic = 5,
};

// Knobs for the non-default sort modes. Every field is 4 bytes, so the struct
// has no padding and can be compared with memcmp. Fields that a mode does not
// read are still validated, so a bad value is caught when it is set.
struct SortSettings
{
    SortMode mode = SortMode::Default;
    uint32_t pixelWindow = 16;          // PixelWindow: 16 or 32; Hybrid uses 8
    uint32_t stableBucketsPerOctave = 1024;
    uint32_t stochasticSamples = 4;     // 4 or 8
    // Stochastic temporal accumulation. Mode 0 keeps a running mean of the frames
    // while the camera is static, capped at temporalFrames, and restarts it on
    // motion. Mode 1 is a running mean up to the point where 1/k reaches
    // temporalEmaWeight, then an exponential average with that weight; during
    // motion the weight is temporalEmaWeight too, and the image is not restarted.
    // The cap and the EMA floor keep the history adapting instead of freezing.
    uint32_t temporalFrames = 32;       // Stochastic: running-mean cap, k <= temporalFrames
    uint32_t temporalMode = 0;          // 0 mean, restart on motion; 1 mean floored at temporalEmaWeight
    float stableHysteresis = 1.0f;      // in buckets
    float temporalEmaWeight = 0.2f;     // mode 1: the floor of the mean, and the weight during motion
};
static_assert(sizeof(SortSettings) == 32, "SortSettings must stay padding-free");

// Per-stage GPU time of one forward run, in milliseconds. Every field is NaN
// and valid is false when timing is off, unsupported by the queue, or the run
// did not stamp (a capacity re-run, or a run that stopped early).
struct ForwardStageTimings
{
    float project = std::numeric_limits<float>::quiet_NaN();
    float scan = std::numeric_limits<float>::quiet_NaN();
    float binning = std::numeric_limits<float>::quiet_NaN();
    float sortKeys = std::numeric_limits<float>::quiet_NaN();
    float sort = std::numeric_limits<float>::quiet_NaN();
    float ranges = std::numeric_limits<float>::quiet_NaN();
    float rasterize = std::numeric_limits<float>::quiet_NaN();
    float resolve = std::numeric_limits<float>::quiet_NaN();
    float total = std::numeric_limits<float>::quiet_NaN();
    bool valid = false;
};

// Throws std::invalid_argument naming the offending field if any value is out
// of range for its mode. Called by every setter that stores sort settings.
void validateSortSettings(const SortSettings & settings);

// Stable, lower-case name of a mode, for logs and error messages.
const char * sortModeName(SortMode mode);

// True for the modes that only change the color order. The relight G-buffer
// pass (gbuffer_rasterize) still uses the default order, so lighting with one
// of these modes would disagree with the lit image.
bool sortModeIncompatibleWithLighting(SortMode mode);

} // namespace sabercat
