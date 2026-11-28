// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// CPU reference renderer for the sort study. It blends the same per-tile
// candidate lists the GPU rasterizer sees, but orders each pixel's contributors
// by a chosen rule, so the study can measure what the sort order alone costs.
// Pure CPU: the only Vulkan-side types are the header structs it reads.
#pragma once

#include <glm/glm.hpp>

#include <cstdint>
#include <vector>

#include "sabercat/forward_pass.h"

namespace sabercat::study
{

// Everything the CPU renderer needs from one captured frame. Filled by
// captureCandidates() (study_capture.h) or built by hand in tests.
struct StudyCandidates
{
    // Grouped by size (4-byte fields first) so the struct carries no padding.
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t tilesX = 0;
    uint32_t tilesY = 0;
    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    std::vector<ProjectedGaussian> projected; // per Gaussian
    std::vector<uint32_t> sortedValues;       // the default sorted list, lastInstanceCount entries
    std::vector<TileRange> tileRanges;        // tilesX * tilesY
    std::vector<float> rayForm;               // 12 floats per Gaussian; empty => center depth
};

// TileOrder reproduces the GPU rasterizer's per-tile list order. ExactRayDepth
// sorts every pixel's contributors by ray depth. WindowRayDepth models a bounded
// per-pixel resort buffer of `window` entries, which is what the GPU kernels
// implement.
enum class ReferenceOrder
{
    TileOrder,
    ExactRayDepth,
    WindowRayDepth,
};

struct ReferenceOptions
{
    ReferenceOrder order = ReferenceOrder::ExactRayDepth;
    uint32_t window = 16; // WindowRayDepth only: buffer capacity K (>= 1)
};

// Peak depth along the ray (dx, dy, -1) from a 12-float whitened ray-depth form
// (columns c0, c1, c2 of P, then u), in double arithmetic. Falls back to
// fallbackDepth when the form is null, degenerate or yields a non-finite depth.
// The null case is an extension of the spec, so a caller can pass
// &rayForm[12 * i] or nullptr without branching.
double evalRayDepth(const float * form12, double fallbackDepth, double dx, double dy);

// Row-major width*height RGBA, top row first. rgb is the accumulated color and
// a = 1 - T, so a pixel no Gaussian reaches is (0, 0, 0, 0). Throws
// std::invalid_argument when the candidates are inconsistent (tile count,
// ranges, ray form size, non-positive focal length for a ray-depth order, or a
// zero window).
std::vector<glm::vec4> renderReference(const StudyCandidates & c, const ReferenceOptions & o);

// Stochastic transparency (the ST4 and ST8 modes). samples is the shader's
// SABERCAT_ST_SAMPLES (4 or 8); frameIndex selects the noise frame whose offset
// the shader receives, so the model can be checked frame by frame.
struct StochasticOptions
{
    uint32_t samples = 4;
    uint32_t frameIndex = 0;
};

// One raw frame of the stochastic rasterizer, in float arithmetic that mirrors
// tile_rasterize_stochastic operation for operation. ranks is the 64 x 64
// blue-noise rank table (generateBlueNoiseRanks), row-major, a permutation of
// 0..4095. Output is row-major width*height premultiplied RGBA: rgb is the mean
// color over the samples, a is the covered fraction. Throws
// std::invalid_argument for a ranks table that is not 4096 entries, samples
// other than 4 or 8, a non-positive focal length, or inconsistent candidates.
std::vector<glm::vec4> renderStochasticModel(const StudyCandidates & c, const std::vector<uint16_t> & ranks,
                                             const StochasticOptions & o);

} // namespace sabercat::study
