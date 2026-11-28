// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Blue-noise rank table and the per-frame and per-splat hashes used by the sort
// study's stochastic transparency. The CPU reference and the GPU host must agree on
// these values exactly, so they are defined once here. Internal to the renderer.
#pragma once

#include <cstdint>
#include <vector>

namespace sabercat
{

constexpr uint32_t kBlueNoiseSize = 64; // the rank table is 64 x 64

// Deterministic void-and-cluster rank table: a permutation of 0..4095, index y * 64 + x.
// Returns the same values on every call, and in Debug and Release builds alike.
std::vector<uint16_t> generateBlueNoiseRanks();

// fract(double(frame) * 0.6180339887498949) cast to float, in [0,1). The golden-ratio
// step spreads consecutive frames evenly over the unit interval.
float stochasticFrameOffset(uint32_t frame);

// lowbias32 integer hash. It is a bijection on 32-bit values; the shader's
// stochasticHash in shaders/sort_common.slangh computes the same function.
uint32_t stochasticHash(uint32_t x);

} // namespace sabercat
