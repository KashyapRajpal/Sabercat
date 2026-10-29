// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>

namespace sabercat
{

class Device;
class GaussianCloud;

// Reads the antimatter15 `.splat` binary format into a GaussianCloud.
//
// Format: 32 bytes per record (headerless), little-endian floats
//   Offset  Size  Field
//   ------  ----  -----
//   0-11    12    Position (3×f32: x, y, z)
//   12-23   12    Scale (3×f32: sx, sy, sz) — already linear, no exp needed
//   24-27    4    Color (4×u8: R, G, B, A) — RGBA, DC-only SH
//   28-31    4    Rotation (4×u8: w, x, y, z) — reorder to xyzw, normalize
//
// The color and opacity are inverted from their display forms to recover
// the raw Gaussian parameters; see the implementation for the exact formulas.
// SH degrees 1-3 are absent (DC-only), so all higher coefficients are zero.
class SplatReader
{
  public:
    static std::shared_ptr<GaussianCloud> read(Device & device, const std::string & path);
};

} // namespace sabercat
