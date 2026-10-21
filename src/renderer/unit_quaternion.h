// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cmath>

// Internal to the renderer (not installed): the unit-length rule for rotations,
// shared by ProjectIO, the layer transform bake and GLTFWriter so all three
// refuse the same broken rotations.
namespace sabercat
{

// Float round-off from the application's own quaternion math stays far inside
// this; anything further from unit length is a broken rotation.
constexpr double kUnitLengthTolerance = 1e-4;

// The Euclidean length of a quaternion given as (w, x, y, z), in double.
inline double quaternionLength(float w, float x, float y, float z)
{
    return std::sqrt(static_cast<double>(w) * w + static_cast<double>(x) * x
                     + static_cast<double>(y) * y + static_cast<double>(z) * z);
}

// True when the quaternion is within kUnitLengthTolerance of unit length. A
// non-finite quaternion is never unit length.
inline bool isUnitLength(float w, float x, float y, float z)
{
    return std::fabs(quaternionLength(w, x, y, z) - 1.0) <= kUnitLengthTolerance;
}

} // namespace sabercat
