// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

// The exact sRGB transfer functions (IEC 61966-2-1) in double precision: the
// host mirror of shaders/color_space.slangh, for the tests' CPU references.
// Plain C, so the C API test can include it too.

#include <math.h>

static inline double srgbToLinear(double c)
{
    return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4);
}

static inline double linearToSrgb(double l)
{
    return l <= 0.0031308 ? 12.92 * l : 1.055 * pow(l, 1.0 / 2.4) - 0.055;
}
