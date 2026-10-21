// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>

namespace sabercat
{

// Approximate floating-point comparison, shared by tests, verification
// passes, and anywhere else that needs to compare computed values against
// expected ones.
//
// The tolerance is INCLUSIVE, so an epsilon of 0 means "bit-exact" rather
// than "never equal" -- a strict `<` makes fabs(a - b) < 0.0f false even
// for identical values, which is a genuinely easy trap to fall into when
// you want an exact check.
inline bool approxEqual(float a, float b, float epsilon = 1e-4f)
{
    return std::fabs(a - b) <= epsilon;
}

inline bool approxEqual(const glm::vec2 & a, const glm::vec2 & b, float epsilon = 1e-4f)
{
    return approxEqual(a.x, b.x, epsilon) && approxEqual(a.y, b.y, epsilon);
}

inline bool approxEqual(const glm::vec3 & a, const glm::vec3 & b, float epsilon = 1e-4f)
{
    return approxEqual(a.x, b.x, epsilon) && approxEqual(a.y, b.y, epsilon)
           && approxEqual(a.z, b.z, epsilon);
}

inline bool approxEqual(const glm::vec4 & a, const glm::vec4 & b, float epsilon = 1e-4f)
{
    return approxEqual(a.x, b.x, epsilon) && approxEqual(a.y, b.y, epsilon)
           && approxEqual(a.z, b.z, epsilon) && approxEqual(a.w, b.w, epsilon);
}

// Compares only the xyz lanes of a padded vec4. The canonical Gaussian
// streams pad xyz data to a 16-byte stride (see gaussian_cloud.h), and the
// w lane is unused padding rather than data -- comparing it would fail on
// whatever happens to be in the pad.
inline bool approxEqualXYZ(const glm::vec4 & padded, const glm::vec3 & expected,
                           float epsilon = 1e-4f)
{
    return approxEqual(glm::vec3(padded), expected, epsilon);
}

// Relative-tolerance comparison, for values whose magnitude varies wildly
// between fields being checked in the same test (e.g. a rasterizer conic
// around 0.01 next to a radius around 30 -- one absolute epsilon can't fit
// both). Falls back to absEpsilon near zero so it doesn't demand
// impossible precision when b is 0.
inline bool approxEqualRelative(float a, float b, float relEpsilon = 1e-3f,
                                float absEpsilon = 1e-5f)
{
    return std::fabs(a - b) <= std::max(absEpsilon, relEpsilon * std::fabs(b));
}

// Largest absolute difference between two equally-sized float ranges.
// Reports HOW FAR off a comparison is rather than just that it failed,
// which is what you want when diffing a rendered image against a reference:
// "max error 0.31" localizes a bug far better than "images differ".
inline float maxAbsDifference(const float * a, const float * b, size_t count)
{
    float worst = 0.0f;
    for (size_t i = 0; i < count; ++i)
    {
        worst = std::fmax(worst, std::fabs(a[i] - b[i]));
    }
    return worst;
}

} // namespace sabercat
