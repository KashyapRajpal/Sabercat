// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "image_compare.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace sabercat::compare
{

namespace
{

// Independent accumulators for the squared-error sum. A fixed-width inner loop
// over eight floats lets the compiler vectorize without fast-math, and the
// order of the additions is fixed, so the result is deterministic.
constexpr size_t kLanes = 8;

std::string describe(const Image & image)
{
    return std::to_string(image.width) + "x" + std::to_string(image.height);
}

// Both images valid and of equal size.
void validatePair(const Image & reference, const Image & test)
{
    validate(reference);
    validate(test);
    if (reference.width != test.width || reference.height != test.height)
    {
        throw std::invalid_argument("image sizes differ: reference is " + describe(reference) +
            ", test is " + describe(test));
    }
}

// Sum of (a - b)^2 over count contiguous samples.
double sumSquaredDifference(const float * a, const float * b, size_t count)
{
    std::array<float, kLanes> lane{};
    const size_t blocked = count - count % kLanes;
    for (size_t i = 0; i < blocked; i += kLanes)
    {
        for (size_t l = 0; l < kLanes; ++l)
        {
            const float d = a[i + l] - b[i + l];
            lane[l] += d * d;
        }
    }
    // The tail starts at a multiple of kLanes, so i & (kLanes - 1) is its lane.
    for (size_t i = blocked; i < count; ++i)
    {
        const float d = a[i] - b[i];
        lane[i & (kLanes - 1)] += d * d;
    }
    double sum = 0.0;
    for (const float v : lane)
    {
        sum += v;
    }
    return sum;
}

} // namespace

void validate(const Image & image)
{
    if (image.width == 0 || image.height == 0)
    {
        throw std::invalid_argument("image has zero width or height: " + describe(image));
    }

    // Count pixels first, so that width * height * 3 cannot wrap a size_t.
    const size_t pixels = static_cast<size_t>(image.width) * image.height;
    if (pixels > kMaxSamples / 3)
    {
        throw std::invalid_argument("image " + describe(image) + " has more than kMaxSamples samples");
    }

    const size_t expected = pixels * 3;
    if (image.rgb.size() != expected)
    {
        throw std::invalid_argument("image " + describe(image) + " has " + std::to_string(image.rgb.size()) +
            " samples, expected " + std::to_string(expected));
    }
}

PsnrResult psnr(const Image & reference, const Image & test)
{
    validatePair(reference, test);

    const size_t rowSamples = static_cast<size_t>(reference.width) * 3;
    double total = 0.0;
    for (uint32_t y = 0; y < reference.height; ++y)
    {
        const size_t offset = static_cast<size_t>(y) * rowSamples;
        total += sumSquaredDifference(reference.rgb.data() + offset, test.rgb.data() + offset, rowSamples);
    }

    // A NaN or infinite sample makes the total non-finite, so one check covers
    // the whole image.
    if (!std::isfinite(total))
    {
        throw std::invalid_argument("non-finite sample");
    }

    PsnrResult result;
    result.mse = total / static_cast<double>(reference.rgb.size());
    result.psnrDb = result.mse == 0.0 ? std::numeric_limits<double>::infinity()
                                      : 10.0 * std::log10(1.0 / result.mse);
    return result;
}

std::vector<float> absoluteDifferenceMap(const Image & reference, const Image & test)
{
    validatePair(reference, test);

    const size_t pixels = static_cast<size_t>(reference.width) * reference.height;
    std::vector<float> map(pixels);
    const float * a = reference.rgb.data();
    const float * b = test.rgb.data();
    float * out = map.data();

    double total = 0.0;
    for (size_t p = 0; p < pixels; ++p)
    {
        const float * pa = a + 3 * p;
        const float * pb = b + 3 * p;
        const float mean = (std::fabs(pa[0] - pb[0]) + std::fabs(pa[1] - pb[1]) + std::fabs(pa[2] - pb[2])) / 3.0f;
        total += mean;
        // std::min(1.0f, NaN) is 1.0f, so the NaN is caught by the total below.
        out[p] = std::min(1.0f, mean);
    }

    if (!std::isfinite(total))
    {
        throw std::invalid_argument("non-finite sample");
    }
    return map;
}

const float * srgbToLinearTable()
{
    static const std::vector<float> table = []
    {
        std::vector<float> values(65536);
        for (size_t k = 0; k < values.size(); ++k)
        {
            const double v = static_cast<double>(k) / 65535.0;
            values[k] = static_cast<float>(v <= 0.04045 ? v / 12.92 : std::pow((v + 0.055) / 1.055, 2.4));
        }
        return values;
    }();
    return table.data();
}

} // namespace sabercat::compare
