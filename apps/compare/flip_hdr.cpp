// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Input helpers for HDR-FLIP and the display transform (flip_hdr.h). These need
// no FLIP.h, so this file is plain C++; the FLIP calls are in flip_metric.cpp.

#include "flip_hdr.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>

namespace sabercat::compare
{

namespace
{

std::string describe(const ImageData & image)
{
    return std::to_string(image.width) + "x" + std::to_string(image.height);
}

// Pixel count of a valid image, from the validated dimensions.
size_t pixelCount(const ImageData & image)
{
    return static_cast<size_t>(image.width) * static_cast<size_t>(image.height);
}

} // namespace

const char * toneMapperName(ToneMapper toneMapper)
{
    switch (toneMapper)
    {
    case ToneMapper::Aces:
        return "aces";
    case ToneMapper::Hable:
        return "hable";
    case ToneMapper::Reinhard:
        return "reinhard";
    }
    return "aces";
}

bool parseToneMapper(const std::string & text, ToneMapper & out)
{
    std::string lower = text;
    for (char & ch : lower)
    {
        if (ch >= 'A' && ch <= 'Z')
        {
            ch = static_cast<char>(ch - 'A' + 'a');
        }
    }
    if (lower == "aces")
    {
        out = ToneMapper::Aces;
        return true;
    }
    if (lower == "hable")
    {
        out = ToneMapper::Hable;
        return true;
    }
    if (lower == "reinhard")
    {
        out = ToneMapper::Reinhard;
        return true;
    }
    return false;
}

void validateImageData(const ImageData & image)
{
    if (image.width == 0 || image.height == 0)
    {
        throw std::invalid_argument("image has zero width or height: " + describe(image));
    }
    // Width and height are 32-bit, so their product fits in 64 bits.
    const uint64_t pixels = static_cast<uint64_t>(image.width) * static_cast<uint64_t>(image.height);
    if (pixels > kMaxSamples / 3)
    {
        throw std::invalid_argument("image " + describe(image) + " is too large for the compare tools");
    }
    const size_t expected = static_cast<size_t>(pixels) * 4;
    if (image.rgba.size() != expected)
    {
        throw std::invalid_argument("image " + describe(image) + " has " + std::to_string(image.rgba.size()) +
            " samples, expected " + std::to_string(expected));
    }
}

uint64_t countNonFiniteRgb(const ImageData & image)
{
    validateImageData(image);
    const size_t pixels = pixelCount(image);
    const float * src = image.rgba.data();
    uint64_t count = 0;
    for (size_t p = 0; p < pixels; ++p)
    {
        const float * pixel = src + 4 * p;
        // Alpha is not compared, so it is not counted.
        for (size_t c = 0; c < 3; ++c)
        {
            if (!std::isfinite(pixel[c]))
            {
                ++count;
            }
        }
    }
    return count;
}

std::vector<float> linearRgb(const ImageData & image)
{
    const uint64_t nonFinite = countNonFiniteRgb(image);
    if (nonFinite != 0)
    {
        throw std::invalid_argument("image " + describe(image) + " has " + std::to_string(nonFinite) +
            " non-finite samples (NaN or infinity); HDR-FLIP cannot evaluate them");
    }

    const size_t pixels = pixelCount(image);
    std::vector<float> linear(pixels * 3);
    const float * src = image.rgba.data();
    float * dst = linear.data();

    if (image.encoding == Encoding::Linear)
    {
        for (size_t p = 0; p < pixels; ++p)
        {
            for (size_t c = 0; c < 3; ++c)
            {
                dst[3 * p + c] = src[4 * p + c];
            }
        }
        return linear;
    }

    // Display: decoded through the same table flip() uses, with the same clamp.
    const float * table = srgbToLinearTable();
    for (size_t p = 0; p < pixels; ++p)
    {
        for (size_t c = 0; c < 3; ++c)
        {
            const float v = std::min(1.0f, std::max(0.0f, src[4 * p + c]));
            dst[3 * p + c] = table[static_cast<int>(v * 65535.0f + 0.5f)];
        }
    }
    return linear;
}

Image displayRgb(const ImageData & image)
{
    const uint64_t nonFinite = countNonFiniteRgb(image);
    if (nonFinite != 0)
    {
        throw std::invalid_argument("image " + describe(image) + " has " + std::to_string(nonFinite) +
            " non-finite samples (NaN or infinity)");
    }

    const size_t pixels = pixelCount(image);
    Image out;
    out.width = image.width;
    out.height = image.height;
    out.rgb.resize(pixels * 3);
    const float * src = image.rgba.data();
    float * dst = out.rgb.data();
    const bool linear = image.encoding == Encoding::Linear;

    for (size_t p = 0; p < pixels; ++p)
    {
        for (size_t c = 0; c < 3; ++c)
        {
            float v = src[4 * p + c];
            if (linear)
            {
                v = std::min(1.0f, std::max(0.0f, v));
                // The sRGB transfer function (IEC 61966-2-1).
                v = v <= 0.0031308f ? v * 12.92f : 1.055f * std::pow(v, 1.0f / 2.4f) - 0.055f;
            }
            dst[3 * p + c] = v;
        }
    }
    return out;
}

double linearMse(const std::vector<float> & a, const std::vector<float> & b)
{
    if (a.size() != b.size())
    {
        throw std::invalid_argument("linearMse: sample counts differ (" + std::to_string(a.size()) + " and " +
            std::to_string(b.size()) + ")");
    }
    double sum = 0.0;
    size_t count = 0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (std::isfinite(a[i]) && std::isfinite(b[i]))
        {
            const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
            sum += d * d;
            ++count;
        }
    }
    if (count == 0)
    {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return sum / static_cast<double>(count);
}

} // namespace sabercat::compare
