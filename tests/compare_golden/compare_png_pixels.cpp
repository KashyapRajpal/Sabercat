// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Decoded-pixel comparison of two PNG files for the golden test (run_golden.cmake).
// Both files are loaded with sabercat::compare::loadImage, then the width, height,
// bit depth and every RGBA float sample are compared exactly. The samples are derived
// from integers, so an exact compare is the right test; the bytes of the files may
// still differ (the golden test checks those separately).
//
// Usage: sabercat_compare_png_pixels <a.png> <b.png>
// Output: one line on stdout, "identical <width>x<height>" or "differ <reason>".
// Exit status: 0 identical, 1 differ, 2 usage or load error (printed as "error <what>").
#include "image_data.h"

#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

namespace
{

// Returns the reason the two images differ, or an empty string when they are identical.
// Both images are expected to come from loadImage, so rgba holds width * height * 4 samples.
std::string differenceReason(const sabercat::compare::ImageData & a,
                             const sabercat::compare::ImageData & b)
{
    if (a.width != b.width || a.height != b.height)
    {
        std::ostringstream out;
        out << "size " << a.width << 'x' << a.height << " vs " << b.width << 'x' << b.height;
        return out.str();
    }
    if (a.bitDepth != b.bitDepth)
    {
        std::ostringstream out;
        out << "bit depth " << a.bitDepth << " vs " << b.bitDepth;
        return out.str();
    }
    if (a.rgba.size() != b.rgba.size())
    {
        std::ostringstream out;
        out << "sample count " << a.rgba.size() << " vs " << b.rgba.size();
        return out.str();
    }

    // Walk the samples in row-major order (top row first, four samples per pixel). The
    // first difference is reported with its pixel and channel; every difference is counted.
    const std::size_t sampleCount = a.rgba.size();
    std::size_t firstIndex = sampleCount;
    std::size_t differing = 0;
    for (std::size_t i = 0; i < sampleCount; ++i)
    {
        if (a.rgba[i] != b.rgba[i])
        {
            if (differing == 0)
            {
                firstIndex = i;
            }
            ++differing;
        }
    }
    if (differing == 0)
    {
        return std::string();
    }

    const std::size_t width = a.width;
    const std::size_t pixel = firstIndex / 4;
    const std::size_t channel = firstIndex % 4;
    const std::size_t x = pixel % width;
    const std::size_t y = pixel / width;

    std::ostringstream out;
    out << std::setprecision(9);
    out << "first difference at pixel (" << x << ',' << y << ") channel " << channel
        << ": " << a.rgba[firstIndex] << " vs " << b.rgba[firstIndex]
        << ", " << differing << " samples differ";
    return out.str();
}

} // namespace

int main(int argc, char ** argv)
{
    if (argc != 3)
    {
        std::cerr << "usage: sabercat_compare_png_pixels <a.png> <b.png>\n";
        return 2;
    }

    try
    {
        const sabercat::compare::ImageData a = sabercat::compare::loadImage(argv[1]);
        const sabercat::compare::ImageData b = sabercat::compare::loadImage(argv[2]);

        const std::string reason = differenceReason(a, b);
        if (reason.empty())
        {
            std::cout << "identical " << a.width << 'x' << a.height << '\n';
            return 0;
        }
        std::cout << "differ " << reason << '\n';
        return 1;
    }
    catch (const std::exception & error)
    {
        std::cout << "error " << error.what() << '\n';
        return 2;
    }
}
