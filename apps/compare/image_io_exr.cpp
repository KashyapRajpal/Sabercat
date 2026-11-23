// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// OpenEXR input through tinyexr v3.4.0 (BSD-3-Clause, pinned in apps/compare/CMakeLists.txt).
// This is the one file that defines the library's implementation; its ZIP decoding
// uses miniz, whose source (deps/miniz, MIT) is in the same tarball and is compiled
// into sabercat_compare_core. The library's C API is used: the header is parsed first,
// and multi-part, deep and tiled files are refused before any pixel is decoded.

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

// Single-threaded, no OpenMP, and miniz for ZIP (its miniz.h is on the include path).
#define TINYEXR_USE_OPENMP 0
#define TINYEXR_USE_THREAD 0
#define TINYEXR_USE_MINIZ 1
#define TINYEXR_IMPLEMENTATION
#include <tinyexr.h>

#include "image_io.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace sabercat::compare
{

namespace
{

// Owns an EXRHeader. InitEXRHeader zeroes it, so freeing it after a failed parse is safe.
class HeaderGuard
{
public:
    HeaderGuard() { InitEXRHeader(&header); }
    ~HeaderGuard() { FreeEXRHeader(&header); }
    HeaderGuard(const HeaderGuard &) = delete;
    HeaderGuard & operator=(const HeaderGuard &) = delete;

    EXRHeader header;
};

// Owns an EXRImage and its decoded planes.
class ImageGuard
{
public:
    ImageGuard() { InitEXRImage(&image); }
    ~ImageGuard() { FreeEXRImage(&image); }
    ImageGuard(const ImageGuard &) = delete;
    ImageGuard & operator=(const ImageGuard &) = delete;

    EXRImage image;
};

[[noreturn]] void fail(const std::string & utf8Path, const std::string & reason)
{
    throw std::runtime_error(utf8Path + ": " + reason);
}

// Throws with tinyexr's own reason. The message belongs to tinyexr and is freed here.
[[noreturn]] void failTinyexr(const std::string & utf8Path, const char * what, const char * tinyexrError)
{
    std::string reason = tinyexrError != nullptr ? tinyexrError : "no detail";
    if (tinyexrError != nullptr)
    {
        FreeEXRErrorMessage(tinyexrError);
    }
    while (!reason.empty() && (reason.back() == '\n' || reason.back() == '\r' || reason.back() == ' '))
    {
        reason.pop_back();
    }
    throw std::runtime_error(utf8Path + ": " + what + " (" + reason + ")");
}

// Index of the channel with this exact name, or -1.
int findChannel(const EXRHeader & header, const char * name)
{
    for (int c = 0; c < header.num_channels; ++c)
    {
        if (std::strncmp(header.channels[c].name, name, sizeof(header.channels[c].name)) == 0)
        {
            return c;
        }
    }
    return -1;
}

} // namespace

ImageData loadExrRgba(const std::string & utf8Path)
{
    const std::vector<unsigned char> bytes = readFileBytes(utf8Path, kMaxFileBytes);
    const unsigned char * const data = bytes.data();
    const size_t size = bytes.size();

    EXRVersion version = {};
    if (ParseEXRVersionFromMemory(&version, data, size) != TINYEXR_SUCCESS)
    {
        fail(utf8Path, "not a valid OpenEXR file");
    }
    if (version.multipart)
    {
        fail(utf8Path, "multi-part EXR is not supported");
    }
    if (version.non_image)
    {
        fail(utf8Path, "deep EXR is not supported");
    }
    if (version.tiled)
    {
        fail(utf8Path, "tiled EXR is not supported");
    }

    HeaderGuard guard;
    EXRHeader & header = guard.header;
    const char * error = nullptr;
    if (ParseEXRHeaderFromMemory(&header, &version, data, size, &error) != TINYEXR_SUCCESS)
    {
        failTinyexr(utf8Path, "invalid OpenEXR header", error);
    }

    // R, G and B are required; A is optional. Every other channel is ignored, but
    // tinyexr still decodes it.
    const int indexR = findChannel(header, "R");
    const int indexG = findChannel(header, "G");
    const int indexB = findChannel(header, "B");
    const int indexA = findChannel(header, "A");
    if (indexR < 0)
    {
        fail(utf8Path, "missing channel R");
    }
    if (indexG < 0)
    {
        fail(utf8Path, "missing channel G");
    }
    if (indexB < 0)
    {
        fail(utf8Path, "missing channel B");
    }

    const int used[4] = {indexR, indexG, indexB, indexA};
    const char * const usedNames[4] = {"R", "G", "B", "A"};
    bool allHalf = true;
    for (int i = 0; i < 4; ++i)
    {
        const int c = used[i];
        if (c < 0)
        {
            continue;
        }
        const int pixelType = header.pixel_types[c];
        if (pixelType != TINYEXR_PIXELTYPE_HALF && pixelType != TINYEXR_PIXELTYPE_FLOAT)
        {
            fail(utf8Path, std::string("channel ") + usedNames[i] + " is not half or float");
        }
        if (header.channels[c].x_sampling != 1 || header.channels[c].y_sampling != 1)
        {
            fail(utf8Path, std::string("subsampled channel ") + usedNames[i] + " is not supported");
        }
        if (pixelType != TINYEXR_PIXELTYPE_HALF)
        {
            allHalf = false;
        }
    }

    // Dimensions and sample counts, checked before anything is decoded.
    const int64_t dataWidth = static_cast<int64_t>(header.data_window.max_x) - header.data_window.min_x + 1;
    const int64_t dataHeight = static_cast<int64_t>(header.data_window.max_y) - header.data_window.min_y + 1;
    if (dataWidth <= 0 || dataHeight <= 0)
    {
        fail(utf8Path, "invalid OpenEXR data window");
    }
    const size_t rgbaCount = rgbaSampleCount(static_cast<uint64_t>(dataWidth), static_cast<uint64_t>(dataHeight), utf8Path);
    // tinyexr decodes every channel, so the samples of all of them are bounded too.
    const uint64_t decodedSamples = static_cast<uint64_t>(dataWidth) * static_cast<uint64_t>(dataHeight)
        * static_cast<uint64_t>(header.num_channels);
    if (decodedSamples > kMaxSamples)
    {
        fail(utf8Path, "image has too many samples for the compare tools");
    }

    // Half channels are decoded as float, so every used plane is float.
    if (header.requested_pixel_types == nullptr)
    {
        fail(utf8Path, "OpenEXR header has no pixel types");
    }
    for (int c = 0; c < header.num_channels; ++c)
    {
        if (header.pixel_types[c] == TINYEXR_PIXELTYPE_HALF)
        {
            header.requested_pixel_types[c] = TINYEXR_PIXELTYPE_FLOAT;
        }
    }

    ImageGuard decoded;
    if (LoadEXRImageFromMemory(&decoded.image, &header, data, size, &error) != TINYEXR_SUCCESS)
    {
        failTinyexr(utf8Path, "could not decode the OpenEXR pixels", error);
    }
    const EXRImage & image = decoded.image;
    if (image.images == nullptr || image.num_channels != header.num_channels
        || static_cast<int64_t>(image.width) != dataWidth || static_cast<int64_t>(image.height) != dataHeight)
    {
        fail(utf8Path, "OpenEXR decoder returned an unexpected image");
    }

    // Scanline files: tinyexr stores rows from the data window's min_y, the top row
    // in OpenEXR's convention, so row 0 here is the first stored row. Nothing is flipped.
    const auto plane = [&image, &utf8Path](int c) -> const float *
    {
        if (c < 0)
        {
            return nullptr;
        }
        if (image.images[c] == nullptr)
        {
            fail(utf8Path, "OpenEXR decoder returned no pixels for a channel");
        }
        return reinterpret_cast<const float *>(image.images[c]);
    };
    const float * const planeR = plane(indexR);
    const float * const planeG = plane(indexG);
    const float * const planeB = plane(indexB);
    const float * const planeA = plane(indexA);

    ImageData out;
    out.width = static_cast<uint32_t>(dataWidth);
    out.height = static_cast<uint32_t>(dataHeight);
    out.rgba.resize(rgbaCount);
    const size_t pixels = static_cast<size_t>(dataWidth * dataHeight);
    for (size_t i = 0; i < pixels; ++i)
    {
        float * const px = out.rgba.data() + i * 4;
        px[0] = planeR[i];
        px[1] = planeG[i];
        px[2] = planeB[i];
        px[3] = planeA != nullptr ? planeA[i] : 1.0f;
    }

    // NaN and infinities are kept as they are and counted, not clamped.
    uint64_t nonFinite = 0;
    for (const float value : out.rgba)
    {
        if (!std::isfinite(value))
        {
            ++nonFinite;
        }
    }
    out.nonFiniteCount = nonFinite;
    out.bitDepth = allHalf ? 16u : 32u;
    out.encoding = Encoding::Linear;
    out.hasAlpha = indexA >= 0;
    return out;
}

} // namespace sabercat::compare
