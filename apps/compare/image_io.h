// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Image input and output: PNG through the Windows Imaging Component (WIC), and the
// RGBA loaders behind loadImage() (image_data.h).
#pragma once

#include "image_compare.h"
#include "image_data.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sabercat::compare
{

// Largest file any loader reads whole into memory.
inline constexpr uint64_t kMaxFileBytes = 0x7FFFFFFFu;

// Decodes a PNG (8 or 16 bits, gray, palette, RGB or with alpha) to
// display-encoded float RGB in [0,1]. Alpha is dropped. Pixel values are the
// file's encoded values divided by the maximum; no gamma conversion is applied.
// Path is UTF-8. Throws std::runtime_error (with the HRESULT) on failure.
Image loadPng(const std::string & utf8Path);

// Writes a 24-bit RGB PNG. Throws std::runtime_error on failure.
void savePngRgb8(const std::string & utf8Path, const Rgb8Image & image);

// Element count of a width x height RGBA buffer (4 floats per pixel). Throws
// std::runtime_error on a zero size or when the count exceeds kMaxSamples. The
// product is computed without overflow, so it is safe before any allocation.
size_t rgbaSampleCount(uint64_t width, uint64_t height, const std::string & utf8Path);

// The whole file, read with std::ifstream on the UTF-8 path. Throws
// std::runtime_error when it is missing, empty, unreadable or larger than maxBytes.
std::vector<unsigned char> readFileBytes(const std::string & utf8Path, uint64_t maxBytes);

// Per-format loaders. Each throws std::runtime_error naming the file.
ImageData loadWicRgba(const std::string & utf8Path);   // PNG, JPEG (Windows only)
ImageData loadWebpRgba(const std::string & utf8Path);  // WebP (image_io_webp.cpp)
ImageData loadExrRgba(const std::string & utf8Path);   // OpenEXR (image_io_exr.cpp)
// .rgba32f with its size read from capture.json next to it (image_io_raw.cpp).
ImageData loadRawCapture(const std::string & utf8Path);

} // namespace sabercat::compare
