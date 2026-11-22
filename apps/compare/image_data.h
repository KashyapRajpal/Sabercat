// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Image input for the compare tools: PNG, JPEG, WebP, OpenEXR and raw captures,
// all loaded into one RGBA float layout. The per-format readers are in
// image_io_*.cpp; loadImage() picks one by file extension.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace sabercat::compare
{

// Encoding of the float samples. Display: the file's own encoded values scaled to
// [0,1] (sRGB for PNG, JPEG and WebP, no gamma conversion). Linear: values as stored
// (EXR).
enum class Encoding : uint8_t
{
    Display,
    Linear
};

struct ImageData
{
    // width * height * 4 floats, row-major, top row first, RGBA. PNG, JPEG and WebP
    // alpha is straight (non-premultiplied). Raw .rgba32f captures are the exception
    // (premultiplied, see below and image_io_raw.cpp): rgb is multiplied by alpha, and
    // alpha is the coverage 1 - T. The loader does not convert it. Consumers composite a
    // premultiplied source over its background (black by default in the GUI); the CLI
    // uses rgb unchanged, which is the capture over black. When hasAlpha is false the
    // alpha samples are 1.
    std::vector<float> rgba;
    // width * height floats, or empty when the file has no depth.
    std::vector<float> depth;
    // NaN and infinite samples in rgba. Counted for EXR only; 0 for every other format.
    uint64_t nonFiniteCount = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    // Bits per channel in the file: 8 or 16 for PNG, JPEG and WebP, 16 (half) or 32
    // (float) for EXR, 32 for raw captures. Palette images report 8.
    uint32_t bitDepth = 0;
    Encoding encoding = Encoding::Display;
    // The source format carries an alpha channel (for palette images: the palette does).
    // Raw captures always set it: their alpha is the renderer's coverage, premultiplied
    // into rgb (see image_io_raw.cpp).
    bool hasAlpha = false;
    // True when rgb is already multiplied by alpha: raw .rgba32f captures (rgb = sum
    // T * alpha * color, alpha = 1 - T_final). False for every other loader.
    bool premultiplied = false;
};

// Loads an image by extension (case-insensitive): .png, .jpg, .jpeg (WIC), .webp
// (libwebp), .exr (tinyexr) or .rgba32f (raw capture; width and height come from
// capture.json in the same folder). The path is UTF-8. Throws std::runtime_error
// with "<path>: <reason>" on any failure, including truncated or corrupt files.
ImageData loadImage(const std::string & utf8Path);

// Loads a raw .rgba32f capture of known size: width * height vec4 float32 values,
// little-endian, rows in the order the renderer's downloadOutputImage() returns
// them (no flip is applied). A sibling .depth32f with the same stem, when present,
// supplies the depth. Throws std::runtime_error on a size mismatch.
ImageData loadRawRgba32f(const std::string & utf8Path, uint32_t width, uint32_t height);

} // namespace sabercat::compare
