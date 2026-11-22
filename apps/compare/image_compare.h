// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Perceptual and numeric image comparison: PSNR, the FLIP perceptual metric
// and error maps. CPU-only and free of file I/O (see image_io.h), so the unit
// tests link it without a window or a Vulkan device.
//
// Images are interleaved RGB floats in [0,1], display-encoded (sRGB). PSNR is
// computed on those values with a peak of 1; FLIP converts to linear RGB itself.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace sabercat::compare
{

struct Image
{
    // Interleaved RGB, width * height * 3 floats, row-major, top row first.
    std::vector<float> rgb;
    uint32_t width = 0;
    uint32_t height = 0;
};

struct PsnrResult
{
    double mse = 0.0;
    // +infinity when mse == 0 (identical images).
    double psnrDb = std::numeric_limits<double>::infinity();
};

struct FlipResult
{
    // One float per pixel in [0,1]; row-major, top row first.
    std::vector<float> errorMap;
    double meanError = 0.0;
};

// 8-bit RGB, 3 bytes per pixel, row-major.
struct Rgb8Image
{
    std::vector<uint8_t> rgb;
    uint32_t width = 0;
    uint32_t height = 0;
};

// Largest accepted image: width * height * 3 must fit in a 32-bit int (FLIP
// indexes with int).
inline constexpr size_t kMaxSamples = static_cast<size_t>(std::numeric_limits<int>::max());

// Throws std::invalid_argument for a zero-sized image, a pixel buffer whose
// size is not width * height * 3, or more than kMaxSamples samples.
void validate(const Image & image);

// PSNR between two images of equal size. Throws std::invalid_argument on a
// size mismatch or on a non-finite sample in either image.
PsnrResult psnr(const Image & reference, const Image & test);

// Accepted range for FLIP's pixels per degree (ppd), inclusive. Values outside
// it, and NaN, are rejected by flip().
inline constexpr float kMinFlipPpd = 1.0f;
inline constexpr float kMaxFlipPpd = 1000.0f;

// FLIP (LDR) between two images of equal size. Samples are clamped to [0,1].
// ppd is pixels per degree and must lie in [kMinFlipPpd, kMaxFlipPpd]. Throws
// std::invalid_argument on a size mismatch, a non-finite sample or a ppd outside
// that range (NaN included). Throws std::runtime_error if FLIP's mean error is
// not finite.
FlipResult flip(const Image & reference, const Image & test, float ppd);

// FLIP's default pixels per degree (0.7 m viewing distance, 3840 px, 0.7 m wide).
float defaultFlipPpd();

// Per-pixel mean over channels of |reference - test|, clamped to [0,1].
// Throws like psnr().
std::vector<float> absoluteDifferenceMap(const Image & reference, const Image & test);

// Maps values in [0,1] (clamped; NaN maps to 0) through the magma color map
// to 8-bit RGB. gray holds width * height values.
Rgb8Image colorizeMagma(const std::vector<float> & gray, uint32_t width, uint32_t height);

// The sRGB decoding table used by flip(): entry k is the linear value of the
// encoded value k / 65535. Exposed for tests.
const float * srgbToLinearTable();

} // namespace sabercat::compare
