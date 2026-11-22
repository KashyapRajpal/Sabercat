// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// HDR-FLIP and the constants of FLIP's filters (apps/compare). Only the functions
// that call FLIP are implemented in flip_metric.cpp, the one file that includes
// FLIP.h (and is built with OpenMP). The helpers declared here are plain C++.
#pragma once

#include "image_compare.h"
#include "image_data.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace sabercat::compare
{

// FLIP's tone mappers for HDR input. The names are the ones FLIP.h uses.
enum class ToneMapper
{
    Aces,
    Hable,
    Reinhard
};

// "aces", "hable" or "reinhard".
const char * toneMapperName(ToneMapper toneMapper);

// Case-insensitive parse of a tone mapper name. Returns false, and leaves out
// unchanged, for any other text.
bool parseToneMapper(const std::string & text, ToneMapper & out);

// Accepted range for the number of HDR exposures, inclusive.
inline constexpr int kMinHdrExposures = 2;
inline constexpr int kMaxHdrExposures = 1000;

// Parameters of an HDR-FLIP run. Unset exposures are derived from the reference
// image, as FLIP::evaluate() does for its HDR path. Field order keeps the struct
// free of padding.
struct HdrFlipParams
{
    ToneMapper toneMapper = ToneMapper::Aces;
    // Pixels per degree, as for flip(). Must lie in [kMinFlipPpd, kMaxFlipPpd].
    float ppd = defaultFlipPpd();
    // Exposure range in stops. An unset value is derived from the reference.
    std::optional<float> startExposure;
    std::optional<float> stopExposure;
    // Number of exposures, in [kMinHdrExposures, kMaxHdrExposures]. Unset: derived as
    // max(2, ceil(stop - start)).
    std::optional<int> numExposures;
};

struct HdrFlipResult
{
    // One float per pixel, row-major, top row first. FLIP's HDR error: the error
    // at the exposure that gives the largest error, in [0,1].
    std::vector<float> errorMap;
    // One float per pixel: the exposure index ratio i / (count - 1) that gave the
    // largest error at that pixel, in [0,1]. This is FLIP's exposure map.
    std::vector<float> exposureMap;
    double meanError = 0.0;
    // The exposure range and count that were used, after the defaults are applied.
    float startExposure = 0.0f;
    float stopExposure = 0.0f;
    int numExposures = 0;
};

// The exposure range and count that flipHdr() uses for a reference image with
// the given tone mapper and no user values.
struct HdrExposures
{
    float start = 0.0f;
    float stop = 0.0f;
    int count = 0;
};

// HDR-FLIP between two images of equal size, on linear RGB (see linearRgb()).
// Throws std::invalid_argument on a size mismatch, a ppd outside the range, a
// NaN or infinite sample (the count is in the message), an invalid exposure range
// or count, or an exposure range that cannot be derived from the reference (for
// example an all-black reference). Throws std::runtime_error if the mean is not
// finite.
HdrFlipResult flipHdr(const ImageData & reference, const ImageData & test, const HdrFlipParams & params);

// The exposure range and count FLIP derives from the reference alone. Throws
// like flipHdr() for a non-finite sample or a reference with no derivable range.
HdrExposures hdrExposures(const ImageData & reference, ToneMapper toneMapper);

// FLIP's rational tone mapping coefficients tc[0..5] (FLIP::ToneMappingCoefficients) for the
// mapper: out = (c*c*tc[0] + c*tc[1] + tc[2]) / (c*c*tc[3] + c*tc[4] + tc[5]).
std::array<float, 6> toneMapperCoefficients(ToneMapper toneMapper);

// FLIP's index of a mapper in ToneMappingCoefficients: Reinhard 0, Aces 1, Hable 2.
int flipToneMapperIndex(ToneMapper toneMapper);

// The exposure range and count for linear RGB samples (width * height * 3 finite floats,
// row-major), with the user values in params (ppd is ignored). Same rules and exceptions as
// flipHdr()'s range derivation (start > stop, count outside 2..1000, derived count above
// 1000: std::invalid_argument), plus: start and stop must be finite and within [-126, 126]
// (std::invalid_argument), and when blackReferenceUsesEpsilon is true a reference whose range
// cannot be derived (no positive luminance) is evaluated as a 1x1 gray image of FLT_EPSILON
// (start = min(start, stop)) instead of throwing. Calls FLIP's computeExposures().
HdrExposures hdrExposuresFromLinear(const std::vector<float> & linearRgb, uint32_t width, uint32_t height,
                                    const HdrFlipParams & params, bool blackReferenceUsesEpsilon);

// FLIP's filter constants and the separable filter taps for one ppd. Every value is
// read from FLIP.h: the filters come from FLIP's own generators
// (image::setSpatialFilters, image::setFeatureFilter), the Gaussian and color
// constants from FLIP's structs, and the color matrices from FLIP's conversion
// functions applied to unit vectors. Only the feature filter radius formula is
// written out (FLIP computes it inline in LDR_FLIP). The HyAB distance has no
// parameters beyond gqc, so it is not stored.
struct FlipKernels
{
    // Pixels per degree the kernels were built for (FLIP::Parameters::PPD).
    float ppd = 0.0f;

    // Spatial (contrast sensitivity) filter. The Y/Cx filter and the Cz filter are
    // separate because the Cz channel is a sum of two Gaussians. Taps are
    // 2 * spatialRadius + 1 wide, from the left edge to the right edge.
    int spatialRadius = 0;
    // Per tap: {Y weight, Cx weight}. Each column sums to 1.
    std::vector<std::array<float, 2>> spatialYCx;
    // Per tap: {weight of the first Gaussian, weight of the second}. The two columns
    // satisfy sqrt(sum1^2 + sum2^2) = 1.
    std::vector<std::array<float, 2>> spatialCz;
    // FLIP's Gaussian constants, one entry per (Y, Cx, Cz) channel: a1 and b1 for
    // the first Gaussian, a2 and b2 for the second (used by Cz only).
    std::array<float, 3> gaussianA1{};
    std::array<float, 3> gaussianB1{};
    std::array<float, 3> gaussianA2{};
    std::array<float, 3> gaussianB2{};

    // Feature (edge and point) filter. Taps are 2 * featureRadius + 1 wide.
    int featureRadius = 0;
    // Per tap: {Gaussian, first derivative, second derivative}. The Gaussian sums to
    // 1. The positive part of the first derivative sums to 1 and the negative part
    // to -1; the same holds for the second derivative.
    std::vector<std::array<float, 3>> feature;

    // Color difference constants (FLIP::FLIPConstants).
    float gqc = 0.0f;   // exponent applied to the HyAB distance
    float gpc = 0.0f;   // fraction of cmax at which the error map turns from linear to the second segment
    float gpt = 0.0f;   // output value at pccmax
    float gw = 0.0f;    // feature filter width factor: stdDev = 0.5 * gw * ppd
    float gqf = 0.0f;   // exponent applied to the feature difference
    float cmax = 0.0f;  // color3::computeMaxDistance(gqc)
    float pccmax = 0.0f; // gpc * cmax

    // Color transforms. Matrices are row-major 3x3: value = M * vector.
    std::array<float, 9> linearRgbToXyz{};   // D65 sRGB primaries
    std::array<float, 9> xyzToLinearRgb{};   // inverse of the above
    std::array<float, 3> referenceIlluminant{};     // D65 white point (XYZ scale)
    std::array<float, 3> invReferenceIlluminant{};  // 1 / referenceIlluminant
    // Hunt adjustment: L * c * huntFactor (color3::Hunt).
    float huntFactor = 0.0f;
};

// The filter set FLIP uses for ppd. Throws std::invalid_argument for a ppd outside
// [kMinFlipPpd, kMaxFlipPpd] (NaN included).
FlipKernels flipKernels(float ppd);

// Throws std::invalid_argument for a zero size, a sample buffer whose size is not
// width * height * 4, or an image too large for FLIP (more than kMaxSamples / 3
// pixels).
void validateImageData(const ImageData & image);

// Linear RGB samples of an image, width * height * 3 floats, row-major. Display
// inputs are decoded with the sRGB table (srgbToLinearTable(), clamped to [0,1]);
// Linear inputs are taken as stored. Alpha is dropped. Throws
// std::invalid_argument when the buffer size is wrong or when any RGB sample is NaN
// or infinite; the message gives the count.
std::vector<float> linearRgb(const ImageData & image);

// The image as a display shows it, for PSNR and the magma maps: Display inputs as
// stored; Linear inputs clamped to [0,1] and encoded with the sRGB transfer
// function. Alpha is dropped. Same validation as linearRgb().
Image displayRgb(const ImageData & image);

// Number of RGB samples of an image that are NaN or infinite (0 for a valid image).
// Throws std::invalid_argument for a wrong buffer size.
uint64_t countNonFiniteRgb(const ImageData & image);

// Mean squared difference over the samples of two equal-length vectors. Pairs with
// a non-finite value on either side are skipped. Returns NaN when no pair counts.
double linearMse(const std::vector<float> & a, const std::vector<float> & b);

} // namespace sabercat::compare
