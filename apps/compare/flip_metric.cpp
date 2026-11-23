// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// FLIP (LDR and HDR) between two images, FLIP's filter constants, and the magma
// color map. This is the only file that includes FLIP.h. It is built with OpenMP
// (apps/compare/CMakeLists.txt): FLIP.h's parallel loops write one output per
// index and use no reductions, so the results do not depend on the thread count.

#include "image_compare.h"
#include "flip_hdr.h"

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// FLIP.h is included in this file only. It defines Min and Max as macros, so
// they are removed right after the include to leave std::min and std::max usable.
#include "FLIP.h"
#undef Max
#undef Min

namespace sabercat::compare
{

namespace
{

using MagmaPalette = std::array<std::array<uint8_t, 3>, 256>;

std::string describe(const Image & image)
{
    return std::to_string(image.width) + "x" + std::to_string(image.height);
}

std::string describe(const ImageData & image)
{
    return std::to_string(image.width) + "x" + std::to_string(image.height);
}

// Display-encoded samples to interleaved linear RGB, through the sRGB table.
// Samples are clamped to [0,1] first. A NaN or infinite sample is caught by
// one check on the running sum, not by a branch per sample.
void linearize(const Image & image, std::vector<float> & linear)
{
    const float * table = srgbToLinearTable();
    const size_t count = image.rgb.size();
    linear.resize(count);

    const float * src = image.rgb.data();
    float * dst = linear.data();
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i)
    {
        sum += src[i];
        // std::max(0, NaN) is 0, so the index stays in range; the sum check rejects the NaN.
        const float v = std::min(1.0f, std::max(0.0f, src[i]));
        const int index = static_cast<int>(v * 65535.0f + 0.5f);
        dst[i] = table[index];
    }

    if (!std::isfinite(sum))
    {
        throw std::invalid_argument("non-finite sample");
    }
}

// Converts a linear channel in [0,1] to 8 bits, rounding to nearest.
uint8_t toByte(const float channel)
{
    return static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, channel * 255.0f) + 0.5f));
}

// FLIP's magma map as 8-bit RGB, built once.
const MagmaPalette & magmaPalette()
{
    static const MagmaPalette palette = []
    {
        MagmaPalette table{};
        for (size_t k = 0; k < table.size(); ++k)
        {
            const FLIP::color3 & c = FLIP::MapMagma[k];
            table[k] = { toByte(c.x), toByte(c.y), toByte(c.z) };
        }
        return table;
    }();
    return palette;
}

// Mean of the error values. Four double accumulators keep the mean accurate and
// let the adds overlap. The order of the adds is fixed, so the result is the same
// on every run.
double meanOf(const float * errors, size_t pixels)
{
    std::array<double, 4> lane{};
    size_t i = 0;
    for (; i + 4 <= pixels; i += 4)
    {
        lane[0] += errors[i];
        lane[1] += errors[i + 1];
        lane[2] += errors[i + 2];
        lane[3] += errors[i + 3];
    }
    for (; i < pixels; ++i)
    {
        lane[0] += errors[i];
    }
    return (lane[0] + lane[1] + lane[2] + lane[3]) / static_cast<double>(pixels);
}

// Written so that NaN fails the comparison and is rejected too.
void requirePpd(float ppd)
{
    if (!(ppd >= kMinFlipPpd && ppd <= kMaxFlipPpd))
    {
        throw std::invalid_argument("ppd must be between 1 and 1000");
    }
}

// Exposure stops must keep 2^e a finite float (hdrExposuresFromLinear checks them).
constexpr float kMaxHdrExposureStops = 126.0f;

// The exposure range and count of an HDR run. resolveHdrRange() gives the values
// FLIP::evaluate() would derive: unset exposures from FLIP's computeExposures() on
// the reference, and the count as FLIP's evaluate() computes it.
struct HdrRange
{
    float start = 0.0f;
    float stop = 0.0f;
    int count = 0;
};

// With blackFallback, a reference whose derived range is not finite (no positive luminance) gets
// the range of a 1x1 gray image of FLT_EPSILON instead, and an auto start above the auto stop is
// lowered to it. Without it the range is refused as before.
HdrRange resolveHdrRange(const std::vector<float> & linearReference, int width, int height, const HdrFlipParams & params,
                         bool blackFallback)
{
    float autoStart = 0.0f;
    float autoStop = 0.0f;
    if (!params.startExposure.has_value() || !params.stopExposure.has_value())
    {
        FLIP::image<FLIP::color3> reference;
        reference.setPixels(linearReference.data(), width, height);
        reference.computeExposures(toneMapperName(params.toneMapper), autoStart, autoStop);
        if (blackFallback && !(std::isfinite(autoStart) && std::isfinite(autoStop)))
        {
            const float epsilonGray[3] = { FLT_EPSILON, FLT_EPSILON, FLT_EPSILON };
            FLIP::image<FLIP::color3> epsilonReference;
            epsilonReference.setPixels(epsilonGray, 1, 1);
            epsilonReference.computeExposures(toneMapperName(params.toneMapper), autoStart, autoStop);
            autoStart = std::min(autoStart, autoStop);
        }
    }

    HdrRange range;
    range.start = params.startExposure.value_or(autoStart);
    range.stop = params.stopExposure.value_or(autoStop);
    // An all-black reference has no luminance, so FLIP's log2 gives infinity.
    if (!std::isfinite(range.start) || !std::isfinite(range.stop))
    {
        throw std::invalid_argument("the exposure range cannot be derived from the reference (it has no luminance); "
                                    "set the start and stop exposures");
    }
    if (range.start > range.stop)
    {
        throw std::invalid_argument("start exposure must not be greater than stop exposure");
    }

    if (params.numExposures.has_value())
    {
        range.count = *params.numExposures;
    }
    else
    {
        // FLIP::evaluate(): max(2, ceil(stop - start)).
        const float span = std::max(static_cast<float>(kMinHdrExposures), std::ceil(range.stop - range.start));
        if (span > static_cast<float>(kMaxHdrExposures))
        {
            throw std::invalid_argument("the derived exposure count exceeds " + std::to_string(kMaxHdrExposures) +
                "; set the number of exposures");
        }
        range.count = static_cast<int>(span);
    }
    if (range.count < kMinHdrExposures || range.count > kMaxHdrExposures)
    {
        throw std::invalid_argument("the number of exposures must be from " + std::to_string(kMinHdrExposures) +
            " to " + std::to_string(kMaxHdrExposures) + ", got " + std::to_string(range.count));
    }
    return range;
}

} // namespace

FlipResult flip(const Image & reference, const Image & test, float ppd)
{
    validate(reference);
    validate(test);
    if (reference.width != test.width || reference.height != test.height)
    {
        throw std::invalid_argument("image sizes differ: reference is " + describe(reference) +
            ", test is " + describe(test));
    }
    requirePpd(ppd);

    std::vector<float> linearReference;
    std::vector<float> linearTest;
    linearize(reference, linearReference);
    linearize(test, linearTest);

    const int width = static_cast<int>(reference.width);
    const int height = static_cast<int>(reference.height);

    FLIP::image<FLIP::color3> referenceImage;
    FLIP::image<FLIP::color3> testImage;
    referenceImage.setPixels(linearReference.data(), width, height);
    testImage.setPixels(linearTest.data(), width, height);

    // The five-argument evaluate() that takes image objects. The overload that
    // allocates its own output is not used.
    FLIP::Parameters parameters;
    parameters.PPD = ppd;
    FLIP::image<float> errorMap(width, height, 0.0f);
    FLIP::evaluate(referenceImage, testImage, false, parameters, errorMap);

    const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    const float * errors = errorMap.getHostData();

    FlipResult result;
    result.errorMap.resize(pixels);
    std::memcpy(result.errorMap.data(), errors, pixels * sizeof(float));
    result.meanError = meanOf(errors, pixels);
    if (!std::isfinite(result.meanError))
    {
        throw std::runtime_error("FLIP produced a non-finite mean error");
    }
    return result;
}

HdrFlipResult flipHdr(const ImageData & reference, const ImageData & test, const HdrFlipParams & params)
{
    validateImageData(reference);
    validateImageData(test);
    if (reference.width != test.width || reference.height != test.height)
    {
        throw std::invalid_argument("image sizes differ: reference is " + describe(reference) +
            ", test is " + describe(test));
    }
    requirePpd(params.ppd);

    // linearRgb() rejects NaN and infinite samples, with the count in the message.
    const std::vector<float> linearReference = linearRgb(reference);
    const std::vector<float> linearTest = linearRgb(test);

    const int width = static_cast<int>(reference.width);
    const int height = static_cast<int>(reference.height);
    const HdrRange range = resolveHdrRange(linearReference, width, height, params, false);

    FLIP::image<FLIP::color3> referenceImage;
    FLIP::image<FLIP::color3> testImage;
    referenceImage.setPixels(linearReference.data(), width, height);
    testImage.setPixels(linearTest.data(), width, height);

    // Every exposure value is set, so evaluate() derives nothing itself. Its HDR
    // path then runs the exposures, tone map, clamp and LDR-FLIP steps and keeps
    // the largest error per pixel, with the exposure that gave it.
    FLIP::Parameters parameters;
    parameters.PPD = params.ppd;
    parameters.startExposure = range.start;
    parameters.stopExposure = range.stop;
    parameters.numExposures = range.count;
    parameters.tonemapper = toneMapperName(params.toneMapper);
    FLIP::image<float> errorMap(width, height, 0.0f);
    FLIP::image<float> exposureMap(width, height, 0.0f);
    FLIP::evaluate(referenceImage, testImage, true, parameters, errorMap, exposureMap);

    const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    const float * errors = errorMap.getHostData();
    const float * exposures = exposureMap.getHostData();

    HdrFlipResult result;
    result.errorMap.assign(errors, errors + pixels);
    result.exposureMap.assign(exposures, exposures + pixels);
    result.meanError = meanOf(errors, pixels);
    if (!std::isfinite(result.meanError))
    {
        throw std::runtime_error("HDR-FLIP produced a non-finite mean error");
    }
    result.startExposure = range.start;
    result.stopExposure = range.stop;
    result.numExposures = range.count;
    return result;
}

HdrExposures hdrExposures(const ImageData & reference, ToneMapper toneMapper)
{
    validateImageData(reference);
    const std::vector<float> linearReference = linearRgb(reference);
    HdrFlipParams params;
    params.toneMapper = toneMapper;
    const HdrRange range = resolveHdrRange(linearReference, static_cast<int>(reference.width),
        static_cast<int>(reference.height), params, false);

    HdrExposures exposures;
    exposures.start = range.start;
    exposures.stop = range.stop;
    exposures.count = range.count;
    return exposures;
}

std::array<float, 6> toneMapperCoefficients(ToneMapper toneMapper)
{
    const float (&coefficients)[6] = FLIP::ToneMappingCoefficients[flipToneMapperIndex(toneMapper)];
    std::array<float, 6> out{};
    std::copy(coefficients, coefficients + 6, out.begin());
    return out;
}

int flipToneMapperIndex(ToneMapper toneMapper)
{
    switch (toneMapper)
    {
    case ToneMapper::Reinhard:
        return 0;
    case ToneMapper::Aces:
        return 1;
    case ToneMapper::Hable:
        return 2;
    }
    throw std::invalid_argument("unknown tone mapper");
}

HdrExposures hdrExposuresFromLinear(const std::vector<float> & linearRgb, uint32_t width, uint32_t height,
                                    const HdrFlipParams & params, bool blackReferenceUsesEpsilon)
{
    if (width == 0 || height == 0)
    {
        throw std::invalid_argument("the reference has no pixels");
    }
    const size_t expected = static_cast<size_t>(width) * height * 3;
    if (linearRgb.size() != expected)
    {
        throw std::invalid_argument("linear RGB has " + std::to_string(linearRgb.size()) + " values, expected " +
            std::to_string(expected));
    }
    // User values are checked before the derivation, so a bad value gets its own message.
    const auto checkLimit = [](const std::optional<float> & value, const char * name)
    {
        if (value.has_value() && !(*value >= -kMaxHdrExposureStops && *value <= kMaxHdrExposureStops))
        {
            throw std::invalid_argument(std::string(name) + " exposure must be finite and within [-126, 126]");
        }
    };
    checkLimit(params.startExposure, "start");
    checkLimit(params.stopExposure, "stop");

    const HdrRange range = resolveHdrRange(linearRgb, static_cast<int>(width), static_cast<int>(height), params,
        blackReferenceUsesEpsilon);
    if (!(range.start >= -kMaxHdrExposureStops && range.start <= kMaxHdrExposureStops) ||
        !(range.stop >= -kMaxHdrExposureStops && range.stop <= kMaxHdrExposureStops))
    {
        throw std::invalid_argument("the exposure range must be finite and within [-126, 126]");
    }

    HdrExposures exposures;
    exposures.start = range.start;
    exposures.stop = range.stop;
    exposures.count = range.count;
    return exposures;
}

FlipKernels flipKernels(float ppd)
{
    requirePpd(ppd);
    FlipKernels kernels;
    kernels.ppd = ppd;

    // Spatial filter: the radius and the taps are FLIP's own.
    kernels.spatialRadius = FLIP::calculateSpatialFilterRadius(ppd);
    const int spatialWidth = 2 * kernels.spatialRadius + 1;
    FLIP::image<FLIP::color3> spatialYCx(spatialWidth, 1);
    FLIP::image<FLIP::color3> spatialCz(spatialWidth, 1);
    FLIP::image<FLIP::color3>::setSpatialFilters(spatialYCx, spatialCz, ppd, kernels.spatialRadius);
    kernels.spatialYCx.resize(static_cast<size_t>(spatialWidth));
    kernels.spatialCz.resize(static_cast<size_t>(spatialWidth));
    for (int x = 0; x < spatialWidth; ++x)
    {
        const FLIP::color3 yCx = spatialYCx.get(x, 0);
        const FLIP::color3 cz = spatialCz.get(x, 0);
        kernels.spatialYCx[static_cast<size_t>(x)] = { yCx.x, yCx.y };
        kernels.spatialCz[static_cast<size_t>(x)] = { cz.x, cz.y };
    }

    const FLIP::xGaussianConstants & gaussian = FLIP::GaussianConstants;
    kernels.gaussianA1 = { gaussian.a1.x, gaussian.a1.y, gaussian.a1.z };
    kernels.gaussianB1 = { gaussian.b1.x, gaussian.b1.y, gaussian.b1.z };
    kernels.gaussianA2 = { gaussian.a2.x, gaussian.a2.y, gaussian.a2.z };
    kernels.gaussianB2 = { gaussian.b2.x, gaussian.b2.y, gaussian.b2.z };

    // Feature filter. FLIP's LDR_FLIP computes the radius inline from the same
    // stdDev, so the formula is repeated here; setFeatureFilter() gives the taps.
    const float stdDev = 0.5f * FLIP::FLIPConstants.gw * ppd;
    kernels.featureRadius = static_cast<int>(std::ceil(3.0f * stdDev));
    const int featureWidth = 2 * kernels.featureRadius + 1;
    FLIP::image<FLIP::color3> featureFilter(featureWidth, 1);
    FLIP::image<FLIP::color3>::setFeatureFilter(featureFilter, ppd);
    kernels.feature.resize(static_cast<size_t>(featureWidth));
    for (int x = 0; x < featureWidth; ++x)
    {
        const FLIP::color3 taps = featureFilter.get(x, 0);
        kernels.feature[static_cast<size_t>(x)] = { taps.x, taps.y, taps.z };
    }

    // Color difference constants.
    kernels.gqc = FLIP::FLIPConstants.gqc;
    kernels.gpc = FLIP::FLIPConstants.gpc;
    kernels.gpt = FLIP::FLIPConstants.gpt;
    kernels.gw = FLIP::FLIPConstants.gw;
    kernels.gqf = FLIP::FLIPConstants.gqf;
    kernels.cmax = FLIP::color3::computeMaxDistance(kernels.gqc);
    kernels.pccmax = kernels.gpc * kernels.cmax;

    // Color transforms, read from FLIP's functions. Applied to a unit vector, a
    // linear map gives one column of its matrix.
    for (int c = 0; c < 3; ++c)
    {
        const FLIP::color3 unit(c == 0 ? 1.0f : 0.0f, c == 1 ? 1.0f : 0.0f, c == 2 ? 1.0f : 0.0f);
        const FLIP::color3 xyz = FLIP::color3::LinearRGBToXYZ(unit);
        const FLIP::color3 rgb = FLIP::color3::XYZToLinearRGB(unit);
        kernels.linearRgbToXyz[static_cast<size_t>(0 * 3 + c)] = xyz.x;
        kernels.linearRgbToXyz[static_cast<size_t>(1 * 3 + c)] = xyz.y;
        kernels.linearRgbToXyz[static_cast<size_t>(2 * 3 + c)] = xyz.z;
        kernels.xyzToLinearRgb[static_cast<size_t>(0 * 3 + c)] = rgb.x;
        kernels.xyzToLinearRgb[static_cast<size_t>(1 * 3 + c)] = rgb.y;
        kernels.xyzToLinearRgb[static_cast<size_t>(2 * 3 + c)] = rgb.z;
    }
    const FLIP::color3 illuminant = DEFAULT_ILLUMINANT;
    const FLIP::color3 invIlluminant = INV_DEFAULT_ILLUMINANT;
    kernels.referenceIlluminant = { illuminant.x, illuminant.y, illuminant.z };
    kernels.invReferenceIlluminant = { invIlluminant.x, invIlluminant.y, invIlluminant.z };
    kernels.huntFactor = FLIP::color3::Hunt(1.0f, 1.0f);

    return kernels;
}

float defaultFlipPpd()
{
    FLIP::Parameters parameters;
    return parameters.PPD;
}

Rgb8Image colorizeMagma(const std::vector<float> & gray, uint32_t width, uint32_t height)
{
    const size_t pixels = static_cast<size_t>(width) * height;
    if (gray.size() != pixels)
    {
        throw std::invalid_argument("gray has " + std::to_string(gray.size()) + " values, expected " +
            std::to_string(pixels));
    }

    const MagmaPalette & palette = magmaPalette();
    Rgb8Image image;
    image.width = width;
    image.height = height;
    image.rgb.resize(pixels * 3);
    uint8_t * out = image.rgb.data();

    for (size_t p = 0; p < pixels; ++p)
    {
        // std::max(0, NaN) is 0, so NaN maps to the first palette entry.
        const float clamped = std::min(1.0f, std::max(0.0f, gray[p]));
        const auto & color = palette[static_cast<size_t>(clamped * 255.0f + 0.5f)];
        out[3 * p] = color[0];
        out[3 * p + 1] = color[1];
        out[3 * p + 2] = color[2];
    }
    return image;
}

} // namespace sabercat::compare
