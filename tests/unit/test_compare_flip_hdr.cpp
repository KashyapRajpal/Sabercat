// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// HDR-FLIP and FLIP's filter constants (apps/compare/flip_hdr.h): parameter checks
// on synthetic images, the tone mapper names and coefficients, the HDR exposure range
// from linear samples, and the filter constants and taps. The
// full-resolution checks against the NVlabs FLIP sample images are in
// compare_flip_reference.cpp, a separate executable.
#include "unit_test.h"

#include "flip_hdr.h"
#include "image_compare.h"
#include "image_data.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace sabercat::unit
{

namespace
{

using sabercat::compare::defaultFlipPpd;
using sabercat::compare::FlipKernels;
using sabercat::compare::HdrExposures;
using sabercat::compare::HdrFlipParams;
using sabercat::compare::HdrFlipResult;
using sabercat::compare::ImageData;
using sabercat::compare::Encoding;
using sabercat::compare::ToneMapper;

// A linear image with every RGB sample set to value and alpha 1.
ImageData makeLinear(uint32_t width, uint32_t height, float value)
{
    ImageData image;
    image.width = width;
    image.height = height;
    image.rgba.assign(static_cast<size_t>(width) * height * 4, value);
    for (size_t i = 3; i < image.rgba.size(); i += 4)
    {
        image.rgba[i] = 1.0f;
    }
    image.bitDepth = 32;
    image.encoding = Encoding::Linear;
    image.hasAlpha = false;
    return image;
}

// The message of the std::invalid_argument fn throws, or "" when it does not throw.
template <typename Fn>
std::string invalidArgumentMessage(Fn && fn)
{
    try
    {
        fn();
    }
    catch (const std::invalid_argument & e)
    {
        return e.what();
    }
    catch (...)
    {
        return "wrong exception type";
    }
    return std::string();
}

bool contains(const std::string & text, const std::string & needle)
{
    return text.find(needle) != std::string::npos;
}

// A gray image with a skewed distribution: pixel i has the value peak * t^4, where t
// runs from 0 to 1 over the pixels. The median is far below the peak, so the exposure
// range derived from it is several stops wide.
ImageData makeSkewedRamp(uint32_t width, uint32_t height, float peak)
{
    ImageData image = makeLinear(width, height, 0.0f);
    const size_t pixels = static_cast<size_t>(width) * height;
    for (size_t i = 0; i < pixels; ++i)
    {
        const float t = static_cast<float>(i) / static_cast<float>(pixels - 1);
        const float value = peak * t * t * t * t;
        image.rgba[i * 4 + 0] = value;
        image.rgba[i * 4 + 1] = value;
        image.rgba[i * 4 + 2] = value;
    }
    return image;
}

// Parameters with a tone mapper and the given exposure range and count. An unset value
// is left out, so FLIP derives it.
HdrFlipParams rangeParams(ToneMapper toneMapper, std::optional<float> start, std::optional<float> stop,
                          std::optional<int> count)
{
    HdrFlipParams params;
    params.toneMapper = toneMapper;
    params.startExposure = start;
    params.stopExposure = stop;
    params.numExposures = count;
    return params;
}

// The count for a range without a user count: max(2, ceil(stop - start)).
int defaultCount(float start, float stop)
{
    return static_cast<int>(std::max(2.0f, std::ceil(stop - start)));
}

// True when fn throws std::invalid_argument.
template <typename Fn>
bool throwsInvalidArgument(Fn && fn)
{
    try
    {
        fn();
    }
    catch (const std::invalid_argument &)
    {
        return true;
    }
    catch (...)
    {
        return false;
    }
    return false;
}

// FLIP's pi as written in FLIP.h, for the closed-form checks.
constexpr double kFlipPi = 3.14159265358979;

} // namespace

void testCompareFlipHdr(Context & ctx)
{
    using sabercat::compare::flipHdr;
    using sabercat::compare::flipKernels;
    using sabercat::compare::hdrExposures;
    using sabercat::compare::linearRgb;
    using sabercat::compare::parseToneMapper;
    using sabercat::compare::toneMapperName;

    // (b) Tone mapper names.
    {
        ToneMapper mapper = ToneMapper::Aces;
        UNIT_CHECK(ctx, parseToneMapper("HABLE", mapper) && mapper == ToneMapper::Hable);
        UNIT_CHECK(ctx, parseToneMapper("Reinhard", mapper) && mapper == ToneMapper::Reinhard);
        UNIT_CHECK(ctx, parseToneMapper("aces", mapper) && mapper == ToneMapper::Aces);
        UNIT_CHECK(ctx, !parseToneMapper("foo", mapper) && mapper == ToneMapper::Aces);
        UNIT_CHECK(ctx, std::string(toneMapperName(ToneMapper::Hable)) == "hable");
    }

    // (c) Invalid HDR parameters, on synthetic images.
    {
        const ImageData gray = makeLinear(4, 3, 0.5f);
        const ImageData dim = makeLinear(4, 3, 0.25f);
        const ImageData other = makeLinear(3, 4, 0.5f);
        const HdrFlipParams good{};

        UNIT_CHECK(ctx, invalidArgumentMessage([&] { flipHdr(gray, dim, good); }).empty());

        const float ppds[] = {0.0f, 0.5f, -3.0f, 1000.5f, std::numeric_limits<float>::quiet_NaN(),
                              std::numeric_limits<float>::infinity()};
        for (const float ppd : ppds)
        {
            HdrFlipParams params;
            params.ppd = ppd;
            UNIT_CHECK(ctx, !invalidArgumentMessage([&] { flipHdr(gray, dim, params); }).empty());
        }

        UNIT_CHECK(ctx, contains(invalidArgumentMessage([&] { flipHdr(gray, other, good); }), "image sizes differ"));

        ImageData nanReference = gray;
        nanReference.rgba[0] = std::numeric_limits<float>::quiet_NaN();
        UNIT_CHECK(ctx, contains(invalidArgumentMessage([&] { flipHdr(nanReference, dim, good); }), "1 non-finite"));

        ImageData infTest = dim;
        infTest.rgba[4] = std::numeric_limits<float>::infinity();
        infTest.rgba[5] = -std::numeric_limits<float>::infinity();
        UNIT_CHECK(ctx, contains(invalidArgumentMessage([&] { flipHdr(gray, infTest, good); }), "2 non-finite"));

        // Alpha is not compared, so a NaN alpha is not an error.
        ImageData nanAlpha = gray;
        nanAlpha.rgba[3] = std::numeric_limits<float>::quiet_NaN();
        UNIT_CHECK(ctx, invalidArgumentMessage([&] { flipHdr(nanAlpha, dim, good); }).empty());

        // Exposure range and count.
        HdrFlipParams reversed;
        reversed.startExposure = 1.0f;
        reversed.stopExposure = 0.0f;
        UNIT_CHECK(ctx, contains(invalidArgumentMessage([&] { flipHdr(gray, dim, reversed); }), "must not be greater"));

        HdrFlipParams infiniteStart;
        infiniteStart.startExposure = std::numeric_limits<float>::infinity();
        UNIT_CHECK(ctx, !invalidArgumentMessage([&] { flipHdr(gray, dim, infiniteStart); }).empty());

        HdrFlipParams oneExposure;
        oneExposure.numExposures = 1;
        UNIT_CHECK(ctx, !invalidArgumentMessage([&] { flipHdr(gray, dim, oneExposure); }).empty());
        HdrFlipParams tooMany;
        tooMany.numExposures = 1001;
        UNIT_CHECK(ctx, !invalidArgumentMessage([&] { flipHdr(gray, dim, tooMany); }).empty());
        HdrFlipParams two;
        two.numExposures = 2;
        UNIT_CHECK(ctx, invalidArgumentMessage([&] { flipHdr(gray, dim, two); }).empty());

        // A black reference has no luminance to derive a range from. With the range
        // given, it is accepted.
        const ImageData black = makeLinear(4, 3, 0.0f);
        UNIT_CHECK(ctx, !invalidArgumentMessage([&] { flipHdr(black, black, good); }).empty());
        HdrFlipParams given;
        given.startExposure = -5.0f;
        given.stopExposure = 0.0f;
        UNIT_CHECK(ctx, invalidArgumentMessage([&] { flipHdr(black, black, given); }).empty());

        // Identical images: zero error everywhere, and no exposure selected.
        const HdrFlipResult same = flipHdr(gray, gray, good);
        UNIT_CHECK_NEAR(ctx, same.meanError, 0.0, 0.0);
        bool allZero = true;
        for (size_t i = 0; i < same.exposureMap.size(); ++i)
        {
            allZero = allZero && same.exposureMap[i] == 0.0f && same.errorMap[i] == 0.0f;
        }
        UNIT_CHECK(ctx, allZero);

        // The default range of a constant image is valid.
        const HdrExposures constant = hdrExposures(gray, ToneMapper::Hable);
        UNIT_CHECK(ctx, constant.count >= 2 && constant.start <= constant.stop);
        UNIT_CHECK(ctx, !invalidArgumentMessage([&] { hdrExposures(nanReference, ToneMapper::Aces); }).empty());
        UNIT_CHECK(ctx, !invalidArgumentMessage([&] { hdrExposures(black, ToneMapper::Aces); }).empty());

        // Linear RGB: sRGB decoding for display input, and the stored values for linear.
        const std::vector<float> linear = linearRgb(gray);
        UNIT_CHECK(ctx, linear.size() == 4u * 3u * 3u);
        UNIT_CHECK_NEAR(ctx, linear[0], 0.5, 1e-6);
        ImageData display = gray;
        display.encoding = Encoding::Display;
        const std::vector<float> decoded = linearRgb(display);
        // sRGB 0.5 decodes to ((0.5 + 0.055) / 1.055)^2.4 = 0.214041.
        UNIT_CHECK_NEAR(ctx, decoded[0], 0.214041, 1e-5);
    }

    // (d) Filter constants and taps, from flipKernels().
    {
        const float ppd = defaultFlipPpd();
        const FlipKernels k = flipKernels(ppd);
        UNIT_CHECK_NEAR(ctx, k.ppd, ppd, 0.0);
        UNIT_CHECK_NEAR(ctx, ppd, 67.02, 0.01);

        // FLIP's constants (FLIP.h, FLIPConstants and GaussianConstants).
        UNIT_CHECK_NEAR(ctx, k.gqc, 0.7, 1e-6);
        UNIT_CHECK_NEAR(ctx, k.gpc, 0.4, 1e-6);
        UNIT_CHECK_NEAR(ctx, k.gpt, 0.95, 1e-6);
        UNIT_CHECK_NEAR(ctx, k.gw, 0.082, 1e-6);
        UNIT_CHECK_NEAR(ctx, k.gqf, 0.5, 1e-6);
        UNIT_CHECK(ctx, k.cmax > 0.0f && std::isfinite(k.cmax));
        UNIT_CHECK_NEAR(ctx, k.pccmax, k.gpc * k.cmax, 1e-7);
        UNIT_CHECK_NEAR(ctx, k.gaussianA1[0], 1.0, 1e-7);
        UNIT_CHECK_NEAR(ctx, k.gaussianA1[2], 34.1, 1e-5);
        UNIT_CHECK_NEAR(ctx, k.gaussianB1[0], 0.0047, 1e-7);
        UNIT_CHECK_NEAR(ctx, k.gaussianB1[1], 0.0053, 1e-7);
        UNIT_CHECK_NEAR(ctx, k.gaussianB1[2], 0.04, 1e-7);
        UNIT_CHECK_NEAR(ctx, k.gaussianA2[2], 13.5, 1e-6);
        UNIT_CHECK_NEAR(ctx, k.gaussianB2[0], 1e-5, 1e-9);
        UNIT_CHECK_NEAR(ctx, k.gaussianB2[2], 0.025, 1e-7);

        // Color transforms: the sRGB-to-XYZ (D65) matrix, its inverse, and the illuminant.
        UNIT_CHECK_NEAR(ctx, k.linearRgbToXyz[0], 0.41239, 1e-4);
        UNIT_CHECK_NEAR(ctx, k.linearRgbToXyz[1], 0.35758, 1e-4);
        UNIT_CHECK_NEAR(ctx, k.linearRgbToXyz[2], 0.18048, 1e-4);
        UNIT_CHECK_NEAR(ctx, k.linearRgbToXyz[4], 0.71517, 1e-4);
        bool inverse = true;
        for (size_t row = 0; row < 3; ++row)
        {
            for (size_t col = 0; col < 3; ++col)
            {
                double sum = 0.0;
                for (size_t mid = 0; mid < 3; ++mid)
                {
                    sum += static_cast<double>(k.xyzToLinearRgb[row * 3 + mid]) * k.linearRgbToXyz[mid * 3 + col];
                }
                inverse = inverse && std::abs(sum - (row == col ? 1.0 : 0.0)) < 1e-4;
            }
        }
        UNIT_CHECK(ctx, inverse);
        UNIT_CHECK_NEAR(ctx, k.referenceIlluminant[0], 0.950428545, 1e-6);
        UNIT_CHECK_NEAR(ctx, k.referenceIlluminant[2], 1.088900371, 1e-6);
        for (size_t c = 0; c < 3; ++c)
        {
            UNIT_CHECK_NEAR(ctx, k.referenceIlluminant[c] * k.invReferenceIlluminant[c], 1.0, 1e-5);
        }
        UNIT_CHECK_NEAR(ctx, k.huntFactor, 0.01, 1e-7);

        // Spatial filter: the radius from FLIP's formula (3 sigma of the largest
        // scale, in pixels). Taps are odd in number, and the column sums are FLIP's.
        const double maxScale = 0.04;
        const double radiusExpected = std::ceil(3.0 * std::sqrt(maxScale / (2.0 * kFlipPi * kFlipPi)) * ppd);
        UNIT_CHECK(ctx, std::abs(static_cast<double>(k.spatialRadius) - radiusExpected) <= 1.0);
        const size_t taps = static_cast<size_t>(2 * k.spatialRadius + 1);
        UNIT_CHECK(ctx, k.spatialYCx.size() == taps && k.spatialCz.size() == taps);
        double sumY = 0.0;
        double sumCx = 0.0;
        double sumCz1 = 0.0;
        double sumCz2 = 0.0;
        for (size_t i = 0; i < taps; ++i)
        {
            sumY += k.spatialYCx[i][0];
            sumCx += k.spatialYCx[i][1];
            sumCz1 += k.spatialCz[i][0];
            sumCz2 += k.spatialCz[i][1];
        }
        UNIT_CHECK_NEAR(ctx, sumY, 1.0, 1e-5);
        UNIT_CHECK_NEAR(ctx, sumCx, 1.0, 1e-5);
        UNIT_CHECK_NEAR(ctx, std::sqrt(sumCz1 * sumCz1 + sumCz2 * sumCz2), 1.0, 1e-5);

        // Symmetry: the taps are mirror images about the center.
        bool symmetric = true;
        for (size_t i = 0; i < taps; ++i)
        {
            for (size_t j = 0; j < 2; ++j)
            {
                symmetric = symmetric && std::abs(k.spatialYCx[i][j] - k.spatialYCx[taps - 1 - i][j]) <= 1e-7f;
                symmetric = symmetric && std::abs(k.spatialCz[i][j] - k.spatialCz[taps - 1 - i][j]) <= 1e-7f;
            }
        }
        UNIT_CHECK(ctx, symmetric);

        // Closed form: each Gaussian's tap at offset x, relative to the center, is
        // exp(-pi^2 (x / ppd)^2 / b) (the FLIP paper's Gaussian in its b form). Y uses
        // b1.x, Cx uses b1.y, and the two Cz Gaussians use b1.z and b2.z.
        const int center = k.spatialRadius;
        bool matchesGaussians = k.spatialYCx[center][0] > 0.0f;
        for (int x = -center; x <= center; ++x)
        {
            const double dist = static_cast<double>(x) / static_cast<double>(ppd);
            const size_t i = static_cast<size_t>(x + center);
            const double expectY = std::exp(-kFlipPi * kFlipPi * dist * dist / k.gaussianB1[0]);
            const double expectCx = std::exp(-kFlipPi * kFlipPi * dist * dist / k.gaussianB1[1]);
            const double expectCz1 = std::exp(-kFlipPi * kFlipPi * dist * dist / k.gaussianB1[2]);
            const double expectCz2 = std::exp(-kFlipPi * kFlipPi * dist * dist / k.gaussianB2[2]);
            const double centerY = k.spatialYCx[static_cast<size_t>(center)][0];
            const double centerCx = k.spatialYCx[static_cast<size_t>(center)][1];
            const double centerCz1 = k.spatialCz[static_cast<size_t>(center)][0];
            const double centerCz2 = k.spatialCz[static_cast<size_t>(center)][1];
            matchesGaussians = matchesGaussians &&
                std::abs(k.spatialYCx[i][0] / centerY - expectY) < 1e-4 &&
                std::abs(k.spatialYCx[i][1] / centerCx - expectCx) < 1e-4 &&
                std::abs(k.spatialCz[i][0] / centerCz1 - expectCz1) < 1e-4 &&
                std::abs(k.spatialCz[i][1] / centerCz2 - expectCz2) < 1e-4;
        }
        UNIT_CHECK(ctx, matchesGaussians);

        // Feature filter: radius = ceil(3 * 0.5 * gw * ppd), with the Gaussian and the
        // derivative normalizations of FLIP's setFeatureFilter.
        const double sigma = 0.5 * 0.082 * static_cast<double>(ppd);
        const double featureExpected = std::ceil(3.0 * sigma);
        UNIT_CHECK(ctx, std::abs(static_cast<double>(k.featureRadius) - featureExpected) <= 1.0);
        const size_t featureTaps = static_cast<size_t>(2 * k.featureRadius + 1);
        UNIT_CHECK(ctx, k.feature.size() == featureTaps);
        double gaussSum = 0.0;
        double dPositive = 0.0;
        double dNegative = 0.0;
        double ddPositive = 0.0;
        double ddNegative = 0.0;
        for (const auto & tap : k.feature)
        {
            gaussSum += tap[0];
            (tap[1] > 0.0f ? dPositive : dNegative) += tap[1];
            (tap[2] > 0.0f ? ddPositive : ddNegative) += tap[2];
        }
        UNIT_CHECK_NEAR(ctx, gaussSum, 1.0, 1e-5);
        UNIT_CHECK_NEAR(ctx, dPositive, 1.0, 1e-5);
        UNIT_CHECK_NEAR(ctx, dNegative, -1.0, 1e-5);
        UNIT_CHECK_NEAR(ctx, ddPositive, 1.0, 1e-5);
        UNIT_CHECK_NEAR(ctx, ddNegative, -1.0, 1e-5);

        bool featureSymmetric = true;
        bool featureGaussian = true;
        const int featureCenter = k.featureRadius;
        for (size_t i = 0; i < featureTaps; ++i)
        {
            const auto & a = k.feature[i];
            const auto & b = k.feature[featureTaps - 1 - i];
            featureSymmetric = featureSymmetric && std::abs(a[0] - b[0]) <= 1e-6f &&
                               std::abs(a[1] + b[1]) <= 1e-6f && std::abs(a[2] - b[2]) <= 1e-6f;
            const double x = static_cast<double>(static_cast<int>(i) - featureCenter);
            const double expect = std::exp(-x * x / (2.0 * sigma * sigma));
            featureGaussian = featureGaussian &&
                std::abs(a[0] / k.feature[static_cast<size_t>(featureCenter)][0] - expect) < 1e-4;
        }
        UNIT_CHECK(ctx, featureSymmetric);
        UNIT_CHECK(ctx, featureGaussian);

        // Invalid ppd values are rejected, as by flip().
        const float badPpds[] = {0.0f, std::numeric_limits<float>::quiet_NaN(), 1000.5f};
        for (const float bad : badPpds)
        {
            UNIT_CHECK(ctx, !invalidArgumentMessage([&] { flipKernels(bad); }).empty());
        }
    }

    // (e) Tone mapper coefficients and FLIP's indices, from FLIP.h (ToneMappingCoefficients).
    {
        using sabercat::compare::flipToneMapperIndex;
        using sabercat::compare::toneMapperCoefficients;

        UNIT_CHECK(ctx, flipToneMapperIndex(ToneMapper::Reinhard) == 0);
        UNIT_CHECK(ctx, flipToneMapperIndex(ToneMapper::Aces) == 1);
        UNIT_CHECK(ctx, flipToneMapperIndex(ToneMapper::Hable) == 2);

        // The rows of FLIP.h lines 138-143, in the same float arithmetic.
        const float reinhard[6] = {0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f};
        const float aces[6] = {0.6f * 0.6f * 2.51f, 0.6f * 0.03f, 0.0f, 0.6f * 0.6f * 2.43f, 0.6f * 0.59f, 0.14f};
        const float hable[6] = {0.231683f, 0.013791f, 0.0f, 0.18f, 0.3f, 0.018f};
        const float * rows[3] = {reinhard, aces, hable};
        const ToneMapper mappers[3] = {ToneMapper::Reinhard, ToneMapper::Aces, ToneMapper::Hable};
        for (size_t m = 0; m < 3; ++m)
        {
            const std::array<float, 6> coefficients = toneMapperCoefficients(mappers[m]);
            for (size_t c = 0; c < 6; ++c)
            {
                UNIT_CHECK_NEAR(ctx, coefficients[c], rows[m][c], 1e-6);
            }
        }
    }

    // (f) The exposure range from linear samples: hdrExposuresFromLinear() agrees with
    // hdrExposures() on the same image, takes the user values as given, and handles the
    // black and uniform references.
    {
        using sabercat::compare::hdrExposuresFromLinear;

        const ImageData ramp = makeSkewedRamp(8, 8, 4.0f);
        const std::vector<float> rampLinear = linearRgb(ramp);
        const ImageData black = makeLinear(4, 3, 0.0f);
        const std::vector<float> blackLinear = linearRgb(black);
        const std::vector<float> uniformLinear = linearRgb(makeLinear(4, 3, 0.5f));
        const ToneMapper mappers[3] = {ToneMapper::Aces, ToneMapper::Hable, ToneMapper::Reinhard};

        for (const ToneMapper mapper : mappers)
        {
            // No user values: the same range and count as hdrExposures().
            const HdrExposures derived = hdrExposures(ramp, mapper);
            const HdrExposures fromLinear = hdrExposuresFromLinear(
                rampLinear, 8, 8, rangeParams(mapper, std::nullopt, std::nullopt, std::nullopt), false);
            UNIT_CHECK(ctx, derived.count >= 2 && derived.start < derived.stop);
            UNIT_CHECK(ctx, fromLinear.start == derived.start && fromLinear.stop == derived.stop &&
                                fromLinear.count == derived.count);

            // Start alone: the stop is FLIP's, and the count follows the span.
            const float startValue = derived.start - 1.0f;
            const HdrExposures startOnly = hdrExposuresFromLinear(
                rampLinear, 8, 8, rangeParams(mapper, startValue, std::nullopt, std::nullopt), false);
            UNIT_CHECK(ctx, startOnly.start == startValue && startOnly.stop == derived.stop);
            UNIT_CHECK(ctx, startOnly.count == defaultCount(startOnly.start, startOnly.stop));

            // Stop alone: the start is FLIP's, and the count follows the span.
            const float stopValue = derived.stop + 2.0f;
            const HdrExposures stopOnly = hdrExposuresFromLinear(
                rampLinear, 8, 8, rangeParams(mapper, std::nullopt, stopValue, std::nullopt), false);
            UNIT_CHECK(ctx, stopOnly.start == derived.start && stopOnly.stop == stopValue);
            UNIT_CHECK(ctx, stopOnly.count == defaultCount(stopOnly.start, stopOnly.stop));

            // Both: taken as given, and the count is max(2, ceil(3 - -2)) = 5.
            const HdrExposures both = hdrExposuresFromLinear(
                rampLinear, 8, 8, rangeParams(mapper, -2.0f, 3.0f, std::nullopt), false);
            UNIT_CHECK(ctx, both.start == -2.0f && both.stop == 3.0f && both.count == 5);

            // Count alone: the range is FLIP's.
            const HdrExposures countOnly = hdrExposuresFromLinear(
                rampLinear, 8, 8, rangeParams(mapper, std::nullopt, std::nullopt, 7), false);
            UNIT_CHECK(ctx, countOnly.start == derived.start && countOnly.stop == derived.stop && countOnly.count == 7);

            // Limits: start and stop within [-126, 126], and the count within 2 to 1000.
            const auto refused = [&](std::optional<float> from, std::optional<float> to, std::optional<int> count) {
                return throwsInvalidArgument([&] {
                    hdrExposuresFromLinear(rampLinear, 8, 8, rangeParams(mapper, from, to, count), false);
                });
            };
            const HdrExposures limits = hdrExposuresFromLinear(
                rampLinear, 8, 8, rangeParams(mapper, -126.0f, 126.0f, std::nullopt), false);
            UNIT_CHECK(ctx, limits.start == -126.0f && limits.stop == 126.0f && limits.count == 252);
            UNIT_CHECK(ctx, hdrExposuresFromLinear(rampLinear, 8, 8, rangeParams(mapper, -2.0f, 3.0f, 2), false).count == 2);
            UNIT_CHECK(ctx, refused(1.0f, 0.0f, std::nullopt));
            UNIT_CHECK(ctx, refused(-2.0f, 3.0f, 1));
            UNIT_CHECK(ctx, refused(-2.0f, 3.0f, 1001));
            UNIT_CHECK(ctx, refused(127.0f, 128.0f, std::nullopt));
            UNIT_CHECK(ctx, refused(-127.0f, 0.0f, std::nullopt));
            UNIT_CHECK(ctx, refused(0.0f, 127.0f, std::nullopt));
            UNIT_CHECK(ctx, refused(std::numeric_limits<float>::quiet_NaN(), 1.0f, std::nullopt));
            UNIT_CHECK(ctx, refused(0.0f, std::numeric_limits<float>::infinity(), std::nullopt));

            // All-black reference: no range can be derived, so it is an error unless the
            // epsilon fallback is allowed. The fallback gives one exposure (count 2).
            const auto noRange = rangeParams(mapper, std::nullopt, std::nullopt, std::nullopt);
            UNIT_CHECK(ctx, throwsInvalidArgument([&] {
                hdrExposuresFromLinear(blackLinear, 4, 3, noRange, false);
            }));
            const HdrExposures epsilon = hdrExposuresFromLinear(blackLinear, 4, 3, noRange, true);
            UNIT_CHECK(ctx, epsilon.count == 2 && epsilon.start == epsilon.stop);
            UNIT_CHECK(ctx, std::isfinite(epsilon.start) && std::isfinite(epsilon.stop));
            // With both values given, the black reference is not evaluated.
            const HdrExposures blackGiven = hdrExposuresFromLinear(
                blackLinear, 4, 3, rangeParams(mapper, -5.0f, 0.0f, std::nullopt), false);
            UNIT_CHECK(ctx, blackGiven.start == -5.0f && blackGiven.stop == 0.0f && blackGiven.count == 5);

            // Uniform reference: the maximum equals the median, so start == stop and count 2.
            const HdrExposures uniform = hdrExposuresFromLinear(uniformLinear, 4, 3, noRange, false);
            UNIT_CHECK(ctx, uniform.count == 2 && uniform.start == uniform.stop);
        }
    }
}

} // namespace sabercat::unit
