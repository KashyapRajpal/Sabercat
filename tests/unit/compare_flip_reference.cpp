// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Full-resolution FLIP checks against the NVlabs FLIP reference values on FLIP's
// 1920x1080 sample images: the LDR and HDR (ACES) means, the HDR exposure range, and
// the OpenMP one-thread vs many-thread bit-identity check. A Debug run takes about two
// minutes, so these checks have their own executable (CTest entry compare_flip_reference)
// and are not in sabercat_unit_test. SABERCAT_FLIP_IMAGES_DIR names FLIP's images folder
// (see tests/unit/CMakeLists.txt), and a missing folder fails the test.
#include "unit_test.h"

#include "flip_hdr.h"
#include "image_compare.h"
#include "image_data.h"

#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef SABERCAT_FLIP_IMAGES_DIR
#define SABERCAT_FLIP_IMAGES_DIR ""
#endif

namespace sabercat::unit
{

namespace
{

using sabercat::compare::defaultFlipPpd;
using sabercat::compare::HdrExposures;
using sabercat::compare::HdrFlipParams;
using sabercat::compare::HdrFlipResult;
using sabercat::compare::ImageData;
using sabercat::compare::ToneMapper;

// The UTF-8 text of a path, for loadImage().
std::string utf8Of(const std::filesystem::path & path)
{
    const auto utf8 = path.u8string();
    return std::string(reinterpret_cast<const char *>(utf8.data()), utf8.size());
}

} // namespace

void testCompareFlipReference(Context & ctx)
{
    using sabercat::compare::displayRgb;
    using sabercat::compare::flip;
    using sabercat::compare::flipHdr;
    using sabercat::compare::hdrExposures;
    using sabercat::compare::loadImage;

    // (a) The FLIP sample images: LDR mean, and HDR (ACES) mean and exposure range.
    const std::filesystem::path flipImages(SABERCAT_FLIP_IMAGES_DIR);
    if (!std::filesystem::is_directory(flipImages))
    {
        std::cerr << __FILE__ << ":" << __LINE__ << ": FLIP image folder is missing: '" << flipImages.string()
                  << "'" << std::endl;
        ++ctx.checks;
        ++ctx.failures;
    }
    else
    {
        try
        {
            const ImageData ldrReference = loadImage(utf8Of(flipImages / "reference.png"));
            const ImageData ldrTest = loadImage(utf8Of(flipImages / "test.png"));
            const sabercat::compare::Image ldrRefImage = displayRgb(ldrReference);
            const sabercat::compare::Image ldrTestImage = displayRgb(ldrTest);
            const sabercat::compare::FlipResult ldr = flip(ldrRefImage, ldrTestImage, defaultFlipPpd());
            UNIT_CHECK_NEAR(ctx, ldr.meanError, 0.159693, 1e-6);

            const ImageData exrReference = loadImage(utf8Of(flipImages / "reference.exr"));
            const ImageData exrTest = loadImage(utf8Of(flipImages / "test.exr"));
            const HdrFlipParams params{};
            const HdrFlipResult hdr = flipHdr(exrReference, exrTest, params);
            UNIT_CHECK_NEAR(ctx, hdr.meanError, 0.283547, 1e-5);
            UNIT_CHECK_NEAR(ctx, hdr.startExposure, -12.5423, 1e-3);
            UNIT_CHECK_NEAR(ctx, hdr.stopExposure, 0.9427, 1e-3);
            UNIT_CHECK(ctx, hdr.numExposures == 14);
            const size_t pixels = static_cast<size_t>(exrReference.width) * exrReference.height;
            UNIT_CHECK(ctx, hdr.errorMap.size() == pixels && hdr.exposureMap.size() == pixels);
            bool inRange = true;
            for (size_t i = 0; i < hdr.exposureMap.size(); ++i)
            {
                inRange = inRange && hdr.exposureMap[i] >= 0.0f && hdr.exposureMap[i] <= 1.0f &&
                          hdr.errorMap[i] >= 0.0f && hdr.errorMap[i] <= 1.0f;
            }
            UNIT_CHECK(ctx, inRange);

            // hdrExposures() gives the same range and count for the reference alone.
            const HdrExposures exposures = hdrExposures(exrReference, ToneMapper::Aces);
            UNIT_CHECK_NEAR(ctx, exposures.start, hdr.startExposure, 0.0);
            UNIT_CHECK_NEAR(ctx, exposures.stop, hdr.stopExposure, 0.0);
            UNIT_CHECK(ctx, exposures.count == hdr.numExposures);

#ifdef _OPENMP
            // FLIP.h's loops write one output per index with no reductions, so one
            // thread and many threads must give the same bytes.
            const int maxThreads = omp_get_max_threads();
            omp_set_num_threads(1);
            const sabercat::compare::FlipResult ldrSingle = flip(ldrRefImage, ldrTestImage, defaultFlipPpd());
            const HdrFlipResult hdrSingle = flipHdr(exrReference, exrTest, params);
            omp_set_num_threads(maxThreads);
            const sabercat::compare::FlipResult ldrMulti = flip(ldrRefImage, ldrTestImage, defaultFlipPpd());
            const HdrFlipResult hdrMulti = flipHdr(exrReference, exrTest, params);
            omp_set_num_threads(maxThreads);

            UNIT_CHECK(ctx, ldrSingle.meanError == ldrMulti.meanError);
            UNIT_CHECK(ctx, ldrSingle.errorMap.size() == ldrMulti.errorMap.size() &&
                                std::memcmp(ldrSingle.errorMap.data(), ldrMulti.errorMap.data(),
                                            ldrSingle.errorMap.size() * sizeof(float)) == 0);
            UNIT_CHECK(ctx, hdrSingle.meanError == hdrMulti.meanError);
            UNIT_CHECK(ctx, hdrSingle.errorMap.size() == hdrMulti.errorMap.size() &&
                                std::memcmp(hdrSingle.errorMap.data(), hdrMulti.errorMap.data(),
                                            hdrSingle.errorMap.size() * sizeof(float)) == 0);
            UNIT_CHECK(ctx, hdrSingle.exposureMap.size() == hdrMulti.exposureMap.size() &&
                                std::memcmp(hdrSingle.exposureMap.data(), hdrMulti.exposureMap.data(),
                                            hdrSingle.exposureMap.size() * sizeof(float)) == 0);
            UNIT_CHECK_NEAR(ctx, ldrSingle.meanError, 0.159693, 1e-6);
#else
            // Without OpenMP the thread comparison cannot run, so the check fails.
            UNIT_CHECK(ctx, false);
            std::cerr << __FILE__ << ":" << __LINE__ << ": built without OpenMP; the thread check cannot run"
                      << std::endl;
#endif
        }
        catch (const std::exception & e)
        {
            UNIT_CHECK(ctx, false);
            std::cerr << __FILE__ << ":" << __LINE__ << ": FLIP sample check threw: " << e.what() << std::endl;
        }
    }
}

} // namespace sabercat::unit

int main()
{
    sabercat::unit::Context ctx;
    sabercat::unit::testCompareFlipReference(ctx);
    std::cout << "compare FLIP reference: " << (ctx.failures == 0 ? "ok" : "FAILED") << std::endl;
    std::cout << ctx.checks << " checks, " << ctx.failures << " failed" << std::endl;
    return ctx.failures == 0 ? 0 : 1;
}
