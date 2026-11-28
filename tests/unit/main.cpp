// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// CPU-only unit tests: no Vulkan device and no window. Run with
//   ctest --test-dir build -C Release -L cpu --output-on-failure
// or run sabercat_unit_test directly. Exits 0 when every check passes.
#include "unit_test.h"

#include <iostream>

namespace sabercat::unit
{

void testOrbitCamera(Context & ctx);
void testAppState(Context & ctx);
void testLayerModel(Context & ctx);
void testEditHistory(Context & ctx);
void testEditMath(Context & ctx);
void testAppPolicy(Context & ctx);
void testSystemInfo(Context & ctx);
void testFileDialog(Context & ctx);
void testFileAssociation(Context & ctx);
void testSurfaceFormat(Context & ctx);
void testBlueNoise(Context & ctx);
#ifdef SABERCAT_HAVE_SORT_STUDY
void testSortReference(Context & ctx);
void testSortMetrics(Context & ctx);
void testSortStableModel(Context & ctx);
void testSortStochastic(Context & ctx);
#endif
#ifdef SABERCAT_HAVE_COMPARE
void testImageCompare(Context & ctx);
void testCompareLaunchMode(Context & ctx);
void testCompareLoaders(Context & ctx);
void testCompareTimeline(Context & ctx);
void testCompareProject(Context & ctx);
void testCompareLayout(Context & ctx);
void testCompareFlipHdr(Context & ctx);
void testCompareReport(Context & ctx);
#endif

} // namespace sabercat::unit

int main()
{
    using sabercat::unit::Context;
    struct Suite
    {
        const char * name;
        void (*run)(Context &);
    };
    const Suite suites[] = {
        {"orbit camera", sabercat::unit::testOrbitCamera},
        {"app state JSON", sabercat::unit::testAppState},
        {"layer model", sabercat::unit::testLayerModel},
        {"edit history", sabercat::unit::testEditHistory},
        {"edit math", sabercat::unit::testEditMath},
        {"app policy, scene scan", sabercat::unit::testAppPolicy},
        {"system info, session logs, log mirror", sabercat::unit::testSystemInfo},
        {"file dialog helpers", sabercat::unit::testFileDialog},
        {"file association", sabercat::unit::testFileAssociation},
        {"swapchain format choice", sabercat::unit::testSurfaceFormat},
        {"blue noise ranks, frame offset, hash", sabercat::unit::testBlueNoise},
#ifdef SABERCAT_HAVE_SORT_STUDY
        {"sort study reference renderer", sabercat::unit::testSortReference},
        {"sort study metrics", sabercat::unit::testSortMetrics},
        {"sort study stable key model", sabercat::unit::testSortStableModel},
        {"sort study stochastic model", sabercat::unit::testSortStochastic},
#endif
#ifdef SABERCAT_HAVE_COMPARE
        {"image comparison", sabercat::unit::testImageCompare},
        {"compare launch mode", sabercat::unit::testCompareLaunchMode},
        {"compare loaders", sabercat::unit::testCompareLoaders},
        {"compare timeline", sabercat::unit::testCompareTimeline},
        {"compare project file", sabercat::unit::testCompareProject},
        {"compare layout", sabercat::unit::testCompareLayout},
        {"compare FLIP HDR and filters", sabercat::unit::testCompareFlipHdr},
        {"compare report", sabercat::unit::testCompareReport},
#endif
    };

    Context ctx;
    for (const Suite & suite : suites)
    {
        const int failuresBefore = ctx.failures;
        suite.run(ctx);
        std::cout << suite.name << ": " << (ctx.failures == failuresBefore ? "ok" : "FAILED") << std::endl;
    }
    std::cout << ctx.checks << " checks, " << ctx.failures << " failed" << std::endl;
    return ctx.failures == 0 ? 0 : 1;
}
