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
void testAppPolicy(Context & ctx);
void testSystemInfo(Context & ctx);
void testFileDialog(Context & ctx);
void testFileAssociation(Context & ctx);

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
        {"app policy, scene scan", sabercat::unit::testAppPolicy},
        {"system info, session logs, log mirror", sabercat::unit::testSystemInfo},
        {"file dialog helpers", sabercat::unit::testFileDialog},
        {"file association", sabercat::unit::testFileAssociation},
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
