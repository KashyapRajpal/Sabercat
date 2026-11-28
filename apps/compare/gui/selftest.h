// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The compare window's headless self-test (`sabercat_compare --selftest <dir>`). It needs
// a Vulkan device but no window.
#pragma once

#include <string>

namespace sabercat::compare::gui
{

// Runs every check and prints one "[selftest] ..." line per check. The final line is
// "[selftest] PASS" (returns 0) or "[selftest] FAIL: <summary>" (returns 1). outDir
// receives the PNG and report outputs and must exist. validation says whether the
// Vulkan validation layer was requested (the caller has resolved it).
int runSelfTest(const std::string & outDir, bool validation);

} // namespace sabercat::compare::gui
