// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The window front end of sabercat_compare (apps/compare/gui.cpp).
#pragma once

#include "launch_mode.h"

namespace sabercat::compare
{

// Runs the window for the decision's files. Returns the process exit code.
int runGui(const LaunchDecision & decision);

} // namespace sabercat::compare
