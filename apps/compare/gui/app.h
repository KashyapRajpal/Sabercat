// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The compare window's application: the GLFW window, the Vulkan host and renderer (the
// Studio's model), and the main loop. Entered from gui.cpp for every GUI launch that is not
// the self-test.
#pragma once

#include "launch_mode.h"

namespace sabercat::compare::gui
{

// Runs the window with the decision's files open at startup. Returns the process exit code.
int runApp(const LaunchDecision & decision);

} // namespace sabercat::compare::gui
