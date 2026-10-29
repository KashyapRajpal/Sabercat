// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Small helpers shared by the apps: the debug-view cycle, the debug-view names
// shown in the UI, and the result check for Dear ImGui's Vulkan backend.
#pragma once

#include "sabercat/renderer.h"

#include <array>

#include <vulkan/vulkan.h>

namespace sabercat::app
{

inline constexpr std::array<DebugView, 5> kDebugViewCycle = {
    DebugView::Final,
    DebugView::Alpha,
    DebugView::Depth,
    DebugView::TileDensity,
    DebugView::SHDegree,
};

// Name of a debug view, as shown in the debug panel.
const char * debugViewName(DebugView view);

// Dear ImGui's Vulkan backend reports failures through this (CheckVkResultFn).
void checkVkResult(VkResult result);

// Whether the apps turn on the Khronos validation layer, in both device modes.
// Requested in Debug builds (!NDEBUG) or with `--validation`, and never with
// `--no-validation`. A requested layer that is not installed is a warning and
// the app runs without it; the caller passes the result on as enableValidation.
bool resolveValidation(bool validationFlag, bool noValidationFlag);

} // namespace sabercat::app
