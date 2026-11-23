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

// Every debug view with its display name (equal to debugViewName(view)) and
// whether it needs the renderer's depth output. Compare's view picker lists
// these; the viewer and Studio keep using kDebugViewCycle.
// Fields are ordered big to small so the struct has no padding between them.
struct DebugViewInfo
{
    const char * name;
    DebugView view;
    bool needsDepth;
};

inline constexpr std::array<DebugViewInfo, 5> kDebugViews = {{
    {"Final", DebugView::Final, false},
    {"Alpha", DebugView::Alpha, false},
    {"Depth", DebugView::Depth, true},
    {"TileDensity", DebugView::TileDensity, false},
    {"SHDegree", DebugView::SHDegree, false},
}};

// The entry whose name equals `name` exactly, or nullptr (also for nullptr).
const DebugViewInfo * findDebugView(const char * name);

// Stable lowercase key a project file stores for a debug view ("final",
// "alpha", "depth", "tileDensity", "shDegree"). Unlike the display name the
// key never changes when the UI wording does.
const char * debugViewKey(DebugView view);

// Inverse of debugViewKey. False (and `view` untouched) for an unknown key.
bool parseDebugViewKey(const char * key, DebugView & view);

// Dear ImGui's Vulkan backend reports failures through this (CheckVkResultFn).
void checkVkResult(VkResult result);

// Whether the apps turn on the Khronos validation layer, in both device modes.
// Requested in Debug builds (!NDEBUG) or with `--validation`, and never with
// `--no-validation`. A requested layer that is not installed is a warning and
// the app runs without it; the caller passes the result on as enableValidation.
bool resolveValidation(bool validationFlag, bool noValidationFlag);

} // namespace sabercat::app
