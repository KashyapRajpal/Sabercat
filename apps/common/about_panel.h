// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Help > About: the name, version, license and copyright of Sabercat, and a
// scrollable read-only view of THIRD_PARTY_NOTICES.md. Shared by the viewer
// and the Studio.
#pragma once

#include <imgui.h>

namespace sabercat::app
{

struct SystemInfoFacts;

// Draws the About window while `open` is true; the window's close button
// clears `open`. The heading is `facts.appName` (e.g. "Sabercat Viewer") with
// the version appended. The window shows the logo (`logo`, an AboutLogo's
// texture; 0 to skip it), the log file, with buttons to open its folder and to
// copy the system info (formatSystemInfo).
void drawAboutWindow(bool & open, const SystemInfoFacts & facts, ImTextureID logo);

} // namespace sabercat::app
