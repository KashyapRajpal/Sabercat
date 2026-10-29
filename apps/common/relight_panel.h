// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The Relight window (lighting, baked visibility and teacher
// samples) as a panel any app can draw. Its state is a RelightPanelState that
// the app owns; nothing here is global.
#pragma once

#include "sabercat/lighting.h"
#include "sabercat/renderer.h"

#include <cstdint>
#include <string>
#include <vector>

namespace sabercat::app
{

// The panel's UI state, persisting across frames. Lighting on/off, what the
// lighting pass shows, exposure, an environment preset, and up to kMaxLights
// editable lights placed relative to the loaded scene's bounds. BakedSH is
// offered once THIS scene's storage has been baked (a new scene changes
// storageGeneration()).
struct RelightPanelState
{
    // startEnabled: start with relighting on (the viewer's `--relight` flag).
    explicit RelightPanelState(bool startEnabled) : enabled(startEnabled) {}

    // Grouped by size, largest first, so the struct has no interior padding.
    std::vector<Light> lights;
    std::string bakeStatus;
    uint64_t bakedGeneration = UINT64_MAX; // storage generation the bake was made for

    int output = 0; // what the lighting pass shows
    float exposure = 1.0f;
    int environmentPreset = 2; // none, uniform, sky
    float environmentIntensity = 0.6f;
    int shadowMode = 0; // off, PCSS, ray traced, baked SH
    int shadowMapResolution = 1024;
    int raySamples = 4;
    int rtQuality = 0; // exact, balanced, fast
    int visibilityDebug = 0; // off, |error| vs reference, sample locations
    int bakeDirections = 256;

    bool enabled = false; // relighting on
    bool progressiveShadows = true;
    bool placed = false; // default lights placed for the current scene
};

// Draws the Relight window and pushes its settings to the renderer. Call once
// per frame between ImGui::NewFrame() and ImGui::Render().
void drawRelightPanel(Renderer & renderer, RelightPanelState & state);

} // namespace sabercat::app
