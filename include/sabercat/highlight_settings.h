// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace sabercat
{

// The look of the selection highlight (Renderer::setSelectionHighlight). 20 bytes,
// no padding: one 4-byte flag followed by four 4-byte floats.
struct HighlightSettings
{
    // 0 = no highlight (the default); nonzero = the selected Gaussians are tinted.
    uint32_t enabled = 0;
    // tint[0..2] is the display-referred sRGB tint color; tint[3] is the blend
    // weight a, in [0, 1]: a selected Gaussian's color moves toward the tint by a.
    float tint[4] = {1.0f, 0.55f, 0.0f, 0.5f};
};
static_assert(sizeof(HighlightSettings) == 20, "HighlightSettings is 20 bytes, no padding");

} // namespace sabercat
