// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <glm/glm.hpp>

#include <cstddef>

#include "sabercat/scene_bounds.h"

namespace sabercat
{

// Bounds of `count` world-space positions: the axis-aligned box over the finite
// positions, plus the robust framing target (median center, 90th-percentile
// radius). Non-finite positions are skipped. Returns SceneBounds{} (valid false)
// when count is 0 or no position is finite. Internal: GaussianCloud::computeBounds
// and the layer merge's host-side bounds (mergedLayerBounds) both use it, so the
// two always agree. Defined in gaussian_cloud.cpp.
SceneBounds computeSceneBounds(const glm::vec4 * positions, size_t count);

} // namespace sabercat
