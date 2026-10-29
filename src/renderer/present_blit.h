// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The copy present() makes from the rendered image into the swapchain image.
// Internal: Renderer::present() calls it, and examples/renderer_test calls it
// directly to check the byte contract without a window.
#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>

namespace sabercat::internal
{

// Records a nearest-filtered blit of `source` (TRANSFER_SRC_OPTIMAL) into
// `destination` (TRANSFER_DST_OPTIMAL), scaling the source rectangle to the
// destination rectangle. The source holds display-referred color and the
// destination is a UNORM format, so the bytes written are round(v * 255).
void recordPresentBlit(VkCommandBuffer cmd, VkImage source, uint32_t sourceWidth,
                       uint32_t sourceHeight, VkImage destination, uint32_t destinationWidth,
                       uint32_t destinationHeight);

} // namespace sabercat::internal
