// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Swapchain format choice, factored out of Renderer::Impl::createSwapchain so
// it can be unit-tested without a device.
//
// The display contract this serves: every image present() blits is already in
// display space (the forward color is sRGB-encoded, like the training photos).
// The swapchain must therefore be a UNORM format, so the hardware stores those
// bytes as they are. An _SRGB swapchain would encode them a second time.
#pragma once

#include <vulkan/vulkan.h>

#include <vector>

namespace sabercat
{

// Preference order: B8G8R8A8_UNORM, then R8G8B8A8_UNORM (both with
// VK_COLOR_SPACE_SRGB_NONLINEAR_KHR); otherwise the surface's first format.
// Returns a value with VK_FORMAT_UNDEFINED for an empty list (the caller
// rejects surfaces with no formats before getting here).
inline VkSurfaceFormatKHR chooseSurfaceFormat(const std::vector<VkSurfaceFormatKHR> & formats)
{
    if (formats.empty())
    {
        return VkSurfaceFormatKHR{};
    }
    const VkFormat preferred[] = {VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM};
    for (const VkFormat wanted : preferred)
    {
        for (const VkSurfaceFormatKHR & f : formats)
        {
            if (f.format == wanted && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
            {
                return f;
            }
        }
    }
    return formats.front();
}

// True for the sRGB-encoding formats the display path must avoid.
inline bool isSrgbSurfaceFormat(VkFormat format)
{
    return format == VK_FORMAT_B8G8R8A8_SRGB || format == VK_FORMAT_R8G8B8A8_SRGB;
}

} // namespace sabercat
