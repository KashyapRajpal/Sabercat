// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vulkan/vulkan.h>

#include <vma/vk_mem_alloc.h>

#include <cstdint>

namespace sabercat
{

class Device;

// A VMA-backed 2D image. The forward rasterizer's offscreen output target
// is its first consumer -- one workgroup-per-tile writes
// into it as a storage image, and Renderer::present() copies it into the
// acquired swapchain image.
class Image
{
  public:
    // Default usage covers the offscreen rasterizer target: written as a
    // storage image, read back via a copy for verification/present.
    Image(Device & device, uint32_t width, uint32_t height,
         VkFormat format = VK_FORMAT_R8G8B8A8_UNORM,
         VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    ~Image();

    // Non-copyable: it owns a unique VMA allocation.
    Image(const Image &) = delete;
    Image & operator=(const Image &) = delete;

    uint32_t getWidth() const
    {
        return m_width;
    }
    uint32_t getHeight() const
    {
        return m_height;
    }
    VkFormat getFormat() const
    {
        return m_format;
    }

    // Get Vulkan handles
    VkImage getHandle() const
    {
        return m_image;
    }
    VkImageView getImageView() const
    {
        return m_imageView;
    }

    // Transition image layout. Uses the same blunt-but-correct
    // ALL_COMMANDS_BIT stage mask as CommandBuffer::imageBarrier -- fine
    // for the low dispatch counts here; worth narrowing once a profiler
    // says so. Access masks are inferred from the layout pair (covers the
    // transitions this engine actually uses: UNDEFINED/GENERAL/
    // TRANSFER_SRC/TRANSFER_DST).
    void transitionLayout(VkCommandBuffer cmd, VkImageLayout oldLayout,
                          VkImageLayout newLayout) const;

  private:
    Device * m_device = nullptr;
    uint32_t m_width;
    uint32_t m_height;
    VkFormat m_format;
    VkImage m_image = VK_NULL_HANDLE;
    VkImageView m_imageView = VK_NULL_HANDLE;
    VmaAllocation m_allocation = VK_NULL_HANDLE;
};

} // namespace sabercat
