// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vulkan/vulkan.h>

#include <vma/vk_mem_alloc.h>

#include <cstddef>

#include "sabercat/buffer_view.h"

namespace sabercat
{

class Device;

// A VMA-backed GPU buffer with staging-buffer upload/download. Foundation for
// every SoA Gaussian attribute buffer (PERFORMANCE.md design rule 1).
class Buffer
{
  public:
    // Default usage covers what every SSBO in this engine needs: storage
    // binding plus both transfer directions for staging upload/download.
    Buffer(Device & device, VkDeviceSize size,
           VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                      | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                      | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    ~Buffer();

    // Non-copyable: it owns a unique VMA allocation.
    Buffer(const Buffer &) = delete;
    Buffer & operator=(const Buffer &) = delete;

    VkDeviceSize size() const
    {
        return m_size;
    }
    VkBuffer handle() const
    {
        return m_buffer;
    }

    // A non-owning view of this whole buffer, for passing to compute passes
    // (the buffer seam -- passes bind views, not owned Buffers).
    // `stride` is the caller's element size; it isn't intrinsic to the
    // allocation.
    BufferView view(uint32_t stride, VkDeviceSize offset = 0) const
    {
        return BufferView{m_buffer, offset, m_size - offset, stride};
    }

    // Copies `size` bytes from `data` into this buffer via a temporary
    // host-visible staging buffer + immediateSubmit copy.
    void upload(const void * data, size_t size);

    // Copies `size` bytes from this buffer into `data` via a temporary
    // host-visible staging buffer + immediateSubmit copy. const: it only
    // copies GPU bytes into the caller's pointer, no observable state on
    // this object changes.
    void download(void * data, size_t size) const;

  private:
    Device * m_device = nullptr;
    VkBuffer m_buffer = VK_NULL_HANDLE;
    VmaAllocation m_allocation = VK_NULL_HANDLE;
    VkDeviceSize m_size = 0;
};

// Copies `size` bytes starting at `view.offset` into `data`, the same
// way Buffer::download does. Works on any buffer view -- owned or borrowed
// (application-owned) -- as long as its buffer has TRANSFER_SRC usage.
void downloadView(Device & device, const BufferView & view, void * data, size_t size);

} // namespace sabercat
