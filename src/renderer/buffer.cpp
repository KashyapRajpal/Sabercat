// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/buffer.h"

#include <cstring>

#include "sabercat/device.h"
#include "sabercat/vk_check.h"

namespace sabercat
{

namespace
{

// A short-lived host-visible buffer used only to shuttle bytes across the
// upload/download copy. Never bound as an SSBO, never outlives one call.
struct StagingBuffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    void * mapped = nullptr;
};

StagingBuffer createStaging(Device & device, VkDeviceSize size, VkBufferUsageFlags usage,
                            VmaAllocationCreateFlags hostAccessFlag)
{
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    // MAPPED_BIT keeps the allocation persistently mapped so we can memcpy
    // through VmaAllocationInfo::pMappedData without a separate vmaMapMemory.
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags = hostAccessFlag | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    StagingBuffer staging;
    VmaAllocationInfo info{};
    VK_CHECK(vmaCreateBuffer(device.allocator(),
                             &bufferInfo,
                             &allocInfo,
                             &staging.buffer,
                             &staging.allocation,
                             &info),
             "Failed to create staging buffer");
    staging.mapped = info.pMappedData;
    return staging;
}

void destroyStaging(Device & device, StagingBuffer & staging)
{
    vmaDestroyBuffer(device.allocator(), staging.buffer, staging.allocation);
    staging.buffer = VK_NULL_HANDLE;
    staging.allocation = VK_NULL_HANDLE;
    staging.mapped = nullptr;
}

} // namespace

Buffer::Buffer(Device & device, VkDeviceSize size, VkBufferUsageFlags usage)
    : m_device(&device), m_size(size)
{
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    // VMA_MEMORY_USAGE_AUTO lets VMA pick the right heap from usage flags +
    // device type; the old VMA_MEMORY_USAGE_GPU_ONLY is deprecated.
    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

    VK_CHECK(vmaCreateBuffer(
                 m_device->allocator(), &bufferInfo, &allocInfo, &m_buffer, &m_allocation, nullptr),
             "Failed to create buffer");
}

Buffer::~Buffer()
{
    if (m_buffer != VK_NULL_HANDLE)
    {
        vmaDestroyBuffer(m_device->allocator(), m_buffer, m_allocation);
    }
}

void Buffer::upload(const void * data, size_t size)
{
    StagingBuffer staging = createStaging(*m_device,
                                          size,
                                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                                          VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

    std::memcpy(staging.mapped, data, size);
    // Not all host-visible memory is coherent; flush makes the CPU write
    // visible to the GPU before we submit the copy that reads it.
    vmaFlushAllocation(m_device->allocator(), staging.allocation, 0, size);

    m_device->immediateSubmit(
        [&](VkCommandBuffer cmd)
        {
            VkBufferCopy region{};
            region.size = size;
            vkCmdCopyBuffer(cmd, staging.buffer, m_buffer, 1, &region);
        });

    destroyStaging(*m_device, staging);
}

void Buffer::download(void * data, size_t size) const
{
    downloadView(*m_device, view(1), data, size);
}

void downloadView(Device & device, const BufferView & view, void * data, size_t size)
{
    StagingBuffer staging = createStaging(device,
                                          size,
                                          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                          VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

    device.immediateSubmit(
        [&](VkCommandBuffer cmd)
        {
            VkBufferCopy region{};
            region.srcOffset = view.offset;
            region.size = size;
            vkCmdCopyBuffer(cmd, view.buffer, staging.buffer, 1, &region);
        });

    // Mirror of flush: makes the GPU's write visible to the CPU read below.
    vmaInvalidateAllocation(device.allocator(), staging.allocation, 0, size);
    std::memcpy(data, staging.mapped, size);

    destroyStaging(device, staging);
}

} // namespace sabercat
