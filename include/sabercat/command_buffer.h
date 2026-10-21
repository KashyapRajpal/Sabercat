// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vulkan/vulkan.h>

namespace sabercat
{

class CommandBuffer
{
  public:
    CommandBuffer(VkDevice device, VkCommandPool pool);
    ~CommandBuffer();

    // Get Vulkan handle
    VkCommandBuffer getHandle() const
    {
        return m_commandBuffer;
    }

    // Record commands
    void begin(VkCommandBufferUsageFlags flags = 0);
    void end();

    // Bind pipeline + descriptor set, then dispatch.
    void dispatchCompute(VkPipeline pipeline, VkPipelineLayout layout, VkDescriptorSet set,
                         uint32_t groupCountX, uint32_t groupCountY, uint32_t groupCountZ);
    // Same, with the group counts read by the GPU from a
    // VkDispatchIndirectCommand at `offset` in `argsBuffer` (which needs
    // VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT).
    void dispatchComputeIndirect(VkPipeline pipeline, VkPipelineLayout layout,
                                 VkDescriptorSet set, VkBuffer argsBuffer, VkDeviceSize offset);

    // Push per-dispatch parameters (camera, sort params, ...) inline in the
    // command buffer -- no Buffer/staging upload needed. Call before
    // dispatchCompute for kernels whose param struct is a
    // [vk::push_constant] ConstantBuffer<T> rather than a bound resource.
    void pushConstants(VkPipelineLayout layout, VkShaderStageFlags stageFlags, uint32_t size,
                       const void * data);

    // Copy buffer to buffer
    void copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size);

    // Transition image layout
    void imageBarrier(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout,
                      VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask);

    // Buffer barrier for synchronization
    void bufferBarrier(VkBuffer buffer, VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask);

  private:
    VkDevice m_device;
    VkCommandPool m_pool;
    VkCommandBuffer m_commandBuffer;
};

} // namespace sabercat
