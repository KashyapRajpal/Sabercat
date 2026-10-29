// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/command_buffer.h"

#include "sabercat/vk_check.h"

namespace sabercat
{

CommandBuffer::CommandBuffer(VkDevice device, VkCommandPool pool)
    : m_device(device), m_pool(pool), m_commandBuffer(VK_NULL_HANDLE)
{
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = pool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(device, &allocInfo, &m_commandBuffer),
             "Failed to allocate command buffer");
}

CommandBuffer::~CommandBuffer()
{
    // Command buffers are automatically freed when the command pool is destroyed
}

void CommandBuffer::begin(VkCommandBufferUsageFlags flags)
{
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = flags;
    VK_CHECK(vkBeginCommandBuffer(m_commandBuffer, &beginInfo), "Failed to begin command buffer");
}

void CommandBuffer::end()
{
    VK_CHECK(vkEndCommandBuffer(m_commandBuffer), "Failed to end command buffer");
}

void CommandBuffer::dispatchCompute(VkPipeline pipeline, VkPipelineLayout layout,
                                    VkDescriptorSet set, uint32_t groupCountX, uint32_t groupCountY,
                                    uint32_t groupCountZ)
{
    vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(m_commandBuffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            layout,
                            /*firstSet=*/0,
                            1,
                            &set,
                            0,
                            nullptr);
    vkCmdDispatch(m_commandBuffer, groupCountX, groupCountY, groupCountZ);
}

void CommandBuffer::dispatchComputeIndirect(VkPipeline pipeline, VkPipelineLayout layout,
                                            VkDescriptorSet set, VkBuffer argsBuffer,
                                            VkDeviceSize offset)
{
    vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(m_commandBuffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            layout,
                            /*firstSet=*/0,
                            1,
                            &set,
                            0,
                            nullptr);
    vkCmdDispatchIndirect(m_commandBuffer, argsBuffer, offset);
}

void CommandBuffer::pushConstants(VkPipelineLayout layout, VkShaderStageFlags stageFlags,
                                  uint32_t size, const void * data)
{
    vkCmdPushConstants(m_commandBuffer, layout, stageFlags, 0, size, data);
}

void CommandBuffer::copyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size)
{
    VkBufferCopy region{};
    region.size = size;
    vkCmdCopyBuffer(m_commandBuffer, src, dst, 1, &region);
}

void CommandBuffer::imageBarrier(VkImage image, VkImageLayout oldLayout, VkImageLayout newLayout,
                                 VkAccessFlags srcAccessMask, VkAccessFlags dstAccessMask)
{
    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.srcAccessMask = srcAccessMask;
    barrier.dstAccessMask = dstAccessMask;
    vkCmdPipelineBarrier(m_commandBuffer,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0,
                         0,
                         nullptr,
                         0,
                         nullptr,
                         1,
                         &barrier);
}

void CommandBuffer::bufferBarrier(VkBuffer buffer, VkAccessFlags srcAccessMask,
                                  VkAccessFlags dstAccessMask)
{
    VkBufferMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer = buffer;
    barrier.offset = 0;
    barrier.size = VK_WHOLE_SIZE;
    barrier.srcAccessMask = srcAccessMask;
    barrier.dstAccessMask = dstAccessMask;
    vkCmdPipelineBarrier(m_commandBuffer,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0,
                         0,
                         nullptr,
                         1,
                         &barrier,
                         0,
                         nullptr);
}

} // namespace sabercat
