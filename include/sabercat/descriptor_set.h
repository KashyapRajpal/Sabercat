// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vulkan/vulkan.h>

#include <vector>

namespace sabercat
{

// Owns a descriptor pool and any layouts it creates. bindBuffer/bindImage
// write vkUpdateDescriptorSets immediately (no batching), which is enough
// for the per-kernel descriptor sets this engine allocates.
class DescriptorSet
{
  public:
    DescriptorSet(VkDevice device);
    ~DescriptorSet();

    // Non-copyable: it owns pool + layout handles.
    DescriptorSet(const DescriptorSet &) = delete;
    DescriptorSet & operator=(const DescriptorSet &) = delete;

    // Create a descriptor set layout from its bindings. Owned by this
    // DescriptorSet (destroyed alongside it).
    VkDescriptorSetLayout createLayout(const std::vector<VkDescriptorSetLayoutBinding> & bindings);

    // Create descriptor pool
    void createPool(uint32_t maxSets, const std::vector<VkDescriptorPoolSize> & poolSizes);

    // Allocate descriptor set
    VkDescriptorSet allocate(VkDescriptorSetLayout layout);

    // Bind buffer to descriptor set
    void bindBuffer(VkDescriptorSet descriptorSet, uint32_t binding, VkBuffer buffer,
                    VkDeviceSize offset, VkDeviceSize range, VkDescriptorType type);

    // Bind image to descriptor set
    void bindImage(VkDescriptorSet descriptorSet, uint32_t binding, VkImageView imageView,
                   VkSampler sampler, VkDescriptorType type);

    // Bind a top-level acceleration structure (a
    // GaussianAccelerationStructure's tlas()) to a
    // VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR binding -- the Slang
    // RaytracingAccelerationStructure a ray-query kernel traces against.
    void bindAccelerationStructure(VkDescriptorSet descriptorSet, uint32_t binding,
                                   VkAccelerationStructureKHR accelerationStructure);

    VkDescriptorPool getPool() const
    {
        return m_pool;
    }

  private:
    VkDevice m_device;
    VkDescriptorPool m_pool;
    std::vector<VkDescriptorSetLayout> m_layouts;
};

} // namespace sabercat
