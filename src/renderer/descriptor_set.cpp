// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/descriptor_set.h"

#include "sabercat/vk_check.h"

namespace sabercat
{

DescriptorSet::DescriptorSet(VkDevice device) : m_device(device), m_pool(VK_NULL_HANDLE)
{
}

DescriptorSet::~DescriptorSet()
{
    // Destroying the pool implicitly frees every set allocated from it.
    if (m_pool != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(m_device, m_pool, nullptr);
    }
    for (VkDescriptorSetLayout layout : m_layouts)
    {
        vkDestroyDescriptorSetLayout(m_device, layout, nullptr);
    }
}

VkDescriptorSetLayout DescriptorSet::createLayout(
    const std::vector<VkDescriptorSetLayoutBinding> & bindings)
{
    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    VkDescriptorSetLayout layout;
    VK_CHECK(vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &layout),
             "Failed to create descriptor set layout");
    m_layouts.push_back(layout);
    return layout;
}

void DescriptorSet::createPool(uint32_t maxSets,
                               const std::vector<VkDescriptorPoolSize> & poolSizes)
{
    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.maxSets = maxSets;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();

    VK_CHECK(vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_pool),
             "Failed to create descriptor pool");
}

VkDescriptorSet DescriptorSet::allocate(VkDescriptorSetLayout layout)
{
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_pool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &layout;

    VkDescriptorSet descriptorSet;
    VK_CHECK(vkAllocateDescriptorSets(m_device, &allocInfo, &descriptorSet),
             "Failed to allocate descriptor set");
    return descriptorSet;
}

void DescriptorSet::bindBuffer(VkDescriptorSet descriptorSet, uint32_t binding, VkBuffer buffer,
                               VkDeviceSize offset, VkDeviceSize range, VkDescriptorType type)
{
    VkDescriptorBufferInfo bufferInfo{};
    bufferInfo.buffer = buffer;
    bufferInfo.offset = offset;
    bufferInfo.range = range;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptorSet;
    write.dstBinding = binding;
    write.descriptorCount = 1;
    write.descriptorType = type;
    write.pBufferInfo = &bufferInfo;

    vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
}

void DescriptorSet::bindImage(VkDescriptorSet descriptorSet, uint32_t binding,
                              VkImageView imageView, VkSampler sampler, VkDescriptorType type)
{
    VkDescriptorImageInfo imageInfo{};
    imageInfo.imageView = imageView;
    imageInfo.sampler = sampler;
    imageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = descriptorSet;
    write.dstBinding = binding;
    write.descriptorCount = 1;
    write.descriptorType = type;
    write.pImageInfo = &imageInfo;

    vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
}

void DescriptorSet::bindAccelerationStructure(VkDescriptorSet descriptorSet, uint32_t binding,
                                              VkAccelerationStructureKHR accelerationStructure)
{
    VkWriteDescriptorSetAccelerationStructureKHR asInfo{};
    asInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    asInfo.accelerationStructureCount = 1;
    asInfo.pAccelerationStructures = &accelerationStructure;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.pNext = &asInfo;
    write.dstSet = descriptorSet;
    write.dstBinding = binding;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

    vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);
}

} // namespace sabercat
