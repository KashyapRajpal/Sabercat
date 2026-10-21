// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/pipeline.h"

#include "sabercat/vk_check.h"

namespace sabercat
{

Pipeline::Pipeline() : m_device(VK_NULL_HANDLE), m_pipeline(VK_NULL_HANDLE)
{
}

Pipeline::~Pipeline()
{
    // Cleanup is done via destroy() method
}

VkPipelineLayout Pipeline::createPipelineLayout(VkDevice device, VkDescriptorSetLayout setLayout,
                                                uint32_t pushConstantSize,
                                                VkShaderStageFlags pushConstantStages)
{
    VkPushConstantRange range{};
    range.stageFlags = pushConstantStages;
    range.offset = 0;
    range.size = pushConstantSize;

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &setLayout;
    if (pushConstantSize > 0)
    {
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &range;
    }

    VK_CHECK(vkCreatePipelineLayout(device, &layoutInfo, nullptr, &m_pipelineLayout),
             "Failed to create pipeline layout");
    return m_pipelineLayout;
}

void Pipeline::createComputePipeline(VkDevice device, VkShaderModule shader,
                                     VkPipelineLayout layout)
{
    m_device = device;

    VkPipelineShaderStageCreateInfo shaderStageInfo{};
    shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageInfo.module = shader;
    shaderStageInfo.pName = "main";

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.layout = layout;
    pipelineInfo.stage = shaderStageInfo;

    VK_CHECK(
        vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline),
        "Failed to create compute pipeline");
}

void Pipeline::createGraphicsPipeline(VkDevice device, const std::vector<VkShaderModule> & shaders,
                                      VkPipelineLayout layout, VkRenderPass renderPass)
{
    m_device = device;

    // TODO: Create graphics pipeline
    // This would involve:
    // - Creating shader stage info for each shader
    // - Setting up vertex input state
    // - Setting up input assembly state
    // - Setting up viewport and scissor state
    // - Setting up rasterizer state
    // - Setting up multisampling state
    // - Setting up color blend state
    // - Creating the full graphics pipeline
}

void Pipeline::destroy(VkDevice device)
{
    if (m_pipeline != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device, m_pipeline, nullptr);
        m_pipeline = VK_NULL_HANDLE;
    }
    if (m_pipelineLayout != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device, m_pipelineLayout, nullptr);
        m_pipelineLayout = VK_NULL_HANDLE;
    }
}

} // namespace sabercat
