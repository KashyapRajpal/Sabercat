// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace sabercat
{

class Pipeline
{
  public:
    Pipeline();
    ~Pipeline();

    // Build a pipeline layout from a single descriptor set layout, plus an
    // optional push-constant range (pushConstantSize = 0 means none). Every
    // A kernel's per-dispatch parameters (camera, sort params, ...) go
    // in as push constants rather than a uniform-buffer Buffer -- avoids a
    // synchronous staging-buffer upload per dispatch. Owned by this
    // Pipeline (destroyed alongside it in destroy()).
    VkPipelineLayout createPipelineLayout(VkDevice device, VkDescriptorSetLayout setLayout,
                                          uint32_t pushConstantSize = 0,
                                          VkShaderStageFlags pushConstantStages
                                          = VK_SHADER_STAGE_COMPUTE_BIT);

    // Create compute pipeline from shader module
    void createComputePipeline(VkDevice device, VkShaderModule shader, VkPipelineLayout layout);

    // Create graphics pipeline (for rendering)
    void createGraphicsPipeline(VkDevice device, const std::vector<VkShaderModule> & shaders,
                                VkPipelineLayout layout, VkRenderPass renderPass);

    // Get Vulkan handle
    VkPipeline getHandle() const
    {
        return m_pipeline;
    }
    VkPipelineLayout getLayout() const
    {
        return m_pipelineLayout;
    }

    // Destroy pipeline + pipeline layout.
    void destroy(VkDevice device);

  private:
    VkDevice m_device;
    VkPipeline m_pipeline;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
};

} // namespace sabercat
