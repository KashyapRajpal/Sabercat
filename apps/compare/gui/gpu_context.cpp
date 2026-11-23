// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "gpu_context.h"

#include <memory>
#include <stdexcept>
#include <utility>

#include "sabercat/buffer.h"
#include "sabercat/device.h"
#include "sabercat/shader_compiler.h"
#include "sabercat/vk_check.h"

namespace sabercat::compare::gui
{

// ---- ComputeKernel ---------------------------------------------------------------

ComputeKernel::ComputeKernel(GpuContext & context, const std::string & shaderName,
                             const std::vector<VkDescriptorType> & bindingTypes, uint32_t pushConstantSize)
    : m_device(context.handle()), m_bindingTypes(bindingTypes), m_descriptors(context.handle()),
      m_pushConstantSize(pushConstantSize)
{
    if (bindingTypes.empty())
    {
        throw std::invalid_argument("compute kernel " + shaderName + " needs at least one binding");
    }

    std::vector<VkDescriptorSetLayoutBinding> layoutBindings(bindingTypes.size());
    std::vector<VkDescriptorPoolSize> poolSizes;
    for (size_t i = 0; i < bindingTypes.size(); ++i)
    {
        VkDescriptorSetLayoutBinding & binding = layoutBindings[i];
        binding.binding = static_cast<uint32_t>(i);
        binding.descriptorType = bindingTypes[i];
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        binding.pImmutableSamplers = nullptr;

        // One pool entry per descriptor type, counting the bindings of that type.
        bool found = false;
        for (VkDescriptorPoolSize & size : poolSizes)
        {
            if (size.type == bindingTypes[i])
            {
                ++size.descriptorCount;
                found = true;
                break;
            }
        }
        if (!found)
        {
            poolSizes.push_back(VkDescriptorPoolSize{bindingTypes[i], 1});
        }
    }

    VkDescriptorSetLayout layout = m_descriptors.createLayout(layoutBindings);
    m_descriptors.createPool(1, poolSizes);
    m_set = m_descriptors.allocate(layout);

    VkShaderModule module = context.loadShader(shaderName);
    VkPipelineLayout pipelineLayout = m_pipeline.createPipelineLayout(
        m_device, layout, pushConstantSize, VK_SHADER_STAGE_COMPUTE_BIT);
    m_pipeline.createComputePipeline(m_device, module, pipelineLayout);
}

ComputeKernel::~ComputeKernel()
{
    m_pipeline.destroy(m_device);
}

void ComputeKernel::bindBuffer(uint32_t binding, const BufferView & view)
{
    if (binding >= m_bindingTypes.size())
    {
        throw std::out_of_range("compute kernel binding out of range");
    }
    m_descriptors.bindBuffer(m_set, binding, view.buffer, view.offset, view.size, m_bindingTypes[binding]);
}

void ComputeKernel::bindImage(uint32_t binding, VkImageView view)
{
    if (binding >= m_bindingTypes.size())
    {
        throw std::out_of_range("compute kernel binding out of range");
    }
    m_descriptors.bindImage(m_set, binding, view, VK_NULL_HANDLE, m_bindingTypes[binding]);
}

void ComputeKernel::record(VkCommandBuffer cmd, uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ,
                           const void * pushData, uint32_t pushSize) const
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline.getHandle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline.getLayout(), 0, 1, &m_set, 0,
                            nullptr);
    if (pushSize > 0)
    {
        if (pushSize != m_pushConstantSize || pushData == nullptr)
        {
            throw std::invalid_argument("push constants do not match the kernel's push constant size");
        }
        vkCmdPushConstants(cmd, m_pipeline.getLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0, pushSize, pushData);
    }
    vkCmdDispatch(cmd, groupsX, groupsY, groupsZ);
}

// ---- GpuContext ------------------------------------------------------------------

GpuContext::GpuContext(Device & device, std::string shaderDir) : m_device(device), m_shaderDir(std::move(shaderDir))
{
}

GpuContext::~GpuContext()
{
    for (const auto & entry : m_shaders)
    {
        vkDestroyShaderModule(m_device.handle(), entry.second, nullptr);
    }
}

VkDevice GpuContext::handle() const
{
    return m_device.handle();
}

VkShaderModule GpuContext::loadShader(const std::string & name)
{
    const auto found = m_shaders.find(name);
    if (found != m_shaders.end())
    {
        return found->second;
    }
    ShaderCompiler compiler(m_device.handle());
    const std::vector<uint32_t> spirv = compiler.loadSPIRV(m_shaderDir + "/" + name + ".comp.spv");
    VkShaderModule module = compiler.createShaderModule(spirv);
    m_shaders.emplace(name, module);
    return module;
}

std::unique_ptr<ComputeKernel> GpuContext::makeComputePipeline(const std::string & name,
                                                               const std::vector<VkDescriptorType> & bindingTypes,
                                                               uint32_t pushConstantSize)
{
    return std::make_unique<ComputeKernel>(*this, name, bindingTypes, pushConstantSize);
}

void GpuContext::submit(const std::function<void(VkCommandBuffer)> & record)
{
    m_device.immediateSubmit(record);
}

void GpuContext::readBuffer(const BufferView & view, void * data, size_t size)
{
    downloadView(m_device, view, data, size);
}

} // namespace sabercat::compare::gui
