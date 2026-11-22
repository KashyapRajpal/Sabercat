// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The GPU side of the compare window: one borrowed sabercat::Device, the directory of
// the compare kernels, and the helpers the compositor and the metrics share (shader
// modules, compute pipelines, one-shot submission, readback). Everything here is
// synchronous: each call waits for the GPU before it returns.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "sabercat/buffer_view.h"
#include "sabercat/descriptor_set.h"
#include "sabercat/pipeline.h"

namespace sabercat
{
class Device;
}

namespace sabercat::compare::gui
{

class GpuContext;

// A compute pipeline with one descriptor set. Binding i has the type given at
// construction (storage buffers or storage images, all in set 0, all visible to the
// compute stage). The kernel owns its pipeline, pipeline layout, descriptor layout and
// descriptor pool, and destroys them in its destructor. It must be destroyed before the
// device.
class ComputeKernel
{
  public:
    ComputeKernel(GpuContext & context, const std::string & shaderName,
                  const std::vector<VkDescriptorType> & bindingTypes, uint32_t pushConstantSize);
    ~ComputeKernel();

    ComputeKernel(const ComputeKernel &) = delete;
    ComputeKernel & operator=(const ComputeKernel &) = delete;

    // Binds the whole range of a view (its offset and size) to a storage buffer binding.
    void bindBuffer(uint32_t binding, const BufferView & view);
    // Binds an image view to a storage image binding.
    void bindImage(uint32_t binding, VkImageView view);

    // Records bind pipeline, bind descriptors, push (when pushSize > 0) and dispatch.
    // The bindings must be set before the command buffer is submitted.
    void record(VkCommandBuffer cmd, uint32_t groupsX, uint32_t groupsY, uint32_t groupsZ,
                const void * pushData = nullptr, uint32_t pushSize = 0) const;

  private:
    VkDevice m_device = VK_NULL_HANDLE;
    std::vector<VkDescriptorType> m_bindingTypes;
    DescriptorSet m_descriptors;
    Pipeline m_pipeline;
    VkDescriptorSet m_set = VK_NULL_HANDLE;
    uint32_t m_pushConstantSize = 0;
};

class GpuContext
{
  public:
    // device is borrowed and must outlive this object. shaderDir holds the compiled
    // <name>.comp.spv files.
    GpuContext(Device & device, std::string shaderDir);
    ~GpuContext();

    GpuContext(const GpuContext &) = delete;
    GpuContext & operator=(const GpuContext &) = delete;

    Device & device() const
    {
        return m_device;
    }
    VkDevice handle() const;
    const std::string & shaderDir() const
    {
        return m_shaderDir;
    }

    // The shader module of <shaderDir>/<name>.comp.spv, created on first use and cached.
    // The context owns it and destroys it in its destructor. Throws std::runtime_error
    // if the file is missing or malformed.
    VkShaderModule loadShader(const std::string & name);

    // A compute kernel for <name>.comp.spv (see ComputeKernel).
    std::unique_ptr<ComputeKernel> makeComputePipeline(const std::string & name,
                                                       const std::vector<VkDescriptorType> & bindingTypes,
                                                       uint32_t pushConstantSize);

    // Runs record() in a one-shot command buffer and waits for it (Device::immediateSubmit).
    void submit(const std::function<void(VkCommandBuffer)> & record);

    // Copies `size` bytes from the start of a view into data, waiting for the copy.
    void readBuffer(const BufferView & view, void * data, size_t size);

  private:
    Device & m_device;
    std::string m_shaderDir;
    std::map<std::string, VkShaderModule> m_shaders;
};

} // namespace sabercat::compare::gui
