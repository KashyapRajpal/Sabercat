// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <string>

#include "sabercat/descriptor_set.h"
#include "sabercat/pipeline.h"

namespace sabercat
{

class Buffer;
class Device;

// Internal to the renderer (not installed): a standalone device-wide exclusive
// prefix sum of uint32 values, over shaders/prefix_sum.slang. The same three-phase
// scan ForwardPass records for its tile-count and radix-offset scans, copied
// rather than shared so ForwardPass stays untouched. Used by the Studio edit
// kernels (EditPass::exclusiveScan) and their tests.
//
// Submissions: record() rebinds the descriptor set's input, output and block-sum
// bindings with vkUpdateDescriptorSets on every call. That is legal only while
// no earlier submission that uses the set is still pending, so every submission
// must complete before the next record(). EditPass satisfies this by recording
// and submitting through Device::immediateSubmit, which waits on its fence.
// Recording a second scan into a command buffer that is not yet submitted would
// retarget the first one's bindings, so record one scan per submission.
class GpuScan
{
  public:
    GpuScan(Device & device, const std::string & shaderDir);
    ~GpuScan();

    // Non-copyable: it owns the pipeline, pool and device buffers.
    GpuScan(const GpuScan &) = delete;
    GpuScan & operator=(const GpuScan &) = delete;

    // Records an exclusive scan of `count` elements into `cmd`:
    //   output[i] = sum of input[j * inputStrideUints + inputOffsetUints] for j < i.
    // `input` and `output` must hold at least count uints (the input at the stride
    // and offset given) and must be created with the default Buffer usage. Both
    // are bound whole (offset 0). count == 0 records nothing.
    void record(VkCommandBuffer cmd, VkBuffer input, VkBuffer output, uint32_t count,
                uint32_t inputStrideUints = 1, uint32_t inputOffsetUints = 0);

  private:
    // Grows m_blockSums to hold one uint per workgroup of `count` elements.
    void ensureBlockSums(uint32_t count);

    Device & m_device;
    DescriptorSet m_descriptors;
    VkDescriptorSetLayout m_layout = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;
    Pipeline m_pipeline;
    // Binding 3: InstanceDispatch-shaped state the shader reads only when a push
    // constant's countSlot is not the count-from-params sentinel. Never read here.
    std::unique_ptr<Buffer> m_dispatchState;
    // Binding 2: one uint per workgroup's block total.
    std::unique_ptr<Buffer> m_blockSums;
};

} // namespace sabercat
