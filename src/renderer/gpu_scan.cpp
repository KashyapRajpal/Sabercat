// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "gpu_scan.h"

#include <algorithm>
#include <vector>

#include "sabercat/buffer.h"
#include "sabercat/device.h"
#include "sabercat/render_config.h"
#include "sabercat/shader_compiler.h"
#include "sabercat/vk_check.h"

namespace sabercat
{

namespace
{

// prefix_sum.slang's ScanParams. Five 4-byte scalars, so the C++ and std430
// layouts agree with no padding.
struct ScanParams
{
    uint32_t elementCount; // used when countSlot == kCountFromParams
    uint32_t countSlot;    // else: uint index into dispatchState holding the count
    uint32_t inputStride;  // in uints
    uint32_t inputOffset;  // in uints, within each element
    uint32_t phase;        // 0 block scan, 1 block-sum scan, 2 add
};
static_assert(sizeof(ScanParams) == 5 * sizeof(uint32_t),
              "ScanParams must match prefix_sum.slang's std430 layout with no padding");

// prefix_sum.slang's SCAN_COUNT_FROM_PARAMS: the count is params.elementCount.
constexpr uint32_t kCountFromParams = 0xFFFFFFFFu;
constexpr uint32_t kScanPhaseCount = 3;
// Phase 1 scans the block sums in a single workgroup.
constexpr uint32_t kBlockSumsPhase = 1;

// Binding slots, as prefix_sum.slang declares them.
constexpr uint32_t kBindingInput = 0;
constexpr uint32_t kBindingOutput = 1;
constexpr uint32_t kBindingBlockSums = 2;
constexpr uint32_t kBindingDispatchState = 3;
// Only the shader's declared size matters here: the state is never read.
constexpr VkDeviceSize kDispatchStateBytes = 16;

VkDescriptorSetLayoutBinding storageBufferBinding(uint32_t binding)
{
    VkDescriptorSetLayoutBinding b{};
    b.binding = binding;
    b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    return b;
}

// Compute -> compute dependency between phases, as forward_pass.cpp records
// it: a global memory barrier rather than per-buffer ones.
void computeBarrier(VkCommandBuffer cmd)
{
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT
                            | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0, nullptr, 0,
                         nullptr);
}

} // namespace

GpuScan::GpuScan(Device & device, const std::string & shaderDir)
    : m_device(device), m_descriptors(device.handle())
{
    VkDevice dev = device.handle();
    // A throw below (a missing shader file is the common one) skips the
    // destructor, so the pipeline created so far is released here. The
    // descriptor objects and buffers clean up through their own destructors.
    try
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings = {
            storageBufferBinding(kBindingInput),
            storageBufferBinding(kBindingOutput),
            storageBufferBinding(kBindingBlockSums),
            storageBufferBinding(kBindingDispatchState),
        };
        m_layout = m_descriptors.createLayout(bindings);
        m_descriptors.createPool(1, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4}});
        m_set = m_descriptors.allocate(m_layout);

        ShaderCompiler compiler(dev);
        VkShaderModule module = compiler.createShaderModule(
            compiler.loadSPIRV(shaderDir + "/prefix_sum.comp.spv"));
        VkPipelineLayout layout = m_pipeline.createPipelineLayout(dev, m_layout, sizeof(ScanParams),
                                                                  VK_SHADER_STAGE_COMPUTE_BIT);
        m_pipeline.createComputePipeline(dev, module, layout);
        compiler.destroyShaderModule(module);

        m_dispatchState = std::make_unique<Buffer>(device, kDispatchStateBytes);
        m_blockSums = std::make_unique<Buffer>(device, sizeof(uint32_t));
        m_descriptors.bindBuffer(m_set, kBindingDispatchState, m_dispatchState->handle(), 0,
                                 m_dispatchState->size(), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    }
    catch (...)
    {
        m_pipeline.destroy(dev);
        throw;
    }
}

GpuScan::~GpuScan()
{
    m_pipeline.destroy(m_device.handle());
}

void GpuScan::ensureBlockSums(uint32_t count)
{
    // One block total per workgroup; at least one uint so the binding is valid.
    const VkDeviceSize neededBytes =
        std::max<VkDeviceSize>(dispatchGroups(count, kScanBlock), 1) * sizeof(uint32_t);
    if (m_blockSums && m_blockSums->size() >= neededBytes)
    {
        return;
    }
    // Geometric growth, never shrunk, as ForwardPass's ensureCapacity does.
    const VkDeviceSize grownBytes = m_blockSums ? m_blockSums->size() * 2 : 0;
    m_blockSums = std::make_unique<Buffer>(m_device, std::max(neededBytes, grownBytes));
}

void GpuScan::record(VkCommandBuffer cmd, VkBuffer input, VkBuffer output, uint32_t count,
                     uint32_t inputStrideUints, uint32_t inputOffsetUints)
{
    if (count == 0)
    {
        return;
    }
    VK_VALIDATE(inputStrideUints > 0, "GpuScan: the input stride must be at least one uint");

    // Grow before recording, so the binding below names the buffer that will
    // be used. Safe to replace it here: see the header on pending submissions.
    ensureBlockSums(count);

    m_descriptors.bindBuffer(m_set, kBindingInput, input, 0, VK_WHOLE_SIZE,
                             VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    m_descriptors.bindBuffer(m_set, kBindingOutput, output, 0, VK_WHOLE_SIZE,
                             VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    m_descriptors.bindBuffer(m_set, kBindingBlockSums, m_blockSums->handle(), 0,
                             m_blockSums->size(), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

    // Pipeline and set stay bound across the three dispatches; only the phase
    // push constant changes.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline.getHandle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline.getLayout(), 0, 1,
                            &m_set, 0, nullptr);

    ScanParams params{};
    params.elementCount = count;
    params.countSlot = kCountFromParams;
    params.inputStride = inputStrideUints;
    params.inputOffset = inputOffsetUints;

    const uint32_t blocks = dispatchGroups(count, kScanBlock);
    for (uint32_t phase = 0; phase < kScanPhaseCount; ++phase)
    {
        params.phase = phase;
        vkCmdPushConstants(cmd, m_pipeline.getLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                           sizeof(params), &params);
        vkCmdDispatch(cmd, phase == kBlockSumsPhase ? 1 : blocks, 1, 1);
        computeBarrier(cmd);
    }
}

} // namespace sabercat
