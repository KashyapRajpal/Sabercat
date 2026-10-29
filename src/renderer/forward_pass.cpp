// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/forward_pass.h"

#include <algorithm>
#include <cstddef>
#include <cstring>

#include "sabercat/buffer.h"
#include "sabercat/command_buffer.h"
#include "sabercat/device.h"
#include "sabercat/render_config.h"
#include "sabercat/shader_compiler.h"
#include "sabercat/vk_check.h"

namespace sabercat
{

namespace
{

static_assert(sizeof(ProjectedGaussian) == 48,
             "ProjectedGaussian must match forward_common.slangh's std430 layout");
static_assert(sizeof(TileRange) == 8, "TileRange must match a GPU uint2");
static_assert(sizeof(CameraUniforms) <= 128,
             "CameraUniforms is pushed as a push constant; 128B is the Vulkan-guaranteed minimum");
static_assert(sizeof(LODParams) == 44,
             "LODParams must match forward_common.slangh's std430 layout");

// Mirrors forward_common.slangh's InstanceDispatch: forward_project's atomic
// instance total, then the sizes instance_dispatch.slang derives from it,
// including the two indirect dispatch commands run() records against this
// buffer. All 4-byte scalars (VkDispatchIndirectCommand is three uint32s),
// so std430 and C++ agree with no padding beyond the explicit tail.
struct InstanceDispatch
{
    uint32_t totalInstances; // forward_project's InterlockedAdd target -- must be first
    uint32_t instanceCount;  // totalInstances, or 0 when it overflowed capacity
    uint32_t sortGroups;
    uint32_t counterCount;
    VkDispatchIndirectCommand instanceGroups;
    VkDispatchIndirectCommand counterScanGroups;
    uint32_t pad0;
    uint32_t pad1;
};
static_assert(sizeof(InstanceDispatch) == 48,
             "InstanceDispatch must match forward_common.slangh's std430 layout");
static_assert(offsetof(InstanceDispatch, totalInstances) == 0,
             "forward_project adds into element 0 of this buffer viewed as uints");
static_assert(offsetof(InstanceDispatch, instanceGroups) % 4 == 0
                  && offsetof(InstanceDispatch, counterScanGroups) % 4 == 0,
             "vkCmdDispatchIndirect offsets must be 4-byte aligned");

// Per-dispatch parameter structs pushed as push constants. Layouts must
// match the corresponding shader struct exactly (std430-esque scalar
// packing -- every field here is a 4-byte uint, so there's no alignment
// subtlety like the vec3-padding elsewhere in this codebase).
struct InstanceDispatchParams
{
    uint32_t instanceCapacity;
};

// prefix_sum.slang's ScanParams.
struct ScanParams
{
    uint32_t elementCount; // used when countSlot == kScanCountFromParams
    uint32_t countSlot;    // else: uint index into InstanceDispatch
    uint32_t inputStride;  // in uints
    uint32_t inputOffset;  // in uints
    uint32_t phase;        // 0 block scan, 1 block-sum scan, 2 add
};

struct SortParams
{
    uint32_t bitShift;
    uint32_t keySelect; // 0 = depth word, 1 = tile word
};

constexpr uint32_t kScanCountFromParams = 0xFFFFFFFFu;

// On a capacity overflow the instance-sized buffers grow to the count plus
// 1/kInstanceHeadroomDivisor (and at least double -- see ensureCapacity), so
// an orbit drifting toward denser views doesn't overflow again next frame.
constexpr uint32_t kInstanceHeadroomDivisor = 4;
constexpr uint32_t kScanPhaseCount = 3;

// The tile-count scan reads projected[i].tileCount straight out of the
// ProjectedGaussian array, viewed as uints.
static_assert(sizeof(ProjectedGaussian) % sizeof(uint32_t) == 0
                  && offsetof(ProjectedGaussian, tileCount) % sizeof(uint32_t) == 0,
             "the tile-count scan strides over ProjectedGaussian in uints");
constexpr uint32_t kProjectedStrideUints = sizeof(ProjectedGaussian) / sizeof(uint32_t);
constexpr uint32_t kTileCountOffsetUints = offsetof(ProjectedGaussian, tileCount) / sizeof(uint32_t);
constexpr uint32_t kCounterCountSlot = offsetof(InstanceDispatch, counterCount) / sizeof(uint32_t);

// Compute -> compute (and -> indirect-argument read) dependency between the
// GPU-driven stages. A global memory barrier rather than per-buffer ones:
// the scan and sort stages hand several buffers along at once, and on the
// hardware this targets the two cost the same.
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

// The three prefix_sum phases, each followed by a barrier. Phase 1 is
// always a single workgroup; phases 0 and 2 take one workgroup per
// SCAN_BLOCK elements -- `directGroups` when the host knows the length,
// else the counterScanGroups command in `indirectArgs` (GPU-sized).
void recordScan(CommandBuffer & cmd, const Pipeline & pipeline, VkDescriptorSet set,
                ScanParams params, VkBuffer indirectArgs, uint32_t directGroups)
{
    for (uint32_t phase = 0; phase < kScanPhaseCount; ++phase)
    {
        params.phase = phase;
        cmd.pushConstants(pipeline.getLayout(), VK_SHADER_STAGE_COMPUTE_BIT, sizeof(params),
                          &params);
        if (phase == 1)
        {
            cmd.dispatchCompute(pipeline.getHandle(), pipeline.getLayout(), set, 1, 1, 1);
        }
        else if (indirectArgs != VK_NULL_HANDLE)
        {
            cmd.dispatchComputeIndirect(pipeline.getHandle(), pipeline.getLayout(), set,
                                        indirectArgs,
                                        offsetof(InstanceDispatch, counterScanGroups));
        }
        else
        {
            cmd.dispatchCompute(pipeline.getHandle(), pipeline.getLayout(), set, directGroups, 1,
                                1);
        }
        computeBarrier(cmd.getHandle());
    }
}

VkDescriptorSetLayoutBinding storageBufferBinding(uint32_t binding)
{
    VkDescriptorSetLayoutBinding b{};
    b.binding = binding;
    b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    return b;
}

VkDescriptorSetLayoutBinding storageImageBinding(uint32_t binding)
{
    VkDescriptorSetLayoutBinding b{};
    b.binding = binding;
    b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    return b;
}

VkShaderModule loadKernel(ShaderCompiler & compiler, const std::string & dir, const char * name)
{
    auto spirv = compiler.loadSPIRV(dir + "/" + name + ".comp.spv");
    return compiler.createShaderModule(spirv);
}

// Allocates (or grows) a device-local storage buffer to at least
// `neededBytes`, geometric growth, never shrunk. Returns true if a
// reallocation happened (caller must rebind whatever descriptor sets
// reference it).
bool ensureCapacity(Device & device, std::unique_ptr<Buffer> & buffer, VkDeviceSize neededBytes)
{
    if (buffer && buffer->size() >= neededBytes)
    {
        return false;
    }
    VkDeviceSize newSize = std::max<VkDeviceSize>(neededBytes, buffer ? buffer->size() * 2 : 0);
    buffer = std::make_unique<Buffer>(device, newSize);
    return true;
}

} // namespace

ForwardPass::ForwardPass(Device & device, uint32_t width, uint32_t height,
                         const std::string & shaderDir, DepthLayout depthLayout)
    : m_device(device), m_width(0), m_height(0), m_projectDescriptors(device.handle()),
      m_dispatchDescriptors(device.handle()), m_prefixSumDescriptors(device.handle()),
      m_tileBinDescriptors(device.handle()), m_histogramDescriptors(device.handle()),
      m_scatterDescriptors(device.handle()), m_tileRangesDescriptors(device.handle()),
      m_rasterizeDescriptors(device.handle())
{
    // A throw anywhere below (a missing shader file is the common one) skips
    // the destructor, so the raw handles created so far are released here.
    try
    {
        // Before resize(): the depth buffer's size comes from the layout.
        m_depthLayout = depthLayout;

        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = device.submitQueueFamily();
        VK_CHECK(vkCreateCommandPool(device.handle(), &poolInfo, nullptr, &m_commandPool),
                 "Failed to create ForwardPass command pool");

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VK_CHECK(vkCreateFence(device.handle(), &fenceInfo, nullptr, &m_fence),
                 "Failed to create ForwardPass fence");

        createPipelines(shaderDir);
        createFixedBuffers();
        resize(width, height);
    }
    catch (...)
    {
        release();
        throw;
    }
}

ForwardPass::~ForwardPass()
{
    release();
}

void ForwardPass::release()
{
    VkDevice dev = m_device.handle();

    m_projectPipeline.destroy(dev);
    m_dispatchPipeline.destroy(dev);
    m_prefixSumPipeline.destroy(dev);
    m_tileBinPipeline.destroy(dev);
    m_histogramPipeline.destroy(dev);
    m_scatterPipeline.destroy(dev);
    m_tileRangesPipeline.destroy(dev);
    m_rasterizePipeline.destroy(dev);

    if (m_instanceCountReadback != VK_NULL_HANDLE)
    {
        vmaDestroyBuffer(m_device.allocator(), m_instanceCountReadback,
                         m_instanceCountReadbackAllocation);
    }

    if (m_fence != VK_NULL_HANDLE)
    {
        vkDestroyFence(dev, m_fence, nullptr);
    }
    if (m_commandPool != VK_NULL_HANDLE)
    {
        vkDestroyCommandPool(dev, m_commandPool, nullptr);
    }
}

void ForwardPass::createPipelines(const std::string & shaderDir)
{
    VkDevice dev = m_device.handle();
    ShaderCompiler compiler(dev);

    // -- forward_project: means, scales, rotations, opacities,
    // shCoefficients, projected(RW), instanceDispatch(RW, the atomic total
    // in its first uint), previousTileRanges
    // (density feedback -- aliases m_tileRanges, see
    // resizeTileBuffers), lodParams (LOD tuning knobs), shDegrees(RW,
    // SH debug view); camera pushed. --
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings = {
            storageBufferBinding(0), storageBufferBinding(1), storageBufferBinding(2),
            storageBufferBinding(3), storageBufferBinding(4), storageBufferBinding(5),
            storageBufferBinding(6), storageBufferBinding(7), storageBufferBinding(8),
            storageBufferBinding(9),
        };
        m_projectLayout = m_projectDescriptors.createLayout(bindings);
        m_projectDescriptors.createPool(1, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10}});
        m_projectSet = m_projectDescriptors.allocate(m_projectLayout);

        VkShaderModule module = loadKernel(compiler, shaderDir, "forward_project");
        VkPipelineLayout layout = m_projectPipeline.createPipelineLayout(
            dev, m_projectLayout, sizeof(CameraUniforms), VK_SHADER_STAGE_COMPUTE_BIT);
        m_projectPipeline.createComputePipeline(dev, module, layout);
        compiler.destroyShaderModule(module);
    }

    // -- instance_dispatch: instanceDispatch(RW); capacity pushed. --
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings = {storageBufferBinding(0)};
        m_dispatchLayout = m_dispatchDescriptors.createLayout(bindings);
        m_dispatchDescriptors.createPool(1, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}});
        m_dispatchSet = m_dispatchDescriptors.allocate(m_dispatchLayout);

        VkShaderModule module = loadKernel(compiler, shaderDir, "instance_dispatch");
        VkPipelineLayout layout = m_dispatchPipeline.createPipelineLayout(
            dev, m_dispatchLayout, sizeof(InstanceDispatchParams), VK_SHADER_STAGE_COMPUTE_BIT);
        m_dispatchPipeline.createComputePipeline(dev, module, layout);
        compiler.destroyShaderModule(module);
    }

    // -- prefix_sum: input, output(RW), blockSums(RW), instanceDispatch;
    // params pushed. Two sets: the per-Gaussian tile-count scan and the radix
    // sort's digit-offset scan. --
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings = {
            storageBufferBinding(0),
            storageBufferBinding(1),
            storageBufferBinding(2),
            storageBufferBinding(3),
        };
        m_prefixSumLayout = m_prefixSumDescriptors.createLayout(bindings);
        m_prefixSumDescriptors.createPool(2, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8}});
        m_prefixSumSet = m_prefixSumDescriptors.allocate(m_prefixSumLayout);
        m_counterScanSet = m_prefixSumDescriptors.allocate(m_prefixSumLayout);

        VkShaderModule module = loadKernel(compiler, shaderDir, "prefix_sum");
        VkPipelineLayout layout = m_prefixSumPipeline.createPipelineLayout(
            dev, m_prefixSumLayout, sizeof(ScanParams), VK_SHADER_STAGE_COMPUTE_BIT);
        m_prefixSumPipeline.createComputePipeline(dev, module, layout);
        compiler.destroyShaderModule(module);
    }

    // -- tile_binning: projected, instanceOffsets, keysTile/keysDepth/values
    // (RW, always writes set A), instanceDispatch; camera pushed. --
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings = {
            storageBufferBinding(0), storageBufferBinding(1), storageBufferBinding(2),
            storageBufferBinding(3), storageBufferBinding(4), storageBufferBinding(5),
        };
        m_tileBinLayout = m_tileBinDescriptors.createLayout(bindings);
        m_tileBinDescriptors.createPool(1, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6}});
        m_tileBinSet = m_tileBinDescriptors.allocate(m_tileBinLayout);

        VkShaderModule module = loadKernel(compiler, shaderDir, "tile_binning");
        VkPipelineLayout layout = m_tileBinPipeline.createPipelineLayout(
            dev, m_tileBinLayout, sizeof(CameraUniforms), VK_SHADER_STAGE_COMPUTE_BIT);
        m_tileBinPipeline.createComputePipeline(dev, module, layout);
        compiler.destroyShaderModule(module);
    }

    // -- radix_sort_histogram: keysTile(src), keysDepth(src), counters(RW),
    // instanceDispatch; params pushed. Two set variants selected by pass
    // parity. --
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings = {
            storageBufferBinding(0),
            storageBufferBinding(1),
            storageBufferBinding(2),
            storageBufferBinding(3),
        };
        m_histogramLayout = m_histogramDescriptors.createLayout(bindings);
        m_histogramDescriptors.createPool(2, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8}});
        m_histogramSetAtoB = m_histogramDescriptors.allocate(m_histogramLayout);
        m_histogramSetBtoA = m_histogramDescriptors.allocate(m_histogramLayout);

        VkShaderModule module = loadKernel(compiler, shaderDir, "radix_sort_histogram");
        VkPipelineLayout layout = m_histogramPipeline.createPipelineLayout(
            dev, m_histogramLayout, sizeof(SortParams), VK_SHADER_STAGE_COMPUTE_BIT);
        m_histogramPipeline.createComputePipeline(dev, module, layout);
        compiler.destroyShaderModule(module);
    }

    // (The radix sort's step 2, the digit-offset scan, is prefix_sum above.)

    // -- radix_sort_scatter: keysTile(src), keysDepth(src), values(src),
    // offsets(src, the scanned counters), outKeysTile/outKeysDepth/outValues
    // (dst, RW), instanceDispatch; params pushed. Two set variants selected
    // by pass parity. --
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings = {
            storageBufferBinding(0), storageBufferBinding(1), storageBufferBinding(2),
            storageBufferBinding(3), storageBufferBinding(4), storageBufferBinding(5),
            storageBufferBinding(6), storageBufferBinding(7),
        };
        m_scatterLayout = m_scatterDescriptors.createLayout(bindings);
        m_scatterDescriptors.createPool(2, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16}});
        m_scatterSetAtoB = m_scatterDescriptors.allocate(m_scatterLayout);
        m_scatterSetBtoA = m_scatterDescriptors.allocate(m_scatterLayout);

        VkShaderModule module = loadKernel(compiler, shaderDir, "radix_sort_scatter");
        VkPipelineLayout layout = m_scatterPipeline.createPipelineLayout(
            dev, m_scatterLayout, sizeof(SortParams), VK_SHADER_STAGE_COMPUTE_BIT);
        m_scatterPipeline.createComputePipeline(dev, module, layout);
        compiler.destroyShaderModule(module);
    }

    // -- tile_ranges: sortedKeysTile (always set A -- see the ping-pong
    // parity note in forward_pass.h), tileRanges(RW), instanceDispatch; no
    // push constants (the count is the GPU's). --
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings = {
            storageBufferBinding(0),
            storageBufferBinding(1),
            storageBufferBinding(2),
        };
        m_tileRangesLayout = m_tileRangesDescriptors.createLayout(bindings);
        m_tileRangesDescriptors.createPool(1, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3}});
        m_tileRangesSet = m_tileRangesDescriptors.allocate(m_tileRangesLayout);

        VkShaderModule module = loadKernel(compiler, shaderDir, "tile_ranges");
        VkPipelineLayout layout = m_tileRangesPipeline.createPipelineLayout(
            dev, m_tileRangesLayout, 0, VK_SHADER_STAGE_COMPUTE_BIT);
        m_tileRangesPipeline.createComputePipeline(dev, module, layout);
        compiler.destroyShaderModule(module);
    }

    // -- tile_rasterize: projected, sortedValues (set A), tileRanges,
    // outputImage(storage image), outputDepth(storage image),
    // batchesFetched (verification-only), pixelState (backward
    // state), outputBuffer (image view), outputDepthBuffer (depth
    // output); camera pushed. --
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings = {
            storageBufferBinding(0),
            storageBufferBinding(1),
            storageBufferBinding(2),
            storageImageBinding(3),
            storageImageBinding(4),
            storageBufferBinding(5),
            storageBufferBinding(6),
            storageBufferBinding(7),
            storageBufferBinding(8),
        };
        m_rasterizeLayout = m_rasterizeDescriptors.createLayout(bindings);
        m_rasterizeDescriptors.createPool(
            1, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2}});
        m_rasterizeSet = m_rasterizeDescriptors.allocate(m_rasterizeLayout);

        VkShaderModule module = loadKernel(compiler, shaderDir, "tile_rasterize");
        VkPipelineLayout layout = m_rasterizePipeline.createPipelineLayout(
            dev, m_rasterizeLayout, sizeof(CameraUniforms), VK_SHADER_STAGE_COMPUTE_BIT);
        m_rasterizePipeline.createComputePipeline(dev, module, layout);
        compiler.destroyShaderModule(module);
    }
}

void ForwardPass::createFixedBuffers()
{
    // One InstanceDispatch, device-local -- forward_project atomically adds
    // tile counts into its first uint every frame, instance_dispatch fills in
    // the rest, and the instance-sized stages dispatch indirectly from it.
    m_instanceDispatch = std::make_unique<Buffer>(
        m_device, sizeof(InstanceDispatch),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT
            | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    const auto bindDispatch = [this](DescriptorSet & descriptors, VkDescriptorSet set,
                                     uint32_t binding)
    {
        descriptors.bindBuffer(set, binding, m_instanceDispatch->handle(), 0,
                               m_instanceDispatch->size(), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    };
    bindDispatch(m_projectDescriptors, m_projectSet, 6);
    bindDispatch(m_dispatchDescriptors, m_dispatchSet, 0);
    bindDispatch(m_prefixSumDescriptors, m_prefixSumSet, 3);
    bindDispatch(m_prefixSumDescriptors, m_counterScanSet, 3);
    bindDispatch(m_tileBinDescriptors, m_tileBinSet, 5);
    bindDispatch(m_histogramDescriptors, m_histogramSetAtoB, 3);
    bindDispatch(m_histogramDescriptors, m_histogramSetBtoA, 3);
    bindDispatch(m_scatterDescriptors, m_scatterSetAtoB, 7);
    bindDispatch(m_scatterDescriptors, m_scatterSetBtoA, 7);
    bindDispatch(m_tileRangesDescriptors, m_tileRangesSet, 2);

    // Persistent host-visible readback target for the instance total,
    // created directly via VMA (Buffer's allocations are always device-local)
    // so every run() can copy it out at the end of its own submission
    // without a throwaway staging buffer per frame.
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = sizeof(uint32_t);
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VmaAllocationInfo info{};
    VK_CHECK(vmaCreateBuffer(m_device.allocator(), &bufferInfo, &allocInfo,
                             &m_instanceCountReadback, &m_instanceCountReadbackAllocation, &info),
             "Failed to create instance-count readback buffer");
    m_instanceCountReadbackMapped = static_cast<const uint32_t *>(info.pMappedData);

    // Seed the dynamic instance-sized buffers at a small starting capacity
    // so descriptor sets always have something valid to bind; run() grows
    // this on demand once it knows the real tile-instance count.
    ensureInstanceCapacity(1);

    // LOD knobs, disabled until setLODParams() is called. Bound here
    // (not in resizeTileBuffers/run()) since it never depends on framebuffer
    // size or gaussian count -- allocated and bound exactly once.
    // Binding 8 is (re)bound per run() from the mode -- see run().
    m_lodParams = std::make_unique<Buffer>(m_device, sizeof(LODParams));
    LODParams disabledLOD{};
    m_lodParams->upload(&disabledLOD, sizeof(disabledLOD));

    // The Differentiable-mode LODParams. Same all-zero contents as
    // the viewer buffer's initial state, but a different buffer that
    // setLODParams() never writes -- so a gradient run cannot see a viewer
    // knob no matter what was set before it.
    m_lodDisabled = std::make_unique<Buffer>(m_device, sizeof(LODParams));
    m_lodDisabled->upload(&disabledLOD, sizeof(disabledLOD));
}

void ForwardPass::resize(uint32_t width, uint32_t height)
{
    if (width == m_width && height == m_height)
    {
        return;
    }
    m_width = width;
    m_height = height;
    resizeTileBuffers(width, height);
}

void ForwardPass::resizeTileBuffers(uint32_t width, uint32_t height)
{
    m_tilesX = dispatchGroups(width, kTileSize);
    m_tilesY = dispatchGroups(height, kTileSize);

    m_tileRanges = std::make_unique<Buffer>(m_device,
                                            static_cast<VkDeviceSize>(m_tilesX) * m_tilesY
                                                * sizeof(TileRange));
    // Zero on creation, not just per-frame: run() only clears this buffer
    // right before tile_ranges writes it (see the comment there), but
    // forward_project's previousTileRanges read, earlier in the same
    // submission, happens BEFORE that -- so the very first run() after a
    // resize/construction would otherwise read whatever was already in this
    // freshly-allocated
    // VMA memory. Vulkan/VMA never guarantee zeroed allocations, so without
    // this, garbage bytes read as a TileRange could produce a huge bogus
    // instance count (e.g. `last - first` underflowing if first > last) and
    // spuriously over-cull for one frame. immediateSubmit is safe here --
    // this only runs at construction/resize, never per-frame.
    m_device.immediateSubmit(
        [this](VkCommandBuffer cmd)
        { vkCmdFillBuffer(cmd, m_tileRanges->handle(), 0, m_tileRanges->size(), 0); });
    // Verification-only counter buffer -- same tilesX*tilesY
    // lifecycle as m_tileRanges, cleared every frame in run() before
    // tile_rasterize runs.
    m_batchesFetched = std::make_unique<Buffer>(
        m_device, static_cast<VkDeviceSize>(m_tilesX) * m_tilesY * sizeof(uint32_t));
    // Backward state; only written by Differentiable runs, so its
    // contents are meaningful only after one (BackwardPass checks lastRun()).
    m_pixelState = std::make_unique<Buffer>(
        m_device, static_cast<VkDeviceSize>(width) * height * 2 * sizeof(uint32_t));
    // Image view; same Differentiable-only lifecycle as m_pixelState.
    m_outputBuffer = std::make_unique<Buffer>(
        m_device, static_cast<VkDeviceSize>(width) * height * sizeof(glm::vec4));
    // Depth output: n floats per pixel for the layout. Under None, a 16-byte
    // placeholder so binding 8 always names a valid buffer.
    const VkDeviceSize depthBytes =
        m_depthLayout == DepthLayout::None
            ? VkDeviceSize{16}
            : static_cast<VkDeviceSize>(width) * height * static_cast<uint32_t>(m_depthLayout)
                  * sizeof(float);
    m_depthBuffer = std::make_unique<Buffer>(m_device, depthBytes);
    m_lastRun = ForwardRunRecord{};
    m_outputImage = std::make_unique<Image>(m_device, width, height,
                                            VK_FORMAT_R32G32B32A32_SFLOAT,
                                            VK_IMAGE_USAGE_STORAGE_BIT
                                                | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    m_depthImage = std::make_unique<Image>(
        m_device, width, height, VK_FORMAT_R32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

    m_tileRangesDescriptors.bindBuffer(m_tileRangesSet,
                                       1,
                                       m_tileRanges->handle(),
                                       0,
                                       m_tileRanges->size(),
                                       VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    m_rasterizeDescriptors.bindBuffer(m_rasterizeSet,
                                      2,
                                      m_tileRanges->handle(),
                                      0,
                                      m_tileRanges->size(),
                                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    m_rasterizeDescriptors.bindImage(m_rasterizeSet,
                                     3,
                                     m_outputImage->getImageView(),
                                     VK_NULL_HANDLE,
                                     VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    m_rasterizeDescriptors.bindImage(m_rasterizeSet,
                                     4,
                                     m_depthImage->getImageView(),
                                     VK_NULL_HANDLE,
                                     VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    m_rasterizeDescriptors.bindBuffer(m_rasterizeSet,
                                      5,
                                      m_batchesFetched->handle(),
                                      0,
                                      m_batchesFetched->size(),
                                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    m_rasterizeDescriptors.bindBuffer(m_rasterizeSet,
                                      6,
                                      m_pixelState->handle(),
                                      0,
                                      m_pixelState->size(),
                                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    m_rasterizeDescriptors.bindBuffer(m_rasterizeSet,
                                      7,
                                      m_outputBuffer->handle(),
                                      0,
                                      m_outputBuffer->size(),
                                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    m_rasterizeDescriptors.bindBuffer(m_rasterizeSet,
                                      8,
                                      m_depthBuffer->handle(),
                                      0,
                                      m_depthBuffer->size(),
                                      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    // Density feedback: forward_project reads THIS SAME buffer as
    // "previous frame's" tile ranges -- see the binding-7 comment in
    // createPipelines() for why that's safe.
    m_projectDescriptors.bindBuffer(m_projectSet,
                                    7,
                                    m_tileRanges->handle(),
                                    0,
                                    m_tileRanges->size(),
                                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
}

void ForwardPass::ensureInstanceCapacity(uint32_t instanceCount)
{
    bool grew = false;
    grew |= ensureCapacity(m_device, m_sortA.keysTile, instanceCount * sizeof(uint32_t));
    grew |= ensureCapacity(m_device, m_sortA.keysDepth, instanceCount * sizeof(uint32_t));
    grew |= ensureCapacity(m_device, m_sortA.values, instanceCount * sizeof(uint32_t));
    grew |= ensureCapacity(m_device, m_sortB.keysTile, instanceCount * sizeof(uint32_t));
    grew |= ensureCapacity(m_device, m_sortB.keysDepth, instanceCount * sizeof(uint32_t));
    grew |= ensureCapacity(m_device, m_sortB.values, instanceCount * sizeof(uint32_t));

    uint32_t capacityElements =
        static_cast<uint32_t>(m_sortA.keysTile->size() / sizeof(uint32_t));
    uint32_t groupCount = dispatchGroups(capacityElements, kWorkgroupSize);
    VkDeviceSize counterElements = static_cast<VkDeviceSize>(kRadixBuckets) * groupCount;
    grew |= ensureCapacity(m_device, m_radixCounters, counterElements * sizeof(uint32_t));
    grew |= ensureCapacity(m_device, m_radixOffsets, counterElements * sizeof(uint32_t));

    if (grew)
    {
        m_instanceCapacity = capacityElements;
        rebindDynamicDescriptors();
    }
    ensureScanCapacity(counterElements);
}

void ForwardPass::ensureScanCapacity(VkDeviceSize elementCount)
{
    VkDeviceSize blocks = std::max<VkDeviceSize>((elementCount + kScanBlock - 1) / kScanBlock, 1);
    if (ensureCapacity(m_device, m_scanBlockSums, blocks * sizeof(uint32_t)))
    {
        for (VkDescriptorSet set : {m_prefixSumSet, m_counterScanSet})
        {
            m_prefixSumDescriptors.bindBuffer(set, 2, m_scanBlockSums->handle(), 0,
                                              m_scanBlockSums->size(),
                                              VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        }
    }
}

void ForwardPass::rebindDynamicDescriptors()
{
    auto bindStorage = [](DescriptorSet & set, VkDescriptorSet target, uint32_t binding,
                          Buffer & buffer)
    {
        set.bindBuffer(
            target, binding, buffer.handle(), 0, buffer.size(), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    };

    // tile_binning always writes set A.
    bindStorage(m_tileBinDescriptors, m_tileBinSet, 2, *m_sortA.keysTile);
    bindStorage(m_tileBinDescriptors, m_tileBinSet, 3, *m_sortA.keysDepth);
    bindStorage(m_tileBinDescriptors, m_tileBinSet, 4, *m_sortA.values);

    // histogram: (src keysTile, src keysDepth, shared counters) per parity.
    bindStorage(m_histogramDescriptors, m_histogramSetAtoB, 0, *m_sortA.keysTile);
    bindStorage(m_histogramDescriptors, m_histogramSetAtoB, 1, *m_sortA.keysDepth);
    bindStorage(m_histogramDescriptors, m_histogramSetAtoB, 2, *m_radixCounters);
    bindStorage(m_histogramDescriptors, m_histogramSetBtoA, 0, *m_sortB.keysTile);
    bindStorage(m_histogramDescriptors, m_histogramSetBtoA, 1, *m_sortB.keysDepth);
    bindStorage(m_histogramDescriptors, m_histogramSetBtoA, 2, *m_radixCounters);

    // digit-offset scan: counters -> offsets (shared by every pass).
    bindStorage(m_prefixSumDescriptors, m_counterScanSet, 0, *m_radixCounters);
    bindStorage(m_prefixSumDescriptors, m_counterScanSet, 1, *m_radixOffsets);

    // scatter: (src x3, offsets, dst x3) per parity.
    bindStorage(m_scatterDescriptors, m_scatterSetAtoB, 0, *m_sortA.keysTile);
    bindStorage(m_scatterDescriptors, m_scatterSetAtoB, 1, *m_sortA.keysDepth);
    bindStorage(m_scatterDescriptors, m_scatterSetAtoB, 2, *m_sortA.values);
    bindStorage(m_scatterDescriptors, m_scatterSetAtoB, 3, *m_radixOffsets);
    bindStorage(m_scatterDescriptors, m_scatterSetAtoB, 4, *m_sortB.keysTile);
    bindStorage(m_scatterDescriptors, m_scatterSetAtoB, 5, *m_sortB.keysDepth);
    bindStorage(m_scatterDescriptors, m_scatterSetAtoB, 6, *m_sortB.values);

    bindStorage(m_scatterDescriptors, m_scatterSetBtoA, 0, *m_sortB.keysTile);
    bindStorage(m_scatterDescriptors, m_scatterSetBtoA, 1, *m_sortB.keysDepth);
    bindStorage(m_scatterDescriptors, m_scatterSetBtoA, 2, *m_sortB.values);
    bindStorage(m_scatterDescriptors, m_scatterSetBtoA, 3, *m_radixOffsets);
    bindStorage(m_scatterDescriptors, m_scatterSetBtoA, 4, *m_sortA.keysTile);
    bindStorage(m_scatterDescriptors, m_scatterSetBtoA, 5, *m_sortA.keysDepth);
    bindStorage(m_scatterDescriptors, m_scatterSetBtoA, 6, *m_sortA.values);

    // Final sorted result always lands back in A after 8 (even) passes.
    bindStorage(m_tileRangesDescriptors, m_tileRangesSet, 0, *m_sortA.keysTile);
    bindStorage(m_rasterizeDescriptors, m_rasterizeSet, 1, *m_sortA.values);
}

const Buffer & ForwardPass::sortedKeysTile() const
{
    return *m_sortA.keysTile;
}
const Buffer & ForwardPass::sortedKeysDepth() const
{
    return *m_sortA.keysDepth;
}
const Buffer & ForwardPass::sortedValues() const
{
    return *m_sortA.values;
}

void ForwardPass::setLODParams(const LODParams & params)
{
    m_lodParams->upload(&params, sizeof(params));
}

void ForwardPass::run(const CanonicalGaussianBufferViews & views,
                      const CameraUniforms & callerCamera, ForwardStage stopAfter,
                      ForwardMode mode)
{
    VkDevice dev = m_device.handle();

    // SAVE_BACKWARD_STATE is ours, derived from the mode -- never trusted
    // from the caller.
    CameraUniforms camera = callerCamera;
    camera.renderFlags &= ~kRenderFlagSaveBackwardState;
    camera.renderFlags &= ~(kRenderFlagDepthFloatsMask << kRenderFlagDepthFloatsShift);
    if (mode == ForwardMode::Differentiable)
    {
        camera.renderFlags |= kRenderFlagSaveBackwardState;
        camera.renderFlags |= static_cast<uint32_t>(m_depthLayout) << kRenderFlagDepthFloatsShift;
    }

    // Viewer knobs reach forward_project only in Viewer mode; see ForwardMode.
    const Buffer & lod = mode == ForwardMode::Differentiable ? *m_lodDisabled : *m_lodParams;
    m_projectDescriptors.bindBuffer(m_projectSet, 8, lod.handle(), 0, lod.size(),
                                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

    // Recorded up front and marked complete only at the very end, so a run
    // that throws or stops early is never mistaken for a differentiable one.
    m_lastRun = ForwardRunRecord{};
    m_lastRun.valid = true;
    m_lastRun.mode = mode;
    m_lastRun.camera = camera;
    m_lastRun.views = views;

    // The gaussianCount-sized buffers are grow-only: densification
    // changes the count every few hundred steps, and the kernels bound
    // themselves by camera.gaussianCount, never by buffer size -- so only a
    // count beyond the current capacity reallocates (and rebinds everything
    // that references them).
    VK_VALIDATE(views.gaussianCount > 0, "ForwardPass::run: gaussianCount must be > 0");
    m_gaussianCount = views.gaussianCount;
    if (m_gaussianCount > m_gaussianCapacity)
    {
        m_gaussianCapacity = m_gaussianCount;
        m_projected = std::make_unique<Buffer>(
            m_device, static_cast<VkDeviceSize>(m_gaussianCapacity) * sizeof(ProjectedGaussian));
        m_instanceOffsets = std::make_unique<Buffer>(
            m_device, static_cast<VkDeviceSize>(m_gaussianCapacity) * sizeof(uint32_t));
        m_shDegrees = std::make_unique<Buffer>(
            m_device, static_cast<VkDeviceSize>(m_gaussianCapacity) * sizeof(float));
        m_projectDescriptors.bindBuffer(m_projectSet, 9, m_shDegrees->handle(), 0,
                                        m_shDegrees->size(), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

        m_projectDescriptors.bindBuffer(m_projectSet, 5, m_projected->handle(), 0,
                                        m_projected->size(), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        m_prefixSumDescriptors.bindBuffer(m_prefixSumSet, 0, m_projected->handle(), 0,
                                          m_projected->size(), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        m_prefixSumDescriptors.bindBuffer(m_prefixSumSet, 1, m_instanceOffsets->handle(), 0,
                                          m_instanceOffsets->size(),
                                          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        m_tileBinDescriptors.bindBuffer(m_tileBinSet, 0, m_projected->handle(), 0,
                                        m_projected->size(), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        m_tileBinDescriptors.bindBuffer(m_tileBinSet, 1, m_instanceOffsets->handle(), 0,
                                        m_instanceOffsets->size(),
                                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        m_rasterizeDescriptors.bindBuffer(m_rasterizeSet, 0, m_projected->handle(), 0,
                                          m_projected->size(), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        ensureScanCapacity(m_gaussianCapacity);
    }

    // views' buffer handles can change frame to frame (a different loaded
    // cloud) even when the count doesn't -- always rebind these.
    // Bound at the view's own offset (a borrowed view need not start at 0).
    const auto bindParam = [this](uint32_t binding, const BufferView & view)
    {
        m_projectDescriptors.bindBuffer(m_projectSet, binding, view.buffer, view.offset, view.size,
                                        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    };
    bindParam(0, views.means);
    bindParam(1, views.scales);
    bindParam(2, views.rotations);
    bindParam(3, views.opacity);
    bindParam(4, views.shCoefficients);

    VK_CHECK(vkResetCommandPool(dev, m_commandPool, 0), "Failed to reset ForwardPass command pool");

    // -- One submission: project, the tile-count scan, then every instance-
    // sized stage, all sized on the GPU (see the class comment). The
    // instance total rides back to the host in the same submission, so
    // lastInstanceCount() is current without an extra wait. --
    submitAndWait(
        [&](CommandBuffer & cmd)
        {
            VkCommandBuffer handle = cmd.getHandle();

            vkCmdFillBuffer(handle, m_instanceDispatch->handle(), 0, m_instanceDispatch->size(),
                            0);
            cmd.bufferBarrier(m_instanceDispatch->handle(), VK_ACCESS_TRANSFER_WRITE_BIT,
                              VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

            cmd.pushConstants(m_projectPipeline.getLayout(), VK_SHADER_STAGE_COMPUTE_BIT,
                              sizeof(CameraUniforms), &camera);
            cmd.dispatchCompute(m_projectPipeline.getHandle(), m_projectPipeline.getLayout(),
                                m_projectSet, dispatchGroups(m_gaussianCount, kWorkgroupSize), 1,
                                1);
            computeBarrier(handle);

            if (stopAfter != ForwardStage::Project)
            {
                // Exclusive scan of projected[].tileCount -> instanceOffsets.
                ScanParams scan{m_gaussianCount, kScanCountFromParams, kProjectedStrideUints,
                                kTileCountOffsetUints, 0};
                recordScan(cmd, m_prefixSumPipeline, m_prefixSumSet, scan, VK_NULL_HANDLE,
                           dispatchGroups(m_gaussianCount, kScanBlock));

                if (stopAfter != ForwardStage::PrefixSum)
                {
                    recordInstanceStages(cmd, camera, stopAfter);
                }
            }

            cmd.bufferBarrier(m_instanceDispatch->handle(),
                              VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                              VK_ACCESS_TRANSFER_READ_BIT);
            VkBufferCopy region{};
            region.srcOffset = offsetof(InstanceDispatch, totalInstances);
            region.size = sizeof(uint32_t);
            vkCmdCopyBuffer(handle, m_instanceDispatch->handle(), m_instanceCountReadback, 1,
                            &region);
        },
        "ForwardPass");

    vmaInvalidateAllocation(m_device.allocator(), m_instanceCountReadbackAllocation, 0,
                            sizeof(uint32_t));
    m_lastInstanceCount = *m_instanceCountReadbackMapped;

    // -- Capacity overflow: instance_dispatch zeroed the count, so the GPU
    // rendered nothing past the scan. Grow (with headroom, so a camera
    // drifting toward denser views doesn't overflow again next frame) and
    // re-record only the instance stages -- projected/instanceOffsets are
    // still valid, and forward_project isn't re-run, so the density
    // feedback it read is unaffected. Once per growth, not per frame. --
    const bool instanceStagesRan =
        stopAfter != ForwardStage::Project && stopAfter != ForwardStage::PrefixSum;
    if (instanceStagesRan && m_lastInstanceCount > m_instanceCapacity)
    {
        uint64_t withHeadroom = static_cast<uint64_t>(m_lastInstanceCount)
                                + m_lastInstanceCount / kInstanceHeadroomDivisor;
        ensureInstanceCapacity(
            static_cast<uint32_t>(std::min<uint64_t>(withHeadroom, UINT32_MAX)));
        submitAndWait([&](CommandBuffer & cmd) { recordInstanceStages(cmd, camera, stopAfter); },
                      "ForwardPass capacity re-run");
    }

    m_lastRun.completed = stopAfter == ForwardStage::Rasterize;
}

void ForwardPass::recordInstanceStages(CommandBuffer & cmd, const CameraUniforms & camera,
                                       ForwardStage stopAfter)
{
    VkCommandBuffer handle = cmd.getHandle();
    VkBuffer indirectArgs = m_instanceDispatch->handle();
    const VkDeviceSize instanceGroupsOffset = offsetof(InstanceDispatch, instanceGroups);

    // Size everything below from forward_project's total, on the GPU.
    InstanceDispatchParams dispatchParams{m_instanceCapacity};
    cmd.pushConstants(m_dispatchPipeline.getLayout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      sizeof(dispatchParams), &dispatchParams);
    cmd.dispatchCompute(m_dispatchPipeline.getHandle(), m_dispatchPipeline.getLayout(),
                        m_dispatchSet, 1, 1, 1);
    computeBarrier(handle);

    cmd.pushConstants(m_tileBinPipeline.getLayout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      sizeof(CameraUniforms), &camera);
    cmd.dispatchCompute(m_tileBinPipeline.getHandle(), m_tileBinPipeline.getLayout(),
                        m_tileBinSet, dispatchGroups(m_gaussianCount, kWorkgroupSize), 1, 1);
    if (stopAfter == ForwardStage::TileBinning)
    {
        return;
    }
    computeBarrier(handle);

    // 8 LSD radix passes: 4 over the depth word (least significant), then 4
    // over the tile word (most significant) -- must finish on the most-
    // significant word for the combined order to be stable. Each pass:
    // histogram -> digit-offset scan (prefix_sum's three phases) -> scatter,
    // the per-instance steps dispatched indirectly.
    ScanParams counterScan{0, kCounterCountSlot, 1, 0, 0};
    for (uint32_t pass = 0; pass < 2 * kRadixPasses; ++pass)
    {
        bool evenPass = (pass % 2) == 0;
        SortParams sortParams{
            (pass % kRadixPasses) * kRadixBits,
            pass < kRadixPasses ? 0u : 1u,
        };
        VkDescriptorSet histogramSet = evenPass ? m_histogramSetAtoB : m_histogramSetBtoA;
        VkDescriptorSet scatterSet = evenPass ? m_scatterSetAtoB : m_scatterSetBtoA;

        cmd.pushConstants(m_histogramPipeline.getLayout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          sizeof(sortParams), &sortParams);
        cmd.dispatchComputeIndirect(m_histogramPipeline.getHandle(),
                                    m_histogramPipeline.getLayout(), histogramSet, indirectArgs,
                                    instanceGroupsOffset);
        computeBarrier(handle);

        recordScan(cmd, m_prefixSumPipeline, m_counterScanSet, counterScan, indirectArgs, 0);

        cmd.pushConstants(m_scatterPipeline.getLayout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          sizeof(sortParams), &sortParams);
        cmd.dispatchComputeIndirect(m_scatterPipeline.getHandle(), m_scatterPipeline.getLayout(),
                                    scatterSet, indirectArgs, instanceGroupsOffset);
        computeBarrier(handle);
    }
    if (stopAfter == ForwardStage::RadixSort)
    {
        return;
    }

    // Tiles with no instances are never touched below -- clear first so they
    // read as an empty [0,0) range rather than last frame's span.
    vkCmdFillBuffer(handle, m_tileRanges->handle(), 0, m_tileRanges->size(), 0);
    cmd.bufferBarrier(m_tileRanges->handle(), VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_ACCESS_SHADER_WRITE_BIT);

    cmd.dispatchComputeIndirect(m_tileRangesPipeline.getHandle(), m_tileRangesPipeline.getLayout(),
                                m_tileRangesSet, indirectArgs, instanceGroupsOffset);
    if (stopAfter == ForwardStage::TileRanges)
    {
        return;
    }
    cmd.bufferBarrier(m_tileRanges->handle(), VK_ACCESS_SHADER_WRITE_BIT,
                      VK_ACCESS_SHADER_READ_BIT);

    // Verification-only: zero this frame's per-tile batch counters
    // before tile_rasterize accumulates into them.
    vkCmdFillBuffer(handle, m_batchesFetched->handle(), 0, m_batchesFetched->size(), 0);
    cmd.bufferBarrier(m_batchesFetched->handle(), VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    m_outputImage->transitionLayout(handle, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    m_depthImage->transitionLayout(handle, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    cmd.pushConstants(m_rasterizePipeline.getLayout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      sizeof(CameraUniforms), &camera);
    cmd.dispatchCompute(m_rasterizePipeline.getHandle(), m_rasterizePipeline.getLayout(),
                        m_rasterizeSet, m_tilesX, m_tilesY, 1);
}

void ForwardPass::submitAndWait(const std::function<void(CommandBuffer &)> & record,
                                const char * what)
{
    VkDevice dev = m_device.handle();

    CommandBuffer cmd(dev, m_commandPool);
    cmd.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    Device::recordGlobalBarrier(cmd.getHandle()); // see Device: queue contract
    record(cmd);
    Device::recordGlobalBarrier(cmd.getHandle()); // see Device: queue contract
    cmd.end();

    VkCommandBuffer handle = cmd.getHandle();
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &handle;
    VK_CHECK(vkResetFences(dev, 1, &m_fence), "Failed to reset ForwardPass fence");
    VK_CHECK(vkQueueSubmit(m_device.submitQueue(), 1, &submitInfo, m_fence),
             std::string("Failed to submit ") + what);
    VK_CHECK(vkWaitForFences(dev, 1, &m_fence, VK_TRUE, UINT64_MAX),
             std::string("Failed to wait for ") + what);
}

void ForwardPass::prepareOutputForRead(VkCommandBuffer cmd)
{
    // GENERAL is what run() always leaves the image in immediately after
    // rasterize; UNDEFINED would also be legal here (discard) but GENERAL
    // documents the actual precondition for this call.
    m_outputImage->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL,
                                    VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
}

void ForwardPass::downloadImageBytes(Image & image, void * dst, VkDeviceSize byteSize)
{
    VkDevice dev = m_device.handle();

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = byteSize;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
    allocInfo.flags =
        VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;

    VkBuffer staging = VK_NULL_HANDLE;
    VmaAllocation stagingAllocation = VK_NULL_HANDLE;
    VmaAllocationInfo info{};
    VK_CHECK(vmaCreateBuffer(m_device.allocator(), &bufferInfo, &allocInfo, &staging,
                             &stagingAllocation, &info),
             "Failed to create image readback staging buffer");

    VK_CHECK(vkResetCommandPool(dev, m_commandPool, 0),
             "Failed to reset ForwardPass command pool");
    submitAndWait(
        [&](CommandBuffer & cmd)
        {
            VkCommandBuffer handle = cmd.getHandle();

            // GENERAL is what run() always leaves both images in immediately
            // after rasterize.
            image.transitionLayout(handle, VK_IMAGE_LAYOUT_GENERAL,
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);

            VkBufferImageCopy region{};
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.layerCount = 1;
            region.imageExtent = {image.getWidth(), image.getHeight(), 1};
            vkCmdCopyImageToBuffer(handle, image.getHandle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   staging, 1, &region);

            // Back to GENERAL, the layout every consumer binds these images in
            // after a run (DebugVisualizer::record and present()'s prepare step
            // do no transition of their own from it). The next run() still
            // starts from UNDEFINED.
            image.transitionLayout(handle, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   VK_IMAGE_LAYOUT_GENERAL);
        },
        "image readback");

    vmaInvalidateAllocation(m_device.allocator(), stagingAllocation, 0, byteSize);
    std::memcpy(dst, info.pMappedData, byteSize);

    vmaDestroyBuffer(m_device.allocator(), staging, stagingAllocation);
}

std::vector<glm::vec4> ForwardPass::downloadOutputImage()
{
    std::vector<glm::vec4> result(static_cast<size_t>(m_width) * m_height);
    downloadImageBytes(*m_outputImage, result.data(), result.size() * sizeof(glm::vec4));
    return result;
}

std::vector<float> ForwardPass::downloadDepthImage()
{
    std::vector<float> result(static_cast<size_t>(m_width) * m_height);
    downloadImageBytes(*m_depthImage, result.data(), result.size() * sizeof(float));
    return result;
}

} // namespace sabercat
