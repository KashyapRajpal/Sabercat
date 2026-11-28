// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/forward_pass.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "blue_noise.h"
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

// sort_stable_update.slang's StableParams: the knobs of the sticky-key update,
// pushed each Stable run. Every field is 4 bytes, so the layout is padding-free.
struct StableParams
{
    uint32_t gaussianCount;
    float bucketsPerOctave;
    float hysteresis;
    uint32_t pad;
};
static_assert(sizeof(StableParams) == 16, "StableParams must match sort_stable_update.slang");

// stochastic_resolve.slang's ResolveParams: the image size and this frame's
// history weight. Four 4-byte fields, so the layout is padding-free.
struct ResolveParams
{
    uint32_t width;
    uint32_t height;
    float weight;
    uint32_t pad;
};
static_assert(sizeof(ResolveParams) == 16, "ResolveParams must match stochastic_resolve.slang");

// Ray-depth form: 12 floats per Gaussian (sort_common.slangh's layout).
constexpr uint32_t kRayFormFloats = 12;
constexpr VkDeviceSize kRayFormBytes = kRayFormFloats * sizeof(float);

// "No key yet" in the sticky keys (see sort_stable_update.slang).
constexpr uint32_t kStickySentinel = 0xFFFFFFFFu;

// The blue-noise rank table: a 64x64 permutation of 0..4095 (see blue_noise.h),
// uploaded once as uints. The params buffer is one uint4 read by the stochastic
// raster: x = the frame offset's bit pattern (a float), y = the noise frame index,
// z and w zero.
constexpr uint32_t kStochNoiseCount = kBlueNoiseSize * kBlueNoiseSize;
constexpr VkDeviceSize kStochNoiseBytes = kStochNoiseCount * sizeof(uint32_t);
constexpr VkDeviceSize kStochParamsBytes = 4 * sizeof(float);
// highlight_tint.slang's HighlightPush: five 4-byte scalars, no padding.
struct HighlightPush
{
    uint32_t count;
    float tintR;
    float tintG;
    float tintB;
    float tintA;
};
static_assert(sizeof(HighlightPush) == 20,
             "HighlightPush must match highlight_tint.slang's push constants");

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

// Timestamp slots of a stamped run, in pipeline order. Each stage's time is the
// difference between two neighbouring slots. In the default chain sortKeys and
// resolve are empty (tile_binning writes the keys, and the resolve slot sits on
// the rasterize stamp). Stochastic mode fills in both: its ray-form dispatch is
// the sortKeys work and its resolve dispatch is the resolve stage.
enum TimestampSlot : uint32_t
{
    kStampStart,
    kStampProject,
    kStampScan,
    kStampBinning,
    kStampSortKeys,
    kStampSort,
    kStampRanges,
    kStampRasterize,
    kStampResolve,
};
constexpr uint32_t kTimestampCount = kStampResolve + 1;

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

// Index into ForwardPass::m_windowStage for a window size K.
uint32_t windowStageIndex(uint32_t window)
{
    switch (window)
    {
    case 8:
        return 0;
    case 16:
        return 1;
    case 32:
        return 2;
    default:
        throw std::logic_error("ForwardPass: no window rasterizer for K = " + std::to_string(window));
    }
}

// The window rasterizer kernel built for window size K (see windowStageIndex).
const char * windowKernelName(uint32_t window)
{
    switch (window)
    {
    case 8:
        return "tile_rasterize_window_k8";
    case 16:
        return "tile_rasterize_window_k16";
    case 32:
        return "tile_rasterize_window_k32";
    default:
        throw std::logic_error("ForwardPass: no window rasterizer for K = " + std::to_string(window));
    }
}

} // namespace

// -- Sort settings (sort_settings.h). Free functions: they are pure value
// checks and names, needing no pass state. --

void validateSortSettings(const SortSettings & settings)
{
    const uint32_t mode = static_cast<uint32_t>(settings.mode);
    if (mode > static_cast<uint32_t>(SortMode::Stochastic))
    {
        throw std::invalid_argument("SortSettings::mode must be 0..5, got "
                                    + std::to_string(mode));
    }

    // Only PixelWindow needs a 16/32 window. Other modes still reject values
    // that are not a window size at all, so a stray value is caught early.
    const uint32_t window = settings.pixelWindow;
    if (settings.mode == SortMode::PixelWindow)
    {
        if (window != 16 && window != 32)
        {
            throw std::invalid_argument("SortSettings::pixelWindow must be 16 or 32 in "
                                        "pixel-window mode, got " + std::to_string(window));
        }
    }
    else if (window != 8 && window != 16 && window != 32)
    {
        throw std::invalid_argument("SortSettings::pixelWindow must be 8, 16 or 32, got "
                                    + std::to_string(window));
    }

    if (settings.stableBucketsPerOctave < 1 || settings.stableBucketsPerOctave > 65536)
    {
        throw std::invalid_argument("SortSettings::stableBucketsPerOctave must be in [1, 65536], got "
                                    + std::to_string(settings.stableBucketsPerOctave));
    }
    if (settings.stochasticSamples != 4 && settings.stochasticSamples != 8)
    {
        throw std::invalid_argument("SortSettings::stochasticSamples must be 4 or 8, got "
                                    + std::to_string(settings.stochasticSamples));
    }
    if (settings.temporalFrames < 1 || settings.temporalFrames > 4096)
    {
        throw std::invalid_argument("SortSettings::temporalFrames must be in [1, 4096], got "
                                    + std::to_string(settings.temporalFrames));
    }
    if (settings.temporalMode > 1)
    {
        throw std::invalid_argument("SortSettings::temporalMode must be 0 or 1, got "
                                    + std::to_string(settings.temporalMode));
    }
    // NaN fails every comparison, so the isfinite test is what rejects it.
    if (!std::isfinite(settings.stableHysteresis) || settings.stableHysteresis < 0.0f
        || settings.stableHysteresis > 16.0f)
    {
        throw std::invalid_argument("SortSettings::stableHysteresis must be finite and in [0, 16], got "
                                    + std::to_string(settings.stableHysteresis));
    }
    if (!std::isfinite(settings.temporalEmaWeight) || settings.temporalEmaWeight <= 0.0f
        || settings.temporalEmaWeight > 1.0f)
    {
        throw std::invalid_argument("SortSettings::temporalEmaWeight must be finite and in (0, 1], got "
                                    + std::to_string(settings.temporalEmaWeight));
    }
}

const char * sortModeName(SortMode mode)
{
    switch (mode)
    {
    case SortMode::Default:
        return "default";
    case SortMode::Stable:
        return "stable";
    case SortMode::TileRayDepth:
        return "tile-ray-depth";
    case SortMode::PixelWindow:
        return "pixel-window";
    case SortMode::Hybrid:
        return "hybrid";
    case SortMode::Stochastic:
        return "stochastic";
    }
    return "unknown";
}

bool sortModeIncompatibleWithLighting(SortMode mode)
{
    return mode == SortMode::PixelWindow || mode == SortMode::Hybrid
           || mode == SortMode::Stochastic;
}

ForwardPass::ForwardPass(Device & device, uint32_t width, uint32_t height,
                         const std::string & shaderDir, DepthLayout depthLayout)
    : m_device(device), m_width(0), m_height(0), m_projectDescriptors(device.handle()),
      m_dispatchDescriptors(device.handle()), m_prefixSumDescriptors(device.handle()),
      m_tileBinDescriptors(device.handle()), m_histogramDescriptors(device.handle()),
      m_scatterDescriptors(device.handle()), m_tileRangesDescriptors(device.handle()),
      m_rasterizeDescriptors(device.handle()), m_rayFormStage(device.handle()),
      m_stableUpdateStage(device.handle()), m_stableKeysStage(device.handle()),
      m_tileRayKeysStage(device.handle()),
      m_windowStage{OptionalStage(device.handle()), OptionalStage(device.handle()),
                    OptionalStage(device.handle())},
      m_stochRasterStage{OptionalStage(device.handle()), OptionalStage(device.handle())},
      m_stochResolveStage(device.handle()), m_highlightDescriptors(device.handle())
{
    // A throw anywhere below (a missing shader file is the common one) skips
    // the destructor, so the raw handles created so far are released here.
    try
    {
        // Kept for the highlight kernel, which is built on its first use.
        m_shaderDir = shaderDir;
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
    m_highlightPipeline.destroy(dev);

    if (m_instanceCountReadback != VK_NULL_HANDLE)
    {
        vmaDestroyBuffer(m_device.allocator(), m_instanceCountReadback,
                         m_instanceCountReadbackAllocation);
    }

    // Every OptionalStage member goes through destroyOptionalStage() here.
    destroyOptionalStage(m_rayFormStage);
    destroyOptionalStage(m_stableUpdateStage);
    destroyOptionalStage(m_stableKeysStage);
    destroyOptionalStage(m_tileRayKeysStage);
    for (OptionalStage & stage : m_windowStage)
    {
        destroyOptionalStage(stage);
    }
    for (OptionalStage & stage : m_stochRasterStage)
    {
        destroyOptionalStage(stage);
    }
    destroyOptionalStage(m_stochResolveStage);

    if (m_timingPool != VK_NULL_HANDLE)
    {
        vkDestroyQueryPool(dev, m_timingPool, nullptr);
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
    // Stable's history is restarted on a framebuffer change, so the first
    // frame at the new size is not sorted against keys from the old one. The
    // Stochastic history is per pixel, so it is restarted the same way.
    m_stickyReset = true;
    m_stochReset = true;
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
    // Stochastic's resolve history is one float4 per pixel. It is not allocated
    // here: default-mode users must not pay for it. ensureSortModeResources()
    // creates it on the first Stochastic run after this resize.
    m_stochHistory.reset();
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

void ForwardPass::setSortSettings(const SortSettings & settings)
{
    validateSortSettings(settings);
    // The sticky keys are only meaningful under the Stable parameters they were
    // built with, so entering or leaving Stable, or changing either knob,
    // starts them over.
    const bool stableChanged =
        (settings.mode == SortMode::Stable) != (m_sort.mode == SortMode::Stable)
        || settings.stableBucketsPerOctave != m_sort.stableBucketsPerOctave
        || settings.stableHysteresis != m_sort.stableHysteresis;
    if (stableChanged)
    {
        m_stickyReset = true;
    }
    // The Stochastic history is a blend of frames rendered under one set of
    // samples and weights, so any change to those (or entering/leaving the mode)
    // starts it over.
    const bool stochasticChanged =
        (settings.mode == SortMode::Stochastic) != (m_sort.mode == SortMode::Stochastic)
        || settings.stochasticSamples != m_sort.stochasticSamples
        || settings.temporalFrames != m_sort.temporalFrames
        || settings.temporalMode != m_sort.temporalMode
        || settings.temporalEmaWeight != m_sort.temporalEmaWeight;
    if (stochasticChanged)
    {
        m_stochReset = true;
    }
    m_sort = settings;
}

void ForwardPass::resetStableSort()
{
    m_stickyReset = true;
}

void ForwardPass::resetStochastic()
{
    m_stochReset = true;
}

const Buffer & ForwardPass::stableKeys() const
{
    if (!m_sticky)
    {
        throw std::runtime_error("ForwardPass::stableKeys: no Stable run yet");
    }
    return *m_sticky;
}

uint64_t ForwardPass::extraBufferBytes() const
{
    uint64_t bytes = 0;
    if (m_sticky)
    {
        bytes += m_sticky->size();
    }
    if (m_rayForm)
    {
        bytes += m_rayForm->size();
    }
    if (m_stochNoise)
    {
        bytes += m_stochNoise->size();
    }
    if (m_stochParams)
    {
        bytes += m_stochParams->size();
    }
    if (m_stochHistory)
    {
        bytes += m_stochHistory->size();
    }
    return bytes;
}

void ForwardPass::setStageTimingEnabled(bool enabled)
{
    m_timingEnabled = enabled;
    if (!enabled || m_timingPool != VK_NULL_HANDLE)
    {
        return;
    }

    // Each timestamp carries validBits meaningful bits, read from the submit
    // queue's family. Zero means the queue cannot write timestamps at all, so
    // no pool is created and every timing stays invalid.
    VkPhysicalDevice physical = m_device.physicalDevice();
    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physical, &familyCount, families.data());
    m_timestampValidBits = families.at(m_device.submitQueueFamily()).timestampValidBits;
    if (m_timestampValidBits == 0)
    {
        return;
    }

    VkQueryPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    poolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    poolInfo.queryCount = kTimestampCount;
    VK_CHECK(vkCreateQueryPool(m_device.handle(), &poolInfo, nullptr, &m_timingPool),
             "Failed to create stage-timing query pool");
}

void ForwardPass::stamp(VkCommandBuffer cmd, uint32_t index)
{
    if (!m_stampThisRun)
    {
        return;
    }
    // BOTTOM_OF_PIPE: the stamp is written once everything recorded before it
    // has finished, so each delta brackets the real GPU work of its stage.
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, m_timingPool, index);
}

void ForwardPass::readStageTimings()
{
    // The submission this reads has already been waited on, so WAIT is only a
    // formality that keeps the call's contract simple.
    std::array<uint64_t, kTimestampCount> ticks{};
    VK_CHECK(vkGetQueryPoolResults(m_device.handle(), m_timingPool, 0, kTimestampCount,
                                   sizeof(ticks), ticks.data(), sizeof(uint64_t),
                                   VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT),
             "Failed to read stage timestamps");

    // Only the low validBits of each counter are meaningful, so each delta is
    // taken modulo 2^validBits.
    const uint32_t validBits = m_timestampValidBits;
    const uint64_t mask = validBits >= 64 ? ~uint64_t{0} : (uint64_t{1} << validBits) - 1;
    const double nanosPerTick = m_device.properties().limits.timestampPeriod;
    const auto millisBetween = [&](uint32_t from, uint32_t to) -> float
    {
        const uint64_t delta = (ticks[to] - ticks[from]) & mask;
        return static_cast<float>(static_cast<double>(delta) * nanosPerTick / 1.0e6);
    };

    m_lastTimings.project = millisBetween(kStampStart, kStampProject);
    m_lastTimings.scan = millisBetween(kStampProject, kStampScan);
    m_lastTimings.binning = millisBetween(kStampScan, kStampBinning);
    m_lastTimings.sortKeys = millisBetween(kStampBinning, kStampSortKeys);
    m_lastTimings.sort = millisBetween(kStampSortKeys, kStampSort);
    m_lastTimings.ranges = millisBetween(kStampSort, kStampRanges);
    m_lastTimings.rasterize = millisBetween(kStampRanges, kStampRasterize);
    m_lastTimings.resolve = millisBetween(kStampRasterize, kStampResolve);
    m_lastTimings.total = millisBetween(kStampStart, kStampResolve);
    m_lastTimings.valid = true;
}

void ForwardPass::requireSortModeAvailable(SortMode mode) const
{
    // setSortSettings() validates the enum, so this only guards a run's
    // effective mode against a value that was never a SortMode.
    if (static_cast<uint32_t>(mode) > static_cast<uint32_t>(SortMode::Stochastic))
    {
        throw std::logic_error("ForwardPass: unknown sort mode "
                               + std::to_string(static_cast<uint32_t>(mode)));
    }
}

ForwardPass::OptionalStage & ForwardPass::ensureOptionalStage(
    OptionalStage & stage, const char * kernelName,
    const std::vector<VkDescriptorSetLayoutBinding> & bindings, uint32_t pushConstantSize)
{
    if (stage.built)
    {
        return stage;
    }
    VkDevice dev = m_device.handle();
    ShaderCompiler compiler(dev);

    stage.layout = stage.descriptors.createLayout(bindings);

    // One set per stage. Pool sizes come from the binding types; a type with
    // no bindings gets no pool entry, since a zero-size pool entry is invalid.
    uint32_t storageBuffers = 0;
    uint32_t storageImages = 0;
    for (const VkDescriptorSetLayoutBinding & binding : bindings)
    {
        if (binding.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
        {
            storageBuffers += binding.descriptorCount;
        }
        else if (binding.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE)
        {
            storageImages += binding.descriptorCount;
        }
    }
    std::vector<VkDescriptorPoolSize> poolSizes;
    if (storageBuffers > 0)
    {
        poolSizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, storageBuffers});
    }
    if (storageImages > 0)
    {
        poolSizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, storageImages});
    }
    stage.descriptors.createPool(1, poolSizes);
    stage.set = stage.descriptors.allocate(stage.layout);

    VkShaderModule module = loadKernel(compiler, m_shaderDir, kernelName);
    VkPipelineLayout layout = stage.pipeline.createPipelineLayout(
        dev, stage.layout, pushConstantSize, VK_SHADER_STAGE_COMPUTE_BIT);
    stage.pipeline.createComputePipeline(dev, module, layout);
    compiler.destroyShaderModule(module);

    stage.built = true;
    return stage;
}

void ForwardPass::destroyOptionalStage(OptionalStage & stage)
{
    // Only the pipeline is destroyed here. The layout, pool and set belong to
    // stage.descriptors and go away with it.
    if (stage.built)
    {
        stage.pipeline.destroy(m_device.handle());
        stage.built = false;
    }
}

void ForwardPass::computeRayDepthForm(const CanonicalGaussianBufferViews & views,
                                      const CameraUniforms & camera)
{
    VkDevice dev = m_device.handle();

    // Standalone: no timestamps. Cleared up front so a throw below leaves
    // rayDepthForm() refusing rather than returning a half-written form.
    m_stampThisRun = false;
    m_rayFormComputed = false;

    VK_VALIDATE(views.gaussianCount > 0,
                "ForwardPass::computeRayDepthForm: gaussianCount must be > 0");
    VK_VALIDATE(camera.gaussianCount == views.gaussianCount,
                "ForwardPass::computeRayDepthForm: camera.gaussianCount must equal "
                "views.gaussianCount");

    ensureRayForm(views.gaussianCount);
    // Views can change every call (a different loaded cloud), so always rebind.
    bindRayFormDescriptors(views);

    VK_CHECK(vkResetCommandPool(dev, m_commandPool, 0),
             "Failed to reset ForwardPass command pool");
    submitAndWait([&](CommandBuffer & cmd) { recordRayFormDispatch(cmd, camera); },
                  "ray-depth form");

    m_rayFormComputed = true;
}

void ForwardPass::ensureRayForm(uint32_t gaussianCount)
{
    // Bindings 0..3 match shaders/ray_depth_form.slang's declaration order.
    ensureOptionalStage(m_rayFormStage, "ray_depth_form",
                        {storageBufferBinding(0), storageBufferBinding(1), storageBufferBinding(2),
                         storageBufferBinding(3)},
                        sizeof(CameraUniforms));
    ensureCapacity(m_device, m_rayForm, static_cast<VkDeviceSize>(gaussianCount) * kRayFormBytes);
}

void ForwardPass::bindRayFormDescriptors(const CanonicalGaussianBufferViews & views)
{
    const auto bindParam = [this](uint32_t binding, const BufferView & view)
    {
        m_rayFormStage.descriptors.bindBuffer(m_rayFormStage.set, binding, view.buffer, view.offset,
                                              view.size, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    };
    bindParam(0, views.means);
    bindParam(1, views.scales);
    bindParam(2, views.rotations);
    m_rayFormStage.descriptors.bindBuffer(m_rayFormStage.set, 3, m_rayForm->handle(), 0,
                                          m_rayForm->size(), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
}

void ForwardPass::recordRayFormDispatch(CommandBuffer & cmd, const CameraUniforms & camera)
{
    // camera.gaussianCount equals the views' count (checked by the callers).
    cmd.pushConstants(m_rayFormStage.pipeline.getLayout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      sizeof(CameraUniforms), &camera);
    cmd.dispatchCompute(m_rayFormStage.pipeline.getHandle(), m_rayFormStage.pipeline.getLayout(),
                        m_rayFormStage.set, dispatchGroups(camera.gaussianCount, kWorkgroupSize), 1,
                        1);
}

void ForwardPass::ensureSortModeResources(uint32_t gaussianCount)
{
    const SortMode mode = m_runSortMode;
    if (mode == SortMode::Default)
    {
        return;
    }

    if (mode == SortMode::Stable)
    {
        // The sticky keys are one uint per Gaussian. A new buffer, or a new
        // Gaussian count, starts them over (their history is per Gaussian).
        if (ensureCapacity(m_device, m_sticky,
                           static_cast<VkDeviceSize>(gaussianCount) * sizeof(uint32_t)))
        {
            m_stickyReset = true;
        }
        if (gaussianCount != m_stickyGaussianCount)
        {
            m_stickyGaussianCount = gaussianCount;
            m_stickyReset = true;
        }

        // Bindings match shaders/sort_stable_update.slang and sort_stable_keys.slang.
        ensureOptionalStage(m_stableUpdateStage, "sort_stable_update",
                            {storageBufferBinding(0), storageBufferBinding(1)},
                            sizeof(StableParams));
        ensureOptionalStage(m_stableKeysStage, "sort_stable_keys",
                            {storageBufferBinding(0), storageBufferBinding(1),
                             storageBufferBinding(2), storageBufferBinding(3)},
                            0);
    }

    if (mode == SortMode::TileRayDepth || mode == SortMode::Hybrid || mode == SortMode::PixelWindow
        || mode == SortMode::Stochastic)
    {
        ensureRayForm(gaussianCount);
    }

    if (mode == SortMode::Stochastic)
    {
        // The blue-noise ranks are a fixed table, uploaded once. Buffer::upload
        // goes through immediateSubmit, which is legal here: run() records
        // nothing before this call. The params buffer is written by
        // vkCmdUpdateBuffer while recording, which needs TRANSFER_DST.
        if (!m_stochNoise)
        {
            const std::vector<uint16_t> ranks = generateBlueNoiseRanks();
            if (ranks.size() != kStochNoiseCount)
            {
                throw std::logic_error("ForwardPass: blue-noise table has the wrong size");
            }
            const std::vector<uint32_t> widened(ranks.begin(), ranks.end());
            const VkBufferUsageFlags uploadUsage =
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            m_stochNoise = std::make_unique<Buffer>(m_device, kStochNoiseBytes, uploadUsage);
            m_stochNoise->upload(widened.data(), kStochNoiseBytes);
        }
        // The history is sized to the framebuffer. resize() drops it and sets
        // m_stochReset, so a recreated history always starts a fresh sequence.
        if (!m_stochHistory)
        {
            m_stochHistory = std::make_unique<Buffer>(
                m_device, static_cast<VkDeviceSize>(m_width) * m_height * sizeof(glm::vec4));
        }
        if (!m_stochParams)
        {
            const VkBufferUsageFlags updateUsage =
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            m_stochParams = std::make_unique<Buffer>(m_device, kStochParamsBytes, updateUsage);
        }

        // Bindings match shaders/tile_rasterize_stochastic.slang: five buffers,
        // then the color and depth images. The S = 4 and S = 8 variants differ only
        // in their -DSABERCAT_ST_SAMPLES define.
        const bool eightSamples = m_sort.stochasticSamples == 8;
        const char * rasterKernel =
            eightSamples ? "tile_rasterize_stochastic_s8" : "tile_rasterize_stochastic_s4";
        ensureOptionalStage(m_stochRasterStage[eightSamples ? 1 : 0], rasterKernel,
                            {storageBufferBinding(0), storageBufferBinding(1),
                             storageBufferBinding(2), storageBufferBinding(3),
                             storageBufferBinding(4), storageBufferBinding(5),
                             storageImageBinding(6), storageImageBinding(7)},
                            sizeof(CameraUniforms));
        // Bindings match shaders/stochastic_resolve.slang: the color image, then the history.
        ensureOptionalStage(m_stochResolveStage, "stochastic_resolve",
                            {storageImageBinding(0), storageBufferBinding(1)},
                            sizeof(ResolveParams));
    }

    if (mode == SortMode::TileRayDepth || mode == SortMode::Hybrid)
    {
        // Bindings match shaders/sort_tile_ray_keys.slang's declaration order.
        ensureOptionalStage(m_tileRayKeysStage, "sort_tile_ray_keys",
                            {storageBufferBinding(0), storageBufferBinding(1),
                             storageBufferBinding(2), storageBufferBinding(3),
                             storageBufferBinding(4), storageBufferBinding(5)},
                            sizeof(CameraUniforms));
    }

    if (mode == SortMode::PixelWindow || mode == SortMode::Hybrid)
    {
        // Bindings match shaders/tile_rasterize_window.slang: four buffers, then
        // the color and depth images.
        ensureOptionalStage(m_windowStage[windowStageIndex(m_runWindow)],
                            windowKernelName(m_runWindow),
                            {storageBufferBinding(0), storageBufferBinding(1),
                             storageBufferBinding(2), storageBufferBinding(3),
                             storageImageBinding(4), storageImageBinding(5)},
                            sizeof(CameraUniforms));
    }
}

void ForwardPass::bindSortModeDescriptors()
{
    const SortMode mode = m_runSortMode;
    if (mode == SortMode::Default)
    {
        return;
    }

    const auto bindStorage = [](DescriptorSet & set, VkDescriptorSet target, uint32_t binding,
                                Buffer & buffer)
    {
        set.bindBuffer(target, binding, buffer.handle(), 0, buffer.size(),
                       VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    };

    if (mode == SortMode::Stable)
    {
        bindStorage(m_stableUpdateStage.descriptors, m_stableUpdateStage.set, 0, *m_projected);
        bindStorage(m_stableUpdateStage.descriptors, m_stableUpdateStage.set, 1, *m_sticky);

        bindStorage(m_stableKeysStage.descriptors, m_stableKeysStage.set, 0, *m_instanceDispatch);
        bindStorage(m_stableKeysStage.descriptors, m_stableKeysStage.set, 1, *m_sortA.values);
        bindStorage(m_stableKeysStage.descriptors, m_stableKeysStage.set, 2, *m_sticky);
        bindStorage(m_stableKeysStage.descriptors, m_stableKeysStage.set, 3, *m_sortA.keysDepth);
    }

    if (mode == SortMode::TileRayDepth || mode == SortMode::Hybrid || mode == SortMode::PixelWindow
        || mode == SortMode::Stochastic)
    {
        bindRayFormDescriptors(m_runViews);
    }

    if (mode == SortMode::Stochastic)
    {
        // Only the selected sample-count variant is bound: the other one may not
        // be built, and a null set cannot be written.
        OptionalStage & raster = m_stochRasterStage[m_sort.stochasticSamples == 8 ? 1 : 0];
        bindStorage(raster.descriptors, raster.set, 0, *m_projected);
        bindStorage(raster.descriptors, raster.set, 1, *m_sortA.values);
        bindStorage(raster.descriptors, raster.set, 2, *m_tileRanges);
        bindStorage(raster.descriptors, raster.set, 3, *m_rayForm);
        bindStorage(raster.descriptors, raster.set, 4, *m_stochNoise);
        bindStorage(raster.descriptors, raster.set, 5, *m_stochParams);
        raster.descriptors.bindImage(raster.set, 6, m_outputImage->getImageView(), VK_NULL_HANDLE,
                                     VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        raster.descriptors.bindImage(raster.set, 7, m_depthImage->getImageView(), VK_NULL_HANDLE,
                                     VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);

        bindStorage(m_stochResolveStage.descriptors, m_stochResolveStage.set, 1, *m_stochHistory);
        m_stochResolveStage.descriptors.bindImage(m_stochResolveStage.set, 0,
                                                  m_outputImage->getImageView(), VK_NULL_HANDLE,
                                                  VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    }

    if (mode == SortMode::TileRayDepth || mode == SortMode::Hybrid)
    {
        DescriptorSet & descriptors = m_tileRayKeysStage.descriptors;
        const VkDescriptorSet set = m_tileRayKeysStage.set;
        bindStorage(descriptors, set, 0, *m_instanceDispatch);
        bindStorage(descriptors, set, 1, *m_projected);
        bindStorage(descriptors, set, 2, *m_rayForm);
        bindStorage(descriptors, set, 3, *m_sortA.keysTile);
        bindStorage(descriptors, set, 4, *m_sortA.values);
        bindStorage(descriptors, set, 5, *m_sortA.keysDepth);
    }

    if (mode == SortMode::PixelWindow || mode == SortMode::Hybrid)
    {
        OptionalStage & stage = m_windowStage[windowStageIndex(m_runWindow)];
        bindStorage(stage.descriptors, stage.set, 0, *m_projected);
        bindStorage(stage.descriptors, stage.set, 1, *m_sortA.values);
        bindStorage(stage.descriptors, stage.set, 2, *m_tileRanges);
        bindStorage(stage.descriptors, stage.set, 3, *m_rayForm);
        stage.descriptors.bindImage(stage.set, 4, m_outputImage->getImageView(), VK_NULL_HANDLE,
                                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
        stage.descriptors.bindImage(stage.set, 5, m_depthImage->getImageView(), VK_NULL_HANDLE,
                                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    }
}

const Buffer & ForwardPass::rayDepthForm() const
{
    if (!m_rayFormComputed)
    {
        throw std::runtime_error("ForwardPass::rayDepthForm: computeRayDepthForm() has not run");
    }
    return *m_rayForm;
}

void ForwardPass::setHighlight(const BufferView & maskWords, const float tint[4])
{
    VK_VALIDATE(maskWords.valid(),
                "ForwardPass::setHighlight: the mask view needs a buffer and a stride");
    m_highlightMask = maskWords;
    std::copy(tint, tint + 4, m_highlightTint);
    m_highlightEnabled = true;
}

void ForwardPass::clearHighlight()
{
    m_highlightEnabled = false;
    m_highlightMask = BufferView{};
}

void ForwardPass::ensureHighlightPipeline()
{
    if (m_highlightPipeline.getHandle() != VK_NULL_HANDLE)
    {
        return;
    }
    VkDevice dev = m_device.handle();
    ShaderCompiler compiler(dev);
    // Loaded first: a missing shader throws before any descriptor state is made.
    VkShaderModule module = loadKernel(compiler, m_shaderDir, "highlight_tint");
    try
    {
        // Made once: a retry after a failed pipeline build reuses the set.
        if (m_highlightSet == VK_NULL_HANDLE)
        {
            m_highlightLayout = m_highlightDescriptors.createLayout(
                {storageBufferBinding(0), storageBufferBinding(1)});
            m_highlightDescriptors.createPool(1, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2}});
            m_highlightSet = m_highlightDescriptors.allocate(m_highlightLayout);
        }
        VkPipelineLayout layout = m_highlightPipeline.createPipelineLayout(
            dev, m_highlightLayout, sizeof(HighlightPush), VK_SHADER_STAGE_COMPUTE_BIT);
        m_highlightPipeline.createComputePipeline(dev, module, layout);
    }
    catch (...)
    {
        m_highlightPipeline.destroy(dev);
        compiler.destroyShaderModule(module);
        throw;
    }
    compiler.destroyShaderModule(module);
}

void ForwardPass::run(const CanonicalGaussianBufferViews & views,
                      const CameraUniforms & callerCamera, ForwardStage stopAfter,
                      ForwardMode mode)
{
    VkDevice dev = m_device.handle();

    // Checked first, before any run state is touched: a refused mode must
    // leave lastRun() and lastStageTimings() as the previous run left them.
    // Differentiable always renders in the default order (BackwardPass
    // assumes it), so the sort mode never reaches a gradient run.
    const SortMode effectiveMode =
        mode == ForwardMode::Differentiable ? SortMode::Default : m_sort.mode;
    requireSortModeAvailable(effectiveMode);
    m_runSortMode = effectiveMode;
    m_runWindow = effectiveMode == SortMode::Hybrid ? 8u : m_sort.pixelWindow;

    // Timings are per run: cleared here, filled only by a run that stamps.
    m_lastTimings = ForwardStageTimings{};
    m_stampThisRun = m_timingEnabled && m_timingPool != VK_NULL_HANDLE
                     && m_timestampValidBits != 0 && stopAfter == ForwardStage::Rasterize;

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

    // Stochastic's noise frame and history weight, from the camera as pushed
    // (flags included). Only a run that reaches the rasterizer advances the
    // history, so only that run computes them. They go into m_runStochastic and
    // are committed at the end, after both submissions have succeeded.
    const bool stochasticRun =
        effectiveMode == SortMode::Stochastic && stopAfter == ForwardStage::Rasterize;
    if (stochasticRun)
    {
        // A new gaussianCount means the history is for a different cloud.
        const bool reset = m_stochReset || views.gaussianCount != m_stochGaussianCount;
        // "changed" is any camera difference, and every reset counts as one.
        const bool cameraMoved =
            std::memcmp(&camera, &m_stochPrevCamera, sizeof(CameraUniforms)) != 0;
        const bool changed = reset || !m_stochPrevValid || cameraMoved;
        StochasticStep step{};
        step.frame = reset ? 0u : m_stochFrame;
        if (reset || (changed && m_sort.temporalMode == 0))
        {
            // Cut: the raw frame replaces the history.
            step.weight = 1.0f;
            step.accumulated = 1;
        }
        else if (changed)
        {
            // EMA mode, camera moved: the new frame enters at temporalEmaWeight. The
            // static frames after it take the floored-mean branch below.
            step.weight = m_sort.temporalEmaWeight;
            step.accumulated = 1;
        }
        else
        {
            // Static camera: running mean of the frames since the last cut, capped
            // at temporalFrames so the history keeps adapting. In EMA mode the mean
            // is floored at temporalEmaWeight: it is a plain mean until 1/k falls to
            // that weight, and an exponential average after that.
            const uint32_t k = std::min(m_stochAccumulated + 1, m_sort.temporalFrames);
            const float mean = 1.0f / static_cast<float>(k);
            step.weight = m_sort.temporalMode == 1 ? std::max(mean, m_sort.temporalEmaWeight) : mean;
            step.accumulated = k;
        }
        m_runStochastic = step;
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
    m_runViews = views;

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

    // Sort-variant stages and their buffers, then their descriptors. Bound
    // here, before the submission records anything, and again after a capacity
    // growth below.
    ensureSortModeResources(m_gaussianCount);
    bindSortModeDescriptors();

    // The selection highlight: a Viewer run tints the projected colors right after
    // projection, when enabled. Its descriptors are written here, before the
    // submission, and the commands below are recorded only when this is true, so a
    // run without a highlight records exactly the default sequence.
    const bool highlightActive = mode == ForwardMode::Viewer && m_highlightEnabled;
    if (highlightActive)
    {
        ensureHighlightPipeline();
        const VkDeviceSize maskBytesNeeded =
            static_cast<VkDeviceSize>((static_cast<uint64_t>(m_gaussianCount) + 31u) / 32u)
            * sizeof(uint32_t);
        VK_VALIDATE(m_highlightMask.buffer != VK_NULL_HANDLE
                        && m_highlightMask.size >= maskBytesNeeded,
                    "ForwardPass::run: the highlight mask holds fewer than ceil(gaussianCount / 32) "
                    "words");
        m_highlightDescriptors.bindBuffer(m_highlightSet, 0, m_projected->handle(), 0,
                                          m_projected->size(), VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        m_highlightDescriptors.bindBuffer(m_highlightSet, 1, m_highlightMask.buffer,
                                          m_highlightMask.offset, m_highlightMask.size,
                                          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    }

    VK_CHECK(vkResetCommandPool(dev, m_commandPool, 0), "Failed to reset ForwardPass command pool");

    // -- One submission: project, the tile-count scan, then every instance-
    // sized stage, all sized on the GPU (see the class comment). The
    // instance total rides back to the host in the same submission, so
    // lastInstanceCount() is current without an extra wait. --
    submitAndWait(
        [&](CommandBuffer & cmd)
        {
            VkCommandBuffer handle = cmd.getHandle();

            // Reset before the first stamp: the pool then holds only this run's
            // values. Nothing is recorded here when timing is off.
            if (m_stampThisRun)
            {
                vkCmdResetQueryPool(handle, m_timingPool, 0, kTimestampCount);
            }

            // Stable's sticky keys start over when asked. This is before the
            // first stamp, so the refill is charged to no stage.
            if (m_runSortMode == SortMode::Stable && m_stickyReset)
            {
                vkCmdFillBuffer(handle, m_sticky->handle(), 0,
                                static_cast<VkDeviceSize>(m_gaussianCount) * sizeof(uint32_t),
                                kStickySentinel);
                cmd.bufferBarrier(m_sticky->handle(), VK_ACCESS_TRANSFER_WRITE_BIT,
                                  VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
                m_stickyReset = false;
            }
            stamp(handle, kStampStart);

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
            stamp(handle, kStampProject);

            if (highlightActive)
            {
                // Tints before the scan and the rasterizer read the colors. The scan
                // reads only tileCount, so this placement holds for every stopAfter.
                const HighlightPush highlight{m_gaussianCount, m_highlightTint[0],
                                              m_highlightTint[1], m_highlightTint[2],
                                              m_highlightTint[3]};
                cmd.pushConstants(m_highlightPipeline.getLayout(), VK_SHADER_STAGE_COMPUTE_BIT,
                                  sizeof(highlight), &highlight);
                cmd.dispatchCompute(m_highlightPipeline.getHandle(),
                                    m_highlightPipeline.getLayout(), m_highlightSet,
                                    dispatchGroups(m_gaussianCount, kWorkgroupSize), 1, 1);
                computeBarrier(handle);
            }

            if (stopAfter != ForwardStage::Project)
            {
                // Exclusive scan of projected[].tileCount -> instanceOffsets.
                ScanParams scan{m_gaussianCount, kScanCountFromParams, kProjectedStrideUints,
                                kTileCountOffsetUints, 0};
                recordScan(cmd, m_prefixSumPipeline, m_prefixSumSet, scan, VK_NULL_HANDLE,
                           dispatchGroups(m_gaussianCount, kScanBlock));
                stamp(handle, kStampScan);

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

    const bool instanceStagesRan =
        stopAfter != ForwardStage::Project && stopAfter != ForwardStage::PrefixSum;
    const bool overflowed = instanceStagesRan && m_lastInstanceCount > m_instanceCapacity;

    // Stage times are read only from a run that rendered in full. An overflowed
    // run's stamps time a frame that drew nothing past the scan, so they are dropped
    // and lastStageTimings() stays invalid.
    if (m_stampThisRun && !overflowed)
    {
        readStageTimings();
    }

    // -- Capacity overflow: instance_dispatch zeroed the count, so the GPU
    // rendered nothing past the scan. Grow (with headroom, so a camera
    // drifting toward denser views doesn't overflow again next frame) and
    // re-record only the instance stages -- projected/instanceOffsets are
    // still valid, and forward_project isn't re-run, so the density
    // feedback it read is unaffected. Once per growth, not per frame. --
    if (overflowed)
    {
        uint64_t withHeadroom = static_cast<uint64_t>(m_lastInstanceCount)
                                + m_lastInstanceCount / kInstanceHeadroomDivisor;
        ensureInstanceCapacity(
            static_cast<uint32_t>(std::min<uint64_t>(withHeadroom, UINT32_MAX)));
        // The sort buffers may have been reallocated; the variant stages point at them.
        bindSortModeDescriptors();
        // The first submission already ran the Stochastic resolve on this frame's
        // empty raster and wrote the history, so the re-run would blend the frame in
        // a second time. A weight of 1 takes the raw frame outright, so the repeat
        // gives the same history, and the accumulation restarts at this frame.
        if (stochasticRun)
        {
            m_runStochastic.weight = 1.0f;
            m_runStochastic.accumulated = 1;
        }
        // The re-run is not stamped. The pool is reset only by the main
        // submission, and this frame's timings are already dropped above.
        m_stampThisRun = false;
        submitAndWait([&](CommandBuffer & cmd) { recordInstanceStages(cmd, camera, stopAfter); },
                      "ForwardPass capacity re-run");
    }

    m_lastRun.completed = stopAfter == ForwardStage::Rasterize;

    // Every submission of this run succeeded, so the Stochastic history and noise
    // sequence now include this frame. A run that threw above never gets here, and
    // its state is unchanged.
    if (stochasticRun)
    {
        m_stochReset = false;
        m_stochPrevValid = true;
        m_stochPrevCamera = camera;
        m_stochGaussianCount = views.gaussianCount;
        m_stochFrame = m_runStochastic.frame + 1;
        m_stochAccumulated = m_runStochastic.accumulated;
        m_stochLastFrame = m_runStochastic.frame;
        m_stochLastWeight = m_runStochastic.weight;
    }
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
    stamp(handle, kStampBinning);

    // Sort-variant key stages. They rewrite keysDepth (or, for the ray-depth
    // form, read it from the instance list), so each ends in a barrier and the
    // radix sort below sees the final keys. The default path records none of
    // them and sorts tile_binning's keys as they are.
    if (m_runSortMode == SortMode::Stable)
    {
        StableParams stableParams{m_gaussianCount, static_cast<float>(m_sort.stableBucketsPerOctave),
                                  m_sort.stableHysteresis, 0};
        cmd.pushConstants(m_stableUpdateStage.pipeline.getLayout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          sizeof(stableParams), &stableParams);
        cmd.dispatchCompute(m_stableUpdateStage.pipeline.getHandle(),
                            m_stableUpdateStage.pipeline.getLayout(), m_stableUpdateStage.set,
                            dispatchGroups(m_gaussianCount, kWorkgroupSize), 1, 1);
        computeBarrier(handle);

        cmd.dispatchComputeIndirect(m_stableKeysStage.pipeline.getHandle(),
                                    m_stableKeysStage.pipeline.getLayout(), m_stableKeysStage.set,
                                    indirectArgs, instanceGroupsOffset);
        computeBarrier(handle);
    }
    else if (m_runSortMode == SortMode::TileRayDepth || m_runSortMode == SortMode::Hybrid)
    {
        recordRayFormDispatch(cmd, camera);
        computeBarrier(handle);
        m_rayFormComputed = true;

        cmd.pushConstants(m_tileRayKeysStage.pipeline.getLayout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          sizeof(CameraUniforms), &camera);
        cmd.dispatchComputeIndirect(m_tileRayKeysStage.pipeline.getHandle(),
                                    m_tileRayKeysStage.pipeline.getLayout(),
                                    m_tileRayKeysStage.set, indirectArgs, instanceGroupsOffset);
        computeBarrier(handle);
    }
    else if (m_runSortMode == SortMode::PixelWindow || m_runSortMode == SortMode::Stochastic)
    {
        // The window and stochastic rasterizers read the form at the end, not the
        // keys here. Stochastic's coverage does not depend on the order, so it
        // needs no key rewrite either.
        recordRayFormDispatch(cmd, camera);
        computeBarrier(handle);
        m_rayFormComputed = true;
    }
    stamp(handle, kStampSortKeys);

    // 8 LSD radix passes: 4 over the depth word (least significant), then 4
    // over the tile word (most significant) -- must finish on the most-
    // significant word for the combined order to be stable. Each pass:
    // histogram -> digit-offset scan (prefix_sum's three phases) -> scatter,
    // the per-instance steps dispatched indirectly.
    // Stochastic starts at pass 4: its coverage is order independent, so the
    // depth word is not needed, and the tile-word passes alone group the list by
    // tile (stably, so each tile keeps the binning's Gaussian-index order). Four
    // passes is an even count, so the result still lands in set A.
    const uint32_t firstPass = m_runSortMode == SortMode::Stochastic ? kRadixPasses : 0;
    ScanParams counterScan{0, kCounterCountSlot, 1, 0, 0};
    for (uint32_t pass = firstPass; pass < 2 * kRadixPasses; ++pass)
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
    stamp(handle, kStampSort);
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
    stamp(handle, kStampRanges);
    if (stopAfter == ForwardStage::TileRanges)
    {
        return;
    }
    cmd.bufferBarrier(m_tileRanges->handle(), VK_ACCESS_SHADER_WRITE_BIT,
                      VK_ACCESS_SHADER_READ_BIT);

    // The window rasterizer (PixelWindow, Hybrid) and the stochastic one take
    // tile_rasterize's place and do not write batchesFetched, so the clear below
    // is skipped for them.
    const bool windowRaster =
        m_runSortMode == SortMode::PixelWindow || m_runSortMode == SortMode::Hybrid;
    const bool stochasticRaster = m_runSortMode == SortMode::Stochastic;

    if (!windowRaster && !stochasticRaster)
    {
        // Verification-only: zero this frame's per-tile batch counters
        // before tile_rasterize accumulates into them.
        vkCmdFillBuffer(handle, m_batchesFetched->handle(), 0, m_batchesFetched->size(), 0);
        cmd.bufferBarrier(m_batchesFetched->handle(), VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    }

    m_outputImage->transitionLayout(handle, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    m_depthImage->transitionLayout(handle, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);

    if (stochasticRaster)
    {
        // The params go in as a 16-byte inline update (no staging needed), then the
        // barrier makes them visible to the raster's read. The layout is uint4
        // {asuint(frameOffset), frameIndex, 0, 0}: the float travels as its bit
        // pattern, so the shader reads it with asfloat and no conversion is
        // involved.
        const float frameOffset = stochasticFrameOffset(m_runStochastic.frame);
        std::array<uint32_t, 4> params{0u, m_runStochastic.frame, 0u, 0u};
        std::memcpy(&params[0], &frameOffset, sizeof(frameOffset));
        vkCmdUpdateBuffer(handle, m_stochParams->handle(), 0, kStochParamsBytes, params.data());
        cmd.bufferBarrier(m_stochParams->handle(), VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_ACCESS_SHADER_READ_BIT);

        OptionalStage & raster = m_stochRasterStage[m_sort.stochasticSamples == 8 ? 1 : 0];
        cmd.pushConstants(raster.pipeline.getLayout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          sizeof(CameraUniforms), &camera);
        cmd.dispatchCompute(raster.pipeline.getHandle(), raster.pipeline.getLayout(), raster.set,
                            m_tilesX, m_tilesY, 1);
        computeBarrier(handle);
        stamp(handle, kStampRasterize);

        // The resolve reads the raw frame the raster just wrote, and folds it into
        // the history with this run's weight.
        const ResolveParams resolve{m_width, m_height, m_runStochastic.weight, 0};
        cmd.pushConstants(m_stochResolveStage.pipeline.getLayout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          sizeof(resolve), &resolve);
        cmd.dispatchCompute(m_stochResolveStage.pipeline.getHandle(),
                            m_stochResolveStage.pipeline.getLayout(), m_stochResolveStage.set,
                            dispatchGroups(m_width, kTileSize), dispatchGroups(m_height, kTileSize),
                            1);
        stamp(handle, kStampResolve);
        return;
    }

    if (windowRaster)
    {
        OptionalStage & stage = m_windowStage[windowStageIndex(m_runWindow)];
        cmd.pushConstants(stage.pipeline.getLayout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          sizeof(CameraUniforms), &camera);
        cmd.dispatchCompute(stage.pipeline.getHandle(), stage.pipeline.getLayout(), stage.set,
                            m_tilesX, m_tilesY, 1);
    }
    else
    {
        cmd.pushConstants(m_rasterizePipeline.getLayout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          sizeof(CameraUniforms), &camera);
        cmd.dispatchCompute(m_rasterizePipeline.getHandle(), m_rasterizePipeline.getLayout(),
                            m_rasterizeSet, m_tilesX, m_tilesY, 1);
    }
    stamp(handle, kStampRasterize);
    // Only Stochastic has a resolve dispatch, so for the other modes its stamp
    // sits right after the rasterize one and its stage time reads as zero.
    stamp(handle, kStampResolve);
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
