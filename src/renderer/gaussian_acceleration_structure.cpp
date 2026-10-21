// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/gaussian_acceleration_structure.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

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

// VkAabbPositionsKHR: six floats, no padding (a multiple of the 8-byte
// stride Vulkan requires). gaussian_aabbs.slang writes exactly this.
constexpr VkDeviceSize kAabbBytes = sizeof(VkAabbPositionsKHR);
static_assert(sizeof(VkAabbPositionsKHR) == 6 * sizeof(float), "AABB layout mirrored by the kernel");

struct AabbParams
{
    uint32_t gaussianCount;
    float minAlpha;
    float invMinAlpha;
    float pad0;
};
static_assert(sizeof(AabbParams) == 16, "AabbParams must match gaussian_aabbs.slang");

// Procedural geometry is never opaque here (the ray must see every candidate)
// and must report each primitive ONCE: gaussianTransmittance multiplies T by
// (1 - alpha) per candidate and never commits, so a duplicate candidate --
// which Vulkan otherwise permits for non-opaque geometry -- would apply a
// Gaussian's alpha twice.
constexpr VkGeometryFlagsKHR kGeometryFlags = VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR;

// BLAS: refit()-able, tuned for tracing (shadow rays are traced per pixel per
// light; a build happens once per parameter change). Update and compaction
// don't mix usefully with a structure refit every optimizer step, so no
// ALLOW_COMPACTION.
constexpr VkBuildAccelerationStructureFlagsKHR kBlasFlags =
    VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR
    | VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
// TLAS: one instance, fully rebuilt every time (a refit of a 1-instance
// TLAS saves nothing), so no ALLOW_UPDATE.
constexpr VkBuildAccelerationStructureFlagsKHR kTlasFlags =
    VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;

void requireArgument(bool condition, const std::string & message)
{
    if (!condition)
    {
        throw std::invalid_argument("GaussianAccelerationStructure: " + message);
    }
}

void requireStream(const char * name, const BufferView & view, uint32_t stride, uint32_t count)
{
    requireArgument(view.buffer != VK_NULL_HANDLE, std::string(name) + " view has no buffer");
    requireArgument(view.stride == stride, std::string(name) + " view stride "
                                               + std::to_string(view.stride) + " is not the canonical "
                                               + std::to_string(stride));
    requireArgument(view.elementCount() >= count,
                    std::string(name) + " view holds " + std::to_string(view.elementCount())
                        + " Gaussians, fewer than gaussianCount " + std::to_string(count));
}

void validateViews(const CanonicalGaussianBufferViews & views)
{
    requireArgument(views.gaussianCount >= 1, "gaussianCount must be >= 1");
    requireStream("means", views.means, kMeansStride, views.gaussianCount);
    requireStream("scales", views.scales, kScalesStride, views.gaussianCount);
    requireStream("rotations", views.rotations, kRotationsStride, views.gaussianCount);
    requireStream("opacity", views.opacity, kOpacityStride, views.gaussianCount);
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

VkDeviceAddress bufferAddress(VkDevice device, VkBuffer buffer)
{
    VkBufferDeviceAddressInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    info.buffer = buffer;
    return vkGetBufferDeviceAddress(device, &info);
}

VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment)
{
    return alignment > 1 ? (value + alignment - 1) / alignment * alignment : value;
}

void memoryBarrier(VkCommandBuffer cmd, VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
                   VkPipelineStageFlags dstStage, VkAccessFlags dstAccess)
{
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

// The BLAS geometry: one AABB per Gaussian, read from `aabbAddress`.
VkAccelerationStructureGeometryKHR aabbGeometry(VkDeviceAddress aabbAddress)
{
    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_AABBS_KHR;
    geometry.flags = kGeometryFlags;
    geometry.geometry.aabbs.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_AABBS_DATA_KHR;
    geometry.geometry.aabbs.data.deviceAddress = aabbAddress;
    geometry.geometry.aabbs.stride = kAabbBytes;
    return geometry;
}

VkAccelerationStructureGeometryKHR instanceGeometry(VkDeviceAddress instanceAddress)
{
    VkAccelerationStructureGeometryKHR geometry{};
    geometry.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    geometry.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geometry.geometry.instances.sType =
        VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geometry.geometry.instances.arrayOfPointers = VK_FALSE;
    geometry.geometry.instances.data.deviceAddress = instanceAddress;
    return geometry;
}

VkAccelerationStructureBuildGeometryInfoKHR buildInfo(VkAccelerationStructureTypeKHR type,
                                                      VkBuildAccelerationStructureFlagsKHR flags,
                                                      const VkAccelerationStructureGeometryKHR * geometry)
{
    VkAccelerationStructureBuildGeometryInfoKHR info{};
    info.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    info.type = type;
    info.flags = flags;
    info.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    info.geometryCount = 1;
    info.pGeometries = geometry;
    return info;
}

} // namespace

GaussianAccelerationStructure::GaussianAccelerationStructure(Device & device,
                                                             const std::string & shaderDir)
    : m_device(device), m_descriptors(device.handle())
{
    VK_VALIDATE(device.features().rayQuery,
                "GaussianAccelerationStructure needs acceleration structures + ray query + buffer "
                "device address (VK_KHR_acceleration_structure, VK_KHR_ray_query), which this "
                "device does not support or did not enable (model B: create it with "
                "DeviceRequirements::Options::rayTracing and declare EnabledDeviceFeatures::rayQuery); "
                "rasterization is unaffected");

    VkDevice dev = device.handle();
    m_createAS = reinterpret_cast<PFN_vkCreateAccelerationStructureKHR>(
        vkGetDeviceProcAddr(dev, "vkCreateAccelerationStructureKHR"));
    m_destroyAS = reinterpret_cast<PFN_vkDestroyAccelerationStructureKHR>(
        vkGetDeviceProcAddr(dev, "vkDestroyAccelerationStructureKHR"));
    m_getBuildSizes = reinterpret_cast<PFN_vkGetAccelerationStructureBuildSizesKHR>(
        vkGetDeviceProcAddr(dev, "vkGetAccelerationStructureBuildSizesKHR"));
    m_cmdBuildAS = reinterpret_cast<PFN_vkCmdBuildAccelerationStructuresKHR>(
        vkGetDeviceProcAddr(dev, "vkCmdBuildAccelerationStructuresKHR"));
    m_getASAddress = reinterpret_cast<PFN_vkGetAccelerationStructureDeviceAddressKHR>(
        vkGetDeviceProcAddr(dev, "vkGetAccelerationStructureDeviceAddressKHR"));
    VK_VALIDATE(m_createAS && m_destroyAS && m_getBuildSizes && m_cmdBuildAS && m_getASAddress,
                "GaussianAccelerationStructure: VK_KHR_acceleration_structure entry points are "
                "missing although rayQuery is enabled");

    VkPhysicalDeviceAccelerationStructurePropertiesKHR asProps{};
    asProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR;
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &asProps;
    vkGetPhysicalDeviceProperties2(device.physicalDevice(), &props2);
    m_scratchAlignment = asProps.minAccelerationStructureScratchOffsetAlignment;

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = device.submitQueueFamily();
    VK_CHECK(vkCreateCommandPool(dev, &poolInfo, nullptr, &m_commandPool),
             "Failed to create GaussianAccelerationStructure command pool");

    // A throw here (a missing shader file is the common one) skips the
    // destructor, so the pool and any pipeline state are released here.
    try
    {
        createPipeline(shaderDir);
    }
    catch (...)
    {
        release();
        throw;
    }
}

GaussianAccelerationStructure::~GaussianAccelerationStructure()
{
    release();
}

void GaussianAccelerationStructure::release()
{
    VkDevice dev = m_device.handle();
    m_pipeline.destroy(dev);
    if (m_tlas != VK_NULL_HANDLE)
    {
        m_destroyAS(dev, m_tlas, nullptr);
    }
    if (m_blas != VK_NULL_HANDLE)
    {
        m_destroyAS(dev, m_blas, nullptr);
    }
    if (m_commandPool != VK_NULL_HANDLE)
    {
        vkDestroyCommandPool(dev, m_commandPool, nullptr);
    }
}

void GaussianAccelerationStructure::createPipeline(const std::string & shaderDir)
{
    // gaussian_aabbs: means, scales, rotations, opacities, aabbs (RW); count pushed.
    VkDevice dev = m_device.handle();
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    for (uint32_t i = 0; i < 5; ++i)
    {
        bindings.push_back(storageBufferBinding(i));
    }
    m_layout = m_descriptors.createLayout(bindings);
    m_descriptors.createPool(1, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5}});
    m_set = m_descriptors.allocate(m_layout);

    ShaderCompiler compiler(dev);
    auto spirv = compiler.loadSPIRV(shaderDir + "/gaussian_aabbs.comp.spv");
    VkShaderModule module = compiler.createShaderModule(spirv);
    VkPipelineLayout layout = m_pipeline.createPipelineLayout(dev, m_layout, sizeof(AabbParams),
                                                              VK_SHADER_STAGE_COMPUTE_BIT);
    m_pipeline.createComputePipeline(dev, module, layout);
    compiler.destroyShaderModule(module);
}

void GaussianAccelerationStructure::ensureCapacity(uint32_t count)
{
    VkDevice dev = m_device.handle();

    // AABBs: written by the kernel, read by the build (and by the test).
    if (count > m_aabbCapacity || !m_aabbs)
    {
        m_aabbs = std::make_unique<Buffer>(
            m_device, static_cast<VkDeviceSize>(count) * kAabbBytes,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
        m_aabbCapacity = count;
    }

    // BLAS sizes for this count (the build/update sizes depend only on the
    // count and flags, not on the AABB contents).
    const VkAccelerationStructureGeometryKHR blasGeometry = aabbGeometry(0);
    VkAccelerationStructureBuildGeometryInfoKHR blasInfo =
        buildInfo(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, kBlasFlags, &blasGeometry);
    VkAccelerationStructureBuildSizesInfoKHR blasSizes{};
    blasSizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    m_getBuildSizes(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &blasInfo, &count,
                    &blasSizes);

    if (blasSizes.accelerationStructureSize > m_blasCapacity || m_blas == VK_NULL_HANDLE)
    {
        if (m_blas != VK_NULL_HANDLE)
        {
            m_destroyAS(dev, m_blas, nullptr);
            m_blas = VK_NULL_HANDLE;
        }
        m_blasStorage = std::make_unique<Buffer>(
            m_device, blasSizes.accelerationStructureSize,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        VkAccelerationStructureCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        createInfo.buffer = m_blasStorage->handle();
        createInfo.size = blasSizes.accelerationStructureSize;
        createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        VK_CHECK(m_createAS(dev, &createInfo, nullptr, &m_blas), "Failed to create Gaussian BLAS");
        m_blasCapacity = blasSizes.accelerationStructureSize;

        VkAccelerationStructureDeviceAddressInfoKHR addressInfo{};
        addressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        addressInfo.accelerationStructure = m_blas;
        m_blasAddress = m_getASAddress(dev, &addressInfo);
    }

    // TLAS: one instance, so its size never changes -- created once, which
    // keeps tlas() stable for the object's lifetime.
    if (m_tlas == VK_NULL_HANDLE)
    {
        m_instance = std::make_unique<Buffer>(
            m_device, sizeof(VkAccelerationStructureInstanceKHR),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
                | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);

        const VkAccelerationStructureGeometryKHR tlasGeometry = instanceGeometry(0);
        VkAccelerationStructureBuildGeometryInfoKHR tlasInfo =
            buildInfo(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, kTlasFlags, &tlasGeometry);
        const uint32_t one = 1;
        VkAccelerationStructureBuildSizesInfoKHR tlasSizes{};
        tlasSizes.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        m_getBuildSizes(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &tlasInfo, &one,
                        &tlasSizes);

        m_tlasStorage = std::make_unique<Buffer>(
            m_device, tlasSizes.accelerationStructureSize,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR
                | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
        VkAccelerationStructureCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        createInfo.buffer = m_tlasStorage->handle();
        createInfo.size = tlasSizes.accelerationStructureSize;
        createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        VK_CHECK(m_createAS(dev, &createInfo, nullptr, &m_tlas), "Failed to create Gaussian TLAS");
        m_tlasBuildScratch = tlasSizes.buildScratchSize;

        VkAccelerationStructureDeviceAddressInfoKHR addressInfo{};
        addressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        addressInfo.accelerationStructure = m_tlas;
        m_tlasAddress = m_getASAddress(dev, &addressInfo);
    }

    // One scratch buffer serves the BLAS build/update and then the TLAS
    // build (a barrier separates them). Over-allocated by the alignment so
    // the address can be rounded up to minAccelerationStructureScratchOffsetAlignment.
    const VkDeviceSize scratchNeeded =
        std::max({blasSizes.buildScratchSize, blasSizes.updateScratchSize, m_tlasBuildScratch})
        + m_scratchAlignment;
    if (!m_scratch || m_scratch->size() < scratchNeeded)
    {
        m_scratch = std::make_unique<Buffer>(m_device, scratchNeeded,
                                             VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                                 | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT);
    }
}

void GaussianAccelerationStructure::build(const CanonicalGaussianBufferViews & views, float minAlpha)
{
    requireArgument(minAlpha >= 1.0f / 255.0f && minAlpha < 1.0f,
                    "minAlpha must be in [1/255, 1), got " + std::to_string(minAlpha));
    validateViews(views);
    m_minAlpha = minAlpha; // after the views check: a rejected build changes nothing
    ensureCapacity(views.gaussianCount);
    record(views, false);
    m_gaussianCount = views.gaussianCount;
    m_refitsSinceBuild = 0;
}

void GaussianAccelerationStructure::refit(const CanonicalGaussianBufferViews & views)
{
    validateViews(views);
    requireArgument(built(), "refit() before the first build()");
    requireArgument(views.gaussianCount == m_gaussianCount,
                    "refit() needs the count of the last build (" + std::to_string(m_gaussianCount)
                        + "), got " + std::to_string(views.gaussianCount)
                        + "; call build() when the count changes");
    record(views, true);
    ++m_refitsSinceBuild;
}

void GaussianAccelerationStructure::record(const CanonicalGaussianBufferViews & views, bool update)
{
    VkDevice dev = m_device.handle();
    const uint32_t count = views.gaussianCount;

    // Bound fresh every call (the call waits for completion, so the set is
    // never updated while in flight). Views bind at their own offsets, so
    // borrowed streams packed into one buffer work as-is.
    const auto bindView = [&](uint32_t binding, const BufferView & view)
    {
        m_descriptors.bindBuffer(m_set, binding, view.buffer, view.offset, view.size,
                                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    };
    bindView(0, views.means);
    bindView(1, views.scales);
    bindView(2, views.rotations);
    bindView(3, views.opacity);
    m_descriptors.bindBuffer(m_set, 4, m_aabbs->handle(), 0, m_aabbs->size(),
                             VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

    const VkDeviceAddress aabbAddress = bufferAddress(dev, m_aabbs->handle());
    const VkDeviceAddress scratchAddress =
        alignUp(bufferAddress(dev, m_scratch->handle()), m_scratchAlignment);
    const VkDeviceAddress instanceAddress = bufferAddress(dev, m_instance->handle());

    VkAccelerationStructureInstanceKHR instance{};
    instance.transform.matrix[0][0] = 1.0f;
    instance.transform.matrix[1][1] = 1.0f;
    instance.transform.matrix[2][2] = 1.0f;
    instance.instanceCustomIndex = 0;
    instance.mask = 0xFF;
    instance.instanceShaderBindingTableRecordOffset = 0;
    instance.flags = 0;
    instance.accelerationStructureReference = m_blasAddress;

    VK_CHECK(vkResetCommandPool(dev, m_commandPool, 0),
             "Failed to reset GaussianAccelerationStructure command pool");
    CommandBuffer cmd(dev, m_commandPool);
    cmd.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VkCommandBuffer handle = cmd.getHandle();
    Device::recordGlobalBarrier(handle); // see Device: queue contract

    // 1. AABBs from the current parameters; the instance (its BLAS address
    //    changes when the BLAS grows) is rewritten alongside.
    // The exact default passes the literal 255 so its boxes stay bit-identical.
    const bool exact = m_minAlpha == 1.0f / 255.0f;
    const AabbParams params{count, m_minAlpha, exact ? 255.0f : 1.0f / m_minAlpha, 0.0f};
    cmd.pushConstants(m_pipeline.getLayout(), VK_SHADER_STAGE_COMPUTE_BIT, sizeof(params), &params);
    cmd.dispatchCompute(m_pipeline.getHandle(), m_pipeline.getLayout(), m_set,
                        dispatchGroups(count, kWorkgroupSize), 1, 1);
    vkCmdUpdateBuffer(handle, m_instance->handle(), 0, sizeof(instance), &instance);
    // Build inputs are read with SHADER_READ at the build stage.
    memoryBarrier(handle,
                  VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                  VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                  VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_ACCESS_SHADER_READ_BIT);

    // 2. BLAS: full build, or an in-place update (src == dst) on refit.
    const VkAccelerationStructureGeometryKHR blasGeometry = aabbGeometry(aabbAddress);
    VkAccelerationStructureBuildGeometryInfoKHR blasInfo =
        buildInfo(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, kBlasFlags, &blasGeometry);
    blasInfo.mode = update ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR
                           : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    blasInfo.srcAccelerationStructure = update ? m_blas : VK_NULL_HANDLE;
    blasInfo.dstAccelerationStructure = m_blas;
    blasInfo.scratchData.deviceAddress = scratchAddress;
    VkAccelerationStructureBuildRangeInfoKHR blasRange{};
    blasRange.primitiveCount = count;
    const VkAccelerationStructureBuildRangeInfoKHR * blasRanges = &blasRange;
    m_cmdBuildAS(handle, 1, &blasInfo, &blasRanges);

    // The TLAS build reads the BLAS and reuses the scratch.
    memoryBarrier(handle, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                  VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                  VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                  VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR
                      | VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR);

    // 3. TLAS: always a full build of the one instance -- the BLAS's bounds
    //    moved with a refit too.
    const VkAccelerationStructureGeometryKHR tlasGeometry = instanceGeometry(instanceAddress);
    VkAccelerationStructureBuildGeometryInfoKHR tlasInfo =
        buildInfo(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, kTlasFlags, &tlasGeometry);
    tlasInfo.dstAccelerationStructure = m_tlas;
    tlasInfo.scratchData.deviceAddress = scratchAddress;
    VkAccelerationStructureBuildRangeInfoKHR tlasRange{};
    tlasRange.primitiveCount = 1;
    const VkAccelerationStructureBuildRangeInfoKHR * tlasRanges = &tlasRange;
    m_cmdBuildAS(handle, 1, &tlasInfo, &tlasRanges);

    Device::recordGlobalBarrier(handle); // see Device: queue contract
    cmd.end();

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence;
    VK_CHECK(vkCreateFence(dev, &fenceInfo, nullptr, &fence), "Failed to create fence");
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &handle;
    VK_CHECK(vkQueueSubmit(m_device.submitQueue(), 1, &submitInfo, fence),
             "Failed to submit GaussianAccelerationStructure build");
    VK_CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX),
             "Failed to wait for GaussianAccelerationStructure build");
    vkDestroyFence(dev, fence, nullptr);
}

} // namespace sabercat
