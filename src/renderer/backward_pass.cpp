// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/backward_pass.h"

#include <cstring>
#include <vector>

#include "sabercat/buffer.h"
#include "sabercat/camera.h"
#include "sabercat/command_buffer.h"
#include "sabercat/device.h"
#include "sabercat/forward_pass.h"
#include "sabercat/render_config.h"
#include "sabercat/shader_compiler.h"
#include "sabercat/vk_check.h"

namespace sabercat
{

namespace
{

constexpr uint32_t kProjectedGradientFloats = SABERCAT_PROJECTED_GRADIENT_FLOATS;
constexpr VkDeviceSize kImageGradientTexelBytes = 4 * sizeof(float); // RGBA

VkDescriptorSetLayoutBinding storageBufferBinding(uint32_t binding)
{
    VkDescriptorSetLayoutBinding b{};
    b.binding = binding;
    b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    return b;
}

void bindView(DescriptorSet & set, VkDescriptorSet target, uint32_t binding, const BufferView & view)
{
    set.bindBuffer(target, binding, view.buffer, view.offset, view.size,
                   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
}

void bindWhole(DescriptorSet & set, VkDescriptorSet target, uint32_t binding, const Buffer & buffer)
{
    set.bindBuffer(target, binding, buffer.handle(), 0, buffer.size(),
                   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
}

} // namespace

BackwardPass::BackwardPass(Device & device, const std::string & shaderDir)
    : m_device(device), m_rasterizeDescriptors(device.handle()),
      m_projectDescriptors(device.handle())
{
    VK_VALIDATE(device.features().shaderBufferFloat32AtomicAdd,
                "BackwardPass needs fp32 atomic add on storage buffers "
                "(VK_EXT_shader_atomic_float shaderBufferFloat32AtomicAdd), which this device "
                "does not support or did not enable; forward rendering is unaffected");

    // A throw below (a missing shader file is the common one) skips the
    // destructor, so the raw handles created so far are released here.
    try
    {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = device.submitQueueFamily();
        VK_CHECK(vkCreateCommandPool(device.handle(), &poolInfo, nullptr, &m_commandPool),
                 "Failed to create BackwardPass command pool");

        createPipelines(shaderDir);
    }
    catch (...)
    {
        release();
        throw;
    }
}

BackwardPass::~BackwardPass()
{
    release();
}

void BackwardPass::release()
{
    VkDevice dev = m_device.handle();
    m_rasterizePipeline.destroy(dev);
    m_projectPipeline.destroy(dev);
    destroyUpload(m_imageUpload);
    destroyUpload(m_depthUpload);
    if (m_commandPool != VK_NULL_HANDLE)
    {
        vkDestroyCommandPool(dev, m_commandPool, nullptr);
    }
}

void BackwardPass::createPipelines(const std::string & shaderDir)
{
    VkDevice dev = m_device.handle();
    ShaderCompiler compiler(dev);

    // -- backward_rasterize: projected, sortedValues, tileRanges, pixelState,
    // dLossDImage, projectedGradients(RW), dLossDDepth; camera pushed. --
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        for (uint32_t i = 0; i < 7; ++i)
        {
            bindings.push_back(storageBufferBinding(i));
        }
        m_rasterizeLayout = m_rasterizeDescriptors.createLayout(bindings);
        m_rasterizeDescriptors.createPool(1, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 7}});
        m_rasterizeSet = m_rasterizeDescriptors.allocate(m_rasterizeLayout);

        auto spirv = compiler.loadSPIRV(shaderDir + "/backward_rasterize.comp.spv");
        VkShaderModule module = compiler.createShaderModule(spirv);
        VkPipelineLayout layout = m_rasterizePipeline.createPipelineLayout(
            dev, m_rasterizeLayout, sizeof(CameraUniforms), VK_SHADER_STAGE_COMPUTE_BIT);
        m_rasterizePipeline.createComputePipeline(dev, module, layout);
        compiler.destroyShaderModule(module);
    }

    // -- backward_project: means, scales, rotations, opacities, sh, projected,
    // projectedGradients, then the six gradient outputs (RW); camera pushed. --
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        for (uint32_t i = 0; i < 13; ++i)
        {
            bindings.push_back(storageBufferBinding(i));
        }
        m_projectLayout = m_projectDescriptors.createLayout(bindings);
        m_projectDescriptors.createPool(1, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 13}});
        m_projectSet = m_projectDescriptors.allocate(m_projectLayout);

        auto spirv = compiler.loadSPIRV(shaderDir + "/backward_project.comp.spv");
        VkShaderModule module = compiler.createShaderModule(spirv);
        VkPipelineLayout layout = m_projectPipeline.createPipelineLayout(
            dev, m_projectLayout, sizeof(CameraUniforms), VK_SHADER_STAGE_COMPUTE_BIT);
        m_projectPipeline.createComputePipeline(dev, module, layout);
        compiler.destroyShaderModule(module);
    }
}

void BackwardPass::destroyUpload(HostUpload & upload)
{
    if (upload.staging != VK_NULL_HANDLE)
    {
        vmaDestroyBuffer(m_device.allocator(), upload.staging, upload.stagingAllocation);
        upload.staging = VK_NULL_HANDLE;
    }
}

void BackwardPass::stageUpload(HostUpload & upload, const float * host, VkDeviceSize bytes)
{
    if (upload.capacity < bytes)
    {
        destroyUpload(upload);

        // Persistent and mapped, like ForwardPass's instance-count readback, so
        // a training loop's per-iteration upload is a memcpy, not a staging
        // buffer create/destroy plus Device::immediateSubmit (load-time-only).
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = bytes;
        bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo allocInfo{};
        allocInfo.usage = VMA_MEMORY_USAGE_AUTO;
        allocInfo.flags =
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        VmaAllocationInfo info{};
        VK_CHECK(vmaCreateBuffer(m_device.allocator(), &bufferInfo, &allocInfo, &upload.staging,
                                 &upload.stagingAllocation, &info),
                 "Failed to create gradient staging buffer");
        upload.stagingMapped = info.pMappedData;

        upload.device = std::make_unique<Buffer>(m_device, bytes);
        upload.capacity = bytes;
    }

    upload.bytes = bytes;
    std::memcpy(upload.stagingMapped, host, static_cast<size_t>(bytes));
    VK_CHECK(vmaFlushAllocation(m_device.allocator(), upload.stagingAllocation, 0, bytes),
             "Failed to flush gradient staging buffer");
}

void BackwardPass::recordUploadCopy(CommandBuffer & cmd, const HostUpload & upload)
{
    VkBufferCopy region{};
    region.size = upload.bytes;
    vkCmdCopyBuffer(cmd.getHandle(), upload.staging, upload.device->handle(), 1, &region);
    cmd.bufferBarrier(upload.device->handle(), VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_ACCESS_SHADER_READ_BIT);
}

void BackwardPass::run(const ForwardPass & forward, const float * dLossDImageHost,
                       const CanonicalGaussianGradientViews & gradients)
{
    run(forward, dLossDImageHost, nullptr, gradients);
}

void BackwardPass::run(const ForwardPass & forward, const float * dLossDImageHost,
                       const float * dLossDDepthHost,
                       const CanonicalGaussianGradientViews & gradients)
{
    VK_VALIDATE(dLossDImageHost != nullptr, "BackwardPass::run: null image gradient");
    const ForwardRunRecord & last = forward.lastRun();
    VK_VALIDATE(last.valid, "BackwardPass::run: no forward run to differentiate");
    // Before anything is uploaded: a rejected call touches no buffer.
    VK_VALIDATE(dLossDDepthHost == nullptr || forward.depthLayout() != DepthLayout::None,
                "BackwardPass::run: a depth gradient was given but the forward has no depth "
                "output (DepthLayout::None)");

    const VkDeviceSize pixels =
        static_cast<VkDeviceSize>(last.camera.imageWidth) * last.camera.imageHeight;
    const VkDeviceSize imageBytes = pixels * kImageGradientTexelBytes;
    stageUpload(m_imageUpload, dLossDImageHost, imageBytes);
    const BufferView imageView =
        m_imageUpload.device->view(static_cast<uint32_t>(kImageGradientTexelBytes));

    if (dLossDDepthHost == nullptr)
    {
        record(forward, imageView, BufferView{}, gradients, &m_imageUpload, nullptr);
        return;
    }

    const uint32_t depthFloats = static_cast<uint32_t>(forward.depthLayout());
    const VkDeviceSize depthTexelBytes = depthFloats * sizeof(float);
    stageUpload(m_depthUpload, dLossDDepthHost, pixels * depthTexelBytes);
    record(forward, imageView, m_depthUpload.device->view(static_cast<uint32_t>(depthTexelBytes)),
           gradients, &m_imageUpload, &m_depthUpload);
}

void BackwardPass::run(const ForwardPass & forward, const BufferView & dLossDImage,
                       const CanonicalGaussianGradientViews & gradients)
{
    run(forward, dLossDImage, BufferView{}, gradients);
}

void BackwardPass::run(const ForwardPass & forward, const BufferView & dLossDImage,
                       const BufferView & dLossDDepth,
                       const CanonicalGaussianGradientViews & gradients)
{
    record(forward, dLossDImage, dLossDDepth, gradients, nullptr, nullptr);
}

void BackwardPass::record(const ForwardPass & forward, const BufferView & dLossDImage,
                          const BufferView & dLossDDepth,
                          const CanonicalGaussianGradientViews & gradients,
                          const HostUpload * imageUpload, const HostUpload * depthUpload)
{
    const ForwardRunRecord & last = forward.lastRun();
    VK_VALIDATE(last.valid && last.completed,
                "BackwardPass::run: the last forward run did not complete (stopped early, "
                "threw, or the pass was resized since)");
    VK_VALIDATE(last.mode == ForwardMode::Differentiable,
                "BackwardPass::run: the last forward run was a Viewer run; viewer LOD knobs may "
                "have dropped Gaussians and no backward state was saved -- run forward in "
                "ForwardMode::Differentiable first");

    // A copy: the forward's recorded flags carry its own DEPTH_FLOATS bits, and
    // only this run's depth gradient may set them.
    CameraUniforms camera = last.camera;
    camera.renderFlags &= ~(kRenderFlagDepthFloatsMask << kRenderFlagDepthFloatsShift);
    const CanonicalGaussianBufferViews & params = last.views;
    const uint32_t count = params.gaussianCount;
    const uint32_t depthFloats = static_cast<uint32_t>(forward.depthLayout());
    const bool depthGradient = dLossDDepth.valid();
    const VkDeviceSize imageBytes =
        static_cast<VkDeviceSize>(camera.imageWidth) * camera.imageHeight * kImageGradientTexelBytes;
    const VkDeviceSize depthBytes =
        static_cast<VkDeviceSize>(camera.imageWidth) * camera.imageHeight * depthFloats * sizeof(float);

    VK_VALIDATE(dLossDImage.valid() && dLossDImage.size >= imageBytes,
                "BackwardPass::run: image gradient buffer is smaller than width*height RGBA");
    if (depthGradient)
    {
        VK_VALIDATE(depthFloats != 0,
                    "BackwardPass::run: a depth gradient was given but the forward has no depth "
                    "output (DepthLayout::None)");
        VK_VALIDATE(dLossDDepth.size >= depthBytes,
                    "BackwardPass::run: depth gradient buffer is smaller than "
                    "width*height*n floats");
        camera.renderFlags |= depthFloats << kRenderFlagDepthFloatsShift;
    }
    VK_VALIDATE(gradients.gaussianCount == count,
                "BackwardPass::run: gradient views are sized for a different Gaussian count");
    VK_VALIDATE(gradients.means.valid() && gradients.scales.valid() && gradients.rotations.valid()
                    && gradients.opacity.valid() && gradients.shCoefficients.valid()
                    && gradients.screenMeans.valid(),
                "BackwardPass::run: every gradient view must be bound");
    if (count == 0)
    {
        return;
    }

    // Grow-only, like ForwardPass's count-sized buffers: densification moves
    // the count often, and only [0, count) is cleared and used below.
    const VkDeviceSize projectedGradientBytes =
        static_cast<VkDeviceSize>(count) * kProjectedGradientFloats * sizeof(float);
    if (count > m_gaussianCapacity || !m_projectedGradients)
    {
        m_gaussianCapacity = count;
        m_projectedGradients = std::make_unique<Buffer>(m_device, projectedGradientBytes);
    }

    // Bound fresh every run: run() waits for completion, so no set is ever
    // updated while a command buffer using it is pending.
    bindWhole(m_rasterizeDescriptors, m_rasterizeSet, 0, forward.projected());
    bindWhole(m_rasterizeDescriptors, m_rasterizeSet, 1, forward.sortedValues());
    bindWhole(m_rasterizeDescriptors, m_rasterizeSet, 2, forward.tileRanges());
    bindWhole(m_rasterizeDescriptors, m_rasterizeSet, 3, forward.pixelState());
    bindView(m_rasterizeDescriptors, m_rasterizeSet, 4, dLossDImage);
    bindWhole(m_rasterizeDescriptors, m_rasterizeSet, 5, *m_projectedGradients);
    // Never read unless the depth flag is set; the forward's depth buffer stands in.
    if (depthGradient)
    {
        bindView(m_rasterizeDescriptors, m_rasterizeSet, 6, dLossDDepth);
    }
    else
    {
        bindWhole(m_rasterizeDescriptors, m_rasterizeSet, 6, forward.depthBuffer());
    }

    bindView(m_projectDescriptors, m_projectSet, 0, params.means);
    bindView(m_projectDescriptors, m_projectSet, 1, params.scales);
    bindView(m_projectDescriptors, m_projectSet, 2, params.rotations);
    bindView(m_projectDescriptors, m_projectSet, 3, params.opacity);
    bindView(m_projectDescriptors, m_projectSet, 4, params.shCoefficients);
    bindWhole(m_projectDescriptors, m_projectSet, 5, forward.projected());
    bindWhole(m_projectDescriptors, m_projectSet, 6, *m_projectedGradients);
    bindView(m_projectDescriptors, m_projectSet, 7, gradients.means);
    bindView(m_projectDescriptors, m_projectSet, 8, gradients.scales);
    bindView(m_projectDescriptors, m_projectSet, 9, gradients.rotations);
    bindView(m_projectDescriptors, m_projectSet, 10, gradients.opacity);
    bindView(m_projectDescriptors, m_projectSet, 11, gradients.shCoefficients);
    bindView(m_projectDescriptors, m_projectSet, 12, gradients.screenMeans);

    VkDevice dev = m_device.handle();
    VK_CHECK(vkResetCommandPool(dev, m_commandPool, 0), "Failed to reset BackwardPass command pool");

    CommandBuffer cmd(dev, m_commandPool);
    cmd.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    Device::recordGlobalBarrier(cmd.getHandle()); // see Device: queue contract
    VkCommandBuffer handle = cmd.getHandle();

    if (imageUpload != nullptr)
    {
        recordUploadCopy(cmd, *imageUpload);
    }
    if (depthUpload != nullptr)
    {
        recordUploadCopy(cmd, *depthUpload);
    }

    // The atomics accumulate, so the scratch must start at zero.
    vkCmdFillBuffer(handle, m_projectedGradients->handle(), 0, projectedGradientBytes, 0);
    cmd.bufferBarrier(m_projectedGradients->handle(), VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    cmd.pushConstants(m_rasterizePipeline.getLayout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      sizeof(CameraUniforms), &camera);
    cmd.dispatchCompute(m_rasterizePipeline.getHandle(), m_rasterizePipeline.getLayout(),
                        m_rasterizeSet, camera.tilesX, camera.tilesY, 1);

    cmd.bufferBarrier(m_projectedGradients->handle(),
                      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT,
                      VK_ACCESS_SHADER_READ_BIT);

    cmd.pushConstants(m_projectPipeline.getLayout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      sizeof(CameraUniforms), &camera);
    cmd.dispatchCompute(m_projectPipeline.getHandle(), m_projectPipeline.getLayout(), m_projectSet,
                        dispatchGroups(count, kWorkgroupSize), 1, 1);

    Device::recordGlobalBarrier(cmd.getHandle()); // see Device: queue contract
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
             "Failed to submit BackwardPass");
    VK_CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX),
             "Failed to wait for BackwardPass");
    vkDestroyFence(dev, fence, nullptr);
}

} // namespace sabercat
