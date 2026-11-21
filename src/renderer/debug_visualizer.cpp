// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/debug_visualizer.h"

#include "sabercat/buffer.h"
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

struct DebugParams
{
    uint32_t mode;
    uint32_t imageWidth;
    uint32_t imageHeight;
    uint32_t tilesX;
    float depthScale;
    float densityScale;
};

VkDescriptorSetLayoutBinding storageImageBinding(uint32_t binding)
{
    VkDescriptorSetLayoutBinding b{};
    b.binding = binding;
    b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    return b;
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

} // namespace

DebugVisualizer::DebugVisualizer(Device & device, uint32_t width, uint32_t height,
                                 const std::string & shaderDir)
    : m_device(device), m_width(0), m_height(0), m_descriptors(device.handle())
{
    VkDevice dev = device.handle();

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        storageImageBinding(0),  // colorImage
        storageImageBinding(1),  // depthImage
        storageBufferBinding(2), // tileRanges
        storageImageBinding(3),  // displayImage
        storageBufferBinding(4), // projected (SHDegree)
        storageBufferBinding(5), // sortedValues (SHDegree)
        storageBufferBinding(6), // shDegrees (SHDegree)
    };
    m_layout = m_descriptors.createLayout(bindings);
    // One set per frame-in-flight (see recordSlot()'s comment on why a single
    // shared set raced a still-pending command buffer), plus one for the
    // headless slot (renderHeadless()) -- pool sizes scale by
    // kMaxFramesInFlight + 1 accordingly.
    m_descriptors.createPool(kMaxFramesInFlight + 1,
                             {{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 3 * (kMaxFramesInFlight + 1)},
                              {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 * (kMaxFramesInFlight + 1)}});
    for (uint32_t i = 0; i < kMaxFramesInFlight; ++i)
    {
        m_sets[i] = m_descriptors.allocate(m_layout);
    }

    ShaderCompiler compiler(dev);
    auto spirv = compiler.loadSPIRV(shaderDir + "/debug_visualize.comp.spv");
    VkShaderModule module = compiler.createShaderModule(spirv);
    VkPipelineLayout layout = m_pipeline.createPipelineLayout(dev, m_layout, sizeof(DebugParams),
                                                              VK_SHADER_STAGE_COMPUTE_BIT);
    m_pipeline.createComputePipeline(dev, module, layout);
    compiler.destroyShaderModule(module);

    resize(width, height);
}

DebugVisualizer::~DebugVisualizer()
{
    m_pipeline.destroy(m_device.handle());
}

void DebugVisualizer::resize(uint32_t width, uint32_t height)
{
    if (width == m_width && height == m_height)
    {
        return;
    }
    m_width = width;
    m_height = height;

    // The headless slot is dropped and recreated lazily at the new size; its
    // descriptor set is kept and rebound on the next renderHeadless().
    m_headlessImage.reset();
    m_headlessBuffer.reset();

    for (uint32_t i = 0; i < kMaxFramesInFlight; ++i)
    {
        m_firstUse[i] = true;

        m_displayImages[i] = std::make_unique<Image>(
            m_device, width, height, VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

        m_descriptors.bindImage(m_sets[i], 3, m_displayImages[i]->getImageView(), VK_NULL_HANDLE,
                                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    }
}

void DebugVisualizer::record(VkCommandBuffer cmd, DebugView mode, const ForwardPass & forward,
                             float depthScale, float densityScale, uint32_t frameIndex)
{
    recordSlot(cmd, mode, forward, depthScale, densityScale, m_sets[frameIndex],
               *m_displayImages[frameIndex], m_firstUse[frameIndex]);
}

const Buffer & DebugVisualizer::renderHeadless(DebugView mode, const ForwardPass & forward,
                                               float depthScale, float densityScale,
                                               VkImageLayout outputLayout)
{
    if (!m_headlessImage)
    {
        m_headlessImage = std::make_unique<Image>(
            m_device, m_width, m_height, VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
        m_headlessFirstUse = true;
        if (m_headlessSet == VK_NULL_HANDLE)
        {
            m_headlessSet = m_descriptors.allocate(m_layout);
        }
        m_descriptors.bindImage(m_headlessSet, 3, m_headlessImage->getImageView(), VK_NULL_HANDLE,
                                VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    }
    if (!m_headlessBuffer)
    {
        // RGBA32F, row-major: 16 B/px. Default usage includes TRANSFER_DST.
        m_headlessBuffer = std::make_unique<Buffer>(
            m_device, VkDeviceSize(m_width) * m_height * 4 * sizeof(float));
    }
    const Buffer & buffer = *m_headlessBuffer;

    // Copies a whole RGBA32F image (in TRANSFER_SRC_OPTIMAL) into the buffer.
    const auto copyToBuffer = [&](VkCommandBuffer cmd, const Image & image)
    {
        VkBufferImageCopy region{};
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.layerCount = 1;
        region.imageExtent = {m_width, m_height, 1};
        vkCmdCopyImageToBuffer(cmd, image.getHandle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               buffer.handle(), 1, &region);
    };

    m_device.immediateSubmit(
        [&](VkCommandBuffer cmd)
        {
            if (mode == DebugView::Final)
            {
                // The color image itself: no dispatch.
                const Image & color = forward.outputImage();
                if (outputLayout != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
                {
                    color.transitionLayout(cmd, outputLayout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
                }
                copyToBuffer(cmd, color);
                color.transitionLayout(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                       VK_IMAGE_LAYOUT_GENERAL);
                return;
            }

            if (outputLayout != VK_IMAGE_LAYOUT_GENERAL)
            {
                forward.outputImage().transitionLayout(cmd, outputLayout,
                                                       VK_IMAGE_LAYOUT_GENERAL);
            }
            recordSlot(cmd, mode, forward, depthScale, densityScale, m_headlessSet,
                       *m_headlessImage, m_headlessFirstUse);
            // recordSlot leaves the display image in TRANSFER_SRC_OPTIMAL.
            copyToBuffer(cmd, *m_headlessImage);
        });

    return buffer;
}

void DebugVisualizer::recordSlot(VkCommandBuffer cmd, DebugView mode, const ForwardPass & forward,
                                 float depthScale, float densityScale, VkDescriptorSet set,
                                 const Image & displayImage, bool & firstUse)
{
    // Rebind the input images every call -- cheap, and keeps this correct
    // even if ForwardPass recreated them (a resize) since the last record().
    // Safe to rewrite THIS frame's set here: the caller only reaches this
    // call after waiting on this exact frame-in-flight slot's own fence
    // (Renderer::present() does so before recording anything), so no
    // pending command buffer can still be referencing it. The headless slot
    // is safe for the same reason: renderHeadless() is synchronous.
    // The buffers are rebound for the same reason: run() grows them
    // (gaussian count, instance count) by reallocating.
    m_descriptors.bindImage(set, 0, forward.outputImage().getImageView(), VK_NULL_HANDLE,
                            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    m_descriptors.bindImage(set, 1, forward.depthImage().getImageView(), VK_NULL_HANDLE,
                            VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    const auto bindWhole = [&](uint32_t binding, const Buffer & buffer)
    {
        m_descriptors.bindBuffer(set, binding, buffer.handle(), 0, buffer.size(),
                                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    };
    bindWhole(2, forward.tileRanges());
    bindWhole(4, forward.projected());
    bindWhole(5, forward.sortedValues());
    bindWhole(6, forward.shDegrees());

    displayImage.transitionLayout(
        cmd,
        firstUse ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_IMAGE_LAYOUT_GENERAL);
    firstUse = false;

    DebugParams params{};
    params.mode = static_cast<uint32_t>(mode);
    params.imageWidth = m_width;
    params.imageHeight = m_height;
    params.tilesX = dispatchGroups(m_width, kTileSize);
    params.depthScale = depthScale;
    params.densityScale = densityScale;

    vkCmdPushConstants(cmd, m_pipeline.getLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                       sizeof(params), &params);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline.getHandle());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline.getLayout(), 0, 1,
                            &set, 0, nullptr);
    vkCmdDispatch(cmd, dispatchGroups(m_width, kTileSize), dispatchGroups(m_height, kTileSize), 1);

    displayImage.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL,
                                  VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
}

} // namespace sabercat
