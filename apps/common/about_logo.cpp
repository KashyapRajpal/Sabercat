// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "about_logo.h"

#include "resources/sabercat_logo_rgba.h"

#include "sabercat/buffer.h"
#include "sabercat/device.h"
#include "sabercat/vk_check.h"

#include <imgui_impl_vulkan.h>

#include <cstdint>

namespace sabercat::app
{

void AboutLogo::create(Device & device)
{
    m_device = &device;
    constexpr uint32_t size = kSabercatLogoSize;
    m_image = std::make_unique<Image>(device, size, size, VK_FORMAT_R8G8B8A8_UNORM,
                                      VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    // The pixels go up through a host-visible staging buffer, then a copy in a
    // one-time submit that leaves the image ready for sampling.
    Buffer staging(device, sizeof(kSabercatLogoRgba));
    staging.upload(kSabercatLogoRgba, sizeof(kSabercatLogoRgba));
    device.immediateSubmit(
        [&](VkCommandBuffer cmd)
        {
            m_image->transitionLayout(cmd, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
            VkBufferImageCopy region{};
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.layerCount = 1;
            region.imageExtent = {size, size, 1};
            vkCmdCopyBufferToImage(cmd, staging.handle(), m_image->getHandle(),
                                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
            m_image->transitionLayout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        });

    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(device.handle(), &samplerInfo, nullptr, &m_sampler),
             "Failed to create the logo sampler");

    // The backend allocates the descriptor set from its own pool; ImGui draws
    // with the set's handle as the texture id.
    m_descriptorSet = ImGui_ImplVulkan_AddTexture(m_sampler, m_image->getImageView(),
                                                  VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    m_texture = static_cast<ImTextureID>(reinterpret_cast<uintptr_t>(m_descriptorSet));
}

void AboutLogo::destroy()
{
    if (m_device == nullptr)
    {
        return;
    }
    // The GPU may still be sampling the texture: wait for it before the
    // registration and the Vulkan objects go.
    vkDeviceWaitIdle(m_device->handle());
    if (m_descriptorSet != VK_NULL_HANDLE)
    {
        ImGui_ImplVulkan_RemoveTexture(m_descriptorSet);
    }
    if (m_sampler != VK_NULL_HANDLE)
    {
        vkDestroySampler(m_device->handle(), m_sampler, nullptr);
    }
    m_image.reset();
    m_sampler = VK_NULL_HANDLE;
    m_descriptorSet = VK_NULL_HANDLE;
    m_texture = 0;
    m_device = nullptr;
}

} // namespace sabercat::app
