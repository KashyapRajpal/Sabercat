// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The Sabercat logo as a Vulkan texture for Dear ImGui's Help > About. The
// pixels are built into the app (resources/sabercat_logo_rgba.h): uploaded
// once, registered with the ImGui Vulkan backend, and destroyed before ImGui
// shuts down.
#pragma once

#include <imgui.h>

#include <vulkan/vulkan.h>

#include "sabercat/image.h"

#include <memory>

namespace sabercat
{
class Device;
}

namespace sabercat::app
{

class AboutLogo
{
  public:
    // Uploads the logo on `device` and registers it with ImGui. Call after
    // ImGui's Vulkan backend is initialized.
    void create(Device & device);

    // Unregisters and frees the texture after the device is idle. Call before
    // ImGui's Vulkan backend shuts down. Safe to call when create() never ran.
    void destroy();

    // The ImGui texture to pass to ImGui::Image; 0 before create().
    ImTextureID texture() const
    {
        return m_texture;
    }

  private:
    Device * m_device = nullptr;
    std::unique_ptr<Image> m_image;
    VkSampler m_sampler = VK_NULL_HANDLE;
    VkDescriptorSet m_descriptorSet = VK_NULL_HANDLE;
    ImTextureID m_texture = 0;
};

} // namespace sabercat::app
