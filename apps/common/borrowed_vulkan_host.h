// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Ownership model B for the apps' --borrowed mode: the app creates the instance
// and device (from sabercat::DeviceRequirements, presentation required) and
// Sabercat borrows them.
#pragma once

#include "sabercat/device.h"

#include <cstdint>
#include <vector>

#include <vulkan/vulkan.h>

namespace sabercat::app
{

// `--borrowed` runs the viewer as an ownership-model-B host -- the
// APP creates the instance and device (from sabercat::DeviceRequirements,
// presentation required) and Sabercat borrows them. Exists to exercise the
// presenting path on a borrowed device end to end; the default is model A.
struct BorrowedVulkanHost
{
    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    VkPhysicalDevice gpu = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t family = UINT32_MAX;
    sabercat::EnabledDeviceFeatures enabled{};

    static VKAPI_ATTR VkBool32 VKAPI_CALL onMessage(
        VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT,
        const VkDebugUtilsMessengerCallbackDataEXT * data, void *);

    // Creates the instance (with `windowExtensions`, plus the validation layer
    // and debug utils when `enableValidation`), picks a device and creates it
    // with one graphics+compute queue. `presentation` false (headless: no
    // window, no swapchain) requires only compute, so an empty
    // `windowExtensions` list is then the normal case. The caller decides
    // `enableValidation` (see resolveValidation in app_util.h): this only
    // enables a layer the caller has checked is installed.
    void create(const std::vector<const char *> & windowExtensions, bool presentation = true,
                bool enableValidation = false);

    sabercat::VulkanContext context() const;

    // After the renderer's shutdown(): borrowers go first.
    void destroy();
};

} // namespace sabercat::app
