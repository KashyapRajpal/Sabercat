// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "borrowed_vulkan_host.h"

#include "sabercat/logger.h"
#include "sabercat/vk_check.h"

#include <string>
#include <vector>

namespace sabercat::app
{

VKAPI_ATTR VkBool32 VKAPI_CALL BorrowedVulkanHost::onMessage(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT * data, void *)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
    {
        Logger::error(std::string("[validation] ") + data->pMessage);
    }
    return VK_FALSE;
}

void BorrowedVulkanHost::create(const std::vector<const char *> & windowExtensions,
                                bool presentation, bool enableValidation)
{
    // The requirements object needs a physical device, which needs an
    // instance: create the instance first with what presenting needs.
    std::vector<const char *> extensions = windowExtensions;
    if (enableValidation)
    {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    const char * layer = "VK_LAYER_KHRONOS_validation";
    VkDebugUtilsMessengerCreateInfoEXT debugInfo{};
    debugInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    debugInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                                | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debugInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                            | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                            | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debugInfo.pfnUserCallback = onMessage;
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Sabercat Viewer (borrowed device)";
    appInfo.apiVersion = sabercat::kRequiredVulkanApiVersion;
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &appInfo;
    if (enableValidation)
    {
        ici.pNext = &debugInfo;
        ici.enabledLayerCount = 1;
        ici.ppEnabledLayerNames = &layer;
    }
    ici.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    ici.ppEnabledExtensionNames = extensions.data();
    VK_CHECK(vkCreateInstance(&ici, nullptr, &instance), "borrowed host: vkCreateInstance");
    if (enableValidation)
    {
        auto createMessenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
        VK_VALIDATE(createMessenger != nullptr, "borrowed host: debug messenger entry point missing");
        VK_CHECK(createMessenger(instance, &debugInfo, nullptr, &messenger),
                 "borrowed host: vkCreateDebugUtilsMessengerEXT");
    }

    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    VK_VALIDATE(count > 0, "No Vulkan-capable GPU found");
    std::vector<VkPhysicalDevice> gpus(count);
    vkEnumeratePhysicalDevices(instance, &count, gpus.data());
    gpu = gpus.at(0);
    for (VkPhysicalDevice candidate : gpus)
    {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(candidate, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        {
            gpu = candidate;
            break;
        }
    }
    // One family that does graphics AND compute: Sabercat submits its
    // compute there, present() its blit, ImGui its draw.
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, families.data());
    const VkQueueFlags both = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
    for (uint32_t i = 0; i < count && family == UINT32_MAX; ++i)
    {
        if ((families[i].queueFlags & both) == both)
        {
            family = i;
        }
    }
    VK_VALIDATE(family != UINT32_MAX, "No queue family with graphics and compute on the GPU");

    sabercat::DeviceRequirements requirements(
        gpu, {sabercat::DeviceRequirements::Need::IfSupported,
              presentation ? sabercat::DeviceRequirements::Need::Required
                           : sabercat::DeviceRequirements::Need::None});
    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = family;
    qci.queueCount = 1;
    qci.pQueuePriorities = &priority;
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.pNext = requirements.featureChain();
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = static_cast<uint32_t>(requirements.deviceExtensions().size());
    dci.ppEnabledExtensionNames = requirements.deviceExtensions().data();
    VK_CHECK(vkCreateDevice(gpu, &dci, nullptr, &device), "borrowed host: vkCreateDevice");
    vkGetDeviceQueue(device, family, 0, &queue);
    enabled = requirements.enabled();
}

sabercat::VulkanContext BorrowedVulkanHost::context() const
{
    sabercat::VulkanContext c;
    c.instance = instance;
    c.physicalDevice = gpu;
    c.device = device;
    c.graphicsQueue = queue;
    c.graphicsFamily = family;
    c.computeQueue = queue;
    c.computeFamily = family;
    return c;
}

void BorrowedVulkanHost::destroy()
{
    if (device != VK_NULL_HANDLE)
    {
        vkDestroyDevice(device, nullptr);
    }
    if (messenger != VK_NULL_HANDLE)
    {
        auto destroyMessenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
        destroyMessenger(instance, messenger, nullptr);
    }
    if (instance != VK_NULL_HANDLE)
    {
        vkDestroyInstance(instance, nullptr);
    }
}

} // namespace sabercat::app
