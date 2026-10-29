// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vulkan/vulkan.h>

#include <stdexcept>
#include <string>

namespace sabercat
{

// Human-readable name for the common VkResult codes, for error messages.
inline const char * vkResultString(VkResult result)
{
    switch (result)
    {
    case VK_SUCCESS:
        return "VK_SUCCESS";
    case VK_NOT_READY:
        return "VK_NOT_READY";
    case VK_TIMEOUT:
        return "VK_TIMEOUT";
    case VK_INCOMPLETE:
        return "VK_INCOMPLETE";
    case VK_ERROR_OUT_OF_HOST_MEMORY:
        return "VK_ERROR_OUT_OF_HOST_MEMORY";
    case VK_ERROR_OUT_OF_DEVICE_MEMORY:
        return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
    case VK_ERROR_INITIALIZATION_FAILED:
        return "VK_ERROR_INITIALIZATION_FAILED";
    case VK_ERROR_DEVICE_LOST:
        return "VK_ERROR_DEVICE_LOST";
    case VK_ERROR_MEMORY_MAP_FAILED:
        return "VK_ERROR_MEMORY_MAP_FAILED";
    case VK_ERROR_LAYER_NOT_PRESENT:
        return "VK_ERROR_LAYER_NOT_PRESENT";
    case VK_ERROR_EXTENSION_NOT_PRESENT:
        return "VK_ERROR_EXTENSION_NOT_PRESENT";
    case VK_ERROR_FEATURE_NOT_PRESENT:
        return "VK_ERROR_FEATURE_NOT_PRESENT";
    case VK_ERROR_INCOMPATIBLE_DRIVER:
        return "VK_ERROR_INCOMPATIBLE_DRIVER";
    case VK_ERROR_TOO_MANY_OBJECTS:
        return "VK_ERROR_TOO_MANY_OBJECTS";
    case VK_ERROR_FORMAT_NOT_SUPPORTED:
        return "VK_ERROR_FORMAT_NOT_SUPPORTED";
    case VK_ERROR_OUT_OF_POOL_MEMORY:
        return "VK_ERROR_OUT_OF_POOL_MEMORY";
    case VK_ERROR_SURFACE_LOST_KHR:
        return "VK_ERROR_SURFACE_LOST_KHR";
    case VK_ERROR_NATIVE_WINDOW_IN_USE_KHR:
        return "VK_ERROR_NATIVE_WINDOW_IN_USE_KHR";
    case VK_SUBOPTIMAL_KHR:
        return "VK_SUBOPTIMAL_KHR";
    case VK_ERROR_OUT_OF_DATE_KHR:
        return "VK_ERROR_OUT_OF_DATE_KHR";
    default:
        return "VK_ERROR_<unknown>";
    }
}

} // namespace sabercat

// VK_CHECK: wrap a Vulkan (or VMA) call that returns a VkResult. Evaluates the
// call exactly once; if it is not VK_SUCCESS, throws std::runtime_error with the
// message, the result code, and the source location.
//
//   VK_CHECK(vkCreateCommandPool(dev, &ci, nullptr, &pool),
//            "Failed to create transfer command pool");
#define VK_CHECK(call, message)                                                                    \
    do                                                                                             \
    {                                                                                              \
        const VkResult vkCheckResult_ = (call);                                                    \
        if (vkCheckResult_ != VK_SUCCESS)                                                          \
        {                                                                                          \
            throw std::runtime_error(std::string(message) + " ["                                   \
                                     + ::sabercat::vkResultString(vkCheckResult_)                  \
                                     + "] at " __FILE__ ":" + std::to_string(__LINE__));           \
        }                                                                                          \
    } while (0)

// VK_VALIDATE: guard an arbitrary boolean condition (things that do NOT return a
// VkResult -- a null function pointer, a count, a capability). Throws with the
// message and source location if the condition is false.
//
//   VK_VALIDATE(deviceCount > 0, "No Vulkan-capable GPU found");
#define VK_VALIDATE(condition, message)                                                            \
    do                                                                                             \
    {                                                                                              \
        if (!(condition))                                                                          \
        {                                                                                          \
            throw std::runtime_error(std::string(message) + " at " __FILE__ ":"                    \
                                     + std::to_string(__LINE__));                                  \
        }                                                                                          \
    } while (0)
