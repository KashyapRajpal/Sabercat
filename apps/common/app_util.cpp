// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "app_util.h"

#include "app_policy.h"

#include "sabercat/logger.h"
#include "sabercat/vk_check.h"

#include <cstring>
#include <vector>

namespace sabercat::app
{

namespace
{

bool validationLayerInstalled()
{
    uint32_t count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> layers(count);
    vkEnumerateInstanceLayerProperties(&count, layers.data());
    for (const VkLayerProperties & layer : layers)
    {
        if (std::strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0)
        {
            return true;
        }
    }
    return false;
}

} // namespace

bool resolveValidation(bool validationFlag, bool noValidationFlag)
{
#ifdef NDEBUG
    const bool requested = validationFlag;
#else
    const bool requested = true;
#endif
    const bool installed = validationLayerInstalled();
    switch (decideValidation(requested, noValidationFlag, installed))
    {
    case ValidationChoice::On:
        Logger::info("Validation layer on");
        return true;
    case ValidationChoice::OffByFlag:
        Logger::info("Validation layer off (--no-validation)");
        return false;
    case ValidationChoice::OffNotRequested:
        Logger::info(installed ? "Validation layer off (Release build; --validation enables it)"
                               : "Validation layer not installed; running without validation");
        return false;
    case ValidationChoice::OffNotInstalled:
        Logger::warning("Validation layer VK_LAYER_KHRONOS_validation requested but not installed; "
                        "continuing without validation");
        return false;
    }
    return false;
}

const char * debugViewName(DebugView view)
{
    switch (view)
    {
    case DebugView::Final:
        return "Final";
    case DebugView::Alpha:
        return "Alpha";
    case DebugView::Depth:
        return "Depth";
    case DebugView::TileDensity:
        return "TileDensity";
    case DebugView::SHDegree:
        return "SHDegree";
    }
    return "?";
}

void checkVkResult(VkResult result)
{
    VK_VALIDATE(result == VK_SUCCESS, "Dear ImGui Vulkan backend call failed");
}

} // namespace sabercat::app
