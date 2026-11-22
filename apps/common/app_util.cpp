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

const DebugViewInfo * findDebugView(const char * name)
{
    if (name == nullptr)
    {
        return nullptr;
    }
    for (const DebugViewInfo & info : kDebugViews)
    {
        if (std::strcmp(info.name, name) == 0)
        {
            return &info;
        }
    }
    return nullptr;
}

const char * debugViewKey(DebugView view)
{
    switch (view)
    {
    case DebugView::Final:
        return "final";
    case DebugView::Alpha:
        return "alpha";
    case DebugView::Depth:
        return "depth";
    case DebugView::TileDensity:
        return "tileDensity";
    case DebugView::SHDegree:
        return "shDegree";
    }
    return "?";
}

bool parseDebugViewKey(const char * key, DebugView & view)
{
    if (key == nullptr)
    {
        return false;
    }
    for (const DebugView candidate : kDebugViewCycle)
    {
        if (std::strcmp(debugViewKey(candidate), key) == 0)
        {
            view = candidate;
            return true;
        }
    }
    return false;
}

void checkVkResult(VkResult result)
{
    VK_VALIDATE(result == VK_SUCCESS, "Dear ImGui Vulkan backend call failed");
}

} // namespace sabercat::app
