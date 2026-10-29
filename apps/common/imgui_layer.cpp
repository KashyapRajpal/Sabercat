// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "imgui_layer.h"

#include "app_util.h"

#include "sabercat/vk_check.h"

#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

namespace sabercat::app
{

void ImGuiLayer::init(GLFWwindow * window, Renderer & renderer)
{
    // -- Dear ImGui: the debug UI. App-level only (renderer.h has no
    // idea this exists) -- registered as a draw callback the renderer
    // invokes at the right point in present() (see setUiRenderCallback).
    // Uses dynamic rendering (vkCmdBeginRendering), not a VkRenderPass:
    // this renderer has none, being pure-compute otherwise, and
    // Device enables VK_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES
    // (core in the 1.3 it targets) specifically so a UI layer never
    // needs one.
    VkDescriptorPoolSize imguiPoolSizes[] = {
        {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8},
    };
    VkDescriptorPoolCreateInfo imguiPoolInfo{};
    imguiPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    imguiPoolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    imguiPoolInfo.maxSets = 8;
    imguiPoolInfo.poolSizeCount = 1;
    imguiPoolInfo.pPoolSizes = imguiPoolSizes;
    VK_CHECK(vkCreateDescriptorPool(renderer.device(), &imguiPoolInfo, nullptr, &m_pool),
            "Failed to create Dear ImGui descriptor pool");

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForVulkan(window, true);

    VkFormat colorFormat = renderer.swapchainFormat();
    VkPipelineRenderingCreateInfo pipelineRenderingInfo{};
    pipelineRenderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    pipelineRenderingInfo.colorAttachmentCount = 1;
    pipelineRenderingInfo.pColorAttachmentFormats = &colorFormat;

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.Instance = renderer.instance();
    initInfo.PhysicalDevice = renderer.physicalDevice();
    initInfo.Device = renderer.device();
    initInfo.QueueFamily = renderer.graphicsQueueFamily();
    initInfo.Queue = renderer.graphicsQueue();
    initInfo.DescriptorPool = m_pool;
    initInfo.MinImageCount = 2;
    initInfo.ImageCount = 2;
    initInfo.UseDynamicRendering = true;
    initInfo.PipelineRenderingCreateInfo = pipelineRenderingInfo;
    initInfo.CheckVkResultFn = checkVkResult;
    ImGui_ImplVulkan_Init(&initInfo);
    m_initialized = true;

    renderer.setUiRenderCallback([](VkCommandBuffer cmd)
                                 { ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd); });
}

void ImGuiLayer::beginFrame()
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void ImGuiLayer::shutdown(Renderer & renderer)
{
    // Dear ImGui's Vulkan backend owns real device resources (font texture,
    // pipeline) that must be destroyed before Device::shutdown() frees the
    // VmaAllocator/VkDevice they depend on -- the same ordering hazard fixed
    // in Renderer::shutdown() for its own GPU-resource-holding members.
    if (m_initialized)
    {
        vkDeviceWaitIdle(renderer.device());
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }
    if (m_pool != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(renderer.device(), m_pool, nullptr);
    }
    m_pool = VK_NULL_HANDLE;
    m_initialized = false;
}

} // namespace sabercat::app
