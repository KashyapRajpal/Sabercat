// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Dear ImGui bootstrap for the apps: the descriptor pool, the GLFW + Vulkan
// backends, and the draw callback the renderer invokes inside present().
#pragma once

#include "sabercat/renderer.h"

#include <vulkan/vulkan.h>

// The GLFW window type (declared the same way glfw3.h declares it).
struct GLFWwindow;

namespace sabercat::app
{

// Order of use: init() once the surface is attached; beginFrame() at the top of
// each frame; shutdown() before renderer.shutdown().
struct ImGuiLayer
{
    void init(GLFWwindow * window, Renderer & renderer);
    void beginFrame();
    void shutdown(Renderer & renderer);

private:
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    bool m_initialized = false;
};

} // namespace sabercat::app
