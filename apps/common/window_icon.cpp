// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "window_icon.h"

#include "resources/sabercat_resource.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

namespace sabercat::app
{

void setWindowIconFromResource(GLFWwindow * window)
{
#ifdef _WIN32
    if (window == nullptr)
    {
        return;
    }
    HWND hwnd = glfwGetWin32Window(window);
    if (hwnd == nullptr)
    {
        return;
    }
    const HINSTANCE instance = GetModuleHandleW(nullptr);
    // LR_SHARED: Windows caches these icons and owns them, so nothing is freed
    // here. Each is loaded at the size the shell asks for.
    HICON big = static_cast<HICON>(LoadImageW(instance, MAKEINTRESOURCEW(IDI_SABERCAT), IMAGE_ICON,
                                              GetSystemMetrics(SM_CXICON), GetSystemMetrics(SM_CYICON),
                                              LR_SHARED));
    HICON small = static_cast<HICON>(LoadImageW(instance, MAKEINTRESOURCEW(IDI_SABERCAT), IMAGE_ICON,
                                                GetSystemMetrics(SM_CXSMICON), GetSystemMetrics(SM_CYSMICON),
                                                LR_SHARED));
    if (big != nullptr)
    {
        SendMessageW(hwnd, WM_SETICON, ICON_BIG, reinterpret_cast<LPARAM>(big));
    }
    if (small != nullptr)
    {
        SendMessageW(hwnd, WM_SETICON, ICON_SMALL, reinterpret_cast<LPARAM>(small));
    }
#else
    (void)window;
#endif
}

} // namespace sabercat::app
