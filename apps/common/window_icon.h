// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The window icon for the apps: the Sabercat icon (IDI_SABERCAT, the
// executable's icon resource) on the title bar and the taskbar button.
#pragma once

// The GLFW window type (declared the same way glfw3.h declares it).
struct GLFWwindow;

namespace sabercat::app
{

// Sets the big and small icons of a window from the executable's IDI_SABERCAT
// resource. Call it once after glfwCreateWindow. A no-op on other platforms
// and for a null window.
void setWindowIconFromResource(GLFWwindow * window);

} // namespace sabercat::app
