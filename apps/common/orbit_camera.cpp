// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Orbit camera input (apps/common): polls the mouse and keyboard each frame.
#include "orbit_camera.h"

#include <GLFW/glfw3.h>
#include <imgui.h>

#include <algorithm>
#include <cmath>

namespace sabercat::app
{

void applyOrbitMouse(OrbitCamera & cam, float deltaX, float deltaY, bool orbit, bool pan, float wheel)
{
    constexpr float kOrbitSpeed = 0.005f;
    constexpr float kPanSpeed = 0.0015f;
    constexpr float kZoomStep = 0.9f; // multiplicative per scroll notch

    if (orbit)
    {
        cam.applyDelta(-deltaX * kOrbitSpeed, -deltaY * kOrbitSpeed, 0.0f);
    }
    else if (pan)
    {
        // cam.right()/up() come straight from the orientation quaternion
        // (not cross(forward, worldUp)) so panning stays well-defined
        // even looking straight up/down, where that cross product would
        // degenerate the same way the old lookAt reconstruction did.
        float panScale = kPanSpeed * cam.distance; // fixed pixel drag -> fixed view fraction
        cam.pivot -= cam.right() * deltaX * panScale;
        cam.pivot += cam.up() * deltaY * panScale;
    }

    // Skipped at zero so a frame without scroll never re-clamps the distance.
    if (wheel != 0.0f)
    {
        cam.distance = std::max(OrbitCamera::kMinDistance, cam.distance * std::pow(kZoomStep, wheel));
    }
}

void updateOrbitCamera(GLFWwindow * window, OrbitCamera & cam, float deltaTime, bool suppressLeftOrbit)
{
    static double lastCursorX = 0.0;
    static double lastCursorY = 0.0;
    static bool haveLastCursor = false;

    double cursorX, cursorY;
    glfwGetCursorPos(window, &cursorX, &cursorY);
    double deltaX = haveLastCursor ? cursorX - lastCursorX : 0.0;
    double deltaY = haveLastCursor ? cursorY - lastCursorY : 0.0;
    lastCursorX = cursorX;
    lastCursorY = cursorY;
    haveLastCursor = true;

    // With suppressLeftOrbit the left button orbits only while Alt is held.
    const bool altHeld = glfwGetKey(window, GLFW_KEY_LEFT_ALT) == GLFW_PRESS
                         || glfwGetKey(window, GLFW_KEY_RIGHT_ALT) == GLFW_PRESS;
    const bool leftOrbitAllowed = !suppressLeftOrbit || altHeld;

    const ImGuiIO & io = ImGui::GetIO();
    if (!io.WantCaptureMouse)
    {
        applyOrbitMouse(cam, static_cast<float>(deltaX), static_cast<float>(deltaY),
                        leftOrbitAllowed && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS,
                        glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS,
                        io.MouseWheel);
    }

    if (!io.WantCaptureKeyboard)
    {
        constexpr float kRollSpeed = 1.5f; // radians/second
        if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS)
        {
            cam.applyDelta(0.0f, 0.0f, -kRollSpeed * deltaTime);
        }
        if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS)
        {
            cam.applyDelta(0.0f, 0.0f, kRollSpeed * deltaTime);
        }
    }
}

} // namespace sabercat::app
