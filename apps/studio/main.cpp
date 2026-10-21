// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Sabercat Studio -- the editing application (docs/studio-design.md). S1 is the
// shell: it owns the Vulkan instance and device (ownership model B, docs/library.md)
// and lends them to the renderer; it edits a project's layers, saves and opens
// .sabercat projects, and exports PLY and glTF.
//
//   sabercat_studio.exe [project.sabercat | scene files...]
//   sabercat_studio.exe --selftest <outDir> <scene>...     (headless, no window)

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <imgui.h>

#include <algorithm>
#include <cctype>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
#endif

#include "about_logo.h"
#include "borrowed_vulkan_host.h"
#include "file_association.h"
#include "imgui_layer.h"
#include "log_files.h"
#include "orbit_camera.h"
#include "shader_dir.h"
#include "app_util.h"
#include "selftest.h"
#include "studio_state.h"
#include "studio_ui.h"
#include "system_info.h"
#include "window_icon.h"

#include "sabercat/logger.h"
#include "sabercat/renderer.h"
#include "sabercat/version.h"
#include "sabercat/vk_check.h"

using namespace sabercat;
using namespace sabercat::app;
using namespace sabercat::studio;

namespace fs = std::filesystem;

// Paths dropped onto the window, picked up by the main loop. GLFW calls this
// from glfwPollEvents on the main thread.
std::vector<std::string> g_droppedPaths;
void onDrop(GLFWwindow *, int count, const char ** paths)
{
    for (int i = 0; i < count; ++i)
    {
        g_droppedPaths.emplace_back(paths[i]);
    }
}

bool isProjectFile(const std::string & path)
{
    std::string extension = fs::u8path(path).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return extension == ".sabercat";
}

// A path from the command line or a drop: a project opens (guarded by the
// unsaved-changes prompt), anything else is added as a layer.
void handlePath(StudioState & state, Renderer & renderer, UiState & ui, const std::string & path)
{
    if (isProjectFile(path))
    {
        Logger::info("Studio: opening project " + path);
        requestOpenProject(state, renderer, ui, path);
    }
    else
    {
        Logger::info("Studio: adding layer " + path);
        requestAddLayer(state, renderer, ui, path);
    }
}

// Command-line arguments as UTF-8. On Windows argv is narrow (the ANSI codepage),
// so a non-ASCII path would arrive mangled; the wide command line is the source.
std::vector<std::string> utf8Arguments(int argc, char * argv[])
{
#ifdef _WIN32
    int wideCount = 0;
    LPWSTR * wide = CommandLineToArgvW(GetCommandLineW(), &wideCount);
    if (wide != nullptr && wideCount == argc)
    {
        std::vector<std::string> result;
        for (int i = 0; i < wideCount; ++i)
        {
            const int bytes = WideCharToMultiByte(CP_UTF8, 0, wide[i], -1, nullptr, 0, nullptr, nullptr);
            std::string text(bytes > 1 ? static_cast<size_t>(bytes - 1) : 0, '\0');
            if (bytes > 1)
            {
                WideCharToMultiByte(CP_UTF8, 0, wide[i], -1, text.data(), bytes, nullptr, nullptr);
            }
            result.push_back(text);
        }
        LocalFree(wide);
        return result;
    }
    if (wide != nullptr)
    {
        LocalFree(wide);
    }
#endif
    return std::vector<std::string>(argv, argv + argc);
}

int main(int argc, char * argv[])
{
    // --validation / --no-validation are the app's own flags, wherever they
    // appear; they are removed here so they are never read as paths.
    // --log <path> and --no-log choose the session log file (see startSessionLog).
    const std::vector<std::string> allArgs = utf8Arguments(argc, argv);
    bool validationFlag = false;
    bool noValidationFlag = false;
    bool noLogFlag = false;
    std::string logPathFlag;
    std::vector<std::string> args;
    for (size_t i = 0; i < allArgs.size(); ++i)
    {
        const std::string & arg = allArgs[i];
        if (arg == "--validation")
        {
            validationFlag = true;
        }
        else if (arg == "--no-validation")
        {
            noValidationFlag = true;
        }
        else if (arg == "--no-log")
        {
            noLogFlag = true;
        }
        else if (arg == "--log" && i + 1 < allArgs.size())
        {
            logPathFlag = allArgs[++i];
        }
        else
        {
            args.push_back(arg);
        }
    }
    std::vector<char *> utf8Argv;
    for (const std::string & arg : args)
    {
        utf8Argv.push_back(const_cast<char *>(arg.c_str()));
    }
    utf8Argv.push_back(nullptr);

    if (args.size() >= 2 && args[1] == "--selftest")
    {
        // The self-test writes no log file unless --log asks for one.
        if (!logPathFlag.empty())
        {
            startSessionLog("studio", logPathFlag, false);
        }
        return runSelfTest(static_cast<int>(args.size()), utf8Argv.data(),
                           resolveValidation(validationFlag, noValidationFlag));
    }
    startSessionLog("studio", logPathFlag, noLogFlag);
    const std::vector<std::string> startupPaths(args.begin() + 1, args.end());

    std::cout << std::endl << "=== Sabercat Studio Starting ===" << std::endl;

    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    const std::string baseTitle = std::string("Sabercat Studio ") + sabercat::kVersionString;
    GLFWwindow * window = glfwCreateWindow(1280, 720, baseTitle.c_str(), nullptr, nullptr);
    setWindowIconFromResource(window);
    glfwSetDropCallback(window, onDrop);

    // The app owns the window; it tells the device which instance extensions
    // its windowing layer needs.
    uint32_t glfwExtCount = 0;
    const char ** glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
    const std::vector<const char *> instanceExtensions(glfwExts, glfwExts + glfwExtCount);

    // Always ownership model B: the Studio creates the device and the renderer
    // borrows it, so the same device can later serve another application too.
    const std::string shaderDir = resolveShaderDir();
    const bool validation = resolveValidation(validationFlag, noValidationFlag);
    Renderer renderer(1280, 720);
    BorrowedVulkanHost host;
    try
    {
        host.create(instanceExtensions, /*presentation=*/true, validation);
        renderer.initializeBorrowed(host.context(), host.enabled, shaderDir);
        Logger::info("Studio running on an app-owned (borrowed) Vulkan device");
    }
    catch (const std::exception & e)
    {
        // e.g. no Vulkan 1.3 GPU, or a required feature is missing: one error
        // naming the reason, then a clean exit.
        Logger::error(std::string("Studio: device setup failed: ") + e.what());
        renderer.shutdown();
        host.destroy();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    // Declared here so the cleanup below can tear ImGui down whether or not the
    // loop threw; ImGui must go before renderer.shutdown().
    ImGuiLayer imgui;
    // The About logo's texture: created once ImGui is up, destroyed with ImGui.
    AboutLogo aboutLogo;
    StudioState state;
    UiState ui;
    ui.window = window; // owns the native file dialogs
    ui.exePath = app::currentExecutablePath(); // for Help > Register .sabercat files
    // Help > About and the session log's system block: the device is up now.
    ui.systemInfo = gatherSystemInfo("Sabercat Studio", &renderer.sabercatDevice(), validation);
    Logger::info(formatSystemInfo(ui.systemInfo, Logger::logFilePath()));
    int result = 0;
    try
    {
        // The app creates the surface from the renderer's instance and hands it
        // back; the release callback destroys it before the instance goes.
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        VK_CHECK(glfwCreateWindowSurface(renderer.instance(), window, nullptr, &surface),
                 "Failed to create window surface");
        renderer.attachSurface(surface,
                               1280,
                               720,
                               [&renderer, surface]()
                               { vkDestroySurfaceKHR(renderer.instance(), surface, nullptr); });

        imgui.init(window, renderer);
        aboutLogo.create(renderer.sabercatDevice());
        ui.aboutLogo = aboutLogo.texture();

        for (const std::string & path : startupPaths)
        {
            handlePath(state, renderer, ui, path);
        }

        double lastFrameTime = glfwGetTime();
        std::string lastTitle;
        while (true)
        {
            glfwPollEvents();

            // Closing the window with unsaved changes asks first; the window
            // is kept open until the discard prompt answers Yes.
            if (glfwWindowShouldClose(window))
            {
                if (!state.dirty || ui.quitConfirmed)
                {
                    break;
                }
                glfwSetWindowShouldClose(window, GLFW_FALSE);
                requestQuit(state, ui);
            }

            // Drops wait until an open dialog has been answered.
            if (!g_droppedPaths.empty() && !ui.dialog.pending())
            {
                const std::vector<std::string> dropped = std::move(g_droppedPaths);
                g_droppedPaths.clear();
                for (const std::string & path : dropped)
                {
                    handlePath(state, renderer, ui, path);
                }
            }

            // At most one rebuild per frame, and only when something changed.
            // It runs before the UI so Frame all and the viewport see this frame's scene.
            if (state.sceneDirty)
            {
                try
                {
                    rebuildScene(state, renderer);
                }
                catch (const std::exception & e)
                {
                    reportError(ui, std::string("Could not build the scene: ") + e.what());
                }
            }
            renderer.setDebugView(kDebugViewCycle[state.debugViewIndex]);
            renderer.setMaxSHDegree(static_cast<uint32_t>(state.shDegree));

            imgui.beginFrame();

            // Camera input reads ImGuiIO's WantCapture flags, so it runs after
            // beginFrame and before the windows that claim the mouse.
            const double now = glfwGetTime();
            const float deltaTime = static_cast<float>(now - lastFrameTime);
            lastFrameTime = now;
            updateOrbitCamera(window, state.camera, deltaTime);
            renderer.setCamera(state.camera.toCamera(renderer.width(), renderer.height()));

            drawStudioUi(state, renderer, ui);
            if (ui.quitConfirmed)
            {
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }

            std::string title = baseTitle;
            if (!state.projectPath.empty())
            {
                const fs::path name = fs::u8path(state.projectPath).filename();
                const std::u8string nameUtf8 = name.u8string();
                title += " - " + std::string(reinterpret_cast<const char *>(nameUtf8.data()), nameUtf8.size());
            }
            if (state.dirty)
            {
                title += " *";
            }
            if (title != lastTitle)
            {
                glfwSetWindowTitle(window, title.c_str());
                lastTitle = title;
            }

            ImGui::Render();
            renderer.forward();
            renderer.present();
        }
    }
    catch (const std::exception & e)
    {
        Logger::error(std::string("Studio FAILED: ") + e.what());
        result = 1;
    }

    // ImGui's Vulkan backend holds device resources, so it goes first. Then the
    // scene's clouds (they point into the device), then the renderer, then the
    // borrowed device last of all. The About logo is registered with ImGui's
    // backend, so it goes before ImGui does.
    aboutLogo.destroy();
    ui.aboutLogo = 0;
    imgui.shutdown(renderer);
    state.project = Project{};
    renderer.shutdown();
    host.destroy();
    glfwDestroyWindow(window);
    glfwTerminate();
    return result;
}
