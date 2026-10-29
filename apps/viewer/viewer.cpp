// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Sabercat Viewer -- interactive viewer application.
//
// Demonstrates the app/library boundary: this app owns the GLFW window and
// injects the required Vulkan instance extensions; the renderer stays
// windowing-agnostic and only ever sees a VkSurfaceKHR.

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <imgui.h>

#include <algorithm>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <iostream>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

#include "about_logo.h"
#include "about_panel.h"
#include "app_policy.h"
#include "app_util.h"
#include "asset_scan.h"
#include "borrowed_vulkan_host.h"
#include "file_dialog.h"
#include "imgui_layer.h"
#include "log_files.h"
#include "orbit_camera.h"
#include "relight_panel.h"
#include "shader_dir.h"
#include "system_info.h"
#include "window_icon.h"

#include "sabercat/device.h"
#include "sabercat/forward_pass.h"
#include "sabercat/lighting.h"
#include "sabercat/logger.h"
#include "sabercat/renderer.h"
#include "sabercat/scene_loader.h"
#include "sabercat/version.h"
#include "sabercat/vk_check.h"

using namespace sabercat;
using namespace sabercat::app;

namespace fs = std::filesystem;

// A file dropped onto the window (any format loadScene understands), picked up
// by the main loop. GLFW calls this from glfwPollEvents on the main thread.
std::string g_droppedPath;
void onDrop(GLFWwindow *, int count, const char ** paths)
{
    if (count > 0)
    {
        g_droppedPath = paths[0];
    }
}

// The last element of a scene path, for the scene list (the full path is the tooltip).
std::string sceneFileLabel(const std::string & path)
{
    const size_t lastSlash = path.find_last_of("/\\");
    return (lastSlash != std::string::npos) ? path.substr(lastSlash + 1) : path;
}

int main(int argc, char * argv[])
{
    // `--log <path>` writes the session log to <path>; `--no-log` writes none.
    // By default the log goes to %LOCALAPPDATA%\Sabercat\logs. It starts first
    // so the whole session is in it.
    std::string logPathFlag;
    bool noLogFlag = false;
    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "--log" && i + 1 < argc)
        {
            logPathFlag = argv[++i];
        }
        else if (std::string(argv[i]) == "--no-log")
        {
            noLogFlag = true;
        }
    }
    startSessionLog("viewer", logPathFlag, noLogFlag);

    std::cout << std::endl << "=== Sabercat Viewer Starting ===" << std::endl;
    std::cout << "CWD: " << fs::current_path().string() << std::endl;

    // Scan assets directory for available scenes
    std::cout << "Scanning for assets..." << std::endl;
    std::vector<std::string> availableScenes = scanAssetsDirectory();
    std::cout << "Asset scan complete. Found " << availableScenes.size() << " scene(s)." << std::endl
             << std::endl;

    // Load initial scene from command-line argument or leave empty for user selection
    std::string currentScenePath;
    bool borrowedDevice = false;
    bool startRelit = false; // `--relight`: start with relighting on
    bool validationFlag = false;   // `--validation`: validation in Release too
    bool noValidationFlag = false; // `--no-validation`: off even in Debug
    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "--borrowed")
        {
            borrowedDevice = true;
        }
        else if (std::string(argv[i]) == "--relight")
        {
            startRelit = true;
        }
        else if (std::string(argv[i]) == "--validation")
        {
            validationFlag = true;
        }
        else if (std::string(argv[i]) == "--no-validation")
        {
            noValidationFlag = true;
        }
        else if (std::string(argv[i]) == "--log" && i + 1 < argc)
        {
            ++i; // the path was taken above
        }
        else if (std::string(argv[i]) == "--no-log")
        {
            // taken above
        }
        else if (currentScenePath.empty())
        {
            currentScenePath = argv[i];
        }
    }
    if (currentScenePath.empty() && !availableScenes.empty())
    {
        // Auto-select first available scene if no argument given
        currentScenePath = availableScenes[0];
    }

    // App owns the window.
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    const std::string windowTitle = std::string("Sabercat Viewer ") + sabercat::kVersionString;
    GLFWwindow * window = glfwCreateWindow(1280, 720, windowTitle.c_str(), nullptr, nullptr);
    setWindowIconFromResource(window);
    glfwSetDropCallback(window, onDrop);

    // App tells the renderer which instance extensions its windowing layer needs.
    uint32_t glfwExtCount = 0;
    const char ** glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
    std::vector<const char *> instanceExtensions(glfwExts, glfwExts + glfwExtCount);

    const std::string shaderDir = resolveShaderDir();
    const bool validation = resolveValidation(validationFlag, noValidationFlag);
    Renderer renderer(1280, 720);
    BorrowedVulkanHost host;
    bool initialized = false;
    if (borrowedDevice)
    {
        try
        {
            host.create(instanceExtensions, /*presentation=*/true, validation);
            renderer.initializeBorrowed(host.context(), host.enabled, shaderDir);
            Logger::info("Viewer running on an APP-OWNED (borrowed) Vulkan device");
            initialized = true;
        }
        catch (const std::exception & e)
        {
            Logger::error(std::string("Borrowed-device setup failed: ") + e.what());
        }
    }
    else
    {
        // Renderer::initialize logs the reason it failed (e.g. no Vulkan 1.3
        // GPU) before returning false.
        initialized = renderer.initialize(instanceExtensions, validation, shaderDir);
    }
    if (!initialized)
    {
        Logger::error("Failed to initialize renderer; see the reason above");
        renderer.shutdown();
        host.destroy();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    // Help > About and the session log's system block: the device is up now.
    const SystemInfoFacts systemInfo = gatherSystemInfo("Sabercat Viewer", &renderer.sabercatDevice(),
                                                        validation);
    Logger::info(formatSystemInfo(systemInfo, Logger::logFilePath()));

    // Declared here (not inside the try) so the cleanup after it can tear
    // ImGui down regardless of whether the loop threw -- these must be
    // destroyed before renderer.shutdown() either way (see the comment
    // there), so an exception can't be allowed to skip past them.
    ImGuiLayer imgui;
    // The About logo's texture: created once ImGui is up, destroyed with ImGui.
    AboutLogo aboutLogo;

    int result = 0;
    try
    {
        // App creates the surface from the renderer's instance, then hands it
        // back. The app still owns it -- attachSurface's callback fires when
        // the renderer no longer needs it (during shutdown, before the
        // instance is destroyed), and the app destroys it there.
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

        // Load initial scene if available
        if (!currentScenePath.empty())
        {
            try
            {
                renderer.loadScene(currentScenePath.c_str());
            }
            catch (const std::exception & e)
            {
                Logger::error("Failed to load initial scene: " + std::string(e.what()));
            }
        }

        // Orbit camera: pivot at the origin, far enough back to frame a
        // scene near it -- the same starting point the old fixed camera
        // used, now adjustable (left-drag orbit, right-drag pan, scroll
        // zoom, Q/E roll).
        OrbitCamera orbitCamera;
        const OrbitCamera kDefaultCamera; // for the Reset button

        // Frame whatever loaded above. Scene scale varies by orders of
        // magnitude across formats, so a fixed default distance cannot work for
        // all of them -- see OrbitCamera::frame().
        const auto frameLoadedScene = [&renderer, &orbitCamera]()
        {
            const SceneBounds bounds = renderer.sceneBounds();
            if (!bounds.valid)
            {
                return;
            }
            orbitCamera.frame(bounds,
                              renderer.height(),
                              static_cast<float>(renderer.height()));
        };
        frameLoadedScene();

        size_t debugViewIndex = 0;
        bool aboutOpen = false;
        bool vWasDown = false;
        bool flipUpAxis = false;
        double lastFrameTime = glfwGetTime();

        // Scene list: full paths, most recent first. It starts with the scanned
        // assets; Browse, a typed path and a drop add to the top for the session.
        std::vector<std::string> scenePaths = availableScenes;
        int selectedSceneIndex = -1;
        if (!currentScenePath.empty())
        {
            selectedSceneIndex = static_cast<int>(addScenePathToList(scenePaths, currentScenePath));
        }
        std::string loadErrorMessage;
        std::string pathMessage; // why a typed path was not added

        // Max SH degree, full quality by default.
        int maxSHDegree = 3;

        // Relight window state; `--relight` starts it enabled.
        RelightPanelState relight(startRelit);

        // Loads any file loadScene understands (.ply .splat .spz .ksplat .sog
        // or a SOG meta.json, .gltf/.glb, a cloud-v1 manifest.json) and
        // re-frames the camera on it. Used by the scene list, the path box and
        // drag-and-drop.
        char pathBuffer[1024] = "";
        // Puts a path at the top of the scene list and selects it. Does not load.
        const auto selectScenePath = [&](const std::string & path)
        {
            selectedSceneIndex = static_cast<int>(addScenePathToList(scenePaths, path));
            pathMessage.clear();
        };
        // The Browse dialog runs on a worker thread, so the window keeps
        // rendering while it is open. Declared before the loop so it is joined
        // before the window is destroyed.
        AsyncFileDialog openDialog;
        const auto loadPath = [&](const std::string & path)
        {
            try
            {
                renderer.loadScene(path.c_str());
                loadErrorMessage.clear();
                currentScenePath = path;
                frameLoadedScene();
                // Clamp the cap to what the new scene actually has -- switching a
                // degree-3 .ply for a DC-only .splat must not leave SH3 selected
                // and unreachable.
                maxSHDegree = std::min(maxSHDegree, static_cast<int>(renderer.sceneSHDegree()));
                renderer.setMaxSHDegree(static_cast<uint32_t>(maxSHDegree));
            }
            catch (const std::exception & e)
            {
                loadErrorMessage = "Error: " + std::string(e.what());
            }
        };

        // Dynamic LOD, off by default -- see LODParams' comment
        // in forward_common.slangh/forward_pass.h for why. Defaults here are
        // starting points for the sliders below, not tuned values.
        bool lodEnabled = false;
        bool appearanceLodEnabled = false;
        LODParams lodParams{};
        lodParams.nearDistance = 10.0f;
        lodParams.farDistance = 100.0f;
        lodParams.maxDropDistance = 0.5f;
        lodParams.maxDropDensity = 0.5f;
        lodParams.densityReference = 4096.0f;
        // Default the appearance thresholds ABOVE the geometry ones
        // so SH quality visibly degrades before Gaussians start disappearing
        // -- a UI default that expresses the intended "graceful falloff"
        // ordering, not a coupling enforced anywhere in the shader.
        lodParams.appearanceMaxDropDistance = 0.8f;
        lodParams.appearanceMaxDropDensity = 0.8f;
        // Band fade over the last half of each degree step: no pop at the
        // threshold, while a Gaussian keeps its full top band for the first
        // half of the step.
        lodParams.appearanceFadeWidth = 0.5f;

        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();

            double now = glfwGetTime();
            float deltaTime = static_cast<float>(now - lastFrameTime);
            lastFrameTime = now;

            // Minimized (zero-size framebuffer): nothing to draw, so wait for an
            // event instead of rendering frames that cannot be shown.
            int framebufferWidth = 0;
            int framebufferHeight = 0;
            glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
            if (framebufferWidth == 0 || framebufferHeight == 0)
            {
                glfwWaitEventsTimeout(0.1);
                lastFrameTime = glfwGetTime();
                continue;
            }

            bool vIsDown = glfwGetKey(window, GLFW_KEY_V) == GLFW_PRESS;
            if (vIsDown && !vWasDown)
            {
                debugViewIndex = (debugViewIndex + 1) % kDebugViewCycle.size();
                renderer.setDebugView(kDebugViewCycle[debugViewIndex]);
            }
            vWasDown = vIsDown;

            // A drop waits until a Browse dialog has been answered.
            if (!g_droppedPath.empty() && !openDialog.pending())
            {
                const std::string dropped = std::move(g_droppedPath);
                g_droppedPath.clear();
                std::snprintf(pathBuffer, sizeof(pathBuffer), "%s", dropped.c_str());
                selectScenePath(dropped);
                loadPath(dropped);
            }

            // A file chosen in the Browse dialog is added to the list, selected,
            // shown in the box and loaded, like a drop. A cancel changes nothing.
            if (const auto picked = openDialog.poll(); picked && *picked)
            {
                std::snprintf(pathBuffer, sizeof(pathBuffer), "%s", (*picked)->c_str());
                selectScenePath(**picked);
                loadPath(**picked);
            }

            imgui.beginFrame();

            // Camera input reads ImGuiIO's WantCaptureMouse/Keyboard (set by
            // the NewFrame calls above), so it must run after them but
            // before the debug panel below is what makes those flags
            // meaningful for THIS frame's drag -- order matters.
            updateOrbitCamera(window, orbitCamera, deltaTime);
            renderer.setCamera(orbitCamera.toCamera(renderer.width(), renderer.height()));

            ImGui::Begin("Sabercat Debug");
            ImGui::Text("%.1f FPS (%.2f ms/frame)", ImGui::GetIO().Framerate,
                       1000.0f / ImGui::GetIO().Framerate);
            ImGui::Text("Gaussians: %u", renderer.gaussianCount());
            ImGui::Separator();

            // Row 1: the scene list and the one Load button. Load is off while a
            // Browse dialog is open.
            const bool dialogOpen = openDialog.pending();
            const bool haveSelection = selectedSceneIndex >= 0
                                       && selectedSceneIndex < static_cast<int>(scenePaths.size());
            if (scenePaths.empty())
            {
                ImGui::TextDisabled("No scene files in assets/. Browse for one or type a path below.");
            }
            // No visible labels on either row: the list and the path box fill the
            // width left after their button, so the button always stays in view.
            const ImGuiStyle & style = ImGui::GetStyle();
            const auto buttonWidth = [&style](const char * label)
            { return ImGui::CalcTextSize(label).x + style.FramePadding.x * 2.0f; };
            ImGui::SetNextItemWidth(-(buttonWidth("Load") + style.ItemSpacing.x));
            const std::string preview = haveSelection ? sceneFileLabel(scenePaths[selectedSceneIndex])
                                                      : std::string("(none)");
            if (ImGui::BeginCombo("##scene", preview.c_str()))
            {
                for (int i = 0; i < static_cast<int>(scenePaths.size()); ++i)
                {
                    const bool selected = (i == selectedSceneIndex);
                    if (ImGui::Selectable(sceneFileLabel(scenePaths[i]).c_str(), selected))
                    {
                        selectedSceneIndex = i;
                    }
                    if (ImGui::IsItemHovered())
                    {
                        ImGui::SetTooltip("%s", scenePaths[i].c_str());
                    }
                    if (selected)
                    {
                        ImGui::SetItemDefaultFocus();
                    }
                }
                ImGui::EndCombo();
            }
            ImGui::SameLine();
            ImGui::BeginDisabled(dialogOpen || !haveSelection);
            if (ImGui::Button("Load"))
            {
                loadPath(scenePaths[selectedSceneIndex]);
            }
            ImGui::EndDisabled();

            // Row 2: the path box. Enter adds the typed path to the list, selects
            // it and loads it; a missing file shows an error and loads nothing.
            // Browse loads the chosen file the same way. Both wait while a dialog
            // is open.
            ImGui::SetNextItemWidth(-(buttonWidth("Browse...") + style.ItemSpacing.x));
            const bool enterPressed = ImGui::InputTextWithHint("##path", "Path to a scene file", pathBuffer,
                                                               sizeof(pathBuffer),
                                                               ImGuiInputTextFlags_EnterReturnsTrue);
            if (ImGui::IsItemHovered())
            {
                ImGui::SetTooltip("Path to a scene file or a cloud-v1 manifest.json. Press Enter to load it, "
                                  "or use Browse... to pick a file.");
            }
            ImGui::SameLine();
            ImGui::BeginDisabled(dialogOpen);
            if (ImGui::Button("Browse..."))
            {
                openDialog.startOpen(window, sceneFileFilters(), "Open scene", "viewer.openScene");
            }
            ImGui::EndDisabled();
            if (enterPressed && !dialogOpen && pathBuffer[0] != '\0')
            {
                std::error_code ec;
                const bool exists = fs::exists(fs::u8path(pathBuffer), ec) && !ec;
                if (exists)
                {
                    selectScenePath(pathBuffer);
                    loadPath(pathBuffer);
                }
                else
                {
                    pathMessage = "No file or folder at that path.";
                }
            }
            if (!pathMessage.empty())
            {
                ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "%s", pathMessage.c_str());
            }
            ImGui::TextDisabled(".ply .splat .spz .ksplat .sog .gltf .glb, a cloud-v1 manifest.json"
                                " (or drop a file here)");

            // Load error display

            if (!loadErrorMessage.empty())
            {
                ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "%s", loadErrorMessage.c_str());
                if (ImGui::IsItemHovered() && ImGui::IsMouseClicked(0))
                {
                    loadErrorMessage.clear();
                }
            }
            ImGui::Separator();
            ImGui::Text("Debug view (or press V to cycle):");
            for (size_t i = 0; i < kDebugViewCycle.size(); ++i)
            {
                if (ImGui::RadioButton(debugViewName(kDebugViewCycle[i]),
                                       debugViewIndex == i))
                {
                    debugViewIndex = i;
                    renderer.setDebugView(kDebugViewCycle[debugViewIndex]);
                }
            }
            ImGui::Separator();
            if (ImGui::Button("About"))
            {
                aboutOpen = true;
            }
            ImGui::Separator();
            ImGui::Text("Camera: left-drag orbit, right-drag pan,");
            ImGui::Text("scroll zoom, Q/E roll");
            if (ImGui::Button("Reset camera"))
            {
                orbitCamera = kDefaultCamera;
            }
            ImGui::SameLine();
            if (ImGui::Button("Frame scene"))
            {
                frameLoadedScene();
            }
            if (ImGui::IsItemHovered())
            {
                ImGui::SetTooltip("Move the camera back far enough to see the whole\n"
                                  "scene. Scenes differ hugely in scale -- the .spz\n"
                                  "samples are ~500 units across, a reference .ply a\n"
                                  "few -- so 'Reset camera' returns to a fixed default\n"
                                  "that only suits small scenes.");
            }
            {
                const SceneBounds bounds = renderer.sceneBounds();
                if (bounds.valid)
                {
                    ImGui::Text("Focus radius %.2f / full %.1f (camera dist %.1f)",
                                bounds.focusRadius, bounds.radius(), orbitCamera.distance);
                    if (ImGui::IsItemHovered())
                    {
                        ImGui::SetTooltip("Framing targets the 90th-percentile radius, not the\n"
                                          "full bounds: outdoor captures carry a distant\n"
                                          "background shell that would otherwise push the\n"
                                          "camera far enough out to make the subject a speck.");
                    }
                }
            }
            ImGui::Separator();
            if (ImGui::Checkbox("Flip up axis", &flipUpAxis))
            {
                renderer.setFlipUpAxis(flipUpAxis);
            }
            if (ImGui::IsItemHovered())
            {
                ImGui::SetTooltip("For scenes authored Y-down (common COLMAP/\n"
                                  "INRIA 3DGS convention) that render upside down.");
            }
            // A property of the loaded scene (Renderer holds it), so
            // read back every frame rather than mirrored in viewer state --
            // a new scene shows its own value without any reset logic here.
            bool antialiased = renderer.sceneAntialiased();
            if (ImGui::Checkbox("Antialiased (opacity compensation)", &antialiased))
            {
                renderer.setAntialiased(antialiased);
            }
            if (ImGui::IsItemHovered())
            {
                ImGui::SetTooltip("Scales each Gaussian's opacity by its true/dilated\n"
                                  "footprint ratio (mip-splatting / gsplat antialiased).\n"
                                  "Set from the file for .spz; .ply/.splat default off\n"
                                  "(vanilla 3DGS trains without it). Turn on for a\n"
                                  "mip-splatting .ply that renders oversaturated.");
            }
            ImGui::Separator();
            // Degrees above what the loaded file carries are disabled rather
            // than hidden: a .splat is DC-only, and a silently dead SH slider
            // reads as a renderer bug instead of a property of the format.
            const int sceneSHDegree = static_cast<int>(renderer.sceneSHDegree());
            ImGui::Text("Max SH degree (scene carries SH%d):", sceneSHDegree);
            for (int degree = 0; degree <= 3; ++degree)
            {
                char label[16];
                std::snprintf(label, sizeof(label), "SH%d", degree);
                const bool available = degree <= sceneSHDegree;
                ImGui::BeginDisabled(!available);
                if (ImGui::RadioButton(label, maxSHDegree == degree) && available)
                {
                    maxSHDegree = degree;
                    renderer.setMaxSHDegree(static_cast<uint32_t>(maxSHDegree));
                }
                ImGui::EndDisabled();
                if (degree < 3)
                {
                    ImGui::SameLine();
                }
            }
            if (ImGui::IsItemHovered())
            {
                ImGui::SetTooltip("SH0 = flat color (no view-dependent shading).\n"
                                  "SH3 = full quality. Lower degrees read fewer\n"
                                  "coefficients per Gaussian. Degrees above what the\n"
                                  "loaded file stores are disabled -- .splat is DC-only.");
            }
            if (sceneSHDegree == 0)
            {
                ImGui::TextDisabled("(DC-only scene: SH degree and appearance LOD are no-ops)");
            }
            ImGui::Separator();
            ImGui::Text("Tile instances: %u (of %u Gaussians)", renderer.lastInstanceCount(),
                       renderer.gaussianCount());
            // Screen-space size cull. Off by default (0): an unconditional 2px
            // threshold was measured culling >95% of racoonfamily.spz, leaving
            // only huge background Gaussians and rendering the scene as a
            // featureless ball. Kept as a knob because it IS a real perf win
            // when a scene can afford it.
            bool lodChanged = ImGui::SliderFloat("Min screen radius (px)",
                                                 &lodParams.minScreenRadius, 0.0f, 8.0f);
            if (ImGui::IsItemHovered())
            {
                ImGui::SetTooltip("Cull Gaussians whose projected footprint is smaller\n"
                                  "than this. 0 = off (default).\n\n"
                                  "Values at or below 1.5px can never fire: the isotropic\n"
                                  "covariance dilation floors every footprint there.\n"
                                  "Raising it drops fine detail before it drops anything\n"
                                  "you would call redundant -- most Gaussians in a real\n"
                                  "scene are sub-pixel at normal viewing distances.");
            }
            lodChanged |= ImGui::Checkbox("Drop Gaussians (geometry LOD)", &lodEnabled);
            if (ImGui::IsItemHovered())
            {
                ImGui::SetTooltip("Distance + tile-density hybrid reduction.\n"
                                  "Viewer-only -- never affects a differentiable pass.");
            }
            lodChanged |= ImGui::Checkbox("Reduce SH degree (appearance LOD)",
                                          &appearanceLodEnabled);
            if (ImGui::IsItemHovered())
            {
                ImGui::SetTooltip("Steps SH degree down under the same distance/density\n"
                                  "pressure. Enable alongside geometry LOD for a\n"
                                  "graceful falloff: quality degrades before splats vanish.");
            }
            if (lodEnabled || appearanceLodEnabled)
            {
                // Shared pressure inputs -- both axes read these.
                lodChanged |= ImGui::SliderFloat("LOD near distance", &lodParams.nearDistance,
                                                 0.0f, 200.0f);
                lodChanged |= ImGui::SliderFloat("LOD far distance", &lodParams.farDistance, 0.0f,
                                                 500.0f);
                lodChanged |= ImGui::SliderFloat("Density reference",
                                                 &lodParams.densityReference, 1.0f, 16384.0f);
            }
            if (lodEnabled)
            {
                lodChanged |= ImGui::SliderFloat("Max drop (distance)",
                                                 &lodParams.maxDropDistance, 0.0f, 1.0f);
                lodChanged |= ImGui::SliderFloat("Max drop (density)",
                                                 &lodParams.maxDropDensity, 0.0f, 1.0f);
            }
            if (appearanceLodEnabled)
            {
                lodChanged |= ImGui::SliderFloat("SH falloff (distance)",
                                                 &lodParams.appearanceMaxDropDistance, 0.0f, 1.0f);
                lodChanged |= ImGui::SliderFloat("SH falloff (density)",
                                                 &lodParams.appearanceMaxDropDensity, 0.0f, 1.0f);
                lodChanged |= ImGui::SliderFloat("SH band fade width",
                                                 &lodParams.appearanceFadeWidth, 0.0f, 1.0f);
                if (ImGui::IsItemHovered())
                {
                    ImGui::SetTooltip("Fraction of each degree step over which the top\n"
                                      "band fades out before the switch, instead of\n"
                                      "popping off at it. 0 = hard switches, 1 = fully\n"
                                      "continuous. The SHDegree debug view shows the result.");
                }
            }
            if (lodChanged)
            {
                lodParams.enabled = lodEnabled ? 1u : 0u;
                lodParams.appearanceEnabled = appearanceLodEnabled ? 1u : 0u;
                renderer.setLODParams(lodParams);
            }
            ImGui::End();

            drawRelightPanel(renderer, relight);
            drawAboutWindow(aboutOpen, systemInfo, aboutLogo.texture());

            ImGui::Render();

            renderer.forward();
            renderer.present();
        }
    }
    catch (const std::exception & e)
    {
        Logger::error(std::string("Viewer FAILED: ") + e.what());
        result = 1;
    }

    // The logo's texture is registered with ImGui's backend, so it goes first.
    aboutLogo.destroy();

    // Dear ImGui's Vulkan backend owns real device resources (font texture,
    // pipeline) that must be destroyed before Device::shutdown() frees the
    // VmaAllocator/VkDevice they depend on -- the same ordering hazard fixed
    // in Renderer::shutdown() for its own GPU-resource-holding members.
    imgui.shutdown(renderer);

    renderer.shutdown();
    host.destroy(); // no-op unless --borrowed; after every borrower
    glfwDestroyWindow(window);
    glfwTerminate();
    return result;
}
