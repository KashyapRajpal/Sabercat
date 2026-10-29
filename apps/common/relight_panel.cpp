// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "relight_panel.h"

#include "sabercat/device.h"

#include <imgui.h>

#include <glm/glm.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

namespace sabercat::app
{

void drawRelightPanel(Renderer & renderer, RelightPanelState & state)
{
    // Named references to the app-owned state, so the body reads as it did
    // when these were function-local statics.
    bool & enabled = state.enabled;
    int & output = state.output;
    float & exposure = state.exposure;
    int & environmentPreset = state.environmentPreset;
    float & environmentIntensity = state.environmentIntensity;
    int & shadowMode = state.shadowMode;
    int & shadowMapResolution = state.shadowMapResolution;
    int & raySamples = state.raySamples;
    bool & progressiveShadows = state.progressiveShadows;
    int & rtQuality = state.rtQuality;
    int & visibilityDebug = state.visibilityDebug;
    int & bakeDirections = state.bakeDirections;
    uint64_t & bakedGeneration = state.bakedGeneration;
    std::string & bakeStatus = state.bakeStatus;
    std::vector<Light> & lights = state.lights;
    bool & placed = state.placed;

    const SceneBounds bounds = renderer.sceneBounds();
    const glm::vec3 center = bounds.valid ? bounds.focusCenter : glm::vec3(0.0f);
    const float radius = bounds.valid ? bounds.focusRadius : 1.0f;
    const auto placeDefaults = [&]
    {
        lights.clear();
        Light key; // a warm key light from above-front
        key.type = LightType::Directional;
        key.direction = glm::normalize(glm::vec3(-0.4f, -1.0f, -0.6f));
        key.color = glm::vec3(1.0f, 0.95f, 0.85f);
        key.intensity = 2.5f;
        lights.push_back(key);
        Light area; // a cool area fill to the side
        area.type = LightType::Rect;
        area.position = center + radius * glm::vec3(1.2f, 0.6f, 0.8f);
        area.halfWidth = area.halfHeight = 0.4f * radius;
        area.color = glm::vec3(0.6f, 0.75f, 1.0f);
        area.intensity = 3.0f;
        lights.push_back(area);
        placed = bounds.valid;
    };
    if (!placed)
    {
        placeDefaults();
    }

    ImGui::Begin("Relight");
    ImGui::Checkbox("Relight (G-buffer + deferred lighting)", &enabled);
    ImGui::BeginDisabled(!enabled);
    const char * outputs[] = {"Lit", "Normal", "Base color", "Roughness", "Metallic",
                              "Ambient occlusion", "Visibility (light 0)", "Depth"};
    ImGui::Combo("Show", &output, outputs, IM_ARRAYSIZE(outputs));
    ImGui::SliderFloat("Exposure", &exposure, 0.05f, 8.0f, "%.2f", ImGuiSliderFlags_Logarithmic);
    const char * presets[] = {"None", "Uniform", "Sky"};
    ImGui::Combo("Environment", &environmentPreset, presets, IM_ARRAYSIZE(presets));
    ImGui::SliderFloat("Environment intensity", &environmentIntensity, 0.0f, 4.0f);

    const bool baked = renderer.storageGeneration() == bakedGeneration;
    const char * shadowModes[] = {"Off", "PCSS (soft, fast)", "Ray traced (reference)",
                                  baked ? "Baked SH (real time)" : "Baked SH (bake first)"};
    ImGui::Combo("Shadows", &shadowMode, shadowModes, IM_ARRAYSIZE(shadowModes));
    if (shadowMode == 1)
    {
        ImGui::SliderInt("Shadow map", &shadowMapResolution, 256, 4096);
    }
    if (shadowMode == 2)
    {
        ImGui::SliderInt("Rays per axis", &raySamples, 1, 16);
        ImGui::Checkbox("Progressive (1 ray/frame, exact once converged)", &progressiveShadows);
        const char * qualities[] = {"Exact", "Balanced (half-res, early stop)", "Fast (+ skip faint splats)"};
        ImGui::Combo("RT quality", &rtQuality, qualities, IM_ARRAYSIZE(qualities));
        if (rtQuality != 0)
        {
            ImGui::TextDisabled("Approximate: check with Visibility debug -> |error|.");
        }
    }
    if (enabled && shadowMode != 0)
    {
        const ShadowFrameStats & stats = renderer.shadowFrameStats();
        if (stats.reusedFrame != 0)
        {
            ImGui::TextDisabled("Shadows: unchanged, reused from the last frame");
        }
        else if (stats.pendingLights + stats.convergedLights > 0)
        {
            ImGui::TextDisabled("Shadows: %u light(s) converged, %u accumulating", stats.convergedLights,
                                stats.pendingLights);
        }
    }

    // Baked per-Gaussian visibility and the teacher samples a trainer learns from.
    ImGui::SeparatorText("Baked visibility / teacher samples");
    const bool rayQuery = renderer.deviceFeatures().rayQuery;
    ImGui::BeginDisabled(!rayQuery);
    ImGui::SliderInt("Bake directions", &bakeDirections, 16, 1024, "%d", ImGuiSliderFlags_Logarithmic);
    if (ImGui::Button("Bake visibility"))
    {
        try
        {
            VisibilityBakeSettings bake;
            bake.directions = static_cast<uint32_t>(bakeDirections);
            const auto start = std::chrono::steady_clock::now();
            renderer.bakeVisibility(bake);
            const double seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            bakedGeneration = renderer.storageGeneration();
            char text[128];
            std::snprintf(text, sizeof(text), "Baked %u directions per Gaussian in %.2f s", bake.directions,
                          seconds);
            bakeStatus = text;
        }
        catch (const std::exception & e)
        {
            bakeStatus = std::string("Bake failed: ") + e.what();
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("Teacher samples"))
    {
        try
        {
            const uint32_t count = renderer.generateTeacherSamples();
            bakeStatus = std::to_string(count) + " teacher samples (" +
                           std::to_string(renderer.teacherSmoothSampleCount()) + " smooth, " +
                           std::to_string(count - renderer.teacherSmoothSampleCount()) + " edge)";
        }
        catch (const std::exception & e)
        {
            bakeStatus = std::string("Teacher samples failed: ") + e.what();
        }
    }
    const char * debugViews[] = {"Off", "|error| vs ray-traced reference", "Teacher sample locations"};
    ImGui::Combo("Visibility debug", &visibilityDebug, debugViews, IM_ARRAYSIZE(debugViews));
    ImGui::EndDisabled();
    if (!rayQuery)
    {
        ImGui::TextDisabled("Needs ray query, which this GPU does not have.");
    }
    else if (visibilityDebug != 0)
    {
        ImGui::TextDisabled("Shown through Show: Visibility (light 0).");
    }
    if (!bakeStatus.empty())
    {
        ImGui::TextWrapped("%s", bakeStatus.c_str());
    }

    const char * types[] = {"Directional", "Point", "Rect"};
    int removeIndex = -1;
    for (size_t i = 0; i < lights.size(); ++i)
    {
        Light & l = lights[i];
        ImGui::PushID(static_cast<int>(i));
        ImGui::Separator();
        int type = static_cast<int>(l.type);
        if (ImGui::Combo("Type", &type, types, IM_ARRAYSIZE(types)))
        {
            l.type = static_cast<LightType>(type);
            if (l.type != LightType::Directional && glm::length(l.position - center) < 1e-6f)
            {
                l.position = center + radius * glm::vec3(0.0f, 1.2f, 1.0f);
            }
        }
        ImGui::ColorEdit3("Color", &l.color.x);
        ImGui::SliderFloat("Intensity", &l.intensity, 0.0f, 20.0f, "%.2f",
                           ImGuiSliderFlags_Logarithmic);
        if (l.type == LightType::Directional)
        {
            if (ImGui::SliderFloat3("Direction", &l.direction.x, -1.0f, 1.0f)
                && glm::length(l.direction) > 1e-4f)
            {
                l.direction = glm::normalize(l.direction);
            }
        }
        else
        {
            ImGui::DragFloat3("Position", &l.position.x, 0.01f * radius);
        }
        if (l.type == LightType::Rect)
        {
            ImGui::DragFloat("Half width", &l.halfWidth, 0.005f * radius, 0.001f, 10.0f * radius);
            ImGui::DragFloat("Half height", &l.halfHeight, 0.005f * radius, 0.001f, 10.0f * radius);
            // Always aimed at the scene center: emits along cross(axisX, axisY).
            const glm::vec3 toScene = center - l.position;
            if (glm::length(toScene) > 1e-6f)
            {
                const glm::vec3 n = glm::normalize(toScene);
                const glm::vec3 helper = std::fabs(n.y) < 0.99f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
                l.axisX = glm::normalize(glm::cross(helper, n));
                l.axisY = glm::cross(n, l.axisX); // cross(axisX, axisY) == n
            }
        }
        bool shadows = l.castsShadows != 0;
        if (ImGui::Checkbox("Casts shadows", &shadows))
        {
            l.castsShadows = shadows ? 1u : 0u;
        }
        if (ImGui::Button("Remove"))
        {
            removeIndex = static_cast<int>(i);
        }
        ImGui::PopID();
    }
    if (removeIndex >= 0)
    {
        lights.erase(lights.begin() + removeIndex);
    }
    ImGui::Separator();
    ImGui::BeginDisabled(lights.size() >= kMaxLights);
    if (ImGui::Button("Add point light"))
    {
        Light p;
        p.type = LightType::Point;
        p.position = center + radius * glm::vec3(0.0f, 1.2f, 1.0f);
        p.intensity = 2.0f * radius * radius;
        lights.push_back(p);
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Reset to scene"))
    {
        placeDefaults();
    }
    ImGui::EndDisabled();
    ImGui::End();

    renderer.setLightingEnabled(enabled);
    renderer.setLights(lights);
    EnvironmentLight environment;
    if (environmentPreset == 1)
    {
        environment = EnvironmentLight::uniform(glm::vec3(1.0f));
    }
    else if (environmentPreset == 2)
    {
        static const EnvironmentLight sky = EnvironmentLight::sky(
            glm::vec3(0.45f, 0.6f, 1.0f), glm::vec3(0.9f, 0.9f, 0.85f), glm::vec3(0.25f, 0.2f, 0.15f),
            glm::vec3(0.0f, 1.0f, 0.0f));
        environment = sky;
    }
    environment.intensity = environmentIntensity;
    renderer.setEnvironment(environment);
    LightingSettings settings = renderer.lightingSettings();
    settings.output = static_cast<LightingOutput>(output);
    settings.exposure = exposure;
    settings.depthScale = bounds.valid ? 0.5f / std::max(radius, 1e-3f) : 0.1f;
    renderer.setLightingSettings(settings);

    ShadowSettings shadows = renderer.shadowSettings();
    // Ray traced needs ray query; fall back to PCSS on devices without it.
    // Baked SH needs this scene's bake; it shows no shadows until then.
    shadows.mode = shadowMode == 2 && !renderer.deviceFeatures().rayQuery
                       ? ShadowMode::PCSS
                       : static_cast<ShadowMode>(shadowMode);
    if (shadows.mode == ShadowMode::BakedSH && renderer.storageGeneration() != bakedGeneration)
    {
        shadows.mode = ShadowMode::None;
    }
    shadows.debug = renderer.deviceFeatures().rayQuery ? static_cast<VisibilityDebug>(visibilityDebug)
                                                       : VisibilityDebug::Off;
    shadows.mapResolution = static_cast<uint32_t>(shadowMapResolution);
    shadows.raySamplesPerAxis = static_cast<uint32_t>(raySamples);
    shadows.progressiveSamplesPerFrame = progressiveShadows ? 1u : 0u;
    // Measured on a 3.5M-Gaussian scene against the exact trace: balanced
    // ~2.7x faster, mean |dV| ~0.036; fast ~7-9x, ~0.085 (docs/architecture.md).
    shadows.rayMinTransmittance = rtQuality == 0 ? 1e-4f : 0.01f;
    shadows.rayMinAlpha = rtQuality == 2 ? 0.02f : 1.0f / 255.0f;
    shadows.rayDownsample = rtQuality == 0 ? 1u : 2u;
    renderer.setShadowSettings(shadows);
}

} // namespace sabercat::app
