// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "studio_state.h"

#include "app_util.h"

#include "sabercat/gaussian_cloud.h"
#include "sabercat/gltf_writer.h"
#include "sabercat/logger.h"
#include "sabercat/ply_writer.h"
#include "sabercat/renderer.h"
#include "sabercat/scene_loader.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <set>
#include <stdexcept>

namespace sabercat::studio
{

namespace
{

namespace fs = std::filesystem;
using json = nlohmann::json;

// The app-state block's own version (the "version" key). Files from a newer
// build still load their known keys; they are logged once.
constexpr int kAppStateVersion = 1;

// The ranges the panels offer (apps/common/relight_panel.cpp): what the app-state
// block may hold. Values outside them are clamped on load.
constexpr int kLightingOutputCount = 8; // the "Show" list
constexpr float kExposureMin = 0.05f;   // "Exposure" slider
constexpr float kExposureMax = 8.0f;
constexpr float kEnvironmentIntensityMax = 4.0f;
constexpr int kShadowMapMin = 256;
constexpr int kShadowMapMax = 4096;
constexpr int kRaySamplesMin = 1;
constexpr int kRaySamplesMax = 16;

// Layer names are UTF-8 (ProjectLayer::name); fs::path::stem() gives the
// native encoding, so convert back explicitly.
std::string utf8(const fs::path & path)
{
    const std::u8string text = path.u8string();
    return std::string(reinterpret_cast<const char *>(text.data()), text.size());
}

// Host arrays of a cloud at the canonical strides, as setGaussians() takes them.
GaussianHostInput hostInputFor(const GaussianCloud & cloud)
{
    GaussianHostInput input;
    input.count = static_cast<uint32_t>(cloud.count());
    input.means = reinterpret_cast<const float *>(cloud.positions().data());
    input.scales = reinterpret_cast<const float *>(cloud.scales().data());
    input.rotations = reinterpret_cast<const float *>(cloud.rotations().data());
    input.opacities = cloud.opacities().data();
    input.shCoefficients = reinterpret_cast<const float *>(cloud.shCoefficients().data());
    input.shDegree = cloud.shDegree();
    input.antialiased = cloud.antialiased();
    if (!cloud.normals().empty())
    {
        input.normals = reinterpret_cast<const float *>(cloud.normals().data());
    }
    if (!cloud.baseColorRoughness().empty())
    {
        input.baseColorRoughness = reinterpret_cast<const float *>(cloud.baseColorRoughness().data());
    }
    if (!cloud.metallic().empty())
    {
        input.metallic = cloud.metallic().data();
    }
    if (!cloud.visibility().empty())
    {
        input.visibility = cloud.visibility().data();
    }
    return input;
}

// --- App-state reads ------------------------------------------------------
//
// A key that is absent takes its default silently (older files lack newer
// keys). A value of the wrong type takes the default, and a value outside the
// range the panel offers is clamped; each such case is one log line. Keys the
// build does not know are ignored.

void reportAppField(const char * key, const std::string & problem)
{
    Logger::warning(std::string("Studio: app state \"") + key + "\" " + problem);
}

// The value at obj[key], or nullptr when obj is not an object or lacks the key.
const json * findField(const json & obj, const char * key)
{
    if (!obj.is_object())
    {
        return nullptr;
    }
    const auto it = obj.find(key);
    return it == obj.end() ? nullptr : &(*it);
}

// obj[key] as an object, or an empty object when it is absent or not an object.
json objectField(const json & obj, const char * key)
{
    const json * value = findField(obj, key);
    if (value == nullptr)
    {
        return json::object();
    }
    if (!value->is_object())
    {
        reportAppField(key, "is not an object; using the defaults");
        return json::object();
    }
    return *value;
}

bool readBool(const json & obj, const char * key, bool fallback)
{
    const json * value = findField(obj, key);
    if (value == nullptr)
    {
        return fallback;
    }
    if (!value->is_boolean())
    {
        reportAppField(key, "is not true or false; using the default");
        return fallback;
    }
    return value->get<bool>();
}

// An integer in [lo, hi]: out-of-range values are clamped to the nearest end.
int readInt(const json & obj, const char * key, int fallback, int lo, int hi)
{
    const json * value = findField(obj, key);
    if (value == nullptr)
    {
        return fallback;
    }
    if (!value->is_number_integer())
    {
        reportAppField(key, "is not an integer; using the default");
        return fallback;
    }
    const double number = value->get<double>();
    if (number < lo || number > hi)
    {
        const int clamped = number < lo ? lo : hi;
        reportAppField(key, "is out of range; clamped to " + std::to_string(clamped));
        return clamped;
    }
    return static_cast<int>(number);
}

// A finite number in [lo, hi]: out-of-range values are clamped to the nearest end.
float readFloat(const json & obj, const char * key, float fallback, float lo, float hi)
{
    const json * value = findField(obj, key);
    if (value == nullptr)
    {
        return fallback;
    }
    if (!value->is_number())
    {
        reportAppField(key, "is not a number; using the default");
        return fallback;
    }
    const double number = value->get<double>();
    if (!std::isfinite(number))
    {
        reportAppField(key, "is not finite; using the default");
        return fallback;
    }
    if (number < lo || number > hi)
    {
        const float clamped = number < lo ? lo : hi;
        reportAppField(key, "is out of range; clamped to " + std::to_string(clamped));
        return clamped;
    }
    return static_cast<float>(number);
}

// Reads obj[key] as N finite numbers into `out`. Returns false, leaving `out`
// untouched, when the key is absent or the value is malformed (logged).
template <size_t N>
bool readFloats(const json & obj, const char * key, float (&out)[N])
{
    const json * value = findField(obj, key);
    if (value == nullptr)
    {
        return false;
    }
    if (!value->is_array() || value->size() != N)
    {
        reportAppField(key, "is not an array of " + std::to_string(N) + " numbers; using the default");
        return false;
    }
    float values[N];
    for (size_t i = 0; i < N; ++i)
    {
        const json & element = (*value)[i];
        if (!element.is_number() || !std::isfinite(element.get<double>()))
        {
            reportAppField(key, "holds a value that is not a finite number; using the default");
            return false;
        }
        values[i] = element.get<float>();
    }
    std::copy(values, values + N, out);
    return true;
}

} // namespace

std::string nextLayerId(const Project & project)
{
    // Only plain decimal ids count: "-1" or "7abc" would otherwise parse as
    // numbers. A non-numeric id cannot collide with the numeric ids handed out.
    std::set<uint64_t> used;
    for (const ProjectLayer & layer : project.layers)
    {
        const bool digitsOnly = !layer.id.empty()
                                && std::all_of(layer.id.begin(), layer.id.end(),
                                               [](unsigned char c) { return std::isdigit(c) != 0; });
        if (!digitsOnly)
        {
            continue;
        }
        try
        {
            used.insert(std::stoull(layer.id));
        }
        catch (const std::out_of_range &)
        {
            // Too large for 64 bits: no id we hand out can equal it.
        }
    }
    if (used.empty())
    {
        return "0";
    }
    if (*used.rbegin() < std::numeric_limits<uint64_t>::max())
    {
        return std::to_string(*used.rbegin() + 1);
    }
    // The largest id is at the top of the range, so "one more" overflows. Take
    // the smallest id not in use instead.
    uint64_t candidate = 0;
    for (const uint64_t id : used)
    {
        if (id == candidate)
        {
            ++candidate;
        }
        else if (id > candidate)
        {
            break;
        }
    }
    return std::to_string(candidate);
}

void addLayerFromFile(StudioState & state, Device & device, const std::string & path)
{
    std::shared_ptr<GaussianCloud> cloud = sabercat::loadScene(device, path);

    ProjectLayer layer;
    layer.cloud = std::move(cloud);
    layer.id = nextLayerId(state.project);
    layer.name = utf8(fs::u8path(path).stem());
    state.project.layers.push_back(std::move(layer));
    state.selectedLayer = static_cast<int>(state.project.layers.size()) - 1;
    state.dirty = true;
    state.sceneDirty = true;
    Logger::info("Studio: added layer " + state.project.layers.back().id + " from " + path);
}

bool removeSelectedLayer(StudioState & state)
{
    if (state.selectedLayer < 0 || state.selectedLayer >= static_cast<int>(state.project.layers.size()))
    {
        return false;
    }
    if (state.project.layers[state.selectedLayer].locked)
    {
        return false;
    }
    state.project.layers.erase(state.project.layers.begin() + state.selectedLayer);
    const int remaining = static_cast<int>(state.project.layers.size());
    state.selectedLayer = std::min(state.selectedLayer, remaining - 1);
    state.dirty = true;
    state.sceneDirty = true;
    return true;
}

bool moveSelectedLayer(StudioState & state, int delta)
{
    const int count = static_cast<int>(state.project.layers.size());
    const int from = state.selectedLayer;
    const int to = from + delta;
    if (delta != -1 && delta != 1)
    {
        return false;
    }
    if (from < 0 || from >= count || to < 0 || to >= count)
    {
        return false;
    }
    // A swap moves both layers, so a locked layer on either side blocks it.
    if (state.project.layers[from].locked || state.project.layers[to].locked)
    {
        return false;
    }
    std::swap(state.project.layers[from], state.project.layers[to]);
    state.selectedLayer = to;
    state.dirty = true;
    state.sceneDirty = true;
    return true;
}

namespace
{

// The layer at `index`, or nullptr when the index is out of range.
ProjectLayer * layerAt(StudioState & state, int index)
{
    if (index < 0 || index >= static_cast<int>(state.project.layers.size()))
    {
        return nullptr;
    }
    return &state.project.layers[index];
}

} // namespace

bool setLayerVisible(StudioState & state, int index, bool visible)
{
    ProjectLayer * layer = layerAt(state, index);
    if (layer == nullptr)
    {
        return false;
    }
    if (layer->visible != visible)
    {
        layer->visible = visible;
        state.dirty = true;
        state.sceneDirty = true;
    }
    return true;
}

bool setLayerLocked(StudioState & state, int index, bool locked)
{
    ProjectLayer * layer = layerAt(state, index);
    if (layer == nullptr)
    {
        return false;
    }
    if (layer->locked != locked)
    {
        layer->locked = locked;
        state.dirty = true;
    }
    return true;
}

bool renameLayer(StudioState & state, int index, const std::string & name)
{
    ProjectLayer * layer = layerAt(state, index);
    if (layer == nullptr || layer->locked)
    {
        return false;
    }
    layer->name = name;
    state.dirty = true;
    return true;
}

bool setLayerTransform(StudioState & state, int index, const LayerTransform & transform)
{
    ProjectLayer * layer = layerAt(state, index);
    if (layer == nullptr || layer->locked)
    {
        return false;
    }
    layer->transform = transform;
    state.dirty = true;
    state.sceneDirty = true;
    return true;
}

std::vector<std::pair<const GaussianCloud *, LayerTransform>> visibleLayers(const StudioState & state)
{
    std::vector<std::pair<const GaussianCloud *, LayerTransform>> visible;
    for (const ProjectLayer & layer : state.project.layers)
    {
        if (layer.visible && layer.cloud)
        {
            visible.emplace_back(layer.cloud.get(), layer.transform);
        }
    }
    return visible;
}

void rebuildScene(StudioState & state, Renderer & renderer)
{
    // Cleared first: a failing rebuild is reported once, not retried each frame.
    state.sceneDirty = false;
    const auto start = std::chrono::steady_clock::now();

    const auto visible = visibleLayers(state);
    if (visible.empty())
    {
        if (state.rendererHasScene)
        {
            // Transparent stand-in: opacity 0 renders nothing, and the renderer
            // has no call to drop a scene.
            static const float kZeroSh[16 * 4] = {};
            static const float kIdentityRotation[4] = {0.0f, 0.0f, 0.0f, 1.0f};
            static const float kUnitScale[4] = {1e-3f, 1e-3f, 1e-3f, 0.0f};
            static const float kPosition[4] = {};
            static const float kOpacity[1] = {0.0f};
            GaussianHostInput placeholder;
            placeholder.count = 1;
            placeholder.means = kPosition;
            placeholder.scales = kUnitScale;
            placeholder.rotations = kIdentityRotation;
            placeholder.opacities = kOpacity;
            placeholder.shCoefficients = kZeroSh;
            placeholder.shDegree = 0;
            renderer.setGaussians(placeholder);
        }
    }
    else
    {
        const std::shared_ptr<GaussianCloud> merged =
            mergeLayers(renderer.sabercatDevice(), visible);
        renderer.setGaussians(hostInputFor(*merged));
        state.rendererHasScene = true;
    }

    const auto end = std::chrono::steady_clock::now();
    state.lastRebuildMs = std::chrono::duration<double, std::milli>(end - start).count();
}

void newProject(StudioState & state)
{
    state.project = Project{};
    state.projectPath.clear();
    state.selectedLayer = -1;
    state.dirty = false;
    state.sceneDirty = true;
    state.camera = app::OrbitCamera{};
    state.shDegree = 3;
    state.debugViewIndex = 0;
    state.relight = app::RelightPanelState(false);
}

void openProject(StudioState & state, Renderer & renderer, const std::string & path)
{
    Project loaded = ProjectIO::load(renderer.sabercatDevice(), path);
    state.project = std::move(loaded);
    state.projectPath = path;
    state.selectedLayer = state.project.layers.empty() ? -1 : 0;
    state.dirty = false;
    state.sceneDirty = true;
    applyAppStateJson(state, state.project.appStateJson);
    resetRelightForNewScene(state);
    Logger::info("Studio: opened " + path + " (" + std::to_string(state.project.layers.size())
                 + " layers)");
}

void saveProject(StudioState & state, const std::string & path)
{
    state.project.appStateJson = buildAppStateJson(state);
    ProjectIO::save(state.project, path);
    state.projectPath = path;
    state.dirty = false;
    Logger::info("Studio: saved " + path);
}

void exportPly(const StudioState & state, Device & device, const std::string & path)
{
    const auto visible = visibleLayers(state);
    if (visible.empty())
    {
        throw std::runtime_error("No visible layers to export");
    }
    const std::shared_ptr<GaussianCloud> merged = mergeLayers(device, visible);
    PLYWriter::write(*merged, path);
    Logger::info("Studio: exported PLY " + path);
}

void exportGlb(const StudioState & state, const std::string & path)
{
    std::vector<GLTFExportLayer> layers;
    for (const ProjectLayer & layer : state.project.layers)
    {
        if (layer.visible && layer.cloud)
        {
            layers.push_back(GLTFExportLayer{layer.cloud.get(), layer.transform, layer.name});
        }
    }
    if (layers.empty())
    {
        throw std::runtime_error("No visible layers to export");
    }
    GLTFWriter::writeGLB(layers, path);
    Logger::info("Studio: exported GLB " + path);
}

std::string buildAppStateJson(const StudioState & state)
{
    const app::OrbitCamera & cam = state.camera;
    const app::RelightPanelState & rl = state.relight;

    json camera;
    camera["pivot"] = json::array({cam.pivot.x, cam.pivot.y, cam.pivot.z});
    camera["distance"] = cam.distance;
    camera["orientation"] = json::array(
        {cam.orientation.w, cam.orientation.x, cam.orientation.y, cam.orientation.z});

    json relight;
    relight["enabled"] = rl.enabled;
    relight["output"] = rl.output;
    relight["exposure"] = rl.exposure;
    relight["environmentPreset"] = rl.environmentPreset;
    relight["environmentIntensity"] = rl.environmentIntensity;
    relight["shadowMode"] = rl.shadowMode;
    relight["shadowMapResolution"] = rl.shadowMapResolution;
    relight["raySamples"] = rl.raySamples;
    relight["rtQuality"] = rl.rtQuality;
    relight["progressiveShadows"] = rl.progressiveShadows;

    json root;
    root["version"] = kAppStateVersion;
    root["camera"] = camera;
    root["shDegree"] = state.shDegree;
    root["debugView"] = state.debugViewIndex;
    root["relight"] = relight;
    return root.dump();
}

void applyAppStateJson(StudioState & state, const std::string & text)
{
    json root = json::object();
    try
    {
        root = json::parse(text);
    }
    catch (const json::exception & e)
    {
        Logger::warning(std::string("Studio: app state is not JSON, using defaults: ") + e.what());
        root = json::object();
    }
    if (!root.is_object())
    {
        Logger::warning("Studio: app state is not a JSON object, using defaults");
        root = json::object();
    }

    const int version = readInt(root, "version", kAppStateVersion, 1, std::numeric_limits<int>::max());
    if (version > kAppStateVersion)
    {
        Logger::warning("Studio: app state version " + std::to_string(version) + " is newer than "
                        + std::to_string(kAppStateVersion) + "; reading the keys this build knows");
    }

    // Every field is assigned, so state from a previous project never leaks in.
    const app::OrbitCamera defaultCamera;
    app::OrbitCamera cam;
    const json camera = objectField(root, "camera");
    float pivot[3] = {defaultCamera.pivot.x, defaultCamera.pivot.y, defaultCamera.pivot.z};
    if (readFloats(camera, "pivot", pivot))
    {
        cam.pivot = glm::vec3(pivot[0], pivot[1], pivot[2]);
    }
    cam.distance = readFloat(camera, "distance", defaultCamera.distance, app::OrbitCamera::kMinDistance,
                             std::numeric_limits<float>::max());
    float quat[4] = {1.0f, 0.0f, 0.0f, 0.0f};
    if (readFloats(camera, "orientation", quat))
    {
        // Kept bit for bit when already unit length (renormalising would
        // perturb the last bits and break the save/open round trip).
        const glm::quat q(quat[0], quat[1], quat[2], quat[3]);
        const float length = glm::length(q);
        if (length <= 1e-6f)
        {
            reportAppField("camera.orientation", "is the zero quaternion; using the default");
        }
        else
        {
            if (std::fabs(length - 1.0f) >= 1e-4f)
            {
                reportAppField("camera.orientation", "is not unit length; normalised");
            }
            cam.orientation = std::fabs(length - 1.0f) < 1e-4f ? q : glm::normalize(q);
        }
    }
    state.camera = cam;

    state.shDegree = readInt(root, "shDegree", 3, 0, 3);
    state.debugViewIndex = readInt(root, "debugView", 0, 0, static_cast<int>(app::kDebugViewCycle.size()) - 1);

    const app::RelightPanelState defaults(false);
    const json relight = objectField(root, "relight");
    state.relight.enabled = readBool(relight, "enabled", defaults.enabled);
    state.relight.output = readInt(relight, "output", defaults.output, 0, kLightingOutputCount - 1);
    state.relight.exposure = readFloat(relight, "exposure", defaults.exposure, kExposureMin, kExposureMax);
    state.relight.environmentPreset = readInt(relight, "environmentPreset", defaults.environmentPreset, 0, 2);
    state.relight.environmentIntensity = readFloat(relight, "environmentIntensity", defaults.environmentIntensity,
                                                   0.0f, kEnvironmentIntensityMax);
    state.relight.shadowMode = readInt(relight, "shadowMode", defaults.shadowMode, 0, 3);
    state.relight.shadowMapResolution = readInt(relight, "shadowMapResolution", defaults.shadowMapResolution,
                                                kShadowMapMin, kShadowMapMax);
    state.relight.raySamples = readInt(relight, "raySamples", defaults.raySamples, kRaySamplesMin, kRaySamplesMax);
    state.relight.rtQuality = readInt(relight, "rtQuality", defaults.rtQuality, 0, 2);
    state.relight.progressiveShadows = readBool(relight, "progressiveShadows", defaults.progressiveShadows);
}

void resetRelightForNewScene(StudioState & state)
{
    state.relight.lights.clear();
    state.relight.placed = false;
    state.relight.bakedGeneration = UINT64_MAX;
    state.relight.bakeStatus.clear();
}

} // namespace sabercat::studio
