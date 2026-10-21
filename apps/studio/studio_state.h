// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Sabercat Studio's model: the project being edited (layers in order), the file
// it lives in, the selection, the viewer-side state the project carries, and the
// scene the renderer shows for the visible layers. Everything here is plain
// C++ over the library; the ImGui code in studio_ui.cpp drives it.
#pragma once

#include "orbit_camera.h"
#include "relight_panel.h"

#include "sabercat/layer_transform.h"
#include "sabercat/project.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace sabercat
{
class Device;
class GaussianCloud;
class Renderer;
} // namespace sabercat

namespace sabercat::studio
{

// Everything the Studio edits and saves. Grouped roughly largest first.
struct StudioState
{
    Project project; // layers in order; appStateJson is rebuilt on every save
    std::string projectPath; // empty until the project is saved or opened
    int selectedLayer = -1;  // index into project.layers, -1 for none
    bool dirty = false;      // unsaved changes to the project (not the camera)

    // The renderer's scene is the merge of the visible layers. Set whenever
    // layers, visibility or transforms change; rebuildScene() clears it.
    bool sceneDirty = true;
    // Whether the renderer has been given a scene yet. The renderer cannot
    // drop its scene, so a project with no visible layers after one was shown
    // stands in a single transparent Gaussian (see rebuildScene).
    bool rendererHasScene = false;
    double lastRebuildMs = 0.0; // wall time of the last rebuildScene(), for the UI

    // Viewer state the project carries in its app-state block.
    app::OrbitCamera camera;
    app::RelightPanelState relight{false};
    int shDegree = 3;       // max SH degree to evaluate, 0..3
    int debugViewIndex = 0; // index into app::kDebugViewCycle
};

// --- Layers ---------------------------------------------------------------

// The next unused layer id, as text: one more than the largest decimal id in use
// (0 for none), or the smallest unused id when that would overflow 64 bits.
std::string nextLayerId(const Project & project);

// Loads any scene file loadScene() accepts and appends it as a layer named after
// the file, with the identity transform. Selects the new layer. Throws on a bad
// file (the project is unchanged).
void addLayerFromFile(StudioState & state, Device & device, const std::string & path);

// Removes the selected layer. Returns false (changing nothing) if there is no
// selection or the layer is locked.
bool removeSelectedLayer(StudioState & state);

// Swaps the selected layer with its neighbour `delta` (-1 or +1) and follows
// it. Returns false, changing nothing, for any other delta, when the swap would
// leave the list, or when the selected layer or the neighbour is locked.
bool moveSelectedLayer(StudioState & state, int delta);

// Edits to one layer by index, as the Layers and Inspector panels make them.
// Each returns false, changing nothing, for an index out of range. A locked
// layer refuses renameLayer and setLayerTransform; visibility and the lock
// flag itself stay settable. An accepted edit marks the project dirty, and a
// visibility or transform edit also marks the scene for a rebuild.
bool setLayerVisible(StudioState & state, int index, bool visible);
bool setLayerLocked(StudioState & state, int index, bool locked);
bool renameLayer(StudioState & state, int index, const std::string & name);
bool setLayerTransform(StudioState & state, int index, const LayerTransform & transform);

// The visible layers, in order, as mergeLayers() takes them.
std::vector<std::pair<const GaussianCloud *, LayerTransform>> visibleLayers(const StudioState & state);

// --- Scene ----------------------------------------------------------------

// Rebuilds the renderer's scene from the visible layers (mergeLayers, then
// setGaussians) and clears sceneDirty. With no visible layers and a scene
// already shown, it shows one fully transparent Gaussian, since the renderer
// has no way to drop a scene; before any scene was shown it does nothing.
// Times the rebuild into lastRebuildMs. Throws if the renderer fails.
void rebuildScene(StudioState & state, Renderer & renderer);

// --- Project files --------------------------------------------------------

// Replaces the project with an empty one and resets the viewer state to its
// defaults.
void newProject(StudioState & state);

// Reads a .sabercat file, restores the app state from it and makes it current.
// Throws (state unchanged) if the file cannot be read.
void openProject(StudioState & state, Renderer & renderer, const std::string & path);

// Writes the project (with the current app state) to `path` and makes `path`
// the current file. Throws (projectPath and dirty unchanged) on failure.
void saveProject(StudioState & state, const std::string & path);

// Writes the merged visible layers as one binary PLY. Throws if none is visible.
void exportPly(const StudioState & state, Device & device, const std::string & path);

// Writes one glTF node per visible layer, transform and name included, as a GLB.
// Throws if none is visible.
void exportGlb(const StudioState & state, const std::string & path);

// --- App state ------------------------------------------------------------

// The app-state block: orbit camera, SH degree, debug view and the relight
// panel's scalar settings, as a JSON object.
std::string buildAppStateJson(const StudioState & state);

// Restores what buildAppStateJson() wrote. A missing key takes its default
// silently, so older files open; unknown keys are ignored, so newer files open
// too. A value of the wrong type, or a non-finite or malformed one, takes the
// default. A value outside the range the panels offer is clamped to the nearest
// end. Each of those cases is one warning line on the log.
void applyAppStateJson(StudioState & state, const std::string & json);

// Relight lights and bake state belong to a scene, not to the project file:
// call after switching projects so the panel places its defaults again.
void resetRelightForNewScene(StudioState & state);

} // namespace sabercat::studio
