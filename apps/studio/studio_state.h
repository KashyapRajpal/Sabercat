// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Sabercat Studio's model: the project being edited (layers in order), the file
// it lives in, the selection, the viewer-side state the project carries, and the
// scene the renderer shows for the visible layers. Everything here is plain
// C++ over the library; the ImGui code in studio_ui.cpp drives it.
//
// Lifetime: state.edit holds GPU masks and an undo history whose commands point at
// this StudioState and at the Renderer. So a StudioState is neither copied nor moved,
// and state.edit.reset() must run before the renderer shuts down (main.cpp does it
// before it drops the project). Selection is not stored in the project file.
#pragma once

#include "orbit_camera.h"
#include "relight_panel.h"
#include "studio_edit.h"

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
    bool dirty = false;      // unsaved changes to the project (not the camera or the selection)

    // The renderer's scene is the merge of the visible layers. Set whenever
    // layers, visibility or transforms change; rebuildScene() clears it.
    bool sceneDirty = true;
    double lastRebuildMs = 0.0; // wall time of the last rebuildScene(), for the UI

    // The editing model of the active layer: selection masks, the undo history,
    // the live transform preview and the highlight (studio_edit.h).
    StudioEdit edit;

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
// file (the project is unchanged). Recorded in the undo history (AddLayerCommand)
// unless a command is replaying. A transform preview on another layer stays.
void addLayerFromFile(StudioState & state, Device & device, const std::string & path);

// Removes the selected layer, with its edit state. Returns false (changing nothing)
// if there is no selection or the layer is locked. Recorded in the undo history
// (RemoveLayerCommand): undo puts the layer back with its selection mask. A transform
// preview on this layer is restored first, so the layer's data is the committed data.
bool removeSelectedLayer(StudioState & state);

// Swaps the selected layer with its neighbor `delta` (-1 or +1) and follows
// it. Returns false, changing nothing, for any other delta, when the swap would
// leave the list, or when the selected layer or the neighbor is locked. Recorded
// in the undo history (MoveLayerCommand).
bool moveSelectedLayer(StudioState & state, int delta);

// Edits to one layer by index, as the Layers and Inspector panels make them.
// Each returns false, changing nothing, for an index out of range. A locked
// layer refuses renameLayer and setLayerTransform; visibility and the lock
// flag itself stay settable. An accepted edit marks the project dirty, and a
// visibility or transform edit also marks the scene for a rebuild.
//
// Undo: a rename and a visibility change that changed the value are recorded
// (RenameLayerCommand, LayerVisibilityCommand). setLayerTransform is NOT recorded,
// because the Inspector calls it on every frame of a drag: the UI calls
// commitLayerTransformEdit() when the drag ends. setLayerLocked is NOT recorded:
// a lock toggle is not undoable (it is a guard the user sets, not an edit).
bool setLayerVisible(StudioState & state, int index, bool visible);
bool setLayerLocked(StudioState & state, int index, bool locked);
bool renameLayer(StudioState & state, int index, const std::string & name);
bool setLayerTransform(StudioState & state, int index, const LayerTransform & transform);

// Records the end of a transform drag on layer `index` (LayerTransformCommand) when its
// transform differs bit for bit from `before`, the transform at the drag's start. The
// command is pushed without running it again: the layer already holds the new transform.
// Returns true when a command was pushed. False, changing nothing, for an index out of
// range, when the transform is unchanged, or while a command is replaying.
bool commitLayerTransformEdit(StudioState & state, int index, const LayerTransform & before);

// The visible layers, in order, as mergeLayers() takes them.
std::vector<std::pair<const GaussianCloud *, LayerTransform>> visibleLayers(const StudioState & state);

// --- Scene ----------------------------------------------------------------

// Rebuilds the renderer's scene from the visible layers and clears sceneDirty.
// Every visible layer is first made GPU resident (ensureLayerGpu). The scene is then
// merged on the GPU (Renderer::mergeLayersOnGpu) and framed by currentSceneBounds().
// If that merge fails once, it is logged and the host merge (mergeLayers, then
// setGaussians, after syncAllToHost) is used for the rest of the session. With no
// visible layers it clears the renderer's scene (Renderer::clearScene). Ends by
// refreshing the selection highlight. Times the rebuild into lastRebuildMs. Throws
// if the renderer fails in a way the host merge cannot recover from.
void rebuildScene(StudioState & state, Renderer & renderer);

// --- Project files --------------------------------------------------------

// Replaces the project with an empty one and resets the viewer state to its
// defaults. Drops the edit state and the history (state.edit.reset()).
void newProject(StudioState & state);

// Reads a .sabercat file, restores the app state from it and makes it current.
// Throws (state unchanged) if the file cannot be read. Otherwise cancels a
// transform preview, then drops the edit state and the history with the old layers.
void openProject(StudioState & state, Renderer & renderer, const std::string & path);

// Writes the project (with the current app state) to `path` and makes `path`
// the current file. First restores any transform preview (cancelTransformPreview(state)),
// so the file never holds previewed values; then syncs stale layers to the host, so the
// file holds the edits. Throws (projectPath and dirty unchanged) on failure.
void saveProject(StudioState & state, const std::string & path);

// Writes the merged visible layers as one binary PLY. Restores any transform preview
// first, then syncs stale layers to the host. Throws if none is visible.
void exportPly(StudioState & state, Device & device, const std::string & path);

// Writes one glTF node per visible layer, transform and name included, as a GLB.
// Restores any transform preview first, then syncs stale layers to the host. Throws if
// none is visible.
void exportGlb(StudioState & state, const std::string & path);

// Writes the merged visible layers as one SPZ (the default container, version 3).
// Restores any transform preview first, then syncs stale layers to the host. Throws if
// none is visible, or if the scene cannot be written (see SPZWriter).
void exportSpz(StudioState & state, Device & device, const std::string & path);

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
