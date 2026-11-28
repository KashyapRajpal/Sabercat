// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Sabercat Studio's editing model (docs/studio-design.md, phase C5, part 1): the
// selection, and the transform, delete and duplicate, of the ACTIVE layer
// (StudioState::selectedLayer), on the GPU, with the undo history and the bounds
// the renderer frames. No ImGui: the UI calls these functions and reads the state
// they leave.
//
// Who owns what.
//   - Each visible layer's GaussianCloud is uploaded to its own GPU buffers
//     (ensureLayerGpu). An edit changes the GPU copy, which is then authoritative:
//     the layer is hostStale and its host vectors catch up lazily (syncLayerToHost).
//     saveProject() and the exporters sync first.
//   - state.edit owns the layers' selection masks and the history. The history's
//     commands hold pointers to the StudioState and the Renderer, so StudioState is
//     neither copyable nor movable, and state.edit.reset() must run before the
//     renderer shuts down (see studio_state.h).
//   - The project file does not store selections. A selection is not part of the
//     project file and does not mark it dirty; a save writes the layers without any
//     selection.
//
// Scope of the undo history: edits to a layer's Gaussians (select, transform,
// delete, duplicate) are undoable, and so are changes to the layer list (add,
// remove, move) and to a layer's properties (rename, visibility, the Inspector's
// transform; studio_edits.h). Not undoable: the lock flag, which is a guard the user
// sets rather than an edit (studio_state.h). Opening or creating a project clears the
// history.
#pragma once

#include "edit_history.h"

#include "sabercat/edit_pass.h"
#include "sabercat/highlight_settings.h"
#include "sabercat/layer_transform.h"
#include "sabercat/project.h"
#include "sabercat/scene_bounds.h"
#include "sabercat/selection_mask.h"

#include <glm/glm.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace sabercat
{
class Renderer;
} // namespace sabercat

namespace sabercat::studio
{

struct StudioState;

// The edit state of one layer, keyed by ProjectLayer::id in StudioEdit::layers.
struct LayerEditState
{
    // The selection: one bit per Gaussian, sized to the cloud's capacity. Created by
    // ensureLayerGpu(); bits at or beyond the cloud's count are always zero.
    std::unique_ptr<SelectionMask> mask;
    // The layer's bounds in its own (layer) space. Exact after a host sync. While
    // hostStale they are conservative: an edit widens them and they never shrink.
    SceneBounds localBounds;
    // The number of selected Gaussians, kept exact by every selection and edit.
    uint32_t selectedCount = 0;
    // True while the GPU copy is ahead of the host vectors (after a transform,
    // delete or duplicate, or their undo or redo).
    bool hostStale = false;
};

// A LayerEditState moved out of StudioEdit::layers (std::map::extract). A layer-list
// command holds one while its layer is out of the project, so the selection mask
// survives a round trip through undo and redo. Empty when the layer had no state.
using LayerEditNode = std::map<std::string, LayerEditState>::node_type;

// A transform applied live, before it is committed. previewTransform() records the
// pre-transform values of the selection and applies the transform; each later call
// restores those values and applies the new transform, so the layer always shows
// exactly one transform on top of the recorded values. commitTransform() turns the
// preview into a TransformCommand; cancelTransformPreview() restores the values.
struct TransformPreview
{
    bool active = false;
    std::string layerId;              // the layer the preview is on
    SparseSnapshot snapshot;          // the pre-transform values (SnapshotScope::Transform)
    std::vector<uint32_t> maskWords;  // the selection the preview was started with
    LayerTransform appliedLayerSpace; // the transform now applied; identity when none is
    // The layer's bounds and staleness before the preview started. Each update widens
    // these (not the previous update's bounds), so repeated updates do not compound;
    // a cancel restores them.
    SceneBounds baseBounds;
    bool hostStaleBefore = false;
    // The renderer the preview was applied with (set when it starts, cleared when it
    // ends). It lets cancelTransformPreview(state) restore the layer without a renderer
    // argument, which saveProject() and the exporters need.
    Renderer * renderer = nullptr;
};

// The Studio's edit state, a value member of StudioState. Default-constructible with
// no GPU work. Not copyable (nor movable: see the header comment).
class StudioEdit
{
  public:
    StudioEdit() = default;
    StudioEdit(const StudioEdit &) = delete;
    StudioEdit & operator=(const StudioEdit &) = delete;

    EditHistory history;                          // a 1 GiB budget by default
    std::map<std::string, LayerEditState> layers; // by ProjectLayer::id
    // Set after one failed GPU layer merge: the host merge is used from then on for
    // this session, and the failure is logged once. Kept across reset().
    bool gpuMergeFailed = false;
    // The renderer's bounds came from mergedLayerBounds() (exact), not the
    // conservative bounds of the edit state.
    bool boundsExact = false;
    // A drag is in progress: the renderer's bounds stay conservative until it ends.
    bool interacting = false;
    // The selection highlight's look, as the UI edits it. enabled = 0 by default.
    HighlightSettings highlight;
    TransformPreview preview;
    // Set while a command runs its undo() or redo() (see ReplayGuard). The state
    // functions that record a command in the history do nothing then, so replaying
    // an edit never records a new one. Restored by ReplayGuard, also on an exception.
    bool replaying = false;

    // Drops every layer's GPU state (the masks), the history and the preview, and
    // clears boundsExact and interacting. Needs no GPU: it destroys the masks that
    // exist. Call it before the renderer shuts down, or before the project's layers
    // are replaced.
    void reset();
};

// Sets StudioEdit::replaying for its lifetime and restores the previous value when it
// ends, including when an exception leaves the scope. Every undo() and redo() of the
// layer commands (studio_edits.h) holds one. Not copyable or movable.
class ReplayGuard
{
  public:
    explicit ReplayGuard(StudioEdit & edit);
    ~ReplayGuard();
    ReplayGuard(const ReplayGuard &) = delete;
    ReplayGuard & operator=(const ReplayGuard &) = delete;

  private:
    StudioEdit * m_edit;
    bool m_previous;
};

// --- Active layer -----------------------------------------------------------

// The active layer when it can be edited: it has a cloud, is visible and is not
// locked. nullptr otherwise, including when nothing is selected.
ProjectLayer * editableLayer(StudioState & state);

// Why editableLayer() is nullptr, as text for the UI: "No layer selected",
// "The layer has no Gaussians", "The layer is locked" or "The layer is hidden".
// Empty when the active layer is editable.
const char * whyNotEditable(const StudioState & state);

// The layer's edit state, created empty on first use. No GPU work.
LayerEditState & editStateOf(StudioState & state, ProjectLayer & layer);

// Makes the layer ready for GPU edits. When its cloud has no GPU storage, uploads
// it (exactly count() Gaussians). Creates the selection mask (zeroed, at the cloud's
// capacity) and the local bounds on first use, or resizes the mask when the cloud's
// capacity has grown. Throws if the cloud is null or the upload fails.
void ensureLayerGpu(StudioState & state, Renderer & renderer, ProjectLayer & layer);

// Downloads a stale layer's GPU copy into its host vectors and recomputes its exact
// local bounds. Silent, and a no-op for a layer that is not stale. Throws if the
// download fails (the layer stays stale).
void syncLayerToHost(StudioState & state, ProjectLayer & layer);
// syncLayerToHost() for every layer of the project.
void syncAllToHost(StudioState & state);

// --- Selection --------------------------------------------------------------
//
// Each selection change runs on the active layer and pushes one SelectionCommand
// (after it has run). It refreshes the cached count and the highlight. It does not
// set dirty (selections are not part of the project file) and never sets sceneDirty:
// the highlight is refreshed on its own. A change
// made while a transform preview is active cancels that preview first (which does
// set sceneDirty, because the layer's data is restored). Each throws, changing
// nothing, when the active layer is not editable (the message is whyNotEditable()).

// An axis-aligned box in world space (its axes are world x, y and z).
void selectBox(StudioState & state, Renderer & renderer, SelectOp op, const glm::vec3 & center,
               const glm::vec3 & halfExtents);
// A sphere in world space.
void selectSphere(StudioState & state, Renderer & renderer, SelectOp op, const glm::vec3 & center,
                  float radius);
// A rectangle in framebuffer pixels, given by any two opposite corners. It is projected
// with the camera of the last completed forward() (Renderer::lastForwardCamera()), so it
// throws std::runtime_error with a message to draw a frame first when there is none.
void selectRectangle(StudioState & state, Renderer & renderer, SelectOp op, const glm::vec2 & cornerA,
                     const glm::vec2 & cornerB);
void selectAllGaussians(StudioState & state, Renderer & renderer);
void selectNone(StudioState & state, Renderer & renderer);
void invertSelection(StudioState & state, Renderer & renderer);

// The cached number of selected Gaussians of the active layer; 0 when there is none.
uint32_t selectedCountOfActiveLayer(const StudioState & state);

// --- Transform --------------------------------------------------------------

// Where a transform's pivot is.
enum class PivotMode
{
    SelectionCentroid, // the world position of the selection's centroid
    LayerOrigin,       // the layer's placement translation
    WorldOrigin,       // the world origin
};

// A world-space edit: x' = scale * R * (x - pivot) + pivot + translation, with R the
// rotation of rotationDegrees (XYZ Euler angles in degrees, glm's convention). The
// fields are ordered so the struct has no padding.
struct TransformParams
{
    glm::vec3 translation{0.0f};
    glm::vec3 rotationDegrees{0.0f};
    float scale = 1.0f;                           // must be positive and finite
    PivotMode pivot = PivotMode::SelectionCentroid;
};

// Starts a live preview of the transform on the selection of the active layer, or
// updates the running one: the first call records the pre-transform values; every
// call restores the previous preview and applies this one. The layer's scene is
// marked for a rebuild (sceneDirty) so the caller shows it; dirty is set only on
// commit. Throws std::invalid_argument for a scale that is not positive and finite,
// std::runtime_error when the active layer is not editable or nothing is selected.
// A selection of more than 250000 Gaussians is still exact, only slower: the
// preview's host snapshot grows with the selection and every update restores it.
void previewTransform(StudioState & state, Renderer & renderer, const TransformParams & params);

// Restores the pre-preview values of the previewed layer and drops the preview. A
// no-op without a preview. Sets sceneDirty.
void cancelTransformPreview(StudioState & state, Renderer & renderer);
// The same, with the renderer the preview was started with (TransformPreview::renderer).
// A no-op without a preview, and then it needs no GPU. saveProject() and the exporters
// call it first, so a file never holds previewed values.
void cancelTransformPreview(StudioState & state);

// Applies the transform (through the preview, starting or updating it as
// previewTransform() does) and pushes one TransformCommand. Sets dirty.
void commitTransform(StudioState & state, Renderer & renderer, const TransformParams & params);

// --- Delete and duplicate (active layer, one command each) ------------------

// Removes the selected Gaussians. Returns false, changing nothing, with the reason in
// whyNot, when the active layer is not editable, nothing is selected, or every
// Gaussian is selected (remove the layer instead). Returns true and clears whyNot
// otherwise.
bool deleteSelection(StudioState & state, Renderer & renderer, std::string & whyNot);
// Appends copies of the selected Gaussians; the copies become the selection. Same
// refusals as deleteSelection(), except that selecting everything is allowed.
bool duplicateSelection(StudioState & state, Renderer & renderer, std::string & whyNot);

// --- History ----------------------------------------------------------------

// Undo and redo, which cancel an active transform preview first. False, changing
// nothing, when there is nothing to undo or redo.
bool undo(StudioState & state, Renderer & renderer);
bool redo(StudioState & state, Renderer & renderer);
// Moves the history cursor to `index` (0 to the number of commands). False, changing
// nothing, when index is past the end.
bool jumpToHistory(StudioState & state, Renderer & renderer, std::size_t index);
// Drops the history without undoing anything. The layer-list and layer-property
// changes are commands now, so nothing in the Studio calls this on a layer change.
void clearEditHistory(StudioState & state);

// --- Highlight and bounds ---------------------------------------------------

// Pushes the selection highlight to the renderer: when it is enabled, the scene is
// current (not sceneDirty) and the GPU merge is in use, the mask of every visible
// layer in merge order; otherwise the highlight is switched off. rebuildScene() calls
// it at its end, because a scene rebuild disables the renderer's highlight.
void refreshHighlight(StudioState & state, Renderer & renderer);

// The bounds the renderer should frame: the exact mergedLayerBounds() of the visible
// layers when none of them is stale and no drag is in progress (and it sets
// boundsExact); otherwise the conservative box of the layers' local bounds, mapped
// by each layer's placement (and it clears boundsExact). Invalid with no visible
// layer.
SceneBounds currentSceneBounds(StudioState & state);

// Makes the renderer's bounds exact: every stale visible layer is synced to the host
// (syncLayerToHost), then the renderer's bounds are set to mergedLayerBounds() of the
// visible layers and boundsExact is set. With no visible layer it does nothing. Used
// before Frame all, so the camera frames the real scene rather than the conservative
// box of a drag. Throws if a sync or the bounds fail (the bounds stay as they were).
void refreshExactBounds(StudioState & state, Renderer & renderer);

// `box` united with the axis-aligned box of its eight corners mapped by `layerSpace`:
// the conservative bounds after a layer-space transform. An invalid box is returned
// unchanged.
SceneBounds widenedBounds(const SceneBounds & box, const LayerTransform & layerSpace);

} // namespace sabercat::studio
