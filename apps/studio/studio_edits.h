// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The Studio's undoable edits (studio_edit.h): one EditCommand per kind of change to a
// layer's Gaussians. Each command is pushed after its change has run. undo() and redo()
// put the layer back in the state it had, or applies it again, on the GPU.
//
// A command holds a pointer to the StudioState and to the Renderer (both stable for the
// session) and the id of its layer. It looks the layer up by id each time it runs: when
// the layer is gone it throws std::runtime_error.
//
// The layer-list and layer-property commands (LayerOpCommand and below) need no Renderer:
// they look their layer up by id in the project, set the layer's own fields, and never
// touch the GPU. Their undo() and redo() run under a ReplayGuard.
//
// Invariant (LIFO): the history undoes the newest command first. So a Gaussian edit of a
// layer never runs while the layer is out of the project: its RemoveLayerCommand is undone
// (the layer comes back with its edit state) before any earlier edit of that layer runs,
// and a later edit is undone before the removal. The edit state moves with the layer:
// AddLayerCommand and RemoveLayerCommand hold the layer's LayerEditState (its selection
// mask) while the layer is out, so the masks survive a round trip.
//
// Budget: a layer-list command's bytes() is the host memory of the cloud it holds (the
// layer is held only while it is out of the project). Eviction removes the oldest
// command, so when a layer-list command has been evicted, nothing older exists.
#pragma once

#include "edit_history.h"
#include "studio_edit.h"

#include "sabercat/edit_pass.h"
#include "sabercat/layer_transform.h"
#include "sabercat/project.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace sabercat
{
class Renderer;
} // namespace sabercat

namespace sabercat::studio
{

struct StudioState;

// The part every command shares: the state, the renderer, and the layer's id.
class LayerEditCommand : public EditCommand
{
  public:
    LayerEditCommand(StudioState & state, Renderer & renderer, std::string layerId);

  protected:
    // The command's layer. Throws std::runtime_error if the project no longer has it.
    ProjectLayer & layer() const;

    StudioState * state;
    Renderer * renderer;
    std::string layerId;
};

// A selection change: the mask words before and after the change (one bit per Gaussian).
// Undo uploads `before`, redo uploads `after`. Label "Select".
class SelectionCommand final : public LayerEditCommand
{
  public:
    SelectionCommand(StudioState & state, Renderer & renderer, std::string layerId,
                     std::vector<uint32_t> before, std::vector<uint32_t> after);

    void undo() override;
    void redo() override;
    std::string label() const override;
    std::size_t bytes() const override;

  private:
    // Puts the mask back to `words` and refreshes what depends on it.
    void restore(const std::vector<uint32_t> & words);

    std::vector<uint32_t> m_before;
    std::vector<uint32_t> m_after;
};

// A committed transform of the selection. The snapshot holds the Transform-scope values
// from before the transform; undo restores them, redo applies the layer-space transform
// again to the same selection. Label "Transform".
class TransformCommand final : public LayerEditCommand
{
  public:
    TransformCommand(StudioState & state, Renderer & renderer, std::string layerId,
                     std::vector<uint32_t> maskWords, SparseSnapshot snapshot,
                     const LayerTransform & layerSpace);

    void undo() override;
    void redo() override;
    std::string label() const override;
    std::size_t bytes() const override;

  private:
    std::vector<uint32_t> m_maskWords;
    SparseSnapshot m_snapshot;
    LayerTransform m_layerSpace;
};

// A deletion of the selected Gaussians, with the record that undoes it. Label "Delete N",
// N being the number of Gaussians removed.
class DeleteCommand final : public LayerEditCommand
{
  public:
    DeleteCommand(StudioState & state, Renderer & renderer, std::string layerId, DeleteRecord record);

    void undo() override;
    void redo() override;
    std::string label() const override;
    std::size_t bytes() const override;

  private:
    DeleteRecord m_record;
};

// A duplication of the selected Gaussians, with the record that undoes it. Label
// "Duplicate N", N being the number of Gaussians appended.
class DuplicateCommand final : public LayerEditCommand
{
  public:
    DuplicateCommand(StudioState & state, Renderer & renderer, std::string layerId,
                     DuplicateRecord record);

    void undo() override;
    void redo() override;
    std::string label() const override;
    std::size_t bytes() const override;

  private:
    DuplicateRecord m_record;
};

// --- Layer list and layer properties ----------------------------------------
//
// Commands for the project's layer list and for a layer's own fields. Each is pushed
// after its change has run, and each undo() and redo() sets the project and the scene
// dirty and keeps state->selectedLayer valid (clamped to the layer list). Each looks its
// layer up by id and throws std::runtime_error when it is missing.
//
// Not undoable, by design: the lock flag (setLayerLocked). A lock is a guard the user
// sets, not an edit, so undoing an edit never unlocks or relocks a layer.

// The part the layer-list and layer-property commands share: the state, the layer's id,
// and the label and budget bytes fixed at construction (while the command is pushed).
class LayerOpCommand : public EditCommand
{
  public:
    LayerOpCommand(StudioState & stateRef, std::string id, std::string text, std::size_t byteCount);

    std::string label() const override;
    std::size_t bytes() const override;

  protected:
    // The command's layer. Throws std::runtime_error if the project no longer has it.
    ProjectLayer & layer();
    // Marks the project and the scene dirty, after an undo or redo.
    void markChanged();
    // Sets the selection to `index`, clamped to the layer list (-1 when it is empty).
    void setSelection(int index);

    StudioState * state;
    std::string layerId;

  private:
    std::string m_label;
    std::size_t m_bytes;
};

// The addition of a layer, pushed after the layer was appended at `index`. The layer
// selected before the add and the one selected after it are stored. Undo takes the layer
// out of the project (keeping it and its edit state for redo); redo puts it back.
// Label "Add layer <name>". bytes() is the cloud's host footprint (0 without a cloud),
// which the command holds while the layer is out.
class AddLayerCommand final : public LayerOpCommand
{
  public:
    AddLayerCommand(StudioState & stateRef, const ProjectLayer & added, int index, int selectedBefore,
                    int selectedAfter);

    void undo() override;
    void redo() override;

  private:
    int m_index;
    int m_selectedBefore;
    int m_selectedAfter;
    std::optional<ProjectLayer> m_held; // the layer while it is out of the project
    LayerEditNode m_node;               // its edit state while it is out
};

// The removal of a layer, pushed after the layer was removed from `index`. The command
// holds the layer and its edit state (possibly empty) from then on. Undo puts them back at
// `index`; redo takes them out again. Label "Remove layer <name>". bytes() is the cloud's
// host footprint (0 without a cloud).
class RemoveLayerCommand final : public LayerOpCommand
{
  public:
    RemoveLayerCommand(StudioState & stateRef, ProjectLayer removed, LayerEditNode node, int index,
                       int selectedBefore, int selectedAfter);

    void undo() override;
    void redo() override;

  private:
    int m_index;
    int m_selectedBefore;
    int m_selectedAfter;
    std::optional<ProjectLayer> m_held; // the layer while it is out of the project
    LayerEditNode m_node;               // its edit state while it is out
};

// A move of a layer from `fromIndex` to `toIndex` (the list's other layers keep their order).
// Undo moves it back, redo moves it again. Label "Move layer <name>".
class MoveLayerCommand final : public LayerOpCommand
{
  public:
    MoveLayerCommand(StudioState & stateRef, std::string layerId, int fromIndex, int toIndex,
                     int selectedBefore, int selectedAfter);

    void undo() override;
    void redo() override;

  private:
    // Moves the layer to `index` in the list.
    void moveTo(int index);

    int m_fromIndex;
    int m_toIndex;
    int m_selectedBefore;
    int m_selectedAfter;
};

// A rename: undo restores the name before, redo sets the name after. Label "Rename layer".
// Applied directly, so a locked layer can be restored.
class RenameLayerCommand final : public LayerOpCommand
{
  public:
    RenameLayerCommand(StudioState & stateRef, std::string layerId, std::string before, std::string after);

    void undo() override;
    void redo() override;

  private:
    std::string m_before;
    std::string m_after;
};

// A visibility change. Label "Show layer <name>" when `after` is true, else "Hide layer <name>".
// Applied directly, so a locked layer can be restored.
class LayerVisibilityCommand final : public LayerOpCommand
{
  public:
    LayerVisibilityCommand(StudioState & stateRef, std::string layerId, bool before, bool after);

    void undo() override;
    void redo() override;

  private:
    bool m_before;
    bool m_after;
};

// A committed transform of a layer (the Inspector's edit, from `before` to `after`).
// Label "Layer transform <name>". Applied directly, so a locked layer can be restored.
class LayerTransformCommand final : public LayerOpCommand
{
  public:
    LayerTransformCommand(StudioState & stateRef, std::string layerId, const LayerTransform & before,
                          const LayerTransform & after);

    void undo() override;
    void redo() override;

  private:
    LayerTransform m_before;
    LayerTransform m_after;
};

} // namespace sabercat::studio
