// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "studio_edits.h"

#include "studio_edit.h"
#include "studio_state.h"

#include "sabercat/gaussian_cloud.h"
#include "sabercat/renderer.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sabercat::studio
{

namespace
{

// What every undo or redo of a Gaussian edit leaves behind: the layer's GPU copy is
// ahead of its host vectors, the project and the scene are dirty, and the cached
// selection count matches the mask again.
void markEditedOnGpu(StudioState & state, Renderer & renderer, ProjectLayer & layer, LayerEditState & es)
{
    es.hostStale = true;
    es.selectedCount = renderer.editPass().countSelected(*es.mask, static_cast<uint32_t>(layer.cloud->count()));
    state.dirty = true;
    state.sceneDirty = true;
}

} // namespace

// --- LayerEditCommand -------------------------------------------------------

LayerEditCommand::LayerEditCommand(StudioState & stateRef, Renderer & rendererRef, std::string id)
    : state(&stateRef), renderer(&rendererRef), layerId(std::move(id))
{
}

ProjectLayer & LayerEditCommand::layer() const
{
    for (ProjectLayer & candidate : state->project.layers)
    {
        if (candidate.id == layerId)
        {
            if (!candidate.cloud)
            {
                throw std::runtime_error("Studio: the edit's layer " + layerId + " has no Gaussians");
            }
            return candidate;
        }
    }
    throw std::runtime_error("Studio: cannot undo or redo an edit of layer " + layerId
                             + ": the layer is no longer in the project");
}

// --- SelectionCommand -------------------------------------------------------

SelectionCommand::SelectionCommand(StudioState & stateRef, Renderer & rendererRef, std::string id,
                                   std::vector<uint32_t> before, std::vector<uint32_t> after)
    : LayerEditCommand(stateRef, rendererRef, std::move(id)), m_before(std::move(before)), m_after(std::move(after))
{
}

void SelectionCommand::undo()
{
    restore(m_before);
}

void SelectionCommand::redo()
{
    restore(m_after);
}

std::string SelectionCommand::label() const
{
    return "Select";
}

std::size_t SelectionCommand::bytes() const
{
    return (m_before.size() + m_after.size()) * sizeof(uint32_t);
}

void SelectionCommand::restore(const std::vector<uint32_t> & words)
{
    ProjectLayer & target = layer();
    ensureLayerGpu(*state, *renderer, target);
    LayerEditState & es = editStateOf(*state, target);
    es.mask->upload(words);
    es.selectedCount = renderer->editPass().countSelected(*es.mask, static_cast<uint32_t>(target.cloud->count()));
    // Selections are not saved in the project, so they do not mark it dirty.
    refreshHighlight(*state, *renderer);
}

// --- TransformCommand -------------------------------------------------------

TransformCommand::TransformCommand(StudioState & stateRef, Renderer & rendererRef, std::string id,
                                   std::vector<uint32_t> maskWords, SparseSnapshot snapshot,
                                   const LayerTransform & layerSpace)
    : LayerEditCommand(stateRef, rendererRef, std::move(id)),
      m_maskWords(std::move(maskWords)),
      m_snapshot(std::move(snapshot)),
      m_layerSpace(layerSpace)
{
}

void TransformCommand::undo()
{
    ProjectLayer & target = layer();
    ensureLayerGpu(*state, *renderer, target);
    LayerEditState & es = editStateOf(*state, target);
    es.mask->upload(m_maskWords);
    renderer->editPass().scatterSnapshot(*target.cloud, m_snapshot);
    // Conservative: never shrink while stale.
    es.localBounds = widenedBounds(es.localBounds, m_layerSpace);
    markEditedOnGpu(*state, *renderer, target, es);
}

void TransformCommand::redo()
{
    ProjectLayer & target = layer();
    ensureLayerGpu(*state, *renderer, target);
    LayerEditState & es = editStateOf(*state, target);
    es.mask->upload(m_maskWords);
    renderer->editPass().transformSelected(*target.cloud, *es.mask, m_layerSpace);
    es.localBounds = widenedBounds(es.localBounds, m_layerSpace);
    markEditedOnGpu(*state, *renderer, target, es);
}

std::string TransformCommand::label() const
{
    return "Transform";
}

std::size_t TransformCommand::bytes() const
{
    return m_snapshot.bytes() + m_maskWords.size() * sizeof(uint32_t);
}

// --- DeleteCommand ----------------------------------------------------------

DeleteCommand::DeleteCommand(StudioState & stateRef, Renderer & rendererRef, std::string id, DeleteRecord record)
    : LayerEditCommand(stateRef, rendererRef, std::move(id)), m_record(std::move(record))
{
}

void DeleteCommand::undo()
{
    ProjectLayer & target = layer();
    ensureLayerGpu(*state, *renderer, target);
    LayerEditState & es = editStateOf(*state, target);
    renderer->editPass().undoDelete(*target.cloud, *es.mask, m_record);
    markEditedOnGpu(*state, *renderer, target, es);
}

void DeleteCommand::redo()
{
    ProjectLayer & target = layer();
    ensureLayerGpu(*state, *renderer, target);
    LayerEditState & es = editStateOf(*state, target);
    es.mask->upload(m_record.selectionWords);
    // The same deletion again; its record replaces this one, so the next undo matches the GPU.
    m_record = renderer->editPass().deleteSelected(*target.cloud, *es.mask);
    markEditedOnGpu(*state, *renderer, target, es);
}

std::string DeleteCommand::label() const
{
    return "Delete " + std::to_string(m_record.deleted.count());
}

std::size_t DeleteCommand::bytes() const
{
    return m_record.deleted.bytes() + m_record.selectionWords.size() * sizeof(uint32_t);
}

// --- DuplicateCommand -------------------------------------------------------

DuplicateCommand::DuplicateCommand(StudioState & stateRef, Renderer & rendererRef, std::string id,
                                   DuplicateRecord record)
    : LayerEditCommand(stateRef, rendererRef, std::move(id)), m_record(std::move(record))
{
}

void DuplicateCommand::undo()
{
    ProjectLayer & target = layer();
    ensureLayerGpu(*state, *renderer, target);
    LayerEditState & es = editStateOf(*state, target);
    renderer->editPass().undoDuplicate(*target.cloud, *es.mask, m_record);
    markEditedOnGpu(*state, *renderer, target, es);
}

void DuplicateCommand::redo()
{
    ProjectLayer & target = layer();
    ensureLayerGpu(*state, *renderer, target);
    LayerEditState & es = editStateOf(*state, target);
    es.mask->upload(m_record.selectionWords);
    // The same duplication again; the new record replaces this one.
    m_record = renderer->editPass().duplicateSelected(*target.cloud, *es.mask);
    markEditedOnGpu(*state, *renderer, target, es);
}

std::string DuplicateCommand::label() const
{
    return "Duplicate " + std::to_string(m_record.added);
}

std::size_t DuplicateCommand::bytes() const
{
    return m_record.selectionWords.size() * sizeof(uint32_t);
}

// --- Layer list and layer properties ----------------------------------------

namespace
{

// The host footprint of a layer's cloud: what a layer-list command holds while its layer
// is out of the project. 0 for a layer without a cloud. The core streams take 308 bytes
// per Gaussian at SH degree 3; each relight stream adds its bytes per Gaussian only when
// its host vector is non-empty (normals 16, base color and roughness 16, metallic 4,
// visibility 64).
std::size_t cloudBytes(const ProjectLayer & layer)
{
    if (!layer.cloud)
        return 0;
    const GaussianCloud & cloud = *layer.cloud;
    std::size_t perGaussian = 308;
    if (!cloud.normals().empty())
        perGaussian += 16;
    if (!cloud.baseColorRoughness().empty())
        perGaussian += 16;
    if (!cloud.metallic().empty())
        perGaussian += 4;
    if (!cloud.visibility().empty())
        perGaussian += 64;
    return cloud.count() * perGaussian;
}

// The error for a layer-list or layer-property command whose layer is not in the project.
std::runtime_error missingLayer(const std::string & id)
{
    return std::runtime_error("Studio: cannot undo or redo a layer change: layer " + id
                              + " is not in the project");
}

// The index of the layer `id` in the project, or -1.
int indexOfLayer(const StudioState & state, const std::string & id)
{
    const std::vector<ProjectLayer> & layers = state.project.layers;
    for (size_t i = 0; i < layers.size(); ++i)
    {
        if (layers[i].id == id)
        {
            return static_cast<int>(i);
        }
    }
    return -1;
}

// The name of the layer `id`, for a label. Throws if the project has no such layer.
std::string nameOfLayer(const StudioState & state, const std::string & id)
{
    const int index = indexOfLayer(state, id);
    if (index < 0)
    {
        throw missingLayer(id);
    }
    return state.project.layers[static_cast<size_t>(index)].name;
}

// Takes the layer `id` out of the project and its edit state out of the edit map (into
// `node`, empty when it had none). The layer must be at `index`, the position the command
// recorded. Throws, changing nothing, when it is not there.
ProjectLayer takeLayer(StudioState & state, const std::string & id, int index, LayerEditNode & node)
{
    const int found = indexOfLayer(state, id);
    if (found < 0)
    {
        throw missingLayer(id);
    }
    if (found != index)
    {
        throw std::runtime_error("Studio: cannot undo or redo a layer change: layer " + id + " is at index "
                                 + std::to_string(found) + ", not " + std::to_string(index));
    }
    std::vector<ProjectLayer> & layers = state.project.layers;
    ProjectLayer taken = std::move(layers[static_cast<size_t>(index)]);
    layers.erase(layers.begin() + index);
    node = state.edit.layers.extract(id);
    return taken;
}

// Puts `layer` back at `index`, with its edit state `node` (moved in when it is not empty).
// Every check runs before anything changes: it throws when the id is already in the project,
// the index is outside the list, or the edit map already has the id.
void restoreLayer(StudioState & state, const ProjectLayer & layer, int index, LayerEditNode & node)
{
    std::vector<ProjectLayer> & layers = state.project.layers;
    if (indexOfLayer(state, layer.id) >= 0)
    {
        throw std::runtime_error("Studio: cannot undo or redo a layer change: layer " + layer.id
                                 + " is already in the project");
    }
    if (index < 0 || static_cast<size_t>(index) > layers.size())
    {
        throw std::runtime_error("Studio: cannot undo or redo a layer change: index " + std::to_string(index)
                                 + " is outside the layer list");
    }
    if (!node.empty() && state.edit.layers.count(node.key()) != 0)
    {
        throw std::runtime_error("Studio: cannot undo or redo a layer change: layer " + layer.id
                                 + " already has an edit state");
    }
    layers.insert(layers.begin() + index, layer);
    if (!node.empty())
    {
        state.edit.layers.insert(std::move(node));
    }
}

} // namespace

// --- LayerOpCommand ---------------------------------------------------------

LayerOpCommand::LayerOpCommand(StudioState & stateRef, std::string id, std::string text, std::size_t byteCount)
    : state(&stateRef), layerId(std::move(id)), m_label(std::move(text)), m_bytes(byteCount)
{
}

std::string LayerOpCommand::label() const
{
    return m_label;
}

std::size_t LayerOpCommand::bytes() const
{
    return m_bytes;
}

ProjectLayer & LayerOpCommand::layer()
{
    const int index = indexOfLayer(*state, layerId);
    if (index < 0)
    {
        throw missingLayer(layerId);
    }
    return state->project.layers[static_cast<size_t>(index)];
}

void LayerOpCommand::markChanged()
{
    state->dirty = true;
    state->sceneDirty = true;
}

void LayerOpCommand::setSelection(int index)
{
    const int count = static_cast<int>(state->project.layers.size());
    state->selectedLayer = std::clamp(index, -1, count - 1);
}

// --- AddLayerCommand --------------------------------------------------------

AddLayerCommand::AddLayerCommand(StudioState & stateRef, const ProjectLayer & added, int index,
                                 int selectedBefore, int selectedAfter)
    : LayerOpCommand(stateRef, added.id, "Add layer " + added.name, cloudBytes(added)),
      m_index(index),
      m_selectedBefore(selectedBefore),
      m_selectedAfter(selectedAfter)
{
    const std::vector<ProjectLayer> & layers = stateRef.project.layers;
    if (index < 0 || static_cast<size_t>(index) >= layers.size() || layers[static_cast<size_t>(index)].id != added.id)
    {
        throw std::invalid_argument("Studio: the added layer " + added.id + " is not at index "
                                    + std::to_string(index));
    }
}

void AddLayerCommand::undo()
{
    ReplayGuard replay(state->edit);
    m_held = takeLayer(*state, layerId, m_index, m_node);
    setSelection(m_selectedBefore);
    markChanged();
}

void AddLayerCommand::redo()
{
    ReplayGuard replay(state->edit);
    if (!m_held)
    {
        throw std::runtime_error("Studio: cannot redo the add of layer " + layerId + ": it is not undone");
    }
    restoreLayer(*state, *m_held, m_index, m_node);
    m_held.reset();
    setSelection(m_selectedAfter);
    markChanged();
}

// --- RemoveLayerCommand -----------------------------------------------------

RemoveLayerCommand::RemoveLayerCommand(StudioState & stateRef, ProjectLayer removed, LayerEditNode node,
                                       int index, int selectedBefore, int selectedAfter)
    : LayerOpCommand(stateRef, removed.id, "Remove layer " + removed.name, cloudBytes(removed)),
      m_index(index),
      m_selectedBefore(selectedBefore),
      m_selectedAfter(selectedAfter),
      m_held(std::move(removed)),
      m_node(std::move(node))
{
}

void RemoveLayerCommand::undo()
{
    ReplayGuard replay(state->edit);
    if (!m_held)
    {
        throw std::runtime_error("Studio: cannot undo the removal of layer " + layerId + ": it is not removed");
    }
    restoreLayer(*state, *m_held, m_index, m_node);
    m_held.reset();
    setSelection(m_selectedBefore);
    markChanged();
}

void RemoveLayerCommand::redo()
{
    ReplayGuard replay(state->edit);
    m_held = takeLayer(*state, layerId, m_index, m_node);
    setSelection(m_selectedAfter);
    markChanged();
}

// --- MoveLayerCommand -------------------------------------------------------

MoveLayerCommand::MoveLayerCommand(StudioState & stateRef, std::string id, int fromIndex, int toIndex,
                                   int selectedBefore, int selectedAfter)
    : LayerOpCommand(stateRef, id, "Move layer " + nameOfLayer(stateRef, id), 0),
      m_fromIndex(fromIndex),
      m_toIndex(toIndex),
      m_selectedBefore(selectedBefore),
      m_selectedAfter(selectedAfter)
{
}

void MoveLayerCommand::undo()
{
    ReplayGuard replay(state->edit);
    moveTo(m_fromIndex);
    setSelection(m_selectedBefore);
    markChanged();
}

void MoveLayerCommand::redo()
{
    ReplayGuard replay(state->edit);
    moveTo(m_toIndex);
    setSelection(m_selectedAfter);
    markChanged();
}

void MoveLayerCommand::moveTo(int index)
{
    std::vector<ProjectLayer> & layers = state->project.layers;
    const int from = indexOfLayer(*state, layerId);
    if (from < 0)
    {
        throw missingLayer(layerId);
    }
    if (index < 0 || static_cast<size_t>(index) >= layers.size())
    {
        throw std::runtime_error("Studio: cannot move layer " + layerId + " to index " + std::to_string(index)
                                 + ": outside the layer list");
    }
    if (from == index)
    {
        return;
    }
    ProjectLayer moved = std::move(layers[static_cast<size_t>(from)]);
    layers.erase(layers.begin() + from);
    layers.insert(layers.begin() + index, std::move(moved));
}

// --- RenameLayerCommand -----------------------------------------------------

RenameLayerCommand::RenameLayerCommand(StudioState & stateRef, std::string id, std::string before,
                                       std::string after)
    : LayerOpCommand(stateRef, id, "Rename layer", before.size() + after.size()),
      m_before(std::move(before)),
      m_after(std::move(after))
{
}

void RenameLayerCommand::undo()
{
    ReplayGuard replay(state->edit);
    layer().name = m_before;
    markChanged();
}

void RenameLayerCommand::redo()
{
    ReplayGuard replay(state->edit);
    layer().name = m_after;
    markChanged();
}

// --- LayerVisibilityCommand -------------------------------------------------

LayerVisibilityCommand::LayerVisibilityCommand(StudioState & stateRef, std::string id, bool before, bool after)
    : LayerOpCommand(stateRef, id, (after ? "Show layer " : "Hide layer ") + nameOfLayer(stateRef, id), 0),
      m_before(before),
      m_after(after)
{
}

void LayerVisibilityCommand::undo()
{
    ReplayGuard replay(state->edit);
    layer().visible = m_before;
    markChanged();
}

void LayerVisibilityCommand::redo()
{
    ReplayGuard replay(state->edit);
    layer().visible = m_after;
    markChanged();
}

// --- LayerTransformCommand --------------------------------------------------

LayerTransformCommand::LayerTransformCommand(StudioState & stateRef, std::string id, const LayerTransform & before,
                                             const LayerTransform & after)
    : LayerOpCommand(stateRef, id, "Layer transform " + nameOfLayer(stateRef, id), 0),
      m_before(before),
      m_after(after)
{
}

void LayerTransformCommand::undo()
{
    ReplayGuard replay(state->edit);
    layer().transform = m_before;
    markChanged();
}

void LayerTransformCommand::redo()
{
    ReplayGuard replay(state->edit);
    layer().transform = m_after;
    markChanged();
}

} // namespace sabercat::studio
