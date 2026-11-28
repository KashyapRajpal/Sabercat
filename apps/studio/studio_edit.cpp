// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "studio_edit.h"

#include "studio_edits.h"
#include "studio_state.h"

#include "sabercat/edit_math.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/renderer.h"

#include <glm/gtc/quaternion.hpp>

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace sabercat::studio
{

namespace
{

// The layer with `id` in the project, or nullptr.
ProjectLayer * findLayer(StudioState & state, const std::string & id)
{
    for (ProjectLayer & layer : state.project.layers)
    {
        if (layer.id == id)
        {
            return &layer;
        }
    }
    return nullptr;
}

// The edit state of the layer `id`, or nullptr when it has none.
LayerEditState * findEdit(StudioState & state, const std::string & id)
{
    const auto it = state.edit.layers.find(id);
    return it == state.edit.layers.end() ? nullptr : &it->second;
}

// The active layer, which must be editable. Otherwise throws, naming the reason.
ProjectLayer & requireEditable(StudioState & state)
{
    ProjectLayer * layer = editableLayer(state);
    if (layer == nullptr)
    {
        throw std::runtime_error(std::string(whyNotEditable(state))
                                 + ": select a visible, unlocked layer to edit it");
    }
    return *layer;
}

// The layer's Gaussian count as the kernels take it.
uint32_t countOf(const ProjectLayer & layer)
{
    return static_cast<uint32_t>(layer.cloud->count());
}

// A world edit that changes nothing: no translation, no rotation, unit scale.
bool isNoOpEdit(const TransformParams & params)
{
    return params.translation == glm::vec3(0.0f) && params.rotationDegrees == glm::vec3(0.0f)
           && params.scale == 1.0f;
}

// Throws std::invalid_argument for a parameter the transform cannot use.
void checkTransformParams(const TransformParams & params)
{
    if (!(params.scale > 0.0f) || !std::isfinite(params.scale))
    {
        throw std::invalid_argument("Studio: the transform scale must be a positive, finite number");
    }
    for (int axis = 0; axis < 3; ++axis)
    {
        if (!std::isfinite(params.translation[axis]) || !std::isfinite(params.rotationDegrees[axis]))
        {
            throw std::invalid_argument("Studio: the transform translation and rotation must be finite");
        }
    }
}

// The layer-space form of the world edit `params`, about its pivot. The pivot of a
// selection centroid is read from the layer's data, so the caller must have restored
// the pre-preview values first.
LayerTransform layerSpaceOf(EditPass & pass, const ProjectLayer & layer, LayerEditState & es,
                            const TransformParams & params)
{
    if (isNoOpEdit(params))
    {
        return LayerTransform{};
    }
    glm::dvec3 pivot(0.0);
    switch (params.pivot)
    {
    case PivotMode::SelectionCentroid:
    {
        const MaskCentroid centroid = pass.maskedCentroid(*layer.cloud, *es.mask);
        pivot = similarityOf(layer.transform).apply(centroid.centroid());
        break;
    }
    case PivotMode::LayerOrigin:
        pivot = glm::dvec3(layer.transform.translation);
        break;
    case PivotMode::WorldOrigin:
        break;
    }
    const glm::dmat3 rotation = glm::mat3_cast(glm::dquat(glm::radians(glm::dvec3(params.rotationDegrees))));
    const Similarity world = worldEditAboutPivot(rotation, static_cast<double>(params.scale),
                                                 glm::dvec3(params.translation), pivot);
    return layerSpaceEdit(layer.transform, world);
}

// Runs one selection change on the active layer. `change(pass, layer, mask, count)`
// edits the mask; the result becomes one SelectionCommand, pushed after it has run.
// A transform preview is canceled first (it is a change of the layer's data).
template <typename Change>
void changeSelection(StudioState & state, Renderer & renderer, Change && change)
{
    ProjectLayer & layer = requireEditable(state);
    cancelTransformPreview(state, renderer);
    ensureLayerGpu(state, renderer, layer);
    LayerEditState & es = editStateOf(state, layer);
    EditPass & pass = renderer.editPass();
    const uint32_t count = countOf(layer);

    std::vector<uint32_t> before = es.mask->download(count);
    change(pass, static_cast<const ProjectLayer &>(layer), *es.mask, count);
    std::vector<uint32_t> after = es.mask->download(count);
    es.selectedCount = pass.countSelected(*es.mask, count);

    // A selection is not part of the project file, so it does not mark the project dirty.
    state.edit.history.push(std::make_unique<SelectionCommand>(state, renderer, layer.id,
                                                               std::move(before), std::move(after)));
    refreshHighlight(state, renderer);
}

} // namespace

// --- StudioEdit -------------------------------------------------------------

void StudioEdit::reset()
{
    preview = TransformPreview{};
    history.clear();
    layers.clear();
    boundsExact = false;
    interacting = false;
}

// --- ReplayGuard ------------------------------------------------------------

ReplayGuard::ReplayGuard(StudioEdit & edit) : m_edit(&edit), m_previous(edit.replaying)
{
    edit.replaying = true;
}

ReplayGuard::~ReplayGuard()
{
    m_edit->replaying = m_previous;
}

// --- Active layer -----------------------------------------------------------

ProjectLayer * editableLayer(StudioState & state)
{
    if (state.selectedLayer < 0 || state.selectedLayer >= static_cast<int>(state.project.layers.size()))
    {
        return nullptr;
    }
    ProjectLayer & layer = state.project.layers[static_cast<size_t>(state.selectedLayer)];
    if (!layer.cloud || !layer.visible || layer.locked)
    {
        return nullptr;
    }
    return &layer;
}

const char * whyNotEditable(const StudioState & state)
{
    if (state.selectedLayer < 0 || state.selectedLayer >= static_cast<int>(state.project.layers.size()))
    {
        return "No layer selected";
    }
    const ProjectLayer & layer = state.project.layers[static_cast<size_t>(state.selectedLayer)];
    if (!layer.cloud)
    {
        return "The layer has no Gaussians";
    }
    if (layer.locked)
    {
        return "The layer is locked";
    }
    if (!layer.visible)
    {
        return "The layer is hidden";
    }
    return "";
}

LayerEditState & editStateOf(StudioState & state, ProjectLayer & layer)
{
    return state.edit.layers[layer.id];
}

void ensureLayerGpu(StudioState & state, Renderer & renderer, ProjectLayer & layer)
{
    if (!layer.cloud)
    {
        throw std::runtime_error("Studio: layer \"" + layer.name + "\" has no Gaussians to edit");
    }
    GaussianCloud & cloud = *layer.cloud;
    if (cloud.capacity() == 0)
    {
        cloud.uploadToGPU(); // exactly count() Gaussians
    }

    LayerEditState & es = editStateOf(state, layer);
    const size_t capacity = cloud.capacity();
    if (!es.mask)
    {
        es.mask = std::make_unique<SelectionMask>(renderer.sabercatDevice(), capacity);
        if (!cloud.bounds().valid)
        {
            cloud.computeBounds();
        }
        es.localBounds = cloud.bounds();
    }
    else if (es.mask->capacity() < capacity)
    {
        es.mask->resize(capacity);
    }
}

void syncLayerToHost(StudioState & state, ProjectLayer & layer)
{
    if (!layer.cloud)
    {
        return;
    }
    const auto it = state.edit.layers.find(layer.id);
    if (it == state.edit.layers.end() || !it->second.hostStale)
    {
        return;
    }
    GaussianCloud & cloud = *layer.cloud;
    cloud.downloadFromGPU();
    cloud.computeBounds();
    it->second.localBounds = cloud.bounds();
    it->second.hostStale = false;
}

void syncAllToHost(StudioState & state)
{
    for (ProjectLayer & layer : state.project.layers)
    {
        syncLayerToHost(state, layer);
    }
}

// --- Selection --------------------------------------------------------------

void selectBox(StudioState & state, Renderer & renderer, SelectOp op, const glm::vec3 & center,
               const glm::vec3 & halfExtents)
{
    if (!(halfExtents.x >= 0.0f && halfExtents.y >= 0.0f && halfExtents.z >= 0.0f))
    {
        throw std::invalid_argument("Studio: the box half extents must be zero or positive");
    }
    SelectionQuery query;
    query.shape = SelectShape::Box;
    query.op = op;
    query.center = center;
    query.halfExtents = halfExtents;
    changeSelection(state, renderer,
                    [&query](EditPass & pass, const ProjectLayer & target, SelectionMask & mask, uint32_t) {
                        pass.select(*target.cloud, target.transform, query, mask);
                    });
}

void selectSphere(StudioState & state, Renderer & renderer, SelectOp op, const glm::vec3 & center,
                  float radius)
{
    if (!(radius >= 0.0f))
    {
        throw std::invalid_argument("Studio: the sphere radius must be zero or positive");
    }
    SelectionQuery query;
    query.shape = SelectShape::Sphere;
    query.op = op;
    query.center = center;
    query.radius = radius;
    changeSelection(state, renderer,
                    [&query](EditPass & pass, const ProjectLayer & target, SelectionMask & mask, uint32_t) {
                        pass.select(*target.cloud, target.transform, query, mask);
                    });
}

void selectRectangle(StudioState & state, Renderer & renderer, SelectOp op, const glm::vec2 & cornerA,
                     const glm::vec2 & cornerB)
{
    // Read before anything changes: with no frame yet there is nothing to select through.
    CameraUniforms camera{};
    try
    {
        camera = renderer.lastForwardCamera();
    }
    catch (const std::runtime_error & e)
    {
        throw std::runtime_error(std::string("Studio: draw a frame before a rectangle selection (")
                                 + e.what() + ")");
    }
    SelectionQuery query;
    query.shape = SelectShape::Rectangle;
    query.op = op;
    query.rect = normalizeRect(cornerA, cornerB);
    query.camera = camera;
    changeSelection(state, renderer,
                    [&query](EditPass & pass, const ProjectLayer & target, SelectionMask & mask, uint32_t) {
                        pass.select(*target.cloud, target.transform, query, mask);
                    });
}

void selectAllGaussians(StudioState & state, Renderer & renderer)
{
    changeSelection(state, renderer,
                    [](EditPass & pass, const ProjectLayer &, SelectionMask & mask, uint32_t count) {
                        pass.selectAll(mask, count);
                    });
}

void selectNone(StudioState & state, Renderer & renderer)
{
    changeSelection(state, renderer,
                    [](EditPass & pass, const ProjectLayer &, SelectionMask & mask, uint32_t) {
                        pass.clearMask(mask);
                    });
}

void invertSelection(StudioState & state, Renderer & renderer)
{
    changeSelection(state, renderer,
                    [](EditPass & pass, const ProjectLayer &, SelectionMask & mask, uint32_t count) {
                        pass.invertMask(mask, count);
                    });
}

uint32_t selectedCountOfActiveLayer(const StudioState & state)
{
    if (state.selectedLayer < 0 || state.selectedLayer >= static_cast<int>(state.project.layers.size()))
    {
        return 0;
    }
    const auto it = state.edit.layers.find(state.project.layers[static_cast<size_t>(state.selectedLayer)].id);
    return it == state.edit.layers.end() ? 0 : it->second.selectedCount;
}

// --- Transform --------------------------------------------------------------

void previewTransform(StudioState & state, Renderer & renderer, const TransformParams & params)
{
    checkTransformParams(params);
    ProjectLayer & layer = requireEditable(state);
    TransformPreview & preview = state.edit.preview;
    if (preview.active && preview.layerId != layer.id)
    {
        // The preview is on another layer: that layer's data goes back first.
        cancelTransformPreview(state, renderer);
    }
    ensureLayerGpu(state, renderer, layer);
    LayerEditState & es = editStateOf(state, layer);
    if (es.selectedCount == 0)
    {
        throw std::runtime_error("Studio: select Gaussians before transforming them");
    }
    EditPass & pass = renderer.editPass();

    if (!preview.active)
    {
        // Gather everything first, so a throw leaves no half-started preview.
        std::vector<uint32_t> words = es.mask->download(countOf(layer));
        SparseSnapshot snapshot = pass.gatherSelected(*layer.cloud, *es.mask, SnapshotScope::Transform);
        preview.maskWords = std::move(words);
        preview.snapshot = std::move(snapshot);
        preview.layerId = layer.id;
        preview.baseBounds = es.localBounds;
        preview.hostStaleBefore = es.hostStale;
        preview.appliedLayerSpace = LayerTransform{};
        preview.renderer = &renderer;
        preview.active = true;
    }
    else if (!preview.appliedLayerSpace.isIdentity())
    {
        pass.scatterSnapshot(*layer.cloud, preview.snapshot);
        preview.appliedLayerSpace = LayerTransform{};
    }

    // After the restore, so a selection-centroid pivot is the original centroid.
    const LayerTransform layerSpace = layerSpaceOf(pass, layer, es, params);
    pass.transformSelected(*layer.cloud, *es.mask, layerSpace);
    preview.appliedLayerSpace = layerSpace;
    es.localBounds = widenedBounds(preview.baseBounds, layerSpace);
    es.hostStale = true;
    state.sceneDirty = true;
}

void cancelTransformPreview(StudioState & state)
{
    // No preview: nothing to restore, and no GPU work.
    if (!state.edit.preview.active)
    {
        return;
    }
    if (state.edit.preview.renderer == nullptr)
    {
        throw std::runtime_error("Studio: a transform preview has no renderer to restore it with");
    }
    cancelTransformPreview(state, *state.edit.preview.renderer);
}

void cancelTransformPreview(StudioState & state, Renderer & renderer)
{
    TransformPreview & preview = state.edit.preview;
    if (!preview.active)
    {
        return;
    }
    ProjectLayer * layer = findLayer(state, preview.layerId);
    LayerEditState * es = findEdit(state, preview.layerId);
    if (layer != nullptr && layer->cloud && !preview.appliedLayerSpace.isIdentity())
    {
        renderer.editPass().scatterSnapshot(*layer->cloud, preview.snapshot);
    }
    if (es != nullptr)
    {
        es->localBounds = preview.baseBounds;
        // The host is stale again if a sync copied the previewed values to it during
        // the preview; otherwise it matches the restored GPU copy, as before.
        es->hostStale = preview.hostStaleBefore || !es->hostStale;
    }
    preview = TransformPreview{};
    state.sceneDirty = true;
}

void commitTransform(StudioState & state, Renderer & renderer, const TransformParams & params)
{
    previewTransform(state, renderer, params);
    TransformPreview & preview = state.edit.preview;
    const std::string layerId = preview.layerId;
    const LayerTransform layerSpace = preview.appliedLayerSpace;
    state.edit.history.push(std::make_unique<TransformCommand>(state, renderer, layerId,
                                                               std::move(preview.maskWords),
                                                               std::move(preview.snapshot), layerSpace));
    preview = TransformPreview{};
    state.dirty = true;
}

// --- Delete and duplicate ---------------------------------------------------

bool deleteSelection(StudioState & state, Renderer & renderer, std::string & whyNot)
{
    whyNot.clear();
    ProjectLayer * target = editableLayer(state);
    if (target == nullptr)
    {
        whyNot = whyNotEditable(state);
        return false;
    }
    ProjectLayer & layer = *target;
    const LayerEditState * known = findEdit(state, layer.id);
    const uint32_t selected = known != nullptr ? known->selectedCount : 0;
    if (selected == 0)
    {
        whyNot = "Nothing is selected";
        return false;
    }
    if (static_cast<size_t>(selected) >= layer.cloud->count())
    {
        whyNot = "Every Gaussian is selected: remove the layer instead";
        return false;
    }

    cancelTransformPreview(state, renderer);
    ensureLayerGpu(state, renderer, layer);
    LayerEditState & es = editStateOf(state, layer);
    DeleteRecord record = renderer.editPass().deleteSelected(*layer.cloud, *es.mask);
    es.selectedCount = 0; // the kernel clears the selection with the Gaussians
    es.hostStale = true;
    state.dirty = true;
    state.sceneDirty = true;
    state.edit.history.push(std::make_unique<DeleteCommand>(state, renderer, layer.id, std::move(record)));
    return true;
}

bool duplicateSelection(StudioState & state, Renderer & renderer, std::string & whyNot)
{
    whyNot.clear();
    ProjectLayer * target = editableLayer(state);
    if (target == nullptr)
    {
        whyNot = whyNotEditable(state);
        return false;
    }
    ProjectLayer & layer = *target;
    const LayerEditState * known = findEdit(state, layer.id);
    if (known == nullptr || known->selectedCount == 0)
    {
        whyNot = "Nothing is selected";
        return false;
    }

    cancelTransformPreview(state, renderer);
    ensureLayerGpu(state, renderer, layer);
    LayerEditState & es = editStateOf(state, layer);
    DuplicateRecord record = renderer.editPass().duplicateSelected(*layer.cloud, *es.mask);
    // The copies are the new selection; the grown mask was resized by the kernel.
    es.selectedCount = record.added;
    es.hostStale = true;
    state.dirty = true;
    state.sceneDirty = true;
    state.edit.history.push(std::make_unique<DuplicateCommand>(state, renderer, layer.id, std::move(record)));
    return true;
}

// --- History ----------------------------------------------------------------

bool undo(StudioState & state, Renderer & renderer)
{
    if (!state.edit.history.canUndo())
    {
        return false;
    }
    cancelTransformPreview(state, renderer);
    return state.edit.history.undo();
}

bool redo(StudioState & state, Renderer & renderer)
{
    if (!state.edit.history.canRedo())
    {
        return false;
    }
    cancelTransformPreview(state, renderer);
    return state.edit.history.redo();
}

bool jumpToHistory(StudioState & state, Renderer & renderer, std::size_t index)
{
    if (index > state.edit.history.size())
    {
        return false;
    }
    if (index == state.edit.history.cursor())
    {
        return true;
    }
    cancelTransformPreview(state, renderer);
    return state.edit.history.jumpTo(index);
}

void clearEditHistory(StudioState & state)
{
    state.edit.history.clear();
}

// --- Highlight and bounds ---------------------------------------------------

void refreshHighlight(StudioState & state, Renderer & renderer)
{
    HighlightSettings off = state.edit.highlight;
    off.enabled = 0;
    // The highlight needs the renderer's merged scene of the current layers. The
    // host merge has no layer ranges to tint, so it never gets one.
    if (state.edit.highlight.enabled == 0 || state.edit.gpuMergeFailed || state.sceneDirty)
    {
        renderer.setSelectionHighlight({}, off);
        return;
    }
    std::vector<BufferView> masks;
    for (ProjectLayer & layer : state.project.layers)
    {
        if (!layer.visible || !layer.cloud)
        {
            continue;
        }
        ensureLayerGpu(state, renderer, layer);
        masks.push_back(editStateOf(state, layer).mask->view());
    }
    if (masks.empty())
    {
        renderer.setSelectionHighlight({}, off);
        return;
    }
    renderer.setSelectionHighlight(masks, state.edit.highlight);
}

SceneBounds currentSceneBounds(StudioState & state)
{
    bool anyStale = false;
    for (const ProjectLayer & layer : state.project.layers)
    {
        if (!layer.visible || !layer.cloud)
        {
            continue;
        }
        const auto it = state.edit.layers.find(layer.id);
        if (it != state.edit.layers.end() && it->second.hostStale)
        {
            anyStale = true;
            break;
        }
    }
    if (!anyStale && !state.edit.interacting)
    {
        state.edit.boundsExact = true;
        return mergedLayerBounds(visibleLayers(state));
    }

    // Conservative: the local box of each visible layer, mapped by its placement.
    state.edit.boundsExact = false;
    glm::vec3 lo(std::numeric_limits<float>::infinity());
    glm::vec3 hi(-std::numeric_limits<float>::infinity());
    bool any = false;
    for (const ProjectLayer & layer : state.project.layers)
    {
        if (!layer.visible || !layer.cloud)
        {
            continue;
        }
        const auto it = state.edit.layers.find(layer.id);
        const SceneBounds & local = it != state.edit.layers.end() ? it->second.localBounds : layer.cloud->bounds();
        if (!local.valid)
        {
            continue;
        }
        const Similarity placement = similarityOf(layer.transform);
        for (int corner = 0; corner < 8; ++corner)
        {
            const glm::dvec3 point((corner & 1) != 0 ? local.max.x : local.min.x,
                                   (corner & 2) != 0 ? local.max.y : local.min.y,
                                   (corner & 4) != 0 ? local.max.z : local.min.z);
            const glm::vec3 world(placement.apply(point));
            lo = glm::min(lo, world);
            hi = glm::max(hi, world);
        }
        any = true;
    }

    SceneBounds out;
    if (!any)
    {
        return out;
    }
    out.min = lo;
    out.max = hi;
    out.valid = true;
    out.focusCenter = out.center();
    out.focusRadius = 0.5f * out.radius();
    return out;
}

void refreshExactBounds(StudioState & state, Renderer & renderer)
{
    bool anyVisible = false;
    for (ProjectLayer & layer : state.project.layers)
    {
        if (layer.visible && layer.cloud)
        {
            anyVisible = true;
            syncLayerToHost(state, layer);
        }
    }
    if (!anyVisible)
    {
        return;
    }
    renderer.setSceneBounds(mergedLayerBounds(visibleLayers(state)));
    state.edit.boundsExact = true;
}

SceneBounds widenedBounds(const SceneBounds & box, const LayerTransform & layerSpace)
{
    if (!box.valid)
    {
        return box;
    }
    const Similarity map = similarityOf(layerSpace);
    glm::vec3 lo = box.min;
    glm::vec3 hi = box.max;
    for (int corner = 0; corner < 8; ++corner)
    {
        const glm::dvec3 point((corner & 1) != 0 ? box.max.x : box.min.x,
                               (corner & 2) != 0 ? box.max.y : box.min.y,
                               (corner & 4) != 0 ? box.max.z : box.min.z);
        const glm::vec3 mapped(map.apply(point));
        lo = glm::min(lo, mapped);
        hi = glm::max(hi, mapped);
    }
    SceneBounds out = box;
    out.min = lo;
    out.max = hi;
    return out;
}

} // namespace sabercat::studio
