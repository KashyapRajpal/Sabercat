// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Section O of edit_test: the borrowed-buffer guard (ownership model B). A layer's five
// canonical parameter streams are application-owned buffers with STORAGE usage only (no
// transfer usage), for 1000 Gaussians, and they are never written: their contents are
// uninitialized. The operations that copy a layer's GPU buffers (gatherSelected,
// scatterSnapshot, deleteSelected, undoDelete, duplicateSelected, undoDuplicate) refuse
// the borrowed layer with a std::runtime_error naming "borrowed", before any GPU work
// or state change: the layer's count, capacity, storage generation and the selection
// mask are checked unchanged after each refusal. select (Sphere) and transformSelected
// (identity) only read or write in place, so they accept the borrowed layer.

#include "edit_test_common.h"

#include <glm/glm.hpp>

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "../common/shader_dir.h"
#include "sabercat/buffer.h"
#include "sabercat/buffer_view.h"
#include "sabercat/device.h"
#include "sabercat/edit_pass.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/layer_transform.h"
#include "sabercat/render_config.h"
#include "sabercat/selection_mask.h"

using namespace sabercat;

namespace edit_test
{
namespace
{

constexpr size_t kBorrowCount = 1000;
constexpr uint32_t kBorrowShDegree = 3;

// What the layer looks like from outside: the counts, the storage generation and the mask.
struct LayerState
{
    size_t count = 0;
    size_t capacity = 0;
    uint64_t generation = 0;
    std::vector<uint32_t> mask;
};

LayerState captureState(const GaussianCloud & layer, const SelectionMask & mask)
{
    LayerState state;
    state.count = layer.count();
    state.capacity = layer.capacity();
    state.generation = layer.storageGeneration();
    state.mask = mask.download(kBorrowCount);
    return state;
}

// The names of the fields that differ between `before` and `after`, leading space each;
// empty when nothing differs.
std::string stateDifference(const LayerState & before, const LayerState & after)
{
    std::string diff;
    if (before.count != after.count)
    {
        diff += " count";
    }
    if (before.capacity != after.capacity)
    {
        diff += " capacity";
    }
    if (before.generation != after.generation)
    {
        diff += " storageGeneration";
    }
    if (before.mask != after.mask)
    {
        diff += " mask";
    }
    return diff;
}

std::string lowerCase(const std::string & text)
{
    std::string out = text;
    std::transform(out.begin(), out.end(), out.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return out;
}

// Runs `fn`, which must throw a std::runtime_error whose message mentions "borrowed"
// (case-insensitive). Then checks that the layer and the mask are as `before` left them.
void expectBorrowRefusal(const std::function<void()> & fn, const std::string & label,
                         const GaussianCloud & layer, const SelectionMask & mask,
                         const LayerState & before)
{
    bool refused = false;
    std::string detail;
    try
    {
        fn();
        detail = "no exception";
    }
    catch (const std::runtime_error & e)
    {
        refused = lowerCase(e.what()).find("borrowed") != std::string::npos;
        detail = e.what();
    }
    catch (const std::exception & e)
    {
        detail = std::string("not a std::runtime_error: ") + e.what();
    }
    catch (...)
    {
        detail = "an unknown exception";
    }
    check(refused, label + ": refused with a runtime_error naming 'borrowed'"
                       + (refused ? "" : " (got: " + detail + ")"));

    const std::string diff = stateDifference(before, captureState(layer, mask));
    check(diff.empty(), label + ": count, capacity, storage generation and mask unchanged"
                            + (diff.empty() ? "" : " (changed:" + diff + ")"));
}

// Runs `fn` and records one check that it does not throw.
void expectNoThrow(const std::function<void()> & fn, const std::string & label)
{
    bool ok = true;
    std::string detail;
    try
    {
        fn();
    }
    catch (const std::exception & e)
    {
        ok = false;
        detail = e.what();
    }
    catch (...)
    {
        ok = false;
        detail = "an unknown exception";
    }
    check(ok, label + (ok ? "" : " (threw: " + detail + ")"));
}

// A Full snapshot of Gaussian 0 with well-formed values. The guard refuses before any
// content is read, so the absent relightable streams of the layer never matter here.
SparseSnapshot makeSnapshot()
{
    SparseSnapshot snap;
    snap.scope = SnapshotScope::Full;
    snap.shDegree = kBorrowShDegree;
    snap.indices = {0};
    snap.means = {glm::vec4(0.0f)};
    snap.scales = {glm::vec4(0.1f, 0.1f, 0.1f, 0.0f)};
    snap.rotations = {glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)};
    snap.shCoefficients.assign(static_cast<size_t>(kShCoefficientsPerGaussian), glm::vec4(0.0f));
    snap.opacities = {0.5f};
    snap.baseColorRoughness = {glm::vec4(0.5f, 0.5f, 0.5f, 0.5f)};
    snap.metallic = {0.0f};
    snap.visibility.assign(kVisibilityCoefficients, 1.0f);
    return snap;
}

// A selection of `count` Gaussians with the first one set, sized to the mask's words.
std::vector<uint32_t> firstBitWords(size_t count)
{
    std::vector<uint32_t> words((count + 31) / 32, 0u);
    words[0] = 1u;
    return words;
}

} // namespace

// O: the borrowed-buffer guard on a layer whose five parameter streams are borrowed.
void checkBorrowGuard(Device & device)
{
    // The application's buffers: STORAGE usage only, no transfer usage, uninitialized.
    const VkBufferUsageFlags storageOnly = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    const VkDeviceSize n = kBorrowCount;
    Buffer means(device, n * kMeansStride, storageOnly);
    Buffer scales(device, n * kScalesStride, storageOnly);
    Buffer rotations(device, n * kRotationsStride, storageOnly);
    Buffer opacity(device, n * kOpacityStride, storageOnly);
    Buffer shCoefficients(device,
                          n * static_cast<VkDeviceSize>(kShCoefficientsPerGaussian) * kShStride,
                          storageOnly);

    CanonicalGaussianBufferViews parameters;
    parameters.means = means.view(kMeansStride);
    parameters.scales = scales.view(kScalesStride);
    parameters.rotations = rotations.view(kRotationsStride);
    parameters.opacity = opacity.view(kOpacityStride);
    parameters.shCoefficients = shCoefficients.view(kShStride);
    parameters.gaussianCount = kBorrowCount;
    parameters.shDegree = kBorrowShDegree;

    GaussianCloud layer(device);
    const uint64_t generationBefore = layer.storageGeneration();
    layer.borrow(parameters, CanonicalGaussianGradientViews{}, RelightBufferViews{}, kBorrowCount);
    check(layer.borrowedParameters(), "O: borrow() leaves the parameters borrowed");
    check(layer.count() == kBorrowCount && layer.capacity() == kBorrowCount,
          "O: borrowed layer count and capacity are 1000");
    check(layer.storageGeneration() > generationBefore,
          "O: borrow() advances storageGeneration()");

    SelectionMask mask(device, kBorrowCount);
    EditPass edit(device, testShaderDir());

    // select only reads the means: a Sphere on the borrowed layer must not throw. Its result
    // is not checked (the means are uninitialized); the mask it leaves is the baseline below.
    SelectionQuery sphere;
    sphere.shape = SelectShape::Sphere;
    sphere.op = SelectOp::Replace;
    sphere.center = glm::vec3(0.0f);
    sphere.radius = 1.0f;
    expectNoThrow([&] { edit.select(layer, LayerTransform{}, sphere, mask); },
                  "O: select (Sphere) on the borrowed layer does not throw");

    // transformSelected with the identity is a no-op: no throw, nothing changes.
    const LayerState baseline = captureState(layer, mask);
    expectNoThrow([&] { edit.transformSelected(layer, mask, LayerTransform{}); },
                  "O: transformSelected (identity) on the borrowed layer does not throw");
    check(stateDifference(baseline, captureState(layer, mask)).empty(),
          "O: identity transformSelected leaves count, capacity, generation and mask unchanged");

    // The copying operations are refused, each leaving the layer and the mask as they were.
    const LayerState before = captureState(layer, mask);
    const SparseSnapshot snapshot = makeSnapshot();

    expectBorrowRefusal([&] { edit.gatherSelected(layer, mask, SnapshotScope::Full); },
                        "O: gatherSelected", layer, mask, before);
    expectBorrowRefusal([&] { edit.scatterSnapshot(layer, snapshot); }, "O: scatterSnapshot",
                        layer, mask, before);
    expectBorrowRefusal([&] { edit.deleteSelected(layer, mask); }, "O: deleteSelected", layer,
                        mask, before);

    // A non-no-op delete record: one deleted Gaussian, 1000 before and 999 kept, so the
    // early return for an empty record is not taken.
    DeleteRecord deleteRecord;
    deleteRecord.deleted = makeSnapshot();
    deleteRecord.selectionWords = firstBitWords(kBorrowCount);
    deleteRecord.oldCount = static_cast<uint32_t>(kBorrowCount);
    deleteRecord.keptCount = static_cast<uint32_t>(kBorrowCount - 1);
    expectBorrowRefusal([&] { edit.undoDelete(layer, mask, deleteRecord); }, "O: undoDelete",
                        layer, mask, before);

    expectBorrowRefusal([&] { edit.duplicateSelected(layer, mask); }, "O: duplicateSelected",
                        layer, mask, before);

    // A non-no-op duplicate record: one appended Gaussian, 999 before and 1000 now, so the
    // early return for added == 0 is not taken.
    DuplicateRecord duplicateRecord;
    duplicateRecord.selectionWords = firstBitWords(kBorrowCount - 1);
    duplicateRecord.oldCount = static_cast<uint32_t>(kBorrowCount - 1);
    duplicateRecord.added = 1;
    expectBorrowRefusal([&] { edit.undoDuplicate(layer, mask, duplicateRecord); },
                        "O: undoDuplicate", layer, mask, before);
}

} // namespace edit_test
