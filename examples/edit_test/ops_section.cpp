// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Sections L and M of edit_test: the C4 stream edits of EditPass, and a seeded random
// sequence of the selection and edit ops with undo and redo.
//
//  L. Each check runs on fresh GPU copies of layer A (100,003 Gaussians, SH3, all relight
//     streams) and layer B (4,099, SH1, no relight streams). The host originals are kept
//     and never changed.
//     L1 transformSelected: the selected Gaussians against the host transformedCopy of the
//        selected subset, with the merge tolerances; every other Gaussian bit-exact; the
//        transform of every Gaussian equal to mergeLayers of the same placement, bit for
//        bit; the identity transform and the empty mask change nothing.
//     L2 gatherSelected and scatterSnapshot: the Transform and Full scopes, the Full copy
//        against the host subset, bytes(), and the restore of a transform.
//     L3 deleteSelected and undoDelete: the remap, the record, the kept order, the undo and
//        the redo; the edge indices 0, n-1, 31, 32 and 33; an empty selection; a selection of
//        every Gaussian, refused.
//     L4 duplicateSelected and undoDuplicate: growth (capacity == count), spare capacity, the
//        undo and the redo; an empty selection.
//  M. Two seeded runs of 200 random ops (select, transform, duplicate, delete). Every op is
//     undone newest first, then all are redone; the state hash is checked after each step
//     and the final streams are compared. Run 1 (20,011 Gaussians, SH3, relight) lives on a
//     Renderer's device and, once every op is undone, is merged by the renderer and read
//     back. Run 2 is layer B.

#include "edit_test_common.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sabercat/buffer.h"
#include "sabercat/buffer_view.h"
#include "sabercat/device.h"
#include "sabercat/edit_math.h"
#include "sabercat/edit_pass.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/layer_transform.h"
#include "sabercat/renderer.h"
#include "sabercat/selection_mask.h"

#include "../common/shader_dir.h"

using namespace sabercat;
using namespace edit_test;

namespace edit_test
{
namespace
{

constexpr size_t kCountA = 100003;
constexpr size_t kCountB = 4099;
constexpr size_t kCountRun1 = 20011;
constexpr size_t kOpsPerRun = 200;
constexpr size_t kMaxCountToDuplicate = 60000;
constexpr uint32_t kUintStride = sizeof(uint32_t);
constexpr uint32_t kNoRemap = 0xFFFFFFFFu;
constexpr uint32_t kWidth = 128;
constexpr uint32_t kHeight = 96;
constexpr double kPi = 3.14159265358979323846;

// A layer under test: the host original (never changed) and its label.
struct Fixture
{
    const GaussianCloud * cloud;
    const char * name;
};

// One entry of the M state stack: how to undo and redo an op, and the state hash after it.
struct StackEntry
{
    std::function<void()> undo;
    std::function<void()> redo;
    uint64_t hashAfter = 0;
};

// How many ops of each kind a run executed.
struct OpCounts
{
    size_t selects = 0;
    size_t transforms = 0;
    size_t duplicates = 0;
    size_t growths = 0;
    size_t deletes = 0;
};

// FNV-1a 64 over little-endian 64-bit lanes, with the tail bytes folded in one at a time.
// Lane-wise rather than byte-wise, so the M hashes stay affordable in a Debug build.
struct Fnv1a64
{
    static constexpr uint64_t kOffset = 14695981039346656037ull;
    static constexpr uint64_t kPrime = 1099511628211ull;
    uint64_t value = kOffset;

    void add(const void * data, size_t size)
    {
        const unsigned char * p = static_cast<const unsigned char *>(data);
        size_t i = 0;
        for (; i + sizeof(uint64_t) <= size; i += sizeof(uint64_t))
        {
            uint64_t lane = 0;
            std::memcpy(&lane, p + i, sizeof lane);
            value = (value ^ lane) * kPrime;
        }
        for (; i < size; ++i)
        {
            value = (value ^ p[i]) * kPrime;
        }
    }

    template <typename T>
    void addVector(const std::vector<T> & v)
    {
        add(v.data(), v.size() * sizeof(T));
    }
};

// Runs `fn`; a throw fails the check `label` and the rest of the section still runs.
template <typename Fn>
void guard(const std::string & label, Fn && fn)
{
    try
    {
        fn();
    }
    catch (const std::exception & e)
    {
        check(false, label + " threw: " + e.what());
    }
}

// True when `fn` throws a std::exception.
template <typename Fn>
bool throwsException(Fn && fn)
{
    try
    {
        fn();
    }
    catch (const std::exception &)
    {
        return true;
    }
    return false;
}

// --- Host helpers -----------------------------------------------------------------------

RelightPresence presenceOf(const GaussianCloud & cloud)
{
    RelightPresence p;
    p.normals = !cloud.normals().empty();
    p.baseColorRoughness = !cloud.baseColorRoughness().empty();
    p.metallic = !cloud.metallic().empty();
    p.visibility = !cloud.visibility().empty();
    return p;
}

// A fresh GPU copy of `source`'s host streams, uploaded with `capacity` (0: capacity ==
// count). The source is not changed.
std::unique_ptr<GaussianCloud> makeGpuCopy(Device & device, const GaussianCloud & source,
                                           size_t capacity)
{
    auto cloud = std::make_unique<GaussianCloud>(device);
    cloud->initialize(source.count());
    cloud->positions() = source.positions();
    cloud->scales() = source.scales();
    cloud->rotations() = source.rotations();
    cloud->shCoefficients() = source.shCoefficients();
    cloud->opacities() = source.opacities();
    cloud->normals() = source.normals();
    cloud->baseColorRoughness() = source.baseColorRoughness();
    cloud->metallic() = source.metallic();
    cloud->visibility() = source.visibility();
    cloud->setShDegree(source.shDegree());
    cloud->setAntialiased(source.antialiased());
    cloud->setBounds(source.bounds());
    cloud->uploadToGPU(capacity);
    return cloud;
}

// The host copy of a GPU layer, after a download.
Streams readBack(GaussianCloud & layer)
{
    layer.downloadFromGPU();
    return copyStreams(layer);
}

// The streams of a Full snapshot, in the Streams layout (absent streams empty).
Streams streamsOf(const SparseSnapshot & s)
{
    Streams out;
    out.positions = s.means;
    out.scales = s.scales;
    out.rotations = s.rotations;
    out.shCoefficients = s.shCoefficients;
    out.opacities = s.opacities;
    out.normals = s.normals;
    out.baseColorRoughness = s.baseColorRoughness;
    out.metallic = s.metallic;
    out.visibility = s.visibility;
    return out;
}

// The elements `indices` of one stream with `n` Gaussians, in order. An absent stream stays
// empty; the element stride is the stream's size over n (16 for SH and visibility).
template <typename T>
std::vector<T> pickStream(const std::vector<T> & src, size_t n,
                            const std::vector<uint32_t> & indices)
{
    std::vector<T> out;
    if (src.empty())
    {
        return out;
    }
    const size_t stride = src.size() / n;
    out.reserve(indices.size() * stride);
    for (const uint32_t i : indices)
    {
        const T * first = src.data() + static_cast<size_t>(i) * stride;
        out.insert(out.end(), first, first + stride);
    }
    return out;
}

Streams pickElements(const Streams & s, size_t n, const std::vector<uint32_t> & indices)
{
    Streams out;
    out.positions = pickStream(s.positions, n, indices);
    out.scales = pickStream(s.scales, n, indices);
    out.rotations = pickStream(s.rotations, n, indices);
    out.shCoefficients = pickStream(s.shCoefficients, n, indices);
    out.opacities = pickStream(s.opacities, n, indices);
    out.normals = pickStream(s.normals, n, indices);
    out.baseColorRoughness = pickStream(s.baseColorRoughness, n, indices);
    out.metallic = pickStream(s.metallic, n, indices);
    out.visibility = pickStream(s.visibility, n, indices);
    return out;
}

template <typename T>
std::vector<T> concatStream(const std::vector<T> & a, const std::vector<T> & b)
{
    std::vector<T> out(a);
    out.insert(out.end(), b.begin(), b.end());
    return out;
}

// `a` followed by `b`, stream by stream.
Streams concatStreams(const Streams & a, const Streams & b)
{
    Streams out;
    out.positions = concatStream(a.positions, b.positions);
    out.scales = concatStream(a.scales, b.scales);
    out.rotations = concatStream(a.rotations, b.rotations);
    out.shCoefficients = concatStream(a.shCoefficients, b.shCoefficients);
    out.opacities = concatStream(a.opacities, b.opacities);
    out.normals = concatStream(a.normals, b.normals);
    out.baseColorRoughness = concatStream(a.baseColorRoughness, b.baseColorRoughness);
    out.metallic = concatStream(a.metallic, b.metallic);
    out.visibility = concatStream(a.visibility, b.visibility);
    return out;
}

// One check for the whole set of streams: bit-equal in every stream, or the names of the
// streams that differ.
void checkBitEqualStreams(const Streams & want, const Streams & got, const std::string & label)
{
    std::string differ;
    const auto note = [&differ](bool same, const char * name)
    {
        if (!same)
        {
            differ += differ.empty() ? "" : ", ";
            differ += name;
        }
    };
    note(bitEqual(want.positions, got.positions), "positions");
    note(bitEqual(want.scales, got.scales), "scales");
    note(bitEqual(want.rotations, got.rotations), "rotations");
    note(bitEqual(want.shCoefficients, got.shCoefficients), "shCoefficients");
    note(bitEqual(want.opacities, got.opacities), "opacities");
    note(bitEqual(want.normals, got.normals), "normals");
    note(bitEqual(want.baseColorRoughness, got.baseColorRoughness), "baseColorRoughness");
    note(bitEqual(want.metallic, got.metallic), "metallic");
    note(bitEqual(want.visibility, got.visibility), "visibility");
    const std::string verdict =
        differ.empty() ? std::string("bit-exact in every stream") : "differs in " + differ;
    check(differ.empty(), label + ": " + verdict);
}

// The set bits of `words` below `count`, ascending.
std::vector<uint32_t> selectedIndices(const std::vector<uint32_t> & words, size_t count)
{
    std::vector<uint32_t> out;
    for (size_t i = 0; i < count; ++i)
    {
        if (((words[i >> 5] >> (i & 31u)) & 1u) != 0u)
        {
            out.push_back(static_cast<uint32_t>(i));
        }
    }
    return out;
}

// The ceil(count / 32) words with the given indices set.
std::vector<uint32_t> packIndices(const std::vector<uint32_t> & indices, size_t count)
{
    std::vector<uint32_t> words((count + 31) / 32, 0u);
    for (const uint32_t i : indices)
    {
        words[i >> 5] |= 1u << (i & 31u);
    }
    return words;
}

// The indices in [from, to), ascending.
std::vector<uint32_t> rangeIndices(size_t from, size_t to)
{
    std::vector<uint32_t> out;
    for (size_t i = from; i < to; ++i)
    {
        out.push_back(static_cast<uint32_t>(i));
    }
    return out;
}

// The indices below `count` that are not in `selected`, ascending.
std::vector<uint32_t> complementIndices(const std::vector<uint32_t> & selected, size_t count)
{
    std::vector<uint8_t> isSelected(count, 0);
    for (const uint32_t i : selected)
    {
        isSelected[i] = 1;
    }
    std::vector<uint32_t> out;
    for (size_t i = 0; i < count; ++i)
    {
        if (isSelected[i] == 0)
        {
            out.push_back(static_cast<uint32_t>(i));
        }
    }
    return out;
}

// Set bits at or beyond `count` in `words`.
size_t bitsBeyond(const std::vector<uint32_t> & words, size_t count)
{
    size_t total = 0;
    for (size_t i = count; i < words.size() * 32; ++i)
    {
        if (((words[i >> 5] >> (i & 31u)) & 1u) != 0u)
        {
            ++total;
        }
    }
    return total;
}

bool allZero(const std::vector<uint32_t> & words)
{
    return std::all_of(words.begin(), words.end(), [](uint32_t w) { return w == 0u; });
}

// --- Queries -----------------------------------------------------------------------------

SelectionQuery boxOnAxes(glm::vec3 center, const glm::mat3 & axes, glm::vec3 halfExtents,
                         SelectOp op)
{
    SelectionQuery q;
    q.shape = SelectShape::Box;
    q.op = op;
    q.center = center;
    q.halfExtents = halfExtents;
    q.boxAxisX = axes[0];
    q.boxAxisY = axes[1];
    q.boxAxisZ = axes[2];
    return q;
}

SelectionQuery sphereAt(glm::vec3 center, float radius, SelectOp op)
{
    SelectionQuery q;
    q.shape = SelectShape::Sphere;
    q.op = op;
    q.center = center;
    q.radius = radius;
    return q;
}

// The section's box: a tilted box in the middle of the cloud, a few percent of Gaussians.
SelectionQuery defaultBox()
{
    const glm::mat3 axes =
        glm::mat3_cast(glm::angleAxis(0.4f, glm::normalize(glm::vec3(1.0f, 2.0f, 0.5f))));
    return boxOnAxes(glm::vec3(0.5f, -0.3f, 0.2f), axes, glm::vec3(2.0f, 1.5f, 3.0f),
                     SelectOp::Replace);
}

// Selects `query` on the identity placement of `layer` into `mask`; returns the selected
// indices.
std::vector<uint32_t> selectOn(EditPass & edit, const GaussianCloud & layer, SelectionMask & mask,
                               const SelectionQuery & query)
{
    const size_t n = layer.count();
    edit.select(layer, identityTransform(), query, mask);
    return selectedIndices(mask.download(n), n);
}

// The indices `query` selects on a fresh GPU copy of `orig` (the copy is discarded).
std::vector<uint32_t> selectionOfCopy(Device & device, EditPass & edit, const GaussianCloud & orig,
                                      const SelectionQuery & query)
{
    std::unique_ptr<GaussianCloud> copy = makeGpuCopy(device, orig, 0);
    SelectionMask mask(device, orig.count());
    return selectOn(edit, *copy, mask, query);
}

// The host subset of a Full selection, as a cloud for transformedCopy (no GPU upload).
std::shared_ptr<GaussianCloud> makeSubsetCloud(Device & device, const GaussianCloud & orig,
                                               const Streams & picked)
{
    auto sub = std::make_shared<GaussianCloud>(device);
    sub->initialize(picked.positions.size());
    sub->positions() = picked.positions;
    sub->scales() = picked.scales;
    sub->rotations() = picked.rotations;
    sub->shCoefficients() = picked.shCoefficients;
    sub->opacities() = picked.opacities;
    sub->normals() = picked.normals;
    sub->baseColorRoughness() = picked.baseColorRoughness;
    sub->metallic() = picked.metallic;
    sub->visibility() = picked.visibility;
    sub->setShDegree(orig.shDegree());
    sub->setAntialiased(orig.antialiased());
    return sub;
}

// The world edit of L1: rotation 35 degrees about (1, 2, 3).
glm::dmat3 rotation35()
{
    return glm::mat3_cast(
        glm::angleAxis(glm::radians(35.0), glm::normalize(glm::dvec3(1.0, 2.0, 3.0))));
}

// --- L1: transform -----------------------------------------------------------------------

void checkTransform(Device & device, EditPass & edit, const Fixture & f)
{
    const GaussianCloud & orig = *f.cloud;
    const std::string label = std::string("L: ") + f.name + " transform";
    const size_t n = orig.count();
    const Streams original = copyStreams(orig);

    std::unique_ptr<GaussianCloud> layerPtr = makeGpuCopy(device, orig, 0);
    GaussianCloud & layer = *layerPtr;
    SelectionMask mask(device, n);
    const std::vector<uint32_t> sel = selectOn(edit, layer, mask, defaultBox());
    check(!sel.empty() && sel.size() < n,
          label + ": the box selects some but not all (" + std::to_string(sel.size()) + ")");
    if (sel.empty() || sel.size() == n)
    {
        return;
    }

    // The layer sits at T1 in the world, so the world pivot is the selection's centroid placed
    // by T1 (maskedCentroid is in layer-local space).
    const LayerTransform placement = transformT1();
    const glm::dvec3 pivot =
        similarityOf(placement).apply(edit.maskedCentroid(layer, mask).centroid());
    const Similarity world =
        worldEditAboutPivot(rotation35(), 1.3, glm::dvec3(0.5, -1.0, 2.0), pivot);
    const LayerTransform layerSpace = layerSpaceEdit(placement, world);

    // The host reference: the selected subset, transformed by the host transformedCopy.
    const Streams picked = pickElements(original, n, sel);
    const std::shared_ptr<GaussianCloud> subset = makeSubsetCloud(device, orig, picked);
    const std::shared_ptr<GaussianCloud> reference = transformedCopy(device, *subset, layerSpace);

    edit.transformSelected(layer, mask, layerSpace);
    const Streams got = readBack(layer);
    check(layer.count() == n && got.positions.size() == n, label + ": count unchanged");

    LayerSpan span;
    span.first = 0;
    span.count = sel.size();
    span.cloud = subset.get();
    span.transform = layerSpace;
    const std::vector<LayerSpan> spans{span};
    compareMergedStreams(spans, copyStreams(*reference), pickElements(got, n, sel), false,
                         label + " selected");

    const std::vector<uint32_t> rest = complementIndices(sel, n);
    checkBitEqualStreams(pickElements(original, n, rest), pickElements(got, n, rest),
                         label + " unselected Gaussians");
    check(bitEqual(original.opacities, got.opacities)
              && bitEqual(original.baseColorRoughness, got.baseColorRoughness)
              && bitEqual(original.metallic, got.metallic)
              && bitEqual(original.visibility, got.visibility),
          label + ": opacities, baseColorRoughness, metallic and visibility of every Gaussian "
                  "bit-exact");

    // Every Gaussian selected: the transform is the merge of the same placement, bit for bit.
    {
        std::unique_ptr<GaussianCloud> allLayer = makeGpuCopy(device, orig, 0);
        SelectionMask all(device, n);
        edit.selectAll(all, static_cast<uint32_t>(n));
        edit.transformSelected(*allLayer, all, layerSpace);
        const Streams transformedAll = readBack(*allLayer);

        std::unique_ptr<GaussianCloud> source = makeGpuCopy(device, orig, 0);
        GaussianCloud destination(device);
        destination.allocateGPU(n, 0, presenceOf(orig));
        const std::vector<MergeLayer> layers{MergeLayer(source.get(), layerSpace)};
        edit.mergeLayers(layers, destination);
        checkBitEqualStreams(readBack(destination), transformedAll,
                             label + " every Gaussian selected equals mergeLayers of the same "
                                     "placement");
    }

    // The identity transform with every Gaussian selected: a no-op.
    {
        std::unique_ptr<GaussianCloud> same = makeGpuCopy(device, orig, 0);
        SelectionMask all(device, n);
        edit.selectAll(all, static_cast<uint32_t>(n));
        edit.transformSelected(*same, all, identityTransform());
        checkBitEqualStreams(original, readBack(*same), label + " identity transform is a no-op");
    }

    // The empty mask: a no-op.
    {
        std::unique_ptr<GaussianCloud> none = makeGpuCopy(device, orig, 0);
        SelectionMask empty(device, n);
        edit.transformSelected(*none, empty, layerSpace);
        checkBitEqualStreams(original, readBack(*none), label + " empty mask is a no-op");
    }
}

// --- L2: sparse snapshot -----------------------------------------------------------------

void checkSnapshot(Device & device, EditPass & edit, const Fixture & f)
{
    const GaussianCloud & orig = *f.cloud;
    const std::string label = std::string("L: ") + f.name + " snapshot";
    const size_t n = orig.count();
    const Streams original = copyStreams(orig);

    std::unique_ptr<GaussianCloud> layerPtr = makeGpuCopy(device, orig, 0);
    GaussianCloud & layer = *layerPtr;
    SelectionMask mask(device, n);
    const std::vector<uint32_t> sel = selectOn(edit, layer, mask, defaultBox());
    check(!sel.empty(), label + ": the box selects something");
    if (sel.empty())
    {
        return;
    }
    const Streams picked = pickElements(original, n, sel);
    const std::vector<uint32_t> maskWords = mask.download(n);

    // Transform scope: the streams a transform rewrites, and nothing else.
    const SparseSnapshot transformScope =
        edit.gatherSelected(layer, mask, SnapshotScope::Transform);
    check(transformScope.scope == SnapshotScope::Transform && transformScope.indices == sel,
          label + " Transform: indices are the selected indices, ascending");
    check(transformScope.opacities.empty() && transformScope.baseColorRoughness.empty()
              && transformScope.metallic.empty() && transformScope.visibility.empty(),
          label + " Transform: no opacities, baseColorRoughness, metallic or visibility");
    Streams transformWant = picked;
    transformWant.opacities.clear();
    transformWant.baseColorRoughness.clear();
    transformWant.metallic.clear();
    transformWant.visibility.clear();
    checkBitEqualStreams(transformWant, streamsOf(transformScope),
                         label + " Transform: the selected values");

    // Full scope: every stream the layer has.
    const SparseSnapshot full = edit.gatherSelected(layer, mask, SnapshotScope::Full);
    check(full.scope == SnapshotScope::Full && full.indices == sel && full.count() == sel.size(),
          label + " Full: indices are the selected indices, ascending");
    checkBitEqualStreams(picked, streamsOf(full), label + " Full: the selected values");
    const size_t bytesWanted = full.indices.size() * sizeof(uint32_t)
                               + full.means.size() * sizeof(glm::vec4)
                               + full.scales.size() * sizeof(glm::vec4)
                               + full.rotations.size() * sizeof(glm::vec4)
                               + full.shCoefficients.size() * sizeof(glm::vec4)
                               + full.opacities.size() * sizeof(float)
                               + full.normals.size() * sizeof(glm::vec4)
                               + full.baseColorRoughness.size() * sizeof(glm::vec4)
                               + full.metallic.size() * sizeof(float)
                               + full.visibility.size() * sizeof(float);
    check(full.bytes() == bytesWanted && bytesWanted > 0,
          label + " Full: bytes() is the sum of the vector payloads");

    // A transform moves the selection; the restore of the Transform snapshot puts it back.
    edit.transformSelected(layer, mask, transformT2());
    const Streams moved = readBack(layer);
    check(!bitEqual(original.positions, moved.positions),
          label + ": the transform moved the selected Gaussians");
    check(mask.download(n) == maskWords, label + ": the transform leaves the selection unchanged");
    edit.scatterSnapshot(layer, transformScope);
    checkBitEqualStreams(original, readBack(layer), label + " restore after a transform");
}

// --- L3: delete --------------------------------------------------------------------------

// Deletes `sel` from a fresh copy of `orig` (selected with `query` when given, else uploaded),
// and checks the delete, the remap, the record, the undo and the redo.
void checkDeleteCase(Device & device, EditPass & edit, const GaussianCloud & orig,
                     const std::string & label, const std::vector<uint32_t> & sel,
                     const SelectionQuery * query)
{
    const size_t n = orig.count();
    const Streams original = copyStreams(orig);
    std::unique_ptr<GaussianCloud> layerPtr = makeGpuCopy(device, orig, 0);
    GaussianCloud & layer = *layerPtr;
    SelectionMask mask(device, n);
    if (query != nullptr)
    {
        check(selectOn(edit, layer, mask, *query) == sel,
              label + ": the GPU selection equals the expected indices");
    }
    else
    {
        mask.upload(packIndices(sel, n));
    }
    const std::vector<uint32_t> selWords = mask.download(n);
    const std::vector<uint32_t> rest = complementIndices(sel, n);
    const uint32_t kept = static_cast<uint32_t>(rest.size());

    Buffer remapBuffer(device, static_cast<VkDeviceSize>(n) * kUintStride);
    const BufferView remapView = remapBuffer.view(kUintStride);
    const DeleteRecord record = edit.deleteSelected(layer, mask, &remapView);
    check(record.oldCount == n && record.keptCount == kept && layer.count() == kept,
          label + ": delete keeps " + std::to_string(kept) + " of " + std::to_string(n));
    check(record.deleted.indices == sel && record.selectionWords == selWords,
          label + ": the record holds the deleted indices and the pre-delete selection words");
    check(allZero(mask.download(n)), label + ": the mask is all zero after the delete");

    std::vector<uint32_t> wantRemap(n, kNoRemap);
    for (size_t j = 0; j < rest.size(); ++j)
    {
        wantRemap[rest[j]] = static_cast<uint32_t>(j);
    }
    std::vector<uint32_t> gotRemap(n, 0u);
    remapBuffer.download(gotRemap.data(), static_cast<size_t>(n) * kUintStride);
    check(gotRemap == wantRemap,
          label + ": the remap holds each kept Gaussian's new index and 0xFFFFFFFF for each "
                  "deleted one");

    const Streams picked = pickElements(original, n, sel);
    checkBitEqualStreams(picked, streamsOf(record.deleted),
                         label + " record.deleted equals the host subset");
    const Streams afterDelete = readBack(layer);
    checkBitEqualStreams(pickElements(original, n, rest), afterDelete,
                         label + " kept Gaussians, in order");

    edit.undoDelete(layer, mask, record);
    check(layer.count() == n, label + " undo: count restored");
    checkBitEqualStreams(original, readBack(layer), label + " undo");
    check(mask.download(n) == selWords, label + " undo: selection words restored");

    // Redo: the selection the delete was made with, then the delete again.
    mask.upload(record.selectionWords);
    const DeleteRecord again = edit.deleteSelected(layer, mask, nullptr);
    check(again.keptCount == kept && again.deleted.indices == sel,
          label + " redo: the same delete (kept count and deleted indices)");
    checkBitEqualStreams(afterDelete, readBack(layer), label + " redo");
}

// An empty selection deletes nothing, and its undo changes nothing.
void checkDeleteEmpty(Device & device, EditPass & edit, const GaussianCloud & orig,
                      const std::string & label)
{
    const size_t n = orig.count();
    const Streams original = copyStreams(orig);
    std::unique_ptr<GaussianCloud> layerPtr = makeGpuCopy(device, orig, 0);
    GaussianCloud & layer = *layerPtr;
    SelectionMask mask(device, n);
    const DeleteRecord record = edit.deleteSelected(layer, mask, nullptr);
    check(record.deleted.count() == 0 && layer.count() == n,
          label + ": an empty selection deletes nothing");
    edit.undoDelete(layer, mask, record);
    check(layer.count() == n, label + ": undo of an empty delete keeps the count");
    checkBitEqualStreams(original, readBack(layer), label + " state after an empty delete");
}

// Selecting every Gaussian throws, and the layer and the selection are unchanged.
void checkDeleteAll(Device & device, EditPass & edit, const GaussianCloud & orig,
                    const std::string & label)
{
    const size_t n = orig.count();
    const Streams original = copyStreams(orig);
    std::unique_ptr<GaussianCloud> layerPtr = makeGpuCopy(device, orig, 0);
    GaussianCloud & layer = *layerPtr;
    SelectionMask mask(device, n);
    edit.selectAll(mask, static_cast<uint32_t>(n));
    const std::vector<uint32_t> allWords = mask.download(n);
    check(throwsException([&] { edit.deleteSelected(layer, mask, nullptr); }),
          label + ": selecting every Gaussian throws");
    check(layer.count() == n && mask.download(n) == allWords,
          label + ": the refused delete leaves the count and the selection unchanged");
    checkBitEqualStreams(original, readBack(layer), label + " state after the refused delete");
}

// --- L4: duplicate -----------------------------------------------------------------------

// Duplicates a box selection on a copy of `orig` with capacity count + spare, and checks the
// growth (spare == 0 grows), the appended copies, the selection, the undo and the redo.
void checkDuplicate(Device & device, EditPass & edit, const GaussianCloud & orig,
                    const std::string & label, size_t spare)
{
    const size_t n = orig.count();
    const Streams original = copyStreams(orig);
    std::unique_ptr<GaussianCloud> layerPtr = makeGpuCopy(device, orig, n + spare);
    GaussianCloud & layer = *layerPtr;
    SelectionMask mask(device, layer.capacity());
    const std::vector<uint32_t> sel = selectOn(edit, layer, mask, defaultBox());
    check(!sel.empty(), label + ": the box selects something");
    if (sel.empty())
    {
        return;
    }
    const std::vector<uint32_t> selWords = mask.download(n);
    const size_t k = sel.size();
    const size_t m = n + k;
    const size_t oldCapacity = layer.capacity();
    const uint64_t oldGeneration = layer.storageGeneration();

    const DuplicateRecord record = edit.duplicateSelected(layer, mask);
    check(record.oldCount == n && record.added == k && record.selectionWords == selWords,
          label + ": the record holds the old count, the added count and the selection");
    check(layer.count() == m, label + ": count is " + std::to_string(m));
    if (spare == 0)
    {
        check(layer.capacity() >= oldCapacity + oldCapacity / 2
                  && layer.storageGeneration() > oldGeneration,
              label + ": capacity grew to " + std::to_string(layer.capacity())
                  + " (at least 1.5x the old capacity, floor rule) and the storage generation "
                    "advanced");
        check(layer.capacity() >= n + k,
              label + ": the new capacity holds count + k (" + std::to_string(n + k) + ")");
    }
    else
    {
        check(layer.capacity() == oldCapacity && layer.storageGeneration() == oldGeneration,
              label + ": spare capacity, so no growth and no new storage generation");
    }
    check(mask.capacity() >= layer.capacity(),
          label + ": the mask holds the layer's capacity");

    const Streams grown = readBack(layer);
    checkBitEqualStreams(concatStreams(original, pickElements(original, n, sel)), grown,
                         label + " duplicate: the first n unchanged and the copies appended");
    const std::vector<uint32_t> wantWords = packIndices(rangeIndices(n, m), m);
    check(mask.download(m) == wantWords && bitsBeyond(mask.download(mask.capacity()), m) == 0,
          label + ": the selection is exactly [n, n + k), with no bits beyond the count");

    const size_t grownCapacity = layer.capacity();
    edit.undoDuplicate(layer, mask, record);
    check(layer.count() == n && layer.capacity() == grownCapacity,
          label + " undo: count back to n, capacity kept");
    checkBitEqualStreams(original, readBack(layer), label + " undo");
    check(mask.download(n) == selWords, label + " undo: selection words restored");

    // Redo: the selection the duplicate was made with, then the duplicate again.
    mask.upload(record.selectionWords);
    const DuplicateRecord again = edit.duplicateSelected(layer, mask);
    check(again.added == k && layer.count() == m && layer.capacity() == grownCapacity,
          label + " redo: the same added count, no further growth");
    checkBitEqualStreams(grown, readBack(layer), label + " redo");
    check(mask.download(m) == wantWords, label + " redo: the selection is [n, n + k) again");
}

// An empty selection duplicates nothing.
void checkDuplicateEmpty(Device & device, EditPass & edit, const GaussianCloud & orig,
                         const std::string & label)
{
    const size_t n = orig.count();
    const Streams original = copyStreams(orig);
    std::unique_ptr<GaussianCloud> layerPtr = makeGpuCopy(device, orig, 0);
    GaussianCloud & layer = *layerPtr;
    SelectionMask mask(device, n);
    const uint64_t generation = layer.storageGeneration();
    const DuplicateRecord record = edit.duplicateSelected(layer, mask);
    check(record.added == 0 && layer.count() == n && layer.storageGeneration() == generation,
          label + ": an empty selection duplicates nothing and grows nothing");
    edit.undoDuplicate(layer, mask, record);
    checkBitEqualStreams(original, readBack(layer), label + " state after an empty duplicate");
}

// --- M: random op sequence ---------------------------------------------------------------

// The state hash of M: the count, every stream over [0, count) and the mask words of
// ceil(count / 32). The layer's host streams are downloaded first.
uint64_t hashState(GaussianCloud & layer, const SelectionMask & mask)
{
    layer.downloadFromGPU();
    const size_t n = layer.count();
    Fnv1a64 h;
    const uint64_t count = n;
    h.add(&count, sizeof count);
    h.addVector(layer.positions());
    h.addVector(layer.scales());
    h.addVector(layer.rotations());
    h.addVector(layer.shCoefficients());
    h.addVector(layer.opacities());
    h.addVector(layer.normals());
    h.addVector(layer.baseColorRoughness());
    h.addVector(layer.metallic());
    h.addVector(layer.visibility());
    h.addVector(mask.download(n));
    return h.value;
}

glm::vec3 randomAxis(XorShift32 & rng)
{
    const glm::vec3 a = signedVec3(rng);
    const float length = glm::length(a);
    return length > 1e-3f ? a / length : glm::vec3(0.0f, 0.0f, 1.0f);
}

glm::quat randomRotation(XorShift32 & rng)
{
    const float angle = (2.0f * rng.unit() - 1.0f) * static_cast<float>(kPi);
    return glm::angleAxis(angle, randomAxis(rng));
}

glm::dmat3 randomRotationD(XorShift32 & rng)
{
    const double angle = (2.0 * static_cast<double>(rng.unit()) - 1.0) * kPi;
    return glm::mat3_cast(glm::angleAxis(angle, glm::dvec3(randomAxis(rng))));
}

// A layer placement: rotation, translation within +-2, scale in [0.7, 1.4].
LayerTransform randomPlacement(XorShift32 & rng)
{
    LayerTransform t;
    t.rotation = randomRotation(rng);
    t.translation = 2.0f * signedVec3(rng);
    t.scale = 0.7f + 0.7f * rng.unit();
    return t;
}

// A random select query over the layer's current host positions. Half the queries are boxes
// (a random orientation, with the box's axis ratios drawn in [0.5, 1.5)), half spheres. The
// size is set by the k-th nearest of the layer's Gaussians, with k a 2 to 5 percent share, so
// each selects a few percent of the layer. The center is uniform in the positions' bounds.
SelectionQuery randomSelection(XorShift32 & rng, const std::vector<glm::vec4> & positions,
                               SelectOp op)
{
    const size_t n = positions.size();
    glm::vec3 lo(std::numeric_limits<float>::max());
    glm::vec3 hi(std::numeric_limits<float>::lowest());
    for (const glm::vec4 & p : positions)
    {
        lo = glm::min(lo, glm::vec3(p));
        hi = glm::max(hi, glm::vec3(p));
    }
    const float ux = rng.unit();
    const float uy = rng.unit();
    const float uz = rng.unit();
    const glm::vec3 center = lo + (hi - lo) * glm::vec3(ux, uy, uz);
    const double fraction = 0.02 + 0.03 * static_cast<double>(rng.unit());
    const size_t k = std::max<size_t>(1, static_cast<size_t>(fraction * static_cast<double>(n)));
    const glm::dvec3 c(center);
    std::vector<double> keys(n);

    if (rng.below(2) == 0)
    {
        const glm::mat3 axes = glm::mat3_cast(randomRotation(rng));
        const float wx = 0.5f + rng.unit();
        const float wy = 0.5f + rng.unit();
        const float wz = 0.5f + rng.unit();
        const glm::vec3 w(wx, wy, wz);
        for (size_t i = 0; i < n; ++i)
        {
            const glm::dvec3 d = glm::dvec3(glm::vec3(positions[i])) - c;
            double key = 0.0;
            for (int a = 0; a < 3; ++a)
            {
                key = std::max(key, std::fabs(glm::dot(glm::dvec3(axes[a]), d))
                                        / static_cast<double>(w[a]));
            }
            keys[i] = key;
        }
        std::nth_element(keys.begin(), keys.begin() + static_cast<std::ptrdiff_t>(k - 1),
                         keys.end());
        const double scale = keys[k - 1];
        return boxOnAxes(center, axes, w * static_cast<float>(scale), op);
    }

    for (size_t i = 0; i < n; ++i)
    {
        keys[i] = glm::length(glm::dvec3(glm::vec3(positions[i])) - c);
    }
    std::nth_element(keys.begin(), keys.begin() + static_cast<std::ptrdiff_t>(k - 1), keys.end());
    return sphereAt(center, static_cast<float>(keys[k - 1]), op);
}

// One random op. Returns false when it is skipped (nothing selected, or the op does not apply
// to this state); otherwise `entry` holds its undo, its redo, and the counts are updated.
bool randomOp(EditPass & edit, GaussianCloud & layer, SelectionMask & mask, XorShift32 & rng,
              OpCounts & counts, size_t & redoRecordMismatches, StackEntry & entry)
{
    const size_t n = layer.count();
    const uint32_t kind = rng.below(100);

    if (kind < 40)
    {
        const std::vector<uint32_t> before = mask.download(n);
        const SelectOp op = static_cast<SelectOp>(rng.below(3));
        const SelectionQuery query = randomSelection(rng, layer.positions(), op);
        edit.select(layer, identityTransform(), query, mask);
        const std::vector<uint32_t> after = mask.download(n);
        entry.undo = [&mask, before]() { mask.upload(before); };
        entry.redo = [&mask, after]() { mask.upload(after); };
        ++counts.selects;
        return true;
    }

    const uint32_t selected = edit.countSelected(mask, static_cast<uint32_t>(n));
    if (selected == 0)
    {
        return false;
    }

    if (kind < 65)
    {
        const std::vector<uint32_t> maskWords = mask.download(n);
        const LayerTransform placement = randomPlacement(rng);
        const glm::dvec3 pivot =
            similarityOf(placement).apply(edit.maskedCentroid(layer, mask).centroid());
        const glm::dmat3 rotation = randomRotationD(rng);
        const double scale = 0.7 + 0.7 * static_cast<double>(rng.unit());
        const glm::dvec3 translation(signedVec3(rng));
        const LayerTransform layerSpace =
            layerSpaceEdit(placement, worldEditAboutPivot(rotation, scale, translation, pivot));
        const SparseSnapshot snapshot = edit.gatherSelected(layer, mask, SnapshotScope::Transform);
        edit.transformSelected(layer, mask, layerSpace);
        entry.undo = [&layer, &mask, &edit, snapshot, maskWords]()
        {
            edit.scatterSnapshot(layer, snapshot);
            mask.upload(maskWords);
        };
        entry.redo = [&layer, &mask, &edit, layerSpace, maskWords]()
        {
            mask.upload(maskWords);
            edit.transformSelected(layer, mask, layerSpace);
        };
        ++counts.transforms;
        return true;
    }

    if (kind < 85)
    {
        if (n > kMaxCountToDuplicate)
        {
            return false;
        }
        const uint64_t generation = layer.storageGeneration();
        const DuplicateRecord record = edit.duplicateSelected(layer, mask);
        if (layer.storageGeneration() != generation)
        {
            ++counts.growths;
        }
        const std::shared_ptr<DuplicateRecord> saved = std::make_shared<DuplicateRecord>(record);
        entry.undo = [&layer, &mask, &edit, saved]() { edit.undoDuplicate(layer, mask, *saved); };
        entry.redo = [&layer, &mask, &edit, saved, &redoRecordMismatches]()
        {
            mask.upload(saved->selectionWords);
            const DuplicateRecord again = edit.duplicateSelected(layer, mask);
            if (again.added != saved->added || again.oldCount != saved->oldCount)
            {
                ++redoRecordMismatches;
            }
        };
        ++counts.duplicates;
        return true;
    }

    if (selected == n)
    {
        return false;
    }
    DeleteRecord record = edit.deleteSelected(layer, mask, nullptr);
    const std::shared_ptr<DeleteRecord> saved = std::make_shared<DeleteRecord>(std::move(record));
    entry.undo = [&layer, &mask, &edit, saved]() { edit.undoDelete(layer, mask, *saved); };
    entry.redo = [&layer, &mask, &edit, saved, &redoRecordMismatches]()
    {
        mask.upload(saved->selectionWords);
        const DeleteRecord again = edit.deleteSelected(layer, mask, nullptr);
        if (again.keptCount != saved->keptCount)
        {
            ++redoRecordMismatches;
        }
    };
    ++counts.deletes;
    return true;
}

// One run: `kOpsPerRun` ops, undone newest first (each undo checked against the hash of the
// step before), then `afterUndoAll`, then redone oldest first (each redo checked against its
// hash), and the final streams compared with the first pass's.
void runRandomSequence(EditPass & edit, GaussianCloud & layer, SelectionMask & mask,
                       const GaussianCloud & orig, uint32_t seed, const std::string & label,
                       const std::function<void()> & afterUndoAll)
{
    const size_t startCount = orig.count();
    const Streams original = copyStreams(orig);
    XorShift32 rng(seed);
    OpCounts counts;
    size_t redoRecordMismatches = 0;
    std::vector<StackEntry> stack;
    const uint64_t hash0 = hashState(layer, mask);

    while (stack.size() < kOpsPerRun)
    {
        mask.resize(layer.capacity());
        StackEntry entry;
        if (randomOp(edit, layer, mask, rng, counts, redoRecordMismatches, entry))
        {
            entry.hashAfter = hashState(layer, mask);
            stack.push_back(std::move(entry));
        }
    }
    const size_t finalCount = layer.count();
    const Streams finalFirst = readBack(layer);
    const std::vector<uint32_t> finalMaskWords = mask.download(finalCount);

    size_t undoMismatches = 0;
    for (size_t i = stack.size(); i-- > 0;)
    {
        stack[i].undo();
        const uint64_t want = i == 0 ? hash0 : stack[i - 1].hashAfter;
        if (hashState(layer, mask) != want)
        {
            ++undoMismatches;
        }
    }
    check(undoMismatches == 0,
          label + ": undoing each of the " + std::to_string(kOpsPerRun)
              + " ops, newest first, restores the hash of the step before it");
    checkBitEqualStreams(original, readBack(layer), label + " after undoing every op");
    afterUndoAll();

    size_t redoMismatches = 0;
    for (size_t i = 0; i < stack.size(); ++i)
    {
        stack[i].redo();
        if (hashState(layer, mask) != stack[i].hashAfter)
        {
            ++redoMismatches;
        }
    }
    check(redoMismatches == 0 && redoRecordMismatches == 0,
          label + ": redoing each op, oldest first, restores its hash and an equivalent record");
    checkBitEqualStreams(finalFirst, readBack(layer),
                         label + " final state after redoing every op");
    check(layer.count() == finalCount && mask.download(finalCount) == finalMaskWords,
          label + " final selection after redoing every op");

    std::printf("  %s: %zu selects, %zu transforms, %zu duplicates (%zu growths), %zu deletes; "
                "count %zu -> %zu\n",
                label.c_str(), counts.selects, counts.transforms, counts.duplicates,
                counts.growths, counts.deletes, startCount, finalCount);
    check(counts.transforms >= 20,
          label + ": at least 20 transforms (" + std::to_string(counts.transforms) + ")");
    check(counts.deletes >= 10,
          label + ": at least 10 deletes (" + std::to_string(counts.deletes) + ")");
    check(counts.duplicates >= 10,
          label + ": at least 10 duplicates (" + std::to_string(counts.duplicates) + ")");
    check(counts.growths >= 1,
          label + ": at least one capacity growth (" + std::to_string(counts.growths) + ")");
}

// The renderer's current core streams, read back with readGaussians.
Streams readRendererCore(Renderer & r)
{
    const size_t n = r.gaussianCount();
    const size_t perGaussian = static_cast<size_t>(kShCoefficientsPerGaussian);
    std::vector<float> means(n * 4);
    std::vector<float> scales(n * 4);
    std::vector<float> rotations(n * 4);
    std::vector<float> opacities(n);
    std::vector<float> sh(n * perGaussian * 4);
    GaussianHostOutput out;
    out.count = static_cast<uint32_t>(n);
    out.means = means.data();
    out.scales = scales.data();
    out.rotations = rotations.data();
    out.opacities = opacities.data();
    out.shCoefficients = sh.data();
    r.readGaussians(out);

    Streams s;
    s.positions.resize(n);
    std::memcpy(s.positions.data(), means.data(), means.size() * sizeof(float));
    s.scales.resize(n);
    std::memcpy(s.scales.data(), scales.data(), scales.size() * sizeof(float));
    s.rotations.resize(n);
    std::memcpy(s.rotations.data(), rotations.data(), rotations.size() * sizeof(float));
    s.opacities = opacities;
    s.shCoefficients.resize(n * perGaussian);
    std::memcpy(s.shCoefficients.data(), sh.data(), sh.size() * sizeof(float));
    return s;
}

// Run 1: a 20,011-Gaussian layer on a Renderer's device. After every op is undone, the
// restored layer is merged by the renderer and the readback must equal the original.
void runRendererRun()
{
    Renderer r(kWidth, kHeight);
    if (!r.initialize({}, true, testShaderDir()))
    {
        check(false, "M: renderer initialize");
        return;
    }
    Device & device = r.sabercatDevice();
    EditPass edit(device, testShaderDir());
    const std::unique_ptr<GaussianCloud> orig =
        makeSynthetic(device, SyntheticSpec{kCountRun1, 3, true, false, 0x4D0001u});
    const Streams original = copyStreams(*orig);
    // uploadToGPU(capacity == count): the first duplicate grows the layer.
    std::unique_ptr<GaussianCloud> layer = makeGpuCopy(device, *orig, kCountRun1);
    SelectionMask mask(device, kCountRun1);
    const std::string label = "M: run 1 (20,011 Gaussians, SH3, relight, renderer device)";

    const auto afterUndoAll = [&]()
    {
        const std::vector<MergeLayer> layers{MergeLayer(layer.get(), identityTransform())};
        r.mergeLayersOnGpu(layers);
        check(r.gaussianCount() == kCountRun1,
              label + ": the renderer merge of the restored layer keeps its count");
        const Streams core = readRendererCore(r);
        check(bitEqual(original.positions, core.positions),
              label + ": renderer readGaussians after undoing every op: means bit-exact");
        check(bitEqual(original.scales, core.scales),
              label + ": renderer readGaussians after undoing every op: scales bit-exact");
        check(bitEqual(original.rotations, core.rotations),
              label + ": renderer readGaussians after undoing every op: rotations bit-exact");
        check(bitEqual(original.opacities, core.opacities),
              label + ": renderer readGaussians after undoing every op: opacities bit-exact");
        check(bitEqual(original.shCoefficients, core.shCoefficients),
              label + ": renderer readGaussians after undoing every op: shCoefficients bit-exact");
    };
    runRandomSequence(edit, *layer, mask, *orig, 0x4D0001u, label, afterUndoAll);
}

// Run 2: layer B (4,099 Gaussians, SH1, no relight) on the main device, capacity == count.
void runDeviceRun(Device & device, EditPass & edit, const GaussianCloud & orig)
{
    std::unique_ptr<GaussianCloud> layer = makeGpuCopy(device, orig, orig.count());
    SelectionMask mask(device, orig.count());
    runRandomSequence(edit, *layer, mask, orig, 0x4D0002u,
                      "M: run 2 (4,099 Gaussians, SH1, no relight)", [] {});
}

} // namespace

void checkOps(Device & device)
{
    EditPass edit(device, testShaderDir());
    const std::unique_ptr<GaussianCloud> origA =
        makeSynthetic(device, SyntheticSpec{kCountA, 3, true, false, 0xA0A0001u});
    const std::unique_ptr<GaussianCloud> origB =
        makeSynthetic(device, SyntheticSpec{kCountB, 1, false, true, 0xB0B0003u});
    const Fixture fixtures[] = {{origA.get(), "A"}, {origB.get(), "B"}};

    for (const Fixture & f : fixtures)
    {
        const std::string name = f.name;
        const GaussianCloud & orig = *f.cloud;
        const size_t n = orig.count();
        const uint32_t last = static_cast<uint32_t>(n - 1);

        guard("L: " + name + " transform", [&] { checkTransform(device, edit, f); });
        guard("L: " + name + " snapshot", [&] { checkSnapshot(device, edit, f); });

        const std::string boxLabel = "L: " + name + " delete (box)";
        const SelectionQuery query = defaultBox();
        guard(boxLabel,
              [&]
              {
                  const std::vector<uint32_t> sel = selectionOfCopy(device, edit, orig, query);
                  checkDeleteCase(device, edit, orig, boxLabel, sel, &query);
              });

        const std::vector<std::pair<std::string, std::vector<uint32_t>>> edges = {
            {"first only", std::vector<uint32_t>{0u}},
            {"last only", std::vector<uint32_t>{last}},
            {"index 31", std::vector<uint32_t>{31u}},
            {"index 32", std::vector<uint32_t>{32u}},
            {"index 33", std::vector<uint32_t>{33u}},
            {"indices 31 to 33", std::vector<uint32_t>{31u, 32u, 33u}},
        };
        for (const auto & edge : edges)
        {
            const std::string label = "L: " + name + " delete (" + edge.first + ")";
            guard(label, [&] { checkDeleteCase(device, edit, orig, label, edge.second, nullptr); });
        }

        const std::string emptyLabel = "L: " + name + " delete (empty selection)";
        guard(emptyLabel, [&] { checkDeleteEmpty(device, edit, orig, emptyLabel); });
        const std::string allLabel = "L: " + name + " delete (every Gaussian)";
        guard(allLabel, [&] { checkDeleteAll(device, edit, orig, allLabel); });

        const std::string dupLabel = "L: " + name + " duplicate (capacity == count)";
        guard(dupLabel, [&] { checkDuplicate(device, edit, orig, dupLabel, 0); });
        const std::string spareLabel = "L: " + name + " duplicate (spare capacity)";
        guard(spareLabel, [&] { checkDuplicate(device, edit, orig, spareLabel, n / 2); });
        const std::string dupEmptyLabel = "L: " + name + " duplicate (empty selection)";
        guard(dupEmptyLabel, [&] { checkDuplicateEmpty(device, edit, orig, dupEmptyLabel); });
    }

    guard("M: run 1 (renderer device)", [] { runRendererRun(); });
    guard("M: run 2 (layer B)", [&] { runDeviceRun(device, edit, *origB); });
}

} // namespace edit_test
