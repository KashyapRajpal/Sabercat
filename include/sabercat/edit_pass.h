// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <glm/glm.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "sabercat/buffer_view.h"
#include "sabercat/camera.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/layer_transform.h"
#include "sabercat/selection_mask.h"

// Studio edit kernels (layer merge, select, transform, delete/compact, duplicate)
// run here, on a layer's buffers in layer-local space. Each kernel is one
// synchronous submission on the single submit queue (Device::submitQueue), so a
// frame never overlaps an edit. Every submission is bracketed by
// Device::recordGlobalBarrier (immediateSubmit does this on both sides). That
// bracket is the compaction barrier: no frame is ever in flight over a buffer
// while it is being compacted, and a compacted buffer is visible to the next
// frame.
//
// Implemented: the exclusive scan that compaction and duplication are built on
// (exclusiveScan), the layer merge (mergeLayers), the selection kernels (select,
// the mask operations and expandMaskToFlags over SelectionMask), and the stream
// edits of Studio phase C4: the sparse snapshot (gatherSelected, scatterSnapshot),
// the in-place similarity transform (transformSelected), and delete and duplicate
// with their undo (deleteSelected, undoDelete, duplicateSelected, undoDuplicate).

namespace sabercat
{

class Buffer;
class Device;
class GpuScan; // internal (src/renderer/gpu_scan.h), not installed

// One layer of a merge: its cloud and the placement it is baked with. The same
// shape as mergeLayers() in layer_transform.h takes.
using MergeLayer = std::pair<const GaussianCloud *, LayerTransform>;

// The layout the merged scene of some layers has, as planLayerMerge() computes it.
struct MergedLayout
{
    size_t count = 0;         // sum of the layers' Gaussian counts
    uint32_t shDegree = 0;    // max of the layers' SH degrees
    bool antialiased = true;  // true only if every layer is antialiased
    RelightPresence presence; // a stream only if every layer has its GPU stream
};

// The layout of the merged scene of `layers`, with mergeLayers()'s rules. A relight
// stream some layers have and others lack is dropped, with one Logger::warning per
// stream. Throws std::runtime_error for an empty list, a null cloud, a cloud with no
// GPU parameters (not uploaded), or an invalid transform. Host-only: no GPU work.
MergedLayout planLayerMerge(const std::vector<MergeLayer> & layers);

// The shape of a selection (SelectionQuery::shape).
enum class SelectShape : uint32_t
{
    Box = 0,       // oriented box, in world space
    Sphere = 1,    // sphere, in world space
    Rectangle = 2, // a screen rectangle, projected through the query's camera
};

// How a selection's result combines with the mask's bits (SelectionQuery::op).
enum class SelectOp : uint32_t
{
    Replace = 0,  // the mask becomes the predicate
    Add = 1,      // the predicate is or-ed into the mask
    Subtract = 2, // the predicate's Gaussians are cleared from the mask
};

// A selection, in world space. The fields are grouped in 16-byte runs (a vec3 with
// the float that follows it), so the struct has no padding holes. Only the fields
// of the chosen shape are read: the box fields for Box, center and radius for
// Sphere, rect and camera for Rectangle.
struct SelectionQuery
{
    glm::vec3 center{0.0f};      // box or sphere center
    float radius = 0.0f;         // sphere radius
    glm::vec3 halfExtents{0.0f}; // box half extents along the three box axes
    float reserved0 = 0.0f;
    glm::vec3 boxAxisX{1.0f, 0.0f, 0.0f}; // unit world-space box axes, supplied orthonormal
    float reserved1 = 0.0f;
    glm::vec3 boxAxisY{0.0f, 1.0f, 0.0f};
    float reserved2 = 0.0f;
    glm::vec3 boxAxisZ{0.0f, 0.0f, 1.0f};
    float reserved3 = 0.0f;
    glm::vec4 rect{0.0f};        // rectangle x0, y0, x1, y1 in framebuffer pixels
    SelectShape shape = SelectShape::Box;
    SelectOp op = SelectOp::Replace;
    // Rectangle only: the camera the rectangle was drawn in, the last forward's
    // (Renderer::lastForwardCamera()).
    CameraUniforms camera{};
};
// 16 * 5 (center/radius, halfExtents/reserved0, the three box axes with their reserved
// floats) + 16 (rect) + 8 (shape, op) + 128 (camera) = 232: no padding.
static_assert(sizeof(SelectionQuery) == 232,
              "SelectionQuery is padding-free (glm vec3/vec4 are 4-byte aligned)");

// The centroid of a masked set of Gaussians, in the layer's local space.
struct MaskCentroid
{
    glm::dvec3 sum{0.0};
    uint64_t count = 0;

    // sum / count, or zero when count is zero.
    glm::dvec3 centroid() const
    {
        if (count == 0)
        {
            return glm::dvec3(0.0);
        }
        return sum / static_cast<double>(count);
    }
};

// Which streams a sparse snapshot holds (gatherSelected).
enum class SnapshotScope : uint32_t
{
    // The streams a similarity transform rewrites: indices, means, scales, rotations,
    // shCoefficients and normals (when the layer has them).
    Transform = 0,
    // Every stream the layer has: what a delete needs to be undone.
    Full = 1,
};

// A host copy of the selected Gaussians' values, in ascending index order. Each stream is
// held in its GPU layout, one element per Gaussian (SH and visibility: 16 per Gaussian). A
// stream the snapshot does not carry is an empty vector.
struct SparseSnapshot
{
    std::vector<uint32_t> indices;             // layer-local index of each element, ascending
    std::vector<glm::vec4> means;
    std::vector<glm::vec4> scales;
    std::vector<glm::vec4> rotations;
    std::vector<glm::vec4> shCoefficients;     // 16 per element
    std::vector<float> opacities;              // Full only
    std::vector<glm::vec4> normals;            // present streams only (empty if the layer has none)
    std::vector<glm::vec4> baseColorRoughness; // Full only
    std::vector<float> metallic;               // Full only
    std::vector<float> visibility;             // Full only; 16 floats per element
    SnapshotScope scope = SnapshotScope::Transform;
    // The layer's SH degree (layer.shDegree()) when the snapshot was gathered. Informational:
    // scatterSnapshot and the undo paths take their degree from the layer.
    uint32_t shDegree = 0;

    size_t count() const
    {
        return indices.size();
    }
    // The host memory the vectors hold: the sum of size() * sizeof(element).
    size_t bytes() const;
};
// Ten vectors, the scope and the SH degree: 10 * 24 + 2 * 4 = 248 bytes, no padding.
static_assert(sizeof(SparseSnapshot) == 10 * sizeof(std::vector<float>) + 2 * sizeof(uint32_t),
              "SparseSnapshot is padding-free");

// What undoDelete needs to put a deleted selection back: the deleted Gaussians (a Full
// snapshot), the selection the delete was made with, and the counts before and after.
struct DeleteRecord
{
    SparseSnapshot deleted;
    std::vector<uint32_t> selectionWords;
    uint32_t oldCount = 0;
    uint32_t keptCount = 0;
};

// What undoDuplicate needs: the selection the duplicate was made with, the count before,
// and how many Gaussians were appended.
struct DuplicateRecord
{
    std::vector<uint32_t> selectionWords;
    uint32_t oldCount = 0;
    uint32_t added = 0;
};

class EditPass
{
  public:
    EditPass(Device & device, const std::string & shaderDir);
    ~EditPass(); // defined in the .cpp, where GpuScan and the kernels are complete

    // Non-copyable: it owns the scan and merge kernels and their device buffers.
    EditPass(const EditPass &) = delete;
    EditPass & operator=(const EditPass &) = delete;

    Device & device() const
    {
        return m_device;
    }

    // Synchronous exclusive scan of `count` uint32 values read from `input` into
    // `output` (which needs count uints). `inputStrideUints` and `inputOffsetUints`
    // select the value in each element, as GpuScan::record does. One
    // Device::immediateSubmit. Returns the total (the sum of all inputs), read
    // back from the last output and the last input. count == 0 returns 0.
    // Both views must start at offset 0, and must lie in different buffers (the
    // total is read back from the input after the scan).
    uint32_t exclusiveScan(const BufferView & input, const BufferView & output, uint32_t count,
                           uint32_t inputStrideUints = 1, uint32_t inputOffsetUints = 0);

    // Synchronous layer merge: bakes `layers` into `destination`'s GPU streams, in
    // the order given, on the GPU. This is mergeLayers() of layer_transform.h
    // without the host copy: the same placement math, in fp32, from the same
    // BakeFrame. Layer i covers the destination Gaussians after the layers before it.
    //
    // Preconditions, each checked before any GPU work (std::runtime_error):
    //   - `destination` has GPU parameters, and its count() is exactly
    //     planLayerMerge(layers).count (size it with allocateGPU first);
    //   - no layer is `destination`;
    //   - a relight stream `destination` has (relightViews() valid) is on every layer.
    // Streams the destination lacks are not written. The destination's shDegree,
    // antialiased and bounds are not set: the caller does that.
    //
    // The placements are uploaded first (one Buffer::upload), then every layer's
    // kernel is recorded into one Device::immediateSubmit. Layers write disjoint
    // destination ranges, so the dispatches need no barriers between them.
    void mergeLayers(const std::vector<MergeLayer> & layers, GaussianCloud & destination);

    // Synchronous selection of a layer's Gaussians by a world-space shape, written
    // into `mask`: the predicate of each Gaussian's mean, placed by `placement` as
    // the merge places it (fp32 from the same BakeFrame), is combined into the mask
    // by query.op. Bits at or beyond layer.count() stay zero.
    //
    // Preconditions, each checked before any GPU work (std::runtime_error):
    //   - `layer` has GPU parameters, and `mask` has capacity >= layer.count();
    //   - `placement` is a valid transform (validateLayerTransform);
    //   - for Rectangle, the query's camera has fx > 0 and fy > 0.
    void select(const GaussianCloud & layer, const LayerTransform & placement,
                const SelectionQuery & query, SelectionMask & mask);

    // Synchronous mask operations. Each is one immediateSubmit over one kernel.
    void clearMask(SelectionMask & mask); // every word 0
    void selectAll(SelectionMask & mask, uint32_t count); // bits [0, count) set, the rest 0
    // Flips the bits below count; bits at or beyond count stay 0.
    void invertMask(SelectionMask & mask, uint32_t count);

    // The number of set bits among the first `count` Gaussians. Per-workgroup partial
    // counts are summed on the host (in 64 bits, returned as uint32). count must not
    // exceed mask.capacity().
    uint32_t countSelected(const SelectionMask & mask, uint32_t count);

    // The centroid of the selected Gaussians of `layer`, in the LAYER-LOCAL space: the
    // host applies the placement in double. Per-workgroup partial sums are added on the
    // host in double. Requires mask.capacity() >= layer.count(), and an empty result
    // (count 0, zero centroid) when nothing is selected.
    MaskCentroid maskedCentroid(const GaussianCloud & layer, const SelectionMask & mask);

    // Writes flags[i] for i < count: 1 for a selected Gaussian, 0 otherwise. With
    // keepUnselected the values are inverted (the flags of a delete: 1 keeps).
    // `flags` must hold count uints, start at offset 0 and be a storage buffer.
    void expandMaskToFlags(const SelectionMask & mask, uint32_t count, bool keepUnselected,
                           const BufferView & flags);

    // ---- Stream edits (Studio phase C4). Every method is synchronous: each GPU step
    // is one Device::immediateSubmit. Each takes the layer's views at the time of its
    // dispatch, so a reserve() made by an earlier call is always seen. Every
    // precondition is checked before the first GPU step, so a throw leaves the layer
    // and the mask as they were (std::runtime_error, or std::invalid_argument where
    // noted).
    //
    // A layer whose parameters are borrowed (ownership model B) is refused, with a
    // std::runtime_error, by every operation that copies its buffers: gatherSelected,
    // scatterSnapshot, deleteSelected, undoDelete, duplicateSelected and undoDuplicate.
    // transformSelected and select only write or read in place and accept borrowed layers.

    // A host copy of the selected Gaussians of `layer` (mask bits below layer.count()),
    // in ascending index order. Scope Transform copies indices, means, scales, rotations,
    // shCoefficients and normals (when present); Full copies every stream the layer has.
    // A stream the scope does not carry stays empty. Requires mask.capacity() >= count().
    SparseSnapshot gatherSelected(const GaussianCloud & layer, const SelectionMask & mask,
                                  SnapshotScope scope);

    // Writes the snapshot's streams back at its indices (the bits of a restore: the
    // words are copied, not recomputed). Only the streams the snapshot holds (non-empty)
    // are written, and each must exist on the layer with the snapshot's element count.
    // Throws if an index is not below layer.count(), or if a stream does not match. An
    // empty snapshot is a no-op.
    void scatterSnapshot(GaussianCloud & layer, const SparseSnapshot & snapshot);

    // Applies a layer-space similarity to the selected Gaussians, in place: the same math
    // as mergeLayers() for one layer, so the result equals a merge of the placement. The
    // placement is validated (validateLayerTransform). An identity transform is a no-op
    // (no GPU work); no selected Gaussians is fine (the kernel writes nothing).
    void transformSelected(GaussianCloud & layer, const SelectionMask & mask,
                           const LayerTransform & layerSpace);

    // Removes the selected Gaussians, compacting every stream, and returns the record
    // undoDelete needs. The kept Gaussians keep their order. The selection is cleared.
    // With `remapOut`, its buffer receives one uint per old index: the new index of a kept
    // Gaussian, 0xFFFFFFFF for a deleted one. remapOut must be a storage buffer at offset
    // 0 with size >= oldCount * 4 (VK_VALIDATE).
    //
    // Selecting nothing returns a record with deleted.count() == 0 and changes nothing.
    // Selecting every Gaussian throws std::invalid_argument (remove the layer instead).
    DeleteRecord deleteSelected(GaussianCloud & layer, SelectionMask & mask,
                                const BufferView * remapOut = nullptr);

    // Restores the record's deleted Gaussians, in their original order, and its selection,
    // bit-exactly. Requires layer.count() == record.keptCount and layer.capacity() >=
    // record.oldCount. A record with deleted.count() == 0 is a no-op.
    // Redo: restore the selection the delete was made with and call deleteSelected again.
    void undoDelete(GaussianCloud & layer, SelectionMask & mask, const DeleteRecord & record);

    // Appends a copy of every selected Gaussian after the layer's last one, growing the
    // layer (reserve, at max(needed, 1.5 x capacity)) and the mask when it has to. The
    // copies become the selection, and the record holds the selection before the copy.
    // Selecting nothing returns a no-op record.
    DuplicateRecord duplicateSelected(GaussianCloud & layer, SelectionMask & mask);

    // Drops the record's appended Gaussians (the count goes back to oldCount; the capacity
    // is kept) and restores the selection. Requires layer.count() == oldCount + added. A
    // record with added == 0 is a no-op.
    void undoDuplicate(GaussianCloud & layer, SelectionMask & mask,
                       const DuplicateRecord & record);

    // The scan kernel, for in-library kernels and the tests. GpuScan is internal:
    // it is not part of the installed API, and a caller outside the library
    // should use exclusiveScan instead.
    GpuScan & scan();

  private:
    // The layer merge's pipeline and descriptor sets, created on the first
    // mergeLayers() call (defined in the .cpp).
    struct MergeKernel;
    // The selection predicate's pipeline and its single descriptor set, created on the
    // first select() call (defined in the .cpp).
    struct SelectKernel;
    // The mask operations' pipeline and its single descriptor set, created on the first
    // mask operation (defined in the .cpp).
    struct MaskKernel;
    // The stream compaction kernel (edit_compact.slang) and the transform kernel
    // (edit_transform.slang): one pipeline and one descriptor set each, created on first
    // use (defined in the .cpp).
    struct CompactKernel;
    struct TransformKernel;

    // One edit_compact.slang dispatch. Bindings left invalid bind the dummy. `threads` is
    // the work items (count * wordsPerElement, or count for INDEX and REMAP). When copyTo is
    // valid, the same submission copies copyBytes from dst (at dst.offset) to copyTo (at
    // copyTo.offset) after the dispatch, behind a compute-to-transfer barrier.
    struct CompactCall
    {
        VkDeviceSize copyBytes = 0;
        uint32_t mode = 0;
        uint32_t count = 0;
        uint32_t wordsPerElement = 1;
        uint32_t base = 0;
        uint32_t threads = 0;
        BufferView flags;
        BufferView scan;
        BufferView src;
        BufferView dst;
        BufferView aux;
        BufferView copyTo;
    };
    void runCompact(const CompactCall & call);

    // Runs one edit_mask_ops.slang mode over `threads` invocations (the kernel's
    // dispatch covers them in workgroups of kWorkgroupSize). `means` and `flags` are bound
    // only by the modes that use them; an invalid view binds the dummy in their place.
    void runMaskOp(uint32_t mode, uint32_t count, uint32_t wordCount, uint32_t flag,
                   uint32_t threads, const SelectionMask & mask, const BufferView & means,
                   const BufferView & flags);

    Device & m_device;
    std::unique_ptr<GpuScan> m_scan;
    std::string m_shaderDir;
    // Binding 0 of the merge kernel: every layer's SimilarityParams, grow-only.
    std::unique_ptr<Buffer> m_mergeParams;
    // Bound to every absent stream and to every unused binding; the kernels never
    // read or write it.
    std::unique_ptr<Buffer> m_dummy;
    std::unique_ptr<MergeKernel> m_merge;
    // Binding 2 of the select kernel: the SelectParams of one call (256 B).
    std::unique_ptr<Buffer> m_selectParams;
    // Per-workgroup partials of the mask operations, grow-only.
    std::unique_ptr<Buffer> m_countPartials;    // one uint per workgroup
    std::unique_ptr<Buffer> m_centroidPartials; // one vec4 (sum.xyz, count) per workgroup
    std::unique_ptr<SelectKernel> m_select;
    std::unique_ptr<MaskKernel> m_maskOps;
    std::unique_ptr<CompactKernel> m_compact;
    std::unique_ptr<TransformKernel> m_transform;
    // Binding 1 of the transform kernel: the SimilarityParams of one transform (480 B).
    std::unique_ptr<Buffer> m_transformParams;
    // Grow-only scratch of the stream edits: the keep or select flags and their scan (one
    // uint per Gaussian), the selected indices (one uint per selected Gaussian), the packed
    // values a scatter uploads, and the stream a compaction or gather writes before it is
    // read back or copied over the layer's stream.
    std::unique_ptr<Buffer> m_editFlags;
    std::unique_ptr<Buffer> m_editScan;
    std::unique_ptr<Buffer> m_editIndices;
    std::unique_ptr<Buffer> m_editPacked;
    std::unique_ptr<Buffer> m_editScratch;
};

} // namespace sabercat
