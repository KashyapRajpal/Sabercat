// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/edit_pass.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "gpu_scan.h"
#include "layer_bake_frame.h"
#include "sabercat/buffer.h"
#include "sabercat/descriptor_set.h"
#include "sabercat/device.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/logger.h"
#include "sabercat/pipeline.h"
#include "sabercat/render_config.h"
#include "sabercat/shader_compiler.h"
#include "sabercat/vk_check.h"

namespace sabercat
{

namespace
{

// SimilarityParams in shaders/edit_common.slangh, byte for byte. Every member is
// a run of 4-byte scalars and the members follow one another with no gap:
// 64 + 16 + 48 + 336 + 4 + 4 + 4 + 4 = 480 bytes. The matrix is column-major,
// glm's memory order.
struct SimilarityParamsHost
{
    float matrix[16];    // node matrix, column-major: matrix[4 * column + row]
    float frameQuat[4];  // unit quaternion xyzw
    float rotRows[3][4]; // rows of the frame rotation, row r in xyz, w 0
    float bands[84];     // SH band matrices: band 1 at [0, 9), band 2 at [9, 34),
                         // band 3 at [34, 83); 83 is padding
    float uniformScale;
    uint32_t count;
    uint32_t shDegree; // min(source degree, 3)
    uint32_t identity; // 1 = straight copy
};
static_assert(sizeof(SimilarityParamsHost) == 480,
              "SimilarityParamsHost must match edit_common.slangh's SimilarityParams");

// LayerMergePush in shaders/layer_merge.slang: four uints, no padding.
struct LayerMergePush
{
    uint32_t first;    // destination Gaussian offset of the layer
    uint32_t layer;    // index into the placements buffer
    uint32_t flags;    // kMergeFlag* bits
    uint32_t reserved; // zero
};
static_assert(sizeof(LayerMergePush) == 4 * sizeof(uint32_t),
              "LayerMergePush must match layer_merge.slang's push constants");

// layer_merge.slang's LAYER_MERGE_FLAG_* bits.
constexpr uint32_t kMergeFlagNormals = 1u;
constexpr uint32_t kMergeFlagBaseColorRoughness = 2u;
constexpr uint32_t kMergeFlagMetallic = 4u;
constexpr uint32_t kMergeFlagVisibility = 8u;

// Binding slots, as layer_merge.slang declares them: 0 the placements, 1..9 the
// source streams and 10..18 the destination streams, both in MergeStream order.
constexpr uint32_t kMergeBindingCount = 19;
constexpr uint32_t kMergeBindingParams = 0;
constexpr uint32_t kMergeSourceBase = 1;
constexpr uint32_t kMergeDestinationBase = 10;

// The stream order of the source and destination bindings.
enum MergeStream : uint32_t
{
    kMeans,
    kScales,
    kRotations,
    kOpacity,
    kSh,
    kNormals,
    kBaseColorRoughness,
    kMetallic,
    kVisibility,
};

// The kernel reads at most degree 3; the canonical SH layout holds 16 coefficients.
constexpr uint32_t kMergeMaxDegree = 3;
// Descriptor sets per pool: one DescriptorSet object (one pool) per 16 dispatches.
constexpr uint32_t kSetsPerPool = 16;
// The byte size of the buffer bound to every absent stream or unused binding.
constexpr VkDeviceSize kDummyBytes = 16;

VkDescriptorSetLayoutBinding storageBufferBinding(uint32_t binding)
{
    VkDescriptorSetLayoutBinding b{};
    b.binding = binding;
    b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    return b;
}

std::vector<VkDescriptorSetLayoutBinding> mergeBindings()
{
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    bindings.reserve(kMergeBindingCount);
    for (uint32_t binding = 0; binding < kMergeBindingCount; ++binding)
    {
        bindings.push_back(storageBufferBinding(binding));
    }
    return bindings;
}

// Grow-only with geometric growth, as ForwardPass's ensureCapacity does.
void ensureBytes(Device & device, std::unique_ptr<Buffer> & buffer, VkDeviceSize neededBytes)
{
    if (buffer && buffer->size() >= neededBytes)
    {
        return;
    }
    const VkDeviceSize grownBytes = buffer ? buffer->size() * 2 : 0;
    buffer = std::make_unique<Buffer>(device, std::max(neededBytes, grownBytes));
}

// Whether the cloud's GPU parameters exist. views() throws (VK_VALIDATE) when they do
// not, so the throw is caught here and turned into a plain answer.
bool hasGpuParameters(const GaussianCloud & cloud)
{
    try
    {
        return cloud.views().means.valid();
    }
    catch (const std::runtime_error &)
    {
        return false;
    }
}

// Which optional relight streams a cloud has on the GPU.
RelightPresence gpuPresence(const GaussianCloud & cloud)
{
    const RelightBufferViews & relight = cloud.relightViews();
    RelightPresence presence;
    presence.normals = relight.normals.valid();
    presence.baseColorRoughness = relight.baseColorRoughness.valid();
    presence.metallic = relight.metallic.valid();
    presence.visibility = relight.visibility.valid();
    return presence;
}

// The view when the stream is carried, an invalid view (bound to the dummy) otherwise.
BufferView streamIf(bool carried, const BufferView & view)
{
    if (carried)
    {
        return view;
    }
    return BufferView{};
}

void warnDropped(const char * name, bool someHave, bool everyHas)
{
    if (someHave && !everyHas)
    {
        Logger::warning(std::string("mergeLayers: relight stream '") + name
                        + "' is missing from some layers; dropped from the merged cloud");
    }
}

// planLayerMerge's rules. `warn` is false for mergeLayers, which plans quietly:
// its caller has already been told by the public planLayerMerge, or it just means
// to write what the layers have.
MergedLayout planLayers(const std::vector<MergeLayer> & layers, bool warn)
{
    if (layers.empty())
    {
        throw std::runtime_error("mergeLayers: no layers to merge");
    }

    MergedLayout plan;
    RelightPresence any;
    RelightPresence all{true, true, true, true};
    for (size_t i = 0; i < layers.size(); ++i)
    {
        const GaussianCloud * cloud = layers[i].first;
        const std::string label = "mergeLayers: layer " + std::to_string(i);
        if (cloud == nullptr)
        {
            throw std::runtime_error(label + " has no cloud");
        }
        validateLayerTransform(layers[i].second, label + " transform");
        if (!hasGpuParameters(*cloud))
        {
            throw std::runtime_error(label + " has no GPU parameters (upload the cloud first)");
        }

        plan.count += cloud->count();
        plan.shDegree = std::max(plan.shDegree, cloud->shDegree());
        plan.antialiased = plan.antialiased && cloud->antialiased();

        const RelightPresence presence = gpuPresence(*cloud);
        any.normals = any.normals || presence.normals;
        any.baseColorRoughness = any.baseColorRoughness || presence.baseColorRoughness;
        any.metallic = any.metallic || presence.metallic;
        any.visibility = any.visibility || presence.visibility;
        all.normals = all.normals && presence.normals;
        all.baseColorRoughness = all.baseColorRoughness && presence.baseColorRoughness;
        all.metallic = all.metallic && presence.metallic;
        all.visibility = all.visibility && presence.visibility;
    }

    if (warn)
    {
        warnDropped("normals", any.normals, all.normals);
        warnDropped("baseColorRoughness", any.baseColorRoughness, all.baseColorRoughness);
        warnDropped("metallic", any.metallic, all.metallic);
        warnDropped("visibility", any.visibility, all.visibility);
    }

    plan.presence = all;
    return plan;
}

// The placement of one layer, as bakeLayer() in layer_transform.cpp uses it: the
// BakeFrame's double values rounded to fp32.
SimilarityParamsHost makePlacement(const LayerTransform & transform, const GaussianCloud & source)
{
    const BakeFrame frame = makeBakeFrame(transform);

    SimilarityParamsHost p{}; // value-initialized: every pad lane is zero
    for (int column = 0; column < 4; ++column)
    {
        for (int row = 0; row < 4; ++row)
        {
            p.matrix[4 * column + row] = static_cast<float>(frame.matrix[column][row]);
        }
    }
    for (int k = 0; k < 4; ++k)
    {
        p.frameQuat[k] = static_cast<float>(frame.rotationQuaternion[k]);
    }
    for (int row = 0; row < 3; ++row)
    {
        for (int col = 0; col < 3; ++col)
        {
            p.rotRows[row][col] = static_cast<float>(frame.rotation[row][col]);
        }
    }
    // Band b is (2b+1)^2 floats, starting at bandBase(b); see edit_common.slangh.
    constexpr uint32_t kBandBase[3] = {0, 9, 34};
    for (int band = 1; band <= 3; ++band)
    {
        const int n = 2 * band + 1;
        const uint32_t base = kBandBase[band - 1];
        for (int row = 0; row < n; ++row)
        {
            for (int col = 0; col < n; ++col)
            {
                p.bands[base + row * n + col] =
                    static_cast<float>(frame.shBands[band - 1][row][col]);
            }
        }
    }
    p.uniformScale = static_cast<float>(frame.uniformScale);
    p.count = static_cast<uint32_t>(source.count());
    p.shDegree = std::min(source.shDegree(), kMergeMaxDegree);
    p.identity = transform.isIdentity() ? 1u : 0u;
    return p;
}

} // namespace

namespace
{

// SelectParams in shaders/edit_select_common.slangh, byte for byte: two matrices and
// four vectors of 16 bytes, then four 4-byte scalars, with no gap.
// 64 + 64 + 48 + 16 + 16 + 16 + 16 + 16 = 256 bytes. The matrices are column-major,
// glm's memory order, as in SimilarityParamsHost.
struct SelectParamsHost
{
    glm::mat4 layerToWorld;  // the layer's node matrix, fp32
    glm::mat4 worldToCamera; // rectangle only: the drawn view's camera, else identity
    glm::vec4 boxAxis[3];    // box only: unit world-space axes, w 0
    glm::vec4 centerRadius;  // xyz: box or sphere center, w: sphere radius
    glm::vec4 halfExtents;   // xyz: box half extents, w 0
    glm::vec4 cameraRect;    // rectangle only: x0, y0, x1, y1, normalized
    glm::vec4 intrinsics;    // rectangle only: fx, fy, cx, cy
    uint32_t shape;
    uint32_t op;
    uint32_t count;
    uint32_t wordCount;
};
static_assert(sizeof(SelectParamsHost) == 256,
              "SelectParamsHost must match edit_select_common.slangh's SelectParams");

// MaskOpsPush in shaders/edit_mask_ops.slang: four uints, no padding.
struct MaskOpsPush
{
    uint32_t mode;
    uint32_t count;
    uint32_t wordCount;
    uint32_t flag;
    uint32_t groups; // workgroups dispatched: the shader's grid stride is groups * 256
};
static_assert(sizeof(MaskOpsPush) == 5 * sizeof(uint32_t),
              "MaskOpsPush must match edit_mask_ops.slang's push constants");

// EDIT_MASK_MODE_* in shaders/edit_mask_ops.slang.
constexpr uint32_t kMaskModeClear = 0u;
constexpr uint32_t kMaskModeAll = 1u;
constexpr uint32_t kMaskModeInvert = 2u;
constexpr uint32_t kMaskModePopcount = 3u;
constexpr uint32_t kMaskModeCentroid = 4u;
constexpr uint32_t kMaskModeExpand = 5u;

// Binding counts. edit_mask_ops.slang binds 0 mask, 1 means, 2 flags, 3 count partials
// and 4 centroid partials. edit_select.slang binds 0 mask, 1 means and 2 params.
constexpr uint32_t kMaskBindingCount = 5;
constexpr uint32_t kSelectBindingCount = 3;

// The number of mask words that hold `count` Gaussians: ceil(count / 32).
uint32_t wordsForCount(uint32_t count)
{
    return static_cast<uint32_t>((static_cast<uint64_t>(count) + 31u) / 32u);
}

// Binds `view` at `binding`, or `dummy` when the view names no buffer. The kernels never
// read or write an unused binding, so the dummy's size only has to be valid.
void bindStorage(DescriptorSet & descriptors, VkDescriptorSet set, uint32_t binding,
                 const BufferView & view, const Buffer & dummy)
{
    if (view.buffer != VK_NULL_HANDLE)
    {
        descriptors.bindBuffer(set, binding, view.buffer, view.offset, view.size,
                               VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    }
    else
    {
        descriptors.bindBuffer(set, binding, dummy.handle(), 0, dummy.size(),
                               VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    }
}

// The pipeline of one compute kernel with a single descriptor set of storage buffers.
// Each EditPass selection or mask operation is its own synchronous submission, so one
// set serves every call: the bindings are written before the submission, and the
// previous submission has completed (immediateSubmit waits on its fence) by then.
class OneSetKernel
{
  public:
    OneSetKernel(Device & device, const std::string & shaderDir, const char * spvName,
                 uint32_t bindingCount, uint32_t pushBytes)
        : device(device), descriptors(device.handle())
    {
        VkDevice dev = device.handle();
        // A throw below (a missing shader file is the common one) skips the destructor,
        // so the pipeline created so far is released here. The descriptor object cleans
        // up through its own destructor.
        try
        {
            std::vector<VkDescriptorSetLayoutBinding> bindings;
            bindings.reserve(bindingCount);
            for (uint32_t binding = 0; binding < bindingCount; ++binding)
            {
                bindings.push_back(storageBufferBinding(binding));
            }
            layout = descriptors.createLayout(bindings);
            descriptors.createPool(1, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, bindingCount}});
            set = descriptors.allocate(layout);

            ShaderCompiler compiler(dev);
            VkShaderModule module =
                compiler.createShaderModule(compiler.loadSPIRV(shaderDir + "/" + spvName));
            try
            {
                VkPipelineLayout pipelineLayout =
                    pipeline.createPipelineLayout(dev, layout, pushBytes,
                                                  VK_SHADER_STAGE_COMPUTE_BIT);
                pipeline.createComputePipeline(dev, module, pipelineLayout);
            }
            catch (...)
            {
                compiler.destroyShaderModule(module);
                throw;
            }
            compiler.destroyShaderModule(module);
        }
        catch (...)
        {
            pipeline.destroy(dev);
            throw;
        }
    }

    ~OneSetKernel()
    {
        pipeline.destroy(device.handle());
    }

    OneSetKernel(const OneSetKernel &) = delete;
    OneSetKernel & operator=(const OneSetKernel &) = delete;

    Device & device;
    DescriptorSet descriptors;
    VkDescriptorSetLayout layout = VK_NULL_HANDLE; // owned by descriptors
    VkDescriptorSet set = VK_NULL_HANDLE;          // allocated from descriptors
    Pipeline pipeline;
};

// EditCompactPush in shaders/edit_compact.slang: five uints, no padding. `groups` is the
// number of workgroups the host dispatched: the shader's grid stride is groups * 256.
struct CompactPush
{
    uint32_t mode;
    uint32_t count;
    uint32_t wordsPerElement;
    uint32_t base;
    uint32_t groups;
};
static_assert(sizeof(CompactPush) == 5 * sizeof(uint32_t),
              "CompactPush must match edit_compact.slang's push constants");

// EDIT_COMPACT_MODE_* in shaders/edit_compact.slang.
constexpr uint32_t kCompactModeGather = 0u;
constexpr uint32_t kCompactModeIndex = 1u;
constexpr uint32_t kCompactModeScatter = 2u;
constexpr uint32_t kCompactModeCompact = 3u;
constexpr uint32_t kCompactModeAppend = 4u;
constexpr uint32_t kCompactModeExpand = 5u;
constexpr uint32_t kCompactModeRemap = 6u;
constexpr uint32_t kCompactBindingCount = 5;

// EditTransformPush in shaders/edit_transform.slang: two uints, no padding.
struct TransformPush
{
    uint32_t count;
    uint32_t hasNormals;
};
static_assert(sizeof(TransformPush) == 2 * sizeof(uint32_t),
              "TransformPush must match edit_transform.slang's push constants");
constexpr uint32_t kTransformBindingCount = 7;

// The most workgroups one dimension of a dispatch may hold (the Vulkan minimum).
constexpr uint32_t kMaxDispatchGroups = 65535;

// The GPU streams of a layer, in the order of EditStreamId.
enum EditStreamId : uint32_t
{
    kStreamMeans,
    kStreamScales,
    kStreamRotations,
    kStreamOpacity,
    kStreamSh,
    kStreamNormals,
    kStreamBaseColorRoughness,
    kStreamMetallic,
    kStreamVisibility,
    kStreamCount,
};

// The uint32 words one Gaussian spans in each stream: 4 for the vec4 streams, 64 for SH
// (16 coefficients of 16 bytes), 1 for the scalars, 16 for the visibility coefficients.
constexpr std::array<uint32_t, kStreamCount> kStreamWords = {
    static_cast<uint32_t>(kMeansStride / sizeof(uint32_t)),
    static_cast<uint32_t>(kScalesStride / sizeof(uint32_t)),
    static_cast<uint32_t>(kRotationsStride / sizeof(uint32_t)),
    static_cast<uint32_t>(kOpacityStride / sizeof(uint32_t)),
    static_cast<uint32_t>(kShCoefficients * kShStride / sizeof(uint32_t)),
    static_cast<uint32_t>(kNormalsStride / sizeof(uint32_t)),
    static_cast<uint32_t>(kBaseColorRoughnessStride / sizeof(uint32_t)),
    static_cast<uint32_t>(kMetallicStride / sizeof(uint32_t)),
    static_cast<uint32_t>(kVisibilityCoefficients * kVisibilityStride / sizeof(uint32_t)),
};

// Largest Gaussian count whose SH words still index in uint32 (the widest stream).
constexpr uint64_t kMaxIndexedGaussians =
    std::numeric_limits<uint32_t>::max() / kStreamWords[kStreamSh];

// One stream of a layer: the whole stream over its capacity (invalid when the layer does
// not carry it) and the words each Gaussian spans.
struct EditStream
{
    BufferView view;
    uint32_t words = 0;

    bool present() const
    {
        return view.valid();
    }
};

// The layer's streams, with their views fetched now (a reserve() replaces them).
// Requires GPU parameters.
std::array<EditStream, kStreamCount> layerStreams(const GaussianCloud & layer)
{
    const CanonicalGaussianBufferViews core = layer.views();
    const RelightBufferViews & relight = layer.relightViews();
    const BufferView views[kStreamCount] = {
        core.means,           core.scales,          core.rotations,
        core.opacity,         core.shCoefficients,  relight.normals,
        relight.baseColorRoughness, relight.metallic, relight.visibility,
    };
    std::array<EditStream, kStreamCount> streams{};
    for (uint32_t id = 0; id < kStreamCount; ++id)
    {
        streams[id].view = views[id];
        streams[id].words = kStreamWords[id];
    }
    return streams;
}

// Whether a stream is part of a Transform snapshot (the rest are Full only).
bool inTransformScope(uint32_t id)
{
    return id == kStreamMeans || id == kStreamScales || id == kStreamRotations
           || id == kStreamSh || id == kStreamNormals;
}

// Host memory of one stream of a snapshot: its data and byte count (zero when empty).
struct HostStream
{
    const void * data = nullptr;
    size_t bytes = 0;
};

HostStream snapshotStream(const SparseSnapshot & s, uint32_t id)
{
    switch (id)
    {
    case kStreamMeans:
        return {s.means.data(), s.means.size() * sizeof(glm::vec4)};
    case kStreamScales:
        return {s.scales.data(), s.scales.size() * sizeof(glm::vec4)};
    case kStreamRotations:
        return {s.rotations.data(), s.rotations.size() * sizeof(glm::vec4)};
    case kStreamOpacity:
        return {s.opacities.data(), s.opacities.size() * sizeof(float)};
    case kStreamSh:
        return {s.shCoefficients.data(), s.shCoefficients.size() * sizeof(glm::vec4)};
    case kStreamNormals:
        return {s.normals.data(), s.normals.size() * sizeof(glm::vec4)};
    case kStreamBaseColorRoughness:
        return {s.baseColorRoughness.data(), s.baseColorRoughness.size() * sizeof(glm::vec4)};
    case kStreamMetallic:
        return {s.metallic.data(), s.metallic.size() * sizeof(float)};
    case kStreamVisibility:
        return {s.visibility.data(), s.visibility.size() * sizeof(float)};
    default:
        return {};
    }
}

// Checks a snapshot's streams against the layer's. A stream the snapshot holds must exist on
// the layer with count * words bytes. With `strict`, a stream the layer has must also be held
// (a delete's record is Full; a Transform snapshot cannot restore a delete).
void checkSnapshotStreams(const SparseSnapshot & snapshot,
                          const std::array<EditStream, kStreamCount> & streams, bool strict,
                          const char * where)
{
    const size_t count = snapshot.count();
    for (uint32_t id = 0; id < kStreamCount; ++id)
    {
        const HostStream host = snapshotStream(snapshot, id);
        const bool present = streams[id].present();
        if (host.bytes == 0)
        {
            VK_VALIDATE(!(strict && present),
                        std::string(where) + ": the snapshot lacks a stream the layer has ("
                            + std::to_string(id) + ")");
            continue;
        }
        VK_VALIDATE(present,
                    std::string(where) + ": the snapshot holds a stream the layer lacks ("
                        + std::to_string(id) + ")");
        VK_VALIDATE(host.bytes == count * kStreamWords[id] * sizeof(uint32_t),
                    std::string(where) + ": a snapshot stream does not match its indices ("
                        + std::to_string(id) + ")");
    }
}

// A view of the first `bytes` of `buffer` (stride one uint), grown to hold them.
BufferView grownView(Device & device, std::unique_ptr<Buffer> & buffer, VkDeviceSize bytes)
{
    ensureBytes(device, buffer, bytes);
    return BufferView{buffer->handle(), 0, bytes, sizeof(uint32_t)};
}

// Reads `bytes` of `view` into `out`, resized to hold them as T.
template <typename T>
void downloadWords(Device & device, const BufferView & view, size_t bytes, std::vector<T> & out)
{
    out.resize(bytes / sizeof(T));
    if (bytes > 0)
    {
        downloadView(device, view, out.data(), bytes);
    }
}

// The mask words (ceil(to / 32) of them) with bits [from, to) set, host side.
std::vector<uint32_t> rangeWords(size_t from, size_t to)
{
    std::vector<uint32_t> words(wordsForCount(static_cast<uint32_t>(to)), 0u);
    for (size_t i = from; i < to; ++i)
    {
        words[i >> 5] |= 1u << (i & 31u);
    }
    return words;
}

// The workgroups a mask operation over `threads` items dispatches: one invocation per item
// where that fits, grid-strided beyond 65535 workgroups (edit_mask_ops.slang).
uint32_t maskDispatchGroups(uint32_t threads)
{
    return std::min(dispatchGroups(threads, kWorkgroupSize), kMaxDispatchGroups);
}

// The requirement of every operation that copies a layer's GPU buffers: a borrowed
// layer (ownership model B) is refused, because Sabercat cannot query the usage flags of
// the borrowed buffers.
void requireOwnedParameters(const GaussianCloud & layer, const char * where)
{
    VK_VALIDATE(!layer.borrowedParameters(),
                std::string(where) + ": the layer's parameters are borrowed (ownership model B); "
                "this operation copies the layer's GPU buffers, which needs owned storage (or "
                "TRANSFER_SRC/TRANSFER_DST usage on the borrowed buffers, which cannot be checked)");
}

// The number of set bits among the first `count` bits of `words`.
uint64_t countBits(const std::vector<uint32_t> & words, uint32_t count)
{
    uint64_t total = 0;
    for (uint32_t i = 0; i < count; ++i)
    {
        if (((words[i >> 5] >> (i & 31u)) & 1u) != 0u)
        {
            ++total;
        }
    }
    return total;
}

} // namespace

// The select kernel (edit_select.slang) and the mask operations (edit_mask_ops.slang).
struct EditPass::SelectKernel : OneSetKernel
{
    using OneSetKernel::OneSetKernel;
};

struct EditPass::MaskKernel : OneSetKernel
{
    using OneSetKernel::OneSetKernel;
};

// The stream compaction kernel (edit_compact.slang) and the transform kernel
// (edit_transform.slang). One set each: every dispatch rewrites it before its own
// submission, and immediateSubmit waits for that submission before the next one.
struct EditPass::CompactKernel : OneSetKernel
{
    using OneSetKernel::OneSetKernel;
};

struct EditPass::TransformKernel : OneSetKernel
{
    using OneSetKernel::OneSetKernel;
};

// The layer merge's pipeline and descriptor sets. The pipeline and the first
// pool are made together, so the layout exists when the pipeline is built. Each
// further pool holds kSetsPerPool sets; each dispatch uses its own set, so the
// sets are never rewritten while a submission that uses them is pending.
struct EditPass::MergeKernel
{
    MergeKernel(Device & device, const std::string & shaderDir);
    ~MergeKernel();

    MergeKernel(const MergeKernel &) = delete;
    MergeKernel & operator=(const MergeKernel &) = delete;

    // The descriptor set for dispatch `index`, allocated on first use.
    VkDescriptorSet set(size_t index);
    // The DescriptorSet object whose pool holds set(index), for bindBuffer.
    DescriptorSet & owner(size_t index);

    // Makes one more pool. The first pool also creates the set layout.
    void addChunk();

    Device & device;
    Pipeline pipeline;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE; // owned by chunks.front()
    std::vector<std::unique_ptr<DescriptorSet>> chunks;
    std::vector<VkDescriptorSet> sets;
};

EditPass::MergeKernel::MergeKernel(Device & device, const std::string & shaderDir)
    : device(device)
{
    VkDevice dev = device.handle();
    // A throw below (a missing shader file is the common one) skips the destructor,
    // so the pipeline created so far is released here. The pools clean up through
    // their own destructors.
    try
    {
        addChunk();

        ShaderCompiler compiler(dev);
        VkShaderModule module = compiler.createShaderModule(
            compiler.loadSPIRV(shaderDir + "/layer_merge.comp.spv"));
        try
        {
            VkPipelineLayout layout = pipeline.createPipelineLayout(
                dev, setLayout, sizeof(LayerMergePush), VK_SHADER_STAGE_COMPUTE_BIT);
            pipeline.createComputePipeline(dev, module, layout);
        }
        catch (...)
        {
            compiler.destroyShaderModule(module);
            throw;
        }
        compiler.destroyShaderModule(module);
    }
    catch (...)
    {
        pipeline.destroy(dev);
        throw;
    }
}

EditPass::MergeKernel::~MergeKernel()
{
    pipeline.destroy(device.handle());
    // The first pool owns the layout the others allocate from, so it goes last.
    while (!chunks.empty())
    {
        chunks.pop_back();
    }
}

void EditPass::MergeKernel::addChunk()
{
    auto chunk = std::make_unique<DescriptorSet>(device.handle());
    if (setLayout == VK_NULL_HANDLE)
    {
        setLayout = chunk->createLayout(mergeBindings());
    }
    chunk->createPool(kSetsPerPool,
                      {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kSetsPerPool * kMergeBindingCount}});
    chunks.push_back(std::move(chunk));
}

VkDescriptorSet EditPass::MergeKernel::set(size_t index)
{
    while (sets.size() <= index)
    {
        // Every existing pool is full: start the next one.
        if (sets.size() == chunks.size() * kSetsPerPool)
        {
            addChunk();
        }
        sets.push_back(chunks.back()->allocate(setLayout));
    }
    return sets[index];
}

DescriptorSet & EditPass::MergeKernel::owner(size_t index)
{
    return *chunks[index / kSetsPerPool];
}

EditPass::EditPass(Device & device, const std::string & shaderDir)
    : m_device(device), m_scan(std::make_unique<GpuScan>(device, shaderDir)),
      m_shaderDir(shaderDir)
{
}

EditPass::~EditPass() = default;

uint32_t EditPass::exclusiveScan(const BufferView & input, const BufferView & output,
                                 uint32_t count, uint32_t inputStrideUints,
                                 uint32_t inputOffsetUints)
{
    if (count == 0)
    {
        return 0;
    }

    // Everything is checked before the submission: a throw inside the record
    // callback would leave the command buffer half-recorded.
    VK_VALIDATE(inputStrideUints > 0,
                "EditPass::exclusiveScan: the input stride must be at least one uint");
    VK_VALIDATE(input.offset == 0 && output.offset == 0,
                "EditPass::exclusiveScan: both views must start at offset 0");
    VK_VALIDATE(input.buffer != VK_NULL_HANDLE && output.buffer != VK_NULL_HANDLE,
                "EditPass::exclusiveScan: both views need a buffer");
    VK_VALIDATE(input.buffer != output.buffer,
                "EditPass::exclusiveScan: input and output must be different buffers");

    const VkDeviceSize uintBytes = sizeof(uint32_t);
    const VkDeviceSize lastIndex = count - 1;
    const VkDeviceSize inputBytesNeeded =
        (lastIndex * inputStrideUints + inputOffsetUints + 1) * uintBytes;
    VK_VALIDATE(input.size >= inputBytesNeeded,
                "EditPass::exclusiveScan: the input view holds fewer than count elements");
    VK_VALIDATE(output.size >= static_cast<VkDeviceSize>(count) * uintBytes,
                "EditPass::exclusiveScan: the output view holds fewer than count uints");

    m_device.immediateSubmit(
        [&](VkCommandBuffer cmd)
        {
            m_scan->record(cmd, input.buffer, output.buffer, count, inputStrideUints,
                           inputOffsetUints);
        });

    // The total is the inclusive end of the scan: the last exclusive output plus
    // the last input. Both are one uint, read back through a staging copy.
    uint32_t lastOutput = 0;
    const BufferView outputView{output.buffer, lastIndex * uintBytes, uintBytes,
                                static_cast<uint32_t>(uintBytes)};
    downloadView(m_device, outputView, &lastOutput, sizeof(lastOutput));

    uint32_t lastInput = 0;
    const VkDeviceSize inputOffset = (lastIndex * inputStrideUints + inputOffsetUints) * uintBytes;
    const BufferView inputView{input.buffer, inputOffset, uintBytes,
                               static_cast<uint32_t>(uintBytes)};
    downloadView(m_device, inputView, &lastInput, sizeof(lastInput));

    return lastOutput + lastInput;
}

MergedLayout planLayerMerge(const std::vector<MergeLayer> & layers)
{
    return planLayers(layers, true);
}

void EditPass::mergeLayers(const std::vector<MergeLayer> & layers, GaussianCloud & destination)
{
    // Every check runs before the first GPU call: a throw after the upload or the
    // record would leave the destination half-written.
    const MergedLayout plan = planLayers(layers, false);
    VK_VALIDATE(hasGpuParameters(destination),
                "EditPass::mergeLayers: the destination has no GPU parameters "
                "(allocate or upload it first)");
    VK_VALIDATE(destination.count() == plan.count,
                "EditPass::mergeLayers: the destination holds " + std::to_string(destination.count())
                    + " Gaussians but the layers merge to " + std::to_string(plan.count)
                    + " (size it with planLayerMerge)");
    VK_VALIDATE(plan.count <= std::numeric_limits<uint32_t>::max(),
                "EditPass::mergeLayers: the merged scene has more Gaussians than a uint32 "
                "index holds");
    for (const MergeLayer & layer : layers)
    {
        VK_VALIDATE(layer.first != &destination,
                    "EditPass::mergeLayers: the destination cannot be one of the layers");
    }

    // A relight stream the destination carries must be on every layer. A stream it
    // lacks is simply not written.
    const RelightPresence carried = gpuPresence(destination);
    VK_VALIDATE(!carried.normals || plan.presence.normals,
                "EditPass::mergeLayers: the destination has a normals stream that not every "
                "layer has");
    VK_VALIDATE(!carried.baseColorRoughness || plan.presence.baseColorRoughness,
                "EditPass::mergeLayers: the destination has a baseColorRoughness stream that "
                "not every layer has");
    VK_VALIDATE(!carried.metallic || plan.presence.metallic,
                "EditPass::mergeLayers: the destination has a metallic stream that not every "
                "layer has");
    VK_VALIDATE(!carried.visibility || plan.presence.visibility,
                "EditPass::mergeLayers: the destination has a visibility stream that not every "
                "layer has");

    uint32_t flags = 0;
    if (carried.normals)
    {
        flags |= kMergeFlagNormals;
    }
    if (carried.baseColorRoughness)
    {
        flags |= kMergeFlagBaseColorRoughness;
    }
    if (carried.metallic)
    {
        flags |= kMergeFlagMetallic;
    }
    if (carried.visibility)
    {
        flags |= kMergeFlagVisibility;
    }

    // Host work first: the placements, and the destination of each layer.
    const size_t layerCount = layers.size();
    std::vector<SimilarityParamsHost> placements;
    std::vector<uint32_t> firsts;
    std::vector<uint32_t> counts;
    placements.reserve(layerCount);
    firsts.reserve(layerCount);
    counts.reserve(layerCount);
    uint32_t first = 0;
    for (const MergeLayer & layer : layers)
    {
        const GaussianCloud & source = *layer.first;
        placements.push_back(makePlacement(layer.second, source));
        firsts.push_back(first);
        counts.push_back(static_cast<uint32_t>(source.count()));
        first += static_cast<uint32_t>(source.count());
    }

    if (!m_merge)
    {
        m_merge = std::make_unique<MergeKernel>(m_device, m_shaderDir);
    }
    if (!m_dummy)
    {
        m_dummy = std::make_unique<Buffer>(m_device, kDummyBytes);
    }

    const VkDeviceSize paramsBytes = layerCount * sizeof(SimilarityParamsHost);
    ensureBytes(m_device, m_mergeParams, paramsBytes);
    m_mergeParams->upload(placements.data(), static_cast<size_t>(paramsBytes));

    // Descriptor sets: each dispatch owns one, written here (before the submission)
    // and used only by that submission.
    const VkBuffer dummy = m_dummy->handle();
    const VkDeviceSize dummyBytes = m_dummy->size();
    const auto bindStream = [&](DescriptorSet & owner, VkDescriptorSet set, uint32_t binding,
                                const BufferView & stream)
    {
        if (stream.valid())
        {
            owner.bindBuffer(set, binding, stream.buffer, stream.offset, stream.size,
                             VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        }
        else
        {
            owner.bindBuffer(set, binding, dummy, 0, dummyBytes,
                             VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        }
    };

    const CanonicalGaussianBufferViews dst = destination.views();
    const RelightBufferViews & dstRelight = destination.relightViews();
    std::vector<VkDescriptorSet> layerSets(layerCount);
    for (size_t i = 0; i < layerCount; ++i)
    {
        const GaussianCloud & source = *layers[i].first;
        const CanonicalGaussianBufferViews src = source.views();
        const RelightBufferViews & srcRelight = source.relightViews();
        const VkDescriptorSet set = m_merge->set(i);
        DescriptorSet & owner = m_merge->owner(i);
        layerSets[i] = set;

        owner.bindBuffer(set, kMergeBindingParams, m_mergeParams->handle(), 0, paramsBytes,
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

        bindStream(owner, set, kMergeSourceBase + kMeans, src.means);
        bindStream(owner, set, kMergeSourceBase + kScales, src.scales);
        bindStream(owner, set, kMergeSourceBase + kRotations, src.rotations);
        bindStream(owner, set, kMergeSourceBase + kOpacity, src.opacity);
        bindStream(owner, set, kMergeSourceBase + kSh, src.shCoefficients);
        bindStream(owner, set, kMergeDestinationBase + kMeans, dst.means);
        bindStream(owner, set, kMergeDestinationBase + kScales, dst.scales);
        bindStream(owner, set, kMergeDestinationBase + kRotations, dst.rotations);
        bindStream(owner, set, kMergeDestinationBase + kOpacity, dst.opacity);
        bindStream(owner, set, kMergeDestinationBase + kSh, dst.shCoefficients);

        bindStream(owner, set, kMergeSourceBase + kNormals,
                   streamIf(carried.normals, srcRelight.normals));
        bindStream(owner, set, kMergeSourceBase + kBaseColorRoughness,
                   streamIf(carried.baseColorRoughness, srcRelight.baseColorRoughness));
        bindStream(owner, set, kMergeSourceBase + kMetallic,
                   streamIf(carried.metallic, srcRelight.metallic));
        bindStream(owner, set, kMergeSourceBase + kVisibility,
                   streamIf(carried.visibility, srcRelight.visibility));
        bindStream(owner, set, kMergeDestinationBase + kNormals,
                   streamIf(carried.normals, dstRelight.normals));
        bindStream(owner, set, kMergeDestinationBase + kBaseColorRoughness,
                   streamIf(carried.baseColorRoughness, dstRelight.baseColorRoughness));
        bindStream(owner, set, kMergeDestinationBase + kMetallic,
                   streamIf(carried.metallic, dstRelight.metallic));
        bindStream(owner, set, kMergeDestinationBase + kVisibility,
                   streamIf(carried.visibility, dstRelight.visibility));
    }

    // One dispatch per layer. Layers write disjoint destination ranges and never read
    // what another writes, so no barrier is needed between them.
    m_device.immediateSubmit(
        [&](VkCommandBuffer cmd)
        {
            const VkPipelineLayout pipelineLayout = m_merge->pipeline.getLayout();
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              m_merge->pipeline.getHandle());
            for (size_t i = 0; i < layerCount; ++i)
            {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
                                        &layerSets[i], 0, nullptr);
                const LayerMergePush push{firsts[i], static_cast<uint32_t>(i), flags, 0};
                vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                   sizeof(push), &push);
                vkCmdDispatch(cmd, dispatchGroups(counts[i], kWorkgroupSize), 1, 1);
            }
        });
}

void EditPass::select(const GaussianCloud & layer, const LayerTransform & placement,
                      const SelectionQuery & query, SelectionMask & mask)
{
    // Every check runs before the first GPU call, so a throw leaves the mask untouched.
    VK_VALIDATE(hasGpuParameters(layer),
                "EditPass::select: the layer has no GPU parameters (upload it first)");
    VK_VALIDATE(mask.capacity() >= layer.count(),
                "EditPass::select: the mask holds " + std::to_string(mask.capacity())
                    + " Gaussians but the layer has " + std::to_string(layer.count()));
    VK_VALIDATE(layer.count() <= std::numeric_limits<uint32_t>::max(),
                "EditPass::select: the layer has more Gaussians than a uint32 index holds");
    validateLayerTransform(placement, "EditPass::select: placement");
    VK_VALIDATE(static_cast<uint32_t>(query.shape) <= 2u,
                "EditPass::select: the query's shape is not Box, Sphere or Rectangle");
    VK_VALIDATE(static_cast<uint32_t>(query.op) <= 2u,
                "EditPass::select: the query's op is not Replace, Add or Subtract");
    if (query.shape == SelectShape::Rectangle)
    {
        VK_VALIDATE(query.camera.fx > 0.0f && query.camera.fy > 0.0f,
                    "EditPass::select: a rectangle needs the camera of its view (fx and fy "
                    "above zero, e.g. Renderer::lastForwardCamera())");
    }

    const uint32_t count = static_cast<uint32_t>(layer.count());
    if (count == 0)
    {
        return;
    }

    // Host work first: the parameter block, uploaded before the submission.
    SelectParamsHost params{};
    if (placement.isIdentity())
    {
        params.layerToWorld = glm::mat4(1.0f);
    }
    else
    {
        // The merge's placement: the BakeFrame's double matrix, rounded to fp32.
        const BakeFrame frame = makeBakeFrame(placement);
        for (int column = 0; column < 4; ++column)
        {
            for (int row = 0; row < 4; ++row)
            {
                params.layerToWorld[column][row] = static_cast<float>(frame.matrix[column][row]);
            }
        }
    }

    params.boxAxis[0] = glm::vec4(query.boxAxisX, 0.0f);
    params.boxAxis[1] = glm::vec4(query.boxAxisY, 0.0f);
    params.boxAxis[2] = glm::vec4(query.boxAxisZ, 0.0f);
    params.centerRadius = glm::vec4(query.center, query.radius);
    params.halfExtents = glm::vec4(query.halfExtents, 0.0f);

    if (query.shape == SelectShape::Rectangle)
    {
        params.worldToCamera = query.camera.worldToCamera;
        params.intrinsics =
            glm::vec4(query.camera.fx, query.camera.fy, query.camera.cx, query.camera.cy);
        // Any corner order: the call normalizes it to x0 <= x1 and y0 <= y1.
        const glm::vec4 r = query.rect;
        params.cameraRect = glm::vec4(std::min(r.x, r.z), std::min(r.y, r.w),
                                      std::max(r.x, r.z), std::max(r.y, r.w));
    }
    else
    {
        params.worldToCamera = glm::mat4(1.0f);
        params.intrinsics = glm::vec4(0.0f);
        params.cameraRect = glm::vec4(0.0f);
    }

    params.shape = static_cast<uint32_t>(query.shape);
    params.op = static_cast<uint32_t>(query.op);
    params.count = count;
    params.wordCount = wordsForCount(count);

    if (!m_selectParams)
    {
        m_selectParams = std::make_unique<Buffer>(m_device, sizeof(SelectParamsHost));
    }
    m_selectParams->upload(&params, sizeof(params));
    if (!m_select)
    {
        m_select = std::make_unique<SelectKernel>(m_device, m_shaderDir, "edit_select.comp.spv",
                                                  kSelectBindingCount, 0u);
    }
    if (!m_dummy)
    {
        m_dummy = std::make_unique<Buffer>(m_device, kDummyBytes);
    }

    // The one descriptor set, written before the submission that uses it.
    const CanonicalGaussianBufferViews src = layer.views();
    const VkDescriptorSet set = m_select->set;
    DescriptorSet & descriptors = m_select->descriptors;
    bindStorage(descriptors, set, 0, mask.view(), *m_dummy);
    bindStorage(descriptors, set, 1, src.means, *m_dummy);
    bindStorage(descriptors, set, 2,
                m_selectParams->view(static_cast<uint32_t>(sizeof(SelectParamsHost))), *m_dummy);

    // One thread per mask word.
    const uint32_t groups = dispatchGroups(params.wordCount, kWorkgroupSize);
    m_device.immediateSubmit(
        [&](VkCommandBuffer cmd)
        {
            const VkPipelineLayout pipelineLayout = m_select->pipeline.getLayout();
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              m_select->pipeline.getHandle());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
                                    &set, 0, nullptr);
            vkCmdDispatch(cmd, groups, 1, 1);
        });
}

void EditPass::runMaskOp(uint32_t mode, uint32_t count, uint32_t wordCount, uint32_t flag,
                         uint32_t threads, const SelectionMask & mask, const BufferView & means,
                         const BufferView & flags)
{
    if (!m_maskOps)
    {
        m_maskOps = std::make_unique<MaskKernel>(m_device, m_shaderDir, "edit_mask_ops.comp.spv",
                                                 kMaskBindingCount,
                                                 static_cast<uint32_t>(sizeof(MaskOpsPush)));
    }
    if (!m_dummy)
    {
        m_dummy = std::make_unique<Buffer>(m_device, kDummyBytes);
    }

    // One partial per workgroup. The dispatch is grid-strided (edit_mask_ops.slang), so it
    // holds at most 65535 workgroups however many items there are.
    const uint32_t groups = maskDispatchGroups(threads);
    const size_t partials = static_cast<size_t>(groups);
    ensureBytes(m_device, m_countPartials, partials * sizeof(uint32_t));
    ensureBytes(m_device, m_centroidPartials, partials * sizeof(glm::vec4));

    const MaskOpsPush push{mode, count, wordCount, flag, groups};
    const VkDescriptorSet set = m_maskOps->set;
    DescriptorSet & descriptors = m_maskOps->descriptors;
    descriptors.bindBuffer(set, 0, mask.buffer().handle(), 0, mask.buffer().size(),
                           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    bindStorage(descriptors, set, 1, means, *m_dummy);
    bindStorage(descriptors, set, 2, flags, *m_dummy);
    descriptors.bindBuffer(set, 3, m_countPartials->handle(), 0, m_countPartials->size(),
                           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    descriptors.bindBuffer(set, 4, m_centroidPartials->handle(), 0, m_centroidPartials->size(),
                           VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

    m_device.immediateSubmit(
        [&](VkCommandBuffer cmd)
        {
            const VkPipelineLayout pipelineLayout = m_maskOps->pipeline.getLayout();
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                              m_maskOps->pipeline.getHandle());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
                                    &set, 0, nullptr);
            vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                               &push);
            vkCmdDispatch(cmd, groups, 1, 1);
        });
}

void EditPass::clearMask(SelectionMask & mask)
{
    const uint32_t words = static_cast<uint32_t>(mask.wordCount());
    runMaskOp(kMaskModeClear, 0, words, 0, words, mask, BufferView{}, BufferView{});
}

void EditPass::selectAll(SelectionMask & mask, uint32_t count)
{
    VK_VALIDATE(count <= mask.capacity(),
                "EditPass::selectAll: " + std::to_string(count)
                    + " Gaussians exceed the mask's capacity of "
                    + std::to_string(mask.capacity()));
    // Every word is written: the words past count become zero.
    const uint32_t words = static_cast<uint32_t>(mask.wordCount());
    runMaskOp(kMaskModeAll, count, words, 0, words, mask, BufferView{}, BufferView{});
}

void EditPass::invertMask(SelectionMask & mask, uint32_t count)
{
    VK_VALIDATE(count <= mask.capacity(),
                "EditPass::invertMask: " + std::to_string(count)
                    + " Gaussians exceed the mask's capacity of "
                    + std::to_string(mask.capacity()));
    const uint32_t words = static_cast<uint32_t>(mask.wordCount());
    runMaskOp(kMaskModeInvert, count, words, 0, words, mask, BufferView{}, BufferView{});
}

uint32_t EditPass::countSelected(const SelectionMask & mask, uint32_t count)
{
    VK_VALIDATE(count <= mask.capacity(),
                "EditPass::countSelected: " + std::to_string(count)
                    + " Gaussians exceed the mask's capacity of "
                    + std::to_string(mask.capacity()));
    if (count == 0)
    {
        return 0;
    }

    const uint32_t words = wordsForCount(count);
    runMaskOp(kMaskModePopcount, count, words, 0, words, mask, BufferView{}, BufferView{});

    // The per-workgroup counts, summed on the host in 64 bits.
    const uint32_t groups = maskDispatchGroups(words);
    std::vector<uint32_t> partials(groups, 0u);
    m_countPartials->download(partials.data(), groups * sizeof(uint32_t));
    uint64_t total = 0;
    for (const uint32_t partial : partials)
    {
        total += partial;
    }
    return static_cast<uint32_t>(total);
}

MaskCentroid EditPass::maskedCentroid(const GaussianCloud & layer, const SelectionMask & mask)
{
    VK_VALIDATE(hasGpuParameters(layer),
                "EditPass::maskedCentroid: the layer has no GPU parameters (upload it first)");
    VK_VALIDATE(mask.capacity() >= layer.count(),
                "EditPass::maskedCentroid: the mask holds " + std::to_string(mask.capacity())
                    + " Gaussians but the layer has " + std::to_string(layer.count()));
    VK_VALIDATE(layer.count() <= std::numeric_limits<uint32_t>::max(),
                "EditPass::maskedCentroid: the layer has more Gaussians than a uint32 index holds");

    MaskCentroid result;
    const uint32_t count = static_cast<uint32_t>(layer.count());
    if (count == 0)
    {
        return result;
    }

    // One thread per Gaussian; the partials are (sum.xyz, count) per workgroup.
    const CanonicalGaussianBufferViews src = layer.views();
    runMaskOp(kMaskModeCentroid, count, static_cast<uint32_t>(mask.wordCount()), 0, count, mask,
              src.means, BufferView{});

    const uint32_t groups = maskDispatchGroups(count);
    std::vector<glm::vec4> partials(groups, glm::vec4(0.0f));
    m_centroidPartials->download(partials.data(), groups * sizeof(glm::vec4));
    for (const glm::vec4 & partial : partials)
    {
        // Each partial's count is a small integer (at most the workgroup size), exact in
        // fp32; the host adds the partials in double.
        result.sum += glm::dvec3(static_cast<double>(partial.x), static_cast<double>(partial.y),
                                 static_cast<double>(partial.z));
        result.count += static_cast<uint64_t>(partial.w);
    }
    return result;
}

void EditPass::expandMaskToFlags(const SelectionMask & mask, uint32_t count, bool keepUnselected,
                                 const BufferView & flags)
{
    VK_VALIDATE(count <= mask.capacity(),
                "EditPass::expandMaskToFlags: " + std::to_string(count)
                    + " Gaussians exceed the mask's capacity of "
                    + std::to_string(mask.capacity()));
    VK_VALIDATE(flags.buffer != VK_NULL_HANDLE && flags.offset == 0,
                "EditPass::expandMaskToFlags: the flags must be a buffer view at offset 0");
    VK_VALIDATE(flags.size >= static_cast<VkDeviceSize>(count) * sizeof(uint32_t),
                "EditPass::expandMaskToFlags: the flags view holds fewer than count uints");
    if (count == 0)
    {
        return;
    }

    const uint32_t flag = keepUnselected ? 1u : 0u;
    runMaskOp(kMaskModeExpand, count, static_cast<uint32_t>(mask.wordCount()), flag, count, mask,
              BufferView{}, flags);
}

size_t SparseSnapshot::bytes() const
{
    return indices.size() * sizeof(uint32_t) + means.size() * sizeof(glm::vec4)
           + scales.size() * sizeof(glm::vec4) + rotations.size() * sizeof(glm::vec4)
           + shCoefficients.size() * sizeof(glm::vec4) + opacities.size() * sizeof(float)
           + normals.size() * sizeof(glm::vec4) + baseColorRoughness.size() * sizeof(glm::vec4)
           + metallic.size() * sizeof(float) + visibility.size() * sizeof(float);
}

void EditPass::runCompact(const CompactCall & call)
{
    if (!m_compact)
    {
        m_compact = std::make_unique<CompactKernel>(m_device, m_shaderDir, "edit_compact.comp.spv",
                                                    kCompactBindingCount,
                                                    static_cast<uint32_t>(sizeof(CompactPush)));
    }
    if (!m_dummy)
    {
        m_dummy = std::make_unique<Buffer>(m_device, kDummyBytes);
    }
    VK_VALIDATE(call.threads > 0, "EditPass: a compaction dispatch needs work items");

    // At most 65535 workgroups; the shader grid-strides over the work items beyond them.
    const uint32_t groups =
        std::min(dispatchGroups(call.threads, kWorkgroupSize), kMaxDispatchGroups);
    const CompactPush push{call.mode, call.count, call.wordsPerElement, call.base, groups};

    CompactKernel & kernel = *m_compact;
    const VkDescriptorSet set = kernel.set;
    DescriptorSet & descriptors = kernel.descriptors;
    bindStorage(descriptors, set, 0, call.flags, *m_dummy);
    bindStorage(descriptors, set, 1, call.scan, *m_dummy);
    bindStorage(descriptors, set, 2, call.src, *m_dummy);
    bindStorage(descriptors, set, 3, call.dst, *m_dummy);
    bindStorage(descriptors, set, 4, call.aux, *m_dummy);

    const bool copy = call.copyTo.valid() && call.copyBytes > 0;
    m_device.immediateSubmit(
        [&](VkCommandBuffer cmd)
        {
            const VkPipelineLayout pipelineLayout = kernel.pipeline.getLayout();
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.pipeline.getHandle());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
                                    &set, 0, nullptr);
            vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                               &push);
            vkCmdDispatch(cmd, groups, 1, 1);
            if (copy)
            {
                // The shader's reads of src and writes to dst finish before the copy reads dst.
                VkMemoryBarrier barrier{};
                barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
                barrier.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
                barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
                vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                     VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, nullptr,
                                     0, nullptr);
                VkBufferCopy region{};
                region.srcOffset = call.dst.offset;
                region.dstOffset = call.copyTo.offset;
                region.size = call.copyBytes;
                vkCmdCopyBuffer(cmd, call.dst.buffer, call.copyTo.buffer, 1, &region);
            }
        });
}

SparseSnapshot EditPass::gatherSelected(const GaussianCloud & layer, const SelectionMask & mask,
                                        SnapshotScope scope)
{
    requireOwnedParameters(layer, "EditPass::gatherSelected");
    VK_VALIDATE(hasGpuParameters(layer),
                "EditPass::gatherSelected: the layer has no GPU parameters (upload it first)");
    VK_VALIDATE(mask.capacity() >= layer.count(),
                "EditPass::gatherSelected: the mask holds " + std::to_string(mask.capacity())
                    + " Gaussians but the layer has " + std::to_string(layer.count()));
    VK_VALIDATE(layer.count() <= kMaxIndexedGaussians,
                "EditPass::gatherSelected: the layer has more Gaussians than a uint32 word index "
                "holds");

    SparseSnapshot snapshot;
    snapshot.scope = scope;
    snapshot.shDegree = layer.shDegree();
    const uint32_t count = static_cast<uint32_t>(layer.count());
    const uint32_t k = countSelected(mask, count);
    if (k == 0)
    {
        return snapshot;
    }
    const bool full = scope == SnapshotScope::Full;

    // The selected flags and their scan: a selected Gaussian's slot in each packed stream.
    const BufferView flags = grownView(m_device, m_editFlags, count * sizeof(uint32_t));
    const BufferView scan = grownView(m_device, m_editScan, count * sizeof(uint32_t));
    expandMaskToFlags(mask, count, false, flags);
    const uint32_t selected = exclusiveScan(flags, scan, count);
    VK_VALIDATE(selected == k,
                "EditPass::gatherSelected: the selection's flags disagree with its count");

    // The layer-local index of each selected Gaussian, ascending.
    const BufferView indexView = grownView(m_device, m_editIndices, k * sizeof(uint32_t));
    CompactCall index;
    index.mode = kCompactModeIndex;
    index.count = count;
    index.threads = count;
    index.flags = flags;
    index.scan = scan;
    index.dst = indexView;
    runCompact(index);
    downloadWords(m_device, indexView, k * sizeof(uint32_t), snapshot.indices);

    // Each stream: GATHER the selected elements into scratch, then read them back.
    const std::array<EditStream, kStreamCount> streams = layerStreams(layer);
    const auto gather = [&](uint32_t id, auto & out)
    {
        const EditStream & stream = streams[id];
        if (!stream.present() || (!full && !inTransformScope(id)))
        {
            return;
        }
        const size_t bytes = static_cast<size_t>(k) * stream.words * sizeof(uint32_t);
        const BufferView dst = grownView(m_device, m_editScratch, bytes);
        CompactCall call;
        call.mode = kCompactModeGather;
        call.count = count;
        call.wordsPerElement = stream.words;
        call.threads = count * stream.words;
        call.flags = flags;
        call.scan = scan;
        call.src = stream.view;
        call.dst = dst;
        runCompact(call);
        downloadWords(m_device, dst, bytes, out);
    };
    gather(kStreamMeans, snapshot.means);
    gather(kStreamScales, snapshot.scales);
    gather(kStreamRotations, snapshot.rotations);
    gather(kStreamOpacity, snapshot.opacities);
    gather(kStreamSh, snapshot.shCoefficients);
    gather(kStreamNormals, snapshot.normals);
    gather(kStreamBaseColorRoughness, snapshot.baseColorRoughness);
    gather(kStreamMetallic, snapshot.metallic);
    gather(kStreamVisibility, snapshot.visibility);
    return snapshot;
}

void EditPass::scatterSnapshot(GaussianCloud & layer, const SparseSnapshot & snapshot)
{
    requireOwnedParameters(layer, "EditPass::scatterSnapshot");
    VK_VALIDATE(hasGpuParameters(layer),
                "EditPass::scatterSnapshot: the layer has no GPU parameters (upload it first)");
    if (snapshot.count() == 0)
    {
        return;
    }
    const uint64_t layerCount = layer.count();
    for (const uint32_t index : snapshot.indices)
    {
        VK_VALIDATE(index < layerCount,
                    "EditPass::scatterSnapshot: index " + std::to_string(index)
                        + " is not below the layer's count " + std::to_string(layerCount));
    }
    const std::array<EditStream, kStreamCount> streams = layerStreams(layer);
    checkSnapshotStreams(snapshot, streams, false, "EditPass::scatterSnapshot");

    // The indices, on the device: the scatter's index list.
    const uint32_t k = static_cast<uint32_t>(snapshot.count());
    const size_t indexBytes = static_cast<size_t>(k) * sizeof(uint32_t);
    ensureBytes(m_device, m_editIndices, indexBytes);
    m_editIndices->upload(snapshot.indices.data(), indexBytes);
    const BufferView indexView{m_editIndices->handle(), 0, indexBytes, sizeof(uint32_t)};

    // Each stream the snapshot holds: upload its values, then SCATTER them into the layer.
    for (uint32_t id = 0; id < kStreamCount; ++id)
    {
        const HostStream host = snapshotStream(snapshot, id);
        if (host.bytes == 0)
        {
            continue;
        }
        const EditStream & stream = streams[id];
        ensureBytes(m_device, m_editPacked, host.bytes);
        m_editPacked->upload(host.data, host.bytes);
        CompactCall call;
        call.mode = kCompactModeScatter;
        call.count = k;
        call.wordsPerElement = stream.words;
        call.threads = k * stream.words;
        call.src = BufferView{m_editPacked->handle(), 0, host.bytes, sizeof(uint32_t)};
        call.dst = stream.view;
        call.aux = indexView;
        runCompact(call);
    }
}

void EditPass::transformSelected(GaussianCloud & layer, const SelectionMask & mask,
                                 const LayerTransform & layerSpace)
{
    VK_VALIDATE(hasGpuParameters(layer),
                "EditPass::transformSelected: the layer has no GPU parameters (upload it first)");
    validateLayerTransform(layerSpace, "EditPass::transformSelected: layer transform");
    VK_VALIDATE(mask.capacity() >= layer.count(),
                "EditPass::transformSelected: the mask holds " + std::to_string(mask.capacity())
                    + " Gaussians but the layer has " + std::to_string(layer.count()));
    VK_VALIDATE(layer.count() <= kMaxIndexedGaussians,
                "EditPass::transformSelected: the layer has more Gaussians than a uint32 word "
                "index holds");
    if (layerSpace.isIdentity() || layer.count() == 0)
    {
        return;
    }

    // The placement, uploaded before the submission. The same BakeFrame and rounding as the
    // merge (makePlacement), with the layer's own count and degree.
    const uint32_t count = static_cast<uint32_t>(layer.count());
    const SimilarityParamsHost placement = makePlacement(layerSpace, layer);
    ensureBytes(m_device, m_transformParams, sizeof(SimilarityParamsHost));
    m_transformParams->upload(&placement, sizeof(placement));
    if (!m_transform)
    {
        m_transform = std::make_unique<TransformKernel>(
            m_device, m_shaderDir, "edit_transform.comp.spv", kTransformBindingCount,
            static_cast<uint32_t>(sizeof(TransformPush)));
    }
    if (!m_dummy)
    {
        m_dummy = std::make_unique<Buffer>(m_device, kDummyBytes);
    }

    const std::array<EditStream, kStreamCount> streams = layerStreams(layer);
    TransformPush push{};
    push.count = count;
    push.hasNormals = streams[kStreamNormals].present() ? 1u : 0u;

    TransformKernel & kernel = *m_transform;
    const VkDescriptorSet set = kernel.set;
    DescriptorSet & descriptors = kernel.descriptors;
    bindStorage(descriptors, set, 0, mask.view(), *m_dummy);
    bindStorage(descriptors, set, 1,
                BufferView{m_transformParams->handle(), 0, sizeof(SimilarityParamsHost),
                           static_cast<uint32_t>(sizeof(SimilarityParamsHost))},
                *m_dummy);
    bindStorage(descriptors, set, 2, streams[kStreamMeans].view, *m_dummy);
    bindStorage(descriptors, set, 3, streams[kStreamScales].view, *m_dummy);
    bindStorage(descriptors, set, 4, streams[kStreamRotations].view, *m_dummy);
    bindStorage(descriptors, set, 5, streams[kStreamSh].view, *m_dummy);
    bindStorage(descriptors, set, 6, streams[kStreamNormals].view, *m_dummy);

    // One invocation per Gaussian. Up to 65535 workgroups in x; a larger layer adds rows in
    // y, and the shader indexes the Gaussian as id.x + id.y * 65535 * WORKGROUP_SIZE.
    const uint32_t groups = dispatchGroups(count, kWorkgroupSize);
    const uint32_t groupsX = std::min(groups, kMaxDispatchGroups);
    const uint32_t groupsY = dispatchGroups(groups, kMaxDispatchGroups);
    m_device.immediateSubmit(
        [&](VkCommandBuffer cmd)
        {
            const VkPipelineLayout pipelineLayout = kernel.pipeline.getLayout();
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.pipeline.getHandle());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
                                    &set, 0, nullptr);
            vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push),
                               &push);
            vkCmdDispatch(cmd, groupsX, groupsY, 1);
        });
}

DeleteRecord EditPass::deleteSelected(GaussianCloud & layer, SelectionMask & mask,
                                      const BufferView * remapOut)
{
    requireOwnedParameters(layer, "EditPass::deleteSelected");
    VK_VALIDATE(hasGpuParameters(layer),
                "EditPass::deleteSelected: the layer has no GPU parameters (upload it first)");
    const uint32_t count = static_cast<uint32_t>(layer.count());
    VK_VALIDATE(layer.count() <= kMaxIndexedGaussians,
                "EditPass::deleteSelected: the layer has more Gaussians than a uint32 word index "
                "holds");
    if (remapOut != nullptr)
    {
        VK_VALIDATE(remapOut->buffer != VK_NULL_HANDLE && remapOut->offset == 0,
                    "EditPass::deleteSelected: remapOut must be a buffer view at offset 0");
        VK_VALIDATE(remapOut->size >= static_cast<VkDeviceSize>(count) * sizeof(uint32_t),
                    "EditPass::deleteSelected: remapOut holds fewer than one uint per Gaussian");
    }

    DeleteRecord record;
    record.oldCount = count;
    record.keptCount = count;
    const uint32_t k = countSelected(mask, count);
    if (k == 0)
    {
        return record;
    }
    if (k == count)
    {
        throw std::invalid_argument("EditPass::deleteSelected: every Gaussian of the layer is "
                                    "selected; remove the layer instead");
    }

    // The record first: the selection and the deleted Gaussians, in full.
    record.selectionWords = mask.download(count);
    record.deleted = gatherSelected(layer, mask, SnapshotScope::Full);

    // The keep flags (1 for an unselected Gaussian) and their scan: the kept slot of each.
    const BufferView flags = grownView(m_device, m_editFlags, count * sizeof(uint32_t));
    const BufferView scan = grownView(m_device, m_editScan, count * sizeof(uint32_t));
    expandMaskToFlags(mask, count, true, flags);
    const uint32_t kept = exclusiveScan(flags, scan, count);
    VK_VALIDATE(kept == count - k,
                "EditPass::deleteSelected: the keep flags disagree with the selection");

    if (remapOut != nullptr)
    {
        CompactCall remap;
        remap.mode = kCompactModeRemap;
        remap.count = count;
        remap.threads = count;
        remap.flags = flags;
        remap.scan = scan;
        remap.dst = BufferView{remapOut->buffer, 0,
                               static_cast<VkDeviceSize>(count) * sizeof(uint32_t),
                               static_cast<uint32_t>(sizeof(uint32_t))};
        runCompact(remap);
    }

    // Each present stream: COMPACT the kept elements into scratch and copy them back over the
    // stream's first kept elements, in one submission.
    const std::array<EditStream, kStreamCount> streams = layerStreams(layer);
    for (uint32_t id = 0; id < kStreamCount; ++id)
    {
        const EditStream & stream = streams[id];
        if (!stream.present())
        {
            continue;
        }
        const size_t keptBytes = static_cast<size_t>(kept) * stream.words * sizeof(uint32_t);
        const BufferView scratch = grownView(m_device, m_editScratch, keptBytes);
        CompactCall call;
        call.mode = kCompactModeCompact;
        call.count = count;
        call.wordsPerElement = stream.words;
        call.threads = count * stream.words;
        call.flags = flags;
        call.scan = scan;
        call.src = stream.view;
        call.dst = scratch;
        call.copyTo = stream.view;
        call.copyBytes = keptBytes;
        runCompact(call);
    }

    layer.setCount(kept);
    clearMask(mask);
    record.keptCount = kept;
    return record;
}

void EditPass::undoDelete(GaussianCloud & layer, SelectionMask & mask, const DeleteRecord & record)
{
    if (record.deleted.count() == 0)
    {
        return;
    }
    requireOwnedParameters(layer, "EditPass::undoDelete");
    VK_VALIDATE(hasGpuParameters(layer),
                "EditPass::undoDelete: the layer has no GPU parameters (upload it first)");
    VK_VALIDATE(layer.count() == record.keptCount,
                "EditPass::undoDelete: the layer holds " + std::to_string(layer.count())
                    + " Gaussians but the delete kept " + std::to_string(record.keptCount));
    VK_VALIDATE(layer.capacity() >= record.oldCount,
                "EditPass::undoDelete: the layer's capacity " + std::to_string(layer.capacity())
                    + " is below the deleted count " + std::to_string(record.oldCount));
    VK_VALIDATE(mask.capacity() >= record.oldCount,
                "EditPass::undoDelete: the mask holds " + std::to_string(mask.capacity())
                    + " Gaussians but the record has " + std::to_string(record.oldCount));
    VK_VALIDATE(static_cast<uint64_t>(record.keptCount) + record.deleted.count()
                    == static_cast<uint64_t>(record.oldCount),
                "EditPass::undoDelete: the record's counts do not add up");
    VK_VALIDATE(record.selectionWords.size() == wordsForCount(record.oldCount),
                "EditPass::undoDelete: the record's selection holds the wrong number of words");
    VK_VALIDATE(countBits(record.selectionWords, record.oldCount) == record.deleted.count(),
                "EditPass::undoDelete: the record's selection does not select its deleted "
                "Gaussians");
    VK_VALIDATE(record.oldCount <= kMaxIndexedGaussians,
                "EditPass::undoDelete: the record holds more Gaussians than a uint32 word index "
                "holds");
    const std::array<EditStream, kStreamCount> streams = layerStreams(layer);
    checkSnapshotStreams(record.deleted, streams, true, "EditPass::undoDelete");

    // The selection the delete was made with: its unselected slots are the kept Gaussians.
    const uint32_t oldCount = record.oldCount;
    mask.upload(record.selectionWords);
    const BufferView flags = grownView(m_device, m_editFlags, oldCount * sizeof(uint32_t));
    const BufferView scan = grownView(m_device, m_editScan, oldCount * sizeof(uint32_t));
    expandMaskToFlags(mask, oldCount, true, flags);
    const uint32_t kept = exclusiveScan(flags, scan, oldCount);
    VK_VALIDATE(kept == record.keptCount,
                "EditPass::undoDelete: the restored selection keeps " + std::to_string(kept)
                    + " Gaussians but the record kept " + std::to_string(record.keptCount));

    // The deleted Gaussians, packed, on the device: one temporary buffer per present stream.
    std::array<std::unique_ptr<Buffer>, kStreamCount> packed;
    for (uint32_t id = 0; id < kStreamCount; ++id)
    {
        if (!streams[id].present())
        {
            continue;
        }
        const HostStream host = snapshotStream(record.deleted, id);
        packed[id] = std::make_unique<Buffer>(m_device, host.bytes);
        packed[id]->upload(host.data, host.bytes);
    }

    // Each present stream: EXPAND the kept elements and the packed deleted ones into scratch,
    // then copy the first oldCount elements back over the stream, in one submission.
    for (uint32_t id = 0; id < kStreamCount; ++id)
    {
        const EditStream & stream = streams[id];
        if (!stream.present())
        {
            continue;
        }
        const size_t bytes = static_cast<size_t>(oldCount) * stream.words * sizeof(uint32_t);
        const BufferView scratch = grownView(m_device, m_editScratch, bytes);
        CompactCall call;
        call.mode = kCompactModeExpand;
        call.count = oldCount;
        call.wordsPerElement = stream.words;
        call.threads = oldCount * stream.words;
        call.flags = flags;
        call.scan = scan;
        call.src = stream.view;
        call.aux = BufferView{packed[id]->handle(), 0, record.deleted.count() * stream.words
                                                           * sizeof(uint32_t),
                              sizeof(uint32_t)};
        call.dst = scratch;
        call.copyTo = stream.view;
        call.copyBytes = bytes;
        runCompact(call);
    }

    layer.setCount(oldCount);
}

DuplicateRecord EditPass::duplicateSelected(GaussianCloud & layer, SelectionMask & mask)
{
    requireOwnedParameters(layer, "EditPass::duplicateSelected");
    VK_VALIDATE(hasGpuParameters(layer),
                "EditPass::duplicateSelected: the layer has no GPU parameters (upload it first)");
    const uint32_t count = static_cast<uint32_t>(layer.count());
    DuplicateRecord record;
    record.oldCount = count;
    const uint32_t k = countSelected(mask, count);
    if (k == 0)
    {
        return record;
    }
    const uint64_t newCount = static_cast<uint64_t>(count) + k;
    VK_VALIDATE(newCount <= kMaxIndexedGaussians,
                "EditPass::duplicateSelected: the layer would hold more Gaussians than a uint32 "
                "word index holds");
    record.selectionWords = mask.download(count);
    record.added = k;

    // Growth: new storage for the streams (the views change) and the mask.
    if (newCount > layer.capacity())
    {
        const size_t capacity = layer.capacity();
        layer.reserve(std::max(static_cast<size_t>(newCount), capacity + capacity / 2));
    }
    mask.resize(layer.capacity());

    // The selected flags and their scan: where each copy lands, after the original count.
    const BufferView flags = grownView(m_device, m_editFlags, count * sizeof(uint32_t));
    const BufferView scan = grownView(m_device, m_editScan, count * sizeof(uint32_t));
    expandMaskToFlags(mask, count, false, flags);
    const uint32_t selected = exclusiveScan(flags, scan, count);
    VK_VALIDATE(selected == k,
                "EditPass::duplicateSelected: the selection's flags disagree with its count");

    // Each present stream: APPEND the selected elements after the original count. Source and
    // destination are the same buffer; the ranges are disjoint.
    const std::array<EditStream, kStreamCount> streams = layerStreams(layer);
    for (uint32_t id = 0; id < kStreamCount; ++id)
    {
        const EditStream & stream = streams[id];
        if (!stream.present())
        {
            continue;
        }
        CompactCall call;
        call.mode = kCompactModeAppend;
        call.count = count;
        call.wordsPerElement = stream.words;
        call.base = count;
        call.threads = count * stream.words;
        call.flags = flags;
        call.scan = scan;
        call.src = stream.view;
        call.dst = stream.view;
        runCompact(call);
    }

    layer.setCount(static_cast<size_t>(newCount));
    // The selection becomes the copies.
    mask.upload(rangeWords(count, static_cast<size_t>(newCount)));
    return record;
}

void EditPass::undoDuplicate(GaussianCloud & layer, SelectionMask & mask,
                             const DuplicateRecord & record)
{
    if (record.added == 0)
    {
        return;
    }
    requireOwnedParameters(layer, "EditPass::undoDuplicate");
    VK_VALIDATE(static_cast<uint64_t>(layer.count())
                    == static_cast<uint64_t>(record.oldCount) + record.added,
                "EditPass::undoDuplicate: the layer holds " + std::to_string(layer.count())
                    + " Gaussians but the duplicate appended to "
                    + std::to_string(record.oldCount));
    VK_VALIDATE(record.selectionWords.size() == wordsForCount(record.oldCount),
                "EditPass::undoDuplicate: the record's selection holds the wrong number of words");
    VK_VALIDATE(mask.capacity() >= record.oldCount,
                "EditPass::undoDuplicate: the mask holds " + std::to_string(mask.capacity())
                    + " Gaussians but the record has " + std::to_string(record.oldCount));
    // The capacity is kept: only the count and the selection go back.
    layer.setCount(record.oldCount);
    mask.upload(record.selectionWords);
}

GpuScan & EditPass::scan()
{
    return *m_scan;
}

} // namespace sabercat
