// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vulkan/vulkan.h>

#include <glm/glm.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "sabercat/buffer_view.h"
#include "sabercat/camera.h"
#include "sabercat/descriptor_set.h"
#include "sabercat/image.h"
#include "sabercat/pipeline.h"
#include "sabercat/render_config.h"

namespace sabercat
{

class Device;
class Buffer;
class CommandBuffer;

// Per-Gaussian output of forward_project, mirrored host-side so
// examples/forward_test can interpret a raw download() of the buffer.
// Matches shaders/forward_common.slangh's ProjectedGaussian exactly --
// std430's rule (vec3 aligns to 16, everything else here is 4-byte
// aligned) happens to equal this struct's natural C++ layout, verified by
// the static_assert in forward_pass.cpp.
struct ProjectedGaussian
{
    glm::vec2 center; // pixel coordinates
    float depth;      // positive in front of the camera
    float radius;     // screen-space footprint, in pixels

    glm::vec3 conic; // inverse 2D covariance (a, b, c) for [[a,b],[b,c]]
    float alpha;      // opacity

    glm::vec3 color; // display-referred sRGB RGB (SH clamped to [0,1]), not linear
    uint32_t tileCount; // tiles this Gaussian touches (0 => culled)
};

// Per-tile [first, last) slice of the sorted instance list. Mirrors
// tile_ranges.slang's `RWStructuredBuffer<uint2>`.
struct TileRange
{
    uint32_t first;
    uint32_t last;
};

// Viewer-only dynamic LOD knobs. Mirrors forward_common.slangh's
// LODParams exactly (see its comment for why this is a separate binding,
// not part of CameraUniforms). `enabled == 0` is the default -- a future
// differentiable backward pass must never have this dropping Gaussians, so
// only Renderer/viewer code should ever set `enabled = 1`.
struct LODParams
{
    uint32_t enabled = 0;
    float nearDistance = 0.0f;
    float farDistance = 0.0f;
    float maxDropDistance = 0.0f;
    float maxDropDensity = 0.0f;
    float densityReference = 1.0f;

    // Appearance LOD: same pressure signals, mapped onto SH degree
    // rather than a keep/drop decision. Gated independently of `enabled`.
    uint32_t appearanceEnabled = 0;
    float appearanceMaxDropDistance = 0.0f;
    float appearanceMaxDropDensity = 0.0f;
    // Band fade width, 0-1 of a degree step (see forward_common.slangh):
    // the outgoing top band ramps to zero over that last fraction of the
    // step instead of popping off at the threshold. 0 = hard switches.
    float appearanceFadeWidth = 0.0f;

    // Screen-space size cull threshold in pixels; 0 (the default) disables it.
    // Was an unconditional compile-time 2.0f until it was measured culling
    // >95% of real scenes -- see shaders/config.slangh. Off by default for the
    // same reason every other knob here is: a differentiable backward pass
    // must never have LOD silently dropping gradient contributors.
    float minScreenRadius = 0.0f;
};

// Where run() should stop recording, for verification harnesses that need
// to inspect a buffer a later stage would otherwise overwrite in place --
// the radix sort's ping-pong buffer set A is tile_binning's raw output
// AND (after an even number of passes) the final sorted result, so
// inspecting tile_binning's output requires stopping before the sort runs.
// Default (Rasterize) is the real per-frame path.
enum class ForwardStage
{
    Project,
    PrefixSum,
    TileBinning,
    RadixSort,
    TileRanges,
    Rasterize,
};

// What a run() is FOR. Decides which knobs are allowed to reach the
// GPU -- structurally, not by convention:
//
//   Viewer         -- binds the LODParams the viewer set via setLODParams()
//                     (geometry LOD, appearance LOD, minScreenRadius). Per-
//                     pixel backward state is not written.
//   Differentiable -- binds a separate, all-zero LODParams buffer uploaded
//                     once at construction and never written again, so no
//                     setLODParams() call can drop or degrade a gradient
//                     contributor; and sets SAVE_BACKWARD_STATE so
//                     tile_rasterize stores what BackwardPass needs. Only a
//                     completed Differentiable run is accepted by
//                     BackwardPass.
enum class ForwardMode
{
    Viewer,
    Differentiable,
};

// What the most recent run() rendered, so BackwardPass differentiates
// exactly that image -- same camera, same parameter buffers, same flags --
// rather than trusting the caller to pass matching arguments twice.
struct ForwardRunRecord
{
    bool valid = false;     // a run() happened since construction/resize
    bool completed = false; // reached ForwardStage::Rasterize
    ForwardMode mode = ForwardMode::Viewer;
    CameraUniforms camera{}; // as pushed, i.e. with ForwardPass's own flag bits
    CanonicalGaussianBufferViews views{};
};

// Host orchestration for the forward rasterizer: project -> prefix
// sum -> tile-bin -> radix sort (8 passes, two 32-bit words) -> tile-ranges
// -> rasterize. Owns every pipeline/descriptor-set/intermediate buffer the
// chain needs and the offscreen storage image the last stage writes into.
//
// run() is GPU-driven: ONE submit-and-wait in the common case. Only the GPU
// knows how many tile instances forward_project emitted, so a one-thread
// kernel (instance_dispatch.slang) turns that count into the sort's element/
// group counts and VkDispatchIndirectCommands, and every instance-sized
// stage reads its size from there instead of from the host. The instance-
// sized buffers are host-allocated with a grow-only capacity; a frame whose
// count exceeds it renders nothing on the GPU, and run() then grows the
// buffers and re-records only the instance stages in a second submission --
// once per growth, not per frame. ForwardPass submits on its own command
// pool on Device::submitQueue() rather than via Device::immediateSubmit,
// which is load-time-only, never the per-frame path.
class ForwardPass
{
  public:
    // shaderDir must contain the compiled *.comp.spv for all 8 kernels
    // (see shaders/CMakeLists.txt's sabercat_shaders target). depthLayout is
    // fixed for this pass's lifetime; see depthBuffer().
    ForwardPass(Device & device, uint32_t width, uint32_t height, const std::string & shaderDir,
                DepthLayout depthLayout = DepthLayout::None);
    ~ForwardPass();

    ForwardPass(const ForwardPass &) = delete;
    ForwardPass & operator=(const ForwardPass &) = delete;

    // Recreates the tile-range buffer and offscreen image for a new
    // framebuffer size.
    void resize(uint32_t width, uint32_t height);

    // Runs the chain over `views` using `camera` (imageWidth/Height/tilesX/
    // tilesY/gaussianCount must already match this pass's size and
    // `views.gaussianCount`), stopping after `stopAfter` -- verification
    // harnesses use this to inspect a buffer a later stage would otherwise
    // overwrite in place (the radix sort's ping-pong set A is both
    // tile_binning's raw output and, after a full run, the final sorted
    // result). The real per-frame path always uses the default (Rasterize).
    // The output image transition always starts from UNDEFINED (discarding
    // whatever layout it was in) since rasterize fully overwrites every
    // pixel -- callers never need to restore it to any particular layout
    // between calls.
    //
    // `mode` picks Viewer vs Differentiable; see ForwardMode. The
    // SAVE_BACKWARD_STATE and DEPTH_FLOATS bits of camera.renderFlags are
    // ForwardPass's own and are overwritten from `mode` and depthLayout(),
    // whatever the caller passed.
    void run(const CanonicalGaussianBufferViews & views, const CameraUniforms & camera,
            ForwardStage stopAfter = ForwardStage::Rasterize,
            ForwardMode mode = ForwardMode::Viewer);

    // The most recent run(), for BackwardPass. Invalidated by resize().
    const ForwardRunRecord & lastRun() const
    {
        return m_lastRun;
    }

    // Uploads new LOD knobs. Rare (only changes when the user drags an
    // ImGui slider), so this goes through Buffer::upload's immediateSubmit
    // path rather than being threaded through run()'s per-frame push
    // constants -- immediateSubmit is load-time-only, never
    // the per-frame path, and this call is exactly that: an occasional,
    // off-the-hot-path update, not a per-frame one.
    void setLODParams(const LODParams & params);

    const Image & outputImage() const
    {
        return *m_outputImage;
    }
    // Depth of the frontmost contributing Gaussian per pixel (R32F, 0 where
    // nothing contributes) -- a debug-view channel, not consumed by any
    // later forward-pass stage.
    const Image & depthImage() const
    {
        return *m_depthImage;
    }

    // Downloads the current output/depth image to the host, for
    // verification checks and debug dumps. Self-contained: does its own
    // GENERAL <-> TRANSFER_SRC_OPTIMAL transitions and submit/wait on
    // ForwardPass's own command pool. Only meaningful after a run() that
    // reached ForwardStage::Rasterize.
    std::vector<glm::vec4> downloadOutputImage();
    std::vector<float> downloadDepthImage();

    // For callers that present/blit the output image themselves (Renderer::
    // present()): records the GENERAL -> TRANSFER_SRC_OPTIMAL transition
    // into a caller-supplied, already-recording command buffer.
    void prepareOutputForRead(VkCommandBuffer cmd);

    // Intermediate-buffer accessors for verification. Each stage writes
    // its own buffer without clobbering earlier ones, so all of these stay
    // inspectable after a run() -- no need to stop the chain partway
    // through to check an individual stage.
    const Buffer & projected() const
    {
        return *m_projected;
    }
    // One float per Gaussian, the effective SH degree
    // forward_project chose (fractional under the band fade). Meaningful
    // only for Gaussians the run kept (tileCount > 0); feeds
    // DebugView::SHDegree.
    const Buffer & shDegrees() const
    {
        return *m_shDegrees;
    }
    // Tile instances the most recent run() emitted (forward_project's
    // total, before any capacity check). Current, not a frame late: it is
    // copied to a host-visible buffer at the end of the same submission
    // run() already waits on, so reading it adds no stall.
    uint32_t lastInstanceCount() const
    {
        return m_lastInstanceCount;
    }
    const Buffer & instanceOffsets() const
    {
        return *m_instanceOffsets;
    }
    // Final sorted order after all 8 radix passes (tile-major, depth-minor).
    const Buffer & sortedKeysTile() const;
    const Buffer & sortedKeysDepth() const;
    const Buffer & sortedValues() const;
    const Buffer & tileRanges() const
    {
        return *m_tileRanges;
    }
    // Verification-only: one uint per tile, counting how many batches
    // tile_rasterize actually fetched for it this run() call (see
    // tile_rasterize.slang's group-wide early-out). Not used by any
    // rendering path.
    const Buffer & batchesFetched() const
    {
        return *m_batchesFetched;
    }
    // width*height uint2 (asuint(final T), nContrib), written only
    // by a Differentiable run -- see tile_rasterize.slang's pixelState.
    const Buffer & pixelState() const
    {
        return *m_pixelState;
    }
    // width*height float4, row-major -- the same values as
    // outputImage(), written only by a Differentiable run (see
    // tile_rasterize.slang's outputBuffer). Recreated by resize().
    const Buffer & outputBuffer() const
    {
        return *m_outputBuffer;
    }
    // The layout this pass was built with (see DepthLayout).
    DepthLayout depthLayout() const
    {
        return m_depthLayout;
    }
    // width*height*n floats, row-major, n = floats per pixel of depthLayout()
    // (see tile_rasterize.slang's outputDepthBuffer). Written only by a
    // Differentiable run, and only when the layout is not None; under None it
    // is a 16-byte placeholder that keeps descriptor binding 8 valid. Recreated
    // by resize().
    const Buffer & depthBuffer() const
    {
        return *m_depthBuffer;
    }

  private:
    struct SortBuffers
    {
        std::unique_ptr<Buffer> keysTile;
        std::unique_ptr<Buffer> keysDepth;
        std::unique_ptr<Buffer> values;
    };

    // Shared by downloadOutputImage()/downloadDepthImage(): transitions
    // `image` GENERAL -> TRANSFER_SRC_OPTIMAL, copies `byteSize` bytes into
    // `dst`, submits and waits on ForwardPass's own command pool.
    void downloadImageBytes(Image & image, void * dst, VkDeviceSize byteSize);

    // The destructor's body, also run by the constructor when it throws
    // part-way (a throwing constructor never reaches the destructor).
    void release();
    void createPipelines(const std::string & shaderDir);
    void createFixedBuffers();
    void resizeTileBuffers(uint32_t width, uint32_t height);
    void ensureInstanceCapacity(uint32_t instanceCount);
    void ensureScanCapacity(VkDeviceSize elementCount);
    void rebindDynamicDescriptors();

    // Records `record` between the two Device::recordGlobalBarrier calls of
    // the queue contract, submits it to Device::submitQueue() and waits.
    void submitAndWait(const std::function<void(CommandBuffer &)> & record, const char * what);
    // Everything sized by the instance count: instance_dispatch, tile_binning,
    // the radix sort, tile_ranges, tile_rasterize (up to `stopAfter`). The
    // part run() re-records after a capacity overflow.
    void recordInstanceStages(CommandBuffer & cmd, const CameraUniforms & camera,
                              ForwardStage stopAfter);

    Device & m_device;
    uint32_t m_width;
    uint32_t m_height;
    uint32_t m_tilesX = 0;
    uint32_t m_tilesY = 0;
    uint32_t m_gaussianCount = 0;
    uint32_t m_gaussianCapacity = 0; // m_projected/m_instanceOffsets size, grow-only
    uint32_t m_instanceCapacity = 0;
    uint32_t m_lastInstanceCount = 0;

    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    VkFence m_fence = VK_NULL_HANDLE; // reused by every submitAndWait()

    // -- Fixed-size (per loaded cloud / per framebuffer resize) buffers --
    std::unique_ptr<Buffer> m_projected;         // gaussianCapacity * ProjectedGaussian
    // One InstanceDispatch (forward_common.slangh): forward_project's atomic
    // instance total plus the GPU-computed sort sizes and indirect dispatch
    // commands. Device-local, INDIRECT usage.
    std::unique_ptr<Buffer> m_instanceDispatch;
    std::unique_ptr<Buffer> m_shDegrees;         // gaussianCapacity * float, see shDegrees()
    std::unique_ptr<Buffer> m_instanceOffsets;   // gaussianCapacity * uint
    // prefix_sum's per-block totals, shared by both scans (they never
    // overlap): ceil(max scan length / SCAN_BLOCK) uints, grow-only.
    std::unique_ptr<Buffer> m_scanBlockSums;
    std::unique_ptr<Buffer> m_tileRanges;        // tilesX*tilesY * TileRange
    std::unique_ptr<Buffer> m_batchesFetched;    // tilesX*tilesY * uint, verification-only
    std::unique_ptr<Image> m_outputImage;
    std::unique_ptr<Image> m_depthImage;
    // Tiny, rarely-updated LOD tuning-knob buffer (see
    // setLODParams()). Uploaded once at construction (disabled) and again
    // only when the viewer changes a knob -- never per-frame.
    std::unique_ptr<Buffer> m_lodParams;
    // The LODParams a Differentiable run binds instead -- all zero,
    // uploaded once at construction, never written again. See ForwardMode.
    std::unique_ptr<Buffer> m_lodDisabled;
    std::unique_ptr<Buffer> m_pixelState; // width*height uint2, see pixelState()
    std::unique_ptr<Buffer> m_outputBuffer; // width*height float4, see outputBuffer()
    DepthLayout m_depthLayout = DepthLayout::None;
    std::unique_ptr<Buffer> m_depthBuffer; // width*height*n floats, see depthBuffer()
    ForwardRunRecord m_lastRun;

    // Persistent host-visible readback target for the instance total
    // (created directly via VMA, not through Buffer -- Buffer's allocations
    // are always device-local; see forward_pass.cpp).
    VkBuffer m_instanceCountReadback = VK_NULL_HANDLE;
    VmaAllocation m_instanceCountReadbackAllocation = VK_NULL_HANDLE;
    const uint32_t * m_instanceCountReadbackMapped = nullptr;

    // -- Dynamic (instance-count-sized) buffers, ping-ponged across the 8
    // radix passes. Grown geometrically when a frame overflows them; never
    // shrunk. bufferA is always where tile_binning writes and, because 8
    // (an even count) passes ping-pong A->B->A->..., always where the
    // final sorted result ends up too.
    SortBuffers m_sortA;
    SortBuffers m_sortB;
    std::unique_ptr<Buffer> m_radixCounters; // RADIX_BUCKETS * groupCount(capacity)
    std::unique_ptr<Buffer> m_radixOffsets;  // same size: prefix_sum of m_radixCounters

    // -- Pipelines + descriptor sets, one set per kernel (two variants for
    // the ping-ponged histogram/scatter kernels, selected by pass parity).
    DescriptorSet m_projectDescriptors;
    VkDescriptorSetLayout m_projectLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_projectSet = VK_NULL_HANDLE;
    Pipeline m_projectPipeline;

    DescriptorSet m_dispatchDescriptors;
    VkDescriptorSetLayout m_dispatchLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_dispatchSet = VK_NULL_HANDLE;
    Pipeline m_dispatchPipeline;

    // prefix_sum serves two scans, one descriptor set each.
    DescriptorSet m_prefixSumDescriptors;
    VkDescriptorSetLayout m_prefixSumLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_prefixSumSet = VK_NULL_HANDLE;   // tile counts -> instanceOffsets
    VkDescriptorSet m_counterScanSet = VK_NULL_HANDLE; // radixCounters -> radixOffsets
    Pipeline m_prefixSumPipeline;

    DescriptorSet m_tileBinDescriptors;
    VkDescriptorSetLayout m_tileBinLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_tileBinSet = VK_NULL_HANDLE;
    Pipeline m_tileBinPipeline;

    DescriptorSet m_histogramDescriptors;
    VkDescriptorSetLayout m_histogramLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_histogramSetAtoB = VK_NULL_HANDLE; // reads A, writes counters for A
    VkDescriptorSet m_histogramSetBtoA = VK_NULL_HANDLE;
    Pipeline m_histogramPipeline;

    DescriptorSet m_scatterDescriptors;
    VkDescriptorSetLayout m_scatterLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_scatterSetAtoB = VK_NULL_HANDLE; // reads A, writes B
    VkDescriptorSet m_scatterSetBtoA = VK_NULL_HANDLE; // reads B, writes A
    Pipeline m_scatterPipeline;

    DescriptorSet m_tileRangesDescriptors;
    VkDescriptorSetLayout m_tileRangesLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_tileRangesSet = VK_NULL_HANDLE;
    Pipeline m_tileRangesPipeline;

    DescriptorSet m_rasterizeDescriptors;
    VkDescriptorSetLayout m_rasterizeLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_rasterizeSet = VK_NULL_HANDLE;
    Pipeline m_rasterizePipeline;
};

} // namespace sabercat
