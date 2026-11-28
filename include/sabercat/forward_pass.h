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
#include "sabercat/sort_settings.h"

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
//
// The non-default SortMode values add optional stages to that chain. Stable
// runs a sticky-key update and a per-instance key rewrite before the radix
// sort; TileRayDepth and Hybrid compute the ray-depth form and rewrite the
// keys from it; PixelWindow computes the form and replaces tile_rasterize
// with the per-pixel window rasterizer (Hybrid too, with K = 8). Stochastic
// computes the form too, skips the depth-word radix passes (the order is
// irrelevant to its coverage), rasterizes S jittered samples per pixel and
// blends the result into a per-pixel history on the GPU. Each stage is built on
// first use, and the default chain records none of them.
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

    // Sort order for Viewer runs. Validates first (throws
    // std::invalid_argument naming the field) and stores; no GPU work. Changing
    // the Stochastic fields (samples, temporalFrames, temporalMode,
    // temporalEmaWeight) or entering/leaving Stochastic restarts its temporal
    // state, like resetStochastic(). Differentiable runs ignore the mode and use
    // Default.
    void setSortSettings(const SortSettings & settings);
    const SortSettings & sortSettings() const
    {
        return m_sort;
    }

    // Opt-in GPU stage timing. Off by default, and while off run() records no
    // extra commands. The first enable creates a timestamp query pool; a queue
    // without timestamp support keeps lastStageTimings() invalid.
    void setStageTimingEnabled(bool enabled);
    bool stageTimingEnabled() const
    {
        return m_timingEnabled;
    }
    // Stage times of the most recent run(). Valid only when that run was
    // stamped: timing on, stopAfter == Rasterize, and no capacity re-run.
    const ForwardStageTimings & lastStageTimings() const
    {
        return m_lastTimings;
    }

    // Verification and study use. Computes the whitened ray-depth form (12
    // floats per Gaussian, 48 bytes; layout in shaders/sort_common.slangh) for views.gaussianCount
    // Gaussians in a standalone submission: its own vkResetCommandPool and
    // submitAndWait, no timestamps, and nothing run() recorded is touched.
    // camera.gaussianCount must equal views.gaussianCount (VK_VALIDATE). The
    // kernel is built on first use.
    void computeRayDepthForm(const CanonicalGaussianBufferViews & views,
                             const CameraUniforms & camera);
    // The form computed by the last computeRayDepthForm(), or by the last run()
    // in TileRayDepth, PixelWindow, Hybrid or Stochastic mode (which computes it
    // for its camera every frame). Throws std::runtime_error if neither ever ran.
    const Buffer & rayDepthForm() const;

    // Stable mode's sticky bucket keys: one uint per Gaussian, the key each
    // Gaussian was last sorted with. Persists across runs. Throws
    // std::runtime_error if no Stable run has happened.
    const Buffer & stableKeys() const;
    // Starts the sticky keys over: the next Stable run refills them with
    // 0xFFFFFFFF (the "no key yet" sentinel) before its first dispatch.
    void resetStableSort();
    // Stochastic mode's temporal state. resetStochastic() restarts the noise
    // sequence (frame 0) and the accumulation: the next Stochastic run is a raw
    // frame. The last* accessors describe the most recent completed Stochastic
    // run; lastStochasticAccumulated() is the count after that run, and reads 1
    // right after a reset.
    void resetStochastic();
    uint32_t lastStochasticFrame() const
    {
        return m_stochLastFrame;
    }
    uint32_t lastStochasticAccumulated() const
    {
        return m_stochAccumulated;
    }
    float lastStochasticWeight() const
    {
        return m_stochLastWeight;
    }
    // Bytes of the mode-specific buffers (sticky keys, ray-depth form, and the
    // Stochastic noise, params and history). They are grow-only or sized with the
    // framebuffer, so this is their allocated size, not the live count.
    uint64_t extraBufferBytes() const;
    // Selection highlight. While set, every Viewer run() moves the color of each
    // Gaussian whose bit is set in maskWords toward tint[0..2] (display-referred
    // sRGB) by tint[3], right after projection (shaders/highlight_tint.slang).
    // maskWords is one bit per Gaussian, word w covering Gaussians [32w, 32w + 32),
    // and must hold at least ceil(gaussianCount / 32) words at each run(); it is
    // read at every run, so a view that moves (a reallocated buffer) is fine as long
    // as setHighlight() is called again with the new view. Differentiable runs never
    // tint. Throws on an invalid view. The tint is recorded only by the runs that
    // use it, so with no highlight set the run's commands are exactly the default.
    void setHighlight(const BufferView & maskWords, const float tint[4]);
    // Turns the highlight off; the next run() records no tint.
    void clearHighlight();

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

    // A compute stage built on first use instead of in createPipelines(), so
    // the default path never creates it. Each stage owns its descriptor
    // set, like the fixed stages below. Members of this type are built by
    // ensureOptionalStage() and torn down by destroyOptionalStage() in release().
    struct OptionalStage
    {
        explicit OptionalStage(VkDevice device) : descriptors(device) {}

        DescriptorSet descriptors;
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
        Pipeline pipeline;
        bool built = false;
    };

    // Builds `stage` from shaderDir's `kernelName`.comp.spv the first time it
    // is called, with one descriptor set of `bindings` and `pushConstantSize`
    // bytes of push constants. Does nothing once built.
    OptionalStage & ensureOptionalStage(OptionalStage & stage, const char * kernelName,
                                        const std::vector<VkDescriptorSetLayoutBinding> & bindings,
                                        uint32_t pushConstantSize);
    // Destroys the pipeline of a stage that ensureOptionalStage() built.
    void destroyOptionalStage(OptionalStage & stage);
    // Checks that `mode` is a SortMode value (an internal guard: the settings
    // were validated when they were set). Every mode can run.
    void requireSortModeAvailable(SortMode mode) const;
    // Builds the stages the run's mode needs and grows the buffers they use
    // (sticky keys, ray-depth form, the Stochastic noise and params). Called by
    // run() before anything is recorded; does nothing in Default mode.
    void ensureSortModeResources(uint32_t gaussianCount);
    // Binds every descriptor set the run's mode uses to the current buffers.
    // Descriptor writes must happen outside command recording, so run() calls
    // this before its submission and again after a capacity growth.
    void bindSortModeDescriptors();
    // Grows m_rayForm to gaussianCount * 12 floats and builds m_rayFormStage.
    void ensureRayForm(uint32_t gaussianCount);
    // Binds the ray-form stage's descriptors to `views` and m_rayForm. Outside
    // command recording, like bindSortModeDescriptors().
    void bindRayFormDescriptors(const CanonicalGaussianBufferViews & views);
    // Records the ray-form dispatch for `camera` (groups from
    // camera.gaussianCount). Its descriptors must already be bound.
    void recordRayFormDispatch(CommandBuffer & cmd, const CameraUniforms & camera);
    // Writes timestamp `index` into the stage-timing pool, but only when the
    // current run is stamping (m_stampThisRun). Otherwise does nothing.
    void stamp(VkCommandBuffer cmd, uint32_t index);
    // Converts the timestamps of the run that just finished into m_lastTimings.
    void readStageTimings();

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
    // Builds the highlight kernel and its descriptor set on the first tinted run.
    void ensureHighlightPipeline();

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
    // The views of the run in progress. The ray-form stage binds them, and
    // the chain's form dispatch re-records from them after a capacity re-run.
    CanonicalGaussianBufferViews m_runViews{};

    // -- Sort settings and stage timing. Grouped by size (pointers, then
    // 4-byte fields, then bools) so the layout carries no interior padding. --
    std::string m_shaderDir; // kept so OptionalStage builds can load kernels
    VkQueryPool m_timingPool = VK_NULL_HANDLE; // kTimestampCount stamps, created on first enable
    SortSettings m_sort;
    ForwardStageTimings m_lastTimings;
    uint32_t m_timestampValidBits = 0; // of the submit queue; 0 = no timestamps
    SortMode m_runSortMode = SortMode::Default; // effective mode of the run in progress
    uint32_t m_runWindow = 0;     // window size K of the run in progress (8, 16 or 32)
    uint32_t m_stickyGaussianCount = 0; // gaussianCount the sticky keys were built for
    bool m_timingEnabled = false;
    bool m_stampThisRun = false; // this run() writes timestamps (see run())

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

    // -- Ray-depth form (computeRayDepthForm, and the chain in TileRayDepth,
    // PixelWindow and Hybrid). Built on first use, so the default path never
    // creates it. m_rayForm is grow-only. --
    OptionalStage m_rayFormStage;
    std::unique_ptr<Buffer> m_rayForm; // gaussianCount * 12 floats, see rayDepthForm()
    bool m_rayFormComputed = false;

    // -- Sort-variant stages, built on first use by ensureSortModeResources().
    // m_windowStage is indexed by windowStageIndex() (K = 8, 16, 32). --
    OptionalStage m_stableUpdateStage;
    OptionalStage m_stableKeysStage;
    OptionalStage m_tileRayKeysStage;
    OptionalStage m_windowStage[3];
    // Stable's sticky bucket keys, one uint per Gaussian (4 B each). Persists
    // across runs and is grow-only. Reset (refilled with the sentinel at the
    // start of the next Stable submission) when m_stickyReset is set. Set on
    // creation, resetStableSort(), resize(), a gaussianCount change, and a
    // change of the Stable parameters or of entering/leaving Stable.
    std::unique_ptr<Buffer> m_sticky;
    bool m_stickyReset = false;

    // -- Stochastic transparency. The stages and the noise and params buffers are
    // built on the first Stochastic run. The history is sized to the framebuffer:
    // resize() drops it, and the next Stochastic run recreates it. --
    OptionalStage m_stochRasterStage[2]; // tile_rasterize_stochastic_s4 / _s8 (index S == 8)
    OptionalStage m_stochResolveStage;   // stochastic_resolve
    std::unique_ptr<Buffer> m_stochNoise;   // 4096 uints: blue-noise ranks, uploaded once
    std::unique_ptr<Buffer> m_stochParams;  // 16 B: the frame offset of the current run
    std::unique_ptr<Buffer> m_stochHistory; // width*height float4; null until a Stochastic run
    // The camera of the last completed run, for the "camera moved" test.
    CameraUniforms m_stochPrevCamera{};
    // The noise frame and accumulation of the run in progress, computed before
    // anything is recorded and committed only when its submission succeeds.
    struct StochasticStep
    {
        uint32_t frame;
        uint32_t accumulated;
        float weight;
    };
    StochasticStep m_runStochastic{0, 1, 1.0f};
    uint32_t m_stochFrame = 0;         // noise frame index of the next run
    uint32_t m_stochAccumulated = 1;   // frames blended into the history
    uint32_t m_stochGaussianCount = 0; // gaussianCount of the last completed run
    uint32_t m_stochLastFrame = 0;     // frame index the last completed run used
    float m_stochLastWeight = 1.0f;    // history weight the last completed run used
    bool m_stochPrevValid = false;     // m_stochPrevCamera holds a completed run's camera
    bool m_stochReset = true;          // the next Stochastic run starts over
    // -- Selection highlight (see setHighlight). The kernel is built lazily, on the
    // first tinted run, so a ForwardPass that never highlights never loads it. --
    DescriptorSet m_highlightDescriptors;
    VkDescriptorSetLayout m_highlightLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_highlightSet = VK_NULL_HANDLE; // bindings: 0 projected, 1 mask
    Pipeline m_highlightPipeline;
    BufferView m_highlightMask;
    float m_highlightTint[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    bool m_highlightEnabled = false;
};

} // namespace sabercat
