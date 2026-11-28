// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <vector>
#include "compare_layout.h"   // PixelRect
#include "flip_hdr.h"         // ToneMapper
#include "gpu_context.h"
#include "media_source.h"
#include "sabercat/buffer_view.h"
namespace sabercat { class Buffer; }
namespace sabercat::compare::gui
{
// User settings of an HDR-FLIP run. Unset start/stop are derived from A (the reference) over the
// overlap with FLIP's computeExposures(); an unset count is max(2, ceil(stop - start)).
struct FlipHdrSettings
{
    ToneMapper toneMapper = ToneMapper::Aces;
    std::optional<float> start;
    std::optional<float> stop;
    std::optional<int32_t> count;
};

// What an HDR-FLIP result adds to FlipGpuResult.
struct FlipHdrInfo
{
    std::vector<float> exposureMap;   // overlap.width * overlap.height floats: i/(count-1) at the exposure with the largest error; 0 where none exceeded 0
    double linearMse = std::numeric_limits<double>::quiet_NaN();  // over the overlap, linear values; NaN = no finite pair
    float start = 0.0f;               // the exposure range and count that were used
    float stop = 0.0f;
    int32_t count = 0;
    ToneMapper toneMapper = ToneMapper::Aces;
};

struct FlipGpuResult
{
    double mean = 0.0;              // FLIP mean over the overlap, in [0,1]
    uint64_t nonFiniteSamples = 0;  // NaN/inf RGB samples of A and B in the overlap (sanitized)
    PixelRect overlap;              // canvas pixels; errorMap covers exactly this rectangle
    float ppd = 0.0f;
    std::vector<float> errorMap;    // overlap.width * overlap.height floats, row-major, top row first (host copy)
    std::optional<FlipHdrInfo> hdr; // set for an HDR-FLIP result (errorMap is then the max over the exposures)
};

class FlipGpu
{
  public:
    explicit FlipGpu(GpuContext & context);
    ~FlipGpu();
    FlipGpu(const FlipGpu &) = delete;
    FlipGpu & operator=(const FlipGpu &) = delete;

    // compute(): unchanged (LDR). It also cancels a running HDR job.
    // Order of checks: a or b not Ready -> nullopt; ppd outside [kMinFlipPpd, kMaxFlipPpd] or NaN ->
    // std::invalid_argument; overlap empty -> nullopt; overlap not inside both sources, color buffer not
    // w*h*16 bytes, or more than kMaxFlipPixels overlap pixels -> std::invalid_argument; not enough free
    // device-local memory (see requiredBytes, uses sceneBudgetOk() from scene_source.h) -> std::runtime_error
    // whose text starts "FLIP needs about <N> MiB of GPU memory" ; mean not finite -> std::runtime_error.
    // Any throw leaves no valid map (errorMapView() invalid). Synchronous.
    std::optional<FlipGpuResult> compute(const MediaSource & a, int32_t aOffsetX, int32_t aOffsetY,
                                         const MediaSource & b, int32_t bOffsetX, int32_t bOffsetY,
                                         PixelRect overlap, float ppd);

    // ---- HDR-FLIP as a job: beginHdr, stepHdr until true, finishHdr. computeHdr does all three.
    // beginHdr: any earlier job is cancelled and the previous map is invalid. Checks, in order:
    // a or b not Ready -> nullopt (no job); ppd outside [kMinFlipPpd, kMaxFlipPpd] or NaN ->
    // std::invalid_argument; overlap empty -> nullopt; overlap not inside both sources, wrong color buffer
    // size, more than kMaxFlipPixels pixels, or too many tiles per axis -> std::invalid_argument (exactly as
    // compute()); the exposure range (hdrExposuresFromLinear with blackReferenceUsesEpsilon =
    // true, from A's overlap crop, see Decision 2/3/4) -> std::invalid_argument on a bad range;
    // device budget (requiredBytesHdr, sceneBudgetOk) -> std::runtime_error "FLIP needs about N MiB of
    // GPU memory, more than is free". Any throw or nullopt leaves no job and no valid map.
    // On success returns true and a job is active. The sources must stay alive and unchanged until
    // finishHdr() or cancelHdr().
    bool beginHdr(const MediaSource & a, int32_t aOffsetX, int32_t aOffsetY,
                  const MediaSource & b, int32_t bOffsetX, int32_t bOffsetY,
                  PixelRect overlap, float ppd, const FlipHdrSettings & settings);
    bool hdrActive() const;            // a job is active (begun, not finished or cancelled)
    uint32_t hdrDone() const;          // exposures finished so far
    uint32_t hdrCount() const;         // exposures of the active job; 0 when none
    // Runs up to maxExposures (at least 1) more exposures in ONE synchronous submit. Returns true when
    // all are done. std::logic_error when no job is active.
    bool stepHdr(uint32_t maxExposures);
    // Requires all exposures done (std::logic_error otherwise). Downloads the maps, mean (finite else
    // std::runtime_error), ends the job. errorMapView() then shows the HDR max map.
    FlipGpuResult finishHdr();
    // Ends the job without a result; errorMapView() stays invalid. Safe with no job.
    void cancelHdr();
    // beginHdr + stepHdr(all) + finishHdr. nullopt exactly when beginHdr returned false.
    std::optional<FlipGpuResult> computeHdr(const MediaSource & a, int32_t aOffsetX, int32_t aOffsetY,
                                            const MediaSource & b, int32_t bOffsetX, int32_t bOffsetY,
                                            PixelRect overlap, float ppd, const FlipHdrSettings & settings);

    // The device error map of the last successful compute() / finishHdr() (the HDR max map after an HDR run).
    BufferView errorMapView() const;
    void release();   // also ends a job
    static uint64_t requiredBytes(uint64_t pixels, float ppd);      // LDR, as today (+ tone table entries)
    static uint64_t requiredBytesHdr(uint64_t pixels, float ppd);   // requiredBytes + 8 * pixels
    static constexpr uint64_t kMaxFlipPixels = 0x7FFFFFFFull / 16ull;

  private:  // implementer's choice
    struct HdrJob;   // the active HDR-FLIP run; defined in flip_gpu.cpp

    // Binds the buffers of every pass to the current scratch. hdr selects the HDR prepare and
    // also binds the accumulate pass (which needs the HDR maps).
    void bindPasses(const BufferView & colorA, const BufferView & colorB, bool hdr);
    // Allocates what a run needs for `pixels` overlap pixels (grow-only). hdr also ensures the HDR maps.
    void ensureBuffers(uint64_t pixels, float ppd, const FlipKernels & kernels, bool hdr);
    // Bytes of every buffer this object holds right now.
    uint64_t heldBytes() const;
    // Frees the per-pixel scratch and the maps (used when the overlap grows).
    void releaseScratch();

    GpuContext & m_context;
    std::unique_ptr<ComputeKernel> m_prepare;        // flip_prepare: display -> YCxCz, non-finite count
    std::unique_ptr<ComputeKernel> m_prepareHdr;     // flip_prepare_hdr: exposure-scaled, tone-mapped YCxCz
    std::unique_ptr<ComputeKernel> m_spatialH;       // flip_spatial_h: horizontal spatial filter
    std::unique_ptr<ComputeKernel> m_spatialVColor;  // flip_spatial_v_color: vertical filter, color difference
    std::unique_ptr<ComputeKernel> m_featureH;       // flip_feature_h: horizontal feature filters
    std::unique_ptr<ComputeKernel> m_featureVFinal;  // flip_feature_v_final: vertical feature filters, final error
    std::unique_ptr<ComputeKernel> m_hdrAccumulate;  // flip_hdr_accumulate: max over the exposures
    std::unique_ptr<sabercat::Buffer> m_srgbTable;   // 65536 floats, uploaded once
    std::unique_ptr<sabercat::Buffer> m_kernelBuffer; // float4 kernel table for m_kernelPpd
    float m_kernelPpd = 0.0f;                        // 0 = no kernel table built
    std::unique_ptr<sabercat::Buffer> m_diag;        // uint[4]: diag[0] = non-finite sample count
    std::unique_ptr<sabercat::Buffer> m_yA;          // float4 per pixel, grow-only
    std::unique_ptr<sabercat::Buffer> m_yB;
    std::unique_ptr<sabercat::Buffer> m_interA;
    std::unique_ptr<sabercat::Buffer> m_interB;
    std::unique_ptr<sabercat::Buffer> m_map;         // float per pixel, grow-only; the LDR map, or one HDR exposure's error
    std::unique_ptr<sabercat::Buffer> m_hdrMax;      // float per pixel: HDR max over the exposures (the HDR map)
    std::unique_ptr<sabercat::Buffer> m_hdrExposure; // float per pixel: i/(count-1) of the largest error
    std::unique_ptr<HdrJob> m_job;                   // the active HDR-FLIP run, or null
    uint64_t m_capacityPixels = 0;                   // pixels the scratch and map buffers hold
    uint64_t m_mapPixels = 0;                        // pixels of the last successful compute
    bool m_mapValid = false;
    bool m_mapIsHdr = false;                         // the valid map is m_hdrMax (an HDR run), else m_map
};
}
