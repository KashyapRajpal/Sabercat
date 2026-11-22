// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// PSNR and MSE between two sources over their overlap, on the GPU. The values are
// those of the CLI: psnr(displayRgb(A crop), displayRgb(B crop)) (flip_hdr.h, image_compare.h).
#pragma once

#include <cstdint>
#include <memory>
#include <optional>

#include "compare_layout.h"
#include "gpu_context.h"
#include "media_source.h"

namespace sabercat
{
class Buffer;
} // namespace sabercat

namespace sabercat::compare::gui
{

struct MetricsResult
{
    // Mean squared error over RGB, display-space, on the overlap: total / (pixelCount * 3).
    double mse = 0.0;
    // 10 log10(1 / mse); +infinity when mse is 0.
    double psnrDb = 0.0;
    // Pixels of the overlap rectangle.
    uint64_t pixelCount = 0;
    // The overlap rectangle, in canvas pixels.
    PixelRect overlap;
};

class MetricsGpu
{
  public:
    explicit MetricsGpu(GpuContext & context);
    ~MetricsGpu();

    MetricsGpu(const MetricsGpu &) = delete;
    MetricsGpu & operator=(const MetricsGpu &) = delete;

    // Metrics over `overlap`, a rectangle of canvas pixels (see overlapRect() in
    // compare_layout.h). aOffsetX and aOffsetY are the canvas position of source a's
    // top-left pixel, likewise for b. Returns nullopt when the overlap is empty or
    // either source is not Ready.
    // Throws std::invalid_argument when the overlap is not inside both sources, when
    // a source's size does not match its color buffer, or when the overlap has 2^31
    // pixels or more. Throws std::runtime_error on a non-finite sample, as the CLI does.
    std::optional<MetricsResult> compute(const MediaSource & a, int32_t aOffsetX, int32_t aOffsetY,
                                         const MediaSource & b, int32_t bOffsetX, int32_t bOffsetY,
                                         PixelRect overlap);

  private:
    GpuContext & m_context;
    std::unique_ptr<ComputeKernel> m_kernel;
    std::unique_ptr<sabercat::Buffer> m_partials;
    uint64_t m_partialBytes = 0;
};

} // namespace sabercat::compare::gui
