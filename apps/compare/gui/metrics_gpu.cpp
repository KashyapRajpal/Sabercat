// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "metrics_gpu.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "sabercat/buffer.h"
#include "sabercat/device.h"

namespace sabercat::compare::gui
{

namespace
{

// Pixels per workgroup: one per thread, the group size of compare_reduce.slang.
constexpr uint32_t kGroupSize = 256;
// Groups per dispatch row. Larger counts wrap into more rows (the shader rebuilds the group index).
constexpr uint32_t kGroupRowLength = 65535;
// Overlaps this large or larger would overflow the shader's 32-bit pixel index.
constexpr uint64_t kMaxOverlapPixels = 0x7FFFFFFFull;

// Push constants of compare_reduce.slang (MetricsParams): 16 x 4 bytes.
struct GpuMetricsParams
{
    int32_t aOffsetX;
    int32_t aOffsetY;
    int32_t bOffsetX;
    int32_t bOffsetY;
    int32_t overlapX;
    int32_t overlapY;
    uint32_t overlapWidth;
    uint32_t overlapHeight;
    uint32_t aWidth;
    uint32_t aHeight;
    uint32_t bWidth;
    uint32_t bHeight;
    uint32_t aLinear;
    uint32_t bLinear;
    uint32_t groupCount;
    uint32_t groupRowLength;
};
static_assert(sizeof(GpuMetricsParams) == 64, "GpuMetricsParams must match the 64-byte push constant block");

// True when every pixel of `inner` lies in `outer` (both non-empty).
bool rectInside(const PixelRect & inner, int32_t left, int32_t top, uint32_t width, uint32_t height)
{
    const int64_t right = static_cast<int64_t>(left) + width;
    const int64_t bottom = static_cast<int64_t>(top) + height;
    return inner.x >= left && inner.y >= top && static_cast<int64_t>(inner.x) + inner.width <= right
        && static_cast<int64_t>(inner.y) + inner.height <= bottom;
}

} // namespace

MetricsGpu::MetricsGpu(GpuContext & context) : m_context(context)
{
    const std::vector<VkDescriptorType> bindings = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 0: source a
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 1: source b
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 2: partial sums
    };
    m_kernel = context.makeComputePipeline("compare_reduce", bindings, static_cast<uint32_t>(sizeof(GpuMetricsParams)));
}

MetricsGpu::~MetricsGpu() = default;

std::optional<MetricsResult> MetricsGpu::compute(const MediaSource & a, int32_t aOffsetX, int32_t aOffsetY,
                                                 const MediaSource & b, int32_t bOffsetX, int32_t bOffsetY,
                                                 PixelRect overlap)
{
    if (a.status() != SourceStatus::Ready || b.status() != SourceStatus::Ready)
    {
        return std::nullopt;
    }
    if (overlap.width <= 0 || overlap.height <= 0)
    {
        return std::nullopt;
    }

    const uint32_t aWidth = a.width();
    const uint32_t aHeight = a.height();
    const uint32_t bWidth = b.width();
    const uint32_t bHeight = b.height();
    if (!rectInside(overlap, aOffsetX, aOffsetY, aWidth, aHeight)
        || !rectInside(overlap, bOffsetX, bOffsetY, bWidth, bHeight))
    {
        throw std::invalid_argument("metrics overlap is not inside both sources");
    }

    const uint64_t pixelCount = static_cast<uint64_t>(overlap.width) * static_cast<uint64_t>(overlap.height);
    if (pixelCount > kMaxOverlapPixels)
    {
        throw std::invalid_argument("metrics overlap has too many pixels for the GPU reduction");
    }

    const BufferView colorA = a.color();
    const BufferView colorB = b.color();
    const VkDeviceSize sizeA = static_cast<VkDeviceSize>(aWidth) * aHeight * 16;
    const VkDeviceSize sizeB = static_cast<VkDeviceSize>(bWidth) * bHeight * 16;
    if (!colorA.valid() || !colorB.valid() || colorA.size != sizeA || colorB.size != sizeB)
    {
        throw std::invalid_argument("a source has no RGBA32F color buffer of its size");
    }

    const uint32_t groupCount = static_cast<uint32_t>((pixelCount + kGroupSize - 1) / kGroupSize);
    const uint32_t groupsX = groupCount < kGroupRowLength ? groupCount : kGroupRowLength;
    const uint32_t groupsY = (groupCount + kGroupRowLength - 1) / kGroupRowLength;

    const uint64_t partialBytes = static_cast<uint64_t>(groupCount) * sizeof(float);
    if (!m_partials || m_partialBytes < partialBytes)
    {
        m_partials = std::make_unique<Buffer>(m_context.device(), partialBytes);
        m_partialBytes = partialBytes;
    }

    m_kernel->bindBuffer(0, colorA);
    m_kernel->bindBuffer(1, colorB);
    m_kernel->bindBuffer(2, m_partials->view(static_cast<uint32_t>(sizeof(float))));

    GpuMetricsParams params{};
    params.aOffsetX = aOffsetX;
    params.aOffsetY = aOffsetY;
    params.bOffsetX = bOffsetX;
    params.bOffsetY = bOffsetY;
    params.overlapX = overlap.x;
    params.overlapY = overlap.y;
    params.overlapWidth = static_cast<uint32_t>(overlap.width);
    params.overlapHeight = static_cast<uint32_t>(overlap.height);
    params.aWidth = aWidth;
    params.aHeight = aHeight;
    params.bWidth = bWidth;
    params.bHeight = bHeight;
    params.aLinear = a.encoding() == Encoding::Linear ? 1u : 0u;
    params.bLinear = b.encoding() == Encoding::Linear ? 1u : 0u;
    params.groupCount = groupCount;
    params.groupRowLength = kGroupRowLength;

    m_context.submit(
        [&](VkCommandBuffer cmd)
        {
            m_kernel->record(cmd, groupsX, groupsY, 1, &params, static_cast<uint32_t>(sizeof(params)));
        });

    std::vector<float> partials(groupCount);
    m_partials->download(partials.data(), static_cast<size_t>(partialBytes));

    // Host sum in double. The partials are non-negative, so no cancellation.
    double total = 0.0;
    for (const float partial : partials)
    {
        total += static_cast<double>(partial);
    }
    if (!std::isfinite(total))
    {
        throw std::runtime_error("non-finite sample in metrics input");
    }

    MetricsResult result;
    result.mse = total / (static_cast<double>(pixelCount) * 3.0);
    result.psnrDb = result.mse == 0.0 ? std::numeric_limits<double>::infinity() : 10.0 * std::log10(1.0 / result.mse);
    result.pixelCount = pixelCount;
    result.overlap = overlap;
    return result;
}

} // namespace sabercat::compare::gui
