// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "flip_gpu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "flip_hdr.h"
#include "image_compare.h"
#include "scene_source.h"
#include "sabercat/buffer.h"
#include "sabercat/device.h"

namespace sabercat::compare::gui
{

namespace
{

// Threads per workgroup side; the kernels use [numthreads(16, 16, 1)].
constexpr uint32_t kTile = 16;
// Vulkan's minimum for each dispatch axis. Larger overlaps are refused (see checkOverlap()).
constexpr uint64_t kMaxGroupsPerAxis = 65535;
// Device-local bytes per overlap pixel: error map 4, yA and yB 16 each, interA and interB 16 each.
constexpr uint64_t kBytesPerPixel = 68;
// Extra bytes per overlap pixel of an HDR run: the max map and the exposure map, 4 each.
constexpr uint64_t kHdrBytesPerPixel = 8;
// The sRGB table: 65,536 floats.
constexpr uint64_t kSrgbEntries = 65536;
constexpr uint64_t kSrgbBytes = kSrgbEntries * sizeof(float);
// The diag buffer: four uints, of which diag[0] is the non-finite count.
constexpr uint64_t kDiagBytes = 4 * sizeof(uint32_t);
// Entries of the kernel table that come before the spatial taps (see buildKernelTable()).
constexpr size_t kKernelHeaderEntries = 10;
// Entries of the kernel table after the feature taps: two per tone mapper (see buildKernelTable()).
constexpr size_t kToneEntries = 6;
constexpr uint64_t kMiB = 1024ull * 1024ull;

// hdrFlags bits read by the HDR shaders (flip_common.slangh).
constexpr uint32_t kHdrFlagCountNonFinite = 1u;
constexpr uint32_t kHdrFlagFirstExposure = 2u;

// Push constants of the FLIP kernels: FlipParams in flip_common.slangh (80 bytes). The LDR passes
// leave the HDR fields at zero; the shaders ignore them.
struct FlipGpuParams
{
    int32_t aOffsetX;
    int32_t aOffsetY;
    int32_t bOffsetX;
    int32_t bOffsetY;
    int32_t overlapX;
    int32_t overlapY;
    uint32_t width;
    uint32_t height;
    uint32_t aWidth;
    uint32_t bWidth;
    uint32_t aLinear;
    uint32_t bLinear;
    uint32_t spatialRadius;
    uint32_t featureRadius;
    uint32_t featureBase;
    uint32_t hdrFlags;
    float exposureScale;
    float exposureFraction;
    uint32_t toneMapper;
    uint32_t hdrPad;
};
static_assert(sizeof(FlipGpuParams) == 80, "FlipGpuParams must match the 80-byte FlipParams push constant");

// One float4 entry of the kernel table.
using KernelEntry = std::array<float, 4>;
static_assert(sizeof(KernelEntry) == 16, "kernel entries are float4");

// Descriptor counts of the pipelines: flip_prepare (and flip_prepare_hdr), flip_spatial_h,
// flip_spatial_v_color, flip_feature_h, flip_feature_v_final, flip_hdr_accumulate.
constexpr uint32_t kPrepareBindings = 7;
constexpr uint32_t kSpatialHBindings = 5;
constexpr uint32_t kSpatialVBindings = 4;
constexpr uint32_t kFeatureHBindings = 5;
constexpr uint32_t kFeatureVBindings = 4;
constexpr uint32_t kHdrAccumulateBindings = 4;

std::vector<VkDescriptorType> storageBindings(uint32_t count)
{
    return std::vector<VkDescriptorType>(count, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
}

// True when every pixel of `inner` lies in the rectangle (left, top, width, height).
bool rectInside(const PixelRect & inner, int32_t left, int32_t top, uint32_t width, uint32_t height)
{
    const int64_t right = static_cast<int64_t>(left) + width;
    const int64_t bottom = static_cast<int64_t>(top) + height;
    return inner.x >= left && inner.y >= top && static_cast<int64_t>(inner.x) + inner.width <= right
        && static_cast<int64_t>(inner.y) + inner.height <= bottom;
}

// Dispatch groups along one axis of an overlap extent.
uint64_t tilesFor(int32_t extent)
{
    return (static_cast<uint64_t>(extent) + kTile - 1) / kTile;
}

// Mean of the error values, in the same order as meanOf() in flip_metric.cpp: four double
// accumulators, fixed order, so the result is the same on every run.
double meanOf(const float * errors, size_t pixels)
{
    std::array<double, 4> lane{};
    size_t i = 0;
    for (; i + 4 <= pixels; i += 4)
    {
        lane[0] += errors[i];
        lane[1] += errors[i + 1];
        lane[2] += errors[i + 2];
        lane[3] += errors[i + 3];
    }
    for (; i < pixels; ++i)
    {
        lane[0] += errors[i];
    }
    return (lane[0] + lane[1] + lane[2] + lane[3]) / static_cast<double>(pixels);
}

// Entries of the kernel table for these filters: the header, the spatial taps, the feature taps and
// the tone mappers.
uint64_t kernelEntryCount(const FlipKernels & kernels)
{
    return kKernelHeaderEntries + kernels.spatialYCx.size() + kernels.feature.size() + kToneEntries;
}

// Device-local bytes compute() needs for `pixels` overlap pixels with these kernels.
uint64_t requiredBytesFor(uint64_t pixels, const FlipKernels & kernels)
{
    return kBytesPerPixel * pixels + kSrgbBytes + kernelEntryCount(kernels) * sizeof(KernelEntry) + kDiagBytes;
}

// The kernel table read by the shaders (kernel[] in flip_common.slangh):
//   0: (cmax, pccmax, gpt, gqc)        1: (gqf, huntFactor, 0, 0)
//   2..4: linearRgbToXyz rows          5..7: xyzToLinearRgb rows
//   8: referenceIlluminant             9: invReferenceIlluminant
//   10 .. 10+sw-1: (Y weight, Cx weight, Cz weight 1, Cz weight 2) per spatial tap
//   featureBase .. +fw-1: (gaussian, first derivative, second derivative, 0) per feature tap
//   toneBase .. +5: two entries per tone mapper, at 2 * flipToneMapperIndex(): the first holds
//     tc[0..2] and the second tc[3..5] of toneMapperCoefficients() (toneBase = featureBase + fw).
// featureBase is 10 + sw.
std::vector<KernelEntry> buildKernelTable(const FlipKernels & kernels)
{
    const size_t spatialWidth = kernels.spatialYCx.size();
    const size_t featureWidth = kernels.feature.size();
    const size_t featureBase = kKernelHeaderEntries + spatialWidth;
    const size_t toneBase = featureBase + featureWidth;
    std::vector<KernelEntry> table(toneBase + kToneEntries);

    table[0] = { kernels.cmax, kernels.pccmax, kernels.gpt, kernels.gqc };
    table[1] = { kernels.gqf, kernels.huntFactor, 0.0f, 0.0f };
    const auto & toXyz = kernels.linearRgbToXyz;
    const auto & toRgb = kernels.xyzToLinearRgb;
    for (size_t row = 0; row < 3; ++row)
    {
        table[2 + row] = { toXyz[row * 3 + 0], toXyz[row * 3 + 1], toXyz[row * 3 + 2], 0.0f };
        table[5 + row] = { toRgb[row * 3 + 0], toRgb[row * 3 + 1], toRgb[row * 3 + 2], 0.0f };
    }
    table[8] = { kernels.referenceIlluminant[0], kernels.referenceIlluminant[1], kernels.referenceIlluminant[2], 0.0f };
    table[9] = { kernels.invReferenceIlluminant[0], kernels.invReferenceIlluminant[1],
                 kernels.invReferenceIlluminant[2], 0.0f };
    for (size_t i = 0; i < spatialWidth; ++i)
    {
        const auto & yCx = kernels.spatialYCx[i];
        const auto & cz = kernels.spatialCz[i];
        table[kKernelHeaderEntries + i] = { yCx[0], yCx[1], cz[0], cz[1] };
    }
    for (size_t i = 0; i < featureWidth; ++i)
    {
        const auto & taps = kernels.feature[i];
        table[featureBase + i] = { taps[0], taps[1], taps[2], 0.0f };
    }
    // The shader indexes the tone mappers by FLIP's numbering, so each mapper goes to its own slot.
    for (const ToneMapper mapper : { ToneMapper::Reinhard, ToneMapper::Aces, ToneMapper::Hable })
    {
        const std::array<float, 6> tc = toneMapperCoefficients(mapper);
        const size_t slot = toneBase + 2 * static_cast<size_t>(flipToneMapperIndex(mapper));
        table[slot] = { tc[0], tc[1], tc[2], 0.0f };
        table[slot + 1] = { tc[3], tc[4], tc[5], 0.0f };
    }
    return table;
}

// Throws the budget error when `needed` bytes, less what is already held, do not fit in the free
// device-local memory (sceneBudgetOk() honors the test override).
void requireBudget(sabercat::Device & device, uint64_t needed, uint64_t held)
{
    const uint64_t extra = needed > held ? needed - held : 0;
    std::string budgetMessage;
    if (!sceneBudgetOk(device, extra, budgetMessage))
    {
        throw std::runtime_error("FLIP needs about " + std::to_string((extra + kMiB - 1) / kMiB)
                                 + " MiB of GPU memory, more than is free");
    }
}

// The checks shared by compute() and beginHdr(), in the order compute() documents. The overlap must
// not be empty (the caller returns first). Sets the color views of both sources and returns the
// overlap's pixel count.
uint64_t checkOverlap(const MediaSource & a, int32_t aOffsetX, int32_t aOffsetY,
                      const MediaSource & b, int32_t bOffsetX, int32_t bOffsetY,
                      const PixelRect & overlap, BufferView & colorA, BufferView & colorB)
{
    const uint32_t aWidth = a.width();
    const uint32_t aHeight = a.height();
    const uint32_t bWidth = b.width();
    const uint32_t bHeight = b.height();
    if (!rectInside(overlap, aOffsetX, aOffsetY, aWidth, aHeight)
        || !rectInside(overlap, bOffsetX, bOffsetY, bWidth, bHeight))
    {
        throw std::invalid_argument("FLIP overlap is not inside both sources");
    }

    const uint64_t pixelCount = static_cast<uint64_t>(overlap.width) * static_cast<uint64_t>(overlap.height);
    if (pixelCount > FlipGpu::kMaxFlipPixels)
    {
        throw std::invalid_argument("FLIP overlap has more pixels than the GPU path supports");
    }

    colorA = a.color();
    colorB = b.color();
    const VkDeviceSize sizeA = static_cast<VkDeviceSize>(aWidth) * aHeight * 16;
    const VkDeviceSize sizeB = static_cast<VkDeviceSize>(bWidth) * bHeight * 16;
    if (!colorA.valid() || !colorB.valid() || colorA.size != sizeA || colorB.size != sizeB)
    {
        throw std::invalid_argument("a source has no RGBA32F color buffer of its size");
    }

    // One dispatch covers the overlap in 16 x 16 tiles, so each axis must fit the dispatch limit.
    if (tilesFor(overlap.width) > kMaxGroupsPerAxis || tilesFor(overlap.height) > kMaxGroupsPerAxis)
    {
        throw std::invalid_argument("FLIP overlap is too wide or too tall for one GPU dispatch");
    }
    return pixelCount;
}

// The push constants that do not depend on the HDR exposure, for the overlap and the two sources.
FlipGpuParams makeParams(const MediaSource & a, int32_t aOffsetX, int32_t aOffsetY,
                         const MediaSource & b, int32_t bOffsetX, int32_t bOffsetY,
                         const PixelRect & overlap, const FlipKernels & kernels)
{
    FlipGpuParams params{};
    params.aOffsetX = aOffsetX;
    params.aOffsetY = aOffsetY;
    params.bOffsetX = bOffsetX;
    params.bOffsetY = bOffsetY;
    params.overlapX = overlap.x;
    params.overlapY = overlap.y;
    params.width = static_cast<uint32_t>(overlap.width);
    params.height = static_cast<uint32_t>(overlap.height);
    params.aWidth = a.width();
    params.bWidth = b.width();
    params.aLinear = a.encoding() == Encoding::Linear ? 1u : 0u;
    params.bLinear = b.encoding() == Encoding::Linear ? 1u : 0u;
    params.spatialRadius = static_cast<uint32_t>(kernels.spatialRadius);
    params.featureRadius = static_cast<uint32_t>(kernels.featureRadius);
    params.featureBase = static_cast<uint32_t>(kKernelHeaderEntries + kernels.spatialYCx.size());
    return params;
}

// The linear value the GPU uses for one RGB sample (Decision 2 and 3). Linear samples are taken as
// stored. Display samples are clamped to [0,1] and decoded with the sRGB table. A non-finite value is 0
// for both kinds, as in the shader.
float linearSample(float value, bool linear, const float * table)
{
    if (!std::isfinite(value))
    {
        return 0.0f;
    }
    if (linear)
    {
        return value;
    }
    // Written so that NaN fails the comparison and is clamped to 0.
    const float clamped = value > 0.0f ? std::min(value, 1.0f) : 0.0f;
    return table[static_cast<size_t>(clamped * 65535.0f + 0.5f)];
}

// What beginHdr() derives on the host: the exposure range and count from A, and linearMse.
struct HdrPlan
{
    HdrExposures exposures;
    double linearMse = std::numeric_limits<double>::quiet_NaN();
};

// The host side of beginHdr(), in one pass over the overlap. The linear values of A (for the
// exposure range) are a temporary that is freed on return, before any device memory is allocated.
HdrPlan planHdrOnHost(const MediaSource & a, int32_t aOffsetX, int32_t aOffsetY,
                      const MediaSource & b, int32_t bOffsetX, int32_t bOffsetY,
                      const PixelRect & overlap, float ppd, const FlipHdrSettings & settings)
{
    const ImageData * hostA = a.hostData();
    const ImageData * hostB = b.hostData();
    if (hostA == nullptr || hostB == nullptr)
    {
        throw std::invalid_argument("a source has no host copy");
    }
    const size_t aWidth = a.width();
    const size_t aHeight = a.height();
    const size_t bWidth = b.width();
    const size_t bHeight = b.height();
    if (hostA->width != aWidth || hostA->height != aHeight || hostA->rgba.size() != aWidth * aHeight * 4
        || hostB->width != bWidth || hostB->height != bHeight || hostB->rgba.size() != bWidth * bHeight * 4)
    {
        throw std::invalid_argument("a source host copy does not match its size");
    }

    const bool aLinear = a.encoding() == Encoding::Linear;
    const bool bLinear = b.encoding() == Encoding::Linear;
    const float * table = srgbToLinearTable();
    const size_t width = static_cast<size_t>(overlap.width);
    const size_t height = static_cast<size_t>(overlap.height);

    std::vector<float> linearA(width * height * 3);
    double sum = 0.0;
    uint64_t samples = 0;
    for (size_t row = 0; row < height; ++row)
    {
        const int64_t canvasY = static_cast<int64_t>(overlap.y) + static_cast<int64_t>(row);
        const size_t aY = static_cast<size_t>(canvasY - aOffsetY);
        const size_t bY = static_cast<size_t>(canvasY - bOffsetY);
        for (size_t col = 0; col < width; ++col)
        {
            const int64_t canvasX = static_cast<int64_t>(overlap.x) + static_cast<int64_t>(col);
            const size_t aIndex = (aY * aWidth + static_cast<size_t>(canvasX - aOffsetX)) * 4;
            const size_t bIndex = (bY * bWidth + static_cast<size_t>(canvasX - bOffsetX)) * 4;
            const size_t out = (row * width + col) * 3;
            for (size_t c = 0; c < 3; ++c)
            {
                const float rawA = hostA->rgba[aIndex + c];
                const float rawB = hostB->rgba[bIndex + c];
                const float linearValueA = linearSample(rawA, aLinear, table);
                const float linearValueB = linearSample(rawB, bLinear, table);
                linearA[out + c] = linearValueA;
                // Same rule as linearMse(): a sample pair counts only when both values are finite.
                if (std::isfinite(rawA) && std::isfinite(rawB))
                {
                    const double d = static_cast<double>(linearValueA) - static_cast<double>(linearValueB);
                    sum += d * d;
                    ++samples;
                }
            }
        }
    }

    HdrPlan plan;
    if (samples > 0)
    {
        plan.linearMse = sum / static_cast<double>(samples);
    }

    HdrFlipParams params;
    params.toneMapper = settings.toneMapper;
    params.ppd = ppd;
    params.startExposure = settings.start;
    params.stopExposure = settings.stop;
    params.numExposures = settings.count;
    plan.exposures = hdrExposuresFromLinear(linearA, static_cast<uint32_t>(overlap.width),
                                            static_cast<uint32_t>(overlap.height), params, true);
    return plan;
}

} // namespace

// The active HDR-FLIP run. The job keeps the color views of both sources, which stay alive until
// finishHdr() or cancelHdr() (see beginHdr()).
struct FlipGpu::HdrJob
{
    FlipGpuParams params{};     // every push constant except the exposure, and the flags set per step
    BufferView colorA;
    BufferView colorB;
    PixelRect overlap;
    ToneMapper toneMapper = ToneMapper::Aces;
    float ppd = 0.0f;
    float start = 0.0f;
    float stop = 0.0f;
    float step = 0.0f;          // (stop - start) / (count - 1), computed as FLIP.h does
    uint32_t count = 0;         // exposures, at least 2
    uint32_t done = 0;          // exposures finished so far
    uint32_t tilesX = 0;        // dispatch groups per axis
    uint32_t tilesY = 0;
    uint64_t pixelCount = 0;
    double linearMse = std::numeric_limits<double>::quiet_NaN();
};

FlipGpu::FlipGpu(GpuContext & context) : m_context(context)
{
    const uint32_t pushSize = static_cast<uint32_t>(sizeof(FlipGpuParams));
    m_prepare = context.makeComputePipeline("flip_prepare", storageBindings(kPrepareBindings), pushSize);
    m_prepareHdr = context.makeComputePipeline("flip_prepare_hdr", storageBindings(kPrepareBindings), pushSize);
    m_spatialH = context.makeComputePipeline("flip_spatial_h", storageBindings(kSpatialHBindings), pushSize);
    m_spatialVColor = context.makeComputePipeline("flip_spatial_v_color", storageBindings(kSpatialVBindings), pushSize);
    m_featureH = context.makeComputePipeline("flip_feature_h", storageBindings(kFeatureHBindings), pushSize);
    m_featureVFinal = context.makeComputePipeline("flip_feature_v_final", storageBindings(kFeatureVBindings), pushSize);
    m_hdrAccumulate = context.makeComputePipeline("flip_hdr_accumulate", storageBindings(kHdrAccumulateBindings), pushSize);
}

FlipGpu::~FlipGpu() = default;

uint64_t FlipGpu::requiredBytes(uint64_t pixels, float ppd)
{
    return requiredBytesFor(pixels, flipKernels(ppd));
}

uint64_t FlipGpu::requiredBytesHdr(uint64_t pixels, float ppd)
{
    return requiredBytes(pixels, ppd) + kHdrBytesPerPixel * pixels;
}

BufferView FlipGpu::errorMapView() const
{
    const std::unique_ptr<sabercat::Buffer> & map = m_mapIsHdr ? m_hdrMax : m_map;
    if (!m_mapValid || !map)
    {
        return {};
    }
    BufferView view = map->view(static_cast<uint32_t>(sizeof(float)));
    view.size = static_cast<VkDeviceSize>(m_mapPixels) * sizeof(float);
    return view;
}

void FlipGpu::release()
{
    cancelHdr();
    m_mapValid = false;
    m_mapIsHdr = false;
    m_mapPixels = 0;
    releaseScratch();
    m_srgbTable.reset();
    m_kernelBuffer.reset();
    m_kernelPpd = 0.0f;
    m_diag.reset();
}

uint64_t FlipGpu::heldBytes() const
{
    uint64_t total = 0;
    const auto add = [&total](const std::unique_ptr<sabercat::Buffer> & buffer)
    {
        if (buffer)
        {
            total += buffer->size();
        }
    };
    add(m_srgbTable);
    add(m_kernelBuffer);
    add(m_diag);
    add(m_yA);
    add(m_yB);
    add(m_interA);
    add(m_interB);
    add(m_map);
    add(m_hdrMax);
    add(m_hdrExposure);
    return total;
}

void FlipGpu::releaseScratch()
{
    m_yA.reset();
    m_yB.reset();
    m_interA.reset();
    m_interB.reset();
    m_map.reset();
    m_hdrMax.reset();
    m_hdrExposure.reset();
    m_capacityPixels = 0;
    m_mapValid = false;
}

void FlipGpu::cancelHdr()
{
    m_job.reset();
}

void FlipGpu::ensureBuffers(uint64_t pixels, float ppd, const FlipKernels & kernels, bool hdr)
{
    sabercat::Device & device = m_context.device();
    if (!m_srgbTable)
    {
        m_srgbTable = std::make_unique<sabercat::Buffer>(device, kSrgbBytes);
        m_srgbTable->upload(srgbToLinearTable(), static_cast<size_t>(kSrgbBytes));
    }
    if (!m_diag)
    {
        m_diag = std::make_unique<sabercat::Buffer>(device, kDiagBytes);
    }
    if (!m_kernelBuffer || m_kernelPpd != ppd)
    {
        m_kernelBuffer.reset();
        m_kernelPpd = 0.0f;
        const std::vector<KernelEntry> table = buildKernelTable(kernels);
        const uint64_t tableBytes = table.size() * sizeof(KernelEntry);
        m_kernelBuffer = std::make_unique<sabercat::Buffer>(device, tableBytes);
        m_kernelBuffer->upload(table.data(), static_cast<size_t>(tableBytes));
        m_kernelPpd = ppd;
    }
    if (pixels > m_capacityPixels)
    {
        releaseScratch();
        m_yA = std::make_unique<sabercat::Buffer>(device, pixels * 16);
        m_yB = std::make_unique<sabercat::Buffer>(device, pixels * 16);
        m_interA = std::make_unique<sabercat::Buffer>(device, pixels * 16);
        m_interB = std::make_unique<sabercat::Buffer>(device, pixels * 16);
        m_map = std::make_unique<sabercat::Buffer>(device, pixels * sizeof(float));
        m_capacityPixels = pixels;
    }
    if (hdr && !m_hdrMax)
    {
        m_hdrMax = std::make_unique<sabercat::Buffer>(device, m_capacityPixels * sizeof(float));
        m_hdrExposure = std::make_unique<sabercat::Buffer>(device, m_capacityPixels * sizeof(float));
    }
}

void FlipGpu::bindPasses(const BufferView & colorA, const BufferView & colorB, bool hdr)
{
    const uint32_t stride16 = 16;
    const uint32_t stride4 = static_cast<uint32_t>(sizeof(float));
    const BufferView kernel = m_kernelBuffer->view(stride16);

    ComputeKernel & prepare = hdr ? *m_prepareHdr : *m_prepare;
    prepare.bindBuffer(0, kernel);
    prepare.bindBuffer(1, colorA);
    prepare.bindBuffer(2, colorB);
    prepare.bindBuffer(3, m_srgbTable->view(stride4));
    prepare.bindBuffer(4, m_yA->view(stride16));
    prepare.bindBuffer(5, m_yB->view(stride16));
    prepare.bindBuffer(6, m_diag->view(stride4));

    m_spatialH->bindBuffer(0, kernel);
    m_spatialH->bindBuffer(1, m_yA->view(stride16));
    m_spatialH->bindBuffer(2, m_yB->view(stride16));
    m_spatialH->bindBuffer(3, m_interA->view(stride16));
    m_spatialH->bindBuffer(4, m_interB->view(stride16));

    m_spatialVColor->bindBuffer(0, kernel);
    m_spatialVColor->bindBuffer(1, m_interA->view(stride16));
    m_spatialVColor->bindBuffer(2, m_interB->view(stride16));
    m_spatialVColor->bindBuffer(3, m_map->view(stride4));

    m_featureH->bindBuffer(0, kernel);
    m_featureH->bindBuffer(1, m_yA->view(stride16));
    m_featureH->bindBuffer(2, m_yB->view(stride16));
    m_featureH->bindBuffer(3, m_interA->view(stride16));
    m_featureH->bindBuffer(4, m_interB->view(stride16));

    m_featureVFinal->bindBuffer(0, kernel);
    m_featureVFinal->bindBuffer(1, m_interA->view(stride16));
    m_featureVFinal->bindBuffer(2, m_interB->view(stride16));
    m_featureVFinal->bindBuffer(3, m_map->view(stride4));

    if (hdr)
    {
        // The accumulate pass reads this exposure's error (m_map) and merges it into the HDR maps.
        m_hdrAccumulate->bindBuffer(0, kernel);
        m_hdrAccumulate->bindBuffer(1, m_map->view(stride4));
        m_hdrAccumulate->bindBuffer(2, m_hdrMax->view(stride4));
        m_hdrAccumulate->bindBuffer(3, m_hdrExposure->view(stride4));
    }
}

std::optional<FlipGpuResult> FlipGpu::compute(const MediaSource & a, int32_t aOffsetX, int32_t aOffsetY,
                                              const MediaSource & b, int32_t bOffsetX, int32_t bOffsetY,
                                              PixelRect overlap, float ppd)
{
    // Any call invalidates the previous map; it is set again only by a successful run.
    cancelHdr();
    m_mapValid = false;

    if (a.status() != SourceStatus::Ready || b.status() != SourceStatus::Ready)
    {
        return std::nullopt;
    }
    // Written so that NaN fails the comparison and is rejected too.
    if (!(ppd >= kMinFlipPpd && ppd <= kMaxFlipPpd))
    {
        throw std::invalid_argument("FLIP ppd must be from 1 to 1000");
    }
    if (overlap.width <= 0 || overlap.height <= 0)
    {
        return std::nullopt;
    }

    BufferView colorA;
    BufferView colorB;
    const uint64_t pixelCount = checkOverlap(a, aOffsetX, aOffsetY, b, bOffsetX, bOffsetY, overlap, colorA, colorB);

    const FlipKernels kernels = flipKernels(ppd);

    // Budget: what this call adds to what the object holds. Buffers that are replaced
    // (an old kernel table, or scratch for a smaller overlap) are released before the
    // new ones are created, so they are free for the new allocations.
    requireBudget(m_context.device(), requiredBytesFor(pixelCount, kernels), heldBytes());

    ensureBuffers(pixelCount, ppd, kernels, false);

    const uint32_t zeroDiag[4] = { 0, 0, 0, 0 };
    m_diag->upload(zeroDiag, sizeof(zeroDiag));

    bindPasses(colorA, colorB, false);

    FlipGpuParams params = makeParams(a, aOffsetX, aOffsetY, b, bOffsetX, bOffsetY, overlap, kernels);

    const uint32_t groupsX = static_cast<uint32_t>(tilesFor(overlap.width));
    const uint32_t groupsY = static_cast<uint32_t>(tilesFor(overlap.height));
    const uint32_t pushSize = static_cast<uint32_t>(sizeof(params));
    m_context.submit(
        [&](VkCommandBuffer cmd)
        {
            // The passes run in order; each one reads the previous pass's output.
            sabercat::Device::recordGlobalBarrier(cmd);
            m_prepare->record(cmd, groupsX, groupsY, 1, &params, pushSize);
            sabercat::Device::recordGlobalBarrier(cmd);
            m_spatialH->record(cmd, groupsX, groupsY, 1, &params, pushSize);
            sabercat::Device::recordGlobalBarrier(cmd);
            m_spatialVColor->record(cmd, groupsX, groupsY, 1, &params, pushSize);
            sabercat::Device::recordGlobalBarrier(cmd);
            m_featureH->record(cmd, groupsX, groupsY, 1, &params, pushSize);
            sabercat::Device::recordGlobalBarrier(cmd);
            m_featureVFinal->record(cmd, groupsX, groupsY, 1, &params, pushSize);
            sabercat::Device::recordGlobalBarrier(cmd);
        });

    std::vector<float> errorMap(static_cast<size_t>(pixelCount));
    m_map->download(errorMap.data(), static_cast<size_t>(pixelCount * sizeof(float)));
    uint32_t diag[4] = { 0, 0, 0, 0 };
    m_diag->download(diag, sizeof(diag));

    const double mean = meanOf(errorMap.data(), static_cast<size_t>(pixelCount));
    if (!std::isfinite(mean))
    {
        throw std::runtime_error("FLIP produced a non-finite mean error");
    }

    m_mapPixels = pixelCount;
    m_mapValid = true;
    m_mapIsHdr = false;

    FlipGpuResult result;
    result.mean = mean;
    result.nonFiniteSamples = diag[0];
    result.overlap = overlap;
    result.ppd = ppd;
    result.errorMap = std::move(errorMap);
    return result;
}

bool FlipGpu::beginHdr(const MediaSource & a, int32_t aOffsetX, int32_t aOffsetY,
                       const MediaSource & b, int32_t bOffsetX, int32_t bOffsetY,
                       PixelRect overlap, float ppd, const FlipHdrSettings & settings)
{
    cancelHdr();
    m_mapValid = false;

    if (a.status() != SourceStatus::Ready || b.status() != SourceStatus::Ready)
    {
        return false;
    }
    if (!(ppd >= kMinFlipPpd && ppd <= kMaxFlipPpd))
    {
        throw std::invalid_argument("FLIP ppd must be from 1 to 1000");
    }
    if (overlap.width <= 0 || overlap.height <= 0)
    {
        return false;
    }

    BufferView colorA;
    BufferView colorB;
    const uint64_t pixelCount = checkOverlap(a, aOffsetX, aOffsetY, b, bOffsetX, bOffsetY, overlap, colorA, colorB);

    // Host side first: the exposure range and linearMse. Its temporaries are gone before any device
    // memory is requested.
    const HdrPlan plan = planHdrOnHost(a, aOffsetX, aOffsetY, b, bOffsetX, bOffsetY, overlap, ppd, settings);

    const FlipKernels kernels = flipKernels(ppd);
    requireBudget(m_context.device(), requiredBytesHdr(pixelCount, ppd), heldBytes());
    ensureBuffers(pixelCount, ppd, kernels, true);

    // The HDR run accumulates into the diag count, so it starts from zero (no per-step clear).
    const uint32_t zeroDiag[4] = { 0, 0, 0, 0 };
    m_diag->upload(zeroDiag, sizeof(zeroDiag));

    auto job = std::make_unique<HdrJob>();
    job->params = makeParams(a, aOffsetX, aOffsetY, b, bOffsetX, bOffsetY, overlap, kernels);
    job->params.toneMapper = static_cast<uint32_t>(flipToneMapperIndex(settings.toneMapper));
    job->colorA = colorA;
    job->colorB = colorB;
    job->overlap = overlap;
    job->toneMapper = settings.toneMapper;
    job->ppd = ppd;
    job->start = plan.exposures.start;
    job->stop = plan.exposures.stop;
    job->count = static_cast<uint32_t>(plan.exposures.count);
    job->step = (job->stop - job->start) / static_cast<float>(job->count - 1);
    job->tilesX = static_cast<uint32_t>(tilesFor(overlap.width));
    job->tilesY = static_cast<uint32_t>(tilesFor(overlap.height));
    job->pixelCount = pixelCount;
    job->linearMse = plan.linearMse;
    m_job = std::move(job);
    return true;
}

bool FlipGpu::hdrActive() const
{
    return m_job != nullptr;
}

uint32_t FlipGpu::hdrDone() const
{
    return m_job ? m_job->done : 0u;
}

uint32_t FlipGpu::hdrCount() const
{
    return m_job ? m_job->count : 0u;
}

bool FlipGpu::stepHdr(uint32_t maxExposures)
{
    if (!m_job)
    {
        throw std::logic_error("no HDR-FLIP run is active");
    }
    HdrJob & job = *m_job;
    const uint32_t first = job.done;
    const uint32_t remaining = job.count - job.done;
    const uint32_t chunk = std::min(std::max(maxExposures, uint32_t{ 1 }), remaining);
    try
    {
        bindPasses(job.colorA, job.colorB, true);
        const uint32_t pushSize = static_cast<uint32_t>(sizeof(FlipGpuParams));
        const uint32_t groupsX = job.tilesX;
        const uint32_t groupsY = job.tilesY;
        m_context.submit(
            [&](VkCommandBuffer cmd)
            {
                for (uint32_t k = 0; k < chunk; ++k)
                {
                    const uint32_t i = first + k;
                    FlipGpuParams params = job.params;
                    // Exposure i: start + i * step, scale 2^exposure, and the index ratio of FLIP.h
                    // (the same float arithmetic as its HDR loop).
                    const float exposure = job.start + static_cast<float>(i) * job.step;
                    params.exposureScale = std::pow(2.0f, exposure);
                    params.exposureFraction = static_cast<float>(i) / static_cast<float>(job.count - 1);
                    // The first exposure also counts the non-finite samples and starts the max map.
                    params.hdrFlags = i == 0 ? (kHdrFlagCountNonFinite | kHdrFlagFirstExposure) : 0u;

                    sabercat::Device::recordGlobalBarrier(cmd);
                    m_prepareHdr->record(cmd, groupsX, groupsY, 1, &params, pushSize);
                    sabercat::Device::recordGlobalBarrier(cmd);
                    m_spatialH->record(cmd, groupsX, groupsY, 1, &params, pushSize);
                    sabercat::Device::recordGlobalBarrier(cmd);
                    m_spatialVColor->record(cmd, groupsX, groupsY, 1, &params, pushSize);
                    sabercat::Device::recordGlobalBarrier(cmd);
                    m_featureH->record(cmd, groupsX, groupsY, 1, &params, pushSize);
                    sabercat::Device::recordGlobalBarrier(cmd);
                    m_featureVFinal->record(cmd, groupsX, groupsY, 1, &params, pushSize);
                    sabercat::Device::recordGlobalBarrier(cmd);
                    m_hdrAccumulate->record(cmd, groupsX, groupsY, 1, &params, pushSize);
                    sabercat::Device::recordGlobalBarrier(cmd);
                }
            });
    }
    catch (...)
    {
        // A failed step leaves no job, so nothing half-done can be finished.
        m_job.reset();
        m_mapValid = false;
        throw;
    }
    job.done += chunk;
    return job.done == job.count;
}

FlipGpuResult FlipGpu::finishHdr()
{
    if (!m_job)
    {
        throw std::logic_error("no HDR-FLIP run is active");
    }
    const HdrJob & job = *m_job;
    const size_t pixels = static_cast<size_t>(job.pixelCount);

    FlipGpuResult result;
    try
    {
        if (job.done != job.count)
        {
            throw std::logic_error("HDR-FLIP exposures are not all done");
        }
        std::vector<float> maxMap(pixels);
        std::vector<float> exposureMap(pixels);
        m_hdrMax->download(maxMap.data(), pixels * sizeof(float));
        m_hdrExposure->download(exposureMap.data(), pixels * sizeof(float));
        uint32_t diag[4] = { 0, 0, 0, 0 };
        m_diag->download(diag, sizeof(diag));

        const double mean = meanOf(maxMap.data(), pixels);
        if (!std::isfinite(mean))
        {
            throw std::runtime_error("HDR-FLIP produced a non-finite mean error");
        }

        FlipHdrInfo info;
        info.exposureMap = std::move(exposureMap);
        info.linearMse = job.linearMse;
        info.start = job.start;
        info.stop = job.stop;
        info.count = static_cast<int32_t>(job.count);
        info.toneMapper = job.toneMapper;

        result.mean = mean;
        result.nonFiniteSamples = diag[0];
        result.overlap = job.overlap;
        result.ppd = job.ppd;
        result.errorMap = std::move(maxMap);
        result.hdr = std::move(info);
    }
    catch (...)
    {
        // Any failure, including finishing early, ends the run without a map.
        m_job.reset();
        m_mapValid = false;
        throw;
    }

    m_job.reset();
    m_mapPixels = pixels;
    m_mapValid = true;
    m_mapIsHdr = true;
    return result;
}

std::optional<FlipGpuResult> FlipGpu::computeHdr(const MediaSource & a, int32_t aOffsetX, int32_t aOffsetY,
                                                 const MediaSource & b, int32_t bOffsetX, int32_t bOffsetY,
                                                 PixelRect overlap, float ppd, const FlipHdrSettings & settings)
{
    if (!beginHdr(a, aOffsetX, aOffsetY, b, bOffsetX, bOffsetY, overlap, ppd, settings))
    {
        return std::nullopt;
    }
    stepHdr(hdrCount());
    return finishHdr();
}

} // namespace sabercat::compare::gui
