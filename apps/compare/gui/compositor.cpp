// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "compositor.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "image_compare.h"
#include "sabercat/buffer.h"
#include "sabercat/device.h"
#include "sabercat/image.h"

namespace sabercat::compare::gui
{

namespace
{

// The shader's "no slot" value.
constexpr uint32_t kGpuNoSlot = 0xFFFFFFFFu;
// Workgroup edge of compare_composite.slang (numthreads(8, 8, 1)).
constexpr uint32_t kTileSize = 8;
// Exposure range of displayTransform() in compare_layout.h.
constexpr float kMaxExposureStops = 32.0f;

// Mirrors of the shader structs in compare_composite.slang. They are std430 with
// 16-byte-aligned records, so the offsets below are the offsets the shader reads.

// One visible source: 48 bytes.
struct alignas(16) GpuSourceRecord
{
    int32_t offsetX;
    int32_t offsetY;
    uint32_t width;
    uint32_t height;
    uint32_t encoding;      // 0 Display, 1 Linear
    uint32_t premultiplied; // 0 or 1
    uint32_t toneMapper;    // ToneMapper value: 0 Aces, 1 Hable, 2 Reinhard
    float exposureStops;
    float background[4];    // rgb, then a pad
};
static_assert(sizeof(GpuSourceRecord) == 48, "GpuSourceRecord must match the shader's 48-byte record");
static_assert(offsetof(GpuSourceRecord, background) == 32, "background must sit at offset 32");

// One pane: 32 bytes. rect is x, y, width, height in view pixels.
struct alignas(16) GpuPaneRecord
{
    int32_t rect[4];
    uint32_t slot;
    uint32_t reserved[3];
};
static_assert(sizeof(GpuPaneRecord) == 32, "GpuPaneRecord must match the shader's 32-byte record");

// The depth record of one source, as the shader reads it: {stride in floats per pixel
// (0 = no depth), front component, coverage component (kGpuNoSlot = color alpha), 0}.
using GpuDepthRecord = uint32_t[4];

// The parameter block: 464 bytes.
struct alignas(16) GpuCompositeParams
{
    int32_t extent[4];       // padded extent: x, y, width, height
    uint32_t view[4];        // view width, view height, mode, nearest
    uint32_t slots[4];       // sourceA slot, sourceB slot, visible count, pane count (Depth and Alpha: chosen slot)
    float transform[4];      // panX, panY, zoom, wipe position
    uint32_t flags[4];       // x: wipe vertical (1) or horizontal (0); y: map uses B; rest unused
    float misc[4];           // x: diff gain; y: depth range min; z: depth range max; w unused
    GpuPaneRecord panes[kMaxVisible];
    GpuSourceRecord sources[kMaxVisible];
    GpuDepthRecord depthA;   // slot A's depth (the chosen source for Depth and Alpha)
    GpuDepthRecord depthB;   // slot B's depth
    int32_t flipRect[4];     // Flip: the overlap the FLIP map covers, x, y, width, height (canvas pixels)
};
static_assert(sizeof(GpuCompositeParams) == 464, "GpuCompositeParams must match the shader's 464-byte block");
static_assert(offsetof(GpuCompositeParams, panes) == 96, "panes must sit at offset 96");
static_assert(offsetof(GpuCompositeParams, sources) == 224, "sources must sit at offset 224");
static_assert(offsetof(GpuCompositeParams, depthA) == 416, "depthA must sit at offset 416");
static_assert(offsetof(GpuCompositeParams, depthB) == 432, "depthB must sit at offset 432");
static_assert(offsetof(GpuCompositeParams, flipRect) == 448, "flipRect must sit at offset 448");

bool isVisible(const LayoutInput & layout, int32_t index)
{
    return index >= 0 && index < layout.sourceCount && index < kMaxSources
        && layout.sources[static_cast<size_t>(index)].visible;
}

uint32_t toGpuSlot(int32_t slot)
{
    return slot < 0 ? kGpuNoSlot : static_cast<uint32_t>(slot);
}

// Exposure as displayTransform() sees it: NaN is 0, clamped to [-32, 32].
float sanitizeExposure(float exposureStops)
{
    if (std::isnan(exposureStops))
    {
        return 0.0f;
    }
    return std::clamp(exposureStops, -kMaxExposureStops, kMaxExposureStops);
}

// The magma table the shader indexes: colorizeMagma() of the 256 values k / 255, one
// float4 per entry (byte / 255, alpha 1). Writing byte / 255 to the UNORM image gives the
// byte back exactly.
std::vector<float> magmaTable()
{
    std::vector<float> gray(256);
    for (size_t k = 0; k < gray.size(); ++k)
    {
        gray[k] = static_cast<float>(k) / 255.0f;
    }
    const Rgb8Image colors = colorizeMagma(gray, 256, 1);
    std::vector<float> table(256 * 4);
    for (size_t k = 0; k < 256; ++k)
    {
        for (size_t c = 0; c < 3; ++c)
        {
            table[4 * k + c] = static_cast<float>(colors.rgb[3 * k + c]) / 255.0f;
        }
        table[4 * k + 3] = 1.0f;
    }
    return table;
}

} // namespace

int32_t paddingCounterpartSlot(int32_t slot, int32_t slotA, int32_t slotB, int32_t sourceCount)
{
    if (slot == slotA && slotB >= 0 && slotB < sourceCount && slotB != slot)
    {
        return slotB;
    }
    if (slot == slotB && slotA >= 0 && slotA < sourceCount && slotA != slot)
    {
        return slotA;
    }
    for (int32_t k = 0; k < sourceCount; ++k)
    {
        if (k != slot)
        {
            return k;
        }
    }
    return -1;
}

// ---- Compositor ------------------------------------------------------------------

Compositor::Compositor(GpuContext & context) : m_device(context.device())
{
    const std::vector<VkDescriptorType> bindings = {
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  // 0: composite
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 1..4: source slots
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 5: parameters
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 6: magma table
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 7: depth of slot A
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 8: depth of slot B
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 9: FLIP error map
    };
    m_kernel = context.makeComputePipeline("compare_composite", bindings, 0);

    m_params = std::make_unique<Buffer>(m_device, sizeof(GpuCompositeParams));
    m_magma = std::make_unique<Buffer>(m_device, 256 * 4 * sizeof(float));
    const std::vector<float> table = magmaTable();
    m_magma->upload(table.data(), table.size() * sizeof(float));

    // Bound to the slots that have no depth. The shader never reads it.
    m_dummy = std::make_unique<Buffer>(m_device, 4 * sizeof(float));
    const float zero[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    m_dummy->upload(zero, sizeof(zero));
}

Compositor::~Compositor() = default;

void Compositor::resize(uint32_t width, uint32_t height)
{
    if (width == m_width && height == m_height)
    {
        return;
    }
    m_image.reset();
    m_width = width;
    m_height = height;
    m_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (m_width == 0 || m_height == 0)
    {
        return;
    }
    m_image = std::make_unique<sabercat::Image>(m_device, m_width, m_height, VK_FORMAT_R8G8B8A8_UNORM,
                                      VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
                                          | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
}

VkImageView Compositor::view() const
{
    return m_image ? m_image->getImageView() : VK_NULL_HANDLE;
}

VkImage Compositor::image() const
{
    return m_image ? m_image->getHandle() : VK_NULL_HANDLE;
}

void Compositor::render(const CompositeInput & input)
{
    const LayoutInput & layout = input.layout;
    if (layout.windowWidth < 0 || layout.windowHeight < 0
        || static_cast<uint32_t>(layout.windowWidth) != m_width
        || static_cast<uint32_t>(layout.windowHeight) != m_height)
    {
        throw std::invalid_argument("composite view size " + std::to_string(layout.windowWidth) + "x"
            + std::to_string(layout.windowHeight) + " does not match the image " + std::to_string(m_width) + "x"
            + std::to_string(m_height) + "; call resize() first");
    }
    if (m_width == 0 || m_height == 0)
    {
        return;
    }
    if (!std::isfinite(input.diffGain) || input.diffGain < 0.0f)
    {
        throw std::invalid_argument("diffGain must be finite and not negative");
    }
    if (!std::isfinite(input.depthMin) || !std::isfinite(input.depthMax) || input.depthMin > input.depthMax)
    {
        throw std::invalid_argument("depth range must be finite with depthMin <= depthMax");
    }
    if (input.mode == CompositeMode::Flip)
    {
        if (!input.flipMap.valid() || input.flipRect.width <= 0 || input.flipRect.height <= 0)
        {
            throw std::invalid_argument("Flip needs a FLIP map and a non-empty flip rectangle");
        }
        const VkDeviceSize needed = static_cast<VkDeviceSize>(input.flipRect.width)
            * static_cast<VkDeviceSize>(input.flipRect.height) * sizeof(float);
        if (input.flipMap.size < needed)
        {
            throw std::invalid_argument("the FLIP map is smaller than its flip rectangle");
        }
    }

    // Slot k is the k-th visible source in index order (the pane order of computePanes).
    std::array<int32_t, kMaxSources> slotOf{};
    slotOf.fill(-1);
    std::array<GpuSourceRecord, kMaxVisible> records{};
    std::array<const MediaSource *, kMaxVisible> media{};
    // Depth view and record of each visible slot. A slot without depth keeps an invalid view
    // and a stride-0 record, which the shader never reads.
    std::array<BufferView, kMaxVisible> depthViews{};
    std::array<GpuDepthRecord, kMaxVisible> depthRecords{};
    int32_t count = 0;
    for (int32_t i = 0; i < layout.sourceCount && i < kMaxSources; ++i)
    {
        const LayoutSource & geometry = layout.sources[static_cast<size_t>(i)];
        if (!geometry.visible)
        {
            continue;
        }
        const CompositeSource & source = input.sources[static_cast<size_t>(i)];
        const std::string name = "source " + std::to_string(i);
        if (source.media == nullptr || source.media->status() != SourceStatus::Ready)
        {
            throw std::invalid_argument(name + " is visible but not ready");
        }
        if (count >= kMaxVisible)
        {
            throw std::invalid_argument("more than " + std::to_string(kMaxVisible) + " visible sources");
        }
        if (source.media->width() != static_cast<uint32_t>(geometry.width)
            || source.media->height() != static_cast<uint32_t>(geometry.height))
        {
            throw std::invalid_argument(name + " has a size that differs from its layout entry");
        }
        const BufferView color = source.media->color();
        if (!color.valid() || color.size != static_cast<VkDeviceSize>(geometry.width) * geometry.height * 16)
        {
            throw std::invalid_argument(name + " has no RGBA32F color buffer of its size");
        }

        slotOf[static_cast<size_t>(i)] = count;
        GpuSourceRecord & record = records[static_cast<size_t>(count)];
        record.offsetX = geometry.offsetX;
        record.offsetY = geometry.offsetY;
        record.width = static_cast<uint32_t>(geometry.width);
        record.height = static_cast<uint32_t>(geometry.height);
        record.encoding = source.media->encoding() == Encoding::Linear ? 1u : 0u;
        record.premultiplied = source.media->premultiplied() ? 1u : 0u;
        record.toneMapper = static_cast<uint32_t>(source.toneMapper);
        record.exposureStops = sanitizeExposure(source.exposureStops);
        record.background[0] = source.background.r;
        record.background[1] = source.background.g;
        record.background[2] = source.background.b;
        record.background[3] = 0.0f;
        media[static_cast<size_t>(count)] = source.media;

        // Depth for the depth and alpha maps. A source that reports depth must have a depth
        // view of exactly the size depthInfo() describes, or the shader would index outside it.
        const DepthInfo depthInfo = source.media->depthInfo();
        GpuDepthRecord & depthRecord = depthRecords[static_cast<size_t>(count)];
        depthRecord[0] = 0u;
        depthRecord[1] = 0u;
        depthRecord[2] = kGpuNoSlot;
        depthRecord[3] = 0u;
        if (depthInfo.floatsPerPixel > 0)
        {
            const BufferView depth = source.media->depth();
            const VkDeviceSize expected = static_cast<VkDeviceSize>(geometry.width) * geometry.height
                * depthInfo.floatsPerPixel * sizeof(float);
            if (!depth.valid() || depth.size != expected)
            {
                throw std::invalid_argument(name + " reports depth but has no depth buffer of its size");
            }
            if (depthInfo.frontComponent >= static_cast<int32_t>(depthInfo.floatsPerPixel)
                || depthInfo.coverageComponent >= static_cast<int32_t>(depthInfo.floatsPerPixel))
            {
                throw std::invalid_argument(name + " names a depth component it does not have");
            }
            depthViews[static_cast<size_t>(count)] = depth;
            depthRecord[0] = depthInfo.floatsPerPixel;
            depthRecord[1] = toGpuSlot(depthInfo.frontComponent);
            depthRecord[2] = toGpuSlot(depthInfo.coverageComponent);
        }
        ++count;
    }

    const int32_t sourceA = layout.view.sourceA;
    const int32_t sourceB = layout.view.sourceB;
    // The source the single-source map modes show: B when mapUseB, else A.
    const int32_t chosen = input.mapUseB ? sourceB : sourceA;
    const bool singleMap = input.mode == CompositeMode::Depth || input.mode == CompositeMode::Alpha;
    const bool pairNeeded = input.mode == CompositeMode::Wipe || input.mode == CompositeMode::Flicker
        || input.mode == CompositeMode::Diff || input.mode == CompositeMode::DepthDiff
        || input.mode == CompositeMode::AlphaDiff || input.mode == CompositeMode::Flip;
    if (pairNeeded && (!isVisible(layout, sourceA) || !isVisible(layout, sourceB)))
    {
        throw std::invalid_argument(
            "Wipe, Flicker, Diff, DepthDiff, AlphaDiff and Flip need the A and B sources visible");
    }
    if (singleMap && !isVisible(layout, chosen))
    {
        throw std::invalid_argument("Depth and Alpha need the chosen source visible");
    }
    const int32_t slotA = isVisible(layout, sourceA) ? slotOf[static_cast<size_t>(sourceA)] : -1;
    const int32_t slotB = isVisible(layout, sourceB) ? slotOf[static_cast<size_t>(sourceB)] : -1;
    const int32_t slotChosen = isVisible(layout, chosen) ? slotOf[static_cast<size_t>(chosen)] : -1;

    // The geometry the composite shows. Every mode but Panes uses the SingleScreen form.
    LayoutInput shown = layout;
    switch (input.mode)
    {
    case CompositeMode::Panes:
        break;
    case CompositeMode::Wipe:
    case CompositeMode::Diff:
    case CompositeMode::DepthDiff:
    case CompositeMode::AlphaDiff:
    case CompositeMode::Flip:
        shown.view.mode = ViewMode::SingleScreen;
        shown.view.abIndex = sourceA;
        break;
    case CompositeMode::Flicker:
        shown.view.mode = ViewMode::SingleScreen;
        shown.view.abIndex = input.flickerShowB ? sourceB : sourceA;
        break;
    case CompositeMode::Depth:
    case CompositeMode::Alpha:
        shown.view.mode = ViewMode::SingleScreen;
        shown.view.abIndex = chosen;
        break;
    }
    const PaneLayout panes = computePanes(shown);
    const PixelRect extent = paddedExtent(shown);
    const double zoom = effectiveZoom(shown);
    const double wipe = clampWipePosition(shown, shown.view.wipePosition);

    GpuCompositeParams params{};
    params.extent[0] = extent.x;
    params.extent[1] = extent.y;
    params.extent[2] = extent.width;
    params.extent[3] = extent.height;
    params.view[0] = m_width;
    params.view[1] = m_height;
    params.view[2] = static_cast<uint32_t>(input.mode);
    params.view[3] = input.nearest ? 1u : 0u;
    // The single-source map modes put the chosen source in the A slot.
    const int32_t firstSlot = singleMap ? slotChosen : slotA;
    const int32_t secondSlot = singleMap ? -1 : slotB;
    params.slots[0] = toGpuSlot(firstSlot);
    params.slots[1] = toGpuSlot(secondSlot);
    params.slots[2] = static_cast<uint32_t>(count);
    params.slots[3] = static_cast<uint32_t>(panes.count);
    params.transform[0] = static_cast<float>(shown.view.panX);
    params.transform[1] = static_cast<float>(shown.view.panY);
    params.transform[2] = static_cast<float>(zoom);
    params.transform[3] = static_cast<float>(wipe);
    params.flags[0] = shown.view.wipeOrientation == WipeOrientation::Vertical ? 1u : 0u;
    params.flags[1] = input.mapUseB ? 1u : 0u;
    params.misc[0] = input.diffGain;
    params.misc[1] = input.depthMin;
    params.misc[2] = input.depthMax;
    if (input.mode == CompositeMode::Flip)
    {
        params.flipRect[0] = input.flipRect.x;
        params.flipRect[1] = input.flipRect.y;
        params.flipRect[2] = input.flipRect.width;
        params.flipRect[3] = input.flipRect.height;
    }
    for (int32_t i = 0; i < panes.count && i < kMaxVisible; ++i)
    {
        const Pane & pane = panes.panes[static_cast<size_t>(i)];
        const int32_t slot = pane.source >= 0 && pane.source < kMaxSources
            ? slotOf[static_cast<size_t>(pane.source)]
            : -1;
        if (slot < 0)
        {
            throw std::logic_error("pane shows a source that has no slot");
        }
        GpuPaneRecord & out = params.panes[i];
        out.rect[0] = pane.rect.x;
        out.rect[1] = pane.rect.y;
        out.rect[2] = pane.rect.width;
        out.rect[3] = pane.rect.height;
        out.slot = static_cast<uint32_t>(slot);
    }
    for (int32_t k = 0; k < count; ++k)
    {
        params.sources[static_cast<size_t>(k)] = records[static_cast<size_t>(k)];
    }

    // Depth records of the two depth slots. The single-source map modes read the chosen
    // source through slot A; slot B is then unused.
    const int32_t depthSlots[2] = {firstSlot, secondSlot};
    const GpuDepthRecord emptyRecord{0u, 0u, kGpuNoSlot, 0u};
    for (uint32_t which = 0; which < 2; ++which)
    {
        const int32_t slot = depthSlots[which];
        const uint32_t * record = slot >= 0 ? depthRecords[static_cast<size_t>(slot)] : emptyRecord;
        uint32_t * out = which == 0 ? params.depthA : params.depthB;
        std::copy(record, record + 4, out);
    }

    m_params->upload(&params, sizeof(params));

    m_kernel->bindImage(0, m_image->getImageView());
    const uint32_t vec4Stride = 4 * sizeof(float);
    for (uint32_t k = 0; k < static_cast<uint32_t>(kMaxVisible); ++k)
    {
        if (k < static_cast<uint32_t>(count))
        {
            m_kernel->bindBuffer(1 + k, media[k]->color());
        }
        else
        {
            m_kernel->bindBuffer(1 + k, m_dummy->view(vec4Stride));
        }
    }
    m_kernel->bindBuffer(5, m_params->view(static_cast<uint32_t>(sizeof(GpuCompositeParams))));
    m_kernel->bindBuffer(6, m_magma->view(vec4Stride));
    // The depth buffers of the depth slots; a slot without depth binds the dummy.
    for (uint32_t which = 0; which < 2; ++which)
    {
        const int32_t slot = depthSlots[which];
        const bool present = slot >= 0 && depthViews[static_cast<size_t>(slot)].valid();
        m_kernel->bindBuffer(7 + which, present ? depthViews[static_cast<size_t>(slot)] : m_dummy->view(vec4Stride));
    }
    // The FLIP map; the other modes bind the dummy, which the shader never reads.
    m_kernel->bindBuffer(9, input.mode == CompositeMode::Flip ? input.flipMap : m_dummy->view(vec4Stride));

    const uint32_t groupsX = (m_width + kTileSize - 1) / kTileSize;
    const uint32_t groupsY = (m_height + kTileSize - 1) / kTileSize;
    m_device.immediateSubmit(
        [&](VkCommandBuffer cmd)
        {
            // The parameter upload and the previous frame's reads are done before the dispatch.
            VkMemoryBarrier barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
            barrier.srcAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
            vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
                                 1, &barrier, 0, nullptr, 0, nullptr);
            m_image->transitionLayout(cmd, m_layout, VK_IMAGE_LAYOUT_GENERAL);
            m_kernel->record(cmd, groupsX, groupsY, 1);
            m_image->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        });
    m_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
}

void Compositor::readback(std::vector<uint8_t> & rgba8)
{
    const size_t bytes = static_cast<size_t>(m_width) * m_height * 4;
    rgba8.assign(bytes, 0);
    if (bytes == 0 || !m_image || m_layout == VK_IMAGE_LAYOUT_UNDEFINED)
    {
        return;
    }
    if (!m_readback || m_readback->size() < bytes)
    {
        m_readback = std::make_unique<Buffer>(m_device, bytes);
    }

    m_device.immediateSubmit(
        [&](VkCommandBuffer cmd)
        {
            m_image->transitionLayout(cmd, m_layout, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
            VkBufferImageCopy region{};
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.layerCount = 1;
            region.imageExtent = {m_width, m_height, 1};
            vkCmdCopyImageToBuffer(cmd, m_image->getHandle(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   m_readback->handle(), 1, &region);
            m_image->transitionLayout(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        });
    m_layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    m_readback->download(rgba8.data(), bytes);
}

} // namespace sabercat::compare::gui
