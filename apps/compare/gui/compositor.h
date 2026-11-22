// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The compare view on the GPU. One compute dispatch writes the view area (the image
// region of the window) into an RGBA8 UNORM storage image. compare_layout.h holds the
// CPU model of the same rules, and the shader (shaders/compare_composite.slang)
// replicates them. The layout (panes, zoom, pan, wipe, padded extent, offsets) is
// computed on the CPU and passed in as data. Nothing about the layout is decided on
// the GPU.
//
// Rules, as the shader applies them:
//   - Window pixel (px, py) has center (px + 0.5, py + 0.5). In a pane with center
//     (cx, cy) it maps to image coordinate panX + (px + 0.5 - cx) / zoom (and likewise
//     for y), as windowToImage() does.
//   - Pixel floor(image coordinate) inside the padded extent is in the view. Outside it
//     the pixel is background, kCompositeBackground gray.
//   - Image pixel: floor(coordinate) inside the pane's source rectangle. Nearest mode
//     shows that pixel. Bilinear mode interpolates the source's display values between
//     source pixel centers, clamping the taps to the source rectangle (no padding is
//     read). Diff is always nearest.
//   - Padding pixel (inside the padded extent, outside the pane's source): the
//     counterpart's pixel at the same canvas position, inverted (1 - display value),
//     nearest. The counterpart is paddingCounterpartSlot(). When it has no pixel
//     there, the pixel is background.
//   - Display value of a source sample: a Premultiplied source is composited over its
//     background first (rgb + (1 - clamp(alpha)) * background). A Display source is then
//     clamped to [0,1]. A Linear source goes through displayTransform() (exposure,
//     tone mapper, sRGB; see compare_layout.h).
//   - Wipe: the source is A where the pixel center's image coordinate (along the wipe
//     axis) is less than the wipe position, else B. The wipe position is clamped to the
//     padded extent by the host.
//   - Flicker: the whole view shows B when flickerShowB, else A.
//   - Diff: mean over RGB of |A - B| with display values as displayRgb() defines them
//     (Display samples as stored, Linear clamped and sRGB encoded, premultiplied rgb
//     unchanged), times diffGain, clamped to 1, through colorizeMagma(). A pixel is
//     "Image" when both A and B have a pixel there; otherwise it is padding and takes
//     the maximum (mapPaddingValue(), top of the magma table). Outside the padded
//     extent: background.
//   - Depth map (Depth): the chosen source's front depth (its depth component
//     `front`, valid when finite and > 0) through the magma table, t = depthMapT() over
//     the depth range. The chosen source is sourceB when mapUseB, else sourceA. The
//     single pane covers the view. No valid depth in the chosen source's rectangle is
//     hatchValue(); padding (inside the extent, outside the source) is hatched too.
//     A source without depth is hatched everywhere inside the extent.
//   - Alpha map (Alpha): the chosen source's coverage as gray (a, a, a), a being the
//     coverage component when the depth buffer names one, else the color alpha. NaN is 0,
//     clamped to [0,1]. Not sRGB encoded. Padding is hatched.
//   - Depth difference (DepthDiff): |dA - dB| over the depth range through the magma
//     table, using sources A and B, where both have a valid depth. Where exactly one does,
//     the pixel is the top of the map. Where neither does, it is hatched. Pixels that are
//     not inside both sources are padding and take the top of the map.
//   - Alpha difference (AlphaDiff): min(1, |aA - aB| * diffGain) through the magma table
//     with the coverage of A and B. Padding takes the top of the map.
//   - FLIP (Flip): the error map computed on the GPU by FlipGpu (flip_gpu.h) over the
//     overlap of A and B, given as flipMap with flipRect (the overlap, canvas pixels).
//     A pixel inside flipRect shows magma(flipMap value). Padding (inside the padded
//     extent, outside flipRect) takes the top of the map, mapPaddingValue(); padding is
//     never run through FLIP. Outside the padded extent: background.
//   - Composite alpha is always 1.
//
// Rounding: the image format is VK_FORMAT_R8G8B8A8_UNORM. The shader stores float
// values and the hardware converts them with round-to-nearest, which the CPU reference
// matches as floor(v * 255 + 0.5) up to exact ties (see the note in the shader).
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include <vulkan/vulkan.h>

#include "compare_layout.h"
#include "gpu_context.h"
#include "media_source.h"
#include "sabercat/buffer_view.h"

namespace sabercat
{
class Buffer;
class Image;
class Device;
} // namespace sabercat

namespace sabercat::compare::gui
{

// The color outside the padded extent, and of padding with no counterpart (linear
// 0.12 in each channel, as a display value).
inline constexpr float kCompositeBackground = 0.12f;

// What the view shows. Panes is the layout as the user set it (split or single screen).
// The other modes show one pane covering the view. The values are the shader's kMode*
// constants (compare_composite.slang), so keep them in step.
enum class CompositeMode : uint8_t
{
    Panes,      // 0
    Wipe,       // 1
    Flicker,    // 2
    Diff,       // 3
    Depth,      // 4: front depth of the chosen source through magma
    Alpha,      // 5: coverage of the chosen source, gray
    DepthDiff,  // 6: |depth A - depth B| through magma
    AlphaDiff,  // 7: |alpha A - alpha B| * diffGain through magma
    Flip        // 8: GPU LDR-FLIP error map over the overlap, through magma
};

// Per layout source, indexed like LayoutInput::sources. media is read only when the
// source is visible in the layout.
struct CompositeSource
{
    RgbaColor background{0.0f, 0.0f, 0.0f, 1.0f}; // r, g, b: premultiplied background
    const MediaSource * media = nullptr;
    float exposureStops = 0.0f;
    ToneMapper toneMapper = ToneMapper::Aces;
};

struct CompositeInput
{
    // The view as the user set it. windowWidth and windowHeight are the view area in
    // pixels and must equal the compositor's image size. For every mode other than Panes
    // the compositor shows one pane, so zoom, pan, extent and the wipe clamp come from the
    // SingleScreen form of this layout: abIndex is A (Wipe, Diff, DepthDiff, AlphaDiff,
    // Flip), B when flickerShowB (Flicker), the chosen source (Depth, Alpha: B when mapUseB,
    // else A), and A for Flicker otherwise. Panes uses the layout unchanged.
    LayoutInput layout;
    std::array<CompositeSource, kMaxSources> sources{};
    CompositeMode mode = CompositeMode::Panes;
    // Nearest sampling for Panes, Wipe and Flicker. Diff and the map modes are always nearest.
    bool nearest = true;
    bool flickerShowB = false;
    // Depth and Alpha: the source shown is B when mapUseB, else A.
    bool mapUseB = false;
    // Diff and AlphaDiff: scale of the mean |A - B| before the clamp to 1. Finite and >= 0.
    float diffGain = 1.0f;
    // Depth and DepthDiff: the range the depth maps over, already resolved by the caller.
    // Finite, depthMin <= depthMax; an empty range maps every valid depth to 0 (DepthDiff:
    // any difference to the top).
    float depthMin = 0.0f;
    float depthMax = 1.0f;
    // Flip only: the FLIP error map (one float per pixel, row-major, top row first) over
    // flipRect, the overlap in canvas pixels (FlipGpuResult::overlap). The view is
    // FlipGpu::errorMapView(); the map must stay valid until render() returns.
    BufferView flipMap;
    PixelRect flipRect;
};

// The slot of the padded-extent counterpart of a pane. Slots number the visible
// sources in index order, which is the pane order of computePanes(). slot is the
// pane's source; slotA and slotB are the A/B pair (-1 when absent); sourceCount is
// the number of visible sources. Returns the other source of the A/B pair when it is
// visible and not the same slot; otherwise the first other visible source; otherwise -1.
int32_t paddingCounterpartSlot(int32_t slot, int32_t slotA, int32_t slotB, int32_t sourceCount);

// The composite of the view area. Owns its image, and its parameter, magma and readback
// buffers. Every call is synchronous.
class Compositor
{
  public:
    explicit Compositor(GpuContext & context);
    ~Compositor();

    Compositor(const Compositor &) = delete;
    Compositor & operator=(const Compositor &) = delete;

    // Recreates the image at the view area size. 0 in either dimension releases the image;
    // render() then does nothing and view() is VK_NULL_HANDLE. Recreation invalidates
    // image() and view().
    void resize(uint32_t width, uint32_t height);

    // Renders the view. Throws std::invalid_argument when the input does not match the
    // image (size, a visible source with no media or not Ready, a size that differs from
    // the source's width and height, or an A/B pair that Wipe, Flicker, Diff, DepthDiff,
    // AlphaDiff or Flip needs and the layout does not have visible, or the chosen source
    // that Depth or Alpha needs). Flip also throws when flipMap is invalid, flipRect is
    // empty, or flipMap holds fewer than flipRect.width * flipRect.height floats. A source
    // with depth whose depth() view is not of the size depthInfo() describes is refused
    // the same way, naming the source.
    void render(const CompositeInput & input);

    // The image after render(), in SHADER_READ_ONLY_OPTIMAL. Its view is sampled by ImGui.
    VkImageView view() const;
    VkImage image() const;
    uint32_t width() const
    {
        return m_width;
    }
    uint32_t height() const
    {
        return m_height;
    }

    // Copies the image to rgba8: width * height * 4 bytes, row-major, top row first,
    // alpha 255. All zero (alpha 0 too) when render() has never run.
    void readback(std::vector<uint8_t> & rgba8);

  private:
    Device & m_device;
    std::unique_ptr<ComputeKernel> m_kernel;
    std::unique_ptr<Buffer> m_params;
    std::unique_ptr<Buffer> m_dummy;  // bound to the slots without depth, never read
    std::unique_ptr<Buffer> m_magma;
    std::unique_ptr<Buffer> m_readback;
    std::unique_ptr<sabercat::Image> m_image;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    VkImageLayout m_layout = VK_IMAGE_LAYOUT_UNDEFINED;
};

} // namespace sabercat::compare::gui
