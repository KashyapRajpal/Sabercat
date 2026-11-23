// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The compare window's self-test. Headless: no window, no screenshot. It checks the GPU
// metric, the GPU FLIP (flip_gpu.h) and the GPU compositor against a CPU reference built from
// compare_layout.h and flip_hdr.h, that the sources' device copies are the host copies bit for
// bit, and that the exports (PNG round trips) match the CPU maps.

#include "selftest.h"

#include "borrowed_vulkan_host.h"
#include "shader_dir.h"

#include "app_util.h"
#include "orbit_camera.h"

#include "compare_layout.h"
#include "compare_project.h"
#include "compare_state.h"
#include "compare_ui.h"
#include "compositor.h"
#include "export.h"
#include "flip_gpu.h"
#include "flip_hdr.h"
#include "gpu_context.h"
#include "image_compare.h"
#include "image_io.h"
#include "image_source.h"
#include "media_source.h"
#include "metrics_gpu.h"
#include "mf_clip_writer.h"
#include "mf_decoder.h"
#include "nv12_converter.h"
#include "report.h"
#include "scene_source.h"
#include "timeline.h"
#include "video_source.h"

#include "sabercat/buffer.h"
#include "sabercat/device.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/ply_writer.h"
#include "sabercat/renderer.h"
#include "sabercat/spz_writer.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <iterator>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sabercat::compare::gui
{

namespace
{

namespace fs = std::filesystem;

using sabercat::app::BorrowedVulkanHost;
using sabercat::app::resolveShaderDir;
using ImagePtrs = std::array<const ImageData *, kMaxSources>;
using MediaList = std::vector<std::unique_ptr<ImageSource>>;

// The metric must agree with the CPU within these.
constexpr double kMaxRelativeMse = 1e-6;
constexpr double kMaxPsnrDeltaDb = 1e-4;
// Composite pixels may differ from the CPU reference by one 8-bit step (rounding, and
// the GPU's float pow and exp2).
constexpr int kMaxByteDelta = 1;
// The Fit zoom is not a power of two, so float and double coordinates can fall on
// different sides of a pixel edge. Fit views may have this fraction of such pixels.
constexpr double kFitMismatchFraction = 0.005;
// GPU FLIP against flip() on the same crops: the mean, the 99.9th percentile of the per-pixel
// difference, and the largest per-pixel difference.
constexpr double kMaxFlipMeanDelta = 5e-5;
constexpr double kMaxFlipP999 = 2e-3;
constexpr double kMaxFlipMaxDelta = 1e-2;
// HDR-FLIP against flipHdr() on the same crops: the mean and the 99.9th percentile as for LDR,
// the share of pixels whose exposure index agrees (percent, within kHdrExposureTolerance).
constexpr double kMaxHdrMeanDelta = 1e-4;
constexpr double kMinHdrExposureAgreement = 99.5;
// With this many exposures the neighbors are near-ties, so the index map is not compared (printed only).
constexpr int kHdrManyExposures = 100;
constexpr double kHdrExposureTolerance = 1e-6;
// The NVlabs HDR pair (reference.exr, test.exr): FLIP's exposure range, count and mean.
constexpr double kNvlabsHdrStart = -12.5423;
constexpr double kNvlabsHdrStop = 0.9427;
constexpr int32_t kNvlabsHdrCount = 14;
constexpr double kNvlabsHdrMean = 0.283547;

// ---- Output and bookkeeping ------------------------------------------------------

template <typename... Args>
std::string strf(const char * format, Args... args)
{
    const int size = std::snprintf(nullptr, 0, format, args...);
    std::string text(static_cast<size_t>(size) + 1, '\0');
    std::snprintf(text.data(), text.size(), format, args...);
    text.resize(static_cast<size_t>(size));
    return text;
}

struct Results
{
    std::vector<std::string> failures;

    void fail(const std::string & what)
    {
        failures.push_back(what);
        std::cout << "[selftest]   FAILED: " << what << '\n';
    }
};

// A UTF-8 string as a path (no deprecated u8path()).
fs::path pathFromUtf8(const std::string & utf8)
{
    return fs::path(std::u8string(reinterpret_cast<const char8_t *>(utf8.data()), utf8.size()));
}

// A path back to UTF-8.
std::string utf8Of(const fs::path & path)
{
    const std::u8string text = path.u8string();
    return std::string(reinterpret_cast<const char *>(text.data()), text.size());
}

std::string joinPath(const std::string & dir, const std::string & name)
{
    return utf8Of(pathFromUtf8(dir) / pathFromUtf8(name));
}

// ---- Fixtures --------------------------------------------------------------------

// Deterministic values in [0, 1).
class Lcg
{
  public:
    explicit Lcg(uint32_t seed) : m_state(seed * 2654435761u + 12345u)
    {
    }

    float next()
    {
        m_state = m_state * 1664525u + 1013904223u;
        return static_cast<float>(m_state >> 8) * (1.0f / 16777216.0f);
    }

  private:
    uint32_t m_state;
};

ImageData blankImage(uint32_t width, uint32_t height, Encoding encoding)
{
    ImageData image;
    image.width = width;
    image.height = height;
    image.encoding = encoding;
    image.bitDepth = encoding == Encoding::Linear ? 32u : 8u;
    image.rgba.assign(static_cast<size_t>(width) * height * 4, 0.0f);
    return image;
}

// A Display gradient with noise, opaque.
ImageData makeDisplay(uint32_t width, uint32_t height, uint32_t seed)
{
    ImageData image = blankImage(width, height, Encoding::Display);
    Lcg rng(seed);
    for (uint32_t y = 0; y < height; ++y)
    {
        for (uint32_t x = 0; x < width; ++x)
        {
            float * p = &image.rgba[(static_cast<size_t>(y) * width + x) * 4];
            p[0] = 0.8f * (static_cast<float>(x) + 0.5f) / static_cast<float>(width) + 0.2f * rng.next();
            p[1] = 0.8f * (static_cast<float>(y) + 0.5f) / static_cast<float>(height) + 0.2f * rng.next();
            p[2] = rng.next();
            p[3] = 1.0f;
        }
    }
    return image;
}

// A Linear (EXR-like) image with values up to 5, opaque.
ImageData makeLinear(uint32_t width, uint32_t height, uint32_t seed)
{
    ImageData image = blankImage(width, height, Encoding::Linear);
    Lcg rng(seed);
    for (size_t i = 0; i < static_cast<size_t>(width) * height; ++i)
    {
        float * p = &image.rgba[i * 4];
        p[0] = 5.0f * rng.next();
        p[1] = 5.0f * rng.next();
        p[2] = 5.0f * rng.next();
        p[3] = 1.0f;
    }
    return image;
}

// A premultiplied capture: rgb = color * alpha, with some fully transparent pixels.
ImageData makePremultiplied(uint32_t width, uint32_t height, uint32_t seed)
{
    ImageData image = blankImage(width, height, Encoding::Display);
    image.hasAlpha = true;
    image.premultiplied = true;
    Lcg rng(seed);
    for (size_t i = 0; i < static_cast<size_t>(width) * height; ++i)
    {
        float * p = &image.rgba[i * 4];
        const float alpha = (i % 7 == 0) ? 0.0f : rng.next();
        p[0] = rng.next() * alpha;
        p[1] = rng.next() * alpha;
        p[2] = rng.next() * alpha;
        p[3] = alpha;
    }
    return image;
}

// A copy of the image with each rgb sample moved by up to +-0.1.
ImageData perturbed(const ImageData & source, uint32_t seed)
{
    ImageData image = source;
    Lcg rng(seed);
    for (size_t i = 0; i < image.rgba.size(); i += 4)
    {
        for (size_t c = 0; c < 3; ++c)
        {
            image.rgba[i + c] = std::clamp(image.rgba[i + c] + 0.2f * (rng.next() - 0.5f), 0.0f, 1.0f);
        }
    }
    return image;
}

struct Case
{
    std::string name;
    std::vector<ImageData> images;
    // The composite settings of each source (media is set when the case runs).
    std::vector<CompositeSource> looks;
    bool centered = false;
    bool grid = false;
    uint32_t windowWidth = 120;
    uint32_t windowHeight = 80;
};

Case makeCase(const std::string & name, std::vector<ImageData> images, bool centered, bool grid = false)
{
    Case c;
    c.name = name;
    c.looks.resize(images.size());
    c.images = std::move(images);
    c.centered = centered;
    c.grid = grid;
    return c;
}

std::vector<Case> buildCases()
{
    std::vector<Case> cases;

    {
        ImageData a = makeDisplay(37, 23, 1);
        ImageData b = perturbed(a, 2);
        cases.push_back(makeCase("same-size", {a, b}, false));
    }
    {
        ImageData a = makeDisplay(37, 23, 3);
        cases.push_back(makeCase("identical", {a, a}, false));
    }
    {
        ImageData a = makeDisplay(40, 30, 4);
        ImageData b = makeDisplay(25, 45, 5);
        cases.push_back(makeCase("sizes-topleft", {a, b}, false));
        cases.push_back(makeCase("sizes-centered", {a, b}, true));
    }
    {
        ImageData a = makePremultiplied(33, 21, 6);
        ImageData b = makeDisplay(33, 21, 7);
        Case c = makeCase("premultiplied", {a, b}, false);
        c.looks[0].background = RgbaColor{0.5f, 0.2f, 0.7f, 1.0f};
        cases.push_back(std::move(c));
    }
    {
        ImageData a = makeLinear(31, 27, 8);
        ImageData b = makeDisplay(31, 27, 9);
        Case c = makeCase("linear-display", {a, b}, false);
        c.looks[0].exposureStops = -1.0f;
        c.looks[0].toneMapper = ToneMapper::Hable;
        c.looks[1].toneMapper = ToneMapper::Reinhard;
        cases.push_back(std::move(c));
    }
    {
        ImageData a = makeDisplay(1, 1, 10);
        ImageData b = perturbed(a, 11);
        cases.push_back(makeCase("one-pixel", {a, b}, false));
    }
    {
        ImageData a = makeDisplay(300, 3, 12);
        ImageData b = perturbed(a, 13);
        cases.push_back(makeCase("wide", {a, b}, false));
    }
    {
        ImageData a = makeDisplay(3, 300, 14);
        ImageData b = perturbed(a, 15);
        cases.push_back(makeCase("tall", {a, b}, false));
    }
    {
        std::vector<ImageData> images;
        images.push_back(makeDisplay(33, 21, 16));
        images.push_back(makeLinear(40, 30, 17));
        images.push_back(makeDisplay(25, 45, 18));
        images.push_back(makeDisplay(37, 23, 19));
        cases.push_back(makeCase("four-sources-grid", std::move(images), true, true));
    }
    {
        std::vector<ImageData> images;
        images.push_back(makeDisplay(30, 20, 20));
        images.push_back(makeDisplay(22, 26, 21));
        images.push_back(makePremultiplied(30, 20, 22));
        Case c = makeCase("three-sources-row", std::move(images), false, false);
        c.looks[2].background = RgbaColor{0.1f, 0.6f, 0.3f, 1.0f};
        cases.push_back(std::move(c));
    }
    return cases;
}

// ---- Layout and composite inputs -------------------------------------------------

LayoutInput makeLayout(const Case & c)
{
    LayoutInput layout;
    for (const ImageData & image : c.images)
    {
        LayoutSource geometry;
        geometry.width = static_cast<int32_t>(image.width);
        geometry.height = static_cast<int32_t>(image.height);
        geometry.encoding = image.encoding;
        geometry.visible = true;
        if (!addSource(layout, geometry))
        {
            throw std::logic_error("addSource refused a source of " + c.name);
        }
    }
    if (c.centered)
    {
        anchorCentered(layout);
    }
    else
    {
        anchorTopLeft(layout);
    }
    layout.view.sourceA = 0;
    layout.view.sourceB = 1;
    layout.view.abIndex = 0;
    layout.view.mode = ViewMode::SplitScreen;
    layout.view.arrangement = c.grid ? PaneArrangement::Grid2x2 : PaneArrangement::Row;
    layout.view.zoomMode = ZoomMode::Fit;
    layout.windowWidth = static_cast<int32_t>(c.windowWidth);
    layout.windowHeight = static_cast<int32_t>(c.windowHeight);
    return layout;
}

CompositeInput makeInput(const Case & c, const LayoutInput & layout, const MediaList & media)
{
    CompositeInput input;
    input.layout = layout;
    for (size_t i = 0; i < c.images.size(); ++i)
    {
        input.sources[i] = c.looks[i];
        input.sources[i].media = media[i].get();
    }
    return input;
}

ImagePtrs imagePointers(const Case & c)
{
    ImagePtrs pointers{};
    for (size_t i = 0; i < c.images.size(); ++i)
    {
        pointers[i] = &c.images[i];
    }
    return pointers;
}

// ---- CPU reference ---------------------------------------------------------------

// Byte of a display value, with the compositor's rounding (floor(v * 255 + 0.5)).
uint8_t byteOf(float v)
{
    if (!(v > 0.0f))
    {
        return 0;
    }
    if (v >= 1.0f)
    {
        return 255;
    }
    return static_cast<uint8_t>(std::floor(v * 255.0f + 0.5f));
}

float clampUnit(float v)
{
    return v > 0.0f ? std::min(v, 1.0f) : 0.0f;
}

int32_t floorPixel(double v)
{
    return static_cast<int32_t>(std::clamp(std::floor(v), -1073741824.0, 1073741824.0));
}

// The display color of a sample of an image as a pane shows it: premultiplied rgb is
// composited over the source's background, then Display is clamped and Linear goes
// through displayTransform(). (x, y) are local pixel coordinates.
void displayColor(const ImageData & image, const CompositeSource & look, int32_t x, int32_t y, float out[3])
{
    const float * s = &image.rgba[(static_cast<size_t>(y) * image.width + static_cast<size_t>(x)) * 4];
    const float holdback = image.premultiplied ? 1.0f - clampUnit(s[3]) : 0.0f;
    const float background[3] = {look.background.r, look.background.g, look.background.b};
    for (size_t c = 0; c < 3; ++c)
    {
        const float value = s[c] + holdback * background[c];
        out[c] = image.encoding == Encoding::Linear
            ? displayTransform(value, look.toneMapper, look.exposureStops)
            : clampUnit(value);
    }
}

// Bilinear display color at a continuous local coordinate. The taps are clamped to the
// source rectangle.
void bilinearColor(const ImageData & image, const CompositeSource & look, float localX, float localY, float out[3])
{
    const float tx = localX - 0.5f;
    const float ty = localY - 0.5f;
    const float fx0 = std::floor(tx);
    const float fy0 = std::floor(ty);
    const float wx = tx - fx0;
    const float wy = ty - fy0;
    const int32_t maxX = static_cast<int32_t>(image.width) - 1;
    const int32_t maxY = static_cast<int32_t>(image.height) - 1;
    const int32_t x0 = static_cast<int32_t>(fx0);
    const int32_t y0 = static_cast<int32_t>(fy0);
    const int32_t xa = std::clamp(x0, 0, maxX);
    const int32_t xb = std::clamp(x0 + 1, 0, maxX);
    const int32_t ya = std::clamp(y0, 0, maxY);
    const int32_t yb = std::clamp(y0 + 1, 0, maxY);
    float c00[3];
    float c10[3];
    float c01[3];
    float c11[3];
    displayColor(image, look, xa, ya, c00);
    displayColor(image, look, xb, ya, c10);
    displayColor(image, look, xa, yb, c01);
    displayColor(image, look, xb, yb, c11);
    for (size_t c = 0; c < 3; ++c)
    {
        const float top = c00[c] + (c10[c] - c00[c]) * wx;
        const float bottom = c01[c] + (c11[c] - c01[c]) * wx;
        out[c] = top + (bottom - top) * wy;
    }
}

// The layout a composite shows: the SingleScreen form for the single-pane modes.
LayoutInput shownLayout(const CompositeInput & input)
{
    LayoutInput shown = input.layout;
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
        shown.view.abIndex = shown.view.sourceA;
        break;
    case CompositeMode::Flicker:
        shown.view.mode = ViewMode::SingleScreen;
        shown.view.abIndex = input.flickerShowB ? shown.view.sourceB : shown.view.sourceA;
        break;
    case CompositeMode::Depth:
    case CompositeMode::Alpha:
        shown.view.mode = ViewMode::SingleScreen;
        shown.view.abIndex = input.mapUseB ? shown.view.sourceB : shown.view.sourceA;
        break;
    }
    return shown;
}

// ---- Map modes on host copies (the rules of compositor.h) --------------------------

// Whether the host copy carries one depth sample per pixel.
bool hasDepthSamples(const ImageData & image)
{
    return image.depth.size() == static_cast<size_t>(image.width) * image.height;
}

// Coverage of a sample: its color alpha, NaN as 0, clamped to [0, 1].
float referenceCoverage(const ImageData & image, size_t index)
{
    return clampUnit(image.rgba[index * 4 + 3]);
}

// The magma color of a map value in [0, 1].
void magmaBytes(float t, uint8_t * out)
{
    const Rgb8Image color = colorizeMagma(std::vector<float>{t}, 1, 1);
    out[0] = color.rgb[0];
    out[1] = color.rgb[1];
    out[2] = color.rgb[2];
}

// The hatch of a no-data pixel: a gray at the view pixel (display value, not magma).
void hatchBytes(int32_t viewX, int32_t viewY, uint8_t * out)
{
    const uint8_t value = byteOf(hatchValue(viewX, viewY));
    out[0] = value;
    out[1] = value;
    out[2] = value;
}

// The other source of a padding pixel: the other of the A/B pair when visible, else the
// first other visible source, else -1. Written from the rule, not from the compositor.
int32_t refCounterpart(const LayoutInput & layout, int32_t source, int32_t a, int32_t b)
{
    const auto visible = [&layout](int32_t k)
    {
        return k >= 0 && k < layout.sourceCount && k < kMaxSources && layout.sources[static_cast<size_t>(k)].visible;
    };
    if (source == a && visible(b) && b != source)
    {
        return b;
    }
    if (source == b && visible(a) && a != source)
    {
        return a;
    }
    for (int32_t k = 0; k < layout.sourceCount && k < kMaxSources; ++k)
    {
        if (k != source && visible(k))
        {
            return k;
        }
    }
    return -1;
}

// The CPU rendering of a composite view, as RGB bytes, top row first. It follows the rules
// of compositor.h, using compare_layout.h for the geometry.
std::vector<uint8_t> referenceComposite(const CompositeInput & input, const ImagePtrs & images, uint32_t width,
                                        uint32_t height)
{
    const LayoutInput shown = shownLayout(input);
    const PixelRect extent = paddedExtent(shown);
    const double wipe = clampWipePosition(shown, shown.view.wipePosition);
    const int32_t a = input.layout.view.sourceA;
    const int32_t b = input.layout.view.sourceB;
    const uint8_t backgroundByte = byteOf(kCompositeBackground);

    Image displayA;
    Image displayB;
    if (input.mode == CompositeMode::Diff)
    {
        displayA = displayRgb(*images[static_cast<size_t>(a)]);
        displayB = displayRgb(*images[static_cast<size_t>(b)]);
    }

    std::vector<uint8_t> out(static_cast<size_t>(width) * height * 3, backgroundByte);
    for (uint32_t py = 0; py < height; ++py)
    {
        for (uint32_t px = 0; px < width; ++px)
        {
            uint8_t * dst = &out[(static_cast<size_t>(py) * width + px) * 3];
            const ImageHit hit = windowToImage(shown, static_cast<int32_t>(px), static_cast<int32_t>(py));
            const int32_t pixelX = floorPixel(hit.imageX);
            const int32_t pixelY = floorPixel(hit.imageY);
            if (hit.pane < 0 || !rectContains(extent, pixelX, pixelY))
            {
                continue; // background, already filled
            }

            if (input.mode == CompositeMode::Diff)
            {
                const LayoutSource & ga = shown.sources[static_cast<size_t>(a)];
                const LayoutSource & gb = shown.sources[static_cast<size_t>(b)];
                float value = mapValueAt(0.0f, PixelKind::Padding);
                if (rectContains(sourceRect(ga), pixelX, pixelY) && rectContains(sourceRect(gb), pixelX, pixelY))
                {
                    const float * pa = &displayA.rgb[(static_cast<size_t>(pixelY - ga.offsetY) * displayA.width
                                                      + static_cast<size_t>(pixelX - ga.offsetX)) * 3];
                    const float * pb = &displayB.rgb[(static_cast<size_t>(pixelY - gb.offsetY) * displayB.width
                                                      + static_cast<size_t>(pixelX - gb.offsetX)) * 3];
                    const float mean = (std::fabs(pa[0] - pb[0]) + std::fabs(pa[1] - pb[1]) + std::fabs(pa[2] - pb[2]))
                        / 3.0f;
                    value = std::min(1.0f, mean * input.diffGain);
                }
                magmaBytes(value, dst);
                continue;
            }

            if (input.mode == CompositeMode::Depth || input.mode == CompositeMode::Alpha)
            {
                // The chosen source (B when mapUseB), one pane over the view. Padding is hatched.
                const int32_t chosen = input.mapUseB ? b : a;
                const LayoutSource & g = shown.sources[static_cast<size_t>(chosen)];
                const ImageData & image = *images[static_cast<size_t>(chosen)];
                const bool inside = rectContains(sourceRect(g), pixelX, pixelY);
                const size_t index = inside
                    ? static_cast<size_t>(pixelY - g.offsetY) * image.width + static_cast<size_t>(pixelX - g.offsetX)
                    : 0;
                if (input.mode == CompositeMode::Depth)
                {
                    if (inside && hasDepthSamples(image) && isValidDepth(image.depth[index]))
                    {
                        magmaBytes(depthMapT(image.depth[index], input.depthMin, input.depthMax), dst);
                    }
                    else
                    {
                        hatchBytes(static_cast<int32_t>(px), static_cast<int32_t>(py), dst);
                    }
                }
                else if (inside)
                {
                    const uint8_t gray = byteOf(referenceCoverage(image, index));
                    dst[0] = gray;
                    dst[1] = gray;
                    dst[2] = gray;
                }
                else
                {
                    hatchBytes(static_cast<int32_t>(px), static_cast<int32_t>(py), dst);
                }
                continue;
            }

            if (input.mode == CompositeMode::DepthDiff || input.mode == CompositeMode::AlphaDiff)
            {
                // A and B, both panes over the view. A pixel outside either rectangle is padding.
                const LayoutSource & ga = shown.sources[static_cast<size_t>(a)];
                const LayoutSource & gb = shown.sources[static_cast<size_t>(b)];
                const ImageData & ia = *images[static_cast<size_t>(a)];
                const ImageData & ib = *images[static_cast<size_t>(b)];
                if (!rectContains(sourceRect(ga), pixelX, pixelY) || !rectContains(sourceRect(gb), pixelX, pixelY))
                {
                    magmaBytes(1.0f, dst);
                    continue;
                }
                const size_t ta = static_cast<size_t>(pixelY - ga.offsetY) * ia.width
                    + static_cast<size_t>(pixelX - ga.offsetX);
                const size_t tb = static_cast<size_t>(pixelY - gb.offsetY) * ib.width
                    + static_cast<size_t>(pixelX - gb.offsetX);
                if (input.mode == CompositeMode::AlphaDiff)
                {
                    const float diff = std::fabs(referenceCoverage(ia, ta) - referenceCoverage(ib, tb));
                    magmaBytes(std::min(1.0f, diff * input.diffGain), dst);
                    continue;
                }
                const bool validA = hasDepthSamples(ia) && isValidDepth(ia.depth[ta]);
                const bool validB = hasDepthSamples(ib) && isValidDepth(ib.depth[tb]);
                if (validA && validB)
                {
                    magmaBytes(depthDiffT(ia.depth[ta], ib.depth[tb], input.depthMin, input.depthMax), dst);
                }
                else if (validA || validB)
                {
                    magmaBytes(1.0f, dst);
                }
                else
                {
                    hatchBytes(static_cast<int32_t>(px), static_cast<int32_t>(py), dst);
                }
                continue;
            }

            int32_t source = hit.source;
            if (input.mode == CompositeMode::Wipe)
            {
                const double coordinate = shown.view.wipeOrientation == WipeOrientation::Vertical ? hit.imageX : hit.imageY;
                source = wipeSideAt(coordinate, wipe) == WipeSide::A ? a : b;
            }
            if (source < 0 || source >= kMaxSources)
            {
                continue;
            }

            float rgb[3] = {0.0f, 0.0f, 0.0f};
            const LayoutSource & g = shown.sources[static_cast<size_t>(source)];
            const ImageData & image = *images[static_cast<size_t>(source)];
            const CompositeSource & look = input.sources[static_cast<size_t>(source)];
            if (rectContains(sourceRect(g), pixelX, pixelY))
            {
                if (input.nearest)
                {
                    displayColor(image, look, pixelX - g.offsetX, pixelY - g.offsetY, rgb);
                }
                else
                {
                    bilinearColor(image, look, static_cast<float>(hit.imageX - g.offsetX),
                                  static_cast<float>(hit.imageY - g.offsetY), rgb);
                }
            }
            else
            {
                const int32_t counterpart = refCounterpart(shown, source, a, b);
                if (counterpart < 0)
                {
                    continue;
                }
                const LayoutSource & gc = shown.sources[static_cast<size_t>(counterpart)];
                if (!rectContains(sourceRect(gc), pixelX, pixelY))
                {
                    continue;
                }
                displayColor(*images[static_cast<size_t>(counterpart)], input.sources[static_cast<size_t>(counterpart)],
                             pixelX - gc.offsetX, pixelY - gc.offsetY, rgb);
                for (float & channel : rgb)
                {
                    channel = 1.0f - channel;
                }
            }
            dst[0] = byteOf(rgb[0]);
            dst[1] = byteOf(rgb[1]);
            dst[2] = byteOf(rgb[2]);
        }
    }
    return out;
}

// ---- Checks ----------------------------------------------------------------------

// Compares GPU bytes (RGBA) with reference bytes (RGB). Exact views allow no pixel beyond
// kMaxByteDelta; approximate views allow kFitMismatchFraction of the pixels.
void compareViewBytes(const std::string & label, const std::vector<uint8_t> & gpu, const std::vector<uint8_t> & ref,
                      uint32_t width, uint32_t height, bool exact, Results & results)
{
    const size_t pixels = static_cast<size_t>(width) * height;
    if (gpu.size() != pixels * 4 || ref.size() != pixels * 3)
    {
        results.fail(label + ": readback size " + std::to_string(gpu.size()) + " bytes");
        return;
    }
    size_t mismatches = 0;
    size_t alphaBad = 0;
    int maxDelta = 0;
    size_t first = pixels;
    for (size_t p = 0; p < pixels; ++p)
    {
        int delta = 0;
        for (size_t c = 0; c < 3; ++c)
        {
            delta = std::max(delta, std::abs(static_cast<int>(gpu[4 * p + c]) - static_cast<int>(ref[3 * p + c])));
        }
        if (gpu[4 * p + 3] != 255)
        {
            ++alphaBad;
        }
        maxDelta = std::max(maxDelta, delta);
        if (delta > kMaxByteDelta)
        {
            if (first == pixels)
            {
                first = p;
            }
            ++mismatches;
        }
    }
    const size_t allowed = exact ? 0 : static_cast<size_t>(std::floor(static_cast<double>(pixels) * kFitMismatchFraction));
    std::cout << strf("[selftest] %s pixels=%zu mismatches=%zu allowed=%zu maxDelta=%d\n", label.c_str(), pixels,
                      mismatches, allowed, maxDelta);
    if (mismatches > allowed)
    {
        std::string detail;
        if (first < pixels)
        {
            const size_t x = first % width;
            const size_t y = first / width;
            detail = strf(" first at (%zu,%zu) gpu=(%u,%u,%u) ref=(%u,%u,%u)", x, y,
                          static_cast<unsigned>(gpu[4 * first]), static_cast<unsigned>(gpu[4 * first + 1]),
                          static_cast<unsigned>(gpu[4 * first + 2]), static_cast<unsigned>(ref[3 * first]),
                          static_cast<unsigned>(ref[3 * first + 1]), static_cast<unsigned>(ref[3 * first + 2]));
        }
        results.fail(label + ": " + std::to_string(mismatches) + " pixels differ by more than 1/255," + detail);
    }
    if (alphaBad != 0)
    {
        results.fail(label + ": " + std::to_string(alphaBad) + " pixels with alpha != 255");
    }
}

// Compares a PNG's pixels with RGB bytes. The PNG is lossless, so every byte must match.
void checkPng(const std::string & label, const std::string & path, const std::vector<uint8_t> & expected,
              uint32_t width, uint32_t height, Results & results)
{
    const ImageData loaded = loadImage(path);
    const size_t pixels = static_cast<size_t>(width) * height;
    if (loaded.width != width || loaded.height != height || loaded.rgba.size() != pixels * 4)
    {
        results.fail(label + ": PNG " + path + " has the wrong size");
        return;
    }
    size_t bad = 0;
    for (size_t p = 0; p < pixels; ++p)
    {
        for (size_t c = 0; c < 3; ++c)
        {
            if (byteOf(loaded.rgba[4 * p + c]) != expected[3 * p + c])
            {
                ++bad;
            }
        }
    }
    std::cout << strf("[selftest] %s png round trip: %zu of %zu samples differ\n", label.c_str(), bad, pixels * 3);
    if (bad != 0)
    {
        results.fail(label + ": PNG round trip differs in " + std::to_string(bad) + " samples");
    }
}

// The host copy (cropped to the rectangle) of an image.
ImageData cropImage(const ImageData & source, int32_t x, int32_t y, int32_t width, int32_t height)
{
    ImageData out;
    out.width = static_cast<uint32_t>(width);
    out.height = static_cast<uint32_t>(height);
    out.encoding = source.encoding;
    out.bitDepth = source.bitDepth;
    out.hasAlpha = source.hasAlpha;
    out.premultiplied = source.premultiplied;
    out.rgba.resize(static_cast<size_t>(width) * height * 4);
    for (int32_t row = 0; row < height; ++row)
    {
        const float * from = &source.rgba[(static_cast<size_t>(y + row) * source.width + static_cast<size_t>(x)) * 4];
        std::memcpy(&out.rgba[static_cast<size_t>(row) * width * 4], from, static_cast<size_t>(width) * 4 * sizeof(float));
    }
    return out;
}

void checkMetrics(const Case & c, const LayoutInput & layout, const MediaList & media, MetricsGpu & metrics,
                  Results & results, std::optional<MetricsResult> & gpuOut)
{
    const PixelRect overlap = overlapRect(layout, 0, 1);
    const int32_t ax = layout.sources[0].offsetX;
    const int32_t ay = layout.sources[0].offsetY;
    const int32_t bx = layout.sources[1].offsetX;
    const int32_t by = layout.sources[1].offsetY;
    if (overlap.width <= 0 || overlap.height <= 0)
    {
        results.fail(c.name + ": the pair has no overlap");
        return;
    }

    const std::optional<MetricsResult> gpu = metrics.compute(*media[0], ax, ay, *media[1], bx, by, overlap);
    if (!gpu)
    {
        results.fail(c.name + ": the GPU metric returned nothing");
        return;
    }
    const ImageData cropA = cropImage(c.images[0], overlap.x - ax, overlap.y - ay, overlap.width, overlap.height);
    const ImageData cropB = cropImage(c.images[1], overlap.x - bx, overlap.y - by, overlap.width, overlap.height);
    const PsnrResult cpu = psnr(displayRgb(cropA), displayRgb(cropB));

    bool mseOk = false;
    double rel = 0.0;
    if (cpu.mse == 0.0)
    {
        rel = gpu->mse;
        mseOk = gpu->mse == 0.0;
    }
    else
    {
        rel = std::fabs(gpu->mse - cpu.mse) / cpu.mse;
        mseOk = rel <= kMaxRelativeMse;
    }
    bool psnrOk = false;
    double delta = 0.0;
    if (std::isinf(cpu.psnrDb) || std::isinf(gpu->psnrDb))
    {
        psnrOk = std::isinf(cpu.psnrDb) && std::isinf(gpu->psnrDb) && (cpu.psnrDb > 0.0) == (gpu->psnrDb > 0.0);
    }
    else
    {
        delta = std::fabs(gpu->psnrDb - cpu.psnrDb);
        psnrOk = delta <= kMaxPsnrDeltaDb;
    }

    std::cout << strf("[selftest] %s metrics gpu mse=%.9g cpu mse=%.9g rel=%.3g gpu psnr=%.6f cpu psnr=%.6f delta=%.3g "
                      "overlap=%dx%d pixels=%llu\n",
                      c.name.c_str(), gpu->mse, cpu.mse, rel, gpu->psnrDb, cpu.psnrDb, delta, overlap.width,
                      overlap.height, static_cast<unsigned long long>(gpu->pixelCount));
    if (!mseOk)
    {
        results.fail(c.name + ": metric MSE differs from the CPU by " + strf("%.3g", rel));
    }
    if (!psnrOk)
    {
        results.fail(c.name + ": metric PSNR differs from the CPU by " + strf("%.3g dB", delta));
    }
    if (gpu->pixelCount != static_cast<uint64_t>(overlap.width) * static_cast<uint64_t>(overlap.height))
    {
        results.fail(c.name + ": metric pixel count is wrong");
    }
    gpuOut = gpu;
}

void checkSourceReadbacks(const Case & c, GpuContext & context, const MediaList & media, Results & results)
{
    for (size_t i = 0; i < c.images.size(); ++i)
    {
        const ImageData & host = c.images[i];
        const size_t floats = host.rgba.size();
        std::vector<float> device(floats, -1.0f);
        context.readBuffer(media[i]->color(), device.data(), floats * sizeof(float));
        const bool same = std::memcmp(device.data(), host.rgba.data(), floats * sizeof(float)) == 0;

        size_t badPixels = 0;
        const uint32_t w = host.width;
        for (uint32_t y = 0; y < host.height; ++y)
        {
            for (uint32_t x = 0; x < w; ++x)
            {
                float px[4];
                const size_t index = (static_cast<size_t>(y) * w + x) * 4;
                if (!media[i]->readPixel(static_cast<int32_t>(x), static_cast<int32_t>(y), px)
                    || std::memcmp(px, &host.rgba[index], sizeof(px)) != 0)
                {
                    ++badPixels;
                }
            }
        }
        std::cout << strf("[selftest] %s source %zu readback bit-exact=%s readPixel mismatches=%zu\n", c.name.c_str(),
                          i, same ? "yes" : "no", badPixels);
        if (!same)
        {
            results.fail(c.name + ": source " + std::to_string(i) + " device copy differs from the host copy");
        }
        if (badPixels != 0)
        {
            results.fail(c.name + ": source " + std::to_string(i) + " readPixel differs from the host copy");
        }
    }
}

// The composite views of a case. Exact views use zoom 100, 200 or 400 with pans on whole
// or half pixels, so every coordinate is exact in float. The one Fit view is approximate.
struct View
{
    std::string label;
    CompositeInput input;
    bool exact = true;
};

std::vector<View> makeViews(const CompositeInput & base)
{
    std::vector<View> views;
    const auto add = [&views, &base](const char * label, CompositeMode mode, bool exact,
                                     const std::function<void(CompositeInput &)> & tune)
    {
        View view;
        view.label = label;
        view.exact = exact;
        view.input = base;
        view.input.mode = mode;
        tune(view.input);
        views.push_back(std::move(view));
    };

    add("split-fit", CompositeMode::Panes, false, [](CompositeInput &) {});
    add("split-100", CompositeMode::Panes, true,
        [](CompositeInput & in)
        {
            setZoomPercent(in.layout, 100.0);
            centerView(in.layout);
            setPan(in.layout, in.layout.view.panX + 0.5, in.layout.view.panY);
        });
    add("split-200-bilinear", CompositeMode::Panes, true,
        [](CompositeInput & in)
        {
            setZoomPercent(in.layout, 200.0);
            centerView(in.layout);
            in.nearest = false;
        });
    add("split-400", CompositeMode::Panes, true,
        [](CompositeInput & in)
        {
            setZoomPercent(in.layout, 400.0);
            centerView(in.layout);
            setPan(in.layout, in.layout.view.panX, in.layout.view.panY + 0.5);
        });
    if (base.layout.sourceCount >= 2)
    {
        add("single-b-100", CompositeMode::Panes, true,
            [](CompositeInput & in)
            {
                in.layout.view.mode = ViewMode::SingleScreen;
                in.layout.view.abIndex = 1;
                setZoomPercent(in.layout, 100.0);
                centerView(in.layout);
            });
    }
    add("wipe-v-100", CompositeMode::Wipe, true,
        [](CompositeInput & in)
        {
            in.layout.view.wipeOrientation = WipeOrientation::Vertical;
            setWipeFraction(in.layout, 0.5);
            setZoomPercent(in.layout, 100.0);
            centerView(in.layout);
        });
    add("wipe-h-200", CompositeMode::Wipe, true,
        [](CompositeInput & in)
        {
            in.layout.view.wipeOrientation = WipeOrientation::Horizontal;
            setWipeFraction(in.layout, 0.375);
            setZoomPercent(in.layout, 200.0);
            centerView(in.layout);
        });
    add("flicker-a-100", CompositeMode::Flicker, true,
        [](CompositeInput & in)
        {
            in.flickerShowB = false;
            setZoomPercent(in.layout, 100.0);
            centerView(in.layout);
        });
    add("flicker-b-200", CompositeMode::Flicker, true,
        [](CompositeInput & in)
        {
            in.flickerShowB = true;
            setZoomPercent(in.layout, 200.0);
            centerView(in.layout);
        });
    add("diff-100", CompositeMode::Diff, true,
        [](CompositeInput & in)
        {
            in.diffGain = 1.0f;
            setZoomPercent(in.layout, 100.0);
            centerView(in.layout);
        });
    add("diff-200", CompositeMode::Diff, true,
        [](CompositeInput & in)
        {
            in.diffGain = 4.0f;
            setZoomPercent(in.layout, 200.0);
            centerView(in.layout);
            setPan(in.layout, in.layout.view.panX + 0.5, in.layout.view.panY);
        });
    add("diff-400", CompositeMode::Diff, true,
        [](CompositeInput & in)
        {
            in.diffGain = 0.5f;
            setZoomPercent(in.layout, 400.0);
            centerView(in.layout);
        });
    return views;
}

void checkComposites(const Case & c, const CompositeInput & base, Compositor & compositor, Results & results)
{
    const ImagePtrs images = imagePointers(c);
    for (const View & view : makeViews(base))
    {
        const std::string label = c.name + " " + view.label;
        const uint32_t width = static_cast<uint32_t>(view.input.layout.windowWidth);
        const uint32_t height = static_cast<uint32_t>(view.input.layout.windowHeight);
        const std::vector<uint8_t> ref = referenceComposite(view.input, images, width, height);
        compositor.resize(width, height);
        try
        {
            compositor.render(view.input);
        }
        catch (const std::exception & e)
        {
            results.fail(label + ": render threw: " + e.what());
            continue;
        }
        std::vector<uint8_t> gpu;
        compositor.readback(gpu);
        compareViewBytes(label, gpu, ref, width, height, view.exact, results);
    }
}

// Writes the split-100 view as a PNG (alpha dropped) and reads it back.
void checkViewExport(const Case & c, const CompositeInput & base, Compositor & compositor, const std::string & outDir,
                     Results & results)
{
    CompositeInput input = base;
    input.mode = CompositeMode::Panes;
    setZoomPercent(input.layout, 100.0);
    centerView(input.layout);
    const uint32_t width = static_cast<uint32_t>(input.layout.windowWidth);
    const uint32_t height = static_cast<uint32_t>(input.layout.windowHeight);
    compositor.resize(width, height);
    compositor.render(input);
    std::vector<uint8_t> gpu;
    compositor.readback(gpu);

    const std::string path = joinPath(outDir, c.name + "-view.png");
    saveRgbaPng(path, gpu.data(), width, height);
    std::vector<uint8_t> rgb(static_cast<size_t>(width) * height * 3);
    for (size_t p = 0; p < rgb.size() / 3; ++p)
    {
        rgb[3 * p] = gpu[4 * p];
        rgb[3 * p + 1] = gpu[4 * p + 1];
        rgb[3 * p + 2] = gpu[4 * p + 2];
    }
    checkPng(c.name + " view export", path, rgb, width, height, results);
}

// Alpha 255 for every pixel of RGB bytes.
std::vector<uint8_t> opaqueRgba(const std::vector<uint8_t> & rgb)
{
    std::vector<uint8_t> rgba(rgb.size() / 3 * 4, 255);
    for (size_t p = 0; p < rgb.size() / 3; ++p)
    {
        rgba[4 * p] = rgb[3 * p];
        rgba[4 * p + 1] = rgb[3 * p + 1];
        rgba[4 * p + 2] = rgb[3 * p + 2];
    }
    return rgba;
}

// The export checks: the diff map and the padded pane images, each against the CPU
// reference (the diff view and the flicker view over the padded extent at 100%) and
// through a PNG round trip. Then the report files.
// The export scene of a case: the A/B pair is sources 0 and 1, with the layout's geometry.
ExportScene caseExportScene(const Case & c, const LayoutInput & layout)
{
    ExportScene scene;
    scene.sourceA = 0;
    scene.sourceB = 1;
    scene.sources.resize(c.images.size());
    for (size_t i = 0; i < c.images.size(); ++i)
    {
        ExportSource & source = scene.sources[i];
        source.image = &c.images[i];
        source.background = c.looks[i].background;
        source.geometry = layout.sources[i];
        source.counterpart = refCounterpart(layout, static_cast<int32_t>(i), 0, 1);
        source.exposureStops = c.looks[i].exposureStops;
        source.toneMapper = c.looks[i].toneMapper;
    }
    return scene;
}

void checkExports(const Case & c, const LayoutInput & layout, const std::optional<MetricsResult> & metrics,
                  const std::string & outDir, Results & results)
{
    const ImagePtrs images = imagePointers(c);
    const ExportScene scene = caseExportScene(c, layout);

    const PixelRect extent = exportExtent(scene);
    const uint32_t width = static_cast<uint32_t>(extent.width);
    const uint32_t height = static_cast<uint32_t>(extent.height);

    // The window over exactly the padded extent, at 100%, for the reference views.
    CompositeInput over;
    over.layout = layout;
    for (size_t i = 0; i < c.images.size(); ++i)
    {
        over.sources[i] = c.looks[i];
    }
    over.layout.windowWidth = extent.width;
    over.layout.windowHeight = extent.height;
    setZoomPercent(over.layout, 100.0);
    setPan(over.layout, extent.x + extent.width * 0.5, extent.y + extent.height * 0.5);

    const float gain = 2.0f;
    const Rgb8Image diff = diffMapImage(scene, gain);
    CompositeInput diffView = over;
    diffView.mode = CompositeMode::Diff;
    diffView.diffGain = gain;
    compareViewBytes(c.name + " diff map export", opaqueRgba(diff.rgb),
                     referenceComposite(diffView, images, width, height), width, height, true, results);
    const std::string diffPath = joinPath(outDir, c.name + "-diff.png");
    savePngRgb8(diffPath, diff);
    checkPng(c.name + " diff map", diffPath, diff.rgb, width, height, results);

    for (size_t i = 0; i < c.images.size(); ++i)
    {
        const Rgb8Image pane = paddedPaneImage(scene, static_cast<int32_t>(i));
        const std::string label = c.name + " pane " + std::to_string(i);
        const std::string path = joinPath(outDir, c.name + "-pane" + std::to_string(i) + ".png");
        savePngRgb8(path, pane);
        checkPng(label, path, pane.rgb, width, height, results);

        // The pane of a pair equals the flicker view of the same source.
        if (c.images.size() == 2)
        {
            CompositeInput flicker = over;
            flicker.mode = CompositeMode::Flicker;
            flicker.flickerShowB = i == 1;
            const std::vector<uint8_t> ref = referenceComposite(flicker, images, width, height);
            compareViewBytes(label + " vs flicker view", opaqueRgba(pane.rgb), ref, width, height, true, results);
        }
    }

    if (metrics)
    {
        ReportInput input;
        input.pathA = c.name + "-a";
        input.pathB = c.name + "-b";
        input.a = &c.images[0];
        input.b = &c.images[1];
        input.aOffsetX = layout.sources[0].offsetX;
        input.aOffsetY = layout.sources[0].offsetY;
        input.overlap = metrics->overlap;
        input.mse = metrics->mse;
        input.psnrDb = metrics->psnrDb;
        input.anchor = c.centered ? "centered" : "topLeft";
        const CompareReport report = makeReport(input);
        const std::string jsonPath = joinPath(outDir, c.name + "-report.json");
        const std::string csvPath = joinPath(outDir, c.name + "-report.csv");
        saveReportFile(jsonPath, report);
        saveReportFile(csvPath, report);
        std::error_code ec;
        const uintmax_t jsonBytes = fs::file_size(pathFromUtf8(jsonPath), ec);
        const uintmax_t csvBytes = fs::file_size(pathFromUtf8(csvPath), ec);
        std::cout << strf("[selftest] %s report overlap=%u,%u %ux%u json=%llu bytes csv=%llu bytes\n", c.name.c_str(),
                          report.overlap.x, report.overlap.y, report.overlap.width, report.overlap.height,
                          static_cast<unsigned long long>(jsonBytes), static_cast<unsigned long long>(csvBytes));
        if (jsonBytes == 0 || csvBytes == 0)
        {
            results.fail(c.name + ": report files are empty");
        }
    }
}

// ---- FLIP on the GPU -------------------------------------------------------------

bool sameRect(const PixelRect & a, const PixelRect & b)
{
    return a.x == b.x && a.y == b.y && a.width == b.width && a.height == b.height;
}

// The GPU FLIP of the pair of a case at ppd, over the overlap at the layout's offsets (as
// checkMetrics runs it). nullopt after a reported failure. The device map stays valid until
// the next compute() on flipGpu.
std::optional<FlipGpuResult> gpuFlipOf(const Case & c, const LayoutInput & layout, const MediaList & media,
                                       FlipGpu & flipGpu, float ppd, Results & results)
{
    const PixelRect overlap = overlapRect(layout, 0, 1);
    try
    {
        std::optional<FlipGpuResult> gpu = flipGpu.compute(*media[0], layout.sources[0].offsetX,
                                                           layout.sources[0].offsetY, *media[1],
                                                           layout.sources[1].offsetX, layout.sources[1].offsetY,
                                                           overlap, ppd);
        if (!gpu)
        {
            results.fail(c.name + ": the GPU FLIP returned nothing");
        }
        return gpu;
    }
    catch (const std::exception & e)
    {
        results.fail(c.name + ": the GPU FLIP threw: " + e.what());
        return std::nullopt;
    }
}

// The host crops of the pair's overlap, as the metric and the CPU reference take them.
std::pair<ImageData, ImageData> overlapCrops(const Case & c, const LayoutInput & layout)
{
    const PixelRect overlap = overlapRect(layout, 0, 1);
    const int32_t ax = layout.sources[0].offsetX;
    const int32_t ay = layout.sources[0].offsetY;
    const int32_t bx = layout.sources[1].offsetX;
    const int32_t by = layout.sources[1].offsetY;
    return {cropImage(c.images[0], overlap.x - ax, overlap.y - ay, overlap.width, overlap.height),
            cropImage(c.images[1], overlap.x - bx, overlap.y - by, overlap.width, overlap.height)};
}

// The CPU reference: flip() on the display values of the overlap crops.
FlipResult cpuFlipOverlap(const Case & c, const LayoutInput & layout, float ppd)
{
    const std::pair<ImageData, ImageData> crops = overlapCrops(c, layout);
    return flip(displayRgb(crops.first), displayRgb(crops.second), ppd);
}

// Differences of a GPU map from the CPU reference over the same pixels.
struct FlipDiff
{
    double mean = 0.0;      // |GPU mean - CPU mean|
    double p999 = 0.0;      // 99.9th percentile of |GPU - CPU| per pixel
    double max = 0.0;       // largest |GPU - CPU| per pixel
    bool sameSize = false;  // both maps have the same number of values
    bool finite = false;    // the GPU mean and every GPU value are finite
};

// Differences of a GPU map from a CPU map over the same pixels, with the two means.
FlipDiff diffMaps(const std::vector<float> & gpuMap, const std::vector<float> & cpuMap, double gpuMean, double cpuMean)
{
    FlipDiff diff;
    diff.sameSize = gpuMap.size() == cpuMap.size();
    diff.finite = std::isfinite(gpuMean);
    if (!diff.sameSize)
    {
        const double infinity = std::numeric_limits<double>::infinity();
        diff.mean = infinity;
        diff.p999 = infinity;
        diff.max = infinity;
        return diff;
    }
    diff.mean = std::fabs(gpuMean - cpuMean);
    std::vector<double> deltas(gpuMap.size());
    for (size_t i = 0; i < deltas.size(); ++i)
    {
        diff.finite = diff.finite && std::isfinite(gpuMap[i]);
        deltas[i] = std::fabs(static_cast<double>(gpuMap[i]) - static_cast<double>(cpuMap[i]));
        diff.max = std::max(diff.max, deltas[i]);
    }
    if (!diff.finite)
    {
        // A NaN would make the sort undefined; a non-finite map fails on every tolerance anyway.
        const double infinity = std::numeric_limits<double>::infinity();
        diff.p999 = infinity;
        diff.max = infinity;
        return diff;
    }
    if (!deltas.empty())
    {
        // The 99.9th percentile is sorted[ceil(0.999 n) - 1].
        std::sort(deltas.begin(), deltas.end());
        const size_t rank = static_cast<size_t>(std::ceil(0.999 * static_cast<double>(deltas.size())));
        diff.p999 = deltas[std::clamp(rank, size_t{1}, deltas.size()) - 1];
    }
    return diff;
}

FlipDiff compareFlip(const FlipGpuResult & gpu, const FlipResult & cpu)
{
    return diffMaps(gpu.errorMap, cpu.errorMap, gpu.mean, cpu.meanError);
}

bool flipAgrees(const FlipDiff & diff)
{
    return diff.sameSize && diff.finite && diff.mean <= kMaxFlipMeanDelta && diff.p999 <= kMaxFlipP999
        && diff.max <= kMaxFlipMaxDelta;
}

void printFlipLine(const std::string & label, const FlipGpuResult & gpu, const FlipResult & cpu,
                   const FlipDiff & diff, float ppd)
{
    std::cout << strf("[selftest] %s flip gpu=%.9g cpu=%.9g dmean=%.3g p999=%.3g max=%.3g ppd=%g\n", label.c_str(),
                      gpu.mean, cpu.meanError, diff.mean, diff.p999, diff.max, static_cast<double>(ppd));
}

void requireFlipAgrees(const std::string & label, const FlipDiff & diff, Results & results)
{
    if (!flipAgrees(diff))
    {
        results.fail(label + strf(": GPU FLIP differs from the CPU reference (dmean=%.3g p999=%.3g max=%.3g)",
                                  diff.mean, diff.p999, diff.max));
    }
}

// Whether render() refuses the input with std::invalid_argument.
bool renderRefused(Compositor & compositor, const CompositeInput & input)
{
    try
    {
        compositor.render(input);
    }
    catch (const std::invalid_argument &)
    {
        return true;
    }
    catch (const std::exception &)
    {
        return false;
    }
    return false;
}

// The GPU FLIP of a two-source pair at ppd, against flip() on the display crops over the
// overlap. The identical pair must give exactly zero everywhere.
void checkFlip(const Case & c, const LayoutInput & layout, const MediaList & media, FlipGpu & flipGpu, float ppd,
               Results & results)
{
    const std::optional<FlipGpuResult> gpu = gpuFlipOf(c, layout, media, flipGpu, ppd, results);
    if (!gpu)
    {
        return;
    }
    const PixelRect overlap = overlapRect(layout, 0, 1);
    const FlipResult cpu = cpuFlipOverlap(c, layout, ppd);
    const FlipDiff diff = compareFlip(*gpu, cpu);
    printFlipLine(c.name, *gpu, cpu, diff, ppd);
    requireFlipAgrees(c.name, diff, results);

    const size_t pixels = static_cast<size_t>(overlap.width) * static_cast<size_t>(overlap.height);
    if (!sameRect(gpu->overlap, overlap) || gpu->errorMap.size() != pixels || gpu->ppd != ppd)
    {
        results.fail(c.name + ": the GPU FLIP result has the wrong overlap, map size or ppd");
    }
    if (gpu->nonFiniteSamples != 0)
    {
        results.fail(c.name + ": the GPU FLIP counted non-finite samples in a finite pair");
    }
    if (c.name == "identical")
    {
        size_t nonZero = 0;
        for (const float value : gpu->errorMap)
        {
            nonZero += value == 0.0f ? 0 : 1;
        }
        if (gpu->mean != 0.0 || nonZero != 0)
        {
            results.fail(c.name + strf(": identical images give mean %.9g and %zu nonzero pixels", gpu->mean,
                                       nonZero));
        }
    }
}

// The GPU FLIP at ppd 1, 67, 200 and 1000 against the CPU reference. Then ppd values outside
// [1, 1000] and NaN and infinity: each is refused with std::invalid_argument, and no map is left
// behind. A valid compute must work after the refusals.
void checkFlipPpd(const Case & c, const LayoutInput & layout, const MediaList & media, FlipGpu & flipGpu,
                  Results & results)
{
    for (const float ppd : {1.0f, 67.0f, 200.0f, 1000.0f})
    {
        const std::optional<FlipGpuResult> gpu = gpuFlipOf(c, layout, media, flipGpu, ppd, results);
        if (!gpu)
        {
            continue;
        }
        const FlipResult cpu = cpuFlipOverlap(c, layout, ppd);
        const FlipDiff diff = compareFlip(*gpu, cpu);
        printFlipLine(c.name, *gpu, cpu, diff, ppd);
        requireFlipAgrees(c.name + strf(" ppd %g", static_cast<double>(ppd)), diff, results);
    }

    const PixelRect overlap = overlapRect(layout, 0, 1);
    const auto computeAt = [&](float ppd)
    {
        return flipGpu.compute(*media[0], layout.sources[0].offsetX, layout.sources[0].offsetY, *media[1],
                               layout.sources[1].offsetX, layout.sources[1].offsetY, overlap, ppd);
    };
    const float notANumber = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    for (const float ppd : {0.0f, 0.5f, 1000.5f, -3.0f, notANumber, infinity})
    {
        bool refused = false;
        std::string what;
        try
        {
            (void)computeAt(ppd);
        }
        catch (const std::invalid_argument & e)
        {
            refused = true;
            what = e.what();
        }
        catch (const std::exception & e)
        {
            what = e.what();
        }
        const bool noMap = !flipGpu.errorMapView().valid();
        std::cout << strf("[selftest] %s flip ppd %g refused=%s no map=%s (\"%s\")\n", c.name.c_str(),
                          static_cast<double>(ppd), refused ? "yes" : "no", noMap ? "yes" : "no", what.c_str());
        if (!refused || !noMap)
        {
            results.fail(c.name + strf(": FLIP ppd %g is not refused with std::invalid_argument and no map",
                                       static_cast<double>(ppd)));
        }
    }

    const std::optional<FlipGpuResult> after = computeAt(defaultFlipPpd());
    const size_t pixels = static_cast<size_t>(overlap.width) * static_cast<size_t>(overlap.height);
    const bool worksAfter = after && after->errorMap.size() == pixels;
    std::cout << strf("[selftest] %s flip after the refused values: gpu=%.9g works=%s\n", c.name.c_str(),
                      after ? after->mean : -1.0, worksAfter ? "yes" : "no");
    if (!worksAfter)
    {
        results.fail(c.name + ": FLIP does not work after a refused ppd");
    }
}

// The CPU rendering of the FLIP view at the layout's zoom, from the GPU's own map, by the rules
// of compositor.h: inside the overlap the magma color of the map, padding the top of the map,
// and outside the padded extent the background. Nearest sampling, as the view uses.
std::vector<uint8_t> referenceFlipComposite(const CompositeInput & input, const FlipGpuResult & gpu, uint32_t width,
                                            uint32_t height)
{
    const LayoutInput shown = shownLayout(input);
    const PixelRect extent = paddedExtent(shown);
    std::vector<uint8_t> out(static_cast<size_t>(width) * height * 3, byteOf(kCompositeBackground));
    for (uint32_t py = 0; py < height; ++py)
    {
        for (uint32_t px = 0; px < width; ++px)
        {
            const ImageHit hit = windowToImage(shown, static_cast<int32_t>(px), static_cast<int32_t>(py));
            const int32_t pixelX = floorPixel(hit.imageX);
            const int32_t pixelY = floorPixel(hit.imageY);
            if (hit.pane < 0 || !rectContains(extent, pixelX, pixelY))
            {
                continue;
            }
            uint8_t * dst = &out[(static_cast<size_t>(py) * width + px) * 3];
            if (rectContains(gpu.overlap, pixelX, pixelY))
            {
                const size_t index = static_cast<size_t>(pixelY - gpu.overlap.y)
                        * static_cast<size_t>(gpu.overlap.width)
                    + static_cast<size_t>(pixelX - gpu.overlap.x);
                magmaBytes(gpu.errorMap[index], dst);
            }
            else
            {
                magmaBytes(mapPaddingValue(), dst);
            }
        }
    }
    return out;
}

// The FLIP view of the sizes-topleft pair at 100%, read back and compared with the CPU rendering
// of the same rules. Flip inputs that are not valid are refused before anything runs.
// The FLIP view of a result (LDR or HDR, the map on flipGpu) at 100% over the padded extent,
// read back and compared with the CPU rendering of the same rules. label names the case in the
// output. Flip inputs that are not valid are refused before anything runs.
void checkFlipView(const std::string & label, const Case & c, const LayoutInput & layout, const MediaList & media,
                   FlipGpu & flipGpu, const FlipGpuResult & gpu, Compositor & compositor, Results & results)
{
    CompositeInput input = makeInput(c, layout, media);
    input.mode = CompositeMode::Flip;
    input.flipMap = flipGpu.errorMapView();
    input.flipRect = gpu.overlap;
    setZoomPercent(input.layout, 100.0);
    centerView(input.layout);
    const uint32_t width = static_cast<uint32_t>(input.layout.windowWidth);
    const uint32_t height = static_cast<uint32_t>(input.layout.windowHeight);
    const std::vector<uint8_t> ref = referenceFlipComposite(input, gpu, width, height);

    std::vector<uint8_t> gpuBytes;
    try
    {
        compositor.resize(width, height);
        compositor.render(input);
        compositor.readback(gpuBytes);
    }
    catch (const std::exception & e)
    {
        results.fail(label + " flip view: render threw: " + e.what());
        return;
    }
    compareViewBytes(label + " flip-100", gpuBytes, ref, width, height, true, results);

    CompositeInput noMap = input;
    noMap.flipMap = BufferView{};
    CompositeInput noRect = input;
    noRect.flipRect = PixelRect{};
    CompositeInput shortMap = input;
    shortMap.flipMap.size = 4;
    const bool refusedNoMap = renderRefused(compositor, noMap);
    const bool refusedNoRect = renderRefused(compositor, noRect);
    const bool refusedShort = renderRefused(compositor, shortMap);
    std::cout << strf("[selftest] %s flip view refusals: no map=%s no rect=%s short map=%s\n", label.c_str(),
                      refusedNoMap ? "yes" : "no", refusedNoRect ? "yes" : "no", refusedShort ? "yes" : "no");
    if (!refusedNoMap || !refusedNoRect || !refusedShort)
    {
        results.fail(label + ": a flip view with an invalid map or rectangle is not refused with invalid_argument");
    }
}

void checkFlipComposite(const Case & c, const LayoutInput & layout, const MediaList & media, FlipGpu & flipGpu,
                        Compositor & compositor, Results & results)
{
    const std::optional<FlipGpuResult> gpu = gpuFlipOf(c, layout, media, flipGpu, defaultFlipPpd(), results);
    if (gpu)
    {
        checkFlipView(c.name, c, layout, media, flipGpu, *gpu, compositor, results);
    }
}

// A map export over the padded extent (flipMapImage: magma of the values inside the overlap, the
// top of the map in the padding): its size, its bytes, and the PNG round trip at path. label names
// the case and what names the map in the output.
void checkMapExport(const std::string & label, const std::string & what, const ExportScene & scene,
                    const std::vector<float> & values, PixelRect overlap, const std::string & path, Results & results)
{
    const PixelRect extent = exportExtent(scene);
    const uint32_t width = static_cast<uint32_t>(extent.width);
    const uint32_t height = static_cast<uint32_t>(extent.height);
    Rgb8Image map;
    try
    {
        map = flipMapImage(scene, values, overlap);
    }
    catch (const std::exception & e)
    {
        results.fail(label + ": " + what + " export threw: " + e.what());
        return;
    }
    if (map.width != width || map.height != height || map.rgb.size() != static_cast<size_t>(width) * height * 3)
    {
        results.fail(label + ": the " + what + " is not the size of the padded extent");
        return;
    }

    // Export pixel (x, y) is canvas pixel (extent.x + x, extent.y + y).
    std::vector<uint8_t> expected(map.rgb.size());
    for (uint32_t y = 0; y < height; ++y)
    {
        for (uint32_t x = 0; x < width; ++x)
        {
            const int32_t canvasX = extent.x + static_cast<int32_t>(x);
            const int32_t canvasY = extent.y + static_cast<int32_t>(y);
            uint8_t * dst = &expected[(static_cast<size_t>(y) * width + x) * 3];
            if (rectContains(overlap, canvasX, canvasY))
            {
                const size_t index = static_cast<size_t>(canvasY - overlap.y) * static_cast<size_t>(overlap.width)
                    + static_cast<size_t>(canvasX - overlap.x);
                magmaBytes(values[index], dst);
            }
            else
            {
                magmaBytes(mapPaddingValue(), dst);
            }
        }
    }
    size_t mismatches = 0;
    for (size_t i = 0; i < expected.size(); ++i)
    {
        mismatches += map.rgb[i] != expected[i] ? 1 : 0;
    }
    std::cout << strf("[selftest] %s %s export: %ux%u samples=%zu mismatches=%zu\n", label.c_str(), what.c_str(),
                      width, height, expected.size(), mismatches);
    if (mismatches != 0)
    {
        results.fail(label + ": the " + what + " export differs from the magma of the GPU map in "
                     + std::to_string(mismatches) + " samples");
    }

    savePngRgb8(path, map);
    checkPng(label + " " + what, path, map.rgb, width, height, results);
}

// The FLIP map export over the padded extent of the sizes-topleft pair (see checkMapExport), and
// the report with flipMean and ppd.
void checkFlipExports(const Case & c, const LayoutInput & layout, const MediaList & media, FlipGpu & flipGpu,
                      const std::string & outDir, Results & results)
{
    const std::optional<FlipGpuResult> gpu = gpuFlipOf(c, layout, media, flipGpu, defaultFlipPpd(), results);
    if (!gpu)
    {
        return;
    }
    const ExportScene scene = caseExportScene(c, layout);
    checkMapExport(c.name, "FLIP map", scene, gpu->errorMap, gpu->overlap, joinPath(outDir, c.name + "-flip.png"),
                   results);

    // The report carries flipMean and ppd, in the JSON and as the last cell of the CSV row.
    const std::pair<ImageData, ImageData> crops = overlapCrops(c, layout);
    const PsnrResult cpuPsnr = psnr(displayRgb(crops.first), displayRgb(crops.second));
    ReportInput input;
    input.pathA = c.name + "-a";
    input.pathB = c.name + "-b";
    input.a = &c.images[0];
    input.b = &c.images[1];
    input.aOffsetX = layout.sources[0].offsetX;
    input.aOffsetY = layout.sources[0].offsetY;
    input.overlap = gpu->overlap;
    input.mse = cpuPsnr.mse;
    input.psnrDb = cpuPsnr.psnrDb;
    input.anchor = c.centered ? "centered" : "topLeft";
    input.flipMean = gpu->mean;
    input.ppd = defaultFlipPpd();
    const CompareReport report = makeReport(input);
    const std::string json = reportToJson(report);
    const std::string csv = reportToCsv(report);

    const std::string meanKey = "\"flipMean\":";
    const size_t meanAt = json.find(meanKey);
    const double jsonMean = meanAt == std::string::npos
        ? std::numeric_limits<double>::quiet_NaN()
        : std::strtod(json.c_str() + meanAt + meanKey.size(), nullptr);
    const std::string ppdKey = "\"ppd\":";
    const size_t ppdAt = json.find(ppdKey);
    const double jsonPpd = ppdAt == std::string::npos
        ? std::numeric_limits<double>::quiet_NaN()
        : std::strtod(json.c_str() + ppdAt + ppdKey.size(), nullptr);
    std::istringstream lines(csv);
    std::string header;
    std::string row;
    std::getline(lines, header);
    std::getline(lines, row);
    const size_t comma = row.rfind(',');
    const std::string csvMean = comma == std::string::npos ? row : row.substr(comma + 1);
    const bool jsonOk = std::fabs(jsonMean - gpu->mean) <= 1e-9 && std::fabs(jsonPpd - defaultFlipPpd()) <= 1e-3;
    const bool csvOk = !csvMean.empty() && std::fabs(std::strtod(csvMean.c_str(), nullptr) - gpu->mean) <= 1e-6;
    std::cout << strf("[selftest] %s FLIP report flipMean json=%.9g csv=%s ppd json=%g\n", c.name.c_str(), jsonMean,
                      csvMean.c_str(), jsonPpd);
    if (!jsonOk || !csvOk)
    {
        results.fail(c.name + ": the report does not carry flipMean and ppd");
    }
}

void runCase(const Case & c, GpuContext & context, MetricsGpu & metrics, FlipGpu & flipGpu, Compositor & compositor,
             const std::string & outDir, Results & results)
{
    std::cout << strf("[selftest] case %s: %zu sources", c.name.c_str(), c.images.size());
    for (const ImageData & image : c.images)
    {
        std::cout << strf(" %ux%u", image.width, image.height);
    }
    std::cout << '\n';

    const LayoutInput layout = makeLayout(c);
    MediaList media;
    for (size_t i = 0; i < c.images.size(); ++i)
    {
        media.push_back(std::make_unique<ImageSource>(context, c.images[i], c.name + "/" + std::to_string(i)));
        if (media.back()->status() != SourceStatus::Ready)
        {
            results.fail(c.name + ": source " + std::to_string(i) + " is not Ready: " + media.back()->statusMessage());
            return;
        }
    }

    checkSourceReadbacks(c, context, media, results);

    std::optional<MetricsResult> gpuMetrics;
    checkMetrics(c, layout, media, metrics, results, gpuMetrics);
    // FLIP runs on the two-source cases; the grid and the three-source row are not paired.
    if (c.images.size() == 2)
    {
        checkFlip(c, layout, media, flipGpu, defaultFlipPpd(), results);
    }
    if (c.name == "same-size" || c.name == "sizes-topleft")
    {
        checkFlipPpd(c, layout, media, flipGpu, results);
    }

    const CompositeInput base = makeInput(c, layout, media);
    checkComposites(c, base, compositor, results);
    checkViewExport(c, base, compositor, outDir, results);
    checkExports(c, layout, gpuMetrics, outDir, results);
    if (c.name == "sizes-topleft")
    {
        checkFlipComposite(c, layout, media, flipGpu, compositor, results);
        checkFlipExports(c, layout, media, flipGpu, outDir, results);
    }
}

// CPU-only: the source and visibility limits of compare_layout.h.
void runCpuChecks(Results & results)
{
    LayoutInput input;
    LayoutSource visible;
    visible.width = 4;
    visible.height = 4;
    visible.visible = true;
    LayoutSource hidden = visible;
    hidden.visible = false;

    bool ok = true;
    for (int i = 0; i < 4; ++i)
    {
        ok = ok && addSource(input, visible);
    }
    ok = ok && !addSource(input, visible); // a 5th visible source is refused
    ok = ok && addSource(input, hidden);   // index 4, hidden
    ok = ok && !setSourceVisible(input, 4, true) && visibleCount(input) == 4;
    ok = ok && setSourceVisible(input, 0, false) && visibleCount(input) == 3;
    ok = ok && setSourceVisible(input, 4, true) && visibleCount(input) == 4;
    ok = ok && addSource(input, hidden) && addSource(input, hidden) && addSource(input, hidden);
    ok = ok && input.sourceCount == kMaxSources;   // 8 sources
    ok = ok && !addSource(input, hidden);          // a 9th is refused
    ok = ok && !setSourceVisible(input, kMaxSources, true);
    std::cout << strf("[selftest] source limits: %d sources, %d visible: %s\n", input.sourceCount,
                      visibleCount(input), ok ? "ok" : "FAILED");
    if (!ok)
    {
        results.fail("source limits: the 8-source or 4-visible cap is not enforced");
    }
}

// A missing file is status Missing, and a file that is not an image is status Error; both
// keep no device buffer and no host copy.
void checkFailedLoads(GpuContext & context, const std::string & outDir, Results & results)
{
    const std::string missingPath = joinPath(outDir, "does-not-exist.png");
    const ImageSource missing(context, missingPath);
    const std::string corruptPath = joinPath(outDir, "corrupt.png");
    {
        std::ofstream out(pathFromUtf8(corruptPath), std::ios::binary);
        out << "not a png";
    }
    const ImageSource corrupt(context, corruptPath);

    const bool missingOk = missing.status() == SourceStatus::Missing
        && missing.statusMessage().find("file not found") != std::string::npos && !missing.color().valid()
        && missing.data() == nullptr;
    const bool corruptOk = corrupt.status() == SourceStatus::Error && !corrupt.statusMessage().empty()
        && !corrupt.color().valid() && corrupt.data() == nullptr;
    std::cout << strf("[selftest] failed loads: missing=%s (\"%s\") corrupt=%s (\"%s\")\n",
                      missingOk ? "Missing" : "WRONG", missing.statusMessage().c_str(),
                      corruptOk ? "Error" : "WRONG", corrupt.statusMessage().c_str());
    if (!missingOk)
    {
        results.fail("a missing file is not status Missing with a file-not-found message");
    }
    if (!corruptOk)
    {
        results.fail("a corrupt file is not status Error with a message");
    }
}


// ---- Media list and project (CompareState) ---------------------------------------

// A display image on disk, for the state checks.
void writeRgbPng(const std::string & path, uint32_t width, uint32_t height, uint32_t seed)
{
    Rgb8Image image;
    image.width = width;
    image.height = height;
    image.rgb.resize(static_cast<size_t>(width) * height * 3);
    Lcg rng(seed);
    for (uint8_t & value : image.rgb)
    {
        value = static_cast<uint8_t>(rng.next() * 255.0f);
    }
    savePngRgb8(path, image);
}

// A file path on a drive other than the one of outDir: the project keeps it absolute.
std::string otherDrivePath(const std::string & outDir)
{
    const std::string root = utf8Of(fs::absolute(pathFromUtf8(outDir)).root_name());
    const char letter = (!root.empty() && (root[0] == 'C' || root[0] == 'c')) ? 'Z' : 'C';
    return std::string(1, letter) + ":\\sabercat-selftest-missing\\other.png";
}

// The media list rules, the view and the layout sync, the metrics availability, the project
// round trip (an other-drive path, a missing file) and the refusal of a newer formatVersion.
void checkState(GpuContext & context, MetricsGpu & metrics, const std::string & outDir, Results & results)
{
    const std::string pathA = joinPath(outDir, "state-a.png");
    const std::string pathB = joinPath(outDir, "state-b.png");
    writeRgbPng(pathA, 16, 12, 11);
    writeRgbPng(pathB, 20, 10, 12);

    const auto report = [&results](const char * name, bool ok, const std::string & detail)
    {
        std::cout << "[selftest] state " << name << ": " << (ok ? "ok" : "FAILED") << " (" << detail << ")\n";
        if (!ok)
        {
            results.fail(std::string("state ") + name + ": " + detail);
        }
    };

    // 1. The cap of 8 sources, the first 4 shown, a fifth visible refused, hidden sources take no role.
    {
        CompareState state(context);
        bool indices = true;
        for (int32_t i = 0; i < kMaxSources; ++i)
        {
            indices = indices && state.addMedia(pathA) == i;
        }
        const bool ninth = state.addMedia(pathA) == -1 && state.status().find("At most 8") != std::string::npos;
        size_t visible = 0;
        for (const MediaEntry & entry : state.entries())
        {
            visible += entry.visible ? 1 : 0;
        }
        const bool firstFour = visible == static_cast<size_t>(kMaxVisible) && !state.entries()[4].visible;
        const bool fifth = !state.setVisible(4, true) && state.status() == "At most 4 sources can be visible";
        const bool hideThenShow = state.setVisible(0, false) && state.setVisible(4, true) && state.size() == 8;
        const bool hiddenRole = !state.setRole(7, project::SourceRole::A);
        report("cap", indices && ninth && firstFour && fifth && hideThenShow && hiddenRole,
               "8 entries, 9th refused, 4 visible, 5th refused, hidden source takes no role");
    }

    // 2. Roles: A and B unique, defaults from the first visible sources.
    {
        CompareState state(context);
        state.addMedia(pathA);
        state.addMedia(pathB);
        state.addMedia(pathA);
        const bool defaults = state.sourceA() == 0 && state.sourceB() == 1;
        const bool moved = state.setRole(2, project::SourceRole::A) && state.sourceA() == 2
            && state.entries()[0].role == project::SourceRole::None && state.sourceB() == 1;
        const bool swapped = state.setRole(2, project::SourceRole::B) && state.sourceB() == 2
            && state.sourceA() == 0;
        int aCount = 0;
        int bCount = 0;
        for (const MediaEntry & entry : state.entries())
        {
            aCount += entry.role == project::SourceRole::A ? 1 : 0;
            bCount += entry.role == project::SourceRole::B ? 1 : 0;
        }
        report("roles", defaults && moved && swapped && aCount == 1 && bCount == 1,
               "defaults A=0 B=1; A and B stay unique after moves");
    }

    // 3. Offsets and anchors reach the layout.
    {
        CompareState state(context);
        state.addMedia(pathA);
        state.addMedia(pathB);
        const bool set = state.setOffset(1, 5, -3);
        const bool custom = state.settings().anchor == project::Anchor::Custom;
        const bool synced = state.layout().sources[1].offsetX == 5 && state.layout().sources[1].offsetY == -3;
        const bool refused = !state.setOffset(1, 1 << 25, 0);
        state.applyAnchor(project::Anchor::TopLeft);
        const bool topLeft = state.layout().sources[0].offsetX == 0 && state.layout().sources[1].offsetY == 0;
        state.applyAnchor(project::Anchor::Centered);
        // The largest size is 20 x 12: a 16 x 12 source centers at (2, 0), a 20 x 10 at (0, 1).
        const bool centered = state.layout().sources[0].offsetX == 2 && state.layout().sources[0].offsetY == 0
            && state.layout().sources[1].offsetX == 0 && state.layout().sources[1].offsetY == 1
            && state.entries()[0].offsetX == 2 && state.entries()[1].offsetY == 1;
        report("offsets", set && custom && synced && refused && topLeft && centered,
               "offsets sync to the layout; TopLeft and Centered anchors; out of range refused");
    }

    // 4. Metrics: none for one source, mse 0 for a source against itself, none without overlap.
    {
        CompareState state(context);
        state.addMedia(pathA);
        state.updateMetrics(metrics);
        const bool single = !state.metrics() && state.metricsReason() == "Add a second image to compare";
        state.addMedia(pathA);
        state.updateMetrics(metrics);
        const bool identical = state.metrics().has_value() && state.metrics()->mse == 0.0
            && std::isinf(state.metrics()->psnrDb) && state.metrics()->overlap.width == 16;
        state.setOffset(1, 1000, 1000);
        state.updateMetrics(metrics);
        const bool apart = !state.metrics() && state.metricsReason() == "A and B do not overlap";
        report("metrics", single && identical && apart,
               "one source: no metrics; the same file: mse 0; no overlap: reason set");
    }

    // 5. Project round trip: an other-drive path, a missing file, an exposure and a background.
    {
        const std::string missingPath = joinPath(outDir, "state-missing.png");
        const std::string otherPath = otherDrivePath(outDir);
        const std::string projectPath = joinPath(outDir, "state-roundtrip.sabercat-compare");
        CompareState source(context);
        source.addMedia(pathA);
        source.addMedia(pathB);
        source.setOffset(1, 3, 2);
        source.setExposure(0, 1.5f);
        source.setBackground(0, project::Rgb{0.25f, 0.5f, 0.75f});
        source.addMedia(missingPath);
        source.addMedia(otherPath);
        std::string error;
        const bool saved = source.saveProject(projectPath, error);

        CompareState loaded(context);
        const bool opened = loaded.openProject(projectPath, error);
        bool same = saved && opened && loaded.size() == source.size();
        for (size_t i = 0; same && i < source.size(); ++i)
        {
            const MediaEntry & x = source.entries()[i];
            const MediaEntry & y = loaded.entries()[i];
            same = x.id == y.id && x.offsetX == y.offsetX && x.offsetY == y.offsetY && x.role == y.role
                && x.visible == y.visible && x.ready() == y.ready() && x.name == y.name;
        }
        const bool look = same && loaded.entries()[0].exposure == 1.5f && loaded.entries()[0].background.g == 0.5f;
        const bool missing = same && !loaded.entries()[2].ready()
            && loaded.entries()[2].message.find("file not found") != std::string::npos;
        // The loader writes the path with forward slashes, so the paths are compared as paths.
        const bool other = same && loaded.entries()[3].otherDriveWarning
            && fs::path(pathFromUtf8(loaded.entries()[3].path)) == pathFromUtf8(otherPath)
            && !loaded.entries()[3].ready();
        bool notice = false;
        for (const std::string & line : loaded.notices())
        {
            notice = notice || line.find("another drive") != std::string::npos;
        }
        const std::string flags = strf("saved=%d opened=%d same=%d look=%d missing=%d other=%d notice=%d clean=%d",
                                       saved ? 1 : 0, opened ? 1 : 0, same ? 1 : 0, look ? 1 : 0, missing ? 1 : 0,
                                       other ? 1 : 0, notice ? 1 : 0, loaded.projectDirty() ? 0 : 1);
        report("project", same && look && missing && other && notice && !loaded.projectDirty(),
               "round trip keeps ids, offsets, roles, visibility; missing and other-drive flagged; " + flags
                   + (error.empty() ? std::string() : " error=" + error));
    }

    // 6. A newer formatVersion is refused and the state is left as it was.
    {
        CompareState state(context);
        state.addMedia(pathA);
        std::string text = project::serializeCompareProject(CompareState(context).toProject(), outDir);
        const size_t key = text.find("\"formatVersion\":");
        const size_t end = key == std::string::npos ? std::string::npos : text.find_first_of(",\n", key);
        bool refused = false;
        if (end != std::string::npos)
        {
            text.replace(key, end - key, "\"formatVersion\": 999");
            const std::string newerPath = joinPath(outDir, "state-newer.sabercat-compare");
            {
                std::ofstream out(pathFromUtf8(newerPath), std::ios::binary);
                out << text;
            }
            std::string error;
            refused = !state.openProject(newerPath, error) && error.find("newer") != std::string::npos
                && state.size() == 1 && state.entries()[0].name == "state-a.png";
        }
        report("newer format", refused, "formatVersion 999 refused, state unchanged");
    }

    // 7. View rules: the composite mode follows the view; the flicker phase is half and half.
    {
        CompareState state(context);
        state.addMedia(pathA);
        state.setSplit(false);
        state.setSingleView(SingleView::Wipe);
        const bool onePane = state.compositeMode() == CompositeMode::Panes;
        state.addMedia(pathB);
        const bool wipe = state.compositeMode() == CompositeMode::Wipe;
        state.setSingleView(SingleView::Flicker);
        const bool flicker = state.compositeMode() == CompositeMode::Flicker;
        const bool phase = !CompareState::flickerShowB(0.0, 1.0) && CompareState::flickerShowB(0.5, 1.0)
            && !CompareState::flickerShowB(1.25, 1.0) && CompareState::flickerShowB(1.75, 1.0);
        report("view", onePane && wipe && flicker && phase, "modes need a pair; flicker phase is half and half");
    }

    // 8. The split divider: Split with exactly two visible sources is one pane with a vertical wipe
    // at the divider. Any other count keeps the panes, and Split refuses the overlay modes.
    {
        constexpr int32_t kDividerWidth = 240;
        constexpr int32_t kDividerHeight = 140;
        CompareState two(context);
        two.addMedia(pathA);
        two.addMedia(pathB);
        two.setViewSize(kDividerWidth, kDividerHeight);
        const bool active = two.dividerActive() && two.sourceA() == 0 && two.sourceB() == 1;
        two.setSingleView(SingleView::Wipe);
        const bool overlayRefused = two.compositeMode() == CompositeMode::Panes;
        const CompositeInput divided = two.buildComposite(0.0);
        const bool wipe = divided.mode == CompositeMode::Wipe && divided.layout.view.mode == ViewMode::SingleScreen
            && divided.layout.view.abIndex == 0 && divided.layout.view.wipeOrientation == WipeOrientation::Vertical
            && two.layout().view.mode == ViewMode::SingleScreen;
        report("divider mode", active && overlayRefused && wipe,
               "two visible sources in Split: one pane, vertical wipe, overlay modes refused in Split");

        CompareState three(context);
        three.addMedia(pathA);
        three.addMedia(pathB);
        three.addMedia(pathA);
        const bool threeOff = !three.dividerActive() && three.buildComposite(0.0).mode == CompositeMode::Panes
            && three.layout().view.mode == ViewMode::SplitScreen;
        CompareState one(context);
        one.addMedia(pathA);
        const bool oneOff = !one.dividerActive() && one.buildComposite(0.0).mode == CompositeMode::Panes
            && one.layout().view.mode == ViewMode::SplitScreen;
        CompareState single(context);
        single.addMedia(pathA);
        single.addMedia(pathB);
        single.setSplit(false);
        single.setSingleView(SingleView::Wipe);
        const bool singleOff = !single.dividerActive() && single.buildComposite(0.0).mode == CompositeMode::Wipe
            && single.layout().view.mode == ViewMode::SingleScreen;
        report("divider counts", threeOff && oneOff && singleOff,
               "one and three visible sources keep the panes; Single keeps its own modes");

        // Split off and on keeps the zoom and the pan; the divider is a window fraction, so it is kept too.
        setZoomPercent(two.viewLayout(), 200.0);
        setPan(two.viewLayout(), 3.5, -2.0);
        two.setSplit(false);
        two.setSplit(true);
        const bool kept = two.layout().view.zoomMode == ZoomMode::Percent && two.layout().view.zoomPercent == 200.0
            && two.layout().view.panX == 3.5 && two.layout().view.panY == -2.0 && two.dividerActive();
        report("divider toggle", kept, "Split off and on keeps zoom, pan and the divider mode");
    }

    // 9. The divider fraction: clamped, NaN is the default, dirty only when it changes, and the
    // project file keeps it (a file without the key is the default, checked in the unit tests).
    {
        const std::string dividerProject = joinPath(outDir, "state-divider.sabercat-compare");
        CompareState state(context);
        state.addMedia(pathA);
        state.addMedia(pathB);
        const bool defaulted = state.dividerFraction() == 0.5;
        state.setDividerFraction(-1.0);
        const bool low = state.dividerFraction() == 0.0;
        state.setDividerFraction(7.0);
        const bool high = state.dividerFraction() == 1.0;
        state.setDividerFraction(std::numeric_limits<double>::quiet_NaN());
        const bool nanDefault = state.dividerFraction() == 0.5;
        std::string error;
        const bool saved0 = state.saveProject(dividerProject, error);
        const bool clean = saved0 && !state.projectDirty();
        state.setDividerFraction(0.5);
        const bool unchangedClean = !state.projectDirty();
        state.setDividerFraction(0.3);
        const bool dirty = state.projectDirty();
        report("divider fraction", defaulted && low && high && nanDefault && clean && unchangedClean && dirty,
               "clamped to [0,1], NaN is 0.5, dirty only on a change");

        const bool saved = state.saveProject(dividerProject, error);
        CompareState loaded(context);
        const bool opened = loaded.openProject(dividerProject, error);
        const bool kept = saved && opened && std::fabs(loaded.dividerFraction() - 0.3) <= 1e-6
            && !loaded.projectDirty();
        report("divider project", kept, "split divider 0.3 survives save and open; clean after open");
    }
}

// ---- Scene sources ---------------------------------------------------------------

constexpr uint32_t kSceneWidth = 160;
constexpr uint32_t kSceneHeight = 120;
constexpr size_t kScenePixels = static_cast<size_t>(kSceneWidth) * kSceneHeight;
constexpr size_t kSyntheticCount = 3000;
// The view area of the map and probe checks, in window pixels.
constexpr uint32_t kViewWidth = 240;
constexpr uint32_t kViewHeight = 140;

const float kNaN = std::numeric_limits<float>::quiet_NaN();
const float kInf = std::numeric_limits<float>::infinity();

struct SceneFiles
{
    std::string ply;
    std::string spz;
};

// Deterministic splats inside the unit cube: SH degree 3 with every band non-zero, and
// opacities from 0.5 to 0.95, so an image has partial coverage, background gaps and some
// fully covered pixels.
// The first `count` Gaussians of the synthetic cloud (the live checks write a smaller variant).
void buildSyntheticCloud(sabercat::GaussianCloud & cloud, size_t count = kSyntheticCount)
{
    cloud.initialize(count);
    Lcg rng(77);
    const size_t shPerGaussian = static_cast<size_t>(kShCoefficientsPerGaussian);
    for (size_t i = 0; i < count; ++i)
    {
        const float px = rng.next() - 0.5f;
        const float py = rng.next() - 0.5f;
        const float pz = rng.next() - 0.5f;
        cloud.positions()[i] = glm::vec4(px, py, pz, 0.0f);
        const float sx = 0.03f + 0.06f * rng.next();
        const float sy = 0.03f + 0.06f * rng.next();
        const float sz = 0.03f + 0.06f * rng.next();
        cloud.scales()[i] = glm::vec4(sx, sy, sz, 0.0f);
        const float qx = rng.next() - 0.5f;
        const float qy = rng.next() - 0.5f;
        const float qz = rng.next() - 0.5f;
        const float qw = rng.next() + 0.25f;
        cloud.rotations()[i] = glm::normalize(glm::vec4(qx, qy, qz, qw));
        cloud.opacities()[i] = 0.5f + 0.45f * rng.next();
        for (size_t k = 0; k < shPerGaussian; ++k)
        {
            // The DC term spans a wide color range; the higher bands are smaller and non-zero.
            const float amplitude = k == 0 ? 4.0f : 0.5f;
            const float r = amplitude * (rng.next() - 0.5f);
            const float g = amplitude * (rng.next() - 0.5f);
            const float b = amplitude * (rng.next() - 0.5f);
            cloud.shCoefficients()[i * shPerGaussian + k] = glm::vec4(r, g, b, 0.0f);
        }
    }
    cloud.setShDegree(3);
    cloud.computeBounds();
}

// The same cloud as a .ply and as a .spz: the two formats the scene checks load.
SceneFiles writeSyntheticScenes(sabercat::Renderer & host, const std::string & outDir)
{
    sabercat::GaussianCloud cloud(host.sabercatDevice());
    buildSyntheticCloud(cloud);
    SceneFiles files;
    files.ply = joinPath(outDir, "synthetic.ply");
    files.spz = joinPath(outDir, "synthetic.spz");
    sabercat::PLYWriter::write(cloud, files.ply);
    sabercat::SPZWriter::write(cloud, files.spz);
    return files;
}

// A renderer of its own on the host's device, loaded with one scene file: the independent
// reference of the scene checks.
class ScratchScene
{
  public:
    ScratchScene(const sabercat::Renderer & host, const std::string & shaderDir, const std::string & path,
                 uint32_t width, uint32_t height)
        : m_renderer(std::make_unique<sabercat::Renderer>(width, height, sabercat::DepthLayout::DepthCoverageFront))
    {
        m_renderer->initializeBorrowed(host.vulkanContext(), host.enabledDeviceFeatures(), shaderDir);
        m_renderer->loadScene(path.c_str());
    }
    ~ScratchScene()
    {
        try
        {
            m_renderer->shutdown();
        }
        catch (const std::exception &)
        {
            // Nothing to recover at teardown.
        }
    }
    ScratchScene(const ScratchScene &) = delete;
    ScratchScene & operator=(const ScratchScene &) = delete;

    sabercat::Renderer & renderer()
    {
        return *m_renderer;
    }

  private:
    std::unique_ptr<sabercat::Renderer> m_renderer;
};

// The floats of a device view, downloaded (empty when the view is invalid or too small).
std::vector<float> downloadFloats(GpuContext & context, const BufferView & view, size_t floats)
{
    if (!view.valid() || view.size < floats * sizeof(float))
    {
        return {};
    }
    std::vector<float> out(floats, -1.0f);
    context.readBuffer(view, out.data(), floats * sizeof(float));
    return out;
}

// Floats whose bits differ (every element when the sizes differ).
size_t bitMismatches(const std::vector<float> & a, const std::vector<float> & b)
{
    if (a.size() != b.size())
    {
        return std::max(a.size(), b.size());
    }
    size_t count = 0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        uint32_t x = 0;
        uint32_t y = 0;
        std::memcpy(&x, &a[i], sizeof(x));
        std::memcpy(&y, &b[i], sizeof(y));
        count += x != y ? 1 : 0;
    }
    return count;
}

struct AlphaCensus
{
    size_t empty = 0;
    size_t partial = 0;
    size_t opaque = 0;
};

// Pixels by coverage from RGBA floats: none, partial and full.
AlphaCensus alphaCensus(const std::vector<float> & rgba)
{
    AlphaCensus census;
    for (size_t p = 0; p + 3 < rgba.size(); p += 4)
    {
        const float alpha = rgba[p + 3];
        if (alpha <= 0.0f)
        {
            ++census.empty;
        }
        else if (alpha >= 1.0f)
        {
            ++census.opaque;
        }
        else
        {
            ++census.partial;
        }
    }
    return census;
}

// The orbit camera of the scene checks: distance 3 at the origin. Pose 1 is orbited the way
// a drag of the mouse would move it.
sabercat::app::OrbitCamera sceneCamera(int pose)
{
    sabercat::app::OrbitCamera camera;
    camera.distance = 3.0f;
    if (pose == 1)
    {
        camera.applyDelta(0.6f, -0.35f, 0.0f);
    }
    return camera;
}

// The bits of two values are equal.
template <typename T>
bool sameBytes(const T & a, const T & b)
{
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

// A display image with a depth buffer of invalid samples (onlyInvalid) or of invalid and valid
// samples, for the in-memory sources.
ImageData makeDepthImage(uint32_t width, uint32_t height, uint32_t seed, bool onlyInvalid)
{
    ImageData image = makeDisplay(width, height, seed);
    // Invalid samples: NaN, infinity, zero and a negative. Valid: 2, 5 and 0.5 (range 0.5..5).
    const float pattern[8] = {kNaN, kInf, 0.0f, -1.0f, 2.0f, 5.0f, -kInf, 0.5f};
    image.depth.resize(static_cast<size_t>(width) * height);
    for (size_t i = 0; i < image.depth.size(); ++i)
    {
        image.depth[i] = onlyInvalid ? pattern[i % 4] : pattern[i % 8];
    }
    return image;
}

// Renders the scene at the camera of pose 0 and at pose 1 through a SceneSource and through
// an independent renderer of the same file; both must agree bit for bit with the readbacks.
void checkSceneRenders(GpuContext & context, sabercat::Renderer & host, const SceneFiles & files,
                       const std::string & shaderDir, Results & results)
{
    const char * formats[2] = {"ply", "spz"};
    const std::string paths[2] = {files.ply, files.spz};
    for (size_t f = 0; f < 2; ++f)
    {
        SceneSettings settings;
        settings.width = kSceneWidth;
        settings.height = kSceneHeight;
        SceneSource source(context, host, paths[f], settings, false);
        if (source.status() != SourceStatus::Ready)
        {
            results.fail(std::string("scene ") + formats[f] + " did not load: " + source.statusMessage());
            continue;
        }
        ScratchScene reference(host, shaderDir, paths[f], kSceneWidth, kSceneHeight);
        reference.renderer().setFlipUpAxis(false);
        reference.renderer().setMaxSHDegree(3);
        reference.renderer().setAntialiased(source.settings().antialias);

        for (int pose = 0; pose < 2; ++pose)
        {
            const sabercat::Camera camera = sceneCamera(pose).toCamera(kSceneWidth, kSceneHeight);
            const bool rendered = source.render(camera, static_cast<uint64_t>(pose) + 1);
            const std::vector<float> color = downloadFloats(context, source.color(), kScenePixels * 4);
            const std::vector<float> depth = downloadFloats(context, source.depth(), kScenePixels * 4);
            const size_t colorVsReadback = bitMismatches(color, source.renderer().readImage());
            const size_t depthVsReadback = bitMismatches(depth, source.renderer().readDepth());

            reference.renderer().setCamera(camera);
            reference.renderer().forward(RenderMode::Differentiable);
            const size_t colorVsReference = bitMismatches(color, reference.renderer().readImage());
            const size_t depthVsReference = bitMismatches(depth, reference.renderer().readDepth());

            const AlphaCensus census = alphaCensus(color);
            std::cout << strf("[selftest] scene %s pose %d: rendered=%d color mismatches=%zu depth mismatches=%zu "
                              "independent renderer color=%zu depth=%zu; coverage empty=%zu partial=%zu opaque=%zu\n",
                              formats[f], pose, rendered ? 1 : 0, colorVsReadback, depthVsReadback, colorVsReference,
                              depthVsReference, census.empty, census.partial, census.opaque);
            if (!rendered || colorVsReadback != 0 || depthVsReadback != 0 || colorVsReference != 0
                || depthVsReference != 0)
            {
                results.fail(std::string("scene ") + formats[f] + " pose " + std::to_string(pose)
                             + " is not bit-exact with its readbacks or an independent renderer");
            }
            if (census.partial == 0 || census.empty == 0)
            {
                results.fail(std::string("scene ") + formats[f] + " has no partial coverage or no background gaps");
            }
        }
    }
}

// Every debug view of the scene's color equals the renderer's readback of that view; Final
// equals readImage(), and switching back to Final restores it.
void checkSceneDebugViews(GpuContext & context, SceneSource & scene, Results & results)
{
    const sabercat::Camera camera = sceneCamera(0).toCamera(kSceneWidth, kSceneHeight);
    for (const sabercat::app::DebugViewInfo & info : sabercat::app::kDebugViews)
    {
        scene.setDebugView(info.view);
        scene.render(camera, 1);
        const std::vector<float> shown = downloadFloats(context, scene.color(), kScenePixels * 4);
        const std::vector<float> expected = info.view == sabercat::DebugView::Final
            ? scene.renderer().readImage()
            : scene.renderer().readDebugView(info.view);
        const size_t mismatches = bitMismatches(shown, expected);
        std::cout << strf("[selftest] scene debug view %s: values=%zu mismatches=%zu\n", info.name, shown.size(),
                          mismatches);
        if (!scene.rendered() || mismatches != 0)
        {
            results.fail(std::string("scene debug view ") + info.name + " differs from readDebugView");
        }
    }
    scene.setDebugView(sabercat::DebugView::Final);
    scene.render(camera, 1);
    const std::vector<float> back = downloadFloats(context, scene.color(), kScenePixels * 4);
    const size_t mismatches = bitMismatches(back, scene.renderer().readImage());
    std::cout << strf("[selftest] scene back to Final: mismatches=%zu vs readImage\n", mismatches);
    if (mismatches != 0)
    {
        results.fail("scene color after switching back to Final differs from readImage");
    }
}

// Two metrics agree: MSE within the relative tolerance and PSNR within the dB tolerance.
void checkMetricNumbers(const std::string & label, const MetricsResult & gpu, const PsnrResult & cpu,
                        Results & results)
{
    double rel = 0.0;
    bool mseOk = false;
    if (cpu.mse == 0.0)
    {
        rel = gpu.mse;
        mseOk = gpu.mse == 0.0;
    }
    else
    {
        rel = std::fabs(gpu.mse - cpu.mse) / cpu.mse;
        mseOk = rel <= kMaxRelativeMse;
    }
    double delta = 0.0;
    bool psnrOk = false;
    if (std::isinf(cpu.psnrDb) || std::isinf(gpu.psnrDb))
    {
        psnrOk = std::isinf(cpu.psnrDb) && std::isinf(gpu.psnrDb);
    }
    else
    {
        delta = std::fabs(gpu.psnrDb - cpu.psnrDb);
        psnrOk = delta <= kMaxPsnrDeltaDb;
    }
    std::cout << strf("[selftest] %s metrics gpu mse=%.9g cpu mse=%.9g rel=%.3g gpu psnr=%.6f cpu psnr=%.6f "
                      "delta=%.3g pixels=%llu\n",
                      label.c_str(), gpu.mse, cpu.mse, rel, gpu.psnrDb, cpu.psnrDb, delta,
                      static_cast<unsigned long long>(gpu.pixelCount));
    if (!mseOk || !psnrOk)
    {
        results.fail(label + ": metric differs from the CPU");
    }
}

// PSNR and MSE of the .ply scene against its .spz twin, through CompareState (the path the UI
// takes), against the CPU core on the host copies.
void checkScenePairMetrics(GpuContext & context, sabercat::Renderer & host, MetricsGpu & metrics,
                           const SceneFiles & files, Results & results)
{
    CompareState state(context);
    state.setSceneHost(&host);
    state.setRenderSize(kSceneWidth, kSceneHeight);
    const int32_t a = state.addMedia(files.ply);
    const int32_t b = state.addMedia(files.spz);
    if (a != 0 || b != 1)
    {
        results.fail("scene metrics: the two scenes were not added as entries 0 and 1");
        return;
    }
    state.updateScenes();
    state.updateMetrics(metrics);
    if (!state.metrics())
    {
        results.fail("scene metrics: no metrics: " + state.metricsReason());
        return;
    }
    const ImageData * hostA = state.entries()[0].media->hostData();
    const ImageData * hostB = state.entries()[1].media->hostData();
    if (hostA == nullptr || hostB == nullptr)
    {
        results.fail("scene metrics: a scene has no host copy");
        return;
    }
    const PsnrResult cpu = psnr(displayRgb(*hostA), displayRgb(*hostB));
    checkMetricNumbers("scene ply vs spz", *state.metrics(), cpu, results);
}

// The scenes' color, depth and debug views are all reached through a SceneSource that
// renders from the shared camera: the shCap 0 image equals a degree-0 renderer, the up-axis
// flip changes the image, and toggling it back restores it bit for bit.
void checkSceneSettings(GpuContext & context, sabercat::Renderer & host, const SceneFiles & files,
                        const std::string & shaderDir, Results & results)
{
    CompareState state(context);
    state.setSceneHost(&host);
    state.setRenderSize(kSceneWidth, kSceneHeight);
    const int32_t index = state.addMedia(files.ply);
    if (index < 0)
    {
        results.fail("scene settings: the scene was not added");
        return;
    }
    const size_t slot = static_cast<size_t>(index);
    const size_t pixels = kScenePixels * 4;
    const auto snapshot = [&]()
    {
        state.updateScenes();
        return downloadFloats(context, state.entries()[slot].media->color(), pixels);
    };
    const std::vector<float> full = snapshot();

    state.setShCap(slot, 0);
    const std::vector<float> dc = snapshot();
    ScratchScene reference(host, shaderDir, files.ply, kSceneWidth, kSceneHeight);
    reference.renderer().setMaxSHDegree(0);
    reference.renderer().setFlipUpAxis(false);
    reference.renderer().setAntialiased(state.entries()[slot].antialias);
    reference.renderer().setCamera(state.camera().toCamera(kSceneWidth, kSceneHeight));
    reference.renderer().forward(RenderMode::Differentiable);
    const size_t dcVsReference = bitMismatches(dc, reference.renderer().readImage());
    const size_t dcVsFull = bitMismatches(dc, full);

    state.setShCap(slot, -1);
    const size_t restoredFull = bitMismatches(snapshot(), full);

    state.setFlipUpAxis(slot, true);
    const std::vector<float> flipped = snapshot();
    const size_t flipChanged = bitMismatches(flipped, full);
    reference.renderer().setFlipUpAxis(true);
    reference.renderer().setMaxSHDegree(3);
    reference.renderer().forward(RenderMode::Differentiable);
    const size_t flipVsReference = bitMismatches(flipped, reference.renderer().readImage());

    state.setFlipUpAxis(slot, false);
    const size_t flipBack = bitMismatches(snapshot(), full);

    std::cout << strf("[selftest] scene settings: shCap 0 vs degree-0 renderer mismatches=%zu, vs full SH "
                      "differing values=%zu; restored full mismatches=%zu; flip changed values=%zu, vs flipped "
                      "renderer mismatches=%zu; flip back mismatches=%zu\n",
                      dcVsReference, dcVsFull, restoredFull, flipChanged, flipVsReference, flipBack);
    if (dcVsReference != 0 || dcVsFull == 0 || restoredFull != 0 || flipChanged == 0 || flipVsReference != 0
        || flipBack != 0)
    {
        results.fail("scene settings: SH cap or up-axis flip does not match the renderer or does not restore");
    }
}

// Failures are per source: a missing file and a file that is not a scene are entries with a
// message, and the other scenes keep rendering. Resizing re-renders at the new size, bad sizes
// are refused, and the same size twice is harmless.
void checkSceneFailures(GpuContext & context, sabercat::Renderer & host, const SceneFiles & files,
                        const std::string & outDir, Results & results)
{
    const std::string garbagePath = joinPath(outDir, "x.ply");
    {
        std::ofstream out(pathFromUtf8(garbagePath), std::ios::binary);
        out << "this is not a splat scene\n";
    }
    CompareState state(context);
    state.setSceneHost(&host);
    state.setRenderSize(kSceneWidth, kSceneHeight);
    const int32_t missing = state.addMedia(joinPath(outDir, "no-such-scene.ply"));
    const int32_t garbage = state.addMedia(garbagePath);
    const int32_t good = state.addMedia(files.ply);
    const bool missingOk = missing >= 0 && state.entries()[static_cast<size_t>(missing)].media->status()
        == SourceStatus::Missing
        && state.entries()[static_cast<size_t>(missing)].message.find("file not found") != std::string::npos;
    const bool garbageOk = garbage >= 0 && state.entries()[static_cast<size_t>(garbage)].media->status()
        == SourceStatus::Error && !state.entries()[static_cast<size_t>(garbage)].message.empty();
    const bool goodOk = good >= 0 && state.entries()[static_cast<size_t>(good)].ready();
    state.updateScenes();
    const auto * goodScene = good >= 0 ? dynamic_cast<SceneSource *>(state.entries()[static_cast<size_t>(good)].media.get()) : nullptr;
    const bool goodRendered = goodScene != nullptr && goodScene->rendered();
    std::cout << strf("[selftest] scene failures: missing=%s (\"%s\") garbage=%s (\"%s\") other scene rendered=%s\n",
                      missingOk ? "Missing" : "WRONG",
                      missing >= 0 ? state.entries()[static_cast<size_t>(missing)].message.c_str() : "",
                      garbageOk ? "Error" : "WRONG",
                      garbage >= 0 ? state.entries()[static_cast<size_t>(garbage)].message.c_str() : "",
                      goodRendered ? "yes" : "no");
    if (!missingOk || !garbageOk || !goodOk || !goodRendered)
    {
        results.fail("scene failures: a bad source does not give its message or stops the other scene");
    }

    // A standalone scene: resize, re-render, refuse bad sizes.
    SceneSettings settings;
    settings.width = kSceneWidth;
    settings.height = kSceneHeight;
    SceneSource scene(context, host, files.spz, settings, false);
    if (scene.status() != SourceStatus::Ready)
    {
        results.fail("scene resize: the scene did not load");
        return;
    }
    scene.render(sceneCamera(0).toCamera(kSceneWidth, kSceneHeight), 1);
    scene.setSize(200, 100);
    const sabercat::Camera resized = sceneCamera(0).toCamera(200, 100);
    const bool resizedRendered = scene.render(resized, 1);
    const std::vector<float> image = downloadFloats(context, scene.color(), 200u * 100u * 4u);
    const size_t resizedMismatches = bitMismatches(image, scene.renderer().readImage());
    scene.setSize(200, 100);
    const bool sameSizeIdle = !scene.render(resized, 1) && scene.rendered();
    bool zeroRefused = false;
    try
    {
        scene.setSize(0, 100);
    }
    catch (const std::invalid_argument &)
    {
        zeroRefused = true;
    }
    bool oversizeRefused = false;
    try
    {
        scene.setSize(static_cast<uint32_t>(kMaxSourceSize) + 1u, 10);
    }
    catch (const std::invalid_argument &)
    {
        oversizeRefused = true;
    }
    const bool keptSize = scene.width() == 200 && scene.height() == 100;
    scene.setSize(kSceneWidth, kSceneHeight);
    scene.render(sceneCamera(0).toCamera(kSceneWidth, kSceneHeight), 1);
    const size_t backMismatches = bitMismatches(downloadFloats(context, scene.color(), kScenePixels * 4),
                                                scene.renderer().readImage());
    std::cout << strf("[selftest] scene resize: 200x100 rendered=%d mismatches=%zu; same size idle=%d; zero refused=%d "
                      "oversize refused=%d; back to 160x120 mismatches=%zu\n",
                      resizedRendered ? 1 : 0, resizedMismatches, sameSizeIdle ? 1 : 0, zeroRefused ? 1 : 0,
                      oversizeRefused ? 1 : 0, backMismatches);
    if (!resizedRendered || resizedMismatches != 0 || !sameSizeIdle || !zeroRefused || !oversizeRefused || !keptSize
        || backMismatches != 0)
    {
        results.fail("scene resize: a size change, a same-size call or a refused size misbehaves");
    }

    // The state resizes every scene and refuses bad sizes.
    const bool stateRefusesZero = !state.setRenderSize(0, 10);
    const bool stateRefusesOver = !state.setRenderSize(static_cast<uint32_t>(kMaxSourceSize) + 1u, 10);
    const bool stateResized = state.setRenderSize(200, 100);
    state.updateScenes();
    const bool stateScene = good >= 0 && state.entries()[static_cast<size_t>(good)].media->width() == 200
        && state.entries()[static_cast<size_t>(good)].media->height() == 100;
    std::cout << strf("[selftest] scene state resize: zero refused=%d oversize refused=%d resized=%d scene at 200x100=%d\n",
                      stateRefusesZero ? 1 : 0, stateRefusesOver ? 1 : 0, stateResized ? 1 : 0, stateScene ? 1 : 0);
    if (!stateRefusesZero || !stateRefusesOver || !stateResized || !stateScene)
    {
        results.fail("scene state resize: bad sizes are not refused or the scene did not resize");
    }
}

// Only the scene sources follow the camera: a camera change re-renders the visible scenes once,
// an image is untouched, and no change renders nothing.
void checkCameraDirtying(GpuContext & context, sabercat::Renderer & host, const SceneFiles & files,
                         const std::string & outDir, Results & results)
{
    const std::string imagePath = joinPath(outDir, "scene-image.png");
    writeRgbPng(imagePath, 96, 72, 31);
    CompareState state(context);
    state.setSceneHost(&host);
    state.setRenderSize(kSceneWidth, kSceneHeight);
    const int32_t image = state.addMedia(imagePath);
    const int32_t scene = state.addMedia(files.ply);
    if (image != 0 || scene != 1)
    {
        results.fail("camera dirtying: the image and the scene were not added");
        return;
    }
    auto * sceneSource = dynamic_cast<SceneSource *>(state.entries()[1].media.get());
    if (sceneSource == nullptr)
    {
        results.fail("camera dirtying: entry 1 is not a scene");
        return;
    }
    const BufferView imageColor = state.entries()[0].media->color();
    state.updateScenes();
    const uint64_t afterFirst = sceneSource->renderCount();
    state.updateScenes();
    const uint64_t afterIdle = sceneSource->renderCount();
    sabercat::app::OrbitCamera moved = state.camera();
    moved.applyDelta(0.4f, 0.1f, 0.0f);
    state.setCamera(moved);
    state.updateScenes();
    const uint64_t afterMove = sceneSource->renderCount();
    const BufferView imageAfter = state.entries()[0].media->color();
    const bool imageUntouched = imageAfter.buffer == imageColor.buffer && imageAfter.offset == imageColor.offset
        && imageAfter.size == imageColor.size;
    std::cout << strf("[selftest] camera dirtying: scene renders first=%llu idle=%llu after move=%llu; image untouched=%s\n",
                      static_cast<unsigned long long>(afterFirst), static_cast<unsigned long long>(afterIdle),
                      static_cast<unsigned long long>(afterMove), imageUntouched ? "yes" : "no");
    if (afterFirst != 1 || afterIdle != 1 || afterMove != 2 || !imageUntouched)
    {
        results.fail("camera dirtying: the camera does not re-render only the scene, or re-renders when idle");
    }
}

// The composite view of a map mode: a mode the state did not show is a failure, and each
// view at fit and at 100% must match the CPU reference.
struct MapView
{
    const char * label;
    bool split;
    SingleView single;
    bool mapUseB;
    CompositeMode mode;
};

// Host copies of the state's sources (null for a source that is not Ready).
ImagePtrs hostCopies(const CompareState & state)
{
    ImagePtrs pointers{};
    for (size_t i = 0; i < state.size(); ++i)
    {
        pointers[i] = state.entries()[i].ready() ? state.entries()[i].media->hostData() : nullptr;
    }
    return pointers;
}

void checkMapViews(const std::string & group, CompareState & state, Compositor & compositor,
                   const std::vector<MapView> & views, Results & results)
{
    state.setViewSize(static_cast<int32_t>(kViewWidth), static_cast<int32_t>(kViewHeight));
    for (const MapView & view : views)
    {
        state.setSplit(view.split);
        state.setSingleView(view.single);
        state.setMapUseB(view.mapUseB);
        for (const bool exact : {false, true})
        {
            if (exact)
            {
                setZoomPercent(state.viewLayout(), 100.0);
                centerView(state.viewLayout());
            }
            else
            {
                setZoomFit(state.viewLayout());
            }
            const std::string label = group + " " + view.label + (exact ? " 100%" : " fit");
            const CompositeInput input = state.buildComposite(0.0);
            if (input.mode != view.mode)
            {
                results.fail(label + ": the state shows composite mode " + std::to_string(static_cast<int>(input.mode))
                             + " instead of " + std::to_string(static_cast<int>(view.mode)));
                continue;
            }
            const ImagePtrs images = hostCopies(state);
            const std::vector<uint8_t> ref = referenceComposite(input, images, kViewWidth, kViewHeight);
            std::vector<uint8_t> gpu;
            try
            {
                compositor.resize(kViewWidth, kViewHeight);
                compositor.render(input);
                compositor.readback(gpu);
            }
            catch (const std::exception & e)
            {
                results.fail(label + ": render threw: " + e.what());
                continue;
            }
            compareViewBytes(label, gpu, ref, kViewWidth, kViewHeight, exact, results);
        }
    }
}

// A display image of one color, opaque.
ImageData solidImage(uint32_t width, uint32_t height, const float (&rgb)[3])
{
    ImageData image = blankImage(width, height, Encoding::Display);
    for (size_t i = 0; i < static_cast<size_t>(width) * height; ++i)
    {
        image.rgba[i * 4 + 0] = rgb[0];
        image.rgba[i * 4 + 1] = rgb[1];
        image.rgba[i * 4 + 2] = rgb[2];
        image.rgba[i * 4 + 3] = 1.0f;
    }
    return image;
}

// How the GPU divider view differs from the A/B rule, counted over the window.
struct DividerCount
{
    size_t pixels = 0;
    size_t mismatches = 0;  // pixels off by more than kMaxByteDelta
    size_t columns = 0;     // of the mismatches, those in columns floor(x) - 1 .. floor(x) + 1
};

// The divider rule, written from the divider's window x and the two rectangles alone (not from
// the reference). A pixel whose center is left of the divider shows A, else B. Inside the
// padded extent a pixel its own source covers shows that source's color; a pixel it does not
// cover shows the other source's color inverted, and a pixel neither covers is background.
DividerCount countDividerMismatches(const CompositeInput & input, const std::vector<uint8_t> & gpu, uint32_t width,
                                    uint32_t height, const float (&colorA)[3], const float (&colorB)[3])
{
    DividerCount count;
    const LayoutInput & layout = input.layout;
    if (gpu.size() != static_cast<size_t>(width) * height * 4)
    {
        return count; // the size failure is reported by compareViewBytes
    }
    const double zoom = effectiveZoom(layout);
    const double divider = dividerWindowX(layout);
    const PixelRect extent = paddedExtent(layout);
    const int32_t columnLeft = floorPixel(divider) - 1;
    const uint8_t background = byteOf(kCompositeBackground);
    for (uint32_t py = 0; py < height; ++py)
    {
        for (uint32_t px = 0; px < width; ++px)
        {
            // The single pane is the whole window, so its center is width / 2.
            const double imageX = layout.view.panX + (px + 0.5 - width * 0.5) / zoom;
            const double imageY = layout.view.panY + (py + 0.5 - height * 0.5) / zoom;
            const int32_t pixelX = floorPixel(imageX);
            const int32_t pixelY = floorPixel(imageY);
            const bool onA = static_cast<double>(px) + 0.5 < divider;
            const bool inA = rectContains(sourceRect(layout.sources[0]), pixelX, pixelY);
            const bool inB = rectContains(sourceRect(layout.sources[1]), pixelX, pixelY);
            const bool covered = onA ? inA : inB;
            const bool counterpartCovers = onA ? inB : inA;
            const float * own = onA ? colorA : colorB;
            const float * other = onA ? colorB : colorA;
            uint8_t expected[3] = {background, background, background};
            if (rectContains(extent, pixelX, pixelY) && (covered || counterpartCovers))
            {
                for (size_t c = 0; c < 3; ++c)
                {
                    expected[c] = covered ? byteOf(own[c]) : byteOf(1.0f - other[c]);
                }
            }
            const uint8_t * dst = &gpu[(static_cast<size_t>(py) * width + px) * 4];
            int delta = 0;
            for (size_t c = 0; c < 3; ++c)
            {
                delta = std::max(delta, std::abs(static_cast<int>(dst[c]) - static_cast<int>(expected[c])));
            }
            ++count.pixels;
            if (delta > kMaxByteDelta)
            {
                ++count.mismatches;
                const int32_t column = static_cast<int32_t>(px);
                if (column >= columnLeft && column <= columnLeft + 2)
                {
                    ++count.columns;
                }
            }
        }
    }
    return count;
}

// Split with exactly two visible solid sources: A red-ish on the left of the divider, B blue-ish on
// the right, at divider fractions 0, 0.25, 0.5 and 1, at Fit and at 100%. The composite must match
// referenceComposite(), and it must match the divider rule in countDividerMismatches(). The smaller
// B leaves a strip of the B side with no B pixel, which shows A inverted.
void checkDividerComposites(GpuContext & context, Compositor & compositor, Results & results)
{
    constexpr uint32_t kWidth = kViewWidth;
    constexpr uint32_t kHeight = kViewHeight;
    const float kRed[3] = {0.8f, 0.2f, 0.1f};
    const float kBlue[3] = {0.1f, 0.3f, 0.9f};
    struct Shape
    {
        const char * name;
        uint32_t aWidth;
        uint32_t aHeight;
        uint32_t bWidth;
        uint32_t bHeight;
    };
    const Shape shapes[] = {{"equal", 60, 40, 60, 40}, {"smaller-b", 60, 40, 40, 30}};
    const double fractions[] = {0.5, 0.25, 0.0, 1.0};
    for (const Shape & shape : shapes)
    {
        for (const double fraction : fractions)
        {
            for (const bool fit : {true, false})
            {
                const std::string label = strf("divider %s fraction %.2f %s", shape.name, fraction,
                                               fit ? "fit" : "100%");
                CompareState state(context);
                state.setViewSize(static_cast<int32_t>(kWidth), static_cast<int32_t>(kHeight));
                const int32_t indexA = state.addMemorySource(
                    std::make_unique<ImageSource>(context, solidImage(shape.aWidth, shape.aHeight, kRed), "divider-a"),
                    "divider-a");
                const int32_t indexB = state.addMemorySource(
                    std::make_unique<ImageSource>(context, solidImage(shape.bWidth, shape.bHeight, kBlue), "divider-b"),
                    "divider-b");
                if (indexA != 0 || indexB != 1)
                {
                    results.fail(label + ": the sources were not added as entries 0 and 1");
                    continue;
                }
                state.setDividerFraction(fraction);
                if (fit)
                {
                    setZoomFit(state.viewLayout());
                }
                else
                {
                    setZoomPercent(state.viewLayout(), 100.0);
                }
                centerView(state.viewLayout());

                const CompositeInput input = state.buildComposite(0.0);
                if (input.mode != CompositeMode::Wipe)
                {
                    results.fail(label + ": the divider composite is not the Wipe mode");
                    continue;
                }
                const ImagePtrs images = hostCopies(state);
                const std::vector<uint8_t> ref = referenceComposite(input, images, kWidth, kHeight);
                std::vector<uint8_t> gpu;
                try
                {
                    compositor.resize(kWidth, kHeight);
                    compositor.render(input);
                    compositor.readback(gpu);
                }
                catch (const std::exception & e)
                {
                    results.fail(label + ": render threw: " + e.what());
                    continue;
                }
                compareViewBytes(label, gpu, ref, kWidth, kHeight, !fit, results);

                const DividerCount count = countDividerMismatches(input, gpu, kWidth, kHeight, kRed, kBlue);
                std::cout << strf("[selftest] %s rule: pixels=%zu mismatches=%zu divider column mismatches=%zu "
                                  "divider x=%.3f\n",
                                  label.c_str(), count.pixels, count.mismatches, count.columns,
                                  dividerWindowX(input.layout));
                if (count.mismatches != 0)
                {
                    results.fail(label + ": " + std::to_string(count.mismatches)
                                 + " pixels break the divider rule (" + std::to_string(count.columns)
                                 + " in the divider columns)");
                }
            }
        }
    }
}

// Three visible sources: Split keeps the pane row, and the divider does not apply.
void checkThreeSourcePanes(GpuContext & context, Compositor & compositor, Results & results)
{
    constexpr uint32_t kWidth = kViewWidth;
    constexpr uint32_t kHeight = kViewHeight;
    const float kRed[3] = {0.8f, 0.2f, 0.1f};
    const float kBlue[3] = {0.1f, 0.3f, 0.9f};
    const float kGreen[3] = {0.2f, 0.7f, 0.3f};
    CompareState state(context);
    state.setViewSize(static_cast<int32_t>(kWidth), static_cast<int32_t>(kHeight));
    state.addMemorySource(std::make_unique<ImageSource>(context, solidImage(60, 40, kRed), "three-a"), "three-a");
    state.addMemorySource(std::make_unique<ImageSource>(context, solidImage(40, 30, kBlue), "three-b"), "three-b");
    state.addMemorySource(std::make_unique<ImageSource>(context, solidImage(50, 50, kGreen), "three-c"), "three-c");
    state.setDividerFraction(0.25);
    setZoomFit(state.viewLayout());
    centerView(state.viewLayout());

    const CompositeInput input = state.buildComposite(0.0);
    if (state.dividerActive() || input.mode != CompositeMode::Panes)
    {
        results.fail("three sources: the divider applies to three visible sources");
        return;
    }
    const std::vector<uint8_t> ref = referenceComposite(input, hostCopies(state), kWidth, kHeight);
    std::vector<uint8_t> gpu;
    try
    {
        compositor.resize(kWidth, kHeight);
        compositor.render(input);
        compositor.readback(gpu);
    }
    catch (const std::exception & e)
    {
        results.fail(std::string("three sources: render threw: ") + e.what());
        return;
    }
    compareViewBytes("three sources panes fit", gpu, ref, kWidth, kHeight, false, results);
}

// The auto depth range is the union of the valid samples of the shown A and B sources, and it
// is always finite (0..1 when there are none).
void checkAutoDepthRange(const std::string & label, const CompareState & state, Results & results)
{
    float lo = 0.0f;
    float hi = 0.0f;
    state.resolvedDepthRange(lo, hi);
    bool found = false;
    float low = 0.0f;
    float high = 0.0f;
    for (const int32_t index : {state.sourceA(), state.sourceB()})
    {
        if (index < 0)
        {
            continue;
        }
        const ImageData * host = state.entries()[static_cast<size_t>(index)].media->hostData();
        if (host == nullptr || !hasDepthSamples(*host))
        {
            continue;
        }
        for (const float d : host->depth)
        {
            if (!isValidDepth(d))
            {
                continue;
            }
            low = found ? std::min(low, d) : d;
            high = found ? std::max(high, d) : d;
            found = true;
        }
    }
    const float expectLo = found ? low : 0.0f;
    const float expectHi = found ? high : 1.0f;
    const bool finite = std::isfinite(lo) && std::isfinite(hi);
    std::cout << strf("[selftest] %s depth range auto: %.9g..%.9g (cpu %.9g..%.9g) finite=%d\n", label.c_str(), lo, hi,
                      expectLo, expectHi, finite ? 1 : 0);
    if (!finite || lo != expectLo || hi != expectHi)
    {
        results.fail(label + ": the auto depth range is not the union of the valid samples");
    }
}

// The probe through the layout the UI uses. Edges and interior samples of each image read
// the exact pixel (color, and for a scene the four depth components); padding and outside
// pixels read nothing.
void checkProbe(GpuContext & context, CompareState & state, Results & results)
{
    (void)context;
    constexpr uint32_t kWidth = 800;
    constexpr uint32_t kHeight = 500;
    state.setSplit(true);
    state.setViewSize(static_cast<int32_t>(kWidth), static_cast<int32_t>(kHeight));
    setZoomPercent(state.viewLayout(), 100.0);
    centerView(state.viewLayout());
    // The pane probe is a layout rule: use the pane form (a two-source split is one divider pane in the state).
    LayoutInput layout = state.layout();
    layout.view.mode = ViewMode::SplitScreen;

    auto * scene = dynamic_cast<SceneSource *>(state.entries()[0].media.get());
    if (scene == nullptr)
    {
        results.fail("probe: entry 0 is not a scene");
        return;
    }
    const std::vector<float> sceneColor = scene->renderer().readImage();
    const std::vector<float> sceneDepth = scene->renderer().readDepth();
    const ImageData * imageHost = state.entries()[1].media->hostData();
    if (imageHost == nullptr)
    {
        results.fail("probe: the image has no host copy");
        return;
    }
    const MediaSource & image = *state.entries()[1].media;

    size_t hits = 0;
    size_t checked = 0;
    size_t edgeChecks = 0;
    size_t mismatches = 0;
    size_t paddingProbes = 0;
    size_t paddingOk = 0;
    size_t outsideProbes = 0;
    size_t outsideOk = 0;
    size_t sceneHits = 0;
    size_t imageHits = 0;
    for (int32_t wy = 0; wy < static_cast<int32_t>(kHeight); ++wy)
    {
        for (int32_t wx = 0; wx < static_cast<int32_t>(kWidth); ++wx)
        {
            const ProbeTarget target = probeTarget(layout, wx, wy);
            const PanePixelClass cls = classifyPanePixel(layout, wx, wy);
            if (cls.kind == PixelKind::Padding)
            {
                // The pane's source has no pixel here: the probe names no source, and readPixel
                // of the pane's source at that pixel is false.
                ++paddingProbes;
                const ImageHit hit = windowToImage(layout, wx, wy);
                const LayoutSource & g = layout.sources[static_cast<size_t>(hit.source)];
                float px[4] = {};
                const bool read = state.entries()[static_cast<size_t>(hit.source)].media->readPixel(
                    hit.pixelX - g.offsetX, hit.pixelY - g.offsetY, px);
                if (target.source < 0 && !read)
                {
                    ++paddingOk;
                }
                continue;
            }
            if (cls.kind == PixelKind::Outside || cls.pane < 0)
            {
                ++outsideProbes;
                bool none = target.source < 0;
                if (cls.pane >= 0)
                {
                    const ImageHit hit = windowToImage(layout, wx, wy);
                    for (size_t i = 0; i < 2; ++i)
                    {
                        const LayoutSource & g = layout.sources[i];
                        float px[4] = {};
                        none = none && !state.entries()[i].media->readPixel(hit.pixelX - g.offsetX,
                                                                             hit.pixelY - g.offsetY, px);
                    }
                }
                outsideOk += none ? 1 : 0;
                continue;
            }
            if (target.source < 0)
            {
                ++mismatches; // an image pixel that the probe does not name
                continue;
            }

            ++hits;
            const size_t src = static_cast<size_t>(target.source);
            const MediaSource & media = *state.entries()[src].media;
            const int32_t w = static_cast<int32_t>(media.width());
            const int32_t h = static_cast<int32_t>(media.height());
            const bool edge = (target.localX == 0 || target.localX == w - 1)
                && (target.localY == 0 || target.localY == h - 1);
            if (!edge && hits % 199 != 0)
            {
                continue;
            }
            ++checked;
            edgeChecks += edge ? 1 : 0;
            (src == 0 ? sceneHits : imageHits) += 1;

            const size_t index = (static_cast<size_t>(target.localY) * static_cast<size_t>(w)
                                  + static_cast<size_t>(target.localX));
            float px[4] = {};
            bool bad = !media.readPixel(target.localX, target.localY, px);
            const float * expect = src == 0 ? &sceneColor[index * 4] : &imageHost->rgba[index * 4];
            bad = bad || std::memcmp(px, expect, sizeof(px)) != 0;
            if (src == 0)
            {
                float raw[4] = {};
                uint32_t count = 0;
                bad = bad || !scene->readDepthRaw(target.localX, target.localY, raw, count) || count != 4
                    || std::memcmp(raw, &sceneDepth[index * 4], sizeof(raw)) != 0;
                float front = 0.0f;
                bad = bad || !scene->readDepth(target.localX, target.localY, front) || !sameBytes(front, raw[2]);
            }
            else
            {
                float raw[4] = {};
                uint32_t count = 0;
                bad = bad || image.readDepthRaw(target.localX, target.localY, raw, count) || count != 0;
            }
            mismatches += bad ? 1 : 0;
        }
    }
    std::cout << strf("[selftest] probe: %zu pixels checked (%zu at corners, scene %zu, image %zu) mismatches=%zu; "
                      "padding %zu of %zu name no source; outside %zu of %zu read nothing\n",
                      checked, edgeChecks, sceneHits, imageHits, mismatches, paddingOk, paddingProbes, outsideOk,
                      outsideProbes);
    if (mismatches != 0 || paddingOk != paddingProbes || outsideOk != outsideProbes || edgeChecks < 4
        || sceneHits == 0 || imageHits == 0 || paddingProbes == 0 || outsideProbes == 0)
    {
        results.fail("probe: a probed pixel differs from the host or readbacks, or a padding/outside probe reads a value");
    }
}

// The composites of scene and image sources, the map modes, the auto range, and the probe.
void checkSceneComposites(GpuContext & context, sabercat::Renderer & host, const SceneFiles & files,
                          const std::string & outDir, Compositor & compositor, Results & results)
{
    const std::string imagePath = joinPath(outDir, "scene-image.png");
    writeRgbPng(imagePath, 96, 72, 31);
    const std::vector<MapView> sceneViews = {
        {"split-divider", true, SingleView::A, false, CompositeMode::Wipe},
        {"single-a", false, SingleView::A, false, CompositeMode::Panes},
        {"wipe", false, SingleView::Wipe, false, CompositeMode::Wipe},
        {"diff", false, SingleView::Diff, false, CompositeMode::Diff},
        {"depth-a-image", false, SingleView::Depth, false, CompositeMode::Depth},
        {"depth-b-scene", false, SingleView::Depth, true, CompositeMode::Depth},
        {"alpha-a-image", false, SingleView::Alpha, false, CompositeMode::Alpha},
        {"alpha-b-scene", false, SingleView::Alpha, true, CompositeMode::Alpha},
        {"depth-diff", false, SingleView::DepthDiff, false, CompositeMode::DepthDiff},
        {"alpha-diff", false, SingleView::AlphaDiff, false, CompositeMode::AlphaDiff},
    };

    // Scene A (160x120) and image B (96x72, no depth), offset so the padded extent has both.
    {
        CompareState state(context);
        state.setSceneHost(&host);
        state.setRenderSize(kSceneWidth, kSceneHeight);
        const int32_t a = state.addMedia(files.ply);
        const int32_t b = state.addMedia(imagePath);
        if (a != 0 || b != 1 || !state.entries()[0].ready() || !state.entries()[1].ready())
        {
            results.fail("scene composites: the scene and the image were not added");
            return;
        }
        state.setOffset(1, 120, 40);
        state.updateScenes();
        checkAutoDepthRange("scene + image", state, results);
        checkMapViews("scene+image", state, compositor, sceneViews, results);
        checkProbe(context, state, results);
    }

    // A source with NaN, infinite, zero and negative depth next to the scene.
    {
        CompareState state(context);
        state.setSceneHost(&host);
        state.setRenderSize(kSceneWidth, kSceneHeight);
        state.addMemorySource(std::make_unique<ImageSource>(context, makeDepthImage(64, 48, 41, false), "nan-depth"),
                              "nan-depth");
        state.addMedia(files.ply);
        if (state.size() != 2 || !state.entries()[0].ready() || !state.entries()[1].ready())
        {
            results.fail("scene composites: the NaN-depth source and the scene were not added");
            return;
        }
        state.setOffset(1, 30, 10);
        state.updateScenes();
        checkAutoDepthRange("nan-depth + scene", state, results);
        checkMapViews("nan-depth+scene", state, compositor,
                      {{"depth-a", false, SingleView::Depth, false, CompositeMode::Depth},
                       {"depth-b", false, SingleView::Depth, true, CompositeMode::Depth},
                       {"depth-diff", false, SingleView::DepthDiff, false, CompositeMode::DepthDiff},
                       {"alpha-a", false, SingleView::Alpha, false, CompositeMode::Alpha},
                       {"alpha-diff", false, SingleView::AlphaDiff, false, CompositeMode::AlphaDiff},
                       {"split-divider", true, SingleView::A, false, CompositeMode::Wipe}},
                      results);
    }

    // A source whose depth has no valid sample: the auto range is 0..1 and the map is hatched.
    {
        CompareState state(context);
        state.setSceneHost(&host);
        state.setRenderSize(kSceneWidth, kSceneHeight);
        state.addMemorySource(std::make_unique<ImageSource>(context, makeDepthImage(40, 30, 43, true), "no-valid-depth"),
                              "no-valid-depth");
        if (!state.entries()[0].ready())
        {
            results.fail("scene composites: the no-valid-depth source was not added");
            return;
        }
        checkAutoDepthRange("no valid depth", state, results);
        checkMapViews("no-valid-depth", state, compositor,
                      {{"depth-a", false, SingleView::Depth, false, CompositeMode::Depth},
                       {"alpha-a", false, SingleView::Alpha, false, CompositeMode::Alpha}},
                      results);
    }
}

// A scene's memory budget: one scene is measured, a second is refused under a small override
// with the message, and four real scenes are each accepted or refused with the message.
void checkSceneBudget(GpuContext & context, sabercat::Renderer & host, const SceneFiles & files, Results & results)
{
    sabercat::Device & device = host.sabercatDevice();
    CompareState state(context);
    state.setSceneHost(&host);
    state.setRenderSize(kSceneWidth, kSceneHeight);
    const int32_t first = state.addMedia(files.ply);
    const auto * firstScene = first >= 0 ? dynamic_cast<SceneSource *>(state.entries()[0].media.get()) : nullptr;
    const uint64_t sceneBytes = firstScene != nullptr ? firstScene->deviceBytes() : 0;
    const uint64_t hostFree = deviceLocalFreeBytes(device);
    const uint64_t freeBytes = hostFree > sceneBytes ? hostFree - sceneBytes : 0;

    setSceneBudgetOverride(std::optional<uint64_t>(1));
    const int32_t second = state.addMedia(files.spz);
    const bool refused = second == -1 && state.status().find("Not enough GPU memory") != std::string::npos;
    const std::string refusal = state.status();
    setSceneBudgetOverride(std::nullopt);
    std::cout << strf("[selftest] scene budget: one scene allocated=%llu bytes (%.1f MiB), free after it=%.1f MiB; "
                      "second scene under a 1-byte override: index=%d \"%s\"\n",
                      static_cast<unsigned long long>(sceneBytes), static_cast<double>(sceneBytes) / 1048576.0,
                      static_cast<double>(freeBytes) / 1048576.0, second, refusal.c_str());
    if (first != 0 || !refused)
    {
        results.fail("scene budget: the override does not refuse a second scene with the message");
    }

    // Four scenes with the real budget: each is accepted or refused with the message.
    CompareState many(context);
    many.setSceneHost(&host);
    many.setRenderSize(kSceneWidth, kSceneHeight);
    std::string which;
    bool messagesOk = true;
    for (int i = 0; i < 4; ++i)
    {
        const int32_t index = many.addMedia(files.ply);
        if (index >= 0)
        {
            which += "accepted ";
        }
        else
        {
            which += "refused ";
            messagesOk = messagesOk && many.status().find("Not enough GPU memory") != std::string::npos;
        }
    }
    std::cout << strf("[selftest] scene budget: four scenes with the real budget: %s\n", which.c_str());
    if (!messagesOk)
    {
        results.fail("scene budget: a refused scene did not give the memory message");
    }
}

// A project keeps the scene settings, the camera and the map view; a view change marks it dirty;
// a project without the new keys still opens.
void checkSceneProject(GpuContext & context, sabercat::Renderer & host, const SceneFiles & files,
                       const std::string & outDir, Results & results)
{
    const std::string imagePath = joinPath(outDir, "scene-image.png");
    const std::string projectPath = joinPath(outDir, "scene-roundtrip.sabercat-compare");
    CompareState source(context);
    source.setSceneHost(&host);
    source.setRenderSize(kSceneWidth, kSceneHeight);
    source.addMedia(files.ply);
    source.addMedia(imagePath);
    source.setShCap(0, 0);
    source.setFlipUpAxis(0, true);
    source.setAntialias(0, true);
    source.setDebugView(0, sabercat::DebugView::Depth);
    sabercat::app::OrbitCamera camera = source.camera();
    camera.pivot = glm::vec3(0.125f, -0.2f, 0.3f);
    camera.applyDelta(0.3f, -0.2f, 0.1f);
    camera.distance = 2.5f;
    source.setCamera(camera);
    source.setSplit(false);
    source.setSingleView(SingleView::Depth);
    source.setMapUseB(true);
    source.setDepthAuto(false);
    source.setDepthRange(0.5f, 3.0f);
    std::string error;
    const bool saved = source.saveProject(projectPath, error);

    CompareState loaded(context);
    loaded.setSceneHost(&host);
    const bool opened = loaded.openProject(projectPath, error);
    const sabercat::app::OrbitCamera & restored = loaded.camera();
    const bool cameraExact = sameBytes(restored.pivot, camera.pivot) && sameBytes(restored.orientation, camera.orientation)
        && sameBytes(restored.distance, camera.distance);
    const bool sceneSettings = loaded.size() == 2 && loaded.entries()[0].shCap == 0
        && loaded.entries()[0].flipUpAxis && loaded.entries()[0].antialias
        && loaded.entries()[0].debugView == sabercat::DebugView::Depth;
    const ViewSettings & view = loaded.settings();
    const bool viewSettings = view.single == SingleView::Depth && !view.split && view.mapUseB && !view.depthAuto
        && view.depthMin == 0.5f && view.depthMax == 3.0f;
    const bool renderSize = loaded.renderWidth() == kSceneWidth && loaded.renderHeight() == kSceneHeight;
    const bool cleanAfterOpen = !loaded.projectDirty();
    setWipeFraction(loaded.viewLayout(), 0.3);
    loaded.viewChanged();
    const bool dirtyAfterWipe = loaded.projectDirty();
    std::cout << strf("[selftest] scene project: saved=%d opened=%d camera exact=%d scene settings=%d view=%d "
                      "render size=%d clean after open=%d dirty after wipe=%d%s\n",
                      saved ? 1 : 0, opened ? 1 : 0, cameraExact ? 1 : 0, sceneSettings ? 1 : 0, viewSettings ? 1 : 0,
                      renderSize ? 1 : 0, cleanAfterOpen ? 1 : 0, dirtyAfterWipe ? 1 : 0,
                      error.empty() ? "" : (" error=" + error).c_str());
    if (!saved || !opened || !cameraExact || !sceneSettings || !viewSettings || !renderSize || !cleanAfterOpen
        || !dirtyAfterWipe)
    {
        results.fail("scene project: the round trip loses a scene setting, the camera or the view");
    }

    // The same project without the keys this change added (singleView, mapSource, depthAuto and
    // the scene keys) still opens, with the defaults.
    std::string text;
    {
        std::ifstream in(pathFromUtf8(projectPath), std::ios::binary);
        std::ostringstream buffer;
        buffer << in.rdbuf();
        text = buffer.str();
    }
    const std::vector<std::string> newKeys = {"\"singleView\"", "\"mapSource\"", "\"depthAuto\"", "\"flipUpAxis\"",
                                              "\"shCap\"", "\"antialias\"", "\"debugView\""};
    std::vector<std::string> kept;
    std::istringstream lines(text);
    for (std::string line; std::getline(lines, line);)
    {
        const size_t first = line.find_first_not_of(" \t");
        const bool drop = first != std::string::npos
            && std::any_of(newKeys.begin(), newKeys.end(),
                           [&line, first](const std::string & key) { return line.compare(first, key.size(), key) == 0; });
        if (!drop)
        {
            kept.push_back(line);
        }
    }
    // Dropping a last member leaves a comma before the closing bracket: remove it.
    for (size_t i = 0; i + 1 < kept.size(); ++i)
    {
        const size_t next = kept[i + 1].find_first_not_of(" \t");
        if (!kept[i].empty() && kept[i].back() == ',' && next != std::string::npos
            && (kept[i + 1][next] == '}' || kept[i + 1][next] == ']'))
        {
            kept[i].pop_back();
        }
    }
    std::string oldText;
    for (const std::string & line : kept)
    {
        oldText += line + "\n";
    }
    const std::string oldPath = joinPath(outDir, "scene-old.sabercat-compare");
    {
        std::ofstream out(pathFromUtf8(oldPath), std::ios::binary);
        out << oldText;
    }
    CompareState old(context);
    old.setSceneHost(&host);
    std::string oldError;
    const bool oldOpened = old.openProject(oldPath, oldError);
    const bool oldDefaults = oldOpened && old.settings().single == SingleView::A && !old.settings().mapUseB
        && old.settings().depthAuto && old.size() == 2 && old.entries()[0].shCap == -1;
    std::cout << strf("[selftest] scene project without the new keys: opened=%d defaults=%d%s\n", oldOpened ? 1 : 0,
                      oldDefaults ? 1 : 0, oldError.empty() ? "" : (" error=" + oldError).c_str());
    if (!oldDefaults)
    {
        results.fail("scene project: a project without the new keys does not open with the defaults: " + oldError);
    }
}

// The debug views: the display names are the names of debugViewName(), and the keys round-trip.
void checkDebugViewKeys(Results & results)
{
    bool ok = true;
    for (const sabercat::app::DebugViewInfo & info : sabercat::app::kDebugViews)
    {
        sabercat::DebugView parsed = sabercat::DebugView::Final;
        const sabercat::app::DebugViewInfo * found = sabercat::app::findDebugView(info.name);
        const bool named = std::strcmp(info.name, sabercat::app::debugViewName(info.view)) == 0;
        const bool keyed = sabercat::app::parseDebugViewKey(sabercat::app::debugViewKey(info.view), parsed)
            && parsed == info.view;
        const bool lookup = found != nullptr && found->view == info.view;
        ok = ok && named && keyed && lookup && info.needsDepth == (info.view == sabercat::DebugView::Depth);
        std::cout << strf("[selftest] debug view %s: key %s name=%d key=%d lookup=%d\n", info.name,
                          sabercat::app::debugViewKey(info.view), named ? 1 : 0, keyed ? 1 : 0, lookup ? 1 : 0);
    }
    sabercat::DebugView none = sabercat::DebugView::Final;
    ok = ok && sabercat::app::findDebugView("no such view") == nullptr
        && !sabercat::app::parseDebugViewKey("no such key", none);
    if (!ok)
    {
        results.fail("debug views: a name or a key does not match debugViewName() or does not round-trip");
    }
}

// Sets the scene budget override (the free bytes sceneBudgetOk() uses) for one scope.
class BudgetOverrideScope
{
  public:
    explicit BudgetOverrideScope(uint64_t freeBytes)
    {
        setSceneBudgetOverride(std::optional<uint64_t>(freeBytes));
    }
    ~BudgetOverrideScope()
    {
        setSceneBudgetOverride(std::nullopt);
    }
    BudgetOverrideScope(const BudgetOverrideScope &) = delete;
    BudgetOverrideScope & operator=(const BudgetOverrideScope &) = delete;
};

// The CPU FLIP over the overlap of a state's A and B (the host copies, at the state's offsets).
FlipResult cpuFlipOfState(const CompareState & state, float ppd, PixelRect & overlap)
{
    const LayoutInput & layout = state.layout();
    overlap = overlapRect(layout, 0, 1);
    const ImageData * hostA = state.entries()[0].media->hostData();
    const ImageData * hostB = state.entries()[1].media->hostData();
    if (hostA == nullptr || hostB == nullptr)
    {
        throw std::logic_error("the state's A or B has no host copy");
    }
    const ImageData cropA = cropImage(*hostA, overlap.x - layout.sources[0].offsetX,
                                      overlap.y - layout.sources[0].offsetY, overlap.width, overlap.height);
    const ImageData cropB = cropImage(*hostB, overlap.x - layout.sources[1].offsetX,
                                      overlap.y - layout.sources[1].offsetY, overlap.width, overlap.height);
    return flip(displayRgb(cropA), displayRgb(cropB), ppd);
}

// ---- Live rendering: the frame clock, the live policy, scene files and the budget ----
//
// The live checks drive a CompareState on a synthetic clock, the way the app drives it: beginFrame,
// updateScenes, an interactive camera move (the UI's orbit) with updateScenes again, then
// updateMetrics. Each item is one function, registered in runSceneChecks.

// The synthetic clock: 60 frames a second, one beginFrame per frame, so frame i is at i / 60 s.
double liveTime(int frame)
{
    return static_cast<double>(frame) / 60.0;
}

// Frames at rest after a drag: 13 frames is 0.22 s, past kLiveIdleSeconds (0.15 s).
constexpr int kLiveRestFrames = 13;

// The count as the unsigned long long that %llu reads.
unsigned long long asULL(uint64_t value)
{
    return static_cast<unsigned long long>(value);
}

// One frame in the app's order. Returns false when the camera refused the move.
bool liveFrame(CompareState & state, MetricsGpu & metrics, int frame, const sabercat::app::OrbitCamera * move)
{
    state.beginFrame(liveTime(frame));
    state.updateScenes();
    bool accepted = true;
    if (move != nullptr)
    {
        accepted = state.setCamera(*move, true);
        state.updateScenes();
    }
    state.updateMetrics(metrics);
    return accepted;
}

// The camera after one orbit step of a drag (the step the smoke hook takes).
sabercat::app::OrbitCamera orbitStep(const CompareState & state)
{
    sabercat::app::OrbitCamera camera = state.camera();
    camera.applyDelta(0.02f, 0.004f, 0.0f);
    return camera;
}

// The entry as a SceneSource (null for an image, or an index past the end).
SceneSource * sceneAt(CompareState & state, size_t index)
{
    if (index >= state.size())
    {
        return nullptr;
    }
    return dynamic_cast<SceneSource *>(state.entries()[index].media.get());
}

// The renderCount of a scene entry (0 for anything else).
uint64_t rendersOf(CompareState & state, size_t index)
{
    const SceneSource * scene = sceneAt(state, index);
    return scene != nullptr ? scene->renderCount() : 0;
}

// A state with the synthetic .ply as entry 0 and the .spz as entry 1, both at the scene size and
// Ready. Returns false when either did not load.
bool openScenePair(CompareState & state, sabercat::Renderer & host, const SceneFiles & files)
{
    state.setSceneHost(&host);
    state.setRenderSize(kSceneWidth, kSceneHeight);
    return state.addMedia(files.ply) == 0 && state.addMedia(files.spz) == 1 && state.entries()[0].ready()
        && state.entries()[1].ready();
}

// The scene the live checks render: SABERCAT_COMPARE_LIVE_SCENE when it names a file, else the
// synthetic .ply.
std::string liveScenePath(const SceneFiles & files)
{
    const char * env = std::getenv("SABERCAT_COMPARE_LIVE_SCENE");
    if (env != nullptr && env[0] != '\0')
    {
        std::error_code ec;
        if (fs::is_regular_file(pathFromUtf8(env), ec))
        {
            return env;
        }
        std::cout << "[selftest] live: SABERCAT_COMPARE_LIVE_SCENE is not a file; using the synthetic scene\n";
    }
    return files.ply;
}

// Frames of the performance check: SABERCAT_COMPARE_LIVE_FRAMES, never fewer than 120.
int liveFrameCount()
{
    const char * text = std::getenv("SABERCAT_COMPARE_LIVE_FRAMES");
    const long value = text != nullptr ? std::strtol(text, nullptr, 10) : 0L;
    return static_cast<int>(std::min(std::max(value, 120L), 1000000L));
}

// Milliseconds since a steady clock time point.
double msSince(std::chrono::steady_clock::time_point start)
{
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

// The value at a fraction of the sorted values (0 when there are none). The values are sorted in
// place, and 0.5 is the lower median for an even count.
double percentileOf(std::vector<double> & values, double fraction)
{
    if (values.empty())
    {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    return values[static_cast<size_t>(fraction * static_cast<double>(values.size() - 1))];
}

// A .ply of the synthetic cloud with count Gaussians.
void writeSyntheticPly(sabercat::Renderer & host, const std::string & path, size_t count)
{
    sabercat::GaussianCloud cloud(host.sabercatDevice());
    buildSyntheticCloud(cloud, count);
    sabercat::PLYWriter::write(cloud, path);
}

// Sets a file's write time to now plus seconds. The scene poll sees a new stamp, whatever the content.
void setWriteTimeAhead(const std::string & path, int seconds)
{
    std::error_code ec;
    fs::last_write_time(pathFromUtf8(path), fs::file_time_type::clock::now() + std::chrono::seconds(seconds), ec);
    if (ec)
    {
        throw std::runtime_error("could not set the write time: " + ec.message());
    }
}

// ---- Live policy items -------------------------------------------------------------

// Idle frames render nothing: with a scene and an image at rest, 30 frames add no scene render, no
// metrics compute and no FLIP compute. The first frame renders the scene once.
void checkLiveIdle(GpuContext & context, sabercat::Renderer & host, const SceneFiles & files,
                   const std::string & outDir, MetricsGpu & metrics, Results & results)
{
    const std::string imagePath = joinPath(outDir, "scene-image.png");
    writeRgbPng(imagePath, 96, 72, 31);
    CompareState state(context);
    state.setSceneHost(&host);
    state.setRenderSize(kSceneWidth, kSceneHeight);
    const int32_t scene = state.addMedia(files.ply);
    const int32_t image = state.addMedia(imagePath);
    if (scene != 0 || image != 1 || !state.entries()[0].ready() || !state.entries()[1].ready())
    {
        results.fail("live idle: the scene and the image were not added");
        return;
    }
    int frame = 0;
    liveFrame(state, metrics, frame++, nullptr);
    const uint64_t firstRenders = rendersOf(state, 0);
    const uint64_t rendersBefore = state.sceneRenderTotal();
    const uint64_t metricsBefore = state.metricsComputeCount();
    const uint64_t flipBefore = state.flipComputeCount();
    for (int k = 0; k < 30; ++k)
    {
        liveFrame(state, metrics, frame++, nullptr);
    }
    const uint64_t renders = state.sceneRenderTotal() - rendersBefore;
    const uint64_t computes = state.metricsComputeCount() - metricsBefore;
    const uint64_t flips = state.flipComputeCount() - flipBefore;
    std::cout << strf("[selftest] live idle: 30 idle frames: scene renders +%llu, metrics computes +%llu, FLIP "
                      "computes +%llu; the first frame rendered the scene %llu time(s)\n",
                      asULL(renders), asULL(computes), asULL(flips), asULL(firstRenders));
    if (firstRenders != 1 || renders != 0 || computes != 0 || flips != 0)
    {
        results.fail("live idle: a frame at rest rendered or computed something");
    }
}

// A drag re-renders each visible scene once per frame and nothing else: the hidden scene and the
// image stay as they were. After the drag the composite equals the CPU reference.
void checkLiveMoves(GpuContext & context, sabercat::Renderer & host, const SceneFiles & files,
                    const std::string & outDir, Compositor & compositor, MetricsGpu & metrics, Results & results)
{
    const std::string imagePath = joinPath(outDir, "scene-image.png");
    writeRgbPng(imagePath, 96, 72, 31);
    CompareState state(context);
    state.setSceneHost(&host);
    state.setRenderSize(kSceneWidth, kSceneHeight);
    const int32_t a = state.addMedia(files.ply);
    const int32_t b = state.addMedia(files.spz);
    const int32_t image = state.addMedia(imagePath);
    const int32_t hidden = state.addMedia(files.ply);
    if (a != 0 || b != 1 || image != 2 || hidden != 3 || !state.setVisible(3, false))
    {
        results.fail("live moves: the sources were not added or the hidden scene did not hide");
        return;
    }
    int frame = 0;
    liveFrame(state, metrics, frame++, nullptr);
    const uint64_t hiddenBefore = rendersOf(state, 3);
    const uint64_t aBefore = rendersOf(state, 0);
    const uint64_t bBefore = rendersOf(state, 1);
    const BufferView imageBefore = state.entries()[2].media->color();
    uint64_t lastA = aBefore;
    uint64_t lastB = bBefore;
    int offByOne = 0;
    for (int k = 0; k < 12; ++k)
    {
        const sabercat::app::OrbitCamera cam = orbitStep(state);
        liveFrame(state, metrics, frame++, &cam);
        const uint64_t nowA = rendersOf(state, 0);
        const uint64_t nowB = rendersOf(state, 1);
        offByOne += (nowA - lastA != 1 || nowB - lastB != 1) ? 1 : 0;
        lastA = nowA;
        lastB = nowB;
    }
    const uint64_t movedA = lastA - aBefore;
    const uint64_t movedB = lastB - bBefore;
    const uint64_t hiddenDelta = rendersOf(state, 3) - hiddenBefore;
    const BufferView imageAfter = state.entries()[2].media->color();
    const bool imageUntouched = imageAfter.buffer == imageBefore.buffer && imageAfter.offset == imageBefore.offset
        && imageAfter.size == imageBefore.size;
    std::cout << strf("[selftest] live moves: 12 drag frames: scene A +%llu, scene B +%llu renders (frames off by "
                      "one=%d); hidden scene +%llu; image untouched=%s\n",
                      asULL(movedA), asULL(movedB), offByOne, asULL(hiddenDelta), imageUntouched ? "yes" : "no");
    if (movedA != 12 || movedB != 12 || offByOne != 0 || hiddenDelta != 0 || !imageUntouched)
    {
        results.fail("live moves: a visible scene did not render once per frame, or the hidden scene or the image moved");
    }

    // The composite at rest (Panes: scene A and scene B) against the CPU reference.
    const std::vector<MapView> panes = {{"panes", true, SingleView::A, false, CompositeMode::Panes}};
    checkMapViews("live moved", state, compositor, panes, results);
}

// The metrics and FLIP are deferred while the camera moves: a deferred frame is pending, and a compute
// runs at most every 1/kDefaultLiveFlipHz seconds. At rest, one frame computes them, and they equal the
// CPU on the downloaded buffers.
void checkLiveFinal(GpuContext & context, sabercat::Renderer & host, const SceneFiles & files, MetricsGpu & metrics,
                    Results & results)
{
    CompareState state(context);
    if (!openScenePair(state, host, files))
    {
        results.fail("live final: the scenes did not load");
        return;
    }
    int frame = 0;
    liveFrame(state, metrics, frame++, nullptr);
    int deferred = 0;
    int computed = 0;
    bool pendingMatches = true;
    for (int k = 0; k < 30; ++k)
    {
        const uint64_t before = state.metricsComputeCount();
        const sabercat::app::OrbitCamera cam = orbitStep(state);
        liveFrame(state, metrics, frame++, &cam);
        const bool ran = state.metricsComputeCount() != before;
        // A frame that computed is not pending, and a deferred one is.
        pendingMatches = pendingMatches && (ran != state.metricsPending());
        if (ran)
        {
            ++computed;
        }
        else
        {
            ++deferred;
        }
    }
    frame += kLiveRestFrames;
    const uint64_t beforeRest = state.metricsComputeCount();
    liveFrame(state, metrics, frame++, nullptr);
    const uint64_t restComputes = state.metricsComputeCount() - beforeRest;
    const bool settled = !state.metricsPending();

    const ImageData * hostA = state.entries()[0].media->hostData();
    const ImageData * hostB = state.entries()[1].media->hostData();
    if (!state.metrics() || hostA == nullptr || hostB == nullptr)
    {
        results.fail("live final: no metrics or no host copies: " + state.metricsReason());
        return;
    }
    checkMetricNumbers("live final", *state.metrics(), psnr(displayRgb(*hostA), displayRgb(*hostB)), results);
    PixelRect overlap;
    const FlipResult cpuFlip = cpuFlipOfState(state, defaultFlipPpd(), overlap);
    const bool flipShown = state.flip().has_value();
    const double gpuFlip = flipShown ? state.flip()->mean : -1.0;
    const double flipDelta = flipShown ? std::fabs(gpuFlip - cpuFlip.meanError) : std::numeric_limits<double>::infinity();
    std::cout << strf("[selftest] live final: drag frames deferred=%d computed=%d (pending on each deferred frame=%s); "
                      "rest frame computes +%llu, pending after=%s; FLIP gpu=%.9g cpu=%.9g dmean=%.3g\n",
                      deferred, computed, pendingMatches ? "yes" : "no", asULL(restComputes), settled ? "no" : "yes",
                      gpuFlip, cpuFlip.meanError, flipDelta);
    if (deferred < 1 || computed < 1 || !pendingMatches || restComputes != 1 || !settled || !flipShown
        || !(flipDelta <= kMaxFlipMeanDelta))
    {
        results.fail("live final: deferred frames, the final compute, or the FLIP does not match the policy or the CPU");
    }
}

// The FLIP compute rate while dragging: a 3 s drag (180 frames) computes at most rate * 3 + 1 times and
// at least once, and the first rest frame adds exactly one. Checked at the default rate and at 1 Hz.
void checkLiveFlipRate(GpuContext & context, sabercat::Renderer & host, const SceneFiles & files,
                       MetricsGpu & metrics, Results & results)
{
    for (const float hz : {kDefaultLiveFlipHz, 1.0f})
    {
        CompareState state(context);
        if (!openScenePair(state, host, files))
        {
            results.fail("live flip rate: the scenes did not load");
            return;
        }
        if (hz != kDefaultLiveFlipHz && !state.setLiveFlipHz(hz))
        {
            results.fail("live flip rate: the rate " + strf("%g", static_cast<double>(hz)) + " was refused");
            return;
        }
        int frame = 0;
        liveFrame(state, metrics, frame++, nullptr);
        const uint64_t before = state.flipComputeCount();
        for (int k = 0; k < 180; ++k)
        {
            const sabercat::app::OrbitCamera cam = orbitStep(state);
            liveFrame(state, metrics, frame++, &cam);
        }
        const uint64_t dragComputes = state.flipComputeCount() - before;
        const uint64_t limit = static_cast<uint64_t>(std::ceil(state.liveFlipHz() * 3.0f)) + 1;
        frame += kLiveRestFrames;
        const uint64_t beforeRest = state.flipComputeCount();
        liveFrame(state, metrics, frame++, nullptr);
        const uint64_t restComputes = state.flipComputeCount() - beforeRest;
        std::cout << strf("[selftest] live flip rate %g Hz: 3 s drag computes=%llu (at least 1, at most %llu); the "
                          "first rest frame adds %llu\n",
                          static_cast<double>(state.liveFlipHz()), asULL(dragComputes), asULL(limit),
                          asULL(restComputes));
        if (dragComputes < 1 || dragComputes > limit || restComputes != 1)
        {
            results.fail(strf("live flip rate %g Hz: the drag computed %llu times, the rest frame %llu",
                              static_cast<double>(state.liveFlipHz()), asULL(dragComputes), asULL(restComputes)));
        }
    }
}

// Live off: a drag renders no scene and computes no metrics, and the metrics stay as they were. The
// first rest frame renders each visible scene once and computes the metrics once, equal to the CPU.
void checkLiveOff(GpuContext & context, sabercat::Renderer & host, const SceneFiles & files, MetricsGpu & metrics,
                  Results & results)
{
    CompareState state(context);
    if (!openScenePair(state, host, files))
    {
        results.fail("live off: the scenes did not load");
        return;
    }
    int frame = 0;
    liveFrame(state, metrics, frame++, nullptr);
    const std::optional<MetricsResult> before = state.metrics();
    if (!before)
    {
        results.fail("live off: no metrics before the drag: " + state.metricsReason());
        return;
    }
    state.setLive(false);
    const uint64_t rendersBefore = state.sceneRenderTotal();
    const uint64_t computesBefore = state.metricsComputeCount();
    for (int k = 0; k < 30; ++k)
    {
        const sabercat::app::OrbitCamera cam = orbitStep(state);
        liveFrame(state, metrics, frame++, &cam);
    }
    const uint64_t dragRenders = state.sceneRenderTotal() - rendersBefore;
    const uint64_t dragComputes = state.metricsComputeCount() - computesBefore;
    const bool kept = state.metrics().has_value() && sameBytes(before->mse, state.metrics()->mse)
        && sameBytes(before->psnrDb, state.metrics()->psnrDb);

    frame += kLiveRestFrames;
    const uint64_t restA0 = rendersOf(state, 0);
    const uint64_t restB0 = rendersOf(state, 1);
    const uint64_t restComputes0 = state.metricsComputeCount();
    liveFrame(state, metrics, frame++, nullptr);
    const uint64_t restA = rendersOf(state, 0) - restA0;
    const uint64_t restB = rendersOf(state, 1) - restB0;
    const uint64_t restComputes = state.metricsComputeCount() - restComputes0;
    state.setLive(true);
    const bool liveBack = state.live();
    const ImageData * hostA = state.entries()[0].media->hostData();
    const ImageData * hostB = state.entries()[1].media->hostData();
    std::cout << strf("[selftest] live off: drag renders +%llu, metrics computes +%llu, metrics kept=%s; rest frame "
                      "renders A +%llu B +%llu, metrics computes +%llu; live on again=%s\n",
                      asULL(dragRenders), asULL(dragComputes), kept ? "yes" : "no", asULL(restA), asULL(restB),
                      asULL(restComputes), liveBack ? "yes" : "no");
    if (dragRenders != 0 || dragComputes != 0 || !kept || restA != 1 || restB != 1 || restComputes != 1 || !liveBack)
    {
        results.fail("live off: a drag rendered or computed, or the rest frame did not do exactly one pass");
    }
    if (!state.metrics() || hostA == nullptr || hostB == nullptr)
    {
        results.fail("live off: no metrics after the rest frame: " + state.metricsReason());
        return;
    }
    checkMetricNumbers("live off final", *state.metrics(), psnr(displayRgb(*hostA), displayRgb(*hostB)), results);
}

// The scene file is polled once a second: an unchanged file reloads nothing; a new stamp reloads
// once, and a timestamp-only touch reloads once more; a broken file keeps the last good render with the
// error and is attempted once per stamp; a valid file with a new stamp loads.
void checkLiveFileReload(GpuContext & context, sabercat::Renderer & host, const SceneFiles & files,
                         const std::string & outDir, MetricsGpu & metrics, Results & results)
{
    const std::string livePath = joinPath(outDir, "live-reload.ply");
    std::error_code ec;
    fs::copy_file(pathFromUtf8(files.ply), pathFromUtf8(livePath), fs::copy_options::overwrite_existing, ec);
    if (ec)
    {
        results.fail("live reload: could not copy the scene file: " + ec.message());
        return;
    }
    CompareState state(context);
    state.setSceneHost(&host);
    state.setRenderSize(kSceneWidth, kSceneHeight);
    if (state.addMedia(livePath) != 0 || !state.entries()[0].ready())
    {
        results.fail("live reload: the copied scene did not load");
        return;
    }
    int frame = 0;
    const auto runFrames = [&](int count)
    {
        for (int k = 0; k < count; ++k)
        {
            liveFrame(state, metrics, frame++, nullptr);
        }
    };
    liveFrame(state, metrics, frame++, nullptr);

    // (a) Unchanged for 4 seconds: nothing reloads.
    runFrames(240);
    const bool quietA = state.sceneReloadCount() == 0 && state.sceneReloadAttempts() == 0;

    // (b) A cloud of 2000 Gaussians, stamped 10 s ahead: one reload and one attempt within 2 s.
    writeSyntheticPly(host, livePath, kSyntheticCount - 1000);
    setWriteTimeAhead(livePath, 10);
    uint64_t reloads = state.sceneReloadCount();
    uint64_t attempts = state.sceneReloadAttempts();
    runFrames(120);
    const uint64_t reloadsB = state.sceneReloadCount() - reloads;
    const uint64_t attemptsB = state.sceneReloadAttempts() - attempts;
    SceneSource * sceneB = sceneAt(state, 0);
    const bool gaussiansB = sceneB != nullptr && sceneB->renderer().gaussianCount() == kSyntheticCount - 1000;
    const bool rendersB = sceneB != nullptr && sceneB->renderCount() == 1;
    const bool generationB = state.entries()[0].sceneGeneration == 1;
    std::cout << strf("[selftest] live reload (b) new cloud: reloads +%llu, attempts +%llu, gaussians=%u (%s), new "
                      "renderer rendered %s, generation %s\n",
                      asULL(reloadsB), asULL(attemptsB), sceneB != nullptr ? sceneB->renderer().gaussianCount() : 0u,
                      gaussiansB ? "as written" : "WRONG", rendersB ? "once" : "NOT once",
                      generationB ? "1" : "WRONG");

    // (c) The same content with a new stamp: one more reload.
    setWriteTimeAhead(livePath, 20);
    reloads = state.sceneReloadCount();
    attempts = state.sceneReloadAttempts();
    runFrames(120);
    const uint64_t reloadsC = state.sceneReloadCount() - reloads;
    const uint64_t attemptsC = state.sceneReloadAttempts() - attempts;
    std::cout << strf("[selftest] live reload (c) touch: reloads +%llu, attempts +%llu\n", asULL(reloadsC),
                      asULL(attemptsC));

    // (d) A broken file with a new stamp: one failed attempt, the last good render stays, and no
    // further attempt for the same stamp.
    {
        std::ofstream out(pathFromUtf8(livePath), std::ios::binary | std::ios::trunc);
        out << "not a splat scene\n";
    }
    setWriteTimeAhead(livePath, 30);
    reloads = state.sceneReloadCount();
    attempts = state.sceneReloadAttempts();
    runFrames(120);
    const uint64_t reloadsD = state.sceneReloadCount() - reloads;
    const uint64_t attemptsD = state.sceneReloadAttempts() - attempts;
    const bool keptD = state.entries()[0].ready() && state.entries()[0].visible
        && state.entries()[0].media->color().valid();
    const std::string errorD = state.entries()[0].reloadError;
    const bool errorTextD = !errorD.empty() && errorD.find("last good render") != std::string::npos;
    attempts = state.sceneReloadAttempts();
    runFrames(180);
    const uint64_t attemptsDMore = state.sceneReloadAttempts() - attempts;
    std::cout << strf("[selftest] live reload (d) broken: reloads +%llu, attempts +%llu, kept ready and visible=%s, "
                      "error \"%s\", 3 s later attempts +%llu\n",
                      asULL(reloadsD), asULL(attemptsD), keptD ? "yes" : "no", errorD.c_str(), asULL(attemptsDMore));

    // (e) A valid file with a new stamp: the reload succeeds and the error clears.
    writeSyntheticPly(host, livePath, kSyntheticCount);
    setWriteTimeAhead(livePath, 40);
    reloads = state.sceneReloadCount();
    runFrames(120);
    const uint64_t reloadsE = state.sceneReloadCount() - reloads;
    const bool errorClearedE = state.entries()[0].reloadError.empty();
    SceneSource * sceneE = sceneAt(state, 0);
    const bool gaussiansE = sceneE != nullptr && sceneE->renderer().gaussianCount() == kSyntheticCount;
    std::cout << strf("[selftest] live reload (e) valid again: reloads +%llu, error cleared=%s, gaussians back to 3000=%s\n",
                      asULL(reloadsE), errorClearedE ? "yes" : "no", gaussiansE ? "yes" : "no");

    if (!quietA || reloadsB != 1 || attemptsB != 1 || !gaussiansB || !rendersB || !generationB || reloadsC != 1
        || attemptsC != 1 || reloadsD != 0 || attemptsD != 1 || !keptD || !errorTextD || attemptsDMore != 0
        || reloadsE != 1 || !errorClearedE || !gaussiansE)
    {
        results.fail("live reload: a reload count, an attempt count, the kept render or the error text is not as the poll promises");
    }
}

// A size change waits for the camera to rest, and a size that does not fit the GPU budget is refused
// with the memory message and the old size is kept. Under a normal budget a larger size is accepted at
// once, and shrinking is never refused.
void checkLiveBudgetAndSize(GpuContext & context, sabercat::Renderer & host, const SceneFiles & files,
                            MetricsGpu & metrics, Results & results)
{
    CompareState state(context);
    state.setSceneHost(&host);
    state.setRenderSize(kSceneWidth, kSceneHeight);
    if (state.addMedia(files.ply) != 0 || !state.entries()[0].ready())
    {
        results.fail("live budget: the scene did not load");
        return;
    }
    int frame = 0;
    liveFrame(state, metrics, frame++, nullptr);

    // Normal budget: a larger size is accepted and the scene follows.
    const bool grown = state.setRenderSize(240, 140);
    const bool grownOk = grown && state.renderWidth() == 240 && state.renderHeight() == 140
        && state.entries()[0].media->width() == 240;

    // A 1 MiB budget: a larger size is refused with the message and the size stays.
    bool refused = false;
    std::string refusal;
    {
        BudgetOverrideScope budget(1048576);
        refused = !state.setRenderSize(1920, 1080);
        refusal = state.status();
    }
    const bool keptAfterRefusal = state.renderWidth() == 240 && state.renderHeight() == 140;
    const bool refusalText = refusal.rfind("Not enough GPU memory for a larger render size", 0) == 0
        && refusal.find("needs about") != std::string::npos;

    // Under the same budget a smaller size is accepted.
    bool shrunk = false;
    {
        BudgetOverrideScope budget(1048576);
        shrunk = state.setRenderSize(80, 60);
    }
    const bool shrunkOk = shrunk && state.renderWidth() == 80 && state.renderHeight() == 60;
    const bool backOk = state.setRenderSize(kSceneWidth, kSceneHeight);

    // While the camera moves a size change is queued and applied once the camera rests.
    sabercat::app::OrbitCamera cam = orbitStep(state);
    state.beginFrame(liveTime(frame++));
    const bool moved = state.setCamera(cam, true);
    const bool queued = state.setRenderSize(200, 100);
    const bool pending = state.renderSizePending();
    const bool stillOld = state.renderWidth() == kSceneWidth && state.renderHeight() == kSceneHeight;
    frame += kLiveRestFrames;
    state.beginFrame(liveTime(frame++));
    state.updateScenes();
    const bool applied = state.renderWidth() == 200 && state.renderHeight() == 100 && !state.renderSizePending()
        && state.entries()[0].media->width() == 200 && state.entries()[0].media->height() == 100;
    state.updateMetrics(metrics);

    // A queued size that no longer fits the budget at apply time is refused with the message, and the
    // size from before the queue stays.
    cam = orbitStep(state);
    state.beginFrame(liveTime(frame++));
    const bool movedAgain = state.setCamera(cam, true);
    const bool queuedBig = state.setRenderSize(1920, 1080);
    const bool bigPending = state.renderSizePending();
    std::string applyStatus;
    {
        BudgetOverrideScope budget(1048576);
        frame += kLiveRestFrames;
        state.beginFrame(liveTime(frame++));
        state.updateScenes();
        applyStatus = state.status();
    }
    const bool bigRefused = applyStatus.rfind("Not enough GPU memory for a larger render size", 0) == 0;
    const bool bigKept = state.renderWidth() == 200 && state.renderHeight() == 100 && !state.renderSizePending();

    std::cout << strf("[selftest] live budget: 240x140 accepted=%s; under 1 MiB 1920x1080 refused=%s \"%s\" "
                      "size kept=%s; 80x60 accepted=%s; queued while moving=%s pending=%s old size=%s; applied at rest "
                      "200x100=%s; queued 1920x1080 pending=%s refused at apply=%s kept=%s\n",
                      grownOk ? "yes" : "no", refused ? "yes" : "no", refusal.c_str(), keptAfterRefusal ? "yes" : "no",
                      shrunkOk ? "yes" : "no", (moved && queued) ? "yes" : "no", pending ? "yes" : "no",
                      stillOld ? "yes" : "no", applied ? "yes" : "no", (movedAgain && queuedBig && bigPending) ? "yes" : "no",
                      bigRefused ? "yes" : "no", bigKept ? "yes" : "no");
    if (!grownOk || !refused || !refusalText || !keptAfterRefusal || !shrunkOk || !backOk || !moved || !queued
        || !pending || !stillOld || !applied || !movedAgain || !queuedBig || !bigPending || !bigRefused || !bigKept)
    {
        results.fail("live budget: a size was applied too early, refused without the message, or kept the wrong size");
    }
}

// Edge cases of the live policy: a non-finite or degenerate camera is refused and changes nothing, a
// scene removed mid-drag and a FLIP pending on a removed source end with a reason and no throw, a
// zero-size view composes without a throw, and four visible scenes dirty at once render once each.
void checkLiveEdgeCases(GpuContext & context, sabercat::Renderer & host, const SceneFiles & files,
                        const std::string & outDir, MetricsGpu & metrics, Results & results)
{
    // A NaN pivot, an infinite distance and a zero distance are refused, and nothing changes.
    [&]
    {
        CompareState state(context);
        if (!openScenePair(state, host, files))
        {
            results.fail("live edge camera: the scenes did not load");
            return;
        }
        int frame = 0;
        liveFrame(state, metrics, frame++, nullptr);
        const uint64_t generation = state.cameraGeneration();
        const uint64_t renders = state.sceneRenderTotal();
        sabercat::app::OrbitCamera nanCam = state.camera();
        nanCam.pivot.x = kNaN;
        const bool nanRefused = !state.setCamera(nanCam);
        const bool nanStatus = state.status().find("not finite") != std::string::npos;
        sabercat::app::OrbitCamera infCam = state.camera();
        infCam.distance = kInf;
        const bool infRefused = !state.setCamera(infCam);
        const bool infStatus = state.status().find("not finite") != std::string::npos;
        sabercat::app::OrbitCamera zeroCam = state.camera();
        zeroCam.distance = 0.0f;
        const bool zeroRefused = !state.setCamera(zeroCam);
        liveFrame(state, metrics, frame++, nullptr);
        const bool unchanged = state.cameraGeneration() == generation && state.sceneRenderTotal() == renders;
        std::cout << strf("[selftest] live edge camera: NaN refused=%s (\"not finite\"=%s); infinite distance refused=%s "
                          "(\"not finite\"=%s); zero distance refused=%s; generation and renders unchanged=%s\n",
                          nanRefused ? "yes" : "no", nanStatus ? "yes" : "no", infRefused ? "yes" : "no",
                          infStatus ? "yes" : "no", zeroRefused ? "yes" : "no", unchanged ? "yes" : "no");
        if (!nanRefused || !nanStatus || !infRefused || !infStatus || !zeroRefused || !unchanged)
        {
            results.fail("live edge camera: a degenerate camera was accepted or changed the state");
        }
    }();

    // A scene removed in the middle of a drag: no throw, and the metrics end with a reason.
    [&]
    {
        CompareState state(context);
        if (!openScenePair(state, host, files))
        {
            results.fail("live edge removal: the scenes did not load");
            return;
        }
        int frame = 0;
        liveFrame(state, metrics, frame++, nullptr);
        bool removed = false;
        bool threw = false;
        std::string throwText;
        try
        {
            for (int k = 0; k < 20; ++k)
            {
                if (k == 10)
                {
                    removed = state.removeSource(1);
                }
                const sabercat::app::OrbitCamera cam = orbitStep(state);
                liveFrame(state, metrics, frame++, &cam);
            }
            frame += kLiveRestFrames;
            liveFrame(state, metrics, frame++, nullptr);
        }
        catch (const std::exception & e)
        {
            threw = true;
            throwText = e.what();
        }
        const bool reason = !state.metricsReason().empty();
        std::cout << strf("[selftest] live edge removal: removed mid-drag=%s; no throw=%s%s; metrics reason \"%s\"\n",
                          removed ? "yes" : "no", threw ? "no" : "yes", threw ? (" (" + throwText + ")").c_str() : "",
                          state.metricsReason().c_str());
        if (!removed || threw || !reason)
        {
            results.fail("live edge removal: the removal threw, or the metrics have no reason");
        }
    }();

    // A FLIP pending on a source that is removed while the camera moves: the next compute ends with a
    // reason and no result, and does not throw. The move is the first one, so the compute runs at once.
    [&]
    {
        CompareState state(context);
        if (!openScenePair(state, host, files))
        {
            results.fail("live edge flip: the scenes did not load");
            return;
        }
        int frame = 0;
        liveFrame(state, metrics, frame++, nullptr);
        const sabercat::app::OrbitCamera cam = orbitStep(state);
        state.beginFrame(liveTime(frame++));
        state.setCamera(cam, true);
        state.updateScenes();
        const bool removed = state.removeSource(1);
        bool threw = false;
        try
        {
            state.updateMetrics(metrics);
            frame += kLiveRestFrames;
            liveFrame(state, metrics, frame++, nullptr);
        }
        catch (const std::exception &)
        {
            threw = true;
        }
        const bool noResult = !state.metrics().has_value() && !state.flip().has_value();
        const bool reasons = !state.metricsReason().empty() && !state.flipReason().empty();
        std::cout << strf("[selftest] live edge flip: source removed while moving=%s; no throw=%s; no result=%s; "
                          "reasons=%s (\"%s\")\n",
                          removed ? "yes" : "no", threw ? "no" : "yes", noResult ? "yes" : "no",
                          reasons ? "yes" : "no", state.flipReason().c_str());
        if (!removed || threw || !noResult || !reasons)
        {
            results.fail("live edge flip: a removed source left a result, threw, or has no reason");
        }
    }();

    // A zero-size view: the scenes update and the composite is built, with no throw.
    [&]
    {
        CompareState state(context);
        if (!openScenePair(state, host, files))
        {
            results.fail("live edge view: the scenes did not load");
            return;
        }
        const std::string imagePath = joinPath(outDir, "scene-image.png");
        writeRgbPng(imagePath, 96, 72, 31);
        if (state.addMedia(imagePath) != 2)
        {
            results.fail("live edge view: the image did not load");
            return;
        }
        int frame = 0;
        liveFrame(state, metrics, frame++, nullptr);
        bool threw = false;
        try
        {
            state.setViewSize(0, 0);
            state.updateScenes();
            static_cast<void>(state.buildComposite(liveTime(frame)));
        }
        catch (const std::exception &)
        {
            threw = true;
        }
        std::cout << strf("[selftest] live edge view: 0x0 view with updateScenes and buildComposite, no throw=%s\n",
                          threw ? "no" : "yes");
        if (threw)
        {
            results.fail("live edge view: a zero-size view threw");
        }
    }();

    // Four visible scenes dirty at once: each renders exactly once per frame, however often the UI
    // calls updateScenes.
    [&]
    {
        CompareState state(context);
        state.setSceneHost(&host);
        state.setRenderSize(kSceneWidth, kSceneHeight);
        bool added = true;
        for (int32_t i = 0; i < 4 && added; ++i)
        {
            added = state.addMedia(files.ply) == i;
        }
        if (!added || state.size() != 4)
        {
            results.fail("live edge four: the four scenes did not all load");
            return;
        }
        int frame = 0;
        const auto firstStart = std::chrono::steady_clock::now();
        state.beginFrame(liveTime(frame++));
        state.updateScenes();
        const double firstMs = msSince(firstStart);
        bool firstOnce = true;
        std::array<uint64_t, 4> before{};
        for (size_t i = 0; i < before.size(); ++i)
        {
            firstOnce = firstOnce && rendersOf(state, i) == 1;
            before[i] = rendersOf(state, i);
        }
        // A camera change while at rest makes every scene dirty at once.
        state.setCamera(orbitStep(state));
        const auto start = std::chrono::steady_clock::now();
        state.beginFrame(liveTime(frame++));
        state.updateScenes();
        state.updateScenes();
        const double allMs = msSince(start);
        bool allOnce = true;
        for (size_t i = 0; i < before.size(); ++i)
        {
            allOnce = allOnce && rendersOf(state, i) == before[i] + 1;
        }
        std::cout << strf("[selftest] live four scenes: first frame %.3f ms (each scene rendered once=%s); all dirty "
                          "at once %.3f ms (each +1 with two updateScenes=%s)\n",
                          firstMs, firstOnce ? "yes" : "no", allMs, allOnce ? "yes" : "no");
        if (!firstOnce || !allOnce)
        {
            results.fail("live edge four: a scene rendered other than once per frame");
        }
    }();
}

// Library decision, printed only: the Viewer and the Differentiable forward of one scene at 640x360 (is
// the color bit-identical, which buffers does each leave valid), and the median forward time of each mode
// at 960x540 and 1920x1080. Never fails, except for an exception that escapes.
void checkLiveViewerVsDifferentiable(GpuContext & context, sabercat::Renderer & host, const SceneFiles & files,
                                     const std::string & shaderDir)
{
    const std::string scenePath = liveScenePath(files);
    const char * sceneName = scenePath == files.ply ? "synthetic" : "env scene";
    constexpr uint32_t width = 640;
    constexpr uint32_t height = 360;
    const size_t floats = static_cast<size_t>(width) * height * 4;
    {
        ScratchScene scratch(host, shaderDir, scenePath, width, height);
        sabercat::Renderer & renderer = scratch.renderer();
        const sabercat::SceneBounds bounds = renderer.sceneBounds();
        if (!bounds.valid)
        {
            std::cout << "[selftest] live viewer vs differentiable: the scene has no bounds, skipped\n";
            return;
        }
        sabercat::app::OrbitCamera orbit;
        orbit.frame(bounds, height, static_cast<float>(height));
        renderer.setCamera(orbit.toCamera(width, height));

        renderer.forward(RenderMode::Differentiable);
        const std::vector<float> differentiable = downloadFloats(context, renderer.imageView(), floats);
        renderer.forward(RenderMode::Viewer);
        const std::vector<float> viewer =
            downloadFloats(context, renderer.renderDebugView(sabercat::DebugView::Final), floats);
        const size_t differing = bitMismatches(differentiable, viewer);
        double maxRgb = 0.0;
        for (size_t i = 0; i < differentiable.size() && i < viewer.size(); ++i)
        {
            if (i % 4 != 3)
            {
                maxRgb = std::max(maxRgb, static_cast<double>(std::fabs(differentiable[i] - viewer[i])));
            }
        }
        bool depthThrew = false;
        bool imageThrew = false;
        try
        {
            static_cast<void>(renderer.depthView());
        }
        catch (const std::exception &)
        {
            depthThrew = true;
        }
        try
        {
            static_cast<void>(renderer.imageView());
        }
        catch (const std::exception &)
        {
            imageThrew = true;
        }
        std::cout << strf("[selftest] live viewer vs differentiable %ux%u (%s): color bit-identical=%s, differing values=%zu, "
                          "max abs rgb difference=%.6g; after a Viewer forward depthView throws=%s, imageView throws=%s\n",
                          width, height, sceneName, differing == 0 ? "yes" : "no", differing, maxRgb,
                          depthThrew ? "yes" : "no", imageThrew ? "yes" : "no");
    }

    // Median of 30 timed forwards per mode (two untimed warm-up forwards first).
    const uint32_t sizes[2][2] = {{960, 540}, {1920, 1080}};
    for (const auto & size : sizes)
    {
        const uint32_t w = size[0];
        const uint32_t h = size[1];
        ScratchScene timed(host, shaderDir, scenePath, w, h);
        const sabercat::SceneBounds bounds = timed.renderer().sceneBounds();
        if (!bounds.valid)
        {
            continue;
        }
        sabercat::app::OrbitCamera framed;
        framed.frame(bounds, h, static_cast<float>(h));
        timed.renderer().setCamera(framed.toCamera(w, h));
        std::vector<double> differentiableMs;
        std::vector<double> viewerMs;
        for (int k = 0; k < 32; ++k)
        {
            const auto start = std::chrono::steady_clock::now();
            timed.renderer().forward(RenderMode::Differentiable);
            const double ms = msSince(start);
            if (k >= 2)
            {
                differentiableMs.push_back(ms);
            }
        }
        for (int k = 0; k < 32; ++k)
        {
            const auto start = std::chrono::steady_clock::now();
            timed.renderer().forward(RenderMode::Viewer);
            const double ms = msSince(start);
            if (k >= 2)
            {
                viewerMs.push_back(ms);
            }
        }
        std::cout << strf("[selftest] live viewer vs differentiable %ux%u (%s): median of 30 forwards: differentiable "
                          "%.3f ms, viewer %.3f ms\n",
                          w, h, sceneName, percentileOf(differentiableMs, 0.5), percentileOf(viewerMs, 0.5));
    }
}

// Live performance, printed only: a drag of SABERCAT_COMPARE_LIVE_FRAMES frames (at least 120) on the
// real clock, with two scenes and with four, at 960x540 and 1920x1080. The first frame loads the scenes
// and is not timed. The bytes per pixel come from one scene at both sizes.
void checkLivePerformance(GpuContext & context, sabercat::Renderer & host, const SceneFiles & files,
                          Compositor & compositor, MetricsGpu & metrics)
{
    const std::string scenePath = liveScenePath(files);
    const int frames = liveFrameCount();
    const uint32_t sizes[2][2] = {{960, 540}, {1920, 1080}};
    for (const auto & size : sizes)
    {
        const uint32_t width = size[0];
        const uint32_t height = size[1];
        for (const int32_t count : {2, 4})
        {
            CompareState state(context);
            state.setSceneHost(&host);
            state.setRenderSize(width, height);
            state.setViewSize(static_cast<int32_t>(width), static_cast<int32_t>(height));
            bool added = true;
            for (int32_t i = 0; i < count && added; ++i)
            {
                added = state.addMedia(scenePath) == i;
            }
            if (!added)
            {
                std::cout << strf("[selftest] live perf %ux%u with %d scenes: refused (\"%s\")\n", width, height,
                                  count, state.status().c_str());
                continue;
            }
            // The second scene differs from the first: DC color only.
            state.setShCap(1, 0);
            compositor.resize(width, height);
            const auto clock = std::chrono::steady_clock::now();
            const auto secondsNow = [&clock]()
            {
                return std::chrono::duration<double>(std::chrono::steady_clock::now() - clock).count();
            };
            state.beginFrame(secondsNow());
            state.updateScenes();
            state.updateMetrics(metrics);
            int visible = 0;
            for (const MediaEntry & entry : state.entries())
            {
                visible += entry.visible ? 1 : 0;
            }

            std::vector<double> renderMs;
            std::vector<double> metricsMs;
            std::vector<double> compositeMs;
            std::vector<double> totalMs;
            for (int k = 0; k < frames; ++k)
            {
                const auto frameStart = std::chrono::steady_clock::now();
                const double now = secondsNow();
                state.beginFrame(now);
                const auto renderStart = std::chrono::steady_clock::now();
                state.updateScenes();
                const sabercat::app::OrbitCamera cam = orbitStep(state);
                state.setCamera(cam, true);
                state.updateScenes();
                renderMs.push_back(msSince(renderStart));
                const uint64_t computesBefore = state.metricsComputeCount();
                const auto metricsStart = std::chrono::steady_clock::now();
                state.updateMetrics(metrics);
                if (state.metricsComputeCount() != computesBefore)
                {
                    metricsMs.push_back(msSince(metricsStart));
                }
                const auto compositeStart = std::chrono::steady_clock::now();
                const CompositeInput input = state.buildComposite(now);
                compositor.render(input);
                compositeMs.push_back(msSince(compositeStart));
                totalMs.push_back(msSince(frameStart));
            }
            const double renderPerScene = percentileOf(renderMs, 0.5) / static_cast<double>(std::max(visible, 1));
            const double metricsMedian = percentileOf(metricsMs, 0.5);
            const double compositeMedian = percentileOf(compositeMs, 0.5);
            const double totalMedian = percentileOf(totalMs, 0.5);
            const double totalP95 = percentileOf(totalMs, 0.95);
            std::cout << strf("[selftest] live perf %ux%u, %d scenes (%d visible), %d frames: render per scene median "
                              "%.3f ms; metrics+FLIP median %.3f ms over %zu frames; composite median %.3f ms; frame "
                              "median %.3f ms, p95 %.3f ms\n",
                              width, height, count, visible, frames, renderPerScene, metricsMedian, metricsMs.size(),
                              compositeMedian, totalMedian, totalP95);
        }
    }

    // Device bytes of one scene at both sizes, and the marginal bytes per extra pixel.
    double bytes[2] = {0.0, 0.0};
    for (size_t s = 0; s < 2; ++s)
    {
        SceneSettings settings;
        settings.width = sizes[s][0];
        settings.height = sizes[s][1];
        SceneSource scene(context, host, scenePath, settings, false);
        if (scene.status() == SourceStatus::Ready)
        {
            scene.render(sceneCamera(0).toCamera(settings.width, settings.height), 1);
            bytes[s] = static_cast<double>(scene.deviceBytes());
        }
    }
    const double pixels1 = 960.0 * 540.0;
    const double pixels2 = 1920.0 * 1080.0;
    std::cout << strf("[selftest] live perf scene device bytes: 960x540 %.0f, 1920x1080 %.0f; %.2f bytes per extra pixel\n",
                      bytes[0], bytes[1], (bytes[1] - bytes[0]) / (pixels2 - pixels1));
}

// Runs one scene check, so that a throw in it is a failure and the others still run.
template <typename Check>
void guardedCheck(const char * name, Results & results, Check check)
{
    try
    {
        check();
    }
    catch (const std::exception & e)
    {
        results.fail(std::string(name) + ": " + e.what());
    }
}

void runSceneChecks(sabercat::Renderer & host, GpuContext & context, MetricsGpu & metrics, Compositor & compositor,
                    const std::string & shaderDir, const std::string & outDir, Results & results)
{
    checkDebugViewKeys(results);
    SceneFiles files;
    try
    {
        files = writeSyntheticScenes(host, outDir);
    }
    catch (const std::exception & e)
    {
        results.fail(std::string("scene files: ") + e.what());
        return;
    }
    guardedCheck("scene renders", results, [&] { checkSceneRenders(context, host, files, shaderDir, results); });
    guardedCheck("scene debug views", results, [&]
    {
        SceneSettings settings;
        settings.width = kSceneWidth;
        settings.height = kSceneHeight;
        SceneSource scene(context, host, files.ply, settings, false);
        checkSceneDebugViews(context, scene, results);
    });
    guardedCheck("scene metrics", results, [&] { checkScenePairMetrics(context, host, metrics, files, results); });
    guardedCheck("scene settings", results, [&] { checkSceneSettings(context, host, files, shaderDir, results); });
    guardedCheck("scene failures", results, [&] { checkSceneFailures(context, host, files, outDir, results); });
    guardedCheck("camera dirtying", results, [&] { checkCameraDirtying(context, host, files, outDir, results); });
    guardedCheck("scene composites", results,
                 [&] { checkSceneComposites(context, host, files, outDir, compositor, results); });
    guardedCheck("scene budget", results, [&] { checkSceneBudget(context, host, files, results); });
    guardedCheck("scene project", results, [&] { checkSceneProject(context, host, files, outDir, results); });
    guardedCheck("live idle", results, [&] { checkLiveIdle(context, host, files, outDir, metrics, results); });
    guardedCheck("live moves", results,
                 [&] { checkLiveMoves(context, host, files, outDir, compositor, metrics, results); });
    guardedCheck("live final", results, [&] { checkLiveFinal(context, host, files, metrics, results); });
    guardedCheck("live flip rate", results, [&] { checkLiveFlipRate(context, host, files, metrics, results); });
    guardedCheck("live off", results, [&] { checkLiveOff(context, host, files, metrics, results); });
    guardedCheck("live file reload", results,
                 [&] { checkLiveFileReload(context, host, files, outDir, metrics, results); });
    guardedCheck("live budget and size", results,
                 [&] { checkLiveBudgetAndSize(context, host, files, metrics, results); });
    guardedCheck("live edge cases", results,
                 [&] { checkLiveEdgeCases(context, host, files, outDir, metrics, results); });
    guardedCheck("live viewer vs differentiable", results,
                 [&] { checkLiveViewerVsDifferentiable(context, host, files, shaderDir); });
    guardedCheck("live performance", results,
                 [&] { checkLivePerformance(context, host, files, compositor, metrics); });
}

// ---- FLIP: state, non-finite samples and the NVlabs pair -------------------------

// The CompareState side of FLIP: the mean the state shows against the CPU, the ppd control and
// its refusals, the reasons when there is no FLIP, the project round trip, the composite mode,
// and the GPU memory refusal with its recovery.
void checkFlipState(GpuContext & context, MetricsGpu & metrics, FlipGpu & flipGpu, const std::string & outDir,
                    Results & results)
{
    const std::string pathA = joinPath(outDir, "flip-a.png");
    const std::string pathB = joinPath(outDir, "flip-b.png");
    writeRgbPng(pathA, 16, 12, 81);
    writeRgbPng(pathB, 20, 10, 82);

    const auto report = [&results](const char * name, bool ok, const std::string & detail)
    {
        std::cout << "[selftest] flip state " << name << ": " << (ok ? "ok" : "FAILED") << " (" << detail << ")\n";
        if (!ok)
        {
            results.fail(std::string("flip state ") + name + ": " + detail);
        }
    };

    // 1. Two Ready sources: the state's FLIP is the CPU's, at the default ppd.
    {
        CompareState state(context);
        state.addMedia(pathA);
        state.addMedia(pathB);
        state.updateMetrics(metrics);
        PixelRect overlap;
        const FlipResult cpu = cpuFlipOfState(state, defaultFlipPpd(), overlap);
        const bool shown = state.flip().has_value();
        const double gpuMean = shown ? state.flip()->mean : -1.0;
        const double delta = shown ? std::fabs(gpuMean - cpu.meanError) : std::numeric_limits<double>::infinity();
        const bool sized = shown
            && state.flip()->errorMap.size() == static_cast<size_t>(overlap.width) * static_cast<size_t>(overlap.height);
        report("mean", shown && sized && delta <= kMaxFlipMeanDelta,
               strf("gpu=%.9g cpu=%.9g dmean=%.3g ppd=%g overlap=%dx%d", gpuMean, cpu.meanError, delta,
                    static_cast<double>(state.flipPpd()), overlap.width, overlap.height));
    }

    // 2. The ppd control: a change recomputes the FLIP only, and the metrics are kept. Refused
    // values leave the ppd and the FLIP as they were.
    {
        CompareState state(context);
        state.addMedia(pathA);
        state.addMedia(pathB);
        state.updateMetrics(metrics);
        const double mean67 = state.flip() ? state.flip()->mean : -1.0;
        const std::optional<MetricsResult> before = state.metrics();
        const bool accepted = state.setFlipPpd(200.0f);
        state.updateMetrics(metrics);
        PixelRect overlap;
        const FlipResult cpu200 = cpuFlipOfState(state, 200.0f, overlap);
        const double mean200 = state.flip() ? state.flip()->mean : -1.0;
        const bool recomputed = state.flip() && std::fabs(mean200 - cpu200.meanError) <= kMaxFlipMeanDelta
            && std::fabs(mean200 - mean67) > 1e-6;
        const bool ppdSet = accepted && state.flipPpd() == 200.0f;
        const bool metricsKept = before && state.metrics()
            && sameBytes(before->mse, state.metrics()->mse) && sameBytes(before->psnrDb, state.metrics()->psnrDb);
        const bool refusedKept = !state.setFlipPpd(0.0f) && !state.setFlipPpd(1001.0f)
            && !state.setFlipPpd(std::numeric_limits<float>::quiet_NaN()) && state.flipPpd() == 200.0f
            && state.status() == "FLIP ppd must be from 1 to 1000";
        report("ppd", recomputed && ppdSet && metricsKept && refusedKept,
               strf("mean 67=%.9g 200=%.9g cpu 200=%.9g ppd=%g metrics kept=%d refusals kept=%d", mean67, mean200,
                    cpu200.meanError, static_cast<double>(state.flipPpd()), metricsKept ? 1 : 0,
                    refusedKept ? 1 : 0));
    }

    // 3. One source: no FLIP, a reason, and no throw.
    {
        CompareState state(context);
        state.addMedia(pathA);
        bool threw = false;
        try
        {
            state.updateMetrics(metrics);
        }
        catch (const std::exception &)
        {
            threw = true;
        }
        const bool ok = !threw && !state.flip() && !state.flipReason().empty();
        report("one source", ok, "reason \"" + state.flipReason() + "\"");
    }

    // 4. Two sources with no overlap: no FLIP and a reason.
    {
        CompareState state(context);
        state.addMedia(pathA);
        state.addMedia(pathB);
        state.setOffset(1, 1000, 1000);
        state.updateMetrics(metrics);
        report("no overlap", !state.flip() && !state.flipReason().empty(), "reason \"" + state.flipReason() + "\"");
    }

    // 5. A missing file as B: it is an entry that is not Ready, so there is no FLIP and no throw.
    {
        CompareState state(context);
        state.addMedia(pathA);
        const int32_t missing = state.addMedia(joinPath(outDir, "flip-missing.png"));
        bool threw = false;
        try
        {
            state.updateMetrics(metrics);
        }
        catch (const std::exception &)
        {
            threw = true;
        }
        const bool ok = missing == 1 && !state.entries()[1].ready() && !threw && !state.flip()
            && !state.flipReason().empty();
        report("missing B", ok, "reason \"" + state.flipReason() + "\"");
    }

    // 6. Project: the ppd and the Flip view survive a save and a load, and as JSON text. The
    // default ppd is saved as 0 and loads as the default. An out-of-range ppd loads as the default
    // with a notice.
    {
        const std::string projectPath = joinPath(outDir, "flip-roundtrip.sabercat-compare");
        CompareState source(context);
        source.addMedia(pathA);
        source.addMedia(pathB);
        source.setFlipPpd(120.0f);
        source.setSplit(false);
        source.setSingleView(SingleView::Flip);
        std::string error;
        const bool saved = source.saveProject(projectPath, error);
        CompareState loaded(context);
        const bool opened = loaded.openProject(projectPath, error);
        const bool fileKept = saved && opened && loaded.flipPpd() == 120.0f
            && loaded.settings().single == SingleView::Flip && !loaded.settings().split;

        bool textKept = false;
        std::string textError;
        try
        {
            const std::string text = project::serializeCompareProject(source.toProject(), outDir);
            CompareState parsed(context);
            parsed.fromProject(project::parseCompareProject(text, outDir, nullptr));
            textKept = parsed.flipPpd() == 120.0f && parsed.settings().single == SingleView::Flip;
        }
        catch (const std::exception & e)
        {
            textError = e.what();
        }

        CompareState plain(context);
        plain.addMedia(pathA);
        plain.addMedia(pathB);
        const std::string plainPath = joinPath(outDir, "flip-default.sabercat-compare");
        const bool plainSaved = plain.saveProject(plainPath, error);
        const bool defaultSavedAsZero = plain.toProject().flip.ppd == 0;
        CompareState plainLoaded(context);
        const bool plainOpened = plainLoaded.openProject(plainPath, error);
        const bool defaultLoads = plainSaved && plainOpened && plainLoaded.flipPpd() == defaultFlipPpd();

        project::CompareProject tooBig = source.toProject();
        tooBig.flip.ppd = 5000;
        const std::string bigPath = joinPath(outDir, "flip-big-ppd.sabercat-compare");
        {
            std::ofstream out(pathFromUtf8(bigPath), std::ios::binary);
            out << project::serializeCompareProject(tooBig, outDir);
        }
        CompareState bigState(context);
        const bool bigOpened = bigState.openProject(bigPath, error);
        bool bigNotice = false;
        for (const std::string & line : bigState.notices())
        {
            bigNotice = bigNotice || line.find("FLIP ppd") != std::string::npos;
        }
        const bool bigDefault = bigOpened && bigState.flipPpd() == defaultFlipPpd() && bigNotice;

        const bool ok = fileKept && textKept && defaultSavedAsZero && defaultLoads && bigDefault;
        report("project", ok,
               strf("saved=%d opened=%d ppd=%g single flip=%d text=%d default saved 0=%d default loads=%d "
                    "out of range default with notice=%d",
                    saved ? 1 : 0, opened ? 1 : 0, static_cast<double>(loaded.flipPpd()),
                    loaded.settings().single == SingleView::Flip ? 1 : 0, textKept ? 1 : 0,
                    defaultSavedAsZero ? 1 : 0, defaultLoads ? 1 : 0, bigDefault ? 1 : 0)
                   + (error.empty() ? std::string() : " error=" + error)
                   + (textError.empty() ? std::string() : " text error=" + textError));
    }

    // 7. The composite mode is Flip only with a FLIP result, and Panes otherwise.
    {
        CompareState state(context);
        state.addMedia(pathA);
        state.addMedia(pathB);
        state.setSplit(false);
        state.setSingleView(SingleView::Flip);
        const bool beforeCompute = state.compositeMode() == CompositeMode::Panes;
        state.updateMetrics(metrics);
        const CompositeInput shown = state.buildComposite(0.0);
        const bool withFlip = state.compositeMode() == CompositeMode::Flip && shown.mode == CompositeMode::Flip
            && shown.flipMap.valid() && state.flip() && sameRect(shown.flipRect, state.flip()->overlap);
        state.setVisible(1, false);
        state.updateMetrics(metrics);
        const bool withoutB = state.compositeMode() == CompositeMode::Panes && !state.flip()
            && !state.flipMapView().valid();
        report("composite mode", beforeCompute && withFlip && withoutB,
               strf("before=%d with pair=%d without B=%d", beforeCompute ? 1 : 0, withFlip ? 1 : 0,
                    withoutB ? 1 : 0));
    }

    // 8. GPU memory: under a free-bytes override too small for the overlap, the state keeps no
    // FLIP and gives the reason; the direct compute throws runtime_error. After the scope the same
    // work succeeds. The direct compute releases the device buffers first, since bytes already held
    // by the object do not count as needed.
    {
        const PixelRect full{0, 0, 160, 120};
        ImageSource bigA(context, makeDisplay(160, 120, 91), "flip-big-a");
        ImageSource bigB(context, makeDisplay(160, 120, 92), "flip-big-b");
        CompareState state(context);
        state.addMemorySource(std::make_unique<ImageSource>(context, makeDisplay(160, 120, 91), "flip-big-a"),
                              "flip-big-a");
        state.addMemorySource(std::make_unique<ImageSource>(context, makeDisplay(160, 120, 92), "flip-big-b"),
                              "flip-big-b");
        std::string stateReason;
        bool stateRefused = false;
        {
            BudgetOverrideScope budget(1048576);
            state.setOffset(1, 0, 1);
            state.updateMetrics(metrics);
            stateRefused = !state.flip() && state.flipReason().find("GPU memory") != std::string::npos;
            stateReason = state.flipReason();
        }
        state.setOffset(1, 0, 0);
        state.updateMetrics(metrics);
        PixelRect overlap;
        const FlipResult cpu = cpuFlipOfState(state, defaultFlipPpd(), overlap);
        const bool restored = state.flip().has_value() && std::fabs(state.flip()->mean - cpu.meanError) <= kMaxFlipMeanDelta;

        flipGpu.release();
        bool directRefused = false;
        std::string directMessage;
        {
            BudgetOverrideScope budget(1048576);
            try
            {
                (void)flipGpu.compute(bigA, 0, 0, bigB, 0, 0, full, defaultFlipPpd());
            }
            catch (const std::runtime_error & e)
            {
                directMessage = e.what();
                directRefused = directMessage.rfind("FLIP needs about", 0) == 0;
            }
        }
        const std::optional<FlipGpuResult> direct = flipGpu.compute(bigA, 0, 0, bigB, 0, 0, full, defaultFlipPpd());
        const bool directWorks = direct && direct->errorMap.size() == static_cast<size_t>(160) * 120;
        report("GPU memory", stateRefused && restored && directRefused && directWorks,
               strf("state refused=%d restored=%d; direct refused=%d works=%d; state reason \"%s\"; direct \"%s\"",
                    stateRefused ? 1 : 0, restored ? 1 : 0, directRefused ? 1 : 0, directWorks ? 1 : 0,
                    stateReason.c_str(), directMessage.c_str()));
    }
}

// Non-finite samples: a source holding NaN and infinity is accepted, the GPU counts them and
// treats them as the display clamp does (NaN and -inf as 0, +inf as 1), and the mean is the CPU
// flip() of the sanitized images.
ImageData sanitizedDisplay(const ImageData & image)
{
    ImageData out = image;
    for (float & value : out.rgba)
    {
        if (!std::isfinite(value))
        {
            value = value > 0.0f ? 1.0f : 0.0f;
        }
    }
    return out;
}

void checkFlipNonFinite(GpuContext & context, FlipGpu & flipGpu, Results & results)
{
    ImageData a = makeDisplay(24, 16, 71);
    ImageData b = makeDisplay(24, 16, 72);
    const float notANumber = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    // Pixel p, channel c is rgba[4 p + c]: three NaN and one +inf in A, one -inf in B.
    a.rgba[4 * 5 + 0] = notANumber;
    a.rgba[4 * 17 + 1] = notANumber;
    a.rgba[4 * 100 + 2] = notANumber;
    a.rgba[4 * 200 + 0] = infinity;
    b.rgba[4 * 50 + 1] = -infinity;
    constexpr uint64_t expectedCount = 5;

    const ImageSource sa(context, a, "nonfinite-a");
    const ImageSource sb(context, b, "nonfinite-b");
    if (sa.status() != SourceStatus::Ready || sb.status() != SourceStatus::Ready)
    {
        results.fail("nonfinite: a source with NaN or infinite samples is not Ready");
        return;
    }
    const PixelRect full{0, 0, 24, 16};
    const std::optional<FlipGpuResult> gpu = flipGpu.compute(sa, 0, 0, sb, 0, 0, full, defaultFlipPpd());
    if (!gpu)
    {
        results.fail("nonfinite: the GPU FLIP returned nothing");
        return;
    }
    const FlipResult cpu = flip(displayRgb(sanitizedDisplay(a)), displayRgb(sanitizedDisplay(b)), defaultFlipPpd());
    const double delta = std::fabs(gpu->mean - cpu.meanError);
    std::cout << strf("[selftest] nonfinite flip gpu=%.9g cpu=%.9g dmean=%.3g nonFinite=%llu ppd=%g\n", gpu->mean,
                      cpu.meanError, delta, static_cast<unsigned long long>(gpu->nonFiniteSamples),
                      static_cast<double>(defaultFlipPpd()));
    if (gpu->nonFiniteSamples != expectedCount || !std::isfinite(gpu->mean) || !(delta <= kMaxFlipMeanDelta))
    {
        results.fail(strf("nonfinite: %llu samples counted (expected 5), mean dmean=%.3g",
                          static_cast<unsigned long long>(gpu->nonFiniteSamples), delta));
    }
}

// The NVlabs FLIP test pair (reference.png and test.png in SABERCAT_FLIP_IMAGES_DIR): the GPU
// mean at the default ppd against FLIP's own value, 0.159693. Release builds also compare the map
// with the CPU flip(), since Debug CPU FLIP takes minutes.
void checkFlipNvlabs(GpuContext & context, FlipGpu & flipGpu, Results & results)
{
#ifdef SABERCAT_FLIP_IMAGES_DIR
    const std::string dir = SABERCAT_FLIP_IMAGES_DIR;
    const std::string referencePath = joinPath(dir, "reference.png");
    const std::string testPath = joinPath(dir, "test.png");
    std::error_code ec;
    if (!fs::exists(pathFromUtf8(referencePath), ec) || !fs::exists(pathFromUtf8(testPath), ec))
    {
        std::cout << "[selftest] nvlabs flip SKIPPED: " << dir << " not found\n";
        return;
    }
    const ImageSource nvRef(context, loadImage(referencePath), "nvlabs-reference");
    const ImageSource nvTest(context, loadImage(testPath), "nvlabs-test");
    if (nvRef.status() != SourceStatus::Ready || nvTest.status() != SourceStatus::Ready)
    {
        results.fail("nvlabs: a test image did not load: " + nvRef.statusMessage() + nvTest.statusMessage());
        return;
    }
    const int32_t w = static_cast<int32_t>(std::min(nvRef.width(), nvTest.width()));
    const int32_t h = static_cast<int32_t>(std::min(nvRef.height(), nvTest.height()));
    const PixelRect overlap{0, 0, w, h};
    const std::optional<FlipGpuResult> gpu = flipGpu.compute(nvRef, 0, 0, nvTest, 0, 0, overlap, defaultFlipPpd());
    if (!gpu)
    {
        results.fail("nvlabs: the GPU FLIP returned nothing");
        return;
    }
    constexpr double expected = 0.159693;
    const double delta = std::fabs(gpu->mean - expected);
    std::cout << strf("[selftest] nvlabs flip gpu=%.9g expected=%.9g delta=%.3g\n", gpu->mean, expected, delta);
    if (!(delta <= 1e-4))
    {
        results.fail(strf("nvlabs: GPU FLIP mean %.9g differs from %.9g by %.3g", gpu->mean, expected, delta));
    }
#ifdef NDEBUG
    const FlipResult cpu = flip(displayRgb(cropImage(*nvRef.hostData(), 0, 0, w, h)),
                                displayRgb(cropImage(*nvTest.hostData(), 0, 0, w, h)), defaultFlipPpd());
    const FlipDiff diff = compareFlip(*gpu, cpu);
    printFlipLine("nvlabs", *gpu, cpu, diff, defaultFlipPpd());
    requireFlipAgrees("nvlabs", diff, results);
#endif
#else
    (void)context;
    (void)flipGpu;
    (void)results;
    std::cout << "[selftest] nvlabs flip SKIPPED: SABERCAT_FLIP_IMAGES_DIR is not defined\n";
#endif
}

// ---- HDR-FLIP: FlipGpu::computeHdr against flipHdr(), and the CompareState side ---------

// A linear image, opaque, of one size. The caller sets the samples.
ImageData hdrImage(uint32_t width, uint32_t height)
{
    ImageData image = blankImage(width, height, Encoding::Linear);
    for (size_t i = 0; i < static_cast<size_t>(width) * height; ++i)
    {
        image.rgba[i * 4 + 3] = 1.0f;
    }
    return image;
}

// A linear image with one value in every rgb sample.
ImageData hdrUniform(uint32_t width, uint32_t height, float value)
{
    ImageData image = hdrImage(width, height);
    for (size_t i = 0; i < image.rgba.size(); i += 4)
    {
        image.rgba[i] = value;
        image.rgba[i + 1] = value;
        image.rgba[i + 2] = value;
    }
    return image;
}

// A linear gradient with noise: values up to about 50, and some samples near 1e-5.
ImageData makeHdr(uint32_t width, uint32_t height, uint32_t seed)
{
    ImageData image = hdrImage(width, height);
    Lcg rng(seed);
    const float channelScale[3] = {1.0f, 0.5f, 2.0f};
    for (uint32_t y = 0; y < height; ++y)
    {
        for (uint32_t x = 0; x < width; ++x)
        {
            float * p = &image.rgba[(static_cast<size_t>(y) * width + x) * 4];
            const float fx = (static_cast<float>(x) + 0.5f) / static_cast<float>(width);
            const float fy = (static_cast<float>(y) + 0.5f) / static_cast<float>(height);
            const float base = 20.0f * (0.2f + fx * fy);
            for (size_t c = 0; c < 3; ++c)
            {
                float value = base * channelScale[c] * (0.25f + rng.next());
                if ((x + 2 * y + c) % 7 == 0)
                {
                    value = 1e-5f * (0.5f + rng.next());
                }
                p[c] = value;
            }
        }
    }
    return image;
}

// A copy with every rgb sample scaled by a factor from 0.9 to 1.1.
ImageData perturbedHdr(const ImageData & source, uint32_t seed)
{
    ImageData image = source;
    Lcg rng(seed);
    for (size_t i = 0; i < image.rgba.size(); i += 4)
    {
        for (size_t c = 0; c < 3; ++c)
        {
            image.rgba[i + c] *= 0.9f + 0.2f * rng.next();
        }
    }
    return image;
}

// makeHdr with values the display never shows: 1e-30, a denormal, 1e4 and a negative sample.
ImageData makeHdrExtremes(uint32_t width, uint32_t height, uint32_t seed)
{
    ImageData image = makeHdr(width, height, seed);
    image.rgba[0] = 1e-30f;
    image.rgba[1] = 1e-40f;
    image.rgba[2] = 1e4f;
    image.rgba[4] = -0.5f;
    return image;
}

// Copies with every non-finite sample set to 0: the rule of the HDR path (a NaN or infinite
// linear sample counts as 0), and the CPU reference's input.
ImageData zeroNonFinite(ImageData image)
{
    for (float & value : image.rgba)
    {
        if (!std::isfinite(value))
        {
            value = 0.0f;
        }
    }
    return image;
}

// The two sources of a pair, alive for the runs that use them.
struct SourcePair
{
    std::unique_ptr<ImageSource> a;
    std::unique_ptr<ImageSource> b;
};

SourcePair makeSourcePair(GpuContext & context, const ImageData & a, const ImageData & b, const std::string & label)
{
    SourcePair pair;
    pair.a = std::make_unique<ImageSource>(context, a, label + "/a");
    pair.b = std::make_unique<ImageSource>(context, b, label + "/b");
    if (pair.a->status() != SourceStatus::Ready || pair.b->status() != SourceStatus::Ready)
    {
        throw std::runtime_error(label + ": a source is not Ready: " + pair.a->statusMessage()
                                 + pair.b->statusMessage());
    }
    return pair;
}

// The share of exposure indices that agree within kHdrExposureTolerance, in percent.
double exposureAgreement(const std::vector<float> & gpu, const std::vector<float> & cpu)
{
    if (gpu.size() != cpu.size() || gpu.empty())
    {
        return 0.0;
    }
    size_t agree = 0;
    for (size_t i = 0; i < gpu.size(); ++i)
    {
        const double difference = std::fabs(static_cast<double>(gpu[i]) - static_cast<double>(cpu[i]));
        agree += difference <= kHdrExposureTolerance ? 1 : 0;
    }
    return 100.0 * static_cast<double>(agree) / static_cast<double>(gpu.size());
}

bool sameBits(const std::vector<float> & a, const std::vector<float> & b)
{
    return bitMismatches(a, b) == 0;
}

// One HDR-FLIP run of a two-source case, and what is checked.
struct HdrCase
{
    Case pair;                  // two images; the layout is makeLayout(pair)
    FlipHdrSettings settings;   // tone mapper and the user range (unset = derived)
    float ppd = 0.0f;
    bool autoRange = false;     // no user range: compare with hdrExposuresFromLinear()
    bool identical = false;     // the pair is one image: every value must be 0
    bool runCpu = true;         // compare the maps with flipHdr() on the crops
    bool cpuOptional = false;   // a CPU refusal is printed, not failed
};

// A case of two images at the default ppd, with the derived range and the given tone mapper.
HdrCase hdrCase(const std::string & name, ImageData a, ImageData b, bool centered, ToneMapper toneMapper)
{
    HdrCase hc;
    hc.pair = makeCase(name, {std::move(a), std::move(b)}, centered);
    hc.settings.toneMapper = toneMapper;
    hc.ppd = defaultFlipPpd();
    hc.autoRange = true;
    return hc;
}

FlipHdrSettings hdrRange(std::optional<float> start, std::optional<float> stop, std::optional<int32_t> count)
{
    FlipHdrSettings settings;
    settings.start = start;
    settings.stop = stop;
    settings.count = count;
    return settings;
}

// Runs one HDR-FLIP pair on the GPU and checks it: the overlap and the sizes, the non-finite
// count, the derived range (against hdrExposuresFromLinear() on the zeroed crop), the identical
// rule, and the maps against flipHdr() run with the GPU's exposures. Returns the GPU result, or
// nullopt after a reported failure. The map stays on flipGpu until the next run.
std::optional<FlipGpuResult> checkHdrCase(GpuContext & context, FlipGpu & flipGpu, const HdrCase & hc,
                                          Results & results)
{
    const std::string & name = hc.pair.name;
    const LayoutInput layout = makeLayout(hc.pair);
    const PixelRect overlap = overlapRect(layout, 0, 1);
    if (overlap.width <= 0 || overlap.height <= 0)
    {
        results.fail(name + ": the pair has no overlap");
        return std::nullopt;
    }
    const int32_t ax = layout.sources[0].offsetX;
    const int32_t ay = layout.sources[0].offsetY;
    const int32_t bx = layout.sources[1].offsetX;
    const int32_t by = layout.sources[1].offsetY;
    const size_t pixels = static_cast<size_t>(overlap.width) * static_cast<size_t>(overlap.height);

    SourcePair sources;
    try
    {
        sources = makeSourcePair(context, hc.pair.images[0], hc.pair.images[1], name);
    }
    catch (const std::exception & e)
    {
        results.fail(e.what());
        return std::nullopt;
    }
    std::optional<FlipGpuResult> gpu;
    try
    {
        gpu = flipGpu.computeHdr(*sources.a, ax, ay, *sources.b, bx, by, overlap, hc.ppd, hc.settings);
    }
    catch (const std::exception & e)
    {
        results.fail(name + ": the GPU HDR-FLIP threw: " + e.what());
        return std::nullopt;
    }
    if (!gpu || !gpu->hdr)
    {
        results.fail(name + ": the GPU HDR-FLIP returned no result");
        return std::nullopt;
    }
    const FlipHdrInfo info = *gpu->hdr;
    if (!sameRect(gpu->overlap, overlap) || gpu->errorMap.size() != pixels || info.exposureMap.size() != pixels
        || info.toneMapper != hc.settings.toneMapper || gpu->ppd != hc.ppd)
    {
        results.fail(name + ": the GPU HDR-FLIP result has the wrong overlap, map sizes, tone mapper or ppd");
    }

    // The overlap crops: the raw ones count the non-finite samples, the zeroed ones are the input
    // of the CPU reference and of the range rule.
    const ImageData rawA = cropImage(hc.pair.images[0], overlap.x - ax, overlap.y - ay, overlap.width, overlap.height);
    const ImageData rawB = cropImage(hc.pair.images[1], overlap.x - bx, overlap.y - by, overlap.width, overlap.height);
    const uint64_t expectedNonFinite = countNonFiniteRgb(rawA) + countNonFiniteRgb(rawB);
    if (gpu->nonFiniteSamples != expectedNonFinite)
    {
        results.fail(name + strf(": the GPU counted %llu non-finite samples, expected %llu",
                                 static_cast<unsigned long long>(gpu->nonFiniteSamples),
                                 static_cast<unsigned long long>(expectedNonFinite)));
    }
    const ImageData refCrop = zeroNonFinite(rawA);
    const ImageData testCrop = zeroNonFinite(rawB);

    if (hc.autoRange)
    {
        HdrFlipParams derive;
        derive.toneMapper = hc.settings.toneMapper;
        derive.ppd = hc.ppd;
        try
        {
            const HdrExposures range = hdrExposuresFromLinear(linearRgb(refCrop), static_cast<uint32_t>(overlap.width),
                                                              static_cast<uint32_t>(overlap.height), derive, true);
            if (range.start != info.start || range.stop != info.stop || range.count != info.count)
            {
                results.fail(name + strf(": the GPU range %.9g..%.9g (%d) differs from hdrExposuresFromLinear "
                                         "%.9g..%.9g (%d)",
                                         static_cast<double>(info.start), static_cast<double>(info.stop), info.count,
                                         static_cast<double>(range.start), static_cast<double>(range.stop),
                                         range.count));
            }
        }
        catch (const std::exception & e)
        {
            results.fail(name + ": hdrExposuresFromLinear threw: " + e.what());
        }
    }

    if (hc.identical)
    {
        size_t nonZero = 0;
        for (const float value : gpu->errorMap)
        {
            nonZero += value == 0.0f ? 0 : 1;
        }
        size_t exposureNonZero = 0;
        for (const float value : info.exposureMap)
        {
            exposureNonZero += value == 0.0f ? 0 : 1;
        }
        if (gpu->mean != 0.0 || nonZero != 0 || exposureNonZero != 0 || info.count < kMinHdrExposures)
        {
            results.fail(name + strf(": identical images give mean %.9g, %zu nonzero pixels and %zu nonzero "
                                     "exposures",
                                     gpu->mean, nonZero, exposureNonZero));
        }
    }

    if (!hc.runCpu)
    {
        std::cout << strf("[selftest] hdr %s gpu=%.9g cpu=skipped start=%.4f stop=%.4f count=%d tone=%s\n",
                          name.c_str(), gpu->mean, static_cast<double>(info.start), static_cast<double>(info.stop),
                          info.count, toneMapperName(info.toneMapper));
        return gpu;
    }

    // The CPU reference runs with the exposures the GPU used, so the maps are comparable.
    HdrFlipParams exact;
    exact.toneMapper = hc.settings.toneMapper;
    exact.ppd = hc.ppd;
    exact.startExposure = info.start;
    exact.stopExposure = info.stop;
    exact.numExposures = info.count;
    HdrFlipResult cpu;
    try
    {
        cpu = flipHdr(refCrop, testCrop, exact);
    }
    catch (const std::exception & e)
    {
        if (hc.cpuOptional)
        {
            std::cout << strf("[selftest] hdr %s gpu=%.9g cpu=refused (%s) tone=%s\n", name.c_str(), gpu->mean,
                              e.what(), toneMapperName(info.toneMapper));
        }
        else
        {
            results.fail(name + ": flipHdr threw: " + e.what());
        }
        return gpu;
    }
    const FlipDiff diff = diffMaps(gpu->errorMap, cpu.errorMap, gpu->mean, cpu.meanError);
    const double agreement = exposureAgreement(info.exposureMap, cpu.exposureMap);
    std::cout << strf("[selftest] hdr %s gpu=%.9g cpu=%.9g dmean=%.3g p999=%.3g expAgree=%.3f%% start=%.4f "
                      "stop=%.4f count=%d tone=%s\n",
                      name.c_str(), gpu->mean, cpu.meanError, diff.mean, diff.p999, agreement,
                      static_cast<double>(info.start), static_cast<double>(info.stop), info.count,
                      toneMapperName(info.toneMapper));
    if (!diff.sameSize || !diff.finite || diff.mean > kMaxHdrMeanDelta || diff.p999 > kMaxFlipP999
        || (agreement < kMinHdrExposureAgreement && info.count < kHdrManyExposures))
    {
        results.fail(name + strf(": GPU HDR-FLIP differs from flipHdr() (dmean=%.3g p999=%.3g expAgree=%.3f%%)",
                                 diff.mean, diff.p999, agreement));
    }
    return gpu;
}

// The synthetic pairs: the same size, the identical pair, the centered and top-left sizes, one
// pixel, a wide and a tall strip, the extreme values, the three tone mappers on one mismatched
// pair, and the two degenerate ranges.
void checkHdrCases(GpuContext & context, FlipGpu & flipGpu, Results & results)
{
    const auto run = [&](const HdrCase & hc) { return checkHdrCase(context, flipGpu, hc, results); };

    {
        const ImageData a = makeHdr(40, 30, 1);
        (void)run(hdrCase("same-size", a, perturbedHdr(a, 2), false, ToneMapper::Aces));
    }
    {
        const ImageData a = makeHdr(37, 23, 3);
        HdrCase hc = hdrCase("identical", a, a, false, ToneMapper::Aces);
        hc.identical = true;
        (void)run(hc);
    }
    {
        (void)run(hdrCase("sizes-centered", makeHdr(40, 30, 4), makeHdr(25, 45, 5), true, ToneMapper::Aces));
    }
    {
        const ImageData a = makeHdr(1, 1, 10);
        (void)run(hdrCase("one-pixel", a, perturbedHdr(a, 11), false, ToneMapper::Aces));
    }
    {
        const ImageData a = makeHdr(300, 3, 12);
        (void)run(hdrCase("wide", a, perturbedHdr(a, 13), false, ToneMapper::Aces));
    }
    {
        const ImageData a = makeHdr(3, 300, 14);
        (void)run(hdrCase("tall", a, perturbedHdr(a, 15), false, ToneMapper::Aces));
    }
    {
        const ImageData a = makeHdrExtremes(40, 30, 20);
        (void)run(hdrCase("extremes", a, perturbedHdr(a, 21), false, ToneMapper::Aces));
    }
    for (const ToneMapper tone : {ToneMapper::Aces, ToneMapper::Hable, ToneMapper::Reinhard})
    {
        (void)run(hdrCase("sizes-topleft", makeHdr(40, 30, 4), makeHdr(25, 45, 5), false, tone));
    }

    // Degenerate ranges: a uniform reference gives two exposures at one value; an all-black
    // reference (the CPU rule throws for it) gives the epsilon fallback, two exposures at one value.
    {
        const ImageData uniform = hdrUniform(32, 24, 0.5f);
        HdrCase hc = hdrCase("uniform", uniform, makeHdr(32, 24, 31), false, ToneMapper::Aces);
        hc.cpuOptional = true;
        const std::optional<FlipGpuResult> gpu = run(hc);
        if (gpu && gpu->hdr)
        {
            const bool ok = gpu->hdr->count == 2 && gpu->hdr->start == gpu->hdr->stop && std::isfinite(gpu->mean);
            if (!ok)
            {
                results.fail("uniform: a uniform reference must give two exposures at one value and a finite mean");
            }
        }
    }
    {
        HdrCase hc = hdrCase("all-black", hdrUniform(32, 24, 0.0f), makeHdr(32, 24, 32), false, ToneMapper::Aces);
        hc.runCpu = false;
        const std::optional<FlipGpuResult> gpu = run(hc);
        if (gpu && gpu->hdr)
        {
            bool finite = std::isfinite(gpu->mean);
            for (const float value : gpu->errorMap)
            {
                finite = finite && std::isfinite(value);
            }
            const bool ok = gpu->hdr->count == 2 && gpu->hdr->start == gpu->hdr->stop && finite;
            if (!ok)
            {
                results.fail("all-black: a black reference must give two exposures at one value and finite values");
            }
        }
    }
}

// NaN and infinite linear samples: counted, treated as 0, never NaN in the result.
void checkHdrNonFinite(GpuContext & context, FlipGpu & flipGpu, Results & results)
{
    ImageData a = makeHdr(24, 16, 71);
    ImageData b = makeHdr(24, 16, 72);
    const float notANumber = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    // Pixel p, channel c is rgba[4 p + c]: three NaN and one +inf in A, one -inf in B.
    a.rgba[4 * 5 + 0] = notANumber;
    a.rgba[4 * 17 + 1] = notANumber;
    a.rgba[4 * 100 + 2] = notANumber;
    a.rgba[4 * 200 + 0] = infinity;
    b.rgba[4 * 50 + 1] = -infinity;

    const std::optional<FlipGpuResult> gpu = checkHdrCase(context, flipGpu, hdrCase("nonfinite", a, b, false,
                                                                                    ToneMapper::Aces),
                                                          results);
    if (gpu)
    {
        std::cout << strf("[selftest] hdr nonfinite nonFinite=%llu mean finite=%s\n",
                          static_cast<unsigned long long>(gpu->nonFiniteSamples),
                          std::isfinite(gpu->mean) ? "yes" : "no");
        if (gpu->nonFiniteSamples != 5 || !std::isfinite(gpu->mean))
        {
            results.fail("hdr nonfinite: the 5 non-finite samples are not counted or the mean is not finite");
        }
    }
}

// The NVlabs HDR pair (reference.exr and test.exr in SABERCAT_FLIP_IMAGES_DIR): FLIP's exposure
// range, count and mean. Release builds also compare the maps with flipHdr(); Debug CPU FLIP takes
// minutes, so there the CPU comparison is skipped.
void checkHdrNvlabs(GpuContext & context, FlipGpu & flipGpu, Results & results)
{
#ifdef SABERCAT_FLIP_IMAGES_DIR
    const std::string dir = SABERCAT_FLIP_IMAGES_DIR;
    const std::string referencePath = joinPath(dir, "reference.exr");
    const std::string testPath = joinPath(dir, "test.exr");
    std::error_code ec;
    if (!fs::exists(pathFromUtf8(referencePath), ec) || !fs::exists(pathFromUtf8(testPath), ec))
    {
        std::cout << "[selftest] hdr nvlabs SKIPPED: " << dir << " not found\n";
        return;
    }
    ImageData reference;
    ImageData test;
    try
    {
        reference = loadImage(referencePath);
        test = loadImage(testPath);
    }
    catch (const std::exception & e)
    {
        results.fail(std::string("hdr nvlabs: an image did not load: ") + e.what());
        return;
    }
    const int32_t w = static_cast<int32_t>(std::min(reference.width, test.width));
    const int32_t h = static_cast<int32_t>(std::min(reference.height, test.height));
    HdrCase hc = hdrCase("nvlabs", cropImage(reference, 0, 0, w, h), cropImage(test, 0, 0, w, h), false,
                         ToneMapper::Aces);
#ifdef NDEBUG
    hc.runCpu = true;
#else
    hc.runCpu = false;
    std::cout << "[selftest] hdr nvlabs CPU comparison SKIPPED in Debug (CPU FLIP takes minutes)\n";
#endif
    const std::optional<FlipGpuResult> gpu = checkHdrCase(context, flipGpu, hc, results);
    if (!gpu || !gpu->hdr)
    {
        return;
    }
    // hdrExposures() is FLIP's rule on the reference alone: the GPU range must equal it exactly.
    const HdrExposures range = hdrExposures(zeroNonFinite(hc.pair.images[0]), ToneMapper::Aces);
    const FlipHdrInfo & info = *gpu->hdr;
    const bool rangeOk = info.count == kNvlabsHdrCount && std::fabs(info.start - kNvlabsHdrStart) < 5e-4
        && std::fabs(info.stop - kNvlabsHdrStop) < 5e-4 && info.start == range.start && info.stop == range.stop
        && info.count == range.count;
    const double delta = std::fabs(gpu->mean - kNvlabsHdrMean);
    std::cout << strf("[selftest] hdr nvlabs gpu=%.9g expected=%.9g delta=%.3g start=%.4f stop=%.4f count=%d\n",
                      gpu->mean, kNvlabsHdrMean, delta, static_cast<double>(info.start),
                      static_cast<double>(info.stop), info.count);
    if (!rangeOk)
    {
        results.fail(strf("hdr nvlabs: the range %.4f..%.4f (%d) is not the NVlabs range or hdrExposures()",
                          static_cast<double>(info.start), static_cast<double>(info.stop), info.count));
    }
    if (!(delta <= 2e-4))
    {
        results.fail(strf("hdr nvlabs: GPU HDR-FLIP mean %.9g differs from %.9g by %.3g", gpu->mean, kNvlabsHdrMean,
                          delta));
    }
#else
    (void)context;
    (void)flipGpu;
    std::cout << "[selftest] hdr nvlabs SKIPPED: SABERCAT_FLIP_IMAGES_DIR is not defined\n";
#endif
    (void)results;
}

// What one run of computeHdr did: "ok", "nullopt", "invalid_argument", "runtime_error", or the
// text of another exception. The result is kept in out when it ran.
std::string runHdrOutcome(FlipGpu & flipGpu, const MediaSource & a, const MediaSource & b, PixelRect overlap,
                          const FlipHdrSettings & settings, std::optional<FlipGpuResult> & out)
{
    try
    {
        out = flipGpu.computeHdr(a, 0, 0, b, 0, 0, overlap, defaultFlipPpd(), settings);
        return out ? "ok" : "nullopt";
    }
    catch (const std::invalid_argument &)
    {
        return "invalid_argument";
    }
    catch (const std::runtime_error &)
    {
        return "runtime_error";
    }
    catch (const std::exception & e)
    {
        return std::string("other: ") + e.what();
    }
}

// A stepped run: beginHdr, stepHdr(perCall) until it reports done, then finishHdr.
struct SteppedRun
{
    std::optional<FlipGpuResult> result;
    uint32_t total = 0;         // exposures of the job
    uint32_t calls = 0;         // stepHdr calls
    bool stepsOk = true;        // every call advanced hdrDone by min(perCall, the rest)
    std::string error;          // empty when the run finished
};

SteppedRun runStepped(FlipGpu & flipGpu, const MediaSource & a, const MediaSource & b, PixelRect overlap,
                      const FlipHdrSettings & settings, uint32_t perCall)
{
    SteppedRun run;
    try
    {
        if (!flipGpu.beginHdr(a, 0, 0, b, 0, 0, overlap, defaultFlipPpd(), settings))
        {
            run.error = "beginHdr refused the pair";
            return run;
        }
        run.total = flipGpu.hdrCount();
        run.stepsOk = flipGpu.hdrActive() && flipGpu.hdrDone() == 0;
        bool done = false;
        while (!done && run.calls < 100000)
        {
            const uint32_t before = flipGpu.hdrDone();
            done = flipGpu.stepHdr(perCall);
            ++run.calls;
            const uint32_t expected = std::min(before + perCall, run.total);
            run.stepsOk = run.stepsOk && flipGpu.hdrDone() == expected;
        }
        run.result = flipGpu.finishHdr();
    }
    catch (const std::exception & e)
    {
        run.error = e.what();
        flipGpu.cancelHdr();
    }
    return run;
}

// The exposure range overrides and the job API, on a 16 x 16 pair and a 40 x 30 pair.
void checkHdrOverrides(GpuContext & context, FlipGpu & flipGpu, Results & results)
{
    const ImageData smallA = makeHdr(16, 16, 151);
    const ImageData smallB = perturbedHdr(smallA, 152);
    const SourcePair small = makeSourcePair(context, smallA, smallB, "hdr overrides");
    const PixelRect smallOverlap{0, 0, 16, 16};

    // Refused: std::invalid_argument, and no map is left behind.
    const float notANumber = std::numeric_limits<float>::quiet_NaN();
    const std::vector<std::pair<std::string, FlipHdrSettings>> refused = {
        {"start above stop", hdrRange(2.0f, -1.0f, std::nullopt)},
        {"count 1", hdrRange(std::nullopt, std::nullopt, 1)},
        {"count 0", hdrRange(std::nullopt, std::nullopt, 0)},
        {"count -1", hdrRange(std::nullopt, std::nullopt, -1)},
        {"count 1001", hdrRange(std::nullopt, std::nullopt, 1001)},
        {"start 200", hdrRange(200.0f, std::nullopt, std::nullopt)},
        {"stop NaN", hdrRange(std::nullopt, notANumber, std::nullopt)},
    };
    for (const auto & [name, settings] : refused)
    {
        std::optional<FlipGpuResult> out;
        const std::string outcome = runHdrOutcome(flipGpu, *small.a, *small.b, smallOverlap, settings, out);
        const bool noMap = !flipGpu.errorMapView().valid();
        std::cout << strf("[selftest] hdr overrides %s: %s, no map=%s\n", name.c_str(), outcome.c_str(),
                           noMap ? "yes" : "no");
        if (outcome != "invalid_argument" || !noMap)
        {
            results.fail("hdr overrides: " + name + " is not refused with std::invalid_argument and no map");
        }
    }

    // Accepted: count 2, count 1000 (CPU only in Release), start == stop, and the derived range after
    // all the refusals.
    {
        HdrCase hc = hdrCase("overrides-count2", smallA, smallB, false, ToneMapper::Aces);
        hc.settings = hdrRange(-2.0f, 2.0f, 2);
        hc.autoRange = false;
        const std::optional<FlipGpuResult> gpu = checkHdrCase(context, flipGpu, hc, results);
        if (gpu && gpu->hdr && gpu->hdr->count != 2)
        {
            results.fail("hdr overrides: count 2 did not give two exposures");
        }
    }
    {
        HdrCase hc = hdrCase("overrides-count1000", smallA, smallB, false, ToneMapper::Aces);
        hc.settings = hdrRange(-2.0f, 2.0f, 1000);
        hc.autoRange = false;
#ifdef NDEBUG
        hc.runCpu = true;
#else
        hc.runCpu = false;
#endif
        const std::optional<FlipGpuResult> gpu = checkHdrCase(context, flipGpu, hc, results);
        if (gpu && gpu->hdr && gpu->hdr->count != 1000)
        {
            results.fail("hdr overrides: count 1000 did not give 1000 exposures");
        }
    }
    {
        HdrCase hc = hdrCase("overrides-equal", smallA, smallB, false, ToneMapper::Aces);
        hc.settings = hdrRange(1.0f, 1.0f, std::nullopt);
        hc.autoRange = false;
        const std::optional<FlipGpuResult> gpu = checkHdrCase(context, flipGpu, hc, results);
        if (gpu && gpu->hdr && !(gpu->hdr->start == 1.0f && gpu->hdr->stop == 1.0f && gpu->hdr->count == 2))
        {
            results.fail("hdr overrides: start == stop did not give a range of two exposures at 1");
        }
    }
    (void)checkHdrCase(context, flipGpu, hdrCase("overrides-auto", smallA, smallB, false, ToneMapper::Aces),
                       results);

    // The job API, on a 40 x 30 pair.
    const ImageData jobA = makeHdr(40, 30, 161);
    const ImageData jobB = perturbedHdr(jobA, 162);
    const SourcePair job = makeSourcePair(context, jobA, jobB, "hdr overrides job");
    const PixelRect jobOverlap{0, 0, 40, 30};
    const FlipHdrSettings autoSettings;
    std::optional<FlipGpuResult> reference;
    try
    {
        reference = flipGpu.computeHdr(*job.a, 0, 0, *job.b, 0, 0, jobOverlap, defaultFlipPpd(), autoSettings);
    }
    catch (const std::exception & e)
    {
        results.fail(std::string("hdr job: the reference run threw: ") + e.what());
        return;
    }
    if (!reference || !reference->hdr)
    {
        results.fail("hdr job: the reference run returned no result");
        return;
    }

    const SteppedRun one = runStepped(flipGpu, *job.a, *job.b, jobOverlap, autoSettings, 1);
    const bool oneOk = one.error.empty() && one.stepsOk && one.calls == one.total
        && static_cast<int32_t>(one.total) == reference->hdr->count && !flipGpu.hdrActive() && one.result
        && one.result->hdr && sameBits(one.result->errorMap, reference->errorMap)
        && sameBits(one.result->hdr->exposureMap, reference->hdr->exposureMap)
        && sameBytes(one.result->mean, reference->mean);
    const SteppedRun three = runStepped(flipGpu, *job.a, *job.b, jobOverlap, autoSettings, 3);
    const bool threeOk = three.error.empty() && three.stepsOk && three.result && three.result->hdr
        && sameBits(three.result->errorMap, reference->errorMap)
        && sameBits(three.result->hdr->exposureMap, reference->hdr->exposureMap)
        && sameBytes(three.result->mean, reference->mean);

    // finishHdr before the exposures are done, and a stepHdr or finishHdr with no job.
    bool finishEarly = false;
    if (flipGpu.beginHdr(*job.a, 0, 0, *job.b, 0, 0, jobOverlap, defaultFlipPpd(), autoSettings))
    {
        try
        {
            (void)flipGpu.finishHdr();
        }
        catch (const std::logic_error &)
        {
            finishEarly = true;
        }
        flipGpu.cancelHdr();
    }
    bool noJobStep = false;
    try
    {
        (void)flipGpu.stepHdr(1);
    }
    catch (const std::logic_error &)
    {
        noJobStep = true;
    }

    // cancelHdr ends a job with no result and no valid map.
    bool cancelled = false;
    if (flipGpu.beginHdr(*job.a, 0, 0, *job.b, 0, 0, jobOverlap, defaultFlipPpd(), autoSettings))
    {
        (void)flipGpu.stepHdr(1);
        flipGpu.cancelHdr();
        cancelled = !flipGpu.hdrActive() && !flipGpu.errorMapView().valid();
    }

    std::cout << strf("[selftest] hdr job: exposures=%u one-by-one=%s three-at-a-time=%s finishEarly=%s "
                       "cancel=%s noJobStep=%s\n",
                       one.total, oneOk ? "same" : "DIFFERENT", threeOk ? "same" : "DIFFERENT",
                       finishEarly ? "yes" : "no", cancelled ? "yes" : "no", noJobStep ? "yes" : "no");
    if (!oneOk)
    {
        results.fail("hdr job: stepHdr(1) does not advance one exposure at a time to the same map: " + one.error);
    }
    if (!threeOk)
    {
        results.fail("hdr job: stepHdr(3) does not give the same map as computeHdr: " + three.error);
    }
    if (!finishEarly || !cancelled || !noJobStep)
    {
        results.fail("hdr job: finishHdr before done, cancelHdr or stepHdr with no job does not behave");
    }
}

// The memory budget: a free-bytes override too small for the pair refuses the run with the
// "FLIP needs about" message and no job; after the scope the same run works.
void checkHdrBudget(GpuContext & context, FlipGpu & flipGpu, Results & results)
{
    const ImageData a = makeHdr(160, 120, 171);
    const SourcePair pair = makeSourcePair(context, a, perturbedHdr(a, 172), "hdr budget");
    const PixelRect full{0, 0, 160, 120};
    // A fresh start: buffers the object already holds do not count against the budget.
    flipGpu.release();
    bool refused = false;
    std::string message;
    {
        BudgetOverrideScope budget(1048576);
        try
        {
            (void)flipGpu.beginHdr(*pair.a, 0, 0, *pair.b, 0, 0, full, defaultFlipPpd(), FlipHdrSettings{});
        }
        catch (const std::runtime_error & e)
        {
            message = e.what();
            refused = message.rfind("FLIP needs about", 0) == 0;
        }
    }
    const bool jobless = !flipGpu.hdrActive() && !flipGpu.errorMapView().valid();
    std::optional<FlipGpuResult> after;
    try
    {
        after = flipGpu.computeHdr(*pair.a, 0, 0, *pair.b, 0, 0, full, defaultFlipPpd(), FlipHdrSettings{});
    }
    catch (const std::exception & e)
    {
        message += std::string(" | after: ") + e.what();
    }
    const bool restored = after && after->hdr && after->errorMap.size() == static_cast<size_t>(160) * 120;
    std::cout << strf("[selftest] hdr budget: refused=%s no job=%s restored=%s (\"%s\")\n", refused ? "yes" : "no",
                      jobless ? "yes" : "no", restored ? "yes" : "no", message.c_str());
    if (!refused || !jobless || !restored)
    {
        results.fail("hdr budget: a run over the GPU budget is not refused with no job, or does not work after");
    }
}

// The CompareState side of HDR-FLIP: availability and the mode, the run and its progress, the
// display-only settings, the range refusals, the mixed-encoding warning, the project, the budget.
void checkHdrState(GpuContext & context, MetricsGpu & metrics, FlipGpu & flipGpu, const std::string & outDir,
                   Results & results)
{
    const auto report = [&results](const std::string & name, bool ok, const std::string & detail)
    {
        std::cout << "[selftest] hdr state " << name << ": " << (ok ? "ok" : "FAILED") << " (" << detail << ")\n";
        if (!ok)
        {
            results.fail("hdr state " + name + ": " + detail);
        }
    };
    const auto addPair = [&context](CompareState & state, ImageData a, ImageData b, const std::string & label)
    {
        state.addMemorySource(std::make_unique<ImageSource>(context, a, label + "-a"), label + "-a");
        state.addMemorySource(std::make_unique<ImageSource>(context, std::move(b), label + "-b"), label + "-b");
    };
    const auto sameRange = [](const FlipHdrSettings & x, const FlipHdrSettings & y)
    {
        return x.toneMapper == y.toneMapper && x.start == y.start && x.stop == y.stop && x.count == y.count;
    };
    const auto stepToEnd = [](CompareState & state)
    {
        bool done = false;
        int calls = 0;
        while (!done && calls < 100000)
        {
            done = state.stepHdr(0.0);
            ++calls;
        }
        return done;
    };
    const ImageData linA = makeHdr(40, 30, 181);
    const ImageData linB = perturbedHdr(linA, 182);

    // (a) A display-only pair: no HDR, the mode stays LDR, the status says why.
    {
        CompareState state(context);
        addPair(state, makeDisplay(40, 30, 183), makeDisplay(40, 30, 184), "hdr-display");
        const bool unavailable = !state.hdrAvailable();
        const bool refused = !state.setFlipMode(project::FlipMode::Hdr);
        const bool ldr = state.flipMode() == project::FlipMode::Ldr;
        const bool status = state.status() == "HDR-FLIP needs a Linear (EXR) source in the A/B pair";
        report("display-only", unavailable && refused && ldr && status, "status \"" + state.status() + "\"");
    }

    // (b) A linear pair: the run waits for the request, steps with progress, and matches computeHdr.
    {
        CompareState state(context);
        addPair(state, linA, linB, "hdr-b");
        if (!state.hasPair())
        {
            report("run", false, "no pair");
            return;
        }
        const bool available = state.hdrAvailable();
        const bool setHdr = state.setFlipMode(project::FlipMode::Hdr);
        state.updateMetrics(metrics);
        const bool pending = !state.flip() && state.flipReason().find("press Run HDR-FLIP") != std::string::npos
            && !state.hdrRunning();
        const bool started = state.requestHdr();
        const bool running = state.hdrRunning();
        bool monotone = true;
        float last = state.hdrProgress();
        bool done = false;
        int calls = 0;
        while (!done && calls < 100000)
        {
            done = state.stepHdr(0.0);
            ++calls;
            if (!done)
            {
                const float progress = state.hdrProgress();
                monotone = monotone && progress >= last;
                last = progress;
            }
        }
        const bool finished = done && state.flip().has_value() && state.flip()->hdr.has_value() && !state.hdrRunning();

        const size_t ia = static_cast<size_t>(state.sourceA());
        const size_t ib = static_cast<size_t>(state.sourceB());
        const PixelRect overlap = overlapRect(state.layout(), static_cast<int>(ia), static_cast<int>(ib));
        const std::optional<FlipGpuResult> reference = flipGpu.computeHdr(
            *state.entries()[ia].media, state.layout().sources[ia].offsetX, state.layout().sources[ia].offsetY,
            *state.entries()[ib].media, state.layout().sources[ib].offsetX, state.layout().sources[ib].offsetY,
            overlap, state.flipPpd(), state.hdrSettings());
        const bool sameMean = state.flip() && reference && std::fabs(state.flip()->mean - reference->mean) <= 1e-12;
        report("run", available && setHdr && pending && started && running && monotone && finished && sameMean,
               strf("calls=%d progress monotone=%d mean=%.9g computeHdr=%.9g", calls, monotone ? 1 : 0,
                    state.flip() ? state.flip()->mean : -1.0, reference ? reference->mean : -1.0));

        // (c) Display EV and tone mapper: the metrics and the HDR result stay bit for bit.
        const std::optional<MetricsResult> metricsBefore = state.metrics();
        const double meanBefore = state.flip() ? state.flip()->mean : 0.0;
        const std::vector<float> mapBefore = state.flip() && state.flip()->hdr ? state.flip()->hdr->exposureMap
                                                                              : std::vector<float>();
        const double linearBefore = state.flip() && state.flip()->hdr ? state.flip()->hdr->linearMse : 0.0;
        const bool exposureSet = state.setExposure(ia, 3.0f);
        const bool toneSet = state.setToneMapper(ib, project::ToneMapper::Hable);
        state.updateMetrics(metrics);
        const bool metricsSame = metricsBefore && state.metrics() && sameBytes(state.metrics()->mse, metricsBefore->mse)
            && sameBytes(state.metrics()->psnrDb, metricsBefore->psnrDb);
        const bool flipKept = state.flip() && state.flip()->hdr && sameBytes(state.flip()->mean, meanBefore)
            && sameBits(state.flip()->hdr->exposureMap, mapBefore)
            && sameBytes(state.flip()->hdr->linearMse, linearBefore);
        report("display-only settings", exposureSet && toneSet && metricsSame && flipKept,
               "exposure and tone mapper change neither the metrics nor the HDR result");

        // (i) and (j): switching back to LDR gives the LDR result again.
        const bool ldrSet = state.setFlipMode(project::FlipMode::Ldr);
        state.updateMetrics(metrics);
        const bool ldrBack = ldrSet && state.flip().has_value() && !state.flip()->hdr.has_value()
            && state.flipReason().empty();
        report("back to LDR", ldrBack, "LDR result, no HDR section");
    }

    // (d) A source change while a job runs cancels it and drops the result; a new request works.
    {
        CompareState state(context);
        addPair(state, linA, linB, "hdr-d");
        state.setFlipMode(project::FlipMode::Hdr);
        state.updateMetrics(metrics);
        const bool started = state.requestHdr();
        const bool running = state.hdrRunning();
        const bool moved = state.setOffset(static_cast<size_t>(state.sourceB()), 2, 1);
        const bool cancelled = !state.hdrRunning() && !state.flip().has_value();
        const bool restarted = state.requestHdr();
        const bool finished = restarted && stepToEnd(state) && state.flip().has_value() && state.flip()->hdr.has_value();
        report("cancel on change", started && running && moved && cancelled && restarted && finished,
               "a source change cancels the job and drops the result; a new request runs");
    }

    // (e) The range: refused values leave the settings and set the status; the accepted ones stand.
    {
        CompareState state(context);
        addPair(state, linA, linB, "hdr-e");
        state.setFlipMode(project::FlipMode::Hdr);
        const FlipHdrSettings before = state.hdrSettings();
        const bool refusedOrder = !state.setHdrRange(2.0f, -1.0f, std::nullopt);
        const bool refusedOne = !state.setHdrRange(std::nullopt, std::nullopt, 1);
        const bool refusedMany = !state.setHdrRange(std::nullopt, std::nullopt, 1001);
        const bool refusedNan = !state.setHdrRange(std::numeric_limits<float>::quiet_NaN(), std::nullopt,
                                                   std::nullopt);
        const bool statusSet = !state.status().empty();
        const bool unchanged = sameRange(before, state.hdrSettings());
        const bool accepted = state.setHdrRange(std::nullopt, std::nullopt, std::nullopt)
            && state.setHdrRange(std::nullopt, std::nullopt, 2) && state.setHdrRange(std::nullopt, std::nullopt, 1000)
            && state.setHdrRange(std::nullopt, std::nullopt, std::nullopt);
        report("range", refusedOrder && refusedOne && refusedMany && refusedNan && statusSet && unchanged && accepted,
               "start above stop, count 1 and 1001, NaN refused; auto, 2 and 1000 accepted");
    }

    // (f) The mixed pair warns and runs; a same-encoding pair does not warn.
    {
        CompareState mixed(context);
        addPair(mixed, makeDisplay(40, 30, 191), makeHdr(40, 30, 192), "hdr-f-mixed");
        const bool warned = !mixed.pairEncodingWarning().empty();
        mixed.setFlipMode(project::FlipMode::Hdr);
        mixed.runHdrBlocking();
        const bool linearFinite = mixed.flip().has_value() && mixed.flip()->hdr.has_value()
            && std::isfinite(mixed.flip()->hdr->linearMse);
        CompareState same(context);
        addPair(same, linA, linB, "hdr-f-same");
        const bool silent = same.pairEncodingWarning().empty();
        report("mixed encodings", warned && silent && linearFinite,
               "warning for a Display and Linear pair only; HDR on the mixed pair has a finite linear MSE");
    }

    // (g) The project: the HDR settings survive toProject, the text, and fromProject; the default is LDR.
    {
        CompareState source(context);
        addPair(source, linA, linB, "hdr-g");
        const bool toneOk = source.setHdrToneMapper(ToneMapper::Hable);
        const bool rangeOk = source.setHdrRange(-3.0f, 2.0f, 7);
        const bool modeOk = source.setFlipMode(project::FlipMode::Hdr);
        const project::CompareProject saved = source.toProject();
        const bool savedFields = saved.flip.mode == project::FlipMode::Hdr
            && saved.flip.toneMapper == project::ToneMapper::Hable && saved.flip.startExposure == -3.0f
            && saved.flip.stopExposure == 2.0f && saved.flip.numExposures == 7;
        const std::string text = project::serializeCompareProject(saved, outDir);
        CompareState restored(context);
        bool restoredOk = false;
        try
        {
            restored.fromProject(project::parseCompareProject(text, outDir, nullptr));
            restoredOk = restored.flipMode() == project::FlipMode::Hdr
                && restored.hdrSettings().toneMapper == ToneMapper::Hable && restored.hdrSettings().start == -3.0f
                && restored.hdrSettings().stop == 2.0f && restored.hdrSettings().count == 7;
        }
        catch (const std::exception & e)
        {
            std::cout << "[selftest] hdr state project: " << e.what() << '\n';
        }
        CompareState plain(context);
        const project::CompareProject plainSaved = plain.toProject();
        const std::string plainText = project::serializeCompareProject(plainSaved, outDir);
        const bool defaultLdr = plainSaved.flip.mode == project::FlipMode::Ldr
            && plainText.find("\"ldr\"") != std::string::npos;
        report("project", toneOk && rangeOk && modeOk && savedFields && restoredOk && defaultLdr,
               "Hdr, Hable, -3..2 by 7 survive toProject, the text and fromProject; the default saves ldr");
    }

    // (h) A GPU budget refusal through the state: the reason says so, and the run works after.
    {
        CompareState state(context);
        const ImageData bigA = makeHdr(160, 120, 201);
        addPair(state, bigA, perturbedHdr(bigA, 202), "hdr-h");
        bool refused = false;
        std::string reason;
        {
            BudgetOverrideScope budget(1048576);
            state.setFlipMode(project::FlipMode::Hdr);
            state.updateMetrics(metrics);
            refused = !state.requestHdr() && !state.flip().has_value() && !state.hdrRunning()
                && state.flipReason().find("GPU memory") != std::string::npos;
            reason = state.flipReason();
        }
        state.runHdrBlocking();
        const bool works = state.flip().has_value() && state.flip()->hdr.has_value();
        report("GPU memory", refused && works, "reason \"" + reason + "\"");
    }

    // (i) runHdrBlocking produces a result on a linear pair.
    {
        CompareState state(context);
        addPair(state, linA, linB, "hdr-i");
        state.setFlipMode(project::FlipMode::Hdr);
        state.runHdrBlocking();
        report("runHdrBlocking", state.flip().has_value() && state.flip()->hdr.has_value(),
               "a result after runHdrBlocking");
    }
}

// The HDR composite, the exposure and FLIP map exports, and the report with the HDR fields, on a
// pair of different sizes with offsets.
void checkHdrOutputs(GpuContext & context, FlipGpu & flipGpu, Compositor & compositor, const std::string & outDir,
                     Results & results)
{
    const Case c = makeCase("hdr-sizes-topleft", {makeHdr(40, 30, 4), makeHdr(25, 45, 5)}, false);
    const LayoutInput layout = makeLayout(c);
    const PixelRect overlap = overlapRect(layout, 0, 1);
    MediaList media;
    for (size_t i = 0; i < c.images.size(); ++i)
    {
        media.push_back(std::make_unique<ImageSource>(context, c.images[i], c.name + "/" + std::to_string(i)));
        if (media.back()->status() != SourceStatus::Ready)
        {
            results.fail(c.name + ": a source is not Ready: " + media.back()->statusMessage());
            return;
        }
    }
    std::optional<FlipGpuResult> gpu;
    try
    {
        gpu = flipGpu.computeHdr(*media[0], layout.sources[0].offsetX, layout.sources[0].offsetY, *media[1],
                                 layout.sources[1].offsetX, layout.sources[1].offsetY, overlap, defaultFlipPpd(),
                                 FlipHdrSettings{});
    }
    catch (const std::exception & e)
    {
        results.fail(c.name + ": the GPU HDR-FLIP threw: " + e.what());
        return;
    }
    if (!gpu || !gpu->hdr)
    {
        results.fail(c.name + ": the GPU HDR-FLIP returned no result");
        return;
    }

    // The composite of the HDR max map: the same rules as the LDR view, 0 mismatching pixels.
    checkFlipView(c.name, c, layout, media, flipGpu, *gpu, compositor, results);

    // The FLIP map and the exposure map over the padded extent.
    const ExportScene scene = caseExportScene(c, layout);
    checkMapExport(c.name, "FLIP map", scene, gpu->errorMap, gpu->overlap, joinPath(outDir, c.name + "-flip.png"),
                   results);
    checkMapExport(c.name, "exposure map", scene, gpu->hdr->exposureMap, gpu->overlap,
                   joinPath(outDir, c.name + "-exposure.png"), results);

    // The report: mode hdr, the tone mapper, the exposures, the linear MSE and the FLIP mean.
    const std::pair<ImageData, ImageData> crops = overlapCrops(c, layout);
    const PsnrResult cpuPsnr = psnr(displayRgb(crops.first), displayRgb(crops.second));
    ReportExposures exposures;
    exposures.toneMapper = toneMapperName(gpu->hdr->toneMapper);
    exposures.start = gpu->hdr->start;
    exposures.stop = gpu->hdr->stop;
    exposures.count = gpu->hdr->count;
    ReportInput input;
    input.pathA = c.name + "-a";
    input.pathB = c.name + "-b";
    input.a = &c.images[0];
    input.b = &c.images[1];
    input.aOffsetX = layout.sources[0].offsetX;
    input.aOffsetY = layout.sources[0].offsetY;
    input.overlap = gpu->overlap;
    input.mse = cpuPsnr.mse;
    input.psnrDb = cpuPsnr.psnrDb;
    input.flipMean = gpu->mean;
    input.ppd = defaultFlipPpd();
    input.hdr = exposures;
    if (std::isfinite(gpu->hdr->linearMse))
    {
        input.linearMse = gpu->hdr->linearMse;
    }
    const CompareReport compareReport = makeReport(input);
    const std::string json = reportToJson(compareReport);
    const bool jsonOk = json.find("\"mode\": \"hdr\"") != std::string::npos
        && json.find("\"toneMapper\": \"aces\"") != std::string::npos
        && json.find("\"exposures\": {") != std::string::npos && json.find("\"linearMse\": null") == std::string::npos
        && json.find("\"linearMse\":") != std::string::npos && json.find("\"flipMean\":") != std::string::npos;

    // CSV: the header names the columns; the HDR cells are filled.
    const std::string csv = reportToCsv(compareReport);
    std::istringstream lines(csv);
    std::string headerLine;
    std::string rowLine;
    std::getline(lines, headerLine);
    std::getline(lines, rowLine);
    const auto split = [](const std::string & line)
    {
        std::vector<std::string> cells;
        std::string cell;
        std::istringstream stream(line);
        while (std::getline(stream, cell, ','))
        {
            cells.push_back(cell);
        }
        return cells;
    };
    const std::vector<std::string> header = split(headerLine);
    const std::vector<std::string> row = split(rowLine);
    const auto cellOf = [&](const std::string & name) -> std::string
    {
        const auto at = std::find(header.begin(), header.end(), name);
        if (at == header.end() || static_cast<size_t>(at - header.begin()) >= row.size())
        {
            return std::string();
        }
        return row[static_cast<size_t>(at - header.begin())];
    };
    const bool csvOk = row.size() == header.size() && !row.empty() && row[0] == "hdr" && cellOf("tone_mapper") == "aces"
        && !cellOf("start_exposure").empty() && !cellOf("stop_exposure").empty() && !cellOf("exposures").empty()
        && !cellOf("linear_mse").empty() && !cellOf("flip_mean").empty();
    std::cout << strf("[selftest] %s HDR report: json ok=%s csv ok=%s tone=%s exposures=%s linear_mse=%s\n",
                      c.name.c_str(), jsonOk ? "yes" : "no", csvOk ? "yes" : "no", cellOf("tone_mapper").c_str(),
                      cellOf("exposures").c_str(), cellOf("linear_mse").c_str());
    if (!jsonOk || !csvOk)
    {
        results.fail(c.name + ": the HDR report does not carry mode hdr, the tone mapper, exposures, linearMse and flipMean");
    }
}

// ---- Video: decoder, clip writer, NV12 converter, VideoSource and timeline (part 1) -----

// The decoder's message for a .webm it cannot decode (mf_decoder.h), verbatim.
constexpr const char * kWebmMessage
    = "WEBM needs the Windows VP9 Video Extensions (free from Microsoft Store); the MP4 path is unaffected.";
// GPU conversion against the CPU reference of the same formula, per channel.
constexpr double kVideoConvertTolerance = 1.0 / 255.0;
// BT.601 must differ from the BT.709 output by more than this on at least one test color.
constexpr double kVideo601Separation = 4.0 / 255.0;
// Seek checks: random requests, the number of backward requests that must occur, and the overrun past the end.
constexpr size_t kVideoSeekRequests = 60;
constexpr size_t kVideoSeekMinBackwards = 25;
constexpr int64_t kVideoSeekOverrun = 50;
// The frames of the .webm that are compared when an encoder exists.
constexpr uint32_t kVideoWebmFrames = 20;
// A free-memory override too small for any clip (the budget refusal check).
constexpr uint64_t kVideoTinyBudget = 1024ull * 1024ull;
// The smallest size limit VideoSource reports.
constexpr uint32_t kVideoMinSizeLimit = 4096;
// clipA is 90 frames at 30 fps: the timeline table assumes this length.
constexpr double kVideoClipSeconds = 3.0;

// One generated test clip. written is false when the encoder could not write it.
struct VideoClip
{
    std::string label; // the name in the [selftest] lines
    std::string path;  // UTF-8
    ClipSpec spec;
    bool written = false;
};

VideoClip makeVideoClip(const std::string & label, const std::string & path, uint32_t width, uint32_t height,
                        uint32_t fps, uint32_t frames, uint32_t keyframeInterval, uint32_t rotationDegrees = 0)
{
    VideoClip clip;
    clip.label = label;
    clip.path = path;
    clip.spec.path = path;
    clip.spec.width = width;
    clip.spec.height = height;
    clip.spec.fpsNumerator = fps;
    clip.spec.fpsDenominator = 1;
    clip.spec.frameCount = frames;
    clip.spec.keyframeInterval = keyframeInterval;
    clip.spec.rotationDegrees = rotationDegrees;
    return clip;
}

double fpsOf(const ClipSpec & spec)
{
    return static_cast<double>(spec.fpsNumerator) / static_cast<double>(spec.fpsDenominator);
}

// Writes the H.264 clip. False, with the reason, when this machine cannot encode it.
bool writeVideoClip(VideoClip & clip, std::string & reason)
{
    try
    {
        writeClipMp4(clip.spec);
        clip.written = true;
        return true;
    }
    catch (const std::exception & e)
    {
        reason = e.what();
        return false;
    }
}

void writeBytes(const std::string & path, const std::vector<uint8_t> & bytes)
{
    std::ofstream out(pathFromUtf8(path), std::ios::binary);
    out.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

void appendTag(std::vector<uint8_t> & bytes, const char * tag)
{
    for (size_t i = 0; i < 4; ++i)
    {
        bytes.push_back(static_cast<uint8_t>(tag[i]));
    }
}

void appendU32(std::vector<uint8_t> & bytes, uint32_t value)
{
    for (uint32_t i = 0; i < 4; ++i)
    {
        bytes.push_back(static_cast<uint8_t>(value >> (8 * i)));
    }
}

void appendU16(std::vector<uint8_t> & bytes, uint16_t value)
{
    bytes.push_back(static_cast<uint8_t>(value & 0xFF));
    bytes.push_back(static_cast<uint8_t>(value >> 8));
}

// The 44 bytes of a PCM WAV header with no samples: audio, not video.
std::vector<uint8_t> wavHeader()
{
    std::vector<uint8_t> bytes;
    appendTag(bytes, "RIFF");
    appendU32(bytes, 36);
    appendTag(bytes, "WAVE");
    appendTag(bytes, "fmt ");
    appendU32(bytes, 16);
    appendU16(bytes, 1);     // PCM
    appendU16(bytes, 1);     // mono
    appendU32(bytes, 8000);  // sample rate
    appendU32(bytes, 16000); // bytes per second
    appendU16(bytes, 2);     // block align
    appendU16(bytes, 16);    // bits per sample
    appendTag(bytes, "data");
    appendU32(bytes, 0);
    return bytes;
}

const char * statusName(SourceStatus status)
{
    switch (status)
    {
    case SourceStatus::Ready:
        return "Ready";
    case SourceStatus::Error:
        return "Error";
    case SourceStatus::Missing:
        return "Missing";
    }
    return "unknown";
}

int codeOf(double value, int low, int high)
{
    return static_cast<int>(std::clamp<long>(std::lround(value), low, high));
}

// A tight NV12 picture of width x height with a gradient in both directions: Y = 16 + x * 219 / (w - 1),
// Cb = 16 + yc * 224 / (h/2 - 1), Cr = 240 - xc * 224 / (w/2 - 1) (integer division).
std::vector<uint8_t> gradientNv12(uint32_t width, uint32_t height)
{
    std::vector<uint8_t> nv12(static_cast<size_t>(width) * height * 3 / 2, 0);
    for (uint32_t y = 0; y < height; ++y)
    {
        for (uint32_t x = 0; x < width; ++x)
        {
            nv12[static_cast<size_t>(y) * width + x] = static_cast<uint8_t>(16 + (x * 219) / (width - 1));
        }
    }
    const size_t chromaBase = static_cast<size_t>(width) * height;
    for (uint32_t yc = 0; yc < height / 2; ++yc)
    {
        for (uint32_t xc = 0; xc < width / 2; ++xc)
        {
            const size_t at = chromaBase + static_cast<size_t>(yc) * width + static_cast<size_t>(xc) * 2;
            nv12[at] = static_cast<uint8_t>(16 + (yc * 224) / (height / 2 - 1));
            nv12[at + 1] = static_cast<uint8_t>(240 - (xc * 224) / (width / 2 - 1));
        }
    }
    return nv12;
}

// CPU reference of the GPU conversion: BT.709 limited, chroma nearest, one texel per pixel, alpha 1.
std::vector<float> referenceRgba(const std::vector<uint8_t> & nv12, uint32_t width, uint32_t height)
{
    std::vector<float> rgba(static_cast<size_t>(width) * height * 4, 1.0f);
    const size_t chromaBase = static_cast<size_t>(width) * height;
    for (uint32_t y = 0; y < height; ++y)
    {
        for (uint32_t x = 0; x < width; ++x)
        {
            const size_t luma = static_cast<size_t>(y) * width + x;
            const size_t chroma = chromaBase + static_cast<size_t>(y / 2) * width + static_cast<size_t>(x / 2) * 2;
            float rgb[3] = {0.0f, 0.0f, 0.0f};
            bt709LimitedToRgb(static_cast<float>(nv12[luma]), static_cast<float>(nv12[chroma]),
                              static_cast<float>(nv12[chroma + 1]), rgb);
            rgba[luma * 4 + 0] = rgb[0];
            rgba[luma * 4 + 1] = rgb[1];
            rgba[luma * 4 + 2] = rgb[2];
        }
    }
    return rgba;
}

// Runs the GPU conversion of one tight NV12 picture and reads the RGBA32F texels back.
std::vector<float> convertOnGpu(GpuContext & context, Nv12Converter & converter, const std::vector<uint8_t> & nv12,
                                uint32_t width, uint32_t height)
{
    const size_t texels = static_cast<size_t>(width) * height;
    sabercat::Buffer nv12Buffer(context.device(), nv12.size());
    nv12Buffer.upload(nv12.data(), nv12.size());
    sabercat::Buffer rgbaBuffer(context.device(), texels * 16);
    converter.convert(nv12Buffer.view(4), rgbaBuffer.view(16), width, height);
    std::vector<float> rgba(texels * 4, -1.0f);
    rgbaBuffer.download(rgba.data(), texels * 16);
    return rgba;
}

// Largest per-channel difference of two RGB triples (alpha not compared).
double maxChannelDiff(const float * a, const float * b)
{
    double worst = 0.0;
    for (size_t c = 0; c < 3; ++c)
    {
        worst = std::max(worst, std::fabs(static_cast<double>(a[c]) - static_cast<double>(b[c])));
    }
    return worst;
}

// Largest per-channel difference over two RGBA images, and whether every alpha of gpu is exactly 1.
double maxDifference(const std::vector<float> & gpu, const std::vector<float> & ref, bool & alphaOne)
{
    double worst = 0.0;
    alphaOne = true;
    for (size_t t = 0; t + 3 < gpu.size(); t += 4)
    {
        worst = std::max(worst, maxChannelDiff(&gpu[t], &ref[t]));
        alphaOne = alphaOne && gpu[t + 3] == 1.0f;
    }
    return worst;
}

// The frame index a VideoSource's host copy shows, read from the frame's own pattern (-1 when there is no copy).
int64_t lumaIndexOf(const VideoSource & source)
{
    const ImageData * host = source.hostData();
    if (!host || host->width == 0 || host->rgba.size() < static_cast<size_t>(host->width) * host->height * 4)
    {
        return -1;
    }
    return static_cast<int64_t>(decodeClipIndexFromRgba(host->rgba.data(), host->width));
}

// Check 1: every frame of the clip decodes to its own index, at the stated size, frame rate and frame count.
void checkVideoExact(const VideoClip & clip, Results & results)
{
    std::string message;
    VideoOpenCode code = VideoOpenCode::Ok;
    const std::unique_ptr<MfDecoder> decoder = MfDecoder::open(clip.path, code, message);
    if (!decoder)
    {
        results.fail(clip.label + ": the decoder did not open it: " + message);
        return;
    }
    const VideoInfo & info = decoder->info();
    const uint32_t width = clip.spec.width;
    const uint32_t height = clip.spec.height;
    const uint32_t frames = clip.spec.frameCount;
    std::cout << strf("[selftest] video info %s size=%ux%u fps=%.6f frames=%lld duration=%.6f yuvMatrix=%u "
                      "nominalRange=%u rotation=%u hasAudio=%d\n",
                      clip.label.c_str(), info.width, info.height, info.fps, static_cast<long long>(info.frameCount),
                      info.durationSeconds, info.yuvMatrix, info.nominalRange, info.rotationDegrees,
                      info.hasAudio ? 1 : 0);
    if (info.width != width || info.height != height || std::fabs(info.fps - fpsOf(clip.spec)) > 1e-6
        || info.frameCount != static_cast<int64_t>(frames))
    {
        results.fail(strf("%s: size %ux%u fps %.6f frames %lld, expected %ux%u fps %.6f frames %u", clip.label.c_str(),
                          info.width, info.height, info.fps, static_cast<long long>(info.frameCount), width, height,
                          fpsOf(clip.spec), frames));
    }

    const size_t bytes = static_cast<size_t>(width) * height * 3 / 2;
    uint32_t mismatches = 0;
    std::string first;
    for (uint32_t k = 0; k < frames; ++k)
    {
        const Nv12Frame * frame = decoder->readFrame(k, message);
        if (!frame)
        {
            mismatches += frames - k;
            first = strf("frame %u: %s", k, message.c_str());
            break;
        }
        if (frame->bytes.size() != bytes || static_cast<int64_t>(decodeClipIndexFromLuma(frame->bytes.data(), width))
                != static_cast<int64_t>(k) || frame->index != static_cast<int64_t>(k))
        {
            ++mismatches;
            if (first.empty())
            {
                first = strf("frame %u: index %lld, %zu bytes", k, static_cast<long long>(frame->index),
                             frame->bytes.size());
            }
        }
    }
    std::cout << strf("[selftest] video exact %s frames=%u mismatches=%u\n", clip.label.c_str(), frames, mismatches);
    if (mismatches > 0)
    {
        results.fail(clip.label + ": " + std::to_string(mismatches) + " frame(s) do not decode to their index; first " + first);
    }
}

// Check 2: random and backward seeks, the first and last frame, and a request past the end (clamped).
void checkVideoSeeks(const VideoClip & clip, uint32_t seed, Results & results)
{
    const int64_t frames = static_cast<int64_t>(clip.spec.frameCount);
    std::vector<int64_t> requests;
    Lcg rng(seed);
    for (size_t i = 0; i < kVideoSeekRequests; ++i)
    {
        int64_t value = std::min<int64_t>(static_cast<int64_t>(rng.next() * static_cast<float>(frames)), frames - 1);
        if (i % 2 == 1 && !requests.empty() && requests.back() > 0 && value >= requests.back())
        {
            value = requests.back() / 2; // forced behind its predecessor
        }
        requests.push_back(value);
    }
    requests.push_back(0);
    requests.push_back(frames - 1);
    requests.push_back(frames + kVideoSeekOverrun);

    size_t backwards = 0;
    for (size_t i = 1; i < requests.size(); ++i)
    {
        if (requests[i] < requests[i - 1])
        {
            ++backwards;
        }
    }

    std::string message;
    VideoOpenCode code = VideoOpenCode::Ok;
    const std::unique_ptr<MfDecoder> decoder = MfDecoder::open(clip.path, code, message);
    if (!decoder)
    {
        results.fail(clip.label + ": the decoder did not open it for the seeks: " + message);
        return;
    }
    uint32_t mismatches = 0;
    for (const int64_t request : requests)
    {
        const int64_t expected = std::min(request, frames - 1);
        const Nv12Frame * frame = decoder->readFrame(request, message);
        if (!frame || frame->index != expected
            || static_cast<int64_t>(decodeClipIndexFromLuma(frame->bytes.data(), clip.spec.width)) != expected)
        {
            ++mismatches;
        }
    }
    std::cout << strf("[selftest] video seeks %s n=%zu backwards=%zu mismatches=%u seekCount=%u\n", clip.label.c_str(),
                      requests.size(), backwards, mismatches, decoder->seekCount());
    if (mismatches > 0 || backwards < kVideoSeekMinBackwards)
    {
        results.fail(strf("%s: seeks: %u mismatch(es), %zu backward request(s) (at least %zu needed)",
                          clip.label.c_str(), mismatches, backwards, kVideoSeekMinBackwards));
    }
}

// Check 3: the rotation tag is observed, not applied. Either value must open and decode.
void checkVideoRotation(const VideoClip & clip, Results & results)
{
    std::string message;
    VideoOpenCode code = VideoOpenCode::Ok;
    const std::unique_ptr<MfDecoder> decoder = MfDecoder::open(clip.path, code, message);
    if (!decoder)
    {
        results.fail(clip.label + ": the decoder did not open it: " + message);
        return;
    }
    const VideoInfo & info = decoder->info();
    const char * applied = "other size";
    if (info.width == 320 && info.height == 192)
    {
        applied = "rotation not applied";
    }
    else if (info.width == 192 && info.height == 320)
    {
        applied = "rotation applied";
    }
    std::cout << strf("[selftest] video rotation info=%u size=%ux%u (%s)\n", info.rotationDegrees, info.width,
                      info.height, applied);
    if (!decoder->readFrame(0, message))
    {
        results.fail(clip.label + ": frame 0 does not decode: " + message);
    }
}

// Check 4: a .webm written by the VP9 encoder when one exists, and a junk .webm, which must give the WEBM message.
void checkVideoWebm(const VideoClip & webm, const std::string & videoDir, Results & results)
{
    const bool encoder = tryWriteClipWebm(webm.spec);
    if (encoder)
    {
        std::string message;
        VideoOpenCode code = VideoOpenCode::Ok;
        const std::unique_ptr<MfDecoder> decoder = MfDecoder::open(webm.path, code, message);
        if (!decoder)
        {
            std::cout << "[selftest] video webm encoder=yes decoder=fail: " << message << '\n';
            if (message != kWebmMessage)
            {
                results.fail("video webm: the decoder message is not the WEBM message");
            }
        }
        else
        {
            std::cout << "[selftest] video webm encoder=yes decoder=ok\n";
            const uint32_t frames = std::min(kVideoWebmFrames, webm.spec.frameCount);
            uint32_t mismatches = 0;
            for (uint32_t k = 0; k < frames; ++k)
            {
                const Nv12Frame * frame = decoder->readFrame(k, message);
                if (!frame || static_cast<int64_t>(decodeClipIndexFromLuma(frame->bytes.data(), webm.spec.width))
                        != static_cast<int64_t>(k) || frame->index != static_cast<int64_t>(k))
                {
                    ++mismatches;
                }
            }
            std::cout << strf("[selftest] video webm exact frames=%u mismatches=%u\n", frames, mismatches);
            if (mismatches > 0)
            {
                results.fail(strf("video webm: %u frame(s) do not decode to their index", mismatches));
            }
        }
    }

    std::vector<uint8_t> junk(64);
    for (size_t i = 0; i < junk.size(); ++i)
    {
        junk[i] = static_cast<uint8_t>(0x5A ^ (i * 37));
    }
    const std::string junkPath = joinPath(videoDir, "bad.webm");
    writeBytes(junkPath, junk);
    std::string message;
    VideoOpenCode code = VideoOpenCode::Ok;
    const std::unique_ptr<MfDecoder> decoder = MfDecoder::open(junkPath, code, message);
    const bool junkOk = !decoder && code == VideoOpenCode::WebmNeedsVp9 && message == kWebmMessage;
    std::cout << strf("[selftest] video webm %sjunkMessage=%s code=%d\n", encoder ? "" : "encoder=no ",
                      message.c_str(), static_cast<int>(code));
    if (!junkOk)
    {
        results.fail("video webm: a junk .webm did not give WebmNeedsVp9 with the WEBM message");
    }
}

// Check 5: sources that cannot play give a non-Ready status with a message, and a valid clip still opens after them.
void checkVideoErrors(GpuContext & context, const std::string & videoDir, const std::string & validPath,
                      Results & results)
{
    std::vector<uint8_t> junk(4096);
    for (size_t i = 0; i < junk.size(); ++i)
    {
        junk[i] = static_cast<uint8_t>((i * 131 + 17) & 0xFF);
    }
    const std::string emptyPath = joinPath(videoDir, "empty.mp4");
    const std::string junkPath = joinPath(videoDir, "junk.mp4");
    const std::string audioPath = joinPath(videoDir, "audio.mp4");
    const std::string missingPath = joinPath(videoDir, "missing.mp4");
    writeBytes(emptyPath, {});
    writeBytes(junkPath, junk);
    writeBytes(audioPath, wavHeader());

    const auto probe = [&](const char * label, const std::string & path, bool expectMissing)
    {
        VideoSource source(context, path);
        const bool refused = source.status() != SourceStatus::Ready && !source.statusMessage().empty()
            && (!expectMissing || source.status() == SourceStatus::Missing);
        std::cout << strf("[selftest] video error %s status=%s message=%s\n", label, statusName(source.status()),
                          source.statusMessage().c_str());
        if (!refused)
        {
            results.fail(std::string("video error ") + label + ": not refused with a status and a message");
        }
    };
    probe("missing", missingPath, true);
    probe("empty", emptyPath, false);
    probe("junk", junkPath, false);
    probe("audio", audioPath, false);

    VideoSource valid(context, validPath);
    if (valid.status() != SourceStatus::Ready)
    {
        results.fail("video error: a valid clip after the refusals is not Ready: " + valid.statusMessage());
    }
}

// Check 6: the NV12 conversion against the CPU formula, on a synthetic gradient, on a decoded frame, and on eight
// saturated colors whose codes must match BT.709 and not BT.601.
void checkVideoConvert(GpuContext & context, const VideoClip & clipA, Results & results)
{
    Nv12Converter converter(context);

    // (a) A 64 x 32 gradient in both directions.
    {
        const std::vector<uint8_t> nv12 = gradientNv12(64, 32);
        const std::vector<float> gpu = convertOnGpu(context, converter, nv12, 64, 32);
        bool alphaOne = false;
        const double worst = maxDifference(gpu, referenceRgba(nv12, 64, 32), alphaOne);
        std::cout << strf("[selftest] video convert gradient max=%.3g alphaOne=%d\n", worst, alphaOne ? 1 : 0);
        if (!(worst <= kVideoConvertTolerance) || !alphaOne)
        {
            results.fail(strf("video convert gradient: max %.3g (limit 1/255), alpha exactly 1: %d", worst,
                              alphaOne ? 1 : 0));
        }
    }

    // (b) Frame 37 of clipA, decoded and converted: the index pattern must read back as 37.
    {
        std::string message;
        VideoOpenCode code = VideoOpenCode::Ok;
        const std::unique_ptr<MfDecoder> decoder = MfDecoder::open(clipA.path, code, message);
        const Nv12Frame * frame = decoder ? decoder->readFrame(37, message) : nullptr;
        if (!frame)
        {
            results.fail("video convert frame 37: the decode failed: " + message);
        }
        else
        {
            const uint32_t width = clipA.spec.width;
            const uint32_t height = clipA.spec.height;
            const std::vector<uint8_t> nv12 = frame->bytes;
            const std::vector<float> gpu = convertOnGpu(context, converter, nv12, width, height);
            bool alphaOne = false;
            const double worst = maxDifference(gpu, referenceRgba(nv12, width, height), alphaOne);
            const uint32_t index = decodeClipIndexFromRgba(gpu.data(), width);
            std::cout << strf("[selftest] video convert frame=37 max=%.3g index=%u\n", worst, index);
            if (!(worst <= kVideoConvertTolerance) || index != 37)
            {
                results.fail(strf("video convert frame 37: max %.3g, index %u (expected 37)", worst, index));
            }
        }
    }

    // (c) Eight saturated colors, one 2 x 2 block each (NV12 16 x 2), coded with BT.709 limited range.
    struct TestColor
    {
        const char * name;
        uint8_t r;
        uint8_t g;
        uint8_t b;
    };
    const std::array<TestColor, 8> colors = {{{"red", 255, 0, 0},
                                             {"green", 0, 255, 0},
                                             {"blue", 0, 0, 255},
                                             {"yellow", 255, 255, 0},
                                             {"cyan", 0, 255, 255},
                                             {"magenta", 255, 0, 255},
                                             {"white", 255, 255, 255},
                                             {"black", 0, 0, 0}}};
    constexpr uint32_t width = 16;
    constexpr uint32_t height = 2;
    std::vector<uint8_t> nv12(static_cast<size_t>(width) * height * 3 / 2, 0);
    std::array<std::array<float, 3>, 8> codes{};
    for (size_t i = 0; i < colors.size(); ++i)
    {
        const TestColor & color = colors[i];
        const double r = color.r / 255.0;
        const double g = color.g / 255.0;
        const double b = color.b / 255.0;
        const double yPrime = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        const int yCode = codeOf(16.0 + 219.0 * yPrime, 16, 235);
        const int cbCode = codeOf(128.0 + 224.0 * (b - yPrime) / 1.8556, 16, 240);
        const int crCode = codeOf(128.0 + 224.0 * (r - yPrime) / 1.5748, 16, 240);
        for (size_t row = 0; row < 2; ++row)
        {
            for (size_t dx = 0; dx < 2; ++dx)
            {
                nv12[row * width + 2 * i + dx] = static_cast<uint8_t>(yCode);
            }
        }
        const size_t chroma = static_cast<size_t>(width) * height + 2 * i;
        nv12[chroma] = static_cast<uint8_t>(cbCode);
        nv12[chroma + 1] = static_cast<uint8_t>(crCode);
        codes[i] = {static_cast<float>(yCode), static_cast<float>(cbCode), static_cast<float>(crCode)};
    }
    const std::vector<float> gpu = convertOnGpu(context, converter, nv12, width, height);
    int differs601 = 0;
    std::string differing;
    for (size_t i = 0; i < colors.size(); ++i)
    {
        const TestColor & color = colors[i];
        const float * texel = &gpu[(2 * i) * 4]; // pixel (2i, 0)
        float rgb709[3] = {0.0f, 0.0f, 0.0f};
        float rgb601[3] = {0.0f, 0.0f, 0.0f};
        bt709LimitedToRgb(codes[i][0], codes[i][1], codes[i][2], rgb709);
        bt601LimitedToRgb(codes[i][0], codes[i][1], codes[i][2], rgb601);
        const float original[3] = {color.r / 255.0f, color.g / 255.0f, color.b / 255.0f};
        double colorError = 0.0;
        for (size_t c = 0; c < 3; ++c)
        {
            colorError = std::max(colorError, static_cast<double>(std::fabs(texel[c] - original[c])));
        }
        const double d709 = maxChannelDiff(texel, rgb709);
        const double d601 = maxChannelDiff(texel, rgb601);
        std::cout << strf("[selftest] video convert color=%s diff709=%.3g diff601=%.3g colorError=%.3g\n", color.name,
                          d709, d601, colorError);
        if (!(d709 <= kVideoConvertTolerance))
        {
            results.fail(strf("video convert %s: differs from the BT.709 reference by %.3g", color.name, d709));
        }
        if (d601 > kVideo601Separation)
        {
            ++differs601;
            differing += std::string(" ") + color.name;
        }
    }
    std::cout << strf("[selftest] video convert BT.601 differs on %d color(s):%s\n", differs601, differing.c_str());
    if (differs601 == 0)
    {
        results.fail("video convert: BT.601 and BT.709 agree on every test color, so the matrix is not checked");
    }
}

// One row of the timeline table. trimOut < 0 is the end of the clip. The expected frame is computed by hand:
// s = time + offset, clamped to [trimIn, trimOut] (or wrapped into it with loop); frame = floor(s * 30) clamped to 89.
struct TimelineRow
{
    const char * label;
    double trimIn;
    double trimOut;
    bool loop;
    double offset;
    double time;
    int64_t frame;
};

constexpr TimelineRow kTimelineRows[] = {
    {"no trim", 0.0, -1.0, false, 0.0, 0.0, 0},
    {"no trim", 0.0, -1.0, false, 0.0, 1.0, 30},
    {"no trim", 0.0, -1.0, false, 0.0, 2.5, 75},
    {"end of the clip", 0.0, -1.0, false, 0.0, 3.0, 89},
    {"past the end clamps", 0.0, -1.0, false, 0.0, 3.5, 89},
    {"loop wraps", 0.0, -1.0, true, 0.0, 3.5, 15},
    {"loop at the end wraps to 0", 0.0, -1.0, true, 0.0, 3.0, 0},
    {"trim start only", 0.5, -1.0, false, 0.0, 0.0, 15},
    {"trim", 0.5, 2.0, false, 0.0, 1.0, 30},
    {"trim end", 0.5, 2.0, false, 0.0, 2.0, 60},
    {"trim past the end clamps", 0.5, 2.0, false, 0.0, 2.7, 60},
    {"trim loop wraps", 0.5, 2.0, true, 0.0, 2.7, 36},
    {"trim loop from before the trim", 0.5, 2.0, true, 0.0, 0.2, 51},
    {"negative offset clamps", 0.0, -1.0, false, -0.5, 0.2, 0},
    {"negative offset loops", 0.0, -1.0, true, -0.5, 0.2, 81},
    {"positive offset past the end", 0.0, -1.0, false, 2.0, 1.5, 89},
    {"positive offset loops", 0.0, -1.0, true, 2.0, 1.5, 15},
    {"one frame trim", 1.0, 1.0333, false, 0.0, 0.0, 30},
    {"one frame trim, past the end", 1.0, 1.0333, false, 0.0, 5.0, 30},
    {"seek past the end", 0.0, -1.0, false, 0.0, 10.0, 89},
    {"trim end at the duration", 1.0, 3.0, false, 0.0, 0.0, 30},
    {"trim end at the duration, time at the end", 1.0, 3.0, false, 0.0, 3.0, 89},
    {"loop at the trim end wraps", 0.5, 2.0, true, 0.0, 2.0, 15},
    {"trim start and offset", 0.5, -1.0, false, 1.0, 1.0, 60},
    {"exactly one frame", 0.0, -1.0, false, 0.0, 1.0 / 30.0, 1},
    {"just before a frame", 0.0, -1.0, false, 0.0, 1.0 / 30.0 - 0.001, 0},
    {"large negative time loops", 0.0, -1.0, true, 0.0, -3.5, 75},
};

// Check 8: the timeline table on clipA, with the trim, loop and offset settings, and the refusals.
void checkVideoTimeline(GpuContext & context, const VideoClip & clipA, Results & results)
{
    VideoSource source(context, clipA.path);
    if (source.status() != SourceStatus::Ready || std::fabs(source.durationSeconds() - kVideoClipSeconds) > 1e-3
        || source.frameCount() != static_cast<int64_t>(clipA.spec.frameCount))
    {
        results.fail("video timeline: clipA is not Ready with 3 s and 90 frames: " + source.statusMessage());
        return;
    }
    uint32_t mismatches = 0;
    for (const TimelineRow & row : kTimelineRows)
    {
        const bool trimOk = source.setTrim(row.trimIn, row.trimOut);
        source.setLoop(row.loop);
        const bool offsetOk = source.setTimeOffset(row.offset);
        if (!trimOk || !offsetOk)
        {
            ++mismatches;
            std::cout << strf("[selftest]   video timeline %s: setting refused (trim %d, offset %d)\n", row.label,
                              trimOk ? 1 : 0, offsetOk ? 1 : 0);
            continue;
        }
        source.update(row.time);
        const int64_t shown = source.frameIndex();
        const int64_t decoded = lumaIndexOf(source);
        const int64_t timed = frameIndexAt(source.timeSource(), row.time);
        if (shown != row.frame || decoded != row.frame || timed != row.frame)
        {
            ++mismatches;
            std::cout << strf("[selftest]   video timeline %s t=%.6f expected=%lld shown=%lld decoded=%lld timed=%lld\n",
                              row.label, row.time, static_cast<long long>(row.frame), static_cast<long long>(shown),
                              static_cast<long long>(decoded), static_cast<long long>(timed));
        }
    }
    std::cout << strf("[selftest] video timeline cases=%zu mismatches=%u\n", std::size(kTimelineRows), mismatches);
    if (mismatches > 0)
    {
        results.fail(strf("video timeline: %u case(s) show the wrong frame", mismatches));
    }

    // Refusals leave every setting as it was.
    const double trimInBefore = source.trimIn();
    const double trimOutBefore = source.trimOut();
    const double offsetBefore = source.timeOffset();
    const bool loopBefore = source.loop();
    const bool toEndBefore = source.trimToEnd();
    const bool refusedA = !source.setTrim(2.0, 1.0);
    const bool refusedB = !source.setTrim(-1.0, 1.0);
    const bool refusedC = !source.setTrim(0.0, 10.0);
    const bool refusedD = !source.setTimeOffset(std::nan(""));
    const bool unchanged = source.trimIn() == trimInBefore && source.trimOut() == trimOutBefore
        && source.timeOffset() == offsetBefore && source.loop() == loopBefore && source.trimToEnd() == toEndBefore;
    const bool refused = refusedA && refusedB && refusedC && refusedD;
    std::cout << strf("[selftest] video timeline refusals=%d unchanged=%d\n", refused ? 1 : 0, unchanged ? 1 : 0);
    if (!refused || !unchanged)
    {
        results.fail("video timeline: a refused setting was applied or changed the other settings");
    }
}

// Check 9: two clips of different frame rates (clipA 30 fps, clipB 24 fps) at the same global times.
void checkVideoFpsMap(GpuContext & context, const VideoClip & clipA, const VideoClip & clipB, Results & results)
{
    // Frame = min(floor(t * fps + 1e-6), last frame): A 30 fps (last 89), B 24 fps (last 71).
    struct FpsRow
    {
        double time;
        int64_t a;
        int64_t b;
    };
    constexpr FpsRow rows[] = {
        {0.0, 0, 0},   {1.0 / 30.0, 1, 0}, {1.0 / 24.0, 1, 1}, {0.5, 15, 12},  {0.99, 29, 23}, {1.0, 30, 24},
        {1.4, 42, 33}, {2.0, 60, 48},      {2.5, 75, 60},      {2.99, 89, 71}, {3.0, 89, 71},  {4.0, 89, 71},
    };
    VideoSource a(context, clipA.path);
    VideoSource b(context, clipB.path);
    if (a.status() != SourceStatus::Ready || b.status() != SourceStatus::Ready)
    {
        results.fail("video fps-map: a clip is not Ready: " + a.statusMessage() + " " + b.statusMessage());
        return;
    }
    uint32_t mismatches = 0;
    for (const FpsRow & row : rows)
    {
        a.update(row.time);
        b.update(row.time);
        const int64_t shownA = a.frameIndex();
        const int64_t shownB = b.frameIndex();
        const int64_t decodedA = lumaIndexOf(a);
        const int64_t decodedB = lumaIndexOf(b);
        const int64_t timedA = frameIndexAt(a.timeSource(), row.time);
        const int64_t timedB = frameIndexAt(b.timeSource(), row.time);
        if (shownA != row.a || decodedA != row.a || timedA != row.a || shownB != row.b || decodedB != row.b
            || timedB != row.b)
        {
            ++mismatches;
            std::cout << strf("[selftest]   video fps-map t=%.6f A expected=%lld shown=%lld decoded=%lld; B expected=%lld "
                              "shown=%lld decoded=%lld\n",
                              row.time, static_cast<long long>(row.a), static_cast<long long>(shownA),
                              static_cast<long long>(decodedA), static_cast<long long>(row.b),
                              static_cast<long long>(shownB), static_cast<long long>(decodedB));
        }
    }
    std::cout << strf("[selftest] video fps-map times=%zu mismatches=%u\n", std::size(rows), mismatches);
    if (mismatches > 0)
    {
        results.fail(strf("video fps-map: %u time(s) show the wrong frame", mismatches));
    }
}

// Check 10: a clip whose path is not ASCII opens, and frame 0 decodes.
void checkVideoUtf8(GpuContext & context, const std::string & path, Results & results)
{
    VideoSource source(context, path);
    const int64_t decoded = lumaIndexOf(source);
    const bool ok = source.status() == SourceStatus::Ready && source.frameIndex() == 0 && decoded == 0;
    std::cout << strf("[selftest] video utf8 status=%s index=%lld decoded=%lld\n", statusName(source.status()),
                      static_cast<long long>(source.frameIndex()), static_cast<long long>(decoded));
    if (!ok)
    {
        results.fail("video utf8 path: " + source.statusMessage());
    }
}

// Check 11: under a free-memory override too small for the clip, the source refuses with the budget message;
// without the override it opens again. Prints the size limit in use.
void checkVideoBudget(GpuContext & context, const VideoClip & clipA, Results & results)
{
    std::string message;
    bool refused = false;
    {
        BudgetOverrideScope scope(kVideoTinyBudget);
        VideoSource source(context, clipA.path);
        refused = source.status() == SourceStatus::Error;
        message = source.statusMessage();
    }
    const bool messageOk = message.rfind("Not enough GPU memory for this video", 0) == 0;
    VideoSource again(context, clipA.path);
    const bool reopened = again.status() == SourceStatus::Ready;
    const uint32_t maxDimension = context.device().properties().limits.maxImageDimension2D;
    const uint32_t limit = std::max(kVideoMinSizeLimit, maxDimension);
    std::cout << strf("[selftest] video budget refused=%d message=%s\n", refused ? 1 : 0, message.c_str());
    std::cout << strf("[selftest] video size limit L=%u maxImageDimension2D=%u reopened=%d\n", limit, maxDimension,
                      reopened ? 1 : 0);
    if (!refused || !messageOk || !reopened)
    {
        results.fail("video budget: the memory refusal or the reopen after it is wrong: " + message);
    }
}

// Mean decode, upload and convert times of a sequential pass over every frame of the clip (the updates that decoded).
bool sequentialTimings(GpuContext & context, const VideoClip & clip, VideoSource::FrameTiming & mean)
{
    VideoSource source(context, clip.path);
    if (source.status() != SourceStatus::Ready)
    {
        return false;
    }
    const double fps = fpsOf(clip.spec);
    VideoSource::FrameTiming sum;
    uint32_t count = 0;
    for (uint32_t k = 0; k < clip.spec.frameCount; ++k)
    {
        if (source.update(static_cast<double>(k) / fps))
        {
            const VideoSource::FrameTiming timing = source.lastTiming();
            sum.decodeSeconds += timing.decodeSeconds;
            sum.uploadSeconds += timing.uploadSeconds;
            sum.convertSeconds += timing.convertSeconds;
            ++count;
        }
    }
    if (count > 0)
    {
        mean.decodeSeconds = sum.decodeSeconds / count;
        mean.uploadSeconds = sum.uploadSeconds / count;
        mean.convertSeconds = sum.convertSeconds / count;
    }
    return true;
}

// Mean Compositor::render seconds of a two-pane composite (clipA as A, clipB as B), one step per frame of clipA.
bool compositeSeconds(GpuContext & context, Compositor & compositor, const VideoClip & clipA, const VideoClip & clipB,
                      double & mean)
{
    CompareState state(context);
    if (state.addMedia(clipA.path) != 0 || state.addMedia(clipB.path) != 1)
    {
        return false;
    }
    state.setViewSize(static_cast<int32_t>(kViewWidth), static_cast<int32_t>(kViewHeight));
    compositor.resize(kViewWidth, kViewHeight);
    const double fps = fpsOf(clipA.spec);
    double total = 0.0;
    uint32_t count = 0;
    for (uint32_t k = 0; k < clipA.spec.frameCount; ++k)
    {
        state.seek(static_cast<double>(k) / fps);
        state.updateVideos();
        const CompositeInput input = state.buildComposite(state.transport().time);
        const auto start = std::chrono::steady_clock::now();
        compositor.render(input);
        total += std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        ++count;
    }
    mean = count > 0 ? total / count : 0.0;
    return true;
}

void printVideoPerf(const VideoClip & clip, const VideoSource::FrameTiming & timing, const std::string & composite)
{
    std::cout << strf("[selftest] video perf %s %ux%u decode=%.3f upload=%.3f convert=%.3f composite=%s\n",
                      clip.label.c_str(), clip.spec.width, clip.spec.height, timing.decodeSeconds * 1000.0,
                      timing.uploadSeconds * 1000.0, timing.convertSeconds * 1000.0, composite.c_str());
}

// Check 12: measurement only (no threshold). Mean decode, upload and convert per frame for clipA and clip1080,
// and the render time of a two-pane composite of clipA and clipB.
void checkVideoPerf(GpuContext & context, Compositor & compositor, const VideoClip & clipA, const VideoClip & clipB,
                    const VideoClip & clip1080, Results & results)
{
    VideoSource::FrameTiming timingA;
    if (!sequentialTimings(context, clipA, timingA))
    {
        results.fail("video perf: clipA is not Ready");
        return;
    }
    double compositeA = 0.0;
    if (!compositeSeconds(context, compositor, clipA, clipB, compositeA))
    {
        results.fail("video perf: the clipA and clipB pair was not added to the state");
        return;
    }
    printVideoPerf(clipA, timingA, strf("%.3f", compositeA * 1000.0));

    VideoSource::FrameTiming timing1080;
    if (!sequentialTimings(context, clip1080, timing1080))
    {
        results.fail("video perf: clip1080 is not Ready");
        return;
    }
    printVideoPerf(clip1080, timing1080, "n/a");
}

// ---- Part 2 (phase P6): the video state checks, items 13-18 -----------------------------

// Counts the checks of one group. A failed check is reported through results.
struct Tally
{
    int checks = 0;
    int failures = 0;

    void check(bool ok, const std::string & what, Results & results)
    {
        ++checks;
        if (!ok)
        {
            ++failures;
            results.fail(what);
        }
    }
};

// The per-channel byte differences of RGBA GPU bytes against RGB reference bytes: the largest, the pixels
// that differ by more than kMaxByteDelta, and the pixels whose alpha is not 255.
struct ByteDiff
{
    int maxDelta = 0;
    size_t over = 0;
    size_t alphaBad = 0;
};

ByteDiff diffRgbaRgb(const std::vector<uint8_t> & gpu, const std::vector<uint8_t> & ref, size_t pixels)
{
    ByteDiff diff;
    for (size_t p = 0; p < pixels; ++p)
    {
        int delta = 0;
        for (size_t c = 0; c < 3; ++c)
        {
            delta = std::max(delta, std::abs(static_cast<int>(gpu[4 * p + c]) - static_cast<int>(ref[3 * p + c])));
        }
        diff.maxDelta = std::max(diff.maxDelta, delta);
        diff.over += delta > kMaxByteDelta ? 1 : 0;
        diff.alphaBad += gpu[4 * p + 3] != 255 ? 1 : 0;
    }
    return diff;
}

// The RGB bytes of RGBA bytes (alpha dropped).
std::vector<uint8_t> rgbOfRgba(const std::vector<uint8_t> & rgba)
{
    const size_t pixels = rgba.size() / 4;
    std::vector<uint8_t> rgb(pixels * 3);
    for (size_t p = 0; p < pixels; ++p)
    {
        rgb[3 * p] = rgba[4 * p];
        rgb[3 * p + 1] = rgba[4 * p + 1];
        rgb[3 * p + 2] = rgba[4 * p + 2];
    }
    return rgb;
}

// The 8-bit RGB image of a Display host copy, rounded as the PNG path rounds it.
Rgb8Image rgb8Of(const ImageData & image)
{
    Rgb8Image out;
    out.width = image.width;
    out.height = image.height;
    const size_t pixels = static_cast<size_t>(image.width) * image.height;
    out.rgb.resize(pixels * 3);
    for (size_t p = 0; p < pixels; ++p)
    {
        for (size_t c = 0; c < 3; ++c)
        {
            out.rgb[3 * p + c] = byteOf(image.rgba[4 * p + c]);
        }
    }
    return out;
}

// The numbers that follow each "key": in JSON text, in order (NaN for null).
std::vector<double> numbersAfter(const std::string & text, const std::string & key)
{
    std::vector<double> values;
    const std::string needle = "\"" + key + "\"";
    for (size_t at = text.find(needle); at != std::string::npos; at = text.find(needle, at + needle.size()))
    {
        size_t p = at + needle.size();
        while (p < text.size() && (text[p] == ':' || text[p] == ' '))
        {
            ++p;
        }
        values.push_back(text.compare(p, 4, "null") == 0 ? std::nan("") : std::strtod(text.c_str() + p, nullptr));
    }
    return values;
}

// The text of a file (empty when it cannot be read).
std::string readText(const std::string & path)
{
    std::ifstream in(pathFromUtf8(path), std::ios::binary);
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

// The decoded frame index of a video entry (its pattern), -1 when the entry is not a Ready video.
int64_t decodedIndex(const CompareState & state, size_t index)
{
    const VideoSource * video = state.videoOf(index);
    return video != nullptr ? lumaIndexOf(*video) : -1;
}

// Adds first as entry 0 and second as entry 1 (both Ready). False, after a reported failure, otherwise.
bool addClipPair(CompareState & state, const VideoClip & first, const VideoClip & second, const char * group,
                 Results & results)
{
    if (state.addMedia(first.path) != 0 || state.addMedia(second.path) != 1 || !state.entries()[0].ready()
        || !state.entries()[1].ready())
    {
        results.fail(std::string(group) + ": the two sources were not added as Ready sources");
        return false;
    }
    return true;
}

// Item 13: one row of the composite table. trimOut < 0 is the end of the clip. Every row sets all the
// settings of both videos, then seeks and updates.
struct CompositeRow
{
    const char * label;
    double inA;
    double outA;
    double offsetA;
    bool loopA;
    double inB;
    double outB;
    double offsetB;
    bool loopB;
    double time;
};

constexpr CompositeRow kCompositeRows[] = {
    {"start", 0.0, -1.0, 0.0, false, 0.0, -1.0, 0.0, false, 0.0},
    {"mid", 0.0, -1.0, 0.0, false, 0.0, -1.0, 0.0, false, 1.5},
    {"end", 0.0, -1.0, 0.0, false, 0.0, -1.0, 0.0, false, 3.0},
    {"past the end", 0.0, -1.0, 0.0, false, 0.0, -1.0, 0.0, false, 3.5},
    {"loop wraps", 0.0, -1.0, 0.0, true, 0.0, -1.0, 0.0, true, 3.5},
    {"trim A", 0.5, 2.0, 0.0, false, 0.0, -1.0, 0.0, false, 1.0},
    {"trim B", 0.0, -1.0, 0.0, false, 0.25, 2.5, 0.0, false, 2.0},
    {"trim both, loop", 0.5, 2.0, 0.0, true, 0.25, 2.5, 0.0, true, 2.7},
    {"A offset +0.5", 0.0, -1.0, 0.5, false, 0.0, -1.0, 0.0, false, 2.2},
    {"B offset -0.5", 0.0, -1.0, 0.0, false, 0.0, -1.0, -0.5, false, 1.0},
    {"A offset -0.25, loop", 0.0, -1.0, -0.25, true, 0.0, -1.0, 0.0, false, 0.3},
    {"trim and offset", 1.0, 2.5, 0.5, false, 0.0, -1.0, 0.0, false, 1.9},
    {"B trim, loop", 0.0, -1.0, 0.0, false, 0.0, 1.0, 0.0, true, 0.75},
    {"A one-frame trim", 1.0, 1.0333, 0.0, false, 0.0, -1.0, 0.0, false, 0.0},
    {"B from 2 s", 0.0, -1.0, 0.0, false, 2.0, -1.0, 0.0, false, 2.5},
    {"A trim to the duration", 0.0, 3.0, 0.0, false, 0.0, -1.0, 0.0, false, 2.99},
    {"both offsets, loop", 0.0, -1.0, 0.1, true, 0.0, -1.0, 0.2, true, 1.234},
    {"time 1/30", 0.0, -1.0, 0.0, false, 0.0, -1.0, 0.0, false, 1.0 / 30.0},
    {"frame boundary at 2 s", 0.0, -1.0, 0.0, false, 0.0, -1.0, 0.0, false, 2.0},
    {"negative time clamps", 0.0, -1.0, 0.0, false, 0.0, -1.0, 0.0, false, -1.0},
    {"A from 0.5 s, B loop", 0.5, -1.0, 0.0, false, 0.0, -1.0, 0.0, true, 0.9},
    {"B loop, trim end", 0.0, -1.0, 0.0, false, 0.5, 2.0, 0.0, true, 2.0},
};

// Item 13: the composite of clipA (A, 320x192) and clipB (B, 256x144) at 100% and the CPU reference, over
// the table. Every byte must be within kMaxByteDelta and every alpha 255.
void checkVideoComposites(GpuContext & context, Compositor & compositor, const VideoClip & clipA,
                          const VideoClip & clipB, Results & results)
{
    CompareState state(context);
    if (!addClipPair(state, clipA, clipB, "video composite", results))
    {
        return;
    }
    state.setViewSize(static_cast<int32_t>(kViewWidth), static_cast<int32_t>(kViewHeight));
    setZoomPercent(state.viewLayout(), 100.0);
    centerView(state.viewLayout());
    compositor.resize(kViewWidth, kViewHeight);

    const size_t pixels = static_cast<size_t>(kViewWidth) * kViewHeight;
    size_t cases = 0;
    int maxDelta = 0;
    for (const CompositeRow & row : kCompositeRows)
    {
        const bool trimA = state.setVideoTrim(0, row.inA, row.outA);
        const bool trimB = state.setVideoTrim(1, row.inB, row.outB);
        const bool loopA = state.setVideoLoop(0, row.loopA);
        const bool loopB = state.setVideoLoop(1, row.loopB);
        const bool offsetA = state.setVideoTimeOffset(0, row.offsetA);
        const bool offsetB = state.setVideoTimeOffset(1, row.offsetB);
        if (!(trimA && trimB && loopA && loopB && offsetA && offsetB))
        {
            results.fail(strf("video composite %s: a setting was refused", row.label));
            continue;
        }
        state.seek(row.time);
        state.updateVideos();

        const ImagePtrs images = hostCopies(state);
        const CompositeInput input = state.buildComposite(0.0);
        std::vector<uint8_t> gpu;
        try
        {
            compositor.render(input);
            compositor.readback(gpu);
        }
        catch (const std::exception & e)
        {
            results.fail(strf("video composite %s: render threw: %s", row.label, e.what()));
            continue;
        }
        if (gpu.size() != pixels * 4)
        {
            results.fail(strf("video composite %s: readback of %zu bytes", row.label, gpu.size()));
            continue;
        }
        const std::vector<uint8_t> ref = referenceComposite(input, images, kViewWidth, kViewHeight);
        const ByteDiff diff = diffRgbaRgb(gpu, ref, pixels);
        ++cases;
        maxDelta = std::max(maxDelta, diff.maxDelta);
        if (diff.over != 0 || diff.alphaBad != 0)
        {
            results.fail(strf("video composite %s: %zu pixels differ by more than %d/255, %zu alpha != 255", row.label,
                              diff.over, kMaxByteDelta, diff.alphaBad));
        }
    }
    std::cout << strf("[selftest] video composite cases=%zu maxByteDelta=%d\n", cases, maxDelta);
}

// Item 14: six frame pairs of clipA and clipB: the GPU metric against the CPU psnr of the host crops, and the
// LDR FLIP mean against the CPU flip of the same crops.
void checkVideoMetrics(GpuContext & context, MetricsGpu & metrics, const VideoClip & clipA, const VideoClip & clipB,
                       Results & results)
{
    CompareState state(context);
    if (!addClipPair(state, clipA, clipB, "video metrics", results))
    {
        return;
    }
    constexpr double kTimes[] = {0.0, 0.5, 1.0, 1.5, 2.0, 2.9};
    for (const double t : kTimes)
    {
        state.seek(t);
        state.updateVideos();
        state.updateMetrics(metrics);
        const std::string label = strf("video metrics t=%.3f", t);
        const LayoutInput & layout = state.layout();
        const PixelRect overlap = overlapRect(layout, 0, 1);
        const ImageData * hostA = state.entries()[0].media->hostData();
        const ImageData * hostB = state.entries()[1].media->hostData();
        if (!state.metrics() || hostA == nullptr || hostB == nullptr)
        {
            results.fail(label + ": no metrics or no host copy: " + state.metricsReason());
            continue;
        }
        const ImageData cropA = cropImage(*hostA, overlap.x - layout.sources[0].offsetX,
                                          overlap.y - layout.sources[0].offsetY, overlap.width, overlap.height);
        const ImageData cropB = cropImage(*hostB, overlap.x - layout.sources[1].offsetX,
                                          overlap.y - layout.sources[1].offsetY, overlap.width, overlap.height);
        const PsnrResult cpu = psnr(displayRgb(cropA), displayRgb(cropB));
        checkMetricNumbers(label, *state.metrics(), cpu, results);

        // LDR FLIP (the default mode): shown, no reason, mean finite and in [0, 1], equal to the CPU flip.
        const bool shown = state.flip().has_value() && state.flipReason().empty();
        const double mean = shown ? state.flip()->mean : std::nan("");
        PixelRect flipOverlap;
        const FlipResult cpuFlip = cpuFlipOfState(state, state.flipPpd(), flipOverlap);
        const double delta = std::fabs(mean - cpuFlip.meanError);
        std::cout << strf("[selftest] %s flip gpu=%.9g cpu=%.9g dmean=%.3g\n", label.c_str(), mean, cpuFlip.meanError,
                          delta);
        if (!shown || !std::isfinite(mean) || mean < 0.0 || mean > 1.0 || !(delta <= kMaxFlipMeanDelta))
        {
            results.fail(label + strf(": FLIP is not shown, or its mean %.9g is outside [0, 1] or differs from the CPU "
                                      "%.9g by %.3g",
                                      mean, cpuFlip.meanError, delta));
        }
    }
}

// Item 15: the transport of clipA and clipB, the short clip beside clipA, the hide and show of the only
// visible video, and the removal of a video. Prints the check count and the failure count.
void checkVideoTransport(GpuContext & context, const VideoClip & clipA, const VideoClip & clipB,
                         const VideoClip & shortClip, Results & results)
{
    Tally tally;
    constexpr double kClockEps = 1e-9;
    constexpr double kDurationEps = 1e-6;

    CompareState state(context);
    if (!addClipPair(state, clipA, clipB, "video transport", results))
    {
        return;
    }
    tally.check(std::fabs(state.transport().duration - kVideoClipSeconds) <= 1e-3,
                strf("video transport: duration %.9g, expected 3 s", state.transport().duration), results);

    // Speed: the presets and the range, and a refusal leaves the speed as it was.
    tally.check(!state.setSpeed(0.0), "video transport: speed 0 is accepted", results);
    tally.check(!state.setSpeed(5.0), "video transport: speed 5 is accepted", results);
    tally.check(!state.setSpeed(std::nan("")), "video transport: a NaN speed is accepted", results);
    tally.check(state.setSpeed(0.25), "video transport: speed 0.25 is refused", results);
    tally.check(state.setSpeed(4.0), "video transport: speed 4 is refused", results);
    tally.check(state.setSpeed(1.0), "video transport: speed 1 is refused", results);

    // Playing: 0.1 s of clock at speed 1 and 0.2 s at speed 2; one 5 s step moves the clock at most 0.25 s.
    state.seek(0.0);
    state.play();
    state.advanceTime(0.1);
    const double speed1 = state.transport().time;
    tally.check(state.transport().playing && std::fabs(speed1 - 0.1) <= kClockEps,
                strf("video transport: speed 1, 0.1 s moved the clock to %.9g", speed1), results);
    state.pause();
    state.seek(0.5);
    state.setSpeed(2.0);
    state.play();
    state.advanceTime(0.1);
    const double speed2 = state.transport().time;
    tally.check(std::fabs(speed2 - 0.7) <= kClockEps,
                strf("video transport: speed 2, 0.1 s from 0.5 moved the clock to %.9g, not 0.7", speed2), results);
    state.pause();
    state.setSpeed(1.0);
    state.seek(0.5);
    state.play();
    state.advanceTime(5.0);
    const double clamped = state.transport().time;
    tally.check(std::fabs(clamped - 0.75) <= kClockEps,
                strf("video transport: advanceTime(5.0) from 0.5 moved the clock to %.9g, not 0.75", clamped), results);
    state.pause();

    // A stall: no calls for a while, then one long step. The clock moves at most 0.25 s.
    state.seek(0.5);
    state.play();
    state.advanceTime(10.0);
    const double stalled = state.transport().time - 0.5;
    std::cout << strf("[selftest] video transport stall: advanceTime(10.0) moved the clock by %.9g\n", stalled);
    tally.check(stalled > 0.0 && stalled <= 0.25 + kClockEps, strf("video transport: a stall moved the clock by %.9g",
                                                                     stalled), results);
    state.pause();

    // Stepping: one frame forward and back of clipA (the reference video), on the frame boundaries, paused.
    state.seek(1.0);
    state.updateVideos();
    state.play();
    state.stepFrames(1);
    tally.check(!state.transport().playing, "video transport: stepFrames did not pause", results);
    state.updateVideos();
    tally.check(std::fabs(state.transport().time - 31.0 / 30.0) <= kClockEps && state.videoFrameIndex(0) == 31
                    && decodedIndex(state, 0) == 31,
                strf("video transport: stepFrames(+1) from 1.0 s gave %.12g s, frame %lld", state.transport().time,
                     static_cast<long long>(state.videoFrameIndex(0))),
                results);
    state.stepFrames(-1);
    state.updateVideos();
    tally.check(std::fabs(state.transport().time - 1.0) <= kClockEps && state.videoFrameIndex(0) == 30
                    && decodedIndex(state, 0) == 30,
                strf("video transport: stepFrames(-1) gave %.12g s, frame %lld", state.transport().time,
                     static_cast<long long>(state.videoFrameIndex(0))),
                results);
    state.seek(0.0);
    state.stepFrames(-1);
    tally.check(state.transport().time == 0.0, strf("video transport: stepFrames(-1) at 0 moved to %.9g",
                                                    state.transport().time), results);

    // Seeks: a NaN is refused; a seek past the end clamps to the duration and shows the last frames.
    tally.check(!state.seek(std::nan("")), "video transport: a NaN seek was accepted", results);
    state.seek(1e9);
    state.updateVideos();
    tally.check(state.transport().time == state.transport().duration && state.videoFrameIndex(0) == 89
                    && decodedIndex(state, 0) == 89 && state.videoFrameIndex(1) == 71 && decodedIndex(state, 1) == 71,
                strf("video transport: seek(1e9) gave %.9g of %.9g, frames %lld and %lld", state.transport().time,
                     state.transport().duration, static_cast<long long>(state.videoFrameIndex(0)),
                     static_cast<long long>(state.videoFrameIndex(1))),
                results);

    // Play at the end without loop: it restarts at 0 (the behavior of play()).
    state.setTransportLoop(false);
    state.pause();
    state.play();
    const bool restarts = state.transport().time == 0.0 && state.transport().playing;
    std::cout << strf("[selftest] video transport: play at the end without loop restarts at 0: %s\n",
                      restarts ? "yes" : "no");
    tally.check(restarts, "video transport: play at the end without loop does not restart at 0", results);
    state.pause();

    // Loop: the clock wraps instead of stopping. Without loop it stops at the duration and is not playing.
    state.setTransportLoop(true);
    state.seek(state.transport().duration - 0.05);
    state.play();
    state.advanceTime(0.1);
    const double wrapped = state.transport().time;
    tally.check(state.transport().playing && std::fabs(wrapped - 0.05) <= kDurationEps,
                strf("video transport: the loop moved the clock to %.9g, not 0.05", wrapped), results);
    state.pause();
    state.setTransportLoop(false);
    state.seek(state.transport().duration - 0.05);
    state.play();
    state.advanceTime(0.1);
    tally.check(state.transport().time == state.transport().duration && !state.transport().playing,
                strf("video transport: without loop the clock is at %.9g, playing %d", state.transport().time,
                     state.transport().playing ? 1 : 0),
                results);
    state.pause();

    // A short clip beside clipA: it holds its last frame (14) at 2.0 s, and with its loop it shows frame 0.
    if (shortClip.written)
    {
        CompareState pair(context);
        if (addClipPair(pair, clipA, shortClip, "video transport short", results))
        {
            pair.seek(2.0);
            pair.updateVideos();
            const int64_t held = pair.videoFrameIndex(1);
            tally.check(held == 14 && decodedIndex(pair, 1) == 14 && pair.videoFrameIndex(0) == 60,
                        strf("video transport: the short clip at 2.0 s shows frame %lld, not 14",
                             static_cast<long long>(held)),
                        results);
            pair.setVideoLoop(1, true);
            pair.updateVideos();
            const int64_t looped = pair.videoFrameIndex(1);
            tally.check(looped == 0 && decodedIndex(pair, 1) == 0,
                        strf("video transport: the looped short clip at 2.0 s shows frame %lld, not 0",
                             static_cast<long long>(looped)),
                        results);
        }
    }
    else
    {
        std::cout << "[selftest] video transport short clip SKIPPED: short.mp4 was not written\n";
    }

    // Hide and show the only visible video: hiding pauses and sets the duration to 0; showing it again catches
    // up to the clock.
    {
        CompareState only(context);
        if (only.addMedia(clipA.path) == 0 && only.entries()[0].ready())
        {
            only.seek(1.0);
            only.updateVideos();
            tally.check(only.setVisible(0, false), "video transport: hiding clipA was refused", results);
            tally.check(only.transport().duration == 0.0 && !only.transport().playing,
                        strf("video transport: hidden, the duration is %.9g and playing %d", only.transport().duration,
                             only.transport().playing ? 1 : 0),
                        results);
            only.play();
            tally.check(!only.transport().playing, "video transport: play() started with no visible video", results);
            tally.check(only.setVisible(0, true), "video transport: showing clipA was refused", results);
            tally.check(std::fabs(only.transport().duration - kVideoClipSeconds) <= 1e-3,
                        strf("video transport: shown again, the duration is %.9g", only.transport().duration), results);
            only.updateVideos();
            const VideoSource * shown = only.videoOf(0);
            const int64_t expected = shown != nullptr ? frameIndexAt(shown->timeSource(), only.transport().time) : -1;
            tally.check(expected >= 0 && only.videoFrameIndex(0) == expected && decodedIndex(only, 0) == expected,
                        strf("video transport: shown again, frame %lld does not match the clock %.9g (frame %lld)",
                             static_cast<long long>(only.videoFrameIndex(0)), only.transport().time,
                             static_cast<long long>(expected)),
                        results);
            only.seek(2.0);
            only.updateVideos();
            tally.check(only.videoFrameIndex(0) == 60 && decodedIndex(only, 0) == 60,
                        strf("video transport: seek(2.0) after showing again shows frame %lld, not 60",
                             static_cast<long long>(only.videoFrameIndex(0))),
                        results);
        }
        else
        {
            results.fail("video transport: clipA was not added as a Ready source for the hide and show check");
        }
    }

    // Remove clipB: clipA and the transport keep working.
    tally.check(state.removeSource(1), "video transport: removing clipB was refused", results);
    state.seek(1.5);
    state.updateVideos();
    state.play();
    state.advanceTime(0.1);
    tally.check(state.size() == 1 && state.sourceB() < 0 && std::fabs(state.transport().duration - kVideoClipSeconds) <= 1e-3
                    && state.videoFrameIndex(0) == 45 && decodedIndex(state, 0) == 45
                    && std::fabs(state.transport().time - 1.6) <= kClockEps,
                strf("video transport: after removing clipB, %zu entries, clock %.9g, frame %lld", state.size(),
                     state.transport().time, static_cast<long long>(state.videoFrameIndex(0))),
                results);
    state.pause();

    std::cout << strf("[selftest] video transport checks=%d failures=%d\n", tally.checks, tally.failures);
}

// Item 16: a project with clipA trimmed, looped and offset, at speed 2 and mid-playback; a project whose video
// was removed; and the smoke project of the pair.
void checkVideoProject(GpuContext & context, const VideoClip & clipA, const VideoClip & clipB,
                       const std::string & videoDir, Results & results)
{
    Tally tally;
    constexpr double kEps = 1e-9;

    // The mid-playback project.
    const std::string midPath = joinPath(videoDir, "mid.sabercat-compare");
    {
        CompareState state(context);
        if (!addClipPair(state, clipA, clipB, "video project", results))
        {
            return;
        }
        tally.check(state.setVideoTrim(0, 0.5, 2.0), "video project: the trim of clipA was refused", results);
        tally.check(state.setVideoLoop(0, true), "video project: the loop of clipA was refused", results);
        tally.check(state.setVideoTimeOffset(0, -0.1), "video project: the offset of clipA was refused", results);
        tally.check(state.setSpeed(2.0), "video project: speed 2 was refused", results);
        state.setTransportLoop(true);
        state.seek(0.5);
        state.play();
        state.advanceTime(0.1);
        state.seek(1.5);
        state.updateVideos();
        std::string error;
        const bool saved = state.saveProject(midPath, error);
        tally.check(saved, "video project: saving mid.sabercat-compare failed: " + error, results);
    }
    CompareState loaded(context);
    std::string error;
    const bool opened = loaded.openProject(midPath, error);
    tally.check(opened, "video project: opening mid.sabercat-compare failed: " + error, results);
    const Transport & transport = loaded.transport();
    const VideoSource * video = loaded.videoOf(0);
    const bool videoOk = video != nullptr && std::fabs(video->trimIn() - 0.5) <= kEps
        && std::fabs(video->trimOut() - 2.0) <= kEps && !video->trimToEnd() && video->loop()
        && std::fabs(video->timeOffset() + 0.1) <= kEps;
    std::cout << strf("[selftest] video project mid: opened=%d time=%.9g speed=%.9g loop=%d playing=%d "
                       "clipA trim=%.9g..%.9g toEnd=%d loop=%d offset=%.9g\n",
                       opened ? 1 : 0, transport.time, transport.speed, transport.loop ? 1 : 0,
                       transport.playing ? 1 : 0, video != nullptr ? video->trimIn() : -1.0,
                       video != nullptr ? video->trimOut() : -1.0, video != nullptr && video->trimToEnd() ? 1 : 0,
                       video != nullptr && video->loop() ? 1 : 0, video != nullptr ? video->timeOffset() : 0.0);
    tally.check(opened && std::fabs(transport.time - 1.5) <= kEps && std::fabs(transport.speed - 2.0) <= kEps
                    && transport.loop && !transport.playing && videoOk,
                "video project: the transport or the clipA settings were not restored exactly", results);

    // The project whose video file is removed after the save.
    const std::string gonePath = joinPath(videoDir, "gone.mp4");
    const std::string goneProject = joinPath(videoDir, "gone.sabercat-compare");
    std::error_code ec;
    fs::copy_file(pathFromUtf8(clipA.path), pathFromUtf8(gonePath), fs::copy_options::overwrite_existing, ec);
    if (ec)
    {
        results.fail("video project: cannot copy clipA for the missing-file check: " + ec.message());
    }
    else
    {
        {
            CompareState state(context);
            const int32_t first = state.addMedia(gonePath);
            const int32_t second = state.addMedia(clipB.path);
            if (first != 0 || second != 1 || !state.entries()[0].ready() || !state.entries()[1].ready())
            {
                results.fail("video project: the copy and clipB were not added as Ready sources");
            }
            else
            {
                state.setRole(0, project::SourceRole::A);
                state.setRole(1, project::SourceRole::B);
                std::string saveError;
                tally.check(state.saveProject(goneProject, saveError),
                            "video project: saving gone.sabercat-compare failed: " + saveError, results);
            }
        }
        fs::remove(pathFromUtf8(gonePath), ec);
        CompareState gone(context);
        std::string goneError;
        const bool goneOpened = gone.openProject(goneProject, goneError);
        const bool entriesOk = gone.entries().size() == 2;
        const bool missing = entriesOk && gone.entries()[0].media->status() == SourceStatus::Missing;
        const std::string message = entriesOk ? (gone.entries()[0].message.empty() ? gone.entries()[0].media->statusMessage()
                                                                                   : gone.entries()[0].message)
                                              : std::string();
        const bool other = entriesOk && gone.entries()[1].ready();
        std::cout << strf("[selftest] video project missing: opened=%d entry A %s (\"%s\") entry B ready=%d\n",
                          goneOpened ? 1 : 0, entriesOk ? statusName(gone.entries()[0].media->status()) : "none",
                          message.c_str(), other ? 1 : 0);
        tally.check(goneOpened && missing && !message.empty() && other,
                    "video project: the removed video is not Missing with a message, or the other source is not Ready",
                    results);
        const bool relocated = gone.relocate(0, clipA.path) && gone.entries()[0].ready();
        std::cout << strf("[selftest] video project missing: relocated to clipA ready=%d\n", relocated ? 1 : 0);
        tally.check(relocated, "video project: relocating the missing video to clipA did not make it Ready", results);
    }

    // The smoke project: clipA as A and clipB as B, at time 0.
    const std::string pairPath = joinPath(videoDir, "pair.sabercat-compare");
    bool pairSaved = false;
    bool pairReopened = false;
    {
        CompareState smoke(context);
        if (addClipPair(smoke, clipA, clipB, "video project pair", results))
        {
            smoke.setRole(0, project::SourceRole::A);
            smoke.setRole(1, project::SourceRole::B);
            std::string saveError;
            pairSaved = smoke.saveProject(pairPath, saveError);
            tally.check(pairSaved && smoke.transport().time == 0.0, "video project pair: the save failed or the clock is not 0",
                        results);
        }
    }
    CompareState reopened(context);
    std::string reopenError;
    pairReopened = reopened.openProject(pairPath, reopenError);
    const bool roles = pairReopened && reopened.sourceA() == 0 && reopened.sourceB() == 1;
    std::cout << strf("[selftest] video project pair: saved=%d reopened=%d roles=%d time=%.9g\n", pairSaved ? 1 : 0,
                      pairReopened ? 1 : 0, roles ? 1 : 0, reopened.transport().time);
    tally.check(pairSaved && pairReopened && roles, "video project pair: the smoke project does not round trip: " + reopenError,
                results);

    std::cout << strf("[selftest] video project checks=%d failures=%d\n", tally.checks, tally.failures);
}

// Item 17: the report JSON of the video pair (frame indices and the clock), the report of an image-only pair
// (neither key), the view PNG of a video composite, and a PNG of a video frame from its host copy.
void checkVideoReport(GpuContext & context, Compositor & compositor, MetricsGpu & metrics, const VideoClip & clipA,
                      const VideoClip & clipB, const std::string & videoDir, Results & results)
{
    Tally tally;
    CompareState state(context);
    if (!addClipPair(state, clipA, clipB, "video report", results))
    {
        return;
    }
    state.setViewSize(static_cast<int32_t>(kViewWidth), static_cast<int32_t>(kViewHeight));
    state.seek(1.0);
    state.updateVideos();
    state.updateMetrics(metrics);

    // The video pair as JSON: frameIndex twice (A at 30, B at 24) and timeSeconds once.
    const std::string reportPath = joinPath(videoDir, "video-report.json");
    exportReport(state, reportPath);
    const std::string text = readText(reportPath);
    const std::vector<double> frames = numbersAfter(text, "frameIndex");
    const std::vector<double> seconds = numbersAfter(text, "timeSeconds");
    const bool framesOk = frames.size() == 2 && frames[0] == 30.0 && frames[1] == 24.0;
    const bool secondsOk = seconds.size() == 1 && std::fabs(seconds[0] - 1.0) <= 1e-5;
    std::cout << strf("[selftest] video report json frameIndex=%zu (%.0f, %.0f) timeSeconds=%zu (%.9g)\n", frames.size(),
                      frames.empty() ? -1.0 : frames[0], frames.size() < 2 ? -1.0 : frames[1], seconds.size(),
                      seconds.empty() ? -1.0 : seconds[0]);
    tally.check(framesOk && secondsOk, "video report: the JSON does not carry the frame indices 30 and 24 and the clock 1 s",
                results);

    // The image-only pair (two Display sources): neither key.
    CompareState images(context);
    images.addMemorySource(std::make_unique<ImageSource>(context, makeDisplay(24, 16, 91), "image-a"), "image-a");
    images.addMemorySource(std::make_unique<ImageSource>(context, makeDisplay(24, 16, 92), "image-b"), "image-b");
    if (images.size() != 2 || !images.entries()[0].ready() || !images.entries()[1].ready())
    {
        results.fail("video report: the image-only pair was not added");
    }
    else
    {
        images.updateMetrics(metrics);
        const std::string imagePath = joinPath(videoDir, "image-report.json");
        exportReport(images, imagePath);
        const std::string imageText = readText(imagePath);
        const bool neither = numbersAfter(imageText, "frameIndex").empty() && numbersAfter(imageText, "timeSeconds").empty();
        std::cout << strf("[selftest] video report image-only keys=%s\n", neither ? "none" : "present");
        tally.check(neither, "video report: an image-only pair has a frameIndex or timeSeconds key", results);
    }

    // The view PNG of the video composite: the saved file reads back as the rendered bytes.
    state.setSplit(true);
    compositor.resize(kViewWidth, kViewHeight);
    compositor.render(state.buildComposite(state.transport().time));
    const std::string viewPath = joinPath(videoDir, "video-view.png");
    exportCurrentView(compositor, viewPath);
    std::vector<uint8_t> rgba;
    compositor.readback(rgba);
    checkPng("video view export", viewPath, rgbOfRgba(rgba), kViewWidth, kViewHeight, results);

    // A PNG of a video frame written from its host copy: it reads back as the 8-bit rounding of that copy.
    const ImageData * host = state.entries()[0].media->hostData();
    if (host == nullptr)
    {
        results.fail("video report: clipA has no host copy");
    }
    else
    {
        const Rgb8Image frame = rgb8Of(*host);
        const std::string framePath = joinPath(videoDir, "video-frame.png");
        savePngRgb8(framePath, frame);
        checkPng("video frame export", framePath, frame.rgb, frame.width, frame.height, results);
    }
    std::cout << strf("[selftest] video report checks=%d failures=%d\n", tally.checks, tally.failures);
}

// Item 18: a Linear A and a Display B with non-finite samples (+inf in one r, NaN in another g, -inf in a third
// b). The GPU counts three, its mean is finite and equals flipHdr() on the B crop with every non-finite sample
// set to 0. The control sets the +inf to 1.0 on the CPU side and must differ from the GPU mean by more than 1e-4.
void checkHdrNonFiniteDisplay(GpuContext & context, FlipGpu & flipGpu, Results & results)
{
    constexpr uint32_t kWidth = 24;
    constexpr uint32_t kHeight = 16;
    constexpr size_t kInfPixel = 5;
    constexpr size_t kNanPixel = 100;
    constexpr size_t kNegPixel = 200;
    constexpr double kControlMinDelta = 1e-4;

    const ImageData a = makeLinear(kWidth, kHeight, 91);
    ImageData b = makeDisplay(kWidth, kHeight, 92);
    b.rgba[4 * kInfPixel + 0] = std::numeric_limits<float>::infinity();
    b.rgba[4 * kNanPixel + 1] = std::numeric_limits<float>::quiet_NaN();
    b.rgba[4 * kNegPixel + 2] = -std::numeric_limits<float>::infinity();

    SourcePair sources;
    try
    {
        sources = makeSourcePair(context, a, b, "nonfinite-display");
    }
    catch (const std::exception & e)
    {
        results.fail(std::string("hdr nonfinite-display: ") + e.what());
        return;
    }
    const PixelRect overlap{0, 0, static_cast<int32_t>(kWidth), static_cast<int32_t>(kHeight)};
    const float ppd = defaultFlipPpd();
    std::optional<FlipGpuResult> gpu;
    try
    {
        gpu = flipGpu.computeHdr(*sources.a, 0, 0, *sources.b, 0, 0, overlap, ppd, FlipHdrSettings{});
    }
    catch (const std::exception & e)
    {
        results.fail(std::string("hdr nonfinite-display: the GPU HDR-FLIP threw: ") + e.what());
        return;
    }
    if (!gpu || !gpu->hdr)
    {
        results.fail("hdr nonfinite-display: the GPU HDR-FLIP returned no result");
        return;
    }
    const FlipHdrInfo info = *gpu->hdr;

    // The CPU reference: the same exposures, B with its non-finite samples set to 0.
    const ImageData zeroed = zeroNonFinite(b);
    HdrFlipParams exact;
    exact.toneMapper = ToneMapper::Aces;
    exact.ppd = ppd;
    exact.startExposure = info.start;
    exact.stopExposure = info.stop;
    exact.numExposures = info.count;
    HdrFlipResult cpu;
    HdrFlipResult control;
    try
    {
        cpu = flipHdr(a, zeroed, exact);
        ImageData withOne = zeroed;
        withOne.rgba[4 * kInfPixel + 0] = 1.0f;
        control = flipHdr(a, withOne, exact);
    }
    catch (const std::exception & e)
    {
        results.fail(std::string("hdr nonfinite-display: flipHdr threw: ") + e.what());
        return;
    }
    const double dmean = std::fabs(gpu->mean - cpu.meanError);
    const double controlDelta = std::fabs(gpu->mean - control.meanError);
    std::cout << strf("[selftest] hdr nonfinite-display gpu=%.9g cpu=%.9g dmean=%.3g control=%.9g\n", gpu->mean,
                      cpu.meanError, dmean, control.meanError);
    if (controlDelta <= kControlMinDelta)
    {
        std::cout << strf("[selftest] hdr nonfinite-display control: the +inf as 1.0 differs from the GPU by only %.3g\n",
                          controlDelta);
    }
    if (gpu->nonFiniteSamples != 3)
    {
        results.fail(strf("hdr nonfinite-display: the GPU counted %llu non-finite samples, expected 3",
                          static_cast<unsigned long long>(gpu->nonFiniteSamples)));
    }
    if (!std::isfinite(gpu->mean) || !std::isfinite(cpu.meanError) || !(dmean <= kMaxHdrMeanDelta))
    {
        results.fail(strf("hdr nonfinite-display: GPU mean %.9g differs from flipHdr() %.9g by %.3g", gpu->mean,
                          cpu.meanError, dmean));
    }
}

// Part 1 of the video checks (phase P6): writes the fixed clips into <outDir>/video and runs items 1-6 and 8-12.
// Part 2 (items 13-18, the state checks) runs at the marker at the end.
void runVideoChecks(GpuContext & context, MetricsGpu & metrics, Compositor & compositor, FlipGpu & flipGpu,
                    const std::string & outDir, Results & results)
{
    const std::string videoDir = joinPath(outDir, "video");
    std::error_code ec;
    fs::create_directories(pathFromUtf8(videoDir), ec);
    if (ec || !fs::is_directory(pathFromUtf8(videoDir), ec))
    {
        results.fail("video: cannot create " + videoDir);
        return;
    }

    // clipA is the reference clip: without it none of the video checks can run.
    VideoClip clipA = makeVideoClip("clipA", joinPath(videoDir, "clipA.mp4"), 320, 192, 30, 90, 10);
    std::string reason;
    if (!writeVideoClip(clipA, reason))
    {
        std::cout << "[selftest] video SKIPPED: H.264 encoder unavailable: " << reason << '\n';
        return;
    }
    VideoClip clipB = makeVideoClip("clipB", joinPath(videoDir, "clipB.mp4"), 256, 144, 24, 72, 10);
    VideoClip shortClip = makeVideoClip("short", joinPath(videoDir, "short.mp4"), 320, 192, 30, 15, 10);
    VideoClip clip1080 = makeVideoClip("clip1080", joinPath(videoDir, "clip1080.mp4"), 1920, 1080, 30, 6, 3);
    VideoClip rot90 = makeVideoClip("rot90", joinPath(videoDir, "rot90.mp4"), 320, 192, 30, 6, 10, 90);
    VideoClip webm = makeVideoClip("clipA.webm", joinPath(videoDir, "clipA.webm"), 320, 192, 30, 90, 10);
    for (VideoClip * clip : {&clipB, &shortClip, &clip1080, &rot90})
    {
        std::string why;
        if (!writeVideoClip(*clip, why))
        {
            std::cout << strf("[selftest] video %s not written, its checks are skipped: %s\n", clip->label.c_str(),
                              why.c_str());
        }
    }

    // A non-ASCII directory and file name (U+00FC, U+65E5) holding a copy of clipA.
    const std::string utf8Dir = joinPath(videoDir, "v\xC3\xBC\xE6\x97\xA5");
    const std::string utf8Path = joinPath(utf8Dir, "clip \xC3\xBC.mp4");
    ec.clear();
    fs::create_directories(pathFromUtf8(utf8Dir), ec);
    ec.clear();
    fs::copy_file(pathFromUtf8(clipA.path), pathFromUtf8(utf8Path), fs::copy_options::overwrite_existing, ec);

    // 1. Exact decode of every frame.
    guardedCheck("video exact", results, [&]
    {
        for (const VideoClip * clip : {&clipA, &clipB, &clip1080})
        {
            if (clip->written)
            {
                checkVideoExact(*clip, results);
            }
        }
    });
    // 2. Seeks.
    guardedCheck("video seeks", results, [&]
    {
        checkVideoSeeks(clipA, 7, results);
        if (clipB.written)
        {
            checkVideoSeeks(clipB, 11, results);
        }
    });
    // 3. Rotation observed.
    guardedCheck("video rotation", results, [&]
    {
        if (rot90.written)
        {
            checkVideoRotation(rot90, results);
        }
    });
    // 4. WebM encoder or refusal, and the junk file.
    guardedCheck("video webm", results, [&] { checkVideoWebm(webm, videoDir, results); });
    // 5. Sources that cannot play.
    guardedCheck("video errors", results, [&] { checkVideoErrors(context, videoDir, clipA.path, results); });
    // 6. The NV12 conversion.
    guardedCheck("video convert", results, [&] { checkVideoConvert(context, clipA, results); });
    // 8. The timeline table and the refusals.
    guardedCheck("video timeline", results, [&] { checkVideoTimeline(context, clipA, results); });
    // 9. Two frame rates at the same global times.
    guardedCheck("video fps-map", results, [&]
    {
        if (clipB.written)
        {
            checkVideoFpsMap(context, clipA, clipB, results);
        }
    });
    // 10. The non-ASCII path.
    guardedCheck("video utf8 path", results, [&]
    {
        if (ec)
        {
            results.fail("video utf8 path: copy failed: " + ec.message());
        }
        else
        {
            checkVideoUtf8(context, utf8Path, results);
        }
    });
    // 11. The memory refusal and the size limit.
    guardedCheck("video budget", results, [&] { checkVideoBudget(context, clipA, results); });
    // 12. Timing (measurement only).
    guardedCheck("video perf", results, [&]
    {
        if (clipB.written && clip1080.written)
        {
            checkVideoPerf(context, compositor, clipA, clipB, clip1080, results);
        }
        else
        {
            std::cout << "[selftest] video perf SKIPPED: clipB or clip1080 was not written\n";
        }
    });

    // ---- S2 video state checks (items 13-18) ----
    // 13. Composites of two videos of different sizes against the CPU reference, over the table.
    guardedCheck("video composites", results, [&]
    {
        if (clipB.written)
        {
            checkVideoComposites(context, compositor, clipA, clipB, results);
        }
        else
        {
            std::cout << "[selftest] video composite SKIPPED: clipB was not written\n";
        }
    });
    // 14. Metrics and LDR FLIP over six frame pairs.
    guardedCheck("video metrics", results, [&]
    {
        if (clipB.written)
        {
            checkVideoMetrics(context, metrics, clipA, clipB, results);
        }
        else
        {
            std::cout << "[selftest] video metrics SKIPPED: clipB was not written\n";
        }
    });
    // 15. The transport: speed, clock, stepping, seeks, loop, the short clip, hide and show, removal.
    guardedCheck("video transport", results, [&]
    {
        if (clipB.written)
        {
            checkVideoTransport(context, clipA, clipB, shortClip, results);
        }
        else
        {
            std::cout << "[selftest] video transport SKIPPED: clipB was not written\n";
        }
    });
    // 16. The project: the mid-playback round trip, a removed video, and the smoke pair.
    guardedCheck("video project", results, [&]
    {
        if (clipB.written)
        {
            checkVideoProject(context, clipA, clipB, videoDir, results);
        }
        else
        {
            std::cout << "[selftest] video project SKIPPED: clipB was not written\n";
        }
    });
    // 17. The report JSON, the image-only report, and the view and frame PNGs.
    guardedCheck("video report", results, [&]
    {
        if (clipB.written)
        {
            checkVideoReport(context, compositor, metrics, clipA, clipB, videoDir, results);
        }
        else
        {
            std::cout << "[selftest] video report SKIPPED: clipB was not written\n";
        }
    });
    // 18. HDR-FLIP with non-finite samples in a Display B (the Display rule).
    guardedCheck("hdr nonfinite-display", results, [&] { checkHdrNonFiniteDisplay(context, flipGpu, results); });
}

void runGpuChecks(sabercat::Renderer & host, const std::string & shaderDir, const std::string & outDir,
                  Results & results)
{
    GpuContext context(host.sabercatDevice(), shaderDir);
    MetricsGpu metrics(context);
    FlipGpu flipGpu(context);
    Compositor compositor(context);
    checkFailedLoads(context, outDir, results);
    try
    {
        checkState(context, metrics, outDir, results);
        checkDividerComposites(context, compositor, results);
        checkThreeSourcePanes(context, compositor, results);
    }
    catch (const std::exception & e)
    {
        results.fail(std::string("state: ") + e.what());
    }
    for (const Case & c : buildCases())
    {
        try
        {
            runCase(c, context, metrics, flipGpu, compositor, outDir, results);
        }
        catch (const std::exception & e)
        {
            results.fail(c.name + ": " + e.what());
        }
    }
    guardedCheck("flip nonfinite", results, [&] { checkFlipNonFinite(context, flipGpu, results); });
    guardedCheck("flip state", results, [&] { checkFlipState(context, metrics, flipGpu, outDir, results); });
    guardedCheck("flip nvlabs", results, [&] { checkFlipNvlabs(context, flipGpu, results); });
    guardedCheck("hdr cases", results, [&] { checkHdrCases(context, flipGpu, results); });
    guardedCheck("hdr nonfinite", results, [&] { checkHdrNonFinite(context, flipGpu, results); });
    guardedCheck("hdr nvlabs", results, [&] { checkHdrNvlabs(context, flipGpu, results); });
    guardedCheck("hdr overrides", results, [&] { checkHdrOverrides(context, flipGpu, results); });
    guardedCheck("hdr budget", results, [&] { checkHdrBudget(context, flipGpu, results); });
    guardedCheck("hdr state", results, [&] { checkHdrState(context, metrics, flipGpu, outDir, results); });
    guardedCheck("hdr outputs", results, [&] { checkHdrOutputs(context, flipGpu, compositor, outDir, results); });
    runSceneChecks(host, context, metrics, compositor, shaderDir, outDir, results);
    guardedCheck("video", results, [&] { runVideoChecks(context, metrics, compositor, flipGpu, outDir, results); });
}

} // namespace

int runSelfTest(const std::string & outDir, bool validation)
{
    Results results;
    std::cout << "[selftest] start: validation " << (validation ? "on" : "off") << '\n';
    try
    {
        std::error_code ec;
        fs::create_directories(pathFromUtf8(outDir), ec);
        if (ec || !fs::is_directory(pathFromUtf8(outDir), ec))
        {
            throw std::runtime_error("cannot create the output directory " + outDir);
        }

        runCpuChecks(results);

        // Headless: no window extensions, no presentation requirement (as the Studio's self-test).
        BorrowedVulkanHost host;
        Renderer renderer(64, 64);
        try
        {
            host.create({}, /*presentation=*/false, validation);
            renderer.initializeBorrowed(host.context(), host.enabled, resolveShaderDir());
            runGpuChecks(renderer, resolveShaderDir(), outDir, results);
        }
        catch (...)
        {
            renderer.shutdown();
            host.destroy();
            throw;
        }
        renderer.shutdown();
        host.destroy();
    }
    catch (const std::exception & e)
    {
        results.fail(std::string("exception: ") + e.what());
    }

    if (results.failures.empty())
    {
        std::cout << "[selftest] PASS" << std::endl;
        return 0;
    }
    std::string summary = std::to_string(results.failures.size()) + " failure(s):";
    for (const std::string & failure : results.failures)
    {
        summary += " " + failure + ";";
    }
    std::cout << "[selftest] FAIL: " << summary << std::endl;
    return 1;
}

} // namespace sabercat::compare::gui
