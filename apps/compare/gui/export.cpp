// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "export.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include "compare_state.h"
#include "image_io.h"

namespace sabercat::compare::gui
{

namespace
{

// Clamp to [0,1]; NaN and negatives give 0 (clampUnit in compare_layout.cpp).
float clampUnit(float v)
{
    return (v > 0.0f) ? std::min(v, 1.0f) : 0.0f;
}

// A display value as an 8-bit channel: floor(v * 255 + 0.5), the rounding of the
// compositor's UNORM store (up to exact ties). NaN gives 0.
uint8_t toByte(float v)
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

// The layout the scene describes: its sources' geometry (hidden ones included).
LayoutInput layoutOf(const ExportScene & scene)
{
    LayoutInput layout;
    layout.sourceCount = static_cast<int32_t>(scene.sources.size());
    for (size_t i = 0; i < scene.sources.size(); ++i)
    {
        layout.sources[i] = scene.sources[i].geometry;
    }
    return layout;
}

// Checks the scene: at most kMaxSources entries, each with an image whose size matches
// its geometry and whose sample count is right.
void validateScene(const ExportScene & scene)
{
    if (scene.sources.size() > static_cast<size_t>(kMaxSources))
    {
        throw std::invalid_argument("export scene has more than " + std::to_string(kMaxSources) + " sources");
    }
    for (const ExportSource & source : scene.sources)
    {
        if (source.image == nullptr)
        {
            throw std::invalid_argument("export source has no image");
        }
        const ImageData & image = *source.image;
        if (image.width != static_cast<uint32_t>(source.geometry.width)
            || image.height != static_cast<uint32_t>(source.geometry.height))
        {
            throw std::invalid_argument("export source size does not match its layout entry");
        }
        if (image.rgba.size() != static_cast<size_t>(image.width) * image.height * 4)
        {
            throw std::invalid_argument("export source has a wrong sample count");
        }
    }
}

// The source at index when it exists and is visible, or nullptr.
const ExportSource * visibleSource(const ExportScene & scene, int32_t index)
{
    if (index < 0 || static_cast<size_t>(index) >= scene.sources.size())
    {
        return nullptr;
    }
    const ExportSource & source = scene.sources[static_cast<size_t>(index)];
    return source.geometry.visible ? &source : nullptr;
}

// The source at index, which must be visible; throws otherwise.
const ExportSource & requireVisible(const ExportScene & scene, int32_t index)
{
    const ExportSource * source = visibleSource(scene, index);
    if (source == nullptr)
    {
        throw std::invalid_argument("export source " + std::to_string(index) + " is not visible");
    }
    return *source;
}

// Display color of one source sample, as the compositor shows it in a pane: premultiplied
// rgb is composited over the background first (rgb + (1 - clamp(a)) * background), then a
// Display sample is clamped and a Linear sample goes through displayTransform(). (x, y)
// are local integer coordinates inside the image.
void paneColor(const ExportSource & source, int32_t x, int32_t y, float out[3])
{
    const ImageData & image = *source.image;
    const size_t index = (static_cast<size_t>(y) * image.width + static_cast<size_t>(x)) * 4;
    const float * samples = &image.rgba[index];
    const float holdback = image.premultiplied ? 1.0f - clampUnit(samples[3]) : 0.0f;
    const float background[3] = {source.background.r, source.background.g, source.background.b};
    const bool linear = image.encoding == Encoding::Linear;
    for (int c = 0; c < 3; ++c)
    {
        const float value = samples[c] + holdback * background[c];
        out[c] = linear ? displayTransform(value, source.toneMapper, source.exposureStops) : clampUnit(value);
    }
}

// Writes one RGB pixel, as bytes, into an image of the given width.
void writePixel(Rgb8Image & image, uint32_t x, uint32_t y, const float rgb[3])
{
    uint8_t * out = &image.rgb[(static_cast<size_t>(y) * image.width + x) * 3];
    out[0] = toByte(rgb[0]);
    out[1] = toByte(rgb[1]);
    out[2] = toByte(rgb[2]);
}

// A zeroed image over the extent. Throws for an empty extent: there is nothing to export.
Rgb8Image emptyImage(const PixelRect & extent)
{
    if (extent.width <= 0 || extent.height <= 0)
    {
        throw std::invalid_argument("the padded extent is empty; nothing to export");
    }
    Rgb8Image image;
    image.width = static_cast<uint32_t>(extent.width);
    image.height = static_cast<uint32_t>(extent.height);
    image.rgb.assign(static_cast<size_t>(image.width) * image.height * 3, 0);
    return image;
}

} // namespace

PixelRect exportExtent(const ExportScene & scene)
{
    validateScene(scene);
    return paddedExtent(layoutOf(scene));
}

Rgb8Image paddedPaneImage(const ExportScene & scene, int32_t index)
{
    validateScene(scene);
    const ExportSource & own = requireVisible(scene, index);
    const PixelRect extent = paddedExtent(layoutOf(scene));
    Rgb8Image image = emptyImage(extent);

    const PixelRect ownRect = sourceRect(own.geometry);
    const ExportSource * counterpart = visibleSource(scene, own.counterpart);
    const float background[3] = {kExportBackground, kExportBackground, kExportBackground};

    for (uint32_t y = 0; y < image.height; ++y)
    {
        for (uint32_t x = 0; x < image.width; ++x)
        {
            const int32_t canvasX = extent.x + static_cast<int32_t>(x);
            const int32_t canvasY = extent.y + static_cast<int32_t>(y);
            if (rectContains(ownRect, canvasX, canvasY))
            {
                float rgb[3];
                paneColor(own, canvasX - own.geometry.offsetX, canvasY - own.geometry.offsetY, rgb);
                writePixel(image, x, y, rgb);
                continue;
            }
            RgbaColor padding;
            if (counterpart != nullptr
                && paddingColorFor(*counterpart->image, canvasX - counterpart->geometry.offsetX,
                                   canvasY - counterpart->geometry.offsetY, counterpart->toneMapper,
                                   counterpart->exposureStops, padding, counterpart->background))
            {
                const float rgb[3] = {padding.r, padding.g, padding.b};
                writePixel(image, x, y, rgb);
            }
            else
            {
                writePixel(image, x, y, background);
            }
        }
    }
    return image;
}

Rgb8Image diffMapImage(const ExportScene & scene, float diffGain)
{
    validateScene(scene);
    if (!std::isfinite(diffGain) || diffGain < 0.0f)
    {
        throw std::invalid_argument("diffGain must be finite and not negative");
    }
    const ExportSource & a = requireVisible(scene, scene.sourceA);
    const ExportSource & b = requireVisible(scene, scene.sourceB);
    const PixelRect extent = paddedExtent(layoutOf(scene));
    const Rgb8Image shape = emptyImage(extent);

    // displayRgb(): Display as stored, Linear clamped and sRGB encoded, premultiplied rgb
    // unchanged. It also rejects non-finite samples.
    const Image displayA = displayRgb(*a.image);
    const Image displayB = displayRgb(*b.image);
    const PixelRect rectA = sourceRect(a.geometry);
    const PixelRect rectB = sourceRect(b.geometry);

    std::vector<float> gray(static_cast<size_t>(shape.width) * shape.height);
    for (uint32_t y = 0; y < shape.height; ++y)
    {
        for (uint32_t x = 0; x < shape.width; ++x)
        {
            const int32_t canvasX = extent.x + static_cast<int32_t>(x);
            const int32_t canvasY = extent.y + static_cast<int32_t>(y);
            float value = mapPaddingValue();
            if (rectContains(rectA, canvasX, canvasY) && rectContains(rectB, canvasX, canvasY))
            {
                const size_t ia = (static_cast<size_t>(canvasY - a.geometry.offsetY) * displayA.width
                                   + static_cast<size_t>(canvasX - a.geometry.offsetX)) * 3;
                const size_t ib = (static_cast<size_t>(canvasY - b.geometry.offsetY) * displayB.width
                                   + static_cast<size_t>(canvasX - b.geometry.offsetX)) * 3;
                const float * pa = &displayA.rgb[ia];
                const float * pb = &displayB.rgb[ib];
                const float mean = (std::fabs(pa[0] - pb[0]) + std::fabs(pa[1] - pb[1]) + std::fabs(pa[2] - pb[2]))
                    / 3.0f;
                value = std::min(1.0f, mean * diffGain);
            }
            gray[static_cast<size_t>(y) * shape.width + x] = value;
        }
    }
    // The padded extent is a rectangle, so every pixel is inside it: no background here.
    return colorizeMagma(gray, shape.width, shape.height);
}

Rgb8Image flipMapImage(const ExportScene & scene, const std::vector<float> & errorMap, PixelRect overlap)
{
    validateScene(scene);
    const PixelRect extent = paddedExtent(layoutOf(scene));
    const Rgb8Image shape = emptyImage(extent);
    if (overlap.width <= 0 || overlap.height <= 0)
    {
        throw std::invalid_argument("the FLIP overlap is empty; nothing to export");
    }
    // Int64 so that the sums cannot wrap for any int32 rectangle.
    if (static_cast<int64_t>(overlap.x) < extent.x || static_cast<int64_t>(overlap.y) < extent.y
        || static_cast<int64_t>(overlap.x) + overlap.width > static_cast<int64_t>(extent.x) + extent.width
        || static_cast<int64_t>(overlap.y) + overlap.height > static_cast<int64_t>(extent.y) + extent.height)
    {
        throw std::invalid_argument("the FLIP overlap is not inside the padded extent");
    }
    const size_t overlapPixels = static_cast<size_t>(overlap.width) * static_cast<size_t>(overlap.height);
    if (errorMap.size() != overlapPixels)
    {
        throw std::invalid_argument("the FLIP map has " + std::to_string(errorMap.size()) + " values, expected "
            + std::to_string(overlapPixels));
    }

    // Padding takes the top of the map. colorizeMagma() clamps each value to [0,1].
    std::vector<float> gray(static_cast<size_t>(shape.width) * shape.height, mapPaddingValue());
    for (uint32_t y = 0; y < shape.height; ++y)
    {
        for (uint32_t x = 0; x < shape.width; ++x)
        {
            const int32_t canvasX = extent.x + static_cast<int32_t>(x);
            const int32_t canvasY = extent.y + static_cast<int32_t>(y);
            if (rectContains(overlap, canvasX, canvasY))
            {
                const size_t index = static_cast<size_t>(canvasY - overlap.y) * static_cast<size_t>(overlap.width)
                    + static_cast<size_t>(canvasX - overlap.x);
                gray[static_cast<size_t>(y) * shape.width + x] = errorMap[index];
            }
        }
    }
    return colorizeMagma(gray, shape.width, shape.height);
}

void saveRgbaPng(const std::string & utf8Path, const uint8_t * rgba, uint32_t width, uint32_t height)
{
    if (rgba == nullptr || width == 0 || height == 0)
    {
        throw std::invalid_argument("saveRgbaPng needs pixels and a non-empty size");
    }
    Rgb8Image image;
    image.width = width;
    image.height = height;
    const size_t pixels = static_cast<size_t>(width) * height;
    image.rgb.resize(pixels * 3);
    for (size_t i = 0; i < pixels; ++i)
    {
        image.rgb[3 * i] = rgba[4 * i];
        image.rgb[3 * i + 1] = rgba[4 * i + 1];
        image.rgb[3 * i + 2] = rgba[4 * i + 2];
    }
    savePngRgb8(utf8Path, image);
}

CompareReport makeReport(const ReportInput & input)
{
    if (input.a == nullptr || input.b == nullptr)
    {
        throw std::invalid_argument("makeReport needs both images");
    }
    const PixelRect & o = input.overlap;
    const int64_t localX = static_cast<int64_t>(o.x) - input.aOffsetX;
    const int64_t localY = static_cast<int64_t>(o.y) - input.aOffsetY;
    if (o.width <= 0 || o.height <= 0 || localX < 0 || localY < 0
        || localX + o.width > static_cast<int64_t>(input.a->width)
        || localY + o.height > static_cast<int64_t>(input.a->height))
    {
        throw std::invalid_argument("the report overlap is empty or outside image A");
    }

    const auto describe = [](const std::string & path, const ImageData & image)
    {
        ReportSource source;
        source.path = path;
        source.width = image.width;
        source.height = image.height;
        source.bitDepth = image.bitDepth;
        source.encoding = image.encoding == Encoding::Linear ? "linear" : "display";
        source.hasAlpha = image.hasAlpha;
        return source;
    };

    CompareReport report;
    report.reference = describe(input.pathA, *input.a);
    report.test = describe(input.pathB, *input.b);
    report.overlap.x = static_cast<uint32_t>(localX);
    report.overlap.y = static_cast<uint32_t>(localY);
    report.overlap.width = static_cast<uint32_t>(o.width);
    report.overlap.height = static_cast<uint32_t>(o.height);
    report.anchor = input.anchor;
    report.ppd = input.ppd;
    report.flipMean = input.flipMean;
    report.hdr = input.hdr;
    if (input.linearMse && std::isfinite(*input.linearMse))
    {
        report.linearMse = input.linearMse;
    }
    report.psnrDb = input.psnrDb;
    report.mse = input.mse;
    report.reference.frameIndex = input.aFrameIndex;
    report.test.frameIndex = input.bFrameIndex;
    if (input.timeSeconds && std::isfinite(*input.timeSeconds))
    {
        report.timeSeconds = input.timeSeconds;
    }
    return report;
}

void addVideoReportFields(const CompareState & state, ReportInput & input)
{
    const int32_t a = state.sourceA();
    const int32_t b = state.sourceB();
    const auto isVideo = [&state](int32_t index)
    {
        return index >= 0 && state.entries()[static_cast<size_t>(index)].media->kind() == project::SourceKind::Video;
    };
    const bool videoA = isVideo(a);
    const bool videoB = isVideo(b);
    if (!videoA && !videoB)
    {
        return;
    }
    // A video that has no frame yet (not Ready) has no index; the report leaves it out.
    const auto frameOf = [&state](int32_t index, bool video) -> std::optional<int64_t>
    {
        if (!video)
        {
            return std::nullopt;
        }
        const int64_t frame = state.videoFrameIndex(static_cast<size_t>(index));
        return frame >= 0 ? std::optional<int64_t>(frame) : std::nullopt;
    };
    input.aFrameIndex = frameOf(a, videoA);
    input.bFrameIndex = frameOf(b, videoB);
    input.timeSeconds = state.transport().time;
}

void saveReportFile(const std::string & utf8Path, const CompareReport & report)
{
    saveReport(utf8Path, report);
}

} // namespace sabercat::compare::gui
