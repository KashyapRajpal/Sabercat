// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Implementation of the compare layout model. The rules and the units are documented
// in compare_layout.h.
#include "compare_layout.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace sabercat::compare
{

namespace
{

// Largest finite half-float: the clamp for linear samples before tone mapping.
constexpr float kMaxLinear = 65504.0f;
constexpr float kMaxExposureStops = 32.0f;
// 2^30: floor() results are saturated to this before the int32 cast.
constexpr double kPixelLimit = 1073741824.0;

// Source at index, or nullptr when the index is out of range.
const LayoutSource * sourceAt(const LayoutInput & input, int32_t index)
{
    if (index < 0 || index >= input.sourceCount || index >= kMaxSources)
    {
        return nullptr;
    }
    return &input.sources[static_cast<size_t>(index)];
}

// Source at index when it exists and is visible, or nullptr.
const LayoutSource * visibleSourceAt(const LayoutInput & input, int32_t index)
{
    const LayoutSource * source = sourceAt(input, index);
    return (source != nullptr && source->visible) ? source : nullptr;
}

// Floor of a continuous coordinate as a pixel index. NaN maps to 0.
int32_t floorToPixel(double value)
{
    if (std::isnan(value))
    {
        return 0;
    }
    return static_cast<int32_t>(std::clamp(std::floor(value), -kPixelLimit, kPixelLimit));
}

// Boundary i of `parts` equal shares of `total` pixels: floor(i * total / parts).
int32_t shareBoundary(int32_t total, int32_t index, int32_t parts)
{
    return static_cast<int32_t>(static_cast<int64_t>(index) * total / parts);
}

// Fit zoom in percent: the padded extent into the smallest pane, centered.
double fitZoomPercent(const LayoutInput & input)
{
    const PixelRect extent = paddedExtent(input);
    const PaneLayout panes = computePanes(input);
    if (extent.width <= 0 || extent.height <= 0 || panes.count <= 0)
    {
        return 100.0;
    }

    int32_t minWidth = std::numeric_limits<int32_t>::max();
    int32_t minHeight = std::numeric_limits<int32_t>::max();
    for (int32_t i = 0; i < panes.count; ++i)
    {
        minWidth = std::min(minWidth, panes.panes[static_cast<size_t>(i)].rect.width);
        minHeight = std::min(minHeight, panes.panes[static_cast<size_t>(i)].rect.height);
    }
    if (minWidth <= 0 || minHeight <= 0)
    {
        return 100.0;
    }

    const double zoom = std::min(static_cast<double>(minWidth) / extent.width,
        static_cast<double>(minHeight) / extent.height);
    return clampZoomPercent(zoom * 100.0);
}

// Start and length of the padded extent along the wipe's axis.
void wipeAxis(const LayoutInput & input, double & start, double & length)
{
    const PixelRect extent = paddedExtent(input);
    if (input.view.wipeOrientation == WipeOrientation::Vertical)
    {
        start = extent.x;
        length = extent.width;
    }
    else
    {
        start = extent.y;
        length = extent.height;
    }
}

// Clamps a sample to [0,1]; NaN and negatives are 0.
float clampUnit(float value)
{
    if (!(value > 0.0f))
    {
        return 0.0f;
    }
    return std::min(value, 1.0f);
}

float srgbTransfer(float linear)
{
    if (linear <= 0.0031308f)
    {
        return 12.92f * linear;
    }
    return 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
}

// Hable's filmic curve (uncharted 2), before normalization.
float hableCurve(float x)
{
    constexpr float a = 0.15f;
    constexpr float b = 0.50f;
    constexpr float c = 0.10f;
    constexpr float d = 0.20f;
    constexpr float e = 0.02f;
    constexpr float f = 0.30f;
    return ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f;
}

// Tone curve for a non-negative linear value. Result is clamped to [0,1] by the caller.
float toneMap(float x, ToneMapper toneMapper)
{
    switch (toneMapper)
    {
    case ToneMapper::Reinhard:
        return x / (1.0f + x);
    case ToneMapper::Aces:
    {
        // Narkowicz's fit of the ACES filmic curve.
        constexpr float a = 2.51f;
        constexpr float b = 0.03f;
        constexpr float c = 2.43f;
        constexpr float d = 0.59f;
        constexpr float e = 0.14f;
        return (x * (a * x + b)) / (x * (c * x + d) + e);
    }
    case ToneMapper::Hable:
        return hableCurve(x) / hableCurve(11.2f);
    }
    return x / (1.0f + x);
}

} // namespace

// ---- Sources ---------------------------------------------------------------------

int32_t visibleCount(const LayoutInput & input)
{
    int32_t count = 0;
    for (int32_t i = 0; i < input.sourceCount && i < kMaxSources; ++i)
    {
        if (input.sources[static_cast<size_t>(i)].visible)
        {
            ++count;
        }
    }
    return count;
}

bool addSource(LayoutInput & input, const LayoutSource & source)
{
    if (input.sourceCount < 0 || input.sourceCount >= kMaxSources)
    {
        return false;
    }
    if (source.width < 1 || source.width > kMaxSourceSize || source.height < 1 || source.height > kMaxSourceSize)
    {
        return false;
    }
    if (source.offsetX < -kMaxSourceOffset || source.offsetX > kMaxSourceOffset || source.offsetY < -kMaxSourceOffset
        || source.offsetY > kMaxSourceOffset)
    {
        return false;
    }
    if (source.visible && visibleCount(input) >= kMaxVisible)
    {
        return false;
    }
    input.sources[static_cast<size_t>(input.sourceCount)] = source;
    ++input.sourceCount;
    return true;
}

bool setSourceVisible(LayoutInput & input, int32_t index, bool visible)
{
    const LayoutSource * found = sourceAt(input, index);
    if (found == nullptr)
    {
        return false;
    }
    LayoutSource & source = input.sources[static_cast<size_t>(index)];
    if (source.visible == visible)
    {
        return true;
    }
    if (visible && visibleCount(input) >= kMaxVisible)
    {
        return false;
    }
    source.visible = visible;
    return true;
}

void anchorTopLeft(LayoutInput & input)
{
    for (int32_t i = 0; i < input.sourceCount && i < kMaxSources; ++i)
    {
        LayoutSource & source = input.sources[static_cast<size_t>(i)];
        source.offsetX = 0;
        source.offsetY = 0;
    }
}

void anchorCentered(LayoutInput & input)
{
    int32_t maxWidth = 0;
    int32_t maxHeight = 0;
    for (int32_t i = 0; i < input.sourceCount && i < kMaxSources; ++i)
    {
        const LayoutSource & source = input.sources[static_cast<size_t>(i)];
        maxWidth = std::max(maxWidth, source.width);
        maxHeight = std::max(maxHeight, source.height);
    }
    for (int32_t i = 0; i < input.sourceCount && i < kMaxSources; ++i)
    {
        LayoutSource & source = input.sources[static_cast<size_t>(i)];
        // The differences are never negative, so integer division rounds down.
        source.offsetX = (maxWidth - source.width) / 2;
        source.offsetY = (maxHeight - source.height) / 2;
    }
}

// ---- Extents ---------------------------------------------------------------------

PixelRect sourceRect(const LayoutSource & source)
{
    return PixelRect{source.offsetX, source.offsetY, source.width, source.height};
}

PixelRect paddedExtent(const LayoutInput & input)
{
    bool any = false;
    int32_t left = 0;
    int32_t top = 0;
    int32_t right = 0;
    int32_t bottom = 0;
    for (int32_t i = 0; i < input.sourceCount && i < kMaxSources; ++i)
    {
        const LayoutSource & source = input.sources[static_cast<size_t>(i)];
        if (!source.visible)
        {
            continue;
        }
        const int32_t l = source.offsetX;
        const int32_t t = source.offsetY;
        const int32_t r = source.offsetX + source.width;
        const int32_t b = source.offsetY + source.height;
        if (!any)
        {
            left = l;
            top = t;
            right = r;
            bottom = b;
            any = true;
        }
        else
        {
            left = std::min(left, l);
            top = std::min(top, t);
            right = std::max(right, r);
            bottom = std::max(bottom, b);
        }
    }
    if (!any)
    {
        return PixelRect{};
    }
    return PixelRect{left, top, right - left, bottom - top};
}

PixelRect intersectRects(const PixelRect & a, const PixelRect & b)
{
    if (a.width <= 0 || a.height <= 0 || b.width <= 0 || b.height <= 0)
    {
        return PixelRect{};
    }
    const int32_t left = std::max(a.x, b.x);
    const int32_t top = std::max(a.y, b.y);
    const int32_t right = std::min(a.x + a.width, b.x + b.width);
    const int32_t bottom = std::min(a.y + a.height, b.y + b.height);
    if (right <= left || bottom <= top)
    {
        return PixelRect{};
    }
    return PixelRect{left, top, right - left, bottom - top};
}

PixelRect overlapRect(const LayoutInput & input, int32_t a, int32_t b)
{
    const LayoutSource * first = visibleSourceAt(input, a);
    const LayoutSource * second = visibleSourceAt(input, b);
    if (first == nullptr || second == nullptr)
    {
        return PixelRect{};
    }
    return intersectRects(sourceRect(*first), sourceRect(*second));
}

bool rectContains(const PixelRect & rect, int32_t x, int32_t y)
{
    // The subtractions cannot overflow: the rectangle corner is within +-2^25.
    return rect.width > 0 && rect.height > 0 && x >= rect.x && y >= rect.y && x - rect.x < rect.width
        && y - rect.y < rect.height;
}

// ---- Panes -----------------------------------------------------------------------

PaneLayout computePanes(const LayoutInput & input)
{
    PaneLayout out;
    // A non-positive side is treated as 0 on its own axis.
    const int32_t width = std::max(0, input.windowWidth);
    const int32_t height = std::max(0, input.windowHeight);

    if (input.view.mode == ViewMode::SingleScreen)
    {
        if (visibleSourceAt(input, input.view.abIndex) == nullptr)
        {
            return out;
        }
        out.panes[0] = Pane{PixelRect{0, 0, width, height}, input.view.abIndex};
        out.count = 1;
        return out;
    }

    const int32_t count = visibleCount(input);
    if (count == 0)
    {
        return out;
    }

    int32_t columns = count;
    int32_t rows = 1;
    if (input.view.arrangement == PaneArrangement::Grid2x2)
    {
        columns = count >= 2 ? 2 : 1;
        rows = count >= 3 ? 2 : 1;
    }

    int32_t slot = 0;
    for (int32_t i = 0; i < input.sourceCount && i < kMaxSources; ++i)
    {
        if (!input.sources[static_cast<size_t>(i)].visible)
        {
            continue;
        }
        const int32_t column = slot % columns;
        const int32_t row = slot / columns;
        const int32_t left = shareBoundary(width, column, columns);
        const int32_t right = shareBoundary(width, column + 1, columns);
        const int32_t top = shareBoundary(height, row, rows);
        const int32_t bottom = shareBoundary(height, row + 1, rows);
        out.panes[static_cast<size_t>(slot)] = Pane{PixelRect{left, top, right - left, bottom - top}, i};
        ++slot;
    }
    out.count = count;
    return out;
}

int32_t paneAt(const PaneLayout & panes, int32_t x, int32_t y)
{
    const int32_t count = std::min(panes.count, kMaxVisible);
    for (int32_t i = 0; i < count; ++i)
    {
        if (rectContains(panes.panes[static_cast<size_t>(i)].rect, x, y))
        {
            return i;
        }
    }
    return -1;
}

// ---- Zoom and pan ----------------------------------------------------------------

double clampZoomPercent(double percent)
{
    if (std::isnan(percent))
    {
        return 100.0;
    }
    return std::clamp(percent, kMinZoomPercent, kMaxZoomPercent);
}

void setZoomFit(LayoutInput & input)
{
    input.view.zoomMode = ZoomMode::Fit;
}

void setZoomPercent(LayoutInput & input, double percent)
{
    input.view.zoomPercent = clampZoomPercent(percent);
    input.view.zoomMode = ZoomMode::Percent;
}

double effectiveZoom(const LayoutInput & input)
{
    const double percent = input.view.zoomMode == ZoomMode::Fit ? fitZoomPercent(input)
                                                                : clampZoomPercent(input.view.zoomPercent);
    return percent / 100.0;
}

bool zoomAt(LayoutInput & input, int32_t windowX, int32_t windowY, double factor)
{
    if (!std::isfinite(factor) || !(factor > 0.0))
    {
        return false;
    }
    const PaneLayout panes = computePanes(input);
    const int32_t index = paneAt(panes, windowX, windowY);
    if (index < 0)
    {
        return false;
    }
    const PixelRect & rect = panes.panes[static_cast<size_t>(index)].rect;
    const double centerX = rect.x + rect.width * 0.5;
    const double centerY = rect.y + rect.height * 0.5;
    const double offsetX = (windowX + 0.5) - centerX;
    const double offsetY = (windowY + 0.5) - centerY;

    // The image point under the anchor, at the old zoom.
    const double oldZoom = effectiveZoom(input);
    const double anchorImageX = input.view.panX + offsetX / oldZoom;
    const double anchorImageY = input.view.panY + offsetY / oldZoom;

    // New zoom, clamped, then the pan that puts the same image point under the anchor.
    const double newPercent = clampZoomPercent(oldZoom * factor * 100.0);
    const double newZoom = newPercent / 100.0;
    input.view.zoomPercent = newPercent;
    input.view.zoomMode = ZoomMode::Percent;
    input.view.panX = anchorImageX - offsetX / newZoom;
    input.view.panY = anchorImageY - offsetY / newZoom;
    return true;
}

void setPan(LayoutInput & input, double imageX, double imageY)
{
    if (std::isfinite(imageX) && std::isfinite(imageY))
    {
        input.view.panX = imageX;
        input.view.panY = imageY;
    }
}

void centerView(LayoutInput & input)
{
    const PixelRect extent = paddedExtent(input);
    input.view.panX = extent.x + extent.width * 0.5;
    input.view.panY = extent.y + extent.height * 0.5;
}

// ---- Mapping ---------------------------------------------------------------------

ImageHit windowToImage(const LayoutInput & input, int32_t windowX, int32_t windowY)
{
    ImageHit hit;
    const PaneLayout panes = computePanes(input);
    const int32_t index = paneAt(panes, windowX, windowY);
    if (index < 0)
    {
        return hit;
    }
    const Pane & pane = panes.panes[static_cast<size_t>(index)];
    const double zoom = effectiveZoom(input);
    const double centerX = pane.rect.x + pane.rect.width * 0.5;
    const double centerY = pane.rect.y + pane.rect.height * 0.5;

    hit.pane = index;
    hit.source = pane.source;
    hit.imageX = input.view.panX + ((windowX + 0.5) - centerX) / zoom;
    hit.imageY = input.view.panY + ((windowY + 0.5) - centerY) / zoom;
    hit.pixelX = floorToPixel(hit.imageX);
    hit.pixelY = floorToPixel(hit.imageY);
    hit.insideExtent = rectContains(paddedExtent(input), hit.pixelX, hit.pixelY);
    return hit;
}

ProbeTarget probeTarget(const LayoutInput & input, int32_t windowX, int32_t windowY)
{
    ProbeTarget target;
    const ImageHit hit = windowToImage(input, windowX, windowY);
    if (hit.pane < 0)
    {
        return target;
    }
    const LayoutSource * source = visibleSourceAt(input, hit.source);
    if (source == nullptr)
    {
        return target;
    }
    const int32_t localX = hit.pixelX - source->offsetX;
    const int32_t localY = hit.pixelY - source->offsetY;
    if (localX < 0 || localY < 0 || localX >= source->width || localY >= source->height)
    {
        return target;
    }
    target.source = hit.source;
    target.localX = localX;
    target.localY = localY;
    return target;
}

PanePixelClass classifyPanePixel(const LayoutInput & input, int32_t windowX, int32_t windowY)
{
    PanePixelClass result;
    const ImageHit hit = windowToImage(input, windowX, windowY);
    if (hit.pane < 0)
    {
        return result;
    }
    result.pane = hit.pane;
    if (!hit.insideExtent)
    {
        result.kind = PixelKind::Outside;
        return result;
    }
    const LayoutSource * source = visibleSourceAt(input, hit.source);
    const bool inSource = source != nullptr && rectContains(sourceRect(*source), hit.pixelX, hit.pixelY);
    result.kind = inSource ? PixelKind::Image : PixelKind::Padding;
    return result;
}

// ---- Wipe ------------------------------------------------------------------------

double clampWipePosition(const LayoutInput & input, double position)
{
    double start = 0.0;
    double length = 0.0;
    wipeAxis(input, start, length);
    if (std::isnan(position))
    {
        return start;
    }
    return std::clamp(position, start, start + length);
}

void setWipeFraction(LayoutInput & input, double fraction)
{
    if (std::isnan(fraction))
    {
        fraction = 0.0;
    }
    fraction = std::clamp(fraction, 0.0, 1.0);
    double start = 0.0;
    double length = 0.0;
    wipeAxis(input, start, length);
    input.view.wipePosition = start + fraction * length;
}

bool wipeWindowCoordinate(const LayoutInput & input, int32_t pane, double & windowCoordinate)
{
    const PaneLayout panes = computePanes(input);
    if (pane < 0 || pane >= panes.count)
    {
        return false;
    }
    const PixelRect & rect = panes.panes[static_cast<size_t>(pane)].rect;
    const double zoom = effectiveZoom(input);
    const double position = clampWipePosition(input, input.view.wipePosition);
    if (input.view.wipeOrientation == WipeOrientation::Vertical)
    {
        const double centerX = rect.x + rect.width * 0.5;
        windowCoordinate = centerX + (position - input.view.panX) * zoom;
    }
    else
    {
        const double centerY = rect.y + rect.height * 0.5;
        windowCoordinate = centerY + (position - input.view.panY) * zoom;
    }
    return true;
}

bool wipePositionAtPixel(const LayoutInput & input, int32_t windowX, int32_t windowY, double & imagePosition)
{
    const ImageHit hit = windowToImage(input, windowX, windowY);
    if (hit.pane < 0)
    {
        return false;
    }
    const double coordinate = input.view.wipeOrientation == WipeOrientation::Vertical ? hit.imageX : hit.imageY;
    imagePosition = clampWipePosition(input, coordinate);
    return true;
}

WipeSide wipeSideAt(double imageCoordinate, double wipePosition)
{
    return imageCoordinate < wipePosition ? WipeSide::A : WipeSide::B;
}

bool wipeSideAtPixel(const LayoutInput & input, int32_t windowX, int32_t windowY, WipeSide & side)
{
    const ImageHit hit = windowToImage(input, windowX, windowY);
    if (hit.pane < 0)
    {
        return false;
    }
    const double coordinate = input.view.wipeOrientation == WipeOrientation::Vertical ? hit.imageX : hit.imageY;
    side = wipeSideAt(coordinate, clampWipePosition(input, input.view.wipePosition));
    return true;
}

// ---- Split-screen divider --------------------------------------------------------

double clampDividerFraction(double fraction)
{
    if (std::isnan(fraction))
    {
        return kDefaultSplitDivider;
    }
    return std::clamp(fraction, 0.0, 1.0);
}

double dividerWindowX(const LayoutInput & input)
{
    return clampDividerFraction(input.view.dividerFraction) * static_cast<double>(std::max(0, input.windowWidth));
}

double dividerFractionAtWindowX(const LayoutInput & input, double windowX)
{
    const int32_t width = std::max(0, input.windowWidth);
    if (width <= 0)
    {
        return kDefaultSplitDivider;
    }
    return clampDividerFraction(windowX / static_cast<double>(width));
}

bool dividerImagePosition(const LayoutInput & input, double & imageX)
{
    const PaneLayout panes = computePanes(input);
    if (panes.count <= 0)
    {
        return false;
    }
    const double zoom = effectiveZoom(input);
    if (!std::isfinite(zoom) || !(zoom > 0.0))
    {
        return false;
    }
    // The same mapping as windowToImage(), at the continuous divider coordinate instead of
    // a pixel center.
    const PixelRect & rect = panes.panes[0].rect;
    const double centerX = rect.x + rect.width * 0.5;
    const double position = input.view.panX + (dividerWindowX(input) - centerX) / zoom;
    if (!std::isfinite(position))
    {
        return false;
    }
    imageX = position;
    return true;
}

// ---- Padding and maps ------------------------------------------------------------

float displayTransform(float linear, ToneMapper toneMapper, float exposureStops)
{
    if (std::isnan(exposureStops))
    {
        exposureStops = 0.0f;
    }
    exposureStops = std::clamp(exposureStops, -kMaxExposureStops, kMaxExposureStops);

    float value = linear * std::exp2(exposureStops);
    if (!(value > 0.0f))
    {
        value = 0.0f;
    }
    value = std::min(value, kMaxLinear);
    return srgbTransfer(clampUnit(toneMap(value, toneMapper)));
}

bool paddingColorFor(const ImageData & other, int32_t x, int32_t y, ToneMapper toneMapper,
    float exposureStops, RgbaColor & out, const RgbaColor & background)
{
    if (x < 0 || y < 0 || other.width == 0 || other.height == 0)
    {
        return false;
    }
    if (static_cast<uint32_t>(x) >= other.width || static_cast<uint32_t>(y) >= other.height)
    {
        return false;
    }
    const size_t pixelCount = static_cast<size_t>(other.width) * static_cast<size_t>(other.height);
    if (other.rgba.size() < pixelCount * 4)
    {
        return false;
    }

    const size_t pixel = static_cast<size_t>(y) * other.width + static_cast<size_t>(x);
    const float * samples = &other.rgba[pixel * 4];
    const bool linear = other.encoding == Encoding::Linear;
    const float alphaOut = 1.0f - clampUnit(samples[3]);
    // Premultiplied: composite over the background first, in the source's own encoding.
    const float holdback = other.premultiplied ? 1.0f - clampUnit(samples[3]) : 0.0f;
    const float backgroundRgb[3] = {background.r, background.g, background.b};
    const auto channel = [&](int index)
    {
        const float sample = samples[index] + holdback * backgroundRgb[index];
        const float display = linear ? displayTransform(sample, toneMapper, exposureStops) : clampUnit(sample);
        return 1.0f - display;
    };
    out.r = channel(0);
    out.g = channel(1);
    out.b = channel(2);
    out.a = alphaOut;
    return true;
}

float mapPaddingValue()
{
    return 1.0f;
}

float mapValueAt(float sample, PixelKind kind)
{
    switch (kind)
    {
    case PixelKind::Image:
        return sample;
    case PixelKind::Padding:
        return mapPaddingValue();
    case PixelKind::Outside:
        break;
    }
    return std::numeric_limits<float>::quiet_NaN();
}

float hatchValue(int32_t viewX, int32_t viewY)
{
    // Diagonal stripes, 8 pixels wide. Callers pass view pixels, which are >= 0.
    return (((viewX + viewY) / 8) & 1) != 0 ? kHatchLight : 0.12f;
}

float depthMapT(float d, float rangeMin, float rangeMax)
{
    const float span = rangeMax - rangeMin;
    if (!(span > 0.0f))
    {
        return 0.0f;
    }
    // clampUnit also maps NaN to 0, so a caller that forgets isValidDepth still gets
    // a defined color instead of a NaN index.
    return clampUnit((d - rangeMin) / span);
}

float depthDiffT(float a, float b, float rangeMin, float rangeMax)
{
    const float diff = std::fabs(a - b);
    const float span = rangeMax - rangeMin;
    if (!(span > 0.0f))
    {
        return diff > 0.0f ? 1.0f : 0.0f;
    }
    return clampUnit(diff / span);
}

} // namespace sabercat::compare
