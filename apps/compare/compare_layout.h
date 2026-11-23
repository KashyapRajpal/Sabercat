// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// CPU model of the compare window's layout: where each source sits in the shared
// canvas, how the window is split into panes, the zoom and pan that map window
// pixels to image pixels, the wipe, and the CPU reference for the padding and map
// rules that the GPU compositor must match. Pure and deterministic: no window, no
// GPU, no Windows headers, no file I/O.
//
// Conventions:
//   - Window pixels are integers, (0,0) at the window's top-left. The center of
//     window pixel (px, py) is (px + 0.5, py + 0.5) in continuous window coordinates.
//   - Image coordinates are continuous, canvas origin at (0,0). Image pixel (ix, iy)
//     covers [ix, ix + 1) x [iy, iy + 1); a continuous coordinate c lies in pixel
//     floor(c).
//   - Rectangles are half-open: [x, x + width) x [y, y + height). An empty rectangle
//     has width or height 0 and contains nothing.
//   - "Padding" is a pixel inside the padded extent (the union of the visible
//     sources) where the pane's own source has no pixel. "Outside" is a pixel outside
//     the padded extent, drawn as background.
#pragma once

#include "flip_hdr.h"
#include "image_data.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

namespace sabercat::compare
{

// Source limits. addSource() and setSourceVisible() refuse anything beyond them.
inline constexpr int32_t kMaxSources = 8;
inline constexpr int32_t kMaxVisible = 4;
// Accepted width and height of a source, [1, kMaxSourceSize], and largest magnitude of
// an offset. They keep every extent sum far from int32 overflow.
inline constexpr int32_t kMaxSourceSize = 1 << 20;
inline constexpr int32_t kMaxSourceOffset = 1 << 24;
// Fixed zoom range in percent (100 is 1:1).
inline constexpr double kMinZoomPercent = 1.0;
inline constexpr double kMaxZoomPercent = 6400.0;
// The split-screen divider at the start: the middle of the window.
inline constexpr double kDefaultSplitDivider = 0.5;

// Depth in a padding pixel: there is no data. Test with std::isnan.
inline constexpr float kPaddingDepth = std::numeric_limits<float>::quiet_NaN();

// Integer rectangle in window or image pixels.
struct PixelRect
{
    int32_t x = 0;
    int32_t y = 0;
    int32_t width = 0;
    int32_t height = 0;
};

// One source. The offset is the image-space position of the source's top-left pixel
// in the shared canvas; it may be negative.
struct LayoutSource
{
    int32_t width = 0;
    int32_t height = 0;
    int32_t offsetX = 0;
    int32_t offsetY = 0;
    Encoding encoding = Encoding::Display;
    bool visible = true;
};

enum class ViewMode : uint8_t
{
    SplitScreen,  // one pane per visible source
    SingleScreen  // one pane showing the source at ViewState::abIndex
};

// How split-screen panes are arranged. Row puts the panes side by side. Grid2x2 uses
// two columns and, for three or four panes, two rows; two panes give two columns.
enum class PaneArrangement : uint8_t
{
    Row,
    Grid2x2
};

enum class ZoomMode : uint8_t
{
    Fit,     // the padded extent fits the smallest pane, centered
    Percent  // ViewState::zoomPercent, clamped to [kMinZoomPercent, kMaxZoomPercent]
};

enum class WipeOrientation : uint8_t
{
    Vertical,   // the wipe is a vertical line at an image x coordinate
    Horizontal  // the wipe is a horizontal line at an image y coordinate
};

// Which source a wipe shows at a pixel.
enum class WipeSide : uint8_t
{
    A,
    B
};

// What a pane pixel shows. Image: inside the pane's own source. Padding: inside the
// padded extent but without a pixel of the pane's source. Outside: outside the padded
// extent (background).
enum class PixelKind : uint8_t
{
    Image,
    Padding,
    Outside
};

// The view: everything the user changes, apart from the sources themselves. Fields
// are ordered by size, so the struct has no padding.
struct ViewState
{
    // Image coordinate shown at the center of every pane (shared by all panes).
    double panX = 0.0;
    double panY = 0.0;
    // Window pixels per image pixel, in percent, used when zoomMode is Percent.
    double zoomPercent = 100.0;
    // Wipe boundary in image coordinates: x for a vertical wipe, y for a horizontal
    // one. Kept in image space so it survives zoom, pan and resize. Stored unclamped;
    // the readers clamp it to the padded extent.
    double wipePosition = 0.0;
    // Split-screen divider (two visible sources, see CompareState::dividerActive()) as a
    // fraction of the window width, 0 at the left edge and 1 at the right. It is a
    // window fraction rather than an image coordinate so that it stays where the user put
    // it through zoom, pan and resize. Stored unclamped; the readers clamp it.
    double dividerFraction = kDefaultSplitDivider;
    // The A/B pair: source indices. Overlap, wipe and padding counterparts use them.
    int32_t sourceA = 0;
    int32_t sourceB = 1;
    // SingleScreen: the source index shown (normally sourceA or sourceB).
    int32_t abIndex = 0;
    ViewMode mode = ViewMode::SplitScreen;
    PaneArrangement arrangement = PaneArrangement::Row;
    ZoomMode zoomMode = ZoomMode::Fit;
    WipeOrientation wipeOrientation = WipeOrientation::Vertical;
};

// Everything the layout depends on. Sources are kept in insertion order; a source's
// index is its position in sources[]. Use addSource() to add one.
struct LayoutInput
{
    std::array<LayoutSource, kMaxSources> sources{};
    ViewState view;
    int32_t sourceCount = 0;
    // Window size in pixels. A side that is 0 or negative is treated as 0 on that axis.
    int32_t windowWidth = 0;
    int32_t windowHeight = 0;
};

// One split-screen or single-screen pane.
struct Pane
{
    PixelRect rect;          // window pixels; empty when the window has no size
    int32_t source = -1;     // source index drawn in this pane
};

struct PaneLayout
{
    std::array<Pane, kMaxVisible> panes{};
    int32_t count = 0;       // 0 when nothing can be drawn
};

// Result of windowToImage().
struct ImageHit
{
    double imageX = 0.0;     // image coordinate of the window pixel's center
    double imageY = 0.0;
    int32_t pixelX = 0;      // floor(imageX), saturated to +-2^30
    int32_t pixelY = 0;
    int32_t pane = -1;       // -1 when the window pixel is in no pane
    int32_t source = -1;     // the pane's source, -1 when there is no pane
    bool insideExtent = false;  // the integer pixel is inside the padded extent
};

// Result of probeTarget(). source is -1 when the window pixel shows no pixel of any
// source: outside the padded extent, in padding, or in no pane.
struct ProbeTarget
{
    int32_t source = -1;
    int32_t localX = 0;      // integer pixel minus the source offset
    int32_t localY = 0;
};

// Result of classifyPanePixel(). pane is -1 only when the window pixel is in no pane.
struct PanePixelClass
{
    PixelKind kind = PixelKind::Outside;
    int32_t pane = -1;
};

// A display-space color with alpha, each channel in [0,1].
struct RgbaColor
{
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
    float a = 0.0f;
};
// ---- Sources ---------------------------------------------------------------------

// Appends a source. Refused (false, nothing changed) when the set already holds
// kMaxSources, when a width or height is outside [1, kMaxSourceSize], when an offset
// magnitude exceeds kMaxSourceOffset, or when the source is visible and kMaxVisible
// sources are already visible.
bool addSource(LayoutInput & input, const LayoutSource & source);

// Shows or hides source `index`. False, with nothing changed, for an index outside
// the set or when showing it would exceed kMaxVisible.
bool setSourceVisible(LayoutInput & input, int32_t index, bool visible);

// Number of visible sources, 0..kMaxVisible.
int32_t visibleCount(const LayoutInput & input);

// Preset: every source gets offset (0, 0).
void anchorTopLeft(LayoutInput & input);

// Preset: centers every source on one common center. With the largest width W and
// height H over all sources (hidden ones included, so toggling visibility never moves a
// source), a source of size (w, h) gets offset ((W - w) / 2, (H - h) / 2) rounded down.
// For an odd difference the centers therefore differ by half a pixel.
void anchorCentered(LayoutInput & input);

// ---- Extents ---------------------------------------------------------------------

// The source's rectangle in the canvas: (offsetX, offsetY, width, height).
PixelRect sourceRect(const LayoutSource & source);

// Union of the visible sources' rectangles. Empty ({0,0,0,0}) when none is visible.
PixelRect paddedExtent(const LayoutInput & input);

// Intersection of the rectangles of sources a and b. Empty when they do not overlap,
// and empty when an index is out of range or the source is hidden. Metrics cover this
// rectangle only.
PixelRect overlapRect(const LayoutInput & input, int32_t a, int32_t b);

// Intersection of two rectangles; empty ({0,0,0,0}) when they do not overlap.
PixelRect intersectRects(const PixelRect & a, const PixelRect & b);

// True when the integer pixel (x, y) lies in the half-open rectangle.
bool rectContains(const PixelRect & rect, int32_t x, int32_t y);

// ---- Panes -----------------------------------------------------------------------

// Splits the window into panes.
//   SingleScreen: one pane covering the window, showing abIndex. No panes when
//                 abIndex is not a visible source.
//   SplitScreen:  one pane per visible source, in source index order.
//                 Row: n columns, one row. Grid2x2: two columns (for n >= 2) and two
//                 rows (for n >= 3). For n = 3 the fourth cell is empty and no pane
//                 covers it.
// Along each axis the pane boundaries are floor(i * size / parts), i = 0..parts. So a
// 7 px window split in two gives widths 3 and 4: the first pane gets the floor and the
// last the odd pixel. The panes of one arrangement tile their cells exactly. A side that
// is 0 or negative is treated as 0 on its axis: a zero-width window gives zero-width panes
// (still full height), and a window with both sides non-positive gives empty panes.
PaneLayout computePanes(const LayoutInput & input);

// Index of the pane containing window pixel (x, y), or -1. A pixel on a shared pane
// boundary belongs to the pane on its right or below.
int32_t paneAt(const PaneLayout & panes, int32_t x, int32_t y);

// ---- Zoom and pan ----------------------------------------------------------------

// Clamps a zoom percentage to [kMinZoomPercent, kMaxZoomPercent]. NaN reads as 100.
double clampZoomPercent(double percent);

// Fit mode: the padded extent is scaled to fit the smallest pane (over all panes, so
// that one zoom works for every pane) and is centered. Fit scales up as well as down.
void setZoomFit(LayoutInput & input);

// Fixed zoom, clamped. 100 is 1:1.
void setZoomPercent(LayoutInput & input, double percent);

// The zoom in window pixels per image pixel, after clamping. In fit mode, 1.0 when
// the extent or a pane is empty (no fit is possible).
double effectiveZoom(const LayoutInput & input);

// Multiplies the zoom by factor (finite and > 0) and switches to percent mode. The
// image point under window pixel (windowX, windowY) stays under it, to rounding
// error, after the clamp. False, with nothing changed, when the pixel is in no pane or
// the factor is invalid.
bool zoomAt(LayoutInput & input, int32_t windowX, int32_t windowY, double factor);

// Sets the pan: the image coordinate shown at every pane's center. Non-finite values
// are ignored.
void setPan(LayoutInput & input, double imageX, double imageY);

// Pans so that the center of the padded extent is at the pane centers.
void centerView(LayoutInput & input);

// ---- Mapping ---------------------------------------------------------------------

// Maps window pixel (windowX, windowY) through the pane that contains it. The image
// coordinate is panX + (windowX + 0.5 - paneCenterX) / zoom, and likewise for y.
ImageHit windowToImage(const LayoutInput & input, int32_t windowX, int32_t windowY);

// Which source pixel the window pixel shows: the pane's source and the integer pixel
// minus that source's offset. source is -1 when the pixel is outside the padded
// extent, in padding (the pane's source has no pixel there), or in no pane.
ProbeTarget probeTarget(const LayoutInput & input, int32_t windowX, int32_t windowY);

// Classifies a pane pixel as Image, Padding or Outside (see PixelKind). The pane
// index is set whenever the pixel is in a pane.
PanePixelClass classifyPanePixel(const LayoutInput & input, int32_t windowX, int32_t windowY);

// ---- Wipe ------------------------------------------------------------------------

// Clamps an image coordinate to the padded extent along the wipe's axis (x for a
// vertical wipe, y for a horizontal one). NaN maps to the extent's start. An empty
// extent gives 0.
double clampWipePosition(const LayoutInput & input, double position);

// Sets the wipe to a fraction of the padded extent along its axis, fraction 0 at
// the start and 1 at the end. Fractions are clamped to [0,1]; NaN is 0.
void setWipeFraction(LayoutInput & input, double fraction);

// Window coordinate of the wipe boundary inside pane `pane`, as a continuous window
// coordinate (x for a vertical wipe, y for a horizontal one). The wipe is clamped
// first. False when the pane index is invalid.
bool wipeWindowCoordinate(const LayoutInput & input, int32_t pane, double & windowCoordinate);

// Image coordinate under the center of window pixel (windowX, windowY), clamped to the
// padded extent along the wipe's axis: the value a wipe position would take from this
// pixel. False when the pixel is in no pane.
bool wipePositionAtPixel(const LayoutInput & input, int32_t windowX, int32_t windowY, double & imagePosition);

// The side a point with image coordinate imageCoordinate shows: A when it is less than
// wipePosition, B otherwise. Far left (position at or below the extent's start) shows
// B everywhere, far right (position at or above the end) shows A everywhere.
WipeSide wipeSideAt(double imageCoordinate, double wipePosition);

// ---- Split-screen divider --------------------------------------------------------
// With exactly two visible sources (A and B), the split view is drawn as one pane with
// a vertical wipe whose boundary is the divider. The divider is kept as a window fraction
// so it survives zoom, pan and resize. Computing it in window pixels would make the
// boundary jump whenever the window changes size.

// Clamps a divider fraction to [0,1]. NaN reads as kDefaultSplitDivider.
double clampDividerFraction(double fraction);

// The divider's continuous window x coordinate: the clamped view.dividerFraction times the
// window width. A width of 0 or less counts as 0. Continuous, not a pixel index: window
// pixel px is on the A side when px + 0.5 is less than this value.
double dividerWindowX(const LayoutInput & input);

// The divider fraction for a continuous window x: windowX / windowWidth, clamped to [0,1]
// (NaN reads as kDefaultSplitDivider). A window with no width gives kDefaultSplitDivider.
double dividerFractionAtWindowX(const LayoutInput & input, double windowX);

// The image x coordinate under the divider, using the same mapping as windowToImage():
// panX + (windowX - paneCenterX) / zoom, with windowX = dividerWindowX() and the pane of
// the SingleScreen form. Precondition: input.view.mode == ViewMode::SingleScreen; the
// caller passes that form (the divider is a single pane, not one of the split panes).
// False when there is no pane, the zoom is not a positive finite number, or the result is
// not finite.
bool dividerImagePosition(const LayoutInput & input, double & imageX);

// The side window pixel (windowX, windowY) shows in wipe mode: wipeSideAt() applied to
// the pixel center's image coordinate, with the clamped wipe. False when the pixel is
// in no pane.
bool wipeSideAtPixel(const LayoutInput & input, int32_t windowX, int32_t windowY, WipeSide & side);

// ---- Padding and maps (CPU reference for the GPU compositor) ----------------------

// Display transform of one linear sample: scale by 2^exposureStops, tone map (Reinhard
// c / (1 + c), ACES (Narkowicz fit) or Hable's filmic curve normalized to white
// point 11.2), then the sRGB transfer function. Input is clamped to [0, 65504] (NaN and
// negatives are 0); exposure is clamped to [-32, 32] (NaN is 0). The result is in [0,1].
float displayTransform(float linear, ToneMapper toneMapper, float exposureStops);

// The padding color for a pane pixel whose pane source has no pixel there: the
// counterpart image's pixel (x, y), inverted. Display samples are clamped to [0,1]
// and inverted. Linear samples go through displayTransform() first and are then
// inverted. Alpha out is 1 - alpha in, with alpha clamped to [0,1]. The counterpart's
// own encoding (other.encoding) selects the path. Returns false, with out unchanged,
// when the counterpart has no pixel there (or its buffer is too short): the result is
// background.
// When other.premultiplied is set (raw captures), rgb is first composited over
// background: rgb' = rgb + (1 - clamp(alpha)) * background.rgb per channel. For a
// Linear source this happens before displayTransform(), with background taken as
// already in the source's encoding. The background is ignored for straight-alpha
// sources. The output alpha is always 1 - clamp(alpha) from the original alpha.
bool paddingColorFor(const ImageData & other, int32_t x, int32_t y, ToneMapper toneMapper,
    float exposureStops, RgbaColor & out, const RgbaColor & background = RgbaColor{0.0f, 0.0f, 0.0f, 1.0f});

// The value a map (absolute difference, FLIP, exposure map, depth difference) draws in
// padding: its maximum, 1.0, so the colorizer gives the top color.
float mapPaddingValue();

// The map value to draw for a pixel of the given kind: the sample for Image, the
// maximum for Padding, and NaN for Outside (the caller draws background).
float mapValueAt(float sample, PixelKind kind);

// ---- Depth and alpha maps ---------------------------------------------------------

// A depth sample that means something: finite and positive. NaN, infinity, zero and
// negative values are "no data" (a background or unfilled pixel) and never enter a
// range or a map.
inline bool isValidDepth(float d)
{
    return std::isfinite(d) && d > 0.0f;
}

// The dark stripe of the hatch is kCompositeBackground (0.12); this is the light one.
inline constexpr float kHatchLight = 0.22f;

// Display value of the hatch drawn where a depth map has no data. Stripes are 8 window
// pixels wide and run along the diagonal, so they read the same at every zoom.
float hatchValue(int32_t viewX, int32_t viewY);

// Position of depth d in the map range, in [0,1]: (d - rangeMin) / span clamped to
// [0,1]. A zero or negative span (one depth, or a degenerate range) gives 0.
float depthMapT(float d, float rangeMin, float rangeMax);

// Position of the difference |a - b| in the map range, in [0,1]. A zero span gives 1
// for any difference and 0 for equal depths, so a flat range still shows change.
float depthDiffT(float a, float b, float rangeMin, float rangeMax);

} // namespace sabercat::compare
