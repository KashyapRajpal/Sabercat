// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Compare layout model (apps/compare/compare_layout.h): source limits and anchors,
// the padded and overlap rectangles, pane splits, zoom and pan, probes and pixel
// classes, the wipe, and the padding and map rules. Pure CPU.
#include "unit_test.h"

#include "compare_layout.h"
#include "image_data.h"

#include <cmath>
#include <cstdint>
#include <limits>

namespace sabercat::unit
{

namespace
{

using sabercat::compare::Encoding;
using sabercat::compare::ImageData;
using sabercat::compare::ImageHit;
using sabercat::compare::LayoutInput;
using sabercat::compare::LayoutSource;
using sabercat::compare::PaneArrangement;
using sabercat::compare::PaneLayout;
using sabercat::compare::PixelKind;
using sabercat::compare::PixelRect;
using sabercat::compare::ProbeTarget;
using sabercat::compare::RgbaColor;
using sabercat::compare::ToneMapper;
using sabercat::compare::ViewMode;
using sabercat::compare::WipeOrientation;
using sabercat::compare::WipeSide;
using sabercat::compare::addSource;
using sabercat::compare::anchorCentered;
using sabercat::compare::anchorTopLeft;
using sabercat::compare::centerView;
using sabercat::compare::classifyPanePixel;
using sabercat::compare::clampDividerFraction;
using sabercat::compare::clampZoomPercent;
using sabercat::compare::computePanes;
using sabercat::compare::displayTransform;
using sabercat::compare::effectiveZoom;
using sabercat::compare::depthDiffT;
using sabercat::compare::depthMapT;
using sabercat::compare::dividerFractionAtWindowX;
using sabercat::compare::dividerImagePosition;
using sabercat::compare::dividerWindowX;
using sabercat::compare::hatchValue;
using sabercat::compare::intersectRects;
using sabercat::compare::isValidDepth;
using sabercat::compare::kHatchLight;
using sabercat::compare::kPaddingDepth;
using sabercat::compare::mapPaddingValue;
using sabercat::compare::mapValueAt;
using sabercat::compare::overlapRect;
using sabercat::compare::paddedExtent;
using sabercat::compare::paddingColorFor;
using sabercat::compare::paneAt;
using sabercat::compare::probeTarget;
using sabercat::compare::setPan;
using sabercat::compare::setSourceVisible;
using sabercat::compare::setWipeFraction;
using sabercat::compare::setZoomFit;
using sabercat::compare::setZoomPercent;
using sabercat::compare::visibleCount;
using sabercat::compare::windowToImage;
using sabercat::compare::wipePositionAtPixel;
using sabercat::compare::wipeSideAt;
using sabercat::compare::wipeSideAtPixel;
using sabercat::compare::wipeWindowCoordinate;
using sabercat::compare::zoomAt;

LayoutInput makeWindow(int32_t width, int32_t height)
{
    LayoutInput input;
    input.windowWidth = width;
    input.windowHeight = height;
    return input;
}

LayoutSource makeSource(int32_t width, int32_t height, int32_t offsetX = 0, int32_t offsetY = 0,
    Encoding encoding = Encoding::Display, bool visible = true)
{
    LayoutSource source;
    source.width = width;
    source.height = height;
    source.offsetX = offsetX;
    source.offsetY = offsetY;
    source.encoding = encoding;
    source.visible = visible;
    return source;
}

bool isRect(const PixelRect & rect, int32_t x, int32_t y, int32_t width, int32_t height)
{
    return rect.x == x && rect.y == y && rect.width == width && rect.height == height;
}

bool isEmpty(const PixelRect & rect)
{
    return rect.width == 0 && rect.height == 0;
}

// The sRGB transfer function in double precision, for the expected values.
double srgbEncode(double linear)
{
    return linear <= 0.0031308 ? 12.92 * linear : 1.055 * std::pow(linear, 1.0 / 2.4) - 0.055;
}

void checkSources(Context & ctx)
{
    // Four visible sources are accepted; a fifth visible one is refused.
    LayoutInput input = makeWindow(8, 4);
    for (int32_t i = 0; i < 4; ++i)
    {
        UNIT_CHECK(ctx, addSource(input, makeSource(4, 4)));
    }
    UNIT_CHECK(ctx, !addSource(input, makeSource(4, 4)));
    UNIT_CHECK(ctx, input.sourceCount == 4);
    UNIT_CHECK(ctx, visibleCount(input) == 4);

    // Hidden sources fill the set to eight; the ninth is refused.
    for (int32_t i = 0; i < 4; ++i)
    {
        UNIT_CHECK(ctx, addSource(input, makeSource(4, 4, 0, 0, Encoding::Display, false)));
    }
    UNIT_CHECK(ctx, input.sourceCount == 8);
    UNIT_CHECK(ctx, !addSource(input, makeSource(4, 4, 0, 0, Encoding::Display, false)));
    UNIT_CHECK(ctx, input.sourceCount == 8);

    // Showing a fifth source is refused; hiding one makes room for it.
    UNIT_CHECK(ctx, !setSourceVisible(input, 4, true));
    UNIT_CHECK(ctx, visibleCount(input) == 4);
    UNIT_CHECK(ctx, setSourceVisible(input, 0, false));
    UNIT_CHECK(ctx, setSourceVisible(input, 4, true));
    UNIT_CHECK(ctx, visibleCount(input) == 4);
    UNIT_CHECK(ctx, setSourceVisible(input, 4, true));
    UNIT_CHECK(ctx, !setSourceVisible(input, 99, true));
    UNIT_CHECK(ctx, !setSourceVisible(input, -1, false));

    // Invalid sizes and offsets are refused and leave the set unchanged.
    LayoutInput other = makeWindow(0, 0);
    UNIT_CHECK(ctx, !addSource(other, makeSource(0, 4)));
    UNIT_CHECK(ctx, !addSource(other, makeSource(4, -1)));
    UNIT_CHECK(ctx, !addSource(other, makeSource(4, 4, 1 << 25, 0)));
    UNIT_CHECK(ctx, other.sourceCount == 0);
}

void checkAnchors(Context & ctx)
{
    LayoutInput input = makeWindow(0, 0);
    addSource(input, makeSource(10, 4, 7, 9));
    addSource(input, makeSource(7, 6, -3, 2));

    anchorTopLeft(input);
    UNIT_CHECK(ctx, input.sources[0].offsetX == 0 && input.sources[0].offsetY == 0);
    UNIT_CHECK(ctx, input.sources[1].offsetX == 0 && input.sources[1].offsetY == 0);

    // The largest source is 10 x 6. The 10 x 4 source moves down 1; the 7 x 6 source
    // moves right by floor(3 / 2) = 1, so its center is half a pixel off.
    anchorCentered(input);
    UNIT_CHECK(ctx, input.sources[0].offsetX == 0 && input.sources[0].offsetY == 1);
    UNIT_CHECK(ctx, input.sources[1].offsetX == 1 && input.sources[1].offsetY == 0);
}

void checkExtents(Context & ctx)
{
    // All offsets negative: the union has a negative origin.
    LayoutInput input = makeWindow(0, 0);
    addSource(input, makeSource(4, 4, -3, -2));
    addSource(input, makeSource(2, 5, -5, -1));
    UNIT_CHECK(ctx, isRect(paddedExtent(input), -5, -2, 6, 6));

    // Hidden sources do not count; with none visible the extent is empty.
    setSourceVisible(input, 1, false);
    UNIT_CHECK(ctx, isRect(paddedExtent(input), -3, -2, 4, 4));
    setSourceVisible(input, 0, false);
    UNIT_CHECK(ctx, isEmpty(paddedExtent(input)));
    setSourceVisible(input, 0, true);
    setSourceVisible(input, 1, true);

    // Touching rectangles do not overlap; the overlap is empty and not an error.
    UNIT_CHECK(ctx, isEmpty(overlapRect(input, 0, 1)));

    // A real intersection: [-3,1) x [-2,2) with [-1,5) x [-1,5) is [-1,1) x [-1,2).
    addSource(input, makeSource(6, 6, -1, -1));
    UNIT_CHECK(ctx, isRect(overlapRect(input, 0, 2), -1, -1, 2, 3));
    setSourceVisible(input, 2, false);
    UNIT_CHECK(ctx, isEmpty(overlapRect(input, 0, 2)));
    UNIT_CHECK(ctx, isEmpty(overlapRect(input, 0, 9)));

    const PixelRect left{0, 0, 2, 2};
    const PixelRect right{2, 0, 2, 2};
    UNIT_CHECK(ctx, isEmpty(intersectRects(left, right)));
}

void checkPanes(Context & ctx)
{
    // Two sources in a 7 px window: the odd pixel goes to the last pane (3 and 4).
    LayoutInput input = makeWindow(7, 5);
    addSource(input, makeSource(4, 4));
    addSource(input, makeSource(4, 4));
    PaneLayout panes = computePanes(input);
    UNIT_CHECK(ctx, panes.count == 2);
    UNIT_CHECK(ctx, isRect(panes.panes[0].rect, 0, 0, 3, 5));
    UNIT_CHECK(ctx, isRect(panes.panes[1].rect, 3, 0, 4, 5));
    UNIT_CHECK(ctx, panes.panes[0].source == 0 && panes.panes[1].source == 1);

    // The last pixel of pane 0 and the first pixel of pane 1 each belong to one pane.
    UNIT_CHECK(ctx, paneAt(panes, 2, 0) == 0);
    UNIT_CHECK(ctx, paneAt(panes, 3, 0) == 1);
    UNIT_CHECK(ctx, paneAt(panes, 6, 4) == 1);
    UNIT_CHECK(ctx, paneAt(panes, 7, 0) == -1);
    UNIT_CHECK(ctx, paneAt(panes, 0, 5) == -1);

    // Three panes in a 7 px window: widths 2, 2 and 3.
    addSource(input, makeSource(4, 4));
    panes = computePanes(input);
    UNIT_CHECK(ctx, panes.count == 3);
    UNIT_CHECK(ctx, isRect(panes.panes[0].rect, 0, 0, 2, 5));
    UNIT_CHECK(ctx, isRect(panes.panes[1].rect, 2, 0, 2, 5));
    UNIT_CHECK(ctx, isRect(panes.panes[2].rect, 4, 0, 3, 5));

    // Grid2x2 with four panes in 7 x 5: columns 3 and 4, rows 2 and 3.
    input.view.arrangement = PaneArrangement::Grid2x2;
    addSource(input, makeSource(4, 4));
    panes = computePanes(input);
    UNIT_CHECK(ctx, panes.count == 4);
    UNIT_CHECK(ctx, isRect(panes.panes[0].rect, 0, 0, 3, 2));
    UNIT_CHECK(ctx, isRect(panes.panes[1].rect, 3, 0, 4, 2));
    UNIT_CHECK(ctx, isRect(panes.panes[2].rect, 0, 2, 3, 3));
    UNIT_CHECK(ctx, isRect(panes.panes[3].rect, 3, 2, 4, 3));

    // Grid2x2 with three panes: the fourth cell (bottom right) is empty.
    LayoutInput three = makeWindow(7, 5);
    three.view.arrangement = PaneArrangement::Grid2x2;
    addSource(three, makeSource(4, 4));
    addSource(three, makeSource(4, 4));
    addSource(three, makeSource(4, 4));
    panes = computePanes(three);
    UNIT_CHECK(ctx, panes.count == 3);
    UNIT_CHECK(ctx, isRect(panes.panes[2].rect, 0, 2, 3, 3));
    UNIT_CHECK(ctx, paneAt(panes, 5, 4) == -1);
    UNIT_CHECK(ctx, paneAt(panes, 5, 1) == 1);

    // Grid2x2 with two panes: two columns, one row.
    LayoutInput two = makeWindow(7, 5);
    two.view.arrangement = PaneArrangement::Grid2x2;
    addSource(two, makeSource(4, 4));
    addSource(two, makeSource(4, 4));
    panes = computePanes(two);
    UNIT_CHECK(ctx, panes.count == 2);
    UNIT_CHECK(ctx, isRect(panes.panes[1].rect, 3, 0, 4, 5));

    // Every Row split of a window from 0 to 13 px, with 1 to 4 panes, tiles it exactly.
    bool tiles = true;
    for (int32_t width = 0; width <= 13; ++width)
    {
        for (int32_t count = 1; count <= 4; ++count)
        {
            LayoutInput tile = makeWindow(width, 6);
            for (int32_t i = 0; i < count; ++i)
            {
                addSource(tile, makeSource(4, 4));
            }
            const PaneLayout tiled = computePanes(tile);
            int32_t next = 0;
            for (int32_t i = 0; i < tiled.count; ++i)
            {
                const PixelRect & rect = tiled.panes[static_cast<size_t>(i)].rect;
                tiles = tiles && rect.x == next && rect.y == 0 && rect.height == 6;
                next += rect.width;
            }
            tiles = tiles && tiled.count == count && next == width;
        }
    }
    UNIT_CHECK(ctx, tiles);

    // A zero-size window (or a negative one) gives empty panes and no pane at any pixel.
    LayoutInput zero = makeWindow(0, 0);
    addSource(zero, makeSource(4, 4));
    addSource(zero, makeSource(4, 4));
    panes = computePanes(zero);
    UNIT_CHECK(ctx, panes.count == 2);
    UNIT_CHECK(ctx, isEmpty(panes.panes[0].rect) && isEmpty(panes.panes[1].rect));
    UNIT_CHECK(ctx, paneAt(panes, 0, 0) == -1);
    UNIT_CHECK(ctx, windowToImage(zero, 0, 0).pane == -1);
    UNIT_CHECK(ctx, classifyPanePixel(zero, 0, 0).kind == PixelKind::Outside);
    UNIT_CHECK(ctx, effectiveZoom(zero) == 1.0);
    LayoutInput negative = makeWindow(-4, 3);
    addSource(negative, makeSource(4, 4));
    UNIT_CHECK(ctx, computePanes(negative).panes[0].rect.width == 0);
    UNIT_CHECK(ctx, windowToImage(negative, 0, 0).pane == -1);

    // Single screen: one pane over the whole window, showing abIndex.
    LayoutInput single = makeWindow(7, 5);
    addSource(single, makeSource(4, 4));
    addSource(single, makeSource(4, 4));
    single.view.mode = ViewMode::SingleScreen;
    single.view.abIndex = 1;
    panes = computePanes(single);
    UNIT_CHECK(ctx, panes.count == 1);
    UNIT_CHECK(ctx, isRect(panes.panes[0].rect, 0, 0, 7, 5));
    UNIT_CHECK(ctx, panes.panes[0].source == 1);
    setSourceVisible(single, 1, false);
    UNIT_CHECK(ctx, computePanes(single).count == 0);
}

void checkZoomAndCentering(Context & ctx)
{
    // Fit centers a small image in a large pane: 10 x 4 in 100 x 100 is 10x, filling
    // the width and centered in the height.
    LayoutInput input = makeWindow(100, 100);
    addSource(input, makeSource(10, 4));
    input.view.mode = ViewMode::SingleScreen;
    UNIT_CHECK_NEAR(ctx, effectiveZoom(input), 10.0, 1e-12);
    centerView(input);
    UNIT_CHECK(ctx, probeTarget(input, 0, 30).source == 0);
    UNIT_CHECK(ctx, probeTarget(input, 0, 29).source == -1);
    UNIT_CHECK(ctx, probeTarget(input, 99, 69).source == 0);
    UNIT_CHECK(ctx, probeTarget(input, 99, 70).source == -1);
    UNIT_CHECK(ctx, probeTarget(input, 99, 30).localX == 9);

    // Percent clamping: [1, 6400] percent, NaN reads as 100.
    setZoomPercent(input, 0.5);
    UNIT_CHECK_NEAR(ctx, effectiveZoom(input), 0.01, 1e-12);
    setZoomPercent(input, 1.0e6);
    UNIT_CHECK_NEAR(ctx, effectiveZoom(input), 64.0, 1e-12);
    setZoomPercent(input, 100.0);
    UNIT_CHECK_NEAR(ctx, effectiveZoom(input), 1.0, 1e-12);
    UNIT_CHECK_NEAR(ctx, clampZoomPercent(std::nan("")), 100.0, 1e-12);
    setZoomFit(input);
    UNIT_CHECK_NEAR(ctx, effectiveZoom(input), 10.0, 1e-12);

    // A 1 x 1 image in a 1 x 1 window: fit is 100%, and the one pixel is probed.
    LayoutInput one = makeWindow(1, 1);
    addSource(one, makeSource(1, 1));
    centerView(one);
    const ImageHit hit = windowToImage(one, 0, 0);
    UNIT_CHECK(ctx, hit.insideExtent && hit.pixelX == 0 && hit.pixelY == 0);
    UNIT_CHECK(ctx, probeTarget(one, 0, 0).source == 0);

    // zoomAt keeps the image point under the anchor fixed, including after the clamp.
    LayoutInput zoomed = makeWindow(100, 80);
    addSource(zoomed, makeSource(30, 20));
    setZoomPercent(zoomed, 200.0);
    setPan(zoomed, 7.3, 4.1);
    ImageHit before = windowToImage(zoomed, 30, 20);
    UNIT_CHECK(ctx, zoomAt(zoomed, 30, 20, 1.5));
    ImageHit after = windowToImage(zoomed, 30, 20);
    UNIT_CHECK_NEAR(ctx, after.imageX, before.imageX, 1e-9);
    UNIT_CHECK_NEAR(ctx, after.imageY, before.imageY, 1e-9);
    UNIT_CHECK_NEAR(ctx, effectiveZoom(zoomed), 3.0, 1e-12);

    before = windowToImage(zoomed, 80, 60);
    UNIT_CHECK(ctx, zoomAt(zoomed, 80, 60, 1.0e9));
    after = windowToImage(zoomed, 80, 60);
    UNIT_CHECK_NEAR(ctx, effectiveZoom(zoomed), 64.0, 1e-12);
    UNIT_CHECK_NEAR(ctx, after.imageX, before.imageX, 1e-9);
    UNIT_CHECK_NEAR(ctx, after.imageY, before.imageY, 1e-9);

    // Zoom out keeps the anchor too.
    before = windowToImage(zoomed, 10, 5);
    UNIT_CHECK(ctx, zoomAt(zoomed, 10, 5, 0.25));
    after = windowToImage(zoomed, 10, 5);
    UNIT_CHECK_NEAR(ctx, after.imageX, before.imageX, 1e-9);
    UNIT_CHECK_NEAR(ctx, after.imageY, before.imageY, 1e-9);

    // Refused: a pixel in no pane, a zero or NaN factor. Nothing changes.
    const double zoomBefore = effectiveZoom(zoomed);
    UNIT_CHECK(ctx, !zoomAt(zoomed, 500, 500, 2.0));
    UNIT_CHECK(ctx, !zoomAt(zoomed, 10, 5, 0.0));
    UNIT_CHECK(ctx, !zoomAt(zoomed, 10, 5, std::nan("")));
    UNIT_CHECK_NEAR(ctx, effectiveZoom(zoomed), zoomBefore, 1e-12);
}

void checkMapping(Context & ctx)
{
    // Two 4 x 4 sources side by side in an 8 x 4 window at 100%. Pan (2, 2) puts image
    // pixel x at window x in pane 0 and at window x - 4 in pane 1 (the centers differ).
    LayoutInput input = makeWindow(8, 4);
    addSource(input, makeSource(4, 4));
    addSource(input, makeSource(4, 4));
    setZoomPercent(input, 100.0);
    setPan(input, 2.0, 2.0);

    // Last pixel of pane 0 and first pixel of pane 1.
    const ProbeTarget last = probeTarget(input, 3, 1);
    UNIT_CHECK(ctx, last.source == 0 && last.localX == 3 && last.localY == 1);
    const ProbeTarget first = probeTarget(input, 4, 1);
    UNIT_CHECK(ctx, first.source == 1 && first.localX == 0 && first.localY == 1);
    UNIT_CHECK(ctx, windowToImage(input, 4, 1).pane == 1);
    UNIT_CHECK(ctx, classifyPanePixel(input, 3, 1).pane == 0);
    UNIT_CHECK(ctx, classifyPanePixel(input, 4, 1).kind == PixelKind::Image);
    UNIT_CHECK(ctx, probeTarget(input, 8, 1).source == -1);

    // Padding: source B sits at x = 2, so the first two pixels of pane 1 have no B
    // pixel but are inside the padded extent [0, 6).
    LayoutInput padded = makeWindow(8, 4);
    addSource(padded, makeSource(4, 4));
    addSource(padded, makeSource(4, 4, 2, 0));
    setZoomPercent(padded, 100.0);
    setPan(padded, 2.0, 2.0);
    UNIT_CHECK(ctx, classifyPanePixel(padded, 3, 1).kind == PixelKind::Image);
    UNIT_CHECK(ctx, classifyPanePixel(padded, 4, 1).kind == PixelKind::Padding);
    UNIT_CHECK(ctx, classifyPanePixel(padded, 4, 1).pane == 1);
    UNIT_CHECK(ctx, classifyPanePixel(padded, 7, 1).kind == PixelKind::Image);
    UNIT_CHECK(ctx, probeTarget(padded, 4, 1).source == -1);
    UNIT_CHECK(ctx, probeTarget(padded, 7, 1).source == 1 && probeTarget(padded, 7, 1).localX == 1);

    // Outside: single screen on A (x [0,2), y [0,2)) with B at x [10,12). The pane
    // center is (4, 2); pan (2, 2) maps window pixel x to image pixel x - 2.
    LayoutInput outside = makeWindow(8, 4);
    addSource(outside, makeSource(2, 2));
    addSource(outside, makeSource(2, 2, 10, 0));
    outside.view.mode = ViewMode::SingleScreen;
    setZoomPercent(outside, 100.0);
    setPan(outside, 2.0, 2.0);
    UNIT_CHECK(ctx, classifyPanePixel(outside, 0, 0).kind == PixelKind::Outside);
    UNIT_CHECK(ctx, classifyPanePixel(outside, 0, 0).pane == 0);
    UNIT_CHECK(ctx, classifyPanePixel(outside, 2, 0).kind == PixelKind::Image);
    UNIT_CHECK(ctx, probeTarget(outside, 2, 0).source == 0);
    UNIT_CHECK(ctx, probeTarget(outside, 2, 0).localX == 0);
    UNIT_CHECK(ctx, classifyPanePixel(outside, 4, 0).kind == PixelKind::Padding);
    UNIT_CHECK(ctx, classifyPanePixel(outside, 2, 2).kind == PixelKind::Outside);

    // Negative offsets: the probe local coordinate subtracts the negative offset.
    LayoutInput negative = makeWindow(4, 4);
    addSource(negative, makeSource(4, 4, -3, -2));
    negative.view.mode = ViewMode::SingleScreen;
    setZoomPercent(negative, 100.0);
    centerView(negative);
    const ProbeTarget corner = probeTarget(negative, 0, 0);
    UNIT_CHECK(ctx, corner.source == 0 && corner.localX == 0 && corner.localY == 0);
    const ProbeTarget far = probeTarget(negative, 3, 3);
    UNIT_CHECK(ctx, far.source == 0 && far.localX == 3 && far.localY == 3);
}

void checkWipe(Context & ctx)
{
    // Single screen, 8 x 4 window, 100%, pan (4, 2): window pixel (px, py) has image
    // coordinate (px + 0.5, py + 0.5). Both sources cover [0, 4) x [0, 4).
    LayoutInput input = makeWindow(8, 4);
    addSource(input, makeSource(4, 4));
    addSource(input, makeSource(4, 4));
    input.view.mode = ViewMode::SingleScreen;
    setZoomPercent(input, 100.0);
    setPan(input, 4.0, 2.0);

    WipeSide side = WipeSide::A;

    // Wipe at 0 (start of the extent): everything in the extent shows B.
    setWipeFraction(input, 0.0);
    UNIT_CHECK_NEAR(ctx, input.view.wipePosition, 0.0, 1e-12);
    UNIT_CHECK(ctx, wipeSideAtPixel(input, 0, 0, side) && side == WipeSide::B);
    UNIT_CHECK(ctx, wipeSideAtPixel(input, 3, 0, side) && side == WipeSide::B);

    // Wipe at 1 (end of the extent): everything in the extent shows A.
    setWipeFraction(input, 1.0);
    UNIT_CHECK_NEAR(ctx, input.view.wipePosition, 4.0, 1e-12);
    UNIT_CHECK(ctx, wipeSideAtPixel(input, 0, 0, side) && side == WipeSide::A);
    UNIT_CHECK(ctx, wipeSideAtPixel(input, 3, 0, side) && side == WipeSide::A);

    // Wipe at the middle: pixels 0 and 1 show A, pixels 2 and 3 show B.
    setWipeFraction(input, 0.5);
    UNIT_CHECK(ctx, wipeSideAtPixel(input, 1, 0, side) && side == WipeSide::A);
    UNIT_CHECK(ctx, wipeSideAtPixel(input, 2, 0, side) && side == WipeSide::B);
    UNIT_CHECK(ctx, wipeSideAt(1.9, 2.0) == WipeSide::A);
    UNIT_CHECK(ctx, wipeSideAt(2.0, 2.0) == WipeSide::B);

    // The stored position is clamped to the extent: far left shows B everywhere,
    // far right shows A everywhere.
    input.view.wipePosition = -100.0;
    UNIT_CHECK(ctx, wipeSideAtPixel(input, 0, 0, side) && side == WipeSide::B);
    input.view.wipePosition = 100.0;
    UNIT_CHECK(ctx, wipeSideAtPixel(input, 3, 0, side) && side == WipeSide::A);

    // Mapping the wipe to the window and back (middle wipe at image x 2 -> window x 2).
    setWipeFraction(input, 0.5);
    double coordinate = 0.0;
    UNIT_CHECK(ctx, wipeWindowCoordinate(input, 0, coordinate));
    UNIT_CHECK_NEAR(ctx, coordinate, 2.0, 1e-12);
    double imagePosition = 0.0;
    UNIT_CHECK(ctx, wipePositionAtPixel(input, 2, 0, imagePosition));
    UNIT_CHECK_NEAR(ctx, imagePosition, 2.5, 1e-12);
    UNIT_CHECK(ctx, !wipeWindowCoordinate(input, 1, coordinate));

    // Horizontal wipe at the middle: rows 0 and 1 show A, rows 2 and 3 show B.
    input.view.wipeOrientation = WipeOrientation::Horizontal;
    setWipeFraction(input, 0.5);
    UNIT_CHECK_NEAR(ctx, input.view.wipePosition, 2.0, 1e-12);
    UNIT_CHECK(ctx, wipeSideAtPixel(input, 0, 1, side) && side == WipeSide::A);
    UNIT_CHECK(ctx, wipeSideAtPixel(input, 0, 2, side) && side == WipeSide::B);
}

void checkPadding(Context & ctx)
{
    // Display counterpart: pixel (0,0) is (0.25, 0.5, 1.5) with alpha 0.25. Each channel
    // is clamped to [0,1] and inverted; alpha is inverted.
    ImageData display;
    display.width = 2;
    display.height = 1;
    display.encoding = Encoding::Display;
    display.rgba = {0.25f, 0.5f, 1.5f, 0.25f, 0.0f, 0.0f, 0.0f, 1.0f};
    RgbaColor color;
    UNIT_CHECK(ctx, paddingColorFor(display, 0, 0, ToneMapper::Reinhard, 0.0f, color));
    UNIT_CHECK_NEAR(ctx, color.r, 0.75, 1e-6);
    UNIT_CHECK_NEAR(ctx, color.g, 0.5, 1e-6);
    UNIT_CHECK_NEAR(ctx, color.b, 0.0, 1e-6);
    UNIT_CHECK_NEAR(ctx, color.a, 0.75, 1e-6);
    UNIT_CHECK(ctx, paddingColorFor(display, 1, 0, ToneMapper::Reinhard, 0.0f, color));
    UNIT_CHECK_NEAR(ctx, color.r, 1.0, 1e-6);
    UNIT_CHECK_NEAR(ctx, color.a, 0.0, 1e-6);

    // No counterpart pixel: background, and the output is left unchanged.
    RgbaColor untouched;
    untouched.r = 0.125f;
    UNIT_CHECK(ctx, !paddingColorFor(display, 2, 0, ToneMapper::Reinhard, 0.0f, untouched));
    UNIT_CHECK(ctx, !paddingColorFor(display, -1, 0, ToneMapper::Reinhard, 0.0f, untouched));
    UNIT_CHECK_NEAR(ctx, untouched.r, 0.125, 1e-7);

    // A buffer shorter than width * height * 4 is treated as no pixel.
    ImageData shortBuffer = display;
    shortBuffer.rgba.resize(4);
    UNIT_CHECK(ctx, !paddingColorFor(shortBuffer, 1, 0, ToneMapper::Reinhard, 0.0f, untouched));

    // Premultiplied (raw capture) counterpart: rgb (0.2, 0, 0) at alpha 0.5. Over the default
    // black the rgb is unchanged, so out = 1 - rgb with alpha 1 - 0.5.
    ImageData premul;
    premul.width = 1;
    premul.height = 1;
    premul.encoding = Encoding::Display;
    premul.premultiplied = true;
    premul.rgba = {0.2f, 0.0f, 0.0f, 0.5f};
    UNIT_CHECK(ctx, paddingColorFor(premul, 0, 0, ToneMapper::Reinhard, 0.0f, color));
    UNIT_CHECK_NEAR(ctx, color.r, 0.8, 1e-6);
    UNIT_CHECK_NEAR(ctx, color.g, 1.0, 1e-6);
    UNIT_CHECK_NEAR(ctx, color.b, 1.0, 1e-6);
    UNIT_CHECK_NEAR(ctx, color.a, 0.5, 1e-6);

    // Over (0, 0.5, 0): g = rgb + (1 - 0.5) * 0.5 = 0.25, so out.g = 0.75; out.r stays 0.8.
    const RgbaColor green{0.0f, 0.5f, 0.0f, 1.0f};
    UNIT_CHECK(ctx, paddingColorFor(premul, 0, 0, ToneMapper::Reinhard, 0.0f, color, green));
    UNIT_CHECK_NEAR(ctx, color.r, 0.8, 1e-6);
    UNIT_CHECK_NEAR(ctx, color.g, 0.75, 1e-6);
    UNIT_CHECK_NEAR(ctx, color.b, 1.0, 1e-6);
    UNIT_CHECK_NEAR(ctx, color.a, 0.5, 1e-6);

    // A straight-alpha source ignores the background: the same numbers, inverted directly.
    ImageData straight = premul;
    straight.premultiplied = false;
    UNIT_CHECK(ctx, paddingColorFor(straight, 0, 0, ToneMapper::Reinhard, 0.0f, color, green));
    UNIT_CHECK_NEAR(ctx, color.r, 0.8, 1e-6);
    UNIT_CHECK_NEAR(ctx, color.g, 1.0, 1e-6);
    UNIT_CHECK_NEAR(ctx, color.b, 1.0, 1e-6);
    UNIT_CHECK_NEAR(ctx, color.a, 0.5, 1e-6);

    // Linear premultiplied counterpart: rgb 0.4 at alpha 0.5 over the background 0.25 gives
    // 0.4 + 0.5 * 0.25 = 0.525 on the linear values, then displayTransform, then invert.
    ImageData linearPremul = premul;
    linearPremul.encoding = Encoding::Linear;
    linearPremul.rgba = {0.4f, 0.0f, 0.0f, 0.5f};
    const RgbaColor warm{0.25f, 0.0f, 0.0f, 1.0f};
    UNIT_CHECK(ctx, paddingColorFor(linearPremul, 0, 0, ToneMapper::Aces, 0.5f, color, warm));
    const float composited = 0.4f + 0.5f * 0.25f;
    UNIT_CHECK_NEAR(ctx, color.r, 1.0 - displayTransform(composited, ToneMapper::Aces, 0.5f), 1e-6);
    UNIT_CHECK_NEAR(ctx, color.a, 0.5, 1e-6);

    // Linear counterpart: 0.5 with Reinhard and no exposure is 0.5 / 1.5 = 1/3 after the
    // tone curve, then the sRGB transfer, then inverted.
    ImageData linear;
    linear.width = 1;
    linear.height = 1;
    linear.encoding = Encoding::Linear;
    linear.rgba = {0.5f, 0.5f, 0.5f, 1.0f};
    UNIT_CHECK(ctx, paddingColorFor(linear, 0, 0, ToneMapper::Reinhard, 0.0f, color));
    const double expected = 1.0 - srgbEncode(1.0 / 3.0);
    UNIT_CHECK_NEAR(ctx, color.r, expected, 1e-5);
    UNIT_CHECK_NEAR(ctx, color.b, expected, 1e-5);
    UNIT_CHECK_NEAR(ctx, color.a, 0.0, 1e-6);

    // Display transform: exposure +1 stop on 0.25 equals exposure 0 on 0.5; the
    // transform of 0.5 is the sRGB encoding of 1/3.
    UNIT_CHECK_NEAR(ctx, displayTransform(0.25f, ToneMapper::Reinhard, 1.0f),
        displayTransform(0.5f, ToneMapper::Reinhard, 0.0f), 1e-6);
    UNIT_CHECK_NEAR(ctx, displayTransform(0.5f, ToneMapper::Reinhard, 0.0f), srgbEncode(1.0 / 3.0), 1e-5);
    UNIT_CHECK_NEAR(ctx, displayTransform(0.0f, ToneMapper::Reinhard, 0.0f), 0.0, 1e-7);
    UNIT_CHECK_NEAR(ctx, displayTransform(0.0f, ToneMapper::Aces, 0.0f), 0.0, 1e-6);
    UNIT_CHECK_NEAR(ctx, displayTransform(0.0f, ToneMapper::Hable, 0.0f), 0.0, 1e-6);
    const float aces = displayTransform(0.5f, ToneMapper::Aces, 0.0f);
    const float hable = displayTransform(0.5f, ToneMapper::Hable, 0.0f);
    UNIT_CHECK(ctx, aces > 0.0f && aces < 1.0f);
    UNIT_CHECK(ctx, hable > 0.0f && hable < 1.0f);
    const float huge = displayTransform(1.0e9f, ToneMapper::Reinhard, 0.0f);
    UNIT_CHECK(ctx, huge >= 0.0f && huge <= 1.0f);
    UNIT_CHECK_NEAR(ctx, displayTransform(0.5f, ToneMapper::Reinhard, std::nanf("")),
        displayTransform(0.5f, ToneMapper::Reinhard, 0.0f), 1e-7);

    // Maps: padding draws the maximum, the sample is kept for images, and outside is NaN.
    UNIT_CHECK(ctx, mapPaddingValue() == 1.0f);
    UNIT_CHECK(ctx, mapValueAt(0.3f, PixelKind::Image) == 0.3f);
    UNIT_CHECK(ctx, mapValueAt(0.3f, PixelKind::Padding) == 1.0f);
    UNIT_CHECK(ctx, std::isnan(mapValueAt(0.3f, PixelKind::Outside)));
    UNIT_CHECK(ctx, std::isnan(kPaddingDepth));
}

// Depth and alpha map helpers: which depths count, the hatch pattern, the map
// positions and the difference positions.
void checkDepthMaps(Context & ctx)
{
    // Valid depth: finite and positive only.
    UNIT_CHECK(ctx, isValidDepth(1.0f));
    UNIT_CHECK(ctx, isValidDepth(1.0e-6f));
    UNIT_CHECK(ctx, !isValidDepth(0.0f));
    UNIT_CHECK(ctx, !isValidDepth(-0.0f));
    UNIT_CHECK(ctx, !isValidDepth(-2.0f));
    UNIT_CHECK(ctx, !isValidDepth(std::nanf("")));
    UNIT_CHECK(ctx, !isValidDepth(std::numeric_limits<float>::infinity()));
    UNIT_CHECK(ctx, !isValidDepth(-std::numeric_limits<float>::infinity()));

    // Hatch: 8-pixel diagonal stripes, dark (0.12) on even bands, light on odd bands.
    UNIT_CHECK(ctx, hatchValue(0, 0) == 0.12f);
    UNIT_CHECK(ctx, hatchValue(7, 0) == 0.12f);
    UNIT_CHECK(ctx, hatchValue(8, 0) == kHatchLight);
    UNIT_CHECK(ctx, hatchValue(4, 4) == kHatchLight);
    UNIT_CHECK(ctx, hatchValue(15, 1) == 0.12f);
    UNIT_CHECK(ctx, hatchValue(8, 8) == 0.12f);
    UNIT_CHECK(ctx, hatchValue(0, 8) == kHatchLight);

    // Map position: linear in the range, clamped at both ends.
    UNIT_CHECK_NEAR(ctx, depthMapT(5.0f, 1.0f, 9.0f), 0.5, 1e-7);
    UNIT_CHECK_NEAR(ctx, depthMapT(1.0f, 1.0f, 9.0f), 0.0, 1e-7);
    UNIT_CHECK_NEAR(ctx, depthMapT(9.0f, 1.0f, 9.0f), 1.0, 1e-7);
    UNIT_CHECK_NEAR(ctx, depthMapT(-3.0f, 1.0f, 9.0f), 0.0, 1e-7);
    UNIT_CHECK_NEAR(ctx, depthMapT(20.0f, 1.0f, 9.0f), 1.0, 1e-7);
    UNIT_CHECK_NEAR(ctx, depthMapT(std::nanf(""), 1.0f, 9.0f), 0.0, 1e-7);

    // Zero and negative spans give 0, never a division by zero.
    UNIT_CHECK_NEAR(ctx, depthMapT(4.0f, 4.0f, 4.0f), 0.0, 1e-7);
    UNIT_CHECK_NEAR(ctx, depthMapT(4.0f, 9.0f, 1.0f), 0.0, 1e-7);

    // Difference position: |a - b| over the span, symmetric in a and b.
    UNIT_CHECK_NEAR(ctx, depthDiffT(3.0f, 7.0f, 0.0f, 8.0f), 0.5, 1e-7);
    UNIT_CHECK_NEAR(ctx, depthDiffT(7.0f, 3.0f, 0.0f, 8.0f), 0.5, 1e-7);
    UNIT_CHECK_NEAR(ctx, depthDiffT(2.0f, 2.0f, 0.0f, 8.0f), 0.0, 1e-7);
    UNIT_CHECK_NEAR(ctx, depthDiffT(0.0f, 100.0f, 0.0f, 8.0f), 1.0, 1e-7);
    // Zero span: any difference is the top of the map, equal depths the bottom.
    UNIT_CHECK_NEAR(ctx, depthDiffT(2.0f, 2.0f, 5.0f, 5.0f), 0.0, 1e-7);
    UNIT_CHECK_NEAR(ctx, depthDiffT(2.0f, 5.0f, 5.0f, 5.0f), 1.0, 1e-7);
    UNIT_CHECK_NEAR(ctx, depthDiffT(5.0f, 2.0f, 5.0f, 5.0f), 1.0, 1e-7);
}

void checkDivider(Context & ctx)
{
    // The fraction is clamped to [0,1]; NaN reads as the default, the middle.
    UNIT_CHECK_NEAR(ctx, clampDividerFraction(-1.0), 0.0, 0.0);
    UNIT_CHECK_NEAR(ctx, clampDividerFraction(0.0), 0.0, 0.0);
    UNIT_CHECK_NEAR(ctx, clampDividerFraction(0.25), 0.25, 0.0);
    UNIT_CHECK_NEAR(ctx, clampDividerFraction(1.0), 1.0, 0.0);
    UNIT_CHECK_NEAR(ctx, clampDividerFraction(2.0), 1.0, 0.0);
    UNIT_CHECK_NEAR(ctx, clampDividerFraction(std::numeric_limits<double>::quiet_NaN()), 0.5, 0.0);
    UNIT_CHECK_NEAR(ctx, clampDividerFraction(std::numeric_limits<double>::infinity()), 1.0, 0.0);
    UNIT_CHECK_NEAR(ctx, clampDividerFraction(-std::numeric_limits<double>::infinity()), 0.0, 0.0);

    // The window x of the divider: the fraction times the window width.
    LayoutInput wide = makeWindow(800, 100);
    wide.view.dividerFraction = 0.25;
    UNIT_CHECK_NEAR(ctx, dividerWindowX(wide), 200.0, 1e-12);
    wide.view.dividerFraction = 0.0;
    UNIT_CHECK_NEAR(ctx, dividerWindowX(wide), 0.0, 1e-12);
    wide.view.dividerFraction = 1.0;
    UNIT_CHECK_NEAR(ctx, dividerWindowX(wide), 800.0, 1e-12);
    LayoutInput narrow = makeWindow(1, 100);
    narrow.view.dividerFraction = 0.5;
    UNIT_CHECK_NEAR(ctx, dividerWindowX(narrow), 0.5, 1e-12);
    // A window with no width puts the divider at 0.
    LayoutInput empty = makeWindow(0, 100);
    UNIT_CHECK_NEAR(ctx, dividerWindowX(empty), 0.0, 1e-12);

    // The fraction at a window x: the inverse of dividerWindowX(), clamped to the window.
    UNIT_CHECK_NEAR(ctx, dividerFractionAtWindowX(wide, 200.0), 0.25, 1e-12);
    UNIT_CHECK_NEAR(ctx, dividerFractionAtWindowX(wide, -50.0), 0.0, 0.0);
    UNIT_CHECK_NEAR(ctx, dividerFractionAtWindowX(wide, 900.0), 1.0, 0.0);
    UNIT_CHECK_NEAR(ctx, dividerFractionAtWindowX(empty, 10.0), 0.5, 0.0);
    wide.view.dividerFraction = 0.3;
    UNIT_CHECK_NEAR(ctx, dividerFractionAtWindowX(wide, dividerWindowX(wide)), 0.3, 1e-12);

    // The image x under the divider in a single pane. A 400 x 100 source fills the window
    // at 100% (800 x 100 window, panned to its center). The divider at window x 200 is image
    // x 0: the same as the boundary between window pixels 199 and 200.
    LayoutInput single = makeWindow(800, 100);
    addSource(single, makeSource(400, 100));
    single.view.mode = ViewMode::SingleScreen;
    setZoomPercent(single, 100.0);
    centerView(single);
    single.view.dividerFraction = 0.25;
    double imageX = 0.0;
    UNIT_CHECK(ctx, dividerImagePosition(single, imageX));
    UNIT_CHECK_NEAR(ctx, imageX, 0.0, 1e-12);
    const ImageHit left = windowToImage(single, 199, 0);
    const ImageHit right = windowToImage(single, 200, 0);
    UNIT_CHECK_NEAR(ctx, imageX, (left.imageX + right.imageX) * 0.5, 1e-12);

    // The same at Fit: the 400 x 100 source fits the 800 x 200 window at 200%. The pan is the
    // extent's center (200), the pane center is window x 400, so window x 200 is 200 window
    // pixels = 100 image pixels left of the pan: image x 100.
    LayoutInput fit = makeWindow(800, 200);
    addSource(fit, makeSource(400, 100));
    fit.view.mode = ViewMode::SingleScreen;
    setZoomFit(fit);
    centerView(fit);
    fit.view.dividerFraction = 0.25;
    UNIT_CHECK(ctx, dividerImagePosition(fit, imageX));
    UNIT_CHECK_NEAR(ctx, imageX, 100.0, 1e-12);
    UNIT_CHECK_NEAR(ctx, imageX, (windowToImage(fit, 199, 0).imageX + windowToImage(fit, 200, 0).imageX) * 0.5, 1e-12);

    // No pane (the single source is hidden, so the screen is empty) gives false.
    LayoutInput none = makeWindow(800, 100);
    addSource(none, makeSource(400, 100, 0, 0, Encoding::Display, false));
    none.view.mode = ViewMode::SingleScreen;
    UNIT_CHECK(ctx, !dividerImagePosition(none, imageX));
}

} // namespace

void testCompareLayout(Context & ctx)
{
    checkSources(ctx);
    checkAnchors(ctx);
    checkExtents(ctx);
    checkPanes(ctx);
    checkZoomAndCentering(ctx);
    checkMapping(ctx);
    checkWipe(ctx);
    checkPadding(ctx);
    checkDepthMaps(ctx);
    checkDivider(ctx);
}

} // namespace sabercat::unit
