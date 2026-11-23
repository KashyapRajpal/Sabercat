// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// CPU-side exports of the compare window: the padded pane images, the full-resolution
// diff and FLIP maps, PNG output and the comparison report. Everything works on host ImageData, so
// the rules are the ones compare_layout.h and flip_hdr.h define; the compositor (GPU)
// follows the same rules and the selftest checks the two against each other. No Vulkan.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "compare_layout.h"
#include "flip_hdr.h"
#include "image_compare.h"
#include "image_data.h"
#include "report.h"

namespace sabercat::compare::gui
{

// The gray of the background outside the padded extent and of padding with no
// counterpart. Equal to kCompositeBackground in compositor.h (kept here so this file
// does not need the GPU headers).
inline constexpr float kExportBackground = 0.12f;

// One source of an export: its host copy and the composite settings the view uses. The
// fields are ordered by size.
struct ExportSource
{
    // Host copy. Its width and height must equal those of geometry. Never null.
    const ImageData * image = nullptr;
    // Background of premultiplied compositing (r, g, b; alpha unused).
    RgbaColor background{0.0f, 0.0f, 0.0f, 1.0f};
    // Layout entry: offset, size and visibility. Hidden sources are not exported.
    LayoutSource geometry;
    // Index into ExportScene::sources of this source's padding counterpart (the
    // paddingCounterpartSlot() rule, applied to the visible sources), or -1 for none.
    int32_t counterpart = -1;
    float exposureStops = 0.0f;
    ToneMapper toneMapper = ToneMapper::Aces;
};

// The sources of one export. Indices are the source indices of the layout.
struct ExportScene
{
    // Up to kMaxSources entries, in index order.
    std::vector<ExportSource> sources;
    // The A/B pair; -1 when absent. The diff map needs both visible.
    int32_t sourceA = -1;
    int32_t sourceB = -1;
};

// Padded extent of the visible sources of a scene (the union of their rectangles).
// Empty when none is visible.
PixelRect exportExtent(const ExportScene & scene);

// The padded pane image of visible source `index`, over exportExtent(scene) (top row
// first, RGB, 8 bits). Inside the source's own rectangle it shows the source's display
// color: premultiplied sources are composited over their background first, Display
// samples are clamped to [0,1], Linear samples go through displayTransform(). In padding
// (inside the extent, outside the source) it shows the counterpart's pixel inverted
// (paddingColorFor()), or the background gray when the counterpart has no pixel there.
// Throws std::invalid_argument for an index that is out of range or hidden, an empty
// extent, or a scene whose geometry does not match its images.
Rgb8Image paddedPaneImage(const ExportScene & scene, int32_t index);

// The full-resolution diff map over exportExtent(scene), colored with colorizeMagma():
// the mean over RGB of |A - B| in display space (displayRgb() values), times diffGain,
// clamped to 1. A pixel that either source lacks is padding and takes the top color.
// Throws std::invalid_argument when A or B is missing or hidden, when diffGain is not
// finite or is negative, or on a bad scene (see paddedPaneImage()). Throws
// std::invalid_argument also when a sample is not finite (displayRgb()).
Rgb8Image diffMapImage(const ExportScene & scene, float diffGain);

// The FLIP map over exportExtent(scene), colored with colorizeMagma(): errorMap is the
// FLIP error map over overlap (one float per overlap pixel, row-major, top row first,
// FlipGpuResult::errorMap). Inside overlap it shows the map's value; elsewhere, in the
// padding, it shows mapPaddingValue() (the top color). Throws std::invalid_argument for an
// empty extent or overlap, an overlap that is not inside the extent, an errorMap whose size
// is not overlap.width * overlap.height, or a bad scene (see paddedPaneImage()).
Rgb8Image flipMapImage(const ExportScene & scene, const std::vector<float> & errorMap, PixelRect overlap);

// Writes RGBA8 pixels (width * height * 4 bytes, top row first) as a 24-bit PNG; the
// alpha channel is dropped. The path is UTF-8. Throws std::invalid_argument for a null
// buffer or an empty size, and std::runtime_error when the file cannot be written.
void saveRgbaPng(const std::string & utf8Path, const uint8_t * rgba, uint32_t width, uint32_t height);

// The inputs of a comparison report for the A/B pair.
struct ReportInput
{
    // Image A (the reference) and B (the test), both as loaded (host copies).
    std::string pathA;
    std::string pathB;
    const ImageData * a = nullptr;
    const ImageData * b = nullptr;
    // Canvas position of A's top-left pixel (the layout offset of A).
    int32_t aOffsetX = 0;
    int32_t aOffsetY = 0;
    // The compared rectangle, in canvas pixels. It must lie inside both sources.
    PixelRect overlap;
    // The metrics the GPU computed over the overlap (MetricsResult).
    double mse = 0.0;
    double psnrDb = 0.0;
    // The FLIP mean over the overlap, when the window has one (FlipGpuResult::mean).
    std::optional<double> flipMean;
    // The FLIP pixels per degree the window uses (CompareState::flipPpd()).
    float ppd = 0.0f;
    // The HDR section, when the FLIP result is an HDR-FLIP one (tone mapper and exposure range).
    std::optional<ReportExposures> hdr;
    // The linear-value MSE of an HDR-FLIP run (FlipHdrInfo::linearMse). Set only when finite.
    std::optional<double> linearMse;
    // "topLeft", "centered" or "custom", the anchor names of the project file.
    std::string anchor = "topLeft";
    // The frame shown by A and by B, set only when that source is a video.
    std::optional<int64_t> aFrameIndex;
    std::optional<int64_t> bFrameIndex;
    // The global clock in seconds, set only when A or B is a video. Must be finite.
    std::optional<double> timeSeconds;
};

class CompareState;

// Fills the video fields of input from the window's state: the frame shown by A and by B
// when each is a video, and the transport time when either is. An image-only pair sets
// nothing. The report export calls this before makeReport().
void addVideoReportFields(const CompareState & state, ReportInput & input);

// The report of the pair. The overlap is written in image A's pixel coordinates. The
// report carries the FLIP mean and ppd from the input and, for an HDR-FLIP run, the HDR
// section and the linear MSE. Throws
// std::invalid_argument when a pointer is null or the overlap is empty or outside A or B.
CompareReport makeReport(const ReportInput & input);

// Writes a report with saveReport(): JSON, or CSV for a .csv path. Throws
// std::runtime_error when the file cannot be written.
void saveReportFile(const std::string & utf8Path, const CompareReport & report);

} // namespace sabercat::compare::gui
