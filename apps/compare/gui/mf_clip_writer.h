// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Test helper: writes short, deterministic video clips with the Media Foundation sink writer,
// and the frame pattern that lets a test read the frame index back from a decoded frame.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sabercat::compare::gui
{

// One frame of a generated test clip: tight NV12 for the given size.
// Frame content (deterministic): a smooth background (Y gradient with a per-frame tint in UV) plus, at the top-left
// corner, kClipIndexBits (= 12) square cells of kClipCellSize (= 16) pixels in one row (so the strip is 192 x 16 px),
// cell b (b = 0 is the leftmost = least significant bit) is white (Y = 235) when bit b of the frame index is set and
// black (Y = 16) otherwise; UV = 128 inside the strip. Index in [0, 4095]. Width must be >= 192 and height >= 16, both even.
inline constexpr uint32_t kClipIndexBits = 12;
inline constexpr uint32_t kClipCellSize = 16;
void makeClipFrameNv12(uint32_t width, uint32_t height, uint32_t index, std::vector<uint8_t> & nv12);

// Reads the frame index back from the decoded luma plane (tight NV12 or any buffer whose first width*height bytes are
// the Y plane): the center sample of each cell > 125 means 1. Returns the index. Used by the tests on NV12 frames.
uint32_t decodeClipIndexFromLuma(const uint8_t * luma, uint32_t width);
// The same from an RGBA32F image (rgb luma 0.2126 r + 0.7152 g + 0.0722 b > 0.5 means 1; the pixel at the cell center).
uint32_t decodeClipIndexFromRgba(const float * rgba, uint32_t width);

struct ClipSpec
{
    std::string path;             // UTF-8 output file (.mp4)
    uint32_t width = 320;
    uint32_t height = 192;
    uint32_t fpsNumerator = 30;
    uint32_t fpsDenominator = 1;
    uint32_t frameCount = 90;     // <= 4096
    uint32_t keyframeInterval = 10;   // frames between keyframes (MF_MT_MAX_KEYFRAME_SPACING and, if the encoder supports it, CODECAPI_AVEncMPVGOPSize)
    uint32_t bitrate = 8000000;   // bits per second (high enough that the black/white cells survive)
    uint32_t rotationDegrees = 0; // MF_MT_VIDEO_ROTATION on the output type when non-zero (0, 90, 180, 270)
};

// Writes the clip with an IMFSinkWriter (MFCreateSinkWriterFromURL on the .mp4 path, H.264 output type, NV12 input,
// no video processing). Sample time of frame i = (i * 10,000,000 * den + num / 2) / num (rounded, exact, not accumulated),
// duration = the next frame's time minus this one. Throws std::runtime_error with the HRESULT text when the machine
// cannot encode (the caller reports "H.264 encoder unavailable" and skips the video tests).
// Throws std::invalid_argument for a spec that cannot be written.
void writeClipMp4(const ClipSpec & spec);
// Same for WEBM/VP9 if an MF VP9 encoder exists; returns false (writes nothing, no throw) when MF has no VP9 encoder.
bool tryWriteClipWebm(const ClipSpec & spec);

} // namespace sabercat::compare::gui
