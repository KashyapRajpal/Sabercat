// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// A video file as a compare source. The frame shown is chosen from one global clock through
// the timeline math (timeline.h): the decoder (mf_decoder.h, Media Foundation, CPU) gives the
// NV12 frame of that index, the frame is uploaded to a device buffer and converted to RGBA32F
// by the shader compare_nv12 (nv12_converter.h). The source then looks like a still image to
// the compositor and the metrics. Trim, loop and time offset are per source.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "image_data.h"
#include "media_source.h"
#include "mf_decoder.h"
#include "timeline.h"

namespace sabercat
{
class Buffer;
} // namespace sabercat

namespace sabercat::compare::gui
{

class GpuContext;
class Nv12Converter;

// True for a path whose extension is .mp4, .m4v or .webm (case-insensitive).
bool isVideoFile(const std::string & utf8Path);

class VideoSource final : public MediaSource
{
  public:
    // Opens the file with MfDecoder, checks the size and the memory budget, creates the device
    // buffers, decodes the first frame (frame 0) and converts it. Never throws. Status Missing for
    // a file that does not exist ("file not found: <path>"), Error otherwise with the decoder's
    // message (the WEBM message verbatim). Refusals (status Error):
    //   "video size WxH is above the limit of L pixels on a side" (L >= 4096: the device's
    //   maxImageDimension2D when that is larger), and "Not enough GPU memory for this video: needs
    //   about N MiB, M MiB free" from deviceBudgetOk().
    VideoSource(GpuContext & context, const std::string & utf8Path);
    // The GPU must be idle (the caller's frame is done), as for ImageSource.
    ~VideoSource() override;

    VideoSource(const VideoSource &) = delete;
    VideoSource & operator=(const VideoSource &) = delete;

    project::SourceKind kind() const override
    {
        return project::SourceKind::Video;
    }
    SourceStatus status() const override
    {
        return m_status;
    }
    // The error text, or after a failed frame decode "video decode failed: ..." while the
    // status stays Ready. Empty otherwise.
    const std::string & statusMessage() const override
    {
        return m_message;
    }
    uint32_t width() const override
    {
        return m_info.width;
    }
    uint32_t height() const override
    {
        return m_info.height;
    }
    // Display (the video's own encoded values, BT.709 limited range converted to R'G'B').
    Encoding encoding() const override
    {
        return Encoding::Display;
    }
    bool premultiplied() const override
    {
        return false;
    }
    bool hasAlpha() const override
    {
        return false;
    }

    // Width * height RGBA32F texels (stride 16). Valid when Ready; the same Buffer object for the
    // life of the source.
    BufferView color() const override;

    // Advances to the frame of globalTime. Returns true when color() changed. A failed decode keeps
    // the previous frame, sets the message and returns false (the status stays Ready).
    bool update(double globalTime) override;

    // A 16-byte readback of the texel (the probe). False when out of range or not Ready.
    bool readPixel(int32_t x, int32_t y, float rgba[4]) const override;
    // A lazily read back host copy of color() (rgba bit for bit, width, height, Display, bitDepth 8,
    // hasAlpha false, premultiplied false, no depth). Cached until the frame changes. Null when not
    // Ready or when the readback fails.
    const ImageData * hostData() const override;

    // ---- Timeline settings (one TimeSource for timeline.*) ----
    // Trim in seconds. trimOut < 0 means the end of the clip. Refused (false, the message is set,
    // nothing changes) unless 0 <= trimIn < the resolved trimOut <= duration (finite values only).
    bool setTrim(double trimIn, double trimOut);
    void setLoop(bool loop);
    // Seconds added to the global clock; finite and |seconds| <= 86400, else refused (false).
    bool setTimeOffset(double seconds);
    double trimIn() const;
    // The resolved end of the trim (never < 0).
    double trimOut() const;
    // True when the trim end is "the end of the clip" (a project writes -1 for it).
    bool trimToEnd() const;
    bool loop() const;
    double timeOffset() const;

    // The TimeSource for timelineDuration() and frame stepping: constant fps, no frame times.
    const TimeSource & timeSource() const;
    double fps() const;
    int64_t frameCount() const;
    double durationSeconds() const;
    // The frame index shown (from the last successful decode) and its source time (index / fps).
    int64_t frameIndex() const;
    double sourceTime() const;
    // The stream facts (matrix, range, rotation, audio) for the UI.
    const VideoInfo & videoInfo() const;

    // Device bytes of this source (NV12 buffer and color buffer) and host bytes (decoder plus the
    // ImageData cache when present).
    uint64_t deviceBytes() const;
    uint64_t hostBytes() const;

    // Timing of the last update() that decoded: decode (MF), upload and convert, in seconds. Zero
    // before the first.
    struct FrameTiming
    {
        double decodeSeconds = 0.0;
        double uploadSeconds = 0.0;
        double convertSeconds = 0.0;
    };
    FrameTiming lastTiming() const;

    // The path as given.
    const std::string & path() const
    {
        return m_path;
    }
    // The decoder (for tests: seeks, sample counts). Null when not Ready.
    MfDecoder * decoder();

  private:
    void fail(SourceStatus status, std::string message);
    // Uploads one decoded frame to the NV12 buffer and converts it into the color buffer. Adds the
    // upload and convert times to timing. Throws std::exception on a GPU or size error.
    void uploadFrame(const Nv12Frame & frame, FrameTiming & timing);

    GpuContext & m_context;
    std::string m_path;
    std::string m_message;
    std::unique_ptr<MfDecoder> m_decoder;
    std::unique_ptr<Nv12Converter> m_converter;
    std::unique_ptr<sabercat::Buffer> m_nv12;
    std::unique_ptr<sabercat::Buffer> m_color;
    VideoInfo m_info;
    TimeSource m_timeSource;
    FrameTiming m_lastTiming;
    mutable ImageData m_host;
    int64_t m_frameIndex = 0;
    // The index last requested from update() and decoded (or kept after a failure).
    int64_t m_requestedIndex = -1;
    SourceStatus m_status = SourceStatus::Error;
    bool m_trimToEnd = true;
    bool m_decodeFailed = false;
    mutable bool m_hostValid = false;
};

} // namespace sabercat::compare::gui
