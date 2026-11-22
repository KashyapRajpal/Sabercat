// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Media Foundation decoding of a video file to tight NV12 frames, on the CPU. The frames
// are host bytes; the GPU side (NV12 to RGBA32F) is nv12_converter.*. MfRuntime is the
// Media Foundation and COM start-up that the decoder and the test clip writer share.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace sabercat::compare::gui
{

enum class VideoOpenCode : uint8_t
{
    Ok,
    Missing,
    NoVideoStream,
    WebmNeedsVp9,
    Unsupported,
    Failed
};

struct VideoInfo
{
    double fps = 0.0;             // nominal frame rate (MF_MT_FRAME_RATE numerator / denominator), > 0
    double durationSeconds = 0.0; // MF_PD_DURATION; > 0
    int64_t frameCount = 1;       // max(1, llround(durationSeconds * fps)); an estimate for variable frame rate
    uint32_t width = 0;           // visible size (after the display aperture crop), even, >= 2
    uint32_t height = 0;
    uint32_t yuvMatrix = 0;       // MF_MT_YUV_MATRIX as reported (MFVideoTransferMatrix_*), 0 = not tagged
    uint32_t nominalRange = 0;    // MF_MT_VIDEO_NOMINAL_RANGE as reported (MFNominalRange_*), 0 = not tagged
    uint32_t rotationDegrees = 0; // MF_MT_VIDEO_ROTATION when the stream carries it, else 0 (informational; NOT applied)
    bool hasAudio = false;        // an audio stream exists (it is deselected and ignored)
};

struct Nv12Frame
{
    std::vector<uint8_t> bytes; // tight NV12: width*height Y bytes, then width*(height/2) interleaved UV bytes
    double timestamp = 0.0;     // seconds since the first frame of the stream (the first frame is 0)
    double duration = 0.0;      // seconds (sample duration; 1/fps when the sample has none)
    int64_t index = 0;          // llround(timestamp * fps)
};

// Starts Media Foundation (MFStartup, refcounted) and COM for its lifetime. ok() is false
// when the required DLLs are missing (an N edition without the Media Feature Pack) or when
// either start-up call fails; error() then says why. Declare it before the objects that
// use Media Foundation, so it is shut down after them.
class MfRuntime
{
  public:
    MfRuntime();
    ~MfRuntime();
    MfRuntime(const MfRuntime &) = delete;
    MfRuntime & operator=(const MfRuntime &) = delete;

    bool ok() const;
    const std::string & error() const;

  private:
    std::string m_error;
    bool m_ok = false;
    bool m_mfStarted = false;
    bool m_uninitializeCom = false;
};

// "0x%08X (system text)" for an HRESULT, the system text from FormatMessage when there is one.
std::string mfHresultText(int32_t hresult);

// Decoder state, defined in mf_decoder.cpp (holds the COM objects).
struct MfDecoderState;

class MfDecoder
{
  public:
    // Opens a file. Does not throw for I/O or codec problems: returns nullptr, sets code and message.
    // Missing -> "file not found: <path>". A .webm that MF cannot open or decode (no VP9 decoder) ->
    // code WebmNeedsVp9, message EXACTLY: "WEBM needs the Windows VP9 Video Extensions (free from Microsoft Store); the MP4 path is unaffected."
    // No video stream -> NoVideoStream "no video stream". Not 4:2:0 8-bit (decoder will not give NV12) -> Unsupported, say why.
    // Odd or zero size -> Unsupported "video size WxH is not supported (needs even width and height)".
    // Open decodes the first frame (so errors surface here) and keeps it as the current frame.
    static std::unique_ptr<MfDecoder> open(const std::string & utf8Path, VideoOpenCode & code, std::string & message);
    ~MfDecoder();
    MfDecoder(const MfDecoder &) = delete;
    MfDecoder & operator=(const MfDecoder &) = delete;

    const VideoInfo & info() const;

    // The frame that is shown at frame index `index` (time index / fps), index clamped to [0, frameCount - 1].
    // Returns the frame whose [timestamp, timestamp + duration) contains index/fps, with a tolerance of
    // 0.05/fps (so a 100 ns timestamp rounding never picks the neighbor). Sequential or short forward requests
    // decode forward without seeking (intermediate frames are decoded and dropped); a request behind the current
    // frame, or more than kSeekAheadFrames (= 30) frames ahead, seeks with IMFSourceReader::SetCurrentPosition
    // and decodes forward from there until the frame covering the target is reached. At end of stream the last
    // decoded frame is returned (the index of that frame is in frame->index; a past-the-end request clamps).
    // The returned pointer is valid until the next readFrame call. Returns nullptr on failure and sets message.
    // If the first sample after a seek is already later than the target, the decoder seeks again further back
    // (target - 1 s, then target - 2 s, ... down to 0) and reports a failure only when that is exhausted.
    const Nv12Frame * readFrame(int64_t index, std::string & message);

    // Bytes the decoder holds on the host (frame buffers + an estimate of 4 frames of decoder pool). For the memory budget.
    uint64_t hostBytes() const;
    // Statistics for the performance measurement: seconds spent in the last readFrame and the number of
    // samples decoded by it (including dropped ones), and the total seeks done since open.
    double lastDecodeSeconds() const;
    uint32_t lastDecodedSamples() const;
    uint32_t seekCount() const;

  private:
    explicit MfDecoder(std::unique_ptr<MfDecoderState> state);

    std::unique_ptr<MfDecoderState> m_state;
};

} // namespace sabercat::compare::gui
