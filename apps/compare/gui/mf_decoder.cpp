// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "mf_decoder.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <objbase.h>
#include <propidl.h>
#include <mfapi.h>
#include <mferror.h>
#include <mfidl.h>
#include <mfobjects.h>
#include <mfreadwrite.h>
#include <wrl/client.h>

namespace sabercat::compare::gui
{

using Microsoft::WRL::ComPtr;

namespace
{

// A decode that moves more than this many frames ahead of the current frame seeks instead.
constexpr int64_t kSeekAheadFrames = 30;
// Media Foundation time is in 100 ns units.
constexpr double kHundredNsPerSecond = 10000000.0;
// Frame rate used when the file does not give one.
constexpr double kDefaultFps = 30.0;
// Stream indices probed for an audio stream.
constexpr DWORD kMaxStreamProbe = 32;

constexpr char kWebmMessage[] =
    "WEBM needs the Windows VP9 Video Extensions (free from Microsoft Store); the MP4 path is unaffected.";
constexpr char kNoMediaFoundationMessage[] =
    "Media Foundation is not available on this Windows (an N edition needs the Media Feature Pack)";

// Frame geometry from a media type: the decoder buffer and the visible area inside it.
struct FrameLayout
{
    uint32_t bufferWidth = 0;  // MF_MT_FRAME_SIZE; the buffer may be padded
    uint32_t bufferHeight = 0;
    uint32_t left = 0;         // visible area offset inside the buffer (even)
    uint32_t top = 0;          // (even)
    uint32_t stride = 0;       // MF_MT_DEFAULT_STRIDE, used for a contiguous (non 2D) buffer
};

// A sample from the source reader; its pixels stay in the buffer until they are copied.
struct RawSample
{
    ComPtr<IMFSample> sample;
    int64_t time = 0;     // 100 ns, as the reader reports it
    int64_t duration = 0; // 100 ns; 0 when the sample has none
};

// Converts a UTF-8 string to UTF-16 for the wide-path calls. Empty on failure.
std::wstring utf8ToWide(const std::string & utf8)
{
    if (utf8.empty())
    {
        return std::wstring();
    }
    const int needed = MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), nullptr, 0);
    if (needed <= 0)
    {
        return std::wstring();
    }
    std::wstring wide(static_cast<size_t>(needed), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), wide.data(), needed);
    return wide;
}

// Converts UTF-16 to UTF-8. Empty on failure.
std::string wideToUtf8(const std::wstring & wide)
{
    if (wide.empty())
    {
        return std::string();
    }
    const int needed = WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()), nullptr, 0,
                                           nullptr, nullptr);
    if (needed <= 0)
    {
        return std::string();
    }
    std::string utf8(static_cast<size_t>(needed), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()), utf8.data(), needed, nullptr,
                        nullptr);
    return utf8;
}

// A UTF-8 string as a path, without the deprecated u8path().
std::filesystem::path pathFromUtf8(const std::string & utf8)
{
    return std::filesystem::path(std::u8string(reinterpret_cast<const char8_t *>(utf8.data()), utf8.size()));
}

// True when the text ends with the suffix, ignoring ASCII case. The suffix is lower case.
bool endsWithIgnoreCase(const std::string & text, const std::string & suffix)
{
    if (text.size() < suffix.size())
    {
        return false;
    }
    std::string tail = text.substr(text.size() - suffix.size());
    for (char & c : tail)
    {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return tail == suffix;
}

// True when the DLL loads from the system directory. Checked before Media Foundation is
// started, so a missing DLL is reported instead of failing on the first call.
bool systemDllLoads(const wchar_t * name)
{
    const HMODULE module = LoadLibraryExW(name, nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (module == nullptr)
    {
        return false;
    }
    FreeLibrary(module);
    return true;
}

// A UINT32 attribute, or fallback when it is absent.
uint32_t uint32Attribute(IMFAttributes * attributes, REFGUID key, uint32_t fallback)
{
    UINT32 value = 0;
    if (attributes != nullptr && SUCCEEDED(attributes->GetUINT32(key, &value)))
    {
        return value;
    }
    return fallback;
}

// The attribute from the current type, or from the native type when the current type has none.
uint32_t uint32AttributeWithFallback(IMFAttributes * current, IMFAttributes * native, REFGUID key)
{
    const uint32_t value = uint32Attribute(current, key, 0);
    return value != 0 ? value : uint32Attribute(native, key, 0);
}

// Media Foundation time (100 ns) to seconds.
double toSeconds(int64_t hundredNs)
{
    return static_cast<double>(hundredNs) / kHundredNsPerSecond;
}

// True when a frame that starts at `start` and lasts `duration` covers time t, with the tolerance.
bool coversTime(double start, double duration, double t, double tol)
{
    return start <= t + tol && t < start + duration - tol;
}

// The frame rate of a media type (MF_MT_FRAME_RATE), or the default when it has none.
double frameRateOf(IMFMediaType * type)
{
    UINT32 numerator = 0;
    UINT32 denominator = 0;
    if (FAILED(MFGetAttributeRatio(type, MF_MT_FRAME_RATE, &numerator, &denominator)) || numerator == 0 ||
        denominator == 0)
    {
        return kDefaultFps;
    }
    return static_cast<double>(numerator) / static_cast<double>(denominator);
}

// The presentation duration in 100 ns, or 0 when the source has none.
int64_t presentationDuration(IMFSourceReader * reader)
{
    PROPVARIANT value;
    PropVariantInit(&value);
    int64_t duration = 0;
    if (SUCCEEDED(reader->GetPresentationAttribute(MF_SOURCE_READER_MEDIASOURCE, MF_PD_DURATION, &value)))
    {
        if (value.vt == VT_UI8)
        {
            duration = static_cast<int64_t>(value.uhVal.QuadPart);
        }
        else if (value.vt == VT_I8)
        {
            duration = value.hVal.QuadPart;
        }
    }
    PropVariantClear(&value);
    return duration;
}

// True when the source has an audio stream.
bool hasAudioStream(IMFSourceReader * reader)
{
    for (DWORD stream = 0; stream < kMaxStreamProbe; ++stream)
    {
        ComPtr<IMFMediaType> type;
        const HRESULT hr = reader->GetNativeMediaType(stream, 0, &type);
        if (hr == MF_E_INVALIDSTREAMNUMBER)
        {
            return false;
        }
        if (FAILED(hr))
        {
            continue;
        }
        GUID major = GUID_NULL;
        if (SUCCEEDED(type->GetMajorType(&major)) && IsEqualGUID(major, MFMediaType_Audio))
        {
            return true;
        }
    }
    return false;
}

// Reads the buffer size and the visible area (MF_MT_MINIMUM_DISPLAY_APERTURE) of a media type.
// The offsets are rounded down to even values, so chroma rows stay aligned with luma rows.
// False when the type has no frame size.
bool readLayout(IMFMediaType * type, FrameLayout & layout, uint32_t & visibleWidth, uint32_t & visibleHeight)
{
    UINT64 frameSize = 0;
    if (FAILED(type->GetUINT64(MF_MT_FRAME_SIZE, &frameSize)))
    {
        return false;
    }
    layout.bufferWidth = static_cast<uint32_t>(frameSize >> 32);
    layout.bufferHeight = static_cast<uint32_t>(frameSize & 0xFFFFFFFFu);
    const int32_t stride = static_cast<int32_t>(uint32Attribute(type, MF_MT_DEFAULT_STRIDE, layout.bufferWidth));
    layout.stride = static_cast<uint32_t>(stride < 0 ? -stride : stride);
    layout.left = 0;
    layout.top = 0;
    visibleWidth = layout.bufferWidth;
    visibleHeight = layout.bufferHeight;

    UINT32 blobSize = 0;
    MFVideoArea area = {};
    if (SUCCEEDED(type->GetBlobSize(MF_MT_MINIMUM_DISPLAY_APERTURE, &blobSize)) && blobSize >= sizeof(MFVideoArea) &&
        SUCCEEDED(type->GetBlob(MF_MT_MINIMUM_DISPLAY_APERTURE, reinterpret_cast<UINT8 *>(&area), sizeof(area),
                                &blobSize)))
    {
        const int64_t left = std::max<int64_t>(0, static_cast<int64_t>(area.OffsetX.value)) & ~int64_t{1};
        const int64_t top = std::max<int64_t>(0, static_cast<int64_t>(area.OffsetY.value)) & ~int64_t{1};
        const int64_t width = area.Area.cx;
        const int64_t height = area.Area.cy;
        if (width > 0 && height > 0 && left + width <= static_cast<int64_t>(layout.bufferWidth) &&
            top + height <= static_cast<int64_t>(layout.bufferHeight))
        {
            layout.left = static_cast<uint32_t>(left);
            layout.top = static_cast<uint32_t>(top);
            visibleWidth = static_cast<uint32_t>(width);
            visibleHeight = static_cast<uint32_t>(height);
        }
    }
    return layout.bufferWidth > 0 && layout.bufferHeight > 0;
}

} // namespace

// Decoder state behind MfDecoder. The runtime is declared first, so Media Foundation is
// shut down after the reader and the samples are released.
struct MfDecoderState
{
    MfRuntime runtime;
    ComPtr<IMFSourceReader> reader;
    VideoInfo info;
    FrameLayout layout;
    int64_t startTime = 0;     // the first sample's time (100 ns); the first frame is time 0
    Nv12Frame frame;           // the current frame
    bool hasFrame = false;
    RawSample ahead;           // a sample read past the last target, not used yet
    bool hasAhead = false;
    double lastDecodeSeconds = 0.0;
    uint32_t lastDecodedSamples = 0;
    uint32_t seekCount = 0;
};

namespace
{

// The outcome of reading toward one target time.
enum class Outcome
{
    Chosen, // `chosen` covers the target, or is the last frame before it (the hold at the end)
    Hold,   // forward decode only: no sample is usable, keep the current frame
    Beyond, // seeking only: the first sample is later than the target, seek further back
    Error
};

// Re-reads the layout from the reader's current type. The visible size must not change while
// decoding. False (message set) otherwise.
bool refreshLayout(MfDecoderState & s, std::string & message)
{
    ComPtr<IMFMediaType> type;
    uint32_t width = 0;
    uint32_t height = 0;
    if (FAILED(s.reader->GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, &type)) ||
        !readLayout(type.Get(), s.layout, width, height))
    {
        message = "the video media type could not be read";
        return false;
    }
    if (width != s.info.width || height != s.info.height)
    {
        message = "video size changed during playback";
        return false;
    }
    return true;
}

// Reads the next video sample. A sample held from an earlier read comes first. Stream ticks
// and native media type changes are skipped (a type change re-reads the layout). Returns 1
// for a sample, 0 at the end of the stream and -1 (message set) on error.
int readSample(MfDecoderState & s, RawSample & out, std::string & message)
{
    if (s.hasAhead)
    {
        out = std::move(s.ahead);
        s.ahead = RawSample();
        s.hasAhead = false;
        return 1;
    }
    for (;;)
    {
        DWORD flags = 0;
        LONGLONG timestamp = 0;
        ComPtr<IMFSample> sample;
        const HRESULT hr = s.reader->ReadSample(MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0, nullptr, &flags,
                                                &timestamp, &sample);
        if (FAILED(hr))
        {
            message = "reading the video failed: " + mfHresultText(hr);
            return -1;
        }
        if ((flags & MF_SOURCE_READERF_ENDOFSTREAM) != 0)
        {
            return 0;
        }
        if ((flags & (MF_SOURCE_READERF_NATIVEMEDIATYPECHANGED | MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED)) != 0 &&
            !refreshLayout(s, message))
        {
            return -1;
        }
        if ((flags & (MF_SOURCE_READERF_STREAMTICK | MF_SOURCE_READERF_NATIVEMEDIATYPECHANGED)) != 0 || !sample)
        {
            continue;
        }
        ++s.lastDecodedSamples;
        LONGLONG duration = 0;
        out.sample = sample;
        out.time = static_cast<int64_t>(timestamp);
        out.duration = SUCCEEDED(sample->GetSampleDuration(&duration)) ? static_cast<int64_t>(duration) : 0;
        return 1;
    }
}

// Copies the visible NV12 area of a sample into a tight buffer: width*height Y bytes, then
// width*(height/2) interleaved UV bytes. The UV plane of the buffer starts at pitch * bufferHeight.
bool copyVisible(const FrameLayout & layout, uint32_t width, uint32_t height, IMFSample * sample,
                 std::vector<uint8_t> & out, std::string & message)
{
    ComPtr<IMFMediaBuffer> buffer;
    HRESULT hr = sample->GetBufferByIndex(0, &buffer);
    if (FAILED(hr))
    {
        message = "the video sample has no buffer: " + mfHresultText(hr);
        return false;
    }

    BYTE * base = nullptr;
    LONG pitch = 0;
    ComPtr<IMF2DBuffer> buffer2d;
    bool locked2d = false;
    if (SUCCEEDED(buffer.As(&buffer2d)) && SUCCEEDED(buffer2d->Lock2D(&base, &pitch)))
    {
        locked2d = true;
    }

    ComPtr<IMFMediaBuffer> contiguous;
    if (!locked2d)
    {
        hr = sample->ConvertToContiguousBuffer(&contiguous);
        if (FAILED(hr))
        {
            message = "the video sample could not be read: " + mfHresultText(hr);
            return false;
        }
        DWORD maxLength = 0;
        DWORD currentLength = 0;
        hr = contiguous->Lock(&base, &maxLength, &currentLength);
        if (FAILED(hr))
        {
            message = "the video sample could not be locked: " + mfHresultText(hr);
            return false;
        }
        const uint64_t needed = static_cast<uint64_t>(layout.stride) * layout.bufferHeight * 3 / 2;
        if (currentLength < needed)
        {
            contiguous->Unlock();
            message = "the video sample is smaller than its frame size";
            return false;
        }
        pitch = static_cast<LONG>(layout.stride);
    }

    const ptrdiff_t stride = pitch;
    const BYTE * yBase = base + static_cast<ptrdiff_t>(layout.top) * stride + layout.left;
    const BYTE * uvBase = base + static_cast<ptrdiff_t>(layout.bufferHeight) * stride +
                          static_cast<ptrdiff_t>(layout.top / 2) * stride + layout.left;
    const size_t w = width;
    const size_t h = height;
    out.resize(w * h + w * (h / 2));
    for (size_t row = 0; row < h; ++row)
    {
        std::memcpy(out.data() + row * w, yBase + static_cast<ptrdiff_t>(row) * stride, w);
    }
    uint8_t * uvOut = out.data() + w * h;
    for (size_t row = 0; row < h / 2; ++row)
    {
        std::memcpy(uvOut + row * w, uvBase + static_cast<ptrdiff_t>(row) * stride, w);
    }

    if (locked2d)
    {
        buffer2d->Unlock2D();
    }
    else
    {
        contiguous->Unlock();
    }
    return true;
}

// Decodes a sample into the current frame and sets its timing. False (message set) on failure.
bool takeFrame(MfDecoderState & s, const RawSample & raw, std::string & message)
{
    if (!copyVisible(s.layout, s.info.width, s.info.height, raw.sample.Get(), s.frame.bytes, message))
    {
        s.hasFrame = false;
        return false;
    }
    s.frame.timestamp = toSeconds(raw.time - s.startTime);
    s.frame.duration = raw.duration > 0 ? toSeconds(raw.duration) : 1.0 / s.info.fps;
    s.frame.index = static_cast<int64_t>(std::llround(s.frame.timestamp * s.info.fps));
    s.hasFrame = true;
    return true;
}

// Moves the reader to `seconds` (relative to the first frame). False (message set) on failure.
bool seekTo(MfDecoderState & s, double seconds, std::string & message)
{
    PROPVARIANT position;
    PropVariantInit(&position);
    position.vt = VT_I8;
    position.hVal.QuadPart = s.startTime + static_cast<LONGLONG>(std::llround(seconds * kHundredNsPerSecond));
    const HRESULT hr = s.reader->SetCurrentPosition(GUID_NULL, position);
    PropVariantClear(&position);
    s.ahead = RawSample();
    s.hasAhead = false;
    if (FAILED(hr))
    {
        message = "seeking failed: " + mfHresultText(hr);
        return false;
    }
    ++s.seekCount;
    return true;
}

// Reads samples from the current position until the target is decided. A sample that covers
// the target is chosen. A sample that ends before the target is kept as the candidate; when a
// later sample starts after the target (or the stream ends), the candidate is chosen. A sample
// that starts after the target is kept for the next call.
Outcome decodeUntil(MfDecoderState & s, double target, double tol, bool seeking, RawSample & chosen,
                    std::string & message)
{
    const double frameDuration = 1.0 / s.info.fps;
    RawSample pending;
    bool hasPending = false;
    for (;;)
    {
        RawSample sample;
        const int got = readSample(s, sample, message);
        if (got < 0)
        {
            return Outcome::Error;
        }
        if (got == 0)
        {
            if (hasPending)
            {
                chosen = std::move(pending);
                return Outcome::Chosen;
            }
            return seeking ? Outcome::Beyond : Outcome::Hold;
        }

        const double start = toSeconds(sample.time - s.startTime);
        const double duration = sample.duration > 0 ? toSeconds(sample.duration) : frameDuration;
        if (start > target + tol)
        {
            s.ahead = std::move(sample);
            s.hasAhead = true;
            if (hasPending)
            {
                chosen = std::move(pending);
                return Outcome::Chosen;
            }
            return seeking ? Outcome::Beyond : Outcome::Hold;
        }
        if (start + duration - tol > target)
        {
            chosen = std::move(sample);
            return Outcome::Chosen;
        }
        pending = std::move(sample);
        hasPending = true;
    }
}

// The frame for a frame index (see MfDecoder::readFrame). Returns the current frame or nullptr.
const Nv12Frame * decodeFrame(MfDecoderState & s, int64_t requested, std::string & message)
{
    const int64_t last = std::max<int64_t>(0, s.info.frameCount - 1);
    const int64_t index = std::clamp<int64_t>(requested, 0, last);
    const double fps = s.info.fps;
    const double target = static_cast<double>(index) / fps;
    const double tol = 0.05 / fps;
    if (s.hasFrame && coversTime(s.frame.timestamp, s.frame.duration, target, tol))
    {
        return &s.frame;
    }

    const bool seek = !s.hasFrame || index < s.frame.index || index - s.frame.index > kSeekAheadFrames;
    if (seek)
    {
        // A first sample after the seek that is already past the target sends the seek further back.
        for (double back = 0.0;; back += 1.0)
        {
            const double from = std::max(0.0, target - back);
            if (!seekTo(s, from, message))
            {
                return nullptr;
            }
            RawSample chosen;
            const Outcome outcome = decodeUntil(s, target, tol, true, chosen, message);
            if (outcome == Outcome::Error)
            {
                return nullptr;
            }
            if (outcome == Outcome::Chosen)
            {
                return takeFrame(s, chosen, message) ? &s.frame : nullptr;
            }
            if (from <= 0.0)
            {
                char text[64];
                std::snprintf(text, sizeof(text), "no video frame at %.3f s", target);
                message = text;
                return nullptr;
            }
        }
    }

    RawSample chosen;
    const Outcome outcome = decodeUntil(s, target, tol, false, chosen, message);
    if (outcome == Outcome::Error)
    {
        return nullptr;
    }
    if (outcome == Outcome::Chosen)
    {
        return takeFrame(s, chosen, message) ? &s.frame : nullptr;
    }
    return &s.frame;
}

} // namespace

std::unique_ptr<MfDecoder> MfDecoder::open(const std::string & utf8Path, VideoOpenCode & code, std::string & message)
{
    code = VideoOpenCode::Failed;
    message.clear();
    const bool webm = endsWithIgnoreCase(utf8Path, ".webm");

    std::error_code ec;
    if (!std::filesystem::exists(pathFromUtf8(utf8Path), ec))
    {
        code = VideoOpenCode::Missing;
        message = "file not found: " + utf8Path;
        return nullptr;
    }

    auto state = std::make_unique<MfDecoderState>();
    if (!state->runtime.ok())
    {
        code = VideoOpenCode::Unsupported;
        message = state->runtime.error();
        return nullptr;
    }

    ComPtr<IMFAttributes> attributes;
    HRESULT hr = MFCreateAttributes(&attributes, 3);
    if (SUCCEEDED(hr))
    {
        // Software decode into NV12 host memory: no video processor, no DXVA, no device manager.
        attributes->SetUINT32(MF_SOURCE_READER_ENABLE_VIDEO_PROCESSING, FALSE);
        attributes->SetUINT32(MF_READWRITE_DISABLE_CONVERTERS, FALSE);
        attributes->SetUINT32(MF_SOURCE_READER_DISABLE_DXVA, TRUE);
    }
    const std::wstring wide = utf8ToWide(utf8Path);
    if (SUCCEEDED(hr))
    {
        hr = MFCreateSourceReaderFromURL(wide.c_str(), attributes.Get(), &state->reader);
    }
    if (FAILED(hr))
    {
        if (webm)
        {
            code = VideoOpenCode::WebmNeedsVp9;
            message = kWebmMessage;
        }
        else
        {
            message = "the video could not be opened: " + mfHresultText(hr);
        }
        return nullptr;
    }

    ComPtr<IMFMediaType> nativeType;
    if (FAILED(state->reader->GetNativeMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, 0, &nativeType)))
    {
        code = VideoOpenCode::NoVideoStream;
        message = "no video stream";
        return nullptr;
    }

    // Only the first video stream is decoded; everything else (audio included) is deselected.
    hr = state->reader->SetStreamSelection(MF_SOURCE_READER_ALL_STREAMS, FALSE);
    if (SUCCEEDED(hr))
    {
        hr = state->reader->SetStreamSelection(MF_SOURCE_READER_FIRST_VIDEO_STREAM, TRUE);
    }
    if (FAILED(hr))
    {
        message = "the video stream could not be selected: " + mfHresultText(hr);
        return nullptr;
    }

    ComPtr<IMFMediaType> nv12;
    hr = MFCreateMediaType(&nv12);
    if (SUCCEEDED(hr))
    {
        nv12->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
        nv12->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_NV12);
        hr = state->reader->SetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, nullptr, nv12.Get());
    }
    if (FAILED(hr))
    {
        if (webm)
        {
            code = VideoOpenCode::WebmNeedsVp9;
            message = kWebmMessage;
        }
        else
        {
            code = VideoOpenCode::Unsupported;
            message = "the video is not 8-bit 4:2:0 (the decoder cannot give NV12): " + mfHresultText(hr);
        }
        return nullptr;
    }

    ComPtr<IMFMediaType> currentType;
    uint32_t visibleWidth = 0;
    uint32_t visibleHeight = 0;
    if (FAILED(state->reader->GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, &currentType)) ||
        !readLayout(currentType.Get(), state->layout, visibleWidth, visibleHeight))
    {
        code = VideoOpenCode::Unsupported;
        message = "the video frame size is not known";
        return nullptr;
    }
    if (visibleWidth < 2 || visibleHeight < 2 || (visibleWidth % 2) != 0 || (visibleHeight % 2) != 0)
    {
        code = VideoOpenCode::Unsupported;
        message = "video size " + std::to_string(visibleWidth) + "x" + std::to_string(visibleHeight) +
                  " is not supported (needs even width and height)";
        return nullptr;
    }

    VideoInfo & info = state->info;
    info.width = visibleWidth;
    info.height = visibleHeight;
    info.fps = frameRateOf(currentType.Get());
    const int64_t durationHundredNs = presentationDuration(state->reader.Get());
    info.durationSeconds = durationHundredNs > 0 ? toSeconds(durationHundredNs) : 1.0 / info.fps;
    info.frameCount = std::max<int64_t>(1, static_cast<int64_t>(std::llround(info.durationSeconds * info.fps)));
    info.yuvMatrix = uint32AttributeWithFallback(currentType.Get(), nativeType.Get(), MF_MT_YUV_MATRIX);
    info.nominalRange = uint32AttributeWithFallback(currentType.Get(), nativeType.Get(), MF_MT_VIDEO_NOMINAL_RANGE);
    info.rotationDegrees = uint32AttributeWithFallback(currentType.Get(), nativeType.Get(), MF_MT_VIDEO_ROTATION);
    info.hasAudio = hasAudioStream(state->reader.Get());

    // The first frame surfaces decode errors now and sets the time origin.
    RawSample first;
    std::string readMessage;
    const int got = readSample(*state, first, readMessage);
    if (got < 0)
    {
        if (webm)
        {
            code = VideoOpenCode::WebmNeedsVp9;
            message = kWebmMessage;
        }
        else
        {
            message = readMessage;
        }
        return nullptr;
    }
    if (got == 0)
    {
        message = "the video has no frames";
        return nullptr;
    }
    state->startTime = first.time;
    if (!takeFrame(*state, first, readMessage))
    {
        if (webm)
        {
            code = VideoOpenCode::WebmNeedsVp9;
            message = kWebmMessage;
        }
        else
        {
            message = readMessage;
        }
        return nullptr;
    }

    // The frame rate attribute is not always right (a WEBM may report a different rate than its
    // timestamps show). Measure the spacing of the next few samples and trust it when it differs by
    // more than 5 percent, snapped to a common rate when within 2 percent of one.
    {
        std::vector<double> times;
        times.push_back(0.0);
        RawSample probe;
        std::string probeMessage;
        while (times.size() < 8 && readSample(*state, probe, probeMessage) == 1)
        {
            times.push_back(toSeconds(probe.time - state->startTime));
        }
        if (times.size() >= 3)
        {
            std::vector<double> gaps;
            for (size_t i = 1; i < times.size(); ++i)
            {
                gaps.push_back(times[i] - times[i - 1]);
            }
            std::sort(gaps.begin(), gaps.end());
            const double gap = gaps[gaps.size() / 2];
            if (gap > 0.0)
            {
                const double measured = 1.0 / gap;
                if (std::fabs(measured / info.fps - 1.0) > 0.05)
                {
                    static const double kCommon[] = {23.976, 24.0, 25.0, 29.97, 30.0, 48.0, 50.0, 59.94, 60.0};
                    double chosen = std::round(measured * 1000.0) / 1000.0;
                    double bestError = 0.02;
                    for (const double rate : kCommon)
                    {
                        const double error = std::fabs(measured / rate - 1.0);
                        if (error < bestError)
                        {
                            bestError = error;
                            chosen = rate;
                        }
                    }
                    info.fps = chosen;
                    info.frameCount = std::max<int64_t>(1, static_cast<int64_t>(std::llround(info.durationSeconds * info.fps)));
                }
            }
        }
        // Back to the first frame, which stays the current frame.
        state->ahead = RawSample();
        state->hasAhead = false;
        if (times.size() > 1 && !seekTo(*state, 0.0, probeMessage))
        {
            message = probeMessage;
            return nullptr;
        }
        state->seekCount = 0;
    }

    code = VideoOpenCode::Ok;
    return std::unique_ptr<MfDecoder>(new MfDecoder(std::move(state)));
}

MfDecoder::MfDecoder(std::unique_ptr<MfDecoderState> state) : m_state(std::move(state)) {}

MfDecoder::~MfDecoder() = default;

const VideoInfo & MfDecoder::info() const
{
    return m_state->info;
}

const Nv12Frame * MfDecoder::readFrame(int64_t index, std::string & message)
{
    const auto started = std::chrono::steady_clock::now();
    m_state->lastDecodedSamples = 0;
    const Nv12Frame * frame = decodeFrame(*m_state, index, message);
    m_state->lastDecodeSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return frame;
}

uint64_t MfDecoder::hostBytes() const
{
    const VideoInfo & info = m_state->info;
    const uint64_t frameBytes = static_cast<uint64_t>(info.width) * info.height * 3 / 2;
    return static_cast<uint64_t>(m_state->frame.bytes.capacity()) + 4 * frameBytes;
}

double MfDecoder::lastDecodeSeconds() const
{
    return m_state->lastDecodeSeconds;
}

uint32_t MfDecoder::lastDecodedSamples() const
{
    return m_state->lastDecodedSamples;
}

uint32_t MfDecoder::seekCount() const
{
    return m_state->seekCount;
}

std::string mfHresultText(int32_t hresult)
{
    char code[16];
    std::snprintf(code, sizeof(code), "0x%08X", static_cast<unsigned int>(static_cast<uint32_t>(hresult)));
    std::string text(code);

    wchar_t * buffer = nullptr;
    const DWORD length = FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
                                            FORMAT_MESSAGE_IGNORE_INSERTS,
                                        nullptr, static_cast<DWORD>(hresult),
                                        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                                        reinterpret_cast<LPWSTR>(&buffer), 0, nullptr);
    if (length > 0 && buffer != nullptr)
    {
        std::wstring system(buffer, length);
        LocalFree(buffer);
        while (!system.empty() &&
               (system.back() == L'\r' || system.back() == L'\n' || system.back() == L' ' || system.back() == L'.'))
        {
            system.pop_back();
        }
        if (!system.empty())
        {
            text += " (" + wideToUtf8(system) + ")";
        }
    }
    return text;
}

MfRuntime::MfRuntime()
{
    if (!systemDllLoads(L"mfplat.dll") || !systemDllLoads(L"mfreadwrite.dll"))
    {
        m_error = kNoMediaFoundationMessage;
        return;
    }

    // RPC_E_CHANGED_MODE: the thread already has COM in the other apartment. COM works, and it is not ours to uninitialize.
    const HRESULT com = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    if (com == S_OK || com == S_FALSE)
    {
        m_uninitializeCom = true;
    }
    else if (com != RPC_E_CHANGED_MODE)
    {
        m_error = "COM did not start: " + mfHresultText(com);
        return;
    }

    const HRESULT mf = MFStartup(MF_VERSION);
    if (FAILED(mf))
    {
        m_error = "Media Foundation did not start: " + mfHresultText(mf);
        if (m_uninitializeCom)
        {
            CoUninitialize();
            m_uninitializeCom = false;
        }
        return;
    }
    m_mfStarted = true;
    m_ok = true;
}

MfRuntime::~MfRuntime()
{
    if (m_mfStarted)
    {
        MFShutdown();
    }
    if (m_uninitializeCom)
    {
        CoUninitialize();
    }
}

bool MfRuntime::ok() const
{
    return m_ok;
}

const std::string & MfRuntime::error() const
{
    return m_error;
}

} // namespace sabercat::compare::gui
