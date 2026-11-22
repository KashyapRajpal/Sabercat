// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "mf_clip_writer.h"

#include <cstddef>
#include <cstring>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mf_decoder.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <objbase.h>
#include <oaidl.h>
#include <codecapi.h>
#include <icodecapi.h>
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

// The index bits of the frame pattern bound the clip length (see kClipIndexBits).
constexpr uint32_t kMaxClipFrames = 1u << kClipIndexBits;
// The strip of cells at the top-left corner: kClipIndexBits cells of kClipCellSize pixels.
constexpr uint32_t kStripWidth = kClipIndexBits * kClipCellSize;
// Media Foundation time is in 100 ns units.
constexpr uint64_t kHundredNsPerSecond = 10000000;

// Luma of a cell that is set (white) and not set (black), and the threshold that reads them back.
constexpr uint8_t kCellWhite = 235;
constexpr uint8_t kCellBlack = 16;
constexpr uint8_t kCellThreshold = 125;

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

// A UTF-8 string as a path, without the deprecated u8path().
std::filesystem::path pathFromUtf8(const std::string & utf8)
{
    return std::filesystem::path(std::u8string(reinterpret_cast<const char8_t *>(utf8.data()), utf8.size()));
}

// Rejects a spec that cannot be written at all (a caller error, not an encoder problem).
void validateSpec(const ClipSpec & spec)
{
    if (spec.path.empty())
    {
        throw std::invalid_argument("clip path is empty");
    }
    if (spec.width < kStripWidth || spec.height < kClipCellSize || (spec.width % 2) != 0 || (spec.height % 2) != 0)
    {
        throw std::invalid_argument("clip size " + std::to_string(spec.width) + "x" + std::to_string(spec.height) +
                                    " is not supported (needs width >= 192, height >= 16, both even)");
    }
    if (spec.fpsNumerator == 0 || spec.fpsDenominator == 0)
    {
        throw std::invalid_argument("clip frame rate must be positive");
    }
    if (spec.frameCount == 0 || spec.frameCount > kMaxClipFrames)
    {
        throw std::invalid_argument("clip frame count must be between 1 and 4096");
    }
    if (spec.keyframeInterval == 0 || spec.bitrate == 0)
    {
        throw std::invalid_argument("clip keyframe interval and bitrate must be positive");
    }
    if (spec.rotationDegrees != 0 && spec.rotationDegrees != 90 && spec.rotationDegrees != 180 &&
        spec.rotationDegrees != 270)
    {
        throw std::invalid_argument("clip rotation must be 0, 90, 180 or 270");
    }
}

// Sample time of frame `index` in 100 ns: index * den / num seconds, rounded to the nearest unit and not
// accumulated from the previous frame.
LONGLONG frameTime(uint64_t index, uint32_t numerator, uint32_t denominator)
{
    const uint64_t units = index * kHundredNsPerSecond * denominator;
    return static_cast<LONGLONG>((units + numerator / 2) / numerator);
}

// The GOP size through the encoder's ICodecAPI, when the sink writer exposes it. Best effort.
void setGopSize(IMFSinkWriter * writer, DWORD streamIndex, uint32_t keyframeInterval)
{
    ComPtr<ICodecAPI> codec;
    if (FAILED(writer->GetServiceForStream(streamIndex, GUID_NULL, IID_PPV_ARGS(&codec))))
    {
        return;
    }
    VARIANT value;
    VariantInit(&value);
    value.vt = VT_UI4;
    value.ulVal = keyframeInterval;
    codec->SetValue(&CODECAPI_AVEncMPVGOPSize, &value);
    VariantClear(&value);
}

// Adds the output stream (codec, bitrate, size, rate, GOP) and sets the NV12 input type.
HRESULT configureStream(IMFSinkWriter * writer, const ClipSpec & spec, const GUID & codec, bool h264,
                        DWORD & streamIndex)
{
    ComPtr<IMFMediaType> output;
    HRESULT hr = MFCreateMediaType(&output);
    if (FAILED(hr))
    {
        return hr;
    }
    output->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
    output->SetGUID(MF_MT_SUBTYPE, codec);
    output->SetUINT32(MF_MT_AVG_BITRATE, spec.bitrate);
    MFSetAttributeSize(output.Get(), MF_MT_FRAME_SIZE, spec.width, spec.height);
    MFSetAttributeRatio(output.Get(), MF_MT_FRAME_RATE, spec.fpsNumerator, spec.fpsDenominator);
    MFSetAttributeRatio(output.Get(), MF_MT_PIXEL_ASPECT_RATIO, 1, 1);
    output->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
    output->SetUINT32(MF_MT_MAX_KEYFRAME_SPACING, spec.keyframeInterval);
    if (h264)
    {
        output->SetUINT32(MF_MT_MPEG2_PROFILE, eAVEncH264VProfile_Main);
    }
    if (spec.rotationDegrees != 0)
    {
        output->SetUINT32(MF_MT_VIDEO_ROTATION, spec.rotationDegrees);
    }
    hr = writer->AddStream(output.Get(), &streamIndex);
    if (FAILED(hr))
    {
        return hr;
    }

    ComPtr<IMFMediaType> input;
    hr = MFCreateMediaType(&input);
    if (FAILED(hr))
    {
        return hr;
    }
    input->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video);
    input->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_NV12);
    MFSetAttributeSize(input.Get(), MF_MT_FRAME_SIZE, spec.width, spec.height);
    MFSetAttributeRatio(input.Get(), MF_MT_FRAME_RATE, spec.fpsNumerator, spec.fpsDenominator);
    MFSetAttributeRatio(input.Get(), MF_MT_PIXEL_ASPECT_RATIO, 1, 1);
    input->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive);
    input->SetUINT32(MF_MT_DEFAULT_STRIDE, spec.width);
    hr = writer->SetInputMediaType(streamIndex, input.Get(), nullptr);
    if (FAILED(hr))
    {
        return hr;
    }

    if (h264)
    {
        setGopSize(writer, streamIndex, spec.keyframeInterval);
    }
    return S_OK;
}

// Writes every frame of the clip. Returns the first failing HRESULT, or S_OK.
HRESULT writeFrames(IMFSinkWriter * writer, DWORD streamIndex, const ClipSpec & spec)
{
    const size_t frameBytes = static_cast<size_t>(spec.width) * spec.height * 3 / 2;
    std::vector<uint8_t> nv12;
    for (uint32_t index = 0; index < spec.frameCount; ++index)
    {
        makeClipFrameNv12(spec.width, spec.height, index, nv12);

        ComPtr<IMFMediaBuffer> buffer;
        HRESULT hr = MFCreateMemoryBuffer(static_cast<DWORD>(frameBytes), &buffer);
        if (FAILED(hr))
        {
            return hr;
        }
        BYTE * data = nullptr;
        DWORD maxLength = 0;
        DWORD currentLength = 0;
        hr = buffer->Lock(&data, &maxLength, &currentLength);
        if (FAILED(hr))
        {
            return hr;
        }
        std::memcpy(data, nv12.data(), frameBytes);
        buffer->Unlock();
        buffer->SetCurrentLength(static_cast<DWORD>(frameBytes));

        ComPtr<IMFSample> sample;
        hr = MFCreateSample(&sample);
        if (FAILED(hr))
        {
            return hr;
        }
        sample->AddBuffer(buffer.Get());
        const LONGLONG start = frameTime(index, spec.fpsNumerator, spec.fpsDenominator);
        const LONGLONG end = frameTime(static_cast<uint64_t>(index) + 1, spec.fpsNumerator, spec.fpsDenominator);
        sample->SetSampleTime(start);
        sample->SetSampleDuration(end - start);
        hr = writer->WriteSample(streamIndex, sample.Get());
        if (FAILED(hr))
        {
            return hr;
        }
    }
    return S_OK;
}

// Writes the clip with the given codec subtype. Returns an empty string on success, else the failure text.
// A partial file is removed. The Media Foundation start-up is inside, so the runtime outlives the writer.
std::string writeClip(const ClipSpec & spec, const GUID & codec, bool h264)
{
    MfRuntime runtime;
    if (!runtime.ok())
    {
        return runtime.error();
    }

    const std::wstring wide = utf8ToWide(spec.path);
    ComPtr<IMFAttributes> attributes;
    HRESULT hr = MFCreateAttributes(&attributes, 2);
    if (SUCCEEDED(hr))
    {
        // Software encoder (deterministic output) and no throttling, so WriteSample never waits for a clock.
        attributes->SetUINT32(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, FALSE);
        attributes->SetUINT32(MF_SINK_WRITER_DISABLE_THROTTLING, TRUE);
    }
    ComPtr<IMFSinkWriter> writer;
    if (SUCCEEDED(hr))
    {
        hr = MFCreateSinkWriterFromURL(wide.c_str(), nullptr, attributes.Get(), &writer);
    }
    DWORD streamIndex = 0;
    if (SUCCEEDED(hr))
    {
        hr = configureStream(writer.Get(), spec, codec, h264, streamIndex);
    }
    bool began = false;
    if (SUCCEEDED(hr))
    {
        hr = writer->BeginWriting();
        began = SUCCEEDED(hr);
    }
    if (SUCCEEDED(hr))
    {
        hr = writeFrames(writer.Get(), streamIndex, spec);
    }

    // Finalize even after a write error (best effort), so the container is closed before the file is removed.
    HRESULT finish = S_OK;
    if (began)
    {
        finish = writer->Finalize();
    }
    writer.Reset();

    if (FAILED(hr) || FAILED(finish))
    {
        std::error_code ec;
        std::filesystem::remove(pathFromUtf8(spec.path), ec);
        return "clip writing failed: " + mfHresultText(FAILED(hr) ? hr : finish);
    }
    return std::string();
}

} // namespace

void makeClipFrameNv12(uint32_t width, uint32_t height, uint32_t index, std::vector<uint8_t> & nv12)
{
    if (width < kStripWidth || height < kClipCellSize || (width % 2) != 0 || (height % 2) != 0)
    {
        throw std::invalid_argument("clip frame size must be at least 192x16 and even");
    }
    if (index >= kMaxClipFrames)
    {
        throw std::invalid_argument("clip frame index must be in [0, 4095]");
    }

    const size_t w = width;
    const size_t h = height;
    nv12.assign(w * h + w * (h / 2), 0);
    uint8_t * luma = nv12.data();
    uint8_t * chroma = luma + w * h;

    // Background: a luma ramp (40..158) and a per-frame chroma tint.
    for (size_t row = 0; row < h; ++row)
    {
        for (size_t col = 0; col < w; ++col)
        {
            luma[row * w + col] = static_cast<uint8_t>(40 + (col * 100) / w + (row * 20) / h);
        }
    }
    const uint8_t tintU = static_cast<uint8_t>(120 + index % 16);
    const uint8_t tintV = static_cast<uint8_t>(136 - index % 16);
    for (size_t row = 0; row < h / 2; ++row)
    {
        for (size_t col = 0; col < w / 2; ++col)
        {
            chroma[row * w + 2 * col] = tintU;
            chroma[row * w + 2 * col + 1] = tintV;
        }
    }

    // The index strip: cell b is white when bit b is set, black otherwise.
    for (uint32_t bit = 0; bit < kClipIndexBits; ++bit)
    {
        const uint8_t value = ((index >> bit) & 1u) != 0 ? kCellWhite : kCellBlack;
        const size_t x0 = static_cast<size_t>(bit) * kClipCellSize;
        for (size_t row = 0; row < kClipCellSize; ++row)
        {
            for (size_t col = x0; col < x0 + kClipCellSize; ++col)
            {
                luma[row * w + col] = value;
            }
        }
    }
    // Neutral chroma under the strip (its luma rows are 0..15, so chroma rows 0..7).
    for (size_t row = 0; row < kClipCellSize / 2; ++row)
    {
        for (size_t col = 0; col < kStripWidth / 2; ++col)
        {
            chroma[row * w + 2 * col] = 128;
            chroma[row * w + 2 * col + 1] = 128;
        }
    }
}

uint32_t decodeClipIndexFromLuma(const uint8_t * luma, uint32_t width)
{
    uint32_t index = 0;
    const size_t centerY = kClipCellSize / 2;
    for (uint32_t bit = 0; bit < kClipIndexBits; ++bit)
    {
        const size_t centerX = static_cast<size_t>(bit) * kClipCellSize + kClipCellSize / 2;
        if (luma[centerY * width + centerX] > kCellThreshold)
        {
            index |= 1u << bit;
        }
    }
    return index;
}

uint32_t decodeClipIndexFromRgba(const float * rgba, uint32_t width)
{
    uint32_t index = 0;
    const size_t centerY = kClipCellSize / 2;
    for (uint32_t bit = 0; bit < kClipIndexBits; ++bit)
    {
        const size_t centerX = static_cast<size_t>(bit) * kClipCellSize + kClipCellSize / 2;
        const float * pixel = rgba + (centerY * width + centerX) * 4;
        const float luma = 0.2126f * pixel[0] + 0.7152f * pixel[1] + 0.0722f * pixel[2];
        if (luma > 0.5f)
        {
            index |= 1u << bit;
        }
    }
    return index;
}

void writeClipMp4(const ClipSpec & spec)
{
    validateSpec(spec);
    const std::string error = writeClip(spec, MFVideoFormat_H264, true);
    if (!error.empty())
    {
        throw std::runtime_error(error);
    }
}

bool tryWriteClipWebm(const ClipSpec & spec)
{
    validateSpec(spec);
    try
    {
        return writeClip(spec, MFVideoFormat_VP90, false).empty();
    }
    catch (const std::exception &)
    {
        return false;
    }
}

} // namespace sabercat::compare::gui
