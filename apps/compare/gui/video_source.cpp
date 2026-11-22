// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "video_source.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <system_error>
#include <utility>

#include <vulkan/vulkan.h>

#include "gpu_context.h"
#include "nv12_converter.h"
#include "scene_source.h"
#include "sabercat/buffer.h"
#include "sabercat/device.h"

namespace sabercat::compare::gui
{

namespace
{

using Clock = std::chrono::steady_clock;

// Bytes per RGBA32F texel (the color buffer's stride).
constexpr uint32_t kTexelBytes = static_cast<uint32_t>(4 * sizeof(float));
// The NV12 buffer is read as 32-bit words by the shader (its stride is one word).
constexpr uint32_t kNv12Stride = static_cast<uint32_t>(sizeof(uint32_t));
// The side limit a video gets on any device: the device's 2D image limit when larger.
constexpr uint32_t kMinSideLimit = 4096;
// The largest time offset, in seconds (one day either way).
constexpr double kMaxTimeOffset = 86400.0;
// Device memory asked for beyond the buffers themselves, in percent.
constexpr uint64_t kBudgetMarginPercent = 25;

// A UTF-8 string as a path, without the deprecated u8path().
std::filesystem::path pathFromUtf8(const std::string & utf8)
{
    return std::filesystem::path(std::u8string(reinterpret_cast<const char8_t *>(utf8.data()), utf8.size()));
}

std::string lowerAscii(std::string text)
{
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return text;
}

// The NV12 bytes of one frame: the Y plane (width * height) and the CbCr plane (width * height / 2).
uint64_t nv12FrameBytes(uint32_t width, uint32_t height)
{
    return static_cast<uint64_t>(width) * height * 3 / 2;
}

// The NV12 device buffer is rounded up to whole words.
uint64_t roundUpToWord(uint64_t bytes)
{
    return (bytes + 3) / 4 * 4;
}

// The largest side a video may have on this device: at least kMinSideLimit.
uint32_t sideLimit(const sabercat::Device & device)
{
    return std::max(kMinSideLimit, device.properties().limits.maxImageDimension2D);
}

std::string formatSeconds(double seconds)
{
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.3f", seconds);
    return buffer;
}

double secondsBetween(Clock::time_point from, Clock::time_point to)
{
    return std::chrono::duration<double>(to - from).count();
}

} // namespace

bool isVideoFile(const std::string & utf8Path)
{
    // Only the file name counts, so a folder named like an extension does not match.
    const size_t separator = utf8Path.find_last_of("/\\");
    const std::string name =
        lowerAscii(separator == std::string::npos ? utf8Path : utf8Path.substr(separator + 1));
    const size_t dot = name.find_last_of('.');
    if (dot == std::string::npos)
    {
        return false;
    }
    const std::string ext = name.substr(dot);
    return ext == ".mp4" || ext == ".m4v" || ext == ".webm";
}

VideoSource::VideoSource(GpuContext & context, const std::string & utf8Path) : m_context(context), m_path(utf8Path)
{
    try
    {
        std::error_code ec;
        if (!std::filesystem::exists(pathFromUtf8(utf8Path), ec))
        {
            fail(SourceStatus::Missing, "file not found: " + utf8Path);
            return;
        }

        VideoOpenCode code = VideoOpenCode::Ok;
        std::string message;
        std::unique_ptr<MfDecoder> decoder = MfDecoder::open(utf8Path, code, message);
        if (!decoder)
        {
            fail(code == VideoOpenCode::Missing ? SourceStatus::Missing : SourceStatus::Error, message);
            return;
        }
        m_decoder = std::move(decoder);
        m_info = m_decoder->info();

        const uint32_t videoWidth = m_info.width;
        const uint32_t videoHeight = m_info.height;
        const uint32_t limit = sideLimit(context.device());
        if (videoWidth > limit || videoHeight > limit)
        {
            fail(SourceStatus::Error, "video size " + std::to_string(videoWidth) + "x" + std::to_string(videoHeight)
                     + " is above the limit of " + std::to_string(limit) + " pixels on a side");
            return;
        }

        // The memory check: the two device buffers plus a margin, before anything is created.
        const uint64_t nv12Bytes = roundUpToWord(nv12FrameBytes(videoWidth, videoHeight));
        const uint64_t colorBytes = static_cast<uint64_t>(videoWidth) * videoHeight * kTexelBytes;
        const uint64_t total = nv12Bytes + colorBytes;
        const uint64_t needed = total + total * kBudgetMarginPercent / 100;
        if (!deviceBudgetOk(context.device(), needed, "this video", message))
        {
            fail(SourceStatus::Error, message);
            return;
        }

        // Constant fps (no frame times): frame i is shown at i / fps. The stored trim is the whole clip.
        m_timeSource.frameTimes.clear();
        m_timeSource.fps = m_info.fps;
        m_timeSource.duration = m_info.durationSeconds;
        m_timeSource.trimIn = 0.0;
        m_timeSource.trimOut = m_info.durationSeconds;
        m_timeSource.timeOffset = 0.0;
        m_timeSource.frameCount = m_info.frameCount;
        m_timeSource.kind = TimeSourceKind::Video;
        m_timeSource.loop = false;
        try
        {
            validateSource(m_timeSource);
        }
        catch (const std::exception & e)
        {
            fail(SourceStatus::Error, std::string("video timing is not valid: ") + e.what());
            return;
        }

        sabercat::Device & device = context.device();
        m_nv12 = std::make_unique<sabercat::Buffer>(device, nv12Bytes);
        m_color = std::make_unique<sabercat::Buffer>(device, colorBytes);
        m_converter = std::make_unique<Nv12Converter>(context);

        // Open already decoded the first frame; readFrame(0) returns it without a seek.
        const Nv12Frame * frame = m_decoder->readFrame(0, message);
        if (frame == nullptr)
        {
            fail(SourceStatus::Error, message);
            return;
        }
        FrameTiming timing;
        uploadFrame(*frame, timing);
        m_lastTiming = {};
        m_frameIndex = frame->index;
        m_requestedIndex = 0;
        m_status = SourceStatus::Ready;
    }
    catch (const std::exception & e)
    {
        fail(SourceStatus::Error, e.what());
    }
    catch (...)
    {
        fail(SourceStatus::Error, "video could not be opened");
    }
}

VideoSource::~VideoSource() = default;

void VideoSource::fail(SourceStatus status, std::string message)
{
    m_color.reset();
    m_nv12.reset();
    m_converter.reset();
    m_decoder.reset();
    m_host = ImageData{};
    m_hostValid = false;
    m_info = VideoInfo{};
    m_status = status;
    m_message = std::move(message);
}

void VideoSource::uploadFrame(const Nv12Frame & frame, FrameTiming & timing)
{
    const uint64_t expected = nv12FrameBytes(m_info.width, m_info.height);
    if (static_cast<uint64_t>(frame.bytes.size()) != expected)
    {
        throw std::runtime_error("decoded frame has " + std::to_string(frame.bytes.size()) + " bytes, expected "
                                 + std::to_string(expected));
    }

    const Clock::time_point uploadStart = Clock::now();
    m_nv12->upload(frame.bytes.data(), frame.bytes.size());
    const Clock::time_point convertStart = Clock::now();
    m_converter->convert(m_nv12->view(kNv12Stride), m_color->view(kTexelBytes), m_info.width, m_info.height);
    const Clock::time_point convertEnd = Clock::now();

    timing.uploadSeconds = secondsBetween(uploadStart, convertStart);
    timing.convertSeconds = secondsBetween(convertStart, convertEnd);
}

BufferView VideoSource::color() const
{
    return (m_status == SourceStatus::Ready && m_color) ? m_color->view(kTexelBytes) : BufferView{};
}

bool VideoSource::update(double globalTime)
{
    if (m_status != SourceStatus::Ready)
    {
        return false;
    }
    const int64_t index = frameIndexAt(m_timeSource, globalTime);
    // The frame is kept while the index is the same; a failed decode is retried on the next call.
    if (!m_decodeFailed && index == m_requestedIndex)
    {
        return false;
    }

    try
    {
        std::string message;
        const Clock::time_point decodeStart = Clock::now();
        const Nv12Frame * frame = m_decoder->readFrame(index, message);
        const Clock::time_point decodeEnd = Clock::now();
        if (frame == nullptr)
        {
            m_decodeFailed = true;
            m_message = "video decode failed: " + message;
            return false;
        }

        FrameTiming timing;
        timing.decodeSeconds = secondsBetween(decodeStart, decodeEnd);
        uploadFrame(*frame, timing);

        m_lastTiming = timing;
        // At the end of the stream the decoder returns its last frame, which may be before the index.
        m_frameIndex = frame->index;
        m_requestedIndex = index;
        m_decodeFailed = false;
        m_message.clear();
        m_hostValid = false;
        return true;
    }
    catch (const std::exception & e)
    {
        m_decodeFailed = true;
        m_message = std::string("video decode failed: ") + e.what();
        return false;
    }
}

bool VideoSource::readPixel(int32_t x, int32_t y, float rgba[4]) const
{
    const uint32_t videoWidth = m_info.width;
    const uint32_t videoHeight = m_info.height;
    if (m_status != SourceStatus::Ready || !m_color || x < 0 || y < 0 || static_cast<uint32_t>(x) >= videoWidth
        || static_cast<uint32_t>(y) >= videoHeight)
    {
        return false;
    }
    // readBuffer copies from view.offset, so the pixel's own view carries the offset.
    const VkDeviceSize offset =
        (static_cast<VkDeviceSize>(y) * videoWidth + static_cast<VkDeviceSize>(x)) * kTexelBytes;
    const BufferView pixel{m_color->handle(), offset, kTexelBytes, kTexelBytes};
    try
    {
        m_context.readBuffer(pixel, rgba, kTexelBytes);
    }
    catch (const std::exception &)
    {
        return false;
    }
    return true;
}

const ImageData * VideoSource::hostData() const
{
    if (m_status != SourceStatus::Ready || !m_color)
    {
        return nullptr;
    }
    if (m_hostValid)
    {
        return &m_host;
    }
    const size_t pixels = static_cast<size_t>(m_info.width) * m_info.height;
    try
    {
        ImageData host;
        host.rgba.resize(pixels * 4);
        m_context.readBuffer(m_color->view(kTexelBytes), host.rgba.data(), host.rgba.size() * sizeof(float));
        host.width = m_info.width;
        host.height = m_info.height;
        host.bitDepth = 8;
        host.encoding = Encoding::Display;
        host.hasAlpha = false;
        host.premultiplied = false;
        m_host = std::move(host);
    }
    catch (const std::exception &)
    {
        m_host = ImageData{};
        return nullptr;
    }
    m_hostValid = true;
    return &m_host;
}

bool VideoSource::setTrim(double trimIn, double trimOut)
{
    if (m_status != SourceStatus::Ready)
    {
        return false;
    }
    if (!std::isfinite(trimIn) || !std::isfinite(trimOut))
    {
        m_message = "trim values must be finite numbers";
        return false;
    }
    const double duration = m_info.durationSeconds;
    const bool toEnd = trimOut < 0.0;
    const double resolvedOut = toEnd ? duration : trimOut;
    // Written so that a NaN fails each comparison.
    if (!(trimIn >= 0.0) || !(trimIn < resolvedOut) || !(resolvedOut <= duration))
    {
        m_message = "trim must satisfy 0 <= in < out <= " + formatSeconds(duration) + " s";
        return false;
    }

    TimeSource next = m_timeSource;
    next.trimIn = trimIn;
    next.trimOut = resolvedOut;
    try
    {
        validateSource(next);
    }
    catch (const std::exception & e)
    {
        m_message = e.what();
        return false;
    }
    m_timeSource = next;
    m_trimToEnd = toEnd;
    m_message.clear();
    return true;
}

void VideoSource::setLoop(bool loop)
{
    m_timeSource.loop = loop;
}

bool VideoSource::setTimeOffset(double seconds)
{
    if (m_status != SourceStatus::Ready)
    {
        return false;
    }
    if (!std::isfinite(seconds) || std::fabs(seconds) > kMaxTimeOffset)
    {
        m_message = "time offset must be a finite number of seconds, at most 86400 in either direction";
        return false;
    }
    m_timeSource.timeOffset = seconds;
    m_message.clear();
    return true;
}

double VideoSource::trimIn() const
{
    return m_timeSource.trimIn;
}

double VideoSource::trimOut() const
{
    return m_timeSource.trimOut;
}

bool VideoSource::trimToEnd() const
{
    return m_trimToEnd;
}

bool VideoSource::loop() const
{
    return m_timeSource.loop;
}

double VideoSource::timeOffset() const
{
    return m_timeSource.timeOffset;
}

const TimeSource & VideoSource::timeSource() const
{
    return m_timeSource;
}

double VideoSource::fps() const
{
    return m_info.fps;
}

int64_t VideoSource::frameCount() const
{
    return m_info.frameCount;
}

double VideoSource::durationSeconds() const
{
    return m_info.durationSeconds;
}

int64_t VideoSource::frameIndex() const
{
    return m_frameIndex;
}

double VideoSource::sourceTime() const
{
    return m_info.fps > 0.0 ? static_cast<double>(m_frameIndex) / m_info.fps : 0.0;
}

const VideoInfo & VideoSource::videoInfo() const
{
    return m_info;
}

uint64_t VideoSource::deviceBytes() const
{
    uint64_t bytes = 0;
    if (m_nv12)
    {
        bytes += m_nv12->size();
    }
    if (m_color)
    {
        bytes += m_color->size();
    }
    return bytes;
}

uint64_t VideoSource::hostBytes() const
{
    uint64_t bytes = m_decoder ? m_decoder->hostBytes() : 0;
    if (m_hostValid)
    {
        bytes += m_host.rgba.size() * sizeof(float);
    }
    return bytes;
}

VideoSource::FrameTiming VideoSource::lastTiming() const
{
    return m_lastTiming;
}

MfDecoder * VideoSource::decoder()
{
    return m_decoder.get();
}

} // namespace sabercat::compare::gui
