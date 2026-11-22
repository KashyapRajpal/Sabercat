// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "image_source.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <utility>

#include <vulkan/vulkan.h>

#include "compare_layout.h"
#include "gpu_context.h"
#include "image_io.h"
#include "sabercat/buffer.h"
#include "sabercat/device.h"

namespace sabercat::compare::gui
{

namespace
{

// Bytes per RGBA32F texel.
constexpr size_t kTexelBytes = 4 * sizeof(float);
// Bytes per depth sample (one float per pixel).
constexpr size_t kDepthBytes = sizeof(float);

// A UTF-8 string as a path, without the deprecated u8path().
std::filesystem::path pathFromUtf8(const std::string & utf8)
{
    return std::filesystem::path(std::u8string(reinterpret_cast<const char8_t *>(utf8.data()), utf8.size()));
}

} // namespace

ImageSource::ImageSource(GpuContext & context, const std::string & utf8Path) : m_path(utf8Path)
{
    try
    {
        std::error_code ec;
        if (!std::filesystem::exists(pathFromUtf8(utf8Path), ec))
        {
            fail(SourceStatus::Missing, "file not found: " + utf8Path);
            return;
        }
        m_data = loadImage(utf8Path);
        uploadTo(context);
        m_status = SourceStatus::Ready;
    }
    catch (const std::exception & e)
    {
        fail(SourceStatus::Error, e.what());
    }
}

ImageSource::ImageSource(GpuContext & context, ImageData data, std::string label)
    : m_path(std::move(label))
{
    try
    {
        const size_t pixels = static_cast<size_t>(data.width) * static_cast<size_t>(data.height);
        if (data.width == 0 || data.height == 0)
        {
            throw std::invalid_argument("image has no pixels");
        }
        if (data.rgba.size() != pixels * 4)
        {
            throw std::invalid_argument("image has " + std::to_string(data.rgba.size()) + " samples, expected "
                + std::to_string(pixels * 4));
        }
        m_data = std::move(data);
        uploadTo(context);
        m_status = SourceStatus::Ready;
    }
    catch (const std::exception & e)
    {
        fail(SourceStatus::Error, e.what());
    }
}

ImageSource::~ImageSource() = default;

void ImageSource::uploadTo(GpuContext & context)
{
    const size_t pixels = static_cast<size_t>(m_data.width) * static_cast<size_t>(m_data.height);
    if (!m_data.depth.empty() && m_data.depth.size() != pixels)
    {
        throw std::invalid_argument("depth has " + std::to_string(m_data.depth.size()) + " samples, expected "
            + std::to_string(pixels));
    }

    const size_t bytes = pixels * kTexelBytes;
    auto buffer = std::make_unique<sabercat::Buffer>(context.device(), bytes,
                                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                                         | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                                         | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    buffer->upload(m_data.rgba.data(), bytes);
    m_buffer = std::move(buffer);

    if (m_data.depth.empty())
    {
        return;
    }
    const size_t depthBytes = pixels * kDepthBytes;
    auto depthBuffer = std::make_unique<sabercat::Buffer>(context.device(), depthBytes,
                                                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                                                              | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                                                              | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    depthBuffer->upload(m_data.depth.data(), depthBytes);
    m_depthBuffer = std::move(depthBuffer);

    // Scanned once here: the host copy never changes after load, so the cached range
    // stays exact for the source's lifetime.
    bool any = false;
    for (const float d : m_data.depth)
    {
        if (!isValidDepth(d))
        {
            continue;
        }
        if (!any)
        {
            m_depthLo = d;
            m_depthHi = d;
            any = true;
        }
        else
        {
            m_depthLo = std::min(m_depthLo, d);
            m_depthHi = std::max(m_depthHi, d);
        }
    }
    m_hasDepthRange = any;
}

void ImageSource::fail(SourceStatus status, std::string message)
{
    m_buffer.reset();
    m_depthBuffer.reset();
    m_hasDepthRange = false;
    m_data = ImageData{};
    m_status = status;
    m_message = std::move(message);
}

BufferView ImageSource::color() const
{
    return m_buffer ? m_buffer->view(static_cast<uint32_t>(kTexelBytes)) : BufferView{};
}

BufferView ImageSource::depth() const
{
    return m_depthBuffer ? m_depthBuffer->view(static_cast<uint32_t>(kDepthBytes)) : BufferView{};
}

DepthInfo ImageSource::depthInfo() const
{
    if (!m_depthBuffer)
    {
        return {};
    }
    return DepthInfo{1, 0, -1};
}

bool ImageSource::readPixel(int32_t x, int32_t y, float rgba[4]) const
{
    if (m_status != SourceStatus::Ready || x < 0 || y < 0)
    {
        return false;
    }
    if (static_cast<uint32_t>(x) >= m_data.width || static_cast<uint32_t>(y) >= m_data.height)
    {
        return false;
    }
    const size_t index = (static_cast<size_t>(y) * m_data.width + static_cast<size_t>(x)) * 4;
    std::memcpy(rgba, &m_data.rgba[index], 4 * sizeof(float));
    return true;
}

bool ImageSource::readDepthRaw(int32_t x, int32_t y, float out[4], uint32_t & count) const
{
    count = 0;
    if (m_status != SourceStatus::Ready || !m_depthBuffer || x < 0 || y < 0)
    {
        return false;
    }
    if (static_cast<uint32_t>(x) >= m_data.width || static_cast<uint32_t>(y) >= m_data.height)
    {
        return false;
    }
    const size_t index = static_cast<size_t>(y) * m_data.width + static_cast<size_t>(x);
    out[0] = m_data.depth[index];
    count = 1;
    return true;
}

bool ImageSource::depthRange(float & lo, float & hi) const
{
    if (m_status != SourceStatus::Ready || !m_hasDepthRange)
    {
        return false;
    }
    lo = m_depthLo;
    hi = m_depthHi;
    return true;
}

} // namespace sabercat::compare::gui
