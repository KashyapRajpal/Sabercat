// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// A still image as a compare source: loaded from a file (or handed in as host data),
// kept on the host for the probe and the CPU reference, and uploaded once to a device
// buffer of RGBA32F texels for the compositor and the metrics. A sibling depth map
// (ImageData::depth) is uploaded the same way, one float per pixel.
#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include "image_data.h"
#include "media_source.h"

namespace sabercat
{
class Buffer;
} // namespace sabercat

namespace sabercat::compare::gui
{

class GpuContext;

class ImageSource final : public MediaSource
{
  public:
    // Loads utf8Path with loadImage(). A file that does not exist gives status Missing
    // ("file not found: <path>"). Any other std::exception gives status Error with its
    // message. The constructor does not throw for these.
    ImageSource(GpuContext & context, const std::string & utf8Path);
    // An in-memory image. data.rgba must hold width * height * 4 floats (width and height
    // at least 1), and data.depth either nothing or width * height floats; otherwise the
    // status is Error. label is returned by path().
    ImageSource(GpuContext & context, ImageData data, std::string label);
    // Frees the device buffers. The GPU must be idle (the caller's frame is done).
    ~ImageSource() override;

    ImageSource(const ImageSource &) = delete;
    ImageSource & operator=(const ImageSource &) = delete;

    project::SourceKind kind() const override
    {
        return project::SourceKind::Image;
    }
    SourceStatus status() const override
    {
        return m_status;
    }
    const std::string & statusMessage() const override
    {
        return m_message;
    }
    uint32_t width() const override
    {
        return m_data.width;
    }
    uint32_t height() const override
    {
        return m_data.height;
    }
    Encoding encoding() const override
    {
        return m_data.encoding;
    }
    bool premultiplied() const override
    {
        return m_data.premultiplied;
    }
    bool hasAlpha() const override
    {
        return m_data.hasAlpha;
    }
    // Width * height RGBA32F texels; invalid unless Ready.
    BufferView color() const override;
    // Width * height floats (one depth per pixel); invalid when the image has no depth.
    BufferView depth() const override;
    // {1, 0, -1} when the image has depth, {} otherwise.
    DepthInfo depthInfo() const override;
    // Reads the host copy, bit for bit. False when not Ready or out of range.
    bool readPixel(int32_t x, int32_t y, float rgba[4]) const override;
    bool readDepthRaw(int32_t x, int32_t y, float out[4], uint32_t & count) const override;
    // Min and max of the finite, positive depth samples, scanned once at load.
    bool depthRange(float & lo, float & hi) const override;
    // The host copy: non-null only when Ready.
    const ImageData * hostData() const override
    {
        return data();
    }

    // The host copy: non-null only when Ready.
    const ImageData * data() const
    {
        return m_status == SourceStatus::Ready ? &m_data : nullptr;
    }
    // The UTF-8 file path, or the label of an in-memory image.
    const std::string & path() const
    {
        return m_path;
    }

  private:
    void uploadTo(GpuContext & context);
    void fail(SourceStatus status, std::string message);

    std::string m_path;
    std::string m_message;
    ImageData m_data;
    std::unique_ptr<sabercat::Buffer> m_buffer;
    std::unique_ptr<sabercat::Buffer> m_depthBuffer;
    float m_depthLo = 0.0f;
    float m_depthHi = 0.0f;
    SourceStatus m_status = SourceStatus::Error;
    bool m_hasDepthRange = false;
};

} // namespace sabercat::compare::gui
