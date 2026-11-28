// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The source contract of the compare window. The compositor, the metrics and the
// probe read a source only through this interface, so image, scene and video sources
// plug in the same way.
#pragma once

#include "compare_project.h"
#include "image_data.h"

#include <cstdint>
#include <string>

#include "sabercat/buffer_view.h"

namespace sabercat::compare::gui
{

enum class SourceStatus : uint8_t
{
    Ready,
    Error,
    Missing
};

// Layout of depth(): floatsPerPixel n (1, 2 or 4; 0 = the source has no depth).
// frontComponent: component index of the depth the maps use (valid sample: finite and > 0), -1 none.
// coverageComponent: component holding coverage (bit-equal to the color alpha), -1 = use color().a.
struct DepthInfo
{
    uint32_t floatsPerPixel = 0;
    int32_t frontComponent = -1;
    int32_t coverageComponent = -1;
};

class MediaSource
{
  public:
    virtual ~MediaSource() = default;

    virtual project::SourceKind kind() const = 0;
    virtual SourceStatus status() const = 0;
    // Error text; empty when Ready.
    virtual const std::string & statusMessage() const = 0;
    virtual uint32_t width() const = 0;
    virtual uint32_t height() const = 0;
    virtual Encoding encoding() const = 0;
    virtual bool premultiplied() const = 0;
    virtual bool hasAlpha() const = 0;

    // RGBA32F, row-major, top row first, stride 16, size width * height * 16. Invalid
    // (no buffer) when the status is not Ready. The source owns the buffer; callers
    // bind it and never free it.
    virtual BufferView color() const = 0;
    // Optional: width * height * floatsPerPixel floats (stride floatsPerPixel * 4).
    // Invalid view when absent. Layout described by depthInfo().
    virtual BufferView depth() const
    {
        return {};
    }
    virtual DepthInfo depthInfo() const
    {
        return {};
    }

    // Advances to global time t in seconds. Returns true when color() changed. Images
    // return false.
    virtual bool update(double /*t*/)
    {
        return false;
    }

    // Exact sample at an integer pixel, for the probe. False when out of range or not
    // Ready. Images read their host copy.
    virtual bool readPixel(int32_t x, int32_t y, float rgba[4]) const = 0;
    // Raw depth components at an integer pixel (count = floatsPerPixel, out[0..count)
    // filled). False when absent, out of range or not Ready.
    virtual bool readDepthRaw(int32_t /*x*/, int32_t /*y*/, float /*out*/[4], uint32_t & /*count*/) const
    {
        return false;
    }
    // Front-component depth at an integer pixel: the frontComponent of readDepthRaw.
    // False when the source has no front depth, or on any failure of readDepthRaw.
    bool readDepth(int32_t x, int32_t y, float & value) const
    {
        const DepthInfo info = depthInfo();
        float raw[4] = {};
        uint32_t count = 0;
        if (info.frontComponent < 0 || !readDepthRaw(x, y, raw, count)
            || static_cast<uint32_t>(info.frontComponent) >= count)
        {
            return false;
        }
        value = raw[info.frontComponent];
        return true;
    }
    // Min and max of the VALID depth samples (finite and > 0) over the whole buffer.
    // False when there are none or no depth. Never returns non-finite values.
    virtual bool depthRange(float & /*lo*/, float & /*hi*/) const
    {
        return false;
    }
    // A host copy: rgba (color() values, bit for bit), depth = front depth per pixel
    // (width * height floats, or empty), width, height, encoding, hasAlpha, premultiplied.
    // Null when not Ready. Used by exports and the report.
    virtual const ImageData * hostData() const
    {
        return nullptr;
    }
};

} // namespace sabercat::compare::gui
