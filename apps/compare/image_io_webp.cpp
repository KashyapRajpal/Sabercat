// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// WebP input through libwebp's decode-only archive (webpdecoder, see the root CMakeLists.txt).

#include "image_io.h"

#include <webp/decode.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace sabercat::compare
{

ImageData loadWebpRgba(const std::string & utf8Path)
{
    const std::vector<unsigned char> bytes = readFileBytes(utf8Path, kMaxFileBytes);

    // Header only: the dimensions, alpha and animation flags, before any pixel work.
    WebPBitstreamFeatures features = {};
    const VP8StatusCode status = WebPGetFeatures(bytes.data(), bytes.size(), &features);
    if (status != VP8_STATUS_OK)
    {
        throw std::runtime_error(utf8Path + ": not a valid WebP file (libwebp status " + std::to_string(static_cast<int>(status)) + ")");
    }
    if (features.has_animation)
    {
        throw std::runtime_error(utf8Path + ": animated WebP is not supported");
    }
    const uint64_t headerWidth = features.width > 0 ? static_cast<uint64_t>(features.width) : 0u;
    const uint64_t headerHeight = features.height > 0 ? static_cast<uint64_t>(features.height) : 0u;
    const size_t samples = rgbaSampleCount(headerWidth, headerHeight, utf8Path);

    int width = 0;
    int height = 0;
    // Straight-alpha RGBA, 8 bits per channel, in the buffer WebPDecodeRGBA allocates.
    std::unique_ptr<uint8_t, void (*)(void *)> decoded(WebPDecodeRGBA(bytes.data(), bytes.size(), &width, &height),
                                                       &WebPFree);
    if (!decoded)
    {
        throw std::runtime_error(utf8Path + ": WebP decoding failed");
    }
    if (static_cast<uint64_t>(width) != headerWidth || static_cast<uint64_t>(height) != headerHeight)
    {
        throw std::runtime_error(utf8Path + ": decoded size differs from the WebP header");
    }

    ImageData image;
    image.width = static_cast<uint32_t>(width);
    image.height = static_cast<uint32_t>(height);
    image.bitDepth = 8;
    image.encoding = Encoding::Display;
    image.hasAlpha = features.has_alpha != 0;
    image.rgba.resize(samples);

    const uint8_t * const in = decoded.get();
    for (size_t i = 0; i < samples; ++i)
    {
        image.rgba[i] = static_cast<float>(in[i]) / 255.0f;
    }
    if (!image.hasAlpha)
    {
        for (size_t i = 3; i < samples; i += 4)
        {
            image.rgba[i] = 1.0f;
        }
    }
    return image;
}

} // namespace sabercat::compare
