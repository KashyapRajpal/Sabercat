// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Raw renderer captures written by examples/dataset_capture: view-NNN.rgba32f (width *
// height vec4 float32, little-endian, rows as downloadOutputImage() returns them), an
// optional view-NNN.depth32f (width * height float32) with the same stem, and a
// capture.json in the same folder holding the top-level "width" and "height".
// Layout, from the renderer (checked against the code, not against a capture):
//  - Row order: row 0 is the TOP image row, as in a PNG. tile_rasterize.slang writes
//    outputImage[pixel] with pixel.y = image row, and forward_project.slang puts the
//    pixel center at y = cy - fy * Y / depth (y grows downward). ForwardPass::
//    downloadOutputImage() copies the VkImage into a tightly packed buffer from
//    offset 0 (vkCmdCopyImageToBuffer, bufferRowLength 0), with no flip. The viewer's
//    present() blit also maps row 0 to the top of the swapchain.
//  - Channel order: r, g, b, a, as the R32G32B32A32_SFLOAT image stores them
//    (glm::vec4 x, y, z, w). Little-endian floats.
//  - Alpha: premultiplied and coverage, the same convention as Renderer::readImage()
//    (renderer.h). The shader writes float4(sum T_i a_i c_i, 1 - T_final): rgb is
//    premultiplied by alpha, and alpha is 1 - transmittance, not a straight opacity.
//    loadRawRgba32f() keeps rgb as stored, so dropping alpha (displayRgb) composites the
//    capture over black, the renderer's default background. Other backgrounds are not
//    applied here.
//  - Color: display-referred sRGB rgb (forward_common.slangh: "not linear"), so the
//    loader marks the capture Encoding::Display.
// The loader does not flip rows.

#include "image_io.h"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace sabercat::compare
{

namespace
{

[[noreturn]] void fail(const std::string & utf8Path, const std::string & reason)
{
    throw std::runtime_error(utf8Path + ": " + reason);
}

// Reads count floats from the file; the size was checked by the caller.
void readFloats(const std::filesystem::path & path, const std::string & utf8Path, float * dst, size_t count)
{
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        fail(utf8Path, "cannot open the file");
    }
    const std::streamsize bytes = static_cast<std::streamsize>(count * sizeof(float));
    in.read(reinterpret_cast<char *>(dst), bytes);
    if (in.gcount() != bytes)
    {
        fail(utf8Path, "could not read the whole file");
    }
}

// One positive "width" or "height" from capture.json, as a 32-bit value.
uint32_t dimensionFromJson(const nlohmann::json & document, const char * key, const std::string & metadataPath)
{
    const auto found = document.find(key);
    if (found == document.end() || !found->is_number_integer())
    {
        fail(metadataPath, std::string("capture.json has no integer \"") + key
                               + "\"; width and height are unknown");
    }
    uint64_t value = 0;
    if (found->is_number_unsigned())
    {
        value = found->get<uint64_t>();
    }
    else
    {
        const int64_t signedValue = found->get<int64_t>();
        if (signedValue < 0)
        {
            fail(metadataPath, std::string("capture.json \"") + key + "\" is negative");
        }
        value = static_cast<uint64_t>(signedValue);
    }
    if (value == 0 || value > UINT32_MAX)
    {
        fail(metadataPath, std::string("capture.json \"") + key + "\" must be between 1 and 4294967295");
    }
    return static_cast<uint32_t>(value);
}

} // namespace

ImageData loadRawRgba32f(const std::string & utf8Path, uint32_t width, uint32_t height)
{
    // Checked before any allocation or file access.
    const size_t samples = rgbaSampleCount(width, height, utf8Path);
    const uint64_t pixels = static_cast<uint64_t>(width) * height;
    const uint64_t expectedBytes = pixels * 16u;

    const std::filesystem::path path = std::filesystem::u8path(utf8Path);
    std::error_code ec;
    const std::uintmax_t size = std::filesystem::file_size(path, ec);
    if (ec)
    {
        fail(utf8Path, "cannot read the file (" + ec.message() + ")");
    }
    if (size != expectedBytes)
    {
        fail(utf8Path, "size mismatch: file has " + std::to_string(size) + " bytes, width*height*16 = "
                           + std::to_string(expectedBytes));
    }

    ImageData image;
    image.width = width;
    image.height = height;
    image.bitDepth = 32;
    image.encoding = Encoding::Display;
    // The capture stores alpha, so the source carries it.
    image.hasAlpha = true;
    // rgb is premultiplied by alpha (see the header comment above). loadRawCapture()
    // goes through this function, so it gets the same flag.
    image.premultiplied = true;
    image.rgba.resize(samples);
    readFloats(path, utf8Path, image.rgba.data(), samples);

    // The depth file is optional; when present its size must match exactly.
    std::filesystem::path depthPath = path;
    depthPath.replace_extension(".depth32f");
    if (std::filesystem::exists(depthPath, ec) && !ec)
    {
        const std::uintmax_t depthSize = std::filesystem::file_size(depthPath, ec);
        if (ec)
        {
            fail(utf8Path, "cannot read the depth file (" + ec.message() + ")");
        }
        const uint64_t depthExpected = pixels * 4u;
        if (depthSize != depthExpected)
        {
            fail(utf8Path, "depth size mismatch: the .depth32f file has " + std::to_string(depthSize)
                               + " bytes, width*height*4 = " + std::to_string(depthExpected));
        }
        image.depth.resize(static_cast<size_t>(pixels));
        readFloats(depthPath, utf8Path, image.depth.data(), image.depth.size());
    }
    return image;
}

ImageData loadRawCapture(const std::string & utf8Path)
{
    // capture.json sits beside the image: the same folder, by UTF-8 string so that
    // non-ASCII folder names need no conversion.
    const size_t separator = utf8Path.find_last_of("/\\");
    const std::string folder = separator == std::string::npos ? std::string() : utf8Path.substr(0, separator + 1);
    const std::string metadataPath = folder + "capture.json";

    std::error_code ec;
    if (!std::filesystem::is_regular_file(std::filesystem::u8path(metadataPath), ec))
    {
        fail(utf8Path, "capture.json not found next to the raw image; width and height are unknown");
    }
    const std::vector<unsigned char> bytes = readFileBytes(metadataPath, kMaxFileBytes);
    const std::string text(bytes.begin(), bytes.end());

    nlohmann::json document;
    try
    {
        document = nlohmann::json::parse(text);
    }
    catch (const nlohmann::json::exception & error)
    {
        fail(metadataPath, std::string("invalid JSON: ") + error.what());
    }
    if (!document.is_object())
    {
        fail(metadataPath, "capture.json is not a JSON object");
    }

    const uint32_t width = dimensionFromJson(document, "width", metadataPath);
    const uint32_t height = dimensionFromJson(document, "height", metadataPath);
    return loadRawRgba32f(utf8Path, width, height);
}

} // namespace sabercat::compare
