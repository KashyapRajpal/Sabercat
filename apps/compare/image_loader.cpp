// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// loadImage(): dispatch by file extension, and the helpers the per-format readers share.

#include "image_io.h"

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

// Lower-case extension of a UTF-8 path, with the dot. Empty when there is none.
// Works on the UTF-8 string, so non-ASCII folder names cannot affect it.
std::string lowerExtension(const std::string & utf8Path)
{
    const size_t separator = utf8Path.find_last_of("/\\");
    const size_t dot = utf8Path.find_last_of('.');
    if (dot == std::string::npos || (separator != std::string::npos && dot < separator))
    {
        return std::string();
    }
    std::string extension = utf8Path.substr(dot);
    for (char & ch : extension)
    {
        if (ch >= 'A' && ch <= 'Z')
        {
            ch = static_cast<char>(ch - 'A' + 'a');
        }
    }
    return extension;
}

} // namespace

size_t rgbaSampleCount(uint64_t width, uint64_t height, const std::string & utf8Path)
{
    if (width == 0 || height == 0)
    {
        throw std::runtime_error(utf8Path + ": image has zero size");
    }
    // kMaxSamples / 4 pixels is the most an RGBA image may have. Testing the
    // quotient before multiplying keeps the product from wrapping.
    constexpr uint64_t kMaxPixels = kMaxSamples / 4u;
    if (width > kMaxPixels || height > kMaxPixels || width > kMaxPixels / height)
    {
        throw std::runtime_error(utf8Path + ": image is too large for the compare tools (" + std::to_string(width)
                                 + "x" + std::to_string(height) + ")");
    }
    return static_cast<size_t>(width * height * 4u);
}

std::vector<unsigned char> readFileBytes(const std::string & utf8Path, uint64_t maxBytes)
{
    const std::filesystem::path path = std::filesystem::u8path(utf8Path);
    std::error_code ec;
    const std::uintmax_t size = std::filesystem::file_size(path, ec);
    if (ec)
    {
        throw std::runtime_error(utf8Path + ": cannot read the file (" + ec.message() + ")");
    }
    if (size == 0)
    {
        throw std::runtime_error(utf8Path + ": the file is empty");
    }
    if (size > maxBytes)
    {
        throw std::runtime_error(utf8Path + ": the file is too large (" + std::to_string(size) + " bytes)");
    }
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        throw std::runtime_error(utf8Path + ": cannot open the file");
    }
    std::vector<unsigned char> bytes(static_cast<size_t>(size));
    in.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (in.gcount() != static_cast<std::streamsize>(bytes.size()))
    {
        throw std::runtime_error(utf8Path + ": could not read the whole file");
    }
    return bytes;
}

ImageData loadImage(const std::string & utf8Path)
{
    if (utf8Path.empty())
    {
        throw std::runtime_error("image path is empty");
    }
    const std::string extension = lowerExtension(utf8Path);
    if (extension == ".png" || extension == ".jpg" || extension == ".jpeg")
    {
        return loadWicRgba(utf8Path);
    }
    if (extension == ".webp")
    {
        return loadWebpRgba(utf8Path);
    }
    if (extension == ".exr")
    {
        return loadExrRgba(utf8Path);
    }
    if (extension == ".rgba32f")
    {
        return loadRawCapture(utf8Path);
    }
    throw std::runtime_error(utf8Path + ": unsupported image type (expected .png, .jpg, .jpeg, .webp, .exr or .rgba32f)");
}

} // namespace sabercat::compare
