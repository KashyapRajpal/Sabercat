// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Image loaders of the comparison core (apps/compare/image_io.h, image_data.h): PNG
// through WIC, raw captures, OpenEXR through tinyexr and WebP. Every fixture is written
// by this file into a scratch folder: PNG bytes come from a small encoder with stored
// (uncompressed) deflate blocks, EXR from tinyexr's own save API, raw captures from
// plain floats. The lossless WebP is committed (tests/compare_golden/fixtures/webp, see
// its README). Windows only, like the rest of the comparison core.
#include "unit_test.h"

#include "flip_hdr.h"
#include "image_io.h"

#include <tinyexr.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

// The committed lossless WebP fixture; the test target defines it (tests/unit/CMakeLists.txt).
#ifndef SABERCAT_COMPARE_FIXTURES_DIR
#define SABERCAT_COMPARE_FIXTURES_DIR ""
#endif

namespace sabercat::unit
{

namespace
{

namespace fs = std::filesystem;
using sabercat::compare::colorizeMagma;
using sabercat::compare::defaultFlipPpd;
using sabercat::compare::displayRgb;
using sabercat::compare::Encoding;
using sabercat::compare::FlipResult;
using sabercat::compare::flip;
using sabercat::compare::Image;
using sabercat::compare::ImageData;
using sabercat::compare::loadImage;
using sabercat::compare::loadPng;
using sabercat::compare::loadRawRgba32f;
using sabercat::compare::PsnrResult;
using sabercat::compare::psnr;
using sabercat::compare::Rgb8Image;
using sabercat::compare::savePngRgb8;

constexpr float kNaN = std::numeric_limits<float>::quiet_NaN();
constexpr float kInf = std::numeric_limits<float>::infinity();

// Scratch folder under the temp directory, removed with its contents on destruction.
class ScratchDir
{
public:
    ScratchDir()
    {
        std::random_device random;
        const std::string name = "sabercat-loaders-" + std::to_string(random()) + "-" + std::to_string(random());
        path_ = fs::temp_directory_path() / fs::u8path(name);
        fs::create_directories(path_);
    }
    ~ScratchDir()
    {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    ScratchDir(const ScratchDir &) = delete;
    ScratchDir & operator=(const ScratchDir &) = delete;

    // A file name inside the folder, as a UTF-8 path string.
    std::string file(const std::string & name) const { return utf8(path_ / fs::u8path(name)); }

    // A subfolder, created, as a UTF-8 path string.
    std::string folder(const std::string & name) const
    {
        const fs::path sub = path_ / fs::u8path(name);
        fs::create_directories(sub);
        return utf8(sub);
    }

private:
    static std::string utf8(const fs::path & path)
    {
        const std::u8string text = path.u8string();
        return std::string(reinterpret_cast<const char *>(text.data()), text.size());
    }

    fs::path path_;
};

void writeBytes(const std::string & utf8Path, const void * data, size_t size)
{
    std::ofstream out(fs::u8path(utf8Path), std::ios::binary | std::ios::trunc);
    if (size > 0)
    {
        out.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
    }
}

void writeBytes(const std::string & utf8Path, const std::vector<uint8_t> & bytes)
{
    writeBytes(utf8Path, bytes.data(), bytes.size());
}

void writeText(const std::string & utf8Path, const std::string & text)
{
    writeBytes(utf8Path, text.data(), text.size());
}

// True when call() throws std::runtime_error whose message contains needle. The
// message is returned through message.
template <class Call>
bool throwsWith(Call && call, const std::string & needle, std::string & message)
{
    try
    {
        call();
    }
    catch (const std::runtime_error & error)
    {
        message = error.what();
        return message.find(needle) != std::string::npos;
    }
    catch (...)
    {
        message = "not a std::runtime_error";
        return false;
    }
    message = "no exception";
    return false;
}

// Loads an image; false with the message on any failure.
bool tryLoad(const std::string & utf8Path, ImageData & out, std::string & error)
{
    try
    {
        out = loadImage(utf8Path);
        return true;
    }
    catch (const std::exception & exception)
    {
        error = exception.what();
        return false;
    }
}

// ---------------------------------------------------------------------------
// PNG encoder: IHDR, optional PLTE and tRNS, IDAT as stored deflate blocks, IEND.
// ---------------------------------------------------------------------------

uint32_t crc32Range(const uint8_t * data, size_t size)
{
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < size; ++i)
    {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit)
        {
            crc = (crc & 1u) != 0 ? (crc >> 1) ^ 0xEDB88320u : (crc >> 1);
        }
    }
    return ~crc;
}

void appendBe32(std::vector<uint8_t> & out, uint32_t value)
{
    out.push_back(static_cast<uint8_t>(value >> 24));
    out.push_back(static_cast<uint8_t>(value >> 16));
    out.push_back(static_cast<uint8_t>(value >> 8));
    out.push_back(static_cast<uint8_t>(value));
}

void appendLe32(std::vector<uint8_t> & out, uint32_t value)
{
    out.push_back(static_cast<uint8_t>(value));
    out.push_back(static_cast<uint8_t>(value >> 8));
    out.push_back(static_cast<uint8_t>(value >> 16));
    out.push_back(static_cast<uint8_t>(value >> 24));
}

void appendText(std::vector<uint8_t> & out, const char * text)
{
    for (const char * p = text; *p != '\0'; ++p)
    {
        out.push_back(static_cast<uint8_t>(*p));
    }
}

uint32_t adler32Of(const std::vector<uint8_t> & bytes)
{
    uint32_t a = 1;
    uint32_t b = 0;
    for (const uint8_t byte : bytes)
    {
        a = (a + byte) % 65521u;
        b = (b + a) % 65521u;
    }
    return (b << 16) | a;
}

// A zlib stream made of stored (uncompressed) deflate blocks.
std::vector<uint8_t> zlibStored(const std::vector<uint8_t> & raw)
{
    std::vector<uint8_t> out = {0x78, 0x01};
    size_t pos = 0;
    do
    {
        const size_t count = std::min<size_t>(65535, raw.size() - pos);
        const bool last = pos + count == raw.size();
        out.push_back(last ? 1 : 0);
        const uint16_t len = static_cast<uint16_t>(count);
        const uint16_t nlen = static_cast<uint16_t>(~len);
        out.push_back(static_cast<uint8_t>(len));
        out.push_back(static_cast<uint8_t>(len >> 8));
        out.push_back(static_cast<uint8_t>(nlen));
        out.push_back(static_cast<uint8_t>(nlen >> 8));
        out.insert(out.end(), raw.data() + pos, raw.data() + pos + count);
        pos += count;
    } while (pos < raw.size());
    appendBe32(out, adler32Of(raw));
    return out;
}

void appendChunk(std::vector<uint8_t> & png, const char * type, const std::vector<uint8_t> & data)
{
    appendBe32(png, static_cast<uint32_t>(data.size()));
    const size_t start = png.size();
    appendText(png, type);
    png.insert(png.end(), data.begin(), data.end());
    appendBe32(png, crc32Range(png.data() + start, png.size() - start));
}

// rows holds height scanlines of rowBytes bytes each, without filter bytes (added here).
std::vector<uint8_t> makePng(uint32_t width, uint32_t height, uint8_t bitDepth, uint8_t colorType,
                             const std::vector<uint8_t> & rows, size_t rowBytes,
                             const std::vector<uint8_t> & palette = {}, const std::vector<uint8_t> & trns = {})
{
    std::vector<uint8_t> png = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};
    std::vector<uint8_t> header;
    appendBe32(header, width);
    appendBe32(header, height);
    header.push_back(bitDepth);
    header.push_back(colorType);
    header.push_back(0); // compression
    header.push_back(0); // filter method
    header.push_back(0); // no interlace
    appendChunk(png, "IHDR", header);
    if (!palette.empty())
    {
        appendChunk(png, "PLTE", palette);
    }
    if (!trns.empty())
    {
        appendChunk(png, "tRNS", trns);
    }
    std::vector<uint8_t> filtered;
    for (uint32_t y = 0; y < height; ++y)
    {
        filtered.push_back(0); // filter type None
        const uint8_t * const row = rows.data() + static_cast<size_t>(y) * rowBytes;
        filtered.insert(filtered.end(), row, row + rowBytes);
    }
    appendChunk(png, "IDAT", zlibStored(filtered));
    appendChunk(png, "IEND", {});
    return png;
}

// The fixtures. Each is small enough that its IDAT starts before byte 40.

// 2x1 RGBA8 with straight alpha: pixel 0 is (200, 100, 50, 128).
std::vector<uint8_t> pngRgba8()
{
    return makePng(2, 1, 8, 6, {200, 100, 50, 128, 10, 20, 30, 255}, 8);
}

// 2x1 RGBA16 (big-endian samples): pixel 0 is (65535, 32768, 0, 16384).
std::vector<uint8_t> pngRgba16()
{
    return makePng(2, 1, 16, 6, {0xFF, 0xFF, 0x80, 0x00, 0x00, 0x00, 0x40, 0x00,
                                 0x00, 0x01, 0x00, 0x02, 0x00, 0x03, 0xFF, 0xFF}, 16);
}

// 2x1 gray8 with values 77 and 255.
std::vector<uint8_t> pngGray8()
{
    return makePng(2, 1, 8, 0, {77, 255}, 2);
}

// 1x1 gray+alpha8: (100, 200).
std::vector<uint8_t> pngGrayAlpha8()
{
    return makePng(1, 1, 8, 4, {100, 200}, 2);
}

// 1x1 gray16 with value 0x8000.
std::vector<uint8_t> pngGray16()
{
    return makePng(1, 1, 16, 0, {0x80, 0x00}, 2);
}

// 2x1 palette8: entry 0 red with alpha 128 (tRNS), entry 1 green (opaque). Pixels are 0, 1.
std::vector<uint8_t> pngPalette8()
{
    return makePng(2, 1, 8, 3, {0, 1}, 2, {255, 0, 0, 0, 255, 0}, {128});
}

// Deterministic bytes that are not an image.
std::vector<uint8_t> garbageBytes(size_t count)
{
    std::vector<uint8_t> bytes(count);
    for (size_t i = 0; i < count; ++i)
    {
        bytes[i] = static_cast<uint8_t>(i * 73u + 11u);
    }
    return bytes;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

constexpr float kTol = 1e-6f;

void testPngDecoding(Context & ctx, const ScratchDir & dir)
{
    ImageData image;
    std::string error;

    // RGBA8, straight alpha: the color is not premultiplied.
    const std::string rgba8 = dir.file("rgba8.png");
    writeBytes(rgba8, pngRgba8());
    UNIT_CHECK(ctx, tryLoad(rgba8, image, error));
    if (image.rgba.size() == 8)
    {
        UNIT_CHECK(ctx, image.width == 2 && image.height == 1);
        UNIT_CHECK(ctx, image.bitDepth == 8);
        UNIT_CHECK(ctx, image.encoding == Encoding::Display);
        UNIT_CHECK(ctx, image.hasAlpha);
        UNIT_CHECK(ctx, image.depth.empty());
        UNIT_CHECK(ctx, image.nonFiniteCount == 0);
        UNIT_CHECK_NEAR(ctx, image.rgba[0], 200.0f / 255.0f, kTol);
        UNIT_CHECK_NEAR(ctx, image.rgba[1], 100.0f / 255.0f, kTol);
        UNIT_CHECK_NEAR(ctx, image.rgba[2], 50.0f / 255.0f, kTol);
        UNIT_CHECK_NEAR(ctx, image.rgba[3], 128.0f / 255.0f, kTol);
        UNIT_CHECK_NEAR(ctx, image.rgba[7], 1.0f, kTol);
    }
    else
    {
        std::cerr << "rgba8: " << error << std::endl;
        UNIT_CHECK(ctx, false);
    }

    // RGBA16: values are code / 65535 and alpha is kept.
    const std::string rgba16 = dir.file("rgba16.png");
    writeBytes(rgba16, pngRgba16());
    UNIT_CHECK(ctx, tryLoad(rgba16, image, error));
    if (image.rgba.size() == 8)
    {
        UNIT_CHECK(ctx, image.bitDepth == 16);
        UNIT_CHECK(ctx, image.hasAlpha);
        UNIT_CHECK_NEAR(ctx, image.rgba[0], 1.0f, kTol);
        UNIT_CHECK_NEAR(ctx, image.rgba[1], 32768.0f / 65535.0f, kTol);
        UNIT_CHECK_NEAR(ctx, image.rgba[2], 0.0f, kTol);
        UNIT_CHECK_NEAR(ctx, image.rgba[3], 16384.0f / 65535.0f, kTol);
    }
    else
    {
        std::cerr << "rgba16: " << error << std::endl;
        UNIT_CHECK(ctx, false);
    }

    // Gray8: replicated, opaque.
    const std::string gray8 = dir.file("gray8.png");
    writeBytes(gray8, pngGray8());
    UNIT_CHECK(ctx, tryLoad(gray8, image, error));
    if (image.rgba.size() == 8)
    {
        UNIT_CHECK(ctx, !image.hasAlpha);
        UNIT_CHECK(ctx, image.bitDepth == 8);
        UNIT_CHECK_NEAR(ctx, image.rgba[0], 77.0f / 255.0f, kTol);
        UNIT_CHECK_NEAR(ctx, image.rgba[1], 77.0f / 255.0f, kTol);
        UNIT_CHECK_NEAR(ctx, image.rgba[2], 77.0f / 255.0f, kTol);
        UNIT_CHECK_NEAR(ctx, image.rgba[3], 1.0f, kTol);
    }
    else
    {
        std::cerr << "gray8: " << error << std::endl;
        UNIT_CHECK(ctx, false);
    }

    // Gray+alpha8.
    const std::string grayAlpha = dir.file("gray-alpha8.png");
    writeBytes(grayAlpha, pngGrayAlpha8());
    UNIT_CHECK(ctx, tryLoad(grayAlpha, image, error));
    if (image.rgba.size() == 4)
    {
        UNIT_CHECK(ctx, image.hasAlpha);
        UNIT_CHECK_NEAR(ctx, image.rgba[0], 100.0f / 255.0f, kTol);
        UNIT_CHECK_NEAR(ctx, image.rgba[3], 200.0f / 255.0f, kTol);
    }
    else
    {
        std::cerr << "gray+alpha8: " << error << std::endl;
        UNIT_CHECK(ctx, false);
    }

    // Gray16, copied as stored.
    const std::string gray16 = dir.file("gray16.png");
    writeBytes(gray16, pngGray16());
    UNIT_CHECK(ctx, tryLoad(gray16, image, error));
    if (image.rgba.size() == 4)
    {
        UNIT_CHECK(ctx, image.bitDepth == 16);
        UNIT_CHECK_NEAR(ctx, image.rgba[0], 32768.0f / 65535.0f, kTol);
        UNIT_CHECK_NEAR(ctx, image.rgba[3], 1.0f, kTol);
    }
    else
    {
        std::cerr << "gray16: " << error << std::endl;
        UNIT_CHECK(ctx, false);
    }

    // Palette with tRNS: the palette supplies the alpha.
    const std::string palette = dir.file("palette8.png");
    writeBytes(palette, pngPalette8());
    UNIT_CHECK(ctx, tryLoad(palette, image, error));
    if (image.rgba.size() == 8)
    {
        UNIT_CHECK(ctx, image.hasAlpha);
        UNIT_CHECK(ctx, image.bitDepth == 8);
        UNIT_CHECK_NEAR(ctx, image.rgba[0], 1.0f, kTol);
        UNIT_CHECK_NEAR(ctx, image.rgba[1], 0.0f, kTol);
        UNIT_CHECK_NEAR(ctx, image.rgba[3], 128.0f / 255.0f, kTol);
        UNIT_CHECK_NEAR(ctx, image.rgba[5], 1.0f, kTol);
        UNIT_CHECK_NEAR(ctx, image.rgba[7], 1.0f, kTol);
    }
    else
    {
        std::cerr << "palette8: " << error << std::endl;
        UNIT_CHECK(ctx, false);
    }

    // RGB written by savePngRgb8 and read back.
    const std::string rgb = dir.file("rgb8.png");
    Rgb8Image source;
    source.width = 2;
    source.height = 1;
    source.rgb = {1, 2, 3, 250, 251, 252};
    savePngRgb8(rgb, source);
    UNIT_CHECK(ctx, tryLoad(rgb, image, error));
    if (image.rgba.size() == 8)
    {
        UNIT_CHECK_NEAR(ctx, image.rgba[0], 1.0f / 255.0f, kTol);
        UNIT_CHECK_NEAR(ctx, image.rgba[6], 252.0f / 255.0f, kTol);
        UNIT_CHECK_NEAR(ctx, image.rgba[7], 1.0f, kTol);
    }

    // loadPng still returns RGB with the same values, alpha dropped.
    const sabercat::compare::Image legacy = loadPng(rgba8);
    UNIT_CHECK(ctx, legacy.width == 2 && legacy.height == 1 && legacy.rgb.size() == 6);
    if (legacy.rgb.size() == 6)
    {
        UNIT_CHECK_NEAR(ctx, legacy.rgb[0], 200.0f / 255.0f, 1e-5f);
        UNIT_CHECK_NEAR(ctx, legacy.rgb[2], 50.0f / 255.0f, 1e-5f);
    }

    // The extension is case-insensitive.
    const std::string upper = dir.file("UPPER.PNG");
    writeBytes(upper, pngRgba8());
    UNIT_CHECK(ctx, tryLoad(upper, image, error));
}

void testPngRejections(Context & ctx, const ScratchDir & dir)
{
    // Truncations inside the header and the start of the image data. Any of them
    // must throw; none may crash.
    const std::vector<uint8_t> full = pngRgba8();
    const size_t cuts[] = {0, 1, 8, 20, 33, 40, full.size() / 2};
    for (const size_t cut : cuts)
    {
        const std::string path = dir.file("truncated-png-" + std::to_string(cut) + ".png");
        writeBytes(path, full.data(), cut);
        ImageData image;
        std::string error;
        UNIT_CHECK(ctx, !tryLoad(path, image, error));
    }

    const std::string garbage = dir.file("garbage.png");
    writeBytes(garbage, garbageBytes(64));
    ImageData image;
    std::string error;
    UNIT_CHECK(ctx, !tryLoad(garbage, image, error));
    UNIT_CHECK(ctx, error.find(garbage) != std::string::npos);
}

void testNonAsciiPaths(Context & ctx, const ScratchDir & dir)
{
    // "café" folder and "été.png" file, UTF-8 encoded.
    const std::string folder = dir.folder("caf\xC3\xA9");
    const std::string pngPath = folder + "/\xC3\xA9t\xC3\xA9.png";
    writeBytes(pngPath, pngRgba8());
    ImageData image;
    std::string error;
    UNIT_CHECK(ctx, tryLoad(pngPath, image, error));
    if (image.rgba.size() == 8)
    {
        UNIT_CHECK_NEAR(ctx, image.rgba[0], 200.0f / 255.0f, kTol);
    }
    else
    {
        std::cerr << "non-ASCII PNG: " << error << std::endl;
        UNIT_CHECK(ctx, false);
    }
}

// Raw capture: width * height vec4 floats, little-endian, plus capture.json.
std::vector<float> rgbaRamp()
{
    std::vector<float> values(16);
    for (size_t i = 0; i < values.size(); ++i)
    {
        values[i] = static_cast<float>(i) * 0.0625f;
    }
    return values;
}

void testRawCaptures(Context & ctx, const ScratchDir & dir)
{
    const std::vector<float> rgba = rgbaRamp();
    const std::vector<float> depth = {1.0f, 2.0f, 3.0f, 4.0f};
    const std::string json = "{\"schemaVersion\": 1, \"width\": 2, \"height\": 2}";

    // A complete capture: rgba and depth.
    const std::string folder = dir.folder("capture");
    writeText(folder + "/capture.json", json);
    writeBytes(folder + "/view-000.rgba32f", rgba.data(), rgba.size() * sizeof(float));
    writeBytes(folder + "/view-000.depth32f", depth.data(), depth.size() * sizeof(float));

    ImageData image;
    std::string error;
    UNIT_CHECK(ctx, tryLoad(folder + "/view-000.rgba32f", image, error));
    if (image.rgba.size() == 16)
    {
        UNIT_CHECK(ctx, image.width == 2 && image.height == 2);
        UNIT_CHECK(ctx, image.bitDepth == 32);
        UNIT_CHECK(ctx, image.encoding == Encoding::Display);
        UNIT_CHECK(ctx, image.hasAlpha);
        UNIT_CHECK(ctx, image.nonFiniteCount == 0);
        UNIT_CHECK(ctx, image.rgba == rgba);
        UNIT_CHECK(ctx, image.depth == depth);
    }
    else
    {
        std::cerr << "raw capture: " << error << std::endl;
        UNIT_CHECK(ctx, false);
    }

    // The sized overload gives the same result.
    ImageData direct = loadRawRgba32f(folder + "/view-000.rgba32f", 2, 2);
    UNIT_CHECK(ctx, direct.rgba == rgba && direct.depth == depth);

    // No depth file: depth is empty and the rest still loads.
    const std::string noDepth = dir.folder("capture-no-depth");
    writeText(noDepth + "/capture.json", json);
    writeBytes(noDepth + "/view-000.rgba32f", rgba.data(), rgba.size() * sizeof(float));
    UNIT_CHECK(ctx, tryLoad(noDepth + "/view-000.rgba32f", image, error));
    UNIT_CHECK(ctx, image.depth.empty() && image.rgba == rgba);

    // A size mismatch (one float short) is an error.
    const std::string shortPath = folder + "/view-short.rgba32f";
    writeBytes(shortPath, rgba.data(), rgba.size() * sizeof(float) - sizeof(float));
    std::string message;
    UNIT_CHECK(ctx, throwsWith([&] { loadImage(shortPath); }, "size mismatch", message));

    // An empty raw file is a size mismatch too.
    const std::string emptyPath = folder + "/view-empty.rgba32f";
    writeBytes(emptyPath, nullptr, 0);
    UNIT_CHECK(ctx, throwsWith([&] { loadImage(emptyPath); }, "size mismatch", message));

    // Every truncation of the valid file throws.
    const size_t byteCount = rgba.size() * sizeof(float);
    const size_t truncations[] = {0, 1, 15, byteCount - 1};
    for (const size_t cut : truncations)
    {
        const std::string path = folder + "/view-cut-" + std::to_string(cut) + ".rgba32f";
        writeBytes(path, rgba.data(), cut);
        UNIT_CHECK(ctx, throwsWith([&] { loadImage(path); }, "size mismatch", message));
    }

    // A depth file of the wrong size is an error.
    const std::string badDepth = dir.folder("capture-bad-depth");
    writeText(badDepth + "/capture.json", json);
    writeBytes(badDepth + "/view-000.rgba32f", rgba.data(), rgba.size() * sizeof(float));
    writeBytes(badDepth + "/view-000.depth32f", depth.data(), 3 * sizeof(float));
    UNIT_CHECK(ctx, throwsWith([&] { loadImage(badDepth + "/view-000.rgba32f"); }, "depth size mismatch", message));

    // capture.json missing.
    const std::string noJson = dir.folder("capture-no-json");
    writeBytes(noJson + "/view-000.rgba32f", rgba.data(), rgba.size() * sizeof(float));
    UNIT_CHECK(ctx, throwsWith([&] { loadImage(noJson + "/view-000.rgba32f"); },
                               "capture.json not found", message));

    // capture.json that is not JSON, or has a bad width.
    const std::string badJson = dir.folder("capture-bad-json");
    writeText(badJson + "/capture.json", "{");
    writeBytes(badJson + "/view-000.rgba32f", rgba.data(), rgba.size() * sizeof(float));
    UNIT_CHECK(ctx, throwsWith([&] { loadImage(badJson + "/view-000.rgba32f"); }, "invalid JSON", message));

    const std::string zeroWidth = dir.folder("capture-zero-width");
    writeText(zeroWidth + "/capture.json", "{\"width\": 0, \"height\": 2}");
    writeBytes(zeroWidth + "/view-000.rgba32f", rgba.data(), rgba.size() * sizeof(float));
    UNIT_CHECK(ctx, throwsWith([&] { loadImage(zeroWidth + "/view-000.rgba32f"); }, "width", message));

    const std::string negativeHeight = dir.folder("capture-negative-height");
    writeText(negativeHeight + "/capture.json", "{\"width\": 2, \"height\": -2}");
    writeBytes(negativeHeight + "/view-000.rgba32f", rgba.data(), rgba.size() * sizeof(float));
    UNIT_CHECK(ctx, throwsWith([&] { loadImage(negativeHeight + "/view-000.rgba32f"); }, "height", message));

    // A huge declared size with a tiny file is refused before anything is allocated.
    const std::string huge = dir.folder("capture-huge");
    writeText(huge + "/capture.json", "{\"width\": 65535, \"height\": 65535}");
    writeBytes(huge + "/view-000.rgba32f", rgba.data(), rgba.size() * sizeof(float));
    UNIT_CHECK(ctx, throwsWith([&] { loadImage(huge + "/view-000.rgba32f"); }, "too large", message));

    // Non-ASCII folder name.
    const std::string accented = dir.folder("caf\xC3\xA9-capture");
    writeText(accented + "/capture.json", json);
    writeBytes(accented + "/view-000.rgba32f", rgba.data(), rgba.size() * sizeof(float));
    UNIT_CHECK(ctx, tryLoad(accented + "/view-000.rgba32f", image, error));
    UNIT_CHECK(ctx, image.rgba == rgba);
}

// ---------------------------------------------------------------------------
// OpenEXR fixtures, written with tinyexr's own save API.
// ---------------------------------------------------------------------------

// RGBA float data as an EXR file (half when requested). Empty on failure.
std::vector<uint8_t> encodeExr(const std::vector<float> & rgba, int width, int height, bool half)
{
    unsigned char * buffer = nullptr;
    const char * error = nullptr;
    const int size = SaveEXRToMemory(rgba.data(), width, height, 4, half ? 1 : 0, &buffer, &error);
    std::vector<uint8_t> out;
    if (size > 0 && buffer != nullptr)
    {
        out.assign(buffer, buffer + size);
    }
    std::free(buffer);
    if (error != nullptr)
    {
        FreeEXRErrorMessage(error);
    }
    return out;
}

// A 2-channel (G, R; no B) float EXR. Built with the header API, since the save helper
// always writes three or four channels.
std::vector<uint8_t> encodeExrGr(int width, int height)
{
    EXRHeader header;
    InitEXRHeader(&header);
    header.compression_type = TINYEXR_COMPRESSIONTYPE_NONE;
    header.num_channels = 2;
    header.channels = static_cast<EXRChannelInfo *>(std::malloc(sizeof(EXRChannelInfo) * 2));
    header.pixel_types = static_cast<int *>(std::malloc(sizeof(int) * 2));
    header.requested_pixel_types = static_cast<int *>(std::malloc(sizeof(int) * 2));
    std::memset(header.channels, 0, sizeof(EXRChannelInfo) * 2);
    std::memcpy(header.channels[0].name, "G", 2);
    std::memcpy(header.channels[1].name, "R", 2);
    for (int c = 0; c < 2; ++c)
    {
        header.channels[c].x_sampling = 1;
        header.channels[c].y_sampling = 1;
        header.pixel_types[c] = TINYEXR_PIXELTYPE_FLOAT;
        header.requested_pixel_types[c] = TINYEXR_PIXELTYPE_FLOAT;
    }

    const size_t count = static_cast<size_t>(width) * static_cast<size_t>(height);
    std::vector<float> green(count, 0.5f);
    std::vector<float> red(count, 0.25f);
    float * planes[2] = {green.data(), red.data()};

    EXRImage image;
    InitEXRImage(&image);
    image.images = reinterpret_cast<unsigned char **>(planes);
    image.num_channels = 2;
    image.width = width;
    image.height = height;

    unsigned char * buffer = nullptr;
    const char * error = nullptr;
    const size_t size = SaveEXRImageToMemory(&image, &header, &buffer, &error);
    std::free(header.channels);
    std::free(header.pixel_types);
    std::free(header.requested_pixel_types);

    std::vector<uint8_t> out;
    if (size > 0 && buffer != nullptr)
    {
        out.assign(buffer, buffer + size);
    }
    std::free(buffer);
    if (error != nullptr)
    {
        FreeEXRErrorMessage(error);
    }
    return out;
}

// 16x16 RGBA ramp with three non-finite samples: NaN in R of pixel 0, +inf in G of
// pixel 5 and -inf in B of pixel 9.
std::vector<float> exrRampWithNonFinite()
{
    std::vector<float> rgba(16 * 16 * 4);
    for (int y = 0; y < 16; ++y)
    {
        for (int x = 0; x < 16; ++x)
        {
            const size_t i = static_cast<size_t>(y * 16 + x) * 4;
            rgba[i] = static_cast<float>(x + y * 16) / 256.0f;
            rgba[i + 1] = 0.5f;
            rgba[i + 2] = 0.25f;
            rgba[i + 3] = 1.0f;
        }
    }
    rgba[0] = kNaN;
    rgba[5 * 4 + 1] = kInf;
    rgba[9 * 4 + 2] = -kInf;
    return rgba;
}

void testExr(Context & ctx, const ScratchDir & dir)
{
    ImageData image;
    std::string error;

    // Float EXR with non-finite samples: counted, not clamped, and the finite values
    // come back exactly. Row 0 is the top row in both directions.
    const std::vector<float> floats = exrRampWithNonFinite();
    const std::vector<uint8_t> floatFile = encodeExr(floats, 16, 16, false);
    UNIT_CHECK(ctx, !floatFile.empty());
    const std::string floatPath = dir.file("float.exr");
    writeBytes(floatPath, floatFile);
    UNIT_CHECK(ctx, tryLoad(floatPath, image, error));
    if (image.rgba.size() == floats.size())
    {
        UNIT_CHECK(ctx, image.width == 16 && image.height == 16);
        UNIT_CHECK(ctx, image.encoding == Encoding::Linear);
        UNIT_CHECK(ctx, image.bitDepth == 32);
        UNIT_CHECK(ctx, image.hasAlpha);
        UNIT_CHECK(ctx, image.nonFiniteCount == 3);
        UNIT_CHECK(ctx, std::isnan(image.rgba[0]));
        UNIT_CHECK(ctx, image.rgba[5 * 4 + 1] == kInf);
        UNIT_CHECK(ctx, image.rgba[9 * 4 + 2] == -kInf);
        UNIT_CHECK(ctx, image.rgba[1] == 0.5f);
        UNIT_CHECK(ctx, image.rgba[(3 * 16 + 2) * 4] == floats[(3 * 16 + 2) * 4]);
        UNIT_CHECK(ctx, image.rgba[(15 * 16 + 15) * 4 + 3] == 1.0f);
    }
    else
    {
        std::cerr << "float EXR: " << error << std::endl;
        UNIT_CHECK(ctx, false);
    }

    // Half EXR: values that are exact in half, plus one +inf.
    std::vector<float> halves(16 * 16 * 4);
    for (size_t i = 0; i < halves.size(); ++i)
    {
        halves[i] = static_cast<float>(i % 64) / 64.0f;
    }
    halves[7] = kInf;
    const std::vector<uint8_t> halfFile = encodeExr(halves, 16, 16, true);
    UNIT_CHECK(ctx, !halfFile.empty());
    const std::string halfPath = dir.file("half.exr");
    writeBytes(halfPath, halfFile);
    UNIT_CHECK(ctx, tryLoad(halfPath, image, error));
    if (image.rgba.size() == halves.size())
    {
        UNIT_CHECK(ctx, image.bitDepth == 16);
        UNIT_CHECK(ctx, image.nonFiniteCount == 1);
        UNIT_CHECK(ctx, image.rgba[7] == kInf);
        UNIT_CHECK(ctx, image.rgba[4] == halves[4]);
        UNIT_CHECK(ctx, image.rgba[10] == halves[10]);
    }
    else
    {
        std::cerr << "half EXR: " << error << std::endl;
        UNIT_CHECK(ctx, false);
    }

    // Non-ASCII folder and file name.
    const std::string folder = dir.folder("caf\xC3\xA9-exr");
    const std::string accented = folder + "/\xC3\xA9t\xC3\xA9.exr";
    writeBytes(accented, floatFile);
    UNIT_CHECK(ctx, tryLoad(accented, image, error));
    UNIT_CHECK(ctx, image.nonFiniteCount == 3);

    // Missing channel B: a 2-channel file.
    const std::vector<uint8_t> grFile = encodeExrGr(4, 4);
    UNIT_CHECK(ctx, !grFile.empty());
    const std::string grPath = dir.file("gr-only.exr");
    writeBytes(grPath, grFile);
    std::string message;
    UNIT_CHECK(ctx, throwsWith([&] { loadImage(grPath); }, "missing channel B", message));

    // Garbage with an .exr name.
    const std::string garbage = dir.file("garbage.exr");
    writeBytes(garbage, garbageBytes(64));
    UNIT_CHECK(ctx, throwsWith([&] { loadImage(garbage); }, "not a valid OpenEXR", message));

    // Truncations: the header, the offset table and the pixel data.
    const size_t cuts[] = {0, 3, 8, 40, floatFile.size() / 2};
    for (const size_t cut : cuts)
    {
        const std::string path = dir.file("truncated-" + std::to_string(cut) + ".exr");
        writeBytes(path, floatFile.data(), cut);
        UNIT_CHECK(ctx, !tryLoad(path, image, error));
    }
}

// ---------------------------------------------------------------------------
// WebP. The animated case is a hand-built header (VP8X with the animation flag and an
// ANIM chunk), so the animation rejection and truncations are checked here. The decode
// path is checked by testWebpLossless, which reads a committed file.
// ---------------------------------------------------------------------------

std::vector<uint8_t> animatedWebp()
{
    std::vector<uint8_t> out;
    appendText(out, "RIFF");
    appendLe32(out, 36); // everything after the first eight bytes: 4 + 18 + 14
    appendText(out, "WEBP");
    // VP8X: flags (bit 1 = animation), three reserved bytes, canvas width - 1 and height - 1 (24 bits each).
    appendText(out, "VP8X");
    appendLe32(out, 10);
    out.insert(out.end(), {0x02, 0, 0, 0, 1, 0, 0, 1, 0, 0});
    // ANIM: background color and loop count.
    appendText(out, "ANIM");
    appendLe32(out, 6);
    out.insert(out.end(), {0, 0, 0, 0, 0, 0});
    return out;
}

void testWebp(Context & ctx, const ScratchDir & dir)
{
    const std::vector<uint8_t> animated = animatedWebp();
    const std::string animPath = dir.file("animated.webp");
    writeBytes(animPath, animated);
    ImageData image;
    std::string message;
    UNIT_CHECK(ctx, throwsWith([&] { loadImage(animPath); }, "animated WebP is not supported", message));

    // Every truncation throws (header not complete, or the animation is refused first).
    for (size_t cut = 0; cut < animated.size(); ++cut)
    {
        const std::string path = dir.file("webp-cut-" + std::to_string(cut) + ".webp");
        writeBytes(path, animated.data(), cut);
        UNIT_CHECK(ctx, !tryLoad(path, image, message));
    }

    const std::string garbage = dir.file("garbage.webp");
    writeBytes(garbage, garbageBytes(64));
    UNIT_CHECK(ctx, !tryLoad(garbage, image, message));
}

void testDispatch(Context & ctx, const ScratchDir & dir)
{
    ImageData image;
    std::string message;

    UNIT_CHECK(ctx, throwsWith([&] { loadImage(""); }, "empty", message));
    const std::string bmp = dir.file("picture.bmp");
    writeBytes(bmp, garbageBytes(16));
    UNIT_CHECK(ctx, throwsWith([&] { loadImage(bmp); }, "unsupported image type", message));
    const std::string noExtension = dir.file("picture");
    writeBytes(noExtension, garbageBytes(16));
    UNIT_CHECK(ctx, throwsWith([&] { loadImage(noExtension); }, "unsupported image type", message));
    UNIT_CHECK(ctx, throwsWith([&] { loadImage(dir.file("does-not-exist.png")); }, "does-not-exist.png", message));
}

// ---------------------------------------------------------------------------
// Two loaders for one PNG file, the alpha of palette and gray+alpha images, the raw
// layout of a dataset_capture file, and the committed lossless WebP fixture.
// ---------------------------------------------------------------------------

constexpr uint32_t kPairWidth = 16;
constexpr uint32_t kPairHeight = 16;

// RGBA8 samples of a kPairWidth x kPairHeight image, straight alpha.
std::vector<uint8_t> pairRgba8(uint32_t seed)
{
    std::vector<uint8_t> rows;
    for (uint32_t i = 0; i < kPairWidth * kPairHeight; ++i)
    {
        const uint32_t n = i * 37u + seed * 101u;
        rows.push_back(static_cast<uint8_t>(n));
        rows.push_back(static_cast<uint8_t>(n * 3u + 7u));
        rows.push_back(static_cast<uint8_t>(n * 5u + 11u));
        rows.push_back(static_cast<uint8_t>(255u - n % 200u));
    }
    return rows;
}

// RGBA16 samples of the same image, big-endian as PNG stores them.
std::vector<uint8_t> pairRgba16(uint32_t seed)
{
    std::vector<uint8_t> rows;
    for (uint32_t i = 0; i < kPairWidth * kPairHeight; ++i)
    {
        for (uint32_t channel = 0; channel < 4; ++channel)
        {
            const uint32_t value = (i * 2531u + channel * 4099u + seed * 977u) & 0xFFFFu;
            rows.push_back(static_cast<uint8_t>(value >> 8));
            rows.push_back(static_cast<uint8_t>(value));
        }
    }
    return rows;
}

// The RGB of the same image, for savePngRgb8.
Rgb8Image pairRgb8(uint32_t seed)
{
    const std::vector<uint8_t> rgba = pairRgba8(seed);
    Rgb8Image image;
    image.width = kPairWidth;
    image.height = kPairHeight;
    for (size_t i = 0; i < static_cast<size_t>(kPairWidth) * kPairHeight; ++i)
    {
        image.rgb.push_back(rgba[i * 4]);
        image.rgb.push_back(rgba[i * 4 + 1]);
        image.rgb.push_back(rgba[i * 4 + 2]);
    }
    return image;
}

// Same size and the same sample bits.
bool sameImageBits(const Image & a, const Image & b)
{
    return a.width == b.width && a.height == b.height && a.rgb.size() == b.rgb.size() &&
           (a.rgb.empty() || std::memcmp(a.rgb.data(), b.rgb.data(), a.rgb.size() * sizeof(float)) == 0);
}

bool sameFloatBits(const std::vector<float> & a, const std::vector<float> & b)
{
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
}

// Checks a reference and a test PNG two ways: loadImage() with alpha dropped (displayRgb,
// the CLI's --report path) and loadPng() (the CLI's plain path). The RGB bits, PSNR, the
// FLIP mean and the FLIP error-map bytes must agree exactly.
void checkPairBothWays(Context & ctx, const std::string & referencePath, const std::string & testPath,
                       const char * label)
{
    ImageData referenceData;
    ImageData testData;
    std::string error;
    if (!tryLoad(referencePath, referenceData, error) || !tryLoad(testPath, testData, error))
    {
        std::cerr << label << ": " << error << std::endl;
        UNIT_CHECK(ctx, false);
        return;
    }
    const Image referenceImage = displayRgb(referenceData);
    const Image testImage = displayRgb(testData);
    const Image referenceLegacy = loadPng(referencePath);
    const Image testLegacy = loadPng(testPath);
    UNIT_CHECK(ctx, sameImageBits(referenceImage, referenceLegacy));
    UNIT_CHECK(ctx, sameImageBits(testImage, testLegacy));

    const PsnrResult psnrImage = psnr(referenceImage, testImage);
    const PsnrResult psnrLegacy = psnr(referenceLegacy, testLegacy);
    UNIT_CHECK(ctx, psnrImage.mse == psnrLegacy.mse);
    UNIT_CHECK(ctx, psnrImage.psnrDb == psnrLegacy.psnrDb);

    const float ppd = defaultFlipPpd();
    const FlipResult flipImage = flip(referenceImage, testImage, ppd);
    const FlipResult flipLegacy = flip(referenceLegacy, testLegacy, ppd);
    UNIT_CHECK(ctx, flipImage.meanError == flipLegacy.meanError);
    UNIT_CHECK(ctx, sameFloatBits(flipImage.errorMap, flipLegacy.errorMap));

    const Rgb8Image mapImage = colorizeMagma(flipImage.errorMap, kPairWidth, kPairHeight);
    const Rgb8Image mapLegacy = colorizeMagma(flipLegacy.errorMap, kPairWidth, kPairHeight);
    UNIT_CHECK(ctx, mapImage.rgb.size() == static_cast<size_t>(kPairWidth) * kPairHeight * 3u);
    UNIT_CHECK(ctx, mapImage.rgb == mapLegacy.rgb);
}

void testPngPathsAgree(Context & ctx, const ScratchDir & dir)
{
    const std::string rgba8Reference = dir.file("pair-rgba8-reference.png");
    const std::string rgba8Test = dir.file("pair-rgba8-test.png");
    writeBytes(rgba8Reference, makePng(kPairWidth, kPairHeight, 8, 6, pairRgba8(1), kPairWidth * 4u));
    writeBytes(rgba8Test, makePng(kPairWidth, kPairHeight, 8, 6, pairRgba8(2), kPairWidth * 4u));
    checkPairBothWays(ctx, rgba8Reference, rgba8Test, "RGBA8 pair");

    const std::string rgba16Reference = dir.file("pair-rgba16-reference.png");
    const std::string rgba16Test = dir.file("pair-rgba16-test.png");
    writeBytes(rgba16Reference, makePng(kPairWidth, kPairHeight, 16, 6, pairRgba16(1), kPairWidth * 8u));
    writeBytes(rgba16Test, makePng(kPairWidth, kPairHeight, 16, 6, pairRgba16(2), kPairWidth * 8u));
    checkPairBothWays(ctx, rgba16Reference, rgba16Test, "RGBA16 pair");

    // The 8-bit RGB PNG that savePngRgb8 writes, against the 16-bit RGBA test image.
    const std::string rgb8Reference = dir.file("pair-rgb8-reference.png");
    savePngRgb8(rgb8Reference, pairRgb8(1));
    checkPairBothWays(ctx, rgb8Reference, rgba16Test, "RGB8 against RGBA16");
}

// Palette with tRNS: four entries with alpha 0, 85, 170 and 255. Pixels 0..3 use entries 0..3.
std::vector<uint8_t> pngPalette4Alpha()
{
    return makePng(4, 1, 8, 3, {0, 1, 2, 3}, 4, {255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255},
                   {0, 85, 170, 255});
}

void testAlphaSources(Context & ctx, const ScratchDir & dir)
{
    ImageData image;
    std::string error;

    // Palette with tRNS: each pixel's alpha is its entry's tRNS byte / 255, exactly.
    const std::string palette = dir.file("palette4-alpha.png");
    writeBytes(palette, pngPalette4Alpha());
    UNIT_CHECK(ctx, tryLoad(palette, image, error));
    if (image.rgba.size() == 16)
    {
        const uint8_t alphas[4] = {0, 85, 170, 255};
        UNIT_CHECK(ctx, image.hasAlpha);
        UNIT_CHECK(ctx, image.bitDepth == 8);
        for (size_t i = 0; i < 4; ++i)
        {
            UNIT_CHECK(ctx, image.rgba[i * 4 + 3] == static_cast<float>(alphas[i]) / 255.0f);
        }
        // The colors come from the palette too: entry 0 is red, entry 3 is white.
        UNIT_CHECK(ctx, image.rgba[0] == 1.0f && image.rgba[1] == 0.0f && image.rgba[2] == 0.0f);
        UNIT_CHECK(ctx, image.rgba[12] == 1.0f && image.rgba[13] == 1.0f && image.rgba[14] == 1.0f);
    }
    else
    {
        std::cerr << "palette tRNS: " << error << std::endl;
        UNIT_CHECK(ctx, false);
    }

    // Gray+alpha8 with two pixels, each with its own alpha: gray replicated, alpha as stored.
    const std::string grayAlpha = dir.file("gray-alpha8-2x1.png");
    writeBytes(grayAlpha, makePng(2, 1, 8, 4, {100, 200, 50, 17}, 4));
    UNIT_CHECK(ctx, tryLoad(grayAlpha, image, error));
    if (image.rgba.size() == 8)
    {
        UNIT_CHECK(ctx, image.hasAlpha);
        UNIT_CHECK(ctx, image.bitDepth == 8);
        UNIT_CHECK(ctx, image.rgba[0] == 100.0f / 255.0f);
        UNIT_CHECK(ctx, image.rgba[1] == 100.0f / 255.0f);
        UNIT_CHECK(ctx, image.rgba[2] == 100.0f / 255.0f);
        UNIT_CHECK(ctx, image.rgba[3] == 200.0f / 255.0f);
        UNIT_CHECK(ctx, image.rgba[4] == 50.0f / 255.0f);
        UNIT_CHECK(ctx, image.rgba[6] == 50.0f / 255.0f);
        UNIT_CHECK(ctx, image.rgba[7] == 17.0f / 255.0f);
    }
    else
    {
        std::cerr << "gray+alpha8 2x1: " << error << std::endl;
        UNIT_CHECK(ctx, false);
    }

    // RGBA16: every sample is code / 65535, alpha included.
    const std::string rgba16 = dir.file("rgba16-exact.png");
    writeBytes(rgba16, pngRgba16());
    UNIT_CHECK(ctx, tryLoad(rgba16, image, error));
    if (image.rgba.size() == 8)
    {
        const uint16_t codes[8] = {65535, 32768, 0, 16384, 1, 2, 3, 65535};
        UNIT_CHECK(ctx, image.hasAlpha);
        UNIT_CHECK(ctx, image.bitDepth == 16);
        for (size_t i = 0; i < 8; ++i)
        {
            UNIT_CHECK(ctx, image.rgba[i] == static_cast<float>(codes[i]) / 65535.0f);
        }
    }
    else
    {
        std::cerr << "rgba16 exact: " << error << std::endl;
        UNIT_CHECK(ctx, false);
    }
}

// A 3x2 capture in the layout examples/dataset_capture writes: view-NNN.rgba32f is
// width * height vec4 floats, row 0 first, and capture.json has the same keys, order and
// formatting. Pixel (x, y) is at float index (y * width + x) * 4.
void testRawLayout(Context & ctx, const ScratchDir & dir)
{
    constexpr uint32_t kWidth = 3;
    constexpr uint32_t kHeight = 2;
    std::vector<float> rgba(static_cast<size_t>(kWidth) * kHeight * 4);
    for (size_t i = 0; i < rgba.size(); ++i)
    {
        rgba[i] = static_cast<float>(i) / 64.0f;
    }
    // Pixel (0, 0) and pixel (2, 1), with values that show a swap of x and y, of
    // channels, or of rows.
    const float topLeft[4] = {0.125f, 0.25f, 0.5f, 0.75f};
    const float bottomRight[4] = {0.875f, 0.625f, 0.375f, 0.0625f};
    std::copy(topLeft, topLeft + 4, rgba.begin());
    std::copy(bottomRight, bottomRight + 4, rgba.begin() + (1 * kWidth + 2) * 4);

    const std::string json =
        "{\n"
        "  \"schemaVersion\": 1,\n"
        "  \"generatorVersion\": \"sabercat-dataset-capture-v1\",\n"
        "  \"sourceFileName\": \"layout.ply\",\n"
        "  \"sourceGaussianCount\": 1,\n"
        "  \"sourceSHDegree\": 0,\n"
        "  \"renderedSHDegree\": 0,\n"
        "  \"antialiased\": false,\n"
        "  \"width\": 3,\n"
        "  \"height\": 2,\n"
        "  \"depthSemantics\": \"frontmost-contributor\",\n"
        "  \"normalization\": {\"center\": [0,0,0], \"scale\": 1},\n"
        "  \"views\": [\n"
        "    {\"id\": \"view-000\", \"rgba\": \"view-000.rgba32f\", \"depth\": \"view-000.depth32f\", "
        "\"fx\": 1.5, \"fy\": 1.5, \"cx\": 1, \"cy\": 0.5, "
        "\"cameraToWorld\": [1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]}\n"
        "  ]\n"
        "}\n";
    const std::string folder = dir.folder("layout-capture");
    writeText(folder + "/capture.json", json);
    writeBytes(folder + "/view-000.rgba32f", rgba.data(), rgba.size() * sizeof(float));

    ImageData image;
    std::string error;
    UNIT_CHECK(ctx, tryLoad(folder + "/view-000.rgba32f", image, error));
    if (image.rgba.size() == rgba.size())
    {
        UNIT_CHECK(ctx, image.width == kWidth && image.height == kHeight);
        UNIT_CHECK(ctx, image.hasAlpha);
        UNIT_CHECK(ctx, image.bitDepth == 32);
        UNIT_CHECK(ctx, image.encoding == Encoding::Display);
        UNIT_CHECK(ctx, image.rgba == rgba);
        // Pixel (0, 0) is the first four floats; pixel (2, 1) is the last four.
        UNIT_CHECK(ctx, image.rgba[0] == topLeft[0] && image.rgba[1] == topLeft[1]);
        UNIT_CHECK(ctx, image.rgba[2] == topLeft[2] && image.rgba[3] == topLeft[3]);
        const size_t last = (static_cast<size_t>(kHeight - 1) * kWidth + (kWidth - 1)) * 4;
        UNIT_CHECK(ctx, last == 20);
        UNIT_CHECK(ctx, image.rgba[last] == bottomRight[0] && image.rgba[last + 1] == bottomRight[1]);
        UNIT_CHECK(ctx, image.rgba[last + 2] == bottomRight[2] && image.rgba[last + 3] == bottomRight[3]);
    }
    else
    {
        std::cerr << "raw layout: " << error << std::endl;
        UNIT_CHECK(ctx, false);
    }
}

// tests/compare_golden/fixtures/webp/lossless_4x3.webp, written by make_webp.py (Pillow
// 12.1.0, libwebp 1.6.0). Every sample is the stored byte / 255, alpha included.
void testWebpLossless(Context & ctx)
{
    const std::string path = std::string(SABERCAT_COMPARE_FIXTURES_DIR) + "/webp/lossless_4x3.webp";
    // Row 0 first; each pixel is r, g, b, a.
    constexpr uint8_t kExpected[4 * 3 * 4] = {
        11, 5, 97, 0,      48, 78, 248, 64,    85, 151, 143, 128,  122, 224, 38, 191,
        159, 41, 189, 255, 196, 114, 84, 17,   233, 187, 235, 200, 14, 4, 130, 254,
        51, 77, 25, 3,     88, 150, 176, 100,  125, 223, 71, 230,  162, 40, 222, 255};

    ImageData image;
    std::string error;
    if (!tryLoad(path, image, error))
    {
        std::cerr << "lossless WebP: " << error << std::endl;
        UNIT_CHECK(ctx, false);
        return;
    }
    UNIT_CHECK(ctx, image.width == 4 && image.height == 3);
    UNIT_CHECK(ctx, image.bitDepth == 8);
    UNIT_CHECK(ctx, image.encoding == Encoding::Display);
    UNIT_CHECK(ctx, image.hasAlpha);
    UNIT_CHECK(ctx, image.rgba.size() == 4 * 3 * 4);
    if (image.rgba.size() == 4 * 3 * 4)
    {
        size_t mismatches = 0;
        for (size_t i = 0; i < image.rgba.size(); ++i)
        {
            if (image.rgba[i] != static_cast<float>(kExpected[i]) / 255.0f)
            {
                if (mismatches == 0)
                {
                    std::cerr << "lossless WebP: sample " << i << " differs" << std::endl;
                }
                ++mismatches;
            }
        }
        UNIT_CHECK(ctx, mismatches == 0);
    }
}

} // namespace

void testCompareLoaders(Context & ctx)
{
    try
    {
        const ScratchDir dir;
        testPngDecoding(ctx, dir);
        testPngRejections(ctx, dir);
        testNonAsciiPaths(ctx, dir);
        testRawCaptures(ctx, dir);
        testRawLayout(ctx, dir);
        testExr(ctx, dir);
        testWebp(ctx, dir);
        testWebpLossless(ctx);
        testDispatch(ctx, dir);
        testPngPathsAgree(ctx, dir);
        testAlphaSources(ctx, dir);
    }
    catch (const std::exception & exception)
    {
        std::cerr << "compare loaders: " << exception.what() << std::endl;
        ++ctx.checks;
        ++ctx.failures;
    }
}

} // namespace sabercat::unit
