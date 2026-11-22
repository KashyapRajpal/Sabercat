// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "image_io.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <objbase.h>
#include <wincodec.h>
#endif

#include <algorithm>

namespace sabercat::compare
{

#ifdef _WIN32

namespace
{

// Converts UTF-8 to UTF-16 for the wide-path WIC calls. Empty on failure.
std::wstring utf8ToWide(const std::string & utf8)
{
    if (utf8.empty())
    {
        return std::wstring();
    }
    const int needed = MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()),
                                           nullptr, 0);
    if (needed <= 0)
    {
        return std::wstring();
    }
    std::wstring wide(static_cast<size_t>(needed), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), wide.data(), needed);
    return wide;
}

// Balances CoInitializeEx on the calling thread. S_OK and S_FALSE succeed and
// must be matched by CoUninitialize. RPC_E_CHANGED_MODE means the thread already
// uses the other apartment model: COM works there and it is not ours to
// uninitialize, so it is not balanced.
class ComApartment
{
public:
    ComApartment() : hr_(CoInitializeEx(nullptr, COINIT_MULTITHREADED)) {}
    ~ComApartment()
    {
        if (SUCCEEDED(hr_))
        {
            CoUninitialize();
        }
    }
    ComApartment(const ComApartment &) = delete;
    ComApartment & operator=(const ComApartment &) = delete;

    bool usable() const { return SUCCEEDED(hr_) || hr_ == RPC_E_CHANGED_MODE; }
    HRESULT hr() const { return hr_; }

private:
    HRESULT hr_;
};

// Releases a COM interface when it leaves scope.
template <class T>
class ComPtr
{
public:
    ComPtr() = default;
    ~ComPtr()
    {
        if (ptr_ != nullptr)
        {
            ptr_->Release();
        }
    }
    ComPtr(const ComPtr &) = delete;
    ComPtr & operator=(const ComPtr &) = delete;
    ComPtr(ComPtr && other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
    ComPtr & operator=(ComPtr &&) = delete;

    T * get() const { return ptr_; }
    T * operator->() const { return ptr_; }
    T ** put() { return &ptr_; }

private:
    T * ptr_ = nullptr;
};

// Throws with the operation, the file and the HRESULT in hex.
[[noreturn]] void throwWic(const char * operation, const std::string & utf8Path, HRESULT hr)
{
    char code[16];
    std::snprintf(code, sizeof(code), "0x%08lX", static_cast<unsigned long>(hr));
    throw std::runtime_error(std::string(operation) + " '" + utf8Path + "' failed (HRESULT " + code + ")");
}

// Checks an HRESULT and throws on failure.
void check(HRESULT hr, const char * operation, const std::string & utf8Path)
{
    if (FAILED(hr))
    {
        throwWic(operation, utf8Path, hr);
    }
}

// Number of interleaved samples for a width x height RGB image, rejecting zero
// sizes and anything above kMaxSamples. Computed in 64 bits so the product
// cannot wrap.
size_t sampleCount(UINT width, UINT height, const std::string & utf8Path)
{
    if (width == 0 || height == 0)
    {
        throw std::runtime_error("image has zero size: '" + utf8Path + "'");
    }
    const uint64_t samples = static_cast<uint64_t>(width) * static_cast<uint64_t>(height) * 3u;
    if (samples > kMaxSamples)
    {
        throw std::runtime_error("image is too large for the compare tools: '" + utf8Path + "'");
    }
    return static_cast<size_t>(samples);
}

ComPtr<IWICImagingFactory> createFactory(const std::string & utf8Path)
{
    ComPtr<IWICImagingFactory> factory;
    const HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER,
                                        IID_PPV_ARGS(factory.put()));
    check(hr, "creating the WIC imaging factory for", utf8Path);
    return factory;
}

bool isIndexed(const WICPixelFormatGUID & format)
{
    return IsEqualGUID(format, GUID_WICPixelFormat1bppIndexed) || IsEqualGUID(format, GUID_WICPixelFormat2bppIndexed)
        || IsEqualGUID(format, GUID_WICPixelFormat4bppIndexed) || IsEqualGUID(format, GUID_WICPixelFormat8bppIndexed);
}

// Whether the format itself carries alpha. Indexed formats are decided by their
// palette instead.
bool formatHasAlpha(const WICPixelFormatGUID & format)
{
    return IsEqualGUID(format, GUID_WICPixelFormat32bppBGRA) || IsEqualGUID(format, GUID_WICPixelFormat32bppRGBA)
        || IsEqualGUID(format, GUID_WICPixelFormat32bppPBGRA) || IsEqualGUID(format, GUID_WICPixelFormat64bppRGBA);
}

// How loadWicRgba() reads a WIC pixel format.
enum class WicLayout
{
    Gray16, // 16-bit gray: copied as stored and replicated, no converter
    Bgra8,  // converted to 32bppBGRA (straight alpha)
    Rgba16, // converted to 64bppRGBA (straight alpha)
    Cmyk,   // CMYK: refused, WIC has no CMYK-to-RGB conversion
    Unknown // anything else: refused
};

WicLayout classifyLayout(const WICPixelFormatGUID & format)
{
    if (IsEqualGUID(format, GUID_WICPixelFormat16bppGray))
    {
        return WicLayout::Gray16;
    }
    if (IsEqualGUID(format, GUID_WICPixelFormat48bppRGB) || IsEqualGUID(format, GUID_WICPixelFormat64bppRGBA))
    {
        return WicLayout::Rgba16;
    }
    if (IsEqualGUID(format, GUID_WICPixelFormat32bppCMYK) || IsEqualGUID(format, GUID_WICPixelFormat40bppCMYKAlpha))
    {
        return WicLayout::Cmyk;
    }
    if (isIndexed(format) || IsEqualGUID(format, GUID_WICPixelFormat2bppGray)
        || IsEqualGUID(format, GUID_WICPixelFormat4bppGray) || IsEqualGUID(format, GUID_WICPixelFormat8bppGray)
        || IsEqualGUID(format, GUID_WICPixelFormat24bppBGR) || IsEqualGUID(format, GUID_WICPixelFormat24bppRGB)
        || IsEqualGUID(format, GUID_WICPixelFormat32bppBGR) || formatHasAlpha(format))
    {
        return WicLayout::Bgra8;
    }
    return WicLayout::Unknown;
}

} // namespace

Image loadPng(const std::string & utf8Path)
{
    ComApartment com;
    if (!com.usable())
    {
        throwWic("COM initialization for", utf8Path, com.hr());
    }
    const std::wstring widePath = utf8ToWide(utf8Path);
    if (widePath.empty())
    {
        throw std::runtime_error("PNG path is empty or not convertible to UTF-16: '" + utf8Path + "'");
    }

    ComPtr<IWICImagingFactory> factory = createFactory(utf8Path);

    ComPtr<IWICBitmapDecoder> decoder;
    check(factory->CreateDecoderFromFilename(widePath.c_str(), nullptr, GENERIC_READ,
                                             WICDecodeMetadataCacheOnDemand, decoder.put()),
          "opening the PNG", utf8Path);

    ComPtr<IWICBitmapFrameDecode> frame;
    check(decoder->GetFrame(0, frame.put()), "reading frame 0 of the PNG", utf8Path);

    UINT width = 0;
    UINT height = 0;
    check(frame->GetSize(&width, &height), "reading the PNG size", utf8Path);
    const size_t samples = sampleCount(width, height, utf8Path);

    WICPixelFormatGUID sourceFormat = {};
    check(frame->GetPixelFormat(&sourceFormat), "reading the PNG pixel format", utf8Path);

    Image image;
    image.width = width;
    image.height = height;
    image.rgb.resize(samples);

    // Divided, not multiplied by a rounded 1/65535: each value is then the correctly
    // rounded code / 65535, the same bits loadWicRgba() returns for the same file.
    constexpr float kMaxValue = 65535.0f;

    if (IsEqualGUID(sourceFormat, GUID_WICPixelFormat16bppGray))
    {
        // 16-bit gray is copied from the frame as stored, with no converter, so
        // each value is the file's encoded value. It is replicated to r = g = b.
        // One row at a time; the row stride must still fit a UINT.
        const uint64_t strideBytes = static_cast<uint64_t>(width) * 2u;
        if (strideBytes > UINT32_MAX)
        {
            throw std::runtime_error("PNG row is too wide: '" + utf8Path + "'");
        }
        const UINT stride = static_cast<UINT>(strideBytes);
        std::vector<uint16_t> rowGray(static_cast<size_t>(width));
        for (UINT y = 0; y < height; ++y)
        {
            const WICRect row{0, static_cast<INT>(y), static_cast<INT>(width), 1};
            check(frame->CopyPixels(&row, stride, stride, reinterpret_cast<BYTE *>(rowGray.data())),
                  "copying PNG pixels", utf8Path);
            float * const out = image.rgb.data() + static_cast<size_t>(y) * width * 3u;
            const uint16_t * const in = rowGray.data();
            for (size_t x = 0; x < width; ++x)
            {
                const float value = static_cast<float>(in[x]) / kMaxValue;
                out[x * 3] = value;
                out[x * 3 + 1] = value;
                out[x * 3 + 2] = value;
            }
        }
        return image;
    }

    // Converts every other format to 16-bit straight-alpha RGBA. No color
    // management: the values are the file's encoded values (8-bit ones scaled
    // by 257). A float target would not do: WIC then converts sRGB-encoded
    // 8-bit and 16-bit data to linear light, which is not what PSNR (or FLIP's
    // own sRGB decode) expects.
    ComPtr<IWICFormatConverter> converter;
    check(factory->CreateFormatConverter(converter.put()), "creating the PNG format converter", utf8Path);
    check(converter->Initialize(frame.get(), GUID_WICPixelFormat64bppRGBA, WICBitmapDitherTypeNone,
                                nullptr, 0.0, WICBitmapPaletteTypeCustom),
          "converting the PNG to 16-bit RGBA", utf8Path);

    // One row at a time, so a large image never needs one CopyPixels buffer
    // size above UINT32_MAX. The row stride must still fit a UINT.
    const uint64_t strideBytes = static_cast<uint64_t>(width) * 8u;
    if (strideBytes > UINT32_MAX)
    {
        throw std::runtime_error("PNG row is too wide: '" + utf8Path + "'");
    }
    const UINT stride = static_cast<UINT>(strideBytes);
    std::vector<uint16_t> rowRgba(static_cast<size_t>(width) * 4u);
    for (UINT y = 0; y < height; ++y)
    {
        const WICRect row{0, static_cast<INT>(y), static_cast<INT>(width), 1};
        check(converter->CopyPixels(&row, stride, stride, reinterpret_cast<BYTE *>(rowRgba.data())),
              "copying PNG pixels", utf8Path);
        // Drop alpha and scale to [0,1].
        float * const out = image.rgb.data() + static_cast<size_t>(y) * width * 3u;
        const uint16_t * const in = rowRgba.data();
        for (size_t x = 0; x < width; ++x)
        {
            out[x * 3] = static_cast<float>(in[x * 4]) / kMaxValue;
            out[x * 3 + 1] = static_cast<float>(in[x * 4 + 1]) / kMaxValue;
            out[x * 3 + 2] = static_cast<float>(in[x * 4 + 2]) / kMaxValue;
        }
    }
    return image;
}

ImageData loadWicRgba(const std::string & utf8Path)
{
    ComApartment com;
    if (!com.usable())
    {
        throwWic("COM initialization for", utf8Path, com.hr());
    }
    const std::wstring widePath = utf8ToWide(utf8Path);
    if (widePath.empty())
    {
        throw std::runtime_error("image path is empty or not convertible to UTF-16: '" + utf8Path + "'");
    }

    ComPtr<IWICImagingFactory> factory = createFactory(utf8Path);

    ComPtr<IWICBitmapDecoder> decoder;
    check(factory->CreateDecoderFromFilename(widePath.c_str(), nullptr, GENERIC_READ,
                                             WICDecodeMetadataCacheOnDemand, decoder.put()),
          "opening the image", utf8Path);

    ComPtr<IWICBitmapFrameDecode> frame;
    check(decoder->GetFrame(0, frame.put()), "reading frame 0 of the image", utf8Path);

    UINT width = 0;
    UINT height = 0;
    check(frame->GetSize(&width, &height), "reading the image size", utf8Path);
    if (width == 0 || height == 0)
    {
        throw std::runtime_error("image has zero size: '" + utf8Path + "'");
    }
    const size_t samples = rgbaSampleCount(width, height, utf8Path);

    WICPixelFormatGUID sourceFormat = {};
    check(frame->GetPixelFormat(&sourceFormat), "reading the image pixel format", utf8Path);
    const WicLayout layout = classifyLayout(sourceFormat);
    if (layout == WicLayout::Cmyk)
    {
        throw std::runtime_error("CMYK images are not supported (WIC cannot convert CMYK to RGB): '" + utf8Path + "'");
    }
    if (layout == WicLayout::Unknown)
    {
        throw std::runtime_error("unsupported pixel format in '" + utf8Path + "'");
    }

    // Indexed images: the palette decides whether there is alpha (tRNS in PNG).
    bool hasAlpha = formatHasAlpha(sourceFormat);
    ComPtr<IWICPalette> palette;
    if (isIndexed(sourceFormat))
    {
        check(factory->CreatePalette(palette.put()), "creating the palette for", utf8Path);
        check(frame->CopyPalette(palette.get()), "reading the palette of", utf8Path);
        BOOL paletteHasAlpha = FALSE;
        check(palette->HasAlpha(&paletteHasAlpha), "reading the palette transparency of", utf8Path);
        hasAlpha = paletteHasAlpha != FALSE;
    }

    ImageData image;
    image.width = width;
    image.height = height;
    image.hasAlpha = hasAlpha;
    image.encoding = Encoding::Display;
    image.rgba.resize(samples);

    if (layout == WicLayout::Gray16)
    {
        // Same as loadPng: copied from the frame as stored (no converter), so each
        // value is the file's encoded value, replicated to r = g = b.
        image.bitDepth = 16;
        const uint64_t strideBytes = static_cast<uint64_t>(width) * 2u;
        if (strideBytes > UINT32_MAX)
        {
            throw std::runtime_error("image row is too wide: '" + utf8Path + "'");
        }
        const UINT stride = static_cast<UINT>(strideBytes);
        std::vector<uint16_t> rowGray(static_cast<size_t>(width));
        for (UINT y = 0; y < height; ++y)
        {
            const WICRect row{0, static_cast<INT>(y), static_cast<INT>(width), 1};
            check(frame->CopyPixels(&row, stride, stride, reinterpret_cast<BYTE *>(rowGray.data())),
                  "copying image pixels", utf8Path);
            float * const out = image.rgba.data() + static_cast<size_t>(y) * width * 4u;
            for (size_t x = 0; x < width; ++x)
            {
                const float value = static_cast<float>(rowGray[x]) / 65535.0f;
                out[x * 4] = value;
                out[x * 4 + 1] = value;
                out[x * 4 + 2] = value;
                out[x * 4 + 3] = 1.0f;
            }
        }
        return image;
    }

    // Everything else is converted with WIC to 8-bit BGRA or 16-bit RGBA, both
    // straight alpha, with no color management: the values are the file's encoded
    // values. Palette entries are expanded through the palette.
    const bool sixteen = layout == WicLayout::Rgba16;
    image.bitDepth = sixteen ? 16u : 8u;
    const WICPixelFormatGUID target = sixteen ? GUID_WICPixelFormat64bppRGBA : GUID_WICPixelFormat32bppBGRA;
    ComPtr<IWICFormatConverter> converter;
    check(factory->CreateFormatConverter(converter.put()), "creating the format converter for", utf8Path);
    check(converter->Initialize(frame.get(), target, WICBitmapDitherTypeNone, palette.get(), 0.0,
                                WICBitmapPaletteTypeCustom),
          "converting the image pixels", utf8Path);

    // One row at a time, so the row buffer stays small. The row stride must fit a UINT.
    const uint64_t bytesPerPixel = sixteen ? 8u : 4u;
    const uint64_t strideBytes = static_cast<uint64_t>(width) * bytesPerPixel;
    if (strideBytes > UINT32_MAX)
    {
        throw std::runtime_error("image row is too wide: '" + utf8Path + "'");
    }
    const UINT stride = static_cast<UINT>(strideBytes);
    std::vector<uint8_t> row(static_cast<size_t>(strideBytes));
    for (UINT y = 0; y < height; ++y)
    {
        const WICRect rect{0, static_cast<INT>(y), static_cast<INT>(width), 1};
        check(converter->CopyPixels(&rect, stride, stride, row.data()), "copying image pixels", utf8Path);
        float * const out = image.rgba.data() + static_cast<size_t>(y) * width * 4u;
        if (sixteen)
        {
            // 64bppRGBA: R, G, B, A as uint16.
            const uint16_t * const in = reinterpret_cast<const uint16_t *>(row.data());
            for (size_t i = 0; i < static_cast<size_t>(width) * 4u; ++i)
            {
                out[i] = static_cast<float>(in[i]) / 65535.0f;
            }
        }
        else
        {
            // 32bppBGRA: B, G, R, A as bytes.
            for (size_t x = 0; x < width; ++x)
            {
                const uint8_t * const px = row.data() + x * 4;
                out[x * 4] = static_cast<float>(px[2]) / 255.0f;
                out[x * 4 + 1] = static_cast<float>(px[1]) / 255.0f;
                out[x * 4 + 2] = static_cast<float>(px[0]) / 255.0f;
                out[x * 4 + 3] = static_cast<float>(px[3]) / 255.0f;
            }
        }
        if (!hasAlpha)
        {
            for (size_t x = 0; x < width; ++x)
            {
                out[x * 4 + 3] = 1.0f;
            }
        }
    }
    return image;
}

void savePngRgb8(const std::string & utf8Path, const Rgb8Image & image)
{
    const size_t samples = sampleCount(image.width, image.height, utf8Path);
    if (image.rgb.size() != samples)
    {
        throw std::runtime_error("PNG pixel buffer holds " + std::to_string(image.rgb.size())
                                 + " bytes, expected " + std::to_string(samples) + " for '" + utf8Path + "'");
    }

    ComApartment com;
    if (!com.usable())
    {
        throwWic("COM initialization for", utf8Path, com.hr());
    }
    const std::wstring widePath = utf8ToWide(utf8Path);
    if (widePath.empty())
    {
        throw std::runtime_error("PNG path is empty or not convertible to UTF-16: '" + utf8Path + "'");
    }

    ComPtr<IWICImagingFactory> factory = createFactory(utf8Path);

    ComPtr<IWICStream> stream;
    check(factory->CreateStream(stream.put()), "creating the output stream for", utf8Path);
    check(stream->InitializeFromFilename(widePath.c_str(), GENERIC_WRITE), "opening the output file", utf8Path);

    ComPtr<IWICBitmapEncoder> encoder;
    check(factory->CreateEncoder(GUID_ContainerFormatPng, nullptr, encoder.put()),
          "creating the PNG encoder for", utf8Path);
    check(encoder->Initialize(stream.get(), WICBitmapEncoderNoCache), "initializing the PNG encoder for", utf8Path);

    ComPtr<IWICBitmapFrameEncode> frame;
    check(encoder->CreateNewFrame(frame.put(), nullptr), "creating the PNG frame for", utf8Path);
    check(frame->Initialize(nullptr), "initializing the PNG frame for", utf8Path);
    check(frame->SetSize(image.width, image.height), "setting the PNG size for", utf8Path);

    // The PNG encoder takes 24-bit BGR, not RGB.
    WICPixelFormatGUID format = GUID_WICPixelFormat24bppBGR;
    check(frame->SetPixelFormat(&format), "setting the PNG pixel format for", utf8Path);
    if (!IsEqualGUID(format, GUID_WICPixelFormat24bppBGR))
    {
        throw std::runtime_error("the WIC PNG encoder does not accept 24-bit BGR for '" + utf8Path + "'");
    }

    // One swapped copy, made once for the whole image.
    std::vector<BYTE> bgr(samples);
    const uint8_t * const src = image.rgb.data();
    BYTE * const dst = bgr.data();
    for (size_t i = 0; i < samples; i += 3)
    {
        dst[i] = src[i + 2];
        dst[i + 1] = src[i + 1];
        dst[i + 2] = src[i];
    }

    // samples <= kMaxSamples, so the stride and the buffer size fit a UINT.
    const UINT stride = image.width * 3u;
    check(frame->WritePixels(image.height, stride, static_cast<UINT>(samples), dst),
          "writing PNG pixels", utf8Path);
    check(frame->Commit(), "committing the PNG frame for", utf8Path);
    check(encoder->Commit(), "committing the PNG file", utf8Path);
}

#else // !_WIN32

// PNG I/O is WIC-based and exists only on Windows.

Image loadPng(const std::string & utf8Path)
{
    throw std::runtime_error("PNG input needs Windows (WIC): '" + utf8Path + "'");
}

ImageData loadWicRgba(const std::string & utf8Path)
{
    throw std::runtime_error("PNG and JPEG input needs Windows (WIC): '" + utf8Path + "'");
}

void savePngRgb8(const std::string & utf8Path, const Rgb8Image &)
{
    throw std::runtime_error("PNG output needs Windows (WIC): '" + utf8Path + "'");
}

#endif // _WIN32

} // namespace sabercat::compare
