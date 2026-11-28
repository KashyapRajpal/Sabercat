// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "nv12_converter.h"

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "gpu_context.h"

namespace sabercat::compare::gui
{

namespace
{

// Workgroup edge of compare_nv12.slang (numthreads(16, 16, 1)).
constexpr uint32_t kTileSize = 16;
// Bytes per RGBA32F texel.
constexpr uint64_t kTexelBytes = 4 * sizeof(float);

// The push constants, mirrored by Nv12Params in compare_nv12.slang (8 bytes).
struct Nv12PushConstants
{
    uint32_t width;
    uint32_t height;
};
static_assert(sizeof(Nv12PushConstants) == 8, "Nv12PushConstants must match the shader's 8-byte block");

// The NV12 bytes a frame of this size needs: Y plane plus the half-size CbCr plane, rounded
// up to a whole number of 32-bit words (the shader reads the packed bytes as uints).
uint64_t nv12BytesNeeded(uint32_t width, uint32_t height)
{
    const uint64_t frame = static_cast<uint64_t>(width) * height * 3 / 2;
    return (frame + 3) / 4 * 4;
}

float clampUnit(float v)
{
    return std::min(std::max(v, 0.0f), 1.0f);
}

} // namespace

Nv12Converter::Nv12Converter(GpuContext & context) : m_context(context)
{
    const std::vector<VkDescriptorType> bindings = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 0: NV12 bytes, packed little endian
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 1: RGBA32F output
    };
    m_kernel = context.makeComputePipeline("compare_nv12", bindings, static_cast<uint32_t>(sizeof(Nv12PushConstants)));
}

Nv12Converter::~Nv12Converter() = default;

Nv12Converter::Matrix Nv12Converter::bt709Limited()
{
    return Matrix{16.0f, 219.0f, 128.0f, 224.0f, 1.5748f, 0.187324f, 0.468124f, 1.8556f};
}

void Nv12Converter::convert(const BufferView & nv12, const BufferView & rgba, uint32_t width, uint32_t height)
{
    if (width == 0 || height == 0 || (width & 1u) != 0 || (height & 1u) != 0)
    {
        throw std::invalid_argument("NV12 conversion needs a non-zero, even width and height, got "
                                    + std::to_string(width) + "x" + std::to_string(height));
    }
    const uint64_t nv12Bytes = nv12BytesNeeded(width, height);
    if (!nv12.valid() || nv12.size < nv12Bytes)
    {
        throw std::invalid_argument("NV12 buffer is missing or smaller than " + std::to_string(nv12Bytes) + " bytes");
    }
    const uint64_t rgbaBytes = static_cast<uint64_t>(width) * height * kTexelBytes;
    if (!rgba.valid() || rgba.size < rgbaBytes)
    {
        throw std::invalid_argument("RGBA32F output buffer is missing or smaller than " + std::to_string(rgbaBytes)
                                    + " bytes");
    }

    m_kernel->bindBuffer(0, nv12);
    m_kernel->bindBuffer(1, rgba);

    const Nv12PushConstants push{width, height};
    const uint32_t groupsX = (width + kTileSize - 1) / kTileSize;
    const uint32_t groupsY = (height + kTileSize - 1) / kTileSize;
    m_context.submit(
        [&](VkCommandBuffer cmd)
        {
            m_kernel->record(cmd, groupsX, groupsY, 1, &push, static_cast<uint32_t>(sizeof(push)));
        });
}

void bt709LimitedToRgb(float y, float cb, float cr, float rgb[3])
{
    const Nv12Converter::Matrix m = Nv12Converter::bt709Limited();
    const float yPrime = (y - m.yOffset) / m.yScale;
    const float cbPrime = (cb - m.cOffset) / m.cScale;
    const float crPrime = (cr - m.cOffset) / m.cScale;
    rgb[0] = clampUnit(yPrime + m.crToR * crPrime);
    rgb[1] = clampUnit(yPrime - m.cbToG * cbPrime - m.crToG * crPrime);
    rgb[2] = clampUnit(yPrime + m.cbToB * cbPrime);
}

void bt601LimitedToRgb(float y, float cb, float cr, float rgb[3])
{
    // Same limited-range offsets and scales as BT.709; only the matrix differs.
    constexpr float kYOffset = 16.0f;
    constexpr float kYScale = 219.0f;
    constexpr float kCOffset = 128.0f;
    constexpr float kCScale = 224.0f;
    constexpr float kCrToR = 1.402f;
    constexpr float kCbToG = 0.344136f;
    constexpr float kCrToG = 0.714136f;
    constexpr float kCbToB = 1.772f;
    const float yPrime = (y - kYOffset) / kYScale;
    const float cbPrime = (cb - kCOffset) / kCScale;
    const float crPrime = (cr - kCOffset) / kCScale;
    rgb[0] = clampUnit(yPrime + kCrToR * crPrime);
    rgb[1] = clampUnit(yPrime - kCbToG * cbPrime - kCrToG * crPrime);
    rgb[2] = clampUnit(yPrime + kCbToB * cbPrime);
}

} // namespace sabercat::compare::gui
