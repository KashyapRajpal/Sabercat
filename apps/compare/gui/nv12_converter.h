// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The GPU step of the video sources: one tight NV12 frame (in a storage buffer) to RGBA32F
// texels, BT.709 limited range, with the shader compare_nv12. The CPU reference of the same
// formula is bt709LimitedToRgb(), and bt601LimitedToRgb() is the BT.601 counterpart that the
// tests use to show the output is not 601.
#pragma once

#include <cstdint>
#include <memory>

#include "sabercat/buffer_view.h"

namespace sabercat::compare::gui
{

class ComputeKernel;
class GpuContext;

// BT.709 limited-range NV12 -> RGBA32F (display-encoded R'G'B', opaque, clamped to [0,1], not
// quantized).
class Nv12Converter
{
  public:
    // The context is borrowed and must outlive the converter. Creates the compare_nv12 kernel.
    explicit Nv12Converter(GpuContext & context);
    ~Nv12Converter();

    Nv12Converter(const Nv12Converter &) = delete;
    Nv12Converter & operator=(const Nv12Converter &) = delete;

    // The constants the shader uses, for the CPU reference (the same numbers). Code values
    // are 8-bit: Y' = (y - yOffset) / yScale, Cb' = (cb - cOffset) / cScale, Cr' likewise.
    struct Matrix
    {
        float yOffset;
        float yScale;
        float cOffset;
        float cScale;
        float crToR;
        float cbToG;
        float crToG;
        float cbToB;
    };
    // yOffset 16, yScale 219, cOffset 128, cScale 224, crToR 1.5748, cbToG 0.187324,
    // crToG 0.468124, cbToB 1.8556.
    static Matrix bt709Limited();

    // nv12: a storage buffer holding the tight NV12 bytes; its size must be at least
    // width * height * 3 / 2 rounded up to a multiple of 4. rgba: a storage buffer of
    // width * height RGBA32F texels (at least that many 16-byte elements). Records one
    // dispatch in its own submit and waits for it.
    // Throws std::invalid_argument when width or height is 0 or odd, or a buffer is invalid
    // or too small.
    void convert(const BufferView & nv12, const BufferView & rgba, uint32_t width, uint32_t height);

  private:
    GpuContext & m_context;
    std::unique_ptr<ComputeKernel> m_kernel;
};

// CPU reference of the BT.709 formula for one pixel: code values y, cb, cr in 0..255 give
// r, g, b in [0,1] (clamped).
void bt709LimitedToRgb(float y, float cb, float cr, float rgb[3]);
// The BT.601 limited-range counterpart (Kr 0.299, Kb 0.114: 1.402, 0.344136, 0.714136, 1.772).
// Only for the test that proves the output is not 601.
void bt601LimitedToRgb(float y, float cb, float cr, float rgb[3]);

} // namespace sabercat::compare::gui
