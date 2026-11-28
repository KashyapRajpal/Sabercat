// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// NV12 to RGBA32F for the video sources: one thread per pixel, row-major (width x height).
// The input is one tight NV12 frame, packed as little-endian bytes (byte i is bits 8 * (i & 3)
// of word i >> 2): the Y plane (width * height bytes), then the interleaved CbCr plane
// (width * height / 2 bytes: each row pair holds one Cb, Cr pair per 2x2 block).
//
// Chroma is nearest (each 2x2 block of pixels reads the same Cb and Cr); there is no
// bilinear upsampling. The color is BT.709 limited range (Y code values 16..235, Cb and Cr
// 16..240 centered on 128), with the luma coefficients Kr 0.2126, Kg 0.7152 and Kb 0.0722:
//
//   Y' = (Y - 16) / 219,  Cb' = (Cb - 128) / 224,  Cr' = (Cr - 128) / 224
//   R = Y' + 1.5748 Cr'
//   G = Y' - 0.187324 Cb' - 0.468124 Cr'
//   B = Y' + 1.8556 Cb'
//
// Each channel is clamped to [0,1]. The output is display-encoded R'G'B' (the values a PNG
// stores): no transfer function is applied, and alpha is 1. The CPU reference of the same
// numbers is bt709LimitedToRgb() in nv12_converter.h.

// Mirrored by Nv12Converter's push constants (8 bytes).
struct Nv12Params
{
    uint width;
    uint height;
};

[vk::push_constant] ConstantBuffer<Nv12Params> params;

[vk::binding(0, 0)] StructuredBuffer<uint> nv12;
[vk::binding(1, 0)] RWStructuredBuffer<float4> outputRgba;

// BT.709 limited range, in 8-bit code values (see the header comment).
static const float kYOffset = 16.0f;
static const float kYScale = 219.0f;
static const float kCOffset = 128.0f;
static const float kCScale = 224.0f;
static const float kCrToR = 1.5748f;
static const float kCbToG = 0.187324f;
static const float kCrToG = 0.468124f;
static const float kCbToB = 1.8556f;

// Byte i of the packed NV12 frame, 0..255.
uint nv12Byte(uint i)
{
    return (nv12[i >> 2u] >> ((i & 3u) * 8u)) & 0xFFu;
}

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    const uint x = id.x;
    const uint y = id.y;
    if (x >= params.width || y >= params.height)
    {
        return;
    }

    const float lumaCode = float(nv12Byte(y * params.width + x));
    // The CbCr plane follows the Y plane; each 2x2 block shares one (Cb, Cr) pair.
    const uint chroma = params.width * params.height + (y >> 1u) * params.width + (x >> 1u) * 2u;
    const float cbCode = float(nv12Byte(chroma));
    const float crCode = float(nv12Byte(chroma + 1u));

    const float yPrime = (lumaCode - kYOffset) / kYScale;
    const float cbPrime = (cbCode - kCOffset) / kCScale;
    const float crPrime = (crCode - kCOffset) / kCScale;

    const float r = clamp(yPrime + kCrToR * crPrime, 0.0f, 1.0f);
    const float g = clamp(yPrime - kCbToG * cbPrime - kCrToG * crPrime, 0.0f, 1.0f);
    const float b = clamp(yPrime + kCbToB * cbPrime, 0.0f, 1.0f);

    outputRgba[y * params.width + x] = float4(r, g, b, 1.0f);
}
