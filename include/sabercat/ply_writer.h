// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <string>

namespace sabercat
{

class GaussianCloud;

// Writes a GaussianCloud's host data as a binary_little_endian 3D Gaussian
// Splatting .ply -- the exact inverse of PLYReader (ply_reader.cpp). Any
// INRIA-style tool reads the result, and so does PLYReader, bit-exactly for
// positions, raw SH and visibility.
//
// Written properties: x y z, nx ny nz, f_dc_0..2, f_rest_0.. (channel-major,
// 3 * ((d + 1)^2 - 1) of them for SH degree d), opacity, scale_0..2, rot_0..3.
// Values are the inverses of PLYReader's activations: opacity as logit(),
// scale as log(), rotation as (w, x, y, z).
//
// When the cloud carries relight streams (see GaussianCloud), it also writes
// the R3DG attributes: base_color_0..2 and roughness (inverse of the reader's
// sigmoid * 0.77 + 0.03 and sigmoid * 0.9 + 0.09), visibility_dc_0 and
// visibility_rest_0..14 (raw), and metallic (raw). See write() for the rules.
class PLYWriter
{
  public:
    // Opacity, base color and roughness are inverted through logit(), which is
    // infinite at 0 and 1. Such values are clamped to [kLogitEpsilon,
    // 1 - kLogitEpsilon] before inversion, so an opacity of 0 is stored as
    // logit(1e-6) and reads back as 1e-6 (not 0). For base color and roughness
    // the clamp is applied to the normalized sigmoid argument, so the reader's
    // output stays inside its open range.
    static constexpr float kLogitEpsilon = 1e-6f;

    // Writes `cloud` to `path` (via a temporary file renamed over `path`, so a
    // failed write never leaves a truncated file in place).
    //
    // Reads the cloud's HOST vectors: a cloud that was uploaded and then had
    // its host copy dropped (borrowed, or never downloaded) must be
    // downloadFromGPU()'d first. Throws std::runtime_error on I/O failure, an
    // empty cloud, a host vector whose size does not match count(), a shDegree
    // above 3, a non-finite position or a scale that is not positive and finite
    // (naming the Gaussian),
    // or a relight stream the format cannot carry (normals or visibility
    // without base color: PLYReader ignores them unless base_color is present).
    //
    // Relight rules: the R3DG attributes are written only when baseColorRoughness
    // is non-empty (it is the marker PLYReader keys on). Then nx ny nz hold the
    // normals, or zeros when normals() is empty -- PLYReader maps zero to zero,
    // so the normals() stream then reads back as zeros rather than absent.
    // metallic is written whenever metallic() is non-empty, because PLYReader
    // reads it independently of base_color. Relight-free clouds carry no
    // base_color, roughness, visibility or metallic property; their nx ny nz are
    // zeros, as in INRIA output, and PLYReader ignores them.
    //
    // Base color and roughness values outside the representable open range
    // are clamped; the count is logged once as a warning.
    static void write(const GaussianCloud & cloud, const std::string & path);
};

} // namespace sabercat
