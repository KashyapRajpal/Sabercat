// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace sabercat
{

class GaussianCloud;

// Writes a GaussianCloud's host data as a Niantic `.spz` file -- the exact
// inverse of SpzReader (spz_reader.cpp). Encoding is delegated to the reference
// implementation (external/spz, MIT, pinned to v3.0.0), so the bytes are the
// ones any spz reader expects; this class owns only the canonical-to-spz
// adaptation (the inverses of SpzReader's activations) and the file write.
//
// Container: version 3 (default) is the gzip-wrapped single NGSP stream;
// version 4 is the NGSP container with per-attribute Zstandard frames.
//
// Coordinate system: written as RUB (right/up/back, three.js), the same frame
// SpzReader unpacks to. No axis change is applied in either direction.
//
// Quantization. spz quantizes on pack, so the values handed to it are the
// inverse of SpzReader's activation, and the stored precision follows spz's
// byte and fixed-point formats:
//   position   xyz        spz 24-bit fixed point with 12 fractional bits:
//                         round(x * 4096) must lie in [-2^23, 2^23 - 1], i.e.
//                         x within about +-2048. Positions outside that range,
//                         or non-finite, are rejected.
//   scale      xyz        cloud holds LINEAR scale; written as ln(s) (the
//                         inverse of SpzReader's exp). spz stores
//                         clamp(round((ln s + 10) * 16)) in one byte, so ln s
//                         covers [-10, 255/16 - 10] = [-10, 5.94]. Values
//                         outside that range (including s <= 0) are clamped to
//                         the range and counted; the count is logged once.
//   rotation   xyzw       stored in the same xyzw order (no reorder). spz packs
//                         it as smallest-three in 4 bytes, so q and -q are
//                         equivalent. Each quaternion is normalized before
//                         packing; a near-zero one (squared length <= 1e-12) is
//                         written as the identity and counted, not rejected.
//   alpha      opacity    cloud holds linear opacity in [0, 1]; written as
//                         logit(p) (the inverse of SpzReader's sigmoid). spz
//                         stores round(sigmoid * 255) in one byte. Values
//                         outside [0, 1] are clamped, and p is clamped to
//                         [1e-6, 1 - 1e-6] (PLYWriter::kLogitEpsilon) so the
//                         logit is finite; each clamp is counted and logged.
//   color      DC (SH 0)  written straight through (it is already the raw
//                         f_dc term). spz stores clamp(round(dc * 0.15 * 255 +
//                         127.5)) in one byte, a representable range of about
//                         +-3.33. Values beyond it are saturated by spz and
//                         counted.
//   sh         k = 1..   coefficient-major, RGB interleaved (sh[k*3 + c]), the
//                         layout SpzReader reads. spz stores
//                         clamp(round(x * 128) + 128) and so covers
//                         [-1, 127/128]; values outside are saturated and
//                         counted. spz buckets degree-1 coefficients to 5 bits
//                         and the rest to 4 bits (its default sh1Bits and
//                         shRestBits), which is lossy beyond the range above.
//
// Fields not in the spz format are not written: relight streams (base color,
// roughness, normals, visibility, metallic), time and layers. The antialiased
// flag and the SH degree are carried. Degree above 3 is rejected rather than
// clamped, since truncating the SH bands here would silently change the scene.
//
// Reads the cloud's HOST vectors: a cloud that was uploaded and then had its
// host copy dropped (borrowed, or never downloaded) must be downloadFromGPU()'d
// first.
class SPZWriter
{
  public:
    static constexpr uint32_t kDefaultVersion = 3; // gzip container; 4 = zstd NGSP container
    static constexpr uint32_t kMaxShDegree = 3;
    // spz does int32 index arithmetic over the packed streams; 72 floats is the
    // per-Gaussian SH size at degree 4 (3 * 24), the largest spz allows.
    static constexpr size_t kMaxGaussians = static_cast<size_t>(INT32_MAX / 72);

    // Writes `cloud` to `path` as a .spz of the given container `version`. The
    // whole file is packed into memory first and written with a single stream
    // write to `path` + ".tmp", then renamed over `path`, so a failed write never
    // leaves a truncated file in place (the temporary is removed on failure).
    //
    // Throws std::runtime_error, with a message beginning "SPZWriter: ", for:
    //   - version other than 3 or 4;
    //   - an empty cloud (count() == 0);
    //   - count() above kMaxGaussians;
    //   - a host vector whose size does not match count() (positions, scales,
    //     rotations, opacities must be count(); shCoefficients must be
    //     count() * kShCoefficientsPerGaussian) -- call downloadFromGPU() first;
    //   - shDegree() above kMaxShDegree (3);
    //   - a non-finite position, or a position outside about +-2048 (names the
    //     Gaussian and the value);
    //   - a non-finite scale, rotation component, opacity, DC color or SH
    //     coefficient (names the Gaussian and the field);
    //   - spz failing to pack the cloud, or an I/O failure on the temporary file
    //     or the rename.
    // Clamped, normalized or saturated values that are in range for the format
    // (see above) do not throw; their counts are logged once as a warning.
    static void write(const GaussianCloud & cloud, const std::string & path,
                      uint32_t version = kDefaultVersion);
};

} // namespace sabercat
