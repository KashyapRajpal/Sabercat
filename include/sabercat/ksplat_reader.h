// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>

namespace sabercat
{

class Device;
class GaussianCloud;

// Reads mkkellogg's GaussianSplats3D `.ksplat` format into a GaussianCloud.
//
// There is no written spec: the format is defined by its JS reference
// implementation (SplatBuffer / KSplatLoader / SplatBufferGenerator in
// github.com/mkkellogg/GaussianSplats3D; derived from v0.4.7, the current
// release, whose SplatBuffer.js matches main). docs/splat-formats.md has the
// derivation with line-level provenance; the short form:
//
//   4096-byte header      version u8@0 (major) u8@1 (minor, >= 1), maxSectionCount u32@4,
//                         maxSplatCount u32@12, compressionLevel u16@20, sceneCenter
//                         3xf32@24, SH 8-bit range min f32@36 / max f32@40
//   N x 1024-byte section headers (N = maxSectionCount)
//                         maxSplatCount u32@4, bucketSize u32@8, bucketCount u32@12,
//                         bucketBlockSize f32@16, bucketStorageSizeBytes u16@20,
//                         compressionScaleRange u32@24, fullBucketCount u32@32,
//                         partiallyFilledBucketCount u32@36, shDegree u16@40
//   N sections, each:     partially-filled bucket lengths (u32 each), bucket
//                         centres (3xf32 each), then fixed-size splat records
//
//   splat record          level 0           level 1           level 2
//     centre              3xf32             3xu16 (bucket)    3xu16 (bucket)
//     scale (LINEAR)      3xf32             3xf16             3xf16
//     rotation (WXYZ)     4xf32             4xf16             4xf16
//     colour RGBA         4xu8              4xu8              4xu8
//     SH bands 1-2        f32               f16               u8 over [min,max]
//
// What the reader does to reach the canonical contract:
//   position  level 0 passthrough; levels 1/2 (u16 - range) * (blockSize/2 / range)
//             + the splat's bucket centre
//   scale     ALREADY LINEAR (the exporter exp()s) -- like .splat, unlike .ply
//   rotation  stored w,x,y,z (the JS names the lanes x,y,z,w but fills them from
//             rot_0..rot_3, and its reader reads lane 0 as w) -> reorder to xyzw,
//             normalize
//   colour    display RGB byte, floor((0.5 + SH_C0 * f_dc) * 255) -> invert with
//             b/255, which is also what the reference viewer renders
//   opacity   floor(sigmoid * 255) -> b/255
//   SH        per BAND, channel-major inside the band (band 1: R1..R3 G1..G3
//             B1..B3; band 2: R4..R8 G4..G8 B4..B8) -> canonical coefficient-major
//
// SH degree is at most 2: the reference defines no degree-3 layout (its tables
// stop at 2 and its .ply parser caps at 2), so a degree-3 section is rejected
// rather than guessed at. Coordinates are the source .ply's, untouched -- like
// .ply, a scene usually needs setFlipUpAxis().
class KsplatReader
{
  public:
    static std::shared_ptr<GaussianCloud> read(Device & device, const std::string & path);
};

} // namespace sabercat
