// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>

namespace sabercat
{

class Device;
class GaussianCloud;

// Loads a Niantic `.spz` scene by delegating to the reference implementation
// (external/spz, MIT, pinned to v3.0.0) and adapting its output onto this
// renderer's canonical contract.
//
// Unlike PLYReader and SplatReader, this class parses NOTHING. `.spz` is a
// versioned container -- v1-3 are a single gzip stream, v4+ is a 32-byte
// plaintext header, a TOC of (compressedSize, uncompressedSize) pairs, and
// per-attribute Zstandard frames -- and a container bug fails only on REAL
// files, which self-synthesized fixtures structurally cannot catch. Delegating
// keeps the only Sabercat-owned code here the activation/layout adaptation,
// which is small and genuinely testable.
//
// What the adapter does (spz's unpacked form -> canonical):
//   position   xyz            -> unchanged
//   scale      xyz, log       -> exp() per axis (same as .ply)
//   rotation   xyzw           -> unchanged ORDER (already canonical -- no
//                                wxyz->xyzw reorder, unlike .ply and .splat),
//                                renormalized defensively
//   alpha      pre-sigmoid    -> sigmoid() (same as .ply)
//   color      raw SH DC      -> straight into SH coefficient 0 (already a
//                                real f_dc, so unlike .splat there is nothing
//                                to invert)
//   sh         coefficient-major, RGB interleaved (sh[k*3 + c]) -> the same
//              band-ascending vec4-per-coefficient layout PLYReader produces.
//              Note this is the OPPOSITE of .ply's channel-major f_rest.
//
// Coordinate system: unpacked as RUB (right/up/back, three.js), which already
// matches this renderer's +Y-up / -Z-forward convention -- so `.spz` should
// need no setFlipUpAxis(), unlike `.ply`'s RDF.
//
// SH degree ceiling: spz supports degree 4 (24 coefficients/channel); this
// renderer's canonical layout holds 16 total (degree 3). Degree-4 input is
// truncated to degree 3 -- a correct band-ascending prefix read, the same
// operation the SH degree cap performs -- and logged, rather than rejected.
class SpzReader
{
  public:
    static std::shared_ptr<GaussianCloud> read(Device & device, const std::string & path);
};

} // namespace sabercat
