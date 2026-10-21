// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>

namespace sabercat
{

class Device;
class GaussianCloud;

// Reads glTF 2.0 (`.gltf` JSON + external/data-URI buffers, or binary `.glb`)
// carrying KHR_gaussian_splatting (ratified) into a GaussianCloud.
//
// Every POINTS primitive with the extension, instantiated by a node of the
// active scene, becomes Gaussians; several instances are concatenated in
// scene-traversal order. Primitives without the extension are ignored.
//
// What the reader does (attribute -> canonical):
//   POSITION                     -> transformed by the node's global matrix
//   KHR_gaussian_splatting:SCALE -> ALREADY LINEAR (like .splat, unlike .ply)
//   ...:ROTATION                 -> ALREADY xyzw (glTF order, like .spz);
//                                   normalized ints renormalized, floats
//                                   passed through unless off-unit
//   ...:OPACITY                  -> ALREADY ACTIVATED [0,1]; out-of-range is
//                                   rejected (the spec calls it invalid)
//   ...:SH_DEGREE_l_COEF_m       -> raw 3DGS SH, same basis, sign convention,
//                                   m-ascending order and +0.5 DC bias as
//                                   this renderer -- passed through, then
//                                   rotated by the node's rotation (bands 1-3)
//
// Coordinate system: glTF is right-handed, +Y up, camera looking down -Z --
// identical to this renderer's (see docs/architecture.md), so no axis flip. Normalized
// integer components decode per the glTF 2.0 spec (e.g. i8: max(c/127, -1)).
// Sparse accessors, bufferView.byteStride and KHR_mesh_quantization-style
// integer POSITIONs are supported.
//
// KHR_gaussian_splatting_compression_spz_2 (nested in the base extension):
// the primitive's data comes from a gzipped SPZ stream in a bufferView,
// decoded with spz in the extension's Left-Up-Front frame, mapped field by
// field to the attributes above (exp scale, sigmoid alpha, colors = SH DC),
// then transformed exactly like uncompressed data. The placeholder
// accessors' counts must match the stream.
//
// See docs/splat-formats.md for the full decision list, including what is
// rejected and why.
class GltfReader
{
  public:
    static std::shared_ptr<GaussianCloud> read(Device & device, const std::string & path);
};

} // namespace sabercat
