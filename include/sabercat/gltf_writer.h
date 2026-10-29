// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "sabercat/layer_transform.h"

namespace sabercat
{

class GaussianCloud;

// One layer of a glTF export: its host vectors, the node transform it is placed
// with, and the node name it is written under.
struct GLTFExportLayer
{
    const GaussianCloud * cloud = nullptr; // host vectors are read
    LayerTransform transform;
    std::string name;
};

// Writes KHR_gaussian_splatting (ratified) as a single .glb: one mesh and one node
// per layer, the node's TRS being the layer transform and its name the layer name.
// Each mesh has one POINTS primitive with POSITION (VEC3), KHR_gaussian_splatting:
// ROTATION (VEC4, xyzw), :SCALE (VEC3, linear), :OPACITY (SCALAR, linear) and
// SH_DEGREE_l_COEF_m for l = 0..shDegree (VEC3, raw SH) -- all float32. Extension
// object: kernel "ellipse", colorSpace "srgb_rec709_display".
//
// extensionsUsed lists KHR_gaussian_splatting; extensionsRequired is not written,
// because the ratified spec only says extensions that build on this one MAY
// require it, and gives a loader that does not support it the base glTF point
// cloud as a fallback. Relight streams are not exported.
class GLTFWriter
{
  public:
    // Throws std::runtime_error on an empty layer list, a layer with no Gaussians,
    // or data the reader would reject (non-finite values, opacity outside [0, 1],
    // negative scale, a rotation that is not unit length). Written to a temporary file and then
    // renamed over `path`, so a failed write never leaves a partial .glb there.
    static void writeGLB(const std::vector<GLTFExportLayer> & layers, const std::string & path);
};

} // namespace sabercat
