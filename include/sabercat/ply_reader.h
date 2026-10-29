// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>

namespace sabercat
{

class Device;
class GaussianCloud;

// Parses a standard (binary_little_endian) 3D Gaussian Splatting .ply --
// the format produced by the reference INRIA trainer and most public
// pretrained scenes -- into a GaussianCloud's CPU-side SoA data. Does not
// upload to the GPU; call GaussianCloud::uploadToGPU() afterward.
//
// Also reads the SuperSplat / PlayCanvas "compressed" .ply (an `element
// chunk` of per-256-splat bounds, bit-packed `packed_*` vertices, optional
// uchar `sh`), decoded as PlayCanvas's splat-transform does; ply_reader.cpp
// has the bit layout. Malformed or unsupported variants are rejected by name.
class PLYReader
{
  public:
    static std::shared_ptr<GaussianCloud> read(Device & device, const std::string & path);
};

} // namespace sabercat
