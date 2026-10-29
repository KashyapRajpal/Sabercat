// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>

#include "sabercat/file_map.h"

namespace sabercat
{

class Device;
class GaussianCloud;

// Reads Sabercat's cloud-v1 exchange format: a directory of little-endian
// float32 SoA streams described by a `manifest.json`. `path` is either that
// directory or the manifest itself. Only the schema id `sabercat.cloud-v1` is
// accepted; see cloud_v1_reader.cpp.
//
// The values are already canonical (activated scale/opacity, normalized xyzw,
// raw SH, right-handed +Y up), so means/scales/rotations/opacities are a
// straight copy at the 16 B / 4 B canonical strides. SH is the one stream that
// is NOT byte-compatible: the fixture splits it into `sh0` ([N,4], 16 B) and
// `sh_rest` ([N, (d+1)^2-1, 3], 12 B per coefficient), so the reader merges
// them into the padded 16-coefficient-per-Gaussian layout. `confidence` is
// optional and ignored (GaussianCloud has no confidence stream).
//
// Every manifest field that would change the meaning of the bytes (schema,
// coordinate system, quaternion order, SH layout, scalar type, shapes,
// strides, byte lengths) is checked against what this reader implements, and
// a mismatch is rejected rather than reinterpreted.
class CloudV1Reader
{
  public:
    static std::shared_ptr<GaussianCloud> read(Device & device, const std::string & path);
    // The same fixture held in memory: files keyed by file name (manifest.json
    // plus the streams, no directory prefix). `label` names it in messages.
    static std::shared_ptr<GaussianCloud> read(Device & device, const FileMap & files,
                                               const std::string & label);
};

} // namespace sabercat
