// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "sabercat/file_map.h"

namespace sabercat
{

class GaussianCloud;

// Writes the cloud's HOST vectors as a cloud-v1 directory (schema
// "sabercat.cloud-v1"), the exact inverse of CloudV1Reader (cloud_v1_reader.h).
// The output has the top-level and per-stream manifest keys, the stream names
// and shapes, and the sh0 / sh_rest split (sh_rest only when shDegree > 0) that
// the reader expects. Relight streams are NOT part of cloud-v1; see
// RelightSidecar.
class CloudV1Writer
{
  public:
    // The fixture as in-memory files: manifest.json plus one .bin per stream,
    // keyed by file name only (no directory prefix).
    static FileMap encode(const GaussianCloud & cloud);
    // encode() written into `directory` (created if missing).
    static void write(const GaussianCloud & cloud, const std::string & directory);
};

// Optional relight streams in a sidecar beside a cloud-v1 fixture, so cloud-v1
// stays as it is defined. manifest.json with schema
// "sabercat.relight-v1", count, and a "streams" object holding whichever of
// these the cloud has (each optional, same entry keys as cloud-v1):
//   normals [N,4], base_color_roughness [N,4], metallic [N], visibility [N,16]
//   (16 = kVisibilityCoefficients).
class RelightSidecar
{
  public:
    // Empty map when the cloud has no relight streams.
    static FileMap encode(const GaussianCloud & cloud);
    // Fills the cloud's relight host vectors from a sidecar (files keyed by file
    // name). A stream the sidecar lacks leaves its vector empty. Validates
    // schema, count == cloud.count(), shapes, byteLengths, scalarType; throws
    // std::runtime_error naming the problem. Needs the cloud already initialized
    // to its count (CloudV1Reader or GaussianCloud::initialize).
    static void decode(const FileMap & files, GaussianCloud & cloud, const std::string & label);
};

} // namespace sabercat
