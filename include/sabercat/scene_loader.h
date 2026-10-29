// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>

namespace sabercat
{

class Device;
class GaussianCloud;

// File format detected by extension and magic bytes.
enum class SceneFormat
{
    PLY,   // 3DGS reference format (.ply)
    SPLAT, // antimatter15 splat format (.splat)
    SPZ,   // Niantic compressed format (.spz)
    GLTF,     // glTF 2.0 + KHR_gaussian_splatting (.gltf, .glb)
    CLOUD_V1, // Sabercat cloud-v1 (manifest.json, or its directory)
    KSPLAT, // mkkellogg GaussianSplats3D (.ksplat)
    SOG,    // PlayCanvas SOG: a .sog ZIP bundle, or an unbundled meta.json
};

// Detect scene format from file path.
// Extension is the primary signal; magic bytes can veto (e.g., a file with
// the wrong extension is rejected).
SceneFormat detectFormat(const std::string & path);

// Load a Gaussian cloud from a file in any supported format.
// Throws std::runtime_error if the format is unsupported or the file is invalid.
std::shared_ptr<GaussianCloud> loadScene(Device & device, const std::string & path);

} // namespace sabercat
