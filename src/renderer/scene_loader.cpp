// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/scene_loader.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "sabercat/cloud_v1_reader.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/gltf_reader.h"
#include "sabercat/ksplat_reader.h"
#include "sabercat/ply_reader.h"
#include "sabercat/sog_reader.h"
#include "sabercat/splat_reader.h"
#include "sabercat/spz_reader.h"
#include "sabercat/vk_check.h"

namespace sabercat
{

namespace
{

std::string getExtension(const std::string & path)
{
    size_t dotPos = path.find_last_of('.');
    if (dotPos == std::string::npos)
    {
        return "";
    }
    std::string ext = path.substr(dotPos);
    // Convert to lowercase for case-insensitive comparison
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return ext;
}

// Case-insensitive "path ends with suffix" -- for formats named by file name
// rather than extension.
bool hasSuffix(const std::string & path, const std::string & suffix)
{
    if (path.size() < suffix.size())
    {
        return false;
    }
    return std::equal(suffix.rbegin(), suffix.rend(), path.rbegin(),
                      [](char a, char b)
                      {
                          return std::tolower(static_cast<unsigned char>(a))
                                 == std::tolower(static_cast<unsigned char>(b));
                      });
}

// Check if file starts with PLY magic header
bool isPlyFile(const std::string & path)
{
    std::ifstream file(std::filesystem::u8path(path), std::ios::binary);
    if (!file.is_open())
    {
        return false;
    }

    char magic[3];
    if (!file.read(magic, 3))
    {
        return false;
    }

    return magic[0] == 'p' && magic[1] == 'l' && magic[2] == 'y';
}

// Check if file starts with the binary glTF magic "glTF"
bool isGlbFile(const std::string & path)
{
    std::ifstream file(std::filesystem::u8path(path), std::ios::binary);
    char magic[4];
    return file.is_open() && file.read(magic, 4) && std::memcmp(magic, "glTF", 4) == 0;
}

// Check if this is a COLMAP/SfM input.ply (diagnostic hint)
// These have x/y/z + red/green/blue but lack opacity, scale_*, rot_*, f_dc_*, f_rest_*
bool looksLikeColmapInput(const std::string & path)
{
    // Simple heuristic: if it's named "input.ply" it's likely COLMAP SfM output
    size_t lastSlash = path.find_last_of("/\\");
    std::string filename = (lastSlash != std::string::npos) ? path.substr(lastSlash + 1) : path;
    return filename == "input.ply";
}

} // namespace

SceneFormat detectFormat(const std::string & path)
{
    // A cloud-v1 fixture is a directory (streams + manifest.json); checked
    // first because a directory has no extension.
    if (std::filesystem::is_directory(std::filesystem::u8path(path)))
    {
        return SceneFormat::CLOUD_V1;
    }

    std::string ext = getExtension(path);

    // Extension-based detection with magic-byte veto for PLY.
    // .splat and .spz are headerless and can only be rejected
    // (they cannot be identified by magic bytes).
    if (ext == ".ply")
    {
        if (isPlyFile(path))
        {
            return SceneFormat::PLY;
        }
        // A file that cannot be opened is not a bad header: say so.
        VK_VALIDATE(std::filesystem::exists(std::filesystem::u8path(path)),
                    "detectFormat: cannot open scene file: " + path);
        // File has .ply extension but doesn't start with "ply" magic
        VK_VALIDATE(false, "detectFormat: file has .ply extension but missing PLY magic header: " + path);
    }

    if (ext == ".splat")
    {
        return SceneFormat::SPLAT;
    }

    if (ext == ".spz")
    {
        return SceneFormat::SPZ;
    }

    if (ext == ".gltf" || ext == ".glb")
    {
        // A missing file falls through so the reader reports "failed to open".
        VK_VALIDATE(ext == ".gltf" || !std::filesystem::exists(std::filesystem::u8path(path))
                        || isGlbFile(path),
                    "detectFormat: file has .glb extension but missing the binary glTF magic: " + path);
        return SceneFormat::GLTF;
    }

    if (ext == ".ksplat")
    {
        return SceneFormat::KSPLAT;
    }

    if (ext == ".sog")
    {
        return SceneFormat::SOG;
    }

    // Unbundled SOG is identified by NAME, as splat-transform does it: the
    // meta.json beside its images (also "<scene>.meta.json"). A streamed
    // lod-meta.json is a different, chunked container -- refused by name.
    if (hasSuffix(path, "meta.json"))
    {
        VK_VALIDATE(!hasSuffix(path, "lod-meta.json"),
                    "detectFormat: streamed SOG (lod-meta.json) is not supported; load a single .sog "
                    "or meta.json: " + path);
        return SceneFormat::SOG;
    }

    if (ext == ".json")
    {
        return SceneFormat::CLOUD_V1; // the reader checks the manifest's schema
    }

    VK_VALIDATE(false, "detectFormat: unsupported file extension (expected .ply, .splat, .spz, "
                       ".ksplat, .sog, a SOG meta.json, .gltf, .glb, or a sabercat-cloud-v1 "
                       "manifest.json/directory): " + path);
    return SceneFormat::PLY; // unreachable
}

std::shared_ptr<GaussianCloud> loadScene(Device & device, const std::string & path)
{
    // Diagnostic: COLMAP/SfM input.ply files lack Gaussian parameters
    if (looksLikeColmapInput(path))
    {
        VK_VALIDATE(false, "loadScene: appears to be COLMAP/SfM output (input.ply) rather than "
                           "a 3DGS scene. This file has camera/structure data but lacks "
                           "Gaussian parameters (opacity, scale_*, rot_*, f_dc_*, f_rest_*). "
                           "Please use the reference 3DGS trainer to generate a scene: " +
                               path);
    }

    SceneFormat format = detectFormat(path);

    switch (format)
    {
    case SceneFormat::PLY:
        return PLYReader::read(device, path);

    case SceneFormat::SPLAT:
        return SplatReader::read(device, path);

    case SceneFormat::SPZ:
        return SpzReader::read(device, path);

    case SceneFormat::GLTF:
        return GltfReader::read(device, path);

    case SceneFormat::CLOUD_V1:
        return CloudV1Reader::read(device, path);
    case SceneFormat::KSPLAT:
        return KsplatReader::read(device, path);

    case SceneFormat::SOG:
        return SogReader::read(device, path);
    }

    return nullptr; // unreachable
}

} // namespace sabercat
