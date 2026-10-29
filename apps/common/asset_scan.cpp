// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "asset_scan.h"

#include "sabercat/scene_loader.h"

#include <algorithm>
#include <cctype>
#include <exception>
#include <filesystem>
#include <iostream>

namespace sabercat::app
{

namespace fs = std::filesystem;

std::vector<std::string> scanSceneDirectory(const std::string & directory)
{
    std::vector<std::string> scenes;
    try
    {
        for (const auto & entry : fs::recursive_directory_iterator(directory))
        {
            if (!entry.is_regular_file())
            {
                continue;
            }

            std::string ext = entry.path().extension().string();
            // Convert to lowercase for comparison
            std::transform(ext.begin(), ext.end(), ext.begin(),
                          [](unsigned char c) { return std::tolower(c); });

            if (ext == ".ply" || ext == ".splat" || ext == ".spz" || ext == ".ksplat" || ext == ".sog" ||
                ext == ".gltf" || ext == ".glb")
            {
                scenes.push_back(entry.path().string());
            }
            else if (ext == ".json")
            {
                // Unbundled SOG is a directory of webp images named by its
                // meta.json. Ask the loader's own detection rather than
                // re-deriving it here, so only what loadScene would read as SOG
                // is listed -- not cloud-v1 manifests or arbitrary .json, and
                // not a streamed lod-meta.json (detectFormat rejects that).
                try
                {
                    if (detectFormat(entry.path().string()) == SceneFormat::SOG)
                    {
                        scenes.push_back(entry.path().string());
                    }
                }
                catch (const std::exception &)
                {
                }
            }
        }
    }
    catch (const std::exception & e)
    {
        // Log but don't crash if directory scanning fails
        std::cerr << "[Viewer] Warning: failed to scan assets directory: " << e.what()
                 << std::endl;
    }

    // Sort for consistent order
    std::sort(scenes.begin(), scenes.end());
    return scenes;
}

// Scan assets directory (recursively) for scene files (.ply, .splat, .spz, .ksplat, .sog)
std::vector<std::string> scanAssetsDirectory()
{
    // Debug: print current working directory
    std::cout << "  [AssetScanner] CWD: " << fs::current_path().string() << std::endl;

    // Try to find assets directory:
    // 1. Check if assets/ exists relative to current working directory
    // 2. Walk up directory tree looking for project root (CMakeLists.txt present)
    // 3. Check standard relative paths from build directories
    std::vector<std::string> searchPaths;
    searchPaths.push_back("assets");

    // Walk up the directory tree looking for a CMakeLists.txt (project root marker)
    fs::path current = fs::current_path();
    for (int i = 0; i < 6; ++i)  // Search up to 6 levels
    {
        if (fs::exists(current / "CMakeLists.txt"))
        {
            searchPaths.push_back((current / "assets").string());
            std::cout << "  [AssetScanner] Found project root at: " << current.string() << std::endl;
            break;
        }
        current = current.parent_path();
        if (current == current.parent_path())
            break;  // reached filesystem root
    }

    // Also add standard build paths
    searchPaths.push_back("../../../assets");  // from build/apps/viewer/Debug
    searchPaths.push_back("../../assets");     // from build/apps/viewer
    searchPaths.push_back("../assets");        // from build/apps

    std::string assetsDir;
    std::cout << "  [AssetScanner] Searching paths:" << std::endl;
    for (const auto & candidate : searchPaths)
    {
        bool exists = fs::exists(candidate) && fs::is_directory(candidate);
        std::cout << "    - " << fs::absolute(candidate).string() << (exists ? " [EXISTS]" : "")
                 << std::endl;
        if (exists && assetsDir.empty())
        {
            assetsDir = candidate;
        }
    }

    if (assetsDir.empty())
    {
        std::cout << "  [AssetScanner] Assets directory NOT FOUND" << std::endl;
        return {};
    }

    std::cout << "[Viewer] Scanning assets directory: " << fs::absolute(assetsDir).string()
             << std::endl;

    std::vector<std::string> scenes = scanSceneDirectory(assetsDir);
    std::cout << "[Viewer] Found " << scenes.size() << " scene file(s)" << std::endl;
    for (const auto & scene : scenes)
    {
        std::cout << "  - " << scene << std::endl;
    }
    return scenes;
}

} // namespace sabercat::app
