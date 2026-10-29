// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Scene-file discovery for the apps' scene lists.
#pragma once

#include <string>
#include <vector>

namespace sabercat::app
{

// Scans the assets directory (recursively) for scene files the loader accepts
// (.ply .splat .spz .ksplat .sog .gltf .glb, and SOG meta.json). Sorted.
std::vector<std::string> scanAssetsDirectory();

// The scene files under one directory, as scanAssetsDirectory() lists them.
// Sorted; empty when the directory is missing or unreadable.
std::vector<std::string> scanSceneDirectory(const std::string & directory);

} // namespace sabercat::app
