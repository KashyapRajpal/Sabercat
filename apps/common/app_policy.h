// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Decisions the apps make that need no Vulkan, window or file system: whether
// the validation layer is on, and which directory holds the SPIR-V. The apps
// call these from their GPU-facing wrappers (resolveValidation,
// resolveShaderDir), and the CPU unit tests call them directly.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace sabercat::app
{

// What the apps do about the Khronos validation layer.
enum class ValidationChoice
{
    On,
    OffByFlag,       // --no-validation
    OffNotRequested, // not requested: a Release build without --validation
    OffNotInstalled, // requested, but VK_LAYER_KHRONOS_validation is not installed
};

// The validation decision. `requested` is the build default (on in Debug) or
// --validation; `noValidationFlag` overrides everything; the layer must be
// installed for it to turn on.
ValidationChoice decideValidation(bool requested, bool noValidationFlag, bool installed);

// The SPIR-V directory, in the order resolveShaderDir() tries: a non-empty
// $SABERCAT_SHADER_DIR, then the install layout when it exists, then the build
// tree's path. `installedDir` is the canonical install path; it is only used
// when `installedExists`.
std::string chooseShaderDir(const std::string & envValue, const std::string & installedDir,
                            bool installedExists, const std::string & buildTreeDir);

// The key two scene paths are compared by: absolute and lexically normal, with
// separators unified and, on Windows, case folded. Paths that name the same file
// have the same key.
std::string scenePathKey(const std::string & path);

// Puts `path` at the top of the scene list (most recent first). An entry with the
// same key is moved up rather than repeated, so the list never holds a duplicate.
// Returns the index of the entry, which is always 0.
size_t addScenePathToList(std::vector<std::string> & list, const std::string & path);

} // namespace sabercat::app
