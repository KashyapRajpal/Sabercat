// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Run-time location of the compiled SPIR-V shaders, shared by the apps.
#pragma once

#include <string>

namespace sabercat::app
{

// Resolves the SPIR-V directory at run time, so a packaged app works away from
// the build tree: $SABERCAT_SHADER_DIR, then the install layout
// (<prefix>/bin/<Config>/ -> <prefix>/share/sabercat/shaders), then the build
// tree's path baked in at compile time (SABERCAT_SHADER_DIR macro of this target).
std::string resolveShaderDir();

} // namespace sabercat::app
