// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>
#include <string>

// The compiled SPIR-V directory for a test: $SABERCAT_SHADER_DIR when set
// (the packaged self-test runs these exes away from the build tree and points
// it at <prefix>/share/sabercat/shaders), else the build tree's SHADER_DIR.
inline std::string testShaderDir()
{
    const char * env = std::getenv("SABERCAT_SHADER_DIR");
    return env != nullptr && *env != '\0' ? std::string(env) : std::string(SHADER_DIR);
}
