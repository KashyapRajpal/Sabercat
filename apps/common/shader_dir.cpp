// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "shader_dir.h"

#include "app_policy.h"

#include <cstdlib>
#include <filesystem>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h> // GetModuleFileNameW
#endif

namespace sabercat::app
{

namespace fs = std::filesystem;

std::string resolveShaderDir()
{
    std::error_code ec;
    std::string env;
    if (const char * value = std::getenv("SABERCAT_SHADER_DIR"))
        env = value;
    std::string installed;
    bool installedExists = false;
#ifdef _WIN32
    std::wstring exePath(MAX_PATH, L'\0');
    const DWORD length = GetModuleFileNameW(nullptr, exePath.data(), static_cast<DWORD>(exePath.size()));
    if (length > 0 && length < exePath.size())
    {
        exePath.resize(length);
        const fs::path candidate = fs::path(exePath).parent_path() / ".." / ".." / "share" / "sabercat" / "shaders";
        if (fs::is_directory(candidate, ec))
        {
            installed = fs::weakly_canonical(candidate, ec).string();
            installedExists = true;
        }
    }
#endif
    return chooseShaderDir(env, installed, installedExists, SABERCAT_SHADER_DIR);
}

} // namespace sabercat::app
