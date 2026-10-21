// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "app_policy.h"

#include <filesystem>
#include <system_error>

namespace sabercat::app
{

ValidationChoice decideValidation(bool requested, bool noValidationFlag, bool installed)
{
    if (noValidationFlag)
    {
        return ValidationChoice::OffByFlag;
    }
    if (!requested)
    {
        return ValidationChoice::OffNotRequested;
    }
    if (!installed)
    {
        return ValidationChoice::OffNotInstalled;
    }
    return ValidationChoice::On;
}

std::string chooseShaderDir(const std::string & envValue, const std::string & installedDir,
                            bool installedExists, const std::string & buildTreeDir)
{
    if (!envValue.empty())
    {
        return envValue;
    }
    if (installedExists)
    {
        return installedDir;
    }
    return buildTreeDir;
}

std::string scenePathKey(const std::string & path)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path normal = fs::absolute(fs::u8path(path), ec);
    if (ec)
    {
        normal = fs::u8path(path);
    }
    normal = normal.lexically_normal();
    const std::u8string utf8 = normal.u8string();
    std::string key(utf8.begin(), utf8.end());
    // A trailing separator names the same entry as without it (not a root).
    if (key.size() > 1 && (key.back() == '/' || key.back() == '\\'))
    {
        key.pop_back();
    }
#ifdef _WIN32
    for (char & c : key)
    {
        if (c == '/')
        {
            c = '\\';
        }
        if (c >= 'A' && c <= 'Z')
        {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
#endif
    return key;
}

size_t addScenePathToList(std::vector<std::string> & list, const std::string & path)
{
    const std::string key = scenePathKey(path);
    for (size_t i = 0; i < list.size(); ++i)
    {
        if (scenePathKey(list[i]) == key)
        {
            list.erase(list.begin() + static_cast<std::ptrdiff_t>(i));
            break;
        }
    }
    list.insert(list.begin(), path);
    return 0;
}

} // namespace sabercat::app
