// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// UTF-8 command-line arguments (apps/compare/app_args.h).

#include "app_args.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <shellapi.h>
#endif

namespace sabercat::compare
{

// Command-line arguments as UTF-8. On Windows argv is narrow (the ANSI codepage),
// so a non-ASCII path would arrive mangled; the wide command line is the source.
std::vector<std::string> utf8Arguments(int argc, char * argv[])
{
#ifdef _WIN32
    int wideCount = 0;
    LPWSTR * wide = CommandLineToArgvW(GetCommandLineW(), &wideCount);
    if (wide != nullptr && wideCount == argc)
    {
        std::vector<std::string> result;
        result.reserve(static_cast<size_t>(wideCount));
        for (int i = 0; i < wideCount; ++i)
        {
            const int bytes = WideCharToMultiByte(CP_UTF8, 0, wide[i], -1, nullptr, 0, nullptr, nullptr);
            std::string text(bytes > 1 ? static_cast<size_t>(bytes - 1) : 0, '\0');
            if (bytes > 1)
            {
                WideCharToMultiByte(CP_UTF8, 0, wide[i], -1, text.data(), bytes, nullptr, nullptr);
            }
            result.push_back(text);
        }
        LocalFree(wide);
        return result;
    }
    if (wide != nullptr)
    {
        LocalFree(wide);
    }
#endif
    return std::vector<std::string>(argv, argv + argc);
}

} // namespace sabercat::compare
