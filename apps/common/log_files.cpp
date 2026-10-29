// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "log_files.h"

#include "sabercat/logger.h"

#include <cstddef>
#include <cstdlib>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <set>
#include <sstream>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace sabercat::app
{

namespace fs = std::filesystem;

namespace
{

constexpr const char * kLogSuffix = ".log";

#ifdef _WIN32
constexpr char kSeparator = '\\';

// The value of an environment variable as UTF-8 ("" when unset).
std::string environmentUtf8(const wchar_t * name)
{
    const DWORD length = GetEnvironmentVariableW(name, nullptr, 0);
    if (length == 0)
    {
        return {};
    }
    std::wstring wide(length, L'\0');
    const DWORD written = GetEnvironmentVariableW(name, wide.data(), length);
    wide.resize(written);
    const int bytes = WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()),
                                          nullptr, 0, nullptr, nullptr);
    std::string utf8(bytes > 0 ? static_cast<size_t>(bytes) : 0, '\0');
    if (bytes > 0)
    {
        WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()), utf8.data(),
                            bytes, nullptr, nullptr);
    }
    return utf8;
}
#else
constexpr char kSeparator = '/';

std::string environmentUtf8(const char * name)
{
    const char * value = std::getenv(name);
    return value != nullptr ? value : "";
}
#endif

std::string currentStamp()
{
    const std::time_t now = std::time(nullptr);
    const std::tm local = *std::localtime(&now);
    std::ostringstream stamp;
    stamp << std::put_time(&local, "%Y%m%d-%H%M%S");
    return stamp.str();
}

// Deletes the old session logs of `app` in `directory`. Failures are warnings.
void pruneOldLogs(const fs::path & directory, const std::string & app, const std::string & newName)
{
    std::vector<std::string> names;
    std::error_code ec;
    for (fs::directory_iterator it(directory, ec), end; !ec && it != end; it.increment(ec))
    {
        names.push_back(it->path().filename().string());
    }
    names.push_back(newName);

    for (const std::string & name : logNamesToPrune(names, app, kKeptLogFiles))
    {
        std::error_code removeError;
        if (!fs::remove(directory / name, removeError) || removeError)
        {
            std::cout << "[WARN] could not delete old log " << name << std::endl;
        }
    }
}

} // namespace

std::string sessionLogFileName(const std::string & app, const std::string & stamp)
{
    return app + "-" + stamp + kLogSuffix;
}

std::vector<std::string> logNamesToPrune(const std::vector<std::string> & names,
                                         const std::string & app, std::size_t keep)
{
    const std::string prefix = app + "-";
    const std::string suffix = kLogSuffix;
    std::set<std::string> ours; // sorted: the stamp makes name order time order
    for (const std::string & name : names)
    {
        const bool ownPrefix = name.size() > prefix.size() + suffix.size()
                               && name.compare(0, prefix.size(), prefix) == 0;
        const bool ownSuffix = name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
        if (ownPrefix && ownSuffix)
        {
            ours.insert(name);
        }
    }

    std::vector<std::string> prune;
    if (ours.size() > keep)
    {
        const std::size_t excess = ours.size() - keep;
        prune.assign(ours.begin(), std::next(ours.begin(), static_cast<std::ptrdiff_t>(excess)));
    }
    return prune;
}

std::string defaultLogDirectory()
{
#ifdef _WIN32
    const std::string localAppData = environmentUtf8(L"LOCALAPPDATA");
#else
    const std::string localAppData = environmentUtf8("LOCALAPPDATA");
#endif
    if (localAppData.empty())
    {
        return {};
    }
    return localAppData + kSeparator + "Sabercat" + kSeparator + "logs";
}

std::string startSessionLog(const std::string & app, const std::string & explicitPath, bool disabled)
{
    if (disabled)
    {
        Logger::setLogFile("");
        return {};
    }

    std::string path = explicitPath;
    if (path.empty())
    {
        const std::string directory = defaultLogDirectory();
        if (directory.empty())
        {
            std::cout << "[WARN] LOCALAPPDATA is not set; running without a log file" << std::endl;
            return {};
        }
        std::error_code ec;
        fs::create_directories(fs::u8path(directory), ec);
        if (ec)
        {
            std::cout << "[WARN] could not create " << directory << "; running without a log file"
                      << std::endl;
            return {};
        }
        const std::string name = sessionLogFileName(app, currentStamp());
        pruneOldLogs(fs::u8path(directory), app, name);
        path = directory + kSeparator + name;
    }

    if (!Logger::setLogFile(path))
    {
        std::cout << "[WARN] could not open log file " << path << "; running without a log file"
                  << std::endl;
        return {};
    }
    return path;
}

} // namespace sabercat::app
