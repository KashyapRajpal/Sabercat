// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "about_panel.h"

#include "log_files.h"
#include "system_info.h"

#include "sabercat/logger.h"
#include "sabercat/version.h"

#include <imgui.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h> // GetModuleFileNameW, MultiByteToWideChar
#include <shellapi.h> // ShellExecuteW
#endif

namespace sabercat::app
{

namespace fs = std::filesystem;

namespace
{

constexpr const char * kNoticesFile = "THIRD_PARTY_NOTICES.md";
constexpr const char * kNoticesMissing = "THIRD_PARTY_NOTICES.md not found";

// The notices file, found in the install prefix (the exe lives in
// <prefix>/bin/<Config>/) or else in the source tree the app was built from.
// Returns an empty path when neither has it.
fs::path findNotices()
{
    std::error_code ec;
#ifdef _WIN32
    std::wstring exePath(MAX_PATH, L'\0');
    const DWORD length = GetModuleFileNameW(nullptr, exePath.data(), static_cast<DWORD>(exePath.size()));
    if (length > 0 && length < exePath.size())
    {
        exePath.resize(length);
        const fs::path installed = fs::path(exePath).parent_path() / ".." / ".." / kNoticesFile;
        if (fs::is_regular_file(installed, ec))
            return fs::weakly_canonical(installed, ec);
    }
#endif
    const fs::path source = fs::path(SABERCAT_SOURCE_DIR) / kNoticesFile;
    if (fs::is_regular_file(source, ec))
        return source;
    return {};
}

// Opens the folder holding the session logs in Explorer: the folder of the
// current log file, or the default log folder when logging is off.
void openLogsFolder(const std::string & logFile)
{
    std::string folder = logFile.empty() ? defaultLogDirectory() : logFile;
    if (!logFile.empty())
    {
        const std::size_t separator = folder.find_last_of("\\/");
        folder = separator == std::string::npos ? std::string() : folder.substr(0, separator);
    }
    std::error_code ec;
    if (folder.empty() || !fs::is_directory(fs::u8path(folder), ec))
    {
        Logger::warning("No logs folder to open yet: " + folder);
        return;
    }
#ifdef _WIN32
    const int length = MultiByteToWideChar(CP_UTF8, 0, folder.c_str(), -1, nullptr, 0);
    std::wstring wideFolder(length > 0 ? static_cast<std::size_t>(length) : 0, L'\0');
    if (length > 0)
    {
        MultiByteToWideChar(CP_UTF8, 0, folder.c_str(), -1, wideFolder.data(), length);
    }
    ShellExecuteW(nullptr, L"open", wideFolder.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#endif
}

// The notices as a NUL-terminated buffer for a read-only multiline text box.
// Read once: the file does not change while the app runs.
std::vector<char> & noticesText()
{
    static std::vector<char> text = []
    {
        std::string content = kNoticesMissing;
        const fs::path path = findNotices();
        if (!path.empty())
        {
            std::ifstream file(path, std::ios::binary);
            content.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        }
        std::vector<char> buffer(content.begin(), content.end());
        buffer.push_back('\0');
        return buffer;
    }();
    return text;
}

} // namespace

void drawAboutWindow(bool & open, const SystemInfoFacts & facts, ImTextureID logo)
{
    if (!open)
    {
        return;
    }
    ImGui::SetNextWindowSize(ImVec2(560.0f, 540.0f), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("About Sabercat", &open))
    {
        ImGui::End();
        return;
    }
    if (logo != 0)
    {
        ImGui::Image(logo, ImVec2(96.0f, 96.0f));
    }
    ImGui::Text("%s %s", facts.appName.c_str(), sabercat::kVersionString);
    ImGui::Text("Apache License 2.0");
    ImGui::Text("Copyright 2026 Kashyap Rajpal");
    ImGui::Separator();

    // The log path is read each frame: the mirror file is set once at startup,
    // but it is "off" with --no-log.
    const std::string logFile = Logger::logFilePath();
    ImGui::TextWrapped("Log file: %s", logFile.empty() ? "off" : logFile.c_str());
    if (ImGui::Button("Open logs folder"))
    {
        openLogsFolder(logFile);
    }
    ImGui::SameLine();
    if (ImGui::Button("Copy system info"))
    {
        ImGui::SetClipboardText(formatSystemInfo(facts, logFile).c_str());
    }
    ImGui::Separator();
    ImGui::Text("Third-party notices:");

    // A read-only multiline box is the scrollable view: it takes the
    // remaining window height and scrolls internally.
    std::vector<char> & text = noticesText();
    ImGui::InputTextMultiline("##notices", text.data(), text.size(), ImVec2(-1.0f, -1.0f),
                              ImGuiInputTextFlags_ReadOnly);
    ImGui::End();
}

} // namespace sabercat::app
