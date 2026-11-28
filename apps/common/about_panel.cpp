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

// One line of the notices as drawn, with the markdown already reduced to text.
struct NoticeLine
{
    std::string text;
    bool heading = false;
};

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

// Trims spaces and tabs from both ends.
std::string trimSpaces(const std::string & text)
{
    const std::size_t first = text.find_first_not_of(" \t");
    if (first == std::string::npos)
    {
        return {};
    }
    return text.substr(first, text.find_last_not_of(" \t") - first + 1);
}

// Drops the inline markdown markers: backticks and '*' go, a link [text](url)
// becomes its text, and a web url is kept in brackets after it.
std::string stripInline(const std::string & text)
{
    std::string out;
    out.reserve(text.size());
    std::size_t i = 0;
    while (i < text.size())
    {
        if (text[i] == '`' || text[i] == '*')
        {
            ++i;
            continue;
        }
        if (text[i] == '[')
        {
            const std::size_t close = text.find(']', i + 1);
            if (close != std::string::npos && close + 1 < text.size() && text[close + 1] == '(')
            {
                const std::size_t end = text.find(')', close + 2);
                if (end != std::string::npos)
                {
                    out += stripInline(text.substr(i + 1, close - i - 1));
                    const std::string url = text.substr(close + 2, end - close - 2);
                    if (url.starts_with("http"))
                    {
                        out += " (" + url + ")";
                    }
                    i = end + 1;
                    continue;
                }
            }
        }
        out += text[i++];
    }
    return out;
}

// A trimmed table row (it starts with a bar) as its cells, each trimmed and
// joined by two spaces.
std::string tableRowText(const std::string & row)
{
    std::string body = row.substr(1);
    if (body.ends_with('|'))
    {
        body.pop_back();
    }
    std::string out;
    std::size_t start = 0;
    while (start <= body.size())
    {
        std::size_t end = body.find('|', start);
        if (end == std::string::npos)
        {
            end = body.size();
        }
        if (start > 0)
        {
            out += "  ";
        }
        out += trimSpaces(stripInline(body.substr(start, end - start)));
        start = end + 1;
    }
    return out;
}

// Turns the markdown notices into plain display lines. Fenced code is kept as
// raw text, table rows become their cells, and headings are flagged for bold.
std::vector<NoticeLine> parseNotices(const std::string & content)
{
    std::vector<NoticeLine> lines;
    bool inFence = false;
    std::size_t start = 0;
    while (start < content.size())
    {
        std::size_t end = content.find('\n', start);
        if (end == std::string::npos)
        {
            end = content.size();
        }
        std::string line = content.substr(start, end - start);
        start = end + 1;
        if (line.ends_with('\r'))
        {
            line.pop_back();
        }

        const std::string trimmedLine = trimSpaces(line);
        if (trimmedLine.starts_with("```"))
        {
            inFence = !inFence;
            continue;
        }
        if (inFence)
        {
            lines.push_back({line, false});
            continue;
        }
        if (trimmedLine.size() >= 3 && trimmedLine.find_first_not_of('-') == std::string::npos)
        {
            continue; // horizontal rule
        }
        if (trimmedLine.starts_with('|'))
        {
            // Separator rows hold only bars, dashes, colons and spaces: no text.
            if (trimmedLine.find_first_not_of("|-: \t") != std::string::npos)
            {
                lines.push_back({tableRowText(trimmedLine), false});
            }
            continue;
        }
        const std::size_t hashes = trimmedLine.find_first_not_of('#');
        if (hashes > 0 && hashes != std::string::npos && trimmedLine[hashes] == ' ')
        {
            lines.push_back({trimSpaces(stripInline(trimmedLine.substr(hashes + 1))), true});
            continue;
        }
        // A list item loses its marker and keeps its indent, plus two spaces.
        std::size_t indent = line.find_first_not_of(' ');
        if (indent == std::string::npos)
        {
            indent = line.size();
        }
        if (line.compare(indent, 2, "- ") == 0 || line.compare(indent, 2, "* ") == 0)
        {
            lines.push_back({std::string(indent, ' ') + "  " + stripInline(line.substr(indent + 2)), false});
            continue;
        }
        lines.push_back({stripInline(line), false});
    }
    return lines;
}

// The notices as display lines. Read once: the file does not change while the app runs.
const std::vector<NoticeLine> & noticeLines()
{
    static const std::vector<NoticeLine> lines = []
    {
        std::string content = kNoticesMissing;
        const fs::path path = findNotices();
        if (!path.empty())
        {
            std::ifstream file(path, std::ios::binary);
            content.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        }
        return parseNotices(content);
    }();
    return lines;
}

// Draws one notices line. ImGui has no bold font, so a heading is drawn a
// second time, 1 px to the right.
void drawNoticeLine(const NoticeLine & line)
{
    if (!line.heading)
    {
        ImGui::TextUnformatted(line.text.c_str());
        return;
    }
    const ImVec2 start = ImGui::GetCursorScreenPos();
    ImGui::TextUnformatted(line.text.c_str());
    ImGui::GetWindowDrawList()->AddText(ImVec2(start.x + 1.0f, start.y), ImGui::GetColorU32(ImGuiCol_Text),
                                        line.text.c_str());
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

    // A bordered child takes the remaining window height and scrolls. Each
    // line is one item of equal height, so the clipper stays exact; long lines
    // scroll sideways instead of wrapping.
    const std::vector<NoticeLine> & lines = noticeLines();
    ImGui::BeginChild("##notices", ImVec2(-1.0f, -1.0f), ImGuiChildFlags_Borders,
                      ImGuiWindowFlags_HorizontalScrollbar);
    ImGuiListClipper clipper;
    clipper.Begin(static_cast<int>(lines.size()));
    while (clipper.Step())
    {
        for (int i = clipper.DisplayStart; i < clipper.DisplayEnd; ++i)
        {
            drawNoticeLine(lines[static_cast<std::size_t>(i)]);
        }
    }
    ImGui::EndChild();
    ImGui::End();
}

} // namespace sabercat::app
