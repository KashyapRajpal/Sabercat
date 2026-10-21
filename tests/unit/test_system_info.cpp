// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The system-info text (apps/common/system_info.h), the session log naming and
// pruning (apps/common/log_files.h) and the Logger's file mirror
// (sabercat/logger.h), on pure inputs and a temp file. No device is created.
#include "unit_test.h"

#include "log_files.h"
#include "system_info.h"

#include "sabercat/logger.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace sabercat::unit
{

namespace
{

namespace fs = std::filesystem;
using app::SystemInfoFacts;

bool contains(const std::string & text, const std::string & needle)
{
    return text.find(needle) != std::string::npos;
}

// The driver string as the apps show it, for each vendor's packing.
void testDriverVersionDecoding(Context & ctx)
{
    // NVIDIA: major in bits 22-31, minor 14-21, secondary 6-13, tertiary 0-5.
    // 552 << 22 = 0x8A000000 and 22 << 14 = 0x00058000, so 0x8A058000 is 552.22.
    UNIT_CHECK(ctx, app::decodeDriverVersion(0x10DE, 0x8A058000u) == "552.22");
    // 551 << 22 = 0x89C00000, 61 << 14 = 0x000F4000, 3 << 6 = 0xC0, 5 = 0x5:
    // 0x89CF40C5. Secondary and tertiary show only when set.
    UNIT_CHECK(ctx, app::decodeDriverVersion(0x10DE, 0x89CF40C5u) == "551.61.3.5");
    // Zero minor and sub-fields: just the major.
    UNIT_CHECK(ctx, app::decodeDriverVersion(0x10DE, 0x8A000000u) == "552.0");

    // Intel on Windows: major = value >> 14, minor = value & 0x3FFF.
    UNIT_CHECK(ctx, app::decodeDriverVersion(0x8086, (27u << 14) | 1234u) == "27.1234");

    // Other vendors: the VK_MAKE_API_VERSION layout (major 22-31, minor 12-21, patch 0-11).
    UNIT_CHECK(ctx, app::decodeDriverVersion(0x1002, (1u << 22) | (2u << 12) | 3u) == "1.2.3");

    // A Vulkan apiVersion: VK_API_VERSION_1_3 is (1 << 22) | (3 << 12).
    UNIT_CHECK(ctx, app::formatVulkanApiVersion((1u << 22) | (3u << 12)) == "1.3.0");
}

SystemInfoFacts sampleFacts()
{
    SystemInfoFacts facts;
    facts.appName = "Sabercat Viewer";
    facts.appVersion = "1.0.0-beta.1";
    facts.gitSha = "1a2b3c4";
    facts.buildConfig = "Release";
    facts.osVersion = "Windows 11 (10.0 build 26200)";
    facts.hasDevice = true;
    facts.gpuName = "NVIDIA GeForce RTX 4070";
    facts.vendorId = 0x10DE;
    facts.deviceId = 0x2786;
    facts.driverVersion = 0x8A058000u;
    facts.apiVersion = (1u << 22) | (3u << 12) | 280u;
    facts.rayQuery = true;
    facts.floatAtomics = false;
    facts.validation = false;
    return facts;
}

void testSystemInfoText(Context & ctx)
{
    const std::string text = app::formatSystemInfo(sampleFacts(), "C:\\logs\\viewer-1.log");
    UNIT_CHECK(ctx, text.rfind("Sabercat Viewer 1.0.0-beta.1 (git 1a2b3c4)\n", 0) == 0);
    UNIT_CHECK(ctx, contains(text, "Build: Release\n"));
    UNIT_CHECK(ctx, contains(text, "OS: Windows 11 (10.0 build 26200)\n"));
    UNIT_CHECK(ctx, contains(text, "GPU: NVIDIA GeForce RTX 4070\n"));
    UNIT_CHECK(ctx, contains(text, "Vendor / device: 0x10DE / 0x2786\n"));
    UNIT_CHECK(ctx, contains(text, "Driver: 552.22 (raw 0x8A058000)\n"));
    UNIT_CHECK(ctx, contains(text, "Vulkan API: 1.3.280\n"));
    UNIT_CHECK(ctx, contains(text, "Ray query: yes\n"));
    UNIT_CHECK(ctx, contains(text, "Float atomics: no\n"));
    UNIT_CHECK(ctx, contains(text, "Validation: off\n"));
    UNIT_CHECK(ctx, contains(text, "Log file: C:\\logs\\viewer-1.log\n"));

    // No device and logging off: the GPU lines are replaced, the log says off.
    SystemInfoFacts noDevice = sampleFacts();
    noDevice.hasDevice = false;
    noDevice.gitSha = "unknown";
    const std::string bare = app::formatSystemInfo(noDevice, "");
    UNIT_CHECK(ctx, bare.rfind("Sabercat Viewer 1.0.0-beta.1 (git unknown)\n", 0) == 0);
    UNIT_CHECK(ctx, contains(bare, "GPU: unavailable (no Vulkan device)\n"));
    UNIT_CHECK(ctx, !contains(bare, "Driver:"));
    UNIT_CHECK(ctx, contains(bare, "Log file: off\n"));
}

void testSessionLogNames(Context & ctx)
{
    UNIT_CHECK(ctx, app::sessionLogFileName("viewer", "20260102-093015") == "viewer-20260102-093015.log");

    // Twelve viewer logs and some other files: keep the newest 10 viewer logs.
    std::vector<std::string> names;
    for (int day = 1; day <= 12; ++day)
    {
        std::ostringstream name;
        name << "viewer-202601" << (day < 10 ? "0" : "") << day << "-120000.log";
        names.push_back(name.str());
    }
    names.push_back("studio-20260101-120000.log"); // another app: never touched
    names.push_back("viewer-notes.txt");           // not a session log
    names.push_back("viewer-20260112-120000.log"); // a duplicate of the newest: counts once

    const std::vector<std::string> prune = app::logNamesToPrune(names, "viewer", 10);
    UNIT_CHECK(ctx, prune.size() == 2);
    UNIT_CHECK(ctx, prune.size() == 2 && prune[0] == "viewer-20260101-120000.log");
    UNIT_CHECK(ctx, prune.size() == 2 && prune[1] == "viewer-20260102-120000.log");

    // Nothing to prune when at or under the limit, and the other app's logs are ignored.
    UNIT_CHECK(ctx, app::logNamesToPrune(names, "studio", 10).empty());
    UNIT_CHECK(ctx, app::logNamesToPrune(names, "viewer", 12).empty());
}

// The Logger mirrors each line to the file while it is set, and turns it off on "".
void testLoggerFileMirror(Context & ctx)
{
    const fs::path file = fs::temp_directory_path() / "sabercat_unit_log_mirror.log";
    const std::string utf8 = reinterpret_cast<const char *>(file.u8string().c_str());
    std::error_code ec;
    fs::remove(file, ec);

    UNIT_CHECK(ctx, Logger::setLogFile(utf8));
    UNIT_CHECK(ctx, Logger::logFilePath() == utf8);
    Logger::info("unit test mirror line");
    UNIT_CHECK(ctx, Logger::setLogFile(""));
    UNIT_CHECK(ctx, Logger::logFilePath().empty());
    Logger::info("unit test after the mirror is off");

    std::ifstream in(file);
    std::stringstream content;
    content << in.rdbuf();
    UNIT_CHECK(ctx, contains(content.str(), "] [INFO] unit test mirror line"));
    UNIT_CHECK(ctx, !contains(content.str(), "unit test after the mirror is off"));
    fs::remove(file, ec);
}

} // namespace

void testSystemInfo(Context & ctx)
{
    testDriverVersionDecoding(ctx);
    testSystemInfoText(ctx);
    testSessionLogNames(ctx);
    testLoggerFileMirror(ctx);
}

} // namespace sabercat::unit
