// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "system_info.h"

#include "sabercat/device.h"
#include "sabercat/version.h"

#include <cstdio>
#include <sstream>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace sabercat::app
{

namespace
{

constexpr uint32_t kVendorNvidia = 0x10DE;
constexpr uint32_t kVendorIntel = 0x8086;

#ifdef NDEBUG
constexpr const char * kBuildConfig = "Release";
#else
constexpr const char * kBuildConfig = "Debug";
#endif

std::string hex(uint32_t value, int digits)
{
    char buffer[16];
    std::snprintf(buffer, sizeof(buffer), "0x%0*X", digits, value);
    return buffer;
}

std::string yesNo(bool value)
{
    return value ? "yes" : "no";
}

// "Windows 11 (10.0 build 26200)" from RtlGetVersion. Windows 11 reports
// major 10 with build 22000 or later.
std::string osVersionText()
{
#ifdef _WIN32
    using RtlGetVersionFn = LONG(WINAPI *)(PRTL_OSVERSIONINFOW);
    const HMODULE ntdll = GetModuleHandleW(L"ntdll.dll");
    const auto rtlGetVersion =
        ntdll != nullptr ? reinterpret_cast<RtlGetVersionFn>(GetProcAddress(ntdll, "RtlGetVersion"))
                         : nullptr;
    if (rtlGetVersion != nullptr)
    {
        OSVERSIONINFOW info{};
        info.dwOSVersionInfoSize = sizeof(info);
        if (rtlGetVersion(&info) == 0)
        {
            const bool windows11 = info.dwMajorVersion == 10 && info.dwBuildNumber >= 22000;
            std::ostringstream text;
            text << (windows11 ? "Windows 11" : "Windows") << (windows11 ? " (" : " ")
                 << info.dwMajorVersion << "." << info.dwMinorVersion << " build "
                 << info.dwBuildNumber << (windows11 ? ")" : "");
            return text.str();
        }
    }
#endif
    return "unknown";
}

} // namespace

std::string decodeDriverVersion(uint32_t vendorId, uint32_t driverVersion)
{
    std::ostringstream text;
    if (vendorId == kVendorNvidia)
    {
        const uint32_t major = driverVersion >> 22;
        const uint32_t minor = (driverVersion >> 14) & 0xFFu;
        const uint32_t secondary = (driverVersion >> 6) & 0xFFu;
        const uint32_t tertiary = driverVersion & 0x3Fu;
        text << major << "." << minor;
        if (secondary != 0 || tertiary != 0)
        {
            text << "." << secondary << "." << tertiary;
        }
    }
    else if (vendorId == kVendorIntel)
    {
        text << (driverVersion >> 14) << "." << (driverVersion & 0x3FFFu);
    }
    else
    {
        text << (driverVersion >> 22) << "." << ((driverVersion >> 12) & 0x3FFu) << "."
             << (driverVersion & 0xFFFu);
    }
    return text.str();
}

std::string formatVulkanApiVersion(uint32_t apiVersion)
{
    std::ostringstream text;
    text << (apiVersion >> 22) << "." << ((apiVersion >> 12) & 0x3FFu) << "."
         << (apiVersion & 0xFFFu);
    return text.str();
}

std::string formatSystemInfo(const SystemInfoFacts & facts, const std::string & logFile)
{
    std::ostringstream out;
    out << facts.appName << " " << facts.appVersion << " (git " << facts.gitSha << ")\n";
    out << "Build: " << facts.buildConfig << "\n";
    out << "OS: " << facts.osVersion << "\n";
    if (facts.hasDevice)
    {
        out << "GPU: " << facts.gpuName << "\n";
        out << "Vendor / device: " << hex(facts.vendorId, 4) << " / " << hex(facts.deviceId, 4)
            << "\n";
        out << "Driver: " << decodeDriverVersion(facts.vendorId, facts.driverVersion) << " (raw "
            << hex(facts.driverVersion, 8) << ")\n";
        out << "Vulkan API: " << formatVulkanApiVersion(facts.apiVersion) << "\n";
        out << "Ray query: " << yesNo(facts.rayQuery) << "\n";
        out << "Float atomics: " << yesNo(facts.floatAtomics) << "\n";
    }
    else
    {
        out << "GPU: unavailable (no Vulkan device)\n";
    }
    out << "Validation: " << (facts.validation ? "on" : "off") << "\n";
    out << "Log file: " << (logFile.empty() ? "off" : logFile) << "\n";
    return out.str();
}

SystemInfoFacts gatherSystemInfo(const std::string & appName, const Device * device, bool validation)
{
    SystemInfoFacts facts;
    facts.appName = appName;
    facts.appVersion = kVersionString;
    facts.gitSha = SABERCAT_GIT_SHA;
    facts.buildConfig = kBuildConfig;
    facts.osVersion = osVersionText();
    facts.validation = validation;
    if (device != nullptr)
    {
        const VkPhysicalDeviceProperties & properties = device->properties();
        facts.hasDevice = true;
        facts.gpuName = properties.deviceName;
        facts.vendorId = properties.vendorID;
        facts.deviceId = properties.deviceID;
        facts.driverVersion = properties.driverVersion;
        facts.apiVersion = properties.apiVersion;
        facts.rayQuery = device->features().rayQuery;
        facts.floatAtomics = device->features().shaderBufferFloat32AtomicAdd;
    }
    return facts;
}

} // namespace sabercat::app
