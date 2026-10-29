// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The system facts behind Help > About > Copy system info: the app and build
// version, the OS, the GPU and its driver, the Vulkan and feature support, and
// the log file. formatSystemInfo() is pure (the CPU unit tests call it);
// gatherSystemInfo() reads the OS and the device.
#pragma once

#include <cstdint>
#include <string>

namespace sabercat
{
class Device;
}

namespace sabercat::app
{

struct SystemInfoFacts
{
    std::string appName;     // "Sabercat Viewer"
    std::string appVersion;  // kVersionString, e.g. "1.0.0-beta.1"
    std::string gitSha;      // short commit, or "unknown"
    std::string buildConfig; // "Debug" or "Release"
    std::string osVersion;   // e.g. "Windows 11 (10.0 build 26200)"
    bool hasDevice = false;  // false when the renderer has no Vulkan device
    std::string gpuName;
    uint32_t vendorId = 0;
    uint32_t deviceId = 0;
    uint32_t driverVersion = 0; // VkPhysicalDeviceProperties::driverVersion, raw
    uint32_t apiVersion = 0;    // VkPhysicalDeviceProperties::apiVersion, raw
    bool rayQuery = false;
    bool floatAtomics = false;  // shaderBufferFloat32AtomicAdd
    bool validation = false;
};

// The driver version as a string, decoded per vendor from the raw
// VkPhysicalDeviceProperties::driverVersion:
//  - NVIDIA (0x10DE) packs major in bits 22-31, minor in 14-21, secondary in
//    6-13 and tertiary in 0-5; the displayed driver "552.22" is major.minor.
//    Secondary and tertiary are shown only when non-zero.
//  - Intel (0x8086) on Windows packs major in bits 14-31 and minor in 0-13.
//  - Anything else uses the Vulkan VK_MAKE_API_VERSION layout (major 22-31,
//    minor 12-21, patch 0-11).
// The raw value is printed next to it by formatSystemInfo.
std::string decodeDriverVersion(uint32_t vendorId, uint32_t driverVersion);

// A Vulkan apiVersion as "major.minor.patch".
std::string formatVulkanApiVersion(uint32_t apiVersion);

// The plain-text block Copy system info puts on the clipboard. `logFile` is
// the mirror log's path, or "" when logging is off.
std::string formatSystemInfo(const SystemInfoFacts & facts, const std::string & logFile);

// Reads the OS version and, when `device` is non-null, the GPU. Call it after the
// device is up; pass nullptr when there is none.
SystemInfoFacts gatherSystemInfo(const std::string & appName, const Device * device,
                                 bool validation);

} // namespace sabercat::app
