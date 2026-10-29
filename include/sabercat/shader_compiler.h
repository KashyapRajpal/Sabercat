// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vulkan/vulkan.h>

#include <string>
#include <vector>

namespace sabercat
{

// Loads SPIR-V compiled ahead-of-time from .slang by slangc (see the
// examples/*/CMakeLists.txt custom commands) and turns it into a
// VkShaderModule. No runtime Slang compilation -- that would require linking
// the Slang compiler library, which this project doesn't need.
class ShaderCompiler
{
  public:
    ShaderCompiler(VkDevice device);
    ~ShaderCompiler();

    // Load precompiled SPIR-V from file.
    std::vector<uint32_t> loadSPIRV(const std::string & path);

    // Create shader module from SPIR-V.
    VkShaderModule createShaderModule(const std::vector<uint32_t> & spirv);

    // Destroy shader module.
    void destroyShaderModule(VkShaderModule module);

  private:
    VkDevice m_device;
};

} // namespace sabercat
