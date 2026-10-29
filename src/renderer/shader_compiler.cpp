// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/shader_compiler.h"

#include <fstream>

#include "sabercat/vk_check.h"

namespace sabercat
{

ShaderCompiler::ShaderCompiler(VkDevice device) : m_device(device)
{
}

ShaderCompiler::~ShaderCompiler()
{
}

std::vector<uint32_t> ShaderCompiler::loadSPIRV(const std::string & path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open SPIR-V file: " + path);
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    VK_VALIDATE(fileSize % sizeof(uint32_t) == 0,
                "SPIR-V file size is not a multiple of 4 bytes: " + path);

    std::vector<uint32_t> spirv(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char *>(spirv.data()), fileSize);
    return spirv;
}

VkShaderModule ShaderCompiler::createShaderModule(const std::vector<uint32_t> & spirv)
{
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = spirv.size() * sizeof(uint32_t);
    createInfo.pCode = spirv.data();

    VkShaderModule shaderModule;
    VK_CHECK(vkCreateShaderModule(m_device, &createInfo, nullptr, &shaderModule),
             "Failed to create shader module");
    return shaderModule;
}

void ShaderCompiler::destroyShaderModule(VkShaderModule module)
{
    if (module != VK_NULL_HANDLE)
    {
        vkDestroyShaderModule(m_device, module, nullptr);
    }
}

} // namespace sabercat
