// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// A bounded fake Vulkan allocator catches allocations retained by repeated
// submissions while the pool stays alive. No GPU or driver calls are needed.
#include "sabercat/command_buffer.h"

#include <cstdio>
#include <stdexcept>
#include <type_traits>

namespace
{
int token;
bool allocated = false;
unsigned allocations = 0;
unsigned frees = 0;
unsigned errors = 0;
VkDevice ownerDevice = VK_NULL_HANDLE;
VkCommandPool ownerPool = VK_NULL_HANDLE;
} // namespace

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkAllocateCommandBuffers(
    VkDevice device, const VkCommandBufferAllocateInfo * info, VkCommandBuffer * out)
{
    if (allocated || info->commandBufferCount != 1)
    {
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    }
    allocated = true;
    ++allocations;
    ownerDevice = device;
    ownerPool = info->commandPool;
    *out = reinterpret_cast<VkCommandBuffer>(&token);
    return VK_SUCCESS;
}

extern "C" VKAPI_ATTR void VKAPI_CALL vkFreeCommandBuffers(
    VkDevice device, VkCommandPool pool, uint32_t count, const VkCommandBuffer * buffers)
{
    if (!allocated || device != ownerDevice || pool != ownerPool || count != 1
        || buffers[0] != reinterpret_cast<VkCommandBuffer>(&token))
    {
        ++errors;
    }
    allocated = false;
    ++frees;
}

int main()
{
    using sabercat::CommandBuffer;
    static_assert(!std::is_copy_constructible_v<CommandBuffer>);
    static_assert(!std::is_copy_assignable_v<CommandBuffer>);
    try
    {
        const auto device = reinterpret_cast<VkDevice>(&ownerDevice);
        const auto pool = reinterpret_cast<VkCommandPool>(&ownerPool);
        for (unsigned frame = 0; frame < 10000; ++frame)
        {
            CommandBuffer cmd(device, pool);
            if (!allocated || cmd.getHandle() != reinterpret_cast<VkCommandBuffer>(&token))
            {
                ++errors;
            }
        }
        // Recording can throw before submission; that allocation also belongs
        // to the scope, rather than remaining live until renderer shutdown.
        try
        {
            CommandBuffer cmd(device, pool);
            throw std::runtime_error("recording failed");
        }
        catch (const std::runtime_error &)
        {
        }
        if (allocated || allocations != 10001 || frees != allocations || errors != 0)
        {
            throw std::runtime_error("command-buffer ownership imbalance");
        }
    }
    catch (const std::exception & e)
    {
        std::fprintf(stderr, "command_buffer_test FAILED: %s\n", e.what());
        return 1;
    }
    std::puts("command_buffer_test PASSED: 10000 submissions and exception cleanup");
    return 0;
}
