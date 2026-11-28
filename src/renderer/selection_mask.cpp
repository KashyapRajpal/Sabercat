// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/selection_mask.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sabercat/buffer.h"
#include "sabercat/device.h"
#include "sabercat/vk_check.h"

namespace sabercat
{

namespace
{

constexpr uint32_t kGaussiansPerWord = 32;
constexpr VkDeviceSize kWordBytes = sizeof(uint32_t);

// Words that hold `gaussians` bits: ceil(gaussians / 32), no minimum.
size_t wordsForGaussians(size_t gaussians)
{
    return (gaussians + kGaussiansPerWord - 1) / kGaussiansPerWord;
}

// Words allocated for a capacity: at least one, so the buffer binding is always valid.
size_t wordsForCapacity(size_t capacity)
{
    return std::max<size_t>(1, wordsForGaussians(capacity));
}

} // namespace

SelectionMask::SelectionMask(Device & device, size_t capacity)
    : m_device(device), m_capacity(capacity), m_wordCount(wordsForCapacity(capacity))
{
    VK_VALIDATE(capacity >= 1, "SelectionMask: the capacity must be at least one Gaussian");

    const VkDeviceSize bytes = static_cast<VkDeviceSize>(m_wordCount) * kWordBytes;
    m_buffer = std::make_unique<Buffer>(device, bytes);
    const VkBuffer handle = m_buffer->handle();
    device.immediateSubmit(
        [&](VkCommandBuffer cmd)
        {
            vkCmdFillBuffer(cmd, handle, 0, VK_WHOLE_SIZE, 0u);
        });
}

SelectionMask::~SelectionMask() = default;

const Buffer & SelectionMask::buffer() const
{
    return *m_buffer;
}

BufferView SelectionMask::view() const
{
    return m_buffer->view(static_cast<uint32_t>(kWordBytes));
}

void SelectionMask::resize(size_t newCapacity)
{
    if (newCapacity <= m_capacity)
    {
        return;
    }

    const size_t newWords = wordsForCapacity(newCapacity);
    if (newWords == m_wordCount)
    {
        // The bits past the old capacity lie in the last word and are already zero,
        // since they are past the count too. Nothing to copy.
        m_capacity = newCapacity;
        return;
    }

    const VkDeviceSize oldBytes = static_cast<VkDeviceSize>(m_wordCount) * kWordBytes;
    const VkDeviceSize newBytes = static_cast<VkDeviceSize>(newWords) * kWordBytes;
    auto grown = std::make_unique<Buffer>(m_device, newBytes);

    const VkBuffer oldHandle = m_buffer->handle();
    const VkBuffer newHandle = grown->handle();
    // The copy covers [0, oldBytes) and the fill covers [oldBytes, newBytes): the
    // two never overlap, so no barrier is needed between them.
    m_device.immediateSubmit(
        [&](VkCommandBuffer cmd)
        {
            VkBufferCopy region{};
            region.srcOffset = 0;
            region.dstOffset = 0;
            region.size = oldBytes;
            vkCmdCopyBuffer(cmd, oldHandle, newHandle, 1, &region);
            vkCmdFillBuffer(cmd, newHandle, oldBytes, newBytes - oldBytes, 0u);
        });

    m_buffer = std::move(grown);
    m_wordCount = newWords;
    m_capacity = newCapacity;
}

std::vector<uint32_t> SelectionMask::download(size_t count) const
{
    VK_VALIDATE(count <= m_capacity,
                "SelectionMask::download: " + std::to_string(count)
                    + " Gaussians exceed the capacity of " + std::to_string(m_capacity));

    std::vector<uint32_t> words(wordsForGaussians(count), 0u);
    if (!words.empty())
    {
        m_buffer->download(words.data(), words.size() * sizeof(uint32_t));
    }
    return words;
}

void SelectionMask::upload(const std::vector<uint32_t> & words)
{
    if (words.size() > m_wordCount)
    {
        throw std::runtime_error("SelectionMask::upload: " + std::to_string(words.size())
                                 + " words exceed the mask's " + std::to_string(m_wordCount)
                                 + " words");
    }

    // The words first, then zeros over the rest of the buffer.
    std::vector<uint32_t> full(m_wordCount, 0u);
    std::copy(words.begin(), words.end(), full.begin());
    m_buffer->upload(full.data(), full.size() * sizeof(uint32_t));
}

} // namespace sabercat
