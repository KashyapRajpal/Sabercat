// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vulkan/vulkan.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "sabercat/buffer_view.h"

namespace sabercat
{

class Buffer;
class Device;

// A layer's selection (Studio phase C3): one bit per Gaussian, packed into uint32
// words. Bit (i & 31) of word (i >> 5) is Gaussian i. The mask is sized to the
// layer's CAPACITY, not its count, and bits at or beyond the layer's count are
// always zero: every kernel and upload keeps that invariant.
//
// The GPU buffer is created with the default Buffer usage (storage plus transfer),
// so the edit kernels bind it and the host reads it back.
class SelectionMask
{
  public:
    // `capacity` is in Gaussians and must be at least 1. Allocates
    // max(1, ceil(capacity / 32)) words, all zero (one immediateSubmit).
    SelectionMask(Device & device, size_t capacity);
    ~SelectionMask();

    // Non-copyable: it owns its device buffer.
    SelectionMask(const SelectionMask &) = delete;
    SelectionMask & operator=(const SelectionMask &) = delete;

    size_t capacity() const
    {
        return m_capacity;
    }
    size_t wordCount() const
    {
        return m_wordCount;
    }

    const Buffer & buffer() const;
    // A non-owning view of the whole buffer, stride 4 (one uint per word). Re-fetch
    // it after resize(): growth replaces the buffer.
    BufferView view() const;

    // Grows the mask to hold `newCapacity` Gaussians. A request at or below the
    // current capacity is a no-op. The contents are kept, by one GPU buffer copy,
    // and the new words are zero. Growth may replace the Buffer, so the handle
    // returned by buffer() and view() can change: fetch them again.
    void resize(size_t newCapacity);

    // Reads back the words of the first `count` Gaussians: ceil(count / 32) words.
    // count must not exceed capacity(); a count of 0 returns an empty vector.
    std::vector<uint32_t> download(size_t count) const;

    // Replaces the mask's words with `words`, zeroing the rest of the buffer.
    // Throws std::runtime_error if words.size() exceeds wordCount().
    void upload(const std::vector<uint32_t> & words);

  private:
    Device & m_device;
    size_t m_capacity = 0;
    size_t m_wordCount = 0;
    std::unique_ptr<Buffer> m_buffer;
};

} // namespace sabercat
