// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Section N of edit_test: the mask operations and the selection of a layer above the
// 65535-workgroup limit (17,000,003 Gaussians, past 65535 * 256 = 16,776,960). Only the
// means stream is filled, with vkCmdFillBuffer (every lane 1.0f); the other streams are
// never touched. The cloud is allocated without a host copy and destroyed as soon as its
// checks are done. Checks: selectAll, countSelected, the masked centroid (exact sums),
// invertMask, a selection of the last 10 Gaussians (past the boundary), and
// expandMaskToFlags in both senses. The section prints its elapsed time.

#include "edit_test_common.h"

#include <glm/glm.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

#include "../common/shader_dir.h"
#include "sabercat/buffer.h"
#include "sabercat/buffer_view.h"
#include "sabercat/device.h"
#include "sabercat/edit_pass.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/render_config.h"
#include "sabercat/selection_mask.h"

using namespace sabercat;

namespace edit_test
{
namespace
{

// Not a multiple of 32: the last mask word is partial.
constexpr size_t kLargeCount = 17000003;
// The first Gaussian index past the largest 1-D dispatch: 65535 groups of 256.
constexpr size_t kGroupLimitGaussians = 65535u * 256u;
constexpr size_t kTailCount = 10;
constexpr uint32_t kOneBits = 0x3F800000u; // 1.0f

// The mask words of `count` Gaussians with bits [from, to) set.
std::vector<uint32_t> bitRange(size_t from, size_t to, size_t count)
{
    std::vector<uint32_t> words((count + 31) / 32, 0u);
    for (size_t i = from; i < to; ++i)
    {
        words[i >> 5] |= 1u << (i & 31u);
    }
    return words;
}

// Whether every component of v is exactly x.
bool exactlyEqual(const glm::dvec3 & v, double x)
{
    return v.x == x && v.y == x && v.z == x;
}

} // namespace

void checkLarge(Device & device)
{
    const auto start = std::chrono::steady_clock::now();
    const uint32_t count = static_cast<uint32_t>(kLargeCount);

    std::unique_ptr<GaussianCloud> cloud;
    try
    {
        cloud = std::make_unique<GaussianCloud>(device);
        cloud->allocateGPU(kLargeCount, 0, RelightPresence{});
    }
    catch (const std::exception & e)
    {
        check(false, std::string("N: allocateGPU(17000003) threw: ") + e.what());
        return;
    }
    check(cloud->count() == kLargeCount && cloud->capacity() == kLargeCount,
          "N: allocateGPU(17000003) count and capacity");

    // The means: every lane 1.0f. Only this stream is written.
    const VkDeviceSize meansBytes = static_cast<VkDeviceSize>(kLargeCount) * kMeansStride;
    const BufferView means = cloud->views().means;
    device.immediateSubmit(
        [&](VkCommandBuffer cmd)
        { vkCmdFillBuffer(cmd, means.buffer, means.offset, meansBytes, kOneBits); });

    EditPass edit(device, testShaderDir());
    SelectionMask mask(device, kLargeCount);

    // Every Gaussian selected.
    edit.selectAll(mask, count);
    check(edit.countSelected(mask, count) == count, "N: selectAll counts 17000003");
    const MaskCentroid all = edit.maskedCentroid(*cloud, mask);
    check(all.count == kLargeCount, "N: centroid counts every Gaussian");
    check(exactlyEqual(all.sum, 17000003.0), "N: centroid sum is (17000003, 17000003, 17000003)");
    check(exactlyEqual(all.centroid(), 1.0), "N: centroid is (1, 1, 1)");

    // Inverted: nothing selected.
    edit.invertMask(mask, count);
    check(edit.countSelected(mask, count) == 0, "N: invertMask of a full selection is empty");

    // Only the last kTailCount Gaussians, beyond the 16,776,960 boundary.
    mask.upload(bitRange(kLargeCount - kTailCount, kLargeCount, kLargeCount));
    check(edit.countSelected(mask, count) == kTailCount, "N: the tail selects 10");
    const MaskCentroid tail = edit.maskedCentroid(*cloud, mask);
    check(tail.count == kTailCount && exactlyEqual(tail.sum, 10.0),
          "N: the tail centroid sum is (10, 10, 10)");

    // The flags of the tail selection, with the selection kept and inverted.
    const VkDeviceSize flagBytes = static_cast<VkDeviceSize>(kLargeCount) * sizeof(uint32_t);
    Buffer flagBuffer(device, flagBytes);
    const BufferView flags{flagBuffer.handle(), 0, flagBytes, sizeof(uint32_t)};
    std::vector<uint32_t> host(kLargeCount);

    edit.expandMaskToFlags(mask, count, false, flags);
    flagBuffer.download(host.data(), flagBytes);
    uint64_t ones = 0;
    for (const uint32_t v : host)
    {
        ones += v;
    }
    bool tailFlagged = true;
    for (size_t i = kLargeCount - kTailCount; i < kLargeCount; ++i)
    {
        tailFlagged = tailFlagged && host[i] == 1u;
    }
    check(tailFlagged, "N: expandMaskToFlags marks the last 10 Gaussians");
    check(host[kLargeCount - kTailCount - 1] == 0u, "N: the Gaussian before the tail is 0");
    check(host[kGroupLimitGaussians + 5] == 0u, "N: a Gaussian past the group limit is 0");
    check(ones == kTailCount, "N: expandMaskToFlags (select) has exactly 10 ones");

    edit.expandMaskToFlags(mask, count, true, flags);
    flagBuffer.download(host.data(), flagBytes);
    ones = 0;
    for (const uint32_t v : host)
    {
        ones += v;
    }
    check(ones == kLargeCount - kTailCount,
          "N: expandMaskToFlags (keep unselected) has count - 10 ones");

    // The cloud is not needed past this point: release its device memory now.
    cloud.reset();

    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("N: large section (17000003 Gaussians) took %.2f s\n", seconds);
}

} // namespace edit_test
