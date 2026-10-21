// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/borrowed.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "sabercat/render_config.h"

namespace sabercat
{

namespace
{

void require(bool condition, const std::string & message)
{
    if (!condition)
    {
        throw std::invalid_argument("borrowed Gaussians: " + message);
    }
}

// One stream of the borrowed scene, with what the canonical contract says
// about it.
struct Stream
{
    const char * name;
    const BufferView * view;
    uint32_t stride;            // canonical element stride
    uint32_t elementsPerGaussian; // 16 for SH, 1 otherwise

    VkDeviceSize bytesPerGaussian() const
    {
        return static_cast<VkDeviceSize>(stride) * elementsPerGaussian;
    }
};

// Checks one stream's own properties; returns how many whole Gaussians it
// holds.
VkDeviceSize checkStream(const Stream & s, VkDevice device, const VkPhysicalDeviceLimits & limits)
{
    const BufferView & v = *s.view;
    const std::string name = s.name;
    require(v.buffer != VK_NULL_HANDLE, name + ": buffer is VK_NULL_HANDLE");
    require(v.stride == s.stride, name + ": stride " + std::to_string(v.stride)
                                      + ", the canonical stride is " + std::to_string(s.stride));
    require(v.offset % limits.minStorageBufferOffsetAlignment == 0,
            name + ": offset " + std::to_string(v.offset)
                + " is not a multiple of minStorageBufferOffsetAlignment ("
                + std::to_string(limits.minStorageBufferOffsetAlignment) + ")");
    require(v.size >= s.bytesPerGaussian(),
            name + ": size " + std::to_string(v.size) + " holds no whole Gaussian ("
                + std::to_string(s.bytesPerGaussian()) + " B each)");
    require(v.size <= limits.maxStorageBufferRange,
            name + ": size " + std::to_string(v.size) + " exceeds maxStorageBufferRange ("
                + std::to_string(limits.maxStorageBufferRange) + ")");
    // A buffer is never larger than its memory requirement, so a range past
    // it is certainly outside the buffer.
    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device, v.buffer, &requirements);
    require(v.offset + v.size <= requirements.size,
            name + ": range [" + std::to_string(v.offset) + ", " + std::to_string(v.offset + v.size)
                + ") runs past the end of its buffer (" + std::to_string(requirements.size)
                + " B)");
    return v.size / s.bytesPerGaussian();
}

} // namespace

uint32_t validateBorrowedGaussians(VkPhysicalDevice physicalDevice, VkDevice device,
                                   const BorrowedGaussians & scene)
{
    require(physicalDevice != VK_NULL_HANDLE && device != VK_NULL_HANDLE,
            "no device to validate against");
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physicalDevice, &props);
    const VkPhysicalDeviceLimits & limits = props.limits;

    const CanonicalGaussianBufferViews & p = scene.parameters;
    const CanonicalGaussianGradientViews & g = scene.gradients;
    const uint32_t sh = kShCoefficients;
    const Stream parameters[] = {
        {"parameters.means", &p.means, kMeansStride, 1},
        {"parameters.scales", &p.scales, kScalesStride, 1},
        {"parameters.rotations", &p.rotations, kRotationsStride, 1},
        {"parameters.opacity", &p.opacity, kOpacityStride, 1},
        {"parameters.shCoefficients", &p.shCoefficients, kShStride, sh},
    };
    const Stream gradients[] = {
        {"gradients.means", &g.means, kMeansStride, 1},
        {"gradients.scales", &g.scales, kScalesStride, 1},
        {"gradients.rotations", &g.rotations, kRotationsStride, 1},
        {"gradients.opacity", &g.opacity, kOpacityStride, 1},
        {"gradients.shCoefficients", &g.shCoefficients, kShStride, sh},
        {"gradients.screenMeans", &g.screenMeans, kMeansStride, 1},
    };

    // Parameters: capacity is the smallest stream's whole-Gaussian count.
    VkDeviceSize capacity = ~VkDeviceSize(0);
    for (const Stream & s : parameters)
    {
        capacity = std::min(capacity, checkStream(s, device, limits));
    }
    require(capacity <= UINT32_MAX, "capacity exceeds 2^32 Gaussians");
    if (p.confidence.buffer != VK_NULL_HANDLE)
    {
        // Optional and unused by the rasterizer, but if given it must be well formed.
        checkStream({"parameters.confidence", &p.confidence, kOpacityStride, 1}, device, limits);
    }
    require(p.shDegree <= kShDegree, "parameters.shDegree " + std::to_string(p.shDegree)
                                         + " exceeds " + std::to_string(kShDegree));
    require(p.gaussianCount >= 1 && p.gaussianCount <= capacity,
            "parameters.gaussianCount " + std::to_string(p.gaussianCount)
                + " is outside [1, capacity = " + std::to_string(capacity) + "]");

    // Gradients: all or none, each covering the parameter capacity (the
    // count may rise to it without a rebind).
    size_t gradientCount = 0;
    for (const Stream & s : gradients)
    {
        gradientCount += s.view->buffer != VK_NULL_HANDLE ? 1 : 0;
    }
    require(gradientCount == 0 || gradientCount == 6,
            "gradients: " + std::to_string(gradientCount)
                + " of 6 streams given -- borrow all six or none");
    if (gradientCount == 6)
    {
        for (const Stream & s : gradients)
        {
            const VkDeviceSize held = checkStream(s, device, limits);
            require(held >= capacity, std::string(s.name) + " holds " + std::to_string(held)
                                          + " Gaussians, fewer than the parameter capacity "
                                          + std::to_string(capacity));
        }
    }

    // Relightable streams, each optional, each covering the
    // capacity when given.
    const RelightBufferViews & r = scene.relight;
    const Stream relight[] = {
        {"relight.normals", &r.normals, kNormalsStride, 1},
        {"relight.baseColorRoughness", &r.baseColorRoughness, kBaseColorRoughnessStride, 1},
        {"relight.metallic", &r.metallic, kMetallicStride, 1},
        {"relight.visibility", &r.visibility, kVisibilityStride, kVisibilityCoefficients},
    };
    for (const Stream & s : relight)
    {
        if (s.view->buffer == VK_NULL_HANDLE)
        {
            continue;
        }
        const VkDeviceSize held = checkStream(s, device, limits);
        require(held >= capacity, std::string(s.name) + " holds " + std::to_string(held)
                                      + " Gaussians, fewer than the parameter capacity "
                                      + std::to_string(capacity));
    }

    // Aliasing: the ranges Sabercat reads or writes (capacity Gaussians from
    // each offset) must be disjoint, or backward() would overwrite
    // parameters, or one stream another.
    std::vector<const Stream *> used;
    for (const Stream & s : parameters)
    {
        used.push_back(&s);
    }
    for (const Stream & s : relight)
    {
        if (s.view->buffer != VK_NULL_HANDLE)
        {
            used.push_back(&s);
        }
    }
    if (gradientCount == 6)
    {
        for (const Stream & s : gradients)
        {
            used.push_back(&s);
        }
    }
    for (size_t i = 0; i < used.size(); ++i)
    {
        for (size_t j = i + 1; j < used.size(); ++j)
        {
            const BufferView & a = *used[i]->view;
            const BufferView & b = *used[j]->view;
            if (a.buffer != b.buffer)
            {
                continue;
            }
            const VkDeviceSize aEnd = a.offset + capacity * used[i]->bytesPerGaussian();
            const VkDeviceSize bEnd = b.offset + capacity * used[j]->bytesPerGaussian();
            require(aEnd <= b.offset || bEnd <= a.offset,
                    std::string(used[i]->name) + " and " + used[j]->name
                        + " overlap in the same buffer");
        }
    }
    return static_cast<uint32_t>(capacity);
}

} // namespace sabercat
