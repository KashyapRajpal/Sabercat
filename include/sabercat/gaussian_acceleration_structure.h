// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <string>

#include "sabercat/buffer_view.h"
#include "sabercat/descriptor_set.h"
#include "sabercat/pipeline.h"

namespace sabercat
{

class Device;
class Buffer;

// Ray-traced visibility for relighting: the Gaussian ray-tracing core. Turns the canonical
// parameter streams into a Vulkan acceleration structure that ray-query
// kernels trace with shaders/gaussian_ray.slangh's gaussianTransmittance:
//
//   gaussian_aabbs   compute: one VkAabbPositionsKHR per Gaussian, the exact
//                    box around its 3-sigma ellipsoid (collapsed to a point
//                    for Gaussians that can never contribute)
//   BLAS             one procedural (AABB) geometry, primitive i = Gaussian i
//   TLAS             one instance of the BLAS, identity transform
//
// Works on owned and borrowed (application-owned) streams alike: the views are bound at
// their offsets and only read. Every call is one synchronous submission to
// Device::submitQueue(), bracketed by Device::recordGlobalBarrier like
// ForwardPass/BackwardPass -- so parameter writes a consumer submitted to that
// queue earlier are seen, and the TLAS is visible to whatever it submits after
// (see docs/library.md, synchronization).
//
// Needs DeviceFeatures::rayQuery (acceleration structures + ray query +
// buffer device address enabled); the constructor throws a named error
// without it, and rasterization never needs it.
class GaussianAccelerationStructure
{
  public:
    // shaderDir must contain gaussian_aabbs.comp.spv.
    GaussianAccelerationStructure(Device & device, const std::string & shaderDir);
    ~GaussianAccelerationStructure();

    GaussianAccelerationStructure(const GaussianAccelerationStructure &) = delete;
    GaussianAccelerationStructure & operator=(const GaussianAccelerationStructure &) = delete;

    // Full rebuild for views.gaussianCount Gaussians (>= 1). The AABB,
    // scratch and BLAS storage only grow; tlas() is created by the first
    // build and never changes afterwards, so descriptor sets binding it stay
    // valid across builds and refits. Throws std::invalid_argument for
    // missing/non-canonical/too-small views.
    //
    // `minAlpha` is the alpha skip the rays traced against it will use
    // (gaussian_ray.slangh's RayCutoffs): boxes bound the support at that
    // cutoff, and Gaussians fainter than it collapse. The default, 1/255, is
    // the exact model; a structure must be traced with the cutoff it was
    // built with (refit() keeps it).
    void build(const CanonicalGaussianBufferViews & views, float minAlpha = 1.0f / 255.0f);

    // Refit (VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR) after the
    // parameters changed but the count did not -- an optimizer step. Same
    // count as the last build() or it throws std::invalid_argument (call
    // build() instead). The BVH topology is kept, so trace quality decays as
    // Gaussians move far from where they were built; rebuild after
    // densification, and periodically (refitsSinceBuild()).
    void refit(const CanonicalGaussianBufferViews & views);

    bool built() const
    {
        return m_tlas != VK_NULL_HANDLE && m_gaussianCount > 0;
    }
    // The Gaussian count of the last build()/refit(); 0 before the first.
    // A kernel tracing tlas() must bind parameter views holding at least
    // this many Gaussians, in the same order.
    uint32_t gaussianCount() const
    {
        return m_gaussianCount;
    }
    uint32_t refitsSinceBuild() const
    {
        return m_refitsSinceBuild;
    }
    float minAlpha() const
    {
        return m_minAlpha;
    }

    // For a VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR binding
    // (DescriptorSet::bindAccelerationStructure).
    VkAccelerationStructureKHR tlas() const
    {
        return m_tlas;
    }
    VkDeviceAddress tlasAddress() const
    {
        return m_tlasAddress;
    }
    VkAccelerationStructureKHR blas() const
    {
        return m_blas;
    }

    // Verification: gaussianCount() VkAabbPositionsKHR (6 floats, 24 B each)
    // as the last build/refit wrote them. Valid after build().
    const Buffer & aabbs() const
    {
        return *m_aabbs;
    }

  private:
    // The destructor's body, also run by the constructor when it throws
    // part-way (a throwing constructor never reaches the destructor).
    void release();
    void createPipeline(const std::string & shaderDir);
    void record(const CanonicalGaussianBufferViews & views, bool update);
    void ensureCapacity(uint32_t count);

    Device & m_device;
    VkCommandPool m_commandPool = VK_NULL_HANDLE;

    // Device-level KHR entry points (the loader does not export them).
    PFN_vkCreateAccelerationStructureKHR m_createAS = nullptr;
    PFN_vkDestroyAccelerationStructureKHR m_destroyAS = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR m_getBuildSizes = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR m_cmdBuildAS = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR m_getASAddress = nullptr;
    VkDeviceSize m_scratchAlignment = 0; // minAccelerationStructureScratchOffsetAlignment

    std::unique_ptr<Buffer> m_aabbs;    // 24 B per Gaussian, grow-only
    std::unique_ptr<Buffer> m_scratch;  // shared by the BLAS and TLAS builds, grow-only
    std::unique_ptr<Buffer> m_blasStorage;
    std::unique_ptr<Buffer> m_tlasStorage;
    std::unique_ptr<Buffer> m_instance; // one VkAccelerationStructureInstanceKHR
    VkAccelerationStructureKHR m_blas = VK_NULL_HANDLE;
    VkAccelerationStructureKHR m_tlas = VK_NULL_HANDLE;
    VkDeviceAddress m_blasAddress = 0;
    VkDeviceAddress m_tlasAddress = 0;
    VkDeviceSize m_blasCapacity = 0;     // bytes m_blas was created with
    VkDeviceSize m_tlasBuildScratch = 0; // the TLAS's build scratch size
    uint32_t m_aabbCapacity = 0;         // Gaussians m_aabbs holds
    uint32_t m_gaussianCount = 0;
    uint32_t m_refitsSinceBuild = 0;
    float m_minAlpha = 1.0f / 255.0f; // the last build's alpha cutoff

    DescriptorSet m_descriptors;
    VkDescriptorSetLayout m_layout = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;
    Pipeline m_pipeline;
};

} // namespace sabercat
