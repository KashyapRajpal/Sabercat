// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vulkan/vulkan.h>

#include <glm/glm.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "sabercat/buffer_view.h"
#include "sabercat/descriptor_set.h"
#include "sabercat/lighting.h"
#include "sabercat/pipeline.h"

namespace sabercat
{

class Device;
class Buffer;
class Image;
class ForwardPass;

// Relighting over a completed forward run.
//
//   runGBuffer  -- relight_prepare (per-Gaussian Surfels: relightable streams
//                  or derived defaults) then gbuffer_rasterize over
//                  ForwardPass's tile-sorted lists (no second sort).
//   [a shadow source fills visibility() -- or it stays all ones]
//   runLighting -- deferred_lighting: G-buffer + lights + environment ->
//                  lit buffer and image.
//
// Each run is one synchronous submission on Device::submitQueue(), bracketed
// by Device::recordGlobalBarrier (see docs/library.md, synchronization), so
// every buffer below may be read on that queue as soon as the call returns.
// All per-pixel buffers are row-major width*height (visibility: one such
// layer per light, kMaxLights layers).
class RelightPass
{
  public:
    RelightPass(Device & device, uint32_t width, uint32_t height, const std::string & shaderDir);
    ~RelightPass();

    RelightPass(const RelightPass &) = delete;
    RelightPass & operator=(const RelightPass &) = delete;

    // Reallocates every per-pixel buffer and the lit image.
    void resize(uint32_t width, uint32_t height);

    // Splats the G-buffer of `forward`'s last completed run (either mode).
    // `relight` streams are each optional (invalid => derived default).
    void runGBuffer(const ForwardPass & forward, const RelightBufferViews & relight);

    // Shades the G-buffer. `resetVisibility` fills every visibility layer
    // with 1 first (no shadows); pass false when a shadow source wrote it.
    void runLighting(const ForwardPass & forward, const std::vector<Light> & lights,
                     const EnvironmentLight & environment, const LightingSettings & settings,
                     bool resetVisibility);

    // Fills every visibility layer with 1 (no shadows). A shadow source that
    // writes only the layers of shadow-casting lights calls this first, so
    // the other layers never hold a previous frame's values.
    void clearVisibility();

    // For present(): GENERAL -> TRANSFER_SRC_OPTIMAL on the lit image.
    void prepareLitForRead(VkCommandBuffer cmd);

    const Buffer & surfels() const
    {
        return *m_surfels;
    }
    const Buffer & gNormalDepth() const
    {
        return *m_gNormalDepth;
    }
    const Buffer & gBaseRoughness() const
    {
        return *m_gBaseRoughness;
    }
    const Buffer & gMetalAmbientCoverage() const
    {
        return *m_gMetalAmbientCoverage;
    }
    // kMaxLights layers of width*height floats: the light visibility the
    // lighting pass multiplies direct lights by.
    const Buffer & visibility() const
    {
        return *m_visibility;
    }
    const Buffer & litBuffer() const
    {
        return *m_litBuffer;
    }
    const Image & litImage() const
    {
        return *m_litImage;
    }
    uint32_t width() const
    {
        return m_width;
    }
    uint32_t height() const
    {
        return m_height;
    }

  private:
    // The destructor's body, also run by the constructor when it throws
    // part-way (a throwing constructor never reaches the destructor).
    void release();
    void createPipelines(const std::string & shaderDir);
    void allocatePixelBuffers();
    void submitAndWait(VkCommandBuffer cmd, const char * what);

    Device & m_device;
    uint32_t m_width;
    uint32_t m_height;
    uint32_t m_surfelCapacity = 0; // grow-only, in Gaussians
    VkCommandPool m_commandPool = VK_NULL_HANDLE;

    std::unique_ptr<Buffer> m_dummy; // bound for absent optional streams
    std::unique_ptr<Buffer> m_surfels;
    std::unique_ptr<Buffer> m_gNormalDepth;
    std::unique_ptr<Buffer> m_gBaseRoughness;
    std::unique_ptr<Buffer> m_gMetalAmbientCoverage;
    std::unique_ptr<Buffer> m_visibility;
    std::unique_ptr<Buffer> m_litBuffer;
    std::unique_ptr<Buffer> m_lights;
    std::unique_ptr<Buffer> m_params;
    std::unique_ptr<Image> m_litImage;

    DescriptorSet m_prepareDescriptors;
    VkDescriptorSetLayout m_prepareLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_prepareSet = VK_NULL_HANDLE;
    Pipeline m_preparePipeline;

    DescriptorSet m_gbufferDescriptors;
    VkDescriptorSetLayout m_gbufferLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_gbufferSet = VK_NULL_HANDLE;
    Pipeline m_gbufferPipeline;

    DescriptorSet m_lightingDescriptors;
    VkDescriptorSetLayout m_lightingLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_lightingSet = VK_NULL_HANDLE;
    Pipeline m_lightingPipeline;
};

} // namespace sabercat
