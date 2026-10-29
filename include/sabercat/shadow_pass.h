// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vulkan/vulkan.h>

#include <glm/glm.hpp>

#include <cstdint>
#include <memory>
#include <string>

#include "sabercat/buffer_view.h"
#include "sabercat/camera.h"
#include "sabercat/descriptor_set.h"
#include "sabercat/lighting.h"
#include "sabercat/pipeline.h"

namespace sabercat
{

class Device;
class Buffer;
class ForwardPass;
class RelightPass;
class GaussianAccelerationStructure;

// Per-light visibility for the lighting pass.
//
// runPCSS renders the splats from a light camera with a private ForwardPass
// (Viewer mode, LOD off), reduces that view to a median-transmittance-depth
// shadow map (shadow_map.slang), then filters it with percentage-closer soft
// shadows (shadow_pcss.slang) into one layer of RelightPass::visibility().
// Each call is synchronous on Device::submitQueue() with the usual
// bracketing barriers (see docs/library.md, synchronization).
class ShadowPass
{
  public:
    ShadowPass(Device & device, const std::string & shaderDir);
    ~ShadowPass();

    ShadowPass(const ShadowPass &) = delete;
    ShadowPass & operator=(const ShadowPass &) = delete;

    // Fills visibility layer `layer` for `light`, for the receivers in
    // `relight`'s G-buffer (seen through `receiverCamera`). The light camera
    // frames the sphere (focusCenter, focusRadius). `antialiased` must match
    // the scene's flag so the shadow map splats what the image splats.
    void runPCSS(const RelightPass & relight, const CameraUniforms & receiverCamera,
                 const CanonicalGaussianBufferViews & scene, bool antialiased, const Light & light,
                 uint32_t layer, const ShadowSettings & settings, const glm::vec3 & focusCenter,
                 float focusRadius);

    // Ray-traced shadows (ground truth): V = mean transmittance of the
    // `settings.raySamplesPerAxis`^2 stratified rays from each receiver to
    // the emitter, traced through the Gaussians of `accelerationStructure`
    // (built over `scene` by the caller). Needs `Device::features().rayQuery`.
    // Fills visibility layer `layer`.
    //
    // Progressive: traces grid samples [firstSample, firstSample +
    // sampleCount), adding to this layer's running sum (firstSample 0
    // restarts it), and writes the mean so far. The caller keeps the G-buffer
    // and inputs fixed between the calls of one accumulation; at n*n samples
    // the layer equals a single full call bit for bit. sampleCount 0 only
    // re-resolves the sum (no rays). The defaults trace the whole grid.
    void runRayTraced(const RelightPass & relight, const CameraUniforms & receiverCamera,
                      const CanonicalGaussianBufferViews & scene,
                      const GaussianAccelerationStructure & accelerationStructure, const Light & light,
                      uint32_t layer, const ShadowSettings & settings, float focusRadius,
                      uint32_t firstSample = 0, uint32_t sampleCount = UINT32_MAX);

    // The last light camera and shadow map (mapResolution^2 float2:
    // median depth, final transmittance) -- verification and debug views.
    const CameraUniforms & lightCamera() const
    {
        return m_lightCamera;
    }
    const Buffer & shadowMap() const
    {
        return *m_shadowMap;
    }

    // The perspective camera PCSS renders `light` from (exposed for tests).
    static CameraUniforms lightCameraFor(const Light & light, uint32_t resolution,
                                         const glm::vec3 & focusCenter, float focusRadius,
                                         uint32_t gaussianCount, bool antialiased);

  private:
    // The destructor's body, also run by the constructor when it throws
    // part-way (a throwing constructor never reaches the destructor).
    void release();
    void createPipelines(const std::string & shaderDir);

    Device & m_device;
    std::string m_shaderDir;
    uint32_t m_mapResolution = 0;
    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    CameraUniforms m_lightCamera{};

    std::unique_ptr<ForwardPass> m_lightForward; // built/resized on demand
    std::unique_ptr<Buffer> m_shadowMap;
    std::unique_ptr<Buffer> m_params;

    DescriptorSet m_mapDescriptors;
    VkDescriptorSetLayout m_mapLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_mapSet = VK_NULL_HANDLE;
    Pipeline m_mapPipeline;

    DescriptorSet m_pcssDescriptors;
    VkDescriptorSetLayout m_pcssLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_pcssSet = VK_NULL_HANDLE;
    Pipeline m_pcssPipeline;

    // Ray-traced path, built on first use.
    std::unique_ptr<Buffer> m_rayParams;
    std::unique_ptr<Buffer> m_rayAccum;  // kMaxLights layers of running sums
    std::unique_ptr<Buffer> m_rayLowRes; // rayDownsample > 1: the traced receivers' results
    DescriptorSet m_upsampleDescriptors;
    VkDescriptorSetLayout m_upsampleLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_upsampleSet = VK_NULL_HANDLE;
    Pipeline m_upsamplePipeline;
    DescriptorSet m_rayDescriptors;
    VkDescriptorSetLayout m_rayLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_raySet = VK_NULL_HANDLE;
    Pipeline m_rayPipeline;
};

} // namespace sabercat
