// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/shadow_pass.h"

#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

#include "sabercat/buffer.h"
#include "sabercat/command_buffer.h"
#include "sabercat/device.h"
#include "sabercat/forward_pass.h"
#include "sabercat/gaussian_acceleration_structure.h"
#include "sabercat/relight_pass.h"
#include "sabercat/render_config.h"
#include "sabercat/shader_compiler.h"
#include "sabercat/vk_check.h"

namespace sabercat
{

namespace
{

// Host mirror of shadow_pcss.slang's ShadowParams (std430, 208 B).
struct ShadowParams
{
    glm::mat4 receiverCameraToWorld;
    glm::mat4 lightWorldToCamera;
    glm::vec4 receiverIntrinsics;
    glm::vec4 lightIntrinsics;
    uint32_t width;
    uint32_t height;
    uint32_t mapWidth;
    uint32_t mapHeight;
    uint32_t layer;
    uint32_t blockerSamplesPerAxis;
    uint32_t filterSamplesPerAxis;
    uint32_t directional;
    float lightSize;
    float depthBias;
    float normalOffset;
    float maxFilterTexels;
};
static_assert(sizeof(ShadowParams) == 208, "ShadowParams must match shadow_pcss.slang");

// Host mirror of shadow_rt.slang's RayShadowParams (std430, 192 B).
struct RayShadowParams
{
    glm::mat4 receiverCameraToWorld;
    glm::vec4 receiverIntrinsics;
    glm::vec4 lightPositionType;
    glm::vec4 lightDirection;
    glm::vec4 lightAxisX;
    glm::vec4 lightAxisY;
    uint32_t width;
    uint32_t height;
    uint32_t layer;
    uint32_t samplesPerAxis;
    float lightSize;
    float normalOffset;
    float rayLength;
    uint32_t firstSample;
    uint32_t sampleCount;
    float minAlpha;
    float minTransmittance;
    uint32_t downsample;
};

// shadow_upsample.slang's push constants.
struct UpsampleParams
{
    uint32_t width;
    uint32_t height;
    uint32_t layer;
    uint32_t downsample;
};
static_assert(sizeof(RayShadowParams) == 192, "RayShadowParams must match shadow_rt.slang");

VkDescriptorSetLayoutBinding storageBufferBinding(uint32_t index)
{
    VkDescriptorSetLayoutBinding b{};
    b.binding = index;
    b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    return b;
}

void bindWhole(DescriptorSet & set, VkDescriptorSet target, uint32_t index, const Buffer & buffer)
{
    set.bindBuffer(target, index, buffer.handle(), 0, buffer.size(),
                   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
}

} // namespace

ShadowPass::ShadowPass(Device & device, const std::string & shaderDir)
    : m_device(device), m_shaderDir(shaderDir), m_mapDescriptors(device.handle()),
      m_pcssDescriptors(device.handle()), m_rayDescriptors(device.handle()),
      m_upsampleDescriptors(device.handle())
{
    // A throw below (a missing shader file is the common one) skips the
    // destructor, so the raw handles created so far are released here.
    try
    {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        poolInfo.queueFamilyIndex = device.submitQueueFamily();
        VK_CHECK(vkCreateCommandPool(device.handle(), &poolInfo, nullptr, &m_commandPool),
                 "Failed to create ShadowPass command pool");
        m_params = std::make_unique<Buffer>(m_device, sizeof(ShadowParams));
        createPipelines(shaderDir);
    }
    catch (...)
    {
        release();
        throw;
    }
}

ShadowPass::~ShadowPass()
{
    release();
}

void ShadowPass::release()
{
    VkDevice dev = m_device.handle();
    m_mapPipeline.destroy(dev);
    m_pcssPipeline.destroy(dev);
    m_rayPipeline.destroy(dev);
    m_upsamplePipeline.destroy(dev);
    if (m_commandPool != VK_NULL_HANDLE)
    {
        vkDestroyCommandPool(dev, m_commandPool, nullptr);
    }
}

void ShadowPass::createPipelines(const std::string & shaderDir)
{
    VkDevice dev = m_device.handle();
    ShaderCompiler compiler(dev);
    const auto build = [&](DescriptorSet & descriptors, VkDescriptorSetLayout & layout,
                           VkDescriptorSet & set, Pipeline & pipeline, const char * kernel,
                           uint32_t buffers, uint32_t pushSize)
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        for (uint32_t i = 0; i < buffers; ++i)
        {
            bindings.push_back(storageBufferBinding(i));
        }
        layout = descriptors.createLayout(bindings);
        descriptors.createPool(1, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, buffers}});
        set = descriptors.allocate(layout);
        auto spirv = compiler.loadSPIRV(shaderDir + "/" + kernel + ".comp.spv");
        VkShaderModule module = compiler.createShaderModule(spirv);
        VkPipelineLayout pipelineLayout =
            pipeline.createPipelineLayout(dev, layout, pushSize, VK_SHADER_STAGE_COMPUTE_BIT);
        pipeline.createComputePipeline(dev, module, pipelineLayout);
        compiler.destroyShaderModule(module);
    };
    // shadow_map: projected, sortedValues, tileRanges, shadowMap(RW); camera pushed.
    build(m_mapDescriptors, m_mapLayout, m_mapSet, m_mapPipeline, "shadow_map", 4,
          sizeof(CameraUniforms));
    // shadow_pcss: gNormalDepth, gMetalAmbientCoverage, shadowMap, params, visibility(RW).
    build(m_pcssDescriptors, m_pcssLayout, m_pcssSet, m_pcssPipeline, "shadow_pcss", 5, 0);
}

void ShadowPass::runRayTraced(const RelightPass & relight, const CameraUniforms & receiverCamera,
                              const CanonicalGaussianBufferViews & scene,
                              const GaussianAccelerationStructure & accelerationStructure,
                              const Light & light, uint32_t layer, const ShadowSettings & settings,
                              float focusRadius, uint32_t firstSample, uint32_t sampleCount)
{
    VK_VALIDATE(m_device.features().rayQuery,
                "ShadowPass::runRayTraced: ray-traced shadows need ray query "
                "(DeviceFeatures::rayQuery); this device has none enabled -- use ShadowMode::PCSS");
    VK_VALIDATE(layer < kMaxLights, "ShadowPass::runRayTraced: layer beyond kMaxLights");
    VK_VALIDATE(focusRadius > 0.0f, "ShadowPass::runRayTraced: focus radius must be > 0");
    VkDevice dev = m_device.handle();

    if (!m_rayParams)
    {
        m_rayParams = std::make_unique<Buffer>(m_device, sizeof(RayShadowParams));

        ShaderCompiler compiler(dev);
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        for (uint32_t i = 0; i < 11; ++i)
        {
            VkDescriptorSetLayoutBinding b = storageBufferBinding(i);
            if (i == 0)
            {
                b.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            }
            bindings.push_back(b);
        }
        m_rayLayout = m_rayDescriptors.createLayout(bindings);
        m_rayDescriptors.createPool(1, {{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
                                        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10}});
        m_raySet = m_rayDescriptors.allocate(m_rayLayout);
        auto spirv = compiler.loadSPIRV(m_shaderDir + "/shadow_rt.comp.spv");
        VkShaderModule module = compiler.createShaderModule(spirv);
        VkPipelineLayout layout =
            m_rayPipeline.createPipelineLayout(dev, m_rayLayout, 0, VK_SHADER_STAGE_COMPUTE_BIT);
        m_rayPipeline.createComputePipeline(dev, module, layout);
        compiler.destroyShaderModule(module);

        // shadow_upsample: gNormalDepth, gMetalAmbientCoverage, lowRes, visibility(RW).
        std::vector<VkDescriptorSetLayoutBinding> upBindings;
        for (uint32_t i = 0; i < 4; ++i)
        {
            upBindings.push_back(storageBufferBinding(i));
        }
        m_upsampleLayout = m_upsampleDescriptors.createLayout(upBindings);
        m_upsampleDescriptors.createPool(1, {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4}});
        m_upsampleSet = m_upsampleDescriptors.allocate(m_upsampleLayout);
        auto upSpirv = compiler.loadSPIRV(m_shaderDir + "/shadow_upsample.comp.spv");
        VkShaderModule upModule = compiler.createShaderModule(upSpirv);
        VkPipelineLayout upLayout = m_upsamplePipeline.createPipelineLayout(
            dev, m_upsampleLayout, sizeof(UpsampleParams), VK_SHADER_STAGE_COMPUTE_BIT);
        m_upsamplePipeline.createComputePipeline(dev, upModule, upLayout);
        compiler.destroyShaderModule(upModule);
    }
    // Sized for full resolution, which also covers any downsampled grid.
    const VkDeviceSize layerBytes =
        static_cast<VkDeviceSize>(kMaxLights) * relight.width() * relight.height() * sizeof(float);
    if (!m_rayAccum || m_rayAccum->size() != layerBytes)
    {
        VK_VALIDATE(firstSample == 0, "ShadowPass::runRayTraced: the image size changed mid-accumulation");
        m_rayAccum = std::make_unique<Buffer>(m_device, layerBytes);
        m_rayLowRes = std::make_unique<Buffer>(m_device, layerBytes);
    }

    RayShadowParams params{};
    params.receiverCameraToWorld = glm::inverse(receiverCamera.worldToCamera);
    params.receiverIntrinsics =
        glm::vec4(receiverCamera.fx, receiverCamera.fy, receiverCamera.cx, receiverCamera.cy);
    params.lightPositionType = glm::vec4(light.position, static_cast<float>(static_cast<uint32_t>(light.type)));
    params.lightDirection = glm::vec4(light.direction, 0.0f);
    params.lightAxisX = glm::vec4(glm::normalize(light.axisX) * light.halfWidth, 0.0f);
    params.lightAxisY = glm::vec4(glm::normalize(light.axisY) * light.halfHeight, 0.0f);
    params.width = relight.width();
    params.height = relight.height();
    params.layer = layer;
    params.samplesPerAxis = std::max(settings.raySamplesPerAxis, 1u);
    params.lightSize = light.type == LightType::Directional ? std::tan(settings.sunAngularRadius)
                                                            : settings.pointLightRadius * focusRadius;
    params.normalOffset = settings.normalOffset * focusRadius;
    params.rayLength = 8.0f * focusRadius;
    const uint32_t gridSamples = params.samplesPerAxis * params.samplesPerAxis;
    VK_VALIDATE(firstSample <= gridSamples, "ShadowPass::runRayTraced: firstSample beyond the emitter grid");
    params.firstSample = firstSample;
    params.sampleCount = std::min(sampleCount, gridSamples - firstSample);
    VK_VALIDATE(params.firstSample + params.sampleCount > 0,
                "ShadowPass::runRayTraced: nothing traced or accumulated yet");
    VK_VALIDATE(settings.rayMinAlpha >= 1.0f / 255.0f && settings.rayMinAlpha < 1.0f
                    && settings.rayMinTransmittance >= 0.0f && settings.rayMinTransmittance < 1.0f
                    && settings.rayDownsample >= 1 && settings.rayDownsample <= 8,
                "ShadowPass::runRayTraced: rayMinAlpha must be in [1/255, 1), rayMinTransmittance in "
                "[0, 1), rayDownsample in [1, 8]");
    VK_VALIDATE(settings.rayMinAlpha == accelerationStructure.minAlpha(),
                "ShadowPass::runRayTraced: the acceleration structure was built for another rayMinAlpha");
    params.minAlpha = settings.rayMinAlpha;
    params.minTransmittance = settings.rayMinTransmittance;
    params.downsample = settings.rayDownsample;
    const uint32_t lowWidth = (params.width + params.downsample - 1) / params.downsample;
    const uint32_t lowHeight = (params.height + params.downsample - 1) / params.downsample;

    m_rayDescriptors.bindAccelerationStructure(m_raySet, 0, accelerationStructure.tlas());
    const auto bindView = [&](uint32_t index, const BufferView & v)
    {
        m_rayDescriptors.bindBuffer(m_raySet, index, v.buffer, v.offset, v.size,
                                    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    };
    bindView(1, scene.means);
    bindView(2, scene.scales);
    bindView(3, scene.rotations);
    bindView(4, scene.opacity);
    bindWhole(m_rayDescriptors, m_raySet, 5, relight.gNormalDepth());
    bindWhole(m_rayDescriptors, m_raySet, 6, relight.gMetalAmbientCoverage());
    bindWhole(m_rayDescriptors, m_raySet, 7, *m_rayParams);
    bindWhole(m_rayDescriptors, m_raySet, 8, relight.visibility());
    bindWhole(m_rayDescriptors, m_raySet, 9, *m_rayAccum);
    bindWhole(m_rayDescriptors, m_raySet, 10, *m_rayLowRes);
    if (params.downsample > 1)
    {
        bindWhole(m_upsampleDescriptors, m_upsampleSet, 0, relight.gNormalDepth());
        bindWhole(m_upsampleDescriptors, m_upsampleSet, 1, relight.gMetalAmbientCoverage());
        bindWhole(m_upsampleDescriptors, m_upsampleSet, 2, *m_rayLowRes);
        bindWhole(m_upsampleDescriptors, m_upsampleSet, 3, relight.visibility());
    }

    VK_CHECK(vkResetCommandPool(dev, m_commandPool, 0), "Failed to reset ShadowPass command pool");
    CommandBuffer cmd(dev, m_commandPool);
    cmd.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VkCommandBuffer handle = cmd.getHandle();
    Device::recordGlobalBarrier(handle); // see Device: queue contract
    vkCmdUpdateBuffer(handle, m_rayParams->handle(), 0, sizeof(params), &params);
    Device::recordGlobalBarrier(handle); // params -> kernel
    cmd.dispatchCompute(m_rayPipeline.getHandle(), m_rayPipeline.getLayout(), m_raySet,
                        dispatchGroups(lowWidth, kTileSize), dispatchGroups(lowHeight, kTileSize), 1);
    if (params.downsample > 1)
    {
        Device::recordGlobalBarrier(handle); // low-res receivers -> upsample
        const UpsampleParams up{params.width, params.height, layer, params.downsample};
        cmd.pushConstants(m_upsamplePipeline.getLayout(), VK_SHADER_STAGE_COMPUTE_BIT, sizeof(up), &up);
        cmd.dispatchCompute(m_upsamplePipeline.getHandle(), m_upsamplePipeline.getLayout(), m_upsampleSet,
                            dispatchGroups(params.width, kTileSize), dispatchGroups(params.height, kTileSize), 1);
    }
    Device::recordGlobalBarrier(handle); // see Device: queue contract
    cmd.end();

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence;
    VK_CHECK(vkCreateFence(dev, &fenceInfo, nullptr, &fence), "Failed to create fence");
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &handle;
    VK_CHECK(vkQueueSubmit(m_device.submitQueue(), 1, &submitInfo, fence),
             "Failed to submit ShadowPass ray tracing");
    VK_CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX),
             "Failed to wait for ShadowPass ray tracing");
    vkDestroyFence(dev, fence, nullptr);
}

CameraUniforms ShadowPass::lightCameraFor(const Light & light, uint32_t resolution,
                                          const glm::vec3 & focusCenter, float focusRadius,
                                          uint32_t gaussianCount, bool antialiased)
{
    // Local lights look from the emitter at the focus; a directional light
    // from a point well outside the focus sphere, against its travel. A
    // perspective camera either way -- the rasterizer is perspective-only,
    // and at 3 radii a directional light's frustum is close to parallel.
    glm::vec3 eye;
    if (light.type == LightType::Directional)
    {
        eye = focusCenter - glm::normalize(light.direction) * (3.0f * focusRadius);
    }
    else
    {
        eye = light.position;
    }
    glm::vec3 forward = focusCenter - eye;
    float distance = glm::length(forward);
    if (distance < 1e-6f)
    {
        forward = glm::vec3(0.0f, -1.0f, 0.0f);
        distance = 1e-6f;
    }
    forward /= distance;
    const glm::vec3 up = std::fabs(forward.y) < 0.99f ? glm::vec3(0.0f, 1.0f, 0.0f)
                                                      : glm::vec3(1.0f, 0.0f, 0.0f);
    // The half-angle that just contains the focus sphere (60 deg when the
    // light is inside it -- such a light cannot see its whole scene anyway).
    const float halfAngle = distance > focusRadius * 1.0001f
                                ? std::asin(std::min(focusRadius / distance, 0.999f)) * 1.05f
                                : 1.0471976f;

    CameraUniforms c{};
    c.worldToCamera = glm::lookAt(eye, focusCenter, up);
    c.fx = c.fy = 0.5f * static_cast<float>(resolution) / std::tan(std::min(halfAngle, 1.3f));
    c.cx = c.cy = 0.5f * static_cast<float>(resolution - 1);
    c.imageWidth = c.imageHeight = resolution;
    c.tilesX = c.tilesY = dispatchGroups(resolution, kTileSize);
    c.gaussianCount = gaussianCount;
    c.shDegree = 0; // color is irrelevant; depth and opacity are not
    c.renderFlags = antialiased ? kRenderFlagAntialiased : 0u;
    c.cameraPositionWorld = glm::vec4(eye, 0.0f);
    return c;
}

void ShadowPass::runPCSS(const RelightPass & relight, const CameraUniforms & receiverCamera,
                         const CanonicalGaussianBufferViews & scene, bool antialiased,
                         const Light & light, uint32_t layer, const ShadowSettings & settings,
                         const glm::vec3 & focusCenter, float focusRadius)
{
    VK_VALIDATE(layer < kMaxLights, "ShadowPass::runPCSS: layer beyond kMaxLights");
    VK_VALIDATE(focusRadius > 0.0f, "ShadowPass::runPCSS: focus radius must be > 0");
    const uint32_t res = std::max(settings.mapResolution, kTileSize);

    // 1. The light's view of the splats.
    if (!m_lightForward)
    {
        m_lightForward = std::make_unique<ForwardPass>(m_device, res, res, m_shaderDir);
    }
    else if (res != m_mapResolution)
    {
        m_lightForward->resize(res, res);
    }
    if (res != m_mapResolution || !m_shadowMap)
    {
        m_shadowMap = std::make_unique<Buffer>(
            m_device, static_cast<VkDeviceSize>(res) * res * 2 * sizeof(float));
        m_mapResolution = res;
    }
    m_lightCamera = lightCameraFor(light, res, focusCenter, focusRadius, scene.gaussianCount,
                                   antialiased);
    m_lightForward->run(scene, m_lightCamera, ForwardStage::Rasterize, ForwardMode::Viewer);

    // 2 + 3. Shadow map, then PCSS into the visibility layer.
    ShadowParams params{};
    params.receiverCameraToWorld = glm::inverse(receiverCamera.worldToCamera);
    params.lightWorldToCamera = m_lightCamera.worldToCamera;
    params.receiverIntrinsics =
        glm::vec4(receiverCamera.fx, receiverCamera.fy, receiverCamera.cx, receiverCamera.cy);
    params.lightIntrinsics =
        glm::vec4(m_lightCamera.fx, m_lightCamera.fy, m_lightCamera.cx, m_lightCamera.cy);
    params.width = relight.width();
    params.height = relight.height();
    params.mapWidth = params.mapHeight = res;
    params.layer = layer;
    params.blockerSamplesPerAxis = settings.blockerSamplesPerAxis;
    params.filterSamplesPerAxis = settings.filterSamplesPerAxis;
    params.directional = light.type == LightType::Directional ? 1u : 0u;
    switch (light.type)
    {
    case LightType::Directional:
        params.lightSize = std::tan(settings.sunAngularRadius);
        break;
    case LightType::Point:
        params.lightSize = settings.pointLightRadius * focusRadius;
        break;
    case LightType::Rect:
        params.lightSize = std::max(light.halfWidth, light.halfHeight);
        break;
    }
    params.depthBias = settings.depthBias * focusRadius;
    params.normalOffset = settings.normalOffset * focusRadius;
    params.maxFilterTexels = settings.maxFilterTexels;

    bindWhole(m_mapDescriptors, m_mapSet, 0, m_lightForward->projected());
    bindWhole(m_mapDescriptors, m_mapSet, 1, m_lightForward->sortedValues());
    bindWhole(m_mapDescriptors, m_mapSet, 2, m_lightForward->tileRanges());
    bindWhole(m_mapDescriptors, m_mapSet, 3, *m_shadowMap);
    bindWhole(m_pcssDescriptors, m_pcssSet, 0, relight.gNormalDepth());
    bindWhole(m_pcssDescriptors, m_pcssSet, 1, relight.gMetalAmbientCoverage());
    bindWhole(m_pcssDescriptors, m_pcssSet, 2, *m_shadowMap);
    bindWhole(m_pcssDescriptors, m_pcssSet, 3, *m_params);
    bindWhole(m_pcssDescriptors, m_pcssSet, 4, relight.visibility());

    VkDevice dev = m_device.handle();
    VK_CHECK(vkResetCommandPool(dev, m_commandPool, 0), "Failed to reset ShadowPass command pool");
    CommandBuffer cmd(dev, m_commandPool);
    cmd.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VkCommandBuffer handle = cmd.getHandle();
    Device::recordGlobalBarrier(handle); // see Device: queue contract

    vkCmdUpdateBuffer(handle, m_params->handle(), 0, sizeof(params), &params);
    cmd.pushConstants(m_mapPipeline.getLayout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      sizeof(CameraUniforms), &m_lightCamera);
    cmd.dispatchCompute(m_mapPipeline.getHandle(), m_mapPipeline.getLayout(), m_mapSet,
                        m_lightCamera.tilesX, m_lightCamera.tilesY, 1);
    Device::recordGlobalBarrier(handle); // shadow map + params -> PCSS
    cmd.dispatchCompute(m_pcssPipeline.getHandle(), m_pcssPipeline.getLayout(), m_pcssSet,
                        dispatchGroups(params.width, kTileSize),
                        dispatchGroups(params.height, kTileSize), 1);

    Device::recordGlobalBarrier(handle); // see Device: queue contract
    cmd.end();

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence;
    VK_CHECK(vkCreateFence(dev, &fenceInfo, nullptr, &fence), "Failed to create fence");
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &handle;
    VK_CHECK(vkQueueSubmit(m_device.submitQueue(), 1, &submitInfo, fence),
             "Failed to submit ShadowPass");
    VK_CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX), "Failed to wait for ShadowPass");
    vkDestroyFence(dev, fence, nullptr);
}

} // namespace sabercat
