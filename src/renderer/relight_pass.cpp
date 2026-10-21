// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/relight_pass.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "sabercat/buffer.h"
#include "sabercat/camera.h"
#include "sabercat/command_buffer.h"
#include "sabercat/device.h"
#include "sabercat/forward_pass.h"
#include "sabercat/image.h"
#include "sabercat/render_config.h"
#include "sabercat/shader_compiler.h"
#include "sabercat/vk_check.h"

namespace sabercat
{

namespace
{

// Host mirrors of relight_common.slangh / relight_prepare.slang (std430).
struct GpuLight
{
    glm::vec4 positionType;
    glm::vec4 direction;
    glm::vec4 axisXShadow;
    glm::vec4 axisY;
    glm::vec4 radiance;
};
static_assert(sizeof(GpuLight) == 80, "GpuLight must match relight_common.slangh");

struct GpuSurfel
{
    glm::vec4 normalMetallic;
    glm::vec4 baseColorRoughness;
    glm::vec4 ambient;
};
static_assert(sizeof(GpuSurfel) == 48, "Surfel must match relight_common.slangh");

struct LightingParams
{
    glm::mat4 cameraToWorld;
    glm::vec4 intrinsics;
    uint32_t width;
    uint32_t height;
    uint32_t lightCount;
    uint32_t output;
    uint32_t rectSamplesPerAxis;
    uint32_t visibilityLight;
    uint32_t hasEnvironment;
    uint32_t pad0;
    glm::vec4 backgroundExposure;
    glm::vec4 cameraPosition;
    glm::vec4 environment[9];
    glm::vec4 depthScale;
};
static_assert(sizeof(LightingParams) == 304, "LightingParams must match relight_common.slangh");

struct PrepareParams
{
    glm::vec4 cameraPosition;
    uint32_t count;
    uint32_t flags;
    uint32_t pad0;
    uint32_t pad1;
};
static_assert(sizeof(PrepareParams) == 32, "PrepareParams must match relight_prepare.slang");

constexpr uint32_t kHasNormals = 1u;
constexpr uint32_t kHasBaseColor = 2u;
constexpr uint32_t kHasMetallic = 4u;
constexpr uint32_t kHasVisibility = 8u;

VkDescriptorSetLayoutBinding binding(uint32_t index, VkDescriptorType type)
{
    VkDescriptorSetLayoutBinding b{};
    b.binding = index;
    b.descriptorType = type;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    return b;
}

void bindView(DescriptorSet & set, VkDescriptorSet target, uint32_t index, const BufferView & view)
{
    set.bindBuffer(target, index, view.buffer, view.offset, view.size,
                   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
}

void bindWhole(DescriptorSet & set, VkDescriptorSet target, uint32_t index, const Buffer & buffer)
{
    set.bindBuffer(target, index, buffer.handle(), 0, buffer.size(),
                   VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
}

GpuLight packLight(const Light & l)
{
    GpuLight g{};
    g.positionType = glm::vec4(l.position, static_cast<float>(static_cast<uint32_t>(l.type)));
    g.direction = glm::vec4(l.direction, 0.0f);
    g.axisXShadow = glm::vec4(glm::normalize(l.axisX) * l.halfWidth, l.castsShadows ? 1.0f : 0.0f);
    g.axisY = glm::vec4(glm::normalize(l.axisY) * l.halfHeight, 0.0f);
    g.radiance = glm::vec4(l.color * l.intensity, 0.0f);
    return g;
}

} // namespace

RelightPass::RelightPass(Device & device, uint32_t width, uint32_t height,
                         const std::string & shaderDir)
    : m_device(device), m_width(width), m_height(height),
      m_prepareDescriptors(device.handle()), m_gbufferDescriptors(device.handle()),
      m_lightingDescriptors(device.handle())
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
                 "Failed to create RelightPass command pool");

        m_dummy = std::make_unique<Buffer>(m_device, 64);
        m_lights = std::make_unique<Buffer>(m_device, kMaxLights * sizeof(GpuLight));
        m_params = std::make_unique<Buffer>(m_device, sizeof(LightingParams));
        createPipelines(shaderDir);
        allocatePixelBuffers();
    }
    catch (...)
    {
        release();
        throw;
    }
}

RelightPass::~RelightPass()
{
    release();
}

void RelightPass::release()
{
    VkDevice dev = m_device.handle();
    m_preparePipeline.destroy(dev);
    m_gbufferPipeline.destroy(dev);
    m_lightingPipeline.destroy(dev);
    if (m_commandPool != VK_NULL_HANDLE)
    {
        vkDestroyCommandPool(dev, m_commandPool, nullptr);
    }
}

void RelightPass::createPipelines(const std::string & shaderDir)
{
    VkDevice dev = m_device.handle();
    ShaderCompiler compiler(dev);
    const auto build = [&](DescriptorSet & descriptors, VkDescriptorSetLayout & layout,
                           VkDescriptorSet & set, Pipeline & pipeline, const char * kernel,
                           uint32_t buffers, uint32_t images, uint32_t pushSize)
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        for (uint32_t i = 0; i < buffers; ++i)
        {
            bindings.push_back(binding(i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER));
        }
        for (uint32_t i = 0; i < images; ++i)
        {
            bindings.push_back(binding(buffers + i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE));
        }
        layout = descriptors.createLayout(bindings);
        std::vector<VkDescriptorPoolSize> sizes = {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, buffers}};
        if (images > 0)
        {
            sizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, images});
        }
        descriptors.createPool(1, sizes);
        set = descriptors.allocate(layout);

        auto spirv = compiler.loadSPIRV(shaderDir + "/" + kernel + ".comp.spv");
        VkShaderModule module = compiler.createShaderModule(spirv);
        VkPipelineLayout pipelineLayout =
            pipeline.createPipelineLayout(dev, layout, pushSize, VK_SHADER_STAGE_COMPUTE_BIT);
        pipeline.createComputePipeline(dev, module, pipelineLayout);
        compiler.destroyShaderModule(module);
    };

    // relight_prepare: means, scales, rotations, sh, normals, baseColorRoughness,
    // metallic, visibility, surfels(RW); PrepareParams pushed.
    build(m_prepareDescriptors, m_prepareLayout, m_prepareSet, m_preparePipeline,
          "relight_prepare", 9, 0, sizeof(PrepareParams));
    // gbuffer_rasterize: projected, sortedValues, tileRanges, surfels, three
    // G-buffer outputs(RW); camera pushed.
    build(m_gbufferDescriptors, m_gbufferLayout, m_gbufferSet, m_gbufferPipeline,
          "gbuffer_rasterize", 7, 0, sizeof(CameraUniforms));
    // deferred_lighting: three G-buffer inputs, lights, params, visibility,
    // litBuffer(RW); litImage (storage image, binding 7).
    build(m_lightingDescriptors, m_lightingLayout, m_lightingSet, m_lightingPipeline,
          "deferred_lighting", 7, 1, 0);
}

void RelightPass::allocatePixelBuffers()
{
    const VkDeviceSize pixels = static_cast<VkDeviceSize>(m_width) * m_height;
    m_gNormalDepth = std::make_unique<Buffer>(m_device, pixels * sizeof(glm::vec4));
    m_gBaseRoughness = std::make_unique<Buffer>(m_device, pixels * sizeof(glm::vec4));
    m_gMetalAmbientCoverage = std::make_unique<Buffer>(m_device, pixels * sizeof(glm::vec4));
    m_visibility = std::make_unique<Buffer>(m_device, pixels * kMaxLights * sizeof(float));
    m_litBuffer = std::make_unique<Buffer>(m_device, pixels * sizeof(glm::vec4));
    m_litImage = std::make_unique<Image>(m_device, m_width, m_height, VK_FORMAT_R32G32B32A32_SFLOAT,
                                         VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    // All ones until a shadow source writes it: no shadows rather than
    // undefined ones if lighting runs before anything resets it.
    m_device.immediateSubmit(
        [this](VkCommandBuffer cmd)
        { vkCmdFillBuffer(cmd, m_visibility->handle(), 0, VK_WHOLE_SIZE, 0x3F800000u); });

    bindWhole(m_gbufferDescriptors, m_gbufferSet, 4, *m_gNormalDepth);
    bindWhole(m_gbufferDescriptors, m_gbufferSet, 5, *m_gBaseRoughness);
    bindWhole(m_gbufferDescriptors, m_gbufferSet, 6, *m_gMetalAmbientCoverage);
    bindWhole(m_lightingDescriptors, m_lightingSet, 0, *m_gNormalDepth);
    bindWhole(m_lightingDescriptors, m_lightingSet, 1, *m_gBaseRoughness);
    bindWhole(m_lightingDescriptors, m_lightingSet, 2, *m_gMetalAmbientCoverage);
    bindWhole(m_lightingDescriptors, m_lightingSet, 3, *m_lights);
    bindWhole(m_lightingDescriptors, m_lightingSet, 4, *m_params);
    bindWhole(m_lightingDescriptors, m_lightingSet, 5, *m_visibility);
    bindWhole(m_lightingDescriptors, m_lightingSet, 6, *m_litBuffer);
    m_lightingDescriptors.bindImage(m_lightingSet, 7, m_litImage->getImageView(), VK_NULL_HANDLE,
                                    VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
}

void RelightPass::resize(uint32_t width, uint32_t height)
{
    m_width = width;
    m_height = height;
    allocatePixelBuffers();
}

void RelightPass::submitAndWait(VkCommandBuffer handle, const char * what)
{
    VkDevice dev = m_device.handle();
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence;
    VK_CHECK(vkCreateFence(dev, &fenceInfo, nullptr, &fence), "Failed to create fence");
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &handle;
    VK_CHECK(vkQueueSubmit(m_device.submitQueue(), 1, &submitInfo, fence), what);
    VK_CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX), what);
    vkDestroyFence(dev, fence, nullptr);
}

void RelightPass::runGBuffer(const ForwardPass & forward, const RelightBufferViews & relight)
{
    const ForwardRunRecord & last = forward.lastRun();
    VK_VALIDATE(last.valid && last.completed,
                "RelightPass::runGBuffer: the last forward run did not complete");
    const CameraUniforms & camera = last.camera;
    VK_VALIDATE(camera.imageWidth == m_width && camera.imageHeight == m_height,
                "RelightPass::runGBuffer: forward image size differs from the relight pass");
    const CanonicalGaussianBufferViews & params = last.views;
    const uint32_t count = params.gaussianCount;

    if (count > m_surfelCapacity || !m_surfels)
    {
        m_surfelCapacity = std::max(count, 1u);
        m_surfels = std::make_unique<Buffer>(m_device, m_surfelCapacity * sizeof(GpuSurfel));
    }

    // Bound fresh every run (run waits for completion, so no pending use).
    bindView(m_prepareDescriptors, m_prepareSet, 0, params.means);
    bindView(m_prepareDescriptors, m_prepareSet, 1, params.scales);
    bindView(m_prepareDescriptors, m_prepareSet, 2, params.rotations);
    bindView(m_prepareDescriptors, m_prepareSet, 3, params.shCoefficients);
    PrepareParams prepare{};
    prepare.cameraPosition = camera.cameraPositionWorld;
    prepare.count = count;
    const auto optional = [&](uint32_t index, const BufferView & view, uint32_t flag)
    {
        if (view.valid())
        {
            bindView(m_prepareDescriptors, m_prepareSet, index, view);
            prepare.flags |= flag;
        }
        else
        {
            bindWhole(m_prepareDescriptors, m_prepareSet, index, *m_dummy);
        }
    };
    optional(4, relight.normals, kHasNormals);
    optional(5, relight.baseColorRoughness, kHasBaseColor);
    optional(6, relight.metallic, kHasMetallic);
    optional(7, relight.visibility, kHasVisibility);
    bindWhole(m_prepareDescriptors, m_prepareSet, 8, *m_surfels);

    bindWhole(m_gbufferDescriptors, m_gbufferSet, 0, forward.projected());
    bindWhole(m_gbufferDescriptors, m_gbufferSet, 1, forward.sortedValues());
    bindWhole(m_gbufferDescriptors, m_gbufferSet, 2, forward.tileRanges());
    bindWhole(m_gbufferDescriptors, m_gbufferSet, 3, *m_surfels);

    VkDevice dev = m_device.handle();
    VK_CHECK(vkResetCommandPool(dev, m_commandPool, 0), "Failed to reset RelightPass command pool");
    CommandBuffer cmd(dev, m_commandPool);
    cmd.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    Device::recordGlobalBarrier(cmd.getHandle()); // see Device: queue contract

    if (count > 0)
    {
        cmd.pushConstants(m_preparePipeline.getLayout(), VK_SHADER_STAGE_COMPUTE_BIT,
                          sizeof(prepare), &prepare);
        cmd.dispatchCompute(m_preparePipeline.getHandle(), m_preparePipeline.getLayout(),
                            m_prepareSet, dispatchGroups(count, kWorkgroupSize), 1, 1);
        cmd.bufferBarrier(m_surfels->handle(), VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    }
    cmd.pushConstants(m_gbufferPipeline.getLayout(), VK_SHADER_STAGE_COMPUTE_BIT,
                      sizeof(CameraUniforms), &camera);
    cmd.dispatchCompute(m_gbufferPipeline.getHandle(), m_gbufferPipeline.getLayout(), m_gbufferSet,
                        camera.tilesX, camera.tilesY, 1);

    Device::recordGlobalBarrier(cmd.getHandle()); // see Device: queue contract
    cmd.end();
    submitAndWait(cmd.getHandle(), "Failed to run RelightPass G-buffer");
}

void RelightPass::runLighting(const ForwardPass & forward, const std::vector<Light> & lights,
                              const EnvironmentLight & environment,
                              const LightingSettings & settings, bool resetVisibility)
{
    const ForwardRunRecord & last = forward.lastRun();
    VK_VALIDATE(last.valid && last.completed,
                "RelightPass::runLighting: the last forward run did not complete");
    VK_VALIDATE(lights.size() <= kMaxLights,
                "RelightPass::runLighting: more than kMaxLights (" + std::to_string(kMaxLights)
                    + ") lights");
    const CameraUniforms & camera = last.camera;

    LightingParams params{};
    params.cameraToWorld = glm::inverse(camera.worldToCamera);
    params.intrinsics = glm::vec4(camera.fx, camera.fy, camera.cx, camera.cy);
    params.width = m_width;
    params.height = m_height;
    params.lightCount = static_cast<uint32_t>(lights.size());
    params.output = static_cast<uint32_t>(settings.output);
    params.rectSamplesPerAxis = std::max(
        1u, static_cast<uint32_t>(std::lround(std::sqrt(static_cast<double>(settings.rectSamples)))));
    params.visibilityLight = settings.visibilityLight;
    params.backgroundExposure = glm::vec4(settings.background, settings.exposure);
    params.cameraPosition = camera.cameraPositionWorld;
    params.depthScale = glm::vec4(settings.depthScale, 0.0f, 0.0f, 0.0f);
    for (size_t k = 0; k < 9; ++k)
    {
        params.environment[k] = glm::vec4(environment.sh[k] * environment.intensity, 0.0f);
        params.hasEnvironment |= (environment.sh[k] != glm::vec3(0.0f)) ? 1u : 0u;
    }
    GpuLight packed[kMaxLights]{};
    for (size_t i = 0; i < lights.size(); ++i)
    {
        packed[i] = packLight(lights[i]);
    }

    VkDevice dev = m_device.handle();
    VK_CHECK(vkResetCommandPool(dev, m_commandPool, 0), "Failed to reset RelightPass command pool");
    CommandBuffer cmd(dev, m_commandPool);
    cmd.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    Device::recordGlobalBarrier(cmd.getHandle()); // see Device: queue contract
    VkCommandBuffer handle = cmd.getHandle();

    vkCmdUpdateBuffer(handle, m_params->handle(), 0, sizeof(params), &params);
    vkCmdUpdateBuffer(handle, m_lights->handle(), 0, sizeof(packed), packed);
    if (resetVisibility)
    {
        vkCmdFillBuffer(handle, m_visibility->handle(), 0, VK_WHOLE_SIZE, 0x3F800000u);
    }
    // Transfer writes above -> the lighting kernel's reads.
    Device::recordGlobalBarrier(handle);

    m_litImage->transitionLayout(handle, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL);
    cmd.dispatchCompute(m_lightingPipeline.getHandle(), m_lightingPipeline.getLayout(),
                        m_lightingSet, dispatchGroups(m_width, kTileSize),
                        dispatchGroups(m_height, kTileSize), 1);

    Device::recordGlobalBarrier(handle); // see Device: queue contract
    cmd.end();
    submitAndWait(handle, "Failed to run RelightPass lighting");
}

void RelightPass::clearVisibility()
{
    m_device.immediateSubmit(
        [this](VkCommandBuffer cmd)
        { vkCmdFillBuffer(cmd, m_visibility->handle(), 0, VK_WHOLE_SIZE, 0x3F800000u); });
}

void RelightPass::prepareLitForRead(VkCommandBuffer cmd)
{
    m_litImage->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
}

} // namespace sabercat
