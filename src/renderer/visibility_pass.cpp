// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/visibility_pass.h"

#include <glm/glm.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <string>

#include "sabercat/buffer.h"
#include "sabercat/command_buffer.h"
#include "sabercat/device.h"
#include "sabercat/forward_pass.h"
#include "sabercat/gaussian_acceleration_structure.h"
#include "sabercat/logger.h"
#include "sabercat/relight_pass.h"
#include "sabercat/render_config.h"
#include "sabercat/shader_compiler.h"
#include "sabercat/vk_check.h"

namespace sabercat
{

namespace
{

// Host mirror of light_visibility.slangh's TraceLight (std430, 80 B).
struct TraceLight
{
    glm::vec4 positionType;
    glm::vec4 direction;
    glm::vec4 axisX;
    glm::vec4 axisY;
    glm::vec4 sizeLength;
};
static_assert(sizeof(TraceLight) == 80, "TraceLight must match light_visibility.slangh");

// Host mirror of teacher.slang's TeacherParams (std430, 160 B).
struct TeacherParams
{
    glm::mat4 receiverCameraToWorld;
    glm::vec4 receiverIntrinsics;
    uint32_t randomTarget;
    uint32_t edgeTarget;
    uint32_t samplesPerAxis;
    uint32_t seed;
    uint32_t shadowLightCount;
    uint32_t width;
    uint32_t height;
    uint32_t pad0;
    float normalOffset;
    float pad1;
    float pad2;
    float pad3;
    glm::uvec4 shadowLights[2];
};
static_assert(sizeof(TeacherParams) == 160, "TeacherParams must match teacher.slang");

// Push constants of bake_visibility / sample_select / visibility_debug.
struct BakeConstants
{
    uint32_t firstGaussian;
    uint32_t count;
    uint32_t directionCount;
    float rayLength;
};
static_assert(sizeof(BakeConstants) == 16, "BakeConstants must match bake_visibility.slang");

struct SelectConstants
{
    uint32_t phase;
    uint32_t width;
    uint32_t height;
    uint32_t pad0;
    float minCoverage;
    float edgeDepthRatio;
    float edgeNormalCos;
    float pad1;
};
static_assert(sizeof(SelectConstants) == 32, "SelectConstants must match sample_select.slang");

struct DebugConstants
{
    uint32_t mode;
    uint32_t width;
    uint32_t height;
    uint32_t sampleCount;
};
static_assert(sizeof(DebugConstants) == 16, "DebugConstants must match visibility_debug.slang");

// visibility_splat.slang's SplatParams.
struct SplatParams
{
    uint32_t lightMask;
    uint32_t lightCount;
    uint32_t pad0;
    uint32_t pad1;
};
static_assert(sizeof(SplatParams) == 16, "SplatParams must match visibility_splat.slang");

// Exactly the per-light values ShadowPass::runRayTraced packs, so a teacher
// target and the RayTraced shadow mode trace the same rays.
TraceLight traceLightFor(const Light & light, const ShadowSettings & shadows, float focusRadius)
{
    TraceLight t{};
    t.positionType = glm::vec4(light.position, static_cast<float>(static_cast<uint32_t>(light.type)));
    t.direction = glm::vec4(light.direction, 0.0f);
    t.axisX = glm::vec4(glm::normalize(light.axisX) * light.halfWidth, 0.0f);
    t.axisY = glm::vec4(glm::normalize(light.axisY) * light.halfHeight, 0.0f);
    const float lightSize = light.type == LightType::Directional ? std::tan(shadows.sunAngularRadius)
                                                                 : shadows.pointLightRadius * focusRadius;
    t.sizeLength = glm::vec4(lightSize, 8.0f * focusRadius, 0.0f, 0.0f);
    return t;
}

constexpr VkDescriptorType kStorage = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
constexpr VkDescriptorType kTlas = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;

} // namespace

VisibilityPass::VisibilityPass(Device & device, const std::string & shaderDir)
    : m_device(device), m_shaderDir(shaderDir), m_bake(device.handle()),
      m_select(device.handle()), m_teacher(device.handle()), m_splat(device.handle()),
      m_debug(device.handle())
{
    // Nothing after the pool can throw: pipelines and buffers are made on
    // first use, so a missing shader never leaks the pool.
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = device.submitQueueFamily();
    VK_CHECK(vkCreateCommandPool(device.handle(), &poolInfo, nullptr, &m_commandPool),
             "Failed to create VisibilityPass command pool");
}

VisibilityPass::~VisibilityPass()
{
    VkDevice dev = m_device.handle();
    for (Kernel * k : {&m_bake, &m_select, &m_teacher, &m_splat, &m_debug})
    {
        k->pipeline.destroy(dev);
    }
    if (m_commandPool != VK_NULL_HANDLE)
    {
        vkDestroyCommandPool(dev, m_commandPool, nullptr);
    }
}

void VisibilityPass::buildKernel(Kernel & kernel, const char * name,
                                 const std::vector<VkDescriptorType> & types, uint32_t pushSize)
{
    if (kernel.ready)
    {
        return;
    }
    VkDevice dev = m_device.handle();
    std::vector<VkDescriptorSetLayoutBinding> bindings;
    uint32_t storageCount = 0;
    uint32_t tlasCount = 0;
    for (uint32_t i = 0; i < types.size(); ++i)
    {
        VkDescriptorSetLayoutBinding b{};
        b.binding = i;
        b.descriptorType = types[i];
        b.descriptorCount = 1;
        b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindings.push_back(b);
        (types[i] == kTlas ? tlasCount : storageCount) += 1;
    }
    std::vector<VkDescriptorPoolSize> sizes;
    if (storageCount > 0)
    {
        sizes.push_back({kStorage, storageCount});
    }
    if (tlasCount > 0)
    {
        sizes.push_back({kTlas, tlasCount});
    }
    kernel.layout = kernel.descriptors.createLayout(bindings);
    kernel.descriptors.createPool(1, sizes);
    kernel.set = kernel.descriptors.allocate(kernel.layout);

    ShaderCompiler compiler(dev);
    auto spirv = compiler.loadSPIRV(m_shaderDir + "/" + name + ".comp.spv");
    VkShaderModule module = compiler.createShaderModule(spirv);
    VkPipelineLayout layout =
        kernel.pipeline.createPipelineLayout(dev, kernel.layout, pushSize, VK_SHADER_STAGE_COMPUTE_BIT);
    kernel.pipeline.createComputePipeline(dev, module, layout);
    compiler.destroyShaderModule(module);
    kernel.ready = true;
}

void VisibilityPass::bindView(Kernel & kernel, uint32_t binding, const BufferView & view)
{
    kernel.descriptors.bindBuffer(kernel.set, binding, view.buffer, view.offset, view.size, kStorage);
}

void VisibilityPass::bindWhole(Kernel & kernel, uint32_t binding, const Buffer & buffer)
{
    kernel.descriptors.bindBuffer(kernel.set, binding, buffer.handle(), 0, buffer.size(), kStorage);
}

void VisibilityPass::ensureBuffer(std::unique_ptr<Buffer> & buffer, VkDeviceSize bytes)
{
    bytes = std::max<VkDeviceSize>(bytes, 16);
    if (!buffer || buffer->size() < bytes)
    {
        buffer = std::make_unique<Buffer>(m_device, bytes);
    }
}

template <typename Record>
void VisibilityPass::submit(const char * what, Record && record)
{
    VkDevice dev = m_device.handle();
    VK_CHECK(vkResetCommandPool(dev, m_commandPool, 0), "Failed to reset VisibilityPass command pool");
    CommandBuffer cmd(dev, m_commandPool);
    cmd.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
    VkCommandBuffer handle = cmd.getHandle();
    Device::recordGlobalBarrier(handle); // see Device: queue contract
    record(cmd);
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
    const VkResult submitted = vkQueueSubmit(m_device.submitQueue(), 1, &submitInfo, fence);
    const VkResult waited =
        submitted == VK_SUCCESS ? vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX) : submitted;
    vkDestroyFence(dev, fence, nullptr);
    VK_CHECK(waited, std::string("VisibilityPass: ") + what + " failed");
}

void VisibilityPass::requireRayQuery() const
{
    VK_VALIDATE(m_device.features().rayQuery,
                "VisibilityPass: baking and teacher samples trace rays and need ray query "
                "(DeviceFeatures::rayQuery); this device has none enabled");
}

void VisibilityPass::bake(const CanonicalGaussianBufferViews & scene,
                          const GaussianAccelerationStructure & as, const BufferView & visibility,
                          const VisibilityBakeSettings & settings, float rayLength)
{
    VK_VALIDATE(scene.gaussianCount > 0, "VisibilityPass::bake: empty scene");
    VK_VALIDATE(settings.directions > 0, "VisibilityPass::bake: directions must be > 0");
    // A Gaussian's directions are never split across submissions, so a cap
    // below them could not be kept.
    VK_VALIDATE(settings.maxRaysPerSubmit >= settings.directions,
                "VisibilityPass::bake: maxRaysPerSubmit must be >= directions (one Gaussian's rays are one "
                "submission)");
    VK_VALIDATE(rayLength > 0.0f && std::isfinite(rayLength), "VisibilityPass::bake: ray length must be > 0");
    VK_VALIDATE(visibility.valid()
                    && visibility.size >= static_cast<VkDeviceSize>(scene.gaussianCount)
                                              * kVisibilityCoefficients * sizeof(float),
                "VisibilityPass::bake: the visibility view must hold " + std::to_string(kVisibilityCoefficients)
                    + " floats per Gaussian");
    requireRayQuery();
    buildKernel(m_bake, "bake_visibility", {kTlas, kStorage, kStorage, kStorage, kStorage, kStorage},
                sizeof(BakeConstants));

    m_bake.descriptors.bindAccelerationStructure(m_bake.set, 0, as.tlas());
    bindView(m_bake, 1, scene.means);
    bindView(m_bake, 2, scene.scales);
    bindView(m_bake, 3, scene.rotations);
    bindView(m_bake, 4, scene.opacity);
    bindView(m_bake, 5, visibility);

    // TDR guard: no submission traces more than maxRaysPerSubmit rays. With a
    // time target, each batch is resized from the previous one's host time
    // (at most 2x either way), so slow and fast scenes both stay near the
    // target. Each Gaussian's rays are independent, so the batching does not
    // change the result.
    VK_VALIDATE(std::isfinite(settings.targetSubmitMilliseconds) && settings.targetSubmitMilliseconds >= 0.0f,
                "VisibilityPass::bake: targetSubmitMilliseconds must be >= 0");
    const uint32_t maxGaussians =
        std::max(1u, std::max(settings.maxRaysPerSubmit, 1u) / settings.directions);
    const bool adaptive = settings.targetSubmitMilliseconds > 0.0f;
    // Adaptive batches never drop below 1M rays. One thread traces all of a
    // Gaussian's directions in sequence, so a submission's time has a floor
    // set by that serial latency: on a dense scene 3 Gaussians took ~95 ms and
    // 256 took ~250 ms. Shrinking below the floor only adds submissions, so the
    // time target governs growth above it, and the first batch starts there.
    const uint32_t minGaussians = std::min(maxGaussians, std::max(1u, (1u << 20) / settings.directions));
    uint32_t batch = adaptive ? minGaussians : maxGaussians;
    uint32_t submissions = 0;
    uint32_t lastBatch = 0;
    for (uint32_t first = 0; first < scene.gaussianCount;)
    {
        BakeConstants constants{};
        constants.firstGaussian = first;
        constants.count = std::min(batch, scene.gaussianCount - first);
        constants.directionCount = settings.directions;
        constants.rayLength = rayLength;
        const auto started = std::chrono::steady_clock::now();
        submit("visibility bake",
               [&](CommandBuffer & cmd)
               {
                   cmd.pushConstants(m_bake.pipeline.getLayout(), VK_SHADER_STAGE_COMPUTE_BIT,
                                     sizeof(constants), &constants);
                   cmd.dispatchCompute(m_bake.pipeline.getHandle(), m_bake.pipeline.getLayout(),
                                       m_bake.set, dispatchGroups(constants.count, kWorkgroupSize), 1, 1);
               });
        ++submissions;
        lastBatch = constants.count;
        first += constants.count;
        if (adaptive)
        {
            // submit() waits for the fence, so this is the submission's full time.
            const double elapsedMs = std::max(
                0.1, std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count());
            const double factor = std::clamp(settings.targetSubmitMilliseconds / elapsedMs, 0.5, 2.0);
            batch = static_cast<uint32_t>(std::clamp(std::round(batch * factor), static_cast<double>(minGaussians),
                                                     static_cast<double>(maxGaussians)));
        }
    }
    Logger::info("VisibilityPass::bake: " + std::to_string(scene.gaussianCount) + " Gaussians x "
                 + std::to_string(settings.directions) + " directions in " + std::to_string(submissions)
                 + " submission(s), last batch " + std::to_string(lastBatch) + " Gaussians"
                 + (adaptive ? " (adaptive)" : " (fixed)"));
}

uint32_t VisibilityPass::generateTeacherSamples(const RelightPass & relight,
                                                const CameraUniforms & receiverCamera,
                                                const CanonicalGaussianBufferViews & scene,
                                                const GaussianAccelerationStructure & as,
                                                const std::vector<Light> & lights,
                                                const TeacherSettings & settings,
                                                const ShadowSettings & shadows, float focusRadius)
{
    VK_VALIDATE(lights.size() <= kMaxLights, "VisibilityPass::generateTeacherSamples: too many lights");
    VK_VALIDATE(focusRadius > 0.0f, "VisibilityPass::generateTeacherSamples: focus radius must be > 0");
    const uint32_t total = settings.randomSamples + settings.edgeSamples;
    VK_VALIDATE(total > 0 && settings.randomSamples <= (1u << 24) && settings.edgeSamples <= (1u << 24),
                "VisibilityPass::generateTeacherSamples: sample counts must be 1..2^24");

    TeacherParams params{};
    for (uint32_t i = 0; i < lights.size(); ++i)
    {
        if (lights[i].castsShadows != 0)
        {
            params.shadowLights[params.shadowLightCount / 4][params.shadowLightCount % 4] = i;
            ++params.shadowLightCount;
        }
    }
    VK_VALIDATE(params.shadowLightCount > 0,
                "VisibilityPass::generateTeacherSamples: no light has castsShadows set");
    requireRayQuery();
    buildKernel(m_select, "sample_select", std::vector<VkDescriptorType>(6, kStorage),
                sizeof(SelectConstants));
    buildKernel(m_teacher, "teacher",
                {kTlas, kStorage, kStorage, kStorage, kStorage, kStorage, kStorage, kStorage, kStorage,
                 kStorage, kStorage, kStorage, kStorage, kStorage},
                0);

    const uint32_t width = relight.width();
    const uint32_t height = relight.height();
    const VkDeviceSize pixels = static_cast<VkDeviceSize>(width) * height;
    ensureBuffer(m_rowCounts, height * 2 * sizeof(uint32_t));
    ensureBuffer(m_counts, 2 * 4 * sizeof(uint32_t));
    ensureBuffer(m_smoothList, pixels * sizeof(uint32_t));
    ensureBuffer(m_edgeList, pixels * sizeof(uint32_t));
    ensureBuffer(m_teacherParams, sizeof(TeacherParams));
    ensureBuffer(m_lights, kMaxLights * sizeof(TraceLight));
    ensureBuffer(m_samples, static_cast<VkDeviceSize>(total) * sizeof(VisibilityTrainingSample));
    ensureBuffer(m_samplePixels, static_cast<VkDeviceSize>(total) * sizeof(uint32_t));

    TraceLight traceLights[kMaxLights]{};
    for (uint32_t i = 0; i < lights.size(); ++i)
    {
        traceLights[i] = traceLightFor(lights[i], shadows, focusRadius);
    }
    params.receiverCameraToWorld = glm::inverse(receiverCamera.worldToCamera);
    params.receiverIntrinsics =
        glm::vec4(receiverCamera.fx, receiverCamera.fy, receiverCamera.cx, receiverCamera.cy);
    params.randomTarget = settings.randomSamples;
    params.edgeTarget = settings.edgeSamples;
    params.samplesPerAxis = std::max(settings.emitterSamplesPerAxis, 1u);
    params.seed = settings.seed;
    params.width = width;
    params.height = height;
    params.normalOffset = shadows.normalOffset * focusRadius;

    // sample_select: gNormalDepth, gMetalAmbientCoverage, rowCounts, counts, smoothList, edgeList.
    bindWhole(m_select, 0, relight.gNormalDepth());
    bindWhole(m_select, 1, relight.gMetalAmbientCoverage());
    bindWhole(m_select, 2, *m_rowCounts);
    bindWhole(m_select, 3, *m_counts);
    bindWhole(m_select, 4, *m_smoothList);
    bindWhole(m_select, 5, *m_edgeList);
    // teacher: tlas, means, scales, rotations, opacities, gNormalDepth,
    // gMetalAmbientCoverage, smoothList, edgeList, counts, lights, params,
    // samples, samplePixels -- declaration order.
    m_teacher.descriptors.bindAccelerationStructure(m_teacher.set, 0, as.tlas());
    bindView(m_teacher, 1, scene.means);
    bindView(m_teacher, 2, scene.scales);
    bindView(m_teacher, 3, scene.rotations);
    bindView(m_teacher, 4, scene.opacity);
    bindWhole(m_teacher, 5, relight.gNormalDepth());
    bindWhole(m_teacher, 6, relight.gMetalAmbientCoverage());
    bindWhole(m_teacher, 7, *m_smoothList);
    bindWhole(m_teacher, 8, *m_edgeList);
    bindWhole(m_teacher, 9, *m_counts);
    bindWhole(m_teacher, 10, *m_lights);
    bindWhole(m_teacher, 11, *m_teacherParams);
    bindWhole(m_teacher, 12, *m_samples);
    bindWhole(m_teacher, 13, *m_samplePixels);

    SelectConstants select{};
    select.width = width;
    select.height = height;
    select.minCoverage = settings.minCoverage;
    select.edgeDepthRatio = settings.edgeDepthRatio;
    select.edgeNormalCos = settings.edgeNormalCos;

    submit("teacher samples",
           [&](CommandBuffer & cmd)
           {
               VkCommandBuffer handle = cmd.getHandle();
               vkCmdUpdateBuffer(handle, m_teacherParams->handle(), 0, sizeof(params), &params);
               vkCmdUpdateBuffer(handle, m_lights->handle(), 0, sizeof(traceLights), traceLights);
               for (uint32_t phase = 0; phase < 3; ++phase)
               {
                   Device::recordGlobalBarrier(handle); // params / previous phase -> this phase
                   select.phase = phase;
                   cmd.pushConstants(m_select.pipeline.getLayout(), VK_SHADER_STAGE_COMPUTE_BIT,
                                     sizeof(select), &select);
                   cmd.dispatchCompute(m_select.pipeline.getHandle(), m_select.pipeline.getLayout(),
                                       m_select.set, phase == 1 ? 1u : dispatchGroups(height, kWorkgroupSize),
                                       1, 1);
               }
               Device::recordGlobalBarrier(handle); // lists + counts -> teacher
               cmd.dispatchCompute(m_teacher.pipeline.getHandle(), m_teacher.pipeline.getLayout(),
                                   m_teacher.set, dispatchGroups(total, kWorkgroupSize), 1, 1);
           });

    uint32_t counts[8] = {};
    m_counts->download(counts, sizeof(counts));
    m_sampleCount = counts[4];
    m_smoothSampleCount = counts[5];
    m_samplesWidth = width;
    m_samplesHeight = height;
    return m_sampleCount;
}

void VisibilityPass::splatBaked(const ForwardPass & forward, const CameraUniforms & camera,
                                const RelightPass & relight, const CanonicalGaussianBufferViews & scene,
                                const BufferView & visibilitySH, const std::vector<Light> & lights)
{
    VK_VALIDATE(visibilitySH.valid(),
                "VisibilityPass::splatBaked: the scene has no visibility stream -- bake one "
                "(Renderer::bakeVisibility) or load/borrow one");
    VK_VALIDATE(lights.size() <= kMaxLights, "VisibilityPass::splatBaked: too many lights");
    buildKernel(m_splat, "visibility_splat", std::vector<VkDescriptorType>(8, kStorage),
                sizeof(CameraUniforms));
    ensureBuffer(m_lights, kMaxLights * sizeof(TraceLight));
    ensureBuffer(m_splatParams, sizeof(SplatParams));

    TraceLight traceLights[kMaxLights]{};
    SplatParams params{};
    params.lightCount = static_cast<uint32_t>(lights.size());
    for (uint32_t i = 0; i < lights.size(); ++i)
    {
        // Only the position/type and direction are read; sizes don't matter.
        traceLights[i] = traceLightFor(lights[i], ShadowSettings{}, 1.0f);
        params.lightMask |= lights[i].castsShadows != 0 ? (1u << i) : 0u;
    }

    // projected, sortedValues, tileRanges, means, visibilitySH, lights, params, visibility.
    bindWhole(m_splat, 0, forward.projected());
    bindWhole(m_splat, 1, forward.sortedValues());
    bindWhole(m_splat, 2, forward.tileRanges());
    bindView(m_splat, 3, scene.means);
    bindView(m_splat, 4, visibilitySH);
    bindWhole(m_splat, 5, *m_lights);
    bindWhole(m_splat, 6, *m_splatParams);
    bindWhole(m_splat, 7, relight.visibility());

    submit("baked visibility splat",
           [&](CommandBuffer & cmd)
           {
               VkCommandBuffer handle = cmd.getHandle();
               vkCmdUpdateBuffer(handle, m_lights->handle(), 0, sizeof(traceLights), traceLights);
               vkCmdUpdateBuffer(handle, m_splatParams->handle(), 0, sizeof(params), &params);
               Device::recordGlobalBarrier(handle); // params -> kernel
               cmd.pushConstants(m_splat.pipeline.getLayout(), VK_SHADER_STAGE_COMPUTE_BIT,
                                 sizeof(CameraUniforms), &camera);
               cmd.dispatchCompute(m_splat.pipeline.getHandle(), m_splat.pipeline.getLayout(),
                                   m_splat.set, camera.tilesX, camera.tilesY, 1);
           });
}

void VisibilityPass::snapshotReference(const RelightPass & relight)
{
    const Buffer & visibility = relight.visibility();
    ensureBuffer(m_reference, visibility.size());
    submit("reference snapshot",
           [&](CommandBuffer & cmd)
           { cmd.copyBuffer(visibility.handle(), m_reference->handle(), visibility.size()); });
}

void VisibilityPass::saveCache(const RelightPass & relight)
{
    const Buffer & visibility = relight.visibility();
    ensureBuffer(m_cache, visibility.size());
    submit("visibility cache save",
           [&](CommandBuffer & cmd) { cmd.copyBuffer(visibility.handle(), m_cache->handle(), visibility.size()); });
}

void VisibilityPass::restoreCache(const RelightPass & relight)
{
    const Buffer & visibility = relight.visibility();
    VK_VALIDATE(m_cache != nullptr && m_cache->size() >= visibility.size(),
                "VisibilityPass::restoreCache: no saved frame of this size");
    submit("visibility cache restore",
           [&](CommandBuffer & cmd) { cmd.copyBuffer(m_cache->handle(), visibility.handle(), visibility.size()); });
}

void VisibilityPass::applyErrorVsReference(const RelightPass & relight)
{
    VK_VALIDATE(m_reference != nullptr && m_reference->size() >= relight.visibility().size(),
                "VisibilityPass::applyErrorVsReference: no reference snapshot of this size");
    buildKernel(m_debug, "visibility_debug", std::vector<VkDescriptorType>(4, kStorage),
                sizeof(DebugConstants));
    ensureBuffer(m_dummy, 64);
    bindWhole(m_debug, 0, relight.visibility());
    bindWhole(m_debug, 1, *m_reference);
    bindWhole(m_debug, 2, *m_dummy);
    bindWhole(m_debug, 3, *m_dummy);
    DebugConstants constants{0, relight.width(), relight.height(), 0};
    submit("visibility error view",
           [&](CommandBuffer & cmd)
           {
               cmd.pushConstants(m_debug.pipeline.getLayout(), VK_SHADER_STAGE_COMPUTE_BIT,
                                 sizeof(constants), &constants);
               cmd.dispatchCompute(m_debug.pipeline.getHandle(), m_debug.pipeline.getLayout(),
                                   m_debug.set,
                                   dispatchGroups(constants.width * constants.height, kWorkgroupSize), 1, 1);
           });
}

void VisibilityPass::markSampleLocations(const RelightPass & relight)
{
    buildKernel(m_debug, "visibility_debug", std::vector<VkDescriptorType>(4, kStorage),
                sizeof(DebugConstants));
    ensureBuffer(m_dummy, 64);
    // Samples of a different image size cannot be placed; show none.
    const bool haveSamples = m_samples && m_sampleCount > 0 && m_samplesWidth == relight.width()
                             && m_samplesHeight == relight.height();
    bindWhole(m_debug, 0, relight.visibility());
    bindWhole(m_debug, 1, *m_dummy);
    bindWhole(m_debug, 2, haveSamples ? *m_samplePixels : *m_dummy);
    bindWhole(m_debug, 3, haveSamples ? *m_samples : *m_dummy);
    DebugConstants clear{1, relight.width(), relight.height(), 0};
    DebugConstants mark{2, relight.width(), relight.height(), haveSamples ? m_sampleCount : 0};
    submit("sample locations view",
           [&](CommandBuffer & cmd)
           {
               VkPipelineLayout layout = m_debug.pipeline.getLayout();
               cmd.pushConstants(layout, VK_SHADER_STAGE_COMPUTE_BIT, sizeof(clear), &clear);
               cmd.dispatchCompute(m_debug.pipeline.getHandle(), layout, m_debug.set,
                                   dispatchGroups(clear.width * clear.height, kWorkgroupSize), 1, 1);
               if (mark.sampleCount > 0)
               {
                   Device::recordGlobalBarrier(cmd.getHandle()); // clear -> mark
                   cmd.pushConstants(layout, VK_SHADER_STAGE_COMPUTE_BIT, sizeof(mark), &mark);
                   cmd.dispatchCompute(m_debug.pipeline.getHandle(), layout, m_debug.set,
                                       dispatchGroups(mark.sampleCount, kWorkgroupSize), 1, 1);
               }
           });
}

void VisibilityPass::copyExternal(const RelightPass & relight, const BufferView & external)
{
    const Buffer & visibility = relight.visibility();
    VK_VALIDATE(external.buffer != VK_NULL_HANDLE && external.size >= visibility.size(),
                "ShadowMode::External: the external visibility view must hold kMaxLights layers of "
                "width*height floats (" + std::to_string(visibility.size()) + " bytes) -- see "
                "Renderer::setExternalVisibility");
    submit("external visibility copy",
           [&](CommandBuffer & cmd)
           {
               VkBufferCopy region{};
               region.srcOffset = external.offset;
               region.size = visibility.size();
               vkCmdCopyBuffer(cmd.getHandle(), external.buffer, visibility.handle(), 1, &region);
           });
}

} // namespace sabercat
