// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

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

// Baked visibility and teacher samples.
//
//   bake               per-Gaussian 16-coefficient visibility SH into the
//                      `visibility` relight stream (bake_visibility.slang),
//                      in submissions small enough to never trip a TDR
//   generateTeacherSamples  sample_select.slang + teacher.slang over a lit
//                      frame's G-buffer: 75% smooth / 25% edge pixels (by
//                      default), ray-traced targets, a borrowable buffer
//   splatBaked         ShadowMode::BakedSH: the visibility SH evaluated toward
//                      each light and splatted into the visibility layers
//   debug helpers      snapshot/compare against a reference, mark samples,
//                      copy an external visibility buffer
//
// Every call is synchronous on Device::submitQueue() with the usual
// bracketing barriers (see docs/library.md, synchronization). Pipelines and
// buffers are created on first use; the ray-traced parts need
// Device::features().rayQuery.
class VisibilityPass
{
  public:
    VisibilityPass(Device & device, const std::string & shaderDir);
    ~VisibilityPass();

    VisibilityPass(const VisibilityPass &) = delete;
    VisibilityPass & operator=(const VisibilityPass &) = delete;

    // Bakes Gaussians [0, scene.gaussianCount) of `scene` into `visibility`
    // (kVisibilityCoefficients floats per Gaussian), tracing `as` -- built
    // over `scene` by the caller.
    void bake(const CanonicalGaussianBufferViews & scene, const GaussianAccelerationStructure & as,
              const BufferView & visibility, const VisibilityBakeSettings & settings, float rayLength);

    // Selects and traces teacher samples for the lit frame in `relight`
    // (rendered through `receiverCamera`). `lights` with castsShadows are
    // assigned round-robin; at least one is required. Light emitter sizes and
    // the ray origin offset follow `shadows` and `focusRadius` exactly as
    // ShadowPass::runRayTraced does. Returns the sample count.
    uint32_t generateTeacherSamples(const RelightPass & relight, const CameraUniforms & receiverCamera,
                                    const CanonicalGaussianBufferViews & scene,
                                    const GaussianAccelerationStructure & as,
                                    const std::vector<Light> & lights,
                                    const TeacherSettings & settings, const ShadowSettings & shadows,
                                    float focusRadius);

    // ShadowMode::BakedSH: fills the visibility layers of the shadow-casting
    // `lights` from the per-Gaussian `visibilitySH`, splatted over
    // `forward`'s last run (`camera` is that run's uniforms).
    void splatBaked(const ForwardPass & forward, const CameraUniforms & camera,
                    const RelightPass & relight, const CanonicalGaussianBufferViews & scene,
                    const BufferView & visibilitySH, const std::vector<Light> & lights);

    // Copies the relight pass's visibility layers aside (the reference for
    // applyErrorVsReference).
    void snapshotReference(const RelightPass & relight);
    // visibility = |visibility - reference| in every layer.
    void applyErrorVsReference(const RelightPass & relight);
    // VisibilityDebug::SampleLocations over the last generated samples.
    void markSampleLocations(const RelightPass & relight);
    // ShadowMode::External: copies `external` (at least kMaxLights *
    // width * height floats, the visibility layout) into the visibility layers.
    void copyExternal(const RelightPass & relight, const BufferView & external);
    // The frame cache (Renderer: reuse a frame's visibility while nothing it
    // depends on changed): save copies the layers aside, restore copies them
    // back. restore requires a save of the same size.
    void saveCache(const RelightPass & relight);
    void restoreCache(const RelightPass & relight);

    // The last generateTeacherSamples() result: sampleCount() samples
    // (VisibilityTrainingSample, smooth ones first), and for each its pixel
    // index with the top bit set for an edge sample.
    uint32_t sampleCount() const
    {
        return m_sampleCount;
    }
    uint32_t smoothSampleCount() const
    {
        return m_smoothSampleCount;
    }
    const Buffer * samples() const
    {
        return m_samples.get();
    }
    const Buffer * samplePixels() const
    {
        return m_samplePixels.get();
    }

  private:
    struct Kernel
    {
        explicit Kernel(VkDevice device) : descriptors(device)
        {
        }
        DescriptorSet descriptors;
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
        Pipeline pipeline;
        bool ready = false;
    };

    // `types` lists each binding's descriptor type, in declaration order.
    void buildKernel(Kernel & kernel, const char * name, const std::vector<VkDescriptorType> & types,
                     uint32_t pushSize);
    void bindView(Kernel & kernel, uint32_t binding, const BufferView & view);
    void bindWhole(Kernel & kernel, uint32_t binding, const Buffer & buffer);
    // Grows `buffer` to at least `bytes` (contents not kept).
    void ensureBuffer(std::unique_ptr<Buffer> & buffer, VkDeviceSize bytes);
    // Records `record` between the contract's bracketing barriers and waits.
    template <typename Record>
    void submit(const char * what, Record && record);
    void requireRayQuery() const;

    Device & m_device;
    std::string m_shaderDir;
    VkCommandPool m_commandPool = VK_NULL_HANDLE;

    Kernel m_bake;
    Kernel m_select;
    Kernel m_teacher;
    Kernel m_splat;
    Kernel m_debug;

    std::unique_ptr<Buffer> m_dummy;
    std::unique_ptr<Buffer> m_cache;
    std::unique_ptr<Buffer> m_rowCounts;
    std::unique_ptr<Buffer> m_counts;
    std::unique_ptr<Buffer> m_smoothList;
    std::unique_ptr<Buffer> m_edgeList;
    std::unique_ptr<Buffer> m_teacherParams;
    std::unique_ptr<Buffer> m_lights;
    std::unique_ptr<Buffer> m_splatParams;
    std::unique_ptr<Buffer> m_samples;
    std::unique_ptr<Buffer> m_samplePixels;
    std::unique_ptr<Buffer> m_reference;
    uint32_t m_sampleCount = 0;
    uint32_t m_smoothSampleCount = 0;
    uint32_t m_samplesWidth = 0; // the image the samples' pixel indices refer to
    uint32_t m_samplesHeight = 0;
};

} // namespace sabercat
