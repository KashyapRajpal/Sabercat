// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vulkan/vulkan.h>

#include <vma/vk_mem_alloc.h>

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
class CommandBuffer;
class ForwardPass;

// Host orchestration for the backward pass: dL/dImage -> dL/d(Gaussian
// parameters). Two kernels in one submit:
//
//   backward_rasterize  per tile: walk each pixel's blend back to front and
//                       atomically accumulate dL/d(center, conic, projected
//                       opacity, color) per Gaussian
//   backward_project    per Gaussian: chain those to the stored parameters,
//                       OVERWRITING the gradient buffers
//
// It differentiates exactly what ForwardPass last rendered: camera, parameter
// buffers and flags are all taken from ForwardPass::lastRun(), never passed
// again, and run() throws unless that record is a completed
// ForwardMode::Differentiable run (which guarantees no viewer LOD knob was
// active and that the per-pixel backward state was written). Pure: no loss,
// no optimizer, no training state.
//
// The depth gradient is optional. Given, it is dL/dD per pixel for the
// forward's depth output D = sum_j T_j alpha_j z_j (z_j the camera depth), in
// the forward's DepthLayout: width*height*n floats, row-major, and only
// component 0 is read. The remaining components are ignored, so a trainer can
// pass the whole depth buffer. The coverage gradient still flows through the
// image's alpha, and the front-depth component is not differentiated. Absent
// (nullptr, or an invalid view) the run is color-only, exactly as before the
// depth overloads existed. Needs a forward built with a depth layout other
// than DepthLayout::None.
//
// Needs fp32 atomic add on storage buffers (DeviceFeatures::
// shaderBufferFloat32AtomicAdd); the constructor throws a named error without
// it. Renderer builds this lazily, so forward-only use never requires it.
class BackwardPass
{
  public:
    // shaderDir must contain backward_rasterize/backward_project.comp.spv.
    BackwardPass(Device & device, const std::string & shaderDir);
    ~BackwardPass();

    BackwardPass(const BackwardPass &) = delete;
    BackwardPass & operator=(const BackwardPass &) = delete;

    // `dLossDImage`: device buffer of width*height RGBA floats, row-major --
    // the same layout as ForwardPass::downloadOutputImage(). Zero-copy path.
    void run(const ForwardPass & forward, const BufferView & dLossDImage,
             const CanonicalGaussianGradientViews & gradients);

    // As above, plus `dLossDDepth`: device buffer of width*height*n floats in
    // the forward's depth layout, n = floats per pixel (see the class comment).
    // An invalid (default) view means no depth gradient.
    void run(const ForwardPass & forward, const BufferView & dLossDImage,
             const BufferView & dLossDDepth, const CanonicalGaussianGradientViews & gradients);

    // Host convenience: copies `dLossDImageHost` (width*height*4 floats) in
    // through a persistent mapped staging buffer, then runs as above.
    void run(const ForwardPass & forward, const float * dLossDImageHost,
             const CanonicalGaussianGradientViews & gradients);

    // Host convenience with a depth gradient: `dLossDDepthHost` is width*height*n
    // floats (n = the forward's layout), staged the same way. nullptr means no
    // depth gradient. Throws if it is given and the forward has DepthLayout::None.
    void run(const ForwardPass & forward, const float * dLossDImageHost,
             const float * dLossDDepthHost, const CanonicalGaussianGradientViews & gradients);

    // Verification: the per-Gaussian screen-space gradients backward_rasterize
    // accumulated (SABERCAT_PROJECTED_GRADIENT_FLOATS floats each).
    const Buffer & projectedGradients() const
    {
        return *m_projectedGradients;
    }

  private:
    // Host path only: a gradient uploaded from the host. A persistent mapped
    // staging buffer, its device copy, and the bytes of the current upload.
    struct HostUpload
    {
        std::unique_ptr<Buffer> device;
        VkBuffer staging = VK_NULL_HANDLE;
        VmaAllocation stagingAllocation = VK_NULL_HANDLE;
        void * stagingMapped = nullptr;
        VkDeviceSize capacity = 0;
        VkDeviceSize bytes = 0;
    };

    // The destructor's body, also run by the constructor when it throws
    // part-way (a throwing constructor never reaches the destructor).
    void release();
    void createPipelines(const std::string & shaderDir);
    void destroyUpload(HostUpload & upload);
    // Grows `upload` to hold `bytes`, copies `host` in and flushes it.
    void stageUpload(HostUpload & upload, const float * host, VkDeviceSize bytes);
    void recordUploadCopy(CommandBuffer & cmd, const HostUpload & upload);
    void record(const ForwardPass & forward, const BufferView & dLossDImage,
                const BufferView & dLossDDepth, const CanonicalGaussianGradientViews & gradients,
                const HostUpload * imageUpload, const HostUpload * depthUpload);

    Device & m_device;
    VkCommandPool m_commandPool = VK_NULL_HANDLE;

    std::unique_ptr<Buffer> m_projectedGradients; // gaussianCount * 48 B
    uint32_t m_gaussianCapacity = 0; // m_projectedGradients size in Gaussians, grow-only

    HostUpload m_imageUpload;
    HostUpload m_depthUpload;

    DescriptorSet m_rasterizeDescriptors;
    VkDescriptorSetLayout m_rasterizeLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_rasterizeSet = VK_NULL_HANDLE;
    Pipeline m_rasterizePipeline;

    DescriptorSet m_projectDescriptors;
    VkDescriptorSetLayout m_projectLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_projectSet = VK_NULL_HANDLE;
    Pipeline m_projectPipeline;
};

} // namespace sabercat
