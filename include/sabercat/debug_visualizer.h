// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <string>

#include "sabercat/descriptor_set.h"
#include "sabercat/image.h"
#include "sabercat/pipeline.h"
#include "sabercat/renderer.h"

namespace sabercat
{

class Device;
class Buffer;
class ForwardPass;

// Presentation-only debug views (see DebugView in renderer.h). Reads
// buffers ForwardPass already produced and writes a normalized RGBA image
// for Renderer::present() to blit -- never runs for DebugView::Final,
// which blits ForwardPass's own output image directly instead. Also owns
// one headless slot (renderHeadless()) that present() never touches, so
// a headless render can never race present()'s per-frame resources. Owned
// by Renderer, not ForwardPass: this is a viewer/debug concern, not part of
// the differentiable rasterizer; the two are kept separate.
class DebugVisualizer
{
  public:
    DebugVisualizer(Device & device, uint32_t width, uint32_t height,
                    const std::string & shaderDir);
    ~DebugVisualizer();

    DebugVisualizer(const DebugVisualizer &) = delete;
    DebugVisualizer & operator=(const DebugVisualizer &) = delete;

    void resize(uint32_t width, uint32_t height);

    // Records the visualize dispatch into `cmd` (a command buffer the
    // caller is already recording, e.g. Renderer::present()'s per-frame
    // buffer), reading `forward`'s output/depth images and the buffers its
    // last run() left behind (tile ranges; projected Gaussians, sorted
    // instances and per-Gaussian SH degrees for DebugView::SHDegree). Must
    // follow a run() that reached ForwardStage::Rasterize, with no run()
    // since -- the images are then in VK_IMAGE_LAYOUT_GENERAL.
    // Does nothing for DebugView::Final -- callers should skip calling this
    // and blit the color image directly in that case. `frameIndex` MUST be
    // the caller's own frames-in-flight index (Renderer::present() passes
    // its currentFrame) -- this selects which of kMaxFramesInFlight
    // descriptor sets/display images to rewrite, so that rebinding this
    // frame's set can never race a still-pending command buffer that
    // referenced a DIFFERENT frame's set. Rebinding a single shared set
    // every call (the previous design) violated
    // VUID-vkUpdateDescriptorSets-None-03047: with frames-in-flight, the
    // other slot's command buffer can still be executing on the GPU when
    // this call happens.
    void record(VkCommandBuffer cmd, DebugView mode, const ForwardPass & forward,
                float depthScale, float densityScale, uint32_t frameIndex);

    // Must be the same frameIndex just passed to record().
    const Image & displayImage(uint32_t frameIndex) const
    {
        return *m_displayImages[frameIndex];
    }

    // Headless: renders `mode` over `forward`'s last run into a private slot
    // (separate from the per-frame ones, so it can never race present()) and
    // returns the result as width*height RGBA32F floats in a device buffer
    // (row-major, 16 B/px). The slot (descriptor set, display image, buffer)
    // is created on the first call. Synchronous: one Device::immediateSubmit.
    // `outputLayout` is the layout forward's COLOR image is in now (GENERAL,
    // or TRANSFER_SRC_OPTIMAL after present() blitted it); it is left in
    // GENERAL. DebugView::Final copies the color image itself; no dispatch.
    // The returned buffer is reused and stays valid until the next call or
    // resize().
    const Buffer & renderHeadless(DebugView mode, const ForwardPass & forward, float depthScale,
                                  float densityScale, VkImageLayout outputLayout);

  private:
    // The body of record(), for any descriptor set / display image pair: the
    // per-frame slots and the headless slot both go through it.
    void recordSlot(VkCommandBuffer cmd, DebugView mode, const ForwardPass & forward,
                    float depthScale, float densityScale, VkDescriptorSet set,
                    const Image & displayImage, bool & firstUse);

    Device & m_device;
    uint32_t m_width;
    uint32_t m_height;
    // Per-frame-in-flight: displayImage's first transition must be from
    // UNDEFINED, tracked independently per slot.
    bool m_firstUse[kMaxFramesInFlight];

    std::unique_ptr<Image> m_displayImages[kMaxFramesInFlight];

    DescriptorSet m_descriptors;
    VkDescriptorSetLayout m_layout = VK_NULL_HANDLE;
    VkDescriptorSet m_sets[kMaxFramesInFlight] = {};
    // Headless slot (renderHeadless()): one extra set, display image and
    // readback buffer, created lazily and recreated on resize().
    VkDescriptorSet m_headlessSet = VK_NULL_HANDLE;
    std::unique_ptr<Image> m_headlessImage;
    std::unique_ptr<Buffer> m_headlessBuffer;
    bool m_headlessFirstUse = true;
    Pipeline m_pipeline;
};

} // namespace sabercat
