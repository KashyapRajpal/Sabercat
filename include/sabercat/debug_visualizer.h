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
// which blits ForwardPass's own output image directly instead. Owned by
// Renderer, not ForwardPass: this is a viewer/debug concern, not part of
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

  private:
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
    Pipeline m_pipeline;
};

} // namespace sabercat
