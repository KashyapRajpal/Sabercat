// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "sabercat/buffer_view.h"

namespace sabercat
{

class Buffer;
class Device;

// Merges the selection masks of a layer merge into one scene-wide mask, on the GPU
// (shaders/selection_merge.slang). Internal, not installed: the Renderer owns one
// for the selection highlight and binds mergedMask() into ForwardPass, which tints
// the selected Gaussians from it (shaders/highlight_tint.slang).
class HighlightPass
{
  public:
    HighlightPass(Device & device, const std::string & shaderDir);
    ~HighlightPass(); // defined in the .cpp, where MergeKernel is complete

    // Non-copyable: it owns the merge kernel and the merged mask buffer.
    HighlightPass(const HighlightPass &) = delete;
    HighlightPass & operator=(const HighlightPass &) = delete;

    // Gathers each layer's selection bits into the merged mask. layerMasks[i] is
    // layer i's layer-local mask (bit g = the layer's Gaussian g) and layerCounts[i]
    // its Gaussian count; the layers are laid out in order, so layer i starts after
    // the layers before it. The counts must sum to totalCount. A layer with count 0
    // contributes nothing. A layer's view must hold at least ceil(count / 32) words.
    // All checks run before any GPU work (VK_VALIDATE). One Device::immediateSubmit.
    //
    // The merged buffer is grow-only: it holds max(1, ceil(totalCount / 32)) words
    // at least, and its contents are a snapshot of the masks at this call.
    void buildMergedMask(const std::vector<BufferView> & layerMasks,
                         const std::vector<uint32_t> & layerCounts, uint32_t totalCount);

    // The merged mask of the last buildMergedMask(): one bit per merged Gaussian,
    // word w covering Gaussians [32w, 32w + 32). Throws before the first build.
    const Buffer & mergedMask() const;

    // Advances whenever mergedMask() is reallocated, so a caller holding a view of
    // it knows the handle changed.
    uint64_t generation() const
    {
        return m_generation;
    }

  private:
    struct MergeKernel; // the pipeline and descriptor sets, built on the first build

    Device & m_device;
    std::string m_shaderDir;
    std::unique_ptr<Buffer> m_merged;
    uint64_t m_generation = 0;
    std::unique_ptr<MergeKernel> m_kernel;
};

} // namespace sabercat
