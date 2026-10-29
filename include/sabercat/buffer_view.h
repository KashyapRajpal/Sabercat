// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>

namespace sabercat
{

// A non-owning reference to a range of a GPU buffer (the "buffer
// seam"). Compute passes bind BufferViews, never owned Buffer objects, so
// the same pass works whether Sabercat allocated the memory (ownership
// model A) or an application handed it in (model B) -- moving between them
// changes who constructs the view, not the passes that consume it.
//
// `stride` is the byte distance between consecutive elements. The canonical
// Gaussian streams use padded strides (16 B for xyz/xyzw data) so buffers
// are byte-identical with a trainer's and can be shared with zero copies.
struct BufferView
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    VkDeviceSize size = 0; // bytes covered by this view
    uint32_t stride = 0;   // bytes per element

    bool valid() const
    {
        return buffer != VK_NULL_HANDLE && stride > 0;
    }

    uint32_t elementCount() const
    {
        return stride > 0 ? static_cast<uint32_t>(size / stride) : 0;
    }
};

// The canonical SoA Gaussian streams, as non-owning views. Mirrors
// CanonicalGaussianBufferViews -- strides are fixed by
// the shared contract (see docs/formats.md), not by whoever allocated them.
//
// Covariance is deliberately absent: it is derived on the GPU from scale +
// rotation rather than stored, so there is no covariance stream to share.
struct CanonicalGaussianBufferViews
{
    BufferView means;          // 16 B: xyz + pad
    BufferView scales;         // 16 B: positive linear std-dev xyz + pad
    BufferView rotations;      // 16 B: quaternion xyzw, normalized
    BufferView opacity;        // 4 B: scalar in [0,1]
    BufferView shCoefficients; // 16 B per coefficient: RGB + pad
    BufferView confidence;     // 4 B, optional (invalid view => absent)

    uint32_t gaussianCount = 0;
    uint32_t shDegree = 0; // coefficient count per Gaussian is (degree+1)^2
};

// The optional relightable streams (see docs/architecture.md, "Relightable
// splats"), as non-owning views. Each is independently optional
// (invalid view => absent, and the relight pass derives a default); all are
// ACTIVATED values in the scene's own (asset) space.
struct RelightBufferViews
{
    BufferView normals;            // 16 B: unit normal xyz + pad
    BufferView baseColorRoughness; // 16 B: linear base color rgb + roughness [0,1]
                                   // (a missing stream's default is the SH DC color,
                                   // display-referred, sRGB-decoded to linear)
    BufferView metallic;           // 4 B: [0,1]
    BufferView visibility;         // 4 B x 16 per Gaussian: mono SH (degree 3)
};

// The backward pass's outputs, as non-owning views. Each stream has
// EXACTLY the stride of the parameter stream it differentiates, so an
// optimizer walks parameters and gradients with the same indexing (and the
// pad lanes of the vec4 streams are written as 0).
//
// Every gradient is w.r.t. the STORED, already-activated value -- the trainer
// owns the activations and chains through them itself:
//   means          dL/d(world xyz)
//   scales         dL/d(linear std-dev)          -> log-space: multiply by scale
//   rotations      dL/d(stored xyzw), through the rotation-matrix formula
//                  applied to the 4 values as-is (no normalization inside the
//                  renderer, same as the reference CUDA rasterizer)
//                                                -> trainer chains its normalize
//   opacity        dL/d(opacity in [0,1])        -> logit: multiply by o(1-o)
//   shCoefficients dL/d(raw coefficient), band-ascending, 16 per Gaussian;
//                  0 above the SH degree the forward evaluated
//   screenMeans    dL/d(u, v) of the projected center in pixels, + 2 pad --
//                  not a parameter: the per-view statistic 3DGS densification
//                  thresholds on, exposed so the trainer needn't recompute it
//
// backward() OVERWRITES these (it is a pure function of the forward state
// and dL/dImage). Summing over several views is the trainer's business.
struct CanonicalGaussianGradientViews
{
    BufferView means;          // 16 B
    BufferView scales;         // 16 B
    BufferView rotations;      // 16 B
    BufferView opacity;        // 4 B
    BufferView shCoefficients; // 16 B per coefficient
    BufferView screenMeans;    // 16 B: (du, dv, 0, 0)

    uint32_t gaussianCount = 0;
};

} // namespace sabercat
