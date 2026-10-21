// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>

#include "sabercat/buffer_view.h"

namespace sabercat
{

// A borrowed (application-owned) scene whose GPU storage the APPLICATION
// owns. Sabercat renders and differentiates it in place and never allocates,
// frees, grows or uploads into these buffers. See Renderer::borrowGaussians().
//
// Every stream is a BufferView at the canonical stride (render_config.h /
// docs/formats.md). Capacity is implied: the number of whole Gaussians the
// smallest parameter stream holds. Gradient streams are either all valid
// (borrowed too) or all invalid (Sabercat allocates its own on first use).
struct BorrowedGaussians
{
    // means/scales/rotations/opacity/shCoefficients required; confidence
    // optional and ignored. gaussianCount = how many to render (1..capacity);
    // shDegree = the highest SH band the data holds (0-3).
    CanonicalGaussianBufferViews parameters;
    // All six valid, or all six invalid (VK_NULL_HANDLE). gaussianCount is
    // ignored.
    CanonicalGaussianGradientViews gradients;
    // The optional relightable streams, each independently
    // (invalid => absent). Checked like the parameter streams.
    RelightBufferViews relight;
    // The scene's opacity-compensation flag (see Renderer::setAntialiased).
    bool antialiased = false;
};

// Checks a borrowed scene against everything Sabercat requires of it and
// returns its capacity in Gaussians. Throws std::invalid_argument naming the
// stream and the problem. Renderer::borrowGaussians() calls it; it is public
// so an application can check its buffers up front. `physicalDevice` and
// `device` must be the ones the buffers were created on.
//
// Checked, per stream: a buffer handle; the canonical stride; the offset a
// multiple of minStorageBufferOffsetAlignment; size <= maxStorageBufferRange;
// offset + size within the buffer (vkGetBufferMemoryRequirements). Overall:
// shDegree <= 3; 1 <= gaussianCount <= capacity; gradients all-or-none and
// each covering the capacity; each given relightable stream covering the
// capacity; no two streams' used ranges overlapping.
//
// NOT checkable through Vulkan, so documented requirements instead
// (docs/library.md, "Requirements on borrowed buffers"): STORAGE_BUFFER usage
// (TRANSFER_SRC too for readGaussians/readGradients); created on `device`;
// bound to memory; EXCLUSIVE sharing on the compute queue's family (or
// CONCURRENT including it); alive until replaced or the renderer shuts down.
uint32_t validateBorrowedGaussians(VkPhysicalDevice physicalDevice, VkDevice device,
                                   const BorrowedGaussians & scene);

} // namespace sabercat
