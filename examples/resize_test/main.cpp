// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Resizable Gaussian storage + the shared-queue synchronization
// contract, driven through the public Renderer API exactly the way
// an application that trains would use it.
//
// An optimizer densifies by writing new Gaussians into the storage tail through
// the zero-copy views -- with its OWN command buffers on Sabercat's compute
// queue -- and then raising the count. So the checks are:
//
//  1. reserveGaussians() grows capacity, replaces handles, advances the
//     storage generation, and preserves the active prefix bit-exactly; a
//     non-growing reserve changes nothing.
//  2. setGaussianCount() keeps every handle and the generation; it
//     invalidates the last differentiable forward (backward must refuse).
//  3. Growing by writing the tail on the GPU + setGaussianCount renders and
//     differentiates EXACTLY like a fresh setGaussians() of the same
//     Gaussians: image bit-identical, gradients equal up to fp32 atomic
//     ordering. Same for shrinking (prune to a prefix), and for a scene
//     created with spare capacity up front.
//  4. The synchronization contract: the tail writes are submitted to
//     Renderer::computeQueue() and NOT waited on or barriered before
//     forward(); the gradients are copied out by a command buffer submitted
//     right after backward() with no barrier. Both must see the right data.
//  5. Gradient views are available before the first backward.
//
// Validation messages are logged as "[validation]"; the run must have none.

#include <vulkan/vulkan.h>

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "sabercat/camera.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/renderer.h"

using namespace sabercat;

namespace
{

constexpr uint32_t kWidth = 64;
constexpr uint32_t kHeight = 48;
constexpr uint32_t kTotal = 12; // Gaussians in the full scene
constexpr uint32_t kCoeffs = 16;

int g_failures = 0;

void check(bool ok, const std::string & what)
{
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok)
    {
        ++g_failures;
    }
}

// Host arrays at the canonical strides.
struct Scene
{
    std::vector<float> means, scales, rotations, opacities, sh;

    explicit Scene(uint32_t n)
        : means(n * 4), scales(n * 4), rotations(n * 4), opacities(n), sh(n * kCoeffs * 4)
    {
    }
};

// Deterministic, all-visible, overlapping Gaussians with every SH band
// non-zero, so a mis-copied or mis-indexed element changes the image.
Scene makeScene()
{
    Scene s(kTotal);
    for (uint32_t i = 0; i < kTotal; ++i)
    {
        const float t = static_cast<float>(i);
        s.means[i * 4 + 0] = -0.9f + 0.16f * t;
        s.means[i * 4 + 1] = 0.35f * std::sin(1.7f * t);
        s.means[i * 4 + 2] = -0.4f * std::cos(0.9f * t);
        s.scales[i * 4 + 0] = 0.10f + 0.02f * static_cast<float>(i % 3);
        s.scales[i * 4 + 1] = 0.07f + 0.015f * static_cast<float>(i % 4);
        s.scales[i * 4 + 2] = 0.09f;
        glm::vec4 q(0.1f * t, 0.3f, -0.2f + 0.05f * t, 1.0f);
        q = glm::normalize(q);
        std::memcpy(&s.rotations[i * 4], &q, sizeof(q));
        s.opacities[i] = 0.35f + 0.05f * static_cast<float>(i % 9);
        for (uint32_t k = 0; k < kCoeffs; ++k)
        {
            for (uint32_t c = 0; c < 3; ++c)
            {
                const float base = k == 0 ? 0.6f : 0.08f;
                s.sh[(i * kCoeffs + k) * 4 + c] =
                    base * std::sin(0.37f * t + 1.3f * static_cast<float>(k) + 2.1f * c);
            }
        }
    }
    return s;
}

GaussianHostInput prefixInput(const Scene & s, uint32_t n, uint32_t capacity = 0)
{
    GaussianHostInput in;
    in.count = n;
    in.means = s.means.data();
    in.scales = s.scales.data();
    in.rotations = s.rotations.data();
    in.opacities = s.opacities.data();
    in.shCoefficients = s.sh.data();
    in.shDegree = 3;
    in.capacity = capacity;
    return in;
}

Camera makeCamera()
{
    Camera c;
    c.width = kWidth;
    c.height = kHeight;
    c.fx = c.fy = 52.0f;
    c.cx = 0.5f * (kWidth - 1);
    c.cy = 0.5f * (kHeight - 1);
    c.cameraToWorld = glm::mat4(1.0f);
    c.cameraToWorld[3] = glm::vec4(0.05f, -0.02f, 3.0f, 1.0f);
    return c;
}

// Fixed pseudo-random dL/dImage (RGBA).
std::vector<float> makeImageGradient()
{
    std::vector<float> g(kWidth * kHeight * 4);
    uint32_t state = 12345u;
    for (float & v : g)
    {
        state = state * 1664525u + 1013904223u;
        v = static_cast<float>(state >> 8) / static_cast<float>(1u << 24) - 0.5f;
    }
    return g;
}

struct Result
{
    std::vector<float> image;
    GaussianGradients gradients;
};

Result renderAndDifferentiate(Renderer & r, const std::vector<float> & dLdImage)
{
    r.forward(RenderMode::Differentiable);
    Result out;
    out.image = r.readImage();
    r.backward(dLdImage.data());
    out.gradients = r.readGradients();
    return out;
}

// Worst |a-b| over a stream, relative to the stream's largest magnitude:
// gradients differ only by fp32 atomic accumulation order, so this should be
// at float-epsilon scale, not "close".
template <typename T>
double relativeError(const std::vector<T> & a, const std::vector<T> & b)
{
    if (a.size() != b.size())
    {
        return 1e30;
    }
    const float * pa = reinterpret_cast<const float *>(a.data());
    const float * pb = reinterpret_cast<const float *>(b.data());
    const size_t n = a.size() * sizeof(T) / sizeof(float);
    double worst = 0.0, scale = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        worst = std::max(worst, std::fabs(static_cast<double>(pa[i]) - pb[i]));
        scale = std::max(scale, std::fabs(static_cast<double>(pb[i])));
    }
    return scale > 0.0 ? worst / scale : worst;
}

double worstGradientError(const GaussianGradients & a, const GaussianGradients & b)
{
    return std::max({relativeError(a.means, b.means), relativeError(a.scales, b.scales),
                     relativeError(a.rotations, b.rotations),
                     relativeError(a.opacities, b.opacities),
                     relativeError(a.shCoefficients, b.shCoefficients),
                     relativeError(a.screenMeans, b.screenMeans)});
}

bool imagesIdentical(const std::vector<float> & a, const std::vector<float> & b)
{
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

void compareTo(const Result & got, const Result & want, const std::string & label)
{
    check(imagesIdentical(got.image, want.image), label + ": image bit-identical to fresh scene");
    const double err = worstGradientError(got.gradients, want.gradients);
    char buf[160];
    std::snprintf(buf, sizeof(buf), "%s: gradients match fresh scene (worst rel. %.3g, tol 1e-5)",
                  label.c_str(), err);
    check(err <= 1e-5, buf);
}

// The optimizer's side of the seam: its own command pool on Sabercat's compute
// queue, used to write into / read out of Sabercat's buffers. Submissions are
// deliberately NOT followed by a wait before the next Sabercat call -- that is
// what the synchronization contract promises is unnecessary.
class ExternalQueueUser
{
  public:
    explicit ExternalQueueUser(Renderer & r) : m_r(r), m_dev(r.device())
    {
        VkCommandPoolCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        info.queueFamilyIndex = r.computeQueueFamily();
        if (vkCreateCommandPool(m_dev, &info, nullptr, &m_pool) != VK_SUCCESS)
        {
            throw std::runtime_error("external command pool");
        }
    }

    ~ExternalQueueUser()
    {
        vkDeviceWaitIdle(m_dev);
        for (VkFence f : m_fences)
        {
            vkDestroyFence(m_dev, f, nullptr);
        }
        if (m_readback != VK_NULL_HANDLE)
        {
            vkDestroyBuffer(m_dev, m_readback, nullptr);
            vkFreeMemory(m_dev, m_readbackMemory, nullptr);
        }
        vkDestroyCommandPool(m_dev, m_pool, nullptr);
    }

    // Records `fn` and submits it to the compute queue. Returns the fence but
    // does not wait on it.
    template <typename F>
    VkFence submit(F && fn)
    {
        VkCommandBufferAllocateInfo alloc{};
        alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc.commandPool = m_pool;
        alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc.commandBufferCount = 1;
        VkCommandBuffer cmd;
        vkAllocateCommandBuffers(m_dev, &alloc, &cmd);
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &begin);
        fn(cmd);
        vkEndCommandBuffer(cmd);

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence;
        vkCreateFence(m_dev, &fenceInfo, nullptr, &fence);
        m_fences.push_back(fence);
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        if (vkQueueSubmit(m_r.computeQueue(), 1, &si, fence) != VK_SUCCESS)
        {
            throw std::runtime_error("external submit");
        }
        return fence;
    }

    void wait(VkFence fence)
    {
        vkWaitForFences(m_dev, 1, &fence, VK_TRUE, UINT64_MAX);
    }

    // Writes Gaussians [first, last) of `s` into the parameter views, the way
    // an optimizer's activation kernel would. vkCmdUpdateBuffer stands in for
    // that kernel (it is a transfer write; Sabercat's barrier covers any
    // write type).
    VkFence writeTail(const CanonicalGaussianBufferViews & v, const Scene & s, uint32_t first,
                      uint32_t last)
    {
        return submit(
            [&](VkCommandBuffer cmd)
            {
                const auto update = [&](const BufferView & view, const float * src,
                                        size_t floatsPerGaussian)
                {
                    const VkDeviceSize bytesPer = floatsPerGaussian * sizeof(float);
                    vkCmdUpdateBuffer(cmd, view.buffer, view.offset + first * bytesPer,
                                      (last - first) * bytesPer, src + first * floatsPerGaussian);
                };
                update(v.means, s.means.data(), 4);
                update(v.scales, s.scales.data(), 4);
                update(v.rotations, s.rotations.data(), 4);
                update(v.opacity, s.opacities.data(), 1);
                update(v.shCoefficients, s.sh.data(), kCoeffs * 4);
            });
    }

    // Copies the first `bytes` of `view` to host memory via a command buffer
    // submitted with no barrier of its own.
    std::vector<float> readBack(const BufferView & view, VkDeviceSize bytes)
    {
        ensureReadback(bytes);
        VkFence f = submit(
            [&](VkCommandBuffer cmd)
            {
                VkBufferCopy region{view.offset, 0, bytes};
                vkCmdCopyBuffer(cmd, view.buffer, m_readback, 1, &region);
            });
        wait(f);
        std::vector<float> out(bytes / sizeof(float));
        void * mapped = nullptr;
        vkMapMemory(m_dev, m_readbackMemory, 0, bytes, 0, &mapped);
        std::memcpy(out.data(), mapped, bytes); // HOST_COHERENT
        vkUnmapMemory(m_dev, m_readbackMemory);
        return out;
    }

  private:
    void ensureReadback(VkDeviceSize bytes)
    {
        if (m_readback != VK_NULL_HANDLE && bytes <= m_readbackSize)
        {
            return;
        }
        if (m_readback != VK_NULL_HANDLE)
        {
            vkDestroyBuffer(m_dev, m_readback, nullptr);
            vkFreeMemory(m_dev, m_readbackMemory, nullptr);
        }
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = bytes;
        bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        vkCreateBuffer(m_dev, &bi, nullptr, &m_readback);
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(m_dev, m_readback, &req);
        VkPhysicalDeviceMemoryProperties props;
        vkGetPhysicalDeviceMemoryProperties(m_r.physicalDevice(), &props);
        const VkMemoryPropertyFlags want =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        uint32_t type = UINT32_MAX;
        for (uint32_t i = 0; i < props.memoryTypeCount; ++i)
        {
            if ((req.memoryTypeBits & (1u << i))
                && (props.memoryTypes[i].propertyFlags & want) == want)
            {
                type = i;
                break;
            }
        }
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = type;
        vkAllocateMemory(m_dev, &ai, nullptr, &m_readbackMemory);
        vkBindBufferMemory(m_dev, m_readback, m_readbackMemory, 0);
        m_readbackSize = bytes;
    }

    Renderer & m_r;
    VkDevice m_dev;
    VkCommandPool m_pool = VK_NULL_HANDLE;
    std::vector<VkFence> m_fences;
    VkBuffer m_readback = VK_NULL_HANDLE;
    VkDeviceMemory m_readbackMemory = VK_NULL_HANDLE;
    VkDeviceSize m_readbackSize = 0;
};

bool throws(const std::function<void()> & fn)
{
    try
    {
        fn();
    }
    catch (const std::exception &)
    {
        return true;
    }
    return false;
}

bool sameHandles(const CanonicalGaussianBufferViews & a, const CanonicalGaussianBufferViews & b)
{
    return a.means.buffer == b.means.buffer && a.scales.buffer == b.scales.buffer
           && a.rotations.buffer == b.rotations.buffer && a.opacity.buffer == b.opacity.buffer
           && a.shCoefficients.buffer == b.shCoefficients.buffer;
}

bool noHandleShared(const CanonicalGaussianBufferViews & a, const CanonicalGaussianBufferViews & b)
{
    return a.means.buffer != b.means.buffer && a.scales.buffer != b.scales.buffer
           && a.rotations.buffer != b.rotations.buffer && a.opacity.buffer != b.opacity.buffer
           && a.shCoefficients.buffer != b.shCoefficients.buffer;
}

} // namespace

int main()
{
    try
    {
        Renderer r(kWidth, kHeight);
        if (!r.initialize({}, true, SHADER_DIR))
        {
            std::printf("[FAIL] renderer initialize\n");
            return 1;
        }
        if (!r.backwardSupported())
        {
            std::printf("[SKIP] device lacks fp32 buffer atomics; backward unavailable\n");
            r.shutdown();
            return 0;
        }
        r.setCamera(makeCamera());

        const Scene scene = makeScene();
        const std::vector<float> dLdImage = makeImageGradient();

        // References: fresh scenes of 5, 9 and 12 Gaussians.
        r.setGaussians(prefixInput(scene, 5));
        const Result ref5 = renderAndDifferentiate(r, dLdImage);
        r.setGaussians(prefixInput(scene, 9));
        const Result ref9 = renderAndDifferentiate(r, dLdImage);
        r.setGaussians(prefixInput(scene, kTotal));
        const Result ref12 = renderAndDifferentiate(r, dLdImage);
        {
            // Degeneracy guard: the three references must actually differ, or
            // "matches the fresh scene" would prove nothing about the tail.
            check(!imagesIdentical(ref5.image, ref9.image)
                      && !imagesIdentical(ref9.image, ref12.image),
                  "references differ (the added Gaussians are visible)");
        }

        // Destroyed before r.shutdown(): its pool/fences are children of
        // Sabercat's device (Sabercat owns the device -- the borrower destroys its
        // own objects first).
        auto extOwner = std::make_unique<ExternalQueueUser>(r);
        ExternalQueueUser & ext = *extOwner;

        // ---- Start from 5 Gaussians with exact capacity. ----
        r.setGaussians(prefixInput(scene, 5));
        check(r.gaussianCount() == 5 && r.gaussianCapacity() == 5, "initial count == capacity == 5");
        const uint64_t gen0 = r.storageGeneration();

        // 5. Gradient views before any backward.
        const CanonicalGaussianGradientViews earlyGrad = r.gradientViews();
        check(earlyGrad.means.valid() && earlyGrad.gaussianCount == 5,
              "gradient views available before the first backward");
        check(r.storageGeneration() > gen0, "allocating gradient buffers advances the generation");

        // 1. reserve: grow, handles replaced, prefix preserved.
        const CanonicalGaussianBufferViews before = r.gaussianViews();
        const uint64_t genBefore = r.storageGeneration();
        check(!r.reserveGaussians(5) && !r.reserveGaussians(3), "non-growing reserve is a no-op");
        check(sameHandles(before, r.gaussianViews()) && r.storageGeneration() == genBefore,
              "no-op reserve keeps handles and generation");
        check(r.reserveGaussians(9), "reserve(9) grows");
        const CanonicalGaussianBufferViews grown = r.gaussianViews();
        check(noHandleShared(before, grown), "growing reserve replaces every parameter handle");
        check(r.storageGeneration() > genBefore, "growing reserve advances the generation");
        check(r.gaussianCount() == 5 && r.gaussianCapacity() == 9 && grown.gaussianCount == 5,
              "count unchanged, capacity 9");
        check(grown.means.size == 9 * kMeansStride
                  && grown.shCoefficients.size == 9ull * kCoeffs * kShStride
                  && grown.opacity.size == 9 * kOpacityStride,
              "views span the full capacity");
        {
            const CanonicalGaussianGradientViews g = r.gradientViews();
            check(g.means.buffer != earlyGrad.means.buffer && g.means.size == 9 * kMeansStride,
                  "existing gradient buffers reallocated at the new capacity");
        }
        {
            Scene back(5);
            GaussianHostOutput out;
            out.count = 5;
            out.means = back.means.data();
            out.scales = back.scales.data();
            out.rotations = back.rotations.data();
            out.opacities = back.opacities.data();
            out.shCoefficients = back.sh.data();
            r.readGaussians(out);
            const bool exact =
                std::memcmp(back.means.data(), scene.means.data(), back.means.size() * 4) == 0
                && std::memcmp(back.scales.data(), scene.scales.data(), back.scales.size() * 4) == 0
                && std::memcmp(back.rotations.data(), scene.rotations.data(),
                               back.rotations.size() * 4)
                       == 0
                && std::memcmp(back.opacities.data(), scene.opacities.data(),
                               back.opacities.size() * 4)
                       == 0
                && std::memcmp(back.sh.data(), scene.sh.data(), back.sh.size() * 4) == 0;
            check(exact, "reserve preserved the 5-Gaussian prefix bit-exactly");
        }

        // 2 + 3 + 4. Densify 5 -> 9: write the tail on the shared queue
        // (no wait), raise the count, render.
        r.forward(RenderMode::Differentiable); // a forward at 5, then resize
        const uint64_t genGrown = r.storageGeneration();
        ext.writeTail(grown, scene, 5, 9); // deliberately not waited on
        r.setGaussianCount(9);
        check(sameHandles(grown, r.gaussianViews()) && r.storageGeneration() == genGrown,
              "setGaussianCount keeps handles and generation");
        check(r.gaussianViews().gaussianCount == 9, "views report the new count");
        check(throws([&] { r.backward(dLdImage.data()); }),
              "backward refuses a forward from before setGaussianCount");
        const Result got9 = renderAndDifferentiate(r, dLdImage);
        compareTo(got9, ref9, "grow 5->9 via tail write");

        // Outbound half of the contract: the optimizer's very next submission
        // after backward() -- no Sabercat call in between, no barrier --
        // copies the gradients out. Then compare with readGradients().
        {
            const CanonicalGaussianGradientViews g = r.gradientViews();
            r.forward(RenderMode::Differentiable);
            r.backward(dLdImage.data());
            const std::vector<float> opac = ext.readBack(g.opacity, 9 * sizeof(float));
            const GaussianGradients host = r.readGradients();
            check(relativeError(opac, host.opacities) <= 1e-5,
                  "gradients read on the shared queue right after backward() are complete");
        }

        // Grow past capacity: reserve again (geometric), write, raise.
        check(r.reserveGaussians(18), "reserve(18) grows");
        const CanonicalGaussianBufferViews grown2 = r.gaussianViews();
        ext.writeTail(grown2, scene, 9, kTotal);
        r.setGaussianCount(kTotal);
        compareTo(renderAndDifferentiate(r, dLdImage), ref12, "grow 9->12 after second reserve");

        // Prune to a prefix: shrinking never touches handles or contents.
        r.setGaussianCount(5);
        check(sameHandles(grown2, r.gaussianViews()) && r.gaussianCapacity() == 18,
              "shrink keeps handles and capacity");
        compareTo(renderAndDifferentiate(r, dLdImage), ref5, "shrink 12->5");
        // ... and growing back over the untouched tail restores the image.
        r.setGaussianCount(9);
        compareTo(renderAndDifferentiate(r, dLdImage), ref9, "regrow 5->9 over kept tail");

        // A scene created with spare capacity from the start.
        r.setGaussians(prefixInput(scene, 5, 16));
        check(r.gaussianCount() == 5 && r.gaussianCapacity() == 16,
              "setGaussians honours an up-front capacity");
        compareTo(renderAndDifferentiate(r, dLdImage), ref5, "up-front capacity, count 5");
        ext.writeTail(r.gaussianViews(), scene, 5, kTotal);
        r.setGaussianCount(kTotal);
        compareTo(renderAndDifferentiate(r, dLdImage), ref12, "up-front capacity, grow to 12");

        // Bounds.
        check(throws([&] { r.setGaussianCount(0); }), "setGaussianCount(0) throws");
        check(throws([&] { r.setGaussianCount(17); }), "setGaussianCount beyond capacity throws");
        check(r.gaussianCount() == kTotal, "a rejected count leaves the count alone");

        extOwner.reset();
        r.shutdown();
    }
    catch (const std::exception & e)
    {
        std::printf("[FAIL] exception: %s\n", e.what());
        return 1;
    }

    std::printf("\n%s (%d failure%s)\n", g_failures == 0 ? "ALL PASSED" : "FAILED", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}
