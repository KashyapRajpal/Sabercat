// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The zero-copy image view (Renderer::imageView()), driven the way
// an optimizer computes its loss on the GPU.
//
//  1. After a Differentiable forward, the view's contents -- copied out by a
//     command buffer submitted to computeQueue() straight after forward(),
//     with no wait or barrier -- are bit-identical to readImage(), and to
//     the image-path readImage() of a Viewer forward of the same frame.
//  2. GPU loss path == host path: the image buffer is copied into a
//     optimizer-owned dL/dImage buffer on the shared queue (not waited on),
//     then backward(BufferView); the gradients must equal backward(host
//     pointer to the same values) up to fp32 atomic order.
//  3. imageView() throws before any forward, after a Viewer forward, after
//     setGaussianCount and after a scene replacement; the handle is stable
//     across forwards.
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
constexpr uint32_t kCount = 12;
constexpr uint32_t kCoeffs = 16;
constexpr VkDeviceSize kImageBytes = static_cast<VkDeviceSize>(kWidth) * kHeight * 16;

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
// non-zero (same construction as resize_test).
Scene makeScene()
{
    Scene s(kCount);
    for (uint32_t i = 0; i < kCount; ++i)
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

GaussianHostInput sceneInput(const Scene & s)
{
    GaussianHostInput in;
    in.count = kCount;
    in.means = s.means.data();
    in.scales = s.scales.data();
    in.rotations = s.rotations.data();
    in.opacities = s.opacities.data();
    in.shCoefficients = s.sh.data();
    in.shDegree = 3;
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

// Worst |a-b| over a stream, relative to the stream's largest magnitude.
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

bool identical(const std::vector<float> & a, const std::vector<float> & b)
{
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// The optimizer's side of the seam: its own command pool on Sabercat's compute
// queue. Submissions are deliberately NOT waited on before the next Sabercat
// call -- the synchronization contract says that is unnecessary.
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
        for (const Allocation & a : m_buffers)
        {
            vkDestroyBuffer(m_dev, a.buffer, nullptr);
            vkFreeMemory(m_dev, a.memory, nullptr);
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

    // An optimizer-owned buffer: host-visible readback target, or a
    // device-local storage buffer (e.g. the optimizer's dL/dImage).
    BufferView createBuffer(VkDeviceSize bytes, VkBufferUsageFlags usage,
                            VkMemoryPropertyFlags want)
    {
        Allocation a;
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = bytes;
        bi.usage = usage;
        vkCreateBuffer(m_dev, &bi, nullptr, &a.buffer);
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(m_dev, a.buffer, &req);
        VkPhysicalDeviceMemoryProperties props;
        vkGetPhysicalDeviceMemoryProperties(m_r.physicalDevice(), &props);
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
        if (type == UINT32_MAX)
        {
            throw std::runtime_error("no suitable memory type");
        }
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = type;
        vkAllocateMemory(m_dev, &ai, nullptr, &a.memory);
        vkBindBufferMemory(m_dev, a.buffer, a.memory, 0);
        m_buffers.push_back(a);

        BufferView view;
        view.buffer = a.buffer;
        view.size = bytes;
        view.stride = 16;
        return view;
    }

    // Copies `src` into `dst` with no barrier of its own; not waited on.
    VkFence copy(const BufferView & src, const BufferView & dst, VkDeviceSize bytes)
    {
        return submit(
            [&](VkCommandBuffer cmd)
            {
                VkBufferCopy region{src.offset, dst.offset, bytes};
                vkCmdCopyBuffer(cmd, src.buffer, dst.buffer, 1, &region);
            });
    }

    // Copies the first `bytes` of `view` to the host. The copy is submitted
    // with no barrier; only its own completion is waited on.
    std::vector<float> readBack(const BufferView & view, VkDeviceSize bytes)
    {
        if (m_readback.buffer == VK_NULL_HANDLE)
        {
            m_readback = createBuffer(kImageBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                                          | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            m_readbackMemory = m_buffers.back().memory;
        }
        VkFence f = copy(view, m_readback, bytes);
        vkWaitForFences(m_dev, 1, &f, VK_TRUE, UINT64_MAX);
        std::vector<float> out(bytes / sizeof(float));
        void * mapped = nullptr;
        vkMapMemory(m_dev, m_readbackMemory, 0, bytes, 0, &mapped);
        std::memcpy(out.data(), mapped, bytes); // HOST_COHERENT
        vkUnmapMemory(m_dev, m_readbackMemory);
        return out;
    }

  private:
    struct Allocation
    {
        VkBuffer buffer = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
    };

    Renderer & m_r;
    VkDevice m_dev;
    VkCommandPool m_pool = VK_NULL_HANDLE;
    std::vector<VkFence> m_fences;
    std::vector<Allocation> m_buffers;
    BufferView m_readback;
    VkDeviceMemory m_readbackMemory = VK_NULL_HANDLE;
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
        r.setGaussians(sceneInput(scene));

        // 3a. Before any forward.
        check(throws([&] { r.imageView(); }), "imageView() throws before any forward");

        // Reference through the image path: readImage() after a Viewer
        // forward still downloads the storage image.
        r.forward(RenderMode::Viewer);
        const std::vector<float> viewerImage = r.readImage();
        check(throws([&] { r.imageView(); }), "imageView() throws after a Viewer forward");
        {
            // Degeneracy guard: an all-zero frame would make every
            // comparison below vacuous.
            const size_t covered =
                std::count_if(viewerImage.begin(), viewerImage.end(), [](float v) { return v != 0.0f; });
            check(covered > viewerImage.size() / 4, "reference frame has substantial coverage");
        }

        // Destroyed before r.shutdown(): its objects are children of
        // Sabercat's device.
        auto extOwner = std::make_unique<ExternalQueueUser>(r);
        ExternalQueueUser & ext = *extOwner;

        // 1. Differentiable forward, then the view read back on the shared
        // queue with no wait or barrier before the copy.
        r.forward(RenderMode::Differentiable);
        const BufferView view = r.imageView();
        check(view.valid() && view.offset == 0 && view.stride == 16 && view.size == kImageBytes,
              "imageView(): offset 0, stride 16, size width*height*16");
        const std::vector<float> fromView = ext.readBack(view, kImageBytes);
        const std::vector<float> fromRead = r.readImage();
        check(identical(fromView, fromRead), "image view bit-identical to readImage()");
        check(identical(fromView, viewerImage),
              "image view bit-identical to the image-path readImage() of a Viewer forward");

        // 3b. Handle stability.
        r.forward(RenderMode::Differentiable);
        check(r.imageView().buffer == view.buffer, "image view handle unchanged across forwards");

        // 2. GPU loss path vs host path. dL/dImage = the image itself
        // (L = 0.5 |I|^2), built on the GPU by a copy that is not waited on.
        {
            const BufferView dLdImage = ext.createBuffer(
                kImageBytes, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            ext.copy(r.imageView(), dLdImage, kImageBytes); // deliberately not waited on
            r.backward(dLdImage);
            const GaussianGradients gpu = r.readGradients();

            const std::vector<float> host = r.readImage();
            r.backward(host.data());
            const GaussianGradients ref = r.readGradients();

            bool nonZero = false;
            for (float v : ref.opacities)
            {
                nonZero |= v != 0.0f;
            }
            check(nonZero, "host-path gradients are non-trivial");
            const double err = worstGradientError(gpu, ref);
            char buf[160];
            std::snprintf(buf, sizeof(buf),
                          "GPU dL/dImage path == host path (worst rel. %.3g, tol 1e-5)", err);
            check(err <= 1e-5, buf);
        }

        // 3c. Invalidation.
        r.setGaussianCount(kCount - 1); // a prune
        check(throws([&] { r.imageView(); }), "imageView() throws after setGaussianCount");
        r.forward(RenderMode::Differentiable);
        check(!throws([&] { r.imageView(); }), "imageView() valid again after a new forward");
        r.setGaussians(sceneInput(scene));
        check(throws([&] { r.imageView(); }), "imageView() throws after a scene replacement");
        r.forward(RenderMode::Differentiable);
        check(r.imageView().buffer == view.buffer,
              "image view handle survives a scene replacement (only resize changes it)");

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
