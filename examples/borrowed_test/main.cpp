// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Borrowed (application-owned) device: the application creates the Vulkan device and
// the canonical Gaussian buffers, and Sabercat renders and differentiates
// them in place.
//
// This program plays the application: it creates its own instance (with its
// own validation messenger) and device from DeviceRequirements, and its own
// parameter and gradient buffers, then checks:
//
//  1. A renderer on the borrowed device, rendering the borrowed buffers,
//     produces an image bit-identical to a model-A renderer given the same
//     Gaussians, and gradients (written straight into the application's
//     gradient buffers) equal to it up to fp32 atomic order.
//  2. Several renderers share one device and one scene: a second renderer
//     at another image size renders the SAME buffers bit-identically to the
//     model-A renderer after setImageSize() to that size; and a renderer
//     borrowing the model-A renderer's own device renders its scene.
//  3. Densification rules: setGaussianCount() moves within the borrowed
//     capacity (a prefix renders like a fresh scene of that prefix);
//     reserveGaussians() refuses; re-borrowing larger buffers advances the
//     storage generation.
//  4. The synchronization contract holds for borrowed buffers: an
//     un-waited write on the compute queue before forward() is seen.
//  5. Every requirement Sabercat checks is enforced: each malformed context
//     or scene throws std::invalid_argument and leaves the renderer as it was.
//  6. setImageSize(): new imageView handle, last forward invalidated.
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

#include "sabercat/borrowed.h"
#include "sabercat/camera.h"
#include "sabercat/device.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/renderer.h"

using namespace sabercat;

namespace
{

constexpr uint32_t kWidth = 64;
constexpr uint32_t kHeight = 48;
constexpr uint32_t kSmallWidth = 40;
constexpr uint32_t kSmallHeight = 30;
constexpr uint32_t kTotal = 12;    // Gaussians in the scene
constexpr uint32_t kCapacity = 16; // what the application's buffers hold
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

// Runs `fn`, expecting std::invalid_argument; returns its message (empty if
// nothing or something else was thrown).
std::string invalidArgument(const std::function<void()> & fn)
{
    try
    {
        fn();
    }
    catch (const std::invalid_argument & e)
    {
        return e.what();
    }
    catch (const std::exception & e)
    {
        return std::string("<wrong exception type: ") + e.what() + ">";
    }
    return "";
}

void expectRejected(const std::function<void()> & fn, const std::string & needle,
                    const std::string & what)
{
    const std::string message = invalidArgument(fn);
    const bool ok = !message.empty() && message.find(needle) != std::string::npos;
    check(ok, what + (ok ? "" : " (got: \"" + message + "\")"));
}

bool throwsAnything(const std::function<void()> & fn)
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

// ---- The scene (same construction as resize_test) ----

struct Scene
{
    std::vector<float> means, scales, rotations, opacities, sh;

    explicit Scene(uint32_t n)
        : means(n * 4), scales(n * 4), rotations(n * 4), opacities(n), sh(n * kCoeffs * 4)
    {
    }
};

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

GaussianHostInput prefixInput(const Scene & s, uint32_t n)
{
    GaussianHostInput in;
    in.count = n;
    in.means = s.means.data();
    in.scales = s.scales.data();
    in.rotations = s.rotations.data();
    in.opacities = s.opacities.data();
    in.shCoefficients = s.sh.data();
    in.shDegree = 3;
    return in;
}

// Intrinsics scale with the image so both sizes frame the same view.
Camera makeCamera(uint32_t width, uint32_t height)
{
    Camera c;
    c.width = width;
    c.height = height;
    c.fx = c.fy = 52.0f * static_cast<float>(width) / static_cast<float>(kWidth);
    c.cx = 0.5f * (width - 1);
    c.cy = 0.5f * (height - 1);
    c.cameraToWorld = glm::mat4(1.0f);
    c.cameraToWorld[3] = glm::vec4(0.05f, -0.02f, 3.0f, 1.0f);
    return c;
}

std::vector<float> makeImageGradient(uint32_t width, uint32_t height)
{
    std::vector<float> g(static_cast<size_t>(width) * height * 4);
    uint32_t state = 12345u;
    for (float & v : g)
    {
        state = state * 1664525u + 1013904223u;
        v = static_cast<float>(state >> 8) / static_cast<float>(1u << 24) - 0.5f;
    }
    return g;
}

template <typename T>
double relativeError(const T * a, const T * b, size_t count)
{
    const float * pa = reinterpret_cast<const float *>(a);
    const float * pb = reinterpret_cast<const float *>(b);
    const size_t n = count * sizeof(T) / sizeof(float);
    double worst = 0.0, scale = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        worst = std::max(worst, std::fabs(static_cast<double>(pa[i]) - pb[i]));
        scale = std::max(scale, std::fabs(static_cast<double>(pb[i])));
    }
    return scale > 0.0 ? worst / scale : worst;
}

bool identical(const std::vector<float> & a, const std::vector<float> & b)
{
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// ---- The application: owns the instance, device and buffers ----

VKAPI_ATTR VkBool32 VKAPI_CALL appDebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                VkDebugUtilsMessageTypeFlagsEXT,
                                                const VkDebugUtilsMessengerCallbackDataEXT * data,
                                                void *)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
    {
        std::printf("[validation] %s\n", data->pMessage);
    }
    return VK_FALSE;
}

struct AppBuffer
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    uint8_t * mapped = nullptr;
};

// A canonical stream inside one of the application's buffers.
struct Slot
{
    VkDeviceSize offset;
    VkDeviceSize bytes;
};

class Application
{
  public:
    Application()
    {
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "borrowed_test";
        appInfo.apiVersion = kRequiredVulkanApiVersion;

        const char * layer = "VK_LAYER_KHRONOS_validation";
        const char * debugExt = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
        VkDebugUtilsMessengerCreateInfoEXT debugInfo{};
        debugInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debugInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                                    | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        debugInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                                | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                                | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        debugInfo.pfnUserCallback = appDebugCallback;

        VkInstanceCreateInfo ici{};
        ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo = &appInfo;
        ici.enabledLayerCount = 1;
        ici.ppEnabledLayerNames = &layer;
        ici.enabledExtensionCount = 1;
        ici.ppEnabledExtensionNames = &debugExt;
        ici.pNext = &debugInfo;
        if (vkCreateInstance(&ici, nullptr, &instance) != VK_SUCCESS)
        {
            throw std::runtime_error("app: vkCreateInstance");
        }
        auto createMessenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
        createMessenger(instance, &debugInfo, nullptr, &messenger);

        uint32_t count = 0;
        vkEnumeratePhysicalDevices(instance, &count, nullptr);
        std::vector<VkPhysicalDevice> gpus(count);
        vkEnumeratePhysicalDevices(instance, &count, gpus.data());
        gpu = gpus.at(0);
        for (VkPhysicalDevice candidate : gpus)
        {
            VkPhysicalDeviceProperties props;
            vkGetPhysicalDeviceProperties(candidate, &props);
            if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
            {
                gpu = candidate;
                break;
            }
        }
        vkGetPhysicalDeviceProperties(gpu, &properties);

        // The queue Sabercat will submit to: the first compute family.
        vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, nullptr);
        families.resize(count);
        vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, families.data());
        for (uint32_t i = 0; i < count; ++i)
        {
            if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
            {
                family = i;
                break;
            }
        }

        // What Sabercat asks of the device: the helper defined below.
        requirements = std::make_unique<DeviceRequirements>(
            gpu, DeviceRequirements::Options{DeviceRequirements::Need::Required,
                                             DeviceRequirements::Need::None});
        const float priority = 1.0f;
        VkDeviceQueueCreateInfo qci{};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = family;
        qci.queueCount = 1;
        qci.pQueuePriorities = &priority;
        VkDeviceCreateInfo dci{};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.pNext = requirements->featureChain();
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        dci.enabledExtensionCount = static_cast<uint32_t>(requirements->deviceExtensions().size());
        dci.ppEnabledExtensionNames = requirements->deviceExtensions().data();
        if (vkCreateDevice(gpu, &dci, nullptr, &device) != VK_SUCCESS)
        {
            throw std::runtime_error("app: vkCreateDevice");
        }
        vkGetDeviceQueue(device, family, 0, &queue);

        VkCommandPoolCreateInfo pci{};
        pci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        pci.queueFamilyIndex = family;
        vkCreateCommandPool(device, &pci, nullptr, &pool);
    }

    ~Application()
    {
        vkDeviceWaitIdle(device);
        for (VkFence f : fences)
        {
            vkDestroyFence(device, f, nullptr);
        }
        for (AppBuffer & b : buffers)
        {
            destroyBuffer(b);
        }
        vkDestroyCommandPool(device, pool, nullptr);
        vkDestroyDevice(device, nullptr);
        auto destroyMessenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
        destroyMessenger(instance, messenger, nullptr);
        vkDestroyInstance(instance, nullptr);
    }

    VulkanContext context() const
    {
        VulkanContext c;
        c.instance = instance;
        c.physicalDevice = gpu;
        c.device = device;
        c.computeQueue = queue;
        c.computeFamily = family;
        return c; // compute only: a headless borrower
    }

    // Host-visible, coherent, so the application reads/writes it directly.
    AppBuffer * createBuffer(VkDeviceSize size)
    {
        AppBuffer b;
        b.size = size;
        VkBufferCreateInfo bi{};
        bi.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bi.size = size;
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT
                   | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        vkCreateBuffer(device, &bi, nullptr, &b.buffer);
        VkMemoryRequirements req;
        vkGetBufferMemoryRequirements(device, b.buffer, &req);
        VkPhysicalDeviceMemoryProperties mem;
        vkGetPhysicalDeviceMemoryProperties(gpu, &mem);
        const VkMemoryPropertyFlags want =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        uint32_t type = UINT32_MAX;
        for (uint32_t i = 0; i < mem.memoryTypeCount; ++i)
        {
            if ((req.memoryTypeBits & (1u << i)) && (mem.memoryTypes[i].propertyFlags & want) == want)
            {
                type = i;
                break;
            }
        }
        VkMemoryAllocateInfo ai{};
        ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = type;
        vkAllocateMemory(device, &ai, nullptr, &b.memory);
        vkBindBufferMemory(device, b.buffer, b.memory, 0);
        void * mapped = nullptr;
        vkMapMemory(device, b.memory, 0, VK_WHOLE_SIZE, 0, &mapped);
        b.mapped = static_cast<uint8_t *>(mapped);
        std::memset(b.mapped, 0, size);
        buffers.push_back(b);
        return &buffers.back();
    }

    // Records `fn` and submits it to the shared queue WITHOUT waiting.
    template <typename F>
    void submitUnwaited(F && fn)
    {
        VkCommandBufferAllocateInfo alloc{};
        alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc.commandPool = pool;
        alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc.commandBufferCount = 1;
        VkCommandBuffer cmd;
        vkAllocateCommandBuffers(device, &alloc, &cmd);
        VkCommandBufferBeginInfo begin{};
        begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &begin);
        fn(cmd);
        vkEndCommandBuffer(cmd);
        VkFenceCreateInfo fi{};
        fi.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        VkFence fence;
        vkCreateFence(device, &fi, nullptr, &fence);
        fences.push_back(fence);
        VkSubmitInfo si{};
        si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        vkQueueSubmit(queue, 1, &si, fence);
    }

    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    VkPhysicalDevice gpu = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties properties{};
    std::vector<VkQueueFamilyProperties> families;
    uint32_t family = UINT32_MAX;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkCommandPool pool = VK_NULL_HANDLE;
    std::unique_ptr<DeviceRequirements> requirements;
    std::vector<VkFence> fences;
    std::vector<AppBuffer> buffers; // reserved up front so pointers stay valid

  private:
    void destroyBuffer(AppBuffer & b)
    {
        vkUnmapMemory(device, b.memory);
        vkDestroyBuffer(device, b.buffer, nullptr);
        vkFreeMemory(device, b.memory, nullptr);
    }
};

VkDeviceSize alignUp(VkDeviceSize v, VkDeviceSize a)
{
    return (v + a - 1) / a * a;
}

// Lays the five parameter streams (or six gradient streams) for `capacity`
// Gaussians out in ONE application buffer, each at an aligned offset -- the
// way an allocator-minded application packs them.
struct Layout
{
    std::vector<Slot> slots;
    VkDeviceSize total = 0;

    Layout(const std::vector<VkDeviceSize> & bytesPerGaussian, uint32_t capacity,
           VkDeviceSize alignment)
    {
        for (VkDeviceSize b : bytesPerGaussian)
        {
            slots.push_back({total, b * capacity});
            total = alignUp(total + b * capacity, alignment);
        }
    }
};

const std::vector<VkDeviceSize> kParamBytes = {16, 16, 16, 4, 16 * kCoeffs};
const std::vector<VkDeviceSize> kGradBytes = {16, 16, 16, 4, 16 * kCoeffs, 16};

BufferView viewOf(const AppBuffer & b, const Slot & s, uint32_t stride)
{
    return BufferView{b.buffer, s.offset, s.bytes, stride};
}

BorrowedGaussians describe(const AppBuffer & params, const Layout & pl, const AppBuffer * grads,
                           const Layout * gl, uint32_t count)
{
    BorrowedGaussians b;
    b.parameters.means = viewOf(params, pl.slots[0], 16);
    b.parameters.scales = viewOf(params, pl.slots[1], 16);
    b.parameters.rotations = viewOf(params, pl.slots[2], 16);
    b.parameters.opacity = viewOf(params, pl.slots[3], 4);
    b.parameters.shCoefficients = viewOf(params, pl.slots[4], 16);
    b.parameters.gaussianCount = count;
    b.parameters.shDegree = 3;
    if (grads)
    {
        b.gradients.means = viewOf(*grads, gl->slots[0], 16);
        b.gradients.scales = viewOf(*grads, gl->slots[1], 16);
        b.gradients.rotations = viewOf(*grads, gl->slots[2], 16);
        b.gradients.opacity = viewOf(*grads, gl->slots[3], 4);
        b.gradients.shCoefficients = viewOf(*grads, gl->slots[4], 16);
        b.gradients.screenMeans = viewOf(*grads, gl->slots[5], 16);
    }
    return b;
}

void writeScene(AppBuffer & params, const Layout & pl, const Scene & s, uint32_t n)
{
    std::memcpy(params.mapped + pl.slots[0].offset, s.means.data(), n * 16);
    std::memcpy(params.mapped + pl.slots[1].offset, s.scales.data(), n * 16);
    std::memcpy(params.mapped + pl.slots[2].offset, s.rotations.data(), n * 16);
    std::memcpy(params.mapped + pl.slots[3].offset, s.opacities.data(), n * 4);
    std::memcpy(params.mapped + pl.slots[4].offset, s.sh.data(), n * 16 * kCoeffs);
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

double worstGradientError(const GaussianGradients & a, const GaussianGradients & b)
{
    return std::max({relativeError(a.means.data(), b.means.data(), a.means.size()),
                     relativeError(a.scales.data(), b.scales.data(), a.scales.size()),
                     relativeError(a.rotations.data(), b.rotations.data(), a.rotations.size()),
                     relativeError(a.opacities.data(), b.opacities.data(), a.opacities.size()),
                     relativeError(a.shCoefficients.data(), b.shCoefficients.data(),
                                   a.shCoefficients.size()),
                     relativeError(a.screenMeans.data(), b.screenMeans.data(),
                                   a.screenMeans.size())});
}

std::string fmt(const char * f, double v)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), f, v);
    return buf;
}

} // namespace

int main()
{
    try
    {
        const Scene scene = makeScene();
        const std::vector<float> dLdImage = makeImageGradient(kWidth, kHeight);
        const std::vector<float> dLdSmall = makeImageGradient(kSmallWidth, kSmallHeight);

        // ---- Model A reference: Sabercat owns everything. ----
        Renderer ref(kWidth, kHeight);
        if (!ref.initialize({}, true, SHADER_DIR))
        {
            std::printf("[FAIL] reference renderer initialize\n");
            return 1;
        }
        if (!ref.backwardSupported())
        {
            std::printf("[SKIP] device lacks fp32 buffer atomics\n");
            ref.shutdown();
            return 0;
        }
        ref.setCamera(makeCamera(kWidth, kHeight));
        ref.setGaussians(prefixInput(scene, 5));
        const Result ref5 = renderAndDifferentiate(ref, dLdImage);
        ref.setGaussians(prefixInput(scene, kTotal));
        const Result ref12 = renderAndDifferentiate(ref, dLdImage);
        check(ref.ownsDevice() && !ref.gaussiansBorrowed(), "model A: owns its device and scene");

        // ---- The application (owns the borrowed device). ----
        auto app = std::make_unique<Application>();
        app->buffers.reserve(16);
        const VkDeviceSize align =
            std::max<VkDeviceSize>(app->properties.limits.minStorageBufferOffsetAlignment, 16);
        const Layout pl(kParamBytes, kCapacity, align);
        const Layout gl(kGradBytes, kCapacity, align);
        AppBuffer & params = *app->createBuffer(pl.total);
        AppBuffer & grads = *app->createBuffer(gl.total);
        writeScene(params, pl, scene, kTotal);

        // 1. Borrowed device + borrowed buffers == Sabercat-owned device.
        {
            const EnabledDeviceFeatures enabled = app->requirements->enabled();
            check(enabled.apiVersion == kRequiredVulkanApiVersion
                      && enabled.shaderBufferFloat32AtomicAdd && !enabled.swapchain,
                  "DeviceRequirements: training enabled, no presentation, Vulkan 1.3");
        }
        Renderer b(kWidth, kHeight);
        b.initializeBorrowed(app->context(), app->requirements->enabled(), SHADER_DIR);
        check(!b.ownsDevice() && b.device() == app->device && b.computeQueue() == app->queue,
              "borrowed renderer runs on the application's device and queue");
        check(b.backwardSupported(), "declared fp32 atomics enable backward()");
        b.setCamera(makeCamera(kWidth, kHeight));
        b.borrowGaussians(describe(params, pl, &grads, &gl, kTotal));
        check(b.gaussiansBorrowed() && b.gaussianCount() == kTotal && b.gaussianCapacity() == kCapacity,
              "borrowGaussians: count 12, capacity 16 (what the buffers hold)");
        {
            const CanonicalGaussianBufferViews v = b.gaussianViews();
            const CanonicalGaussianGradientViews g = b.gradientViews();
            check(v.means.buffer == params.buffer && v.shCoefficients.offset == pl.slots[4].offset
                      && g.opacity.buffer == grads.buffer && g.screenMeans.offset == gl.slots[5].offset,
                  "gaussianViews()/gradientViews() are the borrowed views");
        }
        const Result got = renderAndDifferentiate(b, dLdImage);
        check(identical(got.image, ref12.image), "borrowed render bit-identical to model A");
        check(worstGradientError(got.gradients, ref12.gradients) <= 1e-5,
              "borrowed gradients match model A " + fmt("(worst rel. %.3g, tol 1e-5)",
                                                         worstGradientError(got.gradients, ref12.gradients)));
        {
            // backward() wrote straight into the application's memory.
            std::vector<float> opac(kTotal);
            std::memcpy(opac.data(), grads.mapped + gl.slots[3].offset, kTotal * 4);
            check(std::memcmp(opac.data(), got.gradients.opacities.data(), kTotal * 4) == 0,
                  "gradients land in the application's own buffer");
        }

        // 2a. A second renderer, another image size, the SAME buffers.
        Renderer small(kSmallWidth, kSmallHeight);
        small.initializeBorrowed(app->context(), app->requirements->enabled(), SHADER_DIR);
        small.setCamera(makeCamera(kSmallWidth, kSmallHeight));
        small.borrowGaussians(describe(params, pl, nullptr, nullptr, kTotal));
        small.forward(RenderMode::Differentiable);
        const std::vector<float> smallImage = small.readImage();
        // Model A at that size via setImageSize (6).
        const BufferView bigView = [&]
        {
            ref.forward(RenderMode::Differentiable);
            return ref.imageView();
        }();
        ref.setImageSize(kSmallWidth, kSmallHeight);
        check(throwsAnything([&] { ref.imageView(); }) && throwsAnything([&] { ref.readImage(); }),
              "setImageSize invalidates the last forward (imageView/readImage throw)");
        ref.setCamera(makeCamera(kSmallWidth, kSmallHeight));
        ref.forward(RenderMode::Differentiable);
        check(ref.imageView().buffer != bigView.buffer
                  && ref.imageView().size == kSmallWidth * kSmallHeight * 16ull,
              "setImageSize gives a new image view of the new size");
        check(ref.width() == kSmallWidth && ref.height() == kSmallHeight, "setImageSize updates width/height");
        check(identical(smallImage, ref.readImage()),
              "second renderer (40x30) on the same borrowed buffers == model A after setImageSize");
        {
            // Its gradients are Sabercat-allocated (none borrowed) and match.
            small.backward(dLdSmall.data());
            ref.backward(dLdSmall.data());
            const double err = worstGradientError(small.readGradients(), ref.readGradients());
            check(small.gradientViews().means.buffer != grads.buffer && err <= 1e-5,
                  "unborrowed gradients are Sabercat-allocated and match " + fmt("(%.3g)", err));
        }
        check(!throwsAnything([&] { ref.setImageSize(kSmallWidth, kSmallHeight); }),
              "setImageSize to the same size is a no-op");
        ref.setImageSize(kWidth, kHeight);
        ref.setCamera(makeCamera(kWidth, kHeight));

        // 2b. A renderer borrowing the MODEL A renderer's device and scene.
        {
            Renderer share(kWidth, kHeight);
            share.initializeBorrowed(ref.vulkanContext(), ref.enabledDeviceFeatures(), SHADER_DIR);
            share.setCamera(makeCamera(kWidth, kHeight));
            BorrowedGaussians bg;
            bg.parameters = ref.gaussianViews();
            share.borrowGaussians(bg);
            share.forward(RenderMode::Differentiable);
            check(share.device() == ref.device() && identical(share.readImage(), ref12.image),
                  "renderer on a model-A renderer's device renders its scene bit-identically");
            share.shutdown(); // before the owner's
        }

        // 2c. A borrowed renderer with a depth output, its gradients written
        // into the application's buffers, against model A with the same layout.
        {
            Renderer bd(kWidth, kHeight, DepthLayout::DepthCoverage);
            bd.initializeBorrowed(app->context(), app->requirements->enabled(), SHADER_DIR);
            bd.setCamera(makeCamera(kWidth, kHeight));
            bd.borrowGaussians(describe(params, pl, &grads, &gl, kTotal));
            bd.forward(RenderMode::Differentiable);
            const std::vector<float> image = bd.readImage();
            const std::vector<float> depth = bd.readDepth();
            bd.backward(image.data(), depth.data());
            const GaussianGradients withDepth = bd.readGradients();
            bd.backward(image.data());
            const GaussianGradients colorOnly = bd.readGradients();
            bd.shutdown(); // before the application's device goes

            Renderer refD(kWidth, kHeight, DepthLayout::DepthCoverage);
            if (!refD.initialize({}, true, SHADER_DIR))
            {
                std::printf("[FAIL] depth reference renderer initialize\n");
                return 1;
            }
            refD.setCamera(makeCamera(kWidth, kHeight));
            refD.setGaussians(prefixInput(scene, kTotal));
            refD.forward(RenderMode::Differentiable);
            const std::vector<float> refImage = refD.readImage();
            const std::vector<float> refDepth = refD.readDepth();
            refD.backward(refImage.data(), refDepth.data());
            const GaussianGradients refWithDepth = refD.readGradients();
            refD.shutdown();

            const double err = worstGradientError(withDepth, refWithDepth);
            const double depthEffect = worstGradientError(withDepth, colorOnly);
            check(identical(image, refImage) && err <= 1e-5,
                  "borrowed depth gradient == model A with DepthCoverage " + fmt("(worst rel. %.3g, tol 1e-5)", err));
            check(depthEffect > 1e-2,
                  "borrowed depth gradient differs from the color-only gradient " + fmt("(worst rel. %.3g)", depthEffect));
        }

        // 3. Densification rules on borrowed storage.
        {
            const uint64_t gen = b.storageGeneration();
            b.setGaussianCount(5);
            check(b.storageGeneration() == gen && b.gaussianViews().means.buffer == params.buffer,
                  "setGaussianCount within the borrowed capacity keeps handles and generation");
            const Result got5 = renderAndDifferentiate(b, dLdImage);
            check(identical(got5.image, ref5.image)
                      && worstGradientError(got5.gradients, ref5.gradients) <= 1e-5,
                  "count 5 over borrowed storage == fresh 5-Gaussian scene");
            check(throwsAnything([&] { b.setGaussianCount(kCapacity + 1); }),
                  "setGaussianCount beyond the borrowed capacity throws");
            check(throwsAnything([&] { b.reserveGaussians(64); }) && b.gaussianCapacity() == kCapacity,
                  "reserveGaussians refuses on borrowed storage");
            b.setGaussianCount(kTotal);

            // The owner grows: bigger buffers, copy, borrow again.
            const Layout pl2(kParamBytes, 2 * kCapacity, align);
            AppBuffer & params2 = *app->createBuffer(pl2.total);
            writeScene(params2, pl2, scene, kTotal);
            b.borrowGaussians(describe(params2, pl2, nullptr, nullptr, kTotal));
            check(b.storageGeneration() > gen && b.gaussianCapacity() == 2 * kCapacity
                      && b.gaussianViews().means.buffer == params2.buffer,
                  "re-borrowing larger buffers: new handles, capacity 32, generation advances");
            b.forward(RenderMode::Differentiable);
            check(identical(b.readImage(), ref12.image), "re-borrowed scene renders the same");
            b.borrowGaussians(describe(params, pl, &grads, &gl, kTotal));
        }

        // 4. Synchronization: the application rewrites Gaussian 0's opacity
        // on the shared queue, NOT waited on, then renders.
        {
            ref.setGaussians(prefixInput(scene, kTotal));
            Scene changed = scene;
            changed.opacities[0] = 0.95f;
            ref.setGaussians(prefixInput(changed, kTotal));
            ref.forward(RenderMode::Differentiable);
            const std::vector<float> want = ref.readImage();

            const float value = 0.95f;
            app->submitUnwaited([&](VkCommandBuffer cmd)
                                { vkCmdUpdateBuffer(cmd, params.buffer, pl.slots[3].offset, 4, &value); });
            b.forward(RenderMode::Differentiable);
            check(!identical(want, ref12.image) && identical(b.readImage(), want),
                  "an un-waited write on the compute queue is seen by the next forward()");
            std::memcpy(params.mapped + pl.slots[3].offset, scene.opacities.data(), 4);
            ref.setGaussians(prefixInput(scene, kTotal));
        }

        // 5a. Malformed scenes: rejected, and the current scene survives.
        {
            const BorrowedGaussians good = describe(params, pl, &grads, &gl, kTotal);
            const auto rejects = [&](BorrowedGaussians bad, const std::string & needle,
                                     const std::string & what)
            {
                const uint64_t gen = b.storageGeneration();
                expectRejected([&] { b.borrowGaussians(bad); }, needle, "rejects " + what);
                if (b.storageGeneration() != gen || b.gaussianViews().means.buffer != params.buffer)
                {
                    check(false, "a rejected scene changed the renderer (" + what + ")");
                }
            };
            BorrowedGaussians bad = good;
            bad.parameters.means.buffer = VK_NULL_HANDLE;
            rejects(bad, "parameters.means: buffer is VK_NULL_HANDLE", "a missing parameter stream");
            bad = good;
            bad.parameters.scales.stride = 12;
            rejects(bad, "parameters.scales: stride 12", "a non-canonical stride");
            bad = good;
            bad.parameters.opacity.offset += 4;
            bad.parameters.opacity.size -= 4;
            rejects(bad, "minStorageBufferOffsetAlignment", "a misaligned offset");
            bad = good;
            bad.parameters.shCoefficients.size = pl.total; // runs past the buffer end
            rejects(bad, "runs past the end of its buffer", "a range past the buffer's end");
            bad = good;
            bad.parameters.rotations.size = 8;
            rejects(bad, "holds no whole Gaussian", "a stream smaller than one Gaussian");
            bad = good;
            bad.parameters.gaussianCount = kCapacity + 1;
            rejects(bad, "outside [1, capacity", "a count above the capacity");
            bad = good;
            bad.parameters.gaussianCount = 0;
            rejects(bad, "outside [1, capacity", "a zero count");
            bad = good;
            bad.parameters.shDegree = 4;
            rejects(bad, "shDegree 4 exceeds 3", "SH degree 4");
            bad = good;
            bad.gradients.rotations.buffer = VK_NULL_HANDLE;
            rejects(bad, "5 of 6 streams given", "partial gradient views");
            bad = good;
            bad.gradients.opacity.size = 4 * (kCapacity - 1);
            rejects(bad, "fewer than the parameter capacity", "a gradient stream below capacity");
            bad = good;
            bad.parameters.scales.offset = bad.parameters.means.offset;
            rejects(bad, "parameters.means and parameters.scales overlap", "overlapping parameter streams");
            bad = good;
            bad.gradients.means = bad.parameters.means;
            rejects(bad, "parameters.means and gradients.means overlap",
                    "gradients aliasing parameters");
            bad = good;
            bad.parameters.confidence = BufferView{params.buffer, 0, 64, 16};
            rejects(bad, "parameters.confidence: stride 16", "a malformed optional confidence stream");
            b.forward(RenderMode::Differentiable);
            check(identical(b.readImage(), ref12.image), "after every rejection the scene still renders");
        }

        // 5b. Malformed contexts and declarations.
        {
            const VulkanContext good = app->context();
            const EnabledDeviceFeatures enabled = app->requirements->enabled();
            const auto rejects = [&](const VulkanContext & c, const EnabledDeviceFeatures & e,
                                     const std::string & needle, const std::string & what)
            {
                Renderer r(8, 8);
                expectRejected([&] { r.initializeBorrowed(c, e, SHADER_DIR); }, needle,
                               "rejects " + what);
            };
            VulkanContext c = good;
            c.device = VK_NULL_HANDLE;
            rejects(c, enabled, "device is VK_NULL_HANDLE", "a null device");
            c = good;
            c.computeQueue = VK_NULL_HANDLE;
            rejects(c, enabled, "neither a compute nor a graphics queue", "a context with no queue");
            c = good;
            c.computeFamily = static_cast<uint32_t>(app->families.size());
            rejects(c, enabled, "out of range", "a queue family out of range");
            c = good;
            c.computeFamily = UINT32_MAX;
            rejects(c, enabled, "without its family index", "a queue without its family");
            for (uint32_t f = 0; f < app->families.size(); ++f)
            {
                if (!(app->families[f].queueFlags & VK_QUEUE_COMPUTE_BIT))
                {
                    c = good;
                    c.computeFamily = f;
                    rejects(c, enabled, "lacks VK_QUEUE_COMPUTE_BIT",
                            "a compute queue on a family without compute");
                    break;
                }
            }
            EnabledDeviceFeatures e = enabled;
            e.apiVersion = VK_API_VERSION_1_2;
            rejects(good, e, "declared apiVersion 1.2.0", "a declared Vulkan 1.2");
            e = enabled;
            e.swapchain = true;
            rejects(good, e, "no graphics queue", "a swapchain without a graphics queue");
            c = good;
            if (app->families[app->family].queueFlags & VK_QUEUE_GRAPHICS_BIT)
            {
                c.graphicsQueue = app->queue;
                c.graphicsFamily = app->family;
                rejects(c, e, "VK_KHR_swapchain is not enabled on the device",
                        "a declared swapchain the device does not have");
            }

            // Undeclared atomics: forward works, backward refuses.
            Renderer noAtomics(kWidth, kHeight);
            e = enabled;
            e.shaderBufferFloat32AtomicAdd = false;
            noAtomics.initializeBorrowed(good, e, SHADER_DIR);
            noAtomics.setCamera(makeCamera(kWidth, kHeight));
            noAtomics.borrowGaussians(describe(params, pl, nullptr, nullptr, kTotal));
            noAtomics.forward(RenderMode::Differentiable);
            check(!noAtomics.backwardSupported() && identical(noAtomics.readImage(), ref12.image)
                      && throwsAnything([&] { noAtomics.backward(dLdImage.data()); }),
                  "undeclared fp32 atomics: forward renders, backward refuses");
            check(throwsAnything([&] { noAtomics.attachSurface(VK_NULL_HANDLE, 8, 8, [] {}); })
                      && throwsAnything([&] { noAtomics.setUiRenderCallback([](VkCommandBuffer) {}); }),
                  "headless borrowed device: attachSurface and the UI callback refuse");
            noAtomics.shutdown();
        }

        small.shutdown();
        b.shutdown();
        app.reset(); // the owner destroys its device after every borrower
        ref.shutdown();
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
