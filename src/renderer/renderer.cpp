// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/renderer.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <string>

#include "sabercat/backward_pass.h"
#include "sabercat/borrowed.h"
#include "sabercat/buffer.h"
#include "sabercat/camera.h"
#include "sabercat/debug_visualizer.h"
#include "sabercat/device.h"
#include "sabercat/forward_pass.h"
#include "sabercat/gaussian_acceleration_structure.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/logger.h"
#include "sabercat/relight_pass.h"
#include "sabercat/shadow_pass.h"
#include "sabercat/render_config.h"
#include "sabercat/scene_loader.h"
#include "sabercat/visibility_pass.h"
#include "sabercat/vk_check.h"

#include "present_blit.h"
#include "surface_format.h"

namespace sabercat
{

namespace
{

// Rough defaults for the Depth/TileDensity debug views -- there's no scene-
// bounds tracking to derive a real near/far range from yet, so these are
// fixed "reasonable scene scale" constants, not measured. Revisit once a
// real scene makes them feel wrong.
constexpr float kDebugDepthScale = 50.0f;
constexpr float kDebugDensityScale = 64.0f;

// Names for the swapchain-format log line; other values print their number.
std::string surfaceFormatName(VkFormat format)
{
    switch (format)
    {
    case VK_FORMAT_B8G8R8A8_UNORM:
        return "B8G8R8A8_UNORM";
    case VK_FORMAT_R8G8B8A8_UNORM:
        return "R8G8B8A8_UNORM";
    case VK_FORMAT_B8G8R8A8_SRGB:
        return "B8G8R8A8_SRGB";
    case VK_FORMAT_R8G8B8A8_SRGB:
        return "R8G8B8A8_SRGB";
    default:
        return "VkFormat " + std::to_string(static_cast<int>(format));
    }
}

std::string colorSpaceName(VkColorSpaceKHR space)
{
    return space == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR
               ? "SRGB_NONLINEAR"
               : "VkColorSpaceKHR " + std::to_string(static_cast<int>(space));
}

} // namespace

class Renderer::Impl
{
  public:
    Device device;
    std::shared_ptr<GaussianCloud> gaussianCloud;

    // Built lazily in initialize() (only if a shaderDir was given) and torn
    // down explicitly in shutdown() -- see the comment there for why that
    // can't wait for Impl's destructor.
    std::unique_ptr<ForwardPass> forwardPass;
    std::unique_ptr<DebugVisualizer> debugVisualizer;
    // Built on the first lit forward().
    std::unique_ptr<RelightPass> relightPass;
    // Built on the first shadowed forward().
    std::unique_ptr<ShadowPass> shadowPass;
    // Built on the first bake / teacher call / BakedSH frame.
    std::unique_ptr<VisibilityPass> visibilityPass;
    BufferView externalVisibility; // ShadowMode::External's borrowed source
    CameraUniforms litUniforms{};  // the last lit forward()'s camera (teacher samples)

    // Shared Gaussian acceleration structure (shadows, bake, teacher) and the
    // reuse state. Reuse is only sound while every scene change goes through
    // this Renderer: an owned scene whose parameter views were never handed
    // out (sceneTracked()). Otherwise everything is rebuilt and retraced.
    std::unique_ptr<GaussianAccelerationStructure> accelerationStructure;
    uint64_t accelerationKey = 0;
    bool accelerationValid = false;
    bool viewsExposed = false;       // gaussianViews() was handed out
    uint64_t contentEdits = 0;       // markSceneChanged()
    uint64_t visibilityEdits = 0;    // bakes (the visibility stream BakedSH reads)
    uint64_t forwardSettingsEdits = 0; // LOD (the rest reaches CameraUniforms)
    LODParams lodParams{};
    uint64_t frameCacheKey = 0;
    bool frameCacheValid = false;
    std::array<uint64_t, kMaxLights> progressiveKey{};
    std::array<uint32_t, kMaxLights> progressiveSamples{};
    ShadowFrameStats shadowStats;

    bool sceneTracked() const
    {
        return gaussianCloud && !gaussianCloud->borrowedParameters() && !viewsExposed;
    }
    uint64_t sceneKey() const
    {
        return sceneGeneration * 0x9E3779B97F4A7C15ull ^ contentEdits;
    }

    // Builds the shared acceleration structure unless it is known current.
    GaussianAccelerationStructure & sceneAccelerationStructure(bool tracked, float minAlpha = 1.0f / 255.0f)
    {
        VK_VALIDATE(device.features().rayQuery,
                    "Renderer: tracing rays needs ray query (DeviceFeatures::rayQuery), which this "
                    "device does not have enabled");
        if (!accelerationStructure)
        {
            accelerationStructure = std::make_unique<GaussianAccelerationStructure>(device, shaderDir);
            accelerationValid = false;
        }
        // The structure is built for one alpha cutoff (shadow approximations vs
        // the exact bake/teacher): a different one is a rebuild.
        if (!tracked || !accelerationValid || accelerationKey != sceneKey()
            || accelerationStructure->minAlpha() != minAlpha)
        {
            accelerationStructure->build(gaussianCloud->views(), minAlpha);
            accelerationKey = sceneKey();
            accelerationValid = true;
            ++shadowStats.accelerationBuilds;
        }
        return *accelerationStructure;
    }
    ShadowSettings shadowSettings;
    std::vector<Light> lights;
    EnvironmentLight environment;
    LightingSettings lightingSettings;
    bool lightingEnabled = false;
    bool hasLitFrame = false;
    // Built on the first backward(), so forward-only use never needs
    // fp32 buffer atomics.
    std::unique_ptr<BackwardPass> backwardPass;
    std::string shaderDir;

    // Bumped whenever the scene's PARAMETERS are replaced (load/setGaussians)
    // or its storage/count changes (reserveGaussians/setGaussianCount). A
    // Differentiable forward records it; backward() refuses a mismatch, so
    // it can never differentiate a render of a scene that's since been
    // swapped out or resized. (Camera/flags need no such check: backward reuses the ones
    // ForwardPass recorded.)
    uint64_t sceneGeneration = 0;
    uint64_t differentiableGeneration = UINT64_MAX;
    // Sum of the storage generations of every cloud since replaced, so
    // Renderer::storageGeneration() never repeats a value across scenes.
    uint64_t storageGenerationBase = 0;

    VisibilityPass & ensureVisibilityPass()
    {
        if (!visibilityPass)
        {
            visibilityPass = std::make_unique<VisibilityPass>(device, shaderDir);
        }
        return *visibilityPass;
    }

    // Installs a freshly uploaded cloud in place of the current one.
    void replaceCloud(std::shared_ptr<GaussianCloud> cloud)
    {
        if (gaussianCloud)
        {
            storageGenerationBase += gaussianCloud->storageGeneration();
        }
        gaussianCloud = std::move(cloud);
        ++sceneGeneration;
        viewsExposed = false; // views of the old cloud say nothing about this one
    }

    // Whether the last forward() was a completed Differentiable run of the
    // current scene: what backward() differentiates and imageView() exposes.
    // The ForwardPass record catches what the generation cannot -- a
    // swapchain resize since (which also reallocates the image buffer).
    bool hasDifferentiableForward() const
    {
        if (!forwardPass || !gaussianCloud || differentiableGeneration != sceneGeneration)
        {
            return false;
        }
        const ForwardRunRecord & last = forwardPass->lastRun();
        return last.valid && last.completed && last.mode == ForwardMode::Differentiable;
    }

    // Shared precondition + lazy setup for both backward() overloads.
    BackwardPass & prepareBackward()
    {
        VK_VALIDATE(forwardPass && gaussianCloud,
                    "Renderer::backward: needs a shader directory and a loaded scene");
        VK_VALIDATE(hasDifferentiableForward(),
                    "Renderer::backward: no Differentiable forward() of the current scene to "
                    "differentiate -- call forward(RenderMode::Differentiable) first");
        if (!backwardPass)
        {
            backwardPass = std::make_unique<BackwardPass>(device, shaderDir);
        }
        gaussianCloud->ensureGradientBuffers();
        return *backwardPass;
    }

    Camera camera;
    DebugView debugView = DebugView::Final;
    bool hasRenderedFrame = false;
    bool flipUpAxis = false;
    // Full quality by default -- lowering this is opt-in.
    uint32_t maxSHDegree = kShDegree;

    // App-owned: the app creates this (glfwCreateWindowSurface) and
    // destroys it too, but only once onSurfaceRelease fires -- see
    // teardownSurfaceResources().
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    std::function<void()> onSurfaceRelease;

    VkSwapchainKHR swapchain = VK_NULL_HANDLE;
    std::vector<VkImage> swapchainImages;
    // One view per swapchain image -- needed for vkCmdBeginRendering's color
    // attachment (Dear ImGui's UI pass); the blit/clear path before it only
    // ever needed raw VkImage handles.
    std::vector<VkImageView> swapchainImageViews;
    VkFormat swapchainFormat = VK_FORMAT_UNDEFINED;
    VkExtent2D swapchainExtent{};

    // Optional app-supplied UI draw hook (Dear ImGui) -- see
    // Renderer::setUiRenderCallback.
    std::function<void(VkCommandBuffer)> uiRenderCallback;

    struct FrameContext
    {
        // One command pool per frame-in-flight (design rule 4): never reuse
        // Device::m_transferPool, which is load-time-only, not the render
        // recording path.
        VkCommandPool pool = VK_NULL_HANDLE;
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkSemaphore imageAvailable = VK_NULL_HANDLE;
        VkSemaphore renderFinished = VK_NULL_HANDLE;
        VkFence inFlight = VK_NULL_HANDLE;
    };
    FrameContext frames[kMaxFramesInFlight];
    uint32_t currentFrame = 0;

    bool surfaceAttached() const
    {
        return surface != VK_NULL_HANDLE;
    }

    void createSwapchain(uint32_t width, uint32_t height);
    void destroySwapchain();
    void createFrameContexts();
    void destroyFrameContexts();
    void recreateSwapchain(uint32_t width, uint32_t height);

    // Waits out in-flight GPU work and tears down everything attachSurface()
    // built. Called from both shutdown() and the destructor so a caller that
    // forgets to call shutdown() explicitly still gets a clean teardown.
    void teardownSurfaceResources();
};

void Renderer::Impl::createSwapchain(uint32_t width, uint32_t height)
{
    VkSurfaceCapabilitiesKHR caps;
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device.physicalDevice(), surface, &caps),
             "Failed to query surface capabilities");

    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device.physicalDevice(), surface, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(
        device.physicalDevice(), surface, &formatCount, formats.data());
    VK_VALIDATE(!formats.empty(), "Surface has no supported formats");

    uint32_t presentModeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(
        device.physicalDevice(), surface, &presentModeCount, nullptr);
    std::vector<VkPresentModeKHR> presentModes(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(
        device.physicalDevice(), surface, &presentModeCount, presentModes.data());

    // UNORM, not _SRGB: present() blits display-referred color straight in,
    // and an sRGB swapchain would encode it a second time (see surface_format.h).
    const VkSurfaceFormatKHR surfaceFormat = chooseSurfaceFormat(formats);
    if (isSrgbSurfaceFormat(surfaceFormat.format))
    {
        Logger::warning("Swapchain format is sRGB-encoded; display-referred color will be "
                        "encoded twice (no UNORM format offered by this surface)");
    }

    VkPresentModeKHR presentMode = VK_PRESENT_MODE_FIFO_KHR; // guaranteed available
    for (const auto & m : presentModes)
    {
        if (m == VK_PRESENT_MODE_MAILBOX_KHR)
        {
            presentMode = m;
            break;
        }
    }

    VkExtent2D extent;
    if (caps.currentExtent.width != UINT32_MAX)
    {
        extent = caps.currentExtent;
    }
    else
    {
        extent.width = std::clamp(width, caps.minImageExtent.width, caps.maxImageExtent.width);
        extent.height = std::clamp(height, caps.minImageExtent.height, caps.maxImageExtent.height);
    }
    // A minimized Win32 window reports 0x0; no swapchain can be built then.
    // present() skips such frames, so this only fires for an attach at 0x0.
    VK_VALIDATE(extent.width > 0 && extent.height > 0,
                "Renderer::attachSurface: the surface has no size (minimized?) -- attach it once "
                "it has one");

    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && imageCount > caps.maxImageCount)
    {
        imageCount = caps.maxImageCount;
    }

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = surfaceFormat.format;
    createInfo.imageColorSpace = surfaceFormat.colorSpace;
    createInfo.imageExtent = extent;
    createInfo.imageArrayLayers = 1;
    // TRANSFER_DST covers both the initial clear (vkCmdClearColorImage)
    // and the forward rasterizer's actual path: ForwardPass renders into its own offscreen
    // storage image (decision: keeps the rasterizer's format/usage
    // independent of whatever the swapchain surface happens to support) and
    // present() blits that in via vkCmdBlitImage, which also only needs
    // TRANSFER_DST on this side. No STORAGE_BIT needed here.
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    // Graphics queue family presents too (validated in attachSurface), so a
    // single family owns the image -- no concurrent sharing needed.
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.preTransform = caps.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = presentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    VK_CHECK(vkCreateSwapchainKHR(device.handle(), &createInfo, nullptr, &swapchain),
             "Failed to create swapchain");

    uint32_t actualImageCount = 0;
    vkGetSwapchainImagesKHR(device.handle(), swapchain, &actualImageCount, nullptr);
    swapchainImages.resize(actualImageCount);
    vkGetSwapchainImagesKHR(device.handle(), swapchain, &actualImageCount, swapchainImages.data());

    swapchainFormat = surfaceFormat.format;
    swapchainExtent = extent;
    Logger::info("Swapchain " + std::to_string(extent.width) + "x" + std::to_string(extent.height)
                 + ", format " + surfaceFormatName(surfaceFormat.format) + ", color space "
                 + colorSpaceName(surfaceFormat.colorSpace));

    swapchainImageViews.resize(swapchainImages.size());
    for (size_t i = 0; i < swapchainImages.size(); ++i)
    {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = swapchainImages[i];
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = swapchainFormat;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.layerCount = 1;
        VK_CHECK(vkCreateImageView(device.handle(), &viewInfo, nullptr, &swapchainImageViews[i]),
                 "Failed to create swapchain image view");
    }

    // Keep the offscreen render target's size in lockstep with the actual
    // swapchain extent (which may be clamped by surface capabilities and so
    // differ from the width/height the app asked for).
    if (forwardPass)
    {
        forwardPass->resize(extent.width, extent.height);
    }
    if (debugVisualizer)
    {
        debugVisualizer->resize(extent.width, extent.height);
    }
}

void Renderer::Impl::destroySwapchain()
{
    for (VkImageView view : swapchainImageViews)
    {
        vkDestroyImageView(device.handle(), view, nullptr);
    }
    swapchainImageViews.clear();

    if (swapchain != VK_NULL_HANDLE)
    {
        vkDestroySwapchainKHR(device.handle(), swapchain, nullptr);
        swapchain = VK_NULL_HANDLE;
    }
    swapchainImages.clear();
}

void Renderer::Impl::createFrameContexts()
{
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = device.graphicsQueueFamily();

    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; // first wait() doesn't block forever

    for (auto & frame : frames)
    {
        VK_CHECK(vkCreateCommandPool(device.handle(), &poolInfo, nullptr, &frame.pool),
                 "Failed to create per-frame command pool");

        VkCommandBufferAllocateInfo cmdInfo{};
        cmdInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmdInfo.commandPool = frame.pool;
        cmdInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdInfo.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(device.handle(), &cmdInfo, &frame.cmd),
                 "Failed to allocate per-frame command buffer");

        VK_CHECK(vkCreateSemaphore(device.handle(), &semInfo, nullptr, &frame.imageAvailable),
                 "Failed to create imageAvailable semaphore");
        VK_CHECK(vkCreateSemaphore(device.handle(), &semInfo, nullptr, &frame.renderFinished),
                 "Failed to create renderFinished semaphore");
        VK_CHECK(vkCreateFence(device.handle(), &fenceInfo, nullptr, &frame.inFlight),
                 "Failed to create inFlight fence");
    }
}

void Renderer::Impl::destroyFrameContexts()
{
    for (auto & frame : frames)
    {
        if (frame.inFlight != VK_NULL_HANDLE)
        {
            vkDestroyFence(device.handle(), frame.inFlight, nullptr);
        }
        if (frame.renderFinished != VK_NULL_HANDLE)
        {
            vkDestroySemaphore(device.handle(), frame.renderFinished, nullptr);
        }
        if (frame.imageAvailable != VK_NULL_HANDLE)
        {
            vkDestroySemaphore(device.handle(), frame.imageAvailable, nullptr);
        }
        // Destroying the pool frees the command buffer allocated from it too.
        if (frame.pool != VK_NULL_HANDLE)
        {
            vkDestroyCommandPool(device.handle(), frame.pool, nullptr);
        }
        frame = FrameContext{};
    }
}

void Renderer::Impl::recreateSwapchain(uint32_t width, uint32_t height)
{
    vkDeviceWaitIdle(device.handle());
    destroySwapchain();
    createSwapchain(width, height);
}

void Renderer::Impl::teardownSurfaceResources()
{
    if (device.handle() != VK_NULL_HANDLE)
    {
        vkDeviceWaitIdle(device.handle());
    }
    destroyFrameContexts();
    destroySwapchain();
    // Signal the app that it's safe to destroy the surface now: the
    // swapchain built from it is gone, and the instance it was created from
    // (still valid here -- device.shutdown() hasn't run yet) is about to go
    // away too. The app owns the handle; we just own the timing.
    if (surface != VK_NULL_HANDLE)
    {
        surface = VK_NULL_HANDLE;
        if (onSurfaceRelease)
        {
            onSurfaceRelease();
        }
        onSurfaceRelease = nullptr;
    }
}

Renderer::Renderer(uint32_t width, uint32_t height, DepthLayout depthLayout)
    : m_width(width), m_height(height), m_depthLayout(depthLayout), m_impl(std::make_unique<Impl>())
{
    Logger::info("Creating Sabercat differentiable renderer (" + std::to_string(width) + "x"
                 + std::to_string(height) + ")");
}

Renderer::~Renderer()
{
    // Impl's destructor (implicitly invoked here) tears down surface
    // resources; Device's own destructor then shuts the device down.
}

bool Renderer::initialize(const std::vector<const char *> & instanceExtensions,
                          bool enableValidation, const std::string & shaderDir)
{
    Logger::info("Renderer initialize");
    try
    {
        m_impl->device.initialize(instanceExtensions, enableValidation);
        if (!shaderDir.empty())
        {
            m_impl->shaderDir = shaderDir;
            m_impl->forwardPass = std::make_unique<ForwardPass>(m_impl->device, m_width, m_height,
                                                                shaderDir, m_depthLayout);
            m_impl->debugVisualizer =
                std::make_unique<DebugVisualizer>(m_impl->device, m_width, m_height, shaderDir);
        }
    }
    catch (const std::exception & e)
    {
        Logger::error(std::string("Renderer initialize FAILED: ") + e.what());
        return false;
    }
    return true;
}

void Renderer::initializeBorrowed(const VulkanContext & context,
                                  const EnabledDeviceFeatures & enabled,
                                  const std::string & shaderDir)
{
    Logger::info("Renderer initialize (borrowed device)");
    m_impl->device.adopt(context, enabled); // validates, throws on any problem
    if (!shaderDir.empty())
    {
        m_impl->shaderDir = shaderDir;
        m_impl->forwardPass = std::make_unique<ForwardPass>(m_impl->device, m_width, m_height,
                                                            shaderDir, m_depthLayout);
        m_impl->debugVisualizer =
            std::make_unique<DebugVisualizer>(m_impl->device, m_width, m_height, shaderDir);
    }
}

const VulkanContext & Renderer::vulkanContext() const
{
    return m_impl->device.context();
}

const EnabledDeviceFeatures & Renderer::enabledDeviceFeatures() const
{
    return m_impl->device.enabledFeatures();
}

bool Renderer::ownsDevice() const
{
    return m_impl->device.ownsContext();
}

Device & Renderer::sabercatDevice()
{
    if (!m_impl->device.context().valid())
    {
        throw std::runtime_error("Renderer::sabercatDevice: no device (initialize first)");
    }
    return m_impl->device;
}

void Renderer::shutdown()
{
    Logger::info("Renderer shutdown");
    m_impl->teardownSurfaceResources();
    // Every GPU-resource-holding member must be torn down while the device
    // (and its VmaAllocator, which Buffer/Image destructors call back into)
    // is still alive -- shutdown() can run long before Impl's own
    // destructor does (viewer.cpp calls this, then keeps the Renderer
    // object, and therefore Impl, alive until main() returns), so waiting
    // for Impl's destructor here would destroy these after
    // Device::shutdown() already freed the allocator they depend on.
    m_impl->backwardPass.reset();
    m_impl->visibilityPass.reset();
    m_impl->accelerationStructure.reset();
    m_impl->shadowPass.reset();
    m_impl->relightPass.reset();
    m_impl->forwardPass.reset();
    m_impl->debugVisualizer.reset();
    m_impl->gaussianCloud.reset();
    m_impl->device.shutdown();
}

VkInstance Renderer::instance() const
{
    return m_impl->device.instance();
}

VkPhysicalDevice Renderer::physicalDevice() const
{
    return m_impl->device.physicalDevice();
}

VkDevice Renderer::device() const
{
    return m_impl->device.handle();
}

VkQueue Renderer::graphicsQueue() const
{
    return m_impl->device.graphicsQueue();
}

uint32_t Renderer::graphicsQueueFamily() const
{
    return m_impl->device.graphicsQueueFamily();
}

VkFormat Renderer::swapchainFormat() const
{
    return m_impl->swapchainFormat;
}

VkQueue Renderer::computeQueue() const
{
    return m_impl->device.submitQueue();
}

uint32_t Renderer::computeQueueFamily() const
{
    return m_impl->device.submitQueueFamily();
}

const DeviceFeatures & Renderer::deviceFeatures() const
{
    return m_impl->device.features();
}

void Renderer::setUiRenderCallback(std::function<void(VkCommandBuffer)> callback)
{
    // The callback draws inside vkCmdBeginRendering.
    VK_VALIDATE(!callback || m_impl->device.features().dynamicRendering,
                "Renderer::setUiRenderCallback: dynamicRendering is not enabled on this device "
                "(model B: declare EnabledDeviceFeatures::dynamicRendering)");
    m_impl->uiRenderCallback = std::move(callback);
}

void Renderer::attachSurface(VkSurfaceKHR surface, uint32_t width, uint32_t height,
                             std::function<void()> onSurfaceRelease)
{
    VK_VALIDATE(m_impl->device.swapchainEnabled(),
                "Renderer::attachSurface: VK_KHR_swapchain is not enabled on this device "
                "(model A: pass the windowing library's instance extensions to initialize(); "
                "model B: enable it and declare EnabledDeviceFeatures::swapchain)");
    VK_VALIDATE(m_impl->device.graphicsQueue() != VK_NULL_HANDLE,
                "Renderer::attachSurface: no graphics queue to present on (a borrowed "
                "compute-only context is headless)");
    // `surface` is a non-dispatchable handle (an opaque pointer-sized
    // value under the hood) -- copying it is already as cheap as a move,
    // there's no owned resource to steal. onSurfaceRelease is the part
    // actually worth moving: it's a std::function that may own captured
    // state, and moving avoids copying that state (and a possible heap
    // allocation, if the captures don't fit std::function's inline buffer).
    m_impl->surface = surface;
    m_impl->onSurfaceRelease = std::move(onSurfaceRelease);

    // The logical device (and its queues) was already created headlessly in
    // initialize(), before this surface existed, so we couldn't select a
    // present-capable queue family up front. Validate the assumption that
    // the graphics family also presents (true for every desktop GPU/driver
    // this renderer targets) instead of picking a separate present queue.
    VK_VALIDATE(m_impl->device.supportsPresent(surface),
                "Graphics queue family does not support presenting to this surface");

    m_impl->createSwapchain(width, height);
    // The swapchain extent is the size of the image: on Win32 it is the
    // window's real size whatever was requested here.
    m_width = m_impl->swapchainExtent.width;
    m_height = m_impl->swapchainExtent.height;
    m_impl->createFrameContexts();
}

void Renderer::loadScene(const char * path)
{
    Logger::info("Loading Gaussian cloud from: " + std::string(path));
    std::shared_ptr<GaussianCloud> cloud = sabercat::loadScene(m_impl->device, path);
    cloud->uploadToGPU();
    m_impl->replaceCloud(std::move(cloud));
    const SceneBounds & bounds = m_impl->gaussianCloud->bounds();
    Logger::info("Loaded " + std::to_string(m_impl->gaussianCloud->count())
                 + " Gaussians (SH degree "
                 + std::to_string(m_impl->gaussianCloud->shDegree())
                 + (m_impl->gaussianCloud->antialiased() ? ", antialiased" : "")
                 + (bounds.valid ? ", focus radius " + std::to_string(bounds.focusRadius)
                                       + " of " + std::to_string(bounds.radius())
                                 : "")
                 + ")");
}

void Renderer::setCamera(const Camera & camera)
{
    m_impl->camera = camera;
}

void Renderer::setDebugView(DebugView view)
{
    m_impl->debugView = view;
}

void Renderer::setFlipUpAxis(bool flip)
{
    m_impl->flipUpAxis = flip;
}

void Renderer::setLODParams(const LODParams & params)
{
    // LOD changes what the forward pass keeps, so cached visibility depends
    // on it; LODParams is all 4-byte fields (no padding), so compare bytes.
    if (std::memcmp(&params, &m_impl->lodParams, sizeof(LODParams)) != 0)
    {
        m_impl->lodParams = params;
        ++m_impl->forwardSettingsEdits;
    }
    if (m_impl->forwardPass)
    {
        m_impl->forwardPass->setLODParams(params);
    }
}

void Renderer::setMaxSHDegree(uint32_t degree)
{
    m_impl->maxSHDegree = std::min(degree, kShDegree);
}

void Renderer::setAntialiased(bool antialiased)
{
    if (m_impl->gaussianCloud)
    {
        m_impl->gaussianCloud->setAntialiased(antialiased);
    }
}

bool Renderer::sceneAntialiased() const
{
    return m_impl->gaussianCloud && m_impl->gaussianCloud->antialiased();
}

void Renderer::setGaussians(const GaussianHostInput & input)
{
    VK_VALIDATE(input.count > 0, "Renderer::setGaussians: count must be > 0");
    VK_VALIDATE(input.means && input.scales && input.rotations && input.opacities
                    && input.shCoefficients,
                "Renderer::setGaussians: every array must be non-null");
    VK_VALIDATE(input.shDegree <= kShDegree, "Renderer::setGaussians: shDegree must be <= 3");

    auto cloud = std::make_shared<GaussianCloud>(m_impl->device);
    cloud->initialize(input.count);
    const size_t n = input.count;
    std::memcpy(cloud->positions().data(), input.means, n * 4 * sizeof(float));
    std::memcpy(cloud->scales().data(), input.scales, n * 4 * sizeof(float));
    std::memcpy(cloud->rotations().data(), input.rotations, n * 4 * sizeof(float));
    std::memcpy(cloud->opacities().data(), input.opacities, n * sizeof(float));
    std::memcpy(cloud->shCoefficients().data(), input.shCoefficients,
                n * kShCoefficientsPerGaussian * 4 * sizeof(float));
    cloud->setShDegree(input.shDegree);
    cloud->setAntialiased(input.antialiased);
    // Optional relightable streams.
    const auto copyOptional = [n](auto & dst, const float * src, size_t floatsPerGaussian)
    {
        if (src != nullptr)
        {
            dst.resize(n * floatsPerGaussian / (sizeof(dst[0]) / sizeof(float)));
            std::memcpy(dst.data(), src, n * floatsPerGaussian * sizeof(float));
        }
    };
    copyOptional(cloud->normals(), input.normals, 4);
    copyOptional(cloud->baseColorRoughness(), input.baseColorRoughness, 4);
    copyOptional(cloud->metallic(), input.metallic, 1);
    copyOptional(cloud->visibility(), input.visibility, kVisibilityCoefficients);
    cloud->computeBounds();
    cloud->uploadToGPU(input.capacity);
    m_impl->replaceCloud(std::move(cloud));
}

void Renderer::borrowGaussians(const BorrowedGaussians & scene)
{
    // Validate before touching anything: a rejected scene leaves the
    // current one in place.
    const uint32_t capacity = validateBorrowedGaussians(m_impl->device.physicalDevice(),
                                                        m_impl->device.handle(), scene);
    auto cloud = std::make_shared<GaussianCloud>(m_impl->device);
    cloud->borrow(scene.parameters, scene.gradients, scene.relight, capacity);
    cloud->setAntialiased(scene.antialiased);
    m_impl->replaceCloud(std::move(cloud));
}

bool Renderer::gaussiansBorrowed() const
{
    return m_impl->gaussianCloud && m_impl->gaussianCloud->borrowedParameters();
}

void Renderer::clearScene()
{
    if (!m_impl->gaussianCloud)
    {
        return; // no scene: nothing to clear
    }
    // Nothing rendered from the old scene survives: readImage(), imageView()
    // and readLitImage() throw until the next forward(), and present() clears.
    m_impl->differentiableGeneration = UINT64_MAX;
    m_impl->hasRenderedFrame = false;
    m_impl->hasLitFrame = false;
    m_impl->frameCacheValid = false;
    m_impl->progressiveKey.fill(0);
    m_impl->progressiveSamples.fill(0);
    m_impl->shadowStats = ShadowFrameStats{};
    m_impl->litUniforms = CameraUniforms{};
    // Derived from the old scene: the shared acceleration structure, the
    // baked visibility and the teacher samples (held by the visibility pass).
    m_impl->accelerationStructure.reset();
    m_impl->accelerationValid = false;
    m_impl->visibilityPass.reset();
    // Frees the owned GPU buffers. A borrowed scene's buffers are the
    // application's and are left alone. storageGeneration() must still move
    // past every value the old scene reported, hence the +1.
    m_impl->storageGenerationBase += m_impl->gaussianCloud->storageGeneration() + 1;
    m_impl->gaussianCloud.reset();
    m_impl->viewsExposed = false;
    markSceneChanged();
    ++m_impl->sceneGeneration;
}

void Renderer::setImageSize(uint32_t width, uint32_t height)
{
    VK_VALIDATE(m_impl->swapchain == VK_NULL_HANDLE,
                "Renderer::setImageSize: a surface is attached -- the swapchain sets the size");
    VK_VALIDATE(width > 0 && height > 0, "Renderer::setImageSize: width and height must be > 0");
    if (width == m_width && height == m_height)
    {
        return;
    }
    m_width = width;
    m_height = height;
    // Resizing resets ForwardPass's run record, which is what invalidates
    // backward()/imageView(); readImage() needs a new forward as well.
    if (m_impl->forwardPass)
    {
        m_impl->forwardPass->resize(width, height);
    }
    if (m_impl->debugVisualizer)
    {
        m_impl->debugVisualizer->resize(width, height);
    }
    m_impl->differentiableGeneration = UINT64_MAX;
    m_impl->hasRenderedFrame = false;
    m_impl->hasLitFrame = false;
}

bool Renderer::reserveGaussians(uint32_t capacity)
{
    VK_VALIDATE(m_impl->gaussianCloud, "Renderer::reserveGaussians: no scene loaded");
    if (!m_impl->gaussianCloud->reserve(capacity))
    {
        return false;
    }
    // The last forward's record holds the old handles.
    ++m_impl->sceneGeneration;
    return true;
}

void Renderer::setGaussianCount(uint32_t count)
{
    VK_VALIDATE(m_impl->gaussianCloud, "Renderer::setGaussianCount: no scene loaded");
    m_impl->gaussianCloud->setCount(count);
    ++m_impl->sceneGeneration;
}

uint32_t Renderer::gaussianCapacity() const
{
    return m_impl->gaussianCloud ? static_cast<uint32_t>(m_impl->gaussianCloud->capacity()) : 0;
}

uint64_t Renderer::storageGeneration() const
{
    return m_impl->storageGenerationBase
           + (m_impl->gaussianCloud ? m_impl->gaussianCloud->storageGeneration() : 0);
}

void Renderer::readGaussians(const GaussianHostOutput & out)
{
    VK_VALIDATE(m_impl->gaussianCloud, "Renderer::readGaussians: no scene loaded");
    GaussianCloud & cloud = *m_impl->gaussianCloud;
    VK_VALIDATE(out.count == cloud.count(), "Renderer::readGaussians: count mismatch");
    VK_VALIDATE(out.means && out.scales && out.rotations && out.opacities && out.shCoefficients,
                "Renderer::readGaussians: every array must be non-null");
    VK_VALIDATE(out.visibility == nullptr || cloud.relightViews().visibility.valid(),
                "Renderer::readGaussians: visibility requested but the scene has no visibility stream");
    cloud.downloadFromGPU();
    const size_t n = cloud.count();
    if (out.visibility != nullptr)
    {
        std::memcpy(out.visibility, cloud.visibility().data(), n * kVisibilityCoefficients * sizeof(float));
    }
    std::memcpy(out.means, cloud.positions().data(), n * 4 * sizeof(float));
    std::memcpy(out.scales, cloud.scales().data(), n * 4 * sizeof(float));
    std::memcpy(out.rotations, cloud.rotations().data(), n * 4 * sizeof(float));
    std::memcpy(out.opacities, cloud.opacities().data(), n * sizeof(float));
    std::memcpy(out.shCoefficients, cloud.shCoefficients().data(),
                n * kShCoefficientsPerGaussian * 4 * sizeof(float));
}

void Renderer::forward(RenderMode mode)
{
    if (!m_impl->forwardPass || !m_impl->gaussianCloud || !m_impl->camera.valid())
    {
        return;
    }

    glm::mat4 worldToCamera = m_impl->camera.worldToCamera();
    // Camera's world-space translation (column 3 of cameraToWorld), for the
    // SH view direction.
    glm::vec3 cameraPosition = glm::vec3(m_impl->camera.cameraToWorld[3]);
    if (m_impl->flipUpAxis)
    {
        // A 180-degree rotation about world X: Y and Z are negated, the
        // determinant is +1 (a proper rotation, so handedness is kept), and
        // the scene is rotated rather than the camera's own axes. Depth stays
        // positive for anything in front of the camera. Applied to the raw
        // asset-space points before the camera views them -- post-multiplying
        // worldToCamera, since the shader applies it to the loaded means.
        // Covariance goes through the same matrix via a congruence transform
        // (W * Sigma * W^T in forward_project.slang), so splat shape and
        // orientation rotate consistently with position -- no shader changes.
        glm::mat4 rotation(1.0f);
        rotation[1][1] = -1.0f;
        rotation[2][2] = -1.0f;
        worldToCamera = worldToCamera * rotation;

        // The SH view direction must be expressed in the ASSET's own space,
        // because that's the space its SH coefficients were authored in --
        // and the shader computes it against the raw, unrotated mean. With
        // the rotation active, a Gaussian at asset-space P is viewed as R*P,
        // so the direction seen is (R*P - C). Mapping that back to asset
        // space (R is its own inverse, R^-1 = R) gives (P - R*C): the camera
        // position is what carries the rotation. Skipping this would mirror
        // every view-dependent highlight on flipped scenes.
        cameraPosition.y = -cameraPosition.y;
        cameraPosition.z = -cameraPosition.z;
    }

    CameraUniforms uniforms{};
    uniforms.worldToCamera = worldToCamera;
    uniforms.fx = m_impl->camera.fx;
    uniforms.fy = m_impl->camera.fy;
    uniforms.cx = m_impl->camera.cx;
    uniforms.cy = m_impl->camera.cy;
    uniforms.imageWidth = m_width;
    uniforms.imageHeight = m_height;
    uniforms.tilesX = dispatchGroups(m_width, kTileSize);
    uniforms.tilesY = dispatchGroups(m_height, kTileSize);
    uniforms.gaussianCount = static_cast<uint32_t>(m_impl->gaussianCloud->count());
    uniforms.shDegree = m_impl->maxSHDegree;
    uniforms.renderFlags = m_impl->gaussianCloud->antialiased() ? kRenderFlagAntialiased : 0u;
    uniforms.cameraPositionWorld = glm::vec4(cameraPosition, 0.0f);

    const bool differentiable = mode == RenderMode::Differentiable;
    m_impl->differentiableGeneration = UINT64_MAX; // until this run completes
    m_impl->forwardPass->run(m_impl->gaussianCloud->views(), uniforms, ForwardStage::Rasterize,
                             differentiable ? ForwardMode::Differentiable : ForwardMode::Viewer);
    if (differentiable)
    {
        m_impl->differentiableGeneration = m_impl->sceneGeneration;
    }
    m_impl->hasRenderedFrame = true;

    // G-buffer + deferred lighting (relighting) over this run's sorted lists.
    m_impl->hasLitFrame = false;
    if (m_impl->lightingEnabled)
    {
        if (!m_impl->relightPass)
        {
            m_impl->relightPass = std::make_unique<RelightPass>(
                m_impl->device, uniforms.imageWidth, uniforms.imageHeight, m_impl->shaderDir);
        }
        else if (m_impl->relightPass->width() != uniforms.imageWidth
                 || m_impl->relightPass->height() != uniforms.imageHeight)
        {
            m_impl->relightPass->resize(uniforms.imageWidth, uniforms.imageHeight);
        }
        m_impl->relightPass->runGBuffer(*m_impl->forwardPass, m_impl->gaussianCloud->relightViews());

        m_impl->litUniforms = uniforms;

        // Fill the shadowing lights' visibility layers (ray-traced or baked), then
        // optionally replace them with a debug view.
        const ShadowSettings & shadows = m_impl->shadowSettings;
        const bool shadowed = shadows.mode != ShadowMode::None;
        const bool debugView = shadows.debug != VisibilityDebug::Off;
        m_impl->shadowStats = ShadowFrameStats{};
        if (shadowed || debugView)
        {
            const bool needsRays = shadows.mode == ShadowMode::RayTraced
                                   || shadows.debug == VisibilityDebug::ErrorVsReference;
            VK_VALIDATE(!needsRays || m_impl->device.features().rayQuery,
                        "Renderer::forward: ShadowMode::RayTraced and VisibilityDebug::ErrorVsReference "
                        "need ray query, which this device does not have enabled -- use ShadowMode::PCSS");
            VisibilityPass & visibility = m_impl->ensureVisibilityPass();

            // Reuse across frames: only while every input is known unchanged.
            // The keys cover the scene (generation + markSceneChanged), the
            // bake, the forward settings, the camera uniforms (pose,
            // intrinsics, size, flags) and the shadow settings; a light's
            // progressive key adds that one light, the frame key all of them.
            const bool tracked = m_impl->sceneTracked() && shadows.cacheAcrossFrames != 0;
            const auto mix = [](uint64_t h, const void * data, size_t size)
            {
                const auto * bytes = static_cast<const unsigned char *>(data);
                for (size_t i = 0; i < size; ++i)
                {
                    h = (h ^ bytes[i]) * 0x100000001B3ull; // FNV-1a
                }
                return h;
            };
            uint64_t baseKey = 0xCBF29CE484222325ull;
            const uint64_t edits[3] = {m_impl->sceneKey(), m_impl->visibilityEdits, m_impl->forwardSettingsEdits};
            baseKey = mix(baseKey, edits, sizeof(edits));
            baseKey = mix(baseKey, &uniforms, sizeof(uniforms));
            baseKey = mix(baseKey, &shadows, sizeof(shadows));
            uint64_t frameKey = mix(baseKey, m_impl->lights.data(), m_impl->lights.size() * sizeof(Light));
            const bool progressive = shadows.mode == ShadowMode::RayTraced && tracked
                                     && shadows.progressiveSamplesPerFrame > 0;
            const bool frameCacheable = tracked && !progressive && shadows.debug == VisibilityDebug::Off
                                        && shadows.mode != ShadowMode::External;

            if (frameCacheable && m_impl->frameCacheValid && m_impl->frameCacheKey == frameKey)
            {
                visibility.restoreCache(*m_impl->relightPass);
                m_impl->shadowStats.reusedFrame = 1;
            }
            else
            {
                m_impl->relightPass->clearVisibility();
                const auto runRaysOrPCSS = [&](const ShadowSettings & settings, bool accumulate)
                {
                    glm::vec3 focusCenter = settings.focusCenter;
                    float focusRadius = settings.focusRadius;
                    if (focusRadius <= 0.0f)
                    {
                        const SceneBounds & bounds = m_impl->gaussianCloud->bounds();
                        VK_VALIDATE(bounds.valid,
                                    "Renderer::forward: shadows need a focus -- the scene has no bounds "
                                    "(e.g. it is borrowed), so set ShadowSettings::focusCenter/focusRadius");
                        focusCenter = bounds.focusCenter;
                        focusRadius = bounds.radius();
                    }
                    if (!m_impl->shadowPass)
                    {
                        m_impl->shadowPass = std::make_unique<ShadowPass>(m_impl->device, m_impl->shaderDir);
                    }
                    const GaussianAccelerationStructure * as =
                        settings.mode == ShadowMode::RayTraced
                            ? &m_impl->sceneAccelerationStructure(tracked, settings.rayMinAlpha)
                            : nullptr;
                    const uint32_t n = std::max(settings.raySamplesPerAxis, 1u);
                    // Receivers traced per sample (one per rayDownsample block).
                    const uint32_t step = std::max(settings.rayDownsample, 1u);
                    const uint64_t pixels = static_cast<uint64_t>((uniforms.imageWidth + step - 1) / step)
                                            * ((uniforms.imageHeight + step - 1) / step);
                    for (size_t i = 0; i < m_impl->lights.size(); ++i)
                    {
                        if (m_impl->lights[i].castsShadows == 0)
                        {
                            continue;
                        }
                        const uint32_t layer = static_cast<uint32_t>(i);
                        if (settings.mode != ShadowMode::RayTraced)
                        {
                            m_impl->shadowPass->runPCSS(*m_impl->relightPass, uniforms,
                                                        m_impl->gaussianCloud->views(),
                                                        m_impl->gaussianCloud->antialiased(),
                                                        m_impl->lights[i], layer, settings, focusCenter,
                                                        focusRadius);
                            continue;
                        }
                        uint32_t first = 0;
                        uint32_t count = n * n;
                        if (accumulate)
                        {
                            // Restart this light when anything it depends on changed;
                            // once converged, only re-resolve the sum (no rays).
                            const uint64_t lightKey = mix(baseKey, &m_impl->lights[i], sizeof(Light));
                            if (m_impl->progressiveKey[i] != lightKey)
                            {
                                m_impl->progressiveKey[i] = lightKey;
                                m_impl->progressiveSamples[i] = 0;
                            }
                            first = m_impl->progressiveSamples[i];
                            count = std::min(settings.progressiveSamplesPerFrame, n * n - first);
                            m_impl->progressiveSamples[i] += count;
                            if (m_impl->progressiveSamples[i] == n * n)
                            {
                                ++m_impl->shadowStats.convergedLights;
                            }
                            else
                            {
                                ++m_impl->shadowStats.pendingLights;
                            }
                        }
                        else
                        {
                            // A full trace overwrites this layer's running sum.
                            m_impl->progressiveKey[i] = 0;
                        }
                        m_impl->shadowPass->runRayTraced(*m_impl->relightPass, uniforms,
                                                         m_impl->gaussianCloud->views(), *as,
                                                         m_impl->lights[i], layer, settings, focusRadius,
                                                         first, count);
                        m_impl->shadowStats.emitterSamplesDispatched += pixels * count;
                    }
                };

                // The reference for the |error| view, traced first and
                // set aside; the layers are then refilled by the chosen source.
                if (shadows.debug == VisibilityDebug::ErrorVsReference)
                {
                    ShadowSettings reference = shadows;
                    reference.mode = ShadowMode::RayTraced;
                    reference.raySamplesPerAxis = shadows.referenceSamplesPerAxis;
                    runRaysOrPCSS(reference, false);
                    visibility.snapshotReference(*m_impl->relightPass);
                    m_impl->relightPass->clearVisibility();
                }

                switch (shadows.mode)
                {
                case ShadowMode::None:
                    break; // every layer stays 1
                case ShadowMode::PCSS:
                case ShadowMode::RayTraced:
                    runRaysOrPCSS(shadows, progressive);
                    break;
                case ShadowMode::BakedSH:
                    visibility.splatBaked(*m_impl->forwardPass, uniforms, *m_impl->relightPass,
                                          m_impl->gaussianCloud->views(),
                                          m_impl->gaussianCloud->relightViews().visibility, m_impl->lights);
                    break;
                case ShadowMode::External:
                    VK_VALIDATE(m_impl->externalVisibility.buffer != VK_NULL_HANDLE,
                                "Renderer::forward: ShadowMode::External needs setExternalVisibility()");
                    visibility.copyExternal(*m_impl->relightPass, m_impl->externalVisibility);
                    break;
                }

                if (shadows.debug == VisibilityDebug::ErrorVsReference)
                {
                    visibility.applyErrorVsReference(*m_impl->relightPass);
                }
                else if (shadows.debug == VisibilityDebug::SampleLocations)
                {
                    visibility.markSampleLocations(*m_impl->relightPass);
                }

                m_impl->frameCacheValid = frameCacheable;
                if (frameCacheable)
                {
                    visibility.saveCache(*m_impl->relightPass);
                    m_impl->frameCacheKey = frameKey;
                }
            }
        }
        m_impl->relightPass->runLighting(*m_impl->forwardPass, m_impl->lights,
                                         m_impl->environment, m_impl->lightingSettings,
                                         /*resetVisibility=*/!(shadowed || debugView));
        m_impl->hasLitFrame = true;
    }
}

void Renderer::setLightingEnabled(bool enabled)
{
    m_impl->lightingEnabled = enabled;
}

bool Renderer::lightingEnabled() const
{
    return m_impl->lightingEnabled;
}

void Renderer::setLights(const std::vector<Light> & lights)
{
    VK_VALIDATE(lights.size() <= kMaxLights,
                "Renderer::setLights: at most " + std::to_string(kMaxLights) + " lights");
    m_impl->lights = lights;
}

const std::vector<Light> & Renderer::lights() const
{
    return m_impl->lights;
}

void Renderer::setEnvironment(const EnvironmentLight & environment)
{
    m_impl->environment = environment;
}

void Renderer::setLightingSettings(const LightingSettings & settings)
{
    m_impl->lightingSettings = settings;
}

const LightingSettings & Renderer::lightingSettings() const
{
    return m_impl->lightingSettings;
}

void Renderer::setShadowSettings(const ShadowSettings & settings)
{
    m_impl->shadowSettings = settings;
}

const ShadowSettings & Renderer::shadowSettings() const
{
    return m_impl->shadowSettings;
}

std::vector<float> Renderer::readLitImage()
{
    VK_VALIDATE(m_impl->hasLitFrame,
                "Renderer::readLitImage: the last forward() did not run with lighting enabled");
    const Buffer & lit = m_impl->relightPass->litBuffer();
    std::vector<float> out(lit.size() / sizeof(float));
    lit.download(out.data(), lit.size());
    return out;
}

BufferView Renderer::litImageView() const
{
    VK_VALIDATE(m_impl->relightPass, "Renderer::litImageView: no lit forward() yet");
    return m_impl->relightPass->litBuffer().view(sizeof(glm::vec4));
}

BufferView Renderer::visibilityView() const
{
    VK_VALIDATE(m_impl->relightPass, "Renderer::visibilityView: no lit forward() yet");
    return m_impl->relightPass->visibility().view(sizeof(float));
}

std::vector<float> Renderer::readVisibility()
{
    VK_VALIDATE(m_impl->hasLitFrame,
                "Renderer::readVisibility: the last forward() did not run with lighting enabled");
    const Buffer & visibility = m_impl->relightPass->visibility();
    std::vector<float> out(visibility.size() / sizeof(float));
    visibility.download(out.data(), visibility.size());
    return out;
}

Renderer::GBufferData Renderer::readGBuffer()
{
    VK_VALIDATE(m_impl->hasLitFrame,
                "Renderer::readGBuffer: the last forward() did not run with lighting enabled");
    const RelightPass & pass = *m_impl->relightPass;
    GBufferData out;
    const auto read = [](const Buffer & b, std::vector<glm::vec4> & dst)
    {
        dst.resize(b.size() / sizeof(glm::vec4));
        b.download(dst.data(), b.size());
    };
    read(pass.gNormalDepth(), out.normalDepth);
    read(pass.gBaseRoughness(), out.baseRoughness);
    read(pass.gMetalAmbientCoverage(), out.metalAmbientCoverage);
    return out;
}

void Renderer::bakeVisibility(const VisibilityBakeSettings & settings)
{
    VK_VALIDATE(m_impl->gaussianCloud, "Renderer::bakeVisibility: no scene loaded");
    GaussianCloud & cloud = *m_impl->gaussianCloud;
    float rayLength = settings.rayLength;
    if (rayLength <= 0.0f)
    {
        VK_VALIDATE(cloud.bounds().valid,
                    "Renderer::bakeVisibility: the scene has no bounds (e.g. it is borrowed) -- set "
                    "VisibilityBakeSettings::rayLength");
        rayLength = 4.0f * cloud.bounds().radius();
    }
    const BufferView & visibility = cloud.ensureVisibilityStream();
    const bool tracked = m_impl->sceneTracked() && m_impl->shadowSettings.cacheAcrossFrames != 0;
    m_impl->ensureVisibilityPass().bake(cloud.views(), m_impl->sceneAccelerationStructure(tracked),
                                        visibility, settings, rayLength);
    ++m_impl->visibilityEdits; // BakedSH frames depend on the stream
}

uint32_t Renderer::generateTeacherSamples(const TeacherSettings & settings)
{
    VK_VALIDATE(m_impl->hasLitFrame,
                "Renderer::generateTeacherSamples: needs a lit forward() of the current scene first "
                "(setLightingEnabled(true))");
    const ShadowSettings & shadows = m_impl->shadowSettings;
    float focusRadius = shadows.focusRadius;
    if (focusRadius <= 0.0f)
    {
        const SceneBounds & bounds = m_impl->gaussianCloud->bounds();
        VK_VALIDATE(bounds.valid,
                    "Renderer::generateTeacherSamples: the scene has no bounds (e.g. it is borrowed) "
                    "-- set ShadowSettings::focusRadius");
        focusRadius = bounds.radius();
    }
    const bool tracked = m_impl->sceneTracked() && shadows.cacheAcrossFrames != 0;
    const GaussianAccelerationStructure & as = m_impl->sceneAccelerationStructure(tracked);
    return m_impl->ensureVisibilityPass().generateTeacherSamples(
        *m_impl->relightPass, m_impl->litUniforms, m_impl->gaussianCloud->views(), as, m_impl->lights,
        settings, shadows, focusRadius);
}

void Renderer::markSceneChanged()
{
    ++m_impl->contentEdits;
}

const ShadowFrameStats & Renderer::shadowFrameStats() const
{
    return m_impl->shadowStats;
}

BufferView Renderer::teacherSampleView() const
{
    VK_VALIDATE(m_impl->visibilityPass && m_impl->visibilityPass->samples(),
                "Renderer::teacherSampleView: no generateTeacherSamples() yet");
    BufferView view = m_impl->visibilityPass->samples()->view(sizeof(VisibilityTrainingSample));
    view.size = static_cast<VkDeviceSize>(m_impl->visibilityPass->sampleCount())
                * sizeof(VisibilityTrainingSample);
    return view;
}

std::vector<VisibilityTrainingSample> Renderer::readTeacherSamples()
{
    VK_VALIDATE(m_impl->visibilityPass && m_impl->visibilityPass->samples(),
                "Renderer::readTeacherSamples: no generateTeacherSamples() yet");
    std::vector<VisibilityTrainingSample> out(m_impl->visibilityPass->sampleCount());
    if (!out.empty())
    {
        m_impl->visibilityPass->samples()->download(out.data(), out.size() * sizeof(out[0]));
    }
    return out;
}

std::vector<uint32_t> Renderer::readTeacherSamplePixels()
{
    VK_VALIDATE(m_impl->visibilityPass && m_impl->visibilityPass->samplePixels(),
                "Renderer::readTeacherSamplePixels: no generateTeacherSamples() yet");
    std::vector<uint32_t> out(m_impl->visibilityPass->sampleCount());
    if (!out.empty())
    {
        m_impl->visibilityPass->samplePixels()->download(out.data(), out.size() * sizeof(out[0]));
    }
    return out;
}

uint32_t Renderer::teacherSmoothSampleCount() const
{
    return m_impl->visibilityPass ? m_impl->visibilityPass->smoothSampleCount() : 0u;
}

void Renderer::setExternalVisibility(const BufferView & visibility)
{
    m_impl->externalVisibility = visibility;
}

void Renderer::backward(const float * dLoss_dImage)
{
    backward(dLoss_dImage, static_cast<const float *>(nullptr));
}

void Renderer::backward(const BufferView & dLoss_dImage)
{
    backward(dLoss_dImage, BufferView{});
}

void Renderer::backward(const float * dLoss_dImage, const float * dLoss_dDepth)
{
    VK_VALIDATE(dLoss_dImage != nullptr, "Renderer::backward: null image gradient");
    VK_VALIDATE(dLoss_dDepth == nullptr || m_depthLayout != DepthLayout::None,
                "Renderer::backward: a depth gradient was given but the renderer was created with "
                "DepthLayout::None");
    m_impl->prepareBackward().run(*m_impl->forwardPass, dLoss_dImage, dLoss_dDepth,
                                  m_impl->gaussianCloud->gradientViews());
}

void Renderer::backward(const BufferView & dLoss_dImage, const BufferView & dLoss_dDepth)
{
    VK_VALIDATE(!dLoss_dDepth.valid() || m_depthLayout != DepthLayout::None,
                "Renderer::backward: a depth gradient was given but the renderer was created with "
                "DepthLayout::None");
    m_impl->prepareBackward().run(*m_impl->forwardPass, dLoss_dImage, dLoss_dDepth,
                                  m_impl->gaussianCloud->gradientViews());
}

std::vector<float> Renderer::readImage()
{
    VK_VALIDATE(m_impl->forwardPass && m_impl->hasRenderedFrame,
                "Renderer::readImage: nothing rendered yet");
    if (m_impl->hasDifferentiableForward())
    {
        // Same values, already in the row-major layout: a plain buffer
        // download, no image layout transition.
        const Buffer & buffer = m_impl->forwardPass->outputBuffer();
        std::vector<float> out(buffer.size() / sizeof(float));
        buffer.download(out.data(), buffer.size());
        return out;
    }
    std::vector<glm::vec4> pixels = m_impl->forwardPass->downloadOutputImage();
    std::vector<float> out(pixels.size() * 4);
    std::memcpy(out.data(), pixels.data(), out.size() * sizeof(float));
    return out;
}

GaussianGradients Renderer::readGradients()
{
    VK_VALIDATE(m_impl->gaussianCloud && m_impl->gaussianCloud->hasGradientBuffers(),
                "Renderer::readGradients: no backward() has run on the current scene");
    return m_impl->gaussianCloud->downloadGradients();
}

BufferView Renderer::imageView() const
{
    VK_VALIDATE(m_impl->hasDifferentiableForward(),
                "Renderer::imageView: no Differentiable forward() of the current scene -- call "
                "forward(RenderMode::Differentiable) first");
    const Buffer & buffer = m_impl->forwardPass->outputBuffer();
    BufferView view;
    view.buffer = buffer.handle();
    view.offset = 0;
    view.size = buffer.size();
    view.stride = sizeof(glm::vec4);
    return view;
}

DepthLayout Renderer::depthLayout() const
{
    return m_depthLayout;
}

std::vector<float> Renderer::readDepth()
{
    VK_VALIDATE(m_depthLayout != DepthLayout::None,
                "Renderer::readDepth: the renderer was created with DepthLayout::None");
    VK_VALIDATE(m_impl->hasDifferentiableForward(),
                "Renderer::readDepth: no Differentiable forward() of the current scene -- call "
                "forward(RenderMode::Differentiable) first");
    const Buffer & buffer = m_impl->forwardPass->depthBuffer();
    std::vector<float> out(buffer.size() / sizeof(float));
    buffer.download(out.data(), buffer.size());
    return out;
}

BufferView Renderer::depthView() const
{
    VK_VALIDATE(m_depthLayout != DepthLayout::None,
                "Renderer::depthView: the renderer was created with DepthLayout::None");
    VK_VALIDATE(m_impl->hasDifferentiableForward(),
                "Renderer::depthView: no Differentiable forward() of the current scene -- call "
                "forward(RenderMode::Differentiable) first");
    const Buffer & buffer = m_impl->forwardPass->depthBuffer();
    BufferView view;
    view.buffer = buffer.handle();
    view.offset = 0;
    view.size = buffer.size();
    view.stride = static_cast<uint32_t>(m_depthLayout) * sizeof(float);
    return view;
}

CanonicalGaussianBufferViews Renderer::gaussianViews() const
{
    VK_VALIDATE(m_impl->gaussianCloud, "Renderer::gaussianViews: no scene loaded");
    // The caller can now write the parameters in place, so reuse across
    // frames (acceleration structure, cached shadows) stops for this scene.
    m_impl->viewsExposed = true;
    return m_impl->gaussianCloud->views();
}

CanonicalGaussianGradientViews Renderer::gradientViews() const
{
    VK_VALIDATE(m_impl->gaussianCloud, "Renderer::gradientViews: no scene loaded");
    m_impl->gaussianCloud->ensureGradientBuffers();
    return m_impl->gaussianCloud->gradientViews();
}

bool Renderer::backwardSupported() const
{
    return m_impl->device.features().shaderBufferFloat32AtomicAdd;
}

namespace internal
{

void recordPresentBlit(VkCommandBuffer cmd, VkImage source, uint32_t sourceWidth,
                       uint32_t sourceHeight, VkImage destination, uint32_t destinationWidth,
                       uint32_t destinationHeight)
{
    VkImageBlit blitRegion{};
    blitRegion.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitRegion.srcSubresource.layerCount = 1;
    blitRegion.srcOffsets[1] = {static_cast<int32_t>(sourceWidth),
                                static_cast<int32_t>(sourceHeight), 1};
    blitRegion.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    blitRegion.dstSubresource.layerCount = 1;
    blitRegion.dstOffsets[1] = {static_cast<int32_t>(destinationWidth),
                                static_cast<int32_t>(destinationHeight), 1};
    // NEAREST: always-supported per spec, and this is a same-size blit
    // in the common case (ForwardPass is kept in lockstep with the
    // swapchain extent) so filtering quality isn't in play anyway.
    // No color conversion is wanted: the source is display-referred and the
    // destination is UNORM, so the blit stores the values as they are.
    vkCmdBlitImage(cmd, source, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, destination,
                   VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blitRegion, VK_FILTER_NEAREST);
}

} // namespace internal

// Display contract: every image this blits is already in display space. The
// forward color is sRGB-encoded (3DGS SH color, like the training photos),
// the lit output is display-encoded by the lighting pass, and the debug views
// are raw data shown as-is. The swapchain is therefore a UNORM format
// (chooseSurfaceFormat), so the blit stores these values unchanged. An _SRGB
// swapchain would encode them a second time, and everything would look washed out.
void Renderer::present()
{
    if (!m_impl->surfaceAttached())
    {
        return; // headless: nothing to present
    }

    // The surface's current size decides the swapchain. A zero size means
    // minimized (Win32): there is nothing to show and no swapchain can be
    // built, so the frame is skipped and the app's width()/height() stay at
    // the last real size.
    VkSurfaceCapabilitiesKHR caps{};
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
                 m_impl->device.physicalDevice(), m_impl->surface, &caps),
             "Failed to query surface capabilities");
    if (caps.currentExtent.width == 0 || caps.currentExtent.height == 0)
    {
        return;
    }
    // Recreates the swapchain and adopts its extent as the renderer's size.
    // The image the app just rendered is at the old size and is gone, so
    // the frame is not presented; the next forward() uses the new size.
    const auto recreate = [this]()
    {
        m_impl->recreateSwapchain(m_width, m_height);
        m_width = m_impl->swapchainExtent.width;
        m_height = m_impl->swapchainExtent.height;
        m_impl->differentiableGeneration = UINT64_MAX;
        m_impl->hasRenderedFrame = false;
        m_impl->hasLitFrame = false;
    };
    // On Wayland currentExtent is "undefined" (UINT32_MAX) and the size comes
    // from the request, so there is nothing to compare against.
    if (caps.currentExtent.width != UINT32_MAX
        && (caps.currentExtent.width != m_impl->swapchainExtent.width
            || caps.currentExtent.height != m_impl->swapchainExtent.height))
    {
        recreate();
        return;
    }

    Impl::FrameContext & frame = m_impl->frames[m_impl->currentFrame];
    VkDevice dev = m_impl->device.handle();

    VK_CHECK(vkWaitForFences(dev, 1, &frame.inFlight, VK_TRUE, UINT64_MAX),
             "Failed to wait for in-flight fence");

    uint32_t imageIndex = 0;
    VkResult acquireResult = vkAcquireNextImageKHR(
        dev, m_impl->swapchain, UINT64_MAX, frame.imageAvailable, VK_NULL_HANDLE, &imageIndex);
    if (acquireResult == VK_ERROR_OUT_OF_DATE_KHR)
    {
        recreate();
        return;
    }
    VK_VALIDATE(acquireResult == VK_SUCCESS || acquireResult == VK_SUBOPTIMAL_KHR,
                "Failed to acquire swapchain image");

    VK_CHECK(vkResetFences(dev, 1, &frame.inFlight), "Failed to reset in-flight fence");
    VK_CHECK(vkResetCommandBuffer(frame.cmd, 0), "Failed to reset per-frame command buffer");

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(frame.cmd, &beginInfo),
             "Failed to begin per-frame command buffer");

    VkImage image = m_impl->swapchainImages[imageIndex];

    VkImageSubresourceRange range{};
    range.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    range.levelCount = 1;
    range.layerCount = 1;

    // No render pass / graphics pipeline here -- this library is pure
    // compute (see PERFORMANCE.md). ForwardPass renders into its own
    // offscreen storage image; this barrier just gets the swapchain image
    // ready to receive either that (via blit, once a frame has actually
    // been rendered) or a clear (the pre-forward()/headless fallback).
    VkImageMemoryBarrier toTransferDst{};
    toTransferDst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    toTransferDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    toTransferDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toTransferDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransferDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    toTransferDst.image = image;
    toTransferDst.subresourceRange = range;
    toTransferDst.srcAccessMask = 0;
    toTransferDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(frame.cmd,
                         VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0,
                         0,
                         nullptr,
                         0,
                         nullptr,
                         1,
                         &toTransferDst);

    if (m_impl->hasRenderedFrame && m_impl->forwardPass)
    {
        const Image * source = &m_impl->forwardPass->outputImage();
        if (m_impl->debugView == DebugView::Final && m_impl->hasLitFrame)
        {
            // The lit image (or its selected G-buffer channel).
            m_impl->relightPass->prepareLitForRead(frame.cmd);
            source = &m_impl->relightPass->litImage();
        }
        else if (m_impl->debugView == DebugView::Final)
        {
            m_impl->forwardPass->prepareOutputForRead(frame.cmd);
        }
        else
        {
            m_impl->debugVisualizer->record(frame.cmd, m_impl->debugView, *m_impl->forwardPass,
                                            kDebugDepthScale, kDebugDensityScale,
                                            m_impl->currentFrame);
            source = &m_impl->debugVisualizer->displayImage(m_impl->currentFrame);
        }

        internal::recordPresentBlit(frame.cmd,
                                    source->getHandle(),
                                    source->getWidth(),
                                    source->getHeight(),
                                    image,
                                    m_impl->swapchainExtent.width,
                                    m_impl->swapchainExtent.height);
    }
    else
    {
        // Nothing rendered yet (no forward() call, no cloud loaded, or a
        // headless build with no shaderDir) -- clear so the window isn't
        // showing garbage.
        VkClearColorValue clearColor{};
        clearColor.float32[0] = 0.02f;
        clearColor.float32[1] = 0.02f;
        clearColor.float32[2] = 0.05f;
        clearColor.float32[3] = 1.0f;
        vkCmdClearColorImage(
            frame.cmd, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearColor, 1, &range);
    }

    if (m_impl->uiRenderCallback)
    {
        // The blit/clear above left the image in TRANSFER_DST_OPTIMAL;
        // vkCmdBeginRendering's color attachment needs COLOR_ATTACHMENT_
        // OPTIMAL. LOAD_OP_LOAD preserves what was just blitted/cleared --
        // the UI draws over it, not instead of it.
        VkImageMemoryBarrier toColorAttachment = toTransferDst;
        toColorAttachment.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toColorAttachment.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        toColorAttachment.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toColorAttachment.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(frame.cmd,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                             0,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             1,
                             &toColorAttachment);

        VkRenderingAttachmentInfo colorAttachment{};
        colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        colorAttachment.imageView = m_impl->swapchainImageViews[imageIndex];
        colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

        VkRenderingInfo renderingInfo{};
        renderingInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
        renderingInfo.renderArea.extent = m_impl->swapchainExtent;
        renderingInfo.layerCount = 1;
        renderingInfo.colorAttachmentCount = 1;
        renderingInfo.pColorAttachments = &colorAttachment;

        vkCmdBeginRendering(frame.cmd, &renderingInfo);
        m_impl->uiRenderCallback(frame.cmd);
        vkCmdEndRendering(frame.cmd);
    }

    VkImageMemoryBarrier toPresent = toTransferDst;
    toPresent.oldLayout = m_impl->uiRenderCallback ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL
                                                   : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
    toPresent.srcAccessMask = m_impl->uiRenderCallback ? VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT
                                                       : VK_ACCESS_TRANSFER_WRITE_BIT;
    toPresent.dstAccessMask = 0;
    vkCmdPipelineBarrier(frame.cmd,
                         m_impl->uiRenderCallback ? VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT
                                                  : VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                         0,
                         0,
                         nullptr,
                         0,
                         nullptr,
                         1,
                         &toPresent);

    VK_CHECK(vkEndCommandBuffer(frame.cmd), "Failed to end per-frame command buffer");

    VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &frame.imageAvailable;
    submitInfo.pWaitDstStageMask = &waitStage;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &frame.cmd;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &frame.renderFinished;

    VK_CHECK(vkQueueSubmit(m_impl->device.graphicsQueue(), 1, &submitInfo, frame.inFlight),
             "Failed to submit frame command buffer");

    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = 1;
    presentInfo.pWaitSemaphores = &frame.renderFinished;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &m_impl->swapchain;
    presentInfo.pImageIndices = &imageIndex;

    VkResult presentResult = vkQueuePresentKHR(m_impl->device.graphicsQueue(), &presentInfo);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR)
    {
        recreate();
    }
    else
    {
        VK_VALIDATE(presentResult == VK_SUCCESS, "Failed to present swapchain image");
    }

    m_impl->currentFrame = (m_impl->currentFrame + 1) % kMaxFramesInFlight;
}

uint32_t Renderer::gaussianCount() const
{
    return m_impl->gaussianCloud ? static_cast<uint32_t>(m_impl->gaussianCloud->count()) : 0;
}

SceneBounds Renderer::sceneBounds() const
{
    return m_impl->gaussianCloud ? m_impl->gaussianCloud->bounds() : SceneBounds{};
}

uint32_t Renderer::sceneSHDegree() const
{
    return m_impl->gaussianCloud ? m_impl->gaussianCloud->shDegree() : 0;
}

uint32_t Renderer::lastInstanceCount() const
{
    return m_impl->forwardPass ? m_impl->forwardPass->lastInstanceCount() : 0;
}

} // namespace sabercat
