// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vulkan/vulkan.h>

#include <vma/vk_mem_alloc.h>

#include <cstdint>
#include <functional>
#include <vector>

namespace sabercat
{

// Perf-relevant GPU capabilities, queried and enabled at device creation
// (PERFORMANCE.md invariant #6). Plain data so callers (host code, and later
// shader dispatch decisions) never need to touch a Vulkan pNext chain.
struct DeviceFeatures
{
    // Subgroup (wave/warp) properties -- the sort and tile-blend kernels
    // branch on these.
    uint32_t subgroupSize = 0;
    bool subgroupBasic = false;
    bool subgroupArithmetic = false;
    bool subgroupBallot = false;
    bool subgroupShuffle = false;

    // Precision.
    bool shaderFloat16 = false;
    bool shaderInt8 = false;

    // 16-bit storage (lets SSBOs/UBOs hold fp16 without unpacking to fp32).
    bool storageBuffer16BitAccess = false;
    bool uniformAndStorageBuffer16BitAccess = false;

    // Bindless-ready buffer addressing.
    bool bufferDeviceAddress = false;

    // Core in 1.3 (Device targets VK_API_VERSION_1_3) but still an opt-in
    // feature bit. Lets a UI layer (Dear ImGui) draw into the swapchain via
    // vkCmdBeginRendering instead of needing a VkRenderPass/VkFramebuffer --
    // this renderer has neither, being pure-compute otherwise.
    bool dynamicRendering = false;

    // fp32 atomic add on storage buffers (VK_EXT_shader_atomic_float).
    // The backward pass accumulates per-Gaussian gradients with
    // InterlockedAdd on float SSBOs, which Slang lowers to OpAtomicFAddEXT --
    // a pipeline using it fails validation unless this is ENABLED, not just
    // supported. Forward/viewer never need it, so it gates only BackwardPass
    // (which is built lazily and throws a named error without it). Under
    // ownership model B (adopt()) this reports support only: the owner
    // decided what got enabled at device creation.
    bool shaderBufferFloat32AtomicAdd = false;

    // Ray-traced visibility through the splats
    // (GaussianAccelerationStructure + shaders/gaussian_ray.slangh). True
    // when VK_KHR_acceleration_structure, VK_KHR_ray_query and
    // bufferDeviceAddress are all ENABLED -- model A enables them whenever
    // the GPU has them; under model B this reports the owner's declaration
    // (EnabledDeviceFeatures::rayQuery). Rasterization never needs it.
    bool rayQuery = false;

    // Neural/cooperative extensions -- detected only, not enabled or used by
    // the core rasterizer. Reserved for a future Creator-side neural module.
    bool hasCooperativeMatrix = false;
    bool hasCooperativeVector = false;

    // True if the compute queue is a distinct queue from the graphics queue
    // (own family, or a second queue index within a shared family) --
    // required for the async compute overlap in invariant #5.
    bool hasDedicatedComputeQueue = false;
};

// The Vulkan API version Sabercat targets. A borrowed instance and
// device must have been created for at least this version.
constexpr uint32_t kRequiredVulkanApiVersion = VK_API_VERSION_1_3;
// tile_rasterize runs one 16x16 workgroup per tile; Vulkan only guarantees
// 128 invocations, so this limit is checked on every device.
constexpr uint32_t kRequiredComputeWorkGroupInvocations = 256;

// Borrowed device (ownership model B): what the owner of a borrowed device ENABLED
// when it created it. Vulkan cannot be asked after the fact, so the owner
// declares it and Device::adopt() checks every declaration against what the
// GPU supports (declaring something unsupported is rejected). Sabercat gates
// its optional paths on these: backward() on the atomic, present() on the
// swapchain. DeviceRequirements::enabled() fills this in for an owner that
// created its device from DeviceRequirements.
struct EnabledDeviceFeatures
{
    // VkApplicationInfo::apiVersion the instance was created with (the
    // device inherits it). Must be >= kRequiredVulkanApiVersion.
    uint32_t apiVersion = 0;
    // VK_EXT_shader_atomic_float enabled with shaderBufferFloat32AtomicAdd:
    // required by backward().
    bool shaderBufferFloat32AtomicAdd = false;
    // VK_KHR_surface on the instance and VK_KHR_swapchain on the device:
    // required by attachSurface()/present().
    bool swapchain = false;
    // VkPhysicalDeviceVulkan13Features::dynamicRendering: required by
    // present()'s UI callback (setUiRenderCallback).
    bool dynamicRendering = false;
    // VK_KHR_acceleration_structure (+ VK_KHR_deferred_host_operations)
    // and VK_KHR_ray_query enabled with their accelerationStructure / rayQuery
    // features, plus VkPhysicalDeviceVulkan12Features::bufferDeviceAddress:
    // required by GaussianAccelerationStructure and gaussianTransmittance.
    // (Fills the struct's last byte: 4 + 4 x 1 B, no padding.)
    bool rayQuery = false;
};
static_assert(sizeof(EnabledDeviceFeatures) == 8, "EnabledDeviceFeatures must stay padding-free");

// Everything Sabercat needs a device to be created with, for an
// application creating one to lend (model B) -- and what Device itself uses
// for model A, so the two can never disagree. Usage:
//
//   DeviceRequirements req(gpu, {DeviceRequirements::Need::Required});
//   // instance: apiVersion >= kRequiredVulkanApiVersion,
//   //           + req.instanceExtensions()
//   deviceInfo.pNext = req.featureChain();   // VkPhysicalDeviceFeatures2
//   deviceInfo.pEnabledFeatures = nullptr;
//   // + req.deviceExtensions(), then vkCreateDevice
//   renderer.initializeBorrowed(context, req.enabled(), shaderDir);
//
// The chain is self-referential, so the object is neither copyable nor
// movable; keep it alive until vkCreateDevice returns. To chain feature
// structs of your own, set featureChainTail()->pNext.
class DeviceRequirements
{
  public:
    enum class Need
    {
        None,        // not requested
        IfSupported, // enabled when the GPU supports it, skipped otherwise
        Required,    // the constructor throws if the GPU lacks it
    };
    struct Options
    {
        Need training = Need::Required; // fp32 buffer atomics, for backward()
        Need presentation = Need::None; // swapchain + dynamic rendering, for present()
        // Acceleration structures + ray query + buffer device
        // address, for GaussianAccelerationStructure / gaussianTransmittance.
        Need rayTracing = Need::None;
    };

    // Throws (std::runtime_error) if the GPU fails validatePhysicalDevice()
    // or lacks anything marked Required.
    DeviceRequirements(VkPhysicalDevice physicalDevice, const Options & options);

    DeviceRequirements(const DeviceRequirements &) = delete;
    DeviceRequirements & operator=(const DeviceRequirements &) = delete;

    // VkPhysicalDeviceFeatures2 head of the chain, for VkDeviceCreateInfo::pNext
    // (pEnabledFeatures must then be null).
    void * featureChain()
    {
        return &m_features2;
    }
    // The last struct of the chain, whose pNext the caller may set. (Tracks
    // the chain as built: with training None the atomic-float struct is not
    // in it, so returning that struct would have dropped the caller's.)
    VkBaseOutStructure * featureChainTail()
    {
        return m_tail;
    }
    const std::vector<const char *> & deviceExtensions() const
    {
        return m_deviceExtensions;
    }
    // Instance extensions Sabercat itself needs (VK_KHR_surface when
    // presenting). The windowing library's platform surface extension
    // (e.g. from glfwGetRequiredInstanceExtensions) is the caller's to add.
    const std::vector<const char *> & instanceExtensions() const
    {
        return m_instanceExtensions;
    }
    // What this chain enables, to declare at adopt time.
    EnabledDeviceFeatures enabled() const
    {
        return m_enabled;
    }

  private:
    // Links `next` after the current tail of the chain.
    void append(void * next);

    VkPhysicalDeviceFeatures2 m_features2{};
    VkPhysicalDeviceVulkan11Features m_vulkan11{};
    VkPhysicalDeviceVulkan12Features m_vulkan12{};
    VkPhysicalDeviceVulkan13Features m_vulkan13{};
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT m_atomicFloat{};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR m_accelerationStructure{};
    VkPhysicalDeviceRayQueryFeaturesKHR m_rayQuery{};
    VkBaseOutStructure * m_tail = nullptr;
    std::vector<const char *> m_deviceExtensions;
    std::vector<const char *> m_instanceExtensions;
    EnabledDeviceFeatures m_enabled{};
};

// Checks that a physical device meets Sabercat's hard requirements
// (Vulkan >= kRequiredVulkanApiVersion, >= kRequiredComputeWorkGroupInvocations
// invocations per workgroup). Throws std::runtime_error naming the first
// failure. Used by both ownership models.
void validatePhysicalDevice(VkPhysicalDevice physicalDevice);

// Whether `physicalDevice` offers device extension `name`.
bool deviceExtensionAvailable(VkPhysicalDevice physicalDevice, const char * name);

// A borrowed view of the Vulkan objects the renderer needs, carrying no
// statement about who created them (the "context seam"; mirrors
// the application's borrowed-context type). Everything downstream --
// Buffer, Pipeline, DescriptorSet, the swapchain, every compute pass --
// depends on this view rather than on Device itself, so the same code runs
// whether Sabercat created the device (ownership model A) or an
// application handed one in (model B).
//
// The owner keeps these alive; a borrower destroys none of them.
struct VulkanContext
{
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;

    VkQueue graphicsQueue = VK_NULL_HANDLE;
    uint32_t graphicsFamily = UINT32_MAX;
    VkQueue computeQueue = VK_NULL_HANDLE;
    uint32_t computeFamily = UINT32_MAX;

    // Handles present and at least one queue with its family. A headless
    // borrower (e.g. the application's borrowed context) may supply
    // only the compute queue; presenting then needs the graphics queue too.
    // validateBorrowedContext() checks much more than this.
    bool valid() const
    {
        const bool hasCompute = computeQueue != VK_NULL_HANDLE && computeFamily != UINT32_MAX;
        const bool hasGraphics = graphicsQueue != VK_NULL_HANDLE && graphicsFamily != UINT32_MAX;
        return instance != VK_NULL_HANDLE && physicalDevice != VK_NULL_HANDLE
               && device != VK_NULL_HANDLE && (hasCompute || hasGraphics);
    }
};

// Checks a borrowed context and the owner's declarations before
// Sabercat touches them, throwing std::invalid_argument naming the first
// problem. Every check -- and what Vulkan does not let us check -- is listed
// in docs/library.md ("Requirements on a borrowed device"). Device::adopt() runs it.
void validateBorrowedContext(const VulkanContext & context, const EnabledDeviceFeatures & enabled);

// Owns the core Vulkan objects shared by the whole engine: the instance, the
// chosen GPU, the logical device, the graphics/compute queues, and the VMA
// allocator used for all buffer/image allocations.
//
// This is intentionally headless (no surface/swapchain) so the compute and
// memory paths can be brought up and tested before presentation is wired in.
class Device
{
  public:
    Device();
    ~Device();

    // Non-copyable: it owns unique Vulkan handles.
    Device(const Device &) = delete;
    Device & operator=(const Device &) = delete;

    // Creates the instance, selects a GPU, creates the logical device + queues,
    // and initializes the VMA allocator.
    //
    // instanceExtensions: extra instance extensions the caller needs enabled.
    //   Applications that will present pass their windowing library's required
    //   extensions here (e.g. glfwGetRequiredInstanceExtensions). Leave empty
    //   for headless/offscreen use. This is how the renderer stays windowing-
    //   agnostic: it never calls GLFW, the app injects what it needs.
    // enableValidation: turn on the Khronos validation layer + debug messenger.
    void initialize(const std::vector<const char *> & instanceExtensions = {},
                    bool enableValidation = true);

    // Ownership model B: adopt Vulkan objects created by the application
    // instead of creating them. Sabercat still creates (and destroys) its
    // own VMA allocator and transfer pool over the borrowed device -- the
    // allocator is deliberately not part of the shared context -- but
    // destroys none of `context`'s handles. The caller must keep them alive
    // for this Device's lifetime. `enabled` is what the owner enabled at
    // creation: validateBorrowedContext() checks both first, and features()
    // then reports the declared values, not mere support.
    void adopt(const VulkanContext & context, const EnabledDeviceFeatures & enabled);

    // The borrowed context view, valid under both ownership models.
    // Downstream code should depend on this rather than on Device.
    const VulkanContext & context() const
    {
        return m_context;
    }

    // True when this Device created the context (model A) and will destroy
    // it on shutdown; false when it adopted one (model B).
    bool ownsContext() const
    {
        return m_ownsContext;
    }

    // Tears everything down. Safe to call multiple times.
    void shutdown();

    // Core handle accessors -- thin conveniences over context().
    VkInstance instance() const
    {
        return m_context.instance;
    }
    VkPhysicalDevice physicalDevice() const
    {
        return m_context.physicalDevice;
    }
    VkDevice handle() const
    {
        return m_context.device;
    }
    VmaAllocator allocator() const
    {
        return m_allocator;
    }

    // Queue accessors.
    VkQueue graphicsQueue() const
    {
        return m_context.graphicsQueue;
    }
    VkQueue computeQueue() const
    {
        return m_context.computeQueue;
    }
    uint32_t graphicsQueueFamily() const
    {
        return m_context.graphicsFamily;
    }
    uint32_t computeQueueFamily() const
    {
        return m_context.computeFamily;
    }

    const VkPhysicalDeviceProperties & properties() const
    {
        return m_properties;
    }
    const DeviceFeatures & features() const
    {
        return m_features;
    }
    // Whether VK_KHR_swapchain is enabled on this device (model A: when the
    // instance got a surface extension; model B: as declared).
    bool swapchainEnabled() const
    {
        return m_enabled.swapchain;
    }
    // What is enabled on this device, in the form adopt() takes -- so a
    // second Device (or Renderer) can borrow this one's context as-is.
    const EnabledDeviceFeatures & enabledFeatures() const
    {
        return m_enabled;
    }

    // Synchronous alloc-record-submit-wait helper (load-time / occasional
    // work: uploads, downloads, storage growth). Submits to submitQueue(),
    // bracketed by recordGlobalBarrier() on both sides.
    void immediateSubmit(const std::function<void(VkCommandBuffer)> & record);

    // The one queue every Sabercat submission goes to except present()'s:
    // the compute queue, or the graphics queue if a borrowed context has no
    // compute queue. Consumers sharing the device should submit their work
    // touching Sabercat's buffers here too (see docs/library.md, synchronization).
    VkQueue submitQueue() const
    {
        return m_context.computeQueue != VK_NULL_HANDLE ? m_context.computeQueue
                                                        : m_context.graphicsQueue;
    }
    uint32_t submitQueueFamily() const
    {
        return m_context.computeQueue != VK_NULL_HANDLE ? m_context.computeFamily
                                                        : m_context.graphicsFamily;
    }

    // A full memory barrier (all writes -> all reads/writes, all stages).
    // Recorded first and last in each Sabercat submission so that, on the
    // same queue, work a consumer submitted earlier is complete and visible
    // before Sabercat touches a buffer, and Sabercat's writes are visible to
    // anything submitted later -- the consumer needs no barriers of its own.
    static void recordGlobalBarrier(VkCommandBuffer cmd);

    // Whether the graphics queue family can present to `surface`. The logical
    // device is created headlessly in initialize(), before any app-created
    // surface exists, so this can't be checked/selected at device-creation
    // time -- call it from attachSurface() once a surface is available.
    bool supportsPresent(VkSurfaceKHR surface) const;

  private:
    void createInstance(const std::vector<const char *> & instanceExtensions,
                        bool enableValidation);
    void setupDebugMessenger();
    void pickPhysicalDevice();
    void queryFeatures();
    void createLogicalDevice(bool enableValidation);
    void createAllocator();
    void createTransferPool();

    // The Vulkan handles themselves live in the borrowed view; m_ownsContext
    // decides whether shutdown() destroys them.
    VulkanContext m_context{};
    bool m_ownsContext = false;

    // Always Sabercat-owned, even over a borrowed device: the allocator is
    // deliberately not part of the shared context (see docs/library.md), and the
    // transfer pool is a load-time-only helper.
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    VkCommandPool m_transferPool = VK_NULL_HANDLE;

    VkDebugUtilsMessengerEXT m_debugMessenger = VK_NULL_HANDLE;

    VkPhysicalDeviceProperties m_properties{};
    DeviceFeatures m_features{};
    EnabledDeviceFeatures m_enabled{};
    bool m_validationEnabled = false;
    bool m_surfaceExtensionEnabled = false;
};

} // namespace sabercat
