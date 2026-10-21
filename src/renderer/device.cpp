// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/device.h"

#include <cstring>
#include <stdexcept>
#include <vector>

#include "sabercat/logger.h"
#include "sabercat/vk_check.h"

namespace sabercat
{

namespace
{

const char * const kValidationLayer = "VK_LAYER_KHRONOS_validation";

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                             VkDebugUtilsMessageTypeFlagsEXT /*type*/,
                                             const VkDebugUtilsMessengerCallbackDataEXT * data,
                                             void * /*userData*/)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
    {
        Logger::error(std::string("[validation] ") + data->pMessage);
    }
    else if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
    {
        Logger::warning(std::string("[validation] ") + data->pMessage);
    }
    return VK_FALSE;
}

bool validationLayerAvailable()
{
    uint32_t count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> layers(count);
    vkEnumerateInstanceLayerProperties(&count, layers.data());

    for (const auto & layer : layers)
    {
        if (std::strcmp(layer.layerName, kValidationLayer) == 0)
        {
            return true;
        }
    }
    return false;
}

std::vector<VkQueueFamilyProperties> queueFamilies(VkPhysicalDevice physicalDevice)
{
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &count, families.data());
    return families;
}

std::string versionString(uint32_t version)
{
    return std::to_string(VK_API_VERSION_MAJOR(version)) + "."
           + std::to_string(VK_API_VERSION_MINOR(version)) + "."
           + std::to_string(VK_API_VERSION_PATCH(version));
}

// Borrowed-context checks: std::invalid_argument, because a failure is the
// caller handing Sabercat something it cannot use, not a runtime fault.
void require(bool condition, const std::string & message)
{
    if (!condition)
    {
        throw std::invalid_argument("borrowed Vulkan context: " + message);
    }
}

// What the GPU supports, for the features DeviceRequirements may enable.
struct SupportedFeatures
{
    VkPhysicalDeviceVulkan11Features vulkan11{};
    VkPhysicalDeviceVulkan12Features vulkan12{};
    VkPhysicalDeviceVulkan13Features vulkan13{};
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloat{};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accelerationStructure{};
    VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{};
    bool atomicFloatExtension = false;
    bool swapchainExtension = false;
    // Acceleration structures (and the deferred-host-operations
    // extension they depend on) and ray query.
    bool rayTracingExtensions = false;

    explicit SupportedFeatures(VkPhysicalDevice physicalDevice)
    {
        atomicFloatExtension =
            deviceExtensionAvailable(physicalDevice, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
        swapchainExtension =
            deviceExtensionAvailable(physicalDevice, VK_KHR_SWAPCHAIN_EXTENSION_NAME);
        rayTracingExtensions =
            deviceExtensionAvailable(physicalDevice, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)
            && deviceExtensionAvailable(physicalDevice,
                                        VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME)
            && deviceExtensionAvailable(physicalDevice, VK_KHR_RAY_QUERY_EXTENSION_NAME);

        vulkan11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        vulkan12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        vulkan13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        atomicFloat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
        accelerationStructure.sType =
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
        rayQueryFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
        vulkan11.pNext = &vulkan12;
        vulkan12.pNext = &vulkan13;
        // An extension's feature struct may only be chained when it exists.
        VkBaseOutStructure * tail = reinterpret_cast<VkBaseOutStructure *>(&vulkan13);
        const auto append = [&](void * next)
        {
            tail->pNext = static_cast<VkBaseOutStructure *>(next);
            tail = static_cast<VkBaseOutStructure *>(next);
        };
        if (atomicFloatExtension)
        {
            append(&atomicFloat);
        }
        if (rayTracingExtensions)
        {
            append(&accelerationStructure);
            append(&rayQueryFeatures);
        }

        VkPhysicalDeviceFeatures2 features2{};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &vulkan11;
        vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);
        vulkan13.pNext = nullptr;
        atomicFloat.pNext = nullptr;
        accelerationStructure.pNext = nullptr;
        rayQueryFeatures.pNext = nullptr;
    }

    bool atomicAdd() const
    {
        return atomicFloatExtension && atomicFloat.shaderBufferFloat32AtomicAdd;
    }

    bool rayQuery() const
    {
        return rayTracingExtensions && accelerationStructure.accelerationStructure
               && rayQueryFeatures.rayQuery && vulkan12.bufferDeviceAddress;
    }
};

void populateDebugInfo(VkDebugUtilsMessengerCreateInfoEXT & info)
{
    info = {};
    info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT
                           | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT
                       | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT
                       | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    info.pfnUserCallback = debugCallback;
}

} // namespace

bool deviceExtensionAvailable(VkPhysicalDevice physicalDevice, const char * name)
{
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> extensions(count);
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &count, extensions.data());

    for (const auto & ext : extensions)
    {
        if (std::strcmp(ext.extensionName, name) == 0)
        {
            return true;
        }
    }
    return false;
}

void validatePhysicalDevice(VkPhysicalDevice physicalDevice)
{
    VK_VALIDATE(physicalDevice != VK_NULL_HANDLE, "Sabercat device requirements: no physical device");
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physicalDevice, &props);
    VK_VALIDATE(props.apiVersion >= kRequiredVulkanApiVersion,
                std::string("Sabercat device requirements: ") + props.deviceName + " supports Vulkan "
                    + versionString(props.apiVersion) + ", Sabercat needs "
                    + versionString(kRequiredVulkanApiVersion));
    VK_VALIDATE(props.limits.maxComputeWorkGroupInvocations >= kRequiredComputeWorkGroupInvocations,
                std::string("Sabercat device requirements: ") + props.deviceName + " allows "
                    + std::to_string(props.limits.maxComputeWorkGroupInvocations)
                    + " compute invocations per workgroup, the tile rasterizer needs "
                    + std::to_string(kRequiredComputeWorkGroupInvocations));
}

DeviceRequirements::DeviceRequirements(VkPhysicalDevice physicalDevice, const Options & options)
{
    validatePhysicalDevice(physicalDevice);
    const SupportedFeatures supported(physicalDevice);
    using Need = DeviceRequirements::Need;

    m_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    m_vulkan11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    m_vulkan12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    m_vulkan13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    m_atomicFloat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
    m_accelerationStructure.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    m_rayQuery.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    m_features2.pNext = &m_vulkan11;
    m_vulkan11.pNext = &m_vulkan12;
    m_vulkan12.pNext = &m_vulkan13;
    m_tail = reinterpret_cast<VkBaseOutStructure *>(&m_vulkan13);

    // Perf-oriented extras the kernels may use later (PERFORMANCE.md
    // invariant #6): enabled whenever supported, never required.
    m_vulkan11.storageBuffer16BitAccess = supported.vulkan11.storageBuffer16BitAccess;
    m_vulkan11.uniformAndStorageBuffer16BitAccess =
        supported.vulkan11.uniformAndStorageBuffer16BitAccess;
    m_vulkan12.shaderFloat16 = supported.vulkan12.shaderFloat16;
    m_vulkan12.shaderInt8 = supported.vulkan12.shaderInt8;
    m_vulkan12.bufferDeviceAddress = supported.vulkan12.bufferDeviceAddress;

    // Training: backward()'s fp32 buffer atomics.
    if (options.training != Need::None)
    {
        VK_VALIDATE(supported.atomicAdd() || options.training != Need::Required,
                    "Sabercat device requirements: training needs VK_EXT_shader_atomic_float "
                    "with shaderBufferFloat32AtomicAdd, which this GPU lacks");
        if (supported.atomicAdd())
        {
            m_atomicFloat.shaderBufferFloat32AtomicAdd = VK_TRUE;
            append(&m_atomicFloat);
            m_deviceExtensions.push_back(VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
            m_enabled.shaderBufferFloat32AtomicAdd = true;
        }
    }

    // Presentation: swapchain (device) + surface (instance) + the dynamic
    // rendering present()'s UI callback draws with.
    if (options.presentation != Need::None)
    {
        const bool ok = supported.swapchainExtension && supported.vulkan13.dynamicRendering;
        VK_VALIDATE(ok || options.presentation != Need::Required,
                    "Sabercat device requirements: presentation needs VK_KHR_swapchain and "
                    "dynamicRendering, which this GPU lacks");
        if (supported.swapchainExtension)
        {
            m_deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
            m_instanceExtensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
            m_enabled.swapchain = true;
        }
        if (supported.vulkan13.dynamicRendering)
        {
            m_vulkan13.dynamicRendering = VK_TRUE;
            m_enabled.dynamicRendering = true;
        }
    }

    // Ray tracing: acceleration structures (which need
    // VK_KHR_deferred_host_operations enabled alongside, though Sabercat
    // builds on the device and never defers) + ray query + buffer device
    // address (AS build inputs and the BLAS reference in the TLAS instance
    // are device addresses). All or nothing: a partial set is useless.
    if (options.rayTracing != Need::None)
    {
        VK_VALIDATE(supported.rayQuery() || options.rayTracing != Need::Required,
                    "Sabercat device requirements: ray tracing needs VK_KHR_acceleration_structure, "
                    "VK_KHR_deferred_host_operations and VK_KHR_ray_query with the "
                    "accelerationStructure, rayQuery and bufferDeviceAddress features, which this "
                    "GPU lacks");
        if (supported.rayQuery())
        {
            m_vulkan12.bufferDeviceAddress = VK_TRUE;
            m_accelerationStructure.accelerationStructure = VK_TRUE;
            m_rayQuery.rayQuery = VK_TRUE;
            append(&m_accelerationStructure);
            append(&m_rayQuery);
            m_deviceExtensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
            m_deviceExtensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
            m_deviceExtensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
            m_enabled.rayQuery = true;
        }
    }

    m_enabled.apiVersion = kRequiredVulkanApiVersion;
}

void DeviceRequirements::append(void * next)
{
    m_tail->pNext = static_cast<VkBaseOutStructure *>(next);
    m_tail = static_cast<VkBaseOutStructure *>(next);
}

void validateBorrowedContext(const VulkanContext & context, const EnabledDeviceFeatures & enabled)
{
    require(context.instance != VK_NULL_HANDLE, "instance is VK_NULL_HANDLE");
    require(context.physicalDevice != VK_NULL_HANDLE, "physicalDevice is VK_NULL_HANDLE");
    require(context.device != VK_NULL_HANDLE, "device is VK_NULL_HANDLE");
    require(context.computeQueue != VK_NULL_HANDLE || context.graphicsQueue != VK_NULL_HANDLE,
            "neither a compute nor a graphics queue was given");

    // The physical device itself: the same hard floor a Sabercat-owned device enforces.
    try
    {
        validatePhysicalDevice(context.physicalDevice);
    }
    catch (const std::runtime_error & e)
    {
        throw std::invalid_argument(std::string("borrowed Vulkan context: ") + e.what());
    }

    // Queues: each supplied queue's family must exist and have the
    // capability Sabercat uses it for. Every Sabercat submission goes to the
    // compute queue (or the graphics queue if there is none), so that
    // family must support COMPUTE.
    const std::vector<VkQueueFamilyProperties> families = queueFamilies(context.physicalDevice);
    const auto checkFamily = [&](const char * which, uint32_t family, VkQueueFlags needed,
                                 const char * neededName)
    {
        require(family != UINT32_MAX && family != VK_QUEUE_FAMILY_IGNORED,
                std::string(which) + " queue given without its family index");
        require(family < families.size(),
                std::string(which) + " queue family " + std::to_string(family)
                    + " is out of range (the GPU has " + std::to_string(families.size()) + ")");
        require((families[family].queueFlags & needed) == needed,
                std::string(which) + " queue family " + std::to_string(family) + " lacks "
                    + neededName);
    };
    if (context.computeQueue != VK_NULL_HANDLE)
    {
        checkFamily("compute", context.computeFamily, VK_QUEUE_COMPUTE_BIT, "VK_QUEUE_COMPUTE_BIT");
    }
    if (context.graphicsQueue != VK_NULL_HANDLE)
    {
        checkFamily("graphics", context.graphicsFamily, VK_QUEUE_GRAPHICS_BIT,
                    "VK_QUEUE_GRAPHICS_BIT");
        if (context.computeQueue == VK_NULL_HANDLE)
        {
            checkFamily("graphics (used for all Sabercat work, no compute queue given)",
                        context.graphicsFamily, VK_QUEUE_COMPUTE_BIT, "VK_QUEUE_COMPUTE_BIT");
        }
    }

    // Declarations: the API version, and each feature declared enabled must
    // at least be supported (an owner cannot have enabled the unsupported).
    require(enabled.apiVersion >= kRequiredVulkanApiVersion,
            "declared apiVersion " + versionString(enabled.apiVersion) + " is below the required "
                + versionString(kRequiredVulkanApiVersion)
                + " (EnabledDeviceFeatures::apiVersion must be the instance's apiVersion)");
    const SupportedFeatures supported(context.physicalDevice);
    require(!enabled.shaderBufferFloat32AtomicAdd || supported.atomicAdd(),
            "shaderBufferFloat32AtomicAdd declared enabled, but the GPU does not support "
            "VK_EXT_shader_atomic_float / shaderBufferFloat32AtomicAdd");
    require(!enabled.dynamicRendering || supported.vulkan13.dynamicRendering,
            "dynamicRendering declared enabled, but the GPU does not support it");
    if (enabled.swapchain)
    {
        require(supported.swapchainExtension,
                "swapchain declared enabled, but the GPU lacks VK_KHR_swapchain");
        require(context.graphicsQueue != VK_NULL_HANDLE,
                "swapchain declared enabled, but no graphics queue was given to present on");
        // Device-level commands of an extension that was not enabled resolve
        // to null, so this one declaration Vulkan CAN confirm.
        require(vkGetDeviceProcAddr(context.device, "vkCreateSwapchainKHR") != nullptr,
                "swapchain declared enabled, but VK_KHR_swapchain is not enabled on the device "
                "(vkGetDeviceProcAddr(vkCreateSwapchainKHR) is null)");
    }
    if (enabled.rayQuery)
    {
        require(supported.rayQuery(),
                "rayQuery declared enabled, but the GPU does not support VK_KHR_acceleration_structure "
                "+ VK_KHR_deferred_host_operations + VK_KHR_ray_query with accelerationStructure, "
                "rayQuery and bufferDeviceAddress");
        // The two extensions that have commands can be confirmed the same way
        // as the swapchain. VK_KHR_ray_query (no commands) and the feature
        // bits cannot.
        require(vkGetDeviceProcAddr(context.device, "vkCreateAccelerationStructureKHR") != nullptr,
                "rayQuery declared enabled, but VK_KHR_acceleration_structure is not enabled on the "
                "device (vkGetDeviceProcAddr(vkCreateAccelerationStructureKHR) is null)");
        require(vkGetDeviceProcAddr(context.device, "vkCreateDeferredOperationKHR") != nullptr,
                "rayQuery declared enabled, but VK_KHR_deferred_host_operations is not enabled on "
                "the device (vkGetDeviceProcAddr(vkCreateDeferredOperationKHR) is null)");
    }
}

Device::Device() = default;

Device::~Device()
{
    shutdown();
}

void Device::initialize(const std::vector<const char *> & instanceExtensions, bool enableValidation)
{
    // A second call would overwrite the live handles and leak the first
    // instance and device. shutdown() clears them, so re-initializing after
    // it stays allowed.
    VK_VALIDATE(m_context.instance == VK_NULL_HANDLE && m_context.device == VK_NULL_HANDLE,
                "Device::initialize called on an already-initialized Device");
    m_ownsContext = true;
    m_validationEnabled = enableValidation && validationLayerAvailable();
    if (enableValidation && !m_validationEnabled)
    {
        Logger::warning("Validation layer requested but not available; continuing without it");
    }

    createInstance(instanceExtensions, m_validationEnabled);
    setupDebugMessenger();
    pickPhysicalDevice();
    queryFeatures();
    createLogicalDevice(m_validationEnabled);
    createAllocator();

    createTransferPool();

    Logger::info("Vulkan device initialized");
}

void Device::createInstance(const std::vector<const char *> & instanceExtensions,
                            bool enableValidation)
{
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Sabercat";
    appInfo.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    appInfo.pEngineName = "Sabercat";
    appInfo.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    appInfo.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    // Start from whatever the caller (application) requires, then add our own.
    std::vector<const char *> extensions = instanceExtensions;
    if (enableValidation)
    {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    // VK_KHR_swapchain (a device extension) requires VK_KHR_surface (an
    // instance extension) to be enabled too. Headless callers pass no
    // instance extensions, so remember whether it's actually present here --
    // createLogicalDevice() must not try to enable swapchain support
    // without it, or vkCreateDevice fails validation outright.
    for (const char * ext : extensions)
    {
        if (std::strcmp(ext, VK_KHR_SURFACE_EXTENSION_NAME) == 0)
        {
            m_surfaceExtensionEnabled = true;
            break;
        }
    }

    VkDebugUtilsMessengerCreateInfoEXT debugInfo{};
    if (enableValidation)
    {
        createInfo.enabledLayerCount = 1;
        createInfo.ppEnabledLayerNames = &kValidationLayer;
        populateDebugInfo(debugInfo);
        createInfo.pNext = &debugInfo;
    }

    VK_CHECK(vkCreateInstance(&createInfo, nullptr, &m_context.instance),
             "Failed to create Vulkan instance");
}

void Device::setupDebugMessenger()
{
    if (!m_validationEnabled)
    {
        return;
    }

    VkDebugUtilsMessengerCreateInfoEXT info{};
    populateDebugInfo(info);

    auto func = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(m_context.instance, "vkCreateDebugUtilsMessengerEXT"));
    if (func != nullptr)
    {
        func(m_context.instance, &info, nullptr, &m_debugMessenger);
    }
}

void Device::pickPhysicalDevice()
{
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(m_context.instance, &count, nullptr);
    VK_VALIDATE(count > 0, "No Vulkan-capable GPU found");

    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(m_context.instance, &count, devices.data());

    // Prefer a discrete GPU; fall back to the first device otherwise.
    for (const auto & device : devices)
    {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(device, &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
        {
            m_context.physicalDevice = device;
            break;
        }
    }
    if (m_context.physicalDevice == VK_NULL_HANDLE)
    {
        m_context.physicalDevice = devices[0];
    }

    vkGetPhysicalDeviceProperties(m_context.physicalDevice, &m_properties);
    Logger::info(std::string("Using GPU: ") + m_properties.deviceName);
    validatePhysicalDevice(m_context.physicalDevice);
}

void Device::queryFeatures()
{
    // Subgroup (wave/warp) properties -- queried via the properties2 chain,
    // not the features chain.
    VkPhysicalDeviceSubgroupProperties subgroupProps{};
    subgroupProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;

    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &subgroupProps;
    vkGetPhysicalDeviceProperties2(m_context.physicalDevice, &props2);

    m_features.subgroupSize = subgroupProps.subgroupSize;
    m_features.subgroupBasic =
        (subgroupProps.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) != 0;
    m_features.subgroupArithmetic =
        (subgroupProps.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
    m_features.subgroupBallot =
        (subgroupProps.supportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT) != 0;
    m_features.subgroupShuffle =
        (subgroupProps.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) != 0;

    // Feature support chain -- each struct is chained via pNext so one call
    // queries all of them at once.
    // VK_EXT_shader_atomic_float is an extension (not core in 1.3), so its
    // feature struct may only be chained when the extension exists.
    const bool hasAtomicFloatExtension = deviceExtensionAvailable(
        m_context.physicalDevice, VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloatFeatures{};
    atomicFloatFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;

    VkPhysicalDeviceDynamicRenderingFeatures dynamicRenderingFeatures{};
    dynamicRenderingFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES;
    dynamicRenderingFeatures.pNext = hasAtomicFloatExtension ? &atomicFloatFeatures : nullptr;

    VkPhysicalDeviceBufferDeviceAddressFeatures bdaFeatures{};
    bdaFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES;
    bdaFeatures.pNext = &dynamicRenderingFeatures;

    VkPhysicalDevice16BitStorageFeatures storage16Features{};
    storage16Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    storage16Features.pNext = &bdaFeatures;

    VkPhysicalDeviceShaderFloat16Int8Features float16Int8Features{};
    float16Int8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES;
    float16Int8Features.pNext = &storage16Features;

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &float16Int8Features;
    vkGetPhysicalDeviceFeatures2(m_context.physicalDevice, &features2);

    m_features.shaderBufferFloat32AtomicAdd =
        hasAtomicFloatExtension && atomicFloatFeatures.shaderBufferFloat32AtomicAdd;

    m_features.shaderFloat16 = float16Int8Features.shaderFloat16;
    m_features.shaderInt8 = float16Int8Features.shaderInt8;
    m_features.storageBuffer16BitAccess = storage16Features.storageBuffer16BitAccess;
    m_features.uniformAndStorageBuffer16BitAccess =
        storage16Features.uniformAndStorageBuffer16BitAccess;
    m_features.bufferDeviceAddress = bdaFeatures.bufferDeviceAddress;
    m_features.dynamicRendering = dynamicRenderingFeatures.dynamicRendering;
    // Support only; createLogicalDevice()/adopt() replace it with what is
    // actually enabled/declared.
    m_features.rayQuery = SupportedFeatures(m_context.physicalDevice).rayQuery();

    // Cooperative matrix/vector: detect extension presence only. Not enabled,
    // not used by the core rasterizer -- reserved for a future neural module.
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(m_context.physicalDevice, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> extensions(extCount);
    vkEnumerateDeviceExtensionProperties(
        m_context.physicalDevice, nullptr, &extCount, extensions.data());

    for (const auto & ext : extensions)
    {
        if (std::strcmp(ext.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0)
        {
            m_features.hasCooperativeMatrix = true;
        }
        if (std::strcmp(ext.extensionName, VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME) == 0)
        {
            m_features.hasCooperativeVector = true;
        }
    }

    Logger::info("Device capabilities:");
    Logger::info("  subgroup size: " + std::to_string(m_features.subgroupSize)
                 + " (basic=" + (m_features.subgroupBasic ? "y" : "n")
                 + " arithmetic=" + (m_features.subgroupArithmetic ? "y" : "n")
                 + " ballot=" + (m_features.subgroupBallot ? "y" : "n")
                 + " shuffle=" + (m_features.subgroupShuffle ? "y" : "n") + ")");
    Logger::info(std::string("  shaderFloat16: ") + (m_features.shaderFloat16 ? "yes" : "no"));
    Logger::info(std::string("  shaderInt8: ") + (m_features.shaderInt8 ? "yes" : "no"));
    Logger::info(std::string("  16-bit storage: ")
                 + (m_features.storageBuffer16BitAccess ? "yes" : "no"));
    Logger::info(std::string("  bufferDeviceAddress: ")
                 + (m_features.bufferDeviceAddress ? "yes" : "no"));
    Logger::info(std::string("  cooperative matrix (detected): ")
                 + (m_features.hasCooperativeMatrix ? "yes" : "no"));
    Logger::info(std::string("  cooperative vector (detected): ")
                 + (m_features.hasCooperativeVector ? "yes" : "no"));
    Logger::info(std::string("  dynamicRendering: ") + (m_features.dynamicRendering ? "yes" : "no"));
    Logger::info(std::string("  shaderBufferFloat32AtomicAdd: ")
                 + (m_features.shaderBufferFloat32AtomicAdd ? "yes" : "no (backward unavailable)"));
    Logger::info(std::string("  ray query + acceleration structures: ")
                 + (m_features.rayQuery ? "yes" : "no (Gaussian ray tracing unavailable)"));
}

void Device::createLogicalDevice(bool enableValidation)
{
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(m_context.physicalDevice, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(m_context.physicalDevice, &count, families.data());

    for (uint32_t i = 0; i < count; ++i)
    {
        if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
            && m_context.graphicsFamily == UINT32_MAX)
        {
            m_context.graphicsFamily = i;
        }
        if ((families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
            && m_context.computeFamily == UINT32_MAX)
        {
            m_context.computeFamily = i;
        }
    }

    VK_VALIDATE(m_context.graphicsFamily != UINT32_MAX && m_context.computeFamily != UINT32_MAX,
                "GPU lacks required graphics/compute queues");

    // Prefer a compute family distinct from the graphics family (a true
    // dedicated queue, e.g. AMD's async compute family). If the GPU only
    // exposes one family that does both, fall back to a second queue *index*
    // within that family so async compute (invariant #5) still has a queue
    // of its own to submit to.
    bool sameFamily = (m_context.computeFamily == m_context.graphicsFamily);
    uint32_t sharedFamilyQueueCount =
        sameFamily ? families[m_context.graphicsFamily].queueCount : 0;
    bool secondQueueInSharedFamily = sameFamily && sharedFamilyQueueCount >= 2;
    m_features.hasDedicatedComputeQueue = !sameFamily || secondQueueInSharedFamily;

    std::vector<uint32_t> uniqueFamilies = {m_context.graphicsFamily};
    if (!sameFamily)
    {
        uniqueFamilies.push_back(m_context.computeFamily);
    }

    float priorities[2] = {1.0f, 1.0f};
    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    for (uint32_t family : uniqueFamilies)
    {
        VkDeviceQueueCreateInfo qi{};
        qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qi.queueFamilyIndex = family;
        qi.queueCount = (family == m_context.graphicsFamily && secondQueueInSharedFamily) ? 2 : 1;
        qi.pQueuePriorities = priorities;
        queueInfos.push_back(qi);
    }

    // The same DeviceRequirements a borrowing application uses, so what
    // Sabercat creates and what it asks a borrowed device for cannot drift
    // apart. Training is IfSupported (forward/viewer work without fp32
    // atomics). Presentation is requested only when the instance has a
    // surface extension: VK_KHR_swapchain requires VK_KHR_surface, and
    // headless callers pass no instance extensions. Ray tracing is
    // IfSupported like training: only GaussianAccelerationStructure needs it.
    DeviceRequirements requirements(
        m_context.physicalDevice,
        {DeviceRequirements::Need::IfSupported,
         m_surfaceExtensionEnabled ? DeviceRequirements::Need::IfSupported
                                   : DeviceRequirements::Need::None,
         DeviceRequirements::Need::IfSupported});
    const EnabledDeviceFeatures enabled = requirements.enabled();
    if (m_surfaceExtensionEnabled && !enabled.swapchain)
    {
        Logger::warning("VK_KHR_swapchain not available on this GPU; presentation will be unavailable");
    }
    m_enabled = enabled;
    m_features.shaderBufferFloat32AtomicAdd = enabled.shaderBufferFloat32AtomicAdd;
    m_features.dynamicRendering = enabled.dynamicRendering;
    m_features.rayQuery = enabled.rayQuery;
    // bufferDeviceAddress keeps queryFeatures()'s support value: the
    // requirements chain enables BDA whenever it is supported (a perf extra,
    // and part of rayTracing), so here support == enabled.
    const std::vector<const char *> & deviceExtensions = requirements.deviceExtensions();
    VkPhysicalDeviceFeatures2 & features2 =
        *static_cast<VkPhysicalDeviceFeatures2 *>(requirements.featureChain());

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pNext = &features2;
    createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueInfos.size());
    createInfo.pQueueCreateInfos = queueInfos.data();
    createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    createInfo.ppEnabledExtensionNames = deviceExtensions.data();
    // pEnabledFeatures must be null when pNext carries a
    // VkPhysicalDeviceFeatures2 -- the base VkPhysicalDeviceFeatures fields
    // we don't otherwise touch stay disabled, which is fine for this device.

    if (enableValidation)
    {
        createInfo.enabledLayerCount = 1;
        createInfo.ppEnabledLayerNames = &kValidationLayer;
    }

    VK_CHECK(vkCreateDevice(m_context.physicalDevice, &createInfo, nullptr, &m_context.device),
             "Failed to create logical device");

    vkGetDeviceQueue(m_context.device, m_context.graphicsFamily, 0, &m_context.graphicsQueue);
    vkGetDeviceQueue(m_context.device,
                     m_context.computeFamily,
                     secondQueueInSharedFamily ? 1 : 0,
                     &m_context.computeQueue);

    Logger::info(std::string("  dedicated compute queue: ")
                 + (m_features.hasDedicatedComputeQueue ? "yes" : "no (shared, index 0)"));
}

void Device::createAllocator()
{
    VmaAllocatorCreateInfo info{};
    info.physicalDevice = m_context.physicalDevice;
    info.device = m_context.device;
    info.instance = m_context.instance;
    info.vulkanApiVersion = VK_API_VERSION_1_3;
    // Buffers with SHADER_DEVICE_ADDRESS usage (acceleration-
    // structure storage, build inputs, scratch) need their memory allocated
    // with VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT, which VMA adds only with
    // this flag. Set only when BDA is ENABLED (Sabercat-owned device: whenever supported;
    // borrowed device: as part of the rayQuery declaration) -- that allocate flag on
    // a device without the feature is itself a validation error.
    if (m_features.bufferDeviceAddress)
    {
        info.flags |= VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    }

    VK_CHECK(vmaCreateAllocator(&info, &m_allocator), "Failed to create VMA allocator");
}

void Device::adopt(const VulkanContext & context, const EnabledDeviceFeatures & enabled)
{
    VK_VALIDATE(m_context.device == VK_NULL_HANDLE,
                "Device::adopt called on an already-initialized Device");
    validateBorrowedContext(context, enabled);

    m_ownsContext = false;
    m_context = context;

    // A borrowed context arrives with no debug messenger of our own; the
    // owner decides whether validation is on.
    m_validationEnabled = false;

    vkGetPhysicalDeviceProperties(m_context.physicalDevice, &m_properties);
    Logger::info(std::string("Adopted borrowed Vulkan context on GPU: ") + m_properties.deviceName);

    // Capability detection is a property of the physical device, so it works
    // identically over a borrowed one -- but on a borrowed device the owner chose
    // what was ENABLED, and we cannot enable more after the fact. So the
    // features Sabercat acts on report the owner's (validated) declaration,
    // and the optional extras it never uses report false rather than guess.
    queryFeatures();
    m_features.shaderBufferFloat32AtomicAdd = enabled.shaderBufferFloat32AtomicAdd;
    m_features.dynamicRendering = enabled.dynamicRendering;
    m_features.shaderFloat16 = false;
    m_features.shaderInt8 = false;
    m_features.storageBuffer16BitAccess = false;
    m_features.uniformAndStorageBuffer16BitAccess = false;
    // BDA is part of the rayQuery declaration, not declared alone.
    m_features.bufferDeviceAddress = enabled.rayQuery;
    m_features.rayQuery = enabled.rayQuery;
    m_features.hasDedicatedComputeQueue =
        m_context.computeQueue != VK_NULL_HANDLE && m_context.computeQueue != m_context.graphicsQueue;
    m_enabled = enabled;

    // The allocator and transfer pool are Sabercat-owned even over a
    // borrowed device (docs/library.md: the allocator is not part of the shared
    // context), so we create -- and later destroy -- both.
    createAllocator();
    createTransferPool();
}

void Device::createTransferPool()
{
    // Command pool for one-off transfer / immediate-submit work, on the same
    // queue the compute passes use (one queue for everything
    // Sabercat submits except present() -- see docs/library.md's synchronization
    // contract). TRANSIENT: these command buffers are short-lived.
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    poolInfo.queueFamilyIndex = submitQueueFamily();
    VK_CHECK(vkCreateCommandPool(m_context.device, &poolInfo, nullptr, &m_transferPool),
             "Failed to create transfer command pool");
}

void Device::recordGlobalBarrier(VkCommandBuffer cmd)
{
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_MEMORY_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 1, &barrier, 0, nullptr, 0,
                         nullptr);
}

void Device::shutdown()
{
    // The command pool is owned by the device, so it must be destroyed while the
    // device handle is still valid -- i.e. before vkDestroyDevice.
    if (m_transferPool != VK_NULL_HANDLE)
    {
        vkDestroyCommandPool(m_context.device, m_transferPool, nullptr);
        m_transferPool = VK_NULL_HANDLE;
    }
    if (m_allocator != VK_NULL_HANDLE)
    {
        vmaDestroyAllocator(m_allocator);
        m_allocator = VK_NULL_HANDLE;
    }

    // For a borrowed device the context is borrowed: the application
    // created these handles and destroys them itself. Drop our view and
    // stop here.
    if (!m_ownsContext)
    {
        m_context = VulkanContext{};
        return;
    }

    if (m_context.device != VK_NULL_HANDLE)
    {
        vkDestroyDevice(m_context.device, nullptr);
        m_context.device = VK_NULL_HANDLE;
    }
    if (m_debugMessenger != VK_NULL_HANDLE)
    {
        auto func = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(m_context.instance, "vkDestroyDebugUtilsMessengerEXT"));
        if (func != nullptr)
        {
            func(m_context.instance, m_debugMessenger, nullptr);
        }
        m_debugMessenger = VK_NULL_HANDLE;
    }
    if (m_context.instance != VK_NULL_HANDLE)
    {
        vkDestroyInstance(m_context.instance, nullptr);
        m_context.instance = VK_NULL_HANDLE;
    }
}

void Device::immediateSubmit(const std::function<void(VkCommandBuffer)> & record)
{
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_transferPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(m_context.device, &allocInfo, &cmd),
             "Failed to allocate command buffer");
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo), "Failed to begin command buffer");
    // Bracketed like every compute-pass submission: ordered after whatever a
    // consumer submitted to this queue before calling in, and visible to
    // whatever it submits after (docs/library.md's synchronization contract).
    recordGlobalBarrier(cmd);
    record(cmd);
    recordGlobalBarrier(cmd);
    VK_CHECK(vkEndCommandBuffer(cmd), "Failed to end command buffer");
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence;
    VK_CHECK(vkCreateFence(m_context.device, &fenceInfo, nullptr, &fence),
             "Failed to create fence");
    VK_CHECK(vkQueueSubmit(submitQueue(), 1, &submitInfo, fence), "Failed to submit queue");
    VK_CHECK(vkWaitForFences(m_context.device, 1, &fence, VK_TRUE, UINT64_MAX),
             "Failed to wait for fence");
    vkDestroyFence(m_context.device, fence, nullptr);
    vkFreeCommandBuffers(m_context.device, m_transferPool, 1, &cmd);
}

bool Device::supportsPresent(VkSurfaceKHR surface) const
{
    VkBool32 supported = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(
        m_context.physicalDevice, m_context.graphicsFamily, surface, &supported);
    return supported == VK_TRUE;
}

} // namespace sabercat
