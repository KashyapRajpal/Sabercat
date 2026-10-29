// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Smoke test for sabercat::Device.
// Brings up the full Vulkan context (instance, GPU, queues, VMA allocator)
// headlessly and checks it against an independent query of the same GPU.
// Covers the Device lifecycle (double initialize, adopt() rejections and a
// borrowed round trip), DeviceRequirements, and buffer transfers at several
// sizes and at 64 MiB. Exits non-zero on any failure.

#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "sabercat/buffer.h"
#include "sabercat/device.h"

using namespace sabercat;

namespace
{

int g_failures = 0;

void check(bool condition, const std::string & message)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << message << std::endl;
        ++g_failures;
    }
}

// Round-trips `bytes` through a Buffer of that size and reports the
// mismatch count (bytes that differ).
size_t roundTripMismatches(Device & device, const std::vector<uint8_t> & src)
{
    Buffer buffer(device, src.size());
    buffer.upload(src.data(), src.size());
    std::vector<uint8_t> dst(src.size(), 0xEE);
    buffer.download(dst.data(), dst.size());
    size_t mismatches = 0;
    for (size_t i = 0; i < src.size(); ++i)
    {
        mismatches += dst[i] != src[i] ? 1 : 0;
    }
    return mismatches;
}

// Independent query of the GPU's feature and property support, through the
// pNext chains rather than Sabercat's DeviceFeatures.
struct GpuSupport
{
    VkPhysicalDeviceSubgroupProperties subgroup{};
    VkPhysicalDeviceVulkan11Features v11{};
    VkPhysicalDeviceVulkan12Features v12{};
    VkPhysicalDeviceVulkan13Features v13{};
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT atomicFloat{};
    VkPhysicalDeviceAccelerationStructureFeaturesKHR accel{};
    VkPhysicalDeviceRayQueryFeaturesKHR rayQuery{};
    VkPhysicalDeviceProperties properties{};
};

GpuSupport querySupport(VkPhysicalDevice gpu)
{
    GpuSupport s;
    s.subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &s.subgroup;
    vkGetPhysicalDeviceProperties2(gpu, &props2);
    s.properties = props2.properties;

    s.v11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    s.v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    s.v13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    s.atomicFloat.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
    s.accel.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    s.rayQuery.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    s.v11.pNext = &s.v12;
    s.v12.pNext = &s.v13;
    s.v13.pNext = &s.atomicFloat;
    s.atomicFloat.pNext = &s.accel;
    s.accel.pNext = &s.rayQuery;
    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &s.v11;
    vkGetPhysicalDeviceFeatures2(gpu, &features2);
    return s;
}

bool hasBit(VkQueueFlags flags, VkQueueFlags bit)
{
    return (flags & bit) == bit;
}

// Subgroup size, queue families and handles, and the feature flags against
// the GPU's own answers.
void testDeviceState(Device & device)
{
    const GpuSupport s = querySupport(device.physicalDevice());
    const DeviceFeatures & f = device.features();

    check(f.subgroupSize > 0, "subgroupSize must be positive");
    check((f.subgroupSize & (f.subgroupSize - 1)) == 0 && f.subgroupSize <= 128,
          "subgroupSize must be a power of two no larger than 128");
    check(f.subgroupSize == s.subgroup.subgroupSize,
          "subgroupSize " + std::to_string(f.subgroupSize) + " != GPU's "
              + std::to_string(s.subgroup.subgroupSize));

    const VkSubgroupFeatureFlags ops = s.subgroup.supportedOperations;
    check(f.subgroupBasic == ((ops & VK_SUBGROUP_FEATURE_BASIC_BIT) != 0),
          "subgroupBasic disagrees with supportedOperations");
    check(f.subgroupArithmetic == ((ops & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0),
          "subgroupArithmetic disagrees with supportedOperations");
    check(f.subgroupBallot == ((ops & VK_SUBGROUP_FEATURE_BALLOT_BIT) != 0),
          "subgroupBallot disagrees with supportedOperations");
    check(f.subgroupShuffle == ((ops & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) != 0),
          "subgroupShuffle disagrees with supportedOperations");

    // Queue families: both valid and with the capability each is used for.
    uint32_t familyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(device.physicalDevice(), &familyCount, nullptr);
    std::vector<VkQueueFamilyProperties> families(familyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(device.physicalDevice(), &familyCount,
                                             families.data());
    check(device.graphicsQueueFamily() < familyCount, "graphics queue family out of range");
    check(device.computeQueueFamily() < familyCount, "compute queue family out of range");
    check(device.graphicsQueueFamily() < familyCount
              && hasBit(families[device.graphicsQueueFamily()].queueFlags, VK_QUEUE_GRAPHICS_BIT),
          "graphics queue family lacks VK_QUEUE_GRAPHICS_BIT");
    check(device.computeQueueFamily() < familyCount
              && hasBit(families[device.computeQueueFamily()].queueFlags, VK_QUEUE_COMPUTE_BIT),
          "compute queue family lacks VK_QUEUE_COMPUTE_BIT");
    check(device.computeQueue() != VK_NULL_HANDLE && device.graphicsQueue() != VK_NULL_HANDLE,
          "queue handles must be non-null");
    check(device.submitQueue() == device.computeQueue()
              && device.submitQueueFamily() == device.computeQueueFamily(),
          "submitQueue must be the compute queue when there is one");
    check(f.hasDedicatedComputeQueue == (device.computeQueue() != device.graphicsQueue()),
          "hasDedicatedComputeQueue disagrees with the queue handles");
    check(device.allocator() != VK_NULL_HANDLE, "VMA allocator must be created");

    // Feature flags against the GPU's support. Model A enables every
    // supported optional feature, so the reported flag equals support.
    check(f.shaderFloat16 == (s.v12.shaderFloat16 == VK_TRUE), "shaderFloat16 != GPU support");
    check(f.shaderInt8 == (s.v12.shaderInt8 == VK_TRUE), "shaderInt8 != GPU support");
    check(f.storageBuffer16BitAccess == (s.v11.storageBuffer16BitAccess == VK_TRUE),
          "16-bit storage != GPU support");
    check(f.uniformAndStorageBuffer16BitAccess
              == (s.v11.uniformAndStorageBuffer16BitAccess == VK_TRUE),
          "uniform+storage 16-bit != GPU support");
    check(f.bufferDeviceAddress == (s.v12.bufferDeviceAddress == VK_TRUE),
          "bufferDeviceAddress != GPU support");
    // Presentation is only requested with a surface extension, so a headless
    // device reports dynamicRendering as NOT enabled even on a GPU that has it.
    check(f.dynamicRendering == device.enabledFeatures().dynamicRendering,
          "dynamicRendering must report the enabled declaration");
    check(!f.dynamicRendering || s.v13.dynamicRendering == VK_TRUE,
          "dynamicRendering enabled on a GPU without it");
    check(f.shaderBufferFloat32AtomicAdd == (s.atomicFloat.shaderBufferFloat32AtomicAdd == VK_TRUE),
          "shaderBufferFloat32AtomicAdd != GPU support");
    // Ray query needs all three features together, and buffer device
    // address is part of it, so a set rayQuery implies bufferDeviceAddress.
    const bool rayQuerySupported = s.accel.accelerationStructure == VK_TRUE
                                   && s.rayQuery.rayQuery == VK_TRUE
                                   && s.v12.bufferDeviceAddress == VK_TRUE;
    check(f.rayQuery == rayQuerySupported, "rayQuery != (AS + ray query + BDA support)");
    check(!f.rayQuery || f.bufferDeviceAddress, "rayQuery without bufferDeviceAddress");
    check(device.enabledFeatures().apiVersion >= kRequiredVulkanApiVersion,
          "enabled apiVersion below the required Vulkan 1.3");
    check(s.properties.apiVersion >= kRequiredVulkanApiVersion,
          "GPU reports Vulkan below the required 1.3");

    std::cout << "device state: subgroup " << f.subgroupSize << " (GPU " << s.subgroup.subgroupSize
              << "), families graphics " << device.graphicsQueueFamily() << " / compute "
              << device.computeQueueFamily() << " of " << familyCount << ", dedicated compute "
              << (f.hasDedicatedComputeQueue ? "yes" : "no") << ", rayQuery "
              << (f.rayQuery ? "yes" : "no") << "\n";
}

// Buffer transfers at edge sizes, then a large random buffer.
void testTransfers(Device & device)
{
    // Edges: 1 byte, sizes that are not multiples of 4, and one that is.
    const size_t sizes[] = {1, 3, 4, 7, 255, 4096};
    for (size_t size : sizes)
    {
        std::vector<uint8_t> src(size);
        for (size_t i = 0; i < size; ++i)
        {
            src[i] = static_cast<uint8_t>(i * 37 + 11);
        }
        const size_t mismatches = roundTripMismatches(device, src);
        check(mismatches == 0,
              "round trip of " + std::to_string(size) + " bytes: " + std::to_string(mismatches)
                  + " bytes differ");
    }

    // Scale: 64 MiB of pseudo-random bytes from a fixed seed.
    std::mt19937 rng(20261020u);
    std::vector<uint8_t> big(64u * 1024u * 1024u);
    for (uint8_t & b : big)
    {
        b = static_cast<uint8_t>(rng());
    }
    const size_t mismatches = roundTripMismatches(device, big);
    check(mismatches == 0, "64 MiB round trip: " + std::to_string(mismatches) + " bytes differ");
    std::cout << "buffer transfers: 6 edge sizes (1..4096 B) and 64 MiB, mismatches "
              << mismatches << "\n";
}

// Negative: a second initialize() on a live Device must throw and leave
// the device working.
void testDoubleInitialize()
{
    Device device;
    device.initialize(/*instanceExtensions=*/{}, /*enableValidation=*/true);
    const VkDevice before = device.handle();
    bool threw = false;
    try
    {
        device.initialize(/*instanceExtensions=*/{}, /*enableValidation=*/true);
    }
    catch (const std::runtime_error & e)
    {
        threw = std::string(e.what()).find("already-initialized") != std::string::npos;
    }
    check(threw, "a second initialize() must throw std::runtime_error naming the state");
    check(device.handle() == before, "a rejected initialize() must leave the handles unchanged");
    std::vector<uint8_t> src(256, 0x5A);
    check(roundTripMismatches(device, src) == 0, "device unusable after a rejected initialize()");
    device.shutdown();

    // After shutdown the Device may be initialized again.
    device.initialize(/*instanceExtensions=*/{}, /*enableValidation=*/false);
    check(device.handle() != VK_NULL_HANDLE, "re-initialize after shutdown must succeed");
    device.shutdown();
    std::cout << "double initialize: throws, device unchanged; re-initialize after shutdown OK\n";
}

// adopt() rejects a null device and an out-of-range family, and a Device
// that already owns a context cannot adopt another.
void testAdoptRejections(Device & owner)
{
    VulkanContext bad = owner.context();
    bad.device = VK_NULL_HANDLE;
    bool threw = false;
    Device a;
    try
    {
        a.adopt(bad, owner.enabledFeatures());
    }
    catch (const std::invalid_argument & e)
    {
        threw = std::string(e.what()).find("device") != std::string::npos;
    }
    check(threw, "adopt() with a null device must throw std::invalid_argument");

    bad = owner.context();
    bad.computeFamily = 9999;
    threw = false;
    try
    {
        a.adopt(bad, owner.enabledFeatures());
    }
    catch (const std::invalid_argument & e)
    {
        threw = std::string(e.what()).find("out of range") != std::string::npos;
    }
    check(threw, "adopt() with an out-of-range compute family must throw std::invalid_argument");

    // A rejected adopt leaves `a` empty and usable for a real one.
    check(!a.ownsContext() && a.handle() == VK_NULL_HANDLE,
          "a rejected adopt() must leave the Device without a context");
    threw = false;
    try
    {
        owner.adopt(owner.context(), owner.enabledFeatures());
    }
    catch (const std::runtime_error &)
    {
        threw = true;
    }
    check(threw, "adopt() on an already-initialized Device must throw");
    std::cout << "adopt: null device and out-of-range family rejected with invalid_argument\n";
}

// A borrower adopts the owner's context, works through it, shuts down, and
// the owner's handles survive.
void testBorrowedRoundTrip(Device & owner)
{
    Device borrower;
    borrower.adopt(owner.context(), owner.enabledFeatures());
    check(borrower.handle() == owner.handle() && borrower.submitQueue() == owner.submitQueue(),
          "adopted Device must share the owner's device and submit queue");
    check(!borrower.ownsContext(), "an adopted Device must not own its context");
    std::vector<uint8_t> src(1024);
    for (size_t i = 0; i < src.size(); ++i)
    {
        src[i] = static_cast<uint8_t>(i ^ 0x3C);
    }
    check(roundTripMismatches(borrower, src) == 0, "borrowed device round trip differs");
    borrower.shutdown();

    check(roundTripMismatches(owner, src) == 0,
          "owner unusable after the borrower shut down (it destroyed shared handles)");
    std::cout << "adopt round trip: borrower transfers OK; owner survives the borrower's shutdown\n";
}

// DeviceRequirements against this GPU: optional features enable when
// supported, nothing is enabled with None, and a null GPU is rejected.
void testRequirements(Device & device)
{
    const VkPhysicalDevice gpu = device.physicalDevice();

    DeviceRequirements all(gpu, {DeviceRequirements::Need::IfSupported,
                                 DeviceRequirements::Need::IfSupported,
                                 DeviceRequirements::Need::IfSupported});
    const EnabledDeviceFeatures e = all.enabled();
    check(e.apiVersion == kRequiredVulkanApiVersion, "requirements apiVersion must be 1.3");
    check(e.shaderBufferFloat32AtomicAdd == device.features().shaderBufferFloat32AtomicAdd,
          "requirements must enable the atomic float exactly as Device does");
    check(e.rayQuery == device.features().rayQuery,
          "requirements must enable ray query exactly as Device does");
    check(all.featureChain() != nullptr, "requirements feature chain must be non-null");
    // Presentation IfSupported on this GPU: swapchain, dynamic rendering and
    // the surface instance extension are all requested.
    check(e.swapchain && e.dynamicRendering,
          "presentation IfSupported must enable swapchain and dynamic rendering here");
    check(all.instanceExtensions().size() == 1
              && std::strcmp(all.instanceExtensions()[0], "VK_KHR_surface") == 0,
          "presentation must add exactly VK_KHR_surface to the instance extensions");

    DeviceRequirements none(gpu, {DeviceRequirements::Need::None, DeviceRequirements::Need::None,
                                  DeviceRequirements::Need::None});
    const EnabledDeviceFeatures n = none.enabled();
    check(!n.shaderBufferFloat32AtomicAdd && !n.rayQuery && !n.swapchain && !n.dynamicRendering,
          "Need::None must enable nothing");
    check(none.deviceExtensions().empty(), "Need::None must request no device extensions");

    // Required on a GPU that is absent: the check runs before anything else.
    bool threw = false;
    try
    {
        DeviceRequirements missing(VK_NULL_HANDLE, {DeviceRequirements::Need::Required});
    }
    catch (const std::runtime_error & err)
    {
        threw = std::string(err.what()).find("no physical device") != std::string::npos;
    }
    check(threw, "DeviceRequirements(VK_NULL_HANDLE) must throw naming the missing device");

    // Need::Required on a feature the GPU lacks would throw, but this GPU
    // supports every option, so that path cannot be exercised here.
    std::cout << "requirements: IfSupported enables atomic float "
              << (e.shaderBufferFloat32AtomicAdd ? "yes" : "no") << " and ray query "
              << (e.rayQuery ? "yes" : "no") << "; None enables nothing; null GPU rejected\n";
    std::cout << "[SKIP] Need::Required throwing on a missing feature: this GPU supports every "
                 "optional feature\n";
}

} // namespace

int main()
{
    Device device;
    try
    {
        // Headless: no windowing instance extensions needed.
        device.initialize(/*instanceExtensions=*/{}, /*enableValidation=*/true);
    }
    catch (const std::exception & e)
    {
        std::cerr << "Device init FAILED: " << e.what() << std::endl;
        return 1;
    }

    const auto & f = device.features();
    std::cout << "Device OK\n"
              << "  GPU:                    " << device.properties().deviceName << "\n"
              << "  graphics queue family:  " << device.graphicsQueueFamily() << "\n"
              << "  compute  queue family:  " << device.computeQueueFamily() << "\n"
              << "  VMA allocator:          " << (device.allocator() ? "created" : "null") << "\n"
              << "  --- DeviceFeatures ---\n"
              << "  subgroup size:          " << f.subgroupSize << "\n"
              << "  subgroup basic:         " << (f.subgroupBasic ? "yes" : "no") << "\n"
              << "  subgroup arithmetic:    " << (f.subgroupArithmetic ? "yes" : "no") << "\n"
              << "  subgroup ballot:        " << (f.subgroupBallot ? "yes" : "no") << "\n"
              << "  subgroup shuffle:       " << (f.subgroupShuffle ? "yes" : "no") << "\n"
              << "  shaderFloat16:          " << (f.shaderFloat16 ? "yes" : "no") << "\n"
              << "  shaderInt8:             " << (f.shaderInt8 ? "yes" : "no") << "\n"
              << "  16-bit storage:         " << (f.storageBuffer16BitAccess ? "yes" : "no") << "\n"
              << "  bufferDeviceAddress:    " << (f.bufferDeviceAddress ? "yes" : "no") << "\n"
              << "  cooperative matrix:     " << (f.hasCooperativeMatrix ? "yes" : "no") << "\n"
              << "  cooperative vector:     " << (f.hasCooperativeVector ? "yes" : "no") << "\n"
              << "  dedicated compute queue:" << (f.hasDedicatedComputeQueue ? "yes" : "no")
              << std::endl;

    try
    {
        testDeviceState(device);

        // Verification: the original round trip of 256 ints.
        {
            constexpr size_t kCount = 256;
            std::vector<int> src(kCount);
            for (size_t i = 0; i < kCount; ++i)
            {
                src[i] = static_cast<int>(i) * 3 - 17;
            }
            Buffer buffer(device, kCount * sizeof(int));
            buffer.upload(src.data(), kCount * sizeof(int));
            std::vector<int> dst(kCount, 0);
            buffer.download(dst.data(), kCount * sizeof(int));
            check(dst == src, "int round trip: data mismatch");
            std::cout << "Buffer round-trip OK (" << kCount << " ints)\n";
        }

        testTransfers(device);
        testAdoptRejections(device);
        testBorrowedRoundTrip(device);
        testRequirements(device);
    }
    catch (const std::exception & e)
    {
        std::cerr << "Device test FAILED: " << e.what() << std::endl;
        device.shutdown();
        return 1;
    }
    device.shutdown();

    // These build their own Devices, after the shared one is gone.
    try
    {
        testDoubleInitialize();
    }
    catch (const std::exception & e)
    {
        std::cerr << "Double-initialize test FAILED: " << e.what() << std::endl;
        return 1;
    }

    if (g_failures != 0)
    {
        std::cerr << "device_test FAILED (" << g_failures << " checks)" << std::endl;
        return 1;
    }
    std::cout << "device_test PASSED" << std::endl;
    return 0;
}
