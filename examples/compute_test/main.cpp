// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Smoke test for the compute path: Slang kernels dispatched on the dedicated
// compute queue over Buffers, with every result checked against an exact
// integer reference. Exercises ShaderCompiler, DescriptorSet, Pipeline,
// CommandBuffer and Buffer end to end -- the whole compute path before any
// Gaussian math.
//
// Covered: direct dispatch, push constants, indirect dispatch (CPU- and
// GPU-written argument buffers), copyBuffer, bufferBarrier in a dependent
// chain, Buffer::view(offset) for descriptors and partial downloads, a
// large run, and ShaderCompiler's rejections.

#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "sabercat/buffer.h"
#include "sabercat/command_buffer.h"
#include "sabercat/descriptor_set.h"
#include "sabercat/device.h"
#include "sabercat/pipeline.h"
#include "sabercat/shader_compiler.h"
#include "sabercat/vk_check.h"

using namespace sabercat;

namespace
{

constexpr uint32_t kCount = 256;
constexpr uint32_t kWorkgroupSize = 64; // must match [numthreads] in the kernels

int g_failures = 0;

void check(bool condition, const std::string & message)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << message << std::endl;
        ++g_failures;
    }
}

// Submits `cmd` on `queue`, waits for completion, and cleans up the fence.
void submitAndWait(VkDevice device, VkQueue queue, VkCommandBuffer cmd)
{
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence;
    VK_CHECK(vkCreateFence(device, &fenceInfo, nullptr, &fence), "Failed to create fence");

    VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, fence), "Failed to submit compute queue");
    VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX), "Failed to wait for fence");

    vkDestroyFence(device, fence, nullptr);
}

// One compiled kernel with a single storage binding (binding 0) and an
// optional push-constant block.
struct Kernel
{
    VkShaderModule module = VK_NULL_HANDLE;
    std::unique_ptr<DescriptorSet> descriptors;
    VkDescriptorSet set = VK_NULL_HANDLE;
    Pipeline pipeline;
    VkPipelineLayout layout = VK_NULL_HANDLE;
};

class ComputeTest
{
  public:
    explicit ComputeTest(Device & device)
        : m_device(device), m_compiler(device.handle())
    {
        VkCommandPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        poolInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
        poolInfo.queueFamilyIndex = device.computeQueueFamily();
        VK_CHECK(vkCreateCommandPool(device.handle(), &poolInfo, nullptr, &m_pool),
                 "Failed to create compute command pool");
    }

    ~ComputeTest()
    {
        for (Kernel * k : {&m_double, &m_addOne, &m_affine, &m_countGroups})
        {
            if (k->module != VK_NULL_HANDLE)
            {
                k->pipeline.destroy(m_device.handle());
                m_compiler.destroyShaderModule(k->module);
            }
        }
        vkDestroyCommandPool(m_device.handle(), m_pool, nullptr);
    }

    ComputeTest(const ComputeTest &) = delete;
    ComputeTest & operator=(const ComputeTest &) = delete;

    void loadKernels()
    {
        m_double = makeKernel("double", 0);
        m_addOne = makeKernel("add_one", 0);
        m_affine = makeKernel("affine", sizeof(int32_t) * 2);
        m_countGroups = makeKernel("count_groups", sizeof(uint32_t));
    }

    // Records and submits one dispatch of `kernel` over `set`.
    void dispatch(Kernel & kernel, VkBuffer buffer, VkDeviceSize offset, VkDeviceSize range,
                  const void * pushData, uint32_t pushSize, uint32_t groups)
    {
        kernel.descriptors->bindBuffer(kernel.set, 0, buffer, offset, range,
                                       VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        record([&](CommandBuffer & cmd)
               {
                   if (pushSize > 0)
                   {
                       cmd.pushConstants(kernel.layout, VK_SHADER_STAGE_COMPUTE_BIT, pushSize,
                                         pushData);
                   }
                   cmd.dispatchCompute(kernel.pipeline.getHandle(), kernel.layout, kernel.set,
                                       groups, 1, 1);
               });
    }

    // Records and submits an indirect dispatch: the group counts come from
    // `args` at `argsOffset`.
    void dispatchIndirect(Kernel & kernel, VkBuffer buffer, VkDeviceSize offset,
                          VkDeviceSize range, VkBuffer args, VkDeviceSize argsOffset)
    {
        kernel.descriptors->bindBuffer(kernel.set, 0, buffer, offset, range,
                                       VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        record([&](CommandBuffer & cmd)
               {
                   cmd.dispatchComputeIndirect(kernel.pipeline.getHandle(), kernel.layout,
                                               kernel.set, args, argsOffset);
               });
    }

    // Records the work done by `fn` into one command buffer and waits on it.
    template <typename F> void record(F && fn)
    {
        CommandBuffer cmd(m_device.handle(), m_pool);
        cmd.begin(VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT);
        fn(cmd);
        cmd.end();
        submitAndWait(m_device.handle(), m_device.computeQueue(), cmd.getHandle());
    }

    Kernel & double_() { return m_double; }
    Kernel & addOne() { return m_addOne; }
    Kernel & affine() { return m_affine; }
    Kernel & countGroups() { return m_countGroups; }
    ShaderCompiler & compiler() { return m_compiler; }
    Device & device() { return m_device; }

  private:
    Kernel makeKernel(const std::string & name, uint32_t pushSize)
    {
        Kernel k;
        auto spirv = m_compiler.loadSPIRV(std::string(SHADER_DIR) + "/" + name + ".comp.spv");
        k.module = m_compiler.createShaderModule(spirv);

        k.descriptors = std::make_unique<DescriptorSet>(m_device.handle());
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = 0;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        VkDescriptorSetLayout setLayout = k.descriptors->createLayout({binding});
        std::vector<VkDescriptorPoolSize> poolSizes = {
            {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1},
        };
        k.descriptors->createPool(/*maxSets=*/1, poolSizes);
        k.set = k.descriptors->allocate(setLayout);

        k.layout = k.pipeline.createPipelineLayout(m_device.handle(), setLayout, pushSize,
                                                   VK_SHADER_STAGE_COMPUTE_BIT);
        k.pipeline.createComputePipeline(m_device.handle(), k.module, k.layout);
        return k;
    }

    Device & m_device;
    ShaderCompiler m_compiler;
    VkCommandPool m_pool = VK_NULL_HANDLE;
    Kernel m_double, m_addOne, m_affine, m_countGroups;
};

std::vector<int32_t> iota(size_t n, int32_t start = 0)
{
    std::vector<int32_t> v(n);
    for (size_t i = 0; i < n; ++i)
    {
        v[i] = start + static_cast<int32_t>(i);
    }
    return v;
}

// Reference for add_one dispatched over the first `groups * 64` elements.
std::vector<int32_t> addOneReference(std::vector<int32_t> v, uint32_t groups)
{
    for (size_t i = 0; i < v.size() && i < size_t(groups) * kWorkgroupSize; ++i)
    {
        v[i] += 1;
    }
    return v;
}

// Test: the original doubling kernel, direct dispatch.
void testDouble(ComputeTest & t)
{
    Device & device = t.device();
    Buffer buffer(device, kCount * sizeof(int32_t));
    std::vector<int32_t> src = iota(kCount);
    buffer.upload(src.data(), kCount * sizeof(int32_t));
    t.dispatch(t.double_(), buffer.handle(), 0, buffer.size(), nullptr, 0,
               kCount / kWorkgroupSize);
    std::vector<int32_t> dst(kCount, 0);
    buffer.download(dst.data(), kCount * sizeof(int32_t));
    bool ok = true;
    for (uint32_t i = 0; i < kCount; ++i)
    {
        ok = ok && dst[i] == src[i] * 2;
    }
    check(ok, "double: every element must be doubled exactly");
    std::cout << "double: " << kCount << " ints doubled, exact\n";
}

// Test: push constants reach the kernel, with negative values and a
// multiplier of 1 (an identity edge) included.
void testPushConstants(ComputeTest & t)
{
    Device & device = t.device();
    Buffer buffer(device, kCount * sizeof(int32_t));
    std::vector<int32_t> src = iota(kCount, -128);
    buffer.upload(src.data(), kCount * sizeof(int32_t));

    const int32_t first[2] = {-2, 7};
    t.dispatch(t.affine(), buffer.handle(), 0, buffer.size(), first, sizeof(first), kCount / 64);
    std::vector<int32_t> dst(kCount, 0);
    buffer.download(dst.data(), kCount * sizeof(int32_t));
    bool ok = true;
    for (uint32_t i = 0; i < kCount; ++i)
    {
        ok = ok && dst[i] == src[i] * -2 + 7;
    }
    check(ok, "push constants: data * -2 + 7 must be exact");

    // Identity parameters: the same buffer must come back unchanged.
    const int32_t identity[2] = {1, 0};
    t.dispatch(t.affine(), buffer.handle(), 0, buffer.size(), identity, sizeof(identity),
               kCount / 64);
    std::vector<int32_t> again(kCount, 0);
    buffer.download(again.data(), kCount * sizeof(int32_t));
    check(again == dst, "push constants: multiplier 1, addend 0 must be the identity");
    std::cout << "push constants: data * -2 + 7 exact; identity (1, 0) leaves it unchanged\n";
}

// Test: indirect dispatch from a CPU-written args buffer, including the
// zero-group edge (nothing runs) and a full-coverage one.
void testIndirectFromHost(ComputeTest & t)
{
    Device & device = t.device();
    Buffer buffer(device, kCount * sizeof(int32_t));
    Buffer args(device, 3 * sizeof(uint32_t),
                VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                    | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    std::vector<int32_t> src = iota(kCount);
    const uint32_t groupCases[] = {0, 1, 2, 4};
    for (uint32_t groups : groupCases)
    {
        buffer.upload(src.data(), kCount * sizeof(int32_t));
        const uint32_t command[3] = {groups, 1, 1};
        args.upload(command, sizeof(command));
        t.dispatchIndirect(t.addOne(), buffer.handle(), 0, buffer.size(), args.handle(), 0);
        std::vector<int32_t> dst(kCount, 0);
        buffer.download(dst.data(), kCount * sizeof(int32_t));
        check(dst == addOneReference(src, groups),
              "indirect (host args): " + std::to_string(groups) + " groups must touch exactly "
                  + std::to_string(groups * kWorkgroupSize) + " elements");
    }
    std::cout << "indirect (host args): 0, 1, 2 and 4 groups touched exactly 0, 64, 128 and 256 "
                 "elements\n";
}

// Test: a three-kernel dependent chain recorded into ONE command buffer, so
// the bufferBarrier calls are what orders the stages. count_groups writes
// the args GPU-side, add_one reads them indirectly, affine finishes the data.
void testDependentChain(ComputeTest & t)
{
    Device & device = t.device();
    const uint32_t counts[] = {1, 100, 192}; // groups 1, 2, 3
    for (uint32_t count : counts)
    {
        Buffer data(device, kCount * sizeof(int32_t));
        Buffer args(device, 3 * sizeof(uint32_t),
                    VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
                        | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        std::vector<int32_t> src = iota(kCount);
        data.upload(src.data(), kCount * sizeof(int32_t));

        Kernel & countKernel = t.countGroups();
        Kernel & addKernel = t.addOne();
        Kernel & affineKernel = t.affine();
        countKernel.descriptors->bindBuffer(countKernel.set, 0, args.handle(), 0, args.size(),
                                            VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        addKernel.descriptors->bindBuffer(addKernel.set, 0, data.handle(), 0, data.size(),
                                          VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        affineKernel.descriptors->bindBuffer(affineKernel.set, 0, data.handle(), 0, data.size(),
                                             VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        const int32_t params[2] = {3, -5};

        t.record(
            [&](CommandBuffer & cmd)
            {
                cmd.pushConstants(countKernel.layout, VK_SHADER_STAGE_COMPUTE_BIT, sizeof(count),
                                  &count);
                cmd.dispatchCompute(countKernel.pipeline.getHandle(), countKernel.layout,
                                    countKernel.set, 1, 1, 1);
                cmd.bufferBarrier(args.handle(), VK_ACCESS_SHADER_WRITE_BIT,
                                  VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT);

                cmd.dispatchComputeIndirect(addKernel.pipeline.getHandle(), addKernel.layout,
                                            addKernel.set, args.handle(), 0);
                cmd.bufferBarrier(data.handle(), VK_ACCESS_SHADER_WRITE_BIT,
                                  VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

                cmd.pushConstants(affineKernel.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                  sizeof(params), params);
                cmd.dispatchCompute(affineKernel.pipeline.getHandle(), affineKernel.layout,
                                    affineKernel.set, kCount / kWorkgroupSize, 1, 1);
            });

        std::vector<int32_t> dst(kCount, 0);
        data.download(dst.data(), kCount * sizeof(int32_t));
        std::vector<int32_t> want = addOneReference(src, (count + 63) / 64);
        for (int32_t & v : want)
        {
            v = v * 3 - 5;
        }
        check(dst == want, "dependent chain: count " + std::to_string(count)
                               + " does not match the CPU reference");
    }
    std::cout << "dependent chain (count_groups -> indirect add_one -> affine): counts 1, 100 and "
                 "192 exact\n";
}

// Test: copyBuffer, Buffer::view(offset) as a descriptor range, and
// partial downloads through the view.
void testCopyAndViews(ComputeTest & t)
{
    Device & device = t.device();
    std::vector<int32_t> src = iota(kCount, 1000);

    Buffer source(device, kCount * sizeof(int32_t));
    Buffer copy(device, kCount * sizeof(int32_t));
    source.upload(src.data(), kCount * sizeof(int32_t));
    t.record([&](CommandBuffer & cmd)
             { cmd.copyBuffer(source.handle(), copy.handle(), kCount * sizeof(int32_t)); });
    std::vector<int32_t> copied(kCount, 0);
    copy.download(copied.data(), kCount * sizeof(int32_t));
    check(copied == src, "copyBuffer must reproduce the source bytes exactly");

    // view(offset): a partial download reads from the view's offset.
    const BufferView tail = source.view(sizeof(int32_t), 64 * sizeof(int32_t));
    check(tail.size == (kCount - 64) * sizeof(int32_t) && tail.elementCount() == kCount - 64,
          "view(offset) must cover the rest of the buffer");
    std::vector<int32_t> part(32, 0);
    downloadView(device, tail, part.data(), part.size() * sizeof(int32_t));
    check(std::vector<int32_t>(src.begin() + 64, src.begin() + 96) == part,
          "partial download through view(offset) must read elements 64..95");
    std::vector<int32_t> whole(kCount - 64, 0);
    downloadView(device, tail, whole.data(), whole.size() * sizeof(int32_t));
    check(std::vector<int32_t>(src.begin() + 64, src.end()) == whole,
          "full download through view(offset) must read elements 64..255");

    // A descriptor bound to the view: add_one over the 192 elements from
    // element 64 on. Elements 0..63 must be untouched.
    const BufferView binding = source.view(sizeof(int32_t), 64 * sizeof(int32_t));
    t.dispatch(t.addOne(), binding.buffer, binding.offset, binding.size, nullptr, 0, 3);
    std::vector<int32_t> after(kCount, 0);
    source.download(after.data(), kCount * sizeof(int32_t));
    bool ok = true;
    for (uint32_t i = 0; i < kCount; ++i)
    {
        const int32_t expected = src[i] + (i >= 64 ? 1 : 0);
        ok = ok && after[i] == expected;
    }
    check(ok, "descriptor on view(offset): elements 64..255 +1, 0..63 unchanged");
    std::cout << "copyBuffer exact; view(offset) partial and full downloads exact; descriptor "
                 "bound at offset 256 B touches exactly elements 64..255\n";
}

// Scale: 4 Mi ints through two chained dispatches, every element checked.
void testLarge(ComputeTest & t)
{
    Device & device = t.device();
    constexpr uint32_t n = 4u * 1024u * 1024u;
    std::vector<int32_t> src(n);
    for (uint32_t i = 0; i < n; ++i)
    {
        src[i] = static_cast<int32_t>((i * 2654435761u) >> 12) - 500000;
    }
    Buffer buffer(device, n * sizeof(int32_t));
    buffer.upload(src.data(), n * sizeof(int32_t));
    t.dispatch(t.addOne(), buffer.handle(), 0, buffer.size(), nullptr, 0, n / kWorkgroupSize);
    const int32_t params[2] = {3, -5};
    t.dispatch(t.affine(), buffer.handle(), 0, buffer.size(), params, sizeof(params),
               n / kWorkgroupSize);
    std::vector<int32_t> dst(n, 0);
    buffer.download(dst.data(), n * sizeof(int32_t));
    size_t mismatches = 0;
    for (uint32_t i = 0; i < n; ++i)
    {
        mismatches += dst[i] == (src[i] + 1) * 3 - 5 ? 0 : 1;
    }
    check(mismatches == 0, "large run: " + std::to_string(mismatches) + " mismatches in 4 Mi ints");
    std::cout << "large run: 4 Mi ints, add_one then affine, mismatches " << mismatches << "\n";
}

// Negative: ShaderCompiler rejects a missing file and a size that is not a
// whole number of words, with a message naming the problem.
void testShaderCompilerRejections(ComputeTest & t)
{
    try
    {
        t.compiler().loadSPIRV(std::string(SHADER_DIR) + "/does_not_exist.comp.spv");
        check(false, "a missing SPIR-V file must throw");
    }
    catch (const std::runtime_error & e)
    {
        check(std::string(e.what()).find("Failed to open") != std::string::npos,
              std::string("missing file message: ") + e.what());
    }

    const std::string oddPath = "compute_test_odd_size.spv";
    {
        std::ofstream out(oddPath, std::ios::binary);
        const char bytes[6] = {1, 2, 3, 4, 5, 6};
        out.write(bytes, sizeof(bytes));
    }
    try
    {
        t.compiler().loadSPIRV(oddPath);
        check(false, "a SPIR-V file of 6 bytes must throw");
    }
    catch (const std::runtime_error & e)
    {
        check(std::string(e.what()).find("multiple of 4") != std::string::npos,
              std::string("odd size message: ") + e.what());
    }
    std::remove(oddPath.c_str());
    std::cout << "ShaderCompiler: missing file and 6-byte file both rejected with a message\n";
}

} // namespace

int main()
{
    Device device;
    try
    {
        device.initialize(/*instanceExtensions=*/{}, /*enableValidation=*/true);
    }
    catch (const std::exception & e)
    {
        std::cerr << "Device init FAILED: " << e.what() << std::endl;
        return 1;
    }

    try
    {
        ComputeTest t(device);
        t.loadKernels();
        testDouble(t);
        testPushConstants(t);
        testIndirectFromHost(t);
        testDependentChain(t);
        testCopyAndViews(t);
        testLarge(t);
        testShaderCompilerRejections(t);
    }
    catch (const std::exception & e)
    {
        std::cerr << "Compute test FAILED: " << e.what() << std::endl;
        device.shutdown();
        return 1;
    }

    device.shutdown();
    if (g_failures != 0)
    {
        std::cerr << "compute_test FAILED (" << g_failures << " checks)" << std::endl;
        return 1;
    }
    std::cout << "compute_test PASSED" << std::endl;
    return 0;
}
