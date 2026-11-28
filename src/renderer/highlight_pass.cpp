// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "highlight_pass.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "sabercat/buffer.h"
#include "sabercat/descriptor_set.h"
#include "sabercat/device.h"
#include "sabercat/pipeline.h"
#include "sabercat/render_config.h"
#include "sabercat/shader_compiler.h"
#include "sabercat/vk_check.h"

namespace sabercat
{

namespace
{

// SelectionMergePush in shaders/selection_merge.slang: four 4-byte uints, no padding.
struct SelectionMergePush
{
    uint32_t first;        // the layer's offset in the merged scene
    uint32_t count;        // the layer's Gaussian count
    uint32_t firstWord;    // first >> 5: the first merged word the layer touches
    uint32_t wordsInRange; // merged words the layer touches
};
static_assert(sizeof(SelectionMergePush) == 4 * sizeof(uint32_t),
              "SelectionMergePush must match selection_merge.slang's push constants");

// Binding slots, as selection_merge.slang declares them.
constexpr uint32_t kBindingLayerMask = 0;
constexpr uint32_t kBindingMergedMask = 1;
constexpr uint32_t kBindingCount = 2;

// Descriptor sets per pool: one DescriptorSet object (one pool) per 16 layers, as
// EditPass's merge kernel does.
constexpr uint32_t kSetsPerPool = 16;

constexpr uint32_t kBitsPerWord = 32;

VkDescriptorSetLayoutBinding storageBufferBinding(uint32_t binding)
{
    VkDescriptorSetLayoutBinding b{};
    b.binding = binding;
    b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    return b;
}

// Words needed for `bits` bits, ceil(bits / 32), without overflowing at the top of uint32.
uint32_t wordsFor(uint32_t bits)
{
    return static_cast<uint32_t>((static_cast<uint64_t>(bits) + kBitsPerWord - 1) / kBitsPerWord);
}

// A global memory barrier between two stages of the same command buffer.
void memoryBarrier(VkCommandBuffer cmd, VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
                   VkPipelineStageFlags dstStage, VkAccessFlags dstAccess)
{
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = srcAccess;
    barrier.dstAccessMask = dstAccess;
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 1, &barrier, 0, nullptr, 0, nullptr);
}

} // namespace

// The merge kernel's pipeline and descriptor sets. The pipeline and the first pool
// are made together, so the layout exists when the pipeline is built. Each further
// pool holds kSetsPerPool sets; each dispatch uses its own set, so a set is never
// rewritten while a submission that uses it is pending.
struct HighlightPass::MergeKernel
{
    MergeKernel(Device & device, const std::string & shaderDir);
    ~MergeKernel();

    MergeKernel(const MergeKernel &) = delete;
    MergeKernel & operator=(const MergeKernel &) = delete;

    // The descriptor set for dispatch `index`, allocated on first use.
    VkDescriptorSet set(size_t index);
    // The DescriptorSet object whose pool holds set(index), for bindBuffer.
    DescriptorSet & owner(size_t index);

    // Makes one more pool. The first pool also creates the set layout.
    void addChunk();

    Device & device;
    Pipeline pipeline;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE; // owned by chunks.front()
    std::vector<std::unique_ptr<DescriptorSet>> chunks;
    std::vector<VkDescriptorSet> sets;
};

HighlightPass::MergeKernel::MergeKernel(Device & dev, const std::string & shaderDir)
    : device(dev)
{
    VkDevice handle = device.handle();
    // A throw below (a missing shader file is the common one) skips the destructor,
    // so the pipeline created so far is released here. The pools clean up through
    // their own destructors.
    try
    {
        addChunk();

        ShaderCompiler compiler(handle);
        VkShaderModule module = compiler.createShaderModule(
            compiler.loadSPIRV(shaderDir + "/selection_merge.comp.spv"));
        try
        {
            VkPipelineLayout layout = pipeline.createPipelineLayout(
                handle, setLayout, sizeof(SelectionMergePush), VK_SHADER_STAGE_COMPUTE_BIT);
            pipeline.createComputePipeline(handle, module, layout);
        }
        catch (...)
        {
            compiler.destroyShaderModule(module);
            throw;
        }
        compiler.destroyShaderModule(module);
    }
    catch (...)
    {
        pipeline.destroy(handle);
        throw;
    }
}

HighlightPass::MergeKernel::~MergeKernel()
{
    pipeline.destroy(device.handle());
    // The first pool owns the layout the others allocate from, so it goes last.
    while (!chunks.empty())
    {
        chunks.pop_back();
    }
}

void HighlightPass::MergeKernel::addChunk()
{
    auto chunk = std::make_unique<DescriptorSet>(device.handle());
    if (setLayout == VK_NULL_HANDLE)
    {
        setLayout = chunk->createLayout({storageBufferBinding(kBindingLayerMask),
                                         storageBufferBinding(kBindingMergedMask)});
    }
    chunk->createPool(kSetsPerPool,
                      {{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, kSetsPerPool * kBindingCount}});
    chunks.push_back(std::move(chunk));
}

VkDescriptorSet HighlightPass::MergeKernel::set(size_t index)
{
    while (sets.size() <= index)
    {
        // Every existing pool is full: start the next one.
        if (sets.size() == chunks.size() * kSetsPerPool)
        {
            addChunk();
        }
        sets.push_back(chunks.back()->allocate(setLayout));
    }
    return sets[index];
}

DescriptorSet & HighlightPass::MergeKernel::owner(size_t index)
{
    return *chunks[index / kSetsPerPool];
}

HighlightPass::HighlightPass(Device & device, const std::string & shaderDir)
    : m_device(device), m_shaderDir(shaderDir)
{
}

HighlightPass::~HighlightPass() = default;

void HighlightPass::buildMergedMask(const std::vector<BufferView> & layerMasks,
                                    const std::vector<uint32_t> & layerCounts,
                                    uint32_t totalCount)
{
    // Everything is checked before the first GPU call: a throw inside the record
    // callback would leave the command buffer half-recorded.
    VK_VALIDATE(layerMasks.size() == layerCounts.size(),
                "HighlightPass::buildMergedMask: " + std::to_string(layerMasks.size())
                    + " layer masks for " + std::to_string(layerCounts.size()) + " layer counts");
    uint64_t sum = 0;
    for (const uint32_t count : layerCounts)
    {
        sum += count;
    }
    VK_VALIDATE(sum == totalCount,
                "HighlightPass::buildMergedMask: the layer counts sum to " + std::to_string(sum)
                    + " but totalCount is " + std::to_string(totalCount));
    for (size_t i = 0; i < layerCounts.size(); ++i)
    {
        const uint32_t count = layerCounts[i];
        if (count == 0)
        {
            continue;
        }
        const BufferView & mask = layerMasks[i];
        const std::string label = "HighlightPass::buildMergedMask: layer " + std::to_string(i);
        VK_VALIDATE(mask.buffer != VK_NULL_HANDLE, label + " has no mask buffer");
        const VkDeviceSize neededBytes =
            static_cast<VkDeviceSize>(wordsFor(count)) * sizeof(uint32_t);
        VK_VALIDATE(mask.size >= neededBytes,
                    label + "'s mask holds fewer than ceil(count / 32) words");
    }

    // The merged buffer: grow-only, geometric, as ForwardPass grows its own buffers.
    const uint32_t mergedWords = std::max<uint32_t>(1, wordsFor(totalCount));
    const VkDeviceSize mergedBytes = static_cast<VkDeviceSize>(mergedWords) * sizeof(uint32_t);
    if (!m_merged || m_merged->size() < mergedBytes)
    {
        const VkDeviceSize grownBytes = m_merged ? m_merged->size() * 2 : 0;
        m_merged = std::make_unique<Buffer>(m_device, std::max(mergedBytes, grownBytes));
        ++m_generation;
    }

    if (!m_kernel)
    {
        m_kernel = std::make_unique<MergeKernel>(m_device, m_shaderDir);
    }

    // One dispatch per layer that has Gaussians. Each dispatch's descriptor set is
    // written here, before the submission, and used only by that submission.
    const VkBuffer merged = m_merged->handle();
    const VkDeviceSize mergedSize = m_merged->size();
    struct Dispatch
    {
        VkDescriptorSet set;
        SelectionMergePush push;
    };
    std::vector<Dispatch> dispatches;
    dispatches.reserve(layerCounts.size());
    uint32_t first = 0;
    for (size_t i = 0; i < layerCounts.size(); ++i)
    {
        const uint32_t count = layerCounts[i];
        if (count == 0)
        {
            continue;
        }
        const uint64_t layerEnd = static_cast<uint64_t>(first) + count;
        const uint32_t firstWord = first >> 5;
        const uint32_t wordsInRange =
            static_cast<uint32_t>(((layerEnd + kBitsPerWord - 1) >> 5) - firstWord);

        const size_t index = dispatches.size();
        const VkDescriptorSet set = m_kernel->set(index);
        DescriptorSet & owner = m_kernel->owner(index);
        const BufferView & mask = layerMasks[i];
        owner.bindBuffer(set, kBindingLayerMask, mask.buffer, mask.offset, mask.size,
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        owner.bindBuffer(set, kBindingMergedMask, merged, 0, mergedSize,
                         VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

        dispatches.push_back({set, SelectionMergePush{first, count, firstWord, wordsInRange}});
        first += count;
    }

    const VkPipeline pipeline = m_kernel->pipeline.getHandle();
    const VkPipelineLayout pipelineLayout = m_kernel->pipeline.getLayout();
    m_device.immediateSubmit(
        [&](VkCommandBuffer cmd)
        {
            // Zero the words the merge covers, then the layers are ORed in, in order.
            vkCmdFillBuffer(cmd, merged, 0, mergedBytes, 0);
            memoryBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                          VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            for (const Dispatch & dispatch : dispatches)
            {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1,
                                        &dispatch.set, 0, nullptr);
                vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                   sizeof(dispatch.push), &dispatch.push);
                vkCmdDispatch(cmd, dispatchGroups(dispatch.push.wordsInRange, kWorkgroupSize), 1,
                              1);
                // The next layer reads and rewrites the word this one may share at
                // its boundary, so this dispatch's writes land first.
                memoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                              VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            }
        });
}

const Buffer & HighlightPass::mergedMask() const
{
    VK_VALIDATE(m_merged != nullptr,
                "HighlightPass::mergedMask: no merged mask yet -- call buildMergedMask() first");
    return *m_merged;
}

} // namespace sabercat
