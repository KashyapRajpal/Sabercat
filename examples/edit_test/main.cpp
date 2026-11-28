// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// GPU checks for the Studio edit-path building blocks:
//
//  A. GaussianCloud::downloadFromGPU refreshes every host stream, the
//     relightable ones included, after the host copies have been wiped.
//  B. GaussianCloud::allocateGPU: count, capacity, storage generation, the
//     relight presence flags and empty host vectors; a GPU-to-GPU copy of every
//     stream from the cloud in A, read back with downloadFromGPU; a partial
//     presence; a count of 0 is refused.
//  C. GaussianCloud::setBounds round trip, on the cloud from B.
//  D. GpuScan (the internal exclusive prefix sum) against a CPU scan over the
//     block-size edge cases, with one kernel instance reused, and a strided input.
//  E. EditPass::exclusiveScan against the CPU scan, total included; a count of 0;
//     an input that aliases the output is refused.
//  F. EditPass::mergeLayers against the host mergeLayers(), stream by stream, and
//     planLayerMerge's layout; the mergeLayers refusals (merge_section.cpp).
//  G. Renderer::mergeLayersOnGpu: bounds, readback, storage reuse, forward parity,
//     clearScene (renderer_section.cpp).
//  H. Ray-traced visibility refit and its refit cap (renderer_section.cpp).
//  I. Informational merge timing when SABERCAT_EDIT_TEST_SCENE names a scene file
//     (renderer_section.cpp).
//  J. Selection: EditPass::select against a CPU predicate, the op combinations, the
//     mask operations and SelectionMask (selection_section.cpp).
//  K. The selection highlight on a Renderer (selection_section.cpp).
//  L. The C4 stream edits on fresh layer copies: transform, sparse snapshot, delete with
//     undo and redo, duplicate with growth and undo (ops_section.cpp).
//  M. A seeded random sequence of 200 select, transform, duplicate and delete ops, each undone
//     and redone with the state hash checked, and a Renderer merge of the restored layer
//     (ops_section.cpp).
//  N. The mask operations and selection on a layer above the 65535-workgroup limit
//     (17,000,003 Gaussians; large_section.cpp).
//  O. The borrowed-buffer guard (ownership model B): gatherSelected, scatterSnapshot,
//     deleteSelected, undoDelete, duplicateSelected and undoDuplicate refuse a borrowed layer
//     with a runtime_error and change nothing; select and an identity transform accept it
//     (borrow_section.cpp).
//
// Validation messages are logged as "[validation]" and fail the run through
// CTest, so the Device is created with validation on.

#include <glm/glm.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <type_traits>
#include <vector>

#include <vulkan/vulkan.h>

#include "gpu_scan.h"
#include "edit_test_common.h"
#include "sabercat/buffer.h"
#include "sabercat/buffer_view.h"
#include "sabercat/device.h"
#include "sabercat/edit_pass.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/render_config.h"
#include "sabercat/scene_bounds.h"

#include "../common/shader_dir.h"

using namespace sabercat;
using namespace edit_test;

namespace
{

// Not a multiple of 32, so the last workgroup of every pass is partial.
constexpr size_t kCloudCount = 1237;
constexpr size_t kAllocatedCapacity = 1500;
constexpr size_t kPartialCount = 33;
constexpr uint32_t kShDegree = 3;
constexpr uint32_t kUintStride = sizeof(uint32_t);

XorShift32 g_rng(0x2545F491u);

// Uniform in [0, 1).
float nextUnit()
{
    return g_rng.unit();
}

// Uniform in [-1, 1).
float nextSigned()
{
    return g_rng.signedUnit();
}

// Uniform in [0, bound).
uint32_t nextBelow(uint32_t bound)
{
    return g_rng.below(bound);
}

// One check per stream.
void checkStreams(const Streams & want, const Streams & got, const std::string & label)
{
    check(bitEqual(want.positions, got.positions), label + ": positions bit-equal");
    check(bitEqual(want.scales, got.scales), label + ": scales bit-equal");
    check(bitEqual(want.rotations, got.rotations), label + ": rotations bit-equal");
    check(bitEqual(want.shCoefficients, got.shCoefficients),
          label + ": shCoefficients bit-equal");
    check(bitEqual(want.opacities, got.opacities), label + ": opacities bit-equal");
    check(bitEqual(want.normals, got.normals), label + ": normals bit-equal");
    check(bitEqual(want.baseColorRoughness, got.baseColorRoughness),
          label + ": baseColorRoughness bit-equal");
    check(bitEqual(want.metallic, got.metallic), label + ": metallic bit-equal");
    check(bitEqual(want.visibility, got.visibility), label + ": visibility bit-equal");
}

// A cloud of `count` Gaussians with every host stream, the relightable ones
// included, filled with random values.
void fillRandomCloud(GaussianCloud & cloud, size_t count)
{
    cloud.initialize(count);
    cloud.setShDegree(kShDegree);
    for (glm::vec4 & p : cloud.positions())
    {
        p = glm::vec4(4.0f * nextSigned(), 4.0f * nextSigned(), 4.0f * nextSigned(), 0.0f);
    }
    for (glm::vec4 & s : cloud.scales())
    {
        s = glm::vec4(0.01f + 0.2f * nextUnit(), 0.01f + 0.2f * nextUnit(),
                      0.01f + 0.2f * nextUnit(), 0.0f);
    }
    for (glm::vec4 & q : cloud.rotations())
    {
        q = glm::vec4(nextSigned(), nextSigned(), nextSigned(), nextSigned());
    }
    for (glm::vec4 & c : cloud.shCoefficients())
    {
        c = glm::vec4(nextSigned(), nextSigned(), nextSigned(), 0.0f);
    }
    for (float & o : cloud.opacities())
    {
        o = nextUnit();
    }
    cloud.normals().resize(count);
    for (glm::vec4 & n : cloud.normals())
    {
        n = glm::vec4(nextSigned(), nextSigned(), nextSigned(), 0.0f);
    }
    cloud.baseColorRoughness().resize(count);
    for (glm::vec4 & b : cloud.baseColorRoughness())
    {
        b = glm::vec4(nextUnit(), nextUnit(), nextUnit(), nextUnit());
    }
    cloud.metallic().resize(count);
    for (float & m : cloud.metallic())
    {
        m = nextUnit();
    }
    cloud.visibility().resize(count * kVisibilityCoefficients);
    for (float & v : cloud.visibility())
    {
        v = nextSigned();
    }
}

// The first index where `got` and `want` differ, for the failure message.
// Empty when they match.
std::string firstMismatch(const std::vector<uint32_t> & got, const std::vector<uint32_t> & want)
{
    if (got.size() != want.size())
    {
        return " (size " + std::to_string(got.size()) + ", want " + std::to_string(want.size())
               + ")";
    }
    for (size_t i = 0; i < got.size(); ++i)
    {
        if (got[i] != want[i])
        {
            return " (first mismatch at " + std::to_string(i) + ": got "
                   + std::to_string(got[i]) + ", want " + std::to_string(want[i]) + ")";
        }
    }
    return "";
}

// `count` random values: 0 or 1 when `flags`, else 0..1000.
std::vector<uint32_t> randomValues(size_t count, bool flags)
{
    std::vector<uint32_t> v(count);
    for (uint32_t & x : v)
    {
        x = flags ? nextBelow(2) : nextBelow(1001);
    }
    return v;
}

// CPU reference: out[j] = sum of in[i * stride + offset] over i < j, for j < n.
// Returns the total of the n elements. Wraps on overflow, as uint adds do.
uint32_t cpuExclusiveScan(const std::vector<uint32_t> & in, size_t stride, size_t offset,
                          size_t n, std::vector<uint32_t> & out)
{
    out.assign(n, 0);
    uint32_t running = 0;
    for (size_t j = 0; j < n; ++j)
    {
        out[j] = running;
        running += in[j * stride + offset];
    }
    return running;
}

// One stream copy: source and destination views, and the bytes holding count() elements.
struct StreamCopy
{
    BufferView src;
    BufferView dst;
    VkDeviceSize bytes;
};

// A: downloadFromGPU refreshes every host stream after the host copies are wiped.
void checkDownloadFromGPU(Device & device)
{
    GaussianCloud x(device);
    fillRandomCloud(x, kCloudCount);
    x.uploadToGPU();
    const Streams want = copyStreams(x);

    // Wipe the host copies. The core streams are zeroed as well (not only the
    // relightable ones), so a stream the download did not refresh cannot pass by
    // coincidence.
    x.positions().clear();
    x.normals().clear();
    x.baseColorRoughness().clear();
    x.metallic().clear();
    x.visibility().clear();
    std::fill(x.scales().begin(), x.scales().end(), glm::vec4(0.0f));
    std::fill(x.rotations().begin(), x.rotations().end(), glm::vec4(0.0f));
    std::fill(x.shCoefficients().begin(), x.shCoefficients().end(), glm::vec4(0.0f));
    std::fill(x.opacities().begin(), x.opacities().end(), 0.0f);

    x.downloadFromGPU();
    checkStreams(want, copyStreams(x), "A: downloadFromGPU");
    check(x.positions().size() == kCloudCount, "A: downloaded vectors sized to count()");
}

// B: allocateGPU, a GPU-to-GPU copy of A's cloud into it, a partial presence and the
// count-0 refusal. C: setBounds on the same cloud.
void checkAllocateGPU(Device & device)
{
    GaussianCloud x(device);
    fillRandomCloud(x, kCloudCount);
    x.uploadToGPU();
    const Streams want = copyStreams(x);

    GaussianCloud y(device);
    const uint64_t generationBefore = y.storageGeneration();
    y.allocateGPU(kCloudCount, kAllocatedCapacity, RelightPresence{true, true, true, true});
    check(y.count() == kCloudCount, "B: count() is the allocated count");
    check(y.capacity() == kAllocatedCapacity, "B: capacity() is the allocated capacity");
    check(y.storageGeneration() > generationBefore,
          "B: allocateGPU advances storageGeneration()");
    check(y.views().means.valid(), "B: core views valid after allocateGPU");
    const RelightBufferViews & yr = y.relightViews();
    check(yr.normals.valid() && yr.baseColorRoughness.valid() && yr.metallic.valid()
              && yr.visibility.valid(),
          "B: all four relight views valid");
    check(y.positions().empty() && y.normals().empty(),
          "B: host vectors empty after allocateGPU");

    // Copy every stream of x's GPU buffers into y's, in one submission.
    const CanonicalGaussianBufferViews xv = x.views();
    const CanonicalGaussianBufferViews yv = y.views();
    const RelightBufferViews & xr = x.relightViews();
    const VkDeviceSize n = kCloudCount;
    const StreamCopy copies[] = {
        {xv.means, yv.means, n * kMeansStride},
        {xv.scales, yv.scales, n * kScalesStride},
        {xv.rotations, yv.rotations, n * kRotationsStride},
        {xv.shCoefficients, yv.shCoefficients,
         n * static_cast<VkDeviceSize>(kShCoefficientsPerGaussian) * kShStride},
        {xv.opacity, yv.opacity, n * kOpacityStride},
        {xr.normals, yr.normals, n * kNormalsStride},
        {xr.baseColorRoughness, yr.baseColorRoughness, n * kBaseColorRoughnessStride},
        {xr.metallic, yr.metallic, n * kMetallicStride},
        {xr.visibility, yr.visibility, n * kVisibilityCoefficients * kVisibilityStride},
    };
    device.immediateSubmit(
        [&](VkCommandBuffer cmd)
        {
            for (const StreamCopy & c : copies)
            {
                VkBufferCopy region{};
                region.srcOffset = c.src.offset;
                region.dstOffset = c.dst.offset;
                region.size = c.bytes;
                vkCmdCopyBuffer(cmd, c.src.buffer, c.dst.buffer, 1, &region);
            }
        });

    y.downloadFromGPU();
    checkStreams(want, copyStreams(y), "B: GPU copy, downloaded");
    check(y.positions().size() == y.count()
              && y.visibility().size() == y.count() * kVisibilityCoefficients,
          "B: downloaded vectors sized to count(), not capacity");

    // C: setBounds, read back field by field.
    SceneBounds bounds;
    bounds.min = glm::vec3(-1.0f, -2.0f, -3.0f);
    bounds.max = glm::vec3(4.0f, 5.0f, 6.0f);
    bounds.valid = true;
    bounds.focusCenter = glm::vec3(0.5f, 1.0f, 1.5f);
    bounds.focusRadius = 2.25f;
    y.setBounds(bounds);
    const SceneBounds & got = y.bounds();
    check(got.valid, "C: bounds.valid");
    check(got.min.x == -1.0f && got.min.y == -2.0f && got.min.z == -3.0f, "C: bounds.min");
    check(got.max.x == 4.0f && got.max.y == 5.0f && got.max.z == 6.0f, "C: bounds.max");
    check(got.focusCenter.x == 0.5f && got.focusCenter.y == 1.0f && got.focusCenter.z == 1.5f,
          "C: bounds.focusCenter");
    check(got.focusRadius == 2.25f, "C: bounds.focusRadius");

    // A partial presence: only normals.
    GaussianCloud z(device);
    z.allocateGPU(kPartialCount, 0, RelightPresence{true, false, false, false});
    check(z.count() == kPartialCount && z.capacity() == kPartialCount,
          "B: capacity 0 means capacity == count");
    const RelightBufferViews & zr = z.relightViews();
    check(zr.normals.valid(), "B: normals view valid for a normals-only presence");
    check(!zr.baseColorRoughness.valid() && !zr.metallic.valid() && !zr.visibility.valid(),
          "B: absent relight views invalid");
    z.downloadFromGPU();
    check(z.baseColorRoughness().empty() && z.metallic().empty() && z.visibility().empty(),
          "B: absent host streams stay empty after downloadFromGPU");
    check(z.normals().size() == kPartialCount, "B: present normals downloaded");

    // A count of 0 is refused.
    bool refused = false;
    try
    {
        GaussianCloud w(device);
        w.allocateGPU(0, 0, RelightPresence{});
    }
    catch (const std::exception &)
    {
        refused = true;
    }
    check(refused, "B: allocateGPU(0, ...) throws");
}

// D: GpuScan against the CPU scan, one instance for every case.
void checkGpuScan(Device & device)
{
    GpuScan scan(device, testShaderDir());
    const uint32_t b = kScanBlock;
    const uint32_t sizes[] = {1,         2,     255,   256,     2047,  2048,
                              2049,      4096,  100003, 1000003, b - 1, b,
                              b + 1,     2 * b, 3 * b + 7};
    size_t caseIndex = 0;
    for (uint32_t n : sizes)
    {
        const bool flags = caseIndex++ % 2 == 0;
        const std::vector<uint32_t> in = randomValues(n, flags);
        std::vector<uint32_t> want;
        cpuExclusiveScan(in, 1, 0, n, want);

        const VkDeviceSize bytes = static_cast<VkDeviceSize>(n) * kUintStride;
        Buffer inBuf(device, bytes);
        Buffer outBuf(device, bytes);
        inBuf.upload(in.data(), bytes);
        device.immediateSubmit(
            [&](VkCommandBuffer cmd) { scan.record(cmd, inBuf.handle(), outBuf.handle(), n); });
        std::vector<uint32_t> got(n);
        outBuf.download(got.data(), bytes);
        const std::string label =
            "D: GpuScan n=" + std::to_string(n) + (flags ? " (0/1)" : " (0..1000)");
        check(got == want, label + firstMismatch(got, want));
    }

    // Strided: the values sit at in[3 * j + 1]; the other two lanes are noise.
    const uint32_t strided = 5000;
    const std::vector<uint32_t> stridedIn = randomValues(3 * strided, false);
    std::vector<uint32_t> want;
    cpuExclusiveScan(stridedIn, 3, 1, strided, want);
    const VkDeviceSize inBytes = static_cast<VkDeviceSize>(stridedIn.size()) * kUintStride;
    const VkDeviceSize outBytes = static_cast<VkDeviceSize>(strided) * kUintStride;
    Buffer inBuf(device, inBytes);
    Buffer outBuf(device, outBytes);
    inBuf.upload(stridedIn.data(), inBytes);
    device.immediateSubmit(
        [&](VkCommandBuffer cmd)
        { scan.record(cmd, inBuf.handle(), outBuf.handle(), strided, 3, 1); });
    std::vector<uint32_t> got(strided);
    outBuf.download(got.data(), outBytes);
    check(got == want, "D: GpuScan strided (stride 3, offset 1), n=5000" + firstMismatch(got, want));
}

// E: EditPass::exclusiveScan against the CPU scan, and its refusals.
void checkEditPassScan(Device & device)
{
    EditPass edit(device, testShaderDir());
    const uint32_t sizes[] = {1, 100003};
    for (uint32_t n : sizes)
    {
        const std::vector<uint32_t> in = randomValues(n, true);
        std::vector<uint32_t> want;
        const uint32_t total = cpuExclusiveScan(in, 1, 0, n, want);

        const VkDeviceSize bytes = static_cast<VkDeviceSize>(n) * kUintStride;
        Buffer inBuf(device, bytes);
        Buffer outBuf(device, bytes);
        inBuf.upload(in.data(), bytes);
        const uint32_t got =
            edit.exclusiveScan(inBuf.view(kUintStride), outBuf.view(kUintStride), n);
        std::vector<uint32_t> gotScan(n);
        outBuf.download(gotScan.data(), bytes);

        const std::string label = "E: exclusiveScan n=" + std::to_string(n);
        check(gotScan == want, label + " output" + firstMismatch(gotScan, want));
        check(got == total, label + " total " + std::to_string(got) + ", CPU "
                                + std::to_string(total));
    }

    // A count of 0 scans nothing and returns 0.
    Buffer spareIn(device, 16 * kUintStride);
    Buffer spareOut(device, 16 * kUintStride);
    check(edit.exclusiveScan(spareIn.view(kUintStride), spareOut.view(kUintStride), 0) == 0,
          "E: count 0 returns a total of 0");

    // An input that aliases the output is refused before anything is submitted.
    bool aliasRefused = false;
    try
    {
        edit.exclusiveScan(spareIn.view(kUintStride), spareIn.view(kUintStride), 16);
    }
    catch (const std::exception &)
    {
        aliasRefused = true;
    }
    check(aliasRefused, "E: input == output is refused");
}

// Runs one section. A throw from it fails that section only; the rest still run.
void runSection(const char * name, void (*section)(Device &), Device & device)
{
    try
    {
        section(device);
    }
    catch (const std::exception & e)
    {
        check(false, std::string(name) + " threw: " + e.what());
    }
}

} // namespace

int main()
{
    try
    {
        Device device;
        device.initialize(/*instanceExtensions=*/{}, /*enableValidation=*/true);
        std::printf("validation layer: enabled\n");

        runSection("A", checkDownloadFromGPU, device);
        runSection("B/C", checkAllocateGPU, device);
        runSection("D", checkGpuScan, device);
        runSection("E", checkEditPassScan, device);
        runSection("F", checkMergeLayers, device);
        runSection("G", checkRendererMerge, device);
        runSection("H", checkRefit, device);
        runSection("I", timeMergeScene, device);
        runSection("J", checkSelection, device);
        runSection("K", checkHighlight, device);
        runSection("L/M", checkOps, device);
        runSection("N", checkLarge, device);
        runSection("O", checkBorrowGuard, device);
    }
    catch (const std::exception & e)
    {
        check(false, std::string("device: ") + e.what());
    }

    std::printf("edit_test: %d checks, %d failed\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
