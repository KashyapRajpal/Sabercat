// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The Gaussian ray-tracing core.
//
//  1. AABBs: every box equals a CPU (double) computation of the exact box
//     around the Gaussian's 3-sigma ellipsoid, and contains points sampled
//     on that ellipsoid; degenerate Gaussians get a collapsed box.
//  2. Traversal: gaussianTransmittance, traced on the GPU against the TLAS,
//     equals a CPU brute force over ALL Gaussians with the same formula --
//     random rays (unnormalized directions, assorted [tMin, tMax]) and rays
//     aimed through Gaussian centers (some ending before the center), over a
//     synthetic scene of overlapping anisotropic, rotated, tiny, near-zero-
//     opacity, clamped-opacity and degenerate Gaussians.
//  3. refit() after moving Gaussians (and pushing opacities across 1/255)
//     equals a fresh build; build/refit argument checks; a smaller rebuild
//     keeps the TLAS handle.
//  4. Application-owned device: on a device created with
//     DeviceRequirements::Options::rayTracing the same scene traces the
//     same; on one created without it, the class refuses with a named error
//     and declaring rayQuery is rejected through vkGetDeviceProcAddr.
//  5. Alpha cutoffs: build(minAlpha) at 0.02 and 0.2 -- AABBs, traversal and
//     refit against the CPU at that cutoff. A rejected build changes nothing.
//     Growth from 40 to 20 000 Gaussians, one Gaussian, and 20 successive
//     refits against a fresh build.
//  6. Optionally assets/bonsai-7k.ply (path = argv[1], ./assets, or the
//     source tree's assets/): a 20k-Gaussian subset against the CPU, then the
//     whole scene's build/refit time and rays/s.
//
// Skips ([SKIP]) on a GPU without ray query. Validation messages are logged
// as "[validation]"; the run must have none.

#include <vulkan/vulkan.h>

#include <glm/glm.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "sabercat/buffer.h"
#include "sabercat/command_buffer.h"
#include "sabercat/descriptor_set.h"
#include "sabercat/device.h"
#include "sabercat/gaussian_acceleration_structure.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/pipeline.h"
#include "sabercat/render_config.h"
#include "sabercat/scene_loader.h"
#include "sabercat/shader_compiler.h"

using namespace sabercat;

namespace
{

int g_failures = 0;

void check(bool ok, const std::string & what)
{
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok)
    {
        ++g_failures;
    }
}

std::string fmt(const char * format, double a, double b = 0.0, double c = 0.0)
{
    char buffer[256];
    std::snprintf(buffer, sizeof(buffer), format, a, b, c);
    return buffer;
}

// Runs `fn`; returns the message of the exception it threw ("" if none), and
// whether it was std::invalid_argument.
std::string thrownMessage(const std::function<void()> & fn, bool * invalidArgument = nullptr)
{
    try
    {
        fn();
    }
    catch (const std::invalid_argument & e)
    {
        if (invalidArgument)
        {
            *invalidArgument = true;
        }
        return e.what();
    }
    catch (const std::exception & e)
    {
        if (invalidArgument)
        {
            *invalidArgument = false;
        }
        return e.what();
    }
    return "";
}

bool contains(const std::string & haystack, const std::string & needle)
{
    return haystack.find(needle) != std::string::npos;
}

// ---- Scene + rays (host side) ----

struct Scene
{
    std::vector<glm::vec4> means, scales, rotations;
    std::vector<float> opacities;

    uint32_t size() const
    {
        return static_cast<uint32_t>(means.size());
    }
    void add(glm::vec3 mean, glm::vec3 scale, glm::vec4 rotation, float opacity)
    {
        means.emplace_back(mean, 0.0f);
        scales.emplace_back(scale, 0.0f);
        rotations.push_back(rotation);
        opacities.push_back(opacity);
    }
};

// Mirrors trace_rays.slang's TestRay (two float4s, 32 B).
struct TestRay
{
    glm::vec4 originTMin;
    glm::vec4 directionTMax;
};
static_assert(sizeof(TestRay) == 32, "TestRay mirrors the shader struct");

// The alpha cutoff a structure is built and traced with (RayCutoffs). The
// default is the exact model: the shader's literal 1/255 and 255.
struct Cutoffs
{
    float minAlpha = 1.0f / 255.0f;
    float invMinAlpha = 255.0f;
    float minTransmittance = 1e-4f;

    static Cutoffs withMinAlpha(float minAlpha)
    {
        Cutoffs c;
        c.minAlpha = minAlpha;
        c.invMinAlpha = 1.0f / minAlpha;
        return c;
    }
};

// Mirrors trace_rays.slang's TraceParams (16 B push constants).
struct TraceParams
{
    uint32_t rayCount;
    float minAlpha;
    float invMinAlpha;
    float minTransmittance;
};
static_assert(sizeof(TraceParams) == 16, "TraceParams mirrors the shader push constants");

glm::vec4 randomRotation(std::mt19937 & rng)
{
    std::normal_distribution<float> n(0.0f, 1.0f);
    return glm::normalize(glm::vec4(n(rng), n(rng), n(rng), n(rng)));
}

glm::vec3 randomDirection(std::mt19937 & rng)
{
    std::normal_distribution<float> n(0.0f, 1.0f);
    return glm::normalize(glm::vec3(n(rng), n(rng), n(rng)));
}

Scene makeSyntheticScene(std::mt19937 & rng)
{
    Scene s;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const glm::vec4 identity(0.0f, 0.0f, 0.0f, 1.0f);
    // Two overlapping, strongly anisotropic, rotated Gaussians.
    s.add({0.0f, 0.0f, 0.0f}, {0.6f, 0.15f, 0.05f}, glm::normalize(glm::vec4(0.3f, -0.2f, 0.5f, 0.8f)), 0.8f);
    s.add({0.2f, 0.1f, 0.0f}, {0.08f, 0.5f, 0.2f}, glm::normalize(glm::vec4(-0.6f, 0.1f, 0.2f, 0.7f)), 0.6f);
    // Tiny (millimetre) Gaussians, one very flat.
    s.add({1.0f, -0.5f, 0.3f}, {0.002f, 0.003f, 0.001f}, randomRotation(rng), 0.9f);
    s.add({-1.0f, 0.7f, -0.2f}, {0.01f, 0.01f, 0.0005f}, randomRotation(rng), 0.7f);
    // Opacity just below 1/255 (never contributes: degenerate) and just above
    // (contributes only near its core).
    s.add({0.5f, 0.5f, 0.5f}, {0.3f, 0.3f, 0.3f}, identity, 0.003f);
    s.add({-0.5f, -0.5f, 0.5f}, {0.3f, 0.2f, 0.25f}, identity, 0.0045f);
    // Opacity 1: alpha clamps to 0.99.
    s.add({0.0f, 1.2f, 0.0f}, {0.2f, 0.2f, 0.4f}, randomRotation(rng), 1.0f);
    // A non-unit quaternion (normalized by the core).
    s.add({1.3f, 1.3f, -1.0f}, {0.3f, 0.1f, 0.2f}, 3.0f * glm::normalize(glm::vec4(0.1f, 0.7f, 0.2f, 0.5f)), 0.75f);
    // Degenerate: zero scale axis, zero quaternion, NaN mean, infinite scale,
    // an overflowing scale.
    s.add({0.7f, -1.0f, 0.0f}, {0.2f, 0.0f, 0.2f}, identity, 0.9f);
    s.add({-0.7f, -1.0f, 0.0f}, {0.2f, 0.2f, 0.2f}, glm::vec4(0.0f), 0.9f);
    s.add({nan, 0.0f, 0.0f}, {0.2f, 0.2f, 0.2f}, identity, 0.9f);
    s.add({0.0f, -1.6f, 0.4f}, {0.2f, inf, 0.2f}, identity, 0.9f);
    s.add({0.0f, -1.6f, -0.4f}, {1e37f, 0.2f, 0.2f}, identity, 0.9f);
    // A dense stack: rays through it fall below T = 1e-4 (reported as 0).
    for (int k = 0; k < 6; ++k)
    {
        s.add({-1.5f, 1.5f, 0.05f * static_cast<float>(k)}, {0.15f, 0.12f, 0.1f}, randomRotation(rng), 0.95f);
    }
    // Random population.
    std::uniform_real_distribution<float> pos(-2.0f, 2.0f);
    std::uniform_real_distribution<float> logScale(std::log(0.01f), std::log(0.4f));
    std::uniform_real_distribution<float> opac(0.02f, 1.0f);
    for (int i = 0; i < 300; ++i)
    {
        s.add({pos(rng), pos(rng), pos(rng)},
              {std::exp(logScale(rng)), std::exp(logScale(rng)), std::exp(logScale(rng))},
              randomRotation(rng), opac(rng));
    }
    return s;
}

// ---- CPU reference: gaussian_ray.slangh in double ----

bool finite3(const glm::dvec3 & v)
{
    return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z);
}

struct CpuFrame
{
    glm::dmat3 rt; // R^T as rows: rt[k] = column k of R = local axis k
    glm::dvec3 invScale;
    glm::dvec3 scale;
    glm::dvec3 mean;
    double opacity;
};

// The validity rule, evaluated in float like the shader (it decides on input
// values only: compares, a product, a sum).
bool cpuFrame(const Scene & s, uint32_t i, CpuFrame & f)
{
    const glm::vec3 mean(s.means[i]);
    const glm::vec3 scale(s.scales[i]);
    const glm::vec4 q = s.rotations[i];
    const float opacity = s.opacities[i];
    const float qq = glm::dot(q, q);
    const auto fin = [](glm::vec3 v) { return std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z); };
    if (!(opacity >= 1.0f / 255.0f) || !fin(mean) || !fin(scale)
        || !(std::min(scale.x, std::min(scale.y, scale.z)) > 0.0f) || !(qq > 0.0f) || !std::isfinite(qq))
    {
        return false;
    }
    const float maxScale = std::max(scale.x, std::max(scale.y, scale.z));
    if (!(maxScale <= 1e15f)) // SABERCAT_RAY_MAX_SCALE
    {
        return false;
    }
    const glm::dvec4 n = glm::dvec4(q) / std::sqrt(glm::dot(glm::dvec4(q), glm::dvec4(q)));
    const double x = n.x, y = n.y, z = n.z, w = n.w;
    // R row-major as forward_common.slangh's quaternionToMatrix.
    const double r[3][3] = {{1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)},
                            {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)},
                            {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)}};
    for (int k = 0; k < 3; ++k)
    {
        f.rt[k] = glm::dvec3(r[0][k], r[1][k], r[2][k]); // column k of R
    }
    f.scale = glm::dvec3(scale);
    f.invScale = 1.0 / f.scale;
    f.mean = glm::dvec3(mean);
    f.opacity = std::min(static_cast<double>(opacity), 1.0);
    return true;
}

// R^T v.
glm::dvec3 toLocal(const CpuFrame & f, const glm::dvec3 & v)
{
    return glm::dvec3(glm::dot(f.rt[0], v), glm::dot(f.rt[1], v), glm::dot(f.rt[2], v));
}

// gaussian_ray.slangh's gaussianSupportSigmas: 3 sigma, tightened by opacity
// (a ray clears the alpha skip only for d2 <= 2 ln(o / minAlpha)), plus margin.
double cpuSupportSigmas(double opacity, double minAlpha = 1.0 / 255.0)
{
    const double k2 = 2.0 * std::log(opacity / minAlpha) * 1.001 + 1e-3;
    return std::sqrt(std::clamp(k2, 0.0, static_cast<double>(kRadiusSigmas) * kRadiusSigmas));
}

// A Gaussian that gets a box and can contribute at this cutoff: valid (cpuFrame)
// and not fainter than the alpha skip.
bool contributes(const Scene & s, uint32_t i, double minAlpha, CpuFrame & f)
{
    return cpuFrame(s, i, f) && f.opacity >= minAlpha;
}

struct CpuAabb
{
    glm::dvec3 lo, hi;
};

CpuAabb cpuAabb(const Scene & s, uint32_t i, double minAlpha = 1.0 / 255.0)
{
    CpuFrame f;
    const glm::dvec3 mean(glm::vec3(s.means[i]));
    if (!contributes(s, i, minAlpha, f))
    {
        const glm::dvec3 p = finite3(mean) ? mean : glm::dvec3(0.0);
        return {p, p};
    }
    // Row i of M = R S: (R_i0 s0, R_i1 s1, R_i2 s2); rt[k][i] = R_ik.
    glm::dvec3 extent;
    for (int row = 0; row < 3; ++row)
    {
        const glm::dvec3 m(f.rt[0][row] * f.scale.x, f.rt[1][row] * f.scale.y, f.rt[2][row] * f.scale.z);
        extent[row] = cpuSupportSigmas(f.opacity, minAlpha) * glm::length(m);
    }
    return {mean - extent, mean + extent};
}

struct CpuTrace
{
    double transmittance = 1.0;
    bool borderline = false; // some decision lies within rounding of a threshold
};

CpuTrace cpuTransmittance(const Scene & s, uint32_t count, const TestRay & ray, double minAlpha = 1.0 / 255.0)
{
    const glm::dvec3 origin(glm::vec3(ray.originTMin));
    const glm::dvec3 dir(glm::vec3(ray.directionTMax));
    const double tMin = ray.originTMin.w, tMax = ray.directionTMax.w;
    const double maxD2 = static_cast<double>(kRadiusSigmas) * kRadiusSigmas;
    CpuTrace result;
    for (uint32_t i = 0; i < count; ++i)
    {
        CpuFrame f;
        if (!cpuFrame(s, i, f))
        {
            continue;
        }
        const glm::dvec3 o = toLocal(f, origin - f.mean) * f.invScale;
        const glm::dvec3 d = toLocal(f, dir) * f.invScale;
        const double dd = glm::dot(d, d);
        const glm::dvec3 c = glm::cross(o, d);
        const double d2 = glm::dot(c, c) / dd;
        const double t = -glm::dot(o, d) / dd;
        const double rawAlpha = f.opacity * std::exp(-0.5 * d2);
        const double alpha = std::min(rawAlpha, static_cast<double>(kMaxAlpha));
        const double tTol = 1e-4 * std::max(1.0, std::fabs(t));
        if (std::fabs(d2 - maxD2) < 1e-3 * maxD2 || std::fabs(alpha - minAlpha) < 2e-3 * minAlpha
            || std::fabs(t - tMin) < tTol || std::fabs(t - tMax) < tTol)
        {
            result.borderline = true;
        }
        if (d2 > maxD2 || alpha < minAlpha || t < tMin || t > tMax)
        {
            continue;
        }
        result.transmittance *= 1.0 - alpha;
    }
    if (std::fabs(result.transmittance - 1e-4) < 2e-6)
    {
        result.borderline = true;
    }
    if (result.transmittance < 1e-4)
    {
        result.transmittance = 0.0;
    }
    return result;
}

struct Comparison
{
    double worst = 0.0;
    uint32_t compared = 0;
    uint32_t borderline = 0;
    uint32_t occluded = 0;  // CPU T == 0 (early-out)
    uint32_t partial = 0;   // 0 < T < 1
};

Comparison compare(const Scene & s, uint32_t count, const std::vector<TestRay> & rays,
                   const std::vector<float> & gpu, double minAlpha = 1.0 / 255.0)
{
    Comparison c;
    for (size_t r = 0; r < rays.size(); ++r)
    {
        const CpuTrace ref = cpuTransmittance(s, count, rays[r], minAlpha);
        if (ref.borderline)
        {
            ++c.borderline;
            continue;
        }
        ++c.compared;
        c.occluded += ref.transmittance == 0.0 ? 1 : 0;
        c.partial += (ref.transmittance > 0.0 && ref.transmittance < 1.0) ? 1 : 0;
        c.worst = std::max(c.worst, std::fabs(ref.transmittance - static_cast<double>(gpu[r])));
    }
    return c;
}

std::vector<TestRay> makeRays(const Scene & s, std::mt19937 & rng, uint32_t randomCount,
                              float extent)
{
    std::vector<TestRay> rays;
    std::uniform_real_distribution<float> pos(-extent, extent);
    std::uniform_real_distribution<float> unit(0.0f, 1.0f);
    std::uniform_real_distribution<float> length(0.5f, 2.0f);
    for (uint32_t r = 0; r < randomCount; ++r)
    {
        // Unnormalized direction: t is in units of |dir|.
        const glm::vec3 origin(pos(rng), pos(rng), pos(rng));
        const glm::vec3 dir = randomDirection(rng) * length(rng);
        const float tMin = unit(rng) < 0.3f ? 0.0f : unit(rng);
        const float tMax = tMin + 0.5f + 7.5f * unit(rng);
        rays.push_back({glm::vec4(origin, tMin), glm::vec4(dir, tMax)});
    }
    // Aimed through each finite center from 1.5-4 units away; a quarter of
    // them stop short of the center (t* past tMax -> not counted).
    for (uint32_t i = 0; i < s.size(); ++i)
    {
        const glm::vec3 mean(s.means[i]);
        if (!std::isfinite(mean.x))
        {
            continue;
        }
        const float distance = 1.5f + 2.5f * unit(rng);
        const glm::vec3 origin = mean - randomDirection(rng) * distance;
        const glm::vec3 dir = glm::normalize(mean - origin);
        const float tMax = unit(rng) < 0.25f ? distance * (0.6f + 0.35f * unit(rng))
                                             : distance + 2.0f * unit(rng);
        rays.push_back({glm::vec4(origin, 0.0f), glm::vec4(dir, tMax)});
    }
    return rays;
}

// ---- GPU side ----

// The four parameter streams packed into ONE buffer at aligned offsets -- the
// way a model-B owner lays them out -- so the views carry non-zero offsets.
struct GpuScene
{
    std::unique_ptr<Buffer> buffer;
    CanonicalGaussianBufferViews views{};
    VkDeviceSize offsets[4] = {};
    uint32_t capacity = 0;

    GpuScene(Device & device, uint32_t capacity_) : capacity(capacity_)
    {
        const VkDeviceSize align =
            std::max<VkDeviceSize>(device.properties().limits.minStorageBufferOffsetAlignment, 256);
        const VkDeviceSize bytes[4] = {capacity * 16ull, capacity * 16ull, capacity * 16ull,
                                       capacity * 4ull};
        VkDeviceSize total = align; // start past 0 on purpose
        for (int k = 0; k < 4; ++k)
        {
            offsets[k] = total;
            total = (total + bytes[k] + align - 1) / align * align;
        }
        buffer = std::make_unique<Buffer>(device, total);
        views.means = buffer->view(kMeansStride, offsets[0]);
        views.scales = buffer->view(kScalesStride, offsets[1]);
        views.rotations = buffer->view(kRotationsStride, offsets[2]);
        views.opacity = buffer->view(kOpacityStride, offsets[3]);
        views.means.size = bytes[0];
        views.scales.size = bytes[1];
        views.rotations.size = bytes[2];
        views.opacity.size = bytes[3];
    }

    // Uploads the first `count` Gaussians of `s` (starting at `first`).
    void upload(const Scene & s, uint32_t count, uint32_t first = 0)
    {
        std::vector<uint8_t> host(static_cast<size_t>(buffer->size()), 0);
        std::memcpy(host.data() + offsets[0], s.means.data() + first, count * 16ull);
        std::memcpy(host.data() + offsets[1], s.scales.data() + first, count * 16ull);
        std::memcpy(host.data() + offsets[2], s.rotations.data() + first, count * 16ull);
        std::memcpy(host.data() + offsets[3], s.opacities.data() + first, count * 4ull);
        buffer->upload(host.data(), host.size());
        views.gaussianCount = count;
    }
};

// Runs trace_rays.slang: one gaussianTransmittance per ray.
class Tracer
{
  public:
    Tracer(Device & device) : m_device(device), m_descriptors(device.handle())
    {
        VkDevice dev = device.handle();
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        for (uint32_t i = 0; i < 7; ++i)
        {
            VkDescriptorSetLayoutBinding b{};
            b.binding = i;
            b.descriptorType = i == 0 ? VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR
                                      : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            b.descriptorCount = 1;
            b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings.push_back(b);
        }
        m_layout = m_descriptors.createLayout(bindings);
        m_descriptors.createPool(1, {{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1},
                                     {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 6}});
        m_set = m_descriptors.allocate(m_layout);
        ShaderCompiler compiler(dev);
        auto spirv = compiler.loadSPIRV(std::string(TEST_SHADER_DIR) + "/trace_rays.comp.spv");
        VkShaderModule module = compiler.createShaderModule(spirv);
        VkPipelineLayout layout =
            m_pipeline.createPipelineLayout(dev, m_layout, sizeof(TraceParams), VK_SHADER_STAGE_COMPUTE_BIT);
        m_pipeline.createComputePipeline(dev, module, layout);
        compiler.destroyShaderModule(module);
    }
    ~Tracer()
    {
        m_pipeline.destroy(m_device.handle());
    }

    std::vector<float> trace(const GaussianAccelerationStructure & as,
                             const CanonicalGaussianBufferViews & views,
                             const std::vector<TestRay> & rays, double * seconds = nullptr,
                             const Cutoffs & cutoffs = Cutoffs{})
    {
        const uint32_t count = static_cast<uint32_t>(rays.size());
        const TraceParams params{count, cutoffs.minAlpha, cutoffs.invMinAlpha, cutoffs.minTransmittance};
        Buffer rayBuffer(m_device, rays.size() * sizeof(TestRay));
        rayBuffer.upload(rays.data(), rays.size() * sizeof(TestRay));
        Buffer out(m_device, rays.size() * sizeof(float));

        m_descriptors.bindAccelerationStructure(m_set, 0, as.tlas());
        const BufferView streams[4] = {views.means, views.scales, views.rotations, views.opacity};
        for (uint32_t k = 0; k < 4; ++k)
        {
            m_descriptors.bindBuffer(m_set, 1 + k, streams[k].buffer, streams[k].offset,
                                     streams[k].size, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        }
        m_descriptors.bindBuffer(m_set, 5, rayBuffer.handle(), 0, rayBuffer.size(),
                                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
        m_descriptors.bindBuffer(m_set, 6, out.handle(), 0, out.size(),
                                 VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);

        const auto start = std::chrono::steady_clock::now();
        m_device.immediateSubmit(
            [&](VkCommandBuffer cmd)
            {
                vkCmdPushConstants(cmd, m_pipeline.getLayout(), VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                   sizeof(params), &params);
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline.getHandle());
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline.getLayout(),
                                        0, 1, &m_set, 0, nullptr);
                vkCmdDispatch(cmd, (count + 63) / 64, 1, 1);
            });
        if (seconds)
        {
            *seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        }
        std::vector<float> result(rays.size());
        out.download(result.data(), result.size() * sizeof(float));
        return result;
    }

  private:
    Device & m_device;
    DescriptorSet m_descriptors;
    VkDescriptorSetLayout m_layout = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;
    Pipeline m_pipeline;
};

// Tolerance for |T_gpu - T_cpu| on rays with no decision within rounding of
// a threshold. Each alpha carries fp32 error from the whitening (|o'| up to
// ~4e3 for the millimetre Gaussians, so d2 ~ 2 sqrt(d2) |o'| 2^-24 ~ 1e-3
// absolute at worst, i.e. ~5e-4 relative in that alpha, but only where such
// a Gaussian is hit near its core) plus a few ulp from exp and the product;
// T moves by at most sum(alpha_i * rel_err_i). 1e-4 is ~1000 ulp of 1 and
// still 40x below the smallest alpha that counts (1/255).
constexpr double kTransmittanceTolerance = 1e-4;

void reportComparison(const Comparison & c, const std::string & what)
{
    check(c.worst <= kTransmittanceTolerance && c.compared > 0,
          what + fmt(": worst |T_gpu - T_cpu| %.3g over %.0f rays", c.worst, c.compared)
              + fmt(" (%.0f fully occluded, %.0f partial, ", c.occluded, c.partial)
              + fmt("%.0f borderline skipped; tol 1e-4)", c.borderline));
}

// AABBs vs the CPU, and containment of sampled 3-sigma ellipsoid points.
void checkAabbs(const GaussianAccelerationStructure & as, const Scene & s, uint32_t count,
                std::mt19937 & rng, double minAlpha = 1.0 / 255.0, const std::string & label = "")
{
    std::vector<float> boxes(static_cast<size_t>(count) * 6);
    as.aabbs().download(boxes.data(), boxes.size() * sizeof(float));
    double worst = 0.0, worstOutside = 0.0;
    uint32_t collapsed = 0, collapsedExpected = 0;
    for (uint32_t i = 0; i < count; ++i)
    {
        const CpuAabb ref = cpuAabb(s, i, minAlpha);
        const float * b = &boxes[i * 6];
        const glm::dvec3 lo(b[0], b[1], b[2]), hi(b[3], b[4], b[5]);
        const double scaleRef = 1.0 + glm::length(ref.hi - ref.lo) + glm::length(ref.lo);
        worst = std::max(worst, std::max(glm::length(lo - ref.lo), glm::length(hi - ref.hi)) / scaleRef);
        CpuFrame f;
        if (!contributes(s, i, minAlpha, f))
        {
            ++collapsedExpected;
            collapsed += (lo == hi) ? 1 : 0;
            continue;
        }
        for (int k = 0; k < 64; ++k)
        {
            // mu + k R S u, |u| = 1: on the support ellipsoid.
            const glm::dvec3 u = glm::dvec3(randomDirection(rng)) * cpuSupportSigmas(f.opacity, minAlpha);
            const glm::dvec3 local = u * f.scale;
            const glm::dvec3 p = f.mean + f.rt[0] * local.x + f.rt[1] * local.y + f.rt[2] * local.z;
            const glm::dvec3 below = glm::max(lo - p, glm::dvec3(0.0));
            const glm::dvec3 above = glm::max(p - hi, glm::dvec3(0.0));
            worstOutside = std::max(worstOutside, glm::length(below + above) / scaleRef);
        }
    }
    check(worst <= 1e-6, label + fmt("AABBs match the CPU exact support box (worst rel. %.3g over %.0f)", worst, count));
    check(worstOutside <= 1e-6,
          label + fmt("sampled support-ellipsoid points lie inside their AABB (worst rel. outside %.3g)", worstOutside));
    check(collapsed == collapsedExpected,
          label + fmt("degenerate Gaussians get a collapsed box (%.0f of %.0f)", collapsed, collapsedExpected));
}

// ---- A model-B owner (as borrowed_test's): its own instance and device ----

VKAPI_ATTR VkBool32 VKAPI_CALL appDebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                VkDebugUtilsMessageTypeFlagsEXT,
                                                const VkDebugUtilsMessengerCallbackDataEXT * data, void *)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
    {
        std::printf("[validation] %s\n", data->pMessage);
    }
    return VK_FALSE;
}

class Application
{
  public:
    explicit Application(DeviceRequirements::Need rayTracing)
    {
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "gaussian_raytrace_test";
        appInfo.apiVersion = kRequiredVulkanApiVersion;
        const char * layer = "VK_LAYER_KHRONOS_validation";
        const char * debugExt = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
        VkDebugUtilsMessengerCreateInfoEXT debugInfo{};
        debugInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        debugInfo.messageSeverity =
            VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
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
        vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, nullptr);
        std::vector<VkQueueFamilyProperties> families(count);
        vkGetPhysicalDeviceQueueFamilyProperties(gpu, &count, families.data());
        for (uint32_t i = 0; i < count; ++i)
        {
            if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT)
            {
                family = i;
                break;
            }
        }

        requirements = std::make_unique<DeviceRequirements>(
            gpu, DeviceRequirements::Options{DeviceRequirements::Need::None,
                                             DeviceRequirements::Need::None, rayTracing});
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
    }

    ~Application()
    {
        vkDeviceWaitIdle(device);
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
        return c;
    }

    VkInstance instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    VkPhysicalDevice gpu = VK_NULL_HANDLE;
    uint32_t family = UINT32_MAX;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    std::unique_ptr<DeviceRequirements> requirements;
};

std::string findBonsai(int argc, char ** argv)
{
    std::vector<std::string> candidates;
    if (argc > 1)
    {
        candidates.push_back(argv[1]);
    }
    candidates.push_back("assets/bonsai-7k.ply");
    candidates.push_back(std::string(ASSET_DIR) + "/bonsai-7k.ply");
    for (const std::string & c : candidates)
    {
        if (std::filesystem::exists(c))
        {
            return c;
        }
    }
    return "";
}

Scene sceneFromCloud(GaussianCloud & cloud)
{
    Scene s;
    s.means = cloud.positions();
    s.scales = cloud.scales();
    s.rotations = cloud.rotations();
    s.opacities = cloud.opacities();
    return s;
}

} // namespace

int main(int argc, char ** argv)
{
    try
    {
        std::mt19937 rng(20261007u);

        Device device;
        device.initialize({}, true);
        if (!device.features().rayQuery)
        {
            std::printf("[SKIP] device lacks acceleration structures / ray query\n");
            // The class must still refuse cleanly.
            const std::string message =
                thrownMessage([&] { GaussianAccelerationStructure as(device, SHADER_DIR); });
            check(contains(message, "ray query"), "GaussianAccelerationStructure refuses without ray query");
            device.shutdown();
            return g_failures == 0 ? 0 : 1;
        }
        check(device.features().bufferDeviceAddress && device.enabledFeatures().rayQuery,
              "model A enables ray query + acceleration structures + BDA (IfSupported)");

        // ---- 1-2. Synthetic scene: AABBs and traversal vs the CPU. ----
        const Scene scene = makeSyntheticScene(rng);
        const uint32_t n = scene.size();
        GpuScene gpu(device, n);
        gpu.upload(scene, n);
        GaussianAccelerationStructure as(device, SHADER_DIR);
        check(!as.built() && as.gaussianCount() == 0, "nothing built before build()");
        as.build(gpu.views);
        check(as.built() && as.gaussianCount() == n && as.tlas() != VK_NULL_HANDLE && as.tlasAddress() != 0,
              fmt("build(): TLAS for %.0f Gaussians (views at non-zero offsets)", n));
        checkAabbs(as, scene, n, rng);

        Tracer tracer(device);
        const std::vector<TestRay> rays = makeRays(scene, rng, 4000, 3.0f);
        // Rays through the dense stack: below 1e-4, reported as 0.
        std::vector<TestRay> stackRays;
        for (int k = 0; k < 16; ++k)
        {
            const glm::vec3 target(-1.5f, 1.5f, 0.125f);
            const glm::vec3 origin = target - randomDirection(rng) * 3.0f;
            stackRays.push_back({glm::vec4(origin, 0.0f), glm::vec4(glm::normalize(target - origin), 6.0f)});
        }
        const std::vector<float> stackT = tracer.trace(as, gpu.views, stackRays);
        bool allZero = true;
        for (size_t r = 0; r < stackRays.size(); ++r)
        {
            allZero = allZero && stackT[r] == 0.0f
                      && cpuTransmittance(scene, n, stackRays[r]).transmittance == 0.0;
        }
        check(allZero, "rays through 6 stacked opaque Gaussians: T < 1e-4 is reported as exactly 0");

        const std::vector<float> traced = tracer.trace(as, gpu.views, rays);
        reportComparison(compare(scene, n, rays, traced),
                         "GPU gaussianTransmittance == CPU brute force over all Gaussians");


        // ---- 3. Refit after moving Gaussians == fresh build. ----
        Scene moved = scene;
        {
            std::normal_distribution<float> jitter(0.0f, 0.15f);
            std::uniform_real_distribution<float> grow(0.8f, 1.25f);
            for (uint32_t i = 0; i < n; ++i)
            {
                if (!std::isfinite(moved.means[i].x) || !std::isfinite(moved.scales[i].y)
                    || moved.scales[i].x > 1e30f)
                {
                    continue; // the degenerate inputs stay as they are
                }
                moved.means[i] += glm::vec4(jitter(rng), jitter(rng), jitter(rng), 0.0f);
                moved.scales[i] *= grow(rng);
                if (glm::dot(moved.rotations[i], moved.rotations[i]) > 0.0f)
                {
                    moved.rotations[i] = glm::normalize(
                        moved.rotations[i] + 0.3f * glm::vec4(jitter(rng), jitter(rng), jitter(rng), jitter(rng)));
                }
            }
            // Opacity across 1/255 both ways (a collapsed box comes alive and
            // a live one collapses): the primitives stay active either way.
            moved.opacities[4] = 0.6f;
            moved.opacities[0] = 0.001f;
        }
        const VkAccelerationStructureKHR tlasBefore = as.tlas();
        gpu.upload(moved, n);
        as.refit(gpu.views);
        check(as.refitsSinceBuild() == 1 && as.tlas() == tlasBefore, "refit(): in place, same TLAS handle");
        checkAabbs(as, moved, n, rng);
        const std::vector<TestRay> movedRays = makeRays(moved, rng, 4000, 3.0f);
        const std::vector<float> refitT = tracer.trace(as, gpu.views, movedRays);
        {
            GaussianAccelerationStructure fresh(device, SHADER_DIR);
            fresh.build(gpu.views);
            const std::vector<float> freshT = tracer.trace(fresh, gpu.views, movedRays);
            double worst = 0.0;
            for (size_t r = 0; r < movedRays.size(); ++r)
            {
                worst = std::max(worst, std::fabs(static_cast<double>(refitT[r]) - freshT[r]));
            }
            check(worst <= 1e-6, fmt("refit after moving Gaussians == fresh build (worst %.3g, "
                                     "tol 1e-6: same candidates, only the product order differs)", worst));
        }
        reportComparison(compare(moved, n, movedRays, refitT), "refit structure == CPU brute force");

        // Argument checks.
        {
            bool invalid = false;
            CanonicalGaussianBufferViews bad = gpu.views;
            bad.gaussianCount = n - 1;
            std::string m = thrownMessage([&] { as.refit(bad); }, &invalid);
            check(invalid && contains(m, "call build() when the count changes"),
                  "refit() with a different count is rejected");
            bad = gpu.views;
            bad.scales.stride = 12;
            m = thrownMessage([&] { as.build(bad); }, &invalid);
            check(invalid && contains(m, "scales view stride 12"), "build() rejects a non-canonical stride");
            bad = gpu.views;
            bad.gaussianCount = n + 1;
            m = thrownMessage([&] { as.build(bad); }, &invalid);
            check(invalid && contains(m, "fewer than gaussianCount"), "build() rejects a count beyond the views");
            GaussianAccelerationStructure unbuilt(device, SHADER_DIR);
            m = thrownMessage([&] { unbuilt.refit(gpu.views); }, &invalid);
            check(invalid && contains(m, "before the first build"), "refit() before build() is rejected");
        }

        // A smaller rebuild reuses the (grow-only) storage and keeps the TLAS.
        {
            CanonicalGaussianBufferViews prefix = gpu.views;
            prefix.gaussianCount = 60;
            as.build(prefix);
            const std::vector<float> prefixT = tracer.trace(as, prefix, movedRays);
            check(as.tlas() == tlasBefore && as.gaussianCount() == 60 && as.refitsSinceBuild() == 0,
                  "smaller rebuild: same TLAS handle, count 60");
            reportComparison(compare(moved, 60, movedRays, prefixT), "60-Gaussian prefix == CPU");
            as.build(gpu.views);
        }

        // ---- 3b. Alpha cutoffs, rejected builds, growth, one Gaussian, many refits. ----
        // A structure built for a raised minAlpha boxes the support at that
        // cutoff and collapses fainter Gaussians; traced with the same cutoff it
        // equals the CPU brute force at that cutoff, and refit keeps it.
        for (float minAlpha : {0.02f, 0.2f})
        {
            const std::string tag = fmt("build(minAlpha %.2f): ", minAlpha);
            GpuScene own(device, n);
            own.upload(scene, n);
            GaussianAccelerationStructure ca(device, SHADER_DIR);
            ca.build(own.views, minAlpha);
            check(ca.minAlpha() == minAlpha, tag + "minAlpha() reports the build's cutoff");
            checkAabbs(ca, scene, n, rng, minAlpha, tag);
            const Cutoffs cut = Cutoffs::withMinAlpha(minAlpha);
            const std::vector<float> built = tracer.trace(ca, own.views, rays, nullptr, cut);
            reportComparison(compare(scene, n, rays, built, minAlpha), tag + "GPU == CPU brute force at that cutoff");

            own.upload(moved, n);
            ca.refit(own.views);
            check(ca.minAlpha() == minAlpha && ca.refitsSinceBuild() == 1,
                  tag + "refit keeps the cutoff (minAlpha() and one refit)");
            checkAabbs(ca, moved, n, rng, minAlpha, tag + "after refit: ");
            const std::vector<float> refitted = tracer.trace(ca, own.views, movedRays, nullptr, cut);
            reportComparison(compare(moved, n, movedRays, refitted, minAlpha),
                             tag + "after refit, GPU == CPU brute force at that cutoff");
        }
        {
            // Rejected calls change nothing: the cutoff and the built count stay.
            GpuScene own(device, n);
            own.upload(scene, n);
            GaussianAccelerationStructure ra(device, SHADER_DIR);
            ra.build(own.views);
            bool invalid = false;
            CanonicalGaussianBufferViews bad = own.views;
            bad.scales.stride = 12;
            std::string m = thrownMessage([&] { ra.build(bad, 0.2f); }, &invalid);
            check(invalid && contains(m, "scales view stride 12") && ra.minAlpha() == 1.0f / 255.0f
                      && ra.gaussianCount() == n,
                  "build(minAlpha 0.2) with a bad view is rejected and leaves the cutoff and count unchanged");
            const VkAccelerationStructureKHR before = ra.tlas();
            ra.refit(own.views);
            check(ra.refitsSinceBuild() == 1 && ra.tlas() == before,
                  "refit() after that rejected build still refits in place");
            m = thrownMessage([&] { ra.build(own.views, 0.001f); }, &invalid);
            check(invalid && contains(m, "minAlpha must be in [1/255, 1)"),
                  "build(minAlpha 0.001) is rejected (below the exact 1/255)");
            m = thrownMessage([&] { ra.build(own.views, 1.0f); }, &invalid);
            check(invalid && contains(m, "minAlpha must be in [1/255, 1)"), "build(minAlpha 1) is rejected");
            check(ra.minAlpha() == 1.0f / 255.0f, "rejected cutoffs leave minAlpha at 1/255");
        }
        {
            // Growth: a structure built over 40 Gaussians is rebuilt over 20 000
            // (past its capacity) with the same TLAS handle; a single Gaussian
            // on its own; then many successive refits against a fresh build.
            GpuScene own(device, n);
            GaussianAccelerationStructure ga(device, SHADER_DIR);
            own.upload(scene, 40);
            ga.build(own.views);
            const VkAccelerationStructureKHR firstTlas = ga.tlas();
            reportComparison(compare(scene, 40, rays, tracer.trace(ga, own.views, rays)),
                             "build over 40 Gaussians == CPU brute force over those 40");

            Scene large;
            std::uniform_real_distribution<float> lpos(-4.0f, 4.0f);
            std::uniform_real_distribution<float> lscale(std::log(0.01f), std::log(0.3f));
            std::uniform_real_distribution<float> lopacity(0.005f, 0.3f);
            for (int i = 0; i < 20000; ++i)
            {
                large.add({lpos(rng), lpos(rng), lpos(rng)},
                          {std::exp(lscale(rng)), std::exp(lscale(rng)), std::exp(lscale(rng))},
                          randomRotation(rng), lopacity(rng));
            }
            GpuScene big20k(device, 20000);
            big20k.upload(large, 20000);
            ga.build(big20k.views);
            check(ga.tlas() == firstTlas && ga.gaussianCount() == 20000,
                  "rebuild over 20000 Gaussians (past the 40-Gaussian capacity): same TLAS handle");
            std::vector<TestRay> largeRays;
            for (int r = 0; r < 1500; ++r)
            {
                const glm::vec3 origin(lpos(rng), lpos(rng), lpos(rng));
                largeRays.push_back({glm::vec4(origin, 0.0f), glm::vec4(randomDirection(rng) * 2.0f, 8.0f)});
            }
            reportComparison(compare(large, 20000, largeRays, tracer.trace(ga, big20k.views, largeRays)),
                             "20 000-Gaussian scene: GPU == CPU brute force over all 20 000");
            checkAabbs(ga, large, 20000, rng, 1.0 / 255.0, "20 000-Gaussian scene: ");

            GaussianAccelerationStructure one(device, SHADER_DIR);
            own.upload(scene, 1);
            one.build(own.views);
            check(one.gaussianCount() == 1 && one.tlas() != VK_NULL_HANDLE, "a structure of one Gaussian builds");
            reportComparison(compare(scene, 1, rays, tracer.trace(one, own.views, rays)),
                             "one-Gaussian structure == CPU brute force over that Gaussian");

            Scene step = scene;
            std::mt19937 jitterRng(7u);
            std::normal_distribution<float> jitter(0.0f, 0.02f);
            own.upload(step, n);
            GaussianAccelerationStructure gr(device, SHADER_DIR);
            gr.build(own.views);
            for (int it = 0; it < 20; ++it)
            {
                for (uint32_t i = 0; i < n; ++i)
                {
                    if (!std::isfinite(step.means[i].x) || !std::isfinite(step.scales[i].y)
                        || step.scales[i].x > 1e30f)
                    {
                        continue;
                    }
                    step.means[i] += glm::vec4(jitter(jitterRng), jitter(jitterRng), jitter(jitterRng), 0.0f);
                }
                own.upload(step, n);
                gr.refit(own.views);
            }
            check(gr.refitsSinceBuild() == 20, "20 successive refits are counted (refitsSinceBuild() == 20)");
            checkAabbs(gr, step, n, rng, 1.0 / 255.0, "after 20 refits: ");
            const std::vector<TestRay> stepRays = makeRays(step, rng, 4000, 3.0f);
            const std::vector<float> afterRefits = tracer.trace(gr, own.views, stepRays);
            GaussianAccelerationStructure stepFresh(device, SHADER_DIR);
            stepFresh.build(own.views);
            const std::vector<float> freshStep = tracer.trace(stepFresh, own.views, stepRays);
            double worst = 0.0;
            for (size_t r = 0; r < stepRays.size(); ++r)
            {
                worst = std::max(worst, std::fabs(static_cast<double>(afterRefits[r]) - freshStep[r]));
            }
            check(worst <= 1e-6, fmt("after 20 refits == fresh build (worst %.3g, tol 1e-6)", worst));
            reportComparison(compare(step, n, stepRays, afterRefits), "after 20 refits == CPU brute force");
        }

        // ---- 4. Application-owned device. ----
        {
            auto app = std::make_unique<Application>(DeviceRequirements::Need::IfSupported);
            const EnabledDeviceFeatures enabled = app->requirements->enabled();
            check(enabled.rayQuery, "DeviceRequirements rayTracing IfSupported enables it (declared rayQuery)");
            {
                Device borrowed;
                borrowed.adopt(app->context(), enabled);
                check(borrowed.features().rayQuery && borrowed.features().bufferDeviceAddress,
                      "adopt(): declared rayQuery reports ray query + BDA");
                {
                    // Every Sabercat object dies before the Device (and its
                    // allocator) that made it.
                    GpuScene bscene(borrowed, n);
                    bscene.upload(scene, n);
                    GaussianAccelerationStructure bas(borrowed, SHADER_DIR);
                    bas.build(bscene.views);
                    Tracer btracer(borrowed);
                    const std::vector<float> bT = btracer.trace(bas, bscene.views, rays);
                    double worst = 0.0;
                    for (size_t r = 0; r < rays.size(); ++r)
                    {
                        worst = std::max(worst, std::fabs(static_cast<double>(bT[r]) - traced[r]));
                    }
                    check(worst <= 1e-6,
                          fmt("borrowed device traces the same as model A (worst %.3g)", worst));
                }
                borrowed.shutdown();
            }
            app.reset();

            auto plain = std::make_unique<Application>(DeviceRequirements::Need::None);
            const EnabledDeviceFeatures none = plain->requirements->enabled();
            check(!none.rayQuery, "DeviceRequirements rayTracing None declares no rayQuery");
            {
                Device borrowed;
                borrowed.adopt(plain->context(), none);
                check(!borrowed.features().rayQuery && !borrowed.features().bufferDeviceAddress,
                      "device created without ray tracing reports rayQuery unsupported");
                const std::string m =
                    thrownMessage([&] { GaussianAccelerationStructure bas(borrowed, SHADER_DIR); });
                check(contains(m, "GaussianAccelerationStructure needs acceleration structures + ray query"),
                      "GaussianAccelerationStructure refuses with a named error");
                borrowed.shutdown();
            }
            {
                EnabledDeviceFeatures lie = none;
                lie.rayQuery = true;
                Device borrowed;
                bool invalid = false;
                const std::string m = thrownMessage([&] { borrowed.adopt(plain->context(), lie); }, &invalid);
                check(invalid && contains(m, "VK_KHR_acceleration_structure is not enabled on the device"),
                      "declaring rayQuery on a device created without it is rejected (proc-addr check)");
            }
            plain.reset();
        }

        // ---- 5. A real scene. ----
        const std::string bonsai = findBonsai(argc, argv);
        if (bonsai.empty())
        {
            std::printf("[SKIP] bonsai-7k.ply not found (pass its path as argv[1])\n");
        }
        else
        {
            std::shared_ptr<GaussianCloud> cloud = loadScene(device, bonsai);
            Scene full = sceneFromCloud(*cloud);
            const uint32_t total = full.size();
            const glm::vec3 center = cloud->bounds().focusCenter;
            const float radius = cloud->bounds().focusRadius;

            // Subset: the 20k Gaussians nearest the focus center (dense,
            // overlapping), against the CPU.
            {
                const uint32_t k = std::min<uint32_t>(20000, total);
                std::vector<uint32_t> order(total);
                for (uint32_t i = 0; i < total; ++i)
                {
                    order[i] = i;
                }
                const auto dist = [&](uint32_t i) { return glm::length(glm::vec3(full.means[i]) - center); };
                std::nth_element(order.begin(), order.begin() + k, order.end(),
                                 [&](uint32_t a, uint32_t b) { return dist(a) < dist(b); });
                Scene sub;
                float subRadius = 0.0f;
                for (uint32_t j = 0; j < k; ++j)
                {
                    const uint32_t i = order[j];
                    sub.add(glm::vec3(full.means[i]), glm::vec3(full.scales[i]), full.rotations[i], full.opacities[i]);
                    subRadius = std::max(subRadius, dist(i));
                }
                GpuScene gsub(device, k);
                gsub.upload(sub, k);
                as.build(gsub.views);
                std::vector<TestRay> subRays;
                std::uniform_real_distribution<float> unit(0.0f, 1.0f);
                std::uniform_int_distribution<uint32_t> pick(0, k - 1);
                for (int r = 0; r < 1500; ++r)
                {
                    // From outside the subset toward a random Gaussian center,
                    // stopping at it (a shadow ray toward a point) or passing it.
                    const glm::vec3 target(sub.means[pick(rng)]);
                    const glm::vec3 origin = center + randomDirection(rng) * (subRadius * 1.5f);
                    const float d = glm::length(target - origin);
                    const float tMax = r % 2 == 0 ? d * (0.9f + 0.2f * unit(rng)) : 2.0f * d;
                    subRays.push_back({glm::vec4(origin, 0.0f), glm::vec4((target - origin) / d, tMax)});
                }
                const std::vector<float> subT = tracer.trace(as, gsub.views, subRays);
                reportComparison(compare(sub, k, subRays, subT),
                                 "bonsai 20k-Gaussian core == CPU brute force");
            }

            // Whole scene: build, refit, rays/s.
            GpuScene gfull(device, total);
            gfull.upload(full, total);
            GaussianAccelerationStructure big(device, SHADER_DIR);
            const auto timeIt = [](const std::function<void()> & fn)
            {
                const auto start = std::chrono::steady_clock::now();
                fn();
                return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
            };
            const double firstBuild = timeIt([&] { big.build(gfull.views); });
            double build = 0.0, refit = 0.0;
            for (int i = 0; i < 3; ++i)
            {
                build += timeIt([&] { big.build(gfull.views); }) / 3.0;
                refit += timeIt([&] { big.refit(gfull.views); }) / 3.0;
            }
            std::vector<TestRay> perfRays;
            std::uniform_int_distribution<uint32_t> pick(0, total - 1);
            for (int r = 0; r < 1 << 20; ++r)
            {
                // Shadow-ray-like: from a Gaussian center toward a random
                // direction, starting just off the point, to the scene edge.
                const glm::vec3 origin(full.means[pick(rng)]);
                perfRays.push_back({glm::vec4(origin, 0.01f * radius), glm::vec4(randomDirection(rng), 4.0f * radius)});
            }
            double seconds = 0.0;
            tracer.trace(big, gfull.views, perfRays, &seconds); // warm-up
            const std::vector<float> perfT = tracer.trace(big, gfull.views, perfRays, &seconds);
            double meanT = 0.0;
            for (float t : perfT)
            {
                meanT += t / perfT.size();
            }
            std::printf("[INFO] bonsai %u Gaussians: first build %.1f ms, rebuild %.1f ms, refit %.1f ms "
                        "(each = AABB kernel + BLAS + TLAS, one fenced submit)\n",
                        total, firstBuild, build, refit);
            std::printf("[INFO] %u shadow-like rays (off a Gaussian center, random dir): %.1f ms -> %.1f Mrays/s, "
                        "mean T %.3f\n",
                        static_cast<uint32_t>(perfRays.size()), seconds * 1e3,
                        perfRays.size() / seconds / 1e6, meanT);
            check(big.gaussianCount() == total, "whole bonsai scene builds and traces");
        }
        // Everything above is destroyed before `device` (declared first).
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
