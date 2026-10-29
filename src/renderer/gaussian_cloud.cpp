// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/gaussian_cloud.h"

#include "sabercat/buffer.h"
#include "sabercat/device.h"
#include "sabercat/vk_check.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace sabercat
{

namespace
{

// Compile-time proof that the CPU staging types match the canonical GPU
// strides, so uploadToGPU() can memcpy straight from the vectors. If these
// ever diverge the shaders would read misaligned data -- fail the build
// instead. Every vec4-strided stream is checked individually so a typo in
// one alias can't hide behind another.
static_assert(sizeof(glm::vec4) == kMeansStride, "means stride must match glm::vec4");
static_assert(sizeof(glm::vec4) == kScalesStride, "scales stride must match glm::vec4");
static_assert(sizeof(glm::vec4) == kRotationsStride, "rotations stride must match glm::vec4");
static_assert(sizeof(glm::vec4) == kShStride, "SH stride must match glm::vec4");
static_assert(sizeof(float) == kOpacityStride, "opacity stride must match float");

} // namespace

GaussianCloud::GaussianCloud(Device & device) : m_device(device)
{
}

GaussianCloud::~GaussianCloud() = default;

void GaussianCloud::initialize(size_t count)
{
    m_count = count;
    m_positions.resize(count);
    m_scales.resize(count);
    m_rotations.resize(count);
    m_shCoefficients.resize(count * kShCoefficientsPerGaussian);
    m_opacities.resize(count);
    m_times.resize(count, 0.0f);
}

void GaussianCloud::computeBounds()
{
    m_bounds = SceneBounds{};
    if (m_count == 0 || m_positions.empty())
    {
        return; // leaves valid == false
    }

    glm::vec3 lo(std::numeric_limits<float>::max());
    glm::vec3 hi(std::numeric_limits<float>::lowest());
    bool sawFinite = false;

    for (size_t i = 0; i < m_count; ++i)
    {
        const glm::vec3 p(m_positions[i]);
        // Skip non-finite positions rather than letting one NaN poison the
        // whole result -- a single bad Gaussian would otherwise make every
        // comparison false and leave the bounds at their sentinel values,
        // which then frames the camera at an absurd distance.
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
        {
            continue;
        }
        lo = glm::min(lo, p);
        hi = glm::max(hi, p);
        sawFinite = true;
    }

    if (!sawFinite)
    {
        return;
    }

    m_bounds.min = lo;
    m_bounds.max = hi;
    m_bounds.valid = true;

    // Robust framing volume: median center, 90th-percentile radius. Both use
    // std::nth_element (O(n), no full sort) so this stays cheap even at a
    // million Gaussians. See SceneBounds for why the AABB alone is not a usable
    // framing target.
    std::vector<float> component;
    component.reserve(m_count);
    glm::vec3 medianCenter(0.0f);
    for (int axis = 0; axis < 3; ++axis)
    {
        component.clear();
        for (size_t i = 0; i < m_count; ++i)
        {
            const float v = m_positions[i][axis];
            if (std::isfinite(v))
            {
                component.push_back(v);
            }
        }
        if (component.empty())
        {
            return;
        }
        const size_t mid = component.size() / 2;
        std::nth_element(component.begin(), component.begin() + mid, component.end());
        medianCenter[axis] = component[mid];
    }

    std::vector<float> distances;
    distances.reserve(m_count);
    for (size_t i = 0; i < m_count; ++i)
    {
        const glm::vec3 p(m_positions[i]);
        if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z))
        {
            distances.push_back(glm::length(p - medianCenter));
        }
    }
    if (distances.empty())
    {
        return;
    }

    const size_t percentileIndex =
        static_cast<size_t>(0.9 * static_cast<double>(distances.size() - 1));
    std::nth_element(
        distances.begin(), distances.begin() + percentileIndex, distances.end());

    m_bounds.focusCenter = medianCenter;
    m_bounds.focusRadius = distances[percentileIndex];

    // A scene can legitimately have >90% of its Gaussians at one point (a
    // degenerate or single-object capture); fall back to the full radius rather
    // than handing the viewer a zero-radius target it would divide by.
    if (!(m_bounds.focusRadius > 0.0f))
    {
        m_bounds.focusRadius = m_bounds.radius();
    }
}

void GaussianCloud::uploadToGPU(size_t capacity)
{
    VK_VALIDATE(m_count > 0, "GaussianCloud::uploadToGPU called with no Gaussians loaded");
    // The core streams are copied with memcpy, so a host vector shorter than
    // count() would be read past its end.
    VK_VALIDATE(m_positions.size() == m_count && m_scales.size() == m_count
                    && m_rotations.size() == m_count && m_opacities.size() == m_count
                    && m_shCoefficients.size() == m_count * kShCoefficientsPerGaussian,
                "GaussianCloud::uploadToGPU: host streams are not sized for count() Gaussians");

    m_capacity = std::max(m_count, capacity);
    const VkDeviceSize cap = static_cast<VkDeviceSize>(m_capacity);
    const auto uploadStream = [this](std::unique_ptr<Buffer> & target, VkDeviceSize bufferBytes,
                                     const void * data, size_t byteCount)
    {
        target = std::make_unique<Buffer>(m_device, bufferBytes);
        target->upload(data, byteCount);
    };

    uploadStream(m_positionBuffer, cap * kMeansStride, m_positions.data(),
                 m_count * sizeof(glm::vec4));
    uploadStream(m_scaleBuffer, cap * kScalesStride, m_scales.data(), m_count * sizeof(glm::vec4));
    uploadStream(m_rotationBuffer, cap * kRotationsStride, m_rotations.data(),
                 m_count * sizeof(glm::vec4));
    uploadStream(m_colorBuffer, cap * kShCoefficientsPerGaussian * kShStride,
                 m_shCoefficients.data(),
                 m_count * kShCoefficientsPerGaussian * sizeof(glm::vec4));
    uploadStream(m_opacityBuffer, cap * kOpacityStride, m_opacities.data(),
                 m_count * sizeof(float));

    // Each relightable stream is uploaded if present (sized
    // for the count), dropped otherwise.
    const auto optionalStream = [&](std::unique_ptr<Buffer> & target, size_t hostElements,
                                    size_t elementsPerGaussian, VkDeviceSize bytesPerGaussian,
                                    const void * data, const char * name)
    {
        target.reset();
        if (hostElements == 0)
        {
            return;
        }
        VK_VALIDATE(hostElements == m_count * elementsPerGaussian,
                    std::string("GaussianCloud::uploadToGPU: relightable stream '") + name
                        + "' is not sized for count() Gaussians");
        uploadStream(target, cap * bytesPerGaussian, data, m_count * bytesPerGaussian);
    };
    optionalStream(m_normalBuffer, m_normals.size(), 1, kNormalsStride, m_normals.data(), "normals");
    optionalStream(m_baseColorRoughnessBuffer, m_baseColorRoughness.size(), 1,
                   kBaseColorRoughnessStride, m_baseColorRoughness.data(), "baseColorRoughness");
    optionalStream(m_metallicBuffer, m_metallic.size(), 1, kMetallicStride, m_metallic.data(),
                   "metallic");
    optionalStream(m_visibilityBuffer, m_visibility.size(), kVisibilityCoefficients,
                   kVisibilityCoefficients * kVisibilityStride, m_visibility.data(), "visibility");
    bindOwnedParameters();
    m_borrowedParameters = false;

    // Reallocated on the next backward()/gradientViews().
    dropGradients();
    ++m_storageGeneration;
}

void GaussianCloud::borrow(const CanonicalGaussianBufferViews & parameters,
                           const CanonicalGaussianGradientViews & gradients,
                           const RelightBufferViews & relight, size_t capacity)
{
    VK_VALIDATE(parameters.gaussianCount > 0 && parameters.gaussianCount <= capacity,
                "GaussianCloud::borrow: gaussianCount must be in [1, capacity]");

    // Nothing of the old scene survives: no owned parameter buffers, no
    // host copy (bounds stay invalid -- there is nothing on the host to
    // frame), and gradients either borrowed or re-allocated on demand.
    m_positionBuffer.reset();
    m_scaleBuffer.reset();
    m_rotationBuffer.reset();
    m_colorBuffer.reset();
    m_opacityBuffer.reset();
    m_normalBuffer.reset();
    m_baseColorRoughnessBuffer.reset();
    m_metallicBuffer.reset();
    m_visibilityBuffer.reset();
    m_normals.clear();
    m_baseColorRoughness.clear();
    m_metallic.clear();
    m_visibility.clear();
    m_positions.clear();
    m_scales.clear();
    m_rotations.clear();
    m_shCoefficients.clear();
    m_opacities.clear();
    m_times.clear();
    m_bounds = SceneBounds{};
    dropGradients();

    m_parameters = parameters;
    m_relight = relight;
    m_borrowedParameters = true;
    m_count = parameters.gaussianCount;
    m_capacity = capacity;
    m_shDegree = parameters.shDegree;
    if (gradients.means.valid())
    {
        m_gradients = gradients;
        m_borrowedGradients = true;
    }
    ++m_storageGeneration;
}

void GaussianCloud::bindOwnedParameters()
{
    m_parameters = CanonicalGaussianBufferViews{};
    m_parameters.means = m_positionBuffer->view(kMeansStride);
    m_parameters.scales = m_scaleBuffer->view(kScalesStride);
    m_parameters.rotations = m_rotationBuffer->view(kRotationsStride);
    m_parameters.opacity = m_opacityBuffer->view(kOpacityStride);
    m_parameters.shCoefficients = m_colorBuffer->view(kShStride);
    // confidence stays default-constructed (invalid) -- absent, per contract.

    m_relight = RelightBufferViews{};
    if (m_normalBuffer)
    {
        m_relight.normals = m_normalBuffer->view(kNormalsStride);
    }
    if (m_baseColorRoughnessBuffer)
    {
        m_relight.baseColorRoughness = m_baseColorRoughnessBuffer->view(kBaseColorRoughnessStride);
    }
    if (m_metallicBuffer)
    {
        m_relight.metallic = m_metallicBuffer->view(kMetallicStride);
    }
    if (m_visibilityBuffer)
    {
        m_relight.visibility = m_visibilityBuffer->view(kVisibilityStride);
    }
}

void GaussianCloud::dropGradients()
{
    m_gradMeansBuffer.reset();
    m_gradScalesBuffer.reset();
    m_gradRotationsBuffer.reset();
    m_gradOpacityBuffer.reset();
    m_gradShBuffer.reset();
    m_gradScreenMeansBuffer.reset();
    m_gradients = CanonicalGaussianGradientViews{};
    m_borrowedGradients = false;
}

bool GaussianCloud::reserve(size_t capacity)
{
    VK_VALIDATE(!m_borrowedParameters,
                "GaussianCloud::reserve: the parameter storage is borrowed (ownership model B) -- "
                "its owner grows it: allocate larger buffers, copy, and borrow them again");
    VK_VALIDATE(m_positionBuffer != nullptr, "GaussianCloud::reserve() called before uploadToGPU()");
    if (capacity <= m_capacity)
    {
        return false;
    }

    const VkDeviceSize cap = static_cast<VkDeviceSize>(capacity);
    const VkDeviceSize n = static_cast<VkDeviceSize>(m_count);
    struct Stream
    {
        std::unique_ptr<Buffer> & buffer;
        VkDeviceSize bytesPerGaussian;
    };
    Stream streams[] = {
        {m_positionBuffer, kMeansStride},
        {m_scaleBuffer, kScalesStride},
        {m_rotationBuffer, kRotationsStride},
        {m_colorBuffer, kShCoefficientsPerGaussian * kShStride},
        {m_opacityBuffer, kOpacityStride},
        // The relightable streams, when present.
        {m_normalBuffer, kNormalsStride},
        {m_baseColorRoughnessBuffer, kBaseColorRoughnessStride},
        {m_metallicBuffer, kMetallicStride},
        {m_visibilityBuffer, kVisibilityCoefficients * kVisibilityStride},
    };
    constexpr size_t kStreams = sizeof(streams) / sizeof(streams[0]);

    std::unique_ptr<Buffer> grown[kStreams];
    for (size_t i = 0; i < kStreams; ++i)
    {
        if (streams[i].buffer)
        {
            grown[i] = std::make_unique<Buffer>(m_device, cap * streams[i].bytesPerGaussian);
        }
    }
    // One submission for every prefix copy. immediateSubmit runs on the
    // shared queue after anything a consumer submitted there earlier and
    // waits for it, so the old buffers are idle when they are freed below.
    m_device.immediateSubmit(
        [&](VkCommandBuffer cmd)
        {
            for (size_t i = 0; i < kStreams; ++i)
            {
                if (!streams[i].buffer)
                {
                    continue;
                }
                VkBufferCopy region{};
                region.size = n * streams[i].bytesPerGaussian;
                vkCmdCopyBuffer(cmd, streams[i].buffer->handle(), grown[i]->handle(), 1, &region);
            }
        });
    for (size_t i = 0; i < kStreams; ++i)
    {
        if (streams[i].buffer)
        {
            streams[i].buffer = std::move(grown[i]);
        }
    }
    m_capacity = capacity;
    bindOwnedParameters();

    // Gradients carry nothing across a backward (it overwrites them), so
    // they are reallocated at the new capacity rather than copied -- and
    // eagerly, if they existed, so a consumer re-fetching views after this
    // reserve gets both sets at once.
    const bool hadGradients = hasGradientBuffers();
    dropGradients();
    ++m_storageGeneration;
    if (hadGradients)
    {
        ensureGradientBuffers();
    }
    return true;
}

void GaussianCloud::setCount(size_t count)
{
    VK_VALIDATE(m_parameters.means.valid(),
                "GaussianCloud::setCount() called before uploadToGPU()/borrow()");
    VK_VALIDATE(count > 0 && count <= m_capacity,
                "GaussianCloud::setCount: count must be in [1, capacity()] -- reserve() first");
    m_count = count;
}

void GaussianCloud::ensureGradientBuffers()
{
    VK_VALIDATE(m_parameters.means.valid(),
                "GaussianCloud::ensureGradientBuffers() called before uploadToGPU()/borrow()");
    if (hasGradientBuffers())
    {
        return;
    }
    // Sized to the capacity, like the parameters, so setCount() never
    // invalidates them either.
    const VkDeviceSize n = static_cast<VkDeviceSize>(m_capacity);
    m_gradMeansBuffer = std::make_unique<Buffer>(m_device, n * kMeansStride);
    m_gradScalesBuffer = std::make_unique<Buffer>(m_device, n * kScalesStride);
    m_gradRotationsBuffer = std::make_unique<Buffer>(m_device, n * kRotationsStride);
    m_gradOpacityBuffer = std::make_unique<Buffer>(m_device, n * kOpacityStride);
    m_gradShBuffer =
        std::make_unique<Buffer>(m_device, n * kShCoefficientsPerGaussian * kShStride);
    m_gradScreenMeansBuffer = std::make_unique<Buffer>(m_device, n * kMeansStride);
    m_gradients.means = m_gradMeansBuffer->view(kMeansStride);
    m_gradients.scales = m_gradScalesBuffer->view(kScalesStride);
    m_gradients.rotations = m_gradRotationsBuffer->view(kRotationsStride);
    m_gradients.opacity = m_gradOpacityBuffer->view(kOpacityStride);
    m_gradients.shCoefficients = m_gradShBuffer->view(kShStride);
    m_gradients.screenMeans = m_gradScreenMeansBuffer->view(kMeansStride);
    ++m_storageGeneration;
}

GaussianGradients GaussianCloud::downloadGradients() const
{
    VK_VALIDATE(hasGradientBuffers(),
                "GaussianCloud::downloadGradients() called before ensureGradientBuffers()");
    GaussianGradients g;
    g.means.resize(m_count);
    g.scales.resize(m_count);
    g.rotations.resize(m_count);
    g.opacities.resize(m_count);
    g.shCoefficients.resize(m_count * kShCoefficientsPerGaussian);
    g.screenMeans.resize(m_count);
    downloadView(m_device, m_gradients.means, g.means.data(), g.means.size() * sizeof(glm::vec4));
    downloadView(m_device, m_gradients.scales, g.scales.data(), g.scales.size() * sizeof(glm::vec4));
    downloadView(m_device, m_gradients.rotations, g.rotations.data(),
                 g.rotations.size() * sizeof(glm::vec4));
    downloadView(m_device, m_gradients.opacity, g.opacities.data(),
                 g.opacities.size() * sizeof(float));
    downloadView(m_device, m_gradients.shCoefficients, g.shCoefficients.data(),
                 g.shCoefficients.size() * sizeof(glm::vec4));
    downloadView(m_device, m_gradients.screenMeans, g.screenMeans.data(),
                 g.screenMeans.size() * sizeof(glm::vec4));
    return g;
}

void GaussianCloud::downloadFromGPU()
{
    VK_VALIDATE(m_parameters.means.valid(),
                "GaussianCloud::downloadFromGPU() called before uploadToGPU()/borrow()");
    // setCount() leaves the host vectors alone; match them to the GPU count.
    m_positions.resize(m_count);
    m_scales.resize(m_count);
    m_rotations.resize(m_count);
    m_shCoefficients.resize(m_count * kShCoefficientsPerGaussian);
    m_opacities.resize(m_count);
    m_times.resize(m_count, 0.0f);
    downloadView(m_device, m_parameters.means, m_positions.data(),
                 m_positions.size() * sizeof(glm::vec4));
    downloadView(m_device, m_parameters.scales, m_scales.data(), m_scales.size() * sizeof(glm::vec4));
    downloadView(m_device, m_parameters.rotations, m_rotations.data(),
                 m_rotations.size() * sizeof(glm::vec4));
    downloadView(m_device, m_parameters.shCoefficients, m_shCoefficients.data(),
                 m_shCoefficients.size() * sizeof(glm::vec4));
    downloadView(m_device, m_parameters.opacity, m_opacities.data(),
                 m_opacities.size() * sizeof(float));
    // A baked visibility stream lives only on the GPU until here.
    if (m_relight.visibility.valid())
    {
        m_visibility.resize(m_count * kVisibilityCoefficients);
        downloadView(m_device, m_relight.visibility, m_visibility.data(),
                     m_visibility.size() * sizeof(float));
    }
}

const BufferView & GaussianCloud::ensureVisibilityStream()
{
    VK_VALIDATE(m_parameters.means.valid(),
                "GaussianCloud::ensureVisibilityStream() called before uploadToGPU()/borrow()");
    if (m_relight.visibility.valid())
    {
        return m_relight.visibility;
    }
    VK_VALIDATE(!m_borrowedParameters,
                "GaussianCloud::ensureVisibilityStream: the scene is borrowed and has no visibility "
                "stream -- borrow one (RelightBufferViews::visibility) to bake into");
    m_visibilityBuffer = std::make_unique<Buffer>(
        m_device, static_cast<VkDeviceSize>(m_capacity) * kVisibilityCoefficients * kVisibilityStride);
    m_relight.visibility = m_visibilityBuffer->view(kVisibilityStride);
    ++m_storageGeneration;
    return m_relight.visibility;
}

CanonicalGaussianGradientViews GaussianCloud::gradientViews() const
{
    VK_VALIDATE(hasGradientBuffers(),
                "GaussianCloud::gradientViews() called before ensureGradientBuffers()");

    CanonicalGaussianGradientViews v = m_gradients;
    v.gaussianCount = static_cast<uint32_t>(m_count);
    return v;
}

CanonicalGaussianBufferViews GaussianCloud::views() const
{
    VK_VALIDATE(m_parameters.means.valid(),
                "GaussianCloud::views() called before uploadToGPU()/borrow()");

    CanonicalGaussianBufferViews v = m_parameters;
    v.gaussianCount = static_cast<uint32_t>(m_count);
    v.shDegree = m_shDegree; // discovered by the loader (see setShDegree)
    return v;
}

} // namespace sabercat
