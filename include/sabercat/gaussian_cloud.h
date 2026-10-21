// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <memory>
#include <vector>

#include "sabercat/buffer_view.h"
#include "sabercat/render_config.h"
#include "sabercat/scene_bounds.h"

namespace sabercat
{

class Device;
class Buffer;

// Host copy of the gradient streams, same layout as the GPU buffers
// (see CanonicalGaussianGradientViews): vec4-padded xyz/xyzw streams, 16 SH
// coefficients per Gaussian, scalar opacity.
struct GaussianGradients
{
    std::vector<glm::vec4> means;
    std::vector<glm::vec4> scales;
    std::vector<glm::vec4> rotations;
    std::vector<float> opacities;
    std::vector<glm::vec4> shCoefficients;
    std::vector<glm::vec4> screenMeans;
};

// Standard 3DGS spherical harmonics storage: 1 DC term + 15 higher-order
// coefficients (SH degree 3) per color channel = 16 vec4 coefficients per
// Gaussian. Defined in shaders/config.slangh so host and shaders agree.
constexpr int kShCoefficientsPerGaussian = static_cast<int>(kShCoefficients);

// The canonical stream strides (kMeansStride, kOpacityStride, ...) come
// from render_config.h, which shares them with the Slang kernels -- the
// shaders encode the same strides implicitly by declaring
// StructuredBuffer<float4>, so both sides must read one definition.

// Struct-of-Arrays representation of a Gaussian cloud (PERFORMANCE.md
// invariant #1): one contiguous array per attribute, both on the CPU
// (parsed staging data) and the GPU (one Buffer per attribute).
//
// Covariance is deliberately NOT stored. It is derived on the GPU from
// scale + rotation in the forward pass: storing it would mean a 3x3 matrix
// per Gaussian (more bandwidth than the 3+4 floats it comes from), and the
// optimizer has to update scale/rotation anyway -- you can't
// gradient-descend a covariance matrix directly and keep it
// positive-semi-definite.
//
// The schema reserves a per-Gaussian time value (invariant #8) so 4D
// (time-varying) Gaussians can slot in later without restructuring; it's
// unused (always 0) for the static 3D scenes loaded today.
class GaussianCloud
{
  public:
    explicit GaussianCloud(Device & device);
    ~GaussianCloud();

    GaussianCloud(const GaussianCloud &) = delete;
    GaussianCloud & operator=(const GaussianCloud &) = delete;

    // Allocates the CPU-side SoA vectors for `count` Gaussians. Called by a
    // loader before it fills them in. GPU storage is untouched until
    // uploadToGPU().
    void initialize(size_t count);

    size_t count() const
    {
        return m_count;
    }

    // CPU-side staging data. Stored padded to the canonical strides above
    // so uploadToGPU() is a straight memcpy with no repacking.
    std::vector<glm::vec4> & positions()
    {
        return m_positions;
    }
    const std::vector<glm::vec4> & positions() const
    {
        return m_positions;
    }
    std::vector<glm::vec4> & scales()
    {
        return m_scales;
    }
    const std::vector<glm::vec4> & scales() const
    {
        return m_scales;
    }
    // Quaternions in canonical xyzw component order, normalized.
    std::vector<glm::vec4> & rotations()
    {
        return m_rotations;
    }
    const std::vector<glm::vec4> & rotations() const
    {
        return m_rotations;
    }
    // Flat, kShCoefficientsPerGaussian entries per Gaussian (RGB in xyz).
    std::vector<glm::vec4> & shCoefficients()
    {
        return m_shCoefficients;
    }
    const std::vector<glm::vec4> & shCoefficients() const
    {
        return m_shCoefficients;
    }
    std::vector<float> & opacities()
    {
        return m_opacities;
    }
    const std::vector<float> & opacities() const
    {
        return m_opacities;
    }
    std::vector<float> & times()
    {
        return m_times;
    }
    const std::vector<float> & times() const
    {
        return m_times;
    }

    // Optional relightable streams, host side (see docs/architecture.md, "Relightable
    // splats"). Each is either EMPTY (absent: the relight pass
    // derives a default) or sized for count() Gaussians -- normals and
    // baseColorRoughness one vec4 each, metallic one float,
    // visibility kVisibilityCoefficients floats. Loaders fill them (R3DG
    // .ply); uploadToGPU() uploads whichever are present.
    std::vector<glm::vec4> & normals()
    {
        return m_normals;
    }
    const std::vector<glm::vec4> & normals() const
    {
        return m_normals;
    }
    std::vector<glm::vec4> & baseColorRoughness()
    {
        return m_baseColorRoughness;
    }
    const std::vector<glm::vec4> & baseColorRoughness() const
    {
        return m_baseColorRoughness;
    }
    std::vector<float> & metallic()
    {
        return m_metallic;
    }
    const std::vector<float> & metallic() const
    {
        return m_metallic;
    }
    std::vector<float> & visibility()
    {
        return m_visibility;
    }
    const std::vector<float> & visibility() const
    {
        return m_visibility;
    }

    // Records the maximum spherical-harmonic degree in the loaded cloud.
    // Different formats may have different degree ceilings (e.g., .splat is
    // DC-only, so degree 0; .ply can be 0-3). This is discovered during load
    // and reported via views()'s shDegree field so the viewer can clamp its
    // SH slider and log what was loaded.
    void setShDegree(uint32_t degree)
    {
        m_shDegree = degree;
    }
    uint32_t shDegree() const
    {
        return m_shDegree;
    }

    // Computes the world-space bounds of the loaded positions. Called by each
    // loader once it has filled the SoA vectors -- same place and for the same
    // reason as setShDegree(): it is information the loader has and every
    // consumer otherwise has to rediscover. Cheap next to the parse it follows
    // (a linear min/max over ~1M vec4s costs a couple of ms against a ~65ms
    // parse), so it is a plain serial pass rather than a parallel reduction.
    void computeBounds();

    // Valid only after a loader has called computeBounds(); `valid` is false
    // otherwise, and on an empty cloud.
    const SceneBounds & bounds() const
    {
        return m_bounds;
    }

    // Whether this scene was trained with the dilation opacity
    // compensation (mip-splatting / gsplat "antialiased"). A property of the
    // scene's training, so loaders set it: .spz reads its header flag; .ply
    // and .splat cannot carry one and default to false, because vanilla 3DGS
    // -- which is what produces them -- trains without it. Renderer can
    // override it for a mip-splatting .ply that has no way to say so.
    void setAntialiased(bool antialiased)
    {
        m_antialiased = antialiased;
    }
    bool antialiased() const
    {
        return m_antialiased;
    }

    // Uploads the CPU-side SoA data to one GPU Buffer per attribute, sized
    // for max(count(), capacity) Gaussians. Recreates the buffers (new
    // handles, so storageGeneration() advances) and drops any gradient
    // buffers.
    void uploadToGPU(size_t capacity = 0);

    // Resizable storage for densification -- std::vector semantics
    // over the GPU streams, never touching the host.
    //
    // capacity() is how many Gaussians the GPU buffers can hold; count() is
    // how many forward/backward process. Views always cover the whole
    // capacity (size = capacity * stride) and report count() as
    // gaussianCount, so a trainer can write new Gaussians into the tail
    // before raising the count. Elements in [count, capacity) are undefined
    // until written.
    size_t capacity() const
    {
        return m_capacity;
    }

    // Grows the parameter buffers to hold `capacity` Gaussians, GPU-copying
    // the first count() of each stream (one synchronous submission). The
    // ONLY call besides uploadToGPU() that replaces buffer handles: every
    // parameter and gradient view taken before it is stale afterwards.
    // Existing gradient buffers are reallocated at the new capacity with
    // undefined contents (backward() overwrites them anyway). Returns false,
    // changing nothing, if `capacity` <= capacity().
    bool reserve(size_t capacity);

    // Sets how many Gaussians forward/backward process: 1 <= count <=
    // capacity(). No allocation, no copy, no GPU work -- every handle stays
    // valid. The host-side vectors are NOT resized here; downloadFromGPU()
    // resizes them to the new count when it next runs.
    void setCount(size_t count);

    // Advances whenever buffer handles are replaced (uploadToGPU, a growing
    // reserve, or gradient buffers being (re)allocated) -- never on
    // setCount(). A consumer that cached views or descriptor sets compares
    // this to know when to re-fetch them.
    uint64_t storageGeneration() const
    {
        return m_storageGeneration;
    }

    // Renders application-owned parameter
    // streams instead of uploading host data. `parameters` must already have
    // passed validateBorrowedGaussians(), which also yields `capacity`;
    // gaussianCount/shDegree come from it. Gradient views are borrowed too
    // when all six are valid, otherwise allocated lazily as usual. Drops the
    // host vectors and bounds (there is no host copy) and any owned buffers,
    // and advances storageGeneration(). reserve() refuses on borrowed
    // storage: the owner grows it and calls borrow() again.
    void borrow(const CanonicalGaussianBufferViews & parameters,
                const CanonicalGaussianGradientViews & gradients,
                const RelightBufferViews & relight, size_t capacity);
    bool borrowedParameters() const
    {
        return m_borrowedParameters;
    }
    bool borrowedGradients() const
    {
        return m_borrowedGradients;
    }

    // Allocates the gradient buffers (see
    // CanonicalGaussianGradientViews), sized to capacity(), if they don't
    // exist yet. Lazy so a viewer that never calls backward() never pays for
    // them -- they cost slightly more than the parameters themselves
    // (324 vs 308 B per Gaussian). A no-op when the gradients are borrowed.
    void ensureGradientBuffers();
    bool hasGradientBuffers() const
    {
        return m_gradients.means.valid();
    }
    // Only valid after ensureGradientBuffers().
    CanonicalGaussianGradientViews gradientViews() const;

    // Synchronous readbacks (downloadView -> Device::immediateSubmit):
    // for tests, debugging and the C API's host-copy path -- not a training
    // loop's hot path, which should bind the views instead.
    GaussianGradients downloadGradients() const;
    // Refreshes the CPU-side vectors from the GPU buffers, picking up any
    // on-GPU edits (e.g. an optimizer step through views()).
    void downloadFromGPU();

    // The OWNED buffers backing views(); null when the parameters are
    // borrowed (application-owned). Verification-only (gaussian_cloud_test).
    Buffer * positionBuffer() const
    {
        return m_positionBuffer.get();
    }
    Buffer * scaleBuffer() const
    {
        return m_scaleBuffer.get();
    }
    Buffer * rotationBuffer() const
    {
        return m_rotationBuffer.get();
    }
    Buffer * colorBuffer() const
    {
        return m_colorBuffer.get();
    }
    Buffer * opacityBuffer() const
    {
        return m_opacityBuffer.get();
    }

    // Non-owning views of the uploaded streams, for binding into compute
    // passes (the buffer seam -- passes take views, not owned Buffers).
    // Only valid after uploadToGPU().
    CanonicalGaussianBufferViews views() const;
    // The relightable streams on the GPU, owned or borrowed;
    // each view invalid when that stream is absent. Spans the capacity.
    const RelightBufferViews & relightViews() const
    {
        return m_relight;
    }
    // The GPU visibility stream a bake writes, created if
    // absent -- owned storage gets one sized to capacity() (contents
    // undefined until written; the host vector stays empty until
    // downloadFromGPU()). Borrowed storage must already carry a borrowed
    // visibility view: Sabercat never allocates into a borrowed scene.
    // Advances storageGeneration() when it allocates.
    const BufferView & ensureVisibilityStream();

  private:
    // Points m_parameters (and m_relight) at the owned buffers (after
    // upload/reserve).
    void bindOwnedParameters();
    // Releases owned gradient buffers and forgets any borrowed ones.
    void dropGradients();

    Device & m_device;
    size_t m_count = 0;
    size_t m_capacity = 0; // Gaussians the GPU buffers hold; >= m_count
    uint64_t m_storageGeneration = 0;
    // Max SH degree present in the loaded data. Defaults to 3 (full quality)
    // for PLY files; overridden by loaders that discover a lower ceiling.
    uint32_t m_shDegree = 3;
    bool m_antialiased = false;
    bool m_borrowedParameters = false; // m_parameters are the app's
    bool m_borrowedGradients = false;  // m_gradients are the app's
    SceneBounds m_bounds;

    // What every consumer binds. Backed by the owned buffers below, or by
    // the application's (borrow()); gaussianCount/shDegree are filled in on
    // the way out.
    CanonicalGaussianBufferViews m_parameters{};
    CanonicalGaussianGradientViews m_gradients{};
    RelightBufferViews m_relight{};

    std::vector<glm::vec4> m_positions;
    std::vector<glm::vec4> m_scales;
    std::vector<glm::vec4> m_rotations;
    std::vector<glm::vec4> m_shCoefficients;
    std::vector<float> m_opacities;
    std::vector<float> m_times;
    std::vector<glm::vec4> m_normals;
    std::vector<glm::vec4> m_baseColorRoughness;
    std::vector<float> m_metallic;
    std::vector<float> m_visibility;

    std::unique_ptr<Buffer> m_positionBuffer;
    std::unique_ptr<Buffer> m_scaleBuffer;
    std::unique_ptr<Buffer> m_rotationBuffer;
    std::unique_ptr<Buffer> m_colorBuffer;
    std::unique_ptr<Buffer> m_opacityBuffer;
    // Owned relightable streams (null when absent or borrowed).
    std::unique_ptr<Buffer> m_normalBuffer;
    std::unique_ptr<Buffer> m_baseColorRoughnessBuffer;
    std::unique_ptr<Buffer> m_metallicBuffer;
    std::unique_ptr<Buffer> m_visibilityBuffer;

    std::unique_ptr<Buffer> m_gradMeansBuffer;
    std::unique_ptr<Buffer> m_gradScalesBuffer;
    std::unique_ptr<Buffer> m_gradRotationsBuffer;
    std::unique_ptr<Buffer> m_gradOpacityBuffer;
    std::unique_ptr<Buffer> m_gradShBuffer;
    std::unique_ptr<Buffer> m_gradScreenMeansBuffer;
};

} // namespace sabercat
