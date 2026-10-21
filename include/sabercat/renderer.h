// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vulkan/vulkan.h>

#include <glm/glm.hpp>

#include "sabercat/buffer_view.h"
#include "sabercat/lighting.h"
#include "sabercat/scene_bounds.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace sabercat
{

struct Camera;
struct LODParams;
struct GaussianGradients;
struct DeviceFeatures;
struct VulkanContext;
struct EnabledDeviceFeatures;
struct BorrowedGaussians;
class Device;

// What forward() renders for. Viewer honors the viewer-only knobs
// (setLODParams: geometry/appearance LOD, minScreenRadius); Differentiable
// pins them all off and saves the per-pixel state backward() needs. backward()
// only accepts a Differentiable forward. See ForwardMode for the mechanism.
enum class RenderMode
{
    Viewer,
    Differentiable,
};

// Host-side Gaussian arrays at the canonical strides (see docs/formats.md): means,
// scales, rotations (xyzw) as 4 floats per Gaussian (w/pad lane ignored on
// input), opacities 1 float, SH 16 coefficients x 4 floats (RGB + pad),
// band-ascending. Values are the ACTIVATED ones: linear scale, opacity in
// [0,1], normalized quaternion, raw SH.
struct GaussianHostInput
{
    uint32_t count = 0;
    const float * means = nullptr;
    const float * scales = nullptr;
    const float * rotations = nullptr;
    const float * opacities = nullptr;
    const float * shCoefficients = nullptr;
    uint32_t shDegree = 3;
    bool antialiased = false;
    // GPU storage to reserve, in Gaussians; max(count, capacity) is
    // allocated. 0 = exactly `count`. See reserveGaussians().
    uint32_t capacity = 0;
    // Optional relightable streams (null = absent; the relight
    // pass derives a default). normals and baseColorRoughness 4 floats per
    // Gaussian (unit normal + pad; linear rgb + roughness), metallic 1,
    // visibility kVisibilityCoefficients (mono SH, degree 3).
    const float * normals = nullptr;
    const float * baseColorRoughness = nullptr;
    const float * metallic = nullptr;
    const float * visibility = nullptr;
};

// Same layout, for readback. Every pointer must hold `count` Gaussians'
// worth of floats.
struct GaussianHostOutput
{
    uint32_t count = 0;
    float * means = nullptr;
    float * scales = nullptr;
    float * rotations = nullptr;
    float * opacities = nullptr;
    float * shCoefficients = nullptr;
    // Optional: kVisibilityCoefficients floats per Gaussian
    // (e.g. after bakeVisibility); throws if set and the scene has none.
    float * visibility = nullptr;
};

// What present() shows on screen. Debug channels read buffers ForwardPass
// already produces (or, for TileDensity, derives cheaply from tileRanges)
// -- none of them affect what forward() computes.
enum class DebugView
{
    Final,       // the rasterized color image (default)
    Alpha,       // coverage channel, grayscale
    Depth,       // frontmost-contributor depth, grayscale
    TileDensity, // instances-per-tile heatmap
    // Effective SH degree per pixel -- the blend-weighted mean of
    // the contributing Gaussians' (band-faded, so fractional) degrees, as a
    // blue(0) -> green -> yellow -> red(3) heatmap scaled by coverage.
    SHDegree,
};

// Frames-in-flight count (PERFORMANCE.md invariant #3): command buffers,
// per-frame descriptor sets, and sync objects are all duplicated this many
// times so the CPU can record frame N+1 while the GPU still works on frame
// N. Shared here rather than kept as a renderer.cpp-local constant because
// DebugVisualizer's own per-frame descriptor set/display image must use the
// exact same count to stay in lockstep with Renderer's frame index -- a
// mismatch would silently reopen the same "descriptor set in use by a
// pending command buffer" race N-buffering this constant exists to close.
constexpr uint32_t kMaxFramesInFlight = 2;

// Differentiable 3D Gaussian Splatting renderer -- the reusable rendering core.
//
// It exposes a FORWARD pass (rasterize Gaussians into an image, for real-time
// viewing) and a BACKWARD pass (given the gradient of a loss w.r.t. the output
// image, accumulate gradients w.r.t. the Gaussian parameters). The optimizer,
// loss function, and training loop live in a SEPARATE trainer layer built on
// top of this class -- so a pure viewer app calls forward() only and links no
// training code.
//
// The renderer is windowing-agnostic: it depends on Vulkan only. Applications
// create their own window + surface (from instance()) and hand the surface
// back via attachSurface(). Initial Gaussian clouds are produced by a separate
// generation project and loaded here.
class Renderer
{
  public:
    Renderer(uint32_t width, uint32_t height);
    ~Renderer();

    Renderer(const Renderer &) = delete;
    Renderer & operator=(const Renderer &) = delete;

    // Bring up the Vulkan core. Apps that will present pass the instance
    // extensions their windowing layer requires (e.g. from GLFW); leave empty
    // for headless/offscreen use. shaderDir must contain the compiled
    // forward-rasterizer *.comp.spv (see shaders/CMakeLists.txt's
    // sabercat_shaders target) -- leave empty to skip building the forward
    // pass (forward()/present() become no-ops), for tests that don't render.
    bool initialize(const std::vector<const char *> & instanceExtensions = {},
                    bool enableValidation = true, const std::string & shaderDir = "");

    // Bring up on a device the APPLICATION
    // created -- or another Renderer's (vulkanContext() +
    // enabledDeviceFeatures()), e.g. a second image size over the same
    // scene. `enabled` declares what the owner enabled at vkCreateDevice
    // (DeviceRequirements builds both); every handle and declaration is
    // checked first (validateBorrowedContext) and a failure throws
    // std::invalid_argument naming it. Sabercat creates its own allocator,
    // pools and pipelines on the device and destroys only those; the owner
    // keeps the context alive until this renderer's shutdown(). A
    // compute-only context is enough headless; presenting needs the
    // graphics queue and a declared swapchain. Unlike initialize(), throws
    // instead of returning false.
    void initializeBorrowed(const VulkanContext & context, const EnabledDeviceFeatures & enabled,
                            const std::string & shaderDir);
    void shutdown();

    // This renderer's device, in the form initializeBorrowed()
    // takes. Valid after initialize()/initializeBorrowed().
    const VulkanContext & vulkanContext() const;
    const EnabledDeviceFeatures & enabledDeviceFeatures() const;
    // False when the device is borrowed (initializeBorrowed).
    bool ownsDevice() const;
    // The Sabercat Device this renderer runs on, for the host-side calls that
    // take one (loadScene, mergeLayers, ProjectIO::load, the writers' readers).
    // Valid after initialize()/initializeBorrowed(); throws std::runtime_error
    // otherwise. Borrowers must not shut it down -- shutdown() does that.
    Device & sabercatDevice();

    // Vulkan instance -- the application creates its window surface from this.
    VkInstance instance() const;

    // Raw Vulkan accessors for apps that need to hang their own rendering
    // off this device/queue -- Dear ImGui's Vulkan backend is the motivating
    // case (the borrowed-context seam, but for a same-process UI library
    // rather than another Vulkan library). Valid after initialize();
    // swapchainFormat() is only meaningful after attachSurface().
    VkPhysicalDevice physicalDevice() const;
    VkDevice device() const;
    VkQueue graphicsQueue() const;
    uint32_t graphicsQueueFamily() const;
    VkFormat swapchainFormat() const;

    // The queue every Sabercat submission goes to except present()'s
    // (the compute queue), for a consumer borrowing this device -- e.g.
    // the borrowed context's compute_queue. Submit work
    // that touches Sabercat's buffers HERE and no barriers or waits are
    // needed around Sabercat calls; see docs/library.md, synchronization
    // contract. The queue is externally synchronized: never submit to it
    // from another thread while a Renderer call is running.
    VkQueue computeQueue() const;
    uint32_t computeQueueFamily() const;
    // What the device was created with (fp32 atomics, subgroup size, ...).
    // A borrower cannot enable more features on a device it didn't create.
    const DeviceFeatures & deviceFeatures() const;

    // Lets an app draw its own UI (e.g. Dear ImGui) into the swapchain image
    // after present()'s blit, before the final present transition.
    // `callback` receives an already-recording command buffer and the
    // swapchain image already transitioned to
    // VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL inside an active
    // vkCmdBeginRendering scope (LOAD_OP_LOAD, so it draws over the existing
    // content) -- it should record draw commands only, not its own
    // begin/end-rendering or layout transitions. Pass an empty
    // std::function to remove the hook.
    void setUiRenderCallback(std::function<void(VkCommandBuffer)> callback);

    // Hand the renderer an app-created surface to present into; builds the
    // swapchain. The app retains ownership of the surface and must destroy
    // it itself -- but only once `onSurfaceRelease` fires. That happens
    // during shutdown(), after the swapchain built from the surface is torn
    // down but before the renderer's VkInstance is destroyed (surfaces must
    // die before the instance that created them).
    void attachSurface(VkSurfaceKHR surface, uint32_t width, uint32_t height,
                       std::function<void()> onSurfaceRelease);

    // Load a Gaussian cloud from a file in any supported format (.ply, .splat,
    // .spz, .ksplat, .sog or a SOG meta.json, .gltf/.glb with
    // KHR_gaussian_splatting, or a sabercat-cloud-v1 directory /
    // manifest.json).
    // Throws std::runtime_error if the format is unsupported or the file is invalid.
    void loadScene(const char * path);

    // Camera used by subsequent forward()/backward() passes.
    void setCamera(const Camera & camera);

    // What present() shows. Purely a presentation choice -- switching it
    // doesn't re-run forward().
    void setDebugView(DebugView view);

    // Corrects the common COLMAP/INRIA 3DGS authoring convention (Y-down)
    // some pretrained/exported .ply scenes use, which otherwise renders
    // upside down in this renderer's Y-up convention (see docs/architecture.md). Folds a
    // fixed Y negation into the world->camera transform. Deliberately NOT
    // paired with a Z negation (the textbook COLMAP-to-OpenGL 180-degree-
    // around-X fix): this renderer's depth is `-local.z`, so negating Z
    // would flip every point's depth sign and cull the whole scene as
    // behind the camera. A lone Y negation is a reflection rather than a
    // proper rotation (inverts handedness), which doesn't matter for
    // anything this renderer currently computes (degree-0 SH, a covariance
    // congruence transform that's valid under any invertible matrix) --
    // revisit if a future step (higher-order SH, backward-pass rotation
    // gradients) turns out to care about chirality. Toggleable at runtime;
    // doesn't touch the loaded Gaussian data or require a reload.
    void setFlipUpAxis(bool flip);

    // Dynamic LOD tuning knobs for the viewer only. `enabled` in
    // LODParams defaults to false/0 -- forward() never turns this on by
    // itself, so a caller that never calls this sees the same
    // every-Gaussian-rendered behavior it always did (and a future
    // differentiable backward pass simply never calls it). A no-op before
    // initialize()'s forward pass is built.
    void setLODParams(const LODParams & params);

    // Max spherical-harmonic degree to evaluate, clamped to
    // [0, 3]. Degree 0 is the flat DC term (no view-dependent color); 3 is
    // full quality and the default, so a caller that never touches this sees
    // the best the loaded scene can produce. Lower degrees both skip
    // arithmetic and read fewer coefficients per Gaussian.
    void setMaxSHDegree(uint32_t degree);

    // Overrides whether the loaded scene is rendered with the
    // dilation opacity compensation (mip-splatting/gsplat "antialiased").
    // Loaders set it from the file (.spz header; .ply/.splat cannot say, so
    // off); this is for a mip-splatting .ply that has no way to declare it.
    // Applies to the current scene only -- a new loadScene() resets it.
    void setAntialiased(bool antialiased);
    bool sceneAntialiased() const;

    // Replaces the scene with host arrays (see GaussianHostInput) --
    // the C ABI's and tests' path to a scene without a file. Throws on null
    // pointers or count 0.
    void setGaussians(const GaussianHostInput & input);
    // Reads the CURRENT GPU parameters back (picking up any edits made
    // through gaussianViews()). out.count must equal gaussianCount().
    void readGaussians(const GaussianHostOutput & out);

    // Replaces the scene with application-owned
    // GPU streams, rendered and differentiated in place (no copy). Checked
    // by validateBorrowedGaussians() first -- a failure throws
    // std::invalid_argument naming the stream and leaves the current scene
    // untouched. Gradients are borrowed too when all six views are given,
    // otherwise allocated by Sabercat as usual. Capacity is what the
    // buffers hold: setGaussianCount() moves within it, reserveGaussians()
    // throws (the owner grows its buffers and borrows again). Advances
    // storageGeneration(); gaussianViews()/gradientViews() return the
    // borrowed views. The buffers must outlive this scene (until the next
    // scene replacement or shutdown()). Requirements: see docs/library.md.
    void borrowGaussians(const BorrowedGaussians & scene);
    // Whether the current scene's parameter streams are borrowed.
    bool gaussiansBorrowed() const;

    // --- Resizable storage (densification) ---
    //
    // std::vector semantics over the GPU streams. gaussianCapacity() is how
    // many Gaussians the parameter (and gradient) buffers hold;
    // gaussianCount() is how many forward/backward process. Views always
    // span the full capacity, so a trainer writes new Gaussians into
    // [count, newCount) through gaussianViews() and then raises the count.

    // Grows storage to `capacity` Gaussians, GPU-copying the first
    // gaussianCount() of every parameter stream. The ONLY call (besides
    // loadScene/setGaussians) that replaces buffer handles: views and
    // descriptor sets built from earlier views are stale afterwards, and
    // storageGeneration() advances. Gradient buffers are reallocated with
    // undefined contents. Returns false, changing nothing, if `capacity` <=
    // gaussianCapacity(). Throws without a scene.
    bool reserveGaussians(uint32_t capacity);
    // Sets how many Gaussians forward/backward process, 1 <= count <=
    // gaussianCapacity(). No allocation, no copy, no GPU work: every handle
    // stays valid. Gaussians in [old count, count) are whatever the trainer
    // wrote there (undefined if it wrote nothing). Invalidates the last
    // differentiable forward -- backward() needs a fresh forward() first.
    void setGaussianCount(uint32_t count);
    uint32_t gaussianCapacity() const;

    // Changes the output image size of a HEADLESS renderer (with a
    // surface attached, the swapchain decides the size). Reallocates the
    // image and per-pixel state, so imageView()'s handle changes and the
    // last forward is invalidated: forward() again before readImage(),
    // imageView() or backward(). Camera intrinsics must match the new size.
    // A no-op if the size is unchanged.
    void setImageSize(uint32_t width, uint32_t height);
    // Advances whenever parameter or gradient buffer handles change (scene
    // load/set, a growing reserve, gradient buffers being allocated);
    // never on setGaussianCount(). 0 before any scene.
    uint64_t storageGeneration() const;

    // --- Differentiable rasterizer core ---

    // Forward: rasterize the current Gaussians into the output image.
    // Viewer (default) is the real-time path; Differentiable is what a
    // training step must use before backward().
    void forward(RenderMode mode = RenderMode::Viewer);

    // Backward: given dL/d(output image), compute dL/d(Gaussian params) into
    // the gradient buffers -- OVERWRITING them (pure function; summing views
    // is the trainer's job). dLoss_dImage is width*height RGBA floats,
    // row-major (the readImage() layout); the alpha channel is the coverage
    // 1 - T, so a trainer can composite its own background with correct
    // gradients. Differentiates exactly what the last forward() rendered (its
    // camera, its flags): throws unless that was a Differentiable forward of
    // the current scene. Builds the backward pass on first use and throws if
    // the device lacks fp32 buffer atomics (forward is unaffected).
    void backward(const float * dLoss_dImage);
    // Zero-copy variant: a device buffer holding the same layout.
    void backward(const BufferView & dLoss_dImage);

    // Last forward()'s image, width*height RGBA floats (headless use: the
    // image must not have been present()ed since).
    std::vector<float> readImage();
    // Zero-copy variant: the last forward()'s image as a device buffer,
    // width*height RGBA floats row-major (the readImage()/dL/dImage layout;
    // stride 16, offset 0), so a trainer can compute its loss on the GPU.
    // Throws unless the last forward() was a completed Differentiable run of
    // the current scene (the image buffer is written only by those). The
    // handle is stable across forwards and changes only on a resize;
    // contents are valid from forward() returning until the next forward().
    // Validity rules: see docs/library.md.
    BufferView imageView() const;

    // --- Relighting ---
    //
    // With lighting enabled, every forward() also splats a G-buffer (normal,
    // base colour, roughness, metallic, ambient visibility, depth) over the
    // same sorted lists and shades it against the lights and environment
    // (relightable splats; see docs/architecture.md); present() then shows the lit
    // image (or the G-buffer channel LightingSettings::output selects). The
    // colour image, readImage() and imageView() are unchanged. Lights are in
    // the scene's own (asset) space. Scenes without relightable streams get
    // derived defaults (see GaussianHostInput / relight_prepare.slang).
    void setLightingEnabled(bool enabled);
    bool lightingEnabled() const;
    // At most kMaxLights.
    void setLights(const std::vector<Light> & lights);
    const std::vector<Light> & lights() const;
    void setEnvironment(const EnvironmentLight & environment);
    void setLightingSettings(const LightingSettings & settings);
    const LightingSettings & lightingSettings() const;
    // Shadows for lights with castsShadows set. PCSS renders
    // each such light's view of the splats and filters it; RayTraced traces
    // transmittance through the Gaussians (needs ray query). Light cameras
    // frame ShadowSettings' focus sphere, or the scene's bounds when its
    // radius is 0 (a borrowed scene has no bounds -- set the focus then).
    void setShadowSettings(const ShadowSettings & settings);
    const ShadowSettings & shadowSettings() const;
    // The last lit forward()'s output, width*height RGBA floats (alpha =
    // coverage). Throws unless the last forward() ran with lighting enabled.
    std::vector<float> readLitImage();
    // Zero-copy views of the same, and of the visibility buffer (kMaxLights
    // layers of width*height floats, layer i = light i) that the lighting
    // multiplies direct lights by. Valid until the next resize.
    BufferView litImageView() const;
    BufferView visibilityView() const;
    // Host copy of the visibility buffer (kMaxLights layers of
    // width*height floats) after the last lit forward() -- verification.
    std::vector<float> readVisibility();
    // The premultiplied G-buffer sums of the last lit forward() (verification
    // and tooling): per pixel (sum w*n, sum w*depth), (sum w*base, sum
    // w*roughness), (sum w*metallic, sum w*ambient, coverage, 0).
    struct GBufferData
    {
        std::vector<glm::vec4> normalDepth;
        std::vector<glm::vec4> baseRoughness;
        std::vector<glm::vec4> metalAmbientCoverage;
    };
    GBufferData readGBuffer();

    // --- Baked visibility + teacher samples ---
    //
    // Bakes every Gaussian's visibility SH into the scene's `visibility`
    // stream (created for an owned scene; a borrowed scene must have borrowed
    // one), traced with ray query, in submissions of bounded size. Then
    // ShadowMode::BakedSH shadows in real time, and the environment term's
    // ambient occlusion uses it. Synchronous.
    void bakeVisibility(const VisibilityBakeSettings & settings = {});
    // Selects and traces teacher samples over the last LIT forward()'s
    // G-buffer, for the lights with castsShadows (needs ray query; light sizes
    // and offsets from ShadowSettings, as RayTraced uses them). Returns the
    // count. Synchronous: a trainer reads teacherSampleView() on the compute
    // queue after this returns (one queue; see docs/library.md); the view's
    // contents stay valid until the next call.
    uint32_t generateTeacherSamples(const TeacherSettings & settings = {});
    // count VisibilityTrainingSample (32 B each), smooth samples first.
    BufferView teacherSampleView() const;
    std::vector<VisibilityTrainingSample> readTeacherSamples();
    // Per sample: its pixel index, top bit set for an edge sample.
    std::vector<uint32_t> readTeacherSamplePixels();
    // How many of the samples came from the smooth (random) set.
    uint32_t teacherSmoothSampleCount() const;
    // Reuse across frames (ShadowSettings::cacheAcrossFrames): the Gaussian
    // acceleration structure is rebuilt, and cached or progressively
    // accumulated shadows restarted, only when the scene could have changed.
    // The renderer sees every change to an owned scene made through its API
    // (load, setGaussians, reserve, setGaussianCount), but not in-place GPU
    // writes: once gaussianViews() is handed out, or the scene is borrowed,
    // it stops reusing for that scene and rebuilds every frame. Call this
    // after editing parameters in place through any other route.
    void markSceneChanged();
    // What the last lit forward() did for visibility: rays dispatched,
    // acceleration-structure builds, frame-cache reuse, progressive state.
    const ShadowFrameStats & shadowFrameStats() const;
    // ShadowMode::External's source: kMaxLights layers of width*height floats
    // (visibilityView()'s layout), e.g. a trainer's learned visibility,
    // copied in on every lit forward(). The view is borrowed: keep it valid
    // while the mode is External. An invalid view clears it.
    void setExternalVisibility(const BufferView & visibility);

    // Gradients from the last backward(); see CanonicalGaussianGradientViews
    // for what each stream is w.r.t.
    GaussianGradients readGradients();

    // Zero-copy access for a borrowing application: the parameter streams the
    // renderer reads and the gradient streams backward() writes. Both span
    // gaussianCapacity() (size = capacity * stride) and report
    // gaussianCount(). gradientViews() allocates the gradient buffers on
    // first request, so a trainer can bind them before its first backward()
    // (contents are undefined until one runs); that first allocation
    // advances storageGeneration(). Validity rules: see docs/library.md.
    CanonicalGaussianBufferViews gaussianViews() const;
    CanonicalGaussianGradientViews gradientViews() const;
    // Whether backward() can run on this device (fp32 buffer atomics).
    bool backwardSupported() const;

    // Present the most recent forward() result to the attached surface.
    void present();

    uint32_t gaussianCount() const;

    // World-space bounds of the loaded scene. `valid` is false before a scene
    // loads. Scene scale varies by orders of magnitude across formats and
    // datasets -- the .spz samples are ~500 units across where a reference
    // .ply sits a few units from the origin -- so a viewer needs this to place
    // a camera that can actually see what it loaded.
    SceneBounds sceneBounds() const;
    // Max spherical-harmonic degree the LOADED SCENE actually carries, as
    // discovered by its loader -- not the cap set by setMaxSHDegree(). A
    // .splat is always 0 (DC-only), so the viewer can say so instead of
    // leaving a dead SH slider looking like a bug. 0 before a scene loads.
    uint32_t sceneSHDegree() const;
    // Tile-instance count from the most recent forward() -- a rough proxy
    // for how much culling/LOD is currently doing (not an exact surviving-
    // Gaussian count: one Gaussian can emit several tile instances). 0
    // before the first forward() call.
    uint32_t lastInstanceCount() const;
    uint32_t width() const
    {
        return m_width;
    }
    uint32_t height() const
    {
        return m_height;
    }

  private:
    uint32_t m_width;
    uint32_t m_height;

    class Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace sabercat
