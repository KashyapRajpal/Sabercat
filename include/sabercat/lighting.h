// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <glm/glm.hpp>

#include <array>
#include <cstdint>

#include "sabercat/render_config.h"

namespace sabercat
{

// Relighting splats. Lights live in the scene's own (asset)
// space -- the space its means and normals are in -- so a Y-down scene viewed
// with setFlipUpAxis keeps its lights attached to it.

enum class LightType : uint32_t
{
    Directional = 0, // irradiance `color * intensity` arriving along `direction`
    Point = 1,       // radiant intensity `color * intensity`, inverse-square falloff
    Rect = 2,        // one-sided area emitter of radiance `color * intensity`
};

// One light. Grouped as vec3 + 4-byte pairs, so the struct has no padding.
struct Light
{
    glm::vec3 position{0.0f};          // Point, Rect: center
    float intensity = 1.0f;            // scales `color`
    glm::vec3 direction{0.0f, -1.0f, 0.0f}; // Directional: the way the light travels
    float halfWidth = 0.5f;            // Rect: half extent along axisX
    glm::vec3 axisX{1.0f, 0.0f, 0.0f}; // Rect: unit width axis
    float halfHeight = 0.5f;           // Rect: half extent along axisY
    glm::vec3 axisY{0.0f, 0.0f, 1.0f}; // Rect: unit height axis; emits along cross(axisX, axisY)
    LightType type = LightType::Directional;
    glm::vec3 color{1.0f};
    uint32_t castsShadows = 1; // whether a shadow source is consulted
};

// Distant environment lighting as radiance SH of degree 2 (9 RGB
// coefficients) in Sabercat's SH basis -- the same (3DGS-signed) real basis
// the color SH and the per-Gaussian visibility use, so their products and
// convolutions are consistent. Shaded as diffuse irradiance (exact cosine
// convolution) plus a rough specular approximation, both attenuated by the
// per-Gaussian ambient visibility when the scene has a visibility stream.
struct EnvironmentLight
{
    std::array<glm::vec3, 9> sh{}; // radiance coefficients (all zero = no environment)
    float intensity = 1.0f;

    // A constant radiance from every direction.
    static EnvironmentLight uniform(const glm::vec3 & radiance);
    // A smooth sky: `zenith` straight up (`up`), blending through `horizon`
    // to `ground` below, projected numerically onto the 9 coefficients.
    static EnvironmentLight sky(const glm::vec3 & zenith, const glm::vec3 & horizon,
                                const glm::vec3 & ground, const glm::vec3 & up);
    // Radiance this environment returns from direction `dir` (unit).
    glm::vec3 radiance(const glm::vec3 & dir) const;
};

// What the lighting pass writes to its output (and what present() shows when
// lighting is enabled): the lit image, or one G-buffer channel for
// inspecting a relightable scene. Lit is display-referred: its rgb is
// sRGB-encoded after compositing in linear light. Every other output is a raw
// data value (no encoding), alpha = coverage for all of them.
enum class LightingOutput : uint32_t
{
    Lit = 0,              // display-referred sRGB rgb, composited over the background
    Normal = 1,           // world normal, mapped n*0.5+0.5 (raw data)
    BaseColor = 2,        // linear base color (raw data)
    Roughness = 3,
    Metallic = 4,
    AmbientOcclusion = 5, // splatted per-Gaussian ambient visibility
    Visibility = 6,       // the visibility layer of light `visibilityLight`
    Depth = 7,            // expected depth * depthScale
};

struct LightingSettings
{
    // Composited where coverage < 1. Display-referred (sRGB) for Lit, which
    // decodes it before compositing; used as is by the data outputs.
    glm::vec3 background{0.0f};
    float exposure = 1.0f;      // linear multiplier on the result, before Lit's encode
    LightingOutput output = LightingOutput::Lit;
    uint32_t rectSamples = 16;  // stratified samples per Rect light per pixel (rounded to a square)
    uint32_t visibilityLight = 0; // which light LightingOutput::Visibility shows
    float depthScale = 0.1f;    // LightingOutput::Depth multiplier
};

// Where the lighting pass's per-light visibility comes from.
enum class ShadowMode : uint32_t
{
    None = 0,      // every light fully visible
    PCSS = 1,      // percentage-closer soft shadows over a splat shadow map (baseline)
    RayTraced = 2, // ray-traced transmittance through the Gaussians (ground truth; needs ray query)
    // Baked and external visibility:
    BakedSH = 3,  // the per-Gaussian visibility SH (Renderer::bakeVisibility, or a loaded
                  // R3DG/learned stream) evaluated toward each light and splatted -- real time
    External = 4, // copied from Renderer::setExternalVisibility (e.g. a trainer's learned visibility)
};

// What the visibility layers show after the shadow source ran
// (seen through LightingOutput::Visibility; the lit image uses them too).
enum class VisibilityDebug : uint32_t
{
    Off = 0,
    // |source - ray-traced reference| per layer, the reference traced with
    // ShadowSettings::referenceSamplesPerAxis (needs ray query).
    ErrorVsReference = 1,
    // The last generateTeacherSamples() call's samples: 0 everywhere, 0.5 at
    // a smooth sample's pixel and 1 at an edge sample's, in its light's layer.
    SampleLocations = 2,
};

// Shadow configuration. Light cameras for PCSS frame a "focus" sphere:
// focusRadius > 0 uses focusCenter/focusRadius, otherwise the loaded scene's
// bounds (a borrowed scene has none -- set the focus explicitly then).
struct ShadowSettings
{
    glm::vec3 focusCenter{0.0f};
    float focusRadius = 0.0f;
    ShadowMode mode = ShadowMode::None;
    uint32_t mapResolution = 1024;      // PCSS shadow-map texels per side
    uint32_t blockerSamplesPerAxis = 4; // PCSS blocker search grid
    uint32_t filterSamplesPerAxis = 6;  // PCSS filter grid
    float pointLightRadius = 0.02f;     // emitter radius for Point lights, fraction of focusRadius
    float sunAngularRadius = 0.0093f;   // Directional lights' angular radius (radians; the sun's ~0.27 deg x2)
    float depthBias = 0.004f;           // fraction of focusRadius
    float normalOffset = 0.002f;        // fraction of focusRadius
    float maxFilterTexels = 32.0f;      // PCSS search/filter clamp
    uint32_t raySamplesPerAxis = 4;     // RayTraced: stratified samples per axis on area/point emitters
    // Visibility debugging:
    VisibilityDebug debug = VisibilityDebug::Off;
    uint32_t referenceSamplesPerAxis = 8; // ErrorVsReference: the reference's emitter grid
    // Reuse across frames (only while the renderer can see every scene
    // change: an owned scene whose gaussianViews() were never handed out).
    // While nothing a light's visibility depends on changes (camera, scene,
    // that light, these settings), RayTraced with progressiveSamplesPerFrame
    // = k > 0 adds k emitter samples per frame, converging to exactly the
    // full-grid result (same rays, same summation order), then traces nothing.
    // 0 traces the whole grid every frame. PCSS, BakedSH and full-grid
    // RayTraced frames are reused whole when nothing changed. cacheAcrossFrames
    // = 0 disables all of it.
    uint32_t progressiveSamplesPerFrame = 0;
    uint32_t cacheAcrossFrames = 1;
    // RayTraced approximations -- the defaults are the exact model; grade
    // other values with VisibilityDebug::ErrorVsReference. The bake and
    // teacher samples always use the exact model.
    //   rayMinTransmittance  a shadow ray stops (reports 0) once darker than
    //                        this: per-pixel error < this value. Exact 1e-4.
    //   rayMinAlpha          Gaussians whose peak alpha along a ray is below
    //                        this are skipped, and the ray-tracing structure
    //                        shrinks to match: big savings in faint fog,
    //                        shadows a little lighter. Exact 1/255.
    //   rayDownsample        trace one receiver per s x s block, then fill
    //                        the rest with a depth/normal-aware upsample
    //                        (traced pixels keep their values). Exact 1.
    float rayMinTransmittance = 1e-4f;
    float rayMinAlpha = 1.0f / 255.0f;
    uint32_t rayDownsample = 1;
};

// What the last lit forward() did for visibility (profiling, tests, viewer).
struct ShadowFrameStats
{
    uint64_t emitterSamplesDispatched = 0; // RayTraced: pixels x emitter samples dispatched (rays, upper bound)
    uint32_t accelerationBuilds = 0;       // acceleration-structure builds this frame
    uint32_t reusedFrame = 0;              // 1 = the layers came from the frame cache
    uint32_t convergedLights = 0;          // progressive RayTraced: lights whose accumulation is complete
    uint32_t pendingLights = 0;            // ... still accumulating
};

// Renderer::bakeVisibility. Each Gaussian traces
// `directions` rays from its mean (a fixed spherical Fibonacci set) through
// the other Gaussians and stores the 16-coefficient SH of the transmittance
// in the `visibility` stream. The work is split into submissions, so no single
// one can trip a driver timeout.
struct VisibilityBakeSettings
{
    uint32_t directions = 256;
    uint32_t maxRaysPerSubmit = 8u << 20; // upper cap on the rays of one submission
    float rayLength = 0.0f;               // world units; 0 = 4x the scene's bounding radius (needed for borrowed scenes)
    // Submissions grow (up to maxRaysPerSubmit) or shrink so each takes about
    // this long (host wall clock, adapted from the previous submission's
    // time), but never below 1M rays: a submission's time has a floor set by
    // one thread tracing its Gaussian's directions in sequence, so smaller
    // batches only add submissions. 0 = fixed batches of maxRaysPerSubmit
    // rays. Results do not depend on the batching.
    float targetSubmitMilliseconds = 100.0f;
};
static_assert(sizeof(VisibilityBakeSettings) == 16, "VisibilityBakeSettings is padding-free");

// Renderer::generateTeacherSamples. Candidates are the last lit
// frame's covered pixels: `randomSamples` drawn from the smooth ones and
// `edgeSamples` from depth/normal discontinuities and silhouettes, each set
// stratified over its list (fewer when a list is shorter). Each sample gets
// one shadow-casting light, round-robin, and a ray-traced target over an
// `emitterSamplesPerAxis`^2 emitter grid -- the RayTraced shadow mode's
// rays, so a target equals that mode's visibility at the same grid.
struct TeacherSettings
{
    uint32_t randomSamples = 6144; // 75% of the default 8192
    uint32_t edgeSamples = 2048;   // 25%
    uint32_t emitterSamplesPerAxis = 4; // 16 emitter samples per target
    uint32_t seed = 0;             // stratum jitter; vary per call for fresh samples
    float minCoverage = 0.5f;      // a pixel counts as covered at this alpha
    float edgeDepthRatio = 0.05f;  // neighbor depth differs by more than this fraction
    float edgeNormalCos = 0.9f;    // or its normal by more than acos(this) (~26 deg)
};

// One teacher sample, std430 (teacher.slang). A trainer reads
// these through Renderer::teacherSampleView() -- named apart from a trainer's
// own TrainingSampleLabel on purpose.
struct VisibilityTrainingSample
{
    glm::vec3 position; // asset space, on the surface (not offset along the normal)
    float visibilityTarget; // [0, 1]: mean transmittance toward the light's emitter
    glm::vec3 normal;   // unit, or zero where the splatted normals canceled
    uint32_t lightIndex; // into Renderer::lights()
};
static_assert(sizeof(VisibilityTrainingSample) == 32, "VisibilityTrainingSample must match teacher.slang");

// Sabercat's SH basis up to degree 3 at unit direction `dir` (16 values, the
// order and signs of forward_common.slangh's evaluateSHRaw). Used to project
// environments and visibility, and by the CPU references in tests.
void shBasis16(const glm::vec3 & dir, float out[16]);

} // namespace sabercat
