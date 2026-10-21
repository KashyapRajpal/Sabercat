// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

/*
 * Sabercat stable C ABI -- forward/backward as a bindable op.
 *
 * Exposes the differentiable rasterizer to anything that can call C: the
 * Creator's Python stack (ctypes/cffi), other languages, or a C++ application
 * when it would rather not link the C++ API. Shipped as the `sabercat_c` shared
 * library (see CMakeLists.txt). A C++ application may also use
 * sabercat::Renderer directly -- this ABI is the language-neutral option, not
 * a replacement.
 *
 * Contract:
 *  - Pure op: forward renders, backward computes gradients. No loss, no
 *    optimizer, no training state (PERFORMANCE.md invariant #2).
 *  - Every call returns a SabercatResult; no C++ exception ever crosses the
 *    boundary. On failure, sabercat_last_error() describes it (thread-local,
 *    valid until the next failing call on the same thread).
 *  - A renderer handle is NOT thread-safe: call it from one thread at a time.
 *  - Structs that begin with `struct_size` (set it to sizeof(the struct))
 *    can gain appended fields without breaking callers: an older caller's
 *    smaller struct_size is honoured, and output structs are only written up
 *    to the struct_size the caller passed.
 *  - This header needs only <stdint.h>/<stddef.h>. Vulkan handles appear as
 *    uint64_t (non-dispatchable) or void* (dispatchable).
 *
 * Data layout (see docs/formats.md, canonical strides): means, scales, rotations
 * (xyzw) are 4 floats per Gaussian with the 4th lane padding; opacity is 1
 * float; SH is 16 coefficients x 4 floats (RGB + pad) per Gaussian,
 * band-ascending. Values are the ACTIVATED ones (linear scale, opacity in
 * [0,1], normalized quaternion, raw SH); gradients are w.r.t. those same
 * stored values -- the caller chains through its own activations. Images and
 * image gradients are width*height RGBA floats, row-major, alpha = coverage.
 */
#ifndef SABERCAT_C_API_H
#define SABERCAT_C_API_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(SABERCAT_C_BUILD)
#define SABERCAT_C_API __declspec(dllexport)
#else
#define SABERCAT_C_API __declspec(dllimport)
#endif
#else
#define SABERCAT_C_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Bumped on any incompatible change; additive changes keep it. */
#define SABERCAT_C_API_VERSION 1u

/* Limits: lights per renderer, and SH coefficients in a
 * Gaussian's visibility stream (mono SH, degree 3). */
#define SABERCAT_MAX_LIGHT_COUNT 8u
#define SABERCAT_VISIBILITY_COEFFICIENTS 16u

typedef enum SabercatResult
{
    SABERCAT_OK = 0,
    SABERCAT_ERROR_INVALID_ARGUMENT = 1, /* null pointer, size mismatch, ... */
    SABERCAT_ERROR_INVALID_STATE = 2,    /* e.g. backward before a forward  */
    SABERCAT_ERROR_UNSUPPORTED = 3,      /* e.g. no fp32 buffer atomics     */
    SABERCAT_ERROR_RUNTIME = 4,          /* Vulkan/IO/anything else         */
} SabercatResult;

typedef struct SabercatRenderer SabercatRenderer; /* opaque */

typedef struct SabercatRendererDesc
{
    uint32_t struct_size;
    uint32_t width;
    uint32_t height;
    const char * shader_dir; /* directory holding the compiled *.comp.spv */
    int32_t enable_validation;
} SabercatRendererDesc;

/* Host Gaussian arrays (layout above). For set_gaussians the pointers are
 * read; for get_gaussians they are written and must hold `count` Gaussians. */
typedef struct SabercatGaussians
{
    uint32_t struct_size;
    uint32_t count;
    float * means;
    float * scales;
    float * rotations;
    float * opacities;
    float * sh_coefficients;
    uint32_t sh_degree;  /* set_gaussians only: 0-3 */
    int32_t antialiased; /* set_gaussians only: trained with opacity compensation */
    /* Appended later (callers built against the older header omit it).
     * set_gaussians only: GPU storage to reserve, in Gaussians; max(count,
     * capacity) is allocated, 0 = exactly count. See reserve_gaussians. */
    uint32_t capacity;
    /* Appended later. reserved0 fills what was the earlier struct's
     * tail padding; it is never read (an older caller's padding lands
     * there) -- set it to 0. */
    uint32_t reserved0;
    /* set_gaussians only (get_gaussians ignores them): the optional
     * relightable streams, ACTIVATED values in the scene's own space; each
     * NULL = absent (the relight pass derives a default: base colour = the
     * DC colour, roughness 0.8, metallic 0, ambient visibility 1, normal =
     * the shortest ellipsoid axis facing the camera). normals: 4 floats
     * (unit xyz + pad); base_color_roughness: 4 floats (linear rgb +
     * roughness, [0,1]); metallic: 1 float ([0,1]); visibility:
     * SABERCAT_VISIBILITY_COEFFICIENTS floats (mono SH, degree 3,
     * band-ascending), all per Gaussian. */
    const float * normals;
    const float * base_color_roughness;
    const float * metallic;
    const float * visibility;
} SabercatGaussians;

/* Host gradient arrays, same layout as SabercatGaussians plus screen_means:
 * dL/d(u, v) of each projected centre in pixels (+2 pad), the per-view
 * statistic 3DGS densification thresholds on. */
typedef struct SabercatGradients
{
    uint32_t struct_size;
    uint32_t count;
    float * means;
    float * scales;
    float * rotations;
    float * opacities;
    float * sh_coefficients;
    float * screen_means;
} SabercatGradients;

/* Pinhole camera, Sabercat convention: right-handed, +Y up, looking down
 * local -Z; u = fx x/d + cx, v = cy - fy y/d. camera_to_world is a 4x4
 * column-major matrix (glm/OpenGL order). */
typedef struct SabercatCamera
{
    uint32_t struct_size;
    float camera_to_world[16];
    float fx, fy, cx, cy;
} SabercatCamera;

/* Zero-copy access: a Vulkan buffer range (VkBuffer as uint64_t). */
typedef struct SabercatBufferView
{
    uint64_t buffer;
    uint64_t offset;
    uint64_t size;
    uint32_t stride;
} SabercatBufferView;

typedef struct SabercatGaussianViews
{
    SabercatBufferView means, scales, rotations, opacities, sh_coefficients;
    uint32_t count;
} SabercatGaussianViews;

typedef struct SabercatGradientViews
{
    SabercatBufferView means, scales, rotations, opacities, sh_coefficients, screen_means;
    uint32_t count;
} SabercatGradientViews;

/* The optional relightable streams as caller-owned GPU
 * buffers (layout as in SabercatGaussians). Each is independently optional:
 * buffer 0 = absent. Strides: normals and base_color_roughness 16,
 * metallic 4, visibility 4 (per coefficient, SABERCAT_VISIBILITY_COEFFICIENTS
 * per Gaussian). */
typedef struct SabercatRelightViews
{
    SabercatBufferView normals, base_color_roughness, metallic, visibility;
} SabercatRelightViews;

/* The renderer's Vulkan objects, for a consumer sharing the device
 * (ownership model A: Sabercat created them and destroys them in
 * renderer_destroy; the borrower destroys none). Dispatchable handles as
 * void*. compute_queue is the queue EVERY Sabercat submission goes to:
 * submit work touching Sabercat's buffers there and no barriers or waits
 * are needed around Sabercat calls (see docs/library.md, synchronization).
 * It is externally synchronized -- never submit to it from another thread
 * while a sabercat_* call is running. Output struct: set struct_size. */
typedef struct SabercatDeviceInfo
{
    /* Grouped by size so the struct has no padding (56 B on 64-bit). */
    uint32_t struct_size;
    uint32_t compute_queue_family;
    void * instance;        /* VkInstance */
    void * physical_device; /* VkPhysicalDevice */
    void * device;          /* VkDevice */
    void * compute_queue;   /* VkQueue */
    uint32_t vulkan_api_version; /* VK_API_VERSION_* the instance/device target */
    int32_t float32_atomic_add;  /* shaderBufferFloat32AtomicAdd enabled */
    /* Appended later (a 48-byte struct_size omits them). ray_query:
     * VK_KHR_acceleration_structure (+ VK_KHR_deferred_host_operations) and
     * VK_KHR_ray_query are enabled with their accelerationStructure /
     * rayQuery features and bufferDeviceAddress -- what
     * SABERCAT_SHADOW_RAY_TRACED needs. reserved0 keeps the struct
     * padding-free (56 B): written as 0, never read. */
    int32_t ray_query;
    uint32_t reserved0;
} SabercatDeviceInfo;

/* A borrowed scene whose GPU buffers the caller owns, for
 * sabercat_renderer_set_gaussian_views. Views use the canonical strides
 * (means/scales/rotations/sh 16 B, opacity 4 B, 16 SH coefficients per
 * Gaussian); capacity is what the smallest parameter view holds. Gradient
 * views: all six buffers non-zero (borrowed too) or all zero (Sabercat
 * allocates). The `count` fields inside the two view structs are ignored.
 * Every requirement is checked (INVALID_ARGUMENT names the failing stream);
 * what Vulkan cannot check (usage flags, same device) is in docs/library.md.
 * Grouped by size: the four 4-byte fields keep the views 8-aligned. */
typedef struct SabercatBorrowedGaussians
{
    uint32_t struct_size;
    uint32_t count;     /* Gaussians to render, 1..capacity */
    uint32_t sh_degree; /* highest SH band the data holds, 0-3 */
    int32_t antialiased;
    SabercatGaussianViews parameters;
    SabercatGradientViews gradients;
    /* Appended later (a struct_size from before it omits it): optional
     * relightable streams, each covering the capacity when given and not
     * overlapping any other stream. */
    SabercatRelightViews relight;
} SabercatBorrowedGaussians;

/* The Gaussian storage's shape. Views span `capacity` elements and
 * forward/backward process `count`. `generation` advances whenever buffer
 * handles change (new scene, a growing reserve, gradient buffers allocated)
 * and never on set_gaussian_count: re-fetch views when it moves. Output
 * struct: set struct_size. */
typedef struct SabercatGaussianStorage
{
    /* 3 x uint32 + uint64: 4 B of tail padding is unavoidable here. */
    uint32_t struct_size;
    uint32_t count;
    uint32_t capacity;
    uint64_t generation;
} SabercatGaussianStorage;

/* --- Relighting and shadows ---
 *
 * Enum-typed struct fields are uint32_t (an enum's size is the compiler's
 * choice); the enums below name their values. Every struct here holds only
 * 4-byte fields, so none has padding. */

typedef enum SabercatLightType
{
    SABERCAT_LIGHT_DIRECTIONAL = 0, /* irradiance color*intensity arriving along `direction` */
    SABERCAT_LIGHT_POINT = 1,       /* radiant intensity color*intensity, inverse-square falloff */
    SABERCAT_LIGHT_RECT = 2,        /* one-sided area emitter of radiance color*intensity */
} SabercatLightType;

/* One light, in the scene's own (asset) space -- the space its means and
 * normals are in. Every field is read (the unused ones may be anything
 * finite). In an array, lights[0].struct_size is the stride and every
 * element must carry the same value. */
typedef struct SabercatLight
{
    uint32_t struct_size;
    uint32_t type;         /* SabercatLightType */
    int32_t casts_shadows; /* consult the shadow mode's visibility */
    float intensity;       /* scales color */
    float color[3];        /* linear rgb */
    float position[3];     /* POINT, RECT: centre */
    float direction[3];    /* DIRECTIONAL: the way the light travels (normalized internally) */
    float axis_x[3];       /* RECT: unit width axis */
    float axis_y[3];       /* RECT: unit height axis; emits along cross(axis_x, axis_y) */
    float half_width;      /* RECT: half extent along axis_x */
    float half_height;     /* RECT: half extent along axis_y */
} SabercatLight;

/* Distant environment light: radiance SH of degree 2 (9 RGB coefficients,
 * band-ascending, in Sabercat's 3DGS-signed real SH basis -- the colour SH's
 * basis). All zero = no environment. Shaded as cosine-convolved diffuse
 * irradiance plus a rough specular term, attenuated by each Gaussian's
 * ambient visibility. */
typedef struct SabercatEnvironment
{
    uint32_t struct_size;
    float intensity;
    float sh[9 * 3]; /* coefficient k's rgb at sh[k*3 .. k*3+2] */
} SabercatEnvironment;

typedef enum SabercatLightingOutput
{
    SABERCAT_LIGHTING_OUTPUT_LIT = 0,
    SABERCAT_LIGHTING_OUTPUT_NORMAL = 1,            /* world normal, n*0.5+0.5 */
    SABERCAT_LIGHTING_OUTPUT_BASE_COLOR = 2,
    SABERCAT_LIGHTING_OUTPUT_ROUGHNESS = 3,
    SABERCAT_LIGHTING_OUTPUT_METALLIC = 4,
    SABERCAT_LIGHTING_OUTPUT_AMBIENT_OCCLUSION = 5, /* splatted ambient visibility */
    SABERCAT_LIGHTING_OUTPUT_VISIBILITY = 6,        /* light `visibility_light`'s shadow layer */
    SABERCAT_LIGHTING_OUTPUT_DEPTH = 7,             /* expected depth * depth_scale */
} SabercatLightingOutput;

/* What the lit image holds. Defaults: get_lighting_settings on a new
 * renderer (background 0, exposure 1, LIT, rect_samples 16). */
typedef struct SabercatLightingSettings
{
    uint32_t struct_size;
    uint32_t output;           /* SabercatLightingOutput */
    uint32_t rect_samples;     /* stratified samples per RECT light per pixel (rounded to a square) */
    uint32_t visibility_light; /* which light OUTPUT_VISIBILITY shows */
    float background[3];       /* composited where coverage < 1 */
    float exposure;            /* linear multiplier on the result */
    float depth_scale;         /* OUTPUT_DEPTH multiplier */
} SabercatLightingSettings;

typedef enum SabercatShadowMode
{
    SABERCAT_SHADOW_NONE = 0,       /* every light fully visible (the default) */
    SABERCAT_SHADOW_PCSS = 1,       /* percentage-closer soft shadows over a splat shadow map */
    SABERCAT_SHADOW_RAY_TRACED = 2, /* transmittance traced through the Gaussians; needs
                                       SabercatDeviceInfo::ray_query */
} SabercatShadowMode;

/* Shadows for lights with casts_shadows set. PCSS light cameras frame the
 * focus sphere (focus_center, focus_radius); focus_radius 0 = the loaded
 * scene's bounds, which a borrowed scene (set_gaussian_views) does not
 * have -- set the focus explicitly there, or the forward fails. The
 * fractions are of the focus radius. Defaults: get_shadow_settings on a new
 * renderer. */
typedef struct SabercatShadowSettings
{
    uint32_t struct_size;
    uint32_t mode;                     /* SabercatShadowMode */
    uint32_t map_resolution;           /* PCSS shadow-map texels per side, >= 1 */
    uint32_t blocker_samples_per_axis; /* PCSS blocker search grid, >= 1 */
    uint32_t filter_samples_per_axis;  /* PCSS filter grid, >= 1 */
    uint32_t ray_samples_per_axis;     /* RAY_TRACED stratified emitter samples per axis, >= 1 */
    float focus_center[3];
    float focus_radius;       /* >= 0 */
    float point_light_radius; /* POINT emitter radius, fraction of focus_radius */
    float sun_angular_radius; /* DIRECTIONAL angular radius, radians */
    float depth_bias;         /* fraction of focus_radius */
    float normal_offset;      /* fraction of focus_radius */
    float max_filter_texels;  /* PCSS search/filter clamp */
} SabercatShadowSettings;

SABERCAT_C_API uint32_t sabercat_api_version(void);
SABERCAT_C_API const char * sabercat_last_error(void);

SABERCAT_C_API SabercatResult sabercat_renderer_create(const SabercatRendererDesc * desc,
                                                       SabercatRenderer ** out_renderer);
/* A headless renderer on a device the caller
 * created -- or another renderer's, straight from get_device_info (e.g. a
 * second image size over one scene). `device` is read as a declaration:
 * vulkan_api_version is the instance's apiVersion (>= 1.3) and
 * float32_atomic_add whether shaderBufferFloat32AtomicAdd (with
 * VK_EXT_shader_atomic_float) was enabled; ray_query (read only when
 * struct_size covers it -- a 48-byte struct declares 0) whether the ray
 * query set it describes was. Everything is checked first;
 * INVALID_ARGUMENT names the problem. desc->enable_validation is ignored
 * (the owner decides). Destroy this renderer before the device's owner
 * destroys the device. */
SABERCAT_C_API SabercatResult sabercat_renderer_create_borrowed(const SabercatRendererDesc * desc,
                                                                const SabercatDeviceInfo * device,
                                                                SabercatRenderer ** out_renderer);
/* Accepts NULL. */
SABERCAT_C_API void sabercat_renderer_destroy(SabercatRenderer * renderer);

/* Scene: from a file (.ply/.splat/.spz/.ksplat/.sog) or from host arrays. */
SABERCAT_C_API SabercatResult sabercat_renderer_load_scene(SabercatRenderer * renderer,
                                                           const char * path);
SABERCAT_C_API SabercatResult sabercat_renderer_set_gaussians(SabercatRenderer * renderer,
                                                              const SabercatGaussians * gaussians);
SABERCAT_C_API SabercatResult sabercat_renderer_get_gaussians(SabercatRenderer * renderer,
                                                              SabercatGaussians * out);
SABERCAT_C_API SabercatResult sabercat_renderer_get_gaussian_count(SabercatRenderer * renderer,
                                                                   uint32_t * out_count);

SABERCAT_C_API SabercatResult sabercat_renderer_set_camera(SabercatRenderer * renderer,
                                                           const SabercatCamera * camera);
/* Max SH degree evaluated (0-3); training typically raises it over time. */
SABERCAT_C_API SabercatResult sabercat_renderer_set_sh_degree(SabercatRenderer * renderer,
                                                              uint32_t degree);
/* Overrides the scene's antialiased (opacity compensation) flag. */
SABERCAT_C_API SabercatResult sabercat_renderer_set_antialiased(SabercatRenderer * renderer,
                                                                int32_t antialiased);

/* Always a differentiable forward: viewer LOD knobs never apply here. */
SABERCAT_C_API SabercatResult sabercat_renderer_forward(SabercatRenderer * renderer);
/* float_count must be width*height*4. */
SABERCAT_C_API SabercatResult sabercat_renderer_read_image(SabercatRenderer * renderer,
                                                           float * rgba, size_t float_count);
/* Differentiates the last forward. Gradients are OVERWRITTEN, not summed.
 * float_count must be width*height*4. */
SABERCAT_C_API SabercatResult sabercat_renderer_backward(SabercatRenderer * renderer,
                                                         const float * dl_dimage_rgba,
                                                         size_t float_count);
SABERCAT_C_API SabercatResult sabercat_renderer_read_gradients(SabercatRenderer * renderer,
                                                               SabercatGradients * out);

/* Zero-copy views of the GPU streams. Each spans the storage capacity
 * (size = capacity * stride) and reports the active count. Handles stay
 * valid until the storage generation changes (see SabercatGaussianStorage)
 * or the renderer is destroyed. get_gradient_views allocates the gradient
 * buffers on first call (contents undefined until a backward). */
SABERCAT_C_API SabercatResult sabercat_renderer_get_gaussian_views(SabercatRenderer * renderer,
                                                                   SabercatGaussianViews * out);
SABERCAT_C_API SabercatResult sabercat_renderer_get_gradient_views(SabercatRenderer * renderer,
                                                                   SabercatGradientViews * out);

/* --- Device sharing and resizable storage (densification) --- */

SABERCAT_C_API SabercatResult sabercat_renderer_get_device_info(SabercatRenderer * renderer,
                                                                SabercatDeviceInfo * out);
SABERCAT_C_API SabercatResult sabercat_renderer_get_gaussian_storage(
    SabercatRenderer * renderer, SabercatGaussianStorage * out);
/* Zero-copy view of the last forward's image: width*height RGBA floats,
 * row-major (the read_image / backward layout), stride 16, offset 0.
 * INVALID_STATE without a forward of the current scene. The handle is
 * stable across forwards; contents are valid from forward returning until
 * the next forward. */
SABERCAT_C_API SabercatResult sabercat_renderer_get_image_view(SabercatRenderer * renderer,
                                                               SabercatBufferView * out);
/* Zero-copy backward: dL/dImage read from a device buffer in the read_image
 * layout (stride 16, size >= width*height*16) instead of host memory. The
 * buffer needs VK_BUFFER_USAGE_STORAGE_BUFFER_BIT and must have been written
 * on the compute queue (or host-waited); see docs/library.md, synchronization
 * contract. Otherwise identical to sabercat_renderer_backward. */
SABERCAT_C_API SabercatResult sabercat_renderer_backward_view(SabercatRenderer * renderer,
                                                              const SabercatBufferView * dl_dimage);

/* --- Borrowed device and image size --- */

/* Replaces the scene with caller-owned GPU buffers (see
 * SabercatBorrowedGaussians), rendered in place. reserve_gaussians then
 * fails: grow your buffers and call this again. The buffers must outlive
 * the scene. On INVALID_ARGUMENT the current scene is untouched. */
SABERCAT_C_API SabercatResult sabercat_renderer_set_gaussian_views(
    SabercatRenderer * renderer, const SabercatBorrowedGaussians * scene);
/* Changes the output image size. Invalidates the last forward; the image
 * view handle changes. Set a camera whose intrinsics match. */
SABERCAT_C_API SabercatResult sabercat_renderer_set_image_size(SabercatRenderer * renderer,
                                                               uint32_t width, uint32_t height);
/* Grows storage to `capacity` Gaussians, GPU-copying the active prefix of
 * every parameter stream; gradients are reallocated (undefined until the
 * next backward). Replaces every buffer handle -- re-fetch views after.
 * *out_grew (may be NULL) is 0, with nothing changed, if capacity <= the
 * current capacity. */
SABERCAT_C_API SabercatResult sabercat_renderer_reserve_gaussians(SabercatRenderer * renderer,
                                                                  uint32_t capacity,
                                                                  int32_t * out_grew);
/* Sets how many Gaussians forward/backward process, 1 <= count <= capacity.
 * No allocation, no copy, no GPU work; every handle stays valid. Write new
 * Gaussians into [old count, count) through the views first. A forward is
 * needed before the next backward/read_image/read_gradients. */
SABERCAT_C_API SabercatResult sabercat_renderer_set_gaussian_count(SabercatRenderer * renderer,
                                                                   uint32_t count);

/* --- Relighting and shadows ---
 *
 * Relightable streams come with the scene: SabercatGaussians' normals ..
 * visibility for set_gaussians, SabercatBorrowedGaussians::relight for
 * set_gaussian_views, or the file for load_scene (an R3DG-style .ply). With
 * lighting enabled every forward also splats a G-buffer over the same
 * sorted lists and shades it into the LIT IMAGE; read_image, get_image_view
 * and backward are unchanged (the lit image is not differentiated). Lights,
 * environment and settings persist across scenes and apply from the next
 * forward. */

/* Off by default. */
SABERCAT_C_API SabercatResult sabercat_renderer_set_lighting_enabled(SabercatRenderer * renderer,
                                                                     int32_t enabled);
/* Replaces every light; light_count <= SABERCAT_MAX_LIGHT_COUNT (0 clears;
 * lights may then be NULL). Light i's shadow is visibility layer i. */
SABERCAT_C_API SabercatResult sabercat_renderer_set_lights(SabercatRenderer * renderer,
                                                           const SabercatLight * lights,
                                                           uint32_t light_count);
SABERCAT_C_API SabercatResult sabercat_renderer_set_environment(
    SabercatRenderer * renderer, const SabercatEnvironment * environment);
SABERCAT_C_API SabercatResult sabercat_renderer_set_lighting_settings(
    SabercatRenderer * renderer, const SabercatLightingSettings * settings);
/* Output struct: set struct_size. */
SABERCAT_C_API SabercatResult sabercat_renderer_get_lighting_settings(
    SabercatRenderer * renderer, SabercatLightingSettings * out);
/* UNSUPPORTED for SABERCAT_SHADOW_RAY_TRACED on a device without ray query
 * (SabercatDeviceInfo::ray_query == 0; use PCSS); INVALID_ARGUMENT for an
 * unknown mode, a zero resolution/sample count or a negative focus radius.
 * On any failure the current settings are kept. */
SABERCAT_C_API SabercatResult sabercat_renderer_set_shadow_settings(
    SabercatRenderer * renderer, const SabercatShadowSettings * settings);
/* Output struct: set struct_size. */
SABERCAT_C_API SabercatResult sabercat_renderer_get_shadow_settings(SabercatRenderer * renderer,
                                                                    SabercatShadowSettings * out);
/* The last forward's lit image: width*height RGBA floats, row-major (the
 * read_image layout), rgb = (shaded * coverage + background * (1 -
 * coverage)) * exposure, alpha = coverage. float_count must be
 * width*height*4. INVALID_STATE unless the last forward of the current
 * scene ran with lighting enabled. */
SABERCAT_C_API SabercatResult sabercat_renderer_read_lit_image(SabercatRenderer * renderer,
                                                               float * rgba, size_t float_count);
/* Zero-copy view of the same: stride 16, offset 0, size width*height*16.
 * Same INVALID_STATE rule and validity as get_image_view (contents valid
 * from forward returning until the next forward); the handle changes on the
 * first lit forward after a set_image_size. */
SABERCAT_C_API SabercatResult sabercat_renderer_get_lit_image_view(SabercatRenderer * renderer,
                                                                   SabercatBufferView * out);

#ifdef __cplusplus
}
#endif

#endif /* SABERCAT_C_API_H */
