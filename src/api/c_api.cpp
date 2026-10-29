// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The stable C ABI (include/sabercat/c_api.h) over sabercat::Renderer.
//
// Built into the `sabercat_c` SHARED library, not the static `sabercat` one:
// a C ABI exists to be loaded across a binary boundary (ctypes, another
// toolchain), and building it as a DLL is what makes the C test exercise
// that boundary for real. Every entry point catches everything -- a C++
// exception unwinding into a C caller is undefined behavior.

#include "sabercat/c_api.h"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "sabercat/camera.h"
#include "sabercat/device.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/borrowed.h"
#include "sabercat/device.h"
#include "sabercat/lighting.h"
#include "sabercat/render_config.h"
#include "sabercat/renderer.h"

struct SabercatRenderer
{
    std::unique_ptr<sabercat::Renderer> renderer;
    uint32_t width = 0;
    uint32_t height = 0;
    bool hasCamera = false;
    // Mirrors Renderer's own scene-generation check so a misuse can be
    // reported as INVALID_STATE rather than a generic runtime error.
    bool hasForward = false;
    bool hasBackward = false;
    // The last forward ran with lighting enabled. Only meaningful while
    // hasForward holds (every invalidation clears that one).
    bool hasLitForward = false;
};

namespace
{

thread_local std::string g_lastError;

SabercatResult fail(SabercatResult code, const std::string & message)
{
    g_lastError = message;
    return code;
}

// Runs `body`, translating exceptions: std::invalid_argument (the
// borrowed-device/buffer checks) into SABERCAT_ERROR_INVALID_ARGUMENT,
// anything else into SABERCAT_ERROR_RUNTIME.
template <typename F>
SabercatResult guarded(const char * where, F && body)
{
    try
    {
        return body();
    }
    catch (const std::invalid_argument & e)
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT, std::string(where) + ": " + e.what());
    }
    catch (const std::exception & e)
    {
        return fail(SABERCAT_ERROR_RUNTIME, std::string(where) + ": " + e.what());
    }
    catch (...)
    {
        return fail(SABERCAT_ERROR_RUNTIME, std::string(where) + ": unknown exception");
    }
}

sabercat::GaussianHostOutput toOutput(const SabercatGaussians & g)
{
    sabercat::GaussianHostOutput out;
    out.count = g.count;
    out.means = g.means;
    out.scales = g.scales;
    out.rotations = g.rotations;
    out.opacities = g.opacities;
    out.shCoefficients = g.sh_coefficients;
    return out;
}

sabercat::BufferView fromView(const SabercatBufferView & v)
{
    sabercat::BufferView out;
    out.buffer = reinterpret_cast<VkBuffer>(v.buffer);
    out.offset = v.offset;
    out.size = v.size;
    out.stride = v.stride;
    return out;
}

SabercatBufferView toView(const sabercat::BufferView & v)
{
    SabercatBufferView out;
    out.buffer = reinterpret_cast<uint64_t>(v.buffer);
    out.offset = v.offset;
    out.size = v.size;
    out.stride = v.stride;
    return out;
}

bool gaussianArraysPresent(const SabercatGaussians & g)
{
    return g.means && g.scales && g.rotations && g.opacities && g.sh_coefficients;
}

// SabercatGaussians as its original header declared it, before `capacity` was
// appended: the smallest struct_size still accepted.
constexpr uint32_t kGaussiansV1Size = static_cast<uint32_t>(offsetof(SabercatGaussians, capacity));

// Whether a caller's struct (of size struct_size) includes `field`.
#define SABERCAT_HAS_FIELD(ptr, type, field)                                                       \
    ((ptr)->struct_size >= offsetof(type, field) + sizeof(((type *)nullptr)->field))

// SabercatBorrowedGaussians as its original header declared it, before
// `relight` was appended.
constexpr uint32_t kBorrowedGaussiansV1Size =
    static_cast<uint32_t>(offsetof(SabercatBorrowedGaussians, relight));

// SabercatRendererDesc as its original header declared it, before reserved0
// and depth_layout were appended (they fill its old tail padding, so the
// struct was 32 B and a caller built against that header passes 32). Every
// field read from it lies below this offset.
constexpr uint32_t kRendererDescV1Size =
    static_cast<uint32_t>(offsetof(SabercatRendererDesc, reserved0));

static_assert(sizeof(void *) != 8
                  || (sizeof(SabercatDeviceInfo) == 56
                      && offsetof(SabercatDeviceInfo, ray_query) == 48
                      && offsetof(SabercatDeviceInfo, reserved0) == 52),
              "SabercatDeviceInfo is meant to be padding-free, v2 fields appended at the old 48 B");
// Later fields: reserved0 fills the original struct's 4 B tail padding, so the
// pointers start at its old size (64) and nothing is padded.
static_assert(sizeof(void *) != 8
                  || (offsetof(SabercatGaussians, capacity) == 56
                      && offsetof(SabercatGaussians, reserved0) == 60
                      && offsetof(SabercatGaussians, normals) == 64
                      && sizeof(SabercatGaussians) == 96),
              "SabercatGaussians is meant to be padding-free");
// The four 4-byte fields fill exactly the 16 bytes before the 8-aligned views.
static_assert(offsetof(SabercatBorrowedGaussians, parameters) == 16
                  && offsetof(SabercatBorrowedGaussians, gradients)
                         == 16 + sizeof(SabercatGaussianViews)
                  && offsetof(SabercatBorrowedGaussians, relight)
                         == 16 + sizeof(SabercatGaussianViews) + sizeof(SabercatGradientViews)
                  && sizeof(SabercatBorrowedGaussians)
                         == 16 + sizeof(SabercatGaussianViews) + sizeof(SabercatGradientViews)
                                + sizeof(SabercatRelightViews),
              "SabercatBorrowedGaussians adds no padding of its own");
static_assert(sizeof(SabercatRelightViews) == 4 * sizeof(SabercatBufferView),
              "SabercatRelightViews adds no padding of its own");
// The appended renderer-desc fields fill the 4 B tail padding of the 32 B
// original, and reserved1 leaves the struct padding-free at 40 B.
static_assert(sizeof(void *) != 8
                  || (offsetof(SabercatRendererDesc, enable_validation) == 24
                      && offsetof(SabercatRendererDesc, reserved0) == 28
                      && offsetof(SabercatRendererDesc, depth_layout) == 32
                      && offsetof(SabercatRendererDesc, reserved1) == 36
                      && sizeof(SabercatRendererDesc) == 40),
              "SabercatRendererDesc is meant to be padding-free after its appended fields");
static_assert(SABERCAT_DEPTH_LAYOUT_NONE == static_cast<uint32_t>(sabercat::DepthLayout::None)
                  && SABERCAT_DEPTH_LAYOUT_DEPTH
                         == static_cast<uint32_t>(sabercat::DepthLayout::Depth)
                  && SABERCAT_DEPTH_LAYOUT_DEPTH_COVERAGE
                         == static_cast<uint32_t>(sabercat::DepthLayout::DepthCoverage)
                  && SABERCAT_DEPTH_LAYOUT_DEPTH_COVERAGE_FRONT
                         == static_cast<uint32_t>(sabercat::DepthLayout::DepthCoverageFront),
              "SabercatDepthLayout differs from sabercat::DepthLayout");
// Lighting structs: 4-byte fields only, so the size is the field count.
static_assert(sizeof(SabercatLight) == 21 * 4 && offsetof(SabercatLight, color) == 16
                  && offsetof(SabercatLight, half_width) == 76,
              "SabercatLight is meant to be padding-free");
static_assert(sizeof(SabercatEnvironment) == 29 * 4, "SabercatEnvironment is padding-free");
static_assert(sizeof(SabercatLightingSettings) == 9 * 4
                  && offsetof(SabercatLightingSettings, background) == 16,
              "SabercatLightingSettings is meant to be padding-free");
static_assert(sizeof(SabercatShadowSettings) == 15 * 4
                  && offsetof(SabercatShadowSettings, focus_center) == 24,
              "SabercatShadowSettings is meant to be padding-free");
// The C enums and limits restate the C++ ones; keep them in lockstep.
static_assert(SABERCAT_MAX_LIGHT_COUNT == sabercat::kMaxLights
                  && SABERCAT_VISIBILITY_COEFFICIENTS == sabercat::kVisibilityCoefficients,
              "C ABI limits differ from render_config.h");
static_assert(SABERCAT_LIGHT_DIRECTIONAL == static_cast<uint32_t>(sabercat::LightType::Directional)
                  && SABERCAT_LIGHT_POINT == static_cast<uint32_t>(sabercat::LightType::Point)
                  && SABERCAT_LIGHT_RECT == static_cast<uint32_t>(sabercat::LightType::Rect),
              "SabercatLightType differs from sabercat::LightType");
static_assert(SABERCAT_SHADOW_NONE == static_cast<uint32_t>(sabercat::ShadowMode::None)
                  && SABERCAT_SHADOW_PCSS == static_cast<uint32_t>(sabercat::ShadowMode::PCSS)
                  && SABERCAT_SHADOW_RAY_TRACED
                         == static_cast<uint32_t>(sabercat::ShadowMode::RayTraced),
              "SabercatShadowMode differs from sabercat::ShadowMode");
static_assert(SABERCAT_LIGHTING_OUTPUT_LIT == static_cast<uint32_t>(sabercat::LightingOutput::Lit)
                  && SABERCAT_LIGHTING_OUTPUT_DEPTH
                         == static_cast<uint32_t>(sabercat::LightingOutput::Depth),
              "SabercatLightingOutput differs from sabercat::LightingOutput");

glm::vec3 vec3From(const float v[3])
{
    return glm::vec3(v[0], v[1], v[2]);
}

void vec3To(const glm::vec3 & v, float out[3])
{
    out[0] = v.x;
    out[1] = v.y;
    out[2] = v.z;
}

// Field by field rather than a memcpy: the C struct's order (grouped for
// its own padding) need not follow sabercat::Light's.
sabercat::Light toLight(const SabercatLight & l)
{
    sabercat::Light out;
    out.type = static_cast<sabercat::LightType>(l.type);
    out.castsShadows = l.casts_shadows != 0 ? 1u : 0u;
    out.intensity = l.intensity;
    out.color = vec3From(l.color);
    out.position = vec3From(l.position);
    out.direction = vec3From(l.direction);
    out.axisX = vec3From(l.axis_x);
    out.axisY = vec3From(l.axis_y);
    out.halfWidth = l.half_width;
    out.halfHeight = l.half_height;
    return out;
}

// The depth_layout values a renderer accepts: SabercatDepthLayout's.
bool isDepthLayout(uint32_t value)
{
    return value == SABERCAT_DEPTH_LAYOUT_NONE || value == SABERCAT_DEPTH_LAYOUT_DEPTH
           || value == SABERCAT_DEPTH_LAYOUT_DEPTH_COVERAGE
           || value == SABERCAT_DEPTH_LAYOUT_DEPTH_COVERAGE_FRONT;
}

// The depth layout a caller's desc asks for; a struct from before depth_layout
// was appended reads as NONE.
uint32_t requestedDepthLayout(const SabercatRendererDesc * desc)
{
    if (!SABERCAT_HAS_FIELD(desc, SabercatRendererDesc, depth_layout))
    {
        return SABERCAT_DEPTH_LAYOUT_NONE;
    }
    return desc->depth_layout;
}

// Writes a fully built output struct back, only as far as the caller's
// struct_size reaches -- an older, smaller struct is never overrun.
template <typename T>
void writeOutput(T * out, T value)
{
    const uint32_t callerSize = out->struct_size;
    value.struct_size = callerSize;
    std::memcpy(out, &value, std::min<size_t>(callerSize, sizeof(T)));
}

} // namespace

extern "C" {

uint32_t sabercat_api_version(void)
{
    return SABERCAT_C_API_VERSION;
}

const char * sabercat_last_error(void)
{
    return g_lastError.c_str();
}

SabercatResult sabercat_renderer_create(const SabercatRendererDesc * desc,
                                        SabercatRenderer ** outRenderer)
{
    if (!desc || !outRenderer || desc->struct_size < kRendererDescV1Size)
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                    "sabercat_renderer_create: null argument or struct_size too small");
    }
    *outRenderer = nullptr;
    if (desc->width == 0 || desc->height == 0 || !desc->shader_dir || !desc->shader_dir[0])
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                    "sabercat_renderer_create: width/height must be > 0 and shader_dir set");
    }
    const uint32_t depthLayout = requestedDepthLayout(desc);
    if (!isDepthLayout(depthLayout))
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                    "sabercat_renderer_create: depth_layout must be 0, 1, 2 or 4");
    }
    return guarded("sabercat_renderer_create",
                   [&]
                   {
                       auto handle = std::make_unique<SabercatRenderer>();
                       handle->renderer = std::make_unique<sabercat::Renderer>(
                           desc->width, desc->height,
                           static_cast<sabercat::DepthLayout>(depthLayout));
                       handle->width = desc->width;
                       handle->height = desc->height;
                       // Headless: no instance extensions, no surface.
                       if (!handle->renderer->initialize({}, desc->enable_validation != 0,
                                                         desc->shader_dir))
                       {
                           return fail(SABERCAT_ERROR_RUNTIME,
                                       "sabercat_renderer_create: Vulkan/renderer "
                                       "initialization failed (see the log)");
                       }
                       *outRenderer = handle.release();
                       return SABERCAT_OK;
                   });
}

SabercatResult sabercat_renderer_create_borrowed(const SabercatRendererDesc * desc,
                                                 const SabercatDeviceInfo * device,
                                                 SabercatRenderer ** outRenderer)
{
    if (!desc || !device || !outRenderer || desc->struct_size < kRendererDescV1Size
        || !SABERCAT_HAS_FIELD(device, SabercatDeviceInfo, float32_atomic_add))
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                    "sabercat_renderer_create_borrowed: null argument or struct_size too small");
    }
    *outRenderer = nullptr;
    if (desc->width == 0 || desc->height == 0 || !desc->shader_dir || !desc->shader_dir[0])
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                    "sabercat_renderer_create_borrowed: width/height must be > 0 and shader_dir "
                    "set");
    }
    const uint32_t depthLayout = requestedDepthLayout(desc);
    if (!isDepthLayout(depthLayout))
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                    "sabercat_renderer_create_borrowed: depth_layout must be 0, 1, 2 or 4");
    }
    return guarded("sabercat_renderer_create_borrowed",
                   [&]
                   {
                       sabercat::VulkanContext context;
                       context.instance = static_cast<VkInstance>(device->instance);
                       context.physicalDevice =
                           static_cast<VkPhysicalDevice>(device->physical_device);
                       context.device = static_cast<VkDevice>(device->device);
                       context.computeQueue = static_cast<VkQueue>(device->compute_queue);
                       context.computeFamily = device->compute_queue_family;
                       sabercat::EnabledDeviceFeatures enabled;
                       enabled.apiVersion = device->vulkan_api_version;
                       enabled.shaderBufferFloat32AtomicAdd = device->float32_atomic_add != 0;
                       // A struct from before the ray query field cannot declare it.
                       enabled.rayQuery = SABERCAT_HAS_FIELD(device, SabercatDeviceInfo, ray_query)
                                          && device->ray_query != 0;

                       auto handle = std::make_unique<SabercatRenderer>();
                       handle->renderer = std::make_unique<sabercat::Renderer>(
                           desc->width, desc->height,
                           static_cast<sabercat::DepthLayout>(depthLayout));
                       handle->width = desc->width;
                       handle->height = desc->height;
                       handle->renderer->initializeBorrowed(context, enabled, desc->shader_dir);
                       *outRenderer = handle.release();
                       return SABERCAT_OK;
                   });
}

SabercatResult sabercat_renderer_set_gaussian_views(SabercatRenderer * renderer,
                                                    const SabercatBorrowedGaussians * scene)
{
    if (!renderer || !scene || scene->struct_size < kBorrowedGaussiansV1Size)
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                    "sabercat_renderer_set_gaussian_views: null argument or struct_size too small");
    }
    return guarded("sabercat_renderer_set_gaussian_views",
                   [&]
                   {
                       sabercat::BorrowedGaussians b;
                       b.parameters.means = fromView(scene->parameters.means);
                       b.parameters.scales = fromView(scene->parameters.scales);
                       b.parameters.rotations = fromView(scene->parameters.rotations);
                       b.parameters.opacity = fromView(scene->parameters.opacities);
                       b.parameters.shCoefficients = fromView(scene->parameters.sh_coefficients);
                       b.parameters.gaussianCount = scene->count;
                       b.parameters.shDegree = scene->sh_degree;
                       b.gradients.means = fromView(scene->gradients.means);
                       b.gradients.scales = fromView(scene->gradients.scales);
                       b.gradients.rotations = fromView(scene->gradients.rotations);
                       b.gradients.opacity = fromView(scene->gradients.opacities);
                       b.gradients.shCoefficients = fromView(scene->gradients.sh_coefficients);
                       b.gradients.screenMeans = fromView(scene->gradients.screen_means);
                       if (SABERCAT_HAS_FIELD(scene, SabercatBorrowedGaussians, relight))
                       {
                           b.relight.normals = fromView(scene->relight.normals);
                           b.relight.baseColorRoughness =
                               fromView(scene->relight.base_color_roughness);
                           b.relight.metallic = fromView(scene->relight.metallic);
                           b.relight.visibility = fromView(scene->relight.visibility);
                       }
                       b.antialiased = scene->antialiased != 0;
                       renderer->renderer->borrowGaussians(b);
                       renderer->hasForward = false;
                       renderer->hasBackward = false;
                       return SABERCAT_OK;
                   });
}

SabercatResult sabercat_renderer_set_image_size(SabercatRenderer * renderer, uint32_t width,
                                                uint32_t height)
{
    if (!renderer || width == 0 || height == 0)
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                    "sabercat_renderer_set_image_size: null renderer or zero size");
    }
    return guarded("sabercat_renderer_set_image_size",
                   [&]
                   {
                       renderer->renderer->setImageSize(width, height);
                       renderer->width = width;
                       renderer->height = height;
                       renderer->hasForward = false;
                       return SABERCAT_OK;
                   });
}

void sabercat_renderer_destroy(SabercatRenderer * renderer)
{
    if (!renderer)
    {
        return;
    }
    try
    {
        renderer->renderer->shutdown();
    }
    catch (...)
    {
        // Destruction must not throw across the ABI; nothing useful to report.
    }
    delete renderer;
}

SabercatResult sabercat_renderer_load_scene(SabercatRenderer * renderer, const char * path)
{
    if (!renderer || !path)
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT, "sabercat_renderer_load_scene: null argument");
    }
    return guarded("sabercat_renderer_load_scene",
                   [&]
                   {
                       renderer->renderer->loadScene(path);
                       renderer->hasForward = false;
                       renderer->hasBackward = false;
                       return SABERCAT_OK;
                   });
}

SabercatResult sabercat_renderer_set_gaussians(SabercatRenderer * renderer,
                                               const SabercatGaussians * gaussians)
{
    if (!renderer || !gaussians || gaussians->struct_size < kGaussiansV1Size
        || gaussians->count == 0 || !gaussianArraysPresent(*gaussians) || gaussians->sh_degree > 3)
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                    "sabercat_renderer_set_gaussians: null/short struct, count 0, null array, "
                    "or sh_degree > 3");
    }
    return guarded("sabercat_renderer_set_gaussians",
                   [&]
                   {
                       sabercat::GaussianHostInput input;
                       input.count = gaussians->count;
                       input.means = gaussians->means;
                       input.scales = gaussians->scales;
                       input.rotations = gaussians->rotations;
                       input.opacities = gaussians->opacities;
                       input.shCoefficients = gaussians->sh_coefficients;
                       input.shDegree = gaussians->sh_degree;
                       input.antialiased = gaussians->antialiased != 0;
                       if (SABERCAT_HAS_FIELD(gaussians, SabercatGaussians, capacity))
                       {
                           input.capacity = gaussians->capacity;
                       }
                       // reserved0 is deliberately not read: an older caller's
                       // tail padding occupies it.
                       if (SABERCAT_HAS_FIELD(gaussians, SabercatGaussians, visibility))
                       {
                           input.normals = gaussians->normals;
                           input.baseColorRoughness = gaussians->base_color_roughness;
                           input.metallic = gaussians->metallic;
                           input.visibility = gaussians->visibility;
                       }
                       renderer->renderer->setGaussians(input);
                       renderer->hasForward = false;
                       renderer->hasBackward = false;
                       return SABERCAT_OK;
                   });
}

SabercatResult sabercat_renderer_get_gaussians(SabercatRenderer * renderer, SabercatGaussians * out)
{
    if (!renderer || !out || out->struct_size < kGaussiansV1Size
        || !gaussianArraysPresent(*out))
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                    "sabercat_renderer_get_gaussians: null/short struct or null array");
    }
    if (renderer->renderer->gaussianCount() == 0)
    {
        return fail(SABERCAT_ERROR_INVALID_STATE, "sabercat_renderer_get_gaussians: no scene");
    }
    if (out->count != renderer->renderer->gaussianCount())
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                    "sabercat_renderer_get_gaussians: count does not match the scene");
    }
    return guarded("sabercat_renderer_get_gaussians",
                   [&]
                   {
                       renderer->renderer->readGaussians(toOutput(*out));
                       return SABERCAT_OK;
                   });
}

SabercatResult sabercat_renderer_get_gaussian_count(SabercatRenderer * renderer,
                                                    uint32_t * outCount)
{
    if (!renderer || !outCount)
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                    "sabercat_renderer_get_gaussian_count: null argument");
    }
    *outCount = renderer->renderer->gaussianCount();
    return SABERCAT_OK;
}

SabercatResult sabercat_renderer_clear_scene(SabercatRenderer * renderer)
{
    if (!renderer)
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT, "sabercat_renderer_clear_scene: null argument");
    }
    return guarded("sabercat_renderer_clear_scene",
                   [&]
                   {
                       renderer->renderer->clearScene();
                       renderer->hasForward = false;
                       renderer->hasBackward = false;
                       renderer->hasLitForward = false;
                       return SABERCAT_OK;
                   });
}

SabercatResult sabercat_renderer_set_camera(SabercatRenderer * renderer,
                                            const SabercatCamera * camera)
{
    if (!renderer || !camera || camera->struct_size < sizeof(SabercatCamera)
        || !(camera->fx > 0.0f) || !(camera->fy > 0.0f))
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                    "sabercat_renderer_set_camera: null/short struct or non-positive focal length");
    }
    return guarded("sabercat_renderer_set_camera",
                   [&]
                   {
                       sabercat::Camera c;
                       c.width = renderer->width;
                       c.height = renderer->height;
                       c.fx = camera->fx;
                       c.fy = camera->fy;
                       c.cx = camera->cx;
                       c.cy = camera->cy;
                       c.cameraToWorld = glm::make_mat4(camera->camera_to_world);
                       renderer->renderer->setCamera(c);
                       renderer->hasCamera = true;
                       return SABERCAT_OK;
                   });
}

SabercatResult sabercat_renderer_set_sh_degree(SabercatRenderer * renderer, uint32_t degree)
{
    if (!renderer || degree > 3)
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                    "sabercat_renderer_set_sh_degree: null renderer or degree > 3");
    }
    renderer->renderer->setMaxSHDegree(degree);
    return SABERCAT_OK;
}

SabercatResult sabercat_renderer_set_antialiased(SabercatRenderer * renderer, int32_t antialiased)
{
    if (!renderer)
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT, "sabercat_renderer_set_antialiased: null");
    }
    if (renderer->renderer->gaussianCount() == 0)
    {
        return fail(SABERCAT_ERROR_INVALID_STATE, "sabercat_renderer_set_antialiased: no scene");
    }
    renderer->renderer->setAntialiased(antialiased != 0);
    return SABERCAT_OK;
}

SabercatResult sabercat_renderer_forward(SabercatRenderer * renderer)
{
    if (!renderer)
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT, "sabercat_renderer_forward: null");
    }
    if (renderer->renderer->gaussianCount() == 0 || !renderer->hasCamera)
    {
        return fail(SABERCAT_ERROR_INVALID_STATE,
                    "sabercat_renderer_forward: set a scene and a camera first");
    }
    return guarded("sabercat_renderer_forward",
                   [&]
                   {
                       renderer->hasForward = false;
                       renderer->hasLitForward = false;
                       renderer->renderer->forward(sabercat::RenderMode::Differentiable);
                       renderer->hasForward = true;
                       renderer->hasLitForward = renderer->renderer->lightingEnabled();
                       return SABERCAT_OK;
                   });
}

SabercatResult sabercat_renderer_read_image(SabercatRenderer * renderer, float * rgba,
                                            size_t floatCount)
{
    if (!renderer || !rgba
        || floatCount != static_cast<size_t>(renderer->width) * renderer->height * 4)
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                    "sabercat_renderer_read_image: null argument or float_count != width*height*4");
    }
    if (!renderer->hasForward)
    {
        return fail(SABERCAT_ERROR_INVALID_STATE, "sabercat_renderer_read_image: no forward yet");
    }
    return guarded("sabercat_renderer_read_image",
                   [&]
                   {
                       std::vector<float> image = renderer->renderer->readImage();
                       std::memcpy(rgba, image.data(), floatCount * sizeof(float));
                       return SABERCAT_OK;
                   });
}

SabercatResult sabercat_renderer_backward(SabercatRenderer * renderer, const float * dlDImageRgba,
                                          size_t floatCount)
{
    if (!renderer || !dlDImageRgba
        || floatCount != static_cast<size_t>(renderer->width) * renderer->height * 4)
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                    "sabercat_renderer_backward: null argument or float_count != width*height*4");
    }
    if (!renderer->hasForward)
    {
        return fail(SABERCAT_ERROR_INVALID_STATE,
                    "sabercat_renderer_backward: no forward of the current scene to differentiate");
    }
    if (!renderer->renderer->backwardSupported())
    {
        return fail(SABERCAT_ERROR_UNSUPPORTED,
                    "sabercat_renderer_backward: device lacks fp32 buffer atomic add "
                    "(VK_EXT_shader_atomic_float)");
    }
    return guarded("sabercat_renderer_backward",
                   [&]
                   {
                       renderer->renderer->backward(dlDImageRgba);
                       renderer->hasBackward = true;
                       return SABERCAT_OK;
                   });
}

SabercatResult sabercat_renderer_backward_view(SabercatRenderer * renderer,
                                              const SabercatBufferView * dlDImage)
{
    if (!renderer || !dlDImage || dlDImage->buffer == 0 || dlDImage->stride != 16
        || dlDImage->size < static_cast<uint64_t>(renderer->width) * renderer->height * 16)
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                    "sabercat_renderer_backward_view: null view, stride != 16, or size < "
                    "width*height*16");
    }
    if (!renderer->hasForward)
    {
        return fail(SABERCAT_ERROR_INVALID_STATE,
                    "sabercat_renderer_backward_view: no forward of the current scene to "
                    "differentiate");
    }
    if (!renderer->renderer->backwardSupported())
    {
        return fail(SABERCAT_ERROR_UNSUPPORTED,
                    "sabercat_renderer_backward_view: device lacks fp32 buffer atomic add "
                    "(VK_EXT_shader_atomic_float)");
    }
    return guarded("sabercat_renderer_backward_view",
                   [&]
                   {
                       sabercat::BufferView view;
                       view.buffer = reinterpret_cast<VkBuffer>(dlDImage->buffer);
                       view.offset = dlDImage->offset;
                       view.size = dlDImage->size;
                       view.stride = dlDImage->stride;
                       renderer->renderer->backward(view);
                       renderer->hasBackward = true;
                       return SABERCAT_OK;
                   });
}

SabercatResult sabercat_renderer_backward_depth(SabercatRenderer * renderer,
                                               const float * dlDImageRgba, size_t imageFloatCount,
                                               const float * dlDDepth, size_t depthFloatCount)
{
    if (!renderer || !dlDImageRgba
        || imageFloatCount != static_cast<size_t>(renderer->width) * renderer->height * 4)
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                    "sabercat_renderer_backward_depth: null argument or image_float_count != "
                    "width*height*4");
    }
    if (!dlDDepth)
    {
        if (depthFloatCount != 0)
        {
            return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                        "sabercat_renderer_backward_depth: depth_float_count must be 0 when "
                        "dl_ddepth is NULL");
        }
    }
    else
    {
        const uint32_t layout = static_cast<uint32_t>(renderer->renderer->depthLayout());
        if (layout == SABERCAT_DEPTH_LAYOUT_NONE)
        {
            return fail(SABERCAT_ERROR_INVALID_STATE,
                        "sabercat_renderer_backward_depth: the renderer was created with "
                        "depth_layout NONE");
        }
        if (depthFloatCount != static_cast<size_t>(renderer->width) * renderer->height * layout)
        {
            return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                        "sabercat_renderer_backward_depth: depth_float_count != "
                        "width*height*n (n = depth_layout)");
        }
    }
    if (!renderer->hasForward)
    {
        return fail(SABERCAT_ERROR_INVALID_STATE,
                    "sabercat_renderer_backward_depth: no forward of the current scene to "
                    "differentiate");
    }
    if (!renderer->renderer->backwardSupported())
    {
        return fail(SABERCAT_ERROR_UNSUPPORTED,
                    "sabercat_renderer_backward_depth: device lacks fp32 buffer atomic add "
                    "(VK_EXT_shader_atomic_float)");
    }
    return guarded("sabercat_renderer_backward_depth",
                   [&]
                   {
                       renderer->renderer->backward(dlDImageRgba, dlDDepth);
                       renderer->hasBackward = true;
                       return SABERCAT_OK;
                   });
}

SabercatResult sabercat_renderer_backward_depth_view(SabercatRenderer * renderer,
                                                     const SabercatBufferView * dlDImage,
                                                     const SabercatBufferView * dlDDepth)
{
    if (!renderer || !dlDImage || dlDImage->buffer == 0 || dlDImage->stride != 16
        || dlDImage->size < static_cast<uint64_t>(renderer->width) * renderer->height * 16)
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                    "sabercat_renderer_backward_depth_view: null view, stride != 16, or size < "
                    "width*height*16");
    }
    const bool depthGiven = dlDDepth != nullptr && dlDDepth->buffer != 0;
    if (depthGiven)
    {
        const uint32_t layout = static_cast<uint32_t>(renderer->renderer->depthLayout());
        if (layout == SABERCAT_DEPTH_LAYOUT_NONE)
        {
            return fail(SABERCAT_ERROR_INVALID_STATE,
                        "sabercat_renderer_backward_depth_view: the renderer was created with "
                        "depth_layout NONE");
        }
        const uint64_t depthBytes =
            static_cast<uint64_t>(renderer->width) * renderer->height * layout * 4;
        if (dlDDepth->stride != layout * 4 || dlDDepth->size < depthBytes)
        {
            return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                        "sabercat_renderer_backward_depth_view: depth stride must be 4*n and size "
                        ">= width*height*n*4");
        }
    }
    if (!renderer->hasForward)
    {
        return fail(SABERCAT_ERROR_INVALID_STATE,
                    "sabercat_renderer_backward_depth_view: no forward of the current scene to "
                    "differentiate");
    }
    if (!renderer->renderer->backwardSupported())
    {
        return fail(SABERCAT_ERROR_UNSUPPORTED,
                    "sabercat_renderer_backward_depth_view: device lacks fp32 buffer atomic add "
                    "(VK_EXT_shader_atomic_float)");
    }
    return guarded("sabercat_renderer_backward_depth_view",
                   [&]
                   {
                       sabercat::BufferView image;
                       image.buffer = reinterpret_cast<VkBuffer>(dlDImage->buffer);
                       image.offset = dlDImage->offset;
                       image.size = dlDImage->size;
                       image.stride = dlDImage->stride;
                       sabercat::BufferView depth;
                       if (depthGiven)
                       {
                           depth.buffer = reinterpret_cast<VkBuffer>(dlDDepth->buffer);
                           depth.offset = dlDDepth->offset;
                           depth.size = dlDDepth->size;
                           depth.stride = dlDDepth->stride;
                       }
                       renderer->renderer->backward(image, depth);
                       renderer->hasBackward = true;
                       return SABERCAT_OK;
                   });
}

SabercatResult sabercat_renderer_read_gradients(SabercatRenderer * renderer, SabercatGradients * out)
{
    if (!renderer || !out || out->struct_size < sizeof(SabercatGradients) || !out->means
        || !out->scales || !out->rotations || !out->opacities || !out->sh_coefficients)
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                    "sabercat_renderer_read_gradients: null/short struct or null array "
                    "(screen_means may be NULL)");
    }
    if (!renderer->hasBackward)
    {
        return fail(SABERCAT_ERROR_INVALID_STATE, "sabercat_renderer_read_gradients: no backward yet");
    }
    if (out->count != renderer->renderer->gaussianCount())
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                    "sabercat_renderer_read_gradients: count does not match the scene");
    }
    return guarded("sabercat_renderer_read_gradients",
                   [&]
                   {
                       sabercat::GaussianGradients g = renderer->renderer->readGradients();
                       const size_t n = out->count;
                       std::memcpy(out->means, g.means.data(), n * 4 * sizeof(float));
                       std::memcpy(out->scales, g.scales.data(), n * 4 * sizeof(float));
                       std::memcpy(out->rotations, g.rotations.data(), n * 4 * sizeof(float));
                       std::memcpy(out->opacities, g.opacities.data(), n * sizeof(float));
                       std::memcpy(out->sh_coefficients, g.shCoefficients.data(),
                                   g.shCoefficients.size() * 4 * sizeof(float));
                       if (out->screen_means)
                       {
                           std::memcpy(out->screen_means, g.screenMeans.data(),
                                       n * 4 * sizeof(float));
                       }
                       return SABERCAT_OK;
                   });
}

SabercatResult sabercat_renderer_get_gaussian_views(SabercatRenderer * renderer,
                                                    SabercatGaussianViews * out)
{
    if (!renderer || !out)
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT, "sabercat_renderer_get_gaussian_views: null");
    }
    if (renderer->renderer->gaussianCount() == 0)
    {
        return fail(SABERCAT_ERROR_INVALID_STATE, "sabercat_renderer_get_gaussian_views: no scene");
    }
    return guarded("sabercat_renderer_get_gaussian_views",
                   [&]
                   {
                       sabercat::CanonicalGaussianBufferViews v = renderer->renderer->gaussianViews();
                       out->means = toView(v.means);
                       out->scales = toView(v.scales);
                       out->rotations = toView(v.rotations);
                       out->opacities = toView(v.opacity);
                       out->sh_coefficients = toView(v.shCoefficients);
                       out->count = v.gaussianCount;
                       return SABERCAT_OK;
                   });
}

SabercatResult sabercat_renderer_get_gradient_views(SabercatRenderer * renderer,
                                                    SabercatGradientViews * out)
{
    if (!renderer || !out)
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT, "sabercat_renderer_get_gradient_views: null");
    }
    if (renderer->renderer->gaussianCount() == 0)
    {
        return fail(SABERCAT_ERROR_INVALID_STATE, "sabercat_renderer_get_gradient_views: no scene");
    }
    return guarded("sabercat_renderer_get_gradient_views",
                   [&]
                   {
                       sabercat::CanonicalGaussianGradientViews v =
                           renderer->renderer->gradientViews();
                       out->means = toView(v.means);
                       out->scales = toView(v.scales);
                       out->rotations = toView(v.rotations);
                       out->opacities = toView(v.opacity);
                       out->sh_coefficients = toView(v.shCoefficients);
                       out->screen_means = toView(v.screenMeans);
                       out->count = v.gaussianCount;
                       return SABERCAT_OK;
                   });
}

SabercatResult sabercat_renderer_get_device_info(SabercatRenderer * renderer,
                                                 SabercatDeviceInfo * out)
{
    if (!renderer || !out || !SABERCAT_HAS_FIELD(out, SabercatDeviceInfo, compute_queue))
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                    "sabercat_renderer_get_device_info: null argument or struct_size too small");
    }
    sabercat::Renderer & r = *renderer->renderer;
    SabercatDeviceInfo info{};
    info.instance = r.instance();
    info.physical_device = r.physicalDevice();
    info.device = r.device();
    info.compute_queue = r.computeQueue();
    info.compute_queue_family = r.computeQueueFamily();
    info.vulkan_api_version = r.enabledDeviceFeatures().apiVersion;
    info.float32_atomic_add = r.deviceFeatures().shaderBufferFloat32AtomicAdd ? 1 : 0;
    // What was enabled (Sabercat-owned device) or declared (borrowed device) -- the form
    // create_borrowed reads back.
    info.ray_query = r.enabledDeviceFeatures().rayQuery ? 1 : 0;
    info.reserved0 = 0;
    writeOutput(out, info); // a 48-byte caller gets neither later field
    return SABERCAT_OK;
}

SabercatResult sabercat_renderer_get_image_view(SabercatRenderer * renderer,
                                               SabercatBufferView * out)
{
    if (!renderer || !out)
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT, "sabercat_renderer_get_image_view: null");
    }
    if (!renderer->hasForward)
    {
        return fail(SABERCAT_ERROR_INVALID_STATE,
                    "sabercat_renderer_get_image_view: no forward of the current scene");
    }
    return guarded("sabercat_renderer_get_image_view",
                   [&]
                   {
                       *out = toView(renderer->renderer->imageView());
                       return SABERCAT_OK;
                   });
}

SabercatResult sabercat_renderer_read_depth(SabercatRenderer * renderer, float * values,
                                            size_t floatCount)
{
    if (!renderer || !values)
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT, "sabercat_renderer_read_depth: null argument");
    }
    const uint32_t layout = static_cast<uint32_t>(renderer->renderer->depthLayout());
    if (layout == SABERCAT_DEPTH_LAYOUT_NONE)
    {
        return fail(SABERCAT_ERROR_INVALID_STATE,
                    "sabercat_renderer_read_depth: the renderer was created with depth_layout "
                    "NONE");
    }
    if (floatCount != static_cast<size_t>(renderer->width) * renderer->height * layout)
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                    "sabercat_renderer_read_depth: float_count != width*height*n (n = "
                    "depth_layout)");
    }
    if (!renderer->hasForward)
    {
        return fail(SABERCAT_ERROR_INVALID_STATE,
                    "sabercat_renderer_read_depth: no forward of the current scene");
    }
    return guarded("sabercat_renderer_read_depth",
                   [&]
                   {
                       std::vector<float> depth = renderer->renderer->readDepth();
                       std::memcpy(values, depth.data(), floatCount * sizeof(float));
                       return SABERCAT_OK;
                   });
}

SabercatResult sabercat_renderer_get_depth_view(SabercatRenderer * renderer,
                                                SabercatBufferView * out)
{
    if (!renderer || !out)
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT, "sabercat_renderer_get_depth_view: null");
    }
    if (renderer->renderer->depthLayout() == sabercat::DepthLayout::None)
    {
        return fail(SABERCAT_ERROR_INVALID_STATE,
                    "sabercat_renderer_get_depth_view: the renderer was created with depth_layout "
                    "NONE");
    }
    if (!renderer->hasForward)
    {
        return fail(SABERCAT_ERROR_INVALID_STATE,
                    "sabercat_renderer_get_depth_view: no forward of the current scene");
    }
    return guarded("sabercat_renderer_get_depth_view",
                   [&]
                   {
                       *out = toView(renderer->renderer->depthView());
                       return SABERCAT_OK;
                   });
}

SabercatResult sabercat_renderer_get_gaussian_storage(SabercatRenderer * renderer,
                                                      SabercatGaussianStorage * out)
{
    if (!renderer || !out || !SABERCAT_HAS_FIELD(out, SabercatGaussianStorage, generation))
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                    "sabercat_renderer_get_gaussian_storage: null argument or struct_size too "
                    "small");
    }
    SabercatGaussianStorage storage{};
    storage.count = renderer->renderer->gaussianCount();
    storage.capacity = renderer->renderer->gaussianCapacity();
    storage.generation = renderer->renderer->storageGeneration();
    writeOutput(out, storage);
    return SABERCAT_OK;
}

SabercatResult sabercat_renderer_reserve_gaussians(SabercatRenderer * renderer, uint32_t capacity,
                                                   int32_t * outGrew)
{
    if (!renderer)
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT, "sabercat_renderer_reserve_gaussians: null");
    }
    if (renderer->renderer->gaussianCount() == 0)
    {
        return fail(SABERCAT_ERROR_INVALID_STATE, "sabercat_renderer_reserve_gaussians: no scene");
    }
    return guarded("sabercat_renderer_reserve_gaussians",
                   [&]
                   {
                       const bool grew = renderer->renderer->reserveGaussians(capacity);
                       if (grew)
                       {
                           // Renderer invalidated its forward record; the
                           // gradient buffers were reallocated.
                           renderer->hasForward = false;
                           renderer->hasBackward = false;
                       }
                       if (outGrew)
                       {
                           *outGrew = grew ? 1 : 0;
                       }
                       return SABERCAT_OK;
                   });
}

SabercatResult sabercat_renderer_set_gaussian_count(SabercatRenderer * renderer, uint32_t count)
{
    if (!renderer)
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT, "sabercat_renderer_set_gaussian_count: null");
    }
    if (renderer->renderer->gaussianCount() == 0)
    {
        return fail(SABERCAT_ERROR_INVALID_STATE, "sabercat_renderer_set_gaussian_count: no scene");
    }
    if (count == 0 || count > renderer->renderer->gaussianCapacity())
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                    "sabercat_renderer_set_gaussian_count: count must be in [1, capacity] -- "
                    "reserve_gaussians first");
    }
    return guarded("sabercat_renderer_set_gaussian_count",
                   [&]
                   {
                       renderer->renderer->setGaussianCount(count);
                       renderer->hasForward = false;
                       renderer->hasBackward = false;
                       return SABERCAT_OK;
                   });
}

// --- Relighting and shadows ---

SabercatResult sabercat_renderer_set_lighting_enabled(SabercatRenderer * renderer, int32_t enabled)
{
    if (!renderer)
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT, "sabercat_renderer_set_lighting_enabled: null");
    }
    renderer->renderer->setLightingEnabled(enabled != 0);
    return SABERCAT_OK;
}

SabercatResult sabercat_renderer_set_lights(SabercatRenderer * renderer, const SabercatLight * lights,
                                            uint32_t lightCount)
{
    if (!renderer || (lightCount > 0 && !lights) || lightCount > SABERCAT_MAX_LIGHT_COUNT)
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                    "sabercat_renderer_set_lights: null renderer, null lights with a count, or "
                    "more than SABERCAT_MAX_LIGHT_COUNT lights");
    }
    // A versioned array: the first element's struct_size is the stride.
    const uint32_t stride = lightCount > 0 ? lights[0].struct_size : 0;
    if (lightCount > 0 && (stride < sizeof(SabercatLight) || stride % alignof(SabercatLight) != 0))
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                    "sabercat_renderer_set_lights: lights[0].struct_size too small or not a "
                    "multiple of 4");
    }
    const auto lightAt = [&](uint32_t i) -> const SabercatLight &
    {
        return *reinterpret_cast<const SabercatLight *>(
            reinterpret_cast<const unsigned char *>(lights) + size_t(i) * stride);
    };
    for (uint32_t i = 0; i < lightCount; ++i)
    {
        if (lightAt(i).struct_size != stride || lightAt(i).type > SABERCAT_LIGHT_RECT)
        {
            return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                        "sabercat_renderer_set_lights: light " + std::to_string(i)
                            + " has a struct_size differing from lights[0]'s or an unknown type");
        }
    }
    return guarded("sabercat_renderer_set_lights",
                   [&]
                   {
                       std::vector<sabercat::Light> converted;
                       for (uint32_t i = 0; i < lightCount; ++i)
                       {
                           converted.push_back(toLight(lightAt(i)));
                       }
                       renderer->renderer->setLights(converted);
                       return SABERCAT_OK;
                   });
}

SabercatResult sabercat_renderer_set_environment(SabercatRenderer * renderer,
                                                 const SabercatEnvironment * environment)
{
    if (!renderer || !environment || environment->struct_size < sizeof(SabercatEnvironment))
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                    "sabercat_renderer_set_environment: null argument or struct_size too small");
    }
    sabercat::EnvironmentLight env;
    for (size_t k = 0; k < env.sh.size(); ++k)
    {
        env.sh[k] = vec3From(&environment->sh[k * 3]);
    }
    env.intensity = environment->intensity;
    renderer->renderer->setEnvironment(env);
    return SABERCAT_OK;
}

SabercatResult sabercat_renderer_set_lighting_settings(SabercatRenderer * renderer,
                                                       const SabercatLightingSettings * settings)
{
    if (!renderer || !settings || settings->struct_size < sizeof(SabercatLightingSettings)
        || settings->output > SABERCAT_LIGHTING_OUTPUT_DEPTH)
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                    "sabercat_renderer_set_lighting_settings: null/short struct or unknown output");
    }
    sabercat::LightingSettings s;
    s.background = vec3From(settings->background);
    s.exposure = settings->exposure;
    s.output = static_cast<sabercat::LightingOutput>(settings->output);
    s.rectSamples = settings->rect_samples;
    s.visibilityLight = settings->visibility_light;
    s.depthScale = settings->depth_scale;
    renderer->renderer->setLightingSettings(s);
    return SABERCAT_OK;
}

SabercatResult sabercat_renderer_get_lighting_settings(SabercatRenderer * renderer,
                                                       SabercatLightingSettings * out)
{
    if (!renderer || !out || out->struct_size < sizeof(SabercatLightingSettings))
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                    "sabercat_renderer_get_lighting_settings: null argument or struct_size too "
                    "small");
    }
    const sabercat::LightingSettings & s = renderer->renderer->lightingSettings();
    SabercatLightingSettings value{};
    value.output = static_cast<uint32_t>(s.output);
    value.rect_samples = s.rectSamples;
    value.visibility_light = s.visibilityLight;
    vec3To(s.background, value.background);
    value.exposure = s.exposure;
    value.depth_scale = s.depthScale;
    writeOutput(out, value);
    return SABERCAT_OK;
}

SabercatResult sabercat_renderer_set_shadow_settings(SabercatRenderer * renderer,
                                                     const SabercatShadowSettings * settings)
{
    if (!renderer || !settings || settings->struct_size < sizeof(SabercatShadowSettings)
        || settings->mode > SABERCAT_SHADOW_RAY_TRACED || settings->map_resolution == 0
        || settings->blocker_samples_per_axis == 0 || settings->filter_samples_per_axis == 0
        || settings->ray_samples_per_axis == 0 || !(settings->focus_radius >= 0.0f))
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                    "sabercat_renderer_set_shadow_settings: null/short struct, unknown mode, zero "
                    "resolution/sample count, or negative focus_radius");
    }
    // Checked here rather than left to forward(), so the caller can fall
    // back to PCSS before rendering anything.
    if (settings->mode == SABERCAT_SHADOW_RAY_TRACED
        && !renderer->renderer->enabledDeviceFeatures().rayQuery)
    {
        return fail(SABERCAT_ERROR_UNSUPPORTED,
                    "sabercat_renderer_set_shadow_settings: SABERCAT_SHADOW_RAY_TRACED needs ray "
                    "query (VK_KHR_ray_query + VK_KHR_acceleration_structure), which this device "
                    "does not have enabled -- use SABERCAT_SHADOW_PCSS");
    }
    sabercat::ShadowSettings s;
    s.focusCenter = vec3From(settings->focus_center);
    s.focusRadius = settings->focus_radius;
    s.mode = static_cast<sabercat::ShadowMode>(settings->mode);
    s.mapResolution = settings->map_resolution;
    s.blockerSamplesPerAxis = settings->blocker_samples_per_axis;
    s.filterSamplesPerAxis = settings->filter_samples_per_axis;
    s.pointLightRadius = settings->point_light_radius;
    s.sunAngularRadius = settings->sun_angular_radius;
    s.depthBias = settings->depth_bias;
    s.normalOffset = settings->normal_offset;
    s.maxFilterTexels = settings->max_filter_texels;
    s.raySamplesPerAxis = settings->ray_samples_per_axis;
    renderer->renderer->setShadowSettings(s);
    return SABERCAT_OK;
}

SabercatResult sabercat_renderer_get_shadow_settings(SabercatRenderer * renderer,
                                                     SabercatShadowSettings * out)
{
    if (!renderer || !out || out->struct_size < sizeof(SabercatShadowSettings))
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                    "sabercat_renderer_get_shadow_settings: null argument or struct_size too small");
    }
    const sabercat::ShadowSettings & s = renderer->renderer->shadowSettings();
    SabercatShadowSettings value{};
    value.mode = static_cast<uint32_t>(s.mode);
    value.map_resolution = s.mapResolution;
    value.blocker_samples_per_axis = s.blockerSamplesPerAxis;
    value.filter_samples_per_axis = s.filterSamplesPerAxis;
    value.ray_samples_per_axis = s.raySamplesPerAxis;
    vec3To(s.focusCenter, value.focus_center);
    value.focus_radius = s.focusRadius;
    value.point_light_radius = s.pointLightRadius;
    value.sun_angular_radius = s.sunAngularRadius;
    value.depth_bias = s.depthBias;
    value.normal_offset = s.normalOffset;
    value.max_filter_texels = s.maxFilterTexels;
    writeOutput(out, value);
    return SABERCAT_OK;
}

SabercatResult sabercat_renderer_read_lit_image(SabercatRenderer * renderer, float * rgba,
                                                size_t floatCount)
{
    if (!renderer || !rgba
        || floatCount != static_cast<size_t>(renderer->width) * renderer->height * 4)
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT,
                    "sabercat_renderer_read_lit_image: null argument or float_count != "
                    "width*height*4");
    }
    if (!renderer->hasForward || !renderer->hasLitForward)
    {
        return fail(SABERCAT_ERROR_INVALID_STATE,
                    "sabercat_renderer_read_lit_image: no forward of the current scene with "
                    "lighting enabled");
    }
    return guarded("sabercat_renderer_read_lit_image",
                   [&]
                   {
                       std::vector<float> image = renderer->renderer->readLitImage();
                       if (image.size() < floatCount)
                       {
                           return fail(SABERCAT_ERROR_RUNTIME,
                                       "sabercat_renderer_read_lit_image: lit image smaller "
                                       "than width*height*4");
                       }
                       std::memcpy(rgba, image.data(), floatCount * sizeof(float));
                       return SABERCAT_OK;
                   });
}

SabercatResult sabercat_renderer_get_lit_image_view(SabercatRenderer * renderer,
                                                   SabercatBufferView * out)
{
    if (!renderer || !out)
    {
        return fail(SABERCAT_ERROR_INVALID_ARGUMENT, "sabercat_renderer_get_lit_image_view: null");
    }
    if (!renderer->hasForward || !renderer->hasLitForward)
    {
        return fail(SABERCAT_ERROR_INVALID_STATE,
                    "sabercat_renderer_get_lit_image_view: no forward of the current scene with "
                    "lighting enabled");
    }
    return guarded("sabercat_renderer_get_lit_image_view",
                   [&]
                   {
                       *out = toView(renderer->renderer->litImageView());
                       return SABERCAT_OK;
                   });
}

} // extern "C"
