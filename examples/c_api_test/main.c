// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

/*
 * C ABI round trip, in plain C (C language mode, no C++ anywhere in
 * this translation unit), linked against the sabercat_c shared library -- so
 * it crosses the same DLL boundary a ctypes/cffi binding would.
 *
 * Checks:
 *  - error paths: null/short arguments -> INVALID_ARGUMENT, out-of-order
 *    calls -> INVALID_STATE, each with a non-empty sabercat_last_error();
 *  - set_gaussians -> get_gaussians is bit-exact;
 *  - forward -> read_image renders something;
 *  - backward -> read_gradients is wired to the same pipeline: a central
 *    finite difference of the opacity, taken entirely THROUGH the ABI
 *    (set_gaussians / forward / read_image), agrees with the analytic
 *    gradient. This is a wiring check, not a precision check (that is
 *    examples/backward_test's job, in double, with frozen branches): it runs
 *    in fp32 with a large step and lets alpha cross the 1/255 contour on a
 *    few pixels, so the tolerance is a loose 5%;
 *  - zero-copy views report real buffers with the canonical strides;
 *  - get_image_view is INVALID_STATE before a forward, then a real
 *    buffer of width*height*16 bytes; backward_view fed that image view
 *    (dL/dImage = I) matches host backward fed read_image's copy;
 *  - create_borrowed builds a second renderer on the first one's
 *    device (from get_device_info), set_gaussian_views has it render the
 *    first one's buffers at another size, bit-identical to the first after
 *    set_image_size; malformed declarations and views are INVALID_ARGUMENT.
 *  - relighting: with relight streams from set_gaussians (base color
 *    = DC color, roughness 1, metallic 0, normal +Z), the lit image matches
 *    a closed-form expectation for a directional light along the view axis
 *    and, separately, a uniform environment; the lit image view feeds
 *    backward_view exactly like read_lit_image's copy; the same streams
 *    borrowed through SabercatBorrowedGaussians::relight render
 *    identically; PCSS and (with ray query) ray-traced shadows darken an
 *    occluder's shadow and nothing else; casts_shadows 0 is exactly
 *    unshadowed; RAY_TRACED without ray query is UNSUPPORTED. The 48-byte
 *    (pre-relight) SabercatDeviceInfo still works for get_device_info and
 *    create_borrowed (declaring no ray query), and ray_query is reported;
 *  - depth output: a 32-byte desc reads depth_layout as NONE, layout 3 is
 *    refused, and layout 2's read_depth / get_depth_view match read_image's
 *    alpha bit for bit.
 */
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sabercat/c_api.h"

#include "../common/color_space.h"

#define WIDTH 48
#define HEIGHT 32
#define COUNT 3
/* WIDTH/HEIGHT are the default image size of the first renderer. */
#define COEFFS 16

static int g_failures = 0;

#define CHECK(cond, msg)                                                                           \
    do                                                                                             \
    {                                                                                              \
        if (!(cond))                                                                               \
        {                                                                                          \
            fprintf(stderr, "FAIL: %s (last error: %s)\n", msg, sabercat_last_error());           \
            ++g_failures;                                                                          \
        }                                                                                          \
    } while (0)

static float g_means[COUNT * 4], g_scales[COUNT * 4], g_rotations[COUNT * 4], g_opacities[COUNT],
    g_sh[COUNT * COEFFS * 4];

static SabercatGaussians gaussiansDesc(float * means, float * scales, float * rotations,
                                       float * opacities, float * sh)
{
    SabercatGaussians g;
    memset(&g, 0, sizeof(g));
    g.struct_size = sizeof(g);
    g.count = COUNT;
    g.means = means;
    g.scales = scales;
    g.rotations = rotations;
    g.opacities = opacities;
    g.sh_coefficients = sh;
    g.sh_degree = 3;
    g.antialiased = 0;
    return g;
}

/* Worst |a-b| relative to max |b|. */
static double relativeError(const float * a, const float * b, size_t n)
{
    double worst = 0.0, scale = 0.0;
    size_t i;
    for (i = 0; i < n; ++i)
    {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > worst)
        {
            worst = d;
        }
        if (fabs((double)b[i]) > scale)
        {
            scale = fabs((double)b[i]);
        }
    }
    return scale > 0.0 ? worst / scale : worst;
}

/* ---- Relighting ----
 *
 * Scene: a GRID_X x GRID_Y grid of flat Gaussians in the plane z = 0 (the
 * camera sits at z = 3 looking down -Z, so every one is at depth 3), every
 * normal +Z, roughness 1, metallic 0, base color = its DC color; plus
 * (index PLANE, used only by the shadow checks) a small, nearly opaque
 * occluder at z = 1. */
#define GRID_X 7
#define GRID_Y 5
#define PLANE (GRID_X * GRID_Y)
#define V2_COUNT (PLANE + 1)
#define PI_F 3.14159265358979f
#define SH_C0 0.28209479177387814f

static float v_means[V2_COUNT * 4], v_scales[V2_COUNT * 4], v_rotations[V2_COUNT * 4],
    v_opacities[V2_COUNT], v_sh[V2_COUNT * COEFFS * 4], v_normals[V2_COUNT * 4],
    v_baseRough[V2_COUNT * 4], v_metallic[V2_COUNT];

static void makeRelightScene(void)
{
    int i, c;
    memset(v_sh, 0, sizeof(v_sh));
    for (i = 0; i < V2_COUNT; ++i)
    {
        const int gx = i % GRID_X, gy = i / GRID_X;
        const int occluder = i == PLANE;
        float color[3];
        v_means[i * 4 + 0] = occluder ? -0.6f : -1.2f + 0.4f * (float)gx;
        v_means[i * 4 + 1] = occluder ? 0.0f : -0.8f + 0.4f * (float)gy;
        v_means[i * 4 + 2] = occluder ? 1.0f : 0.0f;
        v_means[i * 4 + 3] = 0.0f;
        v_scales[i * 4 + 0] = v_scales[i * 4 + 1] = occluder ? 0.2f : 0.22f;
        v_scales[i * 4 + 2] = 0.01f;
        v_scales[i * 4 + 3] = 0.0f;
        v_rotations[i * 4 + 0] = v_rotations[i * 4 + 1] = v_rotations[i * 4 + 2] = 0.0f;
        v_rotations[i * 4 + 3] = 1.0f;
        v_opacities[i] = 0.99f;
        color[0] = occluder ? 0.5f : 0.25f + 0.5f * (float)gx / (GRID_X - 1);
        color[1] = occluder ? 0.5f : 0.3f + 0.4f * (float)gy / (GRID_Y - 1);
        color[2] = occluder ? 0.5f : 0.6f - 0.15f * (float)((gx + gy) % 3);
        for (c = 0; c < 3; ++c)
        {
            v_sh[i * COEFFS * 4 + c] = (color[c] - 0.5f) / SH_C0; /* color = 0.5 + C0 * DC */
            v_baseRough[i * 4 + c] = color[c];
        }
        v_baseRough[i * 4 + 3] = 1.0f; /* roughness */
        v_normals[i * 4 + 0] = v_normals[i * 4 + 1] = v_normals[i * 4 + 3] = 0.0f;
        v_normals[i * 4 + 2] = 1.0f;
        v_metallic[i] = 0.0f;
    }
}

/* L = sum of the red channel. */
static double renderRedSum(SabercatRenderer * r, float * image)
{
    double sum = 0.0;
    size_t i;
    if (sabercat_renderer_forward(r) != SABERCAT_OK
        || sabercat_renderer_read_image(r, image, WIDTH * HEIGHT * 4) != SABERCAT_OK)
    {
        return NAN;
    }
    for (i = 0; i < WIDTH * HEIGHT; ++i)
    {
        sum += image[i * 4];
    }
    return sum;
}

static SabercatGaussians relightDesc(uint32_t count)
{
    SabercatGaussians g = gaussiansDesc(v_means, v_scales, v_rotations, v_opacities, v_sh);
    g.count = count;
    g.sh_degree = 0;
    g.normals = v_normals;
    g.base_color_roughness = v_baseRough;
    g.metallic = v_metallic; /* visibility NULL: ambient visibility 1 */
    return g;
}

/* White directional light of irradiance pi traveling along d. */
static SabercatLight directionalLight(float dx, float dy, float dz)
{
    SabercatLight l;
    memset(&l, 0, sizeof(l));
    l.struct_size = sizeof(l);
    l.type = SABERCAT_LIGHT_DIRECTIONAL;
    l.casts_shadows = 1;
    l.intensity = PI_F;
    l.color[0] = l.color[1] = l.color[2] = 1.0f;
    l.direction[0] = dx;
    l.direction[1] = dy;
    l.direction[2] = dz;
    l.axis_x[0] = 1.0f;
    l.axis_y[2] = 1.0f;
    l.half_width = l.half_height = 0.5f;
    return l;
}

static double schlick(double cosTheta)
{
    return 0.04 + 0.96 * pow(1.0 - cosTheta, 5.0);
}

/* Worst |lit - analytic| over every channel of every pixel, for the relight plane
 * lit by (a) a white directional light of irradiance `irradiance` traveling
 * along the view axis (-Z) and/or (b) a uniform environment of radiance
 * `envRadiance`. With N = +Z everywhere, roughness 1 and metallic 0 the
 * shader's BRDF collapses to closed form per pixel: L = N gives
 * NdotL = 1; alpha = roughness^2 = 1 makes the GGX distribution 1/pi for
 * every half vector and the height-correlated Smith term 0.5/(NdotL +
 * NdotV); f0 = 0.04. So the direct light gives
 *     E * [(1 - F(VdotH)) base/pi + F(VdotH) / (2 pi (1 + NdotV))]
 * -- Lambert plus the roughness-1 lobe -- and the uniform environment
 * (irradiance pi Le, specular Le (1 - 0.5 roughness), ambient visibility 1)
 *     Le [(1 - F(NdotV)) base + 0.5 F(NdotV)].
 * V is per pixel from the known depth 3: the shader reconstructs the
 * position at pixel center (u, v) as ((u - cx) d/fx, (cy - v) d/fy, -d).
 * base is the splatted base color = the color image / its alpha (the given
 * linear base color stream, which has the same values as the DC color
 * here). The lit output is display-referred: lit rgb =
 * sRGB(clamp01(alpha * shading)) (background 0, exposure 1), lit alpha =
 * alpha. */
static double litWorst(const float * color, const float * lit, const SabercatCamera * cam,
                       double irradiance, double envRadiance, double * maxValue, int * covered)
{
    double worst = 0.0;
    int x, y, c;
    *maxValue = 0.0;
    *covered = 0;
    for (y = 0; y < HEIGHT; ++y)
    {
        for (x = 0; x < WIDTH; ++x)
        {
            const int p = y * WIDTH + x;
            const double alpha = color[p * 4 + 3];
            double want[4] = {0.0, 0.0, 0.0, alpha};
            if (alpha > 1e-6)
            {
                const double px = (x - cam->cx) * 3.0 / cam->fx, py = (cam->cy - y) * 3.0 / cam->fy;
                const double vl = sqrt(px * px + py * py + 9.0);
                const double vx = -px / vl, vy = -py / vl, vz = 3.0 / vl; /* to the camera */
                const double hl = sqrt(vx * vx + vy * vy + (vz + 1.0) * (vz + 1.0));
                const double vDotH = (vx * vx + vy * vy + vz * (vz + 1.0)) / hl;
                const double fh = schlick(vDotH), fv = schlick(vz);
                *covered += alpha > 0.5 ? 1 : 0;
                for (c = 0; c < 3; ++c)
                {
                    double base = color[p * 4 + c] / alpha, linear;
                    base = base < 0.0 ? 0.0 : (base > 1.0 ? 1.0 : base);
                    linear = alpha
                             * (irradiance
                                    * ((1.0 - fh) * base / PI_F + fh / (2.0 * PI_F * (1.0 + vz)))
                                + envRadiance * ((1.0 - fv) * base + 0.5 * fv));
                    linear = linear < 0.0 ? 0.0 : (linear > 1.0 ? 1.0 : linear);
                    want[c] = linearToSrgb(linear);
                }
            }
            for (c = 0; c < 4; ++c)
            {
                const double e = fabs((double)lit[p * 4 + c] - want[c]);
                worst = e > worst ? e : worst;
                *maxValue = fabs(want[c]) > *maxValue ? fabs(want[c]) : *maxValue;
            }
        }
    }
    return worst;
}

/* Linear red light of the lit image averaged over a 3x3 window around pixel
 * (x, y): the lit image is sRGB-encoded, and shadows are ratios of light. */
static double litRed(const float * lit, int x, int y)
{
    double sum = 0.0;
    int dx, dy;
    for (dy = -1; dy <= 1; ++dy)
    {
        for (dx = -1; dx <= 1; ++dx)
        {
            sum += srgbToLinear(lit[((y + dy) * WIDTH + x + dx) * 4]);
        }
    }
    return sum / 9.0;
}

/* Relighting and shadows through the ABI. r is the main renderer (any scene/size). */
static void testRelighting(SabercatRenderer * r, const SabercatRendererDesc * desc,
                           const SabercatCamera * camera)
{
    static float color[WIDTH * HEIGHT * 4], lit[WIDTH * HEIGHT * 4], lit2[WIDTH * HEIGHT * 4],
        litNone[WIDTH * HEIGHT * 4];
    static float hMeans[V2_COUNT * 4], hScales[V2_COUNT * 4], hRotations[V2_COUNT * 4],
        hOpacities[V2_COUNT], hSh[V2_COUNT * COEFFS * 4], dMeans[V2_COUNT * 4],
        dScales[V2_COUNT * 4], dRotations[V2_COUNT * 4], dOpacities[V2_COUNT],
        dSh[V2_COUNT * COEFFS * 4];
    const size_t pixels4 = (size_t)WIDTH * HEIGHT * 4;
    SabercatGaussians g;
    SabercatLight light, lights[2];
    SabercatLightingSettings ls, lsBad;
    SabercatShadowSettings ss, ssBad, ssGot;
    SabercatEnvironment env;
    SabercatDeviceInfo info, info48;
    SabercatBufferView litView, imageView;
    SabercatRenderer *rLegacy = NULL, *rHelper = NULL;
    double worst, maxValue;
    int covered, rayQuery;

    makeRelightScene();
    CHECK(sabercat_renderer_set_image_size(r, WIDTH, HEIGHT) == SABERCAT_OK
              && sabercat_renderer_set_camera(r, camera) == SABERCAT_OK,
          "v2: restore image size and camera");
    g = relightDesc(PLANE);
    CHECK(sabercat_renderer_set_gaussians(r, &g) == SABERCAT_OK, "v2: set_gaussians with relight streams");

    /* Lighting off: forward works, there is no lit image. */
    CHECK(sabercat_renderer_forward(r) == SABERCAT_OK
              && sabercat_renderer_read_lit_image(r, lit, pixels4) == SABERCAT_ERROR_INVALID_STATE
              && sabercat_renderer_get_lit_image_view(r, &litView) == SABERCAT_ERROR_INVALID_STATE,
          "read_lit_image/get_lit_image_view without lighting must be INVALID_STATE");

    /* Defaults, as the C++ structs initialize them. */
    memset(&ls, 0, sizeof(ls));
    ls.struct_size = sizeof(ls);
    CHECK(sabercat_renderer_get_lighting_settings(r, &ls) == SABERCAT_OK
              && ls.output == SABERCAT_LIGHTING_OUTPUT_LIT && ls.exposure == 1.0f
              && ls.background[0] == 0.0f && ls.rect_samples == 16 && ls.struct_size == sizeof(ls),
          "get_lighting_settings defaults");
    memset(&ss, 0, sizeof(ss));
    ss.struct_size = sizeof(ss);
    CHECK(sabercat_renderer_get_shadow_settings(r, &ss) == SABERCAT_OK
              && ss.mode == SABERCAT_SHADOW_NONE && ss.map_resolution == 1024
              && ss.ray_samples_per_axis == 4 && ss.focus_radius == 0.0f,
          "get_shadow_settings defaults");

    /* Argument errors. */
    light = directionalLight(0.0f, 0.0f, -1.0f);
    lights[0] = lights[1] = light;
    lights[1].type = 7;
    CHECK(sabercat_renderer_set_lights(r, lights, 2) == SABERCAT_ERROR_INVALID_ARGUMENT,
          "set_lights with an unknown type must be INVALID_ARGUMENT");
    lights[1] = light;
    lights[1].struct_size = sizeof(light) + 4;
    CHECK(sabercat_renderer_set_lights(r, lights, 2) == SABERCAT_ERROR_INVALID_ARGUMENT,
          "set_lights with mixed struct_size must be INVALID_ARGUMENT");
    CHECK(sabercat_renderer_set_lights(r, NULL, 1) == SABERCAT_ERROR_INVALID_ARGUMENT
              && sabercat_renderer_set_lights(r, lights, SABERCAT_MAX_LIGHT_COUNT + 1)
                     == SABERCAT_ERROR_INVALID_ARGUMENT,
          "set_lights(NULL, 1) / too many lights must be INVALID_ARGUMENT");
    lsBad = ls;
    lsBad.output = 8;
    CHECK(sabercat_renderer_set_lighting_settings(r, &lsBad) == SABERCAT_ERROR_INVALID_ARGUMENT,
          "set_lighting_settings with an unknown output must be INVALID_ARGUMENT");

    /* -- Analytic case 1: white directional light along the view axis. -- */
    CHECK(sabercat_renderer_set_lights(r, &light, 1) == SABERCAT_OK
              && sabercat_renderer_set_lighting_settings(r, &ls) == SABERCAT_OK
              && sabercat_renderer_set_lighting_enabled(r, 1) == SABERCAT_OK,
          "set_lights/set_lighting_settings/set_lighting_enabled");
    CHECK(sabercat_renderer_forward(r) == SABERCAT_OK
              && sabercat_renderer_read_image(r, color, pixels4) == SABERCAT_OK
              && sabercat_renderer_read_lit_image(r, lit, pixels4) == SABERCAT_OK,
          "lit forward + read_image + read_lit_image");
    worst = litWorst(color, lit, camera, PI_F, 0.0, &maxValue, &covered);
    printf("relight, directional light: worst |lit - analytic| %.3g (max value %.3g, %d "
           "pixels above 50%% coverage)\n",
           worst, maxValue, covered);
    CHECK(covered > WIDTH * HEIGHT / 2, "the plane covers most of the image");
    /* fp32 shading + the G-buffer base color vs the color image (equal to
     * ~2e-6, relight_test) sit around 1e-6; dropping the roughness-1 lobe
     * (0.01 alpha) or the diffuse (1 - F) weight (0.04 base alpha) would
     * each miss by ~1e-2. */
    CHECK(worst <= 1e-4, "lit image disagrees with the analytic Lambert + roughness-1 GGX value");

    /* -- Analytic case 2: a uniform environment only (radiance 0.25 x
     * intensity 2), exercising set_environment's SH layout. The DC
     * coefficient of a constant radiance Le is Le / C0. -- */
    memset(&env, 0, sizeof(env));
    env.struct_size = sizeof(env);
    env.intensity = 2.0f;
    env.sh[0] = env.sh[1] = env.sh[2] = 0.25f / SH_C0;
    CHECK(sabercat_renderer_set_lights(r, NULL, 0) == SABERCAT_OK
              && sabercat_renderer_set_environment(r, &env) == SABERCAT_OK
              && sabercat_renderer_forward(r) == SABERCAT_OK
              && sabercat_renderer_read_lit_image(r, lit, pixels4) == SABERCAT_OK,
          "environment-only lit forward");
    worst = litWorst(color, lit, camera, 0.0, 0.5, &maxValue, &covered);
    printf("relight, uniform environment: worst |lit - analytic| %.3g (max value %.3g)\n", worst,
           maxValue);
    CHECK(worst <= 1e-4, "environment-lit image disagrees with the analytic value");
    memset(env.sh, 0, sizeof(env.sh));
    CHECK(sabercat_renderer_set_environment(r, &env) == SABERCAT_OK
              && sabercat_renderer_set_lights(r, &light, 1) == SABERCAT_OK,
          "back to the directional light");

    /* -- Zero-copy lit image: backward fed the lit image VIEW (dL/dImage =
     * lit) matches backward fed read_lit_image's copy. -- */
    {
        SabercatGradients grads;
        double err = 0.0, e;
        CHECK(sabercat_renderer_forward(r) == SABERCAT_OK
                  && sabercat_renderer_read_lit_image(r, lit, pixels4) == SABERCAT_OK
                  && sabercat_renderer_get_lit_image_view(r, &litView) == SABERCAT_OK
                  && sabercat_renderer_get_image_view(r, &imageView) == SABERCAT_OK,
              "lit forward + get_lit_image_view");
        CHECK(litView.buffer != 0 && litView.buffer != imageView.buffer && litView.offset == 0
                  && litView.stride == 16 && litView.size == (uint64_t)WIDTH * HEIGHT * 16,
              "lit image view wrong");
        memset(&grads, 0, sizeof(grads));
        grads.struct_size = sizeof(grads);
        grads.count = PLANE;
        grads.means = hMeans;
        grads.scales = hScales;
        grads.rotations = hRotations;
        grads.opacities = hOpacities;
        grads.sh_coefficients = hSh;
        CHECK(sabercat_renderer_backward(r, lit, pixels4) == SABERCAT_OK
                  && sabercat_renderer_read_gradients(r, &grads) == SABERCAT_OK,
              "host backward of the lit image");
        grads.means = dMeans;
        grads.scales = dScales;
        grads.rotations = dRotations;
        grads.opacities = dOpacities;
        grads.sh_coefficients = dSh;
        CHECK(sabercat_renderer_backward_view(r, &litView) == SABERCAT_OK
                  && sabercat_renderer_read_gradients(r, &grads) == SABERCAT_OK,
              "backward_view of the lit image view");
        e = relativeError(dMeans, hMeans, PLANE * 4);
        err = e > err ? e : err;
        e = relativeError(dScales, hScales, PLANE * 4);
        err = e > err ? e : err;
        e = relativeError(dRotations, hRotations, PLANE * 4);
        err = e > err ? e : err;
        e = relativeError(dOpacities, hOpacities, PLANE);
        err = e > err ? e : err;
        e = relativeError(dSh, hSh, PLANE * COEFFS * 4);
        err = e > err ? e : err;
        printf("lit image view vs read_lit_image (through backward): worst rel. %.3g\n", err);
        CHECK(hOpacities[PLANE / 2] != 0.0f, "lit-image gradients are degenerate");
        /* Same bytes in, same kernels: only atomic-add ordering differs. */
        CHECK(err <= 1e-5, "lit image view holds something other than the lit image");
    }

    /* -- SabercatDeviceInfo: the relight tail, and the 48-byte pre-relight struct. -- */
    memset(&info, 0, sizeof(info));
    info.struct_size = sizeof(info);
    info.reserved0 = 0xFFu;
    CHECK(sizeof(info) == 56 && sabercat_renderer_get_device_info(r, &info) == SABERCAT_OK
              && (info.ray_query == 0 || info.ray_query == 1) && info.reserved0 == 0,
          "get_device_info must report ray_query 0/1 and zero reserved0");
    rayQuery = info.ray_query;
    printf("device ray query: %s\n", rayQuery ? "enabled" : "not enabled");
    memset(&info48, 0, sizeof(info48));
    info48.struct_size = (uint32_t)offsetof(SabercatDeviceInfo, ray_query); /* 48 */
    info48.ray_query = 77;
    info48.reserved0 = 0xABCDu;
    CHECK(info48.struct_size == 48 && sabercat_renderer_get_device_info(r, &info48) == SABERCAT_OK
              && info48.device == info.device && info48.compute_queue == info.compute_queue
              && info48.float32_atomic_add == info.float32_atomic_add
              && info48.vulkan_api_version == info.vulkan_api_version && info48.ray_query == 77
              && info48.reserved0 == 0xABCDu && info48.struct_size == 48,
          "get_device_info with the 48-byte struct must fill it and write nothing past it");

    /* create_borrowed from the 48-byte struct (ray_query = 77 lies past its
     * struct_size and must be ignored): declares no ray query. */
    CHECK(sabercat_renderer_create_borrowed(desc, &info48, &rLegacy) == SABERCAT_OK && rLegacy,
          "create_borrowed with the 48-byte SabercatDeviceInfo");
    /* The full struct declares what r has enabled. */
    CHECK(sabercat_renderer_create_borrowed(desc, &info, &rHelper) == SABERCAT_OK && rHelper,
          "create_borrowed with the 56-byte SabercatDeviceInfo");
    if (!rLegacy || !rHelper)
    {
        sabercat_renderer_destroy(rLegacy);
        sabercat_renderer_destroy(rHelper);
        return;
    }
    {
        SabercatDeviceInfo got;
        memset(&got, 0, sizeof(got));
        got.struct_size = sizeof(got);
        CHECK(sabercat_renderer_get_device_info(rLegacy, &got) == SABERCAT_OK && got.ray_query == 0,
              "a renderer borrowed through the 48-byte struct must report ray_query 0");
        memset(&got, 0, sizeof(got));
        got.struct_size = sizeof(got);
        CHECK(sabercat_renderer_get_device_info(rHelper, &got) == SABERCAT_OK
                  && got.ray_query == rayQuery,
              "ray_query must round-trip through create_borrowed");
    }

    /* -- Borrowed relight streams: rLegacy renders r's parameter buffers,
     * with relight streams that live in rHelper's (its means hold the
     * normals, its rotations base color + roughness, its opacities
     * metallic -- matching strides). Same bytes as r's host streams, so the
     * lit images must be identical. -- */
    {
        SabercatGaussians h;
        SabercatGaussianViews pv, hv;
        SabercatBorrowedGaussians b;
        h = gaussiansDesc(v_normals, v_scales, v_baseRough, v_metallic, v_sh);
        h.count = PLANE;
        h.sh_degree = 0;
        CHECK(sabercat_renderer_set_gaussians(rHelper, &h) == SABERCAT_OK
                  && sabercat_renderer_get_gaussian_views(rHelper, &hv) == SABERCAT_OK
                  && sabercat_renderer_get_gaussian_views(r, &pv) == SABERCAT_OK,
              "helper streams + views");
        memset(&b, 0, sizeof(b));
        b.struct_size = sizeof(b);
        b.count = PLANE;
        b.sh_degree = 0;
        b.parameters = pv;
        b.relight.normals = hv.means;
        b.relight.base_color_roughness = hv.rotations;
        b.relight.metallic = hv.opacities;
        b.relight.normals.stride = 12;
        CHECK(sabercat_renderer_set_gaussian_views(rLegacy, &b) == SABERCAT_ERROR_INVALID_ARGUMENT
                  && strstr(sabercat_last_error(), "relight.normals") != NULL,
              "a bad relight stride must be INVALID_ARGUMENT naming the stream");
        /* A struct that ends before `relight` never reads it. */
        b.struct_size = (uint32_t)offsetof(SabercatBorrowedGaussians, relight);
        CHECK(sabercat_renderer_set_gaussian_views(rLegacy, &b) == SABERCAT_OK,
              "set_gaussian_views with a struct that ends before relight must ignore relight");
        b.struct_size = sizeof(b);
        b.relight.normals = hv.means;
        CHECK(sabercat_renderer_set_gaussian_views(rLegacy, &b) == SABERCAT_OK,
              "set_gaussian_views with borrowed relight streams");
        CHECK(sabercat_renderer_set_camera(rLegacy, camera) == SABERCAT_OK
                  && sabercat_renderer_set_lights(rLegacy, &light, 1) == SABERCAT_OK
                  && sabercat_renderer_set_lighting_enabled(rLegacy, 1) == SABERCAT_OK
                  && sabercat_renderer_forward(rLegacy) == SABERCAT_OK
                  && sabercat_renderer_read_lit_image(rLegacy, lit2, pixels4) == SABERCAT_OK,
              "lit forward on borrowed streams");
        CHECK(memcmp(lit, lit2, sizeof(lit)) == 0,
              "borrowed relight streams render differently from the same host streams");
    }

    /* -- Shadow mode switching on the renderer without ray query. -- */
    ssBad = ss;
    ssBad.mode = SABERCAT_SHADOW_RAY_TRACED;
    CHECK(sabercat_renderer_set_shadow_settings(rLegacy, &ssBad) == SABERCAT_ERROR_UNSUPPORTED
              && strstr(sabercat_last_error(), "PCSS") != NULL,
          "RAY_TRACED without ray query must be UNSUPPORTED, pointing at PCSS");
    memset(&ssGot, 0, sizeof(ssGot));
    ssGot.struct_size = sizeof(ssGot);
    CHECK(sabercat_renderer_get_shadow_settings(rLegacy, &ssGot) == SABERCAT_OK
              && ssGot.mode == SABERCAT_SHADOW_NONE,
          "a rejected shadow mode must leave the settings unchanged");
    sabercat_renderer_destroy(rLegacy); /* before r: it borrows r's device and buffers */
    sabercat_renderer_destroy(rHelper);

    /* -- Shadows on r: the plane plus the occluder at (-0.6, 0, 1), lit by a
     * directional light traveling along (1, 0, -1)/sqrt(2). The occluder's
     * shadow center falls on (0.4, 0, 0) -- pixel (30.7, 16) -- and is 4
     * occluder sigmas clear of the occluder's own image; pixel (31, 4) is
     * 3.6 sigmas from the shadow, so stays lit. -- */
    g = relightDesc(V2_COUNT);
    light = directionalLight(0.70710678f, 0.0f, -0.70710678f);
    CHECK(sabercat_renderer_set_gaussians(r, &g) == SABERCAT_OK
              && sabercat_renderer_set_lights(r, &light, 1) == SABERCAT_OK
              && sabercat_renderer_forward(r) == SABERCAT_OK
              && sabercat_renderer_read_lit_image(r, litNone, pixels4) == SABERCAT_OK,
          "unshadowed lit forward with the occluder");

    ssBad = ss;
    ssBad.mode = 3;
    CHECK(sabercat_renderer_set_shadow_settings(r, &ssBad) == SABERCAT_ERROR_INVALID_ARGUMENT,
          "an unknown shadow mode must be INVALID_ARGUMENT");
    ssBad = ss;
    ssBad.mode = SABERCAT_SHADOW_PCSS;
    ssBad.map_resolution = 0;
    CHECK(sabercat_renderer_set_shadow_settings(r, &ssBad) == SABERCAT_ERROR_INVALID_ARGUMENT,
          "map_resolution 0 must be INVALID_ARGUMENT");
    ssBad.map_resolution = 1024;
    ssBad.focus_radius = -1.0f;
    CHECK(sabercat_renderer_set_shadow_settings(r, &ssBad) == SABERCAT_ERROR_INVALID_ARGUMENT,
          "a negative focus_radius must be INVALID_ARGUMENT");

    {
        const double noneShadow = litRed(litNone, 31, 16), noneLit = litRed(litNone, 31, 4);
        double pcssShadow, pcssLit;
        ss.mode = SABERCAT_SHADOW_PCSS;
        /* The shadow map splats each disk at its center's depth; at 45
         * degrees these wide disks (sigma 0.22) span ~+-0.15 in light depth
         * and self-shadow in stripes under the default bias (0.004 of the
         * focus radius, sized for small disks). 0.15 of the ~1.5 radius
         * covers that and is far below the occluder's 1.4 clearance. */
        ss.depth_bias = 0.15f;
        CHECK(sabercat_renderer_set_shadow_settings(r, &ss) == SABERCAT_OK
                  && sabercat_renderer_forward(r) == SABERCAT_OK
                  && sabercat_renderer_read_lit_image(r, lit, pixels4) == SABERCAT_OK,
              "PCSS lit forward");
        pcssShadow = litRed(lit, 31, 16) / noneShadow;
        pcssLit = litRed(lit, 31, 4) / noneLit;
        printf("PCSS / no shadows: %.3f in the occluder's shadow, %.3f outside\n", pcssShadow,
               pcssLit);        /* The occluder's transmittance is 0.01 at its center. */
        CHECK(noneShadow > 0.05 && pcssShadow < 0.2, "PCSS does not darken the shadowed pixels");
        CHECK(pcssLit > 0.95, "PCSS darkens pixels the occluder cannot shadow");

        light.casts_shadows = 0;
        CHECK(sabercat_renderer_set_lights(r, &light, 1) == SABERCAT_OK
                  && sabercat_renderer_forward(r) == SABERCAT_OK
                  && sabercat_renderer_read_lit_image(r, lit2, pixels4) == SABERCAT_OK
                  && memcmp(lit2, litNone, sizeof(lit2)) == 0,
              "casts_shadows 0 must render exactly as with no shadows");
        light.casts_shadows = 1;
        CHECK(sabercat_renderer_set_lights(r, &light, 1) == SABERCAT_OK, "casts_shadows back on");

        ssBad = ss;
        ssBad.mode = SABERCAT_SHADOW_RAY_TRACED;
        if (rayQuery)
        {
            double rtShadow, rtLit;
            CHECK(sabercat_renderer_set_shadow_settings(r, &ssBad) == SABERCAT_OK
                      && sabercat_renderer_forward(r) == SABERCAT_OK
                      && sabercat_renderer_read_lit_image(r, lit, pixels4) == SABERCAT_OK,
                  "ray-traced lit forward");
            rtShadow = litRed(lit, 31, 16) / noneShadow;
            rtLit = litRed(lit, 31, 4) / noneLit;
            printf("ray traced / no shadows: %.3f in the occluder's shadow, %.3f outside\n",
                   rtShadow, rtLit);
            CHECK(rtShadow < 0.2, "ray-traced shadows do not darken the shadowed pixels");
            CHECK(rtLit > 0.95, "ray-traced shadows darken pixels the occluder cannot shadow");
        }
        else
        {
            CHECK(sabercat_renderer_set_shadow_settings(r, &ssBad) == SABERCAT_ERROR_UNSUPPORTED,
                  "RAY_TRACED without ray query must be UNSUPPORTED");
        }
        memset(&ssGot, 0, sizeof(ssGot));
        ssGot.struct_size = sizeof(ssGot);
        CHECK(sabercat_renderer_get_shadow_settings(r, &ssGot) == SABERCAT_OK
                  && ssGot.mode == (rayQuery ? SABERCAT_SHADOW_RAY_TRACED : SABERCAT_SHADOW_PCSS)
                  && ssGot.map_resolution == ss.map_resolution && ssGot.depth_bias == 0.15f,
              "get_shadow_settings must return the last accepted settings");
    }
}

/* ---- Additional checks: argument matrix, struct sizes, PLY loading, settings
 * round trips, every lighting output and direct light against a double
 * reference, antialiasing and SH degree, finite differences of every
 * parameter, gradient views, and a large synthetic scene. ---- */

#define PI_D 3.14159265358979323846

/* Creates a renderer from `base` at the given size, or exits. */
static SabercatRenderer * makeRenderer(const SabercatRendererDesc * base, uint32_t width,
                                       uint32_t height)
{
    SabercatRendererDesc d = *base;
    SabercatRenderer * r = NULL;
    d.width = width;
    d.height = height;
    if (sabercat_renderer_create(&d, &r) != SABERCAT_OK)
    {
        fprintf(stderr, "create failed: %s\n", sabercat_last_error());
        exit(1);
    }
    return r;
}

static double maxAbsDiff(const float * a, const float * b, size_t n)
{
    double worst = 0.0;
    size_t i;
    for (i = 0; i < n; ++i)
    {
        const double d = fabs((double)a[i] - (double)b[i]);
        worst = d > worst ? d : worst;
    }
    return worst;
}

/* Expects INVALID_ARGUMENT with a non-empty message. */
#define EXPECT_INVALID(expr, what)                                                                 \
    do                                                                                             \
    {                                                                                              \
        const SabercatResult res_ = (expr);                                                        \
        CHECK(res_ == SABERCAT_ERROR_INVALID_ARGUMENT && strlen(sabercat_last_error()) > 0,        \
              what " must be INVALID_ARGUMENT with a message");                                    \
    } while (0)

/* Every entry point rejects a NULL renderer, NULL pointers and short
 * structs (struct_size below the header's size) with INVALID_ARGUMENT. */
static void testErrorMatrix(const SabercatRendererDesc * desc, const SabercatCamera * camera)
{
    SabercatRenderer * r = makeRenderer(desc, WIDTH, HEIGHT);
    SabercatRenderer * none = NULL;
    SabercatRendererDesc descShort = *desc;
    SabercatDeviceInfo info, infoShort;
    SabercatGaussians g = gaussiansDesc(g_means, g_scales, g_rotations, g_opacities, g_sh);
    SabercatGaussians gShort = g, gOut = g, gFull;
    SabercatCamera camShort = *camera;
    SabercatGaussianViews gv;
    SabercatGradientViews grv;
    SabercatGradients gr, grShort;
    SabercatBufferView bv;
    SabercatGaussianStorage st, stShort;
    SabercatBorrowedGaussians scene;
    SabercatBorrowedGaussians sceneShort;
    SabercatLight light;
    SabercatEnvironment env, envShort;
    SabercatLightingSettings ls, lsShort;
    SabercatShadowSettings ss, ssShort;
    static float buf[WIDTH * HEIGHT * 4], dimg[WIDTH * HEIGHT * 4];
    const size_t n = (size_t)WIDTH * HEIGHT * 4;
    uint32_t count = 0;
    int32_t grew = 0;

    memset(&info, 0, sizeof(info));
    info.struct_size = sizeof(info);
    CHECK(sabercat_renderer_get_device_info(r, &info) == SABERCAT_OK, "matrix: device info");
    infoShort = info;
    infoShort.struct_size = 8;
    gShort.struct_size = 8;
    gOut.struct_size = 8;
    camShort.struct_size = 4;
    descShort.struct_size = 8;
    memset(&gr, 0, sizeof(gr));
    gr.struct_size = sizeof(gr);
    gr.count = COUNT;
    grShort = gr;
    grShort.struct_size = 8;
    memset(&bv, 0, sizeof(bv));
    memset(&gv, 0, sizeof(gv));
    memset(&grv, 0, sizeof(grv));
    memset(&st, 0, sizeof(st));
    st.struct_size = sizeof(st);
    stShort = st;
    stShort.struct_size = 4;
    memset(&scene, 0, sizeof(scene));
    scene.struct_size = sizeof(scene);
    sceneShort = scene;
    sceneShort.struct_size = 8;
    memset(&light, 0, sizeof(light));
    light.struct_size = sizeof(light);
    memset(&env, 0, sizeof(env));
    env.struct_size = sizeof(env);
    envShort = env;
    envShort.struct_size = 4;
    memset(&ls, 0, sizeof(ls));
    ls.struct_size = sizeof(ls);
    lsShort = ls;
    lsShort.struct_size = 8;
    memset(&ss, 0, sizeof(ss));
    ss.struct_size = sizeof(ss);
    ssShort = ss;
    ssShort.struct_size = 8;

    /* Creation and destruction. */
    EXPECT_INVALID(sabercat_renderer_create(NULL, &none), "create(NULL desc)");
    EXPECT_INVALID(sabercat_renderer_create(desc, NULL), "create(NULL out)");
    EXPECT_INVALID(sabercat_renderer_create_borrowed(NULL, &info, &none),
                   "create_borrowed(NULL desc)");
    EXPECT_INVALID(sabercat_renderer_create_borrowed(desc, NULL, &none),
                   "create_borrowed(NULL device)");
    EXPECT_INVALID(sabercat_renderer_create_borrowed(desc, &info, NULL),
                   "create_borrowed(NULL out)");
    EXPECT_INVALID(sabercat_renderer_create_borrowed(&descShort, &info, &none),
                   "create_borrowed(short desc)");
    EXPECT_INVALID(sabercat_renderer_create_borrowed(desc, &infoShort, &none),
                   "create_borrowed(8-byte device info)");
    sabercat_renderer_destroy(NULL); /* a no-op, must not fault */

    /* Scene. */
    EXPECT_INVALID(sabercat_renderer_load_scene(NULL, "scene.ply"), "load_scene(NULL renderer)");
    EXPECT_INVALID(sabercat_renderer_load_scene(r, NULL), "load_scene(NULL path)");
    EXPECT_INVALID(sabercat_renderer_set_gaussians(NULL, &g), "set_gaussians(NULL renderer)");
    EXPECT_INVALID(sabercat_renderer_set_gaussians(r, NULL), "set_gaussians(NULL)");
    EXPECT_INVALID(sabercat_renderer_set_gaussians(r, &gShort), "set_gaussians(short)");
    EXPECT_INVALID(sabercat_renderer_get_gaussians(NULL, &gOut), "get_gaussians(NULL renderer)");
    EXPECT_INVALID(sabercat_renderer_get_gaussians(r, NULL), "get_gaussians(NULL)");
    gFull = gaussiansDesc(g_means, g_scales, g_rotations, g_opacities, g_sh);
    CHECK(sabercat_renderer_get_gaussians(r, &gFull) == SABERCAT_ERROR_INVALID_STATE,
          "get_gaussians with no scene must be INVALID_STATE");
    EXPECT_INVALID(sabercat_renderer_get_gaussian_count(NULL, &count),
                   "get_gaussian_count(NULL renderer)");
    EXPECT_INVALID(sabercat_renderer_get_gaussian_count(r, NULL), "get_gaussian_count(NULL)");
    EXPECT_INVALID(sabercat_renderer_set_gaussian_views(NULL, &scene),
                   "set_gaussian_views(NULL renderer)");
    EXPECT_INVALID(sabercat_renderer_set_gaussian_views(r, NULL), "set_gaussian_views(NULL)");
    EXPECT_INVALID(sabercat_renderer_set_gaussian_views(r, &sceneShort),
                   "set_gaussian_views(short)");
    EXPECT_INVALID(sabercat_renderer_set_gaussian_count(NULL, 1),
                   "set_gaussian_count(NULL renderer)");
    EXPECT_INVALID(sabercat_renderer_reserve_gaussians(NULL, 4, &grew),
                   "reserve_gaussians(NULL renderer)");

    /* Camera, SH and antialiasing. */
    EXPECT_INVALID(sabercat_renderer_set_camera(NULL, camera), "set_camera(NULL renderer)");
    EXPECT_INVALID(sabercat_renderer_set_camera(r, NULL), "set_camera(NULL)");
    EXPECT_INVALID(sabercat_renderer_set_camera(r, &camShort), "set_camera(short)");
    EXPECT_INVALID(sabercat_renderer_set_sh_degree(NULL, 0), "set_sh_degree(NULL renderer)");
    EXPECT_INVALID(sabercat_renderer_set_antialiased(NULL, 0), "set_antialiased(NULL renderer)");

    /* Forward and read-out. */
    EXPECT_INVALID(sabercat_renderer_forward(NULL), "forward(NULL)");
    EXPECT_INVALID(sabercat_renderer_read_image(NULL, buf, n), "read_image(NULL renderer)");
    EXPECT_INVALID(sabercat_renderer_read_image(r, NULL, n), "read_image(NULL)");
    EXPECT_INVALID(sabercat_renderer_read_image(r, buf, n - 1), "read_image(short count)");
    EXPECT_INVALID(sabercat_renderer_backward(NULL, dimg, n), "backward(NULL renderer)");
    EXPECT_INVALID(sabercat_renderer_backward(r, NULL, n), "backward(NULL)");
    EXPECT_INVALID(sabercat_renderer_backward(r, dimg, n + 1), "backward(wrong count)");
    EXPECT_INVALID(sabercat_renderer_backward_view(NULL, &bv), "backward_view(NULL renderer)");
    EXPECT_INVALID(sabercat_renderer_backward_view(r, NULL), "backward_view(NULL)");
    EXPECT_INVALID(sabercat_renderer_read_gradients(NULL, &gr), "read_gradients(NULL renderer)");
    EXPECT_INVALID(sabercat_renderer_read_gradients(r, NULL), "read_gradients(NULL)");
    EXPECT_INVALID(sabercat_renderer_read_gradients(r, &grShort), "read_gradients(short)");

    /* Views and storage. */
    EXPECT_INVALID(sabercat_renderer_get_gaussian_views(NULL, &gv),
                   "get_gaussian_views(NULL renderer)");
    EXPECT_INVALID(sabercat_renderer_get_gaussian_views(r, NULL), "get_gaussian_views(NULL)");
    EXPECT_INVALID(sabercat_renderer_get_gradient_views(NULL, &grv),
                   "get_gradient_views(NULL renderer)");
    EXPECT_INVALID(sabercat_renderer_get_gradient_views(r, NULL), "get_gradient_views(NULL)");
    EXPECT_INVALID(sabercat_renderer_get_device_info(NULL, &info),
                   "get_device_info(NULL renderer)");
    EXPECT_INVALID(sabercat_renderer_get_device_info(r, NULL), "get_device_info(NULL)");
    EXPECT_INVALID(sabercat_renderer_get_device_info(r, &infoShort), "get_device_info(short)");
    EXPECT_INVALID(sabercat_renderer_get_image_view(NULL, &bv), "get_image_view(NULL renderer)");
    EXPECT_INVALID(sabercat_renderer_get_image_view(r, NULL), "get_image_view(NULL)");
    EXPECT_INVALID(sabercat_renderer_get_gaussian_storage(NULL, &st),
                   "get_gaussian_storage(NULL renderer)");
    EXPECT_INVALID(sabercat_renderer_get_gaussian_storage(r, NULL),
                   "get_gaussian_storage(NULL)");
    EXPECT_INVALID(sabercat_renderer_get_gaussian_storage(r, &stShort),
                   "get_gaussian_storage(short)");
    EXPECT_INVALID(sabercat_renderer_set_image_size(NULL, 8, 8), "set_image_size(NULL renderer)");

    /* Lighting. */
    EXPECT_INVALID(sabercat_renderer_set_lighting_enabled(NULL, 1),
                   "set_lighting_enabled(NULL renderer)");
    EXPECT_INVALID(sabercat_renderer_set_lights(NULL, &light, 1), "set_lights(NULL renderer)");
    EXPECT_INVALID(sabercat_renderer_set_lights(r, NULL, 1), "set_lights(NULL, 1)");
    EXPECT_INVALID(sabercat_renderer_set_environment(NULL, &env),
                   "set_environment(NULL renderer)");
    EXPECT_INVALID(sabercat_renderer_set_environment(r, NULL), "set_environment(NULL)");
    EXPECT_INVALID(sabercat_renderer_set_environment(r, &envShort), "set_environment(short)");
    EXPECT_INVALID(sabercat_renderer_set_lighting_settings(NULL, &ls),
                   "set_lighting_settings(NULL renderer)");
    EXPECT_INVALID(sabercat_renderer_set_lighting_settings(r, NULL),
                   "set_lighting_settings(NULL)");
    EXPECT_INVALID(sabercat_renderer_set_lighting_settings(r, &lsShort),
                   "set_lighting_settings(short)");
    EXPECT_INVALID(sabercat_renderer_get_lighting_settings(NULL, &ls),
                   "get_lighting_settings(NULL renderer)");
    EXPECT_INVALID(sabercat_renderer_get_lighting_settings(r, NULL),
                   "get_lighting_settings(NULL)");
    EXPECT_INVALID(sabercat_renderer_get_lighting_settings(r, &lsShort),
                   "get_lighting_settings(short)");
    EXPECT_INVALID(sabercat_renderer_set_shadow_settings(NULL, &ss),
                   "set_shadow_settings(NULL renderer)");
    EXPECT_INVALID(sabercat_renderer_set_shadow_settings(r, NULL), "set_shadow_settings(NULL)");
    EXPECT_INVALID(sabercat_renderer_set_shadow_settings(r, &ssShort),
                   "set_shadow_settings(short)");
    EXPECT_INVALID(sabercat_renderer_get_shadow_settings(NULL, &ss),
                   "get_shadow_settings(NULL renderer)");
    EXPECT_INVALID(sabercat_renderer_get_shadow_settings(r, NULL), "get_shadow_settings(NULL)");
    EXPECT_INVALID(sabercat_renderer_get_shadow_settings(r, &ssShort),
                   "get_shadow_settings(short)");
    EXPECT_INVALID(sabercat_renderer_read_lit_image(NULL, buf, n),
                   "read_lit_image(NULL renderer)");
    EXPECT_INVALID(sabercat_renderer_read_lit_image(r, NULL, n), "read_lit_image(NULL)");
    EXPECT_INVALID(sabercat_renderer_get_lit_image_view(NULL, &bv),
                   "get_lit_image_view(NULL renderer)");
    EXPECT_INVALID(sabercat_renderer_get_lit_image_view(r, NULL), "get_lit_image_view(NULL)");

    /* A state error stays a state error: forward with a camera but no scene
     * is INVALID_STATE, not INVALID_ARGUMENT. */
    CHECK(sabercat_renderer_forward(r) == SABERCAT_ERROR_INVALID_STATE,
          "matrix: forward with no scene must stay INVALID_STATE");
    sabercat_renderer_destroy(r);
    printf("argument matrix: every NULL/short case above rejected with a message\n");
}

/* Struct-size compatibility. A caller's struct larger than the header's
 * (a newer header with appended fields) must be accepted, read only up to
 * the fields the library knows, and never have its tail touched; the
 * library's own output must not spill into that tail either. */
typedef struct
{
    SabercatCamera value;
    unsigned char tail[16];
} PaddedCamera;
typedef struct
{
    SabercatGaussians value;
    unsigned char tail[16];
} PaddedGaussians;
typedef struct
{
    SabercatDeviceInfo value;
    unsigned char tail[16];
} PaddedDeviceInfo;
typedef struct
{
    SabercatLightingSettings value;
    unsigned char tail[16];
} PaddedLightingSettings;

static int tailIntact(const unsigned char * tail, unsigned char fill)
{
    size_t i;
    for (i = 0; i < 16; ++i)
    {
        if (tail[i] != fill)
        {
            return 0;
        }
    }
    return 1;
}

static void testStructSizes(const SabercatRendererDesc * desc, const SabercatCamera * camera)
{
    SabercatRenderer * r = makeRenderer(desc, WIDTH, HEIGHT);
    static float base[WIDTH * HEIGHT * 4], image[WIDTH * HEIGHT * 4];
    static float outM[COUNT * 4], outS[COUNT * 4], outR[COUNT * 4], outO[COUNT], outSh[COUNT * COEFFS * 4];
    const size_t n = (size_t)WIDTH * HEIGHT * 4;
    PaddedCamera pc;
    PaddedGaussians pg;
    PaddedDeviceInfo pd, pdRef;
    PaddedLightingSettings pl;
    SabercatGaussians g = gaussiansDesc(g_means, g_scales, g_rotations, g_opacities, g_sh);
    SabercatDeviceInfo ref;
    SabercatLightingSettings lsRef;
    SabercatCamera cam = *camera;

    CHECK(sabercat_renderer_set_gaussians(r, &g) == SABERCAT_OK, "struct sizes: set_gaussians");
    CHECK(sabercat_renderer_set_camera(r, camera) == SABERCAT_OK
              && sabercat_renderer_forward(r) == SABERCAT_OK
              && sabercat_renderer_read_image(r, base, n) == SABERCAT_OK,
          "struct sizes: baseline forward");

    memset(&pc, 0xAB, sizeof(pc));
    pc.value = cam;
    pc.value.struct_size = (uint32_t)sizeof(pc);
    CHECK(sabercat_renderer_set_camera(r, &pc.value) == SABERCAT_OK,
          "a larger SabercatCamera struct_size must be accepted");
    CHECK(sabercat_renderer_forward(r) == SABERCAT_OK
              && sabercat_renderer_read_image(r, image, n) == SABERCAT_OK
              && memcmp(image, base, sizeof(base)) == 0 && tailIntact(pc.tail, 0xAB),
          "a larger camera struct must render identically and leave its tail alone");

    memset(&pg, 0xAB, sizeof(pg));
    pg.value = g;
    pg.value.struct_size = (uint32_t)sizeof(pg);
    CHECK(sabercat_renderer_set_gaussians(r, &pg.value) == SABERCAT_OK,
          "a larger SabercatGaussians struct_size must be accepted");
    CHECK(sabercat_renderer_forward(r) == SABERCAT_OK
              && sabercat_renderer_read_image(r, image, n) == SABERCAT_OK
              && memcmp(image, base, sizeof(base)) == 0 && tailIntact(pg.tail, 0xAB),
          "a larger gaussians struct must render identically and leave its tail alone");

    memset(&pg, 0x5C, sizeof(pg));
    pg.value = gaussiansDesc(outM, outS, outR, outO, outSh);
    pg.value.struct_size = (uint32_t)sizeof(pg);
    CHECK(sabercat_renderer_get_gaussians(r, &pg.value) == SABERCAT_OK && tailIntact(pg.tail, 0x5C)
              && memcmp(outM, g_means, sizeof(g_means)) == 0
              && memcmp(outSh, g_sh, sizeof(g_sh)) == 0,
          "get_gaussians into a larger struct must fill the arrays and leave its tail alone");

    memset(&pdRef, 0, sizeof(pdRef));
    pdRef.value.struct_size = sizeof(SabercatDeviceInfo);
    CHECK(sabercat_renderer_get_device_info(r, &pdRef.value) == SABERCAT_OK,
          "device info for the reference");
    ref = pdRef.value;
    memset(&pd, 0x77, sizeof(pd));
    pd.value.struct_size = (uint32_t)sizeof(pd);
    CHECK(sabercat_renderer_get_device_info(r, &pd.value) == SABERCAT_OK
              && tailIntact(pd.tail, 0x77) && pd.value.device == ref.device
              && pd.value.compute_queue == ref.compute_queue,
          "get_device_info into a larger struct must match and leave its tail alone");

    memset(&pl, 0x33, sizeof(pl));
    pl.value.struct_size = (uint32_t)sizeof(pl);
    CHECK(sabercat_renderer_get_lighting_settings(r, &pl.value) == SABERCAT_OK
              && tailIntact(pl.tail, 0x33),
          "get_lighting_settings into a larger struct must leave its tail alone");
    memset(&lsRef, 0, sizeof(lsRef));
    lsRef.struct_size = sizeof(lsRef);
    CHECK(sabercat_renderer_get_lighting_settings(r, &lsRef) == SABERCAT_OK
              && lsRef.exposure == pl.value.exposure && lsRef.output == pl.value.output,
          "larger and exact lighting-settings structs must agree");

    /* Narrower than the header is still refused (the minimum is the header's size). */
    cam.struct_size = (uint32_t)sizeof(cam) - 4;
    CHECK(sabercat_renderer_set_camera(r, &cam) == SABERCAT_ERROR_INVALID_ARGUMENT,
          "a camera struct 4 bytes short must be INVALID_ARGUMENT");

    sabercat_renderer_destroy(r);
    printf("struct sizes: larger structs accepted with the tail untouched; short ones refused\n");
}

/* A PLY with the 3DGS property set and a DC-only color, written by hand. */
static void writeTestPly(const char * path, const float * means, const float * dc, unsigned count)
{
    FILE * f = fopen(path, "wb");
    unsigned i;
    int c;
    if (!f)
    {
        fprintf(stderr, "cannot write %s\n", path);
        exit(1);
    }
    fprintf(f, "ply\nformat binary_little_endian 1.0\nelement vertex %u\n"
               "property float x\nproperty float y\nproperty float z\n"
               "property float f_dc_0\nproperty float f_dc_1\nproperty float f_dc_2\n"
               "property float opacity\n"
               "property float scale_0\nproperty float scale_1\nproperty float scale_2\n"
               "property float rot_0\nproperty float rot_1\nproperty float rot_2\n"
               "property float rot_3\nend_header\n",
            count);
    for (i = 0; i < count; ++i)
    {
        /* Raw values: opacity logit 0 -> 0.5, log scales 0 -> 1, rot_0 = w = 1. */
        float row[14];
        for (c = 0; c < 3; ++c)
        {
            row[c] = means[i * 4 + c];
            row[3 + c] = dc[i * 3 + c];
        }
        row[6] = 0.0f;
        row[7] = row[8] = row[9] = 0.0f;
        row[10] = 1.0f;
        row[11] = row[12] = row[13] = 0.0f;
        fwrite(row, sizeof(float), 14, f);
    }
    fclose(f);
}

/* load_scene from a PLY written here, against the same data handed to
 * set_gaussians: the activated values, the image and the readback are
 * bit-identical. Missing and malformed files are RUNTIME with a message. */
static void testLoadScene(const SabercatRendererDesc * desc, const SabercatCamera * camera)
{
    enum { N = 3 };
    static const char path[] = "sabercat_c_api_test_scene.ply";
    static float means[N * 4], dc[N * 3];
    static float expM[N * 4], expS[N * 4], expR[N * 4], expO[N], expSh[N * COEFFS * 4];
    static float gotM[N * 4], gotS[N * 4], gotR[N * 4], gotO[N], gotSh[N * COEFFS * 4];
    static float imgLoad[WIDTH * HEIGHT * 4], imgSet[WIDTH * HEIGHT * 4];
    const size_t pixels4 = (size_t)WIDTH * HEIGHT * 4;
    SabercatRenderer * rl = makeRenderer(desc, WIDTH, HEIGHT);
    SabercatRenderer * rs = makeRenderer(desc, WIDTH, HEIGHT);
    SabercatGaussians setIn, gotG;
    uint32_t count = 0;
    int i, c, k;
    SabercatResult res;

    for (i = 0; i < N; ++i)
    {
        means[i * 4 + 0] = 0.3f * (float)i - 0.25f;
        means[i * 4 + 1] = -0.1f * (float)i + 0.05f;
        means[i * 4 + 2] = -0.4f * (float)i;
        means[i * 4 + 3] = 0.0f;
        dc[i * 3 + 0] = 0.2f + 0.1f * (float)i;
        dc[i * 3 + 1] = 0.5f - 0.1f * (float)i;
        dc[i * 3 + 2] = 0.7f;
        for (c = 0; c < 4; ++c)
        {
            expM[i * 4 + c] = means[i * 4 + c];
            expS[i * 4 + c] = 1.0f;
            expR[i * 4 + c] = c == 3 ? 1.0f : 0.0f; /* xyzw identity */
        }
        expS[i * 4 + 3] = 0.0f;
        expO[i] = 0.5f;
        for (k = 0; k < COEFFS * 4; ++k)
        {
            expSh[i * COEFFS * 4 + k] = 0.0f;
        }
        for (c = 0; c < 3; ++c)
        {
            expSh[i * COEFFS * 4 + c] = dc[i * 3 + c];
        }
    }
    writeTestPly(path, means, dc, N);

    res = sabercat_renderer_load_scene(rl, "sabercat_c_api_test_missing.ply");
    CHECK(res == SABERCAT_ERROR_RUNTIME && strlen(sabercat_last_error()) > 0,
          "load_scene of a missing file must be RUNTIME with a message");
    CHECK(strstr(sabercat_last_error(), "cannot open scene file") != NULL
              && strstr(sabercat_last_error(), "magic") == NULL,
          "a missing file must be reported as unopenable, not as a bad PLY header");
    printf("missing file: %s\n", sabercat_last_error());

    CHECK(sabercat_renderer_load_scene(rl, path) == SABERCAT_OK, "load_scene of a valid PLY");
    CHECK(sabercat_renderer_get_gaussian_count(rl, &count) == SABERCAT_OK && count == N,
          "load_scene must report 3 Gaussians");
    memset(&gotG, 0, sizeof(gotG));
    gotG = gaussiansDesc(gotM, gotS, gotR, gotO, gotSh);
    CHECK(sabercat_renderer_get_gaussians(rl, &gotG) == SABERCAT_OK, "get_gaussians after load");
    CHECK(memcmp(gotM, expM, sizeof(expM)) == 0 && memcmp(gotS, expS, sizeof(expS)) == 0
              && memcmp(gotR, expR, sizeof(expR)) == 0 && memcmp(gotO, expO, sizeof(expO)) == 0
              && memcmp(gotSh, expSh, sizeof(expSh)) == 0,
          "loaded values must be the raw PLY values under the 3DGS activations");

    /* The same data through set_gaussians: the readback must match bit for bit. */
    setIn = gaussiansDesc(gotM, gotS, gotR, gotO, gotSh);
    CHECK(sabercat_renderer_set_gaussians(rs, &setIn) == SABERCAT_OK, "set_gaussians of loaded data");
    CHECK(sabercat_renderer_set_camera(rl, camera) == SABERCAT_OK
              && sabercat_renderer_set_camera(rs, camera) == SABERCAT_OK,
          "set camera on the loaded and set renderers");
    CHECK(sabercat_renderer_forward(rl) == SABERCAT_OK
              && sabercat_renderer_read_image(rl, imgLoad, pixels4) == SABERCAT_OK
              && sabercat_renderer_forward(rs) == SABERCAT_OK
              && sabercat_renderer_read_image(rs, imgSet, pixels4) == SABERCAT_OK,
          "forward both renderers");
    CHECK(memcmp(imgLoad, imgSet, sizeof(imgLoad)) == 0,
          "load_scene must render bit-identically to set_gaussians of the same data");

    /* Malformed file: not a PLY at all. */
    {
        FILE * f = fopen(path, "wb");
        if (f)
        {
            fputs("this is not a ply file\n", f);
            fclose(f);
        }
    }
    res = sabercat_renderer_load_scene(rl, path);
    CHECK(res == SABERCAT_ERROR_RUNTIME && strlen(sabercat_last_error()) > 0,
          "a malformed PLY must be RUNTIME with a message");
    remove(path);

    sabercat_renderer_destroy(rl);
    sabercat_renderer_destroy(rs);
    printf("load_scene: PLY render and readback bit-identical to set_gaussians; missing and "
           "malformed files RUNTIME\n");
}

/* Settings round trips: every field of both structs, every enum value, and
 * the rejected values leave the previous settings in place. */
static void testSettingsRoundTrip(const SabercatRendererDesc * desc, int rayQuery)
{
    SabercatRenderer * r = makeRenderer(desc, WIDTH, HEIGHT);
    SabercatLightingSettings in, out, prev;
    SabercatShadowSettings sin, sout, sprev, sbad;
    uint32_t k;
    int ok;

    memset(&out, 0, sizeof(out));
    out.struct_size = sizeof(out);
    CHECK(sabercat_renderer_get_lighting_settings(r, &out) == SABERCAT_OK && out.exposure == 1.0f
              && out.output == SABERCAT_LIGHTING_OUTPUT_LIT,
          "default lighting settings");

    for (k = 0; k < 8; ++k)
    {
        memset(&in, 0, sizeof(in));
        in.struct_size = sizeof(in);
        in.output = k;
        in.rect_samples = 1 + 3 * k;
        in.visibility_light = k % SABERCAT_MAX_LIGHT_COUNT;
        in.background[0] = 0.125f * (float)k;
        in.background[1] = 0.25f;
        in.background[2] = 0.75f;
        in.exposure = 0.5f + 0.25f * (float)k;
        in.depth_scale = 2.25f - 0.125f * (float)k;
        memset(&out, 0, sizeof(out));
        out.struct_size = sizeof(out);
        CHECK(sabercat_renderer_set_lighting_settings(r, &in) == SABERCAT_OK
                  && sabercat_renderer_get_lighting_settings(r, &out) == SABERCAT_OK,
              "lighting settings set/get");
        ok = out.output == in.output && out.rect_samples == in.rect_samples
             && out.visibility_light == in.visibility_light
             && out.background[0] == in.background[0] && out.background[1] == in.background[1]
             && out.background[2] == in.background[2] && out.exposure == in.exposure
             && out.depth_scale == in.depth_scale;
        CHECK(ok, "lighting settings must round-trip every field bit-exactly");
    }

    /* Unknown output: rejected, previous settings kept. */
    prev = out;
    in.output = 8;
    EXPECT_INVALID(sabercat_renderer_set_lighting_settings(r, &in),
                   "lighting output 8");
    memset(&out, 0, sizeof(out));
    out.struct_size = sizeof(out);
    CHECK(sabercat_renderer_get_lighting_settings(r, &out) == SABERCAT_OK
              && out.output == prev.output && out.exposure == prev.exposure,
          "a rejected lighting output must leave the settings unchanged");

    /* Shadow settings: every field, every mode the device supports. */
    memset(&sin, 0, sizeof(sin));
    sin.struct_size = sizeof(sin);
    sin.mode = SABERCAT_SHADOW_PCSS;
    sin.map_resolution = 512;
    sin.blocker_samples_per_axis = 3;
    sin.filter_samples_per_axis = 5;
    sin.ray_samples_per_axis = 2;
    sin.focus_center[0] = 0.5f;
    sin.focus_center[1] = -0.25f;
    sin.focus_center[2] = 1.0f;
    sin.focus_radius = 2.5f;
    sin.point_light_radius = 0.125f;
    sin.sun_angular_radius = 0.0625f;
    sin.depth_bias = 0.25f;
    sin.normal_offset = 0.5f;
    sin.max_filter_texels = 8.0f;
    CHECK(sabercat_renderer_set_shadow_settings(r, &sin) == SABERCAT_OK, "PCSS settings set");
    memset(&sout, 0, sizeof(sout));
    sout.struct_size = sizeof(sout);
    CHECK(sabercat_renderer_get_shadow_settings(r, &sout) == SABERCAT_OK, "shadow settings get");
    ok = sout.mode == sin.mode && sout.map_resolution == sin.map_resolution
         && sout.blocker_samples_per_axis == sin.blocker_samples_per_axis
         && sout.filter_samples_per_axis == sin.filter_samples_per_axis
         && sout.ray_samples_per_axis == sin.ray_samples_per_axis
         && sout.focus_center[0] == sin.focus_center[0] && sout.focus_center[1] == sin.focus_center[1]
         && sout.focus_center[2] == sin.focus_center[2] && sout.focus_radius == sin.focus_radius
         && sout.point_light_radius == sin.point_light_radius
         && sout.sun_angular_radius == sin.sun_angular_radius && sout.depth_bias == sin.depth_bias
         && sout.normal_offset == sin.normal_offset && sout.max_filter_texels == sin.max_filter_texels;
    CHECK(ok, "shadow settings must round-trip every field bit-exactly");

    /* Ray-traced: round trip where ray query is enabled, UNSUPPORTED where not. */
    sin.mode = SABERCAT_SHADOW_RAY_TRACED;
    if (rayQuery)
    {
        CHECK(sabercat_renderer_set_shadow_settings(r, &sin) == SABERCAT_OK
                  && sabercat_renderer_get_shadow_settings(r, &sout) == SABERCAT_OK
                  && sout.mode == SABERCAT_SHADOW_RAY_TRACED,
              "ray-traced settings round trip");
    }
    else
    {
        CHECK(sabercat_renderer_set_shadow_settings(r, &sin) == SABERCAT_ERROR_UNSUPPORTED,
              "ray-traced without ray query must be UNSUPPORTED");
    }

    /* Each zero count and a negative or NaN radius is rejected; the previous
     * settings stay. */
    sprev = sout;
    sbad = sprev;
    sbad.map_resolution = 0;
    EXPECT_INVALID(sabercat_renderer_set_shadow_settings(r, &sbad), "map_resolution 0");
    sbad = sprev;
    sbad.blocker_samples_per_axis = 0;
    EXPECT_INVALID(sabercat_renderer_set_shadow_settings(r, &sbad), "blocker samples 0");
    sbad = sprev;
    sbad.filter_samples_per_axis = 0;
    EXPECT_INVALID(sabercat_renderer_set_shadow_settings(r, &sbad), "filter samples 0");
    sbad = sprev;
    sbad.ray_samples_per_axis = 0;
    EXPECT_INVALID(sabercat_renderer_set_shadow_settings(r, &sbad), "ray samples 0");
    sbad = sprev;
    sbad.focus_radius = -0.5f;
    EXPECT_INVALID(sabercat_renderer_set_shadow_settings(r, &sbad), "negative focus radius");
    sbad = sprev;
    sbad.focus_radius = (float)NAN;
    EXPECT_INVALID(sabercat_renderer_set_shadow_settings(r, &sbad), "NaN focus radius");
    sbad = sprev;
    sbad.mode = 3;
    EXPECT_INVALID(sabercat_renderer_set_shadow_settings(r, &sbad), "shadow mode 3");
    memset(&sout, 0, sizeof(sout));
    sout.struct_size = sizeof(sout);
    CHECK(sabercat_renderer_get_shadow_settings(r, &sout) == SABERCAT_OK
              && sout.map_resolution == sprev.map_resolution
              && sout.focus_radius == sprev.focus_radius && sout.mode == sprev.mode,
          "rejected shadow settings must leave the previous ones in place");

    sabercat_renderer_destroy(r);
    printf("settings: lighting (8 outputs) and shadow fields round-trip bit-exactly; rejected "
           "values keep the old ones\n");
}

/* ---- Direct light and output checks against a double-precision reference ----
 *
 * The plane scene (relightDesc, flat Gaussians at z = 0, normals +Z,
 * roughness 1, metallic 0, base color = DC color) seen from the camera at
 * z = 3. Per pixel the reference is the shader's own BRDF (see
 * refShade) evaluated in double with the exact per-pixel position. */

static double dot3(const double a[3], const double b[3])
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

static void normalize3(double v[3])
{
    const double len = sqrt(dot3(v, v));
    v[0] /= len;
    v[1] /= len;
    v[2] /= len;
}

/* The shader's BRDF times cos(theta_l) at roughness 1 (alpha 1), metallic 0:
 * GGX D = 1/pi, height-correlated Smith V = 0.5 / (NdotL + NdotV), Schlick
 * F with f0 = 0.04, Lambert diffuse scaled by (1 - F). */
static void refShade(const double n[3], const double v[3], const double l[3],
                     const double base[3], double out[3])
{
    const double nl = dot3(n, l);
    double h[3], nv, vh, f, spec;
    int c;
    if (nl <= 0.0)
    {
        out[0] = out[1] = out[2] = 0.0;
        return;
    }
    /* D = 1/pi for every half vector at alpha 1, so NdotH is not needed. */
    nv = dot3(n, v) > 1e-4 ? dot3(n, v) : 1e-4;
    for (c = 0; c < 3; ++c)
    {
        h[c] = l[c] + v[c];
    }
    normalize3(h);
    vh = dot3(v, h);
    vh = vh < 0.0 ? 0.0 : (vh > 1.0 ? 1.0 : vh);
    f = 0.04 + 0.96 * pow(1.0 - vh, 5.0);
    spec = f * (1.0 / PI_D) * 0.5 / (nl + nv);
    for (c = 0; c < 3; ++c)
    {
        out[c] = ((1.0 - f) * base[c] / PI_D + spec) * nl;
    }
}

/* Irradiance-weighted outgoing radiance from one light at the point p. */
static void refLight(const SabercatLight * l, const double p[3], const double n[3],
                     const double v[3], const double base[3], uint32_t rectN, double out[3])
{
    double one[3], acc[3] = {0.0, 0.0, 0.0}, radiance[3], L[3], toL[3], d2, ax[3], ay[3], emit[3];
    double area, sampleArea, s[3], u, w, cosLight;
    uint32_t sx, sy, c;
    int i;
    for (c = 0; c < 3; ++c)
    {
        radiance[c] = (double)l->color[c] * (double)l->intensity;
    }
    if (l->type == SABERCAT_LIGHT_DIRECTIONAL)
    {
        for (i = 0; i < 3; ++i)
        {
            L[i] = -(double)l->direction[i];
        }
        normalize3(L);
        refShade(n, v, L, base, one);
        for (c = 0; c < 3; ++c)
        {
            out[c] = one[c] * radiance[c];
        }
    }
    else if (l->type == SABERCAT_LIGHT_POINT)
    {
        for (i = 0; i < 3; ++i)
        {
            toL[i] = (double)l->position[i] - p[i];
        }
        d2 = dot3(toL, toL);
        L[0] = toL[0] / sqrt(d2);
        L[1] = toL[1] / sqrt(d2);
        L[2] = toL[2] / sqrt(d2);
        refShade(n, v, L, base, one);
        for (c = 0; c < 3; ++c)
        {
            out[c] = one[c] * radiance[c] / d2;
        }
    }
    else
    {
        /* Rect: a midpoint rule over the emitter, one-sided along cross(x, y). */
        for (i = 0; i < 3; ++i)
        {
            ax[i] = (double)l->axis_x[i] * (double)l->half_width;
            ay[i] = (double)l->axis_y[i] * (double)l->half_height;
        }
        {
            const double ex[3] = {l->axis_x[0], l->axis_x[1], l->axis_x[2]};
            const double ey[3] = {l->axis_y[0], l->axis_y[1], l->axis_y[2]};
            emit[0] = ex[1] * ey[2] - ex[2] * ey[1];
            emit[1] = ex[2] * ey[0] - ex[0] * ey[2];
            emit[2] = ex[0] * ey[1] - ex[1] * ey[0];
        }
        normalize3(emit);
        area = 4.0 * sqrt(dot3(ax, ax)) * sqrt(dot3(ay, ay));
        sampleArea = area / ((double)rectN * (double)rectN);
        for (sy = 0; sy < rectN; ++sy)
        {
            for (sx = 0; sx < rectN; ++sx)
            {
                u = ((double)sx + 0.5) / (double)rectN * 2.0 - 1.0;
                w = ((double)sy + 0.5) / (double)rectN * 2.0 - 1.0;
                for (i = 0; i < 3; ++i)
                {
                    s[i] = (double)l->position[i] + u * ax[i] + w * ay[i];
                    toL[i] = s[i] - p[i];
                }
                d2 = dot3(toL, toL);
                L[0] = toL[0] / sqrt(d2);
                L[1] = toL[1] / sqrt(d2);
                L[2] = toL[2] / sqrt(d2);
                cosLight = -dot3(emit, L);
                if (cosLight <= 0.0)
                {
                    continue;
                }
                refShade(n, v, L, base, one);
                for (c = 0; c < 3; ++c)
                {
                    acc[c] += one[c] * cosLight * sampleArea / d2;
                }
            }
        }
        for (c = 0; c < 3; ++c)
        {
            out[c] = acc[c] * radiance[c];
        }
    }
}

/* Worst |lit - expected| over RGBA for one output. `mode` is the
 * SabercatLightingOutput; `light` is used by LIT only. The composition is
 * the documented one: (shaded * coverage + background * (1 - coverage)) *
 * exposure, alpha = coverage, with background (0.1, 0.2, 0.3), exposure
 * 0.75 and depth_scale 0.5. For LIT the composite is linear (the background
 * decoded from sRGB), clamped and sRGB-encoded; the data outputs are raw. */
static double worstOutput(const float * color, const float * lit, uint32_t mode,
                          const SabercatLight * light, uint32_t rectN,
                          const SabercatCamera * cam, double * maxValue)
{
    const double bg[3] = {0.1, 0.2, 0.3};
    const double exposure = 0.75, depthScale = 0.5;
    double worst = 0.0;
    int x, y, c;
    *maxValue = 0.0;
    for (y = 0; y < HEIGHT; ++y)
    {
        for (x = 0; x < WIDTH; ++x)
        {
            const int p = y * WIDTH + x;
            const double alpha = color[p * 4 + 3];
            double base[3], shaded[3] = {0.0, 0.0, 0.0}, want[4];
            for (c = 0; c < 3; ++c)
            {
                double b = alpha > 1e-6 ? color[p * 4 + c] / alpha : 0.0;
                base[c] = b < 0.0 ? 0.0 : (b > 1.0 ? 1.0 : b);
            }
            if (alpha > 1e-6)
            {
                switch (mode)
                {
                case SABERCAT_LIGHTING_OUTPUT_LIT:
                {
                    /* The pixel center (x, y) at depth 3 in camera space, then to
                     * world (the camera sits at z = 3, the plane at z = 0). */
                    const double px = ((double)x - cam->cx) * 3.0 / cam->fx;
                    const double py = ((double)cam->cy - (double)y) * 3.0 / cam->fy;
                    const double P[3] = {px, py, 0.0};
                    const double N[3] = {0.0, 0.0, 1.0};
                    double V[3] = {-px, -py, 3.0};
                    normalize3(V);
                    refLight(light, P, N, V, base, rectN, shaded);
                    break;
                }
                case SABERCAT_LIGHTING_OUTPUT_NORMAL:
                    shaded[0] = shaded[1] = 0.5;
                    shaded[2] = 1.0;
                    break;
                case SABERCAT_LIGHTING_OUTPUT_BASE_COLOR:
                    for (c = 0; c < 3; ++c)
                    {
                        shaded[c] = base[c];
                    }
                    break;
                case SABERCAT_LIGHTING_OUTPUT_ROUGHNESS:
                case SABERCAT_LIGHTING_OUTPUT_AMBIENT_OCCLUSION:
                case SABERCAT_LIGHTING_OUTPUT_VISIBILITY:
                    shaded[0] = shaded[1] = shaded[2] = 1.0;
                    break;
                case SABERCAT_LIGHTING_OUTPUT_METALLIC:
                    break;
                case SABERCAT_LIGHTING_OUTPUT_DEPTH:
                    shaded[0] = shaded[1] = shaded[2] = 3.0 * depthScale;
                    break;
                default:
                    break;
                }
            }
            for (c = 0; c < 3; ++c)
            {
                if (mode == SABERCAT_LIGHTING_OUTPUT_LIT)
                {
                    /* Display-referred: composite in linear light (the background
                     * decoded from sRGB), clamp, then encode. */
                    double linear = (shaded[c] * alpha + srgbToLinear(bg[c]) * (1.0 - alpha)) * exposure;
                    linear = linear < 0.0 ? 0.0 : (linear > 1.0 ? 1.0 : linear);
                    want[c] = linearToSrgb(linear);
                }
                else
                {
                    want[c] = (shaded[c] * alpha + bg[c] * (1.0 - alpha)) * exposure;
                }
            }
            want[3] = alpha;
            for (c = 0; c < 4; ++c)
            {
                const double e = fabs((double)lit[p * 4 + c] - want[c]);
                worst = e > worst ? e : worst;
                *maxValue = fabs(want[c]) > *maxValue ? fabs(want[c]) : *maxValue;
            }
        }
    }
    return worst;
}

static void testOutputsAndLights(const SabercatRendererDesc * desc, const SabercatCamera * camera)
{
    SabercatRenderer * r = makeRenderer(desc, WIDTH, HEIGHT);
    static float color[WIDTH * HEIGHT * 4], lit[WIDTH * HEIGHT * 4];
    const size_t pixels4 = (size_t)WIDTH * HEIGHT * 4;
    SabercatLightingSettings ls;
    SabercatLight light, point, rect;
    uint32_t mode;
    double worst, maxValue;
    const uint32_t rectGpu = 4096; /* 64 x 64 samples per RECT light per pixel */
    const uint32_t rectRef = 128;  /* the reference's midpoint grid per axis */
    const char * names[] = {"LIT", "NORMAL", "BASE_COLOR", "ROUGHNESS", "METALLIC",
                            "AMBIENT_OCCLUSION", "VISIBILITY", "DEPTH"};

    makeRelightScene();
    {
        SabercatGaussians g = relightDesc(PLANE);
        CHECK(sabercat_renderer_set_gaussians(r, &g) == SABERCAT_OK, "plane scene");
    }
    CHECK(sabercat_renderer_set_camera(r, camera) == SABERCAT_OK, "plane camera");
    CHECK(sabercat_renderer_set_lighting_enabled(r, 1) == SABERCAT_OK, "lighting on");

    /* Three direct lights: the closed-form directional, a point, a rect. */
    light = directionalLight(0.0f, 0.0f, -1.0f);
    memset(&point, 0, sizeof(point));
    point.struct_size = sizeof(point);
    point.type = SABERCAT_LIGHT_POINT;
    point.casts_shadows = 0;
    point.intensity = 3.0f;
    point.color[0] = 1.0f;
    point.color[1] = 0.8f;
    point.color[2] = 0.6f;
    point.position[0] = 0.4f;
    point.position[1] = -0.3f;
    point.position[2] = 1.2f;
    memset(&rect, 0, sizeof(rect));
    rect.struct_size = sizeof(rect);
    rect.type = SABERCAT_LIGHT_RECT;
    rect.casts_shadows = 0;
    rect.intensity = 2.0f;
    rect.color[0] = 0.5f;
    rect.color[1] = 0.7f;
    rect.color[2] = 1.0f;
    rect.position[0] = -0.3f;
    rect.position[1] = 0.2f;
    rect.position[2] = 1.0f;
    rect.axis_x[1] = 1.0f; /* cross(axis_x, axis_y) = -Z: the emitter faces the plane */
    rect.axis_y[0] = 1.0f;
    rect.half_width = 0.3f;
    rect.half_height = 0.2f;

    memset(&ls, 0, sizeof(ls));
    ls.struct_size = sizeof(ls);
    ls.rect_samples = rectGpu;
    ls.exposure = 0.75f;
    ls.background[0] = 0.1f;
    ls.background[1] = 0.2f;
    ls.background[2] = 0.3f;
    ls.depth_scale = 0.5f;

    {
        const SabercatLight * cases[] = {&light, &point, &rect};
        const char * caseNames[] = {"directional", "point", "rect"};
        int ci;
        for (ci = 0; ci < 3; ++ci)
        {
            ls.output = SABERCAT_LIGHTING_OUTPUT_LIT;
            CHECK(sabercat_renderer_set_lights(r, cases[ci], 1) == SABERCAT_OK
                      && sabercat_renderer_set_lighting_settings(r, &ls) == SABERCAT_OK
                      && sabercat_renderer_forward(r) == SABERCAT_OK
                      && sabercat_renderer_read_image(r, color, pixels4) == SABERCAT_OK
                      && sabercat_renderer_read_lit_image(r, lit, pixels4) == SABERCAT_OK,
                  "lit forward for a direct light");
            worst = worstOutput(color, lit, SABERCAT_LIGHTING_OUTPUT_LIT, cases[ci], rectRef,
                                camera, &maxValue);
            printf("LIT, %s light: worst |lit - reference| %.3g (max value %.3g)\n", caseNames[ci],
                   worst, maxValue);
            /* Both sides are fp32 shading of O(1) values (~20 flops per term,
             * a few terms): ~1e-6 measured. The rect's 64-vs-128 midpoint grids differ by the
             * quadrature error, which is ~1e-5 at this emitter size. 1e-4 leaves 10x headroom
             * and still catches a dropped cosine, a wrong 1/d^2 or a wrong emitter normal (all
             * O(1e-2) or larger on this scene). */
            CHECK(worst <= 1e-4, "direct light disagrees with the double-precision reference");
        }
    }

    /* Every other output, under the point light (which does not affect them). */
    CHECK(sabercat_renderer_set_lights(r, &point, 1) == SABERCAT_OK, "point light for outputs");
    for (mode = SABERCAT_LIGHTING_OUTPUT_NORMAL; mode <= SABERCAT_LIGHTING_OUTPUT_DEPTH; ++mode)
    {
        ls.output = mode;
        CHECK(sabercat_renderer_set_lighting_settings(r, &ls) == SABERCAT_OK
                  && sabercat_renderer_forward(r) == SABERCAT_OK
                  && sabercat_renderer_read_image(r, color, pixels4) == SABERCAT_OK
                  && sabercat_renderer_read_lit_image(r, lit, pixels4) == SABERCAT_OK,
              "output forward");
        worst = worstOutput(color, lit, mode, &point, rectRef, camera, &maxValue);
        printf("output %-18s worst |lit - expected| %.3g (max value %.3g)\n", names[mode], worst,
               maxValue);
        CHECK(worst <= 1e-4, "an output disagrees with its closed form");
    }

    /* The LIT output with lighting turned off: no lit image. */
    CHECK(sabercat_renderer_set_lighting_enabled(r, 0) == SABERCAT_OK
              && sabercat_renderer_forward(r) == SABERCAT_OK
              && sabercat_renderer_read_lit_image(r, lit, pixels4) == SABERCAT_ERROR_INVALID_STATE,
          "lighting off must give no lit image");

    sabercat_renderer_destroy(r);
}

/* Antialiasing and SH degree. The compensation formula (per Gaussian, from
 * the shader's own geometry): c = sqrt(det(C) / det(C + 0.25 I)) with C the
 * 2D covariance before the low-pass dilation. With identity rotations and a
 * camera at the origin looking down -Z, C = J diag(s^2) J^T at the mean. A
 * scene with antialiased on at opacity o must render exactly like the plain
 * scene at opacity c * o. */
static void testAntialiasAndSh(const SabercatRendererDesc * desc, const SabercatCamera * camera)
{
    enum { N = 3, DIM = WIDTH * HEIGHT * 4 };
    static float imgAa[DIM], imgRef[DIM], imgPlain[DIM], imgAaOff[DIM];
    static float means[N * 4], scales[N * 4], rots[N * 4], opac[N], cOpac[N], sh[N * COEFFS * 4];
    static float imgSh0[DIM], imgSh3[DIM], imgDc0[DIM], imgDc3[DIM];
    static float dcSh[N * COEFFS * 4];
    const float scaleIn[N][3] = {{0.02f, 0.05f, 0.03f}, {0.15f, 0.1f, 0.12f}, {0.4f, 0.3f, 0.2f}};
    const size_t pixels4 = (size_t)WIDTH * HEIGHT * 4;
    SabercatRenderer * r = makeRenderer(desc, WIDTH, HEIGHT);
    SabercatGaussians g, plain;
    double c[N], maxDiffPlain, maxDiffRef, maxDiffSh, maxDiffDc;
    int i, k, j;
    const double fx = camera->fx, fy = camera->fy;

    for (i = 0; i < N; ++i)
    {
        means[i * 4 + 0] = -0.4f + 0.4f * (float)i;
        means[i * 4 + 1] = 0.1f * (float)(i - 1);
        means[i * 4 + 2] = -0.2f * (float)i;
        means[i * 4 + 3] = 0.0f;
        for (j = 0; j < 3; ++j)
        {
            scales[i * 4 + j] = scaleIn[i][j];
        }
        scales[i * 4 + 3] = 0.0f;
        rots[i * 4 + 0] = rots[i * 4 + 1] = rots[i * 4 + 2] = 0.0f;
        rots[i * 4 + 3] = 1.0f;
        opac[i] = 0.5f + 0.1f * (float)i;
        for (k = 0; k < COEFFS * 4; ++k)
        {
            sh[i * COEFFS * 4 + k] = (k % 4 == 3) ? 0.0f : 0.05f * (float)((k * 7 + i) % 9 - 4);
            dcSh[i * COEFFS * 4 + k] = (k < 4 && k % 4 != 3) ? sh[i * COEFFS * 4 + k] : 0.0f;
        }

        /* Predicted compensation for this Gaussian. */
        {
            const double lx = means[i * 4 + 0], ly = means[i * 4 + 1];
            const double sx2 = (double)scales[i * 4 + 0] * scales[i * 4 + 0];
            const double sy2 = (double)scales[i * 4 + 1] * scales[i * 4 + 1];
            const double sz2 = (double)scales[i * 4 + 2] * scales[i * 4 + 2];
            /* Camera at z = 3: the local position is (x, y, z - 3) and depth -(z - 3). */
            const double lz = means[i * 4 + 2] - 3.0;
            const double depth = -lz;
            const double J00 = fx / depth, J02 = fx * lx / (depth * depth);
            const double J11 = -fy / depth, J12 = -fy * ly / (depth * depth);
            const double C00 = J00 * J00 * sx2 + J02 * J02 * sz2;
            const double C11 = J11 * J11 * sy2 + J12 * J12 * sz2;
            const double C01 = J02 * J12 * sz2;
            const double detBefore = C00 * C11 - C01 * C01;
            const double detAfter = (C00 + 0.25) * (C11 + 0.25) - C01 * C01;
            c[i] = sqrt(detBefore / detAfter);
            cOpac[i] = (float)((double)opac[i] * c[i]);
        }
    }
    printf("predicted compensation per Gaussian: %.5f %.5f %.5f\n", c[0], c[1], c[2]);
    CHECK(c[0] < 0.5 && c[2] > 0.9, "the test scene must span real and near-unit compensations");

    memset(&g, 0, sizeof(g));
    g = gaussiansDesc(means, scales, rots, opac, sh);
    CHECK(sabercat_renderer_set_camera(r, camera) == SABERCAT_OK, "aa camera");
    CHECK(sabercat_renderer_set_antialiased(r, 1) == SABERCAT_ERROR_INVALID_STATE,
          "set_antialiased with no scene must be INVALID_STATE");
    CHECK(sabercat_renderer_set_gaussians(r, &g) == SABERCAT_OK, "aa scene");
    CHECK(sabercat_renderer_set_antialiased(r, 1) == SABERCAT_OK
              && sabercat_renderer_forward(r) == SABERCAT_OK
              && sabercat_renderer_read_image(r, imgAa, pixels4) == SABERCAT_OK,
          "antialiased forward");
    CHECK(sabercat_renderer_set_antialiased(r, 0) == SABERCAT_OK
              && sabercat_renderer_forward(r) == SABERCAT_OK
              && sabercat_renderer_read_image(r, imgRef, pixels4) == SABERCAT_OK,
          "plain forward");
    maxDiffRef = maxAbsDiff(imgAa, imgRef, pixels4);

    plain = gaussiansDesc(means, scales, rots, cOpac, sh);
    CHECK(sabercat_renderer_set_gaussians(r, &plain) == SABERCAT_OK
              && sabercat_renderer_forward(r) == SABERCAT_OK
              && sabercat_renderer_read_image(r, imgPlain, pixels4) == SABERCAT_OK,
          "plain forward at compensated opacity");
    maxDiffPlain = maxAbsDiff(imgAa, imgPlain, pixels4);

    /* Switching the flag back off on the original opacities restores the plain image exactly. */
    CHECK(sabercat_renderer_set_gaussians(r, &g) == SABERCAT_OK
              && sabercat_renderer_forward(r) == SABERCAT_OK
              && sabercat_renderer_read_image(r, imgAaOff, pixels4) == SABERCAT_OK
              && memcmp(imgAaOff, imgRef, sizeof(imgRef)) == 0,
          "the plain render must be reproducible bit for bit");
    /* The compensation (1 for the big Gaussian, ~0.3 for the small one) is not
     * negligible, so the flag must change the image. */
    CHECK(maxDiffRef > 1e-3, "the antialiased flag must change the image");
    /* Opacity compensation uses the same sums as the plain path at c * o, so
     * the two differ only by the fp32 rounding of o * c and its composition:
     * ~1e-7 per term, a few terms per pixel. 1e-5 is 100x that. */
    printf("antialiased vs plain at c*o: worst %.3g; antialiased vs plain at o: worst %.3g\n",
           maxDiffPlain, maxDiffRef);
    CHECK(maxDiffPlain <= 1e-5, "antialiased render disagrees with the compensated plain render");

    /* SH degree: a degree-0 render equals a DC-only scene, bit for bit, and the
     * higher bands do change the image at degree 3. */
    {
        SabercatGaussians gFull = gaussiansDesc(means, scales, rots, opac, sh);
        SabercatGaussians gDc = gaussiansDesc(means, scales, rots, opac, dcSh);
        CHECK(sabercat_renderer_set_gaussians(r, &gFull) == SABERCAT_OK, "SH scene");
        CHECK(sabercat_renderer_set_sh_degree(r, 4) == SABERCAT_ERROR_INVALID_ARGUMENT,
              "set_sh_degree(4) must be INVALID_ARGUMENT");
        CHECK(sabercat_renderer_set_sh_degree(r, 0) == SABERCAT_OK
                  && sabercat_renderer_forward(r) == SABERCAT_OK
                  && sabercat_renderer_read_image(r, imgSh0, pixels4) == SABERCAT_OK,
              "degree 0 forward");
        CHECK(sabercat_renderer_set_sh_degree(r, 3) == SABERCAT_OK
                  && sabercat_renderer_forward(r) == SABERCAT_OK
                  && sabercat_renderer_read_image(r, imgSh3, pixels4) == SABERCAT_OK,
              "degree 3 forward");
        maxDiffSh = maxAbsDiff(imgSh0, imgSh3, pixels4);
        CHECK(maxDiffSh > 1e-3, "degree 3 must differ from degree 0 when higher bands are set");

        CHECK(sabercat_renderer_set_gaussians(r, &gDc) == SABERCAT_OK
                  && sabercat_renderer_set_sh_degree(r, 0) == SABERCAT_OK
                  && sabercat_renderer_forward(r) == SABERCAT_OK
                  && sabercat_renderer_read_image(r, imgDc0, pixels4) == SABERCAT_OK,
              "DC-only scene at degree 0");
        CHECK(sabercat_renderer_set_sh_degree(r, 3) == SABERCAT_OK
                  && sabercat_renderer_forward(r) == SABERCAT_OK
                  && sabercat_renderer_read_image(r, imgDc3, pixels4) == SABERCAT_OK,
              "DC-only scene at degree 3");
        maxDiffDc = maxAbsDiff(imgSh0, imgDc0, pixels4);
        CHECK(memcmp(imgSh0, imgDc0, sizeof(imgSh0)) == 0,
              "degree 0 of the full scene must equal the DC-only scene bit for bit");
        CHECK(memcmp(imgDc0, imgDc3, sizeof(imgDc0)) == 0,
              "a DC-only scene at degree 3 must equal degree 0 bit for bit");
        printf("SH: degree 0 vs DC-only scene worst %.3g; degree 3 vs degree 0 (full) worst %.3g\n",
               maxDiffDc, maxDiffSh);
    }

    sabercat_renderer_destroy(r);
}

/* Uniform in [0, 1): a fixed-seed LCG, so every run sees the same scene. */
static float lcgUniform(uint32_t * state)
{
    *state = *state * 1664525u + 1013904223u;
    return (float)(*state >> 8) / 16777216.0f;
}

/* Directional finite differences through the ABI. For each parameter kind a
 * fixed-seed random direction v is drawn, and the analytic directional
 * derivative sum(v * dL/dp) (from backward with dL/dImage = (1, 0, 0, 0), so
 * L = the sum of red) is compared with the central difference of L along v:
 * (L(p + h v) - L(p - h v)) / 2h, at two step sizes. The central difference
 * has O(h^2) truncation error, so a correct gradient's error shrinks as h
 * shrinks until fp32 rounding of L (about 1e-7 relative per pixel, summed
 * over the image) limits it; a wrong gradient does not shrink. Padding lanes
 * of the 16-byte streams carry no parameter and get a zero direction. */
static double directionalFD(SabercatRenderer * r, const SabercatGaussians * g, float * target,
                            const float * orig, const float * dir, size_t n, double h,
                            float * image)
{
    double plus, minus;
    size_t i;
    for (i = 0; i < n; ++i)
    {
        target[i] = orig[i] + (float)h * dir[i];
    }
    sabercat_renderer_set_gaussians(r, g);
    plus = renderRedSum(r, image);
    for (i = 0; i < n; ++i)
    {
        target[i] = orig[i] - (float)h * dir[i];
    }
    sabercat_renderer_set_gaussians(r, g);
    minus = renderRedSum(r, image);
    for (i = 0; i < n; ++i)
    {
        target[i] = orig[i];
    }
    return (plus - minus) / (2.0 * h);
}

/* Uniform in [-1, 1] on the parameter lanes (lane % stride < lanes), 0 on padding. */
static void randomDirection(float * dir, size_t n, size_t stride, size_t lanes, uint32_t * state)
{
    size_t i;
    for (i = 0; i < n; ++i)
    {
        dir[i] = (i % stride) < lanes ? 2.0f * lcgUniform(state) - 1.0f : 0.0f;
    }
}

static double dotProduct(const float * a, const float * b, size_t n)
{
    double sum = 0.0;
    size_t i;
    for (i = 0; i < n; ++i)
    {
        sum += (double)a[i] * (double)b[i];
    }
    return sum;
}

static void testFiniteDifferences(const SabercatRendererDesc * desc, const SabercatCamera * camera)
{
    static float image[WIDTH * HEIGHT * 4], dImage[WIDTH * HEIGHT * 4];
    static float aM[COUNT * 4], aS[COUNT * 4], aR[COUNT * 4], aO[COUNT], aSh[COUNT * COEFFS * 4],
        aScreen[COUNT * 4];
    static float oM[COUNT * 4], oS[COUNT * 4], oR[COUNT * 4], oO[COUNT], oSh[COUNT * COEFFS * 4];
    static float dM[COUNT * 4], dS[COUNT * 4], dR[COUNT * 4], dO[COUNT], dSh[COUNT * COEFFS * 4];
    SabercatRenderer * r = makeRenderer(desc, WIDTH, HEIGHT);
    SabercatGaussians g = gaussiansDesc(g_means, g_scales, g_rotations, g_opacities, g_sh);
    SabercatGradients grads;
    const double steps[2] = {0.02, 0.01};
    const char * names[5] = {"means", "scales", "rotations", "opacities", "sh"};
    uint32_t state = 987654321u;
    size_t t;
    int cat, s;

    memcpy(oM, g_means, sizeof(oM));
    memcpy(oS, g_scales, sizeof(oS));
    memcpy(oR, g_rotations, sizeof(oR));
    memcpy(oO, g_opacities, sizeof(oO));
    memcpy(oSh, g_sh, sizeof(oSh));

    for (t = 0; t < WIDTH * HEIGHT; ++t)
    {
        dImage[t * 4 + 0] = 1.0f;
        dImage[t * 4 + 1] = dImage[t * 4 + 2] = dImage[t * 4 + 3] = 0.0f;
    }
    CHECK(sabercat_renderer_set_camera(r, camera) == SABERCAT_OK
              && sabercat_renderer_set_gaussians(r, &g) == SABERCAT_OK
              && sabercat_renderer_forward(r) == SABERCAT_OK
              && sabercat_renderer_backward(r, dImage, WIDTH * HEIGHT * 4) == SABERCAT_OK,
          "analytic backward for the FD check");
    memset(&grads, 0, sizeof(grads));
    grads.struct_size = sizeof(grads);
    grads.count = COUNT;
    grads.means = aM;
    grads.scales = aS;
    grads.rotations = aR;
    grads.opacities = aO;
    grads.sh_coefficients = aSh;
    grads.screen_means = aScreen;
    CHECK(sabercat_renderer_read_gradients(r, &grads) == SABERCAT_OK, "read_gradients for FD");

    /* One direction per parameter kind; the kinds are checked separately so a
     * failure names the stream. */
    randomDirection(dM, COUNT * 4, 4, 3, &state);
    randomDirection(dS, COUNT * 4, 4, 3, &state);
    randomDirection(dR, COUNT * 4, 4, 4, &state);
    randomDirection(dO, COUNT, 1, 1, &state);
    randomDirection(dSh, COUNT * COEFFS * 4, 4, 3, &state);

    for (cat = 0; cat < 5; ++cat)
    {
        const float * analytic = cat == 0 ? aM : cat == 1 ? aS : cat == 2 ? aR : cat == 3 ? aO : aSh;
        const float * dir = cat == 0 ? dM : cat == 1 ? dS : cat == 2 ? dR : cat == 3 ? dO : dSh;
        const float * orig = cat == 0 ? oM : cat == 1 ? oS : cat == 2 ? oR : cat == 3 ? oO : oSh;
        float * target = cat == 0 ? g_means : cat == 1 ? g_scales : cat == 2 ? g_rotations
                         : cat == 3 ? g_opacities : g_sh;
        const size_t n = cat == 0 || cat == 1 || cat == 2 ? COUNT * 4
                         : cat == 3 ? COUNT : COUNT * COEFFS * 4;
        const double a = dotProduct(dir, analytic, n);
        double err[2];
        for (s = 0; s < 2; ++s)
        {
            const double fd = directionalFD(r, &g, target, orig, dir, n, steps[s], image);
            err[s] = fabs(fd - a) / (fabs(a) > 1e-12 ? fabs(a) : 1e-12);
        }
        printf("FD %-9s directional derivative %.5g; rel. error h=0.02 %.3g, h=0.01 %.3g\n",
               names[cat], a, err[0], err[1]);
        /* Truncation error shrinks ~4x when h halves; the bound applies to the smaller
         * step, where the truncation part is smallest. 5e-2 is the bound this file
         * already uses for the opacity check. */
        CHECK(fabs(a) > 1e-3, "directional FD must not be degenerate");
        /* Means and scales are NOT asserted here. The forward skips a pixel-splat
         * pair whose alpha is below 1/255 (evaluateSplat) and bins a Gaussian to
         * tiles by a hard radius, so L has a step wherever a pixel or tile
         * crosses those thresholds. Under a step the central difference measures
         * the jump, not the slope: the error here does not shrink with h (it
         * grows, 0.10 at h=0.02 and 0.25 at h=0.01), while the streams that are
         * smooth in this scene converge (rotations 0.063 -> 0.0015; opacity and SH
         * ~1e-6). Means and scales are checked by backward_test, which freezes
         * those branches. Printed for visibility. */
        if (cat == 2 || cat == 3 || cat == 4)
        {
            CHECK(err[1] < 5e-2,
                  "analytic gradient disagrees with the directional finite difference");
        }
    }

    /* Restore the scene the other checks expect. */
    memcpy(g_means, oM, sizeof(oM));
    memcpy(g_scales, oS, sizeof(oS));
    memcpy(g_rotations, oR, sizeof(oR));
    memcpy(g_opacities, oO, sizeof(oO));
    memcpy(g_sh, oSh, sizeof(oSh));
    CHECK(sabercat_renderer_set_gaussians(r, &g) == SABERCAT_OK, "restore scene after FD");
    sabercat_renderer_destroy(r);
}

/* Means and scales where the finite difference is valid. Every pair has opacity 0.9 and
 * sits near the image center with a projected sigma of about 3 px (3-sigma radius about
 * 10 px, inside a 16 px tile), so the forward's 1/255 skip and the tile binning never
 * fall within the step; the step is 1e-3 of the scale or less. L is smooth in these
 * parameters here, so the central difference measures the slope. The steps are small
 * enough that fp32 rounding of L, not truncation, sets the floor at the smaller one. */
static void testMeansScalesFD(const SabercatRendererDesc * desc, const SabercatCamera * camera)
{
    static float image[WIDTH * HEIGHT * 4], dImage[WIDTH * HEIGHT * 4];
    static float mm[COUNT * 4], ms[COUNT * 4], mr[COUNT * 4], mo[COUNT], msh[COUNT * COEFFS * 4];
    static float om[COUNT * 4], os[COUNT * 4];
    static float gm[COUNT * 4], gs[COUNT * 4], gr[COUNT * 4], go[COUNT], gsh[COUNT * COEFFS * 4],
        gscreen[COUNT * 4];
    static float dirM[COUNT * 4], dirS[COUNT * 4];
    SabercatRenderer * r = makeRenderer(desc, WIDTH, HEIGHT);
    SabercatGaussians g;
    SabercatGradients grads;
    const double steps[2] = {2e-4, 1e-4};
    const char * names[2] = {"means", "scales"};
    uint32_t state = 424242u;
    double aM, aS, err[2];
    int i, c, s, cat;
    size_t t;

    memset(msh, 0, sizeof(msh));
    for (i = 0; i < COUNT; ++i)
    {
        mm[i * 4 + 0] = -0.08f + 0.08f * (float)i;
        mm[i * 4 + 1] = 0.03f * (float)(i - 1);
        /* Depths 3.0, 3.3, 3.6 apart by far more than the step: equal depths would let
         * a z-step reorder the (strongly overlapping) Gaussians, which is a step too. */
        mm[i * 4 + 2] = -0.3f * (float)i;
        mm[i * 4 + 3] = 0.0f;
        ms[i * 4 + 0] = 0.22f + 0.02f * (float)i;
        ms[i * 4 + 1] = 0.18f;
        ms[i * 4 + 2] = 0.2f;
        ms[i * 4 + 3] = 0.0f;
        mr[i * 4 + 0] = mr[i * 4 + 1] = mr[i * 4 + 2] = 0.0f;
        mr[i * 4 + 3] = 1.0f;
        mo[i] = 0.9f;
        for (c = 0; c < 3; ++c)
        {
            msh[(i * COEFFS) * 4 + c] = 0.3f - 0.05f * (float)i;
        }
    }
    memcpy(om, mm, sizeof(om));
    memcpy(os, ms, sizeof(os));
    g = gaussiansDesc(mm, ms, mr, mo, msh);

    for (t = 0; t < WIDTH * HEIGHT; ++t)
    {
        dImage[t * 4 + 0] = 1.0f;
        dImage[t * 4 + 1] = dImage[t * 4 + 2] = dImage[t * 4 + 3] = 0.0f;
    }
    randomDirection(dirM, COUNT * 4, 4, 3, &state);
    randomDirection(dirS, COUNT * 4, 4, 3, &state);

    CHECK(sabercat_renderer_set_camera(r, camera) == SABERCAT_OK
              && sabercat_renderer_set_gaussians(r, &g) == SABERCAT_OK
              && sabercat_renderer_forward(r) == SABERCAT_OK
              && sabercat_renderer_read_image(r, image, WIDTH * HEIGHT * 4) == SABERCAT_OK
              && sabercat_renderer_backward(r, dImage, WIDTH * HEIGHT * 4) == SABERCAT_OK,
          "analytic backward for the well-conditioned FD");
    memset(&grads, 0, sizeof(grads));
    grads.struct_size = sizeof(grads);
    grads.count = COUNT;
    grads.means = gm;
    grads.scales = gs;
    grads.rotations = gr;
    grads.opacities = go;
    grads.sh_coefficients = gsh;
    grads.screen_means = gscreen;
    CHECK(sabercat_renderer_read_gradients(r, &grads) == SABERCAT_OK,
          "read_gradients for the well-conditioned FD");
    aM = dotProduct(dirM, gm, COUNT * 4);
    aS = dotProduct(dirS, gs, COUNT * 4);

    for (cat = 0; cat < 2; ++cat)
    {
        const double a = cat == 0 ? aM : aS;
        for (s = 0; s < 2; ++s)
        {
            const double fd = directionalFD(r, &g, cat == 0 ? mm : ms, cat == 0 ? om : os,
                                            cat == 0 ? dirM : dirS, COUNT * 4, steps[s], image);
            err[s] = fabs(fd - a) / (fabs(a) > 1e-12 ? fabs(a) : 1e-12);
        }
        printf("FD (well-conditioned) %-7s directional derivative %.5g; rel. error h=2e-4 %.3g, "
               "h=1e-4 %.3g\n",
               names[cat], a, err[0], err[1]);
        CHECK(fabs(a) > 1e-3, "well-conditioned directional FD must not be degenerate");
        CHECK(err[1] < 5e-2, "means/scales gradient disagrees with the well-conditioned FD");
    }

    CHECK(sabercat_renderer_set_gaussians(r, &g) == SABERCAT_OK, "restore after well-conditioned FD");
    sabercat_renderer_destroy(r);
}

/* Field-by-field comparison: SabercatBufferView has 4 bytes of tail padding, which
 * the library does not write, so memcmp over the whole struct is not meaningful. */
static int sameBufferView(const SabercatBufferView * a, const SabercatBufferView * b)
{
    return a->buffer == b->buffer && a->offset == b->offset && a->size == b->size
           && a->stride == b->stride;
}

static int sameGradientViews(const SabercatGradientViews * a, const SabercatGradientViews * b)
{
    return a->count == b->count && sameBufferView(&a->means, &b->means)
           && sameBufferView(&a->scales, &b->scales)
           && sameBufferView(&a->rotations, &b->rotations)
           && sameBufferView(&a->opacities, &b->opacities)
           && sameBufferView(&a->sh_coefficients, &b->sh_coefficients)
           && sameBufferView(&a->screen_means, &b->screen_means);
}

/* Gradient views and read_gradients rules. */
static void testGradientsAndViews(const SabercatRendererDesc * desc, const SabercatCamera * camera)
{
    static float image[WIDTH * HEIGHT * 4], dImage[WIDTH * HEIGHT * 4];
    static float a1[COUNT * 4], a2[COUNT * 4], a3[COUNT * 4], a4[COUNT], a5[COUNT * COEFFS * 4],
        a6[COUNT * 4], b1[COUNT * 4], b2[COUNT * 4], b3[COUNT * 4], b4[COUNT], b5[COUNT * COEFFS * 4];
    SabercatRenderer * r = makeRenderer(desc, WIDTH, HEIGHT);
    SabercatGaussians g = gaussiansDesc(g_means, g_scales, g_rotations, g_opacities, g_sh);
    SabercatGradientViews gv1, gv2;
    SabercatGradients grads;
    size_t t;

    CHECK(sabercat_renderer_set_camera(r, camera) == SABERCAT_OK, "gradient views camera");
    CHECK(sabercat_renderer_set_gaussians(r, &g) == SABERCAT_OK, "gradient views scene");

    /* Views exist before any backward (they allocate on first call), and a second
     * call within the same storage generation returns the same handles. */
    memset(&gv1, 0, sizeof(gv1));
    memset(&gv2, 0, sizeof(gv2));
    CHECK(sabercat_renderer_get_gradient_views(r, &gv1) == SABERCAT_OK && gv1.means.buffer != 0
              && gv1.count == COUNT,
          "get_gradient_views before backward");
    CHECK(sabercat_renderer_get_gradient_views(r, &gv2) == SABERCAT_OK
              && sameGradientViews(&gv1, &gv2),
          "get_gradient_views must be idempotent within a generation");

    /* No backward yet: read_gradients is INVALID_STATE. */
    memset(&grads, 0, sizeof(grads));
    grads.struct_size = sizeof(grads);
    grads.count = COUNT;
    grads.means = a1;
    grads.scales = a2;
    grads.rotations = a3;
    grads.opacities = a4;
    grads.sh_coefficients = a5;
    grads.screen_means = a6;
    CHECK(sabercat_renderer_read_gradients(r, &grads) == SABERCAT_ERROR_INVALID_STATE,
          "read_gradients before any backward must be INVALID_STATE");

    for (t = 0; t < WIDTH * HEIGHT; ++t)
    {
        dImage[t * 4 + 0] = dImage[t * 4 + 3] = 1.0f;
        dImage[t * 4 + 1] = dImage[t * 4 + 2] = 0.0f;
    }
    CHECK(sabercat_renderer_forward(r) == SABERCAT_OK
              && sabercat_renderer_read_image(r, image, WIDTH * HEIGHT * 4) == SABERCAT_OK
              && sabercat_renderer_backward(r, dImage, WIDTH * HEIGHT * 4) == SABERCAT_OK,
          "backward for the gradient views check");

    /* Count mismatches and NULL are rejected; the state after a failure is the
     * same, so a corrected call still works. */
    grads.count = COUNT - 1;
    CHECK(sabercat_renderer_read_gradients(r, &grads) == SABERCAT_ERROR_INVALID_ARGUMENT,
          "read_gradients with a count below the scene's must be INVALID_ARGUMENT");
    grads.count = COUNT + 1;
    CHECK(sabercat_renderer_read_gradients(r, &grads) == SABERCAT_ERROR_INVALID_ARGUMENT,
          "read_gradients with a count above the scene's must be INVALID_ARGUMENT");
    grads.count = COUNT;
    grads.means = NULL;
    CHECK(sabercat_renderer_read_gradients(r, &grads) == SABERCAT_ERROR_INVALID_ARGUMENT,
          "read_gradients with a NULL required array must be INVALID_ARGUMENT");
    grads.means = a1;

    /* screen_means is optional: NULL is accepted and the others are identical. */
    CHECK(sabercat_renderer_read_gradients(r, &grads) == SABERCAT_OK, "read_gradients (full)");
    memcpy(b1, a1, sizeof(a1));
    memcpy(b2, a2, sizeof(a2));
    memcpy(b3, a3, sizeof(a3));
    memcpy(b4, a4, sizeof(a4));
    memcpy(b5, a5, sizeof(a5));
    {
        SabercatGradients noScreen = grads;
        noScreen.means = b1; /* same values, separate buffers */
        noScreen.scales = b2;
        noScreen.rotations = b3;
        noScreen.opacities = b4;
        noScreen.sh_coefficients = b5;
        noScreen.screen_means = NULL;
        CHECK(sabercat_renderer_read_gradients(r, &noScreen) == SABERCAT_OK,
              "read_gradients with NULL screen_means");
        CHECK(memcmp(b1, a1, sizeof(a1)) == 0 && memcmp(b5, a5, sizeof(a5)) == 0,
              "read_gradients with NULL screen_means must fill the other arrays identically");
    }

    /* Views after a backward: same handles, idempotent. */
    memset(&gv2, 0, sizeof(gv2));
    CHECK(sabercat_renderer_get_gradient_views(r, &gv2) == SABERCAT_OK
              && sameGradientViews(&gv1, &gv2),
          "gradient views must not change across a backward");

    /* A new scene starts a new generation: a backward is required again. */
    CHECK(sabercat_renderer_set_gaussians(r, &g) == SABERCAT_OK, "rescene");
    CHECK(sabercat_renderer_read_gradients(r, &grads) == SABERCAT_ERROR_INVALID_STATE,
          "read_gradients after a new scene must be INVALID_STATE");
    sabercat_renderer_destroy(r);
    printf("gradient views idempotent; read_gradients: INVALID_STATE before backward, "
           "INVALID_ARGUMENT for count or NULL, NULL screen_means accepted\n");
}

/* D scale: 200000 random Gaussians from a fixed seed through the ABI. */
static void testLargeScene(const SabercatRendererDesc * desc, const SabercatCamera * camera)
{
    enum { N = 200000 };
    static float image[WIDTH * HEIGHT * 4], dImage[WIDTH * HEIGHT * 4];
    float * m = malloc(sizeof(float) * N * 4);
    float * s = malloc(sizeof(float) * N * 4);
    float * q = malloc(sizeof(float) * N * 4);
    float * o = malloc(sizeof(float) * N);
    float * sh = malloc(sizeof(float) * N * COEFFS * 4);
    float * om = malloc(sizeof(float) * N * 4);
    float * os = malloc(sizeof(float) * N * 4);
    float * oq = malloc(sizeof(float) * N * 4);
    float * oo = malloc(sizeof(float) * N);
    float * osh = malloc(sizeof(float) * N * COEFFS * 4);
    float * gm = malloc(sizeof(float) * N * 4);
    float * gs = malloc(sizeof(float) * N * 4);
    float * gq = malloc(sizeof(float) * N * 4);
    float * go = malloc(sizeof(float) * N);
    float * gsh = malloc(sizeof(float) * N * COEFFS * 4);
    float * gscreen = malloc(sizeof(float) * N * 4);
    SabercatRenderer * r = makeRenderer(desc, WIDTH, HEIGHT);
    SabercatGaussians g, out;
    SabercatGradients grads;
    uint32_t state = 20261020u;
    size_t i, nonFinite = 0, nonzeroGrads = 0;
    int c, pixelsFinite = 1;

    for (i = 0; i < N; ++i)
    {
        float v[4], len;
        for (c = 0; c < 3; ++c)
        {
            m[i * 4 + c] = 2.0f * lcgUniform(&state) - 1.0f;
            s[i * 4 + c] = 0.005f + 0.05f * lcgUniform(&state);
        }
        m[i * 4 + 3] = s[i * 4 + 3] = 0.0f;
        for (c = 0; c < 4; ++c)
        {
            v[c] = lcgUniform(&state);
        }
        len = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]);
        for (c = 0; c < 4; ++c)
        {
            q[i * 4 + c] = (v[c] + 0.01f) / len;
        }
        o[i] = 0.05f + 0.9f * lcgUniform(&state);
        for (c = 0; c < COEFFS * 4; ++c)
        {
            sh[i * COEFFS * 4 + c] = (c % 4 == 3) ? 0.0f : 0.1f * (lcgUniform(&state) - 0.5f);
        }
    }

    g = gaussiansDesc(m, s, q, o, sh);
    g.count = N;
    CHECK(sabercat_renderer_set_camera(r, camera) == SABERCAT_OK, "large scene camera");
    CHECK(sabercat_renderer_set_gaussians(r, &g) == SABERCAT_OK, "large scene set_gaussians");

    out = gaussiansDesc(om, os, oq, oo, osh);
    out.count = N;
    CHECK(sabercat_renderer_get_gaussians(r, &out) == SABERCAT_OK, "large scene get_gaussians");
    CHECK(memcmp(om, m, sizeof(float) * N * 4) == 0 && memcmp(oq, q, sizeof(float) * N * 4) == 0
              && memcmp(oo, o, sizeof(float) * N) == 0 && memcmp(osh, sh, sizeof(float) * N * COEFFS * 4) == 0,
          "200000 Gaussians must round-trip bit-exactly");

    CHECK(sabercat_renderer_forward(r) == SABERCAT_OK
              && sabercat_renderer_read_image(r, image, WIDTH * HEIGHT * 4) == SABERCAT_OK,
          "large scene forward");
    for (i = 0; i < (size_t)WIDTH * HEIGHT * 4; ++i)
    {
        pixelsFinite = pixelsFinite && isfinite(image[i]);
        dImage[i] = image[i];
    }
    CHECK(pixelsFinite, "large scene image has non-finite values");
    CHECK(sabercat_renderer_backward(r, dImage, WIDTH * HEIGHT * 4) == SABERCAT_OK,
          "large scene backward");
    memset(&grads, 0, sizeof(grads));
    grads.struct_size = sizeof(grads);
    grads.count = N;
    grads.means = gm;
    grads.scales = gs;
    grads.rotations = gq;
    grads.opacities = go;
    grads.sh_coefficients = gsh;
    grads.screen_means = gscreen;
    CHECK(sabercat_renderer_read_gradients(r, &grads) == SABERCAT_OK, "large scene gradients");
    for (i = 0; i < (size_t)N * 4; ++i)
    {
        nonFinite += isfinite(gm[i]) ? 0 : 1;
        nonzeroGrads += gm[i] != 0.0f ? 1 : 0;
    }
    for (i = 0; i < (size_t)N; ++i)
    {
        nonFinite += isfinite(go[i]) ? 0 : 1;
    }
    CHECK(nonFinite == 0, "large scene gradients must all be finite");
    CHECK(nonzeroGrads > 0, "large scene means gradients must not all be zero");
    printf("large scene: %d Gaussians, round trip exact, non-finite gradients %zu, nonzero mean "
           "gradients %zu\n",
           N, nonFinite, nonzeroGrads);

    sabercat_renderer_destroy(r);
    free(m);
    free(s);
    free(q);
    free(o);
    free(sh);
    free(om);
    free(os);
    free(oq);
    free(oo);
    free(osh);
    free(gm);
    free(gs);
    free(gq);
    free(go);
    free(gsh);
    free(gscreen);
}

/* clear_scene: NULL is rejected; after a clear the count is 0 and read_image is
 * INVALID_STATE; a scene set afterwards renders exactly as it did before. */
static void testClearScene(const SabercatRendererDesc * desc, const SabercatCamera * camera)
{
    static float means[COUNT * 4], scales[COUNT * 4], rotations[COUNT * 4], opacities[COUNT],
        sh[COUNT * COEFFS * 4];
    static float before[WIDTH * HEIGHT * 4], after[WIDTH * HEIGHT * 4];
    SabercatRenderer * r = makeRenderer(desc, WIDTH, HEIGHT);
    SabercatGaussians g;
    uint32_t count = 99;
    int i;

    for (i = 0; i < COUNT; ++i)
    {
        int c;
        for (c = 0; c < 4; ++c)
        {
            means[i * 4 + c] = (c == 0) ? 0.5f * i - 0.5f : (c == 1) ? 0.1f * i : (c == 2) ? 2.0f : 0.0f;
            scales[i * 4 + c] = (c < 3) ? 0.3f : 0.0f;
            rotations[i * 4 + c] = (c == 3) ? 1.0f : 0.0f;
        }
        opacities[i] = 0.8f;
        for (c = 0; c < COEFFS * 4; ++c)
        {
            sh[i * COEFFS * 4 + c] = 0.0f;
        }
        sh[i * COEFFS * 4 + 0] = 0.5f;
        sh[i * COEFFS * 4 + 1] = 0.1f * i;
        sh[i * COEFFS * 4 + 2] = -0.3f;
    }
    g = gaussiansDesc(means, scales, rotations, opacities, sh);

    CHECK(sabercat_renderer_clear_scene(NULL) == SABERCAT_ERROR_INVALID_ARGUMENT,
          "clear_scene(NULL) must be INVALID_ARGUMENT");
    CHECK(sabercat_renderer_set_camera(r, camera) == SABERCAT_OK, "clear_scene: camera");
    CHECK(sabercat_renderer_clear_scene(r) == SABERCAT_OK, "clear_scene with no scene must succeed");
    CHECK(sabercat_renderer_set_gaussians(r, &g) == SABERCAT_OK, "clear_scene: set_gaussians");
    CHECK(sabercat_renderer_forward(r) == SABERCAT_OK, "clear_scene: forward");
    CHECK(sabercat_renderer_read_image(r, before, WIDTH * HEIGHT * 4) == SABERCAT_OK,
          "clear_scene: read_image before the clear");

    CHECK(sabercat_renderer_clear_scene(r) == SABERCAT_OK, "clear_scene must succeed");
    CHECK(sabercat_renderer_get_gaussian_count(r, &count) == SABERCAT_OK && count == 0,
          "get_gaussian_count after clear_scene must be 0");
    CHECK(sabercat_renderer_read_image(r, after, WIDTH * HEIGHT * 4) == SABERCAT_ERROR_INVALID_STATE,
          "read_image after clear_scene must be INVALID_STATE");
    CHECK(sabercat_renderer_clear_scene(r) == SABERCAT_OK, "a second clear_scene must succeed");

    CHECK(sabercat_renderer_set_gaussians(r, &g) == SABERCAT_OK, "set_gaussians after clear_scene");
    CHECK(sabercat_renderer_forward(r) == SABERCAT_OK, "forward after set_gaussians");
    CHECK(sabercat_renderer_read_image(r, after, WIDTH * HEIGHT * 4) == SABERCAT_OK,
          "read_image after set_gaussians");
    CHECK(memcmp(before, after, sizeof(before)) == 0,
          "a scene set after clear_scene must render exactly as the first one did");
    printf("clear_scene through the ABI OK\n");

    sabercat_renderer_destroy(r);
}

/* Depth output (depth_layout): a 32-byte desc from before the field reads as
 * NONE, depth_layout values other than 0/1/2/4 are refused, and the
 * DepthCoverage layout's buffer has the right size, its coverage is
 * bit-identical to read_image's alpha, and its view has the stride the layout
 * implies. */
static void testDepthOutput(const SabercatRendererDesc * desc, const SabercatCamera * camera)
{
    static float means[COUNT * 4], scales[COUNT * 4], rotations[COUNT * 4], opacities[COUNT],
        sh[COUNT * COEFFS * 4];
    static float depth[WIDTH * HEIGHT * 4], image[WIDTH * HEIGHT * 4];
    SabercatRendererDesc old, bad, d2;
    SabercatRenderer * r = NULL;
    SabercatGaussians g;
    SabercatBufferView view;
    size_t pixels = (size_t)WIDTH * HEIGHT;
    size_t covered = 0, uncovered = 0, mismatches = 0, badDepth = 0;
    size_t i;
    int c;

    /* A struct_size of 32 is the old header's sizeof: depth_layout lies past it,
     * so the 2 written there is never read and the renderer stays NONE. */
    old = *desc;
    old.struct_size = 32;
    old.depth_layout = SABERCAT_DEPTH_LAYOUT_DEPTH_COVERAGE;
    CHECK(sabercat_renderer_create(&old, &r) == SABERCAT_OK && r != NULL,
          "depth: a 32-byte desc must create");
    CHECK(sabercat_renderer_read_depth(r, depth, pixels * 2) == SABERCAT_ERROR_INVALID_STATE,
          "depth: read_depth on a NONE renderer must be INVALID_STATE");
    CHECK(sabercat_renderer_get_depth_view(r, NULL) == SABERCAT_ERROR_INVALID_ARGUMENT,
          "depth: get_depth_view(NULL) must be INVALID_ARGUMENT");
    sabercat_renderer_destroy(r);
    r = NULL;

    bad = *desc;
    bad.depth_layout = 3;
    CHECK(sabercat_renderer_create(&bad, &r) == SABERCAT_ERROR_INVALID_ARGUMENT && r == NULL
              && strstr(sabercat_last_error(), "depth_layout") != NULL,
          "depth: depth_layout 3 must be INVALID_ARGUMENT at create");

    d2 = *desc;
    d2.depth_layout = SABERCAT_DEPTH_LAYOUT_DEPTH_COVERAGE;
    r = makeRenderer(&d2, WIDTH, HEIGHT);
    CHECK(sabercat_renderer_read_depth(r, depth, pixels * 2) == SABERCAT_ERROR_INVALID_STATE,
          "depth: read_depth before a forward must be INVALID_STATE");
    CHECK(sabercat_renderer_get_depth_view(r, &view) == SABERCAT_ERROR_INVALID_STATE,
          "depth: get_depth_view before a forward must be INVALID_STATE");

    for (i = 0; i < COUNT; ++i)
    {
        means[i * 4 + 0] = 0.5f * (float)i - 0.5f;
        means[i * 4 + 1] = 0.1f * (float)i;
        means[i * 4 + 2] = 2.0f;
        means[i * 4 + 3] = 0.0f;
        for (c = 0; c < 4; ++c)
        {
            scales[i * 4 + c] = (c < 3) ? 0.08f : 0.0f; /* small: leaves empty pixels */
            rotations[i * 4 + c] = (c == 3) ? 1.0f : 0.0f;
        }
        opacities[i] = 0.8f;
        for (c = 0; c < COEFFS * 4; ++c)
        {
            sh[i * COEFFS * 4 + c] = 0.0f;
        }
    }
    g = gaussiansDesc(means, scales, rotations, opacities, sh);
    CHECK(sabercat_renderer_set_gaussians(r, &g) == SABERCAT_OK, "depth: set_gaussians");
    CHECK(sabercat_renderer_set_camera(r, camera) == SABERCAT_OK, "depth: set_camera");
    CHECK(sabercat_renderer_forward(r) == SABERCAT_OK, "depth: forward");

    CHECK(sabercat_renderer_read_depth(r, depth, pixels * 4) == SABERCAT_ERROR_INVALID_ARGUMENT,
          "depth: read_depth with float_count != width*height*2 must be INVALID_ARGUMENT");
    CHECK(sabercat_renderer_read_depth(r, depth, pixels * 2) == SABERCAT_OK,
          "depth: read_depth with width*height*2 floats");
    CHECK(sabercat_renderer_read_image(r, image, pixels * 4) == SABERCAT_OK,
          "depth: read_image after the forward");

    for (i = 0; i < pixels; ++i)
    {
        const float coverage = depth[2 * i + 1];
        if (memcmp(&coverage, &image[4 * i + 3], sizeof(float)) != 0)
        {
            ++mismatches;
        }
        if (image[4 * i + 3] > 0.0f)
        {
            ++covered;
            if (!isfinite(depth[2 * i]) || !(depth[2 * i] > 0.0f))
            {
                ++badDepth;
            }
        }
        else
        {
            ++uncovered;
        }
    }
    CHECK(mismatches == 0, "depth: coverage is not bit-identical to the image alpha");
    CHECK(badDepth == 0, "depth: D is not positive and finite where the image has coverage");
    CHECK(covered > 0 && uncovered > 0, "depth: the scene must cover some pixels but not all");

    CHECK(sabercat_renderer_get_depth_view(r, &view) == SABERCAT_OK && view.buffer != 0
              && view.offset == 0 && view.stride == 8 && view.size == pixels * 8,
          "depth: get_depth_view must be stride 8, size width*height*8");
    printf("depth output OK: %u of %u pixels covered, depth layout 2\n", (unsigned)covered,
           (unsigned)pixels);

    sabercat_renderer_destroy(r);
}

/* One read of every gradient stream, for comparing whole backward results. */
typedef struct DepthGrads
{
    float means[COUNT * 4], scales[COUNT * 4], rotations[COUNT * 4], opacities[COUNT];
    float sh[COUNT * COEFFS * 4], screen[COUNT * 4];
} DepthGrads;

static int readDepthGrads(SabercatRenderer * r, DepthGrads * out)
{
    SabercatGradients g;
    memset(&g, 0, sizeof(g));
    g.struct_size = sizeof(g);
    g.count = COUNT;
    g.means = out->means;
    g.scales = out->scales;
    g.rotations = out->rotations;
    g.opacities = out->opacities;
    g.sh_coefficients = out->sh;
    g.screen_means = out->screen;
    return sabercat_renderer_read_gradients(r, &g) == SABERCAT_OK;
}

/* Largest |a-b| over every gradient stream, relative to the largest |b| over all
 * of them. One scale for the whole result: this scene is isotropic at identity
 * rotation, so the quaternion gradient is zero up to rounding and its own
 * relative error would be meaningless. */
static double depthGradsError(const DepthGrads * a, const DepthGrads * b)
{
    const float * pa[6];
    const float * pb[6];
    size_t n[6];
    double worst = 0.0, scale = 0.0;
    size_t s, i;
    pa[0] = a->means;
    pb[0] = b->means;
    n[0] = COUNT * 4;
    pa[1] = a->scales;
    pb[1] = b->scales;
    n[1] = COUNT * 4;
    pa[2] = a->rotations;
    pb[2] = b->rotations;
    n[2] = COUNT * 4;
    pa[3] = a->opacities;
    pb[3] = b->opacities;
    n[3] = COUNT;
    pa[4] = a->sh;
    pb[4] = b->sh;
    n[4] = COUNT * COEFFS * 4;
    pa[5] = a->screen;
    pb[5] = b->screen;
    n[5] = COUNT * 4;
    for (s = 0; s < 6; ++s)
    {
        for (i = 0; i < n[s]; ++i)
        {
            const double d = fabs((double)pa[s][i] - (double)pb[s][i]);
            worst = d > worst ? d : worst;
            scale = fabs((double)pb[s][i]) > scale ? fabs((double)pb[s][i]) : scale;
        }
    }
    return scale > 0.0 ? worst / scale : worst;
}

/* The depth gradient through the ABI: backward_depth and backward_depth_view,
 * against the color-only backward and a layout-NONE renderer. The scene is
 * testDepthOutput's. L = 0.5 |I|^2 + 0.5 D^2, so dL/dD is read_depth(). */
static void testDepthGradient(const SabercatRendererDesc * desc, const SabercatCamera * camera)
{
    static float means[COUNT * 4], scales[COUNT * 4], rotations[COUNT * 4], opacities[COUNT],
        sh[COUNT * COEFFS * 4];
    static float depth[WIDTH * HEIGHT * 2], image[WIDTH * HEIGHT * 4];
    static float dImage[WIDTH * HEIGHT * 4], dDepth[WIDTH * HEIGHT * 2];
    static DepthGrads plain, color, host, view;
    SabercatRendererDesc d2, noneDesc;
    SabercatRenderer * r = NULL;
    SabercatRenderer * none = NULL;
    SabercatGaussians g;
    SabercatBufferView imageV, depthV, badV, zeroV, fake, noneImage;
    size_t pixels = (size_t)WIDTH * HEIGHT;
    size_t i;
    int c;
    double e;

    for (i = 0; i < COUNT; ++i)
    {
        means[i * 4 + 0] = 0.5f * (float)i - 0.5f;
        means[i * 4 + 1] = 0.1f * (float)i;
        means[i * 4 + 2] = 2.0f;
        means[i * 4 + 3] = 0.0f;
        for (c = 0; c < 4; ++c)
        {
            scales[i * 4 + c] = (c < 3) ? 0.08f : 0.0f;
            rotations[i * 4 + c] = (c == 3) ? 1.0f : 0.0f;
        }
        opacities[i] = 0.8f;
        for (c = 0; c < COEFFS * 4; ++c)
        {
            sh[i * COEFFS * 4 + c] = 0.0f;
        }
    }
    g = gaussiansDesc(means, scales, rotations, opacities, sh);

    d2 = *desc;
    d2.depth_layout = SABERCAT_DEPTH_LAYOUT_DEPTH_COVERAGE;
    r = makeRenderer(&d2, WIDTH, HEIGHT);
    CHECK(sabercat_renderer_set_gaussians(r, &g) == SABERCAT_OK, "depth gradient: set_gaussians");
    CHECK(sabercat_renderer_set_camera(r, camera) == SABERCAT_OK, "depth gradient: set_camera");

    CHECK(sabercat_renderer_backward_depth(NULL, image, pixels * 4, NULL, 0)
              == SABERCAT_ERROR_INVALID_ARGUMENT,
          "depth gradient: backward_depth(NULL renderer) must be INVALID_ARGUMENT");
    CHECK(sabercat_renderer_backward_depth(r, image, pixels * 4, NULL, 0)
              == SABERCAT_ERROR_INVALID_STATE,
          "depth gradient: backward_depth before a forward must be INVALID_STATE");

    CHECK(sabercat_renderer_forward(r) == SABERCAT_OK, "depth gradient: forward");
    CHECK(sabercat_renderer_read_image(r, image, pixels * 4) == SABERCAT_OK
              && sabercat_renderer_read_depth(r, depth, pixels * 2) == SABERCAT_OK,
          "depth gradient: read_image and read_depth");
    memcpy(dImage, image, sizeof(dImage));
    memcpy(dDepth, depth, sizeof(dDepth));

    CHECK(sabercat_renderer_backward_depth(r, dImage, pixels * 4, NULL, 1)
              == SABERCAT_ERROR_INVALID_ARGUMENT,
          "depth gradient: NULL dl_ddepth with depth_float_count != 0 must be INVALID_ARGUMENT");
    CHECK(sabercat_renderer_backward_depth(r, dImage, pixels * 4, dDepth, pixels * 4)
              == SABERCAT_ERROR_INVALID_ARGUMENT,
          "depth gradient: depth_float_count != width*height*2 must be INVALID_ARGUMENT");
    CHECK(sabercat_renderer_backward_depth(r, dImage, pixels * 4 - 1, dDepth, pixels * 2)
              == SABERCAT_ERROR_INVALID_ARGUMENT,
          "depth gradient: image_float_count != width*height*4 must be INVALID_ARGUMENT");
    CHECK(sabercat_renderer_backward_depth(r, NULL, pixels * 4, dDepth, pixels * 2)
              == SABERCAT_ERROR_INVALID_ARGUMENT,
          "depth gradient: NULL dl_dimage_rgba must be INVALID_ARGUMENT");

    /* A NULL depth gradient is the color-only backward. */
    CHECK(sabercat_renderer_backward(r, dImage, pixels * 4) == SABERCAT_OK
              && readDepthGrads(r, &plain),
          "depth gradient: color-only backward");
    CHECK(sabercat_renderer_backward_depth(r, dImage, pixels * 4, NULL, 0) == SABERCAT_OK
              && readDepthGrads(r, &color),
          "depth gradient: backward_depth with a NULL depth gradient");
    e = depthGradsError(&plain, &color);
    CHECK(e <= 1e-5, "depth gradient: NULL depth == backward (within 1e-5)");
    printf("depth gradient: NULL depth vs backward worst rel %.3g\n", e);

    /* Host and view paths agree, and the depth term changes the gradients. */
    CHECK(sabercat_renderer_backward_depth(r, dImage, pixels * 4, dDepth, pixels * 2) == SABERCAT_OK
              && readDepthGrads(r, &host),
          "depth gradient: host backward_depth");
    CHECK(sabercat_renderer_get_image_view(r, &imageV) == SABERCAT_OK
              && sabercat_renderer_get_depth_view(r, &depthV) == SABERCAT_OK,
          "depth gradient: get_image_view and get_depth_view");
    CHECK(sabercat_renderer_backward_depth_view(r, &imageV, &depthV) == SABERCAT_OK
              && readDepthGrads(r, &view),
          "depth gradient: backward_depth_view");
    e = depthGradsError(&host, &view);
    CHECK(e <= 1e-5, "depth gradient: backward_depth_view disagrees with backward_depth");
    printf("depth gradient: host vs view worst rel %.3g, depth vs color-only worst rel %.3g\n", e,
           depthGradsError(&host, &plain));
    CHECK(depthGradsError(&host, &plain) > 1e-2,
          "depth gradient: the depth term does not change the gradients");

    /* Bad views. */
    badV = depthV;
    badV.stride = 4;
    CHECK(sabercat_renderer_backward_depth_view(r, &imageV, &badV)
              == SABERCAT_ERROR_INVALID_ARGUMENT,
          "depth gradient: depth stride 4 on layout 2 must be INVALID_ARGUMENT");
    badV = depthV;
    badV.size -= 8;
    CHECK(sabercat_renderer_backward_depth_view(r, &imageV, &badV)
              == SABERCAT_ERROR_INVALID_ARGUMENT,
          "depth gradient: a depth view one pixel short must be INVALID_ARGUMENT");
    CHECK(sabercat_renderer_backward_depth_view(r, NULL, &depthV) == SABERCAT_ERROR_INVALID_ARGUMENT,
          "depth gradient: NULL image view must be INVALID_ARGUMENT");

    /* NULL and buffer-0 depth views are color-only. */
    CHECK(sabercat_renderer_backward_depth_view(r, &imageV, NULL) == SABERCAT_OK
              && readDepthGrads(r, &view)
              && depthGradsError(&view, &plain) <= 1e-5,
          "depth gradient: NULL depth view is the color-only backward");
    memset(&zeroV, 0, sizeof(zeroV));
    CHECK(sabercat_renderer_backward_depth_view(r, &imageV, &zeroV) == SABERCAT_OK
              && readDepthGrads(r, &view)
              && depthGradsError(&view, &plain) <= 1e-5,
          "depth gradient: depth view with buffer 0 is the color-only backward");

    /* Layout NONE: a depth gradient is refused, a NULL one is not. */
    noneDesc = *desc;
    noneDesc.depth_layout = SABERCAT_DEPTH_LAYOUT_NONE;
    none = makeRenderer(&noneDesc, WIDTH, HEIGHT);
    CHECK(sabercat_renderer_set_gaussians(none, &g) == SABERCAT_OK
              && sabercat_renderer_set_camera(none, camera) == SABERCAT_OK,
          "depth gradient: NONE renderer scene and camera");
    CHECK(sabercat_renderer_backward_depth(none, dImage, pixels * 4, NULL, 0)
              == SABERCAT_ERROR_INVALID_STATE,
          "depth gradient: backward_depth before a NONE forward must be INVALID_STATE");
    CHECK(sabercat_renderer_forward(none) == SABERCAT_OK
              && sabercat_renderer_get_image_view(none, &noneImage) == SABERCAT_OK,
          "depth gradient: NONE forward and image view");
    CHECK(sabercat_renderer_backward_depth(none, dImage, pixels * 4, dDepth, pixels * 2)
              == SABERCAT_ERROR_INVALID_STATE,
          "depth gradient: NONE renderer with a depth gradient must be INVALID_STATE");
    CHECK(sabercat_renderer_backward_depth(none, dImage, pixels * 4, NULL, 0) == SABERCAT_OK,
          "depth gradient: NONE renderer with a NULL depth gradient is OK");
    memset(&fake, 0, sizeof(fake));
    fake.buffer = 1; /* validation precedes any use of the handle */
    fake.stride = 8;
    fake.size = (uint64_t)pixels * 8;
    CHECK(sabercat_renderer_backward_depth_view(none, &noneImage, &fake)
              == SABERCAT_ERROR_INVALID_STATE,
          "depth gradient: NONE renderer with a depth view must be INVALID_STATE");
    CHECK(sabercat_renderer_backward_depth_view(none, &noneImage, NULL) == SABERCAT_OK,
          "depth gradient: NONE renderer with a NULL depth view is OK");

    printf("depth gradient through the ABI OK: %u pixels, depth layout 2\n", (unsigned)pixels);
    sabercat_renderer_destroy(none);
    sabercat_renderer_destroy(r);
}

int main(void)
{
    SabercatRenderer * r = NULL;
    SabercatRendererDesc desc;
    SabercatCamera camera;
    SabercatGaussians in, out;
    SabercatGradients grads;
    SabercatGaussianViews pviews;
    SabercatGradientViews gviews;
    static float outMeans[COUNT * 4], outScales[COUNT * 4], outRotations[COUNT * 4],
        outOpacities[COUNT], outSh[COUNT * COEFFS * 4];
    static float gMeans[COUNT * 4], gScales[COUNT * 4], gRotations[COUNT * 4], gOpacities[COUNT],
        gSh[COUNT * COEFFS * 4], gScreen[COUNT * 4];
    static float image[WIDTH * HEIGHT * 4], dImage[WIDTH * HEIGHT * 4];
    int i;

    printf("Sabercat C API version %u\n", sabercat_api_version());
    CHECK(sabercat_api_version() == SABERCAT_C_API_VERSION, "header/library version mismatch");

    /* -- Error paths before a renderer exists. -- */
    CHECK(sabercat_renderer_create(NULL, &r) == SABERCAT_ERROR_INVALID_ARGUMENT,
          "create(NULL) must be INVALID_ARGUMENT");
    CHECK(strlen(sabercat_last_error()) > 0, "last_error empty after a failure");
    memset(&desc, 0, sizeof(desc));
    /* Too short: below the 28 B the original header's fields need (the struct
     * has grown since, so sizeof - 4 would now be a valid older-header size). */
    desc.struct_size = (uint32_t)offsetof(SabercatRendererDesc, reserved0) - 4;
    desc.width = WIDTH;
    desc.height = HEIGHT;
    /* $SABERCAT_SHADER_DIR lets the packaged self-test run this exe away
       from the build tree (see examples/common/shader_dir.h). */
    const char * envShaderDir = getenv("SABERCAT_SHADER_DIR");
    desc.shader_dir = envShaderDir != NULL && envShaderDir[0] != '\0' ? envShaderDir : SHADER_DIR;
    CHECK(sabercat_renderer_create(&desc, &r) == SABERCAT_ERROR_INVALID_ARGUMENT,
          "short struct_size must be rejected");
    sabercat_renderer_destroy(NULL); /* must be a no-op */

    desc.struct_size = sizeof(desc);
    desc.enable_validation = 1;
    if (sabercat_renderer_create(&desc, &r) != SABERCAT_OK)
    {
        fprintf(stderr, "create failed: %s\n", sabercat_last_error());
        return 1;
    }

    CHECK(sabercat_renderer_forward(r) == SABERCAT_ERROR_INVALID_STATE,
          "forward with no scene must be INVALID_STATE");
    CHECK(sabercat_renderer_backward(r, dImage, WIDTH * HEIGHT * 4)
              == SABERCAT_ERROR_INVALID_STATE,
          "backward before forward must be INVALID_STATE");
    {
        SabercatBufferView fake;
        fake.buffer = 1; /* never dereferenced: the state check comes first */
        fake.offset = 0;
        fake.size = (uint64_t)WIDTH * HEIGHT * 16;
        fake.stride = 16;
        CHECK(sabercat_renderer_backward_view(r, &fake) == SABERCAT_ERROR_INVALID_STATE,
              "backward_view before forward must be INVALID_STATE");
    }
    {
        SabercatBufferView imageView;
        CHECK(sabercat_renderer_get_image_view(r, NULL) == SABERCAT_ERROR_INVALID_ARGUMENT,
              "get_image_view(NULL) must be INVALID_ARGUMENT");
        CHECK(sabercat_renderer_get_image_view(r, &imageView) == SABERCAT_ERROR_INVALID_STATE,
              "get_image_view before forward must be INVALID_STATE");
    }

    /* -- Scene: three Gaussians at depth ~3, distinct values in every lane. -- */
    for (i = 0; i < COUNT; ++i)
    {
        int k;
        g_means[i * 4 + 0] = -0.4f + 0.4f * (float)i;
        g_means[i * 4 + 1] = 0.1f * (float)(i - 1);
        g_means[i * 4 + 2] = -0.2f * (float)i;
        g_means[i * 4 + 3] = 0.0f;
        g_scales[i * 4 + 0] = 0.15f + 0.02f * (float)i;
        g_scales[i * 4 + 1] = 0.12f;
        g_scales[i * 4 + 2] = 0.1f;
        g_scales[i * 4 + 3] = 0.0f;
        g_rotations[i * 4 + 0] = 0.1f;
        g_rotations[i * 4 + 1] = 0.2f * (float)i;
        g_rotations[i * 4 + 2] = -0.1f;
        g_rotations[i * 4 + 3] = 0.95f;
        g_opacities[i] = 0.5f + 0.1f * (float)i;
        for (k = 0; k < COEFFS * 4; ++k)
        {
            g_sh[i * COEFFS * 4 + k] = (k % 4 == 3) ? 0.0f : 0.05f * (float)((k * 7 + i) % 9 - 4);
        }
    }
    in = gaussiansDesc(g_means, g_scales, g_rotations, g_opacities, g_sh);
    CHECK(sabercat_renderer_set_gaussians(r, &in) == SABERCAT_OK, "set_gaussians failed");

    out = gaussiansDesc(outMeans, outScales, outRotations, outOpacities, outSh);
    CHECK(sabercat_renderer_get_gaussians(r, &out) == SABERCAT_OK, "get_gaussians failed");
    CHECK(memcmp(outMeans, g_means, sizeof(g_means)) == 0
              && memcmp(outScales, g_scales, sizeof(g_scales)) == 0
              && memcmp(outRotations, g_rotations, sizeof(g_rotations)) == 0
              && memcmp(outOpacities, g_opacities, sizeof(g_opacities)) == 0
              && memcmp(outSh, g_sh, sizeof(g_sh)) == 0,
          "set_gaussians -> get_gaussians is not bit-exact");

    CHECK(sabercat_renderer_forward(r) == SABERCAT_ERROR_INVALID_STATE,
          "forward with no camera must be INVALID_STATE");

    /* Camera 3 units back along +Z, looking down -Z (column-major identity
     * with translation in elements 12-14). */
    memset(&camera, 0, sizeof(camera));
    camera.struct_size = sizeof(camera);
    camera.camera_to_world[0] = camera.camera_to_world[5] = camera.camera_to_world[10] =
        camera.camera_to_world[15] = 1.0f;
    camera.camera_to_world[14] = 3.0f;
    camera.fx = 50.0f;
    camera.fy = 50.0f;
    camera.cx = WIDTH / 2.0f;
    camera.cy = HEIGHT / 2.0f;
    CHECK(sabercat_renderer_set_camera(r, &camera) == SABERCAT_OK, "set_camera failed");

    /* -- Forward + image. -- */
    {
        double base = renderRedSum(r, image);
        double coverage = 0.0;
        for (i = 0; i < WIDTH * HEIGHT; ++i)
        {
            coverage += image[i * 4 + 3];
        }
        CHECK(!isnan(base) && coverage > 1.0, "forward rendered nothing");
        printf("forward OK: red sum %.4f, coverage sum %.2f\n", base, coverage);
    }

    /* -- Backward with dL/dImage = 1 on red, 0 elsewhere. -- */
    for (i = 0; i < WIDTH * HEIGHT * 4; ++i)
    {
        dImage[i] = (i % 4 == 0) ? 1.0f : 0.0f;
    }
    CHECK(sabercat_renderer_backward(r, dImage, 7) == SABERCAT_ERROR_INVALID_ARGUMENT,
          "backward with a wrong float_count must be INVALID_ARGUMENT");
    CHECK(sabercat_renderer_backward(r, dImage, WIDTH * HEIGHT * 4) == SABERCAT_OK,
          "backward failed");

    memset(&grads, 0, sizeof(grads));
    grads.struct_size = sizeof(grads);
    grads.count = COUNT;
    grads.means = gMeans;
    grads.scales = gScales;
    grads.rotations = gRotations;
    grads.opacities = gOpacities;
    grads.sh_coefficients = gSh;
    grads.screen_means = gScreen;
    CHECK(sabercat_renderer_read_gradients(r, &grads) == SABERCAT_OK, "read_gradients failed");

    /* -- Zero-copy views. -- */
    CHECK(sabercat_renderer_get_gaussian_views(r, &pviews) == SABERCAT_OK
              && pviews.means.buffer != 0 && pviews.means.stride == 16
              && pviews.opacities.stride == 4 && pviews.sh_coefficients.stride == 16
              && pviews.count == COUNT,
          "gaussian views wrong");
    CHECK(sabercat_renderer_get_gradient_views(r, &gviews) == SABERCAT_OK
              && gviews.means.buffer != 0 && gviews.screen_means.buffer != 0
              && gviews.opacities.stride == 4 && gviews.count == COUNT,
          "gradient views wrong");
    {
        SabercatBufferView imageView;
        CHECK(sabercat_renderer_get_image_view(r, &imageView) == SABERCAT_OK
                  && imageView.buffer != 0 && imageView.offset == 0 && imageView.stride == 16
                  && imageView.size == (uint64_t)WIDTH * HEIGHT * 16,
              "image view wrong");
    }

    /* -- Finite difference of Gaussian 0's opacity, through the ABI. -- */
    {
        const float h = 0.01f;
        const float original = g_opacities[0];
        double plus, minus, fd, analytic, rel;
        g_opacities[0] = original + h;
        CHECK(sabercat_renderer_set_gaussians(r, &in) == SABERCAT_OK, "set_gaussians (+h) failed");
        CHECK(sabercat_renderer_backward(r, dImage, WIDTH * HEIGHT * 4)
                  == SABERCAT_ERROR_INVALID_STATE,
              "backward after replacing the scene must be INVALID_STATE");
        plus = renderRedSum(r, image);
        g_opacities[0] = original - h;
        CHECK(sabercat_renderer_set_gaussians(r, &in) == SABERCAT_OK, "set_gaussians (-h) failed");
        minus = renderRedSum(r, image);
        g_opacities[0] = original;

        fd = (plus - minus) / (2.0 * (double)h);
        analytic = gOpacities[0];
        rel = fabs(fd - analytic) / fabs(fd);
        printf("opacity gradient through the ABI: analytic %.5f, finite difference %.5f "
               "(rel diff %.2e)\n",
               analytic, fd, rel);
        CHECK(fabs(fd) > 1.0, "finite difference is degenerate (Gaussian 0 not visible?)");
        CHECK(rel < 5e-2, "ABI backward disagrees with an ABI finite difference");
    }

    /* -- Zero-copy backward. dL/dImage = the image (L = 0.5 |I|^2):
     * host path from read_image, device path straight from the image view. -- */
    {
        static float hMeans[COUNT * 4], hScales[COUNT * 4], hRotations[COUNT * 4],
            hOpacities[COUNT], hSh[COUNT * COEFFS * 4], hScreen[COUNT * 4];
        SabercatBufferView view, bad;
        double err = 0.0, e;
        CHECK(sabercat_renderer_set_gaussians(r, &in) == SABERCAT_OK, "set_gaussians (restore)");
        CHECK(sabercat_renderer_forward(r) == SABERCAT_OK
                  && sabercat_renderer_read_image(r, image, WIDTH * HEIGHT * 4) == SABERCAT_OK
                  && sabercat_renderer_get_image_view(r, &view) == SABERCAT_OK,
              "forward + read_image + get_image_view");

        bad = view;
        bad.stride = 4;
        CHECK(sabercat_renderer_backward_view(r, &bad) == SABERCAT_ERROR_INVALID_ARGUMENT,
              "backward_view with stride 4 must be INVALID_ARGUMENT");
        bad = view;
        bad.size -= 16;
        CHECK(sabercat_renderer_backward_view(r, &bad) == SABERCAT_ERROR_INVALID_ARGUMENT,
              "backward_view with a short buffer must be INVALID_ARGUMENT");
        CHECK(sabercat_renderer_backward_view(r, NULL) == SABERCAT_ERROR_INVALID_ARGUMENT,
              "backward_view(NULL) must be INVALID_ARGUMENT");

        CHECK(sabercat_renderer_backward(r, image, WIDTH * HEIGHT * 4) == SABERCAT_OK
                  && sabercat_renderer_read_gradients(r, &grads) == SABERCAT_OK,
              "host backward");
        memcpy(hMeans, gMeans, sizeof(hMeans));
        memcpy(hScales, gScales, sizeof(hScales));
        memcpy(hRotations, gRotations, sizeof(hRotations));
        memcpy(hOpacities, gOpacities, sizeof(hOpacities));
        memcpy(hSh, gSh, sizeof(hSh));
        memcpy(hScreen, gScreen, sizeof(hScreen));

        CHECK(sabercat_renderer_backward_view(r, &view) == SABERCAT_OK
                  && sabercat_renderer_read_gradients(r, &grads) == SABERCAT_OK,
              "backward_view");
        e = relativeError(gMeans, hMeans, COUNT * 4);
        err = e > err ? e : err;
        e = relativeError(gScales, hScales, COUNT * 4);
        err = e > err ? e : err;
        e = relativeError(gRotations, hRotations, COUNT * 4);
        err = e > err ? e : err;
        e = relativeError(gOpacities, hOpacities, COUNT);
        err = e > err ? e : err;
        e = relativeError(gSh, hSh, COUNT * COEFFS * 4);
        err = e > err ? e : err;
        e = relativeError(gScreen, hScreen, COUNT * 4);
        err = e > err ? e : err;
        printf("backward_view vs host backward: worst rel. %.3g\n", err);
        CHECK(hOpacities[0] != 0.0f, "host-path gradients are degenerate");
        CHECK(err <= 1e-5, "backward_view disagrees with host backward");
    }

    /* -- Device sharing. -- */
    {
        SabercatDeviceInfo info;
        memset(&info, 0, sizeof(info));
        info.struct_size = (uint32_t)offsetof(SabercatDeviceInfo, compute_queue); /* short */
        CHECK(sabercat_renderer_get_device_info(r, &info) == SABERCAT_ERROR_INVALID_ARGUMENT,
              "get_device_info with a short struct must be INVALID_ARGUMENT");

        memset(&info, 0, sizeof(info));
        info.struct_size = sizeof(info);
        CHECK(sabercat_renderer_get_device_info(r, &info) == SABERCAT_OK
                  && info.instance != NULL && info.physical_device != NULL && info.device != NULL
                  && info.compute_queue != NULL && info.compute_queue_family != 0xFFFFFFFFu
                  && info.vulkan_api_version >= ((1u << 22) | (3u << 12))
                  && info.float32_atomic_add == 1 && info.struct_size == sizeof(info),
              "get_device_info returned incomplete handles");

        /* An older caller's smaller struct is honored and never overrun. */
        {
            SabercatDeviceInfo older;
            memset(&older, 0, sizeof(older));
            older.struct_size = (uint32_t)offsetof(SabercatDeviceInfo, vulkan_api_version);
            older.vulkan_api_version = 0xABCDu;
            older.float32_atomic_add = 77;
            CHECK(sabercat_renderer_get_device_info(r, &older) == SABERCAT_OK
                      && older.device == info.device && older.vulkan_api_version == 0xABCDu
                      && older.float32_atomic_add == 77,
                  "get_device_info wrote past the caller's struct_size");
        }
    }

    /* -- Resizable storage. -- */
    {
        SabercatGaussianStorage st0, st;
        SabercatGaussianViews before, after;
        SabercatGaussians v1;
        int32_t grew = -1;
        double sum2Fresh, sum2Pruned, sum3;

        /* The finite difference above left opacity0 - h on the GPU. */
        CHECK(sabercat_renderer_set_gaussians(r, &in) == SABERCAT_OK, "set_gaussians (reset)");
        memset(&st0, 0, sizeof(st0));
        st0.struct_size = sizeof(st0);
        CHECK(sabercat_renderer_get_gaussian_storage(r, &st0) == SABERCAT_OK && st0.count == COUNT
                  && st0.capacity == COUNT,
              "storage after set_gaussians: count == capacity == 3");

        CHECK(sabercat_renderer_set_gaussian_count(r, 0) == SABERCAT_ERROR_INVALID_ARGUMENT,
              "set_gaussian_count(0) must be INVALID_ARGUMENT");
        CHECK(sabercat_renderer_set_gaussian_count(r, COUNT + 1) == SABERCAT_ERROR_INVALID_ARGUMENT,
              "set_gaussian_count beyond capacity must be INVALID_ARGUMENT");

        sabercat_renderer_get_gaussian_views(r, &before);
        CHECK(sabercat_renderer_reserve_gaussians(r, 2, &grew) == SABERCAT_OK && grew == 0,
              "non-growing reserve must report grew = 0");
        CHECK(sabercat_renderer_reserve_gaussians(r, 8, &grew) == SABERCAT_OK && grew == 1,
              "reserve(8) must grow");
        memset(&st, 0, sizeof(st));
        st.struct_size = sizeof(st);
        sabercat_renderer_get_gaussian_storage(r, &st);
        sabercat_renderer_get_gaussian_views(r, &after);
        CHECK(st.count == COUNT && st.capacity == 8 && st.generation > st0.generation,
              "reserve: capacity 8, count kept, generation advanced");
        CHECK(after.means.buffer != before.means.buffer && after.means.size == 8 * 16
                  && after.opacities.size == 8 * 4 && after.count == COUNT,
              "reserve: new handles spanning the capacity");
        CHECK(sabercat_renderer_backward(r, dImage, WIDTH * HEIGHT * 4)
                  == SABERCAT_ERROR_INVALID_STATE,
              "backward after reserve must be INVALID_STATE");
        out = gaussiansDesc(outMeans, outScales, outRotations, outOpacities, outSh);
        CHECK(sabercat_renderer_get_gaussians(r, &out) == SABERCAT_OK
                  && memcmp(outMeans, g_means, sizeof(g_means)) == 0
                  && memcmp(outScales, g_scales, sizeof(g_scales)) == 0
                  && memcmp(outRotations, g_rotations, sizeof(g_rotations)) == 0
                  && memcmp(outOpacities, g_opacities, sizeof(g_opacities)) == 0
                  && memcmp(outSh, g_sh, sizeof(g_sh)) == 0,
              "reserve did not preserve the parameters bit-exactly");
        sum3 = renderRedSum(r, image);

        /* Prune to 2 through the count == a fresh 2-Gaussian scene. */
        CHECK(sabercat_renderer_set_gaussian_count(r, 2) == SABERCAT_OK,
              "set_gaussian_count(2) failed");
        sabercat_renderer_get_gaussian_storage(r, &st);
        CHECK(st.count == 2 && st.capacity == 8, "set_gaussian_count: count 2, capacity kept");
        sum2Pruned = renderRedSum(r, image);
        CHECK(sabercat_renderer_set_gaussian_count(r, COUNT) == SABERCAT_OK
                  && renderRedSum(r, image) == sum3,
              "regrowing the count over the kept tail restores the render exactly");

        /* A SabercatGaussians that ends before the capacity field is accepted. */
        v1 = gaussiansDesc(g_means, g_scales, g_rotations, g_opacities, g_sh);
        v1.struct_size = (uint32_t)offsetof(SabercatGaussians, capacity);
        v1.count = 2;
        v1.capacity = 99; /* beyond the caller's struct: must be ignored */
        CHECK(sabercat_renderer_set_gaussians(r, &v1) == SABERCAT_OK,
              "set_gaussians with a struct that ends before capacity failed");
        sabercat_renderer_get_gaussian_storage(r, &st);
        CHECK(st.count == 2 && st.capacity == 2, "struct without capacity: capacity == count");
        sum2Fresh = renderRedSum(r, image);
        CHECK(sum2Fresh == sum2Pruned, "pruned-by-count render differs from a fresh scene");
        printf("storage OK: red sum 3 Gaussians %.4f, 2 pruned %.4f, 2 fresh %.4f\n", sum3,
               sum2Pruned, sum2Fresh);

        v1.struct_size = sizeof(v1);
        v1.capacity = 6;
        CHECK(sabercat_renderer_set_gaussians(r, &v1) == SABERCAT_OK, "set_gaussians(capacity 6)");
        sabercat_renderer_get_gaussian_storage(r, &st);
        CHECK(st.count == 2 && st.capacity == 6, "set_gaussians honors capacity");
    }

    /* -- Borrowed (application-owned) device through the ABI. A second renderer borrows
     * r's device and r's Gaussian buffers, at a different image size. -- */
    {
#define W2 40
#define H2 24
        static float small1[W2 * H2 * 4], small2[W2 * H2 * 4];
        SabercatDeviceInfo info, bad;
        SabercatRendererDesc desc2;
        SabercatRenderer * r2 = NULL;
        SabercatGaussianViews views;
        SabercatBorrowedGaussians scene;
        SabercatCamera camera2;
        uint32_t count;

        memset(&info, 0, sizeof(info));
        info.struct_size = sizeof(info);
        CHECK(sabercat_renderer_get_device_info(r, &info) == SABERCAT_OK, "get_device_info");
        CHECK(info.vulkan_api_version >= ((1u << 22) | (3u << 12)), "device info reports Vulkan >= 1.3");
        desc2 = desc;
        desc2.width = W2;
        desc2.height = H2;

        bad = info;
        bad.vulkan_api_version = (1u << 22) | (2u << 12); /* 1.2 */
        CHECK(sabercat_renderer_create_borrowed(&desc2, &bad, &r2) == SABERCAT_ERROR_INVALID_ARGUMENT
                  && r2 == NULL && strstr(sabercat_last_error(), "apiVersion") != NULL,
              "create_borrowed with a declared Vulkan 1.2 must be INVALID_ARGUMENT");
        bad = info;
        bad.device = NULL;
        CHECK(sabercat_renderer_create_borrowed(&desc2, &bad, &r2) == SABERCAT_ERROR_INVALID_ARGUMENT,
              "create_borrowed with a null device must be INVALID_ARGUMENT");
        CHECK(sabercat_renderer_create_borrowed(&desc2, &info, &r2) == SABERCAT_OK && r2 != NULL,
              "create_borrowed failed");

        camera2 = camera;
        camera2.fx = camera.fx * W2 / WIDTH;
        camera2.fy = camera.fy * W2 / WIDTH;
        camera2.cx = W2 / 2.0f;
        camera2.cy = H2 / 2.0f;
        CHECK(sabercat_renderer_set_camera(r2, &camera2) == SABERCAT_OK, "set_camera (r2)");

        CHECK(sabercat_renderer_get_gaussian_views(r, &views) == SABERCAT_OK, "get_gaussian_views (r)");
        sabercat_renderer_get_gaussian_count(r, &count);
        memset(&scene, 0, sizeof(scene));
        scene.struct_size = sizeof(scene);
        scene.count = count;
        scene.sh_degree = 3;
        scene.parameters = views; /* gradients all zero: r2 allocates its own */

        scene.parameters.scales.stride = 12;
        CHECK(sabercat_renderer_set_gaussian_views(r2, &scene) == SABERCAT_ERROR_INVALID_ARGUMENT
                  && strstr(sabercat_last_error(), "parameters.scales") != NULL,
              "set_gaussian_views with a bad stride must be INVALID_ARGUMENT naming the stream");
        scene.parameters = views;
        scene.count = views.count + 100;
        CHECK(sabercat_renderer_set_gaussian_views(r2, &scene) == SABERCAT_ERROR_INVALID_ARGUMENT,
              "set_gaussian_views with count above capacity must be INVALID_ARGUMENT");
        scene.count = count;
        CHECK(sabercat_renderer_set_gaussian_views(r2, &scene) == SABERCAT_OK, "set_gaussian_views failed");

        CHECK(sabercat_renderer_forward(r2) == SABERCAT_OK
                  && sabercat_renderer_read_image(r2, small2, W2 * H2 * 4) == SABERCAT_OK,
              "forward/read_image on the borrowing renderer");
        CHECK(sabercat_renderer_reserve_gaussians(r2, 64, NULL) != SABERCAT_OK,
              "reserve_gaussians on borrowed storage must fail");

        /* r at the same size must render the same pixels. */
        CHECK(sabercat_renderer_set_image_size(r, 0, H2) == SABERCAT_ERROR_INVALID_ARGUMENT,
              "set_image_size(0, h) must be INVALID_ARGUMENT");
        CHECK(sabercat_renderer_set_image_size(r, W2, H2) == SABERCAT_OK, "set_image_size failed");
        CHECK(sabercat_renderer_read_image(r, small1, W2 * H2 * 4) == SABERCAT_ERROR_INVALID_STATE,
              "read_image after set_image_size, before a forward, must be INVALID_STATE");
        CHECK(sabercat_renderer_set_camera(r, &camera2) == SABERCAT_OK
                  && sabercat_renderer_forward(r) == SABERCAT_OK
                  && sabercat_renderer_read_image(r, small1, W2 * H2 * 4) == SABERCAT_OK,
              "forward/read_image on r at the new size");
        CHECK(memcmp(small1, small2, sizeof(small1)) == 0,
              "borrowing renderer differs from the owner at the same size");
        printf("model B through the ABI OK: %ux%u, %u Gaussians borrowed\n", W2, H2, count);

        sabercat_renderer_destroy(r2); /* before r, whose device and buffers it borrowed */
#undef W2
#undef H2
    }

    /* -- Relighting and shadows. -- */
    testRelighting(r, &desc, &camera);

    /* -- Argument matrix, struct sizes, scenes, settings, outputs, gradients. -- */
    {
        SabercatDeviceInfo devInfo;
        memset(&devInfo, 0, sizeof(devInfo));
        devInfo.struct_size = sizeof(devInfo);
        CHECK(sabercat_renderer_get_device_info(r, &devInfo) == SABERCAT_OK,
              "device info for the extended checks");
        testErrorMatrix(&desc, &camera);
        testStructSizes(&desc, &camera);
        testLoadScene(&desc, &camera);
        testSettingsRoundTrip(&desc, devInfo.ray_query);
        testOutputsAndLights(&desc, &camera);
        testAntialiasAndSh(&desc, &camera);
        testFiniteDifferences(&desc, &camera);
        testMeansScalesFD(&desc, &camera);
        testGradientsAndViews(&desc, &camera);
        testLargeScene(&desc, &camera);
        testClearScene(&desc, &camera);
        testDepthOutput(&desc, &camera);
        testDepthGradient(&desc, &camera);
    }

    sabercat_renderer_destroy(r);

    if (g_failures == 0)
    {
        printf("c_api_test PASSED\n");
        return 0;
    }
    fprintf(stderr, "c_api_test FAILED (%d checks)\n", g_failures);
    return 1;
}
