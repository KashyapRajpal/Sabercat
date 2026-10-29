# C API guide

`sabercat_c` is the C ABI of Sabercat. It is a shared library, `sabercat_c.dll`
on Windows, with the header `sabercat/c_api.h`. Use it from C, from any language
that can call C (Python ctypes or cffi, for example), or from C++ code that
prefers not to link the C++ library.

The C++ API is in [library.md](library.md). Both APIs drive the same renderer.
The C ABI covers the renderer's main calls, but not all of them. See
[Not in the C ABI](#not-in-the-c-abi).

## Contents

- [Linking](#linking)
- [Lifecycle](#lifecycle)
- [Error handling](#error-handling)
- [Struct versioning](#struct-versioning)
- [Data layout](#data-layout)
- [Threading](#threading)
- [Functions](#functions)
- [Not in the C ABI](#not-in-the-c-abi)

## Linking

| Item | Value |
|---|---|
| Header | `sabercat/c_api.h`. Needs only `<stdint.h>` and `<stddef.h>`. |
| CMake target | `Sabercat::sabercat_c` (shared) |
| Runtime library | `sabercat_c.dll`, beside your executable. `sabercat_copy_runtime_dlls(<target>)` copies it. |
| API version | `SABERCAT_C_API_VERSION` (1), and `sabercat_api_version()` at run time |

On Windows the functions are `__declspec(dllimport)` unless you define
`SABERCAT_C_BUILD` while building the DLL itself. A non-Windows build uses default
visibility, but Sabercat targets Windows.

## Lifecycle

A renderer is created, used, and destroyed by the caller. Each `sabercat_renderer_*`
function that takes a renderer handle needs one from `sabercat_renderer_create`
or `sabercat_renderer_create_borrowed`.

```c
#include <stdio.h>
#include "sabercat/c_api.h"

int main(void)
{
    SabercatRendererDesc desc = {0};
    desc.struct_size = sizeof(desc);
    desc.width = 512;
    desc.height = 512;
    desc.shader_dir = "path/to/shaders";     /* directory with *.comp.spv */
    desc.enable_validation = 0;

    SabercatRenderer * r = NULL;
    if (sabercat_renderer_create(&desc, &r) != SABERCAT_OK)
    {
        fprintf(stderr, "create failed: %s\n", sabercat_last_error());
        return 1;
    }

    if (sabercat_renderer_load_scene(r, "scene.ply") != SABERCAT_OK)
    {
        fprintf(stderr, "load failed: %s\n", sabercat_last_error());
    }

    /* set a camera, render, read the image ... */

    sabercat_renderer_destroy(r);            /* accepts NULL */
    return 0;
}
```

| Function | Creates or destroys |
|---|---|
| `sabercat_renderer_create(desc, &out)` | A renderer on a device Sabercat creates (ownership model A) |
| `sabercat_renderer_create_borrowed(desc, device, &out)` | A renderer on a device the caller created (model B). See below. |
| `sabercat_renderer_destroy(r)` | Releases the renderer. Accepts NULL. |

`SabercatRendererDesc` fields:

| Field | Meaning |
|---|---|
| `struct_size` | `sizeof(SabercatRendererDesc)` |
| `width`, `height` | Output image size |
| `shader_dir` | Directory with the compiled `*.comp.spv`. Required for rendering. |
| `enable_validation` | Non-zero turns on the Khronos validation layer, if installed. Ignored by `sabercat_renderer_create_borrowed`. |
| `reserved0`, `reserved1` | Padding for the ABI layout. Not read. Set them to 0 (a zero-initialized desc does). |
| `depth_layout` | A `SabercatDepthLayout` value (0, 1, 2 or 4) that fixes the depth output for the renderer's life. 0 writes no depth output. Other values return `SABERCAT_ERROR_INVALID_ARGUMENT`. See [Depth output](#depth-output). |

### Borrowed device

`sabercat_renderer_create_borrowed` runs on a device the caller created. Get the
device description from another renderer with `sabercat_renderer_get_device_info`,
or fill a `SabercatDeviceInfo` yourself.

| `SabercatDeviceInfo` field | Meaning |
|---|---|
| `instance`, `physical_device`, `device` | Vulkan handles, as `void *` |
| `compute_queue`, `compute_queue_family` | The queue all Sabercat work runs on |
| `vulkan_api_version` | The instance's API version. Must be at least 1.3. |
| `float32_atomic_add` | Whether `shaderBufferFloat32AtomicAdd` was enabled. Needed for backward. |
| `ray_query` | Whether ray query was enabled. Read only when `struct_size` covers the field. |

The device is checked first. A failure returns `SABERCAT_ERROR_INVALID_ARGUMENT` with
a message that names the problem. The renderer does not own the device. Destroy
the renderer before the owner destroys the device.

## Error handling

Every function returns `SabercatResult`, except `sabercat_api_version`,
`sabercat_last_error` and `sabercat_renderer_destroy`. No C++ exception crosses
the boundary.

| Value | Meaning |
|---|---|
| `SABERCAT_OK` (0) | Success |
| `SABERCAT_ERROR_INVALID_ARGUMENT` (1) | A null pointer, a size mismatch, or a bad value |
| `SABERCAT_ERROR_INVALID_STATE` (2) | For example, backward before a differentiable forward |
| `SABERCAT_ERROR_UNSUPPORTED` (3) | For example, the device lacks fp32 buffer atomics |
| `SABERCAT_ERROR_RUNTIME` (4) | Vulkan, I/O, or any other failure |

`sabercat_last_error()` returns a message for the last failing call on the calling
thread. The string is thread-local. It stays valid until the next failing call on
that thread. Copy it if you need it later.

## Struct versioning

Structs that start with `struct_size` let the library add fields without breaking
callers.

- Set `struct_size` to `sizeof` the struct you pass.
- An older caller with a smaller `struct_size` is honored. Fields after its
  size are not read, and a newer library does not write past it.
- Output structs are written only up to the `struct_size` the caller passed.
- Set `struct_size` on output structs too, for example `SabercatGaussianStorage`,
  `SabercatDeviceInfo` and `SabercatLightingSettings`.
- `SabercatRendererDesc` is 40 bytes on 64-bit. A caller built against the
  earlier 32-byte header passes `struct_size` 32, and its `depth_layout` is then
  read as 0 (no depth output). The create functions accept any `struct_size` of
  28 or more.

These structs carry `struct_size`: `SabercatRendererDesc`, `SabercatGaussians`,
`SabercatGradients`, `SabercatCamera`, `SabercatDeviceInfo`,
`SabercatBorrowedGaussians`, `SabercatGaussianStorage`, `SabercatLight`,
`SabercatEnvironment`, `SabercatLightingSettings` and `SabercatShadowSettings`.

The view structs (`SabercatBufferView`, `SabercatGaussianViews`,
`SabercatGradientViews`, `SabercatRelightViews`) have no `struct_size`. Their
layout is fixed.

Additive changes keep `SABERCAT_C_API_VERSION` at 1. A change that breaks
existing callers raises it.

## Data layout

The layout matches the C++ API. See [library.md](library.md#scenes).

| Data | Layout |
|---|---|
| `means`, `scales`, `rotations` | 4 floats per Gaussian (the 4th is padding). Rotations are `xyzw`. |
| `opacities` | 1 float per Gaussian, in `[0, 1]` |
| `sh_coefficients` | 16 coefficients x 4 floats (RGB + padding), band-ascending |
| Image and image gradients | `width * height` RGBA floats, row-major. Alpha is coverage. |
| Depth output and depth gradients | `width * height * n` floats, row-major, with `n` = 1, 2 or 4 from `depth_layout`. See [Depth output](#depth-output). |
| Values | Activated: linear scale, opacity in `[0, 1]`, normalized quaternion, raw SH |
| Gradients | Taken with respect to the same stored values. The caller applies its own activations. |
| Camera | `camera_to_world` is a column-major 4x4. Convention: right-handed, +Y up, looking down local -Z. |

Vulkan handles in `SabercatBufferView` and `SabercatDeviceInfo` are `uint64_t` for
non-dispatchable handles and `void *` for dispatchable ones.

## Threading

A renderer handle is not thread-safe. Call it from one thread at a time. The
`compute_queue` in `SabercatDeviceInfo` is externally synchronized. Do not submit
to it from another thread while a `sabercat_*` call runs.

Views and the image handle follow the rules in
[library.md](library.md#ownership-and-synchronization). In short: a Sabercat
call is synchronous, and views stay valid until the storage generation changes
or the renderer is destroyed.

## Functions

Most functions return `SabercatResult` (exceptions listed above).

### Version and errors

| Function | Purpose |
|---|---|
| `sabercat_api_version()` | Returns the API version (`uint32_t`) |
| `sabercat_last_error()` | Returns the last error message on this thread |

### Lifecycle

| Function | Purpose |
|---|---|
| `sabercat_renderer_create(desc, out)` | Create on a Sabercat-owned device |
| `sabercat_renderer_create_borrowed(desc, device, out)` | Create on a caller-owned device |
| `sabercat_renderer_destroy(r)` | Destroy (accepts NULL) |
| `sabercat_renderer_get_device_info(r, out)` | Fill `SabercatDeviceInfo` for sharing the device |

### Scene

| Function | Purpose |
|---|---|
| `sabercat_renderer_load_scene(r, path)` | Load any format `loadScene()` accepts |
| `sabercat_renderer_set_gaussians(r, g)` | Replace the scene with host arrays (`SabercatGaussians`). Reads `capacity` and the optional relight pointers. |
| `sabercat_renderer_get_gaussians(r, out)` | Read the current parameters into host arrays. `out->count` must equal the current count. |
| `sabercat_renderer_get_gaussian_count(r, &count)` | The count forward and backward process |
| `sabercat_renderer_clear_scene(r)` | Remove the scene (owned or borrowed) without loading another. Count 0; forward does nothing; `read_image`, `get_image_view` and `read_lit_image` are INVALID_STATE until a forward of a new scene. Borrowed buffers stay yours. Fine with no scene. |
| `sabercat_renderer_set_gaussian_views(r, scene)` | Replace the scene with caller-owned buffers (`SabercatBorrowedGaussians`) |
| `sabercat_renderer_get_gaussian_views(r, out)` | Views of the parameter streams |
| `sabercat_renderer_get_gradient_views(r, out)` | Views of the gradient streams. Allocates the gradient buffers on first call. |
| `sabercat_renderer_get_gaussian_storage(r, out)` | Count, capacity and storage generation (`SabercatGaussianStorage`) |
| `sabercat_renderer_set_gaussian_count(r, count)` | Set the processed count, within capacity. No GPU work. |
| `sabercat_renderer_reserve_gaussians(r, capacity, &grew)` | Grow storage. `*grew` (may be NULL) is 0 if nothing changed. Replaces buffer handles. |
| `sabercat_renderer_set_antialiased(r, flag)` | Override the opacity-compensation flag |

### Camera and rendering

| Function | Purpose |
|---|---|
| `sabercat_renderer_set_camera(r, cam)` | Set the `SabercatCamera` for later forward and backward calls |
| `sabercat_renderer_set_sh_degree(r, d)` | Max SH degree evaluated, 0 to 3 |
| `sabercat_renderer_forward(r)` | Differentiable forward pass |
| `sabercat_renderer_read_image(r, rgba, n)` | Copy the image to host memory. `n` must be `width * height * 4`. |
| `sabercat_renderer_get_image_view(r, out)` | The image as a device buffer view. Needs a forward of the current scene. |
| `sabercat_renderer_set_image_size(r, w, h)` | Change the output size. Invalidates the last forward. |
| `sabercat_renderer_backward(r, dl_dimage, n)` | Backward from a host dL/d(image). Overwrites the gradients. |
| `sabercat_renderer_backward_view(r, dl_dimage)` | Backward from a device buffer, same layout |
| `sabercat_renderer_read_gradients(r, out)` | Copy the gradients to host memory (`SabercatGradients`) |

Backward needs a differentiable forward of the current scene with the same camera.
It returns `SABERCAT_ERROR_INVALID_STATE` otherwise.

### Depth output

The depth output is fixed when the renderer is created, with
`SabercatRendererDesc::depth_layout`. `sabercat_renderer_forward()` writes it.
Each pixel holds `n` floats, row-major, so the host buffer is
`width * height * n` floats.

| `SabercatDepthLayout` | `n` | Components per pixel |
|---|---|---|
| `SABERCAT_DEPTH_LAYOUT_NONE` (0) | 0 | No depth output. The depth functions return `SABERCAT_ERROR_INVALID_STATE`. |
| `SABERCAT_DEPTH_LAYOUT_DEPTH` (1) | 1 | D |
| `SABERCAT_DEPTH_LAYOUT_DEPTH_COVERAGE` (2) | 2 | D, coverage |
| `SABERCAT_DEPTH_LAYOUT_DEPTH_COVERAGE_FRONT` (4) | 4 | D, coverage, front depth, 0 |

- **D** is the sum of `T * alpha * z` over the blended Gaussians, where `z` is
  camera depth. D is **not normalized**. Divide it by coverage to get the
  expected depth.
- **Coverage** is `1 - T`. It is bit-equal to the image's alpha.
- **Front depth** is the camera depth of the frontmost contributing Gaussian,
  or 0 where no Gaussian contributes.

Color is bit-identical whatever the layout.

| Function | Purpose |
|---|---|
| `sabercat_renderer_read_depth(r, values, float_count)` | Copy the depth output to host memory. `float_count` must be `width * height * n`. |
| `sabercat_renderer_get_depth_view(r, out)` | The depth output as a device buffer view: stride `n * 4`, offset 0, size `width * height * n * 4` |
| `sabercat_renderer_backward_depth(r, dl_dimage, image_float_count, dl_ddepth, depth_float_count)` | Backward from a host dL/d(image) and dL/d(depth). See below. |
| `sabercat_renderer_backward_depth_view(r, dl_dimage, dl_ddepth)` | The same, with device buffers. `dl_dimage` follows `sabercat_renderer_backward_view`. |

The read and view functions need a forward of the current scene. Without one,
or with layout NONE, they return `SABERCAT_ERROR_INVALID_STATE`. A wrong
`float_count` returns `SABERCAT_ERROR_INVALID_ARGUMENT`.

Depth gradients. `dl_ddepth` is dL/d(depth output) in the `read_depth` layout.
Only component 0 (dL/dD) is read. The other components are ignored, so the
whole depth output can be passed. Coverage's gradient flows through the alpha
of `dl_dimage`. The front-depth component is not differentiated.

- `dl_ddepth` NULL means color-only. Then `depth_float_count` must be 0, and
  the call is the same as `sabercat_renderer_backward()`.
- Otherwise `depth_float_count` must be `width * height * n`
  (`SABERCAT_ERROR_INVALID_ARGUMENT` if not). Layout NONE returns
  `SABERCAT_ERROR_INVALID_STATE`.
- For the view, a NULL pointer or a buffer of 0 means no depth gradient. The
  stride must be `4 * n`, and the size at least `width * height * n * 4`, or the
  call returns `SABERCAT_ERROR_INVALID_ARGUMENT`.
- Both calls share the usage and queue rules of `sabercat_renderer_backward_view`.
  They need a forward of the current scene, and they return
  `SABERCAT_ERROR_UNSUPPORTED` without fp32 atomic add.
- Gradients are overwritten, as in the color-only backward.

### Lighting and shadows

| Function | Purpose |
|---|---|
| `sabercat_renderer_set_lighting_enabled(r, flag)` | Turn the lit image on or off. Off by default. |
| `sabercat_renderer_set_lights(r, lights, count)` | Replace the lights. `count` is at most `SABERCAT_MAX_LIGHT_COUNT` (8). Zero clears them. |
| `sabercat_renderer_set_environment(r, env)` | Set the environment light (`SabercatEnvironment`) |
| `sabercat_renderer_set_lighting_settings(r, s)` | Set `SabercatLightingSettings` |
| `sabercat_renderer_get_lighting_settings(r, out)` | Read `SabercatLightingSettings` |
| `sabercat_renderer_set_shadow_settings(r, s)` | Set `SabercatShadowSettings`. Ray-traced mode needs `ray_query` on the device. |
| `sabercat_renderer_get_shadow_settings(r, out)` | Read `SabercatShadowSettings` |
| `sabercat_renderer_read_lit_image(r, rgba, n)` | Copy the lit image to host memory. Needs a lit forward. |
| `sabercat_renderer_get_lit_image_view(r, out)` | The lit image as a device buffer view |

Lights and environment persist across scenes. They apply from the next forward.

Color spaces. `read_image` returns display-referred sRGB rgb, premultiplied by
coverage, and alpha = coverage. The lit image is display-referred too, but only
for `SABERCAT_LIGHTING_OUTPUT_LIT`: the shading is composited in linear light and
then sRGB-encoded. The other outputs (normal, base color, roughness, metallic,
ambient occlusion, visibility, depth) are raw data. The base color is linear,
and a default base color derived from the DC color is decoded to linear.
`background` is sRGB for LIT and raw for the data outputs.

Enum values:

| Enum | Values |
|---|---|
| `SabercatLightType` | `SABERCAT_LIGHT_DIRECTIONAL` (0), `SABERCAT_LIGHT_POINT` (1), `SABERCAT_LIGHT_RECT` (2) |
| `SabercatLightingOutput` | `SABERCAT_LIGHTING_OUTPUT_LIT` (0), `_NORMAL` (1), `_BASE_COLOR` (2), `_ROUGHNESS` (3), `_METALLIC` (4), `_AMBIENT_OCCLUSION` (5), `_VISIBILITY` (6), `_DEPTH` (7) |
| `SabercatShadowMode` | `SABERCAT_SHADOW_NONE` (0), `SABERCAT_SHADOW_PCSS` (1), `SABERCAT_SHADOW_RAY_TRACED` (2) |
| `SabercatDepthLayout` | `SABERCAT_DEPTH_LAYOUT_NONE` (0), `_DEPTH` (1), `_DEPTH_COVERAGE` (2), `_DEPTH_COVERAGE_FRONT` (4) |

Enum-typed struct fields are `uint32_t`, because the size of an enum is the
compiler's choice.

## Not in the C ABI

These C++ calls have no C equivalent yet. Use the C++ API for them.

- Baked visibility (`bakeVisibility`) and teacher samples
  (`generateTeacherSamples` and its readers)
- External visibility (`setExternalVisibility`) and `ShadowMode::BakedSH`
- The G-buffer readback (`readGBuffer`) and the visibility layers
  (`visibilityView`, `readVisibility`)
- Viewer controls: LOD parameters, debug views, and the up-axis flip
- Presentation to a window
- Project files (`ProjectIO`) and layer transforms

The C ABI loads scenes from files, but it has no file writers. Use the C++ API
for PLY, glTF and cloud-v1 export.
