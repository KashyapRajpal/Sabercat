# Library guide (C++)

This guide tours the C++ library, `Sabercat::sabercat`. It covers the
renderer, the scene and file APIs, the relighting and shadow passes, the
project format, and the rules for sharing a Vulkan device and buffers with
your own code.

For a C or ctypes caller, see [c-api.md](c-api.md). For how to build and link
a prebuilt package, see [consuming.md](consuming.md). For the file formats,
see [formats.md](formats.md).

Everything is in namespace `sabercat`. Include headers as `sabercat/<name>.h`.
The library needs C++20 and Vulkan 1.3. See [building.md](building.md) for the
toolchain.

## Contents

- [Headers](#headers)
- [Renderer lifecycle](#renderer-lifecycle)
- [Scenes](#scenes)
- [Camera and image size](#camera-and-image-size)
- [Forward and backward](#forward-and-backward)
- [Relighting](#relighting)
- [Shadows](#shadows)
- [Baked visibility and teacher samples](#baked-visibility-and-teacher-samples)
- [Loaders and writers](#loaders-and-writers)
- [Projects](#projects)
- [Layer transforms](#layer-transforms)
- [Borrowed buffers](#borrowed-buffers)
- [Ownership and synchronization](#ownership-and-synchronization)
- [Presenting to a window](#presenting-to-a-window)
- [Errors](#errors)

## Headers

| Header | Contents |
|---|---|
| `sabercat/renderer.h` | `Renderer`, `GaussianHostInput`, `GaussianHostOutput`, `RenderMode`, `DebugView` |
| `sabercat/device.h` | `Device`, `VulkanContext`, `EnabledDeviceFeatures`, `DeviceRequirements`, `DeviceFeatures` |
| `sabercat/borrowed.h` | `BorrowedGaussians`, `validateBorrowedGaussians()` |
| `sabercat/camera.h` | `Camera` |
| `sabercat/scene_bounds.h` | `SceneBounds` |
| `sabercat/lighting.h` | `Light`, `EnvironmentLight`, `LightingSettings`, `ShadowSettings`, `VisibilityBakeSettings`, `TeacherSettings` |
| `sabercat/buffer_view.h` | `BufferView`, `CanonicalGaussianBufferViews`, `RelightBufferViews`, `CanonicalGaussianGradientViews` |
| `sabercat/gaussian_cloud.h` | `GaussianCloud`, `GaussianGradients` |
| `sabercat/render_config.h` | Stride and limit constants (`kMaxLights`, `kVisibilityCoefficients`, ...) |
| `sabercat/scene_loader.h` | `loadScene()`, `detectFormat()`, `SceneFormat` |
| `sabercat/project.h` | `Project`, `ProjectLayer`, `ProjectIO` |
| `sabercat/layer_transform.h` | `LayerTransform`, `transformedCopy()`, `mergeLayers()` |
| `sabercat/ply_reader.h`, `gltf_reader.h`, `splat_reader.h`, `spz_reader.h`, `sog_reader.h`, `ksplat_reader.h`, `cloud_v1_reader.h` | Readers, one class per format |
| `sabercat/ply_writer.h`, `gltf_writer.h`, `cloud_v1_writer.h` | `PLYWriter`, `GLTFWriter`, `CloudV1Writer`, `RelightSidecar` |
| `sabercat/c_api.h` | The C ABI. See [c-api.md](c-api.md). |

## Renderer lifecycle

A `Renderer` is created with a size. It is then brought up in one of two
ways. The owned path lets Sabercat create the Vulkan device. The borrowed
path runs on a device your application created.

```cpp
#include "sabercat/renderer.h"

sabercat::Renderer renderer(1280, 720);
if (!renderer.initialize({}, /*enableValidation=*/true, shaderDir))
{
    // Vulkan bring-up failed. The reason is in the log.
}
// ... use the renderer ...
renderer.shutdown();   // the destructor releases the same objects
```

`initialize()` takes three arguments:

| Argument | Meaning |
|---|---|
| `instanceExtensions` | Extra instance extensions. Presenting apps pass what their windowing library needs. Empty for headless use. |
| `enableValidation` | Turns on the Khronos validation layer. If the layer is not installed, Sabercat continues without it. |
| `shaderDir` | Directory with the compiled `*.comp.spv`. Empty skips the forward pass, so `forward()` and `present()` do nothing. Tests that never render use this. |

The shader directory is a path on disk, not a library resource. The
applications in this repository find it with `resolveShaderDir()` in
`apps/common/shader_dir.cpp`. A package built against the installed prefix
gets it from the CMake variable `SABERCAT_SHADER_DIR`. See
[consuming.md](consuming.md).

### Borrowed device

Use `initializeBorrowed()` when your application creates the Vulkan instance
and device. Sabercat then creates only its own allocator and pools on that
device. It destroys none of your handles.

```cpp
// 1. Declare what you need, before vkCreateDevice.
sabercat::DeviceRequirements requirements(
    physicalDevice,
    sabercat::DeviceRequirements::Options{
        sabercat::DeviceRequirements::Need::Required,   // training: fp32 atomics, for backward()
        sabercat::DeviceRequirements::Need::None,       // presentation
        sabercat::DeviceRequirements::Need::None});     // rayTracing

// 2. Create the instance (apiVersion >= VK_API_VERSION_1_3) and the device with
//    requirements.instanceExtensions(), requirements.deviceExtensions(),
//    requirements.featureChain() as pNext, and pEnabledFeatures = nullptr.

// 3. Lend the device to a renderer.
sabercat::Renderer renderer(width, height);
renderer.initializeBorrowed(context, requirements.enabled(), shaderDir);
```

| Item | Meaning |
|---|---|
| `DeviceRequirements` | Checks the GPU and builds the feature chain and extension lists. Throws `std::runtime_error` if a `Required` need is missing. Not copyable or movable. Keep it alive until `vkCreateDevice` returns. |
| `DeviceRequirements::Need` | `None`, `IfSupported`, or `Required`. |
| `DeviceRequirements::enabled()` | The declaration to pass to `initializeBorrowed()`. |
| `VulkanContext` | The instance, physical device, device, and graphics and compute queues with their family indices. Your application keeps these alive. |
| `EnabledDeviceFeatures` | What you enabled at `vkCreateDevice`. It is checked against the GPU. Sabercat cannot check the enables themselves. |

`initializeBorrowed()` throws `std::invalid_argument` if a handle, queue
family or declaration is wrong. The message names the problem. The full
list of checks is in the [borrowed device requirements](#borrowed-buffers).

A renderer can also borrow the device of another renderer. Pass
`other.vulkanContext()` and `other.enabledDeviceFeatures()`. This is how one
scene is rendered at two sizes on one device.

Other device accessors on `Renderer`:

| Method | Returns |
|---|---|
| `ownsDevice()` | `false` when the device is borrowed |
| `sabercatDevice()` | The `Device` for calls that take one (loaders, `mergeLayers`, `ProjectIO::load`). Do not shut it down. |
| `instance()`, `physicalDevice()`, `device()`, `graphicsQueue()`, `graphicsQueueFamily()` | Raw handles for your own rendering, for example a UI library |
| `computeQueue()`, `computeQueueFamily()` | The queue every Sabercat submission uses (see [synchronization](#ownership-and-synchronization)) |
| `deviceFeatures()` | The features the device was created with |
| `backwardSupported()` | Whether `backward()` can run (fp32 buffer atomics) |

## Scenes

A scene is either loaded from a file or given as host arrays. Its GPU
buffers belong to Sabercat, unless you borrow them (see
[Borrowed buffers](#borrowed-buffers)).

```cpp
renderer.loadScene("scene.ply");   // any supported format, see below
```

`loadScene(const char *)` throws `std::runtime_error` for an unsupported or
invalid file. It takes the format from the extension and checks the magic
bytes.

Set host arrays with `setGaussians()`. Read the current GPU parameters back
with `readGaussians()`.

```cpp
sabercat::GaussianHostInput input;
input.count = n;
input.means = means.data();            // 4 floats per Gaussian (xyz + pad)
input.scales = scales.data();          // 4 floats: linear scale xyz + pad
input.rotations = rotations.data();    // 4 floats: normalized xyzw
input.opacities = opacities.data();    // 1 float in [0, 1]
input.shCoefficients = sh.data();      // 16 coefficients x 4 floats (RGB + pad)
input.shDegree = 3;
input.capacity = 0;                    // 0 = exactly count
renderer.setGaussians(input);
```

| Field | Layout |
|---|---|
| `means`, `scales`, `rotations` | 4 floats per Gaussian. The 4th lane is padding. Rotation is `xyzw`. |
| `opacities` | 1 float per Gaussian, activated (in `[0, 1]`) |
| `shCoefficients` | 16 coefficients x 4 floats, band-ascending. RGB plus padding. |
| `scales` | Linear, not log. Values are activated, as the renderer expects. |
| `capacity` | GPU storage to reserve, in Gaussians. `max(count, capacity)` is allocated. |
| `normals`, `baseColorRoughness`, `metallic`, `visibility` | Optional relightable streams. `nullptr` means absent. |

`setGaussians()` throws on a null pointer or `count == 0`. The output
structure `GaussianHostOutput` has the same layout. Each pointer must hold
`count` Gaussians. `readGaussians()` requires `out.count == gaussianCount()`.

`clearScene()` removes the scene, owned or borrowed, without loading another.
Owned GPU buffers are freed. A borrowed scene's buffers stay yours. Afterwards:

- `gaussianCount()` is 0 and `sceneBounds().valid` is false.
- `storageGeneration()` has advanced.
- `forward()` is a no-op.
- `readImage()`, `imageView()`, `readDepth()`, `depthView()`, `readLitImage()`
  and `backward()` throw until a `forward()` of a new scene.
- `present()` shows the clear color.

The camera, lights and other settings are kept. A scene set afterwards
renders as it would on a fresh renderer. Calling `clearScene()` twice, or with
no scene, is fine. The C ABI has the same call, `sabercat_renderer_clear_scene()`.

```cpp
renderer.clearScene();   // an empty viewport; no placeholder Gaussian needed
```

Scene state that the renderer tracks:

| Method | Meaning |
|---|---|
| `gaussianCount()` | How many Gaussians forward and backward process |
| `gaussianCapacity()` | How many the GPU buffers hold |
| `sceneBounds()` | World-space bounds, with `valid == false` before a scene loads |
| `sceneSHDegree()` | The highest SH degree the loaded data carries |
| `setMaxSHDegree(d)` | The SH degree to evaluate, clamped to 0 to 3 |
| `setAntialiased(bool)` | The opacity compensation flag. Loaders set it from the file. A new `loadScene()` resets it. |
| `setFlipUpAxis(bool)` | Folds a Y negation into the view. For Y-down authoring conventions. Does not touch the data. |
| `setLODParams(LODParams)` | Viewer-only LOD knobs. `forward(RenderMode::Differentiable)` ignores them. |

## Camera and image size

`Camera` follows the convention shared with the rest of the Sabercat
interfaces:

- Right-handed, +Y up. The camera looks down its local -Z.
- Depth is `d = -local.z`.
- `u = fx * x / d + cx` and `v = cy - fy * y / d`. The sign flip on `v` is
  deliberate.
- `cameraToWorld` is a column-major `glm::mat4`.

```cpp
sabercat::Camera camera;
camera.width = 1280;
camera.height = 720;
camera.fx = camera.fy = 1000.0f;
camera.cx = 640.0f;
camera.cy = 360.0f;
camera.cameraToWorld = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, 5.0f));
renderer.setCamera(camera);
```

Scene scale varies widely between formats. Use the bounds to frame the scene.
`SceneBounds::focusCenter` and `focusRadius` use the median and the
90th-percentile radius, so a distant background shell does not dominate.

```cpp
const sabercat::SceneBounds b = renderer.sceneBounds();
// Place the camera at b.focusCenter + an offset that scales with b.focusRadius.
```

`setImageSize(w, h)` changes the output size of a headless renderer. It
reallocates the image, so the last forward is invalidated, and the image view
handle changes. Set a camera whose intrinsics match the new size. It throws
while a surface is attached, because the swapchain decides the size there.

## Forward and backward

`forward()` rasterizes the current Gaussians into the image.

| `RenderMode` | Use |
|---|---|
| `Viewer` (default) | Real-time viewing. Honors the viewer LOD knobs. |
| `Differentiable` | Required before `backward()`. Pins the LOD knobs off and saves the per-pixel state backward needs. |

```cpp
renderer.setCamera(camera);
renderer.forward(sabercat::RenderMode::Differentiable);

std::vector<float> image = renderer.readImage();   // width * height RGBA floats, row-major
```

The image is RGBA, row-major, with alpha equal to coverage. `readImage()`
returns the last forward's image. Its rgb is display-referred: the 3DGS color
is sRGB-encoded, like the training photos, so the forward color is sRGB
values premultiplied by coverage. Composite a background in that same space.

`backward()` takes dL/d(image) in the same layout. It computes gradients with
respect to the stored, activated Gaussian values.

```cpp
std::vector<float> dLdImage = /* same layout as readImage() */;
renderer.backward(dLdImage.data());                    // host copy
// or, zero-copy: a device buffer holding dL/d(image) in the same layout
renderer.backward(lossGradientView);                   // sabercat::BufferView
```

| Method | Behavior |
|---|---|
| `backward(const float *)` | Overwrites the gradient buffers from host memory |
| `backward(const BufferView &)` | Reads dL/d(image) from a device buffer, stride 16 |
| `backward(const float *, const float *)` | As above, plus dL/d(depth) from host memory. See [Depth output](#depth-output). |
| `backward(const BufferView &, const BufferView &)` | As above, with dL/d(depth) from a device buffer |
| `imageView()` | The last differentiable forward's image as a device buffer |
| `readDepth()`, `depthView()` | The last differentiable forward's depth output, as a host copy or a device buffer |
| `readGradients()` | Host copy as `GaussianGradients` |
| `gaussianViews()`, `gradientViews()` | Device views of the parameter and gradient streams |

Rules that `backward()` enforces:

- It throws unless the last forward was a `Differentiable` forward of the
  current scene, with the same camera.
- It throws if the device lacks fp32 buffer atomics. Forward is unaffected.
- Gradients are overwritten, not summed. Summing over views is the caller's
  job.
- The alpha channel of dL/d(image) is coverage (`1 - T`). A caller can
  composite its own background with correct gradients.

Gradient streams and their meaning are described in the header
`sabercat/buffer_view.h`. In short, each gradient has the stride of the
parameter it differentiates. Gradients are with respect to stored values, so
the caller applies its own activations (log for scale, logit for opacity,
normalize for rotation). `screenMeans` is the per-view projected-center
gradient used by densification.

### Depth output

The depth output is chosen when the renderer is constructed. The layout is
fixed for the renderer's life, and the default is `DepthLayout::None`, which
writes nothing.

```cpp
sabercat::Renderer renderer(width, height, sabercat::DepthLayout::DepthCoverageFront);
renderer.setCamera(camera);
renderer.forward(sabercat::RenderMode::Differentiable);

std::vector<float> depth = renderer.readDepth();   // width * height * 4 floats, row-major
const float D = depth[i * 4 + 0];                  // not normalized
const float coverage = depth[i * 4 + 1];           // bit-equal to the image alpha
const float expectedDepth = coverage > 0.0f ? D / coverage : 0.0f;
```

| `DepthLayout` | Floats per pixel | Components |
|---|---|---|
| `None` (default) | 0 | No depth output. `readDepth()` and `depthView()` throw. |
| `Depth` | 1 | D |
| `DepthCoverage` | 2 | D, coverage |
| `DepthCoverageFront` | 4 | D, coverage, front depth, 0 |

- **D** is the sum of `T * alpha * z` over the blended Gaussians, where `z` is
  camera depth. It is not normalized. Divide by coverage for the expected
  depth.
- **Coverage** is `1 - T`, bit-equal to the image's alpha.
- **Front depth** is the camera depth of the frontmost contributing Gaussian,
  or 0 where none contributes.

Only a `Differentiable` forward writes the depth output. A `Viewer` forward
does not. Color is bit-identical whatever the layout.

`readDepth()` returns `width * height * n` floats. `depthView()` has stride
`4 * n`, offset 0 and size `width * height * n * 4` bytes. The rules are the
same as for `readImage()` and `imageView()`: they throw without a
differentiable forward of the current scene with the same camera, and
`depthView()` follows the view rules in
[Ownership and synchronization](#ownership-and-synchronization).

Depth gradients. The `backward()` overloads with a second argument take
dL/d(depth) in the same layout as `readDepth()`:

```cpp
std::vector<float> dLdImage = /* width * height * 4, as readImage() */;
std::vector<float> dLdDepth = /* width * height * n, as readDepth() */;
renderer.backward(dLdImage.data(), dLdDepth.data());
```

- Only component 0 (dL/dD) is read. The other components are ignored, so the
  whole `readDepth()` buffer can be passed.
- The coverage gradient flows through the alpha of `dLdImage`. The front-depth
  component is not differentiated.
- Passing `nullptr`, or a default-constructed `BufferView`, gives the color-only
  backward exactly.
- Throws if a depth gradient is given and the layout is `DepthLayout::None`.
- Gradients are overwritten, as with the color-only backward.

### Densification

Parameter buffers can grow. The renderer treats them like `std::vector`:

| Method | Effect |
|---|---|
| `reserveGaussians(capacity)` | Grows storage and copies the active prefix. Returns `false` with nothing changed if `capacity <= gaussianCapacity()`. Replaces every buffer handle. Throws without a scene. |
| `setGaussianCount(n)` | Changes how many Gaussians are processed, within capacity. No GPU work. Invalidates the last differentiable forward. |
| `storageGeneration()` | Advances when a buffer handle changes. Re-fetch views when it moves. |

A trainer writes new Gaussians into `[oldCount, newCount)` through
`gaussianViews()`, then raises the count. Gaussians in that range are
whatever the trainer wrote.

## Relighting

With lighting enabled, every `forward()` also splats a G-buffer (normal, base
color, roughness, metallic, ambient visibility, depth). It then shades the
G-buffer against the lights and environment. The lit image is separate from
the color image.

```cpp
renderer.setLightingEnabled(true);

sabercat::Light sun;
sun.type = sabercat::LightType::Directional;
sun.direction = glm::normalize(glm::vec3(-0.3f, -1.0f, -0.2f));
sun.color = glm::vec3(1.0f, 0.95f, 0.9f);
sun.intensity = 2.0f;
renderer.setLights({sun});                    // at most kMaxLights (8)

renderer.setEnvironment(sabercat::EnvironmentLight::uniform(glm::vec3(0.1f)));

sabercat::LightingSettings settings;
settings.output = sabercat::LightingOutput::Lit;
renderer.setLightingSettings(settings);

renderer.forward(sabercat::RenderMode::Viewer);
std::vector<float> lit = renderer.readLitImage();   // RGBA; alpha = coverage
```

Color spaces. `Lit` is display-referred: the shading is computed in linear
light, the background is decoded from sRGB, the composite
`(shaded * coverage + background * (1 - coverage)) * exposure` is formed in
linear light and clamped to `[0, 1]`, and the rgb is then sRGB-encoded. Alpha
is coverage. Every other output is a raw data value with no encoding: normal
(`n * 0.5 + 0.5`), linear base color, roughness, metallic, ambient
visibility, visibility, and depth (`depthScale` times the expected depth).
Their composite is `(value * coverage + background * (1 - coverage)) * exposure`,
and the background is used as given.

| Type | Notes |
|---|---|
| `Light` | `type` is `Directional`, `Point` or `Rect`. `castsShadows` says whether shadows apply. `Point` and `Rect` use `position`; `Rect` also uses `axisX`, `axisY`, `halfWidth` and `halfHeight`. |
| `EnvironmentLight` | Radiance SH of degree 2 (9 RGB coefficients). `uniform()` and `sky()` build common cases. All zero means no environment. |
| `LightingSettings` | `output` (`LightingOutput`), `exposure`, `background`, `rectSamples`, `depthScale`, `visibilityLight` |
| `LightingOutput` | `Lit`, `Normal`, `BaseColor`, `Roughness`, `Metallic`, `AmbientOcclusion`, `Visibility`, `Depth` |

Lights are in the scene's own (asset) space, the same space as the means.

Scenes without relightable streams get derived defaults. The defaults are
base color equal to the DC color, clamped to `[0, 1]` and sRGB-decoded to
linear (the DC color is display-referred, so a DC color of 0.5 gives a
linear base color of about 0.214), roughness 0.8, metallic 0, ambient
visibility 1, and the normal along the shortest ellipsoid axis facing the
camera. A base color stream you supply is linear and is used as given.

Other relight accessors:

| Method | Returns |
|---|---|
| `readGBuffer()` | Premultiplied G-buffer sums of the last lit forward (`GBufferData`) |
| `litImageView()` | Device view of the lit image |
| `visibilityView()`, `readVisibility()` | The per-light visibility layers (`kMaxLights` layers of `width * height` floats) |

The lit image is not differentiated. `backward()` still uses the color image.

## Shadows

Shadows apply to lights with `castsShadows` set.

| `ShadowMode` | Meaning |
|---|---|
| `None` | Every light fully visible (default) |
| `PCSS` | Soft shadows over a splat shadow map. The baseline. |
| `RayTraced` | Transmittance traced through the Gaussians. Needs ray query. |
| `BakedSH` | Per-Gaussian visibility SH, evaluated toward each light. Needs a baked or loaded visibility stream. |
| `External` | Visibility supplied by the caller through `setExternalVisibility()`. The view must stay valid while the mode is `External`. |

```cpp
sabercat::ShadowSettings shadows;
shadows.mode = sabercat::ShadowMode::PCSS;
shadows.focusCenter = bounds.focusCenter;     // or leave to the scene bounds
shadows.focusRadius = bounds.focusRadius;     // 0 = use the loaded scene's bounds
renderer.setShadowSettings(shadows);
```

Light cameras frame the focus sphere. When `focusRadius` is 0, they frame the
scene's bounds. A borrowed scene has no bounds, so set the focus explicitly
in that case.

`RayTraced` needs the device's ray-query features. Create the device with
`DeviceRequirements::Options::rayTracing` set. Without ray query,
`RayTraced` is rejected, and PCSS still works.

`shadowFrameStats()` reports rays dispatched, acceleration-structure builds,
and cache reuse for the last lit forward.

`ShadowSettings::cacheAcrossFrames` reuses the acceleration structure and
cached shadows across frames. The renderer sees changes made through its own
API. After editing parameters in place through `gaussianViews()` or a
borrowed scene, call `markSceneChanged()`.

## Baked visibility and teacher samples

These calls are synchronous.

| Method | Effect |
|---|---|
| `bakeVisibility(VisibilityBakeSettings)` | Traces every Gaussian's visibility SH into the `visibility` stream. Needs ray query. Enables `ShadowMode::BakedSH` and ambient occlusion from the environment. A borrowed scene must have a borrowed visibility stream. |
| `generateTeacherSamples(TeacherSettings)` | Selects pixels from the last lit forward and traces a ray-traced visibility target for each. Returns the count. Needs ray query. |
| `teacherSampleView()` | Device buffer of `VisibilityTrainingSample` (32 bytes each). Smooth samples come first. |
| `readTeacherSamples()`, `readTeacherSamplePixels()` | Host copies. A pixel index has its top bit set for an edge sample. |
| `teacherSmoothSampleCount()` | How many samples came from the smooth set |

`VisibilityBakeSettings::rayLength` of 0 means four times the scene's
bounding radius. A borrowed scene has no bounds, so set `rayLength` there.

## Loaders and writers

### Loaders

| Format | Entry point |
|---|---|
| Any supported format, by extension | `Renderer::loadScene(path)`, or `sabercat::loadScene(device, path)` |
| `.ply` (3DGS, and SuperSplat compressed) | `PLYReader::read(device, path)` |
| `.splat` | `SplatReader::read(device, path)` |
| `.spz` | `SpzReader::read(device, path)` |
| `.ksplat` | `KsplatReader::read(device, path)` |
| `.sog` bundle or `meta.json` | `SogReader::read(device, path)` |
| `.gltf`, `.glb` (`KHR_gaussian_splatting`) | `GltfReader::read(device, path)` |
| cloud-v1 directory or `manifest.json` | `CloudV1Reader::read(device, path)` |

`detectFormat(path)` returns the `SceneFormat` that `loadScene()` would use.
Readers return a `std::shared_ptr<GaussianCloud>` with host data only. Call
`GaussianCloud::uploadToGPU()` to make it renderable. A `Renderer` does this
for `loadScene()`.

Each reader maps its format's attributes to the canonical stream layout. The
per-format decisions, including what is rejected, are in
[formats.md](formats.md).

### Writers

| Output | Entry point |
|---|---|
| Binary `.ply` (3DGS, with R3DG attributes when relight streams exist) | `PLYWriter::write(cloud, path)` |
| `.glb` with one node per layer | `GLTFWriter::writeGLB(layers, path)` |
| cloud-v1 directory | `CloudV1Writer::write(cloud, directory)` |
| cloud-v1 in memory | `CloudV1Writer::encode(cloud)` returns a `FileMap` |
| relight sidecar (`sabercat.relight-v1`) | `RelightSidecar::encode(cloud)` and `RelightSidecar::decode(files, cloud, label)` |

Writers read the cloud's host vectors. A cloud that was uploaded and whose
host copy was dropped must call `downloadFromGPU()` first.

`GLTFWriter::writeGLB()` takes `GLTFExportLayer` entries, each with a
`cloud`, a `LayerTransform` and a name. The output extension is
`KHR_gaussian_splatting`. Relight streams are not exported to glTF.

## Projects

A `.sabercat` project holds layers, a placement for each layer, and
application state.

```cpp
sabercat::Project project;
sabercat::ProjectLayer layer;
layer.cloud = cloud;                     // std::shared_ptr<GaussianCloud>
layer.id = "base";                       // [A-Za-z0-9_-], 1 to 64 characters
layer.name = "Base scene";               // any UTF-8 text
layer.transform = sabercat::LayerTransform{};
project.layers.push_back(layer);
project.appStateJson = R"({"camera": {"fx": 1000}})";   // must be a JSON object

sabercat::ProjectIO::save(project, "scene.sabercat");
sabercat::Project loaded = sabercat::ProjectIO::load(renderer.sabercatDevice(), "scene.sabercat");
```

| Item | Meaning |
|---|---|
| `Project::layers` | The layers, in order |
| `Project::appStateJson` | Application state, stored verbatim under `app` in `project.json`. Must parse as a JSON object. `"{}"` when the application has none. |
| `Project::savedWithVersion` | Filled in by `load()`. Ignored by `save()`. |
| `ProjectLayer` | `cloud`, `id`, `name`, `transform`, `visible`, `locked` |

`ProjectIO::save()` writes to a temporary file and renames it over the target.
A failed save leaves the old file in place. It throws `std::runtime_error`
for a null cloud, a bad or duplicate layer id, host arrays with the wrong
size, a non-unit rotation, a scale of 0 or less, or application state that is
not a JSON object.

`ProjectIO::load()` returns clouds with host data only. The round trip is
bit-exact for every host stream, the SH degree, the antialiased flag, the
transform, flags, ids and names. It throws `std::runtime_error` for a corrupt
archive, an unknown format, or a newer format version.

The file layout is in [formats.md](formats.md).

## Layer transforms

`LayerTransform` is a rigid placement: a rotation, a translation and a
uniform scale. Uniform scale keeps every per-Gaussian quantity exact.

| Member | Meaning |
|---|---|
| `rotation` | Unit quaternion (`glm::quat`, constructed as w, x, y, z) |
| `translation` | `glm::vec3` |
| `scale` | Uniform, greater than 0 |
| `matrix()` | `T * R * S` as a `glm::mat4` |
| `isIdentity()` | True only for exact identity |

Two functions apply placements to host data. Neither does GPU work.

| Function | Result |
|---|---|
| `transformedCopy(device, source, transform)` | A copy with the transform baked in, as the glTF reader bakes node transforms. Identity gives a bit-exact copy. |
| `mergeLayers(device, layers)` | Concatenates transformed layers into one cloud. The SH degree is the maximum, and lower-degree layers are zero-padded. A relight stream is kept only if every layer has it. Throws on an empty list. |

Call `computeBounds()` on the result before using its bounds.

## Borrowed buffers

Borrowing lets your application own the buffers that hold the scene.
Sabercat renders and differentiates them in place. It never allocates, frees,
grows or uploads into them.

```cpp
#include "sabercat/borrowed.h"

sabercat::BorrowedGaussians scene;
scene.parameters.means = {meansBuffer, 0, meansBytes, 16};
scene.parameters.scales = {scalesBuffer, 0, scalesBytes, 16};
scene.parameters.rotations = {rotationsBuffer, 0, rotationsBytes, 16};
scene.parameters.opacity = {opacityBuffer, 0, opacityBytes, 4};
scene.parameters.shCoefficients = {shBuffer, 0, shBytes, 16};
scene.parameters.gaussianCount = count;
scene.parameters.shDegree = 3;
// Gradients: all six views, or none (Sabercat then allocates its own).
renderer.borrowGaussians(scene);
```

`validateBorrowedGaussians(physicalDevice, device, scene)` runs the same
checks up front and returns the capacity. It is public, so an application can
check its buffers before borrowing them.

| Check | Sabercat verifies |
|---|---|
| Handles | Each required parameter stream has a buffer. Confidence is optional and unused. |
| Stride | Canonical stride per stream: 16 bytes for xyz, rotation, scale and normals; 4 for opacity; 16 per SH coefficient |
| Offset and size | Offset is a multiple of `minStorageBufferOffsetAlignment`. Size is at least one Gaussian and at most `maxStorageBufferRange`. `offset + size` lies inside the buffer. |
| SH | `shDegree <= 3` |
| Count | `1 <= gaussianCount <= capacity` |
| Gradients | All six given, or none. Each covers the capacity. |
| Overlap | No two streams' used ranges overlap in the same buffer |

Sabercat cannot check these through Vulkan. The owner must guarantee them.

- The buffers have `VK_BUFFER_USAGE_STORAGE_BUFFER_BIT`. Add
  `VK_BUFFER_USAGE_TRANSFER_SRC_BIT` if you use `readGaussians()` or
  `readGradients()`.
- They were created on the renderer's device.
- Memory is bound before the call.
- Sharing mode is exclusive to the compute queue's family, or concurrent with
  that family included.
- They stay alive until the next scene replacement, `clearScene()` or `shutdown()`.

Borrowed scenes have these limits:

- `setGaussianCount()` moves within the borrowed capacity. The handles stay
  the same.
- `reserveGaussians()` throws. To grow, allocate larger buffers, copy the
  data, and call `borrowGaussians()` again. That advances
  `storageGeneration()`.
- `gaussianViews()` and `gradientViews()` return the views you borrowed. The
  count they report is the current one.
- A borrowed scene has no bounds, so set shadow focus and bake ray length
  explicitly.

### Borrowed device requirements

Checked by `initializeBorrowed()` and by `validateBorrowedContext()` in
`device.h`.

| Requirement | Result when unmet |
|---|---|
| `instance`, `physicalDevice`, `device` are non-null | Rejected |
| At least one queue: compute or graphics, each with its family index | Rejected |
| Compute queue's family has compute; graphics queue's family has graphics | Rejected |
| GPU supports Vulkan 1.3 or later | Rejected |
| At least 256 compute invocations per workgroup | Rejected |
| Declared `shaderBufferFloat32AtomicAdd` is supported | Rejected |
| Declared `swapchain` has the extension enabled and a graphics queue | Rejected |
| Declared `rayQuery` is supported and its extensions are enabled | Rejected |

`backward()` needs the atomic-float declaration. `attachSurface()` and
`present()` need `swapchain`. `setUiRenderCallback()` needs `dynamicRendering`.
The owner must enable each declared feature at `vkCreateDevice`. Sabercat
checks only some of these with Vulkan calls.

The validation layer reports most violations in Debug builds. On a borrowed
device, the messenger belongs to the owner. Sabercat installs none.

## Ownership and synchronization

This contract applies when your code and Sabercat share a device and
buffers. It is the rule set for both owned and borrowed devices.

- **Every Sabercat call is synchronous.** `forward()`, `backward()`,
  `setGaussians()`, `reserveGaussians()`, `readImage()`, `readGradients()`
  and `readGaussians()` return only after their GPU work has completed. Between
  calls, the application may read or write any parameter or gradient buffer.
- **One queue.** Sabercat submits to `computeQueue()`. Each submission starts
  and ends with a full memory barrier. Submit your own work that touches
  Sabercat's buffers to `computeQueue()` too. No fences, waits or barriers
  are needed on either side.
- **Other queues need a host wait.** If your work uses a different queue, wait
  for it on the host before calling Sabercat. Sabercat's completion covers
  the other direction.
- **Exclusive sharing.** Sabercat's buffers use `VK_SHARING_MODE_EXCLUSIVE`.
  Queue family ownership transfers are not part of this contract.
- **External synchronization.** `VkQueue` is externally synchronized. Do not
  submit to `computeQueue()` from another thread while a Sabercat call runs. A
  renderer is single-threaded.
- **Zero-copy loss.** `imageView()` holds the last differentiable forward's
  image. A loss kernel on `computeQueue()` can read it right after `forward()`
  returns. Its dL/d(image) output goes to `backward(const BufferView &)` with no
  wait. `imageView()` throws after a viewer forward, after a scene change,
  and before any forward.
- **View lifetime.** Parameter and gradient views are valid until
  `storageGeneration()` changes, or until `shutdown()`. The count inside a
  view is a snapshot. Re-read it after `setGaussianCount()`.
- **Gradient contents** are valid from `backward()` returning until the next
  `backward()` or a growing reserve.
- **Image view.** The handle is stable across forwards and scene changes. It
  changes only when the image is resized. Its contents are valid from
  `forward()` returning until the next `forward()`. `depthView()` follows the
  same rules.
- **Borrowed device.** `computeQueue()` is the queue you lent. If you lent only
  a graphics queue, that queue is used.

## Presenting to a window

Sabercat is windowing-agnostic. The application creates the window and the
surface. A borrowed device needs `presentation` in `DeviceRequirements`, so
that the swapchain extension is enabled.

| Method | Use |
|---|---|
| `instance()` | Create the surface from this instance |
| `attachSurface(surface, width, height, onSurfaceRelease)` | Builds the swapchain. The application destroys the surface when `onSurfaceRelease` fires during `shutdown()`. |
| `present()` | Blits the last forward to the swapchain |
| `setUiRenderCallback(fn)` | Records extra draw commands after the blit. Needs dynamic rendering. Pass an empty function to remove it. |
| `setDebugView(DebugView)` | What `present()` shows: `Final`, `Alpha`, `Depth`, `TileDensity` or `SHDegree`. Does not re-run forward. |
| `setLightingSettings()` with `output` | Chooses the lit image or a G-buffer channel for `present()` |

Instance extensions for a window are the windowing library's responsibility.
Pass them to `initialize()`, or add them to your own instance.

## Errors

Errors reach the caller in one of three ways.

| Source | Behavior |
|---|---|
| `Renderer::initialize()` | Returns `false`. The reason is in the log. |
| `Renderer::initializeBorrowed()`, `validateBorrowedContext()`, `validateBorrowedGaussians()` | Throw `std::invalid_argument`. The message names the stream or declaration. |
| Loaders, writers, `ProjectIO`, and most other calls | Throw `std::runtime_error` |

`Logger` writes messages with levels `debug`, `info`, `warning`, `error` and
`critical`. `Logger::setLevel()` sets the threshold.

The borrowing checks run before any state changes. A `borrowGaussians()` call
that fails leaves the current scene in place. The C ABI makes the same
promise for `sabercat_renderer_set_gaussian_views` and
`sabercat_renderer_set_shadow_settings`. See [c-api.md](c-api.md).
