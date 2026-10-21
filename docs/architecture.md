# Architecture

This document describes how Sabercat is built. It covers the module map,
the data layout, the GPU passes, synchronization, ray tracing, relighting,
baking, and the Studio internals. It also records the measurements taken on
the current code.

Related documents:

- [shaders.md](shaders.md): every compute kernel and shared header.
- [PERFORMANCE.md](../PERFORMANCE.md): the design rules that shape the code.
- [library.md](library.md), [c-api.md](c-api.md): the public APIs.
- [formats.md](formats.md): file formats and their conventions.

Platform: Windows 10/11 x64, Vulkan 1.3, Visual Studio 2022.

---

## 1. Components

| Target | Output | Role |
| --- | --- | --- |
| `sabercat` | static library, `Sabercat::sabercat` | Renderer, passes, loaders, writers, project files |
| `sabercat_c` | shared library, `Sabercat::sabercat_c` | C ABI over `Renderer` |
| `sabercat_viewer` | `apps/viewer` | Viewer: window, orbit camera, ImGui panels |
| `sabercat_studio` | `apps/studio` | Studio: layers, project files, export |
| `sabercat_shaders` | SPIR-V files | Compiles every `.slang` kernel with `slangc` |
| `apps/common` | shared app code | Used by the viewer and Studio |

Layering:

```
apps (viewer, studio)        window, input, ImGui; creates the surface
        | VkSurfaceKHR, device handles (app-owned or Sabercat-owned)
        v
sabercat library             Device, Buffer, passes, loaders, writers
        | SPIR-V from sabercat_shaders
        v
Slang compute kernels
```

Rules:

- The library never links a windowing library.
- The library depends on Vulkan and on decoders for some input formats.
  The decoders are spz, zlib, libwebp and nlohmann/json. They are linked
  privately.
- Only `Device` calls `vkCreateInstance` and `vkCreateDevice`.
- Passes take buffer views, not owned buffers. This is the seam that lets a
  host application own the device and the Gaussian buffers.

---

## 2. Module map

Headers are in `include/sabercat/`. Sources are in `src/renderer/` unless
noted.

| Area | Headers | Sources |
| --- | --- | --- |
| Device and memory | `device.h`, `buffer.h`, `image.h`, `vk_check.h`, `logger.h` | `device.cpp`, `buffer.cpp`, `image.cpp`, `vma_impl.cpp` |
| Compute plumbing | `shader_compiler.h`, `descriptor_set.h`, `pipeline.h`, `command_buffer.h` | `shader_compiler.cpp`, `descriptor_set.cpp`, `pipeline.cpp`, `command_buffer.cpp` |
| Gaussian data | `gaussian_cloud.h`, `buffer_view.h`, `borrowed.h`, `scene_bounds.h`, `render_config.h` | `gaussian_cloud.cpp`, `borrowed.cpp` |
| Loaders | `scene_loader.h`, `ply_reader.h`, `splat_reader.h`, `spz_reader.h`, `gltf_reader.h`, `ksplat_reader.h`, `sog_reader.h`, `cloud_v1_reader.h` | matching `.cpp` files |
| Writers and projects | `ply_writer.h`, `gltf_writer.h`, `cloud_v1_writer.h`, `project.h`, `file_map.h`, `layer_transform.h` | matching `.cpp`, `zip_archive.{h,cpp}` (internal), `sh_rotation.{h,cpp}` (internal) |
| Forward and backward | `forward_pass.h`, `backward_pass.h`, `debug_visualizer.h`, `camera.h` | `forward_pass.cpp`, `backward_pass.cpp`, `debug_visualizer.cpp` |
| Lighting, shadows, bake | `lighting.h`, `relight_pass.h`, `shadow_pass.h`, `visibility_pass.h`, `gaussian_acceleration_structure.h` | `lighting.cpp`, `relight_pass.cpp`, `shadow_pass.cpp`, `visibility_pass.cpp`, `gaussian_acceleration_structure.cpp` |
| Facade and C ABI | `renderer.h`, `c_api.h` | `renderer.cpp`, `src/api/c_api.cpp` |
| Helpers | `parallel_for.h`, `approx.h`, `version.h.in` | none |

Kernels are in `shaders/`. See [shaders.md](shaders.md).

---

## 3. Device

`Device` owns the Vulkan instance, the physical and logical device, the queues,
and the VMA allocator. It runs headless. No surface exists when it is created.

### Creation models

- **Model A (owned).** `Renderer::initialize()` creates the instance and
  device. The caller passes the instance extensions its windowing layer
  needs. `Device` never calls a windowing API.
- **Model B (borrowed).** The host creates the instance and device from a
  `DeviceRequirements` declaration. It passes them to
  `Renderer::initializeBorrowed()`, or to `Device::adopt()`. Sabercat checks
  the declaration (section 11).

VMA and the transfer pool belong to Sabercat in both models.

### Requirements

`DeviceRequirements` (`device.h`) is the single source of what a device must
have. Each capability has a `Need` value: `None`, `IfSupported` or `Required`.

| Option | Enables | Used by |
| --- | --- | --- |
| `training` | `VK_EXT_shader_atomic_float` and the float buffer atomic feature | `backward()` |
| `presentation` | `VK_KHR_swapchain`, dynamic rendering | `present()`, the UI callback |
| `rayTracing` | `VK_KHR_acceleration_structure`, `VK_KHR_deferred_host_operations`, `VK_KHR_ray_query`, `accelerationStructure`, `rayQuery`, `bufferDeviceAddress` | shadows, bake, teacher samples |

Always checked:

- Vulkan 1.3 or later.
- At least 256 compute invocations per workgroup. Vulkan guarantees only 128,
  and `tile_rasterize` needs 256.

`VK_KHR_swapchain` is enabled only when the instance has `VK_KHR_surface`.
Device extensions that depend on instance extensions would otherwise fail
`vkCreateDevice` for headless callers.

The 16-bit, fp16, int8 and buffer-device-address features are enabled when
the GPU supports them. The kernels do not depend on them.

### Queues

- One graphics queue, used for presentation.
- One compute queue, `submitQueue()`, used for every other submission. It
  has its own queue index when the GPU allows it.
  `DeviceFeatures::hasDedicatedComputeQueue` reports which case applied.
- `immediateSubmit()` is for load-time work only. It records on the transfer
  pool and submits to `submitQueue()`.

### Feature report

`DeviceFeatures` lists subgroup size and operations, the 16-bit and int8
storage and arithmetic features, buffer device address, and the cooperative
matrix and vector extensions. The cooperative extensions are detected only.
No kernel uses them.

### Shutdown order

`Device::shutdown()` frees the VMA allocator that every `Buffer` and `Image`
destructor calls. `Renderer::shutdown()` therefore resets the Gaussian cloud,
the forward pass and the debug visualizer, in that order, before it calls
`Device::shutdown()`. A surface is destroyed after its swapchain and before
the instance.

---

## 4. Buffers and compute plumbing

- **`Buffer`** wraps a VMA allocation with `VMA_MEMORY_USAGE_AUTO`. The default
  usage is `STORAGE_BUFFER | TRANSFER_SRC | TRANSFER_DST`. `upload()` and
  `download()` go through a temporary host-visible staging buffer and
  `immediateSubmit`. They flush after writing staging and invalidate before
  reading it.
- **`ShaderCompiler`** loads SPIR-V that `slangc` produced at build time. The
  runtime does not compile Slang.
- **`DescriptorSet`** owns a pool and its layouts. `bindBuffer` and
  `bindImage` update the set immediately.
- **`Pipeline`** creates pipeline layouts and compute pipelines.
- **`CommandBuffer`** is a record-only wrapper. The `examples/*_test` programs use it. The
  renderer records through raw `vkCmd*` calls.

Passes that run per frame record into their own command pools. `ForwardPass`
submits on its own pool. `Renderer` keeps `FrameContext frames[2]`, one pool
per frame in flight, each with its own semaphores and fence.

---

## 5. Data layout

### Canonical Gaussian streams

Every stream is padded to a `vec4` (16-byte stride). A tight `vec3` array
would misalign: `std430` gives `vec3` 16-byte alignment, so every element after
the first would be read from the wrong offset. The strides are defined once in
`shaders/config.slangh` and used by both host and shader.

| Stream | Type | Stride | Meaning |
| --- | --- | --- | --- |
| means | vec4 | 16 B | world position, w unused |
| scales | vec4 | 16 B | linear standard deviation per axis |
| rotations | vec4 | 16 B | quaternion stored as **xyzw**, normalized on load |
| sh | vec4 per coefficient | 16 B | RGB, 16 coefficients (degree 3), band-ascending |
| opacities | float | 4 B | sigmoid-activated, in [0, 1] |

Covariance is never stored. The forward pass derives it from scale and
rotation.

Optional relight streams (section 13):

| Stream | Stride | Notes |
| --- | --- | --- |
| normals | 16 B | unit world normal |
| base_color_roughness | 16 B | linear RGB and roughness |
| metallic | 4 B | in [0, 1] |
| visibility | 16 coefficients x 4 B | baked visibility SH (section 15) |

### `GaussianCloud`

- Host copies are parallel `std::vector`s: `positions()`, `scales()`,
  `rotations()`, `shCoefficients()`, `opacities()`, plus the relight vectors.
- `times()` is reserved for a future 4D schema. No kernel reads it.
- The GPU buffers are sized to a **capacity** (at least the count). Growing
  the capacity reallocates the streams with a copy of the first `count`
  elements. `setCount()` changes the count with no GPU work.
- `shDegree()` is the SH degree the file declared. It is 0 for `.splat`
  (DC only), 3 for a full `.ply`, and the header value capped at 3 for `.spz`.
- `antialiased()` records the compensation flag (section 9).
- Gradient buffers are allocated on the first `backward()` and dropped by
  `uploadToGPU()`. A viewer never allocates them.
- A cloud is either **owned** (Sabercat allocates the buffers) or
  **borrowed** (the host owns them; section 11). A borrowed cloud has no host
  copy and cannot grow.

### Uniform data

| Data | Where | Size | Notes |
| --- | --- | --- | --- |
| `CameraUniforms` | push constant | 128 B | the Vulkan-guaranteed minimum; a `static_assert` pins it |
| `LODParams` | storage buffer, binding 8 | 44 B | geometry and appearance LOD; zero in gradient runs |
| `InstanceDispatch` | storage buffer | 48 B | instance total and indirect dispatch arguments |
| Per-kernel params | push constants | small | each kernel declares its own struct |

`CameraUniforms` holds the camera, the tile grid, the count, the SH degree
cap, the camera world position, and `renderFlags`. The flags are
`ANTIALIASED` (bit 0) and `SAVE_BACKWARD_STATE` (bit 1). Any new field needs
its own binding, because the push-constant block is full.

### Scene bounds

`SceneBounds` holds the world AABB of the finite positions, the median
position `focusCenter`, and `focusRadius`, the 90th percentile distance from
the median. The viewer frames the scene on `focusCenter`. Outdoor captures
have a distant background shell, so the full AABB can be 100 times larger than
the subject.

---

## 6. Loading

### Dispatch

`loadScene(device, path)` (`scene_loader.h`) chooses a reader by extension.
Magic bytes can veto the choice. `detectFormat()` exposes the same test to
the viewer. Readers are plain static functions, not a class hierarchy. Loading
is one-shot and whole-file.

| Format | Reader | Notes |
| --- | --- | --- |
| `.ply` (3DGS) | `PLYReader` | `binary_little_endian`. SH degree comes from the header. Also reads R3DG relight attributes and the SuperSplat compressed chunk layout |
| `.splat` | `SplatReader` | 32-byte records, no header, DC only |
| `.spz` | `SpzReader` | Decoded by the pinned spz submodule (v3.0.0). Degree 4 input is truncated to degree 3 with a warning |
| glTF / GLB | `GltfReader` | `KHR_gaussian_splatting`, plus the `_compression_spz_2` extension |
| `.ksplat` | `KsplatReader` | GaussianSplats3D layout, no dependency |
| `.sog` or `meta.json` | `SogReader` | PlayCanvas SOG bundle or unbundled `meta.json`. Uses libwebp |
| cloud-v1 directory | `CloudV1Reader` | Sabercat's canonical directory format (formats.md) |

### Canonical contract

Every loader produces the same values:

- Opacity is sigmoid-activated, in [0, 1].
- Scale is a linear standard deviation (exp-activated on load).
- Rotation is normalized **xyzw**.
- SH coefficient 0 is the raw DC term. Coefficients are band-ascending, unused
  slots are zero, and the degree is discovered from the file.

The formats need different work to reach this contract:

| | `.ply` | `.splat` | `.spz` |
| --- | --- | --- | --- |
| scale | `exp()` | already linear | `exp()` |
| opacity | `sigmoid()` | already activated, `byte/255` | `sigmoid()` |
| SH DC | raw `f_dc` | display RGB, inverted | raw |
| rotation | `wxyz`, reordered | `wxyz`, reordered | already `xyzw` |
| higher SH | channel-major | absent | coefficient-major, RGB interleaved |

Wrong values produce a plausible image, not an error. Each reader has a test
that fails if one of these steps is skipped.

### Parallel parse

- `PLYReader` resolves every property's byte offset once after the header
  parse. The per-vertex loop then does only `memcpy` and arithmetic.
- `parallelFor()` (`parallel_for.h`) splits large loops into chunks. Each
  worker writes a disjoint range of pre-sized vectors. Below 50,000 elements
  it runs inline.
- Quaternion normalization uses SSE2 with a scalar `sqrt`. The fast
  reciprocal-square-root approximation is not used.

Measured (Release, 16 logical cores): 300k synthetic Gaussians parse in
about 33 ms and upload in about 29 ms. A 1M-Gaussian scene is expected at
100-150 ms.

### Decoder notes

- **`.spz`**: the adapter keeps the canonical contract. spz returns
  pre-activation floats in RUB coordinates, which match the renderer's +Y up,
  -Z forward convention. No axis flip is needed.
- **glTF**: node transforms are composed into the splats. Covariance is
  exact under uniform scale and re-diagonalized (Jacobi) under non-uniform
  scale. SH bands 1-3 are rotated by matrices solved numerically against the
  shader's own basis.
- **SuperSplat compressed PLY**: 11-10-11 position and scale, smallest-three
  quaternion, 8-bit colour and opacity, uchar SH. Malformed files fail with a
  message naming the problem.
- **`.splat` sanity check**: the format has no header, so the reader checks the
  first 64 records for finite, positive values. A wrong file fails loudly.

---

## 7. Forward rendering

### Frame flow

`Renderer::forward()` builds `CameraUniforms` and calls `ForwardPass::run()`.
It does nothing if no pass exists, no cloud is loaded, or the camera is not
valid.

`Renderer::present()`:

1. Waits on the frame's fence and acquires a swapchain image.
2. Records a barrier from `UNDEFINED` to `TRANSFER_DST_OPTIMAL`.
3. Blits the source to the swapchain image, or clears it if nothing has been
   rendered.
4. Runs the UI callback inside `vkCmdBeginRendering`, if one is set.
5. Barrier to `PRESENT_SRC_KHR`, submit, present.

The blit source is:

- the lit image, when `DebugView::Final` and a lit frame exists (section 13);
- the forward colour image, for `DebugView::Final`;
- the `DebugVisualizer` output, for the other debug views.

Out-of-date and suboptimal results from acquire or present trigger
`recreateSwapchain()`. The forward pass renders into its own offscreen image,
so the swapchain format does not matter. No render pass or graphics pipeline
is used.

### Kernel chain

`ForwardPass` owns the pipelines, the descriptor sets, the intermediate buffers,
and the output images. Its stages run in this order.

| Kernel | Work |
| --- | --- |
| `forward_project` | world to camera, project the mean, derive the 3D covariance from scale and rotation, project it, invert to a conic, evaluate SH, compute the footprint, cull. Culled Gaussians get zero tiles |
| `instance_dispatch` | one thread: sum the instance total into the sort's sizes and two indirect dispatch commands |
| `prefix_sum` | 3-phase exclusive scan (2048-element blocks). Used for the tile offsets and, per radix pass, the digit offsets |
| `tile_binning` | one `(tile, depth)` key per overlapped tile. Positive float bits sort like the values, so no key transform is needed |
| `radix_sort_histogram`, `radix_sort_scatter` | LSD radix sort, 8-bit digits over a 64-bit key: 8 passes with a ping-pong buffer. The scatter ranks elements with a serial scan within the workgroup, which keeps the sort stable |
| `tile_ranges` | `[first, last)` slice of the sorted list for each tile |
| `tile_rasterize` | one workgroup per 16x16 tile, one thread per pixel. Shared-memory batches, front-to-back blending, early exit on collapsed transmittance |
| `debug_visualize` | separate from the differentiable path; used only for the debug views |

`tile_rasterize` writes:

- `outputImage`: RGBA, alpha is coverage.
- `outputDepth`: R32F, the depth of the first contributor with alpha at least
  1/255. Used by the Depth view.
- In differentiable mode only: `pixelState` (final transmittance and the last
  contributor), and the RGBA float buffer `outputBuffer`. Both are written
  only when `SAVE_BACKWARD_STATE` is set.

### Submission model

- **One submit-and-wait per frame.** `forward_project` produces the instance
  total. `instance_dispatch` turns it into the sort's element counts and
  indirect dispatch arguments. The histogram, scatter, ranges and block scans
  run through `vkCmdDispatchIndirect`. No readback is needed before the sort.
- **Capacity overflow.** No useful bound exists on instances, because one
  Gaussian can cover every tile. The instance-sized buffers grow. If a frame's
  total exceeds the capacity, `instance_dispatch` zeroes the count, so all
  later dispatches do nothing. The host then grows the buffers (count plus
  25%, at least doubling) and records only the instance stages again.
  `forward_project` does not run again.
- **Grow-only buffers.** `projected`, `instanceOffsets` and the sort buffers
  never shrink. Kernels bound themselves by `camera.gaussianCount`, so the
  count can change every densify without reallocation.
- **`tileRanges` is cleared every frame** before `tile_ranges`. Tiles with no
  instances are never written.
- `lastInstanceCount()` is copied to a host-visible readback in the same
  submission.
- Every submission brackets its work with `Device::recordGlobalBarrier()`
  (section 10).

Measured on RTX 5070 Ti, Release, bonsai-7k.ply (1.16M Gaussians, 1280x720,
200-frame orbit, about 2.29M tile instances): `forward()` 33-139 ms before
the single-submission change, and **8.6-8.8 ms mean** after it (median 8.5 ms,
p95 about 11.7 ms). Per stage, cumulative: project 0.65 ms, prefix 0.03 ms,
binning 0.08 ms, sort 1.6 ms, ranges 0.06 ms, rasterize 6.4 ms. Rasterization
is the cost to attack next. The output is bit-identical to the earlier
two-submission version.

The earlier cost was not the host stall. It was the radix sort's digit-offset
scan, which ran in one 256-thread workgroup and took about 117 ms per frame.
The device-wide scan replaced it.

### Conventions

These are the conventions the kernels use. A mistake in any of them gives an
image that looks plausible but is wrong.

- Right-handed, +Y up. The camera looks down its local -Z. `depth = -z`.
- `u = fx*x/d + cx`, `v = cy - fy*y/d`. The flip in `v` also negates the
  second row of the projection Jacobian. Flipping one and not the other gives
  mis-oriented ellipses.
- Covariance dilation adds `0.25 I` to the 2D covariance. Alpha is clamped to
  `0.99`. Alpha below `1/255` is skipped.
- Pixel centres are at integer coordinates.
- Degree 0 colour: `rgb = sh0 * 0.28209479177387814 + 0.5`, clamped to [0, 1].
- SH is evaluated in world space. The view direction is
  `normalize(mean - cameraPosition)`. When the scene is flipped, the camera
  position is negated on the host, not the mean.

### Verification

`examples/forward_test` runs each stage on a small synthetic cloud and compares
the GPU output with an independent C++ reference. The 256x256 pixel-exact check
has a worst colour error of 1.19e-7 and a worst depth error of 0.

`examples/scene_diag` loads a real asset, places the camera as the viewer does,
and reports the distributions of values, radii, depth, alpha and overdraw.

Swapchain images are not read back.
Renderer-level checks read pixels back with `readImage()` instead.

---

## 8. Level of detail and culling

Every LOD knob is off by default. A caller that never sets them sees every
Gaussian rendered. Viewer-only knobs are disabled structurally in gradient
runs (section 9).

### 8.1 Screen-space size cull

`LODParams::minScreenRadius` culls a Gaussian whose projected radius is below
the threshold. The default is 0 (off), and the viewer exposes a slider.

Dilation puts a floor under every projected radius: `3 * sqrt(0.25) = 1.5 px`.
A threshold at or below that floor never fires. A threshold of 2 px culled
more than 95% of `racoonfamily.spz` (932,560 Gaussians), leaving a featureless
ball, so the default is off.

### 8.2 Tile early-out

`tile_rasterize` counts threads that have finished (`groupshared doneCount`).
The count is updated with `InterlockedAdd` at the batch barrier. When every
thread in the tile is done, the workgroup stops fetching batches.
`examples/forward_test` checks this through `batchesFetched`, a per-tile
counter that exists only for verification.

### 8.3 Geometry LOD (distance and density)

A Gaussian is dropped when a stable per-Gaussian hash falls below a drop
fraction. The fraction combines camera distance and the density of the
Gaussian's centre tile in the previous frame.

- The hash is seeded only by the Gaussian's index, never by frame or time. The
  surviving set is therefore nested as the thresholds move, so the image does
  not flicker.
- The density is read from `previousTileRanges`, the same `tileRanges` buffer
  already on the GPU. The round-1 projection runs before the clear, so the
  values are the previous frame's. The buffer is zeroed once on creation and
  resize.
- Parameters are in `LODParams` (binding 8): enabled, near and far distance,
  max drop distance and density, and the density reference.

### 8.4 Appearance LOD (adaptive SH degree)

The same two fractions drive an effective SH degree:

```text
degradeFraction = max(distanceFraction * appearanceMaxDropDistance,
                      densityFraction  * appearanceMaxDropDensity)
adaptiveDegree  = clamp(3 - floor(degradeFraction * 4), 0, 3)
effectiveDegree = min(camera.shDegree, adaptiveDegree)
```

The global cap can lower the adaptive degree but never raise it. Degree
steps down as pressure rises. It needs no hash, because the degree is a
monotonic function of pressure.

`appearanceFadeWidth` (0 to 1) fades the outgoing top band across the last
part of each degree step. Colour is then continuous in pressure. The fade
never reads extra coefficients and does not apply to DC. A cross-build dump
with the fade set to 0 is byte-identical to the hard-switch output.

`DebugView::SHDegree` shows the blend-weighted mean degree of each pixel. It
uses the same weights (`T_i * alpha_i`) as the colour, so it reports the
degree of the colour that is shown.

### 8.5 SH evaluation

Bands 1-3 are evaluated in `forward_common.slangh`. A degree cap is a prefix
read of the band-ascending coefficients, so lower degrees also read less
memory. Scenes render
with view-dependent colour, and their images differ from the DC-only output.

### 8.6 Occlusion culling: measured, not built

A Hi-Z style occlusion cull was measured with an exact oracle built from a
differentiable run's `pixelState` and `tileRanges`:

- Only **3.5%** of instances are cullable at the default framing (10% at half
  the distance).
- Only 1.9% of tiles ever terminate fully.
- The rasterizer already skips those batches. A perfect cull would save about
  0.06 ms per frame, before the margin needed for reprojection.

A wrong occlusion decision leaves a visible hole, so the cull is not built.

### Rasterizer checks

`examples/sh_lod_test` checks the appearance LOD:

- The fade is inert with the LOD off. A fade-0 adaptive degree d renders the
  same as a global cap d.
- On a 3x3 grid with energy in every band, dollied from 10 to 40 in 0.1 steps,
  hard switches jump 0.197 at a threshold. With fade 0.25, 0.5 and 1.0 the
  jump is 0.015, 0.013 and 0.013, below the dolly's own step. The LOD error
  step drops 23-45 times.
- The SHDegree output matches a C++ reference within 8e-7.

---

## 9. Backward pass and gradients

### Forward state

`ForwardPass::run(..., ForwardMode::Differentiable)` sets
`SAVE_BACKWARD_STATE`. `tile_rasterize` then writes `pixelState` (final
transmittance stored directly, not recovered from alpha, which loses about
three digits near the 1e-4 cut-off) and the colour buffer. Viewer runs write
none of these.

`evaluateSplat` and `computeProjectionGeometry` in `forward_common.slangh` are
shared by the forward and backward kernels. The forward output stayed
pixel-exact after the refactor.

### Kernels

- **`backward_rasterize`**: one workgroup per tile, back to front. Starts at
  the tile's largest contributor count. Each step divides `(1 - alpha)` out of
  transmittance and atomically adds nine floats per Gaussian into a 48-byte
  scratch record. The inputs are RGBA, because coverage is `1 - T_final`.
- **`backward_project`**: one thread per Gaussian, no atomics. It chains back
  through SH (including the view direction), the compensation factor, the
  conic inverse, the covariance products, the scale and rotation, and the
  projection. Culled Gaussians get exact zeros.

The gradients are exact derivatives of the forward model. Two deliberate
differences from the INRIA reference:

- Alpha clamped at 0.99 has **zero** gradient.
- In the frustum-clamped region, the depth term is kept. The x derivative is
  zero.

### Gradient views

`CanonicalGaussianGradientViews` holds the gradients for means, scales,
rotations, opacity, SH, and `screenMeans` (the screen-space gradient used for
densification). Each buffer has the stride of its parameter stream.

- Gradients are with respect to the **stored, activated** values: linear
  scale, opacity in [0, 1], the four quaternion components unnormalized, and
  raw SH.
- `backward()` **overwrites** the gradient buffers. It is a pure function. A
  trainer that wants a sum over views adds them itself.
- Atomics are plain fp32 `InterlockedAdd`, nine per (pixel, Gaussian) pair. No
  subgroup pre-reduction is used.

### Structural guarantees

- `ForwardPass` records each run (`ForwardRunRecord`: mode, completion,
  camera, parameter views). `BackwardPass` reads these from the record. It
  throws if the last run was not a completed differentiable run.
- A differentiable run binds an all-zero `LODParams` buffer, uploaded once. No
  geometry or appearance LOD, and no `minScreenRadius`, can reach a gradient
  run.
- The SH degree cap is allowed in gradient runs. Only the adaptive
  per-Gaussian degree is pinned off.
- A resize invalidates the run record.

### Opacity compensation

`CameraUniforms::renderFlags` bit 0 selects the **antialiased** (mip-splatting)
compensation. When it is set, `forward_project` multiplies alpha by
`sqrt(det(cov2d before dilation) / det(cov2d after dilation))`. The integrated
contribution is preserved. Sub-pixel Gaussians are attenuated and large ones
are not affected.

- `.spz` reads the flag from its header.
- `.ply` and `.splat` default to off, because vanilla 3DGS scenes are trained
  without it. `Renderer::setAntialiased()` and a viewer checkbox override it.
- On `racoonfamily.spz`, median stored opacity 0.808 becomes effective alpha
  0.021. The 99th percentile stays at 0.804.

Dilation is 0.25. The INRIA 3DGS reference and gsplat use 0.3. Public
pretrained scenes are therefore rendered with a slightly narrower low-pass
filter than they were trained with. Changing the value must be done in
`shaders/config.slangh` and in every trainer that matches it.

### Verification

`examples/backward_test` compares every gradient (472 scalars per run, with
compensation off and on, and SH cap 3 and 1) with **frozen-branch central
differences**. Discrete decisions (blend order, clamps, early exit, culls) are
frozen at the base point, so the difference measures a derivative and not a
jump. The worst error is 0.3% of the tolerance. Mutation tests with eight
deliberate gradient errors all fail. `examples/c_api_test` checks an opacity
gradient through the C ABI to 1e-6 relative.

---

## 10. Synchronization

This section is the contract a host must follow when it shares buffers with
Sabercat.

### One queue

Everything except `present()` is submitted to `Device::submitQueue()`. Forward,
backward, readback, `immediateSubmit`, and the relight, shadow and bake passes
all use it.

Every submission starts and ends with `Device::recordGlobalBarrier()`: all
writes before, all reads and writes after, all stages. The barriers cost a
pipeline drain at submission boundaries. Those boundaries were already
serialized by fence waits.

A consumer on the same queue therefore needs no wait and no barrier:

- Its earlier submissions are ordered before the first Sabercat command.
- Sabercat's fence wait covers work that finished before it returned.
- The trailing barrier covers later reads.

### Verification

Khronos synchronization validation is clean on the tests above. Removing a
leading barrier gives a `READ_AFTER_WRITE` report. Removing a trailing barrier is
not reported, because the fence wait retires the work. The barrier stays because
the memory model requires a visibility operation for device-to-device reads.

### Submission helper

`submitAndWait()` wraps each submission with the global barriers, the submit
queue and one persistent fence. `ForwardPass` uses it for its single per-frame
submission.

---

## 11. Borrowed buffers and ownership

### Borrowed scenes

`Renderer::borrowGaussians(BorrowedGaussians)` points a cloud at application-owned GPU buffers (`BufferView`, see `include/sabercat/borrowed.h`).
`validateBorrowedGaussians` checks each stream before the new cloud is built,
so a rejected scene leaves the old one unchanged:

- The buffer has the canonical stride.
- The offset is aligned to `minStorageBufferOffsetAlignment`.
- The range is at least one Gaussian and at most `maxStorageBufferRange`.
- `offset + size` fits in the buffer's memory requirements.
- `shDegree` and the count are in range.
- Gradients are all or none, and each covers the capacity.
- The used ranges do not overlap, so gradients cannot alias parameters.

Usage flags, sharing mode and owning device cannot be queried in Vulkan. The
host must guarantee them.

Borrowed clouds cannot grow. The owner resizes its own memory.

### Image size and sharing

- `setImageSize()` resizes the forward and debug passes. It is refused while a
  swapchain exists.
- `Renderer::vulkanContext()`, `enabledDeviceFeatures()` and `ownsDevice()`
  let one renderer borrow another renderer's device.
- `Device::adopt()` runs `validateBorrowedContext` first. It checks the
  handles, the queue families, the GPU floor, each declaration, and the
  swapchain declaration through `vkGetDeviceProcAddr`.

### Present gating

`attachSurface()` requires a swapchain-enabled device and a graphics queue.
`setUiRenderCallback()` requires dynamic rendering. Both throw with the
model-specific fix in the message.

### C ABI

The C ABI is in the `sabercat_c` DLL. It is described in [c-api.md](c-api.md).
Key points:

- Opaque renderer handle. Every call returns a result code.
- `sabercat_last_error()` is thread-local.
- No C++ exception crosses the boundary.
- Input structs begin with `struct_size`. Output is written only up to the
  caller's `struct_size`.
- `SABERCAT_C_API_VERSION` is 1. Later additions are appended.
- Views are passed as `uint64_t` handles for Vulkan objects.

---

## 12. Ray tracing core

Ray tracing needs `DeviceRequirements::rayTracing`. A device without it keeps
every other capability. `ShadowMode::RayTraced` then throws a named error, and
the viewer falls back to PCSS.

### Acceleration structure

`GaussianAccelerationStructure` (`gaussian_acceleration_structure.h`) builds a
BLAS of procedural AABBs and a single-instance TLAS over the canonical streams.
Primitive `i` is Gaussian `i`.

- `gaussian_aabbs` writes the exact box of each ellipsoid. The support radius
  is `gaussianSupportSigmas`, which depends on opacity: a ray can only be
  shaded by a Gaussian whose peak alpha reaches 1/255 inside the support. The
  bound keeps a 1e-3 relative margin over rounding, so the BVH still matches
  a brute-force sum.
- A Gaussian that can never contribute gets a collapsed box. Such Gaussians
  are: opacity below 1/255, a non-positive or huge scale, a zero quaternion,
  or a non-finite position.
- `build()` is a full rebuild. `refit()` updates in place for same-count
  changes. Refit quality decays as Gaussians move, so the structure is rebuilt
  periodically and after densification.
- Each call is one fenced submission on the submit queue.
- Box volume is 41% of the 3-sigma box on the XXL scene and 69% on bonsai.

### Transmittance query

`gaussianTransmittance(tlas, means, scales, rotations, opacities, origin, dir,
tMin, tMax)` (`gaussian_ray.slangh`) walks the TLAS with a `RayQuery`. It never
commits a hit. For each candidate:

1. Find the ray's point of maximum density in the Gaussian's whitened space:
   `t* = -o'.d' / d'.d'` and `d2min = |o' x d'|^2 / |d'|^2`.
2. Skip if `d2min > 9` (outside 3 sigma), if `t*` is outside `[tMin, tMax]`, or
   if the peak alpha `min(opacity * exp(-0.5 * d2min), 0.99)` is below 1/255.
3. Otherwise `T *= 1 - alpha`. Below 1e-4 the walk stops and returns 0.

The walk does not depend on order, so nothing is sorted. This is the peak
response model of the rasterizer: one alpha per Gaussian. A ray that grazes a
long Gaussian gets its peak alpha once. No covariance dilation or compensation
applies, because a secondary ray has no pixel footprint.

### Shader compilation

Slang's direct SPIR-V output declares `RayTracingKHR`, which needs the
`rayTracingPipeline` feature. Sabercat uses ray query only. The ray-query
kernels are therefore compiled with `-emit-spirv-via-glsl`
(`SABERCAT_RAY_QUERY_SLANGC_FLAGS`), which declares `RayQueryKHR` and SPIR-V
1.4 only.

### Verification

`examples/gaussian_raytrace_test` checks the core: AABBs against a CPU
computation (8e-8 relative), transmittance against a CPU brute force over all
Gaussians (worst error 5e-6 on a 20k-Gaussian bonsai subset, tolerance 1e-4),
refit against a fresh build (1.2e-7), and model-B against model-A results.

---

## 13. Relighting

Relighting needs relight streams. An R3DG `.ply` supplies them. Other inputs use
defaults.

### Relight streams

`PLYReader` reads R3DG attributes when `base_color_0` is present. It applies the
R3DG activations: `sigmoid * 0.77 + 0.03` for colour and `sigmoid * 0.9 + 0.09`
for roughness. Zero normals in a vanilla file are ignored.

### Passes

`RelightPass` (`relight_pass.h`) runs three kernels after a completed forward
run, each as one submission on the submit queue.

| Kernel | Work |
| --- | --- |
| `relight_prepare` | Per Gaussian, resolve normal, base colour, roughness, metallic and ambient visibility. Missing streams get defaults: the shortest ellipsoid axis as normal, the SH DC colour, roughness 0.8, metallic 0. Normals are flipped to face the camera. Ambient visibility is the baked visibility SH convolved with the cosine lobe, using Funk-Hecke weights (1, 2/3, 1/4, 0) |
| `gbuffer_rasterize` | Same sorted lists, same `evaluateSplat`, same early exits as the colour pass. Blends the per-Gaussian surface values as premultiplied sums. Coverage equals the colour alpha bit for bit |
| `deferred_lighting` | Normalizes by coverage. Rebuilds the asset-space position from expected depth. Lambert plus GGX (height-correlated Smith, Schlick Fresnel, metallic-roughness). Directional, point and rectangular area lights, with stratified samples for area lights. A degree-2 SH environment: cosine-convolved irradiance for diffuse, the reflected radiance for rough specular. Direct light is multiplied by the pixel's visibility layer |

`kMaxLights` (8) visibility layers are kept. Each layer is `width * height`
floats. Lights live in the scene's coordinate space.

### Renderer API

`setLightingEnabled`, `setLights`, `setEnvironment`, `setLightingSettings`,
`readLitImage`, `litImageView`, `visibilityView`, `readVisibility` and
`readGBuffer`. `LightingOutput` selects the value that `present()` shows one of:
`Lit`, `Normal`, `BaseColor`, `Roughness`, `Metallic`, `AmbientOcclusion`, `Visibility` or `Depth`.

Colour, `readImage()` and `imageView()` are unchanged by lighting.

### Verification

`examples/relight_test` checks that rendering a per-Gaussian colour equals the
G-buffer channel for each surface value (to 1.5e-8), and that every lighting
output matches a CPU port of the shader (3.8e-6 at roughness 0.3 or more). At
roughness 0.1 the GGX lobe amplifies rounding differences, so that case has a
looser, stated bound.

---

## 14. Shadows

`ShadowMode` selects the source for the visibility layers.

| Mode | Source | Cost | Use |
| --- | --- | --- | --- |
| `None` | every light fully visible | none | default |
| `PCSS` | splat shadow map with blocker search | low | any device |
| `RayTraced` | ray-traced transmittance through the Gaussians | high | reference, needs ray query |
| `BakedSH` | per-Gaussian visibility SH (section 15) | low | indirect and environment-like terms |
| `External` | `setExternalVisibility()` buffer | host-defined | learned or imported visibility |

Each light has `castsShadows`. Visibility layers are cleared before the passes
write, so a light that does not cast shadows never keeps a value from an
earlier frame.

### PCSS

`ShadowPass::runPCSS`:

1. A private `ForwardPass` renders the splats from the light's point of view.
   The camera frames the focus sphere. A point light looks from its position.
   A directional light looks from three radii away.
2. `shadow_map` reduces the view to the **median-opacity depth**: the depth where
   transmittance first falls to 0.5. The front-most contributing depth would
   make faint haze cast hard shadows.
3. `shadow_pcss` finds the blockers, estimates the penumbra width by similar
   triangles, and filters with PCF. Receivers come from the G-buffer, offset
   along their normal.

Each map comparison uses the receiver's plane in the light's view (its depth follows the G-buffer normal's slope), so a tilted receiver does not find its own surface as a blocker. A slope-scaled bias (8 texels of depth per unit tan(theta), added to `depthBias`) covers the median-depth map's splat-centre error on tilted surfaces, and blocker gaps are measured against that plane.

PCSS is a heuristic. It is graded against the ray-traced reference.

### Ray-traced shadows

`ShadowPass::runRayTraced` (`shadow_rt.slang`). For each receiver pixel and
each light, the visibility is the mean transmittance over an N x N stratified
grid on the emitter: a disk for point lights, the rectangle for area lights,
a cone for directional lights. The grid is fixed, so the result is
deterministic. The ray starts at the G-buffer position, offset along the
normal. A receiver does not shadow itself, because its own surface Gaussians
peak behind the offset origin, which gives `t* <= 0`.

The acceleration structure is shared (section 12). It is rebuilt only when the
scene has changed.

### Shadow performance

Scene: "cluster fly XXL", 3.51M Gaussians, 87% with opacity below 0.1. Frame:
1280x720, 447k covered pixels, two shadow-casting lights, RTX 5070 Ti, 4x4
rays per light.

- Before the optimizations below, a full frame took 2.7 s. Each light rebuilt
  the structure (about 9 ms) and traced at 5-7 Mrays/s. Bonsai traced at 35-57
  Mrays/s.
- The Renderer keeps one shared structure. It is rebuilt once per frame, and
  not at all while a tracked scene is unchanged.
- A frame cache restores PCSS, BakedSH and full-grid ray-traced results when
  the inputs are unchanged. The inputs are hashed: scene and bake generations,
  LOD state, camera uniforms, shadow settings and lights.
- Progressive accumulation (`progressiveSamplesPerFrame`) adds k emitter samples
  per frame. After N^2 samples the result is bit-identical to the full grid.
  The viewer uses k = 1.

| Frame | Time |
| --- | --- |
| Full frame | about 1.3-1.5 s (from 2.7 s) |
| Unchanged frame | 43 ms (a lit frame with no shadows: 42 ms) |
| Progressive | 95-120 ms per frame for 16 frames, then 43 ms |

PCSS on unchanged frames went from 75 ms to 42 ms.

#### Exact and approximate settings

The defaults are exact. Three `ShadowSettings` values trade accuracy for speed.
They apply to RayTraced shadows only. The bake and the teacher samples always
trace the exact model.

- `rayMinTransmittance`: a ray stops when it is darker than this. The per-pixel
  error is below the value.
- `rayMinAlpha`: Gaussians whose peak alpha along the ray is below this are
  skipped. Boxes are built for the same cut-off, so faint fog is left out of
  the structure. Shadows can only get lighter.
- `rayDownsample`: one receiver is traced per s x s block. The other pixels
  are filled by a joint-bilateral upsample (`shadow_upsample.slang`) using
  depth and normal similarity. Traced pixels keep their exact values.

Measured on XXL, 4x4 rays, two lights, against the exact frame:

| Setting | Speedup | mean abs dV | max abs dV |
| --- | --- | --- | --- |
| rayMinTransmittance 0.01 | 1.3x | 0.0004 | 0.010 |
| rayMinTransmittance 0.05 | 1.5x | 0.0027 | 0.049 |
| rayMinAlpha 0.01 | 2.1x | 0.026 | 0.30 |
| rayMinAlpha 0.02 | 4.0x | 0.073 | 0.58 |
| rayMinAlpha 0.05 | 12.7x | 0.23 | 0.88 |
| rayDownsample 2 | 2.7x | 0.036 | about 1 (thin features) |
| Balanced: 0.01 + downsample 2 | 2.7x | 0.036 | about 1 |
| Fast: 0.01 + alpha 0.02 + downsample 2 | 6.8-8.6x | 0.085 | about 1 |

Skipping faint splats is the largest lever. In fog-heavy captures the faint
splats add up to real shadow, so the error grows. Half resolution is the best
trade, except for thin structures narrower than a block. The viewer offers
Exact, Balanced and Fast.

Not done: temporal reprojection while the camera moves.

### Shadow measurements

`examples/shadow_test` checks the shadow ring, the umbra, and penumbra growth
with light size and occluder height. The ray-traced penumbra widths are
0.11 / 0.21 / 0.50 as the emitter grows. Mean error against a 32x32 reference
is 0.0038 / 0.0007 / 0.0002 / 0.0000 at 2 / 4 / 8 / 16 samples per axis. PCSS is
within a mean 0.018 of the reference, with narrower penumbras.

On bonsai (1.16M Gaussians, sun light, 96x96), PCSS takes 70 ms and ray-traced
54 ms for the whole lit frame. The two modes differ by a mean visibility of
0.180, so PCSS's bias depends on the scene. The acceleration structure builds in
3-7 ms and refits in about 1 ms, at 35-57 Mrays/s.

Measured and rejected: **oriented octahedron proxies** (8 triangles per
Gaussian). They were exact but 2.4x slower to trace on XXL, 5x slower on bonsai,
and 3-10x slower to build.

NVIDIA extensions on the tested NVIDIA driver are not used. Invocation reorder applies only
to ray-tracing pipelines. Mega geometry speeds up builds that the shared
structure already avoids. Linear swept spheres do not model ellipsoids. Opacity
micromaps apply only to triangles. Cooperative vectors are the candidate for a
learned visibility network.

## 15. Baked visibility and teacher samples

`VisibilityPass` (`visibility_pass.h`) does the baking and the sampling. It is
created lazily, so a missing shader never leaks a command pool. It owns its own
acceleration structure, and each call is one synchronous submission.

### Bake

`Renderer::bakeVisibility()` runs `bake_visibility.slang`:

- One thread per Gaussian traces N directions (default 256, a spherical
  Fibonacci set) through every other Gaussian. The skip-index variant
  `gaussianTransmittanceSkipping` is used.
- The result is projected onto 16 mono SH coefficients:
  `c_k = 4*pi/N * sum_j V(w_j) * Y_k(w_j)`, in the 3DGS sign convention.
- Full visibility bakes `c0 = sqrt(4*pi)`, so the cosine convolution gives
  ambient 1.
- Each ray starts where it leaves the Gaussian's own 3-sigma ellipsoid, not at
  the mean. From the mean, a ray leaving a thin surfel crosses its coplanar
  neighbours at `t* ~ 0+` and takes their full peak alpha. The fixture's open
  ground baked an ambient of 0.08 that way, and 0.965 after the fix.
- The output goes to the `visibility` relight stream. An owned cloud creates it.
  A borrowed cloud must already have one.
- **Device timeout guard.** The work is split into fenced submissions sized
  to a time budget (`targetSubmitMilliseconds`, default 100 ms): each batch
  grows or shrinks by at most 2x from the previous submission's host time,
  between a floor of 1M rays and the cap `maxRaysPerSubmit` (default 8M).
  The floor exists because one thread traces all of a Gaussian's directions
  in sequence, so a submission's time never drops below that serial latency.
  On a dense 139k-Gaussian scene, 3 Gaussians took about 95 ms and 256 took
  about 250 ms, and fixed 8M-ray batches tripped the Windows GPU timeout. That
  scene now bakes in about 21 s as 35 submissions. `targetSubmitMilliseconds
  = 0` restores fixed batches of the cap. Results do not depend on the
  batching. Bonsai (1.16M Gaussians x 256 directions) baked in 9.9 s as 36
  fixed submissions of about 276 ms each, about 30 Mrays/s on RTX 5070 Ti.

### Teacher samples

`Renderer::generateTeacherSamples()` produces training pairs of position,
normal, light and target visibility.

- **Candidates** (`sample_select.slang`) come from the last lit frame's
  G-buffer. Smooth pixels are covered with coverage at least 0.5. Edge pixels are
  covered and have a 4-neighbour that is uncovered, more than 5% away in depth,
  or more than about 26 degrees away in normal.
- **Compaction** runs in three phases: per-row counts, one-thread scan, per-row
  scatter. The lists come out in pixel order with no atomic appends.
- **Drawing** (`teacher.slang`): 6144 samples from the smooth list and 2048 from
  the edge list (75/25 by default, both tunable). Each set is stratified: the
  list is cut into equal strata, with one jittered pick per stratum, seeded per
  call. Smooth samples come first.
- **Lights** are assigned round-robin over the shadow-casting lights.
- **Targets** come from `lightVisibility` (`light_visibility.slangh`), which the
  RayTraced shadow pass also uses. A target therefore equals the RayTraced
  shadow value at that pixel, bit for bit.
- **Output**: `VisibilityTrainingSample` (32 bytes: position, target, normal,
  light index), exposed as `teacherSampleView()` and `readTeacherSamples()`.

### BakedSH shadows

`visibility_splat.slang` evaluates each Gaussian's SH toward each light. A
Hanning window (1, 0.854, 0.5, 0.146 per band) reduces ringing. The result is
splatted over the sorted lists like the G-buffer.

Measured limitation: against the ray-traced reference, BakedSH has a mean
absolute error of 0.188 on the fixture. That is worse than no shadows (0.145).
Lit ground beside the slab reads about 0.8 toward a light it sees clearly,
because degree 3 spreads the nearby slab's solid angle over too many
directions. This follows from the band limit. BakedSH is accurate for the
environment term. For direct shadows use RayTraced, PCSS, or an `External`
source.

### Debug views

`ShadowSettings::debug` selects a view, shown through `LightingOutput::Visibility`:

- `ErrorVsReference`: absolute difference between the source and
  RayTraced at `referenceSamplesPerAxis`.
- `SampleLocations`: 0.5 at smooth samples and 1 at edge samples, in each
  sample's light layer.

### Verification

`examples/visibility_test` checks that baked coefficients match a CPU
brute-force bake within 1.5e-5 (where a ray's decision is near a threshold, the
CPU reference is an interval). A split bake equals a single bake bit for bit.
BakedSH equals the colour pass splatting the same values to 1.8e-7. Teacher
targets equal the ray-traced shadow bit for bit, and sample positions match the
G-buffer reconstruction to 3.8e-7. Sampling is deterministic per seed.

### Build note: a Slang code generation issue

Through `-emit-spirv-via-glsl`, the expression
`samplePixels[outIndex] = pixel | (edge ? 0x80000000u : 0u)` reused the
variable of `outIndex` as the ternary's temporary. Every write went to slot 0
or to bit 31. `teacher.slang` uses `uint(edge) << 31u` instead. When a
ray-query kernel misbehaves, check the GLSL that `slangc -target glsl`
produces.

---

## 16. Studio

Studio (`apps/studio`, target `sabercat_studio`) edits layers and saves
projects. It exports PLY and GLB.

Studio is always an owner of its device (model B). It creates the instance and
device through `BorrowedVulkanHost` (`apps/common`) and calls
`Renderer::initializeBorrowed()`. The viewer uses model A by default and model
B with `--borrowed`.

### Components

| Piece | Where | Role |
| --- | --- | --- |
| `ProjectIO` | `project.h`, `src/renderer/project.cpp` | Saves and loads `.sabercat` projects. Save writes a temporary file and renames it over the target |
| `CloudV1Writer` | `cloud_v1_writer.h` | Exact inverse of `CloudV1Reader` |
| `RelightSidecar` | `cloud_v1_writer.h` | Encodes and decodes relight streams as `sabercat.relight-v1` |
| `FileMap` | `file_map.h` | In-memory name-to-bytes map, shared by the cloud-v1 code and the ZIP code |
| `zip::` (internal) | `src/renderer/zip_archive.{h,cpp}` | STORED writer. Reads STORED and DEFLATE entries and checks CRC. Not installed |
| `PLYWriter` | `ply_writer.h` | Exact inverse of `PLYReader`. Writes R3DG attributes when relight streams exist |
| `GLTFWriter` | `gltf_writer.h` | `KHR_gaussian_splatting` as one `.glb`, one mesh and one node per layer |
| `LayerTransform`, `transformedCopy`, `mergeLayers` | `layer_transform.h` | Rigid placement with uniform scale, and a host-side merge of the visible layers |
| `sh_rotation` (internal) | `src/renderer/sh_rotation.{h,cpp}` | Rotates SH bands 1-3 for a layer rotation |
| `apps/common` | `apps/common/` | Shared app code: window and ImGui, borrowed host, orbit camera, relight panel, asset scan, about panel |

### Project layout

A `.sabercat` file is a ZIP of STORED entries, so float data is not compressed:

```text
project.json                 format "sabercat.project", formatVersion 1, layers, app state
layers/<id>/                 one cloud-v1 directory per layer
layers/<id>/relight/         relight streams (sabercat.relight-v1), only if the layer has any
```

- `<id>` is 1 to 64 characters from `[A-Za-z0-9_-]`, unique in the project.
- `project.json` stores each layer's name, transform, visible and locked flags,
  and an opaque `app` object (the viewer's camera and settings) that `ProjectIO`
  stores verbatim.
- A layer directory is a plain cloud-v1 fixture, so other tools can read it.
- Not written: `sources/`, `thumbnails/`, lights, bake state, undo history and
  selection.

### Relight sidecar (`sabercat.relight-v1`)

`layers/<id>/relight/manifest.json` holds `schema`, `count` and a `streams`
object. Each stream is optional and uses the cloud-v1 entry keys:

| Stream | Shape |
| --- | --- |
| `normals` | `[N, 4]` |
| `base_color_roughness` | `[N, 4]` |
| `metallic` | `[N]` |
| `visibility` | `[N, 16]` |

The sidecar is separate from the cloud-v1 directory, so that directory keeps
the base format. Readers may ignore the sidecar. A relight stream survives a
merge only when every layer has it.

### Exactness

Measured round trips (Release):

| Path | Result |
| --- | --- |
| Project save, then load | Bit-exact: every host stream, `shDegree`, the antialiased flag, transform floats, flags, ids, names |
| GLB, identity layers | Bit-exact |
| GLB, placed layers, against `mergeLayers` | Worst error 0 |
| PLY positions, SH, visibility, metallic | Exact |
| PLY rotations | Exact up to ulps |
| PLY scale | Relative error at most about 5e-7 (log and exp) |
| PLY opacity | Absolute error at most 6e-8 (logit and sigmoid). Clamped to [1e-6, 1 - 1e-6] |

Design choices:

- Layer transforms use **uniform scale only**. Means move with the full matrix,
  rotations compose as quaternions, and SH bands 1-3 rotate with the rotation.
  That keeps every per-Gaussian value exact. A non-uniform placement would need
  every covariance re-diagonalized.
- glTF colour space is `srgb_rec709_display`. 3DGS SH is trained against sRGB
  photographs, so the values are display-referred.
- glTF sets `extensionsUsed` only, not `extensionsRequired`. A loader that does
  not know the extension falls back to the base point cloud. Relight streams
  are not exported to glTF.
- A PLY with normals but no base colour gets zero normals, because `PLYReader`
  reads normals only when base colour is present.

### Export and save performance

Release, cluster "fly" XXL, 3,506,799 Gaussians, one layer:

| Operation | Time | File size |
| --- | --- | --- |
| Project save | 1.01 s | 869.7 MB |
| Project load | 1.23 s | |
| PLY export | 0.63 s | 869.7 MB |
| GLB export | 1.29 s | 827.6 MB |
| Scene rebuild (`mergeLayers` plus `setGaussians`) | 765 ms | |

### Limitations

- When every layer is hidden the viewport is empty (internally Studio
  renders a single fully transparent Gaussian, because the renderer has no clear-scene call).
- Loading reads the whole archive into memory.
- No zip64. An archive over 4 GiB or with more than 65534 entries is refused
  with an error.
- The apps use the native Windows file dialogs, with a typed path as the
  fallback when a dialog cannot be shown.
- Layer merge runs on the host.

### Studio command line

- `sabercat_studio [--validation | --no-validation] [files...]`
- `sabercat_studio --selftest <outDir> [<scenes...>]` runs headless and writes
  its results to `outDir`; with no scenes it uses a synthetic project.

---

## 17. Verification

### Test programs

The programs in `examples/*_test` are CTest tests, registered in
`tests/integration/`. They need a GPU:

```text
ctest --test-dir build -C Release --output-on-failure
```

Continuous integration builds the project only. It does not run the GPU tests.

Tests use independent references: a CPU reimplementation of the same math,
frozen-branch finite differences, bit-exact comparisons where the arithmetic is
identical, and synthetic fixtures created at run time. Each feature has a
**degeneracy guard**, a check that fails when the feature does nothing. For
example, a sub-pixel Gaussian must lose alpha, and the tile early-out must fire.

### Validation

- The Vulkan validation layer is on in Debug builds and with `--validation`,
  and off with `--no-validation`. If the layer is not installed, the program
  runs without it (`apps/common/app_util.cpp`, `resolveValidation`).
- Synchronization validation is used for the shared-queue contract (section 10).

Per-subsystem numbers are in the sections above. A summary is in
[PERFORMANCE.md](../PERFORMANCE.md).

---

## 18. Development notes

- Slang compiles the kernels at build time with `slangc`. The runtime loads the
  SPIR-V files (`SABERCAT_SHADER_DIR`).
- Any change to a header in `shaders/` rebuilds every kernel that includes it.
- The shared constants are in `shaders/config.slangh`. `render_config.h` exports
  them to the host.
- After a `Device` change, run the headless tests as well as the viewer. A
  change that only the windowed app exercises can break validation for every
  headless caller.
- After a change to a `Renderer` member that owns GPU memory, check the
  shutdown order (section 3).
