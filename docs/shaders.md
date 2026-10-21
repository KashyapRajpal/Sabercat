# Shaders

All GPU kernels are written in Slang. `slangc` compiles each `.slang` file to
SPIR-V at build time. The runtime loads the SPIR-V from `SABERCAT_SHADER_DIR`.
No Slang compiler is linked into the library.

Sources are in `shaders/`. The build rules are in `shaders/CMakeLists.txt`.
Each kernel is a compute shader with entry point `main`. Most kernels take
their parameters as push constants (`[vk::push_constant] ConstantBuffer<T>`).

See [architecture.md](architecture.md) for how the passes are chained.

---

## Build

| Kind | Command (per file) | Output |
| --- | --- | --- |
| Standard kernel | `slangc <file> -target spirv -fvk-use-entrypoint-name -stage compute -entry main` | `<name>.comp.spv` |
| Ray-query kernel | same, plus `-emit-spirv-via-glsl` (`SABERCAT_RAY_QUERY_SLANGC_FLAGS`) | `<name>.comp.spv` |

Why the ray-query kernels use the GLSL route: Slang's direct SPIR-V output
declares `RayTracingKHR`, which needs the `rayTracingPipeline` feature. Sabercat
enables ray query only. The GLSL route declares `RayQueryKHR` and SPIR-V 1.4.

Shared headers are listed in `SABERCAT_SHADER_HEADERS`. A change to any of them
rebuilds every kernel.

| Header | Contents |
| --- | --- |
| `config.slangh` | Tile and workgroup sizes, radix width, SH degree and constants, stream strides, rasterizer constants, LOD defaults, relight limits |
| `forward_common.slangh` | Projection geometry, `evaluateSplat`, SH evaluation, wang hash, instance dispatch struct |
| `relight_common.slangh` | Lights, surfel layout, BRDF, SH basis |
| `gaussian_ray.slangh` | Transmittance walk through the Gaussians, with optional skipping and cut-offs |
| `light_visibility.slangh` | Emitter sampling and per-light visibility, shared by ray-traced shadows and teacher samples |

---

## Forward rasterizer

Dispatch order is in [architecture.md](architecture.md), section 7.

| Kernel | Purpose | Main inputs | Main outputs |
| --- | --- | --- | --- |
| `forward_project` | Project each Gaussian to screen space: camera transform, covariance, conic, SH colour, footprint, LOD and culling | canonical streams, `LODParams`, previous tile ranges | projected records, tile counts, instance total, depth and colour per Gaussian, SH degree per Gaussian |
| `instance_dispatch` | One thread. Turns the instance total into sort sizes and indirect dispatch arguments | instance total | `InstanceDispatch` buffer |
| `prefix_sum` | Three-phase exclusive scan over 2048-element blocks. Used for tile offsets and the per-digit offsets of the sort | tile counts, digit counts | offsets |
| `tile_binning` | One `(tile, depth)` key per overlapped tile | projected records, offsets | keys and values |
| `radix_sort_histogram` | Per-workgroup digit histogram, 8-bit digits | keys | digit counts |
| `radix_sort_scatter` | Stable scatter of keys and values for one digit pass | keys, values, offsets | reordered keys and values |
| `tile_ranges` | `[first, last)` slice of the sorted list for each tile | sorted keys | tile ranges (cleared each frame) |
| `tile_rasterize` | One workgroup per 16x16 tile. Shared-memory batches, front-to-back blending, early exit | sorted list, tile ranges, projected records | colour image (RGBA), depth image (R32F), `pixelState`, `outputBuffer` (the last two only in differentiable mode) |

Notes:

- The radix sort has 8 passes over a 64-bit key. Each pass uses a
  histogram, a `prefix_sum` for the offsets, and a scatter.
- `pixelState` holds the final transmittance and the last contributor count per
  pixel. It is written only when `SAVE_BACKWARD_STATE` is set.
- `outputBuffer` is the colour image as a row-major float4 array. It lets a
  trainer read the forward output without a layout transition.

---

## Backward pass

| Kernel | Purpose | Main inputs | Main outputs |
| --- | --- | --- | --- |
| `backward_rasterize` | Differentiates the tile blend. One workgroup per tile, back to front | sorted list, tile ranges, `pixelState`, dL/dImage (RGBA) | per-Gaussian gradients of centre, conic, opacity and colour, fp32 atomics |
| `backward_project` | Chains the screen-space gradients back to the stored parameters. One thread per Gaussian, no atomics | projected records, gradients from `backward_rasterize`, parameters | dL/d(mean, scale, rotation, opacity, SH), dL/d(screen mean) |

Both kernels use the same `evaluateSplat` and `computeProjectionGeometry`
helpers as the forward pass.

The optimizer and densification are not in Sabercat. The library exposes the
gradients. The caller applies its own update.

---

## Presentation and debug views

| Kernel | Purpose |
| --- | --- |
| `debug_visualize` | Writes the Alpha, Depth, TileDensity and SHDegree views as RGBA images. Used by `DebugVisualizer` only |
| `visibility_debug` | Writes the visibility error view and the sample-location view (`LightingOutput::Visibility`) |

---

## Relighting

| Kernel | Purpose |
| --- | --- |
| `relight_prepare` | Per Gaussian: resolves the normal, base colour, roughness, metallic and ambient visibility. Uses the relight streams or defaults |
| `gbuffer_rasterize` | Splats the per-Gaussian surface values over the sorted lists, as premultiplied sums. Coverage equals the colour alpha |
| `deferred_lighting` | Lambert and GGX for directional, point and rectangular lights, plus a degree-2 SH environment. Multiplies direct light by the visibility layer of each light |

---

## Shadows

| Kernel | Purpose |
| --- | --- |
| `shadow_map` | Reduces a light's splat view to the median-opacity depth (PCSS) |
| `shadow_pcss` | Blocker search, penumbra estimate and PCF, writes the light's visibility layer |
| `gaussian_aabbs` | Writes one axis-aligned box per Gaussian for the acceleration structure. Boxes are sized by opacity and skipped for Gaussians that cannot contribute |
| `shadow_rt` (ray query) | Ray-traced visibility through the Gaussians, one light per dispatch, with progressive accumulation |
| `shadow_upsample` | Joint-bilateral fill of pixels not traced in `rayDownsample` mode |
| `visibility_splat` | `ShadowMode::BakedSH`: evaluates each Gaussian's visibility SH toward each light, with a Hanning window, and splats it |

---

## Baked visibility and teacher samples

| Kernel | Purpose |
| --- | --- |
| `bake_visibility` (ray query) | Per Gaussian: traces a fixed set of directions through the other Gaussians and projects the result onto 16 SH coefficients. Dispatched in chunks (`maxRaysPerSubmit`) |
| `sample_select` | Chooses smooth and edge candidate pixels from the G-buffer and compacts them in pixel order (three phases) |
| `teacher` (ray query) | Draws stratified samples (75/25 smooth and edge by default). Each target is the ray-traced shadow value at that pixel |

---

## Known code generation issue

Through `-emit-spirv-via-glsl`, a ternary used as an operand of `|` inside an
array store reused the index variable as a temporary. The result wrote to the
wrong slot. `teacher.slang` therefore writes `uint(edge) << 31u` instead of a
ternary. When a ray-query kernel gives wrong results, check the GLSL that
`slangc -target glsl` produces.

---

## Kernel list

All kernels in `shaders/CMakeLists.txt`, in build order:

- Standard: `forward_project`, `prefix_sum`, `tile_binning`, `instance_dispatch`,
  `radix_sort_histogram`, `radix_sort_scatter`, `tile_ranges`, `tile_rasterize`,
  `debug_visualize`, `backward_rasterize`, `backward_project`,
  `relight_prepare`, `gbuffer_rasterize`, `deferred_lighting`, `shadow_map`,
  `shadow_pcss`, `gaussian_aabbs`, `sample_select`, `visibility_splat`,
  `visibility_debug`, `shadow_upsample`.
- Ray query: `shadow_rt`, `bake_visibility`, `teacher`.

The SPIR-V files are installed to `share/sabercat/shaders` in an installed
package.
