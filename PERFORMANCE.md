# Performance

This document lists the design decisions that shape Sabercat's performance.
For each one it states what the code does now. Where the code does not follow
a rule yet, the entry says so.

The hot paths are the forward rasterizer, the radix sort and the backward
gradient pass. They are GPU compute bound. The module-level details are in
[docs/architecture.md](docs/architecture.md).

---

## Design rules

### 1. Struct-of-arrays data

**Rule.** Large datasets are stored as one buffer or vector per attribute. Both
GPU streams and host arrays follow this rule.

**Status: implemented.** `GaussianCloud` keeps parallel vectors (positions,
scales, rotations, SH, opacities, optional relight streams). Each GPU stream is
its own buffer, and each kernel binds one storage buffer per stream.

### 2. Forward and backward are pure operations

**Rule.** `forward()` and `backward()` read and write GPU buffers. They keep no
training state, and the library contains no loss, optimizer or densification.

**Status: implemented.** `Renderer::forward()` and `Renderer::backward()` take
buffers or handles. The C ABI (`sabercat_c`) exposes the same operations.

### 3. Per-frame resources are duplicated per frame in flight

**Rule.** Command buffers, semaphores and fences are duplicated so the CPU can
record one frame while the GPU runs the previous one.

**Status: implemented for the window path.** `Renderer` keeps two
`FrameContext` entries, each with a command pool, a command buffer, a pair of
semaphores and a fence. The debug visualizer keeps one output image per frame
in flight. `ForwardPass` submits and waits inside each call, so it does not
need duplicating.

### 4. Command recording on multiple threads

**Rule.** Command pools are not shared between threads. Large passes record
in parallel.

**Status: not implemented.** Recording happens on the calling thread, from one
pool per frame in flight. There are no secondary command buffers and no job
system. CPU parallelism is in the loaders and in the layer transform, through
`parallelFor()` (`include/sabercat/parallel_for.h`).

### 5. Async compute on a separate queue

**Rule.** Independent GPU work runs on a separate queue. Synchronization uses
timeline semaphores.

**Status: partly implemented.**

- Every compute pass runs on `Device::submitQueue()`, which is a dedicated
  compute queue when the GPU has one. Only presentation uses the graphics
  queue.
- Synchronization uses fences for host waits and binary semaphores for the
  swapchain. Timeline semaphores are not used.
- Work on the compute queue is ordered by global barriers, not by semaphores
  (see architecture.md, section 10).

### 6. Capability detection and gating

**Rule.** The device reports the performance-relevant features it has. Kernels
and host code choose their path from those reports. Cooperative matrix and
vector support is detected for later use. Calls to Vulkan stay behind a thin
seam, so another graphics API could be added later.

**Status: partly implemented.** `DeviceFeatures` reports subgroup size and
operations, the 16-bit and int8 features, buffer device address, and the
cooperative extensions. The cooperative extensions are detected only.
There is no RHI seam. The passes call Vulkan directly.

### 7. Precision and atomics

**Rule.** Choose the storage precision for each buffer. Gradient accumulation
uses fp32 atomics.

**Status: partly implemented.** All stored values are fp32. Gradient
accumulation uses plain fp32 `InterlockedAdd`, nine per (pixel, Gaussian)
pair. The fp16 features are enabled when supported, but no kernel uses fp16
storage.

### 8. Reserved temporal dimension

**Rule.** The Gaussian schema reserves room for per-Gaussian time, so 4D
Gaussians can be added without changing buffer layouts.

**Status: partly implemented.** `GaussianCloud::times()` is reserved. It is
not uploaded to the GPU, and no kernel reads time.

### 9. Central launch configuration

**Rule.** Tile size, workgroup size, radix digit width and similar constants
are defined once, for both host and shader code.

**Status: implemented.** `shaders/config.slangh` holds the constants and
`include/sabercat/render_config.h` exports them to C++. Stream strides are
defined there as well.

### 10. Memory

**Rule.** Allocations are sub-allocated from pooled blocks. Streaming uploads
reuse a persistent staging ring. Buffer device addresses are enabled.

**Status: partly implemented.**

- VMA sub-allocates every buffer and image (`Device::allocator()`).
- There is no persistent staging ring. `Buffer::upload()` creates a temporary
  staging buffer for each call. Uploads happen at load time or when a parameter
  changes, not per frame.
- Buffer device address is enabled when the GPU supports it. Ray tracing requires it. Bindless
  descriptors are not used.

### 11. Borrowed device and buffers

**Rule.** Passes take buffer views and a context view. They never take
ownership. A host can own the device and the Gaussian buffers and lend them to
the renderer.

**Status: implemented.** `DeviceRequirements` and `Device::adopt()` support
the borrowed model. `Renderer::borrowGaussians()` takes host buffers. All
passes bind views with their offsets. Only `Device` calls `vkCreateInstance`
and `vkCreateDevice`.

---

## Priority

| Lever | Impact here | Status |
| --- | --- | --- |
| Subgroup ops and shared memory in the sort and tile blending | High | Shared memory used. Subgroup operations are not used yet |
| Struct-of-arrays with coalesced access | High | Implemented |
| Frames in flight and async compute | Medium | Frames in flight implemented. Compute queue used for all work |
| CPU job system and per-thread pools | Medium | Not implemented. Loaders use `parallelFor()` |
| fp16 storage and fp32 atomic gradients | Medium | fp32 atomics implemented. fp16 not used |
| Cooperative vector and matrix | Low for the core | Detected only |
| CPU SIMD for parsing | Low to medium | SSE2 quaternion normalization. The rest is scalar |

Design rules 1 to 6 and 11 are the ones that force a rewrite when they are
missing. Rules 7 to 10 are cheap to add later.

---

## Measured results

Hardware: NVIDIA RTX 5070 Ti, Windows, Release build unless stated. The
sections of [docs/architecture.md](docs/architecture.md) give the method and
the context for each number.

| Operation | Scene | Result |
| --- | --- | --- |
| Forward pass, 1280x720 | bonsai-7k.ply, 1.16M Gaussians | 8.6-8.8 ms mean (median 8.5 ms, p95 about 11.7 ms); rasterization 6.4 ms of that |
| Forward pass before the single-submission change | same | 33-139 ms |
| Radix digit-offset scan before the change | same | about 117 ms per frame |
| PLY parse | 300k synthetic Gaussians | about 33 ms |
| PLY upload | 300k synthetic Gaussians | about 29 ms |
| Lit frame, PCSS and ray-traced shadows | bonsai-7k.ply, 96x96 | 70 ms and 54 ms |
| Ray-traced shadows, full frame | cluster XXL, 3.51M Gaussians, two lights, 4x4 rays | about 1.3-1.5 s (was 2.7 s) |
| Ray-traced shadows, unchanged frame | same | 43 ms |
| Ray-traced shadows, progressive | same | 95-120 ms per frame for 16 frames |
| Ray-traced shadows, fast setting | same | 6.8-8.6 times faster, mean error 0.085 |
| Acceleration structure build and refit | bonsai, 1M Gaussians | build 3-7 ms, refit about 1 ms |
| Ray throughput | bonsai; XXL | 35-57 Mrays/s on bonsai; 5-7 Mrays/s on XXL before the structure reuse and frame cache |
| Baked visibility | bonsai-7k.ply, 256 directions | 9.9 s, 36 submissions of about 276 ms |
| Project save and load | cluster XXL | 1.01 s and 1.23 s |
| PLY export | cluster XXL | 0.63 s |
| GLB export | cluster XXL | 1.29 s |
| Scene rebuild from layers | cluster XXL | 765 ms |

Measured and not adopted:

- An occlusion cull based on the previous frame. Only 3.5% of instances are
  cullable, so the gain is about 0.06 ms per frame.
- Oriented octahedron proxies for ray-traced shadows. They were 2.4 to 5 times
  slower to trace and 3 to 10 times slower to build.
