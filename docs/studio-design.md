# Sabercat Studio: design

**Status:** S1 is implemented and ships in 1.0.0. S1 is the application shell,
project files and export (section 5). Milestones S2 to S5 are design only, and
ROADMAP.md lists them. The parts that S1 reuses are the renderer, the relight
and shadow passes, the loaders and the borrowed-device model. See
[docs/library.md](library.md) for the device contract.

**Goal.** One artist tool: drop in a photo or a capture, get interactive
Gaussian splats, then relight, edit and export them, without leaving the app.
Sabercat renders, lights, ray-traces, edits and exports. An optional
trainer or generator plugin does the work that needs an optimizer or network
weights: training, refinement, material and visibility learning, and
single-image inference.

**Non-goals for 1.x:** 4D or animated splats (`.splatv`), mesh authoring,
cloud services or macOS. Linux is not planned for 1.x; possible later (Vulkan keeps the door open),
but packaging and testing are Windows-only for now.

---

## 1. The flow

```
 photo(s) ──► generate ──► refine/train ──► relight ──► edit ──► export
           (generator     (trainer        (Sabercat    (Sabercat  (PLY, glTF
            plugin)        plugin)         relight)     kernels)   + relight data)
```

Every stage reads and writes the same thing: a **layer**. A layer is a
`GaussianCloud`, meaning the canonical streams plus the optional relight
streams, held on the shared device. Stages do not convert formats between
each other. Loaders and exporters only appear at the edges.

### 1.1 Image to splats (generate)

A single-image model regresses a full 3DGS scene in one forward pass. Such
models run in seconds on a desktop GPU. Their weights often carry licenses
that do not allow commercial use. So Studio never bundles model weights.
Models are user-supplied downloads behind a model-agnostic interface.

Two ways to run a model:

| | ONNX Runtime inside a generator DLL | Sidecar process |
|---|---|---|
| Install | Runtime DLLs plus the model file | Python, a deep learning framework and the model |
| GPU | DirectML (any D3D12 GPU) or CUDA execution providers | CUDA in practice; CPU is slow |
| Fidelity | Depends on the export; check it against the reference implementation | The reference implementation |
| New models | Need an export each; unsupported operators can block it | Any repository, often unchanged |
| Failure isolation | A crash or out-of-memory takes the app down | The sidecar dies and the app reports it |
| Hand-off | Host arrays to `setGaussians` | A `.ply` on disk, or shared memory |

Neither path can share the Vulkan device, because ONNX Runtime runs on D3D12
or CUDA. That is acceptable. One image's output is about 1.2 million
Gaussians of 14 floats each, roughly 70 MB. A host copy costs milliseconds
next to inference.

**Recommendation.** Define one generator interface (section 8). Implement it
first with the sidecar, so models can be tried quickly and checked against
their reference. Then add ONNX Runtime in a DLL as the shipping path, once one
model's export is validated numerically. Render the same camera from both
paths and require matching images. The interface hides which backend ran. The
generator also returns the input camera (intrinsics and pose), so Studio can
open on the photo's viewpoint and the refine step has its supervision view.

### 1.2 Refine and train

- **Multi-view capture** (COLMAP or a video): a trainer plugin drives
  Sabercat's forward and backward passes through the C ABI
  ([docs/c-api.md](c-api.md)).
- **Single image:** a short refinement against the one input view sharpens
  what the camera sees. Off-axis views stay as the model produced them.
  Studio exposes this as an optional "refine to photo" with an iteration
  budget.

Training and editing share one queue, so training is **time-sliced**: N
optimizer steps per UI frame, with N adapted to a frame budget (for example
12 ms). Optimizer state (Adam moments) is keyed by Gaussian index. Every edit
that reorders, deletes or appends Gaussians goes through an index remap that
Studio passes to the plugin (section 8).

### 1.3 Relight

Sabercat has a G-buffer with deferred GGX shading, PCSS and ray-traced
shadows, and a baked per-Gaussian visibility term. See
[docs/architecture.md](architecture.md). Learning materials and visibility
from data is a trainer plugin's job. A generated single-image scene has no
learned materials. Studio starts from defaults (the shortest ellipsoid axis as
the normal, base color from the SH DC term, roughness 0.8) and lets the artist
paint over them (section 1.4).

**Known gap:** the DC color still contains the photo's baked lighting.
Delighting needs an intrinsic-decomposition model, which would be another
generator behind the same interface. It is an open question (section 7), not
a planned item.

### 1.4 Edit

Every operation is a compute kernel over the layer's buffers, so a selection
of millions of Gaussians stays interactive.

- **Selection** is a per-Gaussian bit buffer: one `uint` per 32 Gaussians, on
  the GPU.
  - Screen tools: rectangle, lasso, and a brush with depth-aware pick. They
    test each Gaussian's projected center, and optionally its 2D extent. They
    reuse the forward pass's projected Gaussians, so no second projection is
    needed.
  - Volume tools: box and sphere in world space.
  - Similarity growth: color or normal within a tolerance of a seed.
  - Selection operations (add, subtract, invert, grow) are bitwise kernels.
  - A highlight debug view tints selected Gaussians.
- **Transform** applies an affine matrix M to the selection. The mean becomes
  M·μ. The covariance becomes A·Σ·Aᵀ, where A is the linear part of M. It is
  factored back into rotation and scale by a 3×3 symmetric eigen
  decomposition. The factoring is exact for rigid transforms and uniform
  scale. The general case keeps an exact covariance. SH bands 1 to 3 rotate
  with the rotation part of A, using a per-band Wigner rotation (see
  `src/renderer/sh_rotation.cpp`). Interaction uses a gizmo with a live
  preview. The preview is the same kernel applied to a scratch copy, and is
  committed on release.
- **Delete** sets a tombstone (opacity 0, which the forward pass already
  skips below 1/255) for instant feedback. A compaction kernel (prefix sum
  and scatter, reusing `prefix_sum.slang`) runs on commit, save or export. It
  emits the index remap for the plugin's optimizer state.
- **Duplicate** appends the selection into spare capacity (`reserveGaussians`,
  grow-only storage) and selects the copies, so the gizmo can move them at
  once.
- **Material painting** is a brush that writes `baseColorRoughness` and
  `metallic` (and optionally normals) for the Gaussians under it. Falloff is
  radial, weighted by each Gaussian's screen coverage. The brush creates the
  relight streams on first use if the layer lacks them.
- **Acceleration structure.** Transforms and paint refit the ray-tracing
  acceleration structure. Delete compaction and duplicate rebuild it.
- **Undo** uses the command pattern. Each command stores what it needs to
  invert itself:
  - transforms store the matrix and the selection;
  - delete and duplicate store the index ranges and the remap;
  - paint and destructive operations store a sparse snapshot of the touched
    Gaussians' prior values, read back from the GPU for the selected indices
    only.
  - History has a memory budget (default 1 GB of host memory). The oldest
    entries are dropped first, with a warning.

### 1.5 Export

- **PLY:** the standard 3DGS layout. When relight streams exist, it adds the
  R3DG-style attributes (`nx ny nz`, `base_color_*`, `roughness`,
  `visibility_*`), so other 3DGS tools can read it. This is the inverse of
  the PLY reader.
- **glTF with `KHR_gaussian_splatting`:** the ellipse kernel, the
  `srgb_rec709_display` color space, and one node per layer that carries the
  layer transform. The color space is `srgb_rec709_display` because 3DGS SH
  coefficients are trained against sRGB photos, so the values are already
  display-referred. Writing them as linear would brighten and shift every
  color. Studio writes a single `.glb` (`GLTFWriter`).
- **Relight data in glTF (proposed):** a `SABERCAT_gaussian_splatting_relight`
  extension on the same primitive. The vendor prefix stays until a second
  implementation exists. Then it can become an `EXT_` proposal.
  - Normals use glTF's core `NORMAL` attribute (`VEC3`, float). No new
    semantic is needed.
  - `SABERCAT_gaussian_splatting_relight:BASE_COLOR_ROUGHNESS` (`VEC4`, linear
    RGB plus roughness), `:METALLIC` (`SCALAR`), and `:VISIBILITY_SH_0..3`
    (four `VEC4`s holding the 16 mono SH coefficients in the 3DGS-signed basis;
    see `shBasis16` in `include/sabercat/lighting.h`).
  - Lights: `KHR_lights_punctual` for directional, point and spot. The
    environment's order-2 SH irradiance goes in `EXT_lights_image_based`'s
    `irradianceCoefficients`. Rect lights, which no ratified extension covers,
    go in the vendor extension's scene-level `lights` array.
  - Readers that ignore the extension still show a correct unlit splat,
    because the extension is not listed in `extensionsRequired`.

---

## 2. Process and device model

One process. **The Studio executable owns the Vulkan instance and device.**
Sabercat and any plugin borrow them:

```
studio.exe ── VkInstance / VkDevice (requirements = Sabercat's, merged with each plugin's)
  ├── Sabercat (static library)    borrowed context, see docs/library.md
  └── plugin DLL (C ABI)           borrowed context, same handles
                                   (plus any runtime it brings, on its own device path)
```

- **Requirements.** Studio builds one `DeviceRequirements` from Sabercat's
  needs, with training and presentation required. It merges in each plugin's
  declared requirements (section 8) and passes the same
  `EnabledDeviceFeatures` to every user of the device.
- **Buffers.** Studio allocates the layer buffers itself, so Sabercat and the
  plugins borrow them and neither owns the scene. Sabercat renders them in
  place. A trainer steps them in place. Edits are Sabercat kernels on the same
  buffers.
- **Queues and threads.** One compute queue, externally synchronized. A
  single **GPU thread** owns every submission: frame rendering, edit kernels
  and training steps. The UI thread posts commands to it. Inference runs on
  worker threads, off the queue, and hands host arrays to the GPU thread.
- **Failure handling.** An in-process plugin can crash the app, so Studio
  autosaves the project (section 4) every few minutes, and before it runs any
  generator or long training. The sidecar backend is the isolation option for
  untrusted or experimental models.

## 3. UI plan

Dear ImGui on top of the viewer's panels. The viewer's relight
and shadow windows carry over.

| Panel | Contents |
|---|---|
| Viewport | Orbit, fly and photo-camera views; debug view picker; selection overlay; gizmo |
| Layers | Layer list: visibility, lock, Gaussian count, relight streams present; merge and split |
| Inspector | Selected layer or selection: transform, material values, statistics |
| Generate | Image drop target, model picker (installed models), backend, run and cancel, progress |
| Train | Dataset, iterations, time-slice budget, loss curve, pause and resume |
| Light | The relight window: lights, environment, shadow mode, baked visibility |
| Paint | Brush size, falloff, channel (base color, roughness, metallic, normal) |
| History | The undo stack, with click to jump |

Viewport modes: **View**, **Select** (with tool sub-modes), **Transform** and
**Paint**, with conventional hotkeys (Q, W, E, R and B).

## 4. Project file format

S1 writes this layout, as implemented in `src/renderer/project.cpp`. A
`.sabercat` project is a ZIP that stores its entries without compression, so
float data is not recompressed:

```
project.json            format, formatVersion, savedWith, layers, app
layers/<id>/            one sabercat.cloud-v1 directory per layer (the exchange format)
layers/<id>/relight/    relight streams, schema sabercat.relight-v1, when the layer has any
```

Planned, not written in S1:

```
sources/                input photos and capture metadata, copied in
thumbnails/project.png  preview image
```

- `project.json` records, as S1 writes it:
  - `format`, `formatVersion` and `savedWith` (the Sabercat version);
  - per layer: id, name, transform (translation, rotation, scale), and the
    visible, locked and antialiased flags;
  - `app`: Studio's camera, SH degree, debug view and Relight settings.
- Planned, not written in S1: lights and environment; camera bookmarks,
  including a photo camera; training configuration; provenance beyond
  `savedWith` (source files, the generator model's name and hash, and the
  plugin version).
- Layer payloads reuse the cloud-v1 directory, so any tool that reads it can
  read a project layer directly. Relight streams go in the separate
  `layers/<id>/relight/` sidecar, with optional `normals`,
  `base_color_roughness`, `metallic` and `visibility` entries. The cloud-v1
  directory itself is unchanged. A reader may ignore the sidecar.
- Undo history is not saved. Selections are not saved by default.
- Saving writes a temporary file and renames it over the old one, so a crash
  during a save does not corrupt the project.

## 5. Milestones

Each milestone ships through the release flow in
[docs/releasing.md](releasing.md).

| Milestone | Scope | Done when |
|---|---|---|
| S1 | Shell: device owner, layers, project save and load, PLY and glTF export | **Shipped in 1.0.0.** Project save and load is bit-exact (`project_test`). PLY and glTF round trips through the Sabercat readers are measured (`ply_writer_test`, `gltf_writer_test`). Not yet checked in a third-party viewer |
| S2 | Editing: selection tools, transform, delete, duplicate, undo; GPU layer merge | Every operation matches a CPU reference on a synthetic cloud. Undo restores the buffers bit-exactly |
| S3 | Trainer plugin: time-sliced training and refine to photo; index remap after edits | A capture trains inside Studio while staying interactive. Editing mid-training keeps the optimizer consistent |
| S4 | Generation: interface, sidecar backend, then the ONNX Runtime backend | One image becomes splats inside Studio. Both backends render the same result within a stated tolerance |
| S5 | Material painting and relight export (the vendor glTF extension and R3DG-style PLY) | A painted, relit scene exports and reloads with identical streams |

## 6. Verification approach

The same rules as the renderer: CPU references for every kernel, bit-exact
round trips for save and load and for undo, measured tolerances where fp32
operation order differs, and zero validation messages in Debug and Release.
Nothing is signed off from screenshots alone.

## 7. Open questions

- **Delighting** generated scenes (intrinsic decomposition): which model, and
  should its output seed materials or only the base color?
- **Generator licenses:** many published single-image models have
  non-commercial weights. Is a permissively licensed model good enough to be
  the default?
- **Layer rendering:** one merged cloud with per-layer ranges (one sort, one
  forward pass), or per-layer passes composited. Per-layer passes are simpler,
  but wrong where layers interpenetrate. The merged form is the lean choice.
  S1 uses a host merge (`mergeLayers`, applied to the visible layers) to build
  the scene. A GPU merge, which applies layer transforms by kernel into the
  merged buffers, comes with S2.

## 8. Plugin interface (proposed)

Nothing in this section exists yet. It is the shape that S3 and S4 need. The
names are not final.

A **plugin** is a DLL with a C ABI, in the same style as `sabercat_c`. Studio
loads it at start-up. A plugin has one kind: `trainer` or `generator`.

| Call | Direction | Purpose |
|---|---|---|
| `requirements` | plugin to host | Device extensions and features the plugin needs. Studio merges them into the one device |
| `attach` | host to plugin | Passes the borrowed device context and the layer buffer views. Versioned by `struct_size`, like the C API |
| `generate` | host to plugin | Image and options in. Host arrays and the input camera out. Runs off the GPU thread |
| `train_step` | host to plugin | Runs up to a time budget of optimizer steps on the GPU thread. Returns the loss and the steps done |
| `remap` | host to plugin | Old-to-new Gaussian index map after delete or compaction. The plugin reorders its own state |
| `detach` | host to plugin | Releases borrowed views. The plugin must not touch the device after it returns |

Rules that every plugin follows:

- **Borrowing.** The plugin never owns the scene buffers. It only reads and
  writes the views it was given.
- **Synchronization.** All device work happens on the GPU thread, one
  submission at a time. A call that returns before the plugin reads the
  buffers again is synchronous.
- **Relight data.** A trainer may write the optional relight streams. It may
  also leave them alone. Studio does not require them.
- **Failure.** A plugin error is returned as a status code, not an exception.
  A crash in an in-process plugin is survived only by autosave. A sidecar
  plugin isolates the crash.
- **Versioning.** The plugin reports an API version. Studio refuses a plugin
  whose major version it does not know.

Sabercat provides no trainer, generator or model. Those are external.

## 9. Relation to the rest of the repository

- Engine behavior that Studio depends on is documented in
  [docs/architecture.md](architecture.md) and [docs/library.md](library.md).
- Formats are documented in [docs/formats.md](formats.md).
- Remaining engine and format work is in [ROADMAP.md](../ROADMAP.md).
