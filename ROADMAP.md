# Roadmap

This page lists planned work after 1.0.0, grouped by the release it is
aimed at. It has no dates. Items are not promises, and an item can move
between releases. What ships today is in [CHANGELOG.md](CHANGELOG.md) and
[docs/architecture.md](docs/architecture.md).

| Release | Theme |
| --- | --- |
| 1.1 | Evaluation tooling; a sorting study against popping; release packages built in CI; repository checks (headers, secrets, local paths) |
| 1.5 | Hybrid rendering (triangles and splats together) with dynamic rendering and variable rate shading; a material system; reflections |
| 2.0 | Volumetric and neural rendering |

Sabercat Studio phases land during 1.x as they are ready (see
[Sabercat Studio](#sabercat-studio)).

## 1.1

- **Perceptual image comparison.** An image-comparison tool with PSNR and the
  FLIP perceptual metric, with error maps that show where two images differ.
- **Release packages built in CI.** A tag-triggered GitHub Actions workflow
  builds the packages on a clean runner and publishes build-provenance
  attestations, so a download can be verified against its commit. The GPU tests
  keep running on a local GPU before a release, and the CI build is
  cross-checked against the locally tested one.
- **Repository checks in CI.** A workflow that fails when a source file lacks
  the copyright and SPDX header, when a secret or credential is committed
  (secret scanning), or when a file contains an absolute local path (such as
  a drive-letter or user-profile path).
- **Wider test scenes.** Regular checks on user-made phone captures (for
  example Scaniverse PLY exports) in addition to the classic research scenes.
  Scenes stay local and are not redistributed.
- **Sorting study.** The rasterizer sorts per tile by Gaussian center depth,
  which can pop when splats overlap. Measure the alternatives against the
  current sort for quality and cost:
  - per-pixel and hybrid sorting;
  - a stable depth sort across frames, so equal or near-equal depths do not
    swap order from one frame to the next;
  - stochastic transparency: order-independent blending from blue-noise
    samples with temporal reuse and a resolve pass, which avoids sorting
    artifacts at the price of noise.

## 1.5: hybrid rendering and materials

**Hybrid rendering.** Triangle meshes in the same scene as splats, with
correct blending between the two.

- Triangles are rasterized with Vulkan dynamic rendering (no render pass
  objects, matching the rest of the engine).
- Opaque triangles render first and write depth. The splat tile rasterizer
  stops accumulating at each pixel's triangle depth, so opaque geometry
  occludes splats at almost no cost.
- Transparent triangle fragments go into the same per-tile, depth-sorted lists
  as the splats, so both blend in one front-to-back pass. This is exact where
  transparent triangles and splats interpenetrate, at the cost of a larger
  sort.
- Variable rate shading (`VK_KHR_fragment_shading_rate`) for the triangle
  pass, and an evaluation of a matching per-tile shading rate for the splat
  pass in screen regions with low detail.
- An evaluation of a hardware-rasterized splat path (splats drawn as quads
  through the graphics pipeline) next to the compute tile rasterizer, since
  it can share the triangle pass and its depth buffer.
- Shadows and visibility through both: the acceleration structure holds
  triangles next to the Gaussian boxes.

**Material system.** Replaces the single set of per-Gaussian metallic,
roughness and base color values with materials that splats and triangles
share.

- A material table, and a per-Gaussian (and per-triangle) material index.
- A layered model: base metallic-roughness, plus clearcoat, sheen,
  transmission and emission.
- Mapping to and from glTF materials (the core model and the `KHR_materials_*`
  extensions).
- Studio's material painting (phase S5) writes material parameters.

**Reflections.**

- Reflection rays through the splat acceleration structure. Unlike shadow
  rays, which need only an order-independent transmittance product, reflection
  rays composite color in depth order through many semi-transparent splats.
- A high-frequency environment (a cube map or equirectangular map) for
  specular, in place of today's degree-2 spherical-harmonic environment, so
  mirror-like and metallic surfaces reflect sharply.

| Item | Done when |
| --- | --- |
| Hybrid rendering | Opaque and transparent triangles composite with splats to match a CPU reference compositor on test scenes, within a stated tolerance |
| Material system | Every material layer matches a CPU port of the shading model; glTF materials round-trip through import and export |
| Reflections | Reflection rays match a CPU reference ray compositor through the splats; a chrome sphere reflects its surroundings |

## 2.0: volumetric and neural rendering

A volumetric rendering track, shown in a new app (`sabercat_volumes`), with
the core passes in the library so Studio can use them. It is a major release
because the scene and rendering API grow beyond splats and triangles.

| Milestone | Scope | Done when |
| --- | --- | --- |
| V1 | Ray-marched participating media: fog, clouds and smoke with single scattering, light shadowing and a sky. Fog composited into splat scenes using their depth. Gaussian volumes: media stored as volumetric 3D Gaussians, integrated in closed form along rays through the existing Gaussian acceleration structure | Homogeneous media match the analytic transmittance `exp(-sigma * d)`. Every pass matches a CPU reference marcher within a stated tolerance |
| V2 | Neural volumes: a small multilayer perceptron with a hash-grid encoding predicts density and color. It is trained on the GPU to reproduce a procedural or grid volume, then rendered in real time | The forward pass matches a CPU port. Gradients match finite differences. The fitted volume renders within a stated error of its source |
| V3 | Neural volumes trained from multi-view images | A captured scene reconstructs within a stated image error |

## Sabercat Studio

Studio phases are described in [docs/studio-design.md](docs/studio-design.md).

| Phase | Scope | Done when |
| --- | --- | --- |
| S2 | Editing: selection tools (rectangle, lasso, polygon, brush, volume), a screen-space outline around the selected splats, transform, delete, duplicate, undo. GPU layer merge, so layer transforms are applied by a kernel instead of on the host | Every operation matches a CPU reference on a synthetic cloud. Undo restores the buffers bit-exactly |
| S3 | Trainer plugin: time-sliced training and refine-to-photo inside Studio, with an index remap after edits | A capture trains while the viewport stays interactive. Editing during training keeps the optimizer state consistent |
| S4 | Generation: a generator plugin interface, a sidecar backend, then an in-process backend | One image becomes splats inside Studio. Both backends render the same result within a stated tolerance |
| S5 | Material painting and relight export: a brush for material parameters (base color, roughness, metallic, normals, and the material system's layers from 1.5), a vendor glTF extension for relight data, and PLY attributes that other 3DGS tools read | A painted, relit scene exports and reloads with identical streams |

Studio items that are not yet in a phase:

- Lights, environment and bake settings saved in project files.
- **Share as a web page.** Export a scene with a standalone HTML viewer, so it
  can be opened in a browser without installing anything.
- **Automation.** A scripting surface for Studio (for example embedded Python)
  and a Model Context Protocol (MCP) server, so scripts and tools can load,
  edit, render and export scenes.

The plugin interface is a proposal. See the "Plugin interface (proposed)"
section of [docs/studio-design.md](docs/studio-design.md).

## Engine

- **Zip64.** The ZIP reader and writer in `src/renderer/zip_archive.cpp`
  handle only classic ZIP. Archives over 4 GiB, or more than 65,534 entries,
  are refused. Zip64 removes both limits.
- **Occlusion culling** in the forward pass. Deferred when the frustum and
  LOD culling landed.
- **Linux.** Not planned for 1.x; possible later (Vulkan keeps the door open). Only Windows is built, tested and packaged today.
- **Trainer-side examples.** A small example that drives the backward pass
  and the C ABI backward views from a training loop, to show the contract end
  to end.

## Formats

Writers to add, so scenes leave Sabercat in compressed delivery formats as
well as PLY and glTF:

- SPZ (Niantic's compressed format; the reader exists today).
- SOG (PlayCanvas's WebP-based bundle; the reader exists today).

Formats that the readers do not load yet:

- `.splatv` (4D, animated splats). It needs the per-Gaussian time stream
  that the cloud already reserves (`GaussianCloud::times`), and its own
  loader.
- SOG v1, and streamed `lod-meta.json` scenes.
- `.ksplat` files with spherical harmonics degree 3. The reader supports
  degrees 0 to 2.
- glTF compression extensions other than SPZ.

## Not planned

- macOS.
- Mesh authoring. (Hybrid rendering in 1.5 renders imported triangle meshes;
  it does not edit them.)
- Cloud services.
