# Roadmap

This page lists planned work after 1.0.0. It has no dates. Items are not
promises. What ships today is in [CHANGELOG.md](CHANGELOG.md) and
[docs/architecture.md](docs/architecture.md).

Studio phases are described in [docs/studio-design.md](docs/studio-design.md).

## Sabercat Studio

| Phase | Scope | Done when |
|---|---|---|
| S2 | Editing: selection tools, transform, delete, duplicate, undo. GPU layer merge, so layer transforms are applied by a kernel instead of on the host | Every operation matches a CPU reference on a synthetic cloud. Undo restores the buffers bit-exactly |
| S3 | Trainer plugin: time-sliced training and refine-to-photo inside Studio, with an index remap after edits | A capture trains while the viewport stays interactive. Editing during training keeps the optimizer state consistent |
| S4 | Generation: a generator plugin interface, a sidecar backend, then an in-process backend | One image becomes splats inside Studio. Both backends render the same result within a stated tolerance |
| S5 | Material painting and relight export: a brush for base colour, roughness, metallic and normals, a vendor glTF extension for relight data, and PLY attributes that other 3DGS tools read | A painted, relit scene exports and reloads with identical streams |

Studio items that are not yet in a phase:

- Lights, environment and bake settings saved in project files.
- Empty scene handling in the viewport. When every layer is hidden the viewport is empty (internally Studio renders a single fully transparent Gaussian, because the renderer has no clear-scene call).

The plugin interface is a proposal. See the "Plugin interface (proposed)"
section of [docs/studio-design.md](docs/studio-design.md).

## Volumes

A volumetric rendering track, shown in a new app (`sabercat_volumes`), with
the core passes in the library so Studio can use them later.

| Milestone | Scope | Done when |
|---|---|---|
| V1 | Ray-marched participating media: fog, clouds and smoke with single scattering, light shadowing and a sky. Fog composited into splat scenes using their depth. Gaussian volumes: media stored as volumetric 3D Gaussians, integrated in closed form along rays through the existing Gaussian acceleration structure | Homogeneous media match the analytic transmittance `exp(-sigma * d)`. Every pass matches a CPU reference marcher within a stated tolerance |
| V2 | Neural volumes: a small multilayer perceptron with a hash-grid encoding predicts density and colour. It is trained on the GPU to reproduce a procedural or grid volume, then rendered in real time | The forward pass matches a CPU port. Gradients match finite differences. The fitted volume renders within a stated error of its source |
| V3 | Neural volumes trained from multi-view images | A captured scene reconstructs within a stated image error |

## Engine

- **Clear-scene API.** A call that removes the current scene without loading
  another. Today a scene is replaced by `loadScene` or `setGaussians`.
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
- Mesh authoring.
- Cloud services.
