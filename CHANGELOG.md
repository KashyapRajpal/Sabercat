# Changelog

All notable changes to Sabercat are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-10-28

First public release.

### Added

- **Rendering** (`Sabercat::sabercat`, C++20, Vulkan 1.3 compute, Slang shaders)
  - Forward splat rendering with antialiasing, spherical harmonics up to the scene's degree, and level-of-detail parameters.
  - Optional per-pixel depth output from the differentiable forward. `Renderer` takes a `DepthLayout` (`None`, `Depth`, `DepthCoverage` or `DepthCoverageFront`) at construction, read with `readDepth()` or the zero-copy `depthView()`.
  - `Renderer::clearScene()`, for applications that hide every layer.
- **Training support**
  - Backward pass with gradients for the Gaussian parameters, using 32-bit float atomics.
  - Depth gradients: the `Renderer::backward()` overloads take dL/d(depth) in the depth layout. Color-only backward is unchanged.
- **Lighting and relighting**
  - Directional, point and rectangle lights and an environment light, with deferred shading.
  - Ray-traced shadows and a visibility bake (requires ray query). The bake sizes its GPU submissions to a time budget (`VisibilityBakeSettings::targetSubmitMilliseconds`), so dense scenes stay clear of the driver timeout.
- **Scene formats**
  - Read: PLY (including the SuperSplat compressed layout), `.splat`, SPZ, `.ksplat`, SOG (`.sog` bundle or `meta.json`), glTF / GLB with `KHR_gaussian_splatting`, and the cloud-v1 directory format (`sabercat.cloud-v1`).
  - Write: PLY, GLB, and cloud-v1 with its relight sidecar (`sabercat.relight-v1`).
- **Projects**: layer transforms and project files (`.sabercat`, format `sabercat.project`). A project round-trips bit for bit.
- **Integration**
  - Owned or borrowed Vulkan device, so an application can share its own device and Gaussian buffers with the renderer.
  - C ABI (`Sabercat::sabercat_c`, `c_api.h`) with lifecycle functions, error reporting and `struct_size` versioning for input and output structs. The depth layout and depth gradients are available through it (`SabercatRendererDesc::depth_layout`, `sabercat_renderer_read_depth()`, `sabercat_renderer_get_depth_view()`, `sabercat_renderer_backward_depth()` and `sabercat_renderer_backward_depth_view()`).
  - CMake package: `find_package(Sabercat 1.0 CONFIG)` finds any 1.x release.
- **Applications**
  - Sabercat Viewer (`sabercat_viewer`): orbit camera, drag and drop, relighting, and an ImGui debug panel.
  - Sabercat Studio (`sabercat_studio`): layers, an inspector, project save and open, PLY and GLB export, and a `--selftest` mode.
  - Native Windows Open and Save dialogs in both apps. A cloud-v1 folder is opened by its `manifest.json`.
  - Per-user `.sabercat` file association for Studio (Help > Register `.sabercat` files), under HKEY_CURRENT_USER and without administrator rights. See [Studio](docs/studio.md#register-sabercat-files).
  - Session logs, and Help > About with the logo, the log folder and Copy system info.
  - App icons and version information on both executables.
- **Testing**
  - 30 GPU tests registered with CTest. They fail on any Vulkan validation message.
  - A forward-output regression test with stored checksums (`forward_regression_test`).
  - CPU unit tests for the apps' GPU-free logic, run in CI.
- **Packaging**: a Windows x64 release zip (`sabercat-<version>-win64.zip`) with a self-test, and an apps zip (`sabercat-<version>-win64-apps.zip`) for people who only want to run the apps.
- **Documentation and policies**: README, build, viewer, Studio, library, C ABI, consuming, formats and architecture guides. Apache-2.0 license, NOTICE, third-party notices, Code of Conduct, contributing guide and security policy.

### Security

- Release packages do not embed the build machine's source or build paths. The release script builds from a mapped drive and fails if a package contains them.

### Numerics

Numerical behavior is covered by `forward_regression_test`, which stores hashes of the forward output. Any change that alters a stored hash gets an entry under Numerics in the release that makes it.

[Unreleased]: https://github.com/KashyapRajpal/Sabercat/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/KashyapRajpal/Sabercat/releases/tag/v1.0.0
