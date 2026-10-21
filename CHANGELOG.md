# Changelog

All notable changes to Sabercat are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0-beta.1] - 2026-10-21

First public release, as a beta for feedback. The final 1.0.0 follows once
beta feedback is addressed.

### Added

- Native Windows Open and Save dialogs in Sabercat Viewer (Browse... loads the chosen file and adds it to the scene list, Load loads the selected entry, and Enter in the path box loads a typed path) and Sabercat Studio (File > Open..., Save As..., Add Layer..., Export PLY... and Export GLB..., with Ctrl+O and Ctrl+Shift+S). The dialogs run in the background, so the windows keep responding while one is open. A cloud-v1 folder is opened by its `manifest.json`. A "Type a Path" menu in Studio (with Browse...) and the path box in the Viewer remain as the fallback.
- **Renderer library** (`Sabercat::sabercat`, C++20, Vulkan 1.3, Slang shaders)
  - Forward splat rendering with antialiasing, spherical harmonics up to the scene's degree, and level-of-detail parameters.
  - Backward pass with gradients for the Gaussian parameters, using 32-bit float atomics.
  - Relighting with directional, point and rectangle lights and an environment light.
  - Ray-traced shadows and visibility bake (requires ray query). The bake
    sizes its GPU submissions to a time budget
    (`VisibilityBakeSettings::targetSubmitMilliseconds`), so dense scenes stay
    clear of the driver timeout.
  - Owned or borrowed Vulkan device, so an application can share its own device with the renderer.
  - Layer transforms and project files (`.sabercat`, format `sabercat.project`).
- **C ABI** (`Sabercat::sabercat_c`, `c_api.h`) with lifecycle functions, error reporting and `struct_size` versioning for input and output structs.
- **Scene readers**: PLY, `.splat`, SPZ, `.ksplat`, SOG (`.sog` bundle or `meta.json`), glTF / GLB with `KHR_gaussian_splatting`, and the cloud-v1 directory format (`sabercat.cloud-v1`).
- **Scene writers**: PLY, glTF / GLB, and cloud-v1 with its relight sidecar (`sabercat.relight-v1`).
- **Sabercat Viewer** (`sabercat_viewer`): orbit camera, drag and drop, relighting toggle, and an ImGui debug panel.
- **Sabercat Studio** (`sabercat_studio`): layers, inspector, project save and open, PLY and GLB export, and a `--selftest` mode.
- **Tests**: 24 GPU test programs registered with CTest.
- **Packaging**: CMake package `find_package(Sabercat 1.0 CONFIG)` compatible with any 1.x release, and a Windows x64 release zip (`sabercat-<version>-<sha>-win64.zip`).
- **Documentation**: README, build, viewer, Studio, library, C ABI, consuming, formats and architecture guides.
- **Project policies**: Apache-2.0 license, NOTICE, third-party notices, Code of Conduct, contributing guide and security policy.
- App icons: the Sabercat Viewer and Studio executables carry the Sabercat icon and version information, and their windows show the icon in the title bar and taskbar.
- The Sabercat logo in Help > About for both apps, with the logo files in `docs/media/` and the icon files in `apps/common/resources/`.
- Per-user `.sabercat` file association for Sabercat Studio: Help > Register `.sabercat` files and Unregister `.sabercat` files, under HKEY_CURRENT_USER (no administrator rights). Double-clicking a project then opens it in Studio. See [Studio](docs/studio.md#register-sabercat-files).

### Changed

- The cloud-v1 schema id is `sabercat.cloud-v1`. The legacy id written before 1.0 is still read.

### Removed

- Removed the trainer-specific interop header.

## 0.x

- 0.x: private pre-releases.

[Unreleased]: https://github.com/KashyapRajpal/Sabercat/compare/v1.0.0-beta.1...HEAD
[1.0.0-beta.1]: https://github.com/KashyapRajpal/Sabercat/releases/tag/v1.0.0-beta.1
