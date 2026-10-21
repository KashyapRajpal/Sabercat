# Unit tests

CPU-only unit tests (CTest label `cpu`). They need no Vulkan device and no
window, so they run on the hosted CI runner.

- `sabercat_unit_test` covers the apps' GPU-free logic: the orbit camera
  (`apps/common/orbit_camera.h`), the Studio project model and app-state JSON
  (`apps/studio/studio_state.h`), and the app policy and scene scan
  (`apps/common/app_policy.h`, `asset_scan.h`), and the native file dialogs'
  pure helpers: filter specs, default extensions and UTF-8 / UTF-16 conversion
  (`apps/common/file_dialog.h`). No dialog is shown.
- Layers in these tests carry no cloud, because a cloud needs a device. Cloud
  paths are covered by the GPU `studio_selftest`.

Run with `ctest --test-dir build -C Release -L cpu --output-on-failure`. The GPU
programs are in `../integration`.
