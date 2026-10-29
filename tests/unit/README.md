# Unit tests

CPU-only unit tests (CTest label `cpu`). They need no Vulkan device and no
window, so they run on the hosted CI runner.

- `sabercat_unit_test` covers the apps' GPU-free logic: the orbit camera
  (`apps/common/orbit_camera.h`), the Studio project model and app-state JSON
  (`apps/studio/studio_state.h`), and the app policy and scene scan
  (`apps/common/app_policy.h`, `asset_scan.h`), and the native file dialogs'
  pure helpers: filter specs, default extensions and UTF-8 / UTF-16 conversion
  (`apps/common/file_dialog.h`). No dialog is shown.
- `test_surface_format.cpp` covers the swapchain format choice
  (`src/renderer/surface_format.h`), which needs a UNORM format.
- `test_system_info.cpp` covers the system-info text, the session log naming and
  pruning, and the Logger's file mirror.
- `test_file_association.cpp` covers the per-user `.sabercat` association: the
  open command, the icon value and, on Windows, register, query and unregister
  against a scratch registry root.
- Layers in these tests carry no cloud, because a cloud needs a device. Cloud
  paths are covered by the GPU `studio_selftest`.

Run with `ctest --test-dir build -C Release -L cpu --output-on-failure`. The GPU
programs are in `../integration`.
