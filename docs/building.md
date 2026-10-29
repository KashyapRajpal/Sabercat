# Building Sabercat

This guide covers building the library, the two applications and the tests
from source on Windows. For using the built library in your own project, see
[consuming.md](consuming.md).

## Requirements

| Item | Version or detail |
|------|-------------------|
| OS | Windows 10 or 11, x64 |
| Compiler | Visual Studio 2022 (MSVC, C++20) |
| CMake | 3.22 or newer |
| Vulkan SDK | 1.3 or newer. 1.4.309.0 is the tested version. |
| GPU | Vulkan 1.3, see [GPU requirements](#gpu-requirements) |
| Git | With submodule support. The repository has one submodule, `external/spz`. |
| Network | Needed the first time you configure. CMake downloads dependencies (see below). |

The Vulkan SDK provides two things:

- `slangc`, the Slang shader compiler. It is needed at build time only.
  CMake finds it through the `VULKAN_SDK` environment variable (`Bin` or
  `bin` under that folder). If it is missing, configuration stops with an
  error that names `slangc`.
- The `VK_LAYER_KHRONOS_validation` layer. It is optional at run time. See
  [Validation layer](#validation-layer).

### GPU requirements

The device must offer these. The library checks them at start-up and reports
the first one that is missing.

| Requirement | Why |
|-------------|-----|
| Vulkan 1.3 | Core API version the library uses. |
| 256 or more compute invocations per workgroup | Shader workgroup size. |
| 32-bit float buffer atomics (`VK_EXT_shader_atomic_float`) | Needed for the backward pass (training). The viewer and Studio do not need it. |
| Ray query (acceleration structures, `rayQuery` and buffer device address) | Optional. It is used for ray-traced shadows and visibility baking. Without it, those options are disabled. |
| Swapchain and dynamic rendering | Needed by the viewer and Studio to show a window. |

## Get the source

```powershell
git clone --recurse-submodules https://github.com/KashyapRajpal/Sabercat.git
cd Sabercat
```

If you cloned without `--recurse-submodules`, run this inside the clone:

```powershell
git submodule update --init --recursive
```

## Configure

Use the Visual Studio 2022 generator. It is a multi-config generator, so
Debug and Release are both chosen at build time:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
```

The first configure downloads these dependencies. Later configures reuse
them:

| Dependency | Source | Used by |
|------------|--------|---------|
| GLFW 3.4 | Fetched only if no system `glfw3` is found | Viewer and Studio |
| Dear ImGui 1.91.5 | Fetched | Viewer and Studio |
| GLM 1.0.1 | Installed package or the Vulkan SDK if present, otherwise fetched (pinned) | Library |
| VMA 3.2.1 | The Vulkan SDK if it ships VMA, otherwise fetched (pinned) | Library |
| spz | Submodule `external/spz` | Library (`.spz` reader) |
| zlib 1.3.1 | Fetched (pinned by SHA-256) | Library (`.spz`, `.sog`) |
| libwebp 1.6.0 | Fetched (pinned by SHA-256) | Library (`.sog`) |
| nlohmann/json 3.12.0 | Fetched (pinned by SHA-256) | Library (glTF, `.sog`, project files) |

### CMake options

| Option | Default | Effect |
|--------|---------|--------|
| `SABERCAT_BUILD_APPS` | `ON` when Sabercat is the top-level project | Builds `sabercat_viewer` and `sabercat_studio`. |
| `SABERCAT_BUILD_EXAMPLES` | `ON` when Sabercat is the top-level project | Builds the programs in `examples/`, including the `*_test` programs. |
| `SABERCAT_BUILD_TESTS` | `ON` when Sabercat is the top-level project | Builds the tests and registers them with CTest. |
| `SABERCAT_FETCH_HEADER_DEPS` | `OFF` | Fetches the pinned GLM and VMA even when an installed package (GLM) or the Vulkan SDK provides them. Use it for reproducible builds. |

To build only the library, for example:

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DSABERCAT_BUILD_APPS=OFF -DSABERCAT_BUILD_EXAMPLES=OFF -DSABERCAT_BUILD_TESTS=OFF
```

## Build

```powershell
cmake --build build --config Release --parallel
cmake --build build --config Debug --parallel
```

Build outputs go to the default Visual Studio folders. For example:

| Program | Path |
|---------|------|
| Viewer | `build\apps\viewer\Release\sabercat_viewer.exe` |
| Studio | `build\apps\studio\Release\sabercat_studio.exe` |

The build copies `zlib` next to every executable that needs it, so the
programs run from their build folder without extra steps.

You can also open the folder in Visual Studio (File, Open, CMake). Or open
the generated `build\Sabercat.sln`.

## Install

```powershell
cmake --install build --config Release --prefix <prefix>
```

The install tree has this layout:

| Path under `<prefix>` | Contents |
|-----------------------|----------|
| `include\sabercat\` | Public headers, including `version.h` and `shaders\config.slangh` |
| `lib\<Config>\` | `sabercat.lib`, `sabercat_c.lib`, and the vendored libraries |
| `bin\<Config>\` | `sabercat_c.dll`, `sabercat_viewer.exe`, `sabercat_studio.exe`, `zlib` DLL |
| `share\sabercat\shaders\` | Compiled shaders (`*.spv`) |
| `lib\cmake\Sabercat\` | CMake package files |

Debug and Release share one prefix. Each configuration has its own `lib` and
`bin` subfolder, because MSVC cannot mix the two runtimes.

The installed viewer and Studio find their shaders in
`share\sabercat\shaders`, relative to the executable.

## Shader lookup

The apps find the compiled shaders in this order:

1. The folder named by the `SABERCAT_SHADER_DIR` environment variable.
2. `..\..\share\sabercat\shaders` relative to the executable (the install
   layout).
3. The shader folder in the build tree, compiled into the app.

To run an app against a different shader build, set `SABERCAT_SHADER_DIR`
before starting it.

## Validation layer

The Vulkan validation layer (`VK_LAYER_KHRONOS_validation`) is set like this:

| Build or flag | Result |
|---------------|--------|
| Debug build | On |
| Release build | Off |
| `--validation` on the command line | On, also in Release |
| `--no-validation` on the command line | Off, also in Debug |

If the layer is not installed, the app logs a warning and runs without it.
Install the layer from the Vulkan SDK to use it. The flags are accepted by
`sabercat_viewer` and `sabercat_studio`.

## Tests

CTest is the single test list. Two labels split it:

- `cpu` (`tests/unit`): CPU-only unit tests for the apps' GPU-free logic
  (orbit camera, Studio project model, app-state JSON, app policy). No Vulkan
  device is needed. CI runs these after every build.
- `gpu` (`tests/integration`): the `*_test` programs in `examples/`, plus
  `studio_selftest`. They need a GPU with Vulkan 1.3.

```powershell
ctest --test-dir build -C Release -L cpu --output-on-failure
ctest --test-dir build -C Release -L gpu --output-on-failure
```

The continuous integration workflow (`.github/workflows/build.yml`) builds the
project, runs the `cpu` tests and checks the install step. Hosted runners have
no GPU, so the `gpu` tests are not run in CI. Run them on a machine with a GPU
before you change the library.

The tests use their own fixtures. A few also run an extra check on a local
scene under `assets\` when one is present, and skip it otherwise.

## Troubleshooting

| Symptom | Cause and fix |
|---------|---------------|
| Configure stops with `slangc` not found | Install the Vulkan SDK and set `VULKAN_SDK`, or add its `Bin` folder to the search path. |
| Configure fails while downloading | No network, or a proxy blocks GitHub or storage.googleapis.com. Configure again once the download works. |
| `external/spz` is empty | The submodule was not cloned. Run `git submodule update --init --recursive`. |
| The zlib DLL cannot be found at start-up | Run the program from its build or install `bin` folder, where the DLL is copied. |
| Start-up error about the device or a Vulkan feature | The GPU or driver does not meet the requirements above. Update the driver or use a GPU that supports Vulkan 1.3. |
| Start-up error about 256 compute invocations | The GPU reports fewer than 256 invocations per workgroup. It is not supported. |
| Many `[validation]` lines | Expected only with the validation layer on. Read the first message; it names the object and the call. |
