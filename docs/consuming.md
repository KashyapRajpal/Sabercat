# Consuming Sabercat

This guide is for an application or library that builds against Sabercat. There
are two ways to get it: a prebuilt release package, or the source tree. The
package is the recommended route. It needs no Sabercat build on your side.

- The API you call: [library.md](library.md) (C++) or [c-api.md](c-api.md) (C).
- The file formats: [formats.md](formats.md).

## Contents

- [Requirements](#requirements)
- [Option 1: the release package](#option-1-the-release-package)
- [Package contents](#package-contents)
- [Finding the package](#finding-the-package)
- [Targets](#targets)
- [Shader directory](#shader-directory)
- [Runtime DLLs](#runtime-dlls)
- [Version compatibility](#version-compatibility)
- [Option 2: the source tree](#option-2-the-source-tree)
- [Checking a package](#checking-a-package)

## Requirements

| Item | Needed for |
|---|---|
| Windows 10 or 11, x64 | The package and the tested platform |
| Visual Studio 2022 (MSVC v143), x64 | Building your application. The package is built with the same toolset. |
| CMake 3.22 or later | `find_package(Sabercat ...)` |
| Vulkan SDK 1.3 or later | Your configure step. The package's config runs `find_dependency(Vulkan)`. |
| A GPU and driver with Vulkan 1.3 | Running your application |

The Vulkan SDK is needed at configure time, because `SabercatConfig.cmake` looks
up Vulkan. The package's own shaders are compiled SPIR-V, so `slangc` is not
needed.

## Option 1: the release package

Each release is tagged `v<version>` and has a GitHub Release at
<https://github.com/KashyapRajpal/Sabercat/releases>. Download the zip named:

```
sabercat-<version>-win64.zip
```

Unpack it into a directory of your choice. Call that directory `<prefix>` below.
Configure your project with it on the prefix path:

```
cmake -S my_app -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH=<prefix>
cmake --build build --config Release
```

Use stable releases, not the `master` branch. Each release is a fast-forward of
the `stable` branch to a commit that passed every test with no validation
messages. The release notes list what changed. [CHANGELOG.md](../CHANGELOG.md)
has the full history.

## Package contents

| Path | Contents |
|---|---|
| `lib/cmake/Sabercat/` | `SabercatConfig.cmake`, `SabercatConfigVersion.cmake`, `SabercatTargets.cmake`, `SabercatHelpers.cmake`, and one `SabercatDeps-<Config>.cmake` per configuration |
| `include/sabercat/` | All public headers. `version.h` holds the release version. |
| `lib/Debug/`, `lib/Release/` | `sabercat.lib`, `sabercat_c.lib`, and the vendored decoder libraries |
| `bin/Debug/`, `bin/Release/` | `sabercat_c.dll`, the vendored zlib DLL, and the apps: `sabercat_viewer.exe`, `sabercat_studio.exe` |
| `share/sabercat/shaders/` | The compiled SPIR-V (`*.comp.spv`) |
| `include/sabercat/shaders/config.slangh` | The shared shader configuration, for consumers that compile their own shaders |
| `selftest/` | `selftest.ps1` and a few packaged test programs |
| `manifest.json` | Version, commit, build configuration, and per-test results |
| `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`, `README.md`, `CHANGELOG.md` | Licences, notices and the change log |
| `docs/` | `consuming.md`, `library.md`, `c-api.md`, `formats.md` |

Debug and Release share one prefix. Each configuration has its own `lib/<Config>`
and `bin/<Config>` directory. A multi-config generator picks the right one.
Link the same configuration as your application.

## Finding the package

```cmake
cmake_minimum_required(VERSION 3.22)
project(my_app LANGUAGES CXX)

find_package(Sabercat 1.0 CONFIG REQUIRED)

add_executable(my_app main.cpp)
target_link_libraries(my_app PRIVATE Sabercat::sabercat)
sabercat_copy_runtime_dlls(my_app)
```

`find_package(Sabercat 1.0 CONFIG REQUIRED)` finds the package on
`CMAKE_PREFIX_PATH`. You can also pass `-DSabercat_DIR=<prefix>/lib/cmake/Sabercat`.

The config sets these variables and functions:

| Name | Meaning |
|---|---|
| `Sabercat::sabercat` | The C++ library |
| `Sabercat::sabercat_c` | The C ABI shared library |
| `SABERCAT_SHADER_DIR` | The directory with the compiled `*.comp.spv` |
| `sabercat_copy_runtime_dlls(<target>)` | Copies the DLLs an executable needs beside it after each build |

Link only the two `Sabercat::` targets. The decoder libraries behind them are
imported targets that the config defines for you.

## Targets

| Target | Kind | Link it when you use |
|---|---|---|
| `Sabercat::sabercat` | Static library | The C++ API (`sabercat/renderer.h` and the other headers) |
| `Sabercat::sabercat_c` | Shared library, `sabercat_c.dll` | The C API (`sabercat/c_api.h`) |

A C++ application that does not need the C ABI links `Sabercat::sabercat`
only. Both targets pull in what they need, including Vulkan.

## Shader directory

The renderer does not search for its shaders. The caller passes the directory to
`Renderer::initialize()` (C++) or `SabercatRendererDesc::shader_dir` (C).

For a consumer, the simplest choice is the value of the CMake variable.

```cmake
target_compile_definitions(my_app PRIVATE
    MY_APP_SHADER_DIR="${SABERCAT_SHADER_DIR}")
```

```cpp
renderer.initialize({}, /*enableValidation=*/false, MY_APP_SHADER_DIR);
```

Two cases need extra care:

- A relocated package. Pass a path that is correct on the machine that runs
  the application. Or read the directory from your own setting, or from an
  environment variable of your own choice.
- The Sabercat apps. `sabercat_viewer.exe` and `sabercat_studio.exe` look for
  `..\..\share\sabercat\shaders` relative to their own executable. They
  also honour the `SABERCAT_SHADER_DIR` environment variable, which overrides
  that lookup.

An empty shader directory skips the forward pass. `forward()` and `present()`
then do nothing. This is useful for tests that only touch the scene API.

## Runtime DLLs

Windows finds DLLs next to the executable, or on `PATH`. An application that
links only `Sabercat::sabercat` needs the vendored zlib DLL. `sabercat_c.dll` is
needed only when it links `Sabercat::sabercat_c`. A missing DLL stops the
application before `main()`.

Use `sabercat_copy_runtime_dlls(my_app)` as shown above. It copies the DLLs after
every build, and copies only the files that changed.

To run from another place, either copy the DLLs yourself, or put the package's
`bin\<Config>` directory on `PATH`.

## Version compatibility

`find_package(Sabercat 1.0 CONFIG)` accepts any 1.x release. Patch and minor
releases keep the API, the C ABI and the on-disk formats compatible. A new
major version is required for an incompatible change. Check the release notes
before you move to a new major version.

The C ABI has its own version, `SABERCAT_C_API_VERSION` (1). Additive changes
keep it at 1. Only a change that breaks existing C callers raises it. See
[c-api.md](c-api.md).

## Option 2: the source tree

Use the source tree when you need to change Sabercat, or when a package is not
available for your platform. Check out the tag `v<version>`, or the `stable`
branch.

Add it to your build with `add_subdirectory`:

```cmake
set(SABERCAT_BUILD_APPS OFF CACHE BOOL "" FORCE)
set(SABERCAT_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SABERCAT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(path/to/Sabercat)

target_link_libraries(my_app PRIVATE Sabercat::sabercat)
```

When Sabercat is not the top-level project, its apps, examples and tests are
off by default. The options above are the explicit form. The targets and the
`SABERCAT_SHADER_DIR` variable are the same as in the package. The build fetches
its third-party dependencies with `FetchContent`. [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)
lists them.

To build against the source tree without a checkout, clone the repository:

```
git clone https://github.com/KashyapRajpal/Sabercat.git
git -C Sabercat checkout v<version>
```

Building the apps and tests from source is described in [building.md](building.md).

## Checking a package

On a new machine, or after a driver update, run the self-test from the unpacked
package. It runs a few packaged tests against this package's own DLLs and
shaders.

```
powershell -ExecutionPolicy Bypass -File <prefix>\selftest\selftest.ps1
```

The run fails on a non-zero exit code or on any `[validation]` line. It needs a
GPU with Vulkan 1.3.
