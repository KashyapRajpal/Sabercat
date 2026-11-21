# Releasing Sabercat

This is the maintainer guide. It covers how a release is built, tested,
packaged and published. For how consumers use a release, see
[consuming.md](consuming.md).

## Model

- `main` is where development happens.
- `stable` only moves by **fast-forward**, to a `main` commit that
  `tools/release.ps1` built and tested. It is never committed to directly.
- Each promotion gets an annotated tag `v<version>`. Its message lists the
  commits since the previous release.
- Each tag gets a GitHub Release with two zips attached:
  `sabercat-<version>-win64.zip` (the full package) and
  `sabercat-<version>-win64-apps.zip` (the apps for testers; see
  [The apps package](#the-apps-package)). The names carry no commit hash, so
  the README in the released commit can link them directly; the hash is in the
  tag and in `manifest.json`.
- Before a release, update the Download links in README.md to the new
  version. `-Publish` refuses to run if the README does not link both of the
  release's zips.
- A beta or other pre-release is tagged `v<version>-<suffix>`, for example
  `v1.0.0-beta.1`. Its GitHub Release is marked as a pre-release. A
  pre-release does not move `stable` and does not refresh `-StableDir`:
  those track final releases only.
- Release notes point at the matching section of
  [CHANGELOG.md](../CHANGELOG.md). Update that section in the release
  commit.
- `-StableDir` is optional. When it is given, the script also keeps an
  unpacked copy of the latest package at that path, swapped in by rename, so a
  consumer never sees a half-written prefix. Without it, the zip on the GitHub
  Release is the only copy.

## Versioning

The version is `project(Sabercat VERSION x.y.z)` in `CMakeLists.txt`. Bump it
in its own commit, for example `[chore]: release v1.0.1`.

- **Patch and minor (`1.x.y`)**: no change to the consumer-visible contract.
  The package config is `SameMajorVersion`, so `find_package(Sabercat 1.0)`
  keeps working.
- **Major (`2.0.0`)**: a contract or ABI break. Consumers must raise their
  request. Say what changed in the commit message and in CHANGELOG.md.

A pre-release keeps the version in `CMakeLists.txt` and adds a suffix at
release time with `-VersionSuffix`, for example `-VersionSuffix beta.1`. The
suffix must be semver pre-release identifiers (`[0-9A-Za-z.-]+`). The window
titles and Help > About show the full version, `1.0.0-beta.1`. The CMake package
version stays `1.0.0`, so `find_package(Sabercat 1.0)` still matches.

The C API has its own version, `SABERCAT_C_API_VERSION`, in
`include/sabercat/c_api.h`. Additive changes keep it.

## Cutting a release

1. Batch the work on `main` and commit it. Every commit is clean and
   builds.
2. Bump the version (see Versioning) and update CHANGELOG.md, in their own
   commit.
3. Dry run:

   ```powershell
   powershell -ExecutionPolicy Bypass -File tools\release.ps1
   ```

   The dry run configures `out/release-mapped`, builds Debug and Release, and runs
   all 30 GPU tests in both configurations. Any non-zero exit, or any
   `[validation]` line, stops it. It then installs both configurations into
   one prefix, zips the result into `dist/`, unpacks the zip into a fresh
   directory and runs the packaged self-test from there. Logs are written to
   `out/release-mapped/release-logs/`.

   To package uncommitted work for local inspection, add `-AllowDirty`. The zip
   is then named `...-dirty.zip`, and publishing is refused.

   A beta is the same dry run with a suffix:

   ```powershell
   powershell -ExecutionPolicy Bypass -File tools\release.ps1 -VersionSuffix beta.1
   ```

   Publish it the same way (below). The tag and both packages carry
   `1.0.0-beta.1`, and the GitHub Release is a pre-release.

4. Publish:

   ```powershell
   powershell -ExecutionPolicy Bypass -File tools\release.ps1 -Publish
   ```

   This repeats the dry run on the committed tree, then:
   - checks that the tag is new and that `stable` can fast-forward;
   - moves `stable` and creates the annotated tag;
   - pushes the current branch, `stable` and the tag to `origin`;
   - runs `gh release create` with both zips (a pre-release when a suffix is set);
   - refreshes `-StableDir`, if it was given.

## Script options

| Option | Default | Meaning |
|---|---|---|
| `-Publish` | off | Promote to `stable`, tag, push, and create the GitHub Release |
| `-AllowDirty` | off | Allow an uncommitted tree (dry runs only; the zip is marked `-dirty`) |
| `-StableDir` | empty | Unpacked package location to refresh on publish. Empty skips the refresh |
| `-BuildDir` | `out/release-mapped` | Build tree (git-ignored), reused between releases. Only ever configured through the mapped drive (see below) |
| `-Remote` | `origin` | Git remote to fetch from and push to |
| `-VersionSuffix` | empty | Pre-release suffix such as `beta.1`. Sets the tag, the package names and the pre-release flag |

## Build paths and the leak gate

A package must not reveal where the repository lives on the build machine.
Compilers record absolute paths (`__FILE__` in error messages, PDB paths,
`FetchContent` sources under the build tree), so the script does not build
from the repository root:

- After the preconditions it maps the repository root to the first free
  drive letter from `Z:` down to `M:` (`subst Z: <repo>`) and runs configure,
  build, test, install and zip from `Z:\`. Every recorded path is then <!-- repo-check: allow -->
  `Z:\src\...`, which says nothing about the machine. The letter itself is <!-- repo-check: allow -->
  irrelevant. git and `gh` still run against the real checkout.
- The mapping is removed at the end (`subst Z: /D`), also when the release
  fails. If the script is killed hard, run `subst` to see a leftover mapping and
  `subst Z: /D` to remove it.
- CMake caches absolute source paths and refuses a tree configured from
  another path, so the default build tree is `out/release-mapped`, which is
  only configured through the mapped drive. If its cache was made from a
  different path or letter, the script discards the tree and starts afresh.
  Use `out/release` for ordinary builds.

The **leak gate** runs on each zip right after it is created, before its
self-test and before anything is published. It reads every file in the zip
(`manifest.json` included) and searches the bytes, as ASCII and as UTF-16LE,
ignoring case, for the real repository root and `%USERPROFILE%`, each with
backslashes, forward slashes and doubled backslashes. Any hit prints the file
and the matched string and fails the release. The bare user name is not
searched for: it is part of the copyright line.

## Repository checks

`tools/check_repo.py` runs in CI on every push and pull request (workflow
`.github/workflows/repo-checks.yml`), and locally with
`python tools/check_repo.py`. It reads every tracked or untracked, non-ignored
text file and reports:

- **header**: a supported source file (C/C++, shader, PowerShell, CMake, YAML,
  JavaScript and Python files, and `CMakeLists.txt`) must carry the copyright
  and `SPDX-License-Identifier: Apache-2.0` lines near its top. Other files,
  such as Markdown and `tools/package/*.txt`, are exempt from this check.
- **local-path**: no Windows drive-letter path and no home-folder path
  (under a `Users` or `home` folder), so a package never reveals the build
  machine. Checked for every text file.
- **secret**: no known API token format (AWS, GitHub, Slack, Google and
  `sk-` keys, PEM private keys) and no hard-coded password, secret, API key or
  access token literal. Checked for every text file.

Binary files are skipped. A line that carries the marker `repo-check: allow`
is exempt from the local-path and secret checks on that line only. Use it for
intentional literals, such as tests. The header check has no exemption.

**Self-test.** `python tools/check_repo.py --self-test` runs built-in fixtures
through the same checks and fails if any result changes. CI runs it before the
repository check, so a broken checker cannot pass silently.

**Secret scan.** CI also runs gitleaks over the full git history with
`gitleaks detect --source . --no-banner --redact`. It uses the official release
binary, pinned to version 8.30.1. The workflow checks the archive against a
pinned sha256 and stops on a mismatch. The `gitleaks-action` is not used
because of its license. To bump gitleaks, take the new version and the sha256
of the `linux_x64` archive from that release's
`gitleaks_<version>_checksums.txt`, and update `GITLEAKS_VERSION` and
`GITLEAKS_SHA256` in `.github/workflows/repo-checks.yml` in one commit.

**GitHub secret scanning.** GitHub's own secret scanning and push protection
are repository settings, separate from this workflow. The repository owner can
enable them under Settings > Code security.

## What the package contains

The zip has one top-level folder, `sabercat-<version>-win64/`:

| Path | Contents |
|---|---|
| `bin/Debug/`, `bin/Release/` | `sabercat_viewer.exe`, `sabercat_studio.exe`, `sabercat_c.dll` |
| `include/sabercat/` | Public headers, including `shaders/config.slangh` |
| `lib/` | Library files and `lib/cmake/Sabercat/` package config files |
| `share/sabercat/shaders/` | Compiled SPIR-V shaders |
| `docs/` | `consuming.md`, `library.md`, `c-api.md`, `formats.md` |
| `README.md`, `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`, `CHANGELOG.md` | At the package root |
| `selftest/` | `selftest.ps1` and four test executables (Release) |
| `manifest.json` | Package name, version, tag (publish builds only), commit SHA, branch, dirty flag, build time (UTC), platform, configurations, CMake version, `vulkanSdk` (the SDK path from `$env:VULKAN_SDK`, not a version), and the result of every test |

The install step checks that the viewer, the Studio executable, the C DLL,
the shader config, the package config and one compiled shader are present,
and fails the release if any is missing.

### The apps package

`sabercat-<version>-win64-apps.zip` is for people who only want to run
the apps. It has one top-level folder, the same name without `.zip`:

| Path | Contents |
|---|---|
| `bin/Release/` | `sabercat_viewer.exe`, `sabercat_studio.exe`, the DLLs they import (`zlib.dll`, if needed) and the Visual C++ runtime DLLs they import |
| `share/sabercat/shaders/` | Compiled SPIR-V shaders. The apps find them two levels above the executable |
| `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`, `CHANGELOG.md` | At the package root |
| `BETA-README.txt` (with a suffix) or `README.txt` | Requirements, how to run, logs, how to report |

It has no headers, libraries, Debug binaries or CMake files.

The Visual C++ runtime is copied from the Visual Studio redistributable
(`VC\Redist\MSVC\<ver>\x64\Microsoft.VC143.CRT`, found with `vswhere`). Debug
runtimes are never shipped.

A dependency audit runs `dumpbin /dependents` on the executables and every DLL
copied in. The release fails if an import is none of these: a DLL in the
package, a Visual C++ runtime DLL, a Windows system DLL (`System32`, or an
`api-ms-win-*` / `ext-ms-*` API set), or `vulkan-1.dll`, which the GPU driver
installs. The release prints the audit table.

The zip is unpacked into a fresh directory and tested there. The Studio
self-test runs with `PATH` set to `System32` alone and `SABERCAT_SHADER_DIR`
unset. The viewer has no headless mode, so it is started and must still be
running after 8 seconds.

## The tests

The release script runs the 30 GPU tests: 29 programs from `examples/` (each built as
`sabercat_<name>.exe`) plus `studio_selftest`. They are registered with CTest in `tests/integration`.
Each one runs in Debug and Release and must exit 0 with no `[validation]`
line in its output.

They need a GPU with Vulkan 1.3. They are not run in CI. The CI workflow
(`.github/workflows/build.yml`) builds, runs the `cpu` tests and checks the install step. So run the tests on a machine
with a GPU before every release:

```powershell
ctest --test-dir out/release -C Release --output-on-failure
```

The self-test in the package runs four of them (`device_test`, `forward_test`,
`relight_test`, `c_api_test`) from the unpacked copy.

## Requirements for a release machine

- Windows 10 or 11, x64, with a Vulkan 1.3 GPU.
- Visual Studio 2022 with the C++ workload.
- CMake 3.22 or later.
- The Vulkan SDK 1.3 or later. It provides `slangc` and the validation layer.
- The GitHub CLI (`gh`), logged in, with rights to create releases on the
  repository.
- A git remote named `origin` (or the name given by `-Remote`).

## If something fails

- **Dry run fails at a test:** the log is in `out/release/release-logs/`.
  Nothing is packaged.
- **`-Publish` refuses the tag:** the version was not bumped. Bump it in its
  own commit.
- **`-Publish` refuses `stable`:** `stable` is not an ancestor of `HEAD`.
  `stable` must be an ancestor of the commit being released. Release from
  `main` after its history includes the last released commit.
- **`gh release create` fails after the tag is pushed:** the tag and branch are
  already on the remote. Rerun the command the script prints, with the zip and
  the release notes it names.
