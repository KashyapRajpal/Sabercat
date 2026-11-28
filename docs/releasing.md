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
  version. The build modes (`-PushTag`, `-Publish` and `-CiBuild`) refuse to run
  if the README does not link both of the release's zips.
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

0. **Prepare.**

   Commit the batch of work on `main`. Every commit is clean and builds. Then,
   in their own commit, bump the version (see Versioning), update CHANGELOG.md
   and update the README download links to the new version.

   A dry run is optional but recommended before step 1:

   ```powershell
   powershell -ExecutionPolicy Bypass -File tools\release.ps1
   ```

   The dry run configures `out/release-mapped`, builds Debug and Release, and runs
   all 36 GPU tests in both configurations. Any non-zero exit, or any
   `[validation]` line, stops it. It then installs both configurations into
   one prefix, zips the result into `dist\`, unpacks the zip into a fresh
   directory and runs the packaged self-test from there. Logs are written to
   `out/release-mapped/release-logs/`.

   To package uncommitted work for local inspection, add `-AllowDirty`. The zip
   is then named `...-dirty.zip`, and the release modes (`-PushTag`, `-Publish`,
   `-CiBuild`) refuse it.

   A beta is the same dry run with a suffix:

   ```powershell
   powershell -ExecutionPolicy Bypass -File tools\release.ps1 -VersionSuffix beta.1
   ```

   Release it the same way (steps 1 to 4). The tag and both packages carry
   `1.0.0-beta.1`, and the GitHub Release is a pre-release.

1. **Check and tag: `tools\release.ps1 -PushTag`** (optional `-VersionSuffix beta.1`).

   This is the full local release check. It stops at the first failed check.
   It checks:
   - the tree is clean (unless `-AllowDirty`, which this mode refuses anyway);
   - the tag is new, both locally and on the remote;
   - `stable` can fast-forward to `HEAD`;
   - the README links both zips;
   - it builds Debug and Release from a mapped drive letter (see
     [Build paths and the leak gate](#build-paths-and-the-leak-gate));
   - the full GPU test suite passes in both configurations, with zero
     `[validation]` lines;
   - the leak gate passes on both zips;
   - the packaged self-test and the apps self-test pass.

   Then it creates the annotated tag (the message lists the commits since the
   previous release) and pushes **only the tag**. It does not push `main` or
   `stable`, and it creates no GitHub Release. The local zips stay in `dist\`
   for step 3. If the tag push fails, the local tag is removed again.

2. **Build in CI (automatic on the tag push).**

   `.github/workflows/release.yml` runs on `windows-2022` with
   `tools\release.ps1 -CiBuild -ExpectTag <tag>`. The runner has no GPU, so it
   runs the CPU tests and not the GPU tests or the self-tests. The workflow
   writes `SHA256SUMS.txt`, attests both zips and `SHA256SUMS.txt` (build
   provenance), and creates or refreshes a **draft** GitHub Release with the
   files attached. A stale draft's assets are replaced. A published release is
   never touched: the job stops with an "already published" message.

   CI never publishes.

3. **Verify the draft: `tools\release.ps1 -VerifyCi -Tag vX.Y.Z[-suffix]`.**

   - Finds the draft through the releases API (the by-tag lookup does not
     return drafts). It fails if there is no draft, or if the release is
     already published.
   - Downloads the assets to a fresh temp folder outside the repository.
   - Checks `SHA256SUMS.txt` against the zips.
   - Runs `gh attestation verify` for each zip and for `SHA256SUMS.txt`.
   - Cross-checks the CI zips with the local packages from step 1, using
     `tools\compare_packages.ps1`:
     - the same file list;
     - identical contents for non-PE files (text files compare after normalizing
       CRLF to LF, and a difference in line endings only is reported but accepted;
       the generated `cmake_policy(VERSION ...)` line in
       `lib\cmake\Sabercat\SabercatTargets.cmake` may differ, because it reflects
       the CMake version of the build machine, and that difference is accepted;
       any other difference in that file fails);
     - for `.exe` and `.dll`, identical import tables and version info; byte
       differences alone are accepted;
     - manifest: the same version and commit, `ci=true` and `gpuTested=false`
       for CI, `ci=false` and `gpuTested=true` for local, the tag matches, and
       the tree was not dirty.
     When running the compare script by hand, `-Strict` is also available; it
     compares bytes exactly (no line-ending normalization) and also fails on any
     differing binary. Neither exception (line endings, `cmake_policy`) applies
     under `-Strict`.
   - Runs the packaged self-test and the apps self-test on the CI zips (this
     needs a GPU).

   **Known gap:** the full GPU test suite runs from the build tree, so it runs
   on the locally built bits in step 1, not on the CI package.

   Without `-Publish`, nothing remote changes.

4. **Publish the draft: `tools\release.ps1 -VerifyCi -Tag vX.Y.Z -Publish`.**

   This runs the same checks as step 3, then publishes the draft (its notes
   are replaced by the tag message, which lists the commits, because the CI
   notes say "draft") and fast-forwards `stable`. `stable` moves for final releases only; a
   pre-release does not move it. If `-StableDir` was given, it is refreshed.

### Fallback: local publish

`tools\release.ps1 -Publish` alone still builds, tests, tags, pushes `main`,
`stable` and the tag, and creates the GitHub Release with the local zips. Use
it when CI is unavailable.

Because the tag push still triggers `release.yml`, the workflow builds and then
stops with the message that the release is already published. That failure is
expected and harmless.

### Why a tag build can fail

- The tag does not match the version in `CMakeLists.txt`. `-ExpectTag`
  stops the build.
- The README in the tagged commit does not link both zips.
- The build finds a dirty tree or a runner image it does not support.

### Manual verification of a downloaded release

Verify the provenance of a zip with `gh attestation verify <zip> --repo <owner>/<repo>`,
and compare its SHA-256 with the matching line in `SHA256SUMS.txt`.

## Script options

| Option | Default | Meaning |
|---|---|---|
| `-PushTag` | off | Full local checks (step 1), then create the annotated tag and push only the tag |
| `-Publish` | off | Legacy. Alone: local publish (build, test, tag, push `main`, `stable` and the tag, create the release with the local zips). With `-VerifyCi`: publish the verified draft |
| `-VerifyCi` | off | Check the CI draft for `-Tag` (step 3). With `-Publish`, publish it (step 4) |
| `-Tag` | empty | The tag whose draft `-VerifyCi` checks. Required with `-VerifyCi` |
| `-CiBuild` | off | CI use only. Builds and packages without a GPU: CPU tests, no GPU tests and no self-tests |
| `-ExpectTag` | empty | CI use with `-CiBuild`. The build fails unless the tag matches the version in `CMakeLists.txt` and the suffix |
| `-AllowDirty` | off | Allow an uncommitted tree (dry runs only; the zip is marked `-dirty`) |
| `-StableDir` | empty | Unpacked package location to refresh on publish. Empty skips the refresh |
| `-BuildDir` | `out/release-mapped` | Build tree (git-ignored), reused between releases. Only ever configured through the mapped drive (see below) |
| `-Remote` | `origin` | Git remote to fetch from and push to |
| `-VersionSuffix` | empty | Pre-release suffix such as `beta.1`. Sets the tag, the package names and the pre-release flag |

## Build paths and the leak gate

A package must not reveal where the repository lives on the build machine.
Compilers record absolute paths (`__FILE__` in error messages, PDB paths,
`FetchContent` sources under the build tree), so the script does not build
from the repository's real location:

- After the preconditions it maps the repository root to the first free
  drive letter, searching downward from the end of the alphabet
  (`subst <letter>: <repo>`), and runs configure, build, test, install and zip
  from `<letter>:\`. Every recorded path is then `<letter>:\src\...`, which says
  nothing about the machine. The letter itself is irrelevant. git and `gh`
  still run against the real checkout.
- The mapping is removed at the end (`subst <letter>: /D`), also when the
  release fails. If the script is killed hard, run `subst` to see a leftover
  mapping and `subst <letter>: /D` to remove it.
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
| `manifest.json` | Package name, version, `tag` (set by `-PushTag`, `-Publish` and CI builds; empty otherwise), `ci` (true for CI builds), `gpuTested` (true when the GPU suite ran on this build), commit SHA, branch, dirty flag, build time (UTC), platform, configurations, CMake version, `vulkanSdk` (the SDK path from `$env:VULKAN_SDK`, not a version), and the result of every test |

The install step checks that the viewer, the Studio executable, the comparer,
the C DLL, the shader config, the package config and one compiled shader are
present, and fails the release if any is missing.

### The apps package

`sabercat-<version>-win64-apps.zip` is for people who only want to run
the apps. It has one top-level folder, the same name without `.zip`:

| Path | Contents |
|---|---|
| `bin/Release/` | `sabercat_viewer.exe`, `sabercat_studio.exe`, `sabercat_compare.exe`, the DLLs they import (`zlib.dll`, if needed) and the Visual C++ runtime DLLs they import |
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
unset. The comparer runs `sabercat_compare --help` under the same `PATH`. The
viewer has no headless mode, so it is started and must still be running after 8
seconds.

## The tests

The release script runs the 36 GPU tests: 34 programs from `examples/` (each built as
`sabercat_<name>.exe`) plus `studio_selftest` and `compare_selftest`. They are registered with CTest in `tests/integration`.
Each one runs in Debug and Release and must exit 0 with no `[validation]`
line in its output.

They need a GPU with Vulkan 1.3. They are not run in CI. The CI workflow
(`.github/workflows/build.yml`) builds, runs the `cpu` tests and checks the install step.
`-PushTag` runs the full suite (step 1). To run it by hand on a machine with a GPU:

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
- The GitHub CLI (`gh`), logged in with `gh auth login`. The account needs
  rights to read and edit releases (drafts included) and to create releases on
  the repository, and the token must be able to verify attestations.
- A git remote named `origin` (or the name given by `-Remote`).

## If something fails

- **Dry run or `-PushTag` fails at a test:** the log is in
  `out/release-mapped/release-logs/`. Nothing is packaged, and nothing is tagged
  or pushed.
- **`-PushTag` or `-Publish` refuses the tag:** the version was not bumped. Bump it
  in its own commit.
- **`-PushTag` or `-Publish` refuses `stable`:** `stable` is not an ancestor of `HEAD`.
  `stable` must be an ancestor of the commit being released. Release from
  `main` after its history includes the last released commit.
- **`-PushTag` push failed:** the local tag is removed again. Nothing remote
  changed. Fix the cause (usually the remote or the network) and run it again.
- **`-VerifyCi` cannot find the draft:** either the workflow is still running or
  it failed (check the run for the tag on GitHub Actions), or the release is
  already published, in which case there is nothing to verify (see the
  fallback above).
- **`gh release create` fails during `-Publish` after the tag is pushed:** the
  tag and branch are already on the remote. Rerun the command the script prints,
  with the zip and the release notes it names.
