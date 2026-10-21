# Contributing to Sabercat

Thank you for your interest in Sabercat. Bug reports, fixes, documentation and new features are all welcome.

Please follow the [Code of Conduct](CODE_OF_CONDUCT.md) in all project spaces.

## Getting started

1. Fork the repository on GitHub: <https://github.com/KashyapRajpal/Sabercat>.
2. Clone your fork with its submodules:

   ```powershell
   git clone --recurse-submodules https://github.com/<your-user>/Sabercat.git
   cd Sabercat
   ```

3. Create a branch for your change:

   ```powershell
   git checkout -b feat/short-description
   ```

## Build

You need the requirements listed in the [README](README.md#requirements): Windows 10/11 x64, Visual Studio 2022, CMake 3.22 or newer, and the Vulkan SDK 1.3 or newer.

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Debug --parallel
```

Debug builds turn on the Vulkan validation layer. Keep it on while you work. A `[validation]` message is a bug to fix, not noise to ignore.

## Tests

CTest runs two kinds of test. The `cpu` label (`tests/unit`) is CPU-only and runs in continuous integration. The `gpu` label is the GPU programs in `examples/*_test` plus `studio_selftest`, which need a Vulkan 1.3 GPU, so run them on your own machine:

```powershell
cmake --build build --config Release --parallel
ctest --test-dir build -C Release -L cpu --output-on-failure
ctest --test-dir build -C Release -L gpu --output-on-failure
```

Continuous integration builds the project for Debug and Release, runs the `cpu` tests and checks the install step. It does not run the `gpu` tests, because hosted runners have no GPU. Run them yourself before you open a pull request that touches the renderer, the readers or the writers.

Add a test for new behaviour. A new GPU test is an `examples/<name>_test` program that exits with 0 on success, plus an entry in `tests/integration/CMakeLists.txt`.

## Coding standards

- **Language**: C++20. The C ABI in `include/sabercat/c_api.h` stays plain C-compatible.
- **Formatting**: run `clang-format` with the `.clang-format` file at the repository root on the C++ files you change.
- **License header**: new source files start with the same two-line SPDX header as the existing files (`// Copyright 2026 Kashyap Rajpal` and `// SPDX-License-Identifier: Apache-2.0`, or `#` comments in CMake, PowerShell and YAML).
- **Naming**:
  - Types: `PascalCase`.
  - Functions and methods: `camelCase`.
  - Constants: `kPascalCase` for `constexpr` values in the library. Macros and environment names use `UPPER_SNAKE_CASE`.
  - Private member variables: `m_camelCase`.
- **Comments**: explain why the code is the way it is. Do not restate what the code does.
- **Structs**: group fields by size so that the struct has no padding. ABI structs append new fields into the tail padding and keep their size.
- **Warnings**: do not add new compiler warnings.

## Commit messages

Use the repository's style: a category in brackets, a colon, and a short summary in the imperative or as a plain noun phrase. Add a body when the reason is not obvious from the summary.

```
[docs]: describe the cloud-v1 manifest fields

The formats guide omitted the relight sidecar path.
```

Categories: `feat`, `fix`, `refactor`, `docs`, `test`, `perf`, `chore`.

Keep commits small and focused. One concern per commit is easier to review and to revert.

## Pull requests

1. Describe what the change does and why.
2. Link related issues. Use `Fixes #123` in the description to close an issue on merge.
3. Keep the change focused. Split unrelated work into separate pull requests.
4. Update the documentation in `docs/` and the `README.md` for user-facing changes, and add an entry to `CHANGELOG.md` under an "Unreleased" heading.
5. Make sure the Debug and Release builds pass and that the GPU tests pass on your machine.

## AI-assisted contributions

AI coding tools may be used to write contributions. If you used one:

- say so in the pull request description, and
- review, test and understand every line you submit. You are responsible for the code, as the author would be for any other code.

Disclosed and reviewed AI-assisted contributions are welcome. Undisclosed or untested generated code is not.

## Reporting bugs

Open an issue and include:

- a description of the problem, and what you expected to happen,
- steps to reproduce it, using a scene and a command line if possible,
- your GPU model, driver version, and Vulkan SDK version,
- the console output, including any `[validation]` lines.

For security problems, do not open a public issue. Follow [SECURITY.md](SECURITY.md).

## Feature requests

Open an issue that describes the use case, the behaviour you want, and any alternatives you considered. [ROADMAP.md](ROADMAP.md) lists work that is already planned.

## License of contributions

Sabercat is licensed under the Apache License, Version 2.0. Unless you state otherwise, any contribution you submit is licensed under the same terms, as described in section 5 of the license. See [LICENSE](LICENSE).

The Sabercat logo files (`docs/media/sabercat-logo*.png`, `docs/media/social-preview.png`, `apps/common/resources/sabercat.*`) are excluded from the Apache licence and are not contributed under it; see [NOTICE](NOTICE).
