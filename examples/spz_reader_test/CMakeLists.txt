# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Headless test for SpzReader + the loadScene() dispatch.
#
# Unlike splat_reader_test, this one reads REAL files: external/spz/samples ships
# two Niantic-authored .spz scenes. That is deliberate -- .spz decoding is
# delegated to the reference implementation, so the only thing worth testing is
# the adapter onto the canonical contract, and the only honest way to test it is
# against files this project did not write. The samples live in the submodule
# that must be checked out for the build to work at all, so unlike assets/ they
# are always present when this test can run.
add_executable(sabercat_spz_reader_test main.cpp)

# Links spz directly: the test recomputes the expected canonical values from
# spz's own unpacked output, so it needs the same API SpzReader consumes.
target_link_libraries(sabercat_spz_reader_test PRIVATE sabercat spz::spz)

# zlib, to strip the gzip wrapper off a v3 file and build the raw NGSP fixture
# independently of SpzReader's own handling.
target_link_libraries(sabercat_spz_reader_test PRIVATE ${_sabercat_zlib_build})

target_compile_definitions(sabercat_spz_reader_test PRIVATE
    SABERCAT_SPZ_SAMPLE_DIR="${CMAKE_SOURCE_DIR}/external/spz/samples")
