# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# SH band-fade continuity (a camera dolly across the
# degree thresholds, hard switches vs fade), fade-off bit-identity, and the
# DebugView::SHDegree output against a C++ reference.
add_executable(sabercat_sh_lod_test main.cpp)
add_dependencies(sabercat_sh_lod_test sabercat_shaders)
target_link_libraries(sabercat_sh_lod_test PRIVATE sabercat)

target_compile_definitions(sabercat_sh_lod_test PRIVATE
    SHADER_DIR="${SABERCAT_SHADER_DIR}"
)
