# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Splatted G-buffer + deferred lighting, verified against the
# colour pass and a CPU port of the shading.
add_executable(sabercat_relight_test main.cpp)
add_dependencies(sabercat_relight_test sabercat_shaders)
target_link_libraries(sabercat_relight_test PRIVATE sabercat)

target_compile_definitions(sabercat_relight_test PRIVATE
    SHADER_DIR="${SABERCAT_SHADER_DIR}"
)
