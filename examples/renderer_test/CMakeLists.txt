# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Renderer-level contract checks: knobs, flip, debug views and scene bounds,
# driven through the public Renderer API and checked numerically.

add_executable(sabercat_renderer_test main.cpp)
add_dependencies(sabercat_renderer_test sabercat_shaders)
target_link_libraries(sabercat_renderer_test PRIVATE sabercat)

target_compile_definitions(sabercat_renderer_test PRIVATE
    SHADER_DIR="${SABERCAT_SHADER_DIR}"
)
