# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Renderer::renderDebugView()/readDebugView() -- the debug views rendered headless.
add_executable(sabercat_debug_view_test main.cpp)
add_dependencies(sabercat_debug_view_test sabercat_shaders)
target_link_libraries(sabercat_debug_view_test PRIVATE sabercat)

target_compile_definitions(sabercat_debug_view_test PRIVATE
    SHADER_DIR="${SABERCAT_SHADER_DIR}"
)
