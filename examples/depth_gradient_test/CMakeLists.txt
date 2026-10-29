# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# The depth gradient through the Renderer API: backward(image, depth) and its
# BufferView overload, against the color-only path, the layout-None renderer,
# and freshly built renderers after a resize or a count change.
add_executable(sabercat_depth_gradient_test main.cpp)
add_dependencies(sabercat_depth_gradient_test sabercat_shaders)
target_link_libraries(sabercat_depth_gradient_test PRIVATE sabercat)

target_compile_definitions(sabercat_depth_gradient_test PRIVATE
    SHADER_DIR="${SABERCAT_SHADER_DIR}"
)
