# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# The per-pixel depth output of a Differentiable forward (DepthLayout,
# Renderer::readDepth() and depthView()), checked against the color image,
# the lit G-buffer and the Depth lighting output.
add_executable(sabercat_depth_output_test main.cpp)
add_dependencies(sabercat_depth_output_test sabercat_shaders)
target_link_libraries(sabercat_depth_output_test PRIVATE sabercat)

target_compile_definitions(sabercat_depth_output_test PRIVATE
    SHADER_DIR="${SABERCAT_SHADER_DIR}"
)
