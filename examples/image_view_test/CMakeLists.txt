# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# The zero-copy image view (Renderer::imageView()) -- the optimizer
# reads the rendered image and builds dL/dImage on Sabercat's compute queue
# without a host download.
add_executable(sabercat_image_view_test main.cpp)
add_dependencies(sabercat_image_view_test sabercat_shaders)
target_link_libraries(sabercat_image_view_test PRIVATE sabercat)

target_compile_definitions(sabercat_image_view_test PRIVATE
    SHADER_DIR="${SABERCAT_SHADER_DIR}"
)
