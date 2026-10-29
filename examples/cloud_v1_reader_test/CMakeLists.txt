# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Headless test for CloudV1Reader (cloud-v1 format) and its
# loadScene() dispatch. Synthesizes every fixture at runtime; also renders a
# .ply and the cloud-v1 fixture converted from it through Renderer.

add_executable(sabercat_cloud_v1_reader_test main.cpp million_check.cpp)
add_dependencies(sabercat_cloud_v1_reader_test sabercat_shaders)
target_link_libraries(sabercat_cloud_v1_reader_test PRIVATE sabercat)

target_compile_definitions(sabercat_cloud_v1_reader_test PRIVATE
    SHADER_DIR="${SABERCAT_SHADER_DIR}"
)
