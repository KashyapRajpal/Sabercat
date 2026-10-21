# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

add_executable(sabercat_dataset_capture main.cpp)
add_dependencies(sabercat_dataset_capture sabercat_shaders)
target_link_libraries(sabercat_dataset_capture PRIVATE sabercat)

target_compile_definitions(sabercat_dataset_capture PRIVATE
    SHADER_DIR="${SABERCAT_SHADER_DIR}"
)
