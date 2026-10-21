# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# v2 Step 12: baked visibility + teacher samples, against CPU references.
add_executable(sabercat_visibility_test main.cpp)
add_dependencies(sabercat_visibility_test sabercat_shaders)
target_link_libraries(sabercat_visibility_test PRIVATE sabercat)

target_compile_definitions(sabercat_visibility_test PRIVATE
    SHADER_DIR="${SABERCAT_SHADER_DIR}"
)
