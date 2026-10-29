# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# PCSS shadows over splats, on a scene with a known answer.
add_executable(sabercat_shadow_test main.cpp)
add_dependencies(sabercat_shadow_test sabercat_shaders)
target_link_libraries(sabercat_shadow_test PRIVATE sabercat)

target_compile_definitions(sabercat_shadow_test PRIVATE
    SHADER_DIR="${SABERCAT_SHADER_DIR}"
)
