# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Finite-difference gradient check of the backward pass against an
# independent double-precision CPU forward (frozen-branch central
# differences), with compensation on/off and an SH degree cap.
add_executable(sabercat_backward_test main.cpp)
add_dependencies(sabercat_backward_test sabercat_shaders)
target_link_libraries(sabercat_backward_test PRIVATE sabercat)

target_compile_definitions(sabercat_backward_test PRIVATE
    SHADER_DIR="${SABERCAT_SHADER_DIR}"
)
