# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Forward-output regression test: renders fixed, seeded scenes and compares a
# hash of the exact output bytes with stored constants (see main.cpp).

add_executable(sabercat_forward_regression_test main.cpp)
add_dependencies(sabercat_forward_regression_test sabercat_shaders)
target_link_libraries(sabercat_forward_regression_test PRIVATE sabercat)

target_compile_definitions(sabercat_forward_regression_test PRIVATE
    SHADER_DIR="${SABERCAT_SHADER_DIR}"
)
