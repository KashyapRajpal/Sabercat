# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Headless verification harness for ForwardPass: a tiny synthetic
# Gaussian cloud + fixed camera, checked stage by stage against an
# independent C++ reference implementation of the shader math (mirrors how
# gaussian_cloud_test recomputes expected values rather than hardcoding
# decimal literals).

add_executable(sabercat_forward_test main.cpp)
add_dependencies(sabercat_forward_test sabercat_shaders)
target_link_libraries(sabercat_forward_test PRIVATE sabercat)

target_compile_definitions(sabercat_forward_test PRIVATE
    SHADER_DIR="${SABERCAT_SHADER_DIR}"
)
