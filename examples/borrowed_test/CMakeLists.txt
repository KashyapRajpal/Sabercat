# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Borrowed (application-owned) device: the program creates the Vulkan device and the
# canonical Gaussian buffers itself and Sabercat renders/differentiates them
# in place; plus every borrowed-device/buffer requirement being enforced.
add_executable(sabercat_borrowed_test main.cpp)
add_dependencies(sabercat_borrowed_test sabercat_shaders)
target_link_libraries(sabercat_borrowed_test PRIVATE sabercat)

target_compile_definitions(sabercat_borrowed_test PRIVATE
    SHADER_DIR="${SABERCAT_SHADER_DIR}"
)
