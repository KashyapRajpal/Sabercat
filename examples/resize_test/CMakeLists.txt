# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Resizable Gaussian storage (reserve + count) and the shared-queue
# synchronization contract, through the public Renderer API -- the optimizer's
# tail writes and gradient readback run on its own command buffers on
# Sabercat's compute queue.
add_executable(sabercat_resize_test main.cpp)
add_dependencies(sabercat_resize_test sabercat_shaders)
target_link_libraries(sabercat_resize_test PRIVATE sabercat)

target_compile_definitions(sabercat_resize_test PRIVATE
    SHADER_DIR="${SABERCAT_SHADER_DIR}"
)
