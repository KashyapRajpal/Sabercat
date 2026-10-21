# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Headless smoke test for the compute pipeline: dispatch Slang kernels over
# Buffers via Device's dedicated compute queue and verify the readbacks.

find_program(SLANGC
    NAMES slangc
    HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin"
    REQUIRED
)

# One kernel per file: the pipeline names its entry point "main".
set(COMPUTE_TEST_KERNELS double add_one affine count_groups)
set(COMPUTE_TEST_SPV "")
foreach(_kernel IN LISTS COMPUTE_TEST_KERNELS)
    set(_src ${CMAKE_CURRENT_SOURCE_DIR}/${_kernel}.slang)
    set(_spv ${CMAKE_CURRENT_BINARY_DIR}/${_kernel}.comp.spv)
    add_custom_command(
        OUTPUT ${_spv}
        COMMAND ${SLANGC} ${_src} -target spirv -fvk-use-entrypoint-name -stage compute -entry main -o ${_spv}
        DEPENDS ${_src}
        COMMENT "Compiling compute shader ${_kernel} (Slang -> SPIR-V)"
        VERBATIM
    )
    list(APPEND COMPUTE_TEST_SPV ${_spv})
endforeach()

add_custom_target(sabercat_compute_test_shaders DEPENDS ${COMPUTE_TEST_SPV})

add_executable(sabercat_compute_test main.cpp)
add_dependencies(sabercat_compute_test sabercat_compute_test_shaders)
target_link_libraries(sabercat_compute_test PRIVATE sabercat)

# Tell the app where to find the compiled SPIR-V at runtime.
target_compile_definitions(sabercat_compute_test PRIVATE
    SHADER_DIR="${CMAKE_CURRENT_BINARY_DIR}"
)
