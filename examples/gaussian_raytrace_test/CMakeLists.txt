# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# The Gaussian ray-tracing core -- AABBs, BLAS/TLAS build + refit,
# and gaussianTransmittance traced on the GPU against a CPU brute force over
# every Gaussian. trace_rays.slang is a test kernel (one thread per ray); it
# compiles with the ray-query flags shaders/CMakeLists.txt defines.

find_program(SLANGC
    NAMES slangc
    HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin"
    REQUIRED
)

set(SHADER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/trace_rays.slang)
set(SHADER_SPV ${CMAKE_CURRENT_BINARY_DIR}/trace_rays.comp.spv)

add_custom_command(
    OUTPUT ${SHADER_SPV}
    COMMAND ${SLANGC} ${SHADER_SRC} -I ${SABERCAT_SHADER_INCLUDE_DIR} -target spirv
            ${SABERCAT_RAY_QUERY_SLANGC_FLAGS} -fvk-use-entrypoint-name -stage compute -entry main
            -o ${SHADER_SPV}
    DEPENDS ${SHADER_SRC} ${SABERCAT_SHADER_HEADER_FILES}
    COMMENT "Compiling trace_rays.slang (Slang -> SPIR-V, ray query)"
    VERBATIM
)
add_custom_target(sabercat_gaussian_raytrace_test_shaders DEPENDS ${SHADER_SPV})

add_executable(sabercat_gaussian_raytrace_test main.cpp)
add_dependencies(sabercat_gaussian_raytrace_test sabercat_shaders
                 sabercat_gaussian_raytrace_test_shaders)
target_link_libraries(sabercat_gaussian_raytrace_test PRIVATE sabercat)

target_compile_definitions(sabercat_gaussian_raytrace_test PRIVATE
    SHADER_DIR="${SABERCAT_SHADER_DIR}"
    TEST_SHADER_DIR="${CMAKE_CURRENT_BINARY_DIR}"
    ASSET_DIR="${PROJECT_SOURCE_DIR}/assets"
)
