# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Hello-triangle example: validates GLFW + Vulkan + Slang->SPIR-V end to end.

# Locate the Slang compiler shipped with the Vulkan SDK.
find_program(SLANGC
    NAMES slangc
    HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin"
    REQUIRED
)

set(SHADER_SRC ${CMAKE_CURRENT_SOURCE_DIR}/shader.slang)
set(VERT_SPV ${CMAKE_CURRENT_BINARY_DIR}/shader.vert.spv)
set(FRAG_SPV ${CMAKE_CURRENT_BINARY_DIR}/shader.frag.spv)

# -fvk-use-entrypoint-name preserves the original entry-point names in the
# SPIR-V (otherwise Slang renames them to "main"), so the pipeline can request
# "vertexMain" / "fragmentMain" by name.
add_custom_command(
    OUTPUT ${VERT_SPV}
    COMMAND ${SLANGC} ${SHADER_SRC} -target spirv -fvk-use-entrypoint-name -stage vertex -entry vertexMain -o ${VERT_SPV}
    DEPENDS ${SHADER_SRC}
    COMMENT "Compiling vertex shader (Slang -> SPIR-V)"
    VERBATIM
)

add_custom_command(
    OUTPUT ${FRAG_SPV}
    COMMAND ${SLANGC} ${SHADER_SRC} -target spirv -fvk-use-entrypoint-name -stage fragment -entry fragmentMain -o ${FRAG_SPV}
    DEPENDS ${SHADER_SRC}
    COMMENT "Compiling fragment shader (Slang -> SPIR-V)"
    VERBATIM
)

add_custom_target(sabercat_triangle_shaders DEPENDS ${VERT_SPV} ${FRAG_SPV})

add_executable(sabercat_triangle main.cpp)
add_dependencies(sabercat_triangle sabercat_triangle_shaders)

target_link_libraries(sabercat_triangle PRIVATE Vulkan::Vulkan glfw)

# Tell the app where to find the compiled SPIR-V at runtime.
target_compile_definitions(sabercat_triangle PRIVATE
    SHADER_DIR="${CMAKE_CURRENT_BINARY_DIR}"
)
