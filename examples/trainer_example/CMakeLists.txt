# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Trainer example: a differentiable Gaussian-splat fitting loop driven by the
# C ABI (sabercat_c) and raw Vulkan. Three Slang kernels (activate, loss, adam)
# are compiled to SPIR-V here; the host program is main.cpp.

find_program(SLANGC
    NAMES slangc
    HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin"
    REQUIRED
)

# One kernel per file: each pipeline names its entry point "main".
set(TRAINER_KERNELS trainer_activate trainer_loss trainer_adam)
set(TRAINER_SPV "")
foreach(_kernel IN LISTS TRAINER_KERNELS)
    set(_src ${CMAKE_CURRENT_SOURCE_DIR}/${_kernel}.slang)
    set(_spv ${CMAKE_CURRENT_BINARY_DIR}/${_kernel}.comp.spv)
    add_custom_command(
        OUTPUT ${_spv}
        COMMAND ${SLANGC} ${_src} -target spirv -fvk-use-entrypoint-name -stage compute -entry main -o ${_spv}
        DEPENDS ${_src}
        COMMENT "Compiling trainer kernel ${_kernel} (Slang -> SPIR-V)"
        VERBATIM
    )
    list(APPEND TRAINER_SPV ${_spv})
endforeach()

add_custom_target(sabercat_trainer_example_shaders DEPENDS ${TRAINER_SPV})

add_executable(sabercat_trainer_example main.cpp)
add_dependencies(sabercat_trainer_example sabercat_trainer_example_shaders sabercat_shaders)
target_link_libraries(sabercat_trainer_example PRIVATE sabercat_c Vulkan::Vulkan)

# SHADER_DIR: Sabercat's own SPIR-V (the forward/backward kernels, via
# testShaderDir()). TRAINER_SPV_DIR: this example's three kernels.
target_compile_definitions(sabercat_trainer_example PRIVATE
    SHADER_DIR="${SABERCAT_SHADER_DIR}"
    TRAINER_SPV_DIR="${CMAKE_CURRENT_BINARY_DIR}"
)

# The DLL must sit beside the executable (Windows has no rpath).
add_custom_command(TARGET sabercat_trainer_example POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:sabercat_c> $<TARGET_FILE_DIR:sabercat_trainer_example>
)
