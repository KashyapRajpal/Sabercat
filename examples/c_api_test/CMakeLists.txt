# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# C ABI round trip. Compiled as C (main.c) against the sabercat_c
# SHARED library, so it crosses the same binary boundary a ctypes/cffi binding
# would.
add_executable(sabercat_c_api_test main.c)
set_target_properties(sabercat_c_api_test PROPERTIES C_STANDARD 99 C_STANDARD_REQUIRED ON)
add_dependencies(sabercat_c_api_test sabercat_shaders)
target_link_libraries(sabercat_c_api_test PRIVATE sabercat_c)

target_compile_definitions(sabercat_c_api_test PRIVATE
    SHADER_DIR="${SABERCAT_SHADER_DIR}"
)

# The DLL must sit beside the executable (Windows has no rpath).
add_custom_command(TARGET sabercat_c_api_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:sabercat_c> $<TARGET_FILE_DIR:sabercat_c_api_test>
)
