# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Smallest real program: load a scene, render one frame headless, write a PPM.
# Not a CTest test; run it by hand (see main.cpp for the arguments).

add_executable(sabercat_hello_sabercat main.cpp)
add_dependencies(sabercat_hello_sabercat sabercat_shaders)
target_link_libraries(sabercat_hello_sabercat PRIVATE sabercat)

target_compile_definitions(sabercat_hello_sabercat PRIVATE
    SHADER_DIR="${SABERCAT_SHADER_DIR}"
)
