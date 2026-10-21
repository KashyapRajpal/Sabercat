# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Feature-reel demo: renders a fixed sequence of Sabercat's rendering features
# headless to PPM frames (see tools/make_demo.ps1 for the MP4/GIF encode).
# Not a test: not registered with CTest and not in tools/release.ps1.
add_executable(sabercat_demo_reel main.cpp)
add_dependencies(sabercat_demo_reel sabercat_shaders)
target_link_libraries(sabercat_demo_reel PRIVATE sabercat)

target_compile_definitions(sabercat_demo_reel PRIVATE
    SHADER_DIR="${SABERCAT_SHADER_DIR}"
)
