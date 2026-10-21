# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Numeric diagnostic for a real scene file: reports what the loader produced and
# what forward_project actually did with it. Exists because screen capture of
# Vulkan content is unreliable in this environment, so "the render looks wrong"
# has to be turned into numbers before it can be debugged.
add_executable(sabercat_scene_diag main.cpp)
add_dependencies(sabercat_scene_diag sabercat_shaders)
target_link_libraries(sabercat_scene_diag PRIVATE sabercat)

target_compile_definitions(sabercat_scene_diag PRIVATE
    SHADER_DIR="${SABERCAT_SHADER_DIR}"
)
