# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Sabercat Studio -- the editing application (docs/studio-design.md). It owns the
# window (GLFW) and the Vulkan device and lends the device to the
# renderer. The imgui target and the shared app code (apps/common) are defined
# one level up.

# The document model (project, layers, app state): plain C++ over the renderer,
# with no window or ImGui, so the CPU unit tests link it too.
add_library(sabercat_studio_core STATIC
    studio_state.cpp
)
target_include_directories(sabercat_studio_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(sabercat_studio_core PUBLIC sabercat_app_core)
target_link_libraries(sabercat_studio_core PRIVATE nlohmann_json::nlohmann_json)

add_executable(sabercat_studio
    main.cpp
    selftest.cpp
    studio_ui.cpp
)
add_dependencies(sabercat_studio sabercat_shaders)
sabercat_add_app_resources(sabercat_studio "Sabercat Studio")

# Like the viewer, the app injects its windowing bits; the library stays
# windowing-agnostic. nlohmann/json holds the app-state block of a project.
target_link_libraries(sabercat_studio PRIVATE
    sabercat
    sabercat_app_common
    sabercat_studio_core
    glfw
    imgui
    nlohmann_json::nlohmann_json
)

# bin/<Config>/, beside sabercat_viewer, so Debug and Release share a prefix and
# the Studio finds ../../share/sabercat/shaders at run time.
install(TARGETS sabercat_studio DESTINATION ${CMAKE_INSTALL_BINDIR}/$<CONFIG>)
