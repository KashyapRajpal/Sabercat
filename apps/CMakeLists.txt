# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Applications built on top of the renderer library.
# Each app owns its window/input and links its own windowing library.

# Dear ImGui: the debug UI (view picker, stats). App-level only -- the
# renderer library stays UI-framework-agnostic, so this dependency
# lives entirely in apps/, never in libsabercat. ImGui ships no top-
# level CMakeLists.txt of its own, so FetchContent_MakeAvailable just
# populates the source (no add_subdirectory happens) and a small STATIC
# target is built by hand from exactly the files needed: the core library
# plus the GLFW + Vulkan backends (the apps' windowing/graphics stack).
include(FetchContent)
FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.91.5
)
FetchContent_MakeAvailable(imgui)

add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp
)
target_include_directories(imgui PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends
)
target_link_libraries(imgui PUBLIC glfw Vulkan::Vulkan)

# The Windows icon and version resource of an app executable. The resource
# script is generated into the build tree from one template: the shared
# Sabercat icon plus the app's file description. Defined here, before the apps,
# so each app calls it. A no-op off Windows.
set(SABERCAT_ICON_FILE "${CMAKE_CURRENT_SOURCE_DIR}/common/resources/sabercat.ico")
set(SABERCAT_RESOURCE_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/common/resources/sabercat_resource.h")
set(SABERCAT_RC_TEMPLATE "${CMAKE_CURRENT_SOURCE_DIR}/common/resources/sabercat_app.rc.in")
string(REPLACE "." "," SABERCAT_RC_VERSION "${PROJECT_VERSION},0")
function(sabercat_add_app_resources target description)
    if(NOT WIN32)
        return()
    endif()
    set(SABERCAT_APP_NAME "${target}")
    set(SABERCAT_APP_DESCRIPTION "${description}")
    set(rc_file "${CMAKE_CURRENT_BINARY_DIR}/${target}_resources.rc")
    configure_file("${SABERCAT_RC_TEMPLATE}" "${rc_file}" @ONLY)
    set_source_files_properties("${rc_file}" PROPERTIES OBJECT_DEPENDS "${SABERCAT_ICON_FILE}")
    target_sources(${target} PRIVATE "${rc_file}")
endfunction()

# Code shared by the apps (shader lookup, scene scan, orbit camera, relight
# panel, borrowed-device host, ImGui bootstrap). Must precede the apps.
add_subdirectory(common)
add_subdirectory(viewer)
add_subdirectory(studio)
