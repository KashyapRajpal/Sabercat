# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Shared application code (apps/common): the pieces every app needs --
# shader lookup, scene scanning, the orbit camera, the relight panel, the
# borrowed-device host and the Dear ImGui bootstrap. Links the renderer library
# and the glfw/imgui targets defined by apps/; the renderer library never links
# this.
# The GPU-free part: decisions (app_policy), scene discovery (asset_scan) and
# the SPIR-V lookup (shader_dir). It needs only the renderer library, so the CPU
# unit tests link it without a window or a Vulkan device.
add_library(sabercat_app_core STATIC
    app_policy.cpp
    asset_scan.cpp
    log_files.cpp
    shader_dir.cpp
    system_info.cpp
)
target_include_directories(sabercat_app_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(sabercat_app_core PUBLIC sabercat)

# resolveShaderDir()'s build-tree fallback: where the compiled SPIR-V lives.
target_compile_definitions(sabercat_app_core PRIVATE
    SABERCAT_SHADER_DIR="${SABERCAT_SHADER_DIR}"
)

add_library(sabercat_app_common STATIC
    about_logo.cpp
    about_panel.cpp
    app_util.cpp
    borrowed_vulkan_host.cpp
    file_association.cpp
    file_dialog.cpp
    imgui_layer.cpp
    orbit_camera.cpp
    relight_panel.cpp
    window_icon.cpp
)
target_include_directories(sabercat_app_common PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(sabercat_app_common PUBLIC sabercat_app_core glfw imgui)
if(WIN32)
    # ShellExecuteW (Help > About > Open logs folder); the native file dialogs
    # (file_dialog.cpp: CoCreateInstance, the shell's dialog CLSIDs).
    target_link_libraries(sabercat_app_common PRIVATE shell32 ole32 uuid advapi32)
endif()

# about_panel's source-tree fallback for THIRD_PARTY_NOTICES.md.
target_compile_definitions(sabercat_app_common PRIVATE
    SABERCAT_SOURCE_DIR="${PROJECT_SOURCE_DIR}"
)
