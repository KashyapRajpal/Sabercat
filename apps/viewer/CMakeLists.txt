# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Sabercat Viewer -- an application that renders Gaussian clouds interactively.
# It owns the window (GLFW) and drives the renderer's forward pass only.
# The imgui target and the shared app code (apps/common) are defined one level up.

add_executable(sabercat_viewer viewer.cpp)
add_dependencies(sabercat_viewer sabercat_shaders)
sabercat_add_app_resources(sabercat_viewer "Sabercat Viewer")

# The app links the renderer library AND its own windowing library. The library
# itself does not depend on GLFW; the app injects the windowing bits.
target_link_libraries(sabercat_viewer PRIVATE sabercat sabercat_app_common glfw imgui)

# bin/<Config>/, beside sabercat_c and the zlib DLL, like the libraries -- so
# Debug and Release share a prefix and the viewer finds
# ../../share/sabercat/shaders at run time.
install(TARGETS sabercat_viewer DESTINATION ${CMAKE_INSTALL_BINDIR}/$<CONFIG>)
