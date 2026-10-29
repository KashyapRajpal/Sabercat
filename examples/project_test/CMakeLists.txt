# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Headless test for the .sabercat project file (ProjectIO, src/renderer/project.cpp):
# bit-exact round trips, determinism, atomic save, corruption rejection, and
# plain cloud-v1 layout of each layer's folder. Needs no GPU work;
# links sabercat for the library and includes its internal ZIP header.

add_executable(sabercat_project_test main.cpp load_checks.cpp)
add_dependencies(sabercat_project_test sabercat_shaders)
target_include_directories(sabercat_project_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/renderer)
# nlohmann is PRIVATE to sabercat, so the test links it itself to compare the
# app-state JSON as values.
target_link_libraries(sabercat_project_test PRIVATE sabercat nlohmann_json::nlohmann_json)
