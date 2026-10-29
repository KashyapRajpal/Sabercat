# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Unit tests (CPU-only, label `cpu`). See README.md in this directory.
#
# One executable covers the GPU-free parts of the apps: the orbit camera, the
# Studio's document model and the app policy. It links their libraries, so it
# exists only when the apps are built.
if(NOT TARGET sabercat_studio_core)
    message(STATUS "tests/unit: the apps are not built (SABERCAT_BUILD_APPS), so no unit tests")
    return()
endif()

add_executable(sabercat_unit_test
    main.cpp
    test_app_policy.cpp
    test_file_association.cpp
    test_file_dialog.cpp
    test_orbit_camera.cpp
    test_studio_state.cpp
    test_surface_format.cpp
    test_system_info.cpp
)
# surface_format.h is internal to the renderer; it needs only the Vulkan headers.
target_include_directories(sabercat_unit_test PRIVATE ${PROJECT_SOURCE_DIR}/src/renderer)
# sabercat_app_common holds file_dialog.cpp, whose pure helpers are tested here.
# Linking it pulls in the window and ImGui libraries but creates no window.
target_link_libraries(sabercat_unit_test PRIVATE
    sabercat_studio_core
    sabercat_app_core
    sabercat_app_common
    nlohmann_json::nlohmann_json
)

# The renderer library is static, but its spz dependency is a shared zlib: the
# executable needs zlib.dll beside it to start (see the root CMakeLists.txt).
if(TARGET zlib)
    add_custom_command(TARGET sabercat_unit_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:zlib> $<TARGET_FILE_DIR:sabercat_unit_test>)
endif()

add_test(NAME unit_test COMMAND $<TARGET_FILE:sabercat_unit_test>)
set_tests_properties(unit_test PROPERTIES LABELS cpu TIMEOUT 60)
