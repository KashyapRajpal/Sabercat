# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Unit tests (CPU-only, label `cpu`). See README.md in this directory.

# Intercepts command-buffer allocation/free to check bounded ownership without
# a Vulkan device. Keep this independent of the optional application targets.
add_executable(sabercat_command_buffer_test test_command_buffer.cpp)
target_link_libraries(sabercat_command_buffer_test PRIVATE sabercat)
if(TARGET zlib)
    add_custom_command(TARGET sabercat_command_buffer_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:zlib> $<TARGET_FILE_DIR:sabercat_command_buffer_test>)
endif()
add_test(NAME command_buffer_test COMMAND $<TARGET_FILE:sabercat_command_buffer_test>)
set_tests_properties(command_buffer_test PROPERTIES LABELS cpu TIMEOUT 60)

# Golden-output test for the image comparison CLI. It runs sabercat_compare on
# fixed fixtures and compares stdout, stderr, the exit code and any PNG it writes
# byte-for-byte with files captured from a known-good build (see
# tests/compare_golden/run_golden.cmake).
if(TARGET sabercat_compare)
    include(FetchContent)
    # apps/compare fetches FLIP, but FetchContent sets <name>_SOURCE_DIR only in
    # the directory that made the call. The global property works from here.
    FetchContent_GetProperties(flip SOURCE_DIR flip_source_dir)
    add_test(NAME test_cli_golden COMMAND ${CMAKE_COMMAND}
        -DEXE=$<TARGET_FILE:sabercat_compare>
        -DFIXTURES=${CMAKE_CURRENT_SOURCE_DIR}/../compare_golden/fixtures
        -DFLIP_IMAGES=${flip_source_dir}/images
        -DEXPECTED=${CMAKE_CURRENT_SOURCE_DIR}/../compare_golden/expected
        -DWORK=${CMAKE_CURRENT_BINARY_DIR}/golden_work
        -DMODE=check
        -P ${CMAKE_CURRENT_SOURCE_DIR}/../compare_golden/run_golden.cmake)
    set_tests_properties(test_cli_golden PROPERTIES LABELS cpu TIMEOUT 60)

    # CLI checks for the HDR, report and launch-mode flags: exit codes, the output
    # lines, the files written, and the GUI message (see run_cli_checks.cmake).
    add_test(NAME test_cli_hdr COMMAND ${CMAKE_COMMAND}
        -DEXE=$<TARGET_FILE:sabercat_compare>
        -DFLIP_IMAGES=${flip_source_dir}/images
        -DWORK=${CMAKE_CURRENT_BINARY_DIR}/cli_hdr_work
        -P ${CMAKE_CURRENT_SOURCE_DIR}/../compare_golden/run_cli_checks.cmake)
    set_tests_properties(test_cli_hdr PROPERTIES LABELS cpu TIMEOUT 120)
endif()

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
# The image comparison tool is optional with the apps; its test is added only
# when that library is built.
if(TARGET sabercat_compare_core)
    target_sources(sabercat_unit_test PRIVATE test_image_compare.cpp test_compare_launch_mode.cpp test_compare_loaders.cpp
        test_compare_timeline.cpp test_compare_project.cpp test_compare_layout.cpp)
    # The loader test writes EXR fixtures with tinyexr's own save API, so it needs the
    # header (the implementation is in sabercat_compare_core). Populated by apps/compare.
    include(FetchContent)
    FetchContent_GetProperties(tinyexr SOURCE_DIR tinyexr_source_dir)
    target_include_directories(sabercat_unit_test PRIVATE ${tinyexr_source_dir})
    # The cheap FLIP HDR checks are here; the full-resolution reference checks are in
    # compare_flip_reference, below.
    target_sources(sabercat_unit_test PRIVATE test_compare_flip_hdr.cpp test_compare_report.cpp)
    # The WebP loader check reads its committed fixture (tests/compare_golden/fixtures/webp).
    target_compile_definitions(sabercat_unit_test PRIVATE
        SABERCAT_COMPARE_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../compare_golden/fixtures")
endif()
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
if(TARGET sabercat_compare_core)
    target_link_libraries(sabercat_unit_test PRIVATE sabercat_compare_core)
    target_compile_definitions(sabercat_unit_test PRIVATE SABERCAT_HAVE_COMPARE)
endif()

# The full-resolution FLIP reference checks (compare_flip_reference.cpp) read FLIP's sample
# images from the FLIP source tree (not copied into the repo) and take about two minutes in
# Debug, so they have their own executable and CTest entry. A missing folder fails them.
if(TARGET sabercat_compare_core)
    include(FetchContent)
    FetchContent_GetProperties(flip SOURCE_DIR flip_source_dir)
    add_executable(sabercat_compare_flip_reference compare_flip_reference.cpp)
    target_link_libraries(sabercat_compare_flip_reference PRIVATE sabercat_compare_core)
    target_compile_definitions(sabercat_compare_flip_reference PRIVATE
        SABERCAT_FLIP_IMAGES_DIR="${flip_source_dir}/images")
    # omp.h is used by the thread-count check in compare_flip_reference.cpp.
    if(MSVC)
        set_source_files_properties(compare_flip_reference.cpp PROPERTIES COMPILE_OPTIONS "/openmp")
    endif()
    if(TARGET zlib)
        add_custom_command(TARGET sabercat_compare_flip_reference POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    $<TARGET_FILE:zlib> $<TARGET_FILE_DIR:sabercat_compare_flip_reference>)
    endif()
    add_test(NAME compare_flip_reference COMMAND $<TARGET_FILE:sabercat_compare_flip_reference>)
    set_tests_properties(compare_flip_reference PROPERTIES LABELS cpu TIMEOUT 900)
endif()

# The renderer library is static, but its spz dependency is a shared zlib: the
# executable needs zlib.dll beside it to start (see the root CMakeLists.txt).
if(TARGET zlib)
    add_custom_command(TARGET sabercat_unit_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                $<TARGET_FILE:zlib> $<TARGET_FILE_DIR:sabercat_unit_test>)
endif()

add_test(NAME unit_test COMMAND $<TARGET_FILE:sabercat_unit_test>)
# The full-resolution FLIP checks run in compare_flip_reference, with their own timeout, so
# this suite finishes well within 120 s even in Debug.
set_tests_properties(unit_test PROPERTIES LABELS cpu TIMEOUT 120)
