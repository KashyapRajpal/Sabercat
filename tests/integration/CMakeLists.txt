# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Integration tests: the GPU-backed examples/*_test programs, registered with
# CTest. Each one exits 0 on success and prints "[validation]" lines on any
# Vulkan validation-layer error. Hosted CI runners have no GPU, so these run
# locally only (see .github/workflows/build.yml).
#
# Requires SABERCAT_BUILD_EXAMPLES. A target that is not built is skipped
# silently (guarded by if(TARGET)), so the list below can name optional
# programs. This file is processed after examples/ (see tests/CMakeLists.txt).

set(_sabercat_integration_tests
    backward_test
    borrowed_test
    c_api_test
    cloud_v1_reader_test
    compute_test
    depth_gradient_test
    depth_output_test
    device_test
    forward_regression_test
    forward_test
    gaussian_cloud_test
    gaussian_raytrace_test
    gltf_reader_test
    gltf_writer_test
    image_view_test
    ksplat_reader_test
    ply_writer_test
    project_test
    relight_test
    renderer_test
    scene_loader_test
    resize_test
    sh_lod_test
    shadow_test
    sog_reader_test
    splat_reader_test
    spz_reader_test
    visibility_test
    zip_archive_test
)

# A listed program with no target is a configure error when the examples are
# built: a misspelt or removed name must not drop a test silently. Each test is
# given a timeout, so a hang fails instead of stalling the run. The timeout is
# 900 s because the Debug builds (-O0, validation layer) are several times slower
# than Release on the large synthetic scenes.
foreach(_sabercat_t IN LISTS _sabercat_integration_tests)
    if(TARGET sabercat_${_sabercat_t})
        # Run from the source root so the relative asset paths resolve.
        add_test(NAME ${_sabercat_t}
            COMMAND $<TARGET_FILE:sabercat_${_sabercat_t}>
            WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
        set_tests_properties(${_sabercat_t} PROPERTIES LABELS gpu TIMEOUT 900
            FAIL_REGULAR_EXPRESSION "\\[validation\\]")
    elseif(SABERCAT_BUILD_EXAMPLES AND SABERCAT_HAVE_GLFW)
        message(FATAL_ERROR "tests/integration: ${_sabercat_t} is listed, but sabercat_${_sabercat_t} is not a target")
    endif()
endforeach()

# The Studio's headless self-test with no scene files: a synthetic two-layer
# project, its PLY checked against mergeLayers, and export refused with nothing
# visible. Needs the apps; the output directory is per configuration.
if(TARGET sabercat_studio)
    add_test(NAME studio_selftest
        COMMAND $<TARGET_FILE:sabercat_studio> --selftest
                ${CMAKE_BINARY_DIR}/tests/studio_selftest-$<CONFIG>)
    set_tests_properties(studio_selftest PROPERTIES LABELS gpu TIMEOUT 900
        FAIL_REGULAR_EXPRESSION "\\[validation\\]")
endif()
