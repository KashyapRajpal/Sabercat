# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Sabercat Compare -- PSNR, FLIP and error-map comparison of two images. The
# PNG input and output go through the Windows Imaging Component (WIC), so the
# whole directory is Windows-only.
if(NOT WIN32)
    return()
endif()

# FLIP (NVIDIA, BSD-3-Clause; see THIRD_PARTY_NOTICES.md) is header-only. As for
# GLM and VMA in the root CMakeLists.txt, only the source is populated, so FLIP's
# own CMake project is never configured. The repo has no release tags, so the
# download is pinned to a commit.
include(FetchContent)
FetchContent_Declare(
    flip
    URL https://github.com/NVlabs/flip/archive/d3ab02a4425eb167d8a4021b382a2bafc6e6c838.tar.gz
    URL_HASH SHA256=991abec79184de1f983f303c34d97cd64ca463c24f74c42f6285d5bfc68af42e
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR _sabercat_populate_only
)
FetchContent_MakeAvailable(flip)

# tinyexr (BSD-3-Clause) reads OpenEXR. Pinned to the
# v3.4.0 release tarball by SHA256. Populate-only, like FLIP: its CMake project is
# not configured. Only its headers and deps/miniz/miniz.c (MIT, bundled in the same
# tarball) are used, compiled into sabercat_compare_core. The implementation is in
# image_io_exr.cpp.
FetchContent_Declare(
    tinyexr
    URL https://github.com/syoyo/tinyexr/archive/refs/tags/v3.4.0.tar.gz
    URL_HASH SHA256=1eed7356fcc8525d08b72deec4e8c1c822a2af888c8debc8094bc77000bd2481
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    SOURCE_SUBDIR _sabercat_populate_only
)
FetchContent_MakeAvailable(tinyexr)

# The comparison core: no window and no Vulkan, so it stays testable on its own.
add_library(sabercat_compare_core STATIC
    image_compare.cpp
    flip_metric.cpp
    image_io_wic.cpp
    image_io_webp.cpp
    image_io_exr.cpp
    image_io_raw.cpp
    image_loader.cpp
    ${tinyexr_SOURCE_DIR}/deps/miniz/miniz.c
    launch_mode.cpp
    flip_hdr.cpp
    report.cpp
    timeline.cpp
    compare_project.cpp
    compare_layout.cpp
)
# FLIP.h (flip_metric.cpp is the only file that includes it) runs its per-pixel
# loops under '#pragma omp parallel for'. MSVC needs /openmp for them; the loops
# have no reductions, so the output does not depend on the thread count.
if(MSVC)
    set_source_files_properties(flip_metric.cpp PROPERTIES COMPILE_OPTIONS "/openmp")
endif()
target_include_directories(sabercat_compare_core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${flip_SOURCE_DIR}/src/cpp
        ${tinyexr_SOURCE_DIR}
        ${tinyexr_SOURCE_DIR}/deps/miniz
        # webpdecoder's own interface include is libwebp's root; <webp/decode.h> is under src/.
        ${libwebp_SOURCE_DIR}/src
        # sabercat/version.h is configured into the build tree (root CMakeLists.txt).
        ${CMAKE_BINARY_DIR}/include
)
target_link_libraries(sabercat_compare_core PRIVATE ole32 windowscodecs)
# WebP decoding (libwebp, root CMakeLists.txt) and JSON for capture.json.
if(NOT TARGET webpdecoder)
    message(FATAL_ERROR "sabercat_compare_core needs the webpdecoder target (libwebp, root CMakeLists.txt)")
endif()
target_link_libraries(sabercat_compare_core PRIVATE webpdecoder nlohmann_json::nlohmann_json)

# The command-line and window front end. Its icon and version resource come from
# the same template as the other apps.
add_executable(sabercat_compare
    main.cpp
    cli.cpp
    app_args.cpp
    console.cpp
    gui.cpp
)
target_link_libraries(sabercat_compare PRIVATE sabercat_compare_core)
sabercat_add_app_resources(sabercat_compare "Sabercat Compare")

# bin/<Config>/, beside the other apps.
install(TARGETS sabercat_compare DESTINATION ${CMAKE_INSTALL_BINDIR}/$<CONFIG>)

# The GPU side of the window: the compositor, the metrics, the GPU context and the video
# path (Media Foundation decode, NV12 conversion). It is linked into sabercat_compare
# below. The compute kernels are compiled first (their SPIR-V goes to ${SABERCAT_SHADER_DIR},
# set by the root CMakeLists.txt through shaders/).
add_subdirectory(shaders)

add_library(sabercat_compare_gpu STATIC
    gui/gpu_context.cpp
    gui/compositor.cpp
    gui/metrics_gpu.cpp
    gui/flip_gpu.cpp
    gui/image_source.cpp
    gui/mf_decoder.cpp
    gui/mf_clip_writer.cpp
    gui/nv12_converter.cpp
    gui/video_source.cpp
    gui/scene_source.cpp
    gui/export.cpp
    gui/compare_state.cpp
    gui/compare_ui.cpp
    gui/app.cpp
    gui/selftest.cpp
)
target_include_directories(sabercat_compare_gpu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/gui
)
# sabercat_app_common brings the borrowed-device host (the self-test's device) and,
# through it, glfw and imgui.
target_link_libraries(sabercat_compare_gpu PUBLIC
    sabercat
    sabercat_app_core
    sabercat_app_common
    sabercat_compare_core
    Vulkan::Vulkan
    glfw
    imgui
)
# Media Foundation (video decode and the test clip writer) and COM. PUBLIC so the
# front end gets them too. Its delay-load flags are below.
target_link_libraries(sabercat_compare_gpu PUBLIC
    mfplat
    mfreadwrite
    mfuuid
    mf
    ole32
)
add_dependencies(sabercat_compare_gpu sabercat_compare_shaders)
# The FLIP GPU self-test loads the NVlabs reference pair from the FLIP source tree.
target_compile_definitions(sabercat_compare_gpu PRIVATE SABERCAT_FLIP_IMAGES_DIR="${flip_SOURCE_DIR}/images")

# The window front end links the GPU side and the apps' windowing stack. Its shaders
# must exist before it is built, so the dependency comes after sabercat_compare_shaders.
target_link_libraries(sabercat_compare PRIVATE
    sabercat_compare_gpu
    sabercat
    sabercat_app_common
    glfw
    imgui
)
add_dependencies(sabercat_compare sabercat_shaders sabercat_compare_shaders)
if(MSVC)
    # vulkan-1.dll is loaded on first use, so the front end starts (and reports a
    # missing driver) without it. Same as the other apps' Vulkan loading.
    target_link_options(sabercat_compare PRIVATE "/DELAYLOAD:vulkan-1.dll" "/DELAYLOAD:mfplat.dll" "/DELAYLOAD:mfreadwrite.dll" "/DELAYLOAD:mf.dll")
    target_link_libraries(sabercat_compare PRIVATE delayimp)
endif()
