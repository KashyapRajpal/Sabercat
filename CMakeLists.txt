# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22)
# C is enabled only for examples/c_api_test, which must compile as genuine C
# to prove the C ABI header (c_api.h) is C-clean.
project(Sabercat VERSION 1.0.0 LANGUAGES C CXX)

# Pre-release suffix for the version string (cmake -DSABERCAT_VERSION_SUFFIX=beta.1).
# Validated as semver pre-release identifiers. The CMake package version stays
# the plain numeric project version (SabercatConfigVersion.cmake).
set(SABERCAT_VERSION_SUFFIX "" CACHE STRING
    "Pre-release suffix appended to the version string, e.g. beta.1")
if(NOT SABERCAT_VERSION_SUFFIX MATCHES "^[0-9A-Za-z.-]*$")
    message(FATAL_ERROR "SABERCAT_VERSION_SUFFIX '${SABERCAT_VERSION_SUFFIX}' must match [0-9A-Za-z.-]+")
endif()
if(SABERCAT_VERSION_SUFFIX STREQUAL "")
    set(SABERCAT_VERSION_STRING "${PROJECT_VERSION}")
else()
    set(SABERCAT_VERSION_STRING "${PROJECT_VERSION}-${SABERCAT_VERSION_SUFFIX}")
endif()

# Short commit of the source tree, for Help > About > Copy system info. Never
# fails the configure: "unknown" when git is missing or this is not a checkout.
set(SABERCAT_GIT_SHA "unknown")
find_package(Git QUIET)
if(Git_FOUND)
    execute_process(COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        OUTPUT_VARIABLE SABERCAT_GIT_SHA_OUTPUT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE SABERCAT_GIT_SHA_RESULT
        ERROR_QUIET)
    if(SABERCAT_GIT_SHA_RESULT EQUAL 0 AND SABERCAT_GIT_SHA_OUTPUT MATCHES "^[0-9a-f]+$")
        set(SABERCAT_GIT_SHA "${SABERCAT_GIT_SHA_OUTPUT}")
    endif()
endif()

# C++20 for the library and its consumers.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(GNUInstallDirs)
# Consumer helpers (sabercat_copy_runtime_dlls); also installed for
# find_package(Sabercat). Functions are global, so an add_subdirectory()
# consumer can call them too.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/SabercatHelpers.cmake)

# Compiler-specific optimization/debug flags
if(NOT MSVC)
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

find_package(Vulkan REQUIRED)

# Optional dependencies
find_package(glfw3 QUIET)

# GLM and VMA are header-only, and public headers include both, so the install
# carries copies of them (include/glm/, include/vma/). Each comes from the first
# source that has it: an installed GLM package or the Vulkan SDK's optional
# GLM/VMA components, otherwise a pinned FetchContent download. Setting
# SABERCAT_FETCH_HEADER_DEPS=ON skips the first two, so the build uses exactly
# the pinned sources (reproducible builds, and testing the fetch path locally).
option(SABERCAT_FETCH_HEADER_DEPS
    "Fetch GLM and VMA even when an installed package or the Vulkan SDK provides them" OFF)

set(_sabercat_glm_dir "")
if(NOT SABERCAT_FETCH_HEADER_DEPS)
    find_package(glm CONFIG QUIET)
    if(glm_FOUND)
        get_target_property(_sabercat_glm_dir glm::glm INTERFACE_INCLUDE_DIRECTORIES)
        if(_sabercat_glm_dir)
            list(GET _sabercat_glm_dir 0 _sabercat_glm_dir)
            message(STATUS "GLM: installed package (${_sabercat_glm_dir})")
        endif()
    endif()
    if(NOT _sabercat_glm_dir)
        find_path(_sabercat_glm_sdk_dir glm/glm.hpp PATHS "${Vulkan_INCLUDE_DIR}" NO_DEFAULT_PATH)
        if(_sabercat_glm_sdk_dir)
            set(_sabercat_glm_dir "${_sabercat_glm_sdk_dir}")
            message(STATUS "GLM: Vulkan SDK (${_sabercat_glm_dir})")
        endif()
    endif()
endif()
if(NOT _sabercat_glm_dir)
    include(FetchContent)
    # Header-only: populate the source tree only, so GLM's own tests and
    # CMake project are never configured.
    FetchContent_Declare(
        glm
        URL https://github.com/g-truc/glm/archive/refs/tags/1.0.1.tar.gz
        URL_HASH SHA256=9f3174561fd26904b23f0db5e560971cbf9b3cbda0b280f04d5c379d03bf234c
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SOURCE_SUBDIR _sabercat_populate_only
    )
    FetchContent_MakeAvailable(glm)
    set(_sabercat_glm_dir "${glm_SOURCE_DIR}")
    message(STATUS "GLM: fetched 1.0.1 (${_sabercat_glm_dir})")
endif()

set(_sabercat_vma_dir "")
if(NOT SABERCAT_FETCH_HEADER_DEPS)
    find_path(_sabercat_vma_sdk_dir vma/vk_mem_alloc.h PATHS "${Vulkan_INCLUDE_DIR}" NO_DEFAULT_PATH)
    if(_sabercat_vma_sdk_dir)
        set(_sabercat_vma_dir "${_sabercat_vma_sdk_dir}")
        message(STATUS "VMA: Vulkan SDK (${_sabercat_vma_dir})")
    endif()
endif()
if(NOT _sabercat_vma_dir)
    include(FetchContent)
    # Header-only, populate-only as for GLM. Upstream ships include/vk_mem_alloc.h,
    # but the code includes <vma/vk_mem_alloc.h>, so copy it under that name.
    FetchContent_Declare(
        vma
        URL https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator/archive/refs/tags/v3.2.1.tar.gz
        URL_HASH SHA256=5e7749504cb802427ffb7bec38a0b6a15db46ae253f00560acb3e624d9fe695c
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        SOURCE_SUBDIR _sabercat_populate_only
    )
    FetchContent_MakeAvailable(vma)
    set(_sabercat_vma_dir "${CMAKE_BINARY_DIR}/_deps/vma-include")
    configure_file("${vma_SOURCE_DIR}/include/vk_mem_alloc.h"
        "${_sabercat_vma_dir}/vma/vk_mem_alloc.h" COPYONLY)
    message(STATUS "VMA: fetched 3.2.1 (${vma_SOURCE_DIR})")
endif()

# GLFW is for APPLICATIONS (apps/), not the renderer library. Fetch it if it
# isn't installed system-wide so apps can link against it.
if(glfw3_FOUND)
    set(SABERCAT_HAVE_GLFW TRUE)
else()
    message(STATUS "glfw3 not found - fetching via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG        3.4
    )
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(glfw)
    set(SABERCAT_HAVE_GLFW TRUE)
endif()

# Niantic's reference .spz implementation, vendored as a git submodule and
# pinned to v3.0.0. Fetched rather than hand-reimplemented:
# .spz is a versioned container (v1-3 gzip single stream, v4+ independently
# zstd-framed streams behind a TOC) and getting the container wrong fails only
# on REAL files -- which self-synthesized fixtures structurally cannot catch.
# Using the reference decoder means the only code Sabercat owns here is the
# adapter onto the canonical contract, which is small and genuinely testable.
#
# MIT licensed. It vendors its own zlib/zstd, so Sabercat never names those
# directly -- but note this is libsabercat's first non-Vulkan dependency, so
# the "Vulkan only" layering rule now reads "Vulkan, plus a decoder for one
# input format". Linked PRIVATE below to keep it out of the public interface.
#
# Its samples/ ships two real Niantic-authored .spz files, which is what makes
# the verification honest -- see examples/spz_reader_test.
#
# Tools stay ON: `ply_to_spz` generates assets/bonsai-7k.spz from the existing
# bonsai-7k.ply for the same-scene cross-format check.
# WORKAROUND (upstream bug, spz v3.0.0): spz fetches zlib from
# https://zlib.net/zlib-1.3.1.tar.gz, and zlib.net deletes old tarballs when it
# rotates to a new release -- that URL now returns 404, so configuring spz
# fails outright. FetchContent honors the FIRST declaration of a given name,
# so declaring zlib here (before add_subdirectory below) overrides spz's dead
# one without patching the submodule.
#
# Same version and the SAME SHA256 spz itself asserts -- the GitHub release
# asset was verified byte-identical to what zlib.net used to serve, so this
# swaps only the host, never the contents, and integrity checking stays on.
# Declare only: spz's own FetchContent_MakeAvailable(zlib) does the populate.
# Remove once spz points at a URL that stays alive.
include(FetchContent)
FetchContent_Declare(
    zlib
    URL https://github.com/madler/zlib/releases/download/v1.3.1/zlib-1.3.1.tar.gz
    URL_HASH SHA256=9a93b2b7dfdac77ceba5a558a580e74667dd6fede4585b91eefb60f03b72df23
)

set(SPZ_BUILD_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
set(SPZ_BUILD_EXTENSIONS OFF CACHE BOOL "" FORCE)
add_subdirectory(external/spz EXCLUDE_FROM_ALL)

# zlib, named directly (not only through spz) because the .sog reader inflates
# DEFLATE zip entries with it: splat-transform writes STORED entries, but the
# PlayCanvas engine also accepts deflated ones, which is what a re-zipped .sog
# from a general-purpose archiver contains. spz already made it available, so
# this adds no new dependency -- only a direct edge to it. A system zlib's
# imported target is directory-scoped to external/spz, hence the find here.
if(TARGET zlib)
    set(_sabercat_zlib_build zlib)
    set(_sabercat_zlib_install Sabercat::zlib)
else()
    find_package(ZLIB REQUIRED)
    set(_sabercat_zlib_build ZLIB::ZLIB)
    set(_sabercat_zlib_install ZLIB::ZLIB)
endif()

# libwebp, Google's reference WebP codec (BSD-3-Clause), for .sog: a SOG scene
# is a set of lossless WebP images plus meta.json, so decoding one IS decoding
# WebP. Pinned to the official v1.6.0 release tarball by SHA256 (its src/ was
# checked file-for-file identical to the v1.6.0 git tag; the tarball only adds
# autotools outputs). Only the decode-only `webpdecoder` archive is linked.
#
# Populate-only (SOURCE_SUBDIR names a directory that does not exist), then
# added EXCLUDE_FROM_ALL by hand: libwebp's install() rules are unconditional
# (headers, every library, its own WebPConfig.cmake), and an EXCLUDE_FROM_ALL
# directory's install rules never run -- the same reason spz's fetched
# zlib/zstd stay out of the install tree. The one archive Sabercat uses is
# installed and described below, like spz's. (FetchContent_Declare's own
# EXCLUDE_FROM_ALL keyword needs CMake 3.28; this project allows 3.22.)
set(WEBP_BUILD_ANIM_UTILS OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_CWEBP OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_DWEBP OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_GIF2WEBP OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_IMG2WEBP OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_VWEBP OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_WEBPINFO OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_LIBWEBPMUX OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_WEBPMUX OFF CACHE BOOL "" FORCE)
set(WEBP_BUILD_EXTRAS OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
    libwebp
    URL https://storage.googleapis.com/downloads.webmproject.org/releases/webp/libwebp-1.6.0.tar.gz
    URL_HASH SHA256=e4ab7009bf0629fd11982d4c2aa83964cf244cffba7347ecd39019a9e38c4564
    SOURCE_SUBDIR _sabercat_populate_only
)
FetchContent_MakeAvailable(libwebp)
add_subdirectory(${libwebp_SOURCE_DIR} ${libwebp_BINARY_DIR} EXCLUDE_FROM_ALL)

# nlohmann/json (MIT, header-only) parses the JSON of .sog's meta.json, glTF
# KHR_gaussian_splatting (.gltf / the .glb JSON chunk) and the
# sabercat cloud-v1 manifest.json. Header-only, so it costs no
# binary and no install: it is linked PRIVATE behind $<BUILD_INTERFACE:>, which
# keeps it out of the exported link interface entirely -- nothing in an
# installed Sabercat needs it, and no public header includes it. Pinned to the
# v3.12.0 release tarball with the SHA256 published in its release notes.
FetchContent_Declare(
    nlohmann_json
    URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
    URL_HASH SHA256=42f6e95cad6ec532fd372391373363b62a14af6d771056dbfc86160e6dfff7aa
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")
# Guarded: a parent project may already provide the target.
if(NOT TARGET nlohmann_json::nlohmann_json)
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# Main library
add_library(sabercat
    src/renderer/renderer.cpp
    src/renderer/device.cpp
    src/renderer/vma_impl.cpp
    src/renderer/pipeline.cpp
    src/renderer/buffer.cpp
    src/renderer/image.cpp
    src/renderer/command_buffer.cpp
    src/renderer/shader_compiler.cpp
    src/renderer/descriptor_set.cpp
    src/renderer/gaussian_cloud.cpp
    src/renderer/borrowed.cpp
    src/renderer/lighting.cpp
    src/renderer/relight_pass.cpp
    src/renderer/shadow_pass.cpp
    src/renderer/visibility_pass.cpp
    src/renderer/ply_reader.cpp
    src/renderer/ply_writer.cpp
    src/renderer/splat_reader.cpp
    src/renderer/spz_reader.cpp
    src/renderer/gltf_reader.cpp
    src/renderer/gltf_writer.cpp
    src/renderer/layer_transform.cpp
    src/renderer/sh_rotation.cpp
    src/renderer/cloud_v1_reader.cpp
    src/renderer/cloud_v1_writer.cpp
    src/renderer/project.cpp
    src/renderer/ksplat_reader.cpp
    src/renderer/sog_reader.cpp
    src/renderer/zip_archive.cpp
    src/renderer/scene_loader.cpp
    src/renderer/forward_pass.cpp
    src/renderer/backward_pass.cpp
    src/renderer/gaussian_acceleration_structure.cpp
    src/renderer/debug_visualizer.cpp
    src/utils/logger.cpp
)

# render_config.h includes the host/shader shared constants as
# <sabercat/shaders/config.slangh>. Mirror the file into the build tree at that
# path (configure_file re-runs CMake when the source changes, so the copy
# cannot go stale); install() puts it at the same path under include/.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/shaders/config.slangh
    ${CMAKE_CURRENT_BINARY_DIR}/include/sabercat/shaders/config.slangh COPYONLY)

# sabercat/version.h: the release version as integers and a string, for file
# provenance (glTF asset.generator, project files) and the apps' titles. Generated
# into the build tree's include dir (next to the shaders copy above) and installed
# with the other headers below.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/sabercat/version.h.in
    ${CMAKE_CURRENT_BINARY_DIR}/include/sabercat/version.h @ONLY)

# GLM and VMA dirs are listed before the Vulkan include dir (which the SDK may
# also hold copies of them): include order decides which copy is found.
target_include_directories(sabercat
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
        $<BUILD_INTERFACE:${_sabercat_glm_dir}>
        $<BUILD_INTERFACE:${_sabercat_vma_dir}>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        # webpdecoder's own interface include is libwebp's ROOT (it spells
        # its headers "src/webp/decode.h"); this gives the usual <webp/...>.
        $<BUILD_INTERFACE:${libwebp_SOURCE_DIR}/src>
)

# spz is linked PRIVATE, but a static library still carries its private
# dependencies to whoever links it: in the build tree that is the real spz
# target; in an install tree it is Sabercat::spz, an IMPORTED target that
# SabercatConfig.cmake declares over the installed vendored archives. The
# .sog reader's libwebp and zlib follow the same pattern; nlohmann/json is
# header-only, so it has no install-tree counterpart at all.
target_link_libraries(sabercat
    PUBLIC
        Vulkan::Vulkan
    PRIVATE
        $<BUILD_INTERFACE:spz::spz>
        $<INSTALL_INTERFACE:Sabercat::spz>
        $<BUILD_INTERFACE:webpdecoder>
        $<INSTALL_INTERFACE:Sabercat::webpdecoder>
        $<BUILD_INTERFACE:${_sabercat_zlib_build}>
        $<INSTALL_INTERFACE:${_sabercat_zlib_install}>
        # Header-only: build-tree only, nothing to carry into an install.
        $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>
)
target_compile_features(sabercat PUBLIC cxx_std_20)
add_library(Sabercat::sabercat ALIAS sabercat)

# The renderer library is windowing-agnostic: it depends on Vulkan ONLY, never
# on GLFW/SDL/etc. Applications own their window and link their own windowing
# library (see apps/). This is what lets the same renderer back many apps.

# The stable C ABI (include/sabercat/c_api.h), as a SHARED library.
# Deliberately a separate target rather than another source in the static
# `sabercat` library: a C ABI exists to be loaded across a binary boundary
# (ctypes/cffi, another toolchain), and only a DLL actually exercises that.
# It links the static core, so it is self-contained apart from zlib's DLL.
add_library(sabercat_c SHARED src/api/c_api.cpp)
target_link_libraries(sabercat_c PRIVATE sabercat)
target_compile_definitions(sabercat_c PRIVATE SABERCAT_C_BUILD)
target_include_directories(sabercat_c PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
add_library(Sabercat::sabercat_c ALIAS sabercat_c)

# Forward-rasterizer shaders: compiled once here, consumed by
# apps/viewer and examples/forward_test via the sabercat_shaders target.
add_subdirectory(shaders)
# Anything that links the library gets the SPIR-V it needs at runtime built too.
add_dependencies(sabercat sabercat_shaders)
# add_subdirectory() consumers: where the compiled *.comp.spv live (the
# install tree's equivalent comes from SabercatConfig.cmake).
if(NOT PROJECT_IS_TOP_LEVEL)
    set(SABERCAT_SHADER_DIR ${SABERCAT_SHADER_DIR} PARENT_SCOPE)
    # For consumers compiling their own ray-query kernels against
    # shaders/gaussian_ray.slangh (see shaders/CMakeLists.txt).
    set(SABERCAT_SHADER_INCLUDE_DIR ${SABERCAT_SHADER_INCLUDE_DIR} PARENT_SCOPE)
    set(SABERCAT_RAY_QUERY_SLANGC_FLAGS ${SABERCAT_RAY_QUERY_SLANGC_FLAGS} PARENT_SCOPE)
endif()

# Applications: windowed programs that consume the renderer library. Each owns
# its own window/input and links its own windowing library (GLFW here).
option(SABERCAT_BUILD_APPS "Build applications" ${PROJECT_IS_TOP_LEVEL})
if(SABERCAT_BUILD_APPS AND SABERCAT_HAVE_GLFW)
    add_subdirectory(apps)
endif()

# Examples: small demos and smoke tests (triangle, headless device test).
option(SABERCAT_BUILD_EXAMPLES "Build example applications" ${PROJECT_IS_TOP_LEVEL})
if(SABERCAT_BUILD_EXAMPLES AND SABERCAT_HAVE_GLFW)
    add_subdirectory(examples)
    # Make the triangle the default startup project when opening the VS solution.
    if(TARGET sabercat_triangle)
        set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT sabercat_triangle)
    endif()
endif()

# The zlib that the spz submodule fetches builds as a SHARED library -- zlib
# 1.3.1 hardcodes `add_library(zlib SHARED ...)`, so there is no static option
# to select. Every executable linking libsabercat therefore needs zlibd.dll
# beside its .exe or it dies with "zlibd.dll: cannot open shared object file"
# before main() runs.
#
# This is a custom TARGET rather than add_custom_command(TARGET ... POST_BUILD)
# because the POST_BUILD form may only name targets created in the SAME
# directory, and every executable here is created in a subdirectory. Generator
# expressions have no such restriction, so $<TARGET_FILE_DIR:...> resolves each
# executable's output directory fine from up here.
#
# Guarded on the `zlib` target existing, which is only true when spz fetched it
# -- a system zlib links statically and needs nothing copied.
if(TARGET zlib)
    set(_sabercat_dll_consumers
        sabercat_viewer
        sabercat_studio
        sabercat_triangle
        sabercat_hello_sabercat
        sabercat_device_test
        sabercat_compute_test
        sabercat_gaussian_cloud_test
        sabercat_splat_reader_test
        sabercat_ply_writer_test
        sabercat_spz_reader_test
        sabercat_gltf_reader_test
        sabercat_gltf_writer_test
        sabercat_cloud_v1_reader_test
        sabercat_ksplat_reader_test
        sabercat_sog_reader_test
        sabercat_zip_archive_test
        sabercat_forward_test
        sabercat_forward_regression_test
        sabercat_sh_lod_test
        sabercat_backward_test
        sabercat_c_api_test
        sabercat_resize_test
        sabercat_image_view_test
        sabercat_depth_gradient_test
        sabercat_depth_output_test
        sabercat_borrowed_test
        sabercat_relight_test
        sabercat_shadow_test
        sabercat_visibility_test
        sabercat_demo_reel
        sabercat_gaussian_raytrace_test
        sabercat_project_test
        sabercat_scene_diag
        sabercat_renderer_test
        sabercat_scene_loader_test)

    set(_sabercat_copy_commands)
    foreach(_sabercat_exe IN LISTS _sabercat_dll_consumers)
        if(TARGET ${_sabercat_exe})
            # make_directory first: copy_if_different treats a non-existent
            # destination as a FILE name, so copying before MSBuild has created
            # the target's output directory silently produces a file called
            # "Debug" that then blocks the real directory from being created.
            list(APPEND _sabercat_copy_commands
                 COMMAND ${CMAKE_COMMAND} -E make_directory
                         $<TARGET_FILE_DIR:${_sabercat_exe}>
                 COMMAND ${CMAKE_COMMAND} -E copy_if_different
                         $<TARGET_FILE:zlib> $<TARGET_FILE_DIR:${_sabercat_exe}>)
        endif()
    endforeach()

    if(_sabercat_copy_commands)
        add_custom_target(sabercat_copy_runtime_dlls ALL ${_sabercat_copy_commands})
        add_dependencies(sabercat_copy_runtime_dlls zlib)
        # Make each executable depend on the copy so a freshly-built exe always
        # has its DLL, regardless of target build order.
        foreach(_sabercat_exe IN LISTS _sabercat_dll_consumers)
            if(TARGET ${_sabercat_exe})
                add_dependencies(${_sabercat_exe} sabercat_copy_runtime_dlls)
            endif()
        endforeach()
    endif()
endif()

# Tests
option(SABERCAT_BUILD_TESTS "Build tests" ${PROJECT_IS_TOP_LEVEL})
if(SABERCAT_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# Installation + find_package(Sabercat).
#
# Layout: include/, lib/<Config>/ and bin/<Config>/ (so Debug and Release can
# share one prefix -- MSVC cannot mix their runtimes), share/sabercat/shaders/,
# lib/cmake/Sabercat/. Consumers that build Sabercat from source can skip all
# of this and add_subdirectory() it instead.
include(CMakePackageConfigHelpers)
set(SABERCAT_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/Sabercat)
set(SABERCAT_INSTALL_SHADER_DIR ${CMAKE_INSTALL_DATADIR}/sabercat/shaders)
set(_sabercat_libdir ${CMAKE_INSTALL_LIBDIR}/$<CONFIG>)
set(_sabercat_bindir ${CMAKE_INSTALL_BINDIR}/$<CONFIG>)

install(TARGETS sabercat sabercat_c
    EXPORT SabercatTargets
    ARCHIVE DESTINATION ${_sabercat_libdir}
    LIBRARY DESTINATION ${_sabercat_libdir}
    RUNTIME DESTINATION ${_sabercat_bindir}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)
# License, NOTICE and third-party notices sit at the root of every package.
install(FILES LICENSE NOTICE THIRD_PARTY_NOTICES.md DESTINATION .)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR} PATTERN "*.in" EXCLUDE)
# GLM and VMA headers are always shipped, whichever source they were built
# from, so the package needs no Vulkan SDK headers beyond Vulkan's own.
install(DIRECTORY ${_sabercat_glm_dir}/glm DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp" PATTERN "*.inl")
install(FILES ${_sabercat_vma_dir}/vma/vk_mem_alloc.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/vma)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/sabercat/version.h
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/sabercat)
install(FILES shaders/config.slangh
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/sabercat/shaders)
install(EXPORT SabercatTargets
    NAMESPACE Sabercat::
    DESTINATION ${SABERCAT_INSTALL_CMAKEDIR}
)

# spz and the zlib/zstd it fetched are not exportable as targets (zlib's
# carries raw build-tree include paths; spz itself declines to install when
# it fetched them). Install their binaries instead and describe them with one
# generated SabercatDeps-<Config>.cmake per configuration, which
# SabercatConfig.cmake includes to set each IMPORTED target's locations.
if(TARGET zlib)
    set(SABERCAT_VENDORED_ZLIB TRUE)
else()
    set(SABERCAT_VENDORED_ZLIB FALSE)
endif()
if(TARGET libzstd_static)
    set(SABERCAT_VENDORED_ZSTD TRUE)
else()
    set(SABERCAT_VENDORED_ZSTD FALSE)
endif()

set(_sabercat_deps_content [=[
set(_cfg $<UPPER_CASE:$<CONFIG>>)
set_property(TARGET Sabercat::spz APPEND PROPERTY IMPORTED_CONFIGURATIONS ${_cfg})
set_target_properties(Sabercat::spz PROPERTIES
    IMPORTED_LOCATION_${_cfg} "${_SABERCAT_PREFIX}/@LIBDIR@/$<CONFIG>/$<TARGET_FILE_NAME:spz>")
]=])
install(FILES $<TARGET_FILE:spz> DESTINATION ${_sabercat_libdir})
# libwebp is always fetched (there is no system-package path), so it is always
# vendored.
string(APPEND _sabercat_deps_content [=[
set_property(TARGET Sabercat::webpdecoder APPEND PROPERTY IMPORTED_CONFIGURATIONS ${_cfg})
set_target_properties(Sabercat::webpdecoder PROPERTIES
    IMPORTED_LOCATION_${_cfg} "${_SABERCAT_PREFIX}/@LIBDIR@/$<CONFIG>/$<TARGET_FILE_NAME:webpdecoder>")
]=])
install(FILES $<TARGET_FILE:webpdecoder> DESTINATION ${_sabercat_libdir})
if(SABERCAT_VENDORED_ZSTD)
    string(APPEND _sabercat_deps_content [=[
set_property(TARGET Sabercat::zstd APPEND PROPERTY IMPORTED_CONFIGURATIONS ${_cfg})
set_target_properties(Sabercat::zstd PROPERTIES
    IMPORTED_LOCATION_${_cfg} "${_SABERCAT_PREFIX}/@LIBDIR@/$<CONFIG>/$<TARGET_FILE_NAME:libzstd_static>")
]=])
    install(FILES $<TARGET_FILE:libzstd_static> DESTINATION ${_sabercat_libdir})
endif()
if(SABERCAT_VENDORED_ZLIB)
    string(APPEND _sabercat_deps_content [=[
set_property(TARGET Sabercat::zlib APPEND PROPERTY IMPORTED_CONFIGURATIONS ${_cfg})
set_target_properties(Sabercat::zlib PROPERTIES
    IMPORTED_IMPLIB_${_cfg} "${_SABERCAT_PREFIX}/@LIBDIR@/$<CONFIG>/$<TARGET_LINKER_FILE_NAME:zlib>"
    IMPORTED_LOCATION_${_cfg} "${_SABERCAT_PREFIX}/@BINDIR@/$<CONFIG>/$<TARGET_FILE_NAME:zlib>")
]=])
    install(FILES $<TARGET_LINKER_FILE:zlib> DESTINATION ${_sabercat_libdir})
    install(FILES $<TARGET_FILE:zlib> DESTINATION ${_sabercat_bindir})
endif()
string(REPLACE "@LIBDIR@" "${CMAKE_INSTALL_LIBDIR}" _sabercat_deps_content "${_sabercat_deps_content}")
string(REPLACE "@BINDIR@" "${CMAKE_INSTALL_BINDIR}" _sabercat_deps_content "${_sabercat_deps_content}")
file(GENERATE
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/sabercat-package/SabercatDeps-$<CONFIG>.cmake
    CONTENT "${_sabercat_deps_content}"
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/sabercat-package/SabercatDeps-$<CONFIG>.cmake
    DESTINATION ${SABERCAT_INSTALL_CMAKEDIR})

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/SabercatConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/sabercat-package/SabercatConfig.cmake
    INSTALL_DESTINATION ${SABERCAT_INSTALL_CMAKEDIR}
)
write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/sabercat-package/SabercatConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/sabercat-package/SabercatConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/sabercat-package/SabercatConfigVersion.cmake
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/SabercatHelpers.cmake
    DESTINATION ${SABERCAT_INSTALL_CMAKEDIR}
)
