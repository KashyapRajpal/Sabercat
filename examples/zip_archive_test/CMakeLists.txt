# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Headless test for the internal ZIP module (src/renderer/zip_archive.{h,cpp}):
# the STORED writer, the reader for STORED and DEFLATE entries, CRC and
# truncation rejection. Needs no Vulkan device; links sabercat for the module
# and zlib directly to build a DEFLATE archive by hand, so the reader's DEFLATE
# path is tested against bytes this test made, not the code under test.

add_executable(sabercat_zip_archive_test main.cpp)
add_dependencies(sabercat_zip_archive_test sabercat_shaders)
target_include_directories(sabercat_zip_archive_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/renderer)
target_link_libraries(sabercat_zip_archive_test PRIVATE
    sabercat ${_sabercat_zlib_build})
