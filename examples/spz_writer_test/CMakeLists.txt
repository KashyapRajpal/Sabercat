# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Headless test for SPZWriter, the exact inverse of SpzReader. Round-trips
# through SpzReader for the quantizer tolerances, refusals and clamps; the
# byte-level check reads the file back with the reference library directly.

add_executable(sabercat_spz_writer_test main.cpp extra_checks.cpp)

# spz is linked for the byte-level check, which inspects the file with
# spz::loadSpzPacked (the same API spz_reader_test consumes). The test does not
# touch zlib itself: spz does the gzip work.
target_link_libraries(sabercat_spz_writer_test PRIVATE sabercat spz::spz)
