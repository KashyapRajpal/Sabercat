# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Headless round-trip test for PLYWriter (Studio S1): synthesizes clouds at
# runtime, writes them, reads them back with PLYReader, and compares. A real
# scene from assets/ is used when one is present.

add_executable(sabercat_ply_writer_test main.cpp plain_ply_checks.cpp)
target_link_libraries(sabercat_ply_writer_test PRIVATE sabercat)
