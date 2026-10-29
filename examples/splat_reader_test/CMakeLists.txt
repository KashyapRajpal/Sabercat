# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Headless test for SplatReader + the loadScene() dispatch.
# Synthesizes every .splat fixture at runtime -- no network, no assets/.

add_executable(sabercat_splat_reader_test main.cpp)
target_link_libraries(sabercat_splat_reader_test PRIVATE sabercat)
