# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Headless smoke test for GaussianCloud + PLYReader. Hand-writes a
# small synthetic .ply fixture (no network access needed) and verifies the
# parse/activation/covariance/upload path. compressed_ply.cpp does the same
# for the SuperSplat / PlayCanvas compressed .ply, encoded in-test.

add_executable(sabercat_gaussian_cloud_test main.cpp compressed_ply.cpp host_checks.cpp)
target_link_libraries(sabercat_gaussian_cloud_test PRIVATE sabercat)
