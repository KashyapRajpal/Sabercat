# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Tests. Processed after examples/ (root CMakeLists.txt), so the test targets
# exist when tests/integration registers them with CTest.
#
# - unit/        CPU-only unit tests (see unit/README.md)
# - integration/ GPU-backed examples/*_test programs, registered with CTest
#
# Run with: ctest --test-dir build -C Release --output-on-failure

enable_testing()

add_subdirectory(unit)
add_subdirectory(integration)
