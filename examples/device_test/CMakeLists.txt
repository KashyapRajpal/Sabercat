# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Headless smoke test for the Vulkan Device + VMA allocator.

add_executable(sabercat_device_test main.cpp)
target_link_libraries(sabercat_device_test PRIVATE sabercat)
