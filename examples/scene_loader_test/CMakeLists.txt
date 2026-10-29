# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Table-driven test for the scene loader's format dispatch (detectFormat and
# loadScene): every format, the magic and extension rules, and the refusals.
# Fixtures are written at run time, so the test needs no assets.

add_executable(sabercat_scene_loader_test main.cpp)
target_link_libraries(sabercat_scene_loader_test PRIVATE sabercat)
