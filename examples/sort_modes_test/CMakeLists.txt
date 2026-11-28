# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Sort modes: the SortSettings surface (defaults, Differentiable forcing the
# default sort, the lighting gate, validation, stage timings) on a synthetic
# scene, plus a manual --timing benchmark that ctest does not run.
add_executable(sabercat_sort_modes_test main.cpp)
add_dependencies(sabercat_sort_modes_test sabercat_shaders)
target_link_libraries(sabercat_sort_modes_test PRIVATE sabercat sabercat_sort_study_core)

target_compile_definitions(sabercat_sort_modes_test PRIVATE
    SHADER_DIR="${SABERCAT_SHADER_DIR}"
)
