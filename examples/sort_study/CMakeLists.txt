# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Sort study: a manual harness, not a CTest test. It renders the forward pass with
# the default sort and with alternative per-pixel orders on the CPU, then reports
# the error, popping and near-tie swap metrics. The CPU core (cpu_reference,
# sort_metrics) is covered by the cpu-labeled unit tests in tests/unit.
add_library(sabercat_sort_study_core STATIC
    cpu_reference.cpp
    sort_metrics.cpp
    stable_key_model.cpp
)
target_link_libraries(sabercat_sort_study_core PUBLIC sabercat)
target_include_directories(sabercat_sort_study_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
# blue_noise.h is internal to the renderer, so the stochastic model reaches it by
# this include path. The symbols it declares live in the sabercat static library,
# which this core already links, so only the header path is added here.
target_include_directories(sabercat_sort_study_core PUBLIC ${PROJECT_SOURCE_DIR}/src/renderer)

add_executable(sabercat_sort_study
    main.cpp
    study_scenes.cpp
    study_report.cpp
)
add_dependencies(sabercat_sort_study sabercat_shaders)
target_link_libraries(sabercat_sort_study PRIVATE sabercat_sort_study_core)

target_compile_definitions(sabercat_sort_study PRIVATE
    SHADER_DIR="${SABERCAT_SHADER_DIR}"
)

# The optional comparison core supplies the FLIP metric. Its include directory is
# PUBLIC, so linking it is enough for the harness to see its headers.
if(TARGET sabercat_compare_core)
    target_link_libraries(sabercat_sort_study PRIVATE sabercat_compare_core)
    target_compile_definitions(sabercat_sort_study PRIVATE SABERCAT_STUDY_HAVE_COMPARE=1)
endif()
