# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# GPU test for the Studio edit path: GaussianCloud's allocateGPU, setBounds and
# downloadFromGPU (with the relightable streams), the internal GpuScan checked
# against a CPU exclusive scan, EditPass::exclusiveScan, EditPass::mergeLayers
# against the host mergeLayers() (merge_section.cpp), and Renderer::mergeLayersOnGpu,
# the ray-traced visibility refit and the merge timing (renderer_section.cpp).
# The C3 selection (EditPass::select, the mask operations, SelectionMask) and the
# selection highlight (Renderer::setSelectionHighlight): selection_section.cpp.
# The C4 stream edits (EditPass transform, snapshot, delete, duplicate) on fresh layer
# copies, and the seeded random op sequence with undo and redo: ops_section.cpp.
# Section N: the mask operations and selection above the 65535-workgroup limit, on a
# 17,000,003-Gaussian layer (large_section.cpp).
# Section O: the borrowed-buffer guard (ownership model B) on the copying stream edits, and
# select and the identity transform on a borrowed layer (borrow_section.cpp).
# Includes the internal gpu_scan.h, so it also gets the renderer's source dir.

add_executable(sabercat_edit_test
    main.cpp
    merge_section.cpp
    ops_section.cpp
    large_section.cpp
    renderer_section.cpp
    selection_section.cpp
    borrow_section.cpp
)
add_dependencies(sabercat_edit_test sabercat_shaders)
target_include_directories(sabercat_edit_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../src/renderer)
target_link_libraries(sabercat_edit_test PRIVATE sabercat)

target_compile_definitions(sabercat_edit_test PRIVATE
    SHADER_DIR="${SABERCAT_SHADER_DIR}"
)
