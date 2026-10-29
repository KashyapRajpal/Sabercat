# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Renderer-level contract checks: knobs, flip, debug views, scene bounds and the
# present blit, driven through the public Renderer API and checked numerically.

add_executable(sabercat_renderer_test main.cpp)
add_dependencies(sabercat_renderer_test sabercat_shaders)
target_link_libraries(sabercat_renderer_test PRIVATE sabercat)

# The present-blit check calls the internal recordPresentBlit() and the flip
# check uses the internal SH rotation helper, both from src/renderer.
target_include_directories(sabercat_renderer_test PRIVATE ${PROJECT_SOURCE_DIR}/src/renderer)

# The swapchain resize check needs a window: a hidden GLFW window when GLFW is
# available. It prints [SKIP] instead when no display can be opened.
if(SABERCAT_HAVE_GLFW)
    target_link_libraries(sabercat_renderer_test PRIVATE glfw)
    target_compile_definitions(sabercat_renderer_test PRIVATE SABERCAT_RENDERER_TEST_GLFW=1)
endif()

target_compile_definitions(sabercat_renderer_test PRIVATE
    SHADER_DIR="${SABERCAT_SHADER_DIR}"
)
