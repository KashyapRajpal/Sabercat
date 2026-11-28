# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Compiles the forward-rasterizer kernels (.slang -> SPIR-V) once,
# shared by every consumer (examples/forward_test, apps/viewer) instead of
# each target repeating the slangc invocation. First multi-file shader
# target in the repo -- existing examples (compute_test, triangle) each
# only ever compiled one .slang file.

find_program(SLANGC
    NAMES slangc
    HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin"
    REQUIRED
)

set(SABERCAT_SHADER_SOURCES
    forward_project.slang
    prefix_sum.slang
    tile_binning.slang
    instance_dispatch.slang
    radix_sort_histogram.slang
    radix_sort_scatter.slang
    tile_ranges.slang
    tile_rasterize.slang
    debug_visualize.slang
    # backward pass (gradients)
    backward_rasterize.slang
    backward_project.slang
    relight_prepare.slang
    gbuffer_rasterize.slang
    deferred_lighting.slang
    shadow_map.slang
    shadow_pcss.slang
    # Gaussian ray-tracing core (plain compute: no ray query inside)
    gaussian_aabbs.slang
    # teacher-sample selection + baked-visibility splat + debug views
    sample_select.slang
    visibility_splat.slang
    visibility_debug.slang
    shadow_upsample.slang
    # sorting study
    ray_depth_form.slang
    sort_stable_update.slang
    sort_stable_keys.slang
    sort_tile_ray_keys.slang
    stochastic_resolve.slang
    # Studio edit kernels
    layer_merge.slang
    edit_select.slang
    edit_mask_ops.slang
    edit_compact.slang
    edit_transform.slang
    selection_merge.slang
    highlight_tint.slang
)

# The shared headers -- a change to any must invalidate every kernel that
# includes it (directly or via forward_common.slangh -> config.slangh).
set(SABERCAT_SHADER_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/config.slangh
    ${CMAKE_CURRENT_SOURCE_DIR}/forward_common.slangh
    ${CMAKE_CURRENT_SOURCE_DIR}/relight_common.slangh
    ${CMAKE_CURRENT_SOURCE_DIR}/color_space.slangh
    ${CMAKE_CURRENT_SOURCE_DIR}/gaussian_ray.slangh
    ${CMAKE_CURRENT_SOURCE_DIR}/light_visibility.slangh
    ${CMAKE_CURRENT_SOURCE_DIR}/sort_common.slangh
    ${CMAKE_CURRENT_SOURCE_DIR}/edit_common.slangh
    ${CMAKE_CURRENT_SOURCE_DIR}/edit_select_common.slangh
)

# Extra slangc flags for any kernel that TRACES rays (calls
# gaussian_ray.slangh's gaussianTransmittance / uses RayQuery). Slang's direct
# SPIR-V emitter (2025.6) declares OpCapability RayTracingKHR +
# SPV_KHR_ray_tracing for every acceleration-structure type, and that
# capability requires the rayTracingPipeline feature -- which Sabercat does
# not enable (ray query only), so the pipeline would fail validation. The
# GLSL route (Slang -> GLSL -> glslang) declares exactly RayQueryKHR +
# SPV_KHR_ray_query (SPIR-V 1.4). Usage:
#   slangc k.slang -I ${SABERCAT_SHADER_INCLUDE_DIR} -target spirv
#          ${SABERCAT_RAY_QUERY_SLANGC_FLAGS} -fvk-use-entrypoint-name ...
set(SABERCAT_RAY_QUERY_SLANGC_FLAGS -emit-spirv-via-glsl)
set(SABERCAT_RAY_QUERY_SLANGC_FLAGS ${SABERCAT_RAY_QUERY_SLANGC_FLAGS} PARENT_SCOPE)
set(SABERCAT_SHADER_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
set(SABERCAT_SHADER_HEADER_FILES ${SABERCAT_SHADER_HEADERS} PARENT_SCOPE)

set(SABERCAT_SHADER_SPV_FILES)
foreach(shader ${SABERCAT_SHADER_SOURCES})
    get_filename_component(shaderName ${shader} NAME_WE)
    set(shaderSrc ${CMAKE_CURRENT_SOURCE_DIR}/${shader})
    set(shaderSpv ${CMAKE_CURRENT_BINARY_DIR}/${shaderName}.comp.spv)

    add_custom_command(
        OUTPUT ${shaderSpv}
        COMMAND ${SLANGC} ${shaderSrc} -target spirv -fvk-use-entrypoint-name -stage compute -entry main -o ${shaderSpv}
        DEPENDS ${shaderSrc} ${SABERCAT_SHADER_HEADERS}
        COMMENT "Compiling ${shader} (Slang -> SPIR-V)"
        VERBATIM
    )
    list(APPEND SABERCAT_SHADER_SPV_FILES ${shaderSpv})
endforeach()

# Kernels that TRACE rays: same, with the ray-query flags above.
set(SABERCAT_RAY_QUERY_SHADER_SOURCES
    shadow_rt.slang
    # baked visibility and teacher samples
    bake_visibility.slang
    teacher.slang
)
foreach(shader ${SABERCAT_RAY_QUERY_SHADER_SOURCES})
    get_filename_component(shaderName ${shader} NAME_WE)
    set(shaderSrc ${CMAKE_CURRENT_SOURCE_DIR}/${shader})
    set(shaderSpv ${CMAKE_CURRENT_BINARY_DIR}/${shaderName}.comp.spv)

    add_custom_command(
        OUTPUT ${shaderSpv}
        COMMAND ${SLANGC} ${shaderSrc} -I ${CMAKE_CURRENT_SOURCE_DIR} -target spirv
                ${SABERCAT_RAY_QUERY_SLANGC_FLAGS} -fvk-use-entrypoint-name -stage compute
                -entry main -o ${shaderSpv}
        DEPENDS ${shaderSrc} ${SABERCAT_SHADER_HEADERS}
        COMMENT "Compiling ${shader} (Slang -> SPIR-V, ray query)"
        VERBATIM
    )
    list(APPEND SABERCAT_SHADER_SPV_FILES ${shaderSpv})
endforeach()

# Sort-study variants: one .slang compiled again with preprocessor defines, so a
# variant needs no copy of the source. Usage:
#   sabercat_add_shader_variant(<name> <source.slang> <DEFINE=VALUE> ...)
# Writes ${CMAKE_CURRENT_BINARY_DIR}/<name>.comp.spv, the same file name shape
# as the kernels above, and adds it to SABERCAT_SHADER_SPV_FILES. This is a
# macro rather than a function because that list append must reach the caller's
# scope. The window rasterizer and the stochastic rasterizer variants are both declared below.
macro(sabercat_add_shader_variant variantName variantSource)
    set(variantDefines)
    foreach(variantDefine ${ARGN})
        list(APPEND variantDefines -D${variantDefine})
    endforeach()
    set(variantSrc ${CMAKE_CURRENT_SOURCE_DIR}/${variantSource})
    set(variantSpv ${CMAKE_CURRENT_BINARY_DIR}/${variantName}.comp.spv)

    add_custom_command(
        OUTPUT ${variantSpv}
        COMMAND ${SLANGC} ${variantSrc} -target spirv -fvk-use-entrypoint-name -stage compute
                -entry main -I ${CMAKE_CURRENT_SOURCE_DIR} ${variantDefines} -o ${variantSpv}
        DEPENDS ${variantSrc} ${SABERCAT_SHADER_HEADERS}
        COMMENT "Compiling ${variantSource} as ${variantName} (Slang -> SPIR-V, variant)"
        VERBATIM
    )
    list(APPEND SABERCAT_SHADER_SPV_FILES ${variantSpv})
endmacro()

# Window-resort rasterizer, one variant per window size K (the per-pixel window of
# the WindowRayDepth model; K = 8 is Hybrid, 16 and 32 are PixelWindow).
sabercat_add_shader_variant(tile_rasterize_window_k8 tile_rasterize_window.slang SABERCAT_SORT_WINDOW=8)
sabercat_add_shader_variant(tile_rasterize_window_k16 tile_rasterize_window.slang SABERCAT_SORT_WINDOW=16)
sabercat_add_shader_variant(tile_rasterize_window_k32 tile_rasterize_window.slang SABERCAT_SORT_WINDOW=32)

# Stochastic-transparency rasterizer, one variant per samples-per-pixel count S
# (the Stochastic mode's ST4 and ST8).
sabercat_add_shader_variant(tile_rasterize_stochastic_s4 tile_rasterize_stochastic.slang SABERCAT_ST_SAMPLES=4)
sabercat_add_shader_variant(tile_rasterize_stochastic_s8 tile_rasterize_stochastic.slang SABERCAT_ST_SAMPLES=8)

add_custom_target(sabercat_shaders DEPENDS ${SABERCAT_SHADER_SPV_FILES})

# Consumers (examples/forward_test, apps/viewer) add_dependencies() on
# sabercat_shaders and read SABERCAT_SHADER_DIR for the runtime SPIR-V path.
set(SABERCAT_SHADER_DIR ${CMAKE_CURRENT_BINARY_DIR} PARENT_SCOPE)

# Installed for find_package(Sabercat) consumers, who get the path as
# SABERCAT_SHADER_DIR from SabercatConfig.cmake. SPIR-V is configuration-
# independent, so one copy serves Debug and Release.
install(FILES ${SABERCAT_SHADER_SPV_FILES} DESTINATION ${CMAKE_INSTALL_DATADIR}/sabercat/shaders)
