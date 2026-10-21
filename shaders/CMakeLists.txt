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
)

# The shared headers -- a change to any must invalidate every kernel that
# includes it (directly or via forward_common.slangh -> config.slangh).
set(SABERCAT_SHADER_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/config.slangh
    ${CMAKE_CURRENT_SOURCE_DIR}/forward_common.slangh
    ${CMAKE_CURRENT_SOURCE_DIR}/relight_common.slangh
    ${CMAKE_CURRENT_SOURCE_DIR}/gaussian_ray.slangh
    ${CMAKE_CURRENT_SOURCE_DIR}/light_visibility.slangh
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

add_custom_target(sabercat_shaders DEPENDS ${SABERCAT_SHADER_SPV_FILES})

# Consumers (examples/forward_test, apps/viewer) add_dependencies() on
# sabercat_shaders and read SABERCAT_SHADER_DIR for the runtime SPIR-V path.
set(SABERCAT_SHADER_DIR ${CMAKE_CURRENT_BINARY_DIR} PARENT_SCOPE)

# Installed for find_package(Sabercat) consumers, who get the path as
# SABERCAT_SHADER_DIR from SabercatConfig.cmake. SPIR-V is configuration-
# independent, so one copy serves Debug and Release.
install(FILES ${SABERCAT_SHADER_SPV_FILES} DESTINATION ${CMAKE_INSTALL_DATADIR}/sabercat/shaders)
