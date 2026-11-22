# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Sabercat Compare's compute kernels (.slang -> SPIR-V). They are written beside the
# renderer's kernels in ${SABERCAT_SHADER_DIR} (set by shaders/CMakeLists.txt, which
# the root CMakeLists.txt adds before apps/), so one shader lookup serves both.
# SLANGC is the cache variable found there.

set(SABERCAT_COMPARE_SHADER_SOURCES
    compare_composite.slang
    compare_nv12.slang
    compare_reduce.slang
    flip_feature_h.slang
    flip_feature_v_final.slang
    flip_hdr_accumulate.slang
    flip_prepare.slang
    flip_prepare_hdr.slang
    flip_spatial_h.slang
    flip_spatial_v_color.slang
)

# Shared by the FLIP kernels (#include), so every shader rebuilds when it changes.
set(SABERCAT_COMPARE_FLIP_COMMON ${CMAKE_CURRENT_SOURCE_DIR}/flip_common.slangh)

set(SABERCAT_COMPARE_SHADER_SPV_FILES)
foreach(shader ${SABERCAT_COMPARE_SHADER_SOURCES})
    get_filename_component(shaderName ${shader} NAME_WE)
    set(shaderSrc ${CMAKE_CURRENT_SOURCE_DIR}/${shader})
    set(shaderSpv ${SABERCAT_SHADER_DIR}/${shaderName}.comp.spv)

    add_custom_command(
        OUTPUT ${shaderSpv}
        COMMAND ${SLANGC} ${shaderSrc} -target spirv -fvk-use-entrypoint-name -stage compute -entry main -o ${shaderSpv}
        DEPENDS ${shaderSrc} ${SABERCAT_COMPARE_FLIP_COMMON}
        COMMENT "Compiling ${shader} (Slang -> SPIR-V)"
        VERBATIM
    )
    list(APPEND SABERCAT_COMPARE_SHADER_SPV_FILES ${shaderSpv})
endforeach()

add_custom_target(sabercat_compare_shaders DEPENDS ${SABERCAT_COMPARE_SHADER_SPV_FILES})

# Installed beside the renderer's kernels, as for sabercat_shaders.
install(FILES ${SABERCAT_COMPARE_SHADER_SPV_FILES} DESTINATION ${CMAKE_INSTALL_DATADIR}/sabercat/shaders)
