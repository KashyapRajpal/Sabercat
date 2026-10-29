# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Headless test for GltfReader (glTF 2.0 + KHR_gaussian_splatting) and its
# loadScene() dispatch. Synthesizes every fixture at runtime; also renders a
# .ply and the .glb converted from it through Renderer (needs the shaders).

add_executable(sabercat_gltf_reader_test main.cpp)
add_dependencies(sabercat_gltf_reader_test sabercat_shaders)
# Links spz directly: the KHR_gaussian_splatting_compression_spz_2 fixtures
# are packed in memory with spz's own writer.
target_link_libraries(sabercat_gltf_reader_test PRIVATE sabercat spz::spz)

target_compile_definitions(sabercat_gltf_reader_test PRIVATE
    SHADER_DIR="${SABERCAT_SHADER_DIR}"
)
