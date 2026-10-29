# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Headless test for GLTFWriter (KHR_gaussian_splatting .glb export) and the
# host-side LayerTransform operations (transformedCopy, mergeLayers). Round-trips
# through GltfReader; needs no shaders.

add_executable(sabercat_gltf_writer_test main.cpp extra_checks.cpp)
# nlohmann is PRIVATE to sabercat; the structural check parses the .glb JSON chunk itself.
target_link_libraries(sabercat_gltf_writer_test PRIVATE sabercat nlohmann_json::nlohmann_json)
