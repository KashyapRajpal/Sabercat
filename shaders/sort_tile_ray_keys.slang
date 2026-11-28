// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Sorting study, TileRayDepth and Hybrid modes: key each instance by the peak
// depth along its tile's center ray.
//
// The default key is the Gaussian's center depth, which orders a pair of
// overlapping splats wrongly whenever their peaks lie in the opposite order
// along the ray. Here the key is t*, the depth at which the Gaussian is
// brightest along the ray through the tile center (rayDepthAlongRay). It is
// one value per (tile, Gaussian) instance, so the list is ordered per tile by
// the ray that tile actually samples.
//
// The key is a float >= 0 stored as its bit pattern, so the radix sort's
// unsigned compare orders it correctly. NaN and -0 map to +0 (the test
// "t > 0" is false for both).

#include "sort_common.slangh"

StructuredBuffer<InstanceDispatch> dispatch;
StructuredBuffer<ProjectedGaussian> projected;
// SABERCAT_RAY_FORM_FLOATS floats per Gaussian, from ray_depth_form.slang.
StructuredBuffer<float> rayForm;
StructuredBuffer<uint> keysTile;        // tile per instance slot
StructuredBuffer<uint> values;          // Gaussian index per instance slot
RWStructuredBuffer<uint> keysDepth;     // the radix sort's low-word key

[vk::push_constant]
ConstantBuffer<CameraUniforms> camera;

[shader("compute")]
[numthreads(SABERCAT_WORKGROUP_SIZE, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    uint i = threadId.x;
    if (i >= dispatch[0].instanceCount)
    {
        return;
    }

    uint tile = keysTile[i];
    uint tx = tile % camera.tilesX;
    uint ty = tile / camera.tilesX;

    // Center of the tile in pixel coordinates (integer pixel centers, so the
    // middle of a 16-pixel run is at 16*t + 7.5). The reference uses the same
    // point, so the sort keys and the window model sample the same ray.
    float px = float(SABERCAT_TILE_SIZE * tx) + 0.5f * float(SABERCAT_TILE_SIZE - 1);
    float py = float(SABERCAT_TILE_SIZE * ty) + 0.5f * float(SABERCAT_TILE_SIZE - 1);
    float2 dxdy = float2((px - camera.cx) / camera.fx, -(py - camera.cy) / camera.fy);

    uint g = values[i];
    float t = rayDepthAlongRay(rayForm, g * SABERCAT_RAY_FORM_FLOATS, projected[g].depth, dxdy);
    keysDepth[i] = asuint(t > 0.0f ? t : 0.0f);
}
