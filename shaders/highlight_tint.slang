// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Selection highlight (Studio S2, phase C3): tints the selected Gaussians' projected
// colors. ForwardPass records this after forward_project and before the tile-count
// scan, for Viewer runs only. Thread i reads mask bit i and, when it is set, moves
// projected[i].color toward the tint by tintA. Only the color changes: depth,
// footprint, alpha and tileCount keep the projection's values, so the scan, the
// sort, the tiles and the blend are untouched.
//
// The mask holds one bit per Gaussian of the merged scene (selection_merge.slang
// builds it from the layers). Bit (i & 31) of mask[i >> 5] is Gaussian i.

#include "forward_common.slangh"

// Push constants: five 4-byte scalars, 20 bytes, no padding.
struct HighlightPush
{
    uint count;  // Gaussians to visit
    float tintR; // display-referred sRGB tint
    float tintG;
    float tintB;
    float tintA; // blend weight, 0..1
};

[vk::push_constant]
ConstantBuffer<HighlightPush> push;

// Binding 0: the projected Gaussians, tinted in place.
[[vk::binding(0, 0)]] RWStructuredBuffer<ProjectedGaussian> projected;
// Binding 1: the merged selection mask, one bit per Gaussian.
[[vk::binding(1, 0)]] StructuredBuffer<uint> mask;

[shader("compute")]
[numthreads(SABERCAT_WORKGROUP_SIZE, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    const uint i = id.x;
    if (i >= push.count)
    {
        return;
    }

    const bool selected = (mask[i >> 5] & (1u << (i & 31u))) != 0u;
    if (selected)
    {
        const float3 tint = float3(push.tintR, push.tintG, push.tintB);
        projected[i].color = lerp(projected[i].color, tint, push.tintA);
    }
}
