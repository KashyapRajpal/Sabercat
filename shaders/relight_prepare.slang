// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Resolve each Gaussian's surface data once per frame, so the
// G-buffer rasterizer only fetches a 48 B Surfel per (pixel, Gaussian).
//
// Every relightable stream is optional. When one is absent, the default is
// derived here:
//   normal     <- the ellipsoid's shortest axis (a splat's flat direction)
//   base colour<- the SH DC colour (clamp(C0*dc + 0.5)), roughness the default
//   metallic   <- 0
//   ambient    <- 1 (fully visible)
// Normals -- given or derived -- are flipped to face the camera, as R3DG does
// when rendering: a splat is two-sided and its stored sign is arbitrary.

#include "forward_common.slangh"
#include "relight_common.slangh"

StructuredBuffer<float4> means;
StructuredBuffer<float4> scales;
StructuredBuffer<float4> rotations;
StructuredBuffer<float4> shCoefficients; // 16 per Gaussian
StructuredBuffer<float4> normals;            // optional
StructuredBuffer<float4> baseColorRoughness; // optional
StructuredBuffer<float> metallic;            // optional
StructuredBuffer<float> visibility;          // optional, 16 per Gaussian
RWStructuredBuffer<Surfel> surfels;

static const uint kHasNormals = 1u;
static const uint kHasBaseColor = 2u;
static const uint kHasMetallic = 4u;
static const uint kHasVisibility = 8u;

struct PrepareParams
{
    float4 cameraPosition; // asset space
    uint count;
    uint flags;
    uint pad0;
    uint pad1;
};

[vk::push_constant]
ConstantBuffer<PrepareParams> params;

[shader("compute")]
[numthreads(SABERCAT_WORKGROUP_SIZE, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= params.count)
    {
        return;
    }
    float3 mean = means[i].xyz;

    float3 normal;
    if ((params.flags & kHasNormals) != 0)
    {
        normal = normals[i].xyz;
    }
    else
    {
        // Sigma = R diag(s^2) R^T: the axes are R's columns; take the one
        // with the smallest scale.
        float3x3 R = quaternionToMatrix(rotations[i]);
        float3 s = scales[i].xyz;
        uint axis = (s.x <= s.y && s.x <= s.z) ? 0u : ((s.y <= s.z) ? 1u : 2u);
        normal = float3(R[0][axis], R[1][axis], R[2][axis]);
    }
    float len = length(normal);
    normal = len > 1e-8f ? normal / len : float3(0.0f, 0.0f, 1.0f);
    if (dot(normal, params.cameraPosition.xyz - mean) < 0.0f)
    {
        normal = -normal;
    }

    float4 baseRough;
    if ((params.flags & kHasBaseColor) != 0)
    {
        baseRough = baseColorRoughness[i];
    }
    else
    {
        float3 dc = SABERCAT_SH_C0 * shCoefficients[i * SABERCAT_SH_COEFFS].xyz + 0.5f;
        baseRough = float4(clamp(dc, 0.0f, 1.0f), SABERCAT_DEFAULT_ROUGHNESS);
    }

    float metal = (params.flags & kHasMetallic) != 0 ? metallic[i] : 0.0f;

    // Ambient visibility: (1/pi) * integral V(w) max(0, n.w) dw from the
    // visibility SH, by band-wise cosine convolution (exact for degree <= 3).
    float ambient = 1.0f;
    if ((params.flags & kHasVisibility) != 0)
    {
        float b[16];
        shBasis16(normal, b);
        uint base = i * SABERCAT_VISIBILITY_COEFFS;
        float sum = 0.0f;
        for (uint k = 0; k < 16; ++k)
        {
            uint band = k == 0 ? 0u : (k < 4 ? 1u : (k < 9 ? 2u : 3u));
            sum += kCosineBand[band] * visibility[base + k] * b[k];
        }
        ambient = saturate(sum);
    }

    Surfel s;
    s.normalMetallic = float4(normal, metal);
    s.baseColorRoughness = baseRough;
    s.ambient = float4(ambient, 0.0f, 0.0f, 0.0f);
    surfels[i] = s;
}
