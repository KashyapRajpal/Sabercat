// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Sorting study: the whitened camera-space ray-depth form of every Gaussian.
//
// Per Gaussian, writes 12 floats (see sort_common.slangh for the layout): the
// three columns of P = S^-1 R^T W^T and u = P * mu_cam. P^T P is the
// camera-space inverse covariance, so the form carries the same information as
// B and a = B mu, but without the cancellation in a = B mu that cost fp32
// accuracy. Only geometry is read (no opacity, no SH), so the form is valid for
// any opacity. Degenerate Gaussians, and any Gaussian whose P or u is not
// finite, write all zeros, which consumers treat as "use the center depth".
//
// Verification and study use only: ForwardPass::computeRayDepthForm() runs
// this kernel standalone. The per-frame chain never dispatches it.

#include "sort_common.slangh"

// Same stride contract as forward_project.slang: float4 IS the stride for
// the three Gaussian streams, and the form is one scalar float per entry.
StructuredBuffer<float4> means;      // SABERCAT_STRIDE_MEANS
StructuredBuffer<float4> scales;     // SABERCAT_STRIDE_SCALES
StructuredBuffer<float4> rotations;  // SABERCAT_STRIDE_ROTATIONS

// SABERCAT_RAY_FORM_FLOATS floats per Gaussian, indexed index * 12 + k.
RWStructuredBuffer<float> rayForm;

[vk::push_constant]
ConstantBuffer<CameraUniforms> camera;

[shader("compute")]
[numthreads(SABERCAT_WORKGROUP_SIZE, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    uint index = threadId.x;
    if (index >= camera.gaussianCount)
    {
        return;
    }

    uint base = index * SABERCAT_RAY_FORM_FLOATS;

    float3 mean = means[index].xyz;
    float3 scale = scales[index].xyz;
    float4 rotation = rotations[index];

    // Same degeneracy rules as gaussian_ray.slangh, minus opacity (the form
    // depends on geometry only). Written as "not greater than" style checks
    // so NaN lands in the degenerate branch.
    float qq = dot(rotation, rotation);
    float minScale = min(scale.x, min(scale.y, scale.z));
    float maxScale = max(scale.x, max(scale.y, scale.z));
    bool ok = all(isfinite(mean)) && all(isfinite(scale)) && (minScale > 0.0f)
              && !(maxScale > 1e15f) && (qq > 0.0f) && isfinite(qq);

    float3 c0 = float3(0.0f, 0.0f, 0.0f);
    float3 c1 = float3(0.0f, 0.0f, 0.0f);
    float3 c2 = float3(0.0f, 0.0f, 0.0f);
    float3 u = float3(0.0f, 0.0f, 0.0f);
    if (ok)
    {
        // W R is the camera-space rotation of the Gaussian's principal axes,
        // with W and R both orthonormal. P = S^-1 (W R)^T, so P^T P =
        // (W R) S^-2 (W R)^T is the camera-space inverse covariance.
        float3x3 W = (float3x3)camera.worldToCamera;
        float3x3 R = quaternionToMatrix(rotation * rsqrt(qq));
        float3x3 Wr = mul(W, R);
        float3x3 Sinv = float3x3(1.0f / scale.x, 0.0f, 0.0f,
                                 0.0f, 1.0f / scale.y, 0.0f,
                                 0.0f, 0.0f, 1.0f / scale.z);
        float3x3 P = mul(Sinv, transpose(Wr));

        // Columns of P as P applied to the unit vectors, which is the matrix
        // product regardless of how Slang stores float3x3.
        c0 = mul(P, float3(1.0f, 0.0f, 0.0f));
        c1 = mul(P, float3(0.0f, 1.0f, 0.0f));
        c2 = mul(P, float3(0.0f, 0.0f, 1.0f));

        float3 muCam = mul(camera.worldToCamera, float4(mean, 1.0f)).xyz;
        u = mul(P, muCam);

        // A scale so small that 1/scale overflows gives non-finite P or u.
        ok = all(isfinite(c0)) && all(isfinite(c1)) && all(isfinite(c2)) && all(isfinite(u));
    }

    if (!ok)
    {
        for (uint k = 0; k < SABERCAT_RAY_FORM_FLOATS; ++k)
        {
            rayForm[base + k] = 0.0f;
        }
        return;
    }

    rayForm[base + 0] = c0.x;
    rayForm[base + 1] = c0.y;
    rayForm[base + 2] = c0.z;
    rayForm[base + 3] = c1.x;
    rayForm[base + 4] = c1.y;
    rayForm[base + 5] = c1.z;
    rayForm[base + 6] = c2.x;
    rayForm[base + 7] = c2.y;
    rayForm[base + 8] = c2.z;
    rayForm[base + 9] = u.x;
    rayForm[base + 10] = u.y;
    rayForm[base + 11] = u.z;
}
