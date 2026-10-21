// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Backward pass, stage 2: differentiate forward_project.
//
// One thread per Gaussian. Recomputes the projection with the SAME shared
// function forward_project used (computeProjectionGeometry), then chains
// backward_rasterize's per-Gaussian dL/d(center, conic, projected opacity,
// color) to the stored parameters. One writer per Gaussian, so no atomics.
// Output buffers are OVERWRITTEN (backward is a pure function); Gaussians the
// forward culled get exact zeros.
//
// Chain, in reverse of the forward:
//   color      -> SH coefficients, and view direction -> mean
//   alpha      =  opacity * compensation(cov2d)   (compensation = 1 unless
//                 ANTIALIASED, when it differentiates through both
//                 determinants)
//   conic      =  inverse(cov2dRaw + dilation*I)
//   cov2dRaw   =  J Sigma_c J^T,  Sigma_c = W Sigma W^T,  Sigma = (R S)(R S)^T
//   J          =  J(local mean), with the frustum clamp handled exactly
//   center     =  projection of the local mean
//   local mean =  W * world mean + t
//
// Exact derivatives of the forward this renderer actually runs (that is what
// the finite-difference check holds it to). Where that differs from the INRIA
// reference -- the 0.99 alpha clamp (handled in backward_rasterize) and the
// depth term of a frustum-clamped Jacobian -- see architecture.md, backward pass.

#include "forward_common.slangh"

StructuredBuffer<float4> means;
StructuredBuffer<float4> scales;
StructuredBuffer<float4> rotations;
StructuredBuffer<float> opacities;
StructuredBuffer<float4> shCoefficients;
StructuredBuffer<ProjectedGaussian> projected;
StructuredBuffer<float> projectedGradients; // see backward_rasterize.slang

RWStructuredBuffer<float4> gradMeans;
RWStructuredBuffer<float4> gradScales;
RWStructuredBuffer<float4> gradRotations;
RWStructuredBuffer<float> gradOpacities;
RWStructuredBuffer<float4> gradShCoefficients;
RWStructuredBuffer<float4> gradScreenMeans;

[vk::push_constant]
ConstantBuffer<CameraUniforms> camera;

// Writes dL/d(coefficient) for every coefficient (0 above `degree`) and
// returns dL/d(dir). `dColor` is w.r.t. the raw SH sum (clamp already masked).
float3 backwardSH(uint degree, uint base, float3 dir, float3 dColor)
{
    float x = dir.x, y = dir.y, z = dir.z;
    float xx = x * x, yy = y * y, zz = z * z;
    float xy = x * y, yz = y * z, xz = x * z;

    // basis[k] and its gradient w.r.t. dir, band-ascending (matches
    // evaluateSH's coefficient order).
    float basis[SABERCAT_SH_COEFFS];
    float3 basisGrad[SABERCAT_SH_COEFFS];
    basis[0] = SABERCAT_SH_C0;                     basisGrad[0] = float3(0.0f, 0.0f, 0.0f);
    basis[1] = -SABERCAT_SH_C1 * y;                basisGrad[1] = float3(0.0f, -SABERCAT_SH_C1, 0.0f);
    basis[2] = SABERCAT_SH_C1 * z;                 basisGrad[2] = float3(0.0f, 0.0f, SABERCAT_SH_C1);
    basis[3] = -SABERCAT_SH_C1 * x;                basisGrad[3] = float3(-SABERCAT_SH_C1, 0.0f, 0.0f);
    basis[4] = SABERCAT_SH_C2_0 * xy;              basisGrad[4] = SABERCAT_SH_C2_0 * float3(y, x, 0.0f);
    basis[5] = SABERCAT_SH_C2_1 * yz;              basisGrad[5] = SABERCAT_SH_C2_1 * float3(0.0f, z, y);
    basis[6] = SABERCAT_SH_C2_2 * (2.0f * zz - xx - yy);
    basisGrad[6] = SABERCAT_SH_C2_2 * float3(-2.0f * x, -2.0f * y, 4.0f * z);
    basis[7] = SABERCAT_SH_C2_3 * xz;              basisGrad[7] = SABERCAT_SH_C2_3 * float3(z, 0.0f, x);
    basis[8] = SABERCAT_SH_C2_4 * (xx - yy);       basisGrad[8] = SABERCAT_SH_C2_4 * float3(2.0f * x, -2.0f * y, 0.0f);
    basis[9] = SABERCAT_SH_C3_0 * y * (3.0f * xx - yy);
    basisGrad[9] = SABERCAT_SH_C3_0 * float3(6.0f * xy, 3.0f * xx - 3.0f * yy, 0.0f);
    basis[10] = SABERCAT_SH_C3_1 * xy * z;
    basisGrad[10] = SABERCAT_SH_C3_1 * float3(yz, xz, xy);
    basis[11] = SABERCAT_SH_C3_2 * y * (4.0f * zz - xx - yy);
    basisGrad[11] = SABERCAT_SH_C3_2 * float3(-2.0f * xy, 4.0f * zz - xx - 3.0f * yy, 8.0f * yz);
    basis[12] = SABERCAT_SH_C3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy);
    basisGrad[12] = SABERCAT_SH_C3_3 * float3(-6.0f * xz, -6.0f * yz, 6.0f * zz - 3.0f * xx - 3.0f * yy);
    basis[13] = SABERCAT_SH_C3_4 * x * (4.0f * zz - xx - yy);
    basisGrad[13] = SABERCAT_SH_C3_4 * float3(4.0f * zz - 3.0f * xx - yy, -2.0f * xy, 8.0f * xz);
    basis[14] = SABERCAT_SH_C3_5 * z * (xx - yy);
    basisGrad[14] = SABERCAT_SH_C3_5 * float3(2.0f * xz, -2.0f * yz, xx - yy);
    basis[15] = SABERCAT_SH_C3_6 * x * (xx - 3.0f * yy);
    basisGrad[15] = SABERCAT_SH_C3_6 * float3(3.0f * xx - 3.0f * yy, -6.0f * xy, 0.0f);

    uint used = (degree + 1) * (degree + 1);
    float3 dDir = float3(0.0f, 0.0f, 0.0f);
    for (uint k = 0; k < SABERCAT_SH_COEFFS; ++k)
    {
        if (k < used)
        {
            gradShCoefficients[base + k] = float4(basis[k] * dColor, 0.0f);
            dDir += basisGrad[k] * dot(shCoefficients[base + k].xyz, dColor);
        }
        else
        {
            gradShCoefficients[base + k] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
    return dDir;
}

[shader("compute")]
[numthreads(SABERCAT_WORKGROUP_SIZE, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    uint index = threadId.x;
    if (index >= camera.gaussianCount)
    {
        return;
    }

    uint shBase = index * SABERCAT_SH_COEFFS;
    float4 zero4 = float4(0.0f, 0.0f, 0.0f, 0.0f);

    // Culled by the forward (tileCount 0): it touched no pixel, so its
    // gradient is exactly zero. Overwrite, never leave stale values.
    float3 worldMean = means[index].xyz;
    float3 scale = scales[index].xyz;
    float4 q = rotations[index];
    ProjectionGeometry geo = computeProjectionGeometry(camera, worldMean, scale, q);
    if (projected[index].tileCount == 0 || !geo.valid)
    {
        gradMeans[index] = zero4;
        gradScales[index] = zero4;
        gradRotations[index] = zero4;
        gradOpacities[index] = 0.0f;
        gradScreenMeans[index] = zero4;
        for (uint k = 0; k < SABERCAT_SH_COEFFS; ++k)
        {
            gradShCoefficients[shBase + k] = zero4;
        }
        return;
    }

    uint pg = index * SABERCAT_PROJECTED_GRADIENT_FLOATS;
    float2 dCenter = float2(projectedGradients[pg + 0], projectedGradients[pg + 1]);
    float3 dConic = float3(projectedGradients[pg + 2], projectedGradients[pg + 3],
                           projectedGradients[pg + 4]);
    float dAlpha = projectedGradients[pg + 5];
    float3 dColor = float3(projectedGradients[pg + 6], projectedGradients[pg + 7],
                           projectedGradients[pg + 8]);

    gradScreenMeans[index] = float4(dCenter, 0.0f, 0.0f);

    // ---- Color -> SH, + view direction -> mean ----------------------------
    // Differentiable runs bind all-zero LODParams, so the forward's degree
    // was exactly camera.shDegree (no adaptive appearance LOD).
    float3 viewVector = worldMean - camera.cameraPositionWorld.xyz;
    float viewLength = length(viewVector);
    float3 dir = viewVector / viewLength;
    float3 raw = evaluateSHRaw(camera.shDegree, shCoefficients, shBase, dir) + 0.5f;
    // clamp(raw, 0, 1) passes gradient only strictly inside.
    float3 dRaw = float3(raw.x > 0.0f && raw.x < 1.0f ? dColor.x : 0.0f,
                         raw.y > 0.0f && raw.y < 1.0f ? dColor.y : 0.0f,
                         raw.z > 0.0f && raw.z < 1.0f ? dColor.z : 0.0f);
    float3 dDir = backwardSH(camera.shDegree, shBase, dir, dRaw);
    // dir = v/|v|: dv = (dDir - dir (dir . dDir)) / |v|.
    float3 dMean = (dDir - dir * dot(dir, dDir)) / viewLength;

    // ---- Opacity / compensation --------------------------------------------
    float opacity = opacities[index];
    gradOpacities[index] = dAlpha * geo.compensation;

    // Gradient w.r.t. the raw (pre-dilation) 2D covariance, as the symmetric
    // entries r00, r01 (= r10, ONE variable), r11.
    float r00 = geo.cov2dRaw[0][0];
    float r01 = geo.cov2dRaw[0][1];
    float r11 = geo.cov2dRaw[1][1];
    float dR00 = 0.0f, dR01 = 0.0f, dR11 = 0.0f;

    bool antialiased = (camera.renderFlags & SABERCAT_RENDER_FLAG_ANTIALIASED) != 0;
    if (antialiased && geo.detBeforeDilation > 0.0f)
    {
        // comp = sqrt(det0 / det): dcomp = comp/2 (ddet0/det0 - ddet/det).
        float dComp = dAlpha * opacity;
        float a0 = 0.5f * geo.compensation / geo.detBeforeDilation;
        float a1 = -0.5f * geo.compensation / geo.determinant;
        // det0 = r00 r11 - r01^2; det = (r00+k)(r11+k) - r01^2.
        dR00 += dComp * (a0 * r11 + a1 * geo.cov2d[1][1]);
        dR11 += dComp * (a0 * r00 + a1 * geo.cov2d[0][0]);
        dR01 += dComp * (a0 + a1) * (-2.0f * r01);
    }

    // ---- Conic = inverse(cov2d) ---------------------------------------------
    {
        float A = geo.cov2d[0][0];
        float B = geo.cov2d[0][1];
        float C = geo.cov2d[1][1];
        float invDet = 1.0f / geo.determinant;
        float invDet2 = invDet * invDet;
        // conic = (C, -B, A) / det, det = AC - B^2.
        dR00 += dConic.x * (-C * C * invDet2) + dConic.y * (B * C * invDet2)
                + dConic.z * (invDet - A * C * invDet2);
        dR01 += dConic.x * (2.0f * B * C * invDet2)
                + dConic.y * (-invDet - 2.0f * B * B * invDet2)
                + dConic.z * (2.0f * A * B * invDet2);
        dR11 += dConic.x * (invDet - A * C * invDet2) + dConic.y * (A * B * invDet2)
                + dConic.z * (-A * A * invDet2);
    }

    // Symmetric matrix gradient: the shared off-diagonal splits in half.
    float2x2 G2 = float2x2(dR00, 0.5f * dR01, 0.5f * dR01, dR11);

    // ---- cov2dRaw = J Sigma_c J^T --------------------------------------------
    float3x3 dSigmaCamera = mul(mul(transpose(geo.J), G2), geo.J);
    float2x3 dJ = 2.0f * mul(mul(G2, geo.J), geo.covarianceCamera);

    // ---- Sigma_c = W Sigma W^T; Sigma = M M^T, M = R diag(scale) -------------
    float3x3 dSigma = mul(mul(transpose(geo.W), dSigmaCamera), geo.W);
    float3x3 R = quaternionToMatrix(q);
    float3x3 M = float3x3(R[0][0] * scale.x, R[0][1] * scale.y, R[0][2] * scale.z,
                          R[1][0] * scale.x, R[1][1] * scale.y, R[1][2] * scale.z,
                          R[2][0] * scale.x, R[2][1] * scale.y, R[2][2] * scale.z);
    float3x3 dM = 2.0f * mul(dSigma, M);

    float3 dScale = float3(dM[0][0] * R[0][0] + dM[1][0] * R[1][0] + dM[2][0] * R[2][0],
                           dM[0][1] * R[0][1] + dM[1][1] * R[1][1] + dM[2][1] * R[2][1],
                           dM[0][2] * R[0][2] + dM[1][2] * R[1][2] + dM[2][2] * R[2][2]);
    gradScales[index] = float4(dScale, 0.0f);

    // dL/dR = dM diag(scale).
    float3x3 D = float3x3(dM[0][0] * scale.x, dM[0][1] * scale.y, dM[0][2] * scale.z,
                          dM[1][0] * scale.x, dM[1][1] * scale.y, dM[1][2] * scale.z,
                          dM[2][0] * scale.x, dM[2][1] * scale.y, dM[2][2] * scale.z);
    float x = q.x, y = q.y, z = q.z, w = q.w;
    // Through quaternionToMatrix applied to the 4 stored values as-is.
    float4 dQ = 2.0f * float4(
        y * D[0][1] + z * D[0][2] + y * D[1][0] - 2.0f * x * D[1][1] - w * D[1][2]
            + z * D[2][0] + w * D[2][1] - 2.0f * x * D[2][2],
        -2.0f * y * D[0][0] + x * D[0][1] + w * D[0][2] + x * D[1][0] + z * D[1][2]
            - w * D[2][0] + z * D[2][1] - 2.0f * y * D[2][2],
        -2.0f * z * D[0][0] - w * D[0][1] + x * D[0][2] + w * D[1][0] - 2.0f * z * D[1][1]
            + y * D[1][2] + x * D[2][0] + y * D[2][1],
        -z * D[0][1] + y * D[0][2] + z * D[1][0] - x * D[1][2] - y * D[2][0] + x * D[2][1]);
    gradRotations[index] = dQ;

    // ---- J and center -> local mean -------------------------------------------
    float d = geo.depth;
    float invD = 1.0f / d;
    float invD2 = invD * invD;
    float invD3 = invD2 * invD;
    float lx = geo.localMean.x;
    float ly = geo.localMean.y;

    float dLx = 0.0f, dLy = 0.0f, dDepth = 0.0f;
    // J00 = fx/d, J11 = -fy/d.
    dDepth += dJ[0][0] * (-camera.fx * invD2) + dJ[1][1] * (camera.fy * invD2);
    // J02 = fx jx/d^2, J12 = -fy jy/d^2.
    float dJx = dJ[0][2] * camera.fx * invD2;
    float dJy = dJ[1][2] * (-camera.fy * invD2);
    dDepth += dJ[0][2] * (-2.0f * camera.fx * geo.jx * invD3)
              + dJ[1][2] * (2.0f * camera.fy * geo.jy * invD3);
    // jx = x while free; = sign*limX*d while clamped (no x dependence).
    if (geo.clampSignX == 0.0f)
    {
        dLx += dJx;
    }
    else
    {
        dDepth += dJx * geo.clampSignX * geo.limX;
    }
    if (geo.clampSignY == 0.0f)
    {
        dLy += dJy;
    }
    else
    {
        dDepth += dJy * geo.clampSignY * geo.limY;
    }

    // u = fx x/d + cx; v = cy - fy y/d.
    dLx += dCenter.x * camera.fx * invD;
    dDepth += dCenter.x * (-camera.fx * lx * invD2);
    dLy += dCenter.y * (-camera.fy * invD);
    dDepth += dCenter.y * (camera.fy * ly * invD2);

    // depth = -local.z; local = W world + t.
    float3 dLocal = float3(dLx, dLy, -dDepth);
    dMean += mul(transpose(geo.W), dLocal);
    gradMeans[index] = float4(dMean, 0.0f);
}
