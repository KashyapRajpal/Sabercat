// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Forward pass, stage 1: project 3D Gaussians into screen space.
//
// Per Gaussian: world -> camera, perspective-project the mean to pixels,
// push the 3D covariance through the projection Jacobian to get a 2D
// covariance, invert it to a conic, evaluate color, and compute the screen
// footprint. Culled Gaussians are marked with tileCount = 0 and skipped by
// every later stage.

#include "forward_common.slangh"

// float4 here IS the stride assertion: it must equal SABERCAT_STRIDE_VEC4,
// and `float` must equal SABERCAT_STRIDE_SCALAR. The host binds these same
// streams using those constants (see render_config.h), so the two
// declarations are two halves of one contract.
StructuredBuffer<float4> means;      // SABERCAT_STRIDE_MEANS
StructuredBuffer<float4> scales;     // SABERCAT_STRIDE_SCALES
StructuredBuffer<float4> rotations;  // SABERCAT_STRIDE_ROTATIONS
StructuredBuffer<float> opacities;   // SABERCAT_STRIDE_OPACITY
StructuredBuffer<float4> shCoefficients; // SABERCAT_STRIDE_SH, SH_COEFFS per Gaussian

[vk::push_constant]
ConstantBuffer<CameraUniforms> camera;

RWStructuredBuffer<ProjectedGaussian> projected;
// One counter: number of tile instances all visible Gaussians will emit.
// This is element 0 of ForwardPass's InstanceDispatch buffer (see
// forward_common.slangh); instance_dispatch.slang turns it into the sizes and
// indirect dispatches of every later instance-sized stage.
RWStructuredBuffer<uint> tileInstanceCount;

// The previous frame's per-tile [first, last) instance ranges.
// ForwardPass::run() records this dispatch first and doesn't clear/overwrite
// tileRanges until just before tile_ranges, later in the same submission --
// so at this point it still holds last frame's data. Free density feedback,
// no extra buffer or host readback needed.
StructuredBuffer<uint2> previousTileRanges;
// Single-element buffer of viewer-tunable LOD knobs; see LODParams' comment
// in forward_common.slangh for why this isn't part of CameraUniforms.
StructuredBuffer<LODParams> lodParams;
// Each visible Gaussian's effective SH degree, fractional under the
// band fade (degree - 1 + topBandWeight). Read only by the SHDegree debug
// view; culled entries are left stale, as nothing reads them.
RWStructuredBuffer<float> shDegrees;

[shader("compute")]
[numthreads(SABERCAT_WORKGROUP_SIZE, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    uint index = threadId.x;
    if (index >= camera.gaussianCount)
    {
        return;
    }

    ProjectedGaussian result;
    result.center = float2(0.0f, 0.0f);
    result.depth = 0.0f;
    result.radius = 0.0f;
    result.conic = float3(0.0f, 0.0f, 0.0f);
    result.alpha = 0.0f;
    result.color = float3(0.0f, 0.0f, 0.0f);
    result.tileCount = 0;

    float3 worldMean = means[index].xyz;

    // Everything derived from the parameters lives in one shared function
    // (forward_common.slangh) so backward_project recomputes it identically.
    ProjectionGeometry geometry =
        computeProjectionGeometry(camera, worldMean, scales[index].xyz, rotations[index]);
    if (!geometry.valid)
    {
        projected[index] = result;
        return;
    }
    float depth = geometry.depth;
    float2 center = geometry.center;
    float2x2 cov2d = geometry.cov2d;
    float determinant = geometry.determinant;
    result.conic = geometry.conic;

    // Footprint from the larger eigenvalue of the symmetric 2x2.
    float midpoint = 0.5f * (cov2d[0][0] + cov2d[1][1]);
    float offset = sqrt(max(midpoint * midpoint - determinant, 0.0f));
    float maxEigenvalue = midpoint + offset;
    float minEigenvalue = midpoint - offset;
    if (minEigenvalue <= 0.0f)
    {
        projected[index] = result;
        return;
    }
    float radius = SABERCAT_RADIUS_SIGMAS * sqrt(maxEigenvalue);

    // Read the LOD knobs once, up here, because the size cull below needs
    // them as well as the distance/density policy further down.
    LODParams lod = lodParams[0];

    // Screen-space size cull -- OFF unless minScreenRadius is set. Sub-pixel
    // Gaussians are not negligible in aggregate: they carry the fine detail,
    // and culling them made real scenes render as featureless blobs (see
    // config.slangh). The `> 0.0f` guard is what makes 0 mean "disabled"
    // rather than "cull everything at or below zero radius".
    if (lod.minScreenRadius > 0.0f && radius < lod.minScreenRadius)
    {
        projected[index] = result;
        return;
    }

    // Frustum cull: reject Gaussians whose footprint misses the image.
    if (center.x + radius < 0.0f || center.x - radius > (float)camera.imageWidth ||
        center.y + radius < 0.0f || center.y - radius > (float)camera.imageHeight)
    {
        projected[index] = result;
        return;
    }

    // Distance + tile-density hybrid dynamic LOD, along two
    // axes -- whether a Gaussian survives (geometry, `enabled`) and how many
    // SH bands it evaluates (appearance, `appearanceEnabled`). Both are off
    // by default: a future differentiable backward pass must never have
    // either silently altering what it renders, so they stay explicit
    // opt-ins the viewer alone turns on.
    uint effectiveDegree = camera.shDegree;
    float topBandWeight = 1.0f; // Band fade; 1 = the hard degree
    if (lod.enabled != 0 || lod.appearanceEnabled != 0)
    {
        // Both axes read the same pressure signals, so this tile lookup and
        // the two fractions are computed once and shared.
        int centerTileX =
            clamp((int)floor(center.x / SABERCAT_TILE_SIZE), 0, (int)camera.tilesX - 1);
        int centerTileY =
            clamp((int)floor(center.y / SABERCAT_TILE_SIZE), 0, (int)camera.tilesY - 1);
        uint centerTileId = (uint)(centerTileY * (int)camera.tilesX + centerTileX);
        uint2 previousRange = previousTileRanges[centerTileId];
        float previousInstances = (float)(previousRange.y - previousRange.x);

        float distanceFraction = clamp((depth - lod.nearDistance)
                                           / max(lod.farDistance - lod.nearDistance, 1e-4f),
                                       0.0f, 1.0f);
        float densityFraction =
            clamp(previousInstances / max(lod.densityReference, 1.0f), 0.0f, 1.0f);

        if (lod.enabled != 0)
        {
            float dropFraction = max(distanceFraction * lod.maxDropDistance,
                                     densityFraction * lod.maxDropDensity);

            if (hashToUnitFloat(index) < dropFraction)
            {
                projected[index] = result;
                return;
            }
        }

        if (lod.appearanceEnabled != 0)
        {
            // Unlike the geometry drop, this needs no stable-hash trick:
            // degree is already a deterministic, monotonic function of
            // pressure, so it steps down 3 -> 2 -> 1 -> 0 smoothly rather
            // than reshuffling which Gaussians are affected. Scaled by
            // (SH_DEGREE + 1) so a full-pressure drop fraction of 1.0
            // reaches degree 0.
            float degradeFraction = max(distanceFraction * lod.appearanceMaxDropDistance,
                                        densityFraction * lod.appearanceMaxDropDensity);
            float steps = degradeFraction * (float)(SABERCAT_SH_DEGREE + 1);
            int stepsLost = (int)floor(steps);
            int adaptiveDegree = clamp(SABERCAT_SH_DEGREE - stepsLost, 0, SABERCAT_SH_DEGREE);
            // Band fade: over the last fadeWidth of each step the outgoing
            // band ramps 1 -> 0, reaching 0 exactly where the hard switch
            // drops it, so degree is continuous in pressure. Only when the
            // adaptive degree is the binding one (a lower global cap is a
            // fixed choice, not a transition) and never for the DC term.
            // Width capped at 1: wider would start the ramp below 1.
            float fadeWidth = min(lod.appearanceFadeWidth, 1.0f);
            if (fadeWidth > 0.0f && adaptiveDegree >= 1 && (uint)adaptiveDegree <= camera.shDegree)
            {
                float stepRemaining = 1.0f - (steps - (float)stepsLost);
                topBandWeight = saturate(stepRemaining / fadeWidth);
            }
            // min(), never max(): the global cap is a ceiling the adaptive
            // policy may lower but must never raise.
            effectiveDegree = min(effectiveDegree, (uint)adaptiveDegree);
        }
    }

    // Tile range this footprint covers, clamped to the tile grid.
    int minTileX = max((int)floor((center.x - radius) / SABERCAT_TILE_SIZE), 0);
    int minTileY = max((int)floor((center.y - radius) / SABERCAT_TILE_SIZE), 0);
    int maxTileX = min((int)floor((center.x + radius) / SABERCAT_TILE_SIZE), (int)camera.tilesX - 1);
    int maxTileY = min((int)floor((center.y + radius) / SABERCAT_TILE_SIZE), (int)camera.tilesY - 1);
    if (minTileX > maxTileX || minTileY > maxTileY)
    {
        projected[index] = result;
        return;
    }

    uint tileCount = (uint)((maxTileX - minTileX + 1) * (maxTileY - minTileY + 1));

    result.center = center;
    result.depth = depth;
    result.radius = radius;
    // Opacity compensation for the dilation (the "antialiased" /
    // mip-splatting formulation) -- applied only when the scene was trained
    // with it (SABERCAT_RENDER_FLAG_ANTIALIASED; computed in
    // computeProjectionGeometry, 1.0 otherwise).
    //
    // Dilating the covariance spreads a Gaussian over a LARGER area. For a
    // scene trained with compensation, leaving it out deposits each
    // sub-pixel Gaussian's full alpha over an area far bigger than it covers
    // and the image saturates into opaque mush. Measured on racoonfamily.spz
    // (antialiased=1) framed at 9.4 units: the median Gaussian's true
    // projected sigma is 0.145px but the dilation floors its radius at
    // 1.56px -- a 167x area inflation at unchanged alpha, over ~890k
    // Gaussians at median alpha 0.80.
    //
    // Scaling by sqrt(det_before / det_after) preserves the Gaussian's total
    // integrated contribution rather than its peak; it self-cancels for
    // large Gaussians (a 2px one is scaled by ~0.94).
    //
    // It is NOT applied to vanilla 3DGS scenes: they were trained without
    // it, so their opacities already encode the dilated footprint and
    // compensating would dim every small Gaussian the trainer tuned.
    result.alpha = opacities[index] * geometry.compensation;
    // View direction is world-space, camera -> Gaussian (the reference 3DGS
    // convention). worldMean is already loaded above.
    float3 viewDir = normalize(worldMean - camera.cameraPositionWorld.xyz);
    // The unfaded path stays the plain evaluateSH call, so fade-off output is
    // the pre-fade output bit for bit.
    result.color = topBandWeight < 1.0f
                       ? evaluateSHFaded(effectiveDegree, topBandWeight, shCoefficients,
                                         index * SABERCAT_SH_COEFFS, viewDir)
                       : evaluateSH(effectiveDegree, shCoefficients, index * SABERCAT_SH_COEFFS,
                                    viewDir);
    shDegrees[index] = (float)effectiveDegree - (1.0f - topBandWeight);
    result.tileCount = tileCount;
    projected[index] = result;

    // Reserve this Gaussian's slots in the tile-instance list.
    uint unused;
    InterlockedAdd(tileInstanceCount[0], tileCount, unused);
}
