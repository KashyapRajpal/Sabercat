// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Shadow approximation: fill a full-resolution visibility layer from the
// downsampled ray-traced receivers (shadow_rt.slang, downsample s > 1, which
// traced pixel (s*lx, s*ly) of every s x s block).
//
// Each pixel blends the (up to) four traced receivers around it with
// bilinear weights times a joint-bilateral term from the G-buffer -- depth
// within ~5% and similar normals -- so a shadow edge does not bleed across a
// silhouette or crease. A traced pixel gets bilinear weight 1 on its own
// receiver and 0 elsewhere, so it keeps its traced value exactly. When every
// neighbor is rejected the most similar one is used; uncovered pixels read 1.

#include "config.slangh"

struct UpsampleParams
{
    uint width;
    uint height;
    uint layer;
    uint downsample;
};

StructuredBuffer<float4> gNormalDepth;
StructuredBuffer<float4> gMetalAmbientCoverage;
StructuredBuffer<float> lowRes;
RWStructuredBuffer<float> visibility;

[vk::push_constant]
ConstantBuffer<UpsampleParams> params;

// Unit normal and expected depth of pixel p; false where it is uncovered.
bool receiverAt(uint p, out float3 n, out float depth)
{
    float coverage = gMetalAmbientCoverage[p].z;
    float4 nd = gNormalDepth[p];
    float len = length(nd.xyz);
    n = len > 1e-8f ? nd.xyz / len : float3(0.0f, 0.0f, 0.0f);
    depth = coverage > 1e-6f ? nd.w / coverage : 0.0f;
    return coverage > 1e-6f;
}

[shader("compute")]
[numthreads(SABERCAT_TILE_SIZE, SABERCAT_TILE_SIZE, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (id.x >= params.width || id.y >= params.height)
    {
        return;
    }
    uint s = params.downsample;
    uint lw = (params.width + s - 1u) / s;
    uint lh = (params.height + s - 1u) / s;
    uint pixel = id.y * params.width + id.x;
    uint dst = params.layer * params.width * params.height + pixel;

    float3 n;
    float depth;
    if (!receiverAt(pixel, n, depth))
    {
        visibility[dst] = 1.0f;
        return;
    }

    uint lx0 = id.x / s;
    uint ly0 = id.y / s;
    if (id.x == lx0 * s && id.y == ly0 * s)
    {
        visibility[dst] = lowRes[params.layer * lw * lh + ly0 * lw + lx0]; // a traced pixel: exact
        return;
    }
    float fx = float(id.x - lx0 * s) / float(s);
    float fy = float(id.y - ly0 * s) / float(s);
    float sum = 0.0f;
    float weight = 0.0f;
    float bestSimilarity = -1.0f;
    float bestValue = 1.0f;
    for (uint k = 0; k < 4u; ++k)
    {
        uint dx = k & 1u;
        uint dy = k >> 1u;
        uint lx = min(lx0 + dx, lw - 1u);
        uint ly = min(ly0 + dy, lh - 1u);
        float bilinear = (dx == 0u ? 1.0f - fx : fx) * (dy == 0u ? 1.0f - fy : fy);
        uint q = min(ly * s, params.height - 1u) * params.width + min(lx * s, params.width - 1u);
        float3 nq;
        float dq;
        if (!receiverAt(q, nq, dq))
        {
            continue;
        }
        float depthTerm = exp(-abs(dq - depth) / max(0.05f * depth, 1e-6f));
        float normalTerm = (dot(n, n) > 0.0f && dot(nq, nq) > 0.0f) ? pow(saturate(dot(n, nq)), 4.0f) : 1.0f;
        float similarity = depthTerm * normalTerm;
        float value = lowRes[params.layer * lw * lh + ly * lw + lx];
        if (similarity > bestSimilarity)
        {
            bestSimilarity = similarity;
            bestValue = value;
        }
        float w = bilinear * similarity;
        sum += w * value;
        weight += w;
    }
    visibility[dst] = weight > 1e-4f ? sum / weight : bestValue;
}
