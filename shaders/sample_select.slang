// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Candidate pixels for teacher samples, from the splatted
// G-buffer. Two classes:
//   smooth -- covered (coverage >= minCoverage) and not a discontinuity
//   edge   -- covered, and a 4-neighbor is uncovered (silhouette), at a
//             depth more than edgeDepthRatio * depth away, or with a normal
//             more than acos(edgeNormalCos) away
// teacher.slang then draws stratified samples from each list (75% / 25% by
// default -- the random share keeps supervision from following only shadow
// edges).
//
// Deterministic compaction in three dispatches of this one kernel (phase is
// a push constant), so the lists come out in pixel order whatever the GPU's
// scheduling -- no atomic append:
//   0  one thread per row: classify and count -> rowCounts[y] = (smooth, edge)
//   1  one thread: exclusive scan of rowCounts in place; totals -> counts.xy
//   2  one thread per row: classify again and scatter pixel indices into
//      smoothList / edgeList at the row's offsets
// Serial per row is plenty: the whole thing is a few ms at 1080p, and the
// teacher pass that follows traces 16 rays per sample.

#include "config.slangh"

struct SelectConstants
{
    uint phase;
    uint width;
    uint height;
    uint pad0;
    float minCoverage;
    float edgeDepthRatio;
    float edgeNormalCos;
    float pad1;
};

StructuredBuffer<float4> gNormalDepth;
StructuredBuffer<float4> gMetalAmbientCoverage;
RWStructuredBuffer<uint2> rowCounts; // per row: counts (phase 0), then offsets (phase 1)
RWStructuredBuffer<uint4> counts;    // [0]: x smooth total, y edge total; [1]: teacher.slang
RWStructuredBuffer<uint> smoothList; // pixel indices, pixel order
RWStructuredBuffer<uint> edgeList;

[vk::push_constant]
ConstantBuffer<SelectConstants> params;

// 0 = not a candidate, 1 = smooth, 2 = edge.
uint classify(uint x, uint y)
{
    uint p = y * params.width + x;
    float coverage = gMetalAmbientCoverage[p].z;
    if (!(coverage >= params.minCoverage))
    {
        return 0u;
    }
    float4 nd = gNormalDepth[p];
    float depth = nd.w / coverage;
    float nLen = length(nd.xyz);
    float3 n = nLen > 1e-8f ? nd.xyz / nLen : float3(0.0f, 0.0f, 0.0f);

    const int2 offsets[4] = { int2(-1, 0), int2(1, 0), int2(0, -1), int2(0, 1) };
    for (uint k = 0; k < 4; ++k)
    {
        int2 q = int2(int(x), int(y)) + offsets[k];
        if (q.x < 0 || q.y < 0 || q.x >= int(params.width) || q.y >= int(params.height))
        {
            continue; // the image border is not a discontinuity
        }
        uint qp = uint(q.y) * params.width + uint(q.x);
        float qCoverage = gMetalAmbientCoverage[qp].z;
        if (!(qCoverage >= params.minCoverage))
        {
            return 2u; // silhouette
        }
        float4 qnd = gNormalDepth[qp];
        float qDepth = qnd.w / qCoverage;
        if (abs(qDepth - depth) > params.edgeDepthRatio * depth)
        {
            return 2u;
        }
        float qLen = length(qnd.xyz);
        if (nLen > 1e-8f && qLen > 1e-8f && dot(n, qnd.xyz / qLen) < params.edgeNormalCos)
        {
            return 2u;
        }
    }
    return 1u;
}

[shader("compute")]
[numthreads(SABERCAT_WORKGROUP_SIZE, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    if (params.phase == 1u)
    {
        if (id.x != 0u)
        {
            return;
        }
        uint2 running = uint2(0u, 0u);
        for (uint y = 0; y < params.height; ++y)
        {
            uint2 rowCount = rowCounts[y];
            rowCounts[y] = running;
            running += rowCount;
        }
        counts[0] = uint4(running, 0u, 0u);
        return;
    }

    uint y = id.x;
    if (y >= params.height)
    {
        return;
    }
    if (params.phase == 0u)
    {
        uint2 rowCount = uint2(0u, 0u);
        for (uint x = 0; x < params.width; ++x)
        {
            uint c = classify(x, y);
            rowCount.x += c == 1u ? 1u : 0u;
            rowCount.y += c == 2u ? 1u : 0u;
        }
        rowCounts[y] = rowCount;
    }
    else
    {
        uint2 offset = rowCounts[y];
        for (uint x = 0; x < params.width; ++x)
        {
            uint c = classify(x, y);
            uint p = y * params.width + x;
            if (c == 1u)
            {
                smoothList[offset.x++] = p;
            }
            else if (c == 2u)
            {
                edgeList[offset.y++] = p;
            }
        }
    }
}
