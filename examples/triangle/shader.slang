// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Hello-triangle shader (Slang).
// Vertex positions and colors are hardcoded and selected by SV_VertexID,
// so no vertex buffer is needed - the draw call issues 3 vertices.

struct VSOutput
{
    float4 position : SV_Position;
    float3 color : COLOR0;
};

static const float2 kPositions[3] = {
    float2( 0.0, -0.5),
    float2( 0.5,  0.5),
    float2(-0.5,  0.5),
};

static const float3 kColors[3] = {
    float3(1.0, 0.0, 0.0),
    float3(0.0, 1.0, 0.0),
    float3(0.0, 0.0, 1.0),
};

[shader("vertex")]
VSOutput vertexMain(uint vertexID : SV_VertexID)
{
    VSOutput output;
    output.position = float4(kPositions[vertexID], 0.0, 1.0);
    output.color = kColors[vertexID];
    return output;
}

[shader("fragment")]
float4 fragmentMain(VSOutput input) : SV_Target
{
    return float4(input.color, 1.0);
}
