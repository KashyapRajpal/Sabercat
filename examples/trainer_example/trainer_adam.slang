// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Chain rule from gradients w.r.t. the activated Gaussian values to the raw
// parameters, then one Adam step on every trained raw scalar. One thread per
// Gaussian. Layout of `params`, `adamM` and `adamV` as in trainer_activate.slang
// (32 floats per Gaussian); only the real lanes are updated.
//
// Chain rule, all from the CURRENT raw parameters:
//   mean:     dmean = gMeans.xyz
//   scale:    S = exp(logScale), dlogS = gScales.xyz * S
//   rotation: n = |qraw|, qh = qraw / n, dq = (gq - qh * dot(qh, gq)) / n
//   opacity:  a = sigmoid(logit), dlogit = gOpac * a * (1 - a)
//   SH:       gSh rgb lanes of coefficients 0..3, copied unchanged

struct AdamParams
{
    uint count;
    float bias1;
    float bias2;
    float lrMean;
    float lrScale;
    float lrQuat;
    float lrLogit;
    float lrShDc;
    float lrShRest;
};

[[vk::binding(0, 0)]] StructuredBuffer<float4> gMeans;
[[vk::binding(1, 0)]] StructuredBuffer<float4> gScales;
[[vk::binding(2, 0)]] StructuredBuffer<float4> gRots;
[[vk::binding(3, 0)]] StructuredBuffer<float> gOpac;
[[vk::binding(4, 0)]] StructuredBuffer<float4> gSh;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> params;
[[vk::binding(6, 0)]] RWStructuredBuffer<float> adamM;
[[vk::binding(7, 0)]] RWStructuredBuffer<float> adamV;

[vk::push_constant]
ConstantBuffer<AdamParams> pc;

static const float kBeta1 = 0.9f;
static const float kBeta2 = 0.999f;
static const float kEps = 1e-15f;

// One Adam update of the raw scalar at index `idx` with gradient `g`.
void adamStep(uint idx, float g, float lr)
{
    const float m = kBeta1 * adamM[idx] + (1.0f - kBeta1) * g;
    const float v = kBeta2 * adamV[idx] + (1.0f - kBeta2) * g * g;
    adamM[idx] = m;
    adamV[idx] = v;
    params[idx] = params[idx] - lr * (m * pc.bias1) / (sqrt(v * pc.bias2) + kEps);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    const uint g = id.x;
    if (g >= pc.count)
        return;

    const uint base = g * 32u;

    // Read every raw value this thread needs before any of them is updated.
    const float3 logS = float3(params[base + 4u], params[base + 5u], params[base + 6u]);
    const float4 qraw = float4(params[base + 8u], params[base + 9u],
                               params[base + 10u], params[base + 11u]);
    const float logit = params[base + 12u];

    // Mean.
    const float4 gm = gMeans[g];
    adamStep(base + 0u, gm.x, pc.lrMean);
    adamStep(base + 1u, gm.y, pc.lrMean);
    adamStep(base + 2u, gm.z, pc.lrMean);

    // Log scale.
    const float3 dlogS = gScales[g].xyz * exp(logS);
    adamStep(base + 4u, dlogS.x, pc.lrScale);
    adamStep(base + 5u, dlogS.y, pc.lrScale);
    adamStep(base + 6u, dlogS.z, pc.lrScale);

    // Quaternion. The activate kernel maps a zero quaternion to identity, so
    // its gradient is taken as zero here.
    const float n = sqrt(dot(qraw, qraw));
    float4 dq = float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (n > 0.0f)
    {
        const float4 qh = qraw / n;
        const float4 gq = gRots[g];
        dq = (gq - qh * dot(qh, gq)) / n;
    }
    adamStep(base + 8u, dq.x, pc.lrQuat);
    adamStep(base + 9u, dq.y, pc.lrQuat);
    adamStep(base + 10u, dq.z, pc.lrQuat);
    adamStep(base + 11u, dq.w, pc.lrQuat);

    // Opacity logit.
    const float a = 1.0f / (1.0f + exp(-logit));
    adamStep(base + 12u, gOpac[g] * a * (1.0f - a), pc.lrLogit);

    // SH coefficients 0..3, rgb lanes only. Coefficient 0 is DC, 1..3 are band 1.
    [unroll]
    for (uint k = 0; k < 4; ++k)
    {
        const float3 gsh = gSh[g * 16u + k].xyz;
        float lr = pc.lrShRest;
        if (k == 0u)
            lr = pc.lrShDc;
        const uint o = base + 16u + 4u * k;
        adamStep(o + 0u, gsh.x, lr);
        adamStep(o + 1u, gsh.y, lr);
        adamStep(o + 2u, gsh.z, lr);
    }
}
