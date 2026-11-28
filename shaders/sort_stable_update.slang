// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Sorting study, Stable mode: advance each Gaussian's sticky depth bucket.
//
// The default sort keys are the raw float depth bits, so a pair of Gaussians
// whose depths differ by a tiny amount can swap order from one frame to the
// next, and the swap shows as a pop. Stable mode instead keys each Gaussian by
// a logarithmic depth bucket that only changes when the depth moves well past
// the bucket's edge: the stored bucket stays put while the depth stays within
// 0.5 + hysteresis buckets of its center. A pair that sits on a bucket edge
// therefore keeps its previous order instead of flickering.
//
// One thread per Gaussian. The sticky state is one uint per Gaussian, carried
// across frames; sort_stable_keys reads it through the instance's Gaussian
// index. The sentinel 0xFFFFFFFF means "no bucket yet", and a Gaussian with no
// tile coverage this frame drops back to it so it re-enters cleanly.

#include "sort_common.slangh"

StructuredBuffer<ProjectedGaussian> projected;
RWStructuredBuffer<uint> sticky;

// Pushed per frame by ForwardPass (16 bytes). Mirrored in forward_pass.cpp.
struct StableParams
{
    uint gaussianCount;
    float bucketsPerOctave;
    float hysteresis;
    uint pad;
};

[vk::push_constant]
ConstantBuffer<StableParams> stable;

[shader("compute")]
[numthreads(SABERCAT_WORKGROUP_SIZE, 1, 1)]
void main(uint3 threadId : SV_DispatchThreadID)
{
    uint i = threadId.x;
    if (i >= stable.gaussianCount)
    {
        return;
    }

    if (projected[i].tileCount == 0)
    {
        sticky[i] = 0xFFFFFFFFu;
        return;
    }

    // Bucket coordinate in octaves of depth, offset by 64 so a depth down to
    // 2^-64 stays positive before the floor. Computed in float to match the
    // host model (stableKeyStep), which reproduces the same operation order.
    float x = (log2(projected[i].depth) + 64.0f) * stable.bucketsPerOctave;

    uint stored = sticky[i];
    // Keep the stored bucket while x is within half a bucket of its center,
    // widened by the hysteresis. The first assignment (sentinel) always takes.
    if (stored == 0xFFFFFFFFu || abs(x - ((float)stored + 0.5f)) > 0.5f + stable.hysteresis)
    {
        sticky[i] = (uint)floor(max(x, 0.0f));
    }
}
