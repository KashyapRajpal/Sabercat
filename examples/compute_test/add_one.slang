// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Adds one to every element the dispatch reaches. Used with direct and
// indirect dispatches: the number of elements touched is the dispatch's
// group count times 64, so the host can tell exactly what ran.

RWStructuredBuffer<int> data;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    data[id.x] += 1;
}
