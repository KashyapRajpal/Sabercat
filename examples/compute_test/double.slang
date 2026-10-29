// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Trivial compute kernel for the compute smoke test: doubles every element of
// a storage buffer. Proves the pipeline/descriptor/dispatch path before any
// Gaussian math.

RWStructuredBuffer<int> data;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    data[id.x] *= 2;
}
