// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// data[i] = data[i] * mul + add, with mul and add as push constants: the
// per-dispatch parameters travel in the command buffer, not in a buffer.

struct AffineParams
{
    int mul;
    int add;
};

RWStructuredBuffer<int> data;

[vk::push_constant]
ConstantBuffer<AffineParams> params;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID)
{
    data[id.x] = data[id.x] * params.mul + params.add;
}
