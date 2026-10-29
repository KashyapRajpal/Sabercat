// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Writes a VkDispatchIndirectCommand {x, y, z} that covers `count` elements
// with 64-thread groups, so a later indirect dispatch reads GPU-written
// group counts. The count arrives as a push constant.

struct CountParams
{
    uint count;
};

RWStructuredBuffer<uint> args;

[vk::push_constant]
ConstantBuffer<CountParams> params;

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    args[0] = (params.count + 63u) / 64u;
    args[1] = 1u;
    args[2] = 1u;
}
