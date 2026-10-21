// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Sizes every instance-count-dependent stage on the GPU, so the host never
// has to read the count back mid-frame.
//
// Runs as a single thread right after forward_project has atomically summed
// the tile-instance count into dispatch[0].totalInstances. Turns that count
// into the sort's element/group counts and the two indirect dispatch
// commands the host records with vkCmdDispatchIndirect.
//
// Capacity: the instance-sized buffers (sort ping-pong, radix counters) are
// sized by the host from earlier frames. If this frame needs more, nothing
// downstream may touch them -- so the count is zeroed, every indirect
// dispatch becomes a no-op, and the frame renders empty. The host compares
// totalInstances (copied out at the end of the same submission) against the
// capacity it pushed here, grows the buffers, and re-runs only the instance
// stages. Zero rather than clamping: a clamped sort would be wasted work the
// re-run throws away anyway.

#include "forward_common.slangh"

struct InstanceDispatchParams
{
    uint instanceCapacity; // elements in each sort ping-pong buffer
};

[vk::push_constant]
ConstantBuffer<InstanceDispatchParams> params;

RWStructuredBuffer<InstanceDispatch> dispatch;

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    InstanceDispatch d = dispatch[0];

    uint count = d.totalInstances <= params.instanceCapacity ? d.totalInstances : 0;
    uint sortGroups = (count + SABERCAT_WORKGROUP_SIZE - 1) / SABERCAT_WORKGROUP_SIZE;
    uint counterCount = SABERCAT_RADIX_BUCKETS * sortGroups;

    d.instanceCount = count;
    d.sortGroups = sortGroups;
    d.counterCount = counterCount;
    d.instanceGroupsX = sortGroups;
    d.instanceGroupsY = 1;
    d.instanceGroupsZ = 1;
    d.counterScanGroupsX = (counterCount + SABERCAT_SCAN_BLOCK - 1) / SABERCAT_SCAN_BLOCK;
    d.counterScanGroupsY = 1;
    d.counterScanGroupsZ = 1;
    d.pad0 = 0;
    d.pad1 = 0;
    dispatch[0] = d;
}
