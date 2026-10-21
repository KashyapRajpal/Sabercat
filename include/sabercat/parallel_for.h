// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstddef>
#include <future>
#include <thread>
#include <vector>

namespace sabercat
{

// Splits [0, count) into contiguous chunks and runs fn(begin, end) for each
// chunk on its own thread (via std::async), then waits for all of them.
// Each chunk covers a disjoint index range, so callers can write directly
// into pre-sized shared containers from fn without locking.
//
// Falls back to running inline on the calling thread when count is small
// enough that spawning threads wouldn't pay for itself (below minChunkSize).
template <typename Fn> void parallelFor(size_t count, Fn && fn, size_t minChunkSize = 50000)
{
    if (count == 0)
    {
        return;
    }

    size_t hardwareThreads = std::max<size_t>(1, std::thread::hardware_concurrency());
    size_t threadCount = std::min(hardwareThreads, std::max<size_t>(1, count / minChunkSize));

    if (threadCount <= 1)
    {
        fn(size_t{0}, count);
        return;
    }

    size_t chunkSize = (count + threadCount - 1) / threadCount;
    std::vector<std::future<void>> futures;
    futures.reserve(threadCount);

    for (size_t t = 0; t < threadCount; ++t)
    {
        size_t begin = t * chunkSize;
        size_t end = std::min(begin + chunkSize, count);
        if (begin >= end)
        {
            break;
        }
        futures.push_back(std::async(std::launch::async, fn, begin, end));
    }

    for (auto & f : futures)
    {
        f.get(); // rethrows if fn threw inside a worker
    }
}

} // namespace sabercat
