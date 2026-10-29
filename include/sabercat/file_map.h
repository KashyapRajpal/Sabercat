// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace sabercat
{

// In-memory files keyed by a relative path with '/' separators (e.g.
// "layers/0/means.bin"). Ordered, so anything written from it is deterministic.
using FileMap = std::map<std::string, std::vector<uint8_t>>;

} // namespace sabercat
