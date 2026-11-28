// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Host model of the Stable-mode sticky key (shaders/sort_stable_update.slang).
// The GPU keeps one bucket per Gaussian across frames; these functions give the
// same next key for the same inputs, so the sort study can check the GPU's
// output and the unit tests can check the rule itself.
#pragma once

#include <cstdint>

namespace sabercat::study
{

// The sticky key of a Gaussian with no bucket yet. The GPU writes it for a
// Gaussian with no tile coverage, so the next frame it takes a fresh bucket.
constexpr uint32_t kStableSentinel = 0xFFFFFFFFu;

// One step of the kernel, in float, in the kernel's operation order. Returns the
// new stored key: the stored key when it is kept, otherwise floor of the bucket
// coordinate (clamped at 0). A tileCount of 0 returns kStableSentinel.
uint32_t stableKeyStep(uint32_t stored, float depth, uint32_t tileCount, float bucketsPerOctave, float hysteresis);

// True when the float result could legitimately differ between the GPU's log2
// and the host's std::log2. Two cases count: the distance from the stored
// bucket's center is within 0.02 of the hysteresis edge (any stored key), or
// the bucket coordinate x is within 0.02 of an integer and the step writes a
// new key (the sentinel always writes; a stored key is rewritten only past the
// edge). A kept key is never flagged for the floor, since its floor is unused.
// A tileCount of 0 is never ambiguous, because the result is the sentinel.
bool stableKeyAmbiguous(uint32_t stored, float depth, uint32_t tileCount, float bucketsPerOctave, float hysteresis);

} // namespace sabercat::study
