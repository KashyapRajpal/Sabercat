// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Playback timing for the video sources of Sabercat Compare: maps one global clock
// to the frame each source shows. Pure CPU math; nothing is decoded and no file is
// read. Frame lookup assumes a source that passed validateSource(); the caller
// validates once when the source is built.
//
// Time model:
//   sourceTime = globalTime + timeOffset        (the per-source offset is added last;
//                                                the global speed is already in the clock)
//   without loop: sourceTime is clamped to [trimIn, trimOut]
//   with loop:    sourceTime wraps into [trimIn, trimOut)
//   frame index:  the last frame whose timestamp is at or before the source time
//                 (floor), clamped to [0, frameCount - 1]
#pragma once

#include <cstdint>
#include <vector>

namespace sabercat::compare
{

enum class TimeSourceKind : uint8_t
{
    Video,
    StaticImage // always frame 0; the trim, offset and fps fields are not used
};

// Timing of one source. Frame i is at i / fps when frameTimes is empty, else at
// frameTimes[i] (variable frame rate; the list must be strictly increasing and
// have frameCount entries).
struct TimeSource
{
    std::vector<double> frameTimes; // seconds; empty = constant fps
    double fps = 0.0;               // frames per second when frameTimes is empty
    double duration = 0.0;          // clip length in seconds; trimOut must not pass it
    double trimIn = 0.0;            // first source second shown, >= 0
    double trimOut = 0.0;           // end of the trim range; must be > trimIn
    double timeOffset = 0.0;        // added to the global clock; may be negative
    int64_t frameCount = 1;         // >= 1
    TimeSourceKind kind = TimeSourceKind::Video;
    bool loop = false;
};

// The global clock. time is in [0, duration]. speed scales how fast the clock runs
// (0 pauses it); it is never negative.
struct Timeline
{
    double time = 0.0;     // global seconds
    double speed = 1.0;    // >= 0; 0 = paused
    double duration = 0.0; // global length, from timelineDuration()
    bool loop = false;     // wrap at duration instead of stopping there
};

// Throws std::invalid_argument if the source cannot be played: a non-finite or
// non-positive duration, trimIn < 0, trimOut <= trimIn, trimOut past duration,
// frameCount < 1, a constant fps that is not positive and finite, or a frameTimes
// list that is the wrong length, non-finite or not strictly increasing. Static
// images are checked for the trim and duration fields only.
void validateSource(const TimeSource & source);

// Global length of the timeline: the latest trimOut - timeOffset over the video
// sources (the global time at which that source reaches the end of its trim range),
// and 0 when there are none. Static images do not extend it. Validates every video
// source.
double timelineDuration(const std::vector<TimeSource> & sources);

// Source time shown at a global time, after the offset and after clamping or
// wrapping to the trim range. A NaN or infinite global time is treated as 0.
double sourceTimeAt(const TimeSource & source, double globalTime);

// Frame index at a source time. Times up to float error on a frame boundary show
// that frame. Static images and frameCount 1 always return 0.
int64_t frameIndexAtSourceTime(const TimeSource & source, double sourceTime);

// Frame index at a global time: frameIndexAtSourceTime(source, sourceTimeAt(...)).
int64_t frameIndexAt(const TimeSource & source, double globalTime);

// Advances the global clock by dtSeconds * speed. A NaN, infinite or negative dt
// leaves the time where it is (except that a time outside [0, duration] is
// brought back in). With loop the time wraps into [0, duration); without loop it
// stops at duration. A duration of 0 keeps the time at 0. Throws
// std::invalid_argument for a speed that is negative or not finite, or a
// duration that is negative or not finite.
void advance(Timeline & timeline, double dtSeconds);

} // namespace sabercat::compare
