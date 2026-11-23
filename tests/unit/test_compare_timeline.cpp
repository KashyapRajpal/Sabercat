// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Playback timing (apps/compare/timeline.h): the global clock, trims, loops, offsets,
// variable frame rate and mixed frame rates. Pure CPU, no file system.
#include "unit_test.h"

#include "timeline.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace sabercat::unit
{

namespace
{

using sabercat::compare::Timeline;
using sabercat::compare::TimeSource;
using sabercat::compare::TimeSourceKind;
using sabercat::compare::advance;
using sabercat::compare::frameIndexAt;
using sabercat::compare::frameIndexAtSourceTime;
using sabercat::compare::sourceTimeAt;
using sabercat::compare::timelineDuration;
using sabercat::compare::validateSource;

// A constant-rate video source with no trim, no offset and no loop.
TimeSource video(double fps, int64_t frameCount, double duration)
{
    TimeSource source;
    source.fps = fps;
    source.frameCount = frameCount;
    source.duration = duration;
    source.trimIn = 0.0;
    source.trimOut = duration;
    return source;
}

// True if the callable throws std::invalid_argument.
template <typename Fn>
bool refuses(Fn fn)
{
    try
    {
        fn();
    }
    catch (const std::invalid_argument &)
    {
        return true;
    }
    catch (...)
    {
    }
    return false;
}

} // namespace

void testCompareTimeline(Context & ctx)
{
    // Speed 0 pauses the clock; speed 2 doubles the step.
    Timeline paused;
    paused.time = 1.0;
    paused.speed = 0.0;
    paused.duration = 3.0;
    advance(paused, 10.0);
    UNIT_CHECK(ctx, paused.time == 1.0);

    Timeline fast;
    fast.speed = 2.0;
    fast.duration = 3.0;
    advance(fast, 0.5);
    UNIT_CHECK_NEAR(ctx, fast.time, 1.0, 1e-12);

    // Advance stops at the duration without loop, and wraps with loop.
    Timeline stop;
    stop.time = 2.5;
    stop.duration = 3.0;
    advance(stop, 1.0);
    UNIT_CHECK_NEAR(ctx, stop.time, 3.0, 1e-12);

    Timeline wrapped = stop;
    wrapped.time = 2.5;
    wrapped.loop = true;
    advance(wrapped, 1.0);
    UNIT_CHECK_NEAR(ctx, wrapped.time, 0.5, 1e-12);

    // A huge step with loop stays in [0, duration).
    Timeline huge = wrapped;
    huge.time = 0.0;
    advance(huge, 1e9);
    UNIT_CHECK(ctx, huge.time >= 0.0 && huge.time < huge.duration);

    // A NaN or negative step does not move the clock; NaN time reads as 0.
    Timeline still;
    still.time = 1.0;
    still.duration = 3.0;
    advance(still, std::numeric_limits<double>::quiet_NaN());
    UNIT_CHECK_NEAR(ctx, still.time, 1.0, 1e-12);
    advance(still, -4.0);
    UNIT_CHECK_NEAR(ctx, still.time, 1.0, 1e-12);

    Timeline nanTime;
    nanTime.time = std::numeric_limits<double>::quiet_NaN();
    nanTime.duration = 3.0;
    advance(nanTime, 0.0);
    UNIT_CHECK(ctx, nanTime.time == 0.0);

    // Bad speed and bad duration are refused.
    Timeline badSpeed;
    badSpeed.speed = -1.0;
    badSpeed.duration = 3.0;
    UNIT_CHECK(ctx, refuses([&] { advance(badSpeed, 0.1); }));

    // Constant-rate frames: index i is shown at time i / fps, also on the boundary.
    const TimeSource clip = video(30.0, 90, 3.0);
    bool boundaries = true;
    for (int64_t i = 0; i < 90; ++i)
    {
        boundaries = boundaries && frameIndexAtSourceTime(clip, static_cast<double>(i) / 30.0) == i;
    }
    UNIT_CHECK(ctx, boundaries);
    UNIT_CHECK(ctx, frameIndexAtSourceTime(clip, 0.0999) == 2);

    // Negative offset delays the source; positive offset advances it.
    TimeSource behind = clip;
    behind.timeOffset = -1.0;
    UNIT_CHECK_NEAR(ctx, sourceTimeAt(behind, 0.5), 0.0, 1e-12);
    UNIT_CHECK(ctx, frameIndexAt(behind, 0.5) == 0);
    UNIT_CHECK(ctx, frameIndexAt(behind, 1.5) == 15);

    TimeSource ahead = clip;
    ahead.timeOffset = 1.0;
    UNIT_CHECK(ctx, frameIndexAt(ahead, 0.0) == 30);
    UNIT_CHECK(ctx, frameIndexAt(ahead, 1.0) == 60); // source 2 s

    // A trim range with no loop clamps: before the start it shows the first frame of the
    // range, after the end the last frame at or before trimOut.
    TimeSource trimmed = clip;
    trimmed.trimIn = 1.0;
    trimmed.trimOut = 2.0;
    UNIT_CHECK(ctx, frameIndexAt(trimmed, -5.0) == 30);
    UNIT_CHECK(ctx, frameIndexAt(trimmed, 0.5) == 30);
    UNIT_CHECK(ctx, frameIndexAt(trimmed, 9.0) == 60);

    // Looping wraps inside [trimIn, trimOut), including for negative times.
    TimeSource looped = trimmed;
    looped.loop = true;
    UNIT_CHECK(ctx, frameIndexAt(looped, 0.0) == 30);
    UNIT_CHECK(ctx, frameIndexAt(looped, 2.0) == 30); // source 2 s is the excluded end: wraps to 1 s
    UNIT_CHECK(ctx, frameIndexAt(looped, 1.5) == 45);
    UNIT_CHECK(ctx, frameIndexAt(looped, -0.5) == 45);
    UNIT_CHECK(ctx, frameIndexAt(looped, -0.25) == 52); // source 1.75 s -> 52.5 -> frame 52
    UNIT_CHECK_NEAR(ctx, sourceTimeAt(looped, -0.25), 1.75, 1e-12);

    // Variable frame rate: frames at the listed timestamps.
    TimeSource vfr;
    vfr.frameTimes = {0.0, 0.1, 0.15, 0.4, 1.0};
    vfr.frameCount = 5;
    vfr.duration = 1.0;
    vfr.trimIn = 0.0;
    vfr.trimOut = 1.0;
    UNIT_CHECK(ctx, !refuses([&] { validateSource(vfr); }));
    UNIT_CHECK(ctx, frameIndexAt(vfr, 0.12) == 1);
    UNIT_CHECK(ctx, frameIndexAt(vfr, 0.15) == 2);
    UNIT_CHECK(ctx, frameIndexAt(vfr, 0.149) == 1);
    UNIT_CHECK(ctx, frameIndexAt(vfr, 0.9) == 3);
    UNIT_CHECK(ctx, frameIndexAt(vfr, 1.0) == 4);
    UNIT_CHECK(ctx, frameIndexAt(vfr, -0.5) == 0);
    UNIT_CHECK(ctx, frameIndexAt(vfr, 0.0) == 0);

    // Two sources at different rates each show their own frame at the same global time.
    const TimeSource fps30 = video(30.0, 90, 3.0);
    const TimeSource fps24 = video(24.0, 72, 3.0);
    UNIT_CHECK(ctx, frameIndexAt(fps30, 1.5) == 45);
    UNIT_CHECK(ctx, frameIndexAt(fps24, 1.5) == 36);
    UNIT_CHECK(ctx, frameIndexAt(fps24, 1.49) == 35);

    // One frame: always index 0, at any time.
    const TimeSource single = video(30.0, 1, 1.0);
    UNIT_CHECK(ctx, !refuses([&] { validateSource(single); }));
    UNIT_CHECK(ctx, frameIndexAt(single, 0.5) == 0);
    UNIT_CHECK(ctx, frameIndexAt(single, 1e9) == 0);

    // A still image is frame 0 at every time, and does not extend the timeline.
    TimeSource still2d = video(0.0, 1, 1.0);
    still2d.fps = 0.0; // not used for a still image
    still2d.kind = TimeSourceKind::StaticImage;
    UNIT_CHECK(ctx, !refuses([&] { validateSource(still2d); }));
    UNIT_CHECK(ctx, frameIndexAt(still2d, 2.5) == 0);
    UNIT_CHECK(ctx, timelineDuration({still2d}) == 0.0);

    // A huge global time does not crash and clamps or wraps as usual.
    UNIT_CHECK(ctx, frameIndexAt(clip, 1e9) == 90 - 1);
    UNIT_CHECK(ctx, frameIndexAt(clip, -1e9) == 0);
    UNIT_CHECK(ctx, frameIndexAt(looped, 1e9) >= 30 && frameIndexAt(looped, 1e9) < 60);

    // A NaN global time reads as 0.
    UNIT_CHECK(ctx, frameIndexAt(clip, std::numeric_limits<double>::quiet_NaN()) == 0);
    UNIT_CHECK(ctx, frameIndexAt(behind, std::numeric_limits<double>::quiet_NaN()) == 0);

    // Timeline duration: the latest trimOut - timeOffset among the video sources.
    TimeSource late = clip;
    late.timeOffset = -1.0;
    UNIT_CHECK_NEAR(ctx, timelineDuration({clip}), 3.0, 1e-12);
    UNIT_CHECK_NEAR(ctx, timelineDuration({clip, late}), 4.0, 1e-12);
    UNIT_CHECK_NEAR(ctx, timelineDuration({}), 0.0, 1e-12);

    // Invalid sources are refused.
    TimeSource inverted = clip;
    inverted.trimIn = 2.0;
    inverted.trimOut = 1.0;
    UNIT_CHECK(ctx, refuses([&] { validateSource(inverted); }));
    UNIT_CHECK(ctx, refuses([&] { timelineDuration({inverted}); }));

    TimeSource empty = clip;
    empty.trimIn = 1.0;
    empty.trimOut = 1.0;
    UNIT_CHECK(ctx, refuses([&] { validateSource(empty); }));

    TimeSource pastEnd = clip;
    pastEnd.trimOut = 4.0;
    UNIT_CHECK(ctx, refuses([&] { validateSource(pastEnd); }));

    TimeSource noRate = clip;
    noRate.fps = 0.0;
    UNIT_CHECK(ctx, refuses([&] { validateSource(noRate); }));

    TimeSource noFrames = clip;
    noFrames.frameCount = 0;
    UNIT_CHECK(ctx, refuses([&] { validateSource(noFrames); }));

    TimeSource wrongList = vfr;
    wrongList.frameCount = 4;
    UNIT_CHECK(ctx, refuses([&] { validateSource(wrongList); }));

    TimeSource notIncreasing = vfr;
    notIncreasing.frameTimes = {0.0, 0.5, 0.5, 0.6, 0.9};
    UNIT_CHECK(ctx, refuses([&] { validateSource(notIncreasing); }));
}

} // namespace sabercat::unit
