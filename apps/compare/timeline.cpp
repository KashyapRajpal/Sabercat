// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "timeline.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace sabercat::compare
{

namespace
{

// Slack that lets a time on a frame boundary, up to float error, show that frame:
// in frame units for a constant rate, in seconds for a frame list.
constexpr double kIndexEpsilon = 1e-9;
constexpr double kTimeEpsilon = 1e-9;

[[noreturn]] void refuse(const std::string & message)
{
    throw std::invalid_argument("timeline: " + message);
}

double finiteOrZero(double value)
{
    return std::isfinite(value) ? value : 0.0;
}

// x modulo m, in [0, m), for finite x and m > 0. Rounding at the edges folds to 0.
double wrap(double x, double m)
{
    double r = x - m * std::floor(x / m);
    if (!(r >= 0.0) || r >= m)
    {
        r = 0.0;
    }
    return r;
}

double clampTo(double value, double low, double high)
{
    return std::min(std::max(value, low), high);
}

} // namespace

void validateSource(const TimeSource & source)
{
    if (!std::isfinite(source.duration) || !(source.duration > 0.0))
    {
        refuse("duration must be a positive, finite number of seconds");
    }
    if (!std::isfinite(source.trimIn) || !std::isfinite(source.trimOut) || !std::isfinite(source.timeOffset))
    {
        refuse("trim and offset must be finite");
    }
    if (source.trimIn < 0.0)
    {
        refuse("trimIn must not be negative");
    }
    if (!(source.trimOut > source.trimIn))
    {
        refuse("trimOut must be greater than trimIn");
    }
    if (source.trimOut > source.duration)
    {
        refuse("trimOut is past the end of the source");
    }
    if (source.kind == TimeSourceKind::StaticImage)
    {
        return;
    }
    if (source.frameCount < 1)
    {
        refuse("frameCount must be at least 1");
    }
    if (!source.frameTimes.empty())
    {
        if (static_cast<int64_t>(source.frameTimes.size()) != source.frameCount)
        {
            refuse("frameTimes must have frameCount entries");
        }
        double previous = 0.0;
        for (size_t i = 0; i < source.frameTimes.size(); ++i)
        {
            const double t = source.frameTimes[i];
            if (!std::isfinite(t))
            {
                refuse("frameTimes must be finite");
            }
            if (i > 0 && !(t > previous))
            {
                refuse("frameTimes must be strictly increasing");
            }
            previous = t;
        }
    }
    else if (!std::isfinite(source.fps) || !(source.fps > 0.0))
    {
        refuse("fps must be a positive, finite number when frameTimes is empty");
    }
}

double timelineDuration(const std::vector<TimeSource> & sources)
{
    double end = 0.0;
    for (const TimeSource & source : sources)
    {
        validateSource(source);
        if (source.kind == TimeSourceKind::Video)
        {
            end = std::max(end, source.trimOut - source.timeOffset);
        }
    }
    return end;
}

double sourceTimeAt(const TimeSource & source, double globalTime)
{
    const double time = finiteOrZero(globalTime) + source.timeOffset;
    if (source.loop)
    {
        // Wraps inside [trimIn, trimOut): the trim range length is positive by validation.
        return source.trimIn + wrap(time - source.trimIn, source.trimOut - source.trimIn);
    }
    return clampTo(time, source.trimIn, source.trimOut);
}

int64_t frameIndexAtSourceTime(const TimeSource & source, double sourceTime)
{
    if (source.kind == TimeSourceKind::StaticImage || source.frameCount <= 1)
    {
        return 0;
    }
    const double last = static_cast<double>(source.frameCount - 1);
    const double time = finiteOrZero(sourceTime);

    double index = 0.0;
    if (source.frameTimes.empty())
    {
        index = std::floor(time * source.fps + kIndexEpsilon);
    }
    else
    {
        // The first frame after the time, less one: the last frame at or before it.
        const auto next = std::upper_bound(source.frameTimes.begin(), source.frameTimes.end(), time + kTimeEpsilon);
        index = static_cast<double>(next - source.frameTimes.begin()) - 1.0;
    }
    return static_cast<int64_t>(clampTo(index, 0.0, last));
}

int64_t frameIndexAt(const TimeSource & source, double globalTime)
{
    return frameIndexAtSourceTime(source, sourceTimeAt(source, globalTime));
}

void advance(Timeline & timeline, double dtSeconds)
{
    if (!std::isfinite(timeline.speed) || timeline.speed < 0.0)
    {
        refuse("speed must be a finite number, 0 or more");
    }
    if (!std::isfinite(timeline.duration) || timeline.duration < 0.0)
    {
        refuse("duration must be a finite number, 0 or more");
    }
    if (timeline.duration <= 0.0)
    {
        timeline.time = 0.0;
        return;
    }

    double time = finiteOrZero(timeline.time);
    if (std::isfinite(dtSeconds) && dtSeconds > 0.0)
    {
        const double step = dtSeconds * timeline.speed;
        // A step that overflows is past any duration: it lands at the end.
        time = std::isfinite(step) ? time + step : timeline.duration;
    }

    if (timeline.loop)
    {
        timeline.time = wrap(time, timeline.duration);
    }
    else
    {
        timeline.time = clampTo(time, 0.0, timeline.duration);
    }
}

} // namespace sabercat::compare
