// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "compare_project.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iterator>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include "sabercat/version.h"

// The file layout: one JSON object with "format", "formatVersion", "savedWith" and
// the state groups. Keys are written in one fixed order, so the same project always
// gives the same bytes.

namespace sabercat::compare::project
{

namespace
{

namespace fs = std::filesystem;
using Json = nlohmann::ordered_json;

constexpr const char * kFormat = "sabercat.compare";
constexpr int64_t kFormatVersion = 1;
constexpr size_t kMaxIdLength = 64;

[[noreturn]] void fail(const std::string & message)
{
    throw std::runtime_error("compare project: " + message);
}

// A name-to-value pair for an enum stored as a JSON string.
template <typename E>
struct Choice
{
    const char * name;
    E value;
};

constexpr Choice<SourceKind> kKinds[] = {
    {"image", SourceKind::Image}, {"scene", SourceKind::Scene}, {"video", SourceKind::Video}};
constexpr Choice<SourceRole> kRoles[] = {{"A", SourceRole::A}, {"B", SourceRole::B}, {"none", SourceRole::None}};
constexpr Choice<ToneMapper> kToneMappers[] = {
    {"aces", ToneMapper::Aces}, {"hable", ToneMapper::Hable}, {"reinhard", ToneMapper::Reinhard}};
constexpr Choice<ViewMode> kViewModes[] = {{"split", ViewMode::Split}, {"single", ViewMode::Single}};
constexpr Choice<GridLayout> kLayouts[] = {{"row", GridLayout::Row}, {"grid", GridLayout::Grid}};
constexpr Choice<WipeOrientation> kOrientations[] = {
    {"horizontal", WipeOrientation::Horizontal}, {"vertical", WipeOrientation::Vertical}};
constexpr Choice<ZoomMode> kZoomModes[] = {{"fit", ZoomMode::Fit}, {"percent", ZoomMode::Percent}};
constexpr Choice<Anchor> kAnchors[] = {
    {"topLeft", Anchor::TopLeft}, {"centered", Anchor::Centered}, {"custom", Anchor::Custom}};
constexpr Choice<FlipMode> kFlipModes[] = {{"ldr", FlipMode::Ldr}, {"hdr", FlipMode::Hdr}};
constexpr Choice<SingleViewMode> kSingleViews[] = {
    {"a", SingleViewMode::A},
    {"b", SingleViewMode::B},
    {"wipe", SingleViewMode::Wipe},
    {"flicker", SingleViewMode::Flicker},
    {"diff", SingleViewMode::Diff},
    {"depth", SingleViewMode::Depth},
    {"alpha", SingleViewMode::Alpha},
    {"depthDiff", SingleViewMode::DepthDiff},
    {"alphaDiff", SingleViewMode::AlphaDiff},
    {"flip", SingleViewMode::Flip}};
// The map source is a bool in the struct: false is A, true is B.
constexpr Choice<bool> kMapSources[] = {{"a", false}, {"b", true}};

template <typename E, size_t N>
const char * nameOf(E value, const Choice<E> (&choices)[N])
{
    for (const Choice<E> & choice : choices)
    {
        if (choice.value == value)
        {
            return choice.name;
        }
    }
    return "";
}

template <typename E, size_t N>
std::string choiceList(const Choice<E> (&choices)[N])
{
    std::string text;
    for (const Choice<E> & choice : choices)
    {
        if (!text.empty())
        {
            text += ", ";
        }
        text += '"';
        text += choice.name;
        text += '"';
    }
    return text;
}

// The key path of a member: "sources[0].trimIn", or just "format" at the top level.
std::string keyPath(const std::string & where, const char * key)
{
    return where.empty() ? std::string(key) : where + "." + key;
}

bool isKnown(const std::string & key, std::initializer_list<const char *> known)
{
    return std::any_of(known.begin(), known.end(), [&key](const char * name) { return key == name; });
}

// Identifiers: 1 to 64 characters of [A-Za-z0-9_-], as in src/renderer/project.cpp.
bool validId(const std::string & id)
{
    if (id.empty() || id.size() > kMaxIdLength)
    {
        return false;
    }
    return std::all_of(id.begin(), id.end(), [](char c) {
        return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
    });
}

// UTF-8 text to a path. u8path keeps non-ASCII names intact on Windows.
fs::path pathFromUtf8(const std::string & text)
{
    return fs::u8path(text);
}

// The path as UTF-8 with forward slashes. Copying through the byte range works the
// same whether u8string() yields std::string or std::u8string.
std::string utf8Of(const fs::path & path)
{
    const auto text = path.generic_u8string();
    return std::string(text.begin(), text.end());
}

// The project folder as an absolute-looking, normalized path with no trailing
// separator, so that lexically_relative compares whole components.
fs::path folderPath(const std::string & folder)
{
    fs::path normal = pathFromUtf8(folder).lexically_normal();
    if (!normal.empty() && !normal.has_filename())
    {
        normal = normal.parent_path();
    }
    return normal;
}

// The path as it is stored: relative to the folder when possible, else absolute.
std::string storedPath(const std::string & path, const fs::path & folder)
{
    const fs::path normal = pathFromUtf8(path).lexically_normal();
    if (normal.is_relative())
    {
        return utf8Of(normal);
    }
    const fs::path relative = normal.lexically_relative(folder);
    if (relative.empty())
    {
        return utf8Of(normal); // another drive or root: no relative form exists
    }
    return utf8Of(relative);
}

// The absolute path for a stored path. otherDrive is set when the stored path is
// absolute but has no relative form from the folder.
std::string resolvedPath(const std::string & stored, const fs::path & folder, bool & otherDrive)
{
    const fs::path given = pathFromUtf8(stored);
    otherDrive = false;
    if (given.is_relative())
    {
        return utf8Of((folder / given).lexically_normal());
    }
    otherDrive = given.lexically_relative(folder).empty();
    return utf8Of(given.lexically_normal());
}

// A number that must be finite. JSON cannot hold NaN or infinity, but a value can
// overflow when it is narrowed to float.
template <typename T>
void checkFinite(T value, const std::string & name)
{
    if (!std::isfinite(static_cast<double>(value)))
    {
        fail(name + " must be a finite number");
    }
}

// Reads one JSON object. Members that are absent keep the caller's default, so every
// optional key falls back cleanly. Unknown keys add a warning naming the key path.
class Fields
{
public:
    Fields(const Json & json, std::string where, std::initializer_list<const char *> known,
           std::vector<std::string> & warnings)
        : json_(json), where_(std::move(where)), warnings_(warnings)
    {
        if (!json_.is_object())
        {
            fail(where_ + " must be an object");
        }
        for (auto it = json_.begin(); it != json_.end(); ++it)
        {
            const std::string key = it.key();
            if (!isKnown(key, known))
            {
                warnings_.push_back("unknown key '" + keyPath(where_, key.c_str()) + "' ignored");
            }
        }
    }

    // The object under key, or an empty object if it is absent.
    Fields child(const char * key, std::initializer_list<const char *> known) const
    {
        static const Json kEmpty = Json::object();
        const Json * value = find(key);
        return Fields(value ? *value : kEmpty, keyPath(where_, key), known, warnings_);
    }

    const Json * find(const char * key) const
    {
        const auto it = json_.find(key);
        return it == json_.end() ? nullptr : &(*it);
    }

    const Json & required(const char * key) const
    {
        const Json * value = find(key);
        if (!value)
        {
            fail("missing " + keyPath(where_, key));
        }
        return *value;
    }

    std::string requiredString(const char * key) const
    {
        const Json & value = required(key);
        if (!value.is_string())
        {
            fail(keyPath(where_, key) + " must be a string");
        }
        return value.get<std::string>();
    }

    void optString(const char * key, std::string & out) const
    {
        if (const Json * value = find(key))
        {
            if (!value->is_string())
            {
                fail(keyPath(where_, key) + " must be a string");
            }
            out = value->get<std::string>();
        }
    }

    void optBool(const char * key, bool & out) const
    {
        if (const Json * value = find(key))
        {
            if (!value->is_boolean())
            {
                fail(keyPath(where_, key) + " must be true or false");
            }
            out = value->get<bool>();
        }
    }

    void optDouble(const char * key, double & out) const
    {
        if (const Json * value = find(key))
        {
            if (!value->is_number())
            {
                fail(keyPath(where_, key) + " must be a number");
            }
            out = value->get<double>();
        }
    }

    // Narrowed to float. A value outside float's range becomes infinite and is refused
    // by checkProject.
    void optFloat(const char * key, float & out) const
    {
        double value = out;
        optDouble(key, value);
        out = static_cast<float>(value);
    }

    void optInt(const char * key, int32_t & out) const
    {
        if (const Json * value = find(key))
        {
            if (!value->is_number_integer())
            {
                fail(keyPath(where_, key) + " must be an integer");
            }
            const int64_t number = value->get<int64_t>();
            if (number < INT32_MIN || number > INT32_MAX)
            {
                fail(keyPath(where_, key) + " is out of range");
            }
            out = static_cast<int32_t>(number);
        }
    }

    void optOptionalFloat(const char * key, std::optional<float> & out) const
    {
        if (find(key))
        {
            float value = 0.0f;
            optFloat(key, value);
            out = value;
        }
    }

    void optOptionalInt(const char * key, std::optional<int32_t> & out) const
    {
        if (find(key))
        {
            int32_t value = 0;
            optInt(key, value);
            out = value;
        }
    }

    template <typename E, size_t N>
    void optEnum(const char * key, const Choice<E> (&choices)[N], E & out) const
    {
        if (const Json * value = find(key))
        {
            out = enumFrom(*value, choices, keyPath(where_, key));
        }
    }

    template <typename E, size_t N>
    void requiredEnum(const char * key, const Choice<E> (&choices)[N], E & out) const
    {
        out = enumFrom(required(key), choices, keyPath(where_, key));
    }

    // A {min, max} range object under key.
    void optRange(const char * key, FloatRange & out) const
    {
        const Fields range = child(key, {"min", "max"});
        range.optFloat("min", out.min);
        range.optFloat("max", out.max);
    }

private:
    template <typename E, size_t N>
    static E enumFrom(const Json & value, const Choice<E> (&choices)[N], const std::string & name)
    {
        if (value.is_string())
        {
            const std::string text = value.get<std::string>();
            for (const Choice<E> & choice : choices)
            {
                if (text == choice.name)
                {
                    return choice.value;
                }
            }
        }
        fail(name + " must be one of " + choiceList(choices));
    }

    const Json & json_;
    std::string where_;
    std::vector<std::string> & warnings_;
};

// JSON number from a float. The text is the shortest decimal that reads back as the
// same float (std::to_chars), so 0.1f is written as 0.1 and not as its widened value.
// That text is parsed to a double, which nlohmann writes back as the same decimal.
// Reading the file gives a double that narrows to the identical float.
Json number(float value)
{
    char text[32];
    const std::to_chars_result result = std::to_chars(text, text + sizeof(text), value);
    double widened = 0.0;
    static_cast<void>(std::from_chars(text, result.ptr, widened)); // always parses: finite shortest text
    return Json(widened);
}

Json rangeJson(const FloatRange & range)
{
    Json json = Json::object();
    json["min"] = number(range.min);
    json["max"] = number(range.max);
    return json;
}

// Throws unless the whole project is valid. Used on save, and on load after the
// file has been read, so a bad file and a bad in-memory project are refused alike.
void checkSource(const CompareSource & source, const std::string & where)
{
    if (!validId(source.id))
    {
        fail(where + ".id must be 1 to 64 characters of [A-Za-z0-9_-]");
    }
    if (source.path.empty())
    {
        fail(where + ".path must not be empty");
    }
    checkFinite(source.background.r, where + ".background.r");
    checkFinite(source.background.g, where + ".background.g");
    checkFinite(source.background.b, where + ".background.b");
    checkFinite(source.exposure, where + ".exposure");
    if (source.shCap < -1)
    {
        fail(where + ".shCap must be -1 (none) or more");
    }
    checkFinite(source.trimIn, where + ".trimIn");
    if (source.trimIn < 0.0)
    {
        fail(where + ".trimIn must not be negative");
    }
    checkFinite(source.trimOut, where + ".trimOut");
    if (source.trimOut >= 0.0 && !(source.trimOut > source.trimIn))
    {
        fail(where + ".trimOut must be greater than trimIn (or negative for the end of the clip)");
    }
    checkFinite(source.timeOffset, where + ".timeOffset");
}

void checkCamera(const CompareCamera & camera)
{
    checkFinite(camera.pivot.x, "camera.pivot.x");
    checkFinite(camera.pivot.y, "camera.pivot.y");
    checkFinite(camera.pivot.z, "camera.pivot.z");
    checkFinite(camera.orientation.w, "camera.orientation.w");
    checkFinite(camera.orientation.x, "camera.orientation.x");
    checkFinite(camera.orientation.y, "camera.orientation.y");
    checkFinite(camera.orientation.z, "camera.orientation.z");
    checkFinite(camera.distance, "camera.distance");
    checkFinite(camera.fx, "camera.fx");
    checkFinite(camera.fy, "camera.fy");
    checkFinite(camera.cx, "camera.cx");
    checkFinite(camera.cy, "camera.cy");
    if (camera.width < 0 || camera.height < 0)
    {
        fail("camera.width and camera.height must not be negative");
    }
}

void checkView(const CompareView & view)
{
    checkFinite(view.ranges.diff.min, "view.ranges.diff.min");
    checkFinite(view.ranges.diff.max, "view.ranges.diff.max");
    checkFinite(view.ranges.flip.min, "view.ranges.flip.min");
    checkFinite(view.ranges.flip.max, "view.ranges.flip.max");
    checkFinite(view.ranges.exposure.min, "view.ranges.exposure.min");
    checkFinite(view.ranges.exposure.max, "view.ranges.exposure.max");
    checkFinite(view.ranges.depth.min, "view.ranges.depth.min");
    checkFinite(view.ranges.depth.max, "view.ranges.depth.max");
    checkFinite(view.wipe.position, "view.wipe.position");
    checkFinite(view.zoom.percent, "view.zoom.percent");
    checkFinite(view.pan.x, "view.pan.x");
    checkFinite(view.pan.y, "view.pan.y");
    checkFinite(view.diffGain, "view.diffGain");
    checkFinite(view.flickerPeriodSeconds, "view.flickerPeriodSeconds");
    checkFinite(view.splitDivider, "view.splitDivider");
    if (!(view.zoom.percent > 0.0f))
    {
        fail("view.zoom.percent must be greater than 0");
    }
    if (!(view.flickerPeriodSeconds > 0.0f))
    {
        fail("view.flickerPeriodSeconds must be greater than 0");
    }
    if (view.paneCount < 1 || view.paneCount > 4)
    {
        fail("view.paneCount must be from 1 to 4");
    }
}

void checkFlip(const CompareFlip & flip)
{
    if (flip.startExposure)
    {
        checkFinite(*flip.startExposure, "flip.startExposure");
    }
    if (flip.stopExposure)
    {
        checkFinite(*flip.stopExposure, "flip.stopExposure");
    }
    if (flip.numExposures && *flip.numExposures < 1)
    {
        fail("flip.numExposures must be 1 or more");
    }
    if (flip.ppd < 0)
    {
        fail("flip.ppd must not be negative (0 = default)");
    }
    // The negated form also refuses NaN; an infinity (from an overflowing number) fails the upper bound.
    if (!(flip.liveHz == 0.0f || (flip.liveHz >= kMinLiveFlipHz && flip.liveHz <= kMaxLiveFlipHz)))
    {
        fail("flip.liveHz must be 0 (the default) or from 0.1 to 60");
    }
}

void checkTimeline(const CompareTimeline & timeline)
{
    checkFinite(timeline.time, "timeline.time");
    checkFinite(timeline.speed, "timeline.speed");
    if (timeline.time < 0.0)
    {
        fail("timeline.time must not be negative");
    }
    if (timeline.speed < 0.0)
    {
        fail("timeline.speed must not be negative (0 = paused)");
    }
}

// At most one source may hold role A and one may hold role B. Throws
// std::invalid_argument, naming the two ids, on a second holder.
void checkRoles(const CompareProject & project)
{
    for (const SourceRole role : {SourceRole::A, SourceRole::B})
    {
        const CompareSource * holder = nullptr;
        for (const CompareSource & source : project.sources)
        {
            if (source.role != role)
            {
                continue;
            }
            if (holder)
            {
                throw std::invalid_argument("compare project: two sources have role "
                                            + std::string(nameOf(role, kRoles)) + " ('" + holder->id + "' and '"
                                            + source.id + "')");
            }
            holder = &source;
        }
    }
}

// On load, a second holder of role A or B is cleared to none, one warning per role
// cleared, so the first holder keeps the role.
void enforceSingleRoles(CompareProject & project, std::vector<std::string> & warnings)
{
    for (const SourceRole role : {SourceRole::A, SourceRole::B})
    {
        const CompareSource * holder = nullptr;
        for (CompareSource & source : project.sources)
        {
            if (source.role != role)
            {
                continue;
            }
            if (!holder)
            {
                holder = &source;
                continue;
            }
            warnings.push_back("role " + std::string(nameOf(role, kRoles)) + " on source '" + source.id
                               + "' cleared: source '" + holder->id + "' already has it");
            source.role = SourceRole::None;
        }
    }
}

void checkProject(const CompareProject & project)
{
    if (project.sources.size() > kMaxCompareSources)
    {
        fail("sources has " + std::to_string(project.sources.size()) + " entries; at most "
             + std::to_string(kMaxCompareSources) + " are allowed");
    }
    std::set<std::string> ids;
    for (size_t i = 0; i < project.sources.size(); ++i)
    {
        const CompareSource & source = project.sources[i];
        const std::string where = "sources[" + std::to_string(i) + "]";
        checkSource(source, where);
        if (!ids.insert(source.id).second)
        {
            fail(where + ".id: duplicate source id '" + source.id + "'");
        }
    }
    checkRoles(project);
    checkCamera(project.camera);
    checkView(project.view);
    checkFlip(project.flip);
    checkTimeline(project.timeline);
}

Json sourceJson(const CompareSource & source, const fs::path & folder)
{
    Json json = Json::object();
    json["id"] = source.id;
    json["kind"] = nameOf(source.kind, kKinds);
    json["path"] = storedPath(source.path, folder);
    json["role"] = nameOf(source.role, kRoles);

    Json offset = Json::object();
    offset["x"] = source.offset.x;
    offset["y"] = source.offset.y;
    json["offset"] = offset;

    Json background = Json::object();
    background["r"] = number(source.background.r);
    background["g"] = number(source.background.g);
    background["b"] = number(source.background.b);
    json["background"] = background;

    json["exposure"] = number(source.exposure);
    json["toneMapper"] = nameOf(source.toneMapper, kToneMappers);
    json["flipUpAxis"] = source.flipUpAxis;
    json["shCap"] = source.shCap;
    json["antialias"] = source.antialias;
    json["debugView"] = source.debugView;
    json["trimIn"] = source.trimIn;
    json["trimOut"] = source.trimOut;
    json["loop"] = source.loop;
    json["timeOffset"] = source.timeOffset;
    json["visible"] = source.visible;
    return json;
}

Json cameraJson(const CompareCamera & camera)
{
    Json pivot = Json::object();
    pivot["x"] = number(camera.pivot.x);
    pivot["y"] = number(camera.pivot.y);
    pivot["z"] = number(camera.pivot.z);

    Json orientation = Json::object();
    orientation["w"] = number(camera.orientation.w);
    orientation["x"] = number(camera.orientation.x);
    orientation["y"] = number(camera.orientation.y);
    orientation["z"] = number(camera.orientation.z);

    Json json = Json::object();
    json["pivot"] = pivot;
    json["orientation"] = orientation;
    json["distance"] = number(camera.distance);
    json["fx"] = number(camera.fx);
    json["fy"] = number(camera.fy);
    json["cx"] = number(camera.cx);
    json["cy"] = number(camera.cy);
    json["width"] = camera.width;
    json["height"] = camera.height;
    return json;
}

Json viewJson(const CompareView & view)
{
    Json ranges = Json::object();
    ranges["diff"] = rangeJson(view.ranges.diff);
    ranges["flip"] = rangeJson(view.ranges.flip);
    ranges["exposure"] = rangeJson(view.ranges.exposure);
    ranges["depth"] = rangeJson(view.ranges.depth);

    Json wipe = Json::object();
    wipe["orientation"] = nameOf(view.wipe.orientation, kOrientations);
    wipe["position"] = number(view.wipe.position);

    Json zoom = Json::object();
    zoom["mode"] = nameOf(view.zoom.mode, kZoomModes);
    zoom["percent"] = number(view.zoom.percent);

    Json pan = Json::object();
    pan["x"] = number(view.pan.x);
    pan["y"] = number(view.pan.y);

    Json json = Json::object();
    json["mode"] = nameOf(view.mode, kViewModes);
    json["paneCount"] = view.paneCount;
    json["layout"] = nameOf(view.layout, kLayouts);
    json["wipe"] = wipe;
    json["zoom"] = zoom;
    json["pan"] = pan;
    json["diffGain"] = number(view.diffGain);
    json["ranges"] = ranges;
    json["flickerPeriodSeconds"] = number(view.flickerPeriodSeconds);
    json["anchor"] = nameOf(view.anchor, kAnchors);
    json["singleView"] = nameOf(view.singleView, kSingleViews);
    json["mapSource"] = nameOf(view.mapUseB, kMapSources);
    json["depthAuto"] = view.depthAuto;
    json["splitDivider"] = number(view.splitDivider);
    return json;
}

Json flipJson(const CompareFlip & flip)
{
    Json json = Json::object();
    json["mode"] = nameOf(flip.mode, kFlipModes);
    json["ppd"] = flip.ppd;
    // The live rate is written only when it is not the default, so a project that never changed
    // it saves the same bytes as before.
    if (flip.liveHz != 0.0f)
    {
        json["liveHz"] = number(flip.liveHz);
    }
    json["toneMapper"] = nameOf(flip.toneMapper, kToneMappers);
    // The overrides are written only when set, so an absent key means "use the default".
    if (flip.startExposure)
    {
        json["startExposure"] = number(*flip.startExposure);
    }
    if (flip.stopExposure)
    {
        json["stopExposure"] = number(*flip.stopExposure);
    }
    if (flip.numExposures)
    {
        json["numExposures"] = *flip.numExposures;
    }
    return json;
}

Json timelineJson(const CompareTimeline & timeline)
{
    Json json = Json::object();
    json["time"] = timeline.time;
    json["speed"] = timeline.speed;
    json["loop"] = timeline.loop;
    return json;
}

CompareSource readSource(const Json & json, const std::string & where, const fs::path & folder,
                         std::vector<std::string> & warnings)
{
    const Fields fields(json, where,
                        {"id", "kind", "path", "role", "offset", "background", "exposure", "toneMapper",
                         "flipUpAxis", "shCap", "antialias", "debugView", "trimIn", "trimOut", "loop",
                         "timeOffset", "visible"},
                        warnings);
    CompareSource source;
    source.id = fields.requiredString("id");
    fields.requiredEnum("kind", kKinds, source.kind);
    const std::string stored = fields.requiredString("path");
    fields.optEnum("role", kRoles, source.role);
    fields.optEnum("toneMapper", kToneMappers, source.toneMapper);
    fields.optBool("flipUpAxis", source.flipUpAxis);
    fields.optInt("shCap", source.shCap);
    fields.optBool("antialias", source.antialias);
    fields.optString("debugView", source.debugView);
    fields.optDouble("trimIn", source.trimIn);
    fields.optDouble("trimOut", source.trimOut);
    fields.optBool("loop", source.loop);
    fields.optDouble("timeOffset", source.timeOffset);
    fields.optBool("visible", source.visible);
    fields.optFloat("exposure", source.exposure);

    const Fields offset = fields.child("offset", {"x", "y"});
    offset.optInt("x", source.offset.x);
    offset.optInt("y", source.offset.y);

    const Fields background = fields.child("background", {"r", "g", "b"});
    background.optFloat("r", source.background.r);
    background.optFloat("g", source.background.g);
    background.optFloat("b", source.background.b);

    bool otherDrive = false;
    source.path = resolvedPath(stored, folder, otherDrive);
    source.pathIsAbsoluteOtherDrive = otherDrive;
    if (otherDrive)
    {
        warnings.push_back("source '" + source.id + "' is on another drive; the path is absolute");
    }
    std::error_code ec;
    source.missing = !fs::exists(pathFromUtf8(source.path), ec);
    return source;
}

CompareCamera readCamera(const Fields & root)
{
    const Fields fields = root.child("camera", {"pivot", "orientation", "distance", "fx", "fy", "cx", "cy", "width", "height"});
    CompareCamera camera;
    const Fields pivot = fields.child("pivot", {"x", "y", "z"});
    pivot.optFloat("x", camera.pivot.x);
    pivot.optFloat("y", camera.pivot.y);
    pivot.optFloat("z", camera.pivot.z);
    const Fields orientation = fields.child("orientation", {"w", "x", "y", "z"});
    orientation.optFloat("w", camera.orientation.w);
    orientation.optFloat("x", camera.orientation.x);
    orientation.optFloat("y", camera.orientation.y);
    orientation.optFloat("z", camera.orientation.z);
    fields.optFloat("distance", camera.distance);
    fields.optFloat("fx", camera.fx);
    fields.optFloat("fy", camera.fy);
    fields.optFloat("cx", camera.cx);
    fields.optFloat("cy", camera.cy);
    fields.optInt("width", camera.width);
    fields.optInt("height", camera.height);
    return camera;
}

CompareView readView(const Fields & root)
{
    const Fields fields = root.child("view", {"mode", "paneCount", "layout", "wipe", "zoom", "pan", "diffGain",
                                              "ranges", "flickerPeriodSeconds", "anchor", "singleView",
                                              "mapSource", "depthAuto", "splitDivider"});
    CompareView view;
    fields.optEnum("mode", kViewModes, view.mode);
    fields.optInt("paneCount", view.paneCount);
    fields.optEnum("layout", kLayouts, view.layout);
    fields.optEnum("anchor", kAnchors, view.anchor);
    // Files written before the single view had these keys load with the defaults.
    fields.optEnum("singleView", kSingleViews, view.singleView);
    fields.optEnum("mapSource", kMapSources, view.mapUseB);
    fields.optBool("depthAuto", view.depthAuto);
    // Files saved before the divider had this key keep the default, the middle.
    fields.optFloat("splitDivider", view.splitDivider);
    fields.optFloat("diffGain", view.diffGain);
    fields.optFloat("flickerPeriodSeconds", view.flickerPeriodSeconds);

    const Fields wipe = fields.child("wipe", {"orientation", "position"});
    wipe.optEnum("orientation", kOrientations, view.wipe.orientation);
    wipe.optFloat("position", view.wipe.position);

    const Fields zoom = fields.child("zoom", {"mode", "percent"});
    zoom.optEnum("mode", kZoomModes, view.zoom.mode);
    zoom.optFloat("percent", view.zoom.percent);

    const Fields pan = fields.child("pan", {"x", "y"});
    pan.optFloat("x", view.pan.x);
    pan.optFloat("y", view.pan.y);

    const Fields ranges = fields.child("ranges", {"diff", "flip", "exposure", "depth"});
    ranges.optRange("diff", view.ranges.diff);
    ranges.optRange("flip", view.ranges.flip);
    ranges.optRange("exposure", view.ranges.exposure);
    ranges.optRange("depth", view.ranges.depth);
    return view;
}

CompareFlip readFlip(const Fields & root)
{
    const Fields fields = root.child("flip", {"mode", "ppd", "liveHz", "toneMapper", "startExposure", "stopExposure",
                                              "numExposures"});
    CompareFlip flip;
    fields.optEnum("mode", kFlipModes, flip.mode);
    fields.optInt("ppd", flip.ppd);
    fields.optFloat("liveHz", flip.liveHz);
    fields.optEnum("toneMapper", kToneMappers, flip.toneMapper);
    fields.optOptionalFloat("startExposure", flip.startExposure);
    fields.optOptionalFloat("stopExposure", flip.stopExposure);
    fields.optOptionalInt("numExposures", flip.numExposures);
    return flip;
}

CompareTimeline readTimeline(const Fields & root)
{
    const Fields fields = root.child("timeline", {"time", "speed", "loop"});
    CompareTimeline timeline;
    fields.optDouble("time", timeline.time);
    fields.optDouble("speed", timeline.speed);
    fields.optBool("loop", timeline.loop);
    return timeline;
}

// Reads and checks a project from its JSON text. folder is normalized and absolute.
CompareProject parseInternal(const std::string & text, const fs::path & folder, std::vector<std::string> & warnings)
{
    Json root;
    try
    {
        root = Json::parse(text);
    }
    catch (const Json::exception & error)
    {
        fail(std::string("the text is not valid JSON (") + error.what() + ")");
    }
    if (!root.is_object())
    {
        fail("the top level must be a JSON object");
    }

    const Fields fields(root, "",
                        {"format", "formatVersion", "savedWith", "sources", "camera", "view", "flip", "timeline"},
                        warnings);
    if (fields.requiredString("format") != kFormat)
    {
        fail("not a Sabercat compare project (format \"" + fields.requiredString("format") + "\")");
    }
    const Json & versionJson = fields.required("formatVersion");
    if (!versionJson.is_number_integer())
    {
        fail("formatVersion must be an integer");
    }
    const int64_t version = versionJson.get<int64_t>();
    if (version > kFormatVersion)
    {
        fail("project format version " + std::to_string(version) + " is newer than this Sabercat reads ("
             + std::to_string(kFormatVersion) + "); update Sabercat");
    }
    if (version < 1)
    {
        fail("formatVersion " + std::to_string(version) + " is invalid");
    }

    // savedWith is informational. Its object is checked, its contents are not used.
    fields.child("savedWith", {"sabercat"});

    CompareProject project;
    if (const Json * list = fields.find("sources"))
    {
        if (!list->is_array())
        {
            fail("sources must be an array");
        }
        if (list->size() > kMaxCompareSources)
        {
            fail("sources has " + std::to_string(list->size()) + " entries; at most "
                 + std::to_string(kMaxCompareSources) + " are allowed");
        }
        for (size_t i = 0; i < list->size(); ++i)
        {
            project.sources.push_back(
                readSource((*list)[i], "sources[" + std::to_string(i) + "]", folder, warnings));
        }
    }
    enforceSingleRoles(project, warnings);
    project.camera = readCamera(fields);
    project.view = readView(fields);
    project.flip = readFlip(fields);
    project.timeline = readTimeline(fields);
    checkProject(project);
    return project;
}

} // namespace

bool setSourceRole(CompareProject & project, size_t index, SourceRole role)
{
    if (index >= project.sources.size())
    {
        return false;
    }
    if (role != SourceRole::None)
    {
        for (size_t i = 0; i < project.sources.size(); ++i)
        {
            if (i != index && project.sources[i].role == role)
            {
                project.sources[i].role = SourceRole::None;
            }
        }
    }
    project.sources[index].role = role;
    return true;
}

std::string serializeCompareProject(const CompareProject & project, const std::string & projectFolder)
{
    checkProject(project);
    const fs::path folder = folderPath(projectFolder);

    Json sources = Json::array();
    for (const CompareSource & source : project.sources)
    {
        sources.push_back(sourceJson(source, folder));
    }
    Json savedWith = Json::object();
    savedWith["sabercat"] = sabercat::kVersionString;

    Json root = Json::object();
    root["format"] = kFormat;
    root["formatVersion"] = kFormatVersion;
    root["savedWith"] = savedWith;
    root["sources"] = sources;
    root["camera"] = cameraJson(project.camera);
    root["view"] = viewJson(project.view);
    root["flip"] = flipJson(project.flip);
    root["timeline"] = timelineJson(project.timeline);

    try
    {
        return root.dump(2) + "\n";
    }
    catch (const Json::exception & error)
    {
        fail(std::string("the project cannot be written as JSON (") + error.what() + ")");
    }
}

CompareProject parseCompareProject(const std::string & json, const std::string & projectFolder,
                                   std::vector<std::string> * warnings)
{
    std::vector<std::string> found;
    CompareProject project = parseInternal(json, folderPath(projectFolder), found);
    if (warnings)
    {
        warnings->insert(warnings->end(), found.begin(), found.end());
    }
    return project;
}

std::string saveCompareProject(const std::string & utf8Path, const CompareProject & project)
{
    const fs::path target = pathFromUtf8(utf8Path);
    std::error_code ec;
    fs::path absolute = fs::absolute(target, ec);
    if (ec)
    {
        absolute = target;
    }
    const std::string text = serializeCompareProject(project, utf8Of(absolute.parent_path()));

    // Write beside the destination, then rename over it, so a failure leaves the old file.
    const std::string tempName = utf8Path + ".tmp";
    const fs::path temp = pathFromUtf8(tempName);
    if (fs::is_directory(temp, ec))
    {
        // Not ours to remove, so refuse before the try block cleans up.
        throw std::runtime_error("saveCompareProject: " + tempName + " is a folder; the project was not saved");
    }
    try
    {
        {
            std::ofstream out(temp, std::ios::binary | std::ios::trunc);
            if (!out)
            {
                throw std::runtime_error("cannot create " + tempName);
            }
            out.write(text.data(), static_cast<std::streamsize>(text.size()));
            out.close();
            if (out.fail())
            {
                throw std::runtime_error("cannot write " + tempName);
            }
        }
        fs::rename(temp, target, ec);
        if (ec)
        {
            throw std::runtime_error("cannot replace " + utf8Path + " (" + ec.message() + ")");
        }
    }
    catch (const std::exception & error)
    {
        std::error_code ignored;
        fs::remove(temp, ignored);
        throw std::runtime_error(std::string("saveCompareProject: ") + error.what());
    }
    return utf8Path;
}

CompareProject loadCompareProject(const std::string & utf8Path, std::vector<std::string> * warnings)
{
    const fs::path target = pathFromUtf8(utf8Path);
    std::ifstream in(target, std::ios::binary);
    if (!in)
    {
        throw std::runtime_error("loadCompareProject: cannot open " + utf8Path);
    }
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());

    std::error_code ec;
    fs::path absolute = fs::absolute(target, ec);
    if (ec)
    {
        absolute = target;
    }
    return parseCompareProject(text, utf8Of(absolute.parent_path()), warnings);
}

} // namespace sabercat::compare::project
