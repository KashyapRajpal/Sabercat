// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The Sabercat Compare project file (apps/compare/compare_project.h): the string
// round trip, the refusals, the path rules and the atomic save. The in-memory tests
// need no file; the save and load tests use a folder under the temp directory.
#include "unit_test.h"

#include "compare_project.h"
#include "compare_layout.h"
#include "flip_hdr.h"

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sabercat/version.h"

namespace sabercat::unit
{

namespace
{

namespace fs = std::filesystem;

using sabercat::compare::project::CompareProject;
using sabercat::compare::project::CompareSource;
using sabercat::compare::project::FlipMode;
using sabercat::compare::project::GridLayout;
using sabercat::compare::project::SourceKind;
using sabercat::compare::project::SingleViewMode;
using sabercat::compare::project::SourceRole;
using sabercat::compare::project::ToneMapper;
using sabercat::compare::project::WipeOrientation;
using sabercat::compare::project::ZoomMode;
using sabercat::compare::project::loadCompareProject;
using sabercat::compare::project::parseCompareProject;
using sabercat::compare::project::saveCompareProject;
using sabercat::compare::project::serializeCompareProject;
using sabercat::compare::project::setSourceRole;

std::string utf8(const fs::path & path)
{
    const auto text = path.generic_u8string();
    return std::string(text.begin(), text.end());
}

bool contains(const std::string & text, const std::string & part)
{
    return text.find(part) != std::string::npos;
}

bool hasWarning(const std::vector<std::string> & warnings, const std::string & text)
{
    return std::find(warnings.begin(), warnings.end(), text) != warnings.end();
}

// The message of the error fn throws, or "no error".
template <typename Fn>
std::string errorOf(Fn fn)
{
    try
    {
        fn();
    }
    catch (const std::exception & error)
    {
        return error.what();
    }
    return "no error";
}

std::string readFile(const fs::path & path)
{
    std::ifstream in(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void writeFile(const fs::path & path, const std::string & text)
{
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << text;
}

CompareSource makeSource(const std::string & id, const std::string & path)
{
    CompareSource source;
    source.id = id;
    source.path = path;
    return source;
}

// A project with every group set away from its default, for the round trip.
CompareProject sampleProject()
{
    CompareProject project;

    CompareSource left = makeSource("left", "C:/proj/scene.ply"); // repo-check: allow
    left.kind = SourceKind::Scene;
    left.role = SourceRole::A;
    left.offset = {4, -2};
    left.background = {0.1f, 0.2f, 0.3f};
    left.exposure = -1.5f;
    left.toneMapper = ToneMapper::Reinhard;
    left.flipUpAxis = true;
    left.shCap = 2;
    left.antialias = true;
    left.debugView = "depth";
    left.trimIn = 0.25;
    left.trimOut = 3.5;
    left.loop = true;
    left.timeOffset = -0.75;
    left.visible = false;

    CompareSource right = makeSource("right", "C:/proj/sub/clip.mp4"); // repo-check: allow
    right.kind = SourceKind::Video;
    right.role = SourceRole::B;

    project.sources = {left, right};

    project.camera.pivot = {1.0f, 2.0f, 3.0f};
    project.camera.orientation = {0.5f, 0.5f, 0.5f, 0.5f};
    project.camera.distance = 4.5f;
    project.camera.fx = 800.0f;
    project.camera.fy = 800.0f;
    project.camera.cx = 960.0f;
    project.camera.cy = 540.0f;
    project.camera.width = 1920;
    project.camera.height = 1080;

    project.view.wipe.position = 0.3f;
    project.view.wipe.orientation = WipeOrientation::Horizontal;
    project.view.zoom.mode = ZoomMode::Percent;
    project.view.zoom.percent = 200.0f;
    project.view.pan = {10.0f, -5.0f};
    project.view.diffGain = 2.5f;
    project.view.flickerPeriodSeconds = 0.5f;
    project.view.paneCount = 3;
    project.view.layout = GridLayout::Grid;
    project.view.ranges.exposure = {-2.0f, 2.0f};
    project.view.anchor = sabercat::compare::project::Anchor::Custom;

    project.flip.mode = FlipMode::Hdr;
    project.flip.ppd = 60;
    project.flip.toneMapper = ToneMapper::Hable;
    project.flip.startExposure = 1.25f;
    project.flip.numExposures = 4;

    project.timeline.time = 1.5;
    project.timeline.speed = 0.5;
    project.timeline.loop = true;
    return project;
}

// A source list of the given size, with ids s0, s1, ...
std::string sourceListJson(size_t count)
{
    std::string text = "[";
    for (size_t i = 0; i < count; ++i)
    {
        text += (i ? "," : "");
        text += "{\"id\":\"s" + std::to_string(i) + "\",\"kind\":\"image\",\"path\":\"a.png\"}";
    }
    return text + "]";
}

// True when the two floats have the same bits, so -0.0 and 0.0 differ.
bool sameBits(float a, float b)
{
    return std::memcmp(&a, &b, sizeof(float)) == 0;
}

// True when fn throws std::invalid_argument.
template <typename Fn>
bool throwsInvalidArgument(Fn fn)
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
        return false;
    }
    return false;
}

} // namespace

void testCompareProject(Context & ctx)
{
    const std::string folder = "C:/proj"; // repo-check: allow

    // The round trip is byte-exact: parse(serialize(p)) serializes to the same text.
    const CompareProject original = sampleProject();
    const std::string first = serializeCompareProject(original, folder);
    std::vector<std::string> warnings;
    const CompareProject parsed = parseCompareProject(first, folder, &warnings);
    const std::string second = serializeCompareProject(parsed, folder);
    UNIT_CHECK(ctx, first == second);
    UNIT_CHECK(ctx, warnings.empty());
    UNIT_CHECK(ctx, !first.empty() && first.back() == '\n');
    UNIT_CHECK(ctx, contains(first, "\"format\": \"sabercat.compare\""));
    UNIT_CHECK(ctx, contains(first, "\"formatVersion\": 1"));
    UNIT_CHECK(ctx, contains(first, "\"sabercat\": \"" + std::string(sabercat::kVersionString) + "\""));
    UNIT_CHECK(ctx, contains(first, "\"path\": \"scene.ply\""));
    UNIT_CHECK(ctx, contains(first, "\"path\": \"sub/clip.mp4\""));
    UNIT_CHECK(ctx, parsed.sources.size() == 2u);
    UNIT_CHECK(ctx, parsed.sources[0].path == "C:/proj/scene.ply"); // repo-check: allow
    UNIT_CHECK(ctx, parsed.sources[0].trimIn == 0.25);
    UNIT_CHECK(ctx, parsed.sources[0].background.r == 0.1f);
    UNIT_CHECK(ctx, parsed.sources[0].exposure == -1.5f);
    UNIT_CHECK(ctx, parsed.view.paneCount == 3);
    UNIT_CHECK(ctx, parsed.flip.startExposure && *parsed.flip.startExposure == 1.25f);
    UNIT_CHECK(ctx, parsed.timeline.time == 1.5);

    // The split-screen divider: written, read back, and defaulted to the middle when the key is absent.
    CompareProject divider = sampleProject();
    divider.view.splitDivider = 0.3f;
    const CompareProject dividerBack = parseCompareProject(serializeCompareProject(divider, folder), folder, nullptr);
    UNIT_CHECK(ctx, std::fabs(dividerBack.view.splitDivider - 0.3f) <= 1e-6f);
    UNIT_CHECK(ctx, contains(serializeCompareProject(divider, folder), "\"splitDivider\": 0.3"));
    const CompareProject noDivider = parseCompareProject(
        R"({"format":"sabercat.compare","formatVersion":1,"view":{"paneCount":2}})", folder, nullptr);
    UNIT_CHECK(ctx, noDivider.view.splitDivider == 0.5f);
    // Not finite is refused on save and on load, with the key named.
    CompareProject badDivider = sampleProject();
    badDivider.view.splitDivider = std::numeric_limits<float>::quiet_NaN();
    UNIT_CHECK(ctx, contains(errorOf([&] { serializeCompareProject(badDivider, folder); }), "view.splitDivider"));
    UNIT_CHECK(ctx, contains(errorOf([&] {
                                 parseCompareProject(R"({"format":"sabercat.compare","formatVersion":1,)"
                                                     R"("view":{"splitDivider":1e300}})",
                                                     folder, nullptr);
                             }),
                             "view.splitDivider"));

    // Floats are written as the shortest decimal that reads back as the same float, so
    // awkward values round-trip bit for bit and no widened digits appear in the text.
    CompareProject awkward = sampleProject();
    awkward.sources[0].background = {0.1f, 1.0f / 3.0f, 1e-7f};
    awkward.sources[0].exposure = 16777216.0f;
    awkward.view.diffGain = -0.0f;
    awkward.camera.fx = FLT_MAX;
    awkward.camera.fy = FLT_MIN;
    awkward.camera.cx = std::numeric_limits<float>::denorm_min();
    awkward.flip.startExposure = 0.1f;
    const std::string awkwardText = serializeCompareProject(awkward, folder);
    UNIT_CHECK(ctx, contains(awkwardText, "\"r\": 0.1,"));
    UNIT_CHECK(ctx, contains(awkwardText, "\"g\": 0.33333334,"));
    UNIT_CHECK(ctx, contains(awkwardText, "-0.0"));
    UNIT_CHECK(ctx, !contains(awkwardText, "0.10000000149"));
    const CompareProject awkwardBack = parseCompareProject(awkwardText, folder, nullptr);
    UNIT_CHECK(ctx, sameBits(awkwardBack.sources[0].background.r, 0.1f));
    UNIT_CHECK(ctx, sameBits(awkwardBack.sources[0].background.g, 1.0f / 3.0f));
    UNIT_CHECK(ctx, sameBits(awkwardBack.sources[0].background.b, 1e-7f));
    UNIT_CHECK(ctx, sameBits(awkwardBack.sources[0].exposure, 16777216.0f));
    UNIT_CHECK(ctx, sameBits(awkwardBack.view.diffGain, -0.0f));
    UNIT_CHECK(ctx, sameBits(awkwardBack.camera.fx, FLT_MAX));
    UNIT_CHECK(ctx, sameBits(awkwardBack.camera.fy, FLT_MIN));
    UNIT_CHECK(ctx, sameBits(awkwardBack.camera.cx, std::numeric_limits<float>::denorm_min()));
    UNIT_CHECK(ctx, awkwardBack.flip.startExposure && sameBits(*awkwardBack.flip.startExposure, 0.1f));
    UNIT_CHECK(ctx, serializeCompareProject(awkwardBack, folder) == awkwardText);

    // The single view and the map keys: every mode round-trips by its key, and a
    // project with the map keys set round-trips byte for byte.
    const std::pair<SingleViewMode, const char *> singleViewKeys[] = {
        {SingleViewMode::A, "a"},         {SingleViewMode::B, "b"},
        {SingleViewMode::Wipe, "wipe"},   {SingleViewMode::Flicker, "flicker"},
        {SingleViewMode::Diff, "diff"},   {SingleViewMode::Depth, "depth"},
        {SingleViewMode::Alpha, "alpha"}, {SingleViewMode::DepthDiff, "depthDiff"},
        {SingleViewMode::AlphaDiff, "alphaDiff"}, {SingleViewMode::Flip, "flip"}};
    for (const auto & [mode, key] : singleViewKeys)
    {
        CompareProject modal = sampleProject();
        modal.view.singleView = mode;
        const std::string modalText = serializeCompareProject(modal, folder);
        UNIT_CHECK(ctx, contains(modalText, std::string("\"singleView\": \"") + key + "\""));
        std::vector<std::string> modalWarnings;
        const CompareProject modalBack = parseCompareProject(modalText, folder, &modalWarnings);
        UNIT_CHECK(ctx, modalWarnings.empty());
        UNIT_CHECK(ctx, modalBack.view.singleView == mode);
    }

    CompareProject mapped = sampleProject();
    mapped.view.mode = sabercat::compare::project::ViewMode::Single;
    mapped.view.singleView = SingleViewMode::AlphaDiff;
    mapped.view.mapUseB = true;
    mapped.view.depthAuto = false;
    const std::string mappedText = serializeCompareProject(mapped, folder);
    UNIT_CHECK(ctx, contains(mappedText, "\"mapSource\": \"b\""));
    UNIT_CHECK(ctx, contains(mappedText, "\"depthAuto\": false"));
    std::vector<std::string> mappedWarnings;
    const CompareProject mappedBack = parseCompareProject(mappedText, folder, &mappedWarnings);
    UNIT_CHECK(ctx, mappedWarnings.empty());
    UNIT_CHECK(ctx, mappedBack.view.singleView == SingleViewMode::AlphaDiff);
    UNIT_CHECK(ctx, mappedBack.view.mapUseB);
    UNIT_CHECK(ctx, !mappedBack.view.depthAuto);
    UNIT_CHECK(ctx, serializeCompareProject(mappedBack, folder) == mappedText);

    // A file from before the single view had these keys loads with the defaults and no
    // warnings, and written back it gains "a", "a" and true.
    const std::string oldView = R"({"format":"sabercat.compare","formatVersion":1,"sources":[)"
                                R"({"id":"a","kind":"image","path":"x.png"}],)"
                                R"("view":{"mode":"single","paneCount":2}})";
    std::vector<std::string> oldWarnings;
    const CompareProject oldBack = parseCompareProject(oldView, folder, &oldWarnings);
    UNIT_CHECK(ctx, oldWarnings.empty());
    UNIT_CHECK(ctx, oldBack.view.mode == sabercat::compare::project::ViewMode::Single);
    UNIT_CHECK(ctx, oldBack.view.singleView == SingleViewMode::A);
    UNIT_CHECK(ctx, !oldBack.view.mapUseB);
    UNIT_CHECK(ctx, oldBack.view.depthAuto);
    const std::string oldWritten = serializeCompareProject(oldBack, folder);
    UNIT_CHECK(ctx, contains(oldWritten, "\"singleView\": \"a\""));
    UNIT_CHECK(ctx, contains(oldWritten, "\"mapSource\": \"a\""));
    UNIT_CHECK(ctx, contains(oldWritten, "\"depthAuto\": true"));

    // A bad value is refused, and the message names its key.
    const std::string badSingle = R"({"format":"sabercat.compare","formatVersion":1,"sources":[)"
                                  R"({"id":"a","kind":"image","path":"x.png"}],)"
                                  R"("view":{"singleView":"wide"}})";
    UNIT_CHECK(ctx, contains(errorOf([&] { parseCompareProject(badSingle, folder, nullptr); }), "view.singleView"));
    const std::string badMap = R"({"format":"sabercat.compare","formatVersion":1,"sources":[)"
                               R"({"id":"a","kind":"image","path":"x.png"}],)"
                               R"("view":{"mapSource":"c"}})";
    UNIT_CHECK(ctx, contains(errorOf([&] { parseCompareProject(badMap, folder, nullptr); }), "view.mapSource"));

    // Roles: at most one A and one B. Parse keeps the first holder and clears the rest,
    // with one warning per cleared source.
    const std::string twoA = R"({"format":"sabercat.compare","formatVersion":1,"sources":[)"
                             R"({"id":"a","kind":"image","path":"x.png","role":"A"},)"
                             R"({"id":"b","kind":"image","path":"y.png","role":"B"},)"
                             R"({"id":"c","kind":"image","path":"z.png","role":"A"},)"
                             R"({"id":"d","kind":"image","path":"w.png","role":"B"}]})";
    std::vector<std::string> roleWarnings;
    const CompareProject roled = parseCompareProject(twoA, folder, &roleWarnings);
    UNIT_CHECK(ctx, roled.sources.size() == 4u);
    UNIT_CHECK(ctx, roled.sources[0].role == SourceRole::A);
    UNIT_CHECK(ctx, roled.sources[1].role == SourceRole::B);
    UNIT_CHECK(ctx, roled.sources[2].role == SourceRole::None);
    UNIT_CHECK(ctx, roled.sources[3].role == SourceRole::None);
    UNIT_CHECK(ctx, roleWarnings.size() == 2u);
    UNIT_CHECK(ctx, hasWarning(roleWarnings, "role A on source 'c' cleared: source 'a' already has it"));
    UNIT_CHECK(ctx, hasWarning(roleWarnings, "role B on source 'd' cleared: source 'b' already has it"));
    std::vector<std::string> roleAgain;
    parseCompareProject(serializeCompareProject(roled, folder), folder, &roleAgain);
    UNIT_CHECK(ctx, roleAgain.empty());

    const std::string threeA = R"({"format":"sabercat.compare","formatVersion":1,"sources":[)"
                               R"({"id":"a","kind":"image","path":"x.png","role":"A"},)"
                               R"({"id":"b","kind":"image","path":"y.png","role":"A"},)"
                               R"({"id":"c","kind":"image","path":"z.png","role":"A"}]})";
    std::vector<std::string> threeWarnings;
    const CompareProject threeBack = parseCompareProject(threeA, folder, &threeWarnings);
    UNIT_CHECK(ctx, threeWarnings.size() == 2u);
    UNIT_CHECK(ctx, hasWarning(threeWarnings, "role A on source 'b' cleared: source 'a' already has it"));
    UNIT_CHECK(ctx, hasWarning(threeWarnings, "role A on source 'c' cleared: source 'a' already has it"));
    UNIT_CHECK(ctx, threeBack.sources[0].role == SourceRole::A);
    UNIT_CHECK(ctx, threeBack.sources[1].role == SourceRole::None);
    UNIT_CHECK(ctx, threeBack.sources[2].role == SourceRole::None);

    // Serialize refuses two holders of A, or of B, with std::invalid_argument.
    CompareProject twoHolders = sampleProject(); // left is A, right is B
    twoHolders.sources[1].role = SourceRole::A;
    UNIT_CHECK(ctx, throwsInvalidArgument([&] { serializeCompareProject(twoHolders, folder); }));
    UNIT_CHECK(ctx, contains(errorOf([&] { serializeCompareProject(twoHolders, folder); }), "role A"));
    CompareProject twoB = sampleProject();
    twoB.sources[0].role = SourceRole::B;
    UNIT_CHECK(ctx, throwsInvalidArgument([&] { serializeCompareProject(twoB, folder); }));
    UNIT_CHECK(ctx, !throwsInvalidArgument([&] { serializeCompareProject(sampleProject(), folder); }));

    // setSourceRole moves a role: assigning A or B clears it from the previous holder.
    CompareProject moving;
    moving.sources = {makeSource("a", "x.png"), makeSource("b", "y.png"), makeSource("c", "z.png")};
    UNIT_CHECK(ctx, setSourceRole(moving, 0, SourceRole::A));
    UNIT_CHECK(ctx, setSourceRole(moving, 1, SourceRole::B));
    UNIT_CHECK(ctx, setSourceRole(moving, 2, SourceRole::A));
    UNIT_CHECK(ctx, moving.sources[0].role == SourceRole::None);
    UNIT_CHECK(ctx, moving.sources[1].role == SourceRole::B);
    UNIT_CHECK(ctx, moving.sources[2].role == SourceRole::A);
    UNIT_CHECK(ctx, setSourceRole(moving, 0, SourceRole::B));
    UNIT_CHECK(ctx, moving.sources[0].role == SourceRole::B);
    UNIT_CHECK(ctx, moving.sources[1].role == SourceRole::None);
    UNIT_CHECK(ctx, setSourceRole(moving, 2, SourceRole::None));
    UNIT_CHECK(ctx, moving.sources[2].role == SourceRole::None);
    UNIT_CHECK(ctx, !setSourceRole(moving, 3, SourceRole::A));
    UNIT_CHECK(ctx, moving.sources[0].role == SourceRole::B && moving.sources[1].role == SourceRole::None);
    UNIT_CHECK(ctx, !serializeCompareProject(moving, folder).empty());

    // Default flip overrides are not written, so an absent key means the default.
    const std::string plain = serializeCompareProject(CompareProject{}, folder);
    UNIT_CHECK(ctx, !contains(plain, "startExposure"));
    UNIT_CHECK(ctx, !contains(plain, "numExposures"));

    // The live FLIP rate: the default (0) is not written and reads back as 0; a set rate round-trips
    // byte-exact; only 0 or a rate from 0.1 to 60 is accepted, on save and on load.
    UNIT_CHECK(ctx, CompareProject{}.flip.liveHz == 0.0f);
    UNIT_CHECK(ctx, !contains(plain, "liveHz"));
    CompareProject liveProject = sampleProject();
    liveProject.flip.liveHz = 12.5f;
    const std::string liveText = serializeCompareProject(liveProject, folder);
    UNIT_CHECK(ctx, contains(liveText, "\"liveHz\": 12.5"));
    const CompareProject liveBack = parseCompareProject(liveText, folder, nullptr);
    UNIT_CHECK(ctx, sameBits(liveBack.flip.liveHz, 12.5f));
    UNIT_CHECK(ctx, serializeCompareProject(liveBack, folder) == liveText);

    // NaN, infinity, a negative rate, a rate below 0.1 and one above 60 are refused on save.
    const float refusedRates[] = {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(),
                                  -1.0f, 0.05f, 60.5f};
    for (const float badRate : refusedRates)
    {
        CompareProject badLive = sampleProject();
        badLive.flip.liveHz = badRate;
        UNIT_CHECK(ctx, contains(errorOf([&] { serializeCompareProject(badLive, folder); }), "flip.liveHz"));
    }

    // The same refusals on load, with the key named. A number too large for float overflows to infinity.
    UNIT_CHECK(ctx, contains(errorOf([&] {
                                 parseCompareProject(R"({"format":"sabercat.compare","formatVersion":1,)"
                                                     R"("flip":{"liveHz":100}})",
                                                     folder, nullptr);
                             }),
                             "flip.liveHz"));
    UNIT_CHECK(ctx, contains(errorOf([&] {
                                 parseCompareProject(R"({"format":"sabercat.compare","formatVersion":1,)"
                                                     R"("flip":{"liveHz":-2}})",
                                                     folder, nullptr);
                             }),
                             "flip.liveHz"));
    UNIT_CHECK(ctx, contains(errorOf([&] {
                                 parseCompareProject(R"({"format":"sabercat.compare","formatVersion":1,)"
                                                     R"("flip":{"liveHz":1e300}})",
                                                     folder, nullptr);
                             }),
                             "flip.liveHz"));

    // A rate in range loads with no warning (liveHz is a known key); 0 and both ends of the range save.
    std::vector<std::string> liveWarnings;
    const CompareProject liveRead = parseCompareProject(
        R"({"format":"sabercat.compare","formatVersion":1,"flip":{"liveHz":30}})", folder, &liveWarnings);
    UNIT_CHECK(ctx, liveWarnings.empty());
    UNIT_CHECK(ctx, sameBits(liveRead.flip.liveHz, 30.0f));
    for (const float edge : {0.0f, 0.1f, 60.0f})
    {
        CompareProject edgeLive = sampleProject();
        edgeLive.flip.liveHz = edge;
        UNIT_CHECK(ctx, errorOf([&] { serializeCompareProject(edgeLive, folder); }) == "no error");
    }
    UNIT_CHECK(ctx, !contains(serializeCompareProject(sampleProject(), folder), "liveHz"));

    // A newer formatVersion is refused, naming both versions.
    const std::string newer = R"({"format":"sabercat.compare","formatVersion":2,"sources":[]})";
    const std::string newerError = errorOf([&] { parseCompareProject(newer, folder, nullptr); });
    UNIT_CHECK(ctx, contains(newerError, "version 2"));
    UNIT_CHECK(ctx, contains(newerError, "(1)"));

    // A version below 1, a wrong format name and text that is not JSON are refused.
    UNIT_CHECK(ctx, contains(errorOf([&] {
                                 parseCompareProject(R"({"format":"sabercat.compare","formatVersion":0})", folder,
                                                     nullptr);
                             }),
                             "invalid"));
    UNIT_CHECK(ctx, contains(errorOf([&] {
                                 parseCompareProject(R"({"format":"other","formatVersion":1})", folder, nullptr);
                             }),
                             "not a Sabercat compare project"));
    UNIT_CHECK(ctx, contains(errorOf([&] { parseCompareProject("{", folder, nullptr); }), "not valid JSON"));

    // Unknown keys are ignored, each with a warning that names its key path.
    const std::string extra = R"({"format":"sabercat.compare","formatVersion":1,"bogus":1,)"
                              R"("view":{"zzz":2},"sources":[{"id":"a","kind":"image","path":"x.png","extra":true}]})";
    std::vector<std::string> extraWarnings;
    const CompareProject withExtra = parseCompareProject(extra, folder, &extraWarnings);
    UNIT_CHECK(ctx, extraWarnings.size() == 3u);
    UNIT_CHECK(ctx, hasWarning(extraWarnings, "unknown key 'bogus' ignored"));
    UNIT_CHECK(ctx, hasWarning(extraWarnings, "unknown key 'view.zzz' ignored"));
    UNIT_CHECK(ctx, hasWarning(extraWarnings, "unknown key 'sources[0].extra' ignored"));
    UNIT_CHECK(ctx, withExtra.sources.size() == 1u);

    // A relative path with ".." resolves against the project folder and is stored
    // relative again.
    const std::string up = R"({"format":"sabercat.compare","formatVersion":1,)"
                           R"("sources":[{"id":"a","kind":"image","path":"../assets/a.png"}]})";
    const CompareProject climbed = parseCompareProject(up, "C:/proj/scenes", nullptr); // repo-check: allow
    UNIT_CHECK(ctx, climbed.sources[0].path == "C:/proj/assets/a.png"); // repo-check: allow
    UNIT_CHECK(ctx, contains(serializeCompareProject(climbed, "C:/proj/scenes"), "\"path\": \"../assets/a.png\"")); // repo-check: allow

    // Non-ASCII folder and file names survive the round trip (UTF-8: e-acute, CJK).
    const std::string accentFolder = "C:/proj/\xC3\xA9t\xC3\xA9/\xE4\xB8\xAD\xE6\x96\x87"; // repo-check: allow
    CompareProject accented = sampleProject();
    accented.sources[0].path = accentFolder + "/clip-\xC3\xA9.mp4";
    const std::string accentText = serializeCompareProject(accented, accentFolder);
    UNIT_CHECK(ctx, contains(accentText, "\"path\": \"clip-\xC3\xA9.mp4\""));
    const CompareProject accentBack = parseCompareProject(accentText, accentFolder, nullptr);
    UNIT_CHECK(ctx, accentBack.sources[0].path == accentFolder + "/clip-\xC3\xA9.mp4");
    UNIT_CHECK(ctx, serializeCompareProject(accentBack, accentFolder) == accentText);

    // A path on another drive cannot be relative: it is stored absolute, and a load
    // flags it with a warning.
    CompareProject otherDrive = sampleProject();
    otherDrive.sources[0].path = "D:/elsewhere/a.png"; // repo-check: allow
    const std::string otherText = serializeCompareProject(otherDrive, folder);
    UNIT_CHECK(ctx, contains(otherText, "\"path\": \"D:/elsewhere/a.png\"")); // repo-check: allow
    std::vector<std::string> otherWarnings;
    const CompareProject otherBack = parseCompareProject(otherText, folder, &otherWarnings);
    UNIT_CHECK(ctx, hasWarning(otherWarnings, "source 'left' is on another drive; the path is absolute"));
    UNIT_CHECK(ctx, otherBack.sources[0].pathIsAbsoluteOtherDrive);
    UNIT_CHECK(ctx, !otherBack.sources[1].pathIsAbsoluteOtherDrive);

    // An empty sources list is valid, and a minimal file gets every default.
    CompareProject empty;
    UNIT_CHECK(ctx, empty.sources.empty());
    const std::string minimal = R"({"format":"sabercat.compare","formatVersion":1})";
    const CompareProject defaults = parseCompareProject(minimal, folder, nullptr);
    UNIT_CHECK(ctx, defaults.sources.empty());
    UNIT_CHECK(ctx, defaults.view.paneCount == 2);
    UNIT_CHECK(ctx, defaults.camera.distance == 3.0f);
    UNIT_CHECK(ctx, !defaults.flip.startExposure.has_value());

    // Ids: unique, 1 to 64 characters of [A-Za-z0-9_-].
    CompareProject duplicate;
    duplicate.sources = {makeSource("a", "x.png"), makeSource("a", "y.png")};
    UNIT_CHECK(ctx, contains(errorOf([&] { serializeCompareProject(duplicate, folder); }), "duplicate source id 'a'"));

    CompareProject spaced;
    spaced.sources = {makeSource("bad id", "x.png")};
    UNIT_CHECK(ctx, contains(errorOf([&] { serializeCompareProject(spaced, folder); }), "1 to 64"));

    CompareProject blank;
    blank.sources = {makeSource("", "x.png")};
    UNIT_CHECK(ctx, contains(errorOf([&] { serializeCompareProject(blank, folder); }), "1 to 64"));

    CompareProject longId;
    longId.sources = {makeSource(std::string(65, 'a'), "x.png")};
    UNIT_CHECK(ctx, contains(errorOf([&] { serializeCompareProject(longId, folder); }), "1 to 64"));

    const std::string dupJson = R"({"format":"sabercat.compare","formatVersion":1,"sources":)"
                                R"([{"id":"a","kind":"image","path":"x.png"},{"id":"a","kind":"image","path":"y.png"}]})";
    UNIT_CHECK(ctx, contains(errorOf([&] { parseCompareProject(dupJson, folder, nullptr); }), "duplicate source id"));

    // At most 8 sources: 8 are valid, 9 are refused on save and on load.
    CompareProject eight;
    for (size_t i = 0; i < 8; ++i)
    {
        eight.sources.push_back(makeSource("s" + std::to_string(i), "a.png"));
    }
    const std::string eightText = serializeCompareProject(eight, folder);
    UNIT_CHECK(ctx, parseCompareProject(eightText, folder, nullptr).sources.size() == 8u);

    CompareProject nine = eight;
    nine.sources.push_back(makeSource("s8", "a.png"));
    UNIT_CHECK(ctx, contains(errorOf([&] { serializeCompareProject(nine, folder); }), "at most 8"));

    const std::string nineJson = R"({"format":"sabercat.compare","formatVersion":1,"sources":)" + sourceListJson(9)
                                 + "}";
    UNIT_CHECK(ctx, contains(errorOf([&] { parseCompareProject(nineJson, folder, nullptr); }), "at most 8"));

    // Wrong types and bad values name the key.
    UNIT_CHECK(ctx, contains(errorOf([&] {
                                 parseCompareProject(R"({"format":"sabercat.compare","formatVersion":"1"})", folder,
                                                     nullptr);
                             }),
                             "formatVersion must be an integer"));
    UNIT_CHECK(ctx, contains(errorOf([&] {
                                 parseCompareProject(R"({"format":"sabercat.compare","formatVersion":1,)"
                                                     R"("sources":[{"id":"a","kind":"image","path":"x.png","trimIn":"x"}]})",
                                                     folder, nullptr);
                             }),
                             "sources[0].trimIn must be a number"));
    UNIT_CHECK(ctx, contains(errorOf([&] {
                                 parseCompareProject(R"({"format":"sabercat.compare","formatVersion":1,)"
                                                     R"("sources":[{"id":"a","kind":"image","path":"x.png","role":"C"}]})",
                                                     folder, nullptr);
                             }),
                             "sources[0].role must be one of"));
    UNIT_CHECK(ctx, contains(errorOf([&] {
                                 parseCompareProject(R"({"format":"sabercat.compare","formatVersion":1,)"
                                                     R"("view":{"paneCount":5}})",
                                                     folder, nullptr);
                             }),
                             "view.paneCount must be from 1 to 4"));

    CompareProject badTrim = sampleProject();
    badTrim.sources[0].trimIn = 2.0;
    badTrim.sources[0].trimOut = 1.0;
    UNIT_CHECK(ctx, contains(errorOf([&] { serializeCompareProject(badTrim, folder); }), "trimOut must be greater"));

    CompareProject negativeTime = sampleProject();
    negativeTime.timeline.time = -1.0;
    UNIT_CHECK(ctx, contains(errorOf([&] { serializeCompareProject(negativeTime, folder); }), "timeline.time"));

    // Files: atomic save, the .tmp file never left behind, and a load that matches.
    const fs::path root = fs::temp_directory_path() / "sabercat_compare_project_test";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);
    writeFile(root / "present.png", "not a real png");

    CompareProject onDisk;
    onDisk.sources = {makeSource("here", utf8(root / "present.png")), makeSource("gone", utf8(root / "gone.png"))};
    const fs::path target = root / "work.sabercat-compare";
    UNIT_CHECK(ctx, saveCompareProject(utf8(target), onDisk) == utf8(target));
    UNIT_CHECK(ctx, !fs::exists(root / "work.sabercat-compare.tmp"));
    UNIT_CHECK(ctx, contains(readFile(target), "\"path\": \"present.png\""));

    std::vector<std::string> loadWarnings;
    const CompareProject loaded = loadCompareProject(utf8(target), &loadWarnings);
    UNIT_CHECK(ctx, loaded.sources.size() == 2u);
    UNIT_CHECK(ctx, !loaded.sources[0].missing);
    UNIT_CHECK(ctx, loaded.sources[1].missing);
    UNIT_CHECK(ctx, loadWarnings.empty());
    UNIT_CHECK(ctx, loaded.sources[0].path == utf8(root / "present.png"));

    // A save that fails leaves the old file exactly as it was. Here the temp file's
    // name is taken by a folder, so the write cannot start.
    const std::string before = readFile(target);
    const fs::path blocker = root / "work.sabercat-compare.tmp";
    fs::create_directories(blocker, ec);
    CompareProject changed = onDisk;
    changed.timeline.time = 9.0;
    UNIT_CHECK(ctx, errorOf([&] { saveCompareProject(utf8(target), changed); }) != "no error");
    UNIT_CHECK(ctx, readFile(target) == before);
    UNIT_CHECK(ctx, fs::is_directory(blocker));
    fs::remove(blocker, ec);

    // A destination that is a non-empty folder: the rename fails, the folder stays
    // whole and the .tmp file is removed.
    const fs::path asDir = root / "as-dir.sabercat-compare";
    fs::create_directories(asDir, ec);
    writeFile(asDir / "keep.txt", "keep");
    UNIT_CHECK(ctx, errorOf([&] { saveCompareProject(utf8(asDir), onDisk); }) != "no error");
    UNIT_CHECK(ctx, fs::is_directory(asDir));
    UNIT_CHECK(ctx, fs::exists(asDir / "keep.txt"));
    UNIT_CHECK(ctx, !fs::exists(root / "as-dir.sabercat-compare.tmp"));

    // A folder that does not exist: the write fails and nothing is created.
    const fs::path nowhere = root / "no-such-folder" / "x.sabercat-compare";
    UNIT_CHECK(ctx, errorOf([&] { saveCompareProject(utf8(nowhere), onDisk); }) != "no error");
    UNIT_CHECK(ctx, !fs::exists(root / "no-such-folder"));

    // An invalid project is refused before any file is created.
    const fs::path refused = root / "refused.sabercat-compare";
    UNIT_CHECK(ctx, errorOf([&] { saveCompareProject(utf8(refused), nine); }) != "no error");
    UNIT_CHECK(ctx, !fs::exists(refused));
    UNIT_CHECK(ctx, !fs::exists(root / "refused.sabercat-compare.tmp"));

    // A non-ASCII file name saves and loads.
    const fs::path accentFile = fs::u8path(std::string("caf\xC3\xA9.sabercat-compare"));
    UNIT_CHECK(ctx, saveCompareProject(utf8(root / accentFile), onDisk) == utf8(root / accentFile));
    UNIT_CHECK(ctx, loadCompareProject(utf8(root / accentFile), nullptr).sources.size() == 2u);

    // Awkward floats on disk: save, load and save again gives byte-identical files.
    CompareProject floats = awkward;
    floats.sources[0].path = utf8(root / "present.png");
    floats.sources[1].path = utf8(root / "gone.png");
    const fs::path floatsFirst = root / "floats-first.sabercat-compare";
    const fs::path floatsSecond = root / "floats-second.sabercat-compare";
    saveCompareProject(utf8(floatsFirst), floats);
    const CompareProject floatsReloaded = loadCompareProject(utf8(floatsFirst), nullptr);
    saveCompareProject(utf8(floatsSecond), floatsReloaded);
    UNIT_CHECK(ctx, readFile(floatsFirst) == readFile(floatsSecond));
    UNIT_CHECK(ctx, contains(readFile(floatsFirst), "\"r\": 0.1,"));
    UNIT_CHECK(ctx, !contains(readFile(floatsFirst), "0.10000000149"));
    UNIT_CHECK(ctx, sameBits(floatsReloaded.camera.fx, FLT_MAX));
    UNIT_CHECK(ctx, sameBits(floatsReloaded.view.diffGain, -0.0f));

    fs::remove_all(root, ec);
}

} // namespace sabercat::unit
