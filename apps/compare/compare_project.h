// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The Sabercat Compare project file (*.sabercat-compare): the sources, camera, view,
// flip and timeline state of one comparison, stored as UTF-8 JSON. The string forms
// need no file, so the tests and any caller can round-trip a project in memory.
// Plain structs only: no glm, no window, no GPU.
//
// Camera mapping, from apps/common/orbit_camera.h and its toCamera():
//   CompareCamera::pivot        = OrbitCamera::pivot
//   CompareCamera::orientation  = OrbitCamera::orientation, as (w, x, y, z)
//   CompareCamera::distance     = OrbitCamera::distance
//   fx, fy, cx, cy, width, height = the Camera intrinsics that toCamera() sets
//
// Paths: a source path is stored relative to the project file's folder, with forward
// slashes. On load it is resolved to an absolute path. A path on another drive
// cannot be relative; it is stored absolute and flagged.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace sabercat::compare::project
{

// Most sources a project may hold; more is refused on save and on load.
inline constexpr size_t kMaxCompareSources = 8;

// The live FLIP rate range in Hz (flip.liveHz). 0 in the file means the default rate, which
// is not written. The gui re-exports these, so the range is kept in one place.
inline constexpr float kMinLiveFlipHz = 0.1f;
inline constexpr float kMaxLiveFlipHz = 60.0f;

enum class SourceKind : uint8_t
{
    Image,
    Scene,
    Video
};

enum class SourceRole : uint8_t
{
    A,
    B,
    None
};

enum class ToneMapper : uint8_t
{
    Aces,
    Hable,
    Reinhard
};

enum class ViewMode : uint8_t
{
    Split,
    Single
};

// The single-pane view: one source, a wipe or flicker of the pair, or one of the map
// modes. The JSON keys are the lowercase names in the comment and in compare_project.cpp.
enum class SingleViewMode : uint8_t
{
    A,         // "a"
    B,         // "b"
    Wipe,      // "wipe"
    Flicker,   // "flicker"
    Diff,      // "diff"
    Depth,     // "depth"
    Alpha,     // "alpha"
    DepthDiff, // "depthDiff"
    AlphaDiff, // "alphaDiff"
    Flip       // "flip" (the GPU LDR-FLIP map)
};

enum class GridLayout : uint8_t
{
    Row,
    Grid
};

enum class WipeOrientation : uint8_t
{
    Horizontal,
    Vertical
};

enum class ZoomMode : uint8_t
{
    Fit,
    Percent
};

enum class Anchor : uint8_t
{
    TopLeft,
    Centered,
    Custom
};

enum class FlipMode : uint8_t
{
    Ldr,
    Hdr
};

struct Vec2i
{
    int32_t x = 0;
    int32_t y = 0;
};

struct Vec2
{
    float x = 0.0f;
    float y = 0.0f;
};

struct Vec3
{
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct Quat
{
    float w = 1.0f;
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct Rgb
{
    float r = 0.0f;
    float g = 0.0f;
    float b = 0.0f;
};

struct FloatRange
{
    float min = 0.0f;
    float max = 1.0f;
};

// One comparison source. The first group of fields is what the file holds. The last
// two are set on load only and are never written.
struct CompareSource
{
    std::string id;        // unique in the project; [A-Za-z0-9_-], 1 to 64 characters
    std::string path;      // absolute once loaded; stored relative on save
    std::string debugView; // name of the debug view; empty = none

    // Trim and offset, in seconds. trimOut < 0 means the end of the clip. Otherwise
    // trimOut must be greater than trimIn.
    double trimIn = 0.0;
    double trimOut = -1.0;
    double timeOffset = 0.0;

    Rgb background;
    float exposure = 0.0f; // stops
    int32_t shCap = -1;    // spherical harmonics degree cap; -1 = none
    Vec2i offset;

    SourceKind kind = SourceKind::Image;
    SourceRole role = SourceRole::None;
    ToneMapper toneMapper = ToneMapper::Aces;
    bool flipUpAxis = false;
    bool antialias = false;
    bool loop = false;
    bool visible = true;

    // Transient: set by load, never written.
    bool missing = false;                  // the resolved path does not exist
    bool pathIsAbsoluteOtherDrive = false; // the stored path was absolute (another drive)
};

struct CompareCamera
{
    Vec3 pivot;
    Quat orientation;
    float distance = 3.0f;
    float fx = 0.0f;
    float fy = 0.0f;
    float cx = 0.0f;
    float cy = 0.0f;
    int32_t width = 0;
    int32_t height = 0;
};

struct CompareWipe
{
    float position = 0.5f; // image-space coordinate of the wipe line; not clamped
    WipeOrientation orientation = WipeOrientation::Vertical;
};

struct CompareZoom
{
    float percent = 100.0f;
    ZoomMode mode = ZoomMode::Fit;
};

struct CompareRanges
{
    FloatRange diff;
    FloatRange flip;
    FloatRange exposure{-5.0f, 5.0f};
    FloatRange depth;
};

struct CompareView
{
    CompareRanges ranges;
    CompareWipe wipe;
    CompareZoom zoom;
    Vec2 pan;
    float diffGain = 1.0f;
    float flickerPeriodSeconds = 1.0f; // must be greater than 0
    // The split-screen divider as a fraction of the window width (JSON "splitDivider"). Files
    // without the key load with 0.5; the loader clamps it to [0,1].
    float splitDivider = 0.5f;
    int32_t paneCount = 2;             // 1 to 4
    ViewMode mode = ViewMode::Split;
    GridLayout layout = GridLayout::Row;
    Anchor anchor = Anchor::TopLeft;
    // The single-pane view (used when mode is Single). The one-byte fields stay last, so
    // the struct has no interior padding.
    SingleViewMode singleView = SingleViewMode::A;
    // Depth and Alpha show source B when true (JSON "mapSource": "b"), else source A.
    bool mapUseB = false;
    // Depth and depth difference take their range from the sources when true, else from
    // ranges.depth.
    bool depthAuto = true;
};

// Flip (HDR/LDR error metric) settings. The optional fields are written only when set.
struct CompareFlip
{
    std::optional<float> startExposure;
    std::optional<float> stopExposure;
    std::optional<int32_t> numExposures;
    int32_t ppd = 0; // pixels per degree; 0 = default
    // The FLIP rate while the camera moves, in Hz: 0 = the default (4 Hz), which is not
    // written. Otherwise from kMinLiveFlipHz to kMaxLiveFlipHz.
    float liveHz = 0.0f;
    FlipMode mode = FlipMode::Ldr;
    ToneMapper toneMapper = ToneMapper::Aces;
};

struct CompareTimeline
{
    double time = 0.0;  // global clock, seconds, >= 0
    double speed = 1.0; // >= 0; 0 = paused
    bool loop = false;
};

struct CompareProject
{
    std::vector<CompareSource> sources; // at most kMaxCompareSources
    CompareCamera camera;
    CompareView view;
    CompareFlip flip;
    CompareTimeline timeline;
};

// Moves a role to source index: assigning A or B clears that role from the source that
// held it. SourceRole::None only clears the source's own role. Returns false, and
// changes nothing, when index is out of range.
bool setSourceRole(CompareProject & project, size_t index, SourceRole role);

// The JSON text for a project, as it would be saved in projectFolder. Throws
// std::runtime_error if the project is invalid (bad ids, too many sources, values
// that are not finite, and so on) or cannot be written as JSON. Throws
// std::invalid_argument if two sources hold the same role A or role B.
std::string serializeCompareProject(const CompareProject & project, const std::string & projectFolder);

// Parses JSON text as a project in projectFolder. Unknown keys are ignored and each
// one adds a warning naming its key path. If two sources hold role A (or two hold B),
// the first keeps it and the others are set to none, with one warning per cleared
// role. Warnings are appended to *warnings when it is not null. Throws
// std::runtime_error naming the key on bad input, and on a formatVersion newer than
// this reader's.
CompareProject parseCompareProject(const std::string & json, const std::string & projectFolder,
                                   std::vector<std::string> * warnings);

// Writes the project to utf8Path atomically: the text goes to utf8Path + ".tmp", which
// is then renamed over utf8Path. On any failure the .tmp file is removed and the old
// file is left as it was. Returns utf8Path. Throws std::runtime_error.
std::string saveCompareProject(const std::string & utf8Path, const CompareProject & project);

// Reads the project at utf8Path. A missing source file is marked, not an error. Paths
// are resolved against the file's folder. Throws std::runtime_error if the file cannot
// be read or is not a valid project.
CompareProject loadCompareProject(const std::string & utf8Path, std::vector<std::string> * warnings);

} // namespace sabercat::compare::project
