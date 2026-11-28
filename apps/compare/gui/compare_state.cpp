// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "compare_state.h"

#include "app_util.h"
#include "gpu_context.h"
#include "image_source.h"
#include "sabercat/logger.h"
#include "timeline.h"
#include "video_source.h"

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace sabercat::compare::gui
{

namespace
{

namespace fs = std::filesystem;

// Limits of the media panel and the project-facing numbers.
constexpr double kMinFlickerPeriod = 0.1;
constexpr double kMaxFlickerPeriod = 60.0;
constexpr float kMaxDiffGain = 100.0f;
constexpr float kMaxExposureStops = 32.0f;
// The render size of the first scene when no image is loaded.
constexpr uint32_t kDefaultRenderWidth = 1280;
constexpr uint32_t kDefaultRenderHeight = 720;
// The highest spherical-harmonics degree; -1 means this (full).
constexpr int32_t kFullShDegree = 3;

fs::path pathFromUtf8(const std::string & utf8)
{
    return fs::path(std::u8string(reinterpret_cast<const char8_t *>(utf8.data()), utf8.size()));
}

std::string utf8Of(const fs::path & path)
{
    const std::u8string text = path.u8string();
    return std::string(reinterpret_cast<const char *>(text.data()), text.size());
}

// The absolute, normal form of a UTF-8 path. The input is kept when it cannot be resolved.
std::string absoluteUtf8(const std::string & utf8)
{
    std::error_code ec;
    const fs::path absolute = fs::absolute(pathFromUtf8(utf8), ec);
    if (ec)
    {
        return utf8;
    }
    return utf8Of(absolute.lexically_normal());
}

// The last component of a UTF-8 path (the whole text when there is none).
std::string fileNameUtf8(const std::string & utf8)
{
    const std::string name = utf8Of(pathFromUtf8(utf8).filename());
    return name.empty() ? utf8 : name;
}

float finiteOr(float value, float fallback)
{
    return std::isfinite(value) ? value : fallback;
}

float clampFloat(float value, float low, float high)
{
    if (std::isnan(value))
    {
        return low;
    }
    return std::min(std::max(value, low), high);
}

// The compare window's tone mapper, from the project's (same order).
compare::ToneMapper toFlipToneMapper(project::ToneMapper toneMapper)
{
    switch (toneMapper)
    {
    case project::ToneMapper::Aces:
        return compare::ToneMapper::Aces;
    case project::ToneMapper::Hable:
        return compare::ToneMapper::Hable;
    case project::ToneMapper::Reinhard:
        return compare::ToneMapper::Reinhard;
    }
    return compare::ToneMapper::Aces;
}

// The project's tone mapper for the window's one (the inverse of toFlipToneMapper()).
project::ToneMapper toProjectToneMapper(compare::ToneMapper toneMapper)
{
    switch (toneMapper)
    {
    case compare::ToneMapper::Aces:
        return project::ToneMapper::Aces;
    case compare::ToneMapper::Hable:
        return project::ToneMapper::Hable;
    case compare::ToneMapper::Reinhard:
        return project::ToneMapper::Reinhard;
    }
    return project::ToneMapper::Aces;
}

// HDR-FLIP exposure limits, in stops (the GPU scale 2^e is a float).
constexpr float kMaxHdrStops = 126.0f;
// The budget that runs a job to its end in one call (runHdrBlocking).
constexpr double kUnlimitedHdrBudget = 1.0e30;
constexpr const char * kHdrNeedsLinear = "HDR-FLIP needs a Linear (EXR) source in the A/B pair";
// The HDR-FLIP state while a video in the pair plays: a frame change never runs it.
constexpr const char * kHdrNotDuringPlayback = "HDR-FLIP not run during playback";
// The playback speed range (the presets and any value between them) and the largest clock step of one
// presented frame, so a stall or a minimize never jumps the clock.
constexpr double kMinPlaybackSpeed = 0.25;
constexpr double kMaxPlaybackSpeed = 4.0;
constexpr double kMaxFrameStep = 0.25;

bool validHdrExposure(float stops)
{
    return std::isfinite(stops) && stops >= -kMaxHdrStops && stops <= kMaxHdrStops;
}

bool validHdrCount(int32_t count)
{
    return count >= kMinHdrExposures && count <= kMaxHdrExposures;
}

size_t countVisible(const std::vector<MediaEntry> & entries)
{
    size_t count = 0;
    for (const MediaEntry & entry : entries)
    {
        count += entry.visible ? 1 : 0;
    }
    return count;
}

SingleView toSingleView(project::SingleViewMode mode)
{
    switch (mode)
    {
    case project::SingleViewMode::A:
        return SingleView::A;
    case project::SingleViewMode::B:
        return SingleView::B;
    case project::SingleViewMode::Wipe:
        return SingleView::Wipe;
    case project::SingleViewMode::Flicker:
        return SingleView::Flicker;
    case project::SingleViewMode::Diff:
        return SingleView::Diff;
    case project::SingleViewMode::Depth:
        return SingleView::Depth;
    case project::SingleViewMode::Alpha:
        return SingleView::Alpha;
    case project::SingleViewMode::DepthDiff:
        return SingleView::DepthDiff;
    case project::SingleViewMode::AlphaDiff:
        return SingleView::AlphaDiff;
    case project::SingleViewMode::Flip:
        return SingleView::Flip;
    }
    return SingleView::A;
}

project::SingleViewMode toProjectSingleView(SingleView view)
{
    switch (view)
    {
    case SingleView::A:
        return project::SingleViewMode::A;
    case SingleView::B:
        return project::SingleViewMode::B;
    case SingleView::Wipe:
        return project::SingleViewMode::Wipe;
    case SingleView::Flicker:
        return project::SingleViewMode::Flicker;
    case SingleView::Diff:
        return project::SingleViewMode::Diff;
    case SingleView::Depth:
        return project::SingleViewMode::Depth;
    case SingleView::Alpha:
        return project::SingleViewMode::Alpha;
    case SingleView::DepthDiff:
        return project::SingleViewMode::DepthDiff;
    case SingleView::AlphaDiff:
        return project::SingleViewMode::AlphaDiff;
    case SingleView::Flip:
        return project::SingleViewMode::Flip;
    }
    return project::SingleViewMode::A;
}

// The view's name for the log: the key the project file uses for it (compare_project.cpp).
const char * singleViewName(SingleView view)
{
    switch (view)
    {
    case SingleView::A:
        return "a";
    case SingleView::B:
        return "b";
    case SingleView::Wipe:
        return "wipe";
    case SingleView::Flicker:
        return "flicker";
    case SingleView::Diff:
        return "diff";
    case SingleView::Depth:
        return "depth";
    case SingleView::Alpha:
        return "alpha";
    case SingleView::DepthDiff:
        return "depthDiff";
    case SingleView::AlphaDiff:
        return "alphaDiff";
    case SingleView::Flip:
        return "flip";
    }
    return "a";
}

// A number with a fixed count of decimals, for the log lines.
std::string fixedText(double value, int decimals)
{
    char text[64];
    std::snprintf(text, sizeof(text), "%.*f", decimals, value);
    return text;
}

// The scene sources' view of a media object, or null for any other kind.
SceneSource * sceneOf(MediaSource * media)
{
    return dynamic_cast<SceneSource *>(media);
}

// The video sources' view of a media object, or null for any other kind.
VideoSource * videoOfMedia(MediaSource * media)
{
    return dynamic_cast<VideoSource *>(media);
}

// The write time and size of a file, read without throwing. False when either cannot be read,
// which the poll treats as no change.
bool readFileStamp(const std::string & utf8Path, int64_t & time, uint64_t & size)
{
    const fs::path path = pathFromUtf8(utf8Path);
    std::error_code timeError;
    std::error_code sizeError;
    const fs::file_time_type stamp = fs::last_write_time(path, timeError);
    const std::uintmax_t bytes = fs::file_size(path, sizeError);
    if (timeError || sizeError)
    {
        return false;
    }
    time = static_cast<int64_t>(stamp.time_since_epoch().count());
    size = static_cast<uint64_t>(bytes);
    return true;
}

// Records the stamp of a Ready scene's file, so a change after its load is seen by the next poll.
// Any other entry has no stamp.
void stampFile(MediaEntry & entry)
{
    entry.fileStamped = false;
    if (entry.path.empty() || !entry.ready() || sceneOf(entry.media.get()) == nullptr)
    {
        return;
    }
    entry.fileStamped = readFileStamp(entry.path, entry.fileTime, entry.fileSize);
}

// The trim, loop and time offset of one video: what the project keeps for it. The trim out is
// -1 for the end of the clip, as in the project file.
struct VideoSettings
{
    double trimIn = 0.0;
    double trimOut = -1.0;
    double timeOffset = 0.0;
    bool loop = false;
};

VideoSettings settingsOf(const VideoSource & video)
{
    VideoSettings settings;
    settings.trimIn = video.trimIn();
    settings.trimOut = video.trimToEnd() ? -1.0 : video.trimOut();
    settings.timeOffset = video.timeOffset();
    settings.loop = video.loop();
    return settings;
}

VideoSettings settingsOfSource(const project::CompareSource & source)
{
    VideoSettings settings;
    settings.trimIn = source.trimIn;
    settings.trimOut = source.trimOut;
    settings.timeOffset = source.timeOffset;
    settings.loop = source.loop;
    return settings;
}

// Applies saved settings to a Ready video. A value the video refuses falls back to its default
// and adds a notice (name first) to notices.
void applyVideoSettings(VideoSource & video, const VideoSettings & settings, const std::string & name,
                        std::vector<std::string> & notices)
{
    if (!video.setTrim(settings.trimIn, settings.trimOut))
    {
        video.setTrim(0.0, -1.0);
        notices.push_back(name + ": the saved trim does not fit this video; the whole clip is used");
    }
    video.setLoop(settings.loop);
    if (!video.setTimeOffset(settings.timeOffset))
    {
        video.setTimeOffset(0.0);
        notices.push_back(name + ": the saved time offset is out of range; using 0 s");
    }
}

// The text of a refused video setting: the video's own message, or the fallback when it has none.
std::string videoRefusal(const VideoSource & video, const char * fallback)
{
    const std::string & text = video.statusMessage();
    return text.empty() ? std::string(fallback) : text;
}

// A scene entry that cannot render (no scene host, or a placeholder while a project loads).
// It stays in the list with its message, as a failed image does.
class UnavailableScene final : public MediaSource
{
  public:
    explicit UnavailableScene(std::string message) : m_message(std::move(message)) {}

    project::SourceKind kind() const override
    {
        return project::SourceKind::Scene;
    }
    SourceStatus status() const override
    {
        return SourceStatus::Error;
    }
    const std::string & statusMessage() const override
    {
        return m_message;
    }
    uint32_t width() const override
    {
        return 1;
    }
    uint32_t height() const override
    {
        return 1;
    }
    Encoding encoding() const override
    {
        return Encoding::Display;
    }
    bool premultiplied() const override
    {
        return false;
    }
    bool hasAlpha() const override
    {
        return true;
    }
    BufferView color() const override
    {
        return {};
    }
    bool readPixel(int32_t /*x*/, int32_t /*y*/, float /*rgba*/[4]) const override
    {
        return false;
    }

  private:
    std::string m_message;
};

} // namespace

LayoutInput compositeLayout(const CompositeInput & input)
{
    LayoutInput shown = input.layout;
    switch (input.mode)
    {
    case CompositeMode::Panes:
        break;
    case CompositeMode::Wipe:
    case CompositeMode::Diff:
    case CompositeMode::DepthDiff:
    case CompositeMode::AlphaDiff:
    case CompositeMode::Flip:
        shown.view.mode = ViewMode::SingleScreen;
        shown.view.abIndex = shown.view.sourceA;
        break;
    case CompositeMode::Flicker:
        shown.view.mode = ViewMode::SingleScreen;
        shown.view.abIndex = input.flickerShowB ? shown.view.sourceB : shown.view.sourceA;
        break;
    case CompositeMode::Depth:
    case CompositeMode::Alpha:
        // The chosen source: B when mapUseB, else A (the compositor's rule).
        shown.view.mode = ViewMode::SingleScreen;
        shown.view.abIndex = input.mapUseB ? shown.view.sourceB : shown.view.sourceA;
        break;
    }
    return shown;
}

CompareState::CompareState(GpuContext & context) : m_context(context)
{
    syncLayout();
    refreshView();
}

CompareState::~CompareState() = default;

void CompareState::setSceneHost(const sabercat::Renderer * host)
{
    m_sceneHost = host;
}

// ---- Media list -----------------------------------------------------------------------

int32_t CompareState::addMedia(const std::string & utf8Path)
{
    if (m_entries.size() >= static_cast<size_t>(kMaxSources))
    {
        setStatus("At most " + std::to_string(kMaxSources) + " sources: remove one before adding another");
        Logger::warning("Compare: " + fileNameUtf8(utf8Path) + " refused: at most " + std::to_string(kMaxSources)
                        + " sources");
        return -1;
    }
    const std::string path = absoluteUtf8(utf8Path);
    if (isSceneFile(path))
    {
        return addScene(path);
    }
    if (isVideoFile(path))
    {
        auto video = std::make_unique<VideoSource>(m_context, path);
        return addEntry(std::move(video), fileNameUtf8(path), path);
    }
    auto media = std::make_unique<ImageSource>(m_context, path);
    return addEntry(std::move(media), fileNameUtf8(path), path);
}

int32_t CompareState::addScene(const std::string & utf8Path)
{
    if (m_sceneHost == nullptr)
    {
        setStatus("Scene rendering is not available");
        return -1;
    }
    if (!sceneBudgetAllows())
    {
        return -1;
    }
    ensureRenderSize();
    SceneSettings settings;
    settings.width = m_renderWidth;
    settings.height = m_renderHeight;

    // A new file keeps the antialias flag its loader reads; the user can change it after.
    std::unique_ptr<MediaSource> media = makeScene(utf8Path, settings, false);
    // The loaded scene's device-local cost sizes the budget check for the next scene. Each
    // scene renderer has its own allocator, so the scene's own bytes are its whole cost.
    const SceneSource * loaded = sceneOf(media.get());
    if (loaded != nullptr && loaded->status() == SourceStatus::Ready)
    {
        m_lastSceneBytes = loaded->deviceBytes();
    }

    const int32_t index = addEntry(std::move(media), fileNameUtf8(utf8Path), utf8Path);
    if (index < 0)
    {
        return -1;
    }
    MediaEntry & entry = m_entries[static_cast<size_t>(index)];
    if (const SceneSource * scene = sceneOf(entry.media.get()))
    {
        entry.antialias = scene->settings().antialias;
        frameFirstScene(*scene);
    }
    return index;
}

std::unique_ptr<MediaSource> CompareState::makeScene(const std::string & utf8Path, const SceneSettings & settings,
                                                     bool applyAntialias)
{
    if (m_sceneHost == nullptr)
    {
        return std::make_unique<UnavailableScene>("Scene rendering is not available");
    }
    return std::make_unique<SceneSource>(m_context, *m_sceneHost, utf8Path, settings, applyAntialias);
}

// The Ready scenes and the device-local bytes they hold. The device's own allocator does not
// count these, so the budget checks subtract them from the free amount.
uint64_t CompareState::readySceneBytes(size_t & readyScenes) const
{
    readyScenes = 0;
    uint64_t bytes = 0;
    for (const MediaEntry & entry : m_entries)
    {
        if (entry.ready() && entry.media->kind() == project::SourceKind::Scene)
        {
            ++readyScenes;
            if (const SceneSource * scene = sceneOf(entry.media.get()))
            {
                bytes += scene->deviceBytes();
            }
        }
    }
    return bytes;
}

// Another scene is refused when the device has too little free memory for it. The estimate
// is the device-local cost of the last scene loaded, plus a quarter. The bytes the loaded
// scenes hold are not in the device's own allocator, so they are subtracted from the free
// amount here.
bool CompareState::sceneBudgetAllows()
{
    size_t readyScenes = 0;
    const uint64_t bytesInUse = readySceneBytes(readyScenes);
    if (readyScenes == 0 || m_lastSceneBytes == 0)
    {
        return true;
    }
    std::string message;
    if (sceneBudgetOk(m_context.device(), m_lastSceneBytes * 5 / 4, message, bytesInUse))
    {
        return true;
    }
    setStatus(message);
    return false;
}

// The render size of the scenes: the first image's size, else the default. It is kept
// after it is set, until a new project.
void CompareState::ensureRenderSize()
{
    if (m_renderWidth > 0 && m_renderHeight > 0)
    {
        return;
    }
    for (const MediaEntry & entry : m_entries)
    {
        if (entry.ready() && entry.media->kind() == project::SourceKind::Image)
        {
            m_renderWidth = entry.media->width();
            m_renderHeight = entry.media->height();
            return;
        }
    }
    m_renderWidth = kDefaultRenderWidth;
    m_renderHeight = kDefaultRenderHeight;
}

void CompareState::frameCamera(const SceneSource & scene)
{
    const sabercat::SceneBounds bounds = scene.bounds();
    if (!bounds.valid || m_renderHeight == 0)
    {
        return;
    }
    m_camera.frame(bounds, m_renderHeight, static_cast<float>(m_renderHeight));
    m_cameraFramed = true;
    ++m_cameraGeneration;
    m_dirty = true;
}

// The first scene that loads frames the camera, unless a project restored one.
void CompareState::frameFirstScene(const SceneSource & scene)
{
    if (!m_cameraFramed && scene.status() == SourceStatus::Ready)
    {
        frameCamera(scene);
    }
}

// A project camera with a render size restores the pose and the size. Without one the
// camera keeps its defaults and the first scene frames it.
void CompareState::restoreCamera(const project::CompareCamera & camera)
{
    const int64_t limit = kMaxSourceSize;
    const bool restore = camera.width > 0 && camera.height > 0 && camera.width <= limit && camera.height <= limit;
    m_camera = app::OrbitCamera{};
    m_cameraFramed = false;
    m_renderWidth = 0;
    m_renderHeight = 0;
    // A size that waited for the old camera belongs to the old project.
    m_pendingW = 0;
    m_pendingH = 0;
    if (restore)
    {
        m_renderWidth = static_cast<uint32_t>(camera.width);
        m_renderHeight = static_cast<uint32_t>(camera.height);
        m_camera.pivot = glm::vec3(camera.pivot.x, camera.pivot.y, camera.pivot.z);
        const glm::quat orientation(camera.orientation.w, camera.orientation.x, camera.orientation.y,
                                    camera.orientation.z);
        const float length = glm::length(orientation);
        if (std::isfinite(length) && length > 0.0f)
        {
            m_camera.orientation = glm::normalize(orientation);
        }
        if (std::isfinite(camera.distance))
        {
            m_camera.distance = std::max(app::OrbitCamera::kMinDistance, camera.distance);
        }
        m_cameraFramed = true;
    }
    ++m_cameraGeneration;
}

void CompareState::hideScene(MediaEntry & entry, const std::string & message)
{
    entry.visible = false;
    entry.role = project::SourceRole::None;
    entry.message = message;
    setStatus(entry.name + ": " + message);
}

// After a scene operation: a scene that is no longer Ready is hidden and keeps its message.
bool CompareState::sceneFailed(MediaEntry & entry)
{
    if (entry.ready())
    {
        return false;
    }
    hideScene(entry, entry.media->statusMessage());
    return true;
}

MediaEntry * CompareState::sceneEntry(size_t index)
{
    if (index >= m_entries.size())
    {
        return nullptr;
    }
    MediaEntry & entry = m_entries[index];
    if (entry.media->kind() != project::SourceKind::Scene)
    {
        setStatus(entry.name + " is not a scene");
        return nullptr;
    }
    return &entry;
}

void CompareState::sceneEdited()
{
    m_dirty = true;
    markMetricsStale();
}

void CompareState::markMetricsStale()
{
    m_metricsDirty = true;
    markFlipStale();
}

void CompareState::markFlipStale()
{
    cancelHdr();
    m_flipDirty = true;
    // Any stale mark that is not a video frame change clears the frame flag (updateVideos sets it after).
    m_flipFromFrame = false;
}

int32_t CompareState::addMemorySource(std::unique_ptr<MediaSource> media, const std::string & name)
{
    return addEntry(std::move(media), name, std::string());
}

int32_t CompareState::addEntry(std::unique_ptr<MediaSource> media, const std::string & name, const std::string & path)
{
    if (m_entries.size() >= static_cast<size_t>(kMaxSources))
    {
        setStatus("At most " + std::to_string(kMaxSources) + " sources: remove one before adding another");
        Logger::warning("Compare: " + name + " refused: at most " + std::to_string(kMaxSources) + " sources");
        return -1;
    }
    MediaEntry entry;
    entry.media = std::move(media);
    entry.name = name;
    entry.path = path;
    if (entry.ready()
        && (entry.media->width() > static_cast<uint32_t>(kMaxSourceSize)
            || entry.media->height() > static_cast<uint32_t>(kMaxSourceSize)))
    {
        setStatus(name + " is larger than " + std::to_string(kMaxSourceSize) + " pixels on a side");
        Logger::warning("Compare: " + name + " refused: larger than " + std::to_string(kMaxSourceSize)
                        + " pixels on a side");
        return -1;
    }
    entry.id = nextId();
    entry.message = entry.ready() ? std::string() : entry.media->statusMessage();
    stampFile(entry);
    entry.visible = entry.ready() && countVisible(m_entries) < static_cast<size_t>(kMaxVisible);
    m_entries.push_back(std::move(entry));
    const int32_t index = static_cast<int32_t>(m_entries.size()) - 1;
    changed();

    const MediaEntry & added = m_entries.back();
    if (added.ready())
    {
        Logger::info("Compare: added " + name + " (" + (path.empty() ? std::string("memory") : path) + ") "
                     + std::to_string(added.media->width()) + "x" + std::to_string(added.media->height()));
    }
    else
    {
        Logger::warning("Compare: " + name + " failed to load: " + added.message);
    }
    if (!added.ready())
    {
        setStatus(name + ": " + added.message);
    }
    else if (!added.visible)
    {
        setStatus("Added " + name + " hidden: At most 4 sources can be visible");
    }
    else
    {
        setStatus("Added " + name);
    }
    return index;
}

bool CompareState::removeSource(size_t index)
{
    if (index >= m_entries.size())
    {
        return false;
    }
    // The job reads the sources, so it ends before one is destroyed.
    cancelHdr();
    const std::string name = m_entries[index].name;
    m_entries.erase(m_entries.begin() + static_cast<std::ptrdiff_t>(index));
    m_flip.reset();
    m_flipReason.clear();
    changed();
    setStatus("Removed " + name);
    Logger::info("Compare: removed " + name);
    return true;
}

bool CompareState::setVisible(size_t index, bool visible)
{
    if (index >= m_entries.size())
    {
        return false;
    }
    MediaEntry & entry = m_entries[index];
    if (entry.visible == visible)
    {
        return true;
    }
    if (visible)
    {
        if (!entry.ready())
        {
            setStatus(entry.name + " cannot be shown: " + entry.message);
            return false;
        }
        if (countVisible(m_entries) >= static_cast<size_t>(kMaxVisible))
        {
            setStatus("At most 4 sources can be visible");
            return false;
        }
    }
    entry.visible = visible;
    if (!visible)
    {
        entry.role = project::SourceRole::None;
    }
    changed();
    return true;
}

bool CompareState::setRole(size_t index, project::SourceRole role)
{
    if (index >= m_entries.size())
    {
        return false;
    }
    MediaEntry & entry = m_entries[index];
    if (role == project::SourceRole::None)
    {
        entry.role = project::SourceRole::None;
        changed();
        return true;
    }
    if (!entry.visible)
    {
        setStatus(entry.name + " must be shown before it can be A or B");
        return false;
    }
    for (MediaEntry & other : m_entries)
    {
        if (other.role == role)
        {
            other.role = project::SourceRole::None;
        }
    }
    entry.role = role;
    changed();
    return true;
}

bool CompareState::setOffset(size_t index, int32_t x, int32_t y)
{
    if (index >= m_entries.size())
    {
        return false;
    }
    if (std::abs(static_cast<int64_t>(x)) > kMaxSourceOffset || std::abs(static_cast<int64_t>(y)) > kMaxSourceOffset)
    {
        setStatus("Offsets must be within +-" + std::to_string(kMaxSourceOffset) + " pixels");
        return false;
    }
    MediaEntry & entry = m_entries[index];
    if (entry.offsetX == x && entry.offsetY == y)
    {
        return true;
    }
    entry.offsetX = x;
    entry.offsetY = y;
    m_settings.anchor = project::Anchor::Custom;
    changed();
    return true;
}

bool CompareState::setBackground(size_t index, project::Rgb color)
{
    if (index >= m_entries.size())
    {
        return false;
    }
    m_entries[index].background.r = clampFloat(color.r, 0.0f, 1.0f);
    m_entries[index].background.g = clampFloat(color.g, 0.0f, 1.0f);
    m_entries[index].background.b = clampFloat(color.b, 0.0f, 1.0f);
    m_dirty = true;
    return true;
}

bool CompareState::setExposure(size_t index, float stops)
{
    if (index >= m_entries.size())
    {
        return false;
    }
    m_entries[index].exposure = clampFloat(stops, -kMaxExposureStops, kMaxExposureStops);
    m_dirty = true;
    return true;
}

bool CompareState::setToneMapper(size_t index, project::ToneMapper toneMapper)
{
    if (index >= m_entries.size())
    {
        return false;
    }
    m_entries[index].toneMapper = toneMapper;
    m_dirty = true;
    return true;
}

bool CompareState::relocate(size_t index, const std::string & utf8Path)
{
    if (index >= m_entries.size())
    {
        return false;
    }
    const std::string path = absoluteUtf8(utf8Path);
    // The job reads the entry's source, which is replaced below.
    cancelHdr();
    MediaEntry & entry = m_entries[index];
    // A video keeps the trim, loop and offset it had, where the new file accepts them.
    VideoSettings carried;
    if (const VideoSource * old = videoOfMedia(entry.media.get()))
    {
        if (old->status() == SourceStatus::Ready)
        {
            carried = settingsOf(*old);
        }
    }
    if (isSceneFile(path))
    {
        if (m_sceneHost == nullptr)
        {
            setStatus("Scene rendering is not available");
            return false;
        }
        // The entry keeps its scene settings; the new file loads with them.
        ensureRenderSize();
        SceneSettings settings;
        settings.width = m_renderWidth;
        settings.height = m_renderHeight;
        settings.shCap = entry.shCap;
        settings.debugView = entry.debugView;
        settings.flipUpAxis = entry.flipUpAxis;
        settings.antialias = entry.antialias;
        entry.media = makeScene(path, settings, true);
    }
    else if (isVideoFile(path))
    {
        entry.media = std::make_unique<VideoSource>(m_context, path);
        if (VideoSource * video = videoOfMedia(entry.media.get()))
        {
            if (video->status() == SourceStatus::Ready)
            {
                applyVideoSettings(*video, carried, fileNameUtf8(path), m_notices);
            }
        }
    }
    else
    {
        entry.media = std::make_unique<ImageSource>(m_context, path);
    }
    entry.path = path;
    entry.name = fileNameUtf8(path);
    entry.otherDriveWarning = false;
    entry.message = entry.ready() ? std::string() : entry.media->statusMessage();
    // The new file has its own stamp and no reload history, and its first render is not guarded
    // by the frame the old source rendered in.
    entry.reloadError.clear();
    entry.lastRenderFrame = 0;
    stampFile(entry);
    if (!entry.ready())
    {
        entry.visible = false;
        setStatus(entry.name + ": " + entry.message);
    }
    else
    {
        setStatus("Located " + entry.name);
        if (const SceneSource * scene = sceneOf(entry.media.get()))
        {
            entry.antialias = scene->settings().antialias;
            frameFirstScene(*scene);
        }
    }
    changed();
    return entry.ready();
}

void CompareState::applyAnchor(project::Anchor anchor)
{
    m_settings.anchor = anchor;
    if (anchor == project::Anchor::Custom)
    {
        m_dirty = true;
        return;
    }
    LayoutInput placed = m_layout;
    if (anchor == project::Anchor::TopLeft)
    {
        anchorTopLeft(placed);
    }
    else
    {
        anchorCentered(placed);
    }
    for (size_t i = 0; i < m_entries.size(); ++i)
    {
        m_entries[i].offsetX = placed.sources[i].offsetX;
        m_entries[i].offsetY = placed.sources[i].offsetY;
    }
    changed();
}

bool CompareState::setShCap(size_t index, int32_t shCap)
{
    MediaEntry * entry = sceneEntry(index);
    if (entry == nullptr)
    {
        return false;
    }
    const int32_t clamped = std::clamp(shCap, -1, kFullShDegree);
    if (entry->shCap == clamped)
    {
        return true;
    }
    entry->shCap = clamped;
    if (SceneSource * scene = sceneOf(entry->media.get()))
    {
        scene->setMaxSHDegree(clamped);
    }
    sceneEdited();
    return true;
}

bool CompareState::setFlipUpAxis(size_t index, bool flip)
{
    MediaEntry * entry = sceneEntry(index);
    if (entry == nullptr)
    {
        return false;
    }
    if (entry->flipUpAxis == flip)
    {
        return true;
    }
    entry->flipUpAxis = flip;
    if (SceneSource * scene = sceneOf(entry->media.get()))
    {
        scene->setFlipUpAxis(flip);
    }
    sceneEdited();
    return true;
}

bool CompareState::setAntialias(size_t index, bool antialias)
{
    MediaEntry * entry = sceneEntry(index);
    if (entry == nullptr)
    {
        return false;
    }
    if (entry->antialias == antialias)
    {
        return true;
    }
    entry->antialias = antialias;
    if (SceneSource * scene = sceneOf(entry->media.get()))
    {
        scene->setAntialiased(antialias);
    }
    sceneEdited();
    return true;
}

bool CompareState::setDebugView(size_t index, sabercat::DebugView view)
{
    MediaEntry * entry = sceneEntry(index);
    if (entry == nullptr)
    {
        return false;
    }
    if (entry->debugView == view)
    {
        return true;
    }
    entry->debugView = view;
    if (SceneSource * scene = sceneOf(entry->media.get()))
    {
        scene->setDebugView(view);
    }
    sceneEdited();
    return true;
}

int32_t CompareState::indexOfRole(project::SourceRole role) const
{
    for (size_t i = 0; i < m_entries.size(); ++i)
    {
        if (m_entries[i].role == role && m_entries[i].visible)
        {
            return static_cast<int32_t>(i);
        }
    }
    return -1;
}

int32_t CompareState::sourceA() const
{
    return indexOfRole(project::SourceRole::A);
}

int32_t CompareState::sourceB() const
{
    return indexOfRole(project::SourceRole::B);
}

void CompareState::defaultRoles()
{
    // Roles belong to visible sources only. The first holder of each role keeps it.
    bool haveA = false;
    bool haveB = false;
    for (MediaEntry & entry : m_entries)
    {
        if (!entry.visible)
        {
            entry.role = project::SourceRole::None;
        }
        if (entry.role == project::SourceRole::A)
        {
            entry.role = haveA ? project::SourceRole::None : project::SourceRole::A;
            haveA = true;
        }
        else if (entry.role == project::SourceRole::B)
        {
            entry.role = haveB ? project::SourceRole::None : project::SourceRole::B;
            haveB = true;
        }
    }
    // An unset role takes the first visible source that has no role.
    const auto takeFirstFree = [this](project::SourceRole role)
    {
        for (MediaEntry & entry : m_entries)
        {
            if (entry.visible && entry.role == project::SourceRole::None)
            {
                entry.role = role;
                return;
            }
        }
    };
    if (!haveA)
    {
        takeFirstFree(project::SourceRole::A);
    }
    if (!haveB)
    {
        takeFirstFree(project::SourceRole::B);
    }
    syncLayout();
    refreshView();
}

void CompareState::changed()
{
    m_dirty = true;
    markMetricsStale();
    refreshTransport();
    defaultRoles();
    // In Fit the content stays centered as sources come and go.
    if (m_layout.view.zoomMode == ZoomMode::Fit)
    {
        centerView(m_layout);
    }
}

void CompareState::syncLayout()
{
    LayoutInput fresh;
    fresh.view = m_layout.view;
    fresh.windowWidth = m_layout.windowWidth;
    fresh.windowHeight = m_layout.windowHeight;
    for (const MediaEntry & entry : m_entries)
    {
        LayoutSource geometry;
        if (entry.ready())
        {
            geometry.width = static_cast<int32_t>(entry.media->width());
            geometry.height = static_cast<int32_t>(entry.media->height());
            geometry.encoding = entry.media->encoding();
        }
        else
        {
            // A failed source keeps a 1 x 1 placeholder; it is never visible.
            geometry.width = 1;
            geometry.height = 1;
        }
        geometry.offsetX = entry.offsetX;
        geometry.offsetY = entry.offsetY;
        geometry.visible = entry.visible;
        if (!addSource(fresh, geometry))
        {
            throw std::logic_error("the layout refused source " + entry.name);
        }
    }
    fresh.view.sourceA = sourceA();
    fresh.view.sourceB = sourceB();
    m_layout = fresh;
}

// ---- Scenes and the shared camera -----------------------------------------------------

bool CompareState::setCamera(const app::OrbitCamera & camera, bool interactive)
{
    // A non-finite part would render NaN, and a distance that is not above zero or a zero
    // orientation has no view. The camera is kept as it was, and the status says why.
    const bool finite = std::isfinite(camera.pivot.x) && std::isfinite(camera.pivot.y)
        && std::isfinite(camera.pivot.z) && std::isfinite(camera.distance) && std::isfinite(camera.orientation.w)
        && std::isfinite(camera.orientation.x) && std::isfinite(camera.orientation.y)
        && std::isfinite(camera.orientation.z);
    if (!finite)
    {
        setStatus("Camera ignored: it is not finite");
        return false;
    }
    const float length = glm::length(camera.orientation);
    if (!std::isfinite(length))
    {
        setStatus("Camera ignored: it is not finite");
        return false;
    }
    if (!(camera.distance > 0.0f) || !(length > 0.0f))
    {
        setStatus("Camera ignored: the distance must be above 0 and the orientation must not be zero");
        return false;
    }
    m_camera = camera;
    ++m_cameraGeneration;
    m_dirty = true;
    // An interactive move is stamped on the frame clock, so the live policy sees the camera move.
    // Other changes (framing, a project load) are immediate, as they were before.
    if (interactive)
    {
        m_interactiveAt = m_now;
    }
    return true;
}

bool CompareState::frameScene(size_t index)
{
    MediaEntry * entry = sceneEntry(index);
    if (entry == nullptr)
    {
        return false;
    }
    const SceneSource * scene = sceneOf(entry->media.get());
    if (scene == nullptr || !entry->ready() || !scene->bounds().valid)
    {
        setStatus(entry->name + " is not loaded");
        return false;
    }
    frameCamera(*scene);
    return true;
}

bool CompareState::setRenderSize(uint32_t width, uint32_t height)
{
    const auto limit = static_cast<uint32_t>(kMaxSourceSize);
    if (width == 0 || height == 0 || width > limit || height > limit)
    {
        setStatus("The render size must be 1 to " + std::to_string(kMaxSourceSize) + " pixels per side");
        return false;
    }
    if (width == m_renderWidth && height == m_renderHeight)
    {
        // The size in place is the wanted one, so a size that was waiting is dropped.
        m_pendingW = 0;
        m_pendingH = 0;
        return true;
    }
    // A drag must not reallocate the scenes every frame: the size waits for the camera to rest.
    if (cameraMoving())
    {
        m_pendingW = width;
        m_pendingH = height;
        setStatus("Render size change waits until the camera is idle");
        return true;
    }
    return applyRenderSize(width, height);
}

bool CompareState::applyRenderSize(uint32_t width, uint32_t height)
{
    m_pendingW = 0;
    m_pendingH = 0;
    // A larger size needs device memory for every Ready scene's renderer at the new pixel count.
    // Shrinking, and the first size (no size yet), cost nothing new.
    const uint64_t oldPixels = static_cast<uint64_t>(m_renderWidth) * m_renderHeight;
    const uint64_t newPixels = static_cast<uint64_t>(width) * height;
    if (m_renderWidth != 0 && newPixels > oldPixels)
    {
        size_t readyScenes = 0;
        const uint64_t bytesInUse = readySceneBytes(readyScenes);
        if (readyScenes > 0)
        {
            const uint64_t needed = (newPixels - oldPixels) * kSceneBytesPerPixelEstimate * readyScenes;
            std::string message;
            if (!deviceBudgetOk(m_context.device(), needed, "a larger render size", message, bytesInUse))
            {
                setStatus(message);
                return false;
            }
        }
    }
    // A resized scene reallocates the color buffer that a running job reads.
    cancelHdr();
    m_renderWidth = width;
    m_renderHeight = height;
    for (MediaEntry & entry : m_entries)
    {
        SceneSource * scene = sceneOf(entry.media.get());
        if (scene == nullptr || !entry.ready())
        {
            continue;
        }
        try
        {
            scene->setSize(width, height);
        }
        catch (const std::exception & e)
        {
            hideScene(entry, e.what());
            continue;
        }
        sceneFailed(entry);
    }
    // The size changes the scenes' aspect in the layout as well as their pixels.
    changed();
    return true;
}

void CompareState::updateScenes()
{
    // A size that waited for the camera is applied once it rests. A refusal over the budget is set
    // in the status and the old size stays.
    if (renderSizePending() && !cameraMoving())
    {
        applyRenderSize(m_pendingW, m_pendingH);
    }
    // With live off, a moving camera holds every scene until it rests.
    if (!m_live && cameraMoving())
    {
        return;
    }
    bool rendered = false;
    bool hidden = false;
    for (MediaEntry & entry : m_entries)
    {
        SceneSource * scene = sceneOf(entry.media.get());
        // Hidden scenes wait until they are shown, so only visible ones render.
        if (scene == nullptr || !entry.visible || !entry.ready() || !scene->dirty(m_cameraGeneration))
        {
            continue;
        }
        // Once per frame: the UI calls this twice a frame, and the second call has nothing new. Frame 0
        // (before any beginFrame()) has no frame to compare with, so it never skips.
        if (m_frame > 0 && entry.lastRenderFrame == m_frame)
        {
            continue;
        }
        bool didRender = false;
        try
        {
            // A render may reallocate the color buffer that a running job reads.
            cancelHdr();
            didRender = scene->render(m_camera.toCamera(scene->width(), scene->height()), m_cameraGeneration);
        }
        catch (const std::exception & e)
        {
            // SceneSource::render reports its own failures. This keeps any other one from
            // reaching the frame; hiding the scene stops it from failing again each frame.
            hideScene(entry, e.what());
            hidden = true;
            continue;
        }
        if (didRender)
        {
            rendered = true;
            entry.lastRenderFrame = m_frame;
        }
        hidden = sceneFailed(entry) || hidden;
    }
    if (hidden)
    {
        changed();
    }
    if (rendered)
    {
        markMetricsStale();
    }
}

// ---- Frame clock and live updates -----------------------------------------------------

void CompareState::beginFrame(double nowSeconds)
{
    // A NaN clock would make every time comparison false, so the last good time is kept.
    if (std::isfinite(nowSeconds))
    {
        m_now = nowSeconds;
    }
    ++m_frame;
    // The first call polls, because m_lastPoll starts far in the past.
    if (m_now - m_lastPoll >= kScenePollSeconds)
    {
        pollSceneFiles();
        m_lastPoll = m_now;
    }
}

void CompareState::setLive(bool live)
{
    m_live = live;
}

bool CompareState::setLiveFlipHz(float hz)
{
    // The negated form also refuses NaN, and an infinity fails the upper bound.
    if (!(hz >= kMinLiveFlipHz && hz <= kMaxLiveFlipHz))
    {
        setStatus("The live FLIP rate must be from 0.1 to 60 Hz");
        return false;
    }
    if (hz != m_liveFlipHz)
    {
        m_liveFlipHz = hz;
        m_dirty = true;
    }
    return true;
}

bool CompareState::cameraMoving() const
{
    return m_interactiveAt >= 0.0 && m_now - m_interactiveAt < kLiveIdleSeconds;
}

bool CompareState::liveSettled() const
{
    if (cameraMoving() || metricsPending() || renderSizePending())
    {
        return false;
    }
    for (const MediaEntry & entry : m_entries)
    {
        const SceneSource * scene = sceneOf(entry.media.get());
        if (scene != nullptr && entry.visible && entry.ready() && scene->dirty(m_cameraGeneration))
        {
            return false;
        }
    }
    return true;
}

uint64_t CompareState::sceneRenderTotal() const
{
    uint64_t total = 0;
    for (const MediaEntry & entry : m_entries)
    {
        if (const SceneSource * scene = sceneOf(entry.media.get()))
        {
            total += scene->renderCount();
        }
    }
    return total;
}

bool CompareState::reloadScene(size_t index)
{
    if (index >= m_entries.size())
    {
        return false;
    }
    MediaEntry & entry = m_entries[index];
    const SceneSource * current = sceneOf(entry.media.get());
    if (current == nullptr || !entry.ready())
    {
        return false;
    }
    ++m_sceneReloadAttempts;
    // A job reads the scene's color buffer, which the swap below destroys.
    cancelHdr();
    // The new renderer exists beside the old one until the swap, so the budget check is the one for
    // another scene. A refusal keeps the old scene and says why on the entry.
    if (!sceneBudgetAllows())
    {
        entry.reloadError = m_status;
        return false;
    }
    // The file changes; the look the user set (SH cap, debug view, up axis, antialias) does not.
    const SceneSettings settings = current->settings();
    std::unique_ptr<MediaSource> fresh = makeScene(entry.path, settings, true);
    // A source that failed to load may still hold GPU work, and its renderer is destroyed when it
    // goes out of scope below, so the GPU is idle first on both paths.
    vkDeviceWaitIdle(m_context.handle());
    if (fresh->status() != SourceStatus::Ready)
    {
        entry.reloadError = "Reload failed: " + fresh->statusMessage() + "; showing the last good render";
        setStatus(entry.name + ": " + entry.reloadError);
        return false;
    }
    // The old scene is destroyed when its source is replaced, so the GPU is idle at that point too.
    entry.media = std::move(fresh);
    ++entry.sceneGeneration;
    ++m_sceneReloads;
    entry.reloadError.clear();
    // The new scene is not rendered yet, so the next updateScenes() renders it. Frame 0 keeps the
    // once-per-frame guard from skipping it when this reload happens in a frame that already rendered.
    entry.lastRenderFrame = 0;
    if (const SceneSource * loaded = sceneOf(entry.media.get()))
    {
        m_lastSceneBytes = loaded->deviceBytes();
    }
    stampFile(entry);
    markMetricsStale();
    m_dirty = true;
    setStatus("Reloaded " + entry.name);
    return true;
}

void CompareState::pollSceneFiles()
{
    for (size_t i = 0; i < m_entries.size(); ++i)
    {
        MediaEntry & entry = m_entries[i];
        if (sceneOf(entry.media.get()) == nullptr || !entry.ready() || entry.path.empty())
        {
            continue;
        }
        int64_t time = 0;
        uint64_t size = 0;
        if (!readFileStamp(entry.path, time, size))
        {
            // A file that cannot be read now (moved, locked, being replaced) counts as no change.
            continue;
        }
        if (!entry.fileStamped)
        {
            // Baseline only: the scene loaded before this stamp was read, so nothing reloads.
            entry.fileTime = time;
            entry.fileSize = size;
            entry.fileStamped = true;
            continue;
        }
        if (time == entry.fileTime && size == entry.fileSize)
        {
            continue;
        }
        // The new stamp is stored first, so a file that fails to load is tried once per change, not on
        // every poll.
        entry.fileTime = time;
        entry.fileSize = size;
        reloadScene(i);
    }
}

// ---- View -----------------------------------------------------------------------------

CompositeMode CompareState::compositeMode() const
{
    if (m_settings.split)
    {
        return CompositeMode::Panes;
    }
    switch (m_settings.single)
    {
    case SingleView::A:
    case SingleView::B:
        return CompositeMode::Panes;
    case SingleView::Wipe:
        return hasPair() ? CompositeMode::Wipe : CompositeMode::Panes;
    case SingleView::Flicker:
        return hasPair() ? CompositeMode::Flicker : CompositeMode::Panes;
    case SingleView::Diff:
        return hasPair() ? CompositeMode::Diff : CompositeMode::Panes;
    case SingleView::Depth:
    case SingleView::Alpha:
    {
        // The map shows the chosen source, which must be visible and Ready.
        const int32_t chosen = m_settings.mapUseB ? sourceB() : sourceA();
        if (chosen < 0)
        {
            return CompositeMode::Panes;
        }
        return m_settings.single == SingleView::Depth ? CompositeMode::Depth : CompositeMode::Alpha;
    }
    case SingleView::DepthDiff:
        return hasPair() ? CompositeMode::DepthDiff : CompositeMode::Panes;
    case SingleView::AlphaDiff:
        return hasPair() ? CompositeMode::AlphaDiff : CompositeMode::Panes;
    case SingleView::Flip:
        // The map exists only once FLIP has run over the pair (see updateMetrics()).
        return (hasPair() && m_flip) ? CompositeMode::Flip : CompositeMode::Panes;
    }
    return CompositeMode::Panes;
}

int32_t CompareState::resolveAb(int32_t preferred) const
{
    if (preferred >= 0 && static_cast<size_t>(preferred) < m_entries.size()
        && m_entries[static_cast<size_t>(preferred)].visible)
    {
        return preferred;
    }
    for (size_t i = 0; i < m_entries.size(); ++i)
    {
        if (m_entries[i].visible)
        {
            return static_cast<int32_t>(i);
        }
    }
    return -1;
}

bool CompareState::dividerActive() const
{
    return m_settings.split && countVisible(m_entries) == 2 && sourceA() >= 0 && sourceB() >= 0
        && sourceA() != sourceB();
}

void CompareState::setDividerFraction(double fraction)
{
    const double clamped = clampDividerFraction(fraction);
    if (clamped == m_layout.view.dividerFraction)
    {
        return;
    }
    m_layout.view.dividerFraction = clamped;
    m_dirty = true;
}

void CompareState::refreshView()
{
    const CompositeMode mode = compositeMode();
    const bool divider = dividerActive();
    // The divider replaces the two split panes with one pane (see buildComposite()).
    const bool splitPanes = mode == CompositeMode::Panes && m_settings.split && !divider;
    m_layout.view.mode = splitPanes ? ViewMode::SplitScreen : ViewMode::SingleScreen;
    m_layout.view.arrangement = m_settings.arrangement;
    m_layout.view.sourceA = sourceA();
    m_layout.view.sourceB = sourceB();
    if (divider)
    {
        // The A side is left of the divider, so the one pane shows A.
        m_layout.view.abIndex = sourceA();
    }
    else if (mode == CompositeMode::Panes)
    {
        m_layout.view.abIndex = resolveAb(m_settings.single == SingleView::B ? sourceB() : sourceA());
    }
    else if (mode == CompositeMode::Depth || mode == CompositeMode::Alpha)
    {
        // The chosen source: B when mapUseB, else A.
        m_layout.view.abIndex = m_settings.mapUseB ? sourceB() : sourceA();
    }
    else
    {
        // Wipe, Diff, the depth difference and FLIP show A; Flicker picks A or B per frame in
        // buildComposite().
        m_layout.view.abIndex = sourceA();
    }
}

void CompareState::setSplit(bool split)
{
    if (m_settings.split == split)
    {
        return;
    }
    Logger::info(split ? "Compare: split view on" : "Compare: split view off");
    m_settings.split = split;
    m_dirty = true;
    refreshView();
}

void CompareState::setSingleView(SingleView view)
{
    if (m_settings.single != view)
    {
        Logger::info(std::string("Compare: single view ") + singleViewName(view));
    }
    m_settings.single = view;
    m_dirty = true;
    refreshView();
}

void CompareState::setArrangement(PaneArrangement arrangement)
{
    m_settings.arrangement = arrangement;
    m_dirty = true;
    refreshView();
}

void CompareState::setFlickerPeriod(double seconds)
{
    double period = std::isfinite(seconds) ? seconds : 1.0;
    period = std::min(std::max(period, kMinFlickerPeriod), kMaxFlickerPeriod);
    m_settings.flickerPeriod = period;
    m_dirty = true;
}

void CompareState::setDiffGain(float gain)
{
    const float clamped = std::isfinite(gain) ? std::min(std::max(gain, 0.0f), kMaxDiffGain) : 1.0f;
    m_settings.diffGain = clamped;
    m_dirty = true;
}

void CompareState::setNearest(bool nearest)
{
    m_settings.nearest = nearest;
    m_dirty = true;
}

void CompareState::setWipeOrientation(WipeOrientation orientation)
{
    m_layout.view.wipeOrientation = orientation;
    m_dirty = true;
}

void CompareState::setMapUseB(bool useB)
{
    m_settings.mapUseB = useB;
    m_dirty = true;
    refreshView();
}

void CompareState::setDepthAuto(bool depthAuto)
{
    m_settings.depthAuto = depthAuto;
    m_dirty = true;
}

void CompareState::setDepthRange(float min, float max)
{
    if (!std::isfinite(min) || !std::isfinite(max) || !(min < max))
    {
        return;
    }
    m_settings.depthMin = min;
    m_settings.depthMax = max;
    m_dirty = true;
}

void CompareState::resolvedDepthRange(float & lo, float & hi) const
{
    lo = m_settings.depthMin;
    hi = m_settings.depthMax;
    if (!m_settings.depthAuto)
    {
        return;
    }
    // The union of the shown sources' ranges. A source without valid depth adds nothing.
    bool found = false;
    float low = 0.0f;
    float high = 0.0f;
    const int32_t candidates[2] = {sourceA(), sourceB()};
    for (const int32_t index : candidates)
    {
        if (index < 0)
        {
            continue;
        }
        const MediaEntry & entry = m_entries[static_cast<size_t>(index)];
        float rangeLo = 0.0f;
        float rangeHi = 0.0f;
        if (!entry.ready() || !entry.media->depthRange(rangeLo, rangeHi))
        {
            continue;
        }
        if (!found)
        {
            low = rangeLo;
            high = rangeHi;
            found = true;
        }
        else
        {
            low = std::min(low, rangeLo);
            high = std::max(high, rangeHi);
        }
    }
    lo = found ? low : 0.0f;
    hi = found ? high : 1.0f;
}

void CompareState::setViewSize(int32_t width, int32_t height)
{
    m_layout.windowWidth = std::max(0, width);
    m_layout.windowHeight = std::max(0, height);
}

bool CompareState::flickerShowB(double timeSeconds, double period)
{
    if (!(period > 0.0) || !std::isfinite(timeSeconds))
    {
        return false;
    }
    double phase = std::fmod(timeSeconds, period);
    if (phase < 0.0)
    {
        phase += period;
    }
    return phase >= period / 2.0;
}

CompositeInput CompareState::buildComposite(double timeSeconds) const
{
    CompositeInput input;
    input.layout = m_layout;
    input.mode = compositeMode();
    if (dividerActive())
    {
        // The divider is a vertical wipe over one pane: A left of the boundary, B right. The
        // boundary is set on this copy only, so the state's own wipe position is not touched.
        input.mode = CompositeMode::Wipe;
        input.layout.view.mode = ViewMode::SingleScreen;
        input.layout.view.abIndex = input.layout.view.sourceA;
        input.layout.view.wipeOrientation = WipeOrientation::Vertical;
        double position = 0.0;
        if (dividerImagePosition(compositeLayout(input), position))
        {
            input.layout.view.wipePosition = position;
        }
    }
    input.nearest = m_settings.nearest;
    input.flickerShowB = flickerShowB(timeSeconds, m_settings.flickerPeriod);
    input.diffGain = m_settings.diffGain;
    input.mapUseB = m_settings.mapUseB;
    input.depthMin = m_settings.depthMin;
    input.depthMax = m_settings.depthMax;
    // Only the depth modes use the range, and resolving it may read a scene's depth back.
    if (input.mode == CompositeMode::Depth || input.mode == CompositeMode::DepthDiff)
    {
        resolvedDepthRange(input.depthMin, input.depthMax);
    }
    if (input.mode == CompositeMode::Flip && m_flip)
    {
        input.flipMap = flipMapView();
        input.flipRect = m_flip->overlap;
    }
    for (size_t i = 0; i < m_entries.size(); ++i)
    {
        const MediaEntry & entry = m_entries[i];
        CompositeSource & source = input.sources[i];
        source.background = RgbaColor{entry.background.r, entry.background.g, entry.background.b, 1.0f};
        source.media = (entry.visible && entry.ready()) ? entry.media.get() : nullptr;
        source.exposureStops = entry.exposure;
        source.toneMapper = toFlipToneMapper(entry.toneMapper);
    }
    return input;
}

// ---- Metrics --------------------------------------------------------------------------

void CompareState::updateMetrics(MetricsGpu & metrics)
{
    if (!m_metricsDirty && !m_flipDirty)
    {
        m_liveDeferred = false;
        return;
    }
    // While the camera moves, the metrics and FLIP run at most liveFlipHz times a second, and not at
    // all with live off. The old result stays on screen, shown as pending, until the camera rests.
    if (cameraMoving())
    {
        if (!m_live || m_now - m_lastLiveCompute < 1.0 / static_cast<double>(m_liveFlipHz))
        {
            m_liveDeferred = true;
            return;
        }
        m_lastLiveCompute = m_now;
    }
    m_liveDeferred = false;

    // The pair and its overlap, shared by the metrics and FLIP. pairReason is empty when
    // there is a pair with an overlap.
    int32_t a = -1;
    int32_t b = -1;
    PixelRect overlap;
    const std::string pairReason = pairState(a, b, overlap);

    if (m_metricsDirty)
    {
        m_metricsDirty = false;
        m_metrics.reset();
        m_metricsReason.clear();
        if (pairReason.empty())
        {
            const MediaEntry & ea = m_entries[static_cast<size_t>(a)];
            const MediaEntry & eb = m_entries[static_cast<size_t>(b)];
            try
            {
                ++m_metricsComputes;
                m_metrics = metrics.compute(*ea.media, ea.offsetX, ea.offsetY, *eb.media, eb.offsetX, eb.offsetY, overlap);
                if (!m_metrics)
                {
                    m_metricsReason = "A and B have no pixels in common";
                }
            }
            catch (const std::exception & e)
            {
                m_metrics.reset();
                m_metricsReason = e.what();
            }
        }
        else
        {
            m_metricsReason = pairReason;
        }
    }

    if (m_flipDirty)
    {
        m_flipDirty = false;
        updateFlip(pairReason, a, b, overlap);
    }
}

std::string CompareState::pairState(int32_t & a, int32_t & b, PixelRect & overlap) const
{
    a = -1;
    b = -1;
    overlap = PixelRect{};
    if (countVisible(m_entries) < 2)
    {
        return "Add a second image to compare";
    }
    a = sourceA();
    b = sourceB();
    if (a < 0 || b < 0)
    {
        return "Choose the A and B sources";
    }
    overlap = overlapRect(m_layout, a, b);
    if (overlap.width <= 0 || overlap.height <= 0)
    {
        return "A and B do not overlap";
    }
    return std::string();
}

void CompareState::updateFlip(const std::string & pairReason, int32_t a, int32_t b, const PixelRect & overlap)
{
    // A job reads the pair it began with; a new evaluation starts from scratch.
    cancelHdr();
    m_flip.reset();
    m_flipReason.clear();
    // The flag says whether the change that asks for this evaluation was a video frame change.
    // It is consumed here: the next stale mark sets it afresh.
    const bool fromFrame = m_flipFromFrame;
    m_flipFromFrame = false;
    if (m_flipMode == project::FlipMode::Hdr)
    {
        // HDR-FLIP never runs from here unless "run on change" is on; the LDR compute is skipped.
        if (!pairReason.empty())
        {
            m_flipReason = pairReason;
            return;
        }
        const MediaEntry & ea = m_entries[static_cast<size_t>(a)];
        const MediaEntry & eb = m_entries[static_cast<size_t>(b)];
        if (!ea.ready())
        {
            m_flipReason = "no FLIP: A is not ready";
            return;
        }
        if (!eb.ready())
        {
            m_flipReason = "no FLIP: B is not ready";
            return;
        }
        if (!hdrAvailable())
        {
            m_flipReason = kHdrNeedsLinear;
            return;
        }
        // A frame change never runs HDR-FLIP, and "run on change" waits while the clock plays.
        const bool videoPair = m_entries[static_cast<size_t>(a)].media->kind() == project::SourceKind::Video
            || m_entries[static_cast<size_t>(b)].media->kind() == project::SourceKind::Video;
        if (fromFrame && videoPair)
        {
            m_flipReason = kHdrNotDuringPlayback;
            return;
        }
        if (m_hdrAuto && !fromFrame && !m_transport.playing)
        {
            // HDR-FLIP is never run while the camera moves. The retry is marked here, so it runs once
            // the camera rests.
            if (cameraMoving())
            {
                m_flipDirty = true;
                m_flipReason = "HDR-FLIP waits for the camera to rest";
                return;
            }
            requestHdr();
            return;
        }
        m_flipReason = "HDR-FLIP has not run: press Run HDR-FLIP";
        return;
    }
    if (!pairReason.empty())
    {
        m_flipReason = pairReason;
        return;
    }
    const MediaEntry & ea = m_entries[static_cast<size_t>(a)];
    const MediaEntry & eb = m_entries[static_cast<size_t>(b)];
    if (!ea.ready())
    {
        m_flipReason = "no FLIP: A is not ready";
        return;
    }
    if (!eb.ready())
    {
        m_flipReason = "no FLIP: B is not ready";
        return;
    }
    try
    {
        if (!m_flipGpu)
        {
            m_flipGpu = std::make_unique<FlipGpu>(m_context);
        }
        ++m_flipComputes;
        const auto started = std::chrono::steady_clock::now();
        m_flip = m_flipGpu->compute(*ea.media, ea.offsetX, ea.offsetY, *eb.media, eb.offsetX, eb.offsetY, overlap,
                                    m_flipPpd);
        const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        if (!m_flip)
        {
            m_flipReason = "A and B have no pixels in common";
        }
        else if (!fromFrame)
        {
            // Not logged for a video frame change: that runs every frame of playback.
            Logger::info("Compare: FLIP computed in " + fixedText(elapsed, 3) + " s, mean "
                         + fixedText(m_flip->mean, 4));
        }
    }
    catch (const std::exception & e)
    {
        m_flip.reset();
        m_flipReason = e.what();
        if (!fromFrame)
        {
            Logger::warning(std::string("Compare: FLIP failed: ") + e.what());
        }
    }
}

bool CompareState::setFlipPpd(float ppd)
{
    // The negated form also refuses NaN.
    if (!(ppd >= kMinFlipPpd && ppd <= kMaxFlipPpd))
    {
        setStatus("FLIP ppd must be from 1 to 1000");
        return false;
    }
    if (ppd == m_flipPpd)
    {
        return true;
    }
    m_flipPpd = ppd;
    m_dirty = true;
    markFlipStale();
    return true;
}

BufferView CompareState::flipMapView() const
{
    return (m_flip && m_flipGpu) ? m_flipGpu->errorMapView() : BufferView{};
}

// ---- HDR-FLIP -------------------------------------------------------------------------

bool CompareState::hdrAvailable() const
{
    const int32_t a = sourceA();
    const int32_t b = sourceB();
    if (a < 0 || b < 0)
    {
        return false;
    }
    const MediaEntry & ea = m_entries[static_cast<size_t>(a)];
    const MediaEntry & eb = m_entries[static_cast<size_t>(b)];
    if (!ea.ready() || !eb.ready())
    {
        return false;
    }
    return ea.media->encoding() == Encoding::Linear || eb.media->encoding() == Encoding::Linear;
}

bool CompareState::setFlipMode(project::FlipMode mode)
{
    if (mode == m_flipMode)
    {
        return true;
    }
    if (mode == project::FlipMode::Hdr && !hdrAvailable())
    {
        setStatus(kHdrNeedsLinear);
        return false;
    }
    Logger::info(mode == project::FlipMode::Hdr ? "Compare: FLIP mode HDR" : "Compare: FLIP mode LDR");
    m_flipMode = mode;
    m_dirty = true;
    markFlipStale();
    return true;
}

bool CompareState::setHdrToneMapper(ToneMapper toneMapper)
{
    if (toneMapper == m_hdrSettings.toneMapper)
    {
        return true;
    }
    m_hdrSettings.toneMapper = toneMapper;
    m_dirty = true;
    markFlipStale();
    return true;
}

bool CompareState::setHdrRange(std::optional<float> start, std::optional<float> stop, std::optional<int32_t> count)
{
    if (start && !validHdrExposure(*start))
    {
        setStatus("HDR-FLIP start must be a finite value from -126 to 126 stops");
        return false;
    }
    if (stop && !validHdrExposure(*stop))
    {
        setStatus("HDR-FLIP stop must be a finite value from -126 to 126 stops");
        return false;
    }
    if (start && stop && *start > *stop)
    {
        setStatus("HDR-FLIP start must not be above stop");
        return false;
    }
    if (count && !validHdrCount(*count))
    {
        setStatus("HDR-FLIP exposures must be from " + std::to_string(kMinHdrExposures) + " to "
                  + std::to_string(kMaxHdrExposures));
        return false;
    }
    if (start == m_hdrSettings.start && stop == m_hdrSettings.stop && count == m_hdrSettings.count)
    {
        return true;
    }
    m_hdrSettings.start = start;
    m_hdrSettings.stop = stop;
    m_hdrSettings.count = count;
    m_dirty = true;
    markFlipStale();
    return true;
}

void CompareState::setHdrAuto(bool on)
{
    m_hdrAuto = on;
    // Turning it on in HDR mode with no result runs now, on the next metrics update.
    if (on && m_flipMode == project::FlipMode::Hdr && !m_flip && !m_hdrRunning)
    {
        m_flipDirty = true;
    }
}

bool CompareState::requestHdr()
{
    // The result requested now is the fresh one, so the stale flag is cleared here.
    cancelHdr();
    m_flip.reset();
    m_flipReason.clear();
    m_flipDirty = false;
    m_hdrSecondsPerExposure = 0.0f;
    m_hdrSeconds = 0.0;
    if (m_flipMode != project::FlipMode::Hdr)
    {
        m_flipReason = "FLIP is in LDR mode: choose HDR first";
        return false;
    }
    int32_t a = -1;
    int32_t b = -1;
    PixelRect overlap;
    const std::string pairReason = pairState(a, b, overlap);
    if (!pairReason.empty())
    {
        m_flipReason = pairReason;
        return false;
    }
    const MediaEntry & ea = m_entries[static_cast<size_t>(a)];
    const MediaEntry & eb = m_entries[static_cast<size_t>(b)];
    if (!ea.ready())
    {
        m_flipReason = "no FLIP: A is not ready";
        return false;
    }
    if (!eb.ready())
    {
        m_flipReason = "no FLIP: B is not ready";
        return false;
    }
    if (!hdrAvailable())
    {
        m_flipReason = kHdrNeedsLinear;
        return false;
    }
    try
    {
        if (!m_flipGpu)
        {
            m_flipGpu = std::make_unique<FlipGpu>(m_context);
        }
        const bool started = m_flipGpu->beginHdr(*ea.media, ea.offsetX, ea.offsetY, *eb.media, eb.offsetX, eb.offsetY,
                                                 overlap, m_flipPpd, m_hdrSettings);
        if (!started)
        {
            m_flipReason = "A and B have no pixels in common";
            return false;
        }
    }
    catch (const std::exception & e)
    {
        if (m_flipGpu)
        {
            m_flipGpu->cancelHdr();
        }
        m_flipReason = e.what();
        Logger::warning(std::string("Compare: HDR-FLIP could not start: ") + e.what());
        return false;
    }
    m_hdrRunning = true;
    Logger::info("Compare: HDR-FLIP started (" + std::to_string(m_flipGpu->hdrCount()) + " exposures)");
    // An explicit run is the result the next frame change makes stale.
    m_hdrVideoStale = false;
    return true;
}

float CompareState::hdrProgress() const
{
    if (!m_hdrRunning || !m_flipGpu)
    {
        return 0.0f;
    }
    const uint32_t count = m_flipGpu->hdrCount();
    if (count == 0)
    {
        return 0.0f;
    }
    return static_cast<float>(m_flipGpu->hdrDone()) / static_cast<float>(count);
}

void CompareState::cancelHdr()
{
    if (m_flipGpu)
    {
        m_flipGpu->cancelHdr();
    }
    if (m_hdrRunning)
    {
        m_hdrRunning = false;
        m_flipReason = "HDR-FLIP cancelled";
    }
}

bool CompareState::stepHdr(double budgetSeconds)
{
    if (!m_hdrRunning || !m_flipGpu || !m_flipGpu->hdrActive())
    {
        m_hdrRunning = false;
        return false;
    }
    const uint32_t remaining = m_flipGpu->hdrCount() - m_flipGpu->hdrDone();
    // The first chunk is one exposure. After that, as many as the budget fits at the measured
    // time per exposure, and at least one.
    uint32_t chunk = 1;
    if (budgetSeconds > 0.0 && m_hdrSecondsPerExposure > 0.0f)
    {
        const double fits = budgetSeconds / static_cast<double>(m_hdrSecondsPerExposure);
        if (fits >= static_cast<double>(remaining))
        {
            chunk = remaining;
        }
        else if (fits >= 1.0)
        {
            chunk = static_cast<uint32_t>(fits);
        }
    }
    chunk = std::max<uint32_t>(1, std::min(chunk, remaining));

    const auto started = std::chrono::steady_clock::now();
    bool done = false;
    try
    {
        done = m_flipGpu->stepHdr(chunk);
    }
    catch (const std::exception & e)
    {
        m_flipGpu->cancelHdr();
        m_hdrRunning = false;
        m_flip.reset();
        m_flipReason = e.what();
        Logger::warning(std::string("Compare: HDR-FLIP failed: ") + e.what());
        return true;
    }
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    m_hdrSeconds += elapsed;
    // A floor keeps a very fast chunk from reading as free: the next chunk is then the rest of the job.
    m_hdrSecondsPerExposure = static_cast<float>(std::max(elapsed / static_cast<double>(chunk), 1.0e-9));
    if (!done)
    {
        return false;
    }

    m_hdrRunning = false;
    try
    {
        m_flip = m_flipGpu->finishHdr();
        m_flipReason.clear();
        Logger::info("Compare: HDR-FLIP finished in " + fixedText(m_hdrSeconds, 3) + " s");
    }
    catch (const std::exception & e)
    {
        m_flipGpu->cancelHdr();
        m_flip.reset();
        m_flipReason = e.what();
        Logger::error(std::string("Compare: HDR-FLIP failed: ") + e.what());
    }
    return true;
}

void CompareState::runHdrBlocking()
{
    if (!m_hdrRunning && !m_flip && !requestHdr())
    {
        return;
    }
    while (m_hdrRunning)
    {
        stepHdr(kUnlimitedHdrBudget);
    }
}

std::string CompareState::pairEncodingWarning() const
{
    const int32_t a = sourceA();
    const int32_t b = sourceB();
    if (a < 0 || b < 0)
    {
        return std::string();
    }
    const MediaEntry & ea = m_entries[static_cast<size_t>(a)];
    const MediaEntry & eb = m_entries[static_cast<size_t>(b)];
    if (!ea.ready() || !eb.ready() || ea.media->encoding() == eb.media->encoding())
    {
        return std::string();
    }
    const char * tail = "in LDR the linear one is clamped and sRGB-encoded; in HDR the display one is decoded "
                        "from sRGB to linear";
    if (ea.media->encoding() == Encoding::Display)
    {
        return std::string("A is display-encoded and B is linear: ") + tail;
    }
    return std::string("A is linear and B is display-encoded: ") + tail;
}

// ---- Video and the transport ----------------------------------------------------------

bool CompareState::hasVideo() const
{
    return std::any_of(m_entries.begin(), m_entries.end(), [](const MediaEntry & entry)
                       { return entry.media->kind() == project::SourceKind::Video; });
}

bool CompareState::hasVisibleVideo() const
{
    return std::any_of(m_entries.begin(), m_entries.end(), [](const MediaEntry & entry)
                       { return entry.visible && entry.ready() && entry.media->kind() == project::SourceKind::Video; });
}

void CompareState::play()
{
    if (!hasVisibleVideo())
    {
        setStatus("Show a video to play it");
        return;
    }
    if (!(m_transport.duration > 0.0))
    {
        setStatus("The visible videos have no duration to play");
        return;
    }
    // At the end without loop, play starts again from the beginning.
    if (!m_transport.loop && m_transport.time >= m_transport.duration)
    {
        m_transport.time = 0.0;
    }
    m_transport.playing = true;
}

void CompareState::pause()
{
    m_transport.playing = false;
}

void CompareState::togglePlay()
{
    if (m_transport.playing)
    {
        pause();
    }
    else
    {
        play();
    }
}

bool CompareState::seek(double seconds)
{
    if (!std::isfinite(seconds))
    {
        setStatus("The seek time must be a finite number of seconds");
        return false;
    }
    m_transport.time = std::clamp(seconds, 0.0, m_transport.duration);
    m_dirty = true;
    return true;
}

bool CompareState::setSpeed(double speed)
{
    // The negated form also refuses NaN.
    if (!(speed >= kMinPlaybackSpeed && speed <= kMaxPlaybackSpeed))
    {
        setStatus("speed must be between 0.25 and 4");
        return false;
    }
    if (speed != m_transport.speed)
    {
        m_transport.speed = speed;
        m_dirty = true;
    }
    return true;
}

void CompareState::setTransportLoop(bool loop)
{
    m_transport.loop = loop;
    m_dirty = true;
}

void CompareState::stepFrames(int32_t n)
{
    pause();
    VideoSource * video = referenceVideo();
    if (video == nullptr)
    {
        setStatus("Show a video to step its frames");
        return;
    }
    const compare::TimeSource & timing = video->timeSource();
    const int64_t frame = compare::frameIndexAt(timing, m_transport.time);
    const int64_t target = std::clamp<int64_t>(frame + static_cast<int64_t>(n), 0, timing.frameCount - 1);
    // The source time of the target frame, less the video's own offset, is the global time.
    const double globalTime = static_cast<double>(target) / timing.fps - timing.timeOffset;
    m_transport.time = std::clamp(globalTime, 0.0, m_transport.duration);
    m_dirty = true;
}

void CompareState::advanceTime(double dtSeconds)
{
    if (!m_transport.playing)
    {
        return;
    }
    const double dt = std::isfinite(dtSeconds) ? std::clamp(dtSeconds, 0.0, kMaxFrameStep) : 0.0;
    compare::Timeline timeline;
    timeline.time = m_transport.time;
    timeline.speed = m_transport.speed;
    timeline.duration = m_transport.duration;
    timeline.loop = m_transport.loop;
    try
    {
        compare::advance(timeline, dt);
    }
    catch (const std::exception & e)
    {
        m_transport.playing = false;
        setStatus(std::string("Playback stopped: ") + e.what());
        return;
    }
    m_transport.time = timeline.time;
    if (!m_transport.loop && m_transport.time >= m_transport.duration)
    {
        m_transport.time = m_transport.duration;
        m_transport.playing = false;
    }
}

bool CompareState::updateVideos()
{
    bool frameChanged = false;
    for (MediaEntry & entry : m_entries)
    {
        VideoSource * video = videoOfMedia(entry.media.get());
        if (video == nullptr || !entry.visible || !entry.ready())
        {
            continue;
        }
        if (video->update(m_transport.time))
        {
            frameChanged = true;
        }
        // A failed decode keeps the previous frame and the video reports why; the entry stays Ready.
        entry.message = video->statusMessage();
    }
    if (frameChanged)
    {
        ++m_frameGeneration;
        markMetricsStale();
        // Set after the stale mark, which clears it: the FLIP update that follows knows this came from a frame.
        m_flipFromFrame = true;
        m_hdrVideoStale = true;
    }
    return frameChanged;
}

bool CompareState::hdrStaleForVideo() const
{
    if (m_flipMode != project::FlipMode::Hdr || !m_hdrVideoStale)
    {
        return false;
    }
    const auto isVideo = [this](int32_t index)
    {
        return index >= 0 && m_entries[static_cast<size_t>(index)].media->kind() == project::SourceKind::Video;
    };
    return isVideo(sourceA()) || isVideo(sourceB());
}

VideoSource * CompareState::videoOf(size_t index)
{
    if (index >= m_entries.size())
    {
        return nullptr;
    }
    const MediaEntry & entry = m_entries[index];
    return entry.ready() ? videoOfMedia(entry.media.get()) : nullptr;
}

const VideoSource * CompareState::videoOf(size_t index) const
{
    if (index >= m_entries.size())
    {
        return nullptr;
    }
    const MediaEntry & entry = m_entries[index];
    return entry.ready() ? videoOfMedia(entry.media.get()) : nullptr;
}

int64_t CompareState::videoFrameIndex(size_t index) const
{
    const VideoSource * video = videoOf(index);
    return video != nullptr ? video->frameIndex() : -1;
}

bool CompareState::setVideoTrim(size_t index, double trimIn, double trimOut)
{
    VideoSource * video = readyVideo(index);
    if (video == nullptr)
    {
        return false;
    }
    if (!video->setTrim(trimIn, trimOut))
    {
        setStatus(m_entries[index].name + ": " + videoRefusal(*video, "the trim is not valid"));
        return false;
    }
    videoEdited();
    return true;
}

bool CompareState::setVideoLoop(size_t index, bool loop)
{
    VideoSource * video = readyVideo(index);
    if (video == nullptr)
    {
        return false;
    }
    video->setLoop(loop);
    videoEdited();
    return true;
}

bool CompareState::setVideoTimeOffset(size_t index, double seconds)
{
    VideoSource * video = readyVideo(index);
    if (video == nullptr)
    {
        return false;
    }
    if (!video->setTimeOffset(seconds))
    {
        setStatus(m_entries[index].name + ": " + videoRefusal(*video, "the time offset is not valid"));
        return false;
    }
    videoEdited();
    return true;
}

bool CompareState::setVideoTimeOffsetFrames(size_t index, int32_t frames)
{
    const VideoSource * video = readyVideo(index);
    if (video == nullptr)
    {
        return false;
    }
    return setVideoTimeOffset(index, static_cast<double>(frames) / video->fps());
}

void CompareState::videoEdited()
{
    m_dirty = true;
    markMetricsStale();
    refreshTransport();
}

VideoSource * CompareState::readyVideo(size_t index)
{
    if (index >= m_entries.size())
    {
        return nullptr;
    }
    MediaEntry & entry = m_entries[index];
    VideoSource * video = videoOfMedia(entry.media.get());
    if (video == nullptr)
    {
        setStatus(entry.name + " is not a video");
        return nullptr;
    }
    if (!entry.ready())
    {
        setStatus(entry.name + " is not ready: " + entry.message);
        return nullptr;
    }
    return video;
}

VideoSource * CompareState::referenceVideo()
{
    const int32_t candidates[2] = {sourceA(), sourceB()};
    for (const int32_t index : candidates)
    {
        if (index >= 0)
        {
            if (VideoSource * video = videoOfMedia(m_entries[static_cast<size_t>(index)].media.get()))
            {
                return video;
            }
        }
    }
    for (MediaEntry & entry : m_entries)
    {
        if (entry.visible && entry.ready())
        {
            if (VideoSource * video = videoOfMedia(entry.media.get()))
            {
                return video;
            }
        }
    }
    return nullptr;
}

void CompareState::refreshTransport()
{
    std::vector<compare::TimeSource> timings;
    for (const MediaEntry & entry : m_entries)
    {
        if (entry.visible && entry.ready())
        {
            if (const VideoSource * video = videoOfMedia(entry.media.get()))
            {
                timings.push_back(video->timeSource());
            }
        }
    }
    double duration = 0.0;
    if (!timings.empty())
    {
        try
        {
            duration = compare::timelineDuration(timings);
        }
        catch (const std::exception & e)
        {
            duration = 0.0;
            setStatus(std::string("Video timing is not valid: ") + e.what());
        }
    }
    m_transport.duration = (std::isfinite(duration) && duration > 0.0) ? duration : 0.0;
    if (m_transport.duration <= 0.0)
    {
        m_transport.playing = false;
    }
    m_transport.time = timings.empty() ? 0.0 : std::clamp(m_transport.time, 0.0, m_transport.duration);
}

// ---- Project --------------------------------------------------------------------------

project::CompareProject CompareState::toProject() const
{
    project::CompareProject out;
    for (const MediaEntry & entry : m_entries)
    {
        if (entry.path.empty())
        {
            continue;
        }
        project::CompareSource source;
        source.id = entry.id;
        source.path = entry.path;
        source.background = entry.background;
        source.exposure = entry.exposure;
        source.offset = project::Vec2i{entry.offsetX, entry.offsetY};
        source.kind = entry.media->kind();
        source.role = entry.role;
        source.toneMapper = entry.toneMapper;
        source.visible = entry.visible;
        if (source.kind == project::SourceKind::Video)
        {
            // A Ready video writes its settings. One that is not Ready holds none to write, so it
            // keeps the defaults (a saved trim of a missing video is not kept across a save).
            if (const VideoSource * video = videoOfMedia(entry.media.get()))
            {
                if (video->status() == SourceStatus::Ready)
                {
                    const VideoSettings settings = settingsOf(*video);
                    source.trimIn = settings.trimIn;
                    source.trimOut = settings.trimOut;
                    source.timeOffset = settings.timeOffset;
                    source.loop = settings.loop;
                }
            }
        }
        if (source.kind == project::SourceKind::Scene)
        {
            source.flipUpAxis = entry.flipUpAxis;
            source.shCap = entry.shCap;
            source.antialias = entry.antialias;
            source.debugView = app::debugViewKey(entry.debugView);
        }
        out.sources.push_back(source);
    }

    // The camera as toCamera() sets its intrinsics. Width and height are the render size,
    // zero until a scene sets it; a camera with no size is not restored on load.
    out.camera.pivot = project::Vec3{m_camera.pivot.x, m_camera.pivot.y, m_camera.pivot.z};
    out.camera.orientation = project::Quat{m_camera.orientation.w, m_camera.orientation.x, m_camera.orientation.y,
                                           m_camera.orientation.z};
    out.camera.distance = m_camera.distance;
    const float renderHeight = static_cast<float>(m_renderHeight);
    out.camera.fx = renderHeight;
    out.camera.fy = renderHeight;
    out.camera.cx = static_cast<float>(m_renderWidth) * 0.5f;
    out.camera.cy = renderHeight * 0.5f;
    out.camera.width = static_cast<int32_t>(m_renderWidth);
    out.camera.height = static_cast<int32_t>(m_renderHeight);

    out.view.mode = m_settings.split ? project::ViewMode::Split : project::ViewMode::Single;
    out.view.layout = m_settings.arrangement == PaneArrangement::Grid2x2 ? project::GridLayout::Grid
                                                                         : project::GridLayout::Row;
    out.view.anchor = m_settings.anchor;
    out.view.pan = project::Vec2{static_cast<float>(m_layout.view.panX), static_cast<float>(m_layout.view.panY)};
    out.view.zoom.mode = m_layout.view.zoomMode == ZoomMode::Fit ? project::ZoomMode::Fit : project::ZoomMode::Percent;
    out.view.zoom.percent = static_cast<float>(clampZoomPercent(m_layout.view.zoomPercent));
    out.view.wipe.position = static_cast<float>(m_layout.view.wipePosition);
    out.view.wipe.orientation = m_layout.view.wipeOrientation == WipeOrientation::Vertical
        ? project::WipeOrientation::Vertical
        : project::WipeOrientation::Horizontal;
    out.view.splitDivider = static_cast<float>(m_layout.view.dividerFraction);
    out.view.diffGain = m_settings.diffGain;
    out.view.flickerPeriodSeconds = static_cast<float>(m_settings.flickerPeriod);
    out.view.singleView = toProjectSingleView(m_settings.single);
    out.view.mapUseB = m_settings.mapUseB;
    out.view.depthAuto = m_settings.depthAuto;
    out.view.ranges.depth = project::FloatRange{m_settings.depthMin, m_settings.depthMax};
    out.view.paneCount = std::min(std::max(static_cast<int32_t>(countVisible(m_entries)), 1), kMaxVisible);
    // The FLIP ppd. The default is saved as 0, which the file reads as the default.
    out.flip.ppd = m_flipPpd == defaultFlipPpd() ? 0 : static_cast<int32_t>(std::lround(m_flipPpd));
    // The live rate is saved only when it is not the default, as the ppd is.
    out.flip.liveHz = m_liveFlipHz == kDefaultLiveFlipHz ? 0.0f : m_liveFlipHz;
    out.flip.mode = m_flipMode;
    out.flip.toneMapper = toProjectToneMapper(m_hdrSettings.toneMapper);
    out.flip.startExposure = m_hdrSettings.start;
    out.flip.stopExposure = m_hdrSettings.stop;
    out.flip.numExposures = m_hdrSettings.count;
    // The playing state is not saved; the clock, the speed and the loop are.
    out.timeline.time = m_transport.time;
    out.timeline.speed = m_transport.speed;
    out.timeline.loop = m_transport.loop;
    return out;
}

void CompareState::fromProject(const project::CompareProject & project)
{
    // The job reads the sources, which are replaced below.
    cancelHdr();
    m_entries.clear();
    m_transport = Transport{};
    m_hdrVideoStale = false;
    m_metrics.reset();
    m_metricsReason.clear();
    m_flip.reset();
    m_flipReason.clear();
    if (m_flipGpu)
    {
        m_flipGpu->release();
    }
    m_notices.clear();
    m_lastSceneBytes = 0;
    restoreCamera(project.camera);

    // Image entries load now. A scene entry gets a placeholder until the render size is
    // known: the first image sets its default, so images come before scenes are built.
    bool anyScene = false;
    for (const project::CompareSource & source : project.sources)
    {
        MediaEntry entry;
        entry.id = source.id;
        entry.path = source.path;
        entry.name = fileNameUtf8(source.path);
        entry.background = source.background;
        entry.exposure = clampFloat(source.exposure, -kMaxExposureStops, kMaxExposureStops);
        entry.offsetX = std::min(std::max(source.offset.x, -kMaxSourceOffset), kMaxSourceOffset);
        entry.offsetY = std::min(std::max(source.offset.y, -kMaxSourceOffset), kMaxSourceOffset);
        entry.toneMapper = source.toneMapper;
        entry.role = source.role;
        entry.otherDriveWarning = source.pathIsAbsoluteOtherDrive;
        if (isSceneFile(source.path))
        {
            anyScene = true;
            entry.media = std::make_unique<UnavailableScene>(std::string());
            entry.shCap = std::clamp(source.shCap, -1, kFullShDegree);
            entry.flipUpAxis = source.flipUpAxis;
            entry.antialias = source.antialias;
            if (!app::parseDebugViewKey(source.debugView.c_str(), entry.debugView))
            {
                entry.debugView = sabercat::DebugView::Final;
            }
        }
        else
        {
            if (isVideoFile(source.path))
            {
                entry.media = std::make_unique<VideoSource>(m_context, source.path);
                // The saved trim, loop and offset apply once the video is Ready.
                if (VideoSource * video = videoOfMedia(entry.media.get()))
                {
                    if (video->status() == SourceStatus::Ready)
                    {
                        applyVideoSettings(*video, settingsOfSource(source), entry.name, m_notices);
                    }
                }
            }
            else
            {
                entry.media = std::make_unique<ImageSource>(m_context, source.path);
            }
        }
        m_entries.push_back(std::move(entry));
    }

    if (anyScene)
    {
        ensureRenderSize();
    }
    for (size_t i = 0; i < m_entries.size(); ++i)
    {
        MediaEntry & entry = m_entries[i];
        const project::CompareSource & source = project.sources[i];
        if (isSceneFile(source.path))
        {
            SceneSettings settings;
            settings.width = m_renderWidth;
            settings.height = m_renderHeight;
            settings.shCap = entry.shCap;
            settings.debugView = entry.debugView;
            settings.flipUpAxis = entry.flipUpAxis;
            // The saved antialias flag is applied to the scene, not the file's default.
            settings.antialias = entry.antialias;
            entry.media = makeScene(entry.path, settings, true);
        }
        entry.message = entry.ready() ? std::string() : entry.media->statusMessage();
        stampFile(entry);
        entry.visible = source.visible && entry.ready() && countVisible(m_entries) < static_cast<size_t>(kMaxVisible);
        if (const SceneSource * scene = sceneOf(entry.media.get()))
        {
            entry.antialias = scene->settings().antialias;
            frameFirstScene(*scene);
        }
    }

    m_settings = ViewSettings{};
    m_settings.split = project.view.mode == project::ViewMode::Split;
    m_settings.arrangement = project.view.layout == project::GridLayout::Grid ? PaneArrangement::Grid2x2
                                                                              : PaneArrangement::Row;
    m_settings.anchor = project.view.anchor;
    m_settings.single = toSingleView(project.view.singleView);
    m_settings.mapUseB = project.view.mapUseB;
    m_settings.depthAuto = project.view.depthAuto;
    const project::FloatRange & depthRange = project.view.ranges.depth;
    if (std::isfinite(depthRange.min) && std::isfinite(depthRange.max) && depthRange.min < depthRange.max)
    {
        m_settings.depthMin = depthRange.min;
        m_settings.depthMax = depthRange.max;
    }
    m_settings.flickerPeriod = (std::isfinite(project.view.flickerPeriodSeconds) && project.view.flickerPeriodSeconds > 0.0f)
        ? std::min(std::max(static_cast<double>(project.view.flickerPeriodSeconds), kMinFlickerPeriod), kMaxFlickerPeriod)
        : 1.0;
    m_settings.diffGain = finiteOr(project.view.diffGain, 1.0f) < 0.0f
        ? 1.0f
        : std::min(finiteOr(project.view.diffGain, 1.0f), kMaxDiffGain);

    // The FLIP ppd: 0 is the default; a value outside the range loads as the default with a notice.
    const float savedPpd = static_cast<float>(project.flip.ppd);
    if (project.flip.ppd == 0)
    {
        m_flipPpd = defaultFlipPpd();
    }
    else if (savedPpd >= kMinFlipPpd && savedPpd <= kMaxFlipPpd)
    {
        m_flipPpd = savedPpd;
    }
    else
    {
        m_flipPpd = defaultFlipPpd();
        m_notices.push_back("FLIP ppd " + std::to_string(project.flip.ppd) + " is out of range; using the default");
    }

    // The live FLIP rate: 0 is the default; a value outside the range loads as the default with a notice.
    if (project.flip.liveHz == 0.0f)
    {
        m_liveFlipHz = kDefaultLiveFlipHz;
    }
    else if (project.flip.liveHz >= kMinLiveFlipHz && project.flip.liveHz <= kMaxLiveFlipHz)
    {
        m_liveFlipHz = project.flip.liveHz;
    }
    else
    {
        m_liveFlipHz = kDefaultLiveFlipHz;
        m_notices.push_back("FLIP live rate " + std::to_string(project.flip.liveHz) + " is out of range; using the default");
    }

    // The FLIP mode is kept even when HDR is not available yet: the pair decides at the next
    // metrics update. Out-of-range exposure overrides load as automatic, with a notice each.
    m_flipMode = project.flip.mode;
    m_hdrSettings = FlipHdrSettings{};
    m_hdrSettings.toneMapper = toFlipToneMapper(project.flip.toneMapper);
    m_hdrSettings.start = project.flip.startExposure;
    m_hdrSettings.stop = project.flip.stopExposure;
    m_hdrSettings.count = project.flip.numExposures;
    if (m_hdrSettings.start && !validHdrExposure(*m_hdrSettings.start))
    {
        m_hdrSettings.start.reset();
        m_notices.push_back("HDR-FLIP start exposure is out of range; using the automatic value");
    }
    if (m_hdrSettings.stop && !validHdrExposure(*m_hdrSettings.stop))
    {
        m_hdrSettings.stop.reset();
        m_notices.push_back("HDR-FLIP stop exposure is out of range; using the automatic value");
    }
    if (m_hdrSettings.start && m_hdrSettings.stop && *m_hdrSettings.start > *m_hdrSettings.stop)
    {
        m_hdrSettings.start.reset();
        m_hdrSettings.stop.reset();
        m_notices.push_back("HDR-FLIP start exposure is above the stop exposure; using the automatic range");
    }
    if (m_hdrSettings.count && !validHdrCount(*m_hdrSettings.count))
    {
        m_hdrSettings.count.reset();
        m_notices.push_back("HDR-FLIP exposure count is out of range; using the automatic count");
    }

    m_layout.view.panX = std::isfinite(project.view.pan.x) ? project.view.pan.x : 0.0f;
    m_layout.view.panY = std::isfinite(project.view.pan.y) ? project.view.pan.y : 0.0f;
    m_layout.view.zoomMode = project.view.zoom.mode == project::ZoomMode::Fit ? ZoomMode::Fit : ZoomMode::Percent;
    m_layout.view.zoomPercent = clampZoomPercent(project.view.zoom.percent);
    m_layout.view.wipePosition = std::isfinite(project.view.wipe.position) ? project.view.wipe.position : 0.0f;
    m_layout.view.wipeOrientation = project.view.wipe.orientation == project::WipeOrientation::Vertical
        ? WipeOrientation::Vertical
        : WipeOrientation::Horizontal;
    // A file without the key holds the default; clamped, so a file cannot put the divider off the window.
    m_layout.view.dividerFraction = clampDividerFraction(project.view.splitDivider);

    // The transport: paused, with the saved clock and speed. The clock is clamped to the
    // duration of the videos, which changed() computes.
    m_transport.loop = project.timeline.loop;
    m_transport.time = (std::isfinite(project.timeline.time) && project.timeline.time > 0.0) ? project.timeline.time : 0.0;
    if (project.timeline.speed >= kMinPlaybackSpeed && project.timeline.speed <= kMaxPlaybackSpeed)
    {
        m_transport.speed = project.timeline.speed;
    }
    else
    {
        m_transport.speed = 1.0;
        m_notices.push_back("timeline speed " + std::to_string(project.timeline.speed) + " is out of range; using 1");
    }
    changed();
}

void CompareState::newProject()
{
    // The job reads the sources, which are cleared below.
    cancelHdr();
    m_entries.clear();
    m_settings = ViewSettings{};
    m_flipPpd = defaultFlipPpd();
    // The saved live rate and any waiting size belong to the project being replaced.
    m_liveFlipHz = kDefaultLiveFlipHz;
    m_pendingW = 0;
    m_pendingH = 0;
    m_flip.reset();
    m_flipReason.clear();
    if (m_flipGpu)
    {
        m_flipGpu->release();
    }
    m_flipMode = project::FlipMode::Ldr;
    m_hdrSettings = FlipHdrSettings{};
    m_hdrAuto = false;
    m_transport = Transport{};
    m_hdrVideoStale = false;
    const int32_t width = m_layout.windowWidth;
    const int32_t height = m_layout.windowHeight;
    m_layout = LayoutInput{};
    m_layout.windowWidth = width;
    m_layout.windowHeight = height;
    m_camera = app::OrbitCamera{};
    m_cameraFramed = false;
    m_cameraGeneration += 1;
    m_renderWidth = 0;
    m_renderHeight = 0;
    m_lastSceneBytes = 0;
    m_metrics.reset();
    m_metricsReason.clear();
    markMetricsStale();
    m_projectPath.clear();
    m_dirty = false;
    m_notices.clear();
    syncLayout();
    refreshView();
    setStatus("New project");
    Logger::info("Compare: new project");
}

bool CompareState::openProject(const std::string & utf8Path, std::string & error)
{
    std::vector<std::string> warnings;
    project::CompareProject loaded;
    try
    {
        loaded = project::loadCompareProject(utf8Path, &warnings);
    }
    catch (const std::exception & e)
    {
        error = e.what();
        setStatus("Could not open " + fileNameUtf8(utf8Path) + ": " + error);
        Logger::warning("Compare: could not open " + utf8Path + ": " + error);
        return false;
    }
    fromProject(loaded);
    // fromProject() leaves its own notices (the FLIP ppd) in m_notices; keep them.
    const std::vector<std::string> loadNotices = m_notices;
    m_projectPath = absoluteUtf8(utf8Path);
    m_dirty = false;
    m_notices = warnings;
    m_notices.insert(m_notices.end(), loadNotices.begin(), loadNotices.end());
    for (const MediaEntry & entry : m_entries)
    {
        if (entry.otherDriveWarning)
        {
            m_notices.push_back(entry.name + " is stored on another drive, so its path is kept absolute.");
        }
        if (!entry.ready())
        {
            m_notices.push_back(entry.name + " could not be loaded (" + entry.message + "). Use Locate... to find it.");
        }
    }
    setStatus("Opened " + fileNameUtf8(utf8Path));
    Logger::info("Compare: opened project " + m_projectPath + " (" + std::to_string(m_entries.size()) + " sources)");
    return true;
}

bool CompareState::saveProject(const std::string & utf8Path, std::string & error)
{
    try
    {
        project::saveCompareProject(utf8Path, toProject());
    }
    catch (const std::exception & e)
    {
        error = e.what();
        setStatus("Could not save: " + error);
        Logger::warning("Compare: could not save " + utf8Path + ": " + error);
        return false;
    }
    m_projectPath = absoluteUtf8(utf8Path);
    m_dirty = false;
    setStatus("Saved " + fileNameUtf8(utf8Path));
    Logger::info("Compare: saved project " + m_projectPath);
    return true;
}

std::string CompareState::nextId() const
{
    for (int n = 1;; ++n)
    {
        std::string id = "source" + std::to_string(n);
        const bool taken = std::any_of(m_entries.begin(), m_entries.end(),
                                       [&id](const MediaEntry & entry) { return entry.id == id; });
        if (!taken)
        {
            return id;
        }
    }
}

} // namespace sabercat::compare::gui
