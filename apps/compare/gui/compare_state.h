// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The compare window's state: the media list, the view settings, the cached metrics, the
// shared scene camera and the project file. No ImGui and no GLFW, so the self-test can
// drive it directly.
//
// The layout (compare_layout.h) is kept in sync with the media list: layout index i is
// media index i, and a hidden source stays in the layout with visible = false, so indices
// are stable while the user toggles visibility. The layout is rebuilt from the list after
// each change to it; only the view fields (zoom, pan, wipe) are edited in place through
// viewLayout(), and viewChanged() records such an edit for the project file.
#pragma once

#include "compare_layout.h"
#include "compare_project.h"
#include "compositor.h"
#include "flip_gpu.h"
#include "image_compare.h"
#include "media_source.h"
#include "metrics_gpu.h"
#include "scene_source.h"

#include "orbit_camera.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace sabercat::compare::gui
{

// Live scene updates. While the camera moves, visible scenes render every frame and the metrics
// and FLIP wait; the final render and FLIP run once the camera has rested for kLiveIdleSeconds.
// This is the one place the idle time lives.
constexpr double kLiveIdleSeconds = 0.15;
// The FLIP rate while the camera moves. The one place the default lives: the project file
// writes its rate only when it differs from this.
constexpr float kDefaultLiveFlipHz = 4.0f;
// The range of the live FLIP rate. The project file refuses values outside it too.
constexpr float kMinLiveFlipHz = project::kMinLiveFlipHz;
constexpr float kMaxLiveFlipHz = project::kMaxLiveFlipHz;
// How often the scene files are checked for a new write time or size. Two file attributes
// once a second, not a watcher thread, so a changed file reloads within about a second.
constexpr double kScenePollSeconds = 1.0;

class GpuContext;
class VideoSource;

// What the single-screen view shows. Split is the layout's SplitScreen form and is not
// one of these. Depth and Alpha show the chosen source (ViewSettings::mapUseB). Flip shows
// the GPU FLIP error map (LDR or HDR) over the A/B overlap (CompareState::flip()).
enum class SingleView : uint8_t
{
    A,
    B,
    Wipe,
    Flicker,
    Diff,
    Depth,
    Alpha,
    DepthDiff,
    AlphaDiff,
    Flip
};

// One media entry. The scene fields (shCap through antialias) are used only by scene
// sources. Field order keeps the struct close to free of padding.
struct MediaEntry
{
    std::unique_ptr<MediaSource> media; // never null
    // Scene: the frame it last rendered in, which stops a second render in the same frame.
    uint64_t lastRenderFrame = 0;
    // Scene: the file stamp the poll compares with the file now. The size is in bytes and the
    // time is the write time in file_time_type ticks. Valid only while fileStamped is set.
    uint64_t fileSize = 0;
    int64_t fileTime = 0;
    project::Rgb background;            // premultiplied background, r, g, b
    float exposure = 0.0f;              // stops; used for Linear sources only
    int32_t offsetX = 0;
    int32_t offsetY = 0;
    int32_t shCap = -1;                 // scene: spherical-harmonics degree cap, -1 = full
    uint32_t sceneGeneration = 0;       // scene: successful reloads of its file
    sabercat::DebugView debugView = sabercat::DebugView::Final; // scene: the view shown
    project::ToneMapper toneMapper = project::ToneMapper::Aces;
    project::SourceRole role = project::SourceRole::None;
    bool visible = false;               // shown in the layout; only ever true when Ready
    bool otherDriveWarning = false;     // the stored path was absolute on another drive
    bool flipUpAxis = false;            // scene: flip the up axis
    bool antialias = false;             // scene: antialiased rendering
    bool fileStamped = false;           // scene: fileSize and fileTime were read from the file
    std::string id;                     // project id: [A-Za-z0-9_-], 1 to 64 characters
    std::string name;                   // display name: the file name, UTF-8
    std::string path;                   // absolute UTF-8 path; empty for in-memory sources
    std::string message;                // error text when not Ready; empty when Ready
    std::string reloadError;            // scene: why the last reload was refused or failed; empty when none

    bool ready() const
    {
        return media != nullptr && media->status() == SourceStatus::Ready;
    }
};

// The view settings that the project file keeps. Pan, zoom and the wipe position live in
// the layout's view (viewLayout()).
struct ViewSettings
{
    double flickerPeriod = 1.0;                     // seconds, > 0
    float diffGain = 1.0f;                          // finite, >= 0
    float depthMin = 0.0f;                          // manual depth range (depthAuto off)
    float depthMax = 1.0f;
    SingleView single = SingleView::A;              // used when !split
    PaneArrangement arrangement = PaneArrangement::Row;
    project::Anchor anchor = project::Anchor::TopLeft;
    bool split = true;                              // Split (true) or Single
    bool nearest = true;
    bool mapUseB = false;                           // Depth and Alpha show B when true
    bool depthAuto = true;                          // depth range from the sources
};

// The transport: the global clock that every visible video follows (see CompareState::advanceTime).
struct Transport
{
    double time = 0.0;       // global seconds, in [0, duration]
    double speed = 1.0;      // 0.25 .. 4
    double duration = 0.0;   // timelineDuration() over the visible Ready videos; 0 when none
    bool loop = false;       // the global loop (wraps the clock at duration)
    bool playing = false;
};

// The layout the composite and the probe use for an input: the SingleScreen form for the
// single-pane modes, otherwise the layout unchanged. It mirrors the rule the compositor
// applies.
LayoutInput compositeLayout(const CompositeInput & input);

class CompareState
{
  public:
    explicit CompareState(GpuContext & context);
    ~CompareState();

    CompareState(const CompareState &) = delete;
    CompareState & operator=(const CompareState &) = delete;

    GpuContext & context() const
    {
        return m_context;
    }

    // The renderer that scene sources borrow the device from. Without one, scene files
    // cannot be added. Set it once, right after the state is created.
    void setSceneHost(const sabercat::Renderer * host);

    // ---- Media list ---------------------------------------------------------------

    const std::vector<MediaEntry> & entries() const
    {
        return m_entries;
    }
    size_t size() const
    {
        return m_entries.size();
    }
    const LayoutInput & layout() const
    {
        return m_layout;
    }
    // The layout's view fields (zoom, pan, wipe position and orientation) may be edited
    // through this reference. The sources must not be: use the operations below. Call
    // viewChanged() after an edit that the project file should record.
    LayoutInput & viewLayout()
    {
        return m_layout;
    }
    // Marks the view as edited (pan, zoom, wipe, Fit, 100%). Not for hover or probe.
    void viewChanged()
    {
        m_dirty = true;
    }

    // Loads a file as a media entry. A scene file (isSceneFile) becomes a scene source.
    // Returns its index, or -1 when refused (at 8 entries, an image larger than the layout
    // limits, no scene host, or the GPU budget for another scene); the status says why. A
    // file that cannot be loaded is still added, with status Error or Missing and its message.
    int32_t addMedia(const std::string & utf8Path);
    // Adds an in-memory source (tests). name is its display name; path is empty unless
    // the source has a file behind it.
    int32_t addMemorySource(std::unique_ptr<MediaSource> media, const std::string & name);
    // Removes an entry. Later entries move down by one. The caller waits for the GPU first
    // when the entry is a scene.
    bool removeSource(size_t index);
    // Shows or hides an entry. Showing refuses a source that is not Ready and a fifth
    // visible source (the status says why). Hiding clears its role.
    bool setVisible(size_t index, bool visible);
    // Gives an entry A or B (the other entry that held it loses it), or None. Refused for a
    // hidden source.
    bool setRole(size_t index, project::SourceRole role);
    // Sets the layout offset of an entry. Refused when out of range. Makes the anchor Custom.
    bool setOffset(size_t index, int32_t x, int32_t y);
    bool setBackground(size_t index, project::Rgb color);
    bool setExposure(size_t index, float stops);
    bool setToneMapper(size_t index, project::ToneMapper toneMapper);
    // Points an entry at another file (a scene file makes a scene source). The entry keeps
    // its id, role and offset. The caller waits for the GPU first when the entry is a scene.
    bool relocate(size_t index, const std::string & utf8Path);
    // TopLeft puts every offset at (0, 0); Centered centers the sources; Custom changes
    // nothing.
    void applyAnchor(project::Anchor anchor);

    // Scene settings of one entry. Each one marks the project and the metrics dirty; the
    // change reaches the scene at its next render. Refused (false, status says why) for
    // an entry that is not a scene.
    bool setShCap(size_t index, int32_t shCap);
    bool setFlipUpAxis(size_t index, bool flip);
    bool setAntialias(size_t index, bool antialias);
    bool setDebugView(size_t index, sabercat::DebugView view);

    // The A and B entries (visible and Ready), or -1.
    int32_t sourceA() const;
    int32_t sourceB() const;
    bool hasPair() const
    {
        return sourceA() >= 0 && sourceB() >= 0;
    }
    // Gives roles to visible, Ready entries that have none: A and B become the first
    // visible entries when they are unset or invalid. Runs after every change.
    void defaultRoles();

    // ---- Frame clock and live updates -------------------------------------------------
    // The live policy runs on a clock the caller sets, so the tests can advance time without
    // sleeping. The UI calls beginFrame() first in every frame.

    // Sets the clock to nowSeconds and counts a frame. The first call also polls the scene files,
    // and later polls run every kScenePollSeconds. A non-finite time is ignored (the clock keeps
    // its last value) so a bad clock cannot stop the policy from ever settling.
    void beginFrame(double nowSeconds);
    // Live updates, on by default. On: scenes re-render each frame while the camera moves, and
    // metrics and FLIP are rate limited. Off: nothing renders or recomputes until the camera rests.
    // Not saved in the project, because it is how the user is working, not what is compared.
    bool live() const
    {
        return m_live;
    }
    void setLive(bool live);
    // The FLIP rate while the camera moves, in Hz. Starts at kDefaultLiveFlipHz.
    float liveFlipHz() const
    {
        return m_liveFlipHz;
    }
    // Refused (false, status says why) for NaN, infinity or a rate outside [kMinLiveFlipHz,
    // kMaxLiveFlipHz]. A change marks the project dirty, because the rate is saved in it.
    bool setLiveFlipHz(float hz);
    // True while the camera moves: an interactive setCamera() within kLiveIdleSeconds of the clock.
    // With live on, scenes render every frame while this is true; metrics and FLIP run at
    // liveFlipHz() instead. With live off, scenes and metrics wait until it clears.
    bool cameraMoving() const;
    // True when metrics or FLIP are stale only because the live policy deferred them. They are
    // computed when the camera rests, or at the next rate tick while it moves. The UI shows this.
    bool metricsPending() const
    {
        return m_liveDeferred;
    }
    // True when nothing is left for the current view: the camera is at rest, no metrics are
    // deferred, no render size waits, and every visible Ready scene is clean. A test or a smoke run
    // waits on this before it reads the final numbers.
    bool liveSettled() const;
    // True while a render size change waits for the camera to rest (see setRenderSize()).
    bool renderSizePending() const
    {
        return m_pendingW != 0;
    }
    // Counters for the tests and the smoke run. They count from startup, so the initial load is in
    // them. sceneRenderTotal() sums SceneSource::renderCount over the scene entries.
    uint64_t sceneRenderTotal() const;
    uint64_t flipComputeCount() const
    {
        return m_flipComputes;
    }
    uint64_t metricsComputeCount() const
    {
        return m_metricsComputes;
    }
    uint64_t sceneReloadCount() const
    {
        return m_sceneReloads;
    }
    uint64_t sceneReloadAttempts() const
    {
        return m_sceneReloadAttempts;
    }
    // Reloads a scene entry from its file, keeping its settings, role, visibility and offset. False
    // when the entry is not a Ready scene or the reload failed or was refused over the GPU budget.
    // A refusal or failure sets the entry's reloadError and the old render stays on screen.
    bool reloadScene(size_t index);
    // Checks each Ready scene's file against its stored stamp and reloads the ones that changed.
    // beginFrame() runs it once a second; it is public so the tests can run it directly. The new
    // stamp is stored before the reload, so a file that fails to load is tried once per change and
    // not on every poll.
    void pollSceneFiles();

    // ---- Scenes and the shared camera -----------------------------------------------

    const app::OrbitCamera & camera() const
    {
        return m_camera;
    }
    // Replaces the camera. Every visible scene re-renders at the next updateScenes(). An
    // interactive change (a drag, the wheel, a roll) is stamped on the frame clock, which is what
    // makes the camera count as moving. Any other change is immediate, as a framing or a project
    // load is. Refused (false, the camera is kept, status says why) when the pivot, distance or
    // orientation is not finite, the distance is not above zero, or the orientation has length 0.
    bool setCamera(const app::OrbitCamera & camera, bool interactive = false);
    uint64_t cameraGeneration() const
    {
        return m_cameraGeneration;
    }
    // Frames the camera on a Ready scene (its bounds at the render height). False when the
    // entry is not a loaded scene.
    bool frameScene(size_t index);
    // The render size of every scene source, in pixels. Refused (false) outside 1 to
    // kMaxSourceSize per side. Scenes resize and re-render. While the camera moves a different
    // size is kept pending (renderSizePending()) and applied at the next updateScenes() after the
    // camera rests, so a drag never reallocates the scenes; the call still returns true. A larger
    // size over the GPU budget is refused on apply, and the status says why.
    bool setRenderSize(uint32_t width, uint32_t height);
    uint32_t renderWidth() const
    {
        return m_renderWidth;
    }
    uint32_t renderHeight() const
    {
        return m_renderHeight;
    }
    // Renders every visible scene that is dirty for the current camera generation. A render
    // failure hides that scene and sets its message; the other sources keep working. Call
    // before the metrics and the composite of a frame. A scene renders at most once per frame
    // after beginFrame(), because the UI calls this twice a frame. While the camera moves with
    // live off, nothing renders; a pending render size is applied once the camera rests.
    void updateScenes();

    // ---- View ---------------------------------------------------------------------

    const ViewSettings & settings() const
    {
        return m_settings;
    }
    void setSplit(bool split);
    // The divider mode: Split with exactly two visible sources, A and B both set and different.
    // Then the split view is one pane with a vertical wipe at the divider (buildComposite()
    // says so), and the layout is SingleScreen. With one, three or four sources Split keeps
    // its panes.
    bool dividerActive() const;
    // The split divider as a fraction of the window width (see ViewState::dividerFraction).
    double dividerFraction() const
    {
        return m_layout.view.dividerFraction;
    }
    // Clamps to [0,1]; NaN sets the default (kDefaultSplitDivider). Marks the project dirty only
    // when the value changes.
    void setDividerFraction(double fraction);
    void setSingleView(SingleView view);
    void setArrangement(PaneArrangement arrangement);
    void setFlickerPeriod(double seconds);
    void setDiffGain(float gain);
    void setNearest(bool nearest);
    void setWipeOrientation(WipeOrientation orientation);
    // Depth and Alpha: show source B when true, else A.
    void setMapUseB(bool useB);
    void setDepthAuto(bool depthAuto);
    // The manual depth range. Ignored unless both are finite and min < max.
    void setDepthRange(float min, float max);
    // The depth range the depth maps use: the manual range, or with depthAuto the union of
    // the visible Ready A and B depth ranges (0..1 when neither has depth).
    void resolvedDepthRange(float & lo, float & hi) const;
    // The window size of the view area, in pixels. The UI calls this each frame before
    // any view operation.
    void setViewSize(int32_t width, int32_t height);
    int32_t viewWidth() const
    {
        return m_layout.windowWidth;
    }
    int32_t viewHeight() const
    {
        return m_layout.windowHeight;
    }

    // The mode the composite shows: Panes for Split, and for Single A/B; the single-pane
    // modes need their sources and fall back to Panes without them.
    CompositeMode compositeMode() const;
    // Whether the flicker shows B at time t: fmod(t, period) >= period / 2.
    static bool flickerShowB(double timeSeconds, double period);
    // The input for the compositor at the current view size and time t (seconds).
    CompositeInput buildComposite(double timeSeconds) const;

    // ---- Metrics ------------------------------------------------------------------

    // Recomputes A against B over their overlap when a change made the cache stale. Does
    // nothing otherwise.
    void updateMetrics(MetricsGpu & metrics);
    const std::optional<MetricsResult> & metrics() const
    {
        return m_metrics;
    }
    // Why there are no metrics (empty when there are).
    const std::string & metricsReason() const
    {
        return m_metricsReason;
    }

    // ---- FLIP ---------------------------------------------------------------------

    // The FLIP ppd, in [kMinFlipPpd, kMaxFlipPpd]. Starts at defaultFlipPpd().
    float flipPpd() const
    {
        return m_flipPpd;
    }
    // Sets the ppd. Refused (false, status says why) for NaN or a value outside
    // [kMinFlipPpd, kMaxFlipPpd]. A new value makes only the FLIP result stale; the metrics
    // stay as they are until the next updateMetrics().
    bool setFlipPpd(float ppd);
    // The FLIP result over the A/B overlap, computed by updateMetrics() when stale. Empty
    // when there is none (see flipReason()).
    const std::optional<FlipGpuResult> & flip() const
    {
        return m_flip;
    }
    // Why there is no FLIP result (empty when there is one).
    const std::string & flipReason() const
    {
        return m_flipReason;
    }
    // The device error map of flip(), for the compositor. Invalid when there is no flip().
    BufferView flipMapView() const;

    // ---- HDR-FLIP -----------------------------------------------------------------
    // The FLIP mode. Ldr by default. flip() is the active result, LDR or HDR.
    project::FlipMode flipMode() const
    {
        return m_flipMode;
    }
    // A and B are both visible and Ready, and at least one is Encoding::Linear.
    bool hdrAvailable() const;
    // Hdr is refused (false, setStatus("HDR-FLIP needs a Linear (EXR) source in the A/B pair")) unless
    // hdrAvailable(). Ldr always allowed. Same mode -> true, nothing stale. A change marks the project
    // dirty and the FLIP result stale (cancels a job).
    bool setFlipMode(project::FlipMode mode);
    const FlipHdrSettings & hdrSettings() const
    {
        return m_hdrSettings;
    }
    // Stales FLIP when changed.
    bool setHdrToneMapper(ToneMapper toneMapper);
    // Overrides (nullopt = auto). Refused (false, setStatus with the reason, nothing changes) for a
    // non-finite value, a value outside [-126,126], start > stop when both are set, count outside
    // kMinHdrExposures..kMaxHdrExposures. Stales FLIP when changed.
    bool setHdrRange(std::optional<float> start, std::optional<float> stop, std::optional<int32_t> count);
    // "Run on change", default false. Not saved in the project.
    bool hdrAuto() const
    {
        return m_hdrAuto;
    }
    void setHdrAuto(bool on);
    // Starts an HDR-FLIP job for the current pair (mode must be Hdr). Cancels a running job first.
    // Returns false with flipReason() set (and no job) when it cannot start: not Hdr mode, no pair/overlap
    // (the same reasons as metrics), HDR not available, a source not ready, or beginHdr threw / returned
    // nullopt (reason = the exception text, or "A and B have no pixels in common"). Drops the current flip().
    bool requestHdr();
    bool hdrRunning() const
    {
        return m_hdrRunning;
    }
    // done / count of the running job, 0 when not running.
    float hdrProgress() const;
    // Ends a running job; flipReason() = "HDR-FLIP cancelled". Safe with no job.
    void cancelHdr();
    // Advances the running job for about budgetSeconds (at least one exposure per call): steps
    // FlipGpu::stepHdr with a chunk size adapted from the measured time per exposure (start with 1). On
    // completion sets flip() (FlipGpuResult with hdr) and returns true; on an exception sets flipReason()
    // to the text and ends the job (returns true). Returns false while still running or when nothing runs.
    bool stepHdr(double budgetSeconds);
    // requestHdr() if nothing runs and no result, then step with an unlimited budget until done.
    void runHdrBlocking();
    // Empty when the pair does not mix encodings; otherwise a one-line warning saying how a Display and
    // a Linear source are treated in LDR and HDR.
    std::string pairEncodingWarning() const;

    // ---- Video and the transport ----------------------------------------------------

    const Transport & transport() const
    {
        return m_transport;
    }
    // True when at least one entry is a video (Ready or not): the UI shows the transport bar.
    bool hasVideo() const;
    // True when at least one VISIBLE Ready video exists (duration > 0 is possible).
    bool hasVisibleVideo() const;
    // Refused (the status says why) when there is no visible video or the duration is 0. At the
    // end without loop it restarts at 0.
    void play();
    void pause();
    void togglePlay();
    // Clamps to [0, duration]; non-finite refused (false). Marks the project dirty; the videos
    // follow at the next updateVideos().
    bool seek(double seconds);
    // Presets are 0.25, 0.5, 1, 2, 4 but any finite value in [0.25, 4] is accepted; else refused
    // (false, status says why).
    bool setSpeed(double speed);
    void setTransportLoop(bool loop);
    // Step n frames (negative = back) of the reference video: the video of source A when A is a
    // video, else B, else the first visible video. Pauses. Lands exactly on a frame boundary:
    // source frame f = frameIndexAt(ref, time), target f2 = clamp(f + n, 0, frameCount - 1),
    // global time = f2 / fps - timeOffset(ref), then clamped to [0, duration].
    void stepFrames(int32_t n);
    // Advances the clock by dtSeconds * speed when playing. dt is clamped to [0, 0.25] first, so a
    // stall or a minimize never jumps. At the end without loop the clock stops at duration and
    // playing becomes false. Call once per presented frame, before updateVideos().
    void advanceTime(double dtSeconds);
    // Updates every visible Ready video to the clock. When any frame changed: the metrics and FLIP
    // become stale, and HDR-FLIP is never run by a frame change. Returns true when a frame changed.
    // A video whose update failed gets its message copied to the entry while the entry stays Ready.
    bool updateVideos();
    // Per-video settings. All refuse (false, status says why) for an entry that is not a Ready
    // video, or on invalid values (see VideoSource). Each marks the project dirty and the metrics
    // stale and recomputes the transport duration (the clock is clamped to it).
    bool setVideoTrim(size_t index, double trimIn, double trimOut); // trimOut < 0 = end of the clip
    bool setVideoLoop(size_t index, bool loop);
    bool setVideoTimeOffset(size_t index, double seconds);
    // Frame offset: seconds = frames / fps (of that video); same checks.
    bool setVideoTimeOffsetFrames(size_t index, int32_t frames);
    // The VideoSource of an entry when it is a Ready video, else null.
    VideoSource * videoOf(size_t index);
    const VideoSource * videoOf(size_t index) const;
    // Frame index shown by a video entry, -1 for a non-video or a video that is not Ready.
    int64_t videoFrameIndex(size_t index) const;
    // "HDR-FLIP not run during playback" state: true while the mode is Hdr, a visible A or B is a
    // video, and the last frame change happened after the last explicit HDR run.
    bool hdrStaleForVideo() const;
    // Incremented each time a frame changes (for the UI and tests).
    uint64_t frameGeneration() const
    {
        return m_frameGeneration;
    }

    // ---- Project ------------------------------------------------------------------

    // The project as it would be saved. Sources without a path (in-memory) are left out.
    project::CompareProject toProject() const;
    // Replaces the state with a project. Missing files become Missing entries.
    void fromProject(const project::CompareProject & project);
    // Clears everything: no sources, default view and camera, no project file.
    void newProject();
    // Replaces the state with the project at utf8Path. On failure the state is unchanged
    // and error holds the reason (a newer formatVersion is refused here).
    bool openProject(const std::string & utf8Path, std::string & error);
    // Saves to utf8Path and makes it the project file. On failure error holds the reason.
    bool saveProject(const std::string & utf8Path, std::string & error);
    const std::string & projectPath() const
    {
        return m_projectPath;
    }
    bool projectDirty() const
    {
        return m_dirty;
    }
    // Marks the project changed (for edits made outside these operations).
    void markDirty()
    {
        m_dirty = true;
    }
    // Warnings from the last open (unknown keys, cleared roles, other-drive paths).
    const std::vector<std::string> & notices() const
    {
        return m_notices;
    }

    // ---- Status -------------------------------------------------------------------

    const std::string & status() const
    {
        return m_status;
    }
    void setStatus(const std::string & text)
    {
        m_status = text;
    }

  private:
    int32_t addEntry(std::unique_ptr<MediaSource> media, const std::string & name, const std::string & path);
    int32_t addScene(const std::string & utf8Path);
    std::unique_ptr<MediaSource> makeScene(const std::string & utf8Path, const SceneSettings & settings,
                                           bool applyAntialias);
    bool sceneBudgetAllows();
    // The Ready scenes and the device-local bytes they hold (each scene's own renderer).
    uint64_t readySceneBytes(size_t & readyScenes) const;
    // Applies a render size now, or refuses it over the GPU budget. Clears any pending size.
    bool applyRenderSize(uint32_t width, uint32_t height);
    void ensureRenderSize();
    void frameCamera(const SceneSource & scene);
    void frameFirstScene(const SceneSource & scene);
    void restoreCamera(const project::CompareCamera & camera);
    void hideScene(MediaEntry & entry, const std::string & message);
    bool sceneFailed(MediaEntry & entry);
    MediaEntry * sceneEntry(size_t index);
    void sceneEdited();
    // Marks the metrics and the FLIP result stale (and cancels a running HDR-FLIP job).
    void markMetricsStale();
    // Marks only the FLIP result stale (and cancels a running HDR-FLIP job).
    void markFlipStale();
    // The Ready video of an entry, or null (the status says why when the entry is a video that is
    // not Ready, and stays silent for an entry that is not a video).
    VideoSource * readyVideo(size_t index);
    // Recomputes the transport duration from the visible Ready videos and clamps the clock to it.
    // No visible video: duration 0, time 0, not playing.
    void refreshTransport();
    // A video setting changed: the project is dirty, the metrics are stale and the duration is
    // recomputed.
    void videoEdited();
    // The video the frame stepping reads: A, else B, else the first visible Ready video (or null).
    VideoSource * referenceVideo();
    // The A/B pair and its overlap. Returns the reason there is no pair with an overlap (empty when
    // there is one); a, b and overlap are set then.
    std::string pairState(int32_t & a, int32_t & b, PixelRect & overlap) const;
    // Recomputes the FLIP result from the pair, or sets its reason. pairReason is the
    // metrics' reason for no pair or no overlap (empty when there is one); a, b and overlap
    // are then valid.
    void updateFlip(const std::string & pairReason, int32_t a, int32_t b, const PixelRect & overlap);
    void syncLayout();
    void refreshView();
    void changed();
    int32_t resolveAb(int32_t preferred) const;
    int32_t indexOfRole(project::SourceRole role) const;
    std::string nextId() const;

    GpuContext & m_context;
    const sabercat::Renderer * m_sceneHost = nullptr;
    app::OrbitCamera m_camera;
    std::vector<MediaEntry> m_entries;
    LayoutInput m_layout;
    ViewSettings m_settings;
    Transport m_transport;
    std::optional<MetricsResult> m_metrics;
    std::string m_metricsReason;
    std::unique_ptr<FlipGpu> m_flipGpu;  // created on the first FLIP; null until then
    std::optional<FlipGpuResult> m_flip;
    std::string m_flipReason;
    FlipHdrSettings m_hdrSettings;      // HDR-FLIP tone mapper and exposure overrides
    std::string m_projectPath;
    std::vector<std::string> m_notices;
    std::string m_status;
    uint64_t m_cameraGeneration = 0;
    uint64_t m_frameGeneration = 0;     // incremented on each video frame change
    uint64_t m_lastSceneBytes = 0;      // device-local bytes the last scene added
    // The frame clock and the live counters (see beginFrame()). The times are in seconds on the
    // caller's clock. m_lastLiveCompute starts far in the past so the first tick computes at once.
    double m_now = 0.0;
    double m_interactiveAt = -1.0;      // clock time of the last interactive camera move; -1 = none
    double m_lastLiveCompute = -1.0e30; // clock time of the last metrics/FLIP tick while moving
    double m_lastPoll = -1.0e9;         // clock time of the last scene file poll
    uint64_t m_frame = 0;               // frames begun by beginFrame(); 0 before the first
    uint64_t m_flipComputes = 0;        // LDR GPU FLIP computes so far
    uint64_t m_metricsComputes = 0;     // metrics computes so far
    uint64_t m_sceneReloads = 0;        // successful scene reloads
    uint64_t m_sceneReloadAttempts = 0; // reload attempts, successful or not
    uint32_t m_renderWidth = 0;         // 0 until the first scene (or a restored camera)
    uint32_t m_renderHeight = 0;
    uint32_t m_pendingW = 0;            // a render size waiting for the camera to rest; 0 = none
    uint32_t m_pendingH = 0;
    float m_flipPpd = defaultFlipPpd();
    float m_liveFlipHz = kDefaultLiveFlipHz; // the FLIP rate while the camera moves
    float m_hdrSecondsPerExposure = 0.0f; // measured by the running HDR-FLIP job; 0 = none yet
    project::FlipMode m_flipMode = project::FlipMode::Ldr;
    bool m_metricsDirty = true;
    bool m_flipDirty = true;
    bool m_dirty = false;
    bool m_cameraFramed = false;        // the camera was framed on a scene or restored
    bool m_hdrAuto = false;             // HDR-FLIP runs on every change
    bool m_hdrRunning = false;          // an HDR-FLIP job is active in m_flipGpu
    bool m_flipFromFrame = false;       // the pending FLIP staleness came from a video frame change
    bool m_hdrVideoStale = false;       // a video frame changed since the last explicit HDR-FLIP run
    bool m_live = true;                 // live updates (see setLive())
    bool m_liveDeferred = false;        // metrics or FLIP were deferred by the live policy
};

} // namespace sabercat::compare::gui
