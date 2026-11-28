// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "app.h"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <imgui_internal.h>

#include "about_logo.h"
#include "app_util.h"
#include "borrowed_vulkan_host.h"
#include "compare_state.h"
#include "compare_ui.h"
#include "compositor.h"
#include "file_dialog.h"
#include "flip_hdr.h"
#include "gpu_context.h"
#include "imgui_layer.h"
#include "metrics_gpu.h"
#include "orbit_camera.h"
#include "shader_dir.h"
#include "system_info.h"
#include "video_source.h"
#include "window_icon.h"

#include "sabercat/logger.h"
#include "sabercat/renderer.h"
#include "sabercat/version.h"
#include "sabercat/vk_check.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace sabercat::compare::gui
{

namespace
{

namespace fs = std::filesystem;

constexpr int kWindowWidth = 1400;
constexpr int kWindowHeight = 900;
// The swapchain's starting size, as the Studio's (the surface size decides after that).
constexpr int kSwapchainWidth = 1280;
constexpr int kSwapchainHeight = 720;

// The test hook (SABERCAT_COMPARE_SMOKE_DIR): frame numbers of the resize sequence, and
// the frame after which the exports run.
constexpr int kSmokeFrames = 20;
constexpr int kSmokeResizeFrame = 5;
constexpr int kSmokeTinyFrame = 8;
constexpr int kSmokeIconifyFrame = 9;
constexpr int kSmokeRestoreFrame = 12;

// The playback test (SABERCAT_COMPARE_SMOKE_PLAY=<N>, with the smoke directory): the clock starts at
// frame 2 and advances 1/30 s per presented frame, and the exports run after N frames more. N defaults
// to 20 when the value is empty, zero or not a number.
constexpr int kSmokePlayFrames = 20;
constexpr double kSmokePlayFps = 30.0;
constexpr int kSmokePlayStartFrame = 2;

// The frame count of the playback test for the value of SABERCAT_COMPARE_SMOKE_PLAY. Zero when the
// variable is not set.
int smokePlayFrames(const char * text)
{
    if (text == nullptr)
    {
        return 0;
    }
    char * end = nullptr;
    const long value = std::strtol(text, &end, 10);
    const bool valid = end != text && *end == '\0' && value > 0 && value <= 1000000;
    return valid ? static_cast<int>(value) : kSmokePlayFrames;
}

// ---- The ID scan (SABERCAT_COMPARE_SMOKE_IDSCAN=1, with the smoke directory) -----------
// ImGui reports a duplicate ID only for the item under the pointer, so the scan moves the pointer
// over the UI and watches for the report. Each configuration of the view is swept: the side panel
// and the bottom band (the transport bar), one pointer position per frame. Then each media entry's
// settings popup is swept over the window's width. The scan never clicks, so it changes no state
// except the configuration it sets.

// One configuration of the view. A split configuration shows `sources` visible sources (the first
// ready ones, the first two as A and B); the others show A and B.
struct ScanConfig
{
    const char * name;
    int32_t sources;
    bool split;
    SingleView view;
    WipeOrientation orientation;
    bool hdr; // the FLIP mode is HDR; the configuration is skipped without a Linear pair
};

constexpr std::array<ScanConfig, 14> kScanConfigs = {{
    {"split2", 2, true, SingleView::A, WipeOrientation::Vertical, false},
    {"split3", 3, true, SingleView::A, WipeOrientation::Vertical, false},
    {"split4", 4, true, SingleView::A, WipeOrientation::Vertical, false},
    {"singleA", 2, false, SingleView::A, WipeOrientation::Vertical, false},
    {"wipeVertical", 2, false, SingleView::Wipe, WipeOrientation::Vertical, false},
    {"wipeHorizontal", 2, false, SingleView::Wipe, WipeOrientation::Horizontal, false},
    {"flicker", 2, false, SingleView::Flicker, WipeOrientation::Vertical, false},
    {"diff", 2, false, SingleView::Diff, WipeOrientation::Vertical, false},
    {"depth", 2, false, SingleView::Depth, WipeOrientation::Vertical, false},
    {"alpha", 2, false, SingleView::Alpha, WipeOrientation::Vertical, false},
    {"depthDiff", 2, false, SingleView::DepthDiff, WipeOrientation::Vertical, false},
    {"alphaDiff", 2, false, SingleView::AlphaDiff, WipeOrientation::Vertical, false},
    {"flip", 2, false, SingleView::Flip, WipeOrientation::Vertical, false},
    {"hdrFlip", 2, false, SingleView::Flip, WipeOrientation::Vertical, true},
}};

// Shows the first `sources` ready sources as the configuration's view. False when the configuration
// cannot be shown: too few ready sources, or HDR without a Linear pair in A and B.
bool applyScanConfig(CompareState & state, const ScanConfig & config)
{
    std::vector<size_t> ready;
    for (size_t i = 0; i < state.size(); ++i)
    {
        if (state.entries()[i].ready())
        {
            ready.push_back(i);
        }
    }
    if (ready.size() < static_cast<size_t>(config.sources))
    {
        return false;
    }
    for (size_t i = 0; i < state.size(); ++i)
    {
        state.setVisible(i, false);
    }
    for (int32_t k = 0; k < config.sources; ++k)
    {
        state.setVisible(ready[static_cast<size_t>(k)], true);
    }
    state.setRole(ready[0], project::SourceRole::A);
    state.setRole(ready[1], project::SourceRole::B);
    state.setFlipMode(project::FlipMode::Ldr);
    state.setSplit(config.split);
    if (!config.split)
    {
        state.setSingleView(config.view);
        if (config.view == SingleView::Wipe)
        {
            state.setWipeOrientation(config.orientation);
        }
    }
    return !config.hdr || state.setFlipMode(project::FlipMode::Hdr);
}

// The pointer positions of a configuration's sweep: the side panel (x in [8, 340), y from 20 to the
// window's bottom, in steps of 28 and 10), then the bottom band of 110 px over the whole width.
std::vector<ImVec2> sweepPositions(ImVec2 display)
{
    std::vector<ImVec2> positions;
    for (float y = 20.0f; y < display.y; y += 10.0f)
    {
        for (float x = 8.0f; x < 340.0f; x += 28.0f)
        {
            positions.emplace_back(x, y);
        }
    }
    for (float y = std::max(0.0f, display.y - 110.0f); y < display.y; y += 10.0f)
    {
        for (float x = 0.0f; x < display.x; x += 28.0f)
        {
            positions.emplace_back(x, y);
        }
    }
    return positions;
}

// The pointer positions of an entry's popup sweep: x over [8, 760) in steps of 28, y over the window
// in steps of 20 (the popup is one window, so a coarser y keeps the run short).
std::vector<ImVec2> popupPositions(ImVec2 display)
{
    std::vector<ImVec2> positions;
    for (float y = 20.0f; y < display.y; y += 20.0f)
    {
        for (float x = 8.0f; x < 760.0f; x += 28.0f)
        {
            positions.emplace_back(x, y);
        }
    }
    return positions;
}

class IdScan
{
  public:
    // A sweep that hovers no more distinct items than this is invalid: the pointer did not reach the UI.
    static constexpr size_t kMinDistinctHovered = 20;

    explicit IdScan(GLFWwindow * window) : m_window(window) {}

    // Before the frame's beginFrame(): sets this frame's configuration and queues the pointer
    // position (ImGui reads it in NewFrame). False when the scan is over; the caller then calls finish().
    bool prepare(CompareState & state, CompareUi & ui);
    // Before beginFrame(): keeps the previous frame's hover data, which NewFrame resets.
    void snapshot(const ImGuiContext & ctx);
    // After beginFrame(): ImGui reports a duplicate ID for the item hovered in the previous frame.
    void checkConflicts(const ImGuiContext & ctx);
    // After the frame's UI is built: counts the item under the pointer.
    void recordHover(const ImGuiContext & ctx);
    // Prints the summary and returns the exit code: 1 on a conflict or an invalid scan.
    int finish();

  private:
    struct Phase
    {
        bool popup = false;
        size_t index = 0; // the configuration, or the entry of the popup
    };

    void startPhase(const Phase & phase, CompareState & state, CompareUi & ui);
    void invalidate(const std::string & reason);

    GLFWwindow * m_window = nullptr;
    std::vector<Phase> m_phases;
    std::vector<ImVec2> m_positions;
    std::set<ImGuiID> m_hovered;
    std::string m_phaseName;
    std::string m_snapPhase;
    std::string m_snapWindow;
    std::string m_invalidReason;
    float m_x = 0.0f;
    float m_y = 0.0f;
    float m_snapX = 0.0f;
    float m_snapY = 0.0f;
    size_t m_phase = 0;
    size_t m_position = 0;
    size_t m_snapItems = 0;
    size_t m_conflicts = 0;
    size_t m_configs = 0;
    size_t m_popups = 0;
    size_t m_frames = 0;
    bool m_started = false;
    bool m_done = false;
    bool m_invalid = false;
};

void IdScan::invalidate(const std::string & reason)
{
    m_invalid = true;
    m_invalidReason = reason;
    m_done = true;
}

// Starts a phase: applies its state, opens its popup, and fills the pointer positions. A phase that
// cannot be shown leaves the positions empty, so prepare() moves on to the next one.
void IdScan::startPhase(const Phase & phase, CompareState & state, CompareUi & ui)
{
    const ImGuiIO & io = ImGui::GetIO();
    if (phase.popup)
    {
        // The popups are swept over the split pair, as the first configuration shows them.
        const std::string name = "settings:" + state.entries()[phase.index].name;
        if (!applyScanConfig(state, kScanConfigs[0]))
        {
            std::cout << "idscan skip " << name << '\n';
            return;
        }
        ui.smokeOpenSettings(static_cast<int32_t>(phase.index));
        m_phaseName = name;
        m_positions = popupPositions(io.DisplaySize);
        ++m_popups;
    }
    else
    {
        const ScanConfig & config = kScanConfigs[phase.index];
        // Closes the popup the previous sweep left open, then shows the configuration.
        ui.smokeOpenSettings(-1);
        if (!applyScanConfig(state, config))
        {
            std::cout << "idscan skip config=" << config.name << '\n';
            return;
        }
        m_phaseName = config.name;
        m_positions = sweepPositions(io.DisplaySize);
        ++m_configs;
    }
    std::cout << "idscan phase=" << m_phaseName << " positions=" << m_positions.size() << '\n';
}

bool IdScan::prepare(CompareState & state, CompareUi & ui)
{
    if (m_done)
    {
        return false;
    }
    ImGuiIO & io = ImGui::GetIO();
    if (!m_started)
    {
        m_started = true;
        // Each line is flushed as it is written, so a crash in the sweep still leaves the lines before it.
        std::cout << std::unitbuf;
        std::cout << "idscan window focused=" << glfwGetWindowAttrib(m_window, GLFW_FOCUSED) << '\n';
        size_t ready = 0;
        for (const MediaEntry & entry : state.entries())
        {
            ready += entry.ready() ? 1 : 0;
        }
        if (ready < 2)
        {
            invalidate("the sweep needs 2 ready sources, found " + std::to_string(ready));
            return false;
        }
        for (size_t i = 0; i < kScanConfigs.size(); ++i)
        {
            m_phases.push_back(Phase{false, i});
        }
        for (size_t i = 0; i < state.size(); ++i)
        {
            m_phases.push_back(Phase{true, i});
        }
    }
    // ImGui learns the display size in the frame's NewFrame, so the first frame has none yet: the sweep
    // waits for it (a phase sized from an empty display would be skipped whole).
    if (io.DisplaySize.x <= 0.0f || io.DisplaySize.y <= 0.0f)
    {
        return true;
    }
    while (m_position >= m_positions.size())
    {
        if (m_phase >= m_phases.size())
        {
            m_done = true;
            ui.smokeOpenSettings(-1);
            return false;
        }
        m_positions.clear();
        startPhase(m_phases[m_phase++], state, ui);
        m_position = 0;
    }
    const ImVec2 at = m_positions[m_position++];
    // The GLFW backend takes the pointer from the OS cursor while the window is focused, and that
    // would undo an injected position. So the OS cursor moves there too (the real pointer moves
    // during the scan; it clicks nothing).
    if (glfwGetWindowAttrib(m_window, GLFW_FOCUSED) != 0)
    {
        glfwSetCursorPos(m_window, static_cast<double>(at.x), static_cast<double>(at.y));
    }
    io.AddMousePosEvent(at.x, at.y);
    m_x = at.x;
    m_y = at.y;
    ++m_frames;
    return true;
}

void IdScan::snapshot(const ImGuiContext & ctx)
{
    // The count and the hovered window belong to the previous frame: NewFrame resets both.
    m_snapItems = static_cast<size_t>(ctx.HoveredIdPreviousFrameItemCount);
    m_snapWindow = ctx.HoveredWindow != nullptr ? ctx.HoveredWindow->Name : "none";
    m_snapPhase = m_phaseName;
    m_snapX = m_x;
    m_snapY = m_y;
}

void IdScan::checkConflicts(const ImGuiContext & ctx)
{
    if (ctx.DebugDrawIdConflicts == 0)
    {
        return;
    }
    ++m_conflicts;
    std::cout << "idscan conflict config=" << m_snapPhase << " mouse=(" << static_cast<int>(m_snapX) << ","
              << static_cast<int>(m_snapY) << ") hoveredWindow=" << m_snapWindow << " items=" << m_snapItems << '\n';
}

void IdScan::recordHover(const ImGuiContext & ctx)
{
    if (ctx.HoveredId != 0)
    {
        m_hovered.insert(ctx.HoveredId);
    }
}

int IdScan::finish()
{
    std::cout << "idscan frames=" << m_frames << " configs=" << m_configs << " conflicts=" << m_conflicts
              << " distinctHovered=" << m_hovered.size() << '\n';
    std::cout << "idscan popups=" << m_popups << '\n';
    if (m_invalid)
    {
        std::cout << "idscan invalid: " << m_invalidReason << '\n';
        return 1;
    }
    if (m_hovered.size() <= IdScan::kMinDistinctHovered)
    {
        std::cout << "idscan invalid: the pointer hovered " << m_hovered.size() << " distinct items, more than "
                  << IdScan::kMinDistinctHovered << " are needed\n";
        return 1;
    }
    return m_conflicts > 0 ? 1 : 0;
}

// The live test (SABERCAT_COMPARE_SMOKE_LIVE=<N>, with the smoke directory): a camera drag of N frames from
// kSmokeLiveStartFrame, timed on the real clock. The loop then runs until the state has settled, or
// kSmokeLiveSettleFrames more frames; a run that does not settle fails. N defaults to kSmokeLiveDefaultFrames
// when the value is empty or invalid.
constexpr int kSmokeLiveStartFrame = 3;
constexpr int kSmokeLiveDefaultFrames = 30;
constexpr int kSmokeLiveMaxFrames = 100000;
constexpr int kSmokeLiveSettleFrames = 600;

// The frame count of the live test for the value of SABERCAT_COMPARE_SMOKE_LIVE. Zero when the variable
// is not set.
int smokeLiveFrames(const char * text)
{
    if (text == nullptr)
    {
        return 0;
    }
    char * end = nullptr;
    const long value = std::strtol(text, &end, 10);
    const bool valid = end != text && *end == '\0' && value > 0 && value <= kSmokeLiveMaxFrames;
    return valid ? static_cast<int>(value) : kSmokeLiveDefaultFrames;
}

// Paths dropped onto the window, taken by the main loop. GLFW calls onDrop on the main
// thread from glfwPollEvents.
struct WindowHooks
{
    std::vector<std::string> dropped;
};

void onDrop(GLFWwindow * window, int count, const char ** paths)
{
    auto * hooks = static_cast<WindowHooks *>(glfwGetWindowUserPointer(window));
    if (hooks == nullptr)
    {
        return;
    }
    for (int i = 0; i < count; ++i)
    {
        hooks->dropped.emplace_back(paths[i]);
    }
}

// A failure that means the GPU is gone (the loop ends rather than retrying every frame).
bool isDeviceLost(const std::string & what)
{
    return what.find("DEVICE_LOST") != std::string::npos || what.find("device lost") != std::string::npos;
}

// A UTF-8 string as a path.
fs::path pathFromUtf8(const std::string & utf8)
{
    return fs::path(std::u8string(reinterpret_cast<const char8_t *>(utf8.data()), utf8.size()));
}

// directory / name, as UTF-8.
std::string pathJoin(const std::string & directory, const std::string & name)
{
    const std::u8string text = (pathFromUtf8(directory) / pathFromUtf8(name)).u8string();
    return std::string(reinterpret_cast<const char *>(text.data()), text.size());
}

// Drives the resize and minimize sequence of the smoke test (SABERCAT_COMPARE_SMOKE_RESIZE=1).
void smokeWindowSequence(GLFWwindow * window, int frame)
{
    if (frame == kSmokeResizeFrame)
    {
        glfwSetWindowSize(window, 1000, 640);
    }
    else if (frame == kSmokeTinyFrame)
    {
        glfwSetWindowSize(window, 1, 1);
    }
    else if (frame == kSmokeIconifyFrame)
    {
        glfwIconifyWindow(window);
    }
    else if (frame == kSmokeRestoreFrame)
    {
        glfwRestoreWindow(window);
        glfwSetWindowSize(window, 1200, 760);
    }
}

// Drives the HDR mode of the smoke test (SABERCAT_COMPARE_SMOKE_HDR). "toggle" sets HDR mode
// and runs HDR-FLIP at frame 3, sets LDR at frame 10 and sets HDR mode with a run at frame 14.
// Any other value sets HDR mode at frame 2. A refusal goes to stderr; the exports decide the exit code.
void smokeHdrSequence(CompareState & state, int frame, const std::string & mode)
{
    const auto setMode = [&state](project::FlipMode flipMode, bool run)
    {
        if (!state.setFlipMode(flipMode))
        {
            std::cerr << "sabercat_compare: smoke HDR mode refused: " << state.status() << '\n';
            return;
        }
        if (run && !state.requestHdr())
        {
            std::cerr << "sabercat_compare: smoke HDR run refused: " << state.flipReason() << '\n';
        }
    };
    if (mode == "toggle")
    {
        if (frame == 3)
        {
            setMode(project::FlipMode::Hdr, true);
        }
        else if (frame == 10)
        {
            setMode(project::FlipMode::Ldr, false);
        }
        else if (frame == 14)
        {
            setMode(project::FlipMode::Hdr, true);
        }
    }
    else if (frame == 2)
    {
        setMode(project::FlipMode::Hdr, false);
    }
}

// The live test's numbers, written as the live line of smoke.txt. frames is 0 when the test did not run.
struct LiveSmoke
{
    double seconds = 0.0;  // the drag's wall-clock time, from its first frame to the frame after its last
    int frames = 0;        // the drag's frame count
    bool settled = false;  // whether the state had settled when the exports ran
};

// Runs the exports through the same functions the menus use, then writes smoke.txt. Returns
// the exit code: 0 when every export and every startup source succeeded. With expectHdr, an
// HDR-FLIP result must exist too. With expectVideo (the playback test), the transport and each
// visible video's frame are written, and a run with no visible video fails. With live.frames > 0
// (the live test), its line is written and its counts can fail the run.
int runSmokeExports(const std::string & directory, CompareState & state, Compositor & compositor, bool expectHdr,
                    bool expectVideo, const LiveSmoke & live)
{
    int code = 0;
    std::string report;
    std::error_code ec;
    fs::create_directories(pathFromUtf8(directory), ec);

    const auto attempt = [&code, &report](const char * label, const std::function<std::string()> & action)
    {
        try
        {
            report += std::string(label) + " ok: " + action() + "\n";
        }
        catch (const std::exception & e)
        {
            code = 1;
            report += std::string(label) + " FAILED: " + e.what() + "\n";
        }
    };
    attempt("view", [&]() { return exportCurrentView(compositor, pathJoin(directory, "view.png")); });
    attempt("diff", [&]() { return exportDiffMap(state, pathJoin(directory, "diff.png")); });
    // The FLIP map exists only when FLIP ran over the pair; without it the export is skipped.
    if (state.flip())
    {
        attempt("flip", [&]() { return exportFlipMap(state, pathJoin(directory, "flip.png")); });
    }
    // The exposure map exists only for an HDR-FLIP result; without one the export is skipped.
    if (state.flip() && state.flip()->hdr)
    {
        attempt("exposure", [&]() { return exportExposureMap(state, pathJoin(directory, "exposure.png")); });
    }
    attempt("report", [&]() { return exportReport(state, pathJoin(directory, "report.json")); });

    size_t notReady = 0;
    for (const MediaEntry & entry : state.entries())
    {
        notReady += entry.ready() ? 0 : 1;
    }
    if (notReady > 0)
    {
        code = 1;
    }
    report = "startup sources=" + std::to_string(state.size()) + " notReady=" + std::to_string(notReady) + "\n"
        + report;

    if (state.metrics())
    {
        const MetricsResult & metrics = *state.metrics();
        char psnr[64];
        if (std::isinf(metrics.psnrDb))
        {
            std::snprintf(psnr, sizeof(psnr), "inf");
        }
        else
        {
            std::snprintf(psnr, sizeof(psnr), "%.9g", metrics.psnrDb);
        }
        char line[256];
        std::snprintf(line, sizeof(line), "metrics mse=%.17g psnr=%s overlap=%dx%d+%d+%d pixels=%llu\n", metrics.mse,
                      psnr, metrics.overlap.width, metrics.overlap.height, metrics.overlap.x, metrics.overlap.y,
                      static_cast<unsigned long long>(metrics.pixelCount));
        report += line;
    }
    else
    {
        report += "metrics none reason=" + state.metricsReason() + "\n";
        code = 1;
    }
    if (expectVideo)
    {
        // Fixed six decimals for times, and %g for the speed.
        const auto fixed6 = [](double value)
        {
            char text[64];
            std::snprintf(text, sizeof(text), "%.6f", value);
            return std::string(text);
        };
        const auto general = [](double value)
        {
            char text[64];
            std::snprintf(text, sizeof(text), "%g", value);
            return std::string(text);
        };
        size_t visibleVideos = 0;
        for (size_t i = 0; i < state.size(); ++i)
        {
            const MediaEntry & entry = state.entries()[i];
            if (!entry.visible || entry.media->kind() != project::SourceKind::Video)
            {
                continue;
            }
            ++visibleVideos;
            const VideoSource * video = state.videoOf(i);
            report += "video " + entry.name + " frame=" + std::to_string(state.videoFrameIndex(i))
                + " time=" + fixed6(state.transport().time)
                + " sourceTime=" + fixed6(video != nullptr ? video->sourceTime() : 0.0) + "\n";
        }
        if (visibleVideos == 0)
        {
            report += "video none\n";
            code = 1;
        }
        const Transport & transport = state.transport();
        report += "transport time=" + fixed6(transport.time) + " speed=" + general(transport.speed)
            + " playing=" + (transport.playing ? "1" : "0") + "\n";
    }
    // A missing FLIP result is reported, not counted as a failure.
    if (state.flip())
    {
        char flipLine[256];
        std::snprintf(flipLine, sizeof(flipLine), "flip mean=%.9g ppd=%.9g nonFinite=%llu\n", state.flip()->mean,
                      static_cast<double>(state.flipPpd()),
                      static_cast<unsigned long long>(state.flip()->nonFiniteSamples));
        report += flipLine;
    }
    else
    {
        report += "flip none reason=" + state.flipReason() + "\n";
    }
    // An HDR-FLIP result is reported with its settings. Asked for and missing, it fails the run.
    if (state.flip() && state.flip()->hdr)
    {
        const FlipHdrInfo & info = *state.flip()->hdr;
        char hdrLine[320];
        std::snprintf(hdrLine, sizeof(hdrLine),
                      "hdr mean=%.9g tone=%s start=%.9g stop=%.9g count=%d linearMse=%.9g nonFinite=%llu\n",
                      state.flip()->mean, toneMapperName(info.toneMapper), static_cast<double>(info.start),
                      static_cast<double>(info.stop), static_cast<int>(info.count), info.linearMse,
                      static_cast<unsigned long long>(state.flip()->nonFiniteSamples));
        report += hdrLine;
    }
    else if (expectHdr || state.flipMode() == project::FlipMode::Hdr)
    {
        report += "hdr none reason=" + state.flipReason() + "\n";
        if (expectHdr)
        {
            code = 1;
        }
    }
    // The live line. Its counts are totals since startup, so the first load's renders and FLIP are included:
    // that is why the FLIP ceiling has two extra computes. The run fails when a visible scene rendered fewer
    // times than the drag had frames (each visible scene renders once per frame), when the FLIP computes
    // exceed the rate's ceiling for the drag, or when the state had not settled.
    if (live.frames > 0)
    {
        bool visibleScene = false;
        for (const MediaEntry & entry : state.entries())
        {
            if (entry.visible && entry.ready() && entry.media->kind() == project::SourceKind::Scene)
            {
                visibleScene = true;
            }
        }
        const uint64_t sceneRenders = state.sceneRenderTotal();
        const uint64_t flipComputes = state.flipComputeCount();
        const uint64_t metricsComputes = state.metricsComputeCount();
        const double flipCeiling = std::ceil(static_cast<double>(state.liveFlipHz()) * live.seconds) + 2.0;
        if ((visibleScene && sceneRenders < static_cast<uint64_t>(live.frames))
            || static_cast<double>(flipComputes) > flipCeiling || !live.settled)
        {
            code = 1;
        }
        char liveLine[320];
        std::snprintf(liveLine, sizeof(liveLine),
                      "live frames=%d seconds=%.3f sceneRenders=%llu flipComputes=%llu metricsComputes=%llu settled=%d\n",
                      live.frames, live.seconds, static_cast<unsigned long long>(sceneRenders),
                      static_cast<unsigned long long>(flipComputes), static_cast<unsigned long long>(metricsComputes),
                      live.settled ? 1 : 0);
        report += liveLine;
    }
    report += "composite=" + std::to_string(compositor.width()) + "x" + std::to_string(compositor.height()) + "\n";
    report += "exit=" + std::to_string(code) + "\n";

    std::ofstream out(pathFromUtf8(pathJoin(directory, "smoke.txt")), std::ios::binary);
    out << report;
    std::cout << report;
    return code;
}

} // namespace

int runApp(const LaunchDecision & decision)
{
    if (!glfwInit())
    {
        Logger::error("Sabercat Compare: GLFW could not start");
        std::cerr << "sabercat_compare: GLFW could not start\n";
        return 1;
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    const std::string baseTitle = std::string("Sabercat Compare ") + sabercat::kVersionString;
    GLFWwindow * window = glfwCreateWindow(kWindowWidth, kWindowHeight, baseTitle.c_str(), nullptr, nullptr);
    if (window == nullptr)
    {
        Logger::error("Sabercat Compare: the window could not be created");
        std::cerr << "sabercat_compare: the window could not be created\n";
        glfwTerminate();
        return 1;
    }
    Logger::info(std::string("Sabercat Compare ") + sabercat::kVersionString + " starting");
    app::setWindowIconFromResource(window);
    WindowHooks hooks;
    glfwSetWindowUserPointer(window, &hooks);
    glfwSetDropCallback(window, onDrop);

    // The window tells the device which instance extensions its windowing layer needs.
    uint32_t glfwExtCount = 0;
    const char ** glfwExts = glfwGetRequiredInstanceExtensions(&glfwExtCount);
    const std::vector<const char *> instanceExtensions(glfwExts, glfwExts + glfwExtCount);

    // The Studio's ownership model B: the app creates the device and the renderer borrows it.
    const std::string shaderDir = app::resolveShaderDir();
    const bool validation = app::resolveValidation(decision.validation, decision.noValidation);
    const char * smokeEnv = std::getenv("SABERCAT_COMPARE_SMOKE_DIR");
    const std::string smokeDir = smokeEnv != nullptr ? smokeEnv : "";
    const char * resizeEnv = std::getenv("SABERCAT_COMPARE_SMOKE_RESIZE");
    const bool smokeResize = resizeEnv != nullptr && std::string(resizeEnv) == "1";
    const char * hdrEnv = std::getenv("SABERCAT_COMPARE_SMOKE_HDR");
    const std::string smokeHdr = hdrEnv != nullptr ? hdrEnv : "";
    // The playback test runs only with the smoke directory; its frame count sets when the exports run.
    const int smokePlay = smokeDir.empty() ? 0 : smokePlayFrames(std::getenv("SABERCAT_COMPARE_SMOKE_PLAY"));
    const int smokeEndFrame = smokePlay > 0 ? smokePlay + kSmokePlayStartFrame : kSmokeFrames;
    // The live test runs only with the smoke directory; its drag ends at liveDragEnd (the first frame after it).
    const int smokeLive = smokeDir.empty() ? 0 : smokeLiveFrames(std::getenv("SABERCAT_COMPARE_SMOKE_LIVE"));
    const int liveDragEnd = smokeLive > 0 ? kSmokeLiveStartFrame + smokeLive : 0;
    // The ID scan runs only with the smoke directory and SABERCAT_COMPARE_SMOKE_IDSCAN=1; it ends the run itself.
    const char * idScanEnv = std::getenv("SABERCAT_COMPARE_SMOKE_IDSCAN");
    const bool idScanOn = !smokeDir.empty() && idScanEnv != nullptr && std::string(idScanEnv) == "1";

    Renderer renderer(kSwapchainWidth, kSwapchainHeight);
    app::BorrowedVulkanHost host;
    try
    {
        host.create(instanceExtensions, /*presentation=*/true, validation);
        renderer.initializeBorrowed(host.context(), host.enabled, shaderDir);
        Logger::info("Sabercat Compare running on an app-owned (borrowed) Vulkan device");
    }
    catch (const std::exception & e)
    {
        Logger::error(std::string("Sabercat Compare: device setup failed: ") + e.what());
        std::cerr << "sabercat_compare: device setup failed: " << e.what() << '\n';
        renderer.shutdown();
        host.destroy();
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    // Declared here so the teardown below can release everything whether or not the loop
    // threw. Order of release: the UI (its texture), the state and the GPU objects, ImGui,
    // the renderer, the device.
    app::ImGuiLayer imgui;
    app::AboutLogo aboutLogo;
    std::unique_ptr<app::AsyncFileDialog> dialogs;
    std::unique_ptr<GpuContext> gpu;
    std::unique_ptr<MetricsGpu> metrics;
    std::unique_ptr<Compositor> compositor;
    std::unique_ptr<CompareState> state;
    std::unique_ptr<CompareUi> ui;
    app::SystemInfoFacts facts;
    int result = 0;
    bool frameOpen = false;
    try
    {
        // The app creates the surface from the renderer's instance and hands it back; the
        // release callback destroys it before the instance goes.
        VkSurfaceKHR surface = VK_NULL_HANDLE;
        VK_CHECK(glfwCreateWindowSurface(renderer.instance(), window, nullptr, &surface),
                 "Failed to create window surface");
        renderer.attachSurface(surface,
                               kSwapchainWidth,
                               kSwapchainHeight,
                               [&renderer, surface]()
                               { vkDestroySurfaceKHR(renderer.instance(), surface, nullptr); });

        imgui.init(window, renderer);
        // No imgui.ini in the working directory: the window keeps no layout between runs.
        ImGui::GetIO().IniFilename = nullptr;
        aboutLogo.create(renderer.sabercatDevice());

        gpu = std::make_unique<GpuContext>(renderer.sabercatDevice(), shaderDir);
        metrics = std::make_unique<MetricsGpu>(*gpu);
        compositor = std::make_unique<Compositor>(*gpu);
        state = std::make_unique<CompareState>(*gpu);
        // Scene sources borrow the presenter's device through this renderer.
        state->setSceneHost(&renderer);
        dialogs = std::make_unique<app::AsyncFileDialog>();
        facts = app::gatherSystemInfo("Sabercat Compare", &renderer.sabercatDevice(), validation);
        Logger::info(app::formatSystemInfo(facts, Logger::logFilePath()));
        ui = std::make_unique<CompareUi>(*state, *compositor, *metrics, *dialogs, window, renderer.device(), facts);
        ui->setLogoTexture(aboutLogo.texture());

        ui->openPaths(decision.files);

        std::string lastTitle;
        int frame = 0;
        bool smokeDone = false;
        // The live test's drag: its wall-clock start, and whether it has started and been timed.
        LiveSmoke live;
        live.frames = smokeLive;
        double liveStart = 0.0;
        bool liveDragStarted = false;
        bool liveDragTimed = false;
        std::unique_ptr<IdScan> idScan;
        if (idScanOn)
        {
            idScan = std::make_unique<IdScan>(window);
        }
        while (true)
        {
            glfwPollEvents();
            ++frame;
            if (!smokeDir.empty() && smokeResize)
            {
                smokeWindowSequence(window, frame);
            }
            if (!smokeHdr.empty())
            {
                smokeHdrSequence(*state, frame, smokeHdr);
            }
            if (smokePlay > 0 && frame == kSmokePlayStartFrame)
            {
                state->play();
            }

            // Closing with unsaved changes asks first: the window stays open until the
            // prompt answers Discard (requestQuit then sets quitRequested).
            if (glfwWindowShouldClose(window))
            {
                if (!state->projectDirty())
                {
                    break;
                }
                glfwSetWindowShouldClose(window, GLFW_FALSE);
                ui->requestQuit();
            }

            if (!hooks.dropped.empty())
            {
                const std::vector<std::string> dropped = std::move(hooks.dropped);
                hooks.dropped.clear();
                ui->openPaths(dropped);
            }

            // Minimized (zero-size framebuffer): nothing to draw, so wait for an event instead.
            int framebufferWidth = 0;
            int framebufferHeight = 0;
            glfwGetFramebufferSize(window, &framebufferWidth, &framebufferHeight);
            if (framebufferWidth == 0 || framebufferHeight == 0)
            {
                glfwWaitEventsTimeout(0.1);
                continue;
            }

            // The ID scan sets this frame's view and pointer before ImGui reads them, and ends the run when its sweep is done.
            if (idScan)
            {
                idScan->snapshot(*ImGui::GetCurrentContext());
                if (!idScan->prepare(*state, *ui))
                {
                    result = idScan->finish();
                    break;
                }
            }

            try
            {
                imgui.beginFrame();
                frameOpen = true;
                if (idScan)
                {
                    idScan->checkConflicts(*ImGui::GetCurrentContext());
                }
                // The playback test passes a fixed step per presented frame; otherwise the real time.
                const double now = smokePlay > 0 ? static_cast<double>(frame) / kSmokePlayFps : glfwGetTime();
                // The live drag: a fixed camera step per frame through the same setCamera the input uses (as
                // interactive). It runs before ui->draw, so the move is stamped with the previous frame's clock.
                if (smokeLive > 0 && frame >= kSmokeLiveStartFrame && frame < liveDragEnd)
                {
                    if (!liveDragStarted)
                    {
                        liveDragStarted = true;
                        liveStart = glfwGetTime();
                    }
                    app::OrbitCamera cam = state->camera();
                    cam.applyDelta(0.02f, 0.004f, 0.0f);
                    state->setCamera(cam, true);
                }
                else if (liveDragStarted && !liveDragTimed && frame >= liveDragEnd)
                {
                    live.seconds = glfwGetTime() - liveStart;
                    liveDragTimed = true;
                }
                ui->draw(now);
                if (idScan)
                {
                    idScan->recordHover(*ImGui::GetCurrentContext());
                }
                if (ui->quitRequested())
                {
                    ImGui::Render();
                    frameOpen = false;
                    break;
                }

                const std::string title = ui->windowTitle();
                if (title != lastTitle)
                {
                    glfwSetWindowTitle(window, title.c_str());
                    lastTitle = title;
                }

                // The composite is drawn before ImGui's draw data is recorded, since the
                // view area samples it in this frame.
                ui->renderView(now);
                ImGui::Render();
                frameOpen = false;
                renderer.present();

                // The live test ends once its drag has run and the state has settled, or kSmokeLiveSettleFrames
                // later (then the run fails on settled == 0); the other modes end at smokeEndFrame.
                const bool smokeDue = smokeLive > 0
                    ? frame >= liveDragEnd && (state->liveSettled() || frame >= liveDragEnd + kSmokeLiveSettleFrames)
                    : frame >= smokeEndFrame;
                if (!smokeDir.empty() && !idScan && !smokeDone && smokeDue)
                {
                    smokeDone = true;
                    // An HDR run that has not finished (or not started) completes here, so the exports have it.
                    if (state->flipMode() == project::FlipMode::Hdr && !state->flip())
                    {
                        state->runHdrBlocking();
                    }
                    if (smokeLive > 0)
                    {
                        live.settled = state->liveSettled();
                    }
                    result = runSmokeExports(smokeDir, *state, *compositor, !smokeHdr.empty(), smokePlay > 0, live);
                    break;
                }
            }
            catch (const std::exception & e)
            {
                if (frameOpen)
                {
                    ImGui::EndFrame();
                    frameOpen = false;
                }
                const std::string what = e.what();
                Logger::error(std::string("Sabercat Compare: ") + what);
                if (isDeviceLost(what))
                {
                    std::cerr << "sabercat_compare: the GPU was lost: " << what << '\n';
                    result = 1;
                    break;
                }
                // Any other failure is shown in the status line and the loop goes on.
                state->setStatus(std::string("Error: ") + what);
            }
        }
    }
    catch (const std::exception & e)
    {
        Logger::error(std::string("Sabercat Compare FAILED: ") + e.what());
        std::cerr << "sabercat_compare: " << e.what() << '\n';
        result = 1;
    }

    // Teardown. The GPU may still hold work (or be lost): wait, then release everything that
    // holds a device object, then ImGui, then the renderer and the borrowed device.
    if (renderer.device() != VK_NULL_HANDLE)
    {
        vkDeviceWaitIdle(renderer.device());
    }
    ui.reset();
    dialogs.reset();
    state.reset();
    compositor.reset();
    metrics.reset();
    gpu.reset();
    aboutLogo.destroy();
    imgui.shutdown(renderer);
    renderer.shutdown();
    host.destroy();
    glfwDestroyWindow(window);
    glfwTerminate();
    return result;
}

} // namespace sabercat::compare::gui
