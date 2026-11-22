// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The compare window's ImGui front end: the menu bar, the media panel, the view area
// (the compositor's image, drawn with ImGui::Image), the mouse, the probe, the metrics
// and the exports. CompareState holds the data; the compositor and the metrics do the GPU
// work. Everything here runs on the main thread.
#pragma once

#include <imgui.h>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <string>
#include <vector>

#include "compare_layout.h"
#include "system_info.h"

struct GLFWwindow;

namespace sabercat::app
{
class AsyncFileDialog;
} // namespace sabercat::app

namespace sabercat::compare::gui
{

class CompareState;
struct CompositeInput;
class Compositor;
class MetricsGpu;

class CompareUi
{
  public:
    // The objects are borrowed and must outlive the UI. device is the window's device.
    CompareUi(CompareState & state, Compositor & compositor, MetricsGpu & metrics, app::AsyncFileDialog & dialogs,
              GLFWwindow * window, VkDevice device, const app::SystemInfoFacts & facts);
    // Unregisters the view texture and frees its sampler. Call before ImGui shuts down; the
    // device must still exist.
    ~CompareUi();

    CompareUi(const CompareUi &) = delete;
    CompareUi & operator=(const CompareUi &) = delete;

    // The Help > About logo texture (0 to skip it).
    void setLogoTexture(ImTextureID logo)
    {
        m_logo = logo;
    }

    // Opens or adds the given paths: a .sabercat-compare file opens the project (after the
    // unsaved-changes prompt), anything else is added as media. Used by drops and the
    // command line.
    void openPaths(const std::vector<std::string> & paths);

    // Builds this frame's ImGui content. It starts the state's frame first (beginFrame), so the
    // live policy sees this frame's time. Updates the metrics when they are stale, resizes
    // the compositor to the view area when that changed, and handles the mouse and keys.
    // Call between ImGui::NewFrame and ImGui::Render.
    void draw(double nowSeconds);

    // Renders the composite of the state for the view area drawn by draw(). Call after
    // draw() and before ImGui::Render(), since the image is sampled by that frame's draw
    // list.
    void renderView(double nowSeconds);

    // File > Quit, or the window's close button. With unsaved changes it asks first.
    void requestQuit();
    bool quitRequested() const
    {
        return m_quitRequested;
    }

    // "<project> - Sabercat Compare <version>", with " *" when the project is dirty.
    std::string windowTitle() const;

    // Test hook of the ID scan (SABERCAT_COMPARE_SMOKE_IDSCAN): the next draw() opens the settings
    // popup of entry index (drawSourceRow opens it). A negative index closes the popups.
    void smokeOpenSettings(int32_t index);

  private:
    enum class DialogPurpose : uint8_t
    {
        None,
        AddMedia,
        OpenProject,
        SaveProject,
        ExportView,
        ExportDiff,
        ExportFlip,
        ExportExposure,
        ExportPanes,
        ExportReport,
        Locate
    };

    // What runs once the unsaved-changes prompt is answered.
    enum class PendingAction : uint8_t
    {
        None,
        NewProject,
        OpenDialog,
        OpenPath,
        Quit
    };

    // The probe under the mouse, computed in the view area and shown in the side panel.
    // The floats come first, then the ints, then the bytes, then the bools.
    struct Probe
    {
        float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float b[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float depthA[4] = {0.0f, 0.0f, 0.0f, 0.0f}; // raw depth components of A and B
        float depthB[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float flip = 0.0f;   // FLIP error at the pixel, when hasFlip
        int32_t x = 0;       // canvas pixel
        int32_t y = 0;
        int32_t target = -1; // source the pane shows at the pixel, -1 when none
        int32_t localX = 0;
        int32_t localY = 0;
        PixelKind kind = PixelKind::Outside;
        uint8_t depthCountA = 0; // depth components read for A (0 = none)
        uint8_t depthCountB = 0;
        bool valid = false;
        bool hasA = false;
        bool hasB = false;
        bool hasFlip = false; // the pixel is inside the FLIP overlap
    };

    void drawMenuBar();
    void drawSidePanel();
    void drawSourceRow(size_t index);
    void drawSourceSettings(size_t index);
    void drawSceneSettings(size_t index);
    void drawVideoSettings(size_t index);
    void drawCompareSection();
    void drawViewSection();
    void drawProbeSection();
    void drawViewArea(float width, float height);
    // The transport bar under the view area: play, step, seek, time, frames, speed and loop.
    void drawTransportBar();
    // Height of the transport bar in pixels; 0 when there is no video.
    float transportHeight() const;
    void drawStatusLine();
    void drawModals();
    void handleShortcuts();
    void fitView();
    void updateProbe(const LayoutInput & shown, int32_t px, int32_t py);
    bool readSample(int32_t index, int32_t x, int32_t y, float rgba[4]) const;
    // The depth components of source index at a canvas pixel; 0 when there are none.
    uint8_t readDepthSample(int32_t index, int32_t x, int32_t y, float out[4]) const;
    bool isSceneIndex(int32_t index) const;
    // Whether the pane under the pointer shows a scene (the camera takes its input there).
    bool paneShowsScene(const CompositeInput & input, const LayoutInput & shown, int32_t px, int32_t py) const;
    // Applies pointer input to the shared camera; the caller passes only real input.
    void orbitScene(float deltaX, float deltaY, bool orbit, bool pan, float wheel);
    // Waits until the GPU is idle, before a scene is released (removed, relocated or replaced).
    void idleGpu();

    void pollDialog();
    bool beginOpen(DialogPurpose purpose, const std::string & title, bool imagesOnly);
    bool beginSave(DialogPurpose purpose, const std::string & defaultName, const std::string & extension);
    void requestAddMedia();
    void requestLocate(size_t index);
    void requestNew();
    void requestOpenDialog();
    void requestOpenPath(const std::string & path);
    void beginSaveProjectAs();
    void saveProjectOrSaveAs();
    bool saveTo(const std::string & utf8Path);
    void doOpenProject(const std::string & utf8Path);
    void continuePending();
    void runExport(const std::string & label, const std::string & utf8Path, DialogPurpose purpose);

    void resizeView(int32_t width, int32_t height);
    void addTexture();
    void removeTexture();
    ImTextureID textureId() const;

    CompareState & m_state;
    Compositor & m_compositor;
    MetricsGpu & m_metrics;
    app::AsyncFileDialog & m_dialogs;
    GLFWwindow * m_window;
    VkDevice m_device;
    app::SystemInfoFacts m_facts;
    ImTextureID m_logo = 0;

    VkSampler m_sampler = VK_NULL_HANDLE;
    VkDescriptorSet m_viewSet = VK_NULL_HANDLE;

    DialogPurpose m_purpose = DialogPurpose::None;
    size_t m_locateIndex = 0;
    PendingAction m_pending = PendingAction::None;
    std::string m_pendingPath;

    Probe m_probe;
    // The live FLIP rate field, staged like the ppd field: it follows the state unless it is being typed in.
    float m_liveFlipField = 0.0f;
    int m_flipPpdField = 0;          // the ppd field's value while the state is not being typed over
    bool m_flipPpdEditing = false;   // the ppd field was active last frame
    bool m_liveFlipEditing = false;  // the live FLIP rate field was active last frame
    // The HDR-FLIP fields, staged the same way: each follows the state unless it is being typed in.
    float m_hdrStartField = 0.0f;
    float m_hdrStopField = 1.0f;
    int m_hdrCountField = 16;
    bool m_hdrStartEditing = false;
    bool m_hdrStopEditing = false;
    bool m_hdrCountEditing = false;
    // The video fields of the source settings, staged like the HDR-FLIP fields: each follows the
    // state unless it is being typed in, and applies when it is left after an edit.
    double m_videoTrimInField = 0.0;
    double m_videoTrimOutField = 0.0;
    double m_videoOffsetSecondsField = 0.0;
    int m_videoOffsetFramesField = 0;
    bool m_videoTrimInEditing = false;
    bool m_videoTrimOutEditing = false;
    bool m_videoOffsetSecondsEditing = false;
    bool m_videoOffsetFramesEditing = false;
    bool m_wipeDrag = false;    // the drag that started on the single-screen wipe line moves it
    bool m_dividerDrag = false; // the drag that started on the split divider moves it
    bool m_sceneDrag = false;   // the drag that started on a scene pane drives the camera
    bool m_clockStarted = false; // the first draw() moves the transport clock by nothing
    bool m_smokeSettingsClose = false; // smokeOpenSettings(-1) asked to close the popups
    int32_t m_pendingRemove = -1;
    int32_t m_smokeSettings = -1;      // the entry whose settings popup the ID scan opens, or -1
    double m_now = 0.0;

    bool m_quitRequested = false;
    bool m_aboutOpen = false;
    bool m_errorOpen = false;
    bool m_discardOpen = false;
    bool m_noticesOpen = false;
    std::string m_error;
};

// Export actions shared by the menus and the self-test of the window (SABERCAT_COMPARE_SMOKE_DIR).
// Each returns the status line text on success and throws std::runtime_error on failure.
// exportCurrentView() writes the composite as it was last rendered. exportPaneImages() writes
// one padded image per visible source next to utf8Path, named <stem>_<id>.png.
std::string exportCurrentView(Compositor & compositor, const std::string & utf8Path);
std::string exportDiffMap(const CompareState & state, const std::string & utf8Path);
// Throws std::runtime_error("no FLIP: ...") when the state has no FLIP result.
std::string exportFlipMap(const CompareState & state, const std::string & utf8Path);
// Throws std::runtime_error("no HDR-FLIP result") when the state has no HDR-FLIP result.
std::string exportExposureMap(const CompareState & state, const std::string & utf8Path);
std::string exportPaneImages(const CompareState & state, const std::string & utf8Path);
std::string exportReport(const CompareState & state, const std::string & utf8Path);

} // namespace sabercat::compare::gui
