// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "compare_ui.h"

#include "about_panel.h"
#include "app_util.h"
#include "compare_state.h"
#include "compositor.h"
#include "export.h"
#include "file_dialog.h"
#include "image_io.h"
#include "metrics_gpu.h"
#include "orbit_camera.h"
#include "report.h"
#include "system_info.h"
#include "video_source.h"

#include "sabercat/logger.h"
#include "sabercat/version.h"
#include "sabercat/vk_check.h"

#include <imgui_impl_vulkan.h>
#include <imgui_internal.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace sabercat::compare::gui
{

namespace
{

namespace fs = std::filesystem;

constexpr float kSidePanelWidth = 340.0f;
// The grab line of the single-screen wipe and of the split divider, in logical pixels: a press
// within kGrabLineReach of the line, along the image area, grabs it, and so does a press inside
// the round handle at the middle of the line (its radius is kGrabHandleRadius).
constexpr float kGrabLineReach = 8.0f;
constexpr float kGrabHandleRadius = 14.0f;
// The divider hint's text, drawn at the top center of the image area in Split.
constexpr const char * kHintDivider = "Drag the divider to compare \xC2\xB7 "
                                      "drag the image to pan \xC2\xB7 wheel to zoom";
constexpr const char * kHintPan = "Drag the image to pan \xC2\xB7 wheel to zoom";
// Zoom factor per mouse-wheel notch.
constexpr double kWheelZoomFactor = 1.1;
constexpr const char * kDialogTag = "sabercat-compare";
constexpr const char * kImageExtensions = "*.png;*.jpg;*.jpeg;*.webp;*.exr;*.rgba32f";
// The scene files the window loads (see isSceneFile in scene_source.h).
constexpr const char * kSceneExtensions = "*.ply;*.spz;*.splat;*.ksplat;*.sog;*.gltf;*.glb;meta.json";
// The video files the window loads (see isVideoFile in video_source.h).
constexpr const char * kVideoExtensions = "*.mp4;*.m4v;*.webm";
constexpr const char * kProjectExtension = ".sabercat-compare";
// Media Foundation values of the stream tags the video settings note (MFVideoTransferMatrix_BT601,
// MFNominalRange_0_255).
constexpr uint32_t kMfMatrixBt601 = 2;
constexpr uint32_t kMfNominalRange0To255 = 1;
// The speed presets of the transport bar; any value in [0.25, 4] is accepted by the state.
constexpr std::array<double, 5> kSpeedPresets = {0.25, 0.5, 1.0, 2.0, 4.0};
// Q and E roll the camera at this many radians per second, as the viewer does.
constexpr float kRollSpeed = 1.5f;
// The render size limits the window accepts for a scene.
constexpr int kMinRenderSize = 16;
constexpr int kMaxRenderSize = 8192;

const ImVec4 kErrorColor(1.0f, 0.45f, 0.40f, 1.0f);
const ImVec4 kWarningColor(1.0f, 0.80f, 0.30f, 1.0f);

const char * const kRoleNames[] = {"None", "A", "B"};
const char * const kToneMapperNames[] = {"ACES", "Hable", "Reinhard"};
const char * const kAnchorTopLeft = "Top left";
const char * const kAnchorCentered = "Centered";

fs::path pathFromUtf8(const std::string & utf8)
{
    return fs::path(std::u8string(reinterpret_cast<const char8_t *>(utf8.data()), utf8.size()));
}

std::string utf8Of(const fs::path & path)
{
    const std::u8string text = path.u8string();
    return std::string(reinterpret_cast<const char *>(text.data()), text.size());
}

std::string fileNameUtf8(const std::string & utf8)
{
    return utf8Of(pathFromUtf8(utf8).filename());
}

std::string stemUtf8(const std::string & utf8)
{
    return utf8Of(pathFromUtf8(utf8).stem());
}

bool isProjectPath(const std::string & utf8)
{
    std::string extension = utf8Of(pathFromUtf8(utf8).extension());
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return extension == kProjectExtension;
}

// Images, scenes and videos: the add and locate dialogs take all three.
std::vector<app::FileFilter> imageFilters()
{
    return {app::FileFilter{"Images, scenes and videos",
                            std::string(kImageExtensions) + ";" + kSceneExtensions + ";" + kVideoExtensions}};
}

std::vector<app::FileFilter> projectFilters()
{
    return {app::FileFilter{"Sabercat Compare project", "*.sabercat-compare"}};
}

std::vector<app::FileFilter> pngFilters()
{
    return {app::FileFilter{"PNG image", "*.png"}};
}

std::vector<app::FileFilter> reportFilters()
{
    return {app::FileFilter{"Report", "*.json;*.csv"}};
}

const char * anchorName(project::Anchor anchor)
{
    switch (anchor)
    {
    case project::Anchor::TopLeft:
        return "topLeft";
    case project::Anchor::Centered:
        return "centered";
    case project::Anchor::Custom:
        return "custom";
    }
    return "custom";
}

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

// A stand-in host copy for a source that failed to load. It is never shown: the layout
// keeps such a source hidden, so the export ignores it.
const ImageData & placeholderImage()
{
    static const ImageData image = []()
    {
        ImageData data;
        data.width = 1;
        data.height = 1;
        data.rgba.assign(4, 0.0f);
        data.bitDepth = 32;
        return data;
    }();
    return image;
}

// The padding counterpart of source `index`, by the rule of paddingCounterpartSlot(): the
// other of the A/B pair when visible, else the first other visible source, else -1.
int32_t counterpartOf(const CompareState & state, size_t index)
{
    const std::vector<MediaEntry> & entries = state.entries();
    const auto visible = [&entries](int32_t k)
    {
        return k >= 0 && static_cast<size_t>(k) < entries.size() && entries[static_cast<size_t>(k)].visible;
    };
    const int32_t self = static_cast<int32_t>(index);
    const int32_t a = state.sourceA();
    const int32_t b = state.sourceB();
    if (self == a && visible(b) && b != self)
    {
        return b;
    }
    if (self == b && visible(a) && a != self)
    {
        return a;
    }
    for (size_t k = 0; k < entries.size(); ++k)
    {
        if (static_cast<int32_t>(k) != self && entries[k].visible)
        {
            return static_cast<int32_t>(k);
        }
    }
    return -1;
}

// The export scene of the current state: every source's host copy and geometry.
ExportScene buildExportScene(const CompareState & state)
{
    ExportScene scene;
    const std::vector<MediaEntry> & entries = state.entries();
    const LayoutInput & layout = state.layout();
    for (size_t i = 0; i < entries.size(); ++i)
    {
        const MediaEntry & entry = entries[i];
        ExportSource source;
        // A hidden scene is not rendered, so it has no host copy; hidden sources are not
        // exported, so the placeholder stands in for it.
        const ImageData * host = entry.ready() ? entry.media->hostData() : nullptr;
        const bool placeholder = host == nullptr;
        if (placeholder)
        {
            if (entry.ready() && entry.visible)
            {
                throw std::runtime_error(entry.name + " has no host copy to export");
            }
            host = &placeholderImage();
        }
        source.image = host;
        source.background = RgbaColor{entry.background.r, entry.background.g, entry.background.b, 1.0f};
        source.geometry = layout.sources[i];
        if (placeholder)
        {
            // The export checks each image against its geometry, so the placeholder's
            // geometry is its own 1 x 1 size; the hidden source takes no part in the output.
            source.geometry.width = 1;
            source.geometry.height = 1;
        }
        source.counterpart = counterpartOf(state, i);
        source.exposureStops = entry.exposure;
        source.toneMapper = toFlipToneMapper(entry.toneMapper);
        scene.sources.push_back(source);
    }
    scene.sourceA = state.sourceA();
    scene.sourceB = state.sourceB();
    return scene;
}

// A depth or coverage value as the probe prints it: %.9g, with non-finite values spelled out
// (printf's own spelling of them differs between compilers).
std::string probeNumber(float value)
{
    if (std::isnan(value))
    {
        return "nan";
    }
    if (std::isinf(value))
    {
        return value > 0.0f ? "inf" : "-inf";
    }
    char text[32];
    std::snprintf(text, sizeof(text), "%.9g", static_cast<double>(value));
    return text;
}

// The text that says why there is no FLIP result: "no FLIP: <reason>", with the prefix
// written once when the reason already has it.
std::string flipUnavailable(const CompareState & state)
{
    const std::string & reason = state.flipReason();
    if (reason.empty())
    {
        return "no FLIP";
    }
    if (reason.starts_with("no FLIP"))
    {
        return reason;
    }
    return "no FLIP: " + reason;
}

// The pixels per degree as the FLIP lines print it: a whole number as is, else two decimals.
std::string ppdText(float ppd)
{
    char text[32];
    if (ppd == std::floor(ppd))
    {
        std::snprintf(text, sizeof(text), "%.0f", static_cast<double>(ppd));
    }
    else
    {
        std::snprintf(text, sizeof(text), "%.2f", static_cast<double>(ppd));
    }
    return text;
}

// "Depth: v" for one component. Several components are named by the source's layout:
// the front depth, the coverage, and the depth itself (component 0); others are not shown.
std::string depthText(const DepthInfo & info, const float * values, uint32_t count)
{
    if (count == 1)
    {
        return "Depth: " + probeNumber(values[0]);
    }
    std::string text = "Depth:";
    for (uint32_t c = 0; c < count; ++c)
    {
        const auto component = static_cast<int32_t>(c);
        const char * label = nullptr;
        if (component == info.frontComponent)
        {
            label = "front";
        }
        else if (component == info.coverageComponent)
        {
            label = "cov";
        }
        else if (c == 0)
        {
            label = "D";
        }
        if (label != nullptr)
        {
            text += " ";
            text += label;
            text += "=" + probeNumber(values[c]);
        }
    }
    return text;
}

// Window pixels from ImGui units.
int32_t toPixels(float value, float scale)
{
    const float pixels = std::max(0.0f, value) * scale;
    return static_cast<int32_t>(std::min(std::lround(pixels), 16384L));
}

// The grab line and its handle, in logical pixels: the line runs across the image area [min, max]
// at coordinate (x for a vertical line, y for a horizontal one), and the handle sits at its middle.
// The line and the handle are drawn on the window's draw list, after the view's invisible button,
// so they never reach the composite or the exports.
ImVec2 grabHandleCenter(ImVec2 min, ImVec2 max, float coordinate, bool vertical)
{
    return vertical ? ImVec2(coordinate, 0.5f * (min.y + max.y)) : ImVec2(0.5f * (min.x + max.x), coordinate);
}

// Whether a press at pointer grabs the line. The pointer is unclamped: a line at the edge of the
// image area (fraction 0 or 1) keeps its handle, and a pointer outside the area can still grab.
bool grabHit(ImVec2 pointer, ImVec2 min, ImVec2 max, float coordinate, bool vertical)
{
    const float across = vertical ? pointer.x - coordinate : pointer.y - coordinate;
    const float along = vertical ? pointer.y : pointer.x;
    const float from = vertical ? min.y : min.x;
    const float to = vertical ? max.y : max.x;
    if (std::fabs(across) <= kGrabLineReach && along >= from && along <= to)
    {
        return true;
    }
    const ImVec2 center = grabHandleCenter(min, max, coordinate, vertical);
    const float dx = pointer.x - center.x;
    const float dy = pointer.y - center.y;
    return dx * dx + dy * dy <= kGrabHandleRadius * kGrabHandleRadius;
}

// A white line of 2 px with a 1 px dark shadow to its lower right, across the image area.
void drawGrabLine(ImDrawList * drawList, ImVec2 min, ImVec2 max, float coordinate, bool vertical)
{
    const ImVec2 from = vertical ? ImVec2(coordinate, min.y) : ImVec2(min.x, coordinate);
    const ImVec2 to = vertical ? ImVec2(coordinate, max.y) : ImVec2(max.x, coordinate);
    drawList->AddLine(ImVec2(from.x + 1.0f, from.y + 1.0f), ImVec2(to.x + 1.0f, to.y + 1.0f),
                      IM_COL32(0, 0, 0, 160), 2.0f);
    drawList->AddLine(from, to, IM_COL32(255, 255, 255, 255), 2.0f);
}

// The round grab handle: a dark translucent disk with a white outline and two white triangles,
// left and right for a vertical line, up and down for a horizontal one. hot brightens the disk.
void drawGrabHandle(ImDrawList * drawList, ImVec2 center, bool vertical, bool hot)
{
    const ImU32 white = IM_COL32(255, 255, 255, 255);
    drawList->AddCircleFilled(center, kGrabHandleRadius, IM_COL32(20, 20, 20, hot ? 215 : 170), 32);
    drawList->AddCircle(center, kGrabHandleRadius, white, 32, 1.5f);
    if (vertical)
    {
        drawList->AddTriangleFilled(ImVec2(center.x - 8.0f, center.y), ImVec2(center.x - 3.0f, center.y - 4.0f),
                                    ImVec2(center.x - 3.0f, center.y + 4.0f), white);
        drawList->AddTriangleFilled(ImVec2(center.x + 8.0f, center.y), ImVec2(center.x + 3.0f, center.y - 4.0f),
                                    ImVec2(center.x + 3.0f, center.y + 4.0f), white);
    }
    else
    {
        drawList->AddTriangleFilled(ImVec2(center.x, center.y - 8.0f), ImVec2(center.x - 4.0f, center.y - 3.0f),
                                    ImVec2(center.x + 4.0f, center.y - 3.0f), white);
        drawList->AddTriangleFilled(ImVec2(center.x, center.y + 8.0f), ImVec2(center.x - 4.0f, center.y + 3.0f),
                                    ImVec2(center.x + 4.0f, center.y + 3.0f), white);
    }
}

// A one-line hint on a translucent dark rounded box, centered at the top of the image area [min, max].
void drawHintLine(ImDrawList * drawList, ImVec2 min, ImVec2 max, const char * text)
{
    const ImVec2 size = ImGui::CalcTextSize(text);
    const float padX = 8.0f;
    const float padY = 4.0f;
    const float left = 0.5f * (min.x + max.x) - 0.5f * size.x - padX;
    const float top = min.y + 8.0f;
    drawList->AddRectFilled(ImVec2(left, top), ImVec2(left + size.x + 2.0f * padX, top + size.y + 2.0f * padY),
                            IM_COL32(0, 0, 0, 150), 6.0f);
    drawList->AddText(ImVec2(left + padX, top + padY), IM_COL32(235, 235, 235, 255), text);
}

// The source pixel probed at canvas pixel (x, y) when the pane shows source index: Image inside
// the source's rectangle (its local pixel is the target), else Padding inside the padded extent,
// else Outside.
void probeSource(const LayoutInput & shown, int32_t index, int32_t x, int32_t y, ProbeTarget & target,
                 PixelKind & kind)
{
    const LayoutSource & source = shown.sources[static_cast<size_t>(index)];
    target = ProbeTarget{};
    if (rectContains(sourceRect(source), x, y))
    {
        target.source = index;
        target.localX = x - source.offsetX;
        target.localY = y - source.offsetY;
        kind = PixelKind::Image;
    }
    else
    {
        kind = rectContains(paddedExtent(shown), x, y) ? PixelKind::Padding : PixelKind::Outside;
    }
}

} // namespace

// ---- Construction -----------------------------------------------------------------------

CompareUi::CompareUi(CompareState & state, Compositor & compositor, MetricsGpu & metrics,
                     app::AsyncFileDialog & dialogs, GLFWwindow * window, VkDevice device,
                     const app::SystemInfoFacts & facts)
    : m_state(state),
      m_compositor(compositor),
      m_metrics(metrics),
      m_dialogs(dialogs),
      m_window(window),
      m_device(device),
      m_facts(facts)
{
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    // The composite is already at window resolution, so the view is sampled 1:1.
    samplerInfo.magFilter = VK_FILTER_NEAREST;
    samplerInfo.minFilter = VK_FILTER_NEAREST;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(m_device, &samplerInfo, nullptr, &m_sampler), "Failed to create the view sampler");
}

CompareUi::~CompareUi()
{
    if (m_sampler != VK_NULL_HANDLE)
    {
        vkDeviceWaitIdle(m_device);
        removeTexture();
        vkDestroySampler(m_device, m_sampler, nullptr);
        m_sampler = VK_NULL_HANDLE;
    }
}

// ---- Texture of the view area -----------------------------------------------------------

ImTextureID CompareUi::textureId() const
{
    return static_cast<ImTextureID>(reinterpret_cast<uintptr_t>(m_viewSet));
}

void CompareUi::addTexture()
{
    m_viewSet = ImGui_ImplVulkan_AddTexture(m_sampler, m_compositor.view(), VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
}

void CompareUi::removeTexture()
{
    if (m_viewSet != VK_NULL_HANDLE)
    {
        ImGui_ImplVulkan_RemoveTexture(m_viewSet);
        m_viewSet = VK_NULL_HANDLE;
    }
}

void CompareUi::resizeView(int32_t width, int32_t height)
{
    // The old image may still be sampled by the GPU, and its texture is about to go.
    vkDeviceWaitIdle(m_device);
    removeTexture();
    m_compositor.resize(static_cast<uint32_t>(width), static_cast<uint32_t>(height));
    if (m_compositor.view() != VK_NULL_HANDLE)
    {
        addTexture();
    }
}

// ---- Per-frame entry points -------------------------------------------------------------

void CompareUi::smokeOpenSettings(int32_t index)
{
    m_smokeSettings = index;
    m_smokeSettingsClose = index < 0;
}

void CompareUi::draw(double nowSeconds)
{
    // The transport clock moves by the real time since the previous draw. The first draw moves
    // it by nothing, and a clock that went backwards moves it by nothing too.
    const double clockStep = m_clockStarted ? std::max(0.0, nowSeconds - m_now) : 0.0;
    m_clockStarted = true;
    m_now = nowSeconds;
    if (m_smokeSettingsClose)
    {
        // ClosePopupToLevel reads the stack's top entry, so it needs an open popup.
        if (ImGui::GetCurrentContext()->OpenPopupStack.Size > 0)
        {
            ImGui::ClosePopupToLevel(0, true);
        }
        m_smokeSettingsClose = false;
    }
    // Starts the state's frame before anything else reads the clock: the camera-idle timer, the
    // scene file poll and the once-per-frame render guard all key off this call.
    m_state.beginFrame(nowSeconds);
    pollDialog();
    // The videos follow the clock before the scenes and the metrics see this frame.
    m_state.advanceTime(clockStep);
    m_state.updateVideos();
    // Scenes re-render first, so the metrics below see this frame's camera.
    m_state.updateScenes();
    m_state.updateMetrics(m_metrics);
    // Advances a running HDR-FLIP job by a time box; a no-op when none runs.
    m_state.stepHdr(0.008);
    handleShortcuts();

    const ImGuiViewport * viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos);
    ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    const ImGuiWindowFlags rootFlags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize
        | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoBringToFrontOnFocus
        | ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoSavedSettings;
    if (ImGui::Begin("##compare", nullptr, rootFlags))
    {
        drawMenuBar();
        const ImVec2 available = ImGui::GetContentRegionAvail();
        const float bodyHeight = std::max(0.0f, available.y - ImGui::GetFrameHeightWithSpacing());
        // The transport bar sits at the bottom of the view area; the image takes the rest.
        const float transportBar = transportHeight();
        const float viewHeight = std::max(0.0f, bodyHeight - transportBar);
        const float viewWidth = std::max(0.0f, available.x - kSidePanelWidth);

        if (ImGui::BeginChild("##side", ImVec2(kSidePanelWidth, bodyHeight), ImGuiChildFlags_Borders))
        {
            drawSidePanel();
        }
        ImGui::EndChild();

        ImGui::SameLine(0.0f, 0.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        if (ImGui::BeginChild("##view", ImVec2(viewWidth, bodyHeight), ImGuiChildFlags_None,
                              ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse))
        {
            drawViewArea(viewWidth, viewHeight);
            if (transportBar > 0.0f)
            {
                ImGui::SetCursorPos(ImVec2(0.0f, viewHeight));
                drawTransportBar();
            }
        }
        ImGui::EndChild();
        ImGui::PopStyleVar();

        drawStatusLine();
    }
    ImGui::End();
    ImGui::PopStyleVar(2);

    drawModals();
    if (m_aboutOpen)
    {
        app::drawAboutWindow(m_aboutOpen, m_facts, m_logo);
    }
}

void CompareUi::renderView(double nowSeconds)
{
    if (m_compositor.width() == 0 || m_compositor.height() == 0)
    {
        return;
    }
    // The camera may have moved in draw(): render the scenes it changed before compositing.
    m_state.updateScenes();
    m_compositor.render(m_state.buildComposite(nowSeconds));
}

void CompareUi::openPaths(const std::vector<std::string> & paths)
{
    for (const std::string & path : paths)
    {
        if (isProjectPath(path))
        {
            requestOpenPath(path);
        }
        else
        {
            m_state.addMedia(path);
        }
    }
}

void CompareUi::requestQuit()
{
    if (!m_state.projectDirty())
    {
        m_quitRequested = true;
        return;
    }
    m_pending = PendingAction::Quit;
    m_discardOpen = true;
}

std::string CompareUi::windowTitle() const
{
    std::string title = std::string("Sabercat Compare ") + sabercat::kVersionString;
    if (!m_state.projectPath().empty())
    {
        title += " - " + fileNameUtf8(m_state.projectPath());
    }
    if (m_state.projectDirty())
    {
        title += " *";
    }
    return title;
}

// ---- Menus, shortcuts and dialogs -------------------------------------------------------

void CompareUi::drawMenuBar()
{
    if (!ImGui::BeginMenuBar())
    {
        return;
    }
    if (ImGui::BeginMenu("File"))
    {
        if (ImGui::MenuItem("New project"))
        {
            requestNew();
        }
        if (ImGui::MenuItem("Open project...", "Ctrl+O"))
        {
            requestOpenDialog();
        }
        if (ImGui::MenuItem("Save project", "Ctrl+S"))
        {
            saveProjectOrSaveAs();
        }
        if (ImGui::MenuItem("Save project as..."))
        {
            beginSaveProjectAs();
        }
        ImGui::Separator();
        const bool full = m_state.size() >= static_cast<size_t>(kMaxSources);
        if (ImGui::MenuItem("Add media...", nullptr, false, !full))
        {
            requestAddMedia();
        }
        ImGui::Separator();
        if (ImGui::BeginMenu("Export"))
        {
            if (ImGui::MenuItem("Current view PNG..."))
            {
                beginSave(DialogPurpose::ExportView, "view.png", "png");
            }
            if (ImGui::MenuItem("Diff map PNG (full resolution)..."))
            {
                if (m_state.hasPair())
                {
                    beginSave(DialogPurpose::ExportDiff, "diff.png", "png");
                }
                else
                {
                    m_state.setStatus("The diff map needs A and B visible");
                }
            }
            if (ImGui::MenuItem("FLIP map PNG (full resolution)..."))
            {
                if (m_state.flip())
                {
                    beginSave(DialogPurpose::ExportFlip, "flip.png", "png");
                }
                else
                {
                    m_state.setStatus(flipUnavailable(m_state));
                }
            }
            if (ImGui::MenuItem("Exposure map PNG (full resolution)..."))
            {
                const std::optional<FlipGpuResult> & flipResult = m_state.flip();
                if (flipResult && flipResult->hdr)
                {
                    beginSave(DialogPurpose::ExportExposure, "exposure.png", "png");
                }
                else
                {
                    const std::string & reason = m_state.flipReason();
                    m_state.setStatus(reason.empty() ? std::string("no HDR-FLIP result")
                                                     : std::string("no HDR-FLIP result: ") + reason);
                }
            }
            if (ImGui::MenuItem("Padded pane images..."))
            {
                beginSave(DialogPurpose::ExportPanes, "panes.png", "png");
            }
            if (ImGui::MenuItem("Report JSON..."))
            {
                beginSave(DialogPurpose::ExportReport, "report.json", "json");
            }
            if (ImGui::MenuItem("Report CSV..."))
            {
                beginSave(DialogPurpose::ExportReport, "report.csv", "csv");
            }
            ImGui::EndMenu();
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Quit"))
        {
            requestQuit();
        }
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("View"))
    {
        if (ImGui::MenuItem("Zoom 100%", "1"))
        {
            setZoomPercent(m_state.viewLayout(), 100.0);
            m_state.viewChanged();
        }
        if (ImGui::MenuItem("Fit", "F"))
        {
            fitView();
        }
        bool nearest = m_state.settings().nearest;
        if (ImGui::MenuItem("Nearest sampling", nullptr, &nearest))
        {
            m_state.setNearest(nearest);
        }
        bool live = m_state.live();
        if (ImGui::MenuItem("Live scene updates", nullptr, &live))
        {
            m_state.setLive(live);
        }
        if (ImGui::IsItemHovered(ImGuiHoveredFlags_ForTooltip))
        {
            ImGui::SetTooltip("Off: scenes and metrics update only when the camera rests.");
        }
        ImGui::EndMenu();
    }
    if (ImGui::BeginMenu("Help"))
    {
        if (ImGui::MenuItem("About Sabercat Compare"))
        {
            m_aboutOpen = true;
        }
        ImGui::EndMenu();
    }
    ImGui::EndMenuBar();
}

// Fit: the padded extent fills the smallest pane and is centered in every pane.
void CompareUi::fitView()
{
    setZoomFit(m_state.viewLayout());
    centerView(m_state.viewLayout());
    m_state.viewChanged();
}

void CompareUi::handleShortcuts()
{
    ImGuiIO & io = ImGui::GetIO();
    if (io.WantTextInput)
    {
        return;
    }
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_S))
    {
        saveProjectOrSaveAs();
    }
    if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_O))
    {
        requestOpenDialog();
    }
    if (io.KeyCtrl)
    {
        return;
    }
    // The transport keys need a visible video: Space plays or pauses, the arrows step one frame.
    if (m_state.hasVisibleVideo())
    {
        if (ImGui::IsKeyPressed(ImGuiKey_Space, false))
        {
            m_state.togglePlay();
        }
        if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow))
        {
            m_state.stepFrames(-1);
        }
        if (ImGui::IsKeyPressed(ImGuiKey_RightArrow))
        {
            m_state.stepFrames(1);
        }
    }
    if (ImGui::IsKeyPressed(ImGuiKey_F))
    {
        fitView();
    }
    if (ImGui::IsKeyPressed(ImGuiKey_1))
    {
        setZoomPercent(m_state.viewLayout(), 100.0);
        m_state.viewChanged();
    }
}

bool CompareUi::beginOpen(DialogPurpose purpose, const std::string & title, bool imagesOnly)
{
    if (m_dialogs.pending())
    {
        m_state.setStatus("Finish the open file dialog first");
        return false;
    }
    const bool started = m_dialogs.startOpen(m_window, imagesOnly ? imageFilters() : projectFilters(), title,
                                             kDialogTag);
    if (started)
    {
        m_purpose = purpose;
    }
    return started;
}

bool CompareUi::beginSave(DialogPurpose purpose, const std::string & defaultName, const std::string & extension)
{
    if (m_dialogs.pending())
    {
        m_state.setStatus("Finish the open file dialog first");
        return false;
    }
    const std::vector<app::FileFilter> filters = purpose == DialogPurpose::ExportReport
        ? reportFilters()
        : (extension == "png" ? pngFilters() : projectFilters());
    const bool started = m_dialogs.startSave(m_window, filters, defaultName, extension, kDialogTag);
    if (started)
    {
        m_purpose = purpose;
    }
    return started;
}

void CompareUi::beginSaveProjectAs()
{
    const std::string name = m_state.projectPath().empty() ? std::string("Untitled.sabercat-compare")
                                                           : fileNameUtf8(m_state.projectPath());
    beginSave(DialogPurpose::SaveProject, name, "sabercat-compare");
}

void CompareUi::requestAddMedia()
{
    if (m_state.size() >= static_cast<size_t>(kMaxSources))
    {
        m_state.setStatus("At most 8 sources: remove one before adding another");
        return;
    }
    beginOpen(DialogPurpose::AddMedia, "Add media", true);
}

void CompareUi::requestLocate(size_t index)
{
    m_locateIndex = index;
    beginOpen(DialogPurpose::Locate, "Locate the image", true);
}

void CompareUi::requestNew()
{
    if (m_state.projectDirty())
    {
        m_pending = PendingAction::NewProject;
        m_discardOpen = true;
        return;
    }
    idleGpu();
    m_state.newProject();
}

void CompareUi::requestOpenDialog()
{
    if (m_state.projectDirty())
    {
        m_pending = PendingAction::OpenDialog;
        m_discardOpen = true;
        return;
    }
    beginOpen(DialogPurpose::OpenProject, "Open project", false);
}

void CompareUi::requestOpenPath(const std::string & path)
{
    if (m_state.projectDirty())
    {
        m_pending = PendingAction::OpenPath;
        m_pendingPath = path;
        m_discardOpen = true;
        return;
    }
    doOpenProject(path);
}

void CompareUi::idleGpu()
{
    vkDeviceWaitIdle(m_device);
}

void CompareUi::doOpenProject(const std::string & utf8Path)
{
    // Opening replaces the sources, and with them any scene that may still be in flight.
    idleGpu();
    std::string error;
    if (!m_state.openProject(utf8Path, error))
    {
        m_error = "The project could not be opened.\n\n" + error;
        m_errorOpen = true;
        return;
    }
    if (!m_state.notices().empty())
    {
        m_noticesOpen = true;
    }
}

void CompareUi::saveProjectOrSaveAs()
{
    if (m_state.projectPath().empty())
    {
        beginSaveProjectAs();
        return;
    }
    saveTo(m_state.projectPath());
}

bool CompareUi::saveTo(const std::string & utf8Path)
{
    std::string error;
    if (m_state.saveProject(utf8Path, error))
    {
        continuePending();
        return true;
    }
    m_error = "The project could not be saved.\n\n" + error;
    m_errorOpen = true;
    return false;
}

// Runs the action that was waiting for the unsaved-changes answer.
void CompareUi::continuePending()
{
    const PendingAction action = m_pending;
    const std::string path = m_pendingPath;
    m_pending = PendingAction::None;
    m_pendingPath.clear();
    switch (action)
    {
    case PendingAction::None:
        break;
    case PendingAction::NewProject:
        idleGpu();
        m_state.newProject();
        break;
    case PendingAction::OpenDialog:
        beginOpen(DialogPurpose::OpenProject, "Open project", false);
        break;
    case PendingAction::OpenPath:
        doOpenProject(path);
        break;
    case PendingAction::Quit:
        m_quitRequested = true;
        break;
    }
}

void CompareUi::runExport(const std::string & label, const std::string & utf8Path, DialogPurpose purpose)
{
    try
    {
        std::string message;
        switch (purpose)
        {
        case DialogPurpose::ExportView:
            message = exportCurrentView(m_compositor, utf8Path);
            break;
        case DialogPurpose::ExportDiff:
            message = exportDiffMap(m_state, utf8Path);
            break;
        case DialogPurpose::ExportFlip:
            message = exportFlipMap(m_state, utf8Path);
            break;
        case DialogPurpose::ExportExposure:
            message = exportExposureMap(m_state, utf8Path);
            break;
        case DialogPurpose::ExportPanes:
            message = exportPaneImages(m_state, utf8Path);
            break;
        case DialogPurpose::ExportReport:
            message = exportReport(m_state, utf8Path);
            break;
        default:
            throw std::logic_error("not an export");
        }
        m_state.setStatus(message);
        Logger::info("Compare: " + message);
    }
    catch (const std::exception & e)
    {
        m_state.setStatus(label + " failed: " + e.what());
        Logger::error("Compare: " + label + " failed: " + e.what());
    }
}

void CompareUi::pollDialog()
{
    const std::optional<std::optional<std::string>> result = m_dialogs.poll();
    if (!result.has_value())
    {
        return;
    }
    const DialogPurpose purpose = m_purpose;
    m_purpose = DialogPurpose::None;
    if (!result->has_value())
    {
        // Canceled. A canceled save leaves a pending action waiting for nothing.
        if (purpose == DialogPurpose::SaveProject)
        {
            m_pending = PendingAction::None;
        }
        return;
    }
    const std::string path = **result;
    switch (purpose)
    {
    case DialogPurpose::None:
        break;
    case DialogPurpose::AddMedia:
        m_state.addMedia(path);
        break;
    case DialogPurpose::OpenProject:
        doOpenProject(path);
        break;
    case DialogPurpose::SaveProject:
        saveTo(path);
        break;
    case DialogPurpose::ExportView:
        runExport("Export", path, purpose);
        break;
    case DialogPurpose::ExportDiff:
        runExport("Export", path, purpose);
        break;
    case DialogPurpose::ExportFlip:
        runExport("Export", path, purpose);
        break;
    case DialogPurpose::ExportExposure:
        runExport("Export", path, purpose);
        break;
    case DialogPurpose::ExportPanes:
        runExport("Export", path, purpose);
        break;
    case DialogPurpose::ExportReport:
        runExport("Report", path, purpose);
        break;
    case DialogPurpose::Locate:
        idleGpu();
        m_state.relocate(m_locateIndex, path);
        break;
    }
}

// ---- Modals -----------------------------------------------------------------------------

void CompareUi::drawModals()
{
    if (m_discardOpen)
    {
        ImGui::OpenPopup("Unsaved changes");
        m_discardOpen = false;
    }
    if (ImGui::BeginPopupModal("Unsaved changes", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::TextUnformatted("The project has unsaved changes.");
        ImGui::Spacing();
        if (ImGui::Button("Save"))
        {
            ImGui::CloseCurrentPopup();
            if (m_state.projectPath().empty())
            {
                beginSaveProjectAs(); // the pending action runs after the save
            }
            else if (saveTo(m_state.projectPath()) == false)
            {
                m_pending = PendingAction::None;
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Discard"))
        {
            ImGui::CloseCurrentPopup();
            continuePending();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel"))
        {
            ImGui::CloseCurrentPopup();
            m_pending = PendingAction::None;
            m_pendingPath.clear();
        }
        ImGui::EndPopup();
    }

    if (m_errorOpen)
    {
        ImGui::OpenPopup("Error");
        m_errorOpen = false;
    }
    if (ImGui::BeginPopupModal("Error", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::PushTextWrapPos(420.0f);
        ImGui::TextUnformatted(m_error.c_str());
        ImGui::PopTextWrapPos();
        ImGui::Spacing();
        if (ImGui::Button("OK"))
        {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    if (m_noticesOpen)
    {
        ImGui::OpenPopup("Project notes");
        m_noticesOpen = false;
    }
    if (ImGui::BeginPopupModal("Project notes", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::PushTextWrapPos(460.0f);
        for (const std::string & notice : m_state.notices())
        {
            ImGui::TextUnformatted(("- " + notice).c_str());
        }
        ImGui::PopTextWrapPos();
        ImGui::Spacing();
        if (ImGui::Button("OK"))
        {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

// ---- Transport --------------------------------------------------------------------------

float CompareUi::transportHeight() const
{
    if (!m_state.hasVideo())
    {
        return 0.0f;
    }
    // Two rows of controls, and a third row for the HDR-FLIP note while the result is stale.
    const float rows = m_state.hdrStaleForVideo() ? 3.0f : 2.0f;
    return rows * ImGui::GetFrameHeightWithSpacing();
}

// Play, step, the seek slider, the time, one frame label per visible video, the speed and the
// loop. Without a visible video the bar is greyed out, and the Play button says why.
void CompareUi::drawTransportBar()
{
    const Transport & transport = m_state.transport();
    const bool usable = m_state.hasVisibleVideo();
    ImGui::PushID("transport");
    ImGui::BeginDisabled(!usable);

    // Row 1: play, step and the seek slider.
    if (ImGui::Button(transport.playing ? "Pause" : "Play"))
    {
        m_state.togglePlay();
    }
    if (!usable && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
    {
        ImGui::SetTooltip("show a video");
    }
    ImGui::SameLine();
    if (ImGui::Button("Step back"))
    {
        m_state.stepFrames(-1);
    }
    ImGui::SameLine();
    if (ImGui::Button("Step forward"))
    {
        m_state.stepFrames(1);
    }
    ImGui::SameLine();
    // With no duration the slider keeps a unit range, so it never divides by zero.
    const float maxSeconds = transport.duration > 0.0 ? static_cast<float>(transport.duration) : 1.0f;
    float seconds = static_cast<float>(transport.time);
    ImGui::SetNextItemWidth(std::max(120.0f, ImGui::GetContentRegionAvail().x));
    if (ImGui::SliderFloat("##seek", &seconds, 0.0f, maxSeconds, "%.3f s"))
    {
        m_state.seek(static_cast<double>(seconds));
    }

    // Row 2: the time, the frame of each visible video, the speed and the loop.
    ImGui::Text("t %.3f s / %.3f s", transport.time, transport.duration);
    for (size_t i = 0; i < m_state.size(); ++i)
    {
        const MediaEntry & entry = m_state.entries()[i];
        if (!entry.visible || !entry.ready() || entry.media->kind() != project::SourceKind::Video)
        {
            continue;
        }
        const VideoSource * video = m_state.videoOf(i);
        if (video == nullptr)
        {
            continue;
        }
        const char * label = entry.role == project::SourceRole::A ? "A"
            : (entry.role == project::SourceRole::B ? "B" : entry.name.c_str());
        ImGui::SameLine();
        ImGui::Text("%s f%lld/%lld", label, static_cast<long long>(m_state.videoFrameIndex(i)),
                    static_cast<long long>(video->frameCount()));
    }
    ImGui::SameLine();
    char speedText[32];
    std::snprintf(speedText, sizeof(speedText), "%gx", transport.speed);
    ImGui::SetNextItemWidth(90.0f);
    if (ImGui::BeginCombo("Speed", speedText))
    {
        for (const double preset : kSpeedPresets)
        {
            char presetText[32];
            std::snprintf(presetText, sizeof(presetText), "%gx", preset);
            const bool selected = std::fabs(transport.speed - preset) < 1e-9;
            if (ImGui::Selectable(presetText, selected))
            {
                m_state.setSpeed(preset);
            }
        }
        ImGui::EndCombo();
    }
    ImGui::SameLine();
    bool loop = transport.loop;
    if (ImGui::Checkbox("Loop", &loop))
    {
        m_state.setTransportLoop(loop);
    }
    ImGui::EndDisabled();

    // Row 3: the HDR-FLIP note, while a frame change has made its result stale.
    if (m_state.hdrStaleForVideo())
    {
        ImGui::TextColored(kWarningColor, "HDR-FLIP not run during playback");
    }
    ImGui::PopID();
}

void CompareUi::drawStatusLine()
{
    ImGui::TextUnformatted(m_state.status().c_str());
}

// ---- Side panel -------------------------------------------------------------------------

void CompareUi::drawSidePanel()
{
    const size_t count = m_state.size();
    ImGui::Text("Media (%zu of %d)", count, kMaxSources);
    const bool full = count >= static_cast<size_t>(kMaxSources);
    ImGui::BeginDisabled(full || m_dialogs.pending());
    if (ImGui::Button("Add media..."))
    {
        requestAddMedia();
    }
    ImGui::EndDisabled();
    if (full)
    {
        ImGui::TextColored(kWarningColor, "Limit of %d sources reached. Remove one to add another.", kMaxSources);
    }
    ImGui::Separator();

    m_pendingRemove = -1;
    for (size_t i = 0; i < count; ++i)
    {
        drawSourceRow(i);
    }
    if (m_pendingRemove >= 0)
    {
        // The removed source may be a scene whose renderer the GPU still uses.
        idleGpu();
        m_state.removeSource(static_cast<size_t>(m_pendingRemove));
        m_pendingRemove = -1;
    }
    if (count == 0)
    {
        ImGui::TextDisabled("No media yet.");
    }
    ImGui::Separator();
    drawCompareSection();
    ImGui::Separator();
    drawViewSection();
    ImGui::Separator();
    drawProbeSection();
}

void CompareUi::drawSourceRow(size_t index)
{
    const MediaEntry & entry = m_state.entries()[index];
    ImGui::PushID(static_cast<int>(index));

    const bool ready = entry.ready();
    bool visible = entry.visible;
    ImGui::BeginDisabled(!ready && !visible);
    if (ImGui::Checkbox("##visible", &visible))
    {
        m_state.setVisible(index, visible);
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::TextUnformatted(entry.name.c_str());
    if (ready)
    {
        ImGui::SameLine();
        ImGui::TextDisabled("%ux%u", entry.media->width(), entry.media->height());
    }
    if (entry.media->kind() == project::SourceKind::Scene)
    {
        ImGui::SameLine();
        ImGui::TextDisabled("scene");
    }
    if (entry.media->kind() == project::SourceKind::Video)
    {
        ImGui::SameLine();
        ImGui::TextDisabled("video");
    }

    int role = entry.role == project::SourceRole::A ? 1 : (entry.role == project::SourceRole::B ? 2 : 0);
    ImGui::SetNextItemWidth(72.0f);
    ImGui::BeginDisabled(!entry.visible);
    if (ImGui::Combo("##role", &role, kRoleNames, 3))
    {
        const project::SourceRole chosen = role == 1 ? project::SourceRole::A
            : (role == 2 ? project::SourceRole::B : project::SourceRole::None);
        m_state.setRole(index, chosen);
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::SmallButton("Settings"))
    {
        ImGui::OpenPopup("settings");
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("X"))
    {
        m_pendingRemove = static_cast<int32_t>(index);
    }
    // The ID scan opens this entry's settings popup to sweep it (smokeOpenSettings).
    if (m_smokeSettings == static_cast<int32_t>(index))
    {
        ImGui::OpenPopup("settings");
    }
    drawSourceSettings(index);

    if (!ready)
    {
        ImGui::TextColored(kErrorColor, "%s", entry.message.c_str());
        if (ImGui::Button("Locate..."))
        {
            requestLocate(index);
        }
    }
    if (entry.otherDriveWarning)
    {
        ImGui::TextColored(kWarningColor, "Stored on another drive: its path is kept absolute.");
    }
    ImGui::Spacing();
    ImGui::PopID();
}

void CompareUi::drawSourceSettings(size_t index)
{
    if (!ImGui::BeginPopup("settings"))
    {
        return;
    }
    const MediaEntry & entry = m_state.entries()[index];
    ImGui::TextUnformatted(entry.name.c_str());
    ImGui::TextDisabled("%s", entry.path.c_str());
    ImGui::Separator();

    int offsetX = entry.offsetX;
    int offsetY = entry.offsetY;
    ImGui::SetNextItemWidth(110.0f);
    if (ImGui::InputInt("Offset X", &offsetX))
    {
        m_state.setOffset(index, offsetX, entry.offsetY);
    }
    ImGui::SetNextItemWidth(110.0f);
    if (ImGui::InputInt("Offset Y", &offsetY))
    {
        m_state.setOffset(index, entry.offsetX, offsetY);
    }

    float background[3] = {entry.background.r, entry.background.g, entry.background.b};
    if (ImGui::ColorEdit3("Background", background))
    {
        m_state.setBackground(index, project::Rgb{background[0], background[1], background[2]});
    }

    if (entry.ready() && entry.media->encoding() == Encoding::Linear)
    {
        float exposure = entry.exposure;
        ImGui::SetNextItemWidth(160.0f);
        if (ImGui::SliderFloat("Exposure", &exposure, -5.0f, 5.0f, "%.2f stops"))
        {
            m_state.setExposure(index, exposure);
        }
        int toneMapper = static_cast<int>(entry.toneMapper);
        ImGui::SetNextItemWidth(160.0f);
        if (ImGui::Combo("Tone mapper", &toneMapper, kToneMapperNames, 3))
        {
            m_state.setToneMapper(index, static_cast<project::ToneMapper>(toneMapper));
        }
    }

    if (entry.media->kind() == project::SourceKind::Scene)
    {
        drawSceneSettings(index);
    }
    if (entry.ready() && entry.media->kind() == project::SourceKind::Video)
    {
        drawVideoSettings(index);
    }

    int role = entry.role == project::SourceRole::A ? 1 : (entry.role == project::SourceRole::B ? 2 : 0);
    ImGui::SetNextItemWidth(110.0f);
    ImGui::BeginDisabled(!entry.visible);
    if (ImGui::Combo("Role", &role, kRoleNames, 3))
    {
        const project::SourceRole chosen = role == 1 ? project::SourceRole::A
            : (role == 2 ? project::SourceRole::B : project::SourceRole::None);
        m_state.setRole(index, chosen);
    }
    ImGui::EndDisabled();

    ImGui::Separator();
    if (!entry.ready() && ImGui::Button("Locate..."))
    {
        requestLocate(index);
    }
    if (ImGui::Button("Remove source"))
    {
        m_pendingRemove = static_cast<int32_t>(index);
        ImGui::CloseCurrentPopup();
    }
    ImGui::EndPopup();
}

// The controls only a scene source has: its render settings, and the render size shared by
// every scene.
void CompareUi::drawSceneSettings(size_t index)
{
    const MediaEntry & entry = m_state.entries()[index];
    ImGui::Separator();
    ImGui::TextUnformatted("Scene");
    // A failed reload keeps the last good render on screen; the error says why the file was not used.
    if (!entry.reloadError.empty())
    {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(kWarningColor, "%s", entry.reloadError.c_str());
        ImGui::PopTextWrapPos();
    }
    bool flip = entry.flipUpAxis;
    if (ImGui::Checkbox("Up axis flip", &flip))
    {
        m_state.setFlipUpAxis(index, flip);
    }
    bool antialias = entry.antialias;
    if (ImGui::Checkbox("Antialias", &antialias))
    {
        m_state.setAntialias(index, antialias);
    }
    const char * const shNames[] = {"0", "1", "2", "3", "Full (3)"};
    int shIndex = entry.shCap < 0 ? 4 : entry.shCap;
    ImGui::SetNextItemWidth(160.0f);
    if (ImGui::Combo("SH cap", &shIndex, shNames, 5))
    {
        m_state.setShCap(index, shIndex == 4 ? -1 : shIndex);
    }

    std::array<const char *, app::kDebugViews.size()> viewNames{};
    int viewIndex = 0;
    for (size_t i = 0; i < app::kDebugViews.size(); ++i)
    {
        viewNames[i] = app::kDebugViews[i].name;
        if (app::kDebugViews[i].view == entry.debugView)
        {
            viewIndex = static_cast<int>(i);
        }
    }
    ImGui::SetNextItemWidth(160.0f);
    if (ImGui::Combo("Debug view", &viewIndex, viewNames.data(), static_cast<int>(viewNames.size())))
    {
        m_state.setDebugView(index, app::kDebugViews[static_cast<size_t>(viewIndex)].view);
    }

    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("PLY and SPZ of one capture differ in antialias and up-axis defaults: set both the same");
    ImGui::PopTextWrapPos();

    // The size applies when a field is left after an edit, to every scene: the camera keeps its pose.
    ImGui::Text("Render size %ux%u", m_state.renderWidth(), m_state.renderHeight());
    if (m_state.renderSizePending())
    {
        ImGui::TextDisabled("Render size change waits until the camera rests");
    }
    int width = static_cast<int>(m_state.renderWidth());
    int height = static_cast<int>(m_state.renderHeight());
    ImGui::SetNextItemWidth(110.0f);
    ImGui::InputInt("Width", &width, 0, 0);
    bool resize = ImGui::IsItemDeactivatedAfterEdit();
    ImGui::SameLine();
    ImGui::SetNextItemWidth(110.0f);
    ImGui::InputInt("Height", &height, 0, 0);
    resize = ImGui::IsItemDeactivatedAfterEdit() || resize;
    if (resize)
    {
        m_state.setRenderSize(static_cast<uint32_t>(std::clamp(width, kMinRenderSize, kMaxRenderSize)),
                              static_cast<uint32_t>(std::clamp(height, kMinRenderSize, kMaxRenderSize)));
    }
    if (ImGui::Button("Frame scene"))
    {
        m_state.frameScene(index);
    }
}

// The controls only a video source has: its stream facts, its trim, its loop and its time offset.
// Each field follows the state unless it is being typed in, and applies when it is left after an
// edit. A refused value leaves the state as it was, and the status line says why.
void CompareUi::drawVideoSettings(size_t index)
{
    VideoSource * video = m_state.videoOf(index);
    if (video == nullptr)
    {
        return;
    }
    const VideoInfo & info = video->videoInfo();
    ImGui::Separator();
    ImGui::TextUnformatted("Video");
    ImGui::PushTextWrapPos(0.0f);
    ImGui::TextDisabled("%ux%u  %.3f fps  %lld frames  %.3f s", video->width(), video->height(), video->fps(),
                        static_cast<long long>(video->frameCount()), video->durationSeconds());
    if (info.rotationDegrees != 0)
    {
        ImGui::TextDisabled("Rotation metadata (%u deg) is ignored", info.rotationDegrees);
    }
    if (info.yuvMatrix == kMfMatrixBt601 || info.nominalRange == kMfNominalRange0To255)
    {
        ImGui::TextDisabled("Stream is tagged BT.601/full range; shown as BT.709 limited");
    }
    ImGui::PopTextWrapPos();

    if (!m_videoTrimInEditing)
    {
        m_videoTrimInField = video->trimIn();
    }
    ImGui::SetNextItemWidth(120.0f);
    ImGui::InputDouble("Trim in (s)", &m_videoTrimInField, 0.0, 0.0, "%.3f");
    if (ImGui::IsItemDeactivatedAfterEdit())
    {
        m_state.setVideoTrim(index, m_videoTrimInField, video->trimToEnd() ? -1.0 : video->trimOut());
    }
    m_videoTrimInEditing = ImGui::IsItemActive();

    // The trim out field shows the resolved end of the clip; "To end" stores it as the clip's end.
    if (!m_videoTrimOutEditing)
    {
        m_videoTrimOutField = video->trimOut();
    }
    ImGui::SetNextItemWidth(120.0f);
    ImGui::InputDouble("Trim out (s)", &m_videoTrimOutField, 0.0, 0.0, "%.3f");
    if (ImGui::IsItemDeactivatedAfterEdit())
    {
        m_state.setVideoTrim(index, video->trimIn(), m_videoTrimOutField);
    }
    m_videoTrimOutEditing = ImGui::IsItemActive();
    ImGui::SameLine();
    if (ImGui::SmallButton("To end"))
    {
        m_state.setVideoTrim(index, video->trimIn(), -1.0);
    }

    bool loop = video->loop();
    if (ImGui::Checkbox("Loop", &loop))
    {
        m_state.setVideoLoop(index, loop);
    }

    if (!m_videoOffsetSecondsEditing)
    {
        m_videoOffsetSecondsField = video->timeOffset();
    }
    ImGui::SetNextItemWidth(120.0f);
    ImGui::InputDouble("Time offset (s)", &m_videoOffsetSecondsField, 0.0, 0.0, "%.3f");
    if (ImGui::IsItemDeactivatedAfterEdit())
    {
        m_state.setVideoTimeOffset(index, m_videoOffsetSecondsField);
    }
    m_videoOffsetSecondsEditing = ImGui::IsItemActive();

    if (!m_videoOffsetFramesEditing)
    {
        m_videoOffsetFramesField = static_cast<int>(std::lround(video->timeOffset() * video->fps()));
    }
    ImGui::SetNextItemWidth(120.0f);
    ImGui::InputInt("Time offset (frames)", &m_videoOffsetFramesField, 0, 0);
    if (ImGui::IsItemDeactivatedAfterEdit())
    {
        m_state.setVideoTimeOffsetFrames(index, static_cast<int32_t>(m_videoOffsetFramesField));
    }
    m_videoOffsetFramesEditing = ImGui::IsItemActive();
}

void CompareUi::drawCompareSection()
{
    ImGui::TextUnformatted("Compare");
    const int32_t a = m_state.sourceA();
    const int32_t b = m_state.sourceB();
    if (a >= 0 && b >= 0)
    {
        const MediaEntry & ea = m_state.entries()[static_cast<size_t>(a)];
        const MediaEntry & eb = m_state.entries()[static_cast<size_t>(b)];
        const uint32_t wa = ea.media->width();
        const uint32_t ha = ea.media->height();
        const uint32_t wb = eb.media->width();
        const uint32_t hb = eb.media->height();
        const PixelRect overlap = overlapRect(m_state.layout(), a, b);
        if (wa != wb || ha != hb)
        {
            ImGui::PushTextWrapPos(0.0f);
            ImGui::TextColored(kWarningColor, "A is %ux%u, B is %ux%u; metrics cover the %dx%d overlap", wa, ha, wb,
                               hb, overlap.width, overlap.height);
            ImGui::PopTextWrapPos();
        }
        if (overlap.width <= 0 || overlap.height <= 0)
        {
            ImGui::TextColored(kWarningColor, "A and B do not overlap.");
        }
    }

    // Linear values reach PSNR and MSE through the display transform (clamped, then sRGB-encoded),
    // and HDR mode works on them as linear values, so the line says so.
    const auto isLinear = [this](int32_t index)
    {
        if (index < 0 || static_cast<size_t>(index) >= m_state.size())
        {
            return false;
        }
        const MediaEntry & entry = m_state.entries()[static_cast<size_t>(index)];
        return entry.ready() && entry.media->encoding() == Encoding::Linear;
    };
    const bool displayTransform = m_state.flipMode() == project::FlipMode::Hdr || isLinear(a) || isLinear(b);
    const std::string encodingWarning = m_state.pairEncodingWarning();
    if (!encodingWarning.empty())
    {
        ImGui::PushTextWrapPos(0.0f);
        ImGui::TextColored(kWarningColor, "%s", encodingWarning.c_str());
        ImGui::PopTextWrapPos();
    }

    // While the live policy defers the metrics (the camera is moving), the values below are the last
    // ones computed, and they are marked stale until the camera rests.
    if (m_state.metricsPending())
    {
        ImGui::TextDisabled("stale: updating when the camera rests");
    }
    if (m_state.metrics())
    {
        const MetricsResult & metrics = *m_state.metrics();
        if (metrics.mse == 0.0)
        {
            ImGui::Text("identical (MSE 0, PSNR inf)");
        }
        else if (displayTransform)
        {
            ImGui::Text("PSNR %.2f dB (on the display transform)   MSE %.6g", metrics.psnrDb, metrics.mse);
        }
        else
        {
            ImGui::Text("PSNR %.2f dB   MSE %.6g", metrics.psnrDb, metrics.mse);
        }
        const std::optional<FlipGpuResult> & flipForLinear = m_state.flip();
        if (flipForLinear && flipForLinear->hdr && std::isfinite(flipForLinear->hdr->linearMse))
        {
            ImGui::Text("linear MSE %.6g", flipForLinear->hdr->linearMse);
        }
        ImGui::TextDisabled("%llu pixels compared", static_cast<unsigned long long>(metrics.pixelCount));
    }
    else if (!m_state.metricsReason().empty())
    {
        ImGui::TextDisabled("%s", m_state.metricsReason().c_str());
    }

    // FLIP: the LDR or HDR mode and, in HDR mode, its controls. Then the mean over the overlap,
    // or why there is none, and the pixels-per-degree control. The ids here are kept apart by PushID.
    ImGui::PushID("flip");
    const float ppd = m_state.flipPpd();
    const bool hdrMode = m_state.flipMode() == project::FlipMode::Hdr;
    const bool hdrAvailable = m_state.hdrAvailable();
    const std::optional<FlipGpuResult> & flipResult = m_state.flip();
    ImGui::TextUnformatted("FLIP mode");
    if (ImGui::RadioButton("LDR", !hdrMode))
    {
        m_state.setFlipMode(project::FlipMode::Ldr);
    }
    ImGui::SameLine();
    ImGui::BeginDisabled(!hdrAvailable);
    if (ImGui::RadioButton("HDR", hdrMode))
    {
        m_state.setFlipMode(project::FlipMode::Hdr);
    }
    ImGui::EndDisabled();
    if (!hdrAvailable && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
    {
        ImGui::SetTooltip("HDR needs a Linear (EXR) source in the A/B pair");
    }

    if (hdrMode)
    {
        const FlipHdrSettings & hdr = m_state.hdrSettings();
        // What a field takes when it leaves automatic: the last HDR result's value, else a default.
        const bool haveResult = flipResult.has_value() && flipResult->hdr.has_value();
        const float seedStart = haveResult ? flipResult->hdr->start : 0.0f;
        const float seedStop = haveResult ? flipResult->hdr->stop : 1.0f;
        const int32_t seedCount = haveResult ? flipResult->hdr->count : 16;

        int tone = static_cast<int>(hdr.toneMapper);
        ImGui::SetNextItemWidth(160.0f);
        if (ImGui::Combo("FLIP tone mapper", &tone, kToneMapperNames, 3))
        {
            m_state.setHdrToneMapper(static_cast<compare::ToneMapper>(tone));
        }

        bool autoRange = !hdr.start.has_value() && !hdr.stop.has_value();
        if (ImGui::Checkbox("Auto exposure range", &autoRange))
        {
            if (autoRange)
            {
                m_state.setHdrRange(std::nullopt, std::nullopt, hdr.count);
            }
            else
            {
                m_hdrStartField = seedStart;
                m_hdrStopField = seedStop;
                m_state.setHdrRange(seedStart, seedStop, hdr.count);
            }
        }
        // The fields show when the range is set by hand; each commits when it is left after an edit.
        if (hdr.start.has_value() || hdr.stop.has_value())
        {
            if (!m_hdrStartEditing)
            {
                m_hdrStartField = hdr.start.value_or(seedStart);
            }
            ImGui::SetNextItemWidth(120.0f);
            ImGui::InputFloat("Start (stops)", &m_hdrStartField, 0.0f, 0.0f, "%.4f");
            if (ImGui::IsItemDeactivatedAfterEdit())
            {
                m_state.setHdrRange(m_hdrStartField, hdr.stop, hdr.count);
            }
            m_hdrStartEditing = ImGui::IsItemActive();

            if (!m_hdrStopEditing)
            {
                m_hdrStopField = hdr.stop.value_or(seedStop);
            }
            ImGui::SetNextItemWidth(120.0f);
            ImGui::InputFloat("Stop (stops)", &m_hdrStopField, 0.0f, 0.0f, "%.4f");
            if (ImGui::IsItemDeactivatedAfterEdit())
            {
                m_state.setHdrRange(hdr.start, m_hdrStopField, hdr.count);
            }
            m_hdrStopEditing = ImGui::IsItemActive();
        }

        bool autoCount = !hdr.count.has_value();
        if (ImGui::Checkbox("Auto count", &autoCount))
        {
            if (autoCount)
            {
                m_state.setHdrRange(hdr.start, hdr.stop, std::nullopt);
            }
            else
            {
                m_hdrCountField = seedCount;
                m_state.setHdrRange(hdr.start, hdr.stop, m_hdrCountField);
            }
        }
        if (hdr.count.has_value())
        {
            if (!m_hdrCountEditing)
            {
                m_hdrCountField = *hdr.count;
            }
            ImGui::SetNextItemWidth(120.0f);
            ImGui::InputInt("Exposures", &m_hdrCountField, 0, 0);
            if (ImGui::IsItemDeactivatedAfterEdit())
            {
                m_hdrCountField = std::clamp(m_hdrCountField, kMinHdrExposures, kMaxHdrExposures);
                m_state.setHdrRange(hdr.start, hdr.stop, m_hdrCountField);
            }
            m_hdrCountEditing = ImGui::IsItemActive();
        }

        bool runOnChange = m_state.hdrAuto();
        if (ImGui::Checkbox("Run on change", &runOnChange))
        {
            m_state.setHdrAuto(runOnChange);
        }
        const bool running = m_state.hdrRunning();
        ImGui::BeginDisabled(!hdrAvailable || running);
        if (ImGui::Button("Run HDR-FLIP"))
        {
            m_state.requestHdr();
        }
        ImGui::EndDisabled();
        if (running)
        {
            ImGui::ProgressBar(m_state.hdrProgress(), ImVec2(240.0f, 0.0f));
            ImGui::SameLine();
            if (ImGui::Button("Cancel"))
            {
                m_state.cancelHdr();
            }
        }
    }

    if (flipResult)
    {
        const FlipGpuResult & flip = *flipResult;
        if (flip.hdr)
        {
            const FlipHdrInfo & info = *flip.hdr;
            ImGui::Text("FLIP (HDR, %s) %.6f (ppd %s, exposures %.4f..%.4f, %d)", toneMapperName(info.toneMapper),
                        flip.mean, ppdText(ppd).c_str(), info.start, info.stop, static_cast<int>(info.count));
        }
        else
        {
            ImGui::Text("FLIP %.6f (ppd %s)", flip.mean, ppdText(ppd).c_str());
        }
        if (flip.nonFiniteSamples > 0)
        {
            ImGui::TextColored(kWarningColor, "%llu non-finite samples treated as 0 or 1",
                               static_cast<unsigned long long>(flip.nonFiniteSamples));
        }
    }
    else if (!m_state.flipReason().empty())
    {
        ImGui::TextDisabled("%s", flipUnavailable(m_state).c_str());
    }
    // The field follows the state unless the user is typing in it. It applies when the field is left after an edit.
    if (!m_flipPpdEditing)
    {
        m_flipPpdField = static_cast<int>(std::lround(ppd));
    }
    ImGui::SetNextItemWidth(120.0f);
    // InputInt has no Enter-only mode: the value applies when the field is left after an edit.
    ImGui::InputInt("ppd", &m_flipPpdField, 0, 0);
    if (ImGui::IsItemDeactivatedAfterEdit())
    {
        m_flipPpdField = std::clamp(m_flipPpdField, static_cast<int>(kMinFlipPpd), static_cast<int>(kMaxFlipPpd));
        m_state.setFlipPpd(static_cast<float>(m_flipPpdField));
    }
    m_flipPpdEditing = ImGui::IsItemActive();
    ImGui::SameLine();
    if (ImGui::SmallButton("Default"))
    {
        m_state.setFlipPpd(defaultFlipPpd());
        m_flipPpdField = static_cast<int>(std::lround(defaultFlipPpd()));
    }

    // The live FLIP rate: how often the FLIP and the metrics update while the camera moves. It
    // follows the state unless it is being typed in, and applies when it is left after an edit; a
    // refused value leaves the state as it was, and the status line says why.
    if (!m_liveFlipEditing)
    {
        m_liveFlipField = m_state.liveFlipHz();
    }
    ImGui::SetNextItemWidth(120.0f);
    ImGui::InputFloat("Live FLIP rate (Hz)", &m_liveFlipField, 0.0f, 0.0f, "%.2f");
    if (ImGui::IsItemDeactivatedAfterEdit())
    {
        m_state.setLiveFlipHz(m_liveFlipField);
    }
    m_liveFlipEditing = ImGui::IsItemActive();
    ImGui::PopID();
}

void CompareUi::drawViewSection()
{
    ImGui::TextUnformatted("View");
    const ViewSettings & settings = m_state.settings();
    if (ImGui::RadioButton("Split", settings.split))
    {
        m_state.setSplit(true);
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Single", !settings.split))
    {
        m_state.setSplit(false);
    }

    if (settings.split)
    {
        // Two visible sources share one image with a divider, so the arrangement is for three or four.
        if (visibleCount(m_state.layout()) >= 3)
        {
            if (ImGui::RadioButton("Row", settings.arrangement == PaneArrangement::Row))
            {
                m_state.setArrangement(PaneArrangement::Row);
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("2x2", settings.arrangement == PaneArrangement::Grid2x2))
            {
                m_state.setArrangement(PaneArrangement::Grid2x2);
            }
        }
        ImGui::TextDisabled("Overlay modes are in Single screen");
    }
    else
    {
        const struct
        {
            const char * label;
            SingleView view;
        } choices[] = {{"A", SingleView::A},
                       {"B", SingleView::B},
                       {"Wipe", SingleView::Wipe},
                       {"Flicker", SingleView::Flicker},
                       {"Diff", SingleView::Diff},
                       {"Depth", SingleView::Depth},
                       {"Alpha", SingleView::Alpha},
                       {"Depth diff", SingleView::DepthDiff},
                       {"Alpha diff", SingleView::AlphaDiff},
                       {"FLIP", SingleView::Flip}};
        // Rows of five, two, two and one: the map modes sit on lines of their own.
        for (size_t i = 0; i < std::size(choices); ++i)
        {
            if (i != 0 && i != 5 && i != 7 && i != 9)
            {
                ImGui::SameLine();
            }
            if (ImGui::RadioButton(choices[i].label, settings.single == choices[i].view))
            {
                m_state.setSingleView(choices[i].view);
            }
        }
        if (settings.single == SingleView::Wipe)
        {
            int orientation = m_state.layout().view.wipeOrientation == WipeOrientation::Vertical ? 0 : 1;
            const char * const names[] = {"Vertical line", "Horizontal line"};
            ImGui::SetNextItemWidth(160.0f);
            // The hidden suffix keeps this combo's ID apart from the "Wipe" radio button above.
            if (ImGui::Combo("Wipe##orientation", &orientation, names, 2))
            {
                m_state.setWipeOrientation(orientation == 0 ? WipeOrientation::Vertical
                                                            : WipeOrientation::Horizontal);
            }
        }
        if (settings.single == SingleView::Flicker)
        {
            float period = static_cast<float>(settings.flickerPeriod);
            ImGui::SetNextItemWidth(160.0f);
            if (ImGui::SliderFloat("Period (s)", &period, 0.1f, 60.0f, "%.2f"))
            {
                m_state.setFlickerPeriod(period);
            }
        }
        if (settings.single == SingleView::Diff || settings.single == SingleView::AlphaDiff)
        {
            float gain = settings.diffGain;
            ImGui::SetNextItemWidth(160.0f);
            if (ImGui::SliderFloat("Gain", &gain, 0.01f, 100.0f, "%.2f"))
            {
                m_state.setDiffGain(gain);
            }
        }
        if (settings.single == SingleView::Depth || settings.single == SingleView::Alpha)
        {
            if (ImGui::RadioButton("Source A", !settings.mapUseB))
            {
                m_state.setMapUseB(false);
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("Source B", settings.mapUseB))
            {
                m_state.setMapUseB(true);
            }
        }
        if (settings.single == SingleView::Depth || settings.single == SingleView::DepthDiff)
        {
            bool autoRange = settings.depthAuto;
            if (ImGui::Checkbox("Auto range", &autoRange))
            {
                m_state.setDepthAuto(autoRange);
            }
            // With auto range the fields show the range the maps use, and cannot be edited.
            float rangeMin = settings.depthMin;
            float rangeMax = settings.depthMax;
            if (settings.depthAuto)
            {
                m_state.resolvedDepthRange(rangeMin, rangeMax);
            }
            ImGui::BeginDisabled(settings.depthAuto);
            ImGui::SetNextItemWidth(120.0f);
            const bool minChanged = ImGui::InputFloat("Min", &rangeMin, 0.0f, 0.0f, "%.6g");
            ImGui::SetNextItemWidth(120.0f);
            const bool maxChanged = ImGui::InputFloat("Max", &rangeMax, 0.0f, 0.0f, "%.6g");
            ImGui::EndDisabled();
            if (minChanged || maxChanged)
            {
                m_state.setDepthRange(rangeMin, rangeMax);
            }
        }
    }

    bool nearest = settings.nearest;
    if (ImGui::Checkbox("Nearest sampling", &nearest))
    {
        m_state.setNearest(nearest);
    }

    double percent = effectiveZoom(m_state.layout()) * 100.0;
    ImGui::SetNextItemWidth(120.0f);
    if (ImGui::InputDouble("Zoom %", &percent, 10.0, 100.0, "%.0f"))
    {
        setZoomPercent(m_state.viewLayout(), percent);
        m_state.viewChanged();
    }
    if (ImGui::Button("100%"))
    {
        setZoomPercent(m_state.viewLayout(), 100.0);
        m_state.viewChanged();
    }
    ImGui::SameLine();
    if (ImGui::Button("Fit"))
    {
        fitView();
    }

    ImGui::TextUnformatted("Anchor");
    if (ImGui::RadioButton(kAnchorTopLeft, settings.anchor == project::Anchor::TopLeft))
    {
        m_state.applyAnchor(project::Anchor::TopLeft);
    }
    ImGui::SameLine();
    if (ImGui::RadioButton(kAnchorCentered, settings.anchor == project::Anchor::Centered))
    {
        m_state.applyAnchor(project::Anchor::Centered);
    }
    if (settings.anchor == project::Anchor::Custom)
    {
        ImGui::TextDisabled("Offsets are custom.");
    }
}

void CompareUi::drawProbeSection()
{
    ImGui::TextUnformatted("Probe");
    if (!m_probe.valid)
    {
        ImGui::TextDisabled("Hover the view to read a pixel.");
        return;
    }
    ImGui::Text("Pixel (%d, %d)", m_probe.x, m_probe.y);
    switch (m_probe.kind)
    {
    case PixelKind::Image:
        if (m_probe.target >= 0 && static_cast<size_t>(m_probe.target) < m_state.size())
        {
            ImGui::Text("Image: %s at (%d, %d)",
                        m_state.entries()[static_cast<size_t>(m_probe.target)].name.c_str(), m_probe.localX,
                        m_probe.localY);
        }
        break;
    case PixelKind::Padding:
        ImGui::TextUnformatted("Padding: no pixel of this pane's source here.");
        break;
    case PixelKind::Outside:
        ImGui::TextUnformatted("Outside all images.");
        break;
    }
    const auto sample = [](const char * label, bool has, const float v[4])
    {
        if (has)
        {
            ImGui::Text("%s  %.4f %.4f %.4f %.4f", label, v[0], v[1], v[2], v[3]);
        }
        else
        {
            ImGui::Text("%s  no pixel", label);
        }
    };
    sample("A", m_probe.hasA, m_probe.a);
    sample("B", m_probe.hasB, m_probe.b);
    // Depth lines only for sources that have depth at this pixel.
    const auto depthLine = [this](const char * label, int32_t index, const float * values, uint8_t count)
    {
        if (count == 0 || index < 0 || static_cast<size_t>(index) >= m_state.size())
        {
            return;
        }
        const DepthInfo info = m_state.entries()[static_cast<size_t>(index)].media->depthInfo();
        ImGui::Text("%s  %s", label, depthText(info, values, count).c_str());
    };
    depthLine("A", m_state.sourceA(), m_probe.depthA, m_probe.depthCountA);
    depthLine("B", m_state.sourceB(), m_probe.depthB, m_probe.depthCountB);
    if (m_probe.hasA && m_probe.hasB)
    {
        const float difference[4] = {m_probe.a[0] - m_probe.b[0], m_probe.a[1] - m_probe.b[1],
                                     m_probe.a[2] - m_probe.b[2], m_probe.a[3] - m_probe.b[3]};
        sample("A-B", true, difference);
    }
    if (m_probe.hasFlip)
    {
        ImGui::Text("FLIP  %.6f", static_cast<double>(m_probe.flip));
    }
}

// ---- View area --------------------------------------------------------------------------

bool CompareUi::readSample(int32_t index, int32_t x, int32_t y, float rgba[4]) const
{
    if (index < 0 || static_cast<size_t>(index) >= m_state.size())
    {
        return false;
    }
    const MediaEntry & entry = m_state.entries()[static_cast<size_t>(index)];
    if (!entry.ready())
    {
        return false;
    }
    return entry.media->readPixel(x - entry.offsetX, y - entry.offsetY, rgba);
}

uint8_t CompareUi::readDepthSample(int32_t index, int32_t x, int32_t y, float out[4]) const
{
    if (index < 0 || static_cast<size_t>(index) >= m_state.size())
    {
        return 0;
    }
    const MediaEntry & entry = m_state.entries()[static_cast<size_t>(index)];
    if (!entry.ready())
    {
        return 0;
    }
    uint32_t count = 0;
    if (!entry.media->readDepthRaw(x - entry.offsetX, y - entry.offsetY, out, count))
    {
        return 0;
    }
    return static_cast<uint8_t>(std::min<uint32_t>(count, 4));
}

bool CompareUi::isSceneIndex(int32_t index) const
{
    if (index < 0 || static_cast<size_t>(index) >= m_state.size())
    {
        return false;
    }
    return m_state.entries()[static_cast<size_t>(index)].media->kind() == project::SourceKind::Scene;
}

// Panes mode asks the pane under the pointer. A single-pane mode asks the source it shows:
// the chosen one for the maps, either of the pair for the other modes (the camera moves
// the scenes the mode draws).
bool CompareUi::paneShowsScene(const CompositeInput & input, const LayoutInput & shown, int32_t px, int32_t py) const
{
    switch (input.mode)
    {
    case CompositeMode::Panes:
        return isSceneIndex(windowToImage(shown, px, py).source);
    case CompositeMode::Depth:
    case CompositeMode::Alpha:
        return isSceneIndex(input.mapUseB ? input.layout.view.sourceB : input.layout.view.sourceA);
    case CompositeMode::Wipe:
    case CompositeMode::Flicker:
    case CompositeMode::Diff:
    case CompositeMode::DepthDiff:
    case CompositeMode::AlphaDiff:
    case CompositeMode::Flip:
        return isSceneIndex(input.layout.view.sourceA) || isSceneIndex(input.layout.view.sourceB);
    }
    return false;
}

void CompareUi::orbitScene(float deltaX, float deltaY, bool orbit, bool pan, float wheel)
{
    app::OrbitCamera cam = m_state.camera();
    app::applyOrbitMouse(cam, deltaX, deltaY, orbit, pan, wheel);
    // Input moves are interactive: the scenes follow every frame while the camera moves.
    m_state.setCamera(cam, true);
}

void CompareUi::updateProbe(const LayoutInput & shown, int32_t px, int32_t py)
{
    m_probe = Probe{};
    const PanePixelClass cls = classifyPanePixel(shown, px, py);
    if (cls.pane < 0)
    {
        return;
    }
    const ImageHit hit = windowToImage(shown, px, py);
    // The shown layout is A's pane. In divider mode the pixels right of the divider show B, so the
    // probe reads B there: the pixel center decides, as the wipe does.
    ProbeTarget target = probeTarget(shown, px, py);
    PixelKind kind = cls.kind;
    if (m_state.dividerActive() && static_cast<double>(px) + 0.5 >= dividerWindowX(shown))
    {
        probeSource(shown, m_state.sourceB(), hit.pixelX, hit.pixelY, target, kind);
    }
    m_probe.valid = true;
    m_probe.kind = kind;
    m_probe.x = hit.pixelX;
    m_probe.y = hit.pixelY;
    m_probe.target = target.source;
    m_probe.localX = target.localX;
    m_probe.localY = target.localY;
    m_probe.hasA = readSample(m_state.sourceA(), hit.pixelX, hit.pixelY, m_probe.a);
    m_probe.hasB = readSample(m_state.sourceB(), hit.pixelX, hit.pixelY, m_probe.b);
    m_probe.depthCountA = readDepthSample(m_state.sourceA(), hit.pixelX, hit.pixelY, m_probe.depthA);
    m_probe.depthCountB = readDepthSample(m_state.sourceB(), hit.pixelX, hit.pixelY, m_probe.depthB);
    // The FLIP error at this canvas pixel, when the pixel is inside the FLIP overlap.
    const std::optional<FlipGpuResult> & flipResult = m_state.flip();
    if (flipResult && rectContains(flipResult->overlap, hit.pixelX, hit.pixelY))
    {
        const PixelRect & o = flipResult->overlap;
        const size_t index = static_cast<size_t>(hit.pixelY - o.y) * static_cast<size_t>(o.width)
            + static_cast<size_t>(hit.pixelX - o.x);
        if (index < flipResult->errorMap.size())
        {
            m_probe.flip = flipResult->errorMap[index];
            m_probe.hasFlip = true;
        }
    }
}

void CompareUi::drawViewArea(float width, float height)
{
    ImGuiIO & io = ImGui::GetIO();
    const float scale = io.DisplayFramebufferScale.x > 0.0f ? io.DisplayFramebufferScale.x : 1.0f;
    const int32_t pixelWidth = toPixels(width, scale);
    const int32_t pixelHeight = toPixels(height, scale);
    if (pixelWidth != static_cast<int32_t>(m_compositor.width())
        || pixelHeight != static_cast<int32_t>(m_compositor.height()))
    {
        resizeView(pixelWidth, pixelHeight);
    }
    m_state.setViewSize(pixelWidth, pixelHeight);
    if (m_viewSet == VK_NULL_HANDLE && m_compositor.view() != VK_NULL_HANDLE)
    {
        addTexture();
    }

    const ImVec2 origin = ImGui::GetCursorScreenPos();
    if (m_viewSet != VK_NULL_HANDLE && width >= 1.0f && height >= 1.0f)
    {
        ImGui::Image(textureId(), ImVec2(width, height));
    }
    if (m_state.size() == 0)
    {
        ImGui::GetWindowDrawList()->AddText(ImVec2(origin.x + 16.0f, origin.y + 16.0f),
                                            IM_COL32(220, 220, 220, 255),
                                            "Add an image with File > Add media..., or drop image files here.");
    }

    // An invisible button over the image takes the mouse.
    ImGui::SetCursorPos(ImVec2(0.0f, 0.0f));
    // Middle and right presses count too: the camera and the image pan with them.
    ImGui::InvisibleButton("##viewhit", ImVec2(std::max(width, 1.0f), std::max(height, 1.0f)),
                           ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight
                               | ImGuiButtonFlags_MouseButtonMiddle);
    const bool hovered = ImGui::IsItemHovered();
    const ImVec2 min = ImGui::GetItemRectMin();
    const ImVec2 max(min.x + width, min.y + height);

    // A drag ends when no button is down, and when the window loses focus (the release may never come).
    const bool anyButton = ImGui::IsMouseDown(ImGuiMouseButton_Left) || ImGui::IsMouseDown(ImGuiMouseButton_Right)
        || ImGui::IsMouseDown(ImGuiMouseButton_Middle);
    if (!anyButton || io.AppFocusLost)
    {
        m_wipeDrag = false;
        m_dividerDrag = false;
        m_sceneDrag = false;
    }
    if (pixelWidth <= 0 || pixelHeight <= 0)
    {
        m_probe = Probe{};
        return;
    }

    const auto clampPixel = [](float value, int32_t size)
    {
        const int32_t pixel = static_cast<int32_t>(std::floor(value));
        return std::min(std::max(pixel, 0), size - 1);
    };
    const int32_t px = clampPixel((io.MousePos.x - min.x) * scale, pixelWidth);
    const int32_t py = clampPixel((io.MousePos.y - min.y) * scale, pixelHeight);

    const CompositeInput input = m_state.buildComposite(m_now);
    const LayoutInput shown = compositeLayout(input);
    // In divider mode the composite is a wipe, but the boundary is the divider, which the state
    // owns: only the single-screen wipe takes the wipe drag and the right-button wipe.
    const bool dividerMode = m_state.dividerActive();
    const bool wipeMode = input.mode == CompositeMode::Wipe && !dividerMode;
    const bool grabVertical = dividerMode || shown.view.wipeOrientation == WipeOrientation::Vertical;
    // The grab line in logical units (x for a vertical line, y for a horizontal one), if there is one.
    // The pointer is read unclamped, so a line at the edge of the image area stays grabbable.
    const auto grabLineAt = [&](const LayoutInput & layout, float & at) -> bool
    {
        double coordinate = 0.0;
        if (dividerMode)
        {
            coordinate = dividerWindowX(layout);
        }
        else if (!wipeMode || !wipeWindowCoordinate(layout, 0, coordinate))
        {
            return false;
        }
        at = (grabVertical ? min.x : min.y) + static_cast<float>(coordinate / static_cast<double>(scale));
        return true;
    };
    float grabAt = 0.0f;
    const bool hasGrab = grabLineAt(shown, grabAt);
    const bool onGrab = hasGrab && grabHit(io.MousePos, min, max, grabAt, grabVertical);
    // A scene pane takes the camera input; every other pane keeps the image controls.
    const bool sceneHere = hovered && paneShowsScene(input, shown, px, py);
    const bool dragging = ImGui::IsItemActive();
    const bool moved = io.MouseDelta.x != 0.0f || io.MouseDelta.y != 0.0f;

    // Drag the image: the point under the pointer stays under it.
    const auto panImage = [&]()
    {
        const double zoom = effectiveZoom(m_state.layout());
        if (zoom > 0.0)
        {
            const LayoutInput & current = m_state.layout();
            const double dx = static_cast<double>(io.MouseDelta.x) * scale;
            const double dy = static_cast<double>(io.MouseDelta.y) * scale;
            setPan(m_state.viewLayout(), current.view.panX - dx / zoom, current.view.panY - dy / zoom);
            m_state.viewChanged();
        }
    };

    if (hovered)
    {
        if (io.MouseWheel != 0.0f)
        {
            if (sceneHere && !io.KeyCtrl)
            {
                // The wheel dollies the camera; Ctrl+wheel zooms the image.
                orbitScene(0.0f, 0.0f, false, false, io.MouseWheel);
            }
            else
            {
                zoomAt(m_state.viewLayout(), px, py, std::pow(kWheelZoomFactor, static_cast<double>(io.MouseWheel)));
                m_state.viewChanged();
            }
        }
        if (!sceneHere && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left))
        {
            fitView();
        }
        if (ImGui::IsItemActivated())
        {
            // A left press on the grab line takes it: the divider, or the single-screen wipe. The
            // handle wins over the pan and over the scene orbit.
            const bool grabPress = onGrab && ImGui::IsMouseDown(ImGuiMouseButton_Left);
            m_dividerDrag = dividerMode && grabPress;
            m_wipeDrag = wipeMode && grabPress;
            m_sceneDrag = sceneHere && !m_dividerDrag;
        }
        if (wipeMode && !sceneHere && ImGui::IsMouseDown(ImGuiMouseButton_Right))
        {
            double position = 0.0;
            if (wipePositionAtPixel(shown, px, py, position))
            {
                m_state.viewLayout().view.wipePosition = position;
                m_state.viewChanged();
            }
        }
        // Q rolls one way and E the other, while a scene pane is under the pointer.
        if (sceneHere && !io.WantTextInput)
        {
            const float roll = kRollSpeed * io.DeltaTime
                * ((ImGui::IsKeyDown(ImGuiKey_E) ? 1.0f : 0.0f) - (ImGui::IsKeyDown(ImGuiKey_Q) ? 1.0f : 0.0f));
            if (roll != 0.0f)
            {
                app::OrbitCamera cam = m_state.camera();
                cam.applyDelta(0.0f, 0.0f, roll);
                m_state.setCamera(cam, true);
            }
        }
    }

    // A drag that began on the grab line moves it. One that began on a scene pane orbits (left),
    // pans the camera (right) or pans the image (middle). Other drags pan the image with the left button.
    if (dragging && m_wipeDrag && ImGui::IsMouseDown(ImGuiMouseButton_Left))
    {
        double position = 0.0;
        if (wipePositionAtPixel(shown, px, py, position))
        {
            m_state.viewLayout().view.wipePosition = position;
            m_state.viewChanged();
        }
    }
    else if (dragging && m_dividerDrag && ImGui::IsMouseDown(ImGuiMouseButton_Left))
    {
        // The divider follows the pointer, which may leave the view while the button is held; the
        // state clamps the fraction.
        m_state.setDividerFraction(dividerFractionAtWindowX(shown, static_cast<double>(io.MousePos.x - min.x) * scale));
    }
    else if (dragging && m_sceneDrag)
    {
        if (moved)
        {
            const bool orbit = ImGui::IsMouseDown(ImGuiMouseButton_Left);
            const bool pan = !orbit && ImGui::IsMouseDown(ImGuiMouseButton_Right);
            if (orbit || pan)
            {
                orbitScene(io.MouseDelta.x, io.MouseDelta.y, orbit, pan, 0.0f);
            }
            if (ImGui::IsMouseDown(ImGuiMouseButton_Middle))
            {
                panImage();
            }
        }
    }
    else if (dragging && ImGui::IsMouseDown(ImGuiMouseButton_Left) && moved)
    {
        panImage();
    }

    // The grab line as the state now holds it: a drag may have moved it this frame.
    LayoutInput placed = shown;
    placed.view.dividerFraction = m_state.dividerFraction();
    placed.view.wipePosition = m_state.layout().view.wipePosition;
    grabLineAt(placed, grabAt);
    const bool gripping = m_dividerDrag || m_wipeDrag;
    const bool onGrabNow = hasGrab && grabHit(io.MousePos, min, max, grabAt, grabVertical);

    // Cursors: the resize arrow on the grab line (and while it is dragged), the hand while the
    // left or middle drag pans the image (a scene's orbit keeps the arrow).
    if ((dragging && gripping) || (hovered && onGrabNow))
    {
        ImGui::SetMouseCursor(grabVertical ? ImGuiMouseCursor_ResizeEW : ImGuiMouseCursor_ResizeNS);
    }
    else if (dragging && !gripping
             && ((ImGui::IsMouseDown(ImGuiMouseButton_Left) && !m_sceneDrag)
                 || ImGui::IsMouseDown(ImGuiMouseButton_Middle)))
    {
        ImGui::SetMouseCursor(ImGuiMouseCursor_Hand);
    }

    if (hovered || dragging)
    {
        updateProbe(shown, px, py);
    }
    else
    {
        m_probe = Probe{};
    }

    // The grab line and its handle, then the hint. Both are drawn after the invisible button, on the
    // window's draw list, so they never reach the composite or the exports.
    ImDrawList * drawList = ImGui::GetWindowDrawList();
    if (hasGrab)
    {
        drawGrabLine(drawList, min, max, grabAt, grabVertical);
        drawGrabHandle(drawList, grabHandleCenter(min, max, grabAt, grabVertical), grabVertical,
                       gripping || (hovered && onGrabNow));
    }
    if (m_state.settings().split && visibleCount(m_state.layout()) >= 1)
    {
        drawHintLine(drawList, min, max, dividerMode ? kHintDivider : kHintPan);
    }
}

// ---- Public exports ---------------------------------------------------------------------

std::string exportCurrentView(Compositor & compositor, const std::string & utf8Path)
{
    if (compositor.width() == 0 || compositor.height() == 0 || compositor.view() == VK_NULL_HANDLE)
    {
        throw std::runtime_error("the view area is empty");
    }
    std::vector<uint8_t> rgba;
    compositor.readback(rgba);
    saveRgbaPng(utf8Path, rgba.data(), compositor.width(), compositor.height());
    return "Exported the view (" + std::to_string(compositor.width()) + "x" + std::to_string(compositor.height())
        + ") to " + fileNameUtf8(utf8Path);
}

std::string exportDiffMap(const CompareState & state, const std::string & utf8Path)
{
    if (!state.hasPair())
    {
        throw std::runtime_error("the diff map needs A and B visible");
    }
    const ExportScene scene = buildExportScene(state);
    const Rgb8Image image = diffMapImage(scene, state.settings().diffGain);
    savePngRgb8(utf8Path, image);
    return "Exported the diff map (" + std::to_string(image.width) + "x" + std::to_string(image.height) + ") to "
        + fileNameUtf8(utf8Path);
}

std::string exportFlipMap(const CompareState & state, const std::string & utf8Path)
{
    if (!state.flip())
    {
        throw std::runtime_error(flipUnavailable(state));
    }
    const FlipGpuResult & flip = *state.flip();
    const ExportScene scene = buildExportScene(state);
    const Rgb8Image image = flipMapImage(scene, flip.errorMap, flip.overlap);
    savePngRgb8(utf8Path, image);
    return "Exported the FLIP map (" + std::to_string(image.width) + "x" + std::to_string(image.height) + ") to "
        + fileNameUtf8(utf8Path);
}

std::string exportExposureMap(const CompareState & state, const std::string & utf8Path)
{
    if (!state.flip() || !state.flip()->hdr)
    {
        throw std::runtime_error("no HDR-FLIP result");
    }
    const FlipGpuResult & flip = *state.flip();
    const ExportScene scene = buildExportScene(state);
    // The exposure index of each pixel (i / (count - 1)), over the overlap; padding takes the top.
    const Rgb8Image image = flipMapImage(scene, flip.hdr->exposureMap, flip.overlap);
    savePngRgb8(utf8Path, image);
    return "Exported the exposure map (" + std::to_string(image.width) + "x" + std::to_string(image.height) + ") to "
        + fileNameUtf8(utf8Path);
}

std::string exportPaneImages(const CompareState & state, const std::string & utf8Path)
{
    const ExportScene scene = buildExportScene(state);
    const fs::path base = pathFromUtf8(utf8Path);
    const std::string stem = stemUtf8(utf8Path);
    size_t written = 0;
    for (size_t i = 0; i < state.size(); ++i)
    {
        if (!state.entries()[i].visible)
        {
            continue;
        }
        const Rgb8Image image = paddedPaneImage(scene, static_cast<int32_t>(i));
        const fs::path target = base.parent_path() / pathFromUtf8(stem + "_" + state.entries()[i].id + ".png");
        savePngRgb8(utf8Of(target), image);
        ++written;
    }
    if (written == 0)
    {
        throw std::runtime_error("no visible source to export");
    }
    return "Exported " + std::to_string(written) + " padded pane image(s) as " + stem + "_<id>.png";
}

std::string exportReport(const CompareState & state, const std::string & utf8Path)
{
    if (!state.metrics())
    {
        throw std::runtime_error("no metrics to report: " + state.metricsReason());
    }
    const int32_t a = state.sourceA();
    const int32_t b = state.sourceB();
    if (a < 0 || b < 0)
    {
        throw std::runtime_error("the report needs A and B visible");
    }
    const MediaEntry & ea = state.entries()[static_cast<size_t>(a)];
    const MediaEntry & eb = state.entries()[static_cast<size_t>(b)];
    const ImageData * hostA = ea.media->hostData();
    const ImageData * hostB = eb.media->hostData();
    if (hostA == nullptr || hostB == nullptr)
    {
        throw std::runtime_error("the report needs both host copies");
    }
    const MetricsResult & metrics = *state.metrics();
    ReportInput input;
    input.pathA = ea.path;
    input.pathB = eb.path;
    input.a = hostA;
    input.b = hostB;
    input.aOffsetX = ea.offsetX;
    input.aOffsetY = ea.offsetY;
    input.overlap = metrics.overlap;
    input.mse = metrics.mse;
    input.psnrDb = metrics.psnrDb;
    if (state.flip())
    {
        input.flipMean = state.flip()->mean;
    }
    input.ppd = state.flipPpd();
    if (state.flip() && state.flip()->hdr)
    {
        const FlipHdrInfo & info = *state.flip()->hdr;
        ReportExposures exposures;
        exposures.toneMapper = toneMapperName(info.toneMapper);
        exposures.start = info.start;
        exposures.stop = info.stop;
        exposures.count = info.count;
        input.hdr = exposures;
        if (std::isfinite(info.linearMse))
        {
            input.linearMse = info.linearMse;
        }
    }
    input.anchor = anchorName(state.settings().anchor);
    addVideoReportFields(state, input);
    const CompareReport report = makeReport(input);
    saveReportFile(utf8Path, report);
    return "Exported the report to " + fileNameUtf8(utf8Path);
}

} // namespace sabercat::compare::gui
