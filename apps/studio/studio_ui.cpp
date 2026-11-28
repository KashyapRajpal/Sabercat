// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "studio_ui.h"

#include "studio_state.h"

#include "about_panel.h"
#include "app_util.h"
#include "file_dialog.h"
#include "relight_panel.h"

#include "sabercat/edit_math.h"
#include "sabercat/edit_pass.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/logger.h"
#include "sabercat/renderer.h"

#include <imgui.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <functional>
#include <limits>

namespace sabercat::studio
{

namespace
{

constexpr const char * kPathPopup = "Path##studio";
constexpr const char * kDiscardPopup = "Discard unsaved changes?##studio";
constexpr const char * kRenamePopup = "Rename layer##studio";
constexpr const char * kErrorPopup = "Error##studio";
constexpr const char * kAssociationPopup = "Point .sabercat files at this copy of Studio?##studio";

// The outline and the wireframe of the selection tools, in display colors.
constexpr ImU32 kSelectColor = IM_COL32(255, 170, 40, 230);
constexpr ImU32 kSelectFill = IM_COL32(255, 170, 40, 40);
// A drag shorter than this (window pixels, on either axis) is a click, not a rectangle.
constexpr float kClickSlop = 3.0f;
constexpr const char * const kPivotNames[] = {"Selection centroid", "Layer origin", "World origin"};
// The History window's sizes are shown in binary units, as the budget is.
constexpr double kKilobyte = 1024.0;
constexpr double kMegabyte = 1024.0 * 1024.0;

// Runs a file action; a failure becomes the error modal, never a crash.
void runAction(UiState & ui, const std::function<void()> & action)
{
    try
    {
        action();
    }
    catch (const std::exception & e)
    {
        reportError(ui, e.what());
    }
}

void openPathDialog(UiState & ui, PathPurpose purpose, const std::string & initial)
{
    ui.pathPurpose = purpose;
    std::snprintf(ui.pathBuffer, sizeof(ui.pathBuffer), "%s", initial.c_str());
    ui.openPathPopup = true;
}

std::string pathUtf8(const std::filesystem::path & path)
{
    const std::u8string text = path.u8string();
    return std::string(reinterpret_cast<const char *>(text.data()), text.size());
}

// The file a save dialog starts from: the project's own file name, or a new one's.
std::string defaultProjectFileName(const StudioState & state)
{
    if (state.projectPath.empty())
    {
        return "untitled.sabercat";
    }
    return pathUtf8(std::filesystem::u8path(state.projectPath).filename());
}

// The base name an export dialog starts from: "<project>" or "untitled".
std::string exportBaseName(const StudioState & state)
{
    if (state.projectPath.empty())
    {
        return "untitled";
    }
    return pathUtf8(std::filesystem::u8path(state.projectPath).stem());
}

// Starts the native dialog a purpose needs: a save for the project and the
// exports, an open for the rest. The dialog runs on a worker thread, so the
// window keeps drawing; pollDialog() takes the result. Refused while another
// dialog is open.
void startDialogFor(StudioState & state, UiState & ui, PathPurpose purpose, DialogUse use)
{
    bool started = false;
    switch (purpose)
    {
    case PathPurpose::OpenProject:
        started = ui.dialog.startOpen(ui.window, app::projectFileFilters(), "Open project",
                                      "studio.openProject");
        break;
    case PathPurpose::SaveAs:
        started = ui.dialog.startSave(ui.window, app::projectFileFilters(), defaultProjectFileName(state),
                                      "sabercat", "studio.saveProject");
        break;
    case PathPurpose::AddLayer:
        started = ui.dialog.startOpen(ui.window, app::sceneFileFilters(), "Add scene layer",
                                      "studio.addLayer");
        break;
    case PathPurpose::ExportPly:
        started = ui.dialog.startSave(ui.window, app::plyFileFilters(), exportBaseName(state) + ".ply",
                                      "ply", "studio.exportPly");
        break;
    case PathPurpose::ExportGlb:
        started = ui.dialog.startSave(ui.window, app::glbFileFilters(), exportBaseName(state) + ".glb",
                                      "glb", "studio.exportGlb");
        break;
    case PathPurpose::ExportSpz:
        started = ui.dialog.startSave(ui.window, app::spzFileFilters(), exportBaseName(state) + ".spz",
                                      "spz", "studio.exportSpz");
        break;
    }
    if (started)
    {
        ui.dialogPurpose = purpose;
        ui.dialogUse = use;
    }
}

// Save As: the native save dialog. Cancel changes nothing.
void chooseSaveAs(StudioState & state, UiState & ui)
{
    startDialogFor(state, ui, PathPurpose::SaveAs, DialogUse::Action);
}

void requestSave(StudioState & state, UiState & ui)
{
    if (state.projectPath.empty())
    {
        chooseSaveAs(state, ui);
        return;
    }
    runAction(ui, [&]() { saveProject(state, state.projectPath); });
}

void chooseOpenProject(StudioState & state, UiState & ui)
{
    startDialogFor(state, ui, PathPurpose::OpenProject, DialogUse::Action);
}

void chooseAddLayer(StudioState & state, UiState & ui)
{
    startDialogFor(state, ui, PathPurpose::AddLayer, DialogUse::Action);
}

void chooseExport(StudioState & state, UiState & ui, PathPurpose format)
{
    startDialogFor(state, ui, format, DialogUse::Action);
}

// Carries out a confirmed path dialog.
void confirmPath(StudioState & state, Renderer & renderer, UiState & ui, PathPurpose purpose,
                 const std::string & path)
{
    switch (purpose)
    {
    case PathPurpose::OpenProject:
        requestOpenProject(state, renderer, ui, path);
        break;
    case PathPurpose::SaveAs:
        runAction(ui, [&]() { saveProject(state, path); });
        break;
    case PathPurpose::AddLayer:
        requestAddLayer(state, renderer, ui, path);
        break;
    case PathPurpose::ExportPly:
        runAction(ui, [&]() { exportPly(state, renderer.sabercatDevice(), path); });
        break;
    case PathPurpose::ExportGlb:
        runAction(ui, [&]() { exportGlb(state, path); });
        break;
    case PathPurpose::ExportSpz:
        runAction(ui, [&]() { exportSpz(state, renderer.sabercatDevice(), path); });
        break;
    }
}

// Takes a finished dialog's result, if there is one. From Browse in the path
// dialog a chosen path only fills the path box; otherwise the action the dialog
// was started for runs. A cancel changes nothing.
void pollDialog(StudioState & state, Renderer & renderer, UiState & ui)
{
    const auto picked = ui.dialog.poll();
    if (!picked)
    {
        return;
    }
    const DialogUse use = ui.dialogUse;
    ui.dialogUse = DialogUse::None;
    if (!*picked)
    {
        return;
    }
    const std::string & path = **picked;
    if (use == DialogUse::PathBox)
    {
        std::snprintf(ui.pathBuffer, sizeof(ui.pathBuffer), "%s", path.c_str());
    }
    else
    {
        confirmPath(state, renderer, ui, ui.dialogPurpose, path);
    }
}

const char * purposeText(PathPurpose purpose)
{
    switch (purpose)
    {
    case PathPurpose::OpenProject:
        return "Project file to open (.sabercat):";
    case PathPurpose::SaveAs:
        return "Save the project as (.sabercat):";
    case PathPurpose::AddLayer:
        return "Scene file to add as a layer:";
    case PathPurpose::ExportPly:
        return "PLY file to write (visible layers, world space):";
    case PathPurpose::ExportGlb:
        return "GLB file to write (one node per visible layer):";
    case PathPurpose::ExportSpz:
        return "SPZ file to write (visible layers, merged):";
    }
    return "";
}

// Frames the whole visible scene, with exact bounds (refreshExactBounds), as the
// viewer's Frame scene button does.
void frameAll(StudioState & state, Renderer & renderer, UiState & ui)
{
    runAction(ui, [&]()
              {
                  refreshExactBounds(state, renderer);
                  const SceneBounds bounds = renderer.sceneBounds();
                  if (bounds.valid)
                  {
                      state.camera.frame(bounds, renderer.height(), static_cast<float>(renderer.height()));
                  }
              });
}

// Reads whether .sabercat files open with this copy, for the Help menu.
void refreshAssociation(UiState & ui)
{
    ui.association = app::queryProjectAssociation(ui.exePath);
}

// Points .sabercat files at this copy and reports the result.
void registerThisCopy(UiState & ui)
{
    std::string error;
    if (app::registerProjectAssociation(ui.exePath, error))
    {
        Logger::info("Studio: .sabercat files now open with " + app::wideToUtf8(ui.exePath));
        ui.statusMessage = ".sabercat files now open with this copy of Studio.";
    }
    else
    {
        ui.statusMessage = "Could not register .sabercat files.";
        reportError(ui, "Could not register .sabercat files. " + error);
    }
    refreshAssociation(ui);
}

// Removes this copy's registration. Another copy's is refused by the registry code.
void unregisterThisCopy(UiState & ui)
{
    std::string error;
    if (app::unregisterProjectAssociation(ui.exePath, error))
    {
        Logger::info("Studio: .sabercat files are no longer registered to " + app::wideToUtf8(ui.exePath));
        ui.statusMessage = "Removed the .sabercat file association for this copy of Studio.";
    }
    else
    {
        ui.statusMessage = "Could not unregister .sabercat files.";
        reportError(ui, "Could not unregister .sabercat files. " + error);
    }
    refreshAssociation(ui);
}

// The Register and Unregister items and the state they act on. Unregister is
// offered only for this copy's own registration, so it never removes another's.
void drawAssociationItems(UiState & ui)
{
    switch (ui.association)
    {
    case app::AssociationState::ThisExe:
        ImGui::TextDisabled("Currently: registered to this Studio");
        break;
    case app::AssociationState::OtherExe:
        ImGui::TextDisabled("Currently: registered to another copy or program");
        break;
    case app::AssociationState::NotRegistered:
        ImGui::TextDisabled("Currently: not registered");
        break;
    }
    if (ImGui::MenuItem("Register .sabercat files", nullptr,
                        ui.association == app::AssociationState::ThisExe))
    {
        // Taking over another copy's association needs a confirmation first.
        if (ui.association == app::AssociationState::OtherExe)
        {
            ui.openAssociationPopup = true;
        }
        else
        {
            registerThisCopy(ui);
        }
    }
    ImGui::BeginDisabled(ui.association != app::AssociationState::ThisExe);
    if (ImGui::MenuItem("Unregister .sabercat files"))
    {
        unregisterThisCopy(ui);
    }
    ImGui::EndDisabled();
}

// --- Editing helpers ----------------------------------------------------------

// Drops a live transform preview, restoring the layer's values. A no-op without one.
void dropPreview(StudioState & state, Renderer & renderer, UiState & ui)
{
    runAction(ui, [&]() { cancelTransformPreview(state, renderer); });
}

// True when the transform changes nothing: no translation, no rotation, unit scale.
bool isIdentityParams(const TransformParams & params)
{
    return params.translation == glm::vec3(0.0f) && params.rotationDegrees == glm::vec3(0.0f)
           && params.scale == 1.0f;
}

// Restores the Transform fields to identity. The pivot is a setting, not a field, so
// it stays as the user chose it.
void resetTransformFields(TransformUi & tf)
{
    const PivotMode pivot = tf.params.pivot;
    tf.params = TransformParams{};
    tf.params.pivot = pivot;
    tf.dirtyFields = false;
}

// Starts or updates the live preview with `params`; an identity transform drops it
// instead, so a preview never sits on a layer with nothing to show.
void previewOrDrop(StudioState & state, Renderer & renderer, UiState & ui, const TransformParams & params)
{
    runAction(ui, [&]()
              {
                  if (isIdentityParams(params))
                  {
                      cancelTransformPreview(state, renderer);
                  }
                  else
                  {
                      previewTransform(state, renderer, params);
                  }
              });
}

// The Select window's op index (0 Replace, 1 Add, 2 Subtract) as the edit's SelectOp.
SelectOp selectOpOf(int index)
{
    return static_cast<SelectOp>(index);
}

// The index of the layer with `id`, or -1 when no layer has it.
int indexOfLayerId(const StudioState & state, const std::string & id)
{
    for (std::size_t i = 0; i < state.project.layers.size(); ++i)
    {
        if (state.project.layers[i].id == id)
        {
            return static_cast<int>(i);
        }
    }
    return -1;
}

// Ends the held Inspector drag and records it against its layer, found by id, so a layer
// that moved keeps its drag. A layer that is gone takes its drag with it (nothing is
// recorded). Clears the capture either way.
void endLayerTransformCapture(StudioState & state, UiState & ui)
{
    if (ui.layerTransformBeforeIndex < 0)
    {
        return;
    }
    const int index = indexOfLayerId(state, ui.layerTransformBeforeId);
    if (index >= 0)
    {
        runAction(ui, [&]() { commitLayerTransformEdit(state, index, ui.layerTransformBefore); });
    }
    ui.layerTransformBeforeIndex = -1;
    ui.layerTransformBeforeId.clear();
}

// Keeps a held drag on its own layer. When the drag's layer is no longer the selected one
// (the selection moved, or the layer was removed, while the drag was held), the drag is
// ended now, and the widgets edit the selected layer from here on. Returns true when the
// drag was ended that way, so a drag that goes on starts a new capture for that layer.
bool reconcileLayerTransformCapture(StudioState & state, UiState & ui)
{
    if (ui.layerTransformBeforeIndex < 0)
    {
        return false;
    }
    const int index = indexOfLayerId(state, ui.layerTransformBeforeId);
    if (index >= 0 && index == state.selectedLayer)
    {
        // Still the drag's layer. An undo may have moved it, so keep its index current.
        ui.layerTransformBeforeIndex = index;
        return false;
    }
    endLayerTransformCapture(state, ui);
    return true;
}

// The layer's bounds in its own space: the conservative local box while the GPU copy is
// ahead of the host, the host's exact bounds otherwise.
SceneBounds layerLocalBounds(const StudioState & state, const ProjectLayer & layer)
{
    const auto it = state.edit.layers.find(layer.id);
    if (it != state.edit.layers.end() && it->second.hostStale)
    {
        return it->second.localBounds;
    }
    return layer.cloud->bounds();
}

// The axis-aligned world box that holds the layer's bounds: the eight corners of the
// local box, mapped by the layer's placement. False when the layer has no bounds.
bool fitBoxToLayer(const StudioState & state, const ProjectLayer & layer, glm::vec3 & center,
                   glm::vec3 & halfExtents)
{
    const SceneBounds local = layerLocalBounds(state, layer);
    if (!local.valid)
    {
        return false;
    }
    const Similarity placement = similarityOf(layer.transform);
    glm::vec3 lo(std::numeric_limits<float>::infinity());
    glm::vec3 hi(-std::numeric_limits<float>::infinity());
    for (int corner = 0; corner < 8; ++corner)
    {
        const glm::dvec3 point((corner & 1) != 0 ? local.max.x : local.min.x,
                               (corner & 2) != 0 ? local.max.y : local.min.y,
                               (corner & 4) != 0 ? local.max.z : local.min.z);
        const glm::vec3 world(placement.apply(point));
        lo = glm::min(lo, world);
        hi = glm::max(hi, world);
    }
    center = 0.5f * (lo + hi);
    halfExtents = 0.5f * (hi - lo);
    return true;
}

// Deletes the selection of the active layer. A refusal's reason goes to the status line.
void deleteSelected(StudioState & state, Renderer & renderer, UiState & ui)
{
    runAction(ui, [&]()
              {
                  std::string whyNot;
                  if (!deleteSelection(state, renderer, whyNot))
                  {
                      ui.statusMessage = whyNot;
                  }
              });
}

void duplicateSelected(StudioState & state, Renderer & renderer, UiState & ui)
{
    runAction(ui, [&]()
              {
                  std::string whyNot;
                  if (!duplicateSelection(state, renderer, whyNot))
                  {
                      ui.statusMessage = whyNot;
                  }
              });
}

// Edit shortcuts: Ctrl+Z undo, Ctrl+Y or Ctrl+Shift+Z redo, Ctrl+A all, Ctrl+Shift+A
// none, Ctrl+I invert, Delete, Ctrl+D duplicate. Ignored while text is typed, a dialog
// is open or a popup is up. The selection shortcuts run only when the menu items would
// be enabled.
void handleEditShortcuts(StudioState & state, Renderer & renderer, UiState & ui)
{
    const ImGuiIO & io = ImGui::GetIO();
    if (io.WantTextInput || ui.dialog.pending() || ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId))
    {
        return;
    }
    ProjectLayer * layer = editableLayer(state);
    const bool canEdit = layer != nullptr;
    const bool canSelection = canEdit && selectedCountOfActiveLayer(state) > 0;

    if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_Z))
    {
        runAction(ui, [&]() { undo(state, renderer); });
    }
    if (ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_Y)
        || ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_Z))
    {
        runAction(ui, [&]() { redo(state, renderer); });
    }
    if (canEdit && ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_A))
    {
        runAction(ui, [&]() { selectAllGaussians(state, renderer); });
    }
    if (canEdit && ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_A))
    {
        runAction(ui, [&]() { selectNone(state, renderer); });
    }
    if (canEdit && ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_I))
    {
        runAction(ui, [&]() { invertSelection(state, renderer); });
    }
    if (canSelection && ImGui::IsKeyChordPressed(ImGuiKey_Delete))
    {
        deleteSelected(state, renderer, ui);
    }
    if (canSelection && ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_D))
    {
        duplicateSelected(state, renderer, ui);
    }
}

// The Edit menu. Its items show the shortcut that runs them; the shortcuts themselves
// are handled by handleEditShortcuts().
void drawEditMenu(StudioState & state, Renderer & renderer, UiState & ui)
{
    if (!ImGui::BeginMenu("Edit"))
    {
        return;
    }
    EditHistory & history = state.edit.history;
    const bool canUndo = history.canUndo();
    const bool canRedo = history.canRedo();
    std::string undoLabel = "Undo";
    if (canUndo)
    {
        undoLabel += " " + history.label(history.cursor() - 1);
    }
    std::string redoLabel = "Redo";
    if (canRedo)
    {
        redoLabel += " " + history.label(history.cursor());
    }

    ProjectLayer * layer = editableLayer(state);
    const bool canEdit = layer != nullptr;
    const bool canSelection = canEdit && selectedCountOfActiveLayer(state) > 0;

    if (ImGui::MenuItem(undoLabel.c_str(), "Ctrl+Z", false, canUndo))
    {
        runAction(ui, [&]() { undo(state, renderer); });
    }
    if (ImGui::MenuItem(redoLabel.c_str(), "Ctrl+Y", false, canRedo))
    {
        runAction(ui, [&]() { redo(state, renderer); });
    }
    ImGui::Separator();
    if (ImGui::MenuItem("Select All", "Ctrl+A", false, canEdit))
    {
        runAction(ui, [&]() { selectAllGaussians(state, renderer); });
    }
    if (ImGui::MenuItem("Select None", "Ctrl+Shift+A", false, canEdit))
    {
        runAction(ui, [&]() { selectNone(state, renderer); });
    }
    if (ImGui::MenuItem("Invert Selection", "Ctrl+I", false, canEdit))
    {
        runAction(ui, [&]() { invertSelection(state, renderer); });
    }
    ImGui::Separator();
    if (ImGui::MenuItem("Delete Selected", "Delete", false, canSelection))
    {
        deleteSelected(state, renderer, ui);
    }
    if (ImGui::MenuItem("Duplicate", "Ctrl+D", false, canSelection))
    {
        duplicateSelected(state, renderer, ui);
    }
    ImGui::EndMenu();
}

void drawWindowMenu(UiState & ui)
{
    if (ImGui::BeginMenu("Window"))
    {
        if (ImGui::MenuItem("Select", nullptr, ui.select.open))
        {
            ui.select.open = !ui.select.open;
        }
        if (ImGui::MenuItem("Transform", nullptr, ui.transform.open))
        {
            ui.transform.open = !ui.transform.open;
        }
        if (ImGui::MenuItem("History", nullptr, ui.historyOpen))
        {
            ui.historyOpen = !ui.historyOpen;
        }
        ImGui::EndMenu();
    }
}

void drawMenuBar(StudioState & state, Renderer & renderer, UiState & ui)
{
    // Every item that starts a dialog or acts on files waits for the open dialog.
    const bool busy = ui.dialog.pending();
    if (!busy && ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_S))
    {
        requestSave(state, ui);
    }
    if (!busy && ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiKey_O))
    {
        chooseOpenProject(state, ui);
    }
    if (!busy && ImGui::IsKeyChordPressed(ImGuiMod_Ctrl | ImGuiMod_Shift | ImGuiKey_S))
    {
        chooseSaveAs(state, ui);
    }
    handleEditShortcuts(state, renderer, ui);

    if (!ImGui::BeginMainMenuBar())
    {
        return;
    }
    if (ImGui::BeginMenu("File"))
    {
        if (ImGui::MenuItem("New"))
        {
            requestNewProject(state, renderer, ui);
        }
        ImGui::BeginDisabled(busy);
        if (ImGui::MenuItem("Open...", "Ctrl+O"))
        {
            chooseOpenProject(state, ui);
        }
        if (ImGui::MenuItem("Save", "Ctrl+S"))
        {
            requestSave(state, ui);
        }
        if (ImGui::MenuItem("Save As...", "Ctrl+Shift+S"))
        {
            chooseSaveAs(state, ui);
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Add Layer..."))
        {
            chooseAddLayer(state, ui);
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Export PLY..."))
        {
            chooseExport(state, ui, PathPurpose::ExportPly);
        }
        if (ImGui::MenuItem("Export GLB..."))
        {
            chooseExport(state, ui, PathPurpose::ExportGlb);
        }
        if (ImGui::MenuItem("Export SPZ..."))
        {
            chooseExport(state, ui, PathPurpose::ExportSpz);
        }
        ImGui::EndDisabled();
        ImGui::Separator();
        // The typed-path fallback, for when a native dialog cannot be shown.
        if (ImGui::BeginMenu("Type a Path"))
        {
            if (ImGui::MenuItem("Open project..."))
            {
                openPathDialog(ui, PathPurpose::OpenProject, "");
            }
            if (ImGui::MenuItem("Save project as..."))
            {
                openPathDialog(ui, PathPurpose::SaveAs, state.projectPath);
            }
            if (ImGui::MenuItem("Add layer..."))
            {
                openPathDialog(ui, PathPurpose::AddLayer, "");
            }
            if (ImGui::MenuItem("Export PLY..."))
            {
                openPathDialog(ui, PathPurpose::ExportPly, "");
            }
            if (ImGui::MenuItem("Export GLB..."))
            {
                openPathDialog(ui, PathPurpose::ExportGlb, "");
            }
            if (ImGui::MenuItem("Export SPZ..."))
            {
                openPathDialog(ui, PathPurpose::ExportSpz, "");
            }
            ImGui::EndMenu();
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Quit"))
        {
            requestQuit(state, ui);
        }
        ImGui::EndMenu();
    }
    drawEditMenu(state, renderer, ui);
    drawWindowMenu(ui);
    const bool helpOpen = ImGui::BeginMenu("Help");
    if (helpOpen)
    {
        // Read when the menu opens, so the state shown is current.
        if (!ui.helpMenuWasOpen)
        {
            refreshAssociation(ui);
        }
        drawAssociationItems(ui);
        ImGui::Separator();
        if (ImGui::MenuItem("About Sabercat"))
        {
            ui.aboutOpen = true;
        }
        ImGui::EndMenu();
    }
    ui.helpMenuWasOpen = helpOpen;
    if (!ui.statusMessage.empty())
    {
        // The result of the last Help action, dimmed at the right of the menu bar.
        const float width = ImGui::CalcTextSize(ui.statusMessage.c_str()).x;
        ImGui::SameLine(ImGui::GetWindowWidth() - width - 12.0f);
        ImGui::TextDisabled("%s", ui.statusMessage.c_str());
    }
    ImGui::EndMainMenuBar();
}

void drawLayersWindow(StudioState & state, Renderer & renderer, UiState & ui)
{
    ImGui::SetNextWindowPos(ImVec2(12.0f, 32.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(470.0f, 290.0f), ImGuiCond_FirstUseEver);
    ImGui::Begin("Layers");

    const int count = static_cast<int>(state.project.layers.size());
    if (ImGui::BeginTable("##layers", 6,
                          ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg
                              | ImGuiTableFlags_SizingStretchProp))
    {
        ImGui::TableSetupColumn("Vis", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn("Lock", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn("Name");
        ImGui::TableSetupColumn("Gaussians", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn("SH", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableSetupColumn("Relight", ImGuiTableColumnFlags_WidthFixed);
        ImGui::TableHeadersRow();

        for (int i = 0; i < count; ++i)
        {
            ProjectLayer & layer = state.project.layers[i];
            ImGui::PushID(i);
            ImGui::TableNextRow();

            ImGui::TableSetColumnIndex(0);
            bool visible = layer.visible;
            if (ImGui::Checkbox("##visible", &visible))
            {
                setLayerVisible(state, i, visible);
            }

            ImGui::TableSetColumnIndex(1);
            bool locked = layer.locked;
            if (ImGui::Checkbox("##locked", &locked))
            {
                setLayerLocked(state, i, locked);
            }
            if (ImGui::IsItemHovered())
            {
                ImGui::SetTooltip("Lock toggles are not undoable.");
            }

            ImGui::TableSetColumnIndex(2);
            const char * label = layer.name.empty() ? "(unnamed)" : layer.name.c_str();
            if (ImGui::Selectable(label, state.selectedLayer == i,
                                  ImGuiSelectableFlags_SpanAllColumns))
            {
                state.selectedLayer = i;
            }

            const GaussianCloud * cloud = layer.cloud.get();
            ImGui::TableSetColumnIndex(3);
            if (cloud)
            {
                ImGui::Text("%zu", cloud->count());
            }
            else
            {
                ImGui::TextUnformatted("-");
            }

            ImGui::TableSetColumnIndex(4);
            if (cloud)
            {
                ImGui::Text("%u", cloud->shDegree());
            }
            else
            {
                ImGui::TextUnformatted("-");
            }

            // Letters for the relight streams the layer carries: N normals,
            // B base color + roughness, M metallic, V visibility.
            ImGui::TableSetColumnIndex(5);
            std::string relight;
            if (cloud)
            {
                const auto add = [&relight](bool present, char letter)
                {
                    if (present)
                    {
                        if (!relight.empty())
                        {
                            relight += ' ';
                        }
                        relight += letter;
                    }
                };
                add(!cloud->normals().empty(), 'N');
                add(!cloud->baseColorRoughness().empty(), 'B');
                add(!cloud->metallic().empty(), 'M');
                add(!cloud->visibility().empty(), 'V');
            }
            ImGui::TextUnformatted(relight.empty() ? "-" : relight.c_str());

            ImGui::PopID();
        }
        ImGui::EndTable();
    }

    const bool hasSelection = state.selectedLayer >= 0 && state.selectedLayer < count;
    const bool selectedLocked = hasSelection && state.project.layers[state.selectedLayer].locked;

    // Add and Remove wait for an open dialog; Rename and the moves do not touch files.
    const bool busy = ui.dialog.pending();
    ImGui::Separator();
    ImGui::BeginDisabled(busy);
    if (ImGui::Button("Add..."))
    {
        chooseAddLayer(state, ui);
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(!hasSelection || selectedLocked || busy);
    if (ImGui::Button("Remove"))
    {
        // A layer's structure changes here: a live transform preview goes first.
        dropPreview(state, renderer, ui);
        removeSelectedLayer(state);
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(!hasSelection || selectedLocked);
    if (ImGui::Button("Rename..."))
    {
        std::snprintf(ui.nameBuffer, sizeof(ui.nameBuffer), "%s",
                      state.project.layers[state.selectedLayer].name.c_str());
        ui.renameLayerId = state.project.layers[state.selectedLayer].id;
        ui.openRenamePopup = true;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(!hasSelection || state.selectedLayer == 0);
    if (ImGui::Button("Up"))
    {
        dropPreview(state, renderer, ui);
        moveSelectedLayer(state, -1);
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(!hasSelection || state.selectedLayer == count - 1);
    if (ImGui::Button("Down"))
    {
        dropPreview(state, renderer, ui);
        moveSelectedLayer(state, +1);
    }
    ImGui::EndDisabled();
    if (selectedLocked)
    {
        ImGui::TextDisabled("Layer is locked: unlock it to remove, rename or move it.");
    }

    int visibleCount = 0;
    for (const ProjectLayer & layer : state.project.layers)
    {
        visibleCount += layer.visible ? 1 : 0;
    }
    ImGui::TextDisabled("%d layers, %d visible", count, visibleCount);
    ImGui::End();
}

void drawInspectorWindow(StudioState & state, UiState & ui)
{
    const ImGuiIO & io = ImGui::GetIO();
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - 330.0f, 32.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(318.0f, 300.0f), ImGuiCond_FirstUseEver);
    ImGui::Begin("Inspector");

    // A drag held from an earlier frame stays with its own layer (see the helpers above).
    const bool selectionMoved = reconcileLayerTransformCapture(state, ui);

    const int count = static_cast<int>(state.project.layers.size());
    if (state.selectedLayer < 0 || state.selectedLayer >= count)
    {
        ImGui::TextDisabled("No layer selected");
        ImGui::End();
        return;
    }

    const int layerIndex = state.selectedLayer;
    ProjectLayer & layer = state.project.layers[layerIndex];
    ImGui::Text("%s", layer.name.c_str());
    ImGui::TextDisabled("id %s%s", layer.id.c_str(), layer.locked ? " (locked)" : "");
    ImGui::Separator();

    // The quaternion is authoritative: Euler angles are derived from it each
    // frame and only written back (as a new quaternion) when the user edits.
    // The widgets edit a copy, which setLayerTransform() stores (refused while
    // the layer is locked, so the disabled widgets cannot change it either).
    // setLayerTransform() is not recorded: the drag as a whole is, below.
    const LayerTransform frameStart = layer.transform;
    LayerTransform edited = layer.transform;

    // A drag is an item of this block being active. DragFloat3 ends with its label
    // text, so IsItemActivated() after it would never see the drag: the block is
    // checked as a whole, with IsAnyItemActive() before and after it.
    const bool activeBefore = ImGui::IsAnyItemActive();
    ImGui::BeginDisabled(layer.locked);
    if (ImGui::DragFloat3("Translation", &edited.translation.x, 0.01f))
    {
        setLayerTransform(state, layerIndex, edited);
    }
    glm::vec3 euler = glm::degrees(glm::eulerAngles(edited.rotation));
    if (ImGui::DragFloat3("Rotation (deg)", &euler.x, 0.1f))
    {
        edited.rotation = glm::normalize(glm::quat(glm::radians(euler)));
        setLayerTransform(state, layerIndex, edited);
    }
    if (ImGui::DragFloat("Scale", &edited.scale, 0.01f, 1e-4f, 1e6f, "%.4f"))
    {
        setLayerTransform(state, layerIndex, edited);
    }
    const bool activeAfter = ImGui::IsAnyItemActive();

    // A drag starts when the block becomes active, and ends when it stops being active
    // (also when the selection moved under a drag that goes on: selectionMoved). The
    // drag is recorded from the transform at the start of the frame it began in.
    if (ui.layerTransformBeforeIndex < 0)
    {
        if (activeAfter && (!activeBefore || selectionMoved))
        {
            ui.layerTransformBefore = frameStart;
            ui.layerTransformBeforeIndex = layerIndex;
            ui.layerTransformBeforeId = layer.id;
        }
    }
    else if (!activeAfter)
    {
        endLayerTransformCapture(state, ui);
    }

    if (ImGui::Button("Reset transform"))
    {
        const LayerTransform before = layer.transform;
        setLayerTransform(state, layerIndex, LayerTransform{});
        commitLayerTransformEdit(state, layerIndex, before);
    }
    ImGui::EndDisabled();

    ImGui::Separator();
    const GaussianCloud * cloud = layer.cloud.get();
    if (cloud)
    {
        ImGui::Text("Gaussians: %zu", cloud->count());
        ImGui::Text("SH degree: %u", cloud->shDegree());
        ImGui::Text("Antialiased: %s", cloud->antialiased() ? "yes" : "no");
        const SceneBounds & bounds = cloud->bounds();
        if (bounds.valid)
        {
            ImGui::Text("Bounds (layer space):");
            ImGui::Text("  min %.3f %.3f %.3f", bounds.min.x, bounds.min.y, bounds.min.z);
            ImGui::Text("  max %.3f %.3f %.3f", bounds.max.x, bounds.max.y, bounds.max.z);
        }
    }
    ImGui::End();
}

void drawViewportWindow(StudioState & state, Renderer & renderer, UiState & ui)
{
    ImGui::SetNextWindowPos(ImVec2(12.0f, 335.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(470.0f, 260.0f), ImGuiCond_FirstUseEver);
    ImGui::Begin("Viewport");

    const ImGuiIO & io = ImGui::GetIO();
    ImGui::Text("%.1f FPS (%.2f ms/frame)", io.Framerate, 1000.0f / io.Framerate);
    ImGui::Text("Gaussians on screen: %u", renderer.gaussianCount());
    ImGui::Text("Scene rebuild: %.2f ms", state.lastRebuildMs);
    ImGui::Separator();

    if (ImGui::Button("Frame all"))
    {
        frameAll(state, renderer, ui);
    }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Move the camera back until the whole visible scene fits.");
    }
    ImGui::TextDisabled("Camera: left-drag orbit, right-drag pan, scroll zoom, Q/E roll.");
    ImGui::TextDisabled("With the Select window's Rectangle tool: Alt+left-drag orbits.");

    ImGui::Separator();
    ImGui::Text("Debug view:");
    for (size_t i = 0; i < app::kDebugViewCycle.size(); ++i)
    {
        if (ImGui::RadioButton(app::debugViewName(app::kDebugViewCycle[i]),
                               state.debugViewIndex == static_cast<int>(i)))
        {
            state.debugViewIndex = static_cast<int>(i);
        }
    }

    ImGui::Separator();
    // The scene's own degree is shown; the slider still runs to 3, since a
    // layer added later may carry more (the renderer reads what is there).
    const int sceneSHDegree = static_cast<int>(renderer.sceneSHDegree());
    ImGui::SliderInt("Max SH degree", &state.shDegree, 0, 3);
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("SH0 = flat color. SH3 = full view-dependent color.");
    }
    ImGui::TextDisabled("Scene carries SH%d", sceneSHDegree);
    ImGui::End();
}

// The Select window's tool, op and fields, for an editable active layer.
void drawSelectControls(StudioState & state, Renderer & renderer, UiState & ui, const ProjectLayer & layer)
{
    SelectUi & sel = ui.select;
    if (ImGui::RadioButton("Rectangle", sel.tool == SelectTool::Rectangle))
    {
        sel.tool = SelectTool::Rectangle;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Box", sel.tool == SelectTool::Box))
    {
        sel.tool = SelectTool::Box;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Sphere", sel.tool == SelectTool::Sphere))
    {
        sel.tool = SelectTool::Sphere;
    }
    if (ImGui::RadioButton("Replace", sel.op == 0))
    {
        sel.op = 0;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Add", sel.op == 1))
    {
        sel.op = 1;
    }
    ImGui::SameLine();
    if (ImGui::RadioButton("Subtract", sel.op == 2))
    {
        sel.op = 2;
    }
    ImGui::TextDisabled("Shift = Add, Ctrl = Subtract while dragging");
    ImGui::Separator();

    if (sel.tool == SelectTool::Rectangle)
    {
        ImGui::TextDisabled("Drag in the viewport to select; Alt+drag orbits");
    }
    else if (sel.tool == SelectTool::Box)
    {
        ImGui::DragFloat3("Center", &sel.boxCenter.x, 0.01f);
        ImGui::DragFloat3("Half extents", &sel.boxHalfExtents.x, 0.01f, 0.0f, 1e6f);
        if (ImGui::Button("Fit to layer bounds"))
        {
            runAction(ui, [&]()
                      {
                          glm::vec3 center(0.0f);
                          glm::vec3 halfExtents(0.0f);
                          if (fitBoxToLayer(state, layer, center, halfExtents))
                          {
                              sel.boxCenter = center;
                              sel.boxHalfExtents = halfExtents;
                          }
                          else
                          {
                              ui.statusMessage = "The layer has no bounds to fit.";
                          }
                      });
        }
        ImGui::SameLine();
        if (ImGui::Button("Apply"))
        {
            runAction(ui, [&]() { selectBox(state, renderer, selectOpOf(sel.op), sel.boxCenter, sel.boxHalfExtents); });
        }
    }
    else
    {
        ImGui::DragFloat3("Center", &sel.sphereCenter.x, 0.01f);
        ImGui::DragFloat("Radius", &sel.sphereRadius, 0.01f, 0.0f, 1e6f, "%.4f");
        if (ImGui::Button("Apply"))
        {
            runAction(ui, [&]() { selectSphere(state, renderer, selectOpOf(sel.op), sel.sphereCenter, sel.sphereRadius); });
        }
    }

    ImGui::Separator();
    ImGui::Text("Selected %u of %zu", selectedCountOfActiveLayer(state), layer.cloud->count());
    if (ImGui::Button("Clear"))
    {
        runAction(ui, [&]() { selectNone(state, renderer); });
    }
    ImGui::SameLine();
    if (ImGui::Button("All"))
    {
        runAction(ui, [&]() { selectAllGaussians(state, renderer); });
    }
    ImGui::SameLine();
    if (ImGui::Button("Invert"))
    {
        runAction(ui, [&]() { invertSelection(state, renderer); });
    }

    // The highlight is display only; it tints the selected Gaussians in the viewport.
    ImGui::Separator();
    bool highlight = state.edit.highlight.enabled != 0;
    bool highlightChanged = false;
    if (ImGui::Checkbox("Highlight selection", &highlight))
    {
        state.edit.highlight.enabled = highlight ? 1u : 0u;
        highlightChanged = true;
    }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Tints the selected Gaussians in the viewport.");
    }
    if (ImGui::ColorEdit4("Tint", state.edit.highlight.tint))
    {
        highlightChanged = true;
    }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("The color the selected Gaussians move toward. The alpha is the strength.");
    }
    if (highlightChanged)
    {
        runAction(ui, [&]() { refreshHighlight(state, renderer); });
    }
    if (state.relight.enabled)
    {
        ImGui::TextDisabled("Highlight needs Relight off (it tints the unlit image)");
    }
}

void drawSelectWindow(StudioState & state, Renderer & renderer, UiState & ui)
{
    SelectUi & sel = ui.select;
    sel.rectangleArmed = false;
    if (!sel.open)
    {
        return;
    }
    ImGui::SetNextWindowPos(ImVec2(490.0f, 32.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(300.0f, 330.0f), ImGuiCond_FirstUseEver);
    ImGui::Begin("Select", &sel.open);

    ProjectLayer * layer = editableLayer(state);
    if (layer == nullptr)
    {
        ImGui::TextDisabled("%s", whyNotEditable(state));
    }
    else
    {
        drawSelectControls(state, renderer, ui, *layer);
    }
    ImGui::End();

    // The Rectangle tool is armed while its drag can start: the window is open, the
    // tool is Rectangle, the active layer is editable and no dialog or popup is up.
    // The main loop reads this next frame, so left drags select instead of orbiting.
    sel.rectangleArmed = sel.open && sel.tool == SelectTool::Rectangle && layer != nullptr
                         && !ui.dialog.pending() && !ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopupId);
}

// Applies the Transform fields to the active layer's selection, live or on Apply.
void drawTransformWindow(StudioState & state, Renderer & renderer, UiState & ui)
{
    TransformUi & tf = ui.transform;
    if (!tf.open)
    {
        // Closing the window drops a live preview: the layer goes back to its values.
        dropPreview(state, renderer, ui);
        return;
    }
    ImGui::SetNextWindowPos(ImVec2(490.0f, 370.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(300.0f, 330.0f), ImGuiCond_FirstUseEver);
    ImGui::Begin("Transform", &tf.open);

    ProjectLayer * layer = editableLayer(state);
    const uint32_t selected = selectedCountOfActiveLayer(state);
    const bool ready = layer != nullptr && selected > 0;

    // A preview belongs to one layer and one selection: switching either drops it.
    TransformPreview & preview = state.edit.preview;
    if (preview.active && (!ready || layer->id != preview.layerId))
    {
        dropPreview(state, renderer, ui);
    }
    if (!ready)
    {
        ImGui::TextDisabled("%s", layer == nullptr ? whyNotEditable(state) : "Select Gaussians first (Select window)");
    }

    if (ImGui::Checkbox("Live preview", &tf.livePreview))
    {
        if (!tf.livePreview)
        {
            dropPreview(state, renderer, ui);
        }
        else if (ready)
        {
            previewOrDrop(state, renderer, ui, tf.params);
        }
    }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Show the transform on the layer as you edit it. Apply makes it permanent.");
    }

    // Any drag in progress marks the scene as interacting (the renderer keeps its
    // conservative bounds); when it ends, the exact bounds come back.
    bool changed = false;
    bool anyActive = false;
    bool endedEdit = false;

    ImGui::BeginDisabled(!ready);
    int pivot = static_cast<int>(tf.params.pivot);
    if (ImGui::Combo("Pivot", &pivot, kPivotNames, IM_ARRAYSIZE(kPivotNames)))
    {
        tf.params.pivot = static_cast<PivotMode>(pivot);
        changed = true;
    }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("The point the rotation and scale turn about.");
    }
    if (ImGui::DragFloat3("Translate", &tf.params.translation.x, 0.01f))
    {
        changed = true;
    }
    anyActive |= ImGui::IsAnyItemActive();
    endedEdit |= ImGui::IsItemDeactivatedAfterEdit();
    if (ImGui::DragFloat3("Rotate (deg)", &tf.params.rotationDegrees.x, 0.1f))
    {
        changed = true;
    }
    anyActive |= ImGui::IsAnyItemActive();
    endedEdit |= ImGui::IsItemDeactivatedAfterEdit();
    if (ImGui::DragFloat("Scale", &tf.params.scale, 0.005f, 1e-4f, 1e6f, "%.4f"))
    {
        changed = true;
    }
    anyActive |= ImGui::IsItemActive();
    endedEdit |= ImGui::IsItemDeactivatedAfterEdit();
    ImGui::EndDisabled();

    if (anyActive)
    {
        state.edit.interacting = true;
    }
    else if (state.edit.interacting || endedEdit)
    {
        state.edit.interacting = false;
        state.sceneDirty = true;
    }

    if (changed)
    {
        tf.dirtyFields = true;
        if (tf.livePreview && ready)
        {
            previewOrDrop(state, renderer, ui, tf.params);
        }
    }
    if (tf.dirtyFields && !tf.livePreview)
    {
        ImGui::TextDisabled("Press Apply to transform the selection.");
    }

    ImGui::BeginDisabled(!ready || isIdentityParams(tf.params));
    if (ImGui::Button("Apply"))
    {
        bool committed = false;
        runAction(ui, [&]() { commitTransform(state, renderer, tf.params); committed = true; });
        if (committed)
        {
            resetTransformFields(tf);
        }
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(!ready);
    if (ImGui::Button("Reset fields"))
    {
        bool canceled = false;
        runAction(ui, [&]() { cancelTransformPreview(state, renderer); canceled = true; });
        if (canceled)
        {
            resetTransformFields(tf);
        }
    }
    ImGui::EndDisabled();
    ImGui::End();
}

// The History window: every undo step in order, with the cursor between the applied steps
// (normal text) and the undone ones (greyed). The row at the cursor is selected. A click
// moves the cursor to that row's position, undoing or redoing as needed; jumpToHistory()
// cancels a live transform preview first.
void drawHistoryWindow(StudioState & state, Renderer & renderer, UiState & ui)
{
    if (!ui.historyOpen)
    {
        return;
    }
    ImGui::SetNextWindowPos(ImVec2(800.0f, 32.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(300.0f, 330.0f), ImGuiCond_FirstUseEver);
    ImGui::Begin("History", &ui.historyOpen);

    const EditHistory & history = state.edit.history;
    const std::size_t count = history.size();
    const std::size_t cursor = history.cursor();
    const auto jumpTo = [&](std::size_t index)
    {
        runAction(ui, [&]() { jumpToHistory(state, renderer, index); });
    };

    // The list scrolls; the footer below it stays in view.
    const float footerHeight = ImGui::GetTextLineHeightWithSpacing() * 2.0f
                               + ImGui::GetFrameHeightWithSpacing() * 2.0f + ImGui::GetStyle().ItemSpacing.y * 2.0f
                               + 1.0f;
    if (ImGui::BeginChild("##history_list", ImVec2(0.0f, -footerHeight), ImGuiChildFlags_Borders))
    {
        const float sizeWidth = ImGui::CalcTextSize("0000.0 KB").x + ImGui::GetStyle().CellPadding.x * 2.0f;
        if (ImGui::BeginTable("##history_rows", 2, ImGuiTableFlags_RowBg))
        {
            ImGui::TableSetupColumn("Step", ImGuiTableColumnFlags_WidthStretch);
            ImGui::TableSetupColumn("Size", ImGuiTableColumnFlags_WidthFixed, sizeWidth);

            // Position 0: before the first command.
            ImGui::TableNextRow();
            ImGui::TableSetColumnIndex(0);
            if (ImGui::Selectable("(initial state)", cursor == 0, ImGuiSelectableFlags_SpanAllColumns))
            {
                jumpTo(0);
            }

            // Entry i sits at position i + 1: the cursor is after it when it is applied.
            for (std::size_t i = 0; i < count; ++i)
            {
                const bool applied = i < cursor;
                ImGui::PushID(static_cast<int>(i));
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                if (!applied)
                {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImGui::GetStyleColorVec4(ImGuiCol_TextDisabled));
                }
                if (ImGui::Selectable(history.label(i).c_str(), cursor == i + 1, ImGuiSelectableFlags_SpanAllColumns))
                {
                    jumpTo(i + 1);
                }

                ImGui::TableSetColumnIndex(1);
                char size[32];
                std::snprintf(size, sizeof(size), "%.1f KB", static_cast<double>(history.entryBytes(i)) / kKilobyte);
                const float textWidth = ImGui::CalcTextSize(size).x;
                ImGui::SetCursorPosX(ImGui::GetCursorPosX() + ImGui::GetContentRegionAvail().x - textWidth);
                ImGui::TextUnformatted(size);
                if (!applied)
                {
                    ImGui::PopStyleColor();
                }
                ImGui::PopID();
            }
            ImGui::EndTable();
        }
    }
    ImGui::EndChild();

    ImGui::Separator();
    ImGui::Text("%zu entries, cursor %zu", count, cursor);
    if (history.evictedCount() > 0)
    {
        ImGui::TextDisabled("evicted: %llu", static_cast<unsigned long long>(history.evictedCount()));
    }
    const double used = static_cast<double>(history.bytesUsed());
    const double budget = static_cast<double>(history.budget());
    const float fraction = budget > 0.0 ? static_cast<float>(std::min(1.0, used / budget)) : 0.0f;
    char memory[64];
    std::snprintf(memory, sizeof(memory), "%.1f MB of %.1f MB", used / kMegabyte, budget / kMegabyte);
    ImGui::ProgressBar(fraction, ImVec2(-FLT_MIN, 0.0f), memory);

    // The same actions as the Edit menu's Undo and Redo.
    const bool canUndo = history.canUndo();
    const bool canRedo = history.canRedo();
    ImGui::BeginDisabled(!canUndo);
    if (ImGui::Button("Undo"))
    {
        runAction(ui, [&]() { undo(state, renderer); });
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(!canRedo);
    if (ImGui::Button("Redo"))
    {
        runAction(ui, [&]() { redo(state, renderer); });
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(count == 0);
    if (ImGui::Button("Clear history"))
    {
        clearEditHistory(state);
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
    {
        ImGui::SetTooltip("Forgets every undo step. Edits already made stay.");
    }
    ImGui::EndDisabled();
    ImGui::End();
}

// --- Viewport overlay ---------------------------------------------------------

// DisplayFramebufferScale (framebuffer pixels per window pixel), never zero.
ImVec2 framebufferScaleOf(const ImGuiIO & io)
{
    ImVec2 scale = io.DisplayFramebufferScale;
    if (!(scale.x > 0.0f))
    {
        scale.x = 1.0f;
    }
    if (!(scale.y > 0.0f))
    {
        scale.y = 1.0f;
    }
    return scale;
}

// Projects a world point into window pixels with the forward pass's camera. False for a
// point behind the camera (or one that does not project).
bool projectToWindow(const glm::dvec3 & point, const CameraUniforms & camera, const ImVec2 & scale, ImVec2 & out)
{
    glm::dvec2 pixel(0.0);
    double depth = 0.0;
    if (!projectToPixels(point, camera, pixel, depth))
    {
        return false;
    }
    out = ImVec2(static_cast<float>(pixel.x / scale.x), static_cast<float>(pixel.y / scale.y));
    return true;
}

// A wireframe segment; skipped when either end does not project.
void drawWireSegment(ImDrawList * draw, const glm::dvec3 & a, const glm::dvec3 & b, const CameraUniforms & camera,
                     const ImVec2 & scale)
{
    ImVec2 pa(0.0f, 0.0f);
    ImVec2 pb(0.0f, 0.0f);
    if (projectToWindow(a, camera, scale, pa) && projectToWindow(b, camera, scale, pb))
    {
        draw->AddLine(pa, pb, kSelectColor, 1.5f);
    }
}

// The twelve edges of the axis-aligned box `center` +- `halfExtents`.
void drawBoxWire(ImDrawList * draw, const glm::vec3 & center, const glm::vec3 & halfExtents,
                 const CameraUniforms & camera, const ImVec2 & scale)
{
    const glm::dvec3 c(center);
    const glm::dvec3 h(halfExtents);
    glm::dvec3 corners[8];
    for (int i = 0; i < 8; ++i)
    {
        corners[i] = c + glm::dvec3((i & 1) != 0 ? h.x : -h.x, (i & 2) != 0 ? h.y : -h.y, (i & 4) != 0 ? h.z : -h.z);
    }
    for (int i = 0; i < 8; ++i)
    {
        for (int axis = 0; axis < 3; ++axis)
        {
            const int bit = 1 << axis;
            if ((i & bit) == 0)
            {
                drawWireSegment(draw, corners[i], corners[i | bit], camera, scale);
            }
        }
    }
}

// Three great circles of the sphere `center`, `radius`, 48 segments each.
void drawSphereWire(ImDrawList * draw, const glm::vec3 & center, float radius, const CameraUniforms & camera,
                    const ImVec2 & scale)
{
    constexpr int kSegments = 48;
    constexpr double kTwoPi = 6.283185307179586;
    const glm::dvec3 c(center);
    const double r = static_cast<double>(radius);
    // Each circle is spanned by two orthogonal unit axes: XY, YZ and XZ.
    const glm::dvec3 spans[3][2] = {
        {glm::dvec3(1.0, 0.0, 0.0), glm::dvec3(0.0, 1.0, 0.0)},
        {glm::dvec3(0.0, 1.0, 0.0), glm::dvec3(0.0, 0.0, 1.0)},
        {glm::dvec3(1.0, 0.0, 0.0), glm::dvec3(0.0, 0.0, 1.0)},
    };
    for (const auto & span : spans)
    {
        glm::dvec3 previous = c + r * span[0];
        for (int s = 1; s <= kSegments; ++s)
        {
            const double t = kTwoPi * static_cast<double>(s) / static_cast<double>(kSegments);
            const glm::dvec3 next = c + r * (std::cos(t) * span[0] + std::sin(t) * span[1]);
            drawWireSegment(draw, previous, next, camera, scale);
            previous = next;
        }
    }
}

// The viewport overlay: the Rectangle drag (start, rubber band, release), the Box and
// Sphere wireframes, and the "Selected N" label of the active layer.
void drawSelectionOverlay(StudioState & state, Renderer & renderer, UiState & ui)
{
    const ImGuiIO & io = ImGui::GetIO();
    SelectUi & sel = ui.select;
    ImDrawList * draw = ImGui::GetForegroundDrawList();
    const ImVec2 scale = framebufferScaleOf(io);
    const ImVec2 mouse = ImGui::GetMousePos();
    ProjectLayer * layer = editableLayer(state);

    if (sel.tool == SelectTool::Rectangle)
    {
        if (sel.dragging)
        {
            if (!sel.rectangleArmed || ImGui::IsKeyPressed(ImGuiKey_Escape))
            {
                sel.dragging = false;
            }
            else if (ImGui::IsMouseReleased(ImGuiMouseButton_Left))
            {
                sel.dragging = false;
                const float dx = mouse.x - sel.dragStart.x;
                const float dy = mouse.y - sel.dragStart.y;
                if (std::fabs(dx) >= kClickSlop || std::fabs(dy) >= kClickSlop)
                {
                    SelectOp op = selectOpOf(sel.op);
                    if (io.KeyShift)
                    {
                        op = SelectOp::Add;
                    }
                    else if (io.KeyCtrl)
                    {
                        op = SelectOp::Subtract;
                    }
                    const glm::vec2 cornerA(sel.dragStart.x * scale.x, sel.dragStart.y * scale.y);
                    const glm::vec2 cornerB(mouse.x * scale.x, mouse.y * scale.y);
                    runAction(ui, [&]() { selectRectangle(state, renderer, op, cornerA, cornerB); });
                }
            }
            else
            {
                const ImVec2 lo(std::min(sel.dragStart.x, mouse.x), std::min(sel.dragStart.y, mouse.y));
                const ImVec2 hi(std::max(sel.dragStart.x, mouse.x), std::max(sel.dragStart.y, mouse.y));
                draw->AddRectFilled(lo, hi, kSelectFill);
                draw->AddRect(lo, hi, kSelectColor, 0.0f, 0, 1.5f);
            }
        }
        else if (sel.rectangleArmed && !io.WantCaptureMouse && !io.KeyAlt
                 && ImGui::IsMouseClicked(ImGuiMouseButton_Left))
        {
            sel.dragging = true;
            sel.dragStart = mouse;
        }
    }
    else if (sel.open && layer != nullptr)
    {
        // The wireframe is projected with the last completed frame's camera; with no
        // frame yet there is nothing to project through, so nothing is drawn.
        CameraUniforms camera{};
        bool haveCamera = true;
        try
        {
            camera = renderer.lastForwardCamera();
        }
        catch (const std::exception &)
        {
            haveCamera = false;
        }
        if (haveCamera)
        {
            if (sel.tool == SelectTool::Box)
            {
                drawBoxWire(draw, sel.boxCenter, sel.boxHalfExtents, camera, scale);
            }
            else
            {
                drawSphereWire(draw, sel.sphereCenter, sel.sphereRadius, camera, scale);
            }
        }
    }

    if (layer != nullptr)
    {
        char text[32];
        std::snprintf(text, sizeof(text), "Selected %u",
                      static_cast<unsigned int>(selectedCountOfActiveLayer(state)));
        const ImGuiViewport * viewport = ImGui::GetMainViewport();
        const ImVec2 pos(viewport->Pos.x + 12.0f, viewport->Pos.y + ImGui::GetFrameHeight() + 6.0f);
        const ImVec2 size = ImGui::CalcTextSize(text);
        draw->AddRectFilled(ImVec2(pos.x - 4.0f, pos.y - 2.0f), ImVec2(pos.x + size.x + 4.0f, pos.y + size.y + 2.0f),
                            IM_COL32(0, 0, 0, 160), 3.0f);
        draw->AddText(pos, IM_COL32(255, 255, 255, 255), text);
    }
}

void drawDialogs(StudioState & state, Renderer & renderer, UiState & ui)
{
    // Path entry: one modal for every "type a path" action.
    if (ui.openPathPopup)
    {
        ImGui::OpenPopup(kPathPopup);
        ui.openPathPopup = false;
    }
    if (ImGui::BeginPopupModal(kPathPopup, nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::TextUnformatted(purposeText(ui.pathPurpose));
        // Browse fills the box from a native dialog; OK then confirms the path.
        // Both wait while a dialog is open (the box itself stays editable).
        const bool busy = ui.dialog.pending();
        ImGui::SetNextItemWidth(400.0f);
        const bool enter = ImGui::InputText("##path", ui.pathBuffer, sizeof(ui.pathBuffer),
                                            ImGuiInputTextFlags_EnterReturnsTrue);
        ImGui::SameLine();
        ImGui::BeginDisabled(busy);
        if (ImGui::Button("Browse...", ImVec2(110.0f, 0.0f)))
        {
            startDialogFor(state, ui, ui.pathPurpose, DialogUse::PathBox);
        }
        ImGui::SameLine();
        const bool ok = (ImGui::Button("OK", ImVec2(110.0f, 0.0f)) || enter) && !busy;
        ImGui::EndDisabled();
        ImGui::SameLine();
        const bool cancel = ImGui::Button("Cancel", ImVec2(110.0f, 0.0f));
        if (ui.pathPurpose == PathPurpose::OpenProject || ui.pathPurpose == PathPurpose::SaveAs)
        {
            ImGui::TextDisabled("A .sabercat project file.");
        }
        else if (ui.pathPurpose == PathPurpose::AddLayer)
        {
            ImGui::TextDisabled(".ply .splat .spz .ksplat .sog .gltf .glb, or a cloud-v1 manifest.json");
        }
        if (ok && ui.pathBuffer[0] != '\0')
        {
            const std::string path = ui.pathBuffer;
            ImGui::CloseCurrentPopup();
            confirmPath(state, renderer, ui, ui.pathPurpose, path);
        }
        else if (cancel)
        {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    if (ui.openDiscardPopup)
    {
        ImGui::OpenPopup(kDiscardPopup);
        ui.openDiscardPopup = false;
    }
    if (ImGui::BeginPopupModal(kDiscardPopup, nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::TextUnformatted("The project has unsaved changes. Discard them?");
        if (ImGui::Button("Yes", ImVec2(120.0f, 0.0f)))
        {
            const PendingAction action = ui.pending;
            ui.pending = PendingAction::None;
            ImGui::CloseCurrentPopup();
            switch (action)
            {
            case PendingAction::NewProject:
                dropPreview(state, renderer, ui);
                newProject(state);
                break;
            case PendingAction::OpenProject:
            {
                const std::string path = ui.pendingOpenPath;
                runAction(ui, [&]() { openProject(state, renderer, path); });
                break;
            }
            case PendingAction::Quit:
                ui.quitConfirmed = true;
                break;
            case PendingAction::None:
                break;
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("No", ImVec2(120.0f, 0.0f)))
        {
            ui.pending = PendingAction::None;
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    if (ui.openAssociationPopup)
    {
        ImGui::OpenPopup(kAssociationPopup);
        ui.openAssociationPopup = false;
    }
    if (ImGui::BeginPopupModal(kAssociationPopup, nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::TextUnformatted("The .sabercat files now open with another copy of Studio or another program.");
        ImGui::TextUnformatted("They will open with this copy instead.");
        if (ImGui::Button("Yes", ImVec2(120.0f, 0.0f)))
        {
            ImGui::CloseCurrentPopup();
            registerThisCopy(ui);
        }
        ImGui::SameLine();
        if (ImGui::Button("No", ImVec2(120.0f, 0.0f)))
        {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    if (ui.openRenamePopup)
    {
        ImGui::OpenPopup(kRenamePopup);
        ui.openRenamePopup = false;
    }
    if (ImGui::BeginPopupModal(kRenamePopup, nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::InputText("Name", ui.nameBuffer, sizeof(ui.nameBuffer));
        if (ImGui::Button("OK", ImVec2(120.0f, 0.0f)))
        {
            // The layer is found by the id it had when the popup opened; a layer
            // that is gone (or moved) is not renamed by index.
            const int index = indexOfLayerId(state, ui.renameLayerId);
            if (index >= 0)
            {
                renameLayer(state, index, ui.nameBuffer);
            }
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::Button("Cancel", ImVec2(120.0f, 0.0f)))
        {
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }

    if (ui.openErrorPopup)
    {
        ImGui::OpenPopup(kErrorPopup);
        ui.openErrorPopup = false;
    }
    if (ImGui::BeginPopupModal(kErrorPopup, nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::PushTextWrapPos(480.0f);
        ImGui::TextWrapped("%s", ui.errorMessage.c_str());
        ImGui::PopTextWrapPos();
        if (ImGui::Button("OK", ImVec2(120.0f, 0.0f)))
        {
            ui.errorMessage.clear();
            ImGui::CloseCurrentPopup();
        }
        ImGui::EndPopup();
    }
}

} // namespace

void reportError(UiState & ui, const std::string & message)
{
    Logger::error("Studio: " + message);
    ui.errorMessage = message;
    ui.openErrorPopup = true;
}

void requestNewProject(StudioState & state, Renderer & renderer, UiState & ui)
{
    if (state.dirty)
    {
        ui.pending = PendingAction::NewProject;
        ui.openDiscardPopup = true;
        return;
    }
    dropPreview(state, renderer, ui);
    newProject(state);
}

void requestOpenProject(StudioState & state, Renderer & renderer, UiState & ui,
                        const std::string & path)
{
    if (state.dirty)
    {
        ui.pending = PendingAction::OpenProject;
        ui.pendingOpenPath = path;
        ui.openDiscardPopup = true;
        return;
    }
    dropPreview(state, renderer, ui);
    runAction(ui, [&]() { openProject(state, renderer, path); });
}

void requestQuit(StudioState & state, UiState & ui)
{
    if (state.dirty)
    {
        ui.pending = PendingAction::Quit;
        ui.openDiscardPopup = true;
        return;
    }
    ui.quitConfirmed = true;
}

void requestAddLayer(StudioState & state, Renderer & renderer, UiState & ui,
                     const std::string & path)
{
    dropPreview(state, renderer, ui);
    runAction(ui, [&]() { addLayerFromFile(state, renderer.sabercatDevice(), path); });
}

void drawStudioUi(StudioState & state, Renderer & renderer, UiState & ui)
{
    // First, so a result that opens a popup or a discard prompt shows this frame.
    pollDialog(state, renderer, ui);
    drawMenuBar(state, renderer, ui);
    drawLayersWindow(state, renderer, ui);
    drawInspectorWindow(state, ui);
    drawViewportWindow(state, renderer, ui);
    drawSelectWindow(state, renderer, ui);
    drawTransformWindow(state, renderer, ui);
    drawHistoryWindow(state, renderer, ui);
    drawRelightPanel(renderer, state.relight);
    // After the windows, so the overlay sits on top and the mouse state reflects them.
    drawSelectionOverlay(state, renderer, ui);
    app::drawAboutWindow(ui.aboutOpen, ui.systemInfo, ui.aboutLogo);
    drawDialogs(state, renderer, ui);
}

} // namespace sabercat::studio
