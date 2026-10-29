// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "studio_ui.h"

#include "studio_state.h"

#include "about_panel.h"
#include "app_util.h"
#include "file_dialog.h"
#include "relight_panel.h"

#include "sabercat/gaussian_cloud.h"
#include "sabercat/logger.h"
#include "sabercat/renderer.h"

#include <imgui.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <cstdio>
#include <exception>
#include <filesystem>
#include <functional>

namespace sabercat::studio
{

namespace
{

constexpr const char * kPathPopup = "Path##studio";
constexpr const char * kDiscardPopup = "Discard unsaved changes?##studio";
constexpr const char * kRenamePopup = "Rename layer##studio";
constexpr const char * kErrorPopup = "Error##studio";
constexpr const char * kAssociationPopup = "Point .sabercat files at this copy of Studio?##studio";

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

void chooseExport(StudioState & state, UiState & ui, bool asPly)
{
    startDialogFor(state, ui, asPly ? PathPurpose::ExportPly : PathPurpose::ExportGlb, DialogUse::Action);
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
    }
    return "";
}

// Frames the renderer's current scene, as the viewer's Frame scene button does.
void frameAll(StudioState & state, Renderer & renderer)
{
    const SceneBounds bounds = renderer.sceneBounds();
    if (bounds.valid)
    {
        state.camera.frame(bounds, renderer.height(), static_cast<float>(renderer.height()));
    }
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

void drawMenuBar(StudioState & state, UiState & ui)
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

    if (!ImGui::BeginMainMenuBar())
    {
        return;
    }
    if (ImGui::BeginMenu("File"))
    {
        if (ImGui::MenuItem("New"))
        {
            requestNewProject(state, ui);
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
            chooseExport(state, ui, /*asPly=*/true);
        }
        if (ImGui::MenuItem("Export GLB..."))
        {
            chooseExport(state, ui, /*asPly=*/false);
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
            ImGui::EndMenu();
        }
        ImGui::Separator();
        if (ImGui::MenuItem("Quit"))
        {
            requestQuit(state, ui);
        }
        ImGui::EndMenu();
    }
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
        removeSelectedLayer(state);
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(!hasSelection || selectedLocked);
    if (ImGui::Button("Rename..."))
    {
        std::snprintf(ui.nameBuffer, sizeof(ui.nameBuffer), "%s",
                      state.project.layers[state.selectedLayer].name.c_str());
        ui.openRenamePopup = true;
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(!hasSelection || state.selectedLayer == 0);
    if (ImGui::Button("Up"))
    {
        moveSelectedLayer(state, -1);
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    ImGui::BeginDisabled(!hasSelection || state.selectedLayer == count - 1);
    if (ImGui::Button("Down"))
    {
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

void drawInspectorWindow(StudioState & state)
{
    const ImGuiIO & io = ImGui::GetIO();
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - 330.0f, 32.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(318.0f, 300.0f), ImGuiCond_FirstUseEver);
    ImGui::Begin("Inspector");

    const int count = static_cast<int>(state.project.layers.size());
    if (state.selectedLayer < 0 || state.selectedLayer >= count)
    {
        ImGui::TextDisabled("No layer selected");
        ImGui::End();
        return;
    }

    ProjectLayer & layer = state.project.layers[state.selectedLayer];
    ImGui::Text("%s", layer.name.c_str());
    ImGui::TextDisabled("id %s%s", layer.id.c_str(), layer.locked ? " (locked)" : "");
    ImGui::Separator();

    // The quaternion is authoritative: Euler angles are derived from it each
    // frame and only written back (as a new quaternion) when the user edits.
    // The widgets edit a copy, which setLayerTransform() stores (refused while
    // the layer is locked, so the disabled widgets cannot change it either).
    LayerTransform edited = layer.transform;
    ImGui::BeginDisabled(layer.locked);
    if (ImGui::DragFloat3("Translation", &edited.translation.x, 0.01f))
    {
        setLayerTransform(state, state.selectedLayer, edited);
    }
    glm::vec3 euler = glm::degrees(glm::eulerAngles(edited.rotation));
    if (ImGui::DragFloat3("Rotation (deg)", &euler.x, 0.1f))
    {
        edited.rotation = glm::normalize(glm::quat(glm::radians(euler)));
        setLayerTransform(state, state.selectedLayer, edited);
    }
    if (ImGui::DragFloat("Scale", &edited.scale, 0.01f, 1e-4f, 1e6f, "%.4f"))
    {
        setLayerTransform(state, state.selectedLayer, edited);
    }
    if (ImGui::Button("Reset transform"))
    {
        setLayerTransform(state, state.selectedLayer, LayerTransform{});
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

void drawViewportWindow(StudioState & state, Renderer & renderer)
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
        frameAll(state, renderer);
    }
    if (ImGui::IsItemHovered())
    {
        ImGui::SetTooltip("Move the camera back until the whole visible scene fits.");
    }
    ImGui::SameLine();
    ImGui::TextDisabled("Camera: left-drag orbit, right-drag pan, scroll zoom, Q/E roll");

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
            renameLayer(state, state.selectedLayer, ui.nameBuffer);
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

void requestNewProject(StudioState & state, UiState & ui)
{
    if (state.dirty)
    {
        ui.pending = PendingAction::NewProject;
        ui.openDiscardPopup = true;
        return;
    }
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
    runAction(ui, [&]() { addLayerFromFile(state, renderer.sabercatDevice(), path); });
}

void drawStudioUi(StudioState & state, Renderer & renderer, UiState & ui)
{
    // First, so a result that opens a popup or a discard prompt shows this frame.
    pollDialog(state, renderer, ui);
    drawMenuBar(state, ui);
    drawLayersWindow(state, renderer, ui);
    drawInspectorWindow(state);
    drawViewportWindow(state, renderer);
    drawRelightPanel(renderer, state.relight);
    app::drawAboutWindow(ui.aboutOpen, ui.systemInfo, ui.aboutLogo);
    drawDialogs(state, renderer, ui);
}

} // namespace sabercat::studio
