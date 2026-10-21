// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Sabercat Studio's Dear ImGui surface: the main menu, the Layers and Inspector
// windows, the Viewport window, and the modal dialogs (path entry, discard
// confirmation, rename, errors). The UI edits a StudioState and reads the
// renderer; it never touches Vulkan directly.
#pragma once

#include "file_association.h"
#include "file_dialog.h"
#include "system_info.h"

#include <imgui.h>

#include <string>

namespace sabercat
{
class Renderer;
} // namespace sabercat

struct GLFWwindow;

namespace sabercat::studio
{

struct StudioState;

// What the path dialog does with the path once it is confirmed.
enum class PathPurpose
{
    OpenProject,
    SaveAs,
    AddLayer,
    ExportPly,
    ExportGlb,
};

// What a finished native dialog's path is for: the path box (Browse in the
// path dialog), or the action the dialog was started for.
enum class DialogUse
{
    None,
    Action,
    PathBox,
};

// What the discard prompt (or a clean quit) does when the user says Yes.
enum class PendingAction
{
    None,
    NewProject,
    OpenProject,
    Quit,
};

// UI-only state: which dialogs are open, their text buffers, the error to
// show, and the action waiting on the discard prompt.
struct UiState
{
    GLFWwindow * window = nullptr; // owns the native file dialogs; set once at start-up

    // The one native dialog that may be open (it runs off the UI thread). While
    // pending(), the controls that start or act on files are disabled.
    app::AsyncFileDialog dialog;
    PathPurpose dialogPurpose = PathPurpose::OpenProject; // for DialogUse::Action
    DialogUse dialogUse = DialogUse::None;

    std::string errorMessage;    // non-empty: the error modal shows it
    std::string pendingOpenPath; // for PendingAction::OpenProject
    PendingAction pending = PendingAction::None;
    PathPurpose pathPurpose = PathPurpose::OpenProject;
    char pathBuffer[1024] = "";
    char nameBuffer[256] = "";

    // One-shot requests, consumed when the matching popup is opened.
    bool openPathPopup = false;
    bool openDiscardPopup = false;
    bool openRenamePopup = false;
    bool openErrorPopup = false;

    // Set once the user has confirmed quitting; the main loop then closes.
    bool quitConfirmed = false;

    // Help > About is open.
    bool aboutOpen = false;
    // What Help > About shows and copies; filled in once the device is up.
    app::SystemInfoFacts systemInfo;
    // The logo texture for Help > About (AboutLogo::texture); 0 until created.
    ImTextureID aboutLogo = 0;

    // Help > Register / Unregister .sabercat files. The state is read again
    // each time the Help menu opens, and after each change.
    std::wstring exePath; // this executable; set once at start-up
    app::AssociationState association = app::AssociationState::NotRegistered;
    bool helpMenuWasOpen = false;
    bool openAssociationPopup = false; // asks before pointing .sabercat files at this copy
    // The result of the last Help action, shown beside the menu bar.
    std::string statusMessage;
};

// Records an error for the error modal and logs it.
void reportError(UiState & ui, const std::string & message);

// File actions. Each one that would discard unsaved changes opens the discard
// prompt instead and runs when the user says Yes.
void requestNewProject(StudioState & state, UiState & ui);
void requestOpenProject(StudioState & state, Renderer & renderer, UiState & ui,
                        const std::string & path);
void requestQuit(StudioState & state, UiState & ui);
// Adds a scene file as a layer; a failure goes to the error modal.
void requestAddLayer(StudioState & state, Renderer & renderer, UiState & ui,
                     const std::string & path);

// Draws everything for one frame, after the ImGui frame has begun and before
// it is rendered.
void drawStudioUi(StudioState & state, Renderer & renderer, UiState & ui);

} // namespace sabercat::studio
