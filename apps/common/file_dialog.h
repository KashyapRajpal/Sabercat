// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Native Windows file dialogs for the apps: the system's own Open, Save and
// folder-picker windows (IFileOpenDialog / IFileSaveDialog). Every call returns
// the chosen path as UTF-8, or std::nullopt when the user cancels or the dialog
// cannot be shown. A failure is logged as a warning, so the caller falls back
// to the path box and the app keeps working.
//
// The pure helpers at the top (filter specs, default extensions, UTF-8 and
// UTF-16 conversion) hold no COM state and are unit-tested directly.
#pragma once

#include <functional>
#include <future>
#include <optional>
#include <string>
#include <thread>
#include <vector>

struct GLFWwindow;

namespace sabercat::app
{

// One entry of a dialog's file-type list. `patterns` is a semicolon-separated
// list of globs, e.g. "*.ply;*.splat".
struct FileFilter
{
    std::string name;
    std::string patterns;
};

// A filter as the dialog takes it: UTF-16 display name and pattern list.
struct FilterSpec
{
    std::wstring name;
    std::wstring spec;
};

// The file types the apps offer. Scene files are every format the loader reads
// (a SOG meta.json, and a cloud-v1 manifest.json, which opens its folder); the
// export and project lists each name one format.
std::vector<FileFilter> sceneFileFilters();
std::vector<FileFilter> projectFileFilters();
std::vector<FileFilter> plyFileFilters();
std::vector<FileFilter> glbFileFilters();

// Converts each FileFilter into the wide strings COMDLG_FILTERSPEC points at.
std::vector<FilterSpec> buildFilterSpecs(const std::vector<FileFilter> & filters);

// `name` with `extension` appended when it has no extension of its own. The
// extension may be given with or without its leading dot ("sabercat" or
// ".sabercat"). A name with any extension (even another one) is returned
// unchanged; a trailing dot counts as no extension. Dots inside directory names
// are not extensions.
std::string withDefaultExtension(const std::string & name, const std::string & extension);

// UTF-8 <-> UTF-16 (Windows wide strings). Invalid input is replaced with U+FFFD.
std::wstring utf8ToWide(const std::string & utf8);
std::string wideToUtf8(const std::wstring & wide);

// Native dialogs. `owner` is the window the dialog is modal to (may be null).
// `clientGuidTag` names the dialog's remembered settings (the last folder it
// opened): each caller passes its own tag, such as "viewer.open".
std::optional<std::string> openFileDialog(GLFWwindow * owner, const std::vector<FileFilter> & filters,
                                          const std::string & title, const char * clientGuidTag);

// The name is pre-filled; the default extension is appended when the user
// types a name without one. Asks before overwriting an existing file.
std::optional<std::string> saveFileDialog(GLFWwindow * owner, const std::vector<FileFilter> & filters,
                                          const std::string & defaultName,
                                          const std::string & defaultExtension, const char * clientGuidTag);

// One native dialog to run: the open or save form of the calls above.
struct DialogRequest
{
    enum class Kind
    {
        Open,
        Save,
    };

    Kind kind = Kind::Open;
    GLFWwindow * owner = nullptr;
    std::vector<FileFilter> filters;
    std::string title;            // Open only
    std::string defaultName;      // Save only
    std::string defaultExtension; // Save only
    std::string clientGuidTag;
};

// Runs one native dialog at a time on a worker thread, so the owner window keeps
// rendering while the dialog is open. The dialog is modal to the owner window
// (its HWND), so the user cannot touch the window meanwhile; the owner must
// stay alive until the result is polled or the object is destroyed.
//
// Usage: start one with startOpen() or startSave(); call poll() every frame
// until it delivers the result. A start while a result is pending is refused.
class AsyncFileDialog
{
public:
    // What a worker runs for one request: the chosen path, or nullopt on cancel.
    using Runner = std::function<std::optional<std::string>(const DialogRequest &)>;

    // Runs the native dialogs.
    AsyncFileDialog();
    // Runs `runner` instead of the native dialogs (the unit-test seam).
    explicit AsyncFileDialog(Runner runner);

    // Joins the worker. A dialog still open holds the destructor until the user
    // closes it, since the window it is modal to is the caller's.
    ~AsyncFileDialog();

    AsyncFileDialog(const AsyncFileDialog &) = delete;
    AsyncFileDialog & operator=(const AsyncFileDialog &) = delete;

    // Each returns false, and starts nothing, while a result is pending. The
    // dialog's title and filters are copied; `owner` may be null.
    bool startOpen(GLFWwindow * owner, const std::vector<FileFilter> & filters, const std::string & title,
                   const std::string & clientGuidTag);
    bool startSave(GLFWwindow * owner, const std::vector<FileFilter> & filters,
                   const std::string & defaultName, const std::string & defaultExtension,
                   const std::string & clientGuidTag);

    // True from a successful start until poll() has delivered its result. It stays
    // true after the dialog closes, so the controls that start or act on files
    // stay disabled until the result has been handled.
    bool pending() const;

    // Empty while the dialog is open, and when nothing is pending. Otherwise the
    // result, delivered once: an engaged inner optional is the chosen path, a
    // disengaged one means the user canceled (or the dialog could not be shown).
    std::optional<std::optional<std::string>> poll();

private:
    bool start(DialogRequest request);

    Runner runner_;
    std::thread worker_;
    std::future<std::optional<std::string>> result_;
};

} // namespace sabercat::app
