// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "file_dialog.h"

#include "sabercat/logger.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <utility>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <objbase.h>
#include <shobjidl.h>
#endif

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

namespace sabercat::app
{

namespace
{

// Used when a filter has no patterns: every file.
constexpr const char * kAllPatterns = "*.*";

} // namespace

std::vector<FileFilter> sceneFileFilters()
{
    return {{"Gaussian splat scenes", "*.ply;*.splat;*.spz;*.ksplat;*.sog;*.gltf;*.glb;meta.json;manifest.json"},
            {"All files", "*.*"}};
}

std::vector<FileFilter> projectFileFilters()
{
    return {{"Sabercat projects", "*.sabercat"}, {"All files", "*.*"}};
}

std::vector<FileFilter> plyFileFilters()
{
    return {{"PLY", "*.ply"}, {"All files", "*.*"}};
}

std::vector<FileFilter> glbFileFilters()
{
    return {{"glTF binary", "*.glb"}, {"All files", "*.*"}};
}

std::vector<FilterSpec> buildFilterSpecs(const std::vector<FileFilter> & filters)
{
    std::vector<FilterSpec> specs;
    specs.reserve(filters.size());
    for (const FileFilter & filter : filters)
    {
        const std::string patterns = filter.patterns.empty() ? std::string(kAllPatterns) : filter.patterns;
        specs.push_back(FilterSpec{utf8ToWide(filter.name), utf8ToWide(patterns)});
    }
    return specs;
}

std::string withDefaultExtension(const std::string & name, const std::string & extension)
{
    std::string ext = extension;
    if (!ext.empty() && ext.front() == '.')
    {
        ext.erase(0, 1);
    }
    if (name.empty() || ext.empty())
    {
        return name;
    }

    // The last dot after the final path separator is the extension separator.
    // Bytes of a multi-byte UTF-8 sequence are never ASCII, so this is safe for
    // non-ASCII names. A leading dot (".hidden") is not an extension.
    const size_t separator = name.find_last_of("/\\");
    const size_t fileStart = (separator == std::string::npos) ? 0 : separator + 1;
    const size_t dot = name.find_last_of('.');
    const bool hasDot = dot != std::string::npos && dot > fileStart;
    if (hasDot && dot + 1 < name.size())
    {
        return name; // already has an extension, of any kind
    }
    const std::string stem = hasDot ? name.substr(0, dot) : name; // "name." drops its dot
    return stem + "." + ext;
}

#ifdef _WIN32

std::wstring utf8ToWide(const std::string & utf8)
{
    if (utf8.empty())
    {
        return std::wstring();
    }
    const int needed = MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()),
                                           nullptr, 0);
    if (needed <= 0)
    {
        return std::wstring();
    }
    std::wstring wide(static_cast<size_t>(needed), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, utf8.data(), static_cast<int>(utf8.size()), wide.data(), needed);
    return wide;
}

std::string wideToUtf8(const std::wstring & wide)
{
    if (wide.empty())
    {
        return std::string();
    }
    const int needed = WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()),
                                           nullptr, 0, nullptr, nullptr);
    if (needed <= 0)
    {
        return std::string();
    }
    std::string utf8(static_cast<size_t>(needed), '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide.data(), static_cast<int>(wide.size()), utf8.data(), needed,
                        nullptr, nullptr);
    return utf8;
}

namespace
{

// Balances CoInitializeEx on the calling thread. S_OK and S_FALSE both succeed
// and both must be matched by CoUninitialize. RPC_E_CHANGED_MODE means the
// thread is already in the other mode: COM works there, and it is not ours to
// uninitialize.
class ComApartment
{
public:
    ComApartment() : hr_(CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED)) {}
    ~ComApartment()
    {
        if (SUCCEEDED(hr_))
        {
            CoUninitialize();
        }
    }
    ComApartment(const ComApartment &) = delete;
    ComApartment & operator=(const ComApartment &) = delete;

    bool usable() const { return SUCCEEDED(hr_) || hr_ == RPC_E_CHANGED_MODE; }
    HRESULT hr() const { return hr_; }

private:
    HRESULT hr_;
};

// Releases a COM interface when it leaves scope.
template <class T>
class ComHandle
{
public:
    ComHandle() = default;
    ~ComHandle()
    {
        if (ptr_)
        {
            ptr_->Release();
        }
    }
    ComHandle(const ComHandle &) = delete;
    ComHandle & operator=(const ComHandle &) = delete;

    T * get() const { return ptr_; }
    T * operator->() const { return ptr_; }
    T ** put() { return &ptr_; }

private:
    T * ptr_ = nullptr;
};

// Logs a COM failure and gives the caller nothing, so the app falls back to
// the path box.
std::optional<std::string> comFailed(const char * what, HRESULT hr)
{
    char code[16];
    std::snprintf(code, sizeof(code), "0x%08lX", static_cast<unsigned long>(hr));
    Logger::warning(std::string("File dialog: ") + what + " failed (HRESULT " + code
                    + "); type the path instead");
    return std::nullopt;
}

// A stable GUID for a dialog tag (FNV-1a in two lanes). Each tag gets its own
// GUID, so each dialog remembers its own last folder across runs.
GUID clientGuidFor(const char * tag)
{
    uint64_t lo = 0xcbf29ce484222325ull;
    uint64_t hi = 0x84222325cbf29ce4ull;
    for (const char * p = tag; *p != '\0'; ++p)
    {
        const uint64_t byte = static_cast<unsigned char>(*p);
        lo = (lo ^ byte) * 0x100000001b3ull;
        hi = (hi ^ byte) * 0x100000001b3ull + 0x9e3779b97f4a7c15ull;
    }
    GUID guid{};
    guid.Data1 = static_cast<unsigned long>(lo >> 32);
    guid.Data2 = static_cast<unsigned short>(lo >> 16);
    guid.Data3 = static_cast<unsigned short>(lo);
    for (int i = 0; i < 8; ++i)
    {
        guid.Data4[i] = static_cast<unsigned char>(hi >> (56 - 8 * i));
    }
    return guid;
}

// What every dialog shares: the remembered-folder GUID, the file types and the
// title. The filter specs must outlive the Show call.
HRESULT configureDialog(IFileDialog * dialog, const std::vector<FilterSpec> & specs,
                        const std::wstring & title, const char * clientGuidTag)
{
    if (clientGuidTag != nullptr)
    {
        const GUID guid = clientGuidFor(clientGuidTag);
        const HRESULT hr = dialog->SetClientGuid(guid);
        if (FAILED(hr))
        {
            return hr;
        }
    }
    if (!specs.empty())
    {
        std::vector<COMDLG_FILTERSPEC> types;
        types.reserve(specs.size());
        for (const FilterSpec & spec : specs)
        {
            types.push_back(COMDLG_FILTERSPEC{spec.name.c_str(), spec.spec.c_str()});
        }
        HRESULT hr = dialog->SetFileTypes(static_cast<UINT>(types.size()), types.data());
        if (FAILED(hr))
        {
            return hr;
        }
        hr = dialog->SetFileTypeIndex(1);
        if (FAILED(hr))
        {
            return hr;
        }
    }
    if (!title.empty())
    {
        return dialog->SetTitle(title.c_str());
    }
    return S_OK;
}

// Shows a dialog modal to `owner` and returns the chosen filesystem path.
std::optional<std::string> showAndTakePath(IFileDialog * dialog, GLFWwindow * owner, const char * what)
{
    HWND parent = owner != nullptr ? glfwGetWin32Window(owner) : nullptr;
    HRESULT hr = dialog->Show(parent);
    if (hr == HRESULT_FROM_WIN32(ERROR_CANCELLED))
    {
        return std::nullopt;
    }
    if (FAILED(hr))
    {
        return comFailed(what, hr);
    }

    ComHandle<IShellItem> item;
    hr = dialog->GetResult(item.put());
    if (FAILED(hr))
    {
        return comFailed("reading the chosen item", hr);
    }
    PWSTR path = nullptr;
    hr = item->GetDisplayName(SIGDN_FILESYSPATH, &path);
    if (FAILED(hr) || path == nullptr)
    {
        return comFailed("reading the chosen path", FAILED(hr) ? hr : E_FAIL);
    }
    const std::string utf8 = wideToUtf8(path);
    CoTaskMemFree(path);
    return utf8;
}

} // namespace

std::optional<std::string> openFileDialog(GLFWwindow * owner, const std::vector<FileFilter> & filters,
                                          const std::string & title, const char * clientGuidTag)
{
    ComApartment com;
    if (!com.usable())
    {
        return comFailed("COM initialization", com.hr());
    }
    ComHandle<IFileOpenDialog> dialog;
    HRESULT hr = CoCreateInstance(CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
                                  IID_PPV_ARGS(dialog.put()));
    if (FAILED(hr))
    {
        return comFailed("creating the open dialog", hr);
    }

    const std::vector<FilterSpec> specs = buildFilterSpecs(filters);
    const std::wstring wideTitle = utf8ToWide(title);
    hr = configureDialog(dialog.get(), specs, wideTitle, clientGuidTag);
    if (FAILED(hr))
    {
        return comFailed("configuring the open dialog", hr);
    }
    FILEOPENDIALOGOPTIONS options = {};
    hr = dialog->GetOptions(&options);
    if (FAILED(hr))
    {
        return comFailed("reading the open dialog options", hr);
    }
    hr = dialog->SetOptions(static_cast<FILEOPENDIALOGOPTIONS>(
        options | FOS_FORCEFILESYSTEM | FOS_FILEMUSTEXIST | FOS_PATHMUSTEXIST));
    if (FAILED(hr))
    {
        return comFailed("setting the open dialog options", hr);
    }
    return showAndTakePath(dialog.get(), owner, "showing the open dialog");
}

std::optional<std::string> saveFileDialog(GLFWwindow * owner, const std::vector<FileFilter> & filters,
                                          const std::string & defaultName,
                                          const std::string & defaultExtension, const char * clientGuidTag)
{
    ComApartment com;
    if (!com.usable())
    {
        return comFailed("COM initialization", com.hr());
    }
    ComHandle<IFileSaveDialog> dialog;
    HRESULT hr = CoCreateInstance(CLSID_FileSaveDialog, nullptr, CLSCTX_INPROC_SERVER,
                                  IID_PPV_ARGS(dialog.put()));
    if (FAILED(hr))
    {
        return comFailed("creating the save dialog", hr);
    }

    const std::vector<FilterSpec> specs = buildFilterSpecs(filters);
    hr = configureDialog(dialog.get(), specs, std::wstring(), clientGuidTag);
    if (FAILED(hr))
    {
        return comFailed("configuring the save dialog", hr);
    }
    FILEOPENDIALOGOPTIONS options = {};
    hr = dialog->GetOptions(&options);
    if (FAILED(hr))
    {
        return comFailed("reading the save dialog options", hr);
    }
    hr = dialog->SetOptions(static_cast<FILEOPENDIALOGOPTIONS>(
        options | FOS_FORCEFILESYSTEM | FOS_OVERWRITEPROMPT | FOS_PATHMUSTEXIST));
    if (FAILED(hr))
    {
        return comFailed("setting the save dialog options", hr);
    }

    // The dialog appends the default extension itself when the typed name has
    // none; the wide name pre-fills the box.
    std::string extension = defaultExtension;
    if (!extension.empty() && extension.front() == '.')
    {
        extension.erase(0, 1);
    }
    if (!defaultName.empty())
    {
        hr = dialog->SetFileName(utf8ToWide(defaultName).c_str());
        if (FAILED(hr))
        {
            return comFailed("naming the save dialog", hr);
        }
    }
    if (!extension.empty())
    {
        hr = dialog->SetDefaultExtension(utf8ToWide(extension).c_str());
        if (FAILED(hr))
        {
            return comFailed("setting the save extension", hr);
        }
    }

    std::optional<std::string> chosen = showAndTakePath(dialog.get(), owner, "showing the save dialog");
    if (chosen && !extension.empty())
    {
        chosen = withDefaultExtension(*chosen, extension);
    }
    return chosen;
}

#else // !_WIN32

// The apps are Windows programs; elsewhere the dialogs are unavailable and the
// conversions are empty. The path box still works.

std::wstring utf8ToWide(const std::string &)
{
    return std::wstring();
}

std::string wideToUtf8(const std::wstring &)
{
    return std::string();
}

std::optional<std::string> openFileDialog(GLFWwindow *, const std::vector<FileFilter> &,
                                          const std::string &, const char *)
{
    return std::nullopt;
}

std::optional<std::string> saveFileDialog(GLFWwindow *, const std::vector<FileFilter> &,
                                          const std::string &, const std::string &, const char *)
{
    return std::nullopt;
}

#endif // _WIN32

namespace
{

// Runs one request with the native dialog it names. The dialogs initialize COM on
// the calling thread (see ComApartment), so this runs on the worker thread.
std::optional<std::string> runNativeDialog(const DialogRequest & request)
{
    if (request.kind == DialogRequest::Kind::Save)
    {
        return saveFileDialog(request.owner, request.filters, request.defaultName, request.defaultExtension,
                              request.clientGuidTag.c_str());
    }
    return openFileDialog(request.owner, request.filters, request.title, request.clientGuidTag.c_str());
}

} // namespace

AsyncFileDialog::AsyncFileDialog() : AsyncFileDialog(Runner(&runNativeDialog)) {}

AsyncFileDialog::AsyncFileDialog(Runner runner) : runner_(std::move(runner)) {}

AsyncFileDialog::~AsyncFileDialog()
{
    if (worker_.joinable())
    {
        worker_.join();
    }
}

bool AsyncFileDialog::startOpen(GLFWwindow * owner, const std::vector<FileFilter> & filters,
                                const std::string & title, const std::string & clientGuidTag)
{
    DialogRequest request;
    request.kind = DialogRequest::Kind::Open;
    request.owner = owner;
    request.filters = filters;
    request.title = title;
    request.clientGuidTag = clientGuidTag;
    return start(std::move(request));
}

bool AsyncFileDialog::startSave(GLFWwindow * owner, const std::vector<FileFilter> & filters,
                                const std::string & defaultName, const std::string & defaultExtension,
                                const std::string & clientGuidTag)
{
    DialogRequest request;
    request.kind = DialogRequest::Kind::Save;
    request.owner = owner;
    request.filters = filters;
    request.defaultName = defaultName;
    request.defaultExtension = defaultExtension;
    request.clientGuidTag = clientGuidTag;
    return start(std::move(request));
}

bool AsyncFileDialog::start(DialogRequest request)
{
    if (pending())
    {
        return false;
    }
    // A worker whose result was delivered has already been joined in poll().
    std::promise<std::optional<std::string>> promise;
    result_ = promise.get_future();
    worker_ = std::thread(
        [runner = runner_, request = std::move(request), promise = std::move(promise)]() mutable
        {
            try
            {
                promise.set_value(runner(request));
            }
            catch (const std::exception & e)
            {
                Logger::warning(std::string("File dialog: ") + e.what() + "; type the path instead");
                promise.set_value(std::nullopt);
            }
            catch (...)
            {
                Logger::warning("File dialog: unknown failure; type the path instead");
                promise.set_value(std::nullopt);
            }
        });
    return true;
}

bool AsyncFileDialog::pending() const
{
    return result_.valid();
}

std::optional<std::optional<std::string>> AsyncFileDialog::poll()
{
    if (!result_.valid() || result_.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
    {
        return std::nullopt;
    }
    std::optional<std::string> chosen = result_.get();
    if (worker_.joinable())
    {
        worker_.join();
    }
    std::optional<std::optional<std::string>> delivered(std::in_place, std::move(chosen));
    return delivered;
}

} // namespace sabercat::app
