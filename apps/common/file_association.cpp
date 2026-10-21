// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "file_association.h"

#include "resources/sabercat_resource.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shlobj.h>
#include <wchar.h>
#include <vector>
#endif

namespace sabercat::app
{

namespace
{

// The per-user classes key the apps write when no administrator rights are
// available: HKEY_CURRENT_USER\Software\Classes.
const std::wstring kDefaultClassesSubkey = L"Software\\Classes";
const std::wstring kExtension = L".sabercat";
const std::wstring kProgId = L"Sabercat.Project";
const std::wstring kFriendlyName = L"Sabercat project";

std::wstring quoted(const std::wstring & text)
{
    return L"\"" + text + L"\"";
}

} // namespace

std::wstring buildOpenCommand(const std::wstring & exePath)
{
    return quoted(exePath) + L" \"%1\"";
}

std::wstring buildDefaultIconValue(const std::wstring & exePath, int resourceId)
{
    return quoted(exePath) + L",-" + std::to_wstring(resourceId);
}

std::wstring exePathFromOpenCommand(const std::wstring & command)
{
    const size_t start = command.find_first_not_of(L" \t");
    if (start == std::wstring::npos)
    {
        return {};
    }
    if (command[start] == L'"')
    {
        const size_t close = command.find(L'"', start + 1);
        if (close == std::wstring::npos)
        {
            return command.substr(start + 1);
        }
        return command.substr(start + 1, close - start - 1);
    }
    const size_t end = command.find(L' ', start);
    return command.substr(start, end == std::wstring::npos ? std::wstring::npos : end - start);
}

#ifdef _WIN32

namespace
{

std::wstring joinKey(const std::wstring & base, const std::wstring & child)
{
    return base + L"\\" + child;
}

// The keys this module owns, under a classes root.
std::wstring extensionKey(const std::wstring & root)
{
    return joinKey(root, kExtension);
}
std::wstring progIdKey(const std::wstring & root)
{
    return joinKey(root, kProgId);
}
std::wstring openWithKey(const std::wstring & root)
{
    return joinKey(extensionKey(root), L"OpenWithProgids");
}
std::wstring commandKey(const std::wstring & root)
{
    return joinKey(progIdKey(root), L"shell\\open\\command");
}
std::wstring iconKey(const std::wstring & root)
{
    return joinKey(progIdKey(root), L"DefaultIcon");
}

// Sets `error` and returns false unless the registry call succeeded.
bool succeeded(LONG status, std::string & error, const char * action)
{
    if (status == ERROR_SUCCESS)
    {
        return true;
    }
    error = std::string("Could not ") + action + " the registry (error " + std::to_string(status) + ").";
    return false;
}

// A string value under HKEY_CURRENT_USER\<subkey>; empty when the key or value is missing.
std::wstring readString(const std::wstring & subkey, const wchar_t * name)
{
    HKEY key = nullptr;
    if (RegOpenKeyExW(HKEY_CURRENT_USER, subkey.c_str(), 0, KEY_QUERY_VALUE, &key) != ERROR_SUCCESS)
    {
        return {};
    }
    std::wstring value;
    DWORD type = 0;
    DWORD bytes = 0;
    if (RegQueryValueExW(key, name, nullptr, &type, nullptr, &bytes) == ERROR_SUCCESS
        && (type == REG_SZ || type == REG_EXPAND_SZ) && bytes > 0)
    {
        // One spare character, so the value is always terminated.
        std::vector<wchar_t> buffer(bytes / sizeof(wchar_t) + 1, L'\0');
        DWORD size = bytes;
        if (RegQueryValueExW(key, name, nullptr, &type, reinterpret_cast<LPBYTE>(buffer.data()), &size)
            == ERROR_SUCCESS)
        {
            value = buffer.data();
        }
    }
    RegCloseKey(key);
    return value;
}

// Writes one value, creating its key (and any missing parents) first.
bool writeValue(const std::wstring & subkey, const wchar_t * name, DWORD type, const void * data, DWORD bytes,
                std::string & error)
{
    HKEY key = nullptr;
    LONG status = RegCreateKeyExW(HKEY_CURRENT_USER, subkey.c_str(), 0, nullptr, REG_OPTION_NON_VOLATILE,
                                  KEY_WRITE, nullptr, &key, nullptr);
    if (status == ERROR_SUCCESS)
    {
        status = RegSetValueExW(key, name, 0, type, static_cast<const BYTE *>(data), bytes);
        RegCloseKey(key);
    }
    return succeeded(status, error, "write to");
}

bool writeString(const std::wstring & subkey, const wchar_t * name, const std::wstring & value, std::string & error)
{
    return writeValue(subkey, name, REG_SZ, value.c_str(),
                      static_cast<DWORD>((value.size() + 1) * sizeof(wchar_t)), error);
}

// Removes one value. A missing key or value is not an error.
bool removeValue(const std::wstring & subkey, const wchar_t * name, std::string & error)
{
    HKEY key = nullptr;
    if (RegOpenKeyExW(HKEY_CURRENT_USER, subkey.c_str(), 0, KEY_SET_VALUE, &key) != ERROR_SUCCESS)
    {
        return true;
    }
    const LONG status = RegDeleteValueW(key, name);
    RegCloseKey(key);
    return status == ERROR_FILE_NOT_FOUND || succeeded(status, error, "clean up");
}

// Removes a key only when nothing is left in it. A missing key is not an error.
bool removeKeyIfEmpty(const std::wstring & subkey, std::string & error)
{
    HKEY key = nullptr;
    if (RegOpenKeyExW(HKEY_CURRENT_USER, subkey.c_str(), 0, KEY_QUERY_VALUE, &key) != ERROR_SUCCESS)
    {
        return true;
    }
    DWORD subkeys = 0;
    DWORD values = 0;
    const LONG status = RegQueryInfoKeyW(key, nullptr, nullptr, nullptr, &subkeys, nullptr, nullptr, &values,
                                         nullptr, nullptr, nullptr, nullptr);
    RegCloseKey(key);
    if (status != ERROR_SUCCESS || subkeys != 0 || values != 0)
    {
        return true;
    }
    return succeeded(RegDeleteKeyW(HKEY_CURRENT_USER, subkey.c_str()), error, "clean up");
}

// Removes a whole key tree we own. A missing key is not an error.
bool removeTree(const std::wstring & subkey, std::string & error)
{
    const LONG status = RegDeleteTreeW(HKEY_CURRENT_USER, subkey.c_str());
    return status == ERROR_FILE_NOT_FOUND || succeeded(status, error, "clean up");
}

// The shell reads file associations once and caches them; this tells it they
// changed so a double-click picks up the new program at once.
void notifyAssociationChanged()
{
    SHChangeNotify(SHCNE_ASSOCCHANGED, SHCNF_IDLIST, nullptr, nullptr);
}

AssociationState queryAt(const std::wstring & root, const std::wstring & exePath)
{
    const std::wstring assigned = readString(extensionKey(root), L"");
    if (assigned.empty())
    {
        return AssociationState::NotRegistered;
    }
    // Another program owns the extension now.
    if (_wcsicmp(assigned.c_str(), kProgId.c_str()) != 0)
    {
        return AssociationState::OtherExe;
    }
    const std::wstring registered = exePathFromOpenCommand(readString(commandKey(root), L""));
    if (registered.empty())
    {
        return AssociationState::NotRegistered;
    }
    return _wcsicmp(registered.c_str(), exePath.c_str()) == 0 ? AssociationState::ThisExe
                                                              : AssociationState::OtherExe;
}

bool registerAt(const std::wstring & root, const std::wstring & exePath, std::string & error)
{
    const bool ok = writeString(extensionKey(root), L"", kProgId, error)
        && writeValue(openWithKey(root), kProgId.c_str(), REG_NONE, nullptr, 0, error)
        && writeString(progIdKey(root), L"", kFriendlyName, error)
        && writeString(progIdKey(root), L"FriendlyTypeName", kFriendlyName, error)
        && writeString(iconKey(root), L"", buildDefaultIconValue(exePath, IDI_SABERCAT), error)
        && writeString(commandKey(root), L"", buildOpenCommand(exePath), error);
    if (ok)
    {
        notifyAssociationChanged();
    }
    return ok;
}

bool unregisterAt(const std::wstring & root, const std::wstring & exePath, std::string & error)
{
    // Another copy's registration is never removed.
    const std::wstring registered = exePathFromOpenCommand(readString(commandKey(root), L""));
    if (!registered.empty() && _wcsicmp(registered.c_str(), exePath.c_str()) != 0)
    {
        error = "The .sabercat association belongs to another copy of Sabercat Studio, so it was left unchanged.";
        return false;
    }

    // The extension's default value is ours only while it names our ProgID.
    const bool ownsExtension = _wcsicmp(readString(extensionKey(root), L"").c_str(), kProgId.c_str()) == 0;
    bool ok = true;
    if (ownsExtension)
    {
        ok = removeValue(extensionKey(root), L"", error);
    }
    ok = ok && removeValue(openWithKey(root), kProgId.c_str(), error);
    ok = ok && removeKeyIfEmpty(openWithKey(root), error);
    ok = ok && removeKeyIfEmpty(extensionKey(root), error);
    ok = ok && removeTree(progIdKey(root), error);
    if (ok)
    {
        notifyAssociationChanged();
    }
    return ok;
}

} // namespace

AssociationState queryProjectAssociationAt(const std::wstring & classesSubkey, const std::wstring & exePath)
{
    return queryAt(classesSubkey, exePath);
}

bool registerProjectAssociationAt(const std::wstring & classesSubkey, const std::wstring & exePath,
                                  std::string & error)
{
    return registerAt(classesSubkey, exePath, error);
}

bool unregisterProjectAssociationAt(const std::wstring & classesSubkey, const std::wstring & exePath,
                                    std::string & error)
{
    return unregisterAt(classesSubkey, exePath, error);
}

std::wstring currentExecutablePath()
{
    std::vector<wchar_t> buffer(MAX_PATH);
    for (;;)
    {
        const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
        if (length == 0)
        {
            return {};
        }
        if (length < buffer.size())
        {
            return std::wstring(buffer.data(), length);
        }
        buffer.resize(buffer.size() * 2);
    }
}

#else

AssociationState queryProjectAssociationAt(const std::wstring &, const std::wstring &)
{
    return AssociationState::NotRegistered;
}

bool registerProjectAssociationAt(const std::wstring &, const std::wstring &, std::string & error)
{
    error = "File associations are only supported on Windows.";
    return false;
}

bool unregisterProjectAssociationAt(const std::wstring &, const std::wstring &, std::string & error)
{
    error = "File associations are only supported on Windows.";
    return false;
}

std::wstring currentExecutablePath()
{
    return {};
}

#endif

AssociationState queryProjectAssociation(const std::wstring & exePath)
{
    return queryProjectAssociationAt(kDefaultClassesSubkey, exePath);
}

bool registerProjectAssociation(const std::wstring & exePath, std::string & error)
{
    return registerProjectAssociationAt(kDefaultClassesSubkey, exePath, error);
}

bool unregisterProjectAssociation(const std::wstring & exePath, std::string & error)
{
    return unregisterProjectAssociationAt(kDefaultClassesSubkey, exePath, error);
}

} // namespace sabercat::app
