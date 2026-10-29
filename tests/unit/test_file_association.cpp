// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The per-user .sabercat association (apps/common/file_association.h): the open
// command, the icon value and the command parser on pure inputs. On Windows the
// register, query and unregister calls run against a scratch registry root,
// HKEY_CURRENT_USER\Software\SabercatTest, which is removed afterwards even when
// a check fails. The real HKEY_CURRENT_USER\Software\Classes is never touched.
#include "unit_test.h"

#include "file_association.h"
#include "resources/sabercat_resource.h"

#include <string>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace sabercat::unit
{

namespace
{

using app::AssociationState;

// Paths with a space and a non-ASCII folder name, the cases the quoting must
// survive. The non-ASCII character is written as a UTF-16 escape.
const std::wstring kThisExe = L"C:\\Program Files\\Sabercat \u00DC\\bin\\sabercat_studio.exe";
const std::wstring kOtherExe = L"D:\\Tools\\sabercat_studio.exe";

void testOpenCommand(Context & ctx)
{
    UNIT_CHECK(ctx, app::buildOpenCommand(L"C:\\Apps\\sabercat_studio.exe")
                        == L"\"C:\\Apps\\sabercat_studio.exe\" \"%1\"");
    // Spaces and non-ASCII stay inside the quotes, so the shell passes one argument.
    UNIT_CHECK(ctx, app::buildOpenCommand(kThisExe) == L"\"" + kThisExe + L"\" \"%1\"");
}

void testIconValue(Context & ctx)
{
    // The negative index names the resource ID, not the position of the icon.
    UNIT_CHECK(ctx, app::buildDefaultIconValue(L"C:\\Apps\\sabercat_studio.exe", 101)
                        == L"\"C:\\Apps\\sabercat_studio.exe\",-101");
    UNIT_CHECK(ctx, app::buildDefaultIconValue(kThisExe, IDI_SABERCAT) == L"\"" + kThisExe + L"\",-101");
}

void testCommandParsing(Context & ctx)
{
    // What the command was built from comes back out, spaces and non-ASCII included.
    UNIT_CHECK(ctx, app::exePathFromOpenCommand(app::buildOpenCommand(kThisExe)) == kThisExe);
    // An unquoted command: the first word.
    UNIT_CHECK(ctx, app::exePathFromOpenCommand(L"C:\\Apps\\studio.exe \"%1\"") == L"C:\\Apps\\studio.exe");
    UNIT_CHECK(ctx, app::exePathFromOpenCommand(L"  \"C:\\a b\\s.exe\" \"%1\"") == L"C:\\a b\\s.exe");
    UNIT_CHECK(ctx, app::exePathFromOpenCommand(L"") == L"");
    UNIT_CHECK(ctx, app::exePathFromOpenCommand(L"   ") == L"");
}

#ifdef _WIN32

// The scratch root: a subkey of HKEY_CURRENT_USER that only the tests use.
const std::wstring kScratchHive = L"Software\\SabercatTest";
const std::wstring kScratchRoot = kScratchHive + L"\\Classes";

// Removes the scratch key on construction and on destruction, so an earlier
// failed run leaves nothing behind and this run leaves nothing either.
struct ScratchGuard
{
    ScratchGuard()
    {
        RegDeleteTreeW(HKEY_CURRENT_USER, kScratchHive.c_str());
    }
    ~ScratchGuard()
    {
        RegDeleteTreeW(HKEY_CURRENT_USER, kScratchHive.c_str());
    }
};

bool keyPresent(const std::wstring & subkey)
{
    HKEY key = nullptr;
    if (RegOpenKeyExW(HKEY_CURRENT_USER, subkey.c_str(), 0, KEY_QUERY_VALUE, &key) != ERROR_SUCCESS)
    {
        return false;
    }
    RegCloseKey(key);
    return true;
}

// Whether the value `name` exists under the key, whatever its type.
bool valuePresent(const std::wstring & subkey, const wchar_t * name)
{
    HKEY key = nullptr;
    if (RegOpenKeyExW(HKEY_CURRENT_USER, subkey.c_str(), 0, KEY_QUERY_VALUE, &key) != ERROR_SUCCESS)
    {
        return false;
    }
    const bool present = RegQueryValueExW(key, name, nullptr, nullptr, nullptr, nullptr) == ERROR_SUCCESS;
    RegCloseKey(key);
    return present;
}

// A string value as text, or empty when it is missing.
std::wstring valueText(const std::wstring & subkey, const wchar_t * name)
{
    HKEY key = nullptr;
    if (RegOpenKeyExW(HKEY_CURRENT_USER, subkey.c_str(), 0, KEY_QUERY_VALUE, &key) != ERROR_SUCCESS)
    {
        return {};
    }
    wchar_t buffer[1024] = {};
    DWORD bytes = sizeof(buffer);
    DWORD type = 0;
    std::wstring text;
    if (RegQueryValueExW(key, name, nullptr, &type, reinterpret_cast<LPBYTE>(buffer), &bytes) == ERROR_SUCCESS
        && type == REG_SZ)
    {
        text = buffer;
    }
    RegCloseKey(key);
    return text;
}

void setDefaultText(const std::wstring & subkey, const std::wstring & text)
{
    HKEY key = nullptr;
    if (RegCreateKeyExW(HKEY_CURRENT_USER, subkey.c_str(), 0, nullptr, 0, KEY_WRITE, nullptr, &key, nullptr)
        == ERROR_SUCCESS)
    {
        RegSetValueExW(key, L"", 0, REG_SZ, reinterpret_cast<const BYTE *>(text.c_str()),
                       static_cast<DWORD>((text.size() + 1) * sizeof(wchar_t)));
        RegCloseKey(key);
    }
}

void testRegistrationCycle(Context & ctx)
{
    ScratchGuard scratch;
    std::string error;

    UNIT_CHECK(ctx, app::queryProjectAssociationAt(kScratchRoot, kThisExe) == AssociationState::NotRegistered);
    UNIT_CHECK(ctx, app::registerProjectAssociationAt(kScratchRoot, kThisExe, error));
    UNIT_CHECK(ctx, app::queryProjectAssociationAt(kScratchRoot, kThisExe) == AssociationState::ThisExe);
    UNIT_CHECK(ctx, app::queryProjectAssociationAt(kScratchRoot, kOtherExe) == AssociationState::OtherExe);

    // The keys written, with the values the shell reads.
    UNIT_CHECK(ctx, valueText(kScratchRoot + L"\\.sabercat", L"") == L"Sabercat.Project");
    UNIT_CHECK(ctx, valuePresent(kScratchRoot + L"\\.sabercat\\OpenWithProgids", L"Sabercat.Project"));
    UNIT_CHECK(ctx, valueText(kScratchRoot + L"\\Sabercat.Project", L"") == L"Sabercat project");
    UNIT_CHECK(ctx, valueText(kScratchRoot + L"\\Sabercat.Project", L"FriendlyTypeName") == L"Sabercat project");
    UNIT_CHECK(ctx, valueText(kScratchRoot + L"\\Sabercat.Project\\DefaultIcon", L"")
                        == app::buildDefaultIconValue(kThisExe, IDI_SABERCAT));
    UNIT_CHECK(ctx, valueText(kScratchRoot + L"\\Sabercat.Project\\shell\\open\\command", L"")
                        == app::buildOpenCommand(kThisExe));

    // Another copy's path cannot remove this copy's registration.
    UNIT_CHECK(ctx, !app::unregisterProjectAssociationAt(kScratchRoot, kOtherExe, error));
    UNIT_CHECK(ctx, !error.empty());
    UNIT_CHECK(ctx, app::queryProjectAssociationAt(kScratchRoot, kThisExe) == AssociationState::ThisExe);
    UNIT_CHECK(ctx, keyPresent(kScratchRoot + L"\\Sabercat.Project"));

    // Registering from the other path takes the association over.
    UNIT_CHECK(ctx, app::registerProjectAssociationAt(kScratchRoot, kOtherExe, error));
    UNIT_CHECK(ctx, app::queryProjectAssociationAt(kScratchRoot, kOtherExe) == AssociationState::ThisExe);
    UNIT_CHECK(ctx, app::queryProjectAssociationAt(kScratchRoot, kThisExe) == AssociationState::OtherExe);
    UNIT_CHECK(ctx, app::registerProjectAssociationAt(kScratchRoot, kThisExe, error));

    // Unregistering with this copy's path removes every key the module wrote.
    UNIT_CHECK(ctx, app::unregisterProjectAssociationAt(kScratchRoot, kThisExe, error));
    UNIT_CHECK(ctx, app::queryProjectAssociationAt(kScratchRoot, kThisExe) == AssociationState::NotRegistered);
    UNIT_CHECK(ctx, !keyPresent(kScratchRoot + L"\\.sabercat"));
    UNIT_CHECK(ctx, !keyPresent(kScratchRoot + L"\\Sabercat.Project"));
}

void testOtherProgramKeepsExtension(Context & ctx)
{
    ScratchGuard scratch;
    std::string error;

    // Another program owns .sabercat. This copy only reports it, and unregister
    // leaves that program's association in place.
    setDefaultText(kScratchRoot + L"\\.sabercat", L"Other.Program");
    UNIT_CHECK(ctx, app::queryProjectAssociationAt(kScratchRoot, kThisExe) == AssociationState::OtherExe);
    UNIT_CHECK(ctx, app::unregisterProjectAssociationAt(kScratchRoot, kThisExe, error));
    UNIT_CHECK(ctx, valueText(kScratchRoot + L"\\.sabercat", L"") == L"Other.Program");
    UNIT_CHECK(ctx, app::queryProjectAssociationAt(kScratchRoot, kThisExe) == AssociationState::OtherExe);
}

#endif

} // namespace

void testFileAssociation(Context & ctx)
{
    testOpenCommand(ctx);
    testIconValue(ctx);
    testCommandParsing(ctx);
#ifdef _WIN32
    testRegistrationCycle(ctx);
    testOtherProgramKeepsExtension(ctx);
#endif
}

} // namespace sabercat::unit
