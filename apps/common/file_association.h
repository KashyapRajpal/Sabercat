// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Per-user file association for .sabercat projects. Registering makes this
// executable the program that opens a project, under HKEY_CURRENT_USER only, so
// no administrator rights are needed. The pure helpers hold no registry state
// and are unit-tested directly. The registry calls are Windows-only; elsewhere
// they report that the association is not supported.
#pragma once

#include <string>

namespace sabercat::app
{

// Whether .sabercat files open with this executable, with another program or
// copy, or with nothing.
enum class AssociationState
{
    NotRegistered,
    ThisExe,
    OtherExe,
};

// The command the shell runs to open a project: "<exe>" "%1". The path is
// quoted, so spaces and non-ASCII characters survive, and Windows passes the
// file as the first argument.
std::wstring buildOpenCommand(const std::wstring & exePath);

// The DefaultIcon value: "<exe>",-<resourceId>. A negative index names a
// resource ID in the executable, so it stays correct whatever order the icons
// are in.
std::wstring buildDefaultIconValue(const std::wstring & exePath, int resourceId);

// The executable an open command starts: the quoted first word, or the first
// word when it is not quoted. Empty for an empty command.
std::wstring exePathFromOpenCommand(const std::wstring & command);

// Full path of the running executable. Empty off Windows.
std::wstring currentExecutablePath();

// The association under HKEY_CURRENT_USER\Software\Classes. Each call takes the
// executable this copy runs from; registration, the query and unregistration
// only ever touch the keys this module owns.
AssociationState queryProjectAssociation(const std::wstring & exePath);
bool registerProjectAssociation(const std::wstring & exePath, std::string & error);
bool unregisterProjectAssociation(const std::wstring & exePath, std::string & error);

// The same three calls under another per-user root: a subkey of
// HKEY_CURRENT_USER such as L"Software\\SabercatTest\\Classes". The tests use a
// scratch root so the real associations are never touched.
AssociationState queryProjectAssociationAt(const std::wstring & classesSubkey,
                                           const std::wstring & exePath);
bool registerProjectAssociationAt(const std::wstring & classesSubkey, const std::wstring & exePath,
                                  std::string & error);
bool unregisterProjectAssociationAt(const std::wstring & classesSubkey, const std::wstring & exePath,
                                    std::string & error);

} // namespace sabercat::app
