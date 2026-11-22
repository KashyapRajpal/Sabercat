// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Console detach for the GUI launch (apps/compare/console.h).

#include "console.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

namespace sabercat::compare
{

void detachConsoleIfOwned()
{
    // Only the calling process is attached: the console was created for it.
    DWORD processId = 0;
    if (GetConsoleProcessList(&processId, 1) == 1)
    {
        ShowWindow(GetConsoleWindow(), SW_HIDE);
        FreeConsole();
    }
}

} // namespace sabercat::compare
