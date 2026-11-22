// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Console handling for the GUI launch (apps/compare/console.cpp).
#pragma once

namespace sabercat::compare
{

// When Explorer started this process, Windows gave it a console of its own: hide
// that console and release it. From a terminal the console is shared, so nothing
// is done.
void detachConsoleIfOwned();

} // namespace sabercat::compare
