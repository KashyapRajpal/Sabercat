// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Command-line arguments as UTF-8, shared by the CLI and GUI launch paths.
#pragma once

#include <string>
#include <vector>

namespace sabercat::compare
{

// argv as UTF-8 strings. On Windows the wide command line is used, so a
// non-ASCII path is not mangled by the ANSI codepage.
std::vector<std::string> utf8Arguments(int argc, char * argv[]);

} // namespace sabercat::compare
