// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The command-line front end of sabercat_compare (apps/compare/cli.cpp).
#pragma once

#include <string>
#include <vector>

namespace sabercat::compare
{

// Runs the command-line comparison. args[0] is the program name. Returns the
// process exit code: 0 on success, 1 on a failure, 2 on a usage error.
int runCli(const std::vector<std::string> & args);

// The usage text printed by --help and after a usage error.
const char * cliUsage();

} // namespace sabercat::compare
