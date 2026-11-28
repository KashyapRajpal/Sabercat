// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Chooses the front end (CLI, GUI or usage error) from the launch arguments. Pure
// CPU: classification is by file extension and name only, never the file system.
#pragma once

#include <string>
#include <vector>

namespace sabercat::compare
{

enum class LaunchMode
{
    Cli,
    Gui,
    UsageError
};

struct LaunchDecision
{
    std::vector<std::string> files; // positional paths, for the GUI to open
    std::string message;            // the conflict, for UsageError; empty otherwise
    std::string selftestDir;        // the value of --selftest; empty when not given
    std::string logPath;            // the value of --log; empty when not given
    LaunchMode mode = LaunchMode::Cli;
    bool guiFlag = false;           // --gui was given
    bool selftest = false;          // --selftest <dir> was given
    bool validation = false;        // --validation was given
    bool noValidation = false;      // --no-validation was given
    bool noLog = false;             // --no-log was given
};

// args[0] is the program name. Option values are never treated as files.
// --selftest, --validation, --no-validation, --log <path> and --no-log are GUI-only
// arguments, like --gui: any of them selects Gui, and a CLI flag with any of them is a
// usage error. --log takes the next argument as its path, whatever it looks like.
LaunchDecision decideMode(const std::vector<std::string> & args);

} // namespace sabercat::compare
