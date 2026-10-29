// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The apps' session log files: one file per run, named
// "<app>-<yyyymmdd-hhmmss>.log" in %LOCALAPPDATA%\Sabercat\logs, the newest
// kKeptLogFiles per app kept. The pruning decision is pure (the CPU unit tests
// call it); startSessionLog() does the file work and never fails the app.
#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace sabercat::app
{

// How many session logs per app are kept; older ones are deleted at startup.
inline constexpr std::size_t kKeptLogFiles = 10;

// "<app>-<stamp>.log", where stamp is "yyyymmdd-hhmmss".
std::string sessionLogFileName(const std::string & app, const std::string & stamp);

// Of `names` (bare file names in one directory), the ones that belong to `app`
// (named "<app>-*.log") and fall outside the newest `keep`. The stamp sorts by
// time, so the oldest names are returned. Duplicates count once; other names
// are ignored.
std::vector<std::string> logNamesToPrune(const std::vector<std::string> & names,
                                         const std::string & app, std::size_t keep);

// %LOCALAPPDATA%\Sabercat\logs as UTF-8. Empty when LOCALAPPDATA is not set.
std::string defaultLogDirectory();

// Starts the session's log mirror (Logger::setLogFile) and returns its UTF-8
// path, or "" when logging is off. `explicitPath` (from --log) is used as given
// and nothing is pruned; `disabled` (--no-log) turns logging off. Otherwise the
// default directory is created, older logs of `app` are pruned and a new file
// is opened. Any failure prints a warning on stdout and returns "".
std::string startSessionLog(const std::string & app, const std::string & explicitPath,
                            bool disabled);

} // namespace sabercat::app
