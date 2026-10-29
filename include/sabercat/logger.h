// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <fstream>
#include <mutex>
#include <string>

namespace sabercat
{

class Logger
{
  public:
    enum class Level
    {
        Debug,
        Info,
        Warning,
        Error,
        Critical
    };

    static void setLevel(Level level);

    // Mirrors every line that passes the level filter to a file, in addition to
    // the console. An empty path turns the file off. The path is UTF-8; the file
    // is opened for appending and flushed after each line. Returns false (and
    // leaves the file off) when it cannot be opened.
    static bool setLogFile(const std::string & utf8Path);
    // The UTF-8 path of the mirror file, or "" when it is off.
    static std::string logFilePath();

    static void debug(const std::string & message);
    static void info(const std::string & message);
    static void warning(const std::string & message);
    static void error(const std::string & message);
    static void critical(const std::string & message);

  private:
    static Level s_level;
    static std::string s_logFilePath;
    static std::ofstream s_logFile;
    static std::mutex s_mutex;

    static void log(Level level, const std::string & message);
};

} // namespace sabercat
