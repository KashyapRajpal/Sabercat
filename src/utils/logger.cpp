// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/logger.h"

#include <chrono>
#include <ctime>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <utility>

namespace sabercat
{

Logger::Level Logger::s_level = Logger::Level::Info;
std::string Logger::s_logFilePath;
std::ofstream Logger::s_logFile;
std::mutex Logger::s_mutex;

void Logger::setLevel(Level level)
{
    s_level = level;
}

bool Logger::setLogFile(const std::string & utf8Path)
{
    std::lock_guard<std::mutex> lock(s_mutex);
    if (s_logFile.is_open())
    {
        s_logFile.close();
    }
    s_logFilePath.clear();
    if (utf8Path.empty())
    {
        return true;
    }

    std::ofstream file(std::filesystem::u8path(utf8Path), std::ios::out | std::ios::app);
    if (!file)
    {
        std::cerr << "[WARN] could not open log file " << utf8Path << std::endl;
        return false;
    }
    s_logFile = std::move(file);
    s_logFilePath = utf8Path;
    return true;
}

std::string Logger::logFilePath()
{
    std::lock_guard<std::mutex> lock(s_mutex);
    return s_logFilePath;
}

void Logger::debug(const std::string & message)
{
    log(Level::Debug, message);
}

void Logger::info(const std::string & message)
{
    log(Level::Info, message);
}

void Logger::warning(const std::string & message)
{
    log(Level::Warning, message);
}

void Logger::error(const std::string & message)
{
    log(Level::Error, message);
}

void Logger::critical(const std::string & message)
{
    log(Level::Critical, message);
}

void Logger::log(Level level, const std::string & message)
{
    if (level < s_level)
    {
        return;
    }

    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&time);

    std::string level_str;
    std::ostream * output = &std::cout;

    switch (level)
    {
    case Level::Debug:
        level_str = "DEBUG";
        break;
    case Level::Info:
        level_str = "INFO";
        break;
    case Level::Warning:
        level_str = "WARN";
        output = &std::cerr;
        break;
    case Level::Error:
        level_str = "ERROR";
        output = &std::cerr;
        break;
    case Level::Critical:
        level_str = "CRITICAL";
        output = &std::cerr;
        break;
    }

    std::ostringstream line;
    line << "[" << std::put_time(&tm, "%H:%M:%S") << "] [" << level_str << "] " << message;

    std::lock_guard<std::mutex> lock(s_mutex);
    *output << line.str() << std::endl;
    if (s_logFile.is_open())
    {
        s_logFile << line.str() << '\n';
        s_logFile.flush();
    }
}

} // namespace sabercat
