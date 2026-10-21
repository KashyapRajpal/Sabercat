// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Scratch space for tests that write files (corrupted copies for rejection
// checks, format round trips). The constructor makes a fresh
// <system temp>/sabercat_<name> folder and switches the working directory
// into it, so relative scratch paths never land wherever the test happens to
// be run from (e.g. the repo root). The destructor switches back and deletes
// the folder. Tests must reference their real inputs by absolute path (the
// CMake-provided fixture/shader directories), as they already do.

#include <filesystem>
#include <string>
#include <system_error>

namespace examples
{

class ScratchDir
{
  public:
    explicit ScratchDir(const std::string & name)
        : m_original(std::filesystem::current_path()),
          m_dir(std::filesystem::temp_directory_path() / ("sabercat_" + name))
    {
        std::error_code ignored;
        std::filesystem::remove_all(m_dir, ignored);
        std::filesystem::create_directories(m_dir);
        std::filesystem::current_path(m_dir);
    }

    ~ScratchDir()
    {
        std::error_code ignored;
        std::filesystem::current_path(m_original, ignored);
        std::filesystem::remove_all(m_dir, ignored);
    }

    ScratchDir(const ScratchDir &) = delete;
    ScratchDir & operator=(const ScratchDir &) = delete;

    const std::filesystem::path & path() const
    {
        return m_dir;
    }

  private:
    std::filesystem::path m_original;
    std::filesystem::path m_dir;
};

} // namespace examples
