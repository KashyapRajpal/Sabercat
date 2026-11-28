// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The window front end (apps/compare/gui.h). Until the window exists, the only GUI path
// is the headless self-test (`--selftest <dir>`).

#include "gui.h"

#include "app_util.h"
#include "gui/app.h"
#include "gui/selftest.h"
#include "log_files.h"

#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <system_error>

namespace sabercat::compare
{

namespace
{

// Creates the output directory of the self-test. The path is UTF-8.
void createOutputDirectory(const std::string & utf8Path)
{
    const std::u8string text(reinterpret_cast<const char8_t *>(utf8Path.data()), utf8Path.size());
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(text), ec);
    if (ec)
    {
        throw std::runtime_error("cannot create " + utf8Path + ": " + ec.message());
    }
}

} // namespace

int runGui(const LaunchDecision & decision)
{
    // The session log. The self-test and the smoke runs write one only when --log asks for it;
    // a normal window launch logs by default.
    const char * smokeEnv = std::getenv("SABERCAT_COMPARE_SMOKE_DIR");
    const bool smokeRun = smokeEnv != nullptr && *smokeEnv != '\0';
    if (decision.selftest || smokeRun)
    {
        if (!decision.logPath.empty())
        {
            sabercat::app::startSessionLog("compare", decision.logPath, false);
        }
    }
    else
    {
        sabercat::app::startSessionLog("compare", decision.logPath, decision.noLog);
    }

    if (decision.selftest)
    {
        try
        {
            createOutputDirectory(decision.selftestDir);
            const bool validation = sabercat::app::resolveValidation(decision.validation, decision.noValidation);
            return gui::runSelfTest(decision.selftestDir, validation);
        }
        catch (const std::exception & e)
        {
            std::cerr << "sabercat_compare: " << e.what() << '\n';
            return 1;
        }
    }
    try
    {
        return gui::runApp(decision);
    }
    catch (const std::exception & e)
    {
        std::cerr << "sabercat_compare: " << e.what() << '\n';
        return 1;
    }
}

} // namespace sabercat::compare
