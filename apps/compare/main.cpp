// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// sabercat_compare -- one executable with two front ends: PSNR, FLIP and
// difference maps for two images (CLI), and a window for scenes, videos, projects
// and image pairs (GUI). The launch arguments choose (see launch_mode.h); this
// file only dispatches. CLI mode never touches the console; a GUI launch from
// Explorer hides the console Windows created for it.
//
//   sabercat_compare <reference.png> <test.png> [--flip-map out.png]
//                    [--diff-map out.png] [--ppd N] [--no-flip] [--help]

#include "app_args.h"
#include "cli.h"
#include "console.h"
#include "gui.h"
#include "launch_mode.h"

#include <iostream>
#include <string>
#include <vector>

int main(int argc, char * argv[])
{
    using namespace sabercat::compare;

    constexpr int kExitUsage = 2;

    const std::vector<std::string> args = utf8Arguments(argc, argv);
    const LaunchDecision decision = decideMode(args);
    switch (decision.mode)
    {
    case LaunchMode::Cli:
        return runCli(args);
    case LaunchMode::UsageError:
        std::cerr << "error: " << decision.message << "\n" << cliUsage();
        return kExitUsage;
    case LaunchMode::Gui:
        detachConsoleIfOwned();
        return runGui(decision);
    }
    return kExitUsage;
}
