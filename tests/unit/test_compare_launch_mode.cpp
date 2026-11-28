// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Launch-mode decisions (apps/compare/launch_mode.h): which front end a set of
// arguments selects, the files it passes on, and the conflicts that are usage
// errors. Pure CPU, no window and no file system.
#include "unit_test.h"

#include "launch_mode.h"

#include <initializer_list>
#include <string>
#include <vector>

namespace sabercat::unit
{

namespace
{

using sabercat::compare::LaunchDecision;
using sabercat::compare::LaunchMode;
using sabercat::compare::decideMode;

// Runs decideMode on the given arguments, after a program name.
LaunchDecision decide(std::initializer_list<const char *> args)
{
    std::vector<std::string> list = {"sabercat_compare"};
    for (const char * arg : args)
    {
        list.emplace_back(arg);
    }
    return decideMode(list);
}

bool contains(const std::string & text, const std::string & part)
{
    return text.find(part) != std::string::npos;
}

} // namespace

void testCompareLaunchMode(Context & ctx)
{
    // No arguments and a lone file open the window.
    UNIT_CHECK(ctx, decide({}).mode == LaunchMode::Gui);
    UNIT_CHECK(ctx, decide({"one.png"}).mode == LaunchMode::Gui);
    UNIT_CHECK(ctx, decide({"a.png", "b.png", "c.png"}).mode == LaunchMode::Gui);

    // Two plain paths compare, with or without CLI flags.
    UNIT_CHECK(ctx, decide({"ref.png", "test.png"}).mode == LaunchMode::Cli);
    UNIT_CHECK(ctx, decide({"ref.png", "test.png", "--ppd", "30"}).mode == LaunchMode::Cli);
    UNIT_CHECK(ctx, decide({"a.dat", "b.dat"}).mode == LaunchMode::Cli);
    UNIT_CHECK(ctx, decide({"--help"}).mode == LaunchMode::Cli);
    UNIT_CHECK(ctx, decide({"--no-flip", "a.png", "b.png"}).mode == LaunchMode::Cli);
    UNIT_CHECK(ctx, decide({"--hdr", "a.exr", "b.exr"}).mode == LaunchMode::Cli);

    // Uppercase extensions and the EXR pair are classified case-insensitively.
    UNIT_CHECK(ctx, decide({"A.PNG", "B.Png"}).mode == LaunchMode::Cli);
    UNIT_CHECK(ctx, decide({"left.EXR", "right.exr"}).mode == LaunchMode::Cli);
    UNIT_CHECK(ctx, decide({"SCENE.PLY"}).mode == LaunchMode::Gui);
    UNIT_CHECK(ctx, decide({"Scene.GLB"}).mode == LaunchMode::Gui);
    UNIT_CHECK(ctx, decide({"CLIP.MP4"}).mode == LaunchMode::Gui);

    // Scene, video and project files open the window.
    UNIT_CHECK(ctx, decide({"scene.ply"}).mode == LaunchMode::Gui);
    UNIT_CHECK(ctx, decide({"scene.splat"}).mode == LaunchMode::Gui);
    UNIT_CHECK(ctx, decide({"scene.spz"}).mode == LaunchMode::Gui);
    UNIT_CHECK(ctx, decide({"scene.sog"}).mode == LaunchMode::Gui);
    UNIT_CHECK(ctx, decide({"scene.gltf"}).mode == LaunchMode::Gui);
    UNIT_CHECK(ctx, decide({"scene.ksplat"}).mode == LaunchMode::Gui);
    UNIT_CHECK(ctx, decide({"a.png", "scene.ply"}).mode == LaunchMode::Gui);
    UNIT_CHECK(ctx, decide({"clip.mp4"}).mode == LaunchMode::Gui);
    UNIT_CHECK(ctx, decide({"clip.webm"}).mode == LaunchMode::Gui);
    UNIT_CHECK(ctx, decide({"work.sabercat-compare"}).mode == LaunchMode::Gui);

    // A cloud-v1 manifest is recognized by its name, in any directory.
    UNIT_CHECK(ctx, decide({"C:/data/manifest.json"}).mode == LaunchMode::Gui); // repo-check: allow
    UNIT_CHECK(ctx, decide({"a.json", "b.json"}).mode == LaunchMode::Cli);

    // --gui opens the window with any files, and sets guiFlag.
    const LaunchDecision guiPair = decide({"--gui", "a.png", "b.png"});
    UNIT_CHECK(ctx, guiPair.mode == LaunchMode::Gui);
    UNIT_CHECK(ctx, guiPair.guiFlag);
    UNIT_CHECK(ctx, guiPair.files.size() == 2u);
    UNIT_CHECK(ctx, decide({"--gui"}).mode == LaunchMode::Gui);

    // A CLI flag with --gui, or with a scene file, is a usage error that names both.
    const LaunchDecision guiPpd = decide({"--gui", "--ppd", "30"});
    UNIT_CHECK(ctx, guiPpd.mode == LaunchMode::UsageError);
    UNIT_CHECK(ctx, contains(guiPpd.message, "--ppd"));
    UNIT_CHECK(ctx, contains(guiPpd.message, "--gui"));

    const LaunchDecision reportScene = decide({"--report", "out.json", "x.ply"});
    UNIT_CHECK(ctx, reportScene.mode == LaunchMode::UsageError);
    UNIT_CHECK(ctx, contains(reportScene.message, "--report"));
    UNIT_CHECK(ctx, contains(reportScene.message, "x.ply"));

    // Option values are not positional files.
    const LaunchDecision flipMap = decide({"--flip-map", "out.png", "a.png", "b.png"});
    UNIT_CHECK(ctx, flipMap.mode == LaunchMode::Cli);
    UNIT_CHECK(ctx, flipMap.files == (std::vector<std::string>{"a.png", "b.png"}));

    // A value-taking flag with no value is left for the CLI to report.
    UNIT_CHECK(ctx, decide({"--ppd"}).mode == LaunchMode::Cli);
    UNIT_CHECK(ctx, decide({"a.png", "b.png", "--ppd"}).mode == LaunchMode::Cli);

    // Unknown options are CLI flags; the CLI reports them.
    UNIT_CHECK(ctx, decide({"--frobnicate", "a.png", "b.png"}).mode == LaunchMode::Cli);

    // --selftest <dir> opens the GUI, which runs the self test; its value is the
    // directory, never a file, and it is not a CLI flag.
    const LaunchDecision selftest = decide({"--selftest", "out_dir"});
    UNIT_CHECK(ctx, selftest.mode == LaunchMode::Gui);
    UNIT_CHECK(ctx, selftest.selftest);
    UNIT_CHECK(ctx, selftest.selftestDir == "out_dir");
    UNIT_CHECK(ctx, selftest.files.empty());
    UNIT_CHECK(ctx, !selftest.guiFlag && !selftest.validation && !selftest.noValidation);

    // A directory named like a scene file is still the directory, not a scene file.
    const LaunchDecision selftestScene = decide({"--selftest", "scene.ply"});
    UNIT_CHECK(ctx, selftestScene.mode == LaunchMode::Gui);
    UNIT_CHECK(ctx, selftestScene.selftestDir == "scene.ply");
    UNIT_CHECK(ctx, selftestScene.files.empty());

    // Files given with --selftest still reach the GUI; the directory does not.
    const LaunchDecision selftestPair = decide({"--selftest", "dir", "a.png", "b.png"});
    UNIT_CHECK(ctx, selftestPair.mode == LaunchMode::Gui);
    UNIT_CHECK(ctx, selftestPair.selftestDir == "dir");
    UNIT_CHECK(ctx, selftestPair.files == (std::vector<std::string>{"a.png", "b.png"}));

    // --selftest with no directory is a usage error that names the flag.
    const LaunchDecision selftestNoValue = decide({"--selftest"});
    UNIT_CHECK(ctx, selftestNoValue.mode == LaunchMode::UsageError);
    UNIT_CHECK(ctx, contains(selftestNoValue.message, "--selftest"));
    UNIT_CHECK(ctx, contains(selftestNoValue.message, "directory"));
    UNIT_CHECK(ctx, decide({"a.png", "b.png", "--selftest"}).mode == LaunchMode::UsageError);
    UNIT_CHECK(ctx, decide({"--selftest", ""}).mode == LaunchMode::UsageError);

    // --selftest with a CLI flag is a usage error that names both.
    const LaunchDecision selftestPpd = decide({"--selftest", "dir", "--ppd", "30"});
    UNIT_CHECK(ctx, selftestPpd.mode == LaunchMode::UsageError);
    UNIT_CHECK(ctx, contains(selftestPpd.message, "--ppd"));
    UNIT_CHECK(ctx, contains(selftestPpd.message, "--selftest"));

    // --validation alone opens the window, and sets only its own flag.
    const LaunchDecision validation = decide({"--validation"});
    UNIT_CHECK(ctx, validation.mode == LaunchMode::Gui);
    UNIT_CHECK(ctx, validation.validation);
    UNIT_CHECK(ctx, !validation.noValidation && !validation.selftest);
    UNIT_CHECK(ctx, validation.files.empty());

    // A GUI-only flag with two plain files opens the GUI and keeps the files.
    const LaunchDecision validationPair = decide({"a.png", "b.png", "--validation"});
    UNIT_CHECK(ctx, validationPair.mode == LaunchMode::Gui);
    UNIT_CHECK(ctx, validationPair.validation);
    UNIT_CHECK(ctx, validationPair.files.size() == 2u);

    // --no-validation alone opens the window, and sets only its own flag.
    const LaunchDecision noValidation = decide({"--no-validation"});
    UNIT_CHECK(ctx, noValidation.mode == LaunchMode::Gui);
    UNIT_CHECK(ctx, noValidation.noValidation);
    UNIT_CHECK(ctx, !noValidation.validation);

    // --validation with --no-validation conflicts, in either order.
    const LaunchDecision bothValidation = decide({"--validation", "--no-validation"});
    UNIT_CHECK(ctx, bothValidation.mode == LaunchMode::UsageError);
    UNIT_CHECK(ctx, bothValidation.message == "--validation and --no-validation conflict");
    UNIT_CHECK(ctx, decide({"--no-validation", "--validation"}).mode == LaunchMode::UsageError);
    UNIT_CHECK(ctx,
               decide({"--no-validation", "--validation"}).message == "--validation and --no-validation conflict");

    // A CLI flag with --validation is a usage error that names both.
    const LaunchDecision validationPpd = decide({"--validation", "--ppd", "30"});
    UNIT_CHECK(ctx, validationPpd.mode == LaunchMode::UsageError);
    UNIT_CHECK(ctx, contains(validationPpd.message, "--ppd"));
    UNIT_CHECK(ctx, contains(validationPpd.message, "--validation"));

    // --gui with a GUI-only argument is still a GUI launch, and keeps its flags.
    const LaunchDecision guiValidation = decide({"--gui", "--no-validation"});
    UNIT_CHECK(ctx, guiValidation.mode == LaunchMode::Gui);
    UNIT_CHECK(ctx, guiValidation.guiFlag && guiValidation.noValidation);

    // A value-taking CLI flag next to --selftest: the value is not the selftest dir.
    const LaunchDecision flipWithSelftest = decide({"--flip-map", "out.png", "--selftest", "dir"});
    UNIT_CHECK(ctx, flipWithSelftest.mode == LaunchMode::UsageError);
    UNIT_CHECK(ctx, contains(flipWithSelftest.message, "--flip-map"));
    UNIT_CHECK(ctx, flipWithSelftest.selftestDir == "dir");

    // --log <path> opens the window and keeps the path as given; it is not a file.
    const LaunchDecision logPathOnly = decide({"--log", "out.log"});
    UNIT_CHECK(ctx, logPathOnly.mode == LaunchMode::Gui);
    UNIT_CHECK(ctx, logPathOnly.logPath == "out.log");
    UNIT_CHECK(ctx, logPathOnly.files.empty());
    UNIT_CHECK(ctx, !logPathOnly.noLog && !logPathOnly.selftest && !logPathOnly.validation);

    // The value after --log is the path even when it names a scene file or a flag.
    const LaunchDecision logScene = decide({"--log", "scene.ply"});
    UNIT_CHECK(ctx, logScene.mode == LaunchMode::Gui);
    UNIT_CHECK(ctx, logScene.logPath == "scene.ply");
    UNIT_CHECK(ctx, logScene.files.empty());
    UNIT_CHECK(ctx, decide({"--log", "--ppd"}).logPath == "--ppd");

    // Files given with --log still reach the GUI; the path does not.
    const LaunchDecision logPair = decide({"a.png", "b.png", "--log", "x.log"});
    UNIT_CHECK(ctx, logPair.mode == LaunchMode::Gui);
    UNIT_CHECK(ctx, logPair.logPath == "x.log");
    UNIT_CHECK(ctx, logPair.files == (std::vector<std::string>{"a.png", "b.png"}));

    // --log with no path, or an empty one, is a usage error that names the flag.
    const LaunchDecision logNoValue = decide({"--log"});
    UNIT_CHECK(ctx, logNoValue.mode == LaunchMode::UsageError);
    UNIT_CHECK(ctx, logNoValue.message == "--log needs a path");
    UNIT_CHECK(ctx, decide({"a.png", "b.png", "--log"}).mode == LaunchMode::UsageError);
    UNIT_CHECK(ctx, decide({"--log", ""}).mode == LaunchMode::UsageError);

    // --no-log alone opens the window and sets only its own flag.
    const LaunchDecision noLog = decide({"--no-log"});
    UNIT_CHECK(ctx, noLog.mode == LaunchMode::Gui);
    UNIT_CHECK(ctx, noLog.noLog);
    UNIT_CHECK(ctx, noLog.logPath.empty() && !noLog.validation);

    // --log with --no-log conflicts, in either order.
    const LaunchDecision bothLog = decide({"--log", "x.log", "--no-log"});
    UNIT_CHECK(ctx, bothLog.mode == LaunchMode::UsageError);
    UNIT_CHECK(ctx, bothLog.message == "--log and --no-log conflict");
    UNIT_CHECK(ctx, decide({"--no-log", "--log", "x.log"}).message == "--log and --no-log conflict");

    // A CLI flag with --log or --no-log is a usage error that names both.
    const LaunchDecision logPpd = decide({"--log", "x.log", "--ppd", "30"});
    UNIT_CHECK(ctx, logPpd.mode == LaunchMode::UsageError);
    UNIT_CHECK(ctx, contains(logPpd.message, "--ppd"));
    UNIT_CHECK(ctx, contains(logPpd.message, "--log"));
    const LaunchDecision noLogPpd = decide({"--no-log", "--ppd", "30"});
    UNIT_CHECK(ctx, noLogPpd.mode == LaunchMode::UsageError);
    UNIT_CHECK(ctx, contains(noLogPpd.message, "--ppd"));
    UNIT_CHECK(ctx, contains(noLogPpd.message, "--no-log"));
}

} // namespace sabercat::unit
