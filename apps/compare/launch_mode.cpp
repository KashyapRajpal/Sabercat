// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Launch-mode rules (apps/compare/launch_mode.h).

#include "launch_mode.h"

#include <cstddef>
#include <initializer_list>

namespace sabercat::compare
{

namespace
{

enum class FileKind
{
    Scene,
    Video,
    Project,
    Other
};

// ASCII lowercase copy; extensions and the manifest name compare case-insensitively.
std::string lowerAscii(const std::string & text)
{
    std::string result = text;
    for (char & c : result)
    {
        if (c >= 'A' && c <= 'Z')
        {
            c = static_cast<char>(c - 'A' + 'a');
        }
    }
    return result;
}

bool isOneOf(const std::string & text, std::initializer_list<const char *> choices)
{
    for (const char * choice : choices)
    {
        if (text == choice)
        {
            return true;
        }
    }
    return false;
}

// The file name after the last '/' or '\'.
std::string fileNameOf(const std::string & path)
{
    const size_t separator = path.find_last_of("/\\");
    return separator == std::string::npos ? path : path.substr(separator + 1);
}

// The lowercase text after the last dot of the file name; empty if there is none.
std::string extensionOf(const std::string & path)
{
    const std::string name = fileNameOf(path);
    const size_t dot = name.find_last_of('.');
    return dot == std::string::npos ? std::string() : lowerAscii(name.substr(dot + 1));
}

FileKind classify(const std::string & path)
{
    if (lowerAscii(fileNameOf(path)) == "manifest.json")
    {
        return FileKind::Scene;
    }
    const std::string extension = extensionOf(path);
    if (isOneOf(extension, {"ply", "splat", "spz", "sog", "gltf", "glb", "ksplat"}))
    {
        return FileKind::Scene;
    }
    if (isOneOf(extension, {"mp4", "webm"}))
    {
        return FileKind::Video;
    }
    if (extension == "sabercat-compare")
    {
        return FileKind::Project;
    }
    return FileKind::Other;
}

// The flags that take a value in the following argument.
bool takesValue(const std::string & arg)
{
    return isOneOf(arg, {"--flip-map", "--diff-map", "--ppd", "--tone-mapper", "--start-exposure",
                         "--stop-exposure", "--num-exposures", "--exposure-map", "--report"});
}

const char * sceneNoun(FileKind kind)
{
    switch (kind)
    {
    case FileKind::Scene:
        return "scene file";
    case FileKind::Video:
        return "video file";
    case FileKind::Project:
        return "project file";
    case FileKind::Other:
        break;
    }
    return "file";
}

} // namespace

LaunchDecision decideMode(const std::vector<std::string> & args)
{
    bool guiFlag = false;
    bool selftest = false;
    bool selftestMissing = false;
    bool validation = false;
    bool noValidation = false;
    std::string selftestDir;
    std::string firstCliFlag;
    std::string firstGuiOnlyFlag;
    std::vector<std::string> files;
    for (size_t i = 1; i < args.size(); ++i)
    {
        const std::string & arg = args[i];
        if (arg == "--gui")
        {
            guiFlag = true;
            continue;
        }
        if (arg == "--validation" || arg == "--no-validation")
        {
            if (arg == "--validation")
            {
                validation = true;
            }
            else
            {
                noValidation = true;
            }
            if (firstGuiOnlyFlag.empty())
            {
                firstGuiOnlyFlag = arg;
            }
            continue;
        }
        if (arg == "--selftest")
        {
            // The directory is the next argument and is never a file. An absent or
            // empty value is reported below.
            if (firstGuiOnlyFlag.empty())
            {
                firstGuiOnlyFlag = arg;
            }
            const bool hasValue = i + 1 < args.size() && !args[i + 1].empty();
            selftestMissing = !hasValue;
            if (hasValue)
            {
                selftest = true;
                selftestDir = args[++i];
            }
            continue;
        }
        if (arg.size() > 1 && arg.front() == '-')
        {
            // Unknown options count as CLI flags, so cli.cpp reports them.
            if (firstCliFlag.empty())
            {
                firstCliFlag = arg;
            }
            if (takesValue(arg))
            {
                // Skip the value. A missing value is left for cli.cpp to report.
                ++i;
            }
            continue;
        }
        files.push_back(arg);
    }

    // Every decision carries the GUI-only arguments it was given.
    LaunchDecision decision;
    decision.files = files;
    decision.selftestDir = selftestDir;
    decision.guiFlag = guiFlag;
    decision.selftest = selftest;
    decision.validation = validation;
    decision.noValidation = noValidation;

    if (selftestMissing)
    {
        decision.mode = LaunchMode::UsageError;
        decision.message = "--selftest needs a directory";
        return decision;
    }
    if (validation && noValidation)
    {
        decision.mode = LaunchMode::UsageError;
        decision.message = "--validation and --no-validation conflict";
        return decision;
    }

    if (guiFlag || selftest || validation || noValidation)
    {
        if (!firstCliFlag.empty())
        {
            decision.mode = LaunchMode::UsageError;
            decision.message = firstCliFlag + " cannot be used with " + (guiFlag ? "--gui" : firstGuiOnlyFlag);
            return decision;
        }
        decision.mode = LaunchMode::Gui;
        return decision;
    }

    for (const std::string & file : files)
    {
        const FileKind kind = classify(file);
        if (kind == FileKind::Other)
        {
            continue;
        }
        if (!firstCliFlag.empty())
        {
            decision.mode = LaunchMode::UsageError;
            decision.message = firstCliFlag + " cannot be used with " + sceneNoun(kind) + " '" + file + "'";
            return decision;
        }
        decision.mode = LaunchMode::Gui;
        return decision;
    }

    if (!firstCliFlag.empty())
    {
        decision.mode = LaunchMode::Cli;
        return decision;
    }
    // Two plain paths compare as before; any other count opens the GUI.
    decision.mode = files.size() == 2 ? LaunchMode::Cli : LaunchMode::Gui;
    return decision;
}

} // namespace sabercat::compare
