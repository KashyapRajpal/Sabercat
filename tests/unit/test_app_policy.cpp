// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The apps' GPU-free decisions (apps/common/app_policy.h) and the scene scan
// (apps/common/asset_scan.h), on a temp directory of files. No device is created.
#include "unit_test.h"

#include "app_policy.h"
#include "asset_scan.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace sabercat::unit
{

namespace
{

namespace fs = std::filesystem;
using app::ValidationChoice;

// Every combination of the three inputs, with the decision the apps must make.
void testValidationTruthTable(Context & ctx)
{
    struct Row
    {
        bool requested;
        bool noFlag;
        bool installed;
        ValidationChoice expected;
    };
    const Row rows[] = {
        // --no-validation wins over everything.
        {true, true, true, ValidationChoice::OffByFlag},
        {true, true, false, ValidationChoice::OffByFlag},
        {false, true, true, ValidationChoice::OffByFlag},
        {false, true, false, ValidationChoice::OffByFlag},
        // Not requested (a Release build without --validation), installed or not.
        {false, false, true, ValidationChoice::OffNotRequested},
        {false, false, false, ValidationChoice::OffNotRequested},
        // Requested but not installed: run without it.
        {true, false, false, ValidationChoice::OffNotInstalled},
        // Requested and installed: on.
        {true, false, true, ValidationChoice::On},
    };
    for (const Row & row : rows)
    {
        UNIT_CHECK(ctx, app::decideValidation(row.requested, row.noFlag, row.installed) == row.expected);
    }
}

// The SPIR-V directory: $SABERCAT_SHADER_DIR, then the install layout (only when
// it exists), then the build tree.
void testShaderDirOrder(Context & ctx)
{
    const std::string build = "build/share/shaders";
    const std::string installed = "package/share/sabercat/shaders";
    UNIT_CHECK(ctx, app::chooseShaderDir("", "", false, build) == build);
    UNIT_CHECK(ctx, app::chooseShaderDir("", installed, false, build) == build); // ignored when absent
    UNIT_CHECK(ctx, app::chooseShaderDir("", installed, true, build) == installed);
    UNIT_CHECK(ctx, app::chooseShaderDir("override/shaders", installed, true, build) == "override/shaders");
    UNIT_CHECK(ctx, app::chooseShaderDir("override/shaders", "", false, build) == "override/shaders");
}

void writeFile(const fs::path & path, const std::string & text)
{
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    out << text;
}

// Scene files are found recursively, by extension in any case, and listed sorted.
// Other files are skipped, and a .json is listed only if the loader would read it
// as SOG (a cloud-v1 manifest and unreadable JSON are not).
void testSceneScan(Context & ctx)
{
    const fs::path root = fs::temp_directory_path() / "sabercat-unit-scan";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root);

    writeFile(root / "b.PLY", "not parsed, only listed");
    writeFile(root / "a.splat", "x");
    writeFile(root / "e.gltf", "{}");
    writeFile(root / "sub" / "c.spz", "x");
    writeFile(root / "sub" / "deep" / "d.GLB", "x");
    writeFile(root / "notes.txt", "x");
    writeFile(root / "manifest.json", R"({"format":"sabercat-cloud-v1","version":1})");
    writeFile(root / "broken.json", "{");

    const std::vector<std::string> found = app::scanSceneDirectory(root.string());
    std::vector<std::string> expected = {
        (root / "a.splat").string(),
        (root / "b.PLY").string(),
        (root / "e.gltf").string(),
        (root / "sub" / "c.spz").string(),
        (root / "sub" / "deep" / "d.GLB").string(),
    };
    std::sort(expected.begin(), expected.end());
    UNIT_CHECK(ctx, found == expected);

    UNIT_CHECK(ctx, app::scanSceneDirectory((root / "missing").string()).empty());
    fs::create_directories(root / "empty");
    UNIT_CHECK(ctx, app::scanSceneDirectory((root / "empty").string()).empty());

    fs::remove_all(root, ec);
}

// The viewer's scene list: most recent first, no duplicates.
void testScenePathList(Context & ctx)
{
    std::vector<std::string> list;
    // Empty list: the path goes in at index 0.
    UNIT_CHECK(ctx, app::addScenePathToList(list, "a.ply") == 0);
    UNIT_CHECK(ctx, list == std::vector<std::string>{"a.ply"});
    // A new path goes on top.
    UNIT_CHECK(ctx, app::addScenePathToList(list, "b.ply") == 0);
    UNIT_CHECK(ctx, (list == std::vector<std::string>{"b.ply", "a.ply"}));
    // A repeat moves to the top instead of being added again.
    UNIT_CHECK(ctx, app::addScenePathToList(list, "a.ply") == 0);
    UNIT_CHECK(ctx, (list == std::vector<std::string>{"a.ply", "b.ply"}));
    // The same file under another spelling is the same entry: the list keeps
    // the first spelling, and the key matches.
    UNIT_CHECK(ctx, app::scenePathKey("scenes/./a.ply") == app::scenePathKey("scenes/a.ply"));
    UNIT_CHECK(ctx, app::scenePathKey("scenes/sub/../a.ply") == app::scenePathKey("scenes/a.ply"));
    UNIT_CHECK(ctx, app::scenePathKey("scenes/a.ply/") == app::scenePathKey("scenes/a.ply"));
    UNIT_CHECK(ctx, app::scenePathKey("scenes/a.ply") != app::scenePathKey("scenes/b.ply"));
    UNIT_CHECK(ctx, app::addScenePathToList(list, "./b.ply") == 0);
    UNIT_CHECK(ctx, (list == std::vector<std::string>{"./b.ply", "a.ply"}));
    UNIT_CHECK(ctx, list.size() == 2);
#ifdef _WIN32
    // Windows paths compare case-insensitively and with either separator.
    UNIT_CHECK(ctx, app::scenePathKey("C:/Scenes/Room.PLY") == app::scenePathKey("c:\\scenes\\room.ply"));
#endif
}

} // namespace

void testAppPolicy(Context & ctx)
{
    testValidationTruthTable(ctx);
    testShaderDirOrder(ctx);
    testSceneScan(ctx);
    testScenePathList(ctx);
}

} // namespace sabercat::unit
