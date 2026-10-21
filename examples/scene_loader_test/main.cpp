// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Table-driven test for the scene loader's format dispatch (detectFormat and
// loadScene). No assets are needed: every fixture is written at run time into a
// scratch folder.
//
//  (a) detectFormat returns the expected SceneFormat for every value, for
//      lower- and upper-case extensions, a cloud-v1 directory, and a missing
//      file where the extension alone decides (the reader reports the rest).
//  (b) A file whose magic does not match its extension is refused by name.
//  (c) An unsupported or missing extension, a streamed SOG manifest and the
//      input.ply (COLMAP) guard are refused by name.
//  (d) loadScene reports a missing file by name, not with a crash.
//
// Every SceneFormat value must be produced by at least one table row; the
// check at the end fails if a new format is added without a row.

#include <cstdio>
#include <exception>
#include <functional>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "../common/scratch_dir.h"
#include "sabercat/device.h"
#include "sabercat/scene_loader.h"

using namespace sabercat;
namespace fs = std::filesystem;

namespace
{

int g_failures = 0;

void check(bool ok, const std::string & what)
{
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok)
    {
        ++g_failures;
    }
}

const char * formatName(SceneFormat format)
{
    switch (format)
    {
    case SceneFormat::PLY:
        return "PLY";
    case SceneFormat::SPLAT:
        return "SPLAT";
    case SceneFormat::SPZ:
        return "SPZ";
    case SceneFormat::GLTF:
        return "GLTF";
    case SceneFormat::CLOUD_V1:
        return "CLOUD_V1";
    case SceneFormat::KSPLAT:
        return "KSPLAT";
    case SceneFormat::SOG:
        return "SOG";
    }
    return "?";
}

constexpr size_t kFormatCount = 7;

void writeBytes(const std::string & path, const std::string & bytes)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

// Runs `action`, which must throw an exception whose message contains `needle`.
void expectThrow(const std::string & what, const std::string & needle, const std::function<void()> & action)
{
    try
    {
        action();
    }
    catch (const std::exception & error)
    {
        const std::string message = error.what();
        check(message.find(needle) != std::string::npos,
              what + " (expected \"" + needle + "\", got: " + message + ")");
        return;
    }
    check(false, what + " (no exception)");
}

struct DetectCase
{
    const char * path;       // relative to the scratch folder
    SceneFormat expected;
};

} // namespace

int main()
{
    examples::ScratchDir scratch("scene_loader_test");
    Device device;
    int exitCode = 0;
    try
    {
        device.initialize({}, /*enableValidation=*/true);

        // Fixtures. Each carries only the magic the loader inspects.
        writeBytes("scene.ply", "ply\nformat binary_little_endian 1.0\n");
        writeBytes("SCENE_UPPER.PLY", "ply\nformat binary_little_endian 1.0\n");
        writeBytes("scene.glb", "glTF\x02\x00\x00\x00");
        writeBytes("SCENE_UPPER.GLB", "glTF\x02\x00\x00\x00");
        writeBytes("scene.gltf", "{\"asset\":{\"version\":\"2.0\"}}");
        writeBytes("scene.splat", std::string(32, '\0'));
        writeBytes("scene.SPZ", std::string(32, '\0'));
        writeBytes("scene.ksplat", std::string(32, '\0'));
        writeBytes("scene.sog", std::string(32, '\0'));
        writeBytes("scene.meta.json", "{}");
        writeBytes("SCENE.META.JSON", "{}");
        writeBytes("manifest.json", "{}");
        writeBytes("MANIFEST.JSON", "{}");
        fs::create_directories("v1.2");
        writeBytes("v1.2/nested.ply", "ply\nformat binary_little_endian 1.0\n");
        fs::create_directories("cloud.v1.dir");
        writeBytes("cloud.v1.dir/manifest.json", "{}");

        // (a) every format, every case the loader dispatches on.
        const DetectCase detectCases[] = {
            {"scene.ply", SceneFormat::PLY},
            {"SCENE_UPPER.PLY", SceneFormat::PLY},
            {"v1.2/nested.ply", SceneFormat::PLY}, // dot in a folder name
            {"scene.splat", SceneFormat::SPLAT},
            {"missing.splat", SceneFormat::SPLAT}, // extension alone decides
            {"scene.SPZ", SceneFormat::SPZ},
            {"scene.gltf", SceneFormat::GLTF},
            {"scene.glb", SceneFormat::GLTF},
            {"SCENE_UPPER.GLB", SceneFormat::GLTF},
            {"missing.glb", SceneFormat::GLTF}, // a missing file falls through to the reader
            {"scene.ksplat", SceneFormat::KSPLAT},
            {"scene.sog", SceneFormat::SOG},
            {"scene.meta.json", SceneFormat::SOG},
            {"SCENE.META.JSON", SceneFormat::SOG}, // suffix match is case-insensitive
            {"manifest.json", SceneFormat::CLOUD_V1},
            {"MANIFEST.JSON", SceneFormat::CLOUD_V1},
            {"missing.json", SceneFormat::CLOUD_V1},
            {"cloud.v1.dir", SceneFormat::CLOUD_V1}, // a directory, no extension
        };
        std::set<SceneFormat> seen;
        for (const DetectCase & c : detectCases)
        {
            SceneFormat got = SceneFormat::PLY;
            bool threw = false;
            try
            {
                got = detectFormat(c.path);
            }
            catch (const std::exception & error)
            {
                threw = true;
                std::printf("  %s: %s\n", c.path, error.what());
            }
            const bool ok = !threw && got == c.expected;
            check(ok, std::string("detectFormat(") + c.path + ") == " + formatName(c.expected)
                          + (ok ? "" : std::string(", got ") + (threw ? "a throw" : formatName(got))));
            if (ok)
            {
                seen.insert(got);
            }
        }
        check(seen.size() == kFormatCount,
              "every SceneFormat value is produced by a table row (" + std::to_string(seen.size())
                  + " of " + std::to_string(kFormatCount) + ")");

        // (b) the magic must match the extension.
        writeBytes("wrong.ply", "PK\x03\x04 not a ply file");
        writeBytes("wrong.PLY", "PK\x03\x04 not a ply file");
        writeBytes("wrong.glb", "{\"asset\":{}}");
        expectThrow("detectFormat refuses a .ply without the ply magic", "missing PLY magic header",
                    [] { detectFormat("wrong.ply"); });
        expectThrow("detectFormat refuses an upper-case .PLY without the ply magic", "missing PLY magic header",
                    [] { detectFormat("wrong.PLY"); });
        expectThrow("detectFormat refuses a .glb without the glTF magic", "missing the binary glTF magic",
                    [] { detectFormat("wrong.glb"); });
        // A missing .ply is reported as a missing file, not as a magic failure.
        expectThrow("detectFormat refuses a .ply that does not exist",
                    "cannot open scene file", [] { detectFormat("missing.ply"); });

        // (c) unsupported names.
        writeBytes("readme.txt", "hello");
        writeBytes("noextension", "ply\n");
        expectThrow("detectFormat refuses an unsupported extension", "unsupported file extension",
                    [] { detectFormat("readme.txt"); });
        expectThrow("detectFormat refuses a file with no extension", "unsupported file extension",
                    [] { detectFormat("noextension"); });
        expectThrow("detectFormat refuses an empty extension-less name", "unsupported file extension",
                    [] { detectFormat(""); });
        writeBytes("lod-meta.json", "{}");
        expectThrow("detectFormat refuses a streamed SOG (lod-meta.json) by name", "streamed SOG (lod-meta.json)",
                    [] { detectFormat("lod-meta.json"); });

        // (d) loadScene: the COLMAP guard fires on the name before any parse,
        // even when the file itself is a valid PLY.
        writeBytes("input.ply", "ply\nformat binary_little_endian 1.0\n");
        expectThrow("loadScene refuses a COLMAP input.ply by name", "appears to be COLMAP/SfM output",
                    [&] { loadScene(device, "input.ply"); });
        expectThrow("loadScene reports a wrong-magic .ply", "missing PLY magic header",
                    [&] { loadScene(device, "wrong.ply"); });
        expectThrow("loadScene reports a missing .ply by name", "cannot open scene file",
                    [&] { loadScene(device, "does_not_exist.ply"); });
        expectThrow("loadScene reports a missing .glb by name", "failed to open",
                    [&] { loadScene(device, "does_not_exist.glb"); });
        expectThrow("loadScene reports an unsupported extension", "unsupported file extension",
                    [&] { loadScene(device, "readme.txt"); });

        device.shutdown();
    }
    catch (const std::exception & error)
    {
        std::printf("[FAIL] unexpected exception: %s\n", error.what());
        ++g_failures;
    }

    std::printf("\n%s (%d failure%s)\n", g_failures == 0 ? "scene_loader_test OK" : "scene_loader_test FAILED",
                g_failures, g_failures == 1 ? "" : "s");
    exitCode = g_failures == 0 ? 0 : 1;
    return exitCode;
}
