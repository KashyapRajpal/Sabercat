// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Headless test for the .sabercat project file (ProjectIO).
//
// Synthetic clouds come from a seeded RNG, so no assets are needed. Covers:
//  (1) round trip, bit-exact: host streams, shDegree, antialiased, transform
//      floats, flags, ids, UTF-8 names, relight streams, app state as JSON.
//  (2) determinism: saving one project twice gives byte-identical files.
//  (3) JSON float edge values round-trip bitwise; NaN/Inf transforms throw.
//  (4) atomic save: a rejected save leaves the existing file untouched and
//      leaves no temporary file behind.
//  (5) corruption: a flipped data byte (CRC), truncation, a newer format
//      version, and a layer whose files are missing all throw.
//  (6) each layer's layers/<id>/ folder is plain cloud-v1: it reads with
//      CloudV1Reader::read(device, directory) bit-exactly, and a project whose
//      layer manifests carry the legacy schema id loads bit-exactly.
//  (7) empty project round trip; unknown archive entries are ignored.
//
// Validation messages are logged as "[validation]"; the run must have none.

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <nlohmann/json.hpp>

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "sabercat/cloud_v1_reader.h"
#include "sabercat/device.h"
#include "sabercat/file_map.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/project.h"
#include "sabercat/render_config.h"
#include "sabercat/version.h"
#include "zip_archive.h"

using namespace sabercat;
namespace fs = std::filesystem;

// load_checks.cpp: malformed project.json fields, large app state, long layer name.
int runLoadChecks(Device & device, const fs::path & dir);

namespace
{

int g_failures = 0;

using Bytes = std::vector<uint8_t>;

void check(bool ok, const std::string & what)
{
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok)
    {
        ++g_failures;
    }
}

// Runs `action`, which must throw an std::exception whose message contains `needle`.
void expectThrow(const std::string & what, const std::function<void()> & action, const std::string & needle)
{
    try
    {
        action();
    }
    catch (const std::exception & error)
    {
        const std::string message = error.what();
        check(needle.empty() || message.find(needle) != std::string::npos, what + " (" + message + ")");
        return;
    }
    check(false, what + " (no exception)");
}

Bytes readBytes(const fs::path & file)
{
    std::ifstream in(file, std::ios::binary);
    return Bytes(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

void writeBytes(const fs::path & file, const Bytes & bytes)
{
    std::ofstream out(file, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

Bytes bytesOfText(const std::string & text)
{
    return Bytes(text.begin(), text.end());
}

// The UTF-8 form of a path. ProjectIO takes UTF-8 std::strings; fs::path::string()
// would convert through the ANSI codepage on Windows.
std::string utf8String(const fs::path & path)
{
    const std::u8string text = path.u8string();
    return std::string(reinterpret_cast<const char *>(text.data()), text.size());
}

// Bitwise equality: -0.0 differs from 0.0.
bool sameFloat(float a, float b)
{
    return std::memcmp(&a, &b, sizeof(float)) == 0;
}

template <typename T> bool sameBits(const std::vector<T> & a, const std::vector<T> & b)
{
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

bool sameTransform(const LayerTransform & a, const LayerTransform & b)
{
    return sameFloat(a.rotation.w, b.rotation.w) && sameFloat(a.rotation.x, b.rotation.x)
           && sameFloat(a.rotation.y, b.rotation.y) && sameFloat(a.rotation.z, b.rotation.z)
           && sameFloat(a.translation.x, b.translation.x) && sameFloat(a.translation.y, b.translation.y)
           && sameFloat(a.translation.z, b.translation.z) && sameFloat(a.scale, b.scale);
}

// Names the first stream that differs, or returns "" when the clouds match.
// `fullProject` also compares what only the project file carries: the relight
// streams and the antialiased flag (the cloud-v1 fixture has neither).
std::string compareClouds(const GaussianCloud & a, const GaussianCloud & b, bool fullProject)
{
    if (a.count() != b.count())
        return "count";
    if (a.shDegree() != b.shDegree())
        return "shDegree";
    if (fullProject && a.antialiased() != b.antialiased())
        return "antialiased";
    if (!sameBits(a.positions(), b.positions()))
        return "positions";
    if (!sameBits(a.scales(), b.scales()))
        return "scales";
    if (!sameBits(a.rotations(), b.rotations()))
        return "rotations";
    if (!sameBits(a.opacities(), b.opacities()))
        return "opacities";
    if (!sameBits(a.shCoefficients(), b.shCoefficients()))
        return "shCoefficients";
    if (fullProject)
    {
        if (!sameBits(a.normals(), b.normals()))
            return "normals";
        if (!sameBits(a.baseColorRoughness(), b.baseColorRoughness()))
            return "baseColorRoughness";
        if (!sameBits(a.metallic(), b.metallic()))
            return "metallic";
        if (!sameBits(a.visibility(), b.visibility()))
            return "visibility";
    }
    return "";
}

// Which relight streams a synthetic cloud carries.
struct RelightSet
{
    bool normals = false;
    bool baseColorRoughness = false;
    bool metallic = false;
    bool visibility = false;
};

// A seeded synthetic cloud. Lanes the cloud-v1 format does not store (the w of
// means, scales and SH) are zero, as every loader produces them; coefficients
// above `degree` are zero.
std::shared_ptr<GaussianCloud> makeCloud(Device & device, size_t n, uint32_t degree, uint32_t seed,
                                         RelightSet relight)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    std::uniform_real_distribution<float> positive(0.001f, 0.5f);

    auto cloud = std::make_shared<GaussianCloud>(device);
    cloud->initialize(n);
    cloud->setShDegree(degree);
    cloud->setAntialiased(false);

    const uint32_t activeCoefficients = (degree + 1) * (degree + 1);
    for (size_t i = 0; i < n; ++i)
    {
        cloud->positions()[i] = glm::vec4(unit(rng) * 10.0f, unit(rng) * 10.0f, unit(rng) * 10.0f, 0.0f);
        cloud->scales()[i] = glm::vec4(positive(rng), positive(rng), positive(rng), 0.0f);
        const glm::quat q = glm::normalize(glm::quat(unit(rng), unit(rng), unit(rng), unit(rng)));
        cloud->rotations()[i] = glm::vec4(q.x, q.y, q.z, q.w);
        cloud->opacities()[i] = positive(rng);
        for (uint32_t k = 0; k < static_cast<uint32_t>(kShCoefficientsPerGaussian); ++k)
        {
            cloud->shCoefficients()[i * kShCoefficientsPerGaussian + k] =
                k < activeCoefficients ? glm::vec4(unit(rng), unit(rng), unit(rng), 0.0f) : glm::vec4(0.0f);
        }
    }
    if (relight.normals)
    {
        cloud->normals().resize(n);
        for (glm::vec4 & v : cloud->normals())
            v = glm::vec4(unit(rng), unit(rng), unit(rng), unit(rng));
    }
    if (relight.baseColorRoughness)
    {
        cloud->baseColorRoughness().resize(n);
        for (glm::vec4 & v : cloud->baseColorRoughness())
            v = glm::vec4(positive(rng), positive(rng), positive(rng), positive(rng));
    }
    if (relight.metallic)
    {
        cloud->metallic().resize(n);
        for (float & v : cloud->metallic())
            v = positive(rng);
    }
    if (relight.visibility)
    {
        cloud->visibility().resize(n * kVisibilityCoefficients);
        for (float & v : cloud->visibility())
            v = unit(rng);
    }
    return cloud;
}

const char * kAppState =
    R"({"camera":{"position":[0.1,-2.5,3e-7],"fov":45.0,"orbit":{"yaw":-0.0,"pitch":1.0e-5}},"flags":[true,false,null],"label":"x","count":7})";

ProjectLayer makeLayer(const std::string & id, const std::string & name, std::shared_ptr<GaussianCloud> cloud,
                       const LayerTransform & transform, bool visible, bool locked, bool antialiased)
{
    ProjectLayer layer;
    layer.cloud = std::move(cloud);
    layer.cloud->setAntialiased(antialiased);
    layer.id = id;
    layer.name = name;
    layer.transform = transform;
    layer.visible = visible;
    layer.locked = locked;
    return layer;
}

// Three layers covering the cases: plain degree 0; degree 3 with all relight
// streams, a placed transform, antialiased and locked; degree 1 with normals and
// metallic only, a UTF-8 name and hidden.
Project buildProject(Device & device)
{
    Project project;

    project.layers.push_back(
        makeLayer("a", "Base", makeCloud(device, 2000, 0, 1, {}), LayerTransform{}, true, false, false));

    LayerTransform placed;
    placed.rotation = glm::normalize(glm::quat(0.6f, 0.3f, -0.7f, 0.2f));
    placed.translation = glm::vec3(1.5f, -0.25f, 1000.0f);
    placed.scale = 2.5f;
    project.layers.push_back(makeLayer("layer-b_2", "Placed",
                                       makeCloud(device, 3000, 3, 2, {true, true, true, true}), placed, true,
                                       true, true));

    // "Ünïcödé 層" as UTF-8. Each escape is closed before the next literal starts,
    // so no hex escape swallows a following hex digit.
    const char * unicodeName = "\xC3\x9C"
                               "n"
                               "\xC3\xAF"
                               "c"
                               "\xC3\xB6"
                               "d"
                               "\xC3\xA9"
                               " "
                               "\xE5\xB1\xA4";
    LayerTransform hiddenTransform;
    hiddenTransform.scale = 0.1f;
    project.layers.push_back(makeLayer("C", unicodeName, makeCloud(device, 2500, 1, 3, {true, false, true, false}),
                                       hiddenTransform, false, false, false));

    project.appStateJson = kAppState;
    return project;
}

void testRoundTrip(Device & device, const fs::path & file, const Project & original)
{
    ProjectIO::save(original, utf8String(file));
    const Project loaded = ProjectIO::load(device, utf8String(file));

    check(loaded.layers.size() == original.layers.size(), "round trip: layer count");
    for (size_t i = 0; i < original.layers.size() && i < loaded.layers.size(); ++i)
    {
        const ProjectLayer & a = original.layers[i];
        const ProjectLayer & b = loaded.layers[i];
        const std::string where = "round trip layer '" + a.id + "'";
        check(a.id == b.id, where + ": id");
        check(a.name == b.name, where + ": name");
        check(a.visible == b.visible, where + ": visible");
        check(a.locked == b.locked, where + ": locked");
        check(sameTransform(a.transform, b.transform), where + ": transform bits");
        const std::string diff = compareClouds(*a.cloud, *b.cloud, true);
        check(diff.empty(), where + ": cloud bit-exact" + (diff.empty() ? "" : " (differs: " + diff + ")"));
    }

    const auto originalJson = nlohmann::json::parse(original.appStateJson);
    const auto loadedJson = nlohmann::json::parse(loaded.appStateJson);
    check(originalJson.is_object() && originalJson == loadedJson, "round trip: app state equal as JSON");
    check(loaded.savedWithVersion == SABERCAT_VERSION_STRING, "round trip: savedWithVersion == project version");
}

void testDeterminism(const Project & project, const fs::path & dir)
{
    const fs::path first = dir / "determinism_1.sabercat";
    const fs::path second = dir / "determinism_2.sabercat";
    ProjectIO::save(project, first.string());
    ProjectIO::save(project, second.string());
    check(readBytes(first) == readBytes(second), "determinism: same project saves byte-identically");
}

void testFloatEdges(Device & device, const fs::path & dir)
{
    Project project;
    LayerTransform lowEdge;
    lowEdge.rotation = glm::quat(1.0f, -0.0f, 0.0f, 0.0f);
    lowEdge.translation = glm::vec3(std::numeric_limits<float>::denorm_min(), -0.0f, 0.1f);
    lowEdge.scale = 1e-30f;
    project.layers.push_back(makeLayer("low", "low", makeCloud(device, 4, 0, 10, {}), lowEdge, true, false, false));

    LayerTransform highEdge;
    highEdge.translation = glm::vec3(FLT_MAX, -FLT_MAX, FLT_MIN);
    highEdge.scale = FLT_MAX;
    project.layers.push_back(
        makeLayer("high", "high", makeCloud(device, 4, 0, 11, {}), highEdge, true, false, false));

    const fs::path file = dir / "float_edges.sabercat";
    ProjectIO::save(project, file.string());
    const Project loaded = ProjectIO::load(device, file.string());
    check(loaded.layers.size() == 2, "float edges: two layers load");
    if (loaded.layers.size() == 2)
    {
        check(sameTransform(project.layers[0].transform, loaded.layers[0].transform),
              "float edges: denormal, -0.0, 0.1f, 1e-30f transform is bit-exact");
        check(sameTransform(project.layers[1].transform, loaded.layers[1].transform),
              "float edges: FLT_MAX, -FLT_MAX, FLT_MIN, FLT_MAX scale is bit-exact");
    }

    // Invalid transforms are rejected at save time.
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    const fs::path rejected = dir / "rejected_transform.sabercat";
    auto withTransform = [&](const LayerTransform & transform)
    {
        Project bad;
        bad.layers.push_back(makeLayer("bad", "bad", makeCloud(device, 2, 0, 12, {}), transform, true, false, false));
        return bad;
    };
    LayerTransform nanTranslation;
    nanTranslation.translation.x = nan;
    LayerTransform infScale;
    infScale.scale = inf;
    LayerTransform nanRotation;
    nanRotation.rotation = glm::quat(nan, 0.0f, 0.0f, 0.0f);
    LayerTransform zeroScale;
    zeroScale.scale = 0.0f;
    LayerTransform stretched;
    stretched.rotation = glm::quat(2.0f, 0.0f, 0.0f, 0.0f);

    expectThrow("NaN translation is rejected at save",
                [&] { ProjectIO::save(withTransform(nanTranslation), rejected.string()); }, "non-finite");
    expectThrow("Inf scale is rejected at save", [&] { ProjectIO::save(withTransform(infScale), rejected.string()); },
                "non-finite");
    expectThrow("NaN rotation is rejected at save", [&] { ProjectIO::save(withTransform(nanRotation), rejected.string()); },
                "non-finite");
    expectThrow("zero scale is rejected at save", [&] { ProjectIO::save(withTransform(zeroScale), rejected.string()); },
                "scale");
    expectThrow("non-unit rotation is rejected at save",
                [&] { ProjectIO::save(withTransform(stretched), rejected.string()); }, "unit length");
    check(!fs::exists(rejected), "rejected saves create no file");
}

void testAtomicSave(Device & device, const Project & original, const fs::path & dir)
{
    const fs::path target = dir / "atomic.sabercat";
    ProjectIO::save(original, target.string());
    const Bytes before = readBytes(target);

    Project duplicate = original;
    duplicate.layers[1].id = duplicate.layers[0].id;
    expectThrow("duplicate id is rejected at save", [&] { ProjectIO::save(duplicate, target.string()); }, "duplicate");

    Project mismatched = original;
    auto cloud = makeCloud(device, 10, 0, 20, {});
    cloud->positions().pop_back();
    mismatched.layers.push_back(makeLayer("short", "short", cloud, LayerTransform{}, true, false, false));
    expectThrow("host vector size mismatch is rejected at save", [&] { ProjectIO::save(mismatched, target.string()); },
                "layer 'short'");

    Project badId = original;
    badId.layers[0].id = "has space";
    expectThrow("invalid id is rejected at save", [&] { ProjectIO::save(badId, target.string()); }, "id must be");

    Project badApp = original;
    badApp.appStateJson = "[1,2,3]";
    expectThrow("non-object app state is rejected at save", [&] { ProjectIO::save(badApp, target.string()); },
                "JSON object");

    check(readBytes(target) == before, "atomic: existing file is byte-identical after rejected saves");
    check(!fs::exists(target.string() + ".tmp"), "atomic: no .tmp file remains after rejected saves");
}

// Re-reads the archive's entries and returns the file offset of one entry's data.
size_t entryDataOffset(const Bytes & archive, const std::string & name)
{
    const auto entries = zip::parseZip(archive, "project.sabercat", "project_test");
    const auto found = entries.find(name);
    if (found == entries.end())
    {
        return 0;
    }
    return static_cast<size_t>(found->second.dataOffset);
}

void testCorruption(Device & device, const fs::path & good, const fs::path & dir)
{
    const Bytes archive = readBytes(good);

    // A flipped byte inside a stored .bin fails the CRC check.
    const size_t offset = entryDataOffset(archive, "layers/a/means.f32le.bin");
    check(offset > 0, "corruption: means entry found in archive");
    Bytes flipped = archive;
    flipped[offset + 7] ^= 0x40;
    const fs::path flippedFile = dir / "flipped.sabercat";
    writeBytes(flippedFile, flipped);
    expectThrow("corruption: flipped data byte fails to load", [&] { ProjectIO::load(device, flippedFile.string()); },
                "");

    // Truncation.
    Bytes truncated(archive.begin(), archive.begin() + static_cast<std::ptrdiff_t>(archive.size() / 2));
    const fs::path truncatedFile = dir / "truncated.sabercat";
    writeBytes(truncatedFile, truncated);
    expectThrow("corruption: truncated file fails to load", [&] { ProjectIO::load(device, truncatedFile.string()); },
                "");

    // A newer format version is refused by name.
    FileMap newer;
    newer["project.json"] = bytesOfText(R"({"format":"sabercat.project","formatVersion":2,"savedWith":{"sabercat":"9.9.9"},"layers":[],"app":{}})");
    const fs::path newerFile = dir / "newer.sabercat";
    zip::writeStoredZip(newerFile.string(), newer);
    expectThrow("corruption: formatVersion 2 is refused as newer", [&] { ProjectIO::load(device, newerFile.string()); },
                "newer");

    // A wrong format name is refused.
    FileMap wrongFormat;
    wrongFormat["project.json"] = bytesOfText(R"({"format":"other","formatVersion":1,"savedWith":{"sabercat":"x"},"layers":[],"app":{}})");
    const fs::path wrongFile = dir / "wrong_format.sabercat";
    zip::writeStoredZip(wrongFile.string(), wrongFormat);
    expectThrow("corruption: wrong format name is refused", [&] { ProjectIO::load(device, wrongFile.string()); },
                "not a Sabercat project");

    // A layer listed in project.json whose files are absent is named in the error.
    FileMap missingFiles;
    missingFiles["project.json"] = bytesOfText(R"({"format":"sabercat.project","formatVersion":1,"savedWith":{"sabercat":"x"},"layers":[{"id":"ghost","name":"g","visible":true,"locked":false,"antialiased":false,"transform":{"translation":[0,0,0],"rotation":[0,0,0,1],"scale":1}}],"app":{}})");
    const fs::path missingFile = dir / "missing_layer.sabercat";
    zip::writeStoredZip(missingFile.string(), missingFiles);
    expectThrow("corruption: missing layer files throw, naming the layer",
                [&] { ProjectIO::load(device, missingFile.string()); }, "ghost");
}

void testLayerIsPlainCloudV1(Device & device, const fs::path & good, const Project & original, const fs::path & dir)
{
    // Layer "layer-b_2" holds the degree-3 cloud with relight streams. Its
    // cloud-v1 entries (without the relight/ folder) become a plain fixture folder.
    const FileMap archive = zip::readZip(good.string(), "project_test");
    const fs::path fixture = dir / "layer_b_fixture";
    fs::create_directories(fixture);
    const std::string prefix = "layers/layer-b_2/";
    size_t copied = 0;
    for (const auto & [name, bytes] : archive)
    {
        if (name.compare(0, prefix.size(), prefix) != 0)
            continue;
        const std::string inside = name.substr(prefix.size());
        if (inside.compare(0, 8, "relight/") == 0)
            continue;
        writeBytes(fixture / inside, bytes);
        ++copied;
    }
    check(copied == 7, "plain cloud-v1: layer folder holds manifest.json and six streams (degree 3)");

    const std::shared_ptr<GaussianCloud> fromDisk = CloudV1Reader::read(device, fixture.string());
    const ProjectLayer & layer = original.layers[1];
    // The fixture has neither relight streams nor the antialiased flag, so only
    // the canonical streams are compared.
    const std::string diff = compareClouds(*layer.cloud, *fromDisk, false);
    check(diff.empty(), "plain cloud-v1: layers/<id>/ reads with CloudV1Reader bit-exactly" +
                            (diff.empty() ? std::string() : " (differs: " + diff + ")"));

    // A project saved before Sabercat 1.0: the same archive with every layer's
    // cloud-v1 manifest carrying the legacy schema id must load bit-exactly.
    FileMap legacyArchive = archive;
    size_t swapped = 0;
    for (auto & [name, bytes] : legacyArchive)
    {
        if (name.compare(0, 7, "layers/") != 0)
            continue;
        std::string text(bytes.begin(), bytes.end());
        const std::string current = "\"sabercat.cloud-v1\"";
        if (text.find(current) == std::string::npos)
            continue;
        const size_t at = text.find(current);
        text.replace(at, current.size(), "\"gaussmith.sabercat-cloud-v1\"");
        bytes = bytesOfText(text);
        ++swapped;
    }
    check(swapped == original.layers.size(),
          "legacy project: every layer manifest rewritten with the legacy schema id (" + std::to_string(swapped)
              + " layers)");
    const fs::path legacyFile = dir / "legacy_layers.sabercat";
    zip::writeStoredZip(legacyFile.string(), legacyArchive);
    const Project legacyLoaded = ProjectIO::load(device, legacyFile.string());
    check(legacyLoaded.layers.size() == original.layers.size(), "legacy project: layer count");
    for (size_t i = 0; i < original.layers.size() && i < legacyLoaded.layers.size(); ++i)
    {
        const std::string legacyDiff = compareClouds(*original.layers[i].cloud, *legacyLoaded.layers[i].cloud, true);
        check(legacyDiff.empty(), "legacy project: layer '" + original.layers[i].id + "' loads bit-exactly"
                                      + (legacyDiff.empty() ? std::string() : " (differs: " + legacyDiff + ")"));
    }
}

void testEmptyAndUnknown(Device & device, const fs::path & dir)
{
    Project empty;
    const fs::path emptyFile = dir / "empty.sabercat";
    ProjectIO::save(empty, emptyFile.string());
    const Project loadedEmpty = ProjectIO::load(device, emptyFile.string());
    check(loadedEmpty.layers.empty(), "empty project: zero layers round-trip");
    check(nlohmann::json::parse(loadedEmpty.appStateJson) == nlohmann::json::object(),
          "empty project: default app state is an empty object");
    check(loadedEmpty.savedWithVersion == SABERCAT_VERSION_STRING, "empty project: savedWithVersion");

    // Entries no layer names are ignored (with one warning), so a newer writer's
    // extra files do not break this reader.
    FileMap withExtras = zip::readZip((dir / "empty.sabercat").string(), "project_test");
    withExtras["extras/notes.txt"] = bytesOfText("hello");
    withExtras["layers/ghost/means.f32le.bin"] = bytesOfText("not a layer");
    const fs::path extrasFile = dir / "extras.sabercat";
    zip::writeStoredZip(extrasFile.string(), withExtras);
    const Project loadedExtras = ProjectIO::load(device, extrasFile.string());
    check(loadedExtras.layers.empty(), "unknown entries: project still loads with unknown entries present");
}

// Saves and loads a project under a directory and file name that are not ASCII,
// so the UTF-8 path must reach the filesystem intact (not through the ANSI codepage).
void testNonAsciiPath(Device & device, const Project & original, const fs::path & dir)
{
    // "Ünïcödé 層" and "проект.sabercat" as UTF-8 byte escapes (see buildProject).
    const std::string unicodeDir = "\xC3\x9C"
                                   "n"
                                   "\xC3\xAF"
                                   "c"
                                   "\xC3\xB6"
                                   "d"
                                   "\xC3\xA9"
                                   " \xE5\xB1\xA4";
    const std::string cyrillicFile = "\xD0\xBF\xD1\x80\xD0\xBE\xD0\xB5\xD0\xBA\xD1\x82.sabercat";
    const fs::path nonAsciiDir = dir / fs::u8path(unicodeDir);
    fs::create_directories(nonAsciiDir);
    const fs::path file = nonAsciiDir / fs::u8path(cyrillicFile);

    testRoundTrip(device, file, original);
    check(fs::exists(file), "non-ASCII path: project file exists under its UTF-8 name");

    // Saving the loaded project again gives the same bytes on disk.
    const fs::path again = nonAsciiDir / fs::u8path("again_" + cyrillicFile);
    ProjectIO::save(ProjectIO::load(device, utf8String(file)), utf8String(again));
    check(readBytes(file) == readBytes(again), "non-ASCII path: load then save is byte-identical");

    check(!fs::exists(fs::u8path(utf8String(file) + ".tmp")), "non-ASCII path: no .tmp file remains");
    fs::remove_all(nonAsciiDir);
}

} // namespace

int main()
{
    int exitCode = 0;
    try
    {
        const fs::path dir = fs::temp_directory_path() / "sabercat_project_test";
        fs::remove_all(dir);
        fs::create_directories(dir);

        Device device;
        device.initialize({}, /*enableValidation=*/true);
        {
            const Project original = buildProject(device);
            const fs::path good = dir / "good.sabercat";
            testRoundTrip(device, good, original);
            testNonAsciiPath(device, original, dir);
            testDeterminism(original, dir);
            testFloatEdges(device, dir);
            testAtomicSave(device, original, dir);
            testCorruption(device, good, dir);
            testLayerIsPlainCloudV1(device, good, original, dir);
            testEmptyAndUnknown(device, dir);
            g_failures += runLoadChecks(device, dir); // load_checks.cpp
        }
        device.shutdown();
        fs::remove_all(dir);
    }
    catch (const std::exception & error)
    {
        std::printf("[FAIL] unexpected exception: %s\n", error.what());
        ++g_failures;
    }

    std::printf("%s: %d failure(s)\n", g_failures == 0 ? "project_test PASSED" : "project_test FAILED", g_failures);
    exitCode = g_failures == 0 ? 0 : 1;
    return exitCode;
}
