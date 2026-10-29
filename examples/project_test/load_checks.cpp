// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// ProjectIO load checks for malformed project.json fields, and round trips of
// large app state and a long layer name (WP6):
//
//  (a) Each malformed field is rejected by name: layers not an array, a missing
//      layers or app member, app not an object, a non-string name, a non-boolean
//      flag, a transform array of the wrong length, a scale of 0, a rotation that
//      is not unit length, an invalid layer id, a non-integer formatVersion.
//      Every archive holds only project.json, so each refusal is about the
//      field, not a missing layer folder.
//  (b) About 1 MB of app state (a long string, nested arrays and numbers)
//      round-trips, compared as parsed JSON values.
//  (c) A 20,000-character layer name (UTF-8, including multi-byte text) round-trips
//      bit-exactly, next to an ordinary layer.

#include <glm/glm.hpp>

#include <nlohmann/json.hpp>

#include <cstdio>
#include <exception>
#include <filesystem>
#include <functional>
#include <cstdint>
#include <vector>
#include <memory>
#include <string>

#include "sabercat/device.h"
#include "sabercat/file_map.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/project.h"
#include "zip_archive.h"

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

std::vector<uint8_t> bytesOf(const std::string & text)
{
    return std::vector<uint8_t>(text.begin(), text.end());
}

// A layer object with every member valid except the ones a case overrides.
std::string layerJson(const std::string & id = "a", const std::string & name = "\"layer\"",
                      const std::string & transform = R"({"translation":[0,0,0],"rotation":[0,0,0,1],"scale":1})",
                      const std::string & visible = "true", const std::string & antialiased = "false")
{
    return "{\"id\":\"" + id + "\",\"name\":" + name + ",\"visible\":" + visible
           + ",\"locked\":false,\"antialiased\":" + antialiased + ",\"transform\":" + transform + "}";
}

// Writes a project.json-only archive and loads it, expecting a refusal that
// contains `needle`.
void expectRefusal(Device & device, const fs::path & dir, const std::string & what, const std::string & json,
                   const std::string & needle)
{
    const fs::path file = dir / "malformed.sabercat";
    fs::remove(file);
    FileMap files;
    files["project.json"] = bytesOf(json);
    zip::writeStoredZip(file.string(), files);
    try
    {
        ProjectIO::load(device, file.string());
        check(false, what + " (loaded without an error)");
    }
    catch (const std::exception & error)
    {
        const std::string message = error.what();
        check(message.find(needle) != std::string::npos, what + " (" + message + ")");
    }
}

std::string header(const std::string & layers, const std::string & app = R"("app":{})")
{
    return R"({"format":"sabercat.project","formatVersion":1,"savedWith":{"sabercat":"x"},"layers":)" + layers
           + "," + app + "}";
}

void testMalformed(Device & device, const fs::path & dir)
{
    expectRefusal(device, dir, "layers must be an array", header("{}"), "layers must be an array");
    expectRefusal(device, dir, "missing layers member",
                  R"({"format":"sabercat.project","formatVersion":1,"savedWith":{"sabercat":"x"},"app":{}})",
                  "missing \"layers\"");
    expectRefusal(device, dir, "missing app member",
                  R"({"format":"sabercat.project","formatVersion":1,"savedWith":{"sabercat":"x"},"layers":[]})",
                  "missing \"app\"");
    expectRefusal(device, dir, "app must be an object", header("[]", R"("app":[])"), "app must be a JSON object");
    expectRefusal(device, dir, "formatVersion must be an integer",
                  R"({"format":"sabercat.project","formatVersion":1.5,"savedWith":{"sabercat":"x"},"layers":[],"app":{}})",
                  "formatVersion must be an integer");
    expectRefusal(device, dir, "non-string layer name", header("[" + layerJson("a", "5") + "]"),
                  "name must be a string");
    expectRefusal(device, dir, "non-boolean visible flag", header("[" + layerJson("a", "\"n\"", "{\"translation\":[0,0,0],\"rotation\":[0,0,0,1],\"scale\":1}", "1") + "]"),
                  "visible must be true or false");
    expectRefusal(device, dir, "translation of 2 numbers",
                  header("[" + layerJson("a", "\"n\"", R"({"translation":[0,0],"rotation":[0,0,0,1],"scale":1})") + "]"),
                  "translation must be an array of 3 numbers");
    expectRefusal(device, dir, "rotation of 3 numbers",
                  header("[" + layerJson("a", "\"n\"", R"({"translation":[0,0,0],"rotation":[0,0,1],"scale":1})") + "]"),
                  "rotation must be an array of 4 numbers");
    expectRefusal(device, dir, "scale 0", header("[" + layerJson("a", "\"n\"", R"({"translation":[0,0,0],"rotation":[0,0,0,1],"scale":0})") + "]"),
                  "transform scale must be > 0");
    expectRefusal(device, dir, "rotation not unit length",
                  header("[" + layerJson("a", "\"n\"", R"({"translation":[0,0,0],"rotation":[0,0,0,2],"scale":1})") + "]"),
                  "transform rotation is not unit length");
    expectRefusal(device, dir, "layer id with a space", header("[" + layerJson("a b") + "]"),
                  "id must be 1 to 64 characters");
    expectRefusal(device, dir, "missing antialiased flag",
                  header("[{\"id\":\"a\",\"name\":\"n\",\"visible\":true,\"locked\":false,"
                         "\"transform\":{\"translation\":[0,0,0],\"rotation\":[0,0,0,1],\"scale\":1}}]"),
                  "missing \"antialiased\"");
}

// A small cloud with fixed values.
std::shared_ptr<GaussianCloud> smallCloud(Device & device)
{
    auto cloud = std::make_shared<GaussianCloud>(device);
    cloud->initialize(5);
    for (size_t i = 0; i < 5; ++i)
    {
        cloud->positions()[i] = glm::vec4(static_cast<float>(i), 1.0f, -2.0f, 0.0f);
        cloud->scales()[i] = glm::vec4(0.1f, 0.2f, 0.3f, 0.0f);
        cloud->rotations()[i] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
        cloud->opacities()[i] = 0.5f;
        for (int k = 0; k < kShCoefficientsPerGaussian; ++k)
        {
            cloud->shCoefficients()[i * kShCoefficientsPerGaussian + k] = glm::vec4(0.0f);
        }
    }
    cloud->setShDegree(0);
    cloud->computeBounds();
    return cloud;
}

void testLargeRoundTrips(Device & device, const fs::path & dir)
{
    // (b) about 1 MB of app state: a long string, an array of numbers, nested objects.
    nlohmann::json app;
    app["blob"] = std::string(1000000, 'q') + "end";
    nlohmann::json numbers = nlohmann::json::array();
    for (int i = 0; i < 20000; ++i)
    {
        numbers.push_back(i * 0.125 - 1000.0);
    }
    app["numbers"] = numbers;
    app["nested"] = {{"camera", {{"yaw", 0.25}, {"pitch", -1.5}}}, {"flags", {true, false, nullptr}}};
    const std::string appText = app.dump();

    // (c) a 20,000-character layer name: a UTF-8 pattern of ASCII, two- and
    // three-byte characters.
    std::string longName;
    const std::string pattern = "ab\xC3\xBC\xE5\xB1\xA4-";
    while (longName.size() < 20000)
    {
        longName += pattern;
    }

    Project project;
    project.appStateJson = appText;
    ProjectLayer first;
    first.cloud = smallCloud(device);
    first.id = "first";
    first.name = longName;
    first.transform.translation = glm::vec3(1.0f, 2.0f, 3.0f);
    project.layers.push_back(first);
    ProjectLayer second = first;
    second.id = "second";
    second.name = "short";
    project.layers.push_back(second);

    const fs::path file = dir / "large.sabercat";
    ProjectIO::save(project, file.string());
    const Project back = ProjectIO::load(device, file.string());

    check(nlohmann::json::parse(back.appStateJson) == app,
          "app state of about 1 MB round trips (" + std::to_string(appText.size()) + " bytes)");
    check(back.layers.size() == 2 && back.layers[0].name == longName,
          "a 20,000-character UTF-8 layer name round trips exactly (" + std::to_string(longName.size()) + " bytes)");
    check(back.layers.size() == 2 && back.layers[1].name == "short" && back.layers[1].id == "second",
          "the layer after the long name is intact");
}

} // namespace

// Entry point called by project_test's main. Returns the failure count.
int runLoadChecks(Device & device, const fs::path & dir)
{
    g_failures = 0;
    testMalformed(device, dir);
    testLargeRoundTrips(device, dir);
    std::printf("project load checks: %s (%d failure%s)\n", g_failures == 0 ? "OK" : "FAILED", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures;
}
