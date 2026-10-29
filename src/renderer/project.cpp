// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/project.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sabercat/cloud_v1_reader.h"
#include "sabercat/cloud_v1_writer.h"
#include "sabercat/file_map.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/logger.h"
#include "sabercat/version.h"
#include "unit_quaternion.h"
#include "zip_archive.h"

// Stored zip archives are readable by any tool; this file defines only the
// project.json layout and the per-layer folder convention on top of them.

namespace sabercat
{

namespace
{

constexpr const char * kFormat = "sabercat.project";
constexpr int64_t kFormatVersion = 1;
constexpr const char * kProjectEntry = "project.json";
constexpr const char * kLayersFolder = "layers/";
constexpr const char * kRelightFolder = "relight/";
constexpr const char * kManifestName = "manifest.json";
constexpr size_t kMaxIdLength = 64;

using Json = nlohmann::ordered_json;
using Bytes = std::vector<uint8_t>;

[[noreturn]] void fail(const std::string & message)
{
    throw std::runtime_error("ProjectIO: " + message);
}

bool validId(const std::string & id)
{
    if (id.empty() || id.size() > kMaxIdLength)
    {
        return false;
    }
    for (const char c : id)
    {
        const bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9')
                        || c == '_' || c == '-';
        if (!ok)
        {
            return false;
        }
    }
    return true;
}

// Rejects a transform that cannot be baked: non-finite values, a scale that is
// not positive, or a rotation that is not unit length.
void validateTransform(const LayerTransform & transform, const std::string & where)
{
    const glm::quat & q = transform.rotation;
    const float values[] = {q.w,
                            q.x,
                            q.y,
                            q.z,
                            transform.translation.x,
                            transform.translation.y,
                            transform.translation.z,
                            transform.scale};
    for (const float value : values)
    {
        if (!std::isfinite(value))
        {
            fail(where + ": transform has a non-finite value");
        }
    }
    if (!(transform.scale > 0.0f))
    {
        fail(where + ": transform scale must be > 0");
    }
    if (!isUnitLength(q.w, q.x, q.y, q.z))
    {
        fail(where + ": transform rotation is not unit length");
    }
}

Json transformToJson(const LayerTransform & transform)
{
    Json translation = Json::array();
    translation.push_back(static_cast<double>(transform.translation.x));
    translation.push_back(static_cast<double>(transform.translation.y));
    translation.push_back(static_cast<double>(transform.translation.z));

    Json rotation = Json::array();
    rotation.push_back(static_cast<double>(transform.rotation.x));
    rotation.push_back(static_cast<double>(transform.rotation.y));
    rotation.push_back(static_cast<double>(transform.rotation.z));
    rotation.push_back(static_cast<double>(transform.rotation.w));

    Json json = Json::object();
    json["translation"] = translation;
    json["rotation"] = rotation;
    json["scale"] = static_cast<double>(transform.scale);
    return json;
}

// Member lookup that names what is missing.
const Json & member(const Json & object, const char * key, const std::string & where)
{
    if (!object.is_object() || !object.contains(key))
    {
        fail(where + ": missing \"" + key + "\"");
    }
    return object.at(key);
}

std::string asString(const Json & value, const std::string & where)
{
    if (!value.is_string())
    {
        fail(where + " must be a string");
    }
    return value.get<std::string>();
}

bool asBool(const Json & value, const std::string & where)
{
    if (!value.is_boolean())
    {
        fail(where + " must be true or false");
    }
    return value.get<bool>();
}

float asFloat(const Json & value, const std::string & where)
{
    if (!value.is_number())
    {
        fail(where + " must be a number");
    }
    return static_cast<float>(value.get<double>());
}

// An array of exactly `count` numbers, as floats.
std::vector<float> asFloats(const Json & value, size_t count, const std::string & where)
{
    if (!value.is_array() || value.size() != count)
    {
        fail(where + " must be an array of " + std::to_string(count) + " numbers");
    }
    std::vector<float> out;
    out.reserve(count);
    for (size_t i = 0; i < count; ++i)
    {
        out.push_back(asFloat(value[i], where));
    }
    return out;
}

LayerTransform transformFromJson(const Json & json, const std::string & where)
{
    if (!json.is_object())
    {
        fail(where + " transform must be an object");
    }
    const std::vector<float> t = asFloats(member(json, "translation", where), 3, where + " translation");
    const std::vector<float> r = asFloats(member(json, "rotation", where), 4, where + " rotation");

    LayerTransform transform;
    transform.translation = glm::vec3(t[0], t[1], t[2]);
    transform.rotation = glm::quat(r[3], r[0], r[1], r[2]); // glm (w, x, y, z)
    transform.scale = asFloat(member(json, "scale", where), where + " scale");
    validateTransform(transform, where);
    return transform;
}

std::string bytesToString(const Bytes & bytes)
{
    return std::string(bytes.begin(), bytes.end());
}

} // namespace

void ProjectIO::save(const Project & project, const std::string & path)
{
    // Everything is built and checked in memory first, so an invalid project
    // throws before any file is created.
    FileMap files;
    Json layers = Json::array();
    std::set<std::string> ids;
    for (const ProjectLayer & layer : project.layers)
    {
        const std::string where = "save: layer '" + layer.id + "'";
        if (!validId(layer.id))
        {
            fail(where + ": id must be 1 to 64 characters of [A-Za-z0-9_-]");
        }
        if (!ids.insert(layer.id).second)
        {
            fail(where + ": duplicate layer id");
        }
        if (!layer.cloud)
        {
            fail(where + ": null cloud");
        }
        validateTransform(layer.transform, where);

        try
        {
            const FileMap cloudFiles = CloudV1Writer::encode(*layer.cloud);
            for (const auto & [name, bytes] : cloudFiles)
            {
                files["layers/" + layer.id + "/" + name] = bytes;
            }
            const FileMap relightFiles = RelightSidecar::encode(*layer.cloud);
            for (const auto & [name, bytes] : relightFiles)
            {
                files["layers/" + layer.id + "/" + kRelightFolder + name] = bytes;
            }
        }
        catch (const std::runtime_error & error)
        {
            fail(where + ": " + error.what());
        }

        Json entry = Json::object();
        entry["id"] = layer.id;
        entry["name"] = layer.name;
        entry["visible"] = layer.visible;
        entry["locked"] = layer.locked;
        entry["antialiased"] = layer.cloud->antialiased();
        entry["transform"] = transformToJson(layer.transform);
        layers.push_back(entry);
    }

    Json app;
    try
    {
        app = Json::parse(project.appStateJson);
    }
    catch (const Json::exception & error)
    {
        fail(std::string("save: app state is not valid JSON: ") + error.what());
    }
    if (!app.is_object())
    {
        fail("save: app state must be a JSON object");
    }

    std::string text;
    try
    {
        Json root = Json::object();
        root["format"] = kFormat;
        root["formatVersion"] = kFormatVersion;
        Json savedWith = Json::object();
        savedWith["sabercat"] = sabercat::kVersionString;
        root["savedWith"] = savedWith;
        root["layers"] = layers;
        root["app"] = app;
        text = root.dump(2) + "\n";
    }
    catch (const Json::exception & error)
    {
        fail(std::string("save: project.json cannot be written (") + error.what() + ")");
    }
    files[kProjectEntry] = Bytes(text.begin(), text.end());

    // Write beside the destination, then rename over it: a failure at any point
    // leaves the existing file as it was.
    const std::string tempPath = path + ".tmp";
    try
    {
        zip::writeStoredZip(tempPath, files);
        std::filesystem::rename(std::filesystem::u8path(tempPath), std::filesystem::u8path(path));
    }
    catch (...)
    {
        std::error_code ignored;
        std::filesystem::remove(std::filesystem::u8path(tempPath), ignored);
        throw;
    }
}

Project ProjectIO::load(Device & device, const std::string & path)
{
    FileMap files = zip::readZip(path, "ProjectIO::load");

    const auto projectIt = files.find(kProjectEntry);
    if (projectIt == files.end())
    {
        fail("load: " + path + " has no project.json");
    }
    Json root;
    try
    {
        root = Json::parse(bytesToString(projectIt->second));
    }
    catch (const Json::exception & error)
    {
        fail(std::string("load: project.json is not valid JSON: ") + error.what());
    }
    if (!root.is_object())
    {
        fail("load: project.json must be a JSON object");
    }

    const std::string format = asString(member(root, "format", "load"), "load: format");
    if (format != kFormat)
    {
        fail("load: not a Sabercat project (format \"" + format + "\")");
    }
    const Json & versionJson = member(root, "formatVersion", "load");
    if (!versionJson.is_number_integer())
    {
        fail("load: formatVersion must be an integer");
    }
    const int64_t formatVersion = versionJson.get<int64_t>();
    if (formatVersion > kFormatVersion)
    {
        fail("load: project format version " + std::to_string(formatVersion)
             + " is newer than this Sabercat reads (" + std::to_string(kFormatVersion)
             + "); update Sabercat");
    }
    if (formatVersion < 1)
    {
        fail("load: formatVersion " + std::to_string(formatVersion) + " is invalid");
    }

    Project project;
    project.savedWithVersion = asString(
        member(member(root, "savedWith", "load"), "sabercat", "load: savedWith"),
        "load: savedWith.sabercat");

    const Json & layersJson = member(root, "layers", "load");
    if (!layersJson.is_array())
    {
        fail("load: layers must be an array");
    }

    // Layer metadata first, so the archive entries can be matched to layers.
    std::map<std::string, size_t> indexById;
    std::vector<bool> antialiasedFlags;
    for (size_t i = 0; i < layersJson.size(); ++i)
    {
        const Json & entry = layersJson[i];
        const std::string id = asString(member(entry, "id", "load: layer " + std::to_string(i)),
                                        "load: layer " + std::to_string(i) + " id");
        const std::string where = "load: layer '" + id + "'";
        if (!validId(id))
        {
            fail(where + ": id must be 1 to 64 characters of [A-Za-z0-9_-]");
        }
        if (!indexById.emplace(id, i).second)
        {
            fail(where + ": duplicate layer id");
        }

        ProjectLayer layer;
        layer.id = id;
        layer.name = asString(member(entry, "name", where), where + " name");
        layer.visible = asBool(member(entry, "visible", where), where + " visible");
        layer.locked = asBool(member(entry, "locked", where), where + " locked");
        antialiasedFlags.push_back(asBool(member(entry, "antialiased", where), where + " antialiased"));
        layer.transform = transformFromJson(member(entry, "transform", where), where);
        project.layers.push_back(std::move(layer));
    }

    // Split the archive entries by layer. Anything the format does not name is
    // skipped, so a newer writer's extra entries do not break this reader.
    std::vector<FileMap> cloudFiles(project.layers.size());
    std::vector<FileMap> relightFiles(project.layers.size());
    std::vector<std::string> unknown;
    const std::string layersPrefix = kLayersFolder;
    const std::string relightPrefix = kRelightFolder;
    for (auto & [name, bytes] : files)
    {
        if (name == kProjectEntry)
        {
            continue;
        }
        const size_t slash = name.find('/', layersPrefix.size());
        if (name.compare(0, layersPrefix.size(), layersPrefix) != 0 || slash == std::string::npos)
        {
            unknown.push_back(name);
            continue;
        }
        const std::string id = name.substr(layersPrefix.size(), slash - layersPrefix.size());
        const std::string inside = name.substr(slash + 1);
        const auto found = indexById.find(id);
        if (found == indexById.end() || inside.empty())
        {
            unknown.push_back(name);
            continue;
        }
        if (inside.compare(0, relightPrefix.size(), relightPrefix) == 0)
        {
            relightFiles[found->second][inside.substr(relightPrefix.size())] = std::move(bytes);
        }
        else
        {
            cloudFiles[found->second][inside] = std::move(bytes);
        }
    }
    if (!unknown.empty())
    {
        Logger::warning("ProjectIO::load: ignoring " + std::to_string(unknown.size())
                        + " entr" + (unknown.size() == 1 ? "y" : "ies")
                        + " the project does not name, e.g. \"" + unknown.front() + "\"");
    }

    // The clouds, in layer order.
    for (size_t i = 0; i < project.layers.size(); ++i)
    {
        ProjectLayer & layer = project.layers[i];
        const std::string where = "load: layer '" + layer.id + "'";
        if (cloudFiles[i].find(kManifestName) == cloudFiles[i].end())
        {
            fail(where + ": its files are missing (no " + kManifestName + " under " + kLayersFolder
                 + layer.id + "/)");
        }
        const std::string label = std::string(kLayersFolder) + layer.id;
        layer.cloud = CloudV1Reader::read(device, cloudFiles[i], label);
        if (!relightFiles[i].empty())
        {
            RelightSidecar::decode(relightFiles[i], *layer.cloud, label + "/" + kRelightFolder);
        }
        layer.cloud->setAntialiased(antialiasedFlags[i]);
        layer.cloud->computeBounds();
    }

    const Json & app = member(root, "app", "load");
    if (!app.is_object())
    {
        fail("load: app must be a JSON object");
    }
    project.appStateJson = app.dump();
    return project;
}

} // namespace sabercat
