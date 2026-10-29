// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/cloud_v1_writer.h"

#include <glm/glm.hpp>
#include <nlohmann/json.hpp>

#include <bit>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sabercat/gaussian_cloud.h"
#include "sabercat/render_config.h"

// The streams are written as the host's float32 bytes, which are little-endian
// only on a little-endian host (the same assumption the reader makes).
static_assert(std::endian::native == std::endian::little,
              "the cloud-v1 writer assumes a little-endian host (the streams are float32 LE)");
static_assert(sizeof(glm::vec4) == 16, "glm::vec4 must match the 16 B canonical stride");

namespace sabercat
{

namespace
{

// Same literals as cloud_v1_reader.cpp: the reader is this writer's inverse.
constexpr const char * kCloudSchema = "sabercat.cloud-v1";
constexpr const char * kCoordinateSystem = "right-handed-y-up-camera-minus-z";
constexpr const char * kQuaternionOrder = "xyzw";
constexpr const char * kShLayout = "splat-major-coefficient-major-rgb";
constexpr const char * kScalarType = "float32-little-endian";
constexpr const char * kColorSpace = "linear-srgb";
constexpr const char * kManifestName = "manifest.json";
constexpr const char * kRelightSchema = "sabercat.relight-v1";

constexpr int kMaxShDegree = 3; // the canonical layout holds 16 coefficients

using Json = nlohmann::ordered_json;
using Bytes = std::vector<uint8_t>;

int coefficientsAboveDc(int degree)
{
    return (degree + 1) * (degree + 1) - 1;
}

template <typename T> Bytes bytesOf(const std::vector<T> & values)
{
    Bytes bytes(values.size() * sizeof(T));
    if (!bytes.empty())
    {
        std::memcpy(bytes.data(), values.data(), bytes.size());
    }
    return bytes;
}

// The bytes of a [N,4] stream. `zeroW` writes the pad lane as 0 (the reader
// zeroes it on the way in, too).
Bytes vec4Bytes(const std::vector<glm::vec4> & values, bool zeroW)
{
    if (!zeroW)
    {
        return bytesOf(values);
    }
    std::vector<glm::vec4> padded(values);
    for (glm::vec4 & value : padded)
    {
        value.w = 0.0f;
    }
    return bytesOf(padded);
}

void requireSize(const char * who, const char * what, size_t actual, size_t expected)
{
    if (actual != expected)
    {
        throw std::runtime_error(std::string(who) + ": " + what + " has " + std::to_string(actual)
                                 + " entries, expected " + std::to_string(expected));
    }
}

// One stream: its .bin file and its manifest entry (file, shape, scalarType,
// elementStrideBytes, byteLength).
void addStream(Json & streams, FileMap & files, const std::string & name,
               const std::vector<int64_t> & shape, Bytes bytes)
{
    const std::string file = name + ".f32le.bin";
    // elementStrideBytes is numpy's strides[0]: the bytes of one leading element.
    int64_t stride = static_cast<int64_t>(sizeof(float));
    for (size_t i = 1; i < shape.size(); ++i)
    {
        stride *= shape[i];
    }
    Json entry = Json::object();
    entry["file"] = file;
    entry["shape"] = shape;
    entry["scalarType"] = kScalarType;
    entry["elementStrideBytes"] = stride;
    entry["byteLength"] = bytes.size();
    streams[name] = entry;
    files[file] = std::move(bytes);
}

void addManifest(FileMap & files, const Json & manifest)
{
    const std::string text = manifest.dump(2) + "\n";
    files[kManifestName] = Bytes(text.begin(), text.end());
}

void writeFiles(const std::string & directory, const FileMap & files)
{
    const std::filesystem::path dir = std::filesystem::u8path(directory);
    std::filesystem::create_directories(dir);
    for (const auto & [name, bytes] : files)
    {
        const std::filesystem::path file = dir / std::filesystem::u8path(name);
        std::ofstream out(file, std::ios::binary | std::ios::trunc);
        if (!out.is_open())
        {
            throw std::runtime_error("CloudV1Writer: failed to create " + file.string());
        }
        out.write(reinterpret_cast<const char *>(bytes.data()),
                  static_cast<std::streamsize>(bytes.size()));
        out.close();
        if (out.fail())
        {
            throw std::runtime_error("CloudV1Writer: write failed for " + file.string());
        }
    }
}

} // namespace

FileMap CloudV1Writer::encode(const GaussianCloud & cloud)
{
    const GaussianCloud & source = cloud;

    const size_t n = source.count();
    requireSize("CloudV1Writer", "positions", source.positions().size(), n);
    requireSize("CloudV1Writer", "scales", source.scales().size(), n);
    requireSize("CloudV1Writer", "rotations", source.rotations().size(), n);
    requireSize("CloudV1Writer", "opacities", source.opacities().size(), n);
    requireSize("CloudV1Writer", "shCoefficients", source.shCoefficients().size(),
                n * kShCoefficientsPerGaussian);
    const int degree = static_cast<int>(source.shDegree());
    if (degree < 0 || degree > kMaxShDegree)
    {
        throw std::runtime_error("CloudV1Writer: shDegree " + std::to_string(degree)
                                 + " is outside 0.." + std::to_string(kMaxShDegree)
                                 + " (the canonical layout holds 16 coefficients)");
    }

    const int64_t count = static_cast<int64_t>(n);
    const auto & sh = source.shCoefficients();
    FileMap files;
    Json streams = Json::object();

    addStream(streams, files, "means", {count, 4}, vec4Bytes(source.positions(), true));
    addStream(streams, files, "scales", {count, 4}, vec4Bytes(source.scales(), true));
    addStream(streams, files, "rotations", {count, 4}, bytesOf(source.rotations()));
    addStream(streams, files, "opacities", {count}, bytesOf(source.opacities()));

    // SH is split: sh0 is the DC vec4 (coefficient 0), sh_rest is the packed RGB
    // of coefficients 1..K. Coefficients above the degree are not written.
    std::vector<glm::vec4> dc(n);
    for (size_t i = 0; i < n; ++i)
    {
        dc[i] = sh[i * kShCoefficientsPerGaussian];
    }
    addStream(streams, files, "sh0", {count, 4}, vec4Bytes(dc, true));
    if (degree > 0)
    {
        const int restCoefficients = coefficientsAboveDc(degree);
        std::vector<float> rest(n * static_cast<size_t>(restCoefficients) * 3);
        for (size_t i = 0; i < n; ++i)
        {
            for (int k = 0; k < restCoefficients; ++k)
            {
                const glm::vec4 & c = sh[i * kShCoefficientsPerGaussian + 1 + k];
                float * out = &rest[(i * static_cast<size_t>(restCoefficients) + k) * 3];
                out[0] = c.x;
                out[1] = c.y;
                out[2] = c.z;
            }
        }
        addStream(streams, files, "sh_rest", {count, restCoefficients, 3}, bytesOf(rest));
    }

    Json manifest = Json::object();
    manifest["schema"] = kCloudSchema;
    manifest["count"] = count;
    manifest["coordinateSystem"] = kCoordinateSystem;
    manifest["quaternionOrder"] = kQuaternionOrder;
    manifest["colorSpace"] = kColorSpace;
    manifest["shDegree"] = degree;
    manifest["shLayout"] = kShLayout;
    manifest["streams"] = streams;
    addManifest(files, manifest);
    return files;
}

void CloudV1Writer::write(const GaussianCloud & cloud, const std::string & directory)
{
    writeFiles(directory, encode(cloud));
}

FileMap RelightSidecar::encode(const GaussianCloud & cloud)
{
    const GaussianCloud & source = cloud;

    const size_t n = source.count();
    const int64_t count = static_cast<int64_t>(n);
    FileMap files;
    Json streams = Json::object();

    if (!source.normals().empty())
    {
        requireSize("RelightSidecar", "normals", source.normals().size(), n);
        addStream(streams, files, "normals", {count, 4}, bytesOf(source.normals()));
    }
    if (!source.baseColorRoughness().empty())
    {
        requireSize("RelightSidecar", "baseColorRoughness", source.baseColorRoughness().size(), n);
        addStream(streams, files, "base_color_roughness", {count, 4},
                  bytesOf(source.baseColorRoughness()));
    }
    if (!source.metallic().empty())
    {
        requireSize("RelightSidecar", "metallic", source.metallic().size(), n);
        addStream(streams, files, "metallic", {count}, bytesOf(source.metallic()));
    }
    if (!source.visibility().empty())
    {
        requireSize("RelightSidecar", "visibility", source.visibility().size(),
                    n * kVisibilityCoefficients);
        addStream(streams, files, "visibility", {count, kVisibilityCoefficients},
                  bytesOf(source.visibility()));
    }
    if (streams.empty())
    {
        return {};
    }

    Json manifest = Json::object();
    manifest["schema"] = kRelightSchema;
    manifest["count"] = count;
    manifest["streams"] = streams;
    addManifest(files, manifest);
    return files;
}

} // namespace sabercat
