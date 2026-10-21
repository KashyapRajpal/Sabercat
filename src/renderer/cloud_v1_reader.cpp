// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/cloud_v1_reader.h"

#include <glm/glm.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sabercat/cloud_v1_writer.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/logger.h"
#include "sabercat/parallel_for.h"
#include "sabercat/vk_check.h"

// The streams are little-endian float32 and are read straight into the
// canonical vectors.
static_assert(std::endian::native == std::endian::little,
              "CloudV1Reader assumes a little-endian host (the streams are float32 LE)");
static_assert(sizeof(glm::vec4) == 16, "glm::vec4 must match the 16 B canonical stride");

namespace sabercat
{

namespace
{

using Json = nlohmann::json;

// --- The manifest values this reader implements (the writer in
// cloud_v1_writer.cpp emits the same values). Anything else would change the
// meaning of the bytes, so it is rejected rather than reinterpreted. ---
constexpr const char * kSchema = "sabercat.cloud-v1";
// The schema id that files written before Sabercat 1.0 carry. Still read,
// never written.
constexpr const char * kLegacySchema = "gaussmith.sabercat-cloud-v1";
constexpr const char * kCoordinateSystem = "right-handed-y-up-camera-minus-z";
constexpr const char * kQuaternionOrder = "xyzw";
constexpr const char * kShLayout = "splat-major-coefficient-major-rgb";
constexpr const char * kScalarType = "float32-little-endian";
constexpr const char * kColorSpace = "linear-srgb";
constexpr const char * kManifestName = "manifest.json";
// The relight sidecar (RelightSidecar in cloud_v1_writer.h).
constexpr const char * kRelightSchema = "sabercat.relight-v1";

constexpr int kMaxShDegree = 3; // the canonical layout holds 16 coefficients
constexpr int64_t kMaxCount = int64_t{1} << 28;

// Same rule as the glTF reader: a canonical (already unit) quaternion passes
// through bit-exact; anything visibly off the unit sphere is renormalized.
constexpr float kUnitQuaternionTolerance = 1e-6f;
constexpr float kMinQuaternionLengthSquared = 1e-12f;

[[noreturn]] void fail(const std::string & label, const std::string & what)
{
    throw std::runtime_error("CloudV1Reader: " + what + ": " + label);
}

int coefficientsAboveDc(int degree)
{
    return (degree + 1) * (degree + 1) - 1;
}

void requireString(const Json & manifest, const char * key, const char * expected,
                   const std::string & label)
{
    if (!manifest.contains(key))
    {
        fail(label, std::string("manifest is missing '") + key + "'");
    }
    const std::string value = manifest.at(key).get<std::string>();
    if (value != expected)
    {
        fail(label,
             std::string("manifest '") + key + "' is '" + value + "'; this reader implements '"
                 + expected + "' only");
    }
}

// Where the stream files come from: a directory on disk (CloudV1Reader::read by
// path) or a FileMap (CloudV1Reader::read by files, RelightSidecar::decode).
class StreamSource
{
  public:
    virtual ~StreamSource() = default;

    // Copies exactly `byteLength` bytes of the stream file `file` into
    // `destination`. `where` names the stream for error messages.
    virtual void readFile(const std::string & where, const std::string & file, int64_t byteLength,
                          void * destination) const = 0;
};

class DirectorySource final : public StreamSource
{
  public:
    DirectorySource(std::filesystem::path directory, const std::string & label)
        : m_directory(std::move(directory))
        , m_label(label)
    {
    }

    void readFile(const std::string & where, const std::string & file, int64_t byteLength,
                  void * destination) const override
    {
        const std::filesystem::path relative = std::filesystem::u8path(file);
        if (relative.is_absolute() || relative.has_root_name())
        {
            fail(m_label,
                 where + " names an absolute file; stream files are relative to the manifest");
        }
        const std::filesystem::path path = m_directory / relative;
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        if (!stream.is_open())
        {
            fail(m_label, where + ": failed to open '" + path.string() + "'");
        }
        const std::streamsize size = stream.tellg();
        if (size != byteLength)
        {
            fail(m_label,
                 where + ": '" + path.string() + "' is " + std::to_string(size)
                     + " bytes, but the manifest says " + std::to_string(byteLength));
        }
        stream.seekg(0, std::ios::beg);
        if (!stream.read(static_cast<char *>(destination), byteLength))
        {
            fail(m_label, where + ": truncated read of '" + path.string() + "'");
        }
    }

  private:
    std::filesystem::path m_directory;
    std::string m_label;
};

class FileMapSource final : public StreamSource
{
  public:
    FileMapSource(const FileMap & files, const std::string & label)
        : m_files(files)
        , m_label(label)
    {
    }

    void readFile(const std::string & where, const std::string & file, int64_t byteLength,
                  void * destination) const override
    {
        const auto it = m_files.find(file);
        if (it == m_files.end())
        {
            fail(m_label, where + ": no file '" + file + "' in the FileMap");
        }
        if (static_cast<int64_t>(it->second.size()) != byteLength)
        {
            fail(m_label,
                 where + ": '" + file + "' is " + std::to_string(it->second.size())
                     + " bytes, but the manifest says " + std::to_string(byteLength));
        }
        if (byteLength > 0)
        {
            std::memcpy(destination, it->second.data(), static_cast<size_t>(byteLength));
        }
    }

  private:
    const FileMap & m_files;
    std::string m_label;
};

struct StreamReader
{
    const Json & streams;
    const StreamSource & source;
    const std::string & label;

    // Checks one stream entry against the shape this reader expects and reads
    // exactly byteLength bytes of it into `destination`.
    void read(const char * name, const std::vector<int64_t> & shape, void * destination) const
    {
        if (!streams.contains(name))
        {
            fail(label, std::string("missing required stream '") + name + "'");
        }
        const Json & entry = streams.at(name);
        const std::string where = std::string("stream '") + name + "'";

        const std::string scalarType = entry.at("scalarType").get<std::string>();
        if (scalarType != kScalarType)
        {
            fail(label,
                 where + " has scalarType '" + scalarType + "' (expected '" + kScalarType + "')");
        }
        const std::vector<int64_t> actualShape = entry.at("shape").get<std::vector<int64_t>>();
        if (actualShape != shape)
        {
            auto describe = [](const std::vector<int64_t> & s)
            {
                std::string text = "[";
                for (size_t i = 0; i < s.size(); ++i)
                {
                    text += (i ? ", " : "") + std::to_string(s[i]);
                }
                return text + "]";
            };
            fail(label,
                 where + " has shape " + describe(actualShape) + ", expected " + describe(shape));
        }
        int64_t elementFloats = 1;
        for (size_t i = 1; i < shape.size(); ++i)
        {
            elementFloats *= shape[i];
        }
        const int64_t stride = elementFloats * static_cast<int64_t>(sizeof(float));
        const int64_t byteLength = shape[0] * stride;
        if (entry.at("elementStrideBytes").get<int64_t>() != stride)
        {
            fail(label,
                 where + " has elementStrideBytes "
                     + std::to_string(entry.at("elementStrideBytes").get<int64_t>()) + ", expected "
                     + std::to_string(stride));
        }
        if (entry.at("byteLength").get<int64_t>() != byteLength)
        {
            fail(label,
                 where + " has byteLength " + std::to_string(entry.at("byteLength").get<int64_t>())
                     + ", expected " + std::to_string(byteLength));
        }

        source.readFile(where, entry.at("file").get<std::string>(), byteLength, destination);
    }
};

std::shared_ptr<GaussianCloud> readManifest(Device & device, const Json & manifest,
                                            const StreamSource & source, const std::string & label)
{
    if (!manifest.contains("schema"))
    {
        fail(label, "manifest is missing 'schema'");
    }
    const std::string schema = manifest.at("schema").get<std::string>();
    if (schema != kSchema && schema != kLegacySchema)
    {
        fail(label, "manifest 'schema' is '" + schema + "'; this reader implements '" + kSchema
                        + "' and the legacy '" + kLegacySchema + "' only");
    }
    requireString(manifest, "coordinateSystem", kCoordinateSystem, label);
    requireString(manifest, "quaternionOrder", kQuaternionOrder, label);
    requireString(manifest, "shLayout", kShLayout, label);

    // Colour space is a label on the SH values, not a byte layout: nothing to
    // convert (a transfer function cannot be applied to SH coefficients), so
    // an unexpected one is reported, not rejected.
    const std::string colorSpace = manifest.value("colorSpace", std::string());
    if (colorSpace != kColorSpace)
    {
        Logger::warning("CloudV1Reader: colorSpace '" + colorSpace + "' (expected '" + kColorSpace
                        + "'); SH passed through unconverted: " + label);
    }

    const int64_t count = manifest.at("count").get<int64_t>();
    if (count < 1 || count > kMaxCount)
    {
        fail(label, "manifest count " + std::to_string(count) + " is out of range");
    }
    const int64_t degree = manifest.at("shDegree").get<int64_t>();
    if (degree < 0)
    {
        fail(label, "manifest shDegree " + std::to_string(degree) + " is negative");
    }
    // Truncate rather than reject, like SpzReader: a degree cap is a correct
    // band-ascending prefix read.
    const int usedDegree = static_cast<int>(std::min<int64_t>(degree, kMaxShDegree));
    if (degree > kMaxShDegree)
    {
        Logger::warning("CloudV1Reader: shDegree " + std::to_string(degree)
                        + " truncated to 3 (the canonical layout holds 16 coefficients): " + label);
    }

    const Json & streams = manifest.at("streams");
    for (const auto & item : streams.items())
    {
        const std::string & name = item.key();
        if (name != "means" && name != "scales" && name != "rotations" && name != "opacities"
            && name != "sh0" && name != "sh_rest" && name != "confidence")
        {
            Logger::warning("CloudV1Reader: unknown stream '" + name + "' ignored: " + label);
        }
    }
    if (degree == 0 && streams.contains("sh_rest"))
    {
        fail(label, "stream 'sh_rest' is present but shDegree is 0");
    }

    const size_t n = static_cast<size_t>(count);
    auto cloud = std::make_shared<GaussianCloud>(device);
    cloud->initialize(n);
    auto & positions = cloud->positions();
    auto & scales = cloud->scales();
    auto & rotations = cloud->rotations();
    auto & opacities = cloud->opacities();
    auto & sh = cloud->shCoefficients();

    const StreamReader reader{streams, source, label};
    // The four canonical-stride streams land directly in the cloud: same
    // 16 B / 4 B strides as the GPU buffers, so no conversion at all.
    reader.read("means", {count, 4}, positions.data());
    reader.read("scales", {count, 4}, scales.data());
    reader.read("rotations", {count, 4}, rotations.data());
    reader.read("opacities", {count}, opacities.data());

    // SH is the exception: sh0 is padded to 16 B, sh_rest is packed RGB at
    // 12 B per coefficient, and the canonical stream interleaves both at
    // 16 B per coefficient per Gaussian.
    std::vector<float> sh0(n * 4);
    reader.read("sh0", {count, 4}, sh0.data());
    const int restCoefficients = coefficientsAboveDc(static_cast<int>(degree));
    std::vector<float> shRest;
    if (degree > 0)
    {
        shRest.resize(n * static_cast<size_t>(restCoefficients) * 3);
        reader.read("sh_rest", {count, restCoefficients, 3}, shRest.data());
    }
    if (streams.contains("confidence"))
    {
        Logger::info("CloudV1Reader: 'confidence' stream not loaded (GaussianCloud has no "
                     "confidence stream): "
                     + label);
    }

    // --- Validate the canonical promises before anything renders them ---
    for (size_t i = 0; i < n; ++i)
    {
        const glm::vec4 & p = positions[i];
        const glm::vec4 & s = scales[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
        {
            fail(label, "mean " + std::to_string(i) + " is not finite");
        }
        if (!(s.x > 0.0f && s.y > 0.0f && s.z > 0.0f) || !std::isfinite(s.x) || !std::isfinite(s.y)
            || !std::isfinite(s.z))
        {
            fail(label, "scale " + std::to_string(i) + " is not a positive finite linear std-dev");
        }
        if (!(opacities[i] >= 0.0f && opacities[i] <= 1.0f))
        {
            fail(label,
                 "opacity " + std::to_string(i) + " is " + std::to_string(opacities[i])
                     + " (must be in [0, 1])");
        }
        glm::vec4 & q = rotations[i];
        const float lengthSquared = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
        if (!std::isfinite(lengthSquared) || lengthSquared < kMinQuaternionLengthSquared)
        {
            fail(label, "rotation " + std::to_string(i) + " is zero-length or not finite");
        }
        if (std::abs(lengthSquared - 1.0f) > kUnitQuaternionTolerance)
        {
            q *= 1.0f / std::sqrt(lengthSquared);
        }
        // Pad lanes are padding, not data: zero them so a writer that left
        // garbage there cannot leak it into the canonical streams.
        positions[i].w = 0.0f;
        scales[i].w = 0.0f;
    }
    for (const float value : sh0)
    {
        if (!std::isfinite(value))
        {
            fail(label, "stream 'sh0' contains a non-finite value");
        }
    }
    for (const float value : shRest)
    {
        if (!std::isfinite(value))
        {
            fail(label, "stream 'sh_rest' contains a non-finite value");
        }
    }

    const int usedRest = coefficientsAboveDc(usedDegree);
    parallelFor(
        n,
        [&](size_t begin, size_t end)
        {
            for (size_t i = begin; i < end; ++i)
            {
                glm::vec4 * out = &sh[i * kShCoefficientsPerGaussian];
                out[0] = glm::vec4(sh0[i * 4 + 0], sh0[i * 4 + 1], sh0[i * 4 + 2], 0.0f);
                // [N, K, 3]: splat-major, coefficient-major, RGB inner.
                const float * rest = shRest.data() + i * static_cast<size_t>(restCoefficients) * 3;
                for (int k = 0; k < usedRest; ++k)
                {
                    out[1 + k] = glm::vec4(rest[k * 3 + 0], rest[k * 3 + 1], rest[k * 3 + 2], 0.0f);
                }
            }
        });

    // Not in the manifest: cloud-v1 has no anti-aliasing flag, so the cloud
    // renders as vanilla 3DGS (no dilation compensation).
    cloud->setAntialiased(false);
    cloud->setShDegree(static_cast<uint32_t>(usedDegree));
    cloud->computeBounds();
    return cloud;
}

// Parses the manifest text and reads the fixture it describes. Shared by the
// path and FileMap entry points, so every check lives in one place.
std::shared_ptr<GaussianCloud> readFixture(Device & device, const std::string & manifestText,
                                           const StreamSource & source, const std::string & label)
{
    try
    {
        const Json manifest = Json::parse(manifestText);
        return readManifest(device, manifest, source, label);
    }
    catch (const Json::exception & e)
    {
        fail(label, std::string("malformed manifest: ") + e.what());
    }
}

} // namespace

std::shared_ptr<GaussianCloud> CloudV1Reader::read(Device & device, const std::string & path)
{
    std::filesystem::path manifestPath = std::filesystem::u8path(path);
    if (std::filesystem::is_directory(manifestPath))
    {
        manifestPath /= kManifestName;
    }
    std::ifstream stream(manifestPath, std::ios::binary);
    if (!stream.is_open())
    {
        fail(path, "failed to open '" + manifestPath.string() + "'");
    }
    const std::string text((std::istreambuf_iterator<char>(stream)),
                           std::istreambuf_iterator<char>());
    return readFixture(device, text, DirectorySource(manifestPath.parent_path(), path), path);
}

std::shared_ptr<GaussianCloud> CloudV1Reader::read(Device & device, const FileMap & files,
                                                   const std::string & label)
{
    const auto it = files.find(kManifestName);
    if (it == files.end())
    {
        fail(label, "FileMap has no 'manifest.json'");
    }
    const std::string text(it->second.begin(), it->second.end());
    return readFixture(device, text, FileMapSource(files, label), label);
}

void RelightSidecar::decode(const FileMap & files, GaussianCloud & cloud, const std::string & label)
{
    const auto it = files.find(kManifestName);
    if (it == files.end())
    {
        fail(label, "relight sidecar has no 'manifest.json'");
    }
    const std::string text(it->second.begin(), it->second.end());
    const FileMapSource source(files, label);
    const size_t n = cloud.count();
    const int64_t count = static_cast<int64_t>(n);

    // Everything is read into locals and committed at the end, so a rejected
    // sidecar leaves the cloud untouched.
    std::vector<glm::vec4> normals;
    std::vector<glm::vec4> baseColorRoughness;
    std::vector<float> metallic;
    std::vector<float> visibility;
    try
    {
        const Json manifest = Json::parse(text);
        requireString(manifest, "schema", kRelightSchema, label);
        const int64_t sidecarCount = manifest.at("count").get<int64_t>();
        if (sidecarCount != count)
        {
            fail(label,
                 "relight sidecar count " + std::to_string(sidecarCount)
                     + " does not match the cloud's " + std::to_string(count) + " Gaussians");
        }
        if (!manifest.contains("streams"))
        {
            fail(label, "manifest is missing 'streams'");
        }
        const Json & streams = manifest.at("streams");
        for (const auto & item : streams.items())
        {
            const std::string & name = item.key();
            if (name != "normals" && name != "base_color_roughness" && name != "metallic"
                && name != "visibility")
            {
                Logger::warning("CloudV1Reader: unknown relight stream '" + name
                                + "' ignored: " + label);
            }
        }

        const StreamReader reader{streams, source, label};
        if (streams.contains("normals"))
        {
            normals.resize(n);
            reader.read("normals", {count, 4}, normals.data());
        }
        if (streams.contains("base_color_roughness"))
        {
            baseColorRoughness.resize(n);
            reader.read("base_color_roughness", {count, 4}, baseColorRoughness.data());
        }
        if (streams.contains("metallic"))
        {
            metallic.resize(n);
            reader.read("metallic", {count}, metallic.data());
        }
        if (streams.contains("visibility"))
        {
            visibility.resize(n * kVisibilityCoefficients);
            reader.read("visibility", {count, kVisibilityCoefficients}, visibility.data());
        }
    }
    catch (const Json::exception & e)
    {
        fail(label, std::string("malformed relight sidecar manifest: ") + e.what());
    }

    cloud.normals() = std::move(normals);
    cloud.baseColorRoughness() = std::move(baseColorRoughness);
    cloud.metallic() = std::move(metallic);
    cloud.visibility() = std::move(visibility);
}

} // namespace sabercat
