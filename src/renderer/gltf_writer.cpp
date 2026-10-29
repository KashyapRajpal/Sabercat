// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/gltf_writer.h"

#include <glm/glm.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "sabercat/gaussian_cloud.h"
#include "sabercat/render_config.h"
#include "sabercat/version.h"
#include "unit_quaternion.h"

// The .glb bytes are little-endian by definition; floats are copied straight in.
static_assert(std::endian::native == std::endian::little,
              "GLTFWriter assumes a little-endian host (glTF data is little-endian)");

namespace sabercat
{

namespace
{

using Json = nlohmann::json;

// Must match what GltfReader accepts (gltf_reader.cpp).
constexpr const char * kExtensionName = "KHR_gaussian_splatting";
constexpr const char * kAttributePrefix = "KHR_gaussian_splatting:";
constexpr const char * kShAttributePrefix = "KHR_gaussian_splatting:SH_DEGREE_";
constexpr const char * kKernel = "ellipse";
constexpr const char * kColorSpace = "srgb_rec709_display";

// asset.generator. The version comes from project() in CMakeLists.txt.
constexpr const char * kGenerator = "Sabercat " SABERCAT_VERSION_STRING;

constexpr int kComponentFloat = 5126;
constexpr int kModePoints = 0;

constexpr uint32_t kGlbMagic = 0x46546C67;     // "glTF"
constexpr uint32_t kGlbVersion = 2;
constexpr uint32_t kGlbChunkJson = 0x4E4F534A; // "JSON"
constexpr uint32_t kGlbChunkBin = 0x004E4942;  // "BIN\0"

// The reader's SH degree ceiling; the canonical layout holds 16 coefficients.
constexpr uint32_t kMaxShDegree = 3;

std::string layerLabel(size_t index, const GLTFExportLayer & layer)
{
    return "layer " + std::to_string(index) + " ('" + layer.name + "')";
}

// Checks the host streams are sized and hold only what the reader will accept.
void validateLayer(const GLTFExportLayer & layer, size_t index)
{
    const std::string where = "GLTFWriter: " + layerLabel(index, layer);
    // The node rotation is written as given, so it gets the project's rule too.
    const glm::quat & placed = layer.transform.rotation;
    if (!isUnitLength(placed.w, placed.x, placed.y, placed.z))
    {
        throw std::runtime_error(where + ": transform rotation is not unit length (|q| = "
                                 + std::to_string(quaternionLength(placed.w, placed.x, placed.y, placed.z))
                                 + ")");
    }
    if (layer.cloud == nullptr)
    {
        throw std::runtime_error(where + " has no cloud");
    }
    const GaussianCloud & c = *layer.cloud;
    const size_t count = c.count();
    if (count < 1)
    {
        throw std::runtime_error(where + " has no Gaussians (glTF requires at least one)");
    }
    if (c.positions().size() != count || c.scales().size() != count
        || c.rotations().size() != count || c.opacities().size() != count
        || c.shCoefficients().size() != count * kShCoefficientsPerGaussian)
    {
        throw std::runtime_error(where + ": host streams are not sized for "
                                 + std::to_string(count) + " Gaussians");
    }
    for (size_t i = 0; i < count; ++i)
    {
        const glm::vec4 & p = c.positions()[i];
        if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z))
        {
            throw std::runtime_error(where + ": position " + std::to_string(i) + " is not finite");
        }
        const glm::vec4 & s = c.scales()[i];
        if (!std::isfinite(s.x) || !std::isfinite(s.y) || !std::isfinite(s.z) || s.x < 0.0f
            || s.y < 0.0f || s.z < 0.0f)
        {
            throw std::runtime_error(where + ": scale " + std::to_string(i)
                                     + " is not finite and non-negative");
        }
        const glm::vec4 & q = c.rotations()[i];
        if (!std::isfinite(q.x) || !std::isfinite(q.y) || !std::isfinite(q.z) || !std::isfinite(q.w))
        {
            throw std::runtime_error(where + ": Gaussian " + std::to_string(i)
                                     + " rotation is not finite");
        }
        if (!isUnitLength(q.w, q.x, q.y, q.z))
        {
            throw std::runtime_error(where + ": Gaussian " + std::to_string(i)
                                     + " rotation is not unit length (|q| = "
                                     + std::to_string(quaternionLength(q.w, q.x, q.y, q.z)) + ")");
        }
        const float opacity = c.opacities()[i];
        if (!(opacity >= 0.0f && opacity <= 1.0f))
        {
            throw std::runtime_error(where + ": opacity " + std::to_string(i) + " is "
                                     + std::to_string(opacity) + " (must be in [0, 1])");
        }
        for (size_t k = 0; k < kShCoefficientsPerGaussian; ++k)
        {
            const glm::vec4 & sh = c.shCoefficients()[i * kShCoefficientsPerGaussian + k];
            if (!std::isfinite(sh.x) || !std::isfinite(sh.y) || !std::isfinite(sh.z))
            {
                throw std::runtime_error(where + ": SH coefficient " + std::to_string(k)
                                         + " of Gaussian " + std::to_string(i)
                                         + " is not finite");
            }
        }
    }
}

// The SH attribute name for canonical coefficient k (band-ascending, m-ascending):
// k = 0 is SH_DEGREE_0_COEF_0; band l occupies k = l^2 .. (l+1)^2 - 1.
std::string shAttribute(uint32_t k)
{
    uint32_t band = 0;
    while ((band + 1) * (band + 1) <= k)
    {
        ++band;
    }
    const uint32_t m = k - band * band;
    return std::string(kShAttributePrefix) + std::to_string(band) + "_COEF_" + std::to_string(m);
}

// Growing BIN chunk plus the JSON arrays that describe it.
class BinaryChunk
{
  public:
    // Appends `values` as float32 and returns the accessor index. `minMax` adds the
    // min/max glTF requires for POSITION.
    size_t addFloatAccessor(const std::vector<float> & values, size_t count, int components,
                            const char * type, bool withBounds)
    {
        // Every view starts on a 4-byte boundary (it always does for float32 data;
        // the padding is kept explicit so the invariant does not depend on that).
        while (m_bytes.size() % 4 != 0)
        {
            m_bytes.push_back(0);
        }
        const size_t offset = m_bytes.size();
        const size_t byteLength = values.size() * sizeof(float);
        m_bytes.resize(offset + byteLength);
        std::memcpy(m_bytes.data() + offset, values.data(), byteLength);

        Json view = Json::object();
        view["buffer"] = 0;
        view["byteOffset"] = offset;
        view["byteLength"] = byteLength;
        m_bufferViews.push_back(view);

        Json accessor = Json::object();
        accessor["bufferView"] = m_bufferViews.size() - 1;
        accessor["byteOffset"] = 0;
        accessor["componentType"] = kComponentFloat;
        accessor["count"] = count;
        accessor["type"] = type;
        if (withBounds)
        {
            Json lo = Json::array();
            Json hi = Json::array();
            for (int c = 0; c < components; ++c)
            {
                float mn = std::numeric_limits<float>::max();
                float mx = std::numeric_limits<float>::lowest();
                for (size_t i = 0; i < count; ++i)
                {
                    mn = std::min(mn, values[i * components + c]);
                    mx = std::max(mx, values[i * components + c]);
                }
                lo.push_back(static_cast<double>(mn));
                hi.push_back(static_cast<double>(mx));
            }
            accessor["min"] = lo;
            accessor["max"] = hi;
        }
        m_accessors.push_back(accessor);
        return m_accessors.size() - 1;
    }

    const std::vector<uint8_t> & bytes() const
    {
        return m_bytes;
    }
    Json & bufferViews()
    {
        return m_bufferViews;
    }
    Json & accessors()
    {
        return m_accessors;
    }

  private:
    std::vector<uint8_t> m_bytes;
    Json m_bufferViews = Json::array();
    Json m_accessors = Json::array();
};

void appendU32(std::vector<uint8_t> & out, uint32_t value)
{
    for (int shift = 0; shift < 32; shift += 8)
    {
        out.push_back(static_cast<uint8_t>((value >> shift) & 0xFF));
    }
}

// The reader's JSON and BIN chunks: JSON padded with spaces, BIN with zeros, each
// to a 4-byte multiple (glTF 2.0 section 4.4).
std::vector<uint8_t> assembleGlb(std::string jsonText, std::vector<uint8_t> bin)
{
    while (jsonText.size() % 4 != 0)
    {
        jsonText.push_back(' ');
    }
    while (bin.size() % 4 != 0)
    {
        bin.push_back(0);
    }
    const size_t total = 12 + 8 + jsonText.size() + 8 + bin.size();

    std::vector<uint8_t> out;
    out.reserve(total);
    appendU32(out, kGlbMagic);
    appendU32(out, kGlbVersion);
    appendU32(out, static_cast<uint32_t>(total));
    appendU32(out, static_cast<uint32_t>(jsonText.size()));
    appendU32(out, kGlbChunkJson);
    out.insert(out.end(), jsonText.begin(), jsonText.end());
    appendU32(out, static_cast<uint32_t>(bin.size()));
    appendU32(out, kGlbChunkBin);
    out.insert(out.end(), bin.begin(), bin.end());
    return out;
}

} // namespace

void GLTFWriter::writeGLB(const std::vector<GLTFExportLayer> & layers, const std::string & path)
{
    if (layers.empty())
    {
        throw std::runtime_error("GLTFWriter: no layers to write to '" + path + "'");
    }
    for (size_t i = 0; i < layers.size(); ++i)
    {
        validateLayer(layers[i], i);
    }

    BinaryChunk buffer;
    Json meshes = Json::array();
    Json nodes = Json::array();
    Json nodeIndices = Json::array();

    for (size_t i = 0; i < layers.size(); ++i)
    {
        const GLTFExportLayer & layer = layers[i];
        const GaussianCloud & c = *layer.cloud;
        const size_t count = c.count();
        const uint32_t degree = std::min(c.shDegree(), kMaxShDegree);

        // Flat float streams, in the layouts the reader expects.
        std::vector<float> positions(count * 3);
        std::vector<float> rotations(count * 4);
        std::vector<float> scales(count * 3);
        std::vector<float> opacities(count);
        for (size_t g = 0; g < count; ++g)
        {
            const glm::vec4 & p = c.positions()[g];
            positions[g * 3 + 0] = p.x;
            positions[g * 3 + 1] = p.y;
            positions[g * 3 + 2] = p.z;
            const glm::vec4 & q = c.rotations()[g];
            rotations[g * 4 + 0] = q.x;
            rotations[g * 4 + 1] = q.y;
            rotations[g * 4 + 2] = q.z;
            rotations[g * 4 + 3] = q.w;
            const glm::vec4 & s = c.scales()[g];
            scales[g * 3 + 0] = s.x;
            scales[g * 3 + 1] = s.y;
            scales[g * 3 + 2] = s.z;
            opacities[g] = c.opacities()[g];
        }

        Json attributes = Json::object();
        attributes["POSITION"] = buffer.addFloatAccessor(positions, count, 3, "VEC3", true);
        attributes[std::string(kAttributePrefix) + "ROTATION"] =
            buffer.addFloatAccessor(rotations, count, 4, "VEC4", false);
        attributes[std::string(kAttributePrefix) + "SCALE"] =
            buffer.addFloatAccessor(scales, count, 3, "VEC3", false);
        attributes[std::string(kAttributePrefix) + "OPACITY"] =
            buffer.addFloatAccessor(opacities, count, 1, "SCALAR", false);

        // SH: one VEC3 accessor per coefficient up to the layer's degree.
        const uint32_t keptCoefficients = (degree + 1) * (degree + 1);
        for (uint32_t k = 0; k < keptCoefficients; ++k)
        {
            std::vector<float> coefficient(count * 3);
            for (size_t g = 0; g < count; ++g)
            {
                const glm::vec4 & sh = c.shCoefficients()[g * kShCoefficientsPerGaussian + k];
                coefficient[g * 3 + 0] = sh.x;
                coefficient[g * 3 + 1] = sh.y;
                coefficient[g * 3 + 2] = sh.z;
            }
            attributes[shAttribute(k)] = buffer.addFloatAccessor(coefficient, count, 3, "VEC3", false);
        }

        Json extension = Json::object();
        extension["kernel"] = kKernel;
        extension["colorSpace"] = kColorSpace;
        Json extensions = Json::object();
        extensions[kExtensionName] = extension;

        Json primitive = Json::object();
        primitive["mode"] = kModePoints;
        primitive["attributes"] = attributes;
        primitive["extensions"] = extensions;

        Json mesh = Json::object();
        mesh["name"] = layer.name;
        mesh["primitives"] = Json::array({primitive});
        meshes.push_back(mesh);

        // Node TRS = the layer transform, exactly as the reader reads it back.
        const LayerTransform & t = layer.transform;
        Json translation = Json::array();
        translation.push_back(static_cast<double>(t.translation.x));
        translation.push_back(static_cast<double>(t.translation.y));
        translation.push_back(static_cast<double>(t.translation.z));
        Json rotation = Json::array();
        rotation.push_back(static_cast<double>(t.rotation.x));
        rotation.push_back(static_cast<double>(t.rotation.y));
        rotation.push_back(static_cast<double>(t.rotation.z));
        rotation.push_back(static_cast<double>(t.rotation.w));
        Json scale = Json::array();
        for (int k = 0; k < 3; ++k)
        {
            scale.push_back(static_cast<double>(t.scale));
        }

        Json node = Json::object();
        node["name"] = layer.name;
        node["mesh"] = i;
        node["translation"] = translation;
        node["rotation"] = rotation;
        node["scale"] = scale;
        nodes.push_back(node);
        nodeIndices.push_back(i);
    }

    Json root = Json::object();
    Json asset = Json::object();
    asset["version"] = "2.0";
    asset["generator"] = kGenerator;
    root["asset"] = asset;
    root["extensionsUsed"] = Json::array({kExtensionName});
    Json scene = Json::object();
    scene["nodes"] = nodeIndices;
    root["scenes"] = Json::array({scene});
    root["scene"] = 0;
    root["nodes"] = nodes;
    root["meshes"] = meshes;
    root["accessors"] = buffer.accessors();
    root["bufferViews"] = buffer.bufferViews();
    // The declared length is the BIN chunk as assembleGlb() pads it.
    const size_t binLength = (buffer.bytes().size() + 3) / 4 * 4;
    Json buffers = Json::object();
    buffers["byteLength"] = binLength;
    root["buffers"] = Json::array({buffers});

    const std::vector<uint8_t> glb = assembleGlb(root.dump(), buffer.bytes());

    // Write beside the target and rename over it, so a failure never leaves a
    // truncated .glb at `path`.
    const std::filesystem::path target = std::filesystem::u8path(path);
    std::filesystem::path temp = target;
    temp += ".tmp";
    {
        std::ofstream stream(temp, std::ios::binary | std::ios::trunc);
        if (!stream.is_open())
        {
            throw std::runtime_error("GLTFWriter: cannot open '" + temp.string() + "' for writing");
        }
        stream.write(reinterpret_cast<const char *>(glb.data()),
                     static_cast<std::streamsize>(glb.size()));
        stream.close();
        if (stream.fail())
        {
            std::error_code ignored;
            std::filesystem::remove(temp, ignored);
            throw std::runtime_error("GLTFWriter: failed writing '" + temp.string() + "'");
        }
    }
    std::error_code error;
    std::filesystem::rename(temp, target, error);
    if (error)
    {
        std::error_code ignored;
        std::filesystem::remove(temp, ignored);
        throw std::runtime_error("GLTFWriter: cannot rename '" + temp.string() + "' to '" + path
                                 + "': " + error.message());
    }
}

} // namespace sabercat
