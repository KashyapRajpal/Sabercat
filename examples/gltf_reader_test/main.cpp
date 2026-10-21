// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Headless test for the glTF KHR_gaussian_splatting loader (GltfReader + the
// loadScene() dispatch).
//
// Every fixture is synthesized AT RUNTIME by a small glTF writer in this file
// whose JSON is built by hand -- deliberately NOT through the JSON library the
// reader uses -- into the system temp directory. Four parts:
//
//  (a) decode: one scene with five splat primitives that together cover every
//      component type the spec allows (normalized and not), byteStride
//      (interleaved and padded), sparse accessors (with and without a base
//      bufferView, all three index types), SH degrees 0-3, and the three
//      containers (.glb, .gltf + data: URI, .gltf + external percent-encoded
//      .bin). Expected values are recomputed here from the glTF spec's decode
//      rules; float data must round-trip bit-exactly.
//  (b) node transforms: a hand-derived band-1 SH rotation, TRS hierarchy vs
//      the equivalent `matrix` node, non-uniform scale (covariance checked as
//      a matrix), and a degree-3 SH rotation checked against an independent
//      basis evaluation.
//  (c) rejections: each malformed file must fail with a message naming the
//      problem.
//  (d) rendering through sabercat::Renderer, headless: a .ply and the .glb
//      converted from it must render BIT-IDENTICALLY (synthetic degree-3
//      scene always; assets/bonsai-7k.ply when present, or a path given as
//      argv[1]); and a node transform must render the same as moving the
//      camera by its inverse -- the physical meaning of the SH rotation,
//      positions and covariances, checked through the real shaders.
//  (e) KHR_gaussian_splatting_compression_spz_2: a cloud packed in memory by
//      spz (in the extension's LUF frame) and wrapped in a .glb must decode
//      to the source values within spz's quantization bounds, load under a
//      node bit-identically to the same data uncompressed, and reject a
//      missing bufferView, a corrupt stream and a count mismatch.
//
// Validation messages are logged as "[validation]"; the run must have none.

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <load-spz.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "sabercat/camera.h"
#include "sabercat/device.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/render_config.h"
#include "sabercat/renderer.h"
#include "sabercat/scene_loader.h"

using namespace sabercat;

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

std::filesystem::path g_tempDir;

std::string tempPath(const std::string & name)
{
    return (g_tempDir / name).string();
}

// ---------------------------------------------------------------------------
// glTF component types and the spec's decode rules, restated independently
// ---------------------------------------------------------------------------

constexpr int kByte = 5120;
constexpr int kUByte = 5121;
constexpr int kShort = 5122;
constexpr int kUShort = 5123;
constexpr int kUInt = 5125;
constexpr int kFloat = 5126;

size_t sizeOf(int componentType)
{
    return componentType == kByte || componentType == kUByte     ? 1
           : componentType == kShort || componentType == kUShort ? 2
                                                                 : 4;
}

// glTF 2.0: f = max(c / 127.0, -1.0) etc. for normalized; the integer itself
// otherwise.
float expectedDecode(int componentType, bool normalized, double raw)
{
    switch (componentType)
    {
    case kFloat:
        return static_cast<float>(raw);
    case kByte:
        return normalized ? std::max(static_cast<float>(raw) / 127.0f, -1.0f)
                          : static_cast<float>(raw);
    case kUByte:
        return normalized ? static_cast<float>(raw) / 255.0f : static_cast<float>(raw);
    case kShort:
        return normalized ? std::max(static_cast<float>(raw) / 32767.0f, -1.0f)
                          : static_cast<float>(raw);
    case kUShort:
        return normalized ? static_cast<float>(raw) / 65535.0f : static_cast<float>(raw);
    default:
        return static_cast<float>(raw);
    }
}

void appendComponent(std::vector<uint8_t> & out, int componentType, double raw)
{
    uint8_t bytes[4] = {};
    switch (componentType)
    {
    case kFloat:
    {
        const float v = static_cast<float>(raw);
        std::memcpy(bytes, &v, 4);
        break;
    }
    case kByte:
    {
        const int8_t v = static_cast<int8_t>(raw);
        std::memcpy(bytes, &v, 1);
        break;
    }
    case kUByte:
    {
        const uint8_t v = static_cast<uint8_t>(raw);
        std::memcpy(bytes, &v, 1);
        break;
    }
    case kShort:
    {
        const int16_t v = static_cast<int16_t>(raw);
        std::memcpy(bytes, &v, 2);
        break;
    }
    case kUShort:
    {
        const uint16_t v = static_cast<uint16_t>(raw);
        std::memcpy(bytes, &v, 2);
        break;
    }
    default:
    {
        const uint32_t v = static_cast<uint32_t>(raw);
        std::memcpy(bytes, &v, 4);
        break;
    }
    }
    out.insert(out.end(), bytes, bytes + sizeOf(componentType));
}

std::string num(double value)
{
    std::ostringstream s;
    s << std::setprecision(17) << value;
    return s.str();
}

// ---------------------------------------------------------------------------
// A minimal glTF writer: accumulates one binary buffer plus JSON fragments
// ---------------------------------------------------------------------------

constexpr const char * kAttr = "KHR_gaussian_splatting:";
const std::string kDefaultExtension = R"({"kernel":"ellipse","colorSpace":"srgb_rec709_display"})";

struct GltfWriter
{
    std::vector<uint8_t> bin;
    std::vector<std::string> views;
    std::vector<std::string> accessors;
    std::vector<std::string> meshes;
    std::vector<std::string> nodes;
    std::string sceneNodes; // e.g. "[0,1]"; empty = no "scenes"
    std::string extensionsRequired = R"(["KHR_gaussian_splatting"])";
    std::string extraTopLevel; // raw ",\"key\":..." appended to the root object

    void align()
    {
        while (bin.size() % 4 != 0)
        {
            bin.push_back(0);
        }
    }

    int view(const std::vector<uint8_t> & bytes, int stride = 0)
    {
        align();
        const size_t offset = bin.size();
        bin.insert(bin.end(), bytes.begin(), bytes.end());
        std::string v = "{\"buffer\":0,\"byteOffset\":" + std::to_string(offset)
                        + ",\"byteLength\":" + std::to_string(bytes.size());
        if (stride != 0)
        {
            v += ",\"byteStride\":" + std::to_string(stride);
        }
        views.push_back(v + "}");
        return static_cast<int>(views.size() - 1);
    }

    int accessor(int viewIndex, int componentType, bool normalized, size_t count, const char * type,
                 size_t byteOffset = 0, const std::string & extra = "")
    {
        std::string a = "{";
        if (viewIndex >= 0)
        {
            a += "\"bufferView\":" + std::to_string(viewIndex)
                 + ",\"byteOffset\":" + std::to_string(byteOffset) + ",";
        }
        a += "\"componentType\":" + std::to_string(componentType);
        if (normalized)
        {
            a += ",\"normalized\":true";
        }
        a += ",\"count\":" + std::to_string(count) + ",\"type\":\"" + type + "\"" + extra + "}";
        accessors.push_back(a);
        return static_cast<int>(accessors.size() - 1);
    }

    // Packs `raw` (count * components values, already in the stored integer
    // or float domain) as one accessor, optionally padded to `stride` bytes.
    int attribute(int componentType, bool normalized, int components,
                  const std::vector<double> & raw, int stride = 0, const std::string & extra = "")
    {
        const size_t count = raw.size() / static_cast<size_t>(components);
        const size_t element = sizeOf(componentType) * static_cast<size_t>(components);
        const size_t step = stride != 0 ? static_cast<size_t>(stride) : element;
        std::vector<uint8_t> bytes;
        for (size_t i = 0; i < count; ++i)
        {
            for (int c = 0; c < components; ++c)
            {
                appendComponent(bytes, componentType, raw[i * components + c]);
            }
            while (bytes.size() < (i + 1) * step && i + 1 < count)
            {
                bytes.push_back(0xCD); // stride padding: garbage the reader must skip
            }
        }
        const char * type = components == 1 ? "SCALAR" : components == 3 ? "VEC3" : "VEC4";
        return accessor(view(bytes, stride), componentType, normalized, count, type, 0, extra);
    }

    int addMesh(const std::vector<std::string> & primitives)
    {
        std::string m = "{\"primitives\":[";
        for (size_t i = 0; i < primitives.size(); ++i)
        {
            m += (i ? "," : "") + primitives[i];
        }
        meshes.push_back(m + "]}");
        return static_cast<int>(meshes.size() - 1);
    }

    int addNode(const std::string & body)
    {
        nodes.push_back("{" + body + "}");
        return static_cast<int>(nodes.size() - 1);
    }

    std::string json(const std::string & bufferUri) const
    {
        auto list = [](const std::vector<std::string> & items)
        {
            std::string s = "[";
            for (size_t i = 0; i < items.size(); ++i)
            {
                s += (i ? "," : "") + items[i];
            }
            return s + "]";
        };
        std::string j = R"({"asset":{"version":"2.0","generator":"sabercat gltf_reader_test"},)";
        j += R"("extensionsUsed":["KHR_gaussian_splatting","KHR_mesh_quantization"],)";
        if (!extensionsRequired.empty())
        {
            j += "\"extensionsRequired\":" + extensionsRequired + ",";
        }
        if (!sceneNodes.empty())
        {
            j += "\"scene\":0,\"scenes\":[{\"nodes\":" + sceneNodes + "}],";
        }
        if (!nodes.empty())
        {
            j += "\"nodes\":" + list(nodes) + ",";
        }
        j += "\"meshes\":" + list(meshes) + ",";
        j += "\"accessors\":" + list(accessors) + ",";
        j += "\"bufferViews\":" + list(views) + ",";
        j += "\"buffers\":[{" + bufferUri + "\"byteLength\":" + std::to_string(bin.size()) + "}]";
        j += extraTopLevel + "}";
        return j;
    }

    void writeGlb(const std::string & path) const
    {
        std::string text = json("");
        while (text.size() % 4 != 0)
        {
            text.push_back(' ');
        }
        std::vector<uint8_t> payload = bin;
        while (payload.size() % 4 != 0)
        {
            payload.push_back(0);
        }
        const uint32_t total = static_cast<uint32_t>(12 + 8 + text.size() + 8 + payload.size());
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        auto u32 = [&](uint32_t v)
        {
            file.write(reinterpret_cast<const char *>(&v), 4);
        };
        u32(0x46546C67);
        u32(2);
        u32(total);
        u32(static_cast<uint32_t>(text.size()));
        u32(0x4E4F534A);
        file.write(text.data(), static_cast<std::streamsize>(text.size()));
        u32(static_cast<uint32_t>(payload.size()));
        u32(0x004E4942);
        file.write(reinterpret_cast<const char *>(payload.data()),
                   static_cast<std::streamsize>(payload.size()));
    }

    // A GLB whose buffer says "uri": null (some exporters write it). Strictly
    // invalid glTF, but unambiguous: the GLB BIN chunk is the buffer.
    void writeGlbNullUri(const std::string & path) const
    {
        std::string text = json("\"uri\":null,");
        while (text.size() % 4 != 0)
        {
            text.push_back(' ');
        }
        std::vector<uint8_t> payload = bin;
        while (payload.size() % 4 != 0)
        {
            payload.push_back(0);
        }
        const uint32_t total = static_cast<uint32_t>(12 + 8 + text.size() + 8 + payload.size());
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        auto u32 = [&](uint32_t v)
        {
            file.write(reinterpret_cast<const char *>(&v), 4);
        };
        u32(0x46546C67);
        u32(2);
        u32(total);
        u32(static_cast<uint32_t>(text.size()));
        u32(0x4E4F534A);
        file.write(text.data(), static_cast<std::streamsize>(text.size()));
        u32(static_cast<uint32_t>(payload.size()));
        u32(0x004E4942);
        file.write(reinterpret_cast<const char *>(payload.data()),
                   static_cast<std::streamsize>(payload.size()));
    }

    void writeGltfDataUri(const std::string & path) const
    {
        static const char * kAlphabet =
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        std::string encoded;
        for (size_t i = 0; i < bin.size(); i += 3)
        {
            const uint32_t b0 = bin[i];
            const uint32_t b1 = i + 1 < bin.size() ? bin[i + 1] : 0;
            const uint32_t b2 = i + 2 < bin.size() ? bin[i + 2] : 0;
            const uint32_t triple = (b0 << 16) | (b1 << 8) | b2;
            encoded += kAlphabet[(triple >> 18) & 63];
            encoded += kAlphabet[(triple >> 12) & 63];
            encoded += i + 1 < bin.size() ? kAlphabet[(triple >> 6) & 63] : '=';
            encoded += i + 2 < bin.size() ? kAlphabet[triple & 63] : '=';
        }
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        file << json("\"uri\":\"data:application/octet-stream;base64," + encoded + "\",");
    }

    // External buffer with a space in its name, referenced percent-encoded.
    void writeGltfExternal(const std::string & path, const std::string & binName,
                           const std::string & binUri) const
    {
        {
            std::ofstream binFile((std::filesystem::path(path).parent_path() / binName),
                                  std::ios::binary | std::ios::trunc);
            binFile.write(reinterpret_cast<const char *>(bin.data()),
                          static_cast<std::streamsize>(bin.size()));
        }
        std::ofstream file(path, std::ios::binary | std::ios::trunc);
        file << json("\"uri\":\"" + binUri + "\",");
    }
};

std::string primitive(const std::vector<std::pair<std::string, int>> & attributes,
                      const std::string & extension = kDefaultExtension, int mode = 0)
{
    std::string p = "{\"attributes\":{";
    for (size_t i = 0; i < attributes.size(); ++i)
    {
        p += (i ? "," : "")
             + ("\"" + attributes[i].first + "\":" + std::to_string(attributes[i].second));
    }
    p += "}";
    if (mode >= 0)
    {
        p += ",\"mode\":" + std::to_string(mode);
    }
    if (!extension.empty())
    {
        p += ",\"extensions\":{\"KHR_gaussian_splatting\":" + extension + "}";
    }
    return p + "}";
}

std::string shName(int degree, int coefficient)
{
    return std::string(kAttr) + "SH_DEGREE_" + std::to_string(degree) + "_COEF_"
           + std::to_string(coefficient);
}

int coefficientCount(int degree)
{
    return (degree + 1) * (degree + 1);
}

// ---------------------------------------------------------------------------
// Expected canonical values
// ---------------------------------------------------------------------------

struct Expected
{
    std::vector<glm::vec4> positions, scales, rotations, sh;
    std::vector<float> opacities;

    size_t count() const
    {
        return opacities.size();
    }
};

glm::vec4 normalizedQuaternion(float x, float y, float z, float w, bool force)
{
    // Same rule the spec implies for a reader: quantized always renormalized,
    // floats only when off-unit.
    const float lengthSquared = x * x + y * y + z * z + w * w;
    if (!force && std::abs(lengthSquared - 1.0f) <= 1e-6f)
    {
        return glm::vec4(x, y, z, w);
    }
    const float inv = 1.0f / std::sqrt(lengthSquared);
    return glm::vec4(x * inv, y * inv, z * inv, w * inv);
}

// SH test value: distinct per (gaussian, coefficient, channel), DC far from
// anything display-RGB-like.
float shValue(size_t gaussian, int k, int c)
{
    return (k == 0 ? -1.3f + 0.4f * static_cast<float>(c)
                   : 0.05f * static_cast<float>(k) - 0.3f * static_cast<float>(c))
           + 0.07f * static_cast<float>(gaussian);
}

// One splat primitive described in the stored domain plus its encodings.
struct PrimitiveSpec
{
    std::vector<double> position, rotation, scale, opacity; // stored values
    int positionType = kFloat, rotationType = kFloat, scaleType = kFloat, opacityType = kFloat;
    bool positionNorm = false, rotationNorm = false, scaleNorm = false, opacityNorm = false;
    int positionStride = 0, scaleStride = 0;
    int degree = 0;
};

// Adds the primitive's accessors and expected canonical values; returns the
// attribute list (the caller may add sparse/extra edits before meshing).
std::vector<std::pair<std::string, int>> addPrimitive(
    GltfWriter & w, const PrimitiveSpec & p, Expected & expected, size_t gaussianBase,
    const std::map<std::string, std::string> & sparse = {})
{
    auto extraFor = [&](const std::string & key)
    {
        const auto it = sparse.find(key);
        return it == sparse.end() ? std::string() : it->second;
    };
    std::vector<std::pair<std::string, int>> attributes;
    attributes.push_back({"POSITION",
                          w.attribute(p.positionType,
                                      p.positionNorm,
                                      3,
                                      p.position,
                                      p.positionStride,
                                      extraFor("POSITION"))});
    attributes.push_back({std::string(kAttr) + "ROTATION",
                          w.attribute(p.rotationType,
                                      p.rotationNorm,
                                      4,
                                      p.rotation,
                                      0,
                                      extraFor(std::string(kAttr) + "ROTATION"))});
    attributes.push_back({std::string(kAttr) + "SCALE",
                          w.attribute(p.scaleType,
                                      p.scaleNorm,
                                      3,
                                      p.scale,
                                      p.scaleStride,
                                      extraFor(std::string(kAttr) + "SCALE"))});
    attributes.push_back({std::string(kAttr) + "OPACITY",
                          w.attribute(p.opacityType,
                                      p.opacityNorm,
                                      1,
                                      p.opacity,
                                      0,
                                      extraFor(std::string(kAttr) + "OPACITY"))});

    const size_t n = p.opacity.size();
    for (size_t i = 0; i < n; ++i)
    {
        expected.positions.push_back(
            glm::vec4(expectedDecode(p.positionType, p.positionNorm, p.position[i * 3 + 0]),
                      expectedDecode(p.positionType, p.positionNorm, p.position[i * 3 + 1]),
                      expectedDecode(p.positionType, p.positionNorm, p.position[i * 3 + 2]),
                      0.0f));
        expected.scales.push_back(
            glm::vec4(expectedDecode(p.scaleType, p.scaleNorm, p.scale[i * 3 + 0]),
                      expectedDecode(p.scaleType, p.scaleNorm, p.scale[i * 3 + 1]),
                      expectedDecode(p.scaleType, p.scaleNorm, p.scale[i * 3 + 2]),
                      0.0f));
        expected.rotations.push_back(normalizedQuaternion(
            expectedDecode(p.rotationType, p.rotationNorm, p.rotation[i * 4 + 0]),
            expectedDecode(p.rotationType, p.rotationNorm, p.rotation[i * 4 + 1]),
            expectedDecode(p.rotationType, p.rotationNorm, p.rotation[i * 4 + 2]),
            expectedDecode(p.rotationType, p.rotationNorm, p.rotation[i * 4 + 3]),
            p.rotationType != kFloat));
        expected.opacities.push_back(expectedDecode(p.opacityType, p.opacityNorm, p.opacity[i]));
        for (int k = 0; k < kShCoefficientsPerGaussian; ++k)
        {
            expected.sh.push_back(k < coefficientCount(p.degree)
                                      ? glm::vec4(shValue(gaussianBase + i, k, 0),
                                                  shValue(gaussianBase + i, k, 1),
                                                  shValue(gaussianBase + i, k, 2),
                                                  0.0f)
                                      : glm::vec4(0.0f));
        }
    }
    for (int k = 0; k < coefficientCount(p.degree); ++k)
    {
        const int band = static_cast<int>(std::floor(std::sqrt(static_cast<double>(k))));
        const int m = k - band * band;
        std::vector<double> raw;
        for (size_t i = 0; i < n; ++i)
        {
            for (int c = 0; c < 3; ++c)
            {
                raw.push_back(shValue(gaussianBase + i, k, c));
            }
        }
        attributes.push_back(
            {shName(band, m), w.attribute(kFloat, false, 3, raw, 0, extraFor(shName(band, m)))});
    }
    return attributes;
}

bool sameCloud(GaussianCloud & a, GaussianCloud & b)
{
    return a.count() == b.count() && a.shDegree() == b.shDegree()
           && std::memcmp(a.positions().data(), b.positions().data(), a.count() * 16) == 0
           && std::memcmp(a.scales().data(), b.scales().data(), a.count() * 16) == 0
           && std::memcmp(a.rotations().data(), b.rotations().data(), a.count() * 16) == 0
           && std::memcmp(a.opacities().data(), b.opacities().data(), a.count() * 4) == 0
           && std::memcmp(a.shCoefficients().data(),
                          b.shCoefficients().data(),
                          a.count() * 16 * kShCoefficientsPerGaussian)
                  == 0;
}

// ---------------------------------------------------------------------------
// (a) decode
// ---------------------------------------------------------------------------

void testDecode(Device & device)
{
    std::printf("\n-- (a) decode: every component type, stride, sparse, SH degree, container --\n");
    GltfWriter w;
    w.extensionsRequired = R"(["KHR_gaussian_splatting","KHR_mesh_quantization"])";
    Expected expected;
    std::vector<std::string> sceneNodes;
    size_t base = 0;

    auto finish = [&](const std::vector<std::pair<std::string, int>> & attributes, size_t count)
    {
        const int mesh = w.addMesh({primitive(attributes)});
        sceneNodes.push_back(std::to_string(w.addNode("\"mesh\":" + std::to_string(mesh))));
        base += count;
    };

    // P1: all float, degree 3; POSITION and SCALE interleaved in one view
    // (byteStride 24). Rotation lanes all distinct and non-zero (a spurious
    // wxyz<->xyzw reorder cannot pass), one of them non-unit (len 2) to
    // exercise the float renormalize; scales 0.5/2.0 straddle 1 (a spurious
    // exp() cannot pass); opacities 0.25/0.9 (a spurious sigmoid cannot pass).
    {
        PrimitiveSpec p;
        p.degree = 3;
        p.position = {1.5, -2.25, 3.75, 0.1, 0.2, -0.3, -4.0, 5.0, -6.0};
        p.scale = {0.5, 2.0, 0.75, 1.5, 0.25, 3.0, 0.1, 0.2, 0.3};
        const double l = std::sqrt(0.1 * 0.1 + 0.2 * 0.2 + 0.3 * 0.3 + 0.9 * 0.9);
        p.rotation = {0.1 / l, 0.2 / l, 0.3 / l, 0.9 / l, 0.2, 0.4, 0.6, 1.8, -0.5, 0.5, -0.5, 0.5};
        p.opacity = {0.25, 0.9, 1.0};
        // Interleave POSITION + SCALE by hand: one view, two accessors.
        std::vector<uint8_t> interleaved;
        for (size_t i = 0; i < 3; ++i)
        {
            for (int c = 0; c < 3; ++c)
                appendComponent(interleaved, kFloat, p.position[i * 3 + c]);
            for (int c = 0; c < 3; ++c)
                appendComponent(interleaved, kFloat, p.scale[i * 3 + c]);
        }
        const int view = w.view(interleaved, 24);
        const int pos = w.accessor(
            view, kFloat, false, 3, "VEC3", 0, R"(,"min":[-4,0.2,-6],"max":[1.5,5,3.75])");
        const int scale = w.accessor(view, kFloat, false, 3, "VEC3", 12);
        Expected scratch;
        auto attributes = addPrimitive(w, p, scratch, base);
        attributes[0].second = pos;
        attributes[2].second = scale;
        expected.positions.insert(
            expected.positions.end(), scratch.positions.begin(), scratch.positions.end());
        expected.scales.insert(expected.scales.end(), scratch.scales.begin(), scratch.scales.end());
        expected.rotations.insert(
            expected.rotations.end(), scratch.rotations.begin(), scratch.rotations.end());
        expected.opacities.insert(
            expected.opacities.end(), scratch.opacities.begin(), scratch.opacities.end());
        expected.sh.insert(expected.sh.end(), scratch.sh.begin(), scratch.sh.end());
        finish(attributes, 3);
    }

    // P2: signed byte POSITION (KHR_mesh_quantization, stride 4), signed byte
    // normalized ROTATION (incl. -128 -> -1 clamp), unsigned byte SCALE
    // (not normalized, stride 4), unsigned byte normalized OPACITY. Degree 1.
    {
        PrimitiveSpec p;
        p.degree = 1;
        p.positionType = kByte;
        p.positionStride = 4;
        p.position = {1, -2, 3, -128, 127, 0, 5, 6, 7};
        p.rotationType = kByte;
        p.rotationNorm = true;
        p.rotation = {127, -64, 32, -16, -128, 10, 20, 30, -5, 7, -9, 120};
        p.scaleType = kUByte;
        p.scaleStride = 4;
        p.scale = {1, 2, 3, 4, 5, 6, 7, 8, 9};
        p.opacityType = kUByte;
        p.opacityNorm = true;
        p.opacity = {0, 128, 255};
        finish(addPrimitive(w, p, expected, base), 3);
    }

    // P3: signed short normalized POSITION (-32768 -> -1 clamp, stride 8),
    // signed short normalized ROTATION, unsigned byte normalized SCALE,
    // unsigned short normalized OPACITY with a sparse override (ubyte
    // indices), a sparse override on the DC (ushort indices), and one band-2
    // coefficient that is SPARSE-ONLY (no bufferView: zeros + overrides).
    // Degree 2.
    {
        PrimitiveSpec p;
        p.degree = 2;
        p.positionType = kShort;
        p.positionNorm = true;
        p.positionStride = 8;
        p.position = {32767, -32768, 0, 1000, -2000, 3000, -1, 1, 16384};
        p.rotationType = kShort;
        p.rotationNorm = true;
        p.rotation = {1000, 2000, -3000, 30000, -32768, 5, 6, 7, 300, -400, 500, -600};
        p.scaleType = kUByte;
        p.scaleNorm = true;
        p.scaleStride = 4;
        p.scale = {255, 128, 1, 10, 20, 30, 200, 100, 50};
        p.opacityType = kUShort;
        p.opacityNorm = true;
        p.opacity = {0, 65535, 32768};

        std::map<std::string, std::string> sparse;
        {
            std::vector<uint8_t> indices;
            appendComponent(indices, kUByte, 1);
            std::vector<uint8_t> values;
            appendComponent(values, kUShort, 1000);
            const int iv = w.view(indices);
            const int vv = w.view(values);
            sparse["KHR_gaussian_splatting:OPACITY"] =
                ",\"sparse\":{\"count\":1,\"indices\":{\"bufferView\":" + std::to_string(iv)
                + ",\"componentType\":5121},\"values\":{\"bufferView\":" + std::to_string(vv)
                + "}}";
        }
        {
            std::vector<uint8_t> indices;
            appendComponent(indices, kUShort, 0);
            appendComponent(indices, kUShort, 2);
            std::vector<uint8_t> values;
            for (double v : {0.11, 0.22, 0.33, -0.44, -0.55, -0.66})
                appendComponent(values, kFloat, v);
            const int iv = w.view(indices);
            const int vv = w.view(values);
            sparse[shName(0, 0)] =
                ",\"sparse\":{\"count\":2,\"indices\":{\"bufferView\":" + std::to_string(iv)
                + ",\"componentType\":5123},\"values\":{\"bufferView\":" + std::to_string(vv)
                + "}}";
        }
        const size_t first = expected.count();
        auto attributes = addPrimitive(w, p, expected, base, sparse);

        // Band-2 coefficient m=3 (k = 7): sparse-only accessor.
        {
            std::vector<uint8_t> indices;
            appendComponent(indices, kUShort, 1);
            std::vector<uint8_t> values;
            for (double v : {0.5, -0.5, 0.25})
                appendComponent(values, kFloat, v);
            const int iv = w.view(indices);
            const int vv = w.view(values);
            const int acc = w.accessor(
                -1,
                kFloat,
                false,
                3,
                "VEC3",
                0,
                ",\"sparse\":{\"count\":1,\"indices\":{\"bufferView\":" + std::to_string(iv)
                    + ",\"componentType\":5123},\"values\":{\"bufferView\":" + std::to_string(vv)
                    + "}}");
            for (auto & a : attributes)
            {
                if (a.first == shName(2, 3))
                    a.second = acc;
            }
            expected.sh[(first + 0) * 16 + 7] = glm::vec4(0.0f);
            expected.sh[(first + 1) * 16 + 7] = glm::vec4(0.5f, -0.5f, 0.25f, 0.0f);
            expected.sh[(first + 2) * 16 + 7] = glm::vec4(0.0f);
        }
        expected.opacities[first + 1] = 1000.0f / 65535.0f;
        expected.sh[(first + 0) * 16] = glm::vec4(0.11f, 0.22f, 0.33f, 0.0f);
        expected.sh[(first + 2) * 16] = glm::vec4(-0.44f, -0.55f, -0.66f, 0.0f);
        finish(attributes, 3);
    }

    // P4: unsigned byte normalized POSITION, float ROTATION, unsigned short
    // (not normalized) SCALE with stride 8, float OPACITY. Degree 0.
    {
        PrimitiveSpec p;
        p.positionType = kUByte;
        p.positionNorm = true;
        p.positionStride = 4;
        p.position = {0, 128, 255, 17, 34, 51, 200, 100, 50};
        p.rotation = {0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5};
        p.scaleType = kUShort;
        p.scaleStride = 8;
        p.scale = {1, 300, 65535, 2, 3, 4, 10, 20, 30};
        p.opacity = {0.0, 0.5, 0.125};
        finish(addPrimitive(w, p, expected, base), 3);
    }

    // P5: unsigned short POSITION (not normalized, stride 8), signed byte
    // normalized ROTATION, unsigned short normalized SCALE (stride 8),
    // unsigned byte normalized OPACITY with a sparse override using UINT
    // indices. Degree 0.
    {
        PrimitiveSpec p;
        p.positionType = kUShort;
        p.positionStride = 8;
        p.position = {0, 1, 2, 65535, 1000, 7, 3, 4, 5};
        p.rotationType = kByte;
        p.rotationNorm = true;
        p.rotation = {1, 2, 3, 4, -1, -2, -3, -4, 0, 0, 127, 0};
        p.scaleType = kUShort;
        p.scaleNorm = true;
        p.scaleStride = 8;
        p.scale = {65535, 32768, 1, 100, 200, 300, 5, 6, 7};
        p.opacityType = kUByte;
        p.opacityNorm = true;
        p.opacity = {10, 20, 30};
        std::map<std::string, std::string> sparse;
        std::vector<uint8_t> indices;
        appendComponent(indices, kUInt, 0);
        appendComponent(indices, kUInt, 2);
        std::vector<uint8_t> values;
        appendComponent(values, kUByte, 255);
        appendComponent(values, kUByte, 1);
        const int iv = w.view(indices);
        const int vv = w.view(values);
        sparse["KHR_gaussian_splatting:OPACITY"] =
            ",\"sparse\":{\"count\":2,\"indices\":{\"bufferView\":" + std::to_string(iv)
            + ",\"componentType\":5125},\"values\":{\"bufferView\":" + std::to_string(vv) + "}}";
        const size_t first = expected.count();
        auto attributes = addPrimitive(w, p, expected, base, sparse);
        expected.opacities[first + 0] = 1.0f;
        expected.opacities[first + 2] = 1.0f / 255.0f;
        finish(attributes, 3);
    }

    std::string nodesList = "[";
    for (size_t i = 0; i < sceneNodes.size(); ++i)
        nodesList += (i ? "," : "") + sceneNodes[i];
    w.sceneNodes = nodesList + "]";

    const std::string glbPath = tempPath("decode.glb");
    const std::string dataUriPath = tempPath("decode_datauri.gltf");
    const std::string externalPath = tempPath("decode_external.gltf");
    w.writeGlb(glbPath);
    w.writeGltfDataUri(dataUriPath);
    w.writeGltfExternal(externalPath, "decode external.bin", "decode%20external.bin");

    const std::string nullUriPath = tempPath("decode_null_uri.glb");
    w.writeGlbNullUri(nullUriPath);
    {
        auto viaNull = loadScene(device, nullUriPath);
        auto viaGlb = loadScene(device, glbPath);
        bool same = viaNull->count() == viaGlb->count();
        for (size_t i = 0; same && i < viaNull->count(); ++i)
        {
            same = viaNull->positions()[i] == viaGlb->positions()[i] &&
                   viaNull->opacities()[i] == viaGlb->opacities()[i];
        }
        check(same, "decode: a GLB buffer with \"uri\": null reads its BIN chunk like an absent uri");
    }

    auto cloud = loadScene(device, glbPath);
    check(cloud->count() == expected.count(),
          "decode: Gaussian count " + std::to_string(cloud->count())
              + " == " + std::to_string(expected.count()));
    check(cloud->shDegree() == 3, "decode: cloud SH degree is the max over primitives (3)");
    check(!cloud->antialiased(), "decode: antialiased defaults to false");
    if (cloud->count() != expected.count())
        return;

    size_t bad = 0;
    float worstRotation = 0.0f;
    for (size_t i = 0; i < expected.count(); ++i)
    {
        auto report = [&](bool mismatch, const char * field)
        {
            if (mismatch)
            {
                ++bad;
                std::printf("  mismatch: Gaussian %zu %s\n", i, field);
            }
        };
        report(cloud->positions()[i] != expected.positions[i], "position");
        report(cloud->scales()[i] != expected.scales[i], "scale");
        report(cloud->opacities()[i] != expected.opacities[i], "opacity");
        for (int c = 0; c < 4; ++c)
            worstRotation = std::max(worstRotation,
                                     std::abs(cloud->rotations()[i][c] - expected.rotations[i][c]));
        for (int k = 0; k < kShCoefficientsPerGaussian; ++k)
            report(cloud->shCoefficients()[i * 16 + k] != expected.sh[i * 16 + k],
                   "SH coefficient");
    }
    check(bad == 0,
          "decode: positions/scales/opacities/SH bit-exact vs the spec decode ("
              + std::to_string(bad) + " mismatches over 15 Gaussians x 19 fields)");
    check(worstRotation <= 1e-7f,
          "decode: rotations match the normalized spec decode (worst " + num(worstRotation) + ")");

    // Degeneracy guards: prove the fixture exercises what it claims to.
    const glm::vec4 & r1 = cloud->rotations()[0];
    check(r1.x != r1.y && r1.y != r1.z && r1.z != r1.w && r1.x != 0.0f && r1.w > 0.9f,
          "guard: four distinct lanes, w last (a wxyz misread would move 0.9 out of w)");
    check(std::abs(glm::length(cloud->rotations()[1]) - 1.0f) < 1e-6f
              && cloud->rotations()[1].w < 0.95f,
          "guard: the length-2 float quaternion was renormalized");
    check(cloud->scales()[0].x == 0.5f && cloud->scales()[0].y == 2.0f,
          "guard: scales 0.5/2.0 not exp()'d");
    check(cloud->opacities()[0] == 0.25f, "guard: opacity 0.25 not sigmoid()'d");
    check(cloud->shCoefficients()[0].x == shValue(0, 0, 0) && cloud->shCoefficients()[0].x < -1.0f,
          "guard: SH DC is the raw coefficient (-1.3), not display RGB");
    check(cloud->positions()[6].x == 1.0f, "guard: normalized short 32767 decodes to 1.0");
    check(cloud->positions()[6].y == -1.0f, "guard: normalized short -32768 clamps to -1.0");
    check(cloud->positions()[4].x == -128.0f,
          "guard: non-normalized signed byte keeps its integer value");
    bool zeroTail = true;
    for (size_t i = 3; i < 6; ++i) // P2 is degree 1: coefficients 4..15 exactly zero
        for (int k = 4; k < 16; ++k)
            zeroTail = zeroTail && cloud->shCoefficients()[i * 16 + k] == glm::vec4(0.0f);
    for (size_t i = 9; i < 15; ++i) // P4/P5 degree 0: 1..15 exactly zero
        for (int k = 1; k < 16; ++k)
            zeroTail = zeroTail && cloud->shCoefficients()[i * 16 + k] == glm::vec4(0.0f);
    check(zeroTail, "decode: SH above each primitive's degree is exactly zero");
    check(cloud->shCoefficients()[7 * 16 + 7] == glm::vec4(0.5f, -0.5f, 0.25f, 0.0f)
              && cloud->shCoefficients()[6 * 16 + 7] == glm::vec4(0.0f),
          "decode: sparse-only accessor = zeros + override");

    auto fromDataUri = loadScene(device, dataUriPath);
    auto fromExternal = loadScene(device, externalPath);
    check(sameCloud(*cloud, *fromDataUri), "containers: .gltf with base64 data: URI == .glb");
    check(sameCloud(*cloud, *fromExternal),
          "containers: .gltf with external percent-encoded .bin == .glb");
}

// ---------------------------------------------------------------------------
// (b) node transforms
// ---------------------------------------------------------------------------

using DMat3 = std::array<std::array<double, 3>, 3>;

DMat3 quatToMat(double x, double y, double z, double w)
{
    return {{{1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)},
             {2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)},
             {2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)}}};
}

DMat3 mul(const DMat3 & a, const DMat3 & b)
{
    DMat3 r{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                r[i][j] += a[i][k] * b[k][j];
    return r;
}

DMat3 transpose(const DMat3 & a)
{
    DMat3 r{};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            r[i][j] = a[j][i];
    return r;
}

// Covariance R diag(s^2) R^T from canonical scale + xyzw rotation.
DMat3 covariance(const glm::vec4 & scale, const glm::vec4 & rotation)
{
    const DMat3 r = quatToMat(rotation.x, rotation.y, rotation.z, rotation.w);
    DMat3 s{};
    for (int i = 0; i < 3; ++i)
        s[i][i] = static_cast<double>(scale[i]) * scale[i];
    return mul(mul(r, s), transpose(r));
}

// Independent restatement of the spec's SH colour sum (Lighting section),
// bands 1..3, for one channel.
double shColor(const double * c, double x, double y, double z)
{
    const double xx = x * x, yy = y * y, zz = z * z;
    double r = 0.0;
    r += c[1] * y * -0.4886025119029199 + c[2] * z * 0.4886025119029199
         + c[3] * x * -0.4886025119029199;
    r += c[4] * x * y * 1.092548430592079 + c[5] * y * z * -1.092548430592079
         + c[6] * (2 * zz - xx - yy) * 0.3153915652525200 + c[7] * x * z * -1.092548430592079
         + c[8] * (xx - yy) * 0.5462742152960395;
    r += c[9] * y * (3 * xx - yy) * -0.5900435899266435 + c[10] * x * y * z * 2.890611442640554
         + c[11] * y * (4 * zz - xx - yy) * -0.4570457994644657
         + c[12] * z * (2 * zz - 3 * xx - 3 * yy) * 0.3731763325901154
         + c[13] * x * (4 * zz - xx - yy) * -0.4570457994644657
         + c[14] * z * (xx - yy) * 1.445305721320277
         + c[15] * x * (xx - 3 * yy) * -0.5900435899266435;
    return r;
}

// A float, degree-`degree` primitive of n Gaussians with deterministic data.
std::vector<std::pair<std::string, int>> addFloatPrimitive(GltfWriter & w, Expected & e, size_t n,
                                                           int degree, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    PrimitiveSpec p;
    p.degree = degree;
    for (size_t i = 0; i < n; ++i)
    {
        for (int c = 0; c < 3; ++c)
        {
            p.position.push_back(u(rng));
            p.scale.push_back(0.05 + 0.1 * (u(rng) + 1.0));
        }
        double q[4] = {u(rng), u(rng), u(rng), u(rng)};
        const double l = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        for (double v : q)
            p.rotation.push_back(v / l);
        p.opacity.push_back(0.3 + 0.3 * (u(rng) + 1.0));
    }
    return addPrimitive(w, p, e, 0);
}

std::shared_ptr<GaussianCloud> loadSingleNode(Device & device, const std::string & name,
                                              const std::string & nodeJson, int degree,
                                              Expected & local, const std::string & parentJson = "")
{
    GltfWriter w;
    const auto attributes = addFloatPrimitive(w, local, 4, degree, 7);
    const int mesh = w.addMesh({primitive(attributes)});
    if (parentJson.empty())
    {
        w.addNode("\"mesh\":" + std::to_string(mesh) + (nodeJson.empty() ? "" : "," + nodeJson));
        w.sceneNodes = "[0]";
    }
    else
    {
        w.addNode(parentJson + ",\"children\":[1]");
        w.addNode("\"mesh\":" + std::to_string(mesh) + (nodeJson.empty() ? "" : "," + nodeJson));
        w.sceneNodes = "[0]";
    }
    w.writeGlb(tempPath(name));
    return loadScene(device, tempPath(name));
}

void testTransforms(Device & device)
{
    std::printf("\n-- (b) node transforms --\n");
    const double h = std::sqrt(0.5);

    // B1: 90 degrees about +Z, degree 1 -- hand-derived. R maps x->y, y->-x,
    // so a world direction d sees the local colour at R^T d = (y, -x, z).
    // With band 1 = C1 * (-y c0 + z c1 - x c2), that is
    // C1 * (x c0 + z c1 - y c2), i.e. new (c0, c1, c2) = (c2, c1, -c0).
    {
        Expected local;
        auto cloud = loadSingleNode(
            device, "rot_z90.glb", "\"rotation\":[0,0," + num(h) + "," + num(h) + "]", 1, local);
        float worst = 0.0f;
        float worstPos = 0.0f;
        for (size_t i = 0; i < 4; ++i)
        {
            const glm::vec4 * c = &local.sh[i * 16];
            const glm::vec4 * out = &cloud->shCoefficients()[i * 16];
            const glm::vec4 expected[3] = {c[3], c[2], -c[1]};
            for (int k = 0; k < 3; ++k)
                for (int ch = 0; ch < 3; ++ch)
                    worst = std::max(worst, std::abs(out[1 + k][ch] - expected[k][ch]));
            const glm::vec4 & p = local.positions[i];
            const glm::vec4 & q = cloud->positions()[i];
            worstPos =
                std::max({worstPos, std::abs(q.x + p.y), std::abs(q.y - p.x), std::abs(q.z - p.z)});
            if (out[0] != c[0])
                worst = 1.0f; // DC is rotation-invariant: must pass through untouched
        }
        check(worst <= 1e-6f,
              "transform: 90deg-about-Z band-1 SH == hand-derived (c2, c1, -c0), worst "
                  + num(worst));
        check(worstPos <= 1e-6f,
              "transform: 90deg-about-Z positions (x,y,z) -> (-y,x,z), worst " + num(worstPos));
        check(std::abs(local.sh[16 + 1].x - local.sh[16 + 3].x) > 0.05f,
              "guard: band-1 coefficients differ, so the permutation is observable");
    }

    // B2: hierarchy (parent: translation + uniform scale 2; child: rotation)
    // vs B3: one `matrix` node with the same global transform.
    const double qx = 0.2, qy = -0.5, qz = 0.3;
    const double qw = std::sqrt(1.0 - qx * qx - qy * qy - qz * qz);
    const DMat3 rNode = quatToMat(qx, qy, qz, qw);
    const double t[3] = {1.0, -2.0, 3.0};
    const double s = 2.0;
    Expected local;
    auto hierarchy = loadSingleNode(
        device,
        "hierarchy.glb",
        "\"rotation\":[" + num(qx) + "," + num(qy) + "," + num(qz) + "," + num(qw) + "]",
        3,
        local,
        "\"translation\":[1,-2,3],\"scale\":[2,2,2]");
    {
        double worstPos = 0.0, worstCov = 0.0;
        for (size_t i = 0; i < 4; ++i)
        {
            const glm::vec4 & p = local.positions[i];
            for (int r = 0; r < 3; ++r)
            {
                const double expected =
                    t[r] + s * (rNode[r][0] * p.x + rNode[r][1] * p.y + rNode[r][2] * p.z);
                worstPos = std::max(worstPos, std::abs(expected - hierarchy->positions()[i][r]));
            }
            DMat3 cov =
                mul(mul(rNode, covariance(local.scales[i], local.rotations[i])), transpose(rNode));
            const DMat3 got = covariance(hierarchy->scales()[i], hierarchy->rotations()[i]);
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    worstCov = std::max(worstCov, std::abs(s * s * cov[a][b] - got[a][b]));
        }
        check(worstPos <= 1e-5, "transform: hierarchy T*S(2)*R positions, worst " + num(worstPos));
        check(worstCov <= 1e-6,
              "transform: hierarchy covariance == M Sigma M^T, worst " + num(worstCov));
    }

    std::string matrix = "\"matrix\":[";
    for (int c = 0; c < 4; ++c)
        for (int r = 0; r < 4; ++r)
        {
            const double v = c < 3 ? (r < 3 ? s * rNode[r][c] : 0.0) : (r < 3 ? t[r] : 1.0);
            matrix += (c || r ? "," : "") + num(v);
        }
    matrix += "]";
    Expected local2;
    auto fromMatrix = loadSingleNode(device, "matrix.glb", matrix, 3, local2);
    {
        float worst = 0.0f;
        for (size_t i = 0; i < 4; ++i)
        {
            for (int c = 0; c < 4; ++c)
            {
                worst = std::max(
                    worst, std::abs(fromMatrix->positions()[i][c] - hierarchy->positions()[i][c]));
                worst = std::max(worst,
                                 std::abs(fromMatrix->scales()[i][c] - hierarchy->scales()[i][c]));
                // q and -q are the same rotation; compare sign-aligned.
                const float sign =
                    glm::dot(fromMatrix->rotations()[i], hierarchy->rotations()[i]) < 0.0f ? -1.0f
                                                                                           : 1.0f;
                worst = std::max(
                    worst,
                    std::abs(sign * fromMatrix->rotations()[i][c] - hierarchy->rotations()[i][c]));
            }
            for (int k = 0; k < 16; ++k)
                for (int c = 0; c < 3; ++c)
                    worst = std::max(worst,
                                     std::abs(fromMatrix->shCoefficients()[i * 16 + k][c]
                                              - hierarchy->shCoefficients()[i * 16 + k][c]));
        }
        check(worst <= 1e-5f,
              "transform: `matrix` node == equivalent TRS hierarchy, worst " + num(worst));
    }

    // Degree-3 SH rotation vs the spec's colour sum, evaluated independently:
    // colour of the loaded (world) coefficients along d must equal the
    // colour of the local coefficients along R^T d.
    {
        std::mt19937 rng(11);
        std::normal_distribution<double> g(0.0, 1.0);
        double worst = 0.0, moved = 0.0;
        for (size_t i = 0; i < 4; ++i)
        {
            for (int ch = 0; ch < 3; ++ch)
            {
                double localC[16], worldC[16];
                for (int k = 0; k < 16; ++k)
                {
                    localC[k] = local.sh[i * 16 + k][ch];
                    worldC[k] = hierarchy->shCoefficients()[i * 16 + k][ch];
                    moved = std::max(moved, std::abs(localC[k] - worldC[k]));
                }
                for (int sample = 0; sample < 32; ++sample)
                {
                    double d[3] = {g(rng), g(rng), g(rng)};
                    const double l = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
                    for (double & v : d)
                        v /= l;
                    double rd[3];
                    for (int r = 0; r < 3; ++r)
                        rd[r] =
                            rNode[0][r] * d[0] + rNode[1][r] * d[1] + rNode[2][r] * d[2]; // R^T d
                    worst = std::max(worst,
                                     std::abs(shColor(worldC, d[0], d[1], d[2])
                                              - shColor(localC, rd[0], rd[1], rd[2])));
                }
            }
        }
        check(worst <= 2e-6,
              "transform: degree-3 SH rotation reproduces the local colour field, worst "
                  + num(worst));
        check(moved > 0.1,
              "guard: the rotation actually changed the SH coefficients (max change " + num(moved)
                  + ")");
    }

    // B4: non-uniform scale on a rotated node -- each covariance must be
    // re-diagonalized, checked as the full matrix M Sigma M^T.
    {
        Expected localN;
        auto cloud = loadSingleNode(device,
                                    "nonuniform.glb",
                                    "\"rotation\":[" + num(qx) + "," + num(qy) + "," + num(qz) + ","
                                        + num(qw) + "],\"scale\":[1,2,3]",
                                    0,
                                    localN);
        double worstRel = 0.0, worstUnit = 0.0;
        for (size_t i = 0; i < 4; ++i)
        {
            DMat3 m = rNode;
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    m[r][c] *= (c + 1.0);
            const DMat3 expected =
                mul(mul(m, covariance(localN.scales[i], localN.rotations[i])), transpose(m));
            const DMat3 got = covariance(cloud->scales()[i], cloud->rotations()[i]);
            double norm = 0.0, err = 0.0;
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                {
                    norm = std::max(norm, std::abs(expected[a][b]));
                    err = std::max(err, std::abs(expected[a][b] - got[a][b]));
                }
            worstRel = std::max(worstRel, err / norm);
            worstUnit =
                std::max(worstUnit,
                         static_cast<double>(std::abs(glm::length(cloud->rotations()[i]) - 1.0f)));
        }
        check(worstUnit <= 1e-6,
              "transform: re-diagonalized rotations are unit (worst " + num(worstUnit) + ")");
        check(worstRel <= 1e-6,
              "transform: non-uniform scale covariance == M Sigma M^T (relative " + num(worstRel)
                  + ")");
    }
}

// ---------------------------------------------------------------------------
// (c) rejections
// ---------------------------------------------------------------------------

void expectRejected(Device & device, const std::string & path, const std::string & needle,
                    const std::string & what)
{
    std::string message;
    try
    {
        loadScene(device, path);
    }
    catch (const std::exception & e)
    {
        message = e.what();
    }
    const bool ok = !message.empty() && message.find(needle) != std::string::npos;
    check(ok,
          "rejects " + what
              + (message.empty() ? " (ACCEPTED)"
                                 : " -- \"" + message.substr(0, message.rfind(" at ")) + "\""));
}

// A valid degree-1 baseline the rejection cases each break in one place.
struct Baseline
{
    GltfWriter w;
    std::vector<std::pair<std::string, int>> attributes;
    Expected e;

    explicit Baseline(int degree = 1)
    {
        attributes = addFloatPrimitive(w, e, 3, degree, 3);
        w.sceneNodes = "[0]";
    }

    void erase(const std::string & semantic)
    {
        attributes.erase(std::remove_if(attributes.begin(),
                                        attributes.end(),
                                        [&](const auto & a) { return a.first == semantic; }),
                         attributes.end());
    }

    std::string write(const std::string & name, const std::string & extension = kDefaultExtension,
                      int mode = 0, const std::string & nodeExtra = "")
    {
        w.addMesh({primitive(attributes, extension, mode)});
        w.addNode("\"mesh\":0" + nodeExtra);
        const std::string path = tempPath(name);
        w.writeGlb(path);
        return path;
    }
};

void writeText(const std::string & path, const std::string & text)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file << text;
}

void testRejections(Device & device)
{
    std::printf("\n-- (c) malformed files --\n");
    {
        Baseline b;
        b.erase(std::string(kAttr) + "OPACITY");
        expectRejected(device,
                       b.write("missing_opacity.glb"),
                       "missing required attribute",
                       "missing OPACITY");
    }
    {
        Baseline b;
        b.erase(shName(0, 0));
        expectRejected(
            device, b.write("missing_dc.glb"), "SH_DEGREE_0_COEF_0", "missing SH degree 0");
    }
    {
        Baseline b;
        b.erase(shName(1, 2));
        expectRejected(
            device, b.write("partial_sh.glb"), "partially defined", "partial SH degree 1 (2 of 3)");
    }
    {
        Baseline b(2);
        for (int m = 0; m < 3; ++m)
            b.erase(shName(1, m));
        expectRejected(
            device, b.write("sh_gap.glb"), "without degree 1", "SH degree 2 without degree 1");
    }
    {
        Baseline b;
        expectRejected(
            device,
            b.write("kernel.glb", R"({"kernel":"customShape","colorSpace":"srgb_rec709_display"})"),
            "unsupported kernel",
            "kernel \"customShape\"");
    }
    {
        Baseline b;
        expectRejected(device,
                       b.write("no_colorspace.glb", R"({"kernel":"ellipse"})"),
                       "colorSpace",
                       "missing required colorSpace");
    }
    {
        Baseline b;
        expectRejected(
            device,
            b.write(
                "projection.glb",
                R"({"kernel":"ellipse","colorSpace":"srgb_rec709_display","projection":"orthographic"})"),
            "unsupported projection",
            "projection \"orthographic\"");
    }
    {
        Baseline b;
        expectRejected(device,
                       b.write("triangles.glb", kDefaultExtension, -1),
                       "POINTS",
                       "mode absent (TRIANGLES)");
    }
    {
        Baseline b;
        // POSITION's bufferView holds 2 of its 3 elements.
        b.w.views[0].replace(b.w.views[0].find("\"byteLength\":36"), 15, "\"byteLength\":24");
        expectRejected(device,
                       b.write("accessor_bounds.glb"),
                       "past the end",
                       "accessor reading past its bufferView");
    }
    {
        Baseline b;
        b.w.views[0] = "{\"buffer\":0,\"byteOffset\":0,\"byteLength\":1000000}";
        expectRejected(device,
                       b.write("view_bounds.glb"),
                       "out of bounds",
                       "bufferView past the end of its buffer");
    }
    {
        Baseline b;
        b.w.views[0].insert(b.w.views[0].size() - 1, ",\"byteStride\":6");
        expectRejected(
            device, b.write("stride.glb"), "byteStride", "byteStride not a multiple of 4");
    }
    {
        Baseline b;
        b.attributes.push_back(
            {shName(1, 0), b.w.attribute(kUByte, true, 3, {1, 2, 3, 4, 5, 6, 7, 8, 9})});
        b.erase(shName(1, 0));
        b.attributes.push_back({shName(1, 0), static_cast<int>(b.w.accessors.size() - 1)});
        expectRejected(
            device, b.write("sh_type.glb"), "does not allow", "SH stored as unsigned byte");
    }
    {
        Baseline b;
        b.erase(std::string(kAttr) + "OPACITY");
        b.attributes.push_back(
            {std::string(kAttr) + "OPACITY", b.w.attribute(kFloat, false, 1, {0.5, 1.5, 0.2})});
        expectRejected(device, b.write("opacity_range.glb"), "must be in [0, 1]", "opacity 1.5");
    }
    {
        Baseline b;
        b.erase(std::string(kAttr) + "SCALE");
        b.attributes.push_back(
            {std::string(kAttr) + "SCALE",
             b.w.attribute(kFloat, false, 3, {0.1, 0.1, 0.1, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1})});
        expectRejected(device, b.write("negative_scale.glb"), "non-negative", "negative scale");
    }
    {
        Baseline b;
        b.erase(std::string(kAttr) + "OPACITY");
        b.attributes.push_back(
            {std::string(kAttr) + "OPACITY", b.w.attribute(kFloat, false, 1, {0.5, 0.5})});
        expectRejected(device,
                       b.write("count_mismatch.glb"),
                       "elements but POSITION",
                       "attribute count mismatch");
    }
    {
        Baseline b;
        b.erase(std::string(kAttr) + "ROTATION");
        b.attributes.push_back(
            {std::string(kAttr) + "ROTATION",
             b.w.attribute(kFloat, false, 4, {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1})});
        expectRejected(device, b.write("zero_quat.glb"), "zero-length", "zero-length quaternion");
    }
    {
        Baseline b;
        b.w.extensionsRequired = R"(["KHR_gaussian_splatting","KHR_draco_mesh_compression"])";
        expectRejected(device,
                       b.write("required_ext.glb"),
                       "KHR_draco_mesh_compression",
                       "an unsupported extensionsRequired entry");
    }
    {
        Baseline b;
        // Sparse indices not strictly increasing.
        std::vector<uint8_t> indices;
        appendComponent(indices, kUByte, 2);
        appendComponent(indices, kUByte, 1);
        std::vector<uint8_t> values;
        appendComponent(values, kFloat, 0.5);
        appendComponent(values, kFloat, 0.5);
        const int iv = b.w.view(indices);
        const int vv = b.w.view(values);
        b.erase(std::string(kAttr) + "OPACITY");
        b.attributes.push_back(
            {std::string(kAttr) + "OPACITY",
             b.w.attribute(
                 kFloat,
                 false,
                 1,
                 {0.1, 0.2, 0.3},
                 0,
                 ",\"sparse\":{\"count\":2,\"indices\":{\"bufferView\":" + std::to_string(iv)
                     + ",\"componentType\":5121},\"values\":{\"bufferView\":" + std::to_string(vv)
                     + "}}")});
        expectRejected(device,
                       b.write("sparse_order.glb"),
                       "strictly increasing",
                       "non-increasing sparse indices");
    }
    {
        // Shear: non-uniform parent scale over a rotated child.
        Baseline b;
        b.w.addMesh({primitive(b.attributes)});
        b.w.addNode("\"scale\":[1,3,1],\"children\":[1]");
        b.w.addNode("\"mesh\":0,\"rotation\":[0,0,0.38268343236508978,0.92387953251128674]");
        b.w.writeGlb(tempPath("shear2.glb"));
        expectRejected(
            device, tempPath("shear2.glb"), "not decomposable", "a sheared node transform");
    }
    {
        Baseline b;
        expectRejected(device,
                       b.write("mirror.glb", kDefaultExtension, 0, ",\"scale\":[1,-1,1]"),
                       "mirroring",
                       "a mirroring (negative scale) node transform");
    }
    {
        Baseline b;
        expectRejected(device,
                       b.write("no_splats.glb", ""),
                       "no KHR_gaussian_splatting primitive",
                       "a file with no splat primitive");
    }
    {
        Baseline b;
        const std::string path = b.write("bad_length.glb");
        std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
        file.seekp(8);
        const uint32_t wrong = 12345;
        file.write(reinterpret_cast<const char *>(&wrong), 4);
        file.close();
        expectRejected(device, path, "does not match the file size", ".glb header length mismatch");
    }
    writeText(tempPath("not_glb.glb"), "this is not binary glTF");
    expectRejected(device,
                   tempPath("not_glb.glb"),
                   "binary glTF magic",
                   ".glb without the glTF magic (dispatch veto)");
    writeText(tempPath("garbage.gltf"), "{\"asset\": {\"version\": \"2.0\"}, \"meshes\": [ oops");
    expectRejected(device, tempPath("garbage.gltf"), "malformed glTF JSON", "unparseable JSON");
    writeText(tempPath("gltf1.gltf"), R"({"asset":{"version":"1.0"}})");
    expectRejected(device, tempPath("gltf1.gltf"), "asset version", "glTF 1.0");
}

// ---------------------------------------------------------------------------
// (d) rendering through Renderer
// ---------------------------------------------------------------------------

// The standard 3DGS .ply (pre-activation, wxyz, channel-major f_rest), written
// from canonical values -- so PLYReader's output is the SAME floats up to its
// own exp/sigmoid/normalize, which the .glb then copies from.
void writeDegree3Ply(const std::string & path, size_t n, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file << "ply\nformat binary_little_endian 1.0\nelement vertex " << n << "\n";
    std::vector<std::string> names = {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"};
    for (int i = 0; i < 45; ++i)
        names.push_back("f_rest_" + std::to_string(i));
    for (const char * name :
         {"opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"})
        names.push_back(name);
    for (const auto & name : names)
        file << "property float " << name << "\n";
    file << "end_header\n";
    for (size_t i = 0; i < n; ++i)
    {
        std::vector<float> v;
        // A slab in front of a camera on +Z, depths well separated.
        v.push_back(u(rng) * 1.5f);
        v.push_back(u(rng) * 1.0f);
        v.push_back(-static_cast<float>(i) * (2.0f / static_cast<float>(n)) + u(rng) * 0.001f);
        for (int c = 0; c < 3; ++c)
            v.push_back(u(rng) * 1.5f);
        for (int k = 0; k < 45; ++k)
            v.push_back(u(rng) * 0.4f);
        v.push_back(u(rng) * 2.0f);
        for (int c = 0; c < 3; ++c)
            v.push_back(std::log(0.04f + 0.06f * (u(rng) + 1.0f)));
        for (int c = 0; c < 4; ++c)
            v.push_back(u(rng));
        file.write(reinterpret_cast<const char *>(v.data()),
                   static_cast<std::streamsize>(v.size() * 4));
    }
}

// Converter: canonical cloud -> .glb, one float accessor per attribute, the
// SH as (deg+1)^2 VEC3 accessors. No transform, no conversion -- the
// canonical contract IS the glTF one apart from layout.
void convertToGlb(GaussianCloud & cloud, const std::string & path,
                  const std::string & nodeExtra = "")
{
    GltfWriter w;
    const size_t n = cloud.count();
    auto pack = [&](const std::vector<glm::vec4> & v, int components)
    {
        std::vector<uint8_t> bytes(n * static_cast<size_t>(components) * 4);
        for (size_t i = 0; i < n; ++i)
            std::memcpy(&bytes[i * components * 4], &v[i], static_cast<size_t>(components) * 4);
        return bytes;
    };
    std::vector<std::pair<std::string, int>> attributes;
    attributes.push_back(
        {"POSITION", w.accessor(w.view(pack(cloud.positions(), 3)), kFloat, false, n, "VEC3")});
    attributes.push_back(
        {std::string(kAttr) + "ROTATION",
         w.accessor(w.view(pack(cloud.rotations(), 4)), kFloat, false, n, "VEC4")});
    attributes.push_back({std::string(kAttr) + "SCALE",
                          w.accessor(w.view(pack(cloud.scales(), 3)), kFloat, false, n, "VEC3")});
    std::vector<uint8_t> opacity(n * 4);
    std::memcpy(opacity.data(), cloud.opacities().data(), n * 4);
    attributes.push_back(
        {std::string(kAttr) + "OPACITY", w.accessor(w.view(opacity), kFloat, false, n, "SCALAR")});
    const int degree = static_cast<int>(cloud.shDegree());
    for (int k = 0; k < coefficientCount(degree); ++k)
    {
        std::vector<uint8_t> bytes(n * 12);
        for (size_t i = 0; i < n; ++i)
            std::memcpy(&bytes[i * 12], &cloud.shCoefficients()[i * 16 + k], 12);
        const int band = static_cast<int>(std::floor(std::sqrt(static_cast<double>(k))));
        attributes.push_back(
            {shName(band, k - band * band), w.accessor(w.view(bytes), kFloat, false, n, "VEC3")});
    }
    w.addMesh({primitive(attributes)});
    w.addNode("\"mesh\":0" + nodeExtra);
    w.sceneNodes = "[0]";
    w.writeGlb(path);
}

// ---------------------------------------------------------------------------
// (e) KHR_gaussian_splatting_compression_spz_2
// ---------------------------------------------------------------------------

constexpr const char * kSpzExtension = "KHR_gaussian_splatting_compression_spz_2";

// One spz round trip's worst-case error, restated from spz's packGaussians
// (load-spz.cc). Each bound is half a quantization step (plus float slack):
//  - position: 24-bit fixed point, 12 fractional bits -> 0.5 / 4096
//  - scale: byte = round((log s + 10) * 16) -> |log error| <= 1/32
//  - rotation (smallest-three): each of the three stored components is a
//    9-bit magnitude over [0, sqrt(1/2)], off by <= d = sqrt(1/2) / 511 / 2;
//    the largest is rebuilt as sqrt(1 - sum q_i^2), off by
//    <= d * sum|q_i| / q_max <= 3d to first order (sum|q_i| <=
//    sqrt(3 (1 - q_max^2)) and q_max >= 1/2)
//  - opacity: byte = round(sigmoid(a) * 255) -> 0.5 / 255
//  - SH DC: byte = round(c * 0.15 * 255 + 127.5) -> 0.5 / (0.15 * 255)
//  - SH band 1: round(x * 128), then to the nearest 2^(8-5) = 8-unit
//    bucket -> (0.5 + 4) / 128
//  - SH bands 2-3: 2^(8-4) = 16-unit buckets -> (0.5 + 8) / 128
// Source values stay inside every clamp range (|position| < 2048, opacity in
// (0, 1), |DC| < 3.3, |SH| < 1), so clamping never adds to these.
constexpr double kSpzPositionTolerance = 0.5 / 4096.0 + 1e-6;
constexpr double kSpzLogScaleTolerance = 1.0 / 32.0 + 1e-5;
const double kSpzRotationTolerance = 3.0 * std::sqrt(0.5) / 511.0 / 2.0 + 1e-4;
constexpr double kSpzOpacityTolerance = 0.5 / 255.0 + 1e-6;
constexpr double kSpzDcTolerance = 0.5 / (0.15 * 255.0) + 1e-5;
constexpr double kSpzBand1Tolerance = 4.5 / 128.0 + 1e-6;
constexpr double kSpzBandRestTolerance = 8.5 / 128.0 + 1e-6;

// spz's unpacked (pre-activation) form, in the glTF frame: log scales,
// pre-sigmoid alpha, DC as `colors`, coefficient-major channel-inner sh.
spz::GaussianCloud makeSpzSource(size_t n, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    spz::GaussianCloud g;
    g.numPoints = static_cast<int32_t>(n);
    g.shDegree = 3;
    g.antialiased = true; // the header flag the reader must carry through
    for (size_t i = 0; i < n; ++i)
    {
        for (int c = 0; c < 3; ++c)
        {
            g.positions.push_back(2.0f * u(rng));
            g.scales.push_back(-2.5f + 1.5f * u(rng));
            g.colors.push_back(1.5f * u(rng));
        }
        float q[4] = {u(rng), u(rng), u(rng), u(rng)};
        const float l = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        for (float v : q)
            g.rotations.push_back(v / l);
        g.alphas.push_back(2.0f * u(rng));
        for (int k = 0; k < 15 * 3; ++k)
            g.sh.push_back(0.8f * u(rng));
    }
    return g;
}

// A .glb whose one primitive takes its data from `stream` through the
// extension, with bufferView-less placeholder accessors of `accessorCount`
// elements -- the shape Cesium writes. `spzObject` empty means
// {"bufferView": <the stream's view>}.
std::string writeSpzGlb(const std::string & name, const std::vector<uint8_t> & stream,
                        size_t accessorCount, const std::string & spzObject = "",
                        const std::string & nodeExtra = "",
                        const std::string & baseFields =
                            R"("kernel":"ellipse","colorSpace":"srgb_rec709_display",)")
{
    GltfWriter w;
    w.extensionsRequired =
        std::string(R"(["KHR_gaussian_splatting",")") + kSpzExtension + "\"]";
    const int view = w.view(stream);
    std::vector<std::pair<std::string, int>> attributes = {
        {"POSITION", w.accessor(-1, kFloat, false, accessorCount, "VEC3")},
        {std::string(kAttr) + "ROTATION", w.accessor(-1, kFloat, false, accessorCount, "VEC4")},
        {std::string(kAttr) + "SCALE", w.accessor(-1, kFloat, false, accessorCount, "VEC3")},
        {std::string(kAttr) + "OPACITY", w.accessor(-1, kFloat, false, accessorCount, "SCALAR")}};
    for (int k = 0; k < coefficientCount(3); ++k)
    {
        const int band = static_cast<int>(std::floor(std::sqrt(static_cast<double>(k))));
        attributes.push_back(
            {shName(band, k - band * band), w.accessor(-1, kFloat, false, accessorCount, "VEC3")});
    }
    const std::string object =
        spzObject.empty() ? "{\"bufferView\":" + std::to_string(view) + "}" : spzObject;
    w.addMesh({primitive(attributes,
                         "{" + baseFields + "\"extensions\":{\"" + kSpzExtension + "\":" + object
                             + "}}")});
    w.addNode("\"mesh\":0" + nodeExtra);
    w.sceneNodes = "[0]";
    const std::string path = tempPath(name);
    w.writeGlb(path);
    return path;
}

void testSpzCompression(Device & device)
{
    std::printf("\n-- (e) KHR_gaussian_splatting_compression_spz_2 --\n");
    const size_t n = 64;
    const spz::GaussianCloud source = makeSpzSource(n, 21);
    spz::PackOptions pack;
    pack.version = 3;                       // gzipped NGSP, what spz_2 assets carry
    pack.from = spz::CoordinateSystem::LUF; // the extension's mandated frame
    std::vector<uint8_t> stream;
    const bool packed = spz::saveSpz(source, pack, &stream);
    check(packed && !stream.empty(),
          "spz: packed a degree-3 cloud in memory (" + std::to_string(stream.size()) + " bytes)");

    auto cloud = loadScene(device, writeSpzGlb("spz.glb", stream, n));
    check(cloud->count() == n && cloud->shDegree() == 3,
          "spz: " + std::to_string(cloud->count()) + " Gaussians, SH degree "
              + std::to_string(cloud->shDegree()) + " (expected 64, 3)");
    check(cloud->antialiased(), "spz: the stream's antialiased flag reaches the cloud");

    // Against the SOURCE values, within the quantization bounds above.
    double position = 0.0, logScale = 0.0, rotation = 0.0, opacity = 0.0, dc = 0.0, band1 = 0.0,
           bandRest = 0.0;
    for (size_t i = 0; i < n; ++i)
    {
        const glm::vec4 q(source.rotations[i * 4 + 0],
                          source.rotations[i * 4 + 1],
                          source.rotations[i * 4 + 2],
                          source.rotations[i * 4 + 3]);
        // q and -q are the same rotation; compare sign-aligned.
        const float sign = glm::dot(cloud->rotations()[i], q) < 0.0f ? -1.0f : 1.0f;
        for (int c = 0; c < 4; ++c)
            rotation = std::max(rotation, std::abs(double(sign * cloud->rotations()[i][c] - q[c])));
        for (int c = 0; c < 3; ++c)
        {
            position = std::max(
                position, std::abs(double(cloud->positions()[i][c] - source.positions[i * 3 + c])));
            logScale = std::max(logScale,
                                std::abs(std::log(double(cloud->scales()[i][c]))
                                         - double(source.scales[i * 3 + c])));
            const glm::vec4 * sh = &cloud->shCoefficients()[i * kShCoefficientsPerGaussian];
            dc = std::max(dc, std::abs(double(sh[0][c] - source.colors[i * 3 + c])));
            for (int k = 1; k < 16; ++k)
            {
                const double error =
                    std::abs(double(sh[k][c] - source.sh[i * 45 + (k - 1) * 3 + c]));
                double & worst = k <= 3 ? band1 : bandRest;
                worst = std::max(worst, error);
            }
        }
        const double activated = 1.0 / (1.0 + std::exp(-double(source.alphas[i])));
        opacity = std::max(opacity, std::abs(double(cloud->opacities()[i]) - activated));
    }
    auto within = [&](double worst, double bound, const std::string & what)
    {
        check(worst <= bound, "spz: " + what + " within quantization, worst " + num(worst)
                                  + " <= " + num(bound));
    };
    within(position, kSpzPositionTolerance, "positions (24-bit fixed point)");
    within(logScale, kSpzLogScaleTolerance, "log scales (1/16 steps)");
    within(rotation, kSpzRotationTolerance, "rotations (smallest-three)");
    within(opacity, kSpzOpacityTolerance, "opacities (8-bit, sigmoid applied)");
    within(dc, kSpzDcTolerance, "SH DC (from spz colors)");
    within(band1, kSpzBand1Tolerance, "SH band 1 (5-bit)");
    within(bandRest, kSpzBandRestTolerance, "SH bands 2-3 (4-bit)");

    // The frame, anchored to spz itself: the reader's glTF frame is spz's
    // LUF, i.e. its native RUB with x and z negated -- exactly.
    {
        spz::UnpackOptions native;
        native.to = spz::CoordinateSystem::RUB;
        const spz::GaussianCloud rub = spz::loadSpz(stream, native);
        bool exact = rub.numPoints == static_cast<int32_t>(n);
        float largestX = 0.0f;
        for (size_t i = 0; exact && i < n; ++i)
        {
            const glm::vec4 & p = cloud->positions()[i];
            exact = p.x == -rub.positions[i * 3 + 0] && p.y == rub.positions[i * 3 + 1]
                    && p.z == -rub.positions[i * 3 + 2];
            largestX = std::max(largestX, std::abs(p.x));
        }
        check(exact, "spz: positions are spz's LUF (= RUB with x, z negated), bit-exact");
        check(largestX > 100.0f * kSpzPositionTolerance,
              "guard: |x| reaches " + num(largestX)
                  + ", so a RUB (unflipped) decode would fail the source comparison");
    }

    // Node transform: the compressed path must share the uncompressed one
    // exactly -- the same canonical floats, written uncompressed under the
    // same node, load bit-identically.
    {
        const std::string node = ",\"translation\":[1,-2,3],\"rotation\":[0.2,-0.5,0.3,"
                                 + num(std::sqrt(1.0 - 0.04 - 0.25 - 0.09))
                                 + "],\"scale\":[2,2,2]";
        auto compressed = loadScene(device, writeSpzGlb("spz_node.glb", stream, n, "", node));
        convertToGlb(*cloud, tempPath("spz_node_plain.glb"), node);
        auto plain = loadScene(device, tempPath("spz_node_plain.glb"));
        check(sameCloud(*compressed, *plain),
              "spz: under a node transform == the same data uncompressed, bit-exact");
        check(compressed->positions()[0] != cloud->positions()[0],
              "guard: the node transform moved the Gaussians");
    }

    // Pre-ratification output (Cesium's sh_unit_cube) omits kernel and
    // colorSpace; SPZ primitives default them instead of failing.
    {
        auto draft = loadScene(device, writeSpzGlb("spz_draft.glb", stream, n, "", "", ""));
        check(sameCloud(*draft, *cloud), "spz: a draft-era object without kernel/colorSpace loads");
    }

    // Rejections.
    expectRejected(device,
                   writeSpzGlb("spz_no_view.glb", stream, n, "{}"),
                   "has no 'bufferView'",
                   "an spz_2 object without bufferView");
    {
        const std::vector<uint8_t> truncated(stream.begin(), stream.begin() + stream.size() / 2);
        expectRejected(device,
                       writeSpzGlb("spz_truncated.glb", truncated, n),
                       "did not decode",
                       "a truncated SPZ stream");
    }
    expectRejected(device,
                   writeSpzGlb("spz_count.glb", stream, n + 1),
                   "stream holds 64 Gaussians",
                   "a placeholder accessor count != the SPZ point count");
}

constexpr uint32_t kWidth = 256;
constexpr uint32_t kHeight = 192;

Camera makeCamera(const glm::vec3 & center, float radius)
{
    Camera camera;
    camera.width = kWidth;
    camera.height = kHeight;
    camera.fx = camera.fy = 0.9f * kWidth;
    camera.cx = 0.5f * kWidth;
    camera.cy = 0.5f * kHeight;
    const glm::vec3 eye = center + glm::normalize(glm::vec3(0.25f, 0.35f, 1.0f)) * (2.2f * radius);
    camera.cameraToWorld = glm::inverse(glm::lookAt(eye, center, glm::vec3(0.0f, 1.0f, 0.0f)));
    return camera;
}

struct ImageDiff
{
    float maxAbs = 0.0f;
    size_t differing = 0;
    double meanAlpha = 0.0;
};

ImageDiff compareImages(const std::vector<float> & a, const std::vector<float> & b)
{
    ImageDiff d;
    for (size_t i = 0; i < a.size(); ++i)
    {
        const float diff = std::abs(a[i] - b[i]);
        d.maxAbs = std::max(d.maxAbs, diff);
        d.differing += diff != 0.0f ? 1 : 0;
        if (i % 4 == 3)
            d.meanAlpha += a[i];
    }
    d.meanAlpha /= static_cast<double>(a.size() / 4);
    return d;
}

std::vector<float> render(Renderer & renderer, const std::string & path,
                          const Camera * camera = nullptr)
{
    renderer.loadScene(path.c_str());
    if (camera == nullptr)
    {
        const SceneBounds & bounds = renderer.sceneBounds();
        renderer.setCamera(makeCamera(bounds.focusCenter, bounds.focusRadius));
    }
    else
    {
        renderer.setCamera(*camera);
    }
    renderer.forward();
    return renderer.readImage();
}

void crossFormat(Renderer & renderer, Device & device, const std::string & plyPath,
                 const std::string & label)
{
    auto cloud = loadScene(device, plyPath);
    const std::string glbPath = tempPath("cross_" + label + ".glb");
    convertToGlb(*cloud, glbPath);
    auto back = loadScene(device, glbPath);
    check(sameCloud(*cloud, *back),
          label + ": .ply -> .glb -> loadScene is bit-identical in every stream ("
              + std::to_string(cloud->count()) + " Gaussians, SH degree "
              + std::to_string(back->shDegree()) + ")");
    const std::vector<float> fromPly = render(renderer, plyPath);
    const SceneBounds plyBounds = renderer.sceneBounds();
    const Camera camera = makeCamera(plyBounds.focusCenter, plyBounds.focusRadius);
    const std::vector<float> fromGlb = render(renderer, glbPath, &camera);
    const ImageDiff d = compareImages(fromPly, fromGlb);
    check(d.maxAbs == 0.0f && d.meanAlpha > 0.05,
          label + ": .ply and converted .glb render bit-identically (max |diff| " + num(d.maxAbs)
              + ", " + std::to_string(d.differing) + " differing floats, mean coverage "
              + num(d.meanAlpha) + ")");
}

// Rendering M*object with camera C must equal rendering the object with
// camera M^-1 * C. That is the physical meaning of the node transform --
// positions, covariances AND the SH rotation -- checked through the shaders,
// with nothing from the reader's own math on the reference side.
void transformEquivalence(Renderer & renderer, const std::string & label,
                          const std::string & nodeJson, const glm::dmat4 & m, int degree)
{
    GltfWriter local;
    GltfWriter world;
    Expected e1, e2;
    std::vector<std::pair<std::string, int>> a1 = addFloatPrimitive(local, e1, 300, degree, 21);
    std::vector<std::pair<std::string, int>> a2 = addFloatPrimitive(world, e2, 300, degree, 21);
    local.addMesh({primitive(a1)});
    local.addNode("\"mesh\":0");
    local.sceneNodes = "[0]";
    world.addMesh({primitive(a2)});
    world.addNode("\"mesh\":0," + nodeJson);
    world.sceneNodes = "[0]";
    local.writeGlb(tempPath("equiv_local.glb"));
    world.writeGlb(tempPath("equiv_world.glb"));

    renderer.loadScene(tempPath("equiv_world.glb").c_str());
    const SceneBounds bounds = renderer.sceneBounds();
    const Camera camera = makeCamera(bounds.center(), bounds.radius());
    Camera localCamera = camera;
    localCamera.cameraToWorld = glm::mat4(glm::inverse(m) * glm::dmat4(camera.cameraToWorld));

    const std::vector<float> worldImage = render(renderer, tempPath("equiv_world.glb"), &camera);
    const std::vector<float> localImage =
        render(renderer, tempPath("equiv_local.glb"), &localCamera);
    const ImageDiff d = compareImages(worldImage, localImage);
    check(d.maxAbs <= 1e-5f && d.meanAlpha > 0.05,
          label + ": node transform renders as the inverse camera move (max |diff| " + num(d.maxAbs)
              + ", " + std::to_string(d.differing) + " of " + std::to_string(worldImage.size())
              + " floats differ, mean coverage " + num(d.meanAlpha) + ")");
}

void testRendering(Device & device, const std::string & assetPly)
{
    std::printf("\n-- (d) rendering through Renderer --\n");
    Renderer renderer(kWidth, kHeight);
    if (!renderer.initialize({}, true, SHADER_DIR))
    {
        check(false, "renderer initialize");
        return;
    }
    renderer.setMaxSHDegree(3);

    const std::string synthetic = tempPath("synthetic_deg3.ply");
    writeDegree3Ply(synthetic, 2000, 5);
    crossFormat(renderer, device, synthetic, "synthetic degree-3");

    if (std::filesystem::exists(assetPly))
    {
        crossFormat(
            renderer, device, assetPly, std::filesystem::path(assetPly).filename().string());
    }
    else
    {
        std::printf("[SKIP] %s not found -- real-asset cross-format check not run\n",
                    assetPly.c_str());
    }

    // Rotation + uniform scale + translation, degree 3: exercises the SH
    // rotation through the real shader.
    const double qx = 0.3, qy = -0.4, qz = 0.1;
    const double qw = std::sqrt(1.0 - qx * qx - qy * qy - qz * qz);
    const DMat3 r = quatToMat(qx, qy, qz, qw);
    glm::dmat4 m(1.0);
    for (int c = 0; c < 3; ++c)
        for (int row = 0; row < 3; ++row)
            m[c][row] = 1.7 * r[row][c];
    m[3] = glm::dvec4(0.5, -1.0, 2.0, 1.0);
    transformEquivalence(renderer,
                         "rotation+uniform scale, SH degree 3",
                         "\"translation\":[0.5,-1,2],\"rotation\":[" + num(qx) + "," + num(qy) + ","
                             + num(qz) + "," + num(qw) + "],\"scale\":[1.7,1.7,1.7]",
                         m,
                         3);

    // Non-uniform scale, degree 0 (under non-uniform scale view directions are
    // not a pure rotation, so only the geometry is comparable this way).
    glm::dmat4 n(1.0);
    for (int c = 0; c < 3; ++c)
        for (int row = 0; row < 3; ++row)
            n[c][row] = r[row][c] * (c == 0 ? 0.6 : c == 1 ? 1.4 : 2.1);
    transformEquivalence(renderer,
                         "rotation+non-uniform scale, SH degree 0",
                         "\"rotation\":[" + num(qx) + "," + num(qy) + "," + num(qz) + "," + num(qw)
                             + "],\"scale\":[0.6,1.4,2.1]",
                         n,
                         0);
    renderer.shutdown();
}

} // namespace

int main(int argc, char ** argv)
{
    const std::string assetPly = argc > 1 ? argv[1] : "assets/bonsai-7k.ply";
    g_tempDir = std::filesystem::temp_directory_path() / "sabercat_gltf_reader_test";
    std::filesystem::create_directories(g_tempDir);

    try
    {
        Device device;
        device.initialize({}, /*enableValidation=*/true);
        testDecode(device);
        testTransforms(device);
        testRejections(device);
        testSpzCompression(device);
        testRendering(device, assetPly);
        device.shutdown();
    }
    catch (const std::exception & e)
    {
        std::printf("[FAIL] unexpected exception: %s\n", e.what());
        ++g_failures;
    }

    std::filesystem::remove_all(g_tempDir);
    std::printf("\n%s (%d failure%s)\n",
                g_failures == 0 ? "gltf_reader_test OK" : "gltf_reader_test FAILED",
                g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}
