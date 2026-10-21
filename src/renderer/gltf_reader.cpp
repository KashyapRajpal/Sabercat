// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/gltf_reader.h"

#include <glm/glm.hpp>
#include <load-spz.h>
#include <splat-types.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "sabercat/gaussian_cloud.h"
#include "sabercat/logger.h"
#include "sabercat/parallel_for.h"
#include "sabercat/render_config.h"
#include "sabercat/vk_check.h"
#include "sh_rotation.h"

// glTF buffers are little-endian by definition, and accessors are decoded with
// memcpy straight from them.
static_assert(std::endian::native == std::endian::little,
              "GltfReader assumes a little-endian host (glTF data is little-endian)");

namespace sabercat
{

namespace
{

using Json = nlohmann::json;

// --- KHR_gaussian_splatting (ratified) ---
constexpr const char * kExtensionName = "KHR_gaussian_splatting";
constexpr const char * kAttributePrefix = "KHR_gaussian_splatting:";
constexpr const char * kShAttributePrefix = "KHR_gaussian_splatting:SH_DEGREE_";

// --- KHR_gaussian_splatting_compression_spz_2 ---
// Nested in the KHR_gaussian_splatting object's own `extensions`, not on the
// primitive: { "bufferView": n } whose bytes are one gzipped SPZ (NGSP)
// stream. It replaces every splat attribute; their accessors are then
// placeholders (typically no bufferView) whose count must match the stream.
constexpr const char * kSpzExtensionName = "KHR_gaussian_splatting_compression_spz_2";

// The extension's conformance section: pack/unpack with Left-Up-Front, the
// frame spz calls "GLB". It is spz's RUB (what SpzReader asks for) with x and
// z negated -- spz's notion, not a property of the glTF axes -- so the same
// stream reads 180 degrees about Y apart as a .spz file and inside a glTF.
constexpr spz::CoordinateSystem kSpzGltfCoordinateSystem = spz::CoordinateSystem::LUF;

// Extensions this reader can honour when a file lists them in
// extensionsRequired. KHR_mesh_quantization only widens POSITION's allowed
// component types, which the generic accessor decode already handles.
constexpr std::array<const char *, 3> kSupportedRequiredExtensions = {
    "KHR_gaussian_splatting", "KHR_mesh_quantization", kSpzExtensionName};

// --- glTF 2.0 enums ---
constexpr int kComponentByte = 5120;
constexpr int kComponentUnsignedByte = 5121;
constexpr int kComponentShort = 5122;
constexpr int kComponentUnsignedShort = 5123;
constexpr int kComponentUnsignedInt = 5125;
constexpr int kComponentFloat = 5126;

constexpr int kModePoints = 0;
constexpr int kModeDefault = 4; // TRIANGLES -- what an absent `mode` means

// --- .glb container ---
constexpr uint32_t kGlbMagic = 0x46546C67; // "glTF"
constexpr uint32_t kGlbVersion = 2;
constexpr uint32_t kGlbChunkJson = 0x4E4F534A; // "JSON"
constexpr uint32_t kGlbChunkBin = 0x004E4942;  // "BIN\0"
constexpr size_t kGlbHeaderSize = 12;
constexpr size_t kGlbChunkHeaderSize = 8;

// bufferView.byteStride range from the glTF 2.0 schema.
constexpr int64_t kMinByteStride = 4;
constexpr int64_t kMaxByteStride = 252;

// An accessor's count is attacker-controlled and, for a sparse-only accessor
// (no bufferView), not bounded by any buffer -- cap it before allocating.
constexpr int64_t kMaxAccessorCount = int64_t{1} << 28;

constexpr int kMaxShDegree = 3; // the canonical layout holds 16 coefficients

// A float quaternion is passed through bit-exact when it is already unit
// length to within float rounding (the spec says it is), and renormalized
// otherwise. Normalized-integer quaternions are always renormalized: the
// quantization alone moves them off the unit sphere.
constexpr float kUnitQuaternionTolerance = 1e-6f;
constexpr float kMinQuaternionLengthSquared = 1e-12f;

// The spec's decomposability rule: the column-normalized upper 3x3 must have a
// determinant "close to positive one". With unit columns |det| <= 1, with
// equality only for an orthonormal basis (Hadamard), so this one number
// rejects both shear and mirroring.
constexpr double kDecomposableDeterminantTolerance = 1e-3;
// Column lengths this close count as a uniform scale, which keeps the exact
// quaternion-composition path instead of re-diagonalizing each covariance.
constexpr double kUniformScaleTolerance = 1e-6;

// The SH band rotation and the rotation math it shares with the transform
// decomposition live in sh_rotation.h (layer_transform.cpp uses them too).
using sh_rotation::BandMatrix;
using sh_rotation::Mat3;
using sh_rotation::determinant;
using sh_rotation::matrixToQuaternion;
using sh_rotation::quaternionToMatrix;
using sh_rotation::shBandRotation;


[[noreturn]] void fail(const std::string & path, const std::string & what)
{
    throw std::runtime_error("GltfReader: " + what + ": " + path);
}

// ---------------------------------------------------------------------------
// File + container
// ---------------------------------------------------------------------------

std::vector<uint8_t> readWholeFile(const std::filesystem::path & file, const std::string & path)
{
    std::ifstream stream(file, std::ios::binary | std::ios::ate);
    if (!stream.is_open())
    {
        fail(path, "failed to open '" + file.string() + "'");
    }
    const std::streamsize size = stream.tellg();
    stream.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytes(static_cast<size_t>(std::max<std::streamsize>(size, 0)));
    if (size > 0 && !stream.read(reinterpret_cast<char *>(bytes.data()), size))
    {
        fail(path, "truncated read of '" + file.string() + "'");
    }
    return bytes;
}

uint32_t readU32(const uint8_t * bytes)
{
    uint32_t value;
    std::memcpy(&value, bytes, sizeof(value));
    return value;
}

struct Container
{
    std::string jsonText;
    const uint8_t * bin = nullptr; // GLB BIN chunk, if any (points into the file bytes)
    size_t binSize = 0;
};

// A .glb is a 12-byte header then length-prefixed chunks: JSON first
// (required), BIN second (optional); unknown chunk types are skipped, per spec.
Container parseGlb(const std::vector<uint8_t> & file, const std::string & path)
{
    Container container;
    if (file.size() < kGlbHeaderSize + kGlbChunkHeaderSize)
    {
        fail(path, "file is too small to be a .glb (" + std::to_string(file.size()) + " bytes)");
    }
    const uint32_t version = readU32(file.data() + 4);
    const uint32_t length = readU32(file.data() + 8);
    if (version != kGlbVersion)
    {
        fail(path,
             "unsupported .glb container version " + std::to_string(version)
                 + " (only 2 is defined)");
    }
    if (length != file.size())
    {
        fail(path,
             ".glb header length " + std::to_string(length) + " does not match the file size "
                 + std::to_string(file.size()));
    }

    size_t offset = kGlbHeaderSize;
    bool first = true;
    while (offset + kGlbChunkHeaderSize <= file.size())
    {
        const uint64_t chunkLength = readU32(file.data() + offset);
        const uint32_t chunkType = readU32(file.data() + offset + 4);
        const size_t dataOffset = offset + kGlbChunkHeaderSize;
        if (dataOffset + chunkLength > file.size())
        {
            fail(path,
                 ".glb chunk at byte " + std::to_string(offset) + " claims "
                     + std::to_string(chunkLength) + " bytes, past the end of the file");
        }
        if (first)
        {
            if (chunkType != kGlbChunkJson)
            {
                fail(path, "the first .glb chunk must be JSON");
            }
            container.jsonText.assign(reinterpret_cast<const char *>(file.data() + dataOffset),
                                      static_cast<size_t>(chunkLength));
            first = false;
        }
        else if (chunkType == kGlbChunkBin && container.bin == nullptr)
        {
            container.bin = file.data() + dataOffset;
            container.binSize = static_cast<size_t>(chunkLength);
        }
        offset = dataOffset + static_cast<size_t>(chunkLength);
    }
    if (first)
    {
        fail(path, ".glb has no JSON chunk");
    }
    return container;
}

// ---------------------------------------------------------------------------
// Buffers (GLB BIN chunk, data: URIs, external files)
// ---------------------------------------------------------------------------

bool base64Value(char c, uint8_t & value)
{
    if (c >= 'A' && c <= 'Z')
        value = static_cast<uint8_t>(c - 'A');
    else if (c >= 'a' && c <= 'z')
        value = static_cast<uint8_t>(c - 'a' + 26);
    else if (c >= '0' && c <= '9')
        value = static_cast<uint8_t>(c - '0' + 52);
    else if (c == '+')
        value = 62;
    else if (c == '/')
        value = 63;
    else
        return false;
    return true;
}

std::vector<uint8_t> decodeBase64(const std::string & text, size_t begin, const std::string & path)
{
    std::vector<uint8_t> out;
    out.reserve((text.size() - begin) / 4 * 3);
    uint32_t accumulator = 0;
    int bits = 0;
    for (size_t i = begin; i < text.size(); ++i)
    {
        const char c = text[i];
        if (c == '=')
        {
            break; // padding ends the payload
        }
        uint8_t value = 0;
        if (!base64Value(c, value))
        {
            fail(path, "invalid character in a base64 data URI");
        }
        accumulator = (accumulator << 6) | value;
        bits += 6;
        if (bits >= 8)
        {
            bits -= 8;
            out.push_back(static_cast<uint8_t>((accumulator >> bits) & 0xFF));
        }
    }
    return out;
}

// glTF URIs are RFC 3986 references: a relative file name may be
// percent-encoded (e.g. a space as %20).
std::string percentDecode(const std::string & uri)
{
    std::string out;
    out.reserve(uri.size());
    for (size_t i = 0; i < uri.size(); ++i)
    {
        if (uri[i] == '%' && i + 2 < uri.size()
            && std::isxdigit(static_cast<unsigned char>(uri[i + 1]))
            && std::isxdigit(static_cast<unsigned char>(uri[i + 2])))
        {
            out.push_back(static_cast<char>(std::stoi(uri.substr(i + 1, 2), nullptr, 16)));
            i += 2;
        }
        else
        {
            out.push_back(uri[i]);
        }
    }
    return out;
}

struct BufferData
{
    const uint8_t * data = nullptr;
    size_t size = 0; // the buffer's declared byteLength (validated <= what was read)
};

struct Document
{
    Json root;
    std::vector<std::vector<uint8_t>> ownedBuffers; // decoded data URIs / external files
    std::vector<BufferData> buffers;
    std::string path;
};

void loadBuffers(Document & doc, const Container & container, const std::filesystem::path & baseDir)
{
    if (!doc.root.contains("buffers"))
    {
        return;
    }
    const Json & buffers = doc.root.at("buffers");
    doc.ownedBuffers.resize(buffers.size());
    doc.buffers.resize(buffers.size());
    for (size_t i = 0; i < buffers.size(); ++i)
    {
        const Json & buffer = buffers.at(i);
        const std::string where = "buffer " + std::to_string(i);
        const int64_t byteLength = buffer.at("byteLength").get<int64_t>();
        if (byteLength < 1)
        {
            fail(doc.path, where + " has byteLength " + std::to_string(byteLength));
        }

        const uint8_t * data = nullptr;
        size_t available = 0;
        if (!buffer.contains("uri") || buffer.at("uri").is_null())
        {
            // Only the first buffer of a .glb may omit its uri: it IS the BIN chunk.
            if (i != 0 || container.bin == nullptr)
            {
                fail(doc.path, where + " has no uri and there is no .glb BIN chunk for it");
            }
            data = container.bin;
            available = container.binSize;
        }
        else
        {
            const std::string uri = buffer.at("uri").get<std::string>();
            if (uri.rfind("data:", 0) == 0)
            {
                const size_t marker = uri.find(";base64,");
                if (marker == std::string::npos)
                {
                    fail(doc.path, where + " has a data: URI that is not base64");
                }
                doc.ownedBuffers[i] = decodeBase64(uri, marker + 8, doc.path);
            }
            else if (uri.find("://") != std::string::npos)
            {
                fail(doc.path,
                     where + " references a remote URI ('" + uri
                         + "'); only data: URIs and local files are read");
            }
            else
            {
                doc.ownedBuffers[i] =
                    readWholeFile(baseDir / std::filesystem::u8path(percentDecode(uri)), doc.path);
            }
            data = doc.ownedBuffers[i].data();
            available = doc.ownedBuffers[i].size();
        }

        if (static_cast<uint64_t>(byteLength) > available)
        {
            fail(doc.path,
                 where + " declares byteLength " + std::to_string(byteLength) + " but only "
                     + std::to_string(available) + " bytes are present");
        }
        doc.buffers[i] = {data, static_cast<size_t>(byteLength)};
    }
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

const Json & arrayElement(const Document & doc, const char * arrayName, int64_t index,
                          const std::string & referrer)
{
    if (!doc.root.contains(arrayName) || index < 0
        || static_cast<uint64_t>(index) >= doc.root.at(arrayName).size())
    {
        fail(doc.path,
             referrer + " references " + arrayName + "[" + std::to_string(index)
                 + "], which does not exist");
    }
    return doc.root.at(arrayName).at(static_cast<size_t>(index));
}

size_t componentSize(int componentType)
{
    switch (componentType)
    {
    case kComponentByte:
    case kComponentUnsignedByte:
        return 1;
    case kComponentShort:
    case kComponentUnsignedShort:
        return 2;
    case kComponentUnsignedInt:
    case kComponentFloat:
        return 4;
    default:
        return 0;
    }
}

std::string componentName(int componentType, bool normalized)
{
    std::string name;
    switch (componentType)
    {
    case kComponentByte:
        name = "signed byte";
        break;
    case kComponentUnsignedByte:
        name = "unsigned byte";
        break;
    case kComponentShort:
        name = "signed short";
        break;
    case kComponentUnsignedShort:
        name = "unsigned short";
        break;
    case kComponentUnsignedInt:
        name = "unsigned int";
        break;
    case kComponentFloat:
        name = "float";
        break;
    default:
        name = "component type " + std::to_string(componentType);
        break;
    }
    return normalized ? name + " normalized" : name;
}

// The glTF 2.0 normalized-integer decode (f = max(c / 127.0, -1.0) etc.):
// signed types are clamped so -128/-32768 map to -1, not below it.
float decodeComponent(const uint8_t * bytes, int componentType, bool normalized)
{
    switch (componentType)
    {
    case kComponentFloat:
    {
        float value;
        std::memcpy(&value, bytes, sizeof(value));
        return value;
    }
    case kComponentByte:
    {
        const int8_t value = static_cast<int8_t>(bytes[0]);
        return normalized ? std::max(static_cast<float>(value) / 127.0f, -1.0f)
                          : static_cast<float>(value);
    }
    case kComponentUnsignedByte:
        return normalized ? static_cast<float>(bytes[0]) / 255.0f : static_cast<float>(bytes[0]);
    case kComponentShort:
    {
        int16_t value;
        std::memcpy(&value, bytes, sizeof(value));
        return normalized ? std::max(static_cast<float>(value) / 32767.0f, -1.0f)
                          : static_cast<float>(value);
    }
    case kComponentUnsignedShort:
    {
        uint16_t value;
        std::memcpy(&value, bytes, sizeof(value));
        return normalized ? static_cast<float>(value) / 65535.0f : static_cast<float>(value);
    }
    default:
    {
        uint32_t value;
        std::memcpy(&value, bytes, sizeof(value));
        return static_cast<float>(value);
    }
    }
}

struct AllowedFormat
{
    int componentType;
    bool normalized;
};

// The spec's per-attribute component-type tables.
const std::vector<AllowedFormat> kPositionFormats = {
    // Base glTF: float. KHR_mesh_quantization adds the integer types.
    {kComponentFloat, false},
    {kComponentByte, false},
    {kComponentByte, true},
    {kComponentUnsignedByte, false},
    {kComponentUnsignedByte, true},
    {kComponentShort, false},
    {kComponentShort, true},
    {kComponentUnsignedShort, false},
    {kComponentUnsignedShort, true}};
const std::vector<AllowedFormat> kRotationFormats = {
    {kComponentFloat, false}, {kComponentByte, true}, {kComponentShort, true}};
const std::vector<AllowedFormat> kScaleFormats = {{kComponentFloat, false},
                                                  {kComponentUnsignedByte, false},
                                                  {kComponentUnsignedByte, true},
                                                  {kComponentUnsignedShort, false},
                                                  {kComponentUnsignedShort, true}};
const std::vector<AllowedFormat> kOpacityFormats = {
    {kComponentFloat, false}, {kComponentUnsignedByte, true}, {kComponentUnsignedShort, true}};
const std::vector<AllowedFormat> kShFormats = {{kComponentFloat, false}};

int componentsForType(const std::string & type)
{
    if (type == "SCALAR")
        return 1;
    if (type == "VEC2")
        return 2;
    if (type == "VEC3")
        return 3;
    if (type == "VEC4")
        return 4;
    return 0;
}

// Header-level facts about an accessor, checked once in the planning pass so
// the decode pass can trust them.
struct AccessorInfo
{
    int64_t index = -1;
    int64_t count = 0;
    int componentType = 0;
    int components = 0;
    bool normalized = false;
};

AccessorInfo inspectAccessor(const Document & doc, int64_t index, const char * expectedType,
                             const std::vector<AllowedFormat> & allowed,
                             const std::string & semantic)
{
    const std::string where = "attribute '" + semantic + "'";
    const Json & accessor = arrayElement(doc, "accessors", index, where);

    AccessorInfo info;
    info.index = index;
    info.componentType = accessor.at("componentType").get<int>();
    info.normalized = accessor.value("normalized", false);
    info.count = accessor.at("count").get<int64_t>();
    const std::string type = accessor.at("type").get<std::string>();
    info.components = componentsForType(type);

    if (type != expectedType)
    {
        fail(doc.path,
             where + " must be a " + expectedType + " accessor, but accessor "
                 + std::to_string(index) + " is " + type);
    }
    const bool formatAllowed = std::any_of(
        allowed.begin(),
        allowed.end(),
        [&](const AllowedFormat & f)
        { return f.componentType == info.componentType && f.normalized == info.normalized; });
    if (!formatAllowed)
    {
        std::string list;
        for (const AllowedFormat & f : allowed)
        {
            list += (list.empty() ? "" : ", ") + componentName(f.componentType, f.normalized);
        }
        fail(doc.path,
             where + " uses " + componentName(info.componentType, info.normalized)
                 + ", which KHR_gaussian_splatting does not allow (allowed: " + list + ")");
    }
    if (info.count < 1 || info.count > kMaxAccessorCount)
    {
        fail(doc.path,
             where + ": accessor " + std::to_string(index) + " has count "
                 + std::to_string(info.count));
    }
    return info;
}

// A bufferView resolved against its buffer, with the bounds already checked.
struct ViewBytes
{
    const uint8_t * data = nullptr;
    uint64_t length = 0;
    int64_t byteStride = 0; // 0 = not specified (tightly packed)
};

ViewBytes resolveView(const Document & doc, int64_t viewIndex, const std::string & where)
{
    const Json & view = arrayElement(doc, "bufferViews", viewIndex, where);
    const std::string viewWhere = "bufferView " + std::to_string(viewIndex);
    const int64_t bufferIndex = view.at("buffer").get<int64_t>();
    if (bufferIndex < 0 || static_cast<uint64_t>(bufferIndex) >= doc.buffers.size())
    {
        fail(doc.path,
             viewWhere + " references buffer " + std::to_string(bufferIndex)
                 + ", which does not exist");
    }
    const BufferData & buffer = doc.buffers[static_cast<size_t>(bufferIndex)];
    const int64_t byteOffset = view.value("byteOffset", int64_t{0});
    const int64_t byteLength = view.at("byteLength").get<int64_t>();
    if (byteOffset < 0 || byteLength < 1
        || static_cast<uint64_t>(byteOffset) + static_cast<uint64_t>(byteLength) > buffer.size)
    {
        fail(doc.path,
             viewWhere + " (offset " + std::to_string(byteOffset) + ", length "
                 + std::to_string(byteLength) + ") is out of bounds of buffer "
                 + std::to_string(bufferIndex) + " (" + std::to_string(buffer.size) + " bytes)");
    }
    ViewBytes bytes;
    bytes.data = buffer.data + byteOffset;
    bytes.length = static_cast<uint64_t>(byteLength);
    bytes.byteStride = view.value("byteStride", int64_t{0});
    if (view.contains("byteStride")
        && (bytes.byteStride < kMinByteStride || bytes.byteStride > kMaxByteStride
            || bytes.byteStride % 4 != 0))
    {
        fail(doc.path,
             viewWhere + " has byteStride " + std::to_string(bytes.byteStride)
                 + " (must be a multiple of 4 in [4, 252])");
    }
    return bytes;
}

// Decodes a whole accessor to floats (count * components), applying the
// normalized-integer rule, byteStride and sparse substitution. Every byte
// range is bounds-checked before it is read.
std::vector<float> decodeAccessor(const Document & doc, const AccessorInfo & info,
                                  const std::string & semantic)
{
    const std::string where =
        "attribute '" + semantic + "' (accessor " + std::to_string(info.index) + ")";
    const Json & accessor = doc.root.at("accessors").at(static_cast<size_t>(info.index));
    const size_t components = static_cast<size_t>(info.components);
    const size_t count = static_cast<size_t>(info.count);
    const size_t elementSize = components * componentSize(info.componentType);

    // No bufferView: the spec says the accessor is all zeros (then sparse).
    std::vector<float> values(count * components, 0.0f);

    if (accessor.contains("bufferView"))
    {
        const ViewBytes view = resolveView(doc, accessor.at("bufferView").get<int64_t>(), where);
        const int64_t byteOffset = accessor.value("byteOffset", int64_t{0});
        const uint64_t stride =
            view.byteStride != 0 ? static_cast<uint64_t>(view.byteStride) : elementSize;
        if (stride < elementSize)
        {
            fail(doc.path,
                 where + ": byteStride " + std::to_string(stride)
                     + " is smaller than the element size " + std::to_string(elementSize));
        }
        // count <= 2^28 and stride <= 252, so this cannot overflow.
        if (byteOffset < 0
            || static_cast<uint64_t>(byteOffset) + stride * (count - 1) + elementSize > view.length)
        {
            fail(doc.path,
                 where + " reads past the end of its bufferView (offset "
                     + std::to_string(byteOffset) + ", " + std::to_string(count)
                     + " elements, stride " + std::to_string(stride) + ", view length "
                     + std::to_string(view.length) + ")");
        }
        const size_t step = componentSize(info.componentType);
        const uint8_t * base = view.data + byteOffset;
        for (size_t i = 0; i < count; ++i)
        {
            const uint8_t * element = base + i * stride;
            for (size_t c = 0; c < components; ++c)
            {
                values[i * components + c] =
                    decodeComponent(element + c * step, info.componentType, info.normalized);
            }
        }
    }

    if (accessor.contains("sparse"))
    {
        const Json & sparse = accessor.at("sparse");
        const int64_t sparseCount = sparse.at("count").get<int64_t>();
        if (sparseCount < 1 || sparseCount > info.count)
        {
            fail(doc.path,
                 where + ": sparse count " + std::to_string(sparseCount) + " is outside [1, "
                     + std::to_string(info.count) + "]");
        }
        const Json & indices = sparse.at("indices");
        const Json & sparseValues = sparse.at("values");
        const int indexType = indices.at("componentType").get<int>();
        if (indexType != kComponentUnsignedByte && indexType != kComponentUnsignedShort
            && indexType != kComponentUnsignedInt)
        {
            fail(doc.path, where + ": sparse indices must be unsigned byte/short/int");
        }
        const size_t indexSize = componentSize(indexType);
        const ViewBytes indexView =
            resolveView(doc, indices.at("bufferView").get<int64_t>(), where + " sparse indices");
        const ViewBytes valueView = resolveView(
            doc, sparseValues.at("bufferView").get<int64_t>(), where + " sparse values");
        const int64_t indexOffset = indices.value("byteOffset", int64_t{0});
        const int64_t valueOffset = sparseValues.value("byteOffset", int64_t{0});
        const uint64_t n = static_cast<uint64_t>(sparseCount);
        if (indexOffset < 0
            || static_cast<uint64_t>(indexOffset) + n * indexSize > indexView.length)
        {
            fail(doc.path, where + ": sparse indices read past the end of their bufferView");
        }
        if (valueOffset < 0
            || static_cast<uint64_t>(valueOffset) + n * elementSize > valueView.length)
        {
            fail(doc.path, where + ": sparse values read past the end of their bufferView");
        }

        // Sparse indices and values are tightly packed (the spec forbids a
        // byteStride on their views).
        int64_t previous = -1;
        for (uint64_t s = 0; s < n; ++s)
        {
            const int64_t target = static_cast<int64_t>(
                decodeComponent(indexView.data + indexOffset + s * indexSize, indexType, false));
            if (target <= previous || target >= info.count)
            {
                fail(doc.path,
                     where + ": sparse index " + std::to_string(target) + " at position "
                         + std::to_string(s) + " is out of range or not strictly increasing");
            }
            previous = target;
            const uint8_t * element = valueView.data + valueOffset + s * elementSize;
            for (size_t c = 0; c < components; ++c)
            {
                values[static_cast<size_t>(target) * components + c] =
                    decodeComponent(element + c * componentSize(info.componentType),
                                    info.componentType,
                                    info.normalized);
            }
        }
    }
    return values;
}

// ---------------------------------------------------------------------------
// Rotation / transform math (double; row-major Mat3)
// ---------------------------------------------------------------------------

// Symmetric 3x3 eigendecomposition by cyclic Jacobi: a = v diag(w) v^T, with
// the eigenvectors as the COLUMNS of v. Only used for non-uniformly scaled
// node transforms, where M R S no longer factors as rotation * diagonal.
void symmetricEigen(Mat3 a, Mat3 & v, std::array<double, 3> & w)
{
    v = {{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}};
    constexpr int kMaxSweeps = 50;
    constexpr int kPairs[3][2] = {{0, 1}, {0, 2}, {1, 2}};
    for (int sweep = 0; sweep < kMaxSweeps; ++sweep)
    {
        const double offDiagonal = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
        const double diagonal = a[0][0] * a[0][0] + a[1][1] * a[1][1] + a[2][2] * a[2][2];
        if (offDiagonal <= 1e-30 * diagonal || offDiagonal == 0.0)
        {
            break;
        }
        for (const auto & pair : kPairs)
        {
            const int p = pair[0];
            const int q = pair[1];
            const double apq = a[p][q];
            if (apq == 0.0)
            {
                continue;
            }
            // A' = J^T A J with J the (p,q) plane rotation that zeroes a[p][q].
            const double theta = (a[q][q] - a[p][p]) / (2.0 * apq);
            const double t =
                (theta >= 0.0 ? 1.0 : -1.0) / (std::abs(theta) + std::sqrt(theta * theta + 1.0));
            const double c = 1.0 / std::sqrt(t * t + 1.0);
            const double s = t * c;
            for (int k = 0; k < 3; ++k)
            {
                const double akp = a[k][p];
                const double akq = a[k][q];
                a[k][p] = c * akp - s * akq;
                a[k][q] = s * akp + c * akq;
            }
            for (int k = 0; k < 3; ++k)
            {
                const double apk = a[p][k];
                const double aqk = a[q][k];
                a[p][k] = c * apk - s * aqk;
                a[q][k] = s * apk + c * aqk;
            }
            for (int k = 0; k < 3; ++k)
            {
                const double vkp = v[k][p];
                const double vkq = v[k][q];
                v[k][p] = c * vkp - s * vkq;
                v[k][q] = s * vkp + c * vkq;
            }
        }
    }
    w = {a[0][0], a[1][1], a[2][2]};
}

// A node's global transform, decomposed once per instance.
struct InstanceTransform
{
    std::array<BandMatrix, 3> shBands{}; // bands 1..3
    glm::dmat4 matrix{1.0};
    Mat3 linear{};   // upper 3x3 of `matrix`
    Mat3 rotation{}; // its rotation factor, exactly orthonormal
    std::array<double, 4> rotationQuaternion{0.0, 0.0, 0.0, 1.0}; // xyzw
    double uniformScale = 1.0;
    bool identity = true;
    bool uniform = true;
};

InstanceTransform decomposeTransform(const glm::dmat4 & matrix, int64_t nodeIndex,
                                     const std::string & path)
{
    InstanceTransform transform;
    transform.matrix = matrix;
    transform.identity = (matrix == glm::dmat4(1.0));
    if (transform.identity)
    {
        return transform;
    }

    const std::string where = "node " + std::to_string(nodeIndex);
    // glm is column-major: matrix[col][row].
    std::array<double, 3> lengths{};
    for (int c = 0; c < 3; ++c)
    {
        for (int r = 0; r < 3; ++r)
        {
            transform.linear[r][c] = matrix[c][r];
        }
        lengths[c] = std::sqrt(matrix[c][0] * matrix[c][0] + matrix[c][1] * matrix[c][1]
                               + matrix[c][2] * matrix[c][2]);
        if (!std::isfinite(lengths[c]) || lengths[c] <= 0.0)
        {
            fail(path,
                 where
                     + "'s global transform has a zero or non-finite scale column -- "
                       "KHR_gaussian_splatting requires finite positive column lengths");
        }
    }
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            if (!std::isfinite(matrix[i][j]))
            {
                fail(path, where + "'s global transform is not finite");
            }
        }
    }

    Mat3 normalized{};
    for (int r = 0; r < 3; ++r)
    {
        for (int c = 0; c < 3; ++c)
        {
            normalized[r][c] = transform.linear[r][c] / lengths[c];
        }
    }
    const double det = determinant(normalized);
    if (std::abs(det - 1.0) > kDecomposableDeterminantTolerance)
    {
        fail(path,
             where
                 + "'s global transform is not decomposable into translation, rotation and "
                   "positive scale (column-normalized determinant "
                 + std::to_string(det) + (det < 0.0 ? ": a mirroring/negative scale" : ": a shear")
                 + "), which KHR_gaussian_splatting leaves undefined");
    }

    // Round-trip through a unit quaternion so `rotation` is orthonormal to
    // double precision, whatever float noise the authored matrix carried.
    transform.rotationQuaternion = matrixToQuaternion(normalized);
    const auto & q = transform.rotationQuaternion;
    transform.rotation = quaternionToMatrix(q[0], q[1], q[2], q[3]);

    const double minLength = std::min({lengths[0], lengths[1], lengths[2]});
    const double maxLength = std::max({lengths[0], lengths[1], lengths[2]});
    transform.uniform = (maxLength - minLength) <= kUniformScaleTolerance * maxLength;
    transform.uniformScale = (lengths[0] + lengths[1] + lengths[2]) / 3.0;

    for (int band = 1; band <= kMaxShDegree; ++band)
    {
        transform.shBands[band - 1] = shBandRotation(band, transform.rotation);
    }
    return transform;
}

glm::dmat4 nodeLocalMatrix(const Json & node)
{
    if (node.contains("matrix"))
    {
        const std::vector<double> m = node.at("matrix").get<std::vector<double>>();
        if (m.size() != 16)
        {
            throw std::runtime_error("node matrix must have 16 elements");
        }
        glm::dmat4 matrix;
        for (int c = 0; c < 4; ++c)
        {
            for (int r = 0; r < 4; ++r)
            {
                matrix[c][r] = m[static_cast<size_t>(c * 4 + r)]; // glTF is column-major too
            }
        }
        return matrix;
    }

    const std::vector<double> t = node.value("translation", std::vector<double>{0.0, 0.0, 0.0});
    const std::vector<double> q = node.value("rotation", std::vector<double>{0.0, 0.0, 0.0, 1.0});
    const std::vector<double> s = node.value("scale", std::vector<double>{1.0, 1.0, 1.0});
    if (t.size() != 3 || q.size() != 4 || s.size() != 3)
    {
        throw std::runtime_error("node translation/rotation/scale has the wrong element count");
    }
    // glTF composes T * R * S; node rotations are unit xyzw quaternions.
    const Mat3 r = quaternionToMatrix(q[0], q[1], q[2], q[3]);
    glm::dmat4 matrix(1.0);
    for (int c = 0; c < 3; ++c)
    {
        for (int row = 0; row < 3; ++row)
        {
            matrix[c][row] = r[row][c] * s[static_cast<size_t>(c)];
        }
        matrix[3][c] = t[static_cast<size_t>(c)];
    }
    return matrix;
}

// ---------------------------------------------------------------------------
// Planning: which primitives, under which transforms, with what data
// ---------------------------------------------------------------------------

struct SplatPrimitive
{
    // One accessor per SH coefficient, band-ascending (index 0 = DC). Empty
    // for an SPZ-compressed primitive.
    std::vector<AccessorInfo> sh;
    // KHR_gaussian_splatting_compression_spz_2: the decoded stream, already
    // in the glTF frame. Non-null means the accessors below are unused.
    std::shared_ptr<const spz::GaussianCloud> spz;
    AccessorInfo position;
    AccessorInfo rotation;
    AccessorInfo scale;
    AccessorInfo opacity;
    size_t transformIndex = 0;
    size_t first = 0; // first Gaussian index in the output cloud
    int64_t count = 0;
    int meshIndex = 0;
    int primitiveIndex = 0;
    int shDegree = 0;
};

std::string shSemantic(int degree, int coefficient)
{
    return std::string(kShAttributePrefix) + std::to_string(degree) + "_COEF_"
           + std::to_string(coefficient);
}

int64_t attributeIndex(const Json & attributes, const std::string & semantic)
{
    return attributes.contains(semantic) ? attributes.at(semantic).get<int64_t>() : -1;
}

// KHR_gaussian_splatting_compression_spz_2: decodes the primitive's SPZ
// stream now (planning needs its point count) and checks the placeholder
// accessors against it. Their type/component checks are skipped: they
// describe data this reader never reads from them.
void planSpzPrimitive(const Document & doc, const Json & spzExtension, const Json & attributes,
                      const std::string & where, SplatPrimitive & out)
{
    const std::string spzWhere = where + " " + kSpzExtensionName;
    if (!spzExtension.is_object() || !spzExtension.contains("bufferView"))
    {
        fail(doc.path, spzWhere + " has no 'bufferView' (it must name the SPZ stream's view)");
    }
    const int64_t viewIndex = spzExtension.at("bufferView").get<int64_t>();
    const ViewBytes view = resolveView(doc, viewIndex, spzWhere);

    // Like SpzReader, but from memory: loadSpz reports failure as an empty
    // cloud rather than throwing.
    spz::UnpackOptions options;
    options.to = kSpzGltfCoordinateSystem;
    auto source = std::make_shared<spz::GaussianCloud>(
        spz::loadSpz(view.data, static_cast<size_t>(view.length), options));
    if (source->numPoints <= 0)
    {
        fail(doc.path,
             spzWhere + ": bufferView " + std::to_string(viewIndex) + " ("
                 + std::to_string(view.length)
                 + " bytes) did not decode to any Gaussians -- not a gzipped SPZ (NGSP) stream, "
                   "an unsupported SPZ version, corrupt/truncated, or empty");
    }
    const size_t count = static_cast<size_t>(source->numPoints);
    const int sourceDegree = source->shDegree;
    if (sourceDegree < 0)
    {
        fail(doc.path, spzWhere + ": SPZ stream has SH degree " + std::to_string(sourceDegree));
    }

    // Guard the decode pass's indexing, as SpzReader does.
    auto checkStream = [&](const char * name, size_t got, size_t expected)
    {
        if (got != expected)
        {
            fail(doc.path,
                 spzWhere + ": decoded " + name + " stream has " + std::to_string(got)
                     + " floats, expected " + std::to_string(expected) + " for "
                     + std::to_string(count) + " Gaussians");
        }
    };
    const size_t shPerChannel =
        static_cast<size_t>((sourceDegree + 1) * (sourceDegree + 1) - 1);
    checkStream("position", source->positions.size(), count * 3);
    checkStream("scale", source->scales.size(), count * 3);
    checkStream("rotation", source->rotations.size(), count * 4);
    checkStream("alpha", source->alphas.size(), count);
    checkStream("color", source->colors.size(), count * 3);
    checkStream("SH", source->sh.size(), count * shPerChannel * 3);

    // Same truncation as SpzReader: the bands are a band-ascending prefix.
    if (sourceDegree > kMaxShDegree)
    {
        Logger::warning("GltfReader: " + spzWhere + " carries SH degree "
                        + std::to_string(sourceDegree) + "; degree "
                        + std::to_string(kMaxShDegree) + " is kept, the top band(s) dropped");
    }

    // "The accessor count must match the number of points in the compressed
    // SPZ data."
    for (const auto & item : attributes.items())
    {
        const int64_t index = item.value().get<int64_t>();
        const Json & accessor =
            arrayElement(doc, "accessors", index, "attribute '" + item.key() + "'");
        const int64_t accessorCount = accessor.at("count").get<int64_t>();
        if (accessorCount != source->numPoints)
        {
            fail(doc.path,
                 where + ": attribute '" + item.key() + "' (accessor " + std::to_string(index)
                     + ") has count " + std::to_string(accessorCount) + " but the "
                     + kSpzExtensionName + " stream holds " + std::to_string(count)
                     + " Gaussians");
        }
    }

    out.count = source->numPoints;
    out.shDegree = std::min(sourceDegree, kMaxShDegree);
    out.spz = std::move(source);
}

// Validates one primitive against the spec and resolves its accessors.
// Returns false (with no error) if the primitive does not use the extension.
bool planPrimitive(const Document & doc, const Json & primitive, int meshIndex, int primitiveIndex,
                   SplatPrimitive & out)
{
    if (!primitive.contains("extensions") || !primitive.at("extensions").contains(kExtensionName))
    {
        return false;
    }
    const std::string where =
        "mesh " + std::to_string(meshIndex) + " primitive " + std::to_string(primitiveIndex);
    const Json & extension = primitive.at("extensions").at(kExtensionName);
    const bool compressed =
        extension.contains("extensions") && extension.at("extensions").contains(kSpzExtensionName);

    // --- Extension object ---
    // SPZ-compressed glTF in the wild (Cesium's 3D Tiles output) predates the
    // ratified base spec and omits kernel/colorSpace; for those, default to
    // what the draft implied rather than refuse the whole asset.
    auto requiredOrDraftDefault = [&](const char * name, const char * draftDefault)
    {
        if (extension.contains(name))
        {
            return extension.at(name).get<std::string>();
        }
        if (!compressed)
        {
            fail(doc.path,
                 where + ": KHR_gaussian_splatting is missing its required '" + name + "'");
        }
        Logger::warning("GltfReader: " + where + " (SPZ-compressed) has no '" + name
                        + "' (pre-ratification output); assuming \"" + draftDefault + "\"");
        return std::string(draftDefault);
    };
    const std::string kernel = requiredOrDraftDefault("kernel", "ellipse");
    if (kernel != "ellipse")
    {
        // A kernel extension may redefine what the attributes mean, so
        // rendering it as an ellipse would be a plausible-but-wrong image.
        fail(
            doc.path,
            where + ": unsupported kernel '" + kernel + "' (Sabercat implements only \"ellipse\")");
    }
    const std::string colorSpace = requiredOrDraftDefault("colorSpace", "srgb_rec709_display");
    if (colorSpace == "srgb_rec709_display")
    {
        // What 3DGS trains in (sRGB-encoded photos) and what every other
        // format Sabercat loads carries -- the same treatment as a .ply.
    }
    else if (colorSpace == "lin_rec709_display")
    {
        Logger::info("GltfReader: " + where
                     + " is lin_rec709_display -- SH passed through unconverted (a transfer "
                       "function cannot be applied to SH coefficients; the spec requires "
                       "blending before any conversion)");
    }
    else
    {
        Logger::warning("GltfReader: " + where + " uses unknown colorSpace '" + colorSpace
                        + "'; SH passed through unconverted");
    }
    const std::string projection = extension.value("projection", std::string("perspective"));
    if (projection != "perspective")
    {
        fail(doc.path,
             where + ": unsupported projection '" + projection
                 + "' (Sabercat implements only \"perspective\")");
    }
    const std::string sorting = extension.value("sortingMethod", std::string("cameraDistance"));
    if (sorting != "cameraDistance")
    {
        Logger::warning("GltfReader: " + where + " requests sortingMethod '" + sorting
                        + "'; Sabercat sorts per tile by view depth regardless");
    }
    if (extension.contains("extensions") && !extension.at("extensions").empty())
    {
        // Not in extensionsRequired (that was rejected up front), so by the
        // spec's rules the base data is a valid fallback.
        for (const auto & item : extension.at("extensions").items())
        {
            if (item.key() == kSpzExtensionName)
            {
                continue;
            }
            Logger::warning("GltfReader: " + where + " extends KHR_gaussian_splatting with '"
                            + item.key() + "', which is ignored (base attributes used)");
        }
    }

    // --- Primitive ---
    const int mode = primitive.value("mode", kModeDefault);
    if (mode != kModePoints)
    {
        fail(doc.path,
             where + ": mode is " + std::to_string(mode)
                 + "; the ellipse kernel requires POINTS (0)");
    }
    if (primitive.contains("indices"))
    {
        fail(doc.path, where + ": indexed splat primitives are not supported");
    }
    if (primitive.contains("targets"))
    {
        fail(doc.path, where + ": morph targets on a splat primitive are not supported");
    }
    if (!primitive.contains("attributes"))
    {
        fail(doc.path, where + ": primitive has no attributes");
    }
    const Json & attributes = primitive.at("attributes");

    auto require = [&](const std::string & semantic) -> int64_t
    {
        const int64_t index = attributeIndex(attributes, semantic);
        if (index < 0)
        {
            fail(doc.path, where + ": missing required attribute '" + semantic + "'");
        }
        return index;
    };

    out.meshIndex = meshIndex;
    out.primitiveIndex = primitiveIndex;
    if (compressed)
    {
        // The conformance rule: a loader that supports the extension takes
        // the data from it first, whatever the accessors carry.
        planSpzPrimitive(
            doc, extension.at("extensions").at(kSpzExtensionName), attributes, where, out);
        return true;
    }
    out.position = inspectAccessor(doc, require("POSITION"), "VEC3", kPositionFormats, "POSITION");
    out.rotation = inspectAccessor(doc,
                                   require(std::string(kAttributePrefix) + "ROTATION"),
                                   "VEC4",
                                   kRotationFormats,
                                   std::string(kAttributePrefix) + "ROTATION");
    out.scale = inspectAccessor(doc,
                                require(std::string(kAttributePrefix) + "SCALE"),
                                "VEC3",
                                kScaleFormats,
                                std::string(kAttributePrefix) + "SCALE");
    out.opacity = inspectAccessor(doc,
                                  require(std::string(kAttributePrefix) + "OPACITY"),
                                  "SCALAR",
                                  kOpacityFormats,
                                  std::string(kAttributePrefix) + "OPACITY");

    // --- SH: degree discovered from which bands are present, all-or-nothing ---
    out.sh.push_back(
        inspectAccessor(doc, require(shSemantic(0, 0)), "VEC3", kShFormats, shSemantic(0, 0)));
    int degree = 0;
    for (int band = 1; band <= kMaxShDegree; ++band)
    {
        int present = 0;
        for (int m = 0; m < 2 * band + 1; ++m)
        {
            present += attributeIndex(attributes, shSemantic(band, m)) >= 0 ? 1 : 0;
        }
        if (present == 0)
        {
            for (int higher = band + 1; higher <= kMaxShDegree; ++higher)
            {
                if (attributeIndex(attributes, shSemantic(higher, 0)) >= 0)
                {
                    fail(doc.path,
                         where + ": SH degree " + std::to_string(higher)
                             + " is present without degree " + std::to_string(band)
                             + " (lower degrees MUST be defined)");
                }
            }
            break;
        }
        if (present != 2 * band + 1)
        {
            fail(doc.path,
                 where + ": SH degree " + std::to_string(band) + " is partially defined ("
                     + std::to_string(present) + " of " + std::to_string(2 * band + 1)
                     + " coefficients) -- the spec forbids partial degrees");
        }
        for (int m = 0; m < 2 * band + 1; ++m)
        {
            out.sh.push_back(inspectAccessor(doc,
                                             attributeIndex(attributes, shSemantic(band, m)),
                                             "VEC3",
                                             kShFormats,
                                             shSemantic(band, m)));
        }
        degree = band;
    }
    out.shDegree = degree;

    // Anything else under our prefix is unknown to this version of the spec
    // (e.g. a degree-4 band): say so instead of silently dropping it.
    for (const auto & item : attributes.items())
    {
        const std::string & name = item.key();
        if (name.rfind(kAttributePrefix, 0) != 0)
        {
            continue;
        }
        bool known = name == std::string(kAttributePrefix) + "ROTATION"
                     || name == std::string(kAttributePrefix) + "SCALE"
                     || name == std::string(kAttributePrefix) + "OPACITY";
        for (int band = 0; band <= degree && !known; ++band)
        {
            for (int m = 0; m < 2 * band + 1 && !known; ++m)
            {
                known = name == shSemantic(band, m);
            }
        }
        if (!known)
        {
            Logger::warning("GltfReader: " + where + " attribute '" + name
                            + "' is not used (beyond SH degree 3 or unknown)");
        }
    }

    // --- Every attribute describes the same Gaussians ---
    out.count = out.position.count;
    auto sameCount = [&](const AccessorInfo & info, const std::string & semantic)
    {
        if (info.count != out.count)
        {
            fail(doc.path,
                 where + ": attribute '" + semantic + "' has " + std::to_string(info.count)
                     + " elements but POSITION has " + std::to_string(out.count));
        }
    };
    sameCount(out.rotation, std::string(kAttributePrefix) + "ROTATION");
    sameCount(out.scale, std::string(kAttributePrefix) + "SCALE");
    sameCount(out.opacity, std::string(kAttributePrefix) + "OPACITY");
    for (size_t k = 0; k < out.sh.size(); ++k)
    {
        sameCount(out.sh[k], "SH coefficient " + std::to_string(k));
    }
    return true;
}

struct Instance
{
    glm::dmat4 global;
    int64_t mesh;
    int64_t node; // -1: a mesh with no node at all
};

void collectInstances(const Document & doc, int64_t nodeIndex, const glm::dmat4 & parent,
                      std::vector<bool> & visited, std::vector<Instance> & out)
{
    const Json & node = arrayElement(doc, "nodes", nodeIndex, "the node hierarchy");
    if (visited[static_cast<size_t>(nodeIndex)])
    {
        fail(doc.path,
             "node " + std::to_string(nodeIndex)
                 + " is reached twice (a cycle or a shared child) -- glTF requires a tree");
    }
    visited[static_cast<size_t>(nodeIndex)] = true;

    glm::dmat4 local;
    try
    {
        local = nodeLocalMatrix(node);
    }
    catch (const std::runtime_error & e)
    {
        fail(doc.path, "node " + std::to_string(nodeIndex) + ": " + e.what());
    }
    const glm::dmat4 global = parent * local;
    if (node.contains("mesh"))
    {
        if (node.contains("skin"))
        {
            Logger::warning("GltfReader: node " + std::to_string(nodeIndex)
                            + " is skinned; the skin is ignored (rest pose only)");
        }
        out.push_back({global, node.at("mesh").get<int64_t>(), nodeIndex});
    }
    if (node.contains("children"))
    {
        for (const Json & child : node.at("children"))
        {
            collectInstances(doc, child.get<int64_t>(), global, visited, out);
        }
    }
}

// The scene glTF says to show: `scene`, else scene 0. A file with nodes but
// no scenes gets every root node; one with meshes but no nodes gets each mesh
// once, untransformed (nothing else could ever display it).
std::vector<Instance> instantiateScene(const Document & doc)
{
    std::vector<Instance> instances;
    const size_t nodeCount = doc.root.contains("nodes") ? doc.root.at("nodes").size() : 0;
    std::vector<bool> visited(nodeCount, false);

    if (doc.root.contains("scenes") && !doc.root.at("scenes").empty())
    {
        const int64_t sceneIndex = doc.root.value("scene", int64_t{0});
        const Json & scene = arrayElement(doc, "scenes", sceneIndex, "the default scene");
        if (scene.contains("nodes"))
        {
            for (const Json & root : scene.at("nodes"))
            {
                collectInstances(doc, root.get<int64_t>(), glm::dmat4(1.0), visited, instances);
            }
        }
    }
    else if (nodeCount > 0)
    {
        std::vector<bool> isChild(nodeCount, false);
        for (const Json & node : doc.root.at("nodes"))
        {
            if (node.contains("children"))
            {
                for (const Json & child : node.at("children"))
                {
                    const int64_t c = child.get<int64_t>();
                    if (c >= 0 && static_cast<size_t>(c) < nodeCount)
                    {
                        isChild[static_cast<size_t>(c)] = true;
                    }
                }
            }
        }
        for (size_t n = 0; n < nodeCount; ++n)
        {
            if (!isChild[n])
            {
                collectInstances(doc, static_cast<int64_t>(n), glm::dmat4(1.0), visited, instances);
            }
        }
    }
    else if (doc.root.contains("meshes"))
    {
        for (size_t m = 0; m < doc.root.at("meshes").size(); ++m)
        {
            instances.push_back({glm::dmat4(1.0), static_cast<int64_t>(m), -1});
        }
    }
    return instances;
}

// ---------------------------------------------------------------------------
// Decode pass
// ---------------------------------------------------------------------------

// SPZ fields restated as the float arrays decodeAccessor() yields for the
// equivalent uncompressed attributes (the extension's attribute mapping), so
// validation, node transform and SH rotation are shared by both paths. spz
// hands back pre-activation values, like .ply.

std::vector<float> spzScales(const spz::GaussianCloud & source)
{
    std::vector<float> values(source.scales.size());
    for (size_t i = 0; i < values.size(); ++i)
    {
        values[i] = std::exp(source.scales[i]); // log-space -> linear
    }
    return values;
}

std::vector<float> spzOpacities(const spz::GaussianCloud & source)
{
    std::vector<float> values(source.alphas.size());
    for (size_t i = 0; i < values.size(); ++i)
    {
        values[i] = 1.0f / (1.0f + std::exp(-source.alphas[i])); // pre-sigmoid -> [0,1]
    }
    return values;
}

// Coefficient k of the canonical band-ascending layout: k = 0 is `colors`
// (already the raw f_dc, which is exactly SH_DEGREE_0_COEF_0), k >= 1 reads
// spz's coefficient-major, channel-inner `sh`.
std::vector<float> spzShCoefficient(const spz::GaussianCloud & source, int k)
{
    const size_t count = static_cast<size_t>(source.numPoints);
    if (k == 0)
    {
        return source.colors;
    }
    const size_t perPoint =
        static_cast<size_t>((source.shDegree + 1) * (source.shDegree + 1) - 1) * 3;
    std::vector<float> values(count * 3);
    for (size_t i = 0; i < count; ++i)
    {
        for (size_t c = 0; c < 3; ++c)
        {
            values[i * 3 + c] = source.sh[i * perPoint + static_cast<size_t>(k - 1) * 3 + c];
        }
    }
    return values;
}

void decodePrimitive(const Document & doc, const SplatPrimitive & prim,
                     const InstanceTransform & transform, GaussianCloud & cloud)
{
    const std::string where = "mesh " + std::to_string(prim.meshIndex) + " primitive "
                              + std::to_string(prim.primitiveIndex);
    const size_t count = static_cast<size_t>(prim.count);
    const size_t first = prim.first;
    auto & positions = cloud.positions();
    auto & scales = cloud.scales();
    auto & rotations = cloud.rotations();
    auto & opacities = cloud.opacities();
    auto & sh = cloud.shCoefficients();
    const spz::GaussianCloud * spzSource = prim.spz.get(); // null: read the accessors

    // --- Positions ---
    {
        const std::vector<float> values =
            spzSource ? spzSource->positions : decodeAccessor(doc, prim.position, "POSITION");
        for (size_t i = 0; i < count * 3; ++i)
        {
            if (!std::isfinite(values[i]))
            {
                fail(doc.path, where + ": POSITION " + std::to_string(i / 3) + " is not finite");
            }
        }
        const glm::dmat4 & m = transform.matrix;
        const bool identity = transform.identity;
        parallelFor(count,
                    [&](size_t begin, size_t end)
                    {
                        for (size_t i = begin; i < end; ++i)
                        {
                            const float * p = &values[i * 3];
                            if (identity)
                            {
                                positions[first + i] = glm::vec4(p[0], p[1], p[2], 0.0f);
                            }
                            else
                            {
                                const glm::dvec4 world = m * glm::dvec4(p[0], p[1], p[2], 1.0);
                                positions[first + i] = glm::vec4(static_cast<float>(world.x),
                                                                 static_cast<float>(world.y),
                                                                 static_cast<float>(world.z),
                                                                 0.0f);
                            }
                        }
                    });
    }

    // --- Scale + rotation (together: a node transform mixes them) ---
    {
        const std::string scaleName = std::string(kAttributePrefix) + "SCALE";
        const std::string rotationName = std::string(kAttributePrefix) + "ROTATION";
        const std::vector<float> scaleValues =
            spzSource ? spzScales(*spzSource) : decodeAccessor(doc, prim.scale, scaleName);
        std::vector<float> rotationValues = spzSource
                                                ? spzSource->rotations
                                                : decodeAccessor(doc, prim.rotation, rotationName);
        for (size_t i = 0; i < count * 3; ++i)
        {
            // "Scale values are linear and MUST NOT be negative."
            if (!std::isfinite(scaleValues[i]) || scaleValues[i] < 0.0f)
            {
                fail(doc.path,
                     where + ": " + scaleName + " " + std::to_string(i / 3) + " is "
                         + std::to_string(scaleValues[i]) + " (must be finite and non-negative)");
            }
        }
        // SPZ's smallest-three quaternions are quantized too (as SpzReader
        // notes, the round trip does not guarantee unit length).
        const bool quantized = spzSource || prim.rotation.componentType != kComponentFloat;
        for (size_t i = 0; i < count; ++i)
        {
            float * q = &rotationValues[i * 4];
            const float lengthSquared = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
            if (!std::isfinite(lengthSquared) || lengthSquared < kMinQuaternionLengthSquared)
            {
                fail(doc.path,
                     where + ": " + rotationName + " " + std::to_string(i)
                         + " is zero-length or not finite (must be a unit quaternion)");
            }
            // Bit-exact passthrough for an already-unit float quaternion, so
            // a lossless export round-trips exactly.
            if (quantized || std::abs(lengthSquared - 1.0f) > kUnitQuaternionTolerance)
            {
                const float invLength = 1.0f / std::sqrt(lengthSquared);
                for (int c = 0; c < 4; ++c)
                {
                    q[c] *= invLength;
                }
            }
        }

        const InstanceTransform & t = transform;
        parallelFor(
            count,
            [&](size_t begin, size_t end)
            {
                for (size_t i = begin; i < end; ++i)
                {
                    const float * s = &scaleValues[i * 3];
                    const float * q = &rotationValues[i * 4]; // xyzw -- already canonical order
                    if (t.identity)
                    {
                        scales[first + i] = glm::vec4(s[0], s[1], s[2], 0.0f);
                        rotations[first + i] = glm::vec4(q[0], q[1], q[2], q[3]);
                        continue;
                    }
                    if (t.uniform)
                    {
                        // M R S = (s_node R_node) R S: compose the rotations,
                        // multiply the scales. Exact, no decomposition.
                        const auto & a = t.rotationQuaternion;
                        const double bx = q[0], by = q[1], bz = q[2], bw = q[3];
                        double x = a[3] * bx + a[0] * bw + a[1] * bz - a[2] * by;
                        double y = a[3] * by - a[0] * bz + a[1] * bw + a[2] * bx;
                        double z = a[3] * bz + a[0] * by - a[1] * bx + a[2] * bw;
                        double w = a[3] * bw - a[0] * bx - a[1] * by - a[2] * bz;
                        const double inv = 1.0 / std::sqrt(x * x + y * y + z * z + w * w);
                        rotations[first + i] = glm::vec4(static_cast<float>(x * inv),
                                                         static_cast<float>(y * inv),
                                                         static_cast<float>(z * inv),
                                                         static_cast<float>(w * inv));
                        scales[first + i] = glm::vec4(static_cast<float>(t.uniformScale * s[0]),
                                                      static_cast<float>(t.uniformScale * s[1]),
                                                      static_cast<float>(t.uniformScale * s[2]),
                                                      0.0f);
                        continue;
                    }
                    // Non-uniform node scale: Sigma' = (M R S)(M R S)^T no
                    // longer factors as rotation * diagonal directly, so
                    // re-diagonalize it (the spec's Sigma = M C C^T M^T).
                    const Mat3 r = quaternionToMatrix(q[0], q[1], q[2], q[3]);
                    Mat3 a{};
                    for (int row = 0; row < 3; ++row)
                    {
                        for (int col = 0; col < 3; ++col)
                        {
                            double sum = 0.0;
                            for (int k = 0; k < 3; ++k)
                            {
                                sum += t.linear[row][k] * r[k][col];
                            }
                            a[row][col] = sum * s[col];
                        }
                    }
                    Mat3 sigma{};
                    for (int row = 0; row < 3; ++row)
                    {
                        for (int col = 0; col < 3; ++col)
                        {
                            sigma[row][col] = a[row][0] * a[col][0] + a[row][1] * a[col][1]
                                              + a[row][2] * a[col][2];
                        }
                    }
                    Mat3 v{};
                    std::array<double, 3> eigen{};
                    symmetricEigen(sigma, v, eigen);
                    if (determinant(v) < 0.0)
                    {
                        for (int row = 0; row < 3; ++row)
                        {
                            v[row][2] = -v[row][2]; // a proper rotation, not a reflection
                        }
                    }
                    const std::array<double, 4> rq = matrixToQuaternion(v);
                    rotations[first + i] = glm::vec4(static_cast<float>(rq[0]),
                                                     static_cast<float>(rq[1]),
                                                     static_cast<float>(rq[2]),
                                                     static_cast<float>(rq[3]));
                    scales[first + i] =
                        glm::vec4(static_cast<float>(std::sqrt(std::max(eigen[0], 0.0))),
                                  static_cast<float>(std::sqrt(std::max(eigen[1], 0.0))),
                                  static_cast<float>(std::sqrt(std::max(eigen[2], 0.0))),
                                  0.0f);
                }
            });
    }

    // --- Opacity: already activated, [0,1] ---
    {
        const std::string name = std::string(kAttributePrefix) + "OPACITY";
        const std::vector<float> values =
            spzSource ? spzOpacities(*spzSource) : decodeAccessor(doc, prim.opacity, name);
        for (size_t i = 0; i < count; ++i)
        {
            // "Out-of-range values are invalid." (NaN fails both comparisons.)
            if (!(values[i] >= 0.0f && values[i] <= 1.0f))
            {
                fail(doc.path,
                     where + ": " + name + " " + std::to_string(i) + " is "
                         + std::to_string(values[i]) + " (must be in [0, 1])");
            }
            opacities[first + i] = values[i];
        }
    }

    // --- SH: raw 3DGS coefficients, band-ascending, m-ascending ---
    const size_t shCoefficients = static_cast<size_t>((prim.shDegree + 1) * (prim.shDegree + 1));
    for (size_t k = 0; k < shCoefficients; ++k)
    {
        const std::vector<float> values =
            spzSource ? spzShCoefficient(*spzSource, static_cast<int>(k))
                      : decodeAccessor(doc, prim.sh[k], "SH coefficient " + std::to_string(k));
        for (size_t i = 0; i < count; ++i)
        {
            const float * c = &values[i * 3];
            if (!std::isfinite(c[0]) || !std::isfinite(c[1]) || !std::isfinite(c[2]))
            {
                fail(doc.path,
                     where + ": SH coefficient " + std::to_string(k) + " of Gaussian "
                         + std::to_string(i) + " is not finite");
            }
            sh[(first + i) * kShCoefficientsPerGaussian + k] = glm::vec4(c[0], c[1], c[2], 0.0f);
        }
    }

    // --- Rotate bands 1..degree by the node's rotation ---
    // Uniform scale and translation do not change view directions, so only
    // the rotation factor matters. (Under a NON-uniform scale directions are
    // not merely rotated; the spec only asks for the rotation component, and
    // that is what is applied.)
    if (!transform.identity && prim.shDegree > 0)
    {
        parallelFor(count,
                    [&](size_t begin, size_t end)
                    {
                        for (size_t i = begin; i < end; ++i)
                        {
                            sh_rotation::rotateShCoefficients(
                                &sh[(first + i) * kShCoefficientsPerGaussian],
                                prim.shDegree,
                                transform.shBands);
                        }
                    });
    }
}

std::shared_ptr<GaussianCloud> readDocument(Device & device, Document & doc,
                                            const Container & container,
                                            const std::filesystem::path & baseDir)
{
    // --- Asset + required extensions ---
    if (!doc.root.is_object() || !doc.root.contains("asset"))
    {
        fail(doc.path, "not a glTF document (no 'asset' object)");
    }
    const std::string version = doc.root.at("asset").at("version").get<std::string>();
    if (version.rfind("2.", 0) != 0)
    {
        fail(doc.path, "unsupported glTF asset version '" + version + "' (expected 2.x)");
    }
    if (doc.root.contains("extensionsRequired"))
    {
        for (const Json & required : doc.root.at("extensionsRequired"))
        {
            const std::string name = required.get<std::string>();
            const bool supported = std::any_of(kSupportedRequiredExtensions.begin(),
                                               kSupportedRequiredExtensions.end(),
                                               [&](const char * s) { return name == s; });
            if (!supported)
            {
                fail(doc.path,
                     "file requires extension '" + name
                         + "', which Sabercat does not implement (supported: "
                           "KHR_gaussian_splatting, KHR_mesh_quantization, "
                           "KHR_gaussian_splatting_compression_spz_2)");
            }
        }
    }

    loadBuffers(doc, container, baseDir);

    // --- Plan: instances -> splat primitives, all validated before decoding ---
    const std::vector<Instance> instances = instantiateScene(doc);
    std::vector<InstanceTransform> transforms;
    std::vector<SplatPrimitive> primitives;
    size_t total = 0;
    int cloudDegree = 0;
    for (const Instance & instance : instances)
    {
        const Json & mesh =
            arrayElement(doc, "meshes", instance.mesh, "node " + std::to_string(instance.node));
        if (!mesh.contains("primitives"))
        {
            continue;
        }
        const Json & meshPrimitives = mesh.at("primitives");
        size_t transformIndex = transforms.size();
        bool transformMade = false;
        for (size_t p = 0; p < meshPrimitives.size(); ++p)
        {
            SplatPrimitive prim;
            if (!planPrimitive(doc,
                               meshPrimitives.at(p),
                               static_cast<int>(instance.mesh),
                               static_cast<int>(p),
                               prim))
            {
                continue;
            }
            if (!transformMade)
            {
                transforms.push_back(decomposeTransform(instance.global, instance.node, doc.path));
                transformMade = true;
            }
            prim.transformIndex = transformIndex;
            prim.first = total;
            total += static_cast<size_t>(prim.count);
            cloudDegree = std::max(cloudDegree, prim.shDegree);
            primitives.push_back(std::move(prim));
        }
    }
    if (primitives.empty())
    {
        fail(doc.path, "no KHR_gaussian_splatting primitive is instantiated by the scene");
    }

    // --- Decode ---
    auto cloud = std::make_shared<GaussianCloud>(device);
    cloud->initialize(total); // zero-fills, so lower-degree primitives stay zero-padded
    for (const SplatPrimitive & prim : primitives)
    {
        decodePrimitive(doc, prim, transforms[prim.transformIndex], *cloud);
    }

    // glTF cannot declare the dilation opacity compensation; vanilla 3DGS
    // (what the spec's kernel describes) trains without it. An SPZ stream's
    // header can, though (see SpzReader) -- honoured when every primitive is
    // SPZ and agrees, since the flag is per cloud, not per Gaussian.
    const size_t flagged = static_cast<size_t>(
        std::count_if(primitives.begin(),
                      primitives.end(),
                      [](const SplatPrimitive & p) { return p.spz && p.spz->antialiased; }));
    if (flagged != 0 && flagged != primitives.size())
    {
        Logger::warning("GltfReader: " + std::to_string(flagged) + " of "
                        + std::to_string(primitives.size())
                        + " splat primitives are SPZ streams flagged antialiased; the cloud "
                          "can hold one setting, so it is loaded without the compensation");
    }
    cloud->setAntialiased(flagged == primitives.size());
    cloud->setShDegree(static_cast<uint32_t>(cloudDegree));
    cloud->computeBounds();
    return cloud;
}

} // namespace

std::shared_ptr<GaussianCloud> GltfReader::read(Device & device, const std::string & path)
{
    const std::filesystem::path filePath = std::filesystem::u8path(path);
    const std::vector<uint8_t> file = readWholeFile(filePath, path);
    if (file.empty())
    {
        fail(path, "file is empty");
    }

    Container container;
    const bool binary = file.size() >= 4 && readU32(file.data()) == kGlbMagic;
    if (binary)
    {
        container = parseGlb(file, path);
    }
    else
    {
        container.jsonText.assign(reinterpret_cast<const char *>(file.data()), file.size());
    }

    Document doc;
    doc.path = path;
    try
    {
        doc.root = Json::parse(container.jsonText);
        return readDocument(device, doc, container, filePath.parent_path());
    }
    catch (const Json::exception & e)
    {
        // Wrong JSON types or missing required members surface here; name the
        // file rather than leaking a bare nlohmann message.
        fail(path, std::string("malformed glTF JSON: ") + e.what());
    }
}

} // namespace sabercat
