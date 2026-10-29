// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/ply_reader.h"

#include <glm/gtc/quaternion.hpp>

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "sabercat/gaussian_cloud.h"
#include "sabercat/parallel_for.h"
#include "sabercat/render_config.h"
#include "sabercat/vk_check.h"

namespace sabercat
{

namespace
{

struct PlyProperty
{
    std::string name;
    std::string type; // as declared ("float", "uint", "uchar", ...)
    size_t offset;    // byte offset within one element record
    size_t size;      // byte size of this property's type
};

// One `element` block of the header, in file order. Element data is stored
// back to back after end_header, so an element's byte offset is the sum of
// the sizes of the elements declared before it.
struct PlyElement
{
    std::string name;
    std::unordered_map<std::string, PlyProperty> properties;
    size_t count = 0;
    size_t stride = 0;    // bytes per record (meaningless if hasList)
    bool hasList = false; // a `property list ...`: records are variable-size
};

// Byte offsets for the fixed set of properties PLYReader cares about,
// resolved once per file (not per vertex -- see the comment on read()).
struct ResolvedOffsets
{
    size_t x, y, z;
    size_t scale[3];
    size_t opacity;
    size_t rot[4];
    size_t fDc[3];
    std::vector<size_t> fRest;
    // R3DG relightable attributes, present only in R3DG files.
    bool relightable = false;
    size_t normal[3];
    size_t baseColor[3];
    size_t roughness;
    bool hasVisibility = false;
    size_t visibility[16]; // visibility_dc_0, visibility_rest_0..14
    // Optional metallic, read whenever present (independent of base_color).
    bool hasMetallic = false;
    size_t metallic;
};

// --- SuperSplat / PlayCanvas compressed PLY ---
// No written spec: defined by PlayCanvas's writer (splat-transform
// src/lib/writers/compressed-chunk.ts + write-compressed-ply.ts) and its
// readers (splat-transform src/lib/readers/decompress-ply.ts, engine
// src/scene/gsplat/gsplat-compressed-data.js). Splats come in chunks of 256
// sharing one `chunk` record of float bounds.
static constexpr size_t kCompressedChunkSize = 256;
static constexpr const char * kChunkBoundNames[12] = {
    "min_x",       "min_y",       "min_z",       "max_x",       "max_y",       "max_z",
    "min_scale_x", "min_scale_y", "min_scale_z", "max_scale_x", "max_scale_y", "max_scale_z",
};
static constexpr const char * kChunkColorNames[6] = {
    "min_r", "min_g", "min_b", "max_r", "max_g", "max_b"};
static constexpr const char * kPackedNames[4] = {
    "packed_position", "packed_rotation", "packed_scale", "packed_color"};
// f_rest count of the optional `sh` element for bands 1..3 (channel-major,
// like a plain .ply's f_rest).
static constexpr size_t kCompressedShRestCounts[3] = {9, 24, 45};
static constexpr double kSqrt2 = 1.4142135623730951;
static constexpr float kByteToUnit = 1.0f / 255.0f;
static constexpr float kInvShC0 = 1.0f / kShC0;

size_t plyTypeSize(const std::string & type)
{
    if (type == "float" || type == "float32" || type == "int" || type == "int32" || type == "uint"
        || type == "uint32")
    {
        return 4;
    }
    if (type == "double" || type == "float64")
    {
        return 8;
    }
    if (type == "uchar" || type == "uint8" || type == "char" || type == "int8")
    {
        return 1;
    }
    if (type == "short" || type == "int16" || type == "ushort" || type == "uint16")
    {
        return 2;
    }
    throw std::runtime_error("PLYReader: unsupported property type '" + type + "'");
}

std::string trimTrailingCR(std::string line)
{
    if (!line.empty() && line.back() == '\r')
    {
        line.pop_back();
    }
    return line;
}

size_t resolveOffset(const std::unordered_map<std::string, PlyProperty> & props,
                     const std::string & name)
{
    auto it = props.find(name);
    VK_VALIDATE(it != props.end(), "PLYReader: missing required property '" + name + "'");
    VK_VALIDATE(it->second.size == 4,
                "PLYReader: property '" + name + "' is not 4 bytes (expected float)");
    return it->second.offset;
}

// Offset of a compressed-PLY property, which must be declared with exactly
// one of `types` (the PLY spellings of the reference's type).
size_t resolveTypedOffset(const PlyElement & element, const std::string & name,
                          std::initializer_list<const char *> types, const char * expected)
{
    auto it = element.properties.find(name);
    VK_VALIDATE(it != element.properties.end(),
                "PLYReader: compressed PLY element '" + element.name + "' is missing property '"
                    + name + "'");
    const bool typeOk = std::any_of(
        types.begin(), types.end(), [&](const char * t) { return it->second.type == t; });
    VK_VALIDATE(typeOk,
                "PLYReader: compressed PLY property '" + element.name + "." + name + "' is '"
                    + it->second.type + "', expected " + expected);
    return it->second.offset;
}

const PlyElement * findElement(const std::vector<PlyElement> & elements, const std::string & name)
{
    for (const PlyElement & element : elements)
    {
        if (element.name == name)
        {
            return &element;
        }
    }
    return nullptr;
}

// Reads one element's records. The elements before it must be fixed-size
// for its offset to be known (true of every splat PLY).
std::vector<char> readElement(std::ifstream & file, std::streampos dataStart,
                              const std::vector<PlyElement> & elements, const PlyElement & target,
                              const std::string & path)
{
    size_t offset = 0;
    for (const PlyElement & element : elements)
    {
        if (&element == &target)
        {
            break;
        }
        VK_VALIDATE(!element.hasList,
                    "PLYReader: element '" + element.name
                        + "' has a list property and precedes element '" + target.name
                        + "'; its size is unknown: " + path);
        offset += element.count * element.stride;
    }
    VK_VALIDATE(
        !target.hasList,
        "PLYReader: list properties in element '" + target.name + "' are not supported: " + path);

    std::vector<char> raw(target.count * target.stride);
    file.clear();
    file.seekg(dataStart + static_cast<std::streamoff>(offset));
    file.read(raw.data(), static_cast<std::streamsize>(raw.size()));
    VK_VALIDATE(static_cast<size_t>(file.gcount()) == raw.size(),
                "PLYReader: truncated '" + target.name + "' element data in " + path);
    return raw;
}

inline float readFloatAt(const char * vertex, size_t offset)
{
    float value;
    std::memcpy(&value, vertex + offset, sizeof(float));
    return value;
}

inline uint32_t readUintAt(const char * record, size_t offset)
{
    uint32_t value;
    std::memcpy(&value, record + offset, sizeof(uint32_t));
    return value;
}

float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

// Reference 3DGS rotation storage is an unnormalized quaternion; normalizing
// it is a dot product + a scale, done here with SSE2 (guaranteed on x64, no
// runtime feature detection needed). Deliberately NOT using _mm_rsqrt_ps's
// fast approximate reciprocal -- this is opacity/scale-adjacent data in a
// differentiable renderer, and its ~12-bit precision isn't worth the risk
// for one scalar sqrt per Gaussian.
glm::quat normalizeQuatWXYZ(float w, float x, float y, float z)
{
    __m128 q = _mm_set_ps(z, y, x, w); // lane0=w, lane1=x, lane2=y, lane3=z
    __m128 sq = _mm_mul_ps(q, q);

    // Horizontal sum of the 4 squared lanes (SSE2 has no single-instruction
    // horizontal add).
    __m128 shuf = _mm_shuffle_ps(sq, sq, _MM_SHUFFLE(2, 3, 0, 1));
    __m128 sums = _mm_add_ps(sq, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);

    float lengthSquared = _mm_cvtss_f32(sums);
    float invLength = 1.0f / std::sqrt(lengthSquared);

    __m128 normalized = _mm_mul_ps(q, _mm_set1_ps(invLength));
    float out[4];
    _mm_storeu_ps(out, normalized);
    return glm::quat(out[0], out[1], out[2], out[3]); // (w, x, y, z)
}

// The reference's unpackUnorm: the low `bits` of `value` over [0, 1].
inline double unpackUnorm(uint32_t value, uint32_t bits)
{
    const uint32_t mask = (1u << bits) - 1u;
    return static_cast<double>(value & mask) / mask;
}

// The reference's lerp, in double as the JS computes it.
inline double lerp(double a, double b, double t)
{
    return a * (1.0 - t) + b * t;
}

// One chunk's bounds, widened to double once: position min xyz / max xyz,
// log-scale min xyz / max xyz, display-color min rgb / max rgb.
struct ChunkBounds
{
    double position[6];
    double scale[6];
    double color[6];
};

// An 11-10-11 field (x: bits 31..21, y: 20..11, z: 10..0), each unorm lerped
// over [bounds[k], bounds[3 + k]].
glm::dvec3 unpack111011(uint32_t value, const double bounds[6])
{
    return glm::dvec3(lerp(bounds[0], bounds[3], unpackUnorm(value >> 21, 11)),
                      lerp(bounds[1], bounds[4], unpackUnorm(value >> 11, 10)),
                      lerp(bounds[2], bounds[5], unpackUnorm(value, 11)));
}

// Smallest-three quaternion -> canonical xyzw. Lanes are in rot_0..rot_3
// (w, x, y, z) order. When the three stored lanes overshoot unit length the
// dropped one clamps to 0 and the length exceeds 1; otherwise it is 1 --
// never 0, so the renormalize is safe.
glm::vec4 unpackRotation(uint32_t value)
{
    const double stored[3] = {(unpackUnorm(value >> 20, 10) - 0.5) * kSqrt2,
                              (unpackUnorm(value >> 10, 10) - 0.5) * kSqrt2,
                              (unpackUnorm(value, 10) - 0.5) * kSqrt2};
    const uint32_t dropped = value >> 30;
    double q[4];
    for (uint32_t lane = 0, s = 0; lane < 4; ++lane)
    {
        q[lane] = lane == dropped ? 0.0 : stored[s++];
    }
    const double storedSquared =
        stored[0] * stored[0] + stored[1] * stored[1] + stored[2] * stored[2];
    q[dropped] = std::sqrt(std::max(0.0, 1.0 - storedSquared));
    const double invLength = 1.0 / std::sqrt(storedSquared + q[dropped] * q[dropped]);
    return glm::vec4(static_cast<float>(q[1] * invLength),
                     static_cast<float>(q[2] * invLength),
                     static_cast<float>(q[3] * invLength),
                     static_cast<float>(q[0] * invLength));
}

// Decodes a SuperSplat / PlayCanvas compressed PLY (see the constants above
// for provenance). Per splat, uint32 fields, bit 31 first:
//   packed_position  x:11 (31..21) y:10 (20..11) z:11 (10..0), each a unorm
//                    lerped over the chunk's [min_*, max_*]
//   packed_rotation  index:2 (31..30) of the dropped (largest-magnitude,
//                    made positive) lane of the (rot_0..rot_3 = w,x,y,z)
//                    quaternion; the other three lanes in ascending order as
//                    10-bit unorms a:29..20 b:19..10 c:9..0, each
//                    (u - 0.5) * sqrt(2); the dropped lane is sqrt(1 - a^2 -
//                    b^2 - c^2)
//   packed_scale     11-10-11 like position, lerped over the chunk's LOG
//                    scale bounds
//   packed_color     r:8 (31..24) g:8 b:8 a:8 (7..0); r,g,b are display
//                    color (0.5 + SH_C0 * f_dc) lerped over the chunk's
//                    [min_r.., max_r..] when the chunk has them (18 chunk
//                    properties), raw unorms otherwise (12); a is the
//                    ACTIVATED opacity, sigmoid(opacity), as a unorm
//   sh element       uchar f_rest_*, channel-major; byte b -> (n - 0.5) * 8
//                    with n = 0 / 1 at b = 0 / 255, else (b + 0.5) / 256
std::shared_ptr<GaussianCloud> readCompressed(Device & device, std::ifstream & file,
                                              std::streampos dataStart,
                                              const std::vector<PlyElement> & elements,
                                              const std::string & path)
{
    for (const PlyElement & element : elements)
    {
        VK_VALIDATE(element.name == "chunk" || element.name == "vertex" || element.name == "sh",
                    "PLYReader: compressed PLY has unsupported element '" + element.name
                        + "' (expected only chunk, vertex, sh): " + path);
    }
    const PlyElement * chunk = findElement(elements, "chunk");
    const PlyElement * vertex = findElement(elements, "vertex");
    const PlyElement * shElement = findElement(elements, "sh");
    VK_VALIDATE(vertex != nullptr && vertex->count > 0,
                "PLYReader: compressed PLY has no vertices: " + path);

    // 12 chunk properties: bounds only (older SuperSplat); 18: plus color
    // bounds. The reference accepts nothing else.
    const size_t chunkPropertyCount = chunk->properties.size();
    const bool hasChunkColor = chunkPropertyCount == 18;
    VK_VALIDATE(
        chunkPropertyCount == 12 || hasChunkColor,
        "PLYReader: compressed PLY chunk element has " + std::to_string(chunkPropertyCount)
            + " properties; only 12 (bounds) or 18 (bounds + min_r..max_b color bounds) are "
              "supported: "
            + path);
    size_t boundOffset[12];
    for (int i = 0; i < 12; ++i)
    {
        boundOffset[i] =
            resolveTypedOffset(*chunk, kChunkBoundNames[i], {"float", "float32"}, "float");
    }
    size_t colorOffset[6] = {};
    if (hasChunkColor)
    {
        for (int i = 0; i < 6; ++i)
        {
            colorOffset[i] =
                resolveTypedOffset(*chunk, kChunkColorNames[i], {"float", "float32"}, "float");
        }
    }

    size_t packedOffset[4];
    for (int i = 0; i < 4; ++i)
    {
        packedOffset[i] = resolveTypedOffset(*vertex, kPackedNames[i], {"uint", "uint32"}, "uint");
    }
    const size_t count = vertex->count;
    const size_t expectedChunks = (count + kCompressedChunkSize - 1) / kCompressedChunkSize;
    VK_VALIDATE(chunk->count == expectedChunks,
                "PLYReader: compressed PLY has " + std::to_string(chunk->count) + " chunks for "
                    + std::to_string(count) + " vertices (expected "
                    + std::to_string(expectedChunks) + ", one per 256): " + path);

    // Optional SH: 9 / 24 / 45 uchar f_rest_* for bands 1 / 2 / 3.
    uint32_t shBands = 0;
    std::vector<size_t> shOffsets;
    if (shElement != nullptr)
    {
        const size_t restCount = shElement->properties.size();
        for (uint32_t band = 0; band < 3; ++band)
        {
            if (restCount == kCompressedShRestCounts[band])
            {
                shBands = band + 1;
            }
        }
        VK_VALIDATE(shBands > 0,
                    "PLYReader: compressed PLY sh element has " + std::to_string(restCount)
                        + " properties; only 9, 24 or 45 (SH bands 1-3) are supported: " + path);
        VK_VALIDATE(shElement->count == count,
                    "PLYReader: compressed PLY sh element count ("
                        + std::to_string(shElement->count) + ") differs from the vertex count ("
                        + std::to_string(count) + "): " + path);
        shOffsets.resize(restCount);
        for (size_t k = 0; k < restCount; ++k)
        {
            shOffsets[k] = resolveTypedOffset(
                *shElement, "f_rest_" + std::to_string(k), {"uchar", "uint8"}, "uchar");
        }
    }
    const size_t shCoeffs = shOffsets.size() / 3; // per channel

    const std::vector<char> chunkRaw = readElement(file, dataStart, elements, *chunk, path);
    const std::vector<char> vertexRaw = readElement(file, dataStart, elements, *vertex, path);
    std::vector<char> shRaw;
    if (shElement != nullptr)
    {
        shRaw = readElement(file, dataStart, elements, *shElement, path);
    }

    // The 256 possible SH bytes, decoded once.
    float shTable[256];
    for (int b = 0; b < 256; ++b)
    {
        const double n = b == 0 ? 0.0 : b == 255 ? 1.0 : (b + 0.5) / 256.0;
        shTable[b] = static_cast<float>((n - 0.5) * 8.0);
    }

    // Without color bounds the color unorms are used as is -- the same as
    // lerping over [0, 1].
    std::vector<ChunkBounds> chunkBounds(chunk->count);
    for (size_t c = 0; c < chunk->count; ++c)
    {
        const char * record = chunkRaw.data() + c * chunk->stride;
        ChunkBounds & out = chunkBounds[c];
        for (int k = 0; k < 6; ++k)
        {
            out.position[k] = readFloatAt(record, boundOffset[k]);
            out.scale[k] = readFloatAt(record, boundOffset[6 + k]);
            out.color[k] =
                hasChunkColor ? readFloatAt(record, colorOffset[k]) : (k < 3 ? 0.0 : 1.0);
        }
    }

    auto cloud = std::make_shared<GaussianCloud>(device);
    cloud->initialize(count);
    auto & positions = cloud->positions();
    auto & scales = cloud->scales();
    auto & rotations = cloud->rotations();
    auto & sh = cloud->shCoefficients();
    auto & opacities = cloud->opacities();
    const char * vertexData = vertexRaw.data();
    const auto * shData = reinterpret_cast<const unsigned char *>(shRaw.data());
    const size_t vertexStride = vertex->stride;
    const size_t shStride = shElement != nullptr ? shElement->stride : 0;

    // Same disjoint-range contract as the plain path. Everything that could
    // fail was validated above, so the workers cannot throw.
    parallelFor(count,
                [&](size_t begin, size_t end)
                {
                    for (size_t i = begin; i < end; ++i)
                    {
                        const ChunkBounds & bounds = chunkBounds[i / kCompressedChunkSize];
                        const char * record = vertexData + i * vertexStride;

                        const glm::dvec3 position =
                            unpack111011(readUintAt(record, packedOffset[0]), bounds.position);
                        positions[i] = glm::vec4(glm::vec3(position), 0.0f);

                        rotations[i] = unpackRotation(readUintAt(record, packedOffset[1]));

                        // Bounds are LOG scales (the .ply's scale_*): exp() once.
                        const glm::dvec3 logScale =
                            unpack111011(readUintAt(record, packedOffset[2]), bounds.scale);
                        scales[i] = glm::vec4(glm::vec3(glm::exp(logScale)), 0.0f);

                        // Color is DISPLAY space, 0.5 + SH_C0 * f_dc, like
                        // .splat/.ksplat -- inverted to the true DC
                        // coefficient. Alpha is already sigmoid(opacity): the
                        // reference's logit would only be undone by the
                        // canonical sigmoid, so it is taken as is.
                        const uint32_t color = readUintAt(record, packedOffset[3]);
                        glm::vec4 * shOut = &sh[i * kShCoefficientsPerGaussian];
                        for (int c = 0; c < 3; ++c)
                        {
                            const double display = lerp(bounds.color[c],
                                                        bounds.color[3 + c],
                                                        unpackUnorm(color >> (24 - 8 * c), 8));
                            shOut[0][c] = static_cast<float>(display - 0.5) * kInvShC0;
                        }
                        opacities[i] = (color & 0xFFu) * kByteToUnit;

                        // Channel-major like a plain .ply's f_rest.
                        const unsigned char * shRecord = shData + i * shStride;
                        for (size_t k = 0; k < shCoeffs; ++k)
                        {
                            shOut[1 + k] = glm::vec4(shTable[shRecord[shOffsets[k]]],
                                                     shTable[shRecord[shOffsets[k + shCoeffs]]],
                                                     shTable[shRecord[shOffsets[k + 2 * shCoeffs]]],
                                                     0.0f);
                        }
                    }
                });

    // Like a plain .ply, the format cannot declare antialiased training.
    cloud->setAntialiased(false);
    cloud->setShDegree(shBands);
    cloud->computeBounds();
    return cloud;
}

} // namespace

std::shared_ptr<GaussianCloud> PLYReader::read(Device & device, const std::string & path)
{
    std::ifstream file(std::filesystem::u8path(path), std::ios::binary);
    VK_VALIDATE(file.is_open(), "PLYReader: failed to open " + path);

    std::string line;
    VK_VALIDATE(static_cast<bool>(std::getline(file, line)), "PLYReader: empty file " + path);
    VK_VALIDATE(trimTrailingCR(line) == "ply",
                "PLYReader: not a PLY file (missing 'ply' magic): " + path);

    bool sawFormat = false;
    bool sawEndHeader = false;
    std::vector<PlyElement> elements;

    while (std::getline(file, line))
    {
        line = trimTrailingCR(line);
        if (line == "end_header")
        {
            sawEndHeader = true;
            break;
        }

        std::istringstream iss(line);
        std::string keyword;
        iss >> keyword;

        if (keyword == "comment")
        {
            continue;
        }
        if (keyword == "format")
        {
            std::string format;
            std::string version;
            iss >> format >> version;
            VK_VALIDATE(
                format == "binary_little_endian",
                "PLYReader: only binary_little_endian PLY is supported, got '" + format + "'");
            sawFormat = true;
            continue;
        }
        if (keyword == "element")
        {
            PlyElement element;
            iss >> element.name >> element.count;
            elements.push_back(std::move(element));
            continue;
        }
        if (keyword == "property" && !elements.empty())
        {
            PlyElement & element = elements.back();
            std::string type;
            std::string name;
            iss >> type >> name;
            if (type == "list")
            {
                // Variable-size records (e.g. a mesh's face indices). Fine
                // in an element nobody reads; readElement() rejects reading
                // it or anything after it.
                element.hasList = true;
                continue;
            }
            PlyProperty prop{name, type, element.stride, plyTypeSize(type)};
            element.stride += prop.size;
            element.properties[name] = prop;
        }
    }

    VK_VALIDATE(sawFormat, "PLYReader: missing 'format' line in " + path);
    VK_VALIDATE(sawEndHeader, "PLYReader: missing 'end_header' in " + path);
    const std::streampos dataStart = file.tellg();

    // SuperSplat / PlayCanvas compressed PLY: per-256-splat bounds in a
    // `chunk` element and bit-packed vertices. Recognized by the chunk
    // element (or, failing that, by its packed vertex property, so a file
    // missing its chunks is named rather than reported as lacking 'x').
    const PlyElement * vertexElement = findElement(elements, "vertex");
    if (findElement(elements, "chunk") != nullptr)
    {
        return readCompressed(device, file, dataStart, elements, path);
    }
    VK_VALIDATE(
        vertexElement == nullptr || vertexElement->properties.count("packed_position") == 0,
        "PLYReader: vertex element has 'packed_position' (SuperSplat compressed PLY) but no "
        "'chunk' element: "
            + path);

    VK_VALIDATE(vertexElement != nullptr && vertexElement->count > 0,
                "PLYReader: no vertices in " + path);
    VK_VALIDATE(vertexElement->stride > 0, "PLYReader: no vertex properties in " + path);
    const std::unordered_map<std::string, PlyProperty> & propsByName = vertexElement->properties;
    const size_t vertexCount = vertexElement->count;
    const size_t vertexStride = vertexElement->stride;

    // Discover how many f_rest_N properties are present -- SH degree varies
    // across public pretrained scenes (0 to 3), so this isn't hardcoded.
    size_t fRestCount = 0;
    while (propsByName.count("f_rest_" + std::to_string(fRestCount)) > 0)
    {
        ++fRestCount;
    }
    VK_VALIDATE(fRestCount % 3 == 0,
                "PLYReader: f_rest property count (" + std::to_string(fRestCount)
                    + ") is not divisible by 3 channels");
    size_t coeffsPerChannel = fRestCount / 3; // higher-order coefficients, per R/G/B channel
    VK_VALIDATE(coeffsPerChannel <= static_cast<size_t>(kShCoefficientsPerGaussian - 1),
                "PLYReader: SH degree too high for kShCoefficientsPerGaussian ("
                    + std::to_string(coeffsPerChannel) + " coefficients/channel)");

    // Resolve every property's byte offset ONCE here, not per vertex. The
    // original version looked properties up by name (a std::string hash-map
    // lookup, with "f_rest_" + std::to_string(k) allocating a new string)
    // inside the per-vertex loop -- for a million-Gaussian scene at SH
    // degree 3 that's ~45 million string allocations just for f_rest,
    // dwarfing the actual parse work. This was the real bottleneck, well
    // before threading mattered.
    ResolvedOffsets offsets;
    offsets.x = resolveOffset(propsByName, "x");
    offsets.y = resolveOffset(propsByName, "y");
    offsets.z = resolveOffset(propsByName, "z");
    offsets.opacity = resolveOffset(propsByName, "opacity");
    for (int i = 0; i < 3; ++i)
    {
        offsets.scale[i] = resolveOffset(propsByName, "scale_" + std::to_string(i));
        offsets.fDc[i] = resolveOffset(propsByName, "f_dc_" + std::to_string(i));
    }
    for (int i = 0; i < 4; ++i)
    {
        offsets.rot[i] = resolveOffset(propsByName, "rot_" + std::to_string(i));
    }
    offsets.fRest.resize(fRestCount);
    for (size_t i = 0; i < fRestCount; ++i)
    {
        offsets.fRest[i] = resolveOffset(propsByName, "f_rest_" + std::to_string(i));
    }

    // Relightable 3D Gaussian (R3DG) files add a BRDF and
    // baked visibility. The marker is base_color_*: vanilla 3DGS files also
    // carry nx/ny/nz but always write zeros there, so normals are only
    // trusted alongside base_color.
    if (propsByName.count("metallic") > 0)
    {
        offsets.hasMetallic = true;
        offsets.metallic = resolveOffset(propsByName, "metallic");
    }
    if (propsByName.count("base_color_0") > 0)
    {
        offsets.relightable = true;
        for (int i = 0; i < 3; ++i)
        {
            offsets.baseColor[i] = resolveOffset(propsByName, "base_color_" + std::to_string(i));
        }
        offsets.normal[0] = resolveOffset(propsByName, "nx");
        offsets.normal[1] = resolveOffset(propsByName, "ny");
        offsets.normal[2] = resolveOffset(propsByName, "nz");
        offsets.roughness = resolveOffset(propsByName, "roughness");
        if (propsByName.count("visibility_dc_0") > 0)
        {
            offsets.hasVisibility = true;
            offsets.visibility[0] = resolveOffset(propsByName, "visibility_dc_0");
            for (int i = 0; i < 15; ++i)
            {
                offsets.visibility[1 + i] =
                    resolveOffset(propsByName, "visibility_rest_" + std::to_string(i));
            }
        }
    }

    // Read the whole binary vertex blob in one shot.
    const std::vector<char> raw = readElement(file, dataStart, elements, *vertexElement, path);

    auto cloud = std::make_shared<GaussianCloud>(device);
    cloud->initialize(vertexCount);

    // SH degree is (coefficients per channel + 1) square-rooted minus 1:
    // 0 coeffs -> degree 0, 3 -> 1, 8 -> 2, 15 -> 3. Discovered from the
    // header above, never hardcoded, so the viewer can report what the file
    // actually carries.
    // The format cannot declare antialiased training, and vanilla
    // 3DGS (what produces it) trains without the dilation opacity
    // compensation -- so off. Renderer::setAntialiased overrides it.
    cloud->setAntialiased(false);
    cloud->setShDegree(
        static_cast<uint32_t>(std::lround(std::sqrt(static_cast<double>(coeffsPerChannel + 1)))) - 1);

    if (offsets.relightable)
    {
        cloud->normals().resize(vertexCount);
        cloud->baseColorRoughness().resize(vertexCount);
        if (offsets.hasVisibility)
        {
            cloud->visibility().resize(vertexCount * kVisibilityCoefficients);
        }
    }
    auto & normals = cloud->normals();
    auto & baseColorRoughness = cloud->baseColorRoughness();
    auto & visibility = cloud->visibility();
    if (offsets.hasMetallic)
    {
        cloud->metallic().resize(vertexCount);
    }
    auto & metallic = cloud->metallic();

    auto & positions = cloud->positions();
    auto & scales = cloud->scales();
    auto & rotations = cloud->rotations();
    auto & sh = cloud->shCoefficients();
    auto & opacities = cloud->opacities();
    const char * rawData = raw.data();

    // Each worker only touches indices [begin, end) of these pre-sized
    // vectors (initialize() already allocated them), so this is safe
    // without any locking -- disjoint writes, no shared mutable state.
    parallelFor(vertexCount,
                [&](size_t begin, size_t end)
                {
                    for (size_t i = begin; i < end; ++i)
                    {
                        const char * vertex = rawData + i * vertexStride;

                        // Streams are padded to the canonical 16 B stride;
                        // the w lane is unused padding, not data.
                        positions[i] = glm::vec4(readFloatAt(vertex, offsets.x),
                                                 readFloatAt(vertex, offsets.y),
                                                 readFloatAt(vertex, offsets.z),
                                                 0.0f);

                        // Reference 3DGS stores pre-activation params for
                        // gradient-friendly optimization; this loader is for using a
                        // (pretrained) cloud, so store the activated, ready-to-use
                        // values instead.
                        scales[i] = glm::vec4(std::exp(readFloatAt(vertex, offsets.scale[0])),
                                              std::exp(readFloatAt(vertex, offsets.scale[1])),
                                              std::exp(readFloatAt(vertex, offsets.scale[2])),
                                              0.0f);

                        opacities[i] = sigmoid(readFloatAt(vertex, offsets.opacity));

                        // Reference 3DGS stores an UNNORMALIZED quaternion in
                        // (w, x, y, z) order. The canonical GPU layout is xyzw, so
                        // this both normalizes and reorders: rot_0 is w and lands in
                        // the LAST lane.
                        glm::quat q = normalizeQuatWXYZ(readFloatAt(vertex, offsets.rot[0]),
                                                        readFloatAt(vertex, offsets.rot[1]),
                                                        readFloatAt(vertex, offsets.rot[2]),
                                                        readFloatAt(vertex, offsets.rot[3]));
                        rotations[i] = glm::vec4(q.x, q.y, q.z, q.w);

                        // SH coefficient 0 (DC term) is the raw f_dc_* triplet -- stored
                        // as a true SH coefficient, not converted to display RGB.
                        glm::vec4 * shOut = &sh[i * kShCoefficientsPerGaussian];
                        shOut[0] = glm::vec4(readFloatAt(vertex, offsets.fDc[0]),
                                             readFloatAt(vertex, offsets.fDc[1]),
                                             readFloatAt(vertex, offsets.fDc[2]),
                                             0.0f);

                        // f_rest is stored channel-major (all R coefficients, then all
                        // G, then all B) -- NOT interleaved per coefficient. A common
                        // bug in 3DGS loaders is assuming interleaving.
                        for (size_t k = 0; k < coeffsPerChannel; ++k)
                        {
                            float r = readFloatAt(vertex, offsets.fRest[k]);
                            float g = readFloatAt(vertex, offsets.fRest[k + coeffsPerChannel]);
                            float b = readFloatAt(vertex, offsets.fRest[k + 2 * coeffsPerChannel]);
                            shOut[1 + k] = glm::vec4(r, g, b, 0.0f);
                        }
                        // Any remaining coefficient slots (if this scene's SH degree is
                        // lower than kShCoefficientsPerGaussian - 1) stay zero-initialized.

                        if (offsets.relightable)
                        {
                            // R3DG's stored values are pre-activation; its
                            // activations: normal normalized (eps 1e-3),
                            // base color sigmoid*0.77+0.03, roughness
                            // sigmoid*0.9+0.09 (scene/gaussian_model.py).
                            glm::vec3 n(readFloatAt(vertex, offsets.normal[0]),
                                        readFloatAt(vertex, offsets.normal[1]),
                                        readFloatAt(vertex, offsets.normal[2]));
                            n /= std::max(glm::length(n), 1e-3f);
                            normals[i] = glm::vec4(n, 0.0f);
                            baseColorRoughness[i] = glm::vec4(
                                sigmoid(readFloatAt(vertex, offsets.baseColor[0])) * 0.77f + 0.03f,
                                sigmoid(readFloatAt(vertex, offsets.baseColor[1])) * 0.77f + 0.03f,
                                sigmoid(readFloatAt(vertex, offsets.baseColor[2])) * 0.77f + 0.03f,
                                sigmoid(readFloatAt(vertex, offsets.roughness)) * 0.9f + 0.09f);
                            if (offsets.hasVisibility)
                            {
                                // Raw SH, identity activation, same basis as
                                // the color SH.
                                for (size_t k = 0; k < kVisibilityCoefficients; ++k)
                                {
                                    visibility[i * kVisibilityCoefficients + k] =
                                        readFloatAt(vertex, offsets.visibility[k]);
                                }
                            }
                        }

                        if (offsets.hasMetallic)
                        {
                            // Raw, like visibility: no activation, so the value
                            // PLYWriter stores reads back exactly.
                            metallic[i] = readFloatAt(vertex, offsets.metallic);
                        }
                    }
                });

    cloud->computeBounds();
    return cloud;
}

} // namespace sabercat
