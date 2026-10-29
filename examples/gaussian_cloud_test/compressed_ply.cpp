// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0
// Portions derived from PlayCanvas splat-transform (https://github.com/playcanvas/splat-transform), MIT; see THIRD_PARTY_NOTICES.md.

// SuperSplat / PlayCanvas compressed .ply through PLYReader.
//
// There is no written spec for the format, so the fixture is produced here by
// a C++ port of the reference ENCODER (splat-transform
// src/lib/writers/compressed-chunk.ts + write-compressed-ply.ts): a synthetic
// source scene is quantized exactly as the reference quantizes it, written to
// disk, read back through PLYReader, and every attribute is compared to its
// SOURCE value within a bound derived from that attribute's bit width (see
// Tolerances below). Encoding in-test keeps the scene exact and in source
// order, so no position matching is needed. Variants cover both chunk
// layouts (12 floats; 18 with color bounds), SH bands 0-3, a partial last
// chunk, a zero-width chunk axis, the multithreaded decode path, and the
// rejection of malformed variants with messages naming the problem.

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "../common/scratch_dir.h"
#include "sabercat/device.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/ply_reader.h"
#include "sabercat/render_config.h"

using namespace sabercat;

namespace
{

constexpr size_t kChunkSize = 256;
constexpr int kRestPerChannel[4] = {0, 3, 8, 15};
constexpr double kShC0Reference = 0.28209479177387814; // the reference's SH_C0

// --- Tolerances ---
// A b-bit unorm over [lo, hi] rounds to the nearest of 2^b - 1 steps, so it
// is off by at most half a step: (hi - lo) / (2 * (2^b - 1)). Per attribute:
//   position   11/10/11 bits over the chunk's per-axis range
//   log scale  same, over the chunk's log-scale range -> RELATIVE linear scale
//              error exp(half step) - 1
//   color     8 bits over the chunk's display-color range (18-float chunks)
//              or [0, 1] (12-float), / SH_C0 to reach f_dc
//   opacity    8 bits over [0, 1]: 1/510
//   rotation   each stored lane is 10 bits over [-1/sqrt2, 1/sqrt2], half
//              step sqrt2/2046 = 6.9e-4; the dropped (largest, >= 1/2) lane
//              is rebuilt from the other three, |dm| <= |a| |e| / m <=
//              sqrt3 * sqrt3 * 6.9e-4 = 2.1e-3; quaternion chord
//              sqrt(3 * 6.9e-4^2 + 2.1e-3^2) = 2.4e-3 -> bound 2.5e-3
//   SH         8 bits over [-4, 4], decoded at the bin center: 8/256/2 =
//              1/64, except bytes 0 and 255, which decode to -4 / +4 (the
//              range ends), up to a whole bin, 1/32, away
// plus a small float slack: the reader decodes in double but stores float,
// which for positions also costs half an ulp of the (up to ~4000) magnitude.
constexpr double kFloatSlack = 1e-5;
constexpr double kFloatRelativeSlack = 1.0 / (1 << 23);
constexpr double kRotationChordBound = 2.5e-3;
constexpr double kOpacityBound = 1.0 / 510.0;

double halfStep(double lo, double hi, int bits)
{
    return (hi - lo) / (2.0 * ((1 << bits) - 1));
}

// --- Source scene ---

struct SourceSplat
{
    float position[3];
    float logScale[3];
    float rot[4]; // rot_0..rot_3 = w, x, y, z, unnormalized
    float fdc[3];
    float opacity;     // pre-sigmoid
    float rest[3][15]; // [channel][coefficient]
};

uint32_t g_state = 0x2545F491u;
float uniform(float lo, float hi)
{
    g_state ^= g_state << 13;
    g_state ^= g_state >> 17;
    g_state ^= g_state << 5;
    return lo + (hi - lo) * (static_cast<float>(g_state >> 8) * (1.0f / 16777216.0f));
}

// Mixed-sign, varied-magnitude values; a few rows pin the edge cases.
std::vector<SourceSplat> makeScene(size_t count)
{
    std::vector<SourceSplat> scene(count);
    for (size_t i = 0; i < count; ++i)
    {
        SourceSplat & s = scene[i];
        const size_t chunk = i / kChunkSize;
        for (int a = 0; a < 3; ++a)
        {
            // Chunks at different offsets and extents.
            s.position[a] = uniform(-5.0f, 5.0f) * (1.0f + chunk) + 3.0f * chunk;
            s.logScale[a] = uniform(-6.0f, 1.0f);
            s.fdc[a] = uniform(-1.5f, 1.5f);
        }
        for (int lane = 0; lane < 4; ++lane)
        {
            s.rot[lane] = uniform(-2.0f, 2.0f);
        }
        s.opacity = uniform(-6.0f, 6.0f);
        for (int c = 0; c < 3; ++c)
        {
            for (int k = 0; k < 15; ++k)
            {
                s.rest[c][k] = uniform(-3.9f, 3.9f);
            }
        }
    }
    // Identity, a negative largest lane (the encoder flips the sign), and
    // each lane the largest in turn.
    const float rots[6][4] = {{1, 0, 0, 0},
                              {-0.9f, 0.1f, 0.3f, -0.2f},
                              {0.1f, 0.8f, -0.3f, 0.2f},
                              {0.2f, -0.1f, -0.9f, 0.3f},
                              {-0.3f, 0.2f, 0.1f, 0.7f},
                              {0.5f, 0.5f, 0.5f, 0.5f}};
    for (int r = 0; r < 6 && r < static_cast<int>(count); ++r)
    {
        std::copy(rots[r], rots[r] + 4, scene[r].rot);
    }
    // SH at and beyond the range ends (beyond clamps to +-4).
    if (count > 2)
    {
        scene[0].rest[0][0] = 4.0f;
        scene[0].rest[1][0] = -4.0f;
        scene[1].rest[0][0] = 3.98f;
        scene[1].rest[1][0] = -3.98f;
        scene[2].rest[0][0] = 6.0f;
        scene[2].rest[1][0] = -6.0f;
    }
    // A zero-width axis: z constant across the whole last chunk.
    const size_t lastChunkStart = ((count - 1) / kChunkSize) * kChunkSize;
    for (size_t i = lastChunkStart; i < count; ++i)
    {
        scene[i].position[2] = 1.25f;
    }
    return scene;
}

// --- Reference encoder (compressed-chunk.ts / write-compressed-ply.ts) ---

struct Encoded
{
    std::vector<float> chunks;    // chunkFloats per chunk
    std::vector<uint32_t> packed; // position, rotation, scale, color per splat
    std::vector<uint8_t> sh;      // 3 * restPerChannel per splat
    size_t chunkFloats = 18;
    int bands = 0;
};

double clampValue(double v, double lo, double hi)
{
    return std::max(lo, std::min(hi, v));
}

double normalizeTo(double x, double lo, double hi)
{
    if (x <= lo)
    {
        return 0.0;
    }
    if (x >= hi)
    {
        return 1.0;
    }
    return hi - lo < 0.00001 ? 0.0 : (x - lo) / (hi - lo);
}

uint32_t packUnorm(double value, int bits)
{
    const double t = (1 << bits) - 1;
    return static_cast<uint32_t>(clampValue(std::floor(value * t + 0.5), 0.0, t));
}

uint32_t pack111011(double x, double y, double z)
{
    return (packUnorm(x, 11) << 21) | (packUnorm(y, 10) << 11) | packUnorm(z, 11);
}

uint32_t packRot(const float rot[4])
{
    double a[4] = {rot[0], rot[1], rot[2], rot[3]};
    const double length = std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + a[3] * a[3]);
    int largest = 0;
    for (int i = 0; i < 4; ++i)
    {
        a[i] /= length;
        if (std::abs(a[i]) > std::abs(a[largest]))
        {
            largest = i;
        }
    }
    const double sign = a[largest] < 0 ? -1.0 : 1.0;
    const double norm = std::sqrt(2.0) * 0.5;
    uint32_t result = static_cast<uint32_t>(largest);
    for (int i = 0; i < 4; ++i)
    {
        if (i != largest)
        {
            result = (result << 10) | packUnorm(sign * a[i] * norm + 0.5, 10);
        }
    }
    return result;
}

// `chunkColor`: the 18-float layout (color bounds); otherwise the 12-float
// one, color stored as a plain unorm.
Encoded encode(const std::vector<SourceSplat> & scene, bool chunkColor, int bands)
{
    Encoded out;
    out.chunkFloats = chunkColor ? 18 : 12;
    out.bands = bands;
    const size_t count = scene.size();
    const size_t chunks = (count + kChunkSize - 1) / kChunkSize;
    const int rest = kRestPerChannel[bands];
    out.packed.resize(count * 4);
    out.sh.resize(count * 3 * rest);

    for (size_t c = 0; c < chunks; ++c)
    {
        // The writer pads the last chunk by repeating its last splat.
        std::vector<const SourceSplat *> rows(kChunkSize);
        for (size_t j = 0; j < kChunkSize; ++j)
        {
            rows[j] = &scene[std::min(c * kChunkSize + j, count - 1)];
        }
        // Display color, rounded to float as the writer's Float32Array does.
        std::vector<float> display(kChunkSize * 3);
        for (size_t j = 0; j < kChunkSize; ++j)
        {
            for (int a = 0; a < 3; ++a)
            {
                display[j * 3 + a] = static_cast<float>(rows[j]->fdc[a] * kShC0Reference + 0.5);
            }
        }

        float lo[9], hi[9]; // position xyz, log scale xyz, color rgb
        for (int a = 0; a < 9; ++a)
        {
            lo[a] = std::numeric_limits<float>::max();
            hi[a] = std::numeric_limits<float>::lowest();
        }
        for (size_t j = 0; j < kChunkSize; ++j)
        {
            for (int a = 0; a < 3; ++a)
            {
                const float v[3] = {rows[j]->position[a], rows[j]->logScale[a], display[j * 3 + a]};
                for (int g = 0; g < 3; ++g)
                {
                    lo[g * 3 + a] = std::min(lo[g * 3 + a], v[g]);
                    hi[g * 3 + a] = std::max(hi[g * 3 + a], v[g]);
                }
            }
        }
        for (int a = 3; a < 6; ++a)
        {
            lo[a] = static_cast<float>(clampValue(lo[a], -20.0, 20.0));
            hi[a] = static_cast<float>(clampValue(hi[a], -20.0, 20.0));
        }
        const float chunkRecord[18] = {lo[0],
                                       lo[1],
                                       lo[2],
                                       hi[0],
                                       hi[1],
                                       hi[2],
                                       lo[3],
                                       lo[4],
                                       lo[5],
                                       hi[3],
                                       hi[4],
                                       hi[5],
                                       lo[6],
                                       lo[7],
                                       lo[8],
                                       hi[6],
                                       hi[7],
                                       hi[8]};
        out.chunks.insert(out.chunks.end(), chunkRecord, chunkRecord + out.chunkFloats);

        for (size_t j = 0; j < kChunkSize && c * kChunkSize + j < count; ++j)
        {
            const size_t i = c * kChunkSize + j;
            const SourceSplat & s = scene[i];
            double colorUnit[3];
            for (int a = 0; a < 3; ++a)
            {
                colorUnit[a] = chunkColor ? normalizeTo(display[j * 3 + a], lo[6 + a], hi[6 + a])
                                          : normalizeTo(display[j * 3 + a], 0.0, 1.0);
            }
            const double alpha = 1.0 / (1.0 + std::exp(-static_cast<double>(s.opacity)));
            out.packed[i * 4 + 0] = pack111011(normalizeTo(s.position[0], lo[0], hi[0]),
                                               normalizeTo(s.position[1], lo[1], hi[1]),
                                               normalizeTo(s.position[2], lo[2], hi[2]));
            out.packed[i * 4 + 1] = packRot(s.rot);
            out.packed[i * 4 + 2] = pack111011(normalizeTo(s.logScale[0], lo[3], hi[3]),
                                               normalizeTo(s.logScale[1], lo[4], hi[4]),
                                               normalizeTo(s.logScale[2], lo[5], hi[5]));
            out.packed[i * 4 + 3] = (packUnorm(colorUnit[0], 8) << 24)
                                    | (packUnorm(colorUnit[1], 8) << 16)
                                    | (packUnorm(colorUnit[2], 8) << 8) | packUnorm(alpha, 8);

            // f_rest_(channel * rest + k), channel-major.
            for (int ch = 0; ch < 3; ++ch)
            {
                for (int k = 0; k < rest; ++k)
                {
                    const double n = s.rest[ch][k] / 8.0 + 0.5;
                    out.sh[(i * 3 + ch) * rest + k] =
                        static_cast<uint8_t>(clampValue(std::trunc(n * 256.0), 0.0, 255.0));
                }
            }
        }
    }
    return out;
}

const char * kChunkNames[18] = {"min_x",
                                "min_y",
                                "min_z",
                                "max_x",
                                "max_y",
                                "max_z",
                                "min_scale_x",
                                "min_scale_y",
                                "min_scale_z",
                                "max_scale_x",
                                "max_scale_y",
                                "max_scale_z",
                                "min_r",
                                "min_g",
                                "min_b",
                                "max_r",
                                "max_g",
                                "max_b"};

// The reference header, then the three blobs. `vertexType`, `extraChunkProp`,
// `chunkCountDelta`, `withChunk` and `truncateBytes` corrupt it for the
// rejection checks.
struct WriteOptions
{
    const char * vertexType = "uint";
    int chunkCountDelta = 0;
    int shPropertyCount = -1; // -1: the encoded count
    size_t truncateBytes = 0;
    bool withChunk = true;
    bool extraChunkProp = false;
};

void writeCompressed(const std::string & path, const Encoded & e, const WriteOptions & options = {})
{
    const size_t count = e.packed.size() / 4;
    const size_t chunks = e.chunks.size() / e.chunkFloats;
    const int shProps =
        options.shPropertyCount >= 0 ? options.shPropertyCount : 3 * kRestPerChannel[e.bands];
    std::string header =
        "ply\nformat binary_little_endian 1.0\ncomment Generated by sabercat test\n";
    if (options.withChunk)
    {
        header += "element chunk "
                  + std::to_string(static_cast<int>(chunks) + options.chunkCountDelta) + "\n";
        for (size_t k = 0; k < e.chunkFloats; ++k)
        {
            header += std::string("property float ") + kChunkNames[k] + "\n";
        }
        if (options.extraChunkProp)
        {
            header += "property float extra\n";
        }
    }
    header += "element vertex " + std::to_string(count) + "\n";
    for (const char * name : {"packed_position", "packed_rotation", "packed_scale", "packed_color"})
    {
        header += std::string("property ") + options.vertexType + " " + name + "\n";
    }
    if (shProps > 0)
    {
        header += "element sh " + std::to_string(count) + "\n";
        for (int k = 0; k < shProps; ++k)
        {
            header += "property uchar f_rest_" + std::to_string(k) + "\n";
        }
    }
    header += "end_header\n";

    std::string bytes = header;
    if (options.withChunk)
    {
        bytes.append(reinterpret_cast<const char *>(e.chunks.data()),
                     e.chunks.size() * sizeof(float));
    }
    bytes.append(reinterpret_cast<const char *>(e.packed.data()),
                 e.packed.size() * sizeof(uint32_t));
    bytes.append(reinterpret_cast<const char *>(e.sh.data()), e.sh.size());
    bytes.resize(bytes.size() - options.truncateBytes);
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

#define CHECK(condition, message)                                                                  \
    do                                                                                             \
    {                                                                                              \
        if (!(condition))                                                                          \
        {                                                                                          \
            std::cerr << "FAILED: " << message << "\n  at " << __FILE__ << ":" << __LINE__         \
                      << std::endl;                                                                \
            return 1;                                                                              \
        }                                                                                          \
    } while (false)

// Running max of |error| / bound, so one number per attribute says how close
// the decode came to its bound (must stay <= 1).
struct Worst
{
    double ratio = 0.0;
    void add(double error, double bound)
    {
        ratio = std::max(ratio, error / bound);
    }
};

int testVariant(Device & device, size_t count, bool chunkColor, int bands)
{
    const std::vector<SourceSplat> scene = makeScene(count);
    const Encoded encoded = encode(scene, chunkColor, bands);
    const std::string path = "compressed.ply";
    writeCompressed(path, encoded);

    auto cloud = PLYReader::read(device, path);
    const std::string label = std::to_string(count) + " splats, " + (chunkColor ? "18" : "12")
                              + "-float chunks, SH bands " + std::to_string(bands);
    CHECK(cloud->count() == count, label << ": count " << cloud->count());
    CHECK(cloud->shDegree() == static_cast<uint32_t>(bands),
          label << ": SH degree " << cloud->shDegree());
    CHECK(!cloud->antialiased(), label << ": antialiased set");

    Worst position, scale, color, opacity, rotation, sh;
    double maxNormError = 0.0;
    const int rest = kRestPerChannel[bands];
    for (size_t i = 0; i < count; ++i)
    {
        const SourceSplat & s = scene[i];
        const float * bounds = &encoded.chunks[(i / kChunkSize) * encoded.chunkFloats];
        const int bits[3] = {11, 10, 11};

        for (int a = 0; a < 3; ++a)
        {
            const double posBound = halfStep(bounds[a], bounds[3 + a], bits[a]) + kFloatSlack
                                    + std::abs(s.position[a]) * kFloatRelativeSlack;
            position.add(std::abs(cloud->positions()[i][a] - s.position[a]), posBound);

            const double scaleBound =
                std::expm1(halfStep(bounds[6 + a], bounds[9 + a], bits[a])) + kFloatSlack;
            scale.add(std::abs(cloud->scales()[i][a] / std::exp(static_cast<double>(s.logScale[a]))
                               - 1.0),
                      scaleBound);

            const double colorBound =
                (chunkColor ? halfStep(bounds[12 + a], bounds[15 + a], 8) : halfStep(0.0, 1.0, 8))
                    / kShC0Reference
                + kFloatSlack;
            color.add(
                std::abs(cloud->shCoefficients()[i * kShCoefficientsPerGaussian][a] - s.fdc[a]),
                colorBound);
        }

        const double alpha = 1.0 / (1.0 + std::exp(-static_cast<double>(s.opacity)));
        opacity.add(std::abs(cloud->opacities()[i] - alpha), kOpacityBound + kFloatSlack);

        // Source wxyz -> xyzw, normalized; q and -q are the same rotation.
        const glm::vec4 r = cloud->rotations()[i];
        glm::dvec4 expected(s.rot[1], s.rot[2], s.rot[3], s.rot[0]);
        expected /= glm::length(expected);
        if (glm::dot(expected, glm::dvec4(r)) < 0.0)
        {
            expected = -expected;
        }
        rotation.add(glm::length(glm::dvec4(r) - expected), kRotationChordBound);
        maxNormError = std::max(maxNormError, std::abs(glm::length(glm::dvec4(r)) - 1.0));

        const glm::vec4 * shOut = &cloud->shCoefficients()[i * kShCoefficientsPerGaussian];
        for (int k = 0; k < kShCoefficientsPerGaussian - 1; ++k)
        {
            for (int c = 0; c < 3; ++c)
            {
                if (k < rest)
                {
                    const double v = clampValue(s.rest[c][k], -4.0, 4.0);
                    const double bound =
                        (std::abs(v) >= 4.0 - 1.0 / 32.0 ? 1.0 / 32.0 : 1.0 / 64.0) + kFloatSlack;
                    sh.add(std::abs(shOut[1 + k][c] - v), bound);
                }
                else
                {
                    CHECK(shOut[1 + k][c] == 0.0f,
                          label << ": SH slot " << (1 + k) << " not zero at " << i);
                }
            }
        }
    }

    std::printf("  %s: worst error / bound -- position %.3f, scale %.3f, f_dc %.3f, opacity %.3f, "
                "rotation %.3f, SH %.3f; max | |q| - 1 | %.2e\n",
                label.c_str(),
                position.ratio,
                scale.ratio,
                color.ratio,
                opacity.ratio,
                rotation.ratio,
                sh.ratio,
                maxNormError);
    CHECK(position.ratio <= 1.0, label << ": position error over bound");
    CHECK(scale.ratio <= 1.0, label << ": scale error over bound");
    CHECK(color.ratio <= 1.0, label << ": f_dc error over bound");
    CHECK(opacity.ratio <= 1.0, label << ": opacity error over bound");
    CHECK(rotation.ratio <= 1.0, label << ": rotation error over bound");
    CHECK(sh.ratio <= 1.0, label << ": SH error over bound");
    CHECK(maxNormError < 1e-6, label << ": rotation not unit length");
    // Degeneracy guards: a stubbed conversion must not pass. The quantized
    // errors are all well inside their bounds, so each ratio being > 0.1
    // shows the bound is tight, not vacuous.
    CHECK(position.ratio > 0.1 && scale.ratio > 0.1 && rotation.ratio > 0.1,
          label << ": bounds suspiciously loose");
    std::remove(path.c_str());
    return 0;
}

// The reader must refuse with a message naming the problem -- never the
// plain-PLY "missing required property 'x'".
int expectRejection(Device & device, const Encoded & e, const WriteOptions & options,
                    const std::string & expected, const char * label)
{
    const std::string path = "rejected.ply";
    writeCompressed(path, e, options);
    std::string message;
    try
    {
        PLYReader::read(device, path);
    }
    catch (const std::exception & error)
    {
        message = error.what();
    }
    std::remove(path.c_str());
    CHECK(!message.empty(), label << ": was accepted");
    CHECK(message.find(expected) != std::string::npos,
          label << ": message '" << message << "' lacks '" << expected << "'");
    CHECK(message.find("property 'x'") == std::string::npos,
          label << ": misleading message " << message);
    std::printf("  rejects %s: %s\n", label, message.substr(0, message.find(" at ")).c_str());
    return 0;
}

} // namespace

int testCompressedPly(Device & device)
{
    examples::ScratchDir scratch("gaussian_cloud_test_compressed");
    std::cout << "Compressed PLY (SuperSplat / PlayCanvas):\n";

    // 600 = two full chunks + a partial one; 120000 crosses parallelFor's
    // 50000 threshold.
    if (testVariant(device, 600, true, 3) || testVariant(device, 600, false, 0)
        || testVariant(device, 300, false, 1) || testVariant(device, 257, true, 2)
        || testVariant(device, 120000, true, 1))
    {
        return 1;
    }

    const Encoded e = encode(makeScene(300), true, 1);
    WriteOptions badSh;
    badSh.shPropertyCount = 12;
    WriteOptions badChunks;
    badChunks.chunkCountDelta = 1;
    WriteOptions badType;
    badType.vertexType = "float";
    WriteOptions noChunk;
    noChunk.withChunk = false;
    WriteOptions extraProp;
    extraProp.extraChunkProp = true;
    WriteOptions truncated;
    truncated.truncateBytes = 1;
    if (expectRejection(device, e, badSh, "sh element has 12 properties", "SH property count")
        || expectRejection(device, e, badChunks, "3 chunks for 300 vertices", "chunk count")
        || expectRejection(device, e, badType, "expected uint", "packed property type")
        || expectRejection(device, e, noChunk, "no 'chunk' element", "missing chunk element")
        || expectRejection(device, e, extraProp, "chunk element has 19 properties", "chunk layout")
        || expectRejection(device, e, truncated, "truncated 'sh' element", "truncated file"))
    {
        return 1;
    }
    std::cout << "Compressed PLY test OK\n";
    return 0;
}
