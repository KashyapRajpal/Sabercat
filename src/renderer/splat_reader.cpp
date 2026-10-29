// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/splat_reader.h"

#include <glm/gtc/quaternion.hpp>

#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

#include "sabercat/gaussian_cloud.h"
#include "sabercat/parallel_for.h"
#include "sabercat/render_config.h"
#include "sabercat/vk_check.h"

namespace sabercat
{

namespace
{

// --- Record layout (antimatter15 .splat) ---
// Headerless: the file is nothing but back-to-back 32-byte records, so the
// only structural check available is `size % 32 == 0` (see read()).
static constexpr size_t kRecordSize = 32;
static constexpr size_t kPositionOffset = 0;  // 3x f32
static constexpr size_t kScaleOffset = 12;    // 3x f32
static constexpr size_t kColorOffset = 24;    // 4x u8, RGBA
static constexpr size_t kRotationOffset = 28; // 4x u8, wxyz

// --- Canonical-contract conversions ---
// GaussianCloud stores ACTIVATED values (opacity in [0,1], scale as a linear
// std-dev, SH coefficient 0 as a raw f_dc). The .splat exporter has already
// applied exp() to scale and sigmoid() to opacity, and has additionally
// converted f_dc to a display-RGB byte -- so scale passes through untouched,
// opacity is a plain byte->unit rescale, and only color needs inverting.

// Exporter: color = round((f_dc * SH_C0 + 0.5) * 255). Inverting that needs
// the reciprocal of SH_C0 (the degree-0 SH basis constant shared with the
// shaders via config.slangh, so host and GPU can never disagree on it).
static constexpr float kByteToUnit = 1.0f / 255.0f;
static constexpr float kInvShC0 = 1.0f / kShC0;
static constexpr float kDcDisplayOffset = 0.5f;

// Exporter: rot_byte = round(q * 128 + 128) per component, in wxyz order.
static constexpr float kQuatByteScale = 1.0f / 128.0f;
static constexpr float kQuatByteBias = 128.0f;

// --- Sanity guard ---
// A headerless format cannot reject a wrong file that happens to divide by
// 32, so a decoded prefix is checked for finite positions and finite,
// strictly positive scales. A stray .bin then fails loudly here instead of
// rendering as noise (or as nothing, if every scale decodes to garbage).
static constexpr size_t kSanityPrefixCount = 64;

// A quaternion of all-128 bytes decodes to (0,0,0,0), which has no direction
// to normalize toward. Substituting identity keeps one corrupt record from
// producing NaNs that would propagate through the covariance math.
static constexpr float kMinQuatLengthSquared = 1e-12f;

inline float readFloatAt(const char * record, size_t offset)
{
    float value;
    std::memcpy(&value, record + offset, sizeof(float));
    return value;
}

inline uint8_t readByteAt(const char * record, size_t offset)
{
    return static_cast<uint8_t>(record[offset]);
}

// Decodes one quaternion byte to its [-1, 1] float, per the exporter's
// `round(q * 128 + 128)`.
inline float decodeQuatByte(uint8_t byte)
{
    return (static_cast<float>(byte) - kQuatByteBias) * kQuatByteScale;
}

// Verifies a prefix of the file decodes to plausible Gaussians. Deliberately
// NOT a check on the whole file: this is a "wrong file entirely" guard, and
// a full scan would cost a pass over a multi-hundred-megabyte asset to catch
// something the first few records already prove.
void validateDecodedPrefix(const char * data, size_t recordCount, const std::string & path)
{
    const size_t checkCount = std::min(recordCount, kSanityPrefixCount);
    for (size_t i = 0; i < checkCount; ++i)
    {
        const char * record = data + i * kRecordSize;
        for (size_t axis = 0; axis < 3; ++axis)
        {
            const float position = readFloatAt(record, kPositionOffset + axis * sizeof(float));
            VK_VALIDATE(std::isfinite(position),
                        "SplatReader: record " + std::to_string(i)
                            + " has a non-finite position -- this does not look like a .splat "
                              "file: "
                            + path);

            const float scale = readFloatAt(record, kScaleOffset + axis * sizeof(float));
            VK_VALIDATE(std::isfinite(scale) && scale > 0.0f,
                        "SplatReader: record " + std::to_string(i) + " has a non-positive scale ("
                            + std::to_string(scale)
                            + ") -- this does not look like a .splat file: " + path);
        }
    }
}

} // namespace

std::shared_ptr<GaussianCloud> SplatReader::read(Device & device, const std::string & path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    VK_VALIDATE(file.is_open(), "SplatReader: failed to open " + path);

    const std::streamsize fileSize = file.tellg();
    file.seekg(0, std::ios::beg);

    VK_VALIDATE(fileSize > 0, "SplatReader: file is empty: " + path);
    // Name the remainder -- "not a multiple of 32" alone doesn't tell the
    // caller whether they have a truncated file or the wrong format.
    VK_VALIDATE(static_cast<size_t>(fileSize) % kRecordSize == 0,
                "SplatReader: file size " + std::to_string(fileSize)
                    + " is not a multiple of the 32-byte record size ("
                    + std::to_string(static_cast<size_t>(fileSize) % kRecordSize)
                    + " bytes left over): " + path);

    const size_t gaussianCount = static_cast<size_t>(fileSize) / kRecordSize;

    std::vector<char> raw(static_cast<size_t>(fileSize));
    file.read(raw.data(), fileSize);
    VK_VALIDATE(file.gcount() == fileSize, "SplatReader: truncated read of " + path);

    validateDecodedPrefix(raw.data(), gaussianCount, path);

    auto cloud = std::make_shared<GaussianCloud>(device);
    cloud->initialize(gaussianCount);

    auto & positions = cloud->positions();
    auto & scales = cloud->scales();
    auto & rotations = cloud->rotations();
    auto & sh = cloud->shCoefficients();
    auto & opacities = cloud->opacities();
    const char * rawData = raw.data();

    // Same disjoint-range contract as PLYReader: every worker writes only
    // indices [begin, end) of vectors initialize() already sized, so no
    // locking is needed and no worker shares mutable state.
    parallelFor(gaussianCount,
                [&](size_t begin, size_t end)
                {
                    for (size_t i = begin; i < end; ++i)
                    {
                        const char * record = rawData + i * kRecordSize;

                        // Streams pad to the canonical 16 B stride; the w
                        // lane is unused padding, not data.
                        positions[i] = glm::vec4(readFloatAt(record, kPositionOffset),
                                                 readFloatAt(record, kPositionOffset + 4),
                                                 readFloatAt(record, kPositionOffset + 8),
                                                 0.0f);

                        // Scale is ALREADY linear -- the exporter applied
                        // exp() on the way out. Applying it again here would
                        // silently inflate every Gaussian.
                        scales[i] = glm::vec4(readFloatAt(record, kScaleOffset),
                                              readFloatAt(record, kScaleOffset + 4),
                                              readFloatAt(record, kScaleOffset + 8),
                                              0.0f);

                        // Alpha is already sigmoid-activated and quantized,
                        // so recovering the canonical [0,1] opacity is a
                        // plain rescale -- no invSigmoid/sigmoid round trip
                        // (which would also blow up at bytes 0 and 255).
                        opacities[i] = readByteAt(record, kColorOffset + 3) * kByteToUnit;

                        // Color bytes are DISPLAY RGB, not raw SH. Undo the
                        // exporter's `f_dc * SH_C0 + 0.5` so coefficient 0
                        // holds a real f_dc, matching what PLYReader stores
                        // and what the shader's SH evaluation expects.
                        glm::vec4 * shOut = &sh[i * kShCoefficientsPerGaussian];
                        shOut[0] = glm::vec4(
                            (readByteAt(record, kColorOffset + 0) * kByteToUnit - kDcDisplayOffset)
                                * kInvShC0,
                            (readByteAt(record, kColorOffset + 1) * kByteToUnit - kDcDisplayOffset)
                                * kInvShC0,
                            (readByteAt(record, kColorOffset + 2) * kByteToUnit - kDcDisplayOffset)
                                * kInvShC0,
                            0.0f);
                        // Coefficients 1..15 stay zero: .splat carries no
                        // bands above DC. initialize() already zeroed them,
                        // so there is nothing to write here -- and the test
                        // asserts they are EXACTLY zero.

                        // Rotation bytes are wxyz (confirmed by decoding
                        // bonsai-7k.splat against bonsai-7k.ply), and the
                        // decoded quaternion is NOT unit length -- both the
                        // reorder to canonical xyzw and the normalize are
                        // required, and skipping either yields a
                        // plausible-but-wrong image rather than a failure.
                        const float qw = decodeQuatByte(readByteAt(record, kRotationOffset + 0));
                        const float qx = decodeQuatByte(readByteAt(record, kRotationOffset + 1));
                        const float qy = decodeQuatByte(readByteAt(record, kRotationOffset + 2));
                        const float qz = decodeQuatByte(readByteAt(record, kRotationOffset + 3));

                        const float lengthSquared = qw * qw + qx * qx + qy * qy + qz * qz;
                        if (lengthSquared < kMinQuatLengthSquared)
                        {
                            rotations[i] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f); // identity, xyzw
                        }
                        else
                        {
                            const float invLength = 1.0f / std::sqrt(lengthSquared);
                            rotations[i] = glm::vec4(qx * invLength,
                                                     qy * invLength,
                                                     qz * invLength,
                                                     qw * invLength);
                        }
                    }
                });

    // DC-only format: the SH degree control and appearance LOD are
    // no-ops on .splat. Recording the degree is what keeps that from looking
    // like a bug in the viewer.
    // The format cannot declare antialiased training, and vanilla
    // 3DGS (what produces it) trains without the dilation opacity
    // compensation -- so off. Renderer::setAntialiased overrides it.
    cloud->setAntialiased(false);
    cloud->setShDegree(0);
    cloud->computeBounds();

    return cloud;
}

} // namespace sabercat
