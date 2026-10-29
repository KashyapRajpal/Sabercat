// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/ksplat_reader.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "sabercat/gaussian_cloud.h"
#include "sabercat/logger.h"
#include "sabercat/parallel_for.h"
#include "sabercat/render_config.h"
#include "sabercat/vk_check.h"

namespace sabercat
{

namespace
{

// --- File layout (GaussianSplats3D SplatBuffer, v0.1) ---
// Every offset below is the JS reference's typed-array index times the
// element size: e.g. `headerArrayUint16[10]` is byte 20.
static constexpr uint64_t kHeaderSize = 4096;
static constexpr uint64_t kSectionHeaderSize = 1024;

static constexpr size_t kHeaderVersionMajor = 0;       // u8
static constexpr size_t kHeaderVersionMinor = 1;       // u8
static constexpr size_t kHeaderMaxSectionCount = 4;    // u32
static constexpr size_t kHeaderMaxSplatCount = 12;     // u32
static constexpr size_t kHeaderSplatCount = 16;        // u32
static constexpr size_t kHeaderCompressionLevel = 20;  // u16
static constexpr size_t kHeaderMinShCoeff = 36;        // f32
static constexpr size_t kHeaderMaxShCoeff = 40;        // f32

static constexpr size_t kSectionMaxSplatCount = 4;           // u32
static constexpr size_t kSectionBucketSize = 8;              // u32
static constexpr size_t kSectionBucketCount = 12;            // u32
static constexpr size_t kSectionBucketBlockSize = 16;        // f32
static constexpr size_t kSectionBucketStorageSize = 20;      // u16
static constexpr size_t kSectionCompressionScaleRange = 24;  // u32
static constexpr size_t kSectionFullBucketCount = 32;        // u32
static constexpr size_t kSectionPartialBucketCount = 36;     // u32
static constexpr size_t kSectionShDegree = 40;               // u16

// The only version this layout describes. The reference's KSplatLoader
// accepts any minor >= 1 within major 0 (and, optimistically, any later
// major); a later major has no layout we could know, so it is rejected.
static constexpr uint8_t kSupportedMajor = 0;
static constexpr uint8_t kMinSupportedMinor = 1;

// A bucket center is 3 x f32. The reference reads centers with a fixed
// 3-float stride but sizes the storage from this header field, so any other
// value would make the JS misread its own file.
static constexpr uint32_t kBucketStorageBytes = 12;

// SplatBuffer.CompressionLevels[1|2].ScaleRange: the u16 center code's
// zero point, substituted when the section header stores 0.
static constexpr uint32_t kDefaultCompressionScaleRange = 32767;

// Constants.SphericalHarmonics8BitCompressionRange / 2: the reference
// substitutes these when the header's min/max are 0 (or NaN) -- its
// `header[9] || -1.5` -- so a level-2 file is read the same way here.
static constexpr float kDefaultShHalfRange = 1.5f;

// The reference defines layouts for SH degree <= 2 only (component counts
// 0/9/24; its record-size tables and its .ply parser both stop at 2).
static constexpr uint32_t kMaxShDegree = 2;
static constexpr uint32_t kShComponentsDegree1 = 9;  // 3 coefficients x RGB
static constexpr uint32_t kShComponentsDegree2 = 24; // + 5 coefficients x RGB

// Per-level record layout (SplatBuffer.CompressionLevels). Level 0 is all
// f32; levels 1 and 2 halve center/scale/rotation (u16 bucket-relative
// centers, f16 scale and rotation) and differ only in SH width.
struct LevelLayout
{
    uint32_t scaleOffset;
    uint32_t rotationOffset;
    uint32_t colorOffset;
    uint32_t shOffset;
    uint32_t bytesPerShComponent;
};
static constexpr LevelLayout kLevels[3] = {
    {12, 24, 40, 44, 4},
    {6, 12, 20, 24, 2},
    {6, 12, 20, 24, 1},
};

// --- Canonical-contract conversions ---
static constexpr float kByteToUnit = 1.0f / 255.0f;
static constexpr float kInvShC0 = 1.0f / kShC0;
static constexpr float kDcDisplayOffset = 0.5f;
static constexpr float kMinQuatLengthSquared = 1e-12f;

// Everything needed to decode one section, resolved once from its header.
// Fields grouped by size so the struct is padding-free.
struct SectionLayout
{
    uint64_t base = 0;            // section start (partial-bucket lengths)
    uint64_t bucketsOffset = 0;   // first bucket center
    uint64_t dataOffset = 0;      // first splat record
    uint64_t firstSplat = 0;      // global index of this section's splat 0
    double positionScale = 0.0;   // (bucketBlockSize / 2) / compressionScaleRange
    std::vector<uint32_t> partialBucketEnds; // exclusive local end of each partial bucket
    uint32_t splatCount = 0;
    uint32_t bucketSize = 0;
    uint32_t bucketCount = 0;
    uint32_t fullBucketCount = 0;
    uint32_t partialBucketCount = 0;
    uint32_t compressionScaleRange = 0;
    uint32_t shDegree = 0;
    uint32_t bytesPerSplat = 0;
};

template <typename T> T readAt(const uint8_t * data, uint64_t offset)
{
    T value;
    std::memcpy(&value, data + offset, sizeof(T));
    return value;
}

// IEEE 754 binary16 -> float, what THREE.DataUtils.fromHalfFloat computes
// (subnormals, infinities and NaN included).
float halfToFloat(uint16_t half)
{
    const uint32_t sign = static_cast<uint32_t>(half & 0x8000u) << 16;
    const uint32_t exponent = (half >> 10) & 0x1Fu;
    const uint32_t mantissa = half & 0x3FFu;
    if (exponent == 0)
    {
        // Zero or subnormal: mantissa * 2^-24, exact in float.
        const float magnitude = std::ldexp(static_cast<float>(mantissa), -24);
        return sign ? -magnitude : magnitude;
    }
    uint32_t bits;
    if (exponent == 0x1Fu)
    {
        bits = sign | 0x7F800000u | (mantissa << 13); // inf / NaN
    }
    else
    {
        bits = sign | ((exponent + 112u) << 23) | (mantissa << 13); // rebias 15 -> 127
    }
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

uint32_t shComponentsForDegree(uint32_t degree)
{
    return degree == 0 ? 0u : (degree == 1 ? kShComponentsDegree1 : kShComponentsDegree2);
}

// Which bucket a section-local splat sits in: full buckets first, each
// `bucketSize` long, then the partially-filled ones in file order (the
// reference's SplatBuffer.getBucketIndex).
uint32_t bucketIndexFor(const SectionLayout & section, uint32_t localSplat)
{
    const uint64_t fullSplats = static_cast<uint64_t>(section.fullBucketCount) * section.bucketSize;
    if (localSplat < fullSplats)
    {
        return localSplat / section.bucketSize;
    }
    const auto it = std::upper_bound(section.partialBucketEnds.begin(),
                                     section.partialBucketEnds.end(), localSplat);
    return section.fullBucketCount
           + static_cast<uint32_t>(it - section.partialBucketEnds.begin());
}

// Resolves every section's layout, validating each structural claim against
// the file size before anything indexes through it.
std::vector<SectionLayout> parseSections(const uint8_t * data, uint64_t fileSize,
                                         uint32_t compressionLevel, uint32_t maxSectionCount,
                                         const std::string & path)
{
    const LevelLayout & level = kLevels[compressionLevel];
    std::vector<SectionLayout> sections(maxSectionCount);

    uint64_t sectionBase = kHeaderSize + static_cast<uint64_t>(maxSectionCount) * kSectionHeaderSize;
    uint64_t firstSplat = 0;
    for (uint32_t s = 0; s < maxSectionCount; ++s)
    {
        const uint64_t headerOffset = kHeaderSize + static_cast<uint64_t>(s) * kSectionHeaderSize;
        const std::string where = "KsplatReader: section " + std::to_string(s) + " of " + path;
        SectionLayout & section = sections[s];

        section.splatCount = readAt<uint32_t>(data, headerOffset + kSectionMaxSplatCount);
        section.bucketSize = readAt<uint32_t>(data, headerOffset + kSectionBucketSize);
        section.bucketCount = readAt<uint32_t>(data, headerOffset + kSectionBucketCount);
        const float bucketBlockSize = readAt<float>(data, headerOffset + kSectionBucketBlockSize);
        const uint16_t bucketStorageBytes =
            readAt<uint16_t>(data, headerOffset + kSectionBucketStorageSize);
        section.compressionScaleRange =
            readAt<uint32_t>(data, headerOffset + kSectionCompressionScaleRange);
        section.fullBucketCount = readAt<uint32_t>(data, headerOffset + kSectionFullBucketCount);
        section.partialBucketCount =
            readAt<uint32_t>(data, headerOffset + kSectionPartialBucketCount);
        section.shDegree = readAt<uint16_t>(data, headerOffset + kSectionShDegree);

        VK_VALIDATE(section.shDegree <= kMaxShDegree,
                    where + " declares SH degree " + std::to_string(section.shDegree)
                        + "; the .ksplat reference defines a record layout only up to degree 2");
        section.bytesPerSplat = level.shOffset
                                + shComponentsForDegree(section.shDegree) * level.bytesPerShComponent;

        // The references (GaussianSplats3D SplatBuffer.parseSectionHeaders and
        // splat-transform read-ksplat) size bucket storage from the header at
        // EVERY level and the partial-bucket lengths at 4 B each; the splat
        // records follow both. GaussianSplats3D writes bucketStorageSize 0 at
        // level 0, so this is a no-op there -- but splat-transform's level-0
        // files carry a real bucket table (12 B) that must be skipped.
        const uint64_t bucketBytes = static_cast<uint64_t>(bucketStorageBytes) * section.bucketCount;
        const uint64_t partialBytes = static_cast<uint64_t>(section.partialBucketCount) * 4;
        const uint64_t recordBytes = static_cast<uint64_t>(section.bytesPerSplat) * section.splatCount;
        section.base = sectionBase;
        section.bucketsOffset = sectionBase + partialBytes;
        section.dataOffset = section.bucketsOffset + bucketBytes;
        section.firstSplat = firstSplat;
        VK_VALIDATE(section.dataOffset + recordBytes <= fileSize,
                    where + " needs " + std::to_string(section.dataOffset + recordBytes)
                        + " bytes but the file has " + std::to_string(fileSize) + " (truncated?)");

        if (compressionLevel >= 1 && section.splatCount > 0)
        {
            VK_VALIDATE(bucketStorageBytes == kBucketStorageBytes,
                        where + " has a bucket record size of " + std::to_string(bucketStorageBytes)
                            + " bytes; the format stores 3 x f32 (12) per bucket center");
            VK_VALIDATE(std::isfinite(bucketBlockSize) && bucketBlockSize > 0.0f,
                        where + " has a non-positive bucket block size");
            VK_VALIDATE(section.bucketSize > 0, where + " has a zero bucket size");
            VK_VALIDATE(static_cast<uint64_t>(section.fullBucketCount) + section.partialBucketCount
                            == section.bucketCount,
                        where + ": full (" + std::to_string(section.fullBucketCount)
                            + ") + partially-filled (" + std::to_string(section.partialBucketCount)
                            + ") buckets != bucket count (" + std::to_string(section.bucketCount)
                            + ")");

            // Every splat must land in exactly one bucket, or a center index
            // would run past the bucket array.
            uint64_t covered = static_cast<uint64_t>(section.fullBucketCount) * section.bucketSize;
            section.partialBucketEnds.resize(section.partialBucketCount);
            for (uint32_t b = 0; b < section.partialBucketCount; ++b)
            {
                covered += readAt<uint32_t>(data, section.base + static_cast<uint64_t>(b) * 4);
                VK_VALIDATE(covered <= section.splatCount,
                            where + ": bucket lengths cover more splats than the section holds");
                section.partialBucketEnds[b] = static_cast<uint32_t>(covered);
            }
            VK_VALIDATE(covered == section.splatCount,
                        where + ": buckets cover " + std::to_string(covered) + " splats, the section holds "
                            + std::to_string(section.splatCount));

            if (section.compressionScaleRange == 0)
            {
                section.compressionScaleRange = kDefaultCompressionScaleRange;
            }
            // In double, as the JS computes it, so decoded centers match the
            // reference to the last float bit.
            section.positionScale = (static_cast<double>(bucketBlockSize) / 2.0)
                                    / static_cast<double>(section.compressionScaleRange);
        }

        sectionBase = section.dataOffset + recordBytes;
        firstSplat += section.splatCount;
    }

    if (sectionBase < fileSize)
    {
        Logger::warning("KsplatReader: " + std::to_string(fileSize - sectionBase)
                        + " trailing bytes after the last section ignored: " + path);
    }
    return sections;
}

} // namespace

std::shared_ptr<GaussianCloud> KsplatReader::read(Device & device, const std::string & path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    VK_VALIDATE(file.is_open(), "KsplatReader: failed to open " + path);
    const std::streamsize streamSize = file.tellg();
    file.seekg(0, std::ios::beg);
    VK_VALIDATE(streamSize > 0, "KsplatReader: file is empty: " + path);
    const uint64_t fileSize = static_cast<uint64_t>(streamSize);
    VK_VALIDATE(fileSize >= kHeaderSize,
                "KsplatReader: file is " + std::to_string(fileSize)
                    + " bytes, smaller than the 4096-byte .ksplat header: " + path);

    std::vector<uint8_t> raw(static_cast<size_t>(fileSize));
    file.read(reinterpret_cast<char *>(raw.data()), streamSize);
    VK_VALIDATE(file.gcount() == streamSize, "KsplatReader: truncated read of " + path);
    const uint8_t * data = raw.data();

    // --- Header ---
    const uint8_t major = data[kHeaderVersionMajor];
    const uint8_t minor = data[kHeaderVersionMinor];
    VK_VALIDATE(major == kSupportedMajor && minor >= kMinSupportedMinor,
                "KsplatReader: unsupported .ksplat version " + std::to_string(major) + "."
                    + std::to_string(minor)
                    + " (this reader implements 0.1+, the sectioned layout GaussianSplats3D "
                      "writes; re-export older files with a current create-ksplat): "
                    + path);
    const uint32_t compressionLevel = readAt<uint16_t>(data, kHeaderCompressionLevel);
    VK_VALIDATE(compressionLevel <= 2,
                "KsplatReader: unknown compression level " + std::to_string(compressionLevel)
                    + " (the format defines 0, 1 and 2): " + path);
    const uint32_t maxSectionCount = readAt<uint32_t>(data, kHeaderMaxSectionCount);
    const uint32_t headerSplatCount = readAt<uint32_t>(data, kHeaderMaxSplatCount);
    const uint32_t headerLoadedCount = readAt<uint32_t>(data, kHeaderSplatCount);
    VK_VALIDATE(maxSectionCount > 0, "KsplatReader: header declares no sections: " + path);
    VK_VALIDATE(kHeaderSize + static_cast<uint64_t>(maxSectionCount) * kSectionHeaderSize <= fileSize,
                "KsplatReader: header declares " + std::to_string(maxSectionCount)
                    + " section headers, more than the file can hold: " + path);

    float shMin = readAt<float>(data, kHeaderMinShCoeff);
    float shMax = readAt<float>(data, kHeaderMaxShCoeff);
    if (shMin == 0.0f || std::isnan(shMin))
    {
        shMin = -kDefaultShHalfRange;
    }
    if (shMax == 0.0f || std::isnan(shMax))
    {
        shMax = kDefaultShHalfRange;
    }
    VK_VALIDATE(std::isfinite(shMin) && std::isfinite(shMax),
                "KsplatReader: non-finite SH quantization range in the header: " + path);

    const std::vector<SectionLayout> sections =
        parseSections(data, fileSize, compressionLevel, maxSectionCount, path);

    uint64_t total = 0;
    uint32_t shDegree = 0;
    for (const SectionLayout & section : sections)
    {
        total += section.splatCount;
        // A mixed-degree file (never written by the reference generator) is
        // reported at its HIGHEST degree: lower-degree sections keep exactly
        // zero in the bands they lack, which renders identically to capping
        // them, so nothing is lost by not taking the reference's minimum.
        shDegree = std::max(shDegree, section.shDegree);
    }
    // GaussianSplats3D writes maxSplatCount (offset 12) == splatCount (offset
    // 16), but playcanvas/splat-transform writes only splatCount and leaves
    // maxSplatCount (and sectionCount) at 0, and its own reader goes by
    // splatCount. Accept either field.
    VK_VALIDATE(total == headerSplatCount || total == headerLoadedCount,
                "KsplatReader: sections hold " + std::to_string(total) + " splats, the header says "
                    + std::to_string(headerSplatCount) + " (max) / " + std::to_string(headerLoadedCount)
                    + " (loaded): " + path);
    VK_VALIDATE(total > 0, "KsplatReader: file contains no splats: " + path);
    VK_VALIDATE(total <= std::numeric_limits<uint32_t>::max(),
                "KsplatReader: splat count exceeds 2^32: " + path);

    auto cloud = std::make_shared<GaussianCloud>(device);
    cloud->initialize(static_cast<size_t>(total));

    auto & positions = cloud->positions();
    auto & scales = cloud->scales();
    auto & rotations = cloud->rotations();
    auto & sh = cloud->shCoefficients();
    auto & opacities = cloud->opacities();
    const LevelLayout & level = kLevels[compressionLevel];
    const double shRange = static_cast<double>(shMax) - static_cast<double>(shMin);

    // One SH component, widened to float per level. Level 2 is the reference's
    // fromUint8: v / 255 * range + min, in double.
    const auto readSh = [&](const uint8_t * record, uint32_t component) -> float
    {
        const uint8_t * at = record + level.shOffset + component * level.bytesPerShComponent;
        switch (compressionLevel)
        {
        case 0:
            return readAt<float>(at, 0);
        case 1:
            return halfToFloat(readAt<uint16_t>(at, 0));
        default:
            return static_cast<float>(static_cast<double>(*at) / 255.0 * shRange
                                      + static_cast<double>(shMin));
        }
    };
    // Scale/rotation lanes: f32 at level 0, f16 above.
    const auto readFloatLane = [&](const uint8_t * record, uint32_t offset, uint32_t lane) -> float
    {
        return compressionLevel == 0 ? readAt<float>(record, offset + lane * 4)
                                     : halfToFloat(readAt<uint16_t>(record, offset + lane * 2));
    };

    for (const SectionLayout & section : sections)
    {
        // Same disjoint-range contract as the other readers: each worker
        // writes only its own [begin, end) of vectors initialize() sized.
        parallelFor(section.splatCount,
                    [&](size_t begin, size_t end)
                    {
                        for (size_t local = begin; local < end; ++local)
                        {
                            const size_t i = static_cast<size_t>(section.firstSplat) + local;
                            const uint8_t * record =
                                data + section.dataOffset + local * section.bytesPerSplat;

                            if (compressionLevel == 0)
                            {
                                positions[i] = glm::vec4(readAt<float>(record, 0), readAt<float>(record, 4),
                                                         readAt<float>(record, 8), 0.0f);
                            }
                            else
                            {
                                // u16 offset from the bucket center, zero at
                                // compressionScaleRange -- the reference's
                                // (x - sr) * sf + bucketCenter, in double.
                                const uint32_t bucket =
                                    bucketIndexFor(section, static_cast<uint32_t>(local));
                                const uint64_t center =
                                    section.bucketsOffset + static_cast<uint64_t>(bucket) * kBucketStorageBytes;
                                const double range = static_cast<double>(section.compressionScaleRange);
                                float out[3];
                                for (uint32_t axis = 0; axis < 3; ++axis)
                                {
                                    const double code = readAt<uint16_t>(record, axis * 2);
                                    out[axis] = static_cast<float>(
                                        (code - range) * section.positionScale
                                        + static_cast<double>(readAt<float>(data, center + axis * 4)));
                                }
                                positions[i] = glm::vec4(out[0], out[1], out[2], 0.0f);
                            }

                            // ALREADY LINEAR: the exporter exp()s scale_* on
                            // the way in. Applying exp() here would inflate
                            // every Gaussian.
                            scales[i] = glm::vec4(readFloatLane(record, level.scaleOffset, 0),
                                                  readFloatLane(record, level.scaleOffset, 1),
                                                  readFloatLane(record, level.scaleOffset, 2), 0.0f);

                            // Stored w,x,y,z: lane 0 is the reference
                            // reader's W. Reorder to canonical xyzw and
                            // renormalize -- f16 rounding alone moves the
                            // length off 1.
                            const float qw = readFloatLane(record, level.rotationOffset, 0);
                            const float qx = readFloatLane(record, level.rotationOffset, 1);
                            const float qy = readFloatLane(record, level.rotationOffset, 2);
                            const float qz = readFloatLane(record, level.rotationOffset, 3);
                            const float lengthSquared = qw * qw + qx * qx + qy * qy + qz * qz;
                            if (lengthSquared < kMinQuatLengthSquared || !std::isfinite(lengthSquared))
                            {
                                rotations[i] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f); // identity, xyzw
                            }
                            else
                            {
                                const float invLength = 1.0f / std::sqrt(lengthSquared);
                                rotations[i] = glm::vec4(qx * invLength, qy * invLength,
                                                         qz * invLength, qw * invLength);
                            }

                            // Color is a DISPLAY byte, floor((0.5 + SH_C0 *
                            // f_dc) * 255); b/255 is exactly what the
                            // reference viewer feeds its shader, so inverting
                            // that recovers the f_dc it renders.
                            const uint8_t * color = record + level.colorOffset;
                            glm::vec4 * shOut = &sh[i * kShCoefficientsPerGaussian];
                            shOut[0] = glm::vec4((color[0] * kByteToUnit - kDcDisplayOffset) * kInvShC0,
                                                 (color[1] * kByteToUnit - kDcDisplayOffset) * kInvShC0,
                                                 (color[2] * kByteToUnit - kDcDisplayOffset) * kInvShC0,
                                                 0.0f);
                            opacities[i] = color[3] * kByteToUnit;

                            // SH is grouped per BAND and channel-major inside
                            // it: component (band base + c * bandWidth + k) is
                            // channel c of the band's k-th coefficient. The
                            // canonical layout is coefficient-major, so this
                            // is a transpose within each band.
                            if (section.shDegree >= 1)
                            {
                                for (uint32_t k = 0; k < 3; ++k)
                                {
                                    shOut[1 + k] = glm::vec4(readSh(record, 0 * 3 + k),
                                                             readSh(record, 1 * 3 + k),
                                                             readSh(record, 2 * 3 + k), 0.0f);
                                }
                            }
                            if (section.shDegree >= 2)
                            {
                                for (uint32_t k = 0; k < 5; ++k)
                                {
                                    shOut[4 + k] = glm::vec4(readSh(record, 9 + 0 * 5 + k),
                                                             readSh(record, 9 + 1 * 5 + k),
                                                             readSh(record, 9 + 2 * 5 + k), 0.0f);
                                }
                            }
                            // Higher slots stay zero from initialize().
                        }
                    });
    }

    // Structural checks cannot catch garbage records inside a well-formed
    // container. Positions and scales feed the covariance and the bounds, so a
    // non-finite or negative one is rejected here, by index, rather than
    // rendered as noise. (Zero scales are legitimate: f16 flushes very thin
    // axes to 0, and the rasterizer's low-pass dilation handles them.)
    for (size_t i = 0; i < positions.size(); ++i)
    {
        const glm::vec4 & p = positions[i];
        const glm::vec4 & s = scales[i];
        VK_VALIDATE(std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z),
                    "KsplatReader: splat " + std::to_string(i) + " has a non-finite position: " + path);
        VK_VALIDATE(std::isfinite(s.x) && std::isfinite(s.y) && std::isfinite(s.z) && s.x >= 0.0f
                        && s.y >= 0.0f && s.z >= 0.0f,
                    "KsplatReader: splat " + std::to_string(i)
                        + " has a negative or non-finite scale (scales are stored linear): " + path);
    }

    cloud->setShDegree(shDegree);
    // The format cannot declare antialiased training (and the vanilla 3DGS
    // .ply files it is converted from do not train with it) -- off, as for
    // .ply/.splat. Renderer::setAntialiased overrides it.
    cloud->setAntialiased(false);
    cloud->computeBounds();
    return cloud;
}

} // namespace sabercat
