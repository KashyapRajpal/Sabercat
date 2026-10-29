// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/spz_reader.h"

#include <load-spz.h>
#include <splat-types.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "sabercat/gaussian_cloud.h"
#include "sabercat/logger.h"
#include "sabercat/parallel_for.h"
#include "sabercat/vk_check.h"

namespace sabercat
{

namespace
{

// --- Canonical-contract conversions ---
// spz hands back the UNPACKED float form, which (like .ply, unlike .splat) is
// pre-activation. So the adaptation is the same shape as PLYReader's: exp the
// log-space scale, sigmoid the pre-sigmoid alpha, and take the DC term as a
// raw f_dc. Working at this layer means the byte-level quantization detail the
// the spz format defines (b/255, exp(b/16 - 10), the smallest-three quaternion) is
// entirely spz's problem, already applied before we see the data.

// This renderer's canonical layout holds (3+1)^2 = 16 SH coefficients. spz can
// carry degree 4 (25 slots, 24 above DC); anything above degree 3 is truncated.
static constexpr int kMaxSupportedShDegree = 3;

// spz stores its own coordinate frame; ask for RUB explicitly rather than
// relying on the file's default. RUB (right/up/back, three.js) is this
// renderer's convention, so a correctly-authored .spz needs no axis flip.
static constexpr spz::CoordinateSystem kTargetCoordinateSystem =
    spz::CoordinateSystem::RUB;

// Higher-order coefficients PER CHANNEL for a given degree: (d+1)^2 - 1.
// spz's own layout comment lists the x3-channel totals (0/9/24/45/72).
int coefficientsPerChannel(int degree)
{
    return (degree + 1) * (degree + 1) - 1;
}

float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

// Reads the whole file and normalizes it to a container spz::loadSpz accepts.
// spz recognizes two framings by their first bytes: a gzip member (1f 8b)
// wrapping the v1-3 single NGSP stream, and the v4+ NGSP container (zstd
// streams behind a plain "NGSP" header). Some tools write the v1-3 NGSP stream
// raw, without the gzip wrapper -- same "NGSP" magic, but a version below 4,
// which spz rejects as an unsupported v4 container. spz exposes no public entry
// point for an un-gzipped legacy stream, so wrap it in gzip in memory and feed
// spz's normal path; it inflates back to the identical bytes, so the decode is
// bit-exact with the gzipped original.
std::vector<uint8_t> readSpzContainer(const std::string & path)
{
    std::ifstream file(std::filesystem::u8path(path), std::ios::binary | std::ios::ate);
    VK_VALIDATE(file.is_open(), "SpzReader: failed to open '" + path + "'");
    std::vector<uint8_t> bytes(static_cast<size_t>(file.tellg()));
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    VK_VALIDATE(file.good() || bytes.empty(), "SpzReader: failed to read '" + path + "'");

    if (bytes.size() >= 2 && bytes[0] == 0x1f && bytes[1] == 0x8b)
    {
        return bytes; // gzip-wrapped v1-3 stream -- spz's legacy path
    }

    uint32_t magic = 0;
    uint32_t version = 0;
    if (bytes.size() >= 8)
    {
        std::memcpy(&magic, bytes.data(), sizeof(magic));
        std::memcpy(&version, bytes.data() + 4, sizeof(version));
    }
    if (magic == spz::NGSP_MAGIC)
    {
        if (version >= static_cast<uint32_t>(spz::MIN_ZSTD_SPZ_HEADER_VERSION))
        {
            return bytes; // v4+ container -- spz reads it as-is
        }
        std::vector<uint8_t> wrapped;
        VK_VALIDATE(spz::compressGzipped(bytes.data(), bytes.size(), &wrapped),
                    "SpzReader: failed to re-wrap the raw NGSP stream of '" + path + "'");
        return wrapped;
    }

    // Name what was actually found: the usual culprit is a different format
    // under a .spz extension, and its leading bytes say which.
    char found[3 * 4 + 1] = {};
    const size_t shown = std::min<size_t>(bytes.size(), 4);
    for (size_t i = 0; i < shown; ++i)
    {
        std::snprintf(found + i * 3, 4, i + 1 < shown ? "%02x " : "%02x", bytes[i]);
    }
    VK_VALIDATE(false, "SpzReader: '" + path + "' is not a .spz -- expected gzip (1f 8b) or NGSP ("
                           "4e 47 53 50) magic, found "
                           + (shown == 0 ? std::string("an empty file") : std::string(found)));
    return {}; // unreachable
}

} // namespace

std::shared_ptr<GaussianCloud> SpzReader::read(Device & device, const std::string & path)
{
    spz::UnpackOptions options;
    options.to = kTargetCoordinateSystem;

    // loadSpz reports failure by returning an empty cloud rather than throwing,
    // so an unreadable/corrupt/wrong-format file surfaces as numPoints == 0.
    // Turn that into this codebase's normal error path (VK_VALIDATE ->
    // std::runtime_error, caught by apps at the boundary).
    const spz::GaussianCloud source = spz::loadSpz(readSpzContainer(path), options);
    VK_VALIDATE(source.numPoints > 0,
                "SpzReader: failed to load '" + path
                    + "' -- not a readable .spz (bad magic, unsupported version, corrupt "
                      "compressed stream, or empty scene)");

    const size_t pointCount = static_cast<size_t>(source.numPoints);

    // Guard the parallel loop's indexing against a malformed cloud before it
    // can read out of bounds -- spz is trusted, but these are cheap and turn a
    // would-be crash into a named error.
    VK_VALIDATE(source.positions.size() == pointCount * 3,
                "SpzReader: position stream is " + std::to_string(source.positions.size())
                    + " floats, expected " + std::to_string(pointCount * 3) + ": " + path);
    VK_VALIDATE(source.scales.size() == pointCount * 3,
                "SpzReader: scale stream size does not match point count: " + path);
    VK_VALIDATE(source.rotations.size() == pointCount * 4,
                "SpzReader: rotation stream size does not match point count: " + path);
    VK_VALIDATE(source.alphas.size() == pointCount,
                "SpzReader: alpha stream size does not match point count: " + path);
    VK_VALIDATE(source.colors.size() == pointCount * 3,
                "SpzReader: color stream size does not match point count: " + path);

    // Truncate rather than reject: a degree cap is a correct band-ascending
    // prefix read (the same operation the SH evaluation performs at runtime), so a
    // degree-4 file still renders correctly, just without its top band.
    const int sourceDegree = source.shDegree;
    const int usedDegree = std::min(sourceDegree, kMaxSupportedShDegree);
    if (sourceDegree > kMaxSupportedShDegree)
    {
        Logger::warning("SpzReader: file carries SH degree " + std::to_string(sourceDegree)
                     + "; this renderer's canonical layout holds degree "
                     + std::to_string(kMaxSupportedShDegree)
                     + ", so the top band(s) are dropped");
    }

    const int sourceCoeffs = coefficientsPerChannel(sourceDegree);
    const int usedCoeffs = coefficientsPerChannel(usedDegree);
    VK_VALIDATE(source.sh.size() == pointCount * static_cast<size_t>(sourceCoeffs) * 3,
                "SpzReader: SH stream is " + std::to_string(source.sh.size())
                    + " floats, expected "
                    + std::to_string(pointCount * static_cast<size_t>(sourceCoeffs) * 3)
                    + " for degree " + std::to_string(sourceDegree) + ": " + path);
    VK_VALIDATE(usedCoeffs <= kShCoefficientsPerGaussian - 1,
                "SpzReader: SH coefficient count exceeds the canonical layout: " + path);

    auto cloud = std::make_shared<GaussianCloud>(device);
    cloud->initialize(pointCount);

    auto & positions = cloud->positions();
    auto & scales = cloud->scales();
    auto & rotations = cloud->rotations();
    auto & sh = cloud->shCoefficients();
    auto & opacities = cloud->opacities();

    // Same disjoint-range contract as PLYReader/SplatReader: each worker writes
    // only indices [begin, end) of vectors initialize() already sized, so no
    // locking is needed and no worker shares mutable state.
    parallelFor(pointCount,
                [&](size_t begin, size_t end)
                {
                    for (size_t i = begin; i < end; ++i)
                    {
                        // Streams pad to the canonical 16 B stride; the w lane
                        // is unused padding, not data.
                        positions[i] = glm::vec4(source.positions[i * 3 + 0],
                                                 source.positions[i * 3 + 1],
                                                 source.positions[i * 3 + 2],
                                                 0.0f);

                        // Log-space, exactly like .ply's scale_* (and unlike
                        // .splat, which is already linear).
                        scales[i] = glm::vec4(std::exp(source.scales[i * 3 + 0]),
                                              std::exp(source.scales[i * 3 + 1]),
                                              std::exp(source.scales[i * 3 + 2]),
                                              0.0f);

                        opacities[i] = sigmoid(source.alphas[i]);

                        // ALREADY xyzw -- no reorder. This is the one format of
                        // the three where the source order matches canonical,
                        // so "reorder like the others" would be the bug here.
                        // Renormalize anyway: quantization to smallest-three
                        // and back does not guarantee unit length.
                        const float qx = source.rotations[i * 4 + 0];
                        const float qy = source.rotations[i * 4 + 1];
                        const float qz = source.rotations[i * 4 + 2];
                        const float qw = source.rotations[i * 4 + 3];
                        const float lengthSquared = qx * qx + qy * qy + qz * qz + qw * qw;
                        if (lengthSquared > 1e-12f)
                        {
                            const float invLength = 1.0f / std::sqrt(lengthSquared);
                            rotations[i] = glm::vec4(qx * invLength,
                                                     qy * invLength,
                                                     qz * invLength,
                                                     qw * invLength);
                        }
                        else
                        {
                            rotations[i] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f); // identity
                        }

                        glm::vec4 * shOut = &sh[i * kShCoefficientsPerGaussian];

                        // spz's `colors` IS the SH DC term already -- a real
                        // f_dc, not a display value. Nothing to invert (that
                        // inversion is .splat's problem, not this format's).
                        shOut[0] = glm::vec4(source.colors[i * 3 + 0],
                                             source.colors[i * 3 + 1],
                                             source.colors[i * 3 + 2],
                                             0.0f);

                        // Coefficient-major with the CHANNEL as the inner axis
                        // (sh[k*3 + c]) -- the opposite of .ply's channel-major
                        // f_rest, and the classic way to get a .spz loader
                        // subtly wrong: reading it channel-major still produces
                        // a full-looking image, just with the bands scrambled.
                        const size_t base = i * static_cast<size_t>(sourceCoeffs) * 3;
                        for (int k = 0; k < usedCoeffs; ++k)
                        {
                            shOut[1 + k] = glm::vec4(source.sh[base + k * 3 + 0],
                                                     source.sh[base + k * 3 + 1],
                                                     source.sh[base + k * 3 + 2],
                                                     0.0f);
                        }
                        // Slots above usedCoeffs stay zero-initialized, whether
                        // because the scene's degree is lower or because a
                        // degree-4 top band was truncated above.
                    }
                });

    cloud->setShDegree(static_cast<uint32_t>(usedDegree));
    // The header's antialiased flag says the scene was trained with
    // the dilation opacity compensation (mip-splatting/gsplat antialiased
    // mode). It decides the forward model -- and so the gradients -- not just
    // the look: racoonfamily.spz sets it, hornedlizard.spz does not.
    cloud->setAntialiased(source.antialiased);
    cloud->computeBounds();
    return cloud;
}

} // namespace sabercat
