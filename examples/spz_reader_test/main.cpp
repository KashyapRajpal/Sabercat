// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Headless test for SpzReader + the loadScene() dispatch.
//
// The verification story here is deliberately different from splat_reader_test.
// SplatReader parses bytes itself, so that test synthesizes fixtures and checks
// the parse. SpzReader parses NOTHING -- container decoding is delegated to the
// reference implementation in external/spz -- so the only Sabercat-owned code is
// the adapter onto the canonical contract, and that is what gets tested.
//
// The centerpiece is testAdapterAgainstReference(): load a REAL Niantic-authored
// .spz twice, once through SpzReader and once through spz::loadSpz directly, and
// check SpzReader's output against canonical values recomputed independently in
// this file from the format's documented semantics. Both paths start from the
// same decoded floats, so the comparison is EXACT -- no quantization tolerance
// to hide a mistake behind -- and it isolates the adapter from the decoder.
// This is the same "independent C++ reimplementation of the same math" pattern
// forward_test uses for the SH basis.
//
// The sample files live in the submodule that must be checked out for the build
// to work at all, so unlike assets/ they are guaranteed present here.

#include <load-spz.h>
#include <splat-types.h>

#include <glm/glm.hpp>
#include <zlib.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../common/scratch_dir.h"
#include "sabercat/approx.h"
#include "sabercat/device.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/render_config.h"
#include "sabercat/scene_loader.h"

using namespace sabercat;

namespace
{

const std::string kSampleDir = SABERCAT_SPZ_SAMPLE_DIR;

#define CHECK(condition, message)                                                                  \
    do                                                                                             \
    {                                                                                              \
        if (!(condition))                                                                          \
        {                                                                                          \
            std::cerr << "FAILED: " << (message) << "\n  at " << __FILE__ << ":" << __LINE__       \
                      << std::endl;                                                                \
            return 1;                                                                              \
        }                                                                                          \
    } while (false)

float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

int coefficientsPerChannel(int degree)
{
    return (degree + 1) * (degree + 1) - 1;
}

bool fileExists(const std::string & path)
{
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

// --- The centerpiece: adapter vs. an independent reimplementation ---
//
// spz hands back pre-activation floats (like .ply, unlike .splat). The canonical
// contract wants: opacity sigmoid-activated, scale exp-activated, SH coefficient
// 0 = raw DC, rotation normalized xyzw. Recomputed here straight from those
// semantics rather than by calling into spz_reader.cpp.
int testAdapterAgainstReference(Device & device, const std::string & name)
{
    const std::string path = kSampleDir + "/" + name;
    CHECK(fileExists(path), "sample .spz missing (is the spz submodule checked out?): " + path);

    auto cloud = loadScene(device, path);

    spz::UnpackOptions options;
    options.to = spz::CoordinateSystem::RUB;
    const spz::GaussianCloud source = spz::loadSpz(path, options);

    CHECK(source.numPoints > 0, name + ": reference load produced no points");
    CHECK(cloud->count() == static_cast<size_t>(source.numPoints),
          name + ": SpzReader and the reference disagree on point count");

    const int sourceDegree = source.shDegree;
    const int usedDegree = std::min(sourceDegree, 3);
    CHECK(cloud->shDegree() == static_cast<uint32_t>(usedDegree),
          name + ": reported SH degree does not match the file's (capped at 3)");
    // The header's antialiased flag must reach the cloud -- it picks
    // the forward model (opacity compensation on/off) and so the gradients.
    CHECK(cloud->antialiased() == source.antialiased,
          name + ": antialiased flag does not match the file header");
    std::cout << "  " << name << ": antialiased=" << (cloud->antialiased() ? 1 : 0) << std::endl;

    const int sourceCoeffs = coefficientsPerChannel(sourceDegree);
    const int usedCoeffs = coefficientsPerChannel(usedDegree);

    // Both sides start from identical decoded floats, so these are exact up to
    // the float error of one exp()/sigmoid() -- not a quantization tolerance.
    constexpr float kEps = 1e-6f;

    size_t scalesAwayFromOne = 0;
    size_t opacitiesInterior = 0;
    size_t shInterleaveWitnesses = 0;

    for (size_t i = 0; i < cloud->count(); ++i)
    {
        // Position: passthrough.
        CHECK(approxEqual(cloud->positions()[i].x, source.positions[i * 3 + 0], kEps)
                  && approxEqual(cloud->positions()[i].y, source.positions[i * 3 + 1], kEps)
                  && approxEqual(cloud->positions()[i].z, source.positions[i * 3 + 2], kEps),
              name + ": position mismatch");

        // Scale: log -> linear. A missing exp() shows up immediately.
        for (int axis = 0; axis < 3; ++axis)
        {
            const float expected = std::exp(source.scales[i * 3 + axis]);
            const float actual = (axis == 0)   ? cloud->scales()[i].x
                                 : (axis == 1) ? cloud->scales()[i].y
                                               : cloud->scales()[i].z;
            CHECK(approxEqualRelative(actual, expected),
                  name + ": scale is not exp() of the stored log-space value");
            if (std::abs(expected - 1.0f) > 0.05f)
            {
                ++scalesAwayFromOne;
            }
        }

        // Opacity: pre-sigmoid -> [0,1].
        const float expectedOpacity = sigmoid(source.alphas[i]);
        CHECK(approxEqual(cloud->opacities()[i], expectedOpacity, kEps),
              name + ": opacity is not sigmoid() of the stored pre-activation alpha");
        CHECK(cloud->opacities()[i] >= 0.0f && cloud->opacities()[i] <= 1.0f,
              name + ": opacity outside [0,1]");
        if (cloud->opacities()[i] > 0.02f && cloud->opacities()[i] < 0.98f)
        {
            ++opacitiesInterior;
        }

        // Rotation: already xyzw in spz, so the ORDER must be preserved. Any
        // "reorder like .ply/.splat do" would land w in the wrong lane here.
        const float qx = source.rotations[i * 4 + 0];
        const float qy = source.rotations[i * 4 + 1];
        const float qz = source.rotations[i * 4 + 2];
        const float qw = source.rotations[i * 4 + 3];
        const float len = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
        if (len > 1e-6f)
        {
            const glm::vec4 & r = cloud->rotations()[i];
            CHECK(approxEqual(r.x, qx / len, 1e-5f) && approxEqual(r.y, qy / len, 1e-5f)
                      && approxEqual(r.z, qz / len, 1e-5f) && approxEqual(r.w, qw / len, 1e-5f),
                  name + ": rotation mismatch -- check that xyzw order is PRESERVED, not reordered");
            const float storedLen = std::sqrt(r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w);
            CHECK(approxEqual(storedLen, 1.0f, 1e-5f), name + ": stored rotation is not unit length");
        }

        // SH coefficient 0 is spz's `colors` -- already a raw f_dc, so unlike
        // .splat there is nothing to invert. Inverting it anyway would fail here.
        const glm::vec4 & dc = cloud->shCoefficients()[i * kShCoefficientsPerGaussian];
        CHECK(approxEqual(dc.x, source.colors[i * 3 + 0], kEps)
                  && approxEqual(dc.y, source.colors[i * 3 + 1], kEps)
                  && approxEqual(dc.z, source.colors[i * 3 + 2], kEps),
              name + ": SH DC must be spz's colors stream verbatim (it is already raw f_dc)");

        // Higher bands: coefficient-major with the CHANNEL as the inner axis.
        // Reading it channel-major still yields a full-looking image with the
        // bands scrambled, so this is the subtle one.
        const size_t base = i * static_cast<size_t>(sourceCoeffs) * 3;
        for (int k = 0; k < usedCoeffs; ++k)
        {
            const glm::vec4 & c = cloud->shCoefficients()[i * kShCoefficientsPerGaussian + 1 + k];
            const float r = source.sh[base + k * 3 + 0];
            const float g = source.sh[base + k * 3 + 1];
            const float b = source.sh[base + k * 3 + 2];
            CHECK(approxEqual(c.x, r, kEps) && approxEqual(c.y, g, kEps)
                      && approxEqual(c.z, b, kEps),
                  name + ": SH coefficient mismatch -- expected coefficient-major with RGB "
                         "interleaved (sh[k*3 + c]), not .ply's channel-major layout");

            // Degeneracy witness: would a channel-major misread actually differ
            // here? If every coefficient happened to be equal across channels,
            // the check above would pass under either interpretation.
            if (usedCoeffs > 1)
            {
                const float channelMajorAlternative = source.sh[base + k];
                if (std::abs(channelMajorAlternative - r) > 1e-4f)
                {
                    ++shInterleaveWitnesses;
                }
            }
        }

        // Coefficients above the file's degree must be exactly zero.
        for (int k = usedCoeffs; k < kShCoefficientsPerGaussian - 1; ++k)
        {
            const glm::vec4 & c = cloud->shCoefficients()[i * kShCoefficientsPerGaussian + 1 + k];
            CHECK(c.x == 0.0f && c.y == 0.0f && c.z == 0.0f,
                  name + ": SH slots above the file's degree must be exactly zero");
        }
    }

    // --- Guards that the fixture actually exercised what it claims to ---
    CHECK(scalesAwayFromOne > cloud->count() / 10,
          name + ": degenerate -- almost every scale is ~1.0, where a missing exp() would barely "
                 "move the value");
    CHECK(opacitiesInterior > 0,
          name + ": degenerate -- no opacity is strictly inside (0,1), so sigmoid() is untested");
    if (usedDegree > 0)
    {
        CHECK(shInterleaveWitnesses > 0,
              name + ": degenerate -- a channel-major misread would give the same answer "
                     "everywhere, so the SH layout check proves nothing");
    }

    std::printf("  %-22s OK: %zu points, SH degree %d%s\n",
                name.c_str(),
                cloud->count(),
                usedDegree,
                usedDegree == 0 ? " (DC only -- SH layout check skipped)" : "");
    return 0;
}

// --- Controlled round-trip through spz's own writer ---
//
// The real files prove the adapter matches the decoder. This proves the whole
// path end to end (write -> container -> decode -> adapt) on values chosen to
// catch specific mistakes, which a fixed sample file cannot provide. Tolerances
// are quantization-sized because saveSpz quantizes hard: scale to 8 bits in log
// space, alpha and color to 8 bits, SH to 5/4 bits.
int testControlledRoundTrip(Device & device)
{
    spz::GaussianCloud source;
    source.numPoints = 2;
    source.shDegree = 1; // 3 coefficients/channel -- enough to test interleaving
    source.antialiased = false;

    // Scales straddle 1.0 so a spurious or missing exp() is caught in both
    // directions; log-space here, since that is what spz stores.
    source.positions = {1.5f, -2.25f, 3.0f, -4.0f, 0.5f, 8.0f};
    source.scales = {std::log(0.5f), std::log(2.0f), std::log(1.25f),
                     std::log(3.0f), std::log(0.25f), std::log(1.0f)};
    source.rotations = {0.0f, 0.0f, 0.0f, 1.0f,   // identity, xyzw
                        0.5f, -0.5f, 0.5f, 0.5f}; // all lanes non-zero and distinct in magnitude
    source.alphas = {1.2f, -0.7f};                // pre-sigmoid -> ~0.769, ~0.332
    source.colors = {0.8f, -0.6f, 0.2f, -1.0f, 0.4f, 1.1f}; // raw f_dc
    // Coefficient-major, channel inner: (k0_r,k0_g,k0_b, k1_r,...) per point.
    // Deliberately all different so a channel-major misread cannot match.
    source.sh = {0.10f, 0.20f, 0.30f, 0.40f, 0.50f, 0.60f, 0.70f, 0.80f, 0.90f,
                 -0.10f, -0.20f, -0.30f, -0.40f, -0.50f, -0.60f, -0.70f, -0.80f, -0.90f};

    // Set the flag, so the round trip proves it survives write ->
    // container -> decode -> adapter.
    source.antialiased = true;

    spz::PackOptions packOptions;
    packOptions.from = spz::CoordinateSystem::RUB; // no conversion on the way out

    const std::string path = "spz_reader_test_roundtrip.spz";
    CHECK(spz::saveSpz(source, packOptions, path), "round-trip: saveSpz failed");

    auto cloud = loadScene(device, path);
    CHECK(cloud->count() == 2, "round-trip: wrong point count");
    CHECK(cloud->shDegree() == 1, "round-trip: wrong SH degree");
    CHECK(cloud->antialiased(), "round-trip: antialiased flag lost");

    // Quantization steps, derived rather than guessed:
    constexpr float kPositionTol = 0.01f;  // 24-bit fixed point, 12 fractional bits
    constexpr float kOpacityTol = 2.0f / 255.0f;
    constexpr float kDcTol = 0.05f;        // 8-bit over the color range
    constexpr float kScaleRelTol = 0.05f;  // 8-bit in LOG space -> relative error
    constexpr float kShTol = 0.12f;        // 5-bit sh1 -> genuinely coarse

    for (size_t i = 0; i < 2; ++i)
    {
        CHECK(approxEqual(cloud->positions()[i].x, source.positions[i * 3 + 0], kPositionTol),
              "round-trip: position x");
        CHECK(approxEqual(cloud->positions()[i].z, source.positions[i * 3 + 2], kPositionTol),
              "round-trip: position z");

        CHECK(approxEqualRelative(cloud->scales()[i].x, std::exp(source.scales[i * 3 + 0]),
                                  kScaleRelTol),
              "round-trip: scale x -- must be exp() of log-space input, applied exactly once");
        CHECK(approxEqualRelative(cloud->scales()[i].y, std::exp(source.scales[i * 3 + 1]),
                                  kScaleRelTol),
              "round-trip: scale y -- must be exp() of log-space input, applied exactly once");

        CHECK(approxEqual(cloud->opacities()[i], sigmoid(source.alphas[i]), kOpacityTol),
              "round-trip: opacity must be sigmoid() of the pre-activation alpha");

        const glm::vec4 & dc = cloud->shCoefficients()[i * kShCoefficientsPerGaussian];
        CHECK(approxEqual(dc.x, source.colors[i * 3 + 0], kDcTol), "round-trip: SH DC r");
        CHECK(approxEqual(dc.z, source.colors[i * 3 + 2], kDcTol), "round-trip: SH DC b");

        for (int k = 0; k < 3; ++k)
        {
            const glm::vec4 & c = cloud->shCoefficients()[i * kShCoefficientsPerGaussian + 1 + k];
            CHECK(approxEqual(c.x, source.sh[i * 9 + k * 3 + 0], kShTol),
                  "round-trip: SH coefficient r -- check coefficient-major/RGB-interleaved layout");
            CHECK(approxEqual(c.y, source.sh[i * 9 + k * 3 + 1], kShTol),
                  "round-trip: SH coefficient g -- check coefficient-major/RGB-interleaved layout");
        }

        // Bands above degree 1 are absent and must be exactly zero.
        for (int k = 3; k < kShCoefficientsPerGaussian - 1; ++k)
        {
            const glm::vec4 & c = cloud->shCoefficients()[i * kShCoefficientsPerGaussian + 1 + k];
            CHECK(c.x == 0.0f && c.y == 0.0f && c.z == 0.0f,
                  "round-trip: SH slots above the file's degree must be exactly zero");
        }
    }

    std::cout << "  controlled round-trip OK: saveSpz -> loadScene preserves canonical values "
                 "within quantization"
              << std::endl;
    return 0;
}

// Inflates one gzip member -- the inverse of the wrapper spz puts around a
// v1-3 stream, done here with zlib directly so the fixture does not depend on
// the code under test.
bool gunzip(const std::vector<uint8_t> & compressed, std::vector<uint8_t> * out)
{
    z_stream stream = {};
    if (inflateInit2(&stream, 16 + MAX_WBITS) != Z_OK) // 16+: expect a gzip header
    {
        return false;
    }
    stream.next_in = const_cast<Bytef *>(compressed.data());
    stream.avail_in = static_cast<uInt>(compressed.size());
    uint8_t chunk[16384];
    int status = Z_OK;
    while (status == Z_OK)
    {
        stream.next_out = chunk;
        stream.avail_out = sizeof(chunk);
        status = inflate(&stream, Z_NO_FLUSH);
        out->insert(out->end(), chunk, chunk + (sizeof(chunk) - stream.avail_out));
    }
    inflateEnd(&stream);
    return status == Z_STREAM_END;
}

template <typename T>
bool bitEqual(const std::vector<T> & a, const std::vector<T> & b)
{
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

// --- Raw (un-gzipped) NGSP stream ---
//
// Some tools write the v1-3 NGSP stream without its gzip wrapper. The fixture
// is spz's own v3 output with that wrapper stripped, so the two files hold the
// SAME stream and must decode to bit-identical clouds -- any difference means
// the raw path is not the gzipped path minus the gzip.
int testRawNgspStream(Device & device)
{
    // The real sample, re-encoded as v3: real data, so the comparison covers
    // every stream at SH degree 3 rather than a handful of chosen values.
    spz::UnpackOptions unpackOptions;
    unpackOptions.to = spz::CoordinateSystem::RUB;
    const spz::GaussianCloud source = spz::loadSpz(kSampleDir + "/racoonfamily.spz", unpackOptions);
    CHECK(source.numPoints > 0, "raw NGSP: reference load produced no points");

    spz::PackOptions packOptions;
    packOptions.from = spz::CoordinateSystem::RUB;
    packOptions.version = 3; // the last gzip single-stream version
    std::vector<uint8_t> gzipped;
    CHECK(spz::saveSpz(source, packOptions, &gzipped), "raw NGSP: saveSpz failed");
    CHECK(gzipped.size() >= 2 && gzipped[0] == 0x1f && gzipped[1] == 0x8b,
          "raw NGSP: v3 saveSpz output is not gzip -- the fixture premise is wrong");

    std::vector<uint8_t> raw;
    CHECK(gunzip(gzipped, &raw), "raw NGSP: failed to inflate the v3 fixture");
    CHECK(raw.size() >= 8 && std::memcmp(raw.data(), "NGSP", 4) == 0 && raw[4] == 3,
          "raw NGSP: inflated stream does not start with the NGSP magic and version 3");

    const auto writeFile = [](const std::string & path, const std::vector<uint8_t> & bytes)
    {
        std::ofstream f(path, std::ios::binary | std::ios::trunc);
        f.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
        return f.good();
    };
    const std::string gzPath = "spz_reader_test_v3_gzip.spz";
    const std::string rawPath = "spz_reader_test_v3_raw.spz";
    CHECK(writeFile(gzPath, gzipped) && writeFile(rawPath, raw), "raw NGSP: failed to write fixtures");

    auto fromGzip = loadScene(device, gzPath);
    auto fromRaw = loadScene(device, rawPath);
    CHECK(fromRaw->count() == static_cast<size_t>(source.numPoints), "raw NGSP: wrong point count");
    CHECK(fromRaw->count() == fromGzip->count(), "raw NGSP: point count differs from gzipped");
    CHECK(fromRaw->shDegree() == fromGzip->shDegree() && fromRaw->shDegree() == 3,
          "raw NGSP: SH degree differs from gzipped");
    CHECK(fromRaw->antialiased() == fromGzip->antialiased(),
          "raw NGSP: antialiased flag differs from gzipped");
    CHECK(bitEqual(fromRaw->positions(), fromGzip->positions())
              && bitEqual(fromRaw->scales(), fromGzip->scales())
              && bitEqual(fromRaw->rotations(), fromGzip->rotations())
              && bitEqual(fromRaw->opacities(), fromGzip->opacities())
              && bitEqual(fromRaw->shCoefficients(), fromGzip->shCoefficients()),
          "raw NGSP: decoded streams are not bit-identical to the gzipped file's");

    std::cout << "  raw NGSP stream OK: " << raw.size() << " B un-gzipped v3 (" << gzipped.size()
              << " B gzipped) decodes bit-identical, " << fromRaw->count() << " points" << std::endl;
    return 0;
}

int testRejections(Device & device)
{
    std::string lastError;
    const auto rejects = [&device, &lastError](const std::string & path)
    {
        try
        {
            loadScene(device, path);
            return false;
        }
        catch (const std::exception & e)
        {
            lastError = e.what();
            return true;
        }
    };

    const std::string garbage = "spz_reader_test_garbage.spz";
    {
        std::ofstream f(garbage, std::ios::binary | std::ios::trunc);
        const std::vector<char> junk(4096, static_cast<char>(0xAB));
        f.write(junk.data(), static_cast<std::streamsize>(junk.size()));
    }
    CHECK(rejects(garbage), "rejection: a non-.spz file with a .spz extension must be rejected");
    CHECK(lastError.find("found ab ab ab ab") != std::string::npos,
          "rejection: the error must name the bad magic, got: " + lastError);

    const std::string empty = "spz_reader_test_empty.spz";
    {
        std::ofstream f(empty, std::ios::binary | std::ios::trunc);
    }
    CHECK(rejects(empty), "rejection: an empty .spz must be rejected");

    CHECK(rejects("spz_reader_test_does_not_exist.spz"),
          "rejection: a missing file must be rejected");

    std::cout << "  rejections OK: garbage, empty, and missing files all rejected" << std::endl;
    return 0;
}

} // namespace

int main()
{
    // Scratch files go to a temp folder, not the directory the test runs from.
    examples::ScratchDir scratch("spz_reader_test");
    Device device;
    int result = 0;
    try
    {
        device.initialize({}, /*enableValidation=*/true);

        std::cout << "real Niantic sample files (external/spz/samples):" << std::endl;
        result = testAdapterAgainstReference(device, "hornedlizard.spz");
        if (result == 0)
        {
            result = testAdapterAgainstReference(device, "racoonfamily.spz");
        }
        if (result == 0)
        {
            // The two samples disagree on the flag, which is what makes them
            // a real test of it: a reader that hardcoded either value fails.
            const bool horned = loadScene(device, kSampleDir + "/hornedlizard.spz")->antialiased();
            const bool racoon = loadScene(device, kSampleDir + "/racoonfamily.spz")->antialiased();
            if (horned || !racoon)
            {
                std::cerr << "FAIL: expected hornedlizard antialiased=0, racoonfamily=1; got "
                          << horned << ", " << racoon << std::endl;
                result = 1;
            }
        }
        if (result == 0)
        {
            result = testControlledRoundTrip(device);
        }
        if (result == 0)
        {
            result = testRawNgspStream(device);
        }
        if (result == 0)
        {
            result = testRejections(device);
        }

        if (result == 0)
        {
            std::cout << "SpzReader test OK" << std::endl;
        }
    }
    catch (const std::exception & e)
    {
        std::cerr << "SpzReader test FAILED: " << e.what() << std::endl;
        result = 1;
    }

    device.shutdown();
    return result;
}
