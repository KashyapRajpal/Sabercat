// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Headless test for SplatReader + the loadScene() dispatch.
//
// Every fixture is synthesized AT RUNTIME -- no network, and nothing read
// from assets/ (which is gitignored and absent on a fresh clone). Each one
// carries a degeneracy guard, because the failure mode that matters here is
// a loader that "works" while silently skipping a conversion: a .splat that
// loads with un-normalized rotations, un-reordered quaternion lanes, or
// display-RGB left in the SH DC slot renders as a plausible-but-wrong image,
// never as an error. The guards below are chosen so a stubbed-out conversion
// FAILS rather than passing trivially.

#include <glm/glm.hpp>

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

constexpr size_t kRecordSize = 32;

// One .splat record, in the exporter's own terms: floats for position and
// (already-exponentiated) scale, bytes for display RGBA and the wxyz
// quaternion.
struct SplatRecord
{
    float x, y, z;
    float sx, sy, sz;
    uint8_t r, g, b, a;
    uint8_t qw, qx, qy, qz; // wxyz -- the reader must reorder these to xyzw
};
static_assert(sizeof(SplatRecord) == kRecordSize, "SplatRecord must be exactly 32 bytes");

void writeSplat(const std::string & path, const std::vector<SplatRecord> & records)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    for (const auto & record : records)
    {
        file.write(reinterpret_cast<const char *>(&record), sizeof(SplatRecord));
    }
}

void writeRawBytes(const std::string & path, const std::vector<char> & bytes)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

// The exporter's forward conversions, reimplemented independently here so the
// test compares against the format definition rather than against the
// reader's own arithmetic.
uint8_t encodeColorByte(float fDc)
{
    const float display = fDc * kShC0 + 0.5f;
    return static_cast<uint8_t>(std::lround(display * 255.0f));
}

uint8_t encodeAlphaByte(float opacity01)
{
    return static_cast<uint8_t>(std::lround(opacity01 * 255.0f));
}

uint8_t encodeQuatByte(float component)
{
    return static_cast<uint8_t>(std::lround(component * 128.0f + 128.0f));
}

float decodeQuatByte(uint8_t byte)
{
    return (static_cast<float>(byte) - 128.0f) / 128.0f;
}

// Returns true if loadScene() rejected the file, false if it accepted it.
// The rejection paths matter as much as the happy path: a headerless format
// that accepts anything divisible by 32 will happily render a stray .bin as
// noise.
bool rejects(Device & device, const std::string & path)
{
    try
    {
        loadScene(device, path);
        return false;
    }
    catch (const std::exception &)
    {
        return true;
    }
}

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

// --- Fixture 1: decode correctness, with every input guard ---
int testDecode(Device & device)
{
    std::vector<SplatRecord> records;

    // Record 0 -- the guard-carrying record.
    //
    // Rotation bytes (255, 255, 128, 128) decode to (w,x,y,z) = (0.9921875,
    // 0.9921875, 0, 0), norm ~1.4031: a reader that skips the normalize
    // fails. But that alone would not catch a missed wxyz->xyzw reorder, so
    // record 1 below handles that.
    //
    // Scales 0.5 and 2.0 sit on either side of 1.0, so a spurious exp()
    // (giving 1.6487 and 7.389) is caught in both directions -- a value of
    // 1.0 would survive exp() nearly unchanged in relative terms and prove
    // nothing.
    //
    // Colors are deliberately far from 128 (the byte that encodes f_dc = 0),
    // so a reader that passes display RGB straight through to the SH DC slot
    // produces a visibly different number than the inverted f_dc.
    {
        SplatRecord record{};
        record.x = 1.5f;
        record.y = -2.25f;
        record.z = 3.75f;
        record.sx = 0.5f;
        record.sy = 2.0f;
        record.sz = 1.25f;
        record.r = 20;  // far from 128
        record.g = 200; // far from 128
        record.b = 240; // far from 128
        record.a = 51;  // 51/255 = 0.2 exactly
        record.qw = 255;
        record.qx = 255;
        record.qy = 128;
        record.qz = 128;
        records.push_back(record);
    }

    // Record 1 -- all four quaternion lanes distinct and non-zero.
    //
    // This is the reorder guard. The classic (w=1, x=y=z=0) identity fixture
    // passes whether or not the reader reorders wxyz->xyzw, because three
    // lanes are zero and the fourth is symmetric under the swap. Four
    // distinct non-zero lanes cannot survive a missed reorder.
    {
        SplatRecord record{};
        record.x = -4.0f;
        record.y = 0.125f;
        record.z = 8.0f;
        record.sx = 3.0f;
        record.sy = 0.25f;
        record.sz = 1.0f;
        record.r = 0;
        record.g = 128;
        record.b = 255;
        record.a = 255; // opacity 1.0
        record.qw = encodeQuatByte(0.25f);
        record.qx = encodeQuatByte(-0.5f);
        record.qy = encodeQuatByte(0.75f);
        record.qz = encodeQuatByte(-0.875f);
        records.push_back(record);
    }

    const std::string path = "splat_reader_test_decode.splat";
    writeSplat(path, records);

    auto cloud = loadScene(device, path);
    CHECK(cloud->count() == records.size(), "decode: wrong Gaussian count");

    // .splat carries no bands above DC, and the viewer surfaces this so a
    // dead SH slider doesn't read as a bug.
    CHECK(cloud->shDegree() == 0, "decode: .splat must report SH degree 0");

    for (size_t i = 0; i < records.size(); ++i)
    {
        const SplatRecord & record = records[i];

        // Position passes through untouched.
        CHECK(approxEqual(cloud->positions()[i].x, record.x), "decode: position x mismatch");
        CHECK(approxEqual(cloud->positions()[i].y, record.y), "decode: position y mismatch");
        CHECK(approxEqual(cloud->positions()[i].z, record.z), "decode: position z mismatch");

        // Scale is ALREADY linear -- exp() must not be applied a second time.
        CHECK(approxEqual(cloud->scales()[i].x, record.sx), "decode: scale x must not be exp'd");
        CHECK(approxEqual(cloud->scales()[i].y, record.sy), "decode: scale y must not be exp'd");
        CHECK(approxEqual(cloud->scales()[i].z, record.sz), "decode: scale z must not be exp'd");

        // Alpha is already sigmoid-activated: a plain byte->unit rescale.
        CHECK(approxEqual(cloud->opacities()[i], record.a / 255.0f),
              "decode: opacity must be alpha/255, with no invSigmoid round trip");

        // Display RGB must be inverted back to a raw f_dc.
        const glm::vec4 & dc = cloud->shCoefficients()[i * kShCoefficientsPerGaussian];
        const float expectedR = (record.r / 255.0f - 0.5f) / kShC0;
        const float expectedG = (record.g / 255.0f - 0.5f) / kShC0;
        const float expectedB = (record.b / 255.0f - 0.5f) / kShC0;
        CHECK(approxEqual(dc.x, expectedR), "decode: SH DC r not inverted from display RGB");
        CHECK(approxEqual(dc.y, expectedG), "decode: SH DC g not inverted from display RGB");
        CHECK(approxEqual(dc.z, expectedB), "decode: SH DC b not inverted from display RGB");

        // "It decoded" vs "it merely loaded": coefficient 0 must NOT still be
        // the display value. Checked only where display and f_dc actually
        // differ -- byte 128 encodes f_dc ~= 0, where the two legitimately
        // nearly coincide and the assert would be meaningless.
        if (std::abs(record.r / 255.0f - 0.5f) > 0.01f)
        {
            CHECK(!approxEqual(dc.x, record.r / 255.0f),
                  "decode: SH DC r looks like raw display RGB (conversion skipped)");
        }

        // Bands 1-3 are absent from the format and must be EXACTLY zero --
        // not merely small. A reader that scattered leftover bytes into them
        // would fail here.
        for (int c = 1; c < kShCoefficientsPerGaussian; ++c)
        {
            const glm::vec4 & higher = cloud->shCoefficients()[i * kShCoefficientsPerGaussian + c];
            CHECK(higher.x == 0.0f && higher.y == 0.0f && higher.z == 0.0f,
                  "decode: SH coefficients 1..15 must be exactly zero for .splat");
        }

        // Rotation: independently decode, reorder, and normalize here, then
        // compare against what the reader produced.
        const float qw = decodeQuatByte(record.qw);
        const float qx = decodeQuatByte(record.qx);
        const float qy = decodeQuatByte(record.qy);
        const float qz = decodeQuatByte(record.qz);
        const float invLength = 1.0f / std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);

        const glm::vec4 & rotation = cloud->rotations()[i];
        CHECK(approxEqual(rotation.x, qx * invLength), "decode: rotation x (xyzw lane 0) mismatch");
        CHECK(approxEqual(rotation.y, qy * invLength), "decode: rotation y (xyzw lane 1) mismatch");
        CHECK(approxEqual(rotation.z, qz * invLength), "decode: rotation z (xyzw lane 2) mismatch");
        CHECK(approxEqual(rotation.w, qw * invLength), "decode: rotation w (xyzw lane 3) mismatch");

        // The stored quaternion must be unit length regardless of what the
        // bytes decoded to.
        const float storedLength = std::sqrt(rotation.x * rotation.x + rotation.y * rotation.y
                                             + rotation.z * rotation.z + rotation.w * rotation.w);
        CHECK(approxEqual(storedLength, 1.0f), "decode: stored rotation is not unit length");
    }

    // Degeneracy guard on the guard itself: record 0's raw bytes must
    // actually decode to a NON-unit quaternion, or the normalize check above
    // would pass trivially.
    {
        const SplatRecord & record = records[0];
        const float qw = decodeQuatByte(record.qw);
        const float qx = decodeQuatByte(record.qx);
        const float qy = decodeQuatByte(record.qy);
        const float qz = decodeQuatByte(record.qz);
        const float rawLength = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
        CHECK(std::abs(rawLength - 1.0f) > 0.3f,
              "decode: fixture is degenerate -- raw rotation is already ~unit length, so the "
              "normalize check proves nothing");
    }

    // Degeneracy guard: record 1's four lanes must be pairwise distinct and
    // non-zero, or the wxyz->xyzw reorder check proves nothing.
    {
        const glm::vec4 & rotation = cloud->rotations()[1];
        const float lanes[4] = {rotation.x, rotation.y, rotation.z, rotation.w};
        for (int a = 0; a < 4; ++a)
        {
            CHECK(std::abs(lanes[a]) > 1e-3f,
                  "decode: fixture is degenerate -- a quaternion lane is zero, so a missed "
                  "wxyz->xyzw reorder could pass unnoticed");
            for (int b = a + 1; b < 4; ++b)
            {
                CHECK(std::abs(lanes[a] - lanes[b]) > 1e-3f,
                      "decode: fixture is degenerate -- two quaternion lanes are equal, so a "
                      "missed wxyz->xyzw reorder could pass unnoticed");
            }
        }
    }

    std::cout << "splat decode OK: 2 records, rotations normalized + reordered to xyzw, scale "
                 "not re-exp'd, DC inverted from display RGB, bands 1-15 exactly zero"
              << std::endl;
    return 0;
}

// --- Fixture 2: the rejection paths ---
int testRejections(Device & device)
{
    // Empty file.
    const std::string emptyPath = "splat_reader_test_empty.splat";
    writeRawBytes(emptyPath, {});
    CHECK(rejects(device, emptyPath), "rejection: an empty .splat must be rejected");

    // Size not a multiple of 32.
    const std::string raggedPath = "splat_reader_test_ragged.splat";
    writeRawBytes(raggedPath, std::vector<char>(kRecordSize + 7, 0));
    CHECK(rejects(device, raggedPath),
          "rejection: a .splat whose size is not a multiple of 32 must be rejected");

    // The real hazard: a wrong file that happens to divide by 32 evenly. A
    // headerless format has no magic bytes to check, so only a sanity guard
    // on the decoded values can catch this. All-0xFF decodes to NaN floats.
    const std::string garbagePath = "splat_reader_test_garbage.splat";
    writeRawBytes(garbagePath, std::vector<char>(kRecordSize * 8, static_cast<char>(0xFF)));
    CHECK(rejects(device, garbagePath),
          "rejection: a non-.splat file that divides by 32 must fail the decoded-value sanity "
          "guard, not render as noise");

    // A zeroed file also divides by 32 evenly, and decodes to scale 0 --
    // degenerate Gaussians with no extent.
    const std::string zeroedPath = "splat_reader_test_zeroed.splat";
    writeRawBytes(zeroedPath, std::vector<char>(kRecordSize * 8, 0));
    CHECK(rejects(device, zeroedPath),
          "rejection: an all-zero file decodes to zero scales and must be rejected");

    std::cout << "splat rejections OK: empty, non-multiple-of-32, all-0xFF, and all-zero files "
                 "all rejected"
              << std::endl;
    return 0;
}

// --- Fixture 3: cross-format equivalence, asset-free ---
//
// Synthesizes a .ply AND the .splat a correct exporter would produce from
// that same scene, loads both through loadScene(), and asserts they agree.
// This is what catches a conversion that is self-consistently wrong: the
// per-field checks in testDecode() verify SplatReader against the format
// spec, but only comparing against the already-trusted PLY path proves both
// loaders land on the SAME canonical contract.
//
// Tolerances are one quantization step of the .splat encoding, not float
// epsilon -- .splat stores color, opacity, and rotation as bytes, so exact
// agreement is impossible by construction.

// SH degree 0 .ply: f_dc only, no f_rest. Matches what a .splat can express,
// so the comparison is apples-to-apples rather than "the .ply has more data".
struct PlyVertex
{
    float x, y, z;
    float nx = 0.0f, ny = 0.0f, nz = 0.0f;
    float f_dc[3];
    float opacity;  // raw, pre-sigmoid
    float scale[3]; // raw, pre-exp (log-space)
    float rot[4];   // unnormalized, wxyz
};
static_assert(sizeof(PlyVertex) == (6 + 3 + 1 + 3 + 4) * sizeof(float),
              "PlyVertex must be tightly packed with no padding");

void writePly(const std::string & path, const std::vector<PlyVertex> & verts)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file << "ply\n";
    file << "format binary_little_endian 1.0\n";
    file << "element vertex " << verts.size() << "\n";
    for (const char * name : {"x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
                              "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1",
                              "rot_2", "rot_3"})
    {
        file << "property float " << name << "\n";
    }
    file << "end_header\n";
    for (const auto & v : verts)
    {
        file.write(reinterpret_cast<const char *>(&v), sizeof(PlyVertex));
    }
}

// Exactly what a correct .splat exporter does to one PLY vertex: activate
// scale and opacity, normalize the quaternion, then quantize.
SplatRecord exportToSplat(const PlyVertex & v)
{
    SplatRecord record{};
    record.x = v.x;
    record.y = v.y;
    record.z = v.z;

    record.sx = std::exp(v.scale[0]);
    record.sy = std::exp(v.scale[1]);
    record.sz = std::exp(v.scale[2]);

    record.r = encodeColorByte(v.f_dc[0]);
    record.g = encodeColorByte(v.f_dc[1]);
    record.b = encodeColorByte(v.f_dc[2]);
    record.a = encodeAlphaByte(1.0f / (1.0f + std::exp(-v.opacity)));

    const float invLength = 1.0f / std::sqrt(v.rot[0] * v.rot[0] + v.rot[1] * v.rot[1]
                                             + v.rot[2] * v.rot[2] + v.rot[3] * v.rot[3]);
    record.qw = encodeQuatByte(v.rot[0] * invLength);
    record.qx = encodeQuatByte(v.rot[1] * invLength);
    record.qy = encodeQuatByte(v.rot[2] * invLength);
    record.qz = encodeQuatByte(v.rot[3] * invLength);
    return record;
}

int testCrossFormat(Device & device)
{
    std::vector<PlyVertex> verts;

    // Deliberately varied: asymmetric scales, an off-axis unnormalized
    // quaternion with four distinct lanes, and f_dc values on both sides of
    // zero so the DC inversion is exercised in both directions.
    {
        PlyVertex v{};
        v.x = 2.5f;
        v.y = -1.75f;
        v.z = 0.5f;
        v.f_dc[0] = 0.8f;
        v.f_dc[1] = -0.6f;
        v.f_dc[2] = 0.2f;
        v.opacity = 1.2f; // sigmoid -> ~0.769
        v.scale[0] = std::log(0.4f);
        v.scale[1] = std::log(1.6f);
        v.scale[2] = std::log(0.9f);
        v.rot[0] = 0.6f;
        v.rot[1] = -0.4f;
        v.rot[2] = 0.5f;
        v.rot[3] = -0.3f; // unnormalized on purpose
        verts.push_back(v);
    }
    {
        PlyVertex v{};
        v.x = -7.0f;
        v.y = 3.25f;
        v.z = 11.5f;
        v.f_dc[0] = -1.2f;
        v.f_dc[1] = 0.05f;
        v.f_dc[2] = 1.1f;
        v.opacity = -0.7f; // sigmoid -> ~0.332
        v.scale[0] = std::log(2.2f);
        v.scale[1] = std::log(0.15f);
        v.scale[2] = std::log(1.0f);
        v.rot[0] = -0.2f;
        v.rot[1] = 0.7f;
        v.rot[2] = -0.55f;
        v.rot[3] = 0.35f;
        verts.push_back(v);
    }

    std::vector<SplatRecord> records;
    records.reserve(verts.size());
    for (const auto & v : verts)
    {
        records.push_back(exportToSplat(v));
    }

    const std::string plyPath = "splat_reader_test_cross.ply";
    const std::string splatPath = "splat_reader_test_cross.splat";
    writePly(plyPath, verts);
    writeSplat(splatPath, records);

    auto fromPly = loadScene(device, plyPath);
    auto fromSplat = loadScene(device, splatPath);

    CHECK(fromPly->count() == fromSplat->count(), "cross-format: Gaussian counts differ");

    // A degree-0 .ply and a .splat both carry DC only -- if these disagree,
    // one of the two loaders is misreporting what it read.
    CHECK(fromPly->shDegree() == 0, "cross-format: degree-0 .ply must report SH degree 0");
    CHECK(fromSplat->shDegree() == 0, "cross-format: .splat must report SH degree 0");

    // One quantization step of each encoding.
    constexpr float kOpacityTolerance = 1.0f / 255.0f;      // alpha is a byte
    constexpr float kDcTolerance = (1.0f / 255.0f) / kShC0; // color is a byte, then scaled by 1/C0
    constexpr float kRotationTolerance = 2.0f / 128.0f;     // byte step, plus renormalization

    for (size_t i = 0; i < verts.size(); ++i)
    {
        CHECK(approxEqual(fromPly->positions()[i], fromSplat->positions()[i]),
              "cross-format: positions disagree");

        // Both must land on a LINEAR scale: the .ply loader exp()s, the
        // .splat loader passes through. A spurious exp() in either one shows
        // up here as a large mismatch.
        CHECK(approxEqualRelative(fromPly->scales()[i].x, fromSplat->scales()[i].x),
              "cross-format: scale x disagrees -- one loader applied exp() and the other did not");
        CHECK(approxEqualRelative(fromPly->scales()[i].y, fromSplat->scales()[i].y),
              "cross-format: scale y disagrees -- one loader applied exp() and the other did not");
        CHECK(approxEqualRelative(fromPly->scales()[i].z, fromSplat->scales()[i].z),
              "cross-format: scale z disagrees -- one loader applied exp() and the other did not");

        CHECK(approxEqual(fromPly->opacities()[i], fromSplat->opacities()[i], kOpacityTolerance),
              "cross-format: opacity disagrees -- both must be sigmoid-activated [0,1]");

        // Both must store a RAW f_dc in coefficient 0, not display RGB.
        const glm::vec4 & plyDc = fromPly->shCoefficients()[i * kShCoefficientsPerGaussian];
        const glm::vec4 & splatDc = fromSplat->shCoefficients()[i * kShCoefficientsPerGaussian];
        CHECK(approxEqual(plyDc.x, splatDc.x, kDcTolerance), "cross-format: SH DC r disagrees");
        CHECK(approxEqual(plyDc.y, splatDc.y, kDcTolerance), "cross-format: SH DC g disagrees");
        CHECK(approxEqual(plyDc.z, splatDc.z, kDcTolerance), "cross-format: SH DC b disagrees");

        // Both must be normalized and in xyzw order. A missed reorder in
        // either loader lands the lanes in different places and fails here.
        CHECK(approxEqual(fromPly->rotations()[i], fromSplat->rotations()[i], kRotationTolerance),
              "cross-format: rotations disagree -- check the wxyz->xyzw reorder and normalize");
    }

    // Degeneracy guard: the fixture must actually exercise the conversions.
    // If every DC happened to be ~0 or every scale ~1, the comparison above
    // would pass even with both loaders stubbed to constants.
    {
        const glm::vec4 & dc = fromSplat->shCoefficients()[0];
        CHECK(std::abs(dc.x) > 0.1f && std::abs(dc.y) > 0.1f,
              "cross-format: fixture is degenerate -- SH DC is ~0, so the inversion proves "
              "nothing");
        CHECK(std::abs(fromSplat->scales()[0].x - 1.0f) > 0.1f,
              "cross-format: fixture is degenerate -- scale is ~1, where a spurious exp() would "
              "barely move it");
        CHECK(std::abs(fromSplat->opacities()[0] - fromSplat->opacities()[1]) > 0.1f,
              "cross-format: fixture is degenerate -- both opacities are the same");
    }

    std::cout << "cross-format equivalence OK: a .ply and the .splat a correct exporter would "
                 "produce from it load to the same canonical values (within one quantization "
                 "step)"
              << std::endl;
    return 0;
}

} // namespace

int main()
{
    // Scratch files go to a temp folder, not the directory the test runs from.
    examples::ScratchDir scratch("splat_reader_test");
    Device device;
    int result = 0;
    try
    {
        // Headless: no surface, no shaders -- these loaders never touch the
        // render path.
        device.initialize({}, /*enableValidation=*/true);

        result = testDecode(device);
        if (result == 0)
        {
            result = testRejections(device);
        }
        if (result == 0)
        {
            result = testCrossFormat(device);
        }

        if (result == 0)
        {
            std::cout << "SplatReader test OK" << std::endl;
        }
    }
    catch (const std::exception & e)
    {
        std::cerr << "SplatReader test FAILED: " << e.what() << std::endl;
        result = 1;
    }

    device.shutdown();
    return result;
}
