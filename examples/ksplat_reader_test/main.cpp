// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Headless test for KsplatReader + the loadScene() dispatch.
//
// .ksplat has no written spec -- it is whatever mkkellogg's GaussianSplats3D
// writes and reads -- so the fixtures are that project's OWN output:
// fixtures/synthetic_l{0,1,2}.ksplat were produced by its SplatBufferGenerator
// (the code path of its util/create-ksplat.js) from the synthetic scene in
// ../common/compressed_scene_fixture.h, one per compression level. A reader
// that only agreed with a Sabercat-written encoder could not pass this.
//
// The check is "does each file decode back to the .ply it was made from":
// the source is regenerated here, loaded through the already-verified
// PLYReader, matched Gaussian-by-Gaussian by position (the generator reorders
// into spatial buckets), and every attribute is held to the bound its
// encoding implies -- derived below, not tuned. Then the decoded scene and
// the source are rendered through sabercat::Renderer and diffed, and a set of
// byte-level corruptions of the real files must each be rejected by name.
//
// Modes:
//   sabercat_ksplat_reader_test                          the test
//   sabercat_ksplat_reader_test --write-source-ply <p>   the fixtures' source (see generate_fixtures.mjs)
//   sabercat_ksplat_reader_test --compare <src.ply> <scene> [ordered]
//       measurement only: per-attribute errors + render diff of any scene vs
//       its source (`ordered` when <src.ply> is a same-order reference decode)

#include <glm/glm.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "../common/compressed_scene_fixture.h"
#include "../common/scratch_dir.h"
#include "sabercat/device.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/render_config.h"
#include "sabercat/renderer.h"
#include "sabercat/scene_loader.h"

using namespace sabercat;

namespace
{

const std::string kFixtureDir = SABERCAT_KSPLAT_FIXTURE_DIR;
const std::string kSourcePly = "ksplat_reader_test_source.ply";
constexpr size_t kSceneSize = 300;

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

std::vector<uint8_t> readBytes(const std::string & path)
{
    std::ifstream file(path, std::ios::binary);
    return std::vector<uint8_t>(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

void writeBytes(const std::string & path, const std::vector<uint8_t> & bytes)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

template <typename T> T peek(const std::vector<uint8_t> & bytes, size_t offset)
{
    T value;
    std::memcpy(&value, bytes.data() + offset, sizeof(T));
    return value;
}

template <typename T> void poke(std::vector<uint8_t> & bytes, size_t offset, T value)
{
    std::memcpy(bytes.data() + offset, &value, sizeof(T));
}

// Raw header fields the bounds are derived from (offsets: ksplat_reader.cpp).
constexpr size_t kSection0 = 4096;

// --- Decoded fixture vs. its source ---
//
// Bounds, from what each level stores (GaussianSplats3D SplatBuffer):
//   position  level 0: f32 copy -> exact. Levels 1/2: Math.round of
//             delta * range / (block/2) into u16 -> half a step, step =
//             (block/2) / range.
//   scale     linear. Level 0 f32 (the JS exp()s in double, PLYReader in
//             float: 1e-6 relative). Levels 1/2: three.js toHalfFloat
//             TRUNCATES -> under one f16 ulp, 2^-10 relative.
//   rotation  level 0 f32 of a double-normalized quaternion; levels 1/2 four
//             truncated f16 lanes -> angle <= 2 * |dq| <= 2 * 2 * 2^-10.
//   color    floor((0.5 + SH_C0 * f_dc) * 255) read back as b/255: the
//             decoded value is never above the source and at most one byte
//             step below it -> 1/255 display = (1/255)/SH_C0 in f_dc.
//   opacity   floor(sigmoid * 255) -> one-sided, under 1/255.
//   SH 1-2    level 0 f32 copy; level 1 truncated f16 (|v| <= 0.45 ->
//             0.45 * 2^-10); level 2 floor((v - min) / (max - min) * 255) ->
//             one-sided, under one step (max - min) / 255.
//   SH 3      the format stops at degree 2: decoded band 3 must be exactly 0.
int testFixture(Device & device, Renderer & renderer, uint32_t level)
{
    const std::string path = kFixtureDir + "/synthetic_l" + std::to_string(level) + ".ksplat";
    const std::string name = "level " + std::to_string(level);
    const std::vector<uint8_t> bytes = readBytes(path);
    CHECK(!bytes.empty(), "fixture missing: " + path);
    CHECK(detectFormat(path) == SceneFormat::KSPLAT, name + ": .ksplat must dispatch to KsplatReader");

    auto source = loadScene(device, kSourcePly);
    auto cloud = loadScene(device, path);
    CHECK(cloud->count() == kSceneSize, name + ": wrong Gaussian count (alpha threshold dropped some?)");
    CHECK(cloud->shDegree() == 2, name + ": SH degree must be 2 (the format's ceiling; the source has 3)");
    CHECK(!cloud->antialiased(), name + ": .ksplat cannot declare antialiasing");
    CHECK(peek<uint16_t>(bytes, 20) == level, name + ": fixture is not the level its name says");

    // Bounds that depend on the file's own header.
    const float blockSize = peek<float>(bytes, kSection0 + 16);
    uint32_t range = peek<uint32_t>(bytes, kSection0 + 24);
    range = range ? range : 32767;
    const double positionBound = level == 0 ? 1e-6 : 0.5 * (blockSize / 2.0) / range + 1e-6;
    const double scaleBound = level == 0 ? 1e-6 : std::ldexp(1.0, -10) + 1e-7;
    const double rotationBound = level == 0 ? 1e-5 : 4.0 * std::ldexp(1.0, -10) + 1e-5;
    const double colorStep = (1.0 / 255.0) / kShC0;
    const double opacityStep = 1.0 / 255.0;
    const double shStep = (static_cast<double>(peek<float>(bytes, 40)) - peek<float>(bytes, 36)) / 255.0;
    const double shBound = level == 0 ? 1e-7 : (level == 1 ? 0.45 * std::ldexp(1.0, -10) + 1e-7 : shStep + 1e-6);

    // One-sided checks: floor() quantizers never round UP.
    size_t roundedUp = 0;
    const auto oneSided = [&](size_t d, size_t s)
    {
        if (cloud->opacities()[d] > source->opacities()[s] + 1e-6f)
        {
            ++roundedUp;
        }
        for (int c = 0; c < 3; ++c)
        {
            if (cloud->shCoefficients()[d * kShCoefficientsPerGaussian][c]
                > source->shCoefficients()[s * kShCoefficientsPerGaussian][c] + 1e-5f)
            {
                ++roundedUp;
            }
        }
    };
    const fixture::AttributeErrors e = fixture::compareToSource(*cloud, *source, false, 0.01f, oneSided);
    std::printf("  %s (%zu bytes):\n", name.c_str(), bytes.size());
    fixture::printStat("position (abs)", e.position, positionBound);
    fixture::printStat("scale (relative)", e.scaleRelative, scaleBound);
    fixture::printStat("rotation (radians)", e.rotationAngle, rotationBound);
    fixture::printStat("opacity (abs, one-sided)", e.opacity, opacityStep);
    fixture::printStat("SH DC (abs, one-sided)", e.dc, colorStep);
    fixture::printStat("SH band 1 (abs)", e.sh[0], shBound);
    fixture::printStat("SH band 2 (abs)", e.sh[1], shBound);
    fixture::printStat("SH band 3 (|decoded|, must be 0)", e.shAboveDegree, 0.0);

    CHECK(e.matched == kSceneSize && e.unmatched == 0 && e.sourceUsedTwice == 0,
          name + ": decoded Gaussians do not map one-to-one onto the source");
    CHECK(e.position.max <= positionBound, name + ": position error above the format's quantization step");
    CHECK(e.scaleRelative.max <= scaleBound,
          name + ": scale error -- stored LINEAR (exp() applied by the exporter), so never exp() again");
    CHECK(e.rotationAngle.max <= rotationBound,
          name + ": rotation error -- lanes are stored w,x,y,z and must be reordered to xyzw");
    CHECK(e.opacity.max <= opacityStep + 1e-6, name + ": opacity is not byte/255");
    CHECK(e.dc.max <= colorStep + 1e-5, name + ": SH DC is not the inverse of the display-byte color");
    CHECK(roundedUp == 0, name + ": a floor()-quantized value decoded ABOVE its source");
    CHECK(e.sh[0].max <= shBound && e.sh[1].max <= shBound,
          name + ": SH bands 1-2 wrong -- per band, channel-major inside the band");
    CHECK(e.shAboveDegree.max == 0.0, name + ": SH band 3 must be exactly zero (the format has none)");

    // --- Degeneracy guards: each conversion above must be one a mistake would fail ---
    size_t laneWitnesses = 0;  // wxyz read as xyzw would rotate far past the bound
    size_t scaleWitnesses = 0; // a spurious exp() moves the scale by > 10%
    size_t dcWitnesses = 0;    // passing the display byte through as f_dc is visibly wrong
    size_t shWitnesses = 0;    // a coefficient-major misread of a band differs
    for (size_t s = 0; s < source->count(); ++s)
    {
        const glm::vec4 q = source->rotations()[s];
        if (fixture::rotationAngle(q, glm::vec4(q.w, q.x, q.y, q.z)) > 100.0 * rotationBound)
        {
            ++laneWitnesses;
        }
        const float sx = source->scales()[s].x;
        if (std::abs(std::exp(sx) / sx - 1.0f) > 0.1f)
        {
            ++scaleWitnesses;
        }
        const glm::vec4 dc = source->shCoefficients()[s * kShCoefficientsPerGaussian];
        if (std::abs((dc.x * kShC0 + 0.5f) - dc.x) > 0.05f)
        {
            ++dcWitnesses;
        }
        const glm::vec4 c1 = source->shCoefficients()[s * kShCoefficientsPerGaussian + 1];
        const glm::vec4 c2 = source->shCoefficients()[s * kShCoefficientsPerGaussian + 2];
        if (std::abs(c1.y - c2.x) > 10.0 * shBound + 0.01)
        {
            ++shWitnesses;
        }
    }
    CHECK(laneWitnesses > kSceneSize * 9 / 10, name + ": degenerate -- rotations would survive a lane mix-up");
    CHECK(scaleWitnesses == kSceneSize, name + ": degenerate -- scales where exp() would not show");
    CHECK(dcWitnesses > kSceneSize * 9 / 10, name + ": degenerate -- DC values near the display identity");
    CHECK(shWitnesses > kSceneSize * 3 / 4, name + ": degenerate -- SH layout misread would not show");
    if (level >= 1)
    {
        // Both bucket kinds must be exercised: full (index by division) and
        // partially-filled (index through the lengths table).
        const uint32_t full = peek<uint32_t>(bytes, kSection0 + 32);
        const uint32_t partial = peek<uint32_t>(bytes, kSection0 + 36);
        std::printf("    buckets: %u full (256 splats), %u partially filled\n", full, partial);
        CHECK(full >= 1 && partial >= 2, name + ": degenerate -- fixture lacks full or partial buckets");
    }

    // --- Rendered through the real renderer, at the file's own SH degree ---
    const fixture::ImageDiff diff = fixture::renderAndDiff(renderer, kSourcePly, path, 2);
    fixture::printImageDiff("render vs source .ply (SH<=2)", diff);
    CHECK(diff.meanCoverage > 0.02, name + ": degenerate render -- the reference image is nearly empty");
    CHECK(diff.rmse < 0.01, name + ": rendered image differs from the source scene's beyond quantization");
    return 0;
}

// --- Corrupted real files must be rejected, each with its own reason ---
int testRejections(Device & device)
{
    const std::vector<uint8_t> l0 = readBytes(kFixtureDir + "/synthetic_l0.ksplat");
    const std::vector<uint8_t> l1 = readBytes(kFixtureDir + "/synthetic_l1.ksplat");
    CHECK(!l0.empty() && !l1.empty(), "rejection fixtures missing");

    struct Case
    {
        const char * what;
        std::vector<uint8_t> bytes;
        const char * expect; // substring of the error
    };
    std::vector<Case> cases;
    const auto mutated = [](std::vector<uint8_t> bytes, const std::function<void(std::vector<uint8_t> &)> & f)
    {
        f(bytes);
        return bytes;
    };

    cases.push_back({"empty file", {}, "empty"});
    cases.push_back({"shorter than the header", std::vector<uint8_t>(100, 0), "smaller than the 4096-byte"});
    cases.push_back({"future major version", mutated(l1, [](auto & b) { b[0] = 1; }), "unsupported .ksplat version"});
    cases.push_back({"legacy version 0.0", mutated(l1, [](auto & b) { b[1] = 0; }), "unsupported .ksplat version"});
    cases.push_back({"unknown compression level",
                     mutated(l1, [](auto & b) { poke<uint16_t>(b, 20, 3); }), "unknown compression level"});
    cases.push_back({"truncated records", mutated(l1, [](auto & b) { b.resize(b.size() - 100); }), "truncated"});
    cases.push_back({"section count past the file",
                     mutated(l1, [](auto & b) { poke<uint32_t>(b, 4, 100000); }), "more than the file can hold"});
    cases.push_back({"SH degree 3 section",
                     mutated(l1, [](auto & b) { poke<uint16_t>(b, kSection0 + 40, 3); }), "only up to degree 2"});
    // The reader accepts a total matching EITHER header count (offset 12 is
    // GaussianSplats3D's maxSplatCount, 16 is splat-transform's splatCount), so
    // a real disagreement has to break both.
    cases.push_back({"header/section count disagree",
                     mutated(l1,
                             [](auto & b)
                             {
                                 poke<uint32_t>(b, 12, peek<uint32_t>(b, 12) + 1);
                                 poke<uint32_t>(b, 16, peek<uint32_t>(b, 16) + 1);
                             }),
                     "the header says"});
    cases.push_back({"bucket counts disagree",
                     mutated(l1, [](auto & b) { poke<uint32_t>(b, kSection0 + 32, peek<uint32_t>(b, kSection0 + 32) + 1); }),
                     "bucket count"});
    cases.push_back({"bucket lengths do not cover the splats",
                     mutated(l1, [](auto & b) { poke<uint32_t>(b, kSection0 + 8, 255); }), "buckets cover"});
    cases.push_back({"bucket record size",
                     mutated(l1, [](auto & b) { poke<uint16_t>(b, kSection0 + 20, 8); }), "bucket record size"});
    // A NaN scale inside a well-formed container: level 0 stores f32 scales at
    // record offset 12, records after 4096 + 1024 (no buckets at level 0).
    cases.push_back({"non-finite scale",
                     mutated(l0, [](auto & b) { poke<float>(b, 4096 + 1024 + 5 * 140 + 12, std::nanf("")); }),
                     "non-finite scale"});

    for (size_t i = 0; i < cases.size(); ++i)
    {
        const std::string path = "ksplat_reader_test_bad_" + std::to_string(i) + ".ksplat";
        writeBytes(path, cases[i].bytes);
        std::string error;
        try
        {
            loadScene(device, path);
        }
        catch (const std::exception & ex)
        {
            error = ex.what();
        }
        CHECK(!error.empty(), std::string("rejection: accepted a corrupt file -- ") + cases[i].what);
        CHECK(error.find(cases[i].expect) != std::string::npos,
              std::string("rejection: ") + cases[i].what + " rejected for the wrong reason: " + error);
        std::printf("  rejected %-40s %.90s\n", cases[i].what, error.c_str());
    }
    std::string missing;
    try
    {
        loadScene(device, "ksplat_reader_test_does_not_exist.ksplat");
    }
    catch (const std::exception & ex)
    {
        missing = ex.what();
    }
    CHECK(missing.find("failed to open") != std::string::npos, "rejection: missing file");
    CHECK(detectFormat("Scene.KSPLAT") == SceneFormat::KSPLAT, "dispatch: extension must be case-insensitive");
    return 0;
}

int compareMode(const std::string & sourcePath, const std::string & scenePath, bool ordered)
{
    Device device;
    device.initialize({}, /*enableValidation=*/true);
    {
        auto source = loadScene(device, sourcePath);
        auto scene = loadScene(device, scenePath);
        std::printf("%s (%zu Gaussians, SH %u) vs %s (%zu, SH %u)%s\n", scenePath.c_str(), scene->count(),
                    scene->shDegree(), sourcePath.c_str(), source->count(), source->shDegree(),
                    ordered ? " [same order]" : " [matched by position]");
        fixture::printAttributeErrors(fixture::compareToSource(*scene, *source, ordered, 0.02f));
        const uint32_t degree = std::min(scene->shDegree(), source->shDegree());
        source.reset();
        scene.reset();
        Renderer renderer(640, 480);
        if (!renderer.initialize({}, true, SHADER_DIR))
        {
            device.shutdown();
            return 1;
        }
        fixture::printImageDiff(("render (SH<=" + std::to_string(degree) + ")").c_str(),
                                fixture::renderAndDiff(renderer, sourcePath, scenePath, degree));
        renderer.shutdown();
    }
    device.shutdown();
    return 0;
}

} // namespace

// A second tool's .ksplat: playcanvas/splat-transform's own minimal.ksplat
// (MIT) next to the .splat it was written from. Its header carries
// maxSplatCount = sectionCount = 0 (the real count is at offset 16) and a
// level-0 section with a real 12-byte bucket table, both of which
// GaussianSplats3D's files never exercise. Decoded positions, scales and
// opacities must equal the .splat's within the format's own quantization.
int testSplatTransformFile(Device & device)
{
    const std::string ksplat = kFixtureDir + "/splat_transform_minimal.ksplat";
    const std::string splat = kFixtureDir + "/splat_transform_minimal.splat";
    auto fromKsplat = loadScene(device, ksplat);
    auto fromSplat = loadScene(device, splat);
    CHECK(fromKsplat->count() == fromSplat->count() && fromKsplat->count() == 4,
          "splat-transform .ksplat: expected 4 Gaussians, got " + std::to_string(fromKsplat->count()));
    float position = 0.0f, scale = 0.0f, opacity = 0.0f;
    for (size_t i = 0; i < fromKsplat->count(); ++i)
    {
        position = std::max(position, glm::length(glm::vec3(fromKsplat->positions()[i]) -
                                                  glm::vec3(fromSplat->positions()[i])));
        scale = std::max(scale, glm::length(glm::vec3(fromKsplat->scales()[i]) -
                                            glm::vec3(fromSplat->scales()[i])) /
                                    glm::length(glm::vec3(fromSplat->scales()[i])));
        opacity = std::max(opacity, std::fabs(fromKsplat->opacities()[i] - fromSplat->opacities()[i]));
    }
    std::cout << "  splat-transform minimal.ksplat vs minimal.splat: worst position " << position
              << ", relative scale " << scale << ", opacity " << opacity << std::endl;
    CHECK(position <= 1e-3f && scale <= 1e-2f && opacity <= 1.0f / 255.0f + 1e-6f,
          "splat-transform .ksplat decodes differently from the .splat it was written from");
    return 0;
}

int main(int argc, char * argv[])
{
    if (argc == 3 && std::string(argv[1]) == "--write-source-ply")
    {
        fixture::writePly(argv[2], fixture::makeSyntheticScene());
        return 0;
    }
    if (argc >= 4 && std::string(argv[1]) == "--compare")
    {
        try
        {
            return compareMode(argv[2], argv[3], argc >= 5 && std::string(argv[4]) == "ordered");
        }
        catch (const std::exception & e)
        {
            std::cerr << "compare failed: " << e.what() << std::endl;
            return 1;
        }
    }

    // Scratch files go to a temp folder, not the directory the test runs from.
    examples::ScratchDir scratch("ksplat_reader_test");
    fixture::writePly(kSourcePly, fixture::makeSyntheticScene());

    Device device;
    Renderer renderer(320, 240);
    int result = 0;
    try
    {
        device.initialize({}, /*enableValidation=*/true);
        CHECK(renderer.initialize({}, true, SHADER_DIR), "renderer initialize");

        std::cout << "GaussianSplats3D-generated fixtures vs. their source .ply:" << std::endl;
        for (uint32_t level = 0; level <= 2 && result == 0; ++level)
        {
            result = testFixture(device, renderer, level);
        }
        if (result == 0)
        {
            std::cout << "a second tool's file:" << std::endl;
            result = testSplatTransformFile(device);
        }
        if (result == 0)
        {
            std::cout << "malformed inputs:" << std::endl;
            result = testRejections(device);
        }
        if (result == 0)
        {
            std::cout << "KsplatReader test OK" << std::endl;
        }
    }
    catch (const std::exception & e)
    {
        std::cerr << "KsplatReader test FAILED: " << e.what() << std::endl;
        result = 1;
    }
    renderer.shutdown();
    device.shutdown();
    return result;
}
