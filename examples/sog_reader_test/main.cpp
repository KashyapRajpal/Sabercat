// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Headless test for SogReader + the loadScene() dispatch.
//
// fixtures/synthetic.sog is PlayCanvas splat-transform's own output (its CLI,
// bundled .sog, CPU k-means) for the synthetic scene in
// ../common/compressed_scene_fixture.h -- a real file from the tool that
// defines the format, so a reader that only agreed with a Sabercat-written
// encoder could not pass. ../common/generate_fixtures.mjs regenerates it.
//
// What is checked:
//   1. The bundled file decodes back to the .ply it was made from (loaded via
//      the verified PLYReader, matched by position -- SOG reorders into Morton
//      order), every attribute within the bound its encoding implies. For the
//      codebook-quantized attributes the bound is exact rather than a
//      tolerance: splat-transform labels each value with its NEAREST codebook
//      entry, so the decoded value must be the codebook entry nearest the
//      source value (codebooks read here from the fixture's own meta.json).
//   2. The other physical forms decode identically, bit for bit: the same
//      entries unbundled beside a meta.json, and re-zipped with DEFLATE (what
//      a general-purpose archiver produces, and the PlayCanvas engine reads).
//   3. Rendered through sabercat::Renderer against the source .ply.
//   4. Corruptions of the real file -- container, meta.json, images -- are
//      each rejected by name.
//
// Modes: as ksplat_reader_test (--write-source-ply, --compare <src> <scene> [ordered]).

#include <nlohmann/json.hpp>
#include <zlib.h>

#include <glm/glm.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
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
using Json = nlohmann::json;
using Bytes = std::vector<uint8_t>;
using Archive = std::map<std::string, Bytes>;

namespace
{

const std::string kFixture = std::string(SABERCAT_SOG_FIXTURE_DIR) + "/synthetic.sog";
const std::string kSourcePly = "sog_reader_test_source.ply";
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

Bytes readBytes(const std::string & path)
{
    std::ifstream file(path, std::ios::binary);
    return Bytes(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
}

void writeBytes(const std::string & path, const Bytes & bytes)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

template <typename T> T peek(const Bytes & b, size_t at)
{
    T v;
    std::memcpy(&v, b.data() + at, sizeof(T));
    return v;
}

template <typename T> void put(Bytes & b, T v)
{
    const size_t at = b.size();
    b.resize(at + sizeof(T));
    std::memcpy(b.data() + at, &v, sizeof(T));
}

// --- A test-side ZIP, independent of SogReader's ---

// Reads the STORED archive splat-transform writes (sizes from the central
// directory; it streams entries with the data-descriptor flag set).
Archive unzipStored(const Bytes & zip)
{
    Archive out;
    const size_t eocd = zip.size() - 22;
    const uint16_t count = peek<uint16_t>(zip, eocd + 10);
    size_t at = peek<uint32_t>(zip, eocd + 16);
    for (uint16_t i = 0; i < count; ++i)
    {
        const uint32_t size = peek<uint32_t>(zip, at + 20);
        const uint16_t nameLength = peek<uint16_t>(zip, at + 28);
        const uint32_t local = peek<uint32_t>(zip, at + 42);
        const std::string name(reinterpret_cast<const char *>(&zip[at + 46]), nameLength);
        const size_t data = local + 30 + peek<uint16_t>(zip, local + 26) + peek<uint16_t>(zip, local + 28);
        out[name] = Bytes(zip.begin() + static_cast<std::ptrdiff_t>(data),
                          zip.begin() + static_cast<std::ptrdiff_t>(data + size));
        at += 46 + nameLength + peek<uint16_t>(zip, at + 30) + peek<uint16_t>(zip, at + 32);
    }
    return out;
}

Bytes deflateRaw(const Bytes & in)
{
    z_stream stream{};
    deflateInit2(&stream, Z_BEST_COMPRESSION, Z_DEFLATED, -MAX_WBITS, 8, Z_DEFAULT_STRATEGY);
    Bytes out(deflateBound(&stream, static_cast<uLong>(in.size())));
    stream.next_in = const_cast<Bytef *>(in.data());
    stream.avail_in = static_cast<uInt>(in.size());
    stream.next_out = out.data();
    stream.avail_out = static_cast<uInt>(out.size());
    deflate(&stream, Z_FINISH);
    out.resize(stream.total_out);
    deflateEnd(&stream);
    return out;
}

// A minimal, standard ZIP: local headers, central directory, end record.
Bytes zip(const Archive & entries, uint16_t method)
{
    Bytes out;
    Bytes central;
    for (const auto & [name, data] : entries)
    {
        const Bytes payload = method == 8 ? deflateRaw(data) : data;
        const uint32_t crc = static_cast<uint32_t>(crc32(0L, data.data(), static_cast<uInt>(data.size())));
        const uint32_t local = static_cast<uint32_t>(out.size());
        for (Bytes * b : {&out, &central})
        {
            const bool isCentral = b == &central;
            put<uint32_t>(*b, isCentral ? 0x02014b50u : 0x04034b50u);
            if (isCentral)
            {
                put<uint16_t>(*b, 20); // version made by
            }
            put<uint16_t>(*b, 20); // version needed
            put<uint16_t>(*b, 0);  // flags
            put<uint16_t>(*b, method);
            put<uint32_t>(*b, 0); // DOS time + date
            put<uint32_t>(*b, crc);
            put<uint32_t>(*b, static_cast<uint32_t>(payload.size()));
            put<uint32_t>(*b, static_cast<uint32_t>(data.size()));
            put<uint16_t>(*b, static_cast<uint16_t>(name.size()));
            put<uint16_t>(*b, 0); // extra
            if (isCentral)
            {
                put<uint16_t>(*b, 0); // comment
                put<uint16_t>(*b, 0); // disk
                put<uint16_t>(*b, 0); // internal attributes
                put<uint32_t>(*b, 0); // external attributes
                put<uint32_t>(*b, local);
            }
            b->insert(b->end(), name.begin(), name.end());
        }
        out.insert(out.end(), payload.begin(), payload.end());
    }
    const uint32_t centralOffset = static_cast<uint32_t>(out.size());
    out.insert(out.end(), central.begin(), central.end());
    put<uint32_t>(out, 0x06054b50u);
    put<uint32_t>(out, 0); // disks
    put<uint16_t>(out, static_cast<uint16_t>(entries.size()));
    put<uint16_t>(out, static_cast<uint16_t>(entries.size()));
    put<uint32_t>(out, static_cast<uint32_t>(central.size()));
    put<uint32_t>(out, centralOffset);
    put<uint16_t>(out, 0); // comment
    return out;
}

Bytes toBytes(const Json & json)
{
    const std::string text = json.dump();
    return Bytes(text.begin(), text.end());
}

// Nearest-codebook distance: the exact bound for a value labeled with its
// nearest codebook entry.
double nearestDistance(const std::vector<double> & codebook, double value)
{
    double best = INFINITY;
    for (double c : codebook)
    {
        best = std::min(best, std::abs(c - value));
    }
    return best;
}

bool sameCloud(GaussianCloud & a, GaussianCloud & b)
{
    return a.count() == b.count() && a.shDegree() == b.shDegree() && a.positions() == b.positions()
           && a.scales() == b.scales() && a.rotations() == b.rotations() && a.opacities() == b.opacities()
           && a.shCoefficients() == b.shCoefficients();
}

// --- 1 + 3: the real file against its source ---
//
// Bounds (splat-transform's writer):
//   position   x = floor(65535 * (log1p-transform(p) - min) / (max - min)):
//              the decoded value is up to one step BELOW in log space, i.e.
//              |dx| <= (|p| + 1) * (exp(step) - 1), step = (max - min) / 65535.
//   rotation   three lanes trunc(255 * (q * sqrt2 * 0.5 + 0.5)): each lane off
//              by < 2/255 / sqrt2 = 0.0055; the fourth rebuilt from unit
//              length -> angle <= 2 * |dq| < 0.03 rad.
//   scale, DC  nearest codebook entry (log scale / raw f_dc) -- exact.
//   opacity    trunc(sigmoid * 255) -> one-sided, under 1/255.
//   SH 1-3     k-means palette (256 entries for 300 Gaussians) then a
//              256-entry codebook. No closed form per Gaussian: one that owns
//              its centroid decodes to within the codebook spacing (~0.004),
//              while k-means merges the rest with a neighbor, whose error is
//              then the full distance to the cluster mean (up to ~0.45 here).
//              So the check is that most Gaussians are near-exact -- which a
//              layout misread makes impossible (guarded below) -- plus a mean.
//              (Measured separately: this file's decode matches splat-
//              transform's own reader of it to 0 on every SH value, so the
//              tail is the encoder's k-means, not the reader.)
constexpr double kRotationBound = 0.03;
constexpr double kShNearExact = 0.01;
constexpr double kShMeanBound = 0.05;

int testFixture(Device & device, Renderer & renderer)
{
    const Bytes bytes = readBytes(kFixture);
    CHECK(!bytes.empty(), "fixture missing: " + kFixture);
    CHECK(detectFormat(kFixture) == SceneFormat::SOG, ".sog must dispatch to SogReader");
    const Archive archive = unzipStored(bytes);
    CHECK(archive.count("meta.json") == 1, "fixture has no meta.json");
    const Json meta = Json::parse(archive.at("meta.json"));
    std::printf("  synthetic.sog (%zu bytes, %zu entries, %s)\n", bytes.size(), archive.size(),
                meta["asset"]["generator"].get<std::string>().c_str());

    auto source = loadScene(device, kSourcePly);
    auto cloud = loadScene(device, kFixture);
    CHECK(cloud->count() == kSceneSize, "wrong Gaussian count");
    CHECK(cloud->shDegree() == 3, "SH degree must be meta.json's shN.bands (3)");
    CHECK(!cloud->antialiased(), "an untagged model must not be antialiased");

    const std::vector<double> scaleCodebook = meta["scales"]["codebook"].get<std::vector<double>>();
    const std::vector<double> dcCodebook = meta["sh0"]["codebook"].get<std::vector<double>>();
    const std::vector<double> shCodebook = meta["shN"]["codebook"].get<std::vector<double>>();
    const std::vector<double> mins = meta["means"]["mins"].get<std::vector<double>>();
    const std::vector<double> maxs = meta["means"]["maxs"].get<std::vector<double>>();

    size_t positionViolations = 0;
    size_t codebookViolations = 0;
    size_t opacityViolations = 0;
    size_t shNotInCodebook = 0;
    size_t shNearExact = 0; // Gaussians whose every SH value is within kShNearExact
    fixture::ErrorStat positionSlack; // error / per-Gaussian bound
    const auto perPair = [&](size_t d, size_t s)
    {
        for (int a = 0; a < 3; ++a)
        {
            const double p = source->positions()[s][a];
            const double step = (maxs[a] - mins[a]) / 65535.0;
            const double bound = (std::abs(p) + 1.0) * std::expm1(step) * 1.001 + 2e-6;
            const double error = std::abs(cloud->positions()[d][a] - p);
            positionSlack.add(error / bound);
            positionViolations += error > bound ? 1 : 0;

            const double logSource = std::log(static_cast<double>(source->scales()[s][a]));
            const double logDecoded = std::log(static_cast<double>(cloud->scales()[d][a]));
            codebookViolations +=
                std::abs(logDecoded - logSource) > nearestDistance(scaleCodebook, logSource) + 1e-5 ? 1 : 0;
            const double dcSource = source->shCoefficients()[s * kShCoefficientsPerGaussian][a];
            const double dcDecoded = cloud->shCoefficients()[d * kShCoefficientsPerGaussian][a];
            codebookViolations += std::abs(dcDecoded - dcSource) > nearestDistance(dcCodebook, dcSource) + 1e-6 ? 1 : 0;
        }
        const double opacityDrop = static_cast<double>(source->opacities()[s]) - cloud->opacities()[d];
        opacityViolations += (opacityDrop < -1e-6 || opacityDrop > 1.0 / 255.0 + 1e-6) ? 1 : 0;
        double worstSh = 0.0;
        for (int k = 1; k < kShCoefficientsPerGaussian; ++k)
        {
            for (int c = 0; c < 3; ++c)
            {
                worstSh = std::max(worstSh, static_cast<double>(std::abs(
                                                cloud->shCoefficients()[d * kShCoefficientsPerGaussian + k][c]
                                                - source->shCoefficients()[s * kShCoefficientsPerGaussian + k][c])));
                shNotInCodebook +=
                    nearestDistance(shCodebook, cloud->shCoefficients()[d * kShCoefficientsPerGaussian + k][c]) > 1e-6
                        ? 1
                        : 0;
            }
        }
        shNearExact += worstSh <= kShNearExact ? 1 : 0;
    };
    const fixture::AttributeErrors e = fixture::compareToSource(*cloud, *source, false, 0.01f, perPair);
    fixture::printStat("position (abs)", e.position);
    fixture::printStat("position (error / per-Gaussian bound)", positionSlack, 1.0);
    fixture::printStat("scale (relative; nearest codebook)", e.scaleRelative);
    fixture::printStat("rotation (radians)", e.rotationAngle, kRotationBound);
    fixture::printStat("opacity (abs, one-sided)", e.opacity, 1.0 / 255.0);
    fixture::printStat("SH DC (abs; nearest codebook)", e.dc);
    fixture::printStat("SH band 1 (abs)", e.sh[0]);
    fixture::printStat("SH band 2 (abs)", e.sh[1]);
    fixture::printStat("SH band 3 (abs)", e.sh[2]);
    std::printf("    SH near-exact (all 45 values within %.3g): %zu of %zu Gaussians\n", kShNearExact, shNearExact,
                kSceneSize);

    CHECK(e.matched == kSceneSize && e.unmatched == 0 && e.sourceUsedTwice == 0,
          "decoded Gaussians do not map one-to-one onto the source");
    CHECK(positionViolations == 0, "position outside the 16-bit log-space quantization bound");
    CHECK(codebookViolations == 0,
          "scale or SH DC is not the nearest codebook entry to its source (scale codebook is LOG -- exp() "
          "once; sh0 codebook is RAW f_dc -- no display transform)");
    CHECK(e.rotationAngle.max <= kRotationBound,
          "rotation error -- smallest-three lanes are w,x,y,z and must be reordered to xyzw");
    CHECK(opacityViolations == 0, "opacity is not alpha/255 (one-sided, under one step)");
    CHECK(shNotInCodebook == 0, "a decoded SH value is not an shN codebook entry");
    // 256 clusters over 300 Gaussians leave ~212 owning their centroid.
    CHECK(shNearExact >= kSceneSize * 2 / 3,
          "SH: too few Gaussians decode near-exactly -- centroid texels are coefficient-major, RGB = channel");
    for (int band = 0; band < 3; ++band)
    {
        CHECK(e.sh[band].mean() <= kShMeanBound, "SH band " + std::to_string(band + 1) + " mean error too large");
    }

    // --- Degeneracy guards ---
    size_t laneWitnesses = 0;
    size_t scaleWitnesses = 0;
    size_t shWitnesses = 0;
    double misreadShError = 0.0;
    for (size_t s = 0; s < source->count(); ++s)
    {
        const glm::vec4 q = source->rotations()[s];
        laneWitnesses += fixture::rotationAngle(q, glm::vec4(q.w, q.x, q.y, q.z)) > 10.0 * kRotationBound ? 1 : 0;
        scaleWitnesses += std::abs(std::exp(source->scales()[s].x) / source->scales()[s].x - 1.0f) > 0.1f ? 1 : 0;
        // A channel-major read of the palette (channel c, coefficient j at
        // texel c * coeffs + j) would hand coefficient 1's G slot coefficient
        // 2's R value, etc. The source values must differ by far more than
        // the SH bound for that misread to be caught.
        const glm::vec4 * sh = &source->shCoefficients()[s * kShCoefficientsPerGaussian];
        const double d = std::abs(sh[1].y - sh[2].x);
        misreadShError += d;
        shWitnesses += d > 10.0 * kShNearExact ? 1 : 0;
    }
    CHECK(laneWitnesses > kSceneSize * 9 / 10, "degenerate -- rotations would survive a lane mix-up");
    CHECK(scaleWitnesses == kSceneSize, "degenerate -- scales where exp() would not show");
    CHECK(shWitnesses > kSceneSize * 3 / 4 && misreadShError / kSceneSize > 3.0 * kShMeanBound,
          "degenerate -- an SH layout misread would stay within the SH bound");
    CHECK(std::abs(mins[0]) > 1.0 && maxs[0] > 1.0 && mins[0] < 0.0,
          "degenerate -- positions must span both signs through the log transform");

    // Twice: at SH 0 the image sees only geometry, DC and opacity, whose
    // quantization is fine; at SH 3 it also sees the k-means-merged SH tail
    // above, which is the encoder's loss, not the reader's.
    const fixture::ImageDiff dcOnly = fixture::renderAndDiff(renderer, kSourcePly, kFixture, 0);
    const fixture::ImageDiff full = fixture::renderAndDiff(renderer, kSourcePly, kFixture, 3);
    fixture::printImageDiff("render vs source .ply (SH 0)", dcOnly);
    fixture::printImageDiff("render vs source .ply (SH<=3)", full);
    CHECK(dcOnly.meanCoverage > 0.02, "degenerate render -- the reference image is nearly empty");
    CHECK(dcOnly.rmse < 0.01, "rendered image (SH 0) differs from the source scene's beyond quantization");
    CHECK(full.rmse < 0.04, "rendered image (SH<=3) differs from the source scene's beyond quantization");
    return 0;
}

// --- 2: every physical form decodes identically ---
int testForms(Device & device)
{
    const Archive archive = unzipStored(readBytes(kFixture));
    auto bundled = loadScene(device, kFixture);

    // Unbundled: meta.json beside its images, loaded by name.
    const std::filesystem::path dir = "sog_reader_test_unbundled";
    std::filesystem::create_directories(dir);
    for (const auto & [name, data] : archive)
    {
        writeBytes((dir / name).string(), data);
    }
    const std::string metaPath = (dir / "meta.json").string();
    CHECK(detectFormat(metaPath) == SceneFormat::SOG, "meta.json must dispatch to SogReader");
    auto unbundled = loadScene(device, metaPath);
    CHECK(sameCloud(*bundled, *unbundled), "unbundled meta.json + images decode differently from the .sog");

    // Re-zipped with DEFLATE entries.
    const std::string deflated = "sog_reader_test_deflate.sog";
    writeBytes(deflated, zip(archive, 8));
    auto fromDeflate = loadScene(device, deflated);
    CHECK(sameCloud(*bundled, *fromDeflate), "a DEFLATE-zipped .sog decodes differently from the STORED one");

    // The same images re-zipped STORED by this test's writer: proves the
    // reader does not depend on splat-transform's data-descriptor quirk.
    const std::string restored = "sog_reader_test_stored.sog";
    writeBytes(restored, zip(archive, 0));
    CHECK(sameCloud(*bundled, *loadScene(device, restored)), "a re-zipped STORED .sog decodes differently");

    // model: "antialiased" reaches the cloud; nothing else changes.
    Archive tagged = archive;
    Json meta = Json::parse(archive.at("meta.json"));
    meta["model"] = "antialiased";
    tagged["meta.json"] = toBytes(meta);
    writeBytes("sog_reader_test_antialiased.sog", zip(tagged, 0));
    auto antialiased = loadScene(device, "sog_reader_test_antialiased.sog");
    CHECK(antialiased->antialiased() && sameCloud(*bundled, *antialiased),
          "meta.json model 'antialiased' must set the flag and change nothing else");

    // An older v2 draft without shN.bands / shN.count (the PlayCanvas engine's
    // own examples/assets/splats/skull.sog): both derive from the centroid
    // image -- width 64 * coeffs, one row per 64 palette entries.
    Archive legacy = archive;
    Json legacyMeta = Json::parse(archive.at("meta.json"));
    legacyMeta.at("shN").erase("bands");
    legacyMeta.at("shN").erase("count");
    legacy["meta.json"] = toBytes(legacyMeta);
    writeBytes("sog_reader_test_legacy_shn.sog", zip(legacy, 0));
    auto legacyCloud = loadScene(device, "sog_reader_test_legacy_shn.sog");
    CHECK(legacyCloud->shDegree() == 3 && sameCloud(*bundled, *legacyCloud),
          "a meta.json without shN.bands/count must derive them from the centroid image and decode identically");

    std::cout << "  unbundled meta.json, DEFLATE and re-zipped STORED bundles: bit-identical decode; "
                 "model 'antialiased' honored"
              << std::endl;
    return 0;
}

// --- 4: corruptions, each rejected for its own reason ---
int testRejections(Device & device)
{
    const Bytes original = readBytes(kFixture);
    const Archive archive = unzipStored(original);
    const Json meta = Json::parse(archive.at("meta.json"));

    struct Case
    {
        std::string what;
        std::string file;
        Bytes bytes;
        std::string expect;
    };
    std::vector<Case> cases;
    const auto withMeta = [&](const std::string & what, const std::function<void(Json &)> & edit,
                              const std::string & expect)
    {
        Archive a = archive;
        Json m = meta;
        edit(m);
        a["meta.json"] = toBytes(m);
        cases.push_back({what, "", zip(a, 0), expect});
    };

    cases.push_back({"not a ZIP", "", Bytes(4096, 0xAB), "not a ZIP archive"});
    cases.push_back({"empty file", "", {}, "is empty"});
    cases.push_back({"truncated end record", "", Bytes(original.begin(), original.end() - 10), "not a ZIP archive"});
    {
        // Flip a byte inside the first entry's data (STORED: local header 30 B + name).
        Bytes b = original;
        b[30 + peek<uint16_t>(b, 26) + 40] ^= 0x5A;
        cases.push_back({"corrupt entry data", "", b, "CRC-32"});
    }
    {
        Archive a = archive;
        Bytes b = zip(a, 0);
        // Central directory method field of the first entry -> 12 (bzip2).
        const size_t central = peek<uint32_t>(b, b.size() - 22 + 16);
        b[central + 10] = 12;
        cases.push_back({"unsupported ZIP method", "", b, "compression method 12"});
    }
    {
        Archive a = archive;
        a.erase("quats.webp");
        cases.push_back({"image missing from bundle", "", zip(a, 0), "has no entry 'quats.webp'"});
    }
    {
        Archive a = archive;
        a["meta.json"] = Bytes{'{', '"', 'v'};
        cases.push_back({"meta.json not JSON", "", zip(a, 0), "not valid JSON"});
    }
    withMeta("legacy v1 (no version)", [](Json & m) { m.erase("version"); }, "legacy SOG v1");
    withMeta("future version", [](Json & m) { m["version"] = 3; }, "unsupported SOG meta.json version");
    withMeta("missing quats", [](Json & m) { m.erase("quats"); }, "missing 'quats'");
    withMeta("2dgs model", [](Json & m) { m["model"] = "2dgs"; }, "2dgs");
    withMeta("count past the images", [](Json & m) { m["count"] = 1000000; }, "too small for");
    withMeta("bands vs centroid width", [](Json & m) { m["shN"]["bands"] = 2; }, "SH centroid image");
    withMeta("oversized codebook",
             [](Json & m) { m["scales"]["codebook"] = std::vector<double>(300, 0.0); }, "1..256 entries");
    withMeta("short codebook",
             [](Json & m) { m["sh0"]["codebook"] = std::vector<double>(4, 0.0); }, "indexes past");
    withMeta("palette smaller than labels", [](Json & m) { m["shN"]["count"] = 1; }, "outside the 1-entry palette");
    withMeta("means.mins wrong length", [](Json & m) { m["means"]["mins"] = {0.0, 0.0}; }, "expected 3");
    withMeta("not a WebP", [](Json & m) { m["quats"]["files"] = {"meta.json"}; }, "not a decodable WebP");

    for (size_t i = 0; i < cases.size(); ++i)
    {
        const std::string path = "sog_reader_test_bad_" + std::to_string(i) + ".sog";
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
        CHECK(!error.empty(), "rejection: accepted a corrupt file -- " + cases[i].what);
        CHECK(error.find(cases[i].expect) != std::string::npos,
              "rejection: " + cases[i].what + " rejected for the wrong reason: " + error);
        std::printf("  rejected %-30s %.100s\n", cases[i].what.c_str(), error.c_str());
    }

    // Unbundled form: an image missing beside meta.json.
    const std::filesystem::path dir = "sog_reader_test_missing";
    std::filesystem::create_directories(dir);
    writeBytes((dir / "meta.json").string(), archive.at("meta.json"));
    std::string error;
    try
    {
        loadScene(device, (dir / "meta.json").string());
    }
    catch (const std::exception & ex)
    {
        error = ex.what();
    }
    CHECK(error.find("failed to open SOG image") != std::string::npos, "rejection: missing unbundled image: " + error);

    error.clear();
    try
    {
        detectFormat("scene/lod-meta.json");
    }
    catch (const std::exception & ex)
    {
        error = ex.what();
    }
    CHECK(error.find("streamed SOG") != std::string::npos, "rejection: lod-meta.json must be refused by name");
    CHECK(detectFormat("Scene.SOG") == SceneFormat::SOG, "dispatch: extension must be case-insensitive");
    std::cout << "  rejected missing unbundled image and streamed lod-meta.json" << std::endl;
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
    examples::ScratchDir scratch("sog_reader_test");
    fixture::writePly(kSourcePly, fixture::makeSyntheticScene());

    Device device;
    Renderer renderer(320, 240);
    int result = 0;
    try
    {
        device.initialize({}, /*enableValidation=*/true);
        if (!renderer.initialize({}, true, SHADER_DIR))
        {
            std::cerr << "FAILED: renderer initialize" << std::endl;
            result = 1;
        }
        if (result == 0)
        {
            std::cout << "splat-transform-generated fixture vs. its source .ply:" << std::endl;
            result = testFixture(device, renderer);
        }
        if (result == 0)
        {
            std::cout << "physical forms:" << std::endl;
            result = testForms(device);
        }
        if (result == 0)
        {
            std::cout << "malformed inputs:" << std::endl;
            result = testRejections(device);
        }
        if (result == 0)
        {
            std::cout << "SogReader test OK" << std::endl;
        }
    }
    catch (const std::exception & e)
    {
        std::cerr << "SogReader test FAILED: " << e.what() << std::endl;
        result = 1;
    }
    renderer.shutdown();
    device.shutdown();
    return result;
}
