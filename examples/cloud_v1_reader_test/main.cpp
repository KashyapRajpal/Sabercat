// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Headless test for the cloud-v1 loader (CloudV1Reader + the loadScene()
// dispatch).
//
// Fixtures are written AT RUNTIME by a writer in this file that mirrors the
// reference cloud-v1 layout field for field (stream names, shapes,
// "float32-little-endian", elementStrideBytes = strides[0], byteLength, and the
// manifest keys), so the reader is checked against an independent writer.
// Pass a cloud-v1 fixture directory as argv[1] to also load a real one; argv[2]
// overrides the .ply used for the rendering check.
//
//  (a) decode: degree 3 (sh0 + 12-byte-packed sh_rest + confidence) and
//      degree 0 fixtures, loaded by directory and by manifest path, compared
//      bit-exactly -- including the sh0/sh_rest merge into the padded
//      16-coefficient layout, the one stream that is NOT a straight copy.
//      The legacy schema id (files written before Sabercat 1.0) reads
//      bit-exactly too.
//  (b) rejections: every manifest field that changes the meaning of the
//      bytes, size mismatches, values outside the canonical contract, and an
//      unknown schema id.
//  (c) rendering: a .ply and the cloud-v1 fixture converted from it render
//      BIT-IDENTICALLY through Renderer (synthetic degree-3 scene always;
//      assets/bonsai-7k.ply when present).
//
//  (d) writer: encode() -> read(FileMap) and write() -> read(path) reproduce
//      every canonical host stream bit-exactly for degrees 0, 1 and 3; the
//      manifest has exactly the reference keys and the new schema id.
//  (e) relight sidecar: all four streams, and a normals-only subset, decode
//      bit-exactly; a corrupted sidecar (count, truncated stream, schema) is
//      rejected and leaves the cloud untouched.
//
// Validation messages are logged as "[validation]"; the run must have none.

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "sabercat/camera.h"
#include "sabercat/cloud_v1_reader.h"
#include "sabercat/cloud_v1_writer.h"
#include "sabercat/file_map.h"
#include "sabercat/device.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/render_config.h"
#include "sabercat/renderer.h"
#include "sabercat/scene_loader.h"

using namespace sabercat;
namespace fs = std::filesystem;

// million_check.cpp: a 1M-Gaussian degree-3 cloud with relight streams, bit-exact.
int runMillionCloudV1(Device & device, const fs::path & dir);

namespace
{

int g_failures = 0;
fs::path g_tempDir;

void check(bool ok, const std::string & what)
{
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok)
    {
        ++g_failures;
    }
}

std::string num(double value)
{
    std::ostringstream s;
    s << std::setprecision(9) << value;
    return s.str();
}

// ---------------------------------------------------------------------------
// Fixture writer (mirrors the reference cloud-v1 layout)
// ---------------------------------------------------------------------------

struct Stream
{
    std::string name;
    std::vector<int64_t> shape;
    std::vector<float> values;
};

// Logical cloud: means/scales [N,3], rotations [N,4]
// xyzw, opacities [N], sh [N, (D+1)^2, 3], optional confidence [N].
struct Logical
{
    std::vector<float> means, scales, rotations, opacities, sh, confidence;
    size_t count = 0;
    int degree = 0;
};

std::vector<Stream> packStreams(const Logical & c)
{
    const int64_t n = static_cast<int64_t>(c.count);
    auto float4 = [&](const std::vector<float> & xyz)
    {
        std::vector<float> out(c.count * 4, 0.0f); // pack_float4: w = 0
        for (size_t i = 0; i < c.count; ++i)
            for (int k = 0; k < 3; ++k)
                out[i * 4 + k] = xyz[i * 3 + k];
        return out;
    };
    const int coefficients = (c.degree + 1) * (c.degree + 1);
    std::vector<float> dc(c.count * 3);
    for (size_t i = 0; i < c.count; ++i)
        for (int k = 0; k < 3; ++k)
            dc[i * 3 + k] = c.sh[(i * coefficients) * 3 + k];
    std::vector<Stream> streams = {{"means", {n, 4}, float4(c.means)},
                                   {"scales", {n, 4}, float4(c.scales)},
                                   {"rotations", {n, 4}, c.rotations},
                                   {"opacities", {n}, c.opacities},
                                   {"sh0", {n, 4}, float4(dc)}};
    if (coefficients > 1)
    {
        std::vector<float> rest;
        for (size_t i = 0; i < c.count; ++i)
            for (int k = 1; k < coefficients; ++k)
                for (int ch = 0; ch < 3; ++ch)
                    rest.push_back(c.sh[(i * coefficients + k) * 3 + ch]);
        streams.push_back({"sh_rest", {n, coefficients - 1, 3}, rest});
    }
    if (!c.confidence.empty())
        streams.push_back({"confidence", {n}, c.confidence});
    return streams;
}

// Writes the streams + manifest. `edit` may rewrite the manifest text before
// it is saved (the rejection cases each change one thing).
fs::path writeFixture(const std::string & name, const Logical & c,
                      const std::function<void(std::string &)> & edit = nullptr,
                      const std::function<void(std::vector<Stream> &)> & editStreams = nullptr)
{
    const fs::path dir = g_tempDir / name;
    fs::create_directories(dir);
    std::vector<Stream> streams = packStreams(c);
    if (editStreams)
        editStreams(streams);
    std::string entries;
    for (const Stream & s : streams)
    {
        const std::string file = s.name + ".f32le.bin";
        std::ofstream out(dir / file, std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char *>(s.values.data()),
                  static_cast<std::streamsize>(s.values.size() * 4));
        int64_t stride = 4;
        std::string shape = "[";
        for (size_t i = 0; i < s.shape.size(); ++i)
        {
            shape += (i ? ", " : "") + std::to_string(s.shape[i]);
            if (i > 0)
                stride *= s.shape[i];
        }
        shape += "]";
        entries += (entries.empty() ? "" : ",\n") + std::string("    \"") + s.name
                   + "\": {\"file\": \"" + file + "\", \"shape\": " + shape
                   + ", \"scalarType\": \"float32-little-endian\", \"elementStrideBytes\": "
                   + std::to_string(stride)
                   + ", \"byteLength\": " + std::to_string(s.values.size() * 4) + "}";
    }
    std::string manifest =
        "{\n  \"schema\": \"sabercat.cloud-v1\",\n  \"count\": " + std::to_string(c.count)
        + ",\n  \"coordinateSystem\": \"right-handed-y-up-camera-minus-z\",\n"
          "  \"quaternionOrder\": \"xyzw\",\n  \"colorSpace\": \"linear-srgb\",\n  \"shDegree\": "
        + std::to_string(c.degree)
        + ",\n  \"shLayout\": \"splat-major-coefficient-major-rgb\",\n  \"streams\": {\n" + entries
        + "\n  }\n}\n";
    if (edit)
        edit(manifest);
    std::ofstream(dir / "manifest.json", std::ios::binary | std::ios::trunc) << manifest;
    return dir;
}

void replaceOnce(std::string & text, const std::string & from, const std::string & to)
{
    const size_t at = text.find(from);
    if (at != std::string::npos)
        text.replace(at, from.size(), to);
}

Logical makeLogical(size_t n, int degree, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    Logical c;
    c.count = n;
    c.degree = degree;
    const int coefficients = (degree + 1) * (degree + 1);
    for (size_t i = 0; i < n; ++i)
    {
        for (int k = 0; k < 3; ++k)
        {
            c.means.push_back(u(rng) * 3.0f);
            c.scales.push_back(
                0.5f + 1.5f * (u(rng) + 1.0f) * 0.5f); // straddles 1: a spurious exp() shows
        }
        float q[4] = {u(rng), u(rng), u(rng), u(rng)};
        const float l = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
        for (float v : q)
            c.rotations.push_back(v / l);
        c.opacities.push_back(0.1f + 0.8f * (u(rng) + 1.0f) * 0.5f);
        for (int k = 0; k < coefficients * 3; ++k)
            c.sh.push_back(u(rng) * (k < 3 ? 2.0f : 0.5f));
        c.confidence.push_back(0.5f);
    }
    return c;
}

// ---------------------------------------------------------------------------
// (a) decode
// ---------------------------------------------------------------------------

void expectExact(Device & device, const fs::path & path, const Logical & c,
                 const std::string & label)
{
    auto cloud = loadScene(device, path.string());
    check(cloud->count() == c.count && cloud->shDegree() == static_cast<uint32_t>(c.degree),
          label + ": count " + std::to_string(cloud->count()) + ", SH degree "
              + std::to_string(cloud->shDegree()));
    if (cloud->count() != c.count)
        return;
    const int coefficients = (c.degree + 1) * (c.degree + 1);
    size_t bad = 0;
    for (size_t i = 0; i < c.count; ++i)
    {
        bad += cloud->positions()[i]
               != glm::vec4(c.means[i * 3], c.means[i * 3 + 1], c.means[i * 3 + 2], 0.0f);
        bad += cloud->scales()[i]
               != glm::vec4(c.scales[i * 3], c.scales[i * 3 + 1], c.scales[i * 3 + 2], 0.0f);
        bad += cloud->rotations()[i]
               != glm::vec4(c.rotations[i * 4],
                            c.rotations[i * 4 + 1],
                            c.rotations[i * 4 + 2],
                            c.rotations[i * 4 + 3]);
        bad += cloud->opacities()[i] != c.opacities[i];
        for (int k = 0; k < kShCoefficientsPerGaussian; ++k)
        {
            const glm::vec4 expected = k < coefficients
                                           ? glm::vec4(c.sh[(i * coefficients + k) * 3],
                                                       c.sh[(i * coefficients + k) * 3 + 1],
                                                       c.sh[(i * coefficients + k) * 3 + 2],
                                                       0.0f)
                                           : glm::vec4(0.0f);
            bad += cloud->shCoefficients()[i * 16 + k] != expected;
        }
    }
    check(bad == 0,
          label + ": every stream bit-exact, SH merged sh0 + sh_rest, unused slots exactly zero ("
              + std::to_string(bad) + " mismatches)");
    check(!cloud->antialiased(), label + ": antialiased defaults to false");
}

void testDecode(Device & device)
{
    std::printf("\n-- (a) decode --\n");
    const Logical deg3 = makeLogical(7, 3, 1);
    const fs::path dir3 = writeFixture("deg3", deg3);
    expectExact(device, dir3, deg3, "degree 3 by directory");
    expectExact(device, dir3 / "manifest.json", deg3, "degree 3 by manifest.json");

    // Guard: a reader that treated sh_rest as 16-byte padded coefficients (or
    // channel-major) would misplace these distinct values.
    {
        auto cloud = loadScene(device, dir3.string());
        const glm::vec4 c5 = cloud->shCoefficients()[2 * 16 + 5];
        const float * src = &deg3.sh[(2 * 16 + 5) * 3];
        check(c5.x == src[0] && c5.y == src[1] && c5.z == src[2] && src[0] != src[1],
              "guard: sh_rest is read at 12 B per coefficient, RGB inner");
    }

    const Logical deg1 = makeLogical(5, 1, 2);
    expectExact(device, writeFixture("deg1", deg1), deg1, "degree 1");

    // Degree 0 without confidence: the smallest shape a fixture takes.
    Logical deg0 = makeLogical(5, 0, 3);
    deg0.confidence.clear();
    expectExact(device, writeFixture("deg0", deg0), deg0, "degree 0, no confidence");

    // Files written before Sabercat 1.0 carry the legacy schema id: they must
    // still read, bit-exactly, and nothing else in the manifest changes.
    bool swapped = false;
    const fs::path legacy = writeFixture("legacy_deg3", deg3, [&](std::string & m)
                                         {
                                             const std::string current = "\"sabercat.cloud-v1\"";
                                             swapped = m.find(current) != std::string::npos;
                                             replaceOnce(m, current,
                                                         "\"gaussmith.sabercat-cloud-v1\"");
                                         });
    check(swapped, "legacy fixture: the manifest was written with the legacy schema id");
    expectExact(device, legacy, deg3, "legacy schema id, degree 3");
}

// ---------------------------------------------------------------------------
// (b) rejections
// ---------------------------------------------------------------------------

void expectRejected(Device & device, const fs::path & path, const std::string & needle,
                    const std::string & what)
{
    std::string message;
    try
    {
        loadScene(device, path.string());
    }
    catch (const std::exception & e)
    {
        message = e.what();
    }
    const bool ok = !message.empty() && message.find(needle) != std::string::npos;
    check(ok,
          "rejects " + what
              + (message.empty() ? " (ACCEPTED)"
                                 : " -- \"" + message.substr(0, message.find(": C:")) + "\""));
}

void testRejections(Device & device)
{
    std::printf("\n-- (b) malformed fixtures --\n");
    const Logical c = makeLogical(4, 2, 9);
    auto edited = [&](const std::string & name, const std::string & from, const std::string & to)
    {
        return writeFixture(name, c, [&](std::string & m) { replaceOnce(m, from, to); });
    };

    expectRejected(
        device, edited("schema", "sabercat.cloud-v1", "sabercat.cloud-v2"), "schema", "schema v2");
    expectRejected(device,
                   edited("schema_unknown", "sabercat.cloud-v1", "gaussmith.cloud-v1"),
                   "sabercat.cloud-v1",
                   "an unknown schema id (the message names both accepted ids)");
    expectRejected(
        device, edited("quat", "\"xyzw\"", "\"wxyz\""), "quaternionOrder", "quaternionOrder wxyz");
    expectRejected(device,
                   edited("coords", "right-handed-y-up-camera-minus-z", "right-handed-z-up"),
                   "coordinateSystem",
                   "a different coordinate system");
    expectRejected(device,
                   edited("layout", "splat-major-coefficient-major-rgb", "channel-major"),
                   "shLayout",
                   "a channel-major SH layout");
    expectRejected(device,
                   edited("scalar", "float32-little-endian", "float16-little-endian"),
                   "scalarType",
                   "float16 streams");
    expectRejected(device,
                   edited("bytelength", "\"byteLength\": 64}", "\"byteLength\": 60}"),
                   "byteLength",
                   "a byteLength that disagrees with the shape");
    expectRejected(device,
                   edited("stride", "\"elementStrideBytes\": 16", "\"elementStrideBytes\": 12"),
                   "elementStrideBytes",
                   "a 12 B stride on a [N,4] stream");
    expectRejected(device,
                   edited("degree", "\"shDegree\": 2", "\"shDegree\": 3"),
                   "shape",
                   "sh_rest shape inconsistent with shDegree");
    expectRejected(device,
                   edited("count", "\"count\": 4", "\"count\": 5"),
                   "shape",
                   "count disagreeing with the streams");
    expectRejected(device,
                   edited("deg0rest", "\"shDegree\": 2", "\"shDegree\": 0"),
                   "sh_rest",
                   "sh_rest with shDegree 0");
    expectRejected(
        device,
        writeFixture("missing", c, nullptr, [](std::vector<Stream> & s) { s.erase(s.begin()); }),
        "missing required stream 'means'",
        "a missing means stream");
    expectRejected(device,
                   writeFixture("norest",
                                c,
                                nullptr,
                                [](std::vector<Stream> & s)
                                {
                                    s.erase(std::remove_if(s.begin(),
                                                           s.end(),
                                                           [](const Stream & x)
                                                           { return x.name == "sh_rest"; }),
                                            s.end());
                                }),
                   "sh_rest",
                   "degree 2 without sh_rest");
    {
        const fs::path dir = writeFixture("truncated", c);
        fs::resize_file(dir / "scales.f32le.bin", 40);
        expectRejected(device, dir, "bytes, but the manifest says", "a truncated stream file");
    }
    {
        Logical bad = c;
        bad.opacities[1] = 1.25f;
        expectRejected(device, writeFixture("opacity", bad), "opacity 1", "opacity 1.25");
    }
    {
        Logical bad = c;
        bad.scales[4] = -0.1f;
        expectRejected(device, writeFixture("scale", bad), "scale 1", "a negative scale");
    }
    {
        Logical bad = c;
        std::fill(bad.rotations.begin() + 4, bad.rotations.begin() + 8, 0.0f);
        expectRejected(device, writeFixture("quatzero", bad), "zero-length", "a zero quaternion");
    }
    {
        const fs::path dir = g_tempDir / "garbage";
        fs::create_directories(dir);
        std::ofstream(dir / "manifest.json") << "{ \"schema\": ";
        expectRejected(device, dir, "malformed manifest", "unparseable JSON");
    }
    expectRejected(device,
                   g_tempDir / "no_such_dir" / "manifest.json",
                   "failed to open",
                   "a missing manifest");
}

// ---------------------------------------------------------------------------
// (c) rendering
// ---------------------------------------------------------------------------

void writeDegree3Ply(const std::string & path, size_t n, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file << "ply\nformat binary_little_endian 1.0\nelement vertex " << n << "\n";
    std::vector<std::string> names = {"x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2"};
    for (int i = 0; i < 45; ++i)
        names.push_back("f_rest_" + std::to_string(i));
    for (const char * name :
         {"opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"})
        names.push_back(name);
    for (const auto & name : names)
        file << "property float " << name << "\n";
    file << "end_header\n";
    for (size_t i = 0; i < n; ++i)
    {
        std::vector<float> v;
        v.push_back(u(rng) * 1.5f);
        v.push_back(u(rng) * 1.0f);
        v.push_back(-static_cast<float>(i) * (2.0f / static_cast<float>(n)) + u(rng) * 0.001f);
        for (int c = 0; c < 48; ++c)
            v.push_back(u(rng) * (c < 3 ? 1.5f : 0.4f));
        v.push_back(u(rng) * 2.0f);
        for (int c = 0; c < 3; ++c)
            v.push_back(std::log(0.04f + 0.06f * (u(rng) + 1.0f)));
        for (int c = 0; c < 4; ++c)
            v.push_back(u(rng));
        file.write(reinterpret_cast<const char *>(v.data()),
                   static_cast<std::streamsize>(v.size() * 4));
    }
}

// Converter: canonical cloud -> logical tensors -> fixture.
Logical fromCloud(GaussianCloud & cloud)
{
    Logical c;
    c.count = cloud.count();
    c.degree = static_cast<int>(cloud.shDegree());
    const int coefficients = (c.degree + 1) * (c.degree + 1);
    for (size_t i = 0; i < c.count; ++i)
    {
        for (int k = 0; k < 3; ++k)
        {
            c.means.push_back(cloud.positions()[i][k]);
            c.scales.push_back(cloud.scales()[i][k]);
        }
        for (int k = 0; k < 4; ++k)
            c.rotations.push_back(cloud.rotations()[i][k]);
        c.opacities.push_back(cloud.opacities()[i]);
        for (int k = 0; k < coefficients; ++k)
            for (int ch = 0; ch < 3; ++ch)
                c.sh.push_back(cloud.shCoefficients()[i * 16 + k][ch]);
    }
    return c;
}

constexpr uint32_t kWidth = 256;
constexpr uint32_t kHeight = 192;

Camera makeCamera(const glm::vec3 & center, float radius)
{
    Camera camera;
    camera.width = kWidth;
    camera.height = kHeight;
    camera.fx = camera.fy = 0.9f * kWidth;
    camera.cx = 0.5f * kWidth;
    camera.cy = 0.5f * kHeight;
    const glm::vec3 eye = center + glm::normalize(glm::vec3(0.25f, 0.35f, 1.0f)) * (2.2f * radius);
    camera.cameraToWorld = glm::inverse(glm::lookAt(eye, center, glm::vec3(0.0f, 1.0f, 0.0f)));
    return camera;
}

void crossFormat(Renderer & renderer, Device & device, const std::string & plyPath,
                 const std::string & label)
{
    auto cloud = loadScene(device, plyPath);
    const fs::path dir = writeFixture("cross_" + label, fromCloud(*cloud));
    auto back = loadScene(device, dir.string());
    const size_t n = cloud->count();
    const bool same =
        back->count() == n && back->shDegree() == cloud->shDegree()
        && std::memcmp(cloud->positions().data(), back->positions().data(), n * 16) == 0
        && std::memcmp(cloud->scales().data(), back->scales().data(), n * 16) == 0
        && std::memcmp(cloud->rotations().data(), back->rotations().data(), n * 16) == 0
        && std::memcmp(cloud->opacities().data(), back->opacities().data(), n * 4) == 0
        && std::memcmp(cloud->shCoefficients().data(), back->shCoefficients().data(), n * 256) == 0;
    check(same,
          label + ": .ply -> cloud-v1 -> loadScene is bit-identical in every stream ("
              + std::to_string(n) + " Gaussians, SH degree " + std::to_string(back->shDegree())
              + ")");

    renderer.loadScene(plyPath.c_str());
    const SceneBounds bounds = renderer.sceneBounds();
    renderer.setCamera(makeCamera(bounds.focusCenter, bounds.focusRadius));
    renderer.forward();
    const std::vector<float> a = renderer.readImage();
    renderer.loadScene(dir.string().c_str());
    renderer.forward();
    const std::vector<float> b = renderer.readImage();
    float maxAbs = 0.0f;
    size_t differing = 0;
    double coverage = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        maxAbs = std::max(maxAbs, std::abs(a[i] - b[i]));
        differing += a[i] != b[i];
        if (i % 4 == 3)
            coverage += a[i];
    }
    coverage /= static_cast<double>(a.size() / 4);
    check(maxAbs == 0.0f && coverage > 0.05,
          label + ": .ply and cloud-v1 render bit-identically (max |diff| " + num(maxAbs) + ", "
              + std::to_string(differing) + " differing floats, mean coverage " + num(coverage)
              + ")");
}

// ---------------------------------------------------------------------------
// (d) writer: encode / write and the in-memory reader
// ---------------------------------------------------------------------------

std::string toText(const std::vector<uint8_t> & bytes)
{
    return std::string(bytes.begin(), bytes.end());
}

std::vector<uint8_t> toBytes(const std::string & text)
{
    return std::vector<uint8_t>(text.begin(), text.end());
}

// A canonical cloud: unit quaternions, positive scales, opacities in [0, 1],
// zero pad lanes, and SH coefficients above `degree` exactly zero (the unused
// slots of the canonical layout), so a reader must reproduce it bit-for-bit.
std::shared_ptr<GaussianCloud> makeCanonicalCloud(Device & device, size_t n, int degree,
                                                  uint32_t seed)
{
    auto cloud = std::make_shared<GaussianCloud>(device);
    cloud->initialize(n);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    auto & positions = cloud->positions();
    auto & scales = cloud->scales();
    auto & rotations = cloud->rotations();
    auto & opacities = cloud->opacities();
    auto & sh = cloud->shCoefficients();
    const int coefficients = (degree + 1) * (degree + 1);
    for (size_t i = 0; i < n; ++i)
    {
        positions[i] = glm::vec4(u(rng) * 3.0f, u(rng) * 3.0f, u(rng) * 3.0f, 0.0f);
        scales[i] = glm::vec4(0.5f + 0.75f * (u(rng) + 1.0f), 0.5f + 0.75f * (u(rng) + 1.0f),
                              0.5f + 0.75f * (u(rng) + 1.0f), 0.0f);
        rotations[i] = glm::normalize(glm::vec4(u(rng), u(rng), u(rng), u(rng)));
        opacities[i] = 0.1f + 0.4f * (u(rng) + 1.0f);
        for (int k = 0; k < kShCoefficientsPerGaussian; ++k)
        {
            sh[i * kShCoefficientsPerGaussian + k] =
                k < coefficients ? glm::vec4(u(rng), u(rng), u(rng), 0.0f) : glm::vec4(0.0f);
        }
    }
    cloud->setShDegree(static_cast<uint32_t>(degree));
    return cloud;
}

// Names of the canonical host streams that differ between two clouds (memcmp).
std::string differingStreams(GaussianCloud & a, GaussianCloud & b)
{
    if (a.count() != b.count() || a.shDegree() != b.shDegree())
    {
        return "count or shDegree";
    }
    const size_t n = a.count();
    std::string names;
    auto differs = [&](const char * name, const void * x, const void * y, size_t bytes)
    {
        if (std::memcmp(x, y, bytes) != 0)
        {
            names += (names.empty() ? "" : ", ") + std::string(name);
        }
    };
    differs("means", a.positions().data(), b.positions().data(), n * 16);
    differs("scales", a.scales().data(), b.scales().data(), n * 16);
    differs("rotations", a.rotations().data(), b.rotations().data(), n * 16);
    differs("opacities", a.opacities().data(), b.opacities().data(), n * 4);
    differs("shCoefficients", a.shCoefficients().data(), b.shCoefficients().data(),
            n * kShCoefficientsPerGaussian * 16);
    return names;
}

void testWriter(Device & device)
{
    std::printf("\n-- (d) cloud-v1 writer --\n");
    for (int degree : {0, 1, 3})
    {
        const std::string label = "degree " + std::to_string(degree);
        auto cloud = makeCanonicalCloud(device, 37, degree, 100 + static_cast<uint32_t>(degree));
        const FileMap files = CloudV1Writer::encode(*cloud);

        auto fromMemory = CloudV1Reader::read(device, files, label);
        const std::string diffMemory = differingStreams(*cloud, *fromMemory);
        check(diffMemory.empty(),
              "encode -> read(FileMap), " + label + ": every canonical host stream bit-exact"
                  + (diffMemory.empty() ? "" : " (differs: " + diffMemory + ")"));

        const fs::path dir = g_tempDir / ("written_degree_" + std::to_string(degree));
        CloudV1Writer::write(*cloud, dir.string());
        bool onDisk = true;
        for (const auto & [name, bytes] : files)
        {
            std::ifstream in(dir / name, std::ios::binary | std::ios::ate);
            std::vector<uint8_t> got(static_cast<size_t>(in.tellg()));
            in.seekg(0, std::ios::beg);
            in.read(reinterpret_cast<char *>(got.data()), static_cast<std::streamsize>(got.size()));
            onDisk = onDisk && got == bytes;
        }
        check(onDisk && fs::file_size(dir / "manifest.json") > 0,
              "write(), " + label + ": the files on disk are exactly the encode() output ("
                  + std::to_string(files.size()) + " files)");

        auto fromPath = CloudV1Reader::read(device, dir.string());
        const std::string diffPath = differingStreams(*cloud, *fromPath);
        check(diffPath.empty(),
              "write -> read(path), " + label + ": every canonical host stream bit-exact"
                  + (diffPath.empty() ? "" : " (differs: " + diffPath + ")"));

        // The manifest's top-level, stream and entry keys, in the reference order.
        // nlohmann dump(2) indents keys 2, 4 and 6 spaces deep.
        const std::string text = toText(files.at("manifest.json"));
        std::vector<std::string> top, streams, entries;
        // The key on `line` if it starts with exactly `indent` spaces and a quote.
        auto keyAt = [](const std::string & line, size_t indent) -> std::string
        {
            if (line.size() <= indent || line.compare(0, indent, std::string(indent, ' ')) != 0
                || line[indent] != '"')
            {
                return "";
            }
            const size_t close = line.find('"', indent + 1);
            if (close == std::string::npos || line.compare(close, 2, "\":") != 0)
            {
                return "";
            }
            return line.substr(indent + 1, close - indent - 1);
        };
        size_t begin = 0;
        while (begin < text.size())
        {
            const size_t end = text.find('\n', begin);
            const std::string line =
                text.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
            begin = end == std::string::npos ? text.size() : end + 1;
            if (!keyAt(line, 2).empty())
            {
                top.push_back(keyAt(line, 2));
            }
            if (!keyAt(line, 4).empty())
            {
                streams.push_back(keyAt(line, 4));
            }
            if (!keyAt(line, 6).empty())
            {
                entries.push_back(keyAt(line, 6));
            }
        }
        const std::vector<std::string> expectedTop = {"schema",         "count",     "coordinateSystem",
                                                      "quaternionOrder", "colorSpace", "shDegree",
                                                      "shLayout",       "streams"};
        std::vector<std::string> expectedStreams = {"means", "scales", "rotations", "opacities", "sh0"};
        if (degree > 0)
        {
            expectedStreams.push_back("sh_rest");
        }
        std::vector<std::string> expectedEntries;
        for (size_t i = 0; i < expectedStreams.size(); ++i)
        {
            for (const char * key : {"file", "shape", "scalarType", "elementStrideBytes", "byteLength"})
            {
                expectedEntries.push_back(key);
            }
        }
        check(text.find("\"schema\": \"sabercat.cloud-v1\"") != std::string::npos,
              "writer schema, " + label + ": the manifest carries the new schema id");
        check(top == expectedTop && streams == expectedStreams && entries == expectedEntries,
              "manifest keys, " + label + ": the reference top-level, stream and entry keys, in order ("
                  + std::to_string(streams.size()) + " streams)");
        check(files.count("manifest.json") == 1 && files.size() == expectedStreams.size() + 1
                  && files.count("sh_rest.f32le.bin") == (degree > 0 ? 1U : 0U),
              "file names, " + label + ": <stream>.f32le.bin per stream, keyed by file name only");
    }
}

// Rejections of the in-memory reader: a missing manifest, a missing stream file.
void expectMemoryRejected(Device & device, const FileMap & files, const std::string & needle,
                          const std::string & what)
{
    std::string message;
    try
    {
        CloudV1Reader::read(device, files, "memory fixture");
    }
    catch (const std::exception & e)
    {
        message = e.what();
    }
    check(!message.empty() && message.find(needle) != std::string::npos,
          "rejects " + what + (message.empty() ? " (ACCEPTED)" : " -- \"" + message.substr(0, message.find(": C:")) + "\""));
}

// ---------------------------------------------------------------------------
// (e) relight sidecar
// ---------------------------------------------------------------------------

// Relight vectors of `cloud` set to arbitrary values (non-unit w included, so a
// repack or normalize anywhere would show up).
void fillRelight(GaussianCloud & cloud, bool normals, bool others)
{
    const size_t n = cloud.count();
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> u(-2.0f, 2.0f);
    if (normals)
    {
        cloud.normals().resize(n);
        for (glm::vec4 & v : cloud.normals())
        {
            v = glm::vec4(u(rng), u(rng), u(rng), u(rng));
        }
    }
    if (others)
    {
        cloud.baseColorRoughness().resize(n);
        for (glm::vec4 & v : cloud.baseColorRoughness())
        {
            v = glm::vec4(u(rng), u(rng), u(rng), u(rng));
        }
        cloud.metallic().resize(n);
        for (float & v : cloud.metallic())
        {
            v = u(rng);
        }
        cloud.visibility().resize(n * kVisibilityCoefficients);
        for (float & v : cloud.visibility())
        {
            v = u(rng);
        }
    }
}

bool sameFloats(const std::vector<float> & a, const std::vector<float> & b)
{
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * 4) == 0);
}

bool sameVec4s(const std::vector<glm::vec4> & a, const std::vector<glm::vec4> & b)
{
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * 16) == 0);
}

void expectSidecarRejected(Device & device, const FileMap & files, size_t n, const std::string & needle,
                           const std::string & what)
{
    auto target = std::make_shared<GaussianCloud>(device);
    target->initialize(n);
    std::string message;
    try
    {
        RelightSidecar::decode(files, *target, "sidecar");
    }
    catch (const std::exception & e)
    {
        message = e.what();
    }
    const bool untouched = target->normals().empty() && target->metallic().empty()
                           && target->visibility().empty() && target->baseColorRoughness().empty();
    check(!message.empty() && message.find(needle) != std::string::npos && untouched,
          "rejects " + what + (message.empty() ? " (ACCEPTED)" : " -- \"" + message.substr(0, message.find(": C:")) + "\"")
              + ", cloud left untouched");
}

void testSidecar(Device & device)
{
    std::printf("\n-- (e) relight sidecar and in-memory rejections --\n");
    const size_t n = 23;
    auto source = makeCanonicalCloud(device, n, 1, 33);
    fillRelight(*source, true, true);
    const FileMap files = RelightSidecar::encode(*source);
    check(files.count("manifest.json") == 1 && files.size() == 5,
          "sidecar has the manifest and all four relight streams (" + std::to_string(files.size()) + " files)");

    auto target = std::make_shared<GaussianCloud>(device);
    target->initialize(n);
    RelightSidecar::decode(files, *target, "sidecar");
    check(sameVec4s(source->normals(), target->normals())
              && sameVec4s(source->baseColorRoughness(), target->baseColorRoughness())
              && sameFloats(source->metallic(), target->metallic())
              && sameFloats(source->visibility(), target->visibility()),
          "all four relight streams decode bit-exactly (memcmp)");

    // A subset: normals only. The others stay absent (empty).
    auto subset = makeCanonicalCloud(device, n, 1, 33);
    fillRelight(*subset, true, false);
    const FileMap subsetFiles = RelightSidecar::encode(*subset);
    check(subsetFiles.size() == 2 && subsetFiles.count("normals.f32le.bin") == 1,
          "normals-only sidecar holds just the manifest and normals");
    auto subsetTarget = std::make_shared<GaussianCloud>(device);
    subsetTarget->initialize(n);
    RelightSidecar::decode(subsetFiles, *subsetTarget, "sidecar");
    check(sameVec4s(subset->normals(), subsetTarget->normals()) && subsetTarget->metallic().empty()
              && subsetTarget->visibility().empty() && subsetTarget->baseColorRoughness().empty(),
          "normals-only subset decodes bit-exactly; absent streams stay empty");

    auto plain = makeCanonicalCloud(device, n, 1, 33);
    check(RelightSidecar::encode(*plain).empty(), "a cloud with no relight streams encodes to an empty map");

    // Corruptions. Each is a copy of the good sidecar with one thing changed.
    {
        FileMap bad = files;
        std::string manifest = toText(bad["manifest.json"]);
        replaceOnce(manifest, "\"count\": 23", "\"count\": 24");
        bad["manifest.json"] = toBytes(manifest);
        expectSidecarRejected(device, bad, n, "does not match", "a sidecar whose count disagrees with the cloud");
    }
    {
        FileMap bad = files;
        bad["normals.f32le.bin"].resize(bad["normals.f32le.bin"].size() - 40);
        expectSidecarRejected(device, bad, n, "bytes, but the manifest says", "a truncated normals stream");
    }
    {
        FileMap bad = files;
        std::string manifest = toText(bad["manifest.json"]);
        replaceOnce(manifest, "sabercat.relight-v1", "sabercat.relight-v2");
        bad["manifest.json"] = toBytes(manifest);
        expectSidecarRejected(device, bad, n, "schema", "a sidecar with another schema");
    }
    {
        FileMap bad = files;
        bad.erase("manifest.json");
        expectSidecarRejected(device, bad, n, "no 'manifest.json'", "a sidecar without its manifest");
    }

    // The in-memory cloud-v1 reader rejects a missing manifest and a missing stream.
    {
        const FileMap canonical = CloudV1Writer::encode(*plain);
        FileMap bad = canonical;
        bad.erase("means.f32le.bin");
        expectMemoryRejected(device, bad, "no file 'means.f32le.bin'", "a FileMap fixture missing a stream");
        bad = canonical;
        bad.erase("manifest.json");
        expectMemoryRejected(device, bad, "no 'manifest.json'", "a FileMap fixture without a manifest");
    }
}

void testRendering(Device & device, const std::string & assetPly)
{
    std::printf("\n-- (c) rendering through Renderer --\n");
    Renderer renderer(kWidth, kHeight);
    if (!renderer.initialize({}, true, SHADER_DIR))
    {
        check(false, "renderer initialize");
        return;
    }
    const std::string synthetic = (g_tempDir / "synthetic_deg3.ply").string();
    writeDegree3Ply(synthetic, 2000, 5);
    crossFormat(renderer, device, synthetic, "synthetic degree-3");
    if (fs::exists(assetPly))
        crossFormat(renderer, device, assetPly, fs::path(assetPly).stem().string());
    else
        std::printf("[SKIP] %s not found -- real-asset cross-format check not run\n",
                    assetPly.c_str());
    renderer.shutdown();
}

// Optional: a cloud-v1 fixture produced by an external writer.
void testExternal(Device & device, const std::string & dir)
{
    std::printf("\n-- external fixture: %s --\n", dir.c_str());
    try
    {
        auto cloud = loadScene(device, dir);
        const glm::vec4 & p = cloud->positions()[0];
        const glm::vec4 & dc = cloud->shCoefficients()[0];
        check(cloud->count() > 0,
              "external fixture loads: " + std::to_string(cloud->count()) + " Gaussians, SH degree "
                  + std::to_string(cloud->shDegree()) + ", mean[0] (" + num(p.x) + ", " + num(p.y)
                  + ", " + num(p.z) + "), opacity[0] " + num(cloud->opacities()[0]) + ", sh0[0] ("
                  + num(dc.x) + ", " + num(dc.y) + ", " + num(dc.z) + ")");
    }
    catch (const std::exception & e)
    {
        check(false, std::string("external fixture loads: ") + e.what());
    }
}

} // namespace

int main(int argc, char ** argv)
{
    const std::string external = argc > 1 ? argv[1] : "";
    const std::string assetPly = argc > 2 ? argv[2] : "assets/bonsai-7k.ply";
    g_tempDir = fs::temp_directory_path() / "sabercat_cloud_v1_reader_test";
    fs::remove_all(g_tempDir);
    fs::create_directories(g_tempDir);

    try
    {
        Device device;
        device.initialize({}, /*enableValidation=*/true);
        testDecode(device);
        testRejections(device);
        if (!external.empty())
            testExternal(device, external);
        testWriter(device);
        testSidecar(device);
        g_failures += runMillionCloudV1(device, g_tempDir);
        testRendering(device, assetPly);
        device.shutdown();
    }
    catch (const std::exception & e)
    {
        std::printf("[FAIL] unexpected exception: %s\n", e.what());
        ++g_failures;
    }

    fs::remove_all(g_tempDir);
    std::printf("\n%s (%d failure%s)\n",
                g_failures == 0 ? "cloud_v1_reader_test OK" : "cloud_v1_reader_test FAILED",
                g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}
