// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Further GLTFWriter and layer-transform checks (WP6):
//
//  (a) Refusals, each naming the layer and, where there is one, the Gaussian:
//      null cloud, zero Gaussians, a non-finite position or SH coefficient,
//      a negative scale, a rotation not unit length (per Gaussian, or the layer
//      transform), host streams not sized for count(). A refused write leaves no
//      file behind.
//  (b) Accepted on purpose, pinned so a change is noticed: a zero scale and two
//      layers with the same name (glTF names are not keys; both are written and
//      read back).
//  (c) Structural check of the .glb, independent of the reader: the header and
//      chunk lengths, the JSON parsed with nlohmann, and each bufferView and
//      accessor against the glTF 2.0 rules (alignment, byte lengths, counts,
//      min/max on POSITION, mesh and node references).
//  (d) layer_transform refusals (transformedCopy and mergeLayers): a scale <= 0,
//      a non-unit rotation at the project tolerance (1e-4 accepted,
//      2e-4 refused), a non-finite placement, a null cloud, relight streams not
//      sized for count(). A merge of one layer equals transformedCopy bit for bit.
//  (e) A 1,000,000-Gaussian degree-3 layer round-trips through GLTFWriter and
//      GltfReader (timed): positions, scales, opacities and SH bit-exact,
//      rotations within 8 ulp up to sign.

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "sabercat/device.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/gltf_reader.h"
#include "sabercat/gltf_writer.h"
#include "sabercat/layer_transform.h"
#include "sabercat/render_config.h"

using namespace sabercat;
namespace fs = std::filesystem;

namespace
{

int g_failures = 0;

void check(bool ok, const std::string & what)
{
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok)
    {
        ++g_failures;
    }
}

GaussianCloud & hostOf(const GaussianCloud & cloud)
{
    return const_cast<GaussianCloud &>(cloud);
}

uint32_t bitsOf(float value)
{
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof bits);
    return bits;
}

// Seeded random cloud at `degree`, w = 0 everywhere, unit rotations.
std::shared_ptr<GaussianCloud> makeCloud(Device & device, size_t count, uint32_t degree, uint32_t seed)
{
    auto cloud = std::make_shared<GaussianCloud>(device);
    cloud->initialize(count);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    std::uniform_real_distribution<float> opacity(0.05f, 0.95f);
    std::uniform_real_distribution<float> scale(0.01f, 0.3f);
    const size_t kept = static_cast<size_t>((degree + 1) * (degree + 1));
    for (size_t i = 0; i < count; ++i)
    {
        cloud->positions()[i] = glm::vec4(2.0f * unit(rng), 2.0f * unit(rng), 2.0f * unit(rng), 0.0f);
        cloud->scales()[i] = glm::vec4(scale(rng), scale(rng), scale(rng), 0.0f);
        cloud->rotations()[i] = glm::normalize(glm::vec4(unit(rng), unit(rng), unit(rng), unit(rng)));
        cloud->opacities()[i] = opacity(rng);
        for (size_t k = 0; k < kShCoefficientsPerGaussian; ++k)
        {
            cloud->shCoefficients()[i * kShCoefficientsPerGaussian + k] =
                k < kept ? glm::vec4(0.5f * unit(rng), 0.5f * unit(rng), 0.5f * unit(rng), 0.0f)
                         : glm::vec4(0.0f);
        }
    }
    cloud->setShDegree(degree);
    cloud->computeBounds();
    return cloud;
}

// Runs `action`, which must throw an exception whose message contains `needle`,
// and checks that `path` was not created.
void expectRefusal(const std::string & what, const std::string & needle, const fs::path & path,
                   const std::function<void()> & action)
{
    fs::remove(path);
    try
    {
        action();
        check(false, what + " (no exception)");
    }
    catch (const std::exception & error)
    {
        const std::string message = error.what();
        check(message.find(needle) != std::string::npos,
              what + " (" + message + ")");
    }
    check(!fs::exists(path) && !fs::exists(path.string() + ".tmp"),
          what + ": no file left behind");
}

// ---- (a) and (b) writer refusals and accepted edge cases -------------------

void testWriterRefusals(Device & device, const fs::path & dir)
{
    const auto good = makeCloud(device, 10, 1, 11);
    const fs::path out = dir / "refused.glb";
    const auto one = [&](const GaussianCloud * cloud, const std::string & name)
    {
        return std::vector<GLTFExportLayer>{GLTFExportLayer{cloud, LayerTransform{}, name}};
    };

    expectRefusal("null cloud refused, layer named", "layer 0 ('nul') has no cloud", out,
                  [&] { GLTFWriter::writeGLB(one(nullptr, "nul"), out.string()); });

    const auto empty = makeCloud(device, 1, 0, 12);
    hostOf(*empty).initialize(0);
    expectRefusal("zero Gaussians refused, layer named", "layer 0 ('empty') has no Gaussians", out,
                  [&] { GLTFWriter::writeGLB(one(empty.get(), "empty"), out.string()); });

    auto badPosition = makeCloud(device, 10, 1, 13);
    hostOf(*badPosition).positions()[3].y = std::numeric_limits<float>::quiet_NaN();
    expectRefusal("NaN position refused, Gaussian named", "position 3 is not finite", out,
                  [&] { GLTFWriter::writeGLB(one(badPosition.get(), "p"), out.string()); });

    auto badSh = makeCloud(device, 10, 3, 14);
    // Gaussian 7, coefficient 2 (the third RGB triple in band 0..1).
    hostOf(*badSh).shCoefficients()[7 * kShCoefficientsPerGaussian + 2].z = std::numeric_limits<float>::infinity();
    expectRefusal("infinite SH coefficient refused, Gaussian named", "SH coefficient 2 of Gaussian 7 is not finite",
                  out, [&] { GLTFWriter::writeGLB(one(badSh.get(), "sh"), out.string()); });

    auto negativeScale = makeCloud(device, 10, 0, 15);
    hostOf(*negativeScale).scales()[5].x = -1.0f;
    expectRefusal("negative scale refused, Gaussian named", "scale 5 is not finite and non-negative", out,
                  [&] { GLTFWriter::writeGLB(one(negativeScale.get(), "neg"), out.string()); });

    auto zeroRotation = makeCloud(device, 10, 0, 16);
    hostOf(*zeroRotation).rotations()[3] = glm::vec4(0.0f);
    expectRefusal("zero-length rotation refused, Gaussian named", "Gaussian 3 rotation is not unit length", out,
                  [&] { GLTFWriter::writeGLB(one(zeroRotation.get(), "rot"), out.string()); });

    auto shortPositions = makeCloud(device, 10, 0, 17);
    hostOf(*shortPositions).positions().resize(9);
    expectRefusal("positions shorter than count refused, layer named",
                  "layer 0 ('short'): host streams are not sized for 10 Gaussians", out,
                  [&] { GLTFWriter::writeGLB(one(shortPositions.get(), "short"), out.string()); });

    // A bad second layer is named by its index, and the first layer is not written.
    expectRefusal("bad layer 1 named by index", "layer 1 ('second')", out,
                  [&] {
                      GLTFWriter::writeGLB({GLTFExportLayer{good.get(), LayerTransform{}, "first"},
                                            GLTFExportLayer{negativeScale.get(), LayerTransform{}, "second"}},
                                           out.string());
                  });

    // Accepted on purpose. A zero scale is representable in a .glb (the reader
    // accepts it), so the writer accepts it too. Two layers with one name are two
    // meshes and nodes, both read back.
    auto zeroScale = makeCloud(device, 10, 0, 18);
    hostOf(*zeroScale).scales()[2] = glm::vec4(0.0f);
    GLTFWriter::writeGLB(one(zeroScale.get(), "zero"), (dir / "zero_scale.glb").string());
    check(fs::exists(dir / "zero_scale.glb"), "zero scale is accepted (the reader accepts it too)");

    GLTFWriter::writeGLB({GLTFExportLayer{good.get(), LayerTransform{}, "dup"},
                          GLTFExportLayer{good.get(), LayerTransform{}, "dup"}},
                         (dir / "dup.glb").string());
    const auto dup = GltfReader::read(device, (dir / "dup.glb").string());
    check(dup->count() == 2 * good->count(), "two layers with one name are both written and read back");
}

// ---- (c) structural check of the .glb ---------------------------------------

std::vector<uint8_t> readAll(const fs::path & path)
{
    std::ifstream file(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

uint32_t readU32(const std::vector<uint8_t> & bytes, size_t offset)
{
    uint32_t value = 0;
    std::memcpy(&value, bytes.data() + offset, 4);
    return value;
}

// Returns the number of violations found (0 for a valid file). Each violation is
// printed.
int structuralViolations(const fs::path & path)
{
    using Json = nlohmann::json;
    int violations = 0;
    const auto bad = [&](const std::string & what)
    {
        std::printf("  structure: %s\n", what.c_str());
        ++violations;
    };

    const std::vector<uint8_t> file = readAll(path);
    if (file.size() < 28)
    {
        bad("file shorter than the 12-byte header plus two chunk headers");
        return violations;
    }
    if (readU32(file, 0) != 0x46546C67 || readU32(file, 4) != 2)
    {
        bad("header is not glTF magic 2");
    }
    if (readU32(file, 8) != file.size())
    {
        bad("header length " + std::to_string(readU32(file, 8)) + " != file size " + std::to_string(file.size()));
    }
    const uint32_t jsonLength = readU32(file, 12);
    if (readU32(file, 16) != 0x4E4F534A)
    {
        bad("first chunk is not JSON");
    }
    if (jsonLength % 4 != 0 || 20 + jsonLength + 8 > file.size())
    {
        bad("JSON chunk length is not a multiple of 4 or runs past the file");
        return violations;
    }
    const size_t binHeader = 20 + jsonLength;
    const uint32_t binLength = readU32(file, binHeader);
    if (readU32(file, binHeader + 4) != 0x004E4942)
    {
        bad("second chunk is not BIN");
    }
    if (binLength % 4 != 0 || binHeader + 8 + binLength != file.size())
    {
        bad("BIN chunk length is not a multiple of 4 or does not end the file");
        return violations;
    }
    const size_t binStart = binHeader + 8;

    const Json root = Json::parse(std::string(file.begin() + 20, file.begin() + 20 + jsonLength));
    if (root.at("asset").at("version") != "2.0")
    {
        bad("asset.version is not 2.0");
    }
    const auto & used = root.at("extensionsUsed");
    if (std::find(used.begin(), used.end(), "KHR_gaussian_splatting") == used.end())
    {
        bad("extensionsUsed does not list KHR_gaussian_splatting");
    }
    if (root.at("buffers").size() != 1 || root.at("buffers")[0].at("byteLength").get<uint64_t>() != binLength)
    {
        bad("buffers[0].byteLength does not match the BIN chunk");
    }

    const auto & views = root.at("bufferViews");
    for (size_t v = 0; v < views.size(); ++v)
    {
        const auto & view = views[v];
        const uint64_t offset = view.value("byteOffset", uint64_t(0));
        const uint64_t length = view.at("byteLength").get<uint64_t>();
        if (view.value("buffer", 0) != 0)
        {
            bad("bufferView " + std::to_string(v) + " does not reference buffer 0");
        }
        if (offset % 4 != 0)
        {
            bad("bufferView " + std::to_string(v) + " byteOffset is not 4-aligned");
        }
        if (length == 0 || length % 4 != 0)
        {
            bad("bufferView " + std::to_string(v) + " byteLength is zero or not a multiple of 4");
        }
        if (offset + length > binLength)
        {
            bad("bufferView " + std::to_string(v) + " runs past the BIN chunk");
        }
    }

    const auto & accessors = root.at("accessors");
    for (size_t a = 0; a < accessors.size(); ++a)
    {
        const auto & acc = accessors[a];
        const std::string label = "accessor " + std::to_string(a);
        if (acc.at("componentType").get<int>() != 5126)
        {
            bad(label + " is not float32");
        }
        const std::string type = acc.at("type").get<std::string>();
        const int components = type == "SCALAR" ? 1 : type == "VEC3" ? 3 : type == "VEC4" ? 4 : -1;
        if (components < 0)
        {
            bad(label + " has an unexpected type " + type);
            continue;
        }
        const uint64_t count = acc.at("count").get<uint64_t>();
        if (count < 1)
        {
            bad(label + " has count 0 (glTF requires at least one element)");
        }
        const size_t viewIndex = acc.at("bufferView").get<size_t>();
        if (viewIndex >= views.size())
        {
            bad(label + " references a missing bufferView");
            continue;
        }
        const uint64_t viewLength = views[viewIndex].at("byteLength").get<uint64_t>();
        if (acc.value("byteOffset", uint64_t(0)) != 0 || count * components * 4 != viewLength)
        {
            bad(label + " byte length does not match count * components * 4");
        }
        if (acc.contains("min") || acc.contains("max"))
        {
            if (acc.at("min").size() != static_cast<size_t>(components) || acc.at("max").size() != static_cast<size_t>(components))
            {
                bad(label + " min/max length is not " + std::to_string(components));
            }
            else
            {
                for (int c = 0; c < components; ++c)
                {
                    if (acc.at("min")[c].get<double>() > acc.at("max")[c].get<double>())
                    {
                        bad(label + " min exceeds max");
                    }
                }
            }
        }
        // Each float of the accessor is finite (the reader would reject it).
        const uint64_t base = views[viewIndex].value("byteOffset", uint64_t(0));
        for (uint64_t i = 0; i < count * components; ++i)
        {
            float value = 0.0f;
            std::memcpy(&value, file.data() + binStart + base + 4 * i, 4);
            if (!std::isfinite(value))
            {
                bad(label + " holds a non-finite value");
                break;
            }
        }
    }

    const auto & meshes = root.at("meshes");
    const auto & nodes = root.at("nodes");
    if (nodes.size() != meshes.size())
    {
        bad("node count differs from mesh count");
    }
    for (const auto & mesh : meshes)
    {
        const auto & primitive = mesh.at("primitives")[0];
        if (primitive.at("mode").get<int>() != 0)
        {
            bad("primitive mode is not POINTS");
        }
        size_t expectedCount = 0;
        for (const auto & [name, index] : primitive.at("attributes").items())
        {
            const size_t count = accessors[index.get<size_t>()].at("count").get<size_t>();
            if (expectedCount == 0)
            {
                expectedCount = count;
            }
            else if (count != expectedCount)
            {
                bad("attribute " + name + " count differs from the other attributes");
            }
        }
        if (!primitive.at("attributes").contains("POSITION"))
        {
            bad("primitive has no POSITION");
        }
    }
    for (const auto & node : nodes)
    {
        if (node.at("mesh").get<size_t>() >= meshes.size())
        {
            bad("node references a missing mesh");
        }
    }
    return violations;
}

void testStructure(Device & device, const fs::path & dir)
{
    // Two layers with different SH degrees and odd counts, so the padding and the
    // alignment rules are exercised, plus a layer with a transform.
    const auto a = makeCloud(device, 13, 1, 31);
    const auto b = makeCloud(device, 7, 3, 32);
    LayerTransform placed;
    placed.translation = glm::vec3(1.0f, -2.0f, 0.5f);
    placed.scale = 2.0f;
    const fs::path path = dir / "structure.glb";
    GLTFWriter::writeGLB({GLTFExportLayer{a.get(), LayerTransform{}, "a"},
                          GLTFExportLayer{b.get(), placed, "b layer"}},
                         path.string());
    const int violations = structuralViolations(path);
    check(violations == 0, "GLB structure satisfies the glTF 2.0 rules (" + std::to_string(violations) + " violations)");
}

// ---- (d) layer_transform refusals ------------------------------------------

void testLayerTransform(Device & device, const fs::path & dir)
{
    const auto source = makeCloud(device, 50, 1, 41);
    const auto one = [&](LayerTransform t)
    {
        try
        {
            transformedCopy(device, *source, t);
            return std::string();
        }
        catch (const std::exception & error)
        {
            return std::string(error.what());
        }
    };
    const auto mergeOne = [&](const GaussianCloud * cloud, LayerTransform t)
    {
        try
        {
            mergeLayers(device, {{cloud, t}});
            return std::string();
        }
        catch (const std::exception & error)
        {
            return std::string(error.what());
        }
    };

    LayerTransform t;
    t.scale = 0.0f;
    check(one(t).find("scale must be > 0") != std::string::npos, "transformedCopy refuses scale 0");
    t.scale = -2.0f;
    check(one(t).find("scale must be > 0") != std::string::npos, "transformedCopy refuses negative scale");
    t = LayerTransform{};
    t.rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f) * 1.001f; // |q| = 1.001
    check(one(t).find("rotation is not unit length") != std::string::npos,
          "transformedCopy refuses a rotation 1e-3 from unit length");
    t = LayerTransform{};
    t.rotation = glm::quat(1.0f + 5e-5f, 0.0f, 0.0f, 0.0f); // within the project's 1e-4
    check(one(t).empty(), "transformedCopy accepts a rotation 5e-5 from unit length (project tolerance 1e-4)");
    t = LayerTransform{};
    t.rotation = glm::quat(1.0f + 2e-4f, 0.0f, 0.0f, 0.0f); // beyond it
    check(one(t).find("rotation is not unit length") != std::string::npos,
          "transformedCopy refuses a rotation 2e-4 from unit length");
    t = LayerTransform{};
    t.translation.y = std::numeric_limits<float>::infinity();
    check(one(t).find("non-finite value") != std::string::npos, "transformedCopy refuses an infinite translation");
    t = LayerTransform{};
    t.scale = std::numeric_limits<float>::quiet_NaN();
    check(one(t).find("non-finite value") != std::string::npos, "transformedCopy refuses a NaN scale");

    t = LayerTransform{};
    t.scale = 0.0f;
    check(mergeOne(source.get(), t).find("mergeLayers: layer 0 transform scale must be > 0") != std::string::npos,
          "mergeLayers refuses scale 0, naming layer 0");
    try
    {
        mergeLayers(device, {{source.get(), LayerTransform{}}, {nullptr, LayerTransform{}}});
        check(false, "mergeLayers refuses a null cloud in layer 1 (no exception)");
    }
    catch (const std::exception & error)
    {
        check(std::string(error.what()).find("layer 1 has no cloud") != std::string::npos,
              std::string("mergeLayers refuses a null cloud, naming layer 1 (") + error.what() + ")");
    }
    t = LayerTransform{};
    t.rotation = glm::quat(1.0f, 0.5f, 0.0f, 0.0f); // |q| = 1.118
    check(mergeOne(source.get(), t).find("layer 0 transform rotation is not unit length") != std::string::npos,
          "mergeLayers refuses a non-unit rotation, naming the layer");

    // Relight streams not sized for count(): refused by both operations. (A
    // stream present in only some layers is dropped with a warning instead, so
    // the merge's second layer carries normals too.)
    auto mismatched = makeCloud(device, 50, 1, 42);
    hostOf(*mismatched).normals().resize(49);
    try
    {
        transformedCopy(device, *mismatched, LayerTransform{});
        check(false, "transformedCopy refuses a normals stream of the wrong size (no exception)");
    }
    catch (const std::exception & error)
    {
        check(std::string(error.what()).find("a relight stream is not sized for 50 Gaussians") != std::string::npos,
              std::string("transformedCopy refuses a normals stream of the wrong size (") + error.what() + ")");
    }
    auto withNormals = makeCloud(device, 43, 1, 43);
    withNormals->normals().assign(43, glm::vec4(0.0f, 0.0f, 1.0f, 0.0f));
    try
    {
        mergeLayers(device, {{withNormals.get(), LayerTransform{}}, {mismatched.get(), LayerTransform{}}});
        check(false, "mergeLayers refuses a normals stream of the wrong size (no exception)");
    }
    catch (const std::exception & error)
    {
        check(std::string(error.what()).find("a relight stream is not sized") != std::string::npos,
              std::string("mergeLayers refuses a normals stream of the wrong size (") + error.what() + ")");
    }

    // A merge of one layer is the same cloud as transformedCopy, bit for bit.
    LayerTransform placed;
    placed.rotation = glm::normalize(glm::quat(0.9f, 0.1f, -0.3f, 0.2f));
    placed.translation = glm::vec3(3.0f, -1.0f, 2.0f);
    placed.scale = 1.7f;
    const auto copy = transformedCopy(device, *source, placed);
    const auto merged = mergeLayers(device, {{source.get(), placed}});
    bool same = copy->count() == merged->count() && copy->shDegree() == merged->shDegree();
    const auto eq = [](const std::vector<glm::vec4> & x, const std::vector<glm::vec4> & y)
    {
        return x.size() == y.size() && (x.empty() || std::memcmp(x.data(), y.data(), x.size() * sizeof(glm::vec4)) == 0);
    };
    same = same && eq(hostOf(*copy).positions(), hostOf(*merged).positions())
           && eq(hostOf(*copy).scales(), hostOf(*merged).scales())
           && eq(hostOf(*copy).rotations(), hostOf(*merged).rotations())
           && eq(hostOf(*copy).shCoefficients(), hostOf(*merged).shCoefficients());
    check(same, "mergeLayers of one layer equals transformedCopy bit for bit");
    (void)dir;
}

// ---- (e) one million Gaussians through GLB -----------------------------------

void testMillion(Device & device, const fs::path & dir)
{
    const size_t count = 1000000;
    const auto source = makeCloud(device, count, 3, 51);
    const fs::path path = dir / "million.glb";
    const auto writeStart = std::chrono::steady_clock::now();
    GLTFWriter::writeGLB({GLTFExportLayer{source.get(), LayerTransform{}, "million"}}, path.string());
    const double writeMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - writeStart).count();
    const auto readStart = std::chrono::steady_clock::now();
    const auto back = GltfReader::read(device, path.string());
    const double readMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - readStart).count();
    std::printf("1M GLB: write %.0f ms, read %.0f ms, file %.1f MB\n", writeMs, readMs,
                static_cast<double>(fs::file_size(path)) / 1e6);

    check(back->count() == count && back->shDegree() == 3, "1M GLB: count and shDegree round trip");
    size_t mismatches = 0;
    for (size_t i = 0; i < count; ++i)
    {
        const glm::vec4 & a = source->positions()[i];
        const glm::vec4 & b = hostOf(*back).positions()[i];
        if (bitsOf(a.x) != bitsOf(b.x) || bitsOf(a.y) != bitsOf(b.y) || bitsOf(a.z) != bitsOf(b.z))
        {
            ++mismatches;
        }
        if (bitsOf(source->opacities()[i]) != bitsOf(hostOf(*back).opacities()[i]))
        {
            ++mismatches;
        }
        for (size_t c = 0; c < 3; ++c)
        {
            if (bitsOf(source->scales()[i][c]) != bitsOf(hostOf(*back).scales()[i][c]))
            {
                ++mismatches;
            }
        }
        for (size_t k = 0; k < kShCoefficientsPerGaussian; ++k)
        {
            const size_t o = i * kShCoefficientsPerGaussian + k;
            const glm::vec4 & s = source->shCoefficients()[o];
            const glm::vec4 & t = hostOf(*back).shCoefficients()[o];
            if (bitsOf(s.x) != bitsOf(t.x) || bitsOf(s.y) != bitsOf(t.y) || bitsOf(s.z) != bitsOf(t.z))
            {
                ++mismatches;
            }
        }
    }
    check(mismatches == 0, "1M GLB: positions, scales, opacities and SH bit-exact (" + std::to_string(mismatches) + " mismatches)");

    // Rotations: the reader passes a quaternion through when its squared length is
    // within 1e-6 of one, and renormalizes it otherwise. Random unit quaternions
    // sit within a few ulp, so 8 ulp up to sign is the bound.
    double worstUlp = 0.0;
    for (size_t i = 0; i < count; ++i)
    {
        const glm::vec4 & qa = source->rotations()[i];
        const glm::vec4 & qb = hostOf(*back).rotations()[i];
        const glm::vec4 diff = qa - qb;
        const glm::vec4 sum = qa + qb;
        const double direct = std::max({std::fabs(static_cast<double>(diff.x)), std::fabs(static_cast<double>(diff.y)),
                                        std::fabs(static_cast<double>(diff.z)), std::fabs(static_cast<double>(diff.w))});
        const double flipped = std::max({std::fabs(static_cast<double>(sum.x)), std::fabs(static_cast<double>(sum.y)),
                                         std::fabs(static_cast<double>(sum.z)), std::fabs(static_cast<double>(sum.w))});
        worstUlp = std::max(worstUlp, std::min(direct, flipped) / static_cast<double>(FLT_EPSILON));
    }
    std::printf("1M GLB worst rotation error %.2f ulp (bound 8)\n", worstUlp);
    check(worstUlp <= 8.0, "1M GLB: rotations within 8 ulp up to sign");
    fs::remove(path);
}

} // namespace

// Entry point called by gltf_writer_test's main. Returns the failure count.
int runExtraChecks(Device & device, const fs::path & dir)
{
    g_failures = 0;
    testWriterRefusals(device, dir);
    testStructure(device, dir);
    testLayerTransform(device, dir);
    testMillion(device, dir);
    std::printf("extra GLTF and layer-transform checks: %s (%d failure%s)\n",
                g_failures == 0 ? "OK" : "FAILED", g_failures, g_failures == 1 ? "" : "s");
    return g_failures;
}
