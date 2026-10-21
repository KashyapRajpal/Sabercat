// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Headless test for GLTFWriter (KHR_gaussian_splatting .glb export) and the
// host-side layer transform operations (transformedCopy, mergeLayers).
//
//  (a) round trip: a single identity layer at SH degrees 0..3 written with
//      GLTFWriter and read back with GltfReader must match bit-for-bit.
//  (b) two layers (different SH degrees, one with rotation + translation +
//      uniform scale): the reader's merged cloud vs mergeLayers() of the same
//      inputs, worst error per stream.
//  (c) transformedCopy(identity) is bit-exact; transformedCopy(T2) after
//      transformedCopy(T1) equals transformedCopy(T2 * T1); the SH rotation
//      evaluated at R*d equals the original at d.
//  (d) mergeLayers keeps a relight stream present in every layer and drops one
//      present in only some; kept normals are rotated.
//  (e) the writer and the merge refuse what they must.
//
// Validation messages are logged as "[validation]"; the run must have none.

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "sabercat/device.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/gltf_reader.h"
#include "sabercat/gltf_writer.h"
#include "sabercat/layer_transform.h"
#include "sabercat/render_config.h"

using namespace sabercat;

// extra_checks.cpp: writer refusals, GLB structure, layer-transform refusals, 1M round trip.
int runExtraChecks(Device & device, const std::filesystem::path & dir);

namespace
{

int g_failures = 0;

// Tolerance for the composed-path comparisons (merge vs reader, transform
// composition, normals, SH rotation). The data are float32 and the arithmetic in
// between is double, so the only error is the final float rounding of a value
// of magnitude O(1..10): a few float ULPs (1 ULP near 4 is 4.8e-7). 1e-5 leaves a
// 20x margin over that and still catches a wrong sign, axis or order, which
// produce errors of order 1e-1 or more.
constexpr double kTolerance = 1e-5;

void check(bool ok, const std::string & what)
{
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok)
    {
        ++g_failures;
    }
}

std::filesystem::path g_tempDir;

// The host accessors are non-const only; the test reads them through this.
GaussianCloud & hostOf(const GaussianCloud & cloud)
{
    return const_cast<GaussianCloud &>(cloud);
}

glm::quat randomRotation(std::mt19937 & rng)
{
    std::normal_distribution<float> normal(0.0f, 1.0f);
    return glm::normalize(glm::quat(normal(rng), normal(rng), normal(rng), normal(rng)));
}

glm::vec3 randomUnit(std::mt19937 & rng)
{
    std::normal_distribution<float> normal(0.0f, 1.0f);
    return glm::normalize(glm::vec3(normal(rng), normal(rng), normal(rng)));
}

// A cloud of `count` random Gaussians at `degree`, with w = 0 in every vec4 so
// whole-vector comparisons are meaningful. `relight` adds every relight stream.
std::shared_ptr<GaussianCloud> makeCloud(Device & device, size_t count, uint32_t degree,
                                         uint32_t seed, bool relight = false)
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
        const glm::quat q = randomRotation(rng);
        cloud->rotations()[i] = glm::vec4(q.x, q.y, q.z, q.w);
        cloud->opacities()[i] = opacity(rng);
        for (size_t k = 0; k < kShCoefficientsPerGaussian; ++k)
        {
            cloud->shCoefficients()[i * kShCoefficientsPerGaussian + k] =
                k < kept ? glm::vec4(0.5f * unit(rng), 0.5f * unit(rng), 0.5f * unit(rng), 0.0f)
                         : glm::vec4(0.0f);
        }
    }
    cloud->setShDegree(degree);
    cloud->setAntialiased(false);
    if (relight)
    {
        cloud->normals().resize(count);
        cloud->baseColorRoughness().resize(count);
        cloud->metallic().resize(count);
        cloud->visibility().resize(count * kVisibilityCoefficients);
        for (size_t i = 0; i < count; ++i)
        {
            const glm::vec3 n = randomUnit(rng);
            cloud->normals()[i] = glm::vec4(n, 0.0f);
            cloud->baseColorRoughness()[i] = glm::vec4(0.5f * (unit(rng) + 1.0f),
                                                       0.5f * (unit(rng) + 1.0f),
                                                       0.5f * (unit(rng) + 1.0f),
                                                       0.5f * (unit(rng) + 1.0f));
            cloud->metallic()[i] = 0.5f * (unit(rng) + 1.0f);
        }
        for (float & v : cloud->visibility())
        {
            v = 0.5f * (unit(rng) + 1.0f);
        }
    }
    cloud->computeBounds();
    return cloud;
}

// Largest absolute component difference between two vec4 streams.
double worstVec4(const std::vector<glm::vec4> & a, const std::vector<glm::vec4> & b)
{
    if (a.size() != b.size())
    {
        return std::numeric_limits<double>::infinity();
    }
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        for (int c = 0; c < 4; ++c)
        {
            worst = std::max(worst, std::abs(static_cast<double>(a[i][c]) - b[i][c]));
        }
    }
    return worst;
}

// Quaternions q and -q are the same rotation, so compare modulo sign.
double worstRotation(const std::vector<glm::vec4> & a, const std::vector<glm::vec4> & b)
{
    if (a.size() != b.size())
    {
        return std::numeric_limits<double>::infinity();
    }
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        double sameSign = 0.0;
        double flipped = 0.0;
        for (int c = 0; c < 4; ++c)
        {
            sameSign = std::max(sameSign, std::abs(static_cast<double>(a[i][c]) - b[i][c]));
            flipped = std::max(flipped, std::abs(static_cast<double>(a[i][c]) + b[i][c]));
        }
        worst = std::max(worst, std::min(sameSign, flipped));
    }
    return worst;
}

double worstFloat(const std::vector<float> & a, const std::vector<float> & b)
{
    if (a.size() != b.size())
    {
        return std::numeric_limits<double>::infinity();
    }
    double worst = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        worst = std::max(worst, std::abs(static_cast<double>(a[i]) - b[i]));
    }
    return worst;
}

// Bit-for-bit equality of two float/vec4 streams (the same elements, same bytes).
template <typename T> bool bitEqual(const std::vector<T> & a, const std::vector<T> & b)
{
    return a.size() == b.size()
           && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

// Every host stream of two clouds, the same bytes.
bool sameBits(const GaussianCloud & a, const GaussianCloud & b)
{
    GaussianCloud & x = hostOf(a);
    GaussianCloud & y = hostOf(b);
    return bitEqual(x.positions(), y.positions()) && bitEqual(x.scales(), y.scales())
           && bitEqual(x.rotations(), y.rotations())
           && bitEqual(x.shCoefficients(), y.shCoefficients())
           && bitEqual(x.opacities(), y.opacities()) && bitEqual(x.times(), y.times())
           && bitEqual(x.normals(), y.normals())
           && bitEqual(x.baseColorRoughness(), y.baseColorRoughness())
           && bitEqual(x.metallic(), y.metallic()) && bitEqual(x.visibility(), y.visibility())
           && x.shDegree() == y.shDegree() && x.antialiased() == y.antialiased();
}

// Writes the layers, reads the .glb back with GltfReader.
std::shared_ptr<GaussianCloud> roundTrip(Device & device,
                                         const std::vector<GLTFExportLayer> & layers,
                                         const std::string & name)
{
    const std::string path = (g_tempDir / name).string();
    GLTFWriter::writeGLB(layers, path);
    check(!std::filesystem::exists(path + ".tmp"), name + ": no temporary file left behind");
    return GltfReader::read(device, path);
}

// Test (a): one identity layer at each SH degree, read back bit-for-bit.
void testRoundTrip(Device & device)
{
    std::printf("\n-- (a) single identity layer round trip, SH degree 0..3 --\n");
    for (uint32_t degree = 0; degree <= 3; ++degree)
    {
        auto source = makeCloud(device, 2000, degree, 100 + degree);
        const auto read = roundTrip(device,
                                    {GLTFExportLayer{source.get(), LayerTransform{}, "solo"}},
                                    "solo_" + std::to_string(degree) + ".glb");
        const std::string label = "degree " + std::to_string(degree) + ": ";
        check(read->count() == source->count(), label + "count matches");
        check(read->shDegree() == degree, label + "reader reports shDegree " + std::to_string(degree));
        GaussianCloud & r = hostOf(*read);
        GaussianCloud & s = hostOf(*source);
        check(bitEqual(r.positions(), s.positions()), label + "positions bit-exact");
        check(bitEqual(r.scales(), s.scales()), label + "scales bit-exact");
        check(bitEqual(r.rotations(), s.rotations()), label + "rotations bit-exact");
        check(bitEqual(r.opacities(), s.opacities()), label + "opacities bit-exact");
        check(bitEqual(r.shCoefficients(), s.shCoefficients()), label + "SH bit-exact");
    }
}

// Test (b): a degree-1 identity layer plus a degree-3 layer with a rotation,
// translation and uniform scale. The reader merges node transforms; mergeLayers
// bakes the same transforms. Both run the same arithmetic, so the error is the
// float rounding of a shared double computation.
void testMergeAgainstReader(Device & device)
{
    std::printf("\n-- (b) two-layer reader vs mergeLayers --\n");
    auto a = makeCloud(device, 5000, 1, 201);
    auto b = makeCloud(device, 7000, 3, 202);

    LayerTransform identity;
    LayerTransform placed;
    placed.rotation = glm::angleAxis(0.7f, glm::normalize(glm::vec3(1.0f, 2.0f, 3.0f)));
    placed.translation = glm::vec3(1.5f, -0.5f, 2.0f);
    placed.scale = 2.5f;

    const auto read = roundTrip(
        device,
        {GLTFExportLayer{a.get(), identity, "A"}, GLTFExportLayer{b.get(), placed, "B"}},
        "two_layers.glb");
    const auto merged = mergeLayers(device, {{a.get(), identity}, {b.get(), placed}});

    GaussianCloud & r = hostOf(*read);
    GaussianCloud & m = hostOf(*merged);
    check(read->count() == merged->count() && merged->count() == 12000,
          "merged count matches the reader (12000)");
    check(read->shDegree() == 3 && merged->shDegree() == 3, "merged shDegree is 3 on both sides");

    const double positions = worstVec4(r.positions(), m.positions());
    const double scales = worstVec4(r.scales(), m.scales());
    const double rotations = worstVec4(r.rotations(), m.rotations());
    const double opacities = worstFloat(r.opacities(), m.opacities());
    const double sh = worstVec4(r.shCoefficients(), m.shCoefficients());
    std::printf("  worst |reader - merge|: positions %.3e, scales %.3e, rotations %.3e,\n"
                "                          opacities %.3e, SH %.3e\n",
                positions, scales, rotations, opacities, sh);
    check(positions <= kTolerance, "positions agree within tolerance");
    check(scales <= kTolerance, "scales agree within tolerance");
    check(rotations <= kTolerance, "rotations agree within tolerance");
    check(opacities <= kTolerance, "opacities agree within tolerance");
    check(sh <= kTolerance, "SH agree within tolerance");

    // The identity layer occupies the first 5000 merged rows: bit-exact.
    std::vector<glm::vec4> aPositions(hostOf(*a).positions());
    check(std::memcmp(r.positions().data(), aPositions.data(), aPositions.size() * sizeof(glm::vec4))
              == 0,
          "identity layer rows are bit-exact through the reader");
}

// The real SH basis up to degree 3 in the 3DGS convention (Condon-Shortley phase,
// m ascending), written out here rather than taken from the library, so the SH
// rotation check below cross-checks the library's rotation against it.
void shBasis16(double x, double y, double z, double * out)
{
    constexpr double C0 = 0.28209479177387814;
    constexpr double C1 = 0.4886025119029199;
    constexpr double C2[5] = {1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
                              -1.0925484305920792, 0.5462742152960396};
    constexpr double C3[7] = {-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
                              0.3731763325901154, -0.4570457994644658, 1.445305721320277,
                              -0.5900435899266435};
    out[0] = C0;
    out[1] = -C1 * y;
    out[2] = C1 * z;
    out[3] = -C1 * x;
    out[4] = C2[0] * x * y;
    out[5] = C2[1] * y * z;
    out[6] = C2[2] * (2.0 * z * z - x * x - y * y);
    out[7] = C2[3] * x * z;
    out[8] = C2[4] * (x * x - y * y);
    out[9] = C3[0] * y * (3.0 * x * x - y * y);
    out[10] = C3[1] * x * y * z;
    out[11] = C3[2] * y * (4.0 * z * z - x * x - y * y);
    out[12] = C3[3] * z * (2.0 * z * z - 3.0 * x * x - 3.0 * y * y);
    out[13] = C3[4] * x * (4.0 * z * z - x * x - y * y);
    out[14] = C3[5] * z * (x * x - y * y);
    out[15] = C3[6] * x * (x * x - 3.0 * y * y);
}

// Test (c): identity copy, composition, and the SH rotation.
void testTransforms(Device & device)
{
    std::printf("\n-- (c) transformedCopy: identity, composition, SH rotation --\n");

    auto source = makeCloud(device, 3000, 3, 301, /*relight=*/true);
    auto copy = transformedCopy(device, *source, LayerTransform{});
    check(sameBits(*copy, *source), "transformedCopy(identity) is bit-exact (all streams)");

    // Composition: T1 then T2 equals T2*T1 = (rotation q2 q1, scale s2 s1,
    // translation t2 + s2 R2 t1), since uniform scale commutes with rotation.
    LayerTransform t1;
    t1.rotation = glm::angleAxis(0.4f, glm::normalize(glm::vec3(0.3f, -1.0f, 0.5f)));
    t1.translation = glm::vec3(0.3f, 1.0f, -0.7f);
    t1.scale = 1.3f;
    LayerTransform t2;
    t2.rotation = glm::angleAxis(-0.9f, glm::normalize(glm::vec3(-0.6f, 0.2f, 1.0f)));
    t2.translation = glm::vec3(-2.0f, 0.5f, 0.1f);
    t2.scale = 0.8f;

    LayerTransform composed;
    composed.rotation = t2.rotation * t1.rotation;
    composed.scale = t2.scale * t1.scale;
    composed.translation = t2.translation + t2.scale * (t2.rotation * t1.translation);

    auto stepped = transformedCopy(device, *transformedCopy(device, *source, t1), t2);
    auto direct = transformedCopy(device, *source, composed);
    GaussianCloud & s = hostOf(*stepped);
    GaussianCloud & d = hostOf(*direct);
    const double positions = worstVec4(s.positions(), d.positions());
    const double scales = worstVec4(s.scales(), d.scales());
    const double rotations = worstRotation(s.rotations(), d.rotations());
    const double sh = worstVec4(s.shCoefficients(), d.shCoefficients());
    const double normals = worstVec4(s.normals(), d.normals());
    std::printf("  worst |T2(T1 x) - (T2 T1) x|: positions %.3e, scales %.3e, rotations %.3e\n"
                "                                 SH %.3e, normals %.3e\n",
                positions, scales, rotations, sh, normals);
    check(positions <= kTolerance, "composition: positions within tolerance");
    check(scales <= kTolerance, "composition: scales within tolerance");
    check(rotations <= kTolerance, "composition: rotations within tolerance (modulo sign)");
    check(sh <= kTolerance, "composition: SH within tolerance");
    check(normals <= kTolerance, "composition: normals within tolerance");

    // SH rotation physics: a Gaussian rotated by R shows, along R*d, the colour its
    // local data shows along d. Evaluated with an independent degree-3 basis.
    auto onePoint = std::make_shared<GaussianCloud>(device);
    onePoint->initialize(1);
    onePoint->setShDegree(3);
    std::mt19937 rng(401);
    std::uniform_real_distribution<float> coefficient(-0.5f, 0.5f);
    onePoint->positions()[0] = glm::vec4(0.0f);
    onePoint->scales()[0] = glm::vec4(0.1f, 0.1f, 0.1f, 0.0f);
    onePoint->rotations()[0] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
    onePoint->opacities()[0] = 0.5f;
    for (size_t k = 0; k < kShCoefficientsPerGaussian; ++k)
    {
        onePoint->shCoefficients()[k] =
            glm::vec4(coefficient(rng), coefficient(rng), coefficient(rng), 0.0f);
    }
    onePoint->computeBounds();

    LayerTransform rotate;
    rotate.rotation = randomRotation(rng);
    const glm::dmat3 rotation = glm::mat3_cast(glm::dquat(rotate.rotation.w, rotate.rotation.x,
                                                          rotate.rotation.y, rotate.rotation.z));
    auto rotated = transformedCopy(device, *onePoint, rotate);
    const auto & original = hostOf(*onePoint).shCoefficients();
    const auto & turned = hostOf(*rotated).shCoefficients();

    double worstSH = 0.0;
    constexpr int kDirections = 1000;
    for (int n = 0; n < kDirections; ++n)
    {
        const glm::dvec3 dir = glm::dvec3(randomUnit(rng));
        const glm::dvec3 turnedDir = rotation * dir;
        double fOriginal[16];
        double fTurned[16];
        shBasis16(dir.x, dir.y, dir.z, fOriginal);
        shBasis16(turnedDir.x, turnedDir.y, turnedDir.z, fTurned);
        for (int c = 0; c < 3; ++c)
        {
            double valueOriginal = 0.0;
            double valueTurned = 0.0;
            for (size_t k = 0; k < kShCoefficientsPerGaussian; ++k)
            {
                valueOriginal += fOriginal[k] * original[k][c];
                valueTurned += fTurned[k] * turned[k][c];
            }
            worstSH = std::max(worstSH, std::abs(valueOriginal - valueTurned));
        }
    }
    // The DC term is rotation invariant and copied, so it must be bit-exact.
    check(original[0] == turned[0], "SH rotation: DC term is bit-exact");
    std::printf("  worst |f_rotated(R d) - f_original(d)| over %d directions: %.3e\n",
                kDirections, worstSH);
    check(worstSH <= kTolerance, "SH rotation: rotated SH at R*d equals original at d");
}

// Test (d): relight streams across layers.
void testRelightMerge(Device & device)
{
    std::printf("\n-- (d) mergeLayers relight rules --\n");
    auto a = makeCloud(device, 300, 2, 501, /*relight=*/true);
    auto b = makeCloud(device, 200, 1, 502, /*relight=*/true);
    hostOf(*b).metallic().clear(); // present in A only

    LayerTransform identity;
    LayerTransform placed;
    placed.rotation = glm::angleAxis(1.1f, glm::normalize(glm::vec3(0.2f, 0.9f, -0.4f)));
    placed.translation = glm::vec3(4.0f, 0.0f, 0.0f);

    auto merged = mergeLayers(device, {{a.get(), identity}, {b.get(), placed}});
    GaussianCloud & m = hostOf(*merged);
    GaussianCloud & sa = hostOf(*a);
    GaussianCloud & sb = hostOf(*b);

    check(m.metallic().empty(), "metallic (present in one layer only) is dropped");
    check(m.normals().size() == 500, "normals (present in every layer) are kept, 500 Gaussians");
    check(m.baseColorRoughness().size() == 500, "baseColorRoughness kept");
    check(m.visibility().size() == 500 * kVisibilityCoefficients, "visibility kept, full width");

    // Identity layer: normals bit-exact. Placed layer: normals are R*n.
    const auto & mn = m.normals();
    bool aNormalsExact = true;
    for (size_t i = 0; i < 300; ++i)
    {
        aNormalsExact = aNormalsExact && std::memcmp(&mn[i], &sa.normals()[i], sizeof(glm::vec4)) == 0;
    }
    check(aNormalsExact, "identity layer normals are bit-exact");

    const glm::dmat3 rotation = glm::mat3_cast(glm::dquat(placed.rotation.w, placed.rotation.x,
                                                          placed.rotation.y, placed.rotation.z));
    double worstNormal = 0.0;
    for (size_t i = 0; i < 200; ++i)
    {
        const glm::vec4 n = sb.normals()[i];
        const glm::dvec3 expected = rotation * glm::dvec3(n.x, n.y, n.z);
        const glm::vec4 got = mn[300 + i];
        worstNormal = std::max({worstNormal,
                                std::abs(got.x - expected.x),
                                std::abs(got.y - expected.y),
                                std::abs(got.z - expected.z)});
    }
    std::printf("  worst |merged normal - R n| for the placed layer: %.3e\n", worstNormal);
    check(worstNormal <= kTolerance, "placed layer normals are rotated by R");
}

// Test (e): refusals.
void testRefusals(Device & device)
{
    std::printf("\n-- (e) refusals --\n");
    bool threw = false;
    try
    {
        mergeLayers(device, {});
    }
    catch (const std::exception &)
    {
        threw = true;
    }
    check(threw, "mergeLayers throws on an empty list");

    threw = false;
    try
    {
        GLTFWriter::writeGLB({}, (g_tempDir / "empty.glb").string());
    }
    catch (const std::exception &)
    {
        threw = true;
    }
    check(threw, "writeGLB throws on an empty layer list");

    auto bad = makeCloud(device, 10, 0, 601);
    hostOf(*bad).opacities()[3] = 1.5f;
    threw = false;
    try
    {
        GLTFWriter::writeGLB({GLTFExportLayer{bad.get(), LayerTransform{}, "bad"}},
                             (g_tempDir / "bad.glb").string());
    }
    catch (const std::exception &)
    {
        threw = true;
    }
    check(threw, "writeGLB refuses an opacity outside [0, 1]");
    check(!std::filesystem::exists(g_tempDir / "bad.glb"), "a refused write leaves no file");
}

// Test (g): a rotation must be unit length to the project tolerance, per Gaussian
// and on the layer transform. Zero scale and two layers with one name are accepted
// and read back.
void testRotationRules(Device & device)
{
    std::printf("\n-- (g) rotation rules, zero scale, duplicate names --\n");

    // Writes the layers and returns the refusal message, or "" when the write succeeds.
    auto refusalOf = [](const std::vector<GLTFExportLayer> & layers, const std::string & name) {
        const std::string path = (g_tempDir / name).string();
        try
        {
            GLTFWriter::writeGLB(layers, path);
        }
        catch (const std::exception & e)
        {
            check(!std::filesystem::exists(path) && !std::filesystem::exists(path + ".tmp"),
                  name + ": a refused write leaves no file or temporary file");
            return std::string(e.what());
        }
        return std::string();
    };

    auto wide = makeCloud(device, 10, 0, 801);
    hostOf(*wide).rotations()[4] = glm::vec4(0.0f, 0.0f, 0.0f, 1.01f);
    std::string message = refusalOf({GLTFExportLayer{wide.get(), LayerTransform{}, "wide"}}, "wide.glb");
    check(message.find("layer 0 ('wide')") != std::string::npos
              && message.find("Gaussian 4 rotation is not unit length") != std::string::npos,
          "rotation of length 1.01 refused, layer and Gaussian 4 named");

    auto half = makeCloud(device, 10, 0, 802);
    hostOf(*half).rotations()[7] = glm::vec4(0.0f, 0.0f, 0.5f, 0.0f);
    message = refusalOf({GLTFExportLayer{half.get(), LayerTransform{}, "half"}}, "half.glb");
    check(message.find("layer 0 ('half')") != std::string::npos
              && message.find("Gaussian 7 rotation is not unit length") != std::string::npos,
          "rotation of length 0.5 refused, layer and Gaussian 7 named");

    auto tilt = makeCloud(device, 10, 0, 803);
    LayerTransform tilted;
    tilted.rotation = glm::quat(1.01f, 0.0f, 0.0f, 0.0f);
    message = refusalOf({GLTFExportLayer{tilt.get(), tilted, "tilt"}}, "tilt.glb");
    check(message.find("layer 0 ('tilt')") != std::string::npos
              && message.find("transform rotation is not unit length") != std::string::npos,
          "layer transform rotation of length 1.01 refused, layer named");

    // Within 1e-6 of unit length, on both sides: accepted, and the round trip
    // returns the same rotations to float rounding.
    auto near = makeCloud(device, 40, 1, 804);
    for (size_t i = 0; i < 40; ++i)
    {
        hostOf(*near).rotations()[i] *= (i % 2 == 0) ? 1.0f + 1e-6f : 1.0f - 1e-6f;
    }
    auto nearBack = roundTrip(device, {GLTFExportLayer{near.get(), LayerTransform{}, "near"}}, "near_rt.glb");
    check(nearBack->count() == 40, "rotations 1e-6 from unit length are accepted, count matches");
    const double nearError = worstRotation(hostOf(*near).rotations(), hostOf(*nearBack).rotations());
    std::printf("  worst rotation error after round trip: %.3e\n", nearError);
    check(nearError <= 2e-6, "rotations 1e-6 from unit length round-trip to float rounding");

    // A zero scale is a flat splat, which the reader accepts: it must round-trip exactly.
    auto zero = makeCloud(device, 300, 2, 805);
    for (size_t i = 0; i < 300; i += 3)
    {
        hostOf(*zero).scales()[i] = glm::vec4(0.0f);
    }
    hostOf(*zero).scales()[1].y = 0.0f;
    auto zeroBack = roundTrip(device, {GLTFExportLayer{zero.get(), LayerTransform{}, "zero"}}, "zero_scale_rt.glb");
    check(bitEqual(hostOf(*zeroBack).scales(), hostOf(*zero).scales()),
          "zero scale accepted, scales bit-exact after round trip");
    check(bitEqual(hostOf(*zeroBack).positions(), hostOf(*zero).positions()),
          "zero scale accepted, positions bit-exact after round trip");

    // Two layers with one name are two meshes and nodes, both read back in order.
    auto twin = makeCloud(device, 50, 0, 806);
    auto twinBack = roundTrip(device,
                              {GLTFExportLayer{twin.get(), LayerTransform{}, "twin"},
                               GLTFExportLayer{twin.get(), LayerTransform{}, "twin"}},
                              "twins.glb");
    std::vector<glm::vec4> doubled = hostOf(*twin).positions();
    doubled.insert(doubled.end(), hostOf(*twin).positions().begin(), hostOf(*twin).positions().end());
    check(twinBack->count() == 100 && bitEqual(hostOf(*twinBack).positions(), doubled),
          "two layers named 'twin' are both written and read back, in order");
}

} // namespace

// Writes and reads back a .glb under a directory and file name that are not ASCII.
// The writer and reader take UTF-8 std::strings, so the path must reach the
// filesystem through u8path (not the ANSI codepage on Windows).
void testNonAsciiPath(Device & device)
{
    std::printf("\n-- (f) non-ASCII path --\n");
    // "Ünïcödé 層" and "проект.glb" as UTF-8 byte escapes.
    const std::string unicodeDir = "\xC3\x9C"
                                   "n"
                                   "\xC3\xAF"
                                   "c"
                                   "\xC3\xB6"
                                   "d"
                                   "\xC3\xA9"
                                   " \xE5\xB1\xA4";
    const std::string cyrillicFile = "\xD0\xBF\xD1\x80\xD0\xBE\xD0\xB5\xD0\xBA\xD1\x82.glb";
    const std::filesystem::path dir = g_tempDir / std::filesystem::u8path(unicodeDir);
    std::filesystem::create_directories(dir);
    const std::filesystem::path file = dir / std::filesystem::u8path(cyrillicFile);
    const std::u8string fileUtf8 = file.u8string();
    const std::string path(reinterpret_cast<const char *>(fileUtf8.data()), fileUtf8.size());

    auto source = makeCloud(device, 500, 1, 700);
    GLTFWriter::writeGLB({GLTFExportLayer{source.get(), LayerTransform{}, "solo"}}, path);
    check(std::filesystem::exists(file), "non-ASCII path: .glb exists under its UTF-8 name");
    check(!std::filesystem::exists(std::filesystem::u8path(path + ".tmp")),
          "non-ASCII path: no temporary file left behind");
    auto read = GltfReader::read(device, path);
    check(read->count() == source->count(), "non-ASCII path: count matches");
    check(bitEqual(hostOf(*read).positions(), hostOf(*source).positions()),
          "non-ASCII path: positions bit-exact");
    check(bitEqual(hostOf(*read).shCoefficients(), hostOf(*source).shCoefficients()),
          "non-ASCII path: SH bit-exact");
    std::filesystem::remove_all(dir);
}

int main()
{
    g_tempDir = std::filesystem::temp_directory_path() / "sabercat_gltf_writer_test";
    std::filesystem::create_directories(g_tempDir);

    try
    {
        Device device;
        device.initialize({}, /*enableValidation=*/true);
        testRoundTrip(device);
        testMergeAgainstReader(device);
        testTransforms(device);
        testRelightMerge(device);
        testRefusals(device);
        testRotationRules(device);
        testNonAsciiPath(device);
        g_failures += runExtraChecks(device, g_tempDir);
        device.shutdown();
    }
    catch (const std::exception & e)
    {
        std::printf("[FAIL] unexpected exception: %s\n", e.what());
        ++g_failures;
    }

    std::filesystem::remove_all(g_tempDir);
    std::printf("\n%s (%d failure%s)\n",
                g_failures == 0 ? "gltf_writer_test OK" : "gltf_writer_test FAILED",
                g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}
