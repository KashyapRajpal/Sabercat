// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Forward-output regression test.
//
// Renders four fixed scenes built in code and compares an FNV-1a 64-bit hash
// of each output image's exact bytes with the stored constants in kExpected.
// Forward output is bitwise reproducible on a given device and driver, so any
// change to the forward pass's output bits fails this test until the change is
// acknowledged and documented. Backward is not covered: fp32 atomics make it
// non-reproducible.
//
// Coverage: color and alpha from readImage(). The forward depth image has no
// public readback, so it is not hashed.
//
// Inputs come from a seeded xorshift32 generator. Every float is built with
// exact or correctly rounded operations only (integer to float by powers of
// two, +, -, *, /, sqrt); the host never calls a transcendental function. So
// the inputs are the same in Debug and Release and on any host byte order (the
// hash feeds each float's bytes least-significant first).
//
// Each scene is rendered twice, each time on a fresh Renderer. The two hashes
// must agree on every device; a disagreement is reported as nondeterminism,
// separately from a numerics change.
//
// Usage:
//   sabercat_forward_regression_test          check every scene against kExpected
//   sabercat_forward_regression_test --print  print the current hashes and device as table rows

#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sabercat/camera.h"
#include "sabercat/renderer.h"

#include "../common/shader_dir.h"

using namespace sabercat;

namespace
{

// The stored hashes. This table is the only place the expected output lives.
//
// If this test fails after an intentional change to forward numerics, re-run
// with --print to get the new hashes, update this table in the same commit,
// and add a 'Numerics' entry to CHANGELOG.md describing the change and its
// measured effect.
//
// A failure that is not a deliberate numerics change is a bug. A
// NONDETERMINISM line means two renders of the same scene in one process
// disagreed (a driver, race or uninitialized-read problem), not a numerics
// change.
//
// Reference device. The hashes are comparable only on the device and driver
// they were recorded on (kReference). Other GPUs or driver versions can round
// exp() or contract FMA differently, so there the test still runs the
// determinism check, prints the hashes, and reports a [SKIP] line instead of
// comparing. To move the reference to another device or driver, run --print on
// that machine, then update kReference and kExpected in the same commit. If only
// the device or driver changed and Sabercat did not, no 'Numerics' entry is
// needed.
constexpr const char * kPolicy =
    "If this test fails after an intentional change to forward numerics, re-run "
    "with --print to get the new hashes, update this table in the same commit, "
    "and add a 'Numerics' entry to CHANGELOG.md describing the change and its "
    "measured effect.";

struct ReferenceDevice
{
    const char * name;
    uint32_t vendorID;
    uint32_t deviceID;
    uint32_t driverVersion;
};

// Recorded with --print on the reference machine.
constexpr ReferenceDevice kReference = {"NVIDIA GeForce RTX 5070 Ti", 0x10deu, 0x2c05u, 0x98960000u};

struct ExpectedHash
{
    const char * scene;
    uint64_t hash;
};

constexpr ExpectedHash kExpected[] = {
    {"sparse_sh0", 0x2981e7d4c5d6ca3fULL},
    {"rotated_sh3", 0x84e0f48c68dd8c89ULL},
    {"rotated_sh3_aa", 0xe07019535b419da0ULL},
    {"edge_odd_sh3", 0xfa090278f0c41de1ULL},
};

// xorshift32 (Marsaglia, 2003). Its output is an exact function of the seed,
// and the seed must be nonzero.
class Xorshift32
{
  public:
    explicit Xorshift32(uint32_t seed) : m_state(seed)
    {
    }

    uint32_t next()
    {
        m_state ^= m_state << 13;
        m_state ^= m_state >> 17;
        m_state ^= m_state << 5;
        return m_state;
    }

    // [0, 1) from the top 24 bits. Exact: 2^24 fits a float's significand.
    float unit()
    {
        return static_cast<float>(next() >> 8) * (1.0f / 16777216.0f);
    }

    // lo + (hi - lo) * unit(): two correctly rounded float operations.
    float range(float lo, float hi)
    {
        return lo + (hi - lo) * unit();
    }

  private:
    uint32_t m_state;
};

// Host arrays in the GaussianHostInput layout. Each draw from the generator
// gets its own statement: the order of argument evaluation is unspecified in
// C++, so two draws inside one call expression could swap between builds.
struct GaussianSet
{
    std::vector<float> means;     // 4 per Gaussian (xyz, pad)
    std::vector<float> scales;    // 4 per Gaussian (linear, xyz, pad)
    std::vector<float> rotations; // 4 per Gaussian (unit quaternion, xyzw)
    std::vector<float> opacities; // 1 per Gaussian, in [0, 1]
    std::vector<float> sh;        // 64 per Gaussian: 16 coefficients x (rgb, pad)

    uint32_t count() const
    {
        return static_cast<uint32_t>(opacities.size());
    }
};

struct Scene
{
    const char * name = "";
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t shDegree = 0;
    bool antialiased = false;
    Camera camera;
    GaussianSet gaussians;
};

// Identity of the device a renderer runs on, as Vulkan reports it.
struct DeviceIdentity
{
    uint32_t vendorID = 0;
    uint32_t deviceID = 0;
    uint32_t driverVersion = 0;
    std::string name;
};

DeviceIdentity identify(const Renderer & renderer)
{
    VkPhysicalDeviceProperties properties {};
    vkGetPhysicalDeviceProperties(renderer.physicalDevice(), &properties);
    DeviceIdentity id;
    id.vendorID = properties.vendorID;
    id.deviceID = properties.deviceID;
    id.driverVersion = properties.driverVersion;
    id.name = properties.deviceName;
    return id;
}

bool matchesReference(const DeviceIdentity & id)
{
    return id.vendorID == kReference.vendorID && id.deviceID == kReference.deviceID
           && id.driverVersion == kReference.driverVersion;
}

glm::vec3 drawBox(Xorshift32 & rng, glm::vec3 lo, glm::vec3 hi)
{
    glm::vec3 v;
    v.x = rng.range(lo.x, hi.x);
    v.y = rng.range(lo.y, hi.y);
    v.z = rng.range(lo.z, hi.z);
    return v;
}

// Appends one Gaussian: the given center and linear scale, a random unit
// rotation, the given opacity, and random SH (wide DC, small higher bands).
void appendGaussian(GaussianSet & set, Xorshift32 & rng, glm::vec3 center, glm::vec3 scale, float opacity)
{
    set.means.insert(set.means.end(), {center.x, center.y, center.z, 0.0f});
    set.scales.insert(set.scales.end(), {scale.x, scale.y, scale.z, 0.0f});

    float q[4];
    for (float & c : q)
    {
        c = rng.range(-1.0f, 1.0f);
    }
    float norm = std::sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]);
    if (!(norm > 1e-6f))
    {
        q[0] = 0.0f;
        q[1] = 0.0f;
        q[2] = 0.0f;
        q[3] = 1.0f;
        norm = 1.0f;
    }
    for (float & c : q)
    {
        c /= norm;
    }
    set.rotations.insert(set.rotations.end(), {q[0], q[1], q[2], q[3]});

    set.opacities.push_back(opacity);

    for (uint32_t k = 0; k < 16; ++k)
    {
        const float spread = (k == 0) ? 1.5f : 0.15f;
        for (int c = 0; c < 3; ++c)
        {
            const float coefficient = rng.range(-spread, spread);
            set.sh.push_back(coefficient);
        }
        set.sh.push_back(0.0f);
    }
}

void appendScatter(GaussianSet & set, Xorshift32 & rng, uint32_t count, glm::vec3 lo, glm::vec3 hi,
                   float scaleLo, float scaleHi)
{
    for (uint32_t i = 0; i < count; ++i)
    {
        const glm::vec3 center = drawBox(rng, lo, hi);
        const glm::vec3 scale = drawBox(rng, glm::vec3(scaleLo), glm::vec3(scaleHi));
        const float opacity = rng.unit();
        appendGaussian(set, rng, center, scale, opacity);
    }
}

// Camera-to-world transform for a unit quaternion (xyzw) and a position.
// Columns are written out from the standard quaternion-to-matrix formulas.
glm::mat4 makePose(float x, float y, float z, float w, glm::vec3 position)
{
    const float norm = std::sqrt(x * x + y * y + z * z + w * w);
    x /= norm;
    y /= norm;
    z /= norm;
    w /= norm;

    glm::mat4 m(1.0f);
    m[0] = glm::vec4(1.0f - 2.0f * (y * y + z * z), 2.0f * (x * y + z * w), 2.0f * (x * z - y * w), 0.0f);
    m[1] = glm::vec4(2.0f * (x * y - z * w), 1.0f - 2.0f * (x * x + z * z), 2.0f * (y * z + x * w), 0.0f);
    m[2] = glm::vec4(2.0f * (x * z + y * w), 2.0f * (y * z - x * w), 1.0f - 2.0f * (x * x + y * y), 0.0f);
    m[3] = glm::vec4(position, 1.0f);
    return m;
}

Camera makeCamera(uint32_t width, uint32_t height, float fx, float fy, float cx, float cy,
                  const glm::mat4 & cameraToWorld)
{
    Camera camera;
    camera.width = width;
    camera.height = height;
    camera.fx = fx;
    camera.fy = fy;
    camera.cx = cx;
    camera.cy = cy;
    camera.cameraToWorld = cameraToWorld;
    return camera;
}

// Scenes (b) and (c): the same cloud, rotated and translated camera, off-center
// principal point, fx != fy. Only the antialiasing flag differs.
Scene buildRotatedScene(const char * name, bool antialiased)
{
    Scene scene;
    scene.name = name;
    scene.width = 256;
    scene.height = 192;
    scene.shDegree = 3;
    scene.antialiased = antialiased;
    scene.camera = makeCamera(256, 192, 310.0f, 290.0f, 119.5f, 101.25f,
                              makePose(0.04f, -0.06f, 0.02f, 1.0f, glm::vec3(0.4f, -0.3f, 5.0f)));

    Xorshift32 rng(0x2C0FFEE1u);
    appendScatter(scene.gaussians, rng, 5000, glm::vec3(-1.2f, -1.2f, -1.2f), glm::vec3(1.2f, 1.2f, 1.2f),
                  0.03f, 0.2f);
    return scene;
}

// Scene (d): splats at and past the near plane, behind the camera, partly
// off-screen, coincident overlaps, and opacities at the 1/255 threshold.
// Image 97x61 (odd sizes, not tile multiples).
Scene buildEdgeScene()
{
    Scene scene;
    scene.name = "edge_odd_sh3";
    scene.width = 97;
    scene.height = 61;
    scene.shDegree = 3;
    scene.antialiased = false;
    scene.camera = makeCamera(97, 61, 55.0f, 60.0f, 48.5f, 30.5f, glm::mat4(1.0f));

    GaussianSet & set = scene.gaussians;
    Xorshift32 rng(0x0DDBA11u);

    // Near plane: centers 0.02 to 0.5 in front of the camera, small scales.
    for (uint32_t i = 0; i < 120; ++i)
    {
        const glm::vec3 center = drawBox(rng, glm::vec3(-0.3f, -0.3f, -0.5f), glm::vec3(0.3f, 0.3f, -0.02f));
        const glm::vec3 scale = drawBox(rng, glm::vec3(0.005f), glm::vec3(0.08f));
        const float opacity = rng.range(0.3f, 1.0f);
        appendGaussian(set, rng, center, scale, opacity);
    }

    // Behind the camera (positive z): must contribute nothing.
    for (uint32_t i = 0; i < 60; ++i)
    {
        const glm::vec3 center = drawBox(rng, glm::vec3(-1.0f, -1.0f, 0.2f), glm::vec3(1.0f, 1.0f, 2.0f));
        const glm::vec3 scale = drawBox(rng, glm::vec3(0.05f), glm::vec3(0.5f));
        const float opacity = rng.range(0.5f, 1.0f);
        appendGaussian(set, rng, center, scale, opacity);
    }

    // Scatter, partly off-screen, with large scales.
    appendScatter(set, rng, 200, glm::vec3(-4.0f, -3.0f, -6.0f), glm::vec3(4.0f, 3.0f, -1.0f), 0.05f, 1.5f);

    // Overlaps: 15 clusters of 10 Gaussians with identical centers.
    for (uint32_t c = 0; c < 15; ++c)
    {
        const glm::vec3 center = drawBox(rng, glm::vec3(-1.0f, -0.8f, -3.0f), glm::vec3(1.0f, 0.8f, -2.0f));
        for (uint32_t k = 0; k < 10; ++k)
        {
            const glm::vec3 scale = drawBox(rng, glm::vec3(0.3f), glm::vec3(0.6f));
            const float opacity = rng.range(0.5f, 1.0f);
            appendGaussian(set, rng, center, scale, opacity);
        }
    }

    // Opacity at the 1/255 threshold: k/255 for k = 0..3 (exact integer
    // over 255, correctly rounded).
    for (uint32_t i = 0; i < 40; ++i)
    {
        const uint32_t k = rng.next() % 4u;
        const glm::vec3 center = drawBox(rng, glm::vec3(-0.5f, -0.5f, -3.0f), glm::vec3(0.5f, 0.5f, -1.0f));
        const glm::vec3 scale = drawBox(rng, glm::vec3(0.2f), glm::vec3(0.6f));
        const float opacity = static_cast<float>(k) / 255.0f;
        appendGaussian(set, rng, center, scale, opacity);
    }

    // Near-opaque: 254/255 and fully opaque.
    for (uint32_t i = 0; i < 30; ++i)
    {
        const glm::vec3 center = drawBox(rng, glm::vec3(-1.5f, -1.0f, -4.0f), glm::vec3(1.5f, 1.0f, -2.0f));
        const glm::vec3 scale = drawBox(rng, glm::vec3(0.1f), glm::vec3(0.4f));
        const float opacity = (i % 2 == 0) ? 1.0f : 254.0f / 255.0f;
        appendGaussian(set, rng, center, scale, opacity);
    }

    return scene;
}

std::vector<Scene> buildScenes()
{
    std::vector<Scene> scenes;

    // (a) Identity camera, SH degree 0, no antialiasing.
    Scene sparse;
    sparse.name = "sparse_sh0";
    sparse.width = 256;
    sparse.height = 192;
    sparse.shDegree = 0;
    sparse.antialiased = false;
    sparse.camera = makeCamera(256, 192, 200.0f, 200.0f, 128.0f, 96.0f, glm::mat4(1.0f));
    {
        Xorshift32 rng(0x1A2B3C01u);
        appendScatter(sparse.gaussians, rng, 2000, glm::vec3(-2.0f, -1.5f, -8.0f), glm::vec3(2.0f, 1.5f, -3.0f),
                      0.02f, 0.15f);
    }
    scenes.push_back(std::move(sparse));

    // (b) Rotated and translated camera, SH degree 3, antialiasing off.
    scenes.push_back(buildRotatedScene("rotated_sh3", false));

    // (c) As (b), with antialiasing on.
    scenes.push_back(buildRotatedScene("rotated_sh3_aa", true));

    // (d) Edge cases, 97x61.
    scenes.push_back(buildEdgeScene());

    return scenes;
}

// FNV-1a 64 over the image's float bit patterns. Each float contributes its
// four bytes least-significant first, so the hash is the same on any host
// byte order.
uint64_t fnv1a64(const std::vector<float> & values)
{
    uint64_t hash = 0xcbf29ce484222325ULL;
    for (float value : values)
    {
        uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof bits);
        for (int shift = 0; shift < 32; shift += 8)
        {
            hash ^= static_cast<uint64_t>((bits >> shift) & 0xffu);
            hash *= 0x100000001b3ULL;
        }
    }
    return hash;
}

struct Rendered
{
    std::vector<float> image;
    DeviceIdentity device;
};

// Renders one scene on a fresh headless Renderer and returns readImage() and
// the device it ran on.
Rendered renderScene(const Scene & scene)
{
    Renderer renderer(scene.width, scene.height);
    if (!renderer.initialize({}, true, testShaderDir()))
    {
        throw std::runtime_error("Renderer::initialize failed (a Vulkan 1.3 device is required)");
    }

    GaussianHostInput input;
    input.count = scene.gaussians.count();
    input.means = scene.gaussians.means.data();
    input.scales = scene.gaussians.scales.data();
    input.rotations = scene.gaussians.rotations.data();
    input.opacities = scene.gaussians.opacities.data();
    input.shCoefficients = scene.gaussians.sh.data();
    input.shDegree = scene.shDegree;
    input.antialiased = scene.antialiased;

    renderer.setGaussians(input);
    renderer.setCamera(scene.camera);
    renderer.forward();

    Rendered result;
    result.image = renderer.readImage();
    result.device = identify(renderer);
    const size_t expected = static_cast<size_t>(scene.width) * scene.height * 4;
    if (result.image.size() != expected)
    {
        throw std::runtime_error("readImage() returned " + std::to_string(result.image.size()) + " floats, expected "
                                 + std::to_string(expected));
    }
    return result;
}

// Context only, not part of the check: the mean over every RGBA value, and the
// fraction of pixels with any coverage (alpha > 0).
void imageStats(const std::vector<float> & image, double & mean, double & coverage)
{
    double sum = 0.0;
    for (float v : image)
    {
        sum += static_cast<double>(v);
    }
    const size_t pixels = image.size() / 4;
    size_t covered = 0;
    for (size_t p = 0; p < pixels; ++p)
    {
        if (image[4 * p + 3] > 0.0f)
        {
            ++covered;
        }
    }
    mean = image.empty() ? 0.0 : sum / static_cast<double>(image.size());
    coverage = pixels == 0 ? 0.0 : static_cast<double>(covered) / static_cast<double>(pixels);
}

const ExpectedHash * findExpected(const char * scene)
{
    for (const ExpectedHash & entry : kExpected)
    {
        if (std::strcmp(entry.scene, scene) == 0)
        {
            return &entry;
        }
    }
    return nullptr;
}

} // namespace

int main(int argc, char ** argv)
{
    bool printOnly = false;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "--print") == 0)
        {
            printOnly = true;
        }
        else
        {
            std::fprintf(stderr, "usage: %s [--print]\n", argv[0]);
            return 1;
        }
    }

    try
    {
        const std::vector<Scene> scenes = buildScenes();
        bool failed = false;
        bool mismatched = false;
        bool haveDevice = false;
        bool onReference = false;
        std::vector<bool> seen(sizeof(kExpected) / sizeof(kExpected[0]), false);

        if (printOnly)
        {
            std::printf("// Rows for kExpected (scene, hash, then mean and coverage as context):\n");
        }
        else
        {
            std::printf("forward_regression_test: %zu scenes, each rendered twice on fresh renderers\n",
                        scenes.size());
        }

        for (const Scene & scene : scenes)
        {
            const auto start = std::chrono::steady_clock::now();
            const Rendered first = renderScene(scene);
            const Rendered second = renderScene(scene);
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

            if (!haveDevice)
            {
                haveDevice = true;
                onReference = matchesReference(first.device);
                const DeviceIdentity & id = first.device;
                if (printOnly)
                {
                    std::printf("// Recorded on: %s (vendor 0x%04x, device 0x%04x, driver 0x%08x)\n",
                                id.name.c_str(), id.vendorID, id.deviceID, id.driverVersion);
                    std::printf("// Paste as kReference:\n");
                    std::printf("constexpr ReferenceDevice kReference = {\"%s\", 0x%04xu, 0x%04xu, 0x%08xu};\n",
                                id.name.c_str(), id.vendorID, id.deviceID, id.driverVersion);
                }
                else if (!onReference)
                {
                    std::printf("[SKIP] hash comparison: running on %s (vendor 0x%04x, device 0x%04x, driver 0x%08x) "
                                "but the stored hashes were recorded on %s (vendor 0x%04x, device 0x%04x, driver "
                                "0x%08x); determinism checked only. Re-record with --print on the reference machine "
                                "if the reference changes.\n",
                                id.name.c_str(), id.vendorID, id.deviceID, id.driverVersion, kReference.name,
                                kReference.vendorID, kReference.deviceID, kReference.driverVersion);
                }
            }

            const uint64_t hash = fnv1a64(first.image);
            const uint64_t repeat = fnv1a64(second.image);
            double mean = 0.0;
            double coverage = 0.0;
            imageStats(first.image, mean, coverage);

            if (hash != repeat)
            {
                failed = true;
                std::printf("NONDETERMINISM scene '%s': two renders in one process gave 0x%016llx and 0x%016llx. "
                            "This is a driver, race or uninitialized-read problem, not a numerics change.\n",
                            scene.name, static_cast<unsigned long long>(hash),
                            static_cast<unsigned long long>(repeat));
                continue;
            }

            if (printOnly)
            {
                std::printf("    {\"%s\", 0x%016llxULL},  // %ux%u mean %.6f coverage %.6f\n", scene.name,
                            static_cast<unsigned long long>(hash), scene.width, scene.height, mean, coverage);
                continue;
            }

            if (!onReference)
            {
                std::printf("  %-16s %3ux%-3u hash 0x%016llx  mean %.6f  coverage %.6f  not compared  (%.2f s for two renders)\n",
                            scene.name, scene.width, scene.height, static_cast<unsigned long long>(hash), mean,
                            coverage, seconds);
                continue;
            }

            const ExpectedHash * expected = findExpected(scene.name);
            const char * status = "ok";
            if (expected == nullptr)
            {
                failed = true;
                mismatched = true;
                status = "NO STORED HASH";
                std::printf("MISMATCH scene '%s': no entry in kExpected (actual 0x%016llx)\n", scene.name,
                            static_cast<unsigned long long>(hash));
            }
            else
            {
                seen[static_cast<size_t>(expected - kExpected)] = true;
                if (expected->hash != hash)
                {
                    failed = true;
                    mismatched = true;
                    status = "MISMATCH";
                    std::printf("MISMATCH scene '%s': expected 0x%016llx, actual 0x%016llx\n", scene.name,
                                static_cast<unsigned long long>(expected->hash),
                                static_cast<unsigned long long>(hash));
                }
            }

            std::printf("  %-16s %3ux%-3u hash 0x%016llx  mean %.6f  coverage %.6f  %s  (%.2f s for two renders)\n",
                        scene.name, scene.width, scene.height, static_cast<unsigned long long>(hash), mean,
                        coverage, status, seconds);
        }

        if (printOnly)
        {
            return failed ? 1 : 0;
        }

        if (onReference)
        {
            for (size_t i = 0; i < seen.size(); ++i)
            {
                if (!seen[i])
                {
                    failed = true;
                    std::printf("STALE kExpected entry '%s' names no scene in this test\n", kExpected[i].scene);
                }
            }
        }

        if (mismatched)
        {
            std::printf("\n%s\n", kPolicy);
        }
        if (failed)
        {
            std::printf("\nforward_regression_test: FAILED\n");
        }
        else if (onReference)
        {
            std::printf("\nforward_regression_test: PASSED\n");
        }
        else
        {
            std::printf("\nforward_regression_test: PASSED (determinism only; hashes not compared on this device)\n");
        }
        return failed ? 1 : 0;
    }
    catch (const std::exception & e)
    {
        std::fprintf(stderr, "forward_regression_test: %s\n", e.what());
        return 1;
    }
}
