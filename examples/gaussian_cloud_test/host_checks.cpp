// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Host and GPU checks for GaussianCloud's load/store path (WP6):
//
//  (a) uploadToGPU -> downloadFromGPU is bit-exact against the host vectors,
//      for SH degrees 0..3, with and without the relightable streams, for
//      capacity == count, capacity > count, and count 1. The host vectors are
//      zeroed before the download, so every value compared was read back from
//      the GPU. The relightable streams are read from the GPU directly, because
//      downloadFromGPU() only returns visibility.
//  (b) uploadToGPU refuses an empty cloud and host streams that are not sized
//      for count(), naming the problem.
//  (c) computeBounds() against a brute-force reference (sorted copies), on
//      random clouds, a point, two points, a degenerate cloud, an all-NaN
//      cloud and an empty one.

#include <glm/glm.hpp>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "sabercat/buffer.h"
#include "sabercat/device.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/render_config.h"

using namespace sabercat;

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

bool bitEqual(float a, float b)
{
    uint32_t x = 0;
    uint32_t y = 0;
    std::memcpy(&x, &a, sizeof x);
    std::memcpy(&y, &b, sizeof y);
    return x == y;
}

bool bitEqual(const glm::vec4 & a, const glm::vec4 & b)
{
    return bitEqual(a.x, b.x) && bitEqual(a.y, b.y) && bitEqual(a.z, b.z) && bitEqual(a.w, b.w);
}

// Index of the first element whose bits differ, or a.size() when every element
// matches (and the sizes agree).
template <typename T>
size_t firstMismatch(const std::vector<T> & a, const std::vector<T> & b)
{
    if (a.size() != b.size())
    {
        return 0;
    }
    for (size_t i = 0; i < a.size(); ++i)
    {
        if (!bitEqual(a[i], b[i]))
        {
            return i;
        }
    }
    return a.size();
}

template <typename T>
bool sameBits(const std::vector<T> & a, const std::vector<T> & b)
{
    return a.size() == b.size() && firstMismatch(a, b) == a.size();
}

// Fills every host stream with seeded values. The values need not be unit
// quaternions or valid scales: the upload path is a byte copy, so any bit
// pattern must survive it, including the padding lane (w) of each vec4.
struct Host
{
    std::vector<glm::vec4> positions, scales, rotations, sh, normals, baseColorRoughness;
    std::vector<float> opacities, metallic, visibility;
};

Host makeHost(size_t count, uint32_t seed, bool relight)
{
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    const auto vec = [&]() { return glm::vec4(unit(rng), unit(rng), unit(rng), unit(rng)); };

    Host h;
    h.positions.resize(count);
    h.scales.resize(count);
    h.rotations.resize(count);
    h.sh.resize(count * kShCoefficientsPerGaussian);
    h.opacities.resize(count);
    for (auto & v : h.positions) v = vec();
    for (auto & v : h.scales) v = vec();
    for (auto & v : h.rotations) v = vec();
    for (auto & v : h.sh) v = vec();
    for (auto & v : h.opacities) v = unit(rng);
    if (relight)
    {
        h.normals.resize(count);
        h.baseColorRoughness.resize(count);
        h.metallic.resize(count);
        h.visibility.resize(count * kVisibilityCoefficients);
        for (auto & v : h.normals) v = vec();
        for (auto & v : h.baseColorRoughness) v = vec();
        for (auto & v : h.metallic) v = unit(rng);
        for (auto & v : h.visibility) v = unit(rng);
    }
    return h;
}

void install(GaussianCloud & cloud, const Host & h, uint32_t degree)
{
    cloud.initialize(h.positions.size());
    cloud.positions() = h.positions;
    cloud.scales() = h.scales;
    cloud.rotations() = h.rotations;
    cloud.shCoefficients() = h.sh;
    cloud.opacities() = h.opacities;
    cloud.normals() = h.normals;
    cloud.baseColorRoughness() = h.baseColorRoughness;
    cloud.metallic() = h.metallic;
    cloud.visibility() = h.visibility;
    cloud.setShDegree(degree);
}

// Reads `bytes` bytes of one GPU view back into host memory.
std::vector<char> readView(Device & device, const BufferView & view, size_t bytes)
{
    std::vector<char> out(bytes);
    downloadView(device, view, out.data(), bytes);
    return out;
}

template <typename T>
std::vector<T> toVector(const std::vector<char> & bytes)
{
    std::vector<T> out(bytes.size() / sizeof(T));
    std::memcpy(out.data(), bytes.data(), out.size() * sizeof(T));
    return out;
}

// Upload, zero the host copy, download, and compare everything bit for bit.
bool roundTrip(Device & device, size_t count, uint32_t degree, size_t capacity, bool relight,
               uint32_t seed, std::string & detail)
{
    const Host h = makeHost(count, seed, relight);
    GaussianCloud cloud(device);
    install(cloud, h, degree);
    cloud.uploadToGPU(capacity);

    const size_t expectedCapacity = std::max(count, capacity);
    if (cloud.capacity() != expectedCapacity)
    {
        detail = "capacity() is " + std::to_string(cloud.capacity()) + ", expected "
                 + std::to_string(expectedCapacity);
        return false;
    }

    // Zero everything the download is supposed to refill.
    std::fill(cloud.positions().begin(), cloud.positions().end(), glm::vec4(0.0f));
    std::fill(cloud.scales().begin(), cloud.scales().end(), glm::vec4(0.0f));
    std::fill(cloud.rotations().begin(), cloud.rotations().end(), glm::vec4(0.0f));
    std::fill(cloud.shCoefficients().begin(), cloud.shCoefficients().end(), glm::vec4(0.0f));
    std::fill(cloud.opacities().begin(), cloud.opacities().end(), 0.0f);
    std::fill(cloud.visibility().begin(), cloud.visibility().end(), 0.0f);
    cloud.downloadFromGPU();

    if (cloud.count() != count || cloud.shDegree() != degree)
    {
        detail = "count or shDegree changed across the round trip";
        return false;
    }
    // Bit comparisons throughout: NaN payloads and signed zeros count.
    if (!sameBits(cloud.positions(), h.positions) || !sameBits(cloud.scales(), h.scales)
        || !sameBits(cloud.rotations(), h.rotations) || !sameBits(cloud.shCoefficients(), h.sh))
    {
        detail = "a parameter stream differs from the host copy";
        return false;
    }
    if (!sameBits(cloud.opacities(), h.opacities))
    {
        detail = "opacity differs from the host copy";
        return false;
    }

    if (relight)
    {
        // The relightable streams are not read back by downloadFromGPU(),
        // so check the GPU bytes directly.
        const RelightBufferViews & rv = cloud.relightViews();
        if (!rv.normals.valid() || !rv.baseColorRoughness.valid() || !rv.metallic.valid()
            || !rv.visibility.valid())
        {
            detail = "a relightable stream is not on the GPU";
            return false;
        }
        const auto normals = toVector<glm::vec4>(readView(device, rv.normals, count * sizeof(glm::vec4)));
        const auto bcr = toVector<glm::vec4>(readView(device, rv.baseColorRoughness, count * sizeof(glm::vec4)));
        const auto metallic = toVector<float>(readView(device, rv.metallic, count * sizeof(float)));
        const auto visibility = toVector<float>(
            readView(device, rv.visibility, count * kVisibilityCoefficients * sizeof(float)));
        if (!sameBits(normals, h.normals) || !sameBits(bcr, h.baseColorRoughness)
            || !sameBits(metallic, h.metallic) || !sameBits(visibility, h.visibility))
        {
            detail = "a relightable stream on the GPU differs from the host copy";
            return false;
        }
        // downloadFromGPU() refills the visibility host vector from the GPU;
        // it was zeroed above, so this is a GPU readback.
        if (!sameBits(cloud.visibility(), h.visibility))
        {
            detail = "downloadFromGPU() did not refill the visibility stream";
            return false;
        }
    }
    detail = "count " + std::to_string(count) + " degree " + std::to_string(degree) + " capacity "
             + std::to_string(expectedCapacity) + (relight ? " +relight" : "");
    return true;
}

int testRoundTrips(Device & device)
{
    int failures = 0;
    uint32_t seed = 1;
    for (uint32_t degree = 0; degree <= 3; ++degree)
    {
        for (const bool relight : {false, true})
        {
            // count 1 with capacity 1, a larger count with capacity == count,
            // and a larger count with spare capacity.
            const struct Case
            {
                size_t count;
                size_t capacity;
            } cases[] = {{1, 0}, {777, 0}, {777, 1500}};
            for (const Case & c : cases)
            {
                std::string detail;
                const bool ok = roundTrip(device, c.count, degree, c.capacity, relight, seed++, detail);
                check(ok, "upload -> download bit-exact: " + detail);
                failures += ok ? 0 : 1;
            }
        }
    }
    return failures;
}

// Each refusal must throw and name the problem. Returns the failure count.
template <typename Action>
int expectThrow(const std::string & what, const std::string & needle, Action action)
{
    try
    {
        action();
    }
    catch (const std::exception & error)
    {
        const std::string message = error.what();
        const bool ok = message.find(needle) != std::string::npos;
        check(ok, what + " (" + message + ")");
        return ok ? 0 : 1;
    }
    check(false, what + " (no exception)");
    return 1;
}

int testRefusals(Device & device)
{
    int failures = 0;
    failures += expectThrow("uploadToGPU refuses an empty cloud", "no Gaussians loaded", [&]
    {
        GaussianCloud cloud(device);
        cloud.initialize(0);
        cloud.uploadToGPU();
    });
    failures += expectThrow("uploadToGPU refuses positions shorter than count()", "not sized for count()", [&]
    {
        GaussianCloud cloud(device);
        cloud.initialize(10);
        cloud.positions().resize(9);
        cloud.uploadToGPU();
    });
    failures += expectThrow("uploadToGPU refuses SH shorter than count() * 16", "not sized for count()", [&]
    {
        GaussianCloud cloud(device);
        cloud.initialize(10);
        cloud.shCoefficients().resize(10 * kShCoefficientsPerGaussian - 1);
        cloud.uploadToGPU();
    });
    failures += expectThrow("uploadToGPU refuses opacities longer than count()", "not sized for count()", [&]
    {
        GaussianCloud cloud(device);
        cloud.initialize(10);
        cloud.opacities().push_back(0.0f);
        cloud.uploadToGPU();
    });
    failures += expectThrow("uploadToGPU refuses a normals stream of the wrong size", "relightable stream 'normals'", [&]
    {
        GaussianCloud cloud(device);
        cloud.initialize(10);
        cloud.normals().resize(5);
        cloud.uploadToGPU();
    });
    failures += expectThrow("uploadToGPU refuses a visibility stream of the wrong size", "relightable stream 'visibility'", [&]
    {
        GaussianCloud cloud(device);
        cloud.initialize(10);
        cloud.visibility().assign(10, 0.0f); // needs 10 * kVisibilityCoefficients
        cloud.uploadToGPU();
    });
    return failures;
}

// ---- computeBounds ------------------------------------------------------

struct Reference
{
    bool valid = false;
    glm::vec3 min{0.0f};
    glm::vec3 max{0.0f};
    glm::vec3 focusCenter{0.0f};
    float focusRadius = 0.0f;
};

// Brute force, straight from the definitions: min/max over the finite points;
// per-axis median = sorted[n / 2]; focus radius = sorted distances to that
// center at index floor(0.9 * (n - 1)); when that is zero, the bounding-sphere
// radius. Points with any non-finite component are excluded entirely.
Reference bruteForceBounds(const std::vector<glm::vec4> & positions)
{
    std::vector<glm::dvec3> finite;
    for (const glm::vec4 & p : positions)
    {
        if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z))
        {
            finite.emplace_back(p.x, p.y, p.z);
        }
    }
    Reference r;
    if (finite.empty())
    {
        return r;
    }
    r.valid = true;
    r.min = glm::vec3(std::numeric_limits<float>::max());
    r.max = glm::vec3(std::numeric_limits<float>::lowest());
    for (const glm::dvec3 & p : finite)
    {
        for (int a = 0; a < 3; ++a)
        {
            r.min[a] = std::min(r.min[a], static_cast<float>(p[a]));
            r.max[a] = std::max(r.max[a], static_cast<float>(p[a]));
        }
    }
    for (int a = 0; a < 3; ++a)
    {
        std::vector<double> component;
        component.reserve(finite.size());
        for (const glm::dvec3 & p : finite)
        {
            component.push_back(p[a]);
        }
        std::sort(component.begin(), component.end());
        r.focusCenter[a] = static_cast<float>(component[component.size() / 2]);
    }
    const glm::dvec3 center(r.focusCenter);
    std::vector<double> distances;
    distances.reserve(finite.size());
    for (const glm::dvec3 & p : finite)
    {
        distances.push_back(glm::length(p - center));
    }
    std::sort(distances.begin(), distances.end());
    const size_t index = static_cast<size_t>(0.9 * static_cast<double>(distances.size() - 1));
    r.focusRadius = static_cast<float>(distances[index]);
    if (!(r.focusRadius > 0.0f))
    {
        r.focusRadius = glm::length(r.max - r.min) * 0.5f;
    }
    return r;
}

// Measures the worst relative error of computeBounds() against brute force and
// returns false if any value is out of tolerance.
//
// Tolerance: the focus radius is a float distance computed as float(p - c),
// then squared, summed and rooted in float. Each of those roundings is at most
// half an ulp of its result, so the relative error stays below about 4 ulp
// (FLT_EPSILON); the brute force is done in double. 8 ulp is the bound; the
// measured worst is printed.
bool compareBounds(const std::string & label, GaussianCloud & cloud, double & worstRelative)
{
    const Reference ref = bruteForceBounds(cloud.positions());
    cloud.computeBounds();
    const SceneBounds & b = cloud.bounds();
    bool ok = b.valid == ref.valid;
    if (ok && ref.valid)
    {
        ok = bitEqual(b.min.x, ref.min.x) && bitEqual(b.min.y, ref.min.y)
             && bitEqual(b.min.z, ref.min.z) && bitEqual(b.max.x, ref.max.x)
             && bitEqual(b.max.y, ref.max.y) && bitEqual(b.max.z, ref.max.z);
        ok = ok && bitEqual(b.focusCenter.x, ref.focusCenter.x)
             && bitEqual(b.focusCenter.y, ref.focusCenter.y)
             && bitEqual(b.focusCenter.z, ref.focusCenter.z);
        const double denominator = std::max(1e-30, static_cast<double>(std::fabs(ref.focusRadius)));
        const double relative = std::fabs(static_cast<double>(b.focusRadius) - ref.focusRadius) / denominator;
        worstRelative = std::max(worstRelative, relative);
        ok = ok && relative <= 8.0 * static_cast<double>(FLT_EPSILON);
    }
    check(ok, "computeBounds vs brute force: " + label + " (valid " + (b.valid ? "true" : "false")
                  + ", focusRadius " + std::to_string(b.focusRadius) + ", reference "
                  + std::to_string(ref.focusRadius) + ")");
    return ok;
}

int testBounds(Device & device)
{
    int failures = 0;
    double worst = 0.0;

    // Random clouds, with whole-point non-finite values mixed in.
    for (const size_t count : {size_t(2), size_t(3), size_t(1000), size_t(20001), size_t(1000000)})
    {
        // Positions only: the bounds never read the other streams, and a
        // million-point SH stream would be 256 MB for nothing.
        std::mt19937 rng(static_cast<uint32_t>(count));
        std::normal_distribution<float> normal(0.0f, 10.0f);
        GaussianCloud cloud(device);
        cloud.initialize(count);
        for (auto & p : cloud.positions())
        {
            p = glm::vec4(normal(rng), normal(rng), normal(rng), 0.0f);
        }
        for (size_t i = 0; i < count; i += 97)
        {
            const float bad = (i % 2 == 0) ? std::numeric_limits<float>::quiet_NaN()
                                           : std::numeric_limits<float>::infinity();
            cloud.positions()[i] = glm::vec4(bad, bad, bad, 0.0f);
        }
        failures += compareBounds("random " + std::to_string(count) + " with NaN/inf points", cloud, worst) ? 0 : 1;
    }

    // One point: the box is a point and the focus radius falls back to 0.
    {
        GaussianCloud cloud(device);
        cloud.initialize(1);
        cloud.positions()[0] = glm::vec4(1.0f, -2.0f, 3.0f, 0.0f);
        cloud.computeBounds();
        const SceneBounds & b = cloud.bounds();
        const bool ok = b.valid && b.min == glm::vec3(1.0f, -2.0f, 3.0f) && b.max == b.min
                        && b.focusCenter == b.min && b.focusRadius == 0.0f;
        check(ok, "one point: valid, min == max == focusCenter, focusRadius 0");
        failures += ok ? 0 : 1;
    }

    // Degenerate: 95% of the points at one place. The 90th percentile is zero,
    // so the focus radius falls back to the bounding-sphere radius.
    {
        GaussianCloud cloud(device);
        cloud.initialize(1000);
        for (size_t i = 0; i < 1000; ++i)
        {
            cloud.positions()[i] = (i < 950) ? glm::vec4(0.5f, 0.5f, 0.5f, 0.0f)
                                             : glm::vec4(10.0f, 0.5f, 0.5f, 0.0f);
        }
        failures += compareBounds("95% of points coincident (fallback radius)", cloud, worst) ? 0 : 1;
        const bool fallback = cloud.bounds().focusRadius > 0.0f;
        check(fallback, "degenerate cloud: focusRadius falls back to a positive radius");
        failures += fallback ? 0 : 1;
    }

    // All non-finite: no finite point, so no bounds.
    {
        GaussianCloud cloud(device);
        cloud.initialize(5);
        for (auto & p : cloud.positions())
        {
            p = glm::vec4(std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity(), 0.0f, 0.0f);
        }
        cloud.computeBounds();
        const bool ok = !cloud.bounds().valid;
        check(ok, "all-NaN positions: bounds.valid is false");
        failures += ok ? 0 : 1;
    }

    // Empty: count 0, no bounds.
    {
        GaussianCloud cloud(device);
        cloud.initialize(0);
        cloud.computeBounds();
        const bool ok = !cloud.bounds().valid;
        check(ok, "empty cloud: bounds.valid is false");
        failures += ok ? 0 : 1;
    }

    std::printf("computeBounds worst relative focusRadius error: %.3g (tolerance %.3g = 8 ulp)\n",
                worst, 8.0 * static_cast<double>(FLT_EPSILON));
    return failures;
}

} // namespace

// Entry point called by gaussian_cloud_test's main. Returns 0 on success.
int runHostChecks(Device & device)
{
    const int failures = testRoundTrips(device) + testRefusals(device) + testBounds(device);
    std::printf("host checks: %s (%d failure%s)\n", failures == 0 ? "OK" : "FAILED", failures,
                failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
