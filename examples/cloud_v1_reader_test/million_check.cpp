// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Scale check for the cloud-v1 writer and reader (WP6): a 1,000,000-Gaussian
// degree-3 cloud with every relight stream is written with CloudV1Writer and read
// back with CloudV1Reader from the directory. cloud-v1 stores float32 as raw
// bytes, so every host stream must come back BIT-EXACT, including the relight
// sidecar. The write and read are timed.

#include <glm/glm.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "sabercat/cloud_v1_reader.h"
#include "sabercat/cloud_v1_writer.h"
#include "sabercat/device.h"
#include "sabercat/file_map.h"
#include "sabercat/gaussian_cloud.h"
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

template <typename T>
bool sameBits(const std::vector<T> & a, const std::vector<T> & b)
{
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

std::shared_ptr<GaussianCloud> makeMillion(Device & device, size_t count)
{
    auto cloud = std::make_shared<GaussianCloud>(device);
    cloud->initialize(count);
    std::mt19937 rng(7777u);
    std::uniform_real_distribution<float> unit(-1.0f, 1.0f);
    std::uniform_real_distribution<float> positive(0.01f, 1.0f);
    for (size_t i = 0; i < count; ++i)
    {
        cloud->positions()[i] = glm::vec4(10.0f * unit(rng), 10.0f * unit(rng), 10.0f * unit(rng), 0.0f);
        cloud->scales()[i] = glm::vec4(positive(rng), positive(rng), positive(rng), 0.0f);
        // Unit quaternions: the reader passes these through bit-exact (it
        // renormalizes only a quaternion visibly off the unit sphere).
        cloud->rotations()[i] = glm::normalize(glm::vec4(unit(rng), unit(rng), unit(rng), unit(rng)));
        cloud->opacities()[i] = positive(rng);
        for (int k = 0; k < kShCoefficientsPerGaussian; ++k)
        {
            cloud->shCoefficients()[i * kShCoefficientsPerGaussian + k] = glm::vec4(unit(rng), unit(rng), unit(rng), 0.0f);
        }
    }
    cloud->setShDegree(3);
    cloud->normals().resize(count);
    cloud->baseColorRoughness().resize(count);
    cloud->metallic().resize(count);
    cloud->visibility().resize(count * kVisibilityCoefficients);
    for (size_t i = 0; i < count; ++i)
    {
        cloud->normals()[i] = glm::vec4(unit(rng), unit(rng), unit(rng), 0.0f);
        cloud->baseColorRoughness()[i] = glm::vec4(positive(rng), positive(rng), positive(rng), positive(rng));
        cloud->metallic()[i] = positive(rng);
    }
    for (float & v : cloud->visibility())
    {
        v = unit(rng);
    }
    cloud->computeBounds();
    return cloud;
}

} // namespace

// Entry point called by cloud_v1_reader_test's main. Returns the failure count.
int runMillionCloudV1(Device & device, const fs::path & dir)
{
    g_failures = 0;
    const size_t count = 1000000;
    const auto source = makeMillion(device, count);
    const fs::path directory = dir / "million_cloud_v1";

    // cloud-v1 holds the canonical streams only; the relight streams travel in
    // the sidecar (RelightSidecar), which the reader applies on top.
    const auto writeStart = std::chrono::steady_clock::now();
    CloudV1Writer::write(*source, directory.string());
    const FileMap relightFiles = RelightSidecar::encode(*source);
    const double writeMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - writeStart).count();
    const auto readStart = std::chrono::steady_clock::now();
    const auto back = CloudV1Reader::read(device, directory.string());
    RelightSidecar::decode(relightFiles, *back, "million relight sidecar");
    const double readMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - readStart).count();
    std::printf("1M cloud-v1 (degree 3, relight): write %.0f ms, read %.0f ms\n", writeMs, readMs);

    check(back->count() == count && back->shDegree() == 3, "1M cloud-v1: count and shDegree round trip");
    GaussianCloud & s = *source;
    GaussianCloud & b = *back;
    struct Stream
    {
        const char * name;
        bool same;
    };
    const Stream streams[] = {
        {"positions", sameBits(s.positions(), b.positions())},
        {"scales", sameBits(s.scales(), b.scales())},
        {"rotations", sameBits(s.rotations(), b.rotations())},
        {"shCoefficients", sameBits(s.shCoefficients(), b.shCoefficients())},
        {"opacities", sameBits(s.opacities(), b.opacities())},
        {"normals", sameBits(s.normals(), b.normals())},
        {"baseColorRoughness", sameBits(s.baseColorRoughness(), b.baseColorRoughness())},
        {"metallic", sameBits(s.metallic(), b.metallic())},
        {"visibility", sameBits(s.visibility(), b.visibility())},
    };
    for (const Stream & stream : streams)
    {
        check(stream.same, std::string("1M cloud-v1: ") + stream.name + " bit-exact");
    }
    fs::remove_all(directory);
    std::printf("cloud-v1 million check: %s (%d failure%s)\n", g_failures == 0 ? "OK" : "FAILED", g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures;
}
