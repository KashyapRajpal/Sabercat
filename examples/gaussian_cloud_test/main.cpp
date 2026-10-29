// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Smoke test for PLYReader + GaussianCloud.
//
// Hand-writes a small synthetic 3DGS .ply (fully self-contained -- no
// network access needed to fetch a real pretrained scene), loads it via
// PLYReader, and checks the activation math (sigmoid opacity, exp scale),
// quaternion -> covariance derivation, and channel-major SH unpacking
// against independently hand-computed expected values. Then uploads to the
// GPU and checks the buffers exist with the right byte size.

#include <glm/glm.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "sabercat/approx.h"
#include "sabercat/buffer.h"
#include "sabercat/device.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/ply_reader.h"

using namespace sabercat;

// host_checks.cpp: upload/download round trips, refusals and computeBounds.
int runHostChecks(Device & device);

namespace
{

constexpr int kFRestPerChannel = 3; // SH degree 1: 3 higher-order coeffs/channel
constexpr int kFRestCount = kFRestPerChannel * 3;

struct SyntheticVertex
{
    float x, y, z;
    float nx = 0.0f, ny = 0.0f, nz = 0.0f;
    float f_dc[3];
    float f_rest[kFRestCount];
    float opacity;
    float scale[3];
    float rot[4];
};
// Member order above matches the .ply property order written below exactly,
// and every member is a plain float (uniform 4-byte alignment, no padding
// gaps) -- so the whole struct can be written in one shot per vertex.
static_assert(sizeof(SyntheticVertex) == (6 + 3 + kFRestCount + 1 + 3 + 4) * sizeof(float),
              "SyntheticVertex must be tightly packed with no padding");

void writeSyntheticPLY(const std::string & path, const std::vector<SyntheticVertex> & verts)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file << "ply\n";
    file << "format binary_little_endian 1.0\n";
    file << "element vertex " << verts.size() << "\n";
    file << "property float x\n";
    file << "property float y\n";
    file << "property float z\n";
    file << "property float nx\n";
    file << "property float ny\n";
    file << "property float nz\n";
    file << "property float f_dc_0\n";
    file << "property float f_dc_1\n";
    file << "property float f_dc_2\n";
    for (int i = 0; i < kFRestCount; ++i)
    {
        file << "property float f_rest_" << i << "\n";
    }
    file << "property float opacity\n";
    file << "property float scale_0\n";
    file << "property float scale_1\n";
    file << "property float scale_2\n";
    file << "property float rot_0\n";
    file << "property float rot_1\n";
    file << "property float rot_2\n";
    file << "property float rot_3\n";
    file << "end_header\n";

    for (const auto & v : verts)
    {
        file.write(reinterpret_cast<const char *>(&v), sizeof(SyntheticVertex));
    }
}

} // namespace

// compressed_ply.cpp: the SuperSplat / PlayCanvas compressed .ply variant.
int testCompressedPly(Device & device);

int main()
{
    // Large enough to exercise the multithreaded parse path in
    // parallelFor() (default minChunkSize is 50000), not just correctness
    // on a handful of vertices.
    std::vector<SyntheticVertex> verts(300000);
    for (size_t i = 0; i < verts.size(); ++i)
    {
        SyntheticVertex & v = verts[i];
        float fi = static_cast<float>(i);
        v.x = fi;
        v.y = fi * 2.0f;
        v.z = fi * 3.0f;

        v.f_dc[0] = 10.0f * fi + 1.0f;
        v.f_dc[1] = 10.0f * fi + 2.0f;
        v.f_dc[2] = 10.0f * fi + 3.0f;

        // Channel-major storage: R coeffs, then G coeffs, then B coeffs.
        for (int k = 0; k < kFRestPerChannel; ++k)
        {
            float fk = static_cast<float>(k);
            v.f_rest[k] = 100.0f + fi + fk;                        // R
            v.f_rest[kFRestPerChannel + k] = 200.0f + fi + fk;     // G
            v.f_rest[2 * kFRestPerChannel + k] = 300.0f + fi + fk; // B
        }

        v.opacity = (i == 0) ? 0.0f : 2.0f + fi; // raw, pre-sigmoid

        if (i == 0)
        {
            v.scale[0] = v.scale[1] = v.scale[2] = std::log(1.0f); // exp -> (1,1,1)
            v.rot[0] = 1.0f;
            v.rot[1] = v.rot[2] = v.rot[3] = 0.0f; // identity quaternion (w,x,y,z)
        }
        else if (i == 1)
        {
            v.scale[0] = std::log(2.0f);
            v.scale[1] = std::log(1.0f);
            v.scale[2] = std::log(1.0f);
            v.rot[0] = 1.0f;
            v.rot[1] = v.rot[2] = v.rot[3] = 0.0f; // identity
        }
        else
        {
            v.scale[0] = std::log(1.5f);
            v.scale[1] = std::log(0.5f);
            v.scale[2] = std::log(1.0f);
            // 90-degree rotation about Z.
            float half = 0.78539816339f; // 45 degrees in radians
            v.rot[0] = std::cos(half);
            v.rot[1] = 0.0f;
            v.rot[2] = 0.0f;
            v.rot[3] = std::sin(half);
        }
    }

    const std::string plyPath = "gaussian_cloud_test_fixture.ply";
    writeSyntheticPLY(plyPath, verts);

    Device device;
    int result = 0;
    try
    {
        device.initialize(/*instanceExtensions=*/{}, /*enableValidation=*/true);

        auto parseStart = std::chrono::steady_clock::now();
        auto cloud = PLYReader::read(device, plyPath);
        auto parseMs =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - parseStart)
                .count();
        std::cout << "Parsed " << cloud->count() << " Gaussians in " << parseMs << " ms\n";

        if (cloud->count() != verts.size())
        {
            std::cerr << "Gaussian count mismatch: expected " << verts.size() << ", got "
                      << cloud->count() << std::endl;
            throw std::runtime_error("count mismatch");
        }

        auto & positions = cloud->positions();
        for (size_t i = 0; i < verts.size(); ++i)
        {
            glm::vec3 expected(verts[i].x, verts[i].y, verts[i].z);
            if (!approxEqualXYZ(positions[i], expected))
            {
                std::cerr << "Position mismatch at " << i << std::endl;
                throw std::runtime_error("position mismatch");
            }
        }

        auto & opacities = cloud->opacities();
        for (size_t i = 0; i < verts.size(); ++i)
        {
            float expected = 1.0f / (1.0f + std::exp(-verts[i].opacity));
            if (!approxEqual(opacities[i], expected))
            {
                std::cerr << "Opacity activation mismatch at " << i << ": expected " << expected
                          << ", got " << opacities[i] << std::endl;
                throw std::runtime_error("opacity mismatch");
            }
        }

        auto & scales = cloud->scales();
        for (size_t i = 0; i < verts.size(); ++i)
        {
            glm::vec3 expected(std::exp(verts[i].scale[0]),
                               std::exp(verts[i].scale[1]),
                               std::exp(verts[i].scale[2]));
            if (!approxEqualXYZ(scales[i], expected))
            {
                std::cerr << "Scale activation mismatch at " << i << std::endl;
                throw std::runtime_error("scale mismatch");
            }
        }

        // Rotations: the .ply stores (w,x,y,z) but the canonical GPU stream
        // is xyzw, so this checks the REORDER as well as normalization --
        // a silent transposition here would rotate every Gaussian wrongly
        // and be very hard to spot in a rendered image.
        auto & rotations = cloud->rotations();
        for (size_t i = 0; i < verts.size(); ++i)
        {
            const float * r = verts[i].rot; // r[0]=w, r[1]=x, r[2]=y, r[3]=z
            float length = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3]);
            glm::vec4 expected(r[1] / length, r[2] / length, r[3] / length, r[0] / length);
            if (!approxEqual(rotations[i], expected))
            {
                std::cerr << "Rotation xyzw mismatch at " << i << ": expected (" << expected.x
                          << ", " << expected.y << ", " << expected.z << ", " << expected.w
                          << "), got (" << rotations[i].x << ", " << rotations[i].y << ", "
                          << rotations[i].z << ", " << rotations[i].w << ")" << std::endl;
                throw std::runtime_error("rotation mismatch");
            }
            if (!approxEqual(glm::length(rotations[i]), 1.0f))
            {
                std::cerr << "Rotation not normalized at " << i << std::endl;
                throw std::runtime_error("rotation not normalized");
            }
        }

        // SH coefficients: exact comparison is safe here (no activation
        // math, pure byte passthrough).
        auto & sh = cloud->shCoefficients();
        for (size_t i = 0; i < verts.size(); ++i)
        {
            glm::vec4 * base = &sh[i * kShCoefficientsPerGaussian];
            glm::vec3 expectedDC(verts[i].f_dc[0], verts[i].f_dc[1], verts[i].f_dc[2]);
            if (!approxEqualXYZ(base[0], expectedDC, 0.0f))
            {
                std::cerr << "SH DC mismatch at " << i << std::endl;
                throw std::runtime_error("sh dc mismatch");
            }
            for (int k = 0; k < kFRestPerChannel; ++k)
            {
                float fi = static_cast<float>(i);
                float fk = static_cast<float>(k);
                glm::vec3 expected(100.0f + fi + fk, 200.0f + fi + fk, 300.0f + fi + fk);
                if (!approxEqualXYZ(base[1 + k], expected, 0.0f))
                {
                    std::cerr << "SH coefficient " << (1 + k) << " mismatch at Gaussian " << i
                              << std::endl;
                    throw std::runtime_error("sh coefficient mismatch");
                }
            }
            for (int k = 1 + kFRestPerChannel; k < kShCoefficientsPerGaussian; ++k)
            {
                if (!approxEqualXYZ(base[k], glm::vec3(0.0f), 0.0f))
                {
                    std::cerr << "Expected zero-fill at SH slot " << k << " for Gaussian " << i
                              << std::endl;
                    throw std::runtime_error("sh zero-fill mismatch");
                }
            }
        }

        auto uploadStart = std::chrono::steady_clock::now();
        cloud->uploadToGPU();
        auto uploadMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()
                                                                  - uploadStart)
                            .count();
        std::cout << "Uploaded to GPU in " << uploadMs << " ms\n";

        // Every stream must land at its canonical stride -- a stride
        // mismatch is exactly what would silently mis-bind in std430.
        const size_t n = verts.size();
        struct StreamCheck
        {
            const char * name;
            const Buffer * buffer;
            size_t expectedBytes;
        };
        const StreamCheck streams[] = {
            {"means", cloud->positionBuffer(), n * kMeansStride},
            {"scales", cloud->scaleBuffer(), n * kScalesStride},
            {"rotations", cloud->rotationBuffer(), n * kRotationsStride},
            {"opacity", cloud->opacityBuffer(), n * kOpacityStride},
            {"sh", cloud->colorBuffer(), n * kShCoefficientsPerGaussian * kShStride},
        };
        for (const auto & s : streams)
        {
            if (!s.buffer || s.buffer->size() != s.expectedBytes)
            {
                std::cerr << "Stream '" << s.name << "' missing or wrong size: expected "
                          << s.expectedBytes << " bytes, got " << (s.buffer ? s.buffer->size() : 0)
                          << std::endl;
                throw std::runtime_error("stream size mismatch");
            }
        }

        // The views the compute passes will actually bind.
        auto views = cloud->views();
        if (views.gaussianCount != n || views.means.stride != kMeansStride
            || views.rotations.stride != kRotationsStride
            || views.shCoefficients.stride != kShStride || views.confidence.valid())
        {
            std::cerr << "Canonical buffer views are wrong" << std::endl;
            throw std::runtime_error("view mismatch");
        }

        std::cout << "GaussianCloud test OK: " << cloud->count()
                  << " Gaussians parsed, activated, and uploaded correctly\n";

        if (testCompressedPly(device) != 0)
        {
            throw std::runtime_error("compressed PLY mismatch");
        }
        if (runHostChecks(device) != 0)
        {
            throw std::runtime_error("host checks failed");
        }
    }
    catch (const std::exception & e)
    {
        std::cerr << "GaussianCloud test FAILED: " << e.what() << std::endl;
        result = 1;
    }

    device.shutdown();
    std::remove(plyPath.c_str());
    return result;
}
