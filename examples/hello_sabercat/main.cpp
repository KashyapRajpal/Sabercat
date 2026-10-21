// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// hello_sabercat: the smallest real Sabercat program.
//
// Creates a headless Renderer (it owns its Vulkan device; no window), loads a
// scene file, frames it with a camera, renders one image and writes it as a
// binary PPM (P6).
//
//   sabercat_hello_sabercat <scene> <out.ppm>

#include <glm/glm.hpp>

#include <algorithm>
#include <cstdio>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

#include "sabercat/camera.h"
#include "sabercat/renderer.h"
#include "sabercat/scene_bounds.h"

#include "../common/shader_dir.h"

using namespace sabercat;

namespace
{

constexpr uint32_t kWidth = 640;
constexpr uint32_t kHeight = 480;

// readImage() returns RGBA floats, row-major. The colour is already composited
// over a black background, so RGB goes straight to 8-bit.
bool writePPM(const std::string & path, const std::vector<float> & rgba)
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
    {
        return false;
    }
    out << "P6\n" << kWidth << " " << kHeight << "\n255\n";

    const size_t pixels = static_cast<size_t>(kWidth) * kHeight;
    std::vector<unsigned char> bytes(pixels * 3);
    for (size_t i = 0; i < pixels; ++i)
    {
        for (size_t c = 0; c < 3; ++c)
        {
            const float v = std::clamp(rgba[i * 4 + c], 0.0f, 1.0f);
            bytes[i * 3 + c] = static_cast<unsigned char>(v * 255.0f + 0.5f);
        }
    }
    out.write(reinterpret_cast<const char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return out.good();
}

} // namespace

int main(int argc, char ** argv)
{
    if (argc != 3)
    {
        std::fprintf(stderr, "usage: %s <scene> <out.ppm>\n", argv[0]);
        return 2;
    }

    try
    {
        // initialize() creates the Vulkan instance and device. The shader
        // directory holds the compiled forward-rasterizer SPIR-V.
        Renderer renderer(kWidth, kHeight);
        if (!renderer.initialize({}, /*enableValidation=*/true, testShaderDir()))
        {
            std::fprintf(stderr, "error: Vulkan initialization failed\n");
            return 1;
        }

        // Throws std::runtime_error for an unsupported or unreadable file.
        renderer.loadScene(argv[1]);

        // Frame the scene: look down -Z from +Z at the robust focus centre,
        // three focus radii out. With focal length == width the frame's
        // vertical half-angle is about 21 degrees, and the focus sphere
        // subtends about 18, so it fits.
        const SceneBounds bounds = renderer.sceneBounds();
        if (!bounds.valid || bounds.focusRadius <= 0.0f)
        {
            std::fprintf(stderr, "error: scene has no usable bounds\n");
            return 1;
        }

        Camera camera;
        camera.width = kWidth;
        camera.height = kHeight;
        camera.fx = camera.fy = static_cast<float>(kWidth);
        camera.cx = 0.5f * (kWidth - 1);
        camera.cy = 0.5f * (kHeight - 1);
        camera.cameraToWorld = glm::mat4(1.0f);
        camera.cameraToWorld[3] =
            glm::vec4(bounds.focusCenter + glm::vec3(0.0f, 0.0f, 3.0f * bounds.focusRadius), 1.0f);
        renderer.setCamera(camera);

        // Viewer mode is the default: the real-time path, no backward state.
        renderer.forward();
        const std::vector<float> rgba = renderer.readImage();

        if (!writePPM(argv[2], rgba))
        {
            std::fprintf(stderr, "error: cannot write %s\n", argv[2]);
            return 1;
        }
        std::printf("wrote %s (%ux%u)\n", argv[2], kWidth, kHeight);
        return 0;
    }
    catch (const std::exception & e)
    {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
