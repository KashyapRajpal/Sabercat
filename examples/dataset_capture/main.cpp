// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include <glm/glm.hpp>
#include <glm/gtc/matrix_inverse.hpp>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "sabercat/camera.h"
#include "sabercat/device.h"
#include "sabercat/forward_pass.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/render_config.h"
#include "sabercat/scene_loader.h"

using namespace sabercat;

namespace
{

struct Options
{
    std::filesystem::path input;
    std::filesystem::path output;
    uint32_t views = 32;
    uint32_t size = 64;
    float verticalFovDegrees = 55.0f;
    float elevationDegrees = 12.0f;
    float margin = 1.15f;
};

uint32_t parsePositive(const std::string & text, const char * name)
{
    size_t consumed = 0;
    const unsigned long value = std::stoul(text, &consumed);
    if (consumed != text.size() || value == 0 || value > std::numeric_limits<uint32_t>::max())
    {
        throw std::runtime_error(std::string(name) + " must be a positive integer");
    }
    return static_cast<uint32_t>(value);
}

float parseFloat(const std::string & text, const char * name)
{
    size_t consumed = 0;
    const float value = std::stof(text, &consumed);
    if (consumed != text.size() || !std::isfinite(value))
    {
        throw std::runtime_error(std::string(name) + " must be finite");
    }
    return value;
}

Options parseOptions(int argc, char ** argv)
{
    if (argc < 3)
    {
        throw std::runtime_error(
            "usage: sabercat_dataset_capture <scene.ply|scene.spz> <output-dir> "
            "[--views N] [--size N] [--vertical-fov DEG] [--elevation DEG] [--margin X]");
    }
    Options options;
    options.input = argv[1];
    options.output = argv[2];
    for (int i = 3; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (i + 1 >= argc)
        {
            throw std::runtime_error("missing value for " + arg);
        }
        const std::string value = argv[++i];
        if (arg == "--views")
            options.views = parsePositive(value, "--views");
        else if (arg == "--size")
            options.size = parsePositive(value, "--size");
        else if (arg == "--vertical-fov")
            options.verticalFovDegrees = parseFloat(value, "--vertical-fov");
        else if (arg == "--elevation")
            options.elevationDegrees = parseFloat(value, "--elevation");
        else if (arg == "--margin")
            options.margin = parseFloat(value, "--margin");
        else
            throw std::runtime_error("unknown option: " + arg);
    }
    if (options.verticalFovDegrees <= 1.0f || options.verticalFovDegrees >= 170.0f)
        throw std::runtime_error("--vertical-fov must be in (1, 170)");
    if (options.elevationDegrees <= -80.0f || options.elevationDegrees >= 80.0f)
        throw std::runtime_error("--elevation must be in (-80, 80)");
    if (options.margin < 1.0f)
        throw std::runtime_error("--margin must be at least 1");
    return options;
}

glm::mat4 lookAtCameraToWorld(const glm::vec3 & eye)
{
    const glm::vec3 forward = glm::normalize(-eye);
    const glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0.0f, 1.0f, 0.0f)));
    const glm::vec3 up = glm::cross(-forward, right);
    glm::mat4 result(1.0f);
    result[0] = glm::vec4(right, 0.0f);
    result[1] = glm::vec4(up, 0.0f);
    result[2] = glm::vec4(-forward, 0.0f);
    result[3] = glm::vec4(eye, 1.0f);
    return result;
}

std::string jsonEscape(const std::string & value)
{
    std::string escaped;
    for (const char ch : value)
    {
        switch (ch)
        {
        case '\\': escaped += "\\\\"; break;
        case '"': escaped += "\\\""; break;
        case '\n': escaped += "\\n"; break;
        case '\r': escaped += "\\r"; break;
        case '\t': escaped += "\\t"; break;
        default: escaped += ch; break;
        }
    }
    return escaped;
}

template<typename T>
void writeBinary(const std::filesystem::path & path, const std::vector<T> & values)
{
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output)
        throw std::runtime_error("could not open output: " + path.string());
    output.write(reinterpret_cast<const char *>(values.data()),
                 static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!output)
        throw std::runtime_error("could not write output: " + path.string());
}

std::string viewName(uint32_t index, const char * suffix)
{
    std::ostringstream stream;
    stream << "view-" << std::setw(3) << std::setfill('0') << index << suffix;
    return stream.str();
}

void writeMatrix(std::ostream & output, const glm::mat4 & matrix)
{
    output << '[';
    for (int row = 0; row < 4; ++row)
    {
        for (int column = 0; column < 4; ++column)
        {
            if (row != 0 || column != 0)
                output << ',';
            output << matrix[column][row];
        }
    }
    output << ']';
}

} // namespace

int main(int argc, char ** argv)
{
    Device device;
    try
    {
        const Options options = parseOptions(argc, argv);
        std::filesystem::create_directories(options.output);
        device.initialize({}, false);

        auto cloud = loadScene(device, options.input.string());
        const SceneBounds sourceBounds = cloud->bounds();
        if (!sourceBounds.valid || !std::isfinite(sourceBounds.focusRadius)
            || sourceBounds.focusRadius <= 0.0f)
            throw std::runtime_error("scene has invalid focus bounds");

        const size_t sourceCount = cloud->count();
        const uint32_t sourceShDegree = cloud->shDegree();
        const bool antialiased = cloud->antialiased();
        for (auto & position : cloud->positions())
        {
            const glm::vec3 normalized =
                (glm::vec3(position) - sourceBounds.focusCenter) / sourceBounds.focusRadius;
            position = glm::vec4(normalized, position.w);
        }
        for (auto & scale : cloud->scales())
        {
            scale.x /= sourceBounds.focusRadius;
            scale.y /= sourceBounds.focusRadius;
            scale.z /= sourceBounds.focusRadius;
        }
        cloud->computeBounds();
        cloud->uploadToGPU();

        const float pi = 3.14159265358979323846f;
        const float fovRadians = options.verticalFovDegrees * pi / 180.0f;
        const float fy = 0.5f * static_cast<float>(options.size) / std::tan(0.5f * fovRadians);
        const float fx = fy;
        const float cx = 0.5f * static_cast<float>(options.size - 1);
        const float cy = cx;
        const float distance = options.margin / std::sin(0.5f * fovRadians);
        const float elevation = options.elevationDegrees * pi / 180.0f;

        ForwardPass pass(device, options.size, options.size, SHADER_DIR);
        std::vector<glm::mat4> cameras;
        cameras.reserve(options.views);
        for (uint32_t index = 0; index < options.views; ++index)
        {
            const float azimuth = 2.0f * pi * static_cast<float>(index)
                                  / static_cast<float>(options.views);
            const float horizontal = distance * std::cos(elevation);
            const glm::vec3 eye(horizontal * std::sin(azimuth),
                                distance * std::sin(elevation),
                                horizontal * std::cos(azimuth));
            const glm::mat4 cameraToWorld = lookAtCameraToWorld(eye);
            cameras.push_back(cameraToWorld);

            CameraUniforms uniforms{};
            uniforms.worldToCamera = glm::inverse(cameraToWorld);
            uniforms.fx = fx;
            uniforms.fy = fy;
            uniforms.cx = cx;
            uniforms.cy = cy;
            uniforms.imageWidth = options.size;
            uniforms.imageHeight = options.size;
            uniforms.tilesX = (options.size + kTileSize - 1) / kTileSize;
            uniforms.tilesY = uniforms.tilesX;
            uniforms.gaussianCount = static_cast<uint32_t>(sourceCount);
            uniforms.shDegree = 0;
            uniforms.renderFlags = antialiased ? kRenderFlagAntialiased : 0u;
            uniforms.cameraPositionWorld = glm::vec4(eye, 1.0f);

            pass.run(cloud->views(), uniforms, ForwardStage::Rasterize, ForwardMode::Viewer);
            writeBinary(options.output / viewName(index, ".rgba32f"), pass.downloadOutputImage());
            writeBinary(options.output / viewName(index, ".depth32f"), pass.downloadDepthImage());
            std::cout << "captured " << (index + 1) << '/' << options.views << '\r' << std::flush;
        }
        std::cout << std::endl;

        std::ofstream metadata(options.output / "capture.json", std::ios::trunc);
        if (!metadata)
            throw std::runtime_error("could not create capture.json");
        metadata << std::setprecision(9);
        metadata << "{\n"
                 << "  \"schemaVersion\": 1,\n"
                 << "  \"generatorVersion\": \"sabercat-dataset-capture-v1\",\n"
                 << "  \"sourceFileName\": \""
                 << jsonEscape(options.input.filename().string()) << "\",\n"
                 << "  \"sourceGaussianCount\": " << sourceCount << ",\n"
                 << "  \"sourceSHDegree\": " << sourceShDegree << ",\n"
                 << "  \"renderedSHDegree\": 0,\n"
                 << "  \"antialiased\": " << (antialiased ? "true" : "false") << ",\n"
                 << "  \"width\": " << options.size << ",\n"
                 << "  \"height\": " << options.size << ",\n"
                 << "  \"depthSemantics\": \"frontmost-contributor\",\n"
                 << "  \"normalization\": {\"center\": ["
                 << sourceBounds.focusCenter.x << ',' << sourceBounds.focusCenter.y << ','
                 << sourceBounds.focusCenter.z << "], \"scale\": "
                 << sourceBounds.focusRadius << "},\n"
                 << "  \"views\": [\n";
        for (uint32_t index = 0; index < options.views; ++index)
        {
            metadata << "    {\"id\": \"view-" << std::setw(3) << std::setfill('0') << index
                     << "\", \"rgba\": \"" << viewName(index, ".rgba32f")
                     << "\", \"depth\": \"" << viewName(index, ".depth32f")
                     << "\", \"fx\": " << fx << ", \"fy\": " << fy
                     << ", \"cx\": " << cx << ", \"cy\": " << cy
                     << ", \"cameraToWorld\": ";
            writeMatrix(metadata, cameras[index]);
            metadata << '}' << (index + 1 == options.views ? "\n" : ",\n");
        }
        metadata << "  ]\n}\n";
        if (!metadata)
            throw std::runtime_error("could not write capture.json");

        return 0;
    }
    catch (const std::exception & error)
    {
        std::cerr << "dataset_capture FAILED: " << error.what() << std::endl;
        return 1;
    }
}
