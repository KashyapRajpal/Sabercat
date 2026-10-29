// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/ply_writer.h"

#include <glm/glm.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

#include "sabercat/gaussian_cloud.h"
#include "sabercat/logger.h"
#include "sabercat/render_config.h"

namespace sabercat
{

namespace
{

// PLYReader's R3DG activations (ply_reader.cpp): base color is
// sigmoid * 0.77 + 0.03, roughness sigmoid * 0.9 + 0.09. The inverse must use
// the same float constants, so keep these in step with the reader.
constexpr float kBaseColorScale = 0.77f;
constexpr float kBaseColorOffset = 0.03f;
constexpr float kRoughnessScale = 0.9f;
constexpr float kRoughnessOffset = 0.09f;

// Records per write call: bounds the staging buffer to a few MB whatever the
// cloud size.
constexpr size_t kRecordsPerBlock = 65536;

// Inverse of the reader's sigmoid in double: logit(p), with p clamped to
// [eps, 1 - eps] so the result is finite. `clamped` reports whether p changed.
double logitClamped(double p, bool & clamped)
{
    const double eps = static_cast<double>(PLYWriter::kLogitEpsilon);
    clamped = !(p >= eps && p <= 1.0 - eps);
    p = std::min(std::max(p, eps), 1.0 - eps);
    return std::log(p / (1.0 - p));
}

// Inverse of `value = sigmoid(x) * scale + offset`: x = logit((value - offset) / scale).
double invertAffineSigmoid(float value, float offset, float scale, bool & clamped)
{
    const double normalized =
        (static_cast<double>(value) - static_cast<double>(offset)) / static_cast<double>(scale);
    return logitClamped(normalized, clamped);
}

void requireSize(size_t actual, size_t expected, const char * name, const std::string & path)
{
    if (actual != expected)
    {
        throw std::runtime_error("PLYWriter: " + std::string(name) + " has "
                                 + std::to_string(actual) + " entries, expected "
                                 + std::to_string(expected)
                                 + " (call downloadFromGPU() if the host copy was dropped): "
                                 + path);
    }
}

void requireFinite(double value, const char * name, size_t index, const std::string & path)
{
    if (!std::isfinite(value))
    {
        throw std::runtime_error("PLYWriter: Gaussian " + std::to_string(index) + " has a non-finite "
                                 + name + ": " + path);
    }
}

} // namespace

void PLYWriter::write(const GaussianCloud & cloud, const std::string & path)
{
    const size_t count = cloud.count();
    if (count == 0)
    {
        throw std::runtime_error("PLYWriter: cloud is empty, nothing to write: " + path);
    }
    const uint32_t shDegree = cloud.shDegree();
    if (shDegree > 3)
    {
        throw std::runtime_error("PLYWriter: SH degree " + std::to_string(shDegree)
                                 + " is above 3: " + path);
    }
    // Coefficients per color channel above the DC term: 3 for degree 1, 8 for 2, 15 for 3.
    const size_t restPerChannel = static_cast<size_t>((shDegree + 1) * (shDegree + 1) - 1);

    const GaussianCloud & host = cloud;
    const auto & positions = host.positions();
    const auto & scales = host.scales();
    const auto & rotations = host.rotations();
    const auto & shCoefficients = host.shCoefficients();
    const auto & opacities = host.opacities();
    const auto & normals = host.normals();
    const auto & baseColorRoughness = host.baseColorRoughness();
    const auto & metallic = host.metallic();
    const auto & visibility = host.visibility();

    requireSize(positions.size(), count, "positions", path);
    requireSize(scales.size(), count, "scales", path);
    requireSize(rotations.size(), count, "rotations", path);
    requireSize(opacities.size(), count, "opacities", path);
    requireSize(shCoefficients.size(), count * kShCoefficientsPerGaussian, "shCoefficients", path);
    if (!normals.empty())
    {
        requireSize(normals.size(), count, "normals", path);
    }
    if (!baseColorRoughness.empty())
    {
        requireSize(baseColorRoughness.size(), count, "baseColorRoughness", path);
    }
    if (!metallic.empty())
    {
        requireSize(metallic.size(), count, "metallic", path);
    }
    if (!visibility.empty())
    {
        requireSize(visibility.size(), count * kVisibilityCoefficients, "visibility", path);
    }

    // The R3DG marker is base_color (PLYReader keys on it), so normals and
    // visibility without base color could not be read back. Refuse rather than
    // drop them silently.
    const bool relightable = !baseColorRoughness.empty();
    if (!relightable && (!normals.empty() || !visibility.empty()))
    {
        throw std::runtime_error("PLYWriter: normals or visibility without baseColorRoughness "
                                 "cannot be read back (PLYReader needs base_color): " + path);
    }
    const bool writeVisibility = !visibility.empty();
    const bool writeMetallic = !metallic.empty();

    // Property names, in record order. Every property is a float32.
    std::vector<std::string> names = {"x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"};
    for (size_t k = 0; k < 3 * restPerChannel; ++k)
    {
        names.push_back("f_rest_" + std::to_string(k));
    }
    names.push_back("opacity");
    for (int c = 0; c < 3; ++c)
    {
        names.push_back("scale_" + std::to_string(c));
    }
    for (int c = 0; c < 4; ++c)
    {
        names.push_back("rot_" + std::to_string(c));
    }
    if (relightable)
    {
        for (int c = 0; c < 3; ++c)
        {
            names.push_back("base_color_" + std::to_string(c));
        }
        names.push_back("roughness");
    }
    if (writeVisibility)
    {
        names.push_back("visibility_dc_0");
        for (int k = 0; k < 15; ++k)
        {
            names.push_back("visibility_rest_" + std::to_string(k));
        }
    }
    if (writeMetallic)
    {
        names.push_back("metallic");
    }
    const size_t stride = names.size();
    static_assert(kVisibilityCoefficients == 16, "PLYReader reads visibility_dc_0 + 15 rest");

    std::string header = "ply\nformat binary_little_endian 1.0\n"
                         "comment Written by Sabercat PLYWriter\n";
    header += "element vertex " + std::to_string(count) + "\n";
    for (const std::string & name : names)
    {
        header += "property float " + name + "\n";
    }
    header += "end_header\n";

    // Base color / roughness values the inverse had to clamp; reported once below.
    size_t clampedRelight = 0;

    // Fills one record. `i` indexes the host vectors; every value is checked
    // here, so a throw from this lambda leaves no finished file behind.
    auto fillRecord = [&](size_t i, float * rec)
    {
        size_t p = 0;
        auto put = [&](float value) { rec[p++] = value; };

        // A non-finite position would be written and read back as it is; refuse
        // it as the other per-Gaussian values are refused.
        const glm::vec4 & position = positions[i];
        requireFinite(position.x, "x position", i, path);
        requireFinite(position.y, "y position", i, path);
        requireFinite(position.z, "z position", i, path);
        put(position.x);
        put(position.y);
        put(position.z);

        // nx ny nz are always written (zeros without normals), as INRIA's
        // writer does. PLYReader ignores them unless base_color is present.
        const glm::vec3 normal = normals.empty() ? glm::vec3(0.0f) : glm::vec3(normals[i]);
        put(normal.x);
        put(normal.y);
        put(normal.z);

        // Channel-major f_rest, like PLYReader expects: all R, then all G, then all B.
        const glm::vec4 * sh = &shCoefficients[i * kShCoefficientsPerGaussian];
        for (int c = 0; c < 3; ++c)
        {
            put(sh[0][c]);
        }
        for (int c = 0; c < 3; ++c)
        {
            for (size_t k = 0; k < restPerChannel; ++k)
            {
                put(sh[1 + k][c]);
            }
        }

        const double opacity = static_cast<double>(opacities[i]);
        requireFinite(opacity, "opacity", i, path);
        bool opacityClamped = false;
        put(static_cast<float>(logitClamped(opacity, opacityClamped)));

        for (int c = 0; c < 3; ++c)
        {
            const float scale = scales[i][c];
            if (!(scale > 0.0f) || !std::isfinite(scale))
            {
                throw std::runtime_error("PLYWriter: Gaussian " + std::to_string(i) + " scale_"
                                         + std::to_string(c) + " is " + std::to_string(scale)
                                         + ", must be positive and finite: " + path);
            }
            put(static_cast<float>(std::log(static_cast<double>(scale))));
        }

        // Canonical xyzw -> the file's (w, x, y, z).
        const glm::vec4 & q = rotations[i];
        put(q.w);
        put(q.x);
        put(q.y);
        put(q.z);

        if (relightable)
        {
            const glm::vec4 & br = baseColorRoughness[i];
            for (int c = 0; c < 4; ++c)
            {
                requireFinite(br[c], c < 3 ? "base color" : "roughness", i, path);
            }
            for (int c = 0; c < 3; ++c)
            {
                bool clamped = false;
                put(static_cast<float>(
                    invertAffineSigmoid(br[c], kBaseColorOffset, kBaseColorScale, clamped)));
                clampedRelight += clamped ? 1 : 0;
            }
            bool clamped = false;
            put(static_cast<float>(
                invertAffineSigmoid(br.w, kRoughnessOffset, kRoughnessScale, clamped)));
            clampedRelight += clamped ? 1 : 0;
        }
        if (writeVisibility)
        {
            for (size_t k = 0; k < kVisibilityCoefficients; ++k)
            {
                put(visibility[i * kVisibilityCoefficients + k]);
            }
        }
        if (writeMetallic)
        {
            put(metallic[i]);
        }

        // Every name was accounted for exactly once.
        if (p != stride)
        {
            throw std::logic_error("PLYWriter: record layout mismatch");
        }
    };

    // Via a temporary file renamed over the target, so a failed write never
    // leaves a truncated .ply in place of a good one.
    const std::string tmpPath = path + ".tmp";
    const std::filesystem::path target = std::filesystem::u8path(path);
    const std::filesystem::path tmpFile = std::filesystem::u8path(tmpPath);
    try
    {
        std::ofstream file(tmpFile, std::ios::binary | std::ios::trunc);
        if (!file)
        {
            throw std::runtime_error("PLYWriter: failed to open " + tmpPath + " for writing");
        }
        file.write(header.data(), static_cast<std::streamsize>(header.size()));

        std::vector<float> block(std::min(count, kRecordsPerBlock) * stride);
        for (size_t begin = 0; begin < count; begin += kRecordsPerBlock)
        {
            const size_t end = std::min(count, begin + kRecordsPerBlock);
            for (size_t i = begin; i < end; ++i)
            {
                fillRecord(i, block.data() + (i - begin) * stride);
            }
            file.write(reinterpret_cast<const char *>(block.data()),
                       static_cast<std::streamsize>((end - begin) * stride * sizeof(float)));
        }

        file.close();
        if (!file)
        {
            throw std::runtime_error("PLYWriter: failed to write " + tmpPath);
        }

        std::error_code ec;
        std::filesystem::rename(tmpFile, target, ec);
        if (ec)
        {
            throw std::runtime_error("PLYWriter: failed to rename " + tmpPath + " over " + path
                                     + ": " + ec.message());
        }
    }
    catch (...)
    {
        std::error_code ignored;
        std::filesystem::remove(tmpFile, ignored);
        throw;
    }

    if (clampedRelight > 0)
    {
        Logger::warning("PLYWriter: " + std::to_string(clampedRelight)
                        + " base color / roughness values outside the representable range were "
                          "clamped (sigmoid argument to ["
                        + std::to_string(PLYWriter::kLogitEpsilon) + ", 1 - eps]): " + path);
    }
}

} // namespace sabercat
