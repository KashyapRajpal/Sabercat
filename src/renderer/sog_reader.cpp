// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "sabercat/sog_reader.h"

#include <nlohmann/json.hpp>
#include <webp/decode.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "sabercat/gaussian_cloud.h"
#include "sabercat/logger.h"
#include "sabercat/parallel_for.h"
#include "sabercat/render_config.h"
#include "sabercat/vk_check.h"
#include "zip_archive.h"

namespace sabercat
{

namespace
{

using Json = nlohmann::json;
using Bytes = std::vector<uint8_t>;

// --- meta.json ---
// The only meta.json version this reader implements. v1 (no `version` key,
// per-channel mins/maxs instead of codebooks) predates splat-transform 1.0
// and is rejected by name rather than half-supported.
static constexpr int kSupportedMetaVersion = 2;
// Palette rows hold 64 centroids; each centroid is `coeffs` texels wide.
static constexpr uint32_t kCentroidsPerRow = 64;
// Coefficients per channel above DC for bands 1..3: (b+1)^2 - 1.
static constexpr uint32_t kShCoeffsForBands[4] = {0, 3, 8, 15};
// A codebook maps a byte, so it holds at most 256 entries.
static constexpr size_t kMaxCodebookSize = 256;

// --- Quantization constants (splat-transform writer / engine reader) ---
static constexpr double kMeansMaxCode = 65535.0;   // 16-bit position code
static constexpr int kQuatTagBase = 252;           // 252 + index of the dropped component
static constexpr float kByteToUnit = 1.0f / 255.0f;
static constexpr float kMinQuatLengthSquared = 1e-12f;

Bytes readWholeFile(const std::string & path, const char * what)
{
    std::ifstream file(std::filesystem::u8path(path), std::ios::binary | std::ios::ate);
    VK_VALIDATE(file.is_open(), std::string("SogReader: failed to open ") + what + " " + path);
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    VK_VALIDATE(size > 0, std::string("SogReader: ") + what + " is empty: " + path);
    Bytes bytes(static_cast<size_t>(size));
    file.read(reinterpret_cast<char *>(bytes.data()), size);
    VK_VALIDATE(file.gcount() == size, std::string("SogReader: truncated read of ") + path);
    return bytes;
}

// One decoded image, RGBA8 row-major.
struct Image
{
    Bytes rgba;
    uint32_t width = 0;
    uint32_t height = 0;
};

Image decodeWebp(const Bytes & encoded, const std::string & name, const std::string & path)
{
    const std::string where = "SogReader: image '" + name + "' of " + path;
    WebPBitstreamFeatures features;
    VK_VALIDATE(WebPGetFeatures(encoded.data(), encoded.size(), &features) == VP8_STATUS_OK,
                where + " is not a decodable WebP");
    VK_VALIDATE(!features.has_animation, where + " is an animated WebP; SOG images are stills");
    // SOG stores data, not pictures, so its images are LOSSLESS. A lossy one
    // still decodes (and the engine would render it), but every value in it
    // is perturbed -- worth saying, not worth refusing.
    if (features.format == 1)
    {
        Logger::warning(where + " is LOSSY WebP; SOG data images are lossless, so decoded values "
                                "will be approximate");
    }

    int width = 0;
    int height = 0;
    uint8_t * pixels = WebPDecodeRGBA(encoded.data(), encoded.size(), &width, &height);
    VK_VALIDATE(pixels != nullptr && width > 0 && height > 0, where + " failed to decode");
    Image image;
    image.width = static_cast<uint32_t>(width);
    image.height = static_cast<uint32_t>(height);
    image.rgba.assign(pixels, pixels + static_cast<size_t>(width) * static_cast<size_t>(height) * 4);
    WebPFree(pixels);
    return image;
}

// --- meta.json field access, every failure named ---

const Json & member(const Json & object, const char * key, const std::string & where)
{
    VK_VALIDATE(object.is_object() && object.contains(key),
                "SogReader: meta.json is missing '" + where + key + "'");
    return object.at(key);
}

std::vector<double> numberArray(const Json & object, const char * key, size_t expectedSize,
                                const std::string & where)
{
    const Json & array = member(object, key, where);
    VK_VALIDATE(array.is_array(), "SogReader: meta.json '" + where + key + "' is not an array");
    VK_VALIDATE(expectedSize == 0 || array.size() == expectedSize,
                "SogReader: meta.json '" + where + key + "' has " + std::to_string(array.size())
                    + " entries, expected " + std::to_string(expectedSize));
    std::vector<double> values;
    values.reserve(array.size());
    for (const Json & v : array)
    {
        VK_VALIDATE(v.is_number() && std::isfinite(v.get<double>()),
                    "SogReader: meta.json '" + where + key + "' holds a non-finite or non-number entry");
        values.push_back(v.get<double>());
    }
    return values;
}

// A 256-entry (at most) byte -> value table. Stored as float: the reference
// readers load it into a Float32Array, so this is the precision they decode
// at too.
std::vector<float> codebook(const Json & section, const std::string & where)
{
    const std::vector<double> values = numberArray(section, "codebook", 0, where);
    VK_VALIDATE(!values.empty() && values.size() <= kMaxCodebookSize,
                "SogReader: meta.json '" + where + "codebook' must hold 1..256 entries, has "
                    + std::to_string(values.size()));
    std::vector<float> table(values.size());
    std::transform(values.begin(), values.end(), table.begin(),
                   [](double v) { return static_cast<float>(v); });
    return table;
}

std::vector<std::string> fileList(const Json & section, size_t expectedCount, const std::string & where)
{
    const Json & files = member(section, "files", where);
    VK_VALIDATE(files.is_array() && files.size() == expectedCount,
                "SogReader: meta.json '" + where + "files' must list " + std::to_string(expectedCount)
                    + " image(s)");
    std::vector<std::string> names;
    for (const Json & f : files)
    {
        VK_VALIDATE(f.is_string(), "SogReader: meta.json '" + where + "files' holds a non-string entry");
        names.push_back(f.get<std::string>());
    }
    return names;
}

// Every per-Gaussian image must cover `count` pixels.
void requireCoverage(const Image & image, uint64_t count, const std::string & name,
                     const std::string & path)
{
    VK_VALIDATE(static_cast<uint64_t>(image.width) * image.height >= count,
                "SogReader: image '" + name + "' is " + std::to_string(image.width) + "x"
                    + std::to_string(image.height) + ", too small for " + std::to_string(count)
                    + " Gaussians: " + path);
}

// Rejects any byte in channels [0, channels) of the first `count` pixels that
// indexes past a codebook shorter than 256 (the writer always emits 256, so
// this only costs a pass on files that need it).
void requireCodebookIndices(const Image & image, uint64_t count, uint32_t channels, size_t codebookSize,
                            const std::string & name, const std::string & path)
{
    if (codebookSize >= kMaxCodebookSize)
    {
        return;
    }
    for (uint64_t i = 0; i < count; ++i)
    {
        for (uint32_t c = 0; c < channels; ++c)
        {
            VK_VALIDATE(image.rgba[i * 4 + c] < codebookSize,
                        "SogReader: image '" + name + "' pixel " + std::to_string(i)
                            + " indexes past its " + std::to_string(codebookSize)
                            + "-entry codebook: " + path);
        }
    }
}

// Inverse of the writer's logTransform(x) = sign(x) * ln(|x| + 1).
double invLogTransform(double v)
{
    const double magnitude = std::exp(std::abs(v)) - 1.0;
    return v < 0.0 ? -magnitude : magnitude;
}

bool endsWith(const std::string & text, const std::string & suffix)
{
    if (text.size() < suffix.size())
    {
        return false;
    }
    return std::equal(suffix.rbegin(), suffix.rend(), text.rbegin(),
                      [](char a, char b) { return std::tolower(static_cast<unsigned char>(a))
                                                  == std::tolower(static_cast<unsigned char>(b)); });
}

} // namespace

std::shared_ptr<GaussianCloud> SogReader::read(Device & device, const std::string & path)
{
    // --- Where the files come from: a ZIP bundle, or a directory ---
    Bytes archive;
    std::map<std::string, zip::ZipEntry> zipEntries;
    std::filesystem::path directory;
    const bool bundled = !endsWith(path, "meta.json");
    if (bundled)
    {
        archive = readWholeFile(path, "SOG bundle");
        zipEntries = zip::parseZip(archive, path, "SogReader");
    }
    else
    {
        directory = std::filesystem::u8path(path).parent_path();
    }
    const auto load = [&](const std::string & name) -> Bytes
    {
        if (bundled)
        {
            const auto it = zipEntries.find(name);
            VK_VALIDATE(it != zipEntries.end(),
                        "SogReader: " + path + " has no entry '" + name + "' (listed in meta.json)");
            return zip::extractZipEntry(archive, it->second, name, path, "SogReader");
        }
        VK_VALIDATE(!name.empty() && !std::filesystem::u8path(name).is_absolute(),
                    "SogReader: meta.json names an absolute or empty image path '" + name + "'");
        return readWholeFile((directory / name).string(), "SOG image");
    };
    const auto loadImage = [&](const std::string & name) { return decodeWebp(load(name), name, path); };

    // --- meta.json ---
    const Bytes metaBytes = bundled ? load("meta.json") : readWholeFile(path, "meta.json");
    const Json meta = Json::parse(metaBytes.begin(), metaBytes.end(), nullptr, /*allow_exceptions=*/false);
    VK_VALIDATE(!meta.is_discarded() && meta.is_object(), "SogReader: meta.json is not valid JSON: " + path);
    VK_VALIDATE(meta.contains("version"),
                "SogReader: meta.json has no 'version' -- a legacy SOG v1 file, which this reader does not "
                "implement; re-export it with a current splat-transform: "
                    + path);
    VK_VALIDATE(meta.at("version").is_number_integer() && meta.at("version").get<int>() == kSupportedMetaVersion,
                "SogReader: unsupported SOG meta.json version " + meta.at("version").dump()
                    + " (this reader implements version 2): " + path);

    const Json & countJson = member(meta, "count", "");
    VK_VALIDATE(countJson.is_number_unsigned() && countJson.get<uint64_t>() > 0
                    && countJson.get<uint64_t>() <= std::numeric_limits<uint32_t>::max(),
                "SogReader: meta.json 'count' must be a positive integer: " + path);
    const uint64_t count = countJson.get<uint64_t>();

    bool antialiased = false;
    if (meta.contains("model"))
    {
        const std::string model = meta.at("model").is_string() ? meta.at("model").get<std::string>() : "";
        VK_VALIDATE(model != "2dgs", "SogReader: meta.json declares a '2dgs' (surfel) model; this renderer "
                                     "rasterizes 3D Gaussians only: "
                                         + path);
        if (model == "antialiased")
        {
            antialiased = true;
        }
        else if (model != "default")
        {
            // splat-transform's own reader does the same: warn, read as default.
            Logger::warning("SogReader: unrecognized splat model '" + meta.at("model").dump()
                            + "' in meta.json, reading as 'default': " + path);
        }
    }

    const Json & meansJson = member(meta, "means", "");
    const std::vector<double> meanMins = numberArray(meansJson, "mins", 3, "means.");
    const std::vector<double> meanMaxs = numberArray(meansJson, "maxs", 3, "means.");
    const std::vector<std::string> meanFiles = fileList(meansJson, 2, "means.");
    const Json & scalesJson = member(meta, "scales", "");
    const std::vector<float> scaleCodebook = codebook(scalesJson, "scales.");
    const std::vector<std::string> scaleFiles = fileList(scalesJson, 1, "scales.");
    const std::vector<std::string> quatFiles = fileList(member(meta, "quats", ""), 1, "quats.");
    const Json & sh0Json = member(meta, "sh0", "");
    const std::vector<float> sh0Codebook = codebook(sh0Json, "sh0.");
    const std::vector<std::string> sh0Files = fileList(sh0Json, 1, "sh0.");

    // --- Per-Gaussian images ---
    const Image meansLo = loadImage(meanFiles[0]);
    const Image meansHi = loadImage(meanFiles[1]);
    const Image quats = loadImage(quatFiles[0]);
    const Image scaleImage = loadImage(scaleFiles[0]);
    const Image sh0Image = loadImage(sh0Files[0]);
    requireCoverage(meansLo, count, meanFiles[0], path);
    requireCoverage(meansHi, count, meanFiles[1], path);
    requireCoverage(quats, count, quatFiles[0], path);
    requireCoverage(scaleImage, count, scaleFiles[0], path);
    requireCoverage(sh0Image, count, sh0Files[0], path);
    requireCodebookIndices(scaleImage, count, 3, scaleCodebook.size(), scaleFiles[0], path);
    requireCodebookIndices(sh0Image, count, 3, sh0Codebook.size(), sh0Files[0], path);

    // A quaternion tag outside 252..255 means the image is not a SOG quats
    // image at all (both reference readers silently substitute identity; a
    // named error is more useful than a scene of unrotated Gaussians).
    for (uint64_t i = 0; i < count; ++i)
    {
        const int tag = quats.rgba[i * 4 + 3];
        VK_VALIDATE(tag >= kQuatTagBase && tag <= kQuatTagBase + 3,
                    "SogReader: quaternion " + std::to_string(i) + " has tag " + std::to_string(tag)
                        + " (expected 252..255 naming the dropped component): " + path);
    }

    // --- Optional higher SH: label -> centroid palette -> codebook ---
    uint32_t shBands = 0;
    uint32_t shCoeffs = 0;
    std::vector<float> shCodebook;
    Image centroids;
    Image labels;
    if (meta.contains("shN"))
    {
        const Json & shN = meta.at("shN");
        // Load the images first: older v2 files (e.g. the PlayCanvas engine's own
        // examples/assets/splats/skull.sog) carry neither 'shN.bands' nor
        // 'shN.count'. Both are derivable from the centroid image: width =
        // 64 * coeffs (3, 8 or 15), one row per 64 palette entries.
        const std::vector<std::string> shFiles = fileList(shN, 2, "shN.");
        centroids = loadImage(shFiles[0]);
        labels = loadImage(shFiles[1]);
        if (shN.contains("bands"))
        {
            const Json & bandsJson = shN.at("bands");
            VK_VALIDATE(bandsJson.is_number_integer() && bandsJson.get<int>() >= 1 && bandsJson.get<int>() <= 3,
                        "SogReader: meta.json 'shN.bands' must be 1, 2 or 3: " + path);
            shBands = bandsJson.get<uint32_t>();
        }
        else
        {
            for (uint32_t b = 1; b <= 3; ++b)
            {
                if (centroids.width == kCentroidsPerRow * kShCoeffsForBands[b])
                {
                    shBands = b;
                }
            }
            VK_VALIDATE(shBands != 0,
                        "SogReader: meta.json has no 'shN.bands' and the SH centroid image is "
                            + std::to_string(centroids.width) + " wide (expected 64 x 3, 8 or 15): " + path);
        }
        shCoeffs = kShCoeffsForBands[shBands];
        uint64_t paletteCount = static_cast<uint64_t>(centroids.height) * kCentroidsPerRow;
        if (shN.contains("count"))
        {
            const Json & paletteJson = shN.at("count");
            VK_VALIDATE(paletteJson.is_number_unsigned() && paletteJson.get<uint64_t>() > 0,
                        "SogReader: meta.json 'shN.count' must be a positive integer: " + path);
            paletteCount = paletteJson.get<uint64_t>();
        }
        shCodebook = codebook(shN, "shN.");
        requireCoverage(labels, count, shFiles[1], path);
        VK_VALIDATE(centroids.width == kCentroidsPerRow * shCoeffs,
                    "SogReader: SH centroid image '" + shFiles[0] + "' is " + std::to_string(centroids.width)
                        + " wide; " + std::to_string(shBands) + " band(s) need 64 x "
                        + std::to_string(shCoeffs) + " = " + std::to_string(kCentroidsPerRow * shCoeffs)
                        + ": " + path);
        // A label must name a palette entry AND a centroid that exists in
        // the image (the palette may be larger than what k-means filled).
        const uint64_t centroidsInImage = static_cast<uint64_t>(centroids.height) * kCentroidsPerRow;
        const uint64_t usableLabels = std::min(paletteCount, centroidsInImage);
        for (uint64_t i = 0; i < count; ++i)
        {
            const uint32_t label = labels.rgba[i * 4 + 0] | (labels.rgba[i * 4 + 1] << 8);
            VK_VALIDATE(label < usableLabels,
                        "SogReader: SH label " + std::to_string(label) + " of Gaussian " + std::to_string(i)
                            + " is outside the " + std::to_string(usableLabels) + "-entry palette: " + path);
        }
        requireCodebookIndices(centroids, static_cast<uint64_t>(centroids.width) * centroids.height, 3,
                               shCodebook.size(), shFiles[0], path);
    }

    auto cloud = std::make_shared<GaussianCloud>(device);
    cloud->initialize(static_cast<size_t>(count));
    auto & positions = cloud->positions();
    auto & scales = cloud->scales();
    auto & rotations = cloud->rotations();
    auto & sh = cloud->shCoefficients();
    auto & opacities = cloud->opacities();

    // In double, as the reference computes it. A zero-width axis decodes to
    // its min (the reference's `max - min || 1`).
    double meanRange[3];
    for (int axis = 0; axis < 3; ++axis)
    {
        meanRange[axis] = meanMaxs[axis] - meanMins[axis];
        if (meanRange[axis] == 0.0)
        {
            meanRange[axis] = 1.0;
        }
    }
    const float invSqrt2 = 1.0f / std::sqrt(2.0f);

    // Same disjoint-range contract as the other readers. Everything that
    // could fail was validated above, so the workers cannot throw.
    parallelFor(static_cast<size_t>(count),
                [&](size_t begin, size_t end)
                {
                    for (size_t i = begin; i < end; ++i)
                    {
                        const size_t px = i * 4;

                        // 16-bit code over [min, max] in LOG space, then the
                        // inverse of sign(x) * ln(|x| + 1).
                        float position[3];
                        for (int axis = 0; axis < 3; ++axis)
                        {
                            const uint32_t code = meansLo.rgba[px + axis] | (meansHi.rgba[px + axis] << 8);
                            position[axis] = static_cast<float>(
                                invLogTransform(meanMins[axis] + meanRange[axis] * (code / kMeansMaxCode)));
                        }
                        positions[i] = glm::vec4(position[0], position[1], position[2], 0.0f);

                        // Smallest-three in w,x,y,z order: the tag names the
                        // dropped (largest, stored positive) component, the
                        // three bytes fill the others in ascending order.
                        float q[4]; // w, x, y, z
                        const int dropped = quats.rgba[px + 3] - kQuatTagBase;
                        float sumSquares = 0.0f;
                        for (int lane = 0, stored = 0; lane < 4; ++lane)
                        {
                            if (lane == dropped)
                            {
                                continue;
                            }
                            q[lane] = (quats.rgba[px + stored] * kByteToUnit * 2.0f - 1.0f) * invSqrt2;
                            sumSquares += q[lane] * q[lane];
                            ++stored;
                        }
                        q[dropped] = std::sqrt(std::max(0.0f, 1.0f - sumSquares));
                        // Canonical xyzw. Renormalize: 8-bit lanes plus the
                        // clamp above do not land exactly on unit length.
                        const float lengthSquared = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
                        if (lengthSquared < kMinQuatLengthSquared)
                        {
                            rotations[i] = glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
                        }
                        else
                        {
                            const float invLength = 1.0f / std::sqrt(lengthSquared);
                            rotations[i] = glm::vec4(q[1] * invLength, q[2] * invLength, q[3] * invLength,
                                                     q[0] * invLength);
                        }

                        // The scale codebook holds LOG scales (k-means over
                        // the .ply's scale_* columns) -- exp() once.
                        scales[i] = glm::vec4(std::exp(scaleCodebook[scaleImage.rgba[px + 0]]),
                                              std::exp(scaleCodebook[scaleImage.rgba[px + 1]]),
                                              std::exp(scaleCodebook[scaleImage.rgba[px + 2]]), 0.0f);

                        // The sh0 codebook holds RAW f_dc -- unlike .splat and
                        // .ksplat there is no display transform to undo.
                        // Alpha is the activated opacity as a byte.
                        glm::vec4 * shOut = &sh[i * kShCoefficientsPerGaussian];
                        shOut[0] = glm::vec4(sh0Codebook[sh0Image.rgba[px + 0]], sh0Codebook[sh0Image.rgba[px + 1]],
                                             sh0Codebook[sh0Image.rgba[px + 2]], 0.0f);
                        opacities[i] = sh0Image.rgba[px + 3] * kByteToUnit;

                        if (shCoeffs > 0)
                        {
                            // Centroid texels are already coefficient-major,
                            // one texel per coefficient with RGB = channels:
                            // the canonical order, no transpose needed.
                            const uint32_t label = labels.rgba[px + 0] | (labels.rgba[px + 1] << 8);
                            const size_t row = label / kCentroidsPerRow;
                            const size_t column = (label % kCentroidsPerRow) * shCoeffs;
                            const uint8_t * texel = &centroids.rgba[(row * centroids.width + column) * 4];
                            for (uint32_t j = 0; j < shCoeffs; ++j)
                            {
                                shOut[1 + j] = glm::vec4(shCodebook[texel[j * 4 + 0]], shCodebook[texel[j * 4 + 1]],
                                                         shCodebook[texel[j * 4 + 2]], 0.0f);
                            }
                        }
                    }
                });

    cloud->setShDegree(shBands);
    cloud->setAntialiased(antialiased);
    cloud->computeBounds();
    return cloud;
}

} // namespace sabercat
