// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Sort study harness (manual; not a CTest test).
//
// For each scene and size, renders the GPU forward pass (default sort, Viewer
// mode) and the CPU reference models. The CPU models blend the same frozen
// candidates in different orders. Image quality is measured against cpu-exact
// (exact ray-depth order). The harness also measures temporal popping and
// near-tie swaps along an orbit, and GPU stage timings. Results are written to
// <out>/results.json and <out>/results.csv, and a table is printed per scene
// and size.
//
//   sabercat_sort_study --scene <file|synthetic> [--scene ...] --out <dir>
//       [--modes default,stable,tile-ray-depth,pixel-window16,pixel-window32,hybrid,
//                stochastic4,stochastic8,cpu-exact]
//       [--sizes 960x540,1920x1080] [--frames 120] [--stills 8] [--cost-frames 200]
//       [--no-stills] [--no-temporal] [--no-cost] [--no-flip] [--tile-stats] [--smoke]
//
// Each phase (stills, temporal, cost) can be switched off on its own. --tile-stats
// adds the default sort's tile list lengths and rasterizer batch counts at the
// first still's camera (off by default; it runs on its own). Cost-only
// runs use --no-stills --no-temporal; quality runs use --no-cost. With --no-stills,
// --stills is still parsed but unused.
//
// Progress goes to stderr: one line per still, and one per 10 temporal frames,
// each with the seconds since the program started. The synthetic scene's still 0
// also writes a 256x256 center crop of every mode's image as <mode>_<size>.png.
//
// --smoke: tiny settings for a quick run (synthetic scene if none is given,
// 256x192, 2 stills, 6 frames, 24 cost frames with 4 warm-up).
//
// Only a scene's file name is printed or written, never its directory.

#include <glm/glm.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "sabercat/device.h"
#include "sabercat/forward_pass.h"
#include "sabercat/gaussian_cloud.h"
#include "sabercat/sort_settings.h"

#include "cpu_reference.h"
#include "sort_metrics.h"
#include "study_capture.h"
#include "study_report.h"
#include "study_scenes.h"

#include "../common/shader_dir.h"

#ifdef SABERCAT_STUDY_HAVE_COMPARE
#include "image_compare.h"
#include "image_io.h"
#endif

using namespace sabercat;
using namespace sabercat::study;

namespace
{

constexpr char kSyntheticName[] = "synthetic";
constexpr float kFlipPpd = 67.0f;
constexpr uint32_t kCropSize = 256;
constexpr float kDiffGain = 4.0f;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kBytesPerMiB = 1024.0 * 1024.0;
// Ray-depth form: 12 floats per Gaussian. extraBufferBytes() counts it, and the
// stochastic per-pixel figure leaves it out.
constexpr double kRayFormBytesPerGaussian = 48.0;

// Stochastic stills accumulate this many static frames. The RMSE rows read the
// image after frames 1, 8 and 32 (names f1, f8, f32), and the usual metrics use
// the frame-32 image. 32 is also the mode's temporalFrames cap.
constexpr uint32_t kStochasticFrames = 32;

// Quality reference: exact ray-depth order over the default run's candidates.
constexpr study::ReferenceOptions kReference{study::ReferenceOrder::ExactRayDepth, 16};

// Start of the run, for the progress lines. Set at static initialization, before main().
const std::chrono::steady_clock::time_point kProgramStart = std::chrono::steady_clock::now();

double secondsSinceStart()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - kProgramStart).count();
}

// Mode registry. A GPU entry is rendered by the forward pass with its sort
// settings (configureGpuMode), and a CPU entry by renderReference() over the
// frozen candidates. The stochastic entries leave pixelWindow and the stable
// fields at their defaults; the stochastic path does not read them.
struct ModeInfo
{
    const char * name;
    bool gpu;
    study::ReferenceOptions cpu; // CPU entries only
    SortSettings sort;           // GPU entries only; every field is set before use
};

constexpr ModeInfo kModes[] = {
    {"default", true, {}, {SortMode::Default}},
    {"stable", true, {}, {SortMode::Stable}},
    {"tile-ray-depth", true, {}, {SortMode::TileRayDepth}},
    {"pixel-window16", true, {}, {SortMode::PixelWindow, 16}},
    {"pixel-window32", true, {}, {SortMode::PixelWindow, 32}},
    {"hybrid", true, {}, {SortMode::Hybrid}},
    // {mode, pixelWindow, stableBucketsPerOctave, stochasticSamples, temporalFrames, temporalMode}
    {"stochastic4", true, {}, {SortMode::Stochastic, 16, 1024, 4, 32, 0}},
    {"stochastic8", true, {}, {SortMode::Stochastic, 16, 1024, 8, 32, 0}},
    {"cpu-tile", false, {study::ReferenceOrder::TileOrder, 16}, {}},
    {"cpu-exact", false, {study::ReferenceOrder::ExactRayDepth, 16}, {}},
    {"cpu-window8", false, {study::ReferenceOrder::WindowRayDepth, 8}, {}},
    {"cpu-window16", false, {study::ReferenceOrder::WindowRayDepth, 16}, {}},
    {"cpu-window32", false, {study::ReferenceOrder::WindowRayDepth, 32}, {}},
};

// The --modes default. cpu-exact is the reference and is always computed
// internally; the other CPU modes are available by name only.
constexpr const char * kDefaultModes[] = {
    "default",  "stable",      "tile-ray-depth", "pixel-window16", "pixel-window32",
    "hybrid",   "stochastic4", "stochastic8",    "cpu-exact",
};

// Frame-time stage fields, in report order. "total" is the whole forward run.
struct StageField
{
    const char * name;
    float ForwardStageTimings::*member;
};

constexpr StageField kStageFields[] = {
    {"project", &ForwardStageTimings::project},
    {"scan", &ForwardStageTimings::scan},
    {"binning", &ForwardStageTimings::binning},
    {"sortKeys", &ForwardStageTimings::sortKeys},
    {"sort", &ForwardStageTimings::sort},
    {"ranges", &ForwardStageTimings::ranges},
    {"rasterize", &ForwardStageTimings::rasterize},
    {"resolve", &ForwardStageTimings::resolve},
    {"total", &ForwardStageTimings::total},
};
constexpr size_t kStageCount = sizeof(kStageFields) / sizeof(kStageFields[0]);

// Bad command line. main() prints the message and the usage text.
struct UsageError : std::runtime_error
{
    using std::runtime_error::runtime_error;
};

struct Size
{
    uint32_t width = 0;
    uint32_t height = 0;
};

std::string sizeLabel(const Size & size)
{
    return std::to_string(size.width) + "x" + std::to_string(size.height);
}

struct Settings
{
    std::vector<std::string> scenes; // paths as given, or "synthetic"
    std::vector<const ModeInfo *> modes;
    std::vector<Size> sizes;
    std::string outDir;
    uint32_t stills = 8;
    uint32_t frames = 120;
    uint32_t costFrames = 200;
    uint32_t costWarmup = 20;
    bool stillsEnabled = true;
    bool tileStats = false;
    bool temporal = true;
    bool cost = true;
    bool flip = true;
};

// RMSE against the reference after 1, 8 and 32 static frames. NaN for every
// mode that is not stochastic.
struct StochasticRmse
{
    double f1 = kNaN;
    double f8 = kNaN;
    double f32 = kNaN;
};

struct StillMetrics
{
    double mae = 0.0;
    double psnrDb = 0.0;
    double flipMean = 0.0;
    StochasticRmse rmse;
};

// One stochastic still: the frame-32 image and the RMSE checkpoints.
struct StochasticStill
{
    std::vector<glm::vec4> image;
    StochasticRmse rmse;
};

// Appends rows for one scene and size.
class RowSink
{
  public:
    RowSink(std::vector<ResultRow> & rows, std::string scene, std::string size)
        : m_rows(rows), m_scene(std::move(scene)), m_size(std::move(size))
    {
    }

    void add(const std::string & mode, const std::string & metric, double value, const char * unit)
    {
        m_rows.push_back(ResultRow{m_scene, m_size, mode, metric, value, unit});
    }

  private:
    std::vector<ResultRow> & m_rows;
    std::string m_scene;
    std::string m_size;
};

// ---------------------------------------------------------------------------
// Command line

void printUsage(FILE * stream, const char * program)
{
    std::fprintf(stream,
                 "usage: %s --scene <file|synthetic> [--scene ...] --out <dir>\n"
                 "       [--modes <list>] [--sizes <WxH,...>] [--frames N] [--stills N]\n"
                 "       [--cost-frames N] [--no-stills] [--no-temporal] [--no-cost] [--no-flip]\n"
                 "       [--tile-stats] [--smoke]\n"
                 "modes (default list): default,stable,tile-ray-depth,pixel-window16,pixel-window32,hybrid,\n"
                 "                      stochastic4,stochastic8,cpu-exact\n"
                 "other modes: cpu-tile,cpu-window8,cpu-window16,cpu-window32\n",
                 program);
}

std::vector<std::string> splitList(const std::string & text)
{
    std::vector<std::string> items;
    size_t start = 0;
    while (true)
    {
        const size_t comma = text.find(',', start);
        std::string item = text.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
        if (item.empty())
        {
            throw UsageError("empty item in list '" + text + "'");
        }
        items.push_back(std::move(item));
        if (comma == std::string::npos)
        {
            return items;
        }
        start = comma + 1;
    }
}

uint32_t parsePositive(const std::string & text, const char * flag)
{
    if (text.empty() || !std::isdigit(static_cast<unsigned char>(text[0])))
    {
        throw UsageError(std::string(flag) + " expects a positive integer, got '" + text + "'");
    }
    char * end = nullptr;
    const unsigned long value = std::strtoul(text.c_str(), &end, 10);
    if (*end != '\0' || value == 0 || value > std::numeric_limits<uint32_t>::max())
    {
        throw UsageError(std::string(flag) + " expects a positive integer, got '" + text + "'");
    }
    return static_cast<uint32_t>(value);
}

Size parseSize(const std::string & text)
{
    unsigned int width = 0;
    unsigned int height = 0;
    int consumed = 0;
    const bool parsed = std::sscanf(text.c_str(), "%ux%u%n", &width, &height, &consumed) == 2;
    if (!parsed || static_cast<size_t>(consumed) != text.size() || width == 0 || height == 0)
    {
        throw UsageError("--sizes expects WxH entries, got '" + text + "'");
    }
    Size size;
    size.width = width;
    size.height = height;
    return size;
}

const ModeInfo * findMode(const std::string & name)
{
    for (const ModeInfo & mode : kModes)
    {
        if (name == mode.name)
        {
            return &mode;
        }
    }
    throw UsageError("unknown mode '" + name + "'");
}

Settings parseArguments(int argc, char ** argv)
{
    std::vector<std::string> sceneArgs;
    std::string outDir;
    std::vector<std::string> modeNames;
    std::vector<std::string> sizeNames;
    std::optional<uint32_t> stills;
    std::optional<uint32_t> frames;
    std::optional<uint32_t> costFrames;
    bool smoke = false;
    bool noTemporal = false;
    bool noCost = false;
    bool noFlip = false;
    bool noStills = false;
    bool tileStats = false;

    for (int i = 1; i < argc; ++i)
    {
        const std::string arg = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc)
            {
                throw UsageError(arg + " needs a value");
            }
            return argv[++i];
        };

        if (arg == "--scene")
        {
            sceneArgs.push_back(value());
        }
        else if (arg == "--out")
        {
            outDir = value();
        }
        else if (arg == "--modes")
        {
            modeNames = splitList(value());
        }
        else if (arg == "--sizes")
        {
            sizeNames = splitList(value());
        }
        else if (arg == "--frames")
        {
            frames = parsePositive(value(), "--frames");
        }
        else if (arg == "--stills")
        {
            stills = parsePositive(value(), "--stills");
        }
        else if (arg == "--cost-frames")
        {
            costFrames = parsePositive(value(), "--cost-frames");
        }
        else if (arg == "--no-temporal")
        {
            noTemporal = true;
        }
        else if (arg == "--no-cost")
        {
            noCost = true;
        }
        else if (arg == "--no-flip")
        {
            noFlip = true;
        }
        else if (arg == "--no-stills")
        {
            noStills = true;
        }
        else if (arg == "--tile-stats")
        {
            tileStats = true;
        }
        else if (arg == "--smoke")
        {
            smoke = true;
        }
        else
        {
            throw UsageError("unknown argument '" + arg + "'");
        }
    }

    if (outDir.empty())
    {
        throw UsageError("--out <dir> is required");
    }
    if (sceneArgs.empty())
    {
        if (!smoke)
        {
            throw UsageError("at least one --scene is required (or --smoke, which uses the synthetic scene)");
        }
        sceneArgs.push_back(kSyntheticName);
    }

    Settings settings;
    settings.scenes = std::move(sceneArgs);
    settings.outDir = std::move(outDir);

    if (modeNames.empty())
    {
        for (const char * name : kDefaultModes)
        {
            modeNames.push_back(name);
        }
    }
    for (const std::string & name : modeNames)
    {
        const ModeInfo * mode = findMode(name);
        if (std::find(settings.modes.begin(), settings.modes.end(), mode) == settings.modes.end())
        {
            settings.modes.push_back(mode);
        }
    }

    if (sizeNames.empty())
    {
        if (smoke)
        {
            sizeNames = {"256x192"};
        }
        else
        {
            sizeNames = {"960x540", "1920x1080"};
        }
    }
    for (const std::string & name : sizeNames)
    {
        settings.sizes.push_back(parseSize(name));
    }

    // --smoke only supplies values the command line did not set.
    settings.stills = stills.value_or(smoke ? 2u : 8u);
    settings.frames = frames.value_or(smoke ? 6u : 120u);
    settings.costFrames = costFrames.value_or(smoke ? 24u : 200u);
    settings.costWarmup = smoke ? 4u : 20u;
    settings.stillsEnabled = !noStills;
    settings.tileStats = tileStats;
    settings.temporal = !noTemporal;
    settings.cost = !noCost;
    settings.flip = !noFlip;

    if (settings.cost && settings.costFrames <= settings.costWarmup)
    {
        throw UsageError("--cost-frames must exceed the warm-up count (" + std::to_string(settings.costWarmup)
                         + ")");
    }
    return settings;
}

bool hasMode(const Settings & settings, const char * name)
{
    return std::any_of(settings.modes.begin(), settings.modes.end(),
                       [name](const ModeInfo * mode) { return std::strcmp(mode->name, name) == 0; });
}

// ---------------------------------------------------------------------------
// Numbers and images

double meanOf(const std::vector<double> & values)
{
    if (values.empty())
    {
        return kNaN;
    }
    double sum = 0.0;
    for (const double v : values)
    {
        sum += v;
    }
    return sum / static_cast<double>(values.size());
}

// Nearest-rank percentile, or NaN when there are no samples. study::percentile
// returns 0 for an empty list, which would read as a real timing.
double percentileOrNan(const std::vector<double> & values, double p)
{
    return values.empty() ? kNaN : study::percentile(values, p);
}

std::string formatValue(double value)
{
    if (std::isnan(value))
    {
        return "nan";
    }
    if (std::isinf(value))
    {
        return value > 0.0 ? "inf" : "-inf";
    }
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.6g", value);
    return buffer;
}

#ifdef SABERCAT_STUDY_HAVE_COMPARE
compare::Image toCompareImage(const std::vector<glm::vec4> & image, const Size & size)
{
    compare::Image out;
    out.width = size.width;
    out.height = size.height;
    out.rgb.resize(static_cast<size_t>(size.width) * size.height * 3);
    for (size_t i = 0; i < static_cast<size_t>(size.width) * size.height; ++i)
    {
        out.rgb[3 * i + 0] = image[i].r;
        out.rgb[3 * i + 1] = image[i].g;
        out.rgb[3 * i + 2] = image[i].b;
    }
    return out;
}

// Center crop of at most kCropSize x kCropSize, as 8-bit sRGB display values.
// `gain` scales before clamping to [0, 1]. NaN maps to 0.
compare::Rgb8Image cropToRgb8(const std::vector<glm::vec4> & image, const Size & size, float gain)
{
    const uint32_t cropWidth = std::min(kCropSize, size.width);
    const uint32_t cropHeight = std::min(kCropSize, size.height);
    const uint32_t x0 = (size.width - cropWidth) / 2;
    const uint32_t y0 = (size.height - cropHeight) / 2;

    compare::Rgb8Image out;
    out.width = cropWidth;
    out.height = cropHeight;
    out.rgb.resize(static_cast<size_t>(cropWidth) * cropHeight * 3);
    for (uint32_t y = 0; y < cropHeight; ++y)
    {
        for (uint32_t x = 0; x < cropWidth; ++x)
        {
            const glm::vec4 & pixel = image[static_cast<size_t>(y0 + y) * size.width + (x0 + x)];
            for (int c = 0; c < 3; ++c)
            {
                const float scaled = pixel[c] * gain;
                const float clamped = scaled > 0.0f ? (scaled < 1.0f ? scaled : 1.0f) : 0.0f;
                out.rgb[(static_cast<size_t>(y) * cropWidth + x) * 3 + c] =
                    static_cast<uint8_t>(clamped * 255.0f + 0.5f);
            }
        }
    }
    return out;
}

// Writes the synthetic scene's still-0 crops: the reference, the default GPU
// image and |default - reference| * kDiffGain.
void saveCrops(const Settings & settings, const Size & size, const std::vector<glm::vec4> & reference,
               const std::vector<glm::vec4> & gpuDefault)
{
    std::vector<glm::vec4> diff(gpuDefault.size());
    for (size_t i = 0; i < diff.size(); ++i)
    {
        diff[i] = glm::abs(gpuDefault[i] - reference[i]);
    }

    const std::filesystem::path dir(settings.outDir);
    const std::string suffix = "_" + sizeLabel(size) + ".png";
    compare::savePngRgb8((dir / ("ref_exact" + suffix)).string(), cropToRgb8(reference, size, 1.0f));
    compare::savePngRgb8((dir / ("default" + suffix)).string(), cropToRgb8(gpuDefault, size, 1.0f));
    compare::savePngRgb8((dir / ("abs_diff_default" + suffix)).string(), cropToRgb8(diff, size, kDiffGain));
}

// Center crop of one mode's still-0 image, written as <mode>_<size>.png.
void saveModeCrop(const Settings & settings, const Size & size, const char * mode,
                  const std::vector<glm::vec4> & image)
{
    const std::filesystem::path dir(settings.outDir);
    const std::string file = std::string(mode) + "_" + sizeLabel(size) + ".png";
    compare::savePngRgb8((dir / file).string(), cropToRgb8(image, size, 1.0f));
}
#endif

// Root-mean-square error over pixels and RGB (alpha is not compared), with the
// sum in double. NaN for empty images. Throws std::invalid_argument when the
// sizes differ.
double rootMeanSquareError(const std::vector<glm::vec4> & test,
                           const std::vector<glm::vec4> & reference)
{
    if (test.size() != reference.size())
    {
        throw std::invalid_argument("rootMeanSquareError: image sizes differ");
    }
    if (test.empty())
    {
        return kNaN;
    }
    double sum = 0.0;
    for (size_t i = 0; i < test.size(); ++i)
    {
        for (int c = 0; c < 3; ++c)
        {
            const double d = static_cast<double>(test[i][c]) - static_cast<double>(reference[i][c]);
            sum += d * d;
        }
    }
    return std::sqrt(sum / (3.0 * static_cast<double>(test.size())));
}

// FLIP mean error, or NaN when comparison support is not built in.
double flipMeanError(const std::vector<glm::vec4> & reference, const std::vector<glm::vec4> & test,
                     const Size & size)
{
#ifdef SABERCAT_STUDY_HAVE_COMPARE
    return compare::flip(toCompareImage(reference, size), toCompareImage(test, size), kFlipPpd).meanError;
#else
    (void)reference;
    (void)test;
    (void)size;
    return kNaN;
#endif
}

StillMetrics stillMetrics(const std::vector<glm::vec4> & test, const std::vector<glm::vec4> & reference,
                          const Size & size, bool withFlip)
{
    StillMetrics metrics;
    metrics.mae = study::meanAbsoluteError(reference, test);
    metrics.psnrDb = study::psnrDb(reference, test);
    metrics.flipMean = withFlip ? flipMeanError(reference, test, size) : kNaN;
    return metrics;
}

// ---------------------------------------------------------------------------
// Rendering helpers

// Default-sort Viewer run. Re-applies the default settings first, so a GPU mode
// that leaves the pass in another sort order cannot leak into the candidates.
void runDefaultSort(ForwardPass & pass, const CanonicalGaussianBufferViews & views, const CameraUniforms & camera)
{
    pass.setSortSettings(SortSettings{});
    pass.run(views, camera);
}

// True for the GPU entry that is the default sort. Its image is the default run
// that the reference is built from, so it is never rendered a second time.
bool isDefaultSort(const ModeInfo & mode)
{
    return mode.gpu && mode.sort.mode == SortMode::Default;
}

// EXTENSION POINT: sets the pass to the sort of a GPU mode before its run.
// Every field is set explicitly, so no earlier mode's settings carry over.
void configureGpuMode(ForwardPass & pass, const ModeInfo & mode)
{
    pass.setSortSettings(mode.sort);
}

// Renders one non-default GPU mode for a still. The Stable keys are reset first
// so each still starts from the same state, and the default settings are
// restored afterwards so the next reference run is clean.
std::vector<glm::vec4> renderGpuStill(ForwardPass & pass, const CanonicalGaussianBufferViews & views,
                                      const CameraUniforms & camera, const ModeInfo & mode)
{
    pass.resetStableSort();
    configureGpuMode(pass, mode);
    pass.run(views, camera);
    std::vector<glm::vec4> image = pass.downloadOutputImage();
    pass.setSortSettings(SortSettings{});
    return image;
}

// Renders one stochastic mode for a still: kStochasticFrames static frames at the
// still's camera, so the accumulation runs from a fresh history. The RMSE is read
// after frames 1, 8 and 32. The default settings are restored afterwards.
StochasticStill renderStochasticStill(ForwardPass & pass, const CanonicalGaussianBufferViews & views,
                                      const CameraUniforms & camera, const ModeInfo & mode,
                                      const std::vector<glm::vec4> & reference)
{
    pass.resetStochastic();
    configureGpuMode(pass, mode);

    StochasticStill still;
    for (uint32_t frame = 1; frame <= kStochasticFrames; ++frame)
    {
        pass.run(views, camera);

        const bool checkpoint = frame == 1 || frame == 8 || frame == kStochasticFrames;
        if (!checkpoint)
        {
            continue;
        }
        std::vector<glm::vec4> image = pass.downloadOutputImage();
        const double rmse = rootMeanSquareError(image, reference);
        if (frame == 1)
        {
            still.rmse.f1 = rmse;
        }
        else if (frame == 8)
        {
            still.rmse.f8 = rmse;
        }
        else
        {
            still.rmse.f32 = rmse;
            still.image = std::move(image);
        }
    }
    pass.setSortSettings(SortSettings{});
    return still;
}

// True for the stochastic GPU modes. They skip the depth passes, so their lists
// have no sort order: there is no order capture and no near-tie count for them.
bool isStochastic(const ModeInfo & mode)
{
    return mode.gpu && mode.sort.mode == SortMode::Stochastic;
}

// True for the GPU modes whose state carries across the frames of a sequence:
// Stable keeps sticky keys, the stochastic modes keep accumulation history.
bool carriesSequenceState(const ModeInfo & mode)
{
    return mode.gpu && (mode.sort.mode == SortMode::Stable || isStochastic(mode));
}

// Image of one requested mode for a view. `gpuImage` is the mode's own GPU image
// and is only read for GPU modes; `reference` is the cpu-exact image.
std::vector<glm::vec4> modeImage(const ModeInfo & mode, const study::StudyCandidates & candidates,
                                 const std::vector<glm::vec4> & gpuImage,
                                 const std::vector<glm::vec4> & reference)
{
    if (mode.gpu)
    {
        return gpuImage;
    }
    if (mode.cpu.order == study::ReferenceOrder::ExactRayDepth)
    {
        return reference;
    }
    return study::renderReference(candidates, mode.cpu);
}

// ---------------------------------------------------------------------------
// Phases
//
// Stills use `pass` and temporal runs use `pass` for the stateless modes. Stable
// and the stochastic modes carry state across frames, so each has a persistent
// pass of its own (see runSceneSize).

// (1) Stills: quality against cpu-exact at evenly spaced azimuths.
void runStills(ForwardPass & pass, const CanonicalGaussianBufferViews & views, const StudyScene & scene,
               const Settings & settings, const Size & size, RowSink & sink)
{
    std::vector<std::vector<StillMetrics>> perMode(settings.modes.size());
    const bool synthetic = scene.name == kSyntheticName;

    for (uint32_t k = 0; k < settings.stills; ++k)
    {
        const float azimuth = 360.0f * static_cast<float>(k) / static_cast<float>(settings.stills);
        const CameraUniforms camera = makeStudyCamera(scene, size.width, size.height, azimuth);
        runDefaultSort(pass, views, camera);

        const study::StudyCandidates candidates = study::captureCandidates(pass, views, camera, true);
        const std::vector<glm::vec4> gpuDefault = pass.downloadOutputImage();
        // Always computed, whether or not cpu-exact is in the mode list: every
        // quality, popping and RMSE row compares against it.
        const std::vector<glm::vec4> reference = study::renderReference(candidates, kReference);

        if (synthetic && k == 0)
        {
#ifdef SABERCAT_STUDY_HAVE_COMPARE
            saveCrops(settings, size, reference, gpuDefault);
#else
            std::fprintf(stderr, "sort_study: PNG crops need SABERCAT_STUDY_HAVE_COMPARE; skipped\n");
#endif
        }

        const std::string prefix = "still" + std::to_string(k) + ".";
        for (size_t m = 0; m < settings.modes.size(); ++m)
        {
            const ModeInfo & mode = *settings.modes[m];
            std::vector<glm::vec4> gpuImage;
            StochasticRmse rmse;
            if (isDefaultSort(mode))
            {
                gpuImage = gpuDefault;
            }
            else if (isStochastic(mode))
            {
                StochasticStill still = renderStochasticStill(pass, views, camera, mode, reference);
                gpuImage = std::move(still.image);
                rmse = still.rmse;
            }
            else if (mode.gpu)
            {
                gpuImage = renderGpuStill(pass, views, camera, mode);
            }
            const std::vector<glm::vec4> image = modeImage(mode, candidates, gpuImage, reference);
#ifdef SABERCAT_STUDY_HAVE_COMPARE
            // The stochastic image here is the frame-32 accumulation, as the metrics use.
            if (synthetic && k == 0)
            {
                saveModeCrop(settings, size, mode.name, image);
            }
#endif
            StillMetrics metrics = stillMetrics(image, reference, size, settings.flip);
            metrics.rmse = rmse;
            perMode[m].push_back(metrics);

            sink.add(mode.name, prefix + "mae", metrics.mae, "display");
            sink.add(mode.name, prefix + "psnr_db", metrics.psnrDb, "dB");
            sink.add(mode.name, prefix + "flip_mean", metrics.flipMean, "flip");
            if (isStochastic(mode))
            {
                sink.add(mode.name, prefix + "rmse_f1", metrics.rmse.f1, "display");
                sink.add(mode.name, prefix + "rmse_f8", metrics.rmse.f8, "display");
                sink.add(mode.name, prefix + "rmse_f32", metrics.rmse.f32, "display");
            }
        }
        std::fprintf(stderr, "sort_study: %s %s still %u of %u done, %.1f s\n", scene.name.c_str(),
                     sizeLabel(size).c_str(), k + 1, settings.stills, secondsSinceStart());
    }

    for (size_t m = 0; m < settings.modes.size(); ++m)
    {
        std::vector<double> mae;
        std::vector<double> psnr;
        std::vector<double> flipValues;
        std::vector<double> rmseF1;
        std::vector<double> rmseF8;
        std::vector<double> rmseF32;
        for (const StillMetrics & metrics : perMode[m])
        {
            mae.push_back(metrics.mae);
            psnr.push_back(metrics.psnrDb);
            flipValues.push_back(metrics.flipMean);
            rmseF1.push_back(metrics.rmse.f1);
            rmseF8.push_back(metrics.rmse.f8);
            rmseF32.push_back(metrics.rmse.f32);
        }
        const ModeInfo & mode = *settings.modes[m];
        sink.add(mode.name, "mean.mae", meanOf(mae), "display");
        sink.add(mode.name, "mean.psnr_db", meanOf(psnr), "dB");
        sink.add(mode.name, "mean.flip_mean", meanOf(flipValues), "flip");
        if (isStochastic(mode))
        {
            sink.add(mode.name, "mean.rmse_f1", meanOf(rmseF1), "display");
            sink.add(mode.name, "mean.rmse_f8", meanOf(rmseF8), "display");
            sink.add(mode.name, "mean.rmse_f32", meanOf(rmseF32), "display");
        }
    }
}

// (2) Temporal: popping and near-tie swaps over a 0.5 degree per frame orbit.
// Every frame starts with the default run that the reference is built from, and
// each GPU mode then renders the same frame with its own settings. The Stable
// sticky keys must persist across frames, but leaving Stable (which the default
// run does) resets them, so the Stable sequence runs on its own persistent pass.
// The stochastic history needs the same treatment: a settings change resets it,
// so each stochastic mode has a pass of its own too. `statePasses[m]` is that
// pass for every mode that carries state, and null for the others, which run on
// `pass`.
void runTemporal(ForwardPass & pass, const std::vector<std::unique_ptr<ForwardPass>> & statePasses,
                 const CanonicalGaussianBufferViews & views, const StudyScene & scene, const Settings & settings,
                 const Size & size, RowSink & sink)
{
    const size_t modeCount = settings.modes.size();
    const uint32_t gaussianCount = static_cast<uint32_t>(scene.cloud->count());
    const bool defaultListed = hasMode(settings, "default");

    for (size_t m = 0; m < modeCount; ++m)
    {
        if (statePasses[m] == nullptr)
        {
            continue;
        }
        if (isStochastic(*settings.modes[m]))
        {
            statePasses[m]->resetStochastic();
        }
        else
        {
            statePasses[m]->resetStableSort();
        }
    }

    std::vector<study::PoppingAccumulator> popping(modeCount);
    std::vector<std::vector<glm::vec4>> prevImage(modeCount);
    std::vector<study::SortOrderFrame> prevOrder(modeCount);
    std::vector<uint64_t> nearTieSwaps(modeCount, 0);
    std::vector<std::vector<double>> motionRmse(modeCount);
    std::vector<glm::vec4> prevReference;

    for (uint32_t frame = 0; frame < settings.frames; ++frame)
    {
        if (frame % 10 == 0)
        {
            std::fprintf(stderr, "sort_study: %s %s temporal frame %u of %u, %.1f s\n", scene.name.c_str(),
                         sizeLabel(size).c_str(), frame, settings.frames, secondsSinceStart());
        }

        const float azimuth = 0.5f * static_cast<float>(frame);
        const CameraUniforms camera = makeStudyCamera(scene, size.width, size.height, azimuth);
        runDefaultSort(pass, views, camera);

        const study::StudyCandidates candidates = study::captureCandidates(pass, views, camera, true);
        // Read before any other run on this pass changes its lists.
        study::SortOrderFrame defaultOrder;
        if (defaultListed)
        {
            defaultOrder = study::captureOrder(pass, gaussianCount);
        }
        const std::vector<glm::vec4> gpuDefault = pass.downloadOutputImage();
        // Built here whether or not cpu-exact is listed, so popping and RMSE never
        // depend on the mode list.
        std::vector<glm::vec4> reference = study::renderReference(candidates, kReference);

        for (size_t m = 0; m < modeCount; ++m)
        {
            const ModeInfo & mode = *settings.modes[m];
            std::vector<glm::vec4> gpuImage;
            study::SortOrderFrame order;
            if (isDefaultSort(mode))
            {
                gpuImage = gpuDefault;
                order = defaultOrder;
            }
            else if (mode.gpu)
            {
                ForwardPass & modePass = statePasses[m] != nullptr ? *statePasses[m] : pass;
                configureGpuMode(modePass, mode);
                modePass.run(views, camera);
                gpuImage = modePass.downloadOutputImage();
                // A stochastic frame has no sort order to capture.
                if (!isStochastic(mode))
                {
                    order = study::captureOrder(modePass, gaussianCount);
                }
            }

            std::vector<glm::vec4> image = modeImage(mode, candidates, gpuImage, reference);
            if (isStochastic(mode))
            {
                // Every frame of the path counts, the first included. This is how far
                // one raw stochastic frame sits from the exact reference while moving.
                motionRmse[m].push_back(rootMeanSquareError(image, reference));
            }
            if (frame > 0)
            {
                popping[m].addFrame(prevImage[m], image, prevReference, reference);
                // Near-tie swaps compare two sort orders, which a stochastic mode does not have.
                if (mode.gpu && !isStochastic(mode))
                {
                    nearTieSwaps[m] += study::countNearTieSwaps(prevOrder[m], order);
                }
            }
            prevImage[m] = std::move(image);
            prevOrder[m] = std::move(order);
        }
        prevReference = std::move(reference);
    }

    const double transitions = static_cast<double>(settings.frames) - 1.0;
    for (size_t m = 0; m < modeCount; ++m)
    {
        const ModeInfo & mode = *settings.modes[m];
        sink.add(mode.name, "popping_score", popping[m].meanScore(), "display");
        sink.add(mode.name, "pop_pixels_per_frame", popping[m].meanPopPixels(), "pixels");
        if (mode.gpu && !isStochastic(mode))
        {
            // Pairs swapped between consecutive frames of this mode, per transition.
            const double perFrame =
                transitions > 0.0 ? static_cast<double>(nearTieSwaps[m]) / transitions : kNaN;
            sink.add(mode.name, "near_tie_swaps_per_frame", perFrame, "pairs");
        }
        if (isStochastic(mode))
        {
            sink.add(mode.name, "motion_rmse_mean", meanOf(motionRmse[m]), "display");
            sink.add(mode.name, "motion_rmse_p95", percentileOrNan(motionRmse[m], 0.95), "display");
        }
    }
}

// (5) Tile statistics of the default sort, at the first still's camera (azimuth
// 0). Reads each tile's list length and the batches tile_rasterize fetched for it.
// A batch is one staged group of kBatchInstances Gaussians (one per thread of a
// 16x16 tile). batches_total * kBatchInstances is what the rasterizer walked, so
// the rest of the list is what the group-wide early-out skipped.
constexpr uint64_t kBatchInstances = 256;

void runTileStats(ForwardPass & pass, const CanonicalGaussianBufferViews & views, const StudyScene & scene,
                  const Size & size, RowSink & sink)
{
    const CameraUniforms camera = makeStudyCamera(scene, size.width, size.height, 0.0f);
    runDefaultSort(pass, views, camera);

    const size_t tileCount = static_cast<size_t>(camera.tilesX) * camera.tilesY;
    std::vector<TileRange> ranges(tileCount);
    pass.tileRanges().download(ranges.data(), tileCount * sizeof(TileRange));
    std::vector<uint32_t> batches(tileCount);
    pass.batchesFetched().download(batches.data(), tileCount * sizeof(uint32_t));

    // List lengths and batch counts over all tiles; the batch statistics below
    // use only the tiles whose list is nonempty.
    std::vector<double> lengths;
    std::vector<double> nonEmptyBatches;
    lengths.reserve(tileCount);
    uint64_t instances = 0;
    uint64_t batchesTotal = 0;
    for (size_t t = 0; t < tileCount; ++t)
    {
        const TileRange & range = ranges[t];
        if (range.last < range.first)
        {
            throw std::runtime_error("tile_stats: tile " + std::to_string(t) + " has last < first");
        }
        const uint32_t length = range.last - range.first;
        lengths.push_back(static_cast<double>(length));
        instances += length;
        batchesTotal += batches[t];
        if (length > 0)
        {
            nonEmptyBatches.push_back(static_cast<double>(batches[t]));
        }
    }

    const double listMax = lengths.empty() ? kNaN : *std::max_element(lengths.begin(), lengths.end());
    const double batchesMax =
        nonEmptyBatches.empty() ? kNaN : *std::max_element(nonEmptyBatches.begin(), nonEmptyBatches.end());
    // Share of the instances the rasterizer did not walk, clamped because the
    // batch count can cover slightly more than the list when a tile is short.
    const double covered = static_cast<double>(batchesTotal * kBatchInstances);
    const double wasted = 1.0 - covered / static_cast<double>(std::max<uint64_t>(instances, 1));

    sink.add("default", "tile_stats.tiles", static_cast<double>(tileCount), "tiles");
    sink.add("default", "tile_stats.instances", static_cast<double>(instances), "instances");
    sink.add("default", "tile_stats.list_mean", meanOf(lengths), "instances");
    sink.add("default", "tile_stats.list_p50", percentileOrNan(lengths, 0.5), "instances");
    sink.add("default", "tile_stats.list_p99", percentileOrNan(lengths, 0.99), "instances");
    sink.add("default", "tile_stats.list_max", listMax, "instances");
    sink.add("default", "tile_stats.batches_total", static_cast<double>(batchesTotal), "batches");
    sink.add("default", "tile_stats.batches_mean", meanOf(nonEmptyBatches), "batches");
    sink.add("default", "tile_stats.batches_max", batchesMax, "batches");
    sink.add("default", "tile_stats.batches_p99", percentileOrNan(nonEmptyBatches, 0.99), "batches");
    sink.add("default", "tile_stats.wasted_fraction", std::clamp(wasted, 0.0, 1.0), "fraction");
}

// (3) Cost: GPU modes only. Wall time around run() and the GPU stage timings.
// Each mode runs on a pass of its own: buffers that an earlier mode allocated
// (the ray form, the sticky keys) would otherwise count against this mode's memory.
void runCostForMode(Device & device, const CanonicalGaussianBufferViews & views, const StudyScene & scene,
                    const Settings & settings, const Size & size, const ModeInfo & mode, RowSink & sink)
{
    ForwardPass pass(device, size.width, size.height, testShaderDir(), DepthLayout::None);
    configureGpuMode(pass, mode);
    pass.setStageTimingEnabled(true);

    std::vector<double> wallMs;
    std::vector<std::vector<double>> stageMs(kStageCount);
    for (uint32_t frame = 0; frame < settings.costFrames; ++frame)
    {
        const float azimuth = 0.5f * static_cast<float>(frame);
        const CameraUniforms camera = makeStudyCamera(scene, size.width, size.height, azimuth);

        const auto start = std::chrono::steady_clock::now();
        pass.run(views, camera);
        const auto stop = std::chrono::steady_clock::now();

        if (frame < settings.costWarmup)
        {
            continue;
        }
        wallMs.push_back(std::chrono::duration<double, std::milli>(stop - start).count());

        const ForwardStageTimings & timings = pass.lastStageTimings();
        if (!timings.valid)
        {
            continue;
        }
        for (size_t i = 0; i < kStageCount; ++i)
        {
            stageMs[i].push_back(static_cast<double>(timings.*(kStageFields[i].member)));
        }
    }

    const char * name = mode.name;
    sink.add(name, "wall_median", percentileOrNan(wallMs, 0.5), "ms");
    sink.add(name, "wall_p95", percentileOrNan(wallMs, 0.95), "ms");
    for (size_t i = 0; i < kStageCount; ++i)
    {
        sink.add(name, std::string("stage_") + kStageFields[i].name + "_median",
                 percentileOrNan(stageMs[i], 0.5), "ms");
    }
    // Read after the loop, so the allocations of the warm-up frames are included.
    const uint64_t extraBytes = pass.extraBufferBytes();
    const double gaussianCount = static_cast<double>(scene.cloud->count());
    sink.add(name, "extra_gpu_memory_mb", static_cast<double>(extraBytes) / kBytesPerMiB, "MB");
    sink.add(name, "extra_bytes_per_gaussian",
             gaussianCount > 0.0 ? static_cast<double>(extraBytes) / gaussianCount : kNaN, "bytes");
    if (isStochastic(mode))
    {
        // Per-pixel cost of the stochastic buffers (the history, plus the fixed
        // noise table and params). The ray-form bytes are removed, since they
        // are per Gaussian and reported above. Computed in double so a smaller
        // total cannot wrap.
        const double pixels = static_cast<double>(size.width) * static_cast<double>(size.height);
        const double perPixel =
            (static_cast<double>(extraBytes) - kRayFormBytesPerGaussian * gaussianCount) / pixels;
        sink.add(name, "extra_bytes_per_pixel", perPixel, "bytes");
    }
}

void runCost(Device & device, const CanonicalGaussianBufferViews & views, const StudyScene & scene,
             const Settings & settings, const Size & size, RowSink & sink)
{
    for (const ModeInfo * mode : settings.modes)
    {
        if (mode->gpu)
        {
            runCostForMode(device, views, scene, settings, size, *mode, sink);
        }
    }
}

// ---------------------------------------------------------------------------
// Driver

void printTable(const std::vector<ResultRow> & rows, size_t first)
{
    if (first >= rows.size())
    {
        return;
    }
    std::printf("== %s %s ==\n", rows[first].scene.c_str(), rows[first].size.c_str());
    std::printf("  %-14s %-26s %14s  %s\n", "mode", "metric", "value", "unit");
    for (size_t i = first; i < rows.size(); ++i)
    {
        const ResultRow & row = rows[i];
        if (row.metric.rfind("still", 0) == 0)
        {
            continue; // per-still rows go to the files only
        }
        std::printf("  %-14s %-26s %14s  %s\n", row.mode.c_str(), row.metric.c_str(),
                    formatValue(row.value).c_str(), row.unit.c_str());
    }
    std::printf("\n");
}

void runSceneSize(Device & device, const StudyScene & scene, const Settings & settings, const Size & size,
                  std::vector<ResultRow> & rows)
{
    const size_t first = rows.size();
    RowSink sink(rows, scene.name, sizeLabel(size));

    ForwardPass pass(device, size.width, size.height, testShaderDir(), DepthLayout::None);
    const CanonicalGaussianBufferViews views = scene.cloud->views();

    if (settings.stillsEnabled)
    {
        runStills(pass, views, scene, settings, size, sink);
    }
    if (settings.tileStats)
    {
        runTileStats(pass, views, scene, size, sink);
    }
    if (settings.temporal)
    {
        // One pass per listed mode that carries state across frames. Stable's
        // sticky keys and the stochastic history live in these passes, so a
        // mode's settings never reset another mode's state.
        std::vector<std::unique_ptr<ForwardPass>> statePasses(settings.modes.size());
        for (size_t m = 0; m < settings.modes.size(); ++m)
        {
            if (carriesSequenceState(*settings.modes[m]))
            {
                statePasses[m] = std::make_unique<ForwardPass>(device, size.width, size.height,
                                                               testShaderDir(), DepthLayout::None);
            }
        }
        runTemporal(pass, statePasses, views, scene, settings, size, sink);
    }
    if (settings.cost)
    {
        runCost(device, views, scene, settings, size, sink);
    }

    printTable(rows, first);
}

RunInfo makeRunInfo(const Device & device)
{
    const VkPhysicalDeviceProperties & properties = device.properties();

    RunInfo info;
    info.deviceName = properties.deviceName;
    info.driverVersion = properties.driverVersion;
#ifdef NDEBUG
    info.buildConfig = "Release";
#else
    info.buildConfig = "Debug";
#endif
    return info;
}

void runAll(Device & device, const Settings & settings, RunInfo & info, std::vector<ResultRow> & rows)
{
    for (const std::string & sceneArg : settings.scenes)
    {
        const bool synthetic = sceneArg == kSyntheticName;
        const StudyScene scene =
            synthetic ? study::makeSyntheticScene(device) : study::loadStudyScene(device, sceneArg);
        info.scenes.push_back(RunScene{scene.name, static_cast<uint64_t>(scene.cloud->count())});

        for (const Size & size : settings.sizes)
        {
            runSceneSize(device, scene, settings, size, rows);
        }
    }
}

} // namespace

int main(int argc, char ** argv)
{
    try
    {
        const Settings settings = parseArguments(argc, argv);

        // The PNG crops are written while the runs go, so the directory comes first.
        std::filesystem::create_directories(settings.outDir);

        Device device;
        device.initialize({}, true);

        RunInfo info = makeRunInfo(device);
        std::vector<ResultRow> rows;
        runAll(device, settings, info, rows);

        const std::filesystem::path outDir(settings.outDir);
        study::writeJson((outDir / "results.json").string(), info, rows);
        study::writeCsv((outDir / "results.csv").string(), rows);
        std::printf("sort_study: wrote results.json and results.csv (%zu rows)\n", rows.size());
        return 0;
    }
    catch (const UsageError & e)
    {
        std::fprintf(stderr, "sort_study: %s\n", e.what());
        printUsage(stderr, argv[0]);
        return 1;
    }
    catch (const std::exception & e)
    {
        std::fprintf(stderr, "sort_study: %s\n", e.what());
        return 1;
    }
}
