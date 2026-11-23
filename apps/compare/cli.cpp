// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The command-line front end (apps/compare/cli.h): PSNR, FLIP (LDR or HDR), maps
// and a comparison report for two images.
//
//   sabercat_compare <reference> <test> [--flip-map out.png] [--diff-map out.png]
//                    [--ppd N] [--no-flip] [--hdr] [--tone-mapper aces|hable|reinhard]
//                    [--start-exposure X] [--stop-exposure Y] [--num-exposures N]
//                    [--exposure-map out.png] [--report out.json|out.csv] [--help]
//
// Two PNG files without --hdr or --report take the original path (loadPng), whose
// output is unchanged. Other inputs go through loadImage().

#include "cli.h"

#include "flip_hdr.h"
#include "image_compare.h"
#include "image_data.h"
#include "image_io.h"
#include "report.h"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace sabercat::compare
{

namespace
{

constexpr int kExitOk = 0;
constexpr int kExitFailure = 1;
constexpr int kExitUsage = 2;

// The synopsis lists every flag; the detail lines below it describe each one.
const char * const kUsage =
    "usage: sabercat_compare <reference.png> <test.png> [--flip-map out.png] [--diff-map out.png]\n"
    "                        [--ppd N] [--no-flip] [--hdr] [--tone-mapper aces|hable|reinhard]\n"
    "                        [--start-exposure X] [--stop-exposure Y] [--num-exposures N]\n"
    "                        [--exposure-map out.png] [--report out.json|out.csv] [--help]\n"
    "  --ppd N   FLIP pixels per degree, from 1 to 1000 (default about 67)\n"
    "  --hdr     HDR-FLIP on linear inputs (EXR); the values may exceed 1\n"
    "  --tone-mapper aces|hable|reinhard   tone mapper for --hdr (default aces)\n"
    "  --start-exposure X   first exposure in stops, for --hdr\n"
    "  --stop-exposure Y    last exposure in stops, for --hdr\n"
    "  --num-exposures N    exposures from 2 to 1000, for --hdr\n"
    "  --exposure-map out.png   HDR exposure map, for --hdr\n"
    "  --report out.json|out.csv   comparison report (CSV for a .csv name)\n"
    "  inputs may be .png, .jpg, .jpeg, .webp, .exr or .rgba32f; no arguments opens the GUI\n";

// Strict float parse: the whole text must be one number in [kMinFlipPpd,
// kMaxFlipPpd]. The range test is written so that NaN and infinity fail it.
bool parsePpd(const std::string & text, float & out)
{
    float value = 0.0f;
    const char * const last = text.data() + text.size();
    const std::from_chars_result result = std::from_chars(text.data(), last, value);
    if (text.empty() || result.ec != std::errc() || result.ptr != last)
    {
        return false;
    }
    if (!(value >= kMinFlipPpd && value <= kMaxFlipPpd))
    {
        return false;
    }
    out = value;
    return true;
}

// Strict float parse for an exposure: the whole text must be one finite number.
bool parseFiniteFloat(const std::string & text, float & out)
{
    float value = 0.0f;
    const char * const last = text.data() + text.size();
    const std::from_chars_result result = std::from_chars(text.data(), last, value);
    if (text.empty() || result.ec != std::errc() || result.ptr != last)
    {
        return false;
    }
    if (!std::isfinite(value))
    {
        return false;
    }
    out = value;
    return true;
}

// Strict integer parse for --num-exposures, from kMinHdrExposures to kMaxHdrExposures.
bool parseExposureCount(const std::string & text, int & out)
{
    int value = 0;
    const char * const last = text.data() + text.size();
    const std::from_chars_result result = std::from_chars(text.data(), last, value);
    if (text.empty() || result.ec != std::errc() || result.ptr != last)
    {
        return false;
    }
    if (value < kMinHdrExposures || value > kMaxHdrExposures)
    {
        return false;
    }
    out = value;
    return true;
}

struct Options
{
    std::string referencePath;
    std::string testPath;
    std::string flipMapPath;
    std::string diffMapPath;
    float ppd = 0.0f;
    bool flipMapGiven = false;
    bool diffMapGiven = false;
    bool noFlip = false;

    bool hdr = false;
    ToneMapper toneMapper = ToneMapper::Aces;
    bool toneMapperGiven = false;
    float startExposure = 0.0f;
    bool startGiven = false;
    float stopExposure = 0.0f;
    bool stopGiven = false;
    int numExposures = 0;
    bool numGiven = false;
    std::string exposureMapPath;
    bool exposureMapGiven = false;
    std::string reportPath;
    bool reportGiven = false;
};

// Parses the arguments after the program name. Returns an error message for a
// usage problem, or an empty string on success. help is set for --help.
std::string parseArguments(const std::vector<std::string> & args, Options & options, bool & help)
{
    std::vector<std::string> positional;
    bool ppdGiven = false;
    for (size_t i = 1; i < args.size(); ++i)
    {
        const std::string & arg = args[i];
        if (arg == "--help")
        {
            help = true;
            return std::string();
        }
        if (arg == "--no-flip")
        {
            options.noFlip = true;
        }
        else if (arg == "--hdr")
        {
            options.hdr = true;
        }
        else if (arg == "--flip-map" || arg == "--diff-map" || arg == "--ppd" || arg == "--tone-mapper" ||
                 arg == "--start-exposure" || arg == "--stop-exposure" || arg == "--num-exposures" ||
                 arg == "--exposure-map" || arg == "--report")
        {
            if (i + 1 >= args.size())
            {
                return arg + " needs a value";
            }
            const std::string & value = args[++i];
            if (arg == "--flip-map")
            {
                options.flipMapPath = value;
                options.flipMapGiven = true;
            }
            else if (arg == "--diff-map")
            {
                options.diffMapPath = value;
                options.diffMapGiven = true;
            }
            else if (arg == "--ppd")
            {
                if (!parsePpd(value, options.ppd))
                {
                    return "--ppd must be a number from 1 to 1000, got '" + value + "'";
                }
                ppdGiven = true;
            }
            else if (arg == "--tone-mapper")
            {
                if (!parseToneMapper(value, options.toneMapper))
                {
                    return "--tone-mapper must be aces, hable or reinhard, got '" + value + "'";
                }
                options.toneMapperGiven = true;
            }
            else if (arg == "--start-exposure")
            {
                if (!parseFiniteFloat(value, options.startExposure))
                {
                    return "--start-exposure must be a finite number, got '" + value + "'";
                }
                options.startGiven = true;
            }
            else if (arg == "--stop-exposure")
            {
                if (!parseFiniteFloat(value, options.stopExposure))
                {
                    return "--stop-exposure must be a finite number, got '" + value + "'";
                }
                options.stopGiven = true;
            }
            else if (arg == "--num-exposures")
            {
                if (!parseExposureCount(value, options.numExposures))
                {
                    return "--num-exposures must be an integer from 2 to 1000, got '" + value + "'";
                }
                options.numGiven = true;
            }
            else if (arg == "--exposure-map")
            {
                options.exposureMapPath = value;
                options.exposureMapGiven = true;
            }
            else
            {
                options.reportPath = value;
                options.reportGiven = true;
            }
        }
        else if (arg.size() > 1 && arg.front() == '-')
        {
            return "unknown option '" + arg + "'";
        }
        else
        {
            positional.push_back(arg);
        }
    }

    if (positional.size() != 2)
    {
        return "expected a reference image and a test image";
    }
    options.referencePath = positional[0];
    options.testPath = positional[1];
    if (!ppdGiven)
    {
        options.ppd = defaultFlipPpd();
    }
    if (options.noFlip && options.flipMapGiven)
    {
        return "--flip-map needs FLIP, which --no-flip turns off";
    }
    if (options.hdr && options.noFlip)
    {
        return "--no-flip cannot be used with --hdr";
    }
    if (!options.hdr)
    {
        if (options.toneMapperGiven)
        {
            return "--tone-mapper needs --hdr";
        }
        if (options.startGiven)
        {
            return "--start-exposure needs --hdr";
        }
        if (options.stopGiven)
        {
            return "--stop-exposure needs --hdr";
        }
        if (options.numGiven)
        {
            return "--num-exposures needs --hdr";
        }
        if (options.exposureMapGiven)
        {
            return "--exposure-map needs --hdr";
        }
    }
    if (options.startGiven && options.stopGiven && options.startExposure > options.stopExposure)
    {
        return "--start-exposure must not be greater than --stop-exposure";
    }
    return std::string();
}

void printUsageError(const std::string & message)
{
    std::cerr << "error: " << message << "\n" << kUsage;
}

// Lower-case extension of a UTF-8 path, with the dot. Empty when there is none.
std::string lowerExtension(const std::string & utf8Path)
{
    const size_t separator = utf8Path.find_last_of("/\\");
    const size_t dot = utf8Path.find_last_of('.');
    if (dot == std::string::npos || (separator != std::string::npos && dot < separator))
    {
        return std::string();
    }
    std::string extension = utf8Path.substr(dot);
    for (char & ch : extension)
    {
        if (ch >= 'A' && ch <= 'Z')
        {
            ch = static_cast<char>(ch - 'A' + 'a');
        }
    }
    return extension;
}

bool isPngPath(const std::string & utf8Path)
{
    return lowerExtension(utf8Path) == ".png";
}

// The results the LDR printer returns for the report.
struct LdrOutcome
{
    PsnrResult psnr;
    std::optional<double> flipMean;
};

// The LDR lines: size, PSNR, MSE, FLIP, then the maps. This is the output that
// earlier builds printed for two PNG files, byte for byte.
LdrOutcome printLdr(const Image & reference, const Image & test, const Options & options)
{
    LdrOutcome outcome;
    std::printf("size: %ux%u\n", reference.width, reference.height);

    outcome.psnr = psnr(reference, test);
    if (std::isinf(outcome.psnr.psnrDb))
    {
        std::printf("PSNR: inf dB\n");
    }
    else
    {
        std::printf("PSNR: %.2f dB\n", outcome.psnr.psnrDb);
    }
    std::printf("MSE: %.6g\n", outcome.psnr.mse);

    if (!options.noFlip)
    {
        const FlipResult flipResult = flip(reference, test, options.ppd);
        std::printf("FLIP: %.6f (ppd %.2f)\n", flipResult.meanError, static_cast<double>(options.ppd));
        if (options.flipMapGiven)
        {
            savePngRgb8(options.flipMapPath,
                        colorizeMagma(flipResult.errorMap, reference.width, reference.height));
        }
        outcome.flipMean = flipResult.meanError;
    }

    if (options.diffMapGiven)
    {
        const std::vector<float> difference = absoluteDifferenceMap(reference, test);
        savePngRgb8(options.diffMapPath, colorizeMagma(difference, reference.width, reference.height));
    }
    return outcome;
}

ReportSource makeSource(const ImageData & data, const std::string & path)
{
    ReportSource source;
    source.path = path;
    source.width = data.width;
    source.height = data.height;
    source.bitDepth = data.bitDepth;
    source.encoding = data.encoding == Encoding::Linear ? "linear" : "display";
    source.hasAlpha = data.hasAlpha;
    return source;
}

HdrFlipParams makeHdrParams(const Options & options)
{
    HdrFlipParams params;
    params.toneMapper = options.toneMapper;
    params.ppd = options.ppd;
    if (options.startGiven)
    {
        params.startExposure = options.startExposure;
    }
    if (options.stopGiven)
    {
        params.stopExposure = options.stopExposure;
    }
    if (options.numGiven)
    {
        params.numExposures = options.numExposures;
    }
    return params;
}

// The input checks for the general path: a non-finite sample is an error, and the
// notes and warnings about the input's encoding go to stderr.
void checkInput(const ImageData & data, const std::string & path, bool hdr)
{
    const uint64_t nonFinite = countNonFiniteRgb(data);
    if (nonFinite != 0)
    {
        throw std::runtime_error(path + ": " + std::to_string(nonFinite) + " non-finite samples (NaN or infinity)");
    }
    if (data.encoding == Encoding::Linear && !hdr)
    {
        std::cerr << "note: " << path << " is linear (EXR); compared after the sRGB display transform\n";
    }
    if (data.encoding == Encoding::Display && hdr)
    {
        std::cerr << "warning: --hdr with display-encoded input '" << path
                  << "'; values are decoded from sRGB as linear\n";
    }
}

// The general path: any input format, the report, and HDR.
int runGeneral(const Options & options)
{
    const ImageData referenceData = loadImage(options.referencePath);
    const ImageData testData = loadImage(options.testPath);
    if (referenceData.width != testData.width || referenceData.height != testData.height)
    {
        std::cerr << "error: image sizes differ (" << referenceData.width << "x" << referenceData.height << " vs "
                  << testData.width << "x" << testData.height << ")\n";
        return kExitFailure;
    }
    checkInput(referenceData, options.referencePath, options.hdr);
    checkInput(testData, options.testPath, options.hdr);

    CompareReport report;
    report.reference = makeSource(referenceData, options.referencePath);
    report.test = makeSource(testData, options.testPath);
    report.overlap.width = referenceData.width;
    report.overlap.height = referenceData.height;
    report.ppd = options.ppd;

    const Image reference = displayRgb(referenceData);
    const Image test = displayRgb(testData);

    if (!options.hdr)
    {
        const LdrOutcome outcome = printLdr(reference, test, options);
        report.psnrDb = outcome.psnr.psnrDb;
        report.mse = outcome.psnr.mse;
        report.flipMean = outcome.flipMean;
    }
    else
    {
        const HdrFlipParams params = makeHdrParams(options);
        std::printf("size: %ux%u\n", reference.width, reference.height);

        const PsnrResult psnrResult = psnr(reference, test);
        if (std::isinf(psnrResult.psnrDb))
        {
            std::printf("PSNR: inf dB\n");
        }
        else
        {
            std::printf("PSNR: %.2f dB\n", psnrResult.psnrDb);
        }
        std::printf("MSE: %.6g\n", psnrResult.mse);

        const double linear = linearMse(linearRgb(referenceData), linearRgb(testData));
        std::printf("linear MSE: %.6g\n", linear);

        const HdrFlipResult flipResult = flipHdr(referenceData, testData, params);
        std::printf("FLIP (HDR, %s): %.6f (ppd %.2f, exposures %.4f..%.4f, %d)\n", toneMapperName(options.toneMapper),
                    flipResult.meanError, static_cast<double>(options.ppd),
                    static_cast<double>(flipResult.startExposure), static_cast<double>(flipResult.stopExposure),
                    flipResult.numExposures);
        if (options.flipMapGiven)
        {
            savePngRgb8(options.flipMapPath,
                        colorizeMagma(flipResult.errorMap, reference.width, reference.height));
        }
        if (options.exposureMapGiven)
        {
            savePngRgb8(options.exposureMapPath,
                        colorizeMagma(flipResult.exposureMap, reference.width, reference.height));
        }
        if (options.diffMapGiven)
        {
            const std::vector<float> difference = absoluteDifferenceMap(reference, test);
            savePngRgb8(options.diffMapPath, colorizeMagma(difference, reference.width, reference.height));
        }

        report.psnrDb = psnrResult.psnrDb;
        report.mse = psnrResult.mse;
        report.linearMse = linear;
        report.flipMean = flipResult.meanError;
        ReportExposures exposures;
        exposures.toneMapper = toneMapperName(options.toneMapper);
        exposures.start = flipResult.startExposure;
        exposures.stop = flipResult.stopExposure;
        exposures.count = flipResult.numExposures;
        report.hdr = exposures;
    }

    if (options.reportGiven)
    {
        saveReport(options.reportPath, report);
    }
    return kExitOk;
}

} // namespace

const char * cliUsage()
{
    return kUsage;
}

int runCli(const std::vector<std::string> & args)
{
    Options options;
    bool help = false;
    const std::string usageError = parseArguments(args, options, help);
    if (help)
    {
        std::fputs(kUsage, stdout);
        return kExitOk;
    }
    if (!usageError.empty())
    {
        printUsageError(usageError);
        return kExitUsage;
    }

    try
    {
        if (options.hdr || options.reportGiven || !isPngPath(options.referencePath) || !isPngPath(options.testPath))
        {
            return runGeneral(options);
        }

        // Two PNG files without HDR or a report: the original path.
        const Image reference = loadPng(options.referencePath);
        const Image test = loadPng(options.testPath);
        if (reference.width != test.width || reference.height != test.height)
        {
            std::cerr << "error: image sizes differ (" << reference.width << "x" << reference.height << " vs "
                      << test.width << "x" << test.height << ")\n";
            return kExitFailure;
        }
        printLdr(reference, test, options);
    }
    catch (const std::exception & e)
    {
        std::cerr << "error: " << e.what() << "\n";
        return kExitFailure;
    }
    return kExitOk;
}

} // namespace sabercat::compare
