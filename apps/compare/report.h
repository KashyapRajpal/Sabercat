// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The comparison report that sabercat_compare writes with --report (apps/compare).
// Deterministic: the same inputs give the same bytes, and nothing in it is a time
// stamp. JSON (sabercat.compare-report, formatVersion 1) or one-row CSV.
#pragma once

#include <cstdint>
#include <optional>
#include <string>

namespace sabercat::compare
{

// One input image, as the report describes it.
struct ReportSource
{
    std::string path;      // as given on the command line (UTF-8)
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t bitDepth = 0; // bits per channel in the file (ImageData::bitDepth)
    std::string encoding;  // "display" or "linear"
    bool hasAlpha = false;
    // The frame shown. Set only for a video source (the frame index on the window's clock).
    std::optional<int64_t> frameIndex;
};

// The area that was compared. The CLI compares same-size images, so this is the
// whole image and the anchor is "topLeft".
struct ReportRect
{
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t width = 0;
    uint32_t height = 0;
};

// The HDR settings and exposure range of an HDR run.
struct ReportExposures
{
    std::string toneMapper; // "aces", "hable" or "reinhard"
    float start = 0.0f;
    float stop = 0.0f;
    int count = 0;
};

struct CompareReport
{
    ReportSource reference;
    ReportSource test;
    ReportRect overlap;
    std::string anchor = "topLeft";
    float ppd = 0.0f;
    // Set for an HDR run; absent for LDR.
    std::optional<ReportExposures> hdr;
    // PSNR in dB. +infinity (identical images) is null in JSON and "inf" in CSV.
    double psnrDb = 0.0;
    double mse = 0.0;
    // Mean squared error on the linear values. Set for HDR only.
    std::optional<double> linearMse;
    // Mean FLIP error. Absent with --no-flip.
    std::optional<double> flipMean;
    // The global clock in seconds. Set only when A or B is a video.
    std::optional<double> timeSeconds;
};

// JSON text: two-space indent, keys in a fixed order, a trailing newline.
std::string reportToJson(const CompareReport & report);

// CSV text: a header row and one data row, columns in a fixed order, a trailing
// newline. Empty cells for absent values.
std::string reportToCsv(const CompareReport & report);

// Writes the report to a UTF-8 path. A path ending in .csv (any case) gets CSV;
// anything else gets JSON. Throws std::runtime_error when the file cannot be
// written.
void saveReport(const std::string & utf8Path, const CompareReport & report);

} // namespace sabercat::compare
