// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace sabercat::study
{

// One measured value. The still index, when there is one, is part of `metric`
// (for example "still0.mae" or "mean.psnr_db").
struct ResultRow
{
    std::string scene; // file name only, or "synthetic"
    std::string size;  // "WxH"
    std::string mode;
    std::string metric;
    double value = 0.0; // NaN and infinities are written as null (JSON) or nan/inf (CSV)
    std::string unit;
};

struct RunScene
{
    std::string file; // file name only
    uint64_t gaussianCount = 0;
};

struct RunInfo
{
    std::string deviceName;
    std::string buildConfig; // "Debug" or "Release"
    std::vector<RunScene> scenes;
    uint32_t driverVersion = 0; // written as hex
};

// Writes the run description and every row as one JSON document.
void writeJson(const std::string & path, const RunInfo & info, const std::vector<ResultRow> & rows);

// Writes a header line and one line per row.
void writeCsv(const std::string & path, const std::vector<ResultRow> & rows);

} // namespace sabercat::study
