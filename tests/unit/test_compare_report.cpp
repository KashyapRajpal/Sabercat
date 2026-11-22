// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The comparison report (apps/compare/report.h): JSON and CSV content, key order,
// determinism, infinite PSNR, and the file writer.
#include "unit_test.h"

#include "report.h"

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sabercat::unit
{

namespace
{

using sabercat::compare::CompareReport;
using sabercat::compare::ReportExposures;
using sabercat::compare::ReportSource;

CompareReport makeLdrReport()
{
    CompareReport report;
    report.reference = ReportSource{ "ref.png", 4, 3, 8, "display", false };
    report.test = ReportSource{ "test, copy.png", 4, 3, 16, "display", true };
    report.overlap = { 0, 0, 4, 3 };
    report.ppd = 67.0f;
    report.psnrDb = std::numeric_limits<double>::infinity();
    report.mse = 0.0;
    report.flipMean = 0.25;
    return report;
}

CompareReport makeHdrReport()
{
    CompareReport report;
    report.reference = ReportSource{ "reference.exr", 2, 2, 32, "linear", false };
    report.test = ReportSource{ "test.exr", 2, 2, 32, "linear", false };
    report.overlap = { 0, 0, 2, 2 };
    report.ppd = 67.0f;
    ReportExposures exposures;
    exposures.toneMapper = "aces";
    exposures.start = -12.5f;
    exposures.stop = 0.5f;
    exposures.count = 14;
    report.hdr = exposures;
    report.psnrDb = 31.25;
    report.mse = 0.000750;
    report.linearMse = 0.5;
    report.flipMean = 0.283547;
    return report;
}

// The position of the first occurrence of needle, or npos.
size_t at(const std::string & text, const std::string & needle)
{
    return text.find(needle);
}

// True when each key appears after the one before it.
bool inOrder(const std::string & text, const std::vector<std::string> & keys)
{
    size_t previous = 0;
    for (const std::string & key : keys)
    {
        const size_t position = at(text, key);
        if (position == std::string::npos || (position < previous && previous != 0))
        {
            return false;
        }
        previous = position;
    }
    return true;
}

size_t countChar(const std::string & text, char ch)
{
    size_t count = 0;
    for (const char c : text)
    {
        if (c == ch)
        {
            ++count;
        }
    }
    return count;
}

// The lines of a text, without the line breaks.
std::vector<std::string> linesOf(const std::string & text)
{
    std::vector<std::string> lines;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line))
    {
        lines.push_back(line);
    }
    return lines;
}

} // namespace

void testCompareReport(Context & ctx)
{
    // JSON for an LDR report: fixed key order, the header keys first, and the
    // trailing newline. Infinite PSNR and the absent values are null.
    {
        const std::string json = sabercat::compare::reportToJson(makeLdrReport());
        UNIT_CHECK(ctx, inOrder(json, { "\"format\": \"sabercat.compare-report\"", "\"formatVersion\": 1",
                                        "\"savedWith\"", "\"mode\": \"ldr\"", "\"sources\"", "\"reference\"",
                                        "\"test\"", "\"overlap\"", "\"anchor\": \"topLeft\"", "\"ppd\"",
                                        "\"toneMapper\": null", "\"exposures\": null", "\"metrics\"",
                                        "\"psnrDb\": null", "\"mse\"", "\"linearMse\": null",
                                        "\"flipMean\": 0.25" }));
        UNIT_CHECK(ctx, json.size() > 2 && json.compare(json.size() - 2, 2, "}\n") == 0);
        UNIT_CHECK(ctx, json.find("\n  \"format\"") != std::string::npos);
        UNIT_CHECK(ctx, json.find("\"ppd\": 67.0") != std::string::npos);
        UNIT_CHECK(ctx, json.find("\"path\": \"test, copy.png\"") != std::string::npos);
        UNIT_CHECK(ctx, json.find("\"hasAlpha\": true") != std::string::npos);
        UNIT_CHECK(ctx, json.find("inf") == std::string::npos);
    }

    // JSON for an HDR report: mode, tone mapper, exposures and linear MSE.
    {
        const std::string json = sabercat::compare::reportToJson(makeHdrReport());
        UNIT_CHECK(ctx, json.find("\"mode\": \"hdr\"") != std::string::npos);
        UNIT_CHECK(ctx, json.find("\"toneMapper\": \"aces\"") != std::string::npos);
        UNIT_CHECK(ctx, json.find("\"count\": 14") != std::string::npos);
        UNIT_CHECK(ctx, json.find("\"start\": -12.5") != std::string::npos);
        UNIT_CHECK(ctx, json.find("\"linearMse\": 0.5") != std::string::npos);
        UNIT_CHECK(ctx, json.find("\"psnrDb\": 31.25") != std::string::npos);
    }

    // Identical inputs give identical bytes, in both formats.
    {
        const CompareReport report = makeHdrReport();
        UNIT_CHECK(ctx, sabercat::compare::reportToJson(report) == sabercat::compare::reportToJson(report));
        UNIT_CHECK(ctx, sabercat::compare::reportToCsv(report) == sabercat::compare::reportToCsv(report));
    }

    // CSV: the fixed header, one data row, the same column count in both, "inf" for
    // infinite PSNR, empty cells for absent values, and quoting of a comma in a path.
    {
        const std::string header =
            "mode,reference_path,reference_width,reference_height,reference_bit_depth,reference_encoding,"
            "reference_has_alpha,test_path,test_width,test_height,test_bit_depth,test_encoding,test_has_alpha,"
            "overlap_x,overlap_y,overlap_w,overlap_h,anchor,ppd,tone_mapper,start_exposure,stop_exposure,"
            "exposures,psnr_db,mse,linear_mse,flip_mean";
        const std::string csv = sabercat::compare::reportToCsv(makeLdrReport());
        const std::vector<std::string> lines = linesOf(csv);
        UNIT_CHECK(ctx, lines.size() == 2u);
        UNIT_CHECK(ctx, !csv.empty() && csv.back() == '\n');
        if (lines.size() == 2u)
        {
            UNIT_CHECK(ctx, lines[0] == header);
            UNIT_CHECK(ctx, lines[1].rfind("ldr,ref.png,4,3,8,display,false,\"test, copy.png\",4,3,16,display,true,", 0) == 0);
            UNIT_CHECK(ctx, lines[1].find(",inf,") != std::string::npos);
            // The quoted path has one comma inside quotes, so the field count is the
            // header's count plus that one.
            UNIT_CHECK(ctx, countChar(lines[1], ',') == countChar(lines[0], ',') + 1);
            // LDR: tone mapper, start, stop and count are empty; PSNR is "inf", MSE 0,
            // linear MSE empty and the FLIP mean 0.25.
            UNIT_CHECK(ctx, lines[1].find(",,,,,") != std::string::npos);
            const std::string tail = ",inf,0,,0.25";
            UNIT_CHECK(ctx, lines[1].size() >= tail.size()
                && lines[1].compare(lines[1].size() - tail.size(), tail.size(), tail) == 0);
        }
    }
    {
        const std::string csv = sabercat::compare::reportToCsv(makeHdrReport());
        const std::vector<std::string> lines = linesOf(csv);
        UNIT_CHECK(ctx, lines.size() == 2u);
        if (lines.size() == 2u)
        {
            UNIT_CHECK(ctx, countChar(lines[0], ',') == countChar(lines[1], ','));
            UNIT_CHECK(ctx, lines[1].find(",aces,") != std::string::npos);
            UNIT_CHECK(ctx, lines[1].find(",14,") != std::string::npos);
        }
    }

    // The file writer: .csv in any case gets CSV, anything else JSON, and an
    // unwritable path throws.
    {
        const std::filesystem::path dir = std::filesystem::temp_directory_path();
        const std::filesystem::path csvPath = dir / "sabercat_report_test.CSV";
        const std::filesystem::path jsonPath = dir / "sabercat_report_test.json";
        const CompareReport report = makeHdrReport();
        sabercat::compare::saveReport(csvPath.string(), report);
        sabercat::compare::saveReport(jsonPath.string(), report);

        std::ifstream csvIn(csvPath, std::ios::binary);
        std::stringstream csvText;
        csvText << csvIn.rdbuf();
        std::ifstream jsonIn(jsonPath, std::ios::binary);
        std::stringstream jsonText;
        jsonText << jsonIn.rdbuf();
        UNIT_CHECK(ctx, csvText.str() == sabercat::compare::reportToCsv(report));
        UNIT_CHECK(ctx, jsonText.str() == sabercat::compare::reportToJson(report));

        std::error_code ignored;
        std::filesystem::remove(csvPath, ignored);
        std::filesystem::remove(jsonPath, ignored);

        const std::filesystem::path missing = dir / "sabercat_no_such_dir_for_report" / "report.json";
        bool threw = false;
        try
        {
            sabercat::compare::saveReport(missing.string(), report);
        }
        catch (const std::runtime_error &)
        {
            threw = true;
        }
        UNIT_CHECK(ctx, threw);
    }

    // Video fields: the frame index goes in its source after the existing keys, and the
    // clock goes after metrics. CSV adds three columns at the end, an empty cell for a
    // side that is not a video, and header and row keep the same column count.
    {
        const CompareReport plain = makeHdrReport();
        UNIT_CHECK(ctx, sabercat::compare::reportToJson(plain).find("frameIndex") == std::string::npos);
        UNIT_CHECK(ctx, sabercat::compare::reportToJson(plain).find("timeSeconds") == std::string::npos);
        UNIT_CHECK(ctx, sabercat::compare::reportToCsv(plain).find("time_seconds") == std::string::npos);

        CompareReport report = makeHdrReport();
        report.test.frameIndex = 12;
        report.timeSeconds = 0.4;
        const std::string json = sabercat::compare::reportToJson(report);
        UNIT_CHECK(ctx, inOrder(json, { "\"hasAlpha\": false", "\"frameIndex\": 12", "\"metrics\"",
                                        "\"timeSeconds\": 0.4" }));

        const std::string columns = ",time_seconds,reference_frame_index,test_frame_index";
        const std::vector<std::string> lines = linesOf(sabercat::compare::reportToCsv(report));
        UNIT_CHECK(ctx, lines.size() == 2u);
        if (lines.size() == 2u)
        {
            UNIT_CHECK(ctx, lines[0].size() >= columns.size()
                && lines[0].compare(lines[0].size() - columns.size(), columns.size(), columns) == 0);
            const std::string tail = ",0.4,,12";
            UNIT_CHECK(ctx, lines[1].size() >= tail.size()
                && lines[1].compare(lines[1].size() - tail.size(), tail.size(), tail) == 0);
            UNIT_CHECK(ctx, countChar(lines[0], ',') == countChar(lines[1], ','));
        }
    }
}

} // namespace sabercat::unit
