// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// The comparison report: JSON and CSV text, and the file writer (report.h).

#include "report.h"

#include "sabercat/version.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <ios>
#include <stdexcept>
#include <string>

namespace sabercat::compare
{

namespace
{

using Json = nlohmann::ordered_json;

// A float written with six significant digits, as a double. Keeps 67.0f from
// becoming 67.00000762939453 in the JSON text.
double roundedFloat(float value)
{
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.6g", static_cast<double>(value));
    return std::strtod(buffer, nullptr);
}

Json sourceJson(const ReportSource & source)
{
    Json json = Json::object();
    json["path"] = source.path;
    json["width"] = source.width;
    json["height"] = source.height;
    json["bitDepth"] = source.bitDepth;
    json["encoding"] = source.encoding;
    json["hasAlpha"] = source.hasAlpha;
    if (source.frameIndex.has_value())
    {
        json["frameIndex"] = *source.frameIndex;
    }
    return json;
}

// A double for the JSON text; infinity (and NaN) become null.
Json jsonNumberOrNull(double value)
{
    return std::isfinite(value) ? Json(value) : Json(nullptr);
}

// The CSV number text: %.10g, or "inf" for infinity and "nan" for NaN.
std::string csvNumber(double value)
{
    if (std::isinf(value))
    {
        return "inf";
    }
    if (std::isnan(value))
    {
        return "nan";
    }
    char buffer[64];
    std::snprintf(buffer, sizeof(buffer), "%.10g", value);
    return buffer;
}

// RFC 4180: a field with a comma, quote or line break is quoted, with quotes doubled.
std::string csvField(const std::string & text)
{
    if (text.find_first_of(",\"\r\n") == std::string::npos)
    {
        return text;
    }
    std::string quoted = "\"";
    for (const char ch : text)
    {
        if (ch == '"')
        {
            quoted += '"';
        }
        quoted += ch;
    }
    quoted += '"';
    return quoted;
}

bool endsWithCsv(const std::string & path)
{
    if (path.size() < 4)
    {
        return false;
    }
    std::string tail = path.substr(path.size() - 4);
    for (char & ch : tail)
    {
        if (ch >= 'A' && ch <= 'Z')
        {
            ch = static_cast<char>(ch - 'A' + 'a');
        }
    }
    return tail == ".csv";
}

} // namespace

std::string reportToJson(const CompareReport & report)
{
    Json json = Json::object();
    json["format"] = "sabercat.compare-report";
    json["formatVersion"] = 1;
    json["savedWith"] = Json::object({ { "sabercat", kVersionString } });
    json["mode"] = report.hdr.has_value() ? "hdr" : "ldr";
    json["sources"] = Json::object({ { "reference", sourceJson(report.reference) },
                                     { "test", sourceJson(report.test) } });
    json["overlap"] = Json::object({ { "x", report.overlap.x },
                                     { "y", report.overlap.y },
                                     { "w", report.overlap.width },
                                     { "h", report.overlap.height } });
    json["anchor"] = report.anchor;
    json["ppd"] = roundedFloat(report.ppd);
    if (report.hdr.has_value())
    {
        const ReportExposures & hdr = *report.hdr;
        json["toneMapper"] = hdr.toneMapper;
        json["exposures"] = Json::object({ { "start", roundedFloat(hdr.start) },
                                           { "stop", roundedFloat(hdr.stop) },
                                           { "count", hdr.count } });
    }
    else
    {
        json["toneMapper"] = nullptr;
        json["exposures"] = nullptr;
    }
    json["metrics"] = Json::object({
        { "psnrDb", jsonNumberOrNull(report.psnrDb) },
        { "mse", jsonNumberOrNull(report.mse) },
        { "linearMse", report.linearMse.has_value() ? jsonNumberOrNull(*report.linearMse) : Json(nullptr) },
        { "flipMean", report.flipMean.has_value() ? jsonNumberOrNull(*report.flipMean) : Json(nullptr) },
    });
    if (report.timeSeconds.has_value())
    {
        json["timeSeconds"] = jsonNumberOrNull(*report.timeSeconds);
    }

    return json.dump(2) + "\n";
}

std::string reportToCsv(const CompareReport & report)
{
    std::string header =
        "mode,reference_path,reference_width,reference_height,reference_bit_depth,reference_encoding,"
        "reference_has_alpha,test_path,test_width,test_height,test_bit_depth,test_encoding,test_has_alpha,"
        "overlap_x,overlap_y,overlap_w,overlap_h,anchor,ppd,tone_mapper,start_exposure,stop_exposure,"
        "exposures,psnr_db,mse,linear_mse,flip_mean";

    const auto flag = [](bool value) { return std::string(value ? "true" : "false"); };
    const auto number = [](uint32_t value) { return std::to_string(value); };

    std::string row;
    row += report.hdr.has_value() ? "hdr" : "ldr";
    row += "," + csvField(report.reference.path);
    row += "," + number(report.reference.width) + "," + number(report.reference.height);
    row += "," + number(report.reference.bitDepth) + "," + csvField(report.reference.encoding);
    row += "," + flag(report.reference.hasAlpha);
    row += "," + csvField(report.test.path);
    row += "," + number(report.test.width) + "," + number(report.test.height);
    row += "," + number(report.test.bitDepth) + "," + csvField(report.test.encoding);
    row += "," + flag(report.test.hasAlpha);
    row += "," + number(report.overlap.x) + "," + number(report.overlap.y);
    row += "," + number(report.overlap.width) + "," + number(report.overlap.height);
    row += "," + csvField(report.anchor);
    row += "," + csvNumber(roundedFloat(report.ppd));
    if (report.hdr.has_value())
    {
        const ReportExposures & hdr = *report.hdr;
        row += "," + csvField(hdr.toneMapper);
        row += "," + csvNumber(roundedFloat(hdr.start));
        row += "," + csvNumber(roundedFloat(hdr.stop));
        row += "," + std::to_string(hdr.count);
    }
    else
    {
        row += ",,,,";
    }
    row += "," + csvNumber(report.psnrDb);
    row += "," + csvNumber(report.mse);
    row += "," + (report.linearMse.has_value() ? csvNumber(*report.linearMse) : std::string());
    row += "," + (report.flipMean.has_value() ? csvNumber(*report.flipMean) : std::string());
    // Video columns go at the end, and only when the clock is in the report, so an
    // image-only report keeps its bytes.
    if (report.timeSeconds.has_value())
    {
        const auto frameCell = [](const std::optional<int64_t> & frame)
        {
            return frame.has_value() ? std::to_string(*frame) : std::string();
        };
        header += ",time_seconds,reference_frame_index,test_frame_index";
        row += "," + csvNumber(*report.timeSeconds) + "," + frameCell(report.reference.frameIndex) + ","
            + frameCell(report.test.frameIndex);
    }

    return header + "\n" + row + "\n";
}

void saveReport(const std::string & utf8Path, const CompareReport & report)
{
    const std::string text = endsWithCsv(utf8Path) ? reportToCsv(report) : reportToJson(report);
    std::ofstream out(std::filesystem::u8path(utf8Path), std::ios::binary | std::ios::trunc);
    if (!out)
    {
        throw std::runtime_error(utf8Path + ": cannot write the report");
    }
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    out.close();
    if (!out)
    {
        throw std::runtime_error(utf8Path + ": cannot write the report");
    }
}

} // namespace sabercat::compare
