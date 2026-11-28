// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "study_report.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace sabercat::study
{
namespace
{

std::string fileNameOf(const std::string & path)
{
    return std::filesystem::path(path).filename().string();
}

std::ofstream openOutput(const std::string & path)
{
    std::ofstream out(path, std::ios::binary);
    if (!out)
    {
        throw std::runtime_error("cannot write " + fileNameOf(path));
    }
    return out;
}

// JSON string body (without the quotes): escapes quote, backslash and control
// characters. Other bytes, including UTF-8, pass through unchanged.
std::string jsonEscape(const std::string & text)
{
    std::string out;
    out.reserve(text.size() + 2);
    for (const char ch : text)
    {
        const unsigned char c = static_cast<unsigned char>(ch);
        switch (c)
        {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (c < 0x20)
            {
                char buffer[8];
                std::snprintf(buffer, sizeof(buffer), "\\u%04x", static_cast<unsigned int>(c));
                out += buffer;
            }
            else
            {
                out += ch;
            }
        }
    }
    return out;
}

std::string jsonString(const std::string & text)
{
    return "\"" + jsonEscape(text) + "\"";
}

// JSON has no NaN or infinity, so non-finite values become null.
std::string jsonNumber(double value)
{
    if (!std::isfinite(value))
    {
        return "null";
    }
    char buffer[32];
    std::snprintf(buffer, sizeof(buffer), "%.9g", value);
    return buffer;
}

std::string csvNumber(double value)
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
    std::snprintf(buffer, sizeof(buffer), "%.9g", value);
    return buffer;
}

// RFC 4180: a field containing a comma, quote or line break is quoted, with
// embedded quotes doubled.
std::string csvField(const std::string & text)
{
    if (text.find_first_of(",\"\r\n") == std::string::npos)
    {
        return text;
    }
    std::string out = "\"";
    for (const char ch : text)
    {
        if (ch == '"')
        {
            out += '"';
        }
        out += ch;
    }
    out += '"';
    return out;
}

} // namespace

void writeJson(const std::string & path, const RunInfo & info, const std::vector<ResultRow> & rows)
{
    std::ofstream out = openOutput(path);

    char driverHex[16];
    std::snprintf(driverHex, sizeof(driverHex), "0x%08x", static_cast<unsigned int>(info.driverVersion));

    out << "{\n";
    out << "  \"device\": {\"name\": " << jsonString(info.deviceName)
        << ", \"driverVersion\": " << jsonString(driverHex) << "},\n";
    out << "  \"buildConfig\": " << jsonString(info.buildConfig) << ",\n";

    out << "  \"scenes\": [";
    for (size_t i = 0; i < info.scenes.size(); ++i)
    {
        const RunScene & scene = info.scenes[i];
        out << (i == 0 ? "\n" : ",\n");
        out << "    {\"file\": " << jsonString(scene.file) << ", \"gaussians\": " << scene.gaussianCount << "}";
    }
    out << (info.scenes.empty() ? "],\n" : "\n  ],\n");

    out << "  \"rows\": [";
    for (size_t i = 0; i < rows.size(); ++i)
    {
        const ResultRow & row = rows[i];
        out << (i == 0 ? "\n" : ",\n");
        out << "    {\"scene\": " << jsonString(row.scene) << ", \"size\": " << jsonString(row.size)
            << ", \"mode\": " << jsonString(row.mode) << ", \"metric\": " << jsonString(row.metric)
            << ", \"value\": " << jsonNumber(row.value) << ", \"unit\": " << jsonString(row.unit) << "}";
    }
    out << (rows.empty() ? "]\n" : "\n  ]\n");
    out << "}\n";

    if (!out)
    {
        throw std::runtime_error("failed writing " + fileNameOf(path));
    }
}

void writeCsv(const std::string & path, const std::vector<ResultRow> & rows)
{
    std::ofstream out = openOutput(path);

    out << "scene,size,mode,metric,value,unit\n";
    for (const ResultRow & row : rows)
    {
        out << csvField(row.scene) << ',' << csvField(row.size) << ',' << csvField(row.mode) << ','
            << csvField(row.metric) << ',' << csvNumber(row.value) << ',' << csvField(row.unit) << '\n';
    }

    if (!out)
    {
        throw std::runtime_error("failed writing " + fileNameOf(path));
    }
}

} // namespace sabercat::study
