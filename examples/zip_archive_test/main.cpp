// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Headless test for the internal ZIP module (zip::writeStoredZip, zip::readZip).
//
//  (a) round trip: STORED writes read back bit-exactly -- nested paths, an
//      empty entry, UTF-8 names, a 1.5 MB entry -- and a 0-byte map.
//  (b) determinism: two writes of one map are byte-identical files.
//  (c) DEFLATE: an archive whose entries are DEFLATE-compressed by zlib here
//      (not by the code under test) reads back bit-exactly.
//  (d) rejection: a flipped data byte fails the CRC-32 check; truncated
//      archives are rejected; oversized names and too many entries throw
//      before a file is created.
//
// The external-tool check (Expand-Archive) is manual and is not run here.

#include <zlib.h>

#include <cstdint>
#include <cstdio>
#include <exception>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "sabercat/file_map.h"
#include "zip_archive.h"

using namespace sabercat;
namespace fs = std::filesystem;

namespace
{

int g_failures = 0;
fs::path g_tempDir;

void check(bool ok, const std::string & what)
{
    std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok)
    {
        ++g_failures;
    }
}

std::vector<uint8_t> readBytes(const fs::path & path)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    std::vector<uint8_t> bytes(static_cast<size_t>(file.tellg()));
    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    return bytes;
}

void writeBytes(const fs::path & path, const std::vector<uint8_t> & bytes)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    file.write(reinterpret_cast<const char *>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
}

// Returns the message of the exception `action` throws, or "" if it does not throw.
template <typename F> std::string thrownMessage(F && action)
{
    try
    {
        action();
    }
    catch (const std::exception & e)
    {
        return e.what();
    }
    return "";
}

bool contains(const std::string & text, const std::string & needle)
{
    return text.find(needle) != std::string::npos;
}

FileMap sampleFiles()
{
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> byte(0, 255);
    FileMap files;
    files["manifest.json"] = std::vector<uint8_t>{'{', '}', '\n'};
    files["empty.bin"] = {};
    files["layers/0/means.bin"] = std::vector<uint8_t>(64, 0xAB);
    files["layers/1/sh_rest.bin"] = std::vector<uint8_t>{0, 1, 2, 3, 0xFF, 0x80, 0x0D, 0x0A};
    files["caf\xC3\xA9/\xE2\x9C\x93.txt"] = std::vector<uint8_t>{'o', 'k'}; // UTF-8 names
    std::vector<uint8_t> big(1536 * 1024 + 17);
    for (uint8_t & value : big)
    {
        value = static_cast<uint8_t>(byte(rng));
    }
    files["big.bin"] = big;
    return files;
}

// --- A minimal ZIP writer for DEFLATE entries, so the reader's DEFLATE path is
// fed bytes this test produced with zlib directly. ---
void put16(std::vector<uint8_t> & out, uint16_t value)
{
    out.push_back(static_cast<uint8_t>(value));
    out.push_back(static_cast<uint8_t>(value >> 8));
}

void put32(std::vector<uint8_t> & out, uint32_t value)
{
    put16(out, static_cast<uint16_t>(value));
    put16(out, static_cast<uint16_t>(value >> 16));
}

void writeDeflateZip(const fs::path & path, const FileMap & files)
{
    std::vector<uint8_t> archive;
    struct Central
    {
        std::string name;
        uint32_t crc, compressed, uncompressed, offset;
    };
    std::vector<Central> central;
    for (const auto & [name, data] : files)
    {
        z_stream stream{};
        // Negative window bits: raw DEFLATE, which is what ZIP stores.
        deflateInit2(&stream, Z_DEFAULT_COMPRESSION, Z_DEFLATED, -MAX_WBITS, 8, Z_DEFAULT_STRATEGY);
        std::vector<uint8_t> compressed(deflateBound(&stream, data.size()) + 16);
        stream.next_in = const_cast<Bytef *>(data.data());
        stream.avail_in = static_cast<uInt>(data.size());
        stream.next_out = compressed.data();
        stream.avail_out = static_cast<uInt>(compressed.size());
        const int status = deflate(&stream, Z_FINISH);
        const uint32_t compressedSize = static_cast<uint32_t>(stream.total_out);
        deflateEnd(&stream);
        if (status != Z_STREAM_END)
        {
            throw std::runtime_error("test deflate failed for " + name);
        }
        compressed.resize(compressedSize);
        const uint32_t crc = static_cast<uint32_t>(crc32(0L, data.data(), static_cast<uInt>(data.size())));
        const uint32_t offset = static_cast<uint32_t>(archive.size());
        put32(archive, 0x04034b50);
        put16(archive, 20);
        put16(archive, 0x0800);
        put16(archive, 8); // DEFLATE
        put16(archive, 0);
        put16(archive, 0x21);
        put32(archive, crc);
        put32(archive, compressedSize);
        put32(archive, static_cast<uint32_t>(data.size()));
        put16(archive, static_cast<uint16_t>(name.size()));
        put16(archive, 0);
        archive.insert(archive.end(), name.begin(), name.end());
        archive.insert(archive.end(), compressed.begin(), compressed.end());
        central.push_back({name, crc, compressedSize, static_cast<uint32_t>(data.size()), offset});
    }
    const uint32_t centralOffset = static_cast<uint32_t>(archive.size());
    for (const Central & c : central)
    {
        put32(archive, 0x02014b50);
        put16(archive, 20);
        put16(archive, 20);
        put16(archive, 0x0800);
        put16(archive, 8);
        put16(archive, 0);
        put16(archive, 0x21);
        put32(archive, c.crc);
        put32(archive, c.compressed);
        put32(archive, c.uncompressed);
        put16(archive, static_cast<uint16_t>(c.name.size()));
        put16(archive, 0);
        put16(archive, 0);
        put16(archive, 0);
        put16(archive, 0);
        put32(archive, 0);
        put32(archive, c.offset);
        archive.insert(archive.end(), c.name.begin(), c.name.end());
    }
    const uint32_t centralSize = static_cast<uint32_t>(archive.size()) - centralOffset;
    put32(archive, 0x06054b50);
    put16(archive, 0);
    put16(archive, 0);
    put16(archive, static_cast<uint16_t>(central.size()));
    put16(archive, static_cast<uint16_t>(central.size()));
    put32(archive, centralSize);
    put32(archive, centralOffset);
    put16(archive, 0);
    writeBytes(path, archive);
}

void testRoundTrip()
{
    std::printf("\n-- (a) STORED round trip --\n");
    const FileMap files = sampleFiles();
    const fs::path path = g_tempDir / "roundtrip.zip";
    zip::writeStoredZip(path.string(), files);
    const FileMap back = zip::readZip(path.string(), "ZipArchiveTest");
    check(back.size() == files.size(), "round trip keeps " + std::to_string(files.size()) + " entries");
    bool same = back.size() == files.size();
    for (const auto & [name, data] : files)
    {
        const auto it = back.find(name);
        same = same && it != back.end() && it->second == data;
    }
    check(same, "every entry (nested, empty, UTF-8 names, 1.5 MB) reads back bit-exactly");
    check(back.count("empty.bin") == 1 && back.at("empty.bin").empty(), "empty entry round trips as 0 bytes");

    const fs::path empty = g_tempDir / "empty.zip";
    zip::writeStoredZip(empty.string(), FileMap{});
    check(fs::file_size(empty) == 22, "a 0-entry map writes a bare 22-byte end record");
    check(zip::readZip(empty.string(), "ZipArchiveTest").empty(), "a 0-entry archive reads back as an empty map");
}

void testDeterminism()
{
    std::printf("\n-- (b) determinism --\n");
    const FileMap files = sampleFiles();
    const fs::path a = g_tempDir / "det_a.zip";
    const fs::path b = g_tempDir / "det_b.zip";
    zip::writeStoredZip(a.string(), files);
    zip::writeStoredZip(b.string(), files);
    const auto bytesA = readBytes(a);
    const auto bytesB = readBytes(b);
    check(bytesA == bytesB, "two writes of the same map are byte-identical (" + std::to_string(bytesA.size()) + " bytes)");
}

void testDeflate()
{
    std::printf("\n-- (c) DEFLATE entries --\n");
    FileMap files;
    const std::string repeated = "the quick brown fox jumps over the lazy dog; ";
    std::vector<uint8_t> text;
    for (int i = 0; i < 2000; ++i)
    {
        text.insert(text.end(), repeated.begin(), repeated.end());
    }
    files["text.txt"] = text;
    std::mt19937 rng(11);
    std::vector<uint8_t> noise(300000);
    for (uint8_t & value : noise)
    {
        value = static_cast<uint8_t>(rng());
    }
    files["noise.bin"] = noise;
    files["nested/empty.bin"] = {};

    const fs::path path = g_tempDir / "deflate.zip";
    writeDeflateZip(path, files);
    check(fs::file_size(path) < text.size() / 4 + noise.size() + 4096,
          "the test's DEFLATE archive is compressed (" + std::to_string(fs::file_size(path)) + " bytes)");
    const FileMap back = zip::readZip(path.string(), "ZipArchiveTest");
    check(back == files, "DEFLATE entries read back bit-exactly (text, noise, empty)");
}

void testRejections()
{
    std::printf("\n-- (d) rejection --\n");
    FileMap files;
    files["a.bin"] = std::vector<uint8_t>{1, 2, 3, 4, 5, 6, 7, 8};
    files["b.bin"] = std::vector<uint8_t>{9, 9, 9};
    const fs::path good = g_tempDir / "reject_good.zip";
    zip::writeStoredZip(good.string(), files);
    check(zip::readZip(good.string(), "ZipArchiveTest") == files, "the unmodified archive reads back");

    // Local header of "a.bin" is 30 bytes + a 5-byte name, so its data starts at 35.
    {
        std::vector<uint8_t> bytes = readBytes(good);
        bytes[35 + 3] ^= 0x01;
        const fs::path flipped = g_tempDir / "reject_flipped.zip";
        writeBytes(flipped, bytes);
        const std::string message = thrownMessage([&] { zip::readZip(flipped.string(), "ZipArchiveTest"); });
        check(contains(message, "CRC-32") && contains(message, "'a.bin'"),
              "a flipped data byte fails the CRC-32 check (\"" + message.substr(0, message.find(" at ")) + "\")");
    }
    {
        const fs::path truncated = g_tempDir / "reject_truncated.zip";
        const auto bytes = readBytes(good);
        for (size_t keep : {bytes.size() - 1, bytes.size() / 2, size_t{10}})
        {
            writeBytes(truncated, std::vector<uint8_t>(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(keep)));
            const std::string message = thrownMessage([&] { zip::readZip(truncated.string(), "ZipArchiveTest"); });
            check(!message.empty() && contains(message, "ZipArchiveTest"),
                  "a truncated archive (" + std::to_string(keep) + " of " + std::to_string(bytes.size())
                      + " bytes) is rejected");
        }
    }
    {
        // Who-prefixed messages: the caller's name, not a hard-coded one.
        const std::string message = thrownMessage([&] { zip::readZip((g_tempDir / "missing.zip").string(), "Mine"); });
        check(contains(message, "Mine: failed to open ZIP"), "a missing file names the caller in its message");
    }
    {
        const fs::path target = g_tempDir / "reject_long_name.zip";
        FileMap longName;
        longName[std::string(70000, 'n')] = std::vector<uint8_t>{1};
        const std::string message = thrownMessage([&] { zip::writeStoredZip(target.string(), longName); });
        check(contains(message, "65535") && !fs::exists(target),
              "an entry name over 65535 bytes throws naming the limit, and no file is left behind");
    }
    {
        const fs::path target = g_tempDir / "reject_many.zip";
        FileMap many;
        for (size_t i = 0; i < 65535; ++i)
        {
            many["e" + std::to_string(i)] = {};
        }
        const std::string message = thrownMessage([&] { zip::writeStoredZip(target.string(), many); });
        check(contains(message, "65534") && !fs::exists(target),
              "65535 entries throw (a non-zip64 archive holds at most 65534), and no file is left behind");
    }
}

} // namespace

int main()
{
    g_tempDir = fs::temp_directory_path() / "sabercat_zip_archive_test";
    fs::remove_all(g_tempDir);
    fs::create_directories(g_tempDir);

    try
    {
        testRoundTrip();
        testDeterminism();
        testDeflate();
        testRejections();
    }
    catch (const std::exception & e)
    {
        std::printf("[FAIL] unexpected exception: %s\n", e.what());
        ++g_failures;
    }

    fs::remove_all(g_tempDir);
    std::printf("\n%s (%d failure%s)\n",
                g_failures == 0 ? "zip_archive_test OK" : "zip_archive_test FAILED",
                g_failures,
                g_failures == 1 ? "" : "s");
    return g_failures == 0 ? 0 : 1;
}
