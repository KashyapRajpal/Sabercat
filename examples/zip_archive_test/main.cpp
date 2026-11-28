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
//      archives are rejected; oversized names throw before a file is created.
//  (e) zip64 writer: with injected limits, each Zip64 field (sizes, offset,
//      entry count) is written only when it passes its limit, with the exact
//      records and markers; the real 65534/65535/65536 entry counts; a CRC
//      with the high bit set.
//  (f) zip64 reader: hand-built archives, valid variants and malformed ones
//      that must be rejected with a clean exception.
//  (g) robustness: every truncation and several byte flips of a zip64
//      archive either parse-and-extract or throw std::exception.
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
}

// --- Zip64 helpers (tests (e) to (g)) ---

constexpr uint32_t kMarker32 = 0xFFFFFFFF;
constexpr uint16_t kMarker16 = 0xFFFF;
constexpr uint32_t kSigLocal = 0x04034b50;
constexpr uint32_t kSigCentral = 0x02014b50;
constexpr uint32_t kSigEocd = 0x06054b50;
constexpr uint32_t kSigEocd64 = 0x06064b50;
constexpr uint32_t kSigLocator = 0x07064b50;
constexpr uint16_t kExtraZip64Id = 0x0001;

// The one entry of the synthetic archives: "a.bin", 8 bytes, STORED.
const std::vector<uint8_t> kSynthData = {1, 2, 3, 4, 5, 6, 7, 8};

// Little-endian reads and writes at a byte offset. Reads are bounds-checked
// (a short archive throws std::out_of_range, which the callers never expect).
template <typename T> T peekAt(const std::vector<uint8_t> & bytes, size_t at)
{
    uint64_t value = 0;
    for (size_t i = 0; i < sizeof(T); ++i)
    {
        value |= static_cast<uint64_t>(bytes.at(at + i)) << (8 * i);
    }
    return static_cast<T>(value);
}

template <typename T> void pokeAt(std::vector<uint8_t> & bytes, size_t at, T value)
{
    for (size_t i = 0; i < sizeof(T); ++i)
    {
        bytes.at(at + i) = static_cast<uint8_t>(static_cast<uint64_t>(value) >> (8 * i));
    }
}

void put64(std::vector<uint8_t> & out, uint64_t value)
{
    put32(out, static_cast<uint32_t>(value & 0xFFFFFFFF));
    put32(out, static_cast<uint32_t>(value >> 32));
}

struct Written
{
    std::vector<uint8_t> bytes;
    bool roundTrip = false;
};

// Writes `files` with `limits`, reads the file back, and reports whether the
// read equals the input map.
Written writeWithLimits(const std::string & name, const FileMap & files, const zip::WriterLimits & limits)
{
    const fs::path path = g_tempDir / name;
    zip::writeStoredZip(path.string(), files, limits);
    Written written;
    written.bytes = readBytes(path);
    written.roundTrip = zip::readZip(path.string(), "ZipArchiveTest") == files;
    return written;
}

// Deterministic, non-trivial bytes; a different seed gives different content.
std::vector<uint8_t> patternBytes(size_t size, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::vector<uint8_t> bytes(size);
    for (uint8_t & value : bytes)
    {
        value = static_cast<uint8_t>(rng());
    }
    return bytes;
}

// Empty entries "e0" .. "e<count - 1>".
FileMap numberedFiles(size_t count)
{
    FileMap files;
    for (size_t i = 0; i < count; ++i)
    {
        files["e" + std::to_string(i)] = {};
    }
    return files;
}

struct LocalHeader
{
    size_t at = 0;
    std::string name;
};

// Walks the local headers from offset 0. A zip64 compressed size (the classic
// field is the marker) is taken from the zip64 extra: id, size, uncompressed, compressed.
std::vector<LocalHeader> walkLocal(const std::vector<uint8_t> & bytes)
{
    std::vector<LocalHeader> headers;
    size_t at = 0;
    while (at + 30 <= bytes.size() && peekAt<uint32_t>(bytes, at) == kSigLocal)
    {
        const uint16_t nameLength = peekAt<uint16_t>(bytes, at + 26);
        const uint16_t extraLength = peekAt<uint16_t>(bytes, at + 28);
        uint64_t compressed = peekAt<uint32_t>(bytes, at + 18);
        if (compressed == kMarker32)
        {
            compressed = peekAt<uint64_t>(bytes, at + 30 + nameLength + 12);
        }
        headers.push_back(LocalHeader{at, std::string(reinterpret_cast<const char *>(bytes.data() + at + 30), nameLength)});
        at += 30 + nameLength + extraLength + compressed;
    }
    return headers;
}

// The classic end record (the file's last 22 bytes, no comment) and, when the
// locator sits 20 bytes before it, the zip64 end record's offset.
struct EndRecord
{
    size_t eocd = 0;
    uint16_t entries = 0;
    uint32_t size = 0;
    uint32_t offset = 0;
    uint64_t zip64 = 0;
};

EndRecord readEndRecord(const std::vector<uint8_t> & bytes)
{
    EndRecord end;
    end.eocd = bytes.size() - 22;
    end.entries = peekAt<uint16_t>(bytes, end.eocd + 10);
    end.size = peekAt<uint32_t>(bytes, end.eocd + 12);
    end.offset = peekAt<uint32_t>(bytes, end.eocd + 16);
    if (peekAt<uint32_t>(bytes, end.eocd - 20) == kSigLocator)
    {
        end.zip64 = peekAt<uint64_t>(bytes, end.eocd - 12); // locator + 8
    }
    return end;
}

bool hasLocator(const std::vector<uint8_t> & bytes)
{
    return peekAt<uint32_t>(bytes, bytes.size() - 42) == kSigLocator;
}

// Central directory start and entry count, from the zip64 record when the classic field is a marker.
uint64_t centralDirOffsetOf(const std::vector<uint8_t> & bytes)
{
    const EndRecord end = readEndRecord(bytes);
    return end.offset == kMarker32 ? peekAt<uint64_t>(bytes, end.zip64 + 48) : end.offset;
}

uint64_t entryCountOf(const std::vector<uint8_t> & bytes)
{
    const EndRecord end = readEndRecord(bytes);
    return end.entries == kMarker16 ? peekAt<uint64_t>(bytes, end.zip64 + 32) : end.entries;
}

// Start offsets of the central-directory headers, in directory order.
std::vector<size_t> centralHeaders(const std::vector<uint8_t> & bytes)
{
    std::vector<size_t> starts;
    size_t at = static_cast<size_t>(centralDirOffsetOf(bytes));
    const uint64_t count = entryCountOf(bytes);
    for (uint64_t i = 0; i < count; ++i)
    {
        starts.push_back(at);
        at += 46 + peekAt<uint16_t>(bytes, at + 28) + peekAt<uint16_t>(bytes, at + 30)
              + peekAt<uint16_t>(bytes, at + 32);
    }
    return starts;
}

std::string centralName(const std::vector<uint8_t> & bytes, size_t at)
{
    return std::string(reinterpret_cast<const char *>(bytes.data() + at + 46), peekAt<uint16_t>(bytes, at + 28));
}

// Parses and extracts every entry; returns the first exception's message, or "" if none is thrown.
std::string parseAndExtract(const std::vector<uint8_t> & bytes)
{
    return thrownMessage([&] {
        const auto entries = zip::parseZip(bytes, "synthetic.zip", "ZipArchiveTest");
        for (const auto & [name, entry] : entries)
        {
            zip::extractZipEntry(bytes, entry, name, "synthetic.zip", "ZipArchiveTest");
        }
    });
}

// Expects a clean rejection whose message contains one of `needles`.
void expectRejected(const std::string & label, const std::vector<uint8_t> & bytes,
                    const std::vector<std::string> & needles)
{
    const std::string message = parseAndExtract(bytes);
    bool matched = false;
    for (const std::string & needle : needles)
    {
        matched = matched || contains(message, needle);
    }
    check(!message.empty() && matched, label + " (\"" + message + "\")");
}

// True when the archive parses and "a.bin" extracts to kSynthData, with no other entries.
bool readsBack(const std::vector<uint8_t> & bytes)
{
    try
    {
        const auto entries = zip::parseZip(bytes, "synthetic.zip", "ZipArchiveTest");
        const auto found = entries.find("a.bin");
        return entries.size() == 1 && found != entries.end()
               && zip::extractZipEntry(bytes, found->second, "a.bin", "synthetic.zip", "ZipArchiveTest")
                      == kSynthData;
    }
    catch (const std::exception & e)
    {
        std::printf("       threw: %s\n", e.what());
        return false;
    }
}

template <typename F> std::vector<uint8_t> patched(const std::vector<uint8_t> & base, F && mutate)
{
    std::vector<uint8_t> bytes = base;
    mutate(bytes);
    return bytes;
}

// --- Test (e): the writer's zip64 records ---

void testZip64Sizes()
{
    std::printf("\n-- (e1) zip64 sizes only (limit 100) --\n");
    FileMap files;
    files["zero"] = {};
    files["small"] = patternBytes(50, 1);
    files["big1"] = patternBytes(101, 2);
    files["big2"] = patternBytes(300, 3);
    const Written out = writeWithLimits("zip64_sizes.zip", files, zip::WriterLimits{100, 0xFFFE});
    check(out.roundTrip, "entries above the size limit read back bit-exactly (zero, small, big1, big2)");
    const std::vector<LocalHeader> locals = walkLocal(out.bytes);
    check(locals.size() == 4, "four local headers are found");
    for (const LocalHeader & header : locals)
    {
        const size_t at = header.at;
        const std::vector<uint8_t> & bytes = out.bytes;
        if (header.name == "big1" || header.name == "big2")
        {
            const uint64_t size = files.at(header.name).size();
            const size_t extra = at + 30 + header.name.size();
            check(peekAt<uint16_t>(bytes, at + 4) == 45, header.name + ": local version needed is 45");
            check(peekAt<uint32_t>(bytes, at + 18) == kMarker32 && peekAt<uint32_t>(bytes, at + 22) == kMarker32,
                  header.name + ": local 32-bit sizes are 0xFFFFFFFF");
            check(peekAt<uint16_t>(bytes, at + 28) == 20, header.name + ": local extra field is 20 bytes");
            check(peekAt<uint16_t>(bytes, extra) == 1 && peekAt<uint16_t>(bytes, extra + 2) == 16
                      && peekAt<uint64_t>(bytes, extra + 4) == size && peekAt<uint64_t>(bytes, extra + 12) == size,
                  header.name + ": zip64 extra is id 1, size 16, uncompressed and compressed = " + std::to_string(size));
        }
        else
        {
            check(peekAt<uint16_t>(bytes, at + 4) == 10 && peekAt<uint16_t>(bytes, at + 28) == 0,
                  header.name + ": local header is classic (version 10, no extra)");
        }
    }
}

void testZip64Offsets()
{
    std::printf("\n-- (e2) zip64 offsets only (limit 150) --\n");
    FileMap files;
    for (int i = 0; i < 5; ++i)
    {
        files["f" + std::to_string(i)] = patternBytes(60, static_cast<uint32_t>(10 + i));
    }
    const Written out = writeWithLimits("zip64_offsets.zip", files, zip::WriterLimits{150, 0xFFFE});
    check(out.roundTrip, "five 60-byte entries past the offset limit read back bit-exactly");
    const std::vector<size_t> starts = centralHeaders(out.bytes);
    check(starts.size() == 5, "five central headers are found");
    // Each local entry is 30 + 2-byte name + 60 bytes of data: 92 bytes, classic.
    int moved = 0;
    bool movedOk = true;
    bool classicOk = true;
    for (size_t i = 0; i < starts.size(); ++i)
    {
        const size_t at = starts[i];
        const uint64_t localAt = 92 * i;
        const size_t extra = at + 46 + peekAt<uint16_t>(out.bytes, at + 28);
        const bool sizesReal = peekAt<uint32_t>(out.bytes, at + 20) == 60 && peekAt<uint32_t>(out.bytes, at + 24) == 60;
        if (localAt > 150)
        {
            ++moved;
            movedOk = movedOk && sizesReal && peekAt<uint32_t>(out.bytes, at + 42) == kMarker32
                      && peekAt<uint16_t>(out.bytes, at + 30) == 12 && peekAt<uint16_t>(out.bytes, extra) == 1
                      && peekAt<uint16_t>(out.bytes, extra + 2) == 8 && peekAt<uint64_t>(out.bytes, extra + 4) == localAt;
        }
        else
        {
            classicOk = classicOk && sizesReal && peekAt<uint32_t>(out.bytes, at + 42) == localAt
                        && peekAt<uint16_t>(out.bytes, at + 30) == 0;
        }
    }
    check(moved == 3 && movedOk,
          "entries at offsets 184, 276 and 368 have offset 0xFFFFFFFF and a 12-byte extra holding the real offset");
    check(classicOk, "entries at offsets 0 and 92 keep classic offsets and have no extra");
}

void testZip64SizeAndOffset()
{
    std::printf("\n-- (e3) size and offset on one entry (limit 100) --\n");
    FileMap files;
    files["a"] = patternBytes(150, 20);
    files["b"] = patternBytes(150, 21);
    const Written out = writeWithLimits("zip64_both.zip", files, zip::WriterLimits{100, 0xFFFE});
    check(out.roundTrip, "entries with a big size and a big offset read back bit-exactly");
    const std::vector<size_t> starts = centralHeaders(out.bytes);
    check(starts.size() == 2 && centralName(out.bytes, starts[1]) == "b", "the second central header is \"b\"");
    // "a" is big-size: its local header is 30 + 1 + 20 zip64 bytes, then 150 data bytes, so "b" starts at 201.
    const size_t at = starts[1];
    const size_t extra = at + 46 + peekAt<uint16_t>(out.bytes, at + 28);
    check(peekAt<uint32_t>(out.bytes, at + 20) == kMarker32 && peekAt<uint32_t>(out.bytes, at + 24) == kMarker32
              && peekAt<uint32_t>(out.bytes, at + 42) == kMarker32,
          "b's central sizes and offset are all 0xFFFFFFFF");
    check(peekAt<uint16_t>(out.bytes, at + 30) == 28 && peekAt<uint16_t>(out.bytes, extra) == 1
              && peekAt<uint16_t>(out.bytes, extra + 2) == 24,
          "b's extra is 28 bytes: id 1, size 24");
    check(peekAt<uint64_t>(out.bytes, extra + 4) == 150 && peekAt<uint64_t>(out.bytes, extra + 12) == 150
              && peekAt<uint64_t>(out.bytes, extra + 20) == 201,
          "b's extra holds uncompressed 150, compressed 150, offset 201, in that order");
    const std::vector<LocalHeader> locals = walkLocal(out.bytes);
    check(locals.size() == 2 && locals[1].at == 201, "b's local header really is at byte 201");
}

void testZip64EntryCount()
{
    std::printf("\n-- (e4) entry count only (max entries 2) --\n");
    FileMap files;
    files["a"] = patternBytes(5, 30);
    files["b"] = patternBytes(5, 31);
    files["c"] = patternBytes(5, 32);
    const Written out = writeWithLimits("zip64_count.zip", files, zip::WriterLimits{0xFFFFFFFE, 2});
    const std::vector<uint8_t> & bytes = out.bytes;
    const size_t size = bytes.size();
    check(out.roundTrip, "three entries with max entries 2 read back bit-exactly");
    const size_t locator = size - 42;
    check(peekAt<uint32_t>(bytes, locator) == kSigLocator, "zip64 locator sits 42 bytes before the end");
    const size_t record = static_cast<size_t>(peekAt<uint64_t>(bytes, locator + 8));
    check(peekAt<uint32_t>(bytes, record) == kSigEocd64, "the locator names a zip64 end record");
    check(peekAt<uint64_t>(bytes, record + 24) == 3 && peekAt<uint64_t>(bytes, record + 32) == 3,
          "the zip64 record counts 3 entries");
    const uint64_t centralSize = peekAt<uint64_t>(bytes, record + 40);
    const uint64_t centralOffset = peekAt<uint64_t>(bytes, record + 48);
    check(centralOffset + centralSize == record && record + 56 == locator,
          "the central directory ends where the zip64 record starts, and the record ends at the locator");
    check(peekAt<uint16_t>(bytes, size - 22 + 8) == kMarker16 && peekAt<uint16_t>(bytes, size - 22 + 10) == kMarker16,
          "the classic end record's entry counts are 0xFFFF");
    check(peekAt<uint32_t>(bytes, size - 22 + 12) == centralSize && peekAt<uint32_t>(bytes, size - 22 + 16) == centralOffset,
          "the classic end record's size and offset hold the real values");
}

void testZip64Boundary()
{
    std::printf("\n-- (e5) boundary at limit 5000 --\n");
    FileMap small;
    small["x.bin"] = patternBytes(4000, 40);
    const Written classic = writeWithLimits("boundary_small_classic.zip", small, zip::WriterLimits{});
    const Written underLimit = writeWithLimits("boundary_small.zip", small, zip::WriterLimits{5000, 0xFFFE});
    check(underLimit.roundTrip && underLimit.bytes == classic.bytes,
          "a 4000-byte entry under limit 5000 is byte-identical to the default-limit (classic) write");

    // A 5000-byte entry is not classic under limit 5000: its central directory starts at byte 5031.
    FileMap exact;
    exact["x.bin"] = patternBytes(5000, 41);
    const Written exactClassic = writeWithLimits("boundary_exact_classic.zip", exact, zip::WriterLimits{});
    const Written exactLimit = writeWithLimits("boundary_exact.zip", exact, zip::WriterLimits{5000, 0xFFFE});
    check(exactLimit.roundTrip && hasLocator(exactLimit.bytes) && exactLimit.bytes != exactClassic.bytes,
          "a 5000-byte entry under limit 5000 is zip64 anyway, since the central directory starts at 5031");

    FileMap over;
    over["x.bin"] = patternBytes(5001, 42);
    const Written overLimit = writeWithLimits("boundary_over.zip", over, zip::WriterLimits{5000, 0xFFFE});
    const Written overClassic = writeWithLimits("boundary_over_classic.zip", over, zip::WriterLimits{});
    check(overLimit.roundTrip && hasLocator(overLimit.bytes) && overLimit.bytes != overClassic.bytes,
          "a 5001-byte entry under limit 5000 is zip64 (locator present) and reads back bit-exactly");
}

void testZip64ZeroAndBig()
{
    std::printf("\n-- (e6) zero-length and big entries together --\n");
    FileMap files;
    files["a0"] = {};
    files["a1"] = patternBytes(300, 50);
    files["a2"] = {};
    files["a3"] = patternBytes(150, 51);
    files["a4"] = patternBytes(10, 52);
    const Written out = writeWithLimits("zip64_mixed_zero.zip", files, zip::WriterLimits{100, 0xFFFE});
    check(out.roundTrip && hasLocator(out.bytes),
          "zero-length entries beside big ones read back bit-exactly, as zip64");
}

void checkEntryCount(size_t count, bool expectLocator)
{
    const std::string label = std::to_string(count) + " entries";
    const FileMap files = numberedFiles(count);
    const Written out = writeWithLimits("count_" + std::to_string(count) + ".zip", files, zip::WriterLimits{});
    check(out.roundTrip, label + " read back to the identical map");
    check(hasLocator(out.bytes) == expectLocator, label + (expectLocator ? ": zip64 locator present" : ": no zip64 locator"));
    const uint16_t classicCount = peekAt<uint16_t>(out.bytes, out.bytes.size() - 22 + 10);
    check(expectLocator ? classicCount == kMarker16 : classicCount == count,
          label + ": classic entry count is " + (expectLocator ? std::string("0xFFFF") : std::to_string(count)));
}

void testZip64RealCounts()
{
    std::printf("\n-- (e7) real entry counts with default limits --\n");
    checkEntryCount(65534, false);
    checkEntryCount(65535, true);
    checkEntryCount(65536, true);
}

void testZip64CrcHighBit()
{
    std::printf("\n-- (e8) CRC-32 with the high bit set --\n");
    std::vector<uint8_t> data;
    uint32_t crc = 0;
    uint32_t counter = 0;
    do
    {
        data.clear();
        put32(data, counter);
        crc = static_cast<uint32_t>(crc32(0L, data.data(), static_cast<uInt>(data.size())));
    } while (crc < 0x80000000u && ++counter < 1000000);
    check(crc >= 0x80000000u, "a 4-byte buffer with CRC-32 " + std::to_string(crc) + " (high bit set) is found");

    FileMap files;
    files["hi.bin"] = data;
    const Written plain = writeWithLimits("crc_high.zip", files, zip::WriterLimits{});
    const Written wide = writeWithLimits("crc_high_zip64.zip", files, zip::WriterLimits{2, 0xFFFE});
    check(plain.roundTrip, "the high-CRC entry reads back (default limits)");
    check(wide.roundTrip, "the high-CRC entry reads back (zip64 via limit 2)");
    const std::vector<const Written *> variants = {&plain, &wide};
    const std::vector<std::string> labels = {"default limits", "zip64 via limit 2"};
    for (size_t i = 0; i < variants.size(); ++i)
    {
        const std::vector<size_t> starts = centralHeaders(variants[i]->bytes);
        check(starts.size() == 1 && peekAt<uint32_t>(variants[i]->bytes, starts[0] + 16) == crc,
              "central CRC-32 field equals zlib's value as unsigned (" + labels[i] + ")");
    }
}

// --- Test (f): the reader on hand-built zip64 archives ---

// Builds a one-entry zip64 archive by hand. markSizes puts the 0xFFFFFFFF marker
// on the central compressed and uncompressed sizes (the zip64 extra then holds
// uncompressed, then compressed). markOffset does the same for the local offset
// (the extra then holds it last). unrelatedFirst puts an unknown extra record
// (id 0x5455, 5 bytes) before the zip64 one. The local header is classic and
// the classic end record carries the markers.
//
// Layout with markSizes and markOffset set (220 bytes): local header 0..43,
// central header 43..122 (its extra at 94), zip64 end record 122..178,
// locator 178..198, classic end record 198..220.
std::vector<uint8_t> buildSyntheticZip64(bool markSizes, bool markOffset, bool unrelatedFirst)
{
    const std::string name = "a.bin";
    const uint32_t crc = static_cast<uint32_t>(crc32(0L, kSynthData.data(), static_cast<uInt>(kSynthData.size())));
    std::vector<uint8_t> out;
    put32(out, kSigLocal);
    put16(out, 10); // version needed
    put16(out, 0);  // flags
    put16(out, 0);  // STORED
    put16(out, 0);  // time
    put16(out, 0x21); // date
    put32(out, crc);
    put32(out, 8);
    put32(out, 8);
    put16(out, 5);
    put16(out, 0);
    out.insert(out.end(), name.begin(), name.end());
    out.insert(out.end(), kSynthData.begin(), kSynthData.end());

    std::vector<uint8_t> extra;
    if (unrelatedFirst)
    {
        const std::vector<uint8_t> payload = {'x', 'y', 'z', 'w', 'v'};
        put16(extra, 0x5455);
        put16(extra, 5);
        extra.insert(extra.end(), payload.begin(), payload.end());
    }
    if (markSizes || markOffset)
    {
        std::vector<uint8_t> payload;
        if (markSizes)
        {
            put64(payload, 8); // uncompressed
            put64(payload, 8); // compressed
        }
        if (markOffset)
        {
            put64(payload, 0); // local header offset
        }
        put16(extra, kExtraZip64Id);
        put16(extra, static_cast<uint16_t>(payload.size()));
        extra.insert(extra.end(), payload.begin(), payload.end());
    }

    const uint64_t centralAt = out.size();
    put32(out, kSigCentral);
    put16(out, 45); // made by
    put16(out, 45); // version needed
    put16(out, 0);  // flags
    put16(out, 0);  // STORED
    put16(out, 0);  // time
    put16(out, 0x21); // date
    put32(out, crc);
    put32(out, markSizes ? kMarker32 : 8u);
    put32(out, markSizes ? kMarker32 : 8u);
    put16(out, 5);
    put16(out, static_cast<uint16_t>(extra.size()));
    put16(out, 0); // comment
    put16(out, 0); // disk
    put16(out, 0); // internal attributes
    put32(out, 0); // external attributes
    put32(out, markOffset ? kMarker32 : 0u);
    out.insert(out.end(), name.begin(), name.end());
    out.insert(out.end(), extra.begin(), extra.end());

    const uint64_t centralSize = out.size() - centralAt;
    const uint64_t recordAt = out.size();
    put32(out, kSigEocd64);
    put64(out, 44); // bytes after this field
    put16(out, 45);
    put16(out, 45);
    put32(out, 0); // this disk
    put32(out, 0); // disk with the central directory
    put64(out, 1); // entries on this disk
    put64(out, 1); // entries in total
    put64(out, centralSize);
    put64(out, centralAt);
    put32(out, kSigLocator);
    put32(out, 0); // disk with the zip64 end record
    put64(out, recordAt);
    put32(out, 1); // total disks
    put32(out, kSigEocd);
    put16(out, 0);
    put16(out, 0);
    put16(out, kMarker16);
    put16(out, kMarker16);
    put32(out, kMarker32);
    put32(out, kMarker32);
    put16(out, 0);
    return out;
}

// Offsets in buildSyntheticZip64(true, true, false).
constexpr size_t kSynthCentralMethod = 53;
constexpr size_t kSynthExtraId = 94;
constexpr size_t kSynthExtraSize = 96;
constexpr size_t kSynthExtraUncompressed = 98;
constexpr size_t kSynthExtraCompressed = 106;
constexpr size_t kSynthExtraOffset = 114;
constexpr size_t kSynthRecord = 122;
constexpr size_t kSynthRecordSize = 126;
constexpr size_t kSynthRecordDisk = 138;
constexpr size_t kSynthRecordEntries = 154;
constexpr size_t kSynthRecordCentralSize = 162;
constexpr size_t kSynthRecordCentralOffset = 170;
constexpr size_t kSynthLocator = 178;
constexpr size_t kSynthLocatorRecord = 186;
constexpr size_t kSynthEocd = 198;
constexpr size_t kSynthEocdDisk = 202;

void testSyntheticZip64()
{
    std::printf("\n-- (f) hand-built zip64 archives, read side --\n");
    const std::vector<uint8_t> valid = buildSyntheticZip64(true, true, false);
    check(valid.size() == 220 && peekAt<uint16_t>(valid, kSynthExtraId) == 1,
          "the full synthetic archive is 220 bytes and its zip64 extra is where the offsets say");

    FileMap expected;
    expected["a.bin"] = kSynthData;
    const fs::path validPath = g_tempDir / "synth_valid.zip";
    writeBytes(validPath, valid);
    check(zip::readZip(validPath.string(), "ZipArchiveTest") == expected,
          "valid: all three central fields marked, zip64 extra holds the real values; readZip returns a.bin");
    check(readsBack(buildSyntheticZip64(false, true, false)),
          "valid: only the offset marker is set (the extra holds one u64)");
    check(readsBack(buildSyntheticZip64(true, true, true)),
          "valid: an unrelated extra record (0x5455) comes before the zip64 one");

    expectRejected("reject: locator missing",
                   patched(valid, [](std::vector<uint8_t> & b) { b.erase(b.begin() + kSynthLocator, b.begin() + kSynthEocd); }),
                   {"zip64"});
    expectRejected("reject: locator signature wrong",
                   patched(valid, [](std::vector<uint8_t> & b) { pokeAt<uint32_t>(b, kSynthLocator, 0x07064b51); }),
                   {"zip64"});
    expectRejected("reject: locator points past the file",
                   patched(valid, [](std::vector<uint8_t> & b) { pokeAt<uint64_t>(b, kSynthLocatorRecord, 0x100000); }),
                   {"zip64"});
    expectRejected("reject: locator points at a bad signature",
                   patched(valid, [](std::vector<uint8_t> & b) { pokeAt<uint64_t>(b, kSynthLocatorRecord, 121); }),
                   {"zip64"});
    expectRejected("reject: zip64 record size field is 43 (< 44)",
                   patched(valid, [](std::vector<uint8_t> & b) { pokeAt<uint64_t>(b, kSynthRecordSize, 43); }),
                   {"zip64"});
    expectRejected("reject: zip64 extra too short for the markers (payload 8, needs 24)",
                   patched(valid, [](std::vector<uint8_t> & b) { pokeAt<uint16_t>(b, kSynthExtraSize, 8); }),
                   {"zip64", "extra"});
    expectRejected("reject: marker set but no 0x0001 extra record",
                   patched(valid, [](std::vector<uint8_t> & b) { pokeAt<uint16_t>(b, kSynthExtraId, 0x7777); }),
                   {"zip64"});
    expectRejected("reject: extra record payload runs past the extra field",
                   patched(valid, [](std::vector<uint8_t> & b) { pokeAt<uint16_t>(b, kSynthExtraSize, 0x0100); }),
                   {"extra"});
    expectRejected("reject: local offset 2^32 + 10 (must not wrap to 10)",
                   patched(valid, [](std::vector<uint8_t> & b) { pokeAt<uint64_t>(b, kSynthExtraOffset, 0x100000000ull + 10); }),
                   {"no valid local header"});
    expectRejected("reject: compressed size 2^32 + 8 (must not wrap to 8)",
                   patched(valid, [](std::vector<uint8_t> & b) { pokeAt<uint64_t>(b, kSynthExtraCompressed, 0x100000000ull + 8); }),
                   {"runs past"});
    expectRejected("reject: zip64 central directory size outside the archive",
                   patched(valid, [](std::vector<uint8_t> & b) { pokeAt<uint64_t>(b, kSynthRecordCentralSize, 0x1000000); }),
                   {"outside"});
    expectRejected("reject: zip64 central directory offset outside the archive",
                   patched(valid, [](std::vector<uint8_t> & b) { pokeAt<uint64_t>(b, kSynthRecordCentralOffset, 0x100000); }),
                   {"outside"});
    expectRejected("reject: zip64 entry count 2^63 - 1",
                   patched(valid, [](std::vector<uint8_t> & b) { pokeAt<uint64_t>(b, kSynthRecordEntries, 0x7FFFFFFFFFFFFFFFull); }),
                   {"corrupt"});
    expectRejected("reject: DEFLATE entry claiming 2^63 - 1 uncompressed bytes",
                   patched(valid, [](std::vector<uint8_t> & b) {
                       pokeAt<uint16_t>(b, kSynthCentralMethod, 8);
                       pokeAt<uint64_t>(b, kSynthExtraUncompressed, 0x7FFFFFFFFFFFFFFFull);
                   }),
                   {"impossible DEFLATE expansion"});
    expectRejected("reject: zip64 record disk number is 1 (multi-disk)",
                   patched(valid, [](std::vector<uint8_t> & b) { pokeAt<uint32_t>(b, kSynthRecordDisk, 1); }),
                   {"zip64", "multi-disk"});
    expectRejected("reject: classic end record disk number is 1 (multi-disk)",
                   patched(valid, [](std::vector<uint8_t> & b) { pokeAt<uint16_t>(b, kSynthEocdDisk, 1); }),
                   {"multi-disk"});
}

// --- Test (g): robustness over every truncation and byte flip ---

void testFuzz()
{
    std::printf("\n-- (g) robustness fuzz: truncations and byte flips --\n");
    FileMap files;
    files["a.bin"] = patternBytes(16, 60);
    files["b.bin"] = {};
    files["c.bin"] = patternBytes(300, 61);
    files["d.bin"] = patternBytes(120, 62);
    files["e.bin"] = patternBytes(64, 63);
    const Written base = writeWithLimits("fuzz_base.zip", files, zip::WriterLimits{200, 0xFFFE});
    const std::vector<uint8_t> & good = base.bytes;
    const size_t size = good.size();
    check(base.roundTrip && hasLocator(good) && size > 900 && size < 1400,
          "the fuzz base is a zip64 archive of " + std::to_string(size) + " bytes and reads back");
    check(parseAndExtract(good).empty(), "the unmodified fuzz base parses and extracts cleanly");

    size_t cases = 0;
    size_t threw = 0;
    size_t succeeded = 0;
    size_t other = 0;
    const auto runCase = [&](const std::vector<uint8_t> & bytes) {
        ++cases;
        try
        {
            const auto entries = zip::parseZip(bytes, "fuzz.zip", "ZipArchiveTest");
            for (const auto & [name, entry] : entries)
            {
                zip::extractZipEntry(bytes, entry, name, "fuzz.zip", "ZipArchiveTest");
            }
            ++succeeded;
        }
        catch (const std::exception &)
        {
            ++threw;
        }
        catch (...)
        {
            ++other;
        }
    };

    const uint8_t * source = good.data();
    for (size_t keep = 0; keep < size; ++keep)
    {
        runCase(std::vector<uint8_t>(source, source + keep));
    }
    std::printf("  truncations: %zu cases, %zu threw, %zu succeeded\n", cases, threw, succeeded);

    const uint8_t masks[] = {0x01, 0x80, 0xFF};
    for (size_t at = 0; at < size; ++at)
    {
        for (const uint8_t mask : masks)
        {
            std::vector<uint8_t> copy = good;
            copy[at] = static_cast<uint8_t>(copy[at] ^ mask);
            runCase(copy);
        }
    }
    std::printf("  total: %zu cases (archive %zu bytes), %zu threw std::exception, %zu returned, %zu other\n",
                cases, size, threw, succeeded, other);
    check(cases == 4 * size && other == 0, "the fuzz loop completed; every case returned or threw std::exception");
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
        testZip64Sizes();
        testZip64Offsets();
        testZip64SizeAndOffset();
        testZip64EntryCount();
        testZip64Boundary();
        testZip64ZeroAndBig();
        testZip64RealCounts();
        testZip64CrcHighBit();
        testSyntheticZip64();
        testFuzz();
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
