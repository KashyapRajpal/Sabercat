// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#include "zip_archive.h"

#include <zlib.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

#include "sabercat/vk_check.h"

namespace sabercat::zip
{

namespace
{

using Bytes = std::vector<uint8_t>;

// --- ZIP record layout (APPNOTE.TXT section 4.3 / 4.4) ---
static constexpr uint32_t kZipEndOfCentralDirSig = 0x06054b50;
static constexpr uint32_t kZipCentralDirSig = 0x02014b50;
static constexpr uint32_t kZipLocalHeaderSig = 0x04034b50;
static constexpr uint64_t kZipEndOfCentralDirSize = 22;
static constexpr uint64_t kZipCentralDirHeaderSize = 46;
static constexpr uint64_t kZipLocalHeaderSize = 30;
static constexpr uint64_t kZipMaxCommentSize = 0xFFFF;
static constexpr uint16_t kZipMethodStored = 0;
static constexpr uint16_t kZipMethodDeflate = 8;
static constexpr uint16_t kZipFlagEncrypted = 0x1;

// Writer constants. Version 1.0 is all a STORED entry needs to extract.
static constexpr uint16_t kZipVersionNeeded = 10;
static constexpr uint16_t kZipVersionMadeBy = 20;
static constexpr uint16_t kZipFlagUtf8Names = 0x0800;
// DOS date for 1980-01-01 (day 1, month 1, year 0 from 1980), time 00:00:00.
// Fixed so two writes of the same FileMap are byte-identical.
static constexpr uint16_t kDosTime = 0;
static constexpr uint16_t kDosDate = (1 << 5) | 1;
// 0xFFFF entries and 0xFFFFFFFF sizes/offsets are the zip64 markers, so a
// non-zip64 archive can hold one fewer of each.
static constexpr uint64_t kZipMaxEntries = 0xFFFE;
static constexpr uint64_t kZipMaxField32 = 0xFFFFFFFE;

template <typename T> T readAt(const uint8_t * data, uint64_t offset)
{
    T value;
    std::memcpy(&value, data + offset, sizeof(T));
    return value;
}

void putU16(Bytes & out, uint16_t value)
{
    out.push_back(static_cast<uint8_t>(value & 0xFF));
    out.push_back(static_cast<uint8_t>(value >> 8));
}

void putU32(Bytes & out, uint32_t value)
{
    putU16(out, static_cast<uint16_t>(value & 0xFFFF));
    putU16(out, static_cast<uint16_t>(value >> 16));
}

Bytes readArchiveFile(const std::string & path, const std::string & who)
{
    std::ifstream file(std::filesystem::u8path(path), std::ios::binary | std::ios::ate);
    VK_VALIDATE(file.is_open(), who + ": failed to open ZIP " + path);
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    VK_VALIDATE(size > 0, who + ": ZIP is empty: " + path);
    Bytes bytes(static_cast<size_t>(size));
    file.read(reinterpret_cast<char *>(bytes.data()), size);
    VK_VALIDATE(file.gcount() == size, who + ": truncated read of " + path);
    return bytes;
}

} // namespace

std::map<std::string, ZipEntry> parseZip(const Bytes & archive, const std::string & path,
                                         const std::string & who)
{
    const uint8_t * data = archive.data();
    const uint64_t size = archive.size();
    VK_VALIDATE(size >= kZipEndOfCentralDirSize,
                who + ": " + path + " is too small to be a ZIP archive (a .sog is a ZIP)");

    // The end record sits at the very end unless the archive has a comment,
    // which can push it back by up to 64 KiB.
    int64_t eocd = -1;
    const uint64_t searchFloor =
        size > kZipEndOfCentralDirSize + kZipMaxCommentSize ? size - kZipEndOfCentralDirSize - kZipMaxCommentSize : 0;
    for (uint64_t at = size - kZipEndOfCentralDirSize + 1; at-- > searchFloor;)
    {
        if (readAt<uint32_t>(data, at) == kZipEndOfCentralDirSig)
        {
            eocd = static_cast<int64_t>(at);
            break;
        }
    }
    VK_VALIDATE(eocd >= 0, who + ": " + path
                               + " is not a ZIP archive (no end-of-central-directory record); a "
                                 ".sog bundle is a ZIP of meta.json + WebP images");

    const uint16_t thisDisk = readAt<uint16_t>(data, eocd + 4);
    const uint16_t entryCount = readAt<uint16_t>(data, eocd + 10);
    const uint32_t centralDirSize = readAt<uint32_t>(data, eocd + 12);
    const uint32_t centralDirOffset = readAt<uint32_t>(data, eocd + 16);
    VK_VALIDATE(thisDisk == 0, who + ": multi-disk ZIP archives are not supported: " + path);
    VK_VALIDATE(entryCount != 0xFFFF && centralDirSize != 0xFFFFFFFFu && centralDirOffset != 0xFFFFFFFFu,
                who + ": zip64 archives are not supported: " + path);
    VK_VALIDATE(static_cast<uint64_t>(centralDirOffset) + centralDirSize <= static_cast<uint64_t>(eocd),
                who + ": ZIP central directory lies outside the archive: " + path);

    std::map<std::string, ZipEntry> entries;
    uint64_t at = centralDirOffset;
    const uint64_t centralDirEnd = static_cast<uint64_t>(centralDirOffset) + centralDirSize;
    for (uint16_t e = 0; e < entryCount; ++e)
    {
        VK_VALIDATE(at + kZipCentralDirHeaderSize <= centralDirEnd
                        && readAt<uint32_t>(data, at) == kZipCentralDirSig,
                    who + ": corrupt ZIP central directory (entry " + std::to_string(e) + "): " + path);
        ZipEntry entry;
        entry.flags = readAt<uint16_t>(data, at + 8);
        entry.method = readAt<uint16_t>(data, at + 10);
        entry.crc = readAt<uint32_t>(data, at + 16);
        entry.compressedSize = readAt<uint32_t>(data, at + 20);
        entry.uncompressedSize = readAt<uint32_t>(data, at + 24);
        const uint16_t nameLength = readAt<uint16_t>(data, at + 28);
        const uint16_t extraLength = readAt<uint16_t>(data, at + 30);
        const uint16_t commentLength = readAt<uint16_t>(data, at + 32);
        const uint32_t localHeader = readAt<uint32_t>(data, at + 42);
        VK_VALIDATE(at + kZipCentralDirHeaderSize + nameLength <= centralDirEnd,
                    who + ": corrupt ZIP central directory (entry name): " + path);
        const std::string name(reinterpret_cast<const char *>(data + at + kZipCentralDirHeaderSize),
                               nameLength);

        VK_VALIDATE(static_cast<uint64_t>(localHeader) + kZipLocalHeaderSize <= size
                        && readAt<uint32_t>(data, localHeader) == kZipLocalHeaderSig,
                    who + ": ZIP entry '" + name + "' has no valid local header: " + path);
        entry.dataOffset = static_cast<uint64_t>(localHeader) + kZipLocalHeaderSize
                           + readAt<uint16_t>(data, localHeader + 26) + readAt<uint16_t>(data, localHeader + 28);
        VK_VALIDATE(entry.dataOffset + entry.compressedSize <= size,
                    who + ": ZIP entry '" + name + "' runs past the end of the archive (truncated?): " + path);

        entries[name] = entry;
        at += kZipCentralDirHeaderSize + nameLength + extraLength + commentLength;
    }
    return entries;
}

Bytes extractZipEntry(const Bytes & archive, const ZipEntry & entry, const std::string & name,
                      const std::string & path, const std::string & who)
{
    const std::string where = who + ": ZIP entry '" + name + "' in " + path;
    VK_VALIDATE((entry.flags & kZipFlagEncrypted) == 0, where + " is encrypted");
    const uint8_t * source = archive.data() + entry.dataOffset;

    Bytes out;
    if (entry.method == kZipMethodStored)
    {
        VK_VALIDATE(entry.compressedSize == entry.uncompressedSize,
                    where + " is STORED but its sizes disagree");
        out.assign(source, source + entry.compressedSize);
    }
    else if (entry.method == kZipMethodDeflate)
    {
        out.resize(entry.uncompressedSize);
        z_stream stream{};
        // Negative window bits: a raw DEFLATE stream, which is what ZIP
        // stores (no zlib header/trailer).
        VK_VALIDATE(inflateInit2(&stream, -MAX_WBITS) == Z_OK, where + ": inflateInit2 failed");
        stream.next_in = const_cast<Bytef *>(source);
        stream.avail_in = entry.compressedSize;
        // A zero-length entry still has a (2-byte) DEFLATE stream, but zlib wants
        // one byte of output room to reach its end: give it a scratch byte, and
        // the total_out check below still rejects any output it claims.
        uint8_t scratch = 0;
        stream.next_out = out.empty() ? &scratch : out.data();
        stream.avail_out = out.empty() ? 1 : entry.uncompressedSize;
        const int status = inflate(&stream, Z_FINISH);
        const uLong produced = stream.total_out;
        inflateEnd(&stream);
        VK_VALIDATE(status == Z_STREAM_END && produced == entry.uncompressedSize,
                    where + " failed to inflate (corrupt DEFLATE stream)");
    }
    else
    {
        VK_VALIDATE(false, where + " uses ZIP compression method " + std::to_string(entry.method)
                               + "; only STORED (0) and DEFLATE (8) are supported");
    }

    const uLong crc = crc32(crc32(0L, Z_NULL, 0), out.data(), static_cast<uInt>(out.size()));
    VK_VALIDATE(crc == entry.crc, where + " fails its CRC-32 check (corrupt archive)");
    return out;
}

void writeStoredZip(const std::string & path, const FileMap & files)
{
    // Check every limit before the file is touched, so a too-large map never
    // leaves a partial archive behind.
    VK_VALIDATE(files.size() <= kZipMaxEntries,
                "writeStoredZip: " + std::to_string(files.size()) + " entries for " + path
                    + "; a ZIP without zip64 holds at most " + std::to_string(kZipMaxEntries));
    uint64_t total = 0;
    uint64_t centralDirSize = 0;
    for (const auto & [name, data] : files)
    {
        VK_VALIDATE(name.size() <= 0xFFFF,
                    "writeStoredZip: entry name is " + std::to_string(name.size())
                        + " bytes; the ZIP limit is 65535");
        VK_VALIDATE(data.size() <= kZipMaxField32,
                    "writeStoredZip: entry '" + name + "' is " + std::to_string(data.size())
                        + " bytes; a ZIP without zip64 entry is at most "
                        + std::to_string(kZipMaxField32) + " bytes");
        total += kZipLocalHeaderSize + name.size() + data.size();
        centralDirSize += kZipCentralDirHeaderSize + name.size();
    }
    const uint64_t archiveSize = total + centralDirSize + kZipEndOfCentralDirSize;
    VK_VALIDATE(archiveSize <= kZipMaxField32,
                "writeStoredZip: archive for " + path + " would be " + std::to_string(archiveSize)
                    + " bytes; a ZIP without zip64 is at most " + std::to_string(kZipMaxField32)
                    + " bytes");

    const std::filesystem::path target = std::filesystem::u8path(path);
    std::ofstream file(target, std::ios::binary | std::ios::trunc);
    VK_VALIDATE(file.is_open(), "writeStoredZip: failed to create " + path);
    try
    {
        std::vector<uint32_t> offsets;
        offsets.reserve(files.size());
        uint64_t offset = 0;
        Bytes header;
        for (const auto & [name, data] : files)
        {
            const uint32_t crc = static_cast<uint32_t>(
                crc32(crc32(0L, Z_NULL, 0), data.data(), static_cast<uInt>(data.size())));
            const uint32_t size = static_cast<uint32_t>(data.size());
            offsets.push_back(static_cast<uint32_t>(offset));

            header.clear();
            putU32(header, kZipLocalHeaderSig);
            putU16(header, kZipVersionNeeded);
            putU16(header, kZipFlagUtf8Names);
            putU16(header, kZipMethodStored);
            putU16(header, kDosTime);
            putU16(header, kDosDate);
            putU32(header, crc);
            putU32(header, size);
            putU32(header, size);
            putU16(header, static_cast<uint16_t>(name.size()));
            putU16(header, 0); // no extra field
            header.insert(header.end(), name.begin(), name.end());
            file.write(reinterpret_cast<const char *>(header.data()),
                       static_cast<std::streamsize>(header.size()));
            file.write(reinterpret_cast<const char *>(data.data()),
                       static_cast<std::streamsize>(data.size()));
            offset += header.size() + data.size();
        }

        const uint32_t centralDirOffset = static_cast<uint32_t>(offset);
        uint32_t index = 0;
        for (const auto & [name, data] : files)
        {
            const uint32_t crc = static_cast<uint32_t>(
                crc32(crc32(0L, Z_NULL, 0), data.data(), static_cast<uInt>(data.size())));
            const uint32_t size = static_cast<uint32_t>(data.size());

            header.clear();
            putU32(header, kZipCentralDirSig);
            putU16(header, kZipVersionMadeBy);
            putU16(header, kZipVersionNeeded);
            putU16(header, kZipFlagUtf8Names);
            putU16(header, kZipMethodStored);
            putU16(header, kDosTime);
            putU16(header, kDosDate);
            putU32(header, crc);
            putU32(header, size);
            putU32(header, size);
            putU16(header, static_cast<uint16_t>(name.size()));
            putU16(header, 0); // extra field
            putU16(header, 0); // comment
            putU16(header, 0); // disk number start
            putU16(header, 0); // internal attributes
            putU32(header, 0); // external attributes
            putU32(header, offsets[index++]);
            header.insert(header.end(), name.begin(), name.end());
            file.write(reinterpret_cast<const char *>(header.data()),
                       static_cast<std::streamsize>(header.size()));
        }

        header.clear();
        putU32(header, kZipEndOfCentralDirSig);
        putU16(header, 0); // this disk
        putU16(header, 0); // disk with the central directory
        putU16(header, static_cast<uint16_t>(files.size()));
        putU16(header, static_cast<uint16_t>(files.size()));
        putU32(header, static_cast<uint32_t>(centralDirSize));
        putU32(header, centralDirOffset);
        putU16(header, 0); // comment length
        file.write(reinterpret_cast<const char *>(header.data()),
                   static_cast<std::streamsize>(header.size()));
        file.close();
        VK_VALIDATE(!file.fail(), "writeStoredZip: write failed for " + path);
    }
    catch (...)
    {
        file.close();
        std::error_code ignored;
        std::filesystem::remove(target, ignored);
        throw;
    }
}

FileMap readZip(const std::string & path, const std::string & who)
{
    const Bytes archive = readArchiveFile(path, who);
    FileMap files;
    for (const auto & [name, entry] : parseZip(archive, path, who))
    {
        files[name] = extractZipEntry(archive, entry, name, path, who);
    }
    return files;
}

} // namespace sabercat::zip
