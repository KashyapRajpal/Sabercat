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
// zip64 records and the markers that point at them (APPNOTE.TXT section 4.5, 4.3.15).
static constexpr uint32_t kZipEocd64Sig = 0x06064b50;
static constexpr uint32_t kZipEocd64LocatorSig = 0x07064b50;
static constexpr uint64_t kZipEocd64Size = 56;
static constexpr uint64_t kZipEocd64LocatorSize = 20;
static constexpr uint16_t kZipExtraZip64Id = 0x0001;
static constexpr uint16_t kZipMarker16 = 0xFFFF;
static constexpr uint32_t kZipMarker32 = 0xFFFFFFFF;
// zlib takes 32-bit lengths; feed it in pieces of at most this size.
static constexpr uint64_t kZlibChunk = 0x40000000;
// File writes go out in pieces of at most this size, as readArchiveFile reads.
static constexpr uint64_t kWriteChunk = uint64_t{1} << 30;

// Writer constants. Version 1.0 is all a STORED entry needs to extract; version
// 4.5 is the zip64 extensions.
static constexpr uint16_t kZipVersionNeeded = 10;
static constexpr uint16_t kZipVersionZip64 = 45;
static constexpr uint16_t kZipVersionMadeBy = 20;
static constexpr uint16_t kZipFlagUtf8Names = 0x0800;
// DOS date for 1980-01-01 (day 1, month 1, year 0 from 1980), time 00:00:00.
// Fixed so two writes of the same FileMap are byte-identical.
static constexpr uint16_t kDosTime = 0;
static constexpr uint16_t kDosDate = (1 << 5) | 1;

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

void putU64(Bytes & out, uint64_t value)
{
    putU32(out, static_cast<uint32_t>(value & 0xFFFFFFFF));
    putU32(out, static_cast<uint32_t>(value >> 32));
}

Bytes readArchiveFile(const std::string & path, const std::string & who)
{
    std::ifstream file(std::filesystem::u8path(path), std::ios::binary | std::ios::ate);
    VK_VALIDATE(file.is_open(), who + ": failed to open ZIP " + path);
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    VK_VALIDATE(size > 0, who + ": ZIP is empty: " + path);
    Bytes bytes(static_cast<size_t>(size));
    // Read in steps of at most 1 GiB, so one huge request is never handed to the stream at once.
    const std::streamsize kReadChunk = std::streamsize{1} << 30;
    std::streamsize done = 0;
    while (done < size)
    {
        const std::streamsize want = size - done < kReadChunk ? size - done : kReadChunk;
        file.read(reinterpret_cast<char *>(bytes.data()) + done, want);
        const std::streamsize got = file.gcount();
        VK_VALIDATE(got == want, who + ": truncated read of " + path);
        done += got;
    }
    return bytes;
}

// CRC-32 of a buffer of any size. zlib takes a 32-bit length (uInt, which is
// 32 bits on Windows too), so a larger buffer goes through in pieces.
uint32_t crc32Of(const uint8_t * data, uint64_t size)
{
    uLong crc = crc32(0L, Z_NULL, 0);
    while (size > 0)
    {
        const uInt chunk = static_cast<uInt>(size < kZlibChunk ? size : kZlibChunk);
        crc = crc32(crc, data, chunk);
        data += chunk;
        size -= chunk;
    }
    return static_cast<uint32_t>(crc);
}

// Writes `size` bytes in pieces of at most kWriteChunk, so one huge request is
// never handed to the stream at once.
void writeAll(std::ofstream & file, const uint8_t * data, uint64_t size)
{
    while (size > 0)
    {
        const uint64_t chunk = size < kWriteChunk ? size : kWriteChunk;
        file.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(chunk));
        data += chunk;
        size -= chunk;
    }
}

// Reads the zip64 end record that the locator before the classic end record
// points to, and replaces each classic field that still holds its marker with
// the zip64 value. Returns where the zip64 end record starts, which is where
// the central directory must end (the record sits between the two).
uint64_t applyZip64EndRecord(const uint8_t * data, uint64_t eocd, const std::string & path,
                             const std::string & who, uint64_t & entryCount, uint64_t & centralDirSize,
                             uint64_t & centralDirOffset)
{
    const std::string message = who + ": ZIP has a corrupt zip64 end record: " + path;
    VK_VALIDATE(eocd >= kZipEocd64LocatorSize, message);
    const uint64_t locator = eocd - kZipEocd64LocatorSize;
    VK_VALIDATE(readAt<uint32_t>(data, locator) == kZipEocd64LocatorSig, message);
    const uint64_t recordOffset = readAt<uint64_t>(data, locator + 8);
    VK_VALIDATE(readAt<uint32_t>(data, locator + 16) <= 1, message);
    VK_VALIDATE(recordOffset <= locator && locator - recordOffset >= kZipEocd64Size, message);
    VK_VALIDATE(readAt<uint32_t>(data, recordOffset) == kZipEocd64Sig, message);
    VK_VALIDATE(readAt<uint64_t>(data, recordOffset + 4) >= 44, message);
    VK_VALIDATE(readAt<uint32_t>(data, recordOffset + 16) == 0 && readAt<uint32_t>(data, recordOffset + 20) == 0,
                message);
    if (entryCount == kZipMarker16)
    {
        entryCount = readAt<uint64_t>(data, recordOffset + 32);
    }
    if (centralDirSize == kZipMarker32)
    {
        centralDirSize = readAt<uint64_t>(data, recordOffset + 40);
    }
    if (centralDirOffset == kZipMarker32)
    {
        centralDirOffset = readAt<uint64_t>(data, recordOffset + 48);
    }
    return recordOffset;
}

// Takes the zip64 values an entry's extra field holds for its classic fields
// that are at the 0xFFFFFFFF marker, in the order APPNOTE fixes: uncompressed
// size, compressed size, local header offset. Fields without the marker are
// left alone. The caller only calls this when at least one marker is set.
void applyZip64Extra(const uint8_t * data, uint64_t extraStart, uint16_t extraLength, const std::string & name,
                     const std::string & path, const std::string & who, uint64_t & uncompressedSize,
                     uint64_t & compressedSize, uint64_t & localHeader)
{
    const std::string corrupt = who + ": ZIP entry '" + name + "' has a corrupt extra field: " + path;
    const uint64_t end = extraStart + extraLength;
    bool found = false;
    uint64_t at = extraStart;
    while (at < end)
    {
        VK_VALIDATE(end - at >= 4, corrupt);
        const uint16_t id = readAt<uint16_t>(data, at);
        const uint16_t payloadSize = readAt<uint16_t>(data, at + 2);
        at += 4;
        VK_VALIDATE(end - at >= payloadSize, corrupt);
        if (id == kZipExtraZip64Id && !found)
        {
            found = true;
            uint64_t field = at;
            const uint64_t fieldEnd = at + payloadSize;
            const auto takeU64 = [&](uint64_t & value) {
                VK_VALIDATE(fieldEnd - field >= 8, who + ": ZIP entry '" + name
                                                       + "' zip64 extra field is too short: " + path);
                value = readAt<uint64_t>(data, field);
                field += 8;
            };
            if (uncompressedSize == kZipMarker32)
            {
                takeU64(uncompressedSize);
            }
            if (compressedSize == kZipMarker32)
            {
                takeU64(compressedSize);
            }
            if (localHeader == kZipMarker32)
            {
                takeU64(localHeader);
            }
        }
        at += payloadSize;
    }
    VK_VALIDATE(found, who + ": ZIP entry '" + name + "' has no zip64 extra field: " + path);
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
    const uint64_t eocdAt = static_cast<uint64_t>(eocd);

    const uint64_t thisDisk = readAt<uint16_t>(data, eocdAt + 4);
    uint64_t entryCount = readAt<uint16_t>(data, eocdAt + 10);
    uint64_t centralDirSize = readAt<uint32_t>(data, eocdAt + 12);
    uint64_t centralDirOffset = readAt<uint32_t>(data, eocdAt + 16);
    const bool zip64 = entryCount == kZipMarker16 || centralDirSize == kZipMarker32
                       || centralDirOffset == kZipMarker32 || thisDisk == kZipMarker16;
    // A zip64 archive's central directory ends where the zip64 end record starts,
    // not at the classic end record.
    uint64_t centralDirLimit = eocdAt;
    if (zip64)
    {
        VK_VALIDATE(thisDisk == 0 || thisDisk == kZipMarker16,
                    who + ": multi-disk ZIP archives are not supported: " + path);
        centralDirLimit = applyZip64EndRecord(data, eocdAt, path, who, entryCount, centralDirSize, centralDirOffset);
    }
    else
    {
        VK_VALIDATE(thisDisk == 0, who + ": multi-disk ZIP archives are not supported: " + path);
    }
    VK_VALIDATE(centralDirOffset <= centralDirLimit && centralDirSize <= centralDirLimit - centralDirOffset,
                who + ": ZIP central directory lies outside the archive: " + path);
    // Every entry takes at least a fixed header; more entries than that cannot fit.
    VK_VALIDATE(entryCount <= centralDirSize / kZipCentralDirHeaderSize,
                who + ": corrupt ZIP central directory (" + std::to_string(entryCount) + " entries in "
                    + std::to_string(centralDirSize) + " bytes): " + path);

    std::map<std::string, ZipEntry> entries;
    const uint64_t centralDirEnd = centralDirOffset + centralDirSize;
    uint64_t at = centralDirOffset;
    for (uint64_t e = 0; e < entryCount; ++e)
    {
        VK_VALIDATE(centralDirEnd - at >= kZipCentralDirHeaderSize && readAt<uint32_t>(data, at) == kZipCentralDirSig,
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
        uint64_t localHeader = readAt<uint32_t>(data, at + 42);
        VK_VALIDATE(kZipCentralDirHeaderSize + nameLength + extraLength + commentLength <= centralDirEnd - at,
                    who + ": corrupt ZIP central directory (entry " + std::to_string(e) + "): " + path);
        const std::string name(reinterpret_cast<const char *>(data + at + kZipCentralDirHeaderSize), nameLength);

        if (entry.uncompressedSize == kZipMarker32 || entry.compressedSize == kZipMarker32
            || localHeader == kZipMarker32)
        {
            applyZip64Extra(data, at + kZipCentralDirHeaderSize + nameLength, extraLength, name, path, who,
                            entry.uncompressedSize, entry.compressedSize, localHeader);
        }

        VK_VALIDATE(size >= kZipLocalHeaderSize && localHeader <= size - kZipLocalHeaderSize
                        && readAt<uint32_t>(data, localHeader) == kZipLocalHeaderSig,
                    who + ": ZIP entry '" + name + "' has no valid local header: " + path);
        entry.dataOffset = localHeader + kZipLocalHeaderSize + readAt<uint16_t>(data, localHeader + 26)
                           + readAt<uint16_t>(data, localHeader + 28);
        VK_VALIDATE(entry.dataOffset <= size && entry.compressedSize <= size - entry.dataOffset,
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
    const uint8_t * source = archive.data() + static_cast<size_t>(entry.dataOffset);

    Bytes out;
    if (entry.method == kZipMethodStored)
    {
        VK_VALIDATE(entry.compressedSize == entry.uncompressedSize,
                    where + " is STORED but its sizes disagree");
        out.assign(source, source + static_cast<size_t>(entry.compressedSize));
    }
    else if (entry.method == kZipMethodDeflate)
    {
        // DEFLATE cannot expand more than 1032:1. Checking the claim first keeps
        // a bogus uncompressed size from driving a huge resize().
        constexpr uint64_t kMaxDeflateRatio = 1032;
        VK_VALIDATE(entry.uncompressedSize / kMaxDeflateRatio <= entry.compressedSize + 1,
                    where + " claims an impossible DEFLATE expansion (corrupt archive)");
        out.resize(static_cast<size_t>(entry.uncompressedSize));
        z_stream stream{};
        // Negative window bits: a raw DEFLATE stream, which is what ZIP
        // stores (no zlib header/trailer).
        VK_VALIDATE(inflateInit2(&stream, -MAX_WBITS) == Z_OK, where + ": inflateInit2 failed");
        // A zero-length entry still has a (2-byte) DEFLATE stream, but zlib wants
        // one byte of output room to reach its end: give it a scratch byte. Any
        // byte zlib writes there is counted as produced, so it fails the size check.
        const bool emptyOutput = out.empty();
        uint8_t scratch = 0;
        stream.next_in = const_cast<Bytef *>(source);
        stream.next_out = out.data();
        uint64_t remainingIn = entry.compressedSize;
        uint64_t remainingOut = entry.uncompressedSize;
        uint64_t produced = 0;
        int status = Z_OK;
        // The common case is one call. Larger entries loop in zlib-sized pieces
        // until the stream ends or stops making progress.
        for (;;)
        {
            const uInt availIn = static_cast<uInt>(remainingIn < kZlibChunk ? remainingIn : kZlibChunk);
            const uInt availOut = emptyOutput ? uInt{1}
                                              : static_cast<uInt>(remainingOut < kZlibChunk ? remainingOut : kZlibChunk);
            stream.avail_in = availIn;
            stream.avail_out = availOut;
            if (emptyOutput)
            {
                stream.next_out = &scratch;
            }
            status = inflate(&stream, Z_NO_FLUSH);
            const uInt consumed = availIn - stream.avail_in;
            const uInt written = availOut - stream.avail_out;
            remainingIn -= consumed;
            produced += written;
            if (!emptyOutput)
            {
                remainingOut -= written;
            }
            if (status != Z_OK || (consumed == 0 && written == 0))
            {
                break;
            }
        }
        inflateEnd(&stream);
        VK_VALIDATE(status == Z_STREAM_END && produced == entry.uncompressedSize,
                    where + " failed to inflate (corrupt DEFLATE stream)");
    }
    else
    {
        VK_VALIDATE(false, where + " uses ZIP compression method " + std::to_string(entry.method)
                               + "; only STORED (0) and DEFLATE (8) are supported");
    }

    const uint32_t crc = crc32Of(out.data(), out.size());
    VK_VALIDATE(crc == entry.crc, where + " fails its CRC-32 check (corrupt archive)");
    return out;
}

void writeStoredZip(const std::string & path, const FileMap & files, const WriterLimits & limits)
{
    // The name limit is the one Zip64 cannot lift. Check it before the file is
    // touched, so a bad name never leaves a partial archive behind.
    for (const auto & [name, data] : files)
    {
        VK_VALIDATE(name.size() <= 0xFFFF,
                    "writeStoredZip: entry name is " + std::to_string(name.size())
                        + " bytes; the ZIP limit is 65535");
    }

    const std::filesystem::path target = std::filesystem::u8path(path);
    std::ofstream file(target, std::ios::binary | std::ios::trunc);
    VK_VALIDATE(file.is_open(), "writeStoredZip: failed to create " + path);
    try
    {
        const uint64_t maxField32 = limits.maxField32;
        // Offsets and CRCs are only known while streaming, so both are recorded
        // here: the CRC is computed once per entry and reused in the central directory.
        std::vector<uint64_t> offsets;
        std::vector<uint32_t> crcs;
        offsets.reserve(files.size());
        crcs.reserve(files.size());
        uint64_t offset = 0;
        Bytes header;
        for (const auto & [name, data] : files)
        {
            const uint64_t size = data.size();
            const bool bigSize = size > maxField32;
            const uint32_t crc = crc32Of(data.data(), size);
            offsets.push_back(offset);
            crcs.push_back(crc);

            header.clear();
            putU32(header, kZipLocalHeaderSig);
            putU16(header, bigSize ? kZipVersionZip64 : kZipVersionNeeded);
            putU16(header, kZipFlagUtf8Names);
            putU16(header, kZipMethodStored);
            putU16(header, kDosTime);
            putU16(header, kDosDate);
            putU32(header, crc);
            putU32(header, bigSize ? kZipMarker32 : static_cast<uint32_t>(size));
            putU32(header, bigSize ? kZipMarker32 : static_cast<uint32_t>(size));
            putU16(header, static_cast<uint16_t>(name.size()));
            putU16(header, bigSize ? static_cast<uint16_t>(20) : static_cast<uint16_t>(0)); // 4-byte header + two u64
            header.insert(header.end(), name.begin(), name.end());
            if (bigSize)
            {
                putU16(header, kZipExtraZip64Id);
                putU16(header, 16);
                putU64(header, size); // uncompressed
                putU64(header, size); // compressed (STORED)
            }
            file.write(reinterpret_cast<const char *>(header.data()),
                       static_cast<std::streamsize>(header.size()));
            writeAll(file, data.data(), size);
            offset += header.size() + size;
        }

        const uint64_t centralDirOffset = offset;
        size_t index = 0;
        for (const auto & [name, data] : files)
        {
            const uint64_t size = data.size();
            const uint64_t localOffset = offsets[index];
            const bool bigSize = size > maxField32;
            const bool bigOffset = localOffset > maxField32;
            // One zip64 word per field that moved out of the classic header.
            const uint16_t zip64Words = static_cast<uint16_t>((bigSize ? 2 : 0) + (bigOffset ? 1 : 0));
            const bool zip64 = zip64Words != 0;

            header.clear();
            putU32(header, kZipCentralDirSig);
            putU16(header, kZipVersionMadeBy);
            putU16(header, zip64 ? kZipVersionZip64 : kZipVersionNeeded);
            putU16(header, kZipFlagUtf8Names);
            putU16(header, kZipMethodStored);
            putU16(header, kDosTime);
            putU16(header, kDosDate);
            putU32(header, crcs[index]);
            putU32(header, bigSize ? kZipMarker32 : static_cast<uint32_t>(size));
            putU32(header, bigSize ? kZipMarker32 : static_cast<uint32_t>(size));
            putU16(header, static_cast<uint16_t>(name.size()));
            putU16(header, zip64 ? static_cast<uint16_t>(4 + 8 * zip64Words) : 0); // zip64 extra field
            putU16(header, 0); // comment
            putU16(header, 0); // disk number start
            putU16(header, 0); // internal attributes
            putU32(header, 0); // external attributes
            putU32(header, bigOffset ? kZipMarker32 : static_cast<uint32_t>(localOffset));
            header.insert(header.end(), name.begin(), name.end());
            if (zip64)
            {
                putU16(header, kZipExtraZip64Id);
                putU16(header, static_cast<uint16_t>(8 * zip64Words));
                if (bigSize)
                {
                    putU64(header, size); // uncompressed
                    putU64(header, size); // compressed (STORED)
                }
                if (bigOffset)
                {
                    putU64(header, localOffset);
                }
            }
            file.write(reinterpret_cast<const char *>(header.data()),
                       static_cast<std::streamsize>(header.size()));
            offset += header.size();
            ++index;
        }

        // Everything after the local entries is the central directory, so its
        // size is where the stream stands now.
        const uint64_t centralDirSize = offset - centralDirOffset;
        const uint64_t entryCount = files.size();
        const bool entriesBig = entryCount > limits.maxEntries;
        const bool sizeBig = centralDirSize > maxField32;
        const bool offsetBig = centralDirOffset > maxField32;
        if (entriesBig || sizeBig || offsetBig)
        {
            // Zip64 end record (56 bytes), then its locator (20 bytes), which points
            // back at the record. The classic end record below carries the markers.
            header.clear();
            putU32(header, kZipEocd64Sig);
            putU64(header, 44); // bytes after this field
            putU16(header, kZipVersionZip64); // made by
            putU16(header, kZipVersionZip64); // needed
            putU32(header, 0); // this disk
            putU32(header, 0); // disk with the central directory
            putU64(header, entryCount); // entries on this disk
            putU64(header, entryCount); // entries in total
            putU64(header, centralDirSize);
            putU64(header, centralDirOffset);
            putU32(header, kZipEocd64LocatorSig);
            putU32(header, 0); // disk with the zip64 end record
            putU64(header, offset); // where the zip64 end record starts
            putU32(header, 1); // total disks
            file.write(reinterpret_cast<const char *>(header.data()),
                       static_cast<std::streamsize>(header.size()));
        }

        header.clear();
        putU32(header, kZipEndOfCentralDirSig);
        putU16(header, 0); // this disk
        putU16(header, 0); // disk with the central directory
        putU16(header, entriesBig ? kZipMarker16 : static_cast<uint16_t>(entryCount));
        putU16(header, entriesBig ? kZipMarker16 : static_cast<uint16_t>(entryCount));
        putU32(header, sizeBig ? kZipMarker32 : static_cast<uint32_t>(centralDirSize));
        putU32(header, offsetBig ? kZipMarker32 : static_cast<uint32_t>(centralDirOffset));
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
