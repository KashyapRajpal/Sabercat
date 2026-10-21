// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "sabercat/file_map.h"

// Internal ZIP support (not installed): the reader SogReader uses for .sog
// bundles, and a STORED-only writer the tests and fixture tools use.
namespace sabercat::zip
{

// A central-directory entry, resolved to where its bytes live.
// Fields grouped by size so the struct is padding-free.
struct ZipEntry
{
    uint64_t dataOffset = 0;
    uint32_t compressedSize = 0;
    uint32_t uncompressedSize = 0;
    uint32_t crc = 0;
    uint16_t method = 0;
    uint16_t flags = 0;
};

// Just enough ZIP to read: a single-disk archive of STORED or DEFLATE
// entries, no zip64, no encryption -- the same envelope the PlayCanvas
// engine's own parseZipArchive accepts. Sizes and CRCs come from the central
// directory, because splat-transform sets the data-descriptor flag and leaves
// the local headers' sizes zero. `who` prefixes every error message.
std::map<std::string, ZipEntry> parseZip(const std::vector<uint8_t> & archive,
                                         const std::string & path, const std::string & who);

// Inflates (or copies) one entry and checks its CRC-32.
std::vector<uint8_t> extractZipEntry(const std::vector<uint8_t> & archive, const ZipEntry & entry,
                                     const std::string & name, const std::string & path,
                                     const std::string & who);

// Writes `files` as a STORED ZIP: local headers, central directory, end record,
// CRC-32 from zlib, a fixed 1980-01-01 00:00 timestamp (so the output is
// deterministic), the UTF-8 name flag, entries in FileMap order. Stream-writes
// the entries; no zip64, so an entry or archive over 0xFFFFFFFF bytes, or more
// than 65535 entries, throws std::runtime_error naming the limit before any
// file is written.
void writeStoredZip(const std::string & path, const FileMap & files);

// Reads every entry of a ZIP (STORED or DEFLATE), CRC-checked. `who` prefixes
// every error message.
FileMap readZip(const std::string & path, const std::string & who);

} // namespace sabercat::zip
