// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>

namespace sabercat
{

class Device;
class GaussianCloud;

// Reads PlayCanvas SOG ("Spatially Ordered Gaussians", SuperSplat /
// splat-transform) scenes into a GaussianCloud. Two physical forms, one
// format:
//   - `.sog`       a ZIP archive holding meta.json and the WebP images
//                  (STORED entries as splat-transform writes them; DEFLATE
//                  entries too, as the PlayCanvas engine accepts them)
//   - `meta.json`  the same files unbundled, images beside meta.json
//
// Only meta.json `version: 2` (current) is read. Legacy v1 (no `version`,
// per-channel min/max instead of codebooks) is rejected by name; so is a
// streamed `lod-meta.json`, which is a different, chunked container.
//
// Every image is a LOSSLESS WebP decoded to RGBA8 (libwebp); pixel i (row-
// major) belongs to Gaussian i. What each one holds, and what the reader does
// to reach the canonical contract (formulas from splat-transform's
// writer/reader and the engine's GSplatSogData, which agree):
//   means_l/means_u  16-bit (lo | hi << 8) per axis over [mins, maxs] in
//                    LOG space: n = min + (max - min) * v / 65535, then
//                    x = sign(n) * (exp(|n|) - 1)
//   quats            smallest-three: 3 bytes (b / 255 * 2 - 1) / sqrt(2) and a
//                    tag 252 + i naming the dropped largest component, in
//                    w,x,y,z order -> rebuilt, reordered to xyzw, normalized
//   scales           3 indices into a 256-entry codebook of LOG scales -> exp()
//   sh0              3 indices into a codebook of RAW f_dc (no display
//                    transform), alpha = sigmoid(opacity) * 255 -> b / 255
//   shN (optional)   per-Gaussian 16-bit label into a palette of k-means
//                    centroids; centroid c, coefficient j is texel
//                    ((c % 64) * coeffs + j, c / 64) whose RGB are indices into
//                    a codebook -> coefficient-major RGB, bands 1..3
//
// meta.json `model: "antialiased"` sets GaussianCloud::antialiased(); a
// `2dgs` model (surfels, not 3D Gaussians) is rejected. Coordinates are the
// source .ply's (splat-transform bakes to its PLY frame), so like .ply a
// scene usually needs setFlipUpAxis().
class SogReader
{
  public:
    static std::shared_ptr<GaussianCloud> read(Device & device, const std::string & path);
};

} // namespace sabercat
