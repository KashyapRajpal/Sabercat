// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Host-side capture of one forward frame for the sort study. Header-only and
// inline, so sort_modes_test and the study harness share one copy. Call these
// after pass.run(views, camera) in the default mode (Viewer, to Rasterize).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "sabercat/buffer.h"
#include "sabercat/camera.h"
#include "sabercat/forward_pass.h"

#include "cpu_reference.h"
#include "sort_metrics.h"

namespace sabercat::study
{

// Downloads the projected Gaussians, the default sorted list and the tile ranges.
// When withRayForm is set, it also runs computeRayDepthForm() and downloads the
// 12-float form per Gaussian. Empty downloads are skipped, since a zero-size
// staging copy is not valid.
inline StudyCandidates captureCandidates(ForwardPass & pass, const CanonicalGaussianBufferViews & views,
                                         const CameraUniforms & camera, bool withRayForm)
{
    StudyCandidates c;
    c.width = camera.imageWidth;
    c.height = camera.imageHeight;
    c.tilesX = camera.tilesX;
    c.tilesY = camera.tilesY;
    c.fx = camera.fx;
    c.fy = camera.fy;
    c.cx = camera.cx;
    c.cy = camera.cy;

    const size_t gaussianCount = camera.gaussianCount;
    if (gaussianCount > 0)
    {
        c.projected.resize(gaussianCount);
        pass.projected().download(c.projected.data(), gaussianCount * sizeof(ProjectedGaussian));
    }

    const size_t instanceCount = pass.lastInstanceCount();
    if (instanceCount > 0)
    {
        c.sortedValues.resize(instanceCount);
        pass.sortedValues().download(c.sortedValues.data(), instanceCount * sizeof(uint32_t));
    }

    const size_t tileCount = size_t(camera.tilesX) * camera.tilesY;
    if (tileCount > 0)
    {
        c.tileRanges.resize(tileCount);
        pass.tileRanges().download(c.tileRanges.data(), tileCount * sizeof(TileRange));
    }

    if (withRayForm)
    {
        pass.computeRayDepthForm(views, camera);
        if (gaussianCount > 0)
        {
            c.rayForm.resize(gaussianCount * 12);
            pass.rayDepthForm().download(c.rayForm.data(), c.rayForm.size() * sizeof(float));
        }
    }
    return c;
}

// The default sort of the last run(): sorted values, tile ranges and the
// projected depths. The tile count comes from the tile-range buffer's size,
// which is tilesX * tilesY entries.
inline SortOrderFrame captureOrder(ForwardPass & pass, uint32_t gaussianCount)
{
    SortOrderFrame frame;

    const size_t instanceCount = pass.lastInstanceCount();
    if (instanceCount > 0)
    {
        frame.sortedValues.resize(instanceCount);
        pass.sortedValues().download(frame.sortedValues.data(), instanceCount * sizeof(uint32_t));
    }

    const size_t tileCount = static_cast<size_t>(pass.tileRanges().size() / sizeof(TileRange));
    if (tileCount > 0)
    {
        frame.tileRanges.resize(tileCount);
        pass.tileRanges().download(frame.tileRanges.data(), tileCount * sizeof(TileRange));
    }

    if (gaussianCount > 0)
    {
        std::vector<ProjectedGaussian> projected(gaussianCount);
        pass.projected().download(projected.data(), gaussianCount * sizeof(ProjectedGaussian));
        frame.depth.resize(gaussianCount);
        for (size_t i = 0; i < gaussianCount; ++i)
        {
            frame.depth[i] = projected[i].depth;
        }
    }
    return frame;
}

} // namespace sabercat::study
