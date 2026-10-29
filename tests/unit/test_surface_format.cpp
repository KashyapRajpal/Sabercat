// Copyright 2026 Kashyap Rajpal
// SPDX-License-Identifier: Apache-2.0

// Swapchain format choice (src/renderer/surface_format.h): the display path
// needs a UNORM format, since present() writes display-referred bytes as-is.
#include "unit_test.h"

#include "surface_format.h"

#include <vector>

namespace sabercat::unit
{

namespace
{

VkSurfaceFormatKHR format(VkFormat f, VkColorSpaceKHR space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
{
    VkSurfaceFormatKHR result{};
    result.format = f;
    result.colorSpace = space;
    return result;
}

bool same(const VkSurfaceFormatKHR & a, const VkSurfaceFormatKHR & b)
{
    return a.format == b.format && a.colorSpace == b.colorSpace;
}

} // namespace

void testSurfaceFormat(Context & ctx)
{
    const VkSurfaceFormatKHR bgraUnorm = format(VK_FORMAT_B8G8R8A8_UNORM);
    const VkSurfaceFormatKHR rgbaUnorm = format(VK_FORMAT_R8G8B8A8_UNORM);
    const VkSurfaceFormatKHR bgraSrgb = format(VK_FORMAT_B8G8R8A8_SRGB);
    const VkSurfaceFormatKHR rgbaSrgb = format(VK_FORMAT_R8G8B8A8_SRGB);

    // Both UNORM formats offered: B8G8R8A8_UNORM wins, whichever comes first.
    UNIT_CHECK(ctx, same(chooseSurfaceFormat({bgraSrgb, bgraUnorm, rgbaUnorm}), bgraUnorm));
    UNIT_CHECK(ctx, same(chooseSurfaceFormat({rgbaUnorm, bgraUnorm, bgraSrgb}), bgraUnorm));
    UNIT_CHECK(ctx, same(chooseSurfaceFormat({bgraUnorm, rgbaUnorm}), bgraUnorm));
    UNIT_CHECK(ctx, same(chooseSurfaceFormat({rgbaUnorm, bgraUnorm}), bgraUnorm));

    // Only R8G8B8A8_UNORM offered (among sRGB formats).
    UNIT_CHECK(ctx, same(chooseSurfaceFormat({bgraSrgb, rgbaUnorm, rgbaSrgb}), rgbaUnorm));
    UNIT_CHECK(ctx, same(chooseSurfaceFormat({rgbaUnorm}), rgbaUnorm));

    // A UNORM format in another color space is not the sRGB-nonlinear choice.
    const VkSurfaceFormatKHR bgraUnormOther =
        format(VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT);
    UNIT_CHECK(ctx, same(chooseSurfaceFormat({bgraSrgb, bgraUnormOther}), bgraSrgb));

    // No UNORM choice: the surface's first format, whatever it is.
    UNIT_CHECK(ctx, same(chooseSurfaceFormat({bgraSrgb, rgbaSrgb}), bgraSrgb));
    UNIT_CHECK(ctx, same(chooseSurfaceFormat({rgbaSrgb, bgraSrgb}), rgbaSrgb));
    UNIT_CHECK(ctx, same(chooseSurfaceFormat({bgraUnormOther, bgraSrgb}), bgraUnormOther));
    UNIT_CHECK(ctx, same(chooseSurfaceFormat({format(VK_FORMAT_A2B10G10R10_UNORM_PACK32)}),
                         format(VK_FORMAT_A2B10G10R10_UNORM_PACK32)));

    // Empty list: undefined, never a read past the end.
    UNIT_CHECK(ctx, chooseSurfaceFormat({}).format == VK_FORMAT_UNDEFINED);

    UNIT_CHECK(ctx, isSrgbSurfaceFormat(VK_FORMAT_B8G8R8A8_SRGB));
    UNIT_CHECK(ctx, isSrgbSurfaceFormat(VK_FORMAT_R8G8B8A8_SRGB));
    UNIT_CHECK(ctx, !isSrgbSurfaceFormat(VK_FORMAT_B8G8R8A8_UNORM));
    UNIT_CHECK(ctx, !isSrgbSurfaceFormat(VK_FORMAT_R8G8B8A8_UNORM));
}

} // namespace sabercat::unit
