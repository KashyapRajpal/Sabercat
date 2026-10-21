# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

# Regenerates the Sabercat logo assets from the 1254 px master image:
#   docs/media/sabercat-logo.png                 512 px downscale (README, docs)
#   apps/common/resources/sabercat.ico           Windows icon: 16, 24, 32, 48,
#                                                64, 128 and 256 px entries
#   apps/common/resources/sabercat_logo_rgba.h   128 px RGBA pixels for the
#                                                Help > About logo (generated C++)
#   docs/media/social-preview.png                1280x640 GitHub social card
#
# Usage (from the repository root):
#   powershell -NoProfile -ExecutionPolicy Bypass -File tools\make_logo_assets.ps1
#
# Windows only: it uses System.Drawing (GDI+). The master is read as-is; copy a
# new master over docs/media/sabercat-logo-1254.png and rerun to refresh all four.

param(
    [string]$Master = (Join-Path $PSScriptRoot '..\docs\media\sabercat-logo-1254.png')
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing

$root = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$mediaDir = Join-Path $root 'docs\media'
$resourceDir = Join-Path $root 'apps\common\resources'
New-Item -ItemType Directory -Force -Path $mediaDir, $resourceDir | Out-Null
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)

# A new 32 bpp ARGB bitmap, with the logo drawn into it at the given size. The
# high-quality bicubic filter is the best GDI+ offers; the caller chains 2:1
# steps for large reductions, which keeps thin details from aliasing.
function Resize-Bitmap([System.Drawing.Image]$source, [int]$size)
{
    $target = New-Object System.Drawing.Bitmap($size, $size, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $graphics = [System.Drawing.Graphics]::FromImage($target)
    $graphics.Clear([System.Drawing.Color]::FromArgb(0, 0, 0, 0))
    $graphics.CompositingMode = [System.Drawing.Drawing2D.CompositingMode]::SourceOver
    $graphics.CompositingQuality = [System.Drawing.Drawing2D.CompositingQuality]::HighQuality
    $graphics.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $graphics.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $graphics.DrawImage($source, 0, 0, $size, $size)
    $graphics.Dispose()
    return $target
}

function Get-PngBytes([System.Drawing.Image]$image)
{
    $stream = New-Object System.IO.MemoryStream
    $image.Save($stream, [System.Drawing.Imaging.ImageFormat]::Png)
    return , $stream.ToArray()
}

$masterImage = [System.Drawing.Bitmap]::FromFile((Resolve-Path $Master).Path)
try
{
    # -- 512 px downscale ----------------------------------------------------
    $logo512 = Resize-Bitmap $masterImage 512
    $logo512.Save((Join-Path $mediaDir 'sabercat-logo.png'), [System.Drawing.Imaging.ImageFormat]::Png)

    # -- Icon chain: each size is a 2:1 (or 3:4) step from the one above -----
    $logo256 = Resize-Bitmap $logo512 256
    $logo128 = Resize-Bitmap $logo256 128
    $logo64 = Resize-Bitmap $logo128 64
    $logo48 = Resize-Bitmap $logo64 48
    $logo32 = Resize-Bitmap $logo48 32
    $logo24 = Resize-Bitmap $logo32 24
    $logo16 = Resize-Bitmap $logo24 16
    $iconSizes = @(16, 24, 32, 48, 64, 128, 256)
    $iconBitmaps = @($logo16, $logo24, $logo32, $logo48, $logo64, $logo128, $logo256)

    # -- Multi-size .ico: every entry is PNG-compressed (Windows Vista and later
    # read these; the 256 px entry must be PNG). ICONDIR, one ICONDIRENTRY per
    # size, then the image data in the same order.
    $pngs = @()
    foreach ($bitmap in $iconBitmaps)
    {
        $pngs += , (Get-PngBytes $bitmap)
    }
    $ms = New-Object System.IO.MemoryStream
    $writer = New-Object System.IO.BinaryWriter($ms)
    $writer.Write([uint16]0)
    $writer.Write([uint16]1)
    $writer.Write([uint16]$iconSizes.Count)
    $offset = 6 + 16 * $iconSizes.Count
    $index = 0
    foreach ($size in $iconSizes)
    {
        $dimension = if ($size -ge 256) { 0 } else { $size }
        $writer.Write([byte]$dimension)
        $writer.Write([byte]$dimension)
        $writer.Write([byte]0)
        $writer.Write([byte]0)
        $writer.Write([uint16]1)
        $writer.Write([uint16]32)
        $writer.Write([uint32]$pngs[$index].Length)
        $writer.Write([uint32]$offset)
        $offset += $pngs[$index].Length
        $index++
    }
    foreach ($png in $pngs)
    {
        $writer.Write([byte[]]$png)
    }
    $writer.Flush()
    [System.IO.File]::WriteAllBytes((Join-Path $resourceDir 'sabercat.ico'), $ms.ToArray())
    $writer.Dispose()

    # -- 128 px RGBA for the About panel (straight alpha, as ImGui blends) -----
    $pixelBytes = New-Object byte[] (128 * 128 * 4)
    $rect = New-Object System.Drawing.Rectangle(0, 0, 128, 128)
    $data = $logo128.LockBits($rect, [System.Drawing.Imaging.ImageLockMode]::ReadOnly,
                              [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    [System.Runtime.InteropServices.Marshal]::Copy($data.Scan0, $pixelBytes, 0, $pixelBytes.Length)
    $logo128.UnlockBits($data)

    $rgba = New-Object System.Text.StringBuilder
    $perLine = 16
    for ($i = 0; $i -lt $pixelBytes.Length; $i += 4)
    {
        # GDI+ stores 32 bpp ARGB as B, G, R, A in memory; the header wants R, G, B, A.
        [void]$rgba.Append(('0x{0:x2},0x{1:x2},0x{2:x2},0x{3:x2},' -f $pixelBytes[$i + 2], $pixelBytes[$i + 1], $pixelBytes[$i], $pixelBytes[$i + 3]))
        if ((($i / 4) + 1) % $perLine -eq 0)
        {
            [void]$rgba.Append("`n    ")
        }
        else
        {
            [void]$rgba.Append(' ')
        }
    }
    $header = @(
        '// Copyright 2026 Kashyap Rajpal',
        '// SPDX-License-Identifier: Apache-2.0',
        '',
        '// GENERATED by tools/make_logo_assets.ps1 from docs/media/sabercat-logo-1254.png.',
        '// Do not edit by hand: rerun the script after changing the master image.',
        '//',
        '// The Sabercat logo as 128 x 128 RGBA8 pixels, rows top to bottom, straight',
        '// (not premultiplied) alpha. Help > About uploads it as a Vulkan texture.',
        '#pragma once',
        '',
        '#include <cstddef>',
        '',
        'namespace sabercat::app',
        '{',
        '',
        'inline constexpr unsigned kSabercatLogoSize = 128;',
        'inline constexpr unsigned char kSabercatLogoRgba[] = {',
        '    ' + $rgba.ToString().TrimEnd(),
        '};',
        '',
        '} // namespace sabercat::app',
        ''
    ) -join "`n"
    [System.IO.File]::WriteAllText((Join-Path $resourceDir 'sabercat_logo_rgba.h'), $header, $utf8NoBom)

    # -- GitHub social preview: 1280x640, the logo left, the name and tagline right.
    # GitHub crops the edges of the card, so everything sits inside x 120..1160 and
    # y 80..560.
    $card = New-Object System.Drawing.Bitmap(1280, 640, [System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
    $g = [System.Drawing.Graphics]::FromImage($card)
    $g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $background = New-Object System.Drawing.Drawing2D.LinearGradientBrush(
        (New-Object System.Drawing.Rectangle(0, 0, 1280, 640)),
        [System.Drawing.Color]::FromArgb(255, 11, 18, 32),
        [System.Drawing.Color]::FromArgb(255, 19, 30, 52),
        20.0)
    $g.FillRectangle($background, 0, 0, 1280, 640)
    $background.Dispose()
    $g.DrawImage($logo512, 120, 80, 480, 480)

    $textX = 648.0
    $textWidth = 512.0
    $titleText = 'Sabercat'
    $taglineText = 'Real-time Gaussian splat engine on Vulkan'
    $titleBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 245, 247, 250))
    $taglineBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(255, 159, 179, 200))

    # Largest size that fits the text column, so the layout holds if the name changes.
    function Get-FittingFont([string]$text, [float]$startSize, [System.Drawing.FontStyle]$style, [float]$maxWidth)
    {
        $size = $startSize
        while ($true)
        {
            $font = New-Object System.Drawing.Font('Segoe UI', $size, $style, [System.Drawing.GraphicsUnit]::Pixel)
            $measured = $g.MeasureString($text, $font, 4096)
            if ($measured.Width -le $maxWidth -or $size -le 12)
            {
                return $font
            }
            $font.Dispose()
            $size -= 2
        }
    }
    $titleFont = Get-FittingFont $titleText 124 ([System.Drawing.FontStyle]::Bold) $textWidth
    $taglineFont = Get-FittingFont $taglineText 34 ([System.Drawing.FontStyle]::Regular) $textWidth
    $titleSize = $g.MeasureString($titleText, $titleFont, 4096)
    $taglineSize = $g.MeasureString($taglineText, $taglineFont, 4096)
    $gap = 22.0
    $blockTop = (640 - ($titleSize.Height + $gap + $taglineSize.Height)) / 2
    $g.DrawString($titleText, $titleFont, $titleBrush, [float]$textX, [float]$blockTop)
    $g.DrawString($taglineText, $taglineFont, $taglineBrush, [float]$textX,
                  [float]($blockTop + $titleSize.Height + $gap))
    $card.Save((Join-Path $mediaDir 'social-preview.png'), [System.Drawing.Imaging.ImageFormat]::Png)

    $titleFont.Dispose()
    $taglineFont.Dispose()
    $titleBrush.Dispose()
    $taglineBrush.Dispose()
    $g.Dispose()
    $card.Dispose()

    foreach ($bitmap in @($logo512, $logo256, $logo128, $logo64, $logo48, $logo32, $logo24, $logo16))
    {
        $bitmap.Dispose()
    }
}
finally
{
    $masterImage.Dispose()
}

Write-Output 'Wrote docs/media/sabercat-logo.png, apps/common/resources/sabercat.ico,'
Write-Output 'apps/common/resources/sabercat_logo_rgba.h and docs/media/social-preview.png.'
