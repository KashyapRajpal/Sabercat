# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

<#
.SYNOPSIS
Renders the Sabercat feature reel and encodes it to an MP4 and a GIF.

.DESCRIPTION
Runs sabercat_demo_reel.exe on the scene (headless; it builds nothing), which
writes frame_%05d.ppm, segments.txt and stats.txt into -OutDir. Then ffmpeg
produces:
  sabercat-demo.mp4  H.264 (libx264, yuv420p, crf 20, 30 fps): a 2 s title
                     card, the reel with each segment's caption burned in
                     (lower left), and a 2 s end card.
  sabercat-demo.gif  <= 5 MB, 640 px wide when it fits, a 10-15 s highlight
                     cut (orbit, relighting, ray-traced shadows).

Nothing is built here. Build the Release target first:
  cmake --build build --config Release --target sabercat_demo_reel

.PARAMETER Scene
Path to a local scene in any supported format. No scene ships with Sabercat;
the published reel was rendered from a CC0 cactus capture by steam_studio
(https://note.com/steam_studio/n/ne9736d94f162), kept outside the repository.

.PARAMETER OutDir
Output directory, relative to the repository root unless absolute. Default out/demo.

.PARAMETER Exe
The sabercat_demo_reel.exe to run. Default build/examples/demo_reel/Release/sabercat_demo_reel.exe.

.PARAMETER Ffmpeg
ffmpeg executable or name. Default 'ffmpeg' on PATH, else the WinGet link
under %LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe.

.PARAMETER SkipRender
Encode the frames already in -OutDir without running the renderer.

.EXAMPLE
powershell -ExecutionPolicy Bypass -File tools/make_demo.ps1 -Scene path\to\scene.glb
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$Scene,
    [string]$OutDir = 'out/demo',
    [string]$Exe,
    [string]$Ffmpeg = 'ffmpeg',
    [int]$Width = 1280,
    [int]$Height = 720,
    [int]$Fps = 30,
    [switch]$SkipRender
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot

function Get-FullPath([string]$Path, [string]$Base) {
    if ([IO.Path]::IsPathRooted($Path)) { return [IO.Path]::GetFullPath($Path) }
    return [IO.Path]::GetFullPath((Join-Path $Base $Path))
}

function Find-Ffmpeg([string]$Name) {
    if (Test-Path -LiteralPath $Name -PathType Leaf) { return (Resolve-Path -LiteralPath $Name).Path }
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $winget = Join-Path $env:LOCALAPPDATA 'Microsoft\WinGet\Links\ffmpeg.exe'
    if (Test-Path -LiteralPath $winget -PathType Leaf) { return $winget }
    throw "ffmpeg not found ('$Name' is not on PATH and $winget does not exist). Install ffmpeg or pass -Ffmpeg <path>."
}

# Runs ffmpeg with the given arguments and throws on failure. Complex filter
# graphs are passed as files with -/filter_complex (ffmpeg 8+; the older
# -filter_complex_script option was removed).
function Invoke-Ffmpeg([string[]]$Arguments) {
    & $script:ffmpegExe -hide_banner -loglevel error -y @Arguments
    if ($LASTEXITCODE -ne 0) { throw "ffmpeg failed (exit $LASTEXITCODE): $($Arguments -join ' ')" }
}

$sceneFull = Get-FullPath $Scene (Get-Location).Path
if (-not (Test-Path -LiteralPath $sceneFull -PathType Leaf)) {
    throw "Scene not found: $sceneFull"
}
$outFull = Get-FullPath $OutDir $repoRoot

if (-not $Exe) {
    $Exe = Join-Path $repoRoot 'build/examples/demo_reel/Release/sabercat_demo_reel.exe'
}
$exeFull = Get-FullPath $Exe (Get-Location).Path

$script:ffmpegExe = Find-Ffmpeg $Ffmpeg
Write-Host "ffmpeg: $script:ffmpegExe"

$fontPath = 'C:/Windows/Fonts/segoeui.ttf'
if (-not (Test-Path -LiteralPath $fontPath -PathType Leaf)) {
    throw "Caption font not found: $fontPath"
}
# Inside a filtergraph the drive colon must be escaped, and the value quoted so
# the graph parser leaves the escape for drawtext's option parser.
$fontArg = "fontfile='C\:/Windows/Fonts/segoeui.ttf'"

if (-not $SkipRender) {
    if (-not (Test-Path -LiteralPath $exeFull -PathType Leaf)) {
        throw "Demo executable not found: $exeFull (build the sabercat_demo_reel Release target, or pass -Exe)"
    }
    New-Item -ItemType Directory -Force -Path $outFull | Out-Null
    # Stale frames would extend the sequence: clear them first.
    Get-ChildItem -LiteralPath $outFull -Filter 'frame_*.ppm' -ErrorAction SilentlyContinue | Remove-Item -Force
    Write-Host "Rendering $sceneFull -> $outFull"
    & $exeFull $sceneFull $outFull --width $Width --height $Height --fps $Fps
    if ($LASTEXITCODE -ne 0) { throw "sabercat_demo_reel failed (exit $LASTEXITCODE)" }
}

$frameCount = @(Get-ChildItem -LiteralPath $outFull -Filter 'frame_*.ppm').Count
if ($frameCount -eq 0) { throw "No frame_*.ppm files in $outFull" }
$segFile = Join-Path $outFull 'segments.txt'
if (-not (Test-Path -LiteralPath $segFile -PathType Leaf)) { throw "segments.txt missing in $outFull" }

$segments = @()
foreach ($line in Get-Content -LiteralPath $segFile) {
    if ($line -match '^(\d+)\t(\d+)\t(.+)$') {
        $segments += [pscustomobject]@{ First = [int]$matches[1]; Count = [int]$matches[2]; Caption = $matches[3] }
    }
}
if ($segments.Count -eq 0) { throw "No segments in $segFile" }

$cmake = Get-Content -LiteralPath (Join-Path $repoRoot 'CMakeLists.txt') -Raw
if ($cmake -notmatch 'project\(Sabercat VERSION ([0-9][0-9.]*)') { throw 'Cannot read the version from CMakeLists.txt' }
$version = $matches[1]

# Text goes through textfile= (UTF-8 files next to the frames, read with the
# frames as the working directory), so no caption needs filtergraph escaping.
$utf8 = New-Object System.Text.UTF8Encoding($false)
function Write-TextFile([string]$Name, [string]$Text) {
    [IO.File]::WriteAllText((Join-Path $outFull $Name), $Text, $utf8)
    return $Name
}

Push-Location $outFull
try {
    # ---- MP4: title card, captioned reel, end card ----
    $graph = New-Object System.Collections.Generic.List[string]
    $last = '[0:v]'
    for ($i = 0; $i -lt $segments.Count; $i++) {
        $s = $segments[$i]
        $textFile = Write-TextFile ('caption_{0:D2}.txt' -f $i) $s.Caption
        $end = $s.First + $s.Count - 1
        $label = "[c$i]"
        $graph.Add("$last" + "drawtext=$fontArg" + ":textfile=$textFile" +
            ":enable='between(n,$($s.First),$end)'" +
            ":x=48:y=h-th-48:fontsize=34:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=14$label")
        $last = $label
    }
    $titleFile = Write-TextFile 'title.txt' "Sabercat $version"
    $endFile = Write-TextFile 'end.txt' 'github.com/KashyapRajpal/Sabercat - Apache-2.0 - scene: CC0'
    $graph.Add("[1:v]drawtext=$fontArg" + ":textfile=$titleFile" +
        ':x=(w-tw)/2:y=(h-th)/2:fontsize=72:fontcolor=white[title]')
    $graph.Add("[2:v]drawtext=$fontArg" + ":textfile=$endFile" +
        ':x=(w-tw)/2:y=(h-th)/2:fontsize=40:fontcolor=white[end]')
    $graph.Add("[title]${last}[end]concat=n=3:v=1:a=0,format=yuv420p[out]")
    $graphFile = Join-Path $outFull 'demo_mp4.filter.txt'
    [IO.File]::WriteAllText($graphFile, ($graph -join ";`n"), $utf8)

    $mp4 = Join-Path $outFull 'sabercat-demo.mp4'
    Invoke-Ffmpeg @(
        '-framerate', "$Fps", '-i', 'frame_%05d.ppm',
        '-f', 'lavfi', '-t', '2', '-i', "color=c=0x101418:s=${Width}x${Height}:r=$Fps",
        '-f', 'lavfi', '-t', '2', '-i', "color=c=0x101418:s=${Width}x${Height}:r=$Fps",
        '-/filter_complex', $graphFile,
        '-map', '[out]', '-c:v', 'libx264', '-preset', 'medium', '-crf', '20',
        '-pix_fmt', 'yuv420p', '-r', "$Fps", '-movflags', '+faststart',
        $mp4)
    Write-Host "MP4: $mp4 ($([math]::Round((Get-Item $mp4).Length / 1MB, 2)) MB)"

    # ---- GIF: highlight cut (orbit 3 s, relighting 6 s, ray-traced shadows 3 s) ----
    $ranges = New-Object System.Collections.Generic.List[string]
    $pick = {
        param([string[]]$Names, [int]$Frames)
        foreach ($n in $Names) {
            $seg = $segments | Where-Object { $_.Caption -eq $n } | Select-Object -First 1
            if ($seg) { return [pscustomobject]@{ First = $seg.First; Count = [math]::Min($seg.Count, $Frames) } }
        }
        return $null
    }
    $parts = @(
        (& $pick @('Gaussian splats, real time') 90),
        (& $pick @('Relighting') 180),
        (& $pick @('Ray-traced shadows', 'Soft shadows (PCSS)') 90)
    ) | Where-Object { $_ -ne $null }
    if ($parts.Count -eq 0) { throw 'No highlight segments available for the GIF' }
    foreach ($p in $parts) {
        $ranges.Add("between(n,$($p.First),$($p.First + $p.Count - 1))")
    }
    $select = ($ranges -join '+')

    $gif = Join-Path $outFull 'sabercat-demo.gif'
    $limit = 5MB
    $candidates = @(@(12, 640), @(10, 640), @(12, 560), @(10, 480), @(8, 480), @(8, 400))
    $done = $false
    foreach ($c in $candidates) {
        $gfps = $c[0]; $gw = $c[1]
        $gifGraph = "select='$select',setpts=N/FRAME_RATE/TB,fps=$gfps,scale=${gw}:-2:flags=lanczos,split[a][b];" +
            '[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle[out]'
        $gifFile = Join-Path $outFull 'demo_gif.filter.txt'
        [IO.File]::WriteAllText($gifFile, $gifGraph, $utf8)
        Invoke-Ffmpeg @('-framerate', "$Fps", '-i', 'frame_%05d.ppm', '-/filter_complex', $gifFile,
            '-map', '[out]', '-loop', '0', $gif)
        $size = (Get-Item $gif).Length
        Write-Host ("GIF candidate {0} fps, {1} px wide: {2:N2} MB" -f $gfps, $gw, ($size / 1MB))
        if ($size -le $limit) { $done = $true; break }
    }
    if (-not $done) { Write-Warning 'GIF is still over 5 MB at the smallest candidate' }
    Write-Host "GIF: $gif ($([math]::Round((Get-Item $gif).Length / 1MB, 2)) MB)"
}
finally {
    Pop-Location
}
