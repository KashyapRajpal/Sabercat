# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

<#
.SYNOPSIS
    Build, test and package a Sabercat release. The publish flow is split:
    -PushTag tags and pushes only the tag, CI builds the draft release,
    -VerifyCi checks that draft, and -VerifyCi -Publish publishes it.

.DESCRIPTION
    Release model:
      main    -- development.
      stable  -- only ever FAST-FORWARDED to a main commit this script has
                 built and tested; each promotion is an annotated tag
                 v<version> (the version is project(Sabercat VERSION ...) in
                 CMakeLists.txt, bumped by hand in its own commit).
      Release -- every tag gets a GitHub Release carrying the prebuilt package
                 sabercat-<version>-win64.zip (a dry run on a dirty tree adds
                 -dirty), so consumers never need to build Sabercat.
      -StableDir -- optional: an unpacked copy of the latest released package;
                 point CMAKE_PREFIX_PATH at it for find_package(Sabercat).
                 Empty (the default) skips the refresh.

    Publish flow (three commands, in order):
      1. -PushTag: runs the full local pipeline below (steps 1 to 6), then
         creates the annotated tag v<version> on HEAD and pushes ONLY that tag.
         The tag workflow (.github/workflows/release.yml) builds and attests the
         packages in CI and creates a DRAFT GitHub Release. The branch and stable
         are not pushed by this mode.
      2. -VerifyCi -Tag v<version>: run it once the draft exists (the tag
         workflow has finished). It downloads the draft's assets into a fresh temp folder, checks
         the asset names, SHA256SUMS.txt and the GitHub attestations, compares the
         CI packages with the local zips in dist\ (tools\compare_packages.ps1),
         and runs the packaged self-tests on the CI bits. It changes nothing
         remote.
      3. -VerifyCi -Tag v<version> -Publish: the same checks, then publishes the
         draft, fast-forwards stable (final releases only) and refreshes
         -StableDir.
    The legacy -Publish (alone) still does everything locally, see step 7; the
    tag workflow then stops with "already published", so prefer the flow above.

    Build paths: the repo root is mapped to a free drive letter (subst, scanning
    Z down to M) and everything is configured, built, tested and packaged from
    that mapped drive, so the compiler and linker see only paths on the mapped
    drive and the packages do not embed where the repository lives on the build
    machine (__FILE__ in error messages, PDB paths, FetchContent paths). The
    mapping is removed at the end, also on failure. git and gh still run against
    the real repo. CMake caches absolute paths, so -BuildDir (default
    out/release-mapped) is only ever configured through the mapped drive; a
    cache from another letter or path is discarded.

    Steps (any failure stops the script; nothing is published unless every
    step before publishing succeeded):
      1. Preconditions: clean working tree (so the tested bits ARE the commit),
         and with -Publish or -PushTag: the tag is new and `stable` can
         fast-forward. Then the repo root is mapped to a drive letter (see Build
         paths).
      2. Configure -BuildDir, build Debug and Release.
      3. Run the full test list in both configurations from the repo root.
         Any non-zero exit or any "[validation]" line fails the release.
      4. Install both configurations into one staging prefix; add docs,
         manifest.json and selftest/ (a few Release tests + selftest.ps1).
      5. Zip it into dist/, run the leak gate on it, unpack the zip into a fresh
         directory and run the packaged self-test from there (tests the
         artifact actually shipped).
      6. Apps package for testers: the viewer, Studio and comparer with only the
         runtime DLLs they import (the Visual C++ runtime is copied in from the
         Visual Studio redistributable). The dependency audit fails the release
         on any import that is neither in the package, a Windows system DLL nor
         vulkan-1.dll. Zipped and self-tested from a fresh directory with PATH
         reduced to System32.
         Leak gate (steps 5 and 6): every file in each zip, manifest.json
         included, is scanned (ASCII and UTF-16LE, case-insensitive) for the
         real repo root and the user profile directory, each in backslash,
         forward-slash and doubled-backslash form. Any hit fails the release
         and nothing is published.
      7. Publish. -PushTag stops after tagging and pushing the tag (see the
         flow above). Legacy -Publish, alone: fast-forward `stable`, create tag
         v<version>, push main, stable and the tag, create the GitHub Release
         with both zips, then swap the unpacked package into -StableDir. With
         -VerifyCi -Publish, the verified CI draft is published instead.

    The version is the CMake project version, plus -VersionSuffix when given:
    1.0.0-beta.1 gives tag v1.0.0-beta.1, package names with the same version,
    and a GitHub pre-release.

    CI mode (-CiBuild) is for a GitHub-hosted runner without a GPU. It runs
    steps 2, 4, 5 and 6 as above, but the CPU tests (CTest label cpu) run in
    place of the GPU tests, and the two package self-tests are skipped (both
    need a GPU). It never publishes. .github/workflows/release.yml uses it to
    build the packages for a draft release. The GPU tests run locally during
    -PushTag, on the same commit, before the CI draft is verified; the packaged
    self-tests run again on the CI bits during -VerifyCi.
    -ExpectTag names the tag the run was started for; the script fails unless
    it equals the tag derived from CMakeLists.txt, so a pushed tag cannot
    disagree with the version in the tree.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\release.ps1
    Dry run: build, test and package; publishes nothing.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\release.ps1 -AllowDirty -VersionSuffix beta.1
    Dry run of a beta: the packages are named 1.0.0-beta.1; publishing is refused on a dirty tree.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\release.ps1 -PushTag
    Step 1 of the publish flow: build, test, package locally, then tag v<version>
    and push only the tag. The tag workflow then creates the draft release.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\release.ps1 -VerifyCi -Tag v1.1.0
    Step 2: verify the CI-built draft release for v1.1.0 against the local
    packages. Publishes nothing.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\release.ps1 -VerifyCi -Tag v1.1.0 -Publish
    Step 3: verify the draft again, then publish it (and move stable for a final release).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\release.ps1 -Publish
    Legacy local publish, a fallback only: build, test, tag, push and create the
    release here, without the CI draft. Prefer -PushTag, then -VerifyCi.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\release.ps1 -CiBuild -ExpectTag v1.1.0
    CI build on a GPU-less runner: build, CPU tests, packages; publishes nothing.
#>
param(
    # Legacy local publish (the split flow is -PushTag, then -VerifyCi): promote
    # to stable, tag, push, create the GitHub Release, refresh -StableDir. With
    # -VerifyCi, publishes the verified CI draft for -Tag instead.
    [switch]$Publish,
    # Run the full local pipeline, then create the annotated tag and push ONLY the
    # tag; the tag workflow then builds the packages and creates a draft release.
    [switch]$PushTag,
    # Verify the CI-built draft release for -Tag against the local packages; with
    # -Publish, also publish that draft.
    [switch]$VerifyCi,
    # The tag to verify, v<x.y.z> or v<x.y.z>-<suffix>; only with -VerifyCi.
    [string]$Tag = '',
    # Package an uncommitted tree (dry runs only; the zip is marked -dirty).
    [switch]$AllowDirty,
    # Unpacked package location to refresh on publish; empty skips the refresh.
    [string]$StableDir = '',
    # Dedicated build tree (git-ignored), reused incrementally between releases.
    # Configured only through the mapped drive (see Build paths), hence not out/release.
    [string]$BuildDir = 'out/release-mapped',
    [string]$Remote = 'origin',
    # Semver pre-release suffix, e.g. beta.1: the tag becomes v<x.y.z>-<suffix>,
    # the packages carry it, and the GitHub Release is marked as a pre-release.
    [string]$VersionSuffix = '',
    # Build and package on a GPU-less CI runner: skips the GPU tests and the two
    # package self-tests, which need a GPU; runs the CPU tests instead. The GPU
    # tests run locally before a release. Never publishes.
    [switch]$CiBuild,
    # If given, the tag derived from CMakeLists.txt + -VersionSuffix must equal it,
    # so a pushed tag cannot disagree with the version in the tree.
    [string]$ExpectTag = ''
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# Every gpu-labeled CTest test must pass in Debug and Release with zero
# "[validation]" lines. The list is CTest's own (read from the build tree below),
# so a test is added or removed in tests/integration/CMakeLists.txt only.
# Shipped in the package and run by selftest\selftest.ps1. Each resolves its
# SPIR-V through $SABERCAT_SHADER_DIR (apps/common/shader_dir.h).
$SelftestTests = @('device_test', 'forward_test', 'relight_test', 'c_api_test')

# $RealRepo is the checkout as it is on disk; git and gh use it. $Repo becomes
# the mapped drive root once the mapping exists (below), so every path handed
# to cmake, the compilers and the tests is neutral.
$RealRepo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Repo = $RealRepo
$BuildPath = Join-Path $Repo $BuildDir
$DistPath = Join-Path $Repo 'dist'
$LogPath = Join-Path $BuildPath 'release-logs'

function Step([string]$message) { Write-Host ''; Write-Host "== $message" -ForegroundColor Cyan }

function Fail([string]$message) { throw "release: $message" }

# Quote one argument for a Windows command line.
function Quote([string]$arg)
{
    if ($arg -notmatch '[\s"]') { return $arg }
    return '"' + ($arg -replace '"', '\"') + '"'
}

# Run a native program, stdout+stderr to $log, from $workDir; returns the
# exit code. Start-Process keeps native stderr out of PowerShell's error
# stream (Windows PowerShell 5.1 turns redirected native stderr into errors).
function Run([string]$log, [string]$exe, [string[]]$arguments, [string]$workDir = $Repo)
{
    $startArgs = @{
        FilePath = $exe; WorkingDirectory = $workDir; NoNewWindow = $true; PassThru = $true
        RedirectStandardOutput = $log; RedirectStandardError = "$log.stderr"
    }
    # Start-Process rejects an empty -ArgumentList.
    if ($arguments.Count -gt 0) { $startArgs.ArgumentList = ($arguments | ForEach-Object { Quote $_ }) -join ' ' }
    $p = Start-Process @startArgs
    $null = $p.Handle # PS 5.1: keep the handle so ExitCode survives exit
    $p.WaitForExit()
    return $p.ExitCode
}

function LogText([string]$log)
{
    $text = ''
    foreach ($f in @($log, "$log.stderr"))
    {
        if (Test-Path $f) { $t = Get-Content $f -Raw; if ($null -ne $t) { $text += $t } }
    }
    return $text
}

function RunOrFail([string]$what, [string]$log, [string]$exe, [string[]]$arguments)
{
    $code = Run $log $exe $arguments
    if ($code -ne 0)
    {
        ((LogText $log) -split "`n") | Select-Object -Last 30 | ForEach-Object { Write-Host $_ }
        Fail "$what failed (exit $code); see $log"
    }
}

function Invoke-Git([string[]]$arguments)
{
    $out = & git -C $RealRepo @arguments
    if ($LASTEXITCODE -ne 0) { Fail "git $($arguments -join ' ') failed" }
    return $out
}

function RefExists([string]$ref)
{
    & git -C $RealRepo rev-parse --verify --quiet $ref > $null
    return $LASTEXITCODE -eq 0
}

function WriteUtf8([string]$path, [string]$text)
{
    [IO.File]::WriteAllText($path, $text, (New-Object Text.UTF8Encoding $false))
}

# Zip reading and writing (ZipFile) is used by the leak gate and the self-tests.
Add-Type -AssemblyName System.IO.Compression.FileSystem

# Case-insensitive byte search over zip entries, ASCII and UTF-16LE. C# because
# a PowerShell byte loop over the tens of MB in a package takes minutes.
Add-Type -TypeDefinition @'
public static class PathLeakScan
{
    // True if `needle` (already lowercase) occurs in `data`, folding A-Z in
    // `data` to a-z on the fly.
    public static bool Contains(byte[] data, byte[] needle)
    {
        int n = needle.Length;
        for (int i = 0; i + n <= data.Length; i++)
        {
            int j = 0;
            for (; j < n; j++)
            {
                byte c = data[i + j];
                if (c >= 65 && c <= 90) c = (byte)(c + 32);
                if (c != needle[j]) break;
            }
            if (j == n) return true;
        }
        return false;
    }
}
'@

# Zip entries that contain any of $needles (ASCII or UTF-16LE, case-insensitive),
# as "entry: needle (encoding)" strings.
function Find-PathLeaks([string]$zipPath, [string[]]$needles)
{
    $patterns = @()
    foreach ($needle in ($needles | Select-Object -Unique))
    {
        $lower = $needle.ToLowerInvariant()
        $patterns += , @($needle, 'ASCII', [Text.Encoding]::ASCII.GetBytes($lower))
        $patterns += , @($needle, 'UTF-16LE', [Text.Encoding]::Unicode.GetBytes($lower))
    }
    $hits = New-Object System.Collections.Generic.List[string]
    $archive = [IO.Compression.ZipFile]::OpenRead($zipPath)
    try
    {
        foreach ($entry in $archive.Entries)
        {
            if ($entry.Name -eq '') { continue }
            $stream = $entry.Open()
            try
            {
                $buffer = New-Object byte[] ([int]$entry.Length)
                $read = 0
                while ($read -lt $buffer.Length)
                {
                    $n = $stream.Read($buffer, $read, $buffer.Length - $read)
                    if ($n -le 0) { break }
                    $read += $n
                }
            }
            finally { $stream.Dispose() }
            foreach ($pattern in $patterns)
            {
                if ([PathLeakScan]::Contains($buffer, $pattern[2])) { $hits.Add("$($entry.FullName): $($pattern[0]) ($($pattern[1]))") }
            }
        }
    }
    finally { $archive.Dispose() }
    return $hits.ToArray()
}

# The strings that must not appear in a shipped file: the real repo root and the
# user profile directory, each with backslashes, forward slashes and doubled
# backslashes (escaped string literals). Not the bare user name: it also occurs
# in the copyright line.
function Get-LeakNeedles()
{
    $needles = @()
    foreach ($root in @($RealRepo, $env:USERPROFILE))
    {
        if ([string]::IsNullOrWhiteSpace($root)) { continue }
        $root = $root.TrimEnd('\', '/')
        $needles += $root.Replace('/', '\')
        $needles += $root.Replace('\', '/')
        $needles += $root.Replace('/', '\').Replace('\', '\\')
    }
    return $needles
}

function Assert-NoPathLeaks([string]$zipPath)
{
    $hits = @(Find-PathLeaks $zipPath (Get-LeakNeedles))
    if ($hits.Count -gt 0)
    {
        $hits | Select-Object -First 50 | ForEach-Object { Write-Host "LEAK  $_" -ForegroundColor Red }
        Fail "$(Split-Path -Leaf $zipPath) embeds the build machine's paths ($($hits.Count) hit(s), listed above) -- nothing is published"
    }
    Write-Host "PASS  leak gate: no repo or user-profile path in $(Split-Path -Leaf $zipPath)"
}

# Maps $RealRepo to the first free drive letter, Z down to M; returns the letter.
function New-RepoDrive()
{
    $used = @([IO.DriveInfo]::GetDrives() | ForEach-Object { $_.Name.Substring(0, 1).ToUpperInvariant() })
    foreach ($code in 90..77)
    {
        $letter = [string][char]$code
        if ($used -contains $letter -or (Test-Path "${letter}:\")) { continue }
        & subst "${letter}:" $RealRepo
        if ($LASTEXITCODE -eq 0) { return $letter }
    }
    Fail 'no free drive letter between M: and Z: to map the repository to'
}

# Unpacks a package zip into $WorkDir\unpacked and runs its selftest\selftest.ps1
# from there (tests the artifact actually shipped). Returns the unpacked package
# prefix, the only output.
function Invoke-PackageSelfTest([string]$ZipPath, [string]$PackageName, [string]$WorkDir, [string]$LogDir)
{
    $unpacked = Join-Path $WorkDir 'unpacked'
    if (Test-Path $unpacked) { Remove-Item -Recurse -Force $unpacked }
    [IO.Compression.ZipFile]::ExtractToDirectory($ZipPath, $unpacked)
    $prefix = Join-Path $unpacked $PackageName
    $log = Join-Path $LogDir 'selftest.log'
    $code = Run $log 'powershell' @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $prefix 'selftest\selftest.ps1')) $WorkDir
    Write-Host (LogText $log)
    if ($code -ne 0) { Fail 'packaged self-test failed' }
    return $prefix
}

# Apps package self-test from a fresh unpack in $WorkDir\unpacked-apps: PATH
# reduced to System32, Studio --selftest, the comparer --help, then the viewer
# must stay up.
function Invoke-AppsSelfTest([string]$ZipPath, [string]$AppsName, [string]$WorkDir, [string]$LogDir)
{
    $AppsUnpacked = Join-Path $WorkDir 'unpacked-apps'
    if (Test-Path $AppsUnpacked) { Remove-Item -Recurse -Force $AppsUnpacked }
    [IO.Compression.ZipFile]::ExtractToDirectory($ZipPath, $AppsUnpacked)
    $AppsBinDir = Join-Path $AppsUnpacked "$AppsName\bin\Release"
    # PATH is reduced to System32 for the run, so only the app-local DLLs and the
    # system ones can load. SABERCAT_SHADER_DIR is cleared so the apps find their
    # shaders next to the package, as an end user's install would.
    $savedPath = $env:PATH
    $savedShaderDir = $env:SABERCAT_SHADER_DIR
    $env:PATH = Join-Path $env:SystemRoot 'System32'
    Remove-Item Env:SABERCAT_SHADER_DIR -ErrorAction SilentlyContinue
    try
    {
        $studioLog = Join-Path $LogDir 'apps-studio-selftest.log'
        $studioOut = Join-Path $AppsUnpacked 'studio-selftest'
        $code = Run $studioLog (Join-Path $AppsBinDir 'sabercat_studio.exe') @('--selftest', $studioOut) $AppsUnpacked
        $studioText = LogText $studioLog
        $studioValidation = ([regex]::Matches($studioText, '\[validation\]')).Count
        if ($code -ne 0 -or $studioValidation -ne 0 -or $studioText -notmatch 'studio selftest OK')
        {
            Write-Host $studioText
            Fail "apps Studio self-test failed (exit $code, $studioValidation validation line(s))"
        }
        Write-Host 'PASS  sabercat_studio --selftest (PATH = System32 only)'

        $compareLog = Join-Path $LogDir 'apps-compare-help.log'
        $code = Run $compareLog (Join-Path $AppsBinDir 'sabercat_compare.exe') @('--help') $AppsUnpacked
        $compareText = LogText $compareLog
        if ($code -ne 0 -or $compareText -notmatch 'usage: sabercat_compare')
        {
            Write-Host $compareText
            Fail "apps sabercat_compare --help failed (exit $code)"
        }
        Write-Host 'PASS  sabercat_compare --help (PATH = System32 only)'

        # The viewer has no headless mode, so start it and check that it is still
        # running after a few seconds: a missing DLL ends the process at once.
        $viewer = Start-Process -FilePath (Join-Path $AppsBinDir 'sabercat_viewer.exe') -WorkingDirectory $AppsUnpacked -PassThru
        $null = $viewer.Handle
        Start-Sleep -Seconds 8
        $viewer.Refresh()
        if ($viewer.HasExited) { Fail "sabercat_viewer.exe exited at startup (exit $($viewer.ExitCode)) with PATH = System32 only" }
        Stop-Process -Id $viewer.Id -Force
        Write-Host 'PASS  sabercat_viewer starts (PATH = System32 only; still running after 8 s)'
    }
    finally
    {
        $env:PATH = $savedPath
        if ($null -ne $savedShaderDir) { $env:SABERCAT_SHADER_DIR = $savedShaderDir }
    }
}

# Swaps an unpacked package into $StableDir by rename: the package moves into
# place whole, so a consumer never sees a half-written prefix. The old copy is
# kept until the swap succeeds. $SourcePrefix must be a path on the real file
# system (the caller passes it under $RealRepo, not the mapped drive): a
# directory cannot be moved off the mapped drive.
function Update-StableDir([string]$StableDir, [string]$SourcePrefix)
{
    Step "Refresh $StableDir"
    $stableFull = [IO.Path]::GetFullPath($StableDir)
    $incoming = "$stableFull.incoming"
    $outgoing = "$stableFull.old"
    foreach ($d in @($incoming, $outgoing)) { if (Test-Path $d) { Remove-Item -Recurse -Force $d } }
    Move-Item $SourcePrefix $incoming
    if (Test-Path $stableFull) { Move-Item $stableFull $outgoing }
    Move-Item $incoming $stableFull
    if (Test-Path $outgoing) { Remove-Item -Recurse -Force $outgoing }
    Write-Host "find_package prefix: $stableFull"
}

# ---------------------------------------------------------------------------
# Switch combinations. Each message names the offending switch.
if ($Tag -ne '' -and -not $VerifyCi) { Fail '-Tag needs -VerifyCi (it names the tag whose CI draft release is verified)' }
if ($VerifyCi)
{
    if ($Tag -eq '') { Fail '-VerifyCi needs -Tag v<x.y.z> (or v<x.y.z>-<suffix>)' }
    if ($Tag -notmatch '^v[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$') { Fail "-VerifyCi: -Tag '$Tag' must look like v1.2.3 or v1.2.3-beta.1" }
    if ($CiBuild) { Fail '-VerifyCi cannot be combined with -CiBuild' }
    if ($AllowDirty) { Fail '-VerifyCi cannot be combined with -AllowDirty (it verifies released bits only)' }
    if ($VersionSuffix -ne '') { Fail '-VerifyCi cannot be combined with -VersionSuffix (the -Tag already carries the suffix)' }
    if ($ExpectTag -ne '') { Fail '-VerifyCi cannot be combined with -ExpectTag (-Tag already names the tag)' }
}
if ($PushTag)
{
    if ($VerifyCi) { Fail '-PushTag cannot be combined with -VerifyCi (tag first, then verify the CI draft)' }
    if ($Publish) { Fail '-PushTag cannot be combined with -Publish (-PushTag only tags; publish the verified draft with -VerifyCi -Publish)' }
    if ($CiBuild) { Fail '-PushTag cannot be combined with -CiBuild' }
    if ($AllowDirty) { Fail '-PushTag cannot be combined with -AllowDirty (a tag names committed code only)' }
}

# ---------------------------------------------------------------------------
# -VerifyCi: checks the CI-built draft release for $Tag against the local
# packages, then (with -Publish) publishes it. It needs no build and no drive
# mapping: everything runs from $RealRepo and a fresh temp folder, and the block
# exits before Preconditions.
if ($VerifyCi)
{
    # gh runs from the checkout, which resolves the GitHub repository from its remotes.
    function Invoke-Gh([string[]]$arguments)
    {
        Push-Location $RealRepo
        try
        {
            $out = & gh @arguments
            $code = $LASTEXITCODE
        }
        finally { Pop-Location }
        if ($code -ne 0) { Fail "gh $($arguments -join ' ') failed (exit $code)" }
        return $out
    }

    # The one release carrying $Tag. The by-tag endpoint does not return drafts,
    # so all releases are listed and filtered here. The jq filter has no double
    # quotes: Windows PowerShell 5.1 passes embedded quotes to native programs
    # unescaped. jq prints one object per line.
    function Find-TagRelease([string]$Slug)
    {
        $found = @()
        $lines = @(Invoke-Gh @('api', "repos/$Slug/releases", '--paginate', '--jq',
            '.[] | {id, tag_name, draft, prerelease, assets: [.assets[] | {id, name, size}]}'))
        foreach ($line in $lines)
        {
            $text = ([string]$line).Trim()
            if ($text -eq '') { continue }
            $r = ConvertFrom-Json -InputObject $text
            if ($r.tag_name -eq $Tag) { $found += $r }
        }
        if ($found.Count -eq 0) { Fail "no release for $Tag found; has the release workflow finished? (check the Actions tab)" }
        if ($found.Count -gt 1) { Fail "$($found.Count) releases carry the tag $Tag; delete all but one by hand" }
        return $found[0]
    }

    Step "Verify the CI packages for $Tag"
    # Names derived from the tag: v1.1.0-beta.1 gives version 1.1.0-beta.1 and
    # suffix beta.1 (the part after the first '-' of the version).
    $Version = $Tag.Substring(1)
    $dash = $Version.IndexOf('-')
    $VerSuffix = if ($dash -ge 0) { $Version.Substring($dash + 1) } else { '' }
    $PackageName = "sabercat-$Version-win64"
    $AppsName = "$PackageName-apps"
    $LocalZip = Join-Path $RealRepo "dist\$PackageName.zip"
    $LocalApps = Join-Path $RealRepo "dist\$AppsName.zip"
    if (-not (Test-Path $LocalZip) -or -not (Test-Path $LocalApps))
    {
        Fail 'run -PushTag first: the locally built packages are missing from dist\ (-VerifyCi compares the CI packages against them)'
    }
    if ($null -eq (Get-Command gh -ErrorAction SilentlyContinue)) { Fail 'gh (GitHub CLI) is not on PATH' }
    if (@(Invoke-Git @('remote')) -notcontains $Remote) { Fail "no git remote '$Remote'" }

    # --quiet throughout, as in Preconditions: git progress on stderr is an error
    # in Windows PowerShell when the caller redirects output.
    Invoke-Git @('fetch', '--quiet', '--tags', $Remote) | Out-Null
    if (-not (RefExists "refs/tags/$Tag")) { Fail "tag $Tag not found; run -PushTag first" }
    $TagSha = (Invoke-Git @('rev-parse', "refs/tags/$Tag^{commit}")).Trim()

    # Stable ancestry is checked before anything is downloaded.
    $PreviousStable = $null
    if ($Publish -and $VerSuffix -eq '')
    {
        foreach ($ref in @('refs/heads/stable', "refs/remotes/$Remote/stable"))
        {
            if (RefExists $ref)
            {
                & git -C $RealRepo merge-base --is-ancestor $ref $TagSha
                if ($LASTEXITCODE -ne 0) { Fail "$ref is not an ancestor of HEAD -- stable only fast-forwards" }
                $PreviousStable = (Invoke-Git @('rev-parse', $ref)).Trim()
            }
        }
    }

    $Slug = [string](@(Invoke-Gh @('repo', 'view', '--json', 'nameWithOwner', '--jq', '.nameWithOwner')) | Select-Object -First 1)
    $Slug = $Slug.Trim()
    if ($Slug -eq '') { Fail 'gh could not name the GitHub repository for this checkout' }

    $release = Find-TagRelease $Slug
    if ($release.draft -ne $true) { Fail "release $Tag is already published; there is no draft to verify" }
    if ([bool]$release.prerelease -ne ($VerSuffix -ne '')) { Fail "release $Tag prerelease flag is $($release.prerelease), but the tag says $($VerSuffix -ne '')" }

    # ---- download into a fresh temp folder (kept on failure, for inspection)
    $Work = Join-Path ([IO.Path]::GetTempPath()) ('sabercat-verifyci-' + [guid]::NewGuid().ToString('N'))
    if ($Work.StartsWith($RealRepo, [StringComparison]::OrdinalIgnoreCase)) { Fail "temp folder $Work is inside the repository" }
    New-Item -ItemType Directory -Path $Work | Out-Null
    $LogDir = Join-Path $Work 'logs'
    New-Item -ItemType Directory -Path $LogDir | Out-Null
    Write-Host "work folder: $Work (kept if verification fails)"
    $CiZip = Join-Path $Work "$PackageName.zip"
    $CiApps = Join-Path $Work "$AppsName.zip"
    $SumsFile = Join-Path $Work 'SHA256SUMS.txt'

    Step "Download the draft's assets"
    $expectedNames = @("$PackageName.zip", "$AppsName.zip", 'SHA256SUMS.txt')
    $foundNames = @($release.assets | ForEach-Object { $_.name })
    $missing = @($expectedNames | Where-Object { $foundNames -notcontains $_ })
    $extra = @($foundNames | Where-Object { $expectedNames -notcontains $_ })
    if ($missing.Count -gt 0 -or $extra.Count -gt 0)
    {
        Fail "release assets do not match: missing [$($missing -join ', ')], unexpected [$($extra -join ', ')], found [$($foundNames -join ', ')]"
    }
    foreach ($asset in $release.assets)
    {
        $dest = Join-Path $Work $asset.name
        # Raw bytes: the asset endpoint with octet-stream, written straight to the file.
        $code = Run $dest 'gh' @('api', '-H', 'Accept: application/octet-stream', "repos/$Slug/releases/assets/$($asset.id)") $RealRepo
        Remove-Item -Force "$dest.stderr" -ErrorAction SilentlyContinue
        if ($code -ne 0) { Fail "download of $($asset.name) failed (exit $code)" }
        $size = (Get-Item $dest).Length
        if ($size -ne [int64]$asset.size) { Fail "$($asset.name) is $size bytes, the release says $($asset.size)" }
        Write-Host ("downloaded {0} ({1:N0} bytes)" -f $asset.name, $size)
    }

    Step 'Check SHA256SUMS.txt'
    $entries = @()
    foreach ($line in @(Get-Content $SumsFile))
    {
        if ($line.Trim() -eq '') { continue }
        if ($line -notmatch '^([0-9a-fA-F]{64})  (.+)$') { Fail "SHA256SUMS.txt line is not '<sha256>  <name>': $line" }
        $entries += [pscustomobject]@{ hash = $Matches[1].ToLowerInvariant(); name = $Matches[2] }
    }
    $sumNames = @($entries | ForEach-Object { $_.name } | Sort-Object)
    $wantNames = @(@("$PackageName.zip", "$AppsName.zip") | Sort-Object)
    if (($sumNames -join "`n") -ne ($wantNames -join "`n")) { Fail "SHA256SUMS.txt must list exactly $($wantNames -join ' and '); it lists: $($sumNames -join ', ')" }
    foreach ($entry in $entries)
    {
        $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $Work $entry.name)).Hash.ToLowerInvariant()
        if ($actual -ne $entry.hash) { Fail "SHA256 mismatch for $($entry.name): SHA256SUMS.txt says $($entry.hash), the download is $actual" }
        Write-Host "PASS  sha256 $($entry.name)"
    }

    Step 'Check the GitHub attestations'
    foreach ($file in @("$PackageName.zip", "$AppsName.zip", 'SHA256SUMS.txt'))
    {
        $log = Join-Path $LogDir "attest-$file.log"
        $code = Run $log 'gh' @('attestation', 'verify', (Join-Path $Work $file), '--repo', $Slug) $RealRepo
        if ($code -ne 0)
        {
            Write-Host (LogText $log)
            Fail "attestation verification failed for $file (exit $code); see $log"
        }
        Write-Host "PASS  attestation $file"
    }

    # Cheap, so it runs before the self-tests: the CI bits must be the local bits.
    Step 'Compare the CI packages with the local packages'
    & (Join-Path $PSScriptRoot 'compare_packages.ps1') -Local @($LocalZip, $LocalApps) -Ci @($CiZip, $CiApps) `
        -ExpectTag $Tag -ExpectVersion $Version -ExpectSha $TagSha
    if ($LASTEXITCODE -ne 0) { Fail 'the CI packages do not match the local packages (see the comparison above)' }

    Step 'Self-test the CI packages'
    # Fresh unpacks under $Work; the GPU suite is not part of a package.
    $CiPrefix = Invoke-PackageSelfTest $CiZip $PackageName $Work $LogDir
    Write-Host "PASS  packaged self-test ($PackageName)"
    Invoke-AppsSelfTest $CiApps $AppsName $Work $LogDir
    Write-Host "PASS  apps self-test ($AppsName)"

    Step "CI packages verified: $Tag"
    Write-Host "tag $Tag at $($TagSha.Substring(0, 7)), repo $Slug, release id $($release.id)"
    Write-Host 'GPU test suite: not run on CI packages (it runs from the build tree during -PushTag)'

    if (-not $Publish)
    {
        Write-Host 'nothing remote was changed'
        Write-Host "to publish the verified draft: powershell -ExecutionPolicy Bypass -File tools\release.ps1 -VerifyCi -Tag $Tag -Publish"
        Remove-Item -Recurse -Force $Work
        exit 0
    }

    # ---- publish: the draft's notes say DRAFT, so the tag annotation is used instead
    Step "Publish $Tag (verified CI packages)"
    $notesLines = @(Invoke-Git @('for-each-ref', '--format=%(contents)', "refs/tags/$Tag"))
    $notes = ($notesLines -join "`n").Trim()
    if ($notes -eq '') { Fail "tag $Tag has no annotation to use as release notes" }
    $notesFile = Join-Path $Work 'release-notes.txt'
    WriteUtf8 $notesFile ($notes + "`n")

    # Re-query just before publishing: the draft must still be the only release for the tag.
    $release = Find-TagRelease $Slug
    if ($release.draft -ne $true) { Fail "release $Tag is already published; nothing to publish" }
    Invoke-Gh @('api', '-X', 'PATCH', "repos/$Slug/releases/$($release.id)", '-F', 'draft=false', '-F', "body=@$notesFile") | Out-Null
    $published = Find-TagRelease $Slug
    if ([bool]$published.draft -or $published.id -ne $release.id) { Fail "release $Tag is still a draft after the publish request" }
    Write-Host "PASS  release $Tag is published"

    if ($VerSuffix -eq '')
    {
        try
        {
            # Fast-forward only: the ancestry was checked above.
            Invoke-Git @('branch', '-f', 'stable', $TagSha) | Out-Null
            Invoke-Git @('push', '--quiet', $Remote, 'refs/heads/stable') | Out-Null
        }
        catch
        {
            Fail "release $Tag is published, but stable was not moved ($($_.Exception.Message)); move it by hand: git branch -f stable $TagSha, then git push $Remote refs/heads/stable"
        }
    }
    else { Write-Host 'stable not moved (pre-release)' }

    if ($VerSuffix -eq '' -and -not [string]::IsNullOrEmpty($StableDir))
    {
        Update-StableDir $StableDir $CiPrefix
    }

    Remove-Item -Recurse -Force $Work
    Step "Released $Tag (verified CI packages)"
    exit 0
}

# ---------------------------------------------------------------------------
Step 'Preconditions'

if ($VersionSuffix -ne '' -and $VersionSuffix -notmatch '^[0-9A-Za-z.-]+$') { Fail "-VersionSuffix '$VersionSuffix' must be semver pre-release identifiers ([0-9A-Za-z.-]+)" }

$cmakeLists = Get-Content (Join-Path $Repo 'CMakeLists.txt') -Raw
if ($cmakeLists -notmatch 'project\(Sabercat\s+VERSION\s+(\d+\.\d+\.\d+)') { Fail 'no project(Sabercat VERSION x.y.z) in CMakeLists.txt' }
$BaseVersion = $Matches[1]
$Version = if ($VersionSuffix -ne '') { "$BaseVersion-$VersionSuffix" } else { $BaseVersion }
$ReleaseTag = "v$Version"
$Sha = (Invoke-Git @('rev-parse', 'HEAD')).Trim()
$ShortSha = $Sha.Substring(0, 7)
$Branch = (Invoke-Git @('rev-parse', '--abbrev-ref', 'HEAD')).Trim()
$Dirty = [bool](Invoke-Git @('status', '--porcelain'))

if ($Dirty -and -not $AllowDirty) { Fail 'working tree has uncommitted changes (commit them, or -AllowDirty for a dry run)' }
if ($Dirty -and ($Publish -or $PushTag)) { Fail '-Publish and -PushTag need a clean working tree: only committed code is released' }
if ($CiBuild -and $Publish) { Fail '-CiBuild never publishes: CI only builds; the release is published by hand after local verification' }
if ($CiBuild -and $AllowDirty) { Fail '-CiBuild needs a clean checkout' }
if ($ExpectTag -ne '' -and $ExpectTag -ne $ReleaseTag) { Fail "tag $ExpectTag does not match the version in CMakeLists.txt (expected $ReleaseTag) -- correct the tag or the version in CMakeLists.txt" }

$PreviousStable = $null
if ($Publish -or $PushTag)
{
    if (@(Invoke-Git @('remote')) -notcontains $Remote) { Fail "no git remote '$Remote'" }
    # --quiet throughout: git reports progress on stderr, which Windows
    # PowerShell turns into errors when the caller redirects output.
    Invoke-Git @('fetch', '--quiet', '--tags', $Remote) | Out-Null
    if (RefExists "refs/tags/$ReleaseTag") { Fail "tag $ReleaseTag already exists -- bump project(Sabercat VERSION ...) in CMakeLists.txt in its own commit (a patch bump keeps consumers' version requests valid; a minor bump needs them raised)" }
    foreach ($ref in @('refs/heads/stable', "refs/remotes/$Remote/stable"))
    {
        if (RefExists $ref)
        {
            & git -C $RealRepo merge-base --is-ancestor $ref $Sha
            if ($LASTEXITCODE -ne 0) { Fail "$ref is not an ancestor of HEAD -- stable only fast-forwards" }
            $PreviousStable = (Invoke-Git @('rev-parse', $ref)).Trim()
        }
    }
}

# No commit hash in the names, so README.md can link the assets of the release
# it is part of; the hash is in the tag and in manifest.json.
$PackageName = "sabercat-$Version-win64"
$AppsName = "sabercat-$Version-win64-apps"
if ($Publish -or $PushTag -or $CiBuild)
{
    # The README in the released commit must link this release's assets.
    $readme = Get-Content -Raw (Join-Path $Repo 'README.md')
    foreach ($asset in @("$PackageName.zip", "$AppsName.zip"))
    {
        $url = "https://github.com/KashyapRajpal/Sabercat/releases/download/v$Version/$asset"
        if (-not $readme.Contains($url)) { Fail "README.md does not link $url -- update its Download section first" }
    }
}
if ($Dirty) { $PackageName += '-dirty'; $AppsName += '-dirty' }
Write-Host "version $Version, commit $ShortSha ($Branch)$(if ($Dirty) { ', DIRTY' }), packages $PackageName and $AppsName"

# ---------------------------------------------------------------------------
Step 'Map the repository to a neutral drive'
$DriveLetter = New-RepoDrive
$Repo = "${DriveLetter}:\"
Write-Host "subst ${DriveLetter}: -> $RealRepo (configure, build, test and package run from $Repo)"
# Everything below runs inside this try, so the mapping is removed on success,
# on failure and on exit. (The body is not indented, to keep the diff small.)
try
{
Push-Location $Repo
$BuildPath = [IO.Path]::GetFullPath((Join-Path $Repo $BuildDir))
$DistPath = Join-Path $Repo 'dist'
$LogPath = Join-Path $BuildPath 'release-logs'

# CMake refuses a build tree configured from another source path, and would
# bake the old path into the outputs. A tree configured under another drive
# letter (or from the real path) cannot be reused: start it afresh.
$cache = Join-Path $BuildPath 'CMakeCache.txt'
if (Test-Path $cache)
{
    $cachedHome = Select-String -Path $cache -Pattern '^CMAKE_HOME_DIRECTORY:INTERNAL=(.*)$' | Select-Object -First 1
    $expectedHome = $Repo.Replace('\', '/').TrimEnd('/')
    if ($null -ne $cachedHome -and $cachedHome.Matches[0].Groups[1].Value.TrimEnd('/') -ne $expectedHome)
    {
        Write-Host "discarding ${BuildDir}: it was configured from $($cachedHome.Matches[0].Groups[1].Value), not $expectedHome"
        Remove-Item -Recurse -Force $BuildPath
    }
}

New-Item -ItemType Directory -Force -Path $LogPath, $DistPath | Out-Null
Get-ChildItem $LogPath | Remove-Item -Recurse -Force

# ---------------------------------------------------------------------------
Step "Configure $BuildDir"
RunOrFail 'configure' (Join-Path $LogPath 'configure.log') 'cmake' @(
    '-S', $Repo, '-B', $BuildPath, '-G', 'Visual Studio 17 2022', '-A', 'x64',
    '-DSABERCAT_BUILD_TESTS=ON', '-DSABERCAT_BUILD_EXAMPLES=ON', '-DSABERCAT_BUILD_APPS=ON',
    "-DSABERCAT_VERSION_SUFFIX=$VersionSuffix")

foreach ($config in @('Debug', 'Release'))
{
    Step "Build $config"
    RunOrFail "$config build" (Join-Path $LogPath "build-$config.log") 'cmake' @(
        '--build', $BuildPath, '--config', $config, '--', '-m', '-v:m', '-nodeReuse:false')
}

# ---------------------------------------------------------------------------
if (-not $CiBuild)
{
# The gpu test names, as CTest lists them for this build tree.
$listing = & ctest -C Release -N -L gpu --test-dir $BuildPath
if ($LASTEXITCODE -ne 0) { Fail "ctest -N -L gpu failed in $BuildPath" }
$TestNames = @($listing | ForEach-Object { if ($_ -match 'Test\s+#\d+:\s+(\S+)') { $Matches[1] } })
if ($TestNames.Count -eq 0) { Fail 'ctest lists no gpu tests' }

$Results = New-Object System.Collections.Generic.List[object]
$testFailures = 0
foreach ($config in @('Debug', 'Release'))
{
    Step "Test $config"
    foreach ($name in $TestNames)
    {
        $log = Join-Path $LogPath "test-$config-$name.log"
        $timer = [Diagnostics.Stopwatch]::StartNew()
        # One test through CTest, so its command line, working directory and
        # timeout are the ones the test list declares. --no-tests=error fails
        # the run if the name matches nothing; -V keeps the output for the count.
        $code = Run $log 'ctest' @('--test-dir', $BuildPath, '-C', $config, '-L', 'gpu',
            '-R', ('^' + $name + '$'), '--no-tests=error', '-V')
        $seconds = [Math]::Round($timer.Elapsed.TotalSeconds, 1)
        $validation = ([regex]::Matches((LogText $log), '\[validation\]')).Count
        $passed = ($code -eq 0 -and $validation -eq 0)
        if (-not $passed) { $testFailures++ }
        $Results.Add([ordered]@{ config = $config; test = $name; passed = $passed; exitCode = $code; validationLines = $validation; seconds = $seconds })
        Write-Host ("{0}  {1,-24} {2,6:N1}s{3}" -f $(if ($passed) { 'PASS' } else { 'FAIL' }), $name, $seconds,
            $(if ($passed) { '' } else { "  (exit $code, $validation validation line(s); $log)" }))
    }
}
}
else
{
# CI: the GPU tests need a GPU this runner does not have; the CPU tests run instead.
$TestNames = @()
$Results = New-Object System.Collections.Generic.List[object]
$testFailures = 0
foreach ($config in @('Debug', 'Release'))
{
    Step "CPU tests $config"
    RunOrFail "$config cpu tests" (Join-Path $LogPath "test-cpu-$config.log") 'ctest' @('--test-dir', $BuildPath, '-C', $config, '-L', 'cpu', '--no-tests=error', '--output-on-failure')
}
}
if ($testFailures -gt 0) { Fail "$testFailures test run(s) failed -- nothing packaged" }

# ---------------------------------------------------------------------------
Step 'Stage package'
$StageRoot = Join-Path $BuildPath 'stage'
$Stage = Join-Path $StageRoot $PackageName
if (Test-Path $StageRoot) { Remove-Item -Recurse -Force $StageRoot }
New-Item -ItemType Directory -Force -Path $Stage | Out-Null
foreach ($config in @('Debug', 'Release'))
{
    RunOrFail "$config install" (Join-Path $LogPath "install-$config.log") 'cmake' @(
        '--install', $BuildPath, '--config', $config, '--prefix', $Stage)
}
foreach ($required in @('bin\Release\sabercat_viewer.exe', 'bin\Release\sabercat_studio.exe',
                        'bin\Release\sabercat_compare.exe', 'bin\Release\sabercat_c.dll',
                        'include\sabercat\shaders\config.slangh', 'include\glm\glm.hpp', 'include\vma\vk_mem_alloc.h',
                        'lib\cmake\Sabercat\SabercatConfig.cmake',
                        'share\sabercat\shaders\forward_project.comp.spv'))
{
    if (-not (Test-Path (Join-Path $Stage $required))) { Fail "install is missing $required" }
}
foreach ($doc in @('README.md', 'LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md', 'CHANGELOG.md'))
{
    Copy-Item (Join-Path $Repo $doc) $Stage
}
$docsDir = Join-Path $Stage 'docs'
New-Item -ItemType Directory -Force -Path $docsDir | Out-Null
foreach ($doc in @('consuming.md', 'library.md', 'c-api.md', 'formats.md'))
{
    Copy-Item (Join-Path $Repo "docs\$doc") $docsDir
}
$selftestDir = Join-Path $Stage 'selftest'
New-Item -ItemType Directory -Path $selftestDir | Out-Null
Copy-Item (Join-Path $Repo 'tools\package\selftest.ps1') $selftestDir
foreach ($name in $SelftestTests)
{
    Copy-Item (Join-Path $BuildPath "examples\$name\Release\sabercat_$name.exe") $selftestDir
}

$manifest = [ordered]@{
    name = 'sabercat'
    version = $Version
    versionSuffix = $VersionSuffix
    tag = $(if ($Publish -or $PushTag -or $ExpectTag -ne '') { $ReleaseTag } else { $null })
    gitSha = $Sha
    gitBranch = $Branch
    dirty = $Dirty
    ci = [bool]$CiBuild
    gpuTested = (-not $CiBuild)
    builtUtc = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    platform = 'win64'
    configurations = @('Debug', 'Release')
    cmake = ((& cmake --version | Select-Object -First 1) -replace 'cmake version ', '')
    vulkanSdk = $env:VULKAN_SDK
    tests = $Results
    githubRunId = $(if ($env:GITHUB_RUN_ID) { $env:GITHUB_RUN_ID } else { $null })
}
WriteUtf8 (Join-Path $Stage 'manifest.json') ($manifest | ConvertTo-Json -Depth 4)

# ---------------------------------------------------------------------------
Step 'Zip'
$Zip = Join-Path $DistPath "$PackageName.zip"
if (Test-Path $Zip) { Remove-Item -Force $Zip }
[IO.Compression.ZipFile]::CreateFromDirectory($Stage, $Zip, [IO.Compression.CompressionLevel]::Optimal, $true)
Write-Host ("{0} ({1:N1} MB)" -f $Zip, ((Get-Item $Zip).Length / 1MB))
Assert-NoPathLeaks $Zip

if ($CiBuild) { Write-Host 'self-tests skipped (-CiBuild: no GPU on this runner)' }
else {
Step 'Self-test the unpacked zip'
$UnpackedPrefix = Invoke-PackageSelfTest $Zip $PackageName $BuildPath $LogPath
}

# ---------------------------------------------------------------------------
Step 'Apps package: viewer, Studio and comparer for testers'
# Same layout as the full package, minus everything but the three apps and what
# they load at run time: bin\Release\ holds the executables and the DLLs they
# import (apps/common/shader_dir.cpp finds share\ two levels above the exe).
$AppsZip = Join-Path $DistPath "$AppsName.zip"
$AppsTop = Join-Path $StageRoot $AppsName
$AppsBin = Join-Path $AppsTop 'bin\Release'
$StageBin = Join-Path $Stage 'bin\Release'
if (Test-Path $AppsTop) { Remove-Item -Recurse -Force $AppsTop }
New-Item -ItemType Directory -Force -Path $AppsBin | Out-Null
foreach ($exe in @('sabercat_viewer.exe', 'sabercat_studio.exe', 'sabercat_compare.exe'))
{
    Copy-Item (Join-Path $StageBin $exe) $AppsBin
}
Copy-Item (Join-Path $Stage 'share') $AppsTop -Recurse
foreach ($doc in @('LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md', 'CHANGELOG.md'))
{
    Copy-Item (Join-Path $Repo $doc) $AppsTop
}
# The readme's first line carries @SABERCAT_VERSION@, replaced by the full
# version; the "=" underline on line 2 is regenerated to match the new title.
$AppsReadme = if ($VersionSuffix -ne '') { 'BETA-README.txt' } else { 'README.txt' }
$readmeText = [System.IO.File]::ReadAllText((Join-Path $Repo "tools\package\$AppsReadme"))
if (-not $readmeText.Contains('@SABERCAT_VERSION@')) { Fail "tools\package\$AppsReadme has no @SABERCAT_VERSION@ token" }
$readmeText = $readmeText.Replace('@SABERCAT_VERSION@', $Version)
$readmeLines = [regex]::Split($readmeText, "\r?\n")
$readmeLines[1] = '=' * $readmeLines[0].Length
[System.IO.File]::WriteAllText((Join-Path $AppsTop $AppsReadme), (($readmeLines -join "`r`n")), (New-Object System.Text.UTF8Encoding($false)))

# Visual Studio: dumpbin for the audit, and the redistributable CRT for the
# app-local Visual C++ runtime (shipped under Microsoft's redistribution terms).
$VsWhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Test-Path $VsWhere)) { Fail 'vswhere.exe not found -- Visual Studio 2022 with C++ tools is needed for the apps package' }
$VsPath = ([string](& $VsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath | Select-Object -First 1)).Trim()
if ($VsPath -eq '') { Fail 'no Visual Studio installation with the C++ tools' }
$ToolDirs = @(Get-ChildItem (Join-Path $VsPath 'VC\Tools\MSVC') -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '^\d+(\.\d+)+$' } | Sort-Object { [version]$_.Name } -Descending)
if ($ToolDirs.Count -eq 0) { Fail "no MSVC tools under $VsPath" }
$Dumpbin = Join-Path $ToolDirs[0].FullName 'bin\Hostx64\x64\dumpbin.exe'
if (-not (Test-Path $Dumpbin)) { Fail "dumpbin.exe not found at $Dumpbin" }
$CrtDir = $null
foreach ($redist in @(Get-ChildItem (Join-Path $VsPath 'VC\Redist\MSVC') -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '^\d+(\.\d+)+$' } | Sort-Object { [version]$_.Name } -Descending))
{
    $crt = @(Get-ChildItem (Join-Path $redist.FullName 'x64') -Directory -Filter 'Microsoft.VC*.CRT' -ErrorAction SilentlyContinue)
    if ($crt.Count -gt 0) { $CrtDir = $crt[0].FullName; break }
}
if ($null -eq $CrtDir) { Fail "no x64 Visual C++ redistributable CRT under $VsPath\VC\Redist\MSVC" }

# The DLL names a binary imports (dumpbin /dependents).
function Get-Imports([string]$binary)
{
    $lines = & $Dumpbin /nologo /dependents $binary
    if ($LASTEXITCODE -ne 0) { Fail "dumpbin failed on $binary" }
    $names = New-Object System.Collections.Generic.List[string]
    $inList = $false
    foreach ($line in $lines)
    {
        $text = ([string]$line).Trim()
        if ($text -eq 'Image has the following dependencies:') { $inList = $true; continue }
        if (-not $inList) { continue }
        if ($text -match '^Summary') { break }
        if ($text -match '^[^\s]+\.dll$') { $names.Add($text) }
    }
    return $names.ToArray()
}

# Dependency closure from the executables. Each import is resolved, in
# order, from the package (copied in from the full package's bin\Release),
# from the Visual C++ CRT, as vulkan-1.dll or a Windows system DLL. Anything
# else fails the release. Copied DLLs are scanned in turn.
$VcRuntime = '^(vcruntime|msvcp|concrt|vccorlib)\d+(_\d+)?\.dll$'
$Queue = New-Object System.Collections.Generic.Queue[string]
foreach ($exe in @(Get-ChildItem $AppsBin -Filter '*.exe')) { $Queue.Enqueue($exe.FullName) }
$Seen = @{}
$Audit = New-Object System.Collections.Generic.List[object]
while ($Queue.Count -gt 0)
{
    $binary = $Queue.Dequeue()
    $owner = Split-Path -Leaf $binary
    foreach ($dll in (Get-Imports $binary))
    {
        $key = $dll.ToLowerInvariant()
        if ($Seen.ContainsKey($key))
        {
            $source = $Seen[$key]
        }
        else
        {
            if (Test-Path (Join-Path $AppsBin $dll)) { $source = 'package' }
            elseif (Test-Path (Join-Path $StageBin $dll))
            {
                Copy-Item (Join-Path $StageBin $dll) $AppsBin
                $Queue.Enqueue((Join-Path $AppsBin $dll))
                $source = 'package (copied from bin)'
            }
            elseif ($dll -match $VcRuntime)
            {
                if (-not (Test-Path (Join-Path $CrtDir $dll))) { Fail "$owner imports $dll, which is not in $CrtDir" }
                Copy-Item (Join-Path $CrtDir $dll) $AppsBin
                $Queue.Enqueue((Join-Path $AppsBin $dll))
                $source = 'Visual C++ runtime (copied)'
            }
            elseif ($key -eq 'vulkan-1.dll') { $source = 'vulkan-1.dll (GPU driver)' }
            elseif ($key -like 'api-ms-win-*' -or $key -like 'ext-ms-*') { $source = 'Windows API set' }
            elseif (Test-Path (Join-Path $env:SystemRoot "System32\$dll")) { $source = 'Windows system DLL' }
            else { Fail "$owner imports $dll, which is neither in the package, a Windows system DLL nor vulkan-1.dll" }
            $Seen[$key] = $source
        }
        $Audit.Add([pscustomobject]@{ binary = $owner; imports = $dll; resolvedBy = $source })
    }
}
Write-Host 'Dependency audit (dumpbin /dependents):'
$Audit | Format-Table -AutoSize | Out-String -Width 200 | Write-Host

if (Test-Path $AppsZip) { Remove-Item -Force $AppsZip }
[IO.Compression.ZipFile]::CreateFromDirectory($AppsTop, $AppsZip, [IO.Compression.CompressionLevel]::Optimal, $true)
Write-Host ("{0} ({1:N2} MB)" -f $AppsZip, ((Get-Item $AppsZip).Length / 1MB))
Assert-NoPathLeaks $AppsZip
$zipArchive = [IO.Compression.ZipFile]::OpenRead($AppsZip)
try
{
    foreach ($entry in $zipArchive.Entries)
    {
        if ($entry.Name -ne '') { Write-Host ("  {0,11:N0}  {1}" -f $entry.Length, $entry.FullName) }
    }
}
finally { $zipArchive.Dispose() }

if (-not $CiBuild)
{
Step 'Self-test the unpacked apps zip'
Invoke-AppsSelfTest $AppsZip $AppsName $BuildPath $LogPath
}

if (-not $Publish -and -not $PushTag)
{
    Step $(if ($CiBuild) { 'CI build complete (nothing published)' } else { 'Dry run complete (nothing published)' })
    Write-Host "package: $Zip"
    Write-Host "apps package: $AppsZip"
    exit 0
}

# ---------------------------------------------------------------------------
# Release notes, shared by -PushTag (the annotation of the tag) and the legacy
# -Publish (the notes of the GitHub Release). Both are past the dry-run exit above.
if ($null -eq $PreviousStable)
{
    $batch = Invoke-Git @('log', '--oneline', '--no-decorate', '-20', $Sha)
    $range = 'initial stable release; latest 20 commits listed'
}
else
{
    $batch = Invoke-Git @('log', '--oneline', '--no-decorate', "$PreviousStable..$Sha")
    $range = "$($PreviousStable.Substring(0, 7))..$ShortSha"
}
$batch = @($batch)
$notes = "Sabercat $ReleaseTag ($range, $($batch.Count) commit(s))`n`n" +
         "All $($TestNames.Count) tests passed in Debug and Release with zero validation messages.`n" +
         "Package: $PackageName.zip -- unpack, then point CMAKE_PREFIX_PATH at it.`n" +
         "Apps for testers: $AppsName.zip -- unpack, then run bin\Release\sabercat_viewer.exe, sabercat_studio.exe or sabercat_compare.exe.`n" +
         $(if ($VersionSuffix -ne '') { "Pre-release: a beta for testing, not a stable release.`n" } else { '' }) +
         "See CHANGELOG.md in the package for the list of changes.`n`n" +
         "Commits:`n" + (($batch | ForEach-Object { "  $_" }) -join "`n") + "`n"
$notesFile = Join-Path $LogPath 'release-notes.txt'
WriteUtf8 $notesFile $notes

# -PushTag: the tag only. CI builds the packages and creates the draft release,
# which -VerifyCi then checks. Nothing else is pushed, and gh is not called.
if ($PushTag)
{
    Step "Tag $ShortSha as $ReleaseTag (tag only; CI builds the draft release)"
    Invoke-Git @('tag', '-a', $ReleaseTag, '-F', $notesFile, $Sha) | Out-Null
    try { Invoke-Git @('push', '--quiet', $Remote, "refs/tags/$ReleaseTag") | Out-Null }
    catch
    {
        # Do not leave a local tag that the remote does not have.
        Invoke-Git @('tag', '-d', $ReleaseTag) | Out-Null
        Fail "could not push $ReleaseTag to $Remote (the local tag was deleted again): $($_.Exception.Message)"
    }
    Write-Host ''
    Write-Host "Pushed $ReleaseTag to $Remote. The tag workflow (.github/workflows/release.yml) now builds and attests the packages and creates a DRAFT release."
    Write-Host 'Next steps:'
    Write-Host "  1. Wait for the tag workflow to finish (Actions tab)."
    Write-Host "  2. powershell -ExecutionPolicy Bypass -File tools\release.ps1 -VerifyCi -Tag $ReleaseTag"
    Write-Host "  3. powershell -ExecutionPolicy Bypass -File tools\release.ps1 -VerifyCi -Tag $ReleaseTag -Publish"
    Write-Host "Keep the local zips in dist\ ($Zip and $AppsZip): -VerifyCi compares the CI packages against them."
    Write-Host 'This mode does not push the branch or stable. Push the branch yourself if needed.'
    exit 0
}

# Legacy local publish, kept as a fallback. The tag-triggered CI workflow will
# find the release already published and stop.
Write-Warning 'legacy local publish: the tag-triggered CI workflow will stop with "already published". Prefer -PushTag, then -VerifyCi.'

Step $(if ($VersionSuffix -ne '') { "Tag $ShortSha as pre-release $ReleaseTag" } else { "Promote $ShortSha to stable as $ReleaseTag" })
# A pre-release (-VersionSuffix) is tagged and published, but `stable` stays on
# the last final release: stable is what consumers build against.
$pushRefs = @("refs/tags/$ReleaseTag")
if ($VersionSuffix -eq '')
{
    # `branch -f` is a fast-forward here: the ancestry check above guarantees it.
    Invoke-Git @('branch', '-f', 'stable', $Sha) | Out-Null
    $pushRefs = @("refs/heads/stable") + $pushRefs
}
Invoke-Git @('tag', '-a', $ReleaseTag, '-F', $notesFile, $Sha) | Out-Null
if ($Branch -ne 'HEAD') { $pushRefs = @("refs/heads/$Branch") + $pushRefs }
Invoke-Git (@('push', '--quiet', $Remote) + $pushRefs) | Out-Null

Step "GitHub Release $ReleaseTag"
$ghArgs = @('release', 'create', $ReleaseTag, $Zip, $AppsZip, '--title', "Sabercat $ReleaseTag", '--notes-file', $notesFile, '--verify-tag')
if ($VersionSuffix -ne '') { $ghArgs += '--prerelease' }
$ghRerun = "gh release create $ReleaseTag `"$Zip`" `"$AppsZip`" --notes-file `"$notesFile`"$(if ($VersionSuffix -ne '') { ' --prerelease' })"
Push-Location $RealRepo # gh resolves the GitHub repository from the checkout's remotes
try { & gh @ghArgs }
finally { Pop-Location }
if ($LASTEXITCODE -ne 0) { Fail "gh release create failed -- the tag is pushed; rerun: $ghRerun" }

# ---------------------------------------------------------------------------
if ($VersionSuffix -ne '')
{
    Step 'Stable prefix refresh skipped (pre-release)'
}
elseif ([string]::IsNullOrEmpty($StableDir))
{
    Step 'Stable prefix refresh skipped (no -StableDir given)'
}
else
{
    # From the real path: a directory cannot be moved off the mapped drive.
    Update-StableDir $StableDir (Join-Path $RealRepo ($UnpackedPrefix.Substring($Repo.Length)))
}

Step "Released $ReleaseTag"
Write-Host "$(if ($VersionSuffix -ne '') { 'pre-release' } else { "stable -> $ShortSha" }), tag $ReleaseTag, $Zip"
}
finally
{
    # Nothing may hold the drive open (the current directory) when it is removed.
    Pop-Location
    Set-Location $RealRepo
    & subst "${DriveLetter}:" /D
    if ($LASTEXITCODE -ne 0) { Write-Warning "could not remove the drive mapping; run: subst ${DriveLetter}: /D" }
}
