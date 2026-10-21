# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

<#
.SYNOPSIS
    Build, test and package a Sabercat release; with -Publish, promote it to
    the `stable` branch, tag it, upload it and refresh the stable prefix.

.DESCRIPTION
    Release model:
      master  -- development.
      stable  -- only ever FAST-FORWARDED to a master commit this script has
                 built and tested; each promotion is an annotated tag
                 v<version> (the version is project(Sabercat VERSION ...) in
                 CMakeLists.txt, bumped by hand in its own commit).
      Release -- every tag gets a GitHub Release carrying the prebuilt package
                 sabercat-<version>-<sha>-win64.zip, so consumers never
                 need to build Sabercat.
      -StableDir -- optional: an unpacked copy of the latest released package;
                 point CMAKE_PREFIX_PATH at it for find_package(Sabercat).
                 Empty (the default) skips the refresh.

    Steps (any failure stops the script; nothing is published unless every
    step before publishing succeeded):
      1. Preconditions: clean working tree (so the tested bits ARE the commit),
         and with -Publish: the tag is new and `stable` can fast-forward.
      2. Configure -BuildDir, build Debug and Release.
      3. Run the full test list in both configurations from the repo root.
         Any non-zero exit or any "[validation]" line fails the release.
      4. Install both configurations into one staging prefix; add docs,
         manifest.json and selftest/ (a few Release tests + selftest.ps1).
      5. Zip it into dist/, unpack the zip into a fresh directory and run the
         packaged self-test from there (tests the artifact actually shipped).
      6. Apps package for testers: the viewer and Studio with only the runtime
         DLLs they import (the Visual C++ runtime is copied in from the Visual
         Studio redistributable). The dependency audit fails the release on any
         import that is neither in the package, a Windows system DLL nor
         vulkan-1.dll. Zipped and self-tested from a fresh directory with PATH
         reduced to System32.
      7. -Publish: fast-forward `stable`, create tag v<version>, push master,
         stable and the tag, create the GitHub Release with both zips, then swap
         the unpacked package into -StableDir.

    The version is the CMake project version, plus -VersionSuffix when given:
    1.0.0-beta.1 gives tag v1.0.0-beta.1, package names with the same version,
    and a GitHub pre-release.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\release.ps1
    Dry run: build, test and package; publishes nothing.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\release.ps1 -AllowDirty -VersionSuffix beta.1
    Dry run of a beta: the packages are named 1.0.0-beta.1; publishing is refused on a dirty tree.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\release.ps1 -Publish
#>
param(
    # Promote to stable, tag, push, create the GitHub Release, refresh -StableDir.
    [switch]$Publish,
    # Package an uncommitted tree (dry runs only; the zip is marked -dirty).
    [switch]$AllowDirty,
    # Unpacked package location to refresh on publish; empty skips the refresh.
    [string]$StableDir = '',
    # Dedicated build tree (git-ignored), reused incrementally between releases.
    [string]$BuildDir = 'out/release',
    [string]$Remote = 'origin',
    # Semver pre-release suffix, e.g. beta.1: the tag becomes v<x.y.z>-<suffix>,
    # the packages carry it, and the GitHub Release is marked as a pre-release.
    [string]$VersionSuffix = ''
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

# Every gpu-labelled CTest test must pass in Debug and Release with zero
# "[validation]" lines. The list is CTest's own (read from the build tree below),
# so a test is added or removed in tests/integration/CMakeLists.txt only.
# Shipped in the package and run by selftest\selftest.ps1. Each resolves its
# SPIR-V through $SABERCAT_SHADER_DIR (apps/common/shader_dir.h).
$SelftestTests = @('device_test', 'forward_test', 'relight_test', 'c_api_test')

$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
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
    $out = & git -C $Repo @arguments
    if ($LASTEXITCODE -ne 0) { Fail "git $($arguments -join ' ') failed" }
    return $out
}

function RefExists([string]$ref)
{
    & git -C $Repo rev-parse --verify --quiet $ref > $null
    return $LASTEXITCODE -eq 0
}

function WriteUtf8([string]$path, [string]$text)
{
    [IO.File]::WriteAllText($path, $text, (New-Object Text.UTF8Encoding $false))
}

# ---------------------------------------------------------------------------
Step 'Preconditions'

if ($VersionSuffix -ne '' -and $VersionSuffix -notmatch '^[0-9A-Za-z.-]+$') { Fail "-VersionSuffix '$VersionSuffix' must be semver pre-release identifiers ([0-9A-Za-z.-]+)" }

$cmakeLists = Get-Content (Join-Path $Repo 'CMakeLists.txt') -Raw
if ($cmakeLists -notmatch 'project\(Sabercat\s+VERSION\s+(\d+\.\d+\.\d+)') { Fail 'no project(Sabercat VERSION x.y.z) in CMakeLists.txt' }
$BaseVersion = $Matches[1]
$Version = if ($VersionSuffix -ne '') { "$BaseVersion-$VersionSuffix" } else { $BaseVersion }
$Tag = "v$Version"
$Sha = (Invoke-Git @('rev-parse', 'HEAD')).Trim()
$ShortSha = $Sha.Substring(0, 7)
$Branch = (Invoke-Git @('rev-parse', '--abbrev-ref', 'HEAD')).Trim()
$Dirty = [bool](Invoke-Git @('status', '--porcelain'))

if ($Dirty -and -not $AllowDirty) { Fail 'working tree has uncommitted changes (commit them, or -AllowDirty for a dry run)' }
if ($Dirty -and $Publish) { Fail '-Publish needs a clean working tree: only committed code is released' }

$PreviousStable = $null
if ($Publish)
{
    if (@(Invoke-Git @('remote')) -notcontains $Remote) { Fail "no git remote '$Remote'" }
    # --quiet throughout: git reports progress on stderr, which Windows
    # PowerShell turns into errors when the caller redirects output.
    Invoke-Git @('fetch', '--quiet', '--tags', $Remote) | Out-Null
    if (RefExists "refs/tags/$Tag") { Fail "tag $Tag already exists -- bump project(Sabercat VERSION ...) in CMakeLists.txt in its own commit (a patch bump keeps consumers' version requests valid; a minor bump needs them raised)" }
    foreach ($ref in @('refs/heads/stable', "refs/remotes/$Remote/stable"))
    {
        if (RefExists $ref)
        {
            & git -C $Repo merge-base --is-ancestor $ref $Sha
            if ($LASTEXITCODE -ne 0) { Fail "$ref is not an ancestor of HEAD -- stable only fast-forwards" }
            $PreviousStable = (Invoke-Git @('rev-parse', $ref)).Trim()
        }
    }
}

# No commit hash in the names, so README.md can link the assets of the release
# it is part of; the hash is in the tag and in manifest.json.
$PackageName = "sabercat-$Version-win64"
$AppsName = "sabercat-$Version-win64-apps"
if ($Publish)
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
foreach ($required in @('bin\Release\sabercat_viewer.exe', 'bin\Release\sabercat_studio.exe', 'bin\Release\sabercat_c.dll',
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
    tag =$(if ($Publish) { $Tag } else { $null })
    gitSha = $Sha
    gitBranch = $Branch
    dirty = $Dirty
    builtUtc = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    platform = 'win64'
    configurations = @('Debug', 'Release')
    cmake = ((& cmake --version | Select-Object -First 1) -replace 'cmake version ', '')
    vulkanSdk = $env:VULKAN_SDK
    tests = $Results
}
WriteUtf8 (Join-Path $Stage 'manifest.json') ($manifest | ConvertTo-Json -Depth 4)

# ---------------------------------------------------------------------------
Step 'Zip'
Add-Type -AssemblyName System.IO.Compression.FileSystem
$Zip = Join-Path $DistPath "$PackageName.zip"
if (Test-Path $Zip) { Remove-Item -Force $Zip }
[IO.Compression.ZipFile]::CreateFromDirectory($Stage, $Zip, [IO.Compression.CompressionLevel]::Optimal, $true)
Write-Host ("{0} ({1:N1} MB)" -f $Zip, ((Get-Item $Zip).Length / 1MB))

Step 'Self-test the unpacked zip'
$Unpacked = Join-Path $BuildPath 'unpacked'
if (Test-Path $Unpacked) { Remove-Item -Recurse -Force $Unpacked }
[IO.Compression.ZipFile]::ExtractToDirectory($Zip, $Unpacked)
$UnpackedPrefix = Join-Path $Unpacked $PackageName
$code = Run (Join-Path $LogPath 'selftest.log') 'powershell' @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $UnpackedPrefix 'selftest\selftest.ps1'))
Write-Host (LogText (Join-Path $LogPath 'selftest.log'))
if ($code -ne 0) { Fail 'packaged self-test failed' }

# ---------------------------------------------------------------------------
Step 'Apps package: viewer and Studio for testers'
# Same layout as the full package, minus everything but the two apps and what
# they load at run time: bin\Release\ holds the executables and the DLLs they
# import (apps/common/shader_dir.cpp finds share\ two levels above the exe).
$AppsTop = Join-Path $StageRoot $AppsName
$AppsBin = Join-Path $AppsTop 'bin\Release'
$StageBin = Join-Path $Stage 'bin\Release'
if (Test-Path $AppsTop) { Remove-Item -Recurse -Force $AppsTop }
New-Item -ItemType Directory -Force -Path $AppsBin | Out-Null
foreach ($exe in @('sabercat_viewer.exe', 'sabercat_studio.exe'))
{
    Copy-Item (Join-Path $StageBin $exe) $AppsBin
}
Copy-Item (Join-Path $Stage 'share') $AppsTop -Recurse
foreach ($doc in @('LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md', 'CHANGELOG.md'))
{
    Copy-Item (Join-Path $Repo $doc) $AppsTop
}
$AppsReadme = if ($VersionSuffix -ne '') { 'BETA-README.txt' } else { 'README.txt' }
Copy-Item (Join-Path $Repo "tools\package\$AppsReadme") $AppsTop

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

# Dependency closure from the two executables. Each import is resolved, in
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

$AppsZip = Join-Path $DistPath "$AppsName.zip"
if (Test-Path $AppsZip) { Remove-Item -Force $AppsZip }
[IO.Compression.ZipFile]::CreateFromDirectory($AppsTop, $AppsZip, [IO.Compression.CompressionLevel]::Optimal, $true)
Write-Host ("{0} ({1:N2} MB)" -f $AppsZip, ((Get-Item $AppsZip).Length / 1MB))
$zipArchive = [IO.Compression.ZipFile]::OpenRead($AppsZip)
try
{
    foreach ($entry in $zipArchive.Entries)
    {
        if ($entry.Name -ne '') { Write-Host ("  {0,11:N0}  {1}" -f $entry.Length, $entry.FullName) }
    }
}
finally { $zipArchive.Dispose() }

Step 'Self-test the unpacked apps zip'
$AppsUnpacked = Join-Path $BuildPath 'unpacked-apps'
if (Test-Path $AppsUnpacked) { Remove-Item -Recurse -Force $AppsUnpacked }
[IO.Compression.ZipFile]::ExtractToDirectory($AppsZip, $AppsUnpacked)
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
    $studioLog = Join-Path $LogPath 'apps-studio-selftest.log'
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

if (-not $Publish)
{
    Step 'Dry run complete (nothing published)'
    Write-Host "package: $Zip"
    Write-Host "apps package: $AppsZip"
    exit 0
}

# ---------------------------------------------------------------------------
Step $(if ($VersionSuffix -ne '') { "Tag $ShortSha as pre-release $Tag" } else { "Promote $ShortSha to stable as $Tag" })
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
$notes = "Sabercat $Tag ($range, $($batch.Count) commit(s))`n`n" +
         "All $($TestNames.Count) tests passed in Debug and Release with zero validation messages.`n" +
         "Package: $PackageName.zip -- unpack, then point CMAKE_PREFIX_PATH at it.`n" +
         "Apps for testers: $AppsName.zip -- unpack, then run bin\Release\sabercat_viewer.exe or sabercat_studio.exe.`n" +
         $(if ($VersionSuffix -ne '') { "Pre-release: a beta for testing, not a stable release.`n" } else { '' }) +
         "See CHANGELOG.md in the package for the list of changes.`n`n" +
         "Commits:`n" + (($batch | ForEach-Object { "  $_" }) -join "`n") + "`n"
$notesFile = Join-Path $LogPath 'release-notes.txt'
WriteUtf8 $notesFile $notes

# A pre-release (-VersionSuffix) is tagged and published, but `stable` stays on
# the last final release: stable is what consumers build against.
$pushRefs = @("refs/tags/$Tag")
if ($VersionSuffix -eq '')
{
    # `branch -f` is a fast-forward here: the ancestry check above guarantees it.
    Invoke-Git @('branch', '-f', 'stable', $Sha) | Out-Null
    $pushRefs = @("refs/heads/stable") + $pushRefs
}
Invoke-Git @('tag', '-a', $Tag, '-F', $notesFile, $Sha) | Out-Null
if ($Branch -ne 'HEAD') { $pushRefs = @("refs/heads/$Branch") + $pushRefs }
Invoke-Git (@('push', '--quiet', $Remote) + $pushRefs) | Out-Null

Step "GitHub Release $Tag"
$ghArgs = @('release', 'create', $Tag, $Zip, $AppsZip, '--title', "Sabercat $Tag", '--notes-file', $notesFile, '--verify-tag')
if ($VersionSuffix -ne '') { $ghArgs += '--prerelease' }
$ghRerun = "gh release create $Tag `"$Zip`" `"$AppsZip`" --notes-file `"$notesFile`"$(if ($VersionSuffix -ne '') { ' --prerelease' })"
Push-Location $Repo # gh resolves the GitHub repository from the checkout's remotes
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
    Step "Refresh $StableDir"
    # Swap by rename: the unpacked, self-tested package moves into place whole, so
    # a consumer never sees a half-written prefix. The old copy is kept until the
    # swap succeeds (a rename fails, leaving it intact, if something has it open).
    $stableFull = [IO.Path]::GetFullPath($StableDir)
    $incoming = "$stableFull.incoming"
    $outgoing = "$stableFull.old"
    foreach ($d in @($incoming, $outgoing)) { if (Test-Path $d) { Remove-Item -Recurse -Force $d } }
    Move-Item $UnpackedPrefix $incoming
    if (Test-Path $stableFull) { Move-Item $stableFull $outgoing }
    Move-Item $incoming $stableFull
    if (Test-Path $outgoing) { Remove-Item -Recurse -Force $outgoing }
    Write-Host "find_package prefix: $stableFull"
}

Step "Released $Tag"
Write-Host "$(if ($VersionSuffix -ne '') { 'pre-release' } else { "stable -> $ShortSha" }), tag $Tag, $Zip"
