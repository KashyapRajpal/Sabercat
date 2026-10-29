# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

<#
.SYNOPSIS
    Sabercat package self-test: runs a few packaged tests against this
    package's own DLLs and SPIR-V, on this machine's GPU, then checks that a
    consumer can find_package(Sabercat) the package.

.DESCRIPTION
    Shipped in every Sabercat release package (tools/release.ps1). Run it
    after unpacking, or after a driver update, to confirm the package works
    here. Each test compares the GPU against a CPU reference; the run fails
    on any non-zero exit code or any "[validation]" line (the Vulkan
    validation layer is used when installed and skipped otherwise).

    It also runs the packaged Studio self-test (no scene files: a synthetic
    project), then builds a small find_package(Sabercat 1.0) consumer against
    the unpacked package and checks that a request for 2.0 is refused. The
    consumer steps need cmake and Visual Studio 2022; without them they are
    skipped with a message.

    powershell -ExecutionPolicy Bypass -File selftest\selftest.ps1
#>
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$selftestDir = $PSScriptRoot
$prefix = Split-Path -Parent $selftestDir
$binDir = Join-Path $prefix 'bin\Release'
$shaderDir = Join-Path $prefix 'share\sabercat\shaders'
foreach ($required in @($binDir, $shaderDir))
{
    if (-not (Test-Path $required)) { throw "selftest: missing $required -- is this an unpacked Sabercat package?" }
}

# The tests resolve their SPIR-V through $SABERCAT_SHADER_DIR and their DLLs
# (sabercat_c, zlib) through PATH.
$env:SABERCAT_SHADER_DIR = $shaderDir
$env:PATH = "$binDir;$env:PATH"

$workDir = Join-Path ([IO.Path]::GetTempPath()) ("sabercat-selftest-" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $workDir | Out-Null

# Runs a program from $workDir with its output redirected to files, so native
# stderr never reaches PowerShell's error stream. Returns the exit code and text.
function Invoke-Program([string]$exe, [string[]]$arguments, [string]$tag)
{
    $stdout = Join-Path $workDir ($tag + '.out.txt')
    $stderr = Join-Path $workDir ($tag + '.err.txt')
    $startArgs = @{
        FilePath = $exe; WorkingDirectory = $workDir; NoNewWindow = $true; PassThru = $true
        RedirectStandardOutput = $stdout; RedirectStandardError = $stderr
    }
    # Start-Process rejects an empty -ArgumentList; quote arguments with spaces.
    if ($arguments.Count -gt 0)
    {
        $startArgs.ArgumentList = ($arguments | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join ' '
    }
    $p = Start-Process @startArgs
    $null = $p.Handle # PS 5.1: keep the handle so ExitCode survives exit
    $p.WaitForExit()
    $text = [string](Get-Content $stdout -Raw) + [string](Get-Content $stderr -Raw)
    return [pscustomobject]@{ ExitCode = $p.ExitCode; Text = $text }
}

$failed = 0
try
{
    $exes = @(Get-ChildItem -Path $selftestDir -Filter '*.exe' | Sort-Object Name)
    if ($exes.Count -eq 0) { throw "selftest: no test programs in $selftestDir -- the package is incomplete" }
    foreach ($exe in $exes)
    {
        $result = Invoke-Program $exe.FullName @() $exe.BaseName
        $validation = ([regex]::Matches($result.Text, '\[validation\]')).Count
        if ($result.ExitCode -eq 0 -and $validation -eq 0)
        {
            Write-Host ("PASS  {0}" -f $exe.BaseName)
        }
        else
        {
            $failed++
            Write-Host ("FAIL  {0}  (exit {1}, {2} validation line(s))" -f $exe.BaseName, $result.ExitCode, $validation)
            Write-Host $result.Text
        }
    }

    # The packaged Studio, headless, on its synthetic project (no scene files).
    $studio = Join-Path $binDir 'sabercat_studio.exe'
    if (-not (Test-Path $studio)) { throw "selftest: missing $studio" }
    $studioOut = Join-Path $workDir 'studio-selftest'
    $result = Invoke-Program $studio @('--selftest', $studioOut) 'sabercat_studio'
    $validation = ([regex]::Matches($result.Text, '\[validation\]')).Count
    if ($result.ExitCode -eq 0 -and $validation -eq 0 -and $result.Text -match 'studio selftest OK')
    {
        Write-Host 'PASS  sabercat_studio --selftest'
    }
    else
    {
        $failed++
        Write-Host ("FAIL  sabercat_studio --selftest  (exit {0}, {1} validation line(s))" -f $result.ExitCode, $validation)
        Write-Host $result.Text
    }

    # The consumer: find_package(Sabercat 1.0) builds and runs against this
    # package; a request for 2.0 must fail (the package's version file is
    # SameMajorVersion). Needs cmake and VS 2022.
    $cmake = Get-Command cmake -ErrorAction SilentlyContinue
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    $vsFound = (Test-Path $vswhere) -and [bool](& $vswhere -latest -products * -requires Microsoft.Component.MSBuild -property installationPath)
    if ($null -eq $cmake -or -not $vsFound)
    {
        Write-Host 'SKIP  consumer find_package (needs cmake and Visual Studio 2022 on this machine)'
    }
    else
    {
        $consumerTemplate = @'
cmake_minimum_required(VERSION 3.22)
project(consumer LANGUAGES CXX)
find_package(Sabercat VERSION_REQUEST REQUIRED CONFIG)
add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE Sabercat::sabercat)
'@
        $consumerMain = @'
#include <sabercat/version.h>
int main()
{
    return SABERCAT_VERSION_MAJOR == 1 ? 0 : 1;
}
'@
        function New-Consumer([string]$name, [string]$version)
        {
            $dir = Join-Path $workDir $name
            New-Item -ItemType Directory -Path $dir | Out-Null
            [IO.File]::WriteAllText((Join-Path $dir 'CMakeLists.txt'), $consumerTemplate.Replace('VERSION_REQUEST', $version), (New-Object Text.UTF8Encoding $false))
            [IO.File]::WriteAllText((Join-Path $dir 'main.cpp'), $consumerMain, (New-Object Text.UTF8Encoding $false))
            return $dir
        }

        $good = New-Consumer 'consumer-1.0' '1.0'
        $goodBuild = Join-Path $good 'build'
        $configure = Invoke-Program $cmake.Source @('-S', $good, '-B', $goodBuild, '-G', 'Visual Studio 17 2022', '-A', 'x64', "-DCMAKE_PREFIX_PATH=$prefix") 'consumer-configure'
        $built = $false
        if ($configure.ExitCode -eq 0)
        {
            $build = Invoke-Program $cmake.Source @('--build', $goodBuild, '--config', 'Release') 'consumer-build'
            $built = $build.ExitCode -eq 0
        }
        $run = if ($built) { Invoke-Program (Join-Path $goodBuild 'Release\consumer.exe') @() 'consumer-run' } else { [pscustomobject]@{ ExitCode = -1 } }
        if ($built -and $run.ExitCode -eq 0)
        {
            Write-Host 'PASS  consumer find_package(Sabercat 1.0) builds and runs'
        }
        else
        {
            $failed++
            Write-Host ("FAIL  consumer find_package(Sabercat 1.0)  (configure exit {0}, build {1}, run {2})" -f $configure.ExitCode, $built, $run.ExitCode)
            Write-Host $configure.Text
        }

        $newer = New-Consumer 'consumer-2.0' '2.0'
        $refused = Invoke-Program $cmake.Source @('-S', $newer, '-B', (Join-Path $newer 'build'), '-G', 'Visual Studio 17 2022', '-A', 'x64', "-DCMAKE_PREFIX_PATH=$prefix") 'consumer-2.0-configure'
        if ($refused.ExitCode -ne 0)
        {
            Write-Host 'PASS  consumer find_package(Sabercat 2.0) is refused (SameMajorVersion)'
        }
        else
        {
            $failed++
            Write-Host 'FAIL  consumer find_package(Sabercat 2.0) was accepted by a 1.0 package'
        }
    }
}
finally
{
    Remove-Item -Recurse -Force $workDir -ErrorAction SilentlyContinue
}

if ($failed -gt 0)
{
    Write-Host "selftest: $failed test(s) FAILED"
    exit 1
}
Write-Host 'selftest: all tests passed'
exit 0
