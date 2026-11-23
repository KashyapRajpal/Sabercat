# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

<#
.SYNOPSIS
    Cross-check CI-built package zips against the locally built and tested ones.

.DESCRIPTION
    Compares lists of zips pairwise. -Local and -Ci are paired by position: the
    first local zip is compared with the first CI zip, and so on. Both lists must
    have the same length and each pair must have the same file name, otherwise
    the script exits 1 before comparing anything.

    Each pair is checked as follows:

      1. manifest.json identity. The win64 package has a manifest.json; the apps
         package does not. If neither zip of a pair has one, the manifest checks
         are skipped. If only one zip has one, the pair fails. When both have one:
         version, gitSha and tag must match, neither package may be dirty, the
         local manifest must have ci=false and gpuTested=true, and the CI manifest
         must have ci=true and gpuTested=false. -ExpectTag, -ExpectVersion and
         -ExpectSha, when given, must equal the value in both manifests.
      2. The set of entry names, excluding manifest.json. Entries present in only
         one zip are listed and fail the check.
      3. For each shared entry, the SHA-256 of its contents, excluding
         manifest.json (it legitimately differs: build time, CI run, ci and
         gpuTested flags, the test list). A differing non-binary entry is
         compared again after normalizing CRLF to LF. If the two are equal, it
         is reported as "text: line endings differ only" and does not fail. Any
         other difference fails. With -Strict there is no normalization: the
         bytes must be identical.

    For lib/cmake/Sabercat/SabercatTargets.cmake only, a difference limited to
    the generated cmake_policy(VERSION ...) line is accepted and reported as
    "generated cmake_policy version line differs (accepted)". That line reflects
    the CMake version of the build machine. Any other difference in that file
    fails. With -Strict, neither this exception nor the line-ending one applies.

    Entries are classed as PE (.exe, .dll), other binary (.pdb, .lib) or
    non-binary (every other entry). A differing non-binary entry fails, unless
    it is one of the accepted differences described above.
    A differing PE is compared by its import tables (dumpbin /imports, with the
    "Dump of file" and "time date stamp" lines removed) and by its FileVersionInfo
    fields (CompanyName, FileDescription, FileVersion, InternalName,
    LegalCopyright, OriginalFilename, ProductName, ProductVersion). If both match,
    the byte difference is reported as OK. If either differs, the pair fails.
    dumpbin is found on PATH or through vswhere. When it is not found, a NOTE is
    printed and only the version info is compared.
    A differing other binary is a warning only: toolchain changes can alter bytes.

    With -Strict, any differing binary (PE or other) also fails the pair.

    Exit codes: 0 when every pair passes, 1 on any failure or error. Because the
    script exits with these codes, a caller running it in-process (with &) should
    check $LASTEXITCODE.

.PARAMETER Local
    Zips built and tested locally, for example dist\sabercat-1.1.0-win64.zip.
    Paired by position with -Ci.

.PARAMETER Ci
    Zips built by the CI workflow and downloaded from the draft release, in the
    same order as -Local. Each file name must equal the name of its -Local pair.

.PARAMETER Strict
    Require byte-identical contents: no line-ending normalization for text files,
    no cmake_policy exception for SabercatTargets.cmake, and any differing binary
    (PE or other) fails the pair.

.PARAMETER ExpectTag
    When not empty, the tag in both manifest.json files must equal this value.

.PARAMETER ExpectVersion
    When not empty, the version in both manifest.json files must equal this value.

.PARAMETER ExpectSha
    When not empty, the gitSha in both manifest.json files must equal this value.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\compare_packages.ps1 -Local dist\sabercat-1.1.0-win64.zip,dist\sabercat-1.1.0-apps.zip -Ci downloads\sabercat-1.1.0-win64.zip,downloads\sabercat-1.1.0-apps.zip

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\compare_packages.ps1 -Local dist\sabercat-1.1.0-win64.zip,dist\sabercat-1.1.0-apps.zip -Ci downloads\sabercat-1.1.0-win64.zip,downloads\sabercat-1.1.0-apps.zip -Strict -ExpectTag v1.1.0 -ExpectVersion 1.1.0 -ExpectSha <full-commit-sha>
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string[]]$Local,

    [Parameter(Mandatory = $true)]
    [string[]]$Ci,

    [switch]$Strict,

    [string]$ExpectTag = '',

    [string]$ExpectVersion = '',

    [string]$ExpectSha = ''
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$peExtensions = @('.exe', '.dll')
$otherBinaryExtensions = @('.pdb', '.lib')
$versionFields = @('CompanyName', 'FileDescription', 'FileVersion', 'InternalName', 'LegalCopyright', 'OriginalFilename', 'ProductName', 'ProductVersion')

function Get-EntryClass {
    param([string]$Name)
    $ext = [System.IO.Path]::GetExtension($Name).ToLowerInvariant()
    if ($peExtensions -contains $ext) { return 'pe' }
    if ($otherBinaryExtensions -contains $ext) { return 'other-binary' }
    return 'non-binary'
}

function Test-ManifestEntry {
    param([string]$Name)
    # manifest.json sits at the zip root or one folder down (the package folder).
    # Both '/' and '\' are accepted as the separator.
    return ($Name -eq 'manifest.json') -or ($Name -match '^[^/\\]+[/\\]manifest\.json$')
}

function Get-EntryHash {
    param($Entry)
    $stream = $Entry.Open()
    try {
        $sha = [System.Security.Cryptography.SHA256]::Create()
        try {
            $bytes = $sha.ComputeHash($stream)
        } finally {
            $sha.Dispose()
        }
    } finally {
        $stream.Dispose()
    }
    return ([System.BitConverter]::ToString($bytes) -replace '-', '').ToLowerInvariant()
}

function Get-EntryTextHash {
    param($Entry)
    # SHA-256 of the contents with every CRLF turned into LF (line endings normalized).
    $stream = $Entry.Open()
    try {
        $memory = New-Object System.IO.MemoryStream
        try {
            $stream.CopyTo($memory)
            $bytes = $memory.ToArray()
        } finally { $memory.Dispose() }
    } finally { $stream.Dispose() }
    $out = New-Object 'System.Collections.Generic.List[byte]' $bytes.Length
    for ($i = 0; $i -lt $bytes.Length; $i++) {
        if ($bytes[$i] -eq 13 -and ($i + 1) -lt $bytes.Length -and $bytes[$i + 1] -eq 10) { continue }
        $out.Add($bytes[$i])
    }
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try { $hash = $sha.ComputeHash($out.ToArray()) } finally { $sha.Dispose() }
    return ([System.BitConverter]::ToString($hash) -replace '-', '').ToLowerInvariant()
}

function Get-EntryPolicyHash {
    param($Entry)
    # SHA-256 of the contents with CRLF turned into LF and the generated
    # cmake_policy(VERSION ...) line replaced by a fixed placeholder.
    $stream = $Entry.Open()
    try {
        $reader = New-Object System.IO.StreamReader($stream, [System.Text.Encoding]::UTF8)
        try {
            $text = $reader.ReadToEnd()
        } finally {
            $reader.Dispose()
        }
    } finally {
        $stream.Dispose()
    }
    $text = $text.Replace("`r`n", "`n")
    $text = [System.Text.RegularExpressions.Regex]::Replace($text, '(?m)^cmake_policy\(VERSION [^\r\n)]*\)', 'cmake_policy(VERSION <generated>)')
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try { $hash = $sha.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($text)) } finally { $sha.Dispose() }
    return ([System.BitConverter]::ToString($hash) -replace '-', '').ToLowerInvariant()
}

function Test-GeneratedCmakeEntry {
    param([string]$Name)
    # Only this generated file may differ in its cmake_policy(VERSION ...) line.
    return ($Name -replace '\\','/') -match '(^|/)lib/cmake/Sabercat/SabercatTargets\.cmake$'
}

function Get-ManifestObject {
    param($Entry)
    $stream = $Entry.Open()
    try {
        $reader = New-Object System.IO.StreamReader($stream, [System.Text.Encoding]::UTF8)
        try {
            $text = $reader.ReadToEnd()
        } finally {
            $reader.Dispose()
        }
    } finally {
        $stream.Dispose()
    }
    return ($text | ConvertFrom-Json)
}

function Get-ManifestField {
    param($Manifest, [string]$Field)
    if ($null -eq $Manifest) { return $null }
    $prop = $Manifest.PSObject.Properties[$Field]
    if ($null -eq $prop) { return $null }
    return $prop.Value
}

function Format-Field {
    param($Value)
    if ($null -eq $Value) { return '(missing)' }
    return [string]$Value
}

function Get-Text {
    param($Obj, [string]$Field)
    return (Format-Field (Get-ManifestField $Obj $Field))
}

function Format-ManifestLine {
    param($Obj)
    return ('version={0} gitSha={1} tag={2} dirty={3} ci={4} gpuTested={5}' -f `
        (Get-Text $Obj 'version'), (Get-Text $Obj 'gitSha'), (Get-Text $Obj 'tag'), `
        (Get-Text $Obj 'dirty'), (Get-Text $Obj 'ci'), (Get-Text $Obj 'gpuTested'))
}

function Test-ExpectedField {
    param($LocalObj, $CiObj, [string]$Field, [string]$Expected)
    if ($Expected -eq '') { return $true }
    $ok = $true
    $localValue = Get-Text $LocalObj $Field
    $ciValue = Get-Text $CiObj $Field
    if ($localValue -cne $Expected) {
        Write-Host ('FAIL: local {0} is {1}, expected {2}.' -f $Field, $localValue, $Expected)
        $ok = $false
    }
    if ($ciValue -cne $Expected) {
        Write-Host ('FAIL: CI {0} is {1}, expected {2}.' -f $Field, $ciValue, $Expected)
        $ok = $false
    }
    return $ok
}

function Test-ManifestPair {
    param($LocalObj, $CiObj, [string]$ExpectTag, [string]$ExpectVersion, [string]$ExpectSha)
    $bad = $false

    Write-Host ('  local: ' + (Format-ManifestLine $LocalObj))
    Write-Host ('  CI:    ' + (Format-ManifestLine $CiObj))

    foreach ($field in @('version', 'gitSha', 'tag')) {
        if ((Get-Text $LocalObj $field) -cne (Get-Text $CiObj $field)) {
            Write-Host ('FAIL: {0} differs.' -f $field)
            $bad = $true
        }
    }

    $localDirty = (Get-Text $LocalObj 'dirty').ToLowerInvariant()
    $ciDirty = (Get-Text $CiObj 'dirty').ToLowerInvariant()
    if ($localDirty -eq 'true') {
        Write-Host 'FAIL: the local package is dirty.'
        $bad = $true
    }
    if ($ciDirty -eq 'true') {
        Write-Host 'FAIL: the CI package is dirty.'
        $bad = $true
    }

    $localCi = (Get-Text $LocalObj 'ci').ToLowerInvariant()
    $ciCi = (Get-Text $CiObj 'ci').ToLowerInvariant()
    $localGpu = (Get-Text $LocalObj 'gpuTested').ToLowerInvariant()
    $ciGpu = (Get-Text $CiObj 'gpuTested').ToLowerInvariant()
    if ($localCi -ne 'false') {
        Write-Host ('FAIL: local manifest ci must be false (is {0}).' -f $localCi)
        $bad = $true
    }
    if ($ciCi -ne 'true') {
        Write-Host ('FAIL: CI manifest ci must be true (is {0}).' -f $ciCi)
        $bad = $true
    }
    if ($localGpu -ne 'true') {
        Write-Host ('FAIL: local manifest gpuTested must be true (is {0}).' -f $localGpu)
        $bad = $true
    }
    if ($ciGpu -ne 'false') {
        Write-Host ('FAIL: CI manifest gpuTested must be false (is {0}).' -f $ciGpu)
        $bad = $true
    }

    if (-not (Test-ExpectedField $LocalObj $CiObj 'tag' $ExpectTag)) { $bad = $true }
    if (-not (Test-ExpectedField $LocalObj $CiObj 'version' $ExpectVersion)) { $bad = $true }
    if (-not (Test-ExpectedField $LocalObj $CiObj 'gitSha' $ExpectSha)) { $bad = $true }

    return (-not $bad)
}

function Save-Entry {
    param($Entry, [string]$Path)
    $src = $Entry.Open()
    try {
        $dst = [System.IO.File]::Create($Path)
        try {
            $src.CopyTo($dst)
        } finally {
            $dst.Dispose()
        }
    } finally {
        $src.Dispose()
    }
}

function Get-VersionLines {
    param([string]$Path)
    $info = [System.Diagnostics.FileVersionInfo]::GetVersionInfo($Path)
    $lines = @()
    foreach ($field in $versionFields) {
        $value = [string]$info.$field
        $lines += ('{0}={1}' -f $field, $value)
    }
    return ,$lines
}

function Get-ImportLines {
    param([string]$DumpBinPath, [string]$FilePath, [string]$WorkDir, [string]$Tag)
    $outFile = Join-Path $WorkDir ($Tag + '_imports.txt')
    $errFile = Join-Path $WorkDir ($Tag + '_imports_err.txt')
    $proc = Start-Process -FilePath $DumpBinPath -ArgumentList @('/nologo', '/imports', ('"{0}"' -f $FilePath)) `
        -NoNewWindow -Wait -PassThru -RedirectStandardOutput $outFile -RedirectStandardError $errFile
    if ($proc.ExitCode -ne 0) {
        throw ('dumpbin exited with code {0} on {1}' -f $proc.ExitCode, $FilePath)
    }
    $raw = @(Get-Content -LiteralPath $outFile)
    $kept = @()
    foreach ($line in $raw) {
        if ($line -match '^\s*Dump of file ') { continue }
        if ($line -match 'time date stamp') { continue }
        $kept += $line.TrimEnd()
    }
    return ,$kept
}

function Write-LineDiff {
    param([string[]]$Left, [string[]]$Right)
    $max = [Math]::Max($Left.Count, $Right.Count)
    $shown = 0
    for ($i = 0; ($i -lt $max) -and ($shown -lt 5); $i++) {
        $a = if ($i -lt $Left.Count) { $Left[$i] } else { '(end)' }
        $b = if ($i -lt $Right.Count) { $Right[$i] } else { '(end)' }
        if ($a -ne $b) {
            Write-Host ('    local: {0}' -f $a)
            Write-Host ('    CI:    {0}' -f $b)
            $shown++
        }
    }
}

function Find-DumpBin {
    $cmd = Get-Command dumpbin.exe -ErrorAction SilentlyContinue
    if ($null -ne $cmd) { return $cmd.Source }

    $pf86 = ${env:ProgramFiles(x86)}
    if (-not $pf86) { return $null }
    $vswhere = Join-Path $pf86 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path -LiteralPath $vswhere)) { return $null }

    $vsRoot = [string](@(& $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath) | Select-Object -First 1)
    if (-not $vsRoot) { return $null }

    $msvcRoot = Join-Path $vsRoot 'VC\Tools\MSVC'
    if (-not (Test-Path -LiteralPath $msvcRoot)) { return $null }
    $versions = @(Get-ChildItem -LiteralPath $msvcRoot -Directory | Sort-Object { try { [version]$_.Name } catch { [version]'0.0' } } -Descending)
    foreach ($dir in $versions) {
        $candidate = Join-Path $dir.FullName 'bin\Hostx64\x64\dumpbin.exe'
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    return $null
}

function Compare-PeEntry {
    param($LocalEntry, $CiEntry, [string]$Name, [string]$DumpBinPath)

    $workDir = Join-Path ([System.IO.Path]::GetTempPath()) ([System.Guid]::NewGuid().ToString('N'))
    $ok = $true
    $localImports = @()
    $ciImports = @()
    $importsCompared = $false
    $importsDiffer = $false
    try {
        [void][System.IO.Directory]::CreateDirectory($workDir)
        $leafName = [System.IO.Path]::GetFileName($Name)
        $localFile = Join-Path $workDir ('local_' + $leafName)
        $ciFile = Join-Path $workDir ('ci_' + $leafName)
        Save-Entry $LocalEntry $localFile
        Save-Entry $CiEntry $ciFile

        $localVersion = @(Get-VersionLines $localFile)
        $ciVersion = @(Get-VersionLines $ciFile)
        $versionDiffers = (($localVersion -join "`n") -ne ($ciVersion -join "`n"))

        if ($null -ne $DumpBinPath) {
            try {
                $localImports = @(Get-ImportLines $DumpBinPath $localFile $workDir 'local')
                $ciImports = @(Get-ImportLines $DumpBinPath $ciFile $workDir 'ci')
                $importsCompared = $true
                $importsDiffer = (($localImports -join "`n") -ne ($ciImports -join "`n"))
            } catch {
                Write-Host ('  FAIL: dumpbin failed on {0}: {1}' -f $Name, $_.Exception.Message)
                $ok = $false
            }
        }

        if ($importsDiffer) {
            Write-Host ('  FAIL: import tables differ: {0}' -f $Name)
            Write-LineDiff $localImports $ciImports
            $ok = $false
        }
        if ($versionDiffers) {
            Write-Host ('  FAIL: version info differs: {0}' -f $Name)
            Write-LineDiff $localVersion $ciVersion
            $ok = $false
        }
        if ($ok) {
            if ($importsCompared) {
                Write-Host ('  OK (bytes differ; imports and version info identical): {0}' -f $Name)
            } else {
                Write-Host ('  OK (bytes differ; version info identical, imports not compared): {0}' -f $Name)
            }
        }
        return $ok
    } finally {
        if (Test-Path -LiteralPath $workDir) {
            Remove-Item -LiteralPath $workDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

function Compare-Pair {
    param([string]$LocalPath, [string]$CiPath)

    $zipLocal = $null
    $zipCi = $null
    $failed = $false

    try {
        $zipLocal = [System.IO.Compression.ZipFile]::OpenRead($LocalPath)
        $zipCi = [System.IO.Compression.ZipFile]::OpenRead($CiPath)

        $localEntries = @{}
        $ciEntries = @{}
        $localManifest = $null
        $ciManifest = $null

        foreach ($entry in $zipLocal.Entries) {
            if ($entry.FullName.EndsWith('/')) { continue }
            if (Test-ManifestEntry $entry.FullName) {
                $localManifest = $entry
                continue
            }
            $localEntries[$entry.FullName] = $entry
        }
        foreach ($entry in $zipCi.Entries) {
            if ($entry.FullName.EndsWith('/')) { continue }
            if (Test-ManifestEntry $entry.FullName) {
                $ciManifest = $entry
                continue
            }
            $ciEntries[$entry.FullName] = $entry
        }

        Write-Host "Local: $LocalPath"
        Write-Host "CI:    $CiPath"
        Write-Host ''

        # 1. manifest.json identity.
        Write-Host '== manifest.json identity =='
        if ($null -eq $localManifest -and $null -eq $ciManifest) {
            Write-Host 'no manifest.json in either zip (apps package): manifest checks skipped'
        } elseif ($null -eq $localManifest -or $null -eq $ciManifest) {
            Write-Host 'FAIL: manifest.json present in only one zip.'
            $failed = $true
        } else {
            $localObj = Get-ManifestObject $localManifest
            $ciObj = Get-ManifestObject $ciManifest
            if (-not (Test-ManifestPair $localObj $ciObj $ExpectTag $ExpectVersion $ExpectSha)) {
                $failed = $true
            }
        }
        Write-Host ''

        # 2. Name sets.
        Write-Host '== entry names =='
        $onlyLocal = @($localEntries.Keys | Where-Object { -not $ciEntries.ContainsKey($_) } | Sort-Object)
        $onlyCi = @($ciEntries.Keys | Where-Object { -not $localEntries.ContainsKey($_) } | Sort-Object)
        $shared = @($localEntries.Keys | Where-Object { $ciEntries.ContainsKey($_) } | Sort-Object)

        foreach ($name in $onlyLocal) { Write-Host "  only local: $name" }
        foreach ($name in $onlyCi) { Write-Host "  only CI:    $name" }
        if ($onlyLocal.Count -gt 0 -or $onlyCi.Count -gt 0) {
            Write-Host 'FAIL: the name sets differ.'
            $failed = $true
        }
        Write-Host ''

        # 3. Contents of shared entries.
        Write-Host '== contents =='
        $identical = 0
        $eolOnly = @()
        $policyOnly = @()
        $binaryDiff = @()
        $nonBinaryDiff = @()

        foreach ($name in $shared) {
            $localHash = Get-EntryHash $localEntries[$name]
            $ciHash = Get-EntryHash $ciEntries[$name]
            if ($localHash -eq $ciHash) {
                $identical++
            } elseif ((Get-EntryClass $name) -eq 'non-binary') {
                if ((-not $Strict) -and ((Get-EntryTextHash $localEntries[$name]) -eq (Get-EntryTextHash $ciEntries[$name]))) {
                    $eolOnly += $name
                } elseif ((-not $Strict) -and (Test-GeneratedCmakeEntry $name) -and ((Get-EntryPolicyHash $localEntries[$name]) -eq (Get-EntryPolicyHash $ciEntries[$name]))) {
                    $policyOnly += $name
                } else {
                    $nonBinaryDiff += $name
                }
            } else {
                $binaryDiff += $name
            }
        }

        foreach ($name in $nonBinaryDiff) {
            Write-Host "  FAIL: non-binary differences: $name"
            $failed = $true
        }
        foreach ($name in $eolOnly) {
            Write-Host "  text: line endings differ only: $name"
        }
        foreach ($name in $policyOnly) {
            Write-Host "  generated cmake_policy version line differs (accepted): $name"
        }

        $peDiffer = 0
        $peIdentical = 0
        foreach ($name in $binaryDiff) {
            if ((Get-EntryClass $name) -eq 'pe') {
                $peDiffer++
                $peOk = Compare-PeEntry $localEntries[$name] $ciEntries[$name] $name $script:DumpBinPath
                if ($peOk) {
                    $peIdentical++
                } else {
                    $failed = $true
                }
            } else {
                Write-Host "  WARNING: binary differs: $name"
            }
            if ($Strict) {
                Write-Host "  FAIL (-Strict): binary differs: $name"
                $failed = $true
            }
        }
        Write-Host ''

        # Summary.
        $peLabel = if ($null -ne $script:DumpBinPath) { 'identical imports and version info' } else { 'identical version info, imports not compared' }
        Write-Host '== summary =='
        Write-Host ('  entries compared (excluding manifest.json): {0}' -f $shared.Count)
        Write-Host ('  identical:                                  {0}' -f $identical)
        Write-Host ('  text, line endings differ only:             {0}' -f $eolOnly.Count)
        Write-Host ('  generated cmake_policy line, accepted:      {0}' -f $policyOnly.Count)
        Write-Host ('  differing:                                  {0} (binary {1}, non-binary {2})' -f ($binaryDiff.Count + $nonBinaryDiff.Count), $binaryDiff.Count, $nonBinaryDiff.Count)
        Write-Host ('  PE differing with {0}: {1} of {2}' -f $peLabel, $peIdentical, $peDiffer)
        Write-Host ('  only local:                                 {0}' -f $onlyLocal.Count)
        Write-Host ('  only CI:                                    {0}' -f $onlyCi.Count)
        Write-Host ('  mode:                                       {0}' -f $(if ($Strict) { 'strict' } else { 'default' }))
        Write-Host ''

        return (-not $failed)
    } catch {
        Write-Host ('ERROR: {0}' -f $_.Exception.Message)
        return $false
    } finally {
        if ($null -ne $zipLocal) { $zipLocal.Dispose() }
        if ($null -ne $zipCi) { $zipCi.Dispose() }
    }
}

$overallOk = $true
try {
    if ($Local.Count -ne $Ci.Count) {
        throw ('-Local has {0} zip(s) but -Ci has {1}; they must pair up by position.' -f $Local.Count, $Ci.Count)
    }

    $resolvedLocal = @()
    $resolvedCi = @()
    for ($i = 0; $i -lt $Local.Count; $i++) {
        $localPath = (Resolve-Path -LiteralPath $Local[$i]).ProviderPath
        $ciPath = (Resolve-Path -LiteralPath $Ci[$i]).ProviderPath
        $localName = [System.IO.Path]::GetFileName($localPath)
        $ciName = [System.IO.Path]::GetFileName($ciPath)
        if ($localName -ne $ciName) {
            throw ('pair {0}: file names differ: local {1} vs CI {2}' -f ($i + 1), $localName, $ciName)
        }
        $resolvedLocal += $localPath
        $resolvedCi += $ciPath
    }

    $script:DumpBinPath = Find-DumpBin
    if ($null -eq $script:DumpBinPath) {
        Write-Host 'NOTE: dumpbin.exe not found (PATH or Visual Studio). PE import tables are not compared; only PE version info is.'
    } else {
        Write-Host ('dumpbin: {0}' -f $script:DumpBinPath)
    }
    Write-Host ''

    $total = $resolvedLocal.Count
    for ($i = 0; $i -lt $total; $i++) {
        Write-Host ('== pair {0} of {1}: {2} ==' -f ($i + 1), $total, [System.IO.Path]::GetFileName($resolvedLocal[$i]))
        $pairOk = Compare-Pair $resolvedLocal[$i] $resolvedCi[$i]
        Write-Host ('PAIR RESULT: {0}' -f $(if ($pairOk) { 'OK' } else { 'FAIL' }))
        Write-Host ''
        if (-not $pairOk) { $overallOk = $false }
    }
} catch {
    Write-Host ('ERROR: {0}' -f $_.Exception.Message)
    $overallOk = $false
}

if ($overallOk) {
    Write-Host 'RESULT: OK'
    exit 0
}
Write-Host 'RESULT: FAIL'
exit 1
