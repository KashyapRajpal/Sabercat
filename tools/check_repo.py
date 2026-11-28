# Copyright 2026 Kashyap Rajpal
# SPDX-License-Identifier: Apache-2.0

"""Check the tracked and untracked files of a git repository for release hygiene.

Usage:
    python tools/check_repo.py [--root DIR] [--self-test]

--root defaults to the parent of the directory holding this script. The file
list comes from one ``git ls-files`` call, so ignored files are skipped and
submodule gitlinks are dropped. UTF-16 files with a byte order mark are decoded
and checked like text. Other binary files (a NUL byte in the first 8000 bytes,
no byte order mark) are skipped.

--self-test runs the built-in fixtures through the same checks and exits
without reading git or any repository file.

Exit status: 0 when clean, 1 when there are findings, 2 when git fails.

Checks:
    header      Files of a supported type must carry a line
                ``Copyright <year> Kashyap Rajpal`` and a line
                ``SPDX-License-Identifier: Apache-2.0`` within the first five
                lines. Checked for CMakeLists.txt and for the suffixes in
                HEADER_SUFFIXES (C/C++, shader, PowerShell, CMake, YAML,
                JavaScript and Python). Every other suffix is exempt, including
                the tools/package/*.txt files and Markdown.
    local-path  No Windows drive-letter path and no POSIX home-directory path
                (under a Users or home folder) on any line. Checked for every
                text file, exempt suffixes included.
    secret      No known API token shape and no hard-coded password, secret,
                API key or access token literal on any line. Checked for every
                text file, exempt suffixes included.

EXCLUDE_PREFIXES lists path prefixes that are never checked (vendored
third-party code, which carries its own license headers). Nothing else is
excluded.

A line that contains the inline marker "repo-check: allow" is skipped for the
local-path and secret checks on that line only. The header check never honors
the marker. Use the marker only for intentional literals, such as tests.

Findings print as ``path:line: [rule] message`` with forward slashes, sorted
by path and line. Matched text is never printed.
"""

import argparse
import codecs
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, NamedTuple, Optional, Set, Tuple

# Vendored third-party code and submodules carry their own license headers.
EXCLUDE_PREFIXES: Tuple[str, ...] = ("external/",)

DEFAULT_ROOT = Path(__file__).resolve().parent.parent
GIT_LIST_ARGS = ("ls-files", "-z", "--cached", "--others", "--exclude-standard")
BINARY_SNIFF_BYTES = 8000
HEADER_WINDOW = 5
HEADER_BASENAME = "CMakeLists.txt"
HEADER_SUFFIXES = frozenset(
    {".cpp", ".h", ".hpp", ".c", ".slang", ".slangh", ".ps1", ".cmake",
     ".in", ".yml", ".yaml", ".mjs", ".js", ".py"}
)
INLINE_MARKER = "repo-check: " + "allow"

COPYRIGHT_RE = re.compile(r"^(//|#)\s*Copyright (19|20)\d\d\s+Kashyap Rajpal\s*$")
SPDX_RE = re.compile(r"^(//|#)\s*SPDX-License-Identifier: Apache-2\.0\s*$")

DRIVE_PATH_RE = re.compile(r"(?<![A-Za-z0-9_/])[A-Za-z]:[\\/]{1,2}(?![\\/])")
POSIX_HOME_RE = re.compile(r"(?<![\w.])/(?:[U]sers|[h]ome)/[A-Za-z0-9_.-]+")

# Patterns are split or written so that this file does not match its own rules.
SECRET_RULES: Tuple[Tuple["re.Pattern[str]", str], ...] = (
    (re.compile("AK" + "IA[0-9A-Z]{16}"), "AWS access key id"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"), "GitHub token"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{22,}"), "GitHub fine-grained token"),
    (re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"), "Slack token"),
    (re.compile("AI" + "za" + r"[0-9A-Za-z_-]{35}"), "Google API key"),
    (re.compile(r"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}"), "secret key token"),
    (re.compile(r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----"), "PEM private key"),
    (re.compile(
        r'(?i)(?:password|passwd|secret|api[_-]?key|access[_-]?token)'
        r'\s*[:=]\s*["\'][^"\'\s]{12,}["\']'),
     "hard-coded credential literal"),
)

SELF_TEST_HEADER = "// Copyright 2026 Kashyap Rajpal\n// SPDX-License-Identifier: Apache-2.0\n"


class GitError(Exception):
    """Raised when the file list cannot be read from git."""


class Finding(NamedTuple):
    path: str
    line: int
    rule: str
    message: str


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check repository files for copyright headers, local paths and secrets.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="repository root (default: parent of the tools directory)")
    parser.add_argument("--self-test", action="store_true",
                        help="run the built-in fixtures against the checks and exit")
    return parser.parse_args(argv)


def list_files(root: Path) -> List[str]:
    """Return every tracked or untracked, non-ignored path, relative to root."""
    cmd = ["git", "-C", str(root)] + list(GIT_LIST_ARGS)
    try:
        proc = subprocess.run(cmd, capture_output=True, check=False)
    except OSError as exc:
        raise GitError(f"cannot run git: {exc}") from exc
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", errors="replace").strip()
        raise GitError(f"git ls-files failed (exit {proc.returncode}): {detail}")
    return [os.fsdecode(raw) for raw in proc.stdout.split(b"\0") if raw]


def is_excluded(rel: str) -> bool:
    return rel.startswith(EXCLUDE_PREFIXES)


def is_regular_file(root: Path, rel: str) -> bool:
    """True for a real file; false for symlinks, submodule gitlinks and directories."""
    full = root / rel
    return not full.is_symlink() and full.is_file()


def read_bytes(root: Path, rel: str) -> Optional[bytes]:
    """Read a file once. Return None for an unreadable file."""
    try:
        return (root / rel).read_bytes()
    except OSError as exc:
        print(f"warning: cannot read {rel}: {exc}", file=sys.stderr)
        return None


def decode_text(data: bytes) -> Optional[List[str]]:
    """Return the lines of a text file, or None for a binary one.

    The UTF-16 byte order mark is checked first: UTF-16 text has NUL bytes
    that would otherwise look binary.
    """
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return data.decode("utf-16", errors="replace").splitlines()
    if b"\0" in data[:BINARY_SNIFF_BYTES]:
        return None
    if data.startswith(codecs.BOM_UTF8):
        data = data[len(codecs.BOM_UTF8):]
    return data.decode("utf-8", errors="replace").splitlines()


def needs_header(rel: str) -> bool:
    path = Path(rel)
    return path.name == HEADER_BASENAME or path.suffix.lower() in HEADER_SUFFIXES


def check_header(rel: str, lines: List[str]) -> List[Finding]:
    if not needs_header(rel):
        return []
    window = lines[:HEADER_WINDOW]
    findings: List[Finding] = []
    if not any(COPYRIGHT_RE.match(line) for line in window):
        findings.append(Finding(rel, 1, "header", "missing copyright line"))
    if not any(SPDX_RE.match(line) for line in window):
        findings.append(Finding(rel, 1, "header", "missing SPDX-License-Identifier line"))
    return findings


def check_line_rules(rel: str, lines: List[str]) -> List[Finding]:
    """Run the local-path and secret checks on every line, honoring the inline marker."""
    findings: List[Finding] = []
    for number, line in enumerate(lines, start=1):
        if INLINE_MARKER in line:
            continue
        if DRIVE_PATH_RE.search(line):
            findings.append(Finding(rel, number, "local-path", "Windows drive-letter path"))
        if POSIX_HOME_RE.search(line):
            findings.append(Finding(rel, number, "local-path", "POSIX home-directory path"))
        for pattern, kind in SECRET_RULES:
            if pattern.search(line):
                findings.append(Finding(rel, number, "secret", kind))
    return findings


def check_content(rel: str, data: bytes) -> Optional[List[Finding]]:
    """Run every check on one file's bytes. Return None for a binary file."""
    lines = decode_text(data)
    if lines is None:
        return None
    return check_header(rel, lines) + check_line_rules(rel, lines)


def _describe(value: Optional[Set[str]]) -> str:
    if value is None:
        return "skipped"
    if not value:
        return "no findings"
    return ", ".join(sorted(value))


def run_self_test() -> int:
    """Run the built-in fixtures through check_content. Reads no files and no git."""
    header = SELF_TEST_HEADER
    code = "int main() {\n    return 0;\n}\n"
    drive = "C" + ":\\Users\\x"
    posix_home = "/Us" + "ers/name/project"
    marker_line = " // " + INLINE_MARKER
    secret_fixtures = [
        ("AWS access key id", "AK" + "IA" + "A" * 16),
        ("GitHub token", "gh" + "p_" + "a" * 36),
        ("GitHub fine-grained token", "github_" + "pat_" + "a" * 22),
        ("Slack token", "xo" + "xb-" + "1" * 12),
        ("Google API key", "AI" + "za" + "a" * 35),
        ("secret key token", "s" + "k-" + "a" * 24),
        ("PEM private key", "-----BEGIN " + "RSA PRIVATE KEY-----"),
        ("hard-coded credential literal", "pass" + 'word = "' + "a" * 12 + '"'),
    ]

    def text(body: str) -> bytes:
        return (header + body).encode("utf-8")

    # (name, path, bytes, expected rule set; None means skipped as binary)
    cases: List[Tuple[str, str, bytes, Optional[Set[str]]]] = [
        ("header: missing", "x.cpp", code.encode(), {"header"}),
        ("header: present", "x.cpp", text(code), set()),
        ("header: after UTF-8 BOM", "x.cpp", codecs.BOM_UTF8 + text(code), set()),
        ("header: CRLF line endings", "x.cpp", text(code).replace(b"\n", b"\r\n"), set()),
        ("header: after shebang in .py", "x.py",
         ("#!/usr/bin/env python3\n# Copyright 2026 Kashyap Rajpal\n"
          "# SPDX-License-Identifier: Apache-2.0\n" + code).encode(), set()),
        ("header: after pragma in .h", "x.h", ("#pragma once\n" + header + code).encode(), set()),
        ("header: wrong SPDX id", "x.cpp",
         ("// Copyright 2026 Kashyap Rajpal\n// SPDX-License-Identifier: MIT\n" + code).encode(),
         {"header"}),
        ("header: markdown exempt", "README.md", b"# Title\n\nNotes.\n", set()),
        ("header: tools/package txt exempt", "tools/package/notes.txt", b"Plain notes.\n", set()),

        ("local-path: backslash drive", "x.cpp", text(drive + "\n"), {"local-path"}),
        ("local-path: forward-slash drive", "x.cpp", text("D" + ":/GitHub/x\n"), {"local-path"}),
        ("local-path: doubled backslash", "x.cpp",
         text("C" + ":\\\\Users\\\\x\n"), {"local-path"}),
        ("local-path: POSIX /Users", "x.cpp", text(posix_home + "\n"), {"local-path"}),
        ("local-path: POSIX /home", "x.cpp",
         text("/ho" + "me/name/project\n"), {"local-path"}),
        ("local-path: %USERPROFILE% passes", "x.cpp", text("%USERPROFILE%\\x\n"), set()),
        ("local-path: $env:USERPROFILE passes", "x.cpp", text("$env:USERPROFILE\n"), set()),
        ("local-path: URL with drive-like segment passes", "x.cpp",
         text("see https://example.com/C:/x\n"), set()),

        ("marker: skips its own line", "x.cpp", text(drive + marker_line + "\n"), set()),
        ("marker: per line only", "x.cpp",
         text(drive + marker_line + "\n" + drive + "\n"), {"local-path"}),
        ("marker: secret on marker line passes", "x.cpp",
         text("pass" + 'word = "' + "a" * 12 + '"' + marker_line + "\n"), set()),

        ("UTF-16 LE: clean", "x.cpp",
         codecs.BOM_UTF16_LE + (header + code).encode("utf-16-le"), set()),
        ("UTF-16 BE: clean", "x.cpp",
         codecs.BOM_UTF16_BE + (header + code).encode("utf-16-be"), set()),
        ("UTF-16 LE: missing header", "x.cpp",
         codecs.BOM_UTF16_LE + code.encode("utf-16-le"), {"header"}),
        ("UTF-16 BE: missing header", "x.cpp",
         codecs.BOM_UTF16_BE + code.encode("utf-16-be"), {"header"}),
        ("UTF-16 LE: drive path", "x.cpp",
         codecs.BOM_UTF16_LE + (header + drive + "\n").encode("utf-16-le"), {"local-path"}),
        ("UTF-16 BE: drive path", "x.cpp",
         codecs.BOM_UTF16_BE + (header + drive + "\n").encode("utf-16-be"), {"local-path"}),

        ("binary: NUL without BOM skipped", "x.bin",
         b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + drive.encode(), None),
        ("clean file", "x.cpp", text(code), set()),
    ]
    for kind, fixture in secret_fixtures:
        cases.append((f"secret: {kind}", "x.cpp", text(fixture + "\n"), {"secret"}))

    failures = 0
    if len(secret_fixtures) != len(SECRET_RULES):
        failures += 1
        print(f"FAIL: secret fixtures: expected {len(SECRET_RULES)} got {len(secret_fixtures)}")

    for name, rel, data, expected in cases:
        findings = check_content(rel, data)
        got = None if findings is None else {f.rule for f in findings}
        if got != expected:
            failures += 1
            print(f"FAIL: {name}: expected {_describe(expected)} got {_describe(got)}")
    print(f"self-test: {len(cases)} cases, {failures} failed")
    return 1 if failures else 0


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if args.self_test:
        return run_self_test()
    root = args.root.resolve()
    try:
        rel_paths = list_files(root)
    except GitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    findings: List[Finding] = []
    checked = 0
    for rel in rel_paths:
        if is_excluded(rel) or not is_regular_file(root, rel):
            continue
        data = read_bytes(root, rel)
        if data is None:
            continue
        results = check_content(rel, data)
        if results is None:
            continue
        checked += 1
        findings.extend(results)

    findings.sort(key=lambda f: (f.path, f.line))
    for finding in findings:
        print(f"{finding.path}:{finding.line}: [{finding.rule}] {finding.message}")
    print(f"checked {checked} files, {len(findings)} findings")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
