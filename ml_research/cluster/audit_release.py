#!/usr/bin/env python3
"""Check the tree for infrastructure detail that should not ship.

This runs over the released code and fails on anything that looks like it was
carried over from a private cluster: object-store URIs, cloud instance types,
credentials, internal hostnames, absolute paths from a managed training service.

The patterns describe *classes* of leak rather than any specific bucket or host,
so this file is safe to read and safe to publish. Run it before any release:

    python ml_research/cluster/audit_release.py

Exit status is non-zero when anything is found, so it drops straight into CI.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

# (pattern, what it is, why it matters)
PATTERNS: list[tuple[str, str, str]] = [
    (r"s3://[A-Za-z0-9][A-Za-z0-9.\-]{2,}", "object-store URI",
     "storage roots belong in WMRL_STORE, not in source"),
    (r"\b(?:gs|az|abfss)://[A-Za-z0-9][A-Za-z0-9.\-]{2,}", "object-store URI",
     "storage roots belong in WMRL_STORE, not in source"),
    (r"\bp[45][a-z]*\.\d+xlarge\b", "cloud instance type",
     "the design is roles per node, not one vendor's SKU"),
    (r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b", "access key id", "credential"),
    (r"aws_secret_access_key\s*=", "secret key", "credential"),
    (r"\barn:aws[a-z-]*:", "cloud resource ARN", "names a private account resource"),
    (r"\b\d{12}\b(?=[:/])", "account id", "names a private account"),
    (r"/opt/ml/(?:input|model|checkpoints)\b", "managed-service path",
     "hard-codes one scheduler's filesystem layout"),
    (r"\bip-\d{1,3}-\d{1,3}-\d{1,3}-\d{1,3}\b", "private hostname", "names a private host"),
    (r"\b10\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "private address", "names a private host"),
    (r"\b(?:172\.(?:1[6-9]|2\d|3[01])|192\.168)\.\d{1,3}\.\d{1,3}\b", "private address",
     "names a private host"),
    (r"[A-Za-z0-9._%+-]+@(?!example\.)[A-Za-z0-9.-]+\.(?:com|net|org)\b", "email address",
     "personal contact details"),
]

SKIP_DIRS = {".git", "__pycache__", "node_modules", "docs", ".venv", "venv"}
SUFFIXES = {".py", ".sh", ".md", ".json", ".yaml", ".yml", ".txt", ".toml", ".cfg"}
ALLOW = re.compile(r"audit-allow")


def walk(roots):
    for root in roots:
        p = pathlib.Path(root)
        if p.is_file():
            yield p
            continue
        for f in p.rglob("*"):
            if f.is_file() and f.suffix in SUFFIXES and not any(d in f.parts for d in SKIP_DIRS):
                yield f


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("paths", nargs="*", default=None,
                    help="defaults to the whole repository")
    args = ap.parse_args()
    roots = args.paths or ["."]

    findings = 0
    self_path = pathlib.Path(__file__).resolve()

    for f in walk(roots):
        if f.resolve() == self_path:
            continue  # this file is a list of patterns, not a list of secrets
        try:
            text = f.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            if ALLOW.search(line):
                continue
            for pat, what, why in PATTERNS:
                m = re.search(pat, line)
                if m:
                    print(f"{f}:{lineno}: {what} — {why}")
                    print(f"    {line.strip()[:150]}")
                    findings += 1
                    break

    if findings:
        print(f"\n{findings} finding(s). Mark a deliberate one with an `audit-allow` comment.")
        return 1
    print("clean: no infrastructure detail found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
