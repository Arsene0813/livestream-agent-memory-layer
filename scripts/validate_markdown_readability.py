#!/usr/bin/env python3
"""Validate reviewer-facing Markdown readability.

This script catches two problems:

1. Reviewer-facing Markdown compressed into very long physical lines.
2. Overclaiming language that presents future or unsupported scope as current implementation.

It intentionally allows boundary sentences such as:
- "not a completed comparability engine"
- "not a global store ranking"
- "not a universal comparability score"

Those sentences are correct scope-control language for this project.
"""

from pathlib import Path
import re
import sys

CORE_FILES = [
    "README.md",
    "PROJECT_SUMMARY_FOR_ADMISSIONS.md",
    "retail_ops/README.md",
]

# Only block positive overclaims.
# Do not block sentences that explicitly say the project is NOT these things.
FORBIDDEN_POSITIVE_PATTERNS = [
    r"\b(is|as|implements|implemented|provides|delivers|builds|built)\s+(a\s+)?completed comparability engine\b",
    r"\b(is|as|implements|implemented|provides|delivers|builds|built)\s+(a\s+)?fully comparable stores?\b",
    r"\b(is|as|implements|implemented|provides|delivers|builds|built)\s+(a\s+)?production recommendation system\b",
    r"\b(is|as|implements|implemented|provides|delivers|builds|built)\s+(an?\s+)?autonomous cognition\b",
    r"\b(is|as|implements|implemented|provides|delivers|builds|built)\s+(a\s+)?Bayesian belief update\b",
    r"\b(is|as|implements|implemented|provides|delivers|builds|built)\s+(a\s+)?global store ranking\b",
    r"\b(is|as|implements|implemented|provides|delivers|builds|built)\s+(a\s+)?universal comparability score\b",
    r"\b(activity_cost_ratio_pct|activity cost ratio)\s+(is|as|means|represents|equals)\s+(an?\s+)?activity ROI\b",
]

BOUNDARY_ALLOW_PREFIXES = [
    "not ",
    "not a ",
    "not an ",
    "not as ",
    "no ",
    "should not ",
    "must not ",
    "does not ",
    "do not ",
    "is not ",
    "are not ",
    "isn't ",
    "aren't ",
]

errors = []

for file_name in CORE_FILES:
    path = Path(file_name)

    if not path.exists():
        errors.append(f"[MISSING] {file_name}")
        continue

    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    for line_no, line in enumerate(lines, start=1):
        stripped = line.strip()
        lowered = stripped.lower()

        if len(line) > 1500:
            errors.append(
                f"[LONG LINE] {file_name}:{line_no} has {len(line)} characters."
            )

        if line.count("## ") >= 2:
            errors.append(
                f"[COMPRESSED HEADING] {file_name}:{line_no} contains multiple headings on one line."
            )

        if line.count("|---") >= 2 and line.count("|") > 8:
            errors.append(
                f"[COMPRESSED TABLE] {file_name}:{line_no} looks like multiple table rows collapsed into one line."
            )

        # Allow explicit boundary-control lines.
        if any(prefix in lowered for prefix in BOUNDARY_ALLOW_PREFIXES):
            continue

        for pattern in FORBIDDEN_POSITIVE_PATTERNS:
            if re.search(pattern, stripped, flags=re.IGNORECASE):
                errors.append(
                    f"[FORBIDDEN POSITIVE CLAIM] {file_name}:{line_no} matches pattern: {pattern}"
                )

if errors:
    print("[FAIL] Markdown readability validation failed")
    for error in errors:
        print(error)
    sys.exit(1)

print("[OK] Markdown readability validation passed")
