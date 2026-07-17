#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "PROJECT_SUMMARY_FOR_ADMISSIONS.md"

START_MARKER = "## Factor-Aware Grounded Review Layer"
END_MARKER = "## Region and Market-Context Boundary"

REQUIRED_PHRASES = [
    "## Factor-Aware Grounded Review Layer",
    "deterministic source-aware review layer",
    "decomposes an operating question into relevant factors",
    "routes each factor to local evidence or boundary evidence",
    "generates competing hypotheses",
    "applies critique and fact checks",
    (
        "produces a grounded report with confidence, "
        "limitations, source paths, and local evidence snippets"
    ),
    (
        "helps prevent a grounded answer from hiding "
        "missing evidence behind a fluent conclusion"
    ),
]

REQUIRED_EXISTING_PATHS = [
    "rac/DEMO_INDEX.md",
    "rac/outputs/grounded_rac_store_a_attribution_001.md",
    (
        "rac/outputs/"
        "grounded_rac_cross_store_comparability_001.md"
    ),
    "rac/src/grounded_pipeline.py",
]

FORBIDDEN_OVERCLAIMS = [
    "implemented live Meituan backend access",
    "implemented Qdrant retrieval",
    "implemented LLM calls",
    "implemented pairwise comparability gate",
    "proves causality",
    "true Bayesian posterior",
    "implemented autonomous world model",
    "updates neural-network weights",
]


def fail(message: str) -> None:
    raise SystemExit(
        f"[PROJECT_SUMMARY RAC sync validation failed] "
        f"{message}"
    )


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_section(text: str) -> str:
    start = text.find(START_MARKER)
    end = text.find(END_MARKER)

    if start == -1:
        fail(f"Missing section: {START_MARKER}")

    if end == -1:
        fail(f"Missing following section: {END_MARKER}")

    if end <= start:
        fail("RAC section markers are in the wrong order")

    return text[start:end]


def main() -> None:
    if not SUMMARY.exists():
        fail("PROJECT_SUMMARY_FOR_ADMISSIONS.md does not exist")

    text = SUMMARY.read_text(encoding="utf-8")
    section = extract_section(text)

    for phrase in REQUIRED_PHRASES:
        if phrase not in section:
            fail(f"Missing required phrase: {phrase}")

    for relative_path in REQUIRED_EXISTING_PATHS:
        if relative_path not in section:
            fail(
                "RAC section does not reference expected path: "
                f"{relative_path}"
            )

        if not (ROOT / relative_path).exists():
            fail(
                f"Referenced path does not exist: {relative_path}"
            )

    normalized_text = normalize(text)

    required_boundary = (
        "does not claim a completed pairwise "
        "comparability gate"
    )

    if required_boundary not in normalized_text:
        fail(
            "Admissions summary is missing the current "
            "pairwise-comparability boundary"
        )

    for phrase in FORBIDDEN_OVERCLAIMS:
        if normalize(phrase) in normalized_text:
            fail(f"Forbidden overclaim found: {phrase}")

    print("[OK] PROJECT_SUMMARY RAC sync validation passed")
    print(f"[OK] Summary file: {SUMMARY}")
    print(
        f"[OK] Referenced paths: "
        f"{len(REQUIRED_EXISTING_PATHS)}"
    )


if __name__ == "__main__":
    main()
