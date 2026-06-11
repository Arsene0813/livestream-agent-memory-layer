from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "PROJECT_SUMMARY_FOR_ADMISSIONS.md"

START_MARKER = "<!-- RAC_ADMISSIONS_SUMMARY_START -->"
END_MARKER = "<!-- RAC_ADMISSIONS_SUMMARY_END -->"

REQUIRED_PHRASES = [
    "Recent Extension: Retrieval-Augmented Cognition Layer",
    "deterministic Retrieval-Augmented Cognition layer",
    "decompose the question into relevant operating or system-design factors",
    "assign interpretable factor weights",
    "ground each factor in local project evidence snippets",
    "generate competing hypotheses",
    "apply critique and fact-checking",
    "grounded report with confidence, limitations, source paths, line ranges, and local evidence snippets",
    "grounded quality gate",
    "rac/DEMO_INDEX.md",
    "rac/outputs/grounded_rac_store_a_attribution_001.md",
    "rac/outputs/grounded_rac_cross_store_comparability_001.md",
    "rac/outputs/grounded_quality_summary.md",
    "deterministic only",
    "local evidence grounding only",
    "no LLM calls",
    "no Qdrant or vector retrieval integration",
    "no live Meituan backend access",
    "no completed pairwise comparability gate",
    "no causal proof from observational store metrics",
]

REQUIRED_EXISTING_PATHS = [
    "rac/DEMO_INDEX.md",
    "rac/outputs/grounded_rac_store_a_attribution_001.md",
    "rac/outputs/grounded_rac_cross_store_comparability_001.md",
    "rac/outputs/grounded_quality_summary.md",
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
    raise SystemExit(f"[PROJECT_SUMMARY RAC sync validation failed] {message}")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_section(text: str) -> str:
    start = text.find(START_MARKER)
    end = text.find(END_MARKER)

    if start == -1 or end == -1:
        fail("RAC admissions summary markers not found")

    if end <= start:
        fail("RAC admissions summary markers are in the wrong order")

    return text[start:end + len(END_MARKER)]


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
            fail(f"Section does not reference expected path: {relative_path}")

        if not (ROOT / relative_path).exists():
            fail(f"Referenced path does not exist: {relative_path}")

    normalized_section = normalize(section)

    for phrase in FORBIDDEN_OVERCLAIMS:
        if normalize(phrase) in normalized_section:
            fail(f"Forbidden overclaim found: {phrase}")

    if "## Recent Extension: Retrieval-Augmented Cognition Layer" not in text:
        fail("RAC heading missing from admissions summary")

    implemented_heading = text.find("## Implemented Check Summary")
    rac_heading = text.find("## Recent Extension: Retrieval-Augmented Cognition Layer")

    if implemented_heading == -1:
        fail("Could not find ## Implemented Check Summary")

    if rac_heading == -1:
        fail("Could not find RAC extension heading")

    if rac_heading > implemented_heading:
        fail("RAC extension section should appear before ## Implemented Check Summary")

    print("[OK] PROJECT_SUMMARY RAC sync validation passed")
    print(f"[OK] Summary file: {SUMMARY}")
    print(f"[OK] Referenced paths: {len(REQUIRED_EXISTING_PATHS)}")


if __name__ == "__main__":
    main()
