from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOP_README = ROOT / "README.md"

START_MARKER = "<!-- RAC_EXTENSION_START -->"
END_MARKER = "<!-- RAC_EXTENSION_END -->"

REQUIRED_PATHS = [
    "rac/DEMO_INDEX.md",
    "rac/outputs/grounded_rac_store_a_attribution_001.md",
    "rac/outputs/grounded_rac_cross_store_comparability_001.md",
    "rac/outputs/grounded_quality_summary.md",
    "rac/scripts/run_grounded_pipeline.py",
    "rac/scripts/validate_grounded_quality_gate.py",
]

REQUIRED_PHRASES = [
    "Structured Reasoning Scaffold: Factor-Aware Grounded Review",
    "question decomposition",
    "factor expansion",
    "factor weighting",
    "local evidence grounding",
    "competing hypotheses",
    "critique",
    "fact check",
    "confidence and limitations",
    "deterministic only",
    "no LLM calls",
    "no Qdrant retrieval",
    "no live Meituan backend access",
    "no completed pairwise comparability gate",
    "no causal proof from observational store metrics",
]

FORBIDDEN_OVERCLAIMS = [
    "RAC proves causality",
    "RAC has live Meituan backend access",
    "RAC uses Qdrant retrieval",
    "RAC calls an LLM",
    "RAC implements pairwise comparability gate",
    "RAC updates neural-network weights",
]


def fail(message: str) -> None:
    raise SystemExit(f"[Top README RAC pointer validation failed] {message}")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def extract_section(text: str) -> str:
    start = text.find(START_MARKER)
    end = text.find(END_MARKER)

    if start == -1 or end == -1:
        fail("structured reasoning scaffold markers not found in top-level README.md")

    if end <= start:
        fail("RAC extension markers are in the wrong order")

    return text[start:end + len(END_MARKER)]


def main() -> None:
    if not TOP_README.exists():
        fail("Top-level README.md does not exist")

    text = TOP_README.read_text(encoding="utf-8")
    section = extract_section(text)

    for phrase in REQUIRED_PHRASES:
        if phrase not in section:
            fail(f"Missing required phrase in RAC section: {phrase}")

    for relative_path in REQUIRED_PATHS:
        if relative_path not in section:
            fail(f"RAC section does not reference path: {relative_path}")

        if not (ROOT / relative_path).exists():
            fail(f"Referenced path does not exist: {relative_path}")

    normalized_section = normalize(section)

    for phrase in FORBIDDEN_OVERCLAIMS:
        if normalize(phrase) in normalized_section:
            fail(f"Forbidden overclaim found: {phrase}")

    print("[OK] Top README RAC pointer validation passed")
    print(f"[OK] README: {TOP_README}")
    print(f"[OK] Referenced paths: {len(REQUIRED_PATHS)}")


if __name__ == "__main__":
    main()
