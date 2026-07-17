#!/usr/bin/env python3
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TOP_README = ROOT / "README.md"

REQUIRED_PATHS = [
    "PROJECT_SUMMARY_FOR_ADMISSIONS.md",
    "rac/DEMO_INDEX.md",
]

REQUIRED_PHRASES = [
    "Lifecycle-Aware AI Memory Layer for Retail Decision Support",
    "Factor-aware grounded review layer (RAC)",
    "deterministic grounded review",
    "factor expansion",
    "evidence routing",
    "competing hypotheses",
    "critique",
    "fact checking",
    "limitations",
    "future pairwise comparability gate remains question-specific",
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
    raise SystemExit(
        f"[Top README RAC pointer validation failed] "
        f"{message}"
    )


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def main() -> None:
    if not TOP_README.exists():
        fail("Top-level README.md does not exist")

    text = TOP_README.read_text(encoding="utf-8")
    normalized_text = normalize(text)

    for phrase in REQUIRED_PHRASES:
        if normalize(phrase) not in normalized_text:
            fail(f"Missing required phrase: {phrase}")

    for relative_path in REQUIRED_PATHS:
        if relative_path not in text:
            fail(
                f"README does not reference path: {relative_path}"
            )

        if not (ROOT / relative_path).exists():
            fail(
                f"Referenced path does not exist: {relative_path}"
            )

    for phrase in FORBIDDEN_OVERCLAIMS:
        if normalize(phrase) in normalized_text:
            fail(f"Forbidden overclaim found: {phrase}")

    print("[OK] Top README RAC pointer validation passed")
    print(f"[OK] README: {TOP_README}")
    print(f"[OK] Referenced paths: {len(REQUIRED_PATHS)}")


if __name__ == "__main__":
    main()
