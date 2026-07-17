#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

DOCS = [
    ROOT / "README.md",
    ROOT / "rac" / "README.md",
    ROOT / "rac" / "DEMO_INDEX.md",
]

SUMMARY_PATH = (
    ROOT
    / "rac"
    / "outputs"
    / "grounded_quality_summary.json"
)

REQUIRED_BOUNDARY_PHRASES = [
    "source-aware",
    "boundary evidence",
    "pairwise comparability remains future work",
    "pairwise comparability remains future work",
]

REQUIRED_PATHS = [
    "retail_ops/outputs/demo2_cross_store_comparability_output.csv",
    "retail_ops/COMPARABILITY_GATE_V0.md",
    "rac/outputs/grounded_rac_cross_store_comparability_001.md",
    "rac/outputs/grounded_quality_summary.md",
]

FORBIDDEN_STALE_PHRASES = [
    "Total grounded packets: 29",
    "Keyword matched packets: 27",
    "Fallback packets: 1",
]

FORBIDDEN_OVERCLAIMS = [
    "pairwise comparability gate is implemented",
    "completed pairwise comparability engine",
    "proves cross-store comparability",
    "proves causality",
    "live Meituan backend access is implemented",
]


def fail(message: str) -> None:
    raise SystemExit(
        f"[RAC doc alignment validation failed] {message}"
    )


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def load_totals(
    summary: dict[str, Any],
) -> dict[str, int]:
    results = summary.get("results")

    if not isinstance(results, list):
        fail("grounded quality summary has no results list")

    metric_names = [
        "total_packets",
        "keyword_matched_count",
        "boundary_matched_count",
        "fallback_count",
        "source_missing_count",
    ]

    totals = {name: 0 for name in metric_names}

    for result in results:
        if not isinstance(result, dict):
            fail("grounded quality result is not an object")

        metrics = result.get("metrics")
        if not isinstance(metrics, dict):
            fail("grounded quality result has no metrics object")

        for name in metric_names:
            value = metrics.get(name)
            if not isinstance(value, int):
                fail(
                    f"grounded quality metric is not an integer: "
                    f"{name}={value!r}"
                )
            totals[name] += value

    return totals


def main() -> None:
    for doc in DOCS:
        if not doc.exists():
            fail(f"Missing doc: {doc}")

    if not SUMMARY_PATH.exists():
        fail(
            "Missing grounded quality summary: "
            f"{SUMMARY_PATH}"
        )

    combined = "\n\n".join(
        doc.read_text(encoding="utf-8")
        for doc in DOCS
    )
    normalized = normalize(combined)

    summary = json.loads(
        SUMMARY_PATH.read_text(encoding="utf-8")
    )
    totals = load_totals(summary)

    required_metric_phrases = [
        (
            "Total grounded packets: "
            f"{totals['total_packets']}"
        ),
        (
            "Keyword matched packets: "
            f"{totals['keyword_matched_count']}"
        ),
        (
            "Boundary matched packets: "
            f"{totals['boundary_matched_count']}"
        ),
        (
            "Fallback packets: "
            f"{totals['fallback_count']}"
        ),
        (
            "Missing source files: "
            f"{totals['source_missing_count']}"
        ),
    ]

    for phrase in required_metric_phrases:
        if phrase not in combined:
            fail(f"Missing current metric phrase: {phrase}")

    for phrase in REQUIRED_BOUNDARY_PHRASES:
        if normalize(phrase) not in normalized:
            fail(f"Missing boundary phrase: {phrase}")

    for relative_path in REQUIRED_PATHS:
        if relative_path not in combined:
            fail(
                f"Missing required path reference: "
                f"{relative_path}"
            )

        if not (ROOT / relative_path).exists():
            fail(
                f"Referenced path does not exist: "
                f"{relative_path}"
            )

    for phrase in FORBIDDEN_STALE_PHRASES:
        if phrase in combined:
            fail(f"Outdated phrase still present: {phrase}")

    for phrase in FORBIDDEN_OVERCLAIMS:
        if normalize(phrase) in normalized:
            fail(f"Forbidden overclaim found: {phrase}")

    print("[OK] RAC doc alignment validation passed")
    print(f"[OK] Docs checked: {len(DOCS)}")
    print(
        "[OK] Docs aligned with "
        f"{totals['total_packets']} total / "
        f"{totals['keyword_matched_count']} keyword / "
        f"{totals['boundary_matched_count']} boundary / "
        f"{totals['fallback_count']} fallback packets"
    )


if __name__ == "__main__":
    main()
