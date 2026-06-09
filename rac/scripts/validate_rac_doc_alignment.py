from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DOCS = [
    ROOT / "README.md",
    ROOT / "rac" / "README.md",
    ROOT / "rac" / "DEMO_INDEX.md",
]

REQUIRED_PHRASES = [
    "Keyword matched packets: 29",
    "Boundary matched packets: 2",
    "Fallback packets: 1",
    "Missing source files: 0",
    "source-aware",
    "boundary-aware",
    "boundary evidence",
    "Pairwise comparability remains future work",
    "no completed pairwise comparability gate",
]

FORBIDDEN_OLD_PHRASES = [
    "Keyword matched packets: 31",
    "Fallback packets: 5",
    "31 keyword",
    "5 fallback",
]

FORBIDDEN_OVERCLAIMS = [
    "pairwise comparability gate is implemented",
    "completed pairwise comparability engine",
    "proves cross-store comparability",
    "proves causality",
    "live Meituan backend access is implemented",
]


def fail(message: str) -> None:
    raise SystemExit(f"[RAC doc alignment validation failed] {message}")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def main() -> None:
    for doc in DOCS:
        if not doc.exists():
            fail(f"Missing doc: {doc}")

    combined = "\n\n".join(doc.read_text(encoding="utf-8") for doc in DOCS)
    normalized_combined = normalize(combined)

    for phrase in REQUIRED_PHRASES:
        if phrase not in combined:
            fail(f"Missing required phrase: {phrase}")

    for phrase in FORBIDDEN_OLD_PHRASES:
        if phrase in combined:
            fail(f"Outdated phrase still present: {phrase}")

    for phrase in FORBIDDEN_OVERCLAIMS:
        if normalize(phrase) in normalized_combined:
            fail(f"Forbidden overclaim found: {phrase}")

    required_paths = [
        "retail_ops/outputs/demo2_cross_store_comparability_output.csv",
        "retail_ops/COMPARABILITY_GATE_V0.md",
        "rac/outputs/grounded_rac_cross_store_comparability_001.md",
        "rac/outputs/grounded_quality_summary.md",
    ]

    for relative_path in required_paths:
        if relative_path not in combined:
            fail(f"Missing required path reference: {relative_path}")

        if not (ROOT / relative_path).exists():
            fail(f"Referenced path does not exist: {relative_path}")

    print("[OK] RAC doc alignment validation passed")
    print(f"[OK] Docs checked: {len(DOCS)}")
    print("[OK] Docs aligned with 29 keyword / 2 boundary / 1 fallback grounding result")


if __name__ == "__main__":
    main()
