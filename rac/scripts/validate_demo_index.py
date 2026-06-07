from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAC = ROOT / "rac"
DEMO_INDEX = RAC / "DEMO_INDEX.md"

REQUIRED_SECTIONS = [
    "# Structured Reasoning Scaffold Demo Index",
    "## 30-Second Summary",
    "## Current Implementation Status",
    "## How To Run",
    "## Demo Cases",
    "## Recommended Review Order",
    "## What The Grounded Reports Show",
    "## Why This Is Different From Ordinary RAG",
    "## What This Module Does Not Claim",
    "## Current Limitations",
    "## Next Possible Steps",
]

REQUIRED_COMMANDS = [
    "python3 rac/scripts/run_grounded_pipeline.py --all-eval",
    "python3 rac/scripts/validate_grounded_quality_gate.py",
]

REQUIRED_OUTPUT_FILES = [
    "rac/outputs/grounded_rac_store_a_attribution_001.md",
    "rac/outputs/grounded_rac_cross_store_comparability_001.md",
    "rac/outputs/grounded_rac_promotion_strategy_001.md",
    "rac/outputs/grounded_rac_system_design_001.md",
    "rac/outputs/grounded_quality_summary.md",
]

REQUIRED_CODE_FILES = [
    "rac/src/mock_pipeline.py",
    "rac/src/local_evidence_resolver.py",
    "rac/src/grounded_pipeline.py",
    "rac/scripts/validate_grounded_quality_gate.py",
]

REQUIRED_BOUNDARY_PHRASES = [
    "does not call an LLM",
    "This module does not claim",
    "Future work",
    "Not implemented",
    "causal proof",
    "confidence and limitations",
]

FORBIDDEN_OVERCLAIMS = [
    "live Meituan backend access is implemented",
    "true autonomous world model is implemented",
    "updates neural-network weights",
    "completed pairwise comparability gate is implemented",
    "proves causality",
]


def fail(message: str) -> None:
    raise SystemExit(f"[RAC demo index validation failed] {message}")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def main() -> None:
    if not DEMO_INDEX.exists():
        fail("rac/DEMO_INDEX.md does not exist")

    text = DEMO_INDEX.read_text(encoding="utf-8")
    normalized = normalize(text)

    for section in REQUIRED_SECTIONS:
        if section not in text:
            fail(f"Missing required section: {section}")

    for command in REQUIRED_COMMANDS:
        if command not in text:
            fail(f"Missing required command: {command}")

    for relative_path in REQUIRED_OUTPUT_FILES + REQUIRED_CODE_FILES:
        path = ROOT / relative_path
        if not path.exists():
            fail(f"Referenced file does not exist: {relative_path}")

        if relative_path not in text:
            fail(f"Referenced file missing from DEMO_INDEX.md: {relative_path}")

    for phrase in REQUIRED_BOUNDARY_PHRASES:
        if phrase not in text:
            fail(f"Missing boundary phrase: {phrase}")

    for phrase in FORBIDDEN_OVERCLAIMS:
        if normalize(phrase) in normalized:
            fail(f"Forbidden overclaim found: {phrase}")

    print("[OK] RAC demo index validation passed")
    print(f"[OK] Demo index: {DEMO_INDEX}")
    print(f"[OK] Referenced output files: {len(REQUIRED_OUTPUT_FILES)}")
    print(f"[OK] Referenced code files: {len(REQUIRED_CODE_FILES)}")


if __name__ == "__main__":
    main()
