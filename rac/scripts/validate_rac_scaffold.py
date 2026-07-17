#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAC = ROOT / "rac"

REQUIRED_FILES = [
    "README.md",
    "src/state_validation.py",
    "schemas/cognition_state.schema.json",
    "schemas/rac_eval_case.schema.json",
    "prompts/question_analyzer.md",
    "prompts/factor_expander.md",
    "prompts/factor_weighter.md",
    "prompts/evidence_router.md",
    "prompts/hypothesis_generator.md",
    "prompts/critic.md",
    "prompts/fact_checker.md",
    "prompts/belief_updater.md",
    "prompts/final_report_writer.md",
    "eval/rac_eval_cases.json",
]

REQUIRED_README_PHRASES = [
    "Factor-Aware Grounded Review (RAC)",
    "shared review-state contract",
    "factor expansion",
    "factor weights",
    "evidence routing",
    "competing hypotheses",
    "critique",
    "Fact Checks",
    "Interpretation Boundary",
]

REQUIRED_EVAL_KEYS = [
    "case_id",
    "question",
    "question_type",
    "must_include_factors",
    "must_not_claim",
    "expected_confidence",
]


def fail(message: str) -> None:
    raise SystemExit(f"[RAC scaffold validation failed] {message}")


def main() -> None:
    if not RAC.exists():
        fail("rac/ directory does not exist")

    missing = [path for path in REQUIRED_FILES if not (RAC / path).exists()]
    if missing:
        fail(f"Missing files: {missing}")

    readme = (RAC / "README.md").read_text(encoding="utf-8")
    missing_phrases = [phrase for phrase in REQUIRED_README_PHRASES if phrase not in readme]
    if missing_phrases:
        fail(f"README missing required phrases: {missing_phrases}")

    for schema_path in [
        RAC / "schemas/cognition_state.schema.json",
        RAC / "schemas/rac_eval_case.schema.json",
    ]:
        try:
            json.loads(schema_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            fail(f"Invalid JSON schema {schema_path}: {exc}")

    eval_path = RAC / "eval/rac_eval_cases.json"
    try:
        eval_cases = json.loads(eval_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        fail(f"Invalid eval JSON: {exc}")

    if not isinstance(eval_cases, list):
        fail("Eval cases must be a list")

    if len(eval_cases) < 3:
        fail("Expected at least 3 RAC eval cases")

    for idx, case in enumerate(eval_cases):
        for key in REQUIRED_EVAL_KEYS:
            if key not in case:
                fail(f"Eval case index {idx} missing key: {key}")

        if not case["must_include_factors"]:
            fail(f"Eval case {case['case_id']} has empty must_include_factors")

        if not case["must_not_claim"]:
            fail(f"Eval case {case['case_id']} has empty must_not_claim")

    print("[OK] RAC scaffold validation passed")
    print(f"[OK] Required files: {len(REQUIRED_FILES)}")
    print(f"[OK] Eval cases: {len(eval_cases)}")


if __name__ == "__main__":
    main()
