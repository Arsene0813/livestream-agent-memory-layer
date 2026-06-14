from __future__ import annotations

from pathlib import Path

SPEC_PATH = Path("retail_ops/COMPARABILITY_GATE_V0.md")
RESULT_PATH = Path("eval/retail_decision_support_eval_results/eval_future_comparability_gate_contract_result.txt")

REQUIRED_SPEC_TERMS = [
    "reference_store_id",
    "candidate_store_id",
    "comparison_question_type",
    "comparison_decision",
    "comparable",
    "comparable_with_limits",
    "not_comparable",
    "insufficient_evidence",
    "supporting_fields",
    "blocking_or_limiting_factors",
    "allowed_interpretation",
    "unsupported_interpretation",
]


def write_result(lines: list[str]) -> None:
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if not SPEC_PATH.exists():
        lines = [
            "Future comparability gate contract stub FAILED.",
            f"Missing spec file: {SPEC_PATH}",
        ]
        write_result(lines)
        print("\n".join(lines))
        return 1

    spec = SPEC_PATH.read_text(encoding="utf-8")
    missing = [term for term in REQUIRED_SPEC_TERMS if term not in spec]

    if missing:
        lines = [
            "Future comparability gate contract stub FAILED.",
            f"Missing terms: {missing}",
        ]
        write_result(lines)
        print("\n".join(lines))
        return 1

    lines = [
        "[SKIP] Future comparability gate is documented but not implemented.",
        "[PASS] Planned input triple, output enum, and output fields are present in COMPARABILITY_GATE_V0.md.",
        "[PASS] This stub freezes the future contract without claiming an implemented pairwise gate.",
        f"[PASS] Result written to {RESULT_PATH.as_posix()}.",
    ]
    write_result(lines)
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
