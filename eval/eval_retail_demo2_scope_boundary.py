"""Demo 2 scope-boundary evaluation.

This eval asserts that Demo 2 remains a row-level same-period diagnostic.
It also asserts the absence of the future pairwise comparability-gate schema.
"""

from __future__ import annotations

import csv
from pathlib import Path

OUTPUT_PATH = Path("retail_ops/outputs/demo2_cross_store_comparability_output.csv")
RESULTS_PATH = Path("eval/results/eval_retail_demo2_scope_boundary_result.txt")

REQUIRED_FIELDS = [
    "store_id",
    "period_month",
    "region_type",
    "store_type",
    "comparison_scope_flag",
    "comparison_limit_notes",
]

EXPECTED_STORE_IDS = {"B", "C", "D", "E", "F"}


def read_rows() -> tuple[list[str], list[dict[str, str]]]:
    if not OUTPUT_PATH.exists():
        raise FileNotFoundError(f"Missing Demo2 output: {OUTPUT_PATH}")

    with OUTPUT_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)


def main() -> int:
    checks: list[tuple[str, bool, str]] = []

    fieldnames, rows = read_rows()

    checks.append(
        (
            "demo2_output_exists_and_has_rows",
            len(rows) > 0,
            "Demo2 output should exist and contain rows.",
        )
    )

    checks.append(
        (
            "demo2_required_fields_exist",
            all(field in fieldnames for field in REQUIRED_FIELDS),
            f"Demo2 output should contain required fields: {REQUIRED_FIELDS}",
        )
    )

    store_ids = {row.get("store_id", "") for row in rows}

    checks.append(
        (
            "demo2_expected_store_ids_exist",
            EXPECTED_STORE_IDS.issubset(store_ids),
            f"Demo2 output should include store IDs {sorted(EXPECTED_STORE_IDS)}.",
        )
    )

    scope_values = {row.get("comparison_scope_flag", "") for row in rows}

    checks.append(
        (
            "demo2_scope_flag_is_present",
            "" not in scope_values,
            "Every Demo2 row should have comparison_scope_flag.",
        )
    )

    checks.append(
        (
            "future_pairwise_gate_schema_is_absent_from_demo2_output",
            "comparison_question_type" not in fieldnames
            and "pairwise_comparison_decision" not in fieldnames
            and "pairwise_limit_notes" not in fieldnames,
            "Demo2 should remain a row-level same-period diagnostic output, not a pairwise gate output.",
        )
    )

    passed = 0
    failed = 0
    lines: list[str] = []

    for name, ok, detail in checks:
        if ok:
            passed += 1
            lines.append(f"PASS {name}")
        else:
            failed += 1
            lines.append(f"FAIL {name}: {detail}")

    summary = [
        f"Retail Demo2 scope-boundary eval result: {passed}/{len(checks)} passed",
        f"Passed: {passed}",
        f"Failed: {failed}",
        "",
        *lines,
        "",
    ]

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text("\n".join(summary), encoding="utf-8")

    print("\n".join(summary))

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
