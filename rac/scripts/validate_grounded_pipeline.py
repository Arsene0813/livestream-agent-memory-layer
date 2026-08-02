from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rac.src.grounded_pipeline import run_grounded_pipeline
from rac.src.store_a_csv_grounding import (
    FACTOR_FIELDS as STORE_A_FACTOR_FIELDS,
    PERIOD_MONTHS as STORE_A_PERIOD_MONTHS,
    SOURCE_PATH as STORE_A_SOURCE_PATH,
)


REQUIRED_REPORT_SECTIONS = [
    "# Grounded RAC Report",
    "## 1. Direct Answer",
    "## 2. Question Type",
    "## 3. Factor Weights",
    "## 4. Local Evidence Grounding",
    "## 5. Competing Hypotheses",
    "## 6. Critic Findings",
    "## 7. Claim and Definition Check",
    "## 8. Final Judgment",
    "## 9. Evidence-Routing Coverage",
    "## 10. What Cannot Be Concluded",
    "## 11. Review-State Update",
]

ALLOWED_GROUNDING_STATUSES = {
    "record_matched",
    "keyword_matched",
    "boundary_matched",
    "source_found_no_keyword_match"
}


def fail(message: str) -> None:
    raise SystemExit(f"[RAC grounded pipeline validation failed] {message}")


def load_eval_cases() -> list[dict]:
    path = ROOT / "rac" / "eval" / "rac_eval_cases.json"
    return json.loads(path.read_text(encoding="utf-8"))


PROMOTION_BOUNDARY_FACTORS = {
    "sku_margin_structure",
    "competitor_context",
}


def validate_promotion_grounded_rows(
    case_id: str,
    rows: list[dict],
) -> None:
    if case_id != "rac_promotion_strategy_001":
        return

    rows_by_factor = {
        row["factor_id"]: row
        for row in rows
    }

    missing = sorted(
        PROMOTION_BOUNDARY_FACTORS
        - set(rows_by_factor)
    )

    if missing:
        fail(
            "Promotion report is missing factors: "
            f"{missing}"
        )

    actual_boundary = {
        factor_id
        for factor_id, row in rows_by_factor.items()
        if row["grounding_status"] == "boundary_matched"
    }

    if actual_boundary != PROMOTION_BOUNDARY_FACTORS:
        fail(
            "Promotion grounded boundary mismatch. "
            f"Expected {sorted(PROMOTION_BOUNDARY_FACTORS)}, "
            f"found {sorted(actual_boundary)}"
        )

    for factor_id in sorted(PROMOTION_BOUNDARY_FACTORS):
        role = rows_by_factor[factor_id].get(
            "grounding_role"
        )

        if role != "boundary_evidence":
            fail(
                f"{factor_id} must use boundary_evidence, "
                f"found {role}"
            )


def validate_store_a_grounded_case(
    case_id: str,
    state: dict,
) -> None:
    rows = state.get(
        "grounded_evidence_rows",
        [],
    )
    record_rows = {
        row["factor_id"]: row
        for row in rows
        if row["grounding_status"]
        == "record_matched"
    }

    if case_id != "rac_store_a_attribution_001":
        if record_rows:
            fail(
                f"{case_id} unexpectedly used "
                "record grounding"
            )
        return

    if set(record_rows) != set(STORE_A_FACTOR_FIELDS):
        fail("Store A grounded factor mismatch")

    if (
        state["grounded_evidence"]["summary"].get(
            "record_matched_count"
        )
        != 5
    ):
        fail("Store A record count is not five")

    for factor_id, row in record_rows.items():
        if row["source_path"] != STORE_A_SOURCE_PATH:
            fail(
                f"{factor_id} used unexpected source"
            )

        if row["evidence_fields"] != list(
            STORE_A_FACTOR_FIELDS[factor_id]
        ):
            fail(
                f"{factor_id} evidence fields mismatch"
            )

        months = tuple(
            item["row_key"]["period_month"]
            for item in row["evidence_values"]
        )

        if months != STORE_A_PERIOD_MONTHS:
            fail(
                f"{factor_id} period selection mismatch"
            )

    for fragment in [
        "Record matched packets: 5",
        (
            "records: store_id=A; "
            "period_month=2026-03, 2026-04; rows=2"
        ),
        "search_exposure_users=4172",
        "transaction_orders=337",
        "activity_cost_ratio_pct=40.69",
    ]:
        if fragment not in state["final_report"]:
            fail(
                "Store A report missing: "
                + fragment
            )


def main() -> None:
    cases = load_eval_cases()
    if not cases:
        fail("No eval cases found")

    total_packets = 0
    total_record_matches = 0
    total_keyword_matches = 0
    total_boundary_matches = 0
    total_fallbacks = 0

    for case in cases:
        state = run_grounded_pipeline(case["question"], root=ROOT)
        if "grounded_evidence" not in state:
            fail(f"{case['case_id']} missing grounded_evidence")

        summary = state["grounded_evidence"]["summary"]
        rows = state.get("grounded_evidence_rows", [])

        validate_promotion_grounded_rows(
            case["case_id"],
            rows,
        )

        if summary["total_packets"] == 0:
            fail(f"{case['case_id']} has zero grounded packets")

        if summary["source_missing_count"] != 0:
            fail(f"{case['case_id']} has missing sources")

        if not rows:
            fail(f"{case['case_id']} has no grounded evidence rows")

        for row in rows:
            if not row["source_path"]:
                fail(f"{case['case_id']} has row without source path")

            if (
                row["grounding_status"]
                == "record_matched"
            ):
                if (
                    row["line_range"] != "n/a"
                    or row["snippet"]
                ):
                    fail(
                        f"{case['case_id']} record row "
                        "claims text-line grounding"
                    )
            elif not row["snippet"]:
                fail(
                    f"{case['case_id']} has row "
                    "without local snippet"
                )

            if not row.get("grounding_role"):
                fail(f"{case['case_id']} has row without grounding_role")

            if row["grounding_status"] not in ALLOWED_GROUNDING_STATUSES:
                fail(
                    f"{case['case_id']} has invalid grounding status: "
                    f"{row['grounding_status']}"
                )

        report = state["final_report"]
        validate_store_a_grounded_case(
            case["case_id"],
            state,
        )

        for section in REQUIRED_REPORT_SECTIONS:
            if section not in report:
                fail(f"{case['case_id']} report missing section: {section}")

        for column in [
            "Source Locator",
            "Evidence Fields",
            "Selected Values",
        ]:
            if column not in report:
                fail(
                    f"{case['case_id']} report "
                    f"does not expose {column}"
                )

        if "Missing source files: 0" not in report:
            fail(f"{case['case_id']} report does not show zero missing sources")

        total_packets += summary["total_packets"]
        total_record_matches += summary.get(
            "record_matched_count",
            0,
        )
        total_keyword_matches += summary["keyword_matched_count"]
        total_boundary_matches += summary.get("boundary_matched_count", 0)
        total_fallbacks += summary["fallback_count"]

    print("[OK] RAC grounded pipeline validation passed")
    print(f"[OK] Eval cases checked: {len(cases)}")
    print(f"[OK] Total grounded packets: {total_packets}")
    print(
        f"[OK] Record matched packets: "
        f"{total_record_matches}"
    )
    print(f"[OK] Keyword matched packets: {total_keyword_matches}")
    print(f"[OK] Boundary matched packets: {total_boundary_matches}")
    print(f"[OK] Fallback packets: {total_fallbacks}")
    print("[OK] Validation completed without writing outputs")


if __name__ == "__main__":
    main()
