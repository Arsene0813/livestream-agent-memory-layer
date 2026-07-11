#!/usr/bin/env python3
"""Reproduce Demo 2 guardrail notes and test ±5pp threshold shifts."""

from __future__ import annotations

import csv
import re
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SQL_REL = Path("retail_ops/sql/02_demo2_cross_store_comparability.sql")
INPUT_REL = Path("retail_ops/outputs/demo2_cross_store_comparability_output.csv")
SUMMARY_REL = Path("retail_ops/outputs/demo2_guardrail_sensitivity_summary.csv")
RESULT_REL = Path("retail_ops/outputs/demo2_guardrail_sensitivity_result.txt")

SQL_PATH = ROOT / SQL_REL
INPUT_PATH = ROOT / INPUT_REL
SUMMARY_PATH = ROOT / SUMMARY_REL
RESULT_PATH = ROOT / RESULT_REL

BASE_NOTE = "compare_with_region_store_type_activity_product_mix_limits"
REQUIRED_COLUMNS = [
    "store_id",
    "transaction_amount",
    "activity_order_share_pct",
    "top3_sku_transaction_amount",
    "top3_sku_transaction_amount_share_pct",
    "comparison_scope_flag",
    "comparison_limit_notes",
]
OUTPUT_COLUMNS = [
    "scenario",
    "store_id",
    "activity_order_share_pct",
    "top3_sku_transaction_amount_share_pct",
    "sensitivity_limit_notes",
    "current_comparison_scope_flag",
    "current_comparison_limit_notes",
]
THRESHOLD_PATTERNS = {
    "activity_high": (
        r"activity_order_share_pct\s*>=\s*([0-9.]+)\s+THEN\s+"
        r"'high_activity_involvement;\s*'"
    ),
    "activity_moderate": (
        r"activity_order_share_pct\s*>=\s*([0-9.]+)\s+THEN\s+"
        r"'moderate_activity_involvement;\s*'"
    ),
    "top3_concentration": (
        r"top3_sku_transaction_amount_share_pct\s*>=\s*([0-9.]+)\s+THEN\s+"
        r"'top3_sku_amount_concentration;\s*'"
    ),
}
REQUIRED_NOTE_TOKENS = [
    "missing_transaction_amount",
    "missing_top3_sku_amount_evidence",
    "high_activity_involvement",
    "moderate_activity_involvement",
    "top3_sku_amount_concentration",
    BASE_NOTE,
]


def clean(value: str | None) -> str:
    return "" if value is None else value.strip()


def decimal_value(row: dict[str, str], field: str) -> Decimal | None:
    text = clean(row.get(field))
    if text == "":
        return None
    try:
        return Decimal(text)
    except InvalidOperation as exc:
        store_id = clean(row.get("store_id")) or "<unknown>"
        raise SystemExit(
            f"Invalid decimal for Store {store_id} {field}: {text!r}"
        ) from exc


def read_sql_thresholds() -> dict[str, Decimal]:
    if not SQL_PATH.exists():
        raise SystemExit(f"Missing SQL file: {SQL_REL}")

    sql = SQL_PATH.read_text(encoding="utf-8")
    missing_tokens = [token for token in REQUIRED_NOTE_TOKENS if token not in sql]
    if missing_tokens:
        raise SystemExit(f"SQL is missing guardrail note tokens: {missing_tokens}")

    thresholds: dict[str, Decimal] = {}
    for name, pattern in THRESHOLD_PATTERNS.items():
        matches = re.findall(pattern, sql, flags=re.IGNORECASE | re.MULTILINE)
        if len(matches) != 1:
            raise SystemExit(
                f"Expected one SQL threshold for {name}; found {len(matches)}."
            )
        thresholds[name] = Decimal(matches[0])

    if thresholds["activity_high"] <= thresholds["activity_moderate"]:
        raise SystemExit("activity_high must exceed activity_moderate.")
    return thresholds


def read_rows() -> list[dict[str, str]]:
    if not INPUT_PATH.exists():
        raise SystemExit(f"Missing input file: {INPUT_REL}")

    with INPUT_PATH.open(newline="", encoding="utf-8-sig") as file_obj:
        reader = csv.DictReader(file_obj)
        headers = reader.fieldnames or []
        rows = list(reader)

    missing = [field for field in REQUIRED_COLUMNS if field not in headers]
    if missing:
        raise SystemExit(f"Missing input columns: {missing}")
    if not rows:
        raise SystemExit("Demo 2 output has no data rows.")

    store_ids = [clean(row.get("store_id")) for row in rows]
    if any(not store_id for store_id in store_ids):
        raise SystemExit("Demo 2 output contains a blank store_id.")
    if len(store_ids) != len(set(store_ids)):
        raise SystemExit("Demo 2 output contains duplicate store_id rows.")
    return rows


def build_notes(
    row: dict[str, str],
    thresholds: dict[str, Decimal],
) -> str:
    transaction_amount = decimal_value(row, "transaction_amount")
    activity_share = decimal_value(row, "activity_order_share_pct")
    top3_amount = decimal_value(row, "top3_sku_transaction_amount")
    top3_share = decimal_value(
        row,
        "top3_sku_transaction_amount_share_pct",
    )

    notes: list[str] = []
    if transaction_amount is None:
        notes.append("missing_transaction_amount")
    if top3_amount is None:
        notes.append("missing_top3_sku_amount_evidence")

    if activity_share is not None:
        if activity_share >= thresholds["activity_high"]:
            notes.append("high_activity_involvement")
        elif activity_share >= thresholds["activity_moderate"]:
            notes.append("moderate_activity_involvement")

    if (
        top3_share is not None
        and top3_share >= thresholds["top3_concentration"]
    ):
        notes.append("top3_sku_amount_concentration")

    notes.append(BASE_NOTE)
    return "; ".join(notes)


def threshold_scenarios(
    baseline: dict[str, Decimal],
) -> list[tuple[str, dict[str, Decimal]]]:
    shift = Decimal("5")
    easier = {name: value - shift for name, value in baseline.items()}
    harder = {name: value + shift for name, value in baseline.items()}
    if any(value < 0 for value in easier.values()):
        raise SystemExit("The -5pp scenario creates a negative threshold.")
    return [
        ("baseline_sql", baseline),
        ("easier_to_trigger_minus_5pp", easier),
        ("harder_to_trigger_plus_5pp", harder),
    ]


def verify_baseline(
    rows: list[dict[str, str]],
    baseline: dict[str, Decimal],
) -> dict[str, str]:
    notes_by_store: dict[str, str] = {}
    failures: list[tuple[str, str, str]] = []

    for row in rows:
        store_id = clean(row["store_id"])
        derived = build_notes(row, baseline)
        current = clean(row["comparison_limit_notes"])
        notes_by_store[store_id] = derived
        if derived != current:
            failures.append((store_id, current, derived))

    if failures:
        print("[FAIL] SQL-derived baseline does not reproduce the current output.")
        for store_id, current, derived in failures:
            print(f" - Store {store_id}")
            print(f"   Current: {current!r}")
            print(f"   Derived: {derived!r}")
        raise SystemExit(1)
    return notes_by_store


def write_summary(
    rows: list[dict[str, str]],
    baseline: dict[str, Decimal],
) -> dict[str, list[str]]:
    baseline_notes = verify_baseline(rows, baseline)
    output_rows: list[dict[str, str]] = []
    changed: dict[str, list[str]] = {}

    for scenario, thresholds in threshold_scenarios(baseline):
        changed_stores: list[str] = []
        for row in rows:
            store_id = clean(row["store_id"])
            notes = build_notes(row, thresholds)
            if scenario != "baseline_sql" and notes != baseline_notes[store_id]:
                changed_stores.append(store_id)

            output_rows.append(
                {
                    "scenario": scenario,
                    "store_id": store_id,
                    "activity_order_share_pct": clean(
                        row["activity_order_share_pct"]
                    ),
                    "top3_sku_transaction_amount_share_pct": clean(
                        row["top3_sku_transaction_amount_share_pct"]
                    ),
                    "sensitivity_limit_notes": notes,
                    "current_comparison_scope_flag": clean(
                        row["comparison_scope_flag"]
                    ),
                    "current_comparison_limit_notes": clean(
                        row["comparison_limit_notes"]
                    ),
                }
            )
        changed[scenario] = sorted(changed_stores)

    with SUMMARY_PATH.open("w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=OUTPUT_COLUMNS,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)
    return changed


def write_result(
    rows: list[dict[str, str]],
    baseline: dict[str, Decimal],
    changed: dict[str, list[str]],
) -> None:
    all_stores = sorted(clean(row["store_id"]) for row in rows)
    all_changed = sorted(
        set(changed["easier_to_trigger_minus_5pp"])
        | set(changed["harder_to_trigger_plus_5pp"])
    )
    unchanged = sorted(set(all_stores) - set(all_changed))

    def display(values: list[str]) -> str:
        return ", ".join(values) if values else "none"

    high = baseline["activity_high"]
    moderate = baseline["activity_moderate"]
    top3 = baseline["top3_concentration"]

    lines = [
        "Demo 2 guardrail sensitivity check completed.",
        f"Input: {INPUT_REL}",
        f"SQL source: {SQL_REL}",
        f"Output CSV: {SUMMARY_REL}",
        (
            "SQL-derived baseline: PASSED. Extracted activity "
            f"high/moderate thresholds {high}/{moderate} and top-3 "
            f"concentration threshold {top3}; recomputed notes match "
            "every current comparison_limit_notes value."
        ),
        "Threshold scenarios:",
        (
            "- baseline_sql: activity high/moderate "
            f"{high}/{moderate}; top-3 concentration {top3}."
        ),
        (
            "- easier_to_trigger_minus_5pp: each implemented threshold "
            "is lowered by 5 percentage points."
        ),
        (
            "- harder_to_trigger_plus_5pp: each implemented threshold "
            "is raised by 5 percentage points."
        ),
        f"Rows reviewed: {len(all_stores)}",
        (
            "Changed rows under easier_to_trigger_minus_5pp: "
            f"{display(changed['easier_to_trigger_minus_5pp'])}"
        ),
        (
            "Changed rows under harder_to_trigger_plus_5pp: "
            f"{display(changed['harder_to_trigger_plus_5pp'])}"
        ),
        (
            "Rows changed under at least one perturbation: "
            f"{display(all_changed)} ({len(all_changed)}/{len(all_stores)})"
        ),
        f"Rows unchanged under both perturbations: {display(unchanged)}",
        "Interpretation:",
        "- This check does not optimize thresholds or create peer-selection rules.",
        (
            "- A changed row shows local sensitivity of the diagnostic "
            "note set in the current sample."
        ),
        (
            "- An unchanged row does not establish stability outside "
            "the current sample."
        ),
    ]
    RESULT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    baseline = read_sql_thresholds()
    rows = read_rows()
    changed = write_summary(rows, baseline)
    write_result(rows, baseline, changed)
    print("[OK] SQL-derived baseline reproduced all current comparison_limit_notes.")
    print(f"[OK] Output: {SUMMARY_REL}")
    print(f"[OK] Result: {RESULT_REL}")


if __name__ == "__main__":
    main()
