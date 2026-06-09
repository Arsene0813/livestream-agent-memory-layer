#!/usr/bin/env python3
"""Analyze Demo 2 guardrail sensitivity without unclear order-status fields.

This script intentionally excludes valid_orders, invalid_orders, and
invalid_order_pressure_pct. Demo 2 guardrail sensitivity is based only on
fields that remain in the current field contract:

- activity_order_share_pct
- top3_sku_transaction_amount_share_pct
- comparison_scope_flag
- comparison_limit_notes
"""

from __future__ import annotations

import csv
from pathlib import Path


INPUT_PATH = Path("retail_ops/outputs/demo2_cross_store_comparability_output.csv")
OUTPUT_PATH = Path("retail_ops/outputs/demo2_guardrail_sensitivity_summary.csv")

REQUIRED_COLUMNS = [
    "store_id",
    "activity_order_share_pct",
    "top3_sku_transaction_amount_share_pct",
    "comparison_scope_flag",
    "comparison_limit_notes",
]


THRESHOLD_SETS = [
    {
        "scenario": "current",
        "activity_high": 80.0,
        "activity_moderate": 60.0,
        "top3_high": 25.0,
        "top3_moderate": 15.0,
    },
    {
        "scenario": "stricter",
        "activity_high": 75.0,
        "activity_moderate": 55.0,
        "top3_high": 22.0,
        "top3_moderate": 12.0,
    },
    {
        "scenario": "looser",
        "activity_high": 85.0,
        "activity_moderate": 65.0,
        "top3_high": 30.0,
        "top3_moderate": 18.0,
    },
]


def parse_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return 0.0
    return float(value)


def classify(value: float, high: float, moderate: float, name: str) -> str | None:
    if value >= high:
        return f"high_{name}"
    if value >= moderate:
        return f"moderate_{name}"
    return None


def main() -> None:
    if not INPUT_PATH.exists():
        raise SystemExit(f"Missing Demo 2 output: {INPUT_PATH}")

    with INPUT_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)

    missing = [col for col in REQUIRED_COLUMNS if col not in headers]
    if missing:
        raise SystemExit(f"Missing required columns in Demo 2 output: {missing}")

    forbidden = [
        "valid_orders",
        "invalid_orders",
        "invalid_order_pressure_pct",
    ]
    present_forbidden = [col for col in forbidden if col in headers]
    if present_forbidden:
        raise SystemExit(
            "Forbidden unclear order-status fields remain in Demo 2 output: "
            f"{present_forbidden}"
        )

    output_rows: list[dict[str, str]] = []

    for scenario in THRESHOLD_SETS:
        for row in rows:
            notes: list[str] = []

            activity = parse_float(row, "activity_order_share_pct")
            top3 = parse_float(row, "top3_sku_transaction_amount_share_pct")

            for flag in [
                classify(activity, scenario["activity_high"], scenario["activity_moderate"], "activity_involvement"),
                classify(top3, scenario["top3_high"], scenario["top3_moderate"], "top3_sku_concentration"),
            ]:
                if flag:
                    notes.append(flag)

            if notes:
                notes.append("compare_with_region_store_type_activity_product_mix_limits")
            else:
                notes.append("same_period_diagnostic_ready")

            output_rows.append(
                {
                    "scenario": scenario["scenario"],
                    "store_id": row["store_id"],
                    "activity_order_share_pct": f"{activity:.2f}",
                    "top3_sku_transaction_amount_share_pct": f"{top3:.2f}",
                    "sensitivity_limit_notes": "; ".join(notes),
                    "current_comparison_scope_flag": row["comparison_scope_flag"],
                    "current_comparison_limit_notes": row["comparison_limit_notes"],
                }
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scenario",
                "store_id",
                "activity_order_share_pct",
                "top3_sku_transaction_amount_share_pct",
                "sensitivity_limit_notes",
                "current_comparison_scope_flag",
                "current_comparison_limit_notes",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    print("[OK] Demo 2 guardrail sensitivity summary written")
    print(f"[OK] Output: {OUTPUT_PATH}")
    print("[OK] Unclear order-status fields are excluded")


if __name__ == "__main__":
    main()
