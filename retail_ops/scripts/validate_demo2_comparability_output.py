#!/usr/bin/env python3
"""Validate Demo 2 cross-store comparability output.

This validator checks that Demo 2 output stays within the current dictionary-defined comparison schema.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path


OUTPUT_PATH = Path("retail_ops/outputs/demo2_cross_store_comparability_output.csv")

REQUIRED_COLUMNS = [
    "store_id",
    "period_month",
    "period_start",
    "period_end",
    "region_type",
    "store_type",
    "transaction_amount",
    "transaction_orders",
    "average_order_value",
    "exposure_users",
    "exposure_times",
    "store_average_rank",
    "entry_users",
    "entry_times",
    "entry_conversion_rate_pct",
    "order_users",
    "order_times",
    "order_conversion_rate_pct",
    "order_amount",
    "payment_users",
    "payment_amount",
    "payment_conversion_rate_pct",
    "search_exposure_users",
    "search_average_rank",
    "search_entry_users",
    "search_entry_rate_pct",
    "search_entry_share_pct",
    "merchant_list_exposure_users",
    "merchant_list_average_rank",
    "merchant_list_entry_users",
    "activity_original_transaction_amount",
    "activity_orders",
    "activity_cost",
    "merchant_subsidy_amount",
    "platform_subsidy_amount",
    "activity_cost_ratio_pct",
    "activity_order_share_pct",
    "refund_amount",
    "full_refund_orders",
    "refund_orders_all_or_partial",
    "business_district_rank",
    "top3_sku_transaction_amount",
    "top3_sku_transaction_amount_share_pct",
    "comparison_scope_flag",
    "comparison_limit_notes",
]


def parse_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value == "":
        raise AssertionError(f"Missing numeric value for {key} in row {row.get('store_id')}")
    return float(value)


def assert_close(row: dict[str, str], key: str, expected: float, tolerance: float = 0.05) -> None:
    actual = parse_float(row, key)
    if not math.isclose(actual, expected, abs_tol=tolerance):
        raise AssertionError(
            f"{row.get('store_id')} {key} mismatch: actual={actual}, expected={expected}"
        )


def pct(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return round(100.0 * numerator / denominator, 2)


def main() -> None:
    if not OUTPUT_PATH.exists():
        raise SystemExit(f"Missing output file: {OUTPUT_PATH}")

    with OUTPUT_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)

    missing = [col for col in REQUIRED_COLUMNS if col not in headers]
    if missing:
        raise SystemExit(f"Missing expected columns: {missing}")


    if len(rows) != 5:
        raise SystemExit(f"Expected 5 Demo 2 store rows, found {len(rows)}")

    store_ids = [row["store_id"] for row in rows]
    if store_ids != ["B", "C", "D", "E", "F"]:
        raise SystemExit(f"Unexpected Demo 2 store order: {store_ids}")

    for row in rows:
        store_id = row["store_id"]

        transaction_amount = parse_float(row, "transaction_amount")
        transaction_orders = parse_float(row, "transaction_orders")
        search_exposure_users = parse_float(row, "search_exposure_users")
        search_entry_users = parse_float(row, "search_entry_users")
        entry_users = parse_float(row, "entry_users")
        activity_orders = parse_float(row, "activity_orders")
        refund_amount = parse_float(row, "refund_amount")
        top3_sku_transaction_amount = parse_float(row, "top3_sku_transaction_amount")

        assert_close(
            row,
            "average_order_value",
            round(transaction_amount / transaction_orders, 2),
        )
        assert_close(
            row,
            "search_entry_rate_pct",
            pct(search_entry_users, search_exposure_users),
        )
        assert_close(
            row,
            "search_entry_share_pct",
            pct(search_entry_users, entry_users),
        )
        assert_close(
            row,
            "activity_order_share_pct",
            pct(activity_orders, transaction_orders),
        )
        assert_close(
            row,
            "top3_sku_transaction_amount_share_pct",
            pct(top3_sku_transaction_amount, transaction_amount),
        )

        if row["comparison_scope_flag"] != "same_period_diagnostic_ready":
            raise SystemExit(
                f"{store_id} has unexpected comparison_scope_flag: "
                f"{row['comparison_scope_flag']}"
            )

        if not row["comparison_limit_notes"].strip():
            raise SystemExit(f"{store_id} has empty comparison_limit_notes")

    print("[OK] Demo 2 comparability output validation passed")
    print("[OK] Checked required fields for current Demo 2 diagnostic output")
    print("[OK] Checked derived rates and comparison-scope fields")


if __name__ == "__main__":
    main()
