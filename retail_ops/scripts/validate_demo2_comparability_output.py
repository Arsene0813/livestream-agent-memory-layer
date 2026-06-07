from pathlib import Path
import csv

OUTPUT_PATH = Path("retail_ops/outputs/demo2_cross_store_comparability_output.csv")

EXPECTED_STORES = {"B", "C", "D", "E", "F"}

EXPECTED_TOP3_AMOUNT = {
    "B": 1300.90,
    "C": 2004.84,
    "D": 3055.78,
    "E": 726.25,
    "F": 1798.40,
}

EXPECTED_REQUIRED_COLUMNS = [
    "store_id",
    "period_month",
    "period_start",
    "period_end",
    "region_type",
    "store_type",
    "transaction_amount",
    "transaction_orders",
    "valid_orders",
    "invalid_orders",
    "order_amount",
    "payment_amount",
    "search_entry_rate_pct",
    "search_entry_share_pct",
    "activity_order_share_pct",
    "refund_pressure_pct",
    "invalid_order_pressure_pct",
    "top3_sku_transaction_amount",
    "top3_sku_transaction_amount_share_pct",
    "comparison_scope_flag",
    "comparison_limit_notes",
]

with OUTPUT_PATH.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    fields = reader.fieldnames or []

if len(rows) != 5:
    raise SystemExit(f"Expected 5 comparability rows, got {len(rows)}")

missing_columns = [col for col in EXPECTED_REQUIRED_COLUMNS if col not in fields]
if missing_columns:
    raise SystemExit(f"Missing expected columns: {missing_columns}")

stores = {row["store_id"] for row in rows}
if stores != EXPECTED_STORES:
    raise SystemExit(f"Unexpected store set: {sorted(stores)}")

for row in rows:
    store_id = row["store_id"]

    if row["period_month"] != "2026-03":
        raise SystemExit(f"{store_id}: bad period_month")

    if row["period_start"] != "2026-03-01" or row["period_end"] != "2026-03-31":
        raise SystemExit(f"{store_id}: bad period range")

    if row["comparison_scope_flag"] != "same_period_diagnostic_ready":
        raise SystemExit(f"{store_id}: unexpected comparison_scope_flag")

    if "compare_with_region_store_type_activity_refund_limits" not in row["comparison_limit_notes"]:
        raise SystemExit(f"{store_id}: missing comparison limit note")

    transaction_amount = float(row["transaction_amount"])
    transaction_orders = float(row["transaction_orders"])
    valid_orders = float(row["valid_orders"])
    invalid_orders = float(row["invalid_orders"])
    entry_users = float(row["entry_users"])
    search_entry_users = float(row["search_entry_users"])
    search_exposure_users = float(row["search_exposure_users"])
    activity_orders = float(row["activity_orders"])
    refund_amount = float(row["refund_amount"])
    top3_amount = float(row["top3_sku_transaction_amount"])

    checks = [
        ("search_entry_rate_pct", search_entry_users / search_exposure_users * 100),
        ("search_entry_share_pct", search_entry_users / entry_users * 100),
        ("activity_order_share_pct", activity_orders / transaction_orders * 100),
        ("refund_pressure_pct", refund_amount / transaction_amount * 100),
        (
            "invalid_order_pressure_pct",
            invalid_orders / (valid_orders + invalid_orders) * 100,
        ),
        (
            "top3_sku_transaction_amount_share_pct",
            top3_amount / transaction_amount * 100,
        ),
    ]

    for field, expected in checks:
        actual = float(row[field])
        if abs(actual - expected) > 0.02:
            raise SystemExit(
                f"{store_id}: {field} mismatch.\n"
                f"actual={actual:.4f}, expected={expected:.4f}"
            )

    if abs(top3_amount - EXPECTED_TOP3_AMOUNT[store_id]) > 0.02:
        raise SystemExit(
            f"{store_id}: top3_sku_transaction_amount mismatch.\n"
            f"actual={top3_amount:.2f}, expected={EXPECTED_TOP3_AMOUNT[store_id]:.2f}"
        )

print("Demo 2 comparability output validation PASSED.")
print("Checked comparability rows: 5")
print("Checked expected stores: B, C, D, E, F")
print("Checked period alignment: 2026-03-01 to 2026-03-31")
print("Checked derived diagnostics and top3 SKU transaction amounts.")
