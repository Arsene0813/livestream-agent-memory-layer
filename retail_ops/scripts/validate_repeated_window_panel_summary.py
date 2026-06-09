#!/usr/bin/env python3
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

sql_path = ROOT / "retail_ops/sql/04_repeated_window_panel_summary.sql"
output_path = ROOT / "retail_ops/outputs/repeated_window_panel_summary_output.csv"
panel_path = ROOT / "retail_ops/data/store_period_panel_metrics.csv"

required_files = [sql_path, output_path, panel_path]
missing = [str(path.relative_to(ROOT)) for path in required_files if not path.exists()]
if missing:
    print("[FAIL] Missing repeated-window panel summary files:")
    for item in missing:
        print(f" - {item}")
    sys.exit(1)

sql_text = sql_path.read_text(encoding="utf-8")

# Strip SQL comments before checking whether forbidden fields are used.
sql_without_comments = []
for line in sql_text.splitlines():
    stripped = line.strip()
    if stripped.startswith("--"):
        continue
    sql_without_comments.append(line)
query_text = "\n".join(sql_without_comments)

for forbidden in [
    "full_refund_order_count",
    "full_or_partial_refund_order_count",
    "self_operated",
]:
    if re.search(rf"\b{re.escape(forbidden)}\b", query_text):
        print(f"[FAIL] Summary SQL uses forbidden or non-canonical field/value in query body: {forbidden}")
        sys.exit(1)

required_columns = {
    "store_id",
    "region_type",
    "store_type",
    "observed_month_count",
    "feb_transaction_amount",
    "mar_transaction_amount",
    "apr_transaction_amount",
    "transaction_amount_feb_to_apr_delta",
    "transaction_amount_feb_to_apr_pct",
    "transaction_orders_feb_to_apr_pct",
    "exposure_users_feb_to_apr_pct",
    "entry_users_feb_to_apr_pct",
    "entry_conversion_rate_pct_feb_to_apr_delta",
    "order_conversion_rate_pct_feb_to_apr_delta",
    "payment_conversion_rate_pct_feb_to_apr_delta",
    "search_exposure_users_feb_to_apr_pct",
    "search_entry_users_feb_to_apr_pct",
    "activity_cost_ratio_pct_feb_to_apr_delta",
    "repeated_window_summary_flag",
    "summary_boundary_note",
}

allowed_store_types = {"self-operated", "partner"}

with output_path.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fields = set(reader.fieldnames or [])
    rows = list(reader)

missing_columns = sorted(required_columns - fields)
if missing_columns:
    print("[FAIL] Summary output missing required columns:")
    for col in missing_columns:
        print(f" - {col}")
    sys.exit(1)

expected_stores = ["B", "C", "D", "E", "F"]
observed_stores = sorted(row.get("store_id") for row in rows)
if observed_stores != expected_stores:
    print("[FAIL] Summary output should contain exactly Stores B-F.")
    print(f"Expected: {expected_stores}")
    print(f"Observed: {observed_stores}")
    sys.exit(1)

for row in rows:
    store_id = row.get("store_id")

    if row.get("observed_month_count") != "3":
        print(f"[FAIL] Store {store_id} should have observed_month_count = 3.")
        print(row)
        sys.exit(1)

    if row.get("store_type") not in allowed_store_types:
        print(f"[FAIL] Store {store_id} has non-canonical store_type: {row.get('store_type')}")
        print("Allowed values: self-operated, partner")
        sys.exit(1)

    if row.get("repeated_window_summary_flag") != "summary_ready_for_descriptive_review":
        print(f"[FAIL] Store {store_id} has unexpected summary flag.")
        print(row)
        sys.exit(1)

    note = row.get("summary_boundary_note", "")
    required_note_bits = [
        "Descriptive repeated-window summary only",
        "not a store ranking",
        "pairwise comparability gate",
        "causal analysis",
    ]
    missing_bits = [bit for bit in required_note_bits if bit not in note]
    if missing_bits:
        print(f"[FAIL] Store {store_id} summary boundary note is missing required phrases:")
        for bit in missing_bits:
            print(f" - {bit}")
        sys.exit(1)

numeric_check_columns = [
    "transaction_amount_feb_to_apr_pct",
    "transaction_orders_feb_to_apr_pct",
    "exposure_users_feb_to_apr_pct",
    "entry_users_feb_to_apr_pct",
]

for row in rows:
    for col in numeric_check_columns:
        value = row.get(col)
        if value in ("", None):
            print(f"[FAIL] Store {row.get('store_id')} has empty numeric summary column: {col}")
            sys.exit(1)

        try:
            float(value)
        except ValueError:
            print(f"[FAIL] Store {row.get('store_id')} has non-numeric value in {col}: {value}")
            sys.exit(1)

for path in [sql_path, output_path, panel_path]:
    text = path.read_text(encoding="utf-8")
    for alias in ["full_refund_order_count", "full_or_partial_refund_order_count", "self_operated"]:
        if alias in text:
            print(f"[FAIL] Non-canonical alias remains in {path.relative_to(ROOT)}: {alias}")
            sys.exit(1)

print("[PASS] Repeated-window panel summary validation passed.")
print("[PASS] Summary output contains Stores B-F with 3 observed months each.")
print("[PASS] Summary uses canonical store_type values.")
print("[PASS] Summary remains descriptive and boundary-preserving.")
