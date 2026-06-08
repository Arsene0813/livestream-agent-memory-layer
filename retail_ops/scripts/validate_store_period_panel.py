#!/usr/bin/env python3
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

panel_path = ROOT / "retail_ops/data/store_period_panel_metrics.csv"
notes_path = ROOT / "retail_ops/data/store_period_panel_source_notes.md"
sql_path = ROOT / "retail_ops/sql/03_store_period_panel_coverage.sql"
output_path = ROOT / "retail_ops/outputs/store_period_panel_coverage_output.csv"

required_files = [panel_path, notes_path, sql_path, output_path]
missing = [str(path.relative_to(ROOT)) for path in required_files if not path.exists()]
if missing:
    print("[FAIL] Missing panel files:")
    for item in missing:
        print(f"  - {item}")
    sys.exit(1)

forbidden_fields = {
    "valid_orders",
    "invalid_orders",
    "invalid_order_pressure_pct",
}

required_panel_fields = {
    "store_id",
    "period_start",
    "period_end",
    "period_month",
    "region_type",
    "store_type",
    "business_district_rank",
    "transaction_amount",
    "transaction_orders",
    "estimated_income_proxy",
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
    "search_entry_users",
    "search_average_rank",
    "activity_original_transaction_amount",
    "activity_orders",
    "activity_cost",
    "merchant_subsidy_amount",
    "platform_subsidy_amount",
    "activity_cost_ratio_pct",
    "refund_amount",
    "full_refund_order_count",
    "full_or_partial_refund_order_count",
}

with panel_path.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    panel_fields = set(reader.fieldnames or [])
    rows = list(reader)

missing_fields = sorted(required_panel_fields - panel_fields)
if missing_fields:
    print("[FAIL] Missing required panel fields:")
    for field in missing_fields:
        print(f"  - {field}")
    sys.exit(1)

forbidden_present = sorted(forbidden_fields & panel_fields)
if forbidden_present:
    print("[FAIL] Forbidden ambiguous order-status fields are present in panel header:")
    for field in forbidden_present:
        print(f"  - {field}")
    sys.exit(1)

if not rows:
    print("[FAIL] Panel CSV has no data rows.")
    sys.exit(1)

seen = set()
duplicates = []
for row in rows:
    key = (row.get("store_id"), row.get("period_month"))
    if key in seen:
        duplicates.append(key)
    seen.add(key)

if duplicates:
    print("[FAIL] Duplicate store-period rows found:")
    for store_id, period_month in duplicates:
        print(f"  - {store_id} {period_month}")
    sys.exit(1)

store_b_months = sorted(
    row.get("period_month")
    for row in rows
    if row.get("store_id") == "B"
)
if store_b_months != ["2026-02", "2026-03", "2026-04"]:
    print("[FAIL] Store B panel should contain exactly 2026-02, 2026-03, and 2026-04.")
    print(f"Observed: {store_b_months}")
    sys.exit(1)

notes = notes_path.read_text(encoding="utf-8")
required_note_phrases = [
    "The panel intentionally excludes:",
    "`valid_orders`",
    "`invalid_orders`",
    "`invalid_order_pressure_pct`",
    "No alternative hidden definition is inferred.",
]
missing_note_phrases = [phrase for phrase in required_note_phrases if phrase not in notes]
if missing_note_phrases:
    print("[FAIL] Source notes missing required exclusion phrases:")
    for phrase in missing_note_phrases:
        print(f"  - {phrase}")
    sys.exit(1)

with output_path.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    output_rows = list(reader)

b_output = [row for row in output_rows if row.get("store_id") == "B"]
if len(b_output) != 1:
    print(f"[FAIL] Expected one Store B coverage output row, found {len(b_output)}.")
    sys.exit(1)

if b_output[0].get("observed_month_count") != "3":
    print("[FAIL] Store B coverage output should show observed_month_count = 3.")
    print(f"Observed row: {b_output[0]}")
    sys.exit(1)

if b_output[0].get("panel_coverage_flag") != "panel_ready_for_repeated_window_diagnostic":
    print("[FAIL] Store B should be ready for repeated-window diagnostic after adding March.")
    print(f"Observed row: {b_output[0]}")
    sys.exit(1)

print("[PASS] Store-period panel validation passed.")
print("[PASS] Store B has 2026-02, 2026-03, and 2026-04.")
print("[PASS] Ambiguous order-status fields are excluded from the panel.")
