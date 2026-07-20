#!/usr/bin/env python3
import csv
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

panel_path = ROOT / "retail_ops/data/store_period_panel_metrics.csv"
demo2_source_path = ROOT / "retail_ops/data/demo2_store_period_metrics.csv"
notes_path = ROOT / "retail_ops/data/store_period_panel_source_notes.md"
sql_path = ROOT / "retail_ops/sql/03_store_period_panel_coverage.sql"
output_path = ROOT / "retail_ops/outputs/store_period_panel_coverage_output.csv"

required_files = [
    panel_path,
    demo2_source_path,
    notes_path,
    sql_path,
    output_path,
]
missing = [str(path.relative_to(ROOT)) for path in required_files if not path.exists()]
if missing:
    print("[FAIL] Missing repeated-window panel extension files:")
    for item in missing:
        print(f" - {item}")
    sys.exit(1)

forbidden_panel_aliases = {
    "full_refund_order_count",
    "full_or_partial_refund_order_count",
    "self_operated",
    "store_exposure_users",
    "store_exposure_times",
    "entry_visits",
    "order_submissions",
    "full_or_partial_refund_orders",
    "business_area_rank",
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
    "merchant_list_exposure_users",
    "merchant_list_entry_users",
    "merchant_list_average_rank",
    "activity_zone_exposure_users",
    "activity_zone_entry_users",
    "order_page_exposure_users",
    "order_page_entry_users",
    "other_exposure_users",
    "other_entry_users",
    "activity_original_transaction_amount",
    "activity_orders",
    "activity_cost",
    "merchant_subsidy_amount",
    "platform_subsidy_amount",
    "activity_cost_ratio_pct",
    "refund_amount",
    "full_refund_orders",
    "refund_orders_all_or_partial",
}

allowed_store_types = {"self-operated", "partner"}

with panel_path.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    panel_fields = set(reader.fieldnames or [])
    rows = list(reader)

missing_fields = sorted(required_panel_fields - panel_fields)
if missing_fields:
    print("[FAIL] Missing required panel fields:")
    for field in missing_fields:
        print(f" - {field}")
    sys.exit(1)

forbidden_present = sorted(forbidden_panel_aliases & panel_fields)
if forbidden_present:
    print("[FAIL] Forbidden non-canonical fields are present in panel header:")
    for field in forbidden_present:
        print(f" - {field}")
    sys.exit(1)

if not rows:
    print("[FAIL] Panel CSV has no data rows.")
    sys.exit(1)

bad_store_types = sorted(
    {
        row.get("store_type", "")
        for row in rows
        if row.get("store_type", "") not in allowed_store_types
    }
)
if bad_store_types:
    print("[FAIL] Panel CSV contains non-canonical store_type values:")
    for value in bad_store_types:
        print(f" - {value}")
    print("Allowed values: self-operated, partner")
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
        print(f" - {store_id} {period_month}")
    sys.exit(1)

required_complete_stores = {
    "B": ["2026-02", "2026-03", "2026-04"],
    "C": ["2026-02", "2026-03", "2026-04"],
    "D": ["2026-02", "2026-03", "2026-04"],
    "E": ["2026-02", "2026-03", "2026-04"],
    "F": ["2026-02", "2026-03", "2026-04"],
}

for store_id, expected_months in required_complete_stores.items():
    observed_months = sorted(
        row.get("period_month")
        for row in rows
        if row.get("store_id") == store_id
    )
    if observed_months != expected_months:
        print(f"[FAIL] Store {store_id} panel months are incomplete.")
        print(f"Expected: {expected_months}")
        print(f"Observed: {observed_months}")
        sys.exit(1)


# Verify that every Demo 2 source row matches the corresponding panel row.
# Compare by field name rather than column position because the two CSV files
# use different column ordering.
panel_by_key = {
    (row.get("store_id", ""), row.get("period_month", "")): row
    for row in rows
}

with demo2_source_path.open(newline="", encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    demo2_source_fields = reader.fieldnames or []
    demo2_source_rows = list(reader)

if not demo2_source_fields:
    print("[FAIL] Demo 2 source CSV has no header.")
    sys.exit(1)

if not demo2_source_rows:
    print("[FAIL] Demo 2 source CSV has no data rows.")
    sys.exit(1)

missing_source_fields_in_panel = sorted(
    set(demo2_source_fields) - panel_fields
)

if missing_source_fields_in_panel:
    print(
        "[FAIL] Demo 2 source fields are missing from "
        "the repeated-window panel:"
    )
    for field in missing_source_fields_in_panel:
        print(f" - {field}")
    sys.exit(1)

text_fields = {
    "store_id",
    "period_month",
    "period_start",
    "period_end",
    "region_type",
    "store_type",
}


def clean_value(value: str | None) -> str:
    return "" if value is None else value.strip()


def values_match(
    field: str,
    source_value: str | None,
    panel_value: str | None,
) -> bool:
    source_text = clean_value(source_value)
    panel_text = clean_value(panel_value)

    # Missingness is meaningful. A source value must not silently become
    # blank or zero in the panel.
    if source_text == "" or panel_text == "":
        return source_text == panel_text

    if field in text_fields:
        return source_text == panel_text

    try:
        return Decimal(source_text) == Decimal(panel_text)
    except InvalidOperation:
        return source_text == panel_text


source_seen = set()
parity_failures = []

for source_row in demo2_source_rows:
    key = (
        clean_value(source_row.get("store_id")),
        clean_value(source_row.get("period_month")),
    )

    if key in source_seen:
        print(f"[FAIL] Duplicate Demo 2 source row: {key}")
        sys.exit(1)

    source_seen.add(key)
    panel_row = panel_by_key.get(key)

    if panel_row is None:
        parity_failures.append(
            {
                "key": key,
                "field": "<row>",
                "source_value": "present",
                "panel_value": "missing",
            }
        )
        continue

    for field in demo2_source_fields:
        source_value = source_row.get(field)
        panel_value = panel_row.get(field)

        if not values_match(field, source_value, panel_value):
            parity_failures.append(
                {
                    "key": key,
                    "field": field,
                    "source_value": clean_value(source_value),
                    "panel_value": clean_value(panel_value),
                }
            )

if parity_failures:
    print("[FAIL] Demo 2 source-to-panel parity check failed:")
    for failure in parity_failures:
        store_id, period_month = failure["key"]
        print(
            " - "
            f"{store_id} {period_month} "
            f"{failure['field']}: "
            f"source={failure['source_value']!r}, "
            f"panel={failure['panel_value']!r}"
        )
    sys.exit(1)

# Check old aliases in data / SQL / output / notes, not in this validator file.
alias_scan_paths = [
    panel_path,
    demo2_source_path,
    notes_path,
    sql_path,
    output_path,
]
for path in alias_scan_paths:
    text = path.read_text(encoding="utf-8")
    for alias in sorted(forbidden_panel_aliases):
        if alias in text:
            print(f"[FAIL] Non-canonical alias remains in {path.relative_to(ROOT)}: {alias}")
            sys.exit(1)

notes = notes_path.read_text(encoding="utf-8")
required_note_phrases = [
    "Coverage uses selected dictionary-defined store-period fields for repeated-window diagnostic coverage",
    "`refund_amount`",
    "`full_refund_orders`",
    "`refund_orders_all_or_partial`",
    "validated field-by-field against that source table",
]
missing_note_phrases = [phrase for phrase in required_note_phrases if phrase not in notes]
if missing_note_phrases:
    print("[FAIL] Source notes missing required repeated-window or scope phrases:")
    for phrase in missing_note_phrases:
        print(f" - {phrase}")
    sys.exit(1)

with output_path.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    output_rows = list(reader)

output_by_store = {row.get("store_id"): row for row in output_rows}
for store_id in required_complete_stores:
    if store_id not in output_by_store:
        print(f"[FAIL] Missing Store {store_id} coverage output row.")
        sys.exit(1)

    row = output_by_store[store_id]
    if row.get("observed_month_count") != "3":
        print(f"[FAIL] Store {store_id} coverage output should show observed_month_count = 3.")
        print(f"Observed row: {row}")
        sys.exit(1)

    if row.get("panel_coverage_flag") != "panel_ready_for_repeated_window_diagnostic":
        print(f"[FAIL] Store {store_id} should be ready for repeated-window diagnostic.")
        print(f"Observed row: {row}")
        sys.exit(1)

print("[PASS] Repeated-window panel extension validation passed.")
print("[PASS] Store B, Store C, Store D, Store E, and Store F each have 2026-02, 2026-03, and 2026-04.")
print("[PASS] Canonical source field names are preserved where retained.")
print("[PASS] Canonical store_type values are used: self-operated and partner.")
print("[PASS] Demo 2 source rows match panel rows across all shared fields.")
print("[PASS] Panel uses the current dictionary-defined source/output schema.")
