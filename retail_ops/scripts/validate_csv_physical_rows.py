#!/usr/bin/env python3
"""Validate that key Demo 2 CSV files have real physical rows.

This catches the failure mode where a CSV has visible content but all rows are
collapsed into one physical line, which weakens reviewability and reproducibility.

It does not change any files.
"""

from pathlib import Path
import csv
import sys

EXPECTED_MIN_PHYSICAL_LINES = {
    "retail_ops/data/demo2_store_period_metrics.csv": 6,
    "retail_ops/data/demo2_top_search_terms.csv": 16,
    "retail_ops/data/demo2_top_skus_by_sales_volume.csv": 16,
    "retail_ops/data/demo2_top_skus_by_transaction_amount.csv": 16,
    "retail_ops/outputs/demo2_cross_store_comparability_output.csv": 6,
}

errors = []

for file_name, min_lines in EXPECTED_MIN_PHYSICAL_LINES.items():
    path = Path(file_name)

    if not path.exists():
        errors.append(f"[MISSING] {file_name}")
        continue

    text = path.read_text(encoding="utf-8")
    physical_lines = text.splitlines()

    if len(physical_lines) < min_lines:
        errors.append(
            f"[PHYSICAL ROWS] {file_name} has {len(physical_lines)} physical lines; "
            f"expected at least {min_lines}."
        )
        continue

    try:
        rows = list(csv.reader(physical_lines))
    except csv.Error as exc:
        errors.append(f"[CSV PARSE] {file_name}: {exc}")
        continue

    if not rows:
        errors.append(f"[EMPTY] {file_name}")
        continue

    header_len = len(rows[0])

    if header_len == 0:
        errors.append(f"[HEADER] {file_name} has empty header.")
        continue

    for row_no, row in enumerate(rows[1:], start=2):
        if len(row) != header_len:
            errors.append(
                f"[WIDTH] {file_name}:{row_no} has {len(row)} columns; "
                f"expected {header_len}."
            )

if errors:
    print("[FAIL] CSV physical-row validation failed")
    for error in errors:
        print(error)
    sys.exit(1)

print("[OK] CSV physical-row validation passed")
