#!/usr/bin/env python3
"""Regenerate the committed repeated-window summary from source CSV and SQL."""

from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

SOURCE_PATH = (
    ROOT
    / "retail_ops"
    / "data"
    / "store_period_panel_metrics.csv"
)

SQL_PATH = (
    ROOT
    / "retail_ops"
    / "sql"
    / "04_repeated_window_panel_summary.sql"
)

OUTPUT_PATH = (
    ROOT
    / "retail_ops"
    / "outputs"
    / "repeated_window_panel_summary_output.csv"
)

EXPECTED_STORES = ["B", "C", "D", "E", "F"]

BASE_METRICS = [
    "transaction_amount",
    "transaction_orders",
    "exposure_users",
    "entry_users",
    "entry_conversion_rate_pct",
    "order_conversion_rate_pct",
    "payment_conversion_rate_pct",
    "search_exposure_users",
    "search_entry_users",
    "activity_orders",
    "activity_cost_ratio_pct",
]

MONTH_PREFIXES = ["feb", "mar", "apr"]


def fail(message: str) -> None:
    print(f"[FAIL] {message}", file=sys.stderr)
    raise SystemExit(1)


def read_source() -> tuple[list[str], list[dict[str, str]]]:
    if not SOURCE_PATH.exists():
        fail(f"Source CSV not found: {SOURCE_PATH}")

    with SOURCE_PATH.open(
        newline="",
        encoding="utf-8",
    ) as source_file:
        reader = csv.DictReader(source_file)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    if not fields:
        fail("Source CSV has no header.")

    if not rows:
        fail("Source CSV has no data rows.")

    missing_fields = [
        field
        for field in BASE_METRICS
        if field not in fields
    ]

    if missing_fields:
        fail(
            "Source CSV is missing required canonical fields: "
            + ", ".join(missing_fields)
        )

    return fields, rows


def read_sql() -> str:
    if not SQL_PATH.exists():
        fail(f"SQL file not found: {SQL_PATH}")

    sql_text = SQL_PATH.read_text(encoding="utf-8")

    executable_lines = [
        line
        for line in sql_text.splitlines()
        if not line.lstrip().startswith(".")
    ]

    executable_sql = "\n".join(executable_lines).strip()

    if not executable_sql:
        fail("SQL file contains no executable SQL.")

    return executable_sql


def execute_summary(
    source_fields: list[str],
    source_rows: list[dict[str, str]],
    executable_sql: str,
) -> tuple[list[str], list[tuple[object, ...]]]:
    connection = sqlite3.connect(":memory:")

    try:
        column_definition = ", ".join(
            f'"{field}" TEXT'
            for field in source_fields
        )

        connection.execute(
            "CREATE TABLE store_period_panel_metrics "
            f"({column_definition})"
        )

        column_names = ", ".join(
            f'"{field}"'
            for field in source_fields
        )

        placeholders = ", ".join(
            "?"
            for _ in source_fields
        )

        connection.executemany(
            (
                "INSERT INTO store_period_panel_metrics "
                f"({column_names}) VALUES ({placeholders})"
            ),
            [
                [
                    row.get(field, "")
                    for field in source_fields
                ]
                for row in source_rows
            ],
        )

        cursor = connection.execute(executable_sql)

        output_fields = [
            description[0]
            for description in cursor.description or []
        ]

        output_rows = cursor.fetchall()

    finally:
        connection.close()

    return output_fields, output_rows


def validate_output(
    output_fields: list[str],
    output_rows: list[tuple[object, ...]],
) -> None:
    if not output_fields:
        fail("Summary SQL returned no columns.")

    if not output_rows:
        fail("Summary SQL returned no rows.")

    if len(output_fields) != 56:
        fail(
            "Expected 56 summary columns after the three-month patch, "
            f"found {len(output_fields)}."
        )

    if len(output_rows) != len(EXPECTED_STORES):
        fail(
            f"Expected {len(EXPECTED_STORES)} rows, "
            f"found {len(output_rows)}."
        )

    required_aliases = [
        f"{prefix}_{metric}"
        for metric in BASE_METRICS
        for prefix in MONTH_PREFIXES
    ]

    missing_aliases = [
        field
        for field in required_aliases
        if field not in output_fields
    ]

    if missing_aliases:
        fail(
            "Summary output is missing month aliases: "
            + ", ".join(missing_aliases)
        )

    try:
        store_id_index = output_fields.index("store_id")
    except ValueError:
        fail("Summary output has no store_id column.")

    observed_stores = [
        str(row[store_id_index])
        for row in output_rows
    ]

    if observed_stores != EXPECTED_STORES:
        fail(
            "Unexpected store scope or order. "
            f"Expected {EXPECTED_STORES}, found {observed_stores}."
        )


def write_output(
    output_fields: list[str],
    output_rows: list[tuple[object, ...]],
) -> None:
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = OUTPUT_PATH.with_suffix(".csv.tmp")

    with temporary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as output_file:
        writer = csv.writer(
            output_file,
            lineterminator="\n",
        )

        writer.writerow(output_fields)
        writer.writerows(output_rows)

    temporary_path.replace(OUTPUT_PATH)


def main() -> None:
    source_fields, source_rows = read_source()
    executable_sql = read_sql()

    output_fields, output_rows = execute_summary(
        source_fields,
        source_rows,
        executable_sql,
    )

    validate_output(
        output_fields,
        output_rows,
    )

    write_output(
        output_fields,
        output_rows,
    )

    print(
        "[PASS] Regenerated "
        "retail_ops/outputs/"
        "repeated_window_panel_summary_output.csv"
    )

    print(
        f"[PASS] Output contains "
        f"{len(output_rows)} rows and "
        f"{len(output_fields)} columns."
    )

    print(
        "[PASS] All selected metrics expose "
        "February, March, and April aliases."
    )


if __name__ == "__main__":
    main()
