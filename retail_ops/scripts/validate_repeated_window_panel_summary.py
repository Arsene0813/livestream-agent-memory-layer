#!/usr/bin/env python3
"""Check repeated-window source, SQL, output, and field-contract consistency."""

from __future__ import annotations

import csv
import re
import sqlite3
import sys
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

SOURCE = ROOT / "retail_ops/data/store_period_panel_metrics.csv"
SQL = ROOT / "retail_ops/sql/04_repeated_window_panel_summary.sql"
OUTPUT = ROOT / "retail_ops/outputs/repeated_window_panel_summary_output.csv"
DICTIONARY = ROOT / "retail_ops/data/DATA_DICTIONARY.md"

STORES = ["B", "C", "D", "E", "F"]

MONTHS = [
    ("2026-02", "feb"),
    ("2026-03", "mar"),
    ("2026-04", "apr"),
]

METRICS = [
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

ABSOLUTE_DELTAS = [
    "transaction_amount",
    "transaction_orders",
    "exposure_users",
    "entry_users",
    "search_exposure_users",
    "search_entry_users",
    "activity_orders",
]

RELATIVE_CHANGES = [
    "transaction_amount",
    "transaction_orders",
    "exposure_users",
    "entry_users",
    "search_exposure_users",
    "search_entry_users",
]

PERCENTAGE_POINT_DELTAS = [
    "entry_conversion_rate_pct",
    "order_conversion_rate_pct",
    "payment_conversion_rate_pct",
    "activity_cost_ratio_pct",
]

FORBIDDEN_ALIASES = [
    "full_refund_order_count",
    "full_or_partial_refund_order_count",
    "self_operated",
]

ALLOWED_STORE_TYPES = {
    "self-operated",
    "partner",
}

REQUIRED_NOTE_PARTS = [
    "Descriptive repeated-window summary only",
    "not a store ranking",
    "pairwise comparability gate",
    "causal analysis",
]

TWO_DECIMALS = Decimal("0.01")


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    raise SystemExit(1)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        fail(f"Missing file: {path.relative_to(ROOT)}")

    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fields = list(reader.fieldnames or [])
        rows = list(reader)

    if not fields:
        fail(f"{path.relative_to(ROOT)} has no header.")

    if not rows:
        fail(f"{path.relative_to(ROOT)} has no data rows.")

    return fields, rows


def number(value: object, label: str) -> Decimal:
    text = "" if value is None else str(value).strip()

    if not text:
        fail(f"{label} is empty.")

    try:
        return Decimal(text)
    except InvalidOperation:
        fail(f"{label} is not numeric: {value!r}")
        raise AssertionError


def round_two(value: Decimal) -> Decimal:
    return value.quantize(TWO_DECIMALS, rounding=ROUND_HALF_UP)


def assert_number(actual: object, expected: Decimal, label: str) -> None:
    observed = number(actual, label)

    if observed != expected:
        fail(
            f"{label} differs: expected {expected}, "
            f"found {observed}"
        )


def execute_sql(
    source_fields: list[str],
    source_rows: list[dict[str, str]],
) -> tuple[list[str], list[dict[str, str]]]:
    sql_text = SQL.read_text(encoding="utf-8")

    executable_sql = "\n".join(
        line
        for line in sql_text.splitlines()
        if not line.lstrip().startswith(".")
    ).strip()

    connection = sqlite3.connect(":memory:")

    try:
        definitions = ", ".join(
            f'"{field}" TEXT'
            for field in source_fields
        )

        connection.execute(
            f"CREATE TABLE store_period_panel_metrics ({definitions})"
        )

        field_list = ", ".join(
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
                f"({field_list}) VALUES ({placeholders})"
            ),
            [
                [row.get(field, "") for field in source_fields]
                for row in source_rows
            ],
        )

        cursor = connection.execute(executable_sql)

        output_fields = [
            item[0]
            for item in cursor.description or []
        ]

        output_rows = [
            {
                field: "" if value is None else str(value)
                for field, value in zip(output_fields, raw_row)
            }
            for raw_row in cursor.fetchall()
        ]

    finally:
        connection.close()

    return output_fields, output_rows


def expected_month_aliases() -> set[str]:
    return {
        f"{prefix}_{metric}"
        for _, prefix in MONTHS
        for metric in METRICS
    }


def check_dictionary() -> None:
    dictionary_text = DICTIONARY.read_text(encoding="utf-8")
    expected_aliases = expected_month_aliases()

    missing = [
        alias
        for alias in expected_aliases
        if f"`{alias}`" not in dictionary_text
    ]

    if missing:
        fail(
            "DATA_DICTIONARY.md is missing aliases: "
            + ", ".join(missing)
        )


def check_canonical_names() -> None:
    sql_text = SQL.read_text(encoding="utf-8")

    query_text = "\n".join(
        line
        for line in sql_text.splitlines()
        if not line.strip().startswith("--")
    )

    for alias in FORBIDDEN_ALIASES:
        if re.search(rf"\b{re.escape(alias)}\b", query_text):
            fail(f"SQL uses non-canonical alias: {alias}")


def index_source(
    fields: list[str],
    rows: list[dict[str, str]],
) -> dict[tuple[str, str], dict[str, str]]:
    required = {
        "store_id",
        "period_month",
        "region_type",
        "store_type",
        *METRICS,
    }

    missing = sorted(required - set(fields))

    if missing:
        fail(
            "Source panel is missing fields: "
            + ", ".join(missing)
        )

    index = {}

    for row in rows:
        key = (
            row.get("store_id", ""),
            row.get("period_month", ""),
        )

        if key in index:
            fail(f"Duplicate source row: {key[0]} / {key[1]}")

        index[key] = row

    expected = {
        (store, month)
        for store in STORES
        for month, _ in MONTHS
    }

    if set(index) != expected:
        missing_rows = sorted(expected - set(index))
        extra_rows = sorted(set(index) - expected)

        fail(
            "Unexpected source store-month scope. "
            f"Missing={missing_rows}; Extra={extra_rows}"
        )

    return index


def index_output(
    rows: list[dict[str, str]],
) -> dict[str, dict[str, str]]:
    if len(rows) != len(STORES):
        fail(f"Expected 5 output rows, found {len(rows)}.")

    observed = [
        row.get("store_id", "")
        for row in rows
    ]

    if observed != STORES:
        fail(
            f"Expected output order {STORES}, "
            f"found {observed}."
        )

    return {
        row["store_id"]: row
        for row in rows
    }


def check_fresh_sql(
    committed_fields: list[str],
    committed_rows: list[dict[str, str]],
    fresh_fields: list[str],
    fresh_rows: list[dict[str, str]],
) -> None:
    if committed_fields != fresh_fields:
        fail("Committed CSV header differs from fresh SQL output.")

    if committed_rows != fresh_rows:
        for row_number, (committed, fresh) in enumerate(
            zip(committed_rows, fresh_rows),
            start=2,
        ):
            for field in committed_fields:
                if committed.get(field) != fresh.get(field):
                    fail(
                        "Committed CSV differs from fresh SQL at "
                        f"row {row_number}, field {field}: "
                        f"{committed.get(field)!r} != "
                        f"{fresh.get(field)!r}"
                    )

        fail("Committed CSV differs from fresh SQL output.")


def check_month_values(
    source: dict[tuple[str, str], dict[str, str]],
    output: dict[str, dict[str, str]],
) -> None:
    for store in STORES:
        output_row = output[store]

        if output_row.get("observed_month_count") != "3":
            fail(f"Store {store} observed_month_count is not 3.")

        if (
            output_row.get("repeated_window_summary_flag")
            != "summary_ready_for_descriptive_review"
        ):
            fail(f"Store {store} has an unexpected summary flag.")

        note = output_row.get("summary_boundary_note", "")

        missing_note_parts = [
            part
            for part in REQUIRED_NOTE_PARTS
            if part not in note
        ]

        if missing_note_parts:
            fail(
                f"Store {store} boundary note is missing: "
                + ", ".join(missing_note_parts)
            )

        source_types = {
            source[(store, month)]["store_type"]
            for month, _ in MONTHS
        }

        source_regions = {
            source[(store, month)]["region_type"]
            for month, _ in MONTHS
        }

        if len(source_types) != 1 or len(source_regions) != 1:
            fail(
                f"Store {store} metadata changes across months."
            )

        source_type = next(iter(source_types))
        source_region = next(iter(source_regions))

        if source_type not in ALLOWED_STORE_TYPES:
            fail(
                f"Store {store} has non-canonical "
                f"store_type: {source_type}"
            )

        if output_row.get("store_type") != source_type:
            fail(f"Store {store} store_type differs from source.")

        if output_row.get("region_type") != source_region:
            fail(f"Store {store} region_type differs from source.")

        for month, prefix in MONTHS:
            source_row = source[(store, month)]

            for metric in METRICS:
                assert_number(
                    output_row[f"{prefix}_{metric}"],
                    number(
                        source_row[metric],
                        f"source {store}/{month}/{metric}",
                    ),
                    f"output {store}/{prefix}_{metric}",
                )


def check_endpoint_formulas(
    output: dict[str, dict[str, str]],
) -> None:
    for store in STORES:
        row = output[store]

        for metric in ABSOLUTE_DELTAS:
            feb = number(row[f"feb_{metric}"], f"{store} feb {metric}")
            apr = number(row[f"apr_{metric}"], f"{store} apr {metric}")

            assert_number(
                row[f"{metric}_feb_to_apr_delta"],
                round_two(apr - feb),
                f"{store} {metric}_feb_to_apr_delta",
            )

        for metric in RELATIVE_CHANGES:
            feb = number(row[f"feb_{metric}"], f"{store} feb {metric}")
            apr = number(row[f"apr_{metric}"], f"{store} apr {metric}")

            if feb <= 0:
                if row.get(f"{metric}_feb_to_apr_pct", ""):
                    fail(
                        f"{store} {metric}_feb_to_apr_pct "
                        "should be empty when February is not positive."
                    )
                continue

            expected = round_two(
                (apr - feb) * Decimal("100") / feb
            )

            assert_number(
                row[f"{metric}_feb_to_apr_pct"],
                expected,
                f"{store} {metric}_feb_to_apr_pct",
            )

        for metric in PERCENTAGE_POINT_DELTAS:
            feb = number(row[f"feb_{metric}"], f"{store} feb {metric}")
            apr = number(row[f"apr_{metric}"], f"{store} apr {metric}")

            assert_number(
                row[f"{metric}_feb_to_apr_delta"],
                round_two(apr - feb),
                f"{store} {metric}_feb_to_apr_delta",
            )


def main() -> None:
    for path in [SOURCE, SQL, OUTPUT, DICTIONARY]:
        if not path.exists():
            fail(f"Missing file: {path.relative_to(ROOT)}")

    check_dictionary()
    check_canonical_names()

    source_fields, source_rows = read_csv(SOURCE)
    output_fields, output_rows = read_csv(OUTPUT)

    source_index = index_source(source_fields, source_rows)
    output_index = index_output(output_rows)

    expected_aliases = expected_month_aliases()

    missing_aliases = sorted(
        expected_aliases - set(output_fields)
    )

    if missing_aliases:
        fail(
            "Summary output is missing aliases: "
            + ", ".join(missing_aliases)
        )

    fresh_fields, fresh_rows = execute_sql(
        source_fields,
        source_rows,
    )

    check_fresh_sql(
        output_fields,
        output_rows,
        fresh_fields,
        fresh_rows,
    )

    check_month_values(
        source_index,
        output_index,
    )

    check_endpoint_formulas(
        output_index,
    )

    print("[PASS] Repeated-window summary validation passed.")
    print(
        f"[PASS] All {len(expected_aliases)} aliases for month fields "
        "are registered in the data dictionary."
    )
    print("[PASS] Source scope is Stores B-F across three months.")
    print("[PASS] Committed CSV matches fresh SQL execution.")
    print("[PASS] Monthly values match the canonical source panel.")
    print("[PASS] Endpoint movement formulas are reproducible.")
    print("[PASS] Metadata and interpretation boundaries are preserved.")


if __name__ == "__main__":
    main()
