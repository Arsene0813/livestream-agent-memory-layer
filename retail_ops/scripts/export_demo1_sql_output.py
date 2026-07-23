"""Export the Demo 1 SQL result with stable CSV serialization."""

from __future__ import annotations

import csv
import math
import os
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import duckdb


REPO_ROOT = Path(__file__).resolve().parents[2]

SQL_PATH = (
    REPO_ROOT
    / "retail_ops"
    / "sql"
    / "01_store_a_month_over_month_diagnostic.sql"
)

OUTPUT_PATH = (
    REPO_ROOT
    / "retail_ops"
    / "outputs"
    / "store_a_demo1_sql_output.csv"
)


def format_decimal(value: Decimal) -> str:
    text = format(value, "f")

    if "." in text:
        text = text.rstrip("0").rstrip(".")

    return text or "0"


def format_cell(value: Any) -> str:
    """Serialize one SQL value without changing its meaning."""

    if value is None:
        return ""

    if isinstance(value, bool):
        return "true" if value else "false"

    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(
                f"Non-finite numeric value cannot enter CSV: {value}"
            )

        if value.is_integer():
            return str(int(value))

        return str(value)

    if isinstance(value, Decimal):
        return format_decimal(value)

    if isinstance(value, (date, datetime)):
        return value.isoformat()

    return str(value)


def load_query() -> str:
    sql = SQL_PATH.read_text(encoding="utf-8").strip()

    if sql.endswith(";"):
        sql = sql[:-1]

    if not sql:
        raise ValueError("Demo 1 SQL file is empty.")

    return sql


def export_csv() -> None:
    os.chdir(REPO_ROOT)

    sql = load_query()
    connection = duckdb.connect(database=":memory:")

    try:
        cursor = connection.execute(sql)
        columns = [
            description[0]
            for description in cursor.description
        ]
        rows = cursor.fetchall()
    finally:
        connection.close()

    if len(columns) != len(set(columns)):
        raise ValueError(
            "Demo 1 SQL output contains duplicate column names."
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = OUTPUT_PATH.with_suffix(".csv.tmp")

    try:
        with temporary_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as output_file:
            writer = csv.writer(
                output_file,
                lineterminator="\n",
            )

            writer.writerow(columns)

            for row in rows:
                writer.writerow(
                    [
                        format_cell(value)
                        for value in row
                    ]
                )

        temporary_path.replace(OUTPUT_PATH)

    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"Generated: {OUTPUT_PATH}")


if __name__ == "__main__":
    export_csv()