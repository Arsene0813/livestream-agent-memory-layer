"""Run registered retail queries on checked canonical source rows in memory."""

from __future__ import annotations

import argparse
import calendar
import csv
import io
import json
import math
import sqlite3
import sys
from decimal import Decimal
from pathlib import Path

from .ingestion import preview


ROOT = Path(__file__).resolve().parents[1]
QUERIES = {
    "01_store_a_month_over_month_diagnostic.sql":
        ("duckdb", ("store_a_monthly_metrics", "store_a_top_skus")),
    "02_demo2_cross_store_comparability.sql":
        ("sqlite", ("demo2_store_period_metrics", "demo2_top_skus_by_transaction_amount")),
    "03_store_period_panel_coverage.sql": ("sqlite", ("store_period_panel_metrics",)),
    "04_repeated_window_panel_summary.sql": ("sqlite", ("store_period_panel_metrics",)),
}


def numeric_text(field, value):
    """Validate by canonical field, then return exact text or SQL NULL."""
    if field not in preview.COUNTS | preview.RANKS | preview.DECIMALS:
        raise ValueError(f"unregistered SQL numeric field: {field}")
    number = preview._value(field, value)
    if number is None:
        return None
    if isinstance(number, int):
        if number > 2**63 - 1:
            raise ValueError(f"{field}: exceeds SQL integer range")
        return str(number)
    floating = float(number)
    if not math.isfinite(floating) or (number != 0 and floating == 0):
        raise ValueError(f"{field}: outside SQL floating-point range")
    return format(number, "f")


def prepare_rows(dataset_id, fields, rows):
    """Check one already identified canonical table before SQL execution."""
    if dataset_id not in preview.SCHEMAS:
        raise ValueError(f"unregistered SQL source dataset: {dataset_id}")
    schema = preview.SCHEMAS[dataset_id]
    fields = [field for field in fields if field not in preview.IGNORED]
    if not fields or len(fields) != len(set(fields)) or set(fields) - schema:
        raise ValueError(f"{dataset_id}: duplicate or unregistered source columns")
    if not preview.KEYS <= set(fields):
        raise ValueError(f"{dataset_id}: source requires store and complete period metadata")
    grain = preview.ROUTES[dataset_id][0]
    keys = ["store_id", "period_start", "period_end"]
    if grain == "store_sku_period":
        keys.append("sku_rank")
    elif grain == "store_search_term_period":
        keys.append("search_term_rank")
    seen, checked = set(), []
    for index, raw in enumerate(rows, 1):
        if not isinstance(raw, dict) or set(raw) - set(fields) - preview.IGNORED:
            raise ValueError(f"{dataset_id} row {index}: unexpected source cells")
        try:
            values = {field: preview._value(field, raw.get(field)) for field in schema}
            if any(values[field] is None for field in [*keys, "period_month"]):
                raise ValueError("missing logical key or month label")
            start = preview._date(values["period_start"])
            end = preview._date(values["period_end"])
            if start.day != 1 or end != start.replace(day=calendar.monthrange(start.year, start.month)[1]):
                raise ValueError("current SQL sources require complete calendar-month windows")
            if values["period_month"] != values["period_start"][:7]:
                raise ValueError("period_month conflicts with reporting-window dates")
            key = tuple(values[field] for field in keys)
            if key in seen:
                raise ValueError("duplicate logical key; select a source version before SQL")
            seen.add(key)
            checked.append({field: numeric_text(field, value)
                            if field in preview.COUNTS | preview.RANKS | preview.DECIMALS
                            else value for field, value in values.items()})
        except (ValueError, ArithmeticError) as exc:
            raise ValueError(f"{dataset_id} row {index}: {exc}") from exc
    return checked


def read_source(root, dataset_id):
    contracts = preview.load_dataset_contracts(root)
    contract = contracts.get(dataset_id)
    expected_path = f"retail_ops/data/{dataset_id}.csv"
    if (contract is None or dataset_id not in preview.SCHEMAS
            or contract.source_path != expected_path
            or (contract.grain, contract.ranking_basis) != preview.ROUTES[dataset_id]):
        raise ValueError(f"{dataset_id}: source path, schema and route require a registered match")
    keys = ("store_id", "period_start", "period_end")
    if contract.grain == "store_sku_period":
        keys += ("sku_rank",)
    elif contract.grain == "store_search_term_period":
        keys += ("search_term_rank",)
    if contract.key_fields != keys:
        raise ValueError(f"{dataset_id}: source key contract changed; review before SQL")
    data = (root / expected_path).read_text(encoding="utf-8-sig")
    reader = csv.reader(io.StringIO(data, newline=""), strict=True)
    header = next(reader, [])
    if not header or len(header) != len(set(header)):
        raise ValueError(f"{dataset_id}: empty or duplicate CSV header")
    rows = []
    for cells in reader:
        if not cells:
            continue
        if len(cells) != len(header):
            raise ValueError(f"{dataset_id}: wrong cell count at source line {reader.line_num}")
        rows.append(dict(zip(header, cells)))
    return header, rows


def read_query(root, query_name):
    if query_name not in QUERIES:
        raise ValueError("query is not registered")
    text = (root / "retail_ops/sql" / query_name).read_text(encoding="utf-8")
    return "\n".join(line for line in text.splitlines()
                     if line.strip() not in {".mode csv", ".headers on"})


def register_sqlite(connection):
    connection.create_function("retail_value", 2, numeric_text, deterministic=True)


def register_duckdb(connection):
    connection.create_function("retail_value", numeric_text, ["VARCHAR", "VARCHAR"],
                               "VARCHAR", null_handling="special")


def _execute(engine, sources, sql):
    # Validate every row before opening the in-memory database. Source tables
    # are explicitly identified by the caller; this is not an upload router.
    prepared = {name: prepare_rows(name, fields, rows) for name, (fields, rows) in sources.items()}
    if engine == "sqlite":
        connection = sqlite3.connect(":memory:")
    elif engine == "duckdb":
        import duckdb
        connection = duckdb.connect(database=":memory:")
    else:
        raise ValueError("SQL engine is not registered")
    try:
        (register_sqlite if engine == "sqlite" else register_duckdb)(connection)
        for name, rows in prepared.items():
            fields = sorted(preview.SCHEMAS[name])
            definitions = ", ".join(f'"{field}" VARCHAR' for field in fields)
            connection.execute(f'CREATE TABLE "{name}" ({definitions})')
            if rows:
                placeholders = ", ".join("?" for _ in fields)
                connection.executemany(f'INSERT INTO "{name}" VALUES ({placeholders})',
                                       [[row[field] for field in fields] for row in rows])
        cursor = connection.execute(sql)
        columns = [item[0] for item in cursor.description or []]
        records = cursor.fetchall()
        if len(columns) != len(set(columns)):
            raise ValueError("SQL output has duplicate field names")
        for row in records:
            if any(isinstance(value, (float, Decimal)) and not math.isfinite(value) for value in row):
                raise ValueError("SQL output contains a non-finite result")
        return columns, records
    finally:
        connection.close()


def execute_sqlite(sources, sql):
    return _execute("sqlite", sources, sql)


def execute_duckdb(sources, sql):
    return _execute("duckdb", sources, sql)


def run_query(query_name, root=ROOT):
    if query_name not in QUERIES:
        raise ValueError("query is not registered")
    engine, datasets = QUERIES[query_name]
    sources = {name: read_source(root, name) for name in datasets}
    return _execute(engine, sources, read_query(root, query_name))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", choices=QUERIES, required=True)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    try:
        columns, rows = run_query(args.query)
    except Exception as exc:
        parser.exit(2, f"Cannot run retail SQL: {exc}\n")
    if args.summary:
        print(json.dumps({"query": args.query, "engine": QUERIES[args.query][0],
                          "rows": len(rows), "columns": len(columns)}, indent=2))
    else:
        writer = csv.writer(sys.stdout, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
