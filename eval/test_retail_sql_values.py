from __future__ import annotations

import contextlib
import copy
import csv
import io
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from decimal import Decimal, InvalidOperation
from pathlib import Path
from unittest.mock import patch

import duckdb

from retail_ops import sql_runtime as runtime
from retail_ops.scripts import export_demo1_sql_output as demo1_export
from retail_ops.scripts import regenerate_repeated_window_panel_summary as panel_export


ROOT = Path(__file__).resolve().parents[1]
Q1, Q2, Q3, Q4 = runtime.QUERIES
OUTPUTS = ("store_a_demo1_sql_output.csv", "demo2_cross_store_comparability_output.csv",
           "store_period_panel_coverage_output.csv", "repeated_window_panel_summary_output.csv")


class RetailSqlValueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = {name: runtime.read_source(ROOT, name)
                       for _, names in runtime.QUERIES.values() for name in names}

    def tables(self, query):
        return {name: copy.deepcopy(self.sources[name]) for name in runtime.QUERIES[query][1]}

    def execute(self, query, tables):
        engine = runtime.QUERIES[query][0]
        fn = runtime.execute_duckdb if engine == "duckdb" else runtime.execute_sqlite
        fields, rows = fn(tables, runtime.read_query(ROOT, query))
        return [dict(zip(fields, row)) for row in rows]

    def direct_sql(self, query, tables):
        """Check SQL join/aggregation guards even without the source-row precheck."""
        engine = runtime.QUERIES[query][0]
        con = duckdb.connect(":memory:") if engine == "duckdb" else sqlite3.connect(":memory:")
        try:
            (runtime.register_duckdb if engine == "duckdb" else runtime.register_sqlite)(con)
            for name, (fields, rows) in tables.items():
                definitions = ", ".join(f'"{field}" VARCHAR' for field in fields)
                con.execute(f'CREATE TABLE "{name}" ({definitions})')
                marks = ", ".join("?" for _ in fields)
                con.executemany(f'INSERT INTO "{name}" VALUES ({marks})',
                                [[row[field] for field in fields] for row in rows])
            cursor = con.execute(runtime.read_query(ROOT, query))
            fields = [item[0] for item in cursor.description]
            return [dict(zip(fields, row)) for row in cursor.fetchall()]
        finally:
            con.close()

    def test_all_queries_preserve_committed_fixture_fields_and_values(self):
        for query, output in zip(runtime.QUERIES, OUTPUTS):
            with self.subTest(query=query):
                fields, rows = runtime.run_query(query, ROOT)
                with (ROOT / "retail_ops/outputs" / output).open() as handle:
                    committed = list(csv.reader(handle))
                self.assertEqual(fields, committed[0])
                self.assertEqual(len(rows), len(committed) - 1)
                for actual, expected in zip(rows, committed[1:]):
                    for field, value, original in zip(fields, actual, expected):
                        text = demo1_export.format_cell(value)
                        try:
                            equal = Decimal(text) == Decimal(original)
                        except InvalidOperation:
                            equal = text == original
                        self.assertTrue(equal, (query, field, text, original))

    def test_blank_core_value_stays_null_and_zero_stays_zero(self):
        for value in (None, "", " \t ", "0"):
            with self.subTest(value=value):
                tables = self.tables(Q2)
                tables["demo2_store_period_metrics"][1][0]["transaction_amount"] = value
                row = self.execute(Q2, tables)[0]
                self.assertEqual(row["transaction_amount"], 0 if value == "0" else None)
                self.assertIsNone(row["top3_sku_transaction_amount_share_pct"])
                expected = "same_period_diagnostic_ready" if value == "0" else "insufficient_data"
                self.assertEqual(row["comparison_scope_flag"], expected)

    def test_blank_optional_metric_does_not_create_a_new_scope_rule(self):
        tables = self.tables(Q2)
        source = tables["demo2_store_period_metrics"][1][0]
        source["estimated_income_proxy"] = ""
        source["refund_amount"] = ""
        row = self.execute(Q2, tables)[0]
        self.assertIsNone(row["estimated_income_proxy"])
        self.assertIsNone(row["refund_amount"])
        self.assertEqual(row["comparison_scope_flag"], "same_period_diagnostic_ready")

    def test_invalid_source_numbers_stop_before_database_creation(self):
        for field, value in (("transaction_amount", "abc"), ("transaction_amount", "12x"),
                             ("transaction_amount", "NaN"), ("entry_conversion_rate_pct", "12%"),
                             ("transaction_orders", "1.5"), ("transaction_orders", "-1"),
                             ("transaction_orders", True)):
            with self.subTest(field=field, value=value):
                tables = self.tables(Q2)
                tables["demo2_store_period_metrics"][1][0][field] = value
                with patch.object(runtime.sqlite3, "connect") as connect:
                    with self.assertRaises(ValueError):
                        self.execute(Q2, tables)
                    connect.assert_not_called()

    def test_both_sql_functions_reject_invalid_text_instead_of_casting_zero(self):
        for engine in ("sqlite", "duckdb"):
            con = sqlite3.connect(":memory:") if engine == "sqlite" else duckdb.connect(":memory:")
            try:
                (runtime.register_sqlite if engine == "sqlite" else runtime.register_duckdb)(con)
                self.assertEqual(con.execute("SELECT retail_value('transaction_amount', '')").fetchone(), (None,))
                with self.assertRaises((sqlite3.OperationalError, duckdb.InvalidInputException)):
                    con.execute("SELECT retail_value('transaction_amount', 'abc')").fetchall()
            finally:
                con.close()

    def test_top3_missing_amount_is_null_in_both_demos(self):
        for query in (Q1, Q2):
            with self.subTest(query=query):
                tables = self.tables(query)
                sku_table = runtime.QUERIES[query][1][1]
                tables[sku_table][1][0]["sku_transaction_amount"] = ""
                row = self.execute(query, tables)[0]
                self.assertIsNone(row["top3_sku_transaction_amount"])
                self.assertIsNone(row["top3_sku_transaction_amount_share_pct"])
                if query == Q2:
                    self.assertEqual(row["comparison_scope_flag"], "insufficient_data")
                    self.assertIn("missing_top3_sku_amount_evidence", row["comparison_limit_notes"])

    def test_top3_missing_row_and_duplicate_rank_cannot_produce_partial_total(self):
        for query in (Q1, Q2):
            tables = self.tables(query)
            sku_table = runtime.QUERIES[query][1][1]
            tables[sku_table][1].pop(0)
            self.assertIsNone(self.execute(query, tables)[0]["top3_sku_transaction_amount"])
            tables = self.tables(query)
            tables[sku_table][1].append(dict(tables[sku_table][1][0]))
            with self.assertRaisesRegex(ValueError, "duplicate logical key"):
                self.execute(query, tables)
            self.assertIsNone(self.direct_sql(query, tables)[0]["top3_sku_transaction_amount"])

    def test_top3_only_uses_ranks_one_to_three(self):
        for query in (Q1, Q2):
            tables = self.tables(query)
            expected = self.execute(query, tables)[0]["top3_sku_transaction_amount"]
            sku_table = runtime.QUERIES[query][1][1]
            tables[sku_table][1].append(dict(tables[sku_table][1][0], sku_rank="4", sku_transaction_amount="999999"))
            self.assertEqual(self.execute(query, tables)[0]["top3_sku_transaction_amount"], expected)

    def test_top3_sql_join_requires_both_window_dates(self):
        for query in (Q1, Q2):
            tables = self.tables(query)
            sku_table = runtime.QUERIES[query][1][1]
            for row in tables[sku_table][1][:3]:
                row["period_end"] = "2026-03-15"
            self.assertIsNone(self.direct_sql(query, tables)[0]["top3_sku_transaction_amount"])

    def test_coverage_average_requires_every_included_value(self):
        tables = self.tables(Q3)
        tables["store_period_panel_metrics"][1][0]["transaction_amount"] = ""
        row = self.execute(Q3, tables)[0]
        self.assertIsNone(row["avg_transaction_amount"])
        self.assertIsNotNone(row["avg_entry_users"])
        self.assertEqual(row["observed_month_count"], 3)
        for source in tables["store_period_panel_metrics"][1]:
            if source["store_id"] == "B":
                source["transaction_amount"] = "0"
        self.assertEqual(self.execute(Q3, tables)[0]["avg_transaction_amount"], 0)

    def test_monthly_null_and_endpoint_delta_remain_null(self):
        tables = self.tables(Q4)
        tables["store_period_panel_metrics"][1][0]["transaction_amount"] = ""
        row = self.execute(Q4, tables)[0]
        self.assertIsNone(row["feb_transaction_amount"])
        self.assertIsNone(row["transaction_amount_feb_to_apr_delta"])
        self.assertIsNone(row["transaction_amount_feb_to_apr_pct"])
        self.assertIsNotNone(row["mar_transaction_amount"])

    def test_three_different_months_do_not_claim_february_to_april_coverage(self):
        tables = self.tables(Q4)
        rows = tables["store_period_panel_metrics"][1]
        target = next(row for row in rows if row["store_id"] == "B" and row["period_month"] == "2026-03")
        target.update(period_month="2026-01", period_start="2026-01-01", period_end="2026-01-31")
        row = self.execute(Q4, tables)[0]
        self.assertEqual(row["observed_month_count"], 3)
        self.assertEqual(row["repeated_window_summary_flag"], "insufficient_repeated_window_coverage")
        self.assertIsNone(row["mar_transaction_amount"])

    def test_conflicting_context_is_not_selected_using_max(self):
        tables = self.tables(Q4)
        tables["store_period_panel_metrics"][1][0].update(region_type="Yantai", store_type="partner")
        row = self.execute(Q4, tables)[0]
        self.assertIsNone(row["region_type"])
        self.assertIsNone(row["store_type"])

    def test_latest_observation_flag_is_null_when_required_values_are_missing(self):
        for month, field in (("2026-04", "order_conversion_rate_pct"), ("2026-03", "average_order_value")):
            tables = self.tables(Q1)
            target = next(row for row in tables["store_a_monthly_metrics"][1] if row["period_month"] == month)
            target[field] = ""
            self.assertIsNone(self.execute(Q1, tables)[-1]["transaction_recovered_with_conversion_aov_tradeoff"])

    def test_source_keys_dates_and_duplicate_versions_are_checked(self):
        for changes in ({"store_id": ""}, {"period_month": "2026-07"},
                        {"period_start": "2026-02-15"}, {"period_end": ""}):
            tables = self.tables(Q4)
            tables["store_period_panel_metrics"][1][0].update(changes)
            with self.assertRaises(ValueError):
                self.execute(Q4, tables)
        tables = self.tables(Q4)
        tables["store_period_panel_metrics"][1].append(dict(tables["store_period_panel_metrics"][1][0]))
        with self.assertRaisesRegex(ValueError, "duplicate logical key"):
            self.execute(Q4, tables)

    def test_unknown_dataset_column_or_query_has_no_fallback(self):
        fields, rows = copy.deepcopy(self.sources["store_period_panel_metrics"])
        with self.assertRaises(ValueError):
            runtime.prepare_rows("new_report", fields, rows)
        with self.assertRaises(ValueError):
            runtime.prepare_rows("store_period_panel_metrics", [*fields, "new_metric"], rows)
        with self.assertRaises(ValueError):
            runtime.prepare_rows("store_period_panel_metrics", [*fields, fields[0]], rows)
        with self.assertRaises(ValueError):
            runtime.run_query("../new_report.sql", ROOT)

    def test_registered_source_contract_cannot_change_sql_destination_or_keys(self):
        name = "demo2_top_skus_by_transaction_amount"
        contracts = runtime.preview.load_dataset_contracts(ROOT)
        for changes in ({"source_path": "retail_ops/data/demo2_top_skus_by_sales_volume.csv"},
                        {"ranking_basis": "sales_volume"},
                        {"key_fields": ("store_id", "period_start", "period_end")}):
            changed = dict(contracts, **{name: replace(contracts[name], **changes)})
            with patch.object(runtime.preview, "load_dataset_contracts", return_value=changed):
                with self.assertRaises(ValueError):
                    runtime.read_source(ROOT, name)

    def test_sql_numeric_ranges_fail_without_saturation_or_underflow_to_zero(self):
        for field, value in (("transaction_orders", str(2**63)), ("transaction_amount", "9" * 400),
                             ("transaction_amount", "0." + "0" * 400 + "1")):
            with self.assertRaises(ValueError):
                runtime.numeric_text(field, value)
        self.assertEqual(runtime.numeric_text("transaction_orders", "0"), "0")
        self.assertEqual(runtime.numeric_text("estimated_income_proxy", "8078.26"), "8078.26")

    def test_existing_exporters_preserve_fixture_bytes(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            for module, name, function in ((demo1_export, OUTPUTS[0], demo1_export.export_csv),
                                            (panel_export, OUTPUTS[3], panel_export.main)):
                target = Path(tmp) / name
                with patch.object(module, "OUTPUT_PATH", target):
                    function()
                self.assertEqual(target.read_bytes(), (ROOT / "retail_ops/outputs" / name).read_bytes())

    def test_invalid_source_does_not_replace_previous_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "existing.csv"
            target.write_bytes(b"previous output\n")
            fields, rows = copy.deepcopy(self.sources["store_period_panel_metrics"])
            rows[0]["transaction_orders"] = "abc"
            with patch.object(panel_export, "OUTPUT_PATH", target), patch.object(panel_export, "read_source", return_value=(fields, rows)):
                with self.assertRaises(ValueError):
                    panel_export.main()
            self.assertEqual(target.read_bytes(), b"previous output\n")

    def test_cli_runs_registered_query_and_reports_summary(self):
        completed = subprocess.run([sys.executable, "-m", "retail_ops.sql_runtime", "--query", Q2, "--summary"],
                                   cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('"rows": 5', completed.stdout)
        self.assertIn('"columns": 46', completed.stdout)


if __name__ == "__main__":
    unittest.main()
