from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

from retail_ops.ingestion import text_preview as source
from retail_ops.ingestion.preview import preview_json


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "店铺：B\n时间范围2026.3\n"
# Store amounts are an excerpt of the supplied March notes. Product names below
# are test fixtures and are never written into retail_ops/data.
TEXT = PREFIX + "成交金额11665.5\n成交订单量299\n入店转化率15.12%\n区域类型青岛\n店铺类型自营\n"
SEARCH = "本店top3搜索词:护理液（曝光次数512，点击次数36，成单次数15），美瞳（曝光次数83，点击次数5，成单次数0），日抛（曝光次数102，点击次数4，成单次数0）\n"
VOLUME = "商品销量top3交易商品：示例商品甲（颜色随机）（销量45），示例商品乙（销量33），示例商品乙（销量20）\n"
AMOUNT = "top3交易商品成交金额：示例商品甲678.45元，示例商品乙320.95元，示例商品丙301.5元\n"


class RetailTextPreviewTests(unittest.TestCase):
    def run_preview(self, text=TEXT, **kwargs):
        return source.preview_text(ROOT, text.encode(), "manual_text_v1", **kwargs)

    def store_record(self, result, index=0):
        return result["blocks"][index]["groups"][0]["candidate_records"][0]["record"]

    def assert_held(self, result):
        self.assertEqual(result["status"], "quarantined")
        self.assertEqual(result["validated_records"], [])

    def test_store_fields_follow_dictionary_and_source_values(self):
        result = self.run_preview(TEXT + "活动营业总额13938.4\n投入产出比24.12%\n退款订单数（全部退款）23\n退款订单数（全部退款+部分退款）24\n预计收入8078.26\n")
        self.assertEqual(result["status"], "validated", result)
        row = self.store_record(result)
        self.assertEqual(row["transaction_amount"], Decimal("11665.5"))
        self.assertEqual(row["transaction_orders"], 299)
        self.assertEqual(row["activity_original_transaction_amount"], Decimal("13938.4"))
        self.assertEqual(row["activity_cost_ratio_pct"], Decimal("24.12"))
        self.assertEqual(row["full_refund_orders"], 23)
        self.assertEqual(row["refund_orders_all_or_partial"], 24)
        self.assertEqual(row["estimated_income_proxy"], Decimal("8078.26"))
        self.assertEqual((row["region_type"], row["store_type"]), ("Qingdao", "self-operated"))

    def test_blank_metrics_stay_null_and_explicit_zero_stays_zero(self):
        result = self.run_preview(PREFIX + "成交金额\n成交订单量0\n入店转化率\n")
        self.assertEqual(result["status"], "validated")
        row = self.store_record(result)
        self.assertIsNone(row["transaction_amount"])
        self.assertIsNone(row["entry_conversion_rate_pct"])
        self.assertIsNone(row["exposure_users"])
        self.assertEqual(row["transaction_orders"], 0)

    def test_omitted_units_are_held_and_explicit_user_counts_are_mapped(self):
        result = self.run_preview(TEXT + "搜索曝光4390\n搜索入店683\n")
        self.assert_held(result)
        self.assertIsNone(self.store_record(result)["search_exposure_users"])
        self.assertIsNone(self.store_record(result)["search_entry_users"])
        self.assertEqual(len(result["blocks"][0]["issues"]), 2)
        explicit = self.run_preview(TEXT + "搜索曝光人数4390\n搜索入店人数683\n")
        self.assertEqual(explicit["status"], "validated")
        self.assertEqual(self.store_record(explicit)["search_exposure_users"], 4390)

    def test_unknown_section_stops_later_values_from_inheriting_store_scope(self):
        result = self.run_preview(TEXT + "新增商品报表：\n支付金额999\n" + SEARCH)
        self.assert_held(result)
        self.assertIsNone(self.store_record(result)["payment_amount"])
        self.assertEqual(len(result["blocks"][0]["groups"]), 1)
        self.assertEqual(len(result["blocks"][0]["unmapped_lines"]), 2)

    def test_similar_labels_and_unregistered_values_have_no_fallback(self):
        for line in ("成交金额变化99", "店铺曝光人次100", "区域类型新城市", "店铺类型新模式", "区域类型分析："):
            with self.subTest(line=line):
                result = self.run_preview(TEXT + line + "\n支付金额999\n")
                self.assert_held(result)
                self.assertIsNone(self.store_record(result)["payment_amount"])

    def test_bad_number_or_unit_is_not_replaced_by_zero(self):
        for line in ("成交金额:abc", "成交金额NaN", "成交金额1e309", "成交订单量1.5", "成交订单量-1", "入店转化率15.12"):
            with self.subTest(line=line):
                result = self.run_preview(PREFIX + line + "\n支付金额999\n")
                self.assert_held(result)
                self.assertIsNone(self.store_record(result)["payment_amount"])

    def test_unknown_store_and_upload_scope_conflicts_are_held(self):
        for text, stores in ((TEXT.replace("店铺：B", "店铺：Z"), None), (TEXT, ["C"]), (TEXT, ["Z"]), (TEXT, "B")):
            result = self.run_preview(text, expected_store_ids=stores)
            self.assert_held(result)
            self.assertFalse(any(block["groups"] for block in result["blocks"]))
        self.assertEqual(self.run_preview(expected_store_ids=["B"])["status"], "validated")

    def test_current_month_windows_are_not_reinterpreted(self):
        for raw, start, end in (("2026.3", "2026-03-01", "2026-03-31"),
                                ("2026.2.1-2.28", "2026-02-01", "2026-02-28"),
                                ("2026-04", "2026-04-01", "2026-04-30")):
            result = self.run_preview(TEXT.replace("2026.3", raw))
            self.assertEqual(result["status"], "validated")
            self.assertEqual((self.store_record(result)["period_start"], self.store_record(result)["period_end"]), (start, end))
        for raw in ("2026.2.15-3.15", "2026.3.31-3.1", "2026.2.1-2.30", "2026-03-15"):
            self.assert_held(self.run_preview(TEXT.replace("2026.3", raw)))

    def test_missing_late_or_repeated_window_does_not_relabel_data(self):
        for text in ("店铺：B\n成交金额100\n时间范围2026.3\n", TEXT + "时间范围2026.4\n支付金额123\n", "店铺：B\n"):
            self.assert_held(self.run_preview(text))

    def test_multiple_store_blocks_keep_their_own_scope(self):
        text = TEXT + "店铺：C\n时间范围2026.4.1-4.30\n成交金额6756.8\n"
        result = self.run_preview(text)
        self.assertEqual(result["status"], "validated")
        first, second = self.store_record(result), self.store_record(result, 1)
        self.assertEqual((first["store_id"], first["period_month"]), ("B", "2026-03"))
        self.assertEqual((second["store_id"], second["period_month"]), ("C", "2026-04"))
        self.assertEqual(second["transaction_amount"], Decimal("6756.8"))
        self.assertIsNone(second["transaction_orders"])

    def test_search_and_two_sku_rankings_stay_separate(self):
        result = self.run_preview(TEXT + SEARCH + VOLUME + AMOUNT)
        self.assertEqual(result["status"], "validated", result)
        groups = result["blocks"][0]["groups"]
        self.assertEqual(len(groups), 4)
        search, volume, amount = groups[1:]
        self.assertEqual(search["context"]["grain"], "store_search_term_period")
        self.assertEqual(search["candidate_records"][1]["record"]["search_term_order_times"], 0)
        self.assertEqual(volume["context"]["ranking_basis"], "sales_volume")
        self.assertEqual(amount["context"]["ranking_basis"], "transaction_amount")
        self.assertIsNone(volume["candidate_records"][0]["record"]["sku_transaction_amount"])
        self.assertIsNone(amount["candidate_records"][0]["record"]["sales_volume"])
        self.assertEqual(self.store_record(result)["transaction_amount"], Decimal("11665.5"))
        self.assertEqual(volume["candidate_records"][1]["record"]["sku_name"], volume["candidate_records"][2]["record"]["sku_name"])
        self.assertNotEqual(volume["candidate_records"][1]["record"]["sku_rank"], volume["candidate_records"][2]["record"]["sku_rank"])

    def test_missing_list_values_and_labelled_digit_ending_names(self):
        volume = VOLUME.replace("销量45", "销量")
        amount = "top3交易商品成交金额：示例商品2026（成交金额100元），示例商品乙（成交金额），示例商品丙（成交金额0元）\n"
        result = self.run_preview(TEXT + volume + amount)
        self.assertEqual(result["status"], "validated", result)
        groups = result["blocks"][0]["groups"]
        self.assertIsNone(groups[1]["candidate_records"][0]["record"]["sales_volume"])
        self.assertEqual(groups[2]["candidate_records"][0]["record"]["sku_name"], "示例商品2026")
        self.assertIsNone(groups[2]["candidate_records"][1]["record"]["sku_transaction_amount"])

    def test_incomplete_malformed_and_repeated_lists_are_held(self):
        for body in (SEARCH.rsplit("，日抛", 1)[0] + "\n", VOLUME.replace("销量45", "订单数45"),
                     VOLUME.replace("（颜色随机）", "（颜色随机"), AMOUNT + AMOUNT):
            self.assert_held(self.run_preview(TEXT + body))

    def test_duplicate_metrics_and_repeated_snapshots_do_not_choose_a_value(self):
        self.assert_held(self.run_preview(TEXT + "成交金额999\n"))
        self.assert_held(self.run_preview(TEXT + TEXT.replace("11665.5", "10")))

    def test_excluded_lines_leave_no_values_or_retention_entries(self):
        for suffix in ("", "新报表：\n"):
            result = self.run_preview(TEXT + suffix + "有效订单数998877\n无效订单数887766\n")
            serialized = preview_json(result)
            for forbidden in ("有效订单数", "无效订单数", "998877", "887766"):
                self.assertNotIn(forbidden, serialized)

    def test_lineage_quotes_and_raw_file_hash_refer_to_original_text(self):
        text = TEXT + "有效订单数998877\n" + SEARCH + VOLUME + AMOUNT
        result = self.run_preview(text)
        self.assertEqual(result["file_sha256"], source.hashlib.sha256(text.encode()).hexdigest())
        lines = text.splitlines()
        for group in result["blocks"][0]["groups"]:
            for entry in group["candidate_records"]:
                for evidence in entry["lineage"]:
                    self.assertIn(evidence["source_text"], lines[evidence["source_line"] - 1])

    def test_model_proposals_are_checked_against_independent_raw_source(self):
        result = self.run_preview(TEXT + SEARCH)
        proposals = [{"dataset_id": item["context"]["dataset_id"], "grain": item["context"]["grain"],
                      "ranking_basis": item["context"]["ranking_basis"], "record": item["record"]}
                     for item in result["validated_records"]]
        self.assertEqual(self.run_preview(TEXT + SEARCH, proposals=proposals)["status"], "validated")
        proposals[0]["record"] = dict(proposals[0]["record"], transaction_amount=Decimal("999"))
        self.assert_held(self.run_preview(TEXT + SEARCH, proposals=proposals))
        self.assert_held(self.run_preview(proposals=[]))

    def test_unknown_source_profiles_and_new_parser_types_have_no_fallback(self):
        result = source.preview_text(ROOT, TEXT.encode(), "new_report_v1")
        self.assert_held(result)
        self.assertEqual(result["blocks"], [])
        raw = (ROOT / source.PROFILE_PATH).read_bytes()
        profile = json.loads(raw)
        profile["sections"][0]["parser"] = "future_report"
        with patch.object(Path, "read_bytes", return_value=json.dumps(profile).encode()):
            self.assert_held(self.run_preview())

    def test_cli_produces_preview_and_does_not_modify_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notes.txt"
            data = TEXT.encode()
            path.write_bytes(data)
            command = [sys.executable, "-m", "retail_ops.ingestion.text_preview", "--input", str(path), "--profile", "manual_text_v1"]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["status"], "validated")
            result = subprocess.run(command + ["--expected-store-id", "C"], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(json.loads(result.stdout)["validated_records"], [])
            result = subprocess.run(command + ["--summary"], cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads(result.stdout)
            self.assertEqual(summary["store_period_blocks"], 1)
            self.assertEqual(summary["candidate_records"], 1)
            self.assertEqual(summary["validated_records"], 1)
            self.assertEqual(path.read_bytes(), data)


if __name__ == "__main__":
    unittest.main()
