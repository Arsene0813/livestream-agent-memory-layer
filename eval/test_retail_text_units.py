from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from retail_ops.ingestion import text_preview as source


ROOT = Path(__file__).resolve().parents[1]
PREFIX = "店铺：B\n时间范围2026.3\n"
# Verbatim metric values from the supplied B-store March notes.
ALIASES = (
    "搜索曝光4390\n搜索入店683\n商家列表曝光400\n商家列表入店7\n"
    "活动专区曝光8\n活动专区入店 3\n订单页入店46\n其他入店104\n"
)
SEARCH = (
    "本店top3搜索词:隐形眼镜护理液（曝光次数512，点击次数36，成单次数15），"
    "美瞳护理液（曝光次数231，点击次数27，成单次数6），"
    "美瞳日抛（曝光次数231，点击次数28，成单次数11）\n"
)
FIELDS = (
    "search_exposure_users", "search_entry_users", "merchant_list_exposure_users",
    "merchant_list_entry_users", "activity_zone_exposure_users", "activity_zone_entry_users",
    "order_page_entry_users", "other_entry_users",
)


class RetailTextUnitTests(unittest.TestCase):
    def run_preview(self, text=PREFIX + ALIASES, version="manual_text_v2", **kwargs):
        return source.preview_text(ROOT, text.encode(), version, **kwargs)

    def row(self, result):
        return result["blocks"][0]["groups"][0]["candidate_records"][0]["record"]

    def assert_held(self, result):
        self.assertEqual(result["status"], "quarantined")
        self.assertEqual(result["validated_records"], [])

    def test_abbreviations_match_existing_csv_fields_and_source_lines(self):
        result = self.run_preview()
        self.assertEqual(result["status"], "validated", result)
        for name in ("store_period_panel_metrics.csv", "demo2_store_period_metrics.csv"):
            with (ROOT / "retail_ops/data" / name).open() as handle:
                expected = next(row for row in csv.DictReader(handle)
                                if row["store_id"] == "B" and row["period_start"] == "2026-03-01")
            self.assertEqual({field: self.row(result)[field] for field in FIELDS},
                             {field: int(expected[field]) for field in FIELDS})
        entry = result["validated_records"][0]
        self.assertEqual(entry["context"]["grain"], "store_period")
        lines = (PREFIX + ALIASES).splitlines()
        for evidence in entry["lineage"]:
            self.assertEqual(evidence["source_text"], lines[evidence["source_line"] - 1].strip())

    def test_store_user_counts_and_search_term_times_remain_distinct(self):
        text = PREFIX + "店铺曝光人数5094\n店铺曝光次数24196\n入店人数770\n入店次数2676\n" + ALIASES + SEARCH
        result = self.run_preview(text)
        self.assertEqual(result["status"], "validated", result)
        row = self.row(result)
        self.assertEqual((row["exposure_users"], row["exposure_times"],
                          row["entry_users"], row["entry_times"], row["search_exposure_users"]),
                         (5094, 24196, 770, 2676, 4390))
        term_group = result["blocks"][0]["groups"][1]
        self.assertEqual(term_group["context"]["grain"], "store_search_term_period")
        first = term_group["candidate_records"][0]["record"]
        self.assertEqual((first["search_term"], first["search_term_exposure_times"]), ("隐形眼镜护理液", 512))
        self.assertNotIn("search_term_exposure_times", row)
        self.assertNotIn("search_exposure_users", first)

    def test_missing_counts_are_null_and_explicit_zero_is_preserved(self):
        result = self.run_preview(PREFIX + "搜索曝光\n搜索入店0\n商家列表入店\n其他入店0\n")
        self.assertEqual(result["status"], "validated", result)
        row = self.row(result)
        self.assertIsNone(row["search_exposure_users"])
        self.assertIsNone(row["merchant_list_entry_users"])
        self.assertIsNone(row["activity_zone_entry_users"])
        self.assertEqual(row["search_entry_users"], 0)
        self.assertEqual(row["other_entry_users"], 0)

    def test_explicit_times_or_unregistered_labels_cannot_use_user_aliases(self):
        for label in ("搜索曝光次数", "搜索入店次数", "搜索曝光人次", "活动专区曝光变化", "其他入店来源"):
            with self.subTest(label=label):
                result = self.run_preview(PREFIX + label + "100\n支付金额999\n")
                self.assert_held(result)
                self.assertIsNone(self.row(result)["search_exposure_users"])
                self.assertIsNone(self.row(result)["search_entry_users"])
                self.assertIsNone(self.row(result)["payment_amount"])

    def test_invalid_count_does_not_become_zero(self):
        for value in ("1.5", "-1", "abc", "NaN"):
            with self.subTest(value=value):
                result = self.run_preview(PREFIX + "搜索入店：" + value + "\n支付金额999\n")
                self.assert_held(result)
                self.assertIsNone(self.row(result)["search_entry_users"])
                self.assertIsNone(self.row(result)["payment_amount"])

    def test_short_and_full_labels_cannot_overwrite_each_other(self):
        for suffix in ("搜索曝光人数4390\n", "搜索曝光人数999\n", "搜索入店人数683\n"):
            with self.subTest(suffix=suffix):
                self.assert_held(self.run_preview(PREFIX + ALIASES + suffix))

    def test_prior_profile_stays_reproducible_and_version_is_explicit(self):
        old = self.run_preview(version="manual_text_v1")
        new = self.run_preview()
        self.assert_held(old)
        self.assertEqual(len(old["blocks"][0]["issues"]), 8)
        self.assertEqual(new["status"], "validated")
        self.assertEqual(old["file_sha256"], new["file_sha256"])
        self.assertNotEqual(old["source_profile_sha256"], new["source_profile_sha256"])
        self.assertEqual(new["mapping_version"], "manual_text_v2")
        self.assert_held(self.run_preview(version="manual_text_v3"))
        self.assert_held(self.run_preview(version="../../manual_text.v2.json"))

    def test_profile_file_must_match_requested_version(self):
        profile = json.loads((ROOT / source.PROFILE_PATHS["manual_text_v2"]).read_bytes())
        profile["mapping_version"] = "manual_text_v1"
        with patch.object(Path, "read_bytes", return_value=json.dumps(profile).encode()):
            result = self.run_preview()
        self.assert_held(result)
        self.assertEqual(result["blocks"], [])

    def test_model_cannot_substitute_search_term_times_for_store_users(self):
        text = PREFIX + ALIASES + SEARCH
        result = self.run_preview(text)
        proposals = [{"dataset_id": item["context"]["dataset_id"], "grain": item["context"]["grain"],
                      "ranking_basis": item["context"]["ranking_basis"], "record": dict(item["record"])}
                     for item in result["validated_records"]]
        self.assertEqual(self.run_preview(text, proposals=proposals)["status"], "validated")
        proposals[0]["record"]["search_exposure_users"] = 512
        self.assert_held(self.run_preview(text, proposals=proposals))

    def test_cli_uses_selected_profile_and_keeps_source_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notes.txt"
            data = (PREFIX + ALIASES + SEARCH).encode()
            path.write_bytes(data)
            command = [sys.executable, "-m", "retail_ops.ingestion.text_preview", "--input", str(path),
                       "--profile", "manual_text_v2", "--summary"]
            completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads(completed.stdout)
            self.assertEqual(summary["status"], "validated")
            self.assertEqual(summary["candidate_records"], 4)
            self.assertEqual(summary["validated_records"], 4)
            self.assertEqual(summary["issues"], {})
            self.assertEqual(summary["file_sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual(path.read_bytes(), data)


if __name__ == "__main__":
    unittest.main()
