from __future__ import annotations

import unittest
from pathlib import Path

from retail_ops.ingestion.contracts import (
    BatchContractError,
    build_batch_metadata,
    load_dataset_contracts,
    validate_dataset_contracts,
)


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_SOURCE_PATHS = {
    "retail_ops/data/store_a_monthly_metrics.csv",
    "retail_ops/data/demo2_store_period_metrics.csv",
    "retail_ops/data/store_period_panel_metrics.csv",
    "retail_ops/data/store_a_top_skus.csv",
    "retail_ops/data/demo2_top_skus_by_transaction_amount.csv",
    "retail_ops/data/demo2_top_skus_by_sales_volume.csv",
    "retail_ops/data/demo2_top_search_terms.csv",
}


class RetailDatasetContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contracts = load_dataset_contracts(ROOT)

    def test_registry_covers_current_source_csvs(self) -> None:
        actual_paths = {
            contract.source_path
            for contract in self.contracts.values()
        }

        self.assertEqual(actual_paths, EXPECTED_SOURCE_PATHS)

    def test_contracts_match_csvs_and_declare_overlap(self) -> None:
        overlaps = validate_dataset_contracts(ROOT)
        overlap_by_pair = {
            frozenset(
                {
                    overlap.left_dataset_id,
                    overlap.right_dataset_id,
                }
            ): overlap
            for overlap in overlaps
        }
        expected_pair = frozenset(
            {
                "demo2_store_period_metrics",
                "store_period_panel_metrics",
            }
        )

        self.assertIn(expected_pair, overlap_by_pair)
        overlap = overlap_by_pair[expected_pair]
        self.assertEqual(overlap.record_count, 5)
        self.assertEqual(
            overlap.overlap_policy,
            "keep_versions_do_not_sum",
        )

    def test_store_context_fields_are_dimensions(self) -> None:
        store_period_ids = {
            "store_a_monthly_metrics",
            "demo2_store_period_metrics",
            "store_period_panel_metrics",
        }

        for dataset_id in store_period_ids:
            with self.subTest(dataset_id=dataset_id):
                contract = self.contracts[dataset_id]
                self.assertEqual(
                    contract.dimension_fields,
                    ("region_type", "store_type"),
                )
                self.assertNotIn(
                    "visibility_entry_profile",
                    contract.dimension_fields,
                )

    def test_demo2_sku_ranking_basis_is_not_lost(self) -> None:
        amount_contract = self.contracts[
            "demo2_top_skus_by_transaction_amount"
        ]
        volume_contract = self.contracts[
            "demo2_top_skus_by_sales_volume"
        ]

        self.assertEqual(
            amount_contract.ranking_basis,
            "transaction_amount",
        )
        self.assertEqual(
            volume_contract.ranking_basis,
            "sales_volume",
        )
        self.assertNotEqual(
            amount_contract.ranking_basis,
            volume_contract.ranking_basis,
        )
        self.assertEqual(
            self.contracts["store_a_top_skus"].ranking_basis,
            "not_recorded_in_current_fixture",
        )

    def _valid_batch_payload(self) -> dict[str, object]:
        return {
            "batch_id": "batch_20260904T120000Z_001",
            "dataset_id": "demo2_store_period_metrics",
            "source_system": "meituan_merchant_backend",
            "source_name": "demo2_store_period_metrics",
            "source_page": "test_fixture_source_page",
            "extracted_at": "2026-09-04T12:00:00+00:00",
            "received_at": "2026-09-04T12:05:00+00:00",
            "file_sha256": "a" * 64,
            "mapping_version": "mapping-v1",
            "snapshot_semantics": "cumulative_period_snapshot",
            "coverage_start": "2026-03-01",
            "coverage_end": "2026-03-31",
            "status": "received",
        }

    def test_batch_contract_keeps_required_lineage_metadata(self) -> None:
        batch = build_batch_metadata(
            self._valid_batch_payload(),
            self.contracts,
        )

        self.assertEqual(
            batch.dataset_id,
            "demo2_store_period_metrics",
        )
        self.assertEqual(
            batch.snapshot_semantics,
            "cumulative_period_snapshot",
        )
        self.assertEqual(
            batch.source_page,
            "test_fixture_source_page",
        )

    def test_batch_contract_requires_source_page(self) -> None:
        payload = self._valid_batch_payload()
        payload["source_page"] = ""

        with self.assertRaisesRegex(
            BatchContractError,
            "source_page",
        ):
            build_batch_metadata(payload, self.contracts)

    def test_batch_snapshot_semantics_must_match_dataset(self) -> None:
        payload = self._valid_batch_payload()
        payload["snapshot_semantics"] = (
            "incremental_period_extract"
        )

        with self.assertRaisesRegex(
            BatchContractError,
            "snapshot_semantics",
        ):
            build_batch_metadata(payload, self.contracts)


if __name__ == "__main__":
    unittest.main()
