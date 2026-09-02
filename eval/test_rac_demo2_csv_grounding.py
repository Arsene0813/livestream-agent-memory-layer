from __future__ import annotations

import unittest
from pathlib import Path

from rac.src.demo2_csv_grounding import (
    FACTOR_RECORD_SPECS,
    PERIOD_MONTH,
    STORE_IDS,
    resolve_demo2_record,
)


ROOT = Path(__file__).resolve().parents[1]


class Demo2CsvGroundingTests(unittest.TestCase):
    def test_declared_factors_select_five_records(
        self,
    ) -> None:
        packet = {
            "evidence_id": (
                "test_demo2_csv_grounding"
            ),
            "limitations": [],
        }

        for (
            factor_id,
            spec,
        ) in FACTOR_RECORD_SPECS.items():
            with self.subTest(
                factor_id=factor_id
            ):
                result = resolve_demo2_record(
                    packet,
                    factor_id=factor_id,
                    root=ROOT,
                )

                self.assertEqual(
                    result["grounding_status"],
                    "record_matched",
                )
                self.assertEqual(
                    result["source_path"],
                    spec.source_path,
                )
                self.assertEqual(
                    result["grounding_role"],
                    spec.grounding_role,
                )
                self.assertEqual(
                    result["evidence_fields"],
                    list(spec.fields),
                )
                self.assertEqual(
                    result["record_scope"][
                        "row_count"
                    ],
                    len(STORE_IDS),
                )
                self.assertEqual(
                    len(
                        result[
                            "evidence_values"
                        ]
                    ),
                    len(STORE_IDS),
                )

    def test_record_keys_follow_declared_grain(
        self,
    ) -> None:
        packet = {
            "evidence_id": "test_record_keys"
        }

        for (
            factor_id,
            spec,
        ) in FACTOR_RECORD_SPECS.items():
            with self.subTest(
                factor_id=factor_id
            ):
                result = resolve_demo2_record(
                    packet,
                    factor_id=factor_id,
                    root=ROOT,
                )

                actual_keys = [
                    item["row_key"]
                    for item in result[
                        "evidence_values"
                    ]
                ]

                expected_keys = [
                    (
                        {
                            "store_id": store_id,
                            "period_month": (
                                PERIOD_MONTH
                            ),
                        }
                        if "period_month"
                        in spec.key_fields
                        else {
                            "store_id": store_id
                        }
                    )
                    for store_id in STORE_IDS
                ]

                self.assertEqual(
                    actual_keys,
                    expected_keys,
                )


if __name__ == "__main__":
    unittest.main()
