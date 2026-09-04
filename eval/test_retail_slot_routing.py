from __future__ import annotations

import unittest

from api.main import infer_retail_slots


class RetailSlotRoutingTests(unittest.TestCase):
    def test_period_only_does_not_infer_slots(self) -> None:
        for message in (
            "February 2026",
            "March 2026",
            "April 2026",
            "2026-03-01",
            "2026年2月",
            "2026年3月",
            "2026年4月",
        ):
            with self.subTest(message=message):
                self.assertEqual(
                    infer_retail_slots(message),
                    [],
                )

    def test_visibility_query_keeps_visibility_scope(self) -> None:
        self.assertEqual(
            infer_retail_slots(
                "What does Store A's visibility and entry profile "
                "show from February to April 2026?"
            ),
            [
                "visibility_entry_profile",
                "single_metric_attribution_guard",
            ],
        )

    def test_term_boundaries_and_canonical_fields(self) -> None:
        cases = [
            (
                "Summarize the research design for Store B.",
                [],
            ),
            (
                "What was Store B's exposure_users in March 2026?",
                [
                    "visibility_entry_profile",
                    "single_metric_attribution_guard",
                ],
            ),
            (
                "What was Store B's transaction_amount in March 2026?",
                [
                    "transaction_conversion_profile",
                    "single_metric_attribution_guard",
                ],
            ),
            (
                "What was Store B's transaction_orders in March 2026?",
                [
                    "transaction_conversion_profile",
                    "single_metric_attribution_guard",
                ],
            ),
            (
                "What was Store B's order_conversion_rate_pct in March 2026?",
                [
                    "transaction_conversion_profile",
                    "single_metric_attribution_guard",
                ],
            ),
            (
                "How did Store B recover in March 2026?",
                [
                    "transaction_conversion_profile",
                    "single_metric_attribution_guard",
                    "activity_lever_profile",
                ],
            ),
        ]

        for message, expected in cases:
            with self.subTest(message=message):
                self.assertEqual(
                    infer_retail_slots(message),
                    expected,
                )


if __name__ == "__main__":
    unittest.main()
