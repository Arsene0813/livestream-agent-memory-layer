from __future__ import annotations

import unittest

from api.main import (
    infer_retail_slots,
    is_demo2_cross_store_query,
    is_unsupported_demo2_retail_scope,
    is_unsupported_retail_scope,
)


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


    def test_demo2_cross_store_term_boundaries(self) -> None:
        cases = [
            ("Compare Store B and Store C.", True),
            ("Compare Stores B-F.", True),
            ("Review this sub-field definition.", False),
            ("Compare storefront layouts.", False),
        ]

        for message, expected in cases:
            with self.subTest(message=message):
                self.assertEqual(
                    is_demo2_cross_store_query(message),
                    expected,
                )

    def test_demo2_unsupported_scope_term_boundaries(self) -> None:
        cases = [
            ("Summarize all stores.", True),
            ("Summarize small stores.", False),
            ("Which is the best store?", True),
            ("Review the best storefront layout.", False),
        ]

        for message, expected_unsupported in cases:
            with self.subTest(message=message):
                result = is_unsupported_demo2_retail_scope(
                    message,
                    "store_b",
                )

                if expected_unsupported:
                    self.assertIsNotNone(result)
                else:
                    self.assertIsNone(result)


    def test_demo1_scope_boundaries_and_cause_forms(self) -> None:
        cases = [
            ("Review Store B metrics.", True),
            ("Review store billing metrics.", False),
            ("Summarize all stores.", True),
            ("Summarize small stores.", False),
            ("Exposure caused the growth.", True),
            ("Did exposure cause the growth?", True),
            ("Was growth caused by search alone?", False),
        ]

        for message, expected_unsupported in cases:
            with self.subTest(message=message):
                result = is_unsupported_retail_scope(
                    message,
                    "store_a",
                )

                if expected_unsupported:
                    self.assertIsNotNone(result)
                else:
                    self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
