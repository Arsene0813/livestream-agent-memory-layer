from __future__ import annotations

import asyncio
import unittest

from api.main import (
    RetailOpsDemo2KbReq,
    chat_retail_ops_demo2_kb,
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


    def test_single_store_compare_is_not_cross_store(self) -> None:
        self.assertFalse(
            is_demo2_cross_store_query(
                "Compare Store B activity with "
                "its March baseline."
            )
        )

    def test_demo2_message_entity_conflict_is_refused(self) -> None:
        messages = [
            (
                "What was Store C's transaction "
                "amount in March 2026?"
            ),
            (
                "What was Store A's transaction "
                "amount in March 2026?"
            ),
        ]

        for message in messages:
            with self.subTest(message=message):
                result = asyncio.run(
                    chat_retail_ops_demo2_kb(
                        RetailOpsDemo2KbReq(
                            message=message,
                            entity_id="store_b",
                        )
                    )
                )

                self.assertFalse(result["supported"])
                self.assertEqual(result["facts"], [])

    def test_cross_store_selection_uses_requested_entities(self) -> None:
        result = asyncio.run(
            chat_retail_ops_demo2_kb(
                RetailOpsDemo2KbReq(
                    message=(
                        "Compare Store B and Store C "
                        "transaction amounts."
                    ),
                    entity_id=None,
                )
            )
        )

        self.assertTrue(result["supported"])
        self.assertEqual(
            [
                fact["slot"]
                for fact in result["facts"]
            ],
            [
                "transaction_conversion_profile",
                "transaction_conversion_profile",
            ],
        )

        answer = result["answer"]

        self.assertIn("Store B", answer)
        self.assertIn("Store C", answer)
        self.assertNotIn("Store D", answer)
        self.assertNotIn("Store E", answer)
        self.assertNotIn("Store F", answer)


    def test_plural_cross_store_selection_uses_named_entities(
        self,
    ) -> None:
        result = asyncio.run(
            chat_retail_ops_demo2_kb(
                RetailOpsDemo2KbReq(
                    message=(
                        "Compare Stores B and C "
                        "transaction amounts."
                    ),
                    entity_id=None,
                )
            )
        )

        self.assertTrue(result["supported"])
        self.assertEqual(
            [
                fact["slot"]
                for fact in result["facts"]
            ],
            [
                "transaction_conversion_profile",
                "transaction_conversion_profile",
            ],
        )

        answer = result["answer"]

        self.assertIn("Store B", answer)
        self.assertIn("Store C", answer)
        self.assertNotIn("Store D", answer)
        self.assertNotIn("Store E", answer)
        self.assertNotIn("Store F", answer)


    def test_cross_store_top_k_does_not_return_partial_scope(
        self,
    ) -> None:
        cases = [
            (
                (
                    "Are Stores B-F directly comparable "
                    "in March 2026?"
                ),
                1,
            ),
            (
                (
                    "Compare Stores B-F transaction "
                    "amounts and activity."
                ),
                5,
            ),
        ]

        for message, top_k in cases:
            with self.subTest(
                message=message,
                top_k=top_k,
            ):
                result = asyncio.run(
                    chat_retail_ops_demo2_kb(
                        RetailOpsDemo2KbReq(
                            message=message,
                            entity_id=None,
                            top_k=top_k,
                        )
                    )
                )

                self.assertFalse(result["supported"])
                self.assertEqual(result["facts"], [])
                self.assertIn(
                    "top_k",
                    result["answer"],
                )


if __name__ == "__main__":
    unittest.main()
