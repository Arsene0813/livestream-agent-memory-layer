from __future__ import annotations

import asyncio
import unittest

from api.main import (
    RetailOpsDemo2KbReq,
    chat_retail_ops_demo2_kb,
    is_unsupported_retail_scope,
)


class RetailPeriodScopeTests(unittest.TestCase):
    def test_demo1_period_scope(self) -> None:
        cases = [
            ("Store A exposure in February 2026", True),
            ("Store A exposure on 2026-03-15", True),
            ("Store A exposure in 2026年4月", True),
            ("Store A exposure in May 2026", False),
            ("Store A exposure on 2026-05-01", False),
            ("Store A exposure in 2026年5月", False),
            ("Store A exposure in 2027", False),
        ]

        for message, expected_supported in cases:
            with self.subTest(message=message):
                reason = is_unsupported_retail_scope(
                    message,
                    "store_a",
                )

                if expected_supported:
                    self.assertIsNone(reason)
                else:
                    self.assertIsNotNone(reason)

    def test_demo2_period_scope(self) -> None:
        cases = [
            (
                "Store B transaction amount in March 2026",
                True,
            ),
            (
                "Store B transaction amount on 2026-03-15",
                True,
            ),
            (
                "Store B transaction amount in 2026年3月",
                True,
            ),
            (
                "Store B transaction amount in April 2026",
                False,
            ),
            (
                "Store B transaction amount on 2026-04-15",
                False,
            ),
            (
                "Store B transaction amount in 2026年4月",
                False,
            ),
            (
                "Store B transaction amount in 2027",
                False,
            ),
        ]

        for message, expected_supported in cases:
            with self.subTest(message=message):
                result = asyncio.run(
                    chat_retail_ops_demo2_kb(
                        RetailOpsDemo2KbReq(
                            message=message,
                            entity_id="store_b",
                        )
                    )
                )

                self.assertEqual(
                    result["supported"],
                    expected_supported,
                )

                if not expected_supported:
                    self.assertEqual(
                        result["facts"],
                        [],
                    )


if __name__ == "__main__":
    unittest.main()
