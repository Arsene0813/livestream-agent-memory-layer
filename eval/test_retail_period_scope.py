from __future__ import annotations

import asyncio
import unittest

from api.main import (
    RetailOpsDemo2KbReq,
    chat_retail_ops_demo2_kb,
    is_unsupported_demo2_retail_scope,
    is_unsupported_retail_scope,
)


class RetailPeriodScopeTests(unittest.TestCase):
    def test_demo1_period_scope(self) -> None:
        cases = [
            ("Store A exposure in February 2026", True),
            ("Store A exposure on 2026-03-15", False),
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
                False,
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


    def test_explicit_dates_preserve_monthly_window_scope(self) -> None:
        cases = [
            ("2026-03", True),
            ("2026-03-01 to 2026-03-31", True),
            ("2026-03-01 至 2026-03-31", True),
            ("2026/03/01 到 2026/03/31", True),
            ("2026-03-15", False),
            ("2026-03-01", False),
            ("2026-03-01 and 2026-03-31", False),
            ("2026-02-15 to 2026-03-15", False),
            ("2026-03-01 to 2026-03-15", False),
            ("2026-03-31 to 2026-03-01", False),
            ("2026-03-01 to 2026-03-32", False),
            ("2026-03-01 to 2026-03-31T12:00:00", False),
            ("2026-00-01 to 2026-03-31", False),
            ("2026-03-01 to 2026-03-31 and March 15, 2026", False),
            ("2026-03-01 to 2026-03-31 and 2026年3月15日", False),
            ("2026-03-01 to 2026-03-31 daily", False),
            ("2026-03-01 to 2026-03-31 按日", False),
            ("2026-03-01 to 2026-03-31 每小时", False),
            ("2026-05-01 to 2026-05-31", False),
            ("2027-03-01 to 2027-03-31", False),
        ]
        for check, store in (
            (is_unsupported_retail_scope, "store_a"),
            (is_unsupported_demo2_retail_scope, "store_b"),
        ):
            for window, supported in cases:
                with self.subTest(store=store, window=window):
                    reason = check("transaction amount " + window, store)
                    self.assertEqual(reason is None, supported)

        for window in (
            "2026-02-01 to 2026-02-28",
            "2026-02-01 to 2026-04-30",
            "2026-04-01 to 2026-04-30",
        ):
            with self.subTest(window=window):
                self.assertIsNone(is_unsupported_retail_scope(window, "store_a"))
                self.assertIsNotNone(is_unsupported_demo2_retail_scope(window, "store_b"))
        self.assertIsNotNone(
            is_unsupported_retail_scope("2026-02-01 to 2026-02-29", "store_a")
        )


if __name__ == "__main__":
    unittest.main()
