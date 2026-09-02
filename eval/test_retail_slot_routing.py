from __future__ import annotations

import unittest

from api.main import infer_retail_slots


class RetailSlotRoutingTests(unittest.TestCase):
    def test_period_only_does_not_infer_slots(self) -> None:
        for message in ("April 2026", "2026年4月"):
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


if __name__ == "__main__":
    unittest.main()
