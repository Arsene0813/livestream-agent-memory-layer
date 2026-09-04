from __future__ import annotations

import asyncio
import unittest

from api.main import (
    RetailOpsDemo2KbReq,
    chat_retail_ops_demo2_kb,
    load_demo2_retail_facts,
)


TRACEABILITY_FIELDS = (
    "value",
    "entity_id",
    "period_start",
    "period_end",
    "period_label",
    "period_granularity",
    "source_fields",
    "observed_values",
    "calculation",
    "source_path",
    "supporting_source_paths",
    "lineage_path",
)


class RetailEndpointTraceabilityTests(
    unittest.TestCase
):
    def test_demo2_response_preserves_fact_traceability(
        self,
    ) -> None:
        source_facts = load_demo2_retail_facts()
        source_by_key = {
            (
                fact["entity_id"].lower(),
                fact["slot"],
            ): fact
            for fact in source_facts
        }

        result = asyncio.run(
            chat_retail_ops_demo2_kb(
                RetailOpsDemo2KbReq(
                    message=(
                        "What was Store B's transaction "
                        "amount in March 2026?"
                    ),
                    entity_id="store_b",
                )
            )
        )

        self.assertTrue(result["supported"])
        self.assertTrue(result["facts"])

        for response_fact in result["facts"]:
            missing = [
                field
                for field in TRACEABILITY_FIELDS
                if field not in response_fact
            ]

            self.assertEqual(
                missing,
                [],
                msg=(
                    "response fact missing "
                    f"traceability fields: {missing}"
                ),
            )

            source_fact = source_by_key[
                (
                    response_fact["entity_id"].lower(),
                    response_fact["slot"],
                )
            ]

            default_values = {
                "source_fields": [],
                "observed_values": {},
                "supporting_source_paths": [],
            }

            for field in TRACEABILITY_FIELDS:
                expected = source_fact.get(field)

                if field in default_values:
                    expected = (
                        expected
                        or default_values[field]
                    )

                self.assertEqual(
                    response_fact[field],
                    expected,
                    msg=f"{field} changed in API response",
                )


if __name__ == "__main__":
    unittest.main()
