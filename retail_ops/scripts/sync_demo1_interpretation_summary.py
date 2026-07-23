#!/usr/bin/env python3
"""Build or check the Demo 1 summary from canonical memory facts."""

import csv
import io
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

FACTS_PATH = (
    ROOT
    / "retail_ops/outputs/generated_retail_memory_facts.json"
)

SUMMARY_PATH = (
    ROOT
    / "retail_ops/outputs/"
    "store_a_demo1_interpretation_summary.csv"
)

SLOTS = (
    "visibility_entry_profile",
    "activity_lever_profile",
    "transaction_conversion_profile",
    "single_metric_attribution_guard",
    "top3_sku_product_mix_note",
)

FIELDS = (
    "store_id",
    "period_granularity",
    "period_start",
    "period_end",
    "period_label",
    "slot",
    "summary",
)


def build_expected_csv():
    payload = json.loads(
        FACTS_PATH.read_text(encoding="utf-8")
    )

    if not isinstance(payload, list):
        raise ValueError("Demo 1 facts must be a JSON list.")

    facts = [
        fact
        for fact in payload
        if fact.get("entity_id") == "store_A"
    ]

    by_slot = {}

    for fact in facts:
        slot = fact.get("slot")

        if slot in by_slot:
            raise ValueError(f"Duplicate Demo 1 slot: {slot}")

        by_slot[slot] = fact

    if set(by_slot) != set(SLOTS):
        raise ValueError(
            "Demo 1 fact slots do not match the canonical slots.\n"
            f"Expected: {list(SLOTS)}\n"
            f"Current:  {sorted(by_slot)}"
        )

    metadata_fields = (
        "entity_id",
        "period_granularity",
        "period_start",
        "period_end",
        "period_label",
    )

    reference = tuple(
        by_slot[SLOTS[0]].get(field)
        for field in metadata_fields
    )

    rows = []

    for slot in SLOTS:
        fact = by_slot[slot]

        current = tuple(
            fact.get(field)
            for field in metadata_fields
        )

        if current != reference:
            raise ValueError(
                f"Inconsistent period metadata in slot: {slot}"
            )

        if fact.get("is_active") is not True:
            raise ValueError(
                f"Inactive Demo 1 fact cannot enter summary: {slot}"
            )

        value = fact.get("value")

        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Missing fact value for slot: {slot}"
            )

        rows.append(
            {
                "store_id": "A",
                "period_granularity": fact["period_granularity"],
                "period_start": fact["period_start"],
                "period_end": fact["period_end"],
                "period_label": fact["period_label"],
                "slot": slot,
                "summary": " ".join(value.split()),
            }
        )

    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=FIELDS,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)

    return output.getvalue()


def main():
    expected = build_expected_csv()

    if "--check" in sys.argv:
        if not SUMMARY_PATH.exists():
            print("FAILED: interpretation summary is missing.")
            return 1

        current = SUMMARY_PATH.read_text(encoding="utf-8")

        if current != expected:
            print(
                "FAILED: interpretation summary is out of sync."
            )
            print(
                "Run this command to rebuild it:\n"
                "python3 retail_ops/scripts/"
                "sync_demo1_interpretation_summary.py --write"
            )
            return 1

        print(
            "Demo 1 interpretation summary check PASSED."
        )
        return 0

    if "--write" not in sys.argv:
        print("Use --write or --check.")
        return 2

    SUMMARY_PATH.write_text(
        expected,
        encoding="utf-8",
    )

    print(
        "Generated: "
        "retail_ops/outputs/"
        "store_a_demo1_interpretation_summary.csv"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
