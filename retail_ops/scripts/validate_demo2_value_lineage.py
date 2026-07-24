#!/usr/bin/env python3
"""Validate Demo 2 value lineage: source CSV -> output CSV -> memory facts."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "retail_ops/data/demo2_store_period_metrics.csv"
OUTPUT = ROOT / "retail_ops/outputs/demo2_cross_store_comparability_output.csv"
SEARCH = ROOT / "retail_ops/data/demo2_top_search_terms.csv"
SKUS = ROOT / "retail_ops/data/demo2_top_skus_by_transaction_amount.csv"
FACTS = ROOT / "retail_ops/outputs/generated_demo2_retail_memory_facts.json"
RESULT = ROOT / "retail_ops/outputs/demo2_value_lineage_validation_result.txt"

STORES = {"B", "C", "D", "E", "F"}
SLOTS = {
    "visibility_entry_profile",
    "activity_lever_profile",
    "transaction_conversion_profile",
    "top3_sku_product_mix_note",
    "single_metric_attribution_guard",
}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path.relative_to(ROOT)}")

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        rows = list(reader)

    if not headers:
        raise ValueError(f"Missing CSV header: {path.relative_to(ROOT)}")

    return headers, rows


def unique_by_store(
    rows: list[dict[str, str]],
    label: str,
) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}

    for row_number, row in enumerate(rows, start=2):
        store_id = row.get("store_id", "")

        if not store_id:
            raise ValueError(
                f"{label} row {row_number} has empty store_id"
            )

        if store_id in indexed:
            raise ValueError(
                f"{label} has duplicate store_id={store_id}"
            )

        indexed[store_id] = row

    return indexed


def group_by_store(
    rows: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)

    for row in rows:
        grouped[row["store_id"]].append(row)

    return dict(grouped)


def as_decimal(value: Any) -> Decimal | None:
    if value is None or value == "" or isinstance(value, bool):
        return None

    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def same_value(
    left: Any,
    right: Any,
    tolerance: str = "0.0001",
) -> bool:
    if left is None or left == "":
        return right is None or right == ""

    if right is None or right == "":
        return False

    left_number = as_decimal(left)
    right_number = as_decimal(right)

    if left_number is not None and right_number is not None:
        return (
            abs(left_number - right_number)
            <= Decimal(tolerance)
        )

    return str(left) == str(right)


def pct(numerator: str, denominator: str) -> float:
    denominator_value = float(denominator)

    if denominator_value == 0:
        return 0.0

    return round(
        float(numerator) / denominator_value * 100,
        2,
    )


def ratio(numerator: str, denominator: str) -> float:
    denominator_value = float(denominator)

    if denominator_value == 0:
        return 0.0

    return round(
        float(numerator) / denominator_value,
        2,
    )


def report(lines: list[str]) -> None:
    text = "\n".join(lines).rstrip() + "\n"

    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(text, encoding="utf-8")

    print(text, end="")


def main() -> int:
    failures: list[str] = []

    def check(
        label: str,
        store_id: str,
        field: str,
        actual: Any,
        expected: Any,
        tolerance: str = "0.0001",
    ) -> None:
        if not same_value(actual, expected, tolerance):
            failures.append(
                f"{label}: store {store_id} field `{field}` "
                f"mismatch: actual={actual!r}, "
                f"expected={expected!r}"
            )

    try:
        source_headers, source_records = read_csv(SOURCE)
        output_headers, output_records = read_csv(OUTPUT)
        search_headers, search_records = read_csv(SEARCH)
        sku_headers, sku_records = read_csv(SKUS)

        source_rows = unique_by_store(
            source_records,
            "Demo 2 source",
        )
        output_rows = unique_by_store(
            output_records,
            "Demo 2 output",
        )

    except (FileNotFoundError, ValueError) as exc:
        report(
            [
                "Demo 2 value lineage validation FAILED.",
                f"[FAIL] {exc}",
            ]
        )
        return 1

    search_by_store = group_by_store(search_records)
    sku_by_store = group_by_store(sku_records)

    if set(source_rows) != STORES:
        failures.append(
            f"Source stores mismatch: {sorted(source_rows)}"
        )

    if set(output_rows) != STORES:
        failures.append(
            f"Output stores mismatch: {sorted(output_rows)}"
        )

    source_output_checks = 0

    shared_fields = [
        field
        for field in source_headers
        if field in output_headers
    ]

    for store_id in sorted(STORES):
        source_row = source_rows.get(store_id)
        output_row = output_rows.get(store_id)

        if source_row is None or output_row is None:
            continue

        for field in shared_fields:
            source_output_checks += 1

            check(
                "source-to-output",
                store_id,
                field,
                output_row[field],
                source_row[field],
            )

        derived = {
            "average_order_value": ratio(
                output_row["transaction_amount"],
                output_row["transaction_orders"],
            ),
            "entry_conversion_rate_pct": pct(
                output_row["entry_users"],
                output_row["exposure_users"],
            ),
            "order_conversion_rate_pct": pct(
                output_row["order_users"],
                output_row["entry_users"],
            ),
            "payment_conversion_rate_pct": pct(
                output_row["payment_users"],
                output_row["order_users"],
            ),
            "search_entry_rate_pct": pct(
                output_row["search_entry_users"],
                output_row["search_exposure_users"],
            ),
            "search_entry_share_pct": pct(
                output_row["search_entry_users"],
                output_row["entry_users"],
            ),
            "activity_cost_ratio_pct": pct(
                output_row["activity_cost"],
                output_row[
                    "activity_original_transaction_amount"
                ],
            ),
            "activity_order_share_pct": pct(
                output_row["activity_orders"],
                output_row["transaction_orders"],
            ),
        }

        top3_amount = round(
            sum(
                float(row["sku_transaction_amount"])
                for row in sku_by_store.get(store_id, [])
            ),
            2,
        )

        derived["top3_sku_transaction_amount"] = (
            top3_amount
        )

        derived[
            "top3_sku_transaction_amount_share_pct"
        ] = round(
            top3_amount
            / float(output_row["transaction_amount"])
            * 100,
            2,
        )

        derived["activity_cost"] = round(
            float(output_row["merchant_subsidy_amount"])
            + float(output_row["platform_subsidy_amount"]),
            2,
        )

        for field, expected in derived.items():
            source_output_checks += 1

            check(
                "derived-output",
                store_id,
                field,
                output_row[field],
                expected,
                "0.011",
            )

    csv_headers = {
        (
            "retail_ops/outputs/"
            "demo2_cross_store_comparability_output.csv"
        ): set(output_headers),
        (
            "retail_ops/data/"
            "demo2_top_search_terms.csv"
        ): set(search_headers),
        (
            "retail_ops/data/"
            "demo2_top_skus_by_transaction_amount.csv"
        ): set(sku_headers),
    }

    try:
        facts = json.loads(
            FACTS.read_text(encoding="utf-8")
        )

    except FileNotFoundError:
        failures.append(
            f"Missing file: {FACTS.relative_to(ROOT)}"
        )
        facts = []

    except json.JSONDecodeError as exc:
        failures.append(
            f"Invalid JSON in "
            f"{FACTS.relative_to(ROOT)}: {exc}"
        )
        facts = []

    if not isinstance(facts, list):
        failures.append(
            "Generated Demo 2 memory facts "
            "must be a JSON list"
        )
        facts = []

    if len(facts) != len(STORES) * len(SLOTS):
        failures.append(
            f"Expected 25 facts, found {len(facts)}"
        )

    fact_checks = 0
    seen_pairs: set[tuple[str, str]] = set()

    for index, fact in enumerate(facts):
        if not isinstance(fact, dict):
            failures.append(
                f"Fact #{index} is not an object"
            )
            continue

        entity_id = fact.get("entity_id")
        slot = fact.get("slot")

        if (
            not isinstance(entity_id, str)
            or not entity_id.startswith("store_")
        ):
            failures.append(
                f"Fact #{index} has invalid "
                f"entity_id: {entity_id!r}"
            )
            continue

        store_id = entity_id.removeprefix("store_")
        pair = (entity_id, str(slot))

        if pair in seen_pairs:
            failures.append(
                f"Duplicate entity-slot pair: {pair}"
            )

        seen_pairs.add(pair)

        if store_id not in STORES:
            failures.append(
                f"Fact #{index} has unsupported "
                f"store: {store_id!r}"
            )
            continue

        if slot not in SLOTS:
            failures.append(
                f"Fact #{index} has unsupported "
                f"slot: {slot!r}"
            )

        output_row = output_rows.get(store_id)

        if output_row is None:
            continue

        period_values = {
            "period_label": output_row["period_month"],
            "period_start": output_row["period_start"],
            "period_end": output_row["period_end"],
        }

        for field, expected in period_values.items():
            fact_checks += 1

            check(
                f"fact {entity_id}/{slot}",
                store_id,
                field,
                fact.get(field),
                expected,
            )

        source_fields = fact.get(
            "source_fields",
            [],
        )
        source_path = fact.get(
            "source_path"
        )
        supporting_paths = fact.get(
            "supporting_source_paths",
            [],
        )

        if not isinstance(source_fields, list):
            failures.append(
                f"Fact {entity_id}/{slot} "
                "has non-list source_fields"
            )
            source_fields = []

        declared_paths: list[str] = []

        if (
            not isinstance(source_path, str)
            or source_path not in csv_headers
        ):
            failures.append(
                f"Fact {entity_id}/{slot} "
                f"declares unsupported primary CSV "
                f"path `{source_path}`"
            )
        else:
            declared_paths.append(source_path)

        if supporting_paths is None:
            supporting_paths = []

        if not isinstance(supporting_paths, list):
            failures.append(
                f"Fact {entity_id}/{slot} "
                "has non-list supporting_source_paths"
            )
            supporting_paths = []

        for path in supporting_paths:
            if (
                not isinstance(path, str)
                or path not in csv_headers
            ):
                failures.append(
                    f"Fact {entity_id}/{slot} "
                    f"declares unsupported additional "
                    f"CSV path `{path}`"
                )
                continue

            if path == source_path:
                failures.append(
                    f"Fact {entity_id}/{slot} "
                    "repeats source_path in "
                    "supporting_source_paths"
                )
                continue

            declared_paths.append(path)

        declared_headers: set[str] = set()

        for path in dict.fromkeys(
            declared_paths
        ):
            declared_headers.update(
                csv_headers[path]
            )

        for field in source_fields:
            if field not in declared_headers:
                failures.append(
                    f"Fact {entity_id}/{slot} "
                    f"source field `{field}` is absent "
                    "from declared CSV headers"
                )

        observed = fact.get("observed_values")

        if not isinstance(observed, dict):
            failures.append(
                f"Fact {entity_id}/{slot} "
                "has non-object observed_values"
            )
            continue

        for field, actual in observed.items():
            if field == "top_search_terms":
                expected = [
                    {
                        "search_term": row["search_term"],
                        "search_term_en": row[
                            "search_term_en"
                        ],
                        (
                            "search_term_exposure_times"
                        ): int(
                            float(
                                row[
                                    "search_term_exposure_times"
                                ]
                            )
                        ),
                        (
                            "search_term_click_times"
                        ): int(
                            float(
                                row[
                                    "search_term_click_times"
                                ]
                            )
                        ),
                        (
                            "search_term_order_times"
                        ): int(
                            float(
                                row[
                                    "search_term_order_times"
                                ]
                            )
                        ),
                    }
                    for row in sorted(
                        search_by_store.get(
                            store_id,
                            [],
                        ),
                        key=lambda item: int(
                            item["search_term_rank"]
                        ),
                    )
                ]

                fact_checks += 1

                if actual != expected:
                    failures.append(
                        f"fact {entity_id}/{slot}: "
                        f"store {store_id} field "
                        "`top_search_terms` mismatch"
                    )

                required = {
                    "search_term",
                    "search_term_en",
                    "search_term_exposure_times",
                    "search_term_click_times",
                    "search_term_order_times",
                }

                missing = sorted(
                    required - set(source_fields)
                )

                if missing:
                    failures.append(
                        f"Fact {entity_id}/{slot} "
                        "missing nested search "
                        f"source fields: {missing}"
                    )

                continue

            if field == (
                "top_skus_by_transaction_amount"
            ):
                expected = [
                    {
                        "sku_name": row["sku_name"],
                        "sku_name_en": row["sku_name_en"],
                        (
                            "sku_transaction_amount"
                        ): round(
                            float(
                                row[
                                    "sku_transaction_amount"
                                ]
                            ),
                            2,
                        ),
                    }
                    for row in sorted(
                        sku_by_store.get(
                            store_id,
                            [],
                        ),
                        key=lambda item: int(
                            item["sku_rank"]
                        ),
                    )
                ]

                fact_checks += 1

                if actual != expected:
                    failures.append(
                        f"fact {entity_id}/{slot}: "
                        f"store {store_id} field "
                        "`top_skus_by_transaction_amount` "
                        "mismatch"
                    )

                required = {
                    "sku_name",
                    "sku_name_en",
                    "sku_transaction_amount",
                }

                missing = sorted(
                    required - set(source_fields)
                )

                if missing:
                    failures.append(
                        f"Fact {entity_id}/{slot} "
                        "missing nested SKU "
                        f"source fields: {missing}"
                    )

                continue

            if field not in output_row:
                failures.append(
                    f"Fact {entity_id}/{slot} "
                    f"observed field `{field}` has "
                    "no output-row lineage mapping"
                )
                continue

            if field not in source_fields:
                failures.append(
                    f"Fact {entity_id}/{slot} "
                    f"observed field `{field}` is "
                    "absent from source_fields"
                )

            fact_checks += 1

            check(
                f"fact {entity_id}/{slot}",
                store_id,
                field,
                actual,
                output_row[field],
            )

    expected_pairs = {
        (f"store_{store_id}", slot)
        for store_id in STORES
        for slot in SLOTS
    }

    missing_pairs = expected_pairs - seen_pairs
    extra_pairs = seen_pairs - expected_pairs

    if missing_pairs:
        failures.append(
            f"Missing entity-slot facts: "
            f"{sorted(missing_pairs)}"
        )

    if extra_pairs:
        failures.append(
            f"Unexpected entity-slot facts: "
            f"{sorted(extra_pairs)}"
        )

    if failures:
        report(
            [
                "Demo 2 value lineage validation FAILED.",
                *[
                    f"[FAIL] {item}"
                    for item in failures
                ],
            ]
        )
        return 1

    report(
        [
            "Demo 2 value lineage validation PASSED.",
            (
                "Checked source store-period rows: "
                f"{len(source_rows)}"
            ),
            (
                "Checked output store-period rows: "
                f"{len(output_rows)}"
            ),
            (
                "Checked source-to-output and "
                "derived-value comparisons: "
                f"{source_output_checks}"
            ),
            (
                "Checked generated memory facts: "
                f"{len(facts)}"
            ),
            (
                "Checked fact period and "
                "observed-value comparisons: "
                f"{fact_checks}"
            ),
            (
                "Checked top-search-term and top-SKU "
                "nested evidence against supporting "
                "CSV rows."
            ),
            (
                "Checked each observed field against "
                "output-row values and declared "
                "source_fields."
            ),
            (
                "Saved result path: "
                f"{RESULT.relative_to(ROOT)}"
            ),
        ]
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
