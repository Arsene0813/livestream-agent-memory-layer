#!/usr/bin/env python3
"""Validate Demo 1 source -> SQL output -> memory-fact value lineage."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "retail_ops/data/store_a_monthly_metrics.csv"
SKUS = ROOT / "retail_ops/data/store_a_top_skus.csv"
OUTPUT = ROOT / "retail_ops/outputs/store_a_demo1_sql_output.csv"
FACTS = ROOT / "retail_ops/outputs/generated_retail_memory_facts.json"
RESULT = ROOT / "retail_ops/outputs/demo1_value_lineage_validation_result.txt"

STORE = "A"
ENTITY = "store_A"
PERIODS = ("2026-02", "2026-03", "2026-04")
PERIOD_LABEL = "2026-02_to_2026-04"
PERIOD_START = "2026-02-01"
PERIOD_END = "2026-04-30"
PRODUCT_SLOT = "top3_sku_product_mix_note"

SLOTS = {
    "visibility_entry_profile",
    "activity_lever_profile",
    "transaction_conversion_profile",
    "single_metric_attribution_guard",
    PRODUCT_SLOT,
}


def load_csv(
    path: Path,
) -> tuple[list[str], list[dict[str, str]]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path.relative_to(ROOT)}"
        )

    with path.open(
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        rows = list(reader)

    if not headers:
        raise ValueError(
            f"Missing CSV header: {path.relative_to(ROOT)}"
        )

    return headers, rows


def index_periods(
    rows: list[dict[str, str]],
    label: str,
) -> dict[tuple[str, str], dict[str, str]]:
    indexed: dict[
        tuple[str, str],
        dict[str, str],
    ] = {}

    for row_number, row in enumerate(
        rows,
        start=2,
    ):
        key = (
            row.get("store_id", ""),
            row.get("period_month", ""),
        )

        if not all(key):
            raise ValueError(
                f"{label} row {row_number} has empty "
                "store_id or period_month"
            )

        if key in indexed:
            raise ValueError(
                f"{label} has duplicate key={key}"
            )

        indexed[key] = row

    return indexed


def decimal_value(
    value: Any,
) -> Decimal | None:
    if (
        value is None
        or value == ""
        or isinstance(value, bool)
    ):
        return None

    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return None


def bool_value(
    value: Any,
) -> bool | None:
    if isinstance(value, bool):
        return value

    if isinstance(value, str):
        normalized = value.strip().lower()

        if normalized == "true":
            return True

        if normalized == "false":
            return False

    return None


def equal(
    left: Any,
    right: Any,
    tolerance: str = "0.0001",
) -> bool:
    if left is None or left == "":
        return right is None or right == ""

    if right is None or right == "":
        return False

    left_bool = bool_value(left)
    right_bool = bool_value(right)

    if (
        left_bool is not None
        or right_bool is not None
    ):
        return (
            left_bool is not None
            and left_bool == right_bool
        )

    left_number = decimal_value(left)
    right_number = decimal_value(right)

    if (
        left_number is not None
        and right_number is not None
    ):
        return (
            abs(left_number - right_number)
            <= Decimal(tolerance)
        )

    return str(left) == str(right)


def ratio(
    numerator: Any,
    denominator: Any,
    scale: float = 1.0,
) -> float:
    denominator_value = float(denominator)

    if denominator_value == 0:
        return 0.0

    return round(
        float(numerator)
        / denominator_value
        * scale,
        2,
    )


def pct_change(
    current: Any,
    previous: Any,
) -> float | None:
    if (
        previous is None
        or previous == ""
        or float(previous) == 0
    ):
        return None

    return round(
        (
            float(current)
            - float(previous)
        )
        / float(previous)
        * 100,
        2,
    )


def write_result(
    lines: list[str],
) -> None:
    text = "\n".join(lines).rstrip() + "\n"

    RESULT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULT.write_text(
        text,
        encoding="utf-8",
    )

    print(text, end="")


def main() -> int:
    failures: list[str] = []

    def check(
        label: str,
        period: str,
        field: str,
        actual: Any,
        expected: Any,
        tolerance: str = "0.0001",
    ) -> None:
        if not equal(
            actual,
            expected,
            tolerance,
        ):
            failures.append(
                f"{label}: period {period} "
                f"field `{field}` mismatch: "
                f"actual={actual!r}, "
                f"expected={expected!r}"
            )

    try:
        (
            source_headers,
            source_records,
        ) = load_csv(SOURCE)

        (
            sku_headers,
            sku_records,
        ) = load_csv(SKUS)

        (
            output_headers,
            output_records,
        ) = load_csv(OUTPUT)

        source_rows = index_periods(
            source_records,
            "Demo 1 source",
        )

        output_rows = index_periods(
            output_records,
            "Demo 1 output",
        )

    except (
        FileNotFoundError,
        ValueError,
    ) as exc:
        write_result(
            [
                "Demo 1 value lineage "
                "validation FAILED.",
                f"[FAIL] {exc}",
            ]
        )
        return 1

    expected_keys = {
        (STORE, period)
        for period in PERIODS
    }

    if set(source_rows) != expected_keys:
        failures.append(
            "Source store-period keys mismatch: "
            f"{sorted(source_rows)}"
        )

    if set(output_rows) != expected_keys:
        failures.append(
            "Output store-period keys mismatch: "
            f"{sorted(output_rows)}"
        )

    sku_by_period: dict[
        tuple[str, str],
        list[dict[str, str]],
    ] = defaultdict(list)

    for row in sku_records:
        sku_by_period[
            (
                row["store_id"],
                row["period_month"],
            )
        ].append(row)

    if set(sku_by_period) != expected_keys:
        failures.append(
            "Top-SKU store-period keys mismatch: "
            f"{sorted(sku_by_period)}"
        )

    sku_checks = 0

    for key in sorted(expected_keys):
        period = key[1]

        rows = sorted(
            sku_by_period.get(
                key,
                [],
            ),
            key=lambda row: int(
                row["sku_rank"]
            ),
        )

        if len(rows) != 3:
            failures.append(
                f"top-SKU: period {period} "
                "expected 3 rows, "
                f"found {len(rows)}"
            )
            continue

        ranks = [
            int(row["sku_rank"])
            for row in rows
        ]

        if ranks != [1, 2, 3]:
            failures.append(
                f"top-SKU: period {period} "
                f"ranks mismatch: {ranks}"
            )

        source_row = source_rows.get(key)

        if source_row is None:
            continue

        for row in rows:
            for field in (
                "store_id",
                "period_month",
                "period_start",
                "period_end",
            ):
                sku_checks += 1

                check(
                    "source-to-SKU",
                    period,
                    field,
                    row[field],
                    source_row[field],
                )

            for field in (
                "sku_name",
                "sku_transaction_amount",
                "sku_category_note",
            ):
                if not row.get(field):
                    failures.append(
                        f"top-SKU: period {period} "
                        f"has empty {field}"
                    )

    shared_fields = [
        field
        for field in source_headers
        if field in output_headers
    ]

    source_formulas: dict[
        str,
        Callable[
            [dict[str, str]],
            Any,
        ],
    ] = {
        "average_order_value": (
            lambda row: ratio(
                row["transaction_amount"],
                row["transaction_orders"],
            )
        ),
        "entry_conversion_rate_pct": (
            lambda row: ratio(
                row["entry_users"],
                row["exposure_users"],
                100,
            )
        ),
        "order_conversion_rate_pct": (
            lambda row: ratio(
                row["order_users"],
                row["entry_users"],
                100,
            )
        ),
        "payment_conversion_rate_pct": (
            lambda row: ratio(
                row["payment_users"],
                row["order_users"],
                100,
            )
        ),
        "activity_cost": (
            lambda row: round(
                float(
                    row[
                        "merchant_subsidy_amount"
                    ]
                )
                + float(
                    row[
                        "platform_subsidy_amount"
                    ]
                ),
                2,
            )
        ),
        "activity_cost_ratio_pct": (
            lambda row: ratio(
                row["activity_cost"],
                row[
                    "activity_original_transaction_amount"
                ],
                100,
            )
        ),
    }

    output_formulas: dict[
        str,
        Callable[
            [dict[str, str], float],
            Any,
        ],
    ] = {
        "search_exposure_share_pct": (
            lambda row, _: ratio(
                row["search_exposure_users"],
                row["exposure_users"],
                100,
            )
        ),
        "search_entry_share_pct": (
            lambda row, _: ratio(
                row["search_entry_users"],
                row["entry_users"],
                100,
            )
        ),
        "search_entry_rate_pct": (
            lambda row, _: ratio(
                row["search_entry_users"],
                row["search_exposure_users"],
                100,
            )
        ),
        "estimated_income_proxy_ratio_pct": (
            lambda row, _: ratio(
                row["estimated_income_proxy"],
                row["transaction_amount"],
                100,
            )
        ),
        "activity_order_share_pct": (
            lambda row, _: ratio(
                row["activity_orders"],
                row["transaction_orders"],
                100,
            )
        ),
        (
            "merchant_subsidy_share_of_"
            "activity_cost_pct"
        ): (
            lambda row, _: ratio(
                row[
                    "merchant_subsidy_amount"
                ],
                row["activity_cost"],
                100,
            )
        ),
        "top3_sku_transaction_amount": (
            lambda _, top3: top3
        ),
        (
            "top3_sku_transaction_amount_"
            "share_pct"
        ): (
            lambda row, top3: ratio(
                top3,
                row["transaction_amount"],
                100,
            )
        ),
    }

    mom_fields = {
        "transaction_amount_mom_pct": (
            "transaction_amount"
        ),
        "transaction_orders_mom_pct": (
            "transaction_orders"
        ),
        "estimated_income_proxy_mom_pct": (
            "estimated_income_proxy"
        ),
        "exposure_users_mom_pct": (
            "exposure_users"
        ),
        "search_exposure_users_mom_pct": (
            "search_exposure_users"
        ),
        "entry_users_mom_pct": (
            "entry_users"
        ),
        "search_entry_users_mom_pct": (
            "search_entry_users"
        ),
        "order_users_mom_pct": (
            "order_users"
        ),
        "payment_users_mom_pct": (
            "payment_users"
        ),
        "average_order_value_mom_pct": (
            "average_order_value"
        ),
    }

    output_checks = 0
    previous: dict[str, str] | None = None

    for period in PERIODS:
        key = (STORE, period)
        source_row = source_rows.get(key)
        output_row = output_rows.get(key)

        if (
            source_row is None
            or output_row is None
        ):
            continue

        for field in shared_fields:
            output_checks += 1

            check(
                "source-to-output",
                period,
                field,
                output_row[field],
                source_row[field],
            )

        for (
            field,
            formula,
        ) in source_formulas.items():
            output_checks += 1

            check(
                "source-formula",
                period,
                field,
                source_row[field],
                formula(source_row),
                "0.011",
            )

        top3_amount = round(
            sum(
                float(
                    row[
                        "sku_transaction_amount"
                    ]
                )
                for row in sku_by_period.get(
                    key,
                    [],
                )
            ),
            2,
        )

        for (
            field,
            formula,
        ) in output_formulas.items():
            output_checks += 1

            check(
                "derived-output",
                period,
                field,
                output_row[field],
                formula(
                    output_row,
                    top3_amount,
                ),
                "0.011",
            )

        for (
            field,
            source_field,
        ) in mom_fields.items():
            expected = (
                None
                if previous is None
                else pct_change(
                    output_row[source_field],
                    previous[source_field],
                )
            )

            output_checks += 1

            check(
                "month-over-month-output",
                period,
                field,
                output_row[field],
                expected,
                "0.011",
            )

        for (
            field,
            source_field,
        ) in (
            (
                "store_average_rank_change",
                "store_average_rank",
            ),
            (
                "search_average_rank_change",
                "search_average_rank",
            ),
        ):
            expected = (
                None
                if previous is None
                else (
                    float(
                        output_row[
                            source_field
                        ]
                    )
                    - float(
                        previous[
                            source_field
                        ]
                    )
                )
            )

            output_checks += 1

            check(
                "rank-change-output",
                period,
                field,
                output_row[field],
                expected,
            )

        tradeoff = False

        if previous is not None:
            tradeoff = (
                float(
                    output_row[
                        "transaction_amount"
                    ]
                )
                > float(
                    previous[
                        "transaction_amount"
                    ]
                )
                and float(
                    output_row[
                        "transaction_orders"
                    ]
                )
                > float(
                    previous[
                        "transaction_orders"
                    ]
                )
                and float(
                    output_row[
                        "order_conversion_rate_pct"
                    ]
                )
                < float(
                    previous[
                        "order_conversion_rate_pct"
                    ]
                )
                and float(
                    output_row[
                        "average_order_value"
                    ]
                )
                < float(
                    previous[
                        "average_order_value"
                    ]
                )
            )

        output_checks += 1

        check(
            "tradeoff-output",
            period,
            (
                "transaction_recovered_with_"
                "conversion_aov_tradeoff"
            ),
            output_row[
                (
                    "transaction_recovered_with_"
                    "conversion_aov_tradeoff"
                )
            ],
            tradeoff,
        )

        previous = output_row

    try:
        facts = json.loads(
            FACTS.read_text(
                encoding="utf-8",
            )
        )

    except FileNotFoundError:
        failures.append(
            f"Missing file: "
            f"{FACTS.relative_to(ROOT)}"
        )
        facts = []

    except json.JSONDecodeError as exc:
        failures.append(
            f"Invalid JSON in "
            f"{FACTS.relative_to(ROOT)}: "
            f"{exc}"
        )
        facts = []

    if not isinstance(facts, list):
        failures.append(
            "Generated Demo 1 memory facts "
            "must be a JSON list"
        )
        facts = []

    if len(facts) != len(SLOTS):
        failures.append(
            "Expected 5 Demo 1 facts, "
            f"found {len(facts)}"
        )

    facts_by_slot: dict[
        str,
        dict[str, Any],
    ] = {}

    fact_checks = 0

    for index, fact in enumerate(facts):
        if not isinstance(fact, dict):
            failures.append(
                f"Fact #{index} is not an object"
            )
            continue

        slot = fact.get("slot")

        if slot not in SLOTS:
            failures.append(
                f"Fact #{index} has unsupported "
                f"slot: {slot!r}"
            )
            continue

        if slot in facts_by_slot:
            failures.append(
                f"Duplicate Demo 1 slot: {slot}"
            )
            continue

        facts_by_slot[slot] = fact

        metadata = {
            "entity_id": ENTITY,
            "period_label": PERIOD_LABEL,
            "period_start": PERIOD_START,
            "period_end": PERIOD_END,
            "period_granularity": (
                "month_range"
            ),
            "is_active": True,
        }

        for (
            field,
            expected,
        ) in metadata.items():
            fact_checks += 1

            check(
                f"fact {slot}",
                PERIOD_LABEL,
                field,
                fact.get(field),
                expected,
            )

        source_path = (
            (
                "retail_ops/data/"
                "store_a_top_skus.csv"
            )
            if slot == PRODUCT_SLOT
            else (
                "retail_ops/outputs/"
                "store_a_demo1_sql_output.csv"
            )
        )

        fact_checks += 1

        check(
            f"fact {slot}",
            PERIOD_LABEL,
            "source_path",
            fact.get("source_path"),
            source_path,
        )

        source_fields = fact.get(
            "source_fields",
            [],
        )

        if not isinstance(
            source_fields,
            list,
        ):
            failures.append(
                f"Fact {slot} has "
                "non-list source_fields"
            )
            source_fields = []

        allowed_headers = set(
            (
                sku_headers
                if slot == PRODUCT_SLOT
                else output_headers
            )
        )

        for field in source_fields:
            if field not in allowed_headers:
                failures.append(
                    f"Fact {slot} source field "
                    f"`{field}` is absent from "
                    f"{source_path}"
                )

        observed = fact.get(
            "observed_values"
        )

        if not isinstance(
            observed,
            dict,
        ):
            failures.append(
                f"Fact {slot} has non-object "
                "observed_values"
            )
            continue

        if slot == PRODUCT_SLOT:
            fact_checks += 1

            check(
                f"fact {slot}",
                PERIOD_LABEL,
                "evidence_scope",
                observed.get(
                    "evidence_scope"
                ),
                (
                    "top 3 SKU transaction and "
                    "sales evidence only"
                ),
            )

            if set(observed) != {
                "evidence_scope"
            }:
                failures.append(
                    f"Fact {slot} observed_values "
                    "must contain only "
                    "evidence_scope"
                )

            categories = {
                row.get(
                    "sku_category_note",
                    "",
                )
                for row in sku_records
            }

            if categories != {
                "care_solution"
            }:
                failures.append(
                    f"Fact {slot} product-mix "
                    "claim is not supported by "
                    "sku_category_note values: "
                    f"{sorted(categories)}"
                )

            continue

        for (
            period,
            values,
        ) in observed.items():
            if period not in PERIODS:
                failures.append(
                    f"Fact {slot} has unsupported "
                    f"observed period: {period!r}"
                )
                continue

            if not isinstance(
                values,
                dict,
            ):
                failures.append(
                    f"Fact {slot}/{period} "
                    "observed value must be "
                    "an object"
                )
                continue

            output_row = output_rows.get(
                (STORE, period)
            )

            if output_row is None:
                continue

            for (
                field,
                actual,
            ) in values.items():
                if field not in source_fields:
                    failures.append(
                        f"Fact {slot}/{period} "
                        f"observed field `{field}` "
                        "is absent from "
                        "source_fields"
                    )
                    continue

                if field not in output_row:
                    failures.append(
                        f"Fact {slot}/{period} "
                        f"observed field `{field}` "
                        "has no SQL-output "
                        "lineage mapping"
                    )
                    continue

                fact_checks += 1

                check(
                    f"fact {slot}",
                    period,
                    field,
                    actual,
                    output_row[field],
                    "0.011",
                )

    missing_slots = (
        SLOTS - set(facts_by_slot)
    )

    if missing_slots:
        failures.append(
            "Missing Demo 1 fact slots: "
            f"{sorted(missing_slots)}"
        )

    if failures:
        write_result(
            [
                "Demo 1 value lineage "
                "validation FAILED.",
            ]
            + [
                f"[FAIL] {item}"
                for item in failures
            ]
        )
        return 1

    write_result(
        [
            (
                "Demo 1 value lineage "
                "validation PASSED."
            ),
            (
                "Checked source "
                "store-period rows: "
                f"{len(source_rows)}"
            ),
            (
                "Checked SQL output "
                "store-period rows: "
                f"{len(output_rows)}"
            ),
            (
                "Checked top-SKU rows: "
                f"{len(sku_records)}"
            ),
            (
                "Checked source/SKU "
                "period-key comparisons: "
                f"{sku_checks}"
            ),
            (
                "Checked source-to-output, "
                "formula, month-over-month, "
                "rank-change, and tradeoff "
                "comparisons: "
                f"{output_checks}"
            ),
            (
                "Checked generated "
                "memory facts: "
                f"{len(facts)}"
            ),
            (
                "Checked fact metadata and "
                "observed-value comparisons: "
                f"{fact_checks}"
            ),
            (
                "Checked the product-mix fact "
                "against the top-3 SKU evidence "
                "scope and sku_category_note "
                "values."
            ),
            (
                "Checked each period-level "
                "observed field against "
                "SQL-output values and "
                "declared source_fields."
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
