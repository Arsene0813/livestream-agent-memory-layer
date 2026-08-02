from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

OUTPUT_DIR = Path("retail_ops/outputs")
DATA_DIR = Path("retail_ops/data")

COMPARABILITY_OUTPUT = OUTPUT_DIR / "demo2_cross_store_comparability_output.csv"
TOP_SEARCH_TERMS = DATA_DIR / "demo2_top_search_terms.csv"
TOP_SKUS_BY_AMOUNT = DATA_DIR / "demo2_top_skus_by_transaction_amount.csv"

OUTPUT_PATH = OUTPUT_DIR / "generated_demo2_retail_memory_facts.json"

PERIOD_LABEL = "2026-03"
PERIOD_START = "2026-03-01"
PERIOD_END = "2026-03-31"

SOURCE_PATH = "retail_ops/outputs/demo2_cross_store_comparability_output.csv"
TOP_SEARCH_TERMS_SOURCE_PATH = "retail_ops/data/demo2_top_search_terms.csv"
TOP_SKUS_BY_AMOUNT_SOURCE_PATH = "retail_ops/data/demo2_top_skus_by_transaction_amount.csv"
LINEAGE_PATH = "retail_ops/TECHNICAL_APPENDIX.md"

VISIBILITY_ENTRY_LIMITATIONS = [
    "March 2026 same-period diagnostic only.",
    "Traffic-source user counts may overlap.",
    (
        "Visibility and entry metrics do not support "
        "single-metric causal attribution."
    ),
]

ACTIVITY_LEVER_LIMITATIONS = [
    (
        "Activity mechanism details and promotion cycle dates "
        "are not included."
    ),
    (
        "Activity fields record backend activity involvement and related "
        "cost measures; they do not establish causal impact."
    ),
    (
        "activity_cost_ratio_pct follows the platform-defined "
        "cost-ratio formula and is not a traditional ROI measure."
    ),
]

TRANSACTION_CONVERSION_LIMITATIONS = [
    "March 2026 transaction and conversion outcomes only.",
    (
        "order_conversion_rate_pct follows the backend definition "
        "and should not be recomputed from transaction-order counts."
    ),
    (
        "These fields record monthly transaction outcomes; retention and "
        "market-share movement are not separately identified."
    ),
]

TOP3_SKU_LIMITATIONS = [
    (
        "Top-3 SKU evidence only; it is not a full SKU or "
        "category-share view."
    ),
    (
        "Manual category inference should not be treated as a "
        "backend category classification."
    ),
    (
        "English SKU names are helper translations, not backend "
        "source values."
    ),
]

ATTRIBUTION_GUARD_LIMITATIONS = [
    "March 2026 same-period diagnostic only.",
    (
        "comparison_scope_flag is a row-level scope guardrail, "
        "not a pairwise comparability decision."
    ),
    (
        "The available metrics do not support single-metric "
        "attribution or strategy-transfer conclusions."
    ),
]



def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(value: str) -> float:
    return round(float(value), 2)


def as_optional_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    return round(float(value), 2)


def as_int(value: str) -> int:
    return int(float(value))


def group_by_store(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["store_id"], []).append(row)
    return grouped


def make_fact(
    *,
    entity_id: str,
    slot: str,
    value: str,
    observed_values: dict[str, Any],
    calculation: str,
    source_fields: list[str],
    confidence: str,
    limitations: list[str],
    source_path: str = SOURCE_PATH,
    supporting_source_paths: list[str] | None = None,
) -> dict[str, Any]:
    fact: dict[str, Any] = {
        "kind": "retail_memory_fact",
        "type": "retail_metric_profile",
        "entity_id": entity_id,
        "slot": slot,
        "period_label": PERIOD_LABEL,
        "period_start": PERIOD_START,
        "period_end": PERIOD_END,
        "value": value,
        "observed_values": observed_values,
        "calculation": calculation,
        "source_fields": source_fields,
        "confidence": confidence,
        "source_path": source_path,
        "lineage_path": LINEAGE_PATH,
        "limitations": limitations,
        "is_active": True,
        "period_granularity": "month",
    }

    if supporting_source_paths:
        additional_paths = [
            path
            for path in dict.fromkeys(
                supporting_source_paths
            )
            if path != source_path
        ]

        if additional_paths:
            fact["supporting_source_paths"] = (
                additional_paths
            )

    return fact


comparability_rows = read_csv(COMPARABILITY_OUTPUT)
top_search_by_store = group_by_store(read_csv(TOP_SEARCH_TERMS))
top_skus_by_store = group_by_store(read_csv(TOP_SKUS_BY_AMOUNT))

facts: list[dict[str, Any]] = []

for row in comparability_rows:
    store_id = row["store_id"]
    entity_id = f"store_{store_id}"

    top_search_terms = [
        {
            "search_term": item["search_term"],
            "search_term_en": item["search_term_en"],
            "search_term_exposure_times": as_int(item["search_term_exposure_times"]),
            "search_term_click_times": as_int(item["search_term_click_times"]),
            "search_term_order_times": as_int(item["search_term_order_times"]),
        }
        for item in top_search_by_store.get(store_id, [])
    ]

    top_skus_by_amount = [
        {
            "sku_name": item["sku_name"],
            "sku_name_en": item["sku_name_en"],
            "sku_transaction_amount": as_float(item["sku_transaction_amount"]),
        }
        for item in top_skus_by_store.get(store_id, [])
    ]


    facts.append(
        make_fact(
            entity_id=entity_id,
            slot="visibility_entry_profile",
            value=(
                f"Store {store_id}'s March 2026 visibility and entry profile records exposure, rank, entry, "
                f"and search-entry metrics across the being seen → being entered stages. "
                f"It should be read alongside transaction, conversion, activity, and SKU evidence."
            ),
            observed_values={
                "region_type": row["region_type"],
                "store_type": row["store_type"],
                "exposure_users": as_int(row["exposure_users"]),
                "exposure_times": as_int(row["exposure_times"]),
                "store_average_rank": as_float(row["store_average_rank"]),
                "entry_users": as_int(row["entry_users"]),
                "entry_times": as_int(row["entry_times"]),
                "entry_conversion_rate_pct": as_float(row["entry_conversion_rate_pct"]),
                "search_exposure_users": as_int(row["search_exposure_users"]),
                "search_average_rank": as_float(row["search_average_rank"]),
                "search_entry_users": as_int(row["search_entry_users"]),
                "search_entry_rate_pct": as_float(row["search_entry_rate_pct"]),
                "search_entry_share_pct": as_float(row["search_entry_share_pct"]),
                "top_search_terms": top_search_terms,
            },
            calculation=(
                "search_entry_rate_pct = search_entry_users / search_exposure_users * 100; "
                "search_entry_share_pct = search_entry_users / entry_users * 100"
            ),
            source_fields=[
                "region_type",
                "store_type",
                "exposure_users",
                "exposure_times",
                "store_average_rank",
                "entry_users",
                "entry_times",
                "entry_conversion_rate_pct",
                "search_exposure_users",
                "search_average_rank",
                "search_entry_users",
                "search_entry_rate_pct",
                "search_entry_share_pct",
                "search_term",
                "search_term_en",
                "search_term_exposure_times",
                "search_term_click_times",
                "search_term_order_times",
            ],
            confidence="high",
            limitations=VISIBILITY_ENTRY_LIMITATIONS,
            supporting_source_paths=[TOP_SEARCH_TERMS_SOURCE_PATH],
        )
    )

    facts.append(
        make_fact(
            entity_id=entity_id,
            slot="activity_lever_profile",
            value=(
                f"Store {store_id}'s March 2026 activity fields record activity involvement and related cost measures. "
                f"These metrics should be treated as operating-tool evidence, not as proof that activity caused the store's result."
            ),
            observed_values={
                "activity_original_transaction_amount": as_float(row["activity_original_transaction_amount"]),
                "activity_orders": as_int(row["activity_orders"]),
                "transaction_orders": as_int(row["transaction_orders"]),
                "activity_cost": as_float(row["activity_cost"]),
                "merchant_subsidy_amount": as_float(row["merchant_subsidy_amount"]),
                "platform_subsidy_amount": as_float(row["platform_subsidy_amount"]),
                "activity_cost_ratio_pct": as_float(row["activity_cost_ratio_pct"]),
                "activity_order_share_pct": as_float(row["activity_order_share_pct"]),
            },
            calculation=(
                "activity_order_share_pct = activity_orders / transaction_orders * 100; "
                "activity_cost_ratio_pct follows the data dictionary formula activity_cost / activity_original_transaction_amount * 100"
            ),
            source_fields=[
                "activity_original_transaction_amount",
                "activity_orders",
                "transaction_orders",
                "activity_cost",
                "merchant_subsidy_amount",
                "platform_subsidy_amount",
                "activity_cost_ratio_pct",
                "activity_order_share_pct",
            ],
            confidence="high",
            limitations=ACTIVITY_LEVER_LIMITATIONS,
        )
    )

    facts.append(
        make_fact(
            entity_id=entity_id,
            slot="transaction_conversion_profile",
            value=(
                f"Store {store_id}'s March 2026 transaction and conversion profile records transaction amount, order volume, "
                f"order-submission conversion, payment conversion, and average order value as transaction outcomes in the current monthly evidence. "
                f"It should be read alongside visibility, entry, activity, and SKU evidence."
            ),
            observed_values={
                "transaction_amount": as_float(row["transaction_amount"]),
                "transaction_orders": as_int(row["transaction_orders"]),
                "average_order_value": as_float(row["average_order_value"]),
                "order_users": as_int(row["order_users"]),
                "order_times": as_int(row["order_times"]),
                "order_conversion_rate_pct": as_float(row["order_conversion_rate_pct"]),
                "order_amount": as_float(row["order_amount"]),
                "payment_users": as_int(row["payment_users"]),
                "payment_amount": as_float(row["payment_amount"]),
                "payment_conversion_rate_pct": as_float(row["payment_conversion_rate_pct"]),
                "entry_users": as_int(row["entry_users"]),
            },
            calculation=(
                "average_order_value = transaction_amount / transaction_orders; "
                "order_conversion_rate_pct follows the backend definition order_users / entry_users * 100; "
                "payment_conversion_rate_pct = payment_users / order_users * 100"
            ),
            source_fields=[
                "transaction_amount",
                "transaction_orders",
                "average_order_value",
                "order_users",
                "order_times",
                "order_conversion_rate_pct",
                "order_amount",
                "payment_users",
                "payment_amount",
                "payment_conversion_rate_pct",
                "entry_users",
            ],
            confidence="high",
            limitations=TRANSACTION_CONVERSION_LIMITATIONS,
        )
    )

    facts.append(
        make_fact(
            entity_id=entity_id,
            slot="top3_sku_product_mix_note",
            value=(
                f"Store {store_id}'s March 2026 top-3 SKU transaction-amount evidence is retained as lightweight product-mix evidence. "
                f"It should not be treated as full SKU category-share analysis."
            ),
            observed_values={
                "top3_sku_transaction_amount": as_optional_float(row["top3_sku_transaction_amount"]),
                "transaction_amount": as_float(row["transaction_amount"]),
                "top3_sku_transaction_amount_share_pct": as_optional_float(row["top3_sku_transaction_amount_share_pct"]),
                "top_skus_by_transaction_amount": top_skus_by_amount,
            },
            calculation="top3_sku_transaction_amount_share_pct = top3_sku_transaction_amount / transaction_amount * 100",
            source_fields=[
                "sku_name",
                "sku_name_en",
                "sku_transaction_amount",
                "transaction_amount",
                "top3_sku_transaction_amount",
                "top3_sku_transaction_amount_share_pct",
                "sku_category_note",
            ],
            confidence="medium",
            source_path=SOURCE_PATH,
            supporting_source_paths=[TOP_SKUS_BY_AMOUNT_SOURCE_PATH],
            limitations=TOP3_SKU_LIMITATIONS,
        )
    )

    facts.append(
        make_fact(
            entity_id=entity_id,
            slot="single_metric_attribution_guard",
            value=(
                f"Store {store_id}'s March 2026 metrics should not be interpreted from a single metric alone. "
                f"The comparison scope is {row['comparison_scope_flag']}, with limit notes: {row['comparison_limit_notes']}."
            ),
            observed_values={
                "transaction_amount": as_float(row["transaction_amount"]),
                "transaction_orders": as_int(row["transaction_orders"]),
                "average_order_value": as_float(row["average_order_value"]),
                "store_average_rank": as_float(row["store_average_rank"]),
                "entry_conversion_rate_pct": as_float(row["entry_conversion_rate_pct"]),
                "order_conversion_rate_pct": as_float(row["order_conversion_rate_pct"]),
                "search_entry_share_pct": as_float(row["search_entry_share_pct"]),
                "activity_order_share_pct": as_float(row["activity_order_share_pct"]),
                "activity_cost_ratio_pct": as_float(row["activity_cost_ratio_pct"]),
                "top3_sku_transaction_amount_share_pct": as_optional_float(row["top3_sku_transaction_amount_share_pct"]),
                "comparison_scope_flag": row["comparison_scope_flag"],
                "comparison_limit_notes": row["comparison_limit_notes"],
            },
            calculation=(
                "comparison_limit_notes are derived from activity involvement and top-3 SKU concentration checks "
                "in the Demo 2 SQL output"
            ),
            source_fields=[
                "transaction_amount",
                "transaction_orders",
                "average_order_value",
                "store_average_rank",
                "entry_conversion_rate_pct",
                "order_conversion_rate_pct",
                "search_entry_share_pct",
                "activity_order_share_pct",
                "activity_cost_ratio_pct",
                "top3_sku_transaction_amount_share_pct",
                "comparison_scope_flag",
                "comparison_limit_notes",
            ],
            confidence="high",
            limitations=ATTRIBUTION_GUARD_LIMITATIONS,
        )
    )

OUTPUT_PATH.write_text(
    json.dumps(facts, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

print(f"Wrote {len(facts)} facts to {OUTPUT_PATH}")
