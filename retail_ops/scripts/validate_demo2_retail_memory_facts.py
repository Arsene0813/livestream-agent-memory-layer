from __future__ import annotations

import json
from pathlib import Path

FACTS_PATH = Path("retail_ops/outputs/generated_demo2_retail_memory_facts.json")

EXPECTED_ENTITIES = {"store_B", "store_C", "store_D", "store_E", "store_F"}

EXPECTED_SLOTS = {
    "visibility_entry_profile",
    "activity_lever_profile",
    "transaction_conversion_profile",
    "order_quality_pressure_profile",
    "top3_sku_product_mix_note",
    "single_metric_attribution_guard",
}

REQUIRED_KEYS = [
    "kind",
    "type",
    "entity_id",
    "slot",
    "period_label",
    "period_start",
    "period_end",
    "value",
    "observed_values",
    "calculation",
    "source_fields",
    "confidence",
    "source_path",
    "supporting_source_paths",
    "lineage_path",
    "limitations",
    "is_active",
    "period_granularity",
]

REQUIRED_TRANSACTION_FIELDS = {
    "transaction_amount",
    "transaction_orders",
    "average_order_value",
    "estimated_income_proxy",
    "order_users",
    "order_times",
    "order_conversion_rate_pct",
    "order_amount",
    "payment_users",
    "payment_amount",
    "payment_conversion_rate_pct",
}

data = json.loads(FACTS_PATH.read_text(encoding="utf-8"))

expected_fact_count = len(EXPECTED_ENTITIES) * len(EXPECTED_SLOTS)
if len(data) != expected_fact_count:
    raise SystemExit(f"Expected {expected_fact_count} Demo 2 facts, got {len(data)}")

entities = {fact["entity_id"] for fact in data}
slots = {fact["slot"] for fact in data}

if entities != EXPECTED_ENTITIES:
    raise SystemExit(f"Unexpected entities: {sorted(entities)}")

if slots != EXPECTED_SLOTS:
    raise SystemExit(f"Unexpected slots: {sorted(slots)}")

seen_pairs = set()

for fact in data:
    missing_keys = [key for key in REQUIRED_KEYS if key not in fact]
    if missing_keys:
        raise SystemExit(f"Missing keys in fact: {missing_keys}")

    pair = (fact["entity_id"], fact["slot"])

    if pair in seen_pairs:
        raise SystemExit(f"Duplicate entity-slot pair: {pair}")
    seen_pairs.add(pair)

    if fact["kind"] != "retail_memory_fact":
        raise SystemExit(f"{pair}: bad kind")

    if fact["type"] != "retail_metric_profile":
        raise SystemExit(f"{pair}: bad type")

    if fact["period_label"] != "2026-03":
        raise SystemExit(f"{pair}: bad period_label")

    if fact["period_start"] != "2026-03-01":
        raise SystemExit(f"{pair}: bad period_start")

    if fact["period_end"] != "2026-03-31":
        raise SystemExit(f"{pair}: bad period_end")

    if fact["period_granularity"] != "month":
        raise SystemExit(f"{pair}: bad period_granularity")

    if fact["is_active"] is not True:
        raise SystemExit(f"{pair}: fact should be active")

    if fact["confidence"] not in {"high", "medium"}:
        raise SystemExit(f"{pair}: unexpected confidence")

    if not isinstance(fact["observed_values"], dict):
        raise SystemExit(f"{pair}: observed_values must be an object")

    if not isinstance(fact["source_fields"], list) or not fact["source_fields"]:
        raise SystemExit(f"{pair}: source_fields must be a non-empty list")

    if not isinstance(fact["supporting_source_paths"], list) or not fact["supporting_source_paths"]:
        raise SystemExit(f"{pair}: supporting_source_paths must be a non-empty list")

    if fact["source_path"] not in fact["supporting_source_paths"]:
        raise SystemExit(f"{pair}: source_path must be included in supporting_source_paths")

    if any(not isinstance(item, str) or not item for item in fact["supporting_source_paths"]):
        raise SystemExit(f"{pair}: supporting_source_paths must contain non-empty strings")

    if not isinstance(fact["limitations"], list) or not fact["limitations"]:
        raise SystemExit(f"{pair}: limitations must be a non-empty list")

    if fact["slot"] == "visibility_entry_profile":
        if "retail_ops/data/demo2_top_search_terms.csv" not in fact["supporting_source_paths"]:
            raise SystemExit(f"{pair}: visibility fact must include top-search source path")

    if fact["slot"] == "top3_sku_product_mix_note":
        if "retail_ops/data/demo2_top_skus_by_transaction_amount.csv" not in fact["supporting_source_paths"]:
            raise SystemExit(f"{pair}: top-SKU fact must include top-SKU transaction amount source path")

    if fact["slot"] == "transaction_conversion_profile":
        observed_fields = set(fact["observed_values"])
        missing_transaction_fields = sorted(REQUIRED_TRANSACTION_FIELDS - observed_fields)
        if missing_transaction_fields:
            raise SystemExit(
                f"{pair}: missing transaction/conversion observed fields: "
                f"{missing_transaction_fields}"
            )

print("Demo 2 retail memory facts validation PASSED.")
print(f"Checked facts: {expected_fact_count}")
print("Checked entities: store_B, store_C, store_D, store_E, store_F")
print("Checked slots:", ", ".join(sorted(EXPECTED_SLOTS)))
print(
    "Checked schema, period, source fields, supporting source paths, "
    "limitations, transaction/conversion fields, and active status."
)
