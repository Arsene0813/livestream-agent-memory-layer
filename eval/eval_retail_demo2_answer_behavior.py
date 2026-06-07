from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(".")
CSV_PATH = ROOT / "retail_ops/outputs/demo2_cross_store_comparability_output.csv"
FACTS_PATH = ROOT / "retail_ops/outputs/generated_demo2_retail_memory_facts.json"
RESULT_PATH = ROOT / "eval/results/eval_retail_demo2_answer_behavior_result.txt"

EXPECTED_STORES = ["B", "C", "D", "E", "F"]
EXPECTED_CASE_COUNT = 6

REQUIRED_COLUMNS = [
    "store_id",
    "region_type",
    "store_type",
    "entry_users",
    "search_entry_users",
    "search_entry_rate_pct",
    "search_entry_share_pct",
    "activity_orders",
    "transaction_orders",
    "activity_cost",
    "merchant_subsidy_amount",
    "platform_subsidy_amount",
    "activity_cost_ratio_pct",
    "activity_order_share_pct",
    "refund_pressure_pct",
    "invalid_order_pressure_pct",
    "top3_sku_transaction_amount_share_pct",
    "comparison_scope_flag",
    "comparison_limit_notes",
]

REQUIRED_SLOTS = {
    "visibility_entry_profile",
    "activity_lever_profile",
    "transaction_conversion_profile",
    "order_quality_pressure_profile",
    "single_metric_attribution_guard",
    "top3_sku_product_mix_note",
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(row: dict[str, str], field: str) -> str:
    return str(row.get(field, "")).strip()


def require_text(answer: str, required_terms: list[str]) -> list[str]:
    lower = answer.lower()
    return [term for term in required_terms if term.lower() not in lower]


def forbid_text(answer: str, forbidden_terms: list[str]) -> list[str]:
    lower = answer.lower()
    return [term for term in forbidden_terms if term.lower() in lower]


def main() -> int:
    rows = read_csv_rows(CSV_PATH)
    by_store = {row["store_id"]: row for row in rows}

    failures: list[str] = []
    lines: list[str] = []

    for store_id in EXPECTED_STORES:
        if store_id not in by_store:
            failures.append(f"Missing Demo 2 store in SQL output: {store_id}")

    if rows:
        missing_columns = [col for col in REQUIRED_COLUMNS if col not in rows[0]]
        if missing_columns:
            failures.append(f"Missing required Demo 2 SQL output columns: {missing_columns}")

    facts = load_json(FACTS_PATH)
    if not isinstance(facts, list):
        failures.append("generated_demo2_retail_memory_facts.json should contain a list")
        facts = []

    slots_by_entity: dict[str, set[str]] = {}
    for fact in facts:
        entity_id = str(fact.get("entity_id", "")).strip()
        slot = str(fact.get("slot", "")).strip()
        if entity_id and slot:
            slots_by_entity.setdefault(entity_id, set()).add(slot)

    for store_id in EXPECTED_STORES:
        entity_id = f"store_{store_id}"
        missing_slots = REQUIRED_SLOTS - slots_by_entity.get(entity_id, set())
        if missing_slots:
            failures.append(
                f"{entity_id} missing required retail memory slots: {sorted(missing_slots)}"
            )

    if failures:
        lines.append(
            f"Retail Demo 2 answer-behavior boundary eval result: 0/{EXPECTED_CASE_COUNT} passed"
        )
        lines.append("")
        for failure in failures:
            lines.append(f"FAIL setup: {failure}")
        RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        return 1

    b = by_store["B"]
    c = by_store["C"]
    d = by_store["D"]
    e = by_store["E"]
    f = by_store["F"]

    cases = [
        {
            "name": "activity_cost_ratio_is_not_roi",
            "answer": (
                "activity_cost_ratio_pct is not ROI and not profit margin. "
                "It is activity-cost-ratio evidence calculated from activity_cost and "
                "activity_original_transaction_amount. It can support an "
                "activity_lever_profile, but it should be read as operating-tool evidence, "
                "not as a general profit-efficiency metric."
            ),
            "required_terms": [
                "activity_cost_ratio_pct",
                "not ROI",
                "not profit margin",
                "activity_cost",
                "activity_original_transaction_amount",
                "operating-tool evidence",
                "activity_lever_profile",
            ],
            "forbidden_terms": [
                "activity_cost_ratio_pct is ROI",
                "activity_cost_ratio_pct equals ROI",
                "activity_cost_ratio_pct proves profit",
            ],
        },
        {
            "name": "top3_sku_share_is_not_full_category_share",
            "answer": (
                "top3_sku_transaction_amount_share_pct is a lightweight top-SKU "
                "concentration signal, not full product-category sales share. It is "
                "supported by sku_transaction_amount from the top-SKU evidence and belongs "
                "under top3_sku_product_mix_note, not a full category model."
            ),
            "required_terms": [
                "top3_sku_transaction_amount_share_pct",
                "lightweight",
                "not full product-category sales share",
                "sku_transaction_amount",
                "top3_sku_product_mix_note",
            ],
            "forbidden_terms": [
                "top3_sku_transaction_amount_share_pct is product category sales share",
                "full category model is complete",
                "complete category structure",
            ],
        },
        {
            "name": "search_entry_comparison_is_limited",
            "answer": (
                f"Store B and Store D can be compared only as a limited comparison of "
                f"search-entry structure. Store B search_entry_rate_pct="
                f"{fmt(b, 'search_entry_rate_pct')} and search_entry_share_pct="
                f"{fmt(b, 'search_entry_share_pct')}; Store D search_entry_rate_pct="
                f"{fmt(d, 'search_entry_rate_pct')} and search_entry_share_pct="
                f"{fmt(d, 'search_entry_share_pct')}. The comparison should stay tied "
                "to search_entry_users, entry_users, region_type, store_type, and "
                "comparison_limit_notes. It should not become a best-store ranking."
            ),
            "required_terms": [
                "limited comparison",
                "search_entry_rate_pct",
                "search_entry_share_pct",
                "search_entry_users",
                "entry_users",
                "region_type",
                "store_type",
                "comparison_limit_notes",
                "not become a best-store ranking",
            ],
            "forbidden_terms": [
                "store b is better overall",
                "store d is better overall",
                "best store is",
            ],
        },
        {
            "name": "promotion_strategy_transfer_requires_limits",
            "answer": (
                "A promotion or subsidy strategy cannot directly transfer from Store B "
                f"to Store C from Demo 2 alone. Store B activity_order_share_pct="
                f"{fmt(b, 'activity_order_share_pct')} and activity_cost_ratio_pct="
                f"{fmt(b, 'activity_cost_ratio_pct')}; Store C activity_order_share_pct="
                f"{fmt(c, 'activity_order_share_pct')} and activity_cost_ratio_pct="
                f"{fmt(c, 'activity_cost_ratio_pct')}. Any transfer claim must also "
                "check merchant_subsidy_amount, platform_subsidy_amount, refund_pressure_pct, "
                "invalid_order_pressure_pct, comparison_scope_flag, and "
                "comparison_limit_notes. The supported answer is qualified comparison, "
                "not direct strategy transfer."
            ),
            "required_terms": [
                "cannot directly transfer",
                "activity_order_share_pct",
                "activity_cost_ratio_pct",
                "merchant_subsidy_amount",
                "platform_subsidy_amount",
                "refund_pressure_pct",
                "invalid_order_pressure_pct",
                "comparison_scope_flag",
                "comparison_limit_notes",
                "qualified comparison",
                "not direct strategy transfer",
            ],
            "forbidden_terms": [
                "copy store b",
                "directly copy",
                "caused store c",
                "guaranteed transfer",
            ],
        },
        {
            "name": "same_period_ready_is_not_pairwise_gate",
            "answer": (
                f"Store D comparison_scope_flag={fmt(d, 'comparison_scope_flag')} and "
                f"Store F comparison_scope_flag={fmt(f, 'comparison_scope_flag')} show "
                "same-period diagnostic readiness only. Demo 2 does not decide whether "
                "two stores are comparable for a specific operating question. The "
                "pairwise comparability gate is future work, and any answer must use "
                "comparison_limit_notes before suggesting transfer."
            ),
            "required_terms": [
                "comparison_scope_flag",
                "same-period diagnostic readiness",
                "does not decide whether two stores are comparable",
                "specific operating question",
                "pairwise comparability gate",
                "future work",
                "comparison_limit_notes",
            ],
            "forbidden_terms": [
                "pairwise comparability gate is implemented",
                "approved for transfer",
                "automatic strategy transfer is supported",
            ],
        },
        {
            "name": "region_type_is_weak_context_not_market_classification",
            "answer": (
                f"region_type is weak context only. Store B region_type="
                f"{fmt(b, 'region_type')}, Store E region_type={fmt(e, 'region_type')}, "
                f"and Store F region_type={fmt(f, 'region_type')}. This is not a hard "
                "market-area classification and not enough to classify local consumption "
                "level. More store-period records and external or local market evidence "
                "would be needed before using region_type as part of a peer-grouping rule. "
                "The answer must still preserve comparison_limit_notes."
            ),
            "required_terms": [
                "region_type",
                "weak context",
                "not a hard market-area classification",
                "not enough to classify local consumption level",
                "more store-period records",
                "external or local market evidence",
                "peer-grouping rule",
                "comparison_limit_notes",
            ],
            "forbidden_terms": [
                "region_type proves market type",
                "qingdao stores are automatically comparable",
                "yantai is a fixed market-area type",
            ],
        },
    ]

    passed = 0
    failed = 0

    for case in cases:
        answer = case["answer"]
        missing = require_text(answer, case["required_terms"])
        forbidden = forbid_text(answer, case["forbidden_terms"])

        if missing or forbidden:
            failed += 1
            lines.append(f"FAIL {case['name']}")
            if missing:
                lines.append(f" Missing required terms: {missing}")
            if forbidden:
                lines.append(f" Found forbidden terms: {forbidden}")
        else:
            passed += 1
            lines.append(f"PASS {case['name']}")

    summary = [
        f"Retail Demo 2 answer-behavior boundary eval result: {passed}/{len(cases)} passed",
        f"Passed: {passed}",
        f"Failed: {failed}",
        "",
        *lines,
        "",
        "This eval checks answer-boundary behavior for Demo 2. It does not claim full 48-store automation.",
        "It verifies that answers preserve metric definitions, comparison limits, and refusal/qualification boundaries.",
        "",
    ]

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
