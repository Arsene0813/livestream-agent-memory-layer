from __future__ import annotations

import json
import sys
from pathlib import Path


FACTS_PATH = Path("retail_ops/outputs/generated_demo2_retail_memory_facts.json")
RESULTS_PATH = Path("eval/results/eval_retail_demo2_facts_result.txt")

EXPECTED_CASES = [
    {
        "name": "store_b_visibility_entry_profile",
        "entity_id": "store_B",
        "slot": "visibility_entry_profile",
        "expected_terms": [
            "search_entry_share_pct",
            "top_search_terms",
            "traffic-source users may overlap",
        ],
    },
    {
        "name": "store_c_top3_sku_product_mix_note",
        "entity_id": "store_C",
        "slot": "top3_sku_product_mix_note",
        "expected_terms": [
            "top3_sku_transaction_amount_share_pct",
            "top_skus_by_transaction_amount",
            "not full SKU category-share analysis",
        ],
    },
    {
        "name": "store_d_activity_lever_profile",
        "entity_id": "store_D",
        "slot": "activity_lever_profile",
        "expected_terms": [
            "activity_order_share_pct",
            "activity_cost_ratio_pct",
            "activity metrics describe tool usage",
        ],
    },
    {
        "name": "store_e_order_quality_pressure_profile",
        "entity_id": "store_E",
        "slot": "order_quality_pressure_profile",
        "expected_terms": [
            "refund_pressure_pct",
            "invalid_order_pressure_pct",
            "not an exact original-order cohort refund rate",
        ],
    },
    {
        "name": "store_f_transaction_conversion_profile",
        "entity_id": "store_F",
        "slot": "transaction_conversion_profile",
        "expected_terms": [
            "transaction_amount",
            "order_conversion_rate_pct",
            "payment_conversion_rate_pct",
            "estimated_income_proxy is platform-displayed and not audited profit",
        ],
    },
    {
        "name": "store_f_single_metric_attribution_guard",
        "entity_id": "store_F",
        "slot": "single_metric_attribution_guard",
        "expected_terms": [
            "comparison_scope_flag",
            "comparison_limit_notes",
            "not causal attribution",
        ],
    },
]

facts = json.loads(FACTS_PATH.read_text(encoding="utf-8"))
index = {(fact["entity_id"], fact["slot"]): fact for fact in facts}

passed = 0
failed = 0
lines = []

for case in EXPECTED_CASES:
    key = (case["entity_id"], case["slot"])
    fact = index.get(key)

    if fact is None:
        failed += 1
        lines.append(f"FAIL {case['name']}: missing fact {key}")
        continue

    serialized = json.dumps(fact, ensure_ascii=False)
    missing_terms = [term for term in case["expected_terms"] if term not in serialized]

    if missing_terms:
        failed += 1
        lines.append(f"FAIL {case['name']}: missing terms {missing_terms}")
        continue

    passed += 1
    lines.append(f"PASS {case['name']}")

summary = [
    f"Retail Demo 2 facts eval result: {passed}/{len(EXPECTED_CASES)} passed",
    f"Passed: {passed}",
    f"Failed: {failed}",
    "",
    *lines,
    "",
]

RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
RESULTS_PATH.write_text("\n".join(summary), encoding="utf-8")

print("\n".join(summary))
sys.exit(0 if failed == 0 else 1)
