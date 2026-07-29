from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

FACTS_PATH = (
    ROOT
    / "retail_ops"
    / "outputs"
    / "generated_demo2_retail_memory_facts.json"
)

RESULTS_PATH = (
    ROOT
    / "eval"
    / "retail_decision_support_eval_results"
    / "eval_retail_demo2_facts_result.txt"
)

ENTITIES = (
    "store_B",
    "store_C",
    "store_D",
    "store_E",
    "store_F",
)

SLOT_TERMS = {
    "visibility_entry_profile": (
        "search_entry_share_pct",
        "top_search_terms",
        "traffic-source users may overlap",
        "not causal attribution",
    ),
    "activity_lever_profile": (
        "activity_order_share_pct",
        "activity_cost_ratio_pct",
        "activity metrics describe tool usage",
    ),
    "transaction_conversion_profile": (
        "transaction_amount",
        "order_conversion_rate_pct",
        "payment_conversion_rate_pct",
        "average_order_value",
    ),
    "top3_sku_product_mix_note": (
        "top3_sku_transaction_amount_share_pct",
        "top_skus_by_transaction_amount",
        "not full SKU category-share analysis",
    ),
    "single_metric_attribution_guard": (
        "comparison_scope_flag",
        "comparison_limit_notes",
        "not causal attribution",
    ),
}

EXPECTED_PERIOD = {
    "period_label": "2026-03",
    "period_start": "2026-03-01",
    "period_end": "2026-03-31",
    "period_granularity": "month",
}

EXPECTED_SUPPORTING_SOURCE_PATHS = {
    "visibility_entry_profile": (
        "retail_ops/data/demo2_top_search_terms.csv",
    ),
    "top3_sku_product_mix_note": (
        "retail_ops/data/demo2_top_skus_by_transaction_amount.csv",
    ),
}

REQUIRED_FIELDS = (
    "kind",
    "type",
    "entity_id",
    "slot",
    "period_label",
    "period_start",
    "period_end",
    "period_granularity",
    "value",
    "observed_values",
    "calculation",
    "source_fields",
    "confidence",
    "source_path",
    "lineage_path",
    "limitations",
    "is_active",
)


def non_empty_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_string_list(
    value: Any,
    field_name: str,
    errors: list[str],
) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(non_empty_text(item) for item in value)
    ):
        errors.append(
            f"{field_name} must be a non-empty string list"
        )
        return []

    return value


def validate_repository_path(
    value: Any,
    field_name: str,
    errors: list[str],
) -> None:
    if not non_empty_text(value):
        errors.append(f"{field_name} is missing or empty")
        return

    if not (ROOT / value).exists():
        errors.append(
            f"{field_name} does not exist: {value}"
        )


def validate_fact(
    fact: dict[str, Any],
    entity_id: str,
    slot: str,
) -> list[str]:
    errors: list[str] = []

    missing_fields = [
        field
        for field in REQUIRED_FIELDS
        if field not in fact
    ]

    if missing_fields:
        errors.append(
            "missing fields: " + ", ".join(missing_fields)
        )

    expected_values = {
        "kind": "retail_memory_fact",
        "type": "retail_metric_profile",
        "entity_id": entity_id,
        "slot": slot,
        **EXPECTED_PERIOD,
    }

    for field_name, expected_value in expected_values.items():
        if fact.get(field_name) != expected_value:
            errors.append(
                f"{field_name} must equal {expected_value!r}"
            )

    for field_name in (
        "value",
        "calculation",
        "confidence",
    ):
        if not non_empty_text(fact.get(field_name)):
            errors.append(
                f"{field_name} is missing or empty"
            )

    if not isinstance(fact.get("observed_values"), dict):
        errors.append("observed_values must be an object")

    validate_string_list(
        fact.get("source_fields"),
        "source_fields",
        errors,
    )

    supporting_paths: list[str] = []
    if "supporting_source_paths" in fact:
        supporting_paths = validate_string_list(
            fact.get("supporting_source_paths"),
            "supporting_source_paths",
            errors,
        )

    expected_supporting_paths = (
        EXPECTED_SUPPORTING_SOURCE_PATHS.get(slot, ())
    )
    for expected_path in expected_supporting_paths:
        if expected_path not in supporting_paths:
            errors.append(
                "supporting_source_paths must include "
                f"{expected_path}"
            )

    validate_string_list(
        fact.get("limitations"),
        "limitations",
        errors,
    )

    if fact.get("is_active") is not True:
        errors.append("is_active must be true")

    validate_repository_path(
        fact.get("source_path"),
        "source_path",
        errors,
    )

    validate_repository_path(
        fact.get("lineage_path"),
        "lineage_path",
        errors,
    )

    for index, path in enumerate(supporting_paths):
        validate_repository_path(
            path,
            f"supporting_source_paths[{index}]",
            errors,
        )

    serialized = json.dumps(
        fact,
        ensure_ascii=False,
        sort_keys=True,
    )

    missing_terms = [
        term
        for term in SLOT_TERMS[slot]
        if term not in serialized
    ]

    if missing_terms:
        errors.append(
            "missing boundary terms: "
            + ", ".join(missing_terms)
        )

    return errors


def load_facts(
    failures: list[str],
) -> list[dict[str, Any]]:
    try:
        content = json.loads(
            FACTS_PATH.read_text(encoding="utf-8")
        )
    except Exception as exc:
        failures.append(
            f"could not read facts file: {exc}"
        )
        return []

    if not isinstance(content, list):
        failures.append(
            "facts file must contain a JSON list"
        )
        return []

    facts: list[dict[str, Any]] = []

    for index, item in enumerate(content):
        if isinstance(item, dict):
            facts.append(item)
        else:
            failures.append(
                f"fact #{index} is not an object"
            )

    return facts


def main() -> int:
    failures: list[str] = []
    result_lines: list[str] = []
    facts = load_facts(failures)

    expected_keys = {
        (entity_id, slot)
        for entity_id in ENTITIES
        for slot in SLOT_TERMS
    }

    indexed: dict[
        tuple[Any, Any],
        list[dict[str, Any]],
    ] = {}

    for fact in facts:
        key = (
            fact.get("entity_id"),
            fact.get("slot"),
        )
        indexed.setdefault(key, []).append(fact)

    if len(facts) != len(expected_keys):
        failures.append(
            f"expected {len(expected_keys)} facts; "
            f"found {len(facts)}"
        )

    unexpected_keys = sorted(
        set(indexed) - expected_keys,
        key=lambda item: (str(item[0]), str(item[1])),
    )

    if unexpected_keys:
        failures.append(
            "unexpected entity-slot combinations: "
            + ", ".join(
                f"{entity_id}/{slot}"
                for entity_id, slot in unexpected_keys
            )
        )

    passed = 0

    for entity_id in ENTITIES:
        for slot in SLOT_TERMS:
            case_name = f"{entity_id}/{slot}"
            matches = indexed.get((entity_id, slot), [])

            if len(matches) != 1:
                issue = (
                    "missing fact"
                    if not matches
                    else f"{len(matches)} duplicate facts"
                )
                failures.append(f"{case_name}: {issue}")
                result_lines.append(
                    f"FAIL {case_name}: {issue}"
                )
                continue

            errors = validate_fact(
                matches[0],
                entity_id,
                slot,
            )

            if errors:
                failures.extend(
                    f"{case_name}: {error}"
                    for error in errors
                )
                result_lines.append(
                    f"FAIL {case_name}: "
                    + " | ".join(errors)
                )
            else:
                passed += 1
                result_lines.append(
                    f"PASS {case_name}"
                )

    expected_count = len(expected_keys)
    status = "passed" if not failures else "failed"

    summary = [
        (
            "Retail Demo 2 fact-contract coverage "
            f"result: {passed}/{expected_count} "
            "entity-slot contracts passed"
        ),
        (
            f"Expected coverage: {len(ENTITIES)} stores x "
            f"{len(SLOT_TERMS)} slots = {expected_count}"
        ),
        f"Status: {status}",
        f"Passed contracts: {passed}",
        f"Failed checks: {len(failures)}",
        "",
        *result_lines,
    ]

    if failures:
        summary.extend(
            [
                "",
                "Failure details:",
                *[f"- {failure}" for failure in failures],
            ]
        )

    summary.append("")

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_PATH.write_text(
        "\n".join(summary),
        encoding="utf-8",
    )

    print("\n".join(summary))

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
