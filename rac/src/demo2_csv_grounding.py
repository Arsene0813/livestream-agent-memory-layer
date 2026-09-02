from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, NamedTuple


SNAPSHOT_SOURCE_PATH = (
    "retail_ops/outputs/"
    "demo2_cross_store_comparability_output.csv"
)
REPEATED_WINDOW_SOURCE_PATH = (
    "retail_ops/outputs/"
    "repeated_window_panel_summary_output.csv"
)
STORE_IDS = ("B", "C", "D", "E", "F")
PERIOD_MONTH = "2026-03"


class RecordSpec(NamedTuple):
    source_path: str
    grounding_role: str
    key_fields: tuple[str, ...]
    fields: tuple[str, ...]


FACTOR_RECORD_SPECS = {
    "store_type": RecordSpec(
        SNAPSHOT_SOURCE_PATH,
        "context_evidence",
        ("store_id", "period_month"),
        ("store_type",),
    ),
    "order_volume": RecordSpec(
        SNAPSHOT_SOURCE_PATH,
        "quantitative_evidence",
        ("store_id", "period_month"),
        ("transaction_orders",),
    ),
    "transaction_amount": RecordSpec(
        SNAPSHOT_SOURCE_PATH,
        "quantitative_evidence",
        ("store_id", "period_month"),
        ("transaction_amount",),
    ),
    "activity_intensity": RecordSpec(
        SNAPSHOT_SOURCE_PATH,
        "quantitative_evidence",
        ("store_id", "period_month"),
        (
            "activity_orders",
            "activity_order_share_pct",
            "activity_cost",
            "activity_cost_ratio_pct",
        ),
    ),
    "sku_structure": RecordSpec(
        SNAPSHOT_SOURCE_PATH,
        "product_mix_evidence",
        ("store_id", "period_month"),
        (
            "top3_sku_transaction_amount",
            "top3_sku_transaction_amount_share_pct",
        ),
    ),
    "repeated_reporting_windows": RecordSpec(
        REPEATED_WINDOW_SOURCE_PATH,
        "quantitative_evidence",
        ("store_id",),
        (
            "observed_month_count",
            "feb_transaction_amount",
            "mar_transaction_amount",
            "apr_transaction_amount",
            "feb_transaction_orders",
            "mar_transaction_orders",
            "apr_transaction_orders",
        ),
    ),
}


def supports_demo2_record(
    question_type: str | None,
    factor_id: str,
    source_path: str,
) -> bool:
    spec = FACTOR_RECORD_SPECS.get(factor_id)
    return bool(
        question_type == "comparability_judgment"
        and spec
        and source_path == spec.source_path
    )


def resolve_demo2_record(
    packet: dict[str, Any],
    *,
    factor_id: str,
    root: Path,
) -> dict[str, Any]:
    spec = FACTOR_RECORD_SPECS[factor_id]
    path = root / spec.source_path

    result: dict[str, Any] = {
        "factor_id": factor_id,
        "evidence_id": packet.get("evidence_id"),
        "source_type": "csv",
        "source_path": spec.source_path,
        "source_exists": path.is_file(),
        "grounding_role": spec.grounding_role,
        "grounding_status": "source_missing",
        "snippets": [],
        "record_scope": {},
        "evidence_fields": list(spec.fields),
        "evidence_values": [],
        "original_claim_supported": packet.get(
            "claim_supported"
        ),
        "original_limitations": packet.get(
            "limitations",
            [],
        ),
        "resolver_limitations": [
            (
                "Deterministic B-F CSV record selection "
                "for the declared comparison scope."
            ),
            (
                "Interpretation remains bounded by the "
                "source contract and reporting scope."
            ),
        ],
    }

    if not path.is_file():
        result["absolute_path_checked"] = str(path)
        return result

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        rows = list(reader)

    missing_fields = sorted(
        {*spec.key_fields, *spec.fields}
        - set(headers)
    )

    if missing_fields:
        result["grounding_status"] = (
            "record_contract_error"
        )
        result["record_contract_errors"] = [
            "Missing fields: "
            + ", ".join(missing_fields)
        ]
        return result

    selected = [
        row
        for row in rows
        if row["store_id"] in STORE_IDS
        and (
            "period_month" not in spec.key_fields
            or row["period_month"] == PERIOD_MONTH
        )
    ]

    by_key: dict[
        tuple[str, ...],
        dict[str, str],
    ] = {}
    duplicates: set[tuple[str, ...]] = set()

    for row in selected:
        key = tuple(
            row[field]
            for field in spec.key_fields
        )

        if key in by_key:
            duplicates.add(key)

        by_key[key] = row

    expected_keys = [
        (
            (store_id, PERIOD_MONTH)
            if "period_month" in spec.key_fields
            else (store_id,)
        )
        for store_id in STORE_IDS
    ]

    missing_keys = [
        key
        for key in expected_keys
        if key not in by_key
    ]

    if duplicates or missing_keys:
        errors = []

        if duplicates:
            errors.append(
                "Duplicate row keys: "
                + str(sorted(duplicates))
            )

        if missing_keys:
            errors.append(
                "Missing row keys: "
                + str(missing_keys)
            )

        result["grounding_status"] = (
            "record_contract_error"
        )
        result["record_contract_errors"] = errors
        return result

    evidence_values = [
        {
            "row_key": dict(
                zip(spec.key_fields, key)
            ),
            "values": {
                field: by_key[key][field]
                for field in spec.fields
            },
        }
        for key in expected_keys
    ]

    result.update(
        {
            "grounding_status": "record_matched",
            "record_scope": {
                "key_fields": list(
                    spec.key_fields
                ),
                "row_count": len(
                    evidence_values
                ),
                "row_keys": [
                    item["row_key"]
                    for item in evidence_values
                ],
            },
            "evidence_values": evidence_values,
        }
    )

    return result
