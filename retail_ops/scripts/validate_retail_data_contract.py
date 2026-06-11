from __future__ import annotations

import csv
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = ROOT / "retail_ops" / "outputs" / "retail_data_contract_validation_result.txt"

FORBIDDEN_ALIASES = [
    "store_visitors",
    "store_entry_count",
    "search_visitors",
    "activity_gross_revenue",
    "estimated_order_income",
    "paid_users",
    "paid_amount",
]

REQUIRED_CANONICAL_FIELDS = [
    "store_id",
    "period_start",
    "period_end",
    "region_type",
    "store_type",
    "entry_users",
    "entry_times",
    "search_entry_users",
    "payment_users",
    "payment_amount",
    "activity_original_transaction_amount",
    "estimated_income_proxy",
    "exposure_users",
    "exposure_times",
    "order_users",
    "order_times",
    "order_amount",
    "transaction_amount",
    "transaction_orders",
    "average_order_value",
    "activity_orders",
    "activity_cost",
    "merchant_subsidy_amount",
    "platform_subsidy_amount",
    "refund_amount",
]

REQUIRED_DEMO2_OUTPUT_FIELDS = [
    "search_entry_share_pct",
    "activity_order_share_pct",
    "activity_cost_ratio_pct",
    "top3_sku_transaction_amount_share_pct",
    "comparison_scope_flag",
    "comparison_limit_notes",
]

REQUIRED_BOUNDARY_PHRASES = [
    "order_conversion_rate_pct",
    "order_users / entry_users",
    "activity_cost_ratio_pct",
    "not traditional ROI",
    "estimated_income_proxy",
    "not audited profit",
    "top3_sku_transaction_amount_share_pct",
    "not full product-category share",
    "region_type remains weak context only",
    "not a hard market-area classification",
]

CANONICAL_MEMORY_SLOTS = {
    "visibility_entry_profile",
    "activity_lever_profile",
    "transaction_conversion_profile",
    "single_metric_attribution_guard",
    "top3_sku_product_mix_note",
}

KNOWN_HELPER_FIELDS = {
    "search_term",
    "search_term_en",
    "search_term_exposure_times",
    "search_term_click_times",
    "search_term_order_times",
    "sku_name",
    "sku_name_en",
    "sku_transaction_amount",
    "sales_volume",
    "sku_category_note",
}

REQUIRED_FILES = [
    "retail_ops/data/DATA_DICTIONARY.md",
    "retail_ops/TECHNICAL_APPENDIX.md",
    "retail_ops/data/store_a_monthly_metrics.csv",
    "retail_ops/data/store_a_top_skus.csv",
    "retail_ops/data/demo2_store_period_metrics.csv",
    "retail_ops/data/demo2_top_search_terms.csv",
    "retail_ops/data/demo2_top_skus_by_transaction_amount.csv",
    "retail_ops/sql/01_store_a_month_over_month_diagnostic.sql",
    "retail_ops/sql/02_demo2_cross_store_comparability.sql",
    "retail_ops/outputs/store_a_demo1_sql_output.csv",
    "retail_ops/outputs/store_a_demo1_interpretation_summary.csv",
    "retail_ops/outputs/demo2_cross_store_comparability_output.csv",
    "retail_ops/outputs/generated_retail_memory_facts.json",
    "retail_ops/outputs/generated_demo2_retail_memory_facts.json",
]


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def read_csv_headers(relative_path: str) -> set[str]:
    with (ROOT / relative_path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        return set(next(reader))


def extract_backticked_fields(text: str) -> set[str]:
    return set(re.findall(r"`([a-zA-Z_][a-zA-Z0-9_]*)`", text))


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        ROOT / line
        for line in result.stdout.splitlines()
        if (ROOT / line).exists()
    ]
def source_exists(relative_path: str) -> bool:
    return (ROOT / relative_path).exists()


def validate_generated_facts(
    *,
    relative_path: str,
    allowed_entities: set[str],
    known_fields: set[str],
    failures: list[str],
) -> None:
    path = ROOT / relative_path

    try:
        facts = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        failures.append(f"{relative_path} is not valid JSON: {exc}")
        return

    if not isinstance(facts, list):
        failures.append(f"{relative_path} should contain a list of facts")
        return

    required_keys = {
        "kind",
        "type",
        "entity_id",
        "slot",
        "period_start",
        "period_end",
        "value",
        "observed_values",
        "source_fields",
        "confidence",
        "source_path",
        "lineage_path",
        "limitations",
        "is_active",
    }

    for index, fact in enumerate(facts):
        if not isinstance(fact, dict):
            failures.append(f"{relative_path} fact #{index} is not an object")
            continue

        missing_keys = sorted(required_keys - set(fact))
        if missing_keys:
            failures.append(
                f"{relative_path} fact #{index} missing keys: {', '.join(missing_keys)}"
            )

        entity_id = fact.get("entity_id")
        if entity_id not in allowed_entities:
            failures.append(
                f"{relative_path} fact #{index} has unsupported entity_id `{entity_id}`"
            )

        slot = fact.get("slot")
        if slot not in CANONICAL_MEMORY_SLOTS:
            failures.append(f"{relative_path} fact #{index} has non-canonical slot `{slot}`")

        source_path = fact.get("source_path")
        if not isinstance(source_path, str) or not source_path.strip():
            failures.append(f"{relative_path} fact #{index} has missing source_path")
        elif not source_exists(source_path):
            failures.append(
                f"{relative_path} fact #{index} source_path does not exist: {source_path}"
            )

        source_fields = fact.get("source_fields")
        if not isinstance(source_fields, list):
            failures.append(f"{relative_path} fact #{index} has non-list source_fields")
            continue

        for field in source_fields:
            if field not in known_fields and field not in KNOWN_HELPER_FIELDS:
                failures.append(
                    f"{relative_path} fact #{index} source field `{field}` is not known"
                )

        limitations = fact.get("limitations")
        if not isinstance(limitations, list) or not limitations:
            failures.append(f"{relative_path} fact #{index} has missing limitations")


def write_report(lines: list[str]) -> None:
    text = "\n".join(lines).rstrip() + "\n"
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(text, encoding="utf-8")
    print(text, end="")


def main() -> int:
    failures: list[str] = []

    for relative_path in REQUIRED_FILES:
        if not (ROOT / relative_path).exists():
            failures.append(f"Missing required file: {relative_path}")

    if failures:
        report = ["Retail data contract validation FAILED.", *[f"[FAIL] {x}" for x in failures]]
        write_report(report)
        return 1

    dictionary = read_text("retail_ops/data/DATA_DICTIONARY.md")
    lineage = read_text("retail_ops/TECHNICAL_APPENDIX.md")
    demo1_sql = read_text("retail_ops/sql/01_store_a_month_over_month_diagnostic.sql")
    demo2_sql = read_text("retail_ops/sql/02_demo2_cross_store_comparability.sql")

    demo1_source_headers = read_csv_headers("retail_ops/data/store_a_monthly_metrics.csv")
    demo1_top_sku_headers = read_csv_headers("retail_ops/data/store_a_top_skus.csv")
    demo1_output_headers = read_csv_headers("retail_ops/outputs/store_a_demo1_sql_output.csv")
    demo1_summary_headers = read_csv_headers(
        "retail_ops/outputs/store_a_demo1_interpretation_summary.csv"
    )
    demo2_source_headers = read_csv_headers("retail_ops/data/demo2_store_period_metrics.csv")
    demo2_output_headers = read_csv_headers(
        "retail_ops/outputs/demo2_cross_store_comparability_output.csv"
    )

    known_fields = (
        extract_backticked_fields(dictionary)
        | demo1_source_headers
        | demo1_top_sku_headers
        | demo1_output_headers
        | demo1_summary_headers
        | demo2_source_headers
        | demo2_output_headers
        | KNOWN_HELPER_FIELDS
    )

    validator_paths = {
        (ROOT / "retail_ops/scripts/validate_retail_data_contract.py").resolve(),
        (ROOT / "scripts/validate_demo2_retail_endpoint_boundary.py").resolve(),
    }

    for path in tracked_files():
        if path.resolve() in validator_paths:
            continue
        if path.suffix.lower() not in {".md", ".py", ".sql", ".csv", ".json", ".txt", ".yml", ".yaml"}:
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        for alias in FORBIDDEN_ALIASES:
            if alias in text:
                failures.append(f"Forbidden alias `{alias}` found in {path.relative_to(ROOT)}")

    for field in REQUIRED_CANONICAL_FIELDS:
        if field not in dictionary:
            failures.append(f"Required canonical field `{field}` missing from DATA_DICTIONARY.md")
        if field not in known_fields:
            failures.append(f"Required canonical field `{field}` missing from known source/output fields")

    for phrase in REQUIRED_BOUNDARY_PHRASES:
        if phrase not in dictionary:
            failures.append(f"DATA_DICTIONARY.md missing required boundary phrase: {phrase}")

    for field in REQUIRED_DEMO2_OUTPUT_FIELDS:
        if field not in demo2_output_headers:
            failures.append(f"Demo 2 output missing required field `{field}`")
        if field not in demo2_sql:
            failures.append(f"Demo 2 SQL missing required field `{field}`")

    critical_lineage_fields = [
        "entry_users",
        "search_entry_users",
        "activity_original_transaction_amount",
        "estimated_income_proxy",
        "order_conversion_rate_pct",
        "activity_cost_ratio_pct",
        "top3_sku_transaction_amount_share_pct",
    ]
    for field in critical_lineage_fields:
        if field not in lineage:
            failures.append(f"Critical lineage field `{field}` missing from TECHNICAL_APPENDIX.md")

    validate_generated_facts(
        relative_path="retail_ops/outputs/generated_retail_memory_facts.json",
        allowed_entities={"store_A"},
        known_fields=known_fields,
        failures=failures,
    )

    validate_generated_facts(
        relative_path="retail_ops/outputs/generated_demo2_retail_memory_facts.json",
        allowed_entities={"store_B", "store_C", "store_D", "store_E", "store_F"},
        known_fields=known_fields,
        failures=failures,
    )

    if failures:
        report = ["Retail data contract validation FAILED.", *[f"[FAIL] {x}" for x in failures]]
        write_report(report)
        return 1

    report = [
        "Retail data contract validation PASSED.",
        "Checked canonical field presence across dictionary and current source/output files.",
        "Checked Demo 1 source/output headers.",
        "Checked Demo 2 source/output headers.",
        "Checked Demo 2 diagnostic-scope and limitation fields.",
        "Checked generated Demo 1 and Demo 2 memory fact structure.",
        "Checked source_path existence and known source_fields.",
        "Checked critical metric-boundary phrases in DATA_DICTIONARY.md.",
        "Checked current schema aliases and dictionary-defined field boundaries.",
        f"Saved result path: {RESULT_PATH.relative_to(ROOT)}",
    ]
    write_report(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
