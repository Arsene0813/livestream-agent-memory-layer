from __future__ import annotations

import csv
import json
import re
import subprocess
from datetime import date
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
    "store_exposure_users",
    "store_exposure_times",
    "entry_visits",
    "order_submissions",
    "full_or_partial_refund_orders",
    "business_area_rank",
]

# Selected required implemented fields exercised by the current
# Demo 1 and Demo 2 contracts. This is not the complete field
# registry documented in DATA_DICTIONARY.md.
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

REQUIRED_COMPARABILITY_GATE_PHRASES = [
    "`estimated_income_proxy` as weak supplementary backend context only",
    "`estimated_income_proxy` lacks a full calculation breakdown in the current demo data and should not be used as a primary gate factor.",
    "`estimated_income_proxy` as supplementary display context only",
    "Check whether transaction order volume and transaction amount are within a reasonable comparison band.",
]

CANONICAL_MEMORY_SLOTS = {
    "visibility_entry_profile",
    "activity_lever_profile",
    "transaction_conversion_profile",
    "single_metric_attribution_guard",
    "top3_sku_product_mix_note",
}

DEMO1_SUMMARY_FIELD_ORDER = (
    "store_id",
    "period_granularity",
    "period_start",
    "period_end",
    "period_label",
    "slot",
    "summary",
)

DEMO1_SUMMARY_SLOT_ORDER = (
    "visibility_entry_profile",
    "activity_lever_profile",
    "transaction_conversion_profile",
    "single_metric_attribution_guard",
    "top3_sku_product_mix_note",
)

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
    "retail_ops/COMPARABILITY_GATE_V0.md",
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


def declared_source_csv_fields(
    *,
    relative_path: str,
    index: int,
    fact: dict[str, object],
    failures: list[str],
) -> set[str]:
    prefix = f"{relative_path} fact #{index}"
    source_path = fact.get("source_path")
    supporting_paths = fact.get("supporting_source_paths", [])

    paths: list[str] = []
    if isinstance(source_path, str) and source_path.strip():
        paths.append(source_path)

    if supporting_paths is None:
        supporting_paths = []
    if not isinstance(supporting_paths, list):
        failures.append(
            f"{prefix} has non-list supporting_source_paths"
        )
        supporting_paths = []

    for supporting_path in supporting_paths:
        if not isinstance(supporting_path, str) or not supporting_path.strip():
            failures.append(
                f"{prefix} has invalid supporting source path "
                f"`{supporting_path}`"
            )
            continue
        paths.append(supporting_path)

    fields: set[str] = set()
    for declared_path in dict.fromkeys(paths):
        if not source_exists(declared_path):
            failures.append(
                f"{prefix} declared source path does not exist: "
                f"{declared_path}"
            )
            continue

        if Path(declared_path).suffix.lower() != ".csv":
            failures.append(
                f"{prefix} declared source path is not a CSV field "
                f"source: {declared_path}"
            )
            continue

        fields.update(read_csv_headers(declared_path))

    return fields


ALLOWED_PERIOD_GRANULARITIES = {"month", "month_range"}


def validate_period_metadata(
    *,
    relative_path: str,
    index: int,
    fact: dict[str, object],
    failures: list[str],
) -> None:
    period_label = fact.get("period_label")
    period_start = fact.get("period_start")
    period_end = fact.get("period_end")
    granularity = fact.get("period_granularity")
    prefix = f"{relative_path} fact #{index}"

    if granularity not in ALLOWED_PERIOD_GRANULARITIES:
        failures.append(
            f"{prefix} has unsupported period_granularity `{granularity}`"
        )
        return

    if not isinstance(period_label, str) or not period_label.strip():
        failures.append(f"{prefix} has missing period_label")
        return

    try:
        start_date = date.fromisoformat(str(period_start))
        end_date = date.fromisoformat(str(period_end))
    except ValueError:
        failures.append(
            f"{prefix} has invalid ISO period dates: "
            f"period_start={period_start!r}, period_end={period_end!r}"
        )
        return

    if end_date < start_date:
        failures.append(
            f"{prefix} has period_end before period_start"
        )
        return

    start_month = start_date.strftime("%Y-%m")
    end_month = end_date.strftime("%Y-%m")

    if granularity == "month":
        if start_month != end_month:
            failures.append(
                f"{prefix} has period_granularity `month` but spans "
                "multiple calendar months"
            )
        if period_label != start_month:
            failures.append(
                f"{prefix} month period_label `{period_label}` does not "
                f"match `{start_month}`"
            )
        return

    expected_label = f"{start_month}_to_{end_month}"
    if start_month == end_month:
        failures.append(
            f"{prefix} has period_granularity `month_range` but does "
            "not span multiple calendar months"
        )
    if period_label != expected_label:
        failures.append(
            f"{prefix} month_range period_label `{period_label}` does "
            f"not match `{expected_label}`"
        )


def validate_generated_facts(
    *,
    relative_path: str,
    allowed_entities: set[str],
    documented_fields: set[str],
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

        validate_period_metadata(
            relative_path=relative_path,
            index=index,
            fact=fact,
            failures=failures,
        )

        value = fact.get("value")
        if not isinstance(value, str) or not value.strip():
            failures.append(
                f"{relative_path} fact #{index} has missing or empty value"
            )

        calculation = fact.get("calculation")
        if not isinstance(calculation, str) or not calculation.strip():
            failures.append(
                f"{relative_path} fact #{index} has missing or empty calculation"
            )

        source_path = fact.get("source_path")
        if not isinstance(source_path, str) or not source_path.strip():
            failures.append(
                f"{relative_path} fact #{index} has missing source_path"
            )

        declared_fields = declared_source_csv_fields(
            relative_path=relative_path,
            index=index,
            fact=fact,
            failures=failures,
        )

        source_fields = fact.get("source_fields")
        if not isinstance(source_fields, list):
            failures.append(
                f"{relative_path} fact #{index} has non-list source_fields"
            )
            continue

        for field in source_fields:
            if not isinstance(field, str) or not field.strip():
                failures.append(
                    f"{relative_path} fact #{index} has invalid "
                    f"source field `{field}`"
                )
                continue

            if (
                field not in documented_fields
                and field not in KNOWN_HELPER_FIELDS
            ):
                failures.append(
                    f"{relative_path} fact #{index} source field "
                    f"`{field}` is not documented in DATA_DICTIONARY.md "
                    "or approved helper registry"
                )

            if declared_fields and field not in declared_fields:
                failures.append(
                    f"{relative_path} fact #{index} source field "
                    f"`{field}` is not present in declared "
                    "source/supporting CSV headers"
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
    comparability_gate = read_text("retail_ops/COMPARABILITY_GATE_V0.md")
    demo1_sql = read_text("retail_ops/sql/01_store_a_month_over_month_diagnostic.sql")
    demo2_sql = read_text("retail_ops/sql/02_demo2_cross_store_comparability.sql")

    demo1_source_headers = read_csv_headers("retail_ops/data/store_a_monthly_metrics.csv")
    demo1_top_sku_headers = read_csv_headers("retail_ops/data/store_a_top_skus.csv")
    demo1_output_headers = read_csv_headers("retail_ops/outputs/store_a_demo1_sql_output.csv")
    demo1_summary_headers = read_csv_headers(
        "retail_ops/outputs/store_a_demo1_interpretation_summary.csv"
    )

    if demo1_summary_headers != set(DEMO1_SUMMARY_FIELD_ORDER):
        failures.append(
            "Demo 1 interpretation summary headers must be exactly "
            f"{list(DEMO1_SUMMARY_FIELD_ORDER)}"
        )

    with (
        ROOT
        / "retail_ops"
        / "outputs"
        / "store_a_demo1_interpretation_summary.csv"
    ).open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as summary_file:
        summary_reader = csv.DictReader(summary_file)
        summary_field_order = tuple(
            summary_reader.fieldnames or ()
        )
        summary_rows = list(summary_reader)

    summary_slot_order = tuple(
        row.get("slot", "")
        for row in summary_rows
    )

    if summary_field_order != DEMO1_SUMMARY_FIELD_ORDER:
        failures.append(
            "Demo 1 interpretation summary field order does not "
            "match the canonical reviewer-summary contract"
        )

    if summary_slot_order != DEMO1_SUMMARY_SLOT_ORDER:
        failures.append(
            "Demo 1 interpretation summary slot order does not "
            "match the canonical Demo 1 slot order"
        )

    if any(
        not row.get("summary", "").strip()
        for row in summary_rows
    ):
        failures.append(
            "Demo 1 interpretation summary contains an empty "
            "summary value"
        )
    demo2_source_headers = read_csv_headers("retail_ops/data/demo2_store_period_metrics.csv")
    demo2_output_headers = read_csv_headers(
        "retail_ops/outputs/demo2_cross_store_comparability_output.csv"
    )

    documented_fields = extract_backticked_fields(dictionary)
    current_source_output_fields = (
        demo1_source_headers
        | demo1_top_sku_headers
        | demo1_output_headers
        | demo1_summary_headers
        | demo2_source_headers
        | demo2_output_headers
    )

    validator_paths = {
        (ROOT / "retail_ops/scripts/validate_retail_data_contract.py").resolve(),
        (ROOT / "retail_ops/scripts/validate_demo2_staging_data.py").resolve(),
        (ROOT / "retail_ops/scripts/validate_store_period_panel.py").resolve(),
        (ROOT / "scripts/validate_demo2_retail_endpoint_boundary.py").resolve(),
        (ROOT / "retail_ops/data/DATA_DICTIONARY.md").resolve(),
        RESULT_PATH.resolve(),
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
        if field not in current_source_output_fields:
            failures.append(f"Required canonical field `{field}` missing from current source/output fields")

    for phrase in REQUIRED_BOUNDARY_PHRASES:
        if phrase not in dictionary:
            failures.append(f"DATA_DICTIONARY.md missing required boundary phrase: {phrase}")

    for phrase in REQUIRED_COMPARABILITY_GATE_PHRASES:
        if phrase not in comparability_gate:
            failures.append(
                "COMPARABILITY_GATE_V0.md missing required "
                f"estimated-income boundary phrase: {phrase}"
            )

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
        documented_fields=documented_fields,
        failures=failures,
    )

    validate_generated_facts(
        relative_path="retail_ops/outputs/generated_demo2_retail_memory_facts.json",
        allowed_entities={"store_B", "store_C", "store_D", "store_E", "store_F"},
        documented_fields=documented_fields,
        failures=failures,
    )

    if failures:
        report = ["Retail data contract validation FAILED.", *[f"[FAIL] {x}" for x in failures]]
        write_report(report)
        return 1

    report = [
        "Retail data contract validation PASSED.",
        "Checked selected required implemented field presence across the dictionary and current source/output files.",
        "Checked Demo 1 source/output headers and canonical interpretation-summary slots.",
        "Checked Demo 2 source/output headers.",
        "Checked Demo 2 diagnostic-scope and limitation fields.",
        "Checked generated Demo 1 and Demo 2 memory fact structure, non-empty values and calculations, and period metadata.",
        "Checked dictionary-bounded source_fields against declared source and supporting CSV headers.",
        "Checked critical metric-boundary phrases in DATA_DICTIONARY.md.",
        "Checked estimated_income_proxy remains supplementary context rather than a primary comparability-gate factor.",
        "Checked registered non-canonical aliases while preserving DATA_DICTIONARY.md as the naming authority.",
        f"Saved result path: {RESULT_PATH.relative_to(ROOT)}",
    ]
    write_report(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
