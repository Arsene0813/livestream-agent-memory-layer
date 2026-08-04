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

REQUIRED_DEMO2_SENSITIVITY_OUTPUT_FIELDS = {
    "scenario",
    "store_id",
    "activity_order_share_pct",
    "top3_sku_transaction_amount_share_pct",
    "sensitivity_limit_notes",
    "current_comparison_scope_flag",
    "current_comparison_limit_notes",
}

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
    "Check whether transaction order volume and transaction amount are within a reasonable comparison band.",
]

CANONICAL_RETAIL_FACT_KIND = "retail_memory_fact"
CANONICAL_RETAIL_FACT_TYPE = "retail_metric_profile"

CANONICAL_MEMORY_SLOTS = {
    "visibility_entry_profile",
    "activity_lever_profile",
    "transaction_conversion_profile",
    "single_metric_attribution_guard",
    "top3_sku_product_mix_note",
}

ALLOWED_FACT_CONFIDENCE = {
    "high",
    "medium",
}

EXPECTED_CONFIDENCE_BY_SLOT = {
    "visibility_entry_profile": "high",
    "activity_lever_profile": "high",
    "transaction_conversion_profile": "high",
    "single_metric_attribution_guard": "high",
    "top3_sku_product_mix_note": "medium",
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
    "retail_ops/outputs/demo2_guardrail_sensitivity_summary.csv",
    "retail_ops/outputs/generated_retail_memory_facts.json",
    "retail_ops/outputs/generated_demo2_retail_memory_facts.json",
]


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def read_csv_headers(relative_path: str) -> set[str]:
    with (ROOT / relative_path).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        return set(next(reader))


def extract_dictionary_fields(text: str) -> set[str]:
    """Read fields from explicit dictionary registration locations."""

    token_pattern = re.compile(
        r"`([a-zA-Z_][a-zA-Z0-9_]*)`"
    )
    mom_field_pattern = re.compile(
        r"- `([a-zA-Z_][a-zA-Z0-9_]*)`"
    )

    fields: set[str] = set()
    in_mom_diagnostics = False

    for line in text.splitlines():
        stripped = line.strip()

        if stripped == "#### Month-over-month diagnostics":
            in_mom_diagnostics = True
            continue

        heading_match = re.match(
            r"^#{3,6}\s+(.+)$",
            stripped,
        )

        if heading_match is not None:
            in_mom_diagnostics = False
            fields.update(
                token_pattern.findall(
                    heading_match.group(1)
                )
            )
            continue

        if in_mom_diagnostics:
            mom_match = mom_field_pattern.fullmatch(stripped)

            if mom_match is not None:
                fields.add(mom_match.group(1))

    explicit_declarations = {
        "store_id": (
            "`store_id` is the canonical store identifier used in "
            "source CSV files, SQL diagnostics, and metric outputs."
        ),
        "store_average_rank_change": (
            "`store_average_rank_change` and "
            "`search_average_rank_change` compare the current month "
            "with the previous available month for the same "
            "`store_id`."
        ),
        "search_average_rank_change": (
            "`store_average_rank_change` and "
            "`search_average_rank_change` compare the current month "
            "with the previous available month for the same "
            "`store_id`."
        ),
        "transaction_recovered_with_conversion_aov_tradeoff": (
            "`transaction_recovered_with_conversion_aov_tradeoff` "
            "is a SQL-derived supporting observation."
        ),
    }

    for field, declaration in explicit_declarations.items():
        if text.count(declaration) != 1:
            raise ValueError(
                "Expected one explicit dictionary declaration for "
                f"`{field}`"
            )

        fields.add(field)

    return fields


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

        kind = fact.get("kind")
        if kind != CANONICAL_RETAIL_FACT_KIND:
            failures.append(
                f"{relative_path} fact #{index} has unsupported "
                f"kind `{kind}`; expected "
                f"`{CANONICAL_RETAIL_FACT_KIND}`"
            )

        fact_type = fact.get("type")
        if fact_type != CANONICAL_RETAIL_FACT_TYPE:
            failures.append(
                f"{relative_path} fact #{index} has unsupported "
                f"type `{fact_type}`; expected "
                f"`{CANONICAL_RETAIL_FACT_TYPE}`"
            )

        entity_id = fact.get("entity_id")
        if entity_id not in allowed_entities:
            failures.append(
                f"{relative_path} fact #{index} has unsupported entity_id `{entity_id}`"
            )

        slot = fact.get("slot")
        if slot not in CANONICAL_MEMORY_SLOTS:
            failures.append(f"{relative_path} fact #{index} has non-canonical slot `{slot}`")

        is_active = fact.get("is_active")
        if not isinstance(is_active, bool):
            failures.append(
                f"{relative_path} fact #{index} has non-boolean is_active"
            )

        confidence = fact.get("confidence")
        if confidence not in ALLOWED_FACT_CONFIDENCE:
            failures.append(
                f"{relative_path} fact #{index} has unsupported "
                f"confidence `{confidence}`"
            )
        elif (
            is_active is True
            and slot in EXPECTED_CONFIDENCE_BY_SLOT
        ):
            expected_confidence = EXPECTED_CONFIDENCE_BY_SLOT[slot]

            if confidence != expected_confidence:
                failures.append(
                    f"{relative_path} fact #{index} slot `{slot}` "
                    f"requires confidence `{expected_confidence}`, "
                    f"found `{confidence}`"
                )

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

        observed_values = fact.get("observed_values")
        if (
            not isinstance(observed_values, dict)
            or not observed_values
        ):
            failures.append(
                f"{relative_path} fact #{index} has missing "
                "or empty observed_values"
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
        if (
            not isinstance(source_fields, list)
            or not source_fields
        ):
            failures.append(
                f"{relative_path} fact #{index} has missing "
                "or non-list source_fields"
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

        lineage_path = fact.get("lineage_path")
        if not isinstance(lineage_path, str) or not lineage_path.strip():
            failures.append(
                f"{relative_path} fact #{index} has missing lineage_path"
            )
        elif not source_exists(lineage_path):
            failures.append(
                f"{relative_path} fact #{index} lineage_path "
                f"does not exist: `{lineage_path}`"
            )

        limitations = fact.get("limitations")
        if not isinstance(limitations, list) or not limitations:
            failures.append(
                f"{relative_path} fact #{index} has missing limitations"
            )
        elif any(
            not isinstance(item, str) or not item.strip()
            for item in limitations
        ):
            failures.append(
                f"{relative_path} fact #{index} contains an "
                "empty or non-string limitation"
            )


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

    demo2_sensitivity_headers = read_csv_headers(
        "retail_ops/outputs/demo2_guardrail_sensitivity_summary.csv"
    )
    try:
        documented_fields = extract_dictionary_fields(
            dictionary
        )
    except ValueError as exc:
        failures.append(
            "Dictionary field-registry extraction failed: "
            f"{exc}"
        )
        documented_fields = set()
    current_source_output_fields = (
        demo1_source_headers
        | demo1_top_sku_headers
        | demo1_output_headers
        | demo1_summary_headers
        | demo2_source_headers
        | demo2_output_headers
        | demo2_sensitivity_headers
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
        if field not in documented_fields:
            failures.append(
                f"Required canonical field `{field}` missing from "
                "explicit DATA_DICTIONARY.md field registrations"
            )
        if field not in current_source_output_fields:
            failures.append(
                f"Required canonical field `{field}` missing from "
                "current source/output fields"
            )

    for phrase in REQUIRED_BOUNDARY_PHRASES:
        if phrase not in dictionary:
            failures.append(f"DATA_DICTIONARY.md missing required boundary phrase: {phrase}")

    for phrase in REQUIRED_COMPARABILITY_GATE_PHRASES:
        if phrase not in comparability_gate:
            failures.append(
                "COMPARABILITY_GATE_V0.md missing required "
                f"comparability-gate phrase: {phrase}"
            )

    for field in REQUIRED_DEMO2_OUTPUT_FIELDS:
        if field not in demo2_output_headers:
            failures.append(f"Demo 2 output missing required field `{field}`")
        if field not in demo2_sql:
            failures.append(f"Demo 2 SQL missing required field `{field}`")

    if (
        demo2_sensitivity_headers
        != REQUIRED_DEMO2_SENSITIVITY_OUTPUT_FIELDS
    ):
        failures.append(
            "Demo 2 guardrail sensitivity summary headers must be "
            "exactly "
            f"{sorted(REQUIRED_DEMO2_SENSITIVITY_OUTPUT_FIELDS)}"
        )

    for field in sorted(
        REQUIRED_DEMO2_SENSITIVITY_OUTPUT_FIELDS
    ):
        if field not in documented_fields:
            failures.append(
                "Demo 2 guardrail sensitivity field "
                f"`{field}` is not explicitly registered in "
                "DATA_DICTIONARY.md"
            )

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
        "Checked selected required implemented fields against explicit dictionary registrations and current source/output files.",
        "Checked Demo 1 source/output headers and canonical interpretation-summary slots.",
        "Checked Demo 2 source/output headers.",
        "Checked Demo 2 guardrail-sensitivity output headers and dictionary registrations.",
        "Checked Demo 2 diagnostic-scope and limitation fields.",
        "Checked generated Demo 1 and Demo 2 memory fact kind/type discriminators, structure, evidence-trace fields, period metadata, and slot-bounded confidence.",
        "Checked generated source_fields against explicit dictionary field registrations and declared source/supporting CSV headers.",
        "Checked critical metric-boundary phrases in DATA_DICTIONARY.md.",
        "Checked required comparability-gate contract phrases.",
        "Checked registered non-canonical aliases while preserving DATA_DICTIONARY.md as the naming authority.",
        f"Saved result path: {RESULT_PATH.relative_to(ROOT)}",
    ]
    write_report(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
