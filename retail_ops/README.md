# Retail Operations Extension

This folder contains the implemented retail evidence layer: source tables,
the canonical field dictionary, SQL diagnostics, generated facts,
outputs, validation scripts, experiments, and technical references.

For the application first-pass order, use
[`PROJECT_SUMMARY_FOR_ADMISSIONS.md`](../PROJECT_SUMMARY_FOR_ADMISSIONS.md).
Within this folder, this README is the working map for locating retail
evidence and implementation files.

## Folder Workflow

The retail folder follows one practical workflow:

1. preserve backend metric definitions;
2. structure selected store-period data with SQL;
3. generate source-aware retail memory facts;
4. evaluate whether answers preserve entity, period, metric, and scope boundaries;
5. prepare repeated-window evidence for a future question-specific comparability gate.

## Folder Map

| Path | Role |
|---|---|
| `data/DATA_DICTIONARY.md` | Canonical field names, backend metric definitions, derived-field formulas, and field-boundary rules. |
| `data/` | Current selected source data and source notes. |
| `sql/` | SQL transformations for Demo 1, Demo 2, store-period panel coverage, and repeated-window summary. |
| `outputs/` | Generated SQL outputs, generated memory facts, retrieval inspection outputs, and validation result files. |
| `demo/` | Human-readable Demo 1 and Demo 2 diagnostic reports. |
| `scripts/` | Export, generation, validation, and inspection scripts for the retail evidence path. |
| `EXPERIMENT_RESULTS.md` | Experiment map, validation outcomes, pass conditions, and failure modes. |
| `COMPARABILITY_GATE_V0.md` | Future pairwise comparability-gate contract. |
| `TECHNICAL_APPENDIX.md` | Technical lineage, field-usage notes, and architecture-level retail evidence details. |

## Current Retail Evidence

| Layer | Current evidence |
|---|---|
| Demo 1 | Store A month-over-month diagnostic across 2026-02, 2026-03, and 2026-04. |
| Demo 2 | Same-period B-F diagnostic under the March 2026 reporting window. |
| Repeated-window panel | B-F coverage and descriptive summary across 2026-02, 2026-03, and 2026-04. |
| Generated memory facts | Source-aware retail facts with observed values, source fields, confidence labels, and limitations. |
| Evaluation | Data-contract, Demo 2 output, generated-fact, answer-boundary, retrieval-inspection, and future-gate contract checks. |

## Admissions Review Path

The application-level entry point is the
[project summary](../PROJECT_SUMMARY_FOR_ADMISSIONS.md). Within the retail
evidence layer, use this order:

| Step | File | Purpose |
|---:|---|---|
| 1 | [Demo 1: Store A month-over-month diagnostic](demo/demo_1_store_a_month_over_month_diagnostic.md) | Review repeated-window single-store evidence and multi-metric interpretation. |
| 2 | [Demo 2: same-period B-F diagnostic](demo/demo_2_cross_store_comparability_diagnostic.md) | Review multi-store evidence organization and interpretation limits. |
| 3 | [Store-period panel coverage](outputs/store_period_panel_coverage_output.csv) and [repeated-window summary](outputs/repeated_window_panel_summary_output.csv) | Inspect February-April B-F coverage and descriptive movement. |
| 4 | [Experiment results](EXPERIMENT_RESULTS.md) | Review validation procedures, outcomes, sensitivity checks, and failure modes. |
| 5 | [RAC demo index](../rac/DEMO_INDEX.md) | Continue from structured retail evidence into factor-aware grounded review. |

Technical references:

- [Data dictionary](data/DATA_DICTIONARY.md): canonical field names,
  backend metric definitions, formulas, grains, and interpretation limits.
- [Technical appendix](TECHNICAL_APPENDIX.md): source-to-claim lineage and
  architecture-level evidence details.
- [Comparability Gate V0](COMPARABILITY_GATE_V0.md): future
  question-specific pairwise comparability requirements.

## Main Outputs

| Output | Purpose |
|---|---|
| `outputs/store_a_demo1_sql_output.csv` | Demo 1 SQL diagnostic output. |
| `outputs/demo2_cross_store_comparability_output.csv` | Demo 2 same-period diagnostic output. |
| `outputs/generated_retail_memory_facts.json` | Demo 1 generated memory facts. |
| `outputs/generated_demo2_retail_memory_facts.json` | Demo 2 generated memory facts. |
| `outputs/store_period_panel_coverage_output.csv` | Repeated-window B-F coverage output. |
| `outputs/repeated_window_panel_summary_output.csv` | February, March, and April values with February-to-April endpoint summaries. |
| `outputs/retrieval_score_distribution.csv` | Retrieval score inspection output. |
| `outputs/retrieval_query_robustness.csv` | Query robustness inspection output. |

## SQL Execution

Run registered queries from the repository root with the project virtual environment:

```bash
.venv/bin/python3 -m retail_ops.sql_runtime --query 02_demo2_cross_store_comparability.sql --summary
```

The runner checks canonical fields, reporting windows and duplicate keys before calculation.
Missing values stay null. Numeric and aggregation rules are documented in [SQL input checks](sql/README.md).

## Local Checks

| Check | Command |
|---|---|
| Markdown readability | `python3 ../scripts/validate_markdown_readability.py` |
| CSV physical rows | `python3 scripts/validate_csv_physical_rows.py` |
| Retail data contract | `python3 scripts/validate_retail_data_contract.py` |
| Demo 2 output | `python3 scripts/validate_demo2_comparability_output.py` |
| Guardrail sensitivity | `python3 scripts/analyze_demo2_guardrail_sensitivity.py` |
| Store-period panel | `python3 scripts/validate_store_period_panel.py` |
| Repeated-window regeneration | `python3 scripts/regenerate_repeated_window_panel_summary.py` |
| Repeated-window summary | `python3 scripts/validate_repeated_window_panel_summary.py` |
