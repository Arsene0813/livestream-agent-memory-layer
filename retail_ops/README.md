# Retail Operations Extension

This folder contains the retail-operations evidence layer for the Meituan instant-retail decision-support prototype.

The current retail implementation has two completed diagnostic stages and one planned future stage:

| Stage | Current status | Main purpose |
|---|---|---|
| Demo 1 | Implemented | Store A month-over-month diagnostic across February, March, and April 2026. |
| Demo 2 | Implemented | Stores B-F same-period March 2026 diagnostic under one field contract. |
| Pairwise comparability gate | Future work | Decide whether two store-period records can be compared for one selected operating question. |

The authoritative source for retail field names and metric meanings is:

- `retail_ops/data/DATA_DICTIONARY.md`

This README is intentionally a folder map.
Detailed evidence boundaries are kept in the files listed below.

## Folder Scope

| Component | Purpose |
|---|---|
| `data/` | Selected Meituan-style source tables, source notes, and metric definitions. |
| `sql/` | Diagnostic SQL for Demo 1 and Demo 2. |
| `outputs/` | Generated SQL outputs, validation results, retrieval-inspection outputs, and memory facts. |
| `scripts/` | Local validation, generation, sensitivity, and loading scripts. |
| `demo/` | Diagnostic review documents for implemented retail stages. |

## Retail Review Path

Use this order when reviewing the retail extension:

| Step | File | What to check |
|---:|---|---|
| 1 | `../PROJECT_SUMMARY_FOR_ADMISSIONS.md` | Why this prototype exists and where the current implementation stops. |
| 2 | `data/DATA_DICTIONARY.md` | Canonical field names, Meituan backend metric meanings, and field boundaries. |
| 3 | `demo/demo_1_store_a_month_over_month_diagnostic.md` | Store A month-over-month diagnostic evidence. |
| 4 | `demo/demo_2_cross_store_comparability_diagnostic.md` | Same-period B-F diagnostic evidence and current comparison limits. |
| 5 | `EXPERIMENTS.md` | What each current analytical check is designed to test. |
| 6 | `EXPERIMENT_RESULTS.md` | Current validation outcomes and what the checks do not prove. |
| 7 | `COMPARABILITY_GATE_V0.md` | Future pairwise comparability-gate contract. |

This file is only the retail folder entry point. Detailed field boundaries belong in `data/DATA_DICTIONARY.md`; future gate boundaries belong in `COMPARABILITY_GATE_V0.md`; experiment meaning and results belong in `EXPERIMENTS.md` and `EXPERIMENT_RESULTS.md`.

## Technical Appendices

These files are retained for auditability, but they are not separate reviewer entry points.

| Appendix | Owns |
|---|---|
| `ARCHITECTURE.md` | Technical structure of the retail evidence path. |
| `LINEAGE.md` | Source-to-SQL-to-memory lineage and claim traceability. |
| `FIELD_USAGE_REVIEW.md` | Field-name and semantic-change review before future expansion. |

## Current Demos

| Demo | Main file | Core evidence |
|---|---|---|
| Demo 1 | `demo/demo_1_store_a_month_over_month_diagnostic.md` | `data/store_a_monthly_metrics.csv`, `data/store_a_top_skus.csv`, `sql/01_store_a_month_over_month_diagnostic.sql`, `outputs/store_a_demo1_sql_output.csv`, `outputs/generated_retail_memory_facts.json` |
| Demo 2 | `demo/demo_2_cross_store_comparability_diagnostic.md` | `data/demo2_store_period_metrics.csv`, `data/demo2_top_search_terms.csv`, `data/demo2_top_skus_by_sales_volume.csv`, `data/demo2_top_skus_by_transaction_amount.csv`, `sql/02_demo2_cross_store_comparability.sql`, `outputs/demo2_cross_store_comparability_output.csv`, `outputs/generated_demo2_retail_memory_facts.json` |

## Repeated-Window Panel Extension

The repeated-window panel extension adds a small multi-month coverage layer after the current Demo 2 same-period diagnostic.

| Item | Current status |
|---|---|
| Current coverage | Stores B-F across 2026-02, 2026-03, and 2026-04 |
| Source table | `data/store_period_panel_metrics.csv` |
| Source notes | `data/store_period_panel_source_notes.md` |
| Coverage SQL | `sql/03_store_period_panel_coverage.sql` |
| Coverage output | `outputs/store_period_panel_coverage_output.csv` |
| Validator | `scripts/validate_store_period_panel.py` |
| Boundary | Coverage foundation only; not a new numbered demo, pairwise comparability gate, endpoint behavior, generated memory facts, store ranking, or causal analysis. |

The extension intentionally excludes `valid_orders`, `invalid_orders`, and `invalid_order_pressure_pct` because the current backend evidence does not define those order-status fields clearly enough for diagnostic use.

## Script Notes

| Script | Meaning |
|---|---|
| `scripts/validate_demo2_staging_data.py` | Validates Demo 2 source-table structure. |
| `scripts/validate_demo2_comparability_output.py` | Validates the Demo 2 diagnostic output contract, not a pairwise gate. |
| `scripts/generate_demo2_retail_memory_facts.py` | Converts Demo 2 diagnostic output into generated retail memory facts. |
| `scripts/validate_demo2_retail_memory_facts.py` | Validates generated Demo 2 retail memory fact structure. |
| `scripts/analyze_demo2_guardrail_sensitivity.py` | Inspects whether current Demo 2 guardrail notes are sensitive to small threshold changes. |
| `scripts/validate_retail_data_contract.py` | Checks retail field-contract consistency across dictionary, source tables, outputs, and facts. |

## Editing Guardrail

Do not rename retail fields in README files. Any field-name or semantic change must first go through:

- `retail_ops/data/DATA_DICTIONARY.md`

This README is intentionally a folder map.
Detailed evidence boundaries are kept in the files listed below.
- `retail_ops/FIELD_USAGE_REVIEW.md`
- `retail_ops/LINEAGE.md`

Current Demo 2 paths keep `cross_store_comparability` for reference stability, but the implemented meaning remains same-period diagnostic evidence, not a completed pairwise comparability gate.
