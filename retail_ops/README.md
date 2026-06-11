# Retail Operations Extension

This folder contains the retail-operations evidence layer for the Meituan instant-retail decision-support prototype.

The current retail implementation has three implemented evidence layers and one planned future stage:

| Stage | Current status | Main purpose |
|---|---|---|
| Demo 1 | Implemented | Store A month-over-month diagnostic across February, March, and April 2026. |
| Demo 2 | Implemented | Stores B-F same-period March 2026 diagnostic under one reporting window and one field contract. |
| Repeated-window panel | Implemented | Stores B-F coverage and descriptive summary across 2026-02, 2026-03, and 2026-04. |
| Pairwise comparability gate | Future work | Decide whether two store-period records can be compared for one selected operating question. |

The authoritative source for retail field names and metric meanings is:

- `retail_ops/data/DATA_DICTIONARY.md`

Use this README as the retail folder map. Detailed field definitions, evidence lineage, and future-gate rules are kept in the files listed below.

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
| 1 | `../PROJECT_SUMMARY_FOR_ADMISSIONS.md` | Why this prototype exists and how the current evidence path is staged. |
| 2 | `data/DATA_DICTIONARY.md` | Canonical field names, Meituan backend metric meanings, and field boundaries. |
| 3 | `demo/demo_1_store_a_month_over_month_diagnostic.md` | Store A month-over-month diagnostic evidence. |
| 4 | `demo/demo_2_cross_store_comparability_diagnostic.md` | Same-period B-F diagnostic evidence and current comparison limits. |
| 5 | `outputs/store_period_panel_coverage_output.csv` and `outputs/repeated_window_panel_summary_output.csv` | Repeated-window B-F evidence coverage and descriptive summary across 2026-02 to 2026-04. |
| 6 | `EXPERIMENTS.md` | What each current analytical check is designed to test. |
| 7 | `EXPERIMENT_RESULTS.md` | Current validation outcomes and evidence-boundary behavior. |
| 8 | `COMPARABILITY_GATE_V0.md` | Future pairwise comparability-gate contract. |

This file is the retail folder entry point. Detailed field boundaries belong in `data/DATA_DICTIONARY.md`; future-gate boundaries belong in `COMPARABILITY_GATE_V0.md`; experiment meaning and results belong in `EXPERIMENTS.md` and `EXPERIMENT_RESULTS.md`.

## Technical Appendices

These files are retained for auditability, but they are not separate reviewer entry points.

| Appendix | Owns |
|---|---|
| `TECHNICAL_APPENDIX.md` | Consolidated architecture, source-to-claim lineage, and field-usage review. |

## Current Demos

| Demo | Main file | Core evidence |
|---|---|---|
| Demo 1 | `demo/demo_1_store_a_month_over_month_diagnostic.md` | `data/store_a_monthly_metrics.csv`, `data/store_a_top_skus.csv`, `sql/01_store_a_month_over_month_diagnostic.sql`, `outputs/store_a_demo1_sql_output.csv`, `outputs/generated_retail_memory_facts.json` |
| Demo 2 | `demo/demo_2_cross_store_comparability_diagnostic.md` | `data/demo2_store_period_metrics.csv`, `data/demo2_top_search_terms.csv`, `data/demo2_top_skus_by_sales_volume.csv`, `data/demo2_top_skus_by_transaction_amount.csv`, `sql/02_demo2_cross_store_comparability.sql`, `outputs/demo2_cross_store_comparability_output.csv`, `outputs/generated_demo2_retail_memory_facts.json` |

## Repeated-Window Panel Extension

The repeated-window panel extension adds a small multi-month evidence layer after the current Demo 2 same-period diagnostic.

Its role is to make B-F store-period coverage visible before future question-specific pairwise comparability rules are added.

| Item | Current status |
|---|---|
| Current coverage | Stores B-F across 2026-02, 2026-03, and 2026-04 |
| Source table | `data/store_period_panel_metrics.csv` |
| Source notes | `data/store_period_panel_source_notes.md` |
| Coverage SQL | `sql/03_store_period_panel_coverage.sql` |
| Coverage output | `outputs/store_period_panel_coverage_output.csv` |
| Coverage validator | `scripts/validate_store_period_panel.py` |
| Descriptive summary SQL | `sql/04_repeated_window_panel_summary.sql` |
| Descriptive summary output | `outputs/repeated_window_panel_summary_output.csv` |
| Descriptive summary validator | `scripts/validate_repeated_window_panel_summary.py` |
| Current use | Coverage and descriptive summary for repeated store-period evidence before future pairwise comparability decisions. |
