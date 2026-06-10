# Retail Operations Architecture

This file is the technical architecture appendix for the retail-operations evidence path.

It should explain how evidence moves through the implemented prototype. It should not repeat the full admissions narrative, the full field dictionary, or the future pairwise comparability-gate design.

## Document Ownership

| This file owns | Canonical file for related detail |
|---|---|
| Current retail evidence path | `retail_ops/LINEAGE.md` for claim-to-data lineage |
| Endpoint evidence modes | `api/main.py` for endpoint implementation |
| Layer responsibilities | `retail_ops/data/DATA_DICTIONARY.md` for field meanings |
| Architecture-level boundaries | `retail_ops/COMPARABILITY_GATE_V0.md` for future gate design |

## Current Retail Evidence Layers

| Layer | Status | Current role | Reviewer reading |
|---|---|---|---|
| Demo 1: Store A month-over-month diagnostic | Implemented | Structures one store's February-April 2026 backend metrics into SQL diagnostic output and memory facts. | A narrow single-store evidence path. |
| Demo 2: Stores B-F same-period diagnostic | Implemented | Structures selected March 2026 store-period records under one field contract. | A same-period diagnostic layer before stronger cross-store comparability rules. |
| Post-Demo2 repeated-window panel | Implemented | Adds February-April 2026 repeated-window coverage for Stores B-F. | Evidence for checking whether same-store signals persist across reporting windows. |
| Pairwise comparability gate | Documented future work | Plans a question-specific decision contract for judging whether selected store-period records can be compared. | Future decision-support design built from broader evidence coverage. |

## Current Evidence Path

The current file-based data path is:

~~~text
selected Meituan backend metrics
-> canonical CSV tables
-> DATA_DICTIONARY.md field contract
-> SQL diagnostic output
-> generated retail memory facts
-> validation and scenario-based boundary checks
~~~

~~~mermaid
graph TD
  A[Meituan backend evidence] --> B[Canonical CSV source tables]
  B --> C[DATA_DICTIONARY.md field contract]
  C --> D[SQL diagnostics]
  D --> E[Generated retail memory facts]
  E --> F[Retail KB endpoint and answer-boundary checks]
  F --> G{Does the evidence support the question?}
  G -->|Yes| H[Qualified answer with source and scope limits]
  G -->|No| I[Refusal or limitation note]
~~~

The design is intentionally file-based at this stage. The priority is to keep each diagnostic claim traceable before expanding toward broader 48-store workflow support.

## Retrieval Mode Boundary

The repository contains more than one retrieval path. They should not be read as the same level of implementation maturity.

| Path | Current mode | Current role | Boundary |
|---|---|---|---|
| `/chat_livestream_kb` | Qdrant-backed retrieval over livestream/product memory facts. | Tests lifecycle-aware memory behavior, including typed facts, active-state filtering, and fallback/refusal behavior. | Original memory-layer path, not the Meituan multi-store diagnostic system. |
| `/chat_retail_ops_kb` | Retail memory retrieval path for implemented Store A facts. | Tests whether retail memory facts can be retrieved with source fields and limitations. | Limited to the implemented retail facts and current retail evidence path. |
| `/chat_retail_ops_demo2_kb` | File-backed generated Demo 2 retail memory facts. | Tests whether B-F same-period diagnostic facts can be returned or refused inside the current evidence boundary. | Not retrieval-score evaluation, production Meituan API integration, or a pairwise comparability gate. |
| Future pairwise comparability gate | Not implemented. | Planned gate for judging whether two store-period records can be compared for a specific operating question. | Should be implemented only after broader store-period coverage, repeated windows, and stronger market-context evidence exist. |

The retrieval-threshold calibration is a separate offline inspection over the file-backed evidence corpus. It should not be read as the runtime selection logic of `/chat_retail_ops_demo2_kb`.

## Endpoint Evidence Modes

The current retail endpoints do not use one identical evidence path.

| Endpoint | Evidence mode | Boundary |
|---|---|---|
| `/chat_retail_ops_kb` | Qdrant-backed retrieval over the retail memory corpus, with embedding scores used for retrieval behavior analysis. | Retrieval score is not treated as standalone operating evidence. |
| `/chat_retail_ops_demo2_kb` | File-backed Demo 2 generated memory facts with local question routing and boundary refusal behavior. | Demo 2 remains a same-period B-F diagnostic endpoint, not a completed pairwise comparability gate. |

Both paths should preserve metric definitions, entity scope, period scope, source limits, and comparison boundaries before returning an answer.

## Responsibility Split

| Layer | Responsibility | Boundary |
|---|---|---|
| Data dictionary | Preserve backend metric meanings and canonical field names. | Existing Meituan backend metrics stay tied to documented definitions. |
| SQL diagnostics | Compute store-period diagnostic evidence under the documented field contract. | SQL output remains diagnostic evidence, not a final operating decision. |
| Generated memory facts | Store observed values, source fields, calculation notes, confidence labels, and limitations. | Memory facts summarize evidence without replacing raw backend definitions. |
| Boundary checks | Check entity scope, period scope, metric meanings, and comparison limits before answers are accepted. | Evaluation focuses on evidence discipline and answer scope. |
| Future comparability gate | Judge whether two store-period records can be compared for one selected operating question. | The gate is question-specific and depends on broader store-period evidence. |

## Layer Contract

| Layer | Input | Output | Boundary |
|---|---|---|---|
| Backend evidence | Selected Meituan merchant-backend metrics and manually structured evidence tables. | Canonical CSV source files. | Not full automated ingestion. |
| Metric contract | Canonical CSV fields and backend definitions. | `retail_ops/data/DATA_DICTIONARY.md` and `retail_ops/LINEAGE.md`. | Existing Meituan backend metrics should not be silently renamed or redefined. |
| SQL diagnostics | Store-period, search, activity, top-SKU evidence. | SQL output files with ratios, shares, guardrail notes, and limitation notes. | SQL should not assign fixed store-stage labels or final operating decisions. |
| Generated memory facts | SQL outputs and supporting source tables. | Retrieval-facing memory facts with observed values, source fields, calculation notes, confidence, and limitations. | Memory facts are summaries, not raw backend exports. |
| Offline evaluation | Generated facts, SQL outputs, and current-scope docs. | Eval result text files and consistency checks. | Evaluations check evidence boundaries; they are not causal business experiments. |

## Evidence Type Boundary

| Evidence type | Examples | Current role | Interpretation limit |
|---|---|---|---|
| Backend-derived fields | `transaction_amount`, `entry_users`, `order_users`, `activity_orders`, `refund_amount` | Preserve Meituan backend metric meanings under canonical field names. | Observed metrics need context before stronger operating interpretation. |
| SQL-derived diagnostics | `search_entry_rate_pct`, `search_entry_share_pct`, `activity_order_share_pct`, `comparison_limit_notes` | Expose visibility-entry structure, activity involvement, refund and invalid-order pressure where available, product-mix signals, and interpretation limits. | Diagnostic signals are not peer-selection rules. |
| Retrieval-facing memory slots | `visibility_entry_profile`, `activity_lever_profile`, `transaction_conversion_profile`, `single_metric_attribution_guard`, `top3_sku_product_mix_note` | Store evidence with source fields, observed values, calculation notes, confidence, and limitations. | Memory slots keep evidence traceable rather than creating undocumented fields. |
| Future gate fields | `comparison_question_type`, `comparison_decision`, `market_area_type` | Planned contract fields for future pairwise comparability work. | These are future fields, not current Demo 2 output columns. |

## Implemented Source Files

Current source files include:

- `retail_ops/data/store_a_monthly_metrics.csv`
- `retail_ops/data/store_a_top_skus.csv`
- `retail_ops/data/demo2_store_period_metrics.csv`
- `retail_ops/data/demo2_top_search_terms.csv`
- `retail_ops/data/demo2_top_skus_by_transaction_amount.csv`
- `retail_ops/data/demo2_top_skus_by_sales_volume.csv`

## Implemented SQL Files

- `retail_ops/sql/01_store_a_month_over_month_diagnostic.sql`
- `retail_ops/sql/02_demo2_cross_store_comparability.sql`

The second file keeps its historical path name for reference stability; its current implemented meaning is a same-period cross-store diagnostic.

## Implemented Memory Outputs

- `retail_ops/outputs/generated_retail_memory_facts.json`
- `retail_ops/outputs/generated_demo2_retail_memory_facts.json`

The memory-facing facts record:

- store identity
- reporting period
- observed values
- source fields
- calculation notes
- evidence-trace confidence
- limitations



## Current Retail Implementation Scope

The current retail path has three implemented evidence layers.

| Layer | Scope | Purpose | Saved evidence |
|---|---|---|---|
| Demo 1: Store A month-over-month diagnostic | Store A, 2026-02 to 2026-04 | Explain one store's monthly movement with backend metric evidence and boundary notes. | SQL diagnostic output and generated memory facts. |
| Demo 2: same-period cross-store diagnostic | Stores B-F, 2026-03 | Review selected stores under one reporting window and one field contract before stronger cross-store claims. | Cross-store diagnostic output, comparison-scope fields, and generated memory facts. |
| Post-Demo2 repeated-window panel extension | Stores B-F, 2026-02 to 2026-04 | Check whether same-store repeated-window evidence exists before building a future pairwise comparability gate. | Panel coverage output, descriptive repeated-window summary, and validator result files. |

The current system organizes backend evidence, preserves metric definitions, records limitations, and prepares repeated-window evidence for future question-specific comparison.

Post-Demo2 repeated-window panel files:

| File | Role |
|---|---|
| `retail_ops/data/store_period_panel_metrics.csv` | B-F store-period panel for 2026-02 to 2026-04 using dictionary-aligned field names. |
| `retail_ops/data/store_period_panel_source_notes.md` | Source notes and exclusions for repeated-window panel fields. |
| `retail_ops/sql/03_store_period_panel_coverage.sql` | Checks whether each store has the monthly windows needed for descriptive panel review. |
| `retail_ops/sql/04_repeated_window_panel_summary.sql` | Produces descriptive February-to-April movement summaries. |
| `retail_ops/outputs/store_period_panel_coverage_output.csv` | Saved panel coverage output. |
| `retail_ops/outputs/repeated_window_panel_summary_output.csv` | Saved descriptive repeated-window summary output. |
| `retail_ops/scripts/validate_store_period_panel.py` | Validates panel coverage, canonical source fields, canonical `store_type` values, and repeated-window evidence boundaries. |
| `retail_ops/scripts/validate_repeated_window_panel_summary.py` | Validates descriptive summary shape and boundary-preserving summary notes. |
| `retail_ops/outputs/store_period_panel_validation_result.txt` | Saved validation result for the panel coverage layer. |
| `retail_ops/outputs/repeated_window_panel_summary_validation_result.txt` | Saved validation result for the repeated-window summary layer. |
