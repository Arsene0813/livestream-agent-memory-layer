# Retail Operations Architecture

This file gives a compact map of the current retail decision-support prototype.

The project starts from a practical Meituan instant-retail problem: single-store backend reports are detailed, but they are mainly designed for reviewing one store at a time. Cross-store decisions need aligned reporting windows, comparable store context, and explicit interpretation limits.

## Current Demo Scope

| Demo | Status | Purpose |
|---|---|---|
| Demo 1 | Implemented | Store A month-over-month diagnostic. |
| Demo 2 | Implemented | Stores B-F same-period diagnostic structure. |
| Pairwise comparability gate | Future work | Planned gate for judging whether selected store-period records can be compared for a specific operating question. |

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

The design is intentionally file-based at this stage. The priority is to keep each diagnostic claim traceable to source fields and output files before expanding toward a larger 48-store workflow.

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

This diagram describes the implemented evidence path for the current retail demos. It should not be read as production Meituan API integration or as an implemented pairwise comparability gate.

## Retrieval Mode Boundary

The current repository contains more than one retrieval path. They should not be read as the same level of implementation maturity.

| Path | Current mode | Current role | Boundary |
|---|---|---|---|
| `/chat_livestream_kb` | Qdrant-backed retrieval over livestream/product memory facts. | Tests lifecycle-aware memory behavior, including typed facts, active-state filtering, and fallback/refusal behavior. | Original memory-layer path, not the Meituan multi-store diagnostic system. |
| `/chat_retail_ops_kb` | Retail memory retrieval path for implemented Store A facts. | Tests whether retail memory facts can be retrieved with source fields and limitations. | Limited to the implemented retail facts; not full 48-store automation. |
| `/chat_retail_ops_demo2_kb` | File-backed generated Demo 2 retail memory facts. | Tests whether B-F same-period diagnostic facts can be returned or refused inside the current evidence boundary. | Not retrieval-score evaluation, production Meituan API integration, or a pairwise comparability gate. |
| Future pairwise comparability gate | Not implemented. | Planned gate for judging whether two store-period records can be compared for a specific operating question. | Should be implemented only after broader store-period coverage, repeated windows, and stronger market-context evidence exist. |

The Demo 2 endpoint is intentionally file-backed at this stage. Its purpose is to test evidence-boundary behavior after SQL diagnostics have been converted into memory facts. It should not be used to justify a retrieval-score threshold or to claim that pairwise store comparability has already been solved.

The retrieval-threshold calibration is a separate offline inspection over the file-backed evidence corpus. It should not be read as the runtime selection logic of `/chat_retail_ops_demo2_kb`.

## Layer Contract

| Layer | Input | Output | Boundary |
|---|---|---|---|
| Backend evidence | Selected Meituan merchant-backend metrics and manually structured evidence tables. | Canonical CSV source files. | Not full automated ingestion. |
| Metric contract | Canonical CSV fields and backend definitions. | `retail_ops/data/DATA_DICTIONARY.md` and `retail_ops/LINEAGE.md`. | Existing Meituan backend metrics should not be silently renamed or redefined. |
| SQL diagnostics | Store-period, search, activity, refund, order-quality, and top-SKU evidence. | SQL output files with ratios, shares, pressure indicators, and limitation notes. | SQL should not assign fixed store-stage labels or final operating decisions. |
| Generated memory facts | SQL outputs and supporting source tables. | Retrieval-facing memory facts with observed values, source fields, calculation notes, confidence, and limitations. | Memory facts are summaries, not raw backend exports. |
| Offline evaluation | Generated facts, SQL outputs, and current-scope docs. | Eval result text files and consistency checks. | Evaluations check evidence boundaries; they are not causal business experiments. |

## Evidence Type Boundary

| Evidence type | Examples | Current role | Not sufficient for |
|---|---|---|---|
| Backend-derived fields | `transaction_amount`, `entry_users`, `order_users`, `activity_orders`, `refund_amount` | Preserve Meituan backend metric meanings under canonical field names. | Causal explanation or direct cross-store strategy transfer. |
| SQL-derived diagnostics | `search_entry_rate_pct`, `search_entry_share_pct`, `activity_order_share_pct`, `refund_pressure_pct`, `invalid_order_pressure_pct`, `comparison_limit_notes` | Expose visibility-entry structure, operating-context pressure metrics, and interpretation limits. | Peer selection, store ranking, or final operating decisions. |
| Retrieval-facing memory slots | `visibility_entry_profile`, `activity_lever_profile`, `transaction_conversion_profile`, `order_quality_pressure_profile` | Store evidence with source fields, observed values, calculation notes, confidence, and limitations. | Replacing source metrics or inventing undocumented fields. |
| Future gate fields | `comparison_question_type`, `comparison_decision`, `market_area_type` | Planned contract fields for future pairwise comparability work. | Current Demo 2 output or current source-table schema. |

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

The second file keeps its historical path name for reference stability, but its current implemented meaning is a same-period cross-store diagnostic, not an implemented pairwise gate.

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

## Current Boundary

The current implemented retail scope stops at Demo 2.

The future pairwise comparability gate is documented in `retail_ops/COMPARABILITY_GATE_V0.md`, but the current SQL output and generated facts should be read as diagnostic evidence rather than as a transfer rule or store-ranking system.
