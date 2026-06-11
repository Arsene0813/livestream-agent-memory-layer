# Retail Operations Technical Appendix

This file consolidates the technical appendix material for the retail decision-support prototype.

The first reviewer path should remain short: project summary, root README, retail README, and `retail_ops/data/DATA_DICTIONARY.md`.

This appendix is for later audit of architecture, lineage, and field-usage consistency.

## Contents

| Section | Source merged here |
|---|---|
| Architecture | Consolidated from the former architecture appendix. |
| Source-to-Claim Lineage | Consolidated from the former lineage appendix. |
| Field-Usage Review | Consolidated from the former field-usage review appendix. |


---

## Architecture


## Document Ownership

| This file owns | Canonical file for related detail |
|---|---|
| Current retail evidence path | the Source-to-Claim Lineage section below for claim-to-data lineage |
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
| Metric contract | Canonical CSV fields and backend definitions. | `retail_ops/data/DATA_DICTIONARY.md` and the Source-to-Claim Lineage section below. | Existing Meituan backend metrics should not be silently renamed or redefined. |
| SQL diagnostics | Store-period, search, activity, top-SKU evidence. | SQL output files with ratios, shares, guardrail notes, and limitation notes. | SQL should not assign fixed store-stage labels or final operating decisions. |
| Generated memory facts | SQL outputs and supporting source tables. | Retrieval-facing memory facts with observed values, source fields, calculation notes, confidence, and limitations. | Memory facts are summaries, not raw backend exports. |
| Offline evaluation | Generated facts, SQL outputs, and current-scope docs. | Eval result text files and consistency checks. | Evaluations check evidence boundaries; they are not causal business experiments. |

## Evidence Type Boundary

| Evidence type | Examples | Current role | Interpretation limit |
|---|---|---|---|
| Backend-derived fields | `transaction_amount`, `entry_users`, `order_users`, `activity_orders`, `refund_amount` | Preserve Meituan backend metric meanings under canonical field names. | Observed metrics need context before stronger operating interpretation. |
| SQL-derived diagnostics | `search_entry_rate_pct`, `search_entry_share_pct`, `activity_order_share_pct`, `comparison_limit_notes` | Expose visibility-entry structure, activity involvement, product-mix signals, and interpretation limits. | Diagnostic signals are not peer-selection rules. |
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


---

## Source-to-Claim Lineage


This file owns claim-to-data lineage for the retail-operations evidence path.

It traces how selected Meituan backend metrics move from source CSV files into SQL diagnostics, SQL outputs, generated memory facts, and answer-boundary evaluations.

## Document Ownership

| This file owns | Canonical file for related detail |
|---|---|
| Source-to-SQL-to-memory lineage | the Architecture section above for architecture structure |
| Claim-to-field support | `retail_ops/data/DATA_DICTIONARY.md` for field meanings |
| Metric interpretation rules | `retail_ops/data/DATA_DICTIONARY.md` for full definitions |
| Current diagnostic boundaries | `retail_ops/EXPERIMENTS.md` and `retail_ops/EXPERIMENT_RESULTS.md` for test meaning and outcomes |
| Future pairwise comparability lineage | `retail_ops/COMPARABILITY_GATE_V0.md` for the future gate contract |

Path names that include `cross_store_comparability` are retained for reference stability. In the current implementation, Demo 2 means same-period diagnostic evidence and guardrails. The future pairwise comparability gate is documented separately.

## Shared Lineage Contract

Existing Meituan backend metrics are kept under one canonical English field name. This avoids mixing multiple English names for the same Chinese backend metric.

Main field-contract files:

- `retail_ops/data/DATA_DICTIONARY.md`
- `retail_ops/data/store_a_monthly_metrics.csv`
- `retail_ops/data/store_a_top_skus.csv`
- `retail_ops/data/demo2_store_period_metrics.csv`
- `retail_ops/data/demo2_top_search_terms.csv`
- `retail_ops/data/demo2_top_skus_by_sales_volume.csv`
- `retail_ops/data/demo2_top_skus_by_transaction_amount.csv`

Main SQL files:

- `retail_ops/sql/01_store_a_month_over_month_diagnostic.sql`
- `retail_ops/sql/02_demo2_cross_store_comparability.sql`

Main output files:

- `retail_ops/outputs/store_a_demo1_sql_output.csv`
- `retail_ops/outputs/demo2_cross_store_comparability_output.csv`
- `retail_ops/outputs/generated_retail_memory_facts.json`
- `retail_ops/outputs/generated_demo2_retail_memory_facts.json`

## Demo 1 Scope

| Item | Value |
|---|---|
| Store | Store A |
| Period | February 2026 to April 2026 |
| Data source | Manually structured Meituan merchant-backend metrics |
| Processing method | Offline SQL diagnostic |
| Output | SQL-derived CSV, markdown diagnostic, generated retail memory facts |
| Limitation | Single-store demo; not causal attribution; not cross-store comparison |

## Demo 1 Claim-to-Data Lineage

| Claim / diagnostic | Source fields | SQL output / derived metric | Memory slot | Limitation |
|---|---|---|---|---|
| Store A's visibility and entry structure can be described from exposure, ranking, entry, and search-entry metrics. | `exposure_users`, `store_average_rank`, `entry_users`, `search_exposure_users`, `search_average_rank`, `search_entry_users` | `search_exposure_share_pct`, `search_entry_share_pct`, `search_entry_rate_pct` | `visibility_entry_profile` | Describes whether the store was being seen and entered; does not prove causal growth. |
| Store A's activity metrics should be interpreted as operating-lever evidence. | `activity_original_transaction_amount`, `activity_orders`, `activity_cost`, `merchant_subsidy_amount`, `platform_subsidy_amount` | `activity_order_share_pct`, `activity_cost_ratio_pct`, `merchant_subsidy_share_of_activity_cost_pct` | `activity_lever_profile` | Activity is a tool inside the operating chain, not a standalone causal explanation or simple ROI judgment. |
| Store A's transaction and conversion signals moved in different directions. | `transaction_amount`, `transaction_orders`, `order_conversion_rate_pct`, `average_order_value` | `transaction_amount_mom_pct`, `transaction_orders_mom_pct`, `average_order_value_mom_pct` | `transaction_conversion_profile` | Transaction recovery can coexist with weaker conversion or lower average order value. |
| Store A's changes should not be explained by one metric alone. | Visibility, entry, transaction, conversion, activity, and SKU evidence | Combined multi-signal interpretation | `single_metric_attribution_guard` | The demo supports structured comparison of signals, not causal attribution. |
| Top SKU mix appears care-solution-heavy. | Top-3 SKU records | Top-3 SKU observation | `top3_sku_product_mix_note` | Top-3 evidence only; not full SKU category-share analysis. |

## Metric Lineage Rules

### Conversion Rate

`order_conversion_rate_pct` is the store-period backend funnel metric used with `entry_users` and `order_users`.

~~~text
order_conversion_rate_pct = order_users / entry_users * 100
~~~

This keeps the conversion metric tied to its documented denominator and reporting grain.

### Traffic Source

Traffic-source users may overlap. The same customer may see the store through multiple exposure sources, so source-level exposure users should not be summed into total exposure users.

`search_entry_users / entry_users` is used only as a directional source-entry structure signal.

### Activity and Promotion

`activity_cost_ratio_pct` follows the backend formula:

~~~text
activity_cost_ratio_pct = activity_cost / activity_original_transaction_amount * 100
~~~

A smaller value means lower activity cost per unit of activity-driven revenue. This project avoids calling it traditional ROI because traditional ROI is often interpreted in the opposite direction.

### Transaction Metrics

`transaction_amount` and `transaction_orders` refer to same-day paid and same-day not-cancelled orders.

For the transaction metric page:

~~~text
average_order_value = transaction_amount / transaction_orders
~~~

If another backend page defines 单均价 using a different backend-reported denominator, it should be treated as a separate backend-reported metric rather than mixed with transaction fields.

### Estimated Income

`estimated_income_proxy` is treated as a platform-displayed income proxy. It should not be interpreted as audited profit because the current demo does not contain the full platform calculation breakdown.

### Refund


### Ranking

Business-district ranking is only comparable among merchants in the same main category and business district. Ranking may be unavailable when the store has no honeycomb or grid information, or no sales activity.

## SKU Evidence Grain Note

Top-SKU evidence uses SKU-level fields.

For Demo 1, the source is:

- `retail_ops/data/store_a_top_skus.csv`

For Demo 2, the sources are:

- `retail_ops/data/demo2_top_skus_by_sales_volume.csv`
- `retail_ops/data/demo2_top_skus_by_transaction_amount.csv`

Lineage rules:

- `sku_transaction_amount` is SKU-period-level transaction evidence.
- It must not be confused with store-period-level `transaction_amount`.
- Top-SKU evidence is used only as lightweight product-mix support.
- Top-SKU evidence is not full category-level sales-share analysis.

## Demo 2 Same-Period Diagnostic Lineage

Demo 2 extends the retail operations prototype from a single-store month-over-month diagnostic to a same-period cross-store diagnostic.

The current Demo 2 scope is limited to five anonymized stores:

- Store B
- Store C
- Store D
- Store E
- Store F

All Demo 2 records use the same reporting window:

| Field | Value |
|---|---|
| `period_start` | 2026-03-01 |
| `period_end` | 2026-03-31 |
| `period_month` | 2026-03 |

Demo 2 structures selected backend metrics under the same reporting window and field contract, derives cautious diagnostic signals, and preserves interpretation limits before any operating recommendation is made.

## Demo 2 Source Data

Demo 2 source data is stored in:

- `retail_ops/data/demo2_store_period_metrics.csv`
- `retail_ops/data/demo2_top_search_terms.csv`
- `retail_ops/data/demo2_top_skus_by_sales_volume.csv`
- `retail_ops/data/demo2_top_skus_by_transaction_amount.csv`
- `retail_ops/data/demo2_source_notes.md`

The source metrics are manually transcribed from the Meituan merchant-backend UI used for instant-retail store operations and anonymized at the store level.

Original Chinese backend search terms and SKU names are retained for traceability. English helper columns are included only for readability.

## Demo 2 SQL Diagnostic Output

Demo 2 SQL is stored in:

- `retail_ops/sql/02_demo2_cross_store_comparability.sql`

The generated SQL output is stored in:

- `retail_ops/outputs/demo2_cross_store_comparability_output.csv`

The SQL uses the March 2026 reporting window as a Demo 2 fixture contract. This keeps the current sample reproducible, but it should not be read as a reusable production SQL design for arbitrary 48-store reporting windows.

Carried-through canonical or backend-formula fields include:

- `region_type`
- `store_type`
- `business_district_rank`
- `activity_cost_ratio_pct`

SQL-derived diagnostic fields include:

- `search_entry_rate_pct`
- `search_entry_share_pct`
- `activity_order_share_pct`
- `top3_sku_transaction_amount_share_pct`
- `comparison_scope_flag`
- `comparison_limit_notes`

These derived fields are diagnostic summaries. They do not replace Meituan backend definitions, rank stores, assign store stages, or prove causal operating effects.

## Demo 2 Claim-to-Field Mapping

| Claim / diagnostic | Supporting fields | Interpretation limit |
|---|---|---|
| Stores are in the same Demo 2 reporting window. | `period_month`, `period_start`, `period_end` | Same-period alignment improves diagnostic structure but does not remove differences in region, store type, activity conditions, competition, fulfillment, or SKU mix. |
| Visibility and entry can be compared cautiously across stores. | `exposure_users`, `entry_users`, `entry_conversion_rate_pct`, `search_exposure_users`, `search_entry_users`, `search_entry_rate_pct`, `search_entry_share_pct` | Visibility and entry metrics do not prove causal transaction growth. |
| Activity involvement should constrain cross-store transaction comparison. | `activity_orders`, `activity_order_share_pct`, `activity_cost`, `activity_cost_ratio_pct`, `merchant_subsidy_amount`, `platform_subsidy_amount` | Activity mechanism details and promotion cycle dates are not included. |
| Top search terms provide lightweight demand evidence. | `search_term`, `search_term_exposure_times`, `search_term_click_times`, `search_term_order_times` | Top search terms are store-period evidence, not complete regional consumer-preference proof. |
| Top SKU evidence provides lightweight product-mix evidence. | `sku_name`, `sku_transaction_amount`, `sales_volume`, `top3_sku_transaction_amount_share_pct` | Top-3 evidence is not full SKU category-share analysis. |

## Demo 2 Memory Fact Output

Demo 2 generated memory facts are stored in:

- `retail_ops/outputs/generated_demo2_retail_memory_facts.json`

The generation script is:

- `retail_ops/scripts/generate_demo2_retail_memory_facts.py`

The validation script is:

- `retail_ops/scripts/validate_demo2_retail_memory_facts.py`

Demo 2 reuses existing canonical retail memory slots:

- `visibility_entry_profile`
- `activity_lever_profile`
- `transaction_conversion_profile`
- `top3_sku_product_mix_note`
- `single_metric_attribution_guard`

Demo 2 does not introduce store-stage labels or best-store rankings.

## Demo 2 Carry-Through Note: Order and Payment Amount Fields

The current implementation carries `order_amount` and `payment_amount` from:

- `retail_ops/data/demo2_store_period_metrics.csv`

into:

- `retail_ops/outputs/demo2_cross_store_comparability_output.csv`
- `retail_ops/outputs/generated_demo2_retail_memory_facts.json`

Interpretation boundary:

- `order_amount` is read with `order_users`, `order_times`, and `order_conversion_rate_pct`.
- `payment_amount` is read with `payment_users` and `payment_conversion_rate_pct`.
- `transaction_amount` remains a separate transaction metric and should not be merged with order-submission or payment-funnel amount fields.

## Future Comparability-Gate Lineage

The current implemented retail lineage includes Demo 1, Demo 2, and the post-Demo2 repeated-window panel evidence-preparation layer. Demo 1 traces Store A month-over-month evidence, Demo 2 traces same-period B-F diagnostic evidence, and the panel lineage traces B-F repeated-window coverage and descriptive summary outputs.

- selected Meituan backend fields
- `DATA_DICTIONARY.md` definitions
- canonical CSV files
- Demo 1 and Demo 2 SQL diagnostics
- Demo 1 and Demo 2 output CSV files
- generated Demo 1 and Demo 2 retail memory facts
- validation and evaluation for the implemented scope

The future pairwise comparability gate should extend this lineage only after stronger multi-store evidence is available. The detailed future gate contract is kept in:

- `retail_ops/COMPARABILITY_GATE_V0.md`

## Post-Demo2 Repeated-Window Panel Lineage

The repeated-window panel extension follows the same dictionary-first rule as Demo 1 and Demo 2.

Panel coverage lineage:

| Step | Artifact |
|---|---|
| Metric definitions | `retail_ops/data/DATA_DICTIONARY.md` |
| Source panel | `retail_ops/data/store_period_panel_metrics.csv` |
| Coverage SQL | `retail_ops/sql/03_store_period_panel_coverage.sql` |
| Coverage output | `retail_ops/outputs/store_period_panel_coverage_output.csv` |
| Validator | `retail_ops/scripts/validate_store_period_panel.py` |
| Saved validation result | `retail_ops/outputs/store_period_panel_validation_result.txt` |

Repeated-window summary lineage:

| Step | Artifact |
|---|---|
| Source panel | `retail_ops/data/store_period_panel_metrics.csv` |
| Summary SQL | `retail_ops/sql/04_repeated_window_panel_summary.sql` |
| Summary output | `retail_ops/outputs/repeated_window_panel_summary_output.csv` |
| Validator | `retail_ops/scripts/validate_repeated_window_panel_summary.py` |
| Saved validation result | `retail_ops/outputs/repeated_window_panel_summary_validation_result.txt` |

This panel does not create a pairwise comparability gate. It checks whether Stores B-F have repeated monthly evidence across 2026-02, 2026-03, and 2026-04, then summarizes movement descriptively.

The panel keeps selected dictionary-defined refund backend fields without interpreting refund reasons. It also keeps `store_type` values aligned with the existing source data: `self-operated` and `partner`.

## Panel Refund Field Trace

The repeated-window panel retains these dictionary-defined refund backend fields without interpreting refund reasons:

- `refund_amount`
- `full_refund_orders`
- `refund_orders_all_or_partial`


---

## Field-Usage Review


## Document Ownership

This file owns field-name and semantic-change review for the retail-operations evidence path.

It should answer one narrow question: if a field name or field meaning changes, what existing dictionary definition, source file, SQL output, generated memory fact, and evaluation behavior could be affected?

| This file owns | Canonical file for related detail |
|---|---|
| Field-name change review | `retail_ops/data/DATA_DICTIONARY.md` for authoritative field names and metric meanings |
| Existing field usage review | Source CSV files, SQL outputs, generated memory facts, and eval files |
| Future field-addition caution | `retail_ops/COMPARABILITY_GATE_V0.md` for future pairwise gate fields |
| Rename decision tracking | The field review tables in this file |

This file should not repeat the full admissions narrative, the full architecture path, or the future comparability-gate design. It should keep the review narrow: field meaning, field location, rename risk, and whether a change is allowed.


This file records the field-name review before expanding the retail comparability narrative.

Current decision: **no existing source CSV field, SQL output field, generated memory slot, or evaluation field is renamed in this patch**.

The purpose of this review is to protect the Meituan backend metric contract before future comparability-gate work. Backend-derived fields, SQL-derived diagnostic fields, and retrieval-facing memory slots should not be mixed, renamed, or promoted into new meanings without an explicit mapping review.

## Consolidated Scope Notes

This file also preserves the field-name and scope-change guardrails that protect the current retail evidence path.

- `retail_ops/data/DATA_DICTIONARY.md` remains the source of truth for retail field names and Meituan-style metric meanings.
- Demo 1 remains a Store A month-over-month diagnostic.
- Demo 2 remains a same-period B-F diagnostic for March 2026, not a completed pairwise comparability gate.
- `region_type` remains weak region or market-context evidence only; it is not a hard market-area classification, store-stage label, or peer-store grouping rule.
- Activity evidence should remain separated into involvement, intensity, and future explicit campaign status only when campaign-calendar or backend status evidence exists.
- Retrieval-score analysis remains offline inspection, not production retrieval logic.
- `rac/` should be described as a deterministic source-aware review scaffold over the structured retail evidence path, not as autonomous cognition or a separate operating-decision engine.


## Review Rule

Any future field-name change must pass this review first:

| Existing field | Dictionary definition | Current usage | Rename decision |
|---|---|---|---|
| TBD | Must be checked against `retail_ops/data/DATA_DICTIONARY.md`. | Must list CSV, SQL, output, memory-fact, lineage, README/admissions, and eval usage. | Do not rename unless the full source-to-output path is migrated together. |

Future fields such as `activity_status`, `market_area_type`, `market_area_type_source`, `market_area_type_confidence`, `comparison_question_type`, or `comparison_decision` must not be introduced into source CSVs, SQL outputs, generated facts, or eval cases until they are first documented in `retail_ops/data/DATA_DICTIONARY.md` and linked through the Source-to-Claim Lineage section of this appendix.

## Field-Change Migration Order

Any future field rename or semantic change must be migrated in this order:

1. update `retail_ops/data/DATA_DICTIONARY.md`;
2. update the Source-to-Claim Lineage section of this appendix;
3. update this field-usage review table;
4. update source CSV headers only if the field is a source field;
5. update SQL outputs only if the field is a SQL-derived diagnostic field;
6. update generated memory facts and source-field references;
7. update validation scripts and expected outputs;
8. update README, admissions summary, and demo docs only after the data contract is stable.

This rule is intentionally conservative. The project should prefer adding clearly documented future fields over silently changing the meaning of existing Meituan backend-derived fields.


## Current Field Review Table

| Existing field | Dictionary definition or boundary | Current use location | Rename decision |
|---|---|---|---|
| `store_id` | Canonical store identifier used in source CSV files, SQL diagnostics, and metric outputs. | Source CSVs, SQL outputs, demo outputs. | No. |
| `entity_id` | Retrieval-layer identifier generated from `store_id` using `entity_id = "store_" + store_id`. | Generated retail memory facts. | No. |
| `period_start` | First date of the reporting window. | Source CSVs, SQL outputs, generated facts, lineage. | No. |
| `period_end` | Last date of the reporting window. | Source CSVs, SQL outputs, generated facts, lineage. | No. |
| `period_month` | Calendar-month label for monthly demo records. | Source CSVs, SQL outputs. | No. |
| `region_type` | Weak region or market-context metadata. It is not a store-stage label, mature market-area classification, consumption-level group, or sufficient comparability condition by itself. | Demo 2 source metrics, SQL output, generated facts, comparability review. | No. |
| `store_type` | Store operating-type field used as comparison context. | Source CSVs, SQL output, generated facts. | No. |
| `business_district_rank` | Backend contextual ranking among same-main-category merchants in a business district. It is supplementary context, not a hard comparability condition. | Demo 2 source metrics and lineage. | No. |
| `exposure_users` | Backend-reported number of users who saw the merchant in the selected scope. | Source CSVs, SQL outputs, visibility facts. | No. |
| `exposure_times` | Backend-reported number of times the merchant was seen. | Source CSVs and visibility evidence. | No. |
| `store_average_rank` | Backend-reported average exposure rank. Lower means better position. | Source CSVs, SQL outputs, visibility facts. | No. |
| `search_exposure_users` | Backend-reported users who saw the merchant through search-result exposure. | Source CSVs, SQL outputs, visibility facts. | No. |
| `search_average_rank` | Backend-reported average search-result exposure position. | Source CSVs, SQL outputs, visibility facts. | No. |
| `entry_users` | Backend-reported users entering the store during the selected period. | Source CSVs, SQL output, visibility and conversion facts. | No. |
| `entry_times` | Backend-reported store-entry visits/actions. It is not the same as `entry_users`. | Source CSVs and traffic evidence. | No. |
| `entry_conversion_rate_pct` | Backend-style entry conversion rate, interpreted with exposure and entry scope. | Source CSVs, SQL outputs. | No. |
| `search_entry_users` | Backend-reported users entering from search during the selected period. | Source CSVs, SQL output, visibility facts. | No. |
| `search_entry_rate_pct` | SQL-derived search exposure-to-entry diagnostic. | Demo 2 SQL output and lineage. | No. |
| `search_entry_share_pct` | SQL-derived directional structure metric for the share of store entry attributed to the search source. Source-level users may overlap. | Demo 2 SQL output, generated facts, lineage. | No. |
| `order_users` | Backend order-user metric used in the backend order-conversion formula. | Source CSVs, SQL output, transaction/conversion facts. | No. |
| `order_times` | Backend order-submission/action-count metric. It is not the same as `order_users`. | Source CSVs and funnel evidence. | No. |
| `order_amount` | Backend order-submission amount field. It belongs to the order-submission funnel and must not be merged with `transaction_amount`. | Demo 2 source metrics, SQL output, generated facts, lineage. | No. |
| `order_conversion_rate_pct` | Backend formula field: `order_users / entry_users * 100`. It must not be recomputed from an alternative project-side numerator. | Source CSVs, SQL output, lineage, transaction/conversion facts. | No. |
| `payment_users` | Backend successful-payment-user metric. | Source CSVs, SQL output, transaction/conversion facts. | No. |
| `payment_amount` | Backend paid-order commodity amount field. It belongs to the payment funnel and must not be merged with `transaction_amount`. | Demo 2 source metrics, SQL output, generated facts, lineage. | No. |
| `payment_conversion_rate_pct` | Backend payment-conversion metric. | Source CSVs, SQL output, transaction/conversion facts. | No. |
| `transaction_amount` | Backend transaction amount for same-day paid and same-day not-cancelled orders under the selected scope. It must not be mixed with `gross_revenue`, `estimated_income_proxy`, `order_amount`, `payment_amount`, or SKU-level `sku_transaction_amount`. | Source CSVs, SQL output, transaction/conversion facts. | No. |
| `transaction_orders` | Backend transaction-order count for same-day paid and same-day not-cancelled orders. | Source CSVs, SQL output, transaction/conversion facts. | No. |
| `average_order_value` | Backend average-order-value field read with `transaction_amount` and `transaction_orders`. | Source CSVs, SQL output. | No. |
| `estimated_income_proxy` | Platform-displayed estimated income / estimated order income proxy. It is not audited profit. | Source CSVs, SQL output, transaction/conversion facts, evidence-boundary docs. | No. |
| `activity_original_transaction_amount` | Original transaction amount of orders that used activities. | Source CSVs, SQL output, activity facts. | No. |
| `activity_orders` | Backend activity-driven order count. | Source CSVs, SQL output, activity facts. | No. |
| `activity_cost` | Backend activity-cost field. | Source CSVs, SQL output, activity facts. | No. |
| `merchant_subsidy_amount` | Merchant-borne subsidy amount. | Source CSVs, SQL output, activity facts. | No. |
| `platform_subsidy_amount` | Platform-borne subsidy amount. | Source CSVs, SQL output, activity facts. | No. |
| `activity_cost_ratio_pct` | Activity cost divided by activity original transaction amount. It is activity-cost-ratio evidence, not traditional ROI. | Source CSVs, SQL output, activity facts, lineage. | No. |
| `activity_order_share_pct` | SQL-derived activity-order share. It shows activity involvement, not full campaign status, promotion mechanism, causal demand lift, or promotion-transfer readiness. | Demo 2 SQL output, generated facts, comparability review. | No. |
| `sku_name` | SKU-level product name from top-SKU evidence. | Top-SKU source files. | No. |
| `sku_name_en` | English helper column for readability. It does not replace the original Chinese SKU name. | Top-SKU source files. | No. |
| `sku_transaction_amount` | SKU-level transaction amount. It must not be confused with store-period-level `transaction_amount`. | Top-SKU source files and top-SKU evidence. | No. |
| `sales_volume` | SKU-level sales-volume evidence where available. | Top-SKU source files. | No. |
| `top3_sku_transaction_amount_share_pct` | SQL-derived lightweight top-SKU concentration evidence. It is not full product-category sales share. | SQL output and top-SKU memory note. | No. |
| `comparison_scope_flag` | SQL-derived data-readiness and comparison-scope guardrail for Demo 2. It is not a pairwise store-comparability decision. | Demo 2 SQL output and Demo 2 memory facts. | No. |
| `comparison_limit_notes` | SQL-derived interpretation-boundary notes for Demo 2. It records constraints from search, activity, source-field, region/store context, and product-mix evidence. | Demo 2 SQL output and Demo 2 memory facts. | No. |
| `visibility_entry_profile` | Retrieval-facing memory slot for exposure, ranking, entry, and search-entry structure. | Generated retail memory facts. | No. |
| `activity_lever_profile` | Retrieval-facing memory slot for activity orders, activity cost, subsidy, and activity-cost ratio. | Generated retail memory facts. | No. |
| `transaction_conversion_profile` | Retrieval-facing memory slot for transaction scale, order conversion, payment, and average order value. | Generated retail memory facts. | No. |
| `single_metric_attribution_guard` | Retrieval-facing memory slot that prevents unsupported interpretation from one metric alone. | Generated retail memory facts. | No. |
| `top3_sku_product_mix_note` | Retrieval-facing memory slot for limited top-SKU evidence. It is not full category-share analysis. | Generated retail memory facts. | No. |

## Future Comparability-Gate Field Review

Pairwise comparability-gate fields are outside the current implemented retail scope.

A reliable future gate should consider transaction order volume, transaction amount, explicit activity status when source evidence exists, activity involvement, activity intensity, store type, region and market context, competition environment, SKU structure, fulfillment or stockout evidence where available, and repeated reporting windows.

At the current sample size, `region_type` remains weak context only. It must not be used as a hard market-area classification, store-stage label, consumption-level label, or peer-store grouping rule.

Possible future fields such as `activity_status`, `market_area_type`, `market_area_type_source`, `market_area_type_confidence`, `comparison_question_type`, or `comparison_decision` should only be added after they are documented in `retail_ops/data/DATA_DICTIONARY.md` and linked through the Source-to-Claim Lineage section of this appendix.

## Current Decision

No current source CSV field, SQL output field, generated memory slot, or evaluation field is renamed in this patch.
