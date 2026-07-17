# Project Summary for Admissions Review

## Reviewer Reading Path

This file is the single application-facing entry point.

| Step | File | What to inspect |
|---:|---|---|
| 1 | `PROJECT_SUMMARY_FOR_ADMISSIONS.md` | Business origin, evidence coverage, architecture, implemented scope, and decision boundary. |
| 2 | `retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md` | Store A month-over-month evidence and multi-metric interpretation. |
| 3 | `retail_ops/demo/demo_2_cross_store_comparability_diagnostic.md` | Same-period B-F evidence and interpretation guardrails. |
| 4 | `retail_ops/outputs/store_period_panel_coverage_output.csv` and `retail_ops/outputs/repeated_window_panel_summary_output.csv` | Repeated-window B-F coverage and descriptive summary. |
| 5 | `retail_ops/EXPERIMENT_RESULTS.md` | Experiment questions, procedures, results, and failure modes. |
| 6 | `rac/DEMO_INDEX.md` | Factor-aware grounded review cases, reports, pipeline stages, and quality gate. |

The full project evolution is preserved in
`case_studies/from_livestream_to_retail_decision_support.md`.

Use `retail_ops/data/DATA_DICTIONARY.md`,
`retail_ops/TECHNICAL_APPENDIX.md`, and
`retail_ops/COMPARABILITY_GATE_V0.md` as contract and technical
references while reviewing the implemented evidence.

## Project Title

Meituan Instant-Retail Decision Support Prototype

Repository: `livestream-agent-memory-layer`

## Project Summary

This staged local prototype grew from a real operating problem in a 48-store Meituan instant-retail business. The Meituan merchant backend contains detailed single-store metrics, but it is mainly designed for reviewing one store at a time. As the store count increased, the harder question became how to compare store-period records without flattening different reporting windows, activity conditions, product structures, store types, and market contexts into the same judgment.

## Evidence Coverage

| Scope | Current evidence |
|---|---|
| Business context | The operating problem emerged in a 48-store Meituan instant-retail business. |
| Repository evidence | Manually transcribed and anonymized observations for six stores: Store A and selected Stores B-F. |
| Demo 1 | Store A across February, March, and April 2026. |
| Demo 2 | Stores B-F under one March 2026 reporting window. |
| Repeated-window panel | Stores B-F across February, March, and April 2026, using the supporting-table coverage documented in the repository. |
| Current decision use | Descriptive diagnosis, evidence lineage, scope checks, and retrieval or answer-boundary evaluation. |

The current retail decision-support path turns selected, manually transcribed merchant-backend metrics into a traceable evidence chain:

1. Meituan backend metric evidence
2. `DATA_DICTIONARY.md` field contract
3. SQL diagnostic output
4. generated retail memory facts
5. boundary-preserving answer checks

Demo 1 structures Store A February-April 2026 movement as a multi-metric operating profile. Demo 2 structures selected Stores B-F under one March 2026 reporting window and one field contract. The repeated-window panel adds B-F coverage across 2026-02, 2026-03, and 2026-04 as preparation for future question-specific pairwise comparability rules.

The repository also includes a deterministic factor-aware grounded review layer. It decomposes operating questions into factors, routes each factor to local evidence or boundary evidence, applies critique and fact checks, and produces inspectable grounded reports over the structured retail evidence.

The current project does not claim a completed pairwise comparability gate. A future gate should judge whether two store-period records can be compared for one selected operating question, using factors such as reporting-window alignment, transaction order volume, transaction amount, activity involvement, activity intensity, store type, repeated-window stability, weak region context, competition evidence, SKU structure, and fulfillment or stockout context.

## Document Ownership

| File | Primary responsibility |
|---|---|
| `PROJECT_SUMMARY_FOR_ADMISSIONS.md` | Application-facing business, evidence, architecture, and evaluation narrative. |
| `README.md` | Repository-level map, architecture, reproduction commands, and implementation scope. |
| `case_studies/from_livestream_to_retail_decision_support.md` | Complete design evolution from livestream product memory to retail decision support. |
| `retail_ops/data/DATA_DICTIONARY.md` | Authoritative field names, metric definitions, formulas, grains, and interpretation boundaries. |
| `retail_ops/EXPERIMENT_RESULTS.md` | Implemented experiment questions, procedures, results, and failure modes. |
| `rac/DEMO_INDEX.md` | Reviewer-facing RAC cases, grounded reports, execution path, and quality-gate evidence. |
| `retail_ops/TECHNICAL_APPENDIX.md` | Consolidated lineage, field usage, architecture details, and technical limitations. |
| `retail_ops/COMPARABILITY_GATE_V0.md` | Future question-specific pairwise comparability contract. |

## Operating Step Summary

| Operating step | Practical meaning |
|---|---|
| Backend metric evidence | Selected operator-provided values were manually transcribed from the Meituan merchant-backend UI, anonymized at store level, and structured under the documented field contract. |
| Metric dictionary | Backend metric meanings are preserved under canonical project field names. |
| SQL diagnostics | Selected store-period records are structured into diagnostic outputs under one reporting window and one field contract. |
| Generated memory facts | Diagnostic evidence is stored with source fields, observed values, confidence labels, and limitations. |
| Boundary checks | Later answers are checked against entity scope, period scope, metric definitions, and comparison limits. |
| Future comparability gate | Planned pairwise logic will decide whether two store-period records can be compared for a specific operating question. |
| RAC grounded review layer | Factor-aware review layer for decomposing questions, routing evidence, applying critique, checking facts, and producing inspectable grounded reports over local evidence. |

## Business Problem

In Meituan instant retail, store competition is not only about having products online. For standardized products such as contact lenses, care solutions, and related eye-care items, stores compete through a local operating chain:

```text
being seen -> being entered -> being ordered -> being selected again / maintaining share
```

Promotion, subsidy, price adjustment, SKU arrangement, ranking optimization, and fulfillment stability are tools inside this chain. Their meaning depends on store state, local competition, activity intensity, product mix, and reporting-window alignment.

A new store may use activity support to gain first exposure and first orders. A store under local price pressure may use pricing or subsidy tools to defend visibility and market share. A store with high exposure may still underperform if entry quality, order conversion, or SKU concentration creates friction.

The practical question is whether the available evidence is strong enough to compare records and decide which operating interpretation is actually usable.

## Why This Became a Data-Science Problem

The merchant backend provides detailed metrics, but it is mainly organized for single-store monitoring. In a 48-store operation, detailed single-store metrics do not automatically become reliable cross-store decision evidence.

A cross-store decision should preserve metric definitions, align reporting windows, expose comparison limits, and separate observed evidence from operating interpretations that the current data cannot yet support.

The current prototype treats SQL as the structuring layer and the memory layer as the evidence-retention layer. SQL organizes selected source records into diagnostic outputs. Generated memory facts then preserve the store, period, source fields, observed values, `calculation` metadata, confidence labels, and limitations so that later retrieval or answer generation does not lose the boundary around the evidence.

## Current Prototype Layers

| Layer | What it does |
|---|---|
| Metric dictionary | Preserves Meituan backend metric meanings and canonical project field names. |
| SQL diagnostic layer | Converts selected manually structured store-period records into diagnostic outputs under one field contract. |
| Generated memory facts | Stores evidence with source paths, source fields, observed values, `calculation` metadata, confidence labels, and limitations. |
| Boundary evaluations | Checks whether later answers preserve entity scope, period scope, metric definitions, and comparison limits. |
| RAC grounded review layer | Provides deterministic factor expansion, evidence routing, critique, fact checking, evidence-coverage update, and grounded report generation over local project evidence. |
| Future comparability gate | Planned pairwise gate for deciding whether two store-period records can be compared for a specific operating question. |

The memory layer records evidence and limitations so that different stores, periods, activity conditions, product structures, and regional contexts are not collapsed into the same comparison. The RAC layer makes the reasoning path inspectable when a question involves multiple factors and uneven evidence coverage.

## Current Implemented Retail Path

### Demo 1: Store A Month-over-Month Diagnostic

Demo 1 analyzes Store A across February, March, and April 2026. It shows why one metric alone is not enough. The purpose is to preserve a careful operating profile, not to label a month as simply good or bad.

Main file:

- `retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md`

### Demo 2: Same-Period B-F Diagnostic

Demo 2 extends the analysis to selected Stores B-F under the same March 2026 reporting window. Current SKU evidence uses selected top-SKU ranking views rather than full catalogue-level product-category classification.

Main file:

- `retail_ops/demo/demo_2_cross_store_comparability_diagnostic.md`

A guardrail-sensitivity check is included because the current Demo 2 thresholds are prototype interpretation thresholds, not optimized business cutoffs. In the current B-F sample, four of five rows change when the existing SQL thresholds are raised by 5 percentage points, while the easier-to-trigger scenario produces no note changes. This is a small-sample sensitivity result, so the thresholds remain diagnostic warnings rather than optimized peer-comparison rules.

A query-robustness inspection is also included to test whether supported, unsupported, hard-negative, entity or period mismatch, and ambiguous comparison queries behave consistently under small wording variations.

### Repeated-Window B-F Panel

The repeated-window panel adds selected Stores B-F across 2026-02, 2026-03, and 2026-04. Its role is to make repeated store-period coverage visible before the future question-specific pairwise comparability gate exists.

The panel verifies three-month coverage and reports descriptive February-to-April movement for selected fields. It prepares the evidence base for future comparison rules. Testing monthly guardrail stability will require repeated top-SKU evidence and monthly recomputation of `comparison_limit_notes`.

Main outputs:

- `retail_ops/outputs/store_period_panel_coverage_output.csv`
- `retail_ops/outputs/repeated_window_panel_summary_output.csv`

## Factor-Aware Grounded Review Layer

The project also includes a deterministic source-aware review layer over local project evidence. This is an important technical component because it makes multi-factor reasoning inspectable before an answer is written.

The layer decomposes an operating question into relevant factors, routes each factor to local evidence or boundary evidence, generates competing hypotheses, applies critique and fact checks, and produces a grounded report with confidence, limitations, source paths, and local evidence snippets.

This layer is useful when current evidence is strong for one factor but weak, missing, or boundary-only for another. It helps prevent a grounded answer from hiding missing evidence behind a fluent conclusion.

Reviewer entry points:

| File | Purpose |
|---|---|
| `rac/DEMO_INDEX.md` | Index of deterministic RAC demo cases. |
| `rac/outputs/grounded_rac_store_a_attribution_001.md` | Store A attribution-boundary review. |
| `rac/outputs/grounded_rac_cross_store_comparability_001.md` | Demo 2 cross-store comparability boundary review. |
| `rac/src/grounded_pipeline.py` | Deterministic grounded review pipeline. |

## Region and Market-Context Boundary

`region_type` follows the central definition in `retail_ops/data/DATA_DICTIONARY.md`. In this project it is weak market-context metadata for the current demo, not a peer-grouping field, market-area label, consumption-level label, or store-stage label.

Future market-context classification should be added only after broader store coverage and supporting evidence are available. Useful future evidence may include repeated store-period records, local consumption-level evidence, competition evidence, delivery-radius evidence, activity-condition evidence, and stronger geographic context.

## Field Boundary Summary

These are the main fields that are easy to misread in application review. Full definitions remain in `retail_ops/data/DATA_DICTIONARY.md`.

| Field | Boundary |
|---|---|
| `region_type` | Weak region or market-context evidence only. It is not a peer-grouping field, market-area classification, consumption-level label, or store-stage label. |
| `activity_cost_ratio_pct` | Activity cost divided by activity original transaction amount. It is not traditional ROI, profit margin, or audited return. |
| `comparison_scope_flag` | Row-level diagnostic-readiness flag for the current Demo 2 evidence layer. It is not a pairwise comparability decision. |
| `comparison_limit_notes` | Interpretation-limit notes that document how far the current diagnostic evidence can be used. |
| `estimated_income_proxy` | Platform-displayed estimated income proxy. It is retained as weak backend-reported context only and should not be used as audited profit, settlement evidence, ROI, or a primary comparability factor. |
| `transaction_amount` | Transaction amount / transaction scale in the current implemented evidence. It should not be renamed or treated as `gross_revenue` unless a future backend export provides that separate field under the dictionary definition. |
| `top3_sku_transaction_amount_share_pct` | Lightweight top-SKU concentration evidence from selected ranking views. It is not full product-category sales share. |

## Current Evidence Boundary

The current repository evidence covers six selected stores and selected reporting windows. It supports reproducible descriptive diagnostics, field-contract checks, lineage-aware memory facts, and answer-boundary evaluation.

Broader store ranking, promotion transfer, market-area classification, and causal attribution require additional store-period coverage and operating context beyond the current evidence.
