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

The repository turns selected backend observations into a traceable decision-support prototype. Metric definitions are fixed in the data dictionary, diagnostic calculations are reproduced in SQL, derived findings retain their source lineage, and later retrieval and review steps preserve the boundaries attached to the evidence.

## Business Decision Problem

For standardized products such as contact lenses, care solutions, and related eye-care items, store performance develops through a connected operating path:

```text
being seen
-> being entered
-> being ordered
-> being selected again or maintaining share
```

Promotion, subsidy, pricing, SKU arrangement, ranking work, fulfillment support, and local competition can affect different parts of this path. Their interpretation depends on the store, reporting period, activity conditions, product structure, and evidence available for the question.

The central decision problem is therefore not simply which store or metric is highest. It is whether an operating interpretation can be reproduced from documented fields, matched to the correct entity and reporting period, and kept within the limits of the available evidence.

## Evidence Scope

| Scope | Evidence represented in the repository |
| --- | --- |
| Business setting | The operating problem emerged in a 48-store Meituan instant-retail business. |
| Repository evidence | Manually transcribed and anonymized observations for six selected stores: Store A and Stores B-F. |
| Demo 1 | Store A across February, March, and April 2026. |
| Demo 2 | Stores B-F under one March 2026 reporting window. |
| Repeated-window panel | Stores B-F across February, March, and April 2026 using the documented supporting-table coverage. |
| Current analytical use | Descriptive diagnosis, value lineage, evidence routing, scope checks, and answer-boundary evaluation. |

The repository evidence supports analysis of these selected store-period records. Question-specific pairwise comparability remains a separate analytical step because matching reporting windows alone does not establish that two stores are suitable peers for a particular operating decision.

## Prototype Design

### Evidence path

```text
selected Meituan backend metrics
-> DATA_DICTIONARY.md field contract
-> reproducible SQL diagnostics
-> generated retail memory facts
-> contract and boundary evaluations
-> retrieval stress tests
-> RAC grounded review over local evidence
```

`retail_ops/data/DATA_DICTIONARY.md` is the authoritative source for field names, Chinese metric definitions, formulas, grains, and interpretation boundaries.

SQL provides the calculation and structuring layer. It converts selected source records into inspectable diagnostic outputs while preserving the documented meanings of the backend metrics.

Generated memory facts retain the entity, reporting period, evidence slot, source fields, observed values, calculation metadata, source paths, confidence label, and limitations. In this project, the confidence label describes evidence traceability and definition coverage. It is not a probability that an operating conclusion is correct.

### Layer responsibilities

| Implemented layer | Role in the prototype |
| --- | --- |
| Metric dictionary | Preserves canonical project field names and the Chinese definitions of the Meituan backend metrics. |
| SQL diagnostics | Reproduces selected diagnostic values and keeps transformations inspectable. |
| Generated memory facts | Retains structured findings together with source fields, source paths, observed values, calculation metadata, confidence labels, and limitations. |
| Contract and lineage checks | Detects selected naming, formula, header, metadata, path, and source-to-output inconsistencies. |
| Retrieval stress tests | Examines evidence routing under supported, unsupported, hard-negative, mismatched, ambiguous, and wording-variation queries. |
| Boundary evaluations | Checks whether later answers preserve entity, period, metric-definition, and comparison limits. |
| RAC grounded review | Decomposes multi-factor questions, routes local or boundary evidence, develops competing hypotheses, applies critique and rule-based claim and definition checks, updates evidence-coverage state, and produces an inspectable report. |

The layers have distinct responsibilities. SQL establishes reproducible diagnostic values. Memory facts preserve evidence after transformation. Retrieval identifies potentially relevant local evidence. Boundary evaluations and RAC review examine whether later reasoning remains connected to the documented source and scope.

## Evaluation Logic

The evaluation layer checks declared field contracts, value lineage, entity and reporting-period scope, answer boundaries, endpoint behavior, retrieval score distributions, wording variation, and threshold sensitivity.

Repository pass counts refer to defined contract, fixture, lineage, and endpoint checks. Retrieval experiments are interpreted through score distributions and failure cases rather than as a general accuracy measure.

The current results support three narrower conclusions:

1. selected diagnostic values can be reproduced from the committed sample data;
2. generated findings can be traced to documented fields and source paths;
3. retrieval and review behavior can be inspected under both supported and difficult queries.

Hard-negative, entity-mismatch, period-mismatch, and ambiguous-query results remain part of the evidence because they show where semantic similarity is insufficient on its own.

## Next Analytical Experiment

The next analytical experiment is a question-specific pairwise comparability gate. Its purpose is to evaluate whether two store-period records are suitable for one defined operating question before cross-store interpretation is attempted.

Potential gate inputs include:

- reporting-window alignment;
- transaction order volume and transaction amount;
- activity involvement and activity intensity;
- store type;
- repeated-window stability;
- weak region context;
- competition evidence;
- SKU structure;
- fulfillment and stockout context.

The gate design is documented in `retail_ops/COMPARABILITY_GATE_V0.md`. It remains separate from the implemented same-period diagnostic so that future comparison rules can be tested explicitly.

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

The layer decomposes an operating question into relevant factors, routes each factor to local evidence or boundary evidence, generates competing hypotheses, applies critique and rule-based claim and definition checks, and produces a grounded report with template confidence labels, limitations, source paths, source-line audit pointers, and evidence fields.

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
