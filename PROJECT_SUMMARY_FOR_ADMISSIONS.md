# Project Summary for Admissions Review

## Project Title

Meituan Instant-Retail Decision Support Prototype

Repository: `livestream-agent-memory-layer`

## Project Summary

This staged local prototype grew from a real operating problem in a 48-store Meituan instant-retail business. The Meituan merchant backend contains detailed single-store metrics, but it is mainly designed for reviewing one store at a time. As the store count increased, the harder question became how to compare store-period records without flattening different reporting windows, activity conditions, product structures, store types, and market contexts into the same judgment.

The repository turns selected backend observations into a traceable decision-support prototype. Metric definitions are fixed in the data dictionary, diagnostic calculations are reproduced in SQL, derived findings retain their source lineage, and later retrieval and review steps preserve the boundaries attached to the evidence.

## Reviewer Reading Path

This file is the single application-facing entry point. Continue through
the project in this five-step order:

| Step | File | What to inspect |
|---:|---|---|
| 1 | [Project summary](PROJECT_SUMMARY_FOR_ADMISSIONS.md) | Business origin, evidence scope, architecture, implemented work, and decision boundary. |
| 2 | [Design evolution](case_studies/from_livestream_to_retail_decision_support.md) | The progression from livestream product memory to lifecycle-aware retail evidence. |
| 3 | [Demo 1: Store A month-over-month diagnostic](retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md) | Store A repeated-window evidence and multi-metric interpretation. |
| 4 | [Demo 2: same-period B-F diagnostic](retail_ops/demo/demo_2_cross_store_comparability_diagnostic.md) | Multi-store evidence structure and comparison guardrails. |
| 5 | [RAC grounded-review demo index](rac/DEMO_INDEX.md) | Factor-aware evidence routing, record grounding, unresolved requirements, and report-contract validation. |

After this first pass, use:

- [Repeated-window B-F coverage](retail_ops/outputs/store_period_panel_coverage_output.csv)
  and [summary](retail_ops/outputs/repeated_window_panel_summary_output.csv)
  for the February-April supporting panel;
- [experiment results](retail_ops/EXPERIMENT_RESULTS.md) for procedures,
  outcomes, retrieval stress cases, and failure modes;
- the [data dictionary](retail_ops/data/DATA_DICTIONARY.md) and
  [technical appendix](retail_ops/TECHNICAL_APPENDIX.md) for field and
  source-to-claim review;
- [Comparability Gate V0](retail_ops/COMPARABILITY_GATE_V0.md) for the
  deferred question-specific pairwise evidence contract.

## Business Decision Problem

For standardized products such as contact lenses, care solutions, and related eye-care items, store performance develops through a connected operating path:

```text
being seen
-> being entered
-> being ordered
-> being selected again / maintaining share
```

This chain is the business framing for the decision problem. Monthly transaction records provide the continuous sales outcome across reporting periods, while `maintaining share` is not treated as a separately measured result.

Promotion, subsidy, pricing, SKU arrangement, ranking work, fulfillment support, and local competition provide operating context for different parts of this path. Their interpretation depends on the store, reporting period, activity conditions, product structure, and evidence available for the question.

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

The repository supports analysis of these selected store-period records. Question-specific pairwise comparability requires a separate analytical step beyond the current same-period diagnostic because matching reporting windows alone does not establish that two stores are suitable peers for a particular operating decision.

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
| Generated memory facts | Retains structured findings together with source fields, source paths, observed values, calculation metadata, evidence-trace confidence labels, and limitations. |
| Contract and lineage checks | Detects selected naming, formula, header, metadata, path, and source-to-output inconsistencies. |
| Retrieval stress tests | Examines evidence routing under supported, unsupported, hard-negative, mismatched, ambiguous, and wording-variation queries. |
| Boundary evaluations | Checks whether later answers preserve entity, period, metric-definition, and comparison limits. |
| RAC grounded review | Decomposes multi-factor questions, routes local or boundary evidence, develops competing hypotheses, applies critique and rule-based checks for unsupported claims and definition conflicts, updates evidence-coverage state, and produces an inspectable report. |

## Evaluation Logic

The evaluation is organized around reproducibility, boundary
preservation, retrieval failure modes, and grounded review.

The current results are:

| Check | Observed result | How to read it |
|---|---|---|
| Store A value lineage | The check covers 3 source rows, 3 SQL output rows, 9 top-SKU rows, 180 source, formula, movement, ranking, and trade-off comparisons, and 5 generated facts. | The lineage from source tables to SQL outputs and generated facts can be checked field by field, while the operating result remains multi-metric rather than a single-cause conclusion. |
| Demo 2 guardrail sensitivity | Baseline notes reproduce for all 5 rows. Raising the current thresholds by 5 percentage points changes Stores C-F; lowering them by 5 percentage points changes no rows. | The thresholds are prototype diagnostic warnings rather than optimized peer-selection rules. |
| Retrieval wording stress | Supported variants retain expected evidence in 34/34 cases. Hard-negative, entity/period-mismatch, and ambiguous variants cross the `0.5720` reference threshold in 23/33, 15/18, and 5/16 cases. | Semantic similarity helps route evidence but does not establish entity, period, or decision-scope support. |
| Repeated-window B-F panel | Stores B-F each retain February-April 2026 coverage across 11 selected metrics. | The panel supports descriptive review and later rule preparation, not a completed pairwise decision or monthly guardrail-stability result. |

These are descriptive analyses, retrieval stress tests, and contract
checks with different meanings. They are not combined into one accuracy
score. Difficult retrieval cases remain visible because they show where
semantic similarity is insufficient on its own.

## Implemented Retail Path

### Demo 1: Store A Month-over-Month Diagnostic

Demo 1 analyzes Store A across February, March, and April 2026. It shows why one metric alone is not enough. The purpose is to preserve a careful operating profile, not to label a month as simply good or bad.

Main file:

- `retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md`

### Demo 2: Same-Period B-F Diagnostic

Demo 2 extends the analysis to selected Stores B-F under the same March 2026 reporting window. Current SKU evidence uses selected top-SKU ranking views rather than full catalogue-level product-category classification.

Main file:

- `retail_ops/demo/demo_2_cross_store_comparability_diagnostic.md`

A guardrail-sensitivity check is included because the current Demo 2 thresholds are prototype interpretation thresholds, not optimized business cutoffs. In the current B-F sample, four of five rows change when the existing SQL thresholds are raised by 5 percentage points, while the easier-to-trigger scenario produces no note changes. This is a small-sample sensitivity result, so the thresholds remain diagnostic warnings rather than optimized peer-comparison rules.

A query-robustness inspection maps how supported, unsupported, hard-negative, entity or period mismatch, and ambiguous comparison queries respond to small wording variations. The mismatch and hard-negative cases are kept visible because some variants still cross the exploratory similarity threshold.

### Repeated-Window B-F Panel

The repeated-window panel adds selected Stores B-F across 2026-02, 2026-03, and 2026-04, making repeated store-period coverage visible for later question-specific comparability checks.

The panel verifies three-month coverage and places February, March, and April values side by side for selected fields. The existing February-to-April delta fields remain endpoint summaries; March is retained so the middle month is not hidden. The panel prepares the evidence base for future comparison rules. Testing monthly guardrail stability will require repeated top-SKU evidence and monthly recomputation of `comparison_limit_notes`.

Main outputs:

- `retail_ops/outputs/store_period_panel_coverage_output.csv`
- `retail_ops/outputs/repeated_window_panel_summary_output.csv`

## Factor-Aware Grounded Review Layer

RAC is a deterministic source-aware review layer over local project evidence.

The layer decomposes an operating question into relevant factors, routes each factor to local evidence or boundary evidence, generates competing hypotheses, applies critique and rule-based checks for unsupported claims and definition conflicts, and produces a grounded report with scenario-template confidence labels, limitations, source paths, structured-record locators for CSV evidence, source-line pointers for text evidence, and canonical evidence fields.

This layer is useful when current evidence is strong for one factor but weak, missing, or boundary-only for another. It helps prevent a grounded answer from hiding missing evidence behind a fluent conclusion.

Reviewer entry points:

| File | Purpose |
|---|---|
| [RAC demo index](rac/DEMO_INDEX.md) | Reviewer-facing index of deterministic RAC cases. |
| [Store A attribution-boundary report](rac/outputs/grounded_rac_store_a_attribution_001.md) | Shows deterministic CSV record grounding and multi-factor attribution limits. |
| [Cross-store comparability-boundary report](rac/outputs/grounded_rac_cross_store_comparability_001.md) | Shows quantitative evidence routing and explicit unavailable requirements. |
| [Promotion-review report](rac/outputs/grounded_rac_promotion_strategy_001.md) | Separates available cost, subsidy, and conversion evidence from unresolved decision requirements. |
| [Grounded pipeline](rac/src/grounded_pipeline.py) | Deterministic review and report-generation implementation. |

## Deferred Comparability Contract

`retail_ops/COMPARABILITY_GATE_V0.md` records the additional evidence requirements for any future question-specific pairwise decision. The current prototype ends at same-period diagnostic review with explicit comparison limits; aligned reporting windows alone do not establish peer-store suitability or support strategy-transfer decisions.

## Field Boundary Summary

These are the main fields that are easy to misread in application review. Full definitions remain in `retail_ops/data/DATA_DICTIONARY.md`.

| Field | Boundary |
|---|---|
| `region_type` | Weak region or market-context evidence only. It is not a peer-grouping field, market-area classification, consumption-level label, or store-stage label. |
| `activity_cost_ratio_pct` | Activity cost divided by activity original transaction amount. It is not traditional ROI, profit margin, or audited return. |
| `comparison_scope_flag` | Row-level diagnostic-readiness flag for the current Demo 2 evidence layer. It is not a pairwise comparability decision. |
| `comparison_limit_notes` | Interpretation-limit notes that document how far the current diagnostic evidence can be used. |
| `transaction_amount` | Transaction amount / transaction scale in the current implemented evidence. It should not be renamed or treated as `gross_revenue` unless a future backend export provides that separate field under the dictionary definition. |
| `top3_sku_transaction_amount_share_pct` | Lightweight top-SKU concentration evidence from selected ranking views. It is not full product-category sales share. |

## Evidence Boundary

The repository evidence covers six selected stores and the documented
reporting windows. It supports reproducible descriptive diagnostics,
field-contract checks, lineage-aware memory facts, retrieval stress
tests, answer-boundary evaluation, and deterministic grounded review.

Broader store ranking, strategy transfer, market-area classification,
and causal attribution remain outside scope. The monthly transaction
series is treated as a continuous sales outcome rather than a separate
measure of the later-stage business objective.
