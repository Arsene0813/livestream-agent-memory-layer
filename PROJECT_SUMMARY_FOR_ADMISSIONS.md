# Project Summary for Admissions Review

## Project Title

Meituan Instant-Retail Decision Support Prototype

Repository: `livestream-agent-memory-layer`

## Project Summary

This staged local prototype grew from a real operating problem in a 48-store Meituan instant-retail business.

The Meituan merchant backend gives detailed single-store metrics, but it does not directly answer the harder multi-store question: which store-period records can be compared, under what operating conditions, and what kind of operating judgment the available evidence can support.

The current prototype organizes selected backend evidence into this path:

1. Meituan backend metric evidence
2. `DATA_DICTIONARY.md` field contract
3. SQL diagnostic output
4. generated retail memory facts
5. boundary-preserving answer checks
6. deterministic source-aware review scaffold

The current retail path includes Demo 1, Demo 2, and a post-Demo2 repeated-window panel extension for Stores B-F across February-April 2026. Demo 2 structures selected store-period evidence under one reporting window and one field contract; the repeated-window panel checks whether same-store multi-month evidence exists before any stronger pairwise comparability decision is attempted.

A future pairwise comparability gate should judge whether two store-period records can be compared for a specific operating question. It should not produce a global store ranking or universal comparability score.

## Review Note

For admissions review, start with:

1. `README.md` - admissions review path and current implementation boundary.
2. `PROJECT_SUMMARY_FOR_ADMISSIONS.md` - short application-facing project summary.
3. `retail_ops/data/DATA_DICTIONARY.md` - authoritative field names and metric definitions.
4. `retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md` - current single-store month-over-month diagnostic.
5. `retail_ops/demo/demo_2_cross_store_comparability_diagnostic.md` - current same-period B-F diagnostic.
6. `retail_ops/EXPERIMENT_RESULTS.md` - implemented checks and validation outcomes.
7. `retail_ops/COMPARABILITY_GATE_V0.md` - future pairwise comparability-gate contract.

## Operating Step Summary

| Operating step | Practical meaning |
|---|---|
| Backend metric evidence | Real Meituan merchant-backend values are the starting point rather than invented examples. |
| Metric dictionary | Backend metric meanings are preserved under canonical project field names. |
| SQL diagnostics | Selected store-period records are structured into diagnostic outputs under one reporting window and one field contract. |
| Generated memory facts | Diagnostic evidence is stored with source fields, observed values, confidence labels, and limitations. |
| Boundary checks | Later answers are checked against entity scope, period scope, metric definitions, and comparison limits. |
| Future comparability gate | Planned pairwise logic will decide whether two store-period records can be compared for a specific operating question. |

## Business Problem

In Meituan instant retail, store competition is not only about having products online. For standardized products such as contact lenses, care solutions, and related eye-care items, stores compete through a local operating chain: being seen, being entered, being ordered, and then being selected again or maintaining share.

Promotion, subsidy, price adjustment, SKU arrangement, ranking optimization, and fulfillment stability are tools inside this chain. Their meaning depends on store state, local competition, activity intensity, refund pressure, product mix, and reporting-window alignment.

A new store may use activity support to gain first exposure and first orders. A store under local price pressure may use pricing or subsidy tools to defend visibility and market share. A store with high exposure may still underperform if entry quality, order conversion, refund pressure, or SKU concentration create friction.

The practical question is therefore not whether one store looks better in isolation. The useful question is whether the available evidence is strong enough to compare records and transfer any operating interpretation.

## Why This Became a Data-Science Problem

The backend data is detailed, but it is mainly organized for single-store monitoring. In a 48-store operation, detailed single-store metrics do not automatically become reliable cross-store decision evidence.

A cross-store decision should preserve metric definitions, align reporting windows, expose comparison limits, and separate observed evidence from unsupported strategy transfer.

The current prototype treats SQL as the structuring layer and the memory layer as the evidence-retention layer. SQL organizes selected backend metrics into diagnostic outputs. Generated memory facts then preserve the store, period, source fields, observed values, calculation notes, confidence labels, and limitations so that later retrieval or answer generation does not lose the boundary around the evidence.

## Current Prototype Layers

| Layer | What it does |
|---|---|
| Metric dictionary | Preserves Meituan backend metric meanings and canonical project field names. |
| SQL diagnostic layer | Converts selected store-period backend data into diagnostic outputs under one field contract. |
| Generated memory facts | Stores evidence with source paths, source fields, observed values, calculation notes, confidence labels, and limitations. |
| Boundary evaluations | Checks whether later answers preserve entity scope, period scope, metric definitions, and comparison limits. |
| Future comparability gate | Planned pairwise gate for deciding whether two store-period records can be compared for a specific operating question. |

The memory layer does not replace the data. It records evidence and limitations so that different stores, periods, activity conditions, product structures, and regional contexts are not collapsed into the same comparison.

## Current Implemented Retail Path

### Demo 1: Store A Month-over-Month Diagnostic

Demo 1 analyzes Store A across February, March, and April 2026. It shows why one metric alone is not enough.

April 2026 showed recovery in traffic and transaction scale, while order conversion and average order value declined. Refund pressure improved at the same time.

The purpose is to preserve a careful operating profile, not to label a month as simply good or bad.

Main file:

- `retail_ops/demo/demo_1_store_a_month_over_month_diagnostic.md`

### Demo 2: Same-Period B-F Diagnostic

Demo 2 extends the analysis to selected Stores B-F under the same March 2026 reporting window.

The point is to place selected store-period fields under the same reporting window and field contract before any stronger operating interpretation is made. Store type, order volume, transaction amount, activity involvement, activity intensity, search-entry structure, refund pressure, and top-SKU evidence can all affect what a cross-store comparison means.

Current SKU evidence uses selected top-SKU ranking views rather than full catalogue-level product-category classification.

Main file:

- `retail_ops/demo/demo_2_cross_store_comparability_diagnostic.md`

A guardrail-sensitivity check is included because the current Demo 2 thresholds are prototype interpretation thresholds, not optimized business cutoffs. In the current B-F sample, guardrail notes change under plus or minus 5 percentage-point threshold shifts, so these thresholds are treated as diagnostic warnings rather than stable peer-comparison rules.

A query-robustness inspection is also included to test whether supported, unsupported, hard-negative, entity or period mismatch, and ambiguous comparison queries behave consistently under small wording variations.

## Region and Market-Context Boundary

`region_type` follows the central definition in `retail_ops/data/DATA_DICTIONARY.md`.

In this project it is weak market-context metadata for the current demo, not a peer-grouping field or market-area label.

Future market-context classification should be added only after broader store coverage and supporting evidence are available. Useful future evidence may include repeated store-period records, local consumption-level evidence, competition evidence, delivery-radius evidence, activity-condition evidence, and stronger geographic context.

## Field Boundary Summary

These are the main fields that are easy to misread in application review. Full definitions remain in `retail_ops/data/DATA_DICTIONARY.md`.

| Field | Boundary |
|---|---|
| `region_type` | Weak region or market-context evidence only. It is not a peer-grouping field, market-area classification, consumption-level label, or store-stage label. |
| `activity_cost_ratio_pct` | Activity cost divided by activity original transaction amount. It is not traditional ROI, profit margin, or audited return. |
| `comparison_scope_flag` | Row-level diagnostic-readiness flag for the current Demo 2 evidence layer. It is not a pairwise comparability decision. |
| `comparison_limit_notes` | Interpretation-limit notes for current diagnostic evidence. They are not optimized business cutoffs, peer-selection rules, or strategy-transfer approvals. |
| `top3_sku_transaction_amount_share_pct` | Lightweight top-SKU concentration evidence from selected ranking views. It is not full product-category sales share. |

## Factor-Aware Grounded Review Scaffold

The project also includes a deterministic source-aware review scaffold over local project evidence.

This layer decomposes an operating question into relevant factors, routes each factor to local evidence or boundary evidence, generates competing hypotheses, applies critique and fact checks, and produces a grounded report with confidence, limitations, source paths, and local evidence snippets.

The value of this layer is reviewability. It makes the reasoning path inspectable before an answer is written, especially when current evidence is strong for one factor but weak or missing for another.

Reviewer entry points:

| File | Purpose |
|---|---|
| `rac/DEMO_INDEX.md` | Index of deterministic RAC demo cases. |
| `rac/outputs/grounded_rac_store_a_attribution_001.md` | Store A attribution-boundary review. |
| `rac/outputs/grounded_rac_cross_store_comparability_001.md` | Demo 2 cross-store comparability boundary review. |
| `rac/src/grounded_pipeline.py` | Deterministic grounded review pipeline. |
