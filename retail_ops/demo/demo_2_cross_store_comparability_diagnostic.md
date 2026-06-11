# Demo 2: Same-Period B-F Store Diagnostic

## What a Reviewer Should Conclude from Demo 2

Demo 2 supports same-period diagnostic reading across selected Stores B-F for March 2026.


It exposes same-period store-period diagnostic evidence and records the interpretation limits that should be carried into future pairwise comparison.

The future pairwise comparability gate remains separate from Demo 2.


## Naming Note

This file keeps the historical `cross_store_comparability` path wording for reference stability. In the current implementation, Demo 2 means same-period B-F diagnostic evidence and interpretation guardrails, not a completed pairwise comparability gate.

<!-- stable-demo2-scope-boundary -->

## Scope Boundary

Demo 2 answers a narrow implementation question: can selected Stores B-F be organized under one March 2026 reporting window and one field contract without losing the original Meituan metric meanings?


`comparison_scope_flag` is a row-level diagnostic-scope field. It means that a store-period row is structurally usable for the current Demo 2 diagnostic. It is not a pairwise store-matching decision.


## Purpose

This demo tests whether five anonymized instant-retail store records can be placed under the same reporting window and field contract before making any stronger operating interpretation. The purpose is to structure selected backend metrics into a same-period B-F diagnostic format and record the limits that should constrain later comparison.

## Business Problem

Meituan's merchant backend provides detailed store-level metrics, but the backend is mainly designed for reviewing one store at a time.

With many stores, the harder problem is deciding which stores can be compared, under what conditions they can be compared, and which signals are strong enough to support cautious operating judgment.

In this project, instant-retail competition is understood through this operating chain:

```text
being seen -> being entered -> being ordered -> being selected again or maintaining share
```

Promotion, subsidy, price, SKU mix, ranking position, and fulfillment conditions are operating levers inside this chain. They should be interpreted through the store's current operating state and comparison limits, not as isolated goals.

## Scope

| Item | Value |
|---|---|
| Stores | B, C, D, E, F |
| Reporting window | 2026-03-01 to 2026-03-31 |
| Period label | 2026-03 |
| Source | Manually structured Meituan merchant-backend metrics for instant-retail store operations. |
| Processing method | Offline SQL diagnostic. |
| SQL file | `retail_ops/sql/02_demo2_cross_store_comparability.sql` |
| SQL output | `retail_ops/outputs/demo2_cross_store_comparability_output.csv` |
| Memory facts | `retail_ops/outputs/generated_demo2_retail_memory_facts.json` |

Some source traffic-channel fields are retained in the structured source file but not carried into the current Demo 2 diagnostic output. Demo 2 focuses on selected same-period diagnostic signals rather than exhaustive traffic-source decomposition.

The top-SKU source tables are intentionally partial: one view is ranked by sales volume and the other by transaction amount. The available backend evidence for this demo did not provide both `sales_volume` and `sku_transaction_amount` for every ranking view, and missing values are not back-calculated because SKU prices can vary by store, operating stage, activity condition, and local competitive pressure.

`region_type` is kept as weak regional context only. In the current sample, it should not be read as a mature market-area classification or as a hard comparability condition.

A future market-area field would require broader store coverage and external or data-supported evidence such as local consumption level, competitive density, price pressure, and SKU demand structure.

## Demo 2 Evidence Path

| Step | Artifact | Role |
|---|---|---|
| Source store-period metrics | `retail_ops/data/demo2_store_period_metrics.csv` | Stores selected Meituan backend metrics under canonical field names. |
| Source search and SKU tables | `retail_ops/data/demo2_top_search_terms.csv`, `retail_ops/data/demo2_top_skus_by_transaction_amount.csv`, `retail_ops/data/demo2_top_skus_by_sales_volume.csv` | Keeps search-term and selected SKU evidence separate from store-period totals. |
| Memory facts | `retail_ops/outputs/generated_demo2_retail_memory_facts.json` | Converts diagnostic evidence into source-bounded facts with observed values and limitations. |
| Boundary checks | Demo 2 validation and evaluation scripts | Checks that answers stay within the implemented evidence instead of treating the output as a pairwise comparability gate. |

## What the SQL Checks

The SQL prepares a same-period diagnostic output with:

- period alignment;
- region and store-type context;
- exposure, ranking, entry, and search-entry structure;
- activity-order share and activity-cost structure;
- top-3 SKU transaction-amount concentration;
- comparison-scope and comparison-limit notes.

## Key Diagnostic Fields

### `comparison_scope_flag`

This field records whether the row is inside the current Demo 2 comparison scope.

In the current Demo 2 output, all B-F stores use the same March 2026 reporting window and are marked:

  same_period_diagnostic_ready

This means the rows are ready for the current same-period diagnostic. It does not mean the stores are fully comparable in every business sense.

### `comparison_limit_notes`

This field records the main reasons why direct cross-store interpretation should be constrained.

Examples include:

- high or moderate activity involvement;

- top-3 SKU transaction-amount concentration;
- the need to compare with region, store type, activity, product-mix limits.

These notes are interpretation guardrails.

The threshold-based guardrail labels in this demo are prototype diagnostic warnings. They help expose which records need interpretation limits; they are not optimized business cutoffs, peer-selection rules, or pairwise comparability decisions.

A guardrail-sensitivity check is included for this reason: in the current B-F sample, all five store rows change `comparison_limit_notes` under at least one plus-or-minus 5 percentage-point threshold scenario. This makes the current thresholds useful as diagnostic warnings, but not stable enough to become peer-comparison rules or strategy-transfer rules.

## What This Demo Supports

This demo supports same-period diagnostic reading.


## Current Interpretation Boundary

Demo 2 supports same-period diagnostic reading, but it should not be used for three stronger decisions:

1. ranking stores as best or worst;
2. transferring subsidy, pricing, SKU, ranking, or fulfillment actions from one store to another;
3. generalizing the B-F March 2026 sample into a full 48-store decision system.

Profit analysis, complete SKU category-share analysis, automated backend ingestion, causal attribution, and pairwise comparability gating require additional evidence beyond the current Demo 2 scope.

## Why This Matters for the Memory Layer

The memory layer should not answer cross-store questions by retrieving isolated metrics.

It should preserve each store's period, evidence, comparison scope, and interpretation limits.

For this reason, Demo 2 converts SQL diagnostics into generated retail memory facts using the existing retail slots:

- `visibility_entry_profile`
- `activity_lever_profile`
- `transaction_conversion_profile`
- `top3_sku_product_mix_note`
- `single_metric_attribution_guard`

The memory facts are currently file-backed for Demo 2. This is enough to test the data contract, SQL diagnostic output, fact generation, and limitation-preserving answer behavior, but it is not yet a full 48-store decision-support system.

## What the Current Demo 2 Output Shows

### Selected Diagnostic Signals

The table below shows selected numeric signals from the current Demo 2 SQL output. It is ordered by anonymized store ID, not by performance rank.

| Store | transaction_orders | transaction_amount | search_exposure_users | search_average_rank | search_entry_users | search_entry_rate_pct | search_entry_share_pct | entry_conversion_rate_pct | order_conversion_rate_pct | activity_order_share_pct | activity_cost_ratio_pct | top3_sku_transaction_amount_share_pct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 299 | 11665.5 | 4390 | 15 | 683 | 15.56 | 88.7 | 15.12 | 36.88 | 88.63 | 10.76 | 11.15 |
| C | 175 | 7064.09 | 2416 | 13 | 355 | 14.69 | 68.8 | 17.3 | 32.75 | 70.86 | 7.5 | 28.38 |
| D | 404 | 18078.7 | 2663 | 13 | 738 | 27.71 | 87.65 | 13.88 | 46.56 | 83.42 | 8.95 | 16.9 |
| E | 158 | 5784.87 | 2784 | 13 | 355 | 12.75 | 87.01 | 12.19 | 39.46 | 68.99 | 19.1 | 12.55 |
| F | 266 | 9301.8 | 2699 | 13 | 481 | 17.82 | 88.91 | 15.27 | 49.72 | 81.58 | 10.62 | 19.33 |

Search-entry structure should be read as a group: `search_exposure_users`, `search_average_rank`, `search_entry_users`, `search_entry_rate_pct`, and `search_entry_share_pct` describe different parts of visibility and entry.

`search_entry_share_pct` describes source mix among entry users. `search_entry_rate_pct`, `entry_conversion_rate_pct`, and `order_conversion_rate_pct` keep the interpretation tied to exposure, entry, and order conversion instead of treating one share metric as search performance by itself.

These values are diagnostic signals used to explain comparison limits. They are not a store ranking, a profit table, or a pairwise comparability decision.

The current output should be read as row-level diagnostic evidence, not as a pairwise store-comparability decision.

The saved output keeps `comparison_limit_notes` for traceability, but this demo does not use those notes to classify or rank stores. The selected table above is limited to numeric operating signals and keeps interpretation tied to visibility, entry, order conversion, activity context, and product-mix context.

## Derived-Metric Scope Note

Demo 2 intentionally keeps the same-period B-F diagnostic output narrower than Demo 1.

Demo 1 is a month-over-month diagnostic for one store, so it includes more month-level derived indicators. Demo 2 is a same-period B-F diagnostic, so it focuses on field-contract consistency, selected diagnostic evidence, and comparison-boundary behavior.

For that reason, Demo 2 does not expand every derived field defined in `retail_ops/data/DATA_DICTIONARY.md`.
