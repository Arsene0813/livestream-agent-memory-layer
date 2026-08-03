# Demo 2: Same-Period B-F Store Diagnostic

## Demo Result

Demo 2 organizes selected Stores B-F under the March 2026 reporting window and the canonical field contract. The output combines row-level diagnostic evidence with `comparison_scope_flag` and `comparison_limit_notes`, while retaining each record's period, source, and diagnostic context.

## Naming Note

This file keeps the historical `cross_store_comparability` path wording for reference stability. Demo 2 provides a same-period B-F diagnostic under the shared field contract.

<!-- stable-demo2-scope-boundary -->

## Diagnostic Contract

Demo 2 structures five anonymized store records under one March 2026 reporting window and one field contract while preserving the original Meituan metric meanings.

`comparison_scope_flag` records row-level readiness for this diagnostic. `comparison_limit_notes` records the diagnostic notes attached to each row.

## Business Problem

Meituan's merchant backend provides detailed store-level metrics.

With many stores, the analytical task becomes organizing store-period records under consistent reporting windows, field definitions, and operating context.

In this project, instant-retail competition is understood through this operating chain:

```text
being seen -> being entered -> being ordered -> being selected again / maintaining share
```

Promotion, subsidy, price, SKU mix, ranking position, and fulfillment conditions are operating levers inside this chain and are interpreted through the observed store-period context.

## Evidence Coverage

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

The structured source file retains the available traffic-channel fields. The Demo 2 output uses the selected same-period fields documented in the diagnostic contract.

The two top-SKU source tables preserve different backend ranking views: one by sales volume and one by transaction amount. Each table retains the values available in its source.

`region_type` retains the coarse regional context available in the source data and is reviewed together with the other store-period fields.

## Demo 2 Evidence Path

| Step | Artifact | Role |
|---|---|---|
| Source store-period metrics | `retail_ops/data/demo2_store_period_metrics.csv` | Stores selected Meituan backend metrics under canonical field names. |
| Source search and SKU tables | `retail_ops/data/demo2_top_search_terms.csv`, `retail_ops/data/demo2_top_skus_by_transaction_amount.csv`, `retail_ops/data/demo2_top_skus_by_sales_volume.csv` | Keeps search-term and selected SKU evidence separate from store-period totals. |
| Memory facts | `retail_ops/outputs/generated_demo2_retail_memory_facts.json` | Converts diagnostic evidence into source-linked facts with observed values and diagnostic context. |
| Validation checks | Demo 2 validation and evaluation scripts | Checks entity, period, canonical field, formula, source, and response consistency. |

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

This field records whether the row satisfies the current Demo 2 diagnostic conditions.

In the current Demo 2 output, all B-F stores use the same March 2026 reporting window and are marked:

  same_period_diagnostic_ready

For this fixture, the flag checks the exact March window plus non-missing `transaction_amount`, `transaction_orders`, `exposure_users`, `entry_users`, `search_exposure_users`, `search_entry_users`, `activity_orders`, and `top3_sku_transaction_amount`.

The core transaction, funnel, activity-involvement, and lightweight product-mix fields are available for the current row-level same-period diagnostic.

### `comparison_limit_notes`

This field records the diagnostic notes generated from the current thresholds.

Examples include:

- activity involvement;
- top-3 SKU transaction-amount concentration;
- review of region, store type, activity, and product-mix context.

A threshold-sensitivity check records how these notes respond to nearby settings. Baseline reproduction passes for all five rows. Raising the implemented thresholds by 5 percentage points changes `comparison_limit_notes` for Stores C, D, E, and F, while Store B remains unchanged. Lowering the thresholds by 5 percentage points changes no rows.

## What This Demo Supports

Demo 2 provides a same-period diagnostic view of Stores B-F across visibility, entry, transaction, conversion, activity, and listed-SKU evidence.

## Current Use

Each store profile remains linked to its March 2026 reporting period, canonical fields, diagnostic notes, and source records. The shared structure supports consistent review across the five selected stores.

## Why This Matters for the Memory Layer

The memory layer preserves each store's period, evidence slots, diagnostic context, and source references. The file-backed Demo 2 facts connect the data contract, SQL diagnostic output, fact generation, retrieval, and response checks in one traceable path.

## What the Current Demo 2 Output Shows

### Selected Diagnostic Signals

The table below shows selected numeric signals from the current Demo 2 SQL output. Rows follow anonymized store ID order. The generated CSV output remains the numeric source of truth for this table.

| Store | transaction_orders | transaction_amount | search_exposure_users | search_average_rank | search_entry_users | search_entry_rate_pct | search_entry_share_pct | entry_conversion_rate_pct | order_conversion_rate_pct | activity_order_share_pct | activity_cost_ratio_pct | top3_sku_transaction_amount_share_pct |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B | 299 | 11665.5 | 4390 | 15 | 683 | 15.56 | 88.7 | 15.12 | 36.88 | 88.63 | 24.12 | 11.15 |
| C | 175 | 7064.09 | 2416 | 13 | 355 | 14.69 | 68.8 | 17.3 | 32.75 | 70.86 | 9.45 | 28.38 |
| D | 404 | 18078.7 | 2663 | 13 | 738 | 27.71 | 87.65 | 13.88 | 46.56 | 83.42 | 14.92 | 16.9 |
| E | 158 | 5784.87 | 2784 | 13 | 355 | 12.75 | 87.01 | 12.19 | 39.46 | 68.99 | 29.38 | 12.55 |
| F | 266 | 9301.8 | 2699 | 13 | 481 | 17.82 | 88.91 | 15.27 | 49.72 | 81.58 | 12.16 | 19.33 |

The search-entry structure combines `search_exposure_users`, `search_average_rank`, `search_entry_users`, `search_entry_rate_pct`, and `search_entry_share_pct` across visibility and entry.

`search_entry_share_pct` describes source mix among entry users. `search_entry_rate_pct`, `entry_conversion_rate_pct`, and `order_conversion_rate_pct` are interpreted together across exposure, entry, and order conversion.

These values provide the numeric operating profile used by the Demo 2 diagnostic.

The saved output retains `comparison_limit_notes` for traceability. The selected table presents numeric operating signals across visibility, entry, order conversion, activity context, and product-mix context.

## Derived-Metric Design

Demo 2 uses a same-period B-F diagnostic design.

Demo 1 uses month-over-month derived indicators for one store. Demo 2 uses a shared same-period field set for Stores B-F, emphasizing field-contract consistency and selected diagnostic evidence.

The implemented Demo 2 field set is documented in the SQL output and `retail_ops/data/DATA_DICTIONARY.md`.
