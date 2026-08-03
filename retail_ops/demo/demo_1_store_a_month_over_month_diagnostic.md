# Demo 1: Store A Month-over-Month Retail Operations Diagnostic

## Purpose

This demo uses Store A as a narrow month-over-month diagnostic example. It shows how selected Meituan backend metrics can be structured into a field-consistent SQL output and then converted into traceable memory facts.

The current demo covers Store A from February 2026 to April 2026.

The month labels use natural calendar-month windows: `2026-02` means `2026-02-01` to `2026-02-28`, `2026-03` means `2026-03-01` to `2026-03-31`, and `2026-04` means `2026-04-01` to `2026-04-30`.

## Business Context

In this project, Meituan instant-retail stores are understood through a chain of operating conditions:

```text
being seen -> being entered -> being ordered -> being selected again / maintaining share
```

A store may use activity subsidy, pricing, ranking optimization, SKU mix, and fulfillment control as operating levers inside this chain.

## Evidence Coverage

This demo examines Store A's February-April 2026 movement across visibility, entry, transaction, conversion, activity, and listed top-SKU signals under one canonical field contract.

## Source Files

| File | Purpose |
|---|---|
| `retail_ops/data/store_a_monthly_metrics.csv` | Store-period metrics manually transcribed from anonymized Meituan merchant-backend evidence. |
| `retail_ops/data/store_a_top_skus.csv` | Top-SKU evidence for each month. |
| `retail_ops/data/DATA_DICTIONARY.md` | Canonical field definitions and metric rules. |
| `retail_ops/sql/01_store_a_month_over_month_diagnostic.sql` | Offline SQL diagnostic query. |
| `retail_ops/outputs/store_a_demo1_sql_output.csv` | SQL output used by this demo. |
| `retail_ops/outputs/generated_retail_memory_facts.json` | Generated retail memory facts for retrieval. |
| `retail_ops/TECHNICAL_APPENDIX.md` | Consolidated source-to-claim lineage and field-usage review. |

## Metric Definition Rules

This demo follows the project data dictionary.

Important consistency rules:

1. `order_conversion_rate_pct` follows the backend funnel definition:

   ```text
   order_conversion_rate_pct = order_users / entry_users * 100
   ```

   Backend refund fields are retained as separate source fields:

   ```text
   refund_amount
   full_refund_orders
   refund_orders_all_or_partial
   ```

2. `activity_cost_ratio_pct` follows the backend-style formula:

   ```text
   activity_cost_ratio_pct = activity_cost / activity_original_transaction_amount * 100
   ```

   The ratio follows the backend formula recorded in `DATA_DICTIONARY.md`.

3. Traffic-source users may overlap. Total exposure users and total entry users therefore use the backend total fields, while source-level user fields remain separate.

4. Top-SKU evidence provides product-mix context for the listed monthly ranking records.

## Store A Monthly Snapshot

| Month   | Transaction Amount | Transaction Orders | Entry Users | Order Conversion Rate | Average Order Value |
| ------- | -----------------: | -----------------: | ----------: | --------------------: | ------------------: |
| 2026-02 |            9460.70 |                274 |         763 |                37.22% |               34.53 |
| 2026-03 |            6454.84 |                207 |         522 |                42.34% |               31.18 |
| 2026-04 |            9083.72 |                337 |         906 |                37.42% |               26.95 |

## Visibility and Entry Profile

Store A's visibility and entry structure can be described from exposure, ranking, entry, and search-entry metrics.

| Month  | Exposure Users | Store Average Rank | Entry Users | Search Entry Users | Search Entry Share |
| ------- | -------------: | -----------------: | ----------: | -----------------: | -----------------: |
| 2026-02 |      6118 |         18 |     763 |        694 |       90.96% |
| 2026-03 |      4663 |         22 |     522 |        445 |       85.25% |
| 2026-04 |      8366 |         18 |     906 |        839 |       92.60% |

Interpretation:

- March had weaker exposure, weaker average rank, fewer entry users, and fewer search-entry users.
- April recovered in exposure, rank, entry users, and search-entry users.
- These metrics describe whether the store was being seen and entered.
- Together with transaction, conversion, activity, and SKU evidence, these metrics form the Store A operating profile.

## Activity-Lever Profile

Activity metrics are interpreted as operating-lever evidence.

| Month  | Activity Orders | Transaction Orders | Activity Order Share | Activity Cost Ratio |
| ------- | --------------: | -----------------: | -------------------: | ------------------: |
| 2026-02 |       270 |        274 |        98.54% |       40.63% |
| 2026-03 |       201 |        207 |        97.10% |       38.55% |
| 2026-04 |       329 |        337 |        97.63% |       40.69% |

Interpretation:

- Activity orders accounted for a high share of transaction orders in all three months.
- Activity-order structure shows high activity involvement during the observed period.
- The meaning of this activity involvement depends on operating context, competition, price pressure, and ranking pressure.

## Transaction and Conversion Profile

April recovered in transaction scale compared with March.

| Metric        | 2026-03 | 2026-04 | Direction |
| --------------------- | ------: | ------: | --------- |
| Transaction Amount  | 6454.84 | 9083.72 | Up    |
| Transaction Orders  |   207 |   337 | Up    |
| Entry Users      |   522 |   906 | Up    |
| Search Entry Users  |   445 |   839 | Up    |
| Order Conversion Rate | 42.34% | 37.42% | Down   |
| Average Order Value  |  31.18 |  26.95 | Down   |

Interpretation:

The store had more exposure, more entry users, more transaction orders, and higher transaction amount, but order conversion rate and average order value declined.

This supports a cautious operating-signal reading: the store recovered scale, but the recovery coexisted with weaker conversion rate and lower average order value.

## Top-SKU Evidence

All nine listed monthly top-3 SKU records are tagged `care_solution` in the source table.

These records provide product-mix context for the listed monthly top-SKU set.

## Combined Operating Profile

Store A's monthly movement is reviewed through the combined visibility, entry, transaction, conversion, activity, and listed-SKU evidence.

April 2026 showed:

- transaction amount up;
- transaction orders up;
- exposure users up;
- entry users up;
- search-entry users up;
- order conversion rate down;
- average order value down;
- activity-order share still high;
- listed top-SKU records retained as product-mix context.

The memory layer preserves observed signals, source fields, and calculation context so later review can reconstruct the combined month-over-month operating profile.

## Current Retail Memory Slots

The current generated retail memory facts use these slots:

| Slot               | Meaning                                                 |
| --------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `visibility_entry_profile`    | Describes exposure, ranking, entry, and search-entry structure.                     |
| `activity_lever_profile`     | Describes activity orders, activity cost, subsidy, and activity-cost ratio as operating-lever evidence. |
| `transaction_conversion_profile` | Describes transaction scale, order conversion, payment, and average order value.            |
| `single_metric_attribution_guard` | Combines the implemented evidence slots into one Store A operating profile.                  |
| `top3_sku_product_mix_note`    | Records product-mix context from the listed monthly top-SKU evidence.             |

## What This Demo Supports

This demo supports:

- consistent field naming;
- SQL-derived diagnostic metrics;
- multi-metric month-over-month diagnostic for Store A;
- traceable retail memory facts;
- entity, period, canonical-field, formula, and source consistency checks.

## Next Evidence

The current store-period record can be extended with:

- repeated reporting windows across more stores;
- broader SKU classification and category mapping;
- promotion-cycle and campaign context;
- competition and price context;
- fulfillment, stockout, delivery-condition, rating, and review signals.

## Future Work

Demo 1 establishes the single-store diagnostic contract and traceable memory-fact structure. Demo 2 applies the same field contract to Stores B-F, while the repeated-window panel supplies the records used by the question-specific comparison experiment in `retail_ops/COMPARABILITY_GATE_V0.md`.
