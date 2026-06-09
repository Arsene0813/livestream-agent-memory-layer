# Demo 1: Store A Month-over-Month Retail Operations Diagnostic

## Purpose

This demo uses one real Meituan-style retail store as a narrow diagnostic example.

The goal is not to prove that one factor caused growth or decline. The goal is to show how detailed merchant-backend metrics, originally designed for single-store review, can be structured into a consistent, traceable decision-support prototype.

The current demo covers Store A from February 2026 to April 2026.

The month labels use natural calendar-month windows: `2026-02` means `2026-02-01` to `2026-02-28`, `2026-03` means `2026-03-01` to `2026-03-31`, and `2026-04` means `2026-04-01` to `2026-04-30`.

## Business Context

In this project, Meituan instant-retail stores are understood through a chain of operating conditions:

```text
being seen -> being entered -> being ordered -> being selected again or maintaining share
```

A store may use activity subsidy, pricing, ranking optimization, SKU mix, and fulfillment control as operating levers inside this chain.



## Current Demo Boundary

This is a single-store month-over-month diagnostic.

It does not implement:

- cross-store comparison;
- store-stage diagnosis;
- causal attribution;
- margin-aware recommendation;
- full SKU category-share analysis;
- automated ingestion from the Meituan backend.

The current demo only supports cautious interpretation of Store A's observed metrics.

## Source Files

| File | Purpose |
|---|---|
| `retail_ops/data/store_a_monthly_metrics.csv` | Store-period backend metrics manually organized from Meituan-style data. |
| `retail_ops/data/store_a_top_skus.csv` | Top-SKU evidence for each month. |
| `retail_ops/data/DATA_DICTIONARY.md` | Canonical field definitions and metric rules. |
| `retail_ops/sql/01_store_a_month_over_month_diagnostic.sql` | Offline SQL diagnostic query. |
| `retail_ops/outputs/store_a_demo1_sql_output.csv` | SQL output used by this demo. |
| `retail_ops/outputs/generated_retail_memory_facts.json` | Generated retail memory facts for retrieval. |
| `retail_ops/LINEAGE.md` | Claim-to-data lineage and metric consistency rules. |

## Metric Definition Rules

This demo follows the project data dictionary.

Important consistency rules:

1. `order_conversion_rate_pct` follows the backend funnel definition:

   ```text
   order_conversion_rate_pct = order_users / entry_users * 100
   ```

   It should not be recalculated as:

   ```text
   order-status counts / entry_users
   ```

2. `activity_cost_ratio_pct` follows the backend-style formula:

   ```text
   activity_cost_ratio_pct = activity_cost / activity_original_transaction_amount * 100
   ```

  This is a cost ratio. It should not be treated as traditional ROI.

3. Traffic-source users may overlap. Source-level users should not be summed into total exposure users or total entry users.

4. Top-SKU evidence is only lightweight product-mix evidence. It is not full category-level sales-share analysis.

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
- They do not prove that visibility alone caused transaction recovery.

## Activity-Lever Profile

Activity metrics are interpreted as operating-lever evidence.

| Month  | Activity Orders | Transaction Orders | Activity Order Share | Activity Cost Ratio |
| ------- | --------------: | -----------------: | -------------------: | ------------------: |
| 2026-02 |       270 |        274 |        98.54% |       40.63% |
| 2026-03 |       201 |        207 |        97.10% |       38.55% |
| 2026-04 |       329 |        337 |        97.63% |       40.69% |

Interpretation:

- Activity orders accounted for a high share of transaction orders in all three months.
- This does not mean activity alone caused growth.
- It means activity was an important operating lever during the observed period.
- The meaning of this lever depends on operating context, competition, price pressure, ranking pressure,.

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

April's recovery should not be read as a simple quality improvement.

The store had more exposure, more entry users, more transaction orders, and higher transaction amount, but order conversion rate and average order value declined.

This supports a cautious operating-signal reading: the store recovered scale, but the recovery coexisted with weaker conversion rate and lower average order value.



| Metric         | 2026-03 | 2026-04 | Direction |
| ---------------------- | ------: | ------: | --------- |

Interpretation:

This is a source-field signal.



## Top-SKU Evidence

The top-SKU evidence is mainly care-solution-heavy.

This evidence is useful as a lightweight product-mix signal, but it is not full category-level sales-share analysis.

The current demo does not manually classify all SKUs, and it does not claim full SKU-category coverage.

## Single-Metric Attribution Guard

The main lesson of this demo is that Store A's monthly changes should not be explained by one metric alone.

April 2026 showed:

- transaction amount up;
- transaction orders up;
- exposure users up;
- entry users up;
- search-entry users up;
- order conversion rate down;
- average order value down;

- activity-order share still high;
- top-SKU evidence still limited.

This is exactly why the project uses a memory layer with traceable facts and limitations. The system should remember observed signals, source fields, and caveats instead of producing a one-factor explanation.

## Current Retail Memory Slots

The current generated retail memory facts use these slots:

| Slot               | Meaning                                                 |
| --------------------------------- | ------------------------------------------------------------------------------------------------------- |
| `visibility_entry_profile`    | Describes exposure, ranking, entry, and search-entry structure.                     |
| `activity_lever_profile`     | Describes activity orders, activity cost, subsidy, and activity-cost ratio as operating-lever evidence. |
| `transaction_conversion_profile` | Describes transaction scale, order conversion, payment, and average order value.            |
| `single_metric_attribution_guard` | Prevents one-factor explanations of growth or decline.                         |
| `top3_sku_product_mix_note`    | Limits top-SKU evidence to lightweight product-mix support.                       |

## What This Demo Supports

This demo supports:

- consistent field naming;
- SQL-derived diagnostic metrics;
- cautious month-over-month comparison;
- traceable retail memory facts;
- refusal of unsupported cross-store claims;
- refusal of unsupported causal attribution.

## What This Demo Does Not Support

This demo does not support:

- deciding which of 48 stores should receive more subsidy;
- claiming that a strategy from Store A transfers to another store;
- judging another store's operating stage;
- proving that activity caused growth;
- claiming that visibility alone explains growth;
- proving full SKU-category sales structure;
- calculating actual profit or margin.

## Future Work

Demo 1 has already led into the current Demo 2 same-period diagnostic path. The next step is not to turn Store A into a transfer rule. It is to expand repeated store-period evidence under the same field contract, then use a future pairwise comparability gate to decide whether selected store-period records can support one specific operating question.

Future expansion should check whether candidate records are comparable by:

- aligned reporting period;
- transaction order volume;
- transaction amount;
- store type;
- weak region context;
- visibility and ranking profile;
- entry and order-conversion profile;
- activity involvement and activity intensity;
- top-SKU evidence;
- data completeness;
- repeated reporting-window stability.

Only after this kind of comparability check should the system attempt stronger cross-store interpretation. Store-stage diagnosis should remain out of scope until the project has stronger evidence on promotion cycles, competition, fulfillment conditions, rating/review signals, and stockout history.
