# Demo 2 Source Notes

## Source

Demo 2 uses manually transcribed Meituan merchant-backend metrics for anonymized instant-retail stores B-F.

All included stores use the same reporting window:

- `period_start`: `2026-03-01`
- `period_end`: `2026-03-31`
- `period_month`: `2026-03`

The source data is a manually structured research copy of selected backend fields for the decision-support prototype.

## Included Store Records

Demo 2 currently includes five anonymized store-period records:

| store_id | region_type | store_type  | reporting window     |
| -------- | ----------- | ------------- | ------------------------ |
| B    | Qingdao   | self-operated | 2026-03-01 to 2026-03-31 |
| C    | Qingdao   | self-operated | 2026-03-01 to 2026-03-31 |
| D    | Yantai   | self-operated | 2026-03-01 to 2026-03-31 |
| E    | Yantai   | partner    | 2026-03-01 to 2026-03-31 |
| F    | Yantai   | partner    | 2026-03-01 to 2026-03-31 |

## Source Tables

Demo 2 uses four structured source tables:

- `demo2_store_period_metrics.csv`
- `demo2_top_search_terms.csv`
- `demo2_top_skus_by_sales_volume.csv`
- `demo2_top_skus_by_transaction_amount.csv`

## Data Integrity Notes

Demo 2 retains backend-reported values as source values. SQL-derived fields provide the diagnostic layer.

Traffic-source entry metrics retain their backend channel values. Total entry users use the backend `entry_users` field, while channel fields remain separate.

`region_type` retains the coarse region label available in the source data and is reviewed together with the other store-period fields.

`business_district_rank` retains the supplementary ranking value reported by the backend and provides local store-period context.

`activity_cost_ratio_pct` follows the project dictionary definition: `activity_cost / activity_original_transaction_amount * 100`.

The two top-SKU tables preserve different backend ranking views: one by sales volume and one by transaction amount. Each table retains the values available in its source, and missing values remain empty.

The tables provide the listed SKU-concentration evidence used by Demo 2.

## Repository Evidence

The repository provides anonymized structured records, metric definitions, SQL diagnostics, generated memory facts, lineage notes, and validation and evaluation outputs.

## Region Context

In the current Demo 2 data, `region_type` retains the available region labels `Qingdao` and `Yantai` under the canonical field contract.
