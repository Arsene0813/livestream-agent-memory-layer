# Demo 2 Source Notes

## Source

Demo 2 uses manually transcribed Meituan merchant-backend metrics for anonymized instant-retail stores B-F.

All included stores use the same reporting window:

- `period_start`: `2026-03-01`
- `period_end`: `2026-03-31`
- `period_month`: `2026-03`

The source data is not an automated backend export. It is a structured research copy of selected backend fields for a limited decision-support prototype.

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

Demo 2 keeps backend-reported values as source values. SQL-derived fields are used only for diagnostics.

Traffic-source entry metrics are treated as backend-reported channel metrics. They are not assumed to be mutually exclusive, and they should not be summed into total `entry_users`.

`region_type` is weak context only. It is not a hard market-area classification, peer-store grouping rule, or mature regional segmentation.

`business_district_rank` is supplementary backend evidence. It is not a global market ranking and is not a hard comparability condition.

`activity_cost_ratio_pct` follows the project dictionary definition: `activity_cost / activity_original_transaction_amount * 100`.

It should not be described as ROI, profit margin, or operating return.

Top-SKU evidence is used as lightweight product-mix evidence. It is not treated as complete product-category sales share.

The two top-SKU tables reflect two different backend ranking views. The available backend evidence for this demo did not provide both `sales_volume` and `sku_transaction_amount` for every ranking view. Because SKU prices can differ by store, store lifecycle context such as a new-store customer-acquisition period or mature-store steady-state period, activity condition, and local competitive pressure, this project does not back-calculate missing sales volume or transaction amount from the other column.

These tables support selected SKU-concentration notes, not complete category-share, price, margin, or catalogue-level product-mix analysis.

## Screenshot Policy

The repository does not include exhaustive Meituan backend screenshots because the backend contains sensitive store-level operating information.

The project instead provides anonymized structured records, metric definitions, SQL diagnostics, generated memory facts, lineage notes, and validation/evaluation outputs.

## Region Context Note

In the current Demo 2 data, `region_type` values such as `Qingdao` and `Yantai` should be read as coarse available region labels, not as market-area classifications. The field name is retained for data-contract stability. It must not be used alone as a hard peer-store grouping rule or as a pairwise comparability decision.
