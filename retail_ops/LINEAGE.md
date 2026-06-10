# Retail Data Lineage

This file owns claim-to-data lineage for the retail-operations evidence path.

It traces how selected Meituan backend metrics move from source CSV files into SQL diagnostics, SQL outputs, generated memory facts, and answer-boundary evaluations.

## Document Ownership

| This file owns | Canonical file for related detail |
|---|---|
| Source-to-SQL-to-memory lineage | `retail_ops/ARCHITECTURE.md` for architecture structure |
| Claim-to-field support | `retail_ops/data/DATA_DICTIONARY.md` for field meanings |
| Metric interpretation rules | `retail_ops/data/DATA_DICTIONARY.md` for full definitions |
| Current diagnostic boundaries | `retail_ops/EXPERIMENTS.md` and `retail_ops/EXPERIMENT_RESULTS.md` for test meaning and outcomes |
| Future pairwise comparability lineage | `retail_ops/COMPARABILITY_GATE_V0.md` for the future gate contract |

Path names that include `cross_store_comparability` are retained for reference stability. In the current implementation, Demo 2 means same-period diagnostic evidence and guardrails. The future pairwise comparability gate is documented separately.

## Shared Lineage Contract

Existing Meituan backend metrics are kept under one canonical English field name. This avoids mixing multiple English names for the same Chinese backend metric.

Main field-contract files:

- `retail_ops/data/DATA_DICTIONARY.md`
- `retail_ops/data/store_a_monthly_metrics.csv`
- `retail_ops/data/store_a_top_skus.csv`
- `retail_ops/data/demo2_store_period_metrics.csv`
- `retail_ops/data/demo2_top_search_terms.csv`
- `retail_ops/data/demo2_top_skus_by_sales_volume.csv`
- `retail_ops/data/demo2_top_skus_by_transaction_amount.csv`

Main SQL files:

- `retail_ops/sql/01_store_a_month_over_month_diagnostic.sql`
- `retail_ops/sql/02_demo2_cross_store_comparability.sql`

Main output files:

- `retail_ops/outputs/store_a_demo1_sql_output.csv`
- `retail_ops/outputs/demo2_cross_store_comparability_output.csv`
- `retail_ops/outputs/generated_retail_memory_facts.json`
- `retail_ops/outputs/generated_demo2_retail_memory_facts.json`

## Demo 1 Scope

| Item | Value |
|---|---|
| Store | Store A |
| Period | February 2026 to April 2026 |
| Data source | Manually structured Meituan merchant-backend metrics |
| Processing method | Offline SQL diagnostic |
| Output | SQL-derived CSV, markdown diagnostic, generated retail memory facts |
| Limitation | Single-store demo; not causal attribution; not cross-store comparison |

## Demo 1 Claim-to-Data Lineage

| Claim / diagnostic | Source fields | SQL output / derived metric | Memory slot | Limitation |
|---|---|---|---|---|
| Store A's visibility and entry structure can be described from exposure, ranking, entry, and search-entry metrics. | `exposure_users`, `store_average_rank`, `entry_users`, `search_exposure_users`, `search_average_rank`, `search_entry_users` | `search_exposure_share_pct`, `search_entry_share_pct`, `search_entry_rate_pct` | `visibility_entry_profile` | Describes whether the store was being seen and entered; does not prove causal growth. |
| Store A's activity metrics should be interpreted as operating-lever evidence. | `activity_original_transaction_amount`, `activity_orders`, `activity_cost`, `merchant_subsidy_amount`, `platform_subsidy_amount` | `activity_order_share_pct`, `activity_cost_ratio_pct`, `merchant_subsidy_share_of_activity_cost_pct` | `activity_lever_profile` | Activity is a tool inside the operating chain, not a standalone causal explanation or simple ROI judgment. |
| Store A's transaction and conversion signals moved in different directions. | `transaction_amount`, `transaction_orders`, `order_conversion_rate_pct`, `average_order_value` | `transaction_amount_mom_pct`, `transaction_orders_mom_pct`, `average_order_value_mom_pct` | `transaction_conversion_profile` | Transaction recovery can coexist with weaker conversion or lower average order value. |
| Store A's changes should not be explained by one metric alone. | Visibility, entry, transaction, conversion, activity, and SKU evidence | Combined multi-signal interpretation | `single_metric_attribution_guard` | The demo supports structured comparison of signals, not causal attribution. |
| Top SKU mix appears care-solution-heavy. | Top-3 SKU records | Top-3 SKU observation | `top3_sku_product_mix_note` | Top-3 evidence only; not full SKU category-share analysis. |

## Metric Lineage Rules

### Conversion Rate

`order_conversion_rate_pct` follows the backend business definition:

~~~text
order_conversion_rate_pct = order_users / entry_users * 100
~~~

It is not derived from:

~~~text
project-side order-status proxy / entry_users
~~~

Reason: unsupported status-derived metrics and user-level funnel metrics should not be mixed into a substitute conversion formula.

### Traffic Source

Traffic-source users may overlap. The same customer may see the store through multiple exposure sources, so source-level exposure users should not be summed into total exposure users.

`search_entry_users / entry_users` is used only as a directional source-entry structure signal.

### Activity and Promotion

`activity_cost_ratio_pct` follows the backend formula:

~~~text
activity_cost_ratio_pct = activity_cost / activity_original_transaction_amount * 100
~~~

A smaller value means lower activity cost per unit of activity-driven revenue. This project avoids calling it traditional ROI because traditional ROI is often interpreted in the opposite direction.

### Transaction Metrics

`transaction_amount` and `transaction_orders` refer to same-day paid and same-day not-cancelled orders.

For the transaction metric page:

~~~text
average_order_value = transaction_amount / transaction_orders
~~~

If another backend page defines 单均价 using a different backend-reported denominator, it should be treated as a separate backend-reported metric rather than mixed with transaction fields.

### Estimated Income

`estimated_income_proxy` is treated as a platform-displayed income proxy. It should not be interpreted as audited profit because the current demo does not contain the full platform calculation breakdown.

### Refund


### Ranking

Business-district ranking is only comparable among merchants in the same main category and business district. Ranking may be unavailable when the store has no honeycomb or grid information, or no sales activity.

## SKU Evidence Grain Note

Top-SKU evidence uses SKU-level fields.

For Demo 1, the source is:

- `retail_ops/data/store_a_top_skus.csv`

For Demo 2, the sources are:

- `retail_ops/data/demo2_top_skus_by_sales_volume.csv`
- `retail_ops/data/demo2_top_skus_by_transaction_amount.csv`

Lineage rules:

- `sku_transaction_amount` is SKU-period-level transaction evidence.
- It must not be confused with store-period-level `transaction_amount`.
- Top-SKU evidence is used only as lightweight product-mix support.
- Top-SKU evidence is not full category-level sales-share analysis.

## Demo 2 Same-Period Diagnostic Lineage

Demo 2 extends the retail operations prototype from a single-store month-over-month diagnostic to a same-period cross-store diagnostic.

The current Demo 2 scope is limited to five anonymized stores:

- Store B
- Store C
- Store D
- Store E
- Store F

All Demo 2 records use the same reporting window:

| Field | Value |
|---|---|
| `period_start` | 2026-03-01 |
| `period_end` | 2026-03-31 |
| `period_month` | 2026-03 |

Demo 2 structures selected backend metrics under the same reporting window and field contract, derives cautious diagnostic signals, and preserves interpretation limits before any operating recommendation is made.

## Demo 2 Source Data

Demo 2 source data is stored in:

- `retail_ops/data/demo2_store_period_metrics.csv`
- `retail_ops/data/demo2_top_search_terms.csv`
- `retail_ops/data/demo2_top_skus_by_sales_volume.csv`
- `retail_ops/data/demo2_top_skus_by_transaction_amount.csv`
- `retail_ops/data/demo2_source_notes.md`

The source metrics are manually transcribed from the Meituan merchant-backend UI used for instant-retail store operations and anonymized at the store level.

Original Chinese backend search terms and SKU names are retained for traceability. English helper columns are included only for readability.

## Demo 2 SQL Diagnostic Output

Demo 2 SQL is stored in:

- `retail_ops/sql/02_demo2_cross_store_comparability.sql`

The generated SQL output is stored in:

- `retail_ops/outputs/demo2_cross_store_comparability_output.csv`

The SQL uses the March 2026 reporting window as a Demo 2 fixture contract. This keeps the current sample reproducible, but it should not be read as a reusable production SQL design for arbitrary 48-store reporting windows.

Carried-through canonical or backend-formula fields include:

- `region_type`
- `store_type`
- `business_district_rank`
- `activity_cost_ratio_pct`

SQL-derived diagnostic fields include:

- `search_entry_rate_pct`
- `search_entry_share_pct`
- `activity_order_share_pct`
- `top3_sku_transaction_amount_share_pct`
- `comparison_scope_flag`
- `comparison_limit_notes`

These derived fields are diagnostic summaries. They do not replace Meituan backend definitions, rank stores, assign store stages, or prove causal operating effects.

## Demo 2 Claim-to-Field Mapping

| Claim / diagnostic | Supporting fields | Interpretation limit |
|---|---|---|
| Stores are in the same Demo 2 reporting window. | `period_month`, `period_start`, `period_end` | Same-period alignment improves diagnostic structure but does not remove differences in region, store type, activity conditions, competition, fulfillment, or SKU mix. |
| Visibility and entry can be compared cautiously across stores. | `exposure_users`, `entry_users`, `entry_conversion_rate_pct`, `search_exposure_users`, `search_entry_users`, `search_entry_rate_pct`, `search_entry_share_pct` | Visibility and entry metrics do not prove causal transaction growth. |
| Activity involvement should constrain cross-store transaction comparison. | `activity_orders`, `activity_order_share_pct`, `activity_cost`, `activity_cost_ratio_pct`, `merchant_subsidy_amount`, `platform_subsidy_amount` | Activity mechanism details and promotion cycle dates are not included. |
| Top search terms provide lightweight demand evidence. | `search_term`, `search_term_exposure_times`, `search_term_click_times`, `search_term_order_times` | Top search terms are store-period evidence, not complete regional consumer-preference proof. |
| Top SKU evidence provides lightweight product-mix evidence. | `sku_name`, `sku_transaction_amount`, `sales_volume`, `top3_sku_transaction_amount_share_pct` | Top-3 evidence is not full SKU category-share analysis. |

## Demo 2 Memory Fact Output

Demo 2 generated memory facts are stored in:

- `retail_ops/outputs/generated_demo2_retail_memory_facts.json`

The generation script is:

- `retail_ops/scripts/generate_demo2_retail_memory_facts.py`

The validation script is:

- `retail_ops/scripts/validate_demo2_retail_memory_facts.py`

Demo 2 reuses existing canonical retail memory slots:

- `visibility_entry_profile`
- `activity_lever_profile`
- `transaction_conversion_profile`
- `top3_sku_product_mix_note`
- `single_metric_attribution_guard`

Demo 2 does not introduce store-stage labels or best-store rankings.

## Demo 2 Carry-Through Note: Order and Payment Amount Fields

The current implementation carries `order_amount` and `payment_amount` from:

- `retail_ops/data/demo2_store_period_metrics.csv`

into:

- `retail_ops/outputs/demo2_cross_store_comparability_output.csv`
- `retail_ops/outputs/generated_demo2_retail_memory_facts.json`

Interpretation boundary:

- `order_amount` is read with `order_users`, `order_times`, and `order_conversion_rate_pct`.
- `payment_amount` is read with `payment_users` and `payment_conversion_rate_pct`.
- `transaction_amount` remains a separate transaction metric and should not be merged with order-submission or payment-funnel amount fields.

## Future Comparability-Gate Lineage

The current implemented retail lineage includes Demo 1, Demo 2, and the post-Demo2 repeated-window panel evidence-preparation layer. Demo 1 traces Store A month-over-month evidence, Demo 2 traces same-period B-F diagnostic evidence, and the panel lineage traces B-F repeated-window coverage and descriptive summary outputs.

- selected Meituan backend fields
- `DATA_DICTIONARY.md` definitions
- canonical CSV files
- Demo 1 and Demo 2 SQL diagnostics
- Demo 1 and Demo 2 output CSV files
- generated Demo 1 and Demo 2 retail memory facts
- validation and evaluation for the implemented scope

The future pairwise comparability gate should extend this lineage only after stronger multi-store evidence is available. The detailed future gate contract is kept in:

- `retail_ops/COMPARABILITY_GATE_V0.md`

## Post-Demo2 Repeated-Window Panel Lineage

The repeated-window panel extension follows the same dictionary-first rule as Demo 1 and Demo 2.

Panel coverage lineage:

| Step | Artifact |
|---|---|
| Metric definitions | `retail_ops/data/DATA_DICTIONARY.md` |
| Source panel | `retail_ops/data/store_period_panel_metrics.csv` |
| Coverage SQL | `retail_ops/sql/03_store_period_panel_coverage.sql` |
| Coverage output | `retail_ops/outputs/store_period_panel_coverage_output.csv` |
| Validator | `retail_ops/scripts/validate_store_period_panel.py` |
| Saved validation result | `retail_ops/outputs/store_period_panel_validation_result.txt` |

Repeated-window summary lineage:

| Step | Artifact |
|---|---|
| Source panel | `retail_ops/data/store_period_panel_metrics.csv` |
| Summary SQL | `retail_ops/sql/04_repeated_window_panel_summary.sql` |
| Summary output | `retail_ops/outputs/repeated_window_panel_summary_output.csv` |
| Validator | `retail_ops/scripts/validate_repeated_window_panel_summary.py` |
| Saved validation result | `retail_ops/outputs/repeated_window_panel_summary_validation_result.txt` |

This panel does not create a pairwise comparability gate. It checks whether Stores B-F have repeated monthly evidence across 2026-02, 2026-03, and 2026-04, then summarizes movement descriptively.

The panel keeps the dictionary names `full_refund_orders` and `refund_orders_all_or_partial`. It also keeps `store_type` values aligned with the existing source data: `self-operated` and `partner`.

## Raw Backend Refund Fields

The following backend fields are retained for source completeness:

- `refund_amount`
- `full_refund_orders`
- `refund_orders_all_or_partial`
