# Field Usage Review Before Comparability-Gate Expansion

## Document Ownership

This file owns field-name and semantic-change review for the retail-operations evidence path.

It should answer one narrow question: if a field name or field meaning changes, what existing dictionary definition, source file, SQL output, generated memory fact, and evaluation behavior could be affected?

| This file owns | Canonical file for related detail |
|---|---|
| Field-name change review | `retail_ops/data/DATA_DICTIONARY.md` for authoritative field names and metric meanings |
| Existing field usage review | Source CSV files, SQL outputs, generated memory facts, and eval files |
| Future field-addition caution | `retail_ops/COMPARABILITY_GATE_V0.md` for future pairwise gate fields |
| Rename decision tracking | The field review tables in this file |

This file should not repeat the full admissions narrative, the full architecture path, or the future comparability-gate design. It should keep the review narrow: field meaning, field location, rename risk, and whether a change is allowed.


This file records the field-name review before expanding the retail comparability narrative.

Current decision: **no existing source CSV field, SQL output field, generated memory slot, or evaluation field is renamed in this patch**.

The purpose of this review is to protect the Meituan backend metric contract before future comparability-gate work. Backend-derived fields, SQL-derived diagnostic fields, and retrieval-facing memory slots should not be mixed, renamed, or promoted into new meanings without an explicit mapping review.

## Consolidated Scope Notes

This file also preserves the field-name and scope-change guardrails that protect the current retail evidence path.

- `retail_ops/data/DATA_DICTIONARY.md` remains the source of truth for retail field names and Meituan-style metric meanings.
- Demo 1 remains a Store A month-over-month diagnostic.
- Demo 2 remains a same-period B-F diagnostic for March 2026, not a completed pairwise comparability gate.
- `region_type` remains weak region or market-context evidence only; it is not a hard market-area classification, store-stage label, or peer-store grouping rule.
- Activity evidence should remain separated into involvement, intensity, and future explicit campaign status only when campaign-calendar or backend status evidence exists.
- Retrieval-score analysis remains offline inspection, not production retrieval logic.
- `rac/` should be described as a deterministic source-aware review scaffold over the structured retail evidence path, not as autonomous cognition or a separate operating-decision engine.


## Review Rule

Any future field-name change must pass this review first:

| Existing field | Dictionary definition | Current usage | Rename decision |
|---|---|---|---|
| TBD | Must be checked against `retail_ops/data/DATA_DICTIONARY.md`. | Must list CSV, SQL, output, memory-fact, lineage, README/admissions, and eval usage. | Do not rename unless the full source-to-output path is migrated together. |

Future fields such as `activity_status`, `market_area_type`, `market_area_type_source`, `market_area_type_confidence`, `comparison_question_type`, or `comparison_decision` must not be introduced into source CSVs, SQL outputs, generated facts, or eval cases until they are first documented in `retail_ops/data/DATA_DICTIONARY.md` and linked through `retail_ops/LINEAGE.md`.

## Field-Change Migration Order

Any future field rename or semantic change must be migrated in this order:

1. update `retail_ops/data/DATA_DICTIONARY.md`;
2. update `retail_ops/LINEAGE.md`;
3. update this field-usage review table;
4. update source CSV headers only if the field is a source field;
5. update SQL outputs only if the field is a SQL-derived diagnostic field;
6. update generated memory facts and source-field references;
7. update validation scripts and expected outputs;
8. update README, admissions summary, and demo docs only after the data contract is stable.

This rule is intentionally conservative. The project should prefer adding clearly documented future fields over silently changing the meaning of existing Meituan backend-derived fields.


## Current Field Review Table

| Existing field | Dictionary definition or boundary | Current use location | Rename decision |
|---|---|---|---|
| `store_id` | Canonical store identifier used in source CSV files, SQL diagnostics, and metric outputs. | Source CSVs, SQL outputs, demo outputs. | No. |
| `entity_id` | Retrieval-layer identifier generated from `store_id` using `entity_id = "store_" + store_id`. | Generated retail memory facts. | No. |
| `period_start` | First date of the reporting window. | Source CSVs, SQL outputs, generated facts, lineage. | No. |
| `period_end` | Last date of the reporting window. | Source CSVs, SQL outputs, generated facts, lineage. | No. |
| `period_month` | Calendar-month label for monthly demo records. | Source CSVs, SQL outputs. | No. |
| `region_type` | Weak region or market-context metadata. It is not a store-stage label, mature market-area classification, consumption-level group, or sufficient comparability condition by itself. | Demo 2 source metrics, SQL output, generated facts, comparability review. | No. |
| `store_type` | Store operating-type field used as comparison context. | Source CSVs, SQL output, generated facts. | No. |
| `business_district_rank` | Backend contextual ranking among same-main-category merchants in a business district. It is supplementary context, not a hard comparability condition. | Demo 2 source metrics and lineage. | No. |
| `exposure_users` | Backend-reported number of users who saw the merchant in the selected scope. | Source CSVs, SQL outputs, visibility facts. | No. |
| `exposure_times` | Backend-reported number of times the merchant was seen. | Source CSVs and visibility evidence. | No. |
| `store_average_rank` | Backend-reported average exposure rank. Lower means better position. | Source CSVs, SQL outputs, visibility facts. | No. |
| `search_exposure_users` | Backend-reported users who saw the merchant through search-result exposure. | Source CSVs, SQL outputs, visibility facts. | No. |
| `search_average_rank` | Backend-reported average search-result exposure position. | Source CSVs, SQL outputs, visibility facts. | No. |
| `entry_users` | Backend-reported users entering the store during the selected period. | Source CSVs, SQL output, visibility and conversion facts. | No. |
| `entry_times` | Backend-reported store-entry visits/actions. It is not the same as `entry_users`. | Source CSVs and traffic evidence. | No. |
| `entry_conversion_rate_pct` | Backend-style entry conversion rate, interpreted with exposure and entry scope. | Source CSVs, SQL outputs. | No. |
| `search_entry_users` | Backend-reported users entering from search during the selected period. | Source CSVs, SQL output, visibility facts. | No. |
| `search_entry_rate_pct` | SQL-derived search exposure-to-entry diagnostic. | Demo 2 SQL output and lineage. | No. |
| `search_entry_share_pct` | SQL-derived directional structure metric for search-driven entry share. Source-level users may overlap. | Demo 2 SQL output, generated facts, lineage. | No. |
| `order_users` | Backend order-user metric used in the backend order-conversion formula. | Source CSVs, SQL output, transaction/conversion facts. | No. |
| `order_times` | Backend order-submission/action-count metric. It is not the same as `order_users`. | Source CSVs and funnel evidence. | No. |
| `order_amount` | Backend order-submission amount field. It belongs to the order-submission funnel and must not be merged with `transaction_amount`. | Demo 2 source metrics, SQL output, generated facts, lineage. | No. |
| `order_conversion_rate_pct` | Backend formula field: `order_users / entry_users * 100`. It must not be recomputed from project-side order-status proxies. | Source CSVs, SQL output, lineage, transaction/conversion facts. | No. |
| `payment_users` | Backend successful-payment-user metric. | Source CSVs, SQL output, transaction/conversion facts. | No. |
| `payment_amount` | Backend paid-order commodity amount field. It belongs to the payment funnel and must not be merged with `transaction_amount`. | Demo 2 source metrics, SQL output, generated facts, lineage. | No. |
| `payment_conversion_rate_pct` | Backend payment-conversion metric. | Source CSVs, SQL output, transaction/conversion facts. | No. |
| `transaction_amount` | Backend transaction amount for same-day paid and same-day not-cancelled orders under the selected scope. It must not be mixed with `gross_revenue`, `estimated_income_proxy`, `order_amount`, `payment_amount`, or SKU-level `sku_transaction_amount`. | Source CSVs, SQL output, transaction/conversion facts. | No. |
| `transaction_orders` | Backend transaction-order count for same-day paid and same-day not-cancelled orders. | Source CSVs, SQL output, transaction/conversion facts. | No. |
| `average_order_value` | Backend average-order-value field read with `transaction_amount` and `transaction_orders`. | Source CSVs, SQL output. | No. |
| `estimated_income_proxy` | Platform-displayed estimated income / estimated order income proxy. It is not audited profit. | Source CSVs, SQL output, transaction/conversion facts, evidence-boundary docs. | No. |
| `activity_original_transaction_amount` | Original transaction amount of orders that used activities. | Source CSVs, SQL output, activity facts. | No. |
| `activity_orders` | Backend activity-driven order count. | Source CSVs, SQL output, activity facts. | No. |
| `activity_cost` | Backend activity-cost field. | Source CSVs, SQL output, activity facts. | No. |
| `merchant_subsidy_amount` | Merchant-borne subsidy amount. | Source CSVs, SQL output, activity facts. | No. |
| `platform_subsidy_amount` | Platform-borne subsidy amount. | Source CSVs, SQL output, activity facts. | No. |
| `activity_cost_ratio_pct` | Activity cost divided by activity original transaction amount. It is activity-cost-ratio evidence, not traditional ROI. | Source CSVs, SQL output, activity facts, lineage. | No. |
| `activity_order_share_pct` | SQL-derived activity-order share. It shows activity involvement, not full campaign status, promotion mechanism, causal demand lift, or promotion-transfer readiness. | Demo 2 SQL output, generated facts, comparability review. | No. |
| `refund_amount` | Backend refund amount counted by refund-success date. | Source CSVs, SQL output, order-quality facts. | No. |
| `full_refund_orders` | Backend all-refund order count, excluding partial refunds. | Source CSVs and order-quality evidence. | No. |
| `refund_orders_all_or_partial` | Backend refund-order count including all-refund and partial-refund orders. | Source CSVs and order-quality evidence. | No. |
| `refund_pressure_pct` | SQL-derived refund-pressure signal based on `refund_amount / transaction_amount * 100`. | SQL output and order-quality facts. | No. |
| `sku_name` | SKU-level product name from top-SKU evidence. | Top-SKU source files. | No. |
| `sku_name_en` | English helper column for readability. It does not replace the original Chinese SKU name. | Top-SKU source files. | No. |
| `sku_transaction_amount` | SKU-level transaction amount. It must not be confused with store-period-level `transaction_amount`. | Top-SKU source files and top-SKU evidence. | No. |
| `sales_volume` | SKU-level sales-volume evidence where available. | Top-SKU source files. | No. |
| `top3_sku_transaction_amount_share_pct` | SQL-derived lightweight top-SKU concentration evidence. It is not full product-category sales share. | SQL output and top-SKU memory note. | No. |
| `comparison_scope_flag` | SQL-derived data-readiness and comparison-scope guardrail for Demo 2. It is not a pairwise store-comparability decision. | Demo 2 SQL output and Demo 2 memory facts. | No. |
| `comparison_limit_notes` | SQL-derived interpretation-boundary notes for Demo 2. It records constraints from search, activity, refund, order-quality, region/store context, and product-mix evidence. | Demo 2 SQL output and Demo 2 memory facts. | No. |
| `visibility_entry_profile` | Retrieval-facing memory slot for exposure, ranking, entry, and search-entry structure. | Generated retail memory facts. | No. |
| `activity_lever_profile` | Retrieval-facing memory slot for activity orders, activity cost, subsidy, and activity-cost ratio. | Generated retail memory facts. | No. |
| `transaction_conversion_profile` | Retrieval-facing memory slot for transaction scale, order conversion, payment, and average order value. | Generated retail memory facts. | No. |
| `order_quality_pressure_profile` | Retrieval-facing memory slot for refund-pressure evidence and related refund context. | Generated retail memory facts. | No. |
| `single_metric_attribution_guard` | Retrieval-facing memory slot that prevents unsupported interpretation from one metric alone. | Generated retail memory facts. | No. |
| `top3_sku_product_mix_note` | Retrieval-facing memory slot for limited top-SKU evidence. It is not full category-share analysis. | Generated retail memory facts. | No. |

## Future Comparability-Gate Field Review

Pairwise comparability-gate fields are outside the current implemented retail scope.

A reliable future gate should consider transaction order volume, transaction amount, explicit activity status when source evidence exists, activity involvement, activity intensity, store type, region and market context, competition environment, SKU structure, refund evidence, fulfillment or stockout evidence where available, and repeated reporting windows.

At the current sample size, `region_type` remains weak context only. It must not be used as a hard market-area classification, store-stage label, consumption-level label, or peer-store grouping rule.

Possible future fields such as `activity_status`, `market_area_type`, `market_area_type_source`, `market_area_type_confidence`, `comparison_question_type`, or `comparison_decision` should only be added after they are documented in `retail_ops/data/DATA_DICTIONARY.md` and linked through `retail_ops/LINEAGE.md`.

## Current Decision

No current source CSV field, SQL output field, generated memory slot, or evaluation field is renamed in this patch.
