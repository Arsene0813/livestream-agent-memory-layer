# Meituan Backend Metric Dictionary

This file documents the Meituan backend metric definitions used in the retail operations demos.

The purpose is to prevent Meituan backend numbers from being treated as generic business metrics without checking their original platform meaning, reporting window, denominator, and data grain.

## Naming Convention

The canonical English field names in this project are the implemented CSV / SQL field names used in the current retail demos.

Demo 1:

- `retail_ops/data/store_a_monthly_metrics.csv`
- `retail_ops/data/store_a_top_skus.csv`
- `retail_ops/sql/01_store_a_month_over_month_diagnostic.sql`
- `retail_ops/outputs/store_a_demo1_sql_output.csv`
- `retail_ops/outputs/generated_retail_memory_facts.json`

Demo 2:

- `retail_ops/data/demo2_store_period_metrics.csv`
- `retail_ops/data/demo2_top_search_terms.csv`
- `retail_ops/data/demo2_top_skus_by_sales_volume.csv`
- `retail_ops/data/demo2_top_skus_by_transaction_amount.csv`
- `retail_ops/sql/02_demo2_cross_store_comparability.sql`
- `retail_ops/outputs/demo2_cross_store_comparability_output.csv`
- `retail_ops/outputs/generated_demo2_retail_memory_facts.json`

Chinese Meituan backend labels are mapped to these implemented English field names.

### Path / File-Name Terms vs Implemented Meaning

Some current paths contain `cross_store_comparability` for reference stability. In the current implementation, `demo2_cross_store_*` paths mean Demo 2 same-period diagnostic evidence and interpretation guardrails. The planned pairwise gate is documented in `retail_ops/COMPARABILITY_GATE_V0.md` and is not implemented in current Demo 2 outputs.

## Store Entity ID Convention / 门店实体 ID 规则

`store_id` is the canonical store identifier used in source CSV files, SQL diagnostics, and metric outputs.

`entity_id` is the retrieval-layer identifier used in generated retail memory facts.

For store-level retail facts, `entity_id` MUST be generated from `store_id` using the following rule:

`entity_id = "store_" + store_id`

Example:

- `store_id = A`
- `entity_id = store_A`

`store_id` must not be replaced by `entity_id` in metric CSV files or SQL outputs.

`entity_id` must not be used as a raw metric-table key unless a future data contract explicitly documents that change.

---

## Period Metadata Fields / 周期元数据字段

### `period_start`, `period_end`, `period_month`

Current status: these fields define the reporting window for manually structured store-period records.

- `period_start` records the first date of the reporting window.
- `period_end` records the last date of the reporting window.
- `period_month` records the calendar-month label used by the current monthly demo data.

These fields are metadata fields, not Meituan backend performance metrics.

They are required because cross-store diagnostic interpretation depends on aligned reporting windows. Same-period alignment is necessary for the current Demo 2 diagnostic, but it is not sufficient to prove that two stores are comparable for pricing, activity, SKU, ranking, fulfillment, or strategy-transfer decisions.

## Region / Market Context Field Status / 区域与经营环境字段状态

### `region_type`

Current status: `region_type` is the implemented metadata field for weak region or market-context evidence in the current retail data contract. This section is the central project definition for `region_type`; other documents should refer back to this definition instead of redefining the field.

In Demo 2, `region_type` may contain coarse available labels such as city-level region labels. These values are retained for data-contract stability and source traceability, but they are not market-area types, consumption-level groups, maturity labels, or peer-store groups; in short, `region_type` is not a hard market-area classification. Stores with similar visible region labels may still differ in purchasing power, delivery radius, local competition, rent structure, promotion pressure, customer behavior, stockout risk, and fulfillment constraints.

In the current project, `region_type` can only be used as weak context alongside period alignment, store type, order volume, visibility and ranking signals, entry and order conversion, activity profile, SKU evidence, data completeness, and future external market evidence. The current Demo 2 sample is too small to support a reliable regional classification.

Future market-area classification should be introduced only after the project has enough store coverage and supporting evidence. If added, it must use new documented fields rather than silently changing the meaning of `region_type`.

Possible future fields:

- `market_area_type`: a documented data-supported market-area classification.
- `market_area_type_source`: the evidence or rule used for the classification.
- `market_area_type_confidence`: whether the classification is data-supported, manually reviewed, or uncertain.

Until those fields are defined, the system should treat market-area classification as an unresolved comparability issue rather than a hard label.

### `store_type`

Current status: `store_type` is the implemented store-operating-model metadata field used in current retail source CSV files, SQL diagnostics, generated outputs, and future comparability-gate planning.

Current demo values include:

- `self-operated`
- `partner`

Correct use: `store_type` may be used as operating-model context when interpreting selected store-period evidence.

Boundary: `store_type` is not a performance label, market-area classification, peer-store group, or sufficient comparability condition by itself. It should be interpreted together with reporting-window alignment, transaction order volume, transaction amount, activity involvement, activity intensity, region or market context, SKU evidence, data completeness, repeated reporting windows, and future competition or fulfillment evidence.

## Source Metrics vs SQL-Derived Diagnostics / 后台原始指标与 SQL 派生诊断边界

Most canonical fields in this dictionary are normalized representations of metrics observed directly from the Meituan merchant backend.

The current SQL demos do not claim to create or infer those backend metrics. Instead, SQL is used to derive a limited set of diagnostic fields from already-normalized canonical source metrics.

Current derived outputs are separated into two layers.

1. SQL output columns:

- Demo 1 month-over-month change fields, ranking-change fields, and supporting diagnostic flags documented below.
- Demo 2 share diagnostics and scope fields, including `search_entry_share_pct`, `activity_order_share_pct`, `comparison_scope_flag`, and `comparison_limit_notes`.
- Repeated-window coverage and movement fields documented in the repeated-window column conventions below.

2. Memory-facing fields and artifacts:

- entity and period metadata, canonical `slot` values, `value`, and `observed_values`;
- `source_fields`, `source_path`, `supporting_source_paths`, and `lineage_path`;
- `calculation`, evidence-trace `confidence`, `limitations`, and active-state metadata.

Memory-facing slots are generated from multiple source fields and SQL-derived columns. They are not raw Meituan backend fields and should not be treated as SQL output headers. The SQL layer must not silently rename, redefine, or reverse-engineer Meituan backend metrics. It also must not turn one threshold into a fixed store-stage label. For example, `order_conversion_rate_pct` follows the documented backend funnel definition: `order_users / entry_users`.

Any new SQL-derived field must be explicitly documented before it is used in generated outputs or memory facts.

### Future Pairwise Comparability-Gate Fields

Pairwise comparability-gate fields are not currently implemented. Current Demo 2 produces same-period diagnostic evidence.

A future pairwise comparability gate should use broader multi-store evidence to decide whether two store-period records can be compared for a specific operating question. Reliable store comparability should be judged from transaction order volume, transaction amount, store type, region and market context, competition environment, SKU structure, fulfillment or stockout evidence where available, repeated reporting windows, and activity evidence.

Activity evidence should be separated into:

- explicit activity status, only if campaign-calendar or backend status evidence exists;
- activity involvement, currently proxied by `activity_orders` and `activity_order_share_pct`;
- activity intensity, currently proxied by `activity_cost`, `activity_cost_ratio_pct`, `merchant_subsidy_amount`, and `platform_subsidy_amount`.

At the current sample size, `region_type` remains weak context only. Future market-context fields should be documented before use and supported by broader multi-store evidence. If future market-area classification is added, it should use new documented fields such as `market_area_type`, `market_area_type_source`, and `market_area_type_confidence` rather than silently changing the meaning of `region_type`.

### Demo 1 SQL-Derived Diagnostic Details

The following fields are not Meituan backend metrics. They are SQL-derived diagnostics created from normalized source metrics for Store A Demo 1.

#### Month-over-month diagnostics

Fields ending in `_mom_pct` compare the current month with the previous available month for the same `store_id`, ordered by `period_start`.

Formula:

```text
current_metric_mom_pct = (current_metric - previous_metric) / previous_metric * 100
```

Examples:

- `transaction_amount_mom_pct`
- `transaction_orders_mom_pct`
- `estimated_income_proxy_mom_pct`
- `exposure_users_mom_pct`
- `search_exposure_users_mom_pct`
- `entry_users_mom_pct`
- `search_entry_users_mom_pct`
- `order_users_mom_pct`
- `payment_users_mom_pct`
- `average_order_value_mom_pct`

Interpretation limit: MoM diagnostics describe directional change between adjacent observed months. They do not prove causality and should not be used alone to label a store as better or worse.

#### Ranking-change diagnostics

`store_average_rank_change` and `search_average_rank_change` compare the current month with the previous available month for the same `store_id`.

Formula:

```text
rank_change = current_rank - previous_rank
```

Because lower ranking numbers indicate better position, a negative value means the average position improved, while a positive value means the average position worsened.

Interpretation limit: ranking change should be read together with exposure, entry, conversion, activity involvement, SKU evidence, and fulfillment context where available. It should not be treated as a standalone explanation for transaction change.

#### Boolean supporting diagnostics

`transaction_recovered_with_conversion_aov_tradeoff` is a SQL-derived supporting observation. It is true only when the latest observed month for the same `store_id` has higher `transaction_amount` and higher `transaction_orders` than the previous month, while `order_conversion_rate_pct` and `average_order_value` both decline.



## Reviewer Helper Fields / 审稿辅助字段

### `sku_name_en`

Current status: `sku_name_en` is a reviewer-readability helper field used for English explanation of selected SKU names.

It does not replace the original SKU name, does not change the Meituan backend source meaning, and must not be treated as a canonical backend metric. If a future data contract needs translated product names, the project should document the translation source and review status separately.

## Demo 2 Same-Period Diagnostic Guardrail Fields

The following fields are SQL-derived diagnostic fields used in Demo 2. They are not Meituan backend metrics and must not be interpreted as store-stage labels, best-store rankings, or operating recommendations.

### SQL-Derived Share / Pressure Diagnostics

| Field | Formula | Current interpretation | Boundary |
|---|---|---|---|
| `search_entry_share_pct` | `search_entry_users / entry_users * 100` | Directional structure metric computed as search-source entry users divided by total entry users in the same reporting window. | Source-level traffic users may overlap, so the ratio should not be read as user-level attribution or exclusive channel contribution. |
| `activity_order_share_pct` | `activity_orders / transaction_orders * 100` | Measures the share of transaction orders associated with activity orders in the current reporting window. | Indicates activity involvement, not full activity status, campaign mechanism, causal demand lift, or promotion-transfer readiness. |

### `comparison_scope_flag`

Type: SQL-derived diagnostic text field.

Purpose: `comparison_scope_flag` records whether a store-period row is inside the current Demo 2 diagnostic scope before any operating interpretation is made.

Current values:

- `same_period_diagnostic_ready`: the row matches the fixed Demo 2 reporting window and contains the core fields used by the current row-level diagnostic.
- `not_comparable_period_mismatch`: the row does not match the Demo 2 reporting window.
- `insufficient_data`: one or more core diagnostic fields are missing.

Core readiness contract for the current Demo 2 fixture:

- `period_start = 2026-03-01`;
- `period_end = 2026-03-31`;
- non-missing `transaction_amount`;
- non-missing `transaction_orders`;
- non-missing `exposure_users`;
- non-missing `entry_users`;
- non-missing `search_exposure_users`;
- non-missing `search_entry_users`;
- non-missing `activity_orders`;
- non-missing `top3_sku_transaction_amount`.

Correct use: This field is a narrow data-readiness and diagnostic-scope guardrail. It indicates that the current Demo 2 core transaction, funnel, activity-involvement, and lightweight product-mix evidence can be discussed under the fixed March 2026 window. It does not certify completeness of every carried-through output column.

Pairwise-gate boundary: `comparison_scope_flag` is not the future pairwise comparability-gate decision. It does not decide whether one store's pricing, promotion, SKU, ranking, fulfillment, or operating strategy can be transferred to another store.

Design reason: The flag follows the exact fields checked in `retail_ops/sql/02_demo2_cross_store_comparability.sql`. Top-3 SKU transaction-amount evidence is included because Demo 2 uses lightweight product-mix evidence to qualify cross-store interpretation. If that evidence is missing, the row should be treated as `insufficient_data` rather than as comparable with zero SKU concentration.

Not supported: This field must not be used as an all-column completeness certificate, store-stage label, performance ranking, causal explanation, pairwise store-matching result, or operating recommendation.

### `comparison_limit_notes`

Type: SQL-derived diagnostic text field.

Purpose: `comparison_limit_notes` records caution notes generated from documented Demo 2 evidence-completeness checks and diagnostic threshold checks. It explains why cross-store comparison should be constrained even when stores share the same reporting window.

Evidence-completeness notes:

| Note | Trigger | Meaning |
|---|---|---|
| `missing_top3_sku_amount_evidence` | Top-SKU transaction-amount evidence is missing. | Missing product-mix evidence must not be treated as zero concentration. |

Diagnostic threshold registry:

| Note | Field | Current threshold | Meaning | Boundary |
|---|---|---:|---|---|
| `high_activity_involvement` | `activity_order_share_pct` | `>= 80` | A large share of transaction orders involved activity orders. | Activity involvement, not explicit campaign status or promotion-transfer approval. |
| `moderate_activity_involvement` | `activity_order_share_pct` | `>= 65` | Activity orders are a meaningful part of the transaction-order structure. | Diagnostic warning only. |
| `top3_sku_amount_concentration` | `top3_sku_transaction_amount_share_pct` | `>= 25` | Top-3 SKU transaction amount is concentrated. | Lightweight top-SKU evidence, not full product-category share. |
| `compare_with_region_store_type_activity_product_mix_limits` | default note | always included for ready rows | Region, store type, activity and product-mix evidence constrain direct comparison. | Reminder that the current output is a diagnostic evidence layer. |

Correct use: This field is used by the memory layer as an interpretation-boundary note. It preserves comparison limits when answering cross-store questions.

Not supported: This field does not rank stores, assign store stages, prove causality, or decide whether a promotion, subsidy, price change, SKU change, or ranking action should be taken.

## Generated Retail Memory Fact Semantics / 生成零售记忆事实语义

Generated retail memory facts are not raw Meituan backend exports.

They are retrieval-facing summaries grounded in canonical source fields, SQL output columns, and documented limitations.

- `entity_id` is the retrieval-layer identifier derived from `store_id`.
- `period_label` identifies the target period or comparison window of the memory fact.
- `period_start` and `period_end` record the exact date range represented by the memory fact.
- `period_granularity` records the time grain of the memory fact.

  Current allowed values:

  - `month`: the fact represents one calendar month, and `period_label`
    uses `YYYY-MM`.
  - `month_range`: the fact represents a bounded window spanning more
    than one calendar month, and `period_label` uses
    `YYYY-MM_to_YYYY-MM`.

  `period_granularity` is retrieval and interpretation metadata, not a
  direct Meituan backend metric. It must agree with `period_start`,
  `period_end`, and `period_label`.
- `observed_values` may include baseline periods when the fact is comparative.
- `source_fields` lists the canonical fields or SQL-derived diagnostics supporting the fact.
- `source_path` records the primary generated output file supporting the memory fact.
- `supporting_source_paths` is an optional list of additional source files when a memory fact includes evidence that does not appear directly in `source_path`, such as top search-term or top-SKU source tables.
- `confidence` means evidence-trace confidence: whether the fact is directly supported by available source fields and SQL output.
- `confidence` does not mean causal confidence, profit confidence, or cross-store transferability.
- `limitations` must state unsupported interpretations, such as cross-store transfer, causal attribution, unaudited profit, incomplete SKU classification, or unknown promotion cycles.

Generated retail memory facts must not introduce new field names, new metric definitions, or new store-stage labels unless those names are first documented in this data dictionary, lineage file, and slot registry.

## Retail Memory Slot Contract / 零售记忆槽位合同

Retail memory slots are retrieval-facing summaries derived from canonical Meituan backend fields and SQL diagnostics.

They are not raw Meituan backend metrics, not store-stage labels, and not one-threshold flags.

The current Store A Demo 1 uses the following canonical retail memory slots:

| Slot               | Meaning                                              | Correct Use                                                   | Not Supported                                |
| --------------------------------- | -------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| `visibility_entry_profile`    | Exposure, ranking, entry, and search-entry structure.                       | Understanding whether the store was being seen and entered during the reporting window.             | Claiming visibility alone caused transaction growth.            |
| `activity_lever_profile`     | Activity orders, activity cost, subsidy, and activity-cost ratio.                 | Understanding how promotional tools were used in the store's operating context.                 | Treating activity as a standalone cause or traditional ROI result.     |
| `transaction_conversion_profile` | Transaction scale, order conversion, payment, and average order value.               | Reading transaction recovery or decline together with funnel quality.                      | Calling a period good or bad from one transaction metric alone.       |
| `single_metric_attribution_guard` | Guardrail against explaining performance from one metric alone.                  | Preventing unsupported attribution from exposure, ranking, activity, conversion, or SKU evidence alone. | Rejecting those metrics as irrelevant.                   |
| `top3_sku_product_mix_note`    | Lightweight top-SKU evidence.                                   | Describing limited leading-SKU evidence.                                    | Full product-category sales-share analysis.                 |

A new retail memory slot should be added only when the same name is consistently used across:

1. generated retail memory facts;
2. SQL output or source-field lineage;
3. demo documentation;
4. README or project summary, if mentioned there;
5. retail evaluation cases, if retrieval-tested.

Supporting SQL observations such as transaction recovery, exposure movement, order-conversion decline, average-order-value decline, source-field improvement, activity-order share, and activity-cost ratio can support these slots, but they should not become independent canonical slots unless they are first documented here.

## 1. Traffic Exposure Metrics / 流量曝光指标

### `exposure_users` / 曝光人数

中文定义：所选周期内，对应位置看到商家的用户数。

English definition: Number of users who saw the merchant in the corresponding display position during the selected period.

Interpretation: This is a user-count metric, not an impression-count metric.

### `exposure_times` / 曝光次数

中文定义：所选周期内，对应位置商家被用户看到的次数。

English definition: Total number of times the merchant was seen by users in the corresponding display position during the selected period.

Interpretation: This is an impression-count metric, not a user-count metric.

### `store_average_rank` / 店铺曝光平均排名

中文定义：后台展示的店铺曝光平均排名。根据后台说明，平均排名通常指商家在商家列表和搜索列表内曝光位置的平均值。

English definition: Backend-reported average exposure rank for the store-level exposure view. According to the backend definition, average rank usually refers to the average exposure position across merchant-list and search-list contexts.

Interpretation: A lower number indicates a better average exposure position. This field should not be mixed with `search_average_rank` or `merchant_list_average_rank` unless the backend page scope is explicitly aligned.

### `search_exposure_users` / 搜索曝光人数

中文定义：所选周期内，通过搜索列表看到商家的用户数。

English definition: Number of users who saw the merchant through search-result exposure during the selected period.

Interpretation: This is a source-level exposure user metric. It should not be summed with other source-level exposure users as total exposure.

### `search_average_rank` / 搜索平均排名

中文定义：商家在搜索列表内曝光位置的平均值。

English definition: Average exposure position of the merchant in search-result lists.

Interpretation: A lower number indicates a better average search-result exposure position.

### `merchant_list_exposure_users` / 商家列表曝光人数

中文定义：所选周期内，通过商家列表页看到商家的用户数。

English definition: Number of users who saw the merchant through merchant-list exposure during the selected period.

Interpretation: This is a source-level exposure user metric.

### `merchant_list_average_rank` / 商家列表平均排名

中文定义：商家在商家列表内曝光位置的平均值。

English definition: Average exposure position of the merchant in merchant-list pages.

Interpretation: A lower number indicates a better average merchant-list exposure position.

### `activity_zone_exposure_users` / 活动专区曝光人数

中文定义：所选周期内，通过活动专区看到商家的用户数。

English definition: Number of users who saw the merchant through activity-zone exposure during the selected period.

### `order_page_exposure_users` / 订单页曝光人数

中文定义：所选周期内，通过订单页相关入口看到商家的用户数。

English definition: Number of users who saw the merchant through order-page related exposure during the selected period.

### `other_exposure_users` / 其他曝光人数

中文定义：所选周期内，通过其他来源看到商家的用户数。

English definition: Number of users who saw the merchant through other exposure sources during the selected period.

---

## 2. Store-Entry Metrics / 入店指标

### `entry_conversion_rate_pct` / 入店转化率

中文公式：入店转化率 = 入店人数 / 曝光人数

English formula: `entry_conversion_rate_pct = entry_users / exposure_users * 100`

Interpretation: This measures the share of exposed users who entered the merchant page.

### `entry_users` / 入店人数

中文定义：所选周期内，由店外进入到店内页面的用户数。

English definition: Number of users who entered the merchant page from outside the store page during the selected period.

Interpretation: This is a user-count metric.

### `entry_times` / 入店次数

中文定义：所选周期内，用户由店外进入到店内页面的次数。

English definition: Total number of visits from outside the store page into the merchant page during the selected period.

Interpretation: This is a visit/action-count metric, not a user-count metric.

### `entry_visit_duration_seconds` / 入店访问时长（s）

中文定义：所选周期内，用户从开始进入该商家相关页面到离开该商家相关页面所用的平均时间。

English definition: Average time in seconds from entering merchant-related pages to leaving merchant-related pages during the selected period.

Current status: This field is defined for future use. It is not currently required in Demo 1 source CSV.

### `search_entry_users` / 搜索入店人数

中文定义：所选周期内，通过搜索来源进入商家页面的用户数。

English definition: Number of users who entered the merchant page through search traffic during the selected period.

Interpretation: This is a source-level entry user metric. It is used as a directional search-entry signal, not as perfect user-level attribution.

### `merchant_list_entry_users` / 商家列表入店人数

中文定义：所选周期内，通过商家列表页进入商家页面的用户数。

English definition: Number of users who entered the merchant page through merchant-list traffic during the selected period.

### `activity_zone_entry_users` / 活动专区入店人数

中文定义：所选周期内，通过活动专区进入商家页面的用户数。

English definition: Number of users who entered the merchant page through activity-zone traffic during the selected period.

### `order_page_entry_users` / 订单页入店人数

中文定义：所选周期内，通过订单页相关入口进入商家页面的用户数。

English definition: Number of users who entered the merchant page through order-page related traffic during the selected period.

### `other_entry_users` / 其他入店人数

中文定义：所选周期内，通过其他来源进入商家页面的用户数。

English definition: Number of users who entered the merchant page through other traffic sources during the selected period.

---

## 3. Order-Submission Funnel Metrics / 下单漏斗指标

### `order_conversion_rate_pct` / 下单转化率

中文公式：下单转化率 = 下单人数 / 入店人数

English formula: `order_conversion_rate_pct = order_users / entry_users * 100`

中文解释：本 demo 将 `order_conversion_rate_pct` 作为美团后台展示的下单转化率，按 `order_users / entry_users` 的漏斗口径解释。

English interpretation: In this demo, `order_conversion_rate_pct` is treated as the backend-reported order conversion rate at the documented user-funnel grain.

It should be read together with `order_users` / 下单人数 and `entry_users` / 入店人数 because both are user-count funnel metrics in the documented reporting window.

### `order_users` / 下单人数

中文定义：所选周期内，最终提交订单的用户数。

English definition: Number of users who finally submitted orders during the selected period.

Interpretation: This is a user-count metric.

### `order_times` / 下单次数

中文定义：所选周期内，用户在商家最终提交订单的总次数。

English definition: Total number of final order-submission actions at the merchant during the selected period.

Interpretation: This is an order-submission/action-count metric.

### `order_amount` / 下单金额

中文定义：所选周期内，用户提交的订单的商品实付总金额。

English definition: Total actual paid commodity amount of submitted orders during the selected period.

---

## 4. Payment Funnel Metrics / 支付漏斗指标

### `payment_users` / 支付人数

中文定义：所选周期内，提交订单并成功支付的用户数。

English definition: Number of users who submitted orders and successfully paid during the selected period.

Interpretation: This is a user-count metric.

### `payment_amount` / 支付金额

中文定义：所选周期内，用户已支付订单的商品实付总金额。

English definition: Total actual paid commodity amount of paid orders during the selected period.

### `payment_conversion_rate_pct` / 支付转化率

中文公式：支付转化率 = 支付人数 / 下单人数

English formula: `payment_conversion_rate_pct = payment_users / order_users * 100`

Interpretation: This measures the share of order-submitting users who successfully paid.

---

## 5. Transaction Metrics / 成交指标

### `transaction_amount` / 成交金额

中文定义：所选时间周期内，该账号所选择条件下门店的当天支付且当天未取消的订单用户实际支付金额。

English definition: Actual amount paid by users for orders that were paid on the same day and not cancelled on the same day under the selected account, store, and time filters.

Interpretation: This is a same-day paid and same-day not-cancelled store-level transaction amount.

Important grain rule: `transaction_amount` is a store-period-level field in `store_a_monthly_metrics.csv` and SQL outputs. SKU-level transaction amount must use `sku_transaction_amount`.

### `transaction_orders` / 成交订单量

中文定义：所选时间周期内，该账号所选择条件下门店的当天支付且当天未取消的订单量。

English definition: Number of orders that were paid on the same day and not cancelled on the same day under the selected account, store, and time filters.

Interpretation: This is a same-day paid and same-day not-cancelled transaction-order count.

### `average_order_value` / 单均价

中文公式：单均价 = 成交金额 / 成交订单量

English formula: `average_order_value = transaction_amount / transaction_orders`

中文解释：在本 demo 中，`average_order_value` 按成交指标页面口径处理，即成交金额除以成交订单量。

English interpretation: In this demo, `average_order_value` follows the transaction-metric page definition: transaction amount divided by transaction orders.

Consistency note: If another backend page defines 单均价 with a different denominator, that value should be treated as a separate backend-reported metric only after its denominator is verified. It should not be mixed with `average_order_value`.

### `business_district_rank` / 商圈排名

中文定义：该商家该指标在所在商圈内主营品类相同的商家中排名及排名变化情况。例如综合药店商家仅看在综合药店商家的排名区间。当门店无蜂窝信息或者无动销时，无商圈排名信息。

English definition: Ranking and ranking change of the merchant among merchants with the same main category in the same trade area. If the store has no honeycomb/location-cell information or no sales activity, no business-district ranking is available.

Current status: `business_district_rank` is included as supplementary backend-reported context in Demo 2. It is not required in Demo 1 source CSV and should not be used alone as a hard comparability condition.

### `gross_revenue` / 营业额

中文定义：营业额为商家的真实流水总额，包含商品原价、餐盒费。针对自配送、众包配送订单，会同时包含顾客实付配送费；针对其他配送类型订单，营业额将不再包含用户支付的配送费。

English definition: Merchant gross transaction flow, including original item price and packaging fee. For self-delivery and crowdsourced delivery orders, it also includes customer-paid delivery fees; for other delivery types, it excludes customer-paid delivery fees.

Current status: This field is defined for future use. It is not currently required in Demo 1 source CSV.

Important distinction:

- `gross_revenue` = 营业额
- `transaction_amount` = 成交金额
- `estimated_income_proxy` = 预计收入 / 预计订单收入 proxy

These fields must not be treated as interchangeable.

### `estimated_income_proxy` / 预计收入 proxy / 预计订单收入 proxy

中文定义：平台展示的预计收入或预计订单收入类指标。预计订单收入通常指营业额扣除商家支出（包括商家补贴、平台服务费等）后的净收入，仅做数据展示，不做结算使用。

English definition: Platform-displayed estimated income after deducting merchant-side expenses such as merchant subsidies and platform service fees. It is for display only and should not be treated as settlement data.

Interpretation: This metric is treated as weak backend-reported context only. It should not be used as audited profit, settlement evidence, ROI, margin evidence, or a primary comparability factor because the current demo data does not include a full calculation breakdown.

---

## 6. Activity / Promotion Metrics / 活动与促销指标

### `activity_original_transaction_amount` / 活动营业总额

中文定义：享受了活动的订单原价交易额。

English definition: Original transaction amount of orders that received promotional benefits.

### `activity_orders` / 活动订单数

中文定义：营销活动带来的订单数。

English definition: Number of orders brought by marketing activities.

### `activity_cost` / 活动成本

中文公式：活动成本 = 商家补贴金额 + 平台补贴金额

English formula: `activity_cost = merchant_subsidy_amount + platform_subsidy_amount`

### `merchant_subsidy_amount` / 商家补贴金额

中文定义：在营销活动中，由商家承担的那部分活动补贴费用。

English definition: The portion of promotional subsidy borne by the merchant.

### `platform_subsidy_amount` / 平台补贴金额

中文定义：在营销活动中，由平台承担的那部分活动补贴费用。

English definition: The portion of promotional subsidy borne by the platform.

### `activity_cost_ratio_pct` / 投入产出比

中文公式：投入产出比 = 活动成本 / 活动营业总额

English formula used in this project: `activity_cost_ratio_pct = activity_cost / activity_original_transaction_amount * 100`

中文解释（美团后台官方表述）：该公式是成本除以活动带动营业额，因此数值越小，单位活动营业额对应的成本越低，活动效率越好。

项目使用边界：本项目原样保留上述后台解释，并按已记录的 `activity_cost / activity_original_transaction_amount` 比率使用该指标。该比率本身不单独证明增量需求、利润、毛利、因果提升或整体活动效果。

English project interpretation: This formula describes recorded activity cost relative to activity original transaction amount. A smaller value means lower recorded activity cost per unit of activity original transaction amount under this backend formula.

Important naming rule:

- In this project, this metric is called `activity_cost_ratio_pct`.
- It should not be called traditional ROI, because traditional ROI is often interpreted as return divided by cost, where larger is better.
- The backend label is 投入产出比, but the formula behaves like a cost ratio.
- The ratio does not by itself establish incremental demand, profit, margin, causal lift, or overall campaign effectiveness.

---


### `refund_amount` / 退款金额

中文定义：所选时间周期内，该账号所选择条件下门店申请退款成功的实际退款到账金额，包含部分退款，不含保险费和重复支付。日期为退款成功日期。

English definition: Actual refund amount successfully returned for the selected account, store, conditions, and time period. It includes partial refunds and excludes insurance fees and duplicate payments. The date is based on refund-success date.


### `full_refund_orders` / 退款订单量（全部退款）

中文定义：所选时间周期内，该账号所选择条件下门店退款成功的订单量，不含部分退款。日期为退款成功的日期。

English definition: Number of successfully refunded orders under the selected account, store, conditions, and time period, excluding partial refunds. The date is based on refund-success date.


### `refund_orders_all_or_partial` / 退款订单量（全部退款+部分退款）

中文定义：所选时间周期内，该账号所选择条件下门店申请退款成功的订单量，包括全部退和部分退款订单。日期为退款成功日期。

English definition: Number of successfully refunded orders under the selected account, store, conditions, and time period, including both full refunds and partial refunds. The date is based on refund-success date.



### `sku_rank` / SKU 排名

中文定义：当前 top SKU 证据表中，该 SKU 在所选门店、所选周期内的排名。

English definition: Rank of the SKU within the current top-SKU evidence table for the selected store and period.

### `sku_name` / SKU 名称

中文定义：后台或人工整理的商品 / SKU 名称。

English definition: Product or SKU name recorded from backend evidence or manually structured evidence.

### `sku_transaction_amount` / SKU 成交金额

中文定义：所选门店、所选周期内，该 SKU 对应的成交金额。

English definition: Transaction amount attributed to the listed SKU within the selected store and period.

Important grain rule: This is SKU-level evidence only. It must not be confused with store-level `transaction_amount`.

### `sales_volume` / SKU 销量

中文定义：所选门店、所选周期内，该 SKU 的销量；如后台未展示，则可以为空。

English definition: Sales volume of the listed SKU where available.

### `sku_category_note` / SKU 品类备注

中文定义：当前 demo 中用于辅助解释的轻量品类备注，不代表完整自动 SKU 分类。

English definition: Lightweight category note used for demo interpretation; not full automated SKU classification.

### `top3_sku_transaction_amount` / Top 3 SKU 成交金额

中文定义：当前 demo 中 Top 3 SKU 的成交金额合计。

English definition: Total transaction amount of the top 3 SKUs used in the current demo.

Interpretation: This is lightweight product-mix evidence only. It is not full category-level sales-share analysis.

### `top3_sku_transaction_amount_share_pct` / Top 3 SKU 成交金额占比

中文公式：Top 3 SKU 成交金额占比 = Top 3 SKU 成交金额 / 成交金额

English formula: `top3_sku_transaction_amount_share_pct = top3_sku_transaction_amount / transaction_amount * 100`

Interpretation: This is used only as lightweight qualitative evidence of leading SKU mix.

---

## 9. Derived Diagnostic Metrics / SQL 派生诊断指标

These fields are not direct Meituan backend labels. They are SQL-derived diagnostic fields calculated from backend-reported metrics.

### `search_exposure_share_pct` / 搜索曝光占比

Formula: `search_exposure_share_pct = search_exposure_users / exposure_users * 100`

Interpretation: Directional measure of how much total store exposure came from search exposure.

### `search_entry_share_pct` / 搜索入店占比

Formula: `search_entry_share_pct = search_entry_users / entry_users * 100`

Interpretation: Directional ratio of search-source entry users to total entry users in the same reporting window.

Boundary: This preserves the documented numerator and denominator. Source-level traffic users may overlap, so the ratio should not be interpreted as user-level attribution or exclusive channel contribution.

### `search_entry_rate_pct` / 搜索曝光到入店转化率

Formula: `search_entry_rate_pct = search_entry_users / search_exposure_users * 100`

Interpretation: Directional source-level conversion from search exposure to search entry.

### `estimated_income_proxy_ratio_pct` / 预计收入 proxy 占成交金额比例

Formula: `estimated_income_proxy_ratio_pct = estimated_income_proxy / transaction_amount * 100`

Interpretation: This is a platform-displayed income proxy ratio, not audited profit margin.








### `activity_order_share_pct` / 活动订单占比

Formula: `activity_order_share_pct = activity_orders / transaction_orders * 100`

Interpretation: In the current demo, this is used to detect activity-lever profile.

### `merchant_subsidy_share_of_activity_cost_pct` / 商家补贴占活动成本比例

Formula: `merchant_subsidy_share_of_activity_cost_pct = merchant_subsidy_amount / activity_cost * 100`

Interpretation: This measures how much of total activity cost was borne by the merchant.

---

## 10. Traffic-Source Overlap Rule / 流量来源重叠规则

中文规则：不同来源顾客数量之和有可能会大于门店总曝光人数，因为同一个顾客可以通过多个曝光来源看到门店。

English rule: The sum of customer counts from different traffic sources may exceed total store exposure users because the same customer can see the store through multiple exposure sources.

Therefore:

- source-level exposure users should not be summed as total exposure users;
- source-level entry users should be treated as directional traffic-source signals;
- `search_entry_users / entry_users` is used as a search-entry share signal, not as perfect user-level attribution.

因此：

- 不同来源的曝光人数不能简单相加成总曝光人数；
- 来源级入店人数应作为方向性的流量来源信号；
- `search_entry_users / entry_users` 只作为搜索入店占比信号，不代表完美的用户级归因。

---

## 11. Metric Consistency Rules / 指标一致性规则

### Rule 1: Keep order conversion tied to the documented funnel grain.

`order_conversion_rate_pct` follows:

`order_conversion_rate_pct = order_users / entry_users * 100`

It should be interpreted at the same user-count funnel grain as `order_users` and `entry_users`.


### Rule 2: Do not sum traffic-source users into total exposure users.

规则 2：不要把不同流量来源用户数直接相加成总曝光人数。

Traffic-source user counts may overlap.

不同流量来源用户可能重叠。

### Rule 3: Treat activity and subsidy as operating-lever evidence, not causal proof.

规则 3：将活动与补贴视为经营工具证据，而不是因果证明。

High activity-order share or a low activity cost ratio does not prove that activity caused growth.

高活动订单占比或较低活动成本率，不证明增长一定由活动导致。活动、补贴、价格、SKU 结构、排名和履约信号应结合门店阶段与竞争环境一起解释。




### Rule 4: Keep refund metrics tied to refund-success date.

规则 4：退款金额按退款成功日期统计。

Refund amount should be interpreted according to the backend refund-success-date reporting rule. It should not be used as direct evidence of the original transaction-period order quality unless the reporting window is explicitly aligned.

### Rule 5: Use backend-reported metrics as source of truth when scope is unclear.

规则 5：当口径不完全明确时，以后台展示指标作为事实来源。

Manual recomputation is only valid when numerator, denominator, time window, deduplication rule, and backend reporting scope are explicitly aligned.

只有在分子、分母、时间窗口、去重规则和后台统计口径全部明确一致时，才进行手动重算。

### Rule 6: Do not reuse store-level field names for SKU-level evidence.

规则 6：不要把门店级字段名复用到 SKU 粒度。

`transaction_amount` is store-period-level transaction amount.

`sku_transaction_amount` is SKU-period-level transaction amount.

---

## 12. Field Consistency Checklist

Before adding new SQL outputs, memory facts, or evaluation cases:

- Use the same field names as `store_a_monthly_metrics.csv` for store-period backend metrics.
- Use the same field names as `store_a_top_skus.csv` for SKU-period evidence.
- Do not introduce alternative English names for existing Meituan backend metrics.
- If a new derived metric is added, define its numerator, denominator, time window, and interpretation limit.
- If a field is renamed, update CSV, SQL, SQL output, memory facts, lineage, README, admissions documents, and evaluation cases in the same commit.

## Demo 2 Additional Source Tables

Demo 2 adds cross-store March 2026 source tables. These tables are source-data tables, not memory slots and not store-stage labels.

### demo2_store_period_metrics.csv

This table follows the existing Store A source metric naming pattern wherever possible.

Key naming choices:

- exposure_users, not store_exposure_users
- exposure_times, not store_exposure_times
- entry_times, not entry_visits
- order_times, not order_submissions
- refund_orders_all_or_partial, not full_or_partial_refund_orders
- business_district_rank, not business_area_rank

business_district_rank is included as a supplementary backend-reported field. It should not be used alone as a hard comparability condition because business-district boundaries and local competitive contexts may differ across stores.

### `demo2_top_search_terms.csv`

This source table records the top three backend-reported search terms for
each store-period. Its rows are search-term evidence, not store-level totals,
customer-level attribution, keyword-quality scores, or causal explanations.

| Field | Dictionary definition | Data grain | Interpretation boundary |
|---|---|---|---|
| `search_term_rank` | Position of the search term in the backend top-search-term list for the selected store-period. | Store-period-search-term | Ordering metadata only; it is not a performance score or keyword-quality classification. |
| `search_term` | Original search term retained from the available backend evidence. | Store-period-search-term | Source value and source of truth for the term text; it must not be replaced by the English helper field. |
| `search_term_en` | Conservative English helper translation of `search_term` for reviewer readability. | Store-period-search-term | Helper text only; it is not an independently observed backend value or a translated source-of-truth field. |
| `search_term_exposure_times` | Number of recorded search-result exposures associated with the listed search term in the selected reporting window. | Store-period-search-term | Impression count, not unique users; it must not be substituted for `search_exposure_users`. |
| `search_term_click_times` | Number of recorded clicks associated with the listed search term in the selected reporting window. | Store-period-search-term | Click count, not unique click users, entry users, or transaction orders. |
| `search_term_order_times` | Number of recorded order actions attributed to the listed search term in the source evidence. | Store-period-search-term | Search-term evidence only; it is not interchangeable with store-level `transaction_orders` and does not prove causal attribution. |

The original Chinese `search_term` remains the source value.
`search_term_en` is retained only as a conservative reviewer-readability
helper.

Search-term exposure, click, and order counts should be interpreted together.
They describe the limited top-search-term evidence available for the selected
store-period. They do not represent complete search-query coverage, unique
customer attribution, full search-funnel reconstruction, or proof that a
search term caused transaction performance.

### demo2_top_skus_by_sales_volume.csv

This table records the top 3 SKUs by backend-reported sales volume for each store-period.

Fields:

- sku_rank
- sku_name
- sku_name_en
- sku_transaction_amount
- sales_volume
- sku_category_note

When transaction amount is not available for a sales-volume-ranked SKU, sku_transaction_amount is left blank.

### demo2_top_skus_by_transaction_amount.csv

This table records the top 3 SKUs by backend-reported SKU transaction amount for each store-period.

Fields:

- sku_rank
- sku_name
- sku_name_en
- sku_transaction_amount
- sales_volume
- sku_category_note

When sales volume is not available for a transaction-amount-ranked SKU, sales_volume is left blank.

### English helper fields

sku_name and search_term preserve the original Chinese backend values.

sku_name_en and search_term_en are conservative English helper translations for readability. They are not treated as source-of-truth backend values.

### SKU category handling

Demo 2 does not perform full manual SKU category classification.

For Demo 2 source tables, sku_category_note = not_classified means the SKU name is retained as source evidence but not converted into a full product-category taxonomy.

### Current Boundary Wording for Validators

These exact boundary phrases are intentionally preserved for consistency checks:

- `region_type remains weak context only`
- `activity_cost_ratio_pct` is not traditional ROI.
- `top3_sku_transaction_amount_share_pct` is not full product-category share.

## Repeated-Window Summary Column Convention

Repeated-window summary columns are report-derived columns created for the
B-F repeated-window review.

Columns such as `feb_transaction_amount`, `mar_transaction_amount`, and
`apr_transaction_amount` preserve the canonical metric name after a month
prefix. For example, `feb_transaction_amount` means the February value of
the canonical `transaction_amount` field in the repeated-window summary
output.

These month-prefixed columns are not raw Meituan backend fields. They should
not be used as standalone metric definitions outside the repeated-window
summary output, and they must not replace the canonical base fields documented
in this dictionary.

Correct use: these columns make repeated store-period coverage easier to
inspect across February, March, and April 2026.

Boundary: these columns support descriptive coverage review only. They should
not be interpreted as store rankings, pairwise comparability decisions,
strategy-transfer approvals, causal explanations, or generated memory facts
by themselves.

## Panel Coverage Derived Column Convention

The following panel-coverage fields are report-derived output fields for `retail_ops/outputs/store_period_panel_coverage_output.csv`. They are not raw Meituan backend export fields.

### Panel coverage metadata fields

- `observed_month_count`: number of distinct `period_month` values observed for the store in the current repeated-window panel.
- `first_observed_month`: earliest observed `period_month` for the store in the current repeated-window panel.
- `last_observed_month`: latest observed `period_month` for the store in the current repeated-window panel.
- `observed_months`: pipe-separated list of observed `period_month` values used for coverage inspection.

These fields support repeated-window coverage review only. They do not prove store comparability, store quality, market maturity, or operating-transfer readiness.

### Average-value coverage fields

Fields with the `avg_` prefix are arithmetic averages of already reported store-period values across the included repeated-window panel rows. They are not raw Meituan backend fields and they are not recomputed funnel ratios.

- `avg_transaction_amount`: arithmetic average of `transaction_amount` across the included store-period rows.
- `avg_transaction_orders`: arithmetic average of `transaction_orders` across the included store-period rows.
- `avg_exposure_users`: arithmetic average of `exposure_users` across the included store-period rows.
- `avg_entry_users`: arithmetic average of `entry_users` across the included store-period rows.
- `avg_order_conversion_rate_pct`: arithmetic average of the already reported `order_conversion_rate_pct` values. It is not recomputed from summed `order_users` and summed `entry_users`.
- `avg_payment_conversion_rate_pct`: arithmetic average of the already reported `payment_conversion_rate_pct` values. It is not recomputed from summed `payment_users` and summed `order_users`.
- `avg_activity_cost_ratio_pct`: arithmetic average of the already reported `activity_cost_ratio_pct` values. It is not ROI, profit, margin, or audited activity efficiency.

Correct use: these fields make repeated-window panel coverage easier to inspect before later diagnostic work.

Boundary: these fields must not be used as store rankings, causal evidence, operating recommendations, pairwise comparability decisions, or strategy-transfer approvals.

## Output Boundary / Guardrail Fields

The following fields are report-derived guardrail fields. They are not raw Meituan backend metrics.

- `panel_coverage_flag`: output-level coverage status for descriptive repeated-window panel review.
- `panel_scope_note`: plain-language boundary note for the panel coverage output.
- `repeated_window_summary_flag`: output-level flag indicating whether a store has enough repeated-window observations for descriptive summary review.
- `summary_boundary_note`: plain-language boundary note for the repeated-window summary output.

Correct use: these fields help preserve the difference between descriptive diagnostic readiness and stronger operating interpretation.

Boundary: these fields must not introduce ranking, causal, endpoint-behavior, generated-memory, or operating-recommendation claims.

## Repeated-Window Delta / Percentage Column Convention

Fields ending in `_feb_to_apr_delta` and `_feb_to_apr_pct` are report-derived movement fields for the February-to-April repeated-window summary.

- `*_feb_to_apr_delta`: April value minus February value for the same canonical metric.
- `*_feb_to_apr_pct`: February-to-April delta divided by the February value, multiplied by 100, when the February denominator is available and non-zero.

Examples:

- `transaction_amount_feb_to_apr_delta`
- `transaction_amount_feb_to_apr_pct`
- `entry_users_feb_to_apr_delta`
- `entry_users_feb_to_apr_pct`
- `activity_cost_ratio_pct_feb_to_apr_delta`

Correct use: these fields describe directional movement within the selected repeated-window output.

Boundary: they do not explain why the movement happened, and they do not prove cross-store comparability, causal impact, customer satisfaction, product quality, or operating recommendation readiness.

## Repeated-Window Analytical Output Field Contract

This section defines the existing analytical columns in
`retail_ops/outputs/repeated_window_panel_summary_output.csv`.

These columns are derived comparison outputs. They are not raw backend source
fields, causal estimates, forecasts, or complete operating recommendations.
Their current names are retained.

### February and April count snapshots

- `feb_activity_orders`
- `feb_entry_users`
- `feb_exposure_users`
- `feb_search_entry_users`
- `feb_search_exposure_users`
- `feb_transaction_orders`
- `apr_activity_orders`
- `apr_entry_users`
- `apr_exposure_users`
- `apr_search_entry_users`
- `apr_search_exposure_users`
- `apr_transaction_orders`

Each `feb_` field is the corresponding February store-month value. Each
`apr_` field is the corresponding April store-month value.

These fields preserve the unit and grain of their base metric. They do not
represent a sum across February through April.

### February and April percentage snapshots

- `feb_activity_cost_ratio_pct`
- `feb_entry_conversion_rate_pct`
- `feb_order_conversion_rate_pct`
- `feb_payment_conversion_rate_pct`
- `apr_activity_cost_ratio_pct`
- `apr_entry_conversion_rate_pct`
- `apr_order_conversion_rate_pct`
- `apr_payment_conversion_rate_pct`

Each field is the corresponding February or April percentage metric already
calculated for that store-month.

The `_pct` suffix means the stored value is expressed in percentage units.
These snapshot fields are not percentage changes.

### February-to-April absolute count deltas

- `activity_orders_feb_to_apr_delta`
- `exposure_users_feb_to_apr_delta`
- `search_entry_users_feb_to_apr_delta`
- `search_exposure_users_feb_to_apr_delta`
- `transaction_orders_feb_to_apr_delta`

For each field:

`feb_to_apr_delta = April value - February value`

The result preserves the unit of the base count. A positive value means the
April count is higher; a negative value means it is lower. The value is not
normalized for store size, traffic opportunity, market conditions, or
intervening operational changes.

### February-to-April percentage-point deltas

- `entry_conversion_rate_pct_feb_to_apr_delta`
- `order_conversion_rate_pct_feb_to_apr_delta`
- `payment_conversion_rate_pct_feb_to_apr_delta`

For each field:

`feb_to_apr_delta = April percentage value - February percentage value`

The result is a percentage-point difference, not a relative percent change.
For example, movement from 10% to 12% is a `2` percentage-point delta, not a
20% value in these fields.

### February-to-April relative percent changes

- `exposure_users_feb_to_apr_pct`
- `search_entry_users_feb_to_apr_pct`
- `search_exposure_users_feb_to_apr_pct`
- `transaction_orders_feb_to_apr_pct`

For each field:

`feb_to_apr_pct = (April value - February value) / February value * 100`

This calculation requires a valid non-zero February denominator. It describes
relative change in the named metric only. It does not establish causality,
cross-store comparability, or a final operating recommendation.

### Repeated-window interpretation boundary

The February and April snapshots, absolute deltas, percentage-point deltas,
and relative percent changes should be read together with the existing
`observed_month_count`, `repeated_window_summary_flag`, and
`summary_boundary_note` fields.

No repeated-window output alone proves that an activity, search term, product
mix, subsidy, or store characteristic caused the observed change.
