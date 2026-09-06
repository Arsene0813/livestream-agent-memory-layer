-- Execute through retail_ops.sql_runtime to register canonical numeric validation.
-- Demo 2: same-period B-F diagnostic
--
-- Source tables expected:
-- demo2_store_period_metrics
-- demo2_top_skus_by_transaction_amount
--
-- Purpose:
-- Derive same-period diagnostic fields and interpretation-limit notes
-- before any future pairwise comparability gate.
--
-- Boundary:
-- This SQL does not decide pairwise store comparability.
-- It does not rank stores as better or worse.
-- It keeps refund backend fields as raw source fields only.

WITH top3_sku_amount AS (
    SELECT
        store_id,
        period_month,
        period_start,
        period_end,
        CASE
            WHEN COUNT(*) = 3
             AND COUNT(DISTINCT CAST(retail_value('sku_rank', sku_rank) AS INTEGER)) = 3
             AND COUNT(retail_value('sku_transaction_amount', sku_transaction_amount)) = 3
            THEN ROUND(SUM(CAST(retail_value('sku_transaction_amount', sku_transaction_amount) AS REAL)), 2)
        END AS top3_sku_transaction_amount
    FROM demo2_top_skus_by_transaction_amount
    WHERE CAST(retail_value('sku_rank', sku_rank) AS INTEGER) BETWEEN 1 AND 3
    GROUP BY store_id, period_month, period_start, period_end
),

diagnostics AS (
    SELECT
        m.store_id,
        m.period_month,
        m.period_start,
        m.period_end,
        m.region_type,
        m.store_type,

        CAST(retail_value('transaction_amount', m.transaction_amount) AS REAL) AS transaction_amount,
        CAST(retail_value('transaction_orders', m.transaction_orders) AS INTEGER) AS transaction_orders,
        CAST(retail_value('estimated_income_proxy', m.estimated_income_proxy) AS REAL) AS estimated_income_proxy,
        CAST(retail_value('average_order_value', m.average_order_value) AS REAL) AS average_order_value,

        CAST(retail_value('exposure_users', m.exposure_users) AS INTEGER) AS exposure_users,
        CAST(retail_value('exposure_times', m.exposure_times) AS INTEGER) AS exposure_times,
        CAST(retail_value('store_average_rank', m.store_average_rank) AS REAL) AS store_average_rank,

        CAST(retail_value('entry_users', m.entry_users) AS INTEGER) AS entry_users,
        CAST(retail_value('entry_times', m.entry_times) AS INTEGER) AS entry_times,
        CAST(retail_value('entry_conversion_rate_pct', m.entry_conversion_rate_pct) AS REAL) AS entry_conversion_rate_pct,

        CAST(retail_value('order_users', m.order_users) AS INTEGER) AS order_users,
        CAST(retail_value('order_times', m.order_times) AS INTEGER) AS order_times,
        CAST(retail_value('order_conversion_rate_pct', m.order_conversion_rate_pct) AS REAL) AS order_conversion_rate_pct,
        CAST(retail_value('order_amount', m.order_amount) AS REAL) AS order_amount,

        CAST(retail_value('payment_users', m.payment_users) AS INTEGER) AS payment_users,
        CAST(retail_value('payment_amount', m.payment_amount) AS REAL) AS payment_amount,
        CAST(retail_value('payment_conversion_rate_pct', m.payment_conversion_rate_pct) AS REAL) AS payment_conversion_rate_pct,

        CAST(retail_value('search_exposure_users', m.search_exposure_users) AS INTEGER) AS search_exposure_users,
        CAST(retail_value('search_average_rank', m.search_average_rank) AS REAL) AS search_average_rank,
        CAST(retail_value('search_entry_users', m.search_entry_users) AS INTEGER) AS search_entry_users,

        CAST(retail_value('merchant_list_exposure_users', m.merchant_list_exposure_users) AS INTEGER) AS merchant_list_exposure_users,
        CAST(retail_value('merchant_list_average_rank', m.merchant_list_average_rank) AS REAL) AS merchant_list_average_rank,
        CAST(retail_value('merchant_list_entry_users', m.merchant_list_entry_users) AS INTEGER) AS merchant_list_entry_users,

        CAST(retail_value('activity_original_transaction_amount', m.activity_original_transaction_amount) AS REAL) AS activity_original_transaction_amount,
        CAST(retail_value('activity_orders', m.activity_orders) AS INTEGER) AS activity_orders,
        CAST(retail_value('activity_cost', m.activity_cost) AS REAL) AS activity_cost,
        CAST(retail_value('merchant_subsidy_amount', m.merchant_subsidy_amount) AS REAL) AS merchant_subsidy_amount,
        CAST(retail_value('platform_subsidy_amount', m.platform_subsidy_amount) AS REAL) AS platform_subsidy_amount,
        CAST(retail_value('activity_cost_ratio_pct', m.activity_cost_ratio_pct) AS REAL) AS activity_cost_ratio_pct,

        CAST(retail_value('refund_amount', m.refund_amount) AS REAL) AS refund_amount,
        CAST(retail_value('full_refund_orders', m.full_refund_orders) AS INTEGER) AS full_refund_orders,
        CAST(retail_value('refund_orders_all_or_partial', m.refund_orders_all_or_partial) AS INTEGER) AS refund_orders_all_or_partial,

        CAST(retail_value('business_district_rank', m.business_district_rank) AS INTEGER) AS business_district_rank,

        s.top3_sku_transaction_amount AS top3_sku_transaction_amount,

        ROUND(
            CAST(retail_value('search_entry_users', m.search_entry_users) AS REAL)
            / NULLIF(CAST(retail_value('search_exposure_users', m.search_exposure_users) AS REAL), 0) * 100,
            2
        ) AS search_entry_rate_pct,

        ROUND(
            CAST(retail_value('search_entry_users', m.search_entry_users) AS REAL)
            / NULLIF(CAST(retail_value('entry_users', m.entry_users) AS REAL), 0) * 100,
            2
        ) AS search_entry_share_pct,

        ROUND(
            CAST(retail_value('activity_orders', m.activity_orders) AS REAL)
            / NULLIF(CAST(retail_value('transaction_orders', m.transaction_orders) AS REAL), 0) * 100,
            2
        ) AS activity_order_share_pct,

        CASE
            WHEN s.top3_sku_transaction_amount IS NULL
              OR m.transaction_amount IS NULL
              OR m.transaction_amount = ''
            THEN NULL
            ELSE ROUND(
                s.top3_sku_transaction_amount
                / NULLIF(CAST(retail_value('transaction_amount', m.transaction_amount) AS REAL), 0) * 100,
                2
            )
        END AS top3_sku_transaction_amount_share_pct

    FROM demo2_store_period_metrics AS m
    LEFT JOIN top3_sku_amount AS s
        ON m.store_id = s.store_id
       AND m.period_month = s.period_month
       AND m.period_start = s.period_start
       AND m.period_end = s.period_end
)

SELECT
    store_id,
    period_month,
    period_start,
    period_end,
    region_type,
    store_type,

    transaction_amount,
    transaction_orders,
    estimated_income_proxy,
    average_order_value,

    exposure_users,
    exposure_times,
    store_average_rank,

    entry_users,
    entry_times,
    entry_conversion_rate_pct,

    order_users,
    order_times,
    order_conversion_rate_pct,
    order_amount,

    payment_users,
    payment_amount,
    payment_conversion_rate_pct,

    search_exposure_users,
    search_average_rank,
    search_entry_users,
    search_entry_rate_pct,
    search_entry_share_pct,

    merchant_list_exposure_users,
    merchant_list_average_rank,
    merchant_list_entry_users,

    activity_original_transaction_amount,
    activity_orders,
    activity_cost,
    merchant_subsidy_amount,
    platform_subsidy_amount,
    activity_cost_ratio_pct,
    activity_order_share_pct,

    refund_amount,
    full_refund_orders,
    refund_orders_all_or_partial,

    business_district_rank,

    top3_sku_transaction_amount,
    top3_sku_transaction_amount_share_pct,

    -- `same_period_diagnostic_ready` is deliberately narrow.
    -- It confirms only the fixed March 2026 window plus the core fields checked below.
    -- It does not certify every carried-through output field or pairwise comparability.
    CASE
        WHEN period_start != '2026-03-01'
          OR period_end != '2026-03-31'
        THEN 'not_comparable_period_mismatch'
        WHEN transaction_amount IS NULL
          OR transaction_orders IS NULL
          OR exposure_users IS NULL
          OR entry_users IS NULL
          OR search_exposure_users IS NULL
          OR search_entry_users IS NULL
          OR activity_orders IS NULL
          OR top3_sku_transaction_amount IS NULL
        THEN 'insufficient_data'
        ELSE 'same_period_diagnostic_ready'
    END AS comparison_scope_flag,

    TRIM(
        CASE WHEN transaction_amount IS NULL THEN 'missing_transaction_amount; ' ELSE '' END ||
        CASE WHEN top3_sku_transaction_amount IS NULL THEN 'missing_top3_sku_amount_evidence; ' ELSE '' END ||
        CASE
            WHEN activity_order_share_pct >= 80 THEN 'high_activity_involvement; '
            WHEN activity_order_share_pct >= 65 THEN 'moderate_activity_involvement; '
            ELSE ''
        END ||
        CASE
            WHEN top3_sku_transaction_amount_share_pct >= 25 THEN 'top3_sku_amount_concentration; '
            ELSE ''
        END ||
        'compare_with_region_store_type_activity_product_mix_limits'
    ) AS comparison_limit_notes

FROM diagnostics
ORDER BY store_id;
