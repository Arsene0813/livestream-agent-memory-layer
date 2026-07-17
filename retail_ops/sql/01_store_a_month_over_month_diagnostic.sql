-- Demo 1: Store A month-over-month diagnostic
--
-- Purpose:
-- Structure selected Store A monthly backend metrics before interpretation.
-- The SQL preserves backend metric meanings and adds derived diagnostic fields.
--
-- Boundary:
-- This query supports month-over-month diagnostic review.
-- It does not attribute performance changes to one metric alone.

WITH top3_sku_amount AS (
    SELECT
        store_id,
        period_month,
        ROUND(SUM(TRY_CAST(sku_transaction_amount AS DOUBLE)), 2) AS top3_sku_transaction_amount
    FROM read_csv_auto(
        'retail_ops/data/store_a_top_skus.csv',
        header = true
    )
    WHERE TRY_CAST(sku_rank AS INTEGER) <= 3
    GROUP BY store_id, period_month
),

store_monthly_metrics AS (
    SELECT
        m.store_id,
        m.period_month,
        m.period_start,
        m.period_end,
        m.region_type,
        m.store_type,

        TRY_CAST(m.transaction_amount AS DOUBLE) AS transaction_amount,
        TRY_CAST(m.transaction_orders AS BIGINT) AS transaction_orders,
        TRY_CAST(m.estimated_income_proxy AS DOUBLE) AS estimated_income_proxy,
        TRY_CAST(m.average_order_value AS DOUBLE) AS average_order_value,

        TRY_CAST(m.exposure_users AS BIGINT) AS exposure_users,
        TRY_CAST(m.exposure_times AS BIGINT) AS exposure_times,
        TRY_CAST(m.store_average_rank AS DOUBLE) AS store_average_rank,

        TRY_CAST(m.entry_conversion_rate_pct AS DOUBLE) AS entry_conversion_rate_pct,
        TRY_CAST(m.entry_users AS BIGINT) AS entry_users,
        TRY_CAST(m.entry_times AS BIGINT) AS entry_times,

        TRY_CAST(m.order_users AS BIGINT) AS order_users,
        TRY_CAST(m.order_times AS BIGINT) AS order_times,
        TRY_CAST(m.order_conversion_rate_pct AS DOUBLE) AS order_conversion_rate_pct,
        TRY_CAST(m.order_amount AS DOUBLE) AS order_amount,

        TRY_CAST(m.payment_users AS BIGINT) AS payment_users,
        TRY_CAST(m.payment_amount AS DOUBLE) AS payment_amount,
        TRY_CAST(m.payment_conversion_rate_pct AS DOUBLE) AS payment_conversion_rate_pct,

        TRY_CAST(m.search_exposure_users AS BIGINT) AS search_exposure_users,
        TRY_CAST(m.search_average_rank AS DOUBLE) AS search_average_rank,
        TRY_CAST(m.search_entry_users AS BIGINT) AS search_entry_users,

        TRY_CAST(m.activity_original_transaction_amount AS DOUBLE) AS activity_original_transaction_amount,
        TRY_CAST(m.activity_orders AS BIGINT) AS activity_orders,
        TRY_CAST(m.activity_cost AS DOUBLE) AS activity_cost,
        TRY_CAST(m.merchant_subsidy_amount AS DOUBLE) AS merchant_subsidy_amount,
        TRY_CAST(m.platform_subsidy_amount AS DOUBLE) AS platform_subsidy_amount,
        TRY_CAST(m.activity_cost_ratio_pct AS DOUBLE) AS activity_cost_ratio_pct,

        TRY_CAST(m.refund_amount AS DOUBLE) AS refund_amount,
        TRY_CAST(m.full_refund_orders AS BIGINT) AS full_refund_orders,
        TRY_CAST(m.refund_orders_all_or_partial AS BIGINT) AS refund_orders_all_or_partial,

        t.top3_sku_transaction_amount
    FROM read_csv_auto(
        'retail_ops/data/store_a_monthly_metrics.csv',
        header = true
    ) AS m
    LEFT JOIN top3_sku_amount AS t
        ON m.store_id = t.store_id
       AND m.period_month = t.period_month
),

with_previous AS (
    SELECT
        *,
        LAG(transaction_amount) OVER (
            PARTITION BY store_id
            ORDER BY period_start
        ) AS prev_transaction_amount,
        LAG(transaction_orders) OVER (
            PARTITION BY store_id
            ORDER BY period_start
        ) AS prev_transaction_orders,
        LAG(estimated_income_proxy) OVER (
            PARTITION BY store_id
            ORDER BY period_start
        ) AS prev_estimated_income_proxy,
        LAG(exposure_users) OVER (
            PARTITION BY store_id
            ORDER BY period_start
        ) AS prev_exposure_users,
        LAG(search_exposure_users) OVER (
            PARTITION BY store_id
            ORDER BY period_start
        ) AS prev_search_exposure_users,
        LAG(entry_users) OVER (
            PARTITION BY store_id
            ORDER BY period_start
        ) AS prev_entry_users,
        LAG(search_entry_users) OVER (
            PARTITION BY store_id
            ORDER BY period_start
        ) AS prev_search_entry_users,
        LAG(order_users) OVER (
            PARTITION BY store_id
            ORDER BY period_start
        ) AS prev_order_users,
        LAG(payment_users) OVER (
            PARTITION BY store_id
            ORDER BY period_start
        ) AS prev_payment_users,
        LAG(average_order_value) OVER (
            PARTITION BY store_id
            ORDER BY period_start
        ) AS prev_average_order_value,
        LAG(store_average_rank) OVER (
            PARTITION BY store_id
            ORDER BY period_start
        ) AS prev_store_average_rank,
        LAG(search_average_rank) OVER (
            PARTITION BY store_id
            ORDER BY period_start
        ) AS prev_search_average_rank,
        LAG(order_conversion_rate_pct) OVER (
            PARTITION BY store_id
            ORDER BY period_start
        ) AS prev_order_conversion_rate_pct
    FROM store_monthly_metrics
),

final_output AS (
    SELECT
        store_id,
        period_month,
        period_start,
        period_end,

        transaction_amount,
        transaction_orders,
        estimated_income_proxy,
        average_order_value,

        exposure_users,
        exposure_times,
        store_average_rank,

        search_exposure_users,
        search_average_rank,

        entry_users,
        entry_times,
        search_entry_users,

        order_users,
        order_times,
        order_amount,

        payment_users,
        payment_amount,

        activity_original_transaction_amount,
        activity_orders,
        activity_cost,
        merchant_subsidy_amount,
        platform_subsidy_amount,

        refund_amount,
        full_refund_orders,
        refund_orders_all_or_partial,

        top3_sku_transaction_amount,

        entry_conversion_rate_pct,
        order_conversion_rate_pct,
        payment_conversion_rate_pct,

        ROUND(search_exposure_users * 100.0 / NULLIF(exposure_users, 0), 2) AS search_exposure_share_pct,
        ROUND(search_entry_users * 100.0 / NULLIF(entry_users, 0), 2) AS search_entry_share_pct,
        ROUND(search_entry_users * 100.0 / NULLIF(search_exposure_users, 0), 2) AS search_entry_rate_pct,
        ROUND(estimated_income_proxy * 100.0 / NULLIF(transaction_amount, 0), 2) AS estimated_income_proxy_ratio_pct,
        ROUND(activity_orders * 100.0 / NULLIF(transaction_orders, 0), 2) AS activity_order_share_pct,

        activity_cost_ratio_pct,

        ROUND(
            merchant_subsidy_amount * 100.0 / NULLIF(activity_cost, 0),
            2
        ) AS merchant_subsidy_share_of_activity_cost_pct,
        ROUND(
            top3_sku_transaction_amount * 100.0 / NULLIF(transaction_amount, 0),
            2
        ) AS top3_sku_transaction_amount_share_pct,

        ROUND(
            (transaction_amount - prev_transaction_amount) * 100.0 / NULLIF(prev_transaction_amount, 0),
            2
        ) AS transaction_amount_mom_pct,
        ROUND(
            (transaction_orders - prev_transaction_orders) * 100.0 / NULLIF(prev_transaction_orders, 0),
            2
        ) AS transaction_orders_mom_pct,
        ROUND(
            (estimated_income_proxy - prev_estimated_income_proxy) * 100.0 / NULLIF(prev_estimated_income_proxy, 0),
            2
        ) AS estimated_income_proxy_mom_pct,
        ROUND(
            (exposure_users - prev_exposure_users) * 100.0 / NULLIF(prev_exposure_users, 0),
            2
        ) AS exposure_users_mom_pct,
        ROUND(
            (search_exposure_users - prev_search_exposure_users) * 100.0 / NULLIF(prev_search_exposure_users, 0),
            2
        ) AS search_exposure_users_mom_pct,
        ROUND((entry_users - prev_entry_users) * 100.0 / NULLIF(prev_entry_users, 0), 2) AS entry_users_mom_pct,
        ROUND(
            (search_entry_users - prev_search_entry_users) * 100.0 / NULLIF(prev_search_entry_users, 0),
            2
        ) AS search_entry_users_mom_pct,
        ROUND((order_users - prev_order_users) * 100.0 / NULLIF(prev_order_users, 0), 2) AS order_users_mom_pct,
        ROUND((payment_users - prev_payment_users) * 100.0 / NULLIF(prev_payment_users, 0), 2) AS payment_users_mom_pct,
        ROUND(
            (average_order_value - prev_average_order_value) * 100.0 / NULLIF(prev_average_order_value, 0),
            2
        ) AS average_order_value_mom_pct,

        store_average_rank - prev_store_average_rank AS store_average_rank_change,
        search_average_rank - prev_search_average_rank AS search_average_rank_change,

        CASE
            WHEN transaction_amount > prev_transaction_amount
             AND transaction_orders > prev_transaction_orders
             AND order_conversion_rate_pct < prev_order_conversion_rate_pct
             AND average_order_value < prev_average_order_value
            THEN true
            ELSE false
        END AS transaction_recovered_with_conversion_aov_tradeoff
    FROM with_previous
)

SELECT *
FROM final_output
ORDER BY period_start;
