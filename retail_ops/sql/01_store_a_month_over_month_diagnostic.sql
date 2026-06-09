-- Demo 1: Store A month-over-month diagnostic
--
-- Purpose:
--   Structure selected Store A monthly backend metrics before interpretation.
--   The SQL preserves backend metric meanings and adds derived diagnostic fields.
--
-- Boundary:
--   This query supports month-over-month diagnostic review.
--   It does not attribute performance changes to one metric alone.

WITH store_monthly_metrics AS (
    SELECT
        m.store_id,
        m.period_month,
        m.period_start,
        m.period_end,
        m.region_type,
        m.store_type,

        m.transaction_amount,
        m.transaction_orders,
        m.estimated_income_proxy,
        m.average_order_value,

        m.exposure_users,
        m.exposure_times,
        m.store_average_rank,

        m.entry_conversion_rate_pct,
        m.entry_users,
        m.entry_times,

        m.order_users,
        m.order_times,
        m.order_conversion_rate_pct,
        m.order_amount,

        m.payment_users,
        m.payment_amount,
        m.payment_conversion_rate_pct,

        m.search_exposure_users,
        m.search_average_rank,
        m.search_entry_users,

        m.activity_original_transaction_amount,
        m.activity_orders,
        m.activity_cost,
        m.merchant_subsidy_amount,
        m.platform_subsidy_amount,
        m.activity_cost_ratio_pct,

        m.refund_amount,
        m.full_refund_orders,
        m.refund_orders_all_or_partial,

        t.top3_sku_transaction_amount
    FROM read_csv_auto(
        'retail_ops/data/store_a_monthly_metrics.csv',
        header = true
    ) AS m
    LEFT JOIN (
        SELECT
            store_id,
            period_month,
            ROUND(SUM(sku_transaction_amount), 2) AS top3_sku_transaction_amount
        FROM read_csv_auto(
            'retail_ops/data/store_a_top_skus.csv',
            header = true
        )
        WHERE sku_rank <= 3
        GROUP BY
            store_id,
            period_month
    ) AS t
        ON m.store_id = t.store_id
       AND m.period_month = t.period_month
),

derived_metrics AS (
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

        entry_conversion_rate_pct,
        entry_users,
        entry_times,

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

        activity_original_transaction_amount,
        activity_orders,
        activity_cost,
        merchant_subsidy_amount,
        platform_subsidy_amount,
        activity_cost_ratio_pct,

        refund_amount,
        full_refund_orders,
        refund_orders_all_or_partial,

        top3_sku_transaction_amount,

        ROUND(100.0 * search_exposure_users / NULLIF(exposure_users, 0), 2)
            AS search_exposure_share_pct,

        ROUND(100.0 * search_entry_users / NULLIF(entry_users, 0), 2)
            AS search_entry_share_pct,

        ROUND(100.0 * search_entry_users / NULLIF(search_exposure_users, 0), 2)
            AS search_entry_rate_pct,

        ROUND(100.0 * estimated_income_proxy / NULLIF(transaction_amount, 0), 2)
            AS estimated_income_proxy_ratio_pct,

        ROUND(100.0 * activity_orders / NULLIF(transaction_orders, 0), 2)
            AS activity_order_share_pct,

        ROUND(100.0 * merchant_subsidy_amount / NULLIF(activity_cost, 0), 2)
            AS merchant_subsidy_share_of_activity_cost_pct,

        ROUND(100.0 * refund_amount / NULLIF(transaction_amount, 0), 2)
            AS refund_pressure_pct,

        ROUND(100.0 * refund_orders_all_or_partial / NULLIF(transaction_orders, 0), 2)
            AS refund_order_pressure_pct,


        ROUND(100.0 * top3_sku_transaction_amount / NULLIF(transaction_amount, 0), 2)
            AS top3_sku_transaction_amount_share_pct
    FROM store_monthly_metrics
),

with_previous_month AS (
    SELECT
        derived_metrics.*,

        MAX(period_start) OVER (
            PARTITION BY store_id
        ) AS latest_period_start,

        LAG(transaction_amount) OVER month_window AS prev_transaction_amount,
        LAG(transaction_orders) OVER month_window AS prev_transaction_orders,
        LAG(estimated_income_proxy) OVER month_window AS prev_estimated_income_proxy,

        LAG(exposure_users) OVER month_window AS prev_exposure_users,
        LAG(search_exposure_users) OVER month_window AS prev_search_exposure_users,

        LAG(entry_users) OVER month_window AS prev_entry_users,
        LAG(search_entry_users) OVER month_window AS prev_search_entry_users,

        LAG(order_users) OVER month_window AS prev_order_users,
        LAG(payment_users) OVER month_window AS prev_payment_users,

        LAG(average_order_value) OVER month_window AS prev_average_order_value,
        LAG(refund_amount) OVER month_window AS prev_refund_amount,

        LAG(store_average_rank) OVER month_window AS prev_store_average_rank,
        LAG(search_average_rank) OVER month_window AS prev_search_average_rank,

        LAG(order_conversion_rate_pct) OVER month_window
            AS prev_order_conversion_rate_pct,

        LAG(refund_pressure_pct) OVER month_window
            AS prev_refund_pressure_pct
    FROM derived_metrics

    WINDOW month_window AS (
        PARTITION BY store_id
        ORDER BY period_start
    )
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

        search_exposure_share_pct,
        search_entry_share_pct,
        search_entry_rate_pct,

        estimated_income_proxy_ratio_pct,

        activity_order_share_pct,
        activity_cost_ratio_pct,
        merchant_subsidy_share_of_activity_cost_pct,

        refund_pressure_pct,
        refund_order_pressure_pct,

        top3_sku_transaction_amount_share_pct,

        ROUND(
            100.0 * (transaction_amount - prev_transaction_amount)
            / NULLIF(prev_transaction_amount, 0),
            2
        ) AS transaction_amount_mom_pct,

        ROUND(
            100.0 * (transaction_orders - prev_transaction_orders)
            / NULLIF(prev_transaction_orders, 0),
            2
        ) AS transaction_orders_mom_pct,

        ROUND(
            100.0 * (estimated_income_proxy - prev_estimated_income_proxy)
            / NULLIF(prev_estimated_income_proxy, 0),
            2
        ) AS estimated_income_proxy_mom_pct,

        ROUND(
            100.0 * (exposure_users - prev_exposure_users)
            / NULLIF(prev_exposure_users, 0),
            2
        ) AS exposure_users_mom_pct,

        ROUND(
            100.0 * (search_exposure_users - prev_search_exposure_users)
            / NULLIF(prev_search_exposure_users, 0),
            2
        ) AS search_exposure_users_mom_pct,

        ROUND(
            100.0 * (entry_users - prev_entry_users)
            / NULLIF(prev_entry_users, 0),
            2
        ) AS entry_users_mom_pct,

        ROUND(
            100.0 * (search_entry_users - prev_search_entry_users)
            / NULLIF(prev_search_entry_users, 0),
            2
        ) AS search_entry_users_mom_pct,

        ROUND(
            100.0 * (order_users - prev_order_users)
            / NULLIF(prev_order_users, 0),
            2
        ) AS order_users_mom_pct,

        ROUND(
            100.0 * (payment_users - prev_payment_users)
            / NULLIF(prev_payment_users, 0),
            2
        ) AS payment_users_mom_pct,

        ROUND(
            100.0 * (average_order_value - prev_average_order_value)
            / NULLIF(prev_average_order_value, 0),
            2
        ) AS average_order_value_mom_pct,

        ROUND(
            100.0 * (refund_amount - prev_refund_amount)
            / NULLIF(prev_refund_amount, 0),
            2
        ) AS refund_amount_mom_pct,

        store_average_rank - prev_store_average_rank AS store_average_rank_change,
        search_average_rank - prev_search_average_rank AS search_average_rank_change,

        CASE
            WHEN period_start = latest_period_start
             AND transaction_amount > prev_transaction_amount
             AND transaction_orders > prev_transaction_orders
             AND order_conversion_rate_pct < prev_order_conversion_rate_pct
             AND average_order_value < prev_average_order_value
            THEN TRUE
            ELSE FALSE
        END AS transaction_recovered_with_conversion_aov_tradeoff,

        CASE
            WHEN period_start = latest_period_start
             AND refund_pressure_pct < prev_refund_pressure_pct
            THEN TRUE
            ELSE FALSE
        END AS refund_pressure_improved
    FROM with_previous_month
)

SELECT *
FROM final_output
ORDER BY period_start;
