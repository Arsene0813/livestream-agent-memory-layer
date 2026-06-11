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




SELECT *
FROM final_output
ORDER BY period_start;
