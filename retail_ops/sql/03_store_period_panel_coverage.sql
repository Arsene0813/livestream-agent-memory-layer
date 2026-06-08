-- Store-period panel coverage inspection.
--
-- Purpose:
--   Check whether the repeated-window panel has enough clean store-period
--   coverage to support later diagnostic work.
--
-- Boundary:
--   This is not Demo 3.
--   This is not a pairwise comparability gate.
--   This does not use valid_orders, invalid_orders, or invalid_order_pressure_pct.

.mode csv
.headers on

WITH panel AS (
    SELECT
        store_id,
        period_month,
        region_type,
        store_type,
        business_district_rank,
        transaction_amount,
        transaction_orders,
        average_order_value,
        exposure_users,
        entry_users,
        entry_conversion_rate_pct,
        order_users,
        order_conversion_rate_pct,
        payment_users,
        payment_conversion_rate_pct,
        search_exposure_users,
        search_entry_users,
        activity_orders,
        activity_cost_ratio_pct,
        refund_amount,
        full_refund_order_count,
        full_or_partial_refund_order_count
    FROM store_period_panel_metrics
),
coverage AS (
    SELECT
        store_id,
        COUNT(*) AS observed_month_count,
        MIN(period_month) AS first_observed_month,
        MAX(period_month) AS last_observed_month,
        GROUP_CONCAT(period_month, '|') AS observed_months,
        ROUND(AVG(transaction_amount), 2) AS avg_transaction_amount,
        ROUND(AVG(transaction_orders), 2) AS avg_transaction_orders,
        ROUND(AVG(exposure_users), 2) AS avg_exposure_users,
        ROUND(AVG(entry_users), 2) AS avg_entry_users,
        ROUND(AVG(order_conversion_rate_pct), 2) AS avg_order_conversion_rate_pct,
        ROUND(AVG(payment_conversion_rate_pct), 2) AS avg_payment_conversion_rate_pct,
        ROUND(AVG(activity_cost_ratio_pct), 2) AS avg_activity_cost_ratio_pct,
        ROUND(AVG(refund_amount), 2) AS avg_refund_amount,
        CASE
            WHEN COUNT(*) >= 3 THEN 'panel_ready_for_repeated_window_diagnostic'
            ELSE 'panel_seed_only_needs_more_months'
        END AS panel_coverage_flag,
        'valid_orders, invalid_orders, and invalid_order_pressure_pct are excluded because their backend definitions are not clear enough for diagnostic use.' AS excluded_order_status_note
    FROM panel
    GROUP BY store_id
)
SELECT *
FROM coverage
ORDER BY store_id;
