-- Execute through retail_ops.sql_runtime to register canonical numeric validation.
-- Repeated-window panel extension coverage inspection.
--
-- Purpose:
-- Check whether the repeated-window panel extension has enough clean store-period
-- coverage to support later diagnostic work.
--
-- Boundary:
-- This is not a new numbered demo.
-- This is not a pairwise comparability gate.
-- Coverage uses selected dictionary-defined store-period fields for repeated-window diagnostic coverage.

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
        full_refund_orders,
        refund_orders_all_or_partial
    FROM store_period_panel_metrics
),

observed_month_list AS (
    SELECT
        store_id,
        GROUP_CONCAT(period_month, '|') AS observed_months
    FROM (
        SELECT DISTINCT
            store_id,
            period_month
        FROM panel
        ORDER BY store_id, period_month
    )
    GROUP BY store_id
),

coverage AS (
    SELECT
        p.store_id,
        COUNT(DISTINCT p.period_month) AS observed_month_count,
        MIN(p.period_month) AS first_observed_month,
        MAX(p.period_month) AS last_observed_month,
        m.observed_months,

        CASE WHEN COUNT(retail_value('transaction_amount', transaction_amount)) = COUNT(*)
             THEN ROUND(AVG(CAST(retail_value('transaction_amount', transaction_amount) AS REAL)), 2)
        END AS avg_transaction_amount,
        CASE WHEN COUNT(retail_value('transaction_orders', transaction_orders)) = COUNT(*)
             THEN ROUND(AVG(CAST(retail_value('transaction_orders', transaction_orders) AS REAL)), 2)
        END AS avg_transaction_orders,
        CASE WHEN COUNT(retail_value('exposure_users', exposure_users)) = COUNT(*)
             THEN ROUND(AVG(CAST(retail_value('exposure_users', exposure_users) AS REAL)), 2)
        END AS avg_exposure_users,
        CASE WHEN COUNT(retail_value('entry_users', entry_users)) = COUNT(*)
             THEN ROUND(AVG(CAST(retail_value('entry_users', entry_users) AS REAL)), 2)
        END AS avg_entry_users,
        CASE WHEN COUNT(retail_value('order_conversion_rate_pct', order_conversion_rate_pct)) = COUNT(*)
             THEN ROUND(AVG(CAST(retail_value('order_conversion_rate_pct', order_conversion_rate_pct) AS REAL)), 2)
        END AS avg_order_conversion_rate_pct,
        CASE WHEN COUNT(retail_value('payment_conversion_rate_pct', payment_conversion_rate_pct)) = COUNT(*)
             THEN ROUND(AVG(CAST(retail_value('payment_conversion_rate_pct', payment_conversion_rate_pct) AS REAL)), 2)
        END AS avg_payment_conversion_rate_pct,
        CASE WHEN COUNT(retail_value('activity_cost_ratio_pct', activity_cost_ratio_pct)) = COUNT(*)
             THEN ROUND(AVG(CAST(retail_value('activity_cost_ratio_pct', activity_cost_ratio_pct) AS REAL)), 2)
        END AS avg_activity_cost_ratio_pct,

        CASE
            WHEN COUNT(DISTINCT p.period_month) = 3
             AND MIN(p.period_month) = '2026-02'
             AND MAX(p.period_month) = '2026-04'
            THEN 'panel_ready_for_repeated_window_diagnostic'
            ELSE 'insufficient_repeated_window_coverage'
        END AS panel_coverage_flag,

        'Coverage uses selected dictionary-defined store-period fields for repeated-window diagnostic coverage.' AS panel_scope_note
    FROM panel AS p
    LEFT JOIN observed_month_list AS m
        ON p.store_id = m.store_id
    GROUP BY
        p.store_id,
        m.observed_months
)

SELECT *
FROM coverage
ORDER BY store_id;
