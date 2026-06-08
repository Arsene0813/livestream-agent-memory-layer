-- Repeated-window panel extension summary.
--
-- Purpose:
--   Summarize February-to-April directional changes for each B-F store in
--   the repeated-window panel extension.
--
-- Boundary:
--   This is a descriptive panel summary.
--   This is not a pairwise comparability gate.
--   This is not a store ranking.
--   This is not an endpoint behavior test.
--   This is not generated memory facts.
--   This is not causal analysis.
--
-- Field boundary:
--   Order-status fields excluded from the panel extension are not selected
--   or derived here.

.mode csv
.headers on

WITH typed_panel AS (
    SELECT
        store_id,
        period_month,
        region_type,
        store_type,
        CAST(NULLIF(transaction_amount, '') AS REAL) AS transaction_amount,
        CAST(NULLIF(transaction_orders, '') AS REAL) AS transaction_orders,
        CAST(NULLIF(average_order_value, '') AS REAL) AS average_order_value,
        CAST(NULLIF(exposure_users, '') AS REAL) AS exposure_users,
        CAST(NULLIF(entry_users, '') AS REAL) AS entry_users,
        CAST(NULLIF(entry_conversion_rate_pct, '') AS REAL) AS entry_conversion_rate_pct,
        CAST(NULLIF(order_users, '') AS REAL) AS order_users,
        CAST(NULLIF(order_conversion_rate_pct, '') AS REAL) AS order_conversion_rate_pct,
        CAST(NULLIF(payment_conversion_rate_pct, '') AS REAL) AS payment_conversion_rate_pct,
        CAST(NULLIF(search_exposure_users, '') AS REAL) AS search_exposure_users,
        CAST(NULLIF(search_entry_users, '') AS REAL) AS search_entry_users,
        CAST(NULLIF(activity_orders, '') AS REAL) AS activity_orders,
        CAST(NULLIF(activity_cost_ratio_pct, '') AS REAL) AS activity_cost_ratio_pct,
        CAST(NULLIF(refund_amount, '') AS REAL) AS refund_amount,
        CAST(NULLIF(full_refund_order_count, '') AS REAL) AS full_refund_order_count,
        CAST(NULLIF(full_or_partial_refund_order_count, '') AS REAL) AS full_or_partial_refund_order_count
    FROM store_period_panel_metrics
),
monthly_pivot AS (
    SELECT
        store_id,
        MAX(region_type) AS region_type,
        MAX(store_type) AS store_type,
        COUNT(DISTINCT period_month) AS observed_month_count,

        MAX(CASE WHEN period_month = '2026-02' THEN transaction_amount END) AS feb_transaction_amount,
        MAX(CASE WHEN period_month = '2026-03' THEN transaction_amount END) AS mar_transaction_amount,
        MAX(CASE WHEN period_month = '2026-04' THEN transaction_amount END) AS apr_transaction_amount,

        MAX(CASE WHEN period_month = '2026-02' THEN transaction_orders END) AS feb_transaction_orders,
        MAX(CASE WHEN period_month = '2026-04' THEN transaction_orders END) AS apr_transaction_orders,

        MAX(CASE WHEN period_month = '2026-02' THEN exposure_users END) AS feb_exposure_users,
        MAX(CASE WHEN period_month = '2026-04' THEN exposure_users END) AS apr_exposure_users,

        MAX(CASE WHEN period_month = '2026-02' THEN entry_users END) AS feb_entry_users,
        MAX(CASE WHEN period_month = '2026-04' THEN entry_users END) AS apr_entry_users,

        MAX(CASE WHEN period_month = '2026-02' THEN entry_conversion_rate_pct END) AS feb_entry_conversion_rate_pct,
        MAX(CASE WHEN period_month = '2026-04' THEN entry_conversion_rate_pct END) AS apr_entry_conversion_rate_pct,

        MAX(CASE WHEN period_month = '2026-02' THEN order_conversion_rate_pct END) AS feb_order_conversion_rate_pct,
        MAX(CASE WHEN period_month = '2026-04' THEN order_conversion_rate_pct END) AS apr_order_conversion_rate_pct,

        MAX(CASE WHEN period_month = '2026-02' THEN payment_conversion_rate_pct END) AS feb_payment_conversion_rate_pct,
        MAX(CASE WHEN period_month = '2026-04' THEN payment_conversion_rate_pct END) AS apr_payment_conversion_rate_pct,

        MAX(CASE WHEN period_month = '2026-02' THEN search_exposure_users END) AS feb_search_exposure_users,
        MAX(CASE WHEN period_month = '2026-04' THEN search_exposure_users END) AS apr_search_exposure_users,

        MAX(CASE WHEN period_month = '2026-02' THEN search_entry_users END) AS feb_search_entry_users,
        MAX(CASE WHEN period_month = '2026-04' THEN search_entry_users END) AS apr_search_entry_users,

        MAX(CASE WHEN period_month = '2026-02' THEN activity_orders END) AS feb_activity_orders,
        MAX(CASE WHEN period_month = '2026-04' THEN activity_orders END) AS apr_activity_orders,

        MAX(CASE WHEN period_month = '2026-02' THEN activity_cost_ratio_pct END) AS feb_activity_cost_ratio_pct,
        MAX(CASE WHEN period_month = '2026-04' THEN activity_cost_ratio_pct END) AS apr_activity_cost_ratio_pct,

        MAX(CASE WHEN period_month = '2026-02' THEN refund_amount END) AS feb_refund_amount,
        MAX(CASE WHEN period_month = '2026-04' THEN refund_amount END) AS apr_refund_amount,

        MAX(CASE WHEN period_month = '2026-02' THEN full_refund_order_count END) AS feb_full_refund_order_count,
        MAX(CASE WHEN period_month = '2026-04' THEN full_refund_order_count END) AS apr_full_refund_order_count
    FROM typed_panel
    GROUP BY store_id
),
summary AS (
    SELECT
        store_id,
        region_type,
        store_type,
        observed_month_count,

        feb_transaction_amount,
        mar_transaction_amount,
        apr_transaction_amount,
        ROUND(apr_transaction_amount - feb_transaction_amount, 2) AS transaction_amount_feb_to_apr_delta,
        CASE
            WHEN feb_transaction_amount > 0
            THEN ROUND((apr_transaction_amount - feb_transaction_amount) * 100.0 / feb_transaction_amount, 2)
        END AS transaction_amount_feb_to_apr_pct,

        feb_transaction_orders,
        apr_transaction_orders,
        ROUND(apr_transaction_orders - feb_transaction_orders, 2) AS transaction_orders_feb_to_apr_delta,
        CASE
            WHEN feb_transaction_orders > 0
            THEN ROUND((apr_transaction_orders - feb_transaction_orders) * 100.0 / feb_transaction_orders, 2)
        END AS transaction_orders_feb_to_apr_pct,

        feb_exposure_users,
        apr_exposure_users,
        ROUND(apr_exposure_users - feb_exposure_users, 2) AS exposure_users_feb_to_apr_delta,
        CASE
            WHEN feb_exposure_users > 0
            THEN ROUND((apr_exposure_users - feb_exposure_users) * 100.0 / feb_exposure_users, 2)
        END AS exposure_users_feb_to_apr_pct,

        feb_entry_users,
        apr_entry_users,
        ROUND(apr_entry_users - feb_entry_users, 2) AS entry_users_feb_to_apr_delta,
        CASE
            WHEN feb_entry_users > 0
            THEN ROUND((apr_entry_users - feb_entry_users) * 100.0 / feb_entry_users, 2)
        END AS entry_users_feb_to_apr_pct,

        feb_entry_conversion_rate_pct,
        apr_entry_conversion_rate_pct,
        ROUND(apr_entry_conversion_rate_pct - feb_entry_conversion_rate_pct, 2) AS entry_conversion_rate_pct_feb_to_apr_delta,

        feb_order_conversion_rate_pct,
        apr_order_conversion_rate_pct,
        ROUND(apr_order_conversion_rate_pct - feb_order_conversion_rate_pct, 2) AS order_conversion_rate_pct_feb_to_apr_delta,

        feb_payment_conversion_rate_pct,
        apr_payment_conversion_rate_pct,
        ROUND(apr_payment_conversion_rate_pct - feb_payment_conversion_rate_pct, 2) AS payment_conversion_rate_pct_feb_to_apr_delta,

        feb_search_exposure_users,
        apr_search_exposure_users,
        ROUND(apr_search_exposure_users - feb_search_exposure_users, 2) AS search_exposure_users_feb_to_apr_delta,
        CASE
            WHEN feb_search_exposure_users > 0
            THEN ROUND((apr_search_exposure_users - feb_search_exposure_users) * 100.0 / feb_search_exposure_users, 2)
        END AS search_exposure_users_feb_to_apr_pct,

        feb_search_entry_users,
        apr_search_entry_users,
        ROUND(apr_search_entry_users - feb_search_entry_users, 2) AS search_entry_users_feb_to_apr_delta,
        CASE
            WHEN feb_search_entry_users > 0
            THEN ROUND((apr_search_entry_users - feb_search_entry_users) * 100.0 / feb_search_entry_users, 2)
        END AS search_entry_users_feb_to_apr_pct,

        feb_activity_orders,
        apr_activity_orders,
        ROUND(apr_activity_orders - feb_activity_orders, 2) AS activity_orders_feb_to_apr_delta,

        feb_activity_cost_ratio_pct,
        apr_activity_cost_ratio_pct,
        ROUND(apr_activity_cost_ratio_pct - feb_activity_cost_ratio_pct, 2) AS activity_cost_ratio_pct_feb_to_apr_delta,

        feb_refund_amount,
        apr_refund_amount,
        ROUND(apr_refund_amount - feb_refund_amount, 2) AS refund_amount_feb_to_apr_delta,
        CASE
            WHEN feb_refund_amount > 0
            THEN ROUND((apr_refund_amount - feb_refund_amount) * 100.0 / feb_refund_amount, 2)
        END AS refund_amount_feb_to_apr_pct,

        feb_full_refund_order_count,
        apr_full_refund_order_count,
        ROUND(apr_full_refund_order_count - feb_full_refund_order_count, 2) AS full_refund_order_count_feb_to_apr_delta,

        CASE
            WHEN observed_month_count = 3
            THEN 'summary_ready_for_descriptive_review'
            ELSE 'insufficient_repeated_window_coverage'
        END AS repeated_window_summary_flag,

        'Descriptive repeated-window summary only; not a store ranking, pairwise comparability gate, operating recommendation, endpoint behavior, generated memory fact, or causal analysis.' AS summary_boundary_note
    FROM monthly_pivot
)
SELECT *
FROM summary
ORDER BY store_id;
