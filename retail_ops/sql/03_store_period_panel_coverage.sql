-- Repeated-window panel extension coverage inspection.
--
-- Purpose:
--   Check whether the repeated-window panel extension has enough clean store-period
--   coverage to support later diagnostic work.
--
-- Boundary:
--   This is not a new numbered demo.
--   This is not a pairwise comparability gate.
--   Coverage uses selected dictionary-defined store-period fields for repeated-window diagnostic coverage.

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
SELECT *
FROM coverage
ORDER BY store_id;
