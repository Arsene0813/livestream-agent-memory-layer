# Repeated-Window Panel Extension Source Notes

This file documents the repeated-window panel extension for the retail-operations evidence path.

The current panel contains Store B, Store C, Store D, Store E, and Store F for February, March, and April 2026. The March rows are copied from `demo2_store_period_metrics.csv` and validated field-by-field against that source table.

## Scope

| Item | Current value |
|---|---|
| Current panel stores | Store B, Store C, Store D, Store E, and Store F |
| Current months entered | 2026-02, 2026-03, and 2026-04 |
| March source | Copied from `demo2_store_period_metrics.csv` and checked by source-to-panel parity validation |
| Current purpose | Repeated-window coverage and descriptive summary for selected store-period fields |

## Panel Field Scope

The panel uses selected dictionary-defined store-period fields that support the current repeated-window diagnostic.

Coverage uses selected dictionary-defined store-period fields for repeated-window diagnostic coverage.

The operating chain is:

~~~text
being seen -> being entered -> being ordered -> being selected again / maintaining share
~~~

This chain is the business framing, not a set of separately measured stages. The monthly panel records continuous sales outcomes across reporting periods; it does not separately identify repeat selection, customer retention, or market-share movement.

Refund backend fields retained in the current panel are:

- `refund_amount`
- `full_refund_orders`
- `refund_orders_all_or_partial`

## Current Interpretation Boundary

This panel supports repeated-window coverage inspection and descriptive summary for Stores B-F across February-April 2026.

Stronger pairwise comparison should use the future comparability-gate contract after the relevant store-period evidence is available.
