# Repeated-Window Panel Extension Source Notes

This file documents the repeated-window panel extension for the retail-operations evidence path.

The current panel contains Store B, Store C, Store D, Store E, and Store F for February, March, and April 2026. The March rows are copied from `demo2_store_period_metrics.csv` and validated field-by-field against that source table.

## Evidence Coverage

| Item | Current value |
|---|---|
| Current panel stores | Store B, Store C, Store D, Store E, and Store F |
| Current months entered | 2026-02, 2026-03, and 2026-04 |
| March source | Copied from `demo2_store_period_metrics.csv` and checked by source-to-panel parity validation |
| Current purpose | Repeated-window coverage and descriptive summary for selected store-period fields |

## Panel Fields

The panel uses selected dictionary-defined store-period fields for repeated-window coverage and descriptive review.

The operating chain is:

~~~text
being seen -> being entered -> being ordered -> being selected again / maintaining share
~~~

## Current Use

This panel provides repeated-window coverage and descriptive summaries for Stores B-F across February-April 2026. The records also provide the store-period evidence required for future question-specific gate design in `retail_ops/COMPARABILITY_GATE_V0.md`.

## Repeated-Window Coverage

Coverage uses selected dictionary-defined store-period fields for repeated-window diagnostic coverage.

The current panel retains the backend fields `refund_amount`, `full_refund_orders`, and `refund_orders_all_or_partial`.
