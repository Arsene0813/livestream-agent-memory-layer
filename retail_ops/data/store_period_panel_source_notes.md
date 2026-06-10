# Repeated-Window Panel Extension Source Notes

This file documents the repeated-window panel extension for the retail-operations evidence path.

The current panel contains Store B, Store C, Store D, Store E, and Store F for February, March, and April 2026. March rows are copied from the existing Demo 2 source table to avoid manual re-entry drift.

## Scope

| Item | Current value |
|---|---|
| Current panel stores | Store B, Store C, Store D, Store E, and Store F |
| Current months entered | 2026-02, 2026-03, and 2026-04 |
| March source | Copied from existing Demo 2 source data |
| Current purpose | Repeated-window coverage and descriptive summary for selected store-period fields |

## Panel Field Scope

The panel uses selected dictionary-defined store-period fields that support the current repeated-window diagnostic.

The operating chain is:

~~~text
being seen -> being entered -> being ordered -> being selected again / maintaining share
~~~

Refund context fields retained in the current panel are:

- `refund_amount`
- `full_refund_orders`
- `refund_orders_all_or_partial`

## Current Interpretation Boundary

This panel supports repeated-window coverage inspection and descriptive summary for Stores B-F across February-April 2026.

Stronger pairwise comparison should use the future comparability-gate contract after the relevant store-period evidence is available.
