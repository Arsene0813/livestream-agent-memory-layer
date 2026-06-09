# Repeated-Window Panel Extension Source Notes

This file documents the repeated-window panel extension for the retail-operations evidence path.

The current panel contains Store B, Store C, Store D, Store E, and Store F for February, March, and April 2026. March rows are copied from the existing Demo 2 source table to avoid manual re-entry drift.

## Scope

| Item | Current value |
|---|---|
| Current panel stores | Store B, Store C, Store D, Store E, and Store F |
| Current months entered | 2026-02, 2026-03, and 2026-04 |
| March source | Copied from existing Demo 2 source data |
| Current purpose | Repeated-window panel extension |
| Not current purpose | A new numbered demo, pairwise comparability gate, endpoint behavior, or generated memory facts |

## Field Inclusion Rule

This panel keeps only fields that are clear enough to support the current operating-chain diagnostic.

The operating chain is:

~~~text
visibility -> store entry -> order submission -> payment / transaction -> refund pressure
~~~

## Excluded Order-Status Fields

The panel intentionally excludes unclear backend order-status fields because their backend definitions are not clear enough for diagnostic use. Current samples also show that those order-status fields do not consistently align with explicitly labeled refund fields.

No alternative hidden definition is inferred.

Refund-related evidence is kept through explicitly labeled refund fields:

- `refund_amount`
- `full_refund_orders`
- `refund_orders_all_or_partial`
## Current Interpretation Boundary

This panel supports repeated-window coverage inspection.

It does not yet support:

- final store ranking
- cross-store strategy transfer
- pairwise comparability decisions
- market-area classification
- causal attribution
