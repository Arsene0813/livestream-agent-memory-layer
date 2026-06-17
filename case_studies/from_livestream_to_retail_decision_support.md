# Appendix: From Product Memory to Retail Decision Support

This appendix records the earlier project evolution. The admissions-facing project narrative starts from the retail operations decision-support problem: selected Meituan merchant-backend evidence needs to be structured, checked, and reused without turning weak diagnostic signals into unsupported operating conclusions.

## Why the Retail Setting Required Stricter Evidence Boundaries

In livestream product interaction, a memory layer mainly needs to manage changing product knowledge, such as price, promotion, stock status, shipping policy, and product features.

In instant-retail operations, the lifecycle problem is stricter. Store-level metrics also change over time, but they are tied to reporting windows, store type, activity involvement, activity intensity, SKU structure, ranking context, refund pressure, and weak market context. A metric that is useful for one store-period record may not be safe to reuse for another store or another month.

The shared technical problem is knowledge lifecycle management: deciding what information should be stored, when it should be updated, when it becomes stale, how it should be retrieved, and when it should not be reused.

## Connection to This Memory Layer

The memory-layer idea is useful in retail operations because store-level findings need to be retained with their evidence boundaries. A generated memory fact should preserve the store, period, source fields, observed values, confidence label, and limitations.

For example, if one store has higher search exposure and another store has lower orders, the system should not immediately conclude that search exposure caused the difference. It should first check whether the records are aligned by reporting window and whether the current evidence covers transaction volume, transaction amount, store type, weak region context, SKU structure, observed activity-related metrics, repeated-window stability, competition context, and fulfillment or stockout context.

The current project therefore treats cross-store discussion as boundary-aware diagnostic support. It does not treat same-period records as automatically comparable for pricing, promotion, SKU, ranking, fulfillment, or strategy-transfer decisions.

## Example: How Demo 2 Uses Evidence Boundaries

In Demo 2, a store with stronger transaction scale should not automatically be treated as a better operating model. Instead, it should be considered together with the reporting period, order volume, store type, weak region context, SKU structure, and observed activity-related metrics such as activity order share, activity cost, and activity cost ratio.

That means a later answer can discuss the store's observed March 2026 profile, but it should not automatically treat the observation as applicable to another store without additional comparison evidence.

This is the practical role of the memory layer in the retail setting: it stores not only the metric value, but also the context needed to determine whether the observation should be reused in a cross-store comparison.

