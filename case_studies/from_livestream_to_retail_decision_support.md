# From Livestream Product Memory to Retail Decision Support

This case study explains how the same memory-layer design can be extended from customer-facing livestream product interaction to internal retail operations decision support.

In the original livestream setting, the system manages changing product knowledge such as price, promotion, stock status, shipping policy, and product features. In a multi-store retail setting, similar lifecycle problems appear in store-performance observations. These observations are tied to specific reporting periods and may become outdated when reused without context.

The shared problem is managing information that changes over time: deciding what information should be stored, when it should be updated, when it becomes stale, how it should be retrieved, and when it should not be reused.

This is why the livestream memory-layer design can be adapted to the current Meituan decision-support prototype. The memory layer is used to preserve store-period records together with their source fields, reporting window, calculation logic, and interpretation limits. 

In this retail version, SQL first structures selected Meituan backend metrics into diagnostic store-period records. The memory layer then stores supporting evidence and interpretation boundaries so later questions can retrieve not only a metric value, but also the context needed to interpret it: which store it came from, which month it belongs to, which source fields support it, and what conclusions it does or does not support.

The shared design idea is therefore lifecycle-aware memory. In livestream commerce, the lifecycle problem appears in product knowledge. In multi-store instant retail, it appears in recurring operating data and cross-store interpretation. The same memory-layer principle helps both settings avoid using outdated or context-mismatched information.


## Connection to This Memory Layer

The same memory-layer design can support retail operations by preserving store-performance observations as structured records.

In practice, store-level retail data is often uneven and difficult to compare directly. Some stores have strong performance across most metrics, while others have low order volume. Therefore, the first step is not to force a strong interpretation of the data, but to determine whether the available observations are complete, aligned, and recent enough for cautious diagnostic review.

For example, if one store has higher search exposure and another has lower orders, the system should not immediately treat the exposure difference as a sufficient explanation for the order difference. It should first check whether the observations are aligned on time period, order volume, product mix, coarse market context, promotion status, and data completeness.

This makes the memory layer useful not only for storing conclusions, but also for preventing weak or misleading conclusions from being reused in later analysis.


## Example: How Demo 2 Uses Evidence Boundaries

In Demo 2, a store with stronger transaction scale should not automatically be treated as a better operating model. Instead, it should be considered together with the reporting period, order volume, activity status, store type, region, SKU structure, and other supporting observations.

That means a later answer can discuss the store's observed March 2026 profile, but it should not automatically treat the observation as applicable to another store without additional comparison.

This is the practical role of the memory layer in the retail setting: it stores not only the metric value, but also the context needed to determine whether the observation should be reused in a cross-store comparison.
