# Design Evolution: From Livestream Product Memory to Retail Decision Support

This case study preserves the complete project evolution: local model
serving, vector recall, retrieval gating, typed facts, overwrite control,
lifecycle-aware retrieval, product-level routing, evaluation, and the
later retail decision-support extension.

The application first-pass path begins with the current retail evidence and RAC implementation. The remaining sections document the memory-layer mechanisms for update control, traceability, active-state filtering, refusal behavior, and evidence boundaries.

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


## System Evolution

## Local LLM Backend

The initial implementation used local model serving for conversational product explanation and customer-facing interaction in a livestream-oriented setting.

## Raw Vector Chat Memory

Vector-based recall retained relevant past exchanges instead of discarding them entirely.

This made the system more context-aware, but it also exposed an important limitation: retrieval alone did not make memory reliable. Past content could be surfaced based on similarity without any clear standard for whether it should still matter.

## Retrieval Gating and Traceability

To address that limitation, I introduced retrieval gating and fallback or refusal conditions so that memory would only be used when there was enough evidence that it was genuinely relevant.

I also added traceable evidence in retrieval outputs so that memory usage could be inspected afterward rather than remaining hidden inside model behavior.

## Structured Facts and Typed Memory

Once retrieval became more controlled, the next issue became clearer: storing past text was still not the same as representing knowledge in a stable form.

Structured fact extraction and typed memory represent different categories of information with explicit handling rules.

## Update Policies and Overwrite Control

After typed memory was in place, I added overwrite logic so that newer facts could replace older active knowledge in a controlled way rather than letting conflicting information accumulate indefinitely.

Older facts were preserved through soft deactivation instead of being deleted without trace, which made memory updates easier to inspect and reason about.

## Lifecycle-Aware Retrieval

Lifecycle-aware memory treats stored facts as knowledge objects with timestamps, active-state flags, freshness windows, and reuse metadata.

This allowed retrieval to prefer currently valid knowledge while filtering out inactive or stale facts when appropriate.

## Livestream Commerce Knowledge Routing

The same memory architecture supports livestream commerce knowledge such as product price, promotions, stock status, shipping policy, and product features.

On top of that, I added policy-guided fact-type routing so that livestream questions could be matched against configured fact categories without requiring the caller to manually specify the knowledge type.

## Policy-Driven Memory Behavior

As the number of fact types increased, hard-coded rules for overwrite, freshness, routing thresholds, scope, and storage semantics became difficult to maintain.

A centralized in-code fact-policy registry keeps storage, retrieval, overwrite, freshness, routing, and scope behavior inspectable.

## Product-Level Entity Separation

Once livestream facts became more structured, it was no longer sufficient to store all product knowledge under a single default product context.

Lightweight product reference extraction attaches facts to specific product entities when possible and falls back to a default product scope when no explicit reference is available.

## Consolidated Extraction and Non-Fact Filtering

The earlier pipeline separated memory gating from fact extraction, which increased latency and could produce inconsistent behavior.

The current version consolidates this flow so that extraction itself determines whether a message yields a storable fact, while obvious non-fact messages such as greetings are filtered out before memory write.

## Explainable Fallback and Small-Scale Evaluation

At the current stage, the system can not only answer from stored knowledge, but also fall back safely when available knowledge is stale, inactive, unsupported, or insufficiently reliable.

Retrieval outputs expose routing decisions and filtered reasons, making failure cases easier to inspect rather than leaving them ambiguous.

To make the current behavior easier to verify, I added a scenario-based evaluation setup covering successful retrieval, fact-type routing, overwrite behavior, product-level separation, non-fact filtering, and unsupported-query fallback. Freshness filtering is implemented in the retrieval layer, while timestamp-controlled freshness tests remain a planned next step.

## Retail Operations Extension

The same lifecycle-aware memory principle is now applied to Meituan instant-retail operations data.

The practical problem is different from livestream product memory. Meituan's merchant backend provides rich single-store metrics, but multi-store operation requires a stricter question: which store-period records can be compared, under what limits, and which claims should be refused because the evidence is incomplete or not aligned.

The retail extension uses SQL and documented metric definitions before retrieval. The SQL layer organizes selected store-period, traffic, activity, search-term, and top-SKU evidence into same-period, contract-aligned diagnostic outputs.

The memory layer records store-period evidence, `calculation` metadata, confidence, and limitations so that March data is not casually mixed with April data, activity-heavy stores are not treated like low-activity stores, and lightweight top-SKU evidence is not overstated as full category-share analysis.

This is still a staged prototype. Its current purpose is not to automate final operating decisions, but to make cross-store interpretation more traceable, more cautious, and easier to verify as the number of stores increases.

