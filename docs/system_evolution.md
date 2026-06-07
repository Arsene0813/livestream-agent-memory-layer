# System Evolution

## Local LLM Backend

The project began as a simple local model-serving setup for conversational use in a livestream-oriented setting. The initial goal was to make local generation work reliably enough to support product explanation and customer-facing interaction.

## Raw Vector Chat Memory

The next step was to add vector-based recall so that past exchanges could be retrieved instead of being discarded entirely.

This made the system more context-aware, but it also exposed an important limitation: retrieval alone did not make memory reliable. Past content could be surfaced based on similarity without any clear standard for whether it should still matter.

## Retrieval Gating and Traceability

To address that limitation, I introduced retrieval gating and fallback or refusal conditions so that memory would only be used when there was enough evidence that it was genuinely relevant.

I also added traceable evidence in retrieval outputs so that memory usage could be inspected afterward rather than remaining hidden inside model behavior.

## Structured Facts and Typed Memory

Once retrieval became more controlled, the next issue became clearer: storing past text was still not the same as representing knowledge in a stable form.

The system therefore moved from unstructured recall toward structured fact extraction and typed memory, allowing different categories of information to be stored and handled more consistently.

## Update Policies and Overwrite Control

After typed memory was in place, I added overwrite logic so that newer facts could replace older active knowledge in a controlled way rather than letting conflicting information accumulate indefinitely.

Older facts were preserved through soft deactivation instead of being deleted without trace, which made memory updates easier to inspect and reason about.

## Lifecycle-Aware Retrieval

The project then evolved into a lifecycle-aware memory layer. Stored facts were no longer treated as timeless entries, but as knowledge objects with timestamps, active-state flags, freshness windows, and reuse metadata.

This allowed retrieval to prefer currently valid knowledge while filtering out inactive or stale facts when appropriate.

## Livestream Commerce Knowledge Routing

The next iteration extended the same memory architecture into livestream commerce knowledge. Structured facts such as product price, promotions, stock status, shipping policy, and product features could be written into typed memory and retrieved later through the same lifecycle-aware logic.

On top of that, I added policy-guided fact-type routing so that livestream questions could be matched against configured fact categories without requiring the caller to manually specify the knowledge type.

## Policy-Driven Memory Behavior

As the number of fact types increased, hard-coded rules for overwrite, freshness, routing thresholds, scope, and storage semantics became difficult to maintain.

To address this, the system moved toward a centralized in-code fact-policy registry, making storage and retrieval behavior easier to extend and reason about.

## Product-Level Entity Separation

Once livestream facts became more structured, it was no longer sufficient to store all product knowledge under a single default product context.

The system therefore introduced lightweight product reference extraction so that facts could be attached to specific product entities when possible, while still falling back to a default product scope when no explicit reference was available.

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

The retail extension therefore uses SQL and documented metric definitions before retrieval. The SQL layer organizes selected store-period, traffic, activity, refund, invalid-order, search-term, and top-SKU evidence into same-period, contract-aligned diagnostic outputs.

The memory layer records store-period evidence, calculation notes, confidence, and limitations so that March data is not casually mixed with April data, activity-heavy stores are not treated like low-activity stores, and lightweight top-SKU evidence is not overstated as full category-share analysis.

This is still a staged prototype. Its current purpose is not to automate final operating decisions, but to make cross-store interpretation more traceable, more cautious, and easier to verify as the number of stores increases.
