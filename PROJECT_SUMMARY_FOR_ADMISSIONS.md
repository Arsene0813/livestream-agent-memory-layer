# Project Summary for Admissions

This repository is a staged local prototype for turning a real Meituan instant-retail operating problem into a more traceable decision-support workflow.

The project grew from a 48-store Meituan instant-retail operation. The merchant backend provides detailed single-store metrics, but it does not directly answer cross-store decision questions. The practical problem is not simply whether one store performed better than another. The harder problem is whether two store-period records can be compared at all, under what evidence limits, and whether differences in exposure, entry, conversion, activity involvement, refund pressure, invalid-order pressure, or product mix can support any operating interpretation.

The current implementation does not claim to solve the full 48-store decision problem. It preserves metric definitions, structures selected store-period data through SQL, converts diagnostic evidence into generated memory facts with source fields and limitations, and checks that later answers remain inside the documented evidence boundary.

## Business Origin

The retail context is Meituan instant retail for standardized optical products such as contact lenses, care solutions, and related eye-care items.

The operating problem is organized around one chain:

| Operating step | Practical meaning |
|---|---|
| Being seen | Whether the store and products receive enough exposure through search, ranking, listing position, and platform traffic. |
| Being entered | Whether exposure turns into store visits or search-related visits. |
| Being ordered | Whether visits turn into submitted and paid orders under current product, price, activity, and fulfillment conditions. |
| Being selected again or maintaining share | The longer-term operating goal that motivates future data collection, but is not directly measured in the current demos. |

In this setting, promotions, subsidies, pricing, SKU arrangement, ranking optimization, and fulfillment stability are operating levers. They matter only when they help explain where a store sits in this chain and what evidence is still missing before a stronger operating judgment can be made.

## Why This Became a Data-Science Problem

The backend data is detailed, but it is organized mainly for single-store monitoring. That makes cross-store analysis inconvenient, and in many cases the stores are not directly comparable without additional context.

A store with high activity involvement should not be read the same way as a store with low activity involvement. A store with strong search entry may still have refund or invalid-order pressure. A store in one regional context should not be treated as automatically comparable to another store only because both belong to the same business. A single month of evidence is also not enough to infer stable market-area classification or transferable strategy.

The data-science problem is therefore not to produce a global store ranking. It is to preserve definitions, structure evidence, detect comparison limits, and avoid turning weakly comparable records into confident operating advice.

## Prototype Layers

| Layer | What it does |
|---|---|
| Metric dictionary | Preserves Meituan backend field meanings and canonical project field names. |
| SQL diagnostic layer | Converts selected store-period backend data into comparable diagnostic outputs. |
| Generated memory facts | Stores store-period evidence with source paths, source fields, observed values, confidence, and interpretation limits. |
| Evaluation checks | Verifies that answers preserve metric definitions and refuse unsupported comparison claims. |
| Future comparability gate | Planned pairwise gate for deciding whether two store-period records can be compared for a specific analytical question. |

The important design choice is that the memory layer does not replace the data. It records evidence and limitations so that later retrieval or answer generation does not flatten different stores, periods, activity conditions, product structures, or regional contexts into the same comparison.

## Current Implementation

The current retail implementation stops at Demo 2.

Demo 1 is a Store A month-over-month diagnostic using selected 2026-02, 2026-03, and 2026-04 evidence. The purpose is to preserve a careful operating profile, not to label a month as simply good or bad.

Demo 2 is a same-period B-F diagnostic for March 2026. It structures selected store-period evidence under one reporting window and one field contract before any stronger pairwise comparability decision is made.

Demo 2 is not a completed pairwise comparability gate. It is a controlled diagnostic step that exposes why a future gate is needed.

## Implemented Checks

| Implemented check | What it confirms | Current boundary |
|---|---|---|
| Data dictionary contract | Meituan backend metrics are preserved under explicit field names and definitions. | Current data is manually structured from selected backend evidence. |
| Demo 2 SQL diagnostics | Selected B-F March 2026 records can be structured under one reporting window and field contract. | Output remains diagnostic evidence before peer selection, ranking, or strategy transfer. |
| Generated memory facts | Diagnostic evidence can carry source fields, observed values, source paths, confidence, and limitations. | Facts preserve source references and interpretation limits. |
| Answer-boundary evaluations | Later answers can be checked for entity scope, period scope, metric definitions, and unsupported-comparison refusal. | Results are scoped to the current evidence path. |
| Retrieval score inspection | Retrieval behavior can be inspected across supported, unsupported, hard-negative, mismatch, and ambiguous queries. | Retrieval score is inspection evidence, not standalone operating evidence. |
| Grounded RAC quality gate | Factor-specific evidence routing, boundary evidence, critique, and report grounding can be checked. | It remains a deterministic review scaffold over local evidence. |

This table is included to show what has actually been checked. It is not meant to inflate the current scope: the implemented checks still operate over selected local evidence, not a full live Meituan data pipeline.

## Evidence Boundaries

The project keeps several boundaries explicit:

| Field | Boundary |
|---|---|
| `region_type` | Weak region or market-context evidence only; not a hard market-area classification, store-stage label, or peer-store grouping rule. |
| `activity_cost_ratio_pct` | Activity-cost intensity metric; not a traditional ROI metric. |
| `top3_sku_transaction_amount_share_pct` | Lightweight top-SKU concentration signal; not full product-category sales share. |
| `comparison_scope_flag` | Diagnostic readiness or scope flag; not a pairwise store-matching decision. |
| Same-period diagnostic evidence | Useful for controlled interpretation, but not enough by itself to justify strategy transfer between stores. |
| Current system scope | Does not claim full 48-store automation or production recommendation ability. |

These boundaries are part of the decision-support logic. Unsupported comparisons should be qualified or refused rather than turned into confident operating advice.

## Future Work: Pairwise Comparability Gate

The next planned retail stage is a pairwise comparability gate.

The gate should decide whether two store-period records can be compared for a specific analytical question. It should consider transaction order volume, transaction amount, activity involvement and intensity, explicit campaign evidence when available, store type, region or market context, competition, SKU structure, refund pressure, invalid-order pressure, fulfillment or stockout evidence where available, and repeated reporting windows.

This future gate should not produce a global store ranking or universal comparability score. Its purpose is narrower: prevent unsupported cross-store comparison before a later diagnostic or operating interpretation is made.

For example, a new store using heavy activity subsidies to gain visibility should not be judged by the same assumptions as a mature store defending market share under different competitive pressure. Similarly, regional context should not be classified by intuition alone; stronger market-area classification should wait for more store data and external consumption-context evidence.

## Why It Matters

The project matters because the original business problem is not a lack of data. The Meituan backend already provides detailed store-level metrics. The harder problem is that detailed single-store metrics do not automatically become reliable cross-store decision evidence.

For a 48-store operator, the useful question is not simply which store looks better. The useful question is which store-period records can be compared, which differences are meaningful under the current evidence, and which comparisons should be qualified because of activity intensity, store type, region context, refund pressure, invalid-order pressure, product mix, competition, or missing repeated-window evidence.

The current prototype shows a staged path:

1. preserve backend metric definitions;
2. structure selected store-period evidence through SQL;
3. generate source-backed memory facts;
4. evaluate whether later answers stay inside the evidence boundary;
5. document the future pairwise comparability gate needed before stronger cross-store comparison.

The current stage is deliberately limited. It does not automate final operating decisions. It first makes the evidence path inspectable enough to say what is supported, what is only diagnostic, and what still needs more data.

For implementation details, the repository README and the `retail_ops/` documents provide the data dictionary, SQL diagnostics, generated outputs, validation scripts, and future comparability-gate contract.
