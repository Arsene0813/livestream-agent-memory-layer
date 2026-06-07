# Reviewer Guide

The fastest way to read this project is as a staged evidence path, not as an automation claim.

This guide is the fastest reading path for reviewing the current retail decision-support prototype.

## 1. What Problem This Repository Addresses

The project comes from a real multi-store Meituan instant-retail operating problem.

The Meituan merchant backend provides detailed single-store metrics, but it is mainly designed for reviewing one store at a time. With a 48-store operation, the harder question is not simply whether one metric increased or decreased.

The harder question is whether store-period records can be compared, under what conditions they can be compared, and which operating judgment the available evidence can support.

In this project, instant-retail competition is interpreted through this operating chain:

- being seen;
- being entered;
- being ordered;
- being selected again or maintaining share.

Promotion, subsidy, pricing, SKU arrangement, ranking position, and fulfillment stability are operating levers inside this chain. They are not interpreted as isolated goals.

## 2. What Is Implemented

The current implemented retail path is:

1. selected Meituan backend metrics;
2. canonical field dictionary;
3. SQL diagnostic output;
4. generated retail memory facts;
5. boundary-preserving answer checks;
6. deterministic source-aware review scaffold.

| Area | Current implementation |
|---|---|
| Data dictionary | Preserves Meituan-style backend metric meanings and canonical field names. |
| Demo 1 | Store A month-over-month diagnostic across February, March, and April 2026. |
| Demo 2 | Same-period B-F diagnostic for March 2026. |
| SQL diagnostics | Derives limited diagnostic fields such as activity involvement, refund pressure, invalid-order pressure, and top-SKU concentration. |
| Memory facts | Converts diagnostic outputs into source-bounded facts with observed values, source fields, source paths, confidence labels, and limitations. |
| Answer-boundary checks | Tests whether answers stay within entity, period, metric-definition, source, and interpretation boundaries. |
| RAC scaffold | Provides deterministic factor expansion, evidence routing, critique, fact checking, evidence-coverage update, and grounded report generation over local project evidence. |

## 3. Current Scope Boundary

The current prototype is intentionally scoped to evidence preparation, diagnostic structuring, and boundary-preserving review.

| Area | Current boundary |
|---|---|
| Full 48-store peer selection | Current demos use selected manually structured records. |
| Pairwise comparability gate | Future work only. Demo 2 is diagnostic evidence before such a gate. |
| Automated Meituan backend ingestion | Current data is manually structured from available backend evidence. |
| Causal promotion evaluation | Observed backend metrics do not prove causal effects. |
| Final operating recommendation system | Current outputs preserve evidence and limits; they do not approve strategy transfer. |
| Market-area classification | `region_type` remains weak context only. |
| RAC scaffold | Current RAC is deterministic and local-evidence-based. |

## 4. Fast Review Path

For admissions or project review, read in this order:

1. `PROJECT_SUMMARY_FOR_ADMISSIONS.md`
2. `retail_ops/FIELD_USAGE_REVIEW.md`
3. `retail_ops/data/DATA_DICTIONARY.md`
4. `retail_ops/demo/demo_2_cross_store_comparability_diagnostic.md`
5. `retail_ops/EXPERIMENT_RESULTS.md`
6. `retail_ops/EXPERIMENT_REVIEW_MAP.md`
7. `rac/DEMO_INDEX.md`

For technical audit, then inspect:

1. `retail_ops/ARCHITECTURE.md`
2. `retail_ops/LINEAGE.md`
3. `retail_ops/FIELD_USAGE_REVIEW.md`
4. `retail_ops/COMPARABILITY_GATE_V0.md`
5. `retail_ops/sql/`
6. `retail_ops/outputs/`
7. `eval/`
8. `rac/src/`
9. `rac/outputs/`

## 5. How To Read Demo 2

Demo 2 answers one narrow implementation question:

Can selected Stores B-F be organized under one March 2026 reporting window and one field contract without losing the original Meituan metric meanings?

Demo 2 supports cautious same-period diagnostic comparison. It does not decide whether one store is a peer, template, or strategy-transfer target for another store.

`comparison_scope_flag` is a row-level diagnostic-scope field.

`comparison_limit_notes` records operating context for interpretation, including activity involvement, refund pressure, invalid-order pressure, and selected SKU concentration. Search-entry structure remains visible through the dedicated search-entry fields.

## 6. How To Read Future Comparability Work

The future pairwise comparability gate should answer one narrow question:

Can these two store-period records be compared for this specific operating question?

The gate should consider transaction order volume, transaction amount, activity involvement, activity intensity, explicit campaign evidence if available, store type, market context, competition, SKU structure, refund pressure, invalid-order pressure, fulfillment or stockout evidence where available, and repeated reporting windows.

It should not produce a global store ranking or a universal comparability score.
