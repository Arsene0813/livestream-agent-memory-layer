# Retail Operations Experiment Review Map

This file maps the current retail operations checks to the business and data-science questions they are meant to answer.

It does not add a new demo, rename fields, change SQL output, or claim that the future pairwise comparability gate has been implemented.

The purpose is to make the current prototype easier to review: what each check uses as evidence, what it currently shows, and where interpretation should stop.

## Current Evidence Path

Meituan backend metric evidence
-> DATA_DICTIONARY.md field contract
-> SQL diagnostic output
-> generated retail memory facts
-> boundary-preserving answer checks
-> deterministic source-aware review scaffold

## Reviewer Map

| Analytical question | Evidence used | Method | Current result | Interpretation boundary |
|---|---|---|---|---|
| Can a single store's monthly operating movement be interpreted without reducing the result to one metric? | Store A February, March, and April 2026 store-period metrics and selected top-SKU evidence. | SQL month-over-month diagnostic over exposure, entry, ranking, transaction, conversion, activity, refund, invalid-order, and top-SKU signals. | Store A can be described through a multi-metric operating profile instead of a single-cause monthly explanation. | The result does not prove that any one metric caused the monthly change. |
| Can selected B-F store-period rows be structured before cross-store interpretation? | March 2026 B-F store-period metrics, top search-term evidence, and top-SKU transaction-amount evidence. | SQL derives search-entry structure, activity involvement, refund pressure, invalid-order pressure, top-3 SKU concentration, `comparison_scope_flag`, and `comparison_limit_notes`. | Demo 2 creates same-period diagnostic evidence under one field contract. | `comparison_scope_flag` is not a pairwise comparability decision, and `comparison_limit_notes` are diagnostic guardrails rather than peer-selection rules. |
| Can SQL diagnostic output become retrieval-facing memory facts without losing evidence boundaries? | Demo 2 SQL output, top search terms, top-SKU evidence, dictionary definitions, and lineage rules. | Generated memory facts preserve entity, period, slot, observed values, source fields, source paths, confidence labels, and limitations. | Current facts can support bounded Store B-F questions while keeping source paths and limitations visible. | Generated facts do not replace raw backend evidence or prove business causality. |
| Can unsupported claims be blocked or qualified? | Retail evaluation cases, generated facts, data dictionary, and lineage rules. | Scenario-based answer checks test unsupported ranking, ROI, strategy-transfer, region-type, and causal claims. | Current checks verify that answers stay inside entity, period, metric-definition, source, and interpretation boundaries. | Passing these checks does not prove general conversational robustness outside the current evidence path. |
| Are Demo 2 guardrail notes stable enough to become reusable peer-comparison rules? | Demo 2 guardrail sensitivity output. | The guardrail sensitivity script recomputes notes under baseline, looser, and stricter threshold scenarios. | Current notes are inspectable and threshold-sensitive. | The thresholds should remain diagnostic warnings until more store-period windows are added. |
| Can retrieval score behavior be inspected instead of justified from isolated examples? | Generated Demo 1 and Demo 2 retail facts, dictionary notes, source notes, and retrieval threshold cases. | Offline embedding-score inspection records top-k scores, margins, entity matches, slot matches, period checks, and expected-match behavior. | Retrieval score behavior is visible across supported, unsupported, hard-negative, mismatch, and ambiguous queries. | Retrieval score alone is not treated as sufficient evidence for operational conclusions. |
| Does wording variation change retrieval behavior too much? | Query robustness cases over the current file-backed retail evidence corpus. | Wording variants test shortened, paraphrased, noisy, typo-like, and keyword-order changes. | The current corpus can be inspected for retrieval behavior under small query variations. | This is not a production-level semantic robustness benchmark. |
| Can a factor-aware review path make reasoning limits visible before an answer is written? | Local project evidence routed through the deterministic RAC scaffold. | The scaffold expands factors, routes evidence, records boundary evidence, generates competing hypotheses, applies critique, and updates confidence with limitations. | Review reports make factor coverage, missing evidence, and unsupported claims visible. | The scaffold is a local deterministic review layer, not live backend ingestion or a completed pairwise comparability gate. |

## What This Adds for Reviewers

This map makes the current experiment structure easier to read without changing the implementation.

The main point is that the project is not trying to jump directly from limited backend metrics to final recommendations. It first tests whether metric definitions, reporting windows, source paths, and interpretation limits can survive SQL processing, memory-fact generation, retrieval, and answer review.

## Next Experimental Direction

The next data-science step should be repeated-window evidence before implementing a stronger pairwise comparability gate.

The repeated-window check should test whether order volume, transaction amount, activity involvement, activity intensity, refund pressure, invalid-order pressure, and top-SKU concentration remain stable across more store-period records, or whether the current March 2026 guardrail signals are mostly one-period artifacts.

Only after that should a future gate return question-specific decisions such as `comparable`, `comparable_with_limits`, `not_comparable`, or `insufficient_evidence`.
