# Retail Operations Experiment Map

This file records the current analytical experiments in the retail operations extension.

In this repository, "experiment" means a staged analytical check on whether the data path preserves metric definitions and answer boundaries.

It does not mean a randomized business experiment or causal A/B test.

## Current Experiment Scope

The current experiments test whether selected Meituan backend metrics can be:

1. structured into canonical fields;
2. processed through SQL diagnostics;
3. converted into retrieval-facing memory facts;
4. discussed without losing metric definitions or evidence limits.

The current experiments do not prove causal business effects.

## Planned Threshold Review

The current Demo 2 thresholds used to create `comparison_limit_notes` are fixture-stage literal thresholds for this prototype.

They are not estimated optimal cutoffs.

When broader store coverage and repeated reporting windows are added, these thresholds should be reviewed with stability checks and simple sensitivity analysis before being used in a stronger pairwise comparability gate.

## Demo 2 Guardrail Sensitivity Check

The Demo 2 thresholds used in `comparison_limit_notes` are prototype guardrails, not optimized business cutoffs. To make that boundary inspectable, the repository includes a small sensitivity check:

- Script: `retail_ops/scripts/analyze_demo2_guardrail_sensitivity.py`
- Output CSV: `retail_ops/outputs/demo2_guardrail_sensitivity_summary.csv`
- Output note: `retail_ops/outputs/demo2_guardrail_sensitivity_result.txt`

The check recomputes the current guardrail notes under three scenarios: baseline SQL thresholds, a looser minus-5-percentage-point setting, and a stricter plus-5-percentage-point setting. Its purpose is not to choose the best threshold. Its purpose is to show whether the current interpretation notes are fragile under small threshold changes.

A store whose notes change under this check should be treated as threshold-sensitive evidence. That means the current row can still be used for cautious diagnostic discussion, but it should not be used as a stable peer-comparison or strategy-transfer rule.

## Experiment 1: Store A Month-over-Month Diagnostic

| Item           | Content                                                                                                                                                                                                         |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Question       | Can a single store's monthly operating movement be interpreted without reducing the result to one metric?                                                                                                       |
| Input          | Store A February, March, and April 2026 store-period metrics; Store A top-SKU evidence.                                                                                                                         |
| Transformation | `01_store_a_month_over_month_diagnostic.sql` derives month-over-month movement, ranking changes, traffic and conversion tradeoffs, refund evidence and top-SKU concentration evidence. |
| Output         | `retail_ops/outputs/store_a_demo1_sql_output.csv`; generated Store A retail memory facts.                                                                                                                       |
| Pass condition | The system can describe exposure, entry, ranking, transaction, conversion, activity, refund evidence, and top-SKU movement together.                                                                      |
| Failure mode   | Claiming that exposure, ranking, activity, conversion, refund pressure, or top-SKU movement alone caused the monthly result.                                                                                    |

## Experiment 2: Demo 2 Same-Period Cross-Store Diagnostic

| Item           | Content                                                                                                                                                                                                                                                           |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Question       | Can selected B-F store-period rows be structured before cross-store interpretation?                                                                                                                                                                               |
| Input          | `demo2_store_period_metrics.csv`; top search-term evidence; top-SKU transaction-amount evidence.                                                                                                                                                                  |
| Transformation | `02_demo2_cross_store_comparability.sql` derives search-entry share/rate, activity-order share, refund evidence, top-3 SKU concentration, `comparison_scope_flag`, and `comparison_limit_notes`.                                          |
| Output         | `retail_ops/outputs/demo2_cross_store_comparability_output.csv`; generated Demo 2 retail memory facts.                                                                                                                                                            |
| Pass condition | The system can discuss stores only within the documented same-period diagnostic scope and must preserve limits related to region context, store type, activity involvement, refund evidence, product-mix evidence, and data completeness. |
| Failure mode   | Ranking stores globally, treating same-period diagnostic readiness as pairwise comparability, treating `activity_cost_ratio_pct` as ROI, or transferring a promotion, price, or SKU action without checking limits.                                               |

## Experiment 3: Retail Memory Fact Generation

| Item           | Content                                                                                                                                                                                                                                  |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Question       | Can SQL diagnostic outputs be converted into retrieval-facing memory facts without losing source fields, observed values, source paths, and limitations?                                                                                 |
| Input          | `retail_ops/outputs/demo2_cross_store_comparability_output.csv`; `retail_ops/data/demo2_top_search_terms.csv`; `retail_ops/data/demo2_top_skus_by_transaction_amount.csv`.                                                               |
| Transformation | `generate_demo2_retail_memory_facts.py` converts row-level diagnostics into slot-based retail memory facts.                                                                                                                              |
| Output         | `retail_ops/outputs/generated_demo2_retail_memory_facts.json`.                                                                                                                                                                           |
| Pass condition | Each generated fact keeps the store entity, period, slot, observed values, calculation notes, source fields, primary source path, supporting source paths, lineage path, confidence, limitations, active status, and period granularity. |
| Failure mode   | Mixing store-level and SKU-level fields, dropping source evidence, introducing undocumented fields, or letting top-search or top-SKU evidence appear without supporting source paths.                                                    |

## Experiment 4: Unsupported Claim Guard

| Item           | Content                                                                                                                                                                                                                            |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Question       | Does the system avoid overclaiming when the current evidence does not support a conclusion?                                                                                                                                        |
| Input          | Retail evaluation cases; Demo 2 generated facts; data dictionary; lineage rules.                                                                                                                                                   |
| Transformation | Scenario-based answer-behavior checks test whether retrieved evidence is used with the correct metric definitions and limitations.                                                                                                 |
| Output         | Retail evaluation result files under `eval/` and validation outputs under `retail_ops/outputs/`.                                                                                                                                   |
| Pass condition | The system qualifies or refuses unsupported claims about causal attribution, audited profit, full 48-store generalization, final store ranking, promotion decisions, pairwise store comparability, or full product-category share. |
| Failure mode   | Producing fluent but unsupported advice from isolated metrics, treating current Demo 2 as a completed pairwise decision system, or ignoring `comparison_limit_notes`.                                                              |

Additional endpoint-level check:

`eval/eval_retail_demo2_endpoint_behavior.py` verifies that the implemented `/chat_retail_ops_demo2_kb` endpoint returns file-backed Demo 2 facts for supported Store B-F questions and refuses unsupported all-48-store, best-store ranking, final operating recommendation, and out-of-scope entity questions.

## Experiment 5: Retrieval Threshold Calibration

| Item           | Content                                                                                                                                                                                                                                                                                                                                                                                                           |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Question       | Can the current retrieval threshold be explained from score distributions rather than isolated log examples?                                                                                                                                                                                                                                                                                                      |
| Input          | `eval/retrieval_threshold_cases.json`; generated Demo 1 and Demo 2 retail memory facts; selected field-contract notes such as `DATA_DICTIONARY.md` and `demo2_source_notes.md`.                                                                                                                                                                                                                                   |
| Transformation | `eval/analyze_retail_embedding_score_distribution.py` embeds each query and retrieval document with local Ollama `bge-m3`, retrieves top-k evidence, records scores, top-1/top-2 margins, expected matches, entity matches, slot matches, and period-scope checks. This is an offline retrieval-inspection path and should not be described as the runtime threshold for the current file-backed Demo 2 endpoint. |
| Output         | `retail_ops/outputs/retrieval_score_distribution.csv`; `retail_ops/outputs/retrieval_threshold_summary.md`; `retail_ops/outputs/retrieval_score_distribution.png`.                                                                                                                                                                                                                                                |
| Pass condition | The calibration makes retrieval behavior inspectable across supported, unsupported, hard-negative, entity/period mismatch, and ambiguous comparison queries.                                                                                                                                                                                                                                                      |
| Failure mode   | Treating a high retrieval score as sufficient evidence for an operating conclusion, or claiming a production-level threshold from the current small file-backed corpus.                                                                                                                                                                                                                                           |

## Next Data Science Experiment: Repeated-Window Comparability Evidence

The next analytical step is to add repeated reporting windows for more stores before implementing a stronger pairwise comparability gate.

Question:

Can the current guardrail signals remain stable across repeated months, or are they mostly one-period artifacts?

Required additional evidence:

- repeated store-period records;
- activity calendar or campaign-status evidence where available;
- repeated transaction-order and transaction-amount bands;
- local competition or price-pressure notes where available;
- broader SKU evidence beyond top-3 rows.

Pass condition:

The system can distinguish stable operating-context differences from one-month threshold-sensitive warnings.

Failure mode:

Turning one March 2026 threshold result into a reusable peer-comparison rule.

## Future Experiment: Pairwise Comparability Gate Contract Stub

| Item           | Content                                                                                                             |
| -------------- | ------------------------------------------------------------------------------------------------------------------- |
| Question       | Can the future pairwise comparability gate contract be frozen before implementation?                                |
| Input          | `retail_ops/COMPARABILITY_GATE_V0.md`.                                                                              |
| Transformation | A lightweight eval stub checks that the planned input triple, output enum, and blocking-factor list are documented. |
| Output         | `eval/eval_future_comparability_gate_contract.py`.                                                                  |
| Pass condition | The stub confirms the future contract exists while not claiming that the pairwise gate is implemented.              |
| Failure mode   | Treating Demo 2 row-level diagnostic readiness as a pairwise comparability decision.                                |

## Method Notes: What Demo 2 Guardrails Are Trying to Prevent

The Demo 2 thresholds are lightweight interpretation guardrails, not optimized business cutoffs.

Their role is to make possible over-interpretation visible before SQL outputs are converted into memory facts or used in answer-boundary checks.

| Guardrail signal                        | Current trigger in SQL                                                                      | Misreading it is meant to prevent                                                                                                                                              |
| --------------------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `comparison_scope_flag`                 | Period mismatch, missing required fields, or same-period diagnostic readiness               | Treating a row as usable when the reporting window or required evidence is incomplete.                                                                                         |
| `search_entry_share_pct`                | Kept as a directional search-entry structure metric.                                     | Reading search-entry share in isolation, without checking search exposure, search rank, entry conversion, order conversion, activity, refund evidence, store type, and region context. |
| `activity_order_share_pct`              | `>= 80` means `high_activity_involvement`; `>= 65` means `moderate_activity_involvement`    | Treating promotion-supported order structure as normal baseline demand.                                                                                                        |
| `refund_pressure_pct`                   | `>= 15` means `high_refund_pressure`; `>= 10` means `moderate_refund_pressure`              | Reading transaction amount as clean demand when refund pressure may weaken the interpretation.                                                                                 |
| `top3_sku_transaction_amount_share_pct` | `>= 25` means `top3_sku_amount_concentration`                                               | Treating a few high-value SKUs as if they described the full product-category structure.                                                                                       |
| `comparison_limit_notes`                | Concatenated guardrail notes                                                                | Letting a later answer ignore the limits already visible in the diagnostic row.                                                                                                |

These literal thresholds are deliberately simple at the current stage.

A future comparability gate should test their stability and sensitivity across more store-period records before treating them as peer-group or strategy-transfer rules.

## Proposed Repeated-Window Test Design

The next comparability experiment should use repeated store-period windows before turning Demo 2 guardrails into any reusable peer-comparison rule.

| Test target | Added evidence needed | What to check | Failure condition |
|---|---|---|---|
| Order-volume comparability | More store-period rows across multiple months | Whether `transaction_orders` bands remain stable enough to compare stores for a selected question | One-month order volume changes the interpretation too much |
| Transaction-scale comparability | Repeated `transaction_amount` and `average_order_value` records | Whether stores with similar order volume still differ strongly in transaction scale | Similar order count hides very different transaction structures |
| Activity involvement stability | Repeated `activity_orders`, `activity_order_share_pct`, and `activity_cost_ratio_pct` | Whether activity-heavy rows remain activity-heavy across months | One campaign month creates a false peer group |
| Region / market-context evidence | More stores plus local consumption, competition, or delivery-radius notes where available | Whether `region_type` is enough as weak context or new `market_area_type` fields are justified | Region labels alone explain too little |
| SKU-structure comparability | Broader SKU evidence beyond top-3 ranking views | Whether top-SKU concentration is stable enough to constrain comparison | Top-3 evidence changes too much or misses category structure |
| Refund-evidence stability | Repeated refund-evidence signals | Whether refund-related evidence is persistent or one-period noise | One-period refund-related evidence creates misleading store comparison |

## Reviewer Experiment Map

| Check | Business / data question | Data used | Saved output | What it supports | Boundary |
|---|---|---|---|---|---|
| Demo 1: Store A month-over-month diagnostic | Can one store's monthly movement be explained using backend metric evidence? | Store A, 2026-02 to 2026-04 | `retail_ops/outputs/store_a_month_over_month_diagnostic_output.csv` | Metric-preserving single-store diagnosis with source fields and limitation notes. | Does not prove cross-store comparability. |
| Demo 2: B-F same-period diagnostic | Can selected stores be reviewed under one reporting window and one field contract? | Stores B-F, 2026-03 | `retail_ops/outputs/demo2_cross_store_comparability_output.csv` | Same-period cross-store diagnostic structure and comparison-scope boundaries. | Does not implement a pairwise comparability gate. |
| Retrieval threshold calibration | Is the retrieval cutoff supported by saved evidence rather than arbitrary tuning? | Saved retail evidence and threshold cases | `retail_ops/outputs/retrieval_threshold_summary.md` | Offline inspection of retrieval score behavior for the current evidence set. | Does not claim a universal threshold for future data. |
| Repeated-window panel coverage | Do Stores B-F have enough repeated monthly evidence for descriptive panel review? | Stores B-F, 2026-02 to 2026-04 | `retail_ops/outputs/store_period_panel_coverage_output.csv` | Confirms repeated-window coverage before future comparability-gate work. | Does not rank stores or infer causality. |
| Repeated-window panel summary | How did selected metrics move from February to April for each store? | Stores B-F, 2026-02 to 2026-04 | `retail_ops/outputs/repeated_window_panel_summary_output.csv` | Descriptive movement summary under dictionary-aligned fields. | Not a recommendation engine, not a pairwise gate, and not a causal test. |
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
| Can a single store's monthly operating movement be interpreted without reducing the result to one metric? | Store A February, March, and April 2026 store-period metrics and selected top-SKU evidence. | SQL month-over-month diagnostic over exposure, entry, ranking, transaction, conversion, activity, refund evidence, and top-SKU signals. | Store A can be described through a multi-metric operating profile instead of a single-cause monthly explanation. | The result does not prove that any one metric caused the monthly change. |
| Can selected B-F store-period rows be structured before cross-store interpretation? | March 2026 B-F store-period metrics, top search-term evidence, and top-SKU transaction-amount evidence. | SQL derives search-entry structure, activity involvement, refund evidence, top-3 SKU concentration, `comparison_scope_flag`, and `comparison_limit_notes`. | Demo 2 creates same-period diagnostic evidence under one field contract. | `comparison_scope_flag` is not a pairwise comparability decision, and `comparison_limit_notes` are diagnostic guardrails rather than peer-selection rules. |
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

The repeated-window check should test whether order volume, transaction amount, activity involvement, activity intensity, refund evidence, and top-SKU concentration remain stable across more store-period records, or whether the current March 2026 guardrail signals are mostly one-period artifacts.

Only after that should a future gate return question-specific decisions such as `comparable`, `comparable_with_limits`, `not_comparable`, or `insufficient_evidence`.

## Repeated-Window Panel Extension Check

These checks verify whether the post-Demo-2 panel has enough clean repeated-window coverage to support later diagnostic work, then whether the covered panel can produce a descriptive February-to-April summary.

| Check | Question | Input | Pass condition | Boundary |
|---|---|---|---|---|
| Repeated-window panel validation | Do the current B-F stores each have February, March, and April 2026 panel rows under the same field contract? | `data/store_period_panel_metrics.csv`, `data/store_period_panel_source_notes.md`, `outputs/store_period_panel_coverage_output.csv` | `scripts/validate_store_period_panel.py` passes and each B-F store has exactly 2026-02, 2026-03, and 2026-04. | Coverage/readiness check only; not a pairwise comparability gate, endpoint test, memory-fact generation step, ranking, recommendation, or causal experiment. |
| Repeated-window panel summary validation | Can the covered B-F panel produce a descriptive February-to-April summary without using excluded order-status fields? | `sql/04_repeated_window_panel_summary.sql`, `outputs/repeated_window_panel_summary_output.csv` | `scripts/validate_repeated_window_panel_summary.py` passes, B-F each have 3 observed months, and each row is marked `summary_ready_for_descriptive_review`. | Descriptive summary only; not a store ranking, pairwise comparability decision, operating recommendation, endpoint behavior, generated memory fact, or causal analysis. |

The checks also verify that ambiguous order-status fields are excluded from the panel and summary logic:

| Excluded field | Reason |
|---|---|
