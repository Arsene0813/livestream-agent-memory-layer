# Retail Operations Experiment Map and Results

This file records the retail experiment map, validation results, and boundary checks for the current decision-support prototype.

## First-Pass Reviewer Matrix
<<<<<<< HEAD

| Experiment                          | Question                                                                                                   | Input                                                                    | Output                                                                                | What it prevents                                                                       |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Field-contract validation           | Are field names and metric meanings stable across the retail path?                                         | `DATA_DICTIONARY.md`, source CSV files, SQL outputs, generated facts     | Pass/fail validation of names, meanings, and fact structure                           | Alias drift, silent metric redefinition, unsupported generated-fact fields             |
| Demo 1 month-over-month diagnostic  | Can one store be described across repeated months without reducing the result to one metric?               | Store A February-April 2026 backend-derived metrics                      | Multi-metric diagnostic output and memory facts                                       | Single-metric attribution, causal overclaim, month-as-good-or-bad labeling             |
| Demo 2 same-period diagnostic       | Can selected B-F store-period rows be staged under one March 2026 reporting window and one field contract? | Stores B-F March 2026 metric records, top search terms, top-SKU evidence | Row-level diagnostic output with `comparison_scope_flag` and `comparison_limit_notes` | Store ranking, premature pairwise comparability, strategy-transfer claims              |
| Answer-boundary checks              | Do later answers preserve entity, period, metric-definition, source, and comparison limits?                | Generated retail memory facts and boundary test cases                    | Scenario-level pass/fail behavior                                                     | Unsupported recommendations, period mismatch, entity mismatch, ROI/profit overclaim    |
| Retrieval and robustness inspection | Does wording variation still retrieve the intended evidence path?                                          | Local file-backed retail evidence corpus and query variants              | Score-distribution and query-robustness inspection                                    | Fluent answers hiding weak or mismatched evidence                                      |
| Repeated-window B-F panel           | Is there enough repeated store-period coverage to prepare future question-specific comparison rules?       | Stores B-F February-April 2026 panel records                             | Coverage output and descriptive repeated-window summary                               | Premature gate claims, store ranking, causal interpretation from short-window evidence |
| Future comparability-gate contract  | What should the next pairwise decision layer decide, and what should it refuse?                            | Current evidence boundaries and planned gate design                      | Question-specific future gate contract                                                | Treating current Demo 2 as a completed pairwise gate                                   |


## Experiment Dependency Map
=======
>>>>>>> 1bdced6 (Refine retail experiment narrative and dictionary boundaries)

| Layer | Review question | Evidence output | Boundary protected |
|---|---|---|---|
| Field contract | Are field names and metric meanings stable? | Dictionary, source headers, SQL outputs, generated-fact structure | Alias drift and silent metric redefinition |
| SQL diagnostics | Can selected store-period records be structured without changing backend meanings? | Demo 1 month-over-month output and Demo 2 same-period B-F output | Single-metric attribution, store ranking, premature comparison |
| Memory facts and answer boundaries | Can later answers preserve entity, period, source, metric, and limitation fields? | Generated retail memory facts, answer-behavior checks, endpoint checks | Unsupported advice, ROI/profit overclaim, scope mismatch |
| Retrieval and robustness inspection | Does wording variation still route to the intended evidence path? | Score-distribution and query-robustness outputs | Fluent answers hiding weak or mismatched evidence |
| Repeated-window panel | Is there repeated B-F evidence before stronger comparison rules are attempted? | February-April 2026 coverage and descriptive summary outputs | Premature gate claims, causal interpretation, strategy-transfer approval |
| RAC grounded review | Can multi-factor reasoning remain inspectable over local evidence? | Grounded review outputs with confidence, limitations, source paths, and evidence snippets | Hidden evidence jumps and overconfident synthesis |
| Future gate contract | What should the next pairwise decision layer decide or refuse? | Future comparability-gate contract stub | Treating current Demo 2 as a completed pairwise gate |

## How to Read the Experiments

The experiments should be read as one staged evidence path rather than as
separate engineering features:

```text
selected Meituan backend metrics
-> canonical field dictionary
-> SQL diagnostic outputs
-> generated retail memory facts
-> answer-boundary checks
-> retrieval and robustness inspection
-> repeated-window evidence review
-> RAC grounded review over local evidence
```

The early checks protect field names, metric meanings, source paths, and
generated-fact structure. Demo 1 and Demo 2 then test whether selected
store-period evidence can be turned into diagnostic outputs without changing
backend metric definitions.

The answer-boundary, endpoint-boundary, retrieval, and robustness checks test
whether later answers preserve entity, period, source, metric-definition, and
comparison limits.

The repeated-window B-F panel extends the evidence base across February,
March, and April 2026. It supports coverage and stability inspection for
future question-specific comparison rules, while leaving pairwise
comparability decisions, store ranking, strategy-transfer approval, causal
attribution, and all-48-store rollout outside the current implemented scope.

RAC remains an important technical part of the project. Its role here is to
make multi-factor operating reasoning inspectable over local evidence through
factor expansion, evidence routing, critique, fact checks, confidence updates,
limitations, source paths, and local evidence snippets.

## Experiment 1: Store A Month-over-Month Diagnostic

| Item | Content |
|---|---|
| Question | Can selected Meituan backend metrics for one store be organized into a month-over-month diagnostic without changing backend metric meanings or reducing the result to one metric? |
| Input | Store A February, March, and April 2026 store-period metrics; Store A top-SKU evidence. |
| Transformation | `retail_ops/sql/01_store_a_month_over_month_diagnostic.sql` derives month-over-month movement, ranking changes, traffic and conversion tradeoffs, and top-SKU concentration evidence. |
| Output | `retail_ops/outputs/store_a_demo1_sql_output.csv`; `retail_ops/outputs/generated_retail_memory_facts.json`. |
| Expected behavior | The output may describe observed month-over-month movement, but it should not attribute performance change to one metric alone. |
| Current result | Passed current validation check. |
| Checked by | `python3 retail_ops/scripts/validate_retail_data_contract.py` |

## Experiment 2: Demo 2 Same-Period Store Diagnostic

| Item | Content |
|---|---|
| Question | Can selected B-F store-period rows be placed under one March 2026 reporting window and one field contract before any stronger comparison is attempted? |
| Input | `retail_ops/data/demo2_store_period_metrics.csv`; top search-term evidence; top-SKU transaction-amount evidence. |
| Transformation | `retail_ops/sql/02_demo2_cross_store_comparability.sql` derives search-entry share/rate, activity-order share, top-3 SKU concentration, `comparison_scope_flag`, and `comparison_limit_notes`. |
| Output | `retail_ops/outputs/demo2_cross_store_comparability_output.csv`; `retail_ops/outputs/generated_demo2_retail_memory_facts.json`. |
| Expected behavior | The SQL output should include `comparison_scope_flag` and `comparison_limit_notes`, while staying at row-level same-period diagnostic scope. |
| Current result | Passed current Demo 2 scope-boundary validation. |
| Checked by | `python3 eval/eval_retail_demo2_scope_boundary.py` |
| Result path | `eval/retail_decision_support_eval_results/eval_retail_demo2_scope_boundary_result.txt` |
| Failure mode | Ranking stores globally, treating same-period diagnostic readiness as pairwise comparability, treating `activity_cost_ratio_pct` as ROI, or transferring a promotion, price, or SKU action without checking limits. |

## Experiment 2A: Demo 2 Guardrail Sensitivity Check

| Item | Content |
|---|---|
| Question | Are the current Demo 2 `comparison_limit_notes` stable under small threshold changes? |
| Input | `retail_ops/outputs/demo2_cross_store_comparability_output.csv`. |
| Transformation | `retail_ops/scripts/analyze_demo2_guardrail_sensitivity.py` recomputes current guardrail notes under baseline SQL thresholds, a looser minus-5-percentage-point setting, and a stricter plus-5-percentage-point setting. |
| Output | `retail_ops/outputs/demo2_guardrail_sensitivity_summary.csv`; `retail_ops/outputs/demo2_guardrail_sensitivity_result.txt`. |
| Expected behavior | The check should not optimize thresholds or turn them into peer-selection rules. It should only show whether the current threshold-based guardrail notes are sensitive to small threshold shifts. |
| Current result | Completed current guardrail sensitivity inspection. In the current B-F sample, all five stores have guardrail notes that change under at least one +/- 5 percentage-point sensitivity scenario. |
| Interpretation | The current thresholds should be treated as diagnostic warnings, not stable peer-comparison rules. |
| Checked by | `python3 retail_ops/scripts/analyze_demo2_guardrail_sensitivity.py` |

## Experiment 3: Demo 2 Memory-Fact Generation

| Item | Content |
|---|---|
| Question | Can the Demo 2 diagnostic output be converted into retrieval-facing memory facts without losing source fields, observed values, source paths, or limitation notes? |
| Input | `retail_ops/outputs/demo2_cross_store_comparability_output.csv`; `retail_ops/data/demo2_top_search_terms.csv`; `retail_ops/data/demo2_top_skus_by_transaction_amount.csv`. |
| Transformation | `retail_ops/scripts/generate_demo2_retail_memory_facts.py` converts row-level diagnostics into slot-based retail memory facts. |
| Output | `retail_ops/outputs/generated_demo2_retail_memory_facts.json`. |
| Expected behavior | Generated facts should preserve store entity, period, slot, observed values, calculation notes, source fields, primary source path, supporting source paths, lineage path, confidence, limitations, active status, and period granularity. |
| Current result | Passed current generated-fact evaluation. |
| Checked by | `python3 eval/eval_retail_demo2_facts.py` |
| Result path | `eval/retail_decision_support_eval_results/eval_retail_demo2_facts_result.txt` |
| Failure mode | Mixing store-level and SKU-level fields, dropping source evidence, introducing undocumented fields, or letting top-search or top-SKU evidence appear without supporting source paths. |

Implemented Demo 2 evidence slots:

| Slot | Review role |
|---|---|
| `visibility_entry_profile` | Exposure, ranking, entry, and search-entry structure. |
| `activity_lever_profile` | Activity orders, activity cost, subsidy, and activity-cost ratio. |
| `transaction_conversion_profile` | Transaction scale, order conversion, payment, and average order value. |
| `top3_sku_product_mix_note` | Lightweight top-SKU concentration evidence from selected ranking views. |
| `single_metric_attribution_guard` | Guardrail against explaining performance from one metric alone. |

## Experiment 4: Demo 2 Answer-Boundary Contract Check

| Item | Content |
|---|---|
| Question | Can expected answer patterns preserve metric boundaries when Demo 2 evidence is used? |
| Input | Retail evaluation cases; Demo 2 generated facts; data dictionary; lineage rules. |
| Transformation | Scenario-based answer-behavior checks test whether retrieved evidence is used with the correct metric definitions and limitations. |
| Output | Retail evaluation result files under `eval/` and validation outputs under `retail_ops/outputs/`. |
| Expected behavior | The system qualifies or refuses unsupported claims about causal attribution, audited profit, full 48-store generalization, final store ranking, promotion decisions, pairwise store comparability, or full product-category share. |
| Current result | Passed current offline scenario checks. |
| Checked by | `python3 eval/eval_retail_demo2_answer_behavior.py` |
| Result path | `eval/retail_decision_support_eval_results/eval_retail_demo2_answer_behavior_result.txt` |
| Failure mode | Producing fluent but unsupported advice from isolated metrics, treating current Demo 2 as a completed pairwise decision system, or ignoring `comparison_limit_notes`. |

Answer-boundary rules checked here:

| Boundary | Rule |
|---|---|
| Activity cost ratio | `activity_cost_ratio_pct` is not traditional ROI or profit margin. |
| Top-SKU share | `top3_sku_transaction_amount_share_pct` is not full product-category sales share. |
| Search-entry evidence | Search-entry evidence is one visibility-to-entry signal within the current store-period profile. |
| Activity evidence | Activity evidence describes operating-tool usage, not automatic promotion-transfer logic. |
| Same-period readiness | `same_period_diagnostic_ready` is not a finished pairwise comparability decision. |
| Region context | `region_type` is weak context only. |

## Experiment 4A: Demo 2 Endpoint-Boundary Contract Check

| Item | Content |
|---|---|
| Question | Does the implemented `/chat_retail_ops_demo2_kb` endpoint preserve the same evidence boundaries when answering from file-backed Demo 2 memory facts? |
| Input | `/chat_retail_ops_demo2_kb`; `api/main.py`; `retail_ops/outputs/generated_demo2_retail_memory_facts.json`. |
| Transformation | Endpoint-level evaluation checks supported Store B-F questions, unsupported all-48-store questions, best-store ranking requests, final operating-recommendation requests, and out-of-Demo-2 entity questions. |
| Output | `eval/retail_decision_support_eval_results/eval_retail_demo2_endpoint_behavior_result.txt`. |
| Expected behavior | Supported Store B-F questions return file-backed Demo 2 facts; cross-store B-F questions stay at same-period diagnostic scope; unsupported scope or final-decision requests are refused or qualified. |
| Current result | Passed current endpoint-level boundary checks. |
| Checked by | `python3 eval/eval_retail_demo2_endpoint_behavior.py` |

## Experiment 5: Retrieval Threshold Inspection

| Item | Content |
|---|---|
| Question | Can the current retrieval threshold be explained from score distributions rather than isolated successful examples? |
| Input | `eval/retrieval_threshold_cases.json`; generated Demo 1 and Demo 2 retail memory facts; selected field-contract notes such as `DATA_DICTIONARY.md` and `demo2_source_notes.md`. |
| Transformation | `eval/analyze_retail_embedding_score_distribution.py` embeds each query and retrieval document with local Ollama `bge-m3`, retrieves top-k evidence, records scores, top-1/top-2 margins, expected matches, entity matches, slot matches, and period-scope checks. |
| Output | `retail_ops/outputs/retrieval_score_distribution.csv`; `retail_ops/outputs/retrieval_threshold_summary.md`; `retail_ops/outputs/retrieval_score_distribution.png`. |
| Expected behavior | The calibration makes retrieval behavior inspectable across supported, unsupported, hard-negative, entity/period mismatch, and ambiguous comparison queries. |
| Current result | Completed as an offline small-corpus retrieval inspection. |
| Boundary | Offline calibration reference only. Retrieval scores make evidence routing inspectable, but final operating claims still require entity, period, source-path, metric-definition, and interpretation-boundary checks. |
| Failure mode | Treating a high retrieval score as sufficient evidence for an operating conclusion, or claiming a production-level threshold from the current small file-backed corpus. |

## Experiment 6: Query Robustness Inspection

| Item | Content |
|---|---|
| Question | Does retrieval behavior remain reasonably stable when the same query intent is expressed with small wording changes? |
| Input | Current retail retrieval-inspection cases and wording variants. |
| Transformation | `eval/analyze_retail_query_robustness.py` tests shortened, paraphrased, typo/noise, and keyword-order query variants. |
| Output | `retail_ops/outputs/retrieval_query_robustness.csv`; `retail_ops/outputs/retrieval_query_threshold_sweep.csv`; `retail_ops/outputs/retrieval_query_robustness_summary.md`. |
| Expected behavior | Supported cases should generally retain expected evidence in top-k under small wording changes. Unsupported, hard-negative, entity/period-mismatch, and ambiguous comparison cases should still require entity, period, slot, source-path, and interpretation-boundary checks. |
| Current result | Completed as an offline small-corpus robustness inspection. |
| Boundary | Retrieval score is only one signal and should be paired with answer-boundary checks. |

## Experiment 7: Repeated-Window Evidence Check

| Item | Content |
|---|---|
| Question | Can the current guardrail signals remain stable across repeated months, or are they mostly one-period artifacts? |
| Input | B-F store-period records across 2026-02, 2026-03, and 2026-04. |
| Output | `retail_ops/outputs/store_period_panel_coverage_output.csv`; `retail_ops/outputs/repeated_window_panel_summary_output.csv`. |
| Current result | Implemented as coverage and descriptive summary evidence. |
| Current use | Preparation for future question-specific pairwise comparability checks. |
| Failure mode | Turning one March 2026 threshold result into a reusable peer-comparison rule. |

Evidence still missing for a stronger gate:

| Missing evidence | Why it matters |
|---|---|
| Broader repeated store-period records beyond the current B-F three-month panel | Needed before treating guardrails as stable across more stores and months. |
| Activity calendar or campaign-status evidence | Needed before treating activity involvement as full campaign status. |
| Repeated transaction-order and transaction-amount bands | Needed before robust question-specific peer comparison. |
| Local competition or price-pressure notes | Needed before promotion, pricing, or market-share interpretations. |
| Broader SKU evidence beyond top-3 rows | Needed before full product-mix or product-category claims. |

## Experiment 8: Future Gate Contract Check

| Item | Content |
|---|---|
| Question | Can the project document a future pairwise comparability gate without accidentally exposing it as a finished current feature? |
| Input | `retail_ops/COMPARABILITY_GATE_V0.md`. |
| Transformation | `eval/eval_future_comparability_gate_contract.py` checks that the planned input triple, output enum, and blocking-factor list are documented. |
| Output | `eval/retail_decision_support_eval_results/eval_future_comparability_gate_contract_result.txt`. |
| Expected behavior | The future gate may define planned factors such as transaction order volume, transaction amount, activity status, activity intensity, store type, region and market context, SKU structure, and repeated reporting windows. It should not appear as a current implemented gate in Demo 2 outputs. |
| Current result | Documented and checked by the future-gate contract stub. |
| Checked by | `python3 eval/eval_future_comparability_gate_contract.py` |
| Failure mode | Treating Demo 2 row-level diagnostic readiness as a pairwise comparability decision. |

## Method Notes: Demo 2 Guardrails

The Demo 2 thresholds are lightweight interpretation guardrails, not optimized business cutoffs.

The current sensitivity check exists because a future pairwise comparability gate should not inherit one-off threshold choices without repeated-window evidence.

| Item | Meaning |
|---|---|
| Baseline | Current SQL output values and current `comparison_limit_notes`. |
| Looser threshold | Recomputed notes after making selected guardrails 5 percentage points easier to trigger. |
| Stricter threshold | Recomputed notes after making selected guardrails 5 percentage points harder to trigger. |
| Threshold-sensitive row | A store row whose note set changes under at least one threshold scenario. |

Current result: all current B-F rows are threshold-sensitive under at least one scenario. This supports the current decision to treat Demo 2 guardrails as diagnostic warnings rather than stable pairwise-comparison rules.
