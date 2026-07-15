# Retail Operations Experiment Map and Results

This file records the retail experiment map, validation results, and boundary checks for the current decision-support prototype.

## First-Pass Reviewer Matrix


| Experiment                          | Question                                                                                                   | Input                                                                    | Output                                                                                | What it prevents                                                                       |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| Field-contract validation           | Are field names and metric meanings consistent across the retail path?                                         | `DATA_DICTIONARY.md`, source CSV files, SQL outputs, generated facts     | Pass/fail validation of names, meanings, and fact structure                           | Alias drift, silent metric redefinition, unsupported generated-fact fields             |
| Demo 1 month-over-month diagnostic  | Can one store be described across repeated months without reducing the result to one metric?               | Store A February-April 2026 backend-derived metrics                      | Multi-metric diagnostic output and memory facts                                       | Single-metric attribution, causal overclaim, month-as-good-or-bad labeling             |
| Demo 2 same-period diagnostic       | Can selected B-F store-period rows be staged under one March 2026 reporting window and one field contract? | Stores B-F March 2026 metric records, top search terms, top-SKU evidence | Row-level diagnostic output with `comparison_scope_flag` and `comparison_limit_notes` | Store ranking, premature pairwise comparability, strategy-transfer claims              |
| Answer-boundary checks              | Do later answers preserve entity, period, metric-definition, source, and comparison limits?                | Generated retail memory facts and boundary test cases                    | Scenario-level pass/fail behavior                                                     | Unsupported recommendations, period mismatch, entity mismatch, ROI/profit overclaim    |
| Retrieval and robustness inspection | Does wording variation still retrieve the intended evidence path?                                          | Local file-backed retail evidence corpus and query variants              | Score-distribution and query-robustness inspection                                    | Fluent answers hiding weak or mismatched evidence                                      |
| Repeated-window B-F panel           | Is there enough repeated store-period coverage to prepare future question-specific comparison rules?       | Stores B-F February-April 2026 panel records                             | Coverage output and descriptive repeated-window summary                               | Premature gate claims, store ranking, causal interpretation from short-window evidence |
| Future comparability-gate contract  | What should the next pairwise decision layer decide, and what should it refuse?                            | Current evidence boundaries and planned gate design                      | Question-specific future gate contract                                                | Treating current Demo 2 as a completed pairwise gate                                   |


## Experiment Dependency Map



| Layer | Review question | Evidence output | Boundary protected |
|---|---|---|---|
| Field contract | Are field names and metric meanings consistent? | Dictionary, source headers, SQL outputs, generated-fact structure | Alias drift and silent metric redefinition |
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
March, and April 2026. It verifies repeated store-period coverage and reports descriptive movement in selected metrics. Monthly guardrail stability is a separate future test because the current panel does not contain repeated top-SKU evidence or monthly recomputed `comparison_limit_notes`.

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
| Transformation | `retail_ops/scripts/analyze_demo2_guardrail_sensitivity.py` reads the implemented thresholds from the current SQL, reproduces every current-row `comparison_limit_notes` value, then lowers and raises those thresholds by 5 percentage points. |
| Output | `retail_ops/outputs/demo2_guardrail_sensitivity_summary.csv`; `retail_ops/outputs/demo2_guardrail_sensitivity_result.txt`. |
| Expected behavior | The check should not optimize thresholds or turn them into peer-selection rules. It should only show whether the current threshold-based guardrail notes are sensitive to small threshold shifts. |
| Current result | Completed current guardrail sensitivity inspection. Baseline reproduction passes for all five rows. Four rows (C, D, E, and F) change under the harder-to-trigger +5 percentage-point scenario; the easier-to-trigger -5 percentage-point scenario produces no note changes, and Store B remains unchanged. |
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
| Current result | Passed 25/25 entity-slot contracts across five stores and five implemented slots. The evaluation checks exact coverage, required fact metadata, repository-backed paths, and slot-specific boundary terms; schema and dictionary consistency remain covered by `validate_retail_data_contract.py`. |
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

## Experiment 4: Demo 2 Answer-Contract Fixture Validation

| Item | Content |
|---|---|
| Question | Do six manually specified answer contracts preserve the documented Demo 2 metric and comparison boundaries? |
| Input | Six fixed answer-contract fixtures; Demo 2 SQL output and generated facts used for setup checks. |
| Transformation | `eval/eval_retail_demo2_answer_behavior.py` checks each fixed contract text for required and forbidden terms. It does not call `/chat_retail_ops_demo2_kb` and does not generate answers. |
| Output | `eval/retail_decision_support_eval_results/eval_retail_demo2_answer_behavior_result.txt`. |
| Expected behavior | Each fixture preserves the relevant metric definition and scope limit for activity-cost ratio, top-SKU evidence, search-entry comparison, strategy transfer, same-period readiness, or `region_type`. |
| Current result | Six of six manually specified answer contracts passed the current required/forbidden-term checks. |
| Checked by | `python3 eval/eval_retail_demo2_answer_behavior.py` |
| Result path | `eval/retail_decision_support_eval_results/eval_retail_demo2_answer_behavior_result.txt` |
| Failure mode | A contract omits required metric or scope language, or includes a forbidden ranking, causal, profit, or direct strategy-transfer claim. |

### Answer-contract rules checked here

| Boundary | Rule |
|---|---|
| Activity cost ratio | `activity_cost_ratio_pct` is not traditional ROI or profit margin. |
| Top-SKU share | `top3_sku_transaction_amount_share_pct` describes the listed top-SKU evidence rather than full product-category sales share. |
| Search-entry evidence | Search-entry evidence is one visibility-to-entry signal within the current store-period profile. |
| Activity evidence | Activity evidence describes operating-tool usage; it does not establish automatic promotion-transfer logic. |
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

## Experiment 7: Repeated-Window Coverage and Descriptive Movement Check

| Item | Content |
|---|---|
| Question | Do Stores B-F have complete February-April 2026 coverage, and what descriptive movement is visible in selected store-period metrics? |
| Input | B-F store-period records across 2026-02, 2026-03, and 2026-04. |
| Transformation | `retail_ops/sql/03_store_period_panel_coverage.sql` checks observed-month coverage and selected averages. `retail_ops/sql/04_repeated_window_panel_summary.sql` reports February-to-April levels, deltas, and percentage changes for selected metrics. |
| Output | `retail_ops/outputs/store_period_panel_coverage_output.csv`; `retail_ops/outputs/repeated_window_panel_summary_output.csv`. |
| Expected behavior | Confirm repeated-window coverage and report descriptive movement without converting short-window changes into store rankings, causal explanations, pairwise comparability decisions, or guardrail-stability claims. |
| Current result | Stores B-F each contain three observed months. The summary reports February-to-April movement in selected transaction, traffic, conversion, search-entry, and activity-cost-ratio fields. |
| Current use | Evidence preparation for future question-specific comparison rules. |
| Checked by | `python3 retail_ops/scripts/validate_store_period_panel.py`; `python3 retail_ops/scripts/validate_repeated_window_panel_summary.py` |
| Failure mode | Treating the three-month descriptive panel as proof that March guardrail notes remain stable over time. |

Evidence required before a monthly guardrail-stability test:

| Missing evidence | Why it matters |
|---|---|
| Repeated top-SKU evidence for each store-month | The current March guardrail logic includes top-3 SKU concentration, so monthly notes cannot be reproduced without monthly SKU evidence. |
| Monthly recomputation of `comparison_limit_notes` under the implemented SQL contract | Stability must compare like-for-like monthly note sets rather than infer stability from unrelated store-level metrics. |
| Broader repeated store-period records beyond the current B-F three-month panel | More stores and months are needed before treating a local sensitivity pattern as reusable. |
| Activity calendar or campaign-condition evidence | Activity involvement should not be interpreted as full campaign status without operating context. |
| Local competition, price-pressure, fulfillment, or stockout evidence | These factors are needed before promotion, pricing, market-share, or strategy-transfer interpretation. |

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
| SQL baseline | Current SQL output values and current `comparison_limit_notes`. |
| Easier-to-trigger scenario | Recomputed notes after lowering each implemented threshold by 5 percentage points. |
| Harder-to-trigger scenario | Recomputed notes after raising each implemented threshold by 5 percentage points. |
| Threshold-sensitive row | A store row whose note set changes under at least one threshold scenario. |

Current result: four of five B-F rows are threshold-sensitive under the harder-to-trigger scenario, while Store B remains unchanged and the easier-to-trigger scenario changes no rows. This small-sample result supports treating the guardrails as diagnostic warnings rather than optimized pairwise-comparison rules.
